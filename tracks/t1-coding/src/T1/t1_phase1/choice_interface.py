"""A bounded JSON interface over generic components, not task-specific recipes."""
from __future__ import annotations

import json
import re

from . import component_recipe


def _pairs(items):
    out = {}
    for key, value in items:
        if key in out:
            raise ValueError('Duplicate JSON field: ' + key)
        out[key] = value
    return out


def _nonfinite(value):
    raise ValueError('Nonfinite JSON number')


def _load(raw):
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError('Empty or nontext choice response')
    if len(raw.encode()) > 64000:
        raise ValueError('Choice response exceeds 64000 bytes')
    raw = raw.strip()
    fence = re.fullmatch(r'```(?:json)?\s*\n(.*?)\n```', raw, re.S)
    if fence:
        raw = fence[1]
    try:
        value = json.loads(raw, object_pairs_hook=_pairs, parse_constant=_nonfinite)
        json.dumps(value, allow_nan=False)
    except RecursionError as exc:
        raise ValueError('Choice response too deeply nested') from exc
    if not isinstance(value, dict):
        raise ValueError('Choice response must be one JSON object')
    return value


def plan_prompt(context):
    return (context + '\nAVAILABLE GENERIC COMPONENTS AND RECIPE LANGUAGE:\n'
            + component_recipe.catalog() + '''
ACTION: Select components and bind task facts. Return only one JSON object:
{"components":["exact registered function names"],"bindings":{},"uncertainties":[]}
In bindings, record the needed task parameters/units and their short source quotes. In uncertainties,
record any contradictory statements and which explicit formula or visible relation will resolve them.
Do not rederive algorithms already implemented by the selected functions. Do not invent function names.
Use only methods permitted by the task. This is the only planning request. Keep the selection compact.
If no permitted component combination can solve the task, return {"unsupported":"specific missing capability"}.
''')


def validate_plan(raw):
    plan = _load(raw)
    if set(plan) == {'unsupported'} and isinstance(plan['unsupported'], str) and plan['unsupported'].strip():
        return plan
    if set(plan) != {'components', 'bindings', 'uncertainties'}:
        raise ValueError('Plan fields must be components, bindings, uncertainties')
    names = plan['components']
    if (not isinstance(names, list) or len(names) > 64
            or any(not isinstance(n, str) or n not in component_recipe.allowed_functions() for n in names)
            or len(names) != len(set(names))):
        raise ValueError('Plan components must be unique registered function names')
    if not isinstance(plan['bindings'], dict) or not isinstance(plan['uncertainties'], list):
        raise ValueError('Plan bindings must be an object; uncertainties must be a list')
    return plan


def code_prompt(context, contract, recipe, version, feedback, remaining, seconds):
    return (context + '\nAVAILABLE GENERIC COMPONENTS AND RECIPE LANGUAGE:\n'
            + component_recipe.catalog() + '\nSELECTED COMPONENTS AND TASK BINDINGS:\n' + contract
            + '\nCURRENT VERSION: ' + version + '\nCURRENT RECIPE:\n'
            + json.dumps(recipe, ensure_ascii=False) + '\nACTUAL FEEDBACK:\n' + feedback
            + f'\nREMAINING REQUESTS: {remaining}\nREMAINING SECONDS: {max(0, round(seconds, 1))}\n'
            + '''ACTION: Fill the next executable component composition; do not write a Python program.
Return one JSON object: {"base":"CURRENT VERSION","action":"replace" or "append","steps":[...]}
Optional basis is a short string linking methodological/units choices to task evidence.
replace supplies the entire recipe; append adds steps to the existing recipe. A changed earlier step
requires replace with all needed steps. Every accepted recipe reruns from scratch, with clean output.
Choose only registered operations and functions. Fill their arguments with literals, references or the
restricted expressions. Reuse implemented calculations; do not derive their internals. Loops combine
related work without one model request per function. Complete one coherent executable calculation now,
including its small visible consistency check, then advance toward every required full-scale output.
Missing outputs are not success; never insert dummy output. Keep exact samples, methods and RNG contract.
Do not silently resolve a task contradiction using library defaults. Unsupported, without execution:
{"base":"CURRENT VERSION","action":"unsupported","reason":"specific missing generic capability"}.
''')


def transition(raw, current, version, *, input_root, output_root):
    packet = _load(raw)
    if packet.get('base') != version:
        raise ValueError('Stale or missing base; use CURRENT VERSION')
    action = packet.get('action')
    if action == 'unsupported':
        if (set(packet) != {'base', 'action', 'reason'} or not isinstance(packet['reason'], str)
                or not packet['reason'].strip()):
            raise ValueError('Unsupported response requires a specific reason')
        return None, None, packet['reason'][:1000]
    if action not in ('replace', 'append') or not {'base', 'action', 'steps'} <= set(packet):
        raise ValueError('Expected base, action replace/append, and steps')
    if set(packet) - {'base', 'action', 'steps', 'basis'}:
        raise ValueError('Unknown choice fields')
    if not isinstance(packet['steps'], list) or not packet['steps']:
        raise ValueError('Supply at least one complete step')
    steps = (current.get('steps', []) if action == 'append' else []) + packet['steps']
    recipe = {'steps': steps}
    if 'basis' in packet:
        recipe['basis'] = packet['basis']
    component_recipe.validate_recipe(recipe)
    program = component_recipe.render_program(recipe, input_root, output_root)
    return recipe, program, None
