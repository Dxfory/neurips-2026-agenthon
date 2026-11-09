"""Small, explicit solver profiles and bounded, non-secret startup identity.

This module observes configuration; it never mutates the environment or changes
House's default request. Recipes supply defaults. Actual prompt exposure and
helper adoption remain separate evidence, not claims made by this receipt.
"""
import ast
import hashlib
import importlib.util
import os
from pathlib import Path
import re

from .build_identity import IDENTITY

CANDIDATE = 'stack14-thin-v1'
HISTORICAL = ('legacy-v1', 'stack14-candidate-v1', 'stack14-unified-v1',
              'stack14-minimal-v1')  # receipt readers only
FULL_LIBRARY_VERSION = '0.6.2'
FULL_CATALOG_SHA256 = 'ff94460f43dc8fb0bd69799173da5f19beb4c3246eb85b3f6f42aa7e5e24fc23'
LEGACY = 'legacy-v1'
LEGACY_ON = ('T1_PLAN', 'T1_REVIEW', 'T1_FRESH', 'T1_SELFCHECK', 'T1_STAGED',
             'T1_PRECHECK', 'T1_LIBHINT', 'T1_CONTRACT', 'T1_ACCEPT', 'T1_ADOPT', 'T1_RAISEGUARD')
LATEST_ON = ('T1_S10', 'T1_S11_CONV', 'T1_S11_RAISE', 'T1_S11_LINT', 'T1_S11_UNION',
             'T1_S11_LIBNOTES', 'T1_S12_MECH', 'T1_S12_POST', 'T1_S13_COERCE', 'T1_S13_REFINE',
             'T1_CONTRACT2')
EXPERIMENTAL_OFF = ('T1_RUNTIME', 'T1_REQUIREMENTS', 'T1_CONTINUATION_BUDGET', 'T1_S12_LIB',
                    'T1_S12_INERT', 'T1_S12_AUDIT', 'T1_ROUTE', 'T1_CARDS', 'T1_R1THINK',
                    'T1_PLANTHINK', 'T1_PATCHTHINK', 'T1_S10_TAGS')
FLAGS = LEGACY_ON + LATEST_ON + EXPERIMENTAL_OFF + ('T1_LIBRARY86', 'T1_REQUEST_SUPERVISOR', 'T1_S10_STATE', 'T1_MINIMAL', 'T1_THIN', 'T1_THIN_DIRECT_STEP')
MINIMAL_ON = ('T1_MINIMAL', 'T1_REVIEW', 'T1_SELFCHECK', 'T1_PRECHECK', 'T1_LIBHINT',
              'T1_CONTRACT', 'T1_RAISEGUARD', 'T1_S11_CONV', 'T1_S11_RAISE',
              'T1_S11_LINT', 'T1_S11_LIBNOTES', 'T1_S12_MECH', 'T1_S12_POST',
              'T1_CONTRACT2', 'T1_LIBRARY86', 'T1_REQUEST_SUPERVISOR')
THIN_ON = ('T1_THIN', 'T1_LIBRARY86', 'T1_REQUEST_SUPERVISOR')
BUDGETS = ('T1_R1THINK_BUDGET', 'T1_PLANTHINK_BUDGET', 'T1_PATCHTHINK_BUDGET',
           'T1_S10_DECIDE_BUDGET_F', 'T1_S10_DECIDE_BUDGET_D', 'T1_S10_DECIDE_BUDGET_L',
           'T1_S10_CHECK_BUDGET', 'T1_S10_STATE_BUDGET', 'T1_S10_CODE_BUDGET_F',
           'T1_S10_CODE_BUDGET_D', 'T1_S10_CODE_BUDGET_L', 'T1_S10_FLAGFIX_BUDGET',
           'T1_S12_AUDIT_BUDGET', 'T1_R1THINK_PLAN_MAX', 'T1_MINIMAL_PLAN_THINK_BUDGET',
           'T1_THIN_REASONING_BUDGET')
OPTIONS = {'T1_S10_CHECK': ('never', 'auto', 'always'),
           'T1_S10_TIER': ('auto', 'F', 'D', 'L'), 'T1_THINK_FIELD': ('kw', 'top', 'both'),
           'T1_THIN_THINKING': ('on', 'off', 'low_effort'), 'T1_THIN_INTERFACE': ('free', 'choice')}
# Thinking per thin request type: on (native), low_effort, off, or budget:<n> (1-2048 reasoning tokens).
# When a key is absent, the older global T1_THIN_THINKING/T1_THIN_REASONING_BUDGET apply if set, else these.
REQUEST_THINKING = {'T1_THIN_CONTRACT_THINKING': 'low_effort', 'T1_THIN_CODE_THINKING': 'off'}
# Continuation requests per free-mode code reply cut at the output limit (each one counted request).
# T1_THIN_MAX_CONTINUATIONS (0-3) overrides it per run. This one value is the solver default, the image
# recipe's ENV (profile_env) and the N of the isolation gate's exact truncated count, 1 + 5 x (1 + N).
THIN_MAX_CONTINUATIONS = 3
# R2 mechanisms, each behind its own name in T1_THIN_FEATURES: a comma list of distinct names, parsed once at thin
# solve start (thin._feature reads it). The image default is THIN_FEATURES_DEFAULT: a mechanism stays off until it
# passes gate D. assert_rerun (R3) is the run-through fix B, always on through R2 and switched since.
THIN_FEATURES = ('repair_time', 'false_green', 'final_delivery', 'loop', 'k3', 'repair_feedback', 'sanity',
                 'sanity_constant', 'fg_bool_false', 'resample', 'executed_note', 'fg_normalize', 'complete_first',
                 'assert_rerun', 'second_draft', 'task_fidelity', 'stub_check', 'output_review',
                 'progress_time', 'crash_redraft', 'literal_contract', 'lib_expose', 'outputs_fix', 'redraft_fix', 'loop_fix', 'clock_fix', 'input_fidelity', 'lib_review')
# Registered but not wired into thin (S2-12 never landed; decision 10-22 18:30): parse_thin_features refuses them, so a
# run that names one ends as invalid_configuration in thin and is refused before any container starts (docker_runner,
# the gate's --fixture-env), never run as if it were on. A static test keeps every other name read by thin.
THIN_FEATURES_UNWIRED = ('k3',)
THIN_FEATURES_WIRED = tuple(name for name in THIN_FEATURES if name not in THIN_FEATURES_UNWIRED)
# The image default of T1_THIN_FEATURES (audit 10-22 section 3): the mechanisms a frozen candidate runs with when no
# run sets the variable. This one value is profile_env's (hence build_image's rendered recipe, its image contract
# check and identity smoke), the tracked recipes' ENV line (recipe_value; a test keeps both Dockerfiles in step) and
# the isolation gate's 'thin' profile and default receipt (integration_fixture). It must parse (an unknown or
# unwired name such as k3 is refused at import, see THIN_RUN_PARAM_DEFAULTS); scripts/thin_defaults.py rewrites it
# and both recipe lines together. The solver itself, without the variable (tests, any run outside the image), keeps
# every mechanism off: the image always sets it.
THIN_FEATURES_DEFAULT = 'assert_rerun,clock_fix,complete_first,crash_redraft,fg_normalize,input_fidelity,lib_expose,literal_contract,loop_fix,output_review,outputs_fix,progress_time,redraft_fix,stub_check,task_fidelity'
# Continuation requests per unit, summed over all its cut code replies (each retried one counts once), on top of
# the per-reply cap: T1_THIN_MAX_CONTINUATIONS_UNIT is one of these pre-registered candidates (R2A_PARAMS.md).
# 'off', the less-intervention side, leaves only the per-reply cap; it is the solver and image default.
THIN_MAX_CONTINUATIONS_UNIT_CHOICES = ('2', '4', 'off')
THIN_MAX_CONTINUATIONS_UNIT = 'off'
# S2-06 (T1_THIN_FEATURES final_delivery): when the FINAL DELIVERY ask starts, the pre-registered OS-2 candidates
# (R2A_PARAMS.md): REMAINING REQUESTS <= r or REMAINING SECONDS <= s as the prompt shows them. 'A', the
# less-intervention side, is the solver and image default; T1_THIN_FINAL_DELIVERY_TRIGGER names the other per run.
FINAL_DELIVERY_TRIGGERS = {'A': (2, 140), 'B': (3, 200)}
THIN_FINAL_DELIVERY_TRIGGER = 'A'
HEX40 = re.compile(r'[0-9a-f]{40}')
HEX64 = re.compile(r'[0-9a-f]{64}')
VERSION = re.compile(r'[0-9]{1,4}(?:\.[0-9]{1,4}){1,3}')


def parse_thin_reasoning_budget(value):
    """Validate an explicit experiment; absence retains native provider thinking."""
    if (not isinstance(value, str) or not re.fullmatch(r'[1-9][0-9]{0,3}', value)
            or int(value) > 2048):
        raise ValueError('T1_THIN_REASONING_BUDGET must be an integer from 1 to 2048')
    return int(value)


def parse_thin_max_continuations(value):
    """Continuation requests per cut code reply: an explicit integer from 0 (none) to 3."""
    if not isinstance(value, str) or not re.fullmatch(r'[0-3]', value):
        raise ValueError('T1_THIN_MAX_CONTINUATIONS must be an integer from 0 to 3')
    return int(value)


def parse_thin_features(value):
    """T1_THIN_FEATURES: '' (none) or a comma list of distinct THIN_FEATURES names, without spaces; a frozenset.
    Anything else, an unknown name or an unwired one (THIN_FEATURES_UNWIRED) included, is refused (the message never
    repeats the value)."""
    if not isinstance(value, str):
        raise ValueError('T1_THIN_FEATURES must be a string')
    names = value.split(',') if value else []
    if any(name not in THIN_FEATURES for name in names) or len(set(names)) != len(names):
        raise ValueError('T1_THIN_FEATURES must be empty or a comma list of distinct names from: '
                         + ', '.join(THIN_FEATURES_WIRED))
    if any(name in THIN_FEATURES_UNWIRED for name in names):
        raise ValueError('T1_THIN_FEATURES names a mechanism that is registered but not wired: '
                         + ', '.join(THIN_FEATURES_UNWIRED))
    return frozenset(names)


def public_thin_features(value):
    """Whether an untrusted record is a recorded feature set: the sorted list of distinct wired THIN_FEATURES names
    (the loop never runs with an unwired one)."""
    return (type(value) is list and all(type(name) is str and name in THIN_FEATURES_WIRED for name in value)
            and value == sorted(set(value)))


def parse_thin_max_continuations_unit(value):
    """T1_THIN_MAX_CONTINUATIONS_UNIT: 2 or 4 continuations per unit, or None for 'off' (the per-reply cap only)."""
    if not isinstance(value, str) or value not in THIN_MAX_CONTINUATIONS_UNIT_CHOICES:
        raise ValueError('T1_THIN_MAX_CONTINUATIONS_UNIT must be one of: '
                         + ', '.join(THIN_MAX_CONTINUATIONS_UNIT_CHOICES))
    return None if value == 'off' else int(value)


def parse_thin_final_delivery_trigger(value):
    """T1_THIN_FINAL_DELIVERY_TRIGGER: one FINAL_DELIVERY_TRIGGERS name; its (requests, seconds) bound."""
    if not isinstance(value, str) or value not in FINAL_DELIVERY_TRIGGERS:
        raise ValueError('T1_THIN_FINAL_DELIVERY_TRIGGER must be one of: ' + ', '.join(FINAL_DELIVERY_TRIGGERS))
    return FINAL_DELIVERY_TRIGGERS[value]


# R2 run parameters (S2-01/S2-02): every variable a run card may set for one arm, mapped to the solver's own parser
# (it raises ValueError on a value the loop refuses). This one registry is what docker_runner forwards, what identity
# records (thin_run_params; profile_matches is false for any value besides the image default) and what the isolation
# gate's --fixture-env accepts. A mechanism that reads a new T1_THIN_* parameter adds it here, never elsewhere.
THIN_RUN_PARAMS = {'T1_THIN_MAX_CONTINUATIONS': parse_thin_max_continuations,
                   'T1_THIN_FEATURES': parse_thin_features,
                   'T1_THIN_MAX_CONTINUATIONS_UNIT': parse_thin_max_continuations_unit,
                   'T1_THIN_FINAL_DELIVERY_TRIGGER': parse_thin_final_delivery_trigger}  # S2-06's OS-2 candidate
# A run parameter's value as identity records it: every THIN_FEATURES name at once is about 160 characters (R2's
# Y' arm 131), so the bound leaves room for the registry to grow; a test keeps the whole list inside it.
RUN_PARAM_VALUE = re.compile(r'[A-Za-z0-9_,:.-]{0,384}')


def run_param_valid(name, value):
    """Whether `value` is one the solver accepts for the registered run parameter `name`."""
    if name not in THIN_RUN_PARAMS or not isinstance(value, str) or not RUN_PARAM_VALUE.fullmatch(value):
        return False
    try:
        THIN_RUN_PARAMS[name](value)
    except ValueError:
        return False
    return True


# The image default of each registered run parameter that has one, as the recipe's ENV sets it: the one source of
# profile_env's run-parameter entries (in their order there), of the tracked recipes' ENV lines and of the isolation
# gate's default receipt. Only registered run parameters may appear; each value must be one the solver accepts.
THIN_RUN_PARAM_DEFAULTS = {'T1_THIN_MAX_CONTINUATIONS': str(THIN_MAX_CONTINUATIONS),
                           'T1_THIN_MAX_CONTINUATIONS_UNIT': THIN_MAX_CONTINUATIONS_UNIT,
                           'T1_THIN_FEATURES': THIN_FEATURES_DEFAULT,
                           'T1_THIN_FINAL_DELIVERY_TRIGGER': THIN_FINAL_DELIVERY_TRIGGER}


def check_run_param_defaults(defaults):
    """`defaults` ({name: value}) when every name is a registered run parameter and every value one the solver's own
    parser accepts; else ValueError (an unknown or unwired T1_THIN_FEATURES name, such as k3, included)."""
    for name, value in defaults.items():
        if name not in THIN_RUN_PARAMS:
            raise ValueError(name + ' is not a registered run parameter (THIN_RUN_PARAMS)')
        if not isinstance(value, str) or not RUN_PARAM_VALUE.fullmatch(value):
            raise ValueError('the image default of ' + name + ' is not a plain run parameter value')
        THIN_RUN_PARAMS[name](value)  # the solver's parser; its message names the problem (e.g. 'not wired')
    return defaults


check_run_param_defaults(THIN_RUN_PARAM_DEFAULTS)  # fail at import: no image, gate or test runs a refused default


def recipe_value(value):
    """A run parameter's value as the tracked recipes' ENV block writes it: "" for the empty string, else as is
    (RUN_PARAM_VALUE characters need no quoting in a Dockerfile ENV line)."""
    return value if value else '""'


def parse_thin_request_thinking(value):
    """One request type's thinking: the House chat value ('default', True, False or a token budget)."""
    if value in ('on', 'low_effort', 'off'):
        return {'on': 'default', 'low_effort': True, 'off': False}[value]
    if isinstance(value, str) and value.startswith('budget:'):
        try:
            return parse_thin_reasoning_budget(value[7:])
        except ValueError:
            pass
    raise ValueError('thin request thinking must be on, low_effort, off or budget:<1-2048>')


def profile_env(name):
    if name != CANDIDATE:
        raise ValueError('Unknown solver profile')
    return {'T1_SOLVER_PROFILE': name, **dict.fromkeys(FLAGS, '0'),
            **dict.fromkeys(THIN_ON, '1'), 'T1_THINK_FIELD': 'both',
            **REQUEST_THINKING, 'T1_THIN_INTERFACE': 'free', **THIN_RUN_PARAM_DEFAULTS}


def _literals(name, sources):
    try:
        path = sources.get(name) if sources is not None else importlib.util.find_spec(name).origin
        raw = Path(path).read_bytes()
        if len(raw) > 512000:
            return {}
        tree = ast.parse(raw)
        wanted = {'__version__', 'CATALOG', 'CATALOG_040_LEN', 'CATALOG_050_LEN'}
        return {n.targets[0].id: ast.literal_eval(n.value) for n in tree.body
                if isinstance(n, ast.Assign) and len(n.targets) == 1
                and isinstance(n.targets[0], ast.Name) and n.targets[0].id in wanted}
    except (OSError, ValueError, TypeError, AttributeError, SyntaxError, RecursionError, ImportError):
        return {}


def identity(env=None, *, sources=None, provenance=None):
    env = os.environ if env is None else env
    selected = env.get('T1_SOLVER_PROFILE', LEGACY)
    name = selected if selected in (CANDIDATE, *HISTORICAL) else 'unknown'
    flags = {key: env.get(key, '1' if key == 'T1_S10_STATE' else '0') == '1' for key in FLAGS}
    flags['T1_S10_STATE'] = env.get('T1_S10_STATE', '1') != '0'
    effective = dict(flags)
    minimal = flags['T1_MINIMAL']
    s10 = flags['T1_S10'] and not minimal
    effective['T1_S10'] = s10
    for key in ('T1_S11_CONV', 'T1_S11_RAISE', 'T1_S11_LINT', 'T1_S11_LIBNOTES',
                'T1_S12_MECH', 'T1_S12_POST'):
        effective[key] = flags[key] and (s10 or minimal)
    for key in ('T1_S11_UNION', 'T1_S10_STATE', 'T1_S10_TAGS'):
        effective[key] = flags[key] and s10
    for key in ('T1_S12_INERT', 'T1_S12_AUDIT', 'T1_S13_COERCE', 'T1_S13_REFINE'):
        effective[key] = flags[key] and effective['T1_S12_POST']
    for key in ('T1_PLAN', 'T1_FRESH', 'T1_STAGED', 'T1_ROUTE', 'T1_S13_COERCE', 'T1_S13_REFINE'):
        effective[key] = effective[key] and not minimal
    effective['T1_S12_LIB'] = flags['T1_S12_LIB'] or flags['T1_LIBRARY86']
    effective['T1_ADOPT'] = flags['T1_ADOPT'] and flags['T1_LIBHINT'] and not minimal
    effective['T1_CARDS'] = flags['T1_CARDS'] and effective['T1_ADOPT']
    effective['T1_REQUIREMENTS'] = flags['T1_REQUIREMENTS'] and flags['T1_RUNTIME']
    effective['T1_ACCEPT'] = flags['T1_ACCEPT'] and not s10 and not minimal
    effective['T1_SELFCHECK'] = flags['T1_SELFCHECK'] and not effective['T1_ACCEPT'] and not s10
    effective['T1_PLANTHINK'] = flags['T1_PLANTHINK'] and not s10 and not minimal
    effective['T1_PATCHTHINK'] = flags['T1_PATCHTHINK'] or s10
    effective['T1_R1THINK'] = flags['T1_R1THINK'] and not s10 and not minimal
    if flags['T1_THIN']:
        # The thin solve branch returns before the legacy/minimal strategy flow.
        # Full library availability is separate from the old S12 mechanism.
        effective = {key: flags[key] if key in (*THIN_ON, 'T1_THIN_DIRECT_STEP') else False for key in FLAGS}
    # Explicit overrides only; these are inputs, not the clamped per-request budgets.
    budgets = {}
    for key in BUDGETS:
        value = env.get(key, '')
        if key == 'T1_THIN_REASONING_BUDGET':
            try:
                budgets[key] = parse_thin_reasoning_budget(value)
            except ValueError:
                pass
            continue
        if key == 'T1_MINIMAL_PLAN_THINK_BUDGET':
            if isinstance(value, str) and re.fullmatch(r'[0-9]{1,4}', value) and int(value) <= 1024:
                budgets[key] = int(value)
            continue
        bound = 99999 if key == 'T1_R1THINK_PLAN_MAX' else 3999
        if isinstance(value, str) and re.fullmatch(r'[1-9][0-9]{0,4}', value) and int(value) <= bound:
            budgets[key] = int(value)
    options = {k: env.get(k) if env.get(k) in values else values[0] for k, values in OPTIONS.items()}
    thinking = {}
    for key in REQUEST_THINKING:
        try:
            parse_thin_request_thinking(env[key])
            thinking[key] = env[key]
        except (KeyError, ValueError):
            pass
    # The cap the thin loop actually uses: the explicit value, else the default; an invalid value is
    # refused by the loop before any request, so it records no cap and counts as an invalid override.
    try:
        continuations = parse_thin_max_continuations(env.get('T1_THIN_MAX_CONTINUATIONS',
                                                             str(THIN_MAX_CONTINUATIONS)))
    except ValueError:
        continuations = None
    # The R2 switches the loop actually uses (absent: the loop's own defaults; for T1_THIN_FEATURES that is none,
    # which the image never relies on: its recipe sets THIN_FEATURES_DEFAULT); an invalid value is refused by the loop
    # before any request, so it records nothing here and counts as an invalid override.
    try:
        features = sorted(parse_thin_features(env.get('T1_THIN_FEATURES', '')))
    except ValueError:
        features = None
    unit_continuations = env.get('T1_THIN_MAX_CONTINUATIONS_UNIT', THIN_MAX_CONTINUATIONS_UNIT)
    try:
        parse_thin_max_continuations_unit(unit_continuations)
    except ValueError:
        unit_continuations = None
    # Every registered run parameter set in the environment, as given; one the solver refuses is an invalid override.
    run_params = {k: env[k] for k in THIN_RUN_PARAMS if k in env and run_param_valid(k, env[k])}
    # The S2-06 trigger the loop uses (absent: the default); an invalid one is counted below through
    # THIN_RUN_PARAMS and records nothing here.
    trigger = env.get('T1_THIN_FINAL_DELIVERY_TRIGGER', THIN_FINAL_DELIVERY_TRIGGER)
    if trigger not in FINAL_DELIVERY_TRIGGERS:
        trigger = None
    invalid_overrides = sum(k in env and k not in budgets for k in BUDGETS)
    invalid_overrides += sum(k in env and env[k] not in values for k, values in OPTIONS.items())
    invalid_overrides += sum(k in env and k not in thinking for k in REQUEST_THINKING)
    invalid_overrides += sum(k in env and k not in run_params for k in THIN_RUN_PARAMS)
    qf = _literals('qfkit', sources)
    rt = _literals('qfruntime', sources)
    catalog = qf.get('CATALOG')
    scope = 'unavailable'
    if isinstance(catalog, str):
        scope = 'full' if flags['T1_LIBRARY86'] else 'extended_050' if effective['T1_S12_LIB'] else 'legacy_040'
        if scope != 'full':
            bound = qf.get('CATALOG_050_LEN' if scope == 'extended_050' else 'CATALOG_040_LEN')
            if type(bound) is int and bound > 0:
                catalog = catalog[:bound]
            else:
                scope = 'full'
    raw = {
        'schema': 'solver-identity-v1', 'solver_profile': name,
        'profile_matches': name == CANDIDATE and all(env.get(k) == v for k, v in profile_env(name).items())
                           and not budgets and not invalid_overrides
                           and not any(k in env for k in THIN_RUN_PARAMS if k not in profile_env(name))
                           and all(options[k] == ('both' if k == 'T1_THINK_FIELD' else values[0]) for k, values in OPTIONS.items()),
        'raw_budget_overrides': budgets, 'option_values': options, 'thinking_values': thinking,
        'budget_scope':'raw_overrides_before_clamp', 'effective_scope':'configuration_gates_only',
        'invalid_override_count': invalid_overrides,
        'effective_flags': effective, 'configured_flags': flags,
        'invalid_flag_count': sum(env.get(k, '0') not in ('0', '1') for k in FLAGS),
        'qfkit_version': qf.get('__version__'), 'qfruntime_version': rt.get('__version__'),
        'catalog_scope': scope,
        **(provenance if isinstance(provenance, dict) else IDENTITY),
    }
    if continuations is not None:
        raw['thin_max_continuations'] = continuations
    if features is not None:
        raw['thin_features'] = features
    if unit_continuations is not None:
        raw['thin_max_continuations_unit'] = unit_continuations
    if run_params:
        raw['thin_run_params'] = run_params
    if trigger is not None:
        raw['thin_final_delivery_trigger'] = trigger
    if isinstance(catalog, str):
        raw.update(catalog_chars=len(catalog), catalog_sha256=hashlib.sha256(catalog.encode()).hexdigest())
    return safe_identity(raw)


def safe_identity(value):
    """Public exports revalidate an untrusted status, including nested keys/types."""
    if not isinstance(value, dict) or value.get('schema') != 'solver-identity-v1':
        return {}
    out = {'schema': 'solver-identity-v1'}
    for key, values in (('solver_profile', (CANDIDATE, *HISTORICAL, 'unknown')),
                        ('catalog_scope', ('full', 'extended_050', 'legacy_040', 'unavailable'))):
        if value.get(key) in values:
            out[key] = value[key]
    for key, expected in (('budget_scope', 'raw_overrides_before_clamp'),
                          ('effective_scope', 'configuration_gates_only')):
        if value.get(key) == expected:
            out[key] = expected
    if type(value.get('profile_matches')) is bool:
        out['profile_matches'] = value['profile_matches']
    for key in ('configured_flags', 'effective_flags'):
        if isinstance(value.get(key), dict):
            out[key] = {name: value[key][name] for name in FLAGS if type(value[key].get(name)) is bool}
    for key, limit in (('catalog_chars', 512000), ('invalid_flag_count', len(FLAGS)),
                       ('invalid_override_count', len(BUDGETS)+len(OPTIONS)+len(REQUEST_THINKING)+len(THIN_RUN_PARAMS)),
                       ('thin_max_continuations', 3)):
        if type(value.get(key)) is int and 0 <= value[key] <= limit:
            out[key] = value[key]
    if public_thin_features(value.get('thin_features')):
        out['thin_features'] = list(value['thin_features'])
    if (type(value.get('thin_max_continuations_unit')) is str
            and value['thin_max_continuations_unit'] in THIN_MAX_CONTINUATIONS_UNIT_CHOICES):
        out['thin_max_continuations_unit'] = value['thin_max_continuations_unit']
    if isinstance(value.get('thin_run_params'), dict):
        params = {k: v for k, v in value['thin_run_params'].items() if run_param_valid(k, v)}
        if params:
            out['thin_run_params'] = dict(sorted(params.items()))
    if (type(value.get('thin_final_delivery_trigger')) is str
            and value['thin_final_delivery_trigger'] in FINAL_DELIVERY_TRIGGERS):
        out['thin_final_delivery_trigger'] = value['thin_final_delivery_trigger']
    if isinstance(value.get('raw_budget_overrides'), dict):
        out['raw_budget_overrides'] = {k:v for k,v in value['raw_budget_overrides'].items()
            if k in BUDGETS and type(v) is int and
            (0 <= v <= 1024 if k == 'T1_MINIMAL_PLAN_THINK_BUDGET'
             else 1 <= v <= 2048 if k == 'T1_THIN_REASONING_BUDGET'
             else 1 <= v <= (99999 if k == 'T1_R1THINK_PLAN_MAX' else 3999))}
    if isinstance(value.get('option_values'), dict):
        out['option_values'] = {k:value['option_values'][k] for k, values in OPTIONS.items()
                                if value['option_values'].get(k) in values}
    if isinstance(value.get('thinking_values'), dict):
        thinking = {}
        for key in REQUEST_THINKING:
            try:
                parse_thin_request_thinking(value['thinking_values'].get(key))
                thinking[key] = value['thinking_values'][key]
            except ValueError:
                pass
        out['thinking_values'] = thinking
    for key in ('qfkit_version', 'qfruntime_version'):
        if isinstance(value.get(key), str) and VERSION.fullmatch(value[key]):
            out[key] = value[key]
    for key, pattern in (('source_commit', HEX40), ('context_sha256', HEX64), ('catalog_sha256', HEX64)):
        if isinstance(value.get(key), str) and pattern.fullmatch(value[key]):
            out[key] = value[key]
    out['source_state'] = ('frozen' if value.get('source_state') == 'frozen' and
                           'source_commit' in out and 'context_sha256' in out else 'unfrozen')
    return out


def startup_problem(receipt):
    """Validate an explicitly selected runtime profile; unlabelled test paths call no gate.

    Strategy overrides are allowed and reported by profile_matches, but the
    current profile cannot disable its own controller. The full
    installed catalog is mandatory; selecting a prefix or old installed library
    cannot masquerade as the current solver. Build provenance is checked by the
    builder and launcher, not inferred here from the image's configuration.
    """
    if receipt.get('solver_profile') != CANDIDATE:
        return 'unsupported_solver_profile'
    if receipt.get('configured_flags', {}).get('T1_THIN') is not True:
        return 'thin_controller_required'
    if (receipt.get('configured_flags', {}).get('T1_LIBRARY86') is not True
            or receipt.get('catalog_scope') != 'full'
            or receipt.get('qfkit_version') != FULL_LIBRARY_VERSION
            or receipt.get('catalog_sha256') != FULL_CATALOG_SHA256):
        return 'full_library_required'
    return None
