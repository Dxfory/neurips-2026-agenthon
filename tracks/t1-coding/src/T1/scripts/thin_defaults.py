#!/usr/bin/env python3
"""Show or check the image defaults of the registered thin run parameters (THIN_RUN_PARAM_DEFAULTS); set the one
that a frozen candidate flips, T1_THIN_FEATURES.

A frozen candidate turns an R2 mechanism on by default by changing its image default, never through a run's
environment (audit 10-22, section 3). Each default lives in one constant of t1_phase1/solver_profile.py
(THIN_FEATURES_DEFAULT for T1_THIN_FEATURES); profile_env, build_image's rendered recipe and image contract check, the
identity smoke and the isolation gate's 'thin' profile and default receipt all read it. The two tracked recipes
(Dockerfile, Dockerfile.amd64) carry the same value on their ENV line: 'set' rewrites the constant and both lines
together, 'check' verifies that all four registered defaults agree with both recipes.

usage: thin_defaults.py [--root T1] show
       thin_defaults.py [--root T1] check
       thin_defaults.py [--root T1] set T1_THIN_FEATURES=VALUE

'set' takes T1_THIN_FEATURES only (SETTABLE): the other registered run parameters (T1_THIN_MAX_CONTINUATIONS 3,
T1_THIN_MAX_CONTINUATIONS_UNIT off, T1_THIN_FINAL_DELIVERY_TRIGGER A) are fixed at their image defaults by the
pre-registration, and the gate's exact counts and the switch tests pin them, so changing one is a reviewed code
change, never a flip. The value must pass the tree's own parser (an unknown or unwired name such as k3 is refused) and
is written as its sorted names. Nothing is written unless the value is accepted and every file has exactly one line
to change; after a write the tree's solver_profile is loaded again and must report exactly the new defaults. --root
(default: this script's T1 folder) is read through a child interpreter (-B, no bytecode written), never imported here.
Exit 0 on success, 1 when 'check' finds a disagreement, 2 for a refused request.
"""
import argparse
import ast
import json
import os
from pathlib import Path
import re
import subprocess
import sys

HERE_ROOT = Path(__file__).resolve().parents[1]
RECIPES = ('Dockerfile', 'Dockerfile.amd64')
# The run parameter 'set' may change -> the solver_profile constant (a str literal) that holds its image default. The
# other registered run parameters keep their pre-registered image defaults ('check' still covers them).
SETTABLE = {'T1_THIN_FEATURES': 'THIN_FEATURES_DEFAULT'}

# Runs in a child interpreter on the target tree: its defaults, and its own verdict on the proposed values.
CHILD = r'''
import json, sys
sys.path.insert(0, sys.argv[1])
proposed = json.loads(sys.argv[2])
out = {}
try:
    from t1_phase1 import solver_profile as p
except Exception as error:
    out['error'] = 'cannot import t1_phase1.solver_profile: ' + type(error).__name__ + ': ' + str(error)[:300]
else:
    out['defaults'] = dict(p.THIN_RUN_PARAM_DEFAULTS)
    out['registered'] = list(p.THIN_RUN_PARAMS)
    out['recipe'] = {name: p.recipe_value(value) for name, value in p.THIN_RUN_PARAM_DEFAULTS.items()}
    checked, refused = {}, {}
    for name, value in proposed.items():
        try:
            p.check_run_param_defaults({name: value})
        except ValueError as error:
            refused[name] = str(error)[:300]
            continue
        if name == 'T1_THIN_FEATURES':
            value = ','.join(sorted(p.parse_thin_features(value)))
        checked[name] = {'value': value, 'recipe': p.recipe_value(value)}
    out.update(checked=checked, refused=refused)
print(json.dumps(out))
'''


class Refuse(Exception):
    pass


def tree_state(root, proposed=None):
    """The target tree's defaults ({'defaults', 'registered', 'recipe', 'checked', 'refused'}), read in a child."""
    env = {k: v for k, v in os.environ.items() if k in ('PATH', 'HOME', 'LANG', 'LC_ALL', 'TMPDIR')}
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    done = subprocess.run([sys.executable, '-B', '-c', CHILD, str(root), json.dumps(proposed or {})],
                          capture_output=True, text=True, timeout=120, env=env)
    try:
        state = json.loads(done.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        raise Refuse('cannot read the tree\'s solver_profile (exit ' + str(done.returncode) + ')') from None
    if 'error' in state:
        raise Refuse(state['error'])
    return state


def recipe_line(name, recipe_value):
    return '    ' + name + '=' + recipe_value + ' \\'


def problems(root, state):
    """Disagreements between the tree's defaults and its tracked recipes (empty when they agree)."""
    found = []
    for recipe in RECIPES:
        try:
            text = (Path(root) / recipe).read_text()
        except OSError:
            found.append(recipe + ' is missing')
            continue
        for name, value in state['recipe'].items():
            lines = [line for line in text.splitlines() if re.match(r'\s*' + re.escape(name) + '=', line)]
            if lines != [recipe_line(name, value)]:
                found.append(f'{recipe}: {name} is not exactly one ENV line {recipe_line(name, value)!r}')
    return found


def parse_pairs(pairs):
    out = {}
    for pair in pairs:
        name, sep, value = pair.partition('=')
        if not sep or not re.fullmatch(r'T1_THIN_[A-Z0-9_]{1,48}', name):
            raise Refuse('set takes NAME=VALUE with NAME a registered T1_THIN_* run parameter')
        if name in out:
            raise Refuse(name + ' is given twice')
        out[name] = value
    if not out:
        raise Refuse('set needs at least one NAME=VALUE')
    return out


def rewrite_constant(text, constant, value):
    pattern = re.compile(r'(?m)^' + re.escape(constant) + r' = (.+)$')
    matches = pattern.findall(text)
    if len(matches) != 1:
        raise Refuse('solver_profile.py has no single line ' + constant + ' = ...')
    try:
        current = ast.literal_eval(matches[0])
    except (ValueError, SyntaxError):
        raise Refuse('solver_profile.py: ' + constant + ' is not a plain literal') from None
    if type(current) is not str:
        raise Refuse('solver_profile.py: ' + constant + ' is not a str')
    return pattern.sub(lambda m: constant + ' = ' + repr(value), text, count=1)


def rewrite_recipe(text, recipe, name, value):
    pattern = re.compile(r'(?m)^    ' + re.escape(name) + r'=\S+ \\$')
    if len(pattern.findall(text)) != 1 or len(re.findall(r'(?m)^\s*' + re.escape(name) + '=', text)) != 1:
        raise Refuse(recipe + ' has no single ENV line for ' + name)
    return pattern.sub(lambda m: recipe_line(name, value), text, count=1)


def set_defaults(root, pairs):
    """Rewrite the T1_THIN_FEATURES constant and both recipe lines (all or nothing); returns the tree's new defaults."""
    requested = parse_pairs(pairs)
    for name in requested:
        if name not in SETTABLE:
            raise Refuse(name + ' cannot be set here: only ' + ', '.join(SETTABLE) + "'s image default is flipped; "
                         'the other run parameters keep their pre-registered image defaults')
    state = tree_state(root, requested)
    for name in requested:
        if name not in state['defaults']:
            raise Refuse(name + ' is not a registered run parameter with an image default in this tree')
    if state['refused']:
        raise Refuse('; '.join(f'{name}: {reason}' for name, reason in sorted(state['refused'].items())))
    root = Path(root)
    profile = root / 't1_phase1' / 'solver_profile.py'
    texts = {profile: profile.read_text(), **{root / recipe: (root / recipe).read_text() for recipe in RECIPES}}
    for name, checked in state['checked'].items():
        texts[profile] = rewrite_constant(texts[profile], SETTABLE[name], checked['value'])
        for recipe in RECIPES:
            texts[root / recipe] = rewrite_recipe(texts[root / recipe], recipe, name, checked['recipe'])
    originals = {path: path.read_text() for path in texts}
    try:
        for path, text in texts.items():
            if originals[path] != text:
                path.write_text(text)
        after = tree_state(root)
        want = dict(state['defaults'], **{name: checked['value'] for name, checked in state['checked'].items()})
        if after['defaults'] != want:
            raise Refuse('the rewritten solver_profile reports other defaults than requested')
        left = problems(root, after)
        if left:
            raise Refuse('; '.join(left))
    except BaseException:
        for path, text in originals.items():  # all or nothing: a failed rewrite leaves the tree as it was
            path.write_text(text)
        raise
    return after['defaults']


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--root', type=Path, default=HERE_ROOT, help='the T1 folder to read or change')
    sub = ap.add_subparsers(dest='action', required=True)
    sub.add_parser('show')
    sub.add_parser('check')
    setter = sub.add_parser('set')
    setter.add_argument('pairs', nargs='+', metavar='T1_THIN_FEATURES=VALUE')
    args = ap.parse_args(argv)
    try:
        if args.action == 'set':
            defaults = set_defaults(args.root, args.pairs)
            print(json.dumps({'set': True, 'defaults': defaults}, indent=2))
            return 0
        state = tree_state(args.root)
        found = problems(args.root, state)
        if args.action == 'show':
            print(json.dumps({'defaults': state['defaults'], 'recipes_agree': not found}, indent=2))
            return 0
        for problem in found:
            print('thin_defaults: ' + problem, file=sys.stderr)
        print(json.dumps({'check': 'ok' if not found else 'mismatch', 'defaults': state['defaults']}))
        return 0 if not found else 1
    except Refuse as error:
        print('thin_defaults: refused: ' + str(error), file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
