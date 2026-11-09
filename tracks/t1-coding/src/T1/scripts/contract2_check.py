#!/usr/bin/env python3
"""T1_CONTRACT2 validation (stack 10 spec §9 b-d), local only and never in the image: no model, network or docker.

statements: every header statement T1_CONTRACT2 adds over the dev instruction texts (indented blocks), and the exact
headers whose extra columns it no longer protects (cross-file), for review by hand.
outputs: the normaliser re-run over recorded output sets (<runs>/<run>/work/shard-*/results/<attempt>/agent-output),
each copied to a temporary directory, with T1_CONTRACT2 off and on (T1_CONTRACT on in both); a file counts as changed
by CONTRACT2 when the two versions differ. Passing attempts (verifier/reward.txt 1) must show none.

Held-out units (holdout.json) are refused before any of their files is opened: their result directories are skipped
by name and their instructions are never read. Prints counts and file names only (no output contents)."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import tomllib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from t1_phase1 import audit  # noqa: E402

# holdout.json: T1_HOLDOUT_FILE, else the first .t1-local-eval/analysis/holdout.json above this file.
HOLDOUT = os.environ.get('T1_HOLDOUT_FILE') or next(
    (str(p / '.t1-local-eval/analysis/holdout.json') for p in Path(__file__).resolve().parents
     if (p / '.t1-local-eval/analysis/holdout.json').is_file()), None)
SOURCE = Path(os.environ.get('T1_SOURCE_CACHE') or Path.home() / 't1-local-eval/source-cache')
COMMIT = '1a60fc48024c4f0e9978416d84e8370dbb0236cd'


def split(path=None):
    """(dev unit ids, held-out unit ids) from holdout.json."""
    if not (path or HOLDOUT):
        raise SystemExit('holdout.json not found: set T1_HOLDOUT_FILE')
    data = json.loads(Path(path or HOLDOUT).read_text())
    held = {unit for units in data['holdout'].values() for unit in units}
    return sorted({unit for units in data['dev'].values() for unit in units} - held), held


def participant_instruction(unit_dir):
    """instruction.md as the participant view holds it: lines naming a canary removed (staging.stage_unit)."""
    card = tomllib.loads((unit_dir / 'card.toml').read_text())
    values = []

    def walk(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if 'canary' in key.lower() and isinstance(item, str) and item:
                    values.append(item)
                walk(item)
    walk(card)
    text = (unit_dir / 'instruction.md').read_text(errors='replace')
    return ''.join(line for line in text.splitlines(keepends=True)
                   if 'canary' not in line.lower() and not any(value in line for value in values))


def instruction(unit, held, source=None):
    if unit in held:
        raise SystemExit(f'refused: {unit} is held out')
    return participant_instruction(Path(source or SOURCE / COMMIT / 'units') / unit)


def statements(units, held, source=None):
    """[(unit, target, columns, lead, kept)] of indented statements and [(unit, csv, freed columns)] of cross-file
    exposures."""
    added, crossed = [], []
    for unit in units:
        text = instruction(unit, held, source)
        lines = text.splitlines()
        fenced = audit._fenced_lines(lines)
        rules = audit.contract_rules(text, contract2=True)
        for target, names, _, lead in audit._indented_statements(lines, fenced, audit._sections(lines, fenced),
                                                                  set()):
            added.append((unit, target, names, lead.strip(), rules['headers'].get(target) == names))
        for base in sorted(rules['header_exact']):
            others = set().union(*(c for n, c in rules['headers'].items() if n != base))
            freed = sorted((rules['ticked'] & others) - set(rules['headers'][base]))
            if freed:
                crossed.append((unit, base, freed))
    return added, crossed


def _normalized(text, folder, on):
    """{relative path: bytes} of every file after normalize_contract with T1_CONTRACT2 on or off."""
    previous = os.environ.get('T1_CONTRACT2')
    os.environ['T1_CONTRACT2'] = '1' if on else '0'
    try:
        audit.normalize_contract(text, folder)
    finally:
        if previous is None:
            os.environ.pop('T1_CONTRACT2', None)
        else:
            os.environ['T1_CONTRACT2'] = previous
    return {p.relative_to(folder).as_posix(): p.read_bytes() for p in sorted(folder.rglob('*')) if p.is_file()}


def attempts(runs, dev, held):
    """(run, attempt, unit, agent-output, passed) for every recorded dev attempt; held-out ones are skipped unread."""
    for output in sorted(Path(runs).glob('*/work/shard-*/results/*/agent-output')):
        name = output.parent.name
        unit = name.split('__r')[0]
        if unit in held or unit not in dev:
            continue
        reward = output.parent / 'verifier' / 'reward.txt'
        try:
            passed = float(reward.read_text().strip()) == 1.0
        except (OSError, ValueError):
            passed = False
        yield output.parents[4].name, name, unit, output, passed


def check_outputs(runs, dev, held, source=None, limit_sec=None):
    """{'passing': [...], 'failing': [...]} of (run, attempt, file) CONTRACT2 changes, and the attempts read."""
    texts, differs, changed, seen, started = {}, {}, {'passing': [], 'failing': []}, 0, time.monotonic()
    for run, name, unit, output, passed in attempts(runs, dev, held):
        if limit_sec is not None and time.monotonic() - started > limit_sec:
            break
        if unit not in texts:
            texts[unit] = instruction(unit, held, source)
            on, off = (audit.contract_rules(texts[unit], contract2=flag) for flag in (True, False))
            differs[unit] = on['headers'] != off['headers'] or any(
                (on['ticked'] & set().union(*(c for n, c in on['headers'].items() if n != base))) - set(columns)
                for base, columns in on['headers'].items())
        text = texts[unit]
        seen += 1
        if not differs[unit]:
            continue  # the same headers and no column freed: CONTRACT2 cannot change a file
        with tempfile.TemporaryDirectory() as tmp:
            versions = []
            for on in (False, True):
                folder = Path(tmp) / ('on' if on else 'off')
                shutil.copytree(output, folder, symlinks=True)
                versions.append(_normalized(text, folder, on))
        for path in sorted(set(versions[0]) | set(versions[1])):
            if versions[0].get(path) != versions[1].get(path):
                changed['passing' if passed else 'failing'].append((run, name, path))
    return changed, seen


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('what', choices=('statements', 'outputs'))
    parser.add_argument('--runs', type=Path, default=Path.home() / 't1-local-eval/runs')
    parser.add_argument('--run', action='append', help='only these run directories (names under --runs)')
    parser.add_argument('--limit-sec', type=float)
    args = parser.parse_args()
    dev, held = split()
    if args.what == 'statements':
        added, crossed = statements(dev, held)
        for unit, target, names, lead, kept in added:
            print(f'{unit}: {target} <- {",".join(names)} | lead: {lead[:80]} | kept: {kept}')
        for unit, base, freed in crossed:
            print(f'cross-file {unit}: {base} no longer protects {", ".join(freed)}')
        return
    runs = args.runs
    if args.run:
        runs = Path(tempfile.mkdtemp())
        for name in args.run:
            (runs / name).symlink_to(args.runs / name)
    changed, seen = check_outputs(runs, set(dev), held, limit_sec=args.limit_sec)
    print(json.dumps({'attempts_read': seen, 'passing_changed': changed['passing'],
                      'failing_changed': changed['failing']}, indent=1))


if __name__ == '__main__':
    main()
