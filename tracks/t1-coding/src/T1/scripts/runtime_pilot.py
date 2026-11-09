#!/usr/bin/env python3
"""Prepare or explicitly execute a frozen, serial paired development pilot.

Preparation stages the pinned participant view and never calls a model. Execution
uses public_eval's existing container isolation and grader, requires receipts for
both actual image IDs, and stops at the first unmeasured infrastructure outcome.
No generated program executes on the host and no reference answer is interpreted.
The default comparison toggles runtime; adoption compares old and candidate
guidance with runtime enabled in both arms.
"""
import argparse
from collections import Counter
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
import time

# Preserve caller PYTHONPATH shims; append only this script's own directory for
# importlib-based test loading, where Python did not add it automatically.
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.append(str(Path(__file__).resolve().parent))
import public_eval as pe

SCHEMA = 1
ARMS = ('off', 'on')
ARM_LABELS = {'runtime': {'off': 'runtime off', 'on': 'runtime on'},
              'adoption': {'off': 'baseline (old guide)', 'on': 'candidate (adoption fixes)'}}
IMAGE_ID = re.compile(r'sha256:[0-9a-f]{64}\Z')
UNIT_ID = re.compile(r't1-[A-Za-z0-9_-]+\Z')
MODEL_FIELDS = ('MODEL_ENDPOINT', 'MODEL_NAME', 'MODEL_TOKEN')


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True,
                                     allow_nan=False, separators=(',', ':')).encode()).hexdigest()


def _file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _write(path, document):
    path = Path(path)
    if path.is_symlink():
        raise ValueError('Report path must not be a symlink')
    temporary = path.with_name('.' + path.name + '.tmp')
    with temporary.open('x', encoding='utf-8') as stream:
        json.dump(document, stream, indent=2, ensure_ascii=True, allow_nan=False)
        stream.write('\n')
    temporary.replace(path)


def inspect_image(reference, context):
    """Resolve one image without pulling or mutating tags; never emit Docker output."""
    command = ['docker', '--context', context, 'image', 'inspect', reference]
    process = subprocess.run(command, capture_output=True, text=True, timeout=60, check=False)
    if process.returncode:
        raise ValueError('Image inspection failed')
    try:
        values = json.loads(process.stdout)
        if len(values) != 1 or not isinstance(values[0], dict):
            raise ValueError
        return values[0]
    except (TypeError, ValueError, KeyError):
        raise ValueError('Image inspection returned invalid metadata') from None


def _environment(info):
    values = info.get('Config', {}).get('Env')
    if not isinstance(values, list):
        raise ValueError('Image environment is missing')
    result = {}
    for entry in values:
        if not isinstance(entry, str) or '=' not in entry:
            raise ValueError('Malformed image environment')
        key, value = entry.split('=', 1)
        if key in result or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', key):
            raise ValueError('Duplicate or invalid image environment key')
        result[key] = value
    return result


def compare_images(off, on, comparison='runtime'):
    """Require identical filesystem/configuration with one declared environment treatment."""
    if not isinstance(comparison, str) or comparison not in ARM_LABELS:
        raise ValueError('Unknown pilot comparison')
    for info in (off, on):
        if not IMAGE_ID.fullmatch(str(info.get('Id', ''))):
            raise ValueError('An actual immutable image ID is required')
        if info.get('Architecture') != 'amd64' or info.get('Os') != 'linux':
            raise ValueError('Both images must be Linux AMD64')
        if not isinstance(info.get('RootFS'), dict) or not info['RootFS'].get('Layers'):
            raise ValueError('Image RootFS metadata is missing')
        if not isinstance(info.get('Config'), dict):
            raise ValueError('Image execution configuration is missing')
    if off['RootFS'] != on['RootFS']:
        raise ValueError('Paired images have different RootFS layers')
    # Docker may omit unset optional fields (notably Entrypoint). Compare their
    # effective defaults, while retaining all other execution config fields.
    defaults = {'Cmd': None, 'Entrypoint': None, 'User': '', 'WorkingDir': ''}
    configs = [dict(defaults, **{k: v for k, v in info['Config'].items() if k != 'Env'})
               for info in (off, on)]
    if configs[0] != configs[1]:
        raise ValueError('Paired image execution configurations differ')
    environments = [_environment(info) for info in (off, on)]
    variant = 'T1_ADOPTION_VARIANT' if comparison == 'adoption' else 'T1_RUNTIME'
    values = ('baseline', 'candidate') if comparison == 'adoption' else ('0', '1')
    for expected, env in zip(values, environments):
        if env.get(variant) != expected or env.get('T1_S12_LIB') != '1':
            raise ValueError('Both arms require T1_S12_LIB=1 and their declared treatment values')
        if comparison == 'adoption' and env.get('T1_RUNTIME') != '1':
            raise ValueError('Adoption comparison requires runtime enabled in both arms')
    differences = {key for key in set(environments[0]) | set(environments[1])
                   if environments[0].get(key) != environments[1].get(key)}
    if differences != {variant}:
        raise ValueError('Only the declared treatment may differ between image environments')
    common_env = {k: v for k, v in environments[0].items() if k != variant}
    return {'comparison': comparison, 'arm_labels': dict(ARM_LABELS[comparison]),
            'images': {'off': off['Id'], 'on': on['Id']},
            'rootfs_sha256': _hash(off['RootFS']), 'execution_sha256': _hash(configs[0]),
            'common_environment_sha256': _hash(common_env), 'platform': 'linux/amd64'}


def ordered_units(spec, index):
    """Preserve an explicit roster's order; refuse classes, duplicates and unknown units."""
    chosen = re.split(r'[\s,]+', spec.strip()) if isinstance(spec, str) and spec.strip() else []
    if not chosen or len(chosen) != len(set(chosen)) or any(not UNIT_ID.fullmatch(x) for x in chosen):
        raise ValueError('Provide a nonempty explicit roster without duplicates')
    known = set()
    for entry in index.get('files', []):
        raw = entry.get('path')
        if (not isinstance(raw, str) or '\\' in raw
                or any(part in ('', '.', '..') for part in raw.split('/'))):
            raise ValueError('Invalid source index path')
        parts = PurePosixPath(raw).parts
        if len(parts) < 3 or parts[0] != 'units' or not UNIT_ID.fullmatch(parts[1]):
            raise ValueError('Invalid source index unit')
        known.add(parts[1])
    if not set(chosen) <= known:
        raise ValueError('The explicit roster contains unknown units')
    return chosen


def _route():
    # Token rotation is allowed; credentials are never serialized or hashed.
    if not os.environ.get('MODEL_ENDPOINT') or not os.environ.get('MODEL_NAME'):
        return None
    return _hash([os.environ['MODEL_ENDPOINT'], os.environ['MODEL_NAME'], pe.eval_backend()])


def _configuration(args, index):
    if (type(args.cpus) not in (int, float) or not math.isfinite(args.cpus) or args.cpus <= 0
            or not isinstance(args.memory, str) or not re.fullmatch(r'[1-9][0-9]*[kmg]', args.memory)
            or type(args.unit_timeout) is not int or args.unit_timeout <= 0
            or type(args.checker_timeout) is not int or args.checker_timeout <= 0):
        raise ValueError('Invalid pilot resource or time budget')
    if not re.fullmatch(r'[0-9a-f]{40}', str(index.get('commit', ''))):
        raise ValueError('A pinned source commit is required')
    compared = compare_images(inspect_image(args.image_off, args.context), inspect_image(args.image_on, args.context),
                              comparison=getattr(args, 'comparison', 'runtime'))
    return dict(compared, source_commit=index['commit'], index_sha256=_file_hash(args.index),
                units=ordered_units(args.units, index), context=args.context, cpus=float(args.cpus),
                memory=args.memory, unit_timeout=args.unit_timeout, checker_timeout=args.checker_timeout,
                explicit_budget=True, budget_mode='explicit', eval_throttle=False, eval_backend=pe.eval_backend(),
                model_route_sha256=_route(), parallel=1, repeats_per_arm=1, max_requests_per_attempt=20,
                harness_sha256=_file_hash(__file__), public_eval_sha256=_file_hash(pe.__file__))


def _tree_hash(root):
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError('Prepared data tree is missing or linked')
    rows = []
    for path in sorted(root.rglob('*')):
        mode = path.lstat().st_mode
        if stat.S_ISDIR(mode):
            continue
        if not stat.S_ISREG(mode) or path.stat().st_nlink != 1:
            raise ValueError('Prepared data contains links or special files')
        rows.append([path.relative_to(root).as_posix(), _file_hash(path)])
    return _hash(rows)


def _data_fingerprint(work):
    return {'dataset_manifest_sha256': _file_hash(work/'evidence/dataset_manifest.json'),
            'selected_index_sha256': _file_hash(work/'evidence/source_files.json'),
            'datasets_sha256': _tree_hash(work/'datasets'),
            'participant_sha256': _tree_hash(work/'participant')}


def _prefill(cache, selected, datasets):
    cache = Path(cache).resolve()
    for entry in selected['files']:
        path = PurePosixPath(entry['path'])
        relative = PurePosixPath(*path.parts[1:])
        source = cache.joinpath(*path.parts)
        if not source.exists():
            source = cache.joinpath(*relative.parts)
        if not source.is_file() or any(part.is_symlink() for part in (source, *source.parents)
                                       if part != cache.parent):
            raise ValueError('Source cache file is missing or linked')
        target = datasets.joinpath(*relative.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        target.chmod(int(entry['mode'][-3:], 8))


def _schedule(units):
    return [{'unit': unit, 'arm': arm, 'attempt': unit+'__'+arm}
            for index, unit in enumerate(units) for arm in (ARMS if index % 2 == 0 else ARMS[::-1])]


def _prepare(args, index, config, work, out):
    work.mkdir(parents=True, exist_ok=False)
    out.mkdir(parents=True, exist_ok=False)
    (work/'evidence').mkdir()
    selected = dict(index, files=[entry for entry in index['files']
                                 if PurePosixPath(entry['path']).parts[1] in config['units']])
    selected_path, manifest_path = work/'evidence/source_files.json', work/'evidence/dataset_manifest.json'
    _write(selected_path, selected)
    if args.source_cache:
        _prefill(args.source_cache, selected, work/'datasets')
    pe.fetch_sources(selected_path, work/'datasets', manifest_path, workers=args.workers)
    manifest = json.loads(manifest_path.read_text())
    pe.verify_dataset(work/'datasets', manifest, config['source_commit'])
    _, errors = pe.stage_units(work/'datasets', work/'participant', config['units'])
    if errors:
        raise ValueError('Participant staging failed; no attempt was executed')
    plan = {'schema_version': SCHEMA, 'config': config, 'work': str(work), 'out': str(out),
            'data': _data_fingerprint(work), 'schedule': _schedule(config['units'])}
    plan['sha256'] = _hash(plan)
    _write(out/'plan.json', plan)
    return plan


def _validate_plan(plan, config, work, out):
    if (not isinstance(plan, dict) or plan.get('schema_version') != SCHEMA
            or plan.get('sha256') != _hash({k: v for k, v in plan.items() if k != 'sha256'})
            or plan.get('config') != config or plan.get('work') != str(work) or plan.get('out') != str(out)
            or plan.get('schedule') != _schedule(config['units'])):
        raise ValueError('Prepared pilot configuration changed')
    manifest = json.loads((work/'evidence/dataset_manifest.json').read_text())
    pe.verify_dataset(work/'datasets', manifest, config['source_commit'])
    if plan.get('data') != _data_fingerprint(work):
        raise ValueError('Prepared source or participant data changed')


def _measured(row):
    return (not str(row.get('status', '')).startswith(pe.NOT_MEASURED)
            and not str(row.get('verdict', '')).startswith(('checker_', 'infrastructure_', 'not_run'))
            and row.get('status') not in (None, '', 'executed_ungraded', 'missing_status'))


def comparison(rows, units, mode='runtime'):
    """Only complete measured pairs contribute discordances; unmatched rows remain visible."""
    if not isinstance(mode, str) or mode not in ARM_LABELS:
        raise ValueError('Unknown pilot comparison')
    pairs = {unit: {} for unit in units}
    for row in rows:
        if row.get('id') not in pairs or row.get('arm') not in ARMS:
            raise ValueError('Unexpected unit or arm in paired results')
        if row['arm'] in pairs[row['id']]:
            raise ValueError('Duplicate unit/arm results are not a valid paired pilot')
        pairs[row['id']][row['arm']] = row
    result = dict(mode=mode, arm_labels=dict(ARM_LABELS[mode]),
                  complete_pairs=0, both_pass=0, both_fail=0, on_only_pass=0, off_only_pass=0)
    for pair in pairs.values():
        if set(pair) != set(ARMS) or not all(_measured(row) for row in pair.values()):
            continue
        result['complete_pairs'] += 1
        off, on = (pair[arm]['status'] == 'passed' for arm in ARMS)
        result['both_pass' if off and on else 'on_only_pass' if on else
               'off_only_pass' if off else 'both_fail'] += 1
    result['unmatched_units'] = len(units) - result['complete_pairs']
    result['arms'] = {}
    for arm in ARMS:
        group = [row for row in rows if row.get('arm') == arm]
        result['arms'][arm] = {
            'attempted': len(group), 'measured': sum(_measured(row) for row in group),
            'passed': sum(row.get('status') == 'passed' for row in group),
            'requests': sum(row.get('model_calls', 0) for row in group
                            if type(row.get('model_calls')) is int and row['model_calls'] >= 0),
            'unknown_request_attempts': sum(type(row.get('model_calls')) is not int
                                             or row['model_calls'] < 0 for row in group),
            'duration_sec': round(sum(row.get('duration_sec', 0.) for row in group
                                      if type(row.get('duration_sec', 0.)) in (int, float)), 3),
            'runtime_shown': sum(row.get('runtime_shown') is True for row in group),
            'runtime_states': dict(Counter(row.get('runtime_state', 'not_reported') for row in group)),
            'statuses': dict(Counter(row.get('status', 'unknown') for row in group))}
    return result


def _save_result(out, result):
    _write(out/'results.json', result)
    compare = result['comparison']
    labels = compare['arm_labels']
    title = 'Adoption guidance' if compare['mode'] == 'adoption' else 'Runtime'
    lines = [f'# {title} paired development pilot', '', f"State: {result['status']}", '',
             f"Comparison: {compare['mode']}; off = {labels['off']}; on = {labels['on']}.", '',
             'Local public-checker evidence; not an official House score or a significance claim.', '',
             'Runtime shown means guidance exposure. Report states are untrusted diagnostics, not correctness: '
             'missing, empty and skip-only reports remain unverified; nonempty reports may contain only internal errors. '
             'Passed counts use the checker outcome.', '',
             f"Complete measured pairs: {compare['complete_pairs']}; unmatched units: {compare['unmatched_units']}.",
             f"Both pass: {compare['both_pass']}; both fail: {compare['both_fail']}; "
             f"{labels['on']} only pass: {compare['on_only_pass']}; "
             f"{labels['off']} only pass: {compare['off_only_pass']}.", '',
             '| Arm | Attempts | Measured | Passed | Known requests | Unknown request attempts | Seconds | Runtime shown | Check states |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for arm in ARMS:
        row = compare['arms'][arm]
        lines.append(f"| {arm}: {labels[arm]} | {row['attempted']} | {row['measured']} | {row['passed']} | "
                     f"{row['requests']} | {row['unknown_request_attempts']} | {row['duration_sec']} | {row['runtime_shown']} | "
                     f"{json.dumps(row['runtime_states'], sort_keys=True)} |")
    lines.extend(['', '| Unit | Arm | Status | Runtime shown | Check state | Requests | Seconds |',
                  '|---|---|---|---|---|---:|---:|'])
    for row in result['attempts']:
        lines.append(f"| {row['id']} | {row['arm']}: {labels[row['arm']]} | {row.get('status')} | "
                     f"{row.get('runtime_shown', False)} | {row.get('runtime_state', 'not_reported')} | "
                     f"{row.get('model_calls', 0)} | {row.get('duration_sec', 0)} |")
    (out/'comparison.md').write_text('\n'.join(lines)+'\n')


@contextmanager
def _without_throttle():
    previous = os.environ.get('T1_EVAL_THROTTLE')
    os.environ['T1_EVAL_THROTTLE'] = '0'
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop('T1_EVAL_THROTTLE', None)
        else:
            os.environ['T1_EVAL_THROTTLE'] = previous


def run(args):
    work, out = Path(args.work).resolve(), Path(args.out).resolve()
    if work == out or work.is_relative_to(out) or out.is_relative_to(work):
        raise ValueError('Work and output must be distinct non-nested directories')
    index = json.loads(Path(args.index).read_text())
    config = _configuration(args, index)
    plan_path = out/'plan.json'
    if plan_path.exists():
        if not args.execute:
            raise ValueError('Preparation requires fresh directories')
        plan = json.loads(plan_path.read_text())
        _validate_plan(plan, config, work, out)
    else:
        if work.exists() or out.exists():
            raise ValueError('Pilot requires fresh work and output directories')
        plan = _prepare(args, index, config, work, out)
    result = {'schema_version': SCHEMA, 'plan_sha256': plan['sha256'], 'status': 'prepared',
              'scope': 'paired development pilot; not official House or statistical proof',
              'attempts': [], 'comparison': comparison([], config['units'], config['comparison'])}
    if not args.execute:
        _save_result(out, result)
        return result
    if any(not os.environ.get(key) for key in MODEL_FIELDS) or config['model_route_sha256'] is None:
        raise ValueError('Complete MODEL configuration is required; credentials must remain in the environment')
    for arm in ARMS:
        receipt_path = getattr(args, 'gate_'+arm)
        if (receipt_path is None or not pe.gate_passed(json.loads(Path(receipt_path).read_text()),
                                                      config['images'][arm])):
            raise ValueError('A passing isolation receipt is required for each actual image ID')
    # Exclusive claim prevents selective restart after success, failure, or interruption.
    try:
        with (out/'execution.started.json').open('x', encoding='utf-8') as stream:
            json.dump({'plan_sha256': plan['sha256'], 'started_unix': time.time()}, stream)
    except FileExistsError:
        raise ValueError('The frozen pilot has already started; selective restart is refused') from None
    execution = argparse.Namespace(context=args.context, cpus=args.cpus, memory=args.memory,
                                  unit_timeout=args.unit_timeout, checker_timeout=args.checker_timeout,
                                  explicit_budget=True, replay_pubkey=None)
    result['status'] = 'running'
    _save_result(out, result)
    with _without_throttle():
        for item in plan['schedule']:
            started = time.monotonic()
            try:
                row = pe.evaluate_unit(item['unit'], config['images'][item['arm']], work, execution,
                                       attempt=item['attempt'])
            except Exception:
                row = {'id': item['unit'], 'status': 'infrastructure_eval_error', 'model_calls': None,
                       'duration_sec': round(time.monotonic()-started, 3)}
            row = dict(row, id=item['unit'], arm=item['arm'], attempt=item['attempt'])
            result['attempts'].append(row)
            result['comparison'] = comparison(result['attempts'], config['units'], config['comparison'])
            if not _measured(row):
                result['status'] = 'partial_infrastructure'
                _save_result(out, result)
                return result
            _save_result(out, result)
    result['status'] = 'completed'
    _save_result(out, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison', choices=tuple(ARM_LABELS), default='runtime',
                        help='runtime: off/on; adoption: baseline/candidate guidance, both runtime enabled')
    for arm in ARMS:
        parser.add_argument('--image-'+arm, required=True)
        parser.add_argument('--gate-'+arm, type=Path)
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--units', required=True, help='Explicit unit IDs in the predeclared order')
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--context', default='default')
    parser.add_argument('--cpus', type=float, default=2)
    parser.add_argument('--memory', default='6g')
    parser.add_argument('--unit-timeout', type=int, default=420)
    parser.add_argument('--checker-timeout', type=int, default=300)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--source-cache', type=Path)
    parser.add_argument('--execute', action='store_true', help='Allow the predeclared serial container/model runs')
    args = parser.parse_args(argv)
    try:
        result = run(args)
    except (OSError, ValueError, TypeError, KeyError, subprocess.SubprocessError):
        # Never print exception text, raw image config, endpoint, or credentials.
        print('Pilot refused or preparation failed; inspect the frozen plan, source hashes, image pair and gates.',
              file=sys.stderr)
        return 2
    print(json.dumps({'status': result['status'], 'comparison': result['comparison']}, sort_keys=True))
    return 1 if result['status'] == 'partial_infrastructure' else 0


if __name__ == '__main__':
    raise SystemExit(main())
