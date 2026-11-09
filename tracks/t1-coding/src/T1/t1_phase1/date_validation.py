"""Bounded, same-run observation of inferred pandas date-parse loss.

Only the worker imports pandas or examines runtime values. Observations do not
change parsing results or prove a task's date policy. Main-process calls only;
worker subprocesses and alternative date parsers are outside this monitor.
"""
from __future__ import annotations

import ast
import inspect
import json
import os
import re
import stat
from pathlib import Path

SCHEMA = 'date-validation-v1'
REPORT_BYTES = 65536


def _install_worker_monitor():
    # This self-contained function's source is copied into worker sitecustomize.
    import atexit
    import functools
    import inspect
    import json
    import os
    import sys
    import warnings

    path = os.environ.get('T1_DATE_REPORT')
    if not path:
        return
    pid = os.getpid()
    # Spawned Python children inherit this marker; forked children also fail the
    # pid check below. Their site startup must not replace the main report.
    owner = os.environ.get('T1_DATE_ROOT_PID')
    if owner is not None and owner != str(pid):
        return
    os.environ['T1_DATE_ROOT_PID'] = str(pid)
    report = {'schema': 'date-validation-v1', 'status': 'unverified',
              'scope': 'main_process', 'reason': 'worker_not_finished',
              'calls_observed': 0, 'unresolved_inferred_date_loss': 0, 'observations': []}
    broken = []

    def save():
        if os.getpid() != pid:
            return
        try:
            with open(path, 'w') as stream:
                stream.write(json.dumps(report, ensure_ascii=True))
        except Exception:
            pass  # Missing/invalid report is unverified on the controller.

    def finish():
        if os.getpid() != pid:
            return
        report['status'] = 'unverified' if broken else 'complete'
        report['reason'] = broken[0] if broken else None
        save()

    def fail(reason):
        if not broken:
            broken.append(reason)
        report['reason'] = broken[0]
        save()

    save()
    atexit.register(finish)
    try:
        import pandas as pd
        original = pd.to_datetime
        signature = inspect.signature(original)

        def observe(arg, result, bound, line):
            # No slicing/sampling is presented as exhaustive verification.
            if pd.api.types.is_scalar(arg):
                given, got = [arg], [result]
            elif isinstance(arg, (pd.DataFrame, dict)):
                fail('unsupported_date_input')
                return
            else:
                if len(arg) > 20000:
                    fail('date_input_limit')
                    return
                if getattr(arg, 'ndim', 1) != 1 or getattr(result, 'ndim', 1) != 1:
                    fail('unsupported_date_input')
                    return
                given, got = list(arg), list(result)
            if len(given) != len(got):
                fail('date_shape_mismatch')
                return
            nonmissing, lost, positions = 0, [], []
            for position, (value, parsed) in enumerate(zip(given, got)):
                absent = bool(pd.isna(value)) or (isinstance(value, str) and not value.strip())
                if not absent:
                    nonmissing += 1
                    if bool(pd.isna(parsed)):
                        lost.append(value)
                        if len(positions) < 3:
                            positions.append(position)
            if not lost:
                return
            explicit = bound.get('format') is not None
            recovered = 0
            if not explicit:
                # Parse only text: never reinterpret numeric epochs/units as dates.
                texts = [value for value in lost if isinstance(value, str)]
                if texts:
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        mixed = original(texts, errors='coerce', format='mixed',
                                         dayfirst=bound.get('dayfirst', False),
                                         yearfirst=bound.get('yearfirst', False),
                                         utc=bound.get('utc', False))
                    recovered = int(mixed.notna().sum())
            if len(report['observations']) >= 64:
                fail('date_observation_limit')
                return
            # Values never enter the side report: truncating raw text before
            # canary removal could leak the prefix of a protected marker.
            examples = ['position=' + str(position) for position in positions]
            report['observations'].append({'line': line, 'format_explicit': explicit,
                'nonmissing': nonmissing, 'new_missing': len(lost),
                'mixed_recoverable': recovered, 'examples': examples})
            report['unresolved_inferred_date_loss'] += recovered
            save()

        @functools.wraps(original)
        def audited(*args, **kwargs):
            # Match Python's real argument binding, including positional format.
            result = original(*args, **kwargs)
            if os.getpid() != pid:
                return result
            try:
                frame = sys._getframe(1)
                program = os.environ.get('T1_PROGRAM')
                while frame is not None and frame.f_code.co_filename != program:
                    frame = frame.f_back
                if frame is None:
                    return result
                if report['calls_observed'] >= 1024:
                    fail('date_call_limit')
                    return result
                report['calls_observed'] += 1
                bound = signature.bind(*args, **kwargs).arguments
                if bound.get('errors', 'raise') == 'coerce':
                    observe(bound['arg'], result, bound, frame.f_lineno)
            except Exception:
                fail('date_observation_error')
            return result

        audited.__module__, audited.__qualname__ = 'pandas', 'to_datetime'
        pd.to_datetime = audited
    except Exception:
        fail('date_monitor_setup_error')


def worker_source() -> str:
    """Trusted monitor source; avoids importing controller modules in the worker."""
    return inspect.getsource(_install_worker_monitor) + '\n_install_worker_monitor()\n'


def unverified(reason='missing_report') -> dict:
    return {'schema': SCHEMA, 'status': 'unverified', 'scope': 'main_process', 'reason': reason,
            'calls_observed': 0, 'unresolved_inferred_date_loss': 0, 'observations': []}


def read_report(path: Path) -> dict:
    """Read bounded worker-owned diagnostic data, never code or a grading verdict."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, 'rb') as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                return unverified('invalid_report')
            raw = stream.read(REPORT_BYTES + 1)
        if len(raw) > REPORT_BYTES:
            return unverified('report_limit')
        def unique(pairs):
            obj = {}
            for key, value in pairs:
                if key in obj:
                    raise ValueError('duplicate key')
                obj[key] = value
            return obj
        data = json.loads(raw, object_pairs_hook=unique)
        if not isinstance(data, dict) or data.get('schema') != SCHEMA or data.get('scope') != 'main_process':
            raise ValueError('schema')
        if data.get('status') not in ('complete', 'unverified'):
            raise ValueError('status')
        reasons = ('worker_not_finished', 'unsupported_date_input', 'date_input_limit', 'date_shape_mismatch',
                   'date_observation_limit', 'date_call_limit', 'date_observation_error', 'date_monitor_setup_error')
        if ((data['status'] == 'complete' and data.get('reason') is not None) or
                (data['status'] == 'unverified' and data.get('reason') not in reasons)):
            raise ValueError('reason')
        calls, unresolved, rows = data.get('calls_observed'), data.get('unresolved_inferred_date_loss'), data.get('observations')
        if type(calls) is not int or not 0 <= calls <= 1024 or type(unresolved) is not int or unresolved < 0:
            raise ValueError('counts')
        if not isinstance(rows, list) or len(rows) > min(64, calls):
            raise ValueError('rows')
        clean = []
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError('row')
            line, count, lost, recover = [row.get(k) for k in ('line', 'nonmissing', 'new_missing', 'mixed_recoverable')]
            if not all(type(v) is int for v in (line, count, lost, recover)) or not (0 <= line <= 1000000 and 0 <= recover <= lost <= count <= 20000):
                raise ValueError('row counts')
            explicit, examples = row.get('format_explicit'), row.get('examples')
            if type(explicit) is not bool or (explicit and recover) or not isinstance(examples, list) or len(examples) > 3:
                raise ValueError('row fields')
            if not all(isinstance(v, str) and re.fullmatch(r'position=[0-9]{1,5}', v)
                       and int(v.split('=')[1]) < 20000 for v in examples):
                raise ValueError('examples')
            clean.append(dict(line=line, format_explicit=explicit, nonmissing=count,
                              new_missing=lost, mixed_recoverable=recover, examples=examples))
        if sum(row['mixed_recoverable'] for row in clean) != unresolved:
            raise ValueError('unresolved count')
        return dict(schema=SCHEMA, status=data['status'], scope='main_process', reason=data.get('reason'),
                    calls_observed=calls, unresolved_inferred_date_loss=unresolved, observations=clean)
    except OSError:
        return unverified('missing_report')
    except (ValueError, TypeError, RecursionError, UnicodeError):
        return unverified('invalid_report')


def _relevant(code: str) -> bool:
    """Potential inferred/coerce call, not a proof of complete parser coverage.

    Literal explicit formats and raise/default errors do not need this guard.
    Dynamic keyword arguments remain uncertain. This only gates an unverified
    report; an observed unresolved loss still blocks independently.
    """
    try:
        tree = ast.parse(code)
        aliases = {'to_datetime'} | {n.asname or n.name for n in ast.walk(tree)
                                     if isinstance(n, ast.alias) and n.name == 'to_datetime'}
        def target(node):
            return ((isinstance(node, ast.Attribute) and node.attr == 'to_datetime') or
                    (isinstance(node, ast.Name) and node.id in aliases) or
                    (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                     and node.func.id == 'getattr' and len(node.args) > 1
                     and isinstance(node.args[1], ast.Constant) and node.args[1].value == 'to_datetime'))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            direct = target(node.func)
            if not direct and not any(target(arg) for arg in node.args):
                continue
            kwargs = {item.arg: item.value for item in node.keywords}
            if None in kwargs or any(isinstance(arg, ast.Starred) for arg in node.args):
                return True
            errors = kwargs.get('errors', node.args[1] if direct and len(node.args) > 1 else ast.Constant('raise'))
            fmt = kwargs.get('format', node.args[5] if direct and len(node.args) > 5 else ast.Constant(None))
            if isinstance(errors, ast.Constant) and errors.value != 'coerce':
                continue
            if isinstance(fmt, ast.Constant) and fmt.value is not None:
                continue
            return True
        return False
    except (SyntaxError, ValueError, RecursionError):
        return 'to_datetime' in code


def flags(report: dict, code: str) -> list[str]:
    """Blocking observations, separate from the worker's real execution status."""
    result = []
    if report.get('unresolved_inferred_date_loss', 0):
        result.append('unresolved_inferred_date_loss: inferred date parsing discarded nonmissing values '
                      'that parse with mixed formats. Specify the intended format explicitly (strict or mixed) '
                      'according to the TASK; do not automatically recover or drop values.')
    if report.get('status') != 'complete' and _relevant(code):
        result.append('date_validation_unverified: date observations did not complete; inspect the diagnostic '
                      'and validate the date conversion before finishing.')
    return result


def feedback(report: dict) -> str:
    """Bounded untrusted runtime observations, never official answer expectations."""
    header = ('Date observations (main process only; not a task verdict): '
              + str(report.get('status', 'unverified')) + '; reason=' + str(report.get('reason'))
              + '. Observed conversion loss does not establish that final outputs are wrong.')
    rows = []
    for row in report.get('observations', [])[:16]:
        rows.append('line {line}: explicit_format={format_explicit}, nonmissing={nonmissing}, '
                    'new_missing={new_missing}, mixed_recoverable={mixed_recoverable}, examples={examples}'.format(**row))
    return (header + '\n' + '\n'.join(rows))[:5000]
