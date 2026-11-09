#!/usr/bin/env python3
"""Compare narrow semantic parsing with source retention, without solving tasks.

Only direct */instruction.md files are inputs. No checker, task code, reference
answer or model is loaded. A retained source is not an understood requirement.
Reports contain fixed status/error categories, counts and hashes, never source
quotes or exception messages.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import tempfile


PACKAGE = Path(__file__).resolve().parents[1] / 'qfruntime'
sys.path.insert(0, str(PACKAGE))
from qfruntime.requirements import MAX_SOURCE_BYTES, RequirementPlan
from qfruntime.semantics import extract_task_spec


def _hash(raw):
    return hashlib.sha256(raw).hexdigest()


def _blank_row(task_id):
    return {
        'task_id': task_id, 'source_sha256': None, 'source_bytes': None,
        'input_error_category': None, 'spec_status': 'not_run',
        'quantities': None, 'semantic_unresolved': None, 'spec_error_category': None,
        'semantic_source_index_state': None, 'semantic_unresolved_total': None,
        'semantic_unresolved_omitted': None,
        'plan_status': 'not_run', 'plan_error_category': None,
        'blocks': None, 'unassigned': None, 'exact_source_retained': None,
        'source_hash_matches': None, 'block_spans_valid': None,
        'all_blocks_unassigned': None, 'all_nonwhitespace_accounted': None,
        'nonwhitespace_characters': None, 'accounted_nonwhitespace_characters': None,
    }


def _read_source(path, row):
    if path.parent.is_symlink():
        row['input_error_category'] = 'nonregular_source'
        return None
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, 'rb') as stream:
            info = os.fstat(stream.fileno())
            row['source_bytes'] = info.st_size
            if not stat.S_ISREG(info.st_mode):
                row['input_error_category'] = 'nonregular_source'
                return None
            if info.st_size > MAX_SOURCE_BYTES:
                row['input_error_category'] = 'source_too_large'
                return None
            raw = stream.read(MAX_SOURCE_BYTES + 1)
        if len(raw) > MAX_SOURCE_BYTES:
            row['input_error_category'] = 'source_too_large'
            return None
        row['source_bytes'] = len(raw)
        row['source_sha256'] = _hash(raw)
        return raw.decode('utf-8')
    except UnicodeDecodeError:
        row['input_error_category'] = 'invalid_utf8'
    except OSError:
        row['input_error_category'] = 'source_unreadable'
    return None


def _semantic_status(source, row):
    try:
        spec = extract_task_spec(source)
        coverage = spec.unresolved_coverage
        row.update(spec_status='parsed', quantities=len(spec.quantities),
                   semantic_unresolved=len(spec.unresolved),
                   semantic_source_index_state='partial' if coverage is not None else 'complete',
                   semantic_unresolved_total=coverage['total'] if coverage is not None else len(spec.unresolved),
                   semantic_unresolved_omitted=coverage['omitted'] if coverage is not None else 0)
    except ValueError as error:
        category = {'too many unresolved requirements': 'unresolved_limit',
                    'instruction too large': 'source_too_large'}.get(str(error), 'parser_rejected')
        row.update(spec_status='failed', spec_error_category=category)
    except Exception:
        row.update(spec_status='failed', spec_error_category='parser_error')


def _retention_status(source, row):
    try:
        plan = RequirementPlan(source)
        blocks, unassigned = plan.blocks, plan.unassigned
        accounted = bytearray(len(source))
        valid, ids, previous_end = True, [], 0
        for block in blocks:
            start, end = block.get('start'), block.get('end')
            block_id, quote = block.get('id'), block.get('quote')
            if (type(start) is not int or type(end) is not int
                    or not 0 <= start < end <= len(source) or start < previous_end
                    or type(quote) is not str or source[start:end] != quote
                    or not quote.strip() or len(quote) > 500
                    or type(block_id) is not str or not block_id or block_id in ids
                    or block.get('source_id') != plan.source_id):
                valid = False
                continue
            accounted[start:end] = b'\x01' * (end-start)
            ids.append(block_id)
            previous_end = end
        nonwhitespace = sum(not char.isspace() for char in source)
        retained = sum(bool(accounted[index]) and not char.isspace()
                       for index, char in enumerate(source))
        row.update(
            blocks=len(blocks), unassigned=len(unassigned),
            exact_source_retained=plan.source == source,
            source_hash_matches=plan.source_sha256 == row['source_sha256'],
            block_spans_valid=bool(blocks) and valid,
            all_blocks_unassigned=list(unassigned) == ids and len(ids) == len(blocks),
            all_nonwhitespace_accounted=valid and nonwhitespace > 0 and retained == nonwhitespace,
            nonwhitespace_characters=nonwhitespace, accounted_nonwhitespace_characters=retained,
        )
        checks = ('exact_source_retained', 'source_hash_matches', 'block_spans_valid',
                  'all_blocks_unassigned', 'all_nonwhitespace_accounted')
        row['plan_status'] = 'retained' if all(row[key] is True for key in checks) else 'invalid'
        if row['plan_status'] == 'invalid':
            row['plan_error_category'] = 'invalid_source_accounting'
    except ValueError:
        row.update(plan_status='failed', plan_error_category='plan_rejected')
    except Exception:
        row.update(plan_status='failed', plan_error_category='plan_error')


def audit_units(units_root):
    """Read exact UTF-8 instruction bytes and retain failures as unknown counts."""
    root = Path(units_root)
    if not root.is_dir():
        raise ValueError('units_root_unavailable')
    rows = []
    for path in sorted(root.glob('*/instruction.md')):
        row = _blank_row(path.parent.name)
        source = _read_source(path, row)
        if source is not None:
            _semantic_status(source, row)
            _retention_status(source, row)
        rows.append(row)
    complete = bool(rows) and all(row['plan_status'] == 'retained' for row in rows)
    code = {
        'audit_sha256': _hash(Path(__file__).read_bytes()),
        'semantics_sha256': _hash((PACKAGE / 'qfruntime/semantics.py').read_bytes()),
        'requirements_sha256': _hash((PACKAGE / 'qfruntime/requirements.py').read_bytes()),
    }
    return {
        'schema_version': 1, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'source_retention_only', 'status': 'completed' if complete else ('partial' if rows else 'empty'),
        'all_sources_retained': complete, 'semantic_improvement_measured': False,
        'model_calls': 0, 'task_executions': 0, 'checker_executions': 0,
        'task_count': len(rows), 'units_root': str(root.resolve()), 'code': code,
        'summary': {
            'spec_states': dict(sorted(Counter(row['spec_status'] for row in rows).items())),
            'spec_errors': dict(sorted(Counter(row['spec_error_category'] for row in rows
                                              if row['spec_error_category']).items())),
            'plan_states': dict(sorted(Counter(row['plan_status'] for row in rows).items())),
            'quantities_in_parsed_specs': sum(row['quantities'] for row in rows if row['quantities'] is not None),
            'tasks_with_unknown_quantity_count': sum(row['quantities'] is None for row in rows),
            'retained_blocks': sum(row['blocks'] for row in rows if row['plan_status'] == 'retained'),
            'unassigned_blocks': sum(row['unassigned'] for row in rows if row['unassigned'] is not None),
        },
        'tasks': rows,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--units-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.is_symlink() or args.output.resolve() in {
            path.resolve() for path in args.units_root.glob('*/instruction.md')}:
        parser.error('output_must_not_replace_source_or_symlink')
    try:
        report = audit_units(args.units_root)
    except (OSError, ValueError):
        parser.error('units_root_unavailable')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.requirements-audit-', dir=args.output.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(report, stream, ensure_ascii=True, allow_nan=False, indent=2)
            stream.write('\n')
        os.replace(temporary, args.output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    print(json.dumps({'status': report['status'], 'task_count': report['task_count'],
                      'all_sources_retained': report['all_sources_retained']}))
    return 0 if report['all_sources_retained'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
