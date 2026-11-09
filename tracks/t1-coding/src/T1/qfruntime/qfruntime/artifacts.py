"""Bounded, read-only artifact structure checks; no numerical correctness claim.

JSON schemas support only type, nullable, object properties/required and array
items. All array elements are checked. Extra object properties are permitted,
but all JSON content must be finite, nonduplicate and within the depth limit.
JSON integer means a JSON integer token (2.0 does not satisfy integer).

CSV is UTF-8, comma-delimited with a header. columns is the complete expected
set; ordered=True additionally requires its order. Rows count data records.
Numeric fields accept surrounding whitespace; integers use signed decimal
digits, numbers decimal/exponent notation and booleans true/false ignoring case.
Empty numeric/boolean cells fail; no implicit null or missing-value convention.

Every format defaults to a 4 MiB read limit (max_bytes may select 1..64 MiB).
At most 32 diagnostics, 32 nested levels and 100,000 JSON nodes are inspected.
Schema traversal and integer tokens are capped at 4,096 nodes/characters. CSV field parser limits, unsupported
features and resource limits stay unverified. Returned hashes bind the complete
bytes inspected; no hash is returned for partial or concurrently changed reads.
Paths are relative to root, with no symlinks anywhere in root or artifact paths.
"""
from __future__ import annotations

import csv
import errno
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import stat

__all__ = ['validate_artifact']

_DEFAULT_BYTES = 4 * 1024 * 1024
_HARD_BYTES = 64 * 1024 * 1024
_TYPES = frozenset(('object', 'array', 'number', 'integer', 'boolean', 'string'))
_SCALARS = _TYPES - {'object', 'array'}
_INTEGER = re.compile(r'[+-]?[0-9]+\Z')
_NUMBER = re.compile(r'[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\Z')


class _Issue(Exception):
    def __init__(self, code, location='', status='failed'):
        self.code, self.location, self.status = code, location, status


def _pointer(path, part):
    return path + '/' + str(part).replace('~', '~0').replace('/', '~1')


def _reason(code, location):
    return {'code': code, 'location': location[:512]}


def _schema(schema, depth=0, count=None):
    count = [0] if count is None else count
    count[0] += 1
    if depth > 32 or count[0] > 4096:
        raise _Issue('schema_limit', status='unverified')
    if (type(schema) is not dict or type(schema.get('type')) is not str
            or schema['type'] not in _TYPES):
        raise _Issue('invalid_schema')
    if set(schema) - {'type', 'nullable', 'properties', 'required', 'items'}:
        raise _Issue('unsupported_schema', status='unverified')
    if 'nullable' in schema and type(schema['nullable']) is not bool:
        raise _Issue('invalid_schema')
    kind = schema['type']
    if kind != 'object' and ('properties' in schema or 'required' in schema):
        raise _Issue('invalid_schema')
    if kind != 'array' and 'items' in schema:
        raise _Issue('invalid_schema')
    if kind == 'object':
        properties, required = schema.get('properties', {}), schema.get('required', [])
        if (type(properties) is not dict or any(type(key) is not str for key in properties)
                or type(required) is not list or any(type(key) is not str for key in required)
                or len(set(required)) != len(required)):
            raise _Issue('invalid_schema')
        for child in properties.values():
            _schema(child, depth + 1, count)
    if kind == 'array':
        if 'items' not in schema:
            raise _Issue('invalid_schema')
        _schema(schema['items'], depth + 1, count)


def _contract(contract):
    if type(contract) is not dict:
        raise _Issue('invalid_contract')
    kind = contract.get('format')
    if kind not in ('json', 'csv', 'file'):
        raise _Issue('unsupported_format', status='unverified')
    extra = {'json': {'schema'}, 'csv': {'columns', 'types', 'rows', 'ordered'}, 'file': set()}[kind]
    if set(contract) - ({'path', 'format', 'max_bytes'} | extra):
        raise _Issue('unsupported_contract', status='unverified')
    name = contract.get('path')
    if (type(name) is not str or not name or len(name) > 4096 or '\x00' in name
            or '\\' in name or any(part in ('', '.', '..') for part in name.split('/'))):
        raise _Issue('unsafe_path')
    limit = contract.get('max_bytes', _DEFAULT_BYTES)
    if type(limit) is not int or limit <= 0:
        raise _Issue('invalid_limit')
    if limit > _HARD_BYTES:
        raise _Issue('size_limit', status='unverified')
    if kind == 'json':
        _schema(contract.get('schema'))
    elif kind == 'csv':
        columns, types = contract.get('columns'), contract.get('types', {})
        if (type(columns) is not list or not columns or len(columns) > 4096
                or any(type(column) is not str or not column for column in columns)
                or len(set(columns)) != len(columns) or type(types) is not dict
                or any(key not in columns or type(value) is not str or value not in _SCALARS
                       for key, value in types.items())
                or ('ordered' in contract and type(contract['ordered']) is not bool)
                or ('rows' in contract and (type(contract['rows']) is not int or contract['rows'] < 0))):
            raise _Issue('invalid_contract')
    return name, kind, limit


def _read(root, name, limit):
    """Descend with no-follow directory descriptors, then inspect one regular fd."""
    descriptors = []
    try:
        directory = Path(os.path.abspath(os.fspath(root)))
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        current = os.open(directory.anchor, flags)
        descriptors.append(current)
        for component in directory.parts[1:] + tuple(name.split('/')[:-1]):
            current = os.open(component, flags, dir_fd=current)
            descriptors.append(current)
        fd = os.open(name.split('/')[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=current)
        descriptors.append(fd)
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise _Issue('not_regular_file')
        if before.st_size > limit:
            raise _Issue('size_limit', status='unverified')
        chunks, size = [], 0
        while size <= limit:
            chunk = os.read(fd, min(65536, limit + 1 - size))
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
        if size > limit:
            raise _Issue('size_limit', status='unverified')
        after = os.fstat(fd)
        fields = ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
        if size != before.st_size or any(getattr(before, key) != getattr(after, key) for key in fields):
            raise _Issue('file_changed', status='unverified')
        return b''.join(chunks)
    except FileNotFoundError as exc:
        raise _Issue('missing_file') from exc
    except OSError as exc:
        code = 'unsafe_path' if exc.errno in (errno.ELOOP, errno.ENOTDIR) else 'unreadable_file'
        raise _Issue(code) from exc
    except (TypeError, ValueError) as exc:
        raise _Issue('unsafe_path') from exc
    finally:
        for fd in reversed(descriptors):
            os.close(fd)


def _pairs(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise _Issue('duplicate_key')
        out[key] = value
    return out


def _constant(_):
    raise _Issue('nonfinite_json')


def _integer(text):
    if len(text) > 4096:
        raise _Issue('numeric_parse_limit', status='unverified')
    try:
        return int(text)
    except ValueError as exc:
        raise _Issue('numeric_parse_limit', status='unverified') from exc


def _json_bounds(value, location='', depth=0, count=None):
    count = [0] if count is None else count
    count[0] += 1
    if depth > 32 or count[0] > 100000:
        raise _Issue('structure_limit', location, 'unverified')
    if type(value) is float and not math.isfinite(value):
        raise _Issue('nonfinite_json', location)
    if type(value) is dict:
        for key, child in value.items():
            _json_bounds(child, _pointer(location, key), depth + 1, count)
    elif type(value) is list:
        for index, child in enumerate(value):
            _json_bounds(child, _pointer(location, index), depth + 1, count)


def _validate_json(value, schema, reasons, location=''):
    if len(reasons) >= 32:
        return
    if value is None and schema.get('nullable', False):
        return
    kind = schema['type']
    expected = {'object': dict, 'array': list, 'integer': int, 'boolean': bool, 'string': str}
    valid = type(value) in (int, float) if kind == 'number' else type(value) is expected[kind]
    if not valid:
        reasons.append(_reason('type_mismatch', location))
        return
    if kind == 'object':
        for key in schema.get('required', []):
            if key not in value and len(reasons) < 32:
                reasons.append(_reason('missing_required', _pointer(location, key)))
        for key, child in schema.get('properties', {}).items():
            if key in value:
                _validate_json(value[key], child, reasons, _pointer(location, key))
    elif kind == 'array':
        for index, child in enumerate(value):
            _validate_json(child, schema['items'], reasons, _pointer(location, index))
            if len(reasons) >= 32:
                break


def _cell(value, kind):
    text = value.strip()
    if kind == 'string':
        return True
    if kind == 'boolean':
        return text.lower() in ('true', 'false')
    if kind == 'integer':
        return _INTEGER.fullmatch(text) is not None
    return _NUMBER.fullmatch(text) is not None and math.isfinite(float(text))


def _validate_csv(text, contract, reasons):
    reader = csv.reader(io.StringIO(text, newline=''), strict=True)
    header = next(reader, None)
    if header is None:
        reasons.append(_reason('missing_header', ''))
        return
    columns = contract['columns']
    def add(code, location):
        if len(reasons) < 32:
            reasons.append(_reason(code, location))
    if len(set(header)) != len(header):
        add('duplicate_column', '/header')
    for name in columns:
        if name not in header:
            add('missing_column', _pointer('/header', name))
    for name in header:
        if name not in columns:
            add('unexpected_column', _pointer('/header', name))
    if contract.get('ordered', False) and header != columns:
        add('column_order', '/header')
    count = 0
    for count, row in enumerate(reader, 1):
        if len(row) != len(header):
            add('row_width', '/rows/' + str(count))
        for name, value in zip(header, row):
            kind = contract.get('types', {}).get(name)
            if kind and not _cell(value, kind):
                add('type_mismatch', _pointer('/rows/' + str(count), name))
        if len(reasons) >= 32:
            return
    if 'rows' in contract and count != contract['rows']:
        add('row_count', '/rows')


def validate_artifact(root, contract):
    """Return status, complete byte hash and bounded {code, location} diagnostics.

    Locations use JSON Pointer escaping; CSV data row numbers are one-based.
    Unknown contract/schema features never count as checked. This function only
    reads one artifact and never repairs, rewrites, executes or imports its data.
    """
    digest, reasons = None, []
    try:
        name, kind, limit = _contract(contract)
        data = _read(root, name, limit)
        digest = hashlib.sha256(data).hexdigest()
        if kind != 'file':
            text = data.decode('utf-8')
            if kind == 'json':
                value = json.loads(text, object_pairs_hook=_pairs, parse_constant=_constant, parse_int=_integer)
                _json_bounds(value)
                _validate_json(value, contract['schema'], reasons)
            else:
                _validate_csv(text, contract, reasons)
        return {'status': 'failed' if reasons else 'passed', 'sha256': digest, 'reasons': reasons}
    except _Issue as exc:
        return {'status': exc.status, 'sha256': digest, 'reasons': [_reason(exc.code, exc.location)]}
    except UnicodeError:
        code, status = 'invalid_utf8', 'failed'
    except (json.JSONDecodeError, ValueError):
        code, status = 'invalid_json', 'failed'
    except csv.Error as exc:
        code, status = ('csv_field_limit', 'unverified') if 'field larger' in str(exc) else ('invalid_csv', 'failed')
    except RecursionError:
        code, status = 'structure_limit', 'unverified'
    return {'status': status, 'sha256': digest, 'reasons': [_reason(code, '')]}
