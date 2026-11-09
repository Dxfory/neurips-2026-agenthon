"""Second-draft agreement (T1_THIN_FEATURES second_draft): whether two runs' deliverables state the same answer.

Two independently written programs that read the task the same way write the same numbers, up to the rounding of
their own arithmetic; this module tells that apart from a different answer. It never decides which answer is right
and never edits a file. Anything it cannot read the same way on both sides counts as a disagreement, so a failure
here only ever keeps the first run's outputs (thin._second_drafts).
"""
import csv
import io
import json
import math
import re

RTOL = 1e-4
ATOL = 1e-6
_NUMBER = re.compile(r'[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?$')
_TOKEN = re.compile(r'[\s,;:|()\[\]{}=]+')
MAX_CELLS = 2_000_000  # beyond this a file is compared byte for byte only


def _close(a, b):
    """Numbers equal within RTOL/ATOL; NaN equals NaN, an infinity only itself."""
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    if math.isinf(a) or math.isinf(b):
        return a == b
    return abs(a - b) <= ATOL + RTOL * max(abs(a), abs(b))


def _number(text):
    """The float a cell or token states, or None (booleans and words are not numbers)."""
    text = text.strip()
    if not text:
        return None
    lowered = text.lower()
    if lowered in ('nan', '-nan', '+nan'):
        return math.nan
    if lowered in ('inf', '+inf', 'infinity', '+infinity'):
        return math.inf
    if lowered in ('-inf', '-infinity'):
        return -math.inf
    if not _NUMBER.match(text):
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _scalar_equal(a, b):
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return _close(float(a), float(b))
    if isinstance(a, str) and isinstance(b, str):
        x, y = _number(a), _number(b)
        if x is not None and y is not None:
            return _close(x, y)
        return a.strip() == b.strip()
    return a == b


def json_equal(a, b, depth=0):
    """Same JSON structure (keys, lengths, types) with numbers equal within the tolerance."""
    if depth > 200:
        return False
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(json_equal(a[k], b[k], depth + 1) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(json_equal(x, y, depth + 1) for x, y in zip(a, b))
    if isinstance(a, (dict, list)) or isinstance(b, (dict, list)):
        return False
    return _scalar_equal(a, b)


def _text(data):
    return data.decode('utf-8-sig')


def _cell_equal(x, y):
    a, b = _number(x), _number(y)
    if a is not None and b is not None:
        return _close(a, b)
    return x.strip() == y.strip()


MAX_BYTES = 64 * 1024 * 1024  # files above this are compared byte for byte only


def csv_equal(a, b, delimiter=','):
    """Same rows and columns, cell by cell: numbers within the tolerance, other text after stripping."""
    if max(len(a), len(b)) > MAX_BYTES:
        return False
    rows_a = list(csv.reader(io.StringIO(_text(a)), delimiter=delimiter))
    rows_b = list(csv.reader(io.StringIO(_text(b)), delimiter=delimiter))
    rows_a = [row for row in rows_a if any(cell.strip() for cell in row)]
    rows_b = [row for row in rows_b if any(cell.strip() for cell in row)]
    if len(rows_a) != len(rows_b):
        return False
    if sum(len(row) for row in rows_a) > MAX_CELLS:
        return False
    return all(len(x) == len(y) and all(_cell_equal(p, q) for p, q in zip(x, y)) for x, y in zip(rows_a, rows_b))


def parquet_equal(a, b):
    """Same columns, row count and values (numbers within the tolerance) for two Parquet files."""
    import numpy as np
    import pandas as pd
    left, right = pd.read_parquet(io.BytesIO(a)), pd.read_parquet(io.BytesIO(b))
    if list(left.columns) != list(right.columns) or left.shape != right.shape:
        return False
    if left.size > MAX_CELLS:
        return False
    for column in left.columns:
        x, y = left[column], right[column]
        if pd.api.types.is_bool_dtype(x) or pd.api.types.is_bool_dtype(y):
            if not (pd.api.types.is_bool_dtype(x) and pd.api.types.is_bool_dtype(y)):
                return False
            if not x.astype(object).equals(y.astype(object)):
                return False
        elif pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
            p, q = x.to_numpy(dtype=float), y.to_numpy(dtype=float)
            if not np.allclose(p, q, rtol=RTOL, atol=ATOL, equal_nan=True):
                return False
        else:
            p, q = x.astype(str).str.strip().tolist(), y.astype(str).str.strip().tolist()
            if p != q:
                return False
    return True


def text_equal(a, b):
    """Plain text: the same tokens, numbers within the tolerance."""
    left = [t for t in _TOKEN.split(_text(a)) if t]
    right = [t for t in _TOKEN.split(_text(b)) if t]
    return len(left) == len(right) and all(_cell_equal(x, y) for x, y in zip(left, right))


def file_equal(name, a, b):
    """Whether one deliverable states the same thing in both runs; unreadable on either side is unequal."""
    if a == b:
        return True
    suffix = name.rsplit('.', 1)[-1].lower() if '.' in name else ''
    try:
        if suffix == 'json':
            return json_equal(json.loads(_text(a)), json.loads(_text(b)))
        if suffix in ('csv', 'tsv'):
            return csv_equal(a, b, '\t' if suffix == 'tsv' else ',')
        if suffix == 'parquet':
            return parquet_equal(a, b)
        if suffix in ('txt', 'md', 'log', ''):
            return text_equal(a, b)
    except Exception:
        return False
    return False


def outputs_agree(first, second, names=()):
    """(agree, reason) for two {name: bytes} output snapshots: every named deliverable (every file of either run when
    none is named) is present in both and states the same thing (file_equal)."""
    names = tuple(names) or tuple(sorted(set(first) | set(second)))
    if not names:
        return False, 'no_outputs'
    for name in names:
        if name not in first or name not in second:
            return False, 'missing:' + name
        if not file_equal(name, first[name], second[name]):
            return False, 'differs:' + name
    return True, 'agree'
