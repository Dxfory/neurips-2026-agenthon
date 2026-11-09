"""Data summaries, one-generation baseline, and conservative checker interpretation."""
from __future__ import annotations

import ast
import bisect
import builtins
import codecs
import csv
import difflib
import hashlib
import importlib.util
import io
import json
import math
import os
import re
import shlex
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import textwrap
import time
import tokenize
import tomllib
import unicodedata
from collections import Counter
from pathlib import Path, PurePosixPath

from .request_budget import parse_request_cap

from .continuation import (LINT_LIMIT, _arity, _bound_in, _compiles, _lintable, absorb, absorb_defs, anchor_hunk, anchors, api_lint,
                           backend_fixed, bind_missing, call_lint as _direct_call_lint, code_region, completion_binding_conflicts, ensure_guard,
                           first_invalid, function_tail_context, guard_raise, has_name, is_main_guard, keep_complete_defs, lift_free_names, merge,
                           missing_calls, new_global_reads, new_lint, numpy_names_fixed, output_names, own_raise, raise_stubs,
                           repair_target, reply_metrics, s12_lint, splice_hunk, statement_count, string_values, stub_broken_def,
                           stub_defs, ufunc_lint, unverified_global_reads, verified_prefix)
from .audit import (CONTRACT_COUNTERS, _quoted, bound_flags, conflicting_definitions, contract_flags,
                    contract_literal_rules, contract_rules, crosskey_flags, degenerate_flags, digest, format_entry,
                    incomplete_mappings, literal_flags, literal_rules, nonfinite_flags, normalize_contract,
                    normalize_outputs, output_contract, review_parse, same_quote, sanity_flags, selfval_flags,
                    worked_examples)
from . import date_validation, families, minimal, resource_control, runtime, solver_profile
from .house import (HouseClient, HouseConfig, HouseError, eval_throttle, extract_python, safe_transport_detail,
                    strip_reasoning)
from .inputs import (JSON_READ_TOTAL, SHEET_COLUMNS, archive_listing, archive_texts, cut, data_traps, html_outline,
                     json_preview, workbook_preview, xml_outline)
from .staging import _canary_values, canary_banner

FORBIDDEN = {'checks','reference','reference_data','oracle','solution','dev','__pycache__'}
_GUID = re.compile(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', re.I)
_CANARY_MARKER = re.compile(
    rb'canary[\s_-]*guid\W{0,5}([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})', re.I)


def protect_controller():
    """Prevent same-UID children from reading controller /proc environment/memory."""
    import ctypes
    if sys.platform!='linux': raise RuntimeError('Controller protection requires Linux')
    libc=ctypes.CDLL(None,use_errno=True)
    if libc.prctl(4,0,0,0,0)!=0 or libc.prctl(3,0,0,0,0)!=0:
        raise RuntimeError('Controller protection could not be established')


def _write_audit(path: Path, text: str):
    # The generated program may have pre-created a symlink with this reserved name.
    flags=os.O_WRONLY|os.O_CREAT|os.O_NOFOLLOW|os.O_NONBLOCK
    fd=os.open(path,flags,0o644)
    with os.fdopen(fd,'w') as stream:
        info=os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_nlink!=1:
            raise ValueError('Reserved audit path must be a plain file')
        stream.truncate(0)
        stream.write(text)


# Third-party exceptions whose class name (never the module path or message) may be exported.
THIRD_PARTY_ERRORS = frozenset({
    'pandas.errors.ParserError', 'pandas.errors.EmptyDataError', 'pandas.errors.IntCastingNaNError',
    'pandas.errors.InvalidIndexError', 'pandas.errors.MergeError', 'pandas.errors.OutOfBoundsDatetime',
    'pandas._libs.tslibs.np_datetime.OutOfBoundsDatetime', 'numpy.linalg.LinAlgError',
    'numpy.exceptions.AxisError', 'numpy._core._exceptions._UFuncNoLoopError', 'scipy.linalg._misc.LinAlgError',
    'pyarrow.lib.ArrowInvalid', 'pyarrow.lib.ArrowTypeError', 'pyarrow.lib.ArrowNotImplementedError',
    'json.decoder.JSONDecodeError', 'socket.gaierror', 'sklearn.exceptions.NotFittedError',
    'statsmodels.tools.sm_exceptions.MissingDataError'})
SIGNATURES = ('fnf_app_path', 'fnf_input_path', 'fnf_other', 'json_not_serializable', 'removed_numpy_attr',
              'removed_pandas_method', 'removed_scipy_attr', 'to_datetime_format', 'mixed_dtype_reduction',
              'keyerror', 'nan_input', 'module_not_found', 'readonly_fs', 'network_blocked', 'other')
_SIGNATURE_PATTERNS = (  # (signature, pattern on the exception message), first match wins
    ('network_blocked', r'network disabled|Name or service not known|Temporary failure in name resolution'
                        r'|Network is unreachable'),
    ('readonly_fs', r'Read-only file system'),
    ('json_not_serializable', r'is not JSON serializable'),
    ('removed_numpy_attr', r"module 'numpy[\w.]*' has no attribute|cannot import name '\w+' from 'numpy"),
    ('removed_scipy_attr', r"module 'scipy[\w.]*' has no attribute|cannot import name '\w+' from 'scipy"),
    ('removed_pandas_method', r"'(?:DataFrame|Series|Index|DataFrameGroupBy|SeriesGroupBy)' object has no attribute "
                              r"'(?:append|iteritems|mad|lookup|ix|as_matrix|get_value|set_value|swapaxes|"
                              r"is_monotonic|backfill|pad)'|module 'pandas' has no attribute"),
    ('to_datetime_format', r"doesn't match format|does not match format|unconverted data remains"),
    ('nan_input', r'contains? NaN|infs or NaNs|float NaN to integer|non-finite values \(NA or inf\)'),
    ('mixed_dtype_reduction', r"agg function failed|could not convert .{0,300} to numeric|"
                              r"can only concatenate str \(not \"(?:int|float)\"\) to str|"
                              r"unsupported operand type\(s\) for .{1,3}: '(?:str|int|float)' and '(?:str|int|float)'|"
                              r"did not contain a loop with signature matching types"))


def failure_signature(name: str, message: str, input_root=None) -> str:
    """The failure class of one exception line, computed in the container: an enum, never the message."""
    short = name.rsplit('.', 1)[-1]
    missing = _MISSING_FILE.search(message)
    if short == 'gaierror':
        return 'network_blocked'
    for signature, pattern in _SIGNATURE_PATTERNS[:2]:
        if re.search(pattern, message):
            return signature
    if short == 'FileNotFoundError' or missing:
        path = missing[1] if missing else ''
        root = str(input_root).rstrip('/') + '/' if input_root else None
        if path.startswith('/app/') and not path.startswith('/app/output'):
            return 'fnf_app_path'
        return 'fnf_input_path' if root and (path + '/').startswith(root) else 'fnf_other'
    if short == 'ModuleNotFoundError' or message.startswith('No module named'):
        return 'module_not_found'
    for signature, pattern in _SIGNATURE_PATTERNS[2:]:
        if re.search(pattern, message, re.I if signature == 'mixed_dtype_reduction' else 0):
            return signature
    return 'keyerror' if short == 'KeyError' else 'other'


def execution_failure_summary(path: Path, script: Path, input_root=None) -> dict:
    """Read at most 64 KiB; export only numeric metadata, allowlisted exception class names and a signature enum.

    Worker-written stderr is untrusted diagnostic evidence, never a verdict.
    Messages, source lines, paths, custom exception names and raw tails are not saved.
    """
    summary = {'source': 'untrusted_execution_log', 'log_readable': False,
               'exception_type': None, 'generated_line': None, 'signature': None}
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, 'rb') as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                return summary
            stream.seek(max(0, info.st_size - 65536))
            tail = stream.read(65536)
        summary.update(log_readable=True, log_bytes=info.st_size,
                       tail_bytes_read=len(tail), tail_truncated=info.st_size > 65536)
    except (OSError, ValueError):
        return summary
    frame = re.compile(r'  File "' + re.escape(str(script)) + r'", line ([0-9]{1,7})(?:, in .*)?')
    traceback, last = False, None
    for line in tail.decode('utf-8', errors='replace').splitlines():
        match = frame.fullmatch(line)
        if match and 0 < int(match[1]) <= 1_000_000:
            summary['generated_line'] = int(match[1])
        traceback = traceback or line.startswith('Traceback')
        match = re.fullmatch(r'([A-Za-z_][A-Za-z0-9_.]{0,127})(?:: ?(.*))?', line)
        if match and (match[1] in _BUILTIN_ERRORS or match[1] in THIRD_PARTY_ERRORS):
            last = match[1], match[2] or ''
            summary['exception_type'] = match[1].rsplit('.', 1)[-1].lstrip('_')
    if last is not None or traceback:
        summary['signature'] = failure_signature(*last, input_root) if last else 'other'
    return summary


PROFILE_FILES = 40
PROFILE_CAP = 30_000
PROFILE_HEAD = 2000  # characters shown of a data-like text file (.json, .xml, .html, .jsonl, a table in .txt)
PROFILE_TEXT = 12_000  # characters shown of a document (.md, .txt prose), such as the formulas a task refers to
PROFILE_CODE = 20_000  # characters shown of a source file the TASK names (a template it says to fix, say)
_DOCUMENTS = ('.md', '.txt')  # a .txt only when it reads as prose: .txt also holds whitespace tables of numbers
_SOURCES = ('.py',)  # shown only when the TASK names the file: provenance scripts beside the data stay unread
_OUTLINED = {'.xml': xml_outline, '.html': html_outline, '.htm': html_outline, '.xhtml': html_outline}
_WORKBOOKS = ('.xlsx', '.xlsm')
_NUMBER = re.compile(r'(?<![\w.])[-+]?\d[\d.,:/]*(?:[eE][-+]?\d+)?')
_WORD = re.compile(r'[A-Za-z]{2,}')
_NOT_PROFILED = {'dockerfile', 'manifest.json', 'staging_manifest.json'}  # organizer metadata, as in staging


def _names(text: str, name: str) -> bool:
    """Whether text mentions a file name as a whole word (data.py is not a mention of a.py)."""
    return bool(name) and re.search(r'(?<![\w.-])' + re.escape(name) + r'(?![\w-])', text) is not None


def _reads_as_prose(text: str) -> bool:
    """Whether text reads as a document (its words outnumber its numbers) rather than a table of numbers."""
    return len(_WORD.findall(text)) > len(_NUMBER.findall(text))


PROFILE_SNIFF = 8 * 1024  # characters of a table's head that tell its delimiter (csv.Sniffer is quadratic at worst)
_TABLE_DELIMITERS = ',;\t|'
_CELL_VALUE = re.compile(r'\s*(?:[-+]?(?:\d+(?:[.,]\d+)*|[.,]\d+)(?:[eE][-+]?\d+)?%?'
                         r'|\d{4}-\d{2}-\d{2}(?:[ T][0-9:.]+)?)\s*')  # a number (either decimal mark) or a date
PROFILE_UTF8_BYTES = 64 * 1024 * 1024  # bytes of one text input checked as UTF-8 (a larger file: its head)
PROFILE_UTF8_TOTAL = 256 * 1024 * 1024  # bytes checked as UTF-8 over all files of a unit
PROFILE_CHARSET_HEAD = 4096  # bytes of an HTML document searched for its meta charset when it has no </head> or <body>
_TEXT_INPUTS = ('.csv', '.tsv', '.json', '.jsonl', '.txt', '.md', '.xml', '.html', '.htm', '.xhtml')
_MARKUP_INPUTS = ('.html', '.htm', '.xhtml')
_XML_ENCODING = re.compile(rb'(?:\xef\xbb\xbf)?\s*<\?xml\b[^>]{0,200}?\bencoding\s*=\s*["\']([A-Za-z0-9_.:-]{1,40})["\']',
                           re.I)
_META_CHARSET = re.compile(rb'<meta\b[^>]{0,500}?\bcharset\s*=\s*["\']?([A-Za-z0-9_.:-]{1,40})', re.I)
_HTML_HEAD_END = re.compile(rb'</head\b|<body\b', re.I)
_BYTE_ORDER_MARKS = ((b'\xff\xfe\x00\x00', 'UTF-32-LE'), (b'\x00\x00\xfe\xff', 'UTF-32-BE'),
                     (b'\xff\xfe', 'UTF-16-LE'), (b'\xfe\xff', 'UTF-16-BE'))


def table_delimiter(path: Path) -> str:
    """A CSV or TSV file's field delimiter, read from its head (whole lines): a .tsv whose header has a tab keeps the
    tab. Otherwise the one of , ; tab | that splits every line of the head into the same number (2 or more) of fields;
    when several do, the one whose cells read as numbers or dates most often (a decimal comma is not a delimiter),
    then the suffix's (a comma; a tab for .tsv). When none does, csv.Sniffer's choice where it splits the header
    into more fields than the suffix's. Any sniffing error keeps the suffix's."""
    tsv = path.suffix.lower() == '.tsv'
    default = '\t' if tsv else ','
    with path.open(newline='', encoding='utf-8-sig', errors='replace') as stream:
        sample = stream.read(PROFILE_SNIFF)
    lines = sample.splitlines(keepends=True)
    if not lines:
        return default
    if len(sample) >= PROFILE_SNIFF and len(lines) > 1:
        lines = lines[:-1]  # whole lines only
        sample = ''.join(lines)
    if tsv and len(next(csv.reader(lines[:1], delimiter='\t'), [])) > 1:
        return default
    try:
        found = csv.Sniffer().sniff(sample, delimiters=_TABLE_DELIMITERS).delimiter
        records = {d: [record for record in csv.reader(lines, delimiter=d) if record] for d in _TABLE_DELIMITERS}
    except Exception:  # csv.Error when no delimiter reads consistently
        return default
    counts = {d: {len(record) for record in rows} for d, rows in records.items()}
    fits = [d for d in _TABLE_DELIMITERS if len(counts[d]) == 1 and min(counts[d]) >= 2]
    if not fits:
        header = {d: len(rows[0]) if rows else 0 for d, rows in records.items()}
        return found if found in _TABLE_DELIMITERS and header[found] > header[default] else default
    values = {d: sum(bool(_CELL_VALUE.fullmatch(cell)) for record in records[d][1:] for cell in record) for d in fits}
    return max(fits, key=lambda d: (values[d], d == default, d == found))


def _text_codec(label: str) -> str | None:
    """Python's name for a declared text encoding, or None when Python does not know it as one."""
    try:
        info = codecs.lookup(label)
    except LookupError:
        return None
    return info.name if getattr(info, '_is_text_encoding', True) else None


def _decodes(data: bytes, codec: str, whole: bool) -> bool:
    try:
        codecs.getincrementaldecoder(codec)().decode(data, final=whole)
    except (UnicodeError, LookupError, ValueError, TypeError):
        return False
    return True


def _declared_charset(data: bytes, suffix: str) -> str | None:
    """The charset a markup file declares in its own head: an XML declaration at its very start, or an HTML meta
    charset before </head> (or <body>; within its first PROFILE_CHARSET_HEAD bytes when it has neither). A charset
    named in the body, a link or a table cell is not a declaration."""
    if suffix == '.xml' or suffix in _MARKUP_INPUTS:
        prolog = _XML_ENCODING.match(data)
        if prolog:
            return prolog.group(1).decode('ascii')
    if suffix in _MARKUP_INPUTS:
        end = _HTML_HEAD_END.search(data)
        meta = _META_CHARSET.search(data, 0, end.start() if end else PROFILE_CHARSET_HEAD)
        if meta:
            return meta.group(1).decode('ascii')
    return None


def utf8_note(path: Path, budget: int):
    """(note or None, bytes read): a text input that is not valid UTF-8 within its first budget bytes, named with
    the first byte that fails and how to open it: with the charset the file declares (an XML encoding, an HTML meta
    charset) or its byte-order mark only when that encoding decodes the bytes read; otherwise with an explicit
    encoding that does (cp1252, else latin-1) or errors='replace'. A file that declares UTF-8 is said not to be."""
    suffix = path.suffix.lower()
    if suffix not in _TEXT_INPUTS or budget <= 0:
        return None, 0
    with path.open('rb') as stream:
        data = stream.read(budget)
        whole = not stream.read(1)
    try:
        data.decode('utf-8')
        return None, len(data)
    except UnicodeDecodeError as exc:
        if not whole and exc.reason == 'unexpected end of data':
            return None, len(data)  # the read stopped inside a character
        start = exc.start
    for mark, name in _BYTE_ORDER_MARKS:
        if data.startswith(mark) and _decodes(data, name, whole):
            return f'not UTF-8: {name} (byte-order mark); read it with that encoding', len(data)
    where = f'not UTF-8 (byte 0x{data[start]:02x} at offset {start})'
    fallback = 'cp1252' if _decodes(data, 'cp1252', whole) else 'latin-1'
    instead = (f"read it with encoding='{fallback}' (which decodes {'it' if whole else 'the bytes checked'}) "
               "or errors='replace'")
    charset = _declared_charset(data, suffix)
    if charset is None:
        if data.startswith(codecs.BOM_UTF8):
            return f'{where}; it starts with a UTF-8 byte-order mark but is not valid UTF-8: {instead}', len(data)
        return f'{where}; no charset declared: {instead}', len(data)
    codec = _text_codec(charset)
    if codec is None:
        return f'{where}; it declares charset {charset}, which Python does not know: {instead}', len(data)
    if _decodes(data, codec, whole):
        return f'{where}; it declares charset {charset}: read it with that encoding', len(data)
    if codec.startswith('utf-8'):
        return f'{where}; it declares charset {charset} but is not valid UTF-8: {instead}', len(data)
    return f'{where}; it declares charset {charset}, which does not decode it: {instead}', len(data)


def _empty_header(columns: list) -> str | None:
    """A note on the empty cells of a table's header row: pandas.read_csv names each 'Unnamed: <position>'."""
    empty = [index for index, name in enumerate(columns) if name == '']
    if not empty:
        return None
    named = ', '.join(f"column {index} -> 'Unnamed: {index}'" for index in empty[:8])
    more = f' and {len(empty) - 8} more' if len(empty) > 8 else ''
    lead = '; an empty first cell often marks a saved index (index_col=0 reads it as the index)' if empty[0] == 0 else ''
    return f'empty header cells; pandas.read_csv names them: {named}{more}{lead}'


def _profile_file(path: Path) -> dict:
    """One input file's plain row: columns and sample rows of a table (split at its sniffed delimiter, named when
    not the suffix's; empty header cells noted), the head of a text file, a parquet file's schema and first rows."""
    row, suffix = {}, path.suffix.lower()
    if suffix in ('.csv','.tsv'):
        delimiter = table_delimiter(path)
        with path.open(newline='', encoding='utf-8-sig', errors='replace') as f:
            reader=csv.reader(f, delimiter=delimiter)
            row['columns']=next(reader, [])
            row['sample']=[record for _,record in zip(range(3),reader)]
        if delimiter != ('\t' if suffix=='.tsv' else ','):
            row['delimiter']=delimiter
        empty=_empty_header(row['columns'])
        if empty:
            row['empty_header']=empty
    elif suffix == '.json':
        with path.open(encoding='utf-8', errors='replace') as f:
            sample = f.read(PROFILE_HEAD + 40)
        row['head'] = cut(sample, PROFILE_HEAD)
        if len(sample) > len(row['head']):
            row['_json_truncated'] = True  # consumed before row budgeting or prompt serialization
    elif suffix in ('.txt','.md','.xml','.html','.jsonl'):
        text=_head(path, PROFILE_TEXT if suffix in _DOCUMENTS else PROFILE_HEAD)
        row['head']=text if suffix!='.txt' or _reads_as_prose(text) else cut(text, PROFILE_HEAD)
    elif suffix in ('.parquet','.pqt'):
        try:
            import pyarrow.parquet as pq
        except ImportError:
            return {'profile_status': 'parquet_reader_unavailable'}
        pf=pq.ParquetFile(path)
        row['rows']=pf.metadata.num_rows
        row['schema']=str(pf.schema_arrow)
        batch=next(pf.iter_batches(batch_size=3), None)
        row['sample']=cut(str(batch.to_pydict()), 2000) if batch is not None else '{}'
    return row


# T1_S11_RAISE: a table's date columns with their min and max, so a date the TASK or an input names that lies outside
# the data shows in the prompt (an order day after the last bar). Pure Python, bounded in bytes per file and per unit.
PROFILE_RANGE_ROWS = 50  # leading rows that tell a date column
PROFILE_RANGE_SHARE = 0.8  # of their values that are not blank, the share that must read as dates
PROFILE_RANGE_BYTES = 64 * 1024 * 1024  # a larger table is read at its first and last half of this (time-ordered data)
PROFILE_RANGE_TOTAL = 192 * 1024 * 1024  # bytes read for ranges over all files of a unit
PROFILE_RANGE_MIN = 2 * 1024 * 1024  # a table is left without ranges once less than this is left of the total
PROFILE_RANGE_NAME = 80  # characters of a column name shown
_STAMP = re.compile(r'\d{4}([-/])\d{2}\1\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d{1,9})?)?)?(?:Z|[+-]\d{2}:?\d{2})?')
_DMY = re.compile(r'(\d{2})([/.-])(\d{2})\2(\d{4})((?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d{1,9})?)?)?)')  # 04/02/2026
_TIME_WORDS = {'date', 'time', 'timestamp', 'datetime', 'stamp', 'ts', 'dt', 'epoch'}  # an integer date column's name
_EPOCH = {10: 1, 13: 10 ** 3, 16: 10 ** 6, 19: 10 ** 9}  # digits of epoch seconds, ms, us and ns since 1970
_YMD = re.compile(r'(?:19|20)\d\d(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])')  # 20200302 in a column named date


def _column_words(name):
    """A column name's words, lower case: open_time, openTime and OPEN TIME are open and time (update_id has none)."""
    return set(re.split(r'[^a-z0-9]+', re.sub(r'(?<=[a-z0-9])(?=[A-Z])', '_', name).lower()))


def _date_key(kind, value):
    """A value's sort key as a date of kind, or None when it does not read so: ISO text and integers of one length
    order as text; a day-first or month-first date orders by year, month, day and time."""
    if kind == 'stamp':
        return value if _STAMP.fullmatch(value) else None
    if kind == 'ymd':
        return value if _YMD.fullmatch(value) else None
    if kind in ('mdy', 'dmy'):
        match = _DMY.fullmatch(value)
        if match is None:
            return None
        month, day = (match[1], match[3]) if kind == 'mdy' else (match[3], match[1])
        return match[4] + month + day + match[5]
    return value if value.isdigit() and len(value) == int(kind[5:]) else None


def _date_kind(name, values):
    """How a column's leading values read as dates: 'stamp' (ISO), 'mdy' or 'dmy' (04/02/2026; day first only when a
    first field passes 12), 'ymd' (YYYYMMDD in a date column) or 'epoch<digits>' (an integer time column of epoch
    seconds to nanoseconds); None unless at least PROFILE_RANGE_SHARE of the values that are not blank read so (a
    feed with a few bad rows still counts). A header cell that itself reads as YYYYMMDD names a date column (a file
    without a header row)."""
    values = [value.strip() for value in values if value.strip()]
    if not values:
        return None
    need = math.ceil(PROFILE_RANGE_SHARE * len(values))
    if sum(bool(_STAMP.fullmatch(value)) for value in values) >= need:
        return 'stamp'
    found = [match for match in map(_DMY.fullmatch, values) if match]
    if len(found) >= need:
        return 'dmy' if any(int(match[1]) > 12 for match in found) else 'mdy'
    words = _column_words(name)
    dated = 'date' in words or 'datetime' in words or bool(_YMD.fullmatch(name.strip()))
    if dated and sum(bool(_YMD.fullmatch(value)) for value in values) >= need:
        return 'ymd'
    if not words & _TIME_WORDS:
        return None
    digits, count = (Counter(len(value) for value in values if value.isdigit()).most_common(1) or [(0, 0)])[0]
    return f'epoch{digits}' if digits in _EPOCH and count >= need else None


def _date_text(kind, value):
    """A column's min or max as shown: the value, and for epoch integers the UTC time it stands for."""
    if not kind.startswith('epoch'):
        return value
    seconds = int(value) / _EPOCH[int(kind[5:])]
    try:
        stamp = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(seconds))
    except (OverflowError, OSError, ValueError):
        return value
    return f'{value} ({stamp})'


def _table_chunks(path: Path, size: int, budget: int):
    """(texts to scan, whether the header row leads the first, read in part): the whole file within budget bytes,
    else its first and last budget // 2 bytes, each cut at a line end."""
    with path.open('rb') as stream:
        if size <= budget:
            return [stream.read(budget).decode('utf-8-sig', 'replace')], False
        half = budget // 2
        head = stream.read(half)
        stream.seek(max(0, size - half))
        tail = stream.read(half)
    return [head[:head.rfind(b'\n') + 1].decode('utf-8-sig', 'replace'),
            tail[tail.find(b'\n') + 1:].decode('utf-8', 'replace')], True


def _csv_ranges(path: Path, budget: int):
    """({column: 'min .. max'} of a CSV or TSV's date columns, bytes read, read in part); ({}, 0, False) without
    one. Values that do not read as the column's kind are skipped; a header cell that reads so counts as a value."""
    delimiter = table_delimiter(path)
    with path.open(newline='', encoding='utf-8-sig', errors='replace') as stream:
        reader = csv.reader(stream, delimiter=delimiter)
        header = next(reader, [])
        leading = [record for _, record in zip(range(PROFILE_RANGE_ROWS), reader)]
    kinds = {}
    for index, name in enumerate(header):
        kind = _date_kind(name, [record[index] for record in leading if index < len(record)])
        if kind:
            kinds[index] = kind
    if not kinds:
        return {}, 0, False
    size = path.stat().st_size
    texts, partial = _table_chunks(path, size, budget)
    low, high = {}, {}

    def add(index, kind, value):
        key = _date_key(kind, value.strip())
        if key is None:
            return
        if index not in low or key < low[index][0]:
            low[index] = (key, value.strip())
        if index not in high or key > high[index][0]:
            high[index] = (key, value.strip())
    for index, kind in kinds.items():
        add(index, kind, header[index])
    for number, text in enumerate(texts):
        rows = csv.reader(io.StringIO(text, newline=''), delimiter=delimiter)
        if number == 0:
            next(rows, None)
        for record in rows:
            for index, kind in kinds.items():
                if index < len(record):
                    add(index, kind, record[index])
    ranges = {header[index][:PROFILE_RANGE_NAME] or f'column {index + 1}':
              f'{_date_text(kinds[index], low[index][1])} .. {_date_text(kinds[index], high[index][1])}'
              for index in kinds if index in low}
    return ranges, min(size, budget), partial


def _parquet_ranges(path: Path, budget: int):
    """_csv_ranges for a parquet file within budget bytes: its timestamp and date columns, and its text columns whose
    first values read as ISO dates."""
    size = path.stat().st_size
    if size > budget:
        return {}, 0, False
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    batch = next(pf.iter_batches(batch_size=PROFILE_RANGE_ROWS), None)
    names = []
    for field in pf.schema_arrow:
        if pa.types.is_timestamp(field.type) or pa.types.is_date(field.type):
            names.append(field.name)
        elif (pa.types.is_string(field.type) or pa.types.is_large_string(field.type)) and batch is not None:
            values = [value for value in batch.column(field.name).to_pylist() if isinstance(value, str)]
            if _date_kind(field.name, values) == 'stamp':
                names.append(field.name)
    if not names:
        return {}, 0, False
    table = pf.read(columns=names)
    ranges = {}
    for name in names:
        found = pc.min_max(table.column(name)).as_py()
        if found.get('min') is not None and found.get('max') is not None:
            ranges[str(name)[:PROFILE_RANGE_NAME]] = f'{found["min"]} .. {found["max"]}'
    return ranges, size, False


def date_ranges(path: Path, budget: int):
    """(fields for a table's profile row, bytes read) under T1_S11_RAISE: date_ranges {column: 'min .. max'} of its
    date columns (_csv_ranges, _parquet_ranges) and, when only the head and tail were read, date_ranges_read; ({}, 0)
    for another file or a table without one. Raises on an unreadable file."""
    suffix = path.suffix.lower()
    if budget < PROFILE_RANGE_MIN or suffix not in ('.csv', '.tsv', '.parquet', '.pqt'):
        return {}, 0
    ranges, used, partial = (_csv_ranges if suffix in ('.csv', '.tsv') else _parquet_ranges)(path, budget)
    if not ranges:
        return {}, used
    fields = {'date_ranges': ranges}
    if partial:
        fields['date_ranges_read'] = f'first and last {budget // 2 // 1024 // 1024} MB of the file'
    return fields, used


def _head(path: Path, limit: int) -> str:
    """A text file's first limit characters (read a little past them, so a cut never splits a GUID)."""
    with path.open(encoding='utf-8', errors='replace') as f:
        return cut(f.read(limit + 40), limit)


def _previews(path: Path, row: dict, named: bool, wide: bool = False) -> list:
    """What a file's plain row may grow into, largest first, each a dict merged into the row: a source file the
    TASK names in full (or its head); a workbook's sheets and first rows (wide: inputs._sheet's, input_fidelity);
    an archive's members; the element outline of XML or HTML (with the head of an HTML file the plain row lacks).
    Raises on an unreadable file."""
    suffix = path.suffix.lower()
    if suffix in _SOURCES:
        if not named:
            return []
        text = _head(path, PROFILE_CODE)
        return [{'head': text}, {'head': cut(text, PROFILE_HEAD)}] if len(text) > PROFILE_HEAD else [{'head': text}]
    if suffix in _WORKBOOKS:
        return [workbook_preview(path, wide=True)] if wide else [workbook_preview(path)]
    if suffix == '.zip':
        return [archive_listing(path)]
    if suffix in _OUTLINED:
        outline = {'outline': _OUTLINED[suffix](path)}
        if 'head' in row:
            return [outline]
        head = {'head': _head(path, PROFILE_HEAD)}
        return [dict(head, **outline), head]
    return []


def _grow(files: list, index: int, options: list, used: int) -> int:
    """Merge the largest option that fits the room left under PROFILE_CAP into files[index]; the characters it
    took (0 when none fits)."""
    size = len(json.dumps(files[index], ensure_ascii=False))
    for option in options:
        grown = dict(files[index], **option)
        extra = len(json.dumps(grown, ensure_ascii=False)) - size
        if used + extra <= PROFILE_CAP:
            files[index] = grown
            return max(extra, 0)
    return 0


def input_profile(task_dir: Path, stats: dict | None = None, named: str = '', ranges: bool = False,
                  fidelity: bool = False) -> dict:
    """Best effort, never raises: each file is profiled on its own (a failure marks that row), symlinks are
    skipped, files beyond PROFILE_FILES are only counted per extension, and rows past PROFILE_CAP characters
    keep their path and size. Every file first gets its plain row (_profile_file; a document's text at its
    PROFILE_HEAD head), budgeted in path order. Only then, in the room still left under PROFILE_CAP, a row grows
    (_previews; never displacing another row): first the files the TASK (named, its text) names, then the
    previews of workbooks, archives and XML or HTML, then documents (.md, a .txt that reads as prose) shown whole
    up to PROFILE_TEXT characters; each in path order, each taking the largest form that fits. A source file
    (.py) is shown only when the TASK names it, up to PROFILE_CODE characters. stats receives the counts
    (numbers only; previews counts the rows a preview grew, not the documents shown whole). With ranges
    (T1_S11_RAISE) a table's plain row also gets its date columns' min and max (date_ranges, within
    PROFILE_RANGE_TOTAL bytes read over the unit; a file that fails keeps its row without them), and stats
    date_ranges counts those columns. A text input that is not valid UTF-8 gets an encoding note (utf8_note, within
    PROFILE_UTF8_TOTAL bytes read over the unit), kept when its row is cut to its path. With fidelity
    (input_fidelity) a workbook preview scans wide rows (inputs._sheet) and the profile opens with input_traps, the
    data-trap lines of the .csv/.tsv inputs read whole (inputs.data_traps); stats traps and wide_rows count them."""
    csv.field_size_limit(1 << 26)
    stats = {} if stats is None else stats
    files, rest, used, whole, paths_of, tables = [], Counter(), 0, {}, {}, []
    json_left, utf8_left = JSON_READ_TOTAL, PROFILE_UTF8_TOTAL
    stats.update(files=0, errors=0, unlisted=0, symlinks=0, capped=0, previews=0)
    left = PROFILE_RANGE_TOTAL
    if ranges:
        stats['date_ranges'] = 0
    try:
        paths = sorted(task_dir.rglob('*'))
    except Exception:
        paths, stats['errors'] = [], 1
    for path in paths:
        try:
            rel = path.relative_to(task_dir)
            if any(p.lower() in FORBIDDEN or p.lower().startswith(('expected','oracle_','answer_key'))
                   or p.lower() in _NOT_PROFILED for p in rel.parts):
                continue
            if path.is_symlink():
                stats['symlinks'] += 1
                continue
            if not path.is_file() or path.name in ('instruction.md','card.toml'):
                continue
            if len(files) >= PROFILE_FILES:
                rest[path.suffix.lower() or '(none)'] += 1
                continue
            row = {'path': str(rel), 'bytes': path.stat().st_size}
            if fidelity and path.suffix.lower() in ('.csv', '.tsv'):
                tables.append((str(rel), path))
            try:
                row.update(_profile_file(path))
            except Exception:
                row['profile_status'] = 'error'
            if ranges and row.get('profile_status') is None:
                try:
                    fields, used_bytes = date_ranges(path, min(PROFILE_RANGE_BYTES, left))
                except Exception:  # the ranges only help: the row stands without them
                    fields, used_bytes = {}, PROFILE_RANGE_MIN
                left -= used_bytes
                row.update(fields)
            if row.pop('_json_truncated', False):
                fields, read = json_preview(path, row['head'], json_left)
                json_left -= read
                row.update(fields)  # never larger than this file's original serialized head row
            if row.get('profile_status') is None:
                try:
                    note, read = utf8_note(path, min(PROFILE_UTF8_BYTES, utf8_left))
                except OSError:
                    note, read = None, 0
                utf8_left -= read
                if note:
                    row['encoding'] = note
            text = row.get('head', '')
            if len(text) > PROFILE_HEAD:
                row['head'] = cut(text, PROFILE_HEAD)
            size = len(json.dumps(row, ensure_ascii=False))
            if used + size > PROFILE_CAP:
                row = {'path': row['path'], 'bytes': row['bytes'], 'profile_status': 'size_cap',
                       **({'encoding': row['encoding']} if 'encoding' in row else {})}
                size = len(json.dumps(row, ensure_ascii=False))
            else:
                if len(text) > PROFILE_HEAD:
                    whole[len(files)] = text
                if row.get('profile_status') is None:
                    paths_of[len(files)] = path
            used += size
            files.append(row)
            stats['errors'] += row.get('profile_status') == 'error'
            stats['capped'] += row.get('profile_status') == 'size_cap'
            if ranges:
                stats['date_ranges'] += len(row.get('date_ranges', ()))
        except Exception:
            stats['errors'] += 1
    mentioned = {index for index in paths_of if _names(named, paths_of[index].name)}
    tiers = sorted(paths_of, key=lambda index: (index not in mentioned, index in whole, index))
    for index in tiers:  # the TASK's files, then structural previews, then whole documents; path order within
        if used >= PROFILE_CAP:
            break
        try:
            options = [{'head': whole[index]}] if index in whole else []
            options = options or (_previews(paths_of[index], files[index], index in mentioned, wide=True) if fidelity
                                  else _previews(paths_of[index], files[index], index in mentioned))
            extra = _grow(files, index, options, used) if options else 0
        except Exception:  # an odd or unreadable file keeps its plain row
            continue
        used += extra
        stats['previews'] += bool(extra) and index not in whole  # a document shown whole is no preview
    stats.update(files=len(files), unlisted=sum(rest.values()))
    profile = {'files': files}
    if rest:
        profile['unprofiled_files'] = dict(sorted(rest.items()))
    if fidelity:
        stats['wide_rows'] = sum(len(cells) > SHEET_COLUMNS for row in files for sheet in row.get('sheets', ())
                                 for cells in (sheet.get('rows') or {}).values())
        try:
            traps = data_traps(tables)
        except Exception:  # the traps only help: the profile stands without them
            traps, stats['errors'] = [], stats.get('errors', 0) + 1
        stats['traps'] = len(traps)
        if traps:  # first, so a context cut from the profile's tail (thin.task_context) leaves them
            profile = {'input_traps': traps, **profile}
    return profile


CONTAINER_SCAN_BYTES = 256 * 1024 * 1024  # decompressed bytes the canary scan reads from all containers together
_NEAR_CANARY = re.compile(rb'canary.{0,80}?([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})', re.I | re.S)


def _canary_texts(name: str, data: bytes) -> set[str]:
    """GUIDs one file's text names as canaries: a canary marker, or a string under a JSON/TOML key containing
    'canary'."""
    found = {guid.decode().lower() for guid in _CANARY_MARKER.findall(data)}
    suffix = PurePosixPath(name).suffix.lower()
    if suffix in ('.json', '.toml'):
        try:
            value = json.loads(data) if suffix == '.json' else tomllib.loads(data.decode())
        except Exception:
            return found
        for item in _canary_values(value):
            found.update(guid.lower() for guid in _GUID.findall(item))
    return found


def near_canaries(text) -> set[str]:
    """GUIDs within 80 characters after the word canary: in text a preview renders from cells, members or rows
    (a marker split over two cells, say), any GUID that close to the word is withheld."""
    data = text.encode('utf-8', 'replace') if isinstance(text, str) else text
    return {guid.decode().lower() for guid in _NEAR_CANARY.findall(data)}


_PREVIEW_FIELDS = ('sheets', 'members', 'outline', 'json_structure')  # rendered structure, including JSON keys


def preview_texts(profile: dict) -> list:
    """The parts of a profile a preview renders (a workbook's sheets, an archive's members, an element outline, a
    source file the TASK names): the text near_canaries scans. Plain rows (a table's columns and sample, a text
    file's head) are left out, so a data GUID beside a column named like canary stays data."""
    shown = []
    for row in profile.get('files', []):
        shown += [row[key] for key in _PREVIEW_FIELDS if key in row]
        if PurePosixPath(str(row.get('path', ''))).suffix.lower() in _SOURCES and 'head' in row:
            shown.append(row['head'])
    shown += list(profile.get('input_traps', ()))  # input_fidelity's lines quote values
    return shown


def input_canaries(task_dir: Path) -> set[str]:
    """GUIDs a canary marker names in any input file; other GUIDs are ordinary data.

    Markers: `...-canary GUID <g>`, `canary_guid = "<g>"`, or a string under a JSON/TOML key
    containing 'canary'. The official /input keeps them and g2 flags any registered canary in
    an output file, so they are withheld from the prompt and output copies are reported. A zip
    container (an archive, a workbook) is also read member by member, bounded, since the profile
    shows what it holds: there a GUID near the word canary counts too.
    """
    found, left = set(), CONTAINER_SCAN_BYTES
    for path in task_dir.rglob('*'):
        try:
            if path.is_symlink() or not path.is_file() or path.stat().st_size > 64*1024*1024:
                continue
            data = path.read_bytes()
            found.update(_canary_texts(path.name, data))
            if data[:4] == b'PK\x03\x04' and left > 0:
                for name, member in archive_texts(path):
                    found.update(_canary_texts(name, member) | near_canaries(member))
                    left -= len(member)
                    if left <= 0:
                        break
        except Exception:  # an unreadable or unparsable file names no further canary
            continue
    return found


def utf8_safe(text: str) -> str:
    """Text House can encode: a lone surrogate (os.fsdecode's stand-in for an undecodable filename byte)
    becomes U+FFFD, so one odd input name cannot make the whole prompt unsendable."""
    try:
        return text.encode('utf-8', 'surrogateescape').decode('utf-8', 'replace')
    except UnicodeEncodeError:  # a surrogate surrogateescape does not map (e.g. from a JSON escape)
        return text.encode('utf-8', 'surrogatepass').decode('utf-8', 'replace')


def task_instruction(text: str, canaries) -> str:
    """The TASK text as sent: lines naming an input canary and the standard banner line beside it are metadata,
    not task text (staging.stage_unit drops the same banner); every other line is kept verbatim."""
    return ''.join(line for line in text.splitlines(keepends=True)
                   if not any(guid in line.lower() for guid in canaries) and not canary_banner(line))


def withhold_canaries(text: str, canaries) -> str:
    """Every text sent to the model passes here: UTF-8-safe, with each input canary withheld."""
    text = utf8_safe(text)
    for guid in canaries:
        text = re.sub(re.escape(guid), '[withheld]', text, flags=re.I)
    return text


def _dockerfile_mappings(task_dir: Path) -> list[tuple[str, str]]:
    """Literal `COPY data...` lines, parsed leniently one by one: these are prompt hints only.

    Nothing is mounted, so unlike staging.data_mounts a line it cannot read is skipped
    rather than failing the whole unit.
    """
    pairs, root = [], task_dir.resolve()
    for relative in ('environment/Dockerfile', 'Dockerfile'):
        dockerfile = task_dir/relative
        if dockerfile.is_symlink() or not dockerfile.is_file():
            continue
        workdir = PurePosixPath('/')
        for line in dockerfile.read_text(errors='replace').splitlines():
            keyword, _, rest = line.strip().partition(' ')
            try:
                if keyword.upper() == 'WORKDIR':
                    workdir = workdir / rest.strip()
                    continue
                if keyword.upper() != 'COPY':
                    continue
                operands = json.loads(rest) if rest.strip().startswith('[') else shlex.split(rest)
                *sources, target = [x for x in operands if not x.startswith('--')]
            except (ValueError, TypeError):
                continue
            for source in sources:
                path = dockerfile.parent/source.rstrip('/')
                if (not source.startswith('data') or path.is_symlink() or not path.exists()
                        or not path.resolve().is_relative_to(root)):
                    continue
                destination = workdir/target
                if path.is_dir() and destination == PurePosixPath('/app'):
                    # Legacy COPY-to-root: file by file beside /app/output. Every copied input file is named
                    # (the organizer's image has it at /app); only organizer metadata is left out. Locally the
                    # stager has already removed what its leakage filter blocks, so nothing more is hidden.
                    pairs.extend(('/app/' + item.relative_to(path).as_posix(), str(item)) for item in _tree(path)
                                 if not any(part.lower() in _NOT_HINTED for part in item.relative_to(path).parts))
                    continue
                if path.is_file() and (target.endswith('/') or len(sources) > 1):
                    destination /= path.name
                if destination.parts[:2] == ('/', 'app'):
                    pairs.append((destination.as_posix(), str(path)))
    return pairs


HINT_FILES = PROFILE_FILES  # per-file /app pairs in the prompt; a COPY data/ /app/ may name thousands
HINT_DIRS = 5
_NOT_HINTED = {'dockerfile', 'manifest.json', 'staging_manifest.json', '__pycache__', '.git'}


def _tree(path: Path) -> list[Path]:
    """Regular files under path in staging's order (sorted, depth first); links are skipped."""
    if path.is_symlink():
        return []
    if path.is_dir():
        return [item for child in sorted(path.iterdir()) for item in _tree(child)]
    return [path] if path.is_file() else []


def input_mapping(task_dir: Path) -> list[tuple[str, str]]:
    """Map the task's /app input paths into /input; official agents get no /app data binds."""
    manifest = task_dir/'staging_manifest.json'
    if not manifest.exists():
        return _dockerfile_mappings(task_dir)
    try:
        mounts = json.loads(manifest.read_text()).get('data_mounts', [])
        return [(m['destination'], str(task_dir/m['source_relative'])) for m in mounts
                if (task_dir/m['source_relative']).exists()]
    except (OSError, ValueError, TypeError, AttributeError, KeyError):
        return []


def _shared_directory(app: str, path: str):
    """('/app', '<dir>') when path ends in the same relative name as app, else None."""
    a, p = PurePosixPath(app).parts, PurePosixPath(path).parts
    n = 0
    while n < min(len(a), len(p)) - 1 and a[-1 - n] == p[-1 - n]:
        n += 1
    return (PurePosixPath(*a[:-n]).as_posix(), PurePosixPath(*p[:-n]).as_posix()) if n else None


def input_hints(task_dir: Path, limit: int = HINT_FILES) -> str:
    """At most `limit` per-file /app pairs; the rest share a directory-level fallback and are counted."""
    pairs = input_mapping(task_dir)
    hints = [f'{app} -> {path}' for app, path in pairs[:limit]]
    rest = pairs[limit:]
    if rest:
        shared = []
        for app, path in rest:
            pair = _shared_directory(app, path)
            if pair and pair not in shared:
                shared.append(pair)
        hints += [f'{app}/<name> -> {path}/<name>' for app, path in shared[:HINT_DIRS]]
        hints.append(f'{len(rest)} more /app input files are not listed here')
    if not any(app == '/app/data' for app, _ in pairs):
        for root in ('environment/data', 'data'):
            if (task_dir/root).is_dir():
                hints.append(f'/app/data/<name> -> {task_dir}/{root}/<name>')
                break
    if not hints:
        return ''
    return ('Input paths under /app named by the task are not mounted here; read them at: '
            + '; '.join(hints) + '. ')


def output_canary_files(out: Path, canaries) -> dict:
    """Count output files carrying an input canary. g2 would flag them; nothing is rewritten."""
    needles = [guid.encode() for guid in canaries]
    result = {'output_canary_files': 0, 'output_unscanned_files': 0}
    if not needles:
        return result
    for path in sorted(out.rglob('*')):
        if path.is_symlink() or not path.is_file():
            continue
        try:
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            with os.fdopen(fd, 'rb') as stream:
                data = stream.read().lower()
        except OSError:
            result['output_unscanned_files'] += 1
            continue
        if any(needle in data for needle in needles):
            result['output_canary_files'] += 1
    return result


# B2 (cap-synthesis "B2-lead"): verified-prefix continuation, patch repair, measured-rate gates.
MAX_REQUESTS = 25  # the organizer's 25 admitted requests per unit (retries count)
MAX_REPLIES = 4  # the first reply plus continuations and the one rewrite
MAX_REPAIRS = 4  # patch rounds; each is gated on time and requests, and repeats of one failure stop early
MAX_PATH_REWRITES = 3
MAX_RAISE_GUARDS = 2  # zero-request rewrites of a program's own check that a patch left raising (T1_RAISEGUARD)
TRANSIENT_WAITS = (2, 4, 8, 15, 25, 40)  # backoff after 408/409/425/429/5xx or a transport error, while time allows
RETRY_AFTER_CAP = 60
RETRY_STATUSES = {408, 409, 425, 429, 500, 502, 503, 504}
# Local evaluation only (T1_EVAL_THROTTLE=1, which the official image never sets): a public endpoint throttles far
# more than the platform's House. The time slept on a 429 or 503 backoff is credited back to the unit, at most
# EVAL_PAUSE_CAP seconds per unit. Only a refused request (429), which costs no request, is retried past
# TRANSIENT_WAITS, on the longer (still bounded) schedule; every other failure stops where the platform's does.
EVAL_TRANSIENT_WAITS = TRANSIENT_WAITS + (60,) * 8
EVAL_PAUSE_STATUSES = (429, 503)
EVAL_PAUSE_CAP = 900.0
STAGE_CAP = 420.0  # every unit shares one 43,200 s stage clock; the card timeout is only a ceiling
EXEC_FLOOR = 45
STATUS_SOFT_LIMIT = 19000  # evaluator.read_status reads at most 20,000 bytes
PROMPT_CAP = 150_000
FIRST_PROMPT_CAP = PROMPT_CAP - 30_000  # the task prompt leaves room for the program and the asks after it
DEFAULT_RATE = 105.0
OUTPUT_FORMAT = ('OUTPUT FORMAT: Reply with exactly one ```python fence and nothing before or after it. '
                 'Structure the program as small top-level functions (each under about 50 lines), then a short '
                 'main() that calls them, then `if __name__ == "__main__": main()`. Keep comments minimal; no '
                 'docstrings, no explanation.')
REWRITE = ('Your previous reply used the whole output limit without usable code. Output only one ```python '
           'fence now, no reasoning or explanation; keep the program compact (small functions) and keep every '
           'required deliverable.')
CONTINUE = ('Your previous reply hit the output limit. The code above is the verified part of your program: '
            '{n} complete top-level statements, the last one beginning `{last}`; everything after it was '
            'discarded. Reply with exactly one ```python fence containing ONLY the remaining top-level '
            'statements, starting with {start}. Do not repeat code above. No text outside the fence. Finish '
            'with the `if __name__ == "__main__":` block.')
CONTINUE_LINE = ('Your reply hit the output limit inside the last statement. Continue from the line after '
                 '`{line}`, keeping its indentation. Output only the remaining lines in one ```python fence; '
                 'no repetition, no text.')
FILL = ('The program above calls {names} but never defines them. Reply with one ```python fence containing '
        'ONLY the missing top-level definitions. Do not repeat code above. No text outside the fence.')
PATCH_FORMAT = ('Reply with one ```python fence containing ONLY the code to change: complete '
                'def/class blocks, new import lines or corrected UPPER_CASE constants (a def or class with an '
                'existing name replaces the old one), or, for a small change inside a long function, just the '
                'changed lines with the unchanged line before and after them, at their original indentation. Do '
                'not repeat unchanged code. No text outside the fence.')
NO_FALLBACK = ('Where the TASK gives no rule for missing or invalid data, raise an error that says what is missing '
               'instead of silently filling a default; where it states a rule, follow the TASK. ')
# T1_RAISEGUARD: the error is for missing or invalid input data only; a check of the program's own results reports.
RESULT_CHECKS = ('A check of your own computed results (a sum, identity, bound or range) never raises or exits: '
                 '{where}. ')
RESULT_CHECKS_WHERE = ('it belongs in self_check, which only reports', 'print a warning and write the outputs')
FALLBACK_FIX = ('Fix the cause: filling missing or NaN values with a default is not a fix unless the TASK says '
                'so. ')
REPAIR = ('The program above ran and failed with exit code {code}. Its last output lines:\n```\n{tail}\n```\n'
          '{report}{history}{lint}{paths}' + FALLBACK_FIX + PATCH_FORMAT)
LINT = ('A static check of the program (advisory) also found calls of its own functions that cannot work as '
        'written; fix those on the failing path, and the others too if they would fail later:\n{items}\n')
LINT_ALL = ('A static check of the program (advisory) also found code that cannot work as written; fix what is on '
            'the failing path, and the rest too if it would fail later:\n{items}\n')
REPORT = ('Crash report (program frames, innermost last, and the local variables where it failed):\n'
          '```\n{report}\n```\n')
SAME_FAILURE = ('Your previous change{names} did not fix this: the same {exc} is raised {place}{error}. Your change '
                'did not address it, so look for the cause elsewhere, such as the types, keys or columns shown '
                'above.\n')
SAME_ATTRIBUTE = ('Your previous change{names} did not fix this: the same {exc} is raised {place}{error}. The cause is '
                  'on that line: module `{module}` has no `{attr}`. Take `{attr}` from the module that has it (import '
                  'it there and use that name), or use a name `{module}` does have{close}.\n')
_NO_ATTRIBUTE = re.compile(r"AttributeError: module '([\w.]{1,120})' has no attribute '(\w{1,120})'")
RETRY_ASK = ('Reply with one ```python fence containing the complete corrected def/class blocks (or new import lines '
             'or constants) that fix the error; the cause may also be in a caller or where the data is prepared. No '
             'text outside the fence.')
RETRY = ('Your reply did not change the program: {why}. It did not address the error{error}, so the program would '
         'fail the same way{where}. ' + RETRY_ASK)
LINT_RETRY = ('Your change leaves code that cannot work as written; the program would fail there:\n{items}\nReply '
              'with one ```python fence containing the complete corrected def blocks (the callers, or the def) so that '
              'every call matches the function it calls; keep the rest of your change. No text outside the fence.')
SCOPE_LINT_RETRY = ('Your change introduces static risks in the merged program:\n{items}\n'
                   'A reported global read has no statically visible binding in the scopes checked; this is '
                   'limited static evidence, not proof that execution must fail. Check the intended data flow '
                   'and any listed call mismatch. Reply with one ```python fence containing the complete '
                   'corrected def blocks or necessary module bindings; keep the rest of your change. '
                   'Do not guess parameters or data columns. No text outside the fence.')
REVERT = ('Your change brings back `{line}` in `{function}`, which already failed in an earlier run{error}. Do not '
          'go back to code that already failed: fix the error{where} another way. ' + RETRY_ASK)
RERUN = ('Your change turns the program back into a version that already ran and failed{error}. Do not go back to '
         'it: fix the error{where} another way. ' + RETRY_ASK)
REVIEW_RETRY = ('Your reply did not change the program: {why}. Its outputs would still contradict the TASK as '
                'listed; it did not address the first listed problem: {first}. Reply with one ```python fence '
                'containing the complete corrected def/class blocks (or new import lines or constants) that fix the '
                'listed problems; the cause may also be in a caller or where the data is prepared. No text outside '
                'the fence.')
SALVAGE = ('Your reply broke off into invalid Python at the statement that begins `{anchor}` ({error}), so it '
           'and everything after it were discarded. The code above is the verified part of your program: {n} '
           'complete top-level statements, the last one beginning `{last}`. Reply with exactly one ```python '
           'fence containing ONLY the remaining top-level statements, starting with a correct version of that '
           'statement. Do not repeat code above. No text outside the fence. Finish with the '
           '`if __name__ == "__main__":` block.')
PATCH_CONTINUE = ('Your previous reply hit the output limit. The code above is the verified part of your patch; '
                  'everything after it was discarded. Reply with one ```python fence containing ONLY the rest '
                  'of the patch, starting with {start}. No text outside the fence.')
# X2 (image flag T1_STAGED=1): when the continuations of a first program reply cut at the output cap end without a
# complete program, staged generation takes over: a skeleton of stubs, then the stubs' bodies, each reply taking as
# many whole functions as fit.
STAGE_AFTER = ('no_progress', 'chunk_cap', 'empty_code', 'invalid_code', 'no_entry')  # continuation stops staged
STAGE_REPLIES = 8  # replies of staged generation, the skeleton included
STAGE_OUTPUT_NAMES = 30  # output names listed in one request
STAGE_DEF_LINES = 60  # lines asked of each function
STAGE_SKELETON_LINES = 150
STAGE_SPLIT_LINES = 30  # lines asked of a function rewritten to call helpers because it did not fit one reply
STAGE_ITEMS = 25  # pending functions listed in one request
STAGE_SKELETON = ('Your reply hit the output limit: this program is too long for one reply, so it is now written in '
                  'stages. {kept}First reply with the SKELETON of {whole}, in one ```python fence: the imports and '
                  'constants it needs, every top-level function as a stub (its def line with all parameters, a one-line '
                  'docstring saying what it reads, computes or writes and what it returns, and the body `...`), then '
                  'a complete main() that only calls these functions in order and passes their results on, then the '
                  '`if __name__ == "__main__":` block. Split the work into functions of at most about {lines} lines '
                  'each: one per input read, per computed quantity or table, and per output file written. The '
                  'skeleton must stay short (at most about {cap} lines): each stub is written in full in a later '
                  'reply. No text outside the fence.')
STAGE_KEPT = ('The code above is the verified part of your program ({n} complete top-level statements); keep it as it '
              'is and do not repeat it. ')
STAGE_BODIES = ('The code above is the program so far. These functions are still stubs or missing:\n{items}\nReply '
                'with one ```python fence containing their complete definitions, in this order: each a complete def '
                'block with its full body, keeping its name and parameters. Send as many as fit in one reply, '
                'finishing each before starting the next; the rest are asked for next. Keep each under about {lines} '
                'lines by moving a long part into a new helper function, defined in the same reply. Do not repeat '
                'other code. No text outside the fence.')
STAGE_SPLIT = ('Your reply hit the output limit before `{name}` was complete: it is too long for one reply. Reply with '
               'one ```python fence containing `{name}` rewritten as a short function (at most about {lines} lines, '
               'the same name and parameters) that calls new helper functions for its parts, and each new helper as '
               'a stub: its def line, a one-line docstring and the body `...`. The helpers are written in the next '
               'replies. No text outside the fence.')
STAGE_OUTPUTS = (' Your earlier reply already wrote these output files and keys, which the program above does not '
                 'have yet; keep each one, exactly as named, where its output is written: {names}.')
STAGE_AGAIN = 'Your last reply wrote none of what was asked. '
STAGE_INVALID = 'Your last reply had invalid Python ({error}); that statement was not taken. '
STAGE_SHORTER = ('Your last reply hit the output limit too: the skeleton must be short, with the body of every '
                 'function except main() just `...`. ')
PLAN_TIMEOUT = 150  # the planning reply's own timeout, so the program keeps most of the unit's time
PLAN_CHARS = 8000  # plan text carried into the program prompt
PLAN_STATUSES = ('ok', 'reasoning_only', 'empty', 'failed', 'skipped_time')
PLAN_ASK = ('PLAN FIRST: do not write the program yet. Work out how to solve the task exactly as specified, '
            'then reply with a concise numbered implementation plan in plain text (no code, at most 400 words) '
            'covering: (1) every required output file with its exact path, format, keys or columns, units and '
            'rounding; (2) the exact formula and convention for each quantity (annualisation, day counts, return '
            'definitions, ddof, signs, window alignment and inclusive or exclusive bounds), quoting the task '
            'wherever it states them; (3) how each input file is read, cleaned and aligned; (4) the random seed '
            'and the order of random draws when the task simulates; (5) the steps where a plausible choice would '
            'contradict the task, and the choice the task requires, including what happens to missing or invalid '
            'data: the TASK\'s rule, or, where it gives none, an error rather than a default.')
PLAN_FACTS = (' Then list (6) CHECKABLE FACTS, in at most 150 more words: each fact the TASK states that the written '
              'outputs must satisfy and that a program can verify by re-reading them (an identity between outputs, '
              'a count, a sum, a range or bound, an explicit rule or condition, a timing convention), with the TASK '
              'sentence it comes from quoted verbatim.')
PLAN_LIBRARY = (' For each standard method the TASK names without writing out its formula (a named pricing model, '
                'estimator, statistical test, numerical scheme or optimisation), say which function of the optional '
                'library computes it and with which of the TASK\'s conventions as arguments, or that none does; where '
                'the TASK spells out its own formula or convention, follow the TASK instead.')
PLAN_LIBRARY_MAP = (' Then list ({n}) LIBRARY MAP, one line per standard method the TASK names without writing out '
                    'its formula (a named pricing model or approximation, estimator, statistical test, numerical '
                    'scheme or optimisation), exactly `<method> -> qfkit.<function>(<the TASK\'s conventions as '
                    'arguments>)` when a catalog function computes it, or `<method> -> none` when none does or the '
                    'TASK spells out its own formula or convention for it. Name only functions the catalog lists, as '
                    'qfkit.<name>.')
PLAN_LIBRARY_MAP_ADOPT = (' Then list ({n}) LIBRARY MAP, one line per standard method the outputs need (a named '
                          'pricing model or approximation, estimator, statistical test, numerical scheme or '
                          'optimisation): exactly `<method> -> qfkit.<function>(<the TASK\'s conventions as '
                          'arguments>)` when a catalog function computes it, and the program will then call that '
                          'function to compute that output (map no function as optional or for verification only); '
                          '`<method> -> none: "<the TASK sentence, copied verbatim>"` only where the TASK writes out '
                          'a different formula or convention for it than the function computes; `<method> -> none: '
                          'different model: <why>` where a catalog function has the name the TASK uses but computes '
                          'a different model than the TASK means; `<method> -> none` when no catalog function '
                          'computes it. Name only functions the catalog lists, as qfkit.<name>.')
# T1_ROUTE: the plan also names the task's family (families.FAMILIES) and quotes the TASK sentence that pins each
# graded method; the program prompt carries the family's note and repeats those lines (route_prompts).
PLAN_ROUTE = (' Then give ({n}) FAMILY: one line `FAMILY: <label>` with exactly one label from this list, the one that '
              'fits the computation of the graded outputs: {labels}. Then list ({m}) METHOD PINS, in at most 150 more '
              'words: one line per graded output or computation, `<output or computation>: "<the TASK sentence that '
              'pins its method (formula, scheme, seed, path count, estimator, window or convention), copied '
              'verbatim>"`, or `<output or computation>: free` where the TASK leaves its method free.')
ROUTE_NOTES = 'FAMILY NOTES ({label}):\n{note}\n\n'
ROUTE_PINS = ('METHOD PINS (from your plan: the TASK sentence that pins each graded method, or free; follow each '
              'pinned method exactly as written, choose a method only where a line says free, and where a line '
              'differs from the TASK above, the TASK is right):\n{pins}\n\n')
ROUTE_PINS_CAP = 1500  # characters of the METHOD PINS lines the program prompt repeats
ROUTE_PIN_LINES = 40  # lines read from the plan's METHOD PINS
ROUTE_LIST_LABELS = 3  # a FAMILY value naming this many labels or more copies the ask's list: it chooses none
SELFCHECK_NAME = 'self_check'
SELFCHECK_ENV = 'T1_SELFCHECK_FILE'
SELFCHECK_SECONDS = 15.0  # a self_check() call that runs longer is stopped and counts as not run
SELFCHECK_BLOCK = ('SELF-CHECK: define a top-level function self_check() and call it as the last step of main(), after '
                   'every output is written. It re-reads the written outputs (and inputs where needed) and checks each '
                   'fact the TASK states about them that a program can verify: identities between outputs, counts, '
                   'sums, ranges, explicit rules, timing conventions. Define it after main(), just before the '
                   '`if __name__ == "__main__":` block, and keep it short: at most about 25 lines. It must take a few '
                   'seconds at most: it compares '
                   'what the files hold and never repeats the program\'s main computation (a check still running '
                   'after ' + f'{SELFCHECK_SECONDS:.0f}' + ' s is stopped and counts as not run). It writes a JSON list '
                   'of {"quote": <the TASK '
                   'sentence, copied verbatim>, "ok": <true or false>, "detail": <what was compared, in a few '
                   'words>} to the path in the environment variable ' + SELFCHECK_ENV + ' (nothing when it is '
                   'unset), never under the output directory. It only reports: it never raises, exits or changes an '
                   'output, a check that cannot run is left out, and no output is ever adjusted to make a check '
                   'pass.\n\n')
SELFCHECK_STUB = f'def {SELFCHECK_NAME}(*args, **kwargs):\n    return None\n'  # a self-check cut at the cap
SELFCHECK_LIMIT = 64 * 1024  # bytes of the side file read
SELFCHECK_ITEMS = 40
SELFCHECK_FLAGS = 8
REVIEW_ROUNDS = 2  # guarded fix rounds of the review when the self-check is on (one otherwise)
PLAN_BLOCK = ('YOUR IMPLEMENTATION PLAN (from your own analysis in a previous step; where it differs from the '
              'TASK above, the TASK is right):\n{plan}\n\n')
# The thinking lever, each part behind its own switch (unset: off, every request as before): a bounded reasoning
# budget, in tokens of the same 4,000-token reply, for the first program request (T1_R1THINK), the plan (T1_PLANTHINK,
# with the plan guard) and the first patch request of a repair round (T1_PATCHTHINK). A budget goes out only while the
# reply, its reasoning at the measured rate, the reserve after it and THINK_MARGIN fit (think_fits); otherwise that
# request goes as without the switch. Low-effort plan thinking yields about 20 reasoning tokens.
THINK_MARGIN = 60
R1THINK_BUDGET = 1000
R1THINK_PLAN_MAX = 2000  # plan reply completion tokens; a longer plan predicts a program the reasoning would cut
PLANTHINK_BUDGET = 1000
PATCHTHINK_BUDGET = 1000  # probe 10-01: at 2,000 one of two patches turned into a whole program and was cut
# A budgeted plan's visible answer grew to 2,800-3,000 tokens in the probe and was cut in 4 of 4 units: under
# T1_PLANTHINK the plan ask ends by keeping the written plan short (the reasoning stays hidden from the program).
PLANTHINK_SHORT = (' Your reasoning is not passed on; only the written plan is. Keep the written plan to at most '
                   '350 words in total, all items included: state each decision once, without code and without '
                   'restating your reasoning.')
R1THINK_STATUSES = ('used', 'skipped_no_plan', 'skipped_plan_long', 'skipped_time')
PLAN_MAP_STATUSES = ('recovered', 'no_map', 'failed', 'skipped')
PLAN_CODE_CUT = '[code omitted]'  # T1_PLANTHINK: a code block of a plan over PLAN_CHARS
PLAN_MAP_ASK = ('Your plan hit the output limit before its LIBRARY MAP. Reply now with ONLY the LIBRARY MAP lines, '
                'one per standard method the outputs need, exactly in the format asked above; no other text and no '
                'code.')
RUN_CAP_FLOOR = 220.0  # seconds; every clean run seen so far took at most 201 s
STACK_LEAD = 3.0  # seconds between the stack sample and the stop of a capped run
MAX_TIMEOUT_REPAIRS = 1
TIMEOUT = ('The program above was stopped after {sec:.0f} s without finishing; about {left:.0f} s remain for the '
           'whole task, so it must finish well within that. Where it was when stopped (most recent call '
           'first):\n```\n{tail}\n```\nMake it fast without changing what it computes: replace Python loops '
           'over observations, paths or grid points with numpy array operations, compute constant weights and '
           'densities once outside loops, never call scipy.stats functions on single numbers inside a loop, and '
           'keep the TASK\'s methods, parameters and sample sizes. ' + PATCH_FORMAT)
FRESH_STATUSES = ('kept', 'rejected', 'duplicate', 'skipped_time', 'failed')
FRESH_DEAD_ENDS = ('noop', 'repeated_failure', 'max_repairs', 'request_failed')  # crash endings that never pass
FRESH_CRASH_REPAIRS = 2  # a garbled first patch must not end the fresh program
REVIEW_STATUSES = ('clean', 'fixed', 'fix_rejected', 'no_patch', 'failed', 'skipped_time', 'too_large')
REVIEW_SNAPSHOT_CAP = 16 * 1024 * 1024  # bytes of outputs kept in memory to restore them after a rejected fix
REVIEW = ('The program above ran and exited 0. Its outputs:\n```\n{digest}\n```\n{flags}Check these outputs '
          'against the TASK. List every place where an output contradicts an explicit statement of the TASK: a '
          'required file or key that is missing or misnamed; a value outside a tolerance, bound, range, identity, '
          'sign, count or ordering that the TASK states; a constant or placeholder where the TASK says to compute '
          'something. One line each, exactly: QUOTE: <the TASK sentence, copied verbatim> | OUTPUT: <file:key or '
          'file:column> | PROBLEM: <what is wrong>. Only contradictions of what the TASK says; no style advice. '
          'If there is none, reply exactly NONE.')
REVIEW_FLAGS = 'A check of the outputs against the TASK already found:\n{flags}\n'
REVIEW_HINTS = ('An automatic scan also noticed the following; each may still be what the TASK asks for, so list '
                'one only where it contradicts the TASK:\n{hints}\n')
FIX_OUTPUTS = ('The program above ran and exited 0, but its outputs contradict the TASK:\n{items}\nFix how the '
               'affected quantities are computed or written, and change nothing else. Re-derive what each symbol '
               'in the TASK formulas denotes; a symbol the TASK never defines is the unknown of its equation, not '
               'a quantity listed elsewhere. Never change or loosen a check, never compute a quantity from the '
               'identity that checks it, and fix every listed item in this one reply. {note}' + FALLBACK_FIX
               + PATCH_FORMAT)
CHECK_FIX = ('A self-check can be wrong too: where the quoted TASK sentence supports the outputs rather than the '
             'check, correct self_check() and leave the outputs as they are. ')
# T1_ACCEPT (image flag): after the first clean run, the model writes a separate checker of the conditions the TASK
# states about the outputs; it runs beside the program, and a failing condition opens a guarded fix round. It takes
# the place of the in-program self-check (SELFCHECK_BLOCK), whose code cut the program's own reply short.
ACCEPT_STATUSES = ('clean', 'fixed', 'fix_rejected', 'no_patch', 'no_checker', 'checker_failed', 'no_conditions',
                   'skipped_time', 'failed', 'too_large')
ACCEPT_SECONDS = 30.0  # the checker's own time limit
ACCEPT_ROUNDS = 2  # guarded fix rounds; a rejected fix is asked again once, saying why it was not kept
ACCEPT_ITEMS = 30  # conditions read from the checker's list
ACCEPT_LISTED = 8  # failing conditions listed in one fix request
ACCEPT_LIMIT = 64 * 1024  # bytes of the checker's printed output read
ACCEPT_LINES = 60
ACCEPT_KEEP_REQUESTS = 3  # requests left after the acceptance step: the review, its fix and the re-review
ACCEPT_ASK = ('The program above ran and exited 0. Its outputs (in {out}):\n```\n{digest}\n```\nNow write a separate '
              'short checker script (at most about {lines} lines; numpy, pandas, scipy and the standard library only) '
              'that re-reads these output files, and the inputs where needed, and evaluates ONLY the conditions the '
              'TASK itself states that the outputs must satisfy: acceptance or validation criteria, tolerances, '
              'identities and parity relations, bounds and ranges, counts, orderings and monotonicity, residual '
              'thresholds, targets. Check what the TASK says the results must satisfy, not the steps it describes for '
              'computing them, and never repeat the program\'s main computation: compare what the files hold. Use the '
              'TASK\'s own tolerance; where it states none, a loose one that only a real error exceeds (about 1e-3 '
              'relative). Leave out file names, columns, keys and number formats (another check covers them). Wrap '
              'each condition in its own try/except and leave out one that cannot be evaluated. Print exactly one JSON '
              'list and nothing else: [{{"quote": <the TASK sentence that states the condition, copied verbatim>, '
              '"ok": <true or false>, "detail": <the compared values, in a few words>}}]. It must finish within '
              '{sec:.0f} s and never write or change a file. Reply with one ```python fence and nothing else.')
ACCEPT_FIX = ('The program above ran and exited 0, but a check of its outputs against conditions the TASK states found '
              'that these do not hold:\n{items}\n{note}Fix how the affected quantities are computed or written so that '
              'the outputs satisfy the TASK as stated, and change nothing else. Re-derive each from the formula and '
              'conventions the TASK states. Never compute a quantity from the condition that checks it, never adjust an '
              'output just to pass a check and never loosen a tolerance; where a listed condition misreads the TASK, '
              'leave those outputs as they are. A property the TASK expects of a correct result can be loose for these '
              'inputs: never change a correct formula to force it. ' + FALLBACK_FIX + PATCH_FORMAT)
ACCEPT_AGAIN = 'Your previous change was not kept: {why}. '
_MISSING_FILE = re.compile(r"No such file or directory: '([^'\n]{1,400})'")


def card_timeout(task_dir: Path):
    try:
        timeout = tomllib.loads((task_dir/'card.toml').read_text())['agent']['timeout_sec']
    except (OSError, ValueError, KeyError, TypeError, UnicodeError):
        return None
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
        return None
    return float(timeout)


def card_budget(task_dir: Path, default: float = STAGE_CAP, cap: float = STAGE_CAP) -> float:
    """[agent].timeout_sec (SUBMISSION_CLI) bounds the container; the solve budget keeps a margin
    under it and never exceeds the unit's share of the stage clock (cap)."""
    timeout = card_timeout(task_dir)
    budget = default if timeout is None else max(60.0, min(timeout * 0.8, timeout - 120))
    return float(min(cap, budget))


def _seconds(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def solve_budget(task_dir: Path, env=None) -> tuple:
    """(budget, source). The image sets UNIT_BUDGET_SEC (the unit's share of the stage clock), applied under
    the card's ceiling; the platform passes nothing else. T1_TIMEOUT_SEC, set only by local harnesses, is an
    explicit override clamped to STAGE_CAP (an invalid one reaches solve, which records the error)."""
    env = os.environ if env is None else env
    if env.get('T1_TIMEOUT_SEC'):
        explicit = _seconds(env['T1_TIMEOUT_SEC'])
        return (min(explicit, STAGE_CAP) if explicit else math.nan), 'explicit'
    unit = _seconds(env.get('UNIT_BUDGET_SEC'))
    return card_budget(task_dir, unit or STAGE_CAP, cap=unit or STAGE_CAP), 'image' if unit else 'default'


class PauseClock:
    """The unit's budget clock: every deadline in this module is read against now(). It is time.monotonic() itself
    unless T1_EVAL_THROTTLE (local evaluation only) credited back time slept on a throttled backoff, at most
    EVAL_PAUSE_CAP seconds. One per process, as solve runs one unit per container; solve resets it."""

    def __init__(self):
        self.credited = 0.0

    def reset(self):
        self.credited = 0.0

    def now(self):
        now = time.monotonic()
        return now - self.credited if self.credited else now

    def creditable(self, wait):
        """How much of a wait the cap still lets be credited."""
        return max(0.0, min(float(wait), EVAL_PAUSE_CAP - self.credited))

    def credit(self, seconds):
        self.credited += max(0.0, min(seconds, EVAL_PAUSE_CAP - self.credited))


CLOCK = PauseClock()


class Pace:
    """Measured House output rate; gates and timeouts assume a full 4,000-token reply. measured: a reply has set the
    rate (T1_S10's gates assume at most S10_PRIOR_RATE until then). Under T1_S10 the rate is the lowest measured so
    far, not the last sample: the probe's rate varied 62-168 tok/s within an hour, and a single fast DECIDE sample
    set timeouts that killed the slower replies after it. s12_halved (T1_S12_MECH) is the lowest measured rate that a
    note request's timeout halved (_request): the next sample replaces the halved guess, since a note reply can time
    out for its length as well as for a slow House (r13b bl-regime: STATE timed out after DECIDE's 65.5 tok/s, and the
    next replies ran at 87 and 91.5)."""

    def __init__(self, rate=DEFAULT_RATE):
        self.rate = rate
        self.measured = False

    def full(self):
        return 4000 / self.rate + 6

    def patch(self):
        return 2500 / self.rate + 6

    def measure(self, client, reply):
        tokens, seconds = reply['usage'].get('completion_tokens'), client.diagnostics['requests'][-1].get('duration_sec')
        if type(tokens) is int and tokens >= 500 and seconds:
            sample = min(400.0, max(20.0, tokens / seconds))
            low = getattr(self, 's12_halved', None)  # T1_S12_MECH: a note timeout's halving holds until this sample
            self.rate = min(self.rate if low is None else low, sample) if self.measured and s10_on() else sample
            self.measured, self.s12_halved = True, None


REPLAY_FILE_CAP = 512 * 1024
REPLAY_TOTAL_CAP = 16 * 1024 * 1024
REPLAY_LOG_TAIL = 65536


class Replay:
    """Local public evaluation only: the raw prompt, every reply, every program that ran and its log tail, for
    offline replay. On only when T1_REPLAY_DIR names an existing directory apart from the output tree (the
    platform never sets it). Best-effort: every error is swallowed, nothing is read back or sent to the model,
    and solve's behaviour and status are the same with or without it."""

    def __init__(self, out=None, env=None):
        self.root, self.used, self.seq, self.first = None, 0, 0, None
        try:
            value = (os.environ if env is None else env).get('T1_REPLAY_DIR')
            root = Path(value).resolve(strict=True) if value else None
            outputs = [Path(p).resolve() for p in ('/app/output', '/output', *([out] if out else []))]
            if root is not None and root.is_dir() and not any(
                    root.is_relative_to(p) or p.is_relative_to(root) for p in outputs):
                self.root = root
        except Exception:
            pass

    def put(self, name, data):
        if self.root is None:
            return
        try:
            data = (data if isinstance(data, bytes) else str(data).encode('utf-8', 'replace'))[:REPLAY_FILE_CAP]
            data = data[:max(0, REPLAY_TOTAL_CAP - self.used)]
            self.seq += 1
            if not data:
                return
            fd = os.open(self.root/f'{self.seq:03d}-{name}', os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW, 0o644)
            with os.fdopen(fd, 'wb') as stream:
                stream.write(data)
            self.used += len(data)
        except Exception:
            pass

    def prompt(self, text):
        if self.first is None:
            self.first = text
            self.put('prompt.txt', text)

    def request(self, client, kind, messages, reply=None, error=None):
        if self.root is None:
            return
        try:
            last = messages[-1]['content']
            record = {'kind': kind, 'turns': len(messages), 'user_is_prompt': last == self.first,
                      'user': None if last == self.first else last,
                      'request': dict(client.diagnostics['requests'][-1]) if client.diagnostics['requests'] else {}}
            details = client.diagnostics.get('transport_requests')
            if isinstance(details, list) and details and len(details) == len(client.diagnostics['requests']):
                detail = safe_transport_detail(details[-1])
                if detail:
                    record['transport_detail'] = detail
            if reply is not None:
                record.update(content=reply['content'], finish=reply['finish_reason_raw'], tokens=reply['usage'])
                if reply.get('reasoning'):
                    record['reasoning'] = reply['reasoning']
            if error is not None:
                record['error'] = error
            self.put(f'request-{kind}.json', json.dumps(record, ensure_ascii=False, indent=1))
        except Exception:
            pass

    def run(self, log, status):
        if self.root is None:
            return
        try:
            fd = os.open(log, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            with os.fdopen(fd, 'rb') as stream:
                info = os.fstat(stream.fileno()); size = info.st_size
                if not stat.S_ISREG(info.st_mode):
                    return
                stream.seek(max(0, size - REPLAY_LOG_TAIL))
                self.put('run.log', stream.read(REPLAY_LOG_TAIL))
            self.put('run.json', json.dumps({'status': status.get('status'), 'exit_code': status.get('exit_code'),
                                             'log_bytes': size}))
        except Exception:
            pass


_NO_REPLAY = Replay(env={})


class _Stop(Exception):
    """A request that is not sent: its gate or a cap failed. reason is a status enum."""

    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


def _env_tokens(name, default):
    """A whole number of tokens (1-3999) from the environment; default when it is unset or not one."""
    value = os.environ.get(name, '')
    return int(value) if re.fullmatch(r'[1-9]\d{0,3}', value) and int(value) < 4000 else default


def r1think_budget():
    """T1_R1THINK=1: the first program request's reasoning budget (T1_R1THINK_BUDGET, else R1THINK_BUDGET); None
    when the switch is off."""
    return _env_tokens('T1_R1THINK_BUDGET', R1THINK_BUDGET) if os.environ.get('T1_R1THINK') == '1' else None


def planthink_budget():
    """T1_PLANTHINK=1: the plan request's reasoning budget in place of low effort (T1_PLANTHINK_BUDGET, else
    PLANTHINK_BUDGET), with the plan guard (_plan_text, plan_map); None when the switch is off, and under T1_S10,
    whose fallback plan is stack9's plan with thinking off (_s10_fallback)."""
    return (_env_tokens('T1_PLANTHINK_BUDGET', PLANTHINK_BUDGET)
            if os.environ.get('T1_PLANTHINK') == '1' and not s10_on() else None)


def patchthink_budget(status=None):
    """T1_PATCHTHINK=1: the reasoning budget of a repair round's first patch request (T1_PATCHTHINK_BUDGET, else
    PATCHTHINK_BUDGET); None when the switch is off. Under T1_S10, whatever T1_PATCHTHINK says: T1_PATCHTHINK_BUDGET,
    else S10_PATCH_BUDGET, clamped to the token rule (s10_budget, which counts a clamp in status)."""
    if s10_on():
        return s10_budget('patch', status=status)
    return _env_tokens('T1_PATCHTHINK_BUDGET', PATCHTHINK_BUDGET) if os.environ.get('T1_PATCHTHINK') == '1' else None


def think_fits(deadline, pace, budget, need=None, reserve=EXEC_FLOOR, margin=THINK_MARGIN):
    """Whether a request may carry budget tokens of reasoning: its reply (a full one when need is None), the
    reasoning at the measured rate, the reserve after it and margin (THINK_MARGIN; S10_MARGIN for T1_S10's note
    requests) still fit before the deadline."""
    return (deadline - CLOCK.now()
            >= (pace.full() if need is None else need) + budget / pace.rate + reserve + margin)


def _continuation_capacity(remaining, pace, reserve):
    """An output allowance at the current estimated rate, not a prediction of code needed.

    Keep the original six-second reply margin and execution reserve. These are
    admission estimates, not a guarantee that transport respects its timeout.
    """
    rate = pace.rate
    if not (math.isfinite(remaining) and math.isfinite(rate) and rate > 0):
        return None
    full = pace.full()
    if remaining >= full + reserve:
        return 4000, full
    tokens = min(3999, math.floor((remaining - reserve - 6) * rate))
    if tokens < 512:
        return None
    need = tokens / rate + 6
    if need + reserve > remaining:  # conservative at floating-point boundaries
        tokens -= 1
        need = tokens / rate + 6
    return (tokens, need) if tokens >= 512 and need + reserve <= remaining else None


def _request(client, messages, deadline, pace, kind, sleep=None, need=None, reserve=EXEC_FLOOR, cap=180,
             thinking=False, retry_floor=0.0, keep=0, budget=None, strict=False, think_margin=THINK_MARGIN,
             timeout_floor=0.0, note=False, expect=None, max_tokens=None, once=False, continuation_budget=False):
    """One reply, retrying only transient service errors. need=None is the first request, sent whenever
    time remains (B0's short-budget allowance); later ones need a full reply plus the reserve. A retry of
    the first request is sent only while a full reply, plus retry_floor seconds kept for later work, can
    still arrive before the final reserve. Neither a request nor a retry is sent where it would leave fewer than
    keep requests for later steps. Under T1_EVAL_THROTTLE (the client's throttle; local evaluation only)
    a 429 or 503 wait is credited back to the clock, and a 429 alone is retried on to the end of
    EVAL_TRANSIENT_WAITS. budget (a thinking switch's reasoning tokens) replaces thinking on each attempt that
    think_fits (with think_margin) allows; the record of an attempt sent with it carries reasoning_budget. With
    strict (T1_S10's note requests) an attempt the budget does not fit raises _Stop('think_gate') instead of going
    without it, a transport retry included, and at most S10_NOTE_RETRIES transient retries go out (each is a counted
    request). timeout_floor (T1_S10's note requests: S10_NOTE_TIMEOUT_FLOOR) is the least timeout of a later request
    where the clock allows it, and a request that dies at its timeout leaves a measured rate alone (the rate is the
    lowest measured) instead of halving it. Under T1_S10 a reply sent with a budget feeds the breaker (_s10_breaker);
    once it has tripped (client.s10_think_off) no attempt reasons, and a strict one is not sent. Under T1_S12_MECH,
    expect (the seconds its reply is expected to take) sets the timeout to S12_TIMEOUT_X x expect, with timeout_floor
    only where that is lower and the clock holds the floor; a note request (note) that dies at its timeout halves the
    rate after all, until the next sample (Pace.s12_halved), and is not retried (client.s12_note_timeouts counts it),
    and nor is a crash patch's first request (once), whose light retry follows instead; max_tokens caps a reply below
    4,000. continuation_budget applies only to a code continuation, recomputing a 512..4000
    allowance before each attempt and retry wait."""
    sleep = sleep or time.sleep
    throttle = getattr(client, 'throttle', False) is True
    waits = EVAL_TRANSIENT_WAITS if throttle else TRANSIENT_WAITS
    retries = S10_NOTE_RETRIES if strict else len(waits)
    if need is not None:
        if client.attempts + keep >= client.max_requests:
            raise _Stop('request_cap')
        if sum(len(m['content'].encode()) for m in messages) >= PROMPT_CAP:
            raise _Stop('prompt_cap')
    error = None
    bounded = continuation_budget and kind in ('continue_stmt', 'continue_line') and need is not None
    for attempt in range(retries + 1):
        remaining = deadline - CLOCK.now()
        allowance = _continuation_capacity(remaining, pace, reserve) if bounded else None
        if bounded:
            if allowance is None:
                raise error or _Stop('time_gate')
            tokens, need = allowance
            if max_tokens and max_tokens < tokens:
                tokens, need = max_tokens, max_tokens / pace.rate + 6
        budget_record = ({'max_tokens': tokens, 'continuation_need_sec': round(need, 6),
                          'continuation_remaining_sec': round(remaining, 6),
                          'continuation_reserve_sec': reserve,
                          'continuation_rate_tok_s': round(pace.rate, 6)} if bounded else {})
        if need is None:
            if attempt and remaining - 20 < pace.full() + retry_floor:
                raise error
            if remaining <= 0:
                raise HouseError('No generation time remaining', kind='local_budget', code='generation_budget')
            timeout = max(1, min(cap, remaining - 20))
        elif remaining < need + reserve:
            raise error or _Stop('time_gate')
        else:
            timeout = max(need, min(remaining - reserve, max(1.5 * pace.full() + 10, timeout_floor)))
            if expect is not None and s12_mech_on():  # T1_S12_MECH: twice the expected reply; the floor where it fits
                room = remaining - reserve
                timeout = max(need, min(room, max(S12_TIMEOUT_X * expect, timeout_floor if timeout_floor <= room
                                                  else 0.0)))
        first = len(client.diagnostics['requests'])
        off = getattr(client, 's10_think_off', False) is True  # T1_S10's breaker tripped earlier in the unit
        fits = bool(budget) and not off and think_fits(deadline, pace, budget, need, reserve, think_margin)
        if strict and budget and not fits:
            raise _Stop('think_gate')  # a budgeted step is skipped, never sent unreasoned
        think = budget if fits else (False if off else thinking)
        try:
            options = {'thinking': think} if think else {}
            if max_tokens:  # S12 light patch; never enlarged by continuation admission
                options['max_tokens'] = max_tokens
            if bounded and tokens < 4000:
                options['max_tokens'] = min(tokens, options.get('max_tokens', 4000))
            try:
                reply = client.chat(messages, timeout=timeout, **options)
            finally:
                if budget_record:  # before replay snapshots, including a failed transport attempt
                    for record in client.diagnostics['requests'][first:]:
                        record.update(budget_record)
            getattr(client, 'replay', _NO_REPLAY).request(client, kind, messages, reply=reply)
            if type(think) is int and s10_on():
                _s10_breaker(client, client.diagnostics['requests'][-1], think, reply)
            return reply
        except HouseError as exc:
            getattr(client, 'replay', _NO_REPLAY).request(client, kind, messages, error=exc.code)
            error = exc
            records = client.diagnostics['requests'][first:]
            timed_out = bool(exc.code == 'transport_failure' and records
                             and records[-1]['duration_sec'] >= 0.9 * timeout)
            halt = timed_out and (note or once) and s12_mech_on()  # T1_S12_MECH: a timed-out note or patch: no retry
            if timed_out and (halt or not (timeout_floor and pace.measured)):  # a note step keeps the lowest rate
                if halt and note and pace.measured:  # T1_S12_MECH: until the next sample (Pace.s12_halved)
                    pace.s12_halved = getattr(pace, 's12_halved', None) or pace.rate
                pace.rate = max(20.0, pace.rate / 2)  # a request timeout: the House is slower than measured
            if halt:
                if note:
                    client.s12_note_timeouts = getattr(client, 's12_note_timeouts', 0) + 1
                raise
            transient = exc.code == 'transport_failure' or (
                exc.code == 'http_error' and client.diagnostics.get('http_status') in RETRY_STATUSES)
            if not transient or attempt == retries or client.attempts + keep >= client.max_requests:
                raise
            # Past the platform's schedule (T1_EVAL_THROTTLE only) a refused 429 alone is retried.
            if attempt >= len(TRANSIENT_WAITS) and not (exc.code == 'http_error'
                                                        and client.diagnostics.get('http_status') == 429):
                raise
            wait = waits[attempt]
            hint = client.diagnostics.get('retry_after_sec')
            if type(hint) is int and hint > 0:
                wait = max(wait, min(hint, RETRY_AFTER_CAP))
            # The part of a throttled wait (Retry-After included) that the clock will credit back costs no budget.
            credit = (CLOCK.creditable(wait) if throttle and exc.code == 'http_error'
                      and client.diagnostics.get('http_status') in EVAL_PAUSE_STATUSES else 0)
            left = deadline - CLOCK.now() - (wait - credit)
            # Wait only if a reply can still arrive after it: a full one for R1, the gate for the rest.
            if bounded:
                if left < 10 or _continuation_capacity(left, pace, reserve) is None:
                    raise  # preserve the actual House error; no viable retry allowance remains
            elif left < 10 or (left - 20 < pace.full() + retry_floor if need is None else left < need + reserve):
                raise
            if credit:  # never more than was slept, so the clock never runs ahead of a wait
                began = time.monotonic()
                sleep(wait)
                CLOCK.credit(min(credit, time.monotonic() - began))
            else:
                sleep(wait)
            client.diagnostics['backoff_sec'] = client.diagnostics.get('backoff_sec', 0) + wait
        finally:
            for record in client.diagnostics['requests'][first:]:
                record.update(kind=kind, attempt=attempt, timeout_sec=round(timeout, 1))


def _text(reply):
    return reply['content'] if isinstance(reply['content'], str) else ''


def _count(status, key, amount=1):
    status[key] = status.get(key, 0) + amount


def _fence(code):
    return '```python\n' + code + '```'


_COUNTERS = {'continue_stmt': 'continuations', 'continue_line': 'continuations', 'rewrite': 'rewrites',
             'fill': 'fill_requests', 'salvage': 'salvages'}


def _entered(program):
    """Whether running the program does anything: a __main__ guard, or a top-level statement other than a
    def, class or import that calls something (lambda bodies do not count). Unparsable means no."""
    try:
        body = ast.parse(program).body
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return False
    inert = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom)
    for node in body:
        if is_main_guard(node):
            return True
        if isinstance(node, inert):
            continue
        todo = [node]
        while todo:
            item = todo.pop()
            if isinstance(item, ast.Call):
                return True
            if not isinstance(item, ast.Lambda):
                todo.extend(ast.iter_child_nodes(item))
    return False


def _next_request(kind, finish, triage, progress, program, tail, zero, status):
    """What follows one generation reply: (stop reason or None, next request kind, zero-progress count)."""
    if kind == 'fill' or (finish == 'stop' and not tail.strip() and program.strip()):
        if not _entered(program) and not ensure_guard(program)[1]:
            return 'no_entry', None, zero  # it would only define names and exit 0
        return (None, 'fill', zero) if kind != 'fill' and missing_calls(program) else ('complete', None, zero)
    if finish == 'stop':
        if program.strip() and tail.strip() and not status.get('salvages') and kind != 'fill':
            return None, 'salvage', zero  # once per unit: a garbled statement ends a finished reply
        return 'invalid_code' if program.strip() else 'empty_code', None, zero
    if finish != 'length':
        return 'finish_other', None, zero
    if triage != 'none':  # one rewrite per unit, never an identical resend
        return ('empty_code', None, zero) if status.get('rewrites') else (None, 'rewrite', zero)
    context = function_tail_context(tail)
    kind, zero = ('continue_stmt', 0) if progress else ('continue_line', zero + 1)
    if context:
        kind = 'continue_line'
    if zero >= 2:
        return 'no_progress', None, zero
    if not program.strip() and not context:  # no safe pending body: retain the original rewrite policy
        if not status.get('rewrites'):
            kind = 'rewrite'
        elif '\n' not in tail:
            return 'empty_code', None, zero
    if 1 + status.get('continuations', 0) + status.get('rewrites', 0) >= MAX_REPLIES:
        return 'chunk_cap', None, zero
    return None, kind, zero


def _conversation(kind, messages, program, tail, canaries):
    """Task messages, the verified program as the assistant turn, then what to send next. The
    original prose and reasoning are never resent."""
    if kind == 'rewrite' and not program.strip():
        return messages[:-1] + [{'role': 'user', 'content': messages[-1]['content'] + '\n\n' + REWRITE}]
    last, anchor, line = anchors(program, tail)
    if kind == 'fill':
        turn, ask = _fence(program), FILL.format(names=', '.join(missing_calls(program)[:10]))
    elif kind == 'continue_line':
        turn = '```python\n' + program + ''.join(text + '\n' for text in tail.split('\n')[:-1])
        ask = CONTINUE_LINE.format(line=line)
        context = function_tail_context(tail)
        if context:
            ask = ('Your reply hit the output limit inside an unfinished function. The code above retains '
                   'its complete physical lines, including its unchanged header and body; it is not yet a '
                   'verified complete function. The partial final line was removed. Continue after '
                   f'\x60{line}\x60 without restarting the function or repeating retained lines. '
                   f'The enclosing logical statement starts at {context["block_indent"]} spaces. '
                   'Indentation inside an open delimiter is argument alignment, not the indentation of '
                   'the next statement after that delimiter closes. Keep Python block structure; dedent '
                   'only when leaving a block, never copy argument alignment onto ordinary statements. '
                   'Output only the remaining lines in one \x60\x60\x60python fence, then finish the program entry block.')
    elif kind == 'salvage':
        turn = _fence(program)
        ask = SALVAGE.format(anchor=anchor or tail.strip().split('\n')[0][:80], error=_syntax_error(tail),
                             n=statement_count(program), last=last)
    else:
        missing = missing_calls(program)[:10]
        turn = _fence(program)
        ask = CONTINUE.format(n=statement_count(program), last=last,
                              start=f'the statement that begins `{anchor}`' if anchor else 'the next statement')
        ask += (' Still undefined: ' + ', '.join(missing) + ' - define them.' if missing else '')
        ask += ' ' + REWRITE if kind == 'rewrite' else ''
    return messages + [{'role': 'assistant', 'content': withhold_canaries(turn, canaries)},
                       {'role': 'user', 'content': withhold_canaries(ask, canaries)}]


def _syntax_error(code):
    try:
        compile(code, '<reply>', 'exec', dont_inherit=True)
    except SyntaxError as exc:
        return f'SyntaxError: {exc.msg}'
    except (ValueError, RecursionError, MemoryError, OverflowError):
        pass
    return 'it does not compile'


_PLAN_FENCE = re.compile(r'\s*(`{3,}|~{3,})')


def _without_code(text):
    """(text with each fenced block replaced by PLAN_CODE_CUT and the LIBRARY MAP lines it holds, the number of
    blocks): a map the plan fenced is kept. A fence left open runs to the end, as in a plan cut at the cap inside a
    program it wrote."""
    kept, fence, blocks, body = [], None, 0, []
    for line in text.split('\n'):
        match = _PLAN_FENCE.match(line)
        if fence is not None:
            if match and not line.strip().strip(fence[0]) and len(line.strip()) >= len(fence):
                fence, kept = None, kept + _map_lines('\n'.join(body))
            else:
                body.append(line)
        elif match:
            blocks += 1
            kept.append(PLAN_CODE_CUT)
            rest = line.strip()[len(match.group(1)):]
            fence = None if rest.rstrip().endswith(match.group(1)) else match.group(1)  # ```python x = 1```
            body = []
        else:
            kept.append(line)
    return '\n'.join(kept + (_map_lines('\n'.join(body)) if fence is not None else [])), blocks


def _plan_text(reply, canaries, cut=True, guard=False, status=None):
    """(text, status) from a planning reply: its answer after any reasoning, else the reasoning itself when the
    reply was cut inside it; long text keeps its head and tail (unless cut is False). Canaries are withheld. With
    guard (T1_PLANTHINK), text over PLAN_CHARS first loses its code blocks, so a LIBRARY MAP before a program the
    plan went on to write is not the middle that the cut drops; status['plan_code_blocks_cut'] counts them."""
    text, state = strip_reasoning(_text(reply)).strip(), 'ok'
    if not text:
        text, state = (reply.get('reasoning') or '').strip(), 'reasoning_only'
    if not text:
        return None, 'empty'
    if cut and guard and len(text) > PLAN_CHARS:
        text, blocks = _without_code(text)
        if blocks and status is not None:
            _count(status, 'plan_code_blocks_cut', blocks)
    if cut and len(text) > PLAN_CHARS:
        half = PLAN_CHARS // 2
        text = text[:half] + '\n[...]\n' + text[-half:]
    return withhold_canaries(text, canaries), state


def plan_notes(client, prompt, deadline, status, pace, *, canaries=(), sleep=None, facts=False, full=None,
               thinking=True):
    """One thinking request (low effort; thinking=False, T1_S10's fallback plan, sends it with thinking off) that
    plans the program before it is written. It is sent only while the plan, a
    full program reply and execution all fit; any failure leaves the plain prompt. status['plan_status']
    records the outcome (PLAN_STATUSES). With facts (the self-check on), the plan also lists the facts the
    TASK states that the outputs must satisfy. When the prompt offers the optional library, the plan also maps
    each method the TASK only names to the library function that computes it; under T1_LIBHINT that map is a
    numbered item of fixed lines, one per method, naming a function or none (PLAN_LIBRARY_MAP); under T1_ADOPT a
    mapped function computes its output and a none line quotes the TASK (PLAN_LIBRARY_MAP_ADOPT), and full, a list,
    receives the plan's whole text (before PLAN_CHARS cuts it) for library_map. Under T1_PLANTHINK the plan
    reasons within its budget in place of low effort, a plan over PLAN_CHARS loses its code first (_plan_text), and
    under T1_ADOPT a plan cut at the cap without a LIBRARY MAP line gets those lines from one more request
    (plan_map), appended to full. Under T1_ROUTE the plan ends with its FAMILY and METHOD PINS items (PLAN_ROUTE),
    numbered after the others, which route_prompts reads from full."""
    if deadline - CLOCK.now() < PLAN_TIMEOUT + pace.full() + EXEC_FLOOR:
        status['plan_status'] = 'skipped_time'
        return None
    budget = planthink_budget()
    adopt = adopt_on()
    offered = (LIBRARY_BLOCK_ADOPT if adopt else LIBRARY_BLOCK).split('`')[0] + f'`import {LIBRARY}`' in prompt
    mapped = PLAN_LIBRARY_MAP_ADOPT if adopt else PLAN_LIBRARY_MAP
    ask = (prompt.removesuffix(OUTPUT_FORMAT) + PLAN_ASK + (PLAN_FACTS if facts else '')
           + ((mapped.format(n=7 if facts else 6) if libhint_on() else PLAN_LIBRARY) if offered else ''))
    if route_on():
        n = 6 + bool(facts) + bool(offered and libhint_on())  # after the facts and the numbered library map
        ask += PLAN_ROUTE.format(n=n, m=n + 1, labels='; '.join(label for _, label, _ in families.FAMILIES))
    if budget is not None:
        ask += PLANTHINK_SHORT
    try:
        # A retry must leave the retried plan, a full program reply and execution their time.
        reply = _request(client, [{'role': 'user', 'content': ask}], deadline, pace, 'plan', sleep,
                         cap=PLAN_TIMEOUT, thinking=thinking, retry_floor=PLAN_TIMEOUT + EXEC_FLOOR, budget=budget)
    except (HouseError, _Stop):
        status['plan_status'] = 'failed'
        return None
    pace.measure(client, reply)
    plan, status['plan_status'] = _plan_text(reply, canaries, guard=budget is not None, status=status)
    status['plan_chars'] = len(plan) if plan else 0
    if plan and full is not None:
        full.append(_plan_text(reply, canaries, cut=False)[0])
        if (budget is not None and adopt and offered and reply['finish_reason_raw'] == 'length'
                and not _map_lines(full[0])):
            lines = plan_map(client, ask, full[0], deadline, status, pace, canaries=canaries, sleep=sleep)
            # Under T1_ROUTE the recovered lines get their item's name, so method_pins never reads them as pins.
            full[0] += '\nLIBRARY MAP' + lines if lines and route_on() else lines
    return plan


_DEF_HEAD = re.compile(r'\s*(?:async\s+)?def\s+\w+\s*\(')  # def f(x) -> None: is a signature, not a map line


def _map_lines(text):
    """The lines of text that library_map reads as LIBRARY MAP lines: `<method> -> qfkit.<function>...` or
    `<method> -> none...`; a Python signature with a return annotation is not one."""
    found = []
    for raw in (text or '').splitlines():
        parts = _map_split(raw.replace('`', '').replace('*', ''))
        target = parts[-1].strip()
        if (len(parts) == 2 and _map_method(_MAP_BULLET.sub('', parts[0])) and not _DEF_HEAD.match(parts[0])
                and (re.match(rf'{LIBRARY}\.[A-Za-z_]', target) or re.match(r'none\b', target, re.I))):
            found.append(raw.strip())
    return found


def plan_map(client, ask, plan, deadline, status, pace, *, canaries=(), sleep=None, need=None, reserve=None):
    """T1_PLANTHINK under T1_ADOPT, for a plan cut at the cap before any LIBRARY MAP line: one thinking-off request
    (plan_map) for those lines alone, sent only while the request cap still leaves the program its request and
    the clock its full reply and execution (T1_S10's DECIDE passes its own gate as need and reserve: the answer's
    cost and R_code). The map lines it gives, each on a new line ('' for none); status['plan_map'] records the
    outcome (PLAN_MAP_STATUSES)."""
    conversation = [{'role': 'user', 'content': ask}, {'role': 'assistant', 'content': plan},
                    {'role': 'user', 'content': PLAN_MAP_ASK}]
    try:  # keep=1: the program request after it
        reply = _request(client, conversation, deadline, pace, 'plan_map', sleep,
                         need=pace.patch() if need is None else need,
                         reserve=pace.full() + EXEC_FLOOR if reserve is None else reserve, keep=1)
    except _Stop:
        status['plan_map'] = 'skipped'
        return ''
    except HouseError:
        status['plan_map'] = 'failed'
        return ''
    lines = _map_lines(withhold_canaries(strip_reasoning(_text(reply)), canaries))
    status['plan_map'] = 'recovered' if lines else 'no_map'
    return ''.join('\n' + line for line in lines)


def with_plan(prompt, plan):
    """The program prompt with the plan between the task and the output format."""
    return prompt.removesuffix(OUTPUT_FORMAT) + PLAN_BLOCK.format(plan=plan) + OUTPUT_FORMAT


def _plan_tokens(record):
    """A plan reply's completion tokens, less the reasoning's share by characters when it reasoned within a budget
    (T1_PLANTHINK): R1THINK_PLAN_MAX measures low-effort plans, whose reasoning is a few dozen tokens. All of
    them when the reply has no answer text."""
    tokens, answer, reasoning = (record.get(key) for key in ('completion_tokens', 'content_chars', 'reasoning_chars'))
    if 'reasoning_budget' not in record or type(answer) is not int or type(reasoning) is not int or answer <= 0:
        return tokens
    return round(tokens * answer / (answer + max(reasoning, 0)))


def r1think(client, status, deadline, pace):
    """T1_R1THINK=1: the reasoning budget of the first program request (r1think_budget), or None. It is given only
    when the plan reply took at most T1_R1THINK_PLAN_MAX completion tokens (R1THINK_PLAN_MAX; a long plan predicts a
    long program, which the reasoning would push past the cap, and without a plan reply the gate is closed; a plan
    that reasoned within a budget counts its answer alone, _plan_tokens) and the clock still leaves a full reply,
    the reasoning, execution and THINK_MARGIN (think_fits). status['r1think'] records the outcome
    (R1THINK_STATUSES). With T1_PLANTHINK on, PLANTHINK_SHORT holds every written plan near 350 words, so the
    length gate stays open in practice and the budget rests on the clock alone; plan length was a weak predictor of a
    cut program anyway (codereq study: AUC 0.41), and a cut program still goes to continuation."""
    budget = r1think_budget()
    if budget is None:
        return None
    tokens = [_plan_tokens(record) for record in client.diagnostics['requests']
              if record.get('kind') == 'plan' and type(record.get('completion_tokens')) is int]
    limit = int(os.environ['T1_R1THINK_PLAN_MAX']) if re.fullmatch(
        r'[1-9]\d{0,4}', os.environ.get('T1_R1THINK_PLAN_MAX', '')) else R1THINK_PLAN_MAX
    reason = ('skipped_no_plan' if not tokens else 'skipped_plan_long' if tokens[-1] > limit
              else None if think_fits(deadline, pace, budget) else 'skipped_time')
    status['r1think'] = reason or 'used'
    return None if reason else budget


def route_on() -> bool:
    """T1_ROUTE=1 (off in the image): the plan also names the task's family from a fixed list and quotes the TASK
    sentence that pins each graded method, or says free (PLAN_ROUTE); a family it names adds that family's note to
    the program and fresh-program prompts and its METHOD PINS lines to the program prompt (route_prompts). Off, or
    with no family named, requests and programs are as before."""
    return os.environ.get('T1_ROUTE') == '1' and not minimal_on()


# Bullets, an item number and markdown before a plan item's name: `(8) FAMILY:`, `- **FAMILY:**`, `9. Method pins -`.
_ROUTE_LEAD = r'^[ \t>#*_`+\-\u2022]*(?:\(?\d{1,2}[.):][ \t]*)?[ \t*_`]*'
_ROUTE_SEPARATOR = r'[ \t*_`]*(?::|=|[-\u2013\u2014](?=[ \t]))'  # not the hyphen of family-level
_ROUTE_FAMILY = re.compile(_ROUTE_LEAD + r'family\b(?:' + _ROUTE_SEPARATOR + r'(.*)|[ \t*_`]*)$', re.I)
_ROUTE_PINS = re.compile(_ROUTE_LEAD + r'method[ \t*_`-]*pins?\b(?:' + _ROUTE_SEPARATOR + r')?(.*)$', re.I)
_ROUTE_SECTION = re.compile(_ROUTE_LEAD + r'(?:family|library[ \t]+map|checkable[ \t]+facts|method[ \t*_`-]*pins?)\b',
                            re.I)
_ROUTE_ITEM = re.compile(r'^[ \t]*(?:[-*+\u2022]|\(?\d{1,2}[.)])[ \t]')
_ROUTE_BULLET = re.compile(r'^[ \t>*+\-\u2022]*(?:\(?\d{1,2}[.)][ \t]+)?')
_ROUTE_QUOTE = re.compile(r'["\u201c]')


def _route_words(text):
    return ' '.join(re.findall(r'[a-z0-9]+', text.lower()))


_ROUTE_LABELS = tuple((slug, re.compile(rf'\b{re.escape(_route_words(label))}\b'))
                      for slug, label, synonyms in families.FAMILIES if synonyms)
_ROUTE_SYNONYMS = tuple((slug, re.compile(rf'\b(?:{"|".join(synonyms)})\b'))
                        for slug, _, synonyms in families.FAMILIES if synonyms)
# A label without synonyms (other) counts only when no family is named: `rates (other: FX)` is rates.
_ROUTE_REST = tuple((slug, re.compile(rf'\b{re.escape(_route_words(label))}\b'))
                    for slug, label, synonyms in families.FAMILIES if not synonyms)


def route_slug(text):
    """The family a FAMILY value names: the label that occurs earliest in its words (case, punctuation and markdown
    ignored), else the family of the earliest synonym (families.FAMILIES), else other; None for any other text and
    for a value naming ROUTE_LIST_LABELS labels or more, which copies the ask's list (route_family reads on)."""
    words = _route_words(text)
    for patterns in (_ROUTE_LABELS, _ROUTE_SYNONYMS, _ROUTE_REST):
        found = [(match.start(), index, slug) for index, (slug, pattern) in enumerate(patterns)
                 for match in [pattern.search(words)] if match]
        if patterns is _ROUTE_LABELS and len(found) >= ROUTE_LIST_LABELS:
            return None
        if found:
            return min(found)[2]
    return None


def route_family(plan):
    """The family of the first `FAMILY: <label>` line of a plan that names one (route_slug), in any case, but the
    lines that spell FAMILY as the ask does first (a column described as `family: ...` comes second); under a FAMILY
    heading with no label of its own, the label is read from the next line. None without one."""
    lines = (plan or '').splitlines()
    found = [(index, match) for index, line in enumerate(lines) for match in [_ROUTE_FAMILY.match(line)] if match]
    for index, match in sorted(found, key=lambda item: ('FAMILY' not in lines[item[0]], item[0])):
        value = match.group(1) or ''
        if not value.strip(' \t*_`'):
            if 'FAMILY' not in lines[index]:
                continue
            value = next((later for later in lines[index + 1:] if later.strip()), '')
        slug = route_slug(value)
        if slug is not None:
            return slug
    return None


def _route_pin(line):
    return ' '.join(_ROUTE_BULLET.sub('', line).replace('**', '').split())


def method_pins(plan):
    """The lines of a plan's METHOD PINS item, bullets and bold dropped: text after its name on the same line, then
    the lines below it up to a blank line that no further list item follows, or another item of the plan ask
    (FAMILY, LIBRARY MAP, CHECKABLE FACTS); at most ROUTE_PIN_LINES. [] without the item."""
    lines = (plan or '').splitlines()
    start = next((index for index, line in enumerate(lines) if _ROUTE_PINS.match(line)), None)
    if start is None:
        return []
    pins = []
    inline = _route_pin(_ROUTE_PINS.match(lines[start]).group(1))
    if inline and not inline.endswith(':'):  # `(one per line):` introduces the lines; it is not one
        pins.append(inline)
    rest = lines[start + 1:]
    for index, line in enumerate(rest):
        if len(pins) >= ROUTE_PIN_LINES or _ROUTE_SECTION.match(line):
            break
        if not line.strip():
            following = next((later for later in rest[index + 1:] if later.strip()), '')
            if pins and not (_ROUTE_ITEM.match(following) and not _ROUTE_SECTION.match(following)):
                break
            continue
        pin = _route_pin(line)
        if pin:
            pins.append(pin)
    return pins


def free_pin(pin):
    """A METHOD PINS line that leaves the method free: it says free and quotes no TASK sentence."""
    return bool(re.search(r'\bfree\b', pin, re.I)) and not _ROUTE_QUOTE.search(pin)


def quoted_pin(pin, instruction):
    """A METHOD PINS line whose quote is a TASK sentence (audit._quoted: verbatim, or most of it), not a paraphrase."""
    return any(_quoted(a or b, instruction) for a, b in _MAP_QUOTE.findall(pin))


def route_note(slug, functions) -> str:
    """The family's note (families.NOTES): its first line, the library functions among functions (the prompt's
    catalog), the items whose functions are all among them, and the common rules."""
    library, items = families.NOTES[slug]
    names = [name for name in library if name in functions]
    lines = [families.NOTE_HEAD] + ([families.LIBRARY_LINE.format(names=', '.join(names))] if names else [])
    lines += ['- ' + text for text, needs in items if set(needs) <= set(functions)]
    return '\n'.join(lines + [families.COMMON])


def _pins_text(pins):
    """The METHOD PINS lines as a list, whole lines within ROUTE_PINS_CAP characters (a first line longer than the
    cap is cut)."""
    lines, used = [], 0
    for pin in pins:
        line = '- ' + pin
        if used + len(line) > ROUTE_PINS_CAP:
            if not lines:
                lines.append(line[:ROUTE_PINS_CAP - 3] + '...')
            break
        lines.append(line)
        used += len(line) + 1
    return '\n'.join(lines)


def with_route(prompt, block):
    """The prompt with a route block before the output format; the prompt as it is without one."""
    return prompt.removesuffix(OUTPUT_FORMAT) + block + OUTPUT_FORMAT if block else prompt


def route_prompts(prompt, plain, plan, status, instruction=''):
    """(program prompt, fresh-program prompt) under T1_ROUTE. The family the plan names (route_family) adds its note
    (ROUTE_NOTES; route_note with the catalog's functions when the prompt shows the library, none otherwise) to both
    prompts, after the library sections and REQUIRED LIBRARY CALLS, and the plan's METHOD PINS that quote the TASK
    or say free (ROUTE_PINS, whole lines within ROUTE_PINS_CAP) to the program prompt after it, before the output
    format; the fresh program stays
    free of the plan's own text. A block goes in only while the task prompt with it stays under FIRST_PROMPT_CAP,
    the pins first to go: first_prompt's own trimming is never redone for them. No family, the prompts as they are.
    status: route_family (a slug, or none), route_note, route_pins (lines read), route_pins_free, route_pins_quoted
    (lines whose quote the instruction holds: quoted_pin), route_trimmed."""
    slug, pins = route_family(plan), method_pins(plan)
    status.update(route_family=slug or 'none', route_note=False, route_pins=len(pins),
                  route_pins_free=sum(map(free_pin, pins)),
                  route_pins_quoted=sum(quoted_pin(pin, instruction) for pin in pins), route_trimmed=False)
    if slug is None or not library_allowed_by_task(instruction):
        return prompt, plain
    functions = library_functions() if status.get('library_shown') else set()
    notes = ROUTE_NOTES.format(label=families.LABELS[slug], note=route_note(slug, functions))
    # Only lines that quote a TASK sentence, or leave the method free, are repeated: a paraphrase may change it.
    shown = _pins_text([pin for pin in pins if free_pin(pin) or quoted_pin(pin, instruction)])
    full = notes + (ROUTE_PINS.format(pins=shown) if shown else '')
    for block in (full, notes):
        if len(with_route(plain, block).encode()) < FIRST_PROMPT_CAP:
            status.update(route_note=True, route_trimmed=block != full)
            return with_route(prompt, block), with_route(plain, notes)
    status['route_trimmed'] = True
    return prompt, plain


# Stack 10 (T1_S10, off in the image; spec stack10/spec.md): reasoning moves out of the code reply into short note
# requests that each reason within a fixed budget and write only numbered lines (DECIDE, then CHECK and STATE when
# needed), which the code request implements. A no-request triage sets the word caps and budgets. Every budgeted note
# request gates itself on the clock (_s10_gate) and is skipped, never sent unreasoned (_request strict), and a
# unit-wide breaker turns thinking off once a reply shows its budget was not honoured (_s10_breaker).
S10_TOKEN_CEILING = 3700  # the 4,000-token reply less 300 for reasoning that runs slightly past its budget
S10_TPW = 5.5  # worst answer tokens per asked word: decision lists overshoot their word caps 1.3-1.5x
S10_EPW = 4.0  # expected answer tokens per asked word
S10_FLOOR = 1200  # a note step's budget shrinks to fit the clock down to this, then the step is skipped
S10_MARGIN = 30  # seconds every S10 gate keeps free (think_fits' margin)
S10_PRIOR_RATE = 60  # tok/s the gates assume at most until a reply is measured (r11 ran at 52)
S10_KEEP = 4  # requests a note request leaves: the code request, a continuation, a patch and its retry
S10_CHECK_USED = 0.85  # DECIDE reasoning at this share of its budget triggers CHECK (T1_S10_CHECK=auto)
# v1.1: CHECK is off unless T1_S10_CHECK says auto or always (in the probe it fixed no item and broke none, for a
# request and about 30 s).
S10_CHECK_MODES, S10_CHECK_DEFAULT = ('auto', 'always', 'never'), 'never'
S10_OVERRUN, S10_OVERRUN_SLACK = 1.25, 150  # the breaker trips at reasoning above 1.25 x the budget + 150 tokens
# The breaker's placement-independent trip (the official endpoint may show no reasoning text at all): a reply's token
# count past its budget, its answer's worst size (S10_TPW tokens per word; S10_ANSWER_CPT characters per token on
# dense code) and this slack, the 300-token margin of S10_TOKEN_CEILING.
S10_COMPLETION_SLACK = 300
# Note requests (DECIDE, CHECK, STATE, the distills): a timeout of at least 4,000 tokens at 40 tok/s + 10 s where the
# clock allows it (the probe's one-sample timeouts killed 3 of about 29 replies after DECIDE), and at most one
# transient retry on a strict step (each is a counted request).
S10_NOTE_TIMEOUT_FLOOR = 110
S10_NOTE_RETRIES = 1
S10_DISTILL_TAIL = 6000  # characters of a DECIDE's reasoning pasted back once when it gave no answer (DISTILL)
S10_DISTILL_MIN = 1000  # reasoning characters below which such a DECIDE is resent with thinking off instead
S10_SPILL_MIN = 200  # v1.1: characters of prose, outside every note item, that make a note reply spill-over
S10_REASONING_CPT = 3.7  # characters per reasoning token
S10_ANSWER_CPT = 3.0  # characters per answer token: low on code, so a long code reply never trips the breaker
S10_TIER_L_CHARS, S10_TIER_L_FILES = 7000, 9  # an L-tier TASK: this long, or naming this many files
S10_TIER_F_CHARS, S10_TIER_F_FILES = 4500, 4  # an F-tier TASK: shorter, fewer files, not stateful, no template
S10_STATE_HITS = 8  # loop words that make a unit stateful without asking (14 of the 64 dev units)
S10_FILE = re.compile(r'(?:/app/output/|output/|`)([\w\-./]+\.(?:csv|json|txt|parquet|png|md|xlsx))', re.I)
S10_STATEFUL = re.compile(r'\b(backtest\w*|ledger|rebalanc\w*|trailing|stop[- ]loss|position\w*|hedg\w*|iterat\w*|'
                          r'EM\b|Baum|expectation[- ]maximi\w*|until convergence|converge\w*|'
                          r'each (?:day|step|date|period)|daily loop|event loop|order\w*|fill\w*|carry forward)', re.I)
S10_TIERS = ('F', 'D', 'L')
# v1.1: an L-tier DECIDE reasons 1,600 tokens and writes at most 12 D lines (the probe's wrote about 27 lines, 900
# words, and the reasoning with them reached the cap before the deciding items late in the list).
S10_DECIDE_BUDGET = {'F': 2600, 'D': 2300, 'L': 1600}
S10_DECIDE_WORDS = {'F': 200, 'D': 250, 'L': 250}
S10_CODE_BUDGET = {'F': 1000, 'D': 700, 'L': 0}  # an L-tier program (mean first reply 3.6k tokens) gets none
S10_CHECK_BUDGET, S10_CHECK_WORDS = 2600, 150
S10_STATE_BUDGET, S10_STATE_WORDS = 1800, 200  # v1.1: from 2,400; the probe's STATE replies spilled over
S10_DISTILL_WORDS = 250
S10_PATCH_BUDGET = 1200  # a repair round's first patch (cliquet's fix came about 0.4k tokens into its reasoning)
S10_FLAGFIX_BUDGET = 500
S10_D_MIN, S10_S_MIN = 2, 3  # D lines a DECIDE needs to stand; S lines a STATE needs
S10_D_LINES, S10_D_CHARS = 16, 400
S10_STATE_LINES, S10_STATE_CHARS, S10_STATE_TOTAL = 25, 300, 2500
S10_OUTPUTS_CHARS = 600
S10_CHECKED_CHARS = 1500
S10_BODY_MIN = 200  # characters, code aside, that a DECIDE text without a complete D line needs to stand (no_lines)
S10_DECIDE_STATUSES = ('ok', 'cut_kept', 'no_lines', 'distilled', 'retried', 'failed', 'fallback_plan')
S10_CHECK_STATUSES = ('ok', 'cut_kept', 'no_lines', 'failed', 'skipped_time', 'skipped_requests', 'skipped_breaker',
                      'not_triggered', 'off')
S10_TRIGGERS = ('cut', 'open', 'unsure', 'free', 'budget', 'always', 'none')
S10_STATE_STATUSES = ('ok', 'cut_kept', 'no_lines', 'failed', 'skipped_time', 'skipped_requests', 'not_stateful',
                      'off')
S10_STATEFUL_FROM = ('hits', 'reply', 'both', 'none')
# v1.1: s10_decide_spill, s10_check_spill and s10_state_spill, the spill-over DISTILL's outcome for each step's reply.
S10_SPILL_STATUSES = ('none', 'used', 'fewer', 'no_lines', 'failed', 'skipped_time', 'skipped_requests')
S10_BREAKER = ('none', 'overrun', 'reasoning_only', 'overrun_tokens')
S10_KINDS = ('s10_decide', 's10_distill', 's10_decide_retry', 's10_check', 's10_state')
# v1.1: DECIDE, CHECK and STATE ask for the reply as soon as the model has decided (the probe's replies often went
# on reasoning in the answer once the budget ran out).
S10_PROMPTLY = ('Write your reply as soon as you have decided; where you cannot settle an item, end its line with '
                'UNSURE rather than keep reasoning in the reply.')
S10_DECIDE = ('DECIDE FIRST: do not write the program yet. Your reasoning is not passed on; only your written reply '
              'is, and the program will be written from it, line by line. Work out how to compute every required '
              'output exactly as the TASK specifies. ' + S10_PROMPTLY + ' Then reply in plain text, without code, in '
              'exactly this order and at most {words} words in all, LIBRARY MAP lines included:\n'
              'STATE: yes if the outputs come from a loop that carries values from one step to the next (positions, '
              'cash, pending orders, running values, or an estimate iterated until it converges); otherwise no.\n'
              '{family}'
              'OUTPUTS: the path of every output file the TASK requires, separated by commas.\n'
              'DECISIONS: one line per decision, those where a plausible choice would contradict the TASK first, each '
              'exactly\n'
              'D<n> | <quantity or step> | <the decision as the program must apply it: the formula with its '
              'variables, its sign, units and scale; the convention; the value of a setting; or the library call with '
              'what each argument means> | "<the TASK words it follows, at most 12, copied verbatim>" or free\n'
              'Cover: each formula the outputs need that the TASK names without writing out; the sign of every cash '
              'flow, position and P&L; which of several similar objects, inputs or variants each output uses; units '
              'and scale (per unit, per basis point or per percent; annualised or not; percent or fraction); each '
              'convention (annualisation, day count, ddof, compounding, discounting, timing and alignment of dates '
              'and windows, inclusive or exclusive bounds, counting from 0 or 1); each numerical setting (step size, '
              'grid, tolerance, stopping rule, seed and the order of random draws); how each input is read and '
              'aligned; and each rule for missing or invalid data (the TASK\'s rule, or else raise an error). Write '
              'free only where the TASK leaves the choice open. End a line with UNSURE where you could not verify '
              'it. {limit}\n'
              '{template}{library}'
              'OPEN: each point you could not settle, separated by `;`, or none.')
# v1.1: the tier's limit on D lines and quotes (the probe's L-tier replies wrote about 27 lines).
S10_DECIDE_LIMIT = {'F': 'Quote at most 12 TASK words in a line.',
                    'D': 'Write at most 16 D lines. Quote at most 12 TASK words in a line.',
                    'L': ('Write at most 12 D lines, the error-prone decisions first (conventions, signs, formulas, '
                          'library arguments, bookkeeping), and skip routine input and output handling that the '
                          'TASK spells out. Quote at most 12 TASK words in a line.')}
S10_FAMILY = ('FAMILY: exactly one label from this list, the one that fits the computation of the graded outputs: '
              '{labels}.\n')
S10_TEMPLATE = ('The TASK gives code to fix or reuse: add one D line per defect it needs fixed, as `D<n> | <function> '
                '| <the current code> -> <the corrected code> | "<TASK words>"`.\n')
# stack9's map items, so that library_map and _map_lines read DECIDE's LIBRARY MAP lines unchanged.
S10_LIBRARY_MAP_ADOPT = ('LIBRARY MAP:' + PLAN_LIBRARY_MAP_ADOPT.split('LIBRARY MAP,', 1)[1]
                         + ' Settle what each argument you pass means in a D line.')
S10_LIBRARY_MAP = ('LIBRARY MAP:' + PLAN_LIBRARY_MAP.split('LIBRARY MAP,', 1)[1]
                   + ' Settle what each argument you pass means in a D line.')
S10_DISTILL = ('YOUR NOTES FROM AN EARLIER ATTEMPT (your own reasoning, cut at the output limit before you wrote the '
               'reply; unchecked; they may include choices you later rejected):\n{tail}\n\n'
               'Reply now with only the parts asked above, in the same format and order, at most {words} words, taking '
               'what the notes conclude last.')
# v1.1: the spill-over DISTILL (s10_spill): a DECIDE, CHECK or STATE reply that went on reasoning in its visible text
# once the budget ran out (the phase-1 probe's had dozens to 140 lines of prose and no ID line), or wrote a program
# instead of its lines, is sent back once, thinking off, after the step's own ask, for its lines alone.
S10_SPILL = ('YOUR REPLY TO THE ASK ABOVE (your own visible reply; it gave reasoning or code instead of only the '
             'lines asked; unchecked; it may include choices you later rejected):\n{tail}\n\n'
             'Reply now with only the lines the ask above asks for, in its format and order, at most {words} words, '
             'taking what the reply concludes last. Write no reasoning and no code; where the reply leaves an item '
             'unsettled, end its line with UNSURE.')
S10_CHECK = ('YOUR DECISIONS SO FAR (written by you in an earlier step; your reasoning then is not shown):\n{notes}\n\n'
             '{family}'
             'CHECK: do not write the program yet. Your reasoning is not passed on; only your written reply is. Check '
             'the decisions above against the TASK, starting with the lines marked UNSURE or free and the OPEN points. '
             'For each, derive it again from the TASK\'s own words; check the sign from the TASK\'s point of view, the '
             'units and scale, the exact object or variant each output uses, where counting starts, which quantities '
             'are discounted, and what each library argument means; and test each formula on a small or limiting case '
             'whose answer you know. ' + S10_PROMPTLY + ' Reply in plain text, without code, at most {words} words: '
             'one line in the same `D<n> | ...` format for each decision you change (keep its number) or add (the '
             'next free number), then a last line `OTHERS CONFIRMED`. Do not repeat a decision that stays as it is.')
S10_STATE = ('YOUR DECISIONS (written by you in earlier steps; your reasoning then is not shown):\n{notes}\n\n'
             'STATE SPEC: do not write the program yet. Your reasoning is not passed on; only your written reply is, '
             'and the program will follow it. The outputs come from a loop that carries values from one step to the '
             'next. ' + S10_PROMPTLY + ' Write how the program runs that loop, in plain text, without code, at most '
             '{words} words, in exactly this order:\n'
             'INIT: the starting value of every carried variable (cash, positions, pending orders, counters and the '
             'first number or id, running values, the first estimate).\n'
             'S1, S2, ...: one line per operation, in the order the program performs them within one step: what is '
             'known at that point and what only later; signals; when an order or decision made now takes effect and '
             'at which price or value; cash and position updates; exits and the reason recorded; counters and '
             'running values.\n'
             'RECORD: for each stored record or output row, each field, the S line that sets it, and from which '
             'variable.\n'
             'LABELS: numbering (from 0 or 1) and labels exactly as the TASK writes them; an internal index is not a '
             'label.\n'
             'END: what happens at the last step or at convergence, and exactly which quantity the stopping test '
             'compares.\n'
             'Follow the TASK and the decisions above; where they differ, the TASK is right.')
S10_NOTES = ('YOUR DECISIONS (written by you in earlier steps, whose reasoning is not shown; implement each exactly as '
             'written; where one differs from the TASK above, the TASK is right):\n{body}\n\n')
S10_NOTES_STATE = 'STATE SPEC (follow this order within each step of the loop):\n{lines}'
S10_CHECKED = 'CHECKED (later; where it differs from a line above, it replaces it):\n{text}'
S10_TAGS = 'End the line of code that implements each decision with a comment naming it, such as `# D3`.'
S10_PATCH_KEEP = ('Change only what this error needs. Keep each of YOUR DECISIONS unless the error shows that the '
                  'decision itself is wrong (for example, a library function rejects the value it chose for an '
                  'argument); then change that decision in the code and mark the line `# D<n> changed`.')
# T1_S11_CONV (off unless set; acts only under T1_S10): the deep dive's fix 1. A near miss fails one check on one
# convention the TASK leaves unstated (a count, a window, discounting, a sign), and thinking longer does not find it
# (r11 at 16k tokens repeated stack7's choices); a named default does, if DECIDE has to write it down, since prose
# notes were ignored in exactly the units they targeted. DECIDE writes each listed item that applies as a D line
# ending in the TASK words that settle it, or in `default` where the TASK is silent, and the code copies it.
# F and D keep stack 10's reasoning budgets and word caps: in r12 and r13b, 10 of the 17 passing F and D DECIDEs
# reasoned past the budgets the token rule would leave under larger caps, and F holds 5 of the 11 reliable units and
# none of the fix's targets; their defaults fold into the lines that cover them. L, the one tier with slack
# (1,600 + 5.5 x 250 = 2,975 of S10_TOKEN_CEILING), gains 50 words and 3 lines with its budget unchanged.
S11_CONV_DECIDE_WORDS = {'F': 200, 'D': 250, 'L': 300}
S11_CONV_D_LINES = 20  # D lines a parse keeps (S10_D_LINES): replies overran their asked limits in r12 and r13b
# D lines ending in default that a parse keeps past S11_CONV_D_LINES (17 of 43 DECIDE parses hit the cap)
S11_CONV_D_EXTRA = 6
S11_CONV_LIMIT = dict(S10_DECIDE_LIMIT, L=('Write at most 15 D lines, the error-prone decisions first (conventions, '
                                           'signs, formulas, library arguments, bookkeeping), and skip routine input '
                                           'and output handling that the TASK spells out. Quote at most 12 TASK '
                                           'words in a line.'))
# The D line's last field may say default (DECIDE's format line, S10_FIELD, gains it under the switch).
S10_FIELD = '"<the TASK words it follows, at most 12, copied verbatim>" or free'
S11_CONV_FIELD = '"<the TASK words it follows, at most 12, copied verbatim>", default or free'
# New default lines go before routine lines: a cut at the output cap (14 of 43 DECIDEs) or the line cap drops the last.
S11_CONV_DEFAULTS = (
    'DEFAULTS, where the TASK is silent: for each item below that applies to this TASK, one D line states the '
    'decision and ends with the TASK words that settle it, or with default where the TASK is silent and the program '
    'follows the default. Use the line that already covers that quantity; where none does, put the new line right '
    'after the decisions a plausible choice would get wrong, ahead of routine input and output handling. The TASK '
    'always wins; skip an item that does not apply.\n'
    '(a) Counts: a count key holds what its own name counts (a key named for prices counts prices, one named for '
    'returns counts returns). A count of valid rows is taken after cleaning the whole input, before any window.\n'
    '(b) Names: a key or label the TASK leaves unnamed uses the TASK\'s own words for that thing.\n'
    '(c) Units: a unit suffix in a key (_days, _bps, _pct, _periods) sets its unit; a word keeps one meaning in all '
    'of the TASK\'s keys.\n'
    '(d) Windows: an estimation window or lookback with no other rule is the trailing N rows ending at the last date.\n'
    '(e) Discounting and draws: a payoff is undiscounted, and the price, value or pv of that payoff is discounted to '
    'today; Greeks of a seeded Monte Carlo use common random numbers (the same draws in every revaluation); a sample '
    'standard deviation uses ddof=1.\n'
    '(f) Signs: a duration-type sensitivity is positive for a long position; a cash flow\'s sign follows who pays it '
    '(a short option receives its premium).\n'
    '(g) Precedence: a unit or scale stated in an output section overrides an earlier formula; a pipeline counter is '
    'taken after every step listed before it; a date-only closed end includes that whole day.\n'
    '(h) Variants: where the TASK defines several variants of a quantity or model, the line of each output that uses '
    'one names it, with the TASK words that choose it.')
# Hygiene (deep dive §3 runners-up): a STATE gate that fails while more than this share of the clock DECIDE began
# with remains no longer skips STATE (in r12 a stateful unit was skipped with about 290 s left, its gate holding R_code
# and a margin back after STATE at a 37 tok/s DECIDE rate, and its program then came in 14 s). The rescued STATE and
# its spill-over DISTILL keep one full code reply and one run free (_s11_rescue_reserve; a request's timeout keeps its
# reserve however long the reply runs): the rescue gives up R_code's patch and second run, never the program.
S11_CONV_HALF = 0.5
S11_CONV_RESCUES = ('none', 'floor', 'think_off')
# T1_S11_RAISE (deep dive fix 3; only under T1_S10): "raise an error ... instead of silently filling a default" made
# programs invent preconditions (32 of 132 dev crash dead ends died on the program's own raise, and the patch loop
# never removed one). The task prompt raises only for a missing input, turns the TASK's validate/ensure/clean lines
# into filters that drop and count, keeps RESULT_CHECKS, and, where the TASK is silent about missing rows, builds
# time grids from its parameters (an order day after the last bar); DECIDE asks the same; a crash on the program's
# own check asks whether the TASK states it (a raise in an except clause stands in for a library's error: no note).
S11_RAISE_INPUT = ('Raise an error only when a required input file, column or key is missing, or where the TASK says '
                   'to raise. A TASK line that says to validate, ensure, check, handle or clean data (input rows, '
                   'quotes, points to fit) is a filter: drop what fails it, count what you drop and print the count; '
                   'never raise for it. Where the TASK is silent about missing or invalid values, drop and count those '
                   'rows the same way instead of silently filling a default; where it states a rule, follow the TASK. ')
S11_RAISE_GRID = ('A date, session or bar that the TASK or an input names may lie outside the data\'s range, which the '
                  'INPUT PROFILE shows for each date column. Where the TASK defines a time grid by parameters (start, '
                  'end, step, bar or bin size, session times) and is silent about rows missing from the data, build '
                  'the grid from those parameters, not from the timestamps present; where the TASK says to use the '
                  'rows or observations present, follow the TASK. Where the TASK is silent, look data up only on dates '
                  'the data has (such as the last row of each period), never on a calendar date built by hand. ')
S11_DECIDE_FROM = ('aligned; and each rule for missing or invalid data (the TASK\'s rule, or else raise an error). ')
S11_DECIDE_TO = ('aligned; each rule for missing or invalid data (the TASK\'s rule; where the TASK is silent, drop and '
                 'count the bad rows, and raise only for a missing input file, column or key); and each time grid '
                 'the TASK defines (the TASK\'s rule; where the TASK is silent about rows missing from the data, '
                 'built from its parameters rather than from the dates present). ')
S11_DECIDE = S10_DECIDE.replace(S11_DECIDE_FROM, S11_DECIDE_TO)
S11_OWN_RAISE = ('This error was raised by your program\'s own check{where}, not by Python or a library. If the '
                 'check guards a numerical step the TASK needs (a root, a bracket, a fit or an iteration that must '
                 'converge), make that step succeed. Otherwise, does the TASK state this requirement? If it does '
                 'not, delete the check instead of satisfying it, and make the code after it handle that case as the '
                 'TASK describes (drop and count the rows that fail; where the TASK is silent about rows missing '
                 'from the data, build a time grid from its parameters; print a warning about a computed result); a '
                 'decision that asked for the check is then wrong too. If the TASK states it, or a required input '
                 'file, column or key is really missing, fix the cause.\n')
S11_OWN_AGAIN = ('Your previous change{names} did not fix this: your program\'s own check raised the same {exc} '
                 'again{error}.\n')
# T1_S11_LINT: a crash ask whose run the operating system killed (exit -9, or 137 through a shell) and whose log
# shows no traceback needs evidence before attributing the signal to memory.
S11_OOM = ('The program was killed without an error message; the cause is unknown. Signal 9 alone does not '
           'prove an out-of-memory event. Inspect available execution/resource evidence and all large array '
           'allocations and callers. If memory is responsible, use bounded batches or equivalent sufficient '
           'state, preserving the required sample count, time grid, precision and random draw order.\n')
S11_OOM_CODES = (-9, 137)
S11_TRACEBACK = 'Traceback (most recent call last)'
S11_OUTPUT_ITEMS = 6  # output lint flags listed in the flags-only round (code lints: LINT_LIMIT)
# T1_S11_UNION (off unless set; acts only under T1_S10; deep dive §4): after DECIDE (A) one more budgeted DECIDE (B,
# s11_decide_b) whose ask is A's with the D lines asked in a reviewer's order, output by output (S11_ORDER_B for
# S11_ORDER_A), so that even a greedy backend answers it apart. A's lines all stand; s11_union drops a B line A
# covers, appends a new one, and marks A's line UNSURE with B's reading where the two decide one quoted TASK sentence
# differently. On the frozen checklist one DECIDE sample scored 0.55 and the union of two 0.65 (r11's level).
S11_ORDER_A = 'those where a plausible choice would contradict the TASK first, each exactly'
S11_ORDER_B = ('in the order a reviewer checks the program against the TASK (output by output: first what each '
               'required output holds, its keys or columns with their units, sign and count, then the formula, '
               'convention, setting and input rule behind it), each exactly')
# The L tier's limit puts the error-prone decisions first and skips routine output handling: its order goes output by
# output through those decisions alone, never through the keys and columns.
S11_ORDER_B_L = ('in the order a reviewer checks the program against the TASK (output by output: the error-prone '
                 'decision behind each required output, its formula, convention, sign, units or library argument), '
                 'each exactly')
S11_UNSURE = (' UNSURE (two readings: as written, or {reading}; take the one the TASK\'s words support, else as '
              'written)')
S11_D_LINES = 24  # D lines the merged notes carry: A's (at most S10_D_LINES), then B's; see s11_union_cap
# B is extra time before the program: its gate keeps, past R_code (and STATE on a stateful unit), this many more repair
# rounds, each a patch and a rerun at the speed repair's reserve (Pace.patch() + EXEC_FLOOR + 10) at the gate rate.
# Of r13b's two passes near the clock, one needed its last patch retry past R_code (kept: at its rate B is skipped);
# the other's first run took all of RUN_CAP_FLOOR, which no reserve that leaves B room covers. Two rounds would skip
# B at any rate below about 125 tok/s.
S11_REPAIR_ROUNDS = 1
S11_READING_CHARS = 300  # characters of B's reading that a marked line carries
S11_BINDING_READING_CHARS = 600  # Complete unverified alternative plus source; never sliced.
S11_QUANTITY = 0.25  # quantity-word overlap (Jaccard) at which two lines quoting one TASK sentence decide one item
S11_SAME_QUANTITY = 0.5  # the overlap at which a TASK-quoted B line and an A line without a TASK quote do
S11_COVER = 0.35  # quantity-and-decision word overlap at which an A line covers a B line (the deep dive's matcher)
S11_QUOTE_MIN = 8  # characters a quote needs, normalized, to be placed in a TASK sentence
# Two readings of one item differ only on a plain contradiction (_s11_differs): the probe's DECIDE pairs mostly
# paraphrase each other, and a paraphrase marked as a conflict is noise the base never had. The poles of a convention
# (one reading names one end and not the other, the other reading the reverse), numbers neither reading's set holds
# all of, or a negation alone between two readings sharing S11_NEGATED of their words.
S11_POLES = (('discounted', 'undiscounted'), ('inclusive', 'exclusive'), ('open', 'close'), ('long', 'short'),
             ('positive', 'negative'), ('before', 'after'), ('start', 'end'), ('first', 'last'), ('simple', 'log'),
             ('continuous', 'discrete'), ('absolute', 'relative'), ('daily', 'annualized'), ('daily', 'annualised'),
             ('trading', 'calendar'), ('sample', 'population'), ('call', 'put'), ('arithmetic', 'geometric'),
             ('upper', 'lower'), ('gross', 'net'), ('bid', 'ask'), ('mean', 'median'), ('percent', 'fraction'))
S11_NEGATION = ('not', 'no', 'never', 'without')
S11_NEGATED = 0.6
# A figure as _s11_differs compares it, so one value written two ways is not a contradiction: a percent is a fraction
# and basis points are one, except `x% of` something, a relative size that an absolute fraction can contradict (a bump
# of 1% of a value against an absolute 0.01).
_S11_FIGURE = re.compile(r'(?<![\w.])(\d+(?:\.\d+)?)(?:\s*(%|percent\b|pct\b|bps\b|bp\b|basis points?\b)(\s*of\b)?)?'
                         r'(?!\w|\.\d)', re.I)
# A B line read after A's lines loses its neighbours: one that only points back (`as above`, `same call`) says nothing
# there, and its `D3` would name A's D3 (s11_union drops the one, renumbers the other).
_S11_BACK = re.compile(r'\b(?:as (?:above|before|previously|earlier)|see above|same (?:call|function|routine|formula|'
                       r'method|way)|ditto|likewise)\b', re.I)
_S11_REF = re.compile(r'(?<!\w)D(\d{1,2})\b')
S11_STATUSES = ('ok', 'cut_kept', 'no_lines', 'spilled', 'failed', 'skipped_time', 'skipped_requests',
                'skipped_breaker', 'no_notes')
_S11_STOP = ('a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'each', 'for', 'from', 'in', 'is', 'it', 'its', 'of',
             'on', 'or', 'that', 'the', 'this', 'to', 'use', 'used', 'uses', 'using', 'with')
# Frozen before held-out scoring (spec §3): the texts and the thresholds of §2.0, §5 and §7. A test pins it.
S10_PROMPT_SHA = hashlib.sha256(json.dumps([
    S10_DECIDE, S10_FAMILY, S10_TEMPLATE, S10_LIBRARY_MAP_ADOPT, S10_LIBRARY_MAP, S10_DISTILL, S10_CHECK, S10_STATE,
    S10_NOTES, S10_NOTES_STATE, S10_CHECKED, S10_TAGS, S10_PATCH_KEEP, PLAN_LIBRARY,
    S10_FILE.pattern, S10_STATEFUL.pattern, S10_TIER_L_CHARS, S10_TIER_L_FILES, S10_TIER_F_CHARS, S10_TIER_F_FILES,
    S10_STATE_HITS, S10_TOKEN_CEILING, S10_TPW, S10_EPW, S10_DECIDE_BUDGET, S10_DECIDE_WORDS, S10_CODE_BUDGET,
    S10_CHECK_BUDGET, S10_CHECK_WORDS, S10_STATE_BUDGET, S10_STATE_WORDS, S10_DISTILL_WORDS, S10_PATCH_BUDGET,
    S10_FLAGFIX_BUDGET, S10_FLOOR, S10_MARGIN, S10_PRIOR_RATE, S10_KEEP, S10_CHECK_USED, S10_OVERRUN,
    S10_OVERRUN_SLACK, S10_DISTILL_TAIL, S10_DISTILL_MIN, S10_REASONING_CPT, S10_ANSWER_CPT, S10_D_MIN, S10_S_MIN,
    S10_D_LINES, S10_D_CHARS, S10_STATE_LINES, S10_STATE_CHARS, S10_STATE_TOTAL, S10_OUTPUTS_CHARS,
    S10_CHECKED_CHARS, S10_BODY_MIN, S10_PROMPTLY, S10_DECIDE_LIMIT, S10_CHECK_MODES, S10_CHECK_DEFAULT, S10_SPILL,
    S10_SPILL_MIN, S11_CONV_DECIDE_WORDS, S11_CONV_D_LINES, S11_CONV_D_EXTRA, S11_CONV_LIMIT, S11_CONV_DEFAULTS,
    S11_CONV_FIELD, S11_CONV_HALF, S11_RAISE_INPUT, S11_RAISE_GRID, S11_DECIDE_FROM, S11_DECIDE_TO, S11_OWN_RAISE,
    S11_OWN_AGAIN, PROFILE_RANGE_ROWS, PROFILE_RANGE_SHARE, PROFILE_RANGE_BYTES, PROFILE_RANGE_TOTAL,
    PROFILE_RANGE_MIN, PROFILE_RANGE_NAME, _STAMP.pattern, _DMY.pattern, sorted(_TIME_WORDS), _YMD.pattern, S11_OOM,
    S11_OOM_CODES, S11_OUTPUT_ITEMS, S11_ORDER_A, S11_ORDER_B, S11_ORDER_B_L, S11_UNSURE, S11_D_LINES,
    S11_READING_CHARS, S11_QUANTITY, S11_SAME_QUANTITY, S11_COVER, S11_QUOTE_MIN, S11_POLES, S11_NEGATION,
    S11_NEGATED, _S11_STOP, S11_REPAIR_ROUNDS, _S11_FIGURE.pattern, _S11_BACK.pattern, _S11_REF.pattern],
    sort_keys=True).encode()).hexdigest()
# T1_S12_MECH (off unless set; acts only under T1_S10): stop wasting time and requests. In r13b units stopped after the
# first clean run with 190-370 s and 15-18 requests unused, and three lost 110-220 s to requests that timed out with no
# reply. (1) STATE honours DECIDE's `STATE: no` unless the TASK's loop words are far above S10_STATE_HITS: of the 64
# dev units 7 have 23-55 (backtests, ledgers, hedges) and the next has 12, so any line from 13 to 23 splits them alike.
S12_STATE_STRONG = 2 * S10_STATE_HITS
S12_STATE_FROM = ('reply', 'strong', 'hits', 'no_honoured', 'none')
# (2) A note request's timeout is S12_TIMEOUT_X times its expected reply at the gate rate (its budget and S10_EPW
# tokens per asked word), with v1.2's S10_NOTE_TIMEOUT_FLOOR only where that is lower and the clock holds the floor.
# One that ends without a reply ends the notes stage: a timeout halves the rate (v1.2 kept it) and is not retried, the
# later note steps are skipped, and the code request goes with thinking off (r13b: a STATE timed out at 110 s at a
# measured 68 tok/s, and the code request still reasoned 700 tokens at 23.6 tok/s).
S12_TIMEOUT_X = 2.0
S12_NOTE_KINDS = S10_KINDS + ('s11_decide_b',)
# (3) A program reply without a fence whose loose code verifies to less than S12_CODE_SHARE of itself (prose with an
# outline of code: all 5 unfenced replies of r13b verified 0 characters) holds no code: it is rewritten, never
# continued or salvaged. Only a reply asked for a whole program (S12_FENCE_KINDS; a rewrite after nothing verified):
# a continuation goes on inside the fence the conversation left open, often without one (4 of 15 cut continue_line
# replies in the dev replays), and its indented first line never verifies.
S12_CODE_SHARE = 0.5
S12_FENCE_KINDS = ('initial', 'rewrite')
# (4) Where a full reply no longer fits, the continuation gate needs what the cut program has left to write: the cut
# statement again (continue_stmt), the rest of it, each stub or function called but not defined, the __main__ block
# and each output file the TASK names that the program does not name yet. Where re-sending the cut statement does
# not fit, its kept lines are continued in place (continue_line). dupire was stopped with 181 s left at 23.7 tok/s
# (a full reply and a run: 220 s) with about 2k tokens left to write after an 11k-character main().
S12_CODE_CPT = 3.3  # characters per token of code (dupire's cut reply: 11.3k characters in about 3.35k tokens)
S12_OPEN_TOKENS = 200  # the rest of the statement the cut ended
S12_DEF_TOKENS = 300  # a stub, or a function called but not defined yet
S12_OUTPUT_TOKENS = 150  # an output file the TASK names that the program does not name yet
S12_GUARD_TOKENS = 50  # the `if __name__ == "__main__":` block
S12_CONT_MIN = 300
S12_CONT_KINDS = ('continue_stmt', 'continue_line')
# (5) A crash patch that returns nothing usable (no reply, a reply cut at the cap that left the crash site as it was,
# or prose without code) is retried once with thinking off and at most S12_PATCH_CAP tokens, for the changed lines
# (on top of what a cut patch merged); where that light retry comes back with no change, stack 11's RETRY follows on
# its own gate. The crash ask names the crash site where the traceback gives one (cme-hdd's first patch rewrote the
# whole program and was cut at 4,000 tokens; bl-regime's timed out). Both asks offer the crash site's def whole only
# where it fills at most S12_DEF_SHARE of the cap: 17 of r13b's 25 crash sites lie in a def over 1,500 tokens (a
# main() of 1.8k-3.7k), and the RETRY replies after a cut first patch were whole main()s of 1.9k-3.9k tokens.
S12_PATCH_CAP = 1500  # first patches answered in 175 tokens (median) and 725 (p90), spec §5
S12_DEF_SHARE = 0.8
S12_UNUSABLE = ('failed', 'cut', 'prose')
S12_CRASH_SITE = ('The error is raised{where}. Send only the code you change, at the crash site or in the caller or '
                  'data preparation that causes it: the changed lines with the unchanged line before and after '
                  'them{whole}; never the whole program. ')
S12_PATCH_WHOLE = (', or the complete def that holds them', ', not the whole `{function}`')
S12_PATCH_WHY = {'failed': 'did not arrive in time', 'cut': 'hit the output limit before it changed the failing code',
                 'prose': 'contained no code'}
S12_PATCH_LIGHT = ('Your previous reply {why}. Reply now with a short patch and no reasoning: {site}one ```python '
                   'fence with only the changed lines (at the crash site, or in the caller or data preparation that '
                   'causes it) and the unchanged line before and after them, at their original indentation{whole}; '
                   'never the whole program. No text outside the fence.')
S12_PATCH_SITE = 'the error is raised{where}; '
S12_PATCH_TURN = {'failed': '(no reply)', 'cut': '(cut at the output limit)', 'prose': '(no code)'}
# (6) T1_S11_CONV's defaults gain these items (the deep dive's windows item is S11_CONV_DEFAULTS' (d) already).
S12_CONV_MORE = (
    '(i) Leverage: total or gross leverage is the sum of the absolute weights, unless the TASK says otherwise.\n'
    '(j) Risk measures: VaR and ES are positive losses at the stated confidence (the 95% VaR is the 95th percentile of '
    'the loss distribution), unless the TASK says otherwise.\n'
    '(k) Year fractions: between timestamps that carry a time of day, a year fraction is the exact elapsed time, '
    'seconds / (365 x 86400), unless the TASK names a day count.\n'
    '(l) Category keys: the keys of a count or breakdown by category are the TASK\'s own category words, verbatim, '
    'unless the TASK says otherwise.')
# Frozen with S10_PROMPT_SHA's rule: a test pins it.
S12_MECH_SHA = hashlib.sha256(json.dumps([
    S12_STATE_STRONG, S12_TIMEOUT_X, S12_CODE_SHARE, S12_CODE_CPT, S12_OPEN_TOKENS, S12_DEF_TOKENS, S12_OUTPUT_TOKENS,
    S12_GUARD_TOKENS, S12_CONT_MIN, S12_PATCH_CAP, S12_CRASH_SITE, S12_PATCH_WHY, S12_PATCH_LIGHT, S12_PATCH_SITE,
    S12_CONV_MORE, S12_DEF_SHARE, S12_PATCH_WHOLE, S12_PATCH_TURN, S12_FENCE_KINDS],
    sort_keys=True).encode()).hexdigest()


def thin_on() -> bool:
    """Current thin solve loop; the older profiles remain historical."""
    return os.environ.get("T1_THIN") == "1"


def input_fidelity_on() -> bool:
    """input_fidelity (T1_THIN_FEATURES) for what is built outside thin.solve's own reading of the switches: the
    INPUT PROFILE (built before solve) and the crash report's key hint (inside the program, T1_KEY_HINT_CUT). Only
    under T1_THIN; a value solve would refuse reads as off here (solve then stops before any request)."""
    if not thin_on():
        return False
    try:
        return 'input_fidelity' in solver_profile.parse_thin_features(os.environ.get('T1_THIN_FEATURES', ''))
    except ValueError:
        return False


def minimal_on() -> bool:
    """The current short-plan/component controller; excludes legacy planners."""
    return os.environ.get('T1_MINIMAL') == '1'


def s10_on() -> bool:
    """T1_S10=1 (off in the image): stack 10's notes stage. Off, every T1_S10_* switch is ignored and every request
    is stack9's."""
    return os.environ.get('T1_S10') == '1' and not minimal_on()


def s11_raise_on() -> bool:
    """T1_S11_RAISE=1, under T1_S10 only (deep dive fix 3): the task prompt's narrowed raise rule and time grids
    (_raise_rule), DECIDE's (S11_DECIDE), the crash ask's note on the program's own raise (s11_own_raise) and the
    profile's date ranges. Off, every request is stack10's."""
    return (s10_on() or minimal_on()) and os.environ.get('T1_S11_RAISE') == '1'


def s11_lint_on() -> bool:
    """T1_S11_LINT=1 (off in the image), only under T1_S10: zero-request lints. A pyplot backend call that cannot work
    is rewritten before every run (library_fixed), output and code lints after a clean run open the flags-only fix
    round (s11_flags, review_outputs), and a crash ask lists an lstsq misread and, for a kill without a traceback,
    says the run ran out of memory (repair_program). Off, every request and program is stack 10's."""
    return os.environ.get('T1_S11_LINT') == '1' and (s10_on() or minimal_on())


def s11_flags(instruction, out, program, known=()):
    """(output, code): T1_S11_LINT's flags after a clean run. Output: a non-finite number in a field the TASK names
    (nonfinite_flags; one known, the contract's flags, already reports is left out) and one quantity with two values
    in two JSON outputs (crosskey_flags), at most S11_OUTPUT_ITEMS. Code: an lstsq result read as residuals, or a numpy
    pareto draw left unshifted where the TASK names a Pareto with a minimum (api_lint), at most LINT_LIMIT. Never
    raises: a lint that fails finds nothing."""
    try:
        output = (nonfinite_flags(instruction, out, known) + crosskey_flags(instruction, out))[:S11_OUTPUT_ITEMS]
    except Exception:
        output = []
    try:
        code = api_lint(program, instruction)[:LINT_LIMIT]
    except Exception:
        code = []
    return output, code


def s10_tags_on() -> bool:
    """T1_S10_TAGS=1: the notes block asks for a `# D<n>` comment on the line that implements each decision."""
    return os.environ.get('T1_S10_TAGS') == '1'


def s11_conv_on() -> bool:
    """T1_S11_CONV=1 under T1_S10 (off unless set): DECIDE's DEFAULTS items, the L tier's larger caps and the default
    lines a parse keeps past its cap, STATE's rescue from a failed gate, and a CHECK reply without D lines dropped.
    Off, every payload is stack 10's."""
    return (s10_on() or minimal_on()) and os.environ.get('T1_S11_CONV') == '1'


def s12_mech_on() -> bool:
    """T1_S12_MECH=1 under T1_S10 (off unless set): STATE honours DECIDE's `STATE: no` below S12_STATE_STRONG loop
    words (s12_stateful); a note request's timeout follows its expected reply, and one that ends without a reply ends
    the notes stage (_s12_note_failed); a program reply without a fence is no code (s12_no_fence); the continuation
    gate is sized to what remains (s12_continuation); a crash patch that returns nothing usable gets one light retry
    (_s12_light_patch) and the crash ask names the crash site; under T1_S11_CONV DECIDE's defaults gain S12_CONV_MORE.
    Off, every payload is stack 11's."""
    return (s10_on() or minimal_on()) and os.environ.get('T1_S12_MECH') == '1'


def s12_stateful(hits, state):
    """T1_S12_MECH: (whether STATE runs, why, S12_STATE_FROM) from the TASK's loop words (hits) and DECIDE's STATE
    answer: yes (reply); S12_STATE_STRONG hits or more, whatever DECIDE said (strong); S10_STATE_HITS or more where
    DECIDE gave no answer (hits) and not where it said no (no_honoured); otherwise not (none)."""
    if state == 'yes':
        return True, 'reply'
    if hits >= S12_STATE_STRONG:
        return True, 'strong'
    if hits >= S10_STATE_HITS:
        return (False, 'no_honoured') if state == 'no' else (True, 'hits')
    return False, 'none'


def _s12_note_failed(client, status, kind, tried):
    """T1_S12_MECH: a note request that went out (tried) and ended without a reply ends the notes stage; the first
    such request's kind is client.s12_note_failed and status['s12_note_failed']. Nothing with the switch off."""
    if s12_mech_on() and tried and getattr(client, 's12_note_failed', None) is None:
        client.s12_note_failed = status['s12_note_failed'] = kind


def _s12_notes_ended(client, status):
    """T1_S12_MECH: whether an earlier note request ended without a reply, so this note step is skipped
    (s12_notes_skipped counts it). False with the switch off."""
    if not s12_mech_on() or getattr(client, 's12_note_failed', None) is None:
        return False
    _count(status, 's12_notes_skipped')
    return True


def s10_decide_words(tier):
    """DECIDE's word cap on a tier: S11_CONV_DECIDE_WORDS under T1_S11_CONV, else S10_DECIDE_WORDS."""
    return (S11_CONV_DECIDE_WORDS if s11_conv_on() else S10_DECIDE_WORDS)[tier]


def s10_distill_words(tier):
    """The reasoning-only DISTILL's word cap: S10_DISTILL_WORDS, or under T1_S11_CONV DECIDE's where that is larger
    (its ask has the DEFAULTS lines to write)."""
    return max(S10_DISTILL_WORDS, s10_decide_words(tier)) if s11_conv_on() else S10_DISTILL_WORDS


def s10_d_lines():
    """The D lines a parse keeps and the notes carry: S11_CONV_D_LINES under T1_S11_CONV, else S10_D_LINES
    (T1_S11_CONV adds up to S11_CONV_D_EXTRA past it that end in default: s10_parse, s10_kept_ids)."""
    return S11_CONV_D_LINES if s11_conv_on() else S10_D_LINES


def s10_kept_ids(s10):
    """The IDs of the D lines CHECK, STATE and the notes read: the first s10['d_cap'] (T1_S11_UNION's merged lines,
    s11_union_cap) or s10_d_lines(), then under T1_S11_CONV up to S11_CONV_D_EXTRA more that end in default
    (_s11_default)."""
    order, cap = s10['order'], s10.get('d_cap', s10_d_lines())
    if not s11_conv_on():
        return list(order[:cap])
    return list(order[:cap]) + [n for n in order[cap:] if _s11_default(s10['decisions'][n])][:S11_CONV_D_EXTRA]


def s10_budget(step, tier='D', status=None):
    """A step's reasoning budget in tokens: its T1_S10_* override (_env_tokens semantics) or the S10 constant, clamped
    to the token rule budget + S10_TPW x the step's word cap <= S10_TOKEN_CEILING (status['s10_budget_clamped']
    counts each clamp). Steps: decide (the tier's), check, state, code (the tier's), flagfix, patch (whose override
    is T1_PATCHTHINK_BUDGET) and s12_audit (T1_S12_POST's audit); code, flagfix and patch ask no word cap. DECIDE's
    word cap is s10_decide_words'."""
    if minimal_on():
        return 0  # component generation and evidence-based patches do not run the old reasoning budget
    name, default, words = {
        'decide': (f'T1_S10_DECIDE_BUDGET_{tier}', S10_DECIDE_BUDGET[tier], s10_decide_words(tier)),
        'check': ('T1_S10_CHECK_BUDGET', S10_CHECK_BUDGET, S10_CHECK_WORDS),
        'state': ('T1_S10_STATE_BUDGET', S10_STATE_BUDGET, S10_STATE_WORDS),
        'code': (f'T1_S10_CODE_BUDGET_{tier}', S10_CODE_BUDGET[tier], 0),
        'flagfix': ('T1_S10_FLAGFIX_BUDGET', S10_FLAGFIX_BUDGET, 0),
        'patch': ('T1_PATCHTHINK_BUDGET', S10_PATCH_BUDGET, 0),
        's12_audit': ('T1_S12_AUDIT_BUDGET', S12_AUDIT_BUDGET, S12_AUDIT_WORDS)}[step]
    budget, cap = _env_tokens(name, default), math.floor(S10_TOKEN_CEILING - S10_TPW * words)
    if budget > cap:
        if status is not None:
            _count(status, 's10_budget_clamped')
        budget = cap
    return budget


def s10_triage(instruction, profile):
    """Stack 10's no-request triage (spec §2.0) of the TASK with its canary lines removed: chars; files, the data file
    names it mentions (inputs too: a size signal, not a contract); state_hits (S10_STATEFUL); template, an input .py
    file it names (which the profile then shows in full); stateful, state_hits >= S10_STATE_HITS; and the tier: L for
    a long TASK or many files, F for a short one with few files, no loop words and no template, D otherwise
    (T1_S10_TIER overrides it in local probes). The tier sets DECIDE's word cap and budget and the code budget only,
    never the clock."""
    chars, files = len(instruction), len({name.lower() for name in S10_FILE.findall(instruction)})
    hits = len(S10_STATEFUL.findall(instruction))
    paths = [PurePosixPath(str(row.get('path', ''))) for row in profile.get('files', []) if isinstance(row, dict)]
    template = any(path.suffix.lower() in _SOURCES and _names(instruction, path.name) for path in paths)
    tier = ('L' if chars >= S10_TIER_L_CHARS or files >= S10_TIER_L_FILES else
            'F' if chars < S10_TIER_F_CHARS and files <= S10_TIER_F_FILES and hits < S10_STATE_HITS and not template
            else 'D')
    forced = os.environ.get('T1_S10_TIER')
    return {'tier': forced if forced in S10_TIERS else tier, 'chars': chars, 'files': files, 'state_hits': hits,
            'template': template, 'stateful': hits >= S10_STATE_HITS}


def s10_triage_status(status, triage):
    """The triage's status fields (s10, s10_tier, s10_chars, s10_files, s10_state_hits, s10_template), the prompt
    hash, the clamp counter and the spill-over DISTILL counter (s10_distill_spill): solve sets them under T1_S10
    whether or not the notes stage runs (T1_PLAN), with s11_conv under T1_S11_CONV and s12_mech, s12_note_failed,
    s12_note_timeouts and s12_notes_skipped under T1_S12_MECH (and s12_conv_more, DECIDE's S12_CONV_MORE, under
    both)."""
    status.update(s10=True, s10_tier=triage['tier'], s10_chars=triage['chars'], s10_files=triage['files'],
                  s10_state_hits=triage['state_hits'], s10_template=triage['template'],
                  s10_prompt_sha=S10_PROMPT_SHA[:12])
    if s11_conv_on():
        status['s11_conv'] = True
    if s12_mech_on():  # T1_S12_MECH's fields start here; the notes stage and solve count on them
        status['s12_mech'] = True
        if s11_conv_on():
            status['s12_conv_more'] = True
        for key, value in (('s12_note_failed', 'none'), ('s12_note_timeouts', 0), ('s12_notes_skipped', 0)):
            status.setdefault(key, value)
    status.setdefault('s10_budget_clamped', 0)
    status.setdefault('s10_distill_spill', 0)


def _s10_rate(pace):
    """The rate the S10 gates read: the measured one, else at most S10_PRIOR_RATE."""
    return pace.rate if pace.measured else min(pace.rate, S10_PRIOR_RATE)


def _s10_cost(tokens, rate):
    return tokens / rate + 6


def _s10_reserve(rate):
    """R_code: one full code reply, one patch and two runs (Pace.full() + Pace.patch() + 2 x EXEC_FLOOR) at rate."""
    return (4000 / rate + 6) + (2500 / rate + 6) + 2 * EXEC_FLOOR


def _s10_gate(step, b_max, words, later, deadline, pace):
    """The reasoning budget a note step (decide, check or state) may carry now, else None (spec §7.2): the most, up to
    b_max, for which its reply (the budget plus S10_TPW tokens per asked word, at the gate rate), the seconds later
    steps reserve (later), R_code and S10_MARGIN still fit before the deadline. One token under the exact fit, so the
    strict send's own check a moment later passes. Below the floor (S10_FLOOR, or b_max when lower) the step is
    skipped: it is never sent with less reasoning than that."""
    rate = _s10_rate(pace)
    room = deadline - CLOCK.now() - later - _s10_reserve(rate) - S10_MARGIN - 6
    budget = min(b_max, math.floor(room * rate - S10_TPW * words) - 1)
    return budget if b_max > 0 and budget >= min(S10_FLOOR, b_max) else None


def _s10_send(client, messages, kind, budget, words, later, deadline, pace, sleep):
    """A note request with its budget, strict: _request's think_fits checks _s10_gate's inequality again; its timeout
    is at least S10_NOTE_TIMEOUT_FLOOR where the clock allows it (T1_S12_MECH: S12_TIMEOUT_X x the reply _s12_expect
    gives, with the floor where that is lower and the clock holds it)."""
    rate = _s10_rate(pace)
    return _request(client, messages, deadline, pace, kind, sleep, need=_s10_cost(S10_TPW * words, rate),
                    reserve=later + _s10_reserve(rate), keep=S10_KEEP, budget=budget, strict=True,
                    think_margin=S10_MARGIN, timeout_floor=S10_NOTE_TIMEOUT_FLOOR, note=True,
                    expect=_s12_expect(budget, words, rate))


def _s12_expect(budget, words, rate):
    """T1_S12_MECH: the seconds a note reply is expected to take at rate, its reasoning budget and S10_EPW tokens per
    asked word (the token rule's expected total); None with the switch off."""
    return _s10_cost((budget or 0) + S10_EPW * words, rate) if s12_mech_on() else None


def _s10_reasoning(record):
    """A reply's reasoning tokens, estimated: its reasoning characters / S10_REASONING_CPT, or without reasoning text
    the completion tokens less the answer characters / S10_ANSWER_CPT."""
    reasoning, answer, tokens = (record.get(key) for key in ('reasoning_chars', 'content_chars', 'completion_tokens'))
    if type(reasoning) is int and reasoning > 0:
        return reasoning / S10_REASONING_CPT
    if type(tokens) is int:
        return max(0.0, tokens - (answer if type(answer) is int else 0) / S10_ANSWER_CPT)
    return 0.0


def _s10_breaker(client, record, budget, reply=None):
    """T1_S10's unit-wide breaker, run by _request on every reply sent with a budget: it adds the reply's estimated
    reasoning to client.s10_reasoning_tokens, keeps the largest reasoning / budget in client.s10_overrun, and trips on
    reasoning past S10_OVERRUN x budget + S10_OVERRUN_SLACK (overrun), a reply cut at the cap without answer text
    (reasoning_only), or, when the endpoint shows no reasoning text, a token count past the budget, the answer's
    worst size (S10_TPW tokens per word of the answer text, or S10_ANSWER_CPT characters per token on dense code) and
    S10_COMPLETION_SLACK (overrun_tokens): client.s10_think_off then holds for the rest of the unit and
    client.s10_breaker records the first cause. Returns the cause or None."""
    tokens = _s10_reasoning(record)
    client.s10_reasoning_tokens = getattr(client, 's10_reasoning_tokens', 0.0) + tokens
    client.s10_overrun = max(getattr(client, 's10_overrun', 0.0), tokens / budget)
    text = strip_reasoning(_text(reply)).strip() if reply is not None else None
    answered = bool(text) if text is not None else (record.get('content_chars') or 0) > 0
    chars, words = (len(text), len(text.split())) if text is not None else (record.get('content_chars') or 0, 0)
    completion = record.get('completion_tokens')
    # Only where the endpoint shows no reasoning text: with the text, the reasoning-length test above is the measure.
    counted = (not record.get('reasoning_chars') and type(completion) is int
               and completion > budget + max(S10_TPW * words, chars / S10_ANSWER_CPT) + S10_COMPLETION_SLACK)
    cause = ('overrun' if tokens > S10_OVERRUN * budget + S10_OVERRUN_SLACK else
             'reasoning_only' if record.get('finish_reason') == 'length' and not answered else
             'overrun_tokens' if counted else None)
    if cause:
        client.s10_think_off = True
        if getattr(client, 's10_breaker', 'none') == 'none':
            client.s10_breaker = cause
    return cause


def s10_client_status(client, status):
    """The breaker's fields: s10_breaker (S10_BREAKER), s10_overrun (the largest reasoning / budget, 2 decimals) and
    s10_reasoning_tokens (estimated, summed over the budgeted replies); under T1_S12_MECH s12_note_timeouts."""
    status.update(s10_breaker=getattr(client, 's10_breaker', 'none'),
                  s10_overrun=round(getattr(client, 's10_overrun', 0.0), 2),
                  s10_reasoning_tokens=int(round(getattr(client, 's10_reasoning_tokens', 0.0))))
    if s12_mech_on():  # T1_S12_MECH: the note requests that died at their timeout
        status['s12_note_timeouts'] = getattr(client, 's12_note_timeouts', 0)


# A line's lead before an ID or a section word: a bullet, a list number (`1.`, `2)`) or a table row's `|`. A label in
# parentheses before the separator (`S2 (signal):`) is kept as the start of the text; the markdown that closes a bold
# ID after its separator (`**D1:** text`) is not part of it. No two quantifiers of a pattern can share a run of
# whitespace, so a long run costs linear time (the parse runs on model text, on the unit clock).
_S10_LEAD = r'^\s*(?:(?:[-*+\u2022]|\d{1,2}[.)]|\|)\s*)?'
_S10_LABEL = r'[*_`]*\s*(?:\(([^()\n]*)\)[*_`]*\s*)?'
_S10_D = re.compile(_S10_LEAD + r'[*_`]*D(\d{1,2})' + _S10_LABEL + r'(?:\||[:.)])[*_]*\s*(.+)$')
_S10_S = re.compile(_S10_LEAD + r'[*_`]*S(\d{1,2})' + _S10_LABEL + r'[:.)|][*_]*\s*(.*)$')
_S10_ID = re.compile(_S10_LEAD + r'[*_`]*[DS]\d{1,2}\b')  # an ID the item forms above do not read
_S10_WORD = re.compile(r'^(?:\s*(?:[-*+\u2022]|\d{1,2}[.)]|\|))?[*_`#\s]*(STATE|FAMILY|OUTPUTS|DECISIONS|LIBRARY MAP|'
                       r'OPEN|INIT|RECORD|LABELS|END|OTHERS CONFIRMED)\b[*_`]*\s*(?:\([^()\n]*\)[*_`\s]*)?'
                       r'(?::[*_`]*\s*(.*)|$)', re.I)
_S10_STEPS = ('INIT', 'RECORD', 'LABELS', 'END')
_S10_CONFIRMED = re.compile(r'^[\W_]*OTHERS CONFIRMED[\W_]*$', re.I)
_S10_JOINED = ('OUTPUTS', 'OPEN')  # header lines whose value may wrap onto the next lines
_S10_NONE = re.compile(r'[*_`"\s]*none\b', re.I)  # an OPEN value the parse reads as no point at all
_S10_NOTE_INFO = re.compile(r'(?:text|plaintext|txt|markdown|md)?', re.I)  # a fence info string that is not code
_S10_NOTHING = (['none'], ['n', 'a'], ['na'], ['nothing'], ['nil'])


def _s10_labelled(match):
    label, text = (match.group(2) or '').strip(), re.sub(r'\s*\|$', '', match.group(3).strip())  # a table row's |
    return f'{label} | {text}' if label and text else label or text


def _s10_starts(line):
    """Whether a line starts a note item: a D or S line, or a section word (not a `#` comment)."""
    return bool(_S10_D.match(line) or _S10_S.match(line)
                or (not line.lstrip().startswith('#') and _S10_WORD.match(line)))


def _s10_unfenced(text):
    """text without the fences of notes wrapped in a plain, text or markdown fence (one whose first line starts a note
    item), so _without_code does not remove the notes as code. Every other fence, and what it holds, stays."""
    kept, lines, index = [], text.split('\n'), 0
    while index < len(lines):
        match = _PLAN_FENCE.match(lines[index])
        fence = match.group(1) if match else ''
        info = lines[index].strip()[len(fence):]
        if not match or info.rstrip().endswith(fence):  # not a fence, or one closed on its own line
            kept.append(lines[index])
            index += 1
            continue
        end = next((k for k in range(index + 1, len(lines)) if _PLAN_FENCE.match(lines[k])
                    and not lines[k].strip().strip(fence[0]) and len(lines[k].strip()) >= len(fence)), len(lines))
        body = lines[index + 1:end]
        notes = _S10_NOTE_INFO.fullmatch(info.strip()) and _s10_starts(next((x for x in body if x.strip()), ''))
        kept += body if notes else lines[index:end + 1]
        index = end + 1
    return '\n'.join(kept)


def _s10_plain(text):
    """A note reply's text as the parsers read it: notes unwrapped from a plain fence (_s10_unfenced), then each code
    block replaced by PLAN_CODE_CUT (_without_code)."""
    return _without_code(_s10_unfenced(text or ''))[0]


def _s10_scan(lines):
    """[kind, key, value, last line, first line] for each item of a note reply's lines (_s10_items)."""
    items, current = [], None
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped == PLAN_CODE_CUT:  # tested first: no pattern runs on a blank line
            current = None
            continue
        decision, step, word = _S10_D.match(line), _S10_S.match(line), _S10_WORD.match(line)
        if word and not (decision or step) and word.group(1).upper() == 'DECISIONS':
            decision = _S10_D.match(word.group(2) or '')  # `DECISIONS: D1 | ...` opens the first line
        if _map_lines(line) and not decision:
            current = None
            continue
        if not (decision or step) and _S10_ID.match(line):  # `S1, S2, ...:` echoing the ask starts no item
            current = None
            continue
        if decision:
            current = ['D', int(decision.group(1)), _s10_labelled(decision), index, index]
        elif step:
            current = ['S', f'S{int(step.group(1))}', _s10_labelled(step), index, index]
        elif word:
            key = ' '.join(word.group(1).upper().split())
            current = ['S' if key in _S10_STEPS else 'section', key, (word.group(2) or '').strip(), index, index]
        elif current is not None and (current[0] in ('D', 'S') or current[1] in _S10_JOINED):
            current[2], current[3] = (current[2] + ' ' + stripped).strip(), index
            continue
        else:
            current = None
            continue
        items.append(current)
    return items


def _s10_last(lines):
    return max((index for index, line in enumerate(lines) if line.strip()), default=-1)


def _s10_items(text, cut=False):
    """[kind, key, value, last line, first line] for each item of a note reply, read from _s10_plain's text: a D line
    ('D', its number), a STATE spec line ('S': INIT, S<k>, RECORD, LABELS or END) or a header ('section': its word,
    value after its colon). A D line, a spec line, OUTPUTS and OPEN take the lines below them up to a blank line, a new
    item or another line that starts with an ID, a LIBRARY MAP line or removed code. With cut (the reply hit the
    output limit), the item its last line belongs to is left out: the cut may have ended it anywhere."""
    lines = _s10_plain(text).split('\n')
    last = _s10_last(lines)
    return [item for item in _s10_scan(lines) if not (cut and item[3] == last)]


def _s10_whole(text, cut=False):
    """A note reply's text as its text fallbacks keep it (_s10_plain's): with cut, without the item the cut ended or,
    outside any item, without its last line."""
    lines = _s10_plain(text).split('\n')
    if cut:
        last = _s10_last(lines)
        lines = lines[:max(0, next((item[4] for item in _s10_scan(lines) if item[3] == last), last))]
    return '\n'.join(lines).strip()


def _s10_nothing(text):
    """OPEN's value or one of its items saying there is nothing: no word at all, or only none, n/a, nothing or nil,
    whatever bullets, brackets and punctuation surround it."""
    words = re.sub(r'[\W_]+', ' ', text).lower().split()
    return not words or words in _S10_NOTHING


def s10_parse(text, cut=False):
    """A DECIDE, DISTILL or CHECK reply as data (spec §4): decisions {n: the line after its ID} and their order (the
    first line of an ID wins; at most s10_d_lines() IDs of S10_D_CHARS characters, and under T1_S11_CONV up to
    S11_CONV_D_EXTRA more that end in default), state ('yes', 'no' or None), outputs (at most S10_OUTPUTS_CHARS
    characters, or None) and open (OPEN's items split on ';', none for 'none', 'n/a' or the like). With cut, the
    item the cut ended is left out (_s10_items)."""
    found = {'decisions': {}, 'order': [], 'state': None, 'outputs': None, 'open': []}
    seen, limit = set(), s10_d_lines()
    extra = S11_CONV_D_EXTRA if s11_conv_on() else 0
    for kind, key, value, *_ in _s10_items(text, cut):
        if kind == 'D':
            kept = len(found['order'])
            if key not in found['decisions'] and value and (
                    kept < limit or (kept < limit + extra and _s11_default(value[:S10_D_CHARS]))):
                found['decisions'][key] = value[:S10_D_CHARS]
                found['order'].append(key)
        elif kind == 'section' and key not in seen:
            seen.add(key)
            if key == 'STATE':
                answer = re.match(r'[*_`"\s]*(yes|no)\b', value, re.I)
                found['state'] = answer.group(1).lower() if answer else None
            elif key == 'OUTPUTS':
                found['outputs'] = value[:S10_OUTPUTS_CHARS] or None
            elif key == 'OPEN' and not _S10_NONE.match(value):
                found['open'] = [item.strip() for item in value.split(';') if not _s10_nothing(item)]
    return found


def s10_state_parse(text, cut=False):
    """A STATE reply's spec lines in order, `<INIT, S<k>, RECORD, LABELS or END>: <text>` with continuation lines
    joined: at most S10_STATE_LINES lines of S10_STATE_CHARS characters and S10_STATE_TOTAL in all. With cut, the
    line the cut ended is left out."""
    lines, used = [], 0
    for kind, key, value, *_ in _s10_items(text, cut):
        if kind != 'S' or not value:
            continue
        line = f'{key}: {value}'[:S10_STATE_CHARS]
        if len(lines) >= S10_STATE_LINES or used + len(line) > S10_STATE_TOTAL:
            break
        lines.append(line)
        used += len(line) + 1
    return lines


def _s10_steps(lines):
    return sum(bool(re.match(r'S\d{1,2}:', line)) for line in lines)


def _s10_norm(text):
    return ' '.join(re.sub(r'[`*_]', '', text).lower().split())


def s10_merge(s10, checked):
    """CHECK's D lines merged into the decisions by ID (spec §4): a known ID replaces its line, counted as changed
    when the text differs beyond case, spacing and markdown; an unknown one is appended (added). Returns (changed,
    added)."""
    changed = added = 0
    for n in checked['order']:
        text = checked['decisions'][n]
        if n not in s10['decisions']:
            s10['decisions'][n] = text
            s10['order'].append(n)
            added += 1
        elif _s10_norm(text) != _s10_norm(s10['decisions'][n]):
            s10['decisions'][n] = text
            changed += 1
    return changed, added


def _s10_unsure(line):
    return re.search(r'\bUNSURE\b', line[-20:]) is not None


def _s10_free(line):
    """A D line whose last field is `free` (a trailing UNSURE aside); without fields, one that ends in free."""
    last = re.sub(r'\bUNSURE\b', '', line.split('|')[-1]).strip(' \t*_`.;:,-')
    return last.lower() == 'free' if '|' in line else re.search(r'\bfree$', last, re.I) is not None


def _s11_default(line):
    """A D line whose last field says default (T1_S11_CONV's mark of a default the TASK is silent on), a trailing
    UNSURE, quotes or a reason after it aside."""
    last = re.sub(r'\bUNSURE\b', '', line.split('|')[-1]).strip(' \t*_`.;:,-"\'')
    return re.match(r'default\b', last, re.I) is not None


def _s10_quote(line):
    """The last quoted span of a D line's last field, or None."""
    quotes = [a or b for a, b in _MAP_QUOTE.findall(line.split('|')[-1])]
    return quotes[-1] if quotes else None


def _s10_answer(reply, canaries):
    """A note reply's answer text, canaries withheld."""
    return withhold_canaries(strip_reasoning(_text(reply)).strip(), canaries)


def _s10_python(run):
    """Whether a run of lines outside every note item is Python rather than prose: two non-empty lines or more that
    parse once dedented (a program written after the notes without a fence)."""
    if sum(bool(line.strip()) for line in run) < 2:
        return False
    try:
        ast.parse(textwrap.dedent('\n'.join(run)))
    except (SyntaxError, ValueError, RecursionError, MemoryError, OverflowError):
        return False
    return True


def _s10_glued(item):
    """Whether a note item ran on past what its parse keeps (S10_D_CHARS of a D line, S10_STATE_CHARS of a spec line
    with its key, S10_OUTPUTS_CHARS of OUTPUTS, nothing of an OPEN that says none): prose written straight after it,
    without a blank line between, that _s10_scan joined to it."""
    kind, key, value = item[:3]
    cap = (S10_D_CHARS if kind == 'D' else S10_STATE_CHARS - len(f'{key}: ') if kind == 'S' else
           S10_OUTPUTS_CHARS if key == 'OUTPUTS' else 0 if key == 'OPEN' and _S10_NONE.match(value) else None)
    return cap is not None and len(value) > cap


def _s10_prose(text):
    """(prose lines, kept lines, code blocks) of a note reply, read from _s10_plain's text: a non-empty line is kept
    when it belongs to a note item (_s10_scan; the item a cut ended too; of an item that ran on past what its parse
    keeps, _s10_glued, its first line only), is a LIBRARY MAP line or says OTHERS CONFIRMED; each run of the other
    lines is prose, or a code block when it parses as Python (_s10_python); each fenced block removed counts as a
    code block too."""
    lines = _s10_plain(text).split('\n')
    inside = {index for item in _s10_scan(lines)
              for index in range(item[4], (item[4] if _s10_glued(item) else item[3]) + 1)}
    prose, kept, code, run = [], 0, 0, []
    for index, line in enumerate(lines + ['']):
        stripped = line.strip()
        closes = index == len(lines) or stripped == PLAN_CODE_CUT or index in inside
        if closes or _map_lines(line) or _S10_CONFIRMED.match(line):
            python = _s10_python(run)
            prose += [] if python else [x for x in run if x.strip()]
            code += python + (stripped == PLAN_CODE_CUT)
            kept, run = kept + (bool(stripped) and stripped != PLAN_CODE_CUT), []
        else:
            run.append(line)
    return prose, kept, code


def s10_spill(text, ids):
    """v1.1: whether a DECIDE, CHECK or STATE reply is spill-over, the model going on in its visible text once its
    budget ran out instead of writing the lines asked (_s10_prose's counts). Either the step's parse kept no ID line
    (ids: the D lines, or the spec lines, it keeps) and the reply holds code or S10_SPILL_MIN characters of prose, or
    that much prose outnumbers the lines the parse keeps (it keeps under half of the reply's lines)."""
    prose, kept, code = _s10_prose(text)
    reasoned = sum(len(line.strip()) for line in prose) >= S10_SPILL_MIN
    return (not ids and (reasoned or code > 0)) or (reasoned and kept < len(prose))


def _s10_spill(client, ask, answer, step, words, later, deadline, status, pace, canaries, sleep, replay,
               rescued=False):
    """v1.1's spill-over DISTILL for a step's reply that s10_spill flags: one thinking-off request (s10_distill, gated
    as DISTILL at the answer's cost, with R_code and the seconds later steps reserve, or after T1_S11_CONV's rescued
    STATE (rescued) the rescue's reserve, _s11_rescue_reserve) with the step's ask and S10_SPILL holding the last
    S10_DISTILL_TAIL characters of the reply (canaries withheld, code included: a program written instead of the
    lines is all such a reply holds), asking for the step's lines alone within its word cap. s10_distill_spill
    counts the requests that went out. (answer, cut) of its reply, or None when it was not sent or failed:
    s10_<step>_spill then says why (_s10_stopped)."""
    content = withhold_canaries(ask + '\n\n' + S10_SPILL.format(tail=answer[-S10_DISTILL_TAIL:], words=words),
                                canaries)
    rate, before = _s10_rate(pace), len(client.diagnostics['requests'])
    if _s12_notes_ended(client, status):  # T1_S12_MECH: an earlier note request ended without a reply
        status[f's10_{step}_spill'] = 'skipped_time'
        return None
    try:
        reply = _request(client, [{'role': 'user', 'content': content}], deadline, pace, 's10_distill', sleep,
                         need=_s10_cost(S10_TPW * words, rate),
                         reserve=_s11_rescue_reserve(rate) if rescued else later + _s10_reserve(rate), keep=S10_KEEP,
                         timeout_floor=S10_NOTE_TIMEOUT_FLOOR, note=True, expect=_s12_expect(0, words, rate))
    except (HouseError, _Stop) as exc:
        tried = _s10_tried(client, before)
        _count(status, 's10_distill_spill', int(tried))
        status[f's10_{step}_spill'] = _s10_stopped(exc, tried)
        _s12_note_failed(client, status, 's10_distill', tried)
        return None
    _count(status, 's10_distill_spill')
    pace.measure(client, reply)
    text = _s10_answer(reply, canaries)
    replay.put(f's10-{step}-spill.txt', text)
    return text, reply['finish_reason_raw'] == 'length'


def _s10_spill_used(status, step, found, needed):
    """Whether the spill-over DISTILL's reply replaces the step's: it gave needed ID lines or more (used), needed being
    the least the step stands with, never the count the step's reply kept (those lines are drafts its prose went on to
    revise, and the distill's ask caps it at the step's word cap and on the L tier at 12 D lines); otherwise the step's
    own reply stands as without it (fewer: some lines; no_lines: none). Sets s10_<step>_spill."""
    status[f's10_{step}_spill'] = 'used' if found >= needed else 'fewer' if found else 'no_lines'
    return found >= needed


def _s10_text(s10, open_items=False):
    """The decisions as CHECK and STATE read them: OUTPUTS, the DECIDE text when it gave too few D lines, the D lines,
    CHECK's text when it gave none, and (for CHECK) the OPEN points."""
    parts = [f'OUTPUTS: {s10["outputs"]}'] if s10['outputs'] and not s10['body'] else []
    parts += [s10['body']] if s10['body'] else []
    parts += [f'D{n} | {s10["decisions"][n]}' for n in s10_kept_ids(s10)]
    parts += [S10_CHECKED.format(text=s10['checked'])] if s10['checked'] else []
    parts += [f'OPEN: {"; ".join(s10["open"])}'] if open_items and s10['open'] else []
    return '\n'.join(parts)


_S10_MAP_HEADER = ('UNVERIFIED A CALL PROPOSALS (optional, not decisions or required calls; TASK takes precedence. '
                   'Verify the function, argument quantities, units and axes before use):\n')
_S10_MAP_SECTION = re.compile(r'^\s*(?:LIBRARY\s+)?MAP\s*:?', re.I)
_S10_MAP_LABEL = re.compile(r'[A-Za-z][\w /(),.+-]{0,159}')


def _s10_map_rows(text, cut=False):
    """Standalone raw arrow rows outside fences, strings, comments and Python continuations. A cut reply's last
    nonblank line is never complete evidence; sources are scanned separately before any legacy MAP gap-fill."""
    lines, fence, hidden = text.split('\n'), None, set()
    for i, line in enumerate(lines):
        match = _PLAN_FENCE.match(line)
        if fence is not None:
            hidden.add(i)
            if match and not line.strip().strip(fence[0]) and len(line.strip()) >= len(fence):
                fence = None
        elif match:
            hidden.add(i)
            fence = match.group(1)
    # Tokenization never executes text. It only excludes manufactured lines; incomplete later text does not
    # invalidate a complete earlier row. Fences are blanked without changing line positions.
    depth, first, fstrings = 0, set(), []
    try:
        for token in tokenize.generate_tokens(io.StringIO('\n'.join(
                '' if i in hidden else line for i, line in enumerate(lines))).readline):
            row, col = token.start[0] - 1, token.start[1]
            if token.type == getattr(tokenize, 'FSTRING_START', None):  # Python 3.12+; older versions use STRING
                fstrings.append(row)
            if fstrings:
                hidden.update(range(row, token.end[0]))
            if token.type == getattr(tokenize, 'FSTRING_END', None) and fstrings:
                fstrings.pop()
            if token.type in (tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.ENDMARKER):
                continue
            if row not in first:
                if depth:
                    hidden.add(row)
                first.add(row)
            if token.type == tokenize.COMMENT or (token.type == tokenize.STRING and
                    (token.end[0] > token.start[0] or col <= len(lines[row]) - len(lines[row].lstrip()))):
                hidden.update(range(row, token.end[0]))
            if token.type == tokenize.OP:
                depth += int(token.string in ('(', '[', '{')) - int(token.string in (')', ']', '}'))
    except tokenize.TokenError as exc:
        if fstrings:
            hidden.update(range(fstrings[0], len(lines)))
        elif 'multi-line string' in exc.args[0]:
            hidden.update(range(exc.args[1][0] - 1, len(lines)))
    except (IndentationError, SyntaxError, ValueError):
        return []  # unknown lexical context is not a standalone MAP contract
    last = _s10_last(lines)
    return [line.strip() for i, line in enumerate(lines)
            if i not in hidden and not (cut and i == last) and not line.startswith(('    ', '\t'))
            and not (i and lines[i - 1].rstrip().endswith('\\'))
            and _MAP_ARROW.search(line) and 'qfkit' in line]


def _s10_map_call(line, functions):
    """A closed, bounded call's original spelling, never a signature or an executable/semantic validation."""
    if len(line) > 600:
        return False
    split = _MAP_ARROW.split(line, maxsplit=1)
    label = _MAP_BULLET.sub('', split[0]).strip()
    if not _S10_MAP_LABEL.fullmatch(label) or not re.match(r'qfkit\.[A-Za-z_]\w*\s*\(', split[1].strip()):
        return False
    try:
        node = ast.parse(split[1].strip(), mode='eval').body
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return False
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name) and node.func.value.id == 'qfkit'
            and node.func.attr in functions):
        return False

    def value(item):
        if isinstance(item, ast.Name):
            return True
        if isinstance(item, ast.Constant):
            return item.value is not Ellipsis and (not isinstance(item.value, (float, complex)) or
                                                   (isinstance(item.value, float) and math.isfinite(item.value)))
        if isinstance(item, (ast.Tuple, ast.List)):
            return all(value(part) for part in item.elts)
        return (isinstance(item, ast.UnaryOp) and isinstance(item.op, (ast.UAdd, ast.USub))
                and isinstance(item.operand, ast.Constant) and type(item.operand.value) in (int, float)
                and value(item.operand))

    keys = [kw.arg for kw in node.keywords]
    return (None not in keys and len(keys) == len(set(keys))
            and all(value(item) for item in node.args) and all(value(kw.value) for kw in node.keywords))


def _s10_map_proposals(s10, instruction):
    """Admit only selected A source segments under the current task/library contract. B and later ADOPT map
    recovery do not write these sources. Counts are bounded carriage diagnostics, not adoption or correctness."""
    items, seen, omitted = [], set(), 0
    functions = library_functions() if library_allowed_by_task(instruction) else set()
    for text, cut in s10.get('map_sources', ()):
        for line in _s10_map_rows(text, cut):
            if line in seen:
                continue
            seen.add(line)
            if _s10_map_call(line, functions):
                items.append(line)
            else:
                omitted += 1
    s10['map_items'], s10['map_rejected'] = items, min(999, omitted)


def _s10_without_map_body(body):
    """A fallback body's raw MAP never becomes a mandatory decision, even when its proposal is inadmissible.
    Remove only MAP sections and explicit qfkit arrow rows; no-MAP text is byte-for-byte unchanged."""
    if not body:
        return body
    kept, section = [], False
    for line in body.split('\n'):
        if _S10_MAP_SECTION.fullmatch(line):
            section = True
            continue
        word = _S10_WORD.match(line)
        spec = (_S10_D.match(line) or _S10_S.match(line) or (word and
                ' '.join(word.group(1).upper().split()) in (*_S10_STEPS, 'STATE', 'OUTPUTS', 'OPEN', 'DECISIONS')))
        if spec:
            section = False
            kept.append(line)
            continue
        if section and (not line.strip() or _s10_starts(line)):
            section = False
        if section or (_MAP_ARROW.search(line) and 'qfkit' in line):
            continue
        kept.append(line)
    return '\n'.join(kept)


def _s10_append_maps(s10, notes):
    """Use only remaining notes space; whole items and header count toward both caps. Rerenders do not count twice."""
    kept, extra = [], ''
    for item in s10.get('map_items', ()):
        candidate = _S10_MAP_HEADER + '\n'.join(kept + [item]) + '\n\n'
        if len(kept) < 8 and len(candidate) <= min(2000, PLAN_CHARS - len(notes)):
            kept.append(item)
            extra = candidate
    s10['map_retained'] = len(kept)
    s10['map_omitted'] = min(999, s10.get('map_rejected', 0) + len(s10.get('map_items', ())) - len(kept))
    return notes + extra


def s10_notes(s10, tags=False):
    """The notes block of the code prompt (S10_NOTES; '' when DECIDE left no notes): OUTPUTS, the D lines (or DECIDE's
    own text when it gave too few), CHECK's text when it gave no lines, the STATE SPEC lines, and with tags
    (T1_S10_TAGS) the tag sentence. At most PLAN_CHARS characters: D lines past that are dropped from the end, then
    the text cut (under T1_S11_CONV the last line that does not end in default goes first). s10['kept'] receives the
    IDs the block shows."""
    s10['kept'] = []
    if not s10.get('notes'):
        return ''
    head = [f'OUTPUTS: {s10["outputs"]}'] if s10['outputs'] and not s10['body'] else []
    tail = [S10_CHECKED.format(text=s10['checked'])] if s10['checked'] else []
    tail += [S10_NOTES_STATE.format(lines='\n'.join(s10['state_lines']))] if s10['state_lines'] else []
    tail += [S10_TAGS] if tags else []
    kept = s10_kept_ids(s10)  # T1_S11_UNION's merged lines: s10['d_cap']

    def block(ids, body):
        return S10_NOTES.format(body='\n'.join(head + ([body] if body else [])
                                               + [f'D{n} | {s10["decisions"][n]}' for n in ids] + tail))
    body = _s10_without_map_body(s10['body']) or ''
    while kept and len(block(kept, body)) > PLAN_CHARS:
        routine = [i for i, n in enumerate(kept) if not _s11_default(s10['decisions'][n])] if s11_conv_on() else []
        kept.pop(routine[-1] if routine else -1)
    if len(block(kept, body)) > PLAN_CHARS:
        body = body[:max(0, len(body) - (len(block(kept, body)) - PLAN_CHARS))]
    s10['kept'] = kept
    return _s10_append_maps(s10, block(kept, body)[:PLAN_CHARS])


def with_s10_notes(prompt, notes):
    """The program prompt with the notes block last before the output format (it replaces PLAN_BLOCK)."""
    return prompt.removesuffix(OUTPUT_FORMAT) + notes + OUTPUT_FORMAT if notes else prompt


def s10_noted(messages):
    """Whether a conversation's program prompt (messages[0]) carries the S10 notes block, under T1_S10: the crash ask
    then asks to keep the decisions (S10_PATCH_KEEP). A fresh program's prompt never does."""
    head = S10_NOTES.split('{body}', 1)[0]
    return (s10_on() and bool(messages) and isinstance(messages[0].get('content'), str)
            and head in messages[0]['content'])


def with_s10_keep(ask):
    """A crash ask with S10_PATCH_KEEP before its PATCH_FORMAT."""
    return ask.removesuffix(PATCH_FORMAT) + S10_PATCH_KEEP + ' ' + PATCH_FORMAT if ask.endswith(PATCH_FORMAT) else ask


_S10_TAG = re.compile(r'#\s*D(\d{1,2})\b')


def s10_tag_ids(code):
    """The decisions the code's comments name (`# D3`, `# D3 changed`), as numbers; a cut reply's code counts too."""
    try:
        comments = [token.string for token in tokenize.generate_tokens(io.StringIO(code).readline)
                    if token.type == tokenize.COMMENT]
    except (tokenize.TokenError, SyntaxError, ValueError):
        comments = [line[line.index('#'):] for line in code.splitlines() if '#' in line]
    return {int(n) for comment in comments for n in _S10_TAG.findall(comment)}


def s10_tags(code):
    """The number of distinct decisions the program's comments name (`# D3`, `# D3 changed`): a metric only."""
    return len(s10_tag_ids(code))


def s10_decide_ask(prompt, triage):
    """(DECIDE's ask, whether the prompt offers the library): the task prompt without its output format, then
    S10_DECIDE with the tier's word cap and limit on lines and quotes (S10_DECIDE_LIMIT), the FAMILY line under
    T1_ROUTE, the TEMPLATE item for a template unit, and the library item where the library is offered (plan_notes'
    test): S10_LIBRARY_MAP_ADOPT under T1_ADOPT, S10_LIBRARY_MAP under T1_LIBHINT, PLAN_LIBRARY otherwise. Under
    T1_S11_CONV the D line may end in default (S11_CONV_FIELD), the word cap is s10_decide_words' and the limit
    S11_CONV_LIMIT's, followed by S11_CONV_DEFAULTS (and under T1_S12_MECH S12_CONV_MORE). Under T1_S11_RAISE,
    S11_DECIDE in place of S10_DECIDE (its rule for missing data and its time grids); the two compose (S10_FIELD is
    outside S11_DECIDE_FROM)."""
    adopt = adopt_on()
    offered = (LIBRARY_BLOCK_ADOPT if adopt else LIBRARY_BLOCK).split('`')[0] + f'`import {LIBRARY}`' in prompt
    library = ((S10_LIBRARY_MAP_ADOPT if adopt else S10_LIBRARY_MAP if libhint_on() else PLAN_LIBRARY.strip())
               + '\n' if offered else '')
    family = S10_FAMILY.format(labels='; '.join(label for _, label, _ in families.FAMILIES)) if route_on() else ''
    tier = triage['tier']
    conv = s11_conv_on()
    limit = S11_CONV_LIMIT[tier] + '\n' + S11_CONV_DEFAULTS if conv else S10_DECIDE_LIMIT[tier]
    if conv and s12_mech_on():  # T1_S12_MECH: more defaults, after T1_S11_CONV's
        limit += '\n' + S12_CONV_MORE
    if runtime.enabled() and runtime.PROMPT in prompt:
        limit += '\n' + runtime.PLAN_CHECK
    decide = S11_DECIDE if s11_raise_on() else S10_DECIDE
    decide = decide.replace(S10_FIELD, S11_CONV_FIELD) if conv else decide
    ask = decide.format(words=s10_decide_words(tier), family=family, limit=limit,
                        template=S10_TEMPLATE if triage['template'] else '', library=library)
    return prompt.removesuffix(OUTPUT_FORMAT) + ask, offered


def _s10_body(answer, parsed, cut):
    """The text a DECIDE answer with too few D lines stands on as the decisions: its text without code and, when cut,
    without what the cut ended (_s10_whole); None when that holds no decision, with no complete D line and under
    S10_BODY_MIN characters once code is removed (a lead-in to a program, or the header lines alone)."""
    body = _s10_whole(answer, cut)
    prose = '\n'.join(line for line in body.split('\n') if line.strip() != PLAN_CODE_CUT).strip()
    return body[:PLAN_CHARS] if parsed['order'] or len(prose) >= S10_BODY_MIN else None


def _s10_take(s10, parsed, answer, body=None, *, cut=False, map_sources=None):
    """DECIDE's (or DISTILL's) parsed answer into the carried state; body (too few D lines: _s10_body) is then the
    decisions, and CHECK's lines are added after it."""
    s10.update(outputs=parsed['outputs'], open=parsed['open'], state=parsed['state'], full=answer, notes=True,
               decisions={} if body else dict(parsed['decisions']), order=[] if body else list(parsed['order']),
               body=body or None, map_sources=[(answer, cut)] if map_sources is None else map_sources)
    if route_on():
        s10['family'] = route_family(answer)


def _s10_fallback(client, prompt, deadline, status, pace, s10, reason, canaries, sleep):
    """DECIDE left no notes (reason: fallback_plan or failed): stack9's plan (plan_notes without facts, which gates
    itself on the clock) for the with_plan path, with thinking off (low effort is as unpromised officially as the
    budget); s10['plan'] holds it, s10['full'] its whole text for ADOPT."""
    status['s10_decide'] = reason
    full = [] if adopt_on() or route_on() else None
    s10['plan'] = plan_notes(client, prompt, deadline, status, pace, canaries=canaries, sleep=sleep, facts=False,
                             full=full, thinking=False)
    s10['full'] = full[0] if full else ''
    return s10


def _s10_stopped(exc, tried=False):
    """The status of a note step whose request failed, or was not sent: once an attempt went out (tried), a retry the
    request cap or the clock then refused is a failure too, not a skip."""
    reason = exc.reason if isinstance(exc, _Stop) and not tried else None
    return ('skipped_requests' if reason == 'request_cap' else
            'skipped_time' if reason in ('time_gate', 'think_gate') else 'failed')


def _s10_tried(client, before):
    """Whether an attempt went out since the client held before request records (one that failed included)."""
    return len(client.diagnostics['requests']) > before


def s10_decide(client, prompt, triage, deadline, status, pace, *, canaries=(), sleep=None, replay=_NO_REPLAY):
    """T1_S10's first note request (spec §2.1) on the task prompt (first_prompt's): DECIDE, with the tier's budget and
    word cap when its gate passes (_s10_gate; until a reply is measured the pace reads at most S10_PRIOR_RATE). A
    finished answer with S10_D_MIN or more D lines stands (ok), a cut one keeps its complete lines (cut_kept), and an
    answer with fewer becomes the decisions as text, without code or the line a cut ended (no_lines, _s10_body). An
    answer that went on reasoning in its text (s10_spill) gets the spill-over DISTILL (_s10_spill) first, whose reply
    replaces it with S10_D_MIN D lines or more, however many the answer kept (distilled; s10_decide_spill says how it
    went), keeping the answer's LIBRARY MAP lines, FAMILY, OUTPUTS and STATE where the new reply gives none. Under
    T1_ADOPT, a DECIDE cut before any LIBRARY MAP line gets those lines from plan_map. Without answer text, or with
    one that decides nothing (a program, the header lines alone), DISTILL (the last S10_DISTILL_TAIL characters
    of its reasoning, thinking off) or, with less reasoning than S10_DISTILL_MIN characters, the same ask with
    thinking off (retried); either stands with one D line or more (DISTILL's word cap is s10_distill_words'). A gate
    that fails at the floor or a DISTILL or resend without D lines (fallback_plan), or a failed request (failed),
    gets stack9's plan instead (_s10_fallback); under T1_S12_MECH a request that went out and failed leaves no plan
    (failed; _s12_note_failed), and nothing after it in the notes stage is sent.
    status: s10, the triage, s10_prompt_sha, s10_decide, s10_decide_budget, s10_rate_after_decide, s10_decide_spill
    (S10_SPILL_STATUSES; none without a spill-over DISTILL), s10_distill_spill. Returns the carried state (spec §4),
    never raises a request error."""
    started = CLOCK.now()
    if not pace.measured:  # gates, timeouts and think_fits all read the prior until a reply is measured
        pace.rate = min(pace.rate, S10_PRIOR_RATE)
    tier = triage['tier']
    s10 = {'tier': tier, 'stateful': triage['stateful'], 'started': started, 'outputs': None, 'decisions': {},
           'order': [], 'body': None, 'checked': None, 'open': [], 'state': None, 'state_lines': [], 'full': '',
           'plan': None, 'notes': False, 'reasoning': 0.0, 'budget': 0, 'family': None, 'kept': [], 'block': ''}
    s10_triage_status(status, triage)
    status.update(s10_decide_budget=0, s10_decide_spill='none')
    ask, offered = s10_decide_ask(prompt, triage)
    ask, words = withhold_canaries(ask, canaries), s10_decide_words(tier)
    budget = _s10_gate('decide', s10_budget('decide', tier, status), words, 0, deadline, pace)
    if budget is None:
        return _s10_fallback(client, prompt, deadline, status, pace, s10, 'fallback_plan', canaries, sleep)
    s10['budget'], before = budget, len(client.diagnostics['requests'])
    try:
        reply = _s10_send(client, [{'role': 'user', 'content': ask}], 's10_decide', budget, words, 0, deadline, pace,
                          sleep)
    except (HouseError, _Stop):
        tried = _s10_tried(client, before)
        status['s10_decide_budget'] = budget if tried else 0  # 0: not sent
        _s12_note_failed(client, status, 's10_decide', tried)
        if _s12_notes_ended(client, status):  # T1_S12_MECH: no fallback plan after a request that went unanswered
            status['s10_decide'] = 'failed'
            return s10
        return _s10_fallback(client, prompt, deadline, status, pace, s10, 'failed', canaries, sleep)
    status['s10_decide_budget'] = budget
    pace.measure(client, reply)
    status['s10_rate_after_decide'] = round(pace.rate, 1)
    s10['reasoning'] = _s10_reasoning(client.diagnostics['requests'][-1])
    answer, cut = _s10_answer(reply, canaries), reply['finish_reason_raw'] == 'length'
    if answer:
        replay.put('s10-decide.txt', answer)
    parsed = s10_parse(answer, cut)
    lined = len(parsed['order']) >= S10_D_MIN
    body = None if lined or not answer else _s10_body(answer, parsed, cut)
    if lined or body:
        state = ('cut_kept' if cut else 'ok') if lined else 'no_lines'
        spilled = (_s10_spill(client, ask, answer, 'decide', words, 0, deadline, status, pace, canaries, sleep, replay)
                   if s10_spill(answer, len(parsed['order'])) else None)
        again = s10_parse(*spilled) if spilled else None
        used = again is not None and _s10_spill_used(status, 'decide', len(again['order']), S10_D_MIN)
        first = answer
        map_sources = [(answer, cut)]
        if spilled:
            status['s10_rate_after_decide'] = round(pace.rate, 1)
        if used:  # its lines replace the reply's; the reply's map lines, FAMILY, OUTPUTS and STATE fill its gaps
            maps = [] if _map_lines(spilled[0]) else _map_lines(first)
            map_sources = [spilled] + (map_sources if maps else [])
            answer, cut = spilled[0] + ('\nLIBRARY MAP:\n' + '\n'.join(maps) if maps else ''), spilled[1]
            parsed = dict(again, outputs=again['outputs'] or parsed['outputs'], state=again['state'] or parsed['state'])
            state, body = 'distilled', None
        _s10_take(s10, parsed, answer, body, cut=cut, map_sources=map_sources)
        if used and route_on() and s10['family'] is None:
            s10['family'] = route_family(first)
        if adopt_on() and offered and cut and not _map_lines(answer) and not _s12_notes_ended(client, status):
            rate = _s10_rate(pace)  # spec §7.2: the answer's cost and R_code, as DISTILL
            lines = plan_map(client, ask, answer, deadline, status, pace, canaries=canaries, sleep=sleep,
                             need=_s10_cost(S10_TPW * words, rate), reserve=_s10_reserve(rate))
            s10['full'] += '\nLIBRARY MAP' + lines if lines and route_on() else lines
    else:
        reasoning = withhold_canaries(reply.get('reasoning') or '', canaries).strip()
        distill = len(reasoning) >= S10_DISTILL_MIN
        cap = s10_distill_words(tier)  # T1_S11_CONV: DECIDE's where larger
        kind, state, asked = (('s10_distill', 'distilled', cap) if distill else ('s10_decide_retry', 'retried', words))
        content = (ask + '\n\n' + S10_DISTILL.format(tail=reasoning[-S10_DISTILL_TAIL:], words=cap)
                   if distill else ask)
        rate, before = _s10_rate(pace), len(client.diagnostics['requests'])
        try:  # thinking off; gated at the answer's cost only
            reply = _request(client, [{'role': 'user', 'content': withhold_canaries(content, canaries)}], deadline,
                             pace, kind, sleep, need=_s10_cost(S10_TPW * asked, rate), reserve=_s10_reserve(rate),
                             keep=S10_KEEP, timeout_floor=S10_NOTE_TIMEOUT_FLOOR, note=True,
                             expect=_s12_expect(0, asked, rate))
        except (HouseError, _Stop):
            _s12_note_failed(client, status, kind, _s10_tried(client, before))
            if _s12_notes_ended(client, status):  # T1_S12_MECH: no fallback plan after a request that went unanswered
                status['s10_decide'] = 'failed'
                return s10
            return _s10_fallback(client, prompt, deadline, status, pace, s10, 'failed', canaries, sleep)
        pace.measure(client, reply)
        status['s10_rate_after_decide'] = round(pace.rate, 1)
        answer = _s10_answer(reply, canaries)
        replay.put('s10-distill.txt' if distill else 's10-decide.txt', answer)
        cut = reply['finish_reason_raw'] == 'length'
        parsed = s10_parse(answer, cut)
        if not parsed['order']:
            return _s10_fallback(client, prompt, deadline, status, pace, s10, 'fallback_plan', canaries, sleep)
        _s10_take(s10, parsed, answer, cut=cut)
    status['s10_decide'] = state
    return s10


def s11_union_on() -> bool:
    """T1_S11_UNION=1 under T1_S10: a second DECIDE sample merged into the first's lines (s11_decide_b)."""
    return s10_on() and os.environ.get('T1_S11_UNION') == '1'


def s11_union_cap():
    """The D lines T1_S11_UNION's merged notes carry: S11_D_LINES, B's room past A's parse cap (S11_D_LINES -
    S10_D_LINES) kept where T1_S11_CONV raises that cap (s10_d_lines), so neither switch takes the other's lines."""
    return S11_D_LINES - S10_D_LINES + s10_d_lines()


def s11_decide_b_ask(prompt, triage):
    """DECIDE B's ask: DECIDE's (s10_decide_ask, same format, caps and word count) with its D lines asked in a
    reviewer's order, output by output (S11_ORDER_B in place of S11_ORDER_A; S11_ORDER_B_L on the L tier, whose limit
    skips routine output handling), the task prompt left as it is."""
    ask, _ = s10_decide_ask(prompt, triage)
    head = prompt.removesuffix(OUTPUT_FORMAT)
    return head + ask[len(head):].replace(S11_ORDER_A, S11_ORDER_B_L if triage['tier'] == 'L' else S11_ORDER_B, 1)


def _s11_words(text):
    return set(re.findall(r'[a-z0-9]+', text.lower())) - set(_S11_STOP)


def _s11_figures(text):
    """[(start, end, value)] of a text's stand-alone figures, never the digit of a name (S0, d1, r_0, T2): a percent
    (not `x% of`) and basis points as fractions (_S11_FIGURE), each value rounded to 12 places."""
    found = []
    for match in _S11_FIGURE.finditer(text):
        unit, value = (match[2] or '').lower(), float(match[1])
        value = value / 10000 if unit.startswith('b') else value / 100 if unit and not match[3] else value
        found.append((match.start(), match.end(), round(value, 12)))
    return found


def _s11_numbers(text):
    """The stand-alone figures of a text as values (_s11_figures)."""
    return {value for _, _, value in _s11_figures(text)}


def _s11_sums(text):
    """The values a + b and a - b of two figures written next to each other (`1 + 0.001`, `(1 - 0.5)`): one value
    written as a sum, which _s11_differs takes as written too."""
    figures, found = _s11_figures(text), set()
    for (_, end, first), (start, _, second) in zip(figures, figures[1:]):
        sign = re.fullmatch(r'\s*([+-])\s*\(?\s*', text[end:start])
        if sign:
            found.add(round(first + second if sign[1] == '+' else first - second, 12))
    return found


def _s11_overlap(a, b):
    return len(a & b) / len(a | b) if a and b else 0.0


def _s11_sentences(instruction):
    """The TASK's sentences, normalized (_s10_norm): split at a blank line, a list item, a heading or a sentence end;
    a line wrapped inside a sentence is joined."""
    text = re.sub(r'\n[ \t]*(?=[-*+][ \t]|\d{1,2}[.)][ \t]|#)', '\n\n', instruction)
    return [_s10_norm(part) for part in re.split(r'\n[ \t]*\n|(?<=[.!?])\s+(?=[A-Z0-9`*_"(])', text)
            if part.strip()]


def _s11_place(quote, sentences):
    """(index of the TASK sentence holding the quote or None, the quote normalized: its longest piece between
    ellipses, without surrounding quotes and punctuation)."""
    pieces = re.split(r'\.\.\.|\u2026', _s10_norm(quote or ''))
    text = max(pieces, key=len).strip(' "\'.,;:!?()[]')
    if len(text) < S11_QUOTE_MIN:
        return None, text
    return next((index for index, sentence in enumerate(sentences) if text in sentence), None), text


def _s11_line(value, sentences, instruction):
    """A D line's parts as s11_union compares them: quantity and decision words, the decision's numbers (and the sums
    of figures written next to each other) and text, its last field, its quote's TASK sentence and normalized text,
    and whether that quote is TASK words (audit._quoted)."""
    fields = [field.strip() for field in value.split('|')]
    quantity, decision, source = ((fields[0], '; '.join(fields[1:-1]), fields[-1]) if len(fields) >= 3 else
                                  (fields[0], fields[1], '') if len(fields) == 2 else ('', value, ''))
    quote = _s10_quote(value) if len(fields) >= 2 else None
    sentence, text = _s11_place(quote, sentences)
    return {'quantity': _s11_words(quantity), 'decision': _s11_words(decision), 'text': decision,
            'numbers': _s11_numbers(decision), 'sums': _s11_sums(decision), 'source': source, 'sentence': sentence,
            'quote': text if quote else '', 'quoted': bool(quote) and _quoted(quote, instruction)}


def _s11_same_sentence(a, b):
    """Whether two lines quote one TASK sentence: both placed in it, or (either not placed) the same quote, or one
    quote inside the other."""
    if not (a['quote'] and b['quote']):
        return False
    if a['sentence'] is not None and b['sentence'] is not None:
        return a['sentence'] == b['sentence']
    short, long = sorted((a['quote'], b['quote']), key=len)
    return short == long or (len(short) >= S11_QUOTE_MIN and short in long)


def _s11_differs(a, b):
    """Whether two readings of one item contradict each other: opposite poles of a convention (S11_POLES), numbers
    where neither reading's set holds the other's (one more specific than the other is not a contradiction; a
    percent, basis points and the sum of two figures count as the value they write: _s11_figures, _s11_sums), or a
    negation (S11_NEGATION) in one alone where their decision words overlap at S11_NEGATED or more."""
    words, other, mine, theirs = a['decision'], b['decision'], a['numbers'], b['numbers']
    poles = any((x in words) != (y in words) and (x in words) == (y in other) and (x in other) != (y in other)
                for x, y in S11_POLES)
    numbers = bool(mine and theirs) and not (mine <= theirs | b['sums'] or theirs <= mine | a['sums'])
    negation = (bool(set(S11_NEGATION) & words) != bool(set(S11_NEGATION) & other)
                and _s11_overlap(words, other) >= S11_NEGATED)
    return poles or numbers or negation


def _s11_split(value):
    """(the part of a D line that may name other D lines, its last field with its `|`: the TASK quote or free, never
    renumbered); a line or reading of fewer than three fields is all the first part."""
    head, bar, last = value.rpartition('|')
    return (head, bar + last) if value.count('|') >= 2 else (value, '')


def _s11_refs(value, other):
    """The IDs of B's lines (other) that a line or reading names (`from D3`); a D<n> that is none of them is a name
    (a D0 dividend), not a reference."""
    return {int(n) for n in _S11_REF.findall(_s11_split(value)[0])} & set(other['decisions'])


def _s11_renumber(value, other, known):
    """value with each D<n> naming one of B's lines (other) as that line's ID in the merge (known)."""
    head, tail = _s11_split(value)
    return _S11_REF.sub(lambda m: f'D{known[int(m[1])]}' if int(m[1]) in other['decisions'] else m[0], head) + tail


def _s11_stands(value, other):
    """Whether a B line or reading says something once it follows A's lines: not one that only points back at its
    neighbours in its own reply (_S11_BACK) without naming a D line."""
    return not (_S11_BACK.search(_s11_split(value)[0]) and not _s11_refs(value, other))


def _s11_call_bindings(text):
    """Syntactic (callee, keyword, value AST) bindings in bounded, closed calls in a decision, never its source.
    Tokenization excludes calls quoted as strings or in comments. No expression is executed. Dynamic callees,
    argument expansion and duplicate keywords are not comparable; complex values are skipped per keyword so a
    comprehension does not erase another keyword's scalar binding. A binding is not evidence it obeys the TASK.
    """
    if len(text) > 2 * S10_D_CHARS:
        return set()
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(text.strip()).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError, ValueError, RecursionError, MemoryError):
        return set()
    if any(t.type == tokenize.ERRORTOKEN and t.string in ('"', "'") for t in tokens):
        return set()
    text = text.strip()
    lines, starts = text.splitlines(keepends=True), [0]
    for line in lines:
        starts.append(starts[-1] + len(line))
    safe = (ast.Constant, ast.Name, ast.Load, ast.Attribute, ast.Subscript, ast.Slice,
            ast.UnaryOp, ast.UAdd, ast.USub, ast.BinOp, ast.Add, ast.Sub, ast.Mult, ast.Div,
            ast.FloorDiv, ast.Mod, ast.Pow, ast.Tuple, ast.List, ast.Set, ast.Dict)
    found, i, calls = set(), 0, 0
    while i < len(tokens) and calls < 16:
        if tokens[i].type == getattr(tokenize, 'FSTRING_START', None):
            depth = 1  # Python 3.12+ tokenizes interpolation code; none of this string is a call specimen.
            i += 1
            while i < len(tokens) and depth:
                depth += (tokens[i].type == tokenize.FSTRING_START) - (tokens[i].type == tokenize.FSTRING_END)
                i += 1
            continue
        if tokens[i].type == tokenize.NAME and tokens[i].string == 'lambda':
            break  # No standalone lambda body is a decision call; keywords inside a closed call are below.
        if tokens[i].type != tokenize.NAME or (i and tokens[i - 1].string == '.'):
            i += 1
            continue
        j = i + 1
        while j + 1 < len(tokens) and tokens[j].string == '.' and tokens[j + 1].type == tokenize.NAME:
            j += 2
        if j >= len(tokens) or tokens[j].string != '(':
            i += 1
            continue
        depth, end = 0, None
        for k in range(j, len(tokens)):
            token = tokens[k]
            if token.type != tokenize.OP:
                continue
            if token.string in ('(', '[', '{'):
                depth += 1
            elif token.string in (')', ']', '}'):
                depth -= 1
                if depth == 0:
                    end = k
                    break
        if end is None:
            break
        start, stop = tokens[i].start, tokens[end].end
        candidate = text[starts[start[0] - 1] + start[1]:starts[stop[0] - 1] + stop[1]]
        declaration = i and tokens[i - 1].type == tokenize.NAME and tokens[i - 1].string in ('def', 'class')
        i, calls = end + 1, calls + 1
        if declaration:  # Also async def; do not reinterpret calls inside a declaration header's defaults.
            continue
        try:
            node = ast.parse(candidate, mode='eval').body
        except (SyntaxError, ValueError, RecursionError, MemoryError):
            continue
        if not isinstance(node, ast.Call) or len(list(ast.walk(node))) > 200:
            continue
        callee, tail = node.func, []
        while isinstance(callee, ast.Attribute):
            tail.append(callee.attr)
            callee = callee.value
        if not isinstance(callee, ast.Name) or any(isinstance(arg, ast.Starred) for arg in node.args):
            continue
        names = [kw.arg for kw in node.keywords]
        if None in names or len(set(names)) != len(names):
            continue
        name = '.'.join([callee.id] + list(reversed(tail)))
        for kw in node.keywords:
            nodes = list(ast.walk(kw.value))
            if (all(isinstance(part, safe) for part in nodes)
                    and not any(isinstance(part, ast.Dict) and None in part.keys for part in nodes)):
                found.add((name, kw.arg, ast.dump(kw.value, include_attributes=False)))
    return found


def _s11_binding_addition(mine, other):
    """A same-quantity B decision has closed-call keyword information A does not express; not a semantic verdict."""
    return (_s11_overlap(mine['quantity'], other['quantity']) >= S11_SAME_QUANTITY
            and bool(_s11_call_bindings(other['text']) - _s11_call_bindings(mine['text'])))



def _s11_binding_reading(value, other, known):
    """Renumber only bare D tokens in a binding alternative; literals, comments and full f-strings stay exact.
    Attribute-chain names, static callees and keyword/assignment keys keep their spelling, not a decision's ID.
    None means tokenization or the reference contract is unsafe. Sources are appended separately by the caller.
    This is only for supplemental bindings; genuine conflicts keep their original reference rules.
    """
    if len(value) > 2 * S10_D_CHARS:
        return None
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(value).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError, ValueError, RecursionError, MemoryError):
        return None
    if any(t.string != '`' and ((t.type == tokenize.ERRORTOKEN and not t.string.isspace())
                               or (t.type == tokenize.OP and t.string not in tokenize.EXACT_TOKEN_TYPES))
           for t in tokens):
        return None  # Backticks are permitted prose markup; broken string delimiters and other errors are not.
    starts = [0]
    for line in value.splitlines(keepends=True):
        starts.append(starts[-1] + len(line))
    def index(position):
        return starts[position[0] - 1] + position[1]
    trivia = (tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.ENDMARKER)
    significant = [n for n, token in enumerate(tokens) if token.type not in trivia]
    neighbours = {n: (tokens[significant[k - 1]].string if k else '',
                      tokens[significant[k + 1]].string if k + 1 < len(significant) else '')
                  for k, n in enumerate(significant)}
    refs, changes, plain, i = set(), [], list(value), 0
    while i < len(tokens):
        token, end = tokens[i], tokens[i].end
        protected = token.type in (tokenize.STRING, tokenize.COMMENT)
        if token.type == getattr(tokenize, 'FSTRING_START', None):
            protected, depth = True, 1
            while depth and i + 1 < len(tokens):
                i += 1
                depth += (tokens[i].type == tokenize.FSTRING_START) - (tokens[i].type == tokenize.FSTRING_END)
            if depth:
                return None
            end = tokens[i].end
        if protected:
            start, stop = index(token.start), index(end)
            plain[start:stop] = ' ' * (stop - start)  # Only a mask for the back-reference check, never output.
        elif token.type == tokenize.NAME:
            match = _S11_REF.fullmatch(token.string)
            before, after = neighbours[i]
            identity = before == '.' or after in ('.', '(', '=')  # Attributes, callees and keyword/assignment keys.
            if match and not identity and int(match[1]) in other['decisions']:
                ref = int(match[1])
                refs.add(ref)
                if ref in known:
                    changes.append((index(token.start), index(token.end), f'D{known[ref]}'))
        i += 1
    if not refs <= set(known) or (_S11_BACK.search(''.join(plain)) and not refs):
        return None
    for start, stop, replacement in reversed(changes):
        value = value[:start] + replacement + value[stop:]
    return value


def s11_union(s10, other, instruction):
    """DECIDE B's parsed lines (other) merged into the carried state, in B's order (deep dive §4); every A line
    stands. A B line decides the same item as the first A line that quotes the same TASK sentence and shares
    S11_QUANTITY of its quantity words (Jaccard), or, B quoting TASK words (audit._quoted), the first A line without
    such a quote that shares S11_SAME_QUANTITY: where the two contradict each other (_s11_differs), A's line is
    marked once with B's reading (S11_UNSURE; a conflict), and B's line is covered either way. A B line without TASK
    words that shares S11_SAME_QUANTITY with an A line is covered too (A's stands), and so is one with S11_COVER of
    its quantity and decision words in an A line (or in a line of A's text, when DECIDE gave too few D lines). The
    rest are appended with the next free IDs, up to s11_union_cap() lines in all (s10['d_cap'], which the notes read).
    Each B line and reading must stand in the merge's numbering: one that only points back (_S11_BACK) without naming
    a D line is left out, a D<n> naming a B line becomes that line's ID in the merge (the A line that decides or
    covers it, or its new ID; _s11_renumber), and one naming a B line with no ID there (left out, or covered by A's
    text alone) is left out. Covered same-quantity calls with additional explicit keyword bindings are retained
    as unverified UNSURE alternatives, even without TASK quotes, after genuine conflicts. A stands; neither
    argument meaning nor B's source is verified. An alternative over the reading cap is skipped whole.
    s11_binding_alternatives/skipped count retained/omitted binding alternatives, not conflicts or correctness.
    A's OUTPUTS and STATE stand unless A gave none. Returns (lines added, genuine conflicts)."""
    sentences = _s11_sentences(instruction)
    s10.update(s11_binding_alternatives=0, s11_binding_skipped=0)
    s10['d_cap'] = cap = s11_union_cap()
    s10['outputs'] = s10['outputs'] or other['outputs']
    s10['state'] = s10['state'] or other['state']
    ours = [(n, _s11_line(s10['decisions'][n], sentences, instruction)) for n in s10['order']]
    text = [_s11_line(line, sentences, instruction) for line in (s10['body'] or '').split('\n') if line.strip()]
    mapped, ids, marks, new = dict(ours), {}, [], []  # ids: a B line's ID -> the A line that decides or covers it
    for key in other['order']:
        value = other['decisions'][key]
        line = _s11_line(value, sentences, instruction)
        same = next((n for n, mine in ours if _s11_same_sentence(mine, line)
                     and _s11_overlap(mine['quantity'], line['quantity']) >= S11_QUANTITY), None)
        close = next((n for n, mine in ours if not (mine['quoted'] and line['quoted'])
                      and _s11_overlap(mine['quantity'], line['quantity']) >= S11_SAME_QUANTITY), None)
        if same is None and line['quoted']:
            same = close  # B quotes TASK words where A's line on the item quotes none
        if same is not None:
            ids[key] = same
            if _s11_differs(mapped[same], line):
                marks.append((same, line, False))
            elif _s11_binding_addition(mapped[same], line):
                marks.append((same, line, True))
            continue
        words = line['quantity'] | line['decision']
        cover = close if close is not None else next(
            (n for n, mine in ours if _s11_overlap(mine['quantity'] | mine['decision'], words) >= S11_COVER), None)
        if cover is not None:
            ids[key] = cover
            if _s11_binding_addition(mapped[cover], line):
                marks.append((cover, line, True))
        elif not any(_s11_overlap(mine['quantity'] | mine['decision'], words) >= S11_COVER for mine in text):
            new.append(key)
    new = [key for key in new if _s11_stands(other['decisions'][key], other)]
    while True:  # one left out may be one that another names
        kept = new[:max(0, cap - len(s10['order']))]
        start = max(s10['order'], default=0) + 1
        known = ids | {key: start + index for index, key in enumerate(kept)}
        dropped = {key for key in kept if not _s11_refs(other['decisions'][key], other) <= set(known)}
        if not dropped:
            break
        new = [key for key in new if key not in dropped]
    for key in kept:
        s10['decisions'][known[key]] = _s11_renumber(other['decisions'][key], other, known)
        s10['order'].append(known[key])
    marked, conflicts = set(), 0
    for same, line, binding in sorted(marks, key=lambda item: item[2]):
        if binding:
            reading = None if same in marked else _s11_binding_reading(line['text'], other, known)
            if reading is None:
                _count(s10, 's11_binding_skipped')
                continue
        else:
            if (same in marked or not _s11_stands(line['text'], other)
                    or not _s11_refs(line['text'], other) <= set(known)):
                continue
            reading = _s11_renumber(line['text'], other, known)
        # B's TASK words where A quotes others; no `|`, so the line keeps A's last field
        reading += '' if line['quote'] == mapped[same]['quote'] else f' per {line["source"]}'
        if binding and len(reading) > S11_BINDING_READING_CHARS:
            _count(s10, 's11_binding_skipped')
            continue
        s10['decisions'][same] += S11_UNSURE.format(reading=reading if binding else reading[:S11_READING_CHARS])
        if binding:
            _count(s10, 's11_binding_alternatives')
        else:
            conflicts += 1
        marked.add(same)
    return len(kept), conflicts


def _s11_reserve(rate):
    """The seconds DECIDE B's gate keeps past R_code for the repair loop: S11_REPAIR_ROUNDS rounds of a patch and a
    rerun at the speed repair's reserve (Pace.patch() + EXEC_FLOOR + 10), at rate."""
    return S11_REPAIR_ROUNDS * ((2500 / rate + 6) + EXEC_FLOOR + 10)


def s11_decide_b(client, prompt, triage, s10, deadline, status, pace, *, instruction='', canaries=(), sleep=None,
                 replay=_NO_REPLAY):
    """T1_S11_UNION's DECIDE B, right after DECIDE on the same task prompt, when DECIDE left notes (not a fallback
    plan): one request (s11_decide_b) of s11_decide_b_ask, sequential, with DECIDE's tier budget and word cap and
    so the token rule, strict on _s10_gate as every note step (skipped below the floor, never sent unreasoned),
    leaving S10_KEEP requests, skipped once the breaker has tripped, its reply feeding the breaker. Its gate keeps
    S11_REPAIR_ROUNDS repair rounds past R_code (_s11_reserve) and, on a stateful unit (as s10_refine decides it),
    STATE at its full budget. A reply with S10_D_MIN D lines or more that did not spill over (s10_spill) is merged
    (s11_union); otherwise A's notes stand as they were.
    status: s11_decide_b (S11_STATUSES), s11_b_budget (0: not sent), s11_b_lines, s11_lines_added, s11_conflicts,
    s11_binding_alternatives and s11_binding_skipped (syntactic alternatives, not correctness).
    Returns s10; never raises a request error."""
    status.update(s11_decide_b='no_notes', s11_b_budget=0, s11_b_lines=0, s11_lines_added=0, s11_conflicts=0,
                  s11_binding_alternatives=0, s11_binding_skipped=0)
    if not s10.get('notes'):
        return s10
    if getattr(client, 's10_think_off', False) is True:
        status['s11_decide_b'] = 'skipped_breaker'
        return s10
    if _s12_notes_ended(client, status):  # T1_S12_MECH: an earlier note request ended without a reply
        status['s11_decide_b'] = 'skipped_time'
        return s10
    tier, rate = triage['tier'], _s10_rate(pace)
    words = s10_decide_words(tier)  # its ask's (s10_decide_ask): T1_S11_CONV's where that is on
    stateful = os.environ.get('T1_S10_STATE') != '0' and (
        s12_stateful(triage['state_hits'], s10['state'])[0] if s12_mech_on()  # T1_S12_MECH: DECIDE's no can hold
        else triage['stateful'] or s10['state'] == 'yes')
    later = _s11_reserve(rate) + (_s10_cost(s10_budget('state') + S10_TPW * S10_STATE_WORDS, rate) if stateful else 0)
    budget = _s10_gate('decide', s10_budget('decide', tier, status), words, later, deadline, pace)
    if budget is None:
        status['s11_decide_b'] = 'skipped_time'
        return s10
    ask, before = withhold_canaries(s11_decide_b_ask(prompt, triage), canaries), len(client.diagnostics['requests'])
    try:
        reply = _s10_send(client, [{'role': 'user', 'content': ask}], 's11_decide_b', budget, words, later, deadline,
                          pace, sleep)
    except (HouseError, _Stop) as exc:
        tried = _s10_tried(client, before)
        status.update(s11_decide_b=_s10_stopped(exc, tried), s11_b_budget=budget if tried else 0)
        _s12_note_failed(client, status, 's11_decide_b', tried)
        return s10
    status['s11_b_budget'] = budget
    pace.measure(client, reply)
    answer, cut = _s10_answer(reply, canaries), reply['finish_reason_raw'] == 'length'
    if answer:
        replay.put('s11-decide-b.txt', answer)
    parsed = s10_parse(answer, cut)
    status['s11_b_lines'] = len(parsed['order'])
    if not answer:
        status['s11_decide_b'] = 'failed'
    elif s10_spill(answer, len(parsed['order'])):  # drafts its prose went on to revise: no spill-over request
        status['s11_decide_b'] = 'spilled'
    elif len(parsed['order']) < S10_D_MIN:
        status['s11_decide_b'] = 'no_lines'
    else:
        added, conflicts = s11_union(s10, parsed, instruction)
        status.update(s11_decide_b='cut_kept' if cut else 'ok', s11_lines_added=added, s11_conflicts=conflicts,
                      s11_binding_alternatives=s10['s11_binding_alternatives'],
                      s11_binding_skipped=s10['s11_binding_skipped'])
    return s10


def s10_adopt(prompt, s10, instruction, status):
    """T1_ADOPT after DECIDE (or its fallback plan), as the stack9 block in solve: the REQUIRED LIBRARY CALLS the
    whole answer's LIBRARY MAP names (library_map) in the prompt CHECK, STATE and the program read, with the adopt_*
    status. (prompt, required)."""
    if not adopt_on() or not s10.get('full'):
        return prompt, []
    counts = {}
    required, dropped = library_map(s10['full'], instruction, matched_functions([{'content': prompt}]), counts)
    status.update(adopt_required=len(required), adopt_none_dropped=dropped,
                  adopt_none_different=counts.get('different', 0))
    return with_required(prompt, required), required


def s10_check_mode():
    """T1_S10_CHECK: auto (CHECK on its trigger), always, or never; unset or any other value is S10_CHECK_DEFAULT
    (never, from v1.1)."""
    mode = os.environ.get('T1_S10_CHECK')
    return mode if mode in S10_CHECK_MODES else S10_CHECK_DEFAULT


def _s10_check_trigger(s10, status):
    """Why CHECK runs (spec §2.2), the first that holds: DECIDE cut or retried, an OPEN point, an UNSURE line, a free
    line, or DECIDE's reasoning at S10_CHECK_USED of its budget (T1_S10_CHECK=auto); always under always; None
    otherwise, and under never."""
    lines = [s10['decisions'][n] for n in s10['order']]
    mode = s10_check_mode()
    if mode != 'auto':
        return 'always' if mode == 'always' else None
    return ('cut' if status.get('s10_decide') in ('cut_kept', 'no_lines', 'distilled', 'retried') else
            'open' if s10['open'] else
            'unsure' if any(map(_s10_unsure, lines)) else
            'free' if any(map(_s10_free, lines)) else
            'budget' if s10['budget'] and s10['reasoning'] >= S10_CHECK_USED * s10['budget'] else None)


def _s10_rest(answer, cut):
    """A CHECK reply's text as CHECKED keeps it: without code, OTHERS CONFIRMED or what a cut ended."""
    return '\n'.join(line for line in _s10_whole(answer, cut).split('\n')
                     if line.strip() and line.strip() != PLAN_CODE_CUT and not _S10_CONFIRMED.match(line)).strip()


def _s10_check(client, prompt, s10, deadline, status, pace, stateful, canaries, sleep, replay):
    trigger = _s10_check_trigger(s10, status)
    status.update(s10_check_trigger=trigger or 'none', s10_check_budget=0, s10_d_changed=0, s10_d_added=0,
                  s10_check_spill='none')
    if s10_check_mode() == 'never':
        status['s10_check'] = 'off'
        return
    if trigger is None:
        status['s10_check'] = 'not_triggered'
        return
    if getattr(client, 's10_think_off', False) is True:
        status['s10_check'] = 'skipped_breaker'
        return
    if _s12_notes_ended(client, status):  # T1_S12_MECH: an earlier note request ended without a reply
        status['s10_check'] = 'skipped_time'
        return
    rate = _s10_rate(pace)
    # On a stateful unit STATE outranks CHECK: CHECK's gate reserves STATE at its floor.
    later = _s10_cost(S10_FLOOR + S10_TPW * S10_STATE_WORDS, rate) if stateful else 0
    budget = _s10_gate('check', s10_budget('check', status=status), S10_CHECK_WORDS, later, deadline, pace)
    if budget is None:
        status['s10_check'] = 'skipped_time'
        return
    family = ''
    if route_on() and s10['family'] and not status.get('library_restricted'):
        functions = library_functions() if status.get('library_shown') else set()
        family = ROUTE_NOTES.format(label=families.LABELS[s10['family']], note=route_note(s10['family'], functions))
    ask = prompt.removesuffix(OUTPUT_FORMAT) + S10_CHECK.format(notes=_s10_text(s10, open_items=True), family=family,
                                                                words=S10_CHECK_WORDS)
    before = len(client.diagnostics['requests'])
    try:
        reply = _s10_send(client, [{'role': 'user', 'content': withhold_canaries(ask, canaries)}], 's10_check',
                          budget, S10_CHECK_WORDS, later, deadline, pace, sleep)
    except (HouseError, _Stop) as exc:
        tried = _s10_tried(client, before)
        status.update(s10_check=_s10_stopped(exc, tried), s10_check_budget=budget if tried else 0)
        _s12_note_failed(client, status, 's10_check', tried)
        return
    status['s10_check_budget'] = budget
    pace.measure(client, reply)
    answer, cut = _s10_answer(reply, canaries), reply['finish_reason_raw'] == 'length'
    replay.put('s10-check.txt', answer)
    parsed = s10_parse(answer, cut)
    spilled = (_s10_spill(client, withhold_canaries(ask, canaries), answer, 'check', S10_CHECK_WORDS, later, deadline,
                          status, pace, canaries, sleep, replay) if s10_spill(answer, len(parsed['order'])) else None)
    if spilled:
        again = s10_parse(*spilled)
        # OTHERS CONFIRMED without a D line changes nothing, whatever the reply writes before it (kept as CHECKED)
        confirms = not again['order'] and any(map(_S10_CONFIRMED.match, _s10_whole(*spilled).split('\n')))
        if _s10_spill_used(status, 'check', len(again['order']) + confirms, 1):
            (answer, cut), parsed = spilled, again
    # Without a D line, the text is kept as CHECKED: without code, OTHERS CONFIRMED, or what a cut ended.
    rest = _s10_rest(answer, cut)
    if parsed['order']:
        status['s10_d_changed'], status['s10_d_added'] = s10_merge(s10, parsed)
        status['s10_check'] = 'cut_kept' if cut else 'ok'
    elif rest:
        if not s11_conv_on():  # T1_S11_CONV drops it: probes pasted a refusal and a whole program as CHECKED
            s10['checked'] = rest[:S10_CHECKED_CHARS]
        status['s10_check'] = 'no_lines'
    else:
        status['s10_check'] = 'ok' if answer and not cut else 'failed'


def _s11_half_left(s10, deadline):
    """T1_S11_CONV: whether more than S11_CONV_HALF of the clock DECIDE began with (s10['started']) remains."""
    return deadline - CLOCK.now() > S11_CONV_HALF * (deadline - s10['started'])


def _s11_state_rescue(b_max, deadline, status, pace):
    """T1_S11_CONV: the budget of STATE after its gate failed while _s11_half_left holds, which reserves one full code
    reply and one run after STATE's own reply (_s11_rescue_reserve), not R_code and the gate's margin: the floor
    (S10_FLOOR, or b_max when lower) where think_fits allows it, sent strict (floor), else 0, sent with thinking off
    at the answer's cost as after a breaker trip (think_off). Sets s11_state_rescue."""
    rate, floor = _s10_rate(pace), min(S10_FLOOR, b_max)
    need, reserve = _s10_cost(S10_TPW * S10_STATE_WORDS, rate), _s11_rescue_reserve(rate)
    budget = floor if floor > 0 and think_fits(deadline, pace, floor, need, reserve, S10_MARGIN) else 0
    status['s11_state_rescue'] = 'floor' if budget else 'think_off'
    return budget


def _s11_rescue_reserve(rate):
    """What STATE's rescue, and its spill-over DISTILL, keep after their replies: one full code reply at rate
    (Pace.full's) and one run (EXEC_FLOOR)."""
    return _s10_cost(4000, rate) + EXEC_FLOOR


def _s10_state(client, prompt, s10, deadline, status, pace, canaries, sleep, replay):
    status.update(s10_state_budget=0, s10_state_spill='none')
    if s11_conv_on():
        status['s11_state_rescue'] = 'none'
    if _s12_notes_ended(client, status):  # T1_S12_MECH: an earlier note request ended without a reply (no rescue)
        status['s10_state'] = 'skipped_time'
        return
    rate = _s10_rate(pace)
    ask = prompt.removesuffix(OUTPUT_FORMAT) + S10_STATE.format(notes=_s10_text(s10), words=S10_STATE_WORDS)
    messages = [{'role': 'user', 'content': withhold_canaries(ask, canaries)}]
    budget, before = 0, len(client.diagnostics['requests'])
    try:
        if getattr(client, 's10_think_off', False) is True:  # after a breaker trip: thinking off, at answer cost
            need, reserve = _s10_cost(S10_TPW * S10_STATE_WORDS, rate), _s10_reserve(rate)
            if s11_conv_on() and deadline - CLOCK.now() < need + reserve and _s11_half_left(s10, deadline):
                reserve = _s11_rescue_reserve(rate)  # T1_S11_CONV: R_code does not fit, a code reply and a run do
                status['s11_state_rescue'] = 'think_off'
            reply = _request(client, messages, deadline, pace, 's10_state', sleep, need=need, reserve=reserve,
                             keep=S10_KEEP, note=True, expect=_s12_expect(0, S10_STATE_WORDS, rate))
        else:
            b_max = s10_budget('state', status=status)
            budget = _s10_gate('state', b_max, S10_STATE_WORDS, 0, deadline, pace)
            if budget is None and s11_conv_on() and _s11_half_left(s10, deadline):
                budget = _s11_state_rescue(b_max, deadline, status, pace)  # a time gate then reads skipped_time
                reply = _request(client, messages, deadline, pace, 's10_state', sleep,
                                 need=_s10_cost(S10_TPW * S10_STATE_WORDS, rate), reserve=_s11_rescue_reserve(rate),
                                 keep=S10_KEEP, budget=budget or None, strict=bool(budget), think_margin=S10_MARGIN,
                                 timeout_floor=S10_NOTE_TIMEOUT_FLOOR, note=True,
                                 expect=_s12_expect(budget, S10_STATE_WORDS, rate))
            elif budget is None:
                status['s10_state'] = 'skipped_time'
                return
            else:
                reply = _s10_send(client, messages, 's10_state', budget, S10_STATE_WORDS, 0, deadline, pace, sleep)
    except (HouseError, _Stop) as exc:
        tried = _s10_tried(client, before)
        status.update(s10_state=_s10_stopped(exc, tried), s10_state_budget=budget if tried else 0)
        _s12_note_failed(client, status, 's10_state', tried)
        return
    status['s10_state_budget'] = budget
    pace.measure(client, reply)
    answer, cut = _s10_answer(reply, canaries), reply['finish_reason_raw'] == 'length'
    replay.put('s10-state.txt', answer)
    lines = s10_state_parse(answer, cut)
    rescued = status.get('s11_state_rescue', 'none') != 'none'  # T1_S11_CONV: its DISTILL keeps the rescue's reserve
    spilled = (_s10_spill(client, messages[0]['content'], answer, 'state', S10_STATE_WORDS, 0, deadline, status, pace,
                          canaries, sleep, replay, rescued) if s10_spill(answer, len(lines)) else None)
    if spilled:
        again = s10_state_parse(*spilled)
        if _s10_spill_used(status, 'state', _s10_steps(again), S10_S_MIN):
            (answer, cut), lines = spilled, again
    if _s10_steps(lines) >= S10_S_MIN:
        s10['state_lines'] = lines
        status['s10_state'] = 'cut_kept' if cut else 'ok'
    else:
        status['s10_state'] = 'no_lines' if answer else 'failed'


def s10_refine(client, prompt, instruction, s10, deadline, status, pace, *, canaries=(), sleep=None,
               replay=_NO_REPLAY):
    """T1_S10's CHECK and STATE (spec §2.2, §2.3) on prompt, the program prompt with REQUIRED (s10_adopt), then the
    notes stage's status fields. Both need DECIDE's notes; without them (a fallback plan) neither runs and their
    fields stay unset. CHECK runs under T1_S10_CHECK=auto on its trigger (_s10_check_trigger) or under always (unset,
    it is off: s10_check_mode), unless the breaker has tripped (skipped_breaker) or its gate fails at the floor
    (skipped_time); its D lines are merged by ID (s10_merge), and without any its text is added (no_lines). STATE
    runs on a stateful unit (S10_STATE_HITS loop words, or DECIDE's STATE: yes; T1_S10_STATE=0 never), with thinking
    off after a breaker trip, and stands with S10_S_MIN S lines or more. A CHECK or STATE reply that went on
    reasoning in its text, or wrote code instead of its lines (s10_spill), gets the spill-over DISTILL (_s10_spill;
    CHECK's reserves what CHECK's gate did), whose reply replaces it with the least the step stands with, however
    many lines the reply kept (_s10_spill_used; CHECK: one D line or more, or OTHERS CONFIRMED without one, any text
    before it kept as CHECKED; STATE: S10_S_MIN S lines or more); s10_check_spill and s10_state_spill say how it
    went. Under T1_S11_CONV a CHECK reply without D lines is dropped (no_lines, nothing added); while more than
    S11_CONV_HALF of DECIDE's clock remains, STATE whose gate fails is rescued (_s11_state_rescue), not skipped, and
    STATE after a breaker trip reserves one full code reply and one run where R_code no longer fits (think_off); a
    rescued STATE's spill-over DISTILL keeps the same reserve. s11_state_rescue says which, and s11_d_default counts
    the D lines the notes carry that end in default. Under T1_S12_MECH, STATE runs as s12_stateful says (s12_state),
    and a note request that ended without a reply skips the steps after it (_s12_notes_ended), STATE's rescue
    included. Returns s10, with the notes block (s10_notes; '' without notes)
    in s10['block']."""
    if s10['notes']:
        state_on = os.environ.get('T1_S10_STATE') != '0'
        hits, said = s10['stateful'], s10['state'] == 'yes'
        status['s10_stateful'] = 'both' if hits and said else 'hits' if hits else 'reply' if said else 'none'
        stateful = state_on and (hits or said)
        if s12_mech_on():  # T1_S12_MECH: DECIDE's STATE: no holds below S12_STATE_STRONG loop words
            stateful, status['s12_state'] = s12_stateful(status.get('s10_state_hits', 0), s10['state'])
            stateful = state_on and stateful
        _s10_check(client, prompt, s10, deadline, status, pace, stateful, canaries, sleep, replay)
        if not state_on:
            status['s10_state'] = 'off'
        elif not stateful:
            status['s10_state'] = 'not_stateful'
        else:
            _s10_state(client, prompt, s10, deadline, status, pace, canaries, sleep, replay)
    _s10_map_proposals(s10, instruction)
    notes = s10['block'] = s10_notes(s10, tags=s10_tags_on())
    if s10.get('map_items') or s10.get('map_rejected'):
        status.update(method_map_retained=s10['map_retained'], method_map_omitted=s10['map_omitted'])
    lines = [s10['decisions'][n] for n in s10['kept']]
    status.update(s10_d_lines=len(lines), s10_d_quoted=sum(bool(quote) and _quoted(quote, instruction)
                                                           for quote in map(_s10_quote, lines)),
                  s10_d_free=sum(map(_s10_free, lines)), s10_d_unsure=sum(map(_s10_unsure, lines)),
                  s10_open=len(s10['open']), s10_state_lines=len(s10['state_lines']), s10_notes_chars=len(notes),
                  s10_notes_sec=round(CLOCK.now() - s10['started'], 1))
    if s11_conv_on():
        status['s11_d_default'] = sum(map(_s11_default, lines))
    s10_client_status(client, status)
    return s10


def s12_no_fence(content, complete):
    """T1_S12_MECH: whether a program reply holds no code: no fence anywhere in its answer, and no loose code, or
    loose code (code_region's) that verifies, as a finished reply when complete, to less than S12_CODE_SHARE of its
    characters: prose with an outline of code, where a program written without its fence verifies whole. False with
    the switch off."""
    if not s12_mech_on() or not isinstance(content, str):
        return False
    text = strip_reasoning(content)
    if '```' in text or '~~~' in text:
        return False
    code = code_region(text).strip()
    return not code or len(verified_prefix(code, complete)[0].strip()) < S12_CODE_SHARE * len(code)


_S12_DEF = re.compile(r'^[ \t]*(?:async[ \t]+)?def[ \t]+(\w+)', re.M)


def s12_rest_tokens(kind, program, tail, outputs=()):
    """T1_S12_MECH: the tokens a continuation of a cut program needs, estimated: the cut statement again (a
    continue_stmt re-sends it whole, at S12_CODE_CPT characters a token), the rest of it, S12_DEF_TOKENS for each stub
    or function called but not defined (_pending; one the cut statement defines is under way), the __main__ block, and
    S12_OUTPUT_TOKENS for each output file name of outputs that neither names; at least S12_CONT_MIN, at most 4,000."""
    text, under_way = program + tail, set(_S12_DEF.findall(tail))
    pending = [name for name, _ in _pending(program)
               if name not in under_way and not (name == 'main' and '__main__' in tail)]
    tokens = ((len(tail.strip()) / S12_CODE_CPT if kind == 'continue_stmt' else 0)
              + (S12_OPEN_TOKENS if tail.strip() else 0) + S12_DEF_TOKENS * len(pending)
              + (0 if '__main__' in text else S12_GUARD_TOKENS)
              + S12_OUTPUT_TOKENS * sum(name not in text for name in set(outputs)))
    return int(min(4000, max(S12_CONT_MIN, tokens)))


def s12_continuation(kind, program, tail, deadline, pace, status, outputs=()):
    """T1_S12_MECH: (kind, need) of a continuation of a cut program. While a full reply and a run fit, as before
    (Pace.full()). Otherwise the gate needs the reply s12_rest_tokens estimates at the measured rate (s12_cont_sized
    counts those that then fit), and a continue_stmt whose cut statement does not fit again goes on from its last
    complete line instead (continue_line; s12_cont_line) where that fits. Where neither fits, the time gate stops it
    as before. The request's timeout still keeps the run's reserve."""
    full, left = pace.full(), deadline - CLOCK.now()
    if left >= full + EXEC_FLOOR:
        return kind, full
    need = min(full, s12_rest_tokens(kind, program, tail, outputs) / pace.rate + 6)
    if left >= need + EXEC_FLOOR:
        _count(status, 's12_cont_sized')
        return kind, need
    if kind == 'continue_stmt' and '\n' in tail:
        line = min(full, s12_rest_tokens('continue_line', program, tail, outputs) / pace.rate + 6)
        if left >= line + EXEC_FLOOR:
            _count(status, 's12_cont_line')
            return 'continue_line', line
    return kind, need


def generate_program(client, messages, deadline, status, *, canaries=(), pace=None, sleep=None, think=None,
                     tags=None, outputs=()):
    """R1, then verified-prefix continuations until the program is complete; only a complete program
    is returned. Counters live in status so they survive a failure. With T1_STAGED=1 (the image's flag), a
    program reply cut at the output cap (the first one, or the rewrite that replaces it) is continued as without
    the flag, and staged_program takes over only when the continuations end without a complete program: on a
    stop in STAGE_AFTER (no_progress, chunk_cap, empty_code, invalid_code or no_entry) that leaves the verified
    program empty or structurally unfinished (_pending: no main to run it, a stub, or a function called but not
    defined), while the request cap and the clock still leave staging its first replies (_stage_fits);
    staged_after_continuation marks it. Continuing first keeps what the cut reply planned: in recorded runs a
    continuation completed 63 of 93 cut first replies within MAX_REPLIES, and 13 of the 22 continuations that
    added no statement were followed by one that completed the program, so staging waits for the loop's own
    stops (two replies without progress, or the reply cap) rather than the first reply without progress. A
    length reply with a degenerate tail may stage immediately when a separate nondegenerate verified prefix
    has complete functions and still has pending work; its original module bindings are protected during
    completion. A reply that finished is never staged, a gate stop (time_gate, request_cap, prompt_cap,
    request_failed) is not either, and neither is a cut program that only lacks statements at its end (module-level code, say). With
    the self-check on (SELFCHECK_BLOCK in the prompt), a cut reply whose only pending name is self_check is
    complete without it: self_check becomes a stub that checks nothing (selfcheck_stubbed) and nothing more is
    asked for; and a finished reply that does not compile only because of its self_check has it stubbed
    (_selfcheck_stubbed) before a salvage, continuation or fill is spent on it. think (T1_R1THINK, from r1think;
    under T1_S10 the tier's code budget, s10_code_budget) is the first request's reasoning budget; every later
    request goes without it. A thinking first reply cut at the cap is rewritten or continued as any cut reply, and
    one that finished without a usable program (nothing to salvage) is taken as cut there, so that the rewrite
    replaces it where B0 would end the unit. A code request that fails after an attempt went out with that budget
    still reports it in s10_code_budget. tags (T1_S10, a set) gathers the decisions that the code of every reply,
    staged ones included, names in its comments (s10_tag_ids): the merge of a cut reply's statements keeps no
    comment after a simple statement (an import or a constant), so the program alone would not show them all. Under
    T1_S12_MECH a reply asked for a whole program (S12_FENCE_KINDS) without a fence whose loose code is mostly not
    code (s12_no_fence; s12_no_fence counts it) is no code (rewrite_no_code), a finished one included, and where a full
    reply no longer fits, a continuation's gate needs what the cut program has left to write (s12_continuation;
    outputs: the output file names the TASK gives)."""
    pace = pace or Pace()
    staged = os.environ.get('T1_STAGED') == '1'
    selfcheck = SELFCHECK_BLOCK in messages[0]['content']
    kind, sent, program, tail, zero, cut = 'initial', messages, '', '', 0, False
    first = len(client.diagnostics['requests'])
    try:
        reply = _request(client, messages, deadline, pace, kind, sleep, budget=think)
    except HouseError:
        if s10_on() and think is not None:  # the unit ends here, but a budgeted attempt was sent
            sent_budget = any('reasoning_budget' in r for r in client.diagnostics['requests'][first:])
            status['s10_code_budget'] = think if sent_budget else 0
        raise
    thought = bool(think) and 'reasoning_budget' in client.diagnostics['requests'][-1]
    if s10_on() and think is not None:  # T1_S10's code budget by tier, as sent (0: the clock or the breaker said no)
        status['s10_code_budget'] = think if thought else 0
    elif thought:
        status['r1think_reasoning_chars'] = client.diagnostics['requests'][-1].get('reasoning_chars', 0)
    elif think:  # a retry the clock no longer let think went without the budget
        status['r1think'] = 'skipped_time'
    while True:
        _count(status, 'generation_replies')
        pace.measure(client, reply)
        status['rate_tok_s'] = round(pace.rate, 1)
        content, finish = _text(reply), reply['finish_reason_raw']
        if tags is not None:
            tags |= s10_tag_ids(code_region(content))
        salvage = unfenced = False
        if kind == 'initial' and finish != 'length':  # a finished (or refused) R1 keeps B0's extraction
            try:
                code = extract_python(content, finish)
            except HouseError as exc:
                code = (_selfcheck_stubbed(code_region(content), status)
                        if selfcheck and exc.code == 'invalid_python' and finish == 'stop' else None)
                if code is not None:  # only its self-check did not compile
                    status['stop_reason'] = 'complete'
                    return code.strip() + '\n'
                if exc.line is not None:
                    client.diagnostics['code_line'] = exc.line
                if exc.code == 'invalid_python' and finish == 'stop' and s12_no_fence(content, True):
                    finish, unfenced = 'length', True  # T1_S12_MECH: prose without a fence: rewritten, not salvaged
                    _count(status, 's12_no_fence')
                elif (exc.code != 'invalid_python' or finish != 'stop'
                        or not verified_prefix(code_region(content), True)[0].strip()):
                    if not thought or finish != 'stop':
                        raise
                    finish = 'length'  # it reasoned, then wrote no usable program: rewritten as a cut reply would be
                else:
                    salvage = True  # a garbled statement in a finished reply: keep what verifies, ask for the rest
            if not salvage and finish != 'length':
                status['stop_reason'] = 'complete'
                return code
        entire = kind in S12_FENCE_KINDS and not program.strip()  # T1_S12_MECH: a reply asked for a whole program
        if kind == 'fill' or (kind == 'rewrite' and not program.strip()):
            tail = ''
        record = client.diagnostics['requests'][-1]
        metrics = reply_metrics(content, reply['usage'].get('completion_tokens'))
        triage = 'none'
        if finish != 'stop' and kind != 'fill':
            triage = ('rewrite_no_code' if metrics['code_chars'] < 400 else
                      'rewrite_degenerate' if metrics['repeat_ratio'] > 6 else 'none')
            if unfenced or (triage == 'none' and entire and s12_mech_on() and s12_no_fence(content, False)):
                if not unfenced:
                    _count(status, 's12_no_fence')
                triage = 'rewrite_no_code'  # T1_S12_MECH: prose with an outline of code, without a fence
        record.update(triage=triage, code_chars=metrics['code_chars'], repeat_ratio=metrics['repeat_ratio'])
        progress = False
        if staged and finish == 'length' and triage == 'rewrite_degenerate':
            # The repeated tail is not permission to discard independent completed
            # functions. Keep only verified code, then use the existing stage gates.
            prefix, rest = verified_prefix(code_region(content), False)
            if program.strip():
                candidate = program
                if reply_metrics(prefix, None)['repeat_ratio'] <= 6:
                    candidate, _ = keep_complete_defs(program, merge(program, prefix, 'continue', False)[0])
                conflicts = completion_binding_conflicts(program, candidate)
                if conflicts:
                    # Reject the incompatible addition, not the existing source.
                    status['prefix_binding_conflicts'] = len(conflicts)
                prefix = program if conflicts else candidate
            stubs = {name for name, _, _ in stub_defs(prefix)}
            definitions = [node for node in ast.parse(prefix).body
                           if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name not in stubs]
            if (definitions and reply_metrics(prefix, None)['repeat_ratio'] <= 6 and _pending(prefix)
                    and _stage_fits(client, prefix, deadline, pace)):
                status['staged_preserved_prefix_defs'] = len(definitions)
                status['staged_preserved_prefix_sha256'] = hashlib.sha256(prefix.encode('utf-8')).hexdigest()
                return staged_program(client, messages, prefix, deadline, status, canaries=canaries, pace=pace,
                                      sleep=sleep, tail=tail + '\n' + rest,
                                      tags=tags, preserve_bindings=True)
        if triage == 'none':
            if selfcheck and finish == 'stop' and not salvage:
                content = _stub_reply(content, status)
            program, tail, st = absorb(program, tail, content, finish, 'line' if kind == 'continue_line' else 'continue')
            if selfcheck and finish == 'stop' and tail.strip() and kind != 'fill':  # a line join, say
                whole = _selfcheck_stubbed(program + tail, status)
                program, tail = (whole, '') if whole is not None else (program, tail)
            progress = st['stmts_added'] > 0 or st['replaced'] > 0 or st.get('tail_progress', False)
            record.update(candidate=st['candidate'], verified_stmts=st['verified_stmts_total'],
                          stmts_added=st['stmts_added'], tail_progress=bool(st.get('tail_progress', False)),
                          pending_lines=st['pending_lines'])
            if (staged and selfcheck and finish == 'length'
                    and [name for name, _ in _pending(program)] == [SELFCHECK_NAME]):
                # Cut inside the self-check alone: the program is complete without it, so it is stubbed, not continued.
                program, appended = ensure_guard(merge(program, SELFCHECK_STUB, 'repair', True)[0])
                status.update(stop_reason='complete', guard_appended=appended, selfcheck_stubbed=True,
                              final_missing_calls_count=len(missing_calls(program)))
                return program
        cut = cut or (staged and finish == 'length' and kind in ('initial', 'rewrite'))
        stop, kind, zero = _next_request(kind, finish, triage, progress, program, tail, zero, status)
        if not stop:
            need = pace.patch() if kind == 'fill' else pace.full()
            if kind in S12_CONT_KINDS and s12_mech_on():  # T1_S12_MECH: the gate needs what is left to write
                kind, need = s12_continuation(kind, program, tail, deadline, pace, status, outputs)
            conversation = _conversation(kind, messages, program, tail, canaries)
            stop = 'no_progress' if conversation == sent else None
        if not stop:
            attempts = client.attempts
            try:
                bounded = (os.environ.get('T1_CONTINUATION_BUDGET') == '1'
                           and kind in ('continue_stmt', 'continue_line') and bool(program.strip()))
                reply = _request(client, conversation, deadline, pace, kind, sleep,
                                 need=need,
                                 **({'continuation_budget': True} if bounded else {}))
                sent = conversation
            except (_Stop, HouseError) as exc:  # a fill that cannot be sent leaves the program complete
                stop = 'complete' if kind == 'fill' else getattr(exc, 'reason', 'request_failed')
            finally:
                if client.attempts > attempts:
                    _count(status, _COUNTERS[kind])
        if stop == 'complete':
            program, appended = ensure_guard(program)
            status.update(stop_reason='complete', guard_appended=appended,
                          final_missing_calls_count=len(missing_calls(program)))
            return program
        if stop:
            if (cut and stop in STAGE_AFTER and (not program.strip() or _pending(program))
                    and _stage_fits(client, program, deadline, pace)):
                status['staged_after_continuation'] = True
                return staged_program(client, messages, program, deadline, status, canaries=canaries, pace=pace,
                                      sleep=sleep, tail=tail, tags=tags)
            status['stop_reason'] = stop
            raise HouseError('No complete program was generated; nothing executed', kind='model_output',
                             code='invalid_python' if kind == 'salvage' else  # the salvage was never answered
                             {'empty_code': 'empty_program', 'invalid_code': 'invalid_python'}.get(
                                 stop, 'incomplete_code'))


def _runs(program):
    """Whether running the program would run its own functions: a __main__ guard, a main() the guard can be
    appended for, or a top-level statement that calls one of its functions. A script without functions runs when
    it calls anything at all (_entered); a setup call beside functions nothing calls does not count."""
    try:
        body = ast.parse(program).body
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return False
    own = {node.name for node in body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
    if not own:
        return _entered(program)
    if any(is_main_guard(node) for node in body) or ensure_guard(program)[1]:
        return True
    return any(isinstance(item, ast.Call) and isinstance(item.func, ast.Name) and item.func.id in own
               for node in body if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
               for item in ast.walk(node))


def _pending(program):
    """What staged generation still has to write, in order: [(name, line shown to the model)] for each stub, each
    function called but never defined, then main() when nothing would run the program's functions (_runs)."""
    items = [(name, f'- `{header}`' + (f' - {doc}' if doc else '')) for name, header, doc in stub_defs(program)]
    named = {name for name, _ in items}
    items += [(name, f'- `{name}` - called but not defined yet') for name in missing_calls(program) if name not in named]
    if not _runs(program):
        items.append(('main', '- `main()`, which calls the functions in order, and the `if __name__ == "__main__":` '
                              'block - missing'))
    return items


def _stage_fits(client, program, deadline, pace):
    """Whether staged generation can still reach a program that runs: its first replies (a skeleton and one stage,
    or one stage when the verified main already runs its functions, as staged_program skips the skeleton then)
    within the request cap, each a full reply, before the execution reserve."""
    replies = 1 if program.strip() and all(name != 'main' for name, _ in _pending(program)) else 2
    return (client.attempts + replies <= client.max_requests
            and deadline - CLOCK.now() >= replies * pace.full() + EXEC_FLOOR)


def _stage_conversation(kind, messages, program, pending, target, note, canaries, missing=()):
    """The next staged request: the task messages, the program so far as the assistant turn, then the ask (after
    note, which says what was wrong with the last reply). A skeleton or stage ask ends by naming missing: the
    output names the cut reply wrote that the program does not have yet (STAGE_OUTPUTS)."""
    outputs = STAGE_OUTPUTS.format(names=', '.join(f'`{name}`' for name in missing[:STAGE_OUTPUT_NAMES]))
    outputs = outputs if missing and kind != 'split' else ''
    if kind == 'skeleton':
        kept = STAGE_KEPT.format(n=statement_count(program)) if program.strip() else ''
        ask = note + STAGE_SKELETON.format(kept=kept, whole='the rest of the program' if kept else 'the whole program',
                                           lines=STAGE_DEF_LINES, cap=STAGE_SKELETON_LINES) + outputs
        if not program.strip():
            return messages[:-1] + [{'role': 'user', 'content': messages[-1]['content'] + '\n\n'
                                     + withhold_canaries(ask, canaries)}]
    elif kind == 'split':
        ask = note + STAGE_SPLIT.format(name=target, lines=STAGE_SPLIT_LINES)
    else:
        ask = note + STAGE_BODIES.format(items='\n'.join(line for _, line in pending[:STAGE_ITEMS]),
                                         lines=STAGE_DEF_LINES) + outputs
    return messages + [{'role': 'assistant', 'content': withhold_canaries(_fence(program), canaries)},
                       {'role': 'user', 'content': withhold_canaries(ask, canaries)}]


def _stub_selfcheck(program, on, status):
    """With the self-check on (on: the prompt carried SELFCHECK_BLOCK), staged generation's program with a pending
    self_check (a stub, or called but not defined) given SELFCHECK_STUB instead (selfcheck_stubbed): the stage
    requests leave the SELF-CHECK contract out, so a self_check written there could not report and only spent
    output. The program unchanged otherwise."""
    if not on or all(name != SELFCHECK_NAME for name, _ in _pending(program)):
        return program
    stubbed = merge(program, SELFCHECK_STUB, 'repair', True)[0]
    if any(name == SELFCHECK_NAME for name, _ in _pending(stubbed)):
        return program
    status['selfcheck_stubbed'] = True
    return stubbed


def _selfcheck_stubbed(code, status):
    """Code that does not compile only because of its self_check, with that self_check given SELFCHECK_STUB
    (stub_broken_def; selfcheck_stubbed, and selfcheck_syntax_stubs counts it): a self-check the model got wrong
    never costs the program a salvage, a continuation, a stage or a patch retry. None otherwise. Callers pass
    code only with the self-check on (SELFCHECK_BLOCK in the prompt)."""
    stubbed = stub_broken_def(code, SELFCHECK_NAME, SELFCHECK_STUB)
    if stubbed == code:
        return None
    _count(status, 'selfcheck_syntax_stubs')
    status['selfcheck_stubbed'] = True
    return stubbed


def _stub_reply(content, status):
    """A finished reply with its code region (its last copy in the text) passed through _selfcheck_stubbed, so
    every merge takes the whole region; the reply unchanged when that changes nothing."""
    region = code_region(content) if isinstance(content, str) else ''
    at = content.rfind(region) if region.strip() else -1
    stubbed = _selfcheck_stubbed(region, status) if at >= 0 else None
    return content if stubbed is None else content[:at] + stubbed + content[at + len(region):]


def staged_program(client, messages, program, deadline, status, *, canaries=(), pace=None, sleep=None, tail='',
                   tags=None, preserve_bindings=False):
    """Staged generation after the continuations of a program reply cut at the output cap ended without a complete
    program (program: the verified program so far; tail: the unverified text after it). First a
    skeleton: the imports, constants, every function as a stub with a one-line docstring, and a main() that only
    calls them. A verified prefix that already runs its functions (main complete, _pending names no main) needs
    no skeleton: its own main fixes the structure, so staging starts with the pending functions
    (stage_skeleton_skipped). No reply, skeleton or stage, replaces or re-signs a function the verified program
    defines completely (keep_complete_defs; stage_defs_kept counts the restores). The output names the verified
    program and its tail write (output_names: dict keys and output file names, the self-check's left out) are
    kept too: each skeleton or stage request names those the program does not have yet (STAGE_OUTPUTS), and
    stage_outputs_missing counts those still missing at the end. Then
    the stubs (and any function called but never defined) are asked for in order, each reply taking as many whole
    functions as fit (absorb_defs keeps the complete ones of a cut reply, and those after an invalid statement,
    which the next request names; stage_salvaged counts the statements so kept). The size gate: when a reply is
    cut before the first function asked for is complete, that function is asked for again rewritten as a short
    one calling new helper stubs. A skeleton cut at the cap is not taken but asked for again, shorter, and the
    program is complete only once a skeleton was taken (or skipped) and nothing is pending. The self-check block
    is left out of these requests: the program already overflows. So, when the prompt carried it, a self_check
    that is pending (a stub, or called but not defined) is never asked for: it becomes the stub that checks
    nothing (_stub_selfcheck, selfcheck_stubbed), as a first reply cut inside it alone does, and a finished reply
    that does not compile only because of a self_check has it stubbed too (_stub_reply). Every request passes
    the usual request and
    time gates; at most STAGE_REPLIES replies. Returns the complete program. When the replies run out or two in
    a row make no progress after a skeleton, while main can run, the program is returned with each remaining
    stub raising NotImplementedError (stop_reason staged_partial), so the crash repair can finish it; otherwise
    HouseError as for any incomplete program. Counters: staged_generation, stage_replies, stage_splits,
    stage_stubs_left, stage_skeleton_skipped, stage_defs_kept, stage_outputs_missing (numbers and booleans
    only). preserve_bindings is used only for a degenerate-tail handoff: a completion that changes the
    original module binding statements is rejected, noted for the next existing request, and remains
    unresolved until a compatible completion arrives. Explicit later crash/semantic repairs are unaffected."""
    pace = pace or Pace()
    status['staged_generation'] = True
    selfcheck = SELFCHECK_BLOCK in messages[0]['content']
    base = [dict(messages[0], content=messages[0]['content'].replace(SELFCHECK_BLOCK, '', 1))] + messages[1:]
    kind, target, zero, stop, note, skeleton = 'skeleton', None, 0, None, '', False
    outputs = output_names(program.rstrip('\n') + '\n' + tail, SELFCHECK_NAME if selfcheck else '')
    program = verified = _stub_selfcheck(program, selfcheck, status)
    if preserve_bindings:
        status['stage_binding_unresolved'] = 0
    if program.strip() and all(name != 'main' for name, _ in _pending(program)):
        kind, skeleton = 'stage', True  # the verified main already calls the functions in order
        status['stage_skeleton_skipped'] = True
    for _ in range(STAGE_REPLIES):
        pending = _pending(program)
        if skeleton and not pending:
            break
        values = string_values(program)
        missing = [name for name in outputs if not has_name(values, name)]
        conversation = _stage_conversation(kind, base, program, pending, target, note, canaries, missing)
        attempts = client.attempts
        try:
            reply = _request(client, conversation, deadline, pace, kind, sleep, need=pace.full())
        except _Stop as exc:
            stop = exc.reason
            break
        except HouseError:
            stop = 'request_failed'
            break
        finally:
            if client.attempts > attempts:
                _count(status, 'stage_replies')
        _count(status, 'generation_replies')
        pace.measure(client, reply)
        content, finish = _text(reply), reply['finish_reason_raw']
        if tags is not None:  # T1_S10: generate_program's tags
            tags |= s10_tag_ids(code_region(content))
        if selfcheck and finish == 'stop':
            content = _stub_reply(content, status)
        before, note, kept = program, '', 0
        if kind == 'skeleton':  # a skeleton cut at the cap is not taken: it would leave the program's end unknown
            if finish != 'length':
                program, kept = keep_complete_defs(before, absorb(program, '', content, finish, 'continue')[0])
            progress = skeleton = _shape(program) != _shape(before)
            note = STAGE_SHORTER if finish == 'length' else ''
        else:
            program, merged = absorb_defs(program, content, finish)
            program, kept = keep_complete_defs(verified, program)
            if merged['salvaged']:
                _count(status, 'stage_salvaged', merged['salvaged'])
            if merged['dropped']:  # the functions after an invalid statement were kept; say which one was not
                note = STAGE_INVALID.format(error=first_invalid(content) or 'it does not compile')
        conflicts = completion_binding_conflicts(verified, program) if preserve_bindings else []
        if conflicts:
            program, progress = before, False
            if kind == 'skeleton':
                skeleton = False
            status['stage_binding_unresolved'] = len(conflicts)
            _count(status, 'stage_binding_conflicts')
            note = ('The completion changed existing module bindings: '
                    + ', '.join(conflicts[:10])
                    + '. It was not accepted. Keep their original definitions, imports and values; '
                    'supply only the missing code.\n')
        else:
            if preserve_bindings:
                status['stage_binding_unresolved'] = 0
            if kept:
                _count(status, 'stage_defs_kept', kept)
        program = _stub_selfcheck(program, selfcheck, status)  # a skeleton may stub it or have main call it
        left = {name for name, _ in _pending(program)}
        if kind == 'split':
            progress = target not in left
        elif kind == 'stage':
            progress = bool({name for name, _ in pending} - left)
        client.diagnostics['requests'][-1].update(stage_pending=len(left))
        if progress:
            kind, zero = 'stage', 0
        elif kind == 'stage' and finish == 'length' and pending and pending[0][0] in left:
            kind, target, zero = 'split', pending[0][0], 0  # the size gate: the first function did not fit
            _count(status, 'stage_splits')
        else:
            zero += 1
            note = note or STAGE_AGAIN
            if zero >= 2:
                stop = 'no_progress'
                break
    pending = _pending(program)
    values = string_values(program)
    status.update(stage_stubs_left=len(pending),
                  stage_outputs_missing=sum(not has_name(values, name) for name in outputs))
    if skeleton and not pending:
        program, appended = ensure_guard(program)
        status.update(stop_reason='complete', guard_appended=appended,
                      final_missing_calls_count=len(missing_calls(program)))
        return program
    stop = stop or 'stage_cap'
    if (skeleton and stop in ('stage_cap', 'no_progress') and not status.get('stage_binding_unresolved')
            and all(name != 'main' for name, _ in pending)):
        program, appended = ensure_guard(raise_stubs(program))
        status.update(stop_reason='staged_partial', guard_appended=appended,
                      final_missing_calls_count=len(missing_calls(program)))
        return program
    status['stop_reason'] = stop
    raise HouseError('No complete program was generated; nothing executed', kind='model_output',
                     code='incomplete_code')


def precheck_on() -> bool:
    """T1_PRECHECK=1 (the image's flag): the static checks around a repair. The crash request's lint also finds a
    function that calls itself on every call before any return (call_lint recursion), a merged patch is linted
    before it runs (_lint_gate), and a newly generated program gets its static fixes before its first run
    (precheck_fixed). Off, programs and repairs are as before."""
    return os.environ.get('T1_PRECHECK') == '1'


def raiseguard_on() -> bool:
    """T1_RAISEGUARD=1 (the image's flag): the task prompt keeps its error for missing or invalid input data but says
    a check of the program's own results never raises (RESULT_CHECKS), and a raise of such a check that a patch
    left failing the same way is rewritten into a warning and run again, with zero requests (guard_raise). Off,
    prompts and programs are as before."""
    return os.environ.get('T1_RAISEGUARD') == '1'


def callback_call_lint(code, limit=LINT_LIMIT):
    """Binding advice for a finite, definite local wrapper -> callback link.

    Read AST/signatures only. A wrapper must invoke its callback before any
    possible exit, through simple statements or exhaustive if/else paths. Only
    invocation shapes shared by every finite exit/fallthrough path are checked;
    unknown control flow keeps only the already-guaranteed prefix. Rebindings,
    escapes, reflection and dynamic arguments/signatures are left alone. Each
    caller/callback/shape is reported once; no value is inserted or executed.
    """
    try:
        tree = ast.parse(code)
        compile(tree, '<callback-call-lint>', 'exec', dont_inherit=True)
        nodes = list(ast.walk(tree))
        dynamic = {'exec', 'eval', 'globals', 'locals', 'vars', 'getattr', 'setattr', 'delattr',
                   '__import__', '__dict__', '__globals__', '__defaults__', '__kwdefaults__',
                   '__code__', 'f_globals', 'f_locals', 'modules'}
        if any((isinstance(n, ast.Name) and n.id in dynamic)
               or (isinstance(n, ast.Attribute) and n.attr in dynamic)
               or (isinstance(n, ast.ImportFrom) and any(a.name == '*' or a.name in dynamic for a in n.names))
               for n in nodes):
            return []
        functions = {name: fn for name, fn in _lintable(tree).items()
                     if isinstance(fn, ast.FunctionDef) and fn.args.vararg is None and fn.args.kwarg is None
                     and not any(isinstance(n, ast.MatchMapping) and n.rest == name for n in nodes)}
        parents = {child: n for n in nodes for child in ast.iter_child_nodes(n)}
        conditional = (ast.IfExp, ast.BoolOp, ast.Lambda, ast.ListComp, ast.SetComp,
                       ast.DictComp, ast.GeneratorExp, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)

        def simple_calls(node):
            todo, calls = [node], []
            while todo:
                item = todo.pop()
                if isinstance(item, conditional):
                    continue
                if isinstance(item, ast.Call):
                    calls.append(item)
                if isinstance(item, ast.AnnAssign):
                    todo.extend([item.target, item.value] if item.value is not None else [])
                else:
                    todo.extend(ast.iter_child_nodes(item))
            return calls

        def shape(call):
            return len(call.args), tuple(sorted(k.arg for k in call.keywords))

        def bound(call, fn):
            if _arity(call, fn) is not None or any(isinstance(a, ast.Starred) for a in call.args) \
                    or any(k.arg is None for k in call.keywords):
                return None
            spec = fn.args
            positional = spec.posonlyargs + spec.args
            result = {a.arg: value for a, value in zip(positional[-len(spec.defaults):], spec.defaults)} \
                if spec.defaults else {}
            result.update((a.arg, value) for a, value in zip(spec.kwonlyargs, spec.kw_defaults)
                          if value is not None)
            result.update((a.arg, value) for a, value in zip(positional, call.args))
            result.update((k.arg, k.value) for k in call.keywords)
            return result

        wrappers = {}
        for name, fn in functions.items():
            arguments = {a.arg for a in fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs}
            local = _bound_in(fn)
            own = []
            todo = list(fn.body)
            while todo:
                item = todo.pop()
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                    continue
                own.append(item)
                todo.extend(ast.iter_child_nodes(item))
            if any(isinstance(n, (ast.Yield, ast.YieldFrom)) for n in own):
                continue
            sites = [n for n in own if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                     and n.func.id in arguments]
            for parameter in arguments:
                calls = [n for n in sites if n.func.id == parameter]
                if not calls or local.get(parameter) != [0] \
                        or any(isinstance(n, ast.MatchMapping) and n.rest == parameter for n in own) \
                        or any(isinstance(n, (ast.Global, ast.Nonlocal)) and parameter in n.names for n in own):
                    continue
                if any(any(isinstance(a, ast.Starred) for a in call.args)
                       or any(k.arg is None for k in call.keywords) for call in calls):
                    continue
                # A callback object used anywhere else can escape or be mutated.
                if any(isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id == parameter
                       and (parents.get(n) not in calls or parents[n].func is not n) for n in ast.walk(fn)):
                    continue

                def meet(*states):
                    values = [value for value in states if value is not None]
                    return set.intersection(*(set(value) for value in values)) if values else None

                reached = set()

                def flow(block, prefix=()):
                    # None = no paths; a set = shapes every such path already invokes.
                    continuing, exited = set(prefix), None
                    for statement in block:
                        if isinstance(statement, (ast.Expr, ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Return)):
                            live = [call for call in simple_calls(statement) if call in calls]
                            reached.update(live)
                            continuing.update(shape(call) for call in live)
                            if isinstance(statement, ast.Return):
                                exited, continuing = meet(exited, continuing), None
                                break
                        elif isinstance(statement, ast.If):
                            if isinstance(statement.test, ast.Constant) and type(statement.test.value) is bool:
                                sides = [statement.body if statement.test.value else statement.orelse]
                            else:
                                sides = [statement.body, statement.orelse]
                            results = [flow(side, continuing) for side in sides]
                            exited = meet(exited, *(result[1] for result in results))
                            continuing = meet(*(result[0] for result in results))
                            if continuing is None:
                                break
                        elif isinstance(statement, (ast.Raise, ast.For, ast.AsyncFor, ast.While,
                                                    ast.Try, ast.TryStar, ast.With, ast.AsyncWith, ast.Match)):
                            exited, continuing = meet(exited, continuing), None
                            break
                    return continuing, exited

                guaranteed = meet(*flow(fn.body))
                if not guaranteed:
                    continue
                unique = {}
                for call in sorted(reached, key=lambda n: (n.lineno, n.col_offset)):
                    if shape(call) in guaranteed:
                        unique.setdefault(shape(call), call)
                wrappers.setdefault(name, {})[parameter] = list(unique.values())
        callers = [n for n in nodes if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                   and n.func.id in wrappers]
        bindings = {call: bound(call, functions[call.func.id]) for call in callers}
        safe_arguments = {values[parameter] for call, values in bindings.items() if values is not None
                          for parameter in wrappers[call.func.id] if parameter in values}
        escaped = set()
        for node in nodes:
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id in functions:
                parent = parents.get(node)
                if node not in safe_arguments and not (isinstance(parent, ast.Call) and parent.func is node):
                    escaped.add(node.id)
        if escaped.intersection(wrappers):
            return []
        lines, found = code.split('\n'), []
        for caller, values in bindings.items():
            if values is None:
                continue
            for parameter, calls in wrappers[caller.func.id].items():
                value = values.get(parameter)
                target = (ast.FunctionDef(name='lambda', args=value.args, body=[], decorator_list=[])
                          if isinstance(value, ast.Lambda) else functions.get(value.id)
                          if isinstance(value, ast.Name) and value.id not in escaped else None)
                if target is None or target.args.vararg is not None or target.args.kwarg is not None:
                    continue
                for call in calls:
                    problem = _arity(call, target)
                    if problem:
                        found.append((caller.lineno, caller.col_offset,
                                      f'line {caller.lineno}: `{lines[caller.lineno - 1].strip()[:120]}` - '
                                      f'callback `{parameter}` in `{caller.func.id}` (line {call.lineno}) {problem}'))
        return [text for _, _, text in sorted(found)][:limit]
    except (SyntaxError, ValueError, TypeError, RecursionError, MemoryError):
        return []


def call_lint(program, limit=LINT_LIMIT, recursion=False):
    """Existing direct advice, plus finite callback advice only under T1_PRECHECK."""
    found = _direct_call_lint(program, limit=None, recursion=recursion)
    if precheck_on():
        found += callback_call_lint(program, limit=None)
    return sorted(found, key=lambda item: int(item.split(':')[0][5:]))[:limit]


def _lint_gate(client, conversation, code, program, content, deadline, status, reserve, canaries, pace, sleep, near,
               normalize):
    """Advise once on call binding and narrow global-read risks under T1_PRECHECK.

    The existing lint_retry request keeps its budget gates. Accept a reply only when
    it reduces the combined finding count (repair_lint_fixed counts accepted
    improvements, not complete repairs). Previously observed scope risks that move
    outside the analysis count as unknown, not repaired. Library binding remains
    arity-only, and includes definite old faults a partial patch leaves behind.
    Finite callback binding has the same advisory retry; it never participates
    in automatic missing-value binding. The lint, scope, library-call and callback
    *_unresolved counters accumulate final findings across gate invocations;
    repair_scope_unknown separately accumulates unverified tracked
    risks. Neither zero counters nor an advisory finding proves runtime
    success or failure. Unresolved patches remain eligible for the ordinary run.
    """
    new = new_lint(code, program, recursion=True)
    scope = new_global_reads(code, program)
    library = library_call_lint(program, limit=None)
    callback = callback_call_lint(program, limit=None)
    if not new and not scope and not library and not callback:
        if status.get('repair_callback_call_flags') or status.get('repair_callback_call_pending'):
            _count(status, 'repair_callback_call_unresolved', 0)
        return program
    if new:
        _count(status, 'repair_lint_new', len(new))
    if scope:
        _count(status, 'repair_scope_new', len(scope))
    if library:
        _count(status, 'repair_library_call_pending', len(library))
    if callback:
        _count(status, 'repair_callback_call_pending', len(callback))
    shown = code_region(content) if isinstance(content, str) else ''
    # Without line numbers: they are the merged program's, which the model has not seen.
    prompt = SCOPE_LINT_RETRY if scope else LINT_RETRY
    ask = prompt.format(items='\n'.join('- ' + item.split(': ', 1)[-1]
                                      for item in (scope + new + library + callback)[:LINT_LIMIT]))
    conversation = conversation + [
        {'role': 'assistant', 'content': withhold_canaries(_fence(shown) if shown.strip() else '(no code)', canaries)},
        {'role': 'user', 'content': withhold_canaries(ask, canaries)}]
    try:
        again = _patch(client, conversation, program, deadline, status, reserve, canaries, pace, sleep, near,
                       'lint_retry', normalize)[0]
    except (_Stop, HouseError):
        again = None
    left = new
    observed = program
    if again is not None:
        again_lint = new_lint(code, again, recursion=True)
        again_scope = new_global_reads(code, again)
        again_library = library_call_lint(again, limit=None)
        again_callback = callback_call_lint(again, limit=None)
        unknown = unverified_global_reads(code, observed, again)
        if len(again_lint) + len(again_scope) + len(again_library) + len(again_callback) + len(unknown) \
                < len(new) + len(scope) + len(library) + len(callback):
            program, left = again, again_lint
            _count(status, 'repair_lint_fixed')
    if new and left:  # scope advice does not expand automatic parameter binding
        program, calls = bind_missing(program)
        if calls:
            _count(status, 'repair_lint_bound', calls)
    _count(status, 'repair_lint_unresolved', len(new_lint(code, program, recursion=True)))
    _count(status, 'repair_scope_unresolved', len(new_global_reads(code, program)))
    _count(status, 'repair_scope_unknown', len(unverified_global_reads(code, observed, program)))
    _count(status, 'repair_library_call_unresolved', len(library_call_lint(program, limit=None)))
    left_callback = callback_call_lint(program, limit=None)
    if left_callback or status.get('repair_callback_call_flags') or status.get('repair_callback_call_pending'):
        _count(status, 'repair_callback_call_unresolved', len(left_callback))
    return program


def repair_program(client, messages, code, tail, exit_code, files, deadline, status, reserve, *,
                   canaries=(), pace=None, sleep=None, report='', history='', near=None, normalize=None, ask=None,
                   ran=(), failed=()):
    """One patch (plus at most one continuation) merged into the program. A reply that makes no progress gets
    one retry that says so, naming the error it did not address: one that changes nothing (then None when the
    retry changes nothing either), and one that goes back to what already failed: a program whose shape is in
    ran, or one that brings back a line of failed ((function, line text, error) of earlier crashes) that the
    program no longer has. After that retry the reverting patch itself stands when the retry gives nothing
    better, so the caller's own checks decide it as before. The crash request also lists what call_lint finds
    in the program (advice only; repair_lint_flags counts it). Under T1_PRECHECK,
    definite exported library calls and finite local callback links also get
    binding advice from their source signatures; partial fixes use the existing
    lint retry, without inserting guessed values.
    The shared path supplies LIBRARY USE when the caller did not carry it in
    history, including a fresh program's crash repair. ask replaces the crash request (a speed repair).
    Under T1_PRECHECK the lint also lists endless self-recursion and names the program takes from the optional
    library that it does not have, and the merged patch passes _lint_gate before the revert checks. Under
    T1_LIBHINT the crash request points at a matched library function the program does not call (library_fix).
    Under T1_S10, with the notes block in the program prompt (s10_noted), the crash request (not a speed repair's
    ask) also asks to keep each decision the error does not show wrong (S10_PATCH_KEEP). Under T1_S11_RAISE, a crash
    on the program's own raise (s11_own_raise) gets S11_OWN_RAISE after the history (s11_own_raise counts it); under
    T1_S11_LINT a kill without a traceback gets S11_OOM before it (s11_oom_hints). Under T1_S12_POST (and
    T1_PRECHECK) the static check also lists a numpy ufunc given more positional arguments than it has inputs
    (ufunc_lint; s12_lint_repair counts them). Under T1_S12_MECH the crash ask names the crash site where the
    traceback gives one (S12_CRASH_SITE; s12_crash_site counts it), and a patch that returns nothing usable
    (_s12_unusable; s12_patch_unusable) gets the light retry (_s12_light_patch) first, on top of a cut patch's partial
    merge: one that is not sent or fails ends as before (a request error, or no change); one that comes back with no
    change leaves a partial merge standing, and otherwise RETRY follows (s12_patch_retry counts it). Both asks offer
    the crash site's def whole only where it is short (_s12_whole)."""
    pace = pace or Pace()
    paths = 'Input files that exist: ' + ', '.join(files) + '\n' if files else ''
    if ask is None:
        checked = precheck_on()
        found = call_lint(code, limit=None if checked else LINT_LIMIT, recursion=checked)
        if checked:  # and the names the program takes from the optional library that it does not have
            callbacks = callback_call_lint(code, limit=None)
            if callbacks:
                _count(status, 'repair_callback_call_flags', len(callbacks))
            calls = library_call_lint(code, limit=None)
            if calls:
                _count(status, 'repair_library_call_flags', len(calls))
            misread = api_lint(code, pareto=False) if s11_lint_on() else []  # T1_S11_LINT: an lstsq misread too
            if misread:
                _count(status, 's11_lint_repair', len(misread))
            arity = ufunc_lint(code) if s12_post_on() else []  # T1_S12_POST: a ufunc given too many arguments
            if arity:
                _count(status, 's12_lint_repair', len(arity))
                misread += arity
            found = sorted(found + library_references(code)[1] + misread + calls,
                           key=lambda item: int(item.split(':')[0][5:]))
            found = found[:LINT_LIMIT]
        _count(status, 'repair_lint_flags', len(found))
        lint = (LINT_ALL if checked else LINT).format(items='\n'.join('- ' + item for item in found)) if found else ''
        lint += library_fix(messages, code, 'the failing code is', required=False)
        line = near if near is not None else status.get('execution_diagnostics', {}).get('generated_line')
        if 'LIBRARY USE:' not in history:
            use, _ = library_use(code, line, _exception_line(report, tail), tail, report)
            if use:
                history = use + history
                _count(status, 'repair_library_use')
        own = s11_own_raise(code, line, _exception_line(report, tail))
        if own:  # T1_S11_RAISE: does the TASK state the check that ended the run?
            _count(status, 's11_own_raise')
            history += S11_OWN_RAISE.format(**own)
        oom = ''
        if s11_lint_on() and exit_code in S11_OOM_CODES and S11_TRACEBACK not in (tail or ''):
            evidence = status.get('execution_resource')
            oom = resource_control.repair_context(code, evidence) if isinstance(evidence, dict) else S11_OOM
            _count(status, 's11_oom_hints')
        elif (s11_lint_on() and status.get('execution_resource', {}).get('memory_event')
              == 'cgroup_oom_observed'):
            oom = resource_control.repair_context(code, status['execution_resource'])
            _count(status, 's11_oom_hints')
        ask = REPAIR.format(code=exit_code, tail=tail, report=REPORT.format(report=report) if report else '',
                            history=oom + history, paths=paths, lint=lint)
        if s10_noted(messages):  # T1_S10: patches may overturn a decision but keep the rest
            ask = with_s10_keep(ask)
        site = _where(code, line) if s12_mech_on() and ask.endswith(PATCH_FORMAT) else ''
        if site:  # T1_S12_MECH: the lines at the crash site, not the whole program
            ask = (ask.removesuffix(PATCH_FORMAT) + S12_CRASH_SITE.format(where=site, whole=_s12_whole(code, line))
                   + PATCH_FORMAT)
            _count(status, 's12_crash_site')
    conversation = messages + [{'role': 'assistant', 'content': withhold_canaries(_fence(code), canaries)},
                               {'role': 'user', 'content': withhold_canaries(ask, canaries)}]
    info, lost = ({} if s12_mech_on() else None), None
    try:
        program, why, content = _patch(client, conversation, code, deadline, status, reserve, canaries, pace, sleep,
                                       near, 'patch', normalize, info=info)
    except HouseError as exc:
        if info is None or not info.get('sent'):
            raise
        program, why, content, lost = None, '', None, exc  # T1_S12_MECH: sent, and no reply came back
    error = _error_note(_exception_line(report, tail))
    where = _where(code, near)
    unusable = _s12_unusable(code, program, why, info, near, lost) if info is not None else None
    if unusable:  # T1_S12_MECH: one light retry first, on top of what a cut patch merged
        _count(status, 's12_patch_unusable')
        light, answered = _s12_light_patch(client, conversation, code if program is None else program, unusable,
                                           where, _s12_whole(code, near), deadline, status, reserve, canaries, pace,
                                           sleep, near, normalize)
        if light is not None:
            program, why, content = light
        elif lost is not None:
            raise lost
        elif program is None and not answered:
            return None  # not sent or failed: stack 11's larger RETRY would not fit or fare better
        elif program is None:  # it came back with no change: stack 11's RETRY follows on its own gate
            attempts = client.attempts
            program = _retry_patch(client, conversation, code, content, _unchanged_retry(why, error, where),
                                   deadline, status, reserve, canaries, pace, sleep, near, 'patch_retry', normalize)
            if client.attempts > attempts:
                _count(status, 's12_patch_retry')
            return program
    if program is None:
        return _retry_patch(client, conversation, code, content, _unchanged_retry(why, error, where),
                            deadline, status, reserve, canaries, pace, sleep, near, 'patch_retry', normalize)
    if precheck_on():
        program = _lint_gate(client, conversation, code, program, content, deadline, status, reserve, canaries, pace,
                             sleep, near, normalize)
    shape = _shape((normalize or (lambda text: text))(program))
    back = _restored(code, program, failed) if shape not in ran else None
    if shape not in ran and back is None:
        return program
    _count(status, 'repair_reverts_rejected')
    retry = (RERUN.format(error=error, where=where) if back is None else
             REVERT.format(line=back[1][:160], function=back[0], error=_error_note(back[2]), where=where))
    again = _retry_patch(client, conversation, code, content, retry, deadline, status, reserve, canaries, pace,
                         sleep, near, 'patch_retry', normalize)
    return program if again is None else again


def _s12_unusable(code, program, why, info, near, lost=None):
    """T1_S12_MECH: why a crash patch gave nothing usable (S12_UNUSABLE), else None: no reply came back (lost, its
    request error: failed); it ended at the cap (a continuation that finished ends that) and changed nothing, or left
    the failing line (near) as it was in its function (cut: a whole program cut before it reached the fix); or it held
    no code (prose)."""
    if lost is not None:
        return 'failed'
    if info.get('cut'):
        site = _failing_statement(code, near)
        return 'cut' if program is None or (site[1] and site in _function_lines(program)) else None
    return 'prose' if program is None and why == 'it contained no code' else None


def _s12_light_patch(client, conversation, base, unusable, where, whole, deadline, status, reserve, canaries, pace,
                     sleep, near, normalize, *, ask=None, no_change=None):
    """T1_S12_MECH: the one light retry (patch_retry) of a crash patch that gave nothing usable (unusable,
    S12_UNUSABLE): thinking off, each reply at most S12_PATCH_CAP tokens, its gate and timeout sized to that, asking
    for the changed lines (S12_PATCH_LIGHT, where: _where's; whole: _s12_whole's, the crash site's def offered whole
    only where it is short) after a stand-in for the reply that came (S12_PATCH_TURN: a large cut reply is not shown
    again). Its patch goes on top of base: the program, or what a cut patch merged into it. s12_patch_light counts
    the retries that went out, s12_patch_light_used those whose program is taken. ((program, why, content) or None,
    whether a reply came back): None when it is not sent, fails or changes nothing. The optional no_change dict
    observes only a valid last stop reply of complete unchanged ordinary definitions, retaining its reason, text
    and conversation for S12 post-run feedback. It sends no request and leaves the default crash path unchanged."""
    if ask is None:  # Preserve the crash prompt; post-run flags supply their own short request.
        ask = S12_PATCH_LIGHT.format(why=S12_PATCH_WHY[unusable],
                                     site=S12_PATCH_SITE.format(where=where) if where else '', whole=whole)
    conversation = conversation + [{'role': 'assistant', 'content': S12_PATCH_TURN[unusable]},
                                   {'role': 'user', 'content': withhold_canaries(ask, canaries)}]
    attempts = client.attempts
    outcome = {} if no_change is not None else None
    try:
        program, why, content = _patch(client, conversation, base, deadline, status, reserve, canaries, pace, sleep,
                                       near, 'patch_retry', normalize, cap=S12_PATCH_CAP, outcome=outcome)
    except (_Stop, HouseError):
        return None, False
    finally:
        if client.attempts > attempts:
            _count(status, 's12_patch_light')
    if program is None:
        if (no_change is not None and outcome.get('reply_returned') is True
                and client.diagnostics['requests'][-1].get('finish_reason') == 'stop'
                and _same_repair_defs(base, content)):
            no_change.update(why=why, content=content, conversation=conversation)
        return None, True
    _count(status, 's12_patch_light_used')
    return (program, why, content), True


def _s12_whole(code, line):
    """T1_S12_MECH: S12_PATCH_WHOLE's words for the def that holds the failing line of code: offered whole where it
    fills at most S12_DEF_SHARE of S12_PATCH_CAP (at S12_CODE_CPT characters a token), else named as not to be sent
    whole; offered as before where no def holds the line or the code does not parse."""
    lines = code.split('\n')
    try:
        spans = [(node.lineno, node.end_lineno or node.lineno, node.name) for node in ast.walk(ast.parse(code))
                 if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                 and type(line) is int and node.lineno <= line <= (node.end_lineno or node.lineno)]
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        spans = []
    if not spans:
        return S12_PATCH_WHOLE[0]
    start, end, name = max(spans)
    size = len('\n'.join(lines[start - 1:end])) / S12_CODE_CPT
    return S12_PATCH_WHOLE[0] if size <= S12_DEF_SHARE * S12_PATCH_CAP else S12_PATCH_WHOLE[1].format(function=name)


def s11_own_raise(code, line, error):
    """T1_S11_RAISE: S11_OWN_RAISE's fields when the program raised the crash's exception itself (own_raise; error is
    the crash's `Type: message` line): where, as _where gives it (' at line N in `f`: `source`'); {} for a library's
    or Python's error, a re-raise, a raise in an except clause (it stands in for the error it caught), a raise the
    program needs to end a loop, and with the switch off."""
    if not s11_raise_on() or not error:
        return {}
    return dict(where=_where(code, line)) if own_raise(code, line, error.split(':', 1)[0].rsplit('.', 1)[-1]) else {}


def _exception_line(report, tail):
    """The failed run's exception line (`Type: message`), from the crash report or else the log tail; '' when
    there is none. Model input only: never exported."""
    lines = [line.strip() for line in (report or '').split('\n')[:1]] + [
        line.strip() for line in reversed((tail or '').split('\n'))]
    for line in lines:
        match = re.fullmatch(r'([A-Za-z_][\w.]{0,127})(?::\s?(.*))?', line)
        if match and re.search(r'(?:Error|Exception|Exit|Interrupt)$', match[1].rsplit('.', 1)[-1]):
            return line[:240]
    return ''


def _error_note(error):
    return f' (`{error}`)' if error else ''


def _function_lines(code):
    """{(innermost enclosing function or '<module>', stripped line text)} of a program's non-blank lines."""
    try:
        spans = sorted((node.lineno, node.end_lineno or node.lineno, node.name) for node in ast.walk(ast.parse(code))
                       if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)))
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return set()
    found = set()
    for number, line in enumerate(code.split('\n'), 1):
        if line.strip():
            inside = [name for start, end, name in spans if start <= number <= end]
            found.add((inside[-1] if inside else '<module>', line.strip()))
    return found


def _restored(before, after, failed):
    """The first (function, line text, error) of failed that after brings back: its line is in that function of
    after but not of before (a line kept all along is not brought back). None when there is none."""
    if not failed:
        return None
    old, new = _function_lines(before), _function_lines(after)
    return next((entry for entry in failed if entry[0] and entry[1]
                 and (entry[0], entry[1]) in new and (entry[0], entry[1]) not in old), None)


def _retry_patch(client, conversation, code, content, ask, deadline, status, reserve, canaries, pace, sleep, near,
                 kind, normalize, *, cap=None, one_request=False):
    """The one retry after a patch that changed nothing (identical code, or lines that did not splice): the
    reply as it came, then ask, which says why. None when this patch changes nothing either. Optional cap and
    one_request are used only by S12's conditional same-definition feedback; defaults retain the old retry."""
    shown = code_region(content) if isinstance(content, str) else ''
    conversation = conversation + [
        {'role': 'assistant', 'content': withhold_canaries(_fence(shown) if shown.strip() else '(no code)', canaries)},
        {'role': 'user', 'content': withhold_canaries(ask, canaries)}]
    try:
        return _patch(client, conversation, code, deadline, status, reserve, canaries, pace, sleep, near, kind,
                      normalize, cap=cap, one_request=one_request)[0]
    except (_Stop, HouseError):
        return None  # the first patch's outcome (no change) stands


def _output_files(out: Path) -> set:
    return {p.relative_to(out).as_posix() for p in out.rglob('*') if p.is_file() and not p.is_symlink()}


def _snapshot(out: Path):
    """The output tree in memory ({relative path: bytes}), or None when it exceeds REVIEW_SNAPSHOT_CAP."""
    kept, size = {}, 0
    for name in sorted(_output_files(out)):
        data = (out / name).read_bytes()
        size += len(data)
        if size > REVIEW_SNAPSHOT_CAP:
            return None
        kept[name] = data
    return kept


def _restore(out: Path, kept: dict) -> None:
    _clear(out)
    for name, data in kept.items():
        (out / name).parent.mkdir(parents=True, exist_ok=True)
        (out / name).write_bytes(data)


NORMALIZE_COUNTERS = ('normalized_files', 'normalized_csv_decimals', 'normalized_csv_booleans',
                      'normalized_json_rounded', 'normalized_json_numbers', 'normalized_json_booleans',
                      'normalized_json_integers') + CONTRACT_COUNTERS


def contract_on() -> bool:
    """T1_CONTRACT=1 (the image's flag): normalization also applies the output contract the TASK states
    (audit.normalize_contract): a CSV's stated header (same columns reordered, extra columns dropped, one column
    renamed in place), JSON template keys renamed by a constant prefix or suffix and put in the template's order
    where the TASK says so, integer cells in integer, timestamp and id columns, and the slot kind of keys a
    template elides with '...'. A file keeps these only when the TASK's statements find fewer problems in it and
    none new. The advisory flags then read the same rules. Off, outputs and flags are as before."""
    return os.environ.get('T1_CONTRACT') == '1'


def _flag_rules(instruction):
    """literal_rules, with the contract's rules when contract_on() (literal_rules alone if those fail)."""
    rules = literal_rules(instruction)
    if not contract_on():
        return rules
    try:
        return contract_literal_rules(rules, contract_rules(instruction))
    except Exception:
        return rules


def normalize_literals(instruction, out, status=None, *, integers=None):
    """Rewrite the outputs where the TASK states their literal format (audit.normalize_outputs, or
    audit.normalize_contract when contract_on()) and add the counters to status (numbers only). Never raises: a
    failure leaves the outputs as the program wrote them (or as normalize_outputs alone left them).
    The counters describe the outputs as graded: a rerun whose outputs are kept replaces them (_reset_counts).
    With `integers` (a dict, thin's outputs_fix) normalize_outputs keeps whole-number id, count and timestamp
    columns as plain integers (audit.integer_guard_rules), recording them in `integers`."""
    try:
        if contract_on() or integers is None:
            counts = normalize_contract(instruction, out) if contract_on() else normalize_outputs(
                literal_rules(instruction), out)
        else:
            from .audit import integer_guard_rules
            counts = normalize_outputs(integer_guard_rules(literal_rules(instruction), instruction, integers), out)
    except Exception:
        counts = {}
    if status is not None:
        _count_all(status, counts)
    return counts


def _count_all(status, counts):
    _count(status, 'normalized_files', 0)
    for key, value in counts.items():
        _count(status, key, value)


def _reset_counts(status, counts):
    """The normalization counters of a rerun whose outputs replaced the earlier ones: the files and values
    rewritten in the outputs now kept, not a sum over every run."""
    for key in NORMALIZE_COUNTERS:
        status.pop(key, None)
    _count_all(status, counts)


def advisory_flags(instruction, out):
    """(literal, degenerate): literal-format problems left after normalize_literals(), and degenerate outputs.
    Both are shown to the review and must not get worse under a fix; neither is a reason for a fix on its own."""
    try:
        rules = _flag_rules(instruction)
        return literal_flags(rules, out), degenerate_flags(rules, out)
    except Exception:
        return [], []


def _flag_key(flag):
    """A flag's identity across two runs: its text with the numbers left out."""
    return re.sub(r'[-+]?\d[\d.eE+-]*', '#', flag)


def _no_worse(before, after):
    """No flag that the earlier outputs lacked: a fix may leave a flag standing but never add one."""
    known = {_flag_key(flag) for flag in before}
    return all(_flag_key(flag) in known for flag in after)


def _resolved(flags, after, entries, again, failed=(), fits_gone=False):
    """Whether the review and the flags judge a fix (whose outputs changed: review_outputs checks that first)
    to have resolved what it was asked to fix: no quote of the original review items or failed self-checks is
    listed again by the review of the new outputs (items it lists anew do not count: the first review never
    listed them), and at least one original problem is gone (fits_gone: a fit report the fix was asked about no
    longer appears). The program's own self-check never decides this: a check may be wrong, so only the review
    and the deterministic flags judge. Without a second review only a fix of deterministic flags (or fit reports)
    alone is judged."""
    quotes = [quote for quote, _, _ in entries] + [quote for quote, _ in failed]
    if quotes and again is None:
        return False
    if any(same_quote(quote, other) for quote in quotes for other, _, _ in again or ()):
        return False
    return bool(quotes) or bool({_flag_key(f) for f in flags} - {_flag_key(f) for f in after}) or fits_gone


def _review(client, messages, code, out, instruction, flags, deadline, pace, sleep, canaries, reserve, hints=(),
            status=None):
    """(entries, digest) from one review request: the review items (quote, output, problem) that quote the task
    and name a shown output, given the contract flags and the advisory hints. Items dropped because they only
    confirm compliance are counted in status['review_affirmations']."""
    shown = digest(out)  # withheld below with the whole request
    listed = REVIEW_FLAGS.format(flags='\n'.join('- ' + f for f in flags)) if flags else ''
    listed += REVIEW_HINTS.format(hints='\n'.join('- ' + h for h in hints)) if hints else ''
    conversation = messages + [{'role': 'assistant', 'content': withhold_canaries(_fence(code), canaries)},
                               {'role': 'user', 'content': withhold_canaries(
                                   REVIEW.format(digest=shown, flags=listed), canaries)}]
    reply = _request(client, conversation, deadline, pace, 'review', sleep, need=pace.patch(), reserve=reserve)
    entries, affirmations = review_parse(_text(reply), instruction, shown)
    if status is not None:
        _count(status, 'review_affirmations', affirmations)
    return entries, shown


def review_outputs(client, messages, code, out, instruction, deadline, status, pace, last_exec, *, canaries=(),
                   replay=_NO_REPLAY, task_dir=None, path_index=None, sleep=None, checks=None, rounds=1,
                   health=None, model=True):
    """After a clean exit: the instruction's output contract is checked, the model reviews the outputs against
    the task, and a guarded fix round follows when either finds a problem (with checks, a failed self-check
    whose quote is the TASK's own words is one too). A fix that changes nothing gets one retry. It is kept only
    when its rerun exits 0, keeps every output file, changes the outputs, adds no deterministic flag (contract,
    tolerance, sanity or advisory) and resolves what it was asked to fix (_resolved); otherwise the earlier
    program's outputs and measured fields are restored. A kept fix is followed by another round, up to rounds,
    on what the second review and the checks still find. When the review request itself fails (HouseError) the
    round still runs on the deterministic flags alone, if any (review_failed_fix): with no review item to judge,
    a fix is then kept only when those flags fall. Returns (program, request fields of the last kept fix, or
    None). With health (the last clean run's FitWarnings), the fit non-convergence that run reported goes to the
    review as advisory hints (fit_warnings counts them): like the other hints, a fix may leave one standing but
    never add one (a report counts by its message, whatever its count and whether the hints show it). Under
    T1_LIBHINT, when the TASK names a method that a library fit computes and the program does not call it
    (library_fits), each such report is also an item of the fix round, which it opens by itself like a
    deterministic flag (fit_fix_flags counts them): a fix is then kept when one of those reports is gone, and not
    only silenced (fit_reports_gone). With model False (T1_S10) no review request is sent, neither before the round
    nor after a fix: the round runs on the deterministic flags and fit reports alone, as review_failed_fix's does,
    and a fix is kept under the same rule, judged by the flags alone (_resolved without a second review). With
    neither flags nor fit reports it is clean before the time gate, which only guards a request it would send.
    Under T1_S11_LINT the deterministic flags also take s11_flags, of the outputs and of the program that ran, so a
    fix is kept only when one of them clears and none is added; review_contract_flags leaves them out (it decides a
    fresh program, which is judged by the contract flags alone), and s11_lint_output and s11_lint_code count them."""
    contract = output_contract(instruction)
    health = health if health is not None else FitWarnings()
    lint, linted = s11_lint_on(), {}  # T1_S11_LINT: its flags join the deterministic ones, for the program that ran

    def deterministic(program):
        found = contract_flags(contract, out) + bound_flags(instruction, out) + sanity_flags(out)
        if lint:
            output, static = s11_flags(instruction, out, program, found)
            linted.update(output=len(output), code=len(static))
            found = found + output + static
        return found

    def observed(program):
        literal, degenerate = advisory_flags(instruction, out)
        return deterministic(program), literal + degenerate, failed_checks(checks, instruction)[0]

    def hints_for(hints, failed):  # with the fit reports of the last clean run
        return hints + health.hints() + [check_flag(quote, detail) for quote, detail in failed]
    flags = deterministic(code)
    literal, degenerate = advisory_flags(instruction, out)
    warned = health.hints()
    hints, fits = literal + degenerate, health.keys()  # fit reports are compared by message, whatever their count
    failed, unquoted, literal_owned = failed_checks(checks, instruction)
    fit_named = library_fits(messages, code) if warned else []
    fitted, fit_open = health.fix_items(fit_named)
    status.update(review_contract_flags=len(flags) - linted.get('output', 0) - linted.get('code', 0), review_items=0,
                  literal_flags=len(literal), degenerate_flags=len(degenerate), fit_warnings=len(warned))
    if lint:  # the contract count stays the contract's: it decides a fresh program after a rejected fix
        status.update(s11_lint_output=linted['output'], s11_lint_code=linted['code'])
    if fitted:
        status['fit_fix_flags'] = len(fitted)
    if checks is not None:
        status.update(selfcheck_written=checks.written, selfcheck_total=len(checks.items),
                      selfcheck_failed=len(failed), selfcheck_unquoted=unquoted, selfcheck_literal=literal_owned)
    if not model and not (flags or fitted):  # T1_S10: nothing for the flags-only round to fix, whatever the clock
        status['review_status'] = 'clean'
        return code, None
    if deadline - CLOCK.now() < pace.patch() + EXEC_FLOOR:
        status['review_status'] = 'skipped_time'
        return code, None
    try:
        if not model:  # T1_S10: the flags-only round (a failed self-check needs the review to judge its fix)
            entries, failed = [], []
        else:
            entries, _ = _review(client, messages, code, out, instruction, flags, deadline, pace, sleep, canaries,
                                 EXEC_FLOOR, hints_for(hints, failed), status)
    except (_Stop, HouseError) as exc:
        status['review_status'] = 'failed'
        if isinstance(exc, _Stop) or not (flags or fitted):
            return code, None
        # The request failed, not the outputs: the deterministic flags still stand, so the fix round runs on them
        # alone (a failed self-check needs the review to judge its fix). An unparseable reply lists nothing and
        # already reaches the round the same way.
        _count(status, 'review_failed_fix')
        entries, failed = [], []
    status['review_items'] = len(entries)
    normalize = (lambda text: prepared(text, task_dir, path_index)) if task_dir else None
    kept_fields = None
    for round_ in range(rounds):
        listed = [(quote, detail) for quote, detail in failed
                  if not any(same_quote(quote, other) for other, _, _ in entries)]
        problems = ([f'- {flag}' for flag in flags] + [f'- {format_entry(entry)}' for entry in entries]
                    + [f'- {check_flag(quote, detail)}' for quote, detail in listed] + [f'- {item}' for item in fitted])
        if not problems:
            status.setdefault('review_status', 'clean')
            break
        kept = _snapshot(out)
        if kept is None:
            status.setdefault('review_status', 'too_large')
            break
        kept_checks = (list(checks.items), checks.written) if checks is not None else None
        kept_health = list(health.items)
        _count(status, 'review_rounds')
        if failed:  # also when the review repeated a failed check as its own item
            _count(status, 'selfcheck_rounds')
        ask = FIX_OUTPUTS.format(items='\n'.join(problems), note=(CHECK_FIX if failed else '')
                                 + library_fix(messages, code, 'a listed output comes from'))
        conversation = messages + [{'role': 'assistant', 'content': withhold_canaries(_fence(code), canaries)},
                                   {'role': 'user', 'content': withhold_canaries(ask, canaries)}]
        try:
            fixed, why, content = _patch(client, conversation, code, deadline, status, last_exec + 10, canaries,
                                         pace, sleep, None, 'review_patch', normalize)
            if fixed is None:  # the same code, or lines that did not splice: one retry that says why
                fixed = _retry_patch(client, conversation, code, content,
                                     REVIEW_RETRY.format(why=why, first=problems[0][2:].strip()[:300]), deadline,
                                     status, last_exec + 10, canaries, pace, sleep, None, 'review_retry', normalize)
                if fixed is not None:
                    _count(status, 'review_retry_patched')
        except (_Stop, HouseError):
            fixed = None
        if fixed is None:
            status.setdefault('review_status', 'no_patch')
            break
        fields = {k: client.diagnostics.get(k) for k in ('http_status', 'finish_reason')}
        if task_dir is not None:
            fixed = prepared(fixed, task_dir, path_index)
        measured = {k: status[k] for k in MEASURED if k in status}
        accepted, again = False, None
        try:
            status['program_sha256'] = hashlib.sha256(fixed.encode()).hexdigest()
            replay.put('program.py', fixed)
            started = CLOCK.now()
            tail, report = _run_program(fixed, out, deadline, canaries, status, clear=True, replay=replay,
                                        input_root=task_dir, checks=checks, health=health)
            if status['status'] == 'execution_failed':  # a fix that crashes gets one crash repair first
                _count(status, 'review_repairs')
                repaired = repair_program(client, messages, fixed, tail, status.get('exit_code'), [], deadline,
                                          status, CLOCK.now() - started + 10, canaries=canaries, pace=pace,
                                          report=report, normalize=normalize)
                if repaired is not None:
                    fixed = prepared(repaired, task_dir, path_index) if task_dir is not None else repaired
                    fields = {k: client.diagnostics.get(k) for k in ('http_status', 'finish_reason')}
                    status['program_sha256'] = hashlib.sha256(fixed.encode()).hexdigest()
                    replay.put('program.py', fixed)
                    _run_program(fixed, out, deadline, canaries, status, clear=True, replay=replay,
                                 input_root=task_dir, checks=checks, health=health)
            if status['status'] == 'executed_ungraded' and set(kept) <= _output_files(out):
                rewritten = normalize_literals(instruction, out)
                after, hints_after, failed_after = observed(fixed)
                if _snapshot(out) == kept:  # outputs that did not change cannot resolve what the outputs contradict
                    _count(status, 'review_unchanged')
                elif (_no_worse(flags, after) and _no_worse(hints, hints_after) and health.keys() <= fits
                      and not runtime.regression(measured, status)):
                    try:
                        again = (_review(client, messages, fixed, out, instruction, after, deadline, pace, sleep,
                                         canaries, 10, hints_for(hints_after, failed_after), status)[0]
                                 if model else None)
                    except (_Stop, HouseError):
                        again = None
                    accepted = _resolved(flags, after, entries, again, listed,
                                         fit_reports_gone(code, fixed, fit_open, health, fit_named))
        except Exception:  # a setup failure or no time left: the earlier outputs stand
            accepted = False
        if not accepted:
            _restore(out, kept)
            if checks is not None:
                checks.items, checks.written = kept_checks
            health.items = kept_health
            for key in MEASURED:
                status.pop(key, None)
            status.update(measured)
            status.setdefault('review_status', 'fix_rejected')
            break
        _reset_counts(status, rewritten)
        if failed:
            _count(status, 'selfcheck_accepted')
        status['review_status'] = 'fixed'
        code, kept_fields = fixed, fields
        flags, hints, failed, entries = after, hints_after, failed_after, again or []
        fits = health.keys()
        fit_named = library_fits(messages, code) if health.items else []
        fitted, fit_open = health.fix_items(fit_named)
    return code, kept_fields


def fresh_attempt(client, prompt, code, messages, out, instruction, deadline, status, pace, last_exec, *,
                  canaries=(), replay=_NO_REPLAY, task_dir=None, path_index=None, sleep=None, ran=(), checks=None,
                  health=None):
    """One independent program from the task prompt without the plan, after repair or the output fix reached a
    dead end (a wrong plan fixes the same error in every program written from it). The new program gets up to
    FRESH_CRASH_REPAIRS crash repairs, placed by the failing line, and one that goes back to a program or a line
    that already failed gets one retry saying so, as in the main chain. It is kept only if it does better: a clean exit where the old program crashed, or, where
    both exit cleanly, every old output file and fewer contract and tolerance problems. The model's review
    never decides it: it also flags programs that pass. Otherwise the old outputs and measured fields are
    restored. Returns (program, messages, request fields), with None fields when nothing changed."""
    if client.attempts > client.max_requests - 2 or deadline - CLOCK.now() < pace.full() + last_exec + EXEC_FLOOR:
        status['fresh_status'] = 'skipped_time'
        return code, messages, None
    contract = output_contract(instruction)
    problems = lambda: (len(contract_flags(contract, out)) + len(bound_flags(instruction, out))
                        + len(sanity_flags(out)))
    hints = lambda: sum(advisory_flags(instruction, out), [])
    old_ok = status.get('status') == 'executed_ungraded'
    old_flags, old_hints = (problems(), hints()) if old_ok else (None, None)
    kept = _snapshot(out)
    if kept is None:
        status['fresh_status'] = 'failed'
        return code, messages, None
    measured = {k: status[k] for k in MEASURED if k in status}
    kept_checks = (list(checks.items), checks.written) if checks is not None else None
    kept_health = list(health.items) if health is not None else None
    fresh = [{'role': 'user', 'content': prompt}]
    normalize = (lambda text: prepared(text, task_dir, path_index)) if task_dir else None
    better, fields, new = False, None, None
    try:
        new = generate_program(client, fresh, deadline, status, canaries=canaries, pace=pace, sleep=sleep)
        new = precheck_fixed(new, status, (SELFCHECK_NAME,) if checks is not None else ())
        fields = {k: client.diagnostics.get(k) for k in ('http_status', 'finish_reason')}
        tried, failed_lines = set(ran) | {_shape(code)}, []
        if _shape(normalize(new) if normalize else new) in tried:
            status['fresh_status'] = 'duplicate'  # the same program fails the same way: nothing is run
            return code, messages, None
        for attempt in range(FRESH_CRASH_REPAIRS + 1):
            new = normalize(new) if normalize else new
            tried.add(_shape(new))
            status['program_sha256'] = hashlib.sha256(new.encode()).hexdigest()
            replay.put('program.py', new)
            started = CLOCK.now()
            tail, report = _run_program(new, out, deadline, canaries, status, clear=True, replay=replay,
                                        input_root=task_dir, checks=checks, health=health)
            if status['status'] != 'execution_failed' or attempt == FRESH_CRASH_REPAIRS:
                break
            # As in the main repair chain: a patch that goes back to what already failed gets one retry saying so.
            line = status.get('execution_diagnostics', {}).get('generated_line')
            failure = _failing_statement(new, line) + (_exception_line(report, tail),)
            if failure[0] and failure[1]:
                failed_lines.append(failure)
            repaired = repair_program(client, fresh, new, tail, status.get('exit_code'), [], deadline, status,
                                      CLOCK.now() - started + 10, canaries=canaries, pace=pace, report=report,
                                      normalize=normalize, near=line, ran=tried, failed=failed_lines)
            if repaired is None or _shape(normalize(repaired) if normalize else repaired) in tried:
                break  # as in the main chain (noop): a program that already failed is not run again
            new, fields = repaired, {k: client.diagnostics.get(k) for k in ('http_status', 'finish_reason')}
        if status['status'] == 'executed_ungraded':
            rewritten = normalize_literals(instruction, out)
            if not old_ok:
                better = True
            elif set(kept) <= _output_files(out):  # the model's review may veto a fix but never promote one
                better = problems() < old_flags and _no_worse(old_hints, hints())
    except (_Stop, HouseError):
        better = False
    except Exception:  # a setup failure: the old outcome stands
        better = False
    if better and runtime.regression(measured, status):
        better = False
    if better:
        _reset_counts(status, rewritten)
        status['fresh_status'] = 'kept'
        return new, fresh, fields
    _restore(out, kept)
    if checks is not None:
        checks.items, checks.written = kept_checks
    if health is not None:
        health.items = kept_health
    for key in MEASURED:
        status.pop(key, None)
    status.update(measured, fresh_status='rejected')
    return code, messages, None


# T1_S12_POST (off unless set; acts only under T1_S10): post-run verification. Stack 10 stops at the first clean run
# with 190-370 s and 15-18 requests unused (r13b), yet in 8 of 14 failures the error showed in the program's own
# outputs or in flags the controller computed and dropped (degenerate_flags on intraday and creditmetrics). After the
# clean run, and after review_outputs' flags-only round, s12_post computes flags with zero requests: the advisory flags
# (a header-only CSV, an all-zero container, a NaN column, a literal format), the program's own failed validations and
# comparison errors (selfval_flags), and code lints (s12_lint). T1_S12_INERT reruns the program with each numeric input
# parameter moved (s12_inert) and flags one that no output depends on; T1_S12_AUDIT asks once whether the code
# does what the D lines and the TASK say (s12_audit). Up to S12_ROUNDS fix rounds then send the flags as evidence. A
# fix is kept only when its rerun exits 0, every output keeps its keys, columns and row counts, no flag of any kind is
# added and every self-validation and lint flag it was asked to fix is gone (S12_HARD). The ask lets an advisory,
# inputs or audit item stand where the TASK wants it so, so those need not clear; a round of them alone is kept only
# when one moved (a flag gone, a probed input now read, or, for audit items, changed outputs). Otherwise the program
# and its outputs go back.
S12_ROUNDS = 2
S12_ITEMS = 8  # flags listed in one fix request; the rest wait for the next round
S12_FLAG_CHARS = 700  # characters of one listed flag
S12_KINDS = ('degenerate', 'literal', 'selfval', 'lint', 'inert', 'audit')
S12_HARD = ('selfval', 'lint')  # kinds a kept fix must clear; the others may be what the TASK asks
S12_STOPS = ('clean', 'cleared', 'rounds', 'no_patch', 'reverted', 'skipped_time', 'skipped_requests', 'failed',
             'too_large')
S12_AUDIT_BUDGET, S12_AUDIT_WORDS = 1500, 300  # 1,500 + 5.5 x 300 = 3,150 of S10_TOKEN_CEILING
S12_AUDIT_ITEMS = 5
S12_AUDIT_KEEP = 2  # requests the audit leaves: the fix and its continuation
S12_AUDIT_CODE_MIN = 8  # characters of an audit item's code line, which must be one of the program's
S12_AUDIT_STATUSES = ('items', 'none', 'unparsed', 'no_notes', 'skipped_time', 'skipped_requests', 'skipped_breaker',
                      'failed')
S12_INERT_KEYS = 6
S12_INERT_RUNS = 3  # the probe runs only while the time left exceeds this x runtime x keys + a fix round's reserve
S12_INERT_BYTES = 64 * 1024  # a larger parameter file is data
S12_INERT_DEPTH = 3  # JSON object levels searched for scalars
S12_INERT_SCALARS = 64  # a file with more numeric scalars holds data, not parameters
# Outputs the same up to this relative difference on every number count as unmoved: ewma's ignored window moved its
# outputs by about 1e-15 relative (rounding in a recursion one step shorter), which a byte comparison calls a change.
S12_INERT_RTOL = 1e-9
S12_INERT_STATUSES = ('flagged', 'none', 'unverified', 'no_params', 'no_paths', 'skipped_time', 'too_large', 'failed')
# Keys whose small change need not move any output: a seed, a solver's tolerance or iteration cap, output formatting.
S12_INERT_SKIP = frozenset({'seed', 'rng', 'canary', 'tol', 'tolerance', 'eps', 'epsilon', 'iter', 'iters',
                            'iterations', 'maxiter', 'precision', 'decimals', 'digits', 'round', 'rounding', 'verbose',
                            'debug', 'dpi', 'timeout', 'figsize'})
S12_TIMED_BYTES = 64 * 1024  # bytes of each input text file read for a time of day
S12_TIMED = re.compile(r'\b\d{4}-\d{2}-\d{2}[T ](\d{2}):(\d{2})(?::(\d{2}))?')
S12_TEXT_INPUTS = ('.csv', '.tsv', '.json', '.jsonl', '.txt', '.yaml', '.yml', '.toml')
S12_EMPTY = ('csv', 'tsv', 'json', 'jsonl', 'txt', 'md')  # output kinds whose empty file is a degenerate flag
S12_FIX = ('The program above ran and exited 0, but checks of its own outputs and code found:\n{items}\n{note}Fix the '
           'cause of each item and change nothing else. An item marked (advisory), (inputs) or (audit) may still be '
           'what the TASK asks: then leave it. Never weaken, remove or bypass a check, never write a flag or a value '
           'directly to make a check pass, and keep every output file, key, column and row. ' + FALLBACK_FIX
           + PATCH_FORMAT)
S12_POST_LIGHT = ('Your reply hit the output limit without a usable change. The program exited 0, but the listed '
                  'output issues remain. Fix only their cause: send one short complete def, or a small changed hunk '
                  'with unique unchanged context. Do not send the whole program, change the TASK semantics or '
                  'weaken checks. No explanation. ' + PATCH_FORMAT)
S12_KEEP = ('Keep each of YOUR DECISIONS unless an item shows that the decision itself is wrong (a tolerance in the '
            'wrong units, a transform paired the wrong way, a window in the wrong place); then change that decision in '
            'the code and mark the line `# D<n> changed`.')
S12_AGAIN = 'Your previous change was not kept: {why}. '
S12_WHY = {'failed': 'the program then failed', 'missing': 'an output file was no longer written',
           'shape': 'it changed the shape of an output ({detail})', 'added': 'it added a problem ({detail})',
           'left': 'this was still found ({detail})', 'unchanged': 'its outputs did not change',
           'error': 'the program could not be run again'}
S12_AUDIT = ('AUDIT: the program above ran and exited 0. Its outputs:\n```\n{digest}\n```\nDo not write code. Compare '
             'the program with YOUR DECISIONS (the D lines in the first message) and with the TASK: list at most '
             '{items} places where the code does not do what a D line or a TASK sentence says (a different formula, '
             'index, label, order, window, unit, tail or stopping rule). One line each, exactly: ITEM: <the D line ID, '
             'or the TASK words in quotes> | CODE: <the code line, copied exactly> | OUTPUT: <the output value it '
             'affects, or n/a> | PROBLEM: <what differs>. Only places where the code differs from the line or the '
             'TASK, not style and not a decision you would now take differently. If there is none, reply exactly '
             'NONE. Write at most {words} words in all.')
S12_AUDIT_FLAG = '(audit) {ref}: the code line `{code}`{output} {problem}'
S12_INERT_FLAG = ('(inputs) {file}: changing `{key}` from {old} to {new} left every output the same (each number '
                  'within 1e-9 relative): the program does not use it, or not where the TASK applies it{told}. Apply '
                  'it as the TASK says; where the TASK says it has no effect on these outputs, leave the program as it '
                  'is')
S12_TOLD = '; the TASK says: "{quote}"'
S12_QUOTE_CHARS = 300
_S12_ITEM = re.compile(r'ITEM:\s*(.+?)\s*\|\s*CODE:\s*(.+?)\s*\|\s*OUTPUT:\s*(.+?)\s*\|\s*PROBLEM:\s*(.+)')
_S12_NOTE_ID = re.compile(r'^D(\d{1,2}) \| ', re.M)
_S12_LINES = {'.toml': re.compile(r'^(\s*)([A-Za-z_][\w\-]*)(\s*=\s*)([-+]?\d[\d_]*(?:\.\d*)?(?:[eE][-+]?\d+)?)'
                                  r'(\s*(?:#.*)?)$'),
              '.yaml': re.compile(r'^(\s*)([A-Za-z_][\w\-]*)(\s*:\s*)([-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?)'
                                  r'(\s*(?:#.*)?)$')}
_S12_LINES['.yml'] = _S12_LINES['.yaml']
_S12_TOML_TABLE = re.compile(r'^\s*\[\s*([^\[\]#]+?)\s*\](?!\])\s*(?:#.*)?$')
_S12_YAML_PARENT = re.compile(r'^(\s*)([A-Za-z_][\w\-]*)\s*:\s*([|>][^#]*)?(?:#.*)?$')
# Frozen with the texts and thresholds above; a test pins it.
S12_PROMPT_SHA = hashlib.sha256(json.dumps([
    S12_ROUNDS, S12_ITEMS, S12_FLAG_CHARS, S12_HARD, S12_AUDIT_BUDGET, S12_AUDIT_WORDS, S12_AUDIT_ITEMS, S12_AUDIT_KEEP,
    S12_AUDIT_CODE_MIN, S12_INERT_KEYS, S12_INERT_RUNS, S12_INERT_BYTES, S12_INERT_DEPTH, S12_INERT_SCALARS,
    S12_INERT_RTOL, sorted(S12_INERT_SKIP),
    S12_TIMED_BYTES, S12_TIMED.pattern, S12_EMPTY, S12_FIX, S12_POST_LIGHT, S12_KEEP, S12_AGAIN, S12_WHY, S12_AUDIT,
    S12_AUDIT_FLAG, S12_INERT_FLAG, S12_TOLD, S12_QUOTE_CHARS], sort_keys=True).encode()).hexdigest()


def s12_post_on() -> bool:
    """T1_S12_POST=1 (off in the image), only under T1_S10: post-run verification after the last clean run (s12_post),
    a crash ask that lists a numpy ufunc given too many arguments (repair_program), and before every run a call of a
    numpy name the program never binds made numpy's (library_fixed). Off, every request and program is stack 11's."""
    return os.environ.get('T1_S12_POST') == '1' and (s10_on() or minimal_on())


def s12_inert_on() -> bool:
    """T1_S12_INERT=1, only under T1_S12_POST: the input parameter probe (s12_inert)."""
    return os.environ.get('T1_S12_INERT') == '1' and s12_post_on()


def s12_audit_on() -> bool:
    """T1_S12_AUDIT=1, only under T1_S12_POST: one audit request of the code against the notes (s12_audit)."""
    return os.environ.get('T1_S12_AUDIT') == '1' and s12_post_on()


def s12_timed(files) -> bool:
    """Whether an input text file carries a time of day (a timestamp not at midnight) in its first S12_TIMED_BYTES
    bytes: the day-count lint's condition."""
    for name in files:
        path = Path(name)
        if path.suffix.lower() not in S12_TEXT_INPUTS or path.is_symlink():
            continue
        try:
            with path.open('rb') as stream:
                text = stream.read(S12_TIMED_BYTES).decode('utf-8', 'replace')
        except OSError:
            continue
        if any(any(int(part or 0) for part in match.groups()) for match in S12_TIMED.finditer(text)):
            return True
    return False


def _s12_words(key):
    return set(re.findall(r'[a-z0-9]+', re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', str(key)).lower()))


def _s12_left_out(label, canaries):
    """A key the probe leaves alone: one naming a seed, a tolerance, an iteration cap, a precision or formatting
    (S12_INERT_SKIP), a canary, or private ('_')."""
    words = _s12_words(label)
    return (bool(words & S12_INERT_SKIP) or {'random', 'state'} <= words or label.startswith('_')
            or label.split('.')[-1].startswith('_') or bool(near_canaries(label))
            or any(guid.lower() in label.lower() for guid in canaries))


def _s12_moved(value):
    """The probe's value: an integer plus one, a float plus 10%; None for a float zero, which no percentage moves."""
    if isinstance(value, int):
        return value + 1
    return value * 1.1 if value else None


def _s12_scalars(data, path=(), depth=0):
    """(key path, number) of a JSON document's numeric scalars under object keys, never inside a list."""
    if isinstance(data, dict) and depth < S12_INERT_DEPTH:
        for key, value in data.items():
            if not isinstance(key, str):
                continue
            if (isinstance(value, int) and not isinstance(value, bool)) or (
                    isinstance(value, float) and math.isfinite(value)):
                yield path + (key,), value
            elif isinstance(value, dict):
                yield from _s12_scalars(value, path + (key,), depth + 1)


def _s12_parents(lines, suffix):
    """{line index: dotted path of the tables or mappings above it} for the YAML or TOML lines a scalar may sit on, so
    that two keys of one name label apart (ewma.window, garch.window); a line inside a list, an array of tables or a
    block scalar has none (a row's data, as a JSON scalar inside a list is never probed)."""
    parents, stack, table = {}, [], ()
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            continue
        if suffix == '.toml':
            match = _S12_TOML_TABLE.match(line)
            if match or stripped.startswith('['):
                table = tuple(part.strip().strip('"\'') for part in match.group(1).split('.')) if match else None
                continue
            if table is not None:
                parents[i] = table
            continue
        indent = len(line) - len(line.lstrip(' '))
        while stack and stack[-1][0] >= indent:
            stack.pop()
        if any(key is None for _, key in stack):
            continue  # a list item's or a block scalar's line
        parent = _S12_YAML_PARENT.match(line)
        if stripped.startswith('-'):
            stack.append((indent, None))
        elif parent:
            stack.append((indent, None if parent.group(3) else parent.group(2)))
        else:
            parents[i] = tuple(key for _, key in stack)
    return parents


def _s12_file_edits(path, rel, canaries):
    """[(rel, label, old, new, text)] for one parameter file: each numeric scalar moved (_s12_moved) in a copy of the
    file's text; a JSON copy is the parsed document written again, a YAML or TOML copy the text with that one line
    changed, labelled with its parents' path (_s12_parents; a TOML copy must still parse and hold the new value at
    that path). [] for a file of more than S12_INERT_SCALARS scalars (data)."""
    suffix, found = path.suffix.lower(), []
    try:
        if path.is_symlink() or not path.is_file() or path.stat().st_size > S12_INERT_BYTES:
            return []
        raw = path.read_text(encoding='utf-8')
    except (OSError, UnicodeError):
        return []
    if suffix == '.json':
        try:
            data = json.loads(raw)
        except (ValueError, RecursionError):
            return []
        scalars = list(_s12_scalars(data))
        if len(scalars) > S12_INERT_SCALARS:
            return []
        for keys, value in scalars:
            label, new = '.'.join(keys), _s12_moved(value)
            if new is None or _s12_left_out(label, canaries):
                continue
            copy = json.loads(raw)
            node = copy
            for key in keys[:-1]:
                node = node[key]
            node[keys[-1]] = new
            found.append((rel, label, value, new, json.dumps(copy, indent=2, ensure_ascii=False) + '\n'))
        return found
    lines = raw.split('\n')
    matches = [(i, _S12_LINES[suffix].match(line)) for i, line in enumerate(lines)]
    matches = [(i, match) for i, match in matches if match]
    if len(matches) > S12_INERT_SCALARS:
        return []
    parents = _s12_parents(lines, suffix)
    labels = {i: '.'.join(parents[i] + (match.group(2),)) for i, match in matches
              if i in parents and len(parents[i]) < S12_INERT_DEPTH}
    named = Counter(labels.values())
    for i, match in matches:
        if named[labels.get(i)] != 1:
            continue  # inside a list, too deep, or a label two lines share: which one a flag names is unclear
        text, label = match.group(4), labels[i]
        try:
            value = int(text) if re.fullmatch(r'[-+]?\d[\d_]*', text) else float(text)
        except ValueError:
            continue
        new = _s12_moved(value)
        if new is None or _s12_left_out(label, canaries):
            continue
        body = '\n'.join(lines[:i] + [match.group(1) + match.group(2) + match.group(3) + repr(new) + match.group(5)]
                         + lines[i + 1:])
        if suffix == '.toml':
            try:
                node = tomllib.loads(body)
                for part in label.split('.'):
                    node = node[part]
            except (tomllib.TOMLDecodeError, ValueError, KeyError, TypeError):
                continue
            if node != new:
                continue  # the line is not the key its label names (a multi-line string's text)
        found.append((rel, label, value, new, body))
    return found


def s12_param_edits(task_dir, files, instruction, canaries=(), limit=S12_INERT_KEYS):
    """[(relative path, label, old, new, new text)]: at most limit numeric scalars of the input parameter files (JSON,
    YAML or TOML of at most S12_INERT_BYTES bytes), each moved in a copy of its file (_s12_file_edits): keys the TASK
    names first, then in file order."""
    found = []
    for name in files:
        path = Path(name)
        if path.suffix.lower() not in ('.json', '.yaml', '.yml', '.toml'):
            continue
        try:
            rel = path.relative_to(task_dir).as_posix()
        except ValueError:
            continue
        found += _s12_file_edits(path, rel, canaries)
    named = [edit for edit in found if _names(instruction, edit[1].rsplit('.', 1)[-1])]
    return (named + [edit for edit in found if edit not in named])[:limit]


def _s12_mirror(task_dir, root):
    """A tree at root of links to every input file under task_dir (directories made, links in task_dir skipped), so
    that one file can be swapped for a moved copy; False when it would hold more than PATH_INDEX_LIMIT files or
    cannot be made."""
    count = 0
    try:
        for folder, directories, names in os.walk(task_dir):
            directories[:] = sorted(d for d in directories if not os.path.islink(os.path.join(folder, d)))
            rel = os.path.relpath(folder, task_dir)
            target = root if rel == '.' else root / rel
            target.mkdir(parents=True, exist_ok=True)
            for name in sorted(names):
                source = os.path.join(folder, name)
                if os.path.islink(source):
                    continue
                count += 1
                if count > PATH_INDEX_LIMIT:
                    return False
                os.symlink(source, target / name)
    except OSError:
        return False
    return True


def _s12_redirect(code, root, mirror):
    """(code, edits): every string literal, or an f-string's leading part, that starts with the input root as a whole
    path (root, or root/...) names the mirror instead; placed as rewrite_input_paths places its edits."""
    if root not in code:
        return code, 0
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(code).readline))
    except (tokenize.TokenError, SyntaxError, ValueError):
        return code, 0
    middle = getattr(tokenize, 'FSTRING_MIDDLE', None)
    edits, previous = [], None
    for token in tokens:
        if token.start[0] == token.end[0] and (token.type == tokenize.STRING or (
                token.type == middle and previous is not None and previous.type == tokenize.FSTRING_START)):
            text, at = token.string, token.string.find(root)
            while at >= 0:
                end = at + len(root)
                lead = at == 0 if token.type == middle else at > 0 and text[at - 1] in '\'"'
                if lead and (end == len(text) or text[end] in '/\'"'):
                    edits.append((token.start[0], token.start[1] + at, token.start[1] + end))
                at = text.find(root, at + 1)
        previous = token
    if not edits:
        return code, 0
    lines = code.split('\n')
    for row, begin, end in sorted(edits, reverse=True):
        lines[row - 1] = lines[row - 1][:begin] + mirror + lines[row - 1][end:]
    fixed = '\n'.join(lines)
    try:
        ast.parse(fixed)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return code, 0
    return fixed, len(edits)


def _s12_close(a, b):
    """Whether two parsed values are the same up to S12_INERT_RTOL on every number (NaN equal to NaN)."""
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a == b or math.isclose(a, b, rel_tol=S12_INERT_RTOL, abs_tol=1e-15) or (a != a and b != b)
    if isinstance(a, dict) and isinstance(b, dict):
        return list(a) == list(b) and all(_s12_close(a[key], b[key]) for key in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_s12_close(x, y) for x, y in zip(a, b))
    return a == b


def _s12_cells(data, delimiter):
    """A CSV's rows with each cell a number where it reads as one."""
    rows = []
    for row in csv.reader(io.StringIO(data.decode('utf-8-sig', errors='replace')), delimiter=delimiter):
        cells = []
        for cell in row:
            try:
                cells.append(float(cell))
            except ValueError:
                cells.append(cell)
        rows.append(cells)
    return rows


def _s12_same(before, after):
    """Whether two output snapshots hold the same files with the same content: equal bytes, or JSON or CSV whose numbers
    agree within S12_INERT_RTOL and whose other content is equal (_s12_close). None for a snapshot: not the same."""
    if before is None or after is None or before.keys() != after.keys():
        return False
    for name, data in before.items():
        new, suffix = after[name], name.rsplit('.', 1)[-1].lower() if '.' in name else ''
        if data == new:
            continue
        try:
            if suffix == 'json':
                same = _s12_close(json.loads(data.decode('utf-8-sig')), json.loads(new.decode('utf-8-sig')))
            elif suffix in ('csv', 'tsv'):
                delimiter = '\t' if suffix == 'tsv' else ','
                same = _s12_close(_s12_cells(data, delimiter), _s12_cells(new, delimiter))
            else:
                same = False
        except (ValueError, RecursionError, csv.Error):
            same = False
        if not same:
            return False
    return True


def _s12_probe(code, out, task_dir, edits, instruction, canaries, status, health, deadline, last_exec, reserve):
    """{(file, label): whether the outputs changed (_s12_same), None where the run failed or had no time} for each
    edit: the program runs on a mirror of the inputs (_s12_mirror, _s12_redirect) with that one file moved, each run
    within twice the measured runtime plus 10 s and never into reserve; then the outputs, the measured fields, the run
    records and the fit reports go back. A status enum instead when the probe cannot run: too_large, no_paths or
    failed."""
    kept = _snapshot(out)
    if kept is None:
        return 'too_large'
    measured = {k: status[k] for k in MEASURED if k in status}
    runs, fits = list(status.get('runs', [])), list(health.items) if health is not None else None
    results = {}
    try:
        with tempfile.TemporaryDirectory(prefix='t1-s12-', dir='/tmp', ignore_cleanup_errors=True) as tmp:
            mirror = Path(tmp) / 'input'
            if not _s12_mirror(task_dir, mirror):
                return 'failed'
            probe, moved = _s12_redirect(code, str(task_dir), str(mirror))
            if not moved:
                return 'no_paths'
            for rel, label, _, _, text in edits:
                limit = min(deadline - reserve, CLOCK.now() + 2 * last_exec + 10)
                if limit - CLOCK.now() < last_exec + 5:
                    break
                target = mirror / rel
                target.unlink()
                target.write_text(text)
                try:
                    _run_program(probe, out, limit, canaries, status, clear=True, input_root=task_dir)
                    _count(status, 's12_inert_runs')
                    if status['status'] == 'executed_ungraded':
                        normalize_literals(instruction, out)
                        results[rel, label] = not _s12_same(kept, _snapshot(out))
                    else:
                        results[rel, label] = None
                finally:
                    target.unlink()
                    os.symlink(task_dir / rel, target)
    except (OSError, HouseError):
        return 'failed'
    finally:
        _restore(out, kept)
        for key in MEASURED:
            status.pop(key, None)
        status.update(measured)
        status['runs'] = runs
        if fits is not None:
            health.items = fits
    return results


def s12_inert(code, out, task_dir, files, instruction, canaries, status, health, deadline, last_exec, reserve):
    """T1_S12_INERT's probe: ({identity: ('inert', flag)}, the edits flagged). Each of at most S12_INERT_KEYS input
    parameters (s12_param_edits) moved once in a rerun of the final program (_s12_probe), only while the time left
    exceeds S12_INERT_RUNS x the measured runtime x the keys probed + reserve (fewer keys while it does not). A key is
    flagged when its move changes no output beyond S12_INERT_RTOL on any number and another key of its file does (so
    the program reads the moved copy). s12_inert counts the flags; s12_inert_status (S12_INERT_STATUSES) says how it
    went."""
    edits = s12_param_edits(task_dir, files, instruction, canaries)
    if not edits:
        status['s12_inert_status'] = 'no_params'
        return {}, []
    runtime, n = max(last_exec, 1.0), len(edits)
    while n and deadline - CLOCK.now() <= S12_INERT_RUNS * runtime * n + reserve:
        n -= 1
    if not n:
        status['s12_inert_status'] = 'skipped_time'
        return {}, []
    results = _s12_probe(code, out, task_dir, edits[:n], instruction, canaries, status, health, deadline, last_exec,
                         reserve)
    if isinstance(results, str):
        status['s12_inert_status'] = results
        return {}, []
    read = {rel for (rel, _), changed in results.items() if changed}
    inert = [edit for edit in edits[:n] if results.get(edit[:2]) is False and edit[0] in read]
    unread = any(changed is False and rel not in read for (rel, _), changed in results.items())
    status.update(s12_inert=len(inert), s12_inert_status='flagged' if inert else 'unverified' if unread else 'none')
    flags = {}
    for rel, label, old, new, _ in inert:
        key = label.rsplit('.', 1)[-1]
        quote = next((line.strip() for line in instruction.splitlines() if _names(line, key)), '')
        told = S12_TOLD.format(quote=quote[:S12_QUOTE_CHARS]) if quote else ''
        flags['inert', rel, label] = ('inert', S12_INERT_FLAG.format(file=rel, key=label, old=old, new=new, told=told))
    return flags, inert


def _s12_ids(messages):
    """The D line IDs the notes block of the program prompt shows."""
    content = messages[0]['content'] if messages and isinstance(messages[0].get('content'), str) else ''
    head = S10_NOTES.split('{body}', 1)[0]
    return {int(n) for n in _S12_NOTE_ID.findall(content[content.find(head):])} if head in content else set()


def s12_audit_items(text, code, instruction, ids):
    """{identity: ('audit', flag)} of an audit reply's complete item lines (_S12_ITEM) whose code line is one of the
    program's (whitespace aside, at least S12_AUDIT_CODE_MIN characters) and whose reference is a D line the notes show
    (ids) or TASK words (_quoted); at most S12_AUDIT_ITEMS."""
    flat, items = ' '.join(code.split()), {}
    for line in text.split('\n'):
        match = _S12_ITEM.search(line)
        if not match or len(items) >= S12_AUDIT_ITEMS:
            continue
        ref, source, output, problem = (part.strip() for part in match.groups())
        source = source.strip('`').strip()
        if len(source) < S12_AUDIT_CODE_MIN or ' '.join(source.split()) not in flat:
            continue
        named = re.fullmatch(r'[\W_]*D(\d{1,2})\b[\W_]*', ref)
        if not (int(named.group(1)) in ids if named else _quoted(ref, instruction)):
            continue
        shown = '' if re.fullmatch(r'[\W_]*(?:n/?a|none)?[\W_]*', output, re.I) else f' gives {output[:200]};'
        items['audit', len(items)] = ('audit', S12_AUDIT_FLAG.format(ref=ref[:300], code=source[:300], output=shown,
                                                                      problem=problem[:400]))
    return items


def s12_audit(client, messages, code, out, instruction, deadline, status, pace, reserve, *, canaries=(),
              sleep=None):
    """T1_S12_AUDIT's one request (kind s12_audit) after the first clean run: the code, the outputs' digest and the
    notes' D lines in the conversation, asking where the code does not do what a D line or a TASK sentence says
    (S12_AUDIT). It reasons within s10_budget('s12_audit') under the token rule and is sent strict: only while its
    reply, its reasoning, reserve (a fix round) and S10_MARGIN fit, with S12_AUDIT_KEEP requests left and the breaker
    untripped; otherwise it is skipped. Returns s12_audit_items' {identity: ('audit', flag)}; s12_audit
    (S12_AUDIT_STATUSES) says how it went."""
    ids = _s12_ids(messages)
    if not ids:
        status['s12_audit'] = 'no_notes'
        return {}
    if getattr(client, 's10_think_off', False) is True:
        status['s12_audit'] = 'skipped_breaker'
        return {}
    ask = S12_AUDIT.format(digest=digest(out), items=S12_AUDIT_ITEMS, words=S12_AUDIT_WORDS)
    conversation = messages + [{'role': 'assistant', 'content': withhold_canaries(_fence(code), canaries)},
                               {'role': 'user', 'content': withhold_canaries(ask, canaries)}]
    budget = s10_budget('s12_audit', status=status)
    try:
        reply = _request(client, conversation, deadline, pace, 's12_audit', sleep,
                         need=_s10_cost(S10_TPW * S12_AUDIT_WORDS, _s10_rate(pace)), reserve=reserve,
                         keep=S12_AUDIT_KEEP, budget=budget, strict=True, think_margin=S10_MARGIN)
    except _Stop as stop:
        status['s12_audit'] = 'skipped_requests' if stop.reason in ('request_cap', 'prompt_cap') else 'skipped_time'
        return {}
    except HouseError:
        status['s12_audit'] = 'failed'
        return {}
    pace.measure(client, reply)
    answer = _s10_answer(reply, canaries)
    items = s12_audit_items(answer, code, instruction, ids)
    status['s12_audit'] = 'items' if items else 'none' if re.fullmatch(r'[\W_]*NONE[\W_]*', answer) else 'unparsed'
    return items


def s12_shape_changes(before, after, limit=3):
    """What a fix changed in the outputs' shape (two _snapshot trees): a file no longer written, a CSV column or JSON
    key path lost (shape_losses), or a CSV's row count or a JSON list's length changed, except from none (a header-only
    CSV or an empty list may fill). [] when every output keeps its shape."""
    changes = [f'{name[-80:]} is no longer written' for name in sorted(before) if name not in after]
    changes += shape_losses({k: ('file', None, v) for k, v in before.items()},
                            {k: ('file', None, v) for k, v in after.items()}, limit)
    for name, data in sorted(before.items()):
        new, suffix = after.get(name), name.rsplit('.', 1)[-1].lower() if '.' in name else ''
        if new is None:
            continue
        if suffix in ('csv', 'tsv'):
            old, now = (_table_shape(text, '\t' if suffix == 'tsv' else ',') for text in (data, new))
            if old and now and old[1] and now[1] > old[1]:
                changes.append(f'{name[-80:]} went from {old[1]} to {now[1]} rows')
        elif suffix == 'json':
            old, now = _json_shape(data), _json_shape(new)
            grown = sorted(place for place, length in (old[1] if old and now else {}).items()
                           if length and now[1].get(place, 0) > length)
            if grown:
                place = min(grown, key=len)
                changes.append(f'{name[-80:]}: the list at {_json_path(place)} went from {old[1][place]} to '
                               f'{now[1][place]} items')
    return changes[:limit]


def _s12_lint_key(text):
    """A code lint's identity across a patch: its text without the line number and with numbers left out."""
    return _flag_key(text.split(': ', 1)[-1])


def s12_flags(instruction, out, program, timed):
    """{identity: (kind, flag)} of S12's zero-request flags on the outputs now written and the program that wrote them:
    advisory (degenerate, with an output file left empty, which degenerate_flags does not read; literal; marked as
    such), selfval and lint, in that order. Never raises: a check that fails finds nothing."""
    found = {}
    literal, degenerate = advisory_flags(instruction, out)
    try:
        degenerate = [f'{name}: the file is empty' for name in sorted(_output_files(out))
                      if name.rsplit('.', 1)[-1].lower() in S12_EMPTY and not (out / name).read_bytes().strip()
                      ] + degenerate
    except OSError:
        pass
    for kind, items in (('degenerate', degenerate), ('literal', literal)):
        for text in items:
            found.setdefault((kind, _flag_key(text)), (kind, '(advisory) ' + text))
    try:
        for ident, text in selfval_flags(instruction, out):
            found.setdefault(ident, ('selfval', text))
    except Exception:
        pass
    try:
        for text in s12_lint(program, instruction, timed):
            found.setdefault(('lint', _s12_lint_key(text)), ('lint', text))
    except Exception:
        pass
    return found


def _s12_numeric_context(out, listed, canaries=()):
    """Bounded observed JSON scalars, only for listed self-validation flags.

    This is evidence context, never a flag, threshold or expected value. Lists,
    strings, metadata, links and oversized/deep outputs are deliberately omitted.
    At most six files, 16 KiB each/64 KiB total, 40 scalar leaves and 2,000 characters;
    file paths have at most three parts and object keys at most four. Discovery
    examines at most 128 entries; named flag files are checked directly first.
    """
    wanted = {ident[1] for ident, (kind, _) in listed
              if kind == 'selfval' and isinstance(ident, tuple) and len(ident) > 1
              and isinstance(ident[1], str)}
    if not wanted or out.is_symlink():
        return ''
    paths, omitted, scanned = [], False, 0
    forbidden = {'checks', 'reference', 'grading', 'verifier', 'metadata', '__pycache__'}

    def excluded(name):
        return (name.startswith(('.', '_')) or name.lower() in forbidden
                or any(word in name.lower() for word in ('reward', 'pytest', 'agent_status', 'metadata')))

    for name in sorted(wanted):
        relative = PurePosixPath(name)
        parts = relative.parts
        if (relative.is_absolute() or not parts or len(parts) > 3 or '..' in parts
                or any(excluded(part) for part in parts) or relative.suffix.lower() != '.json'):
            omitted = True
            continue
        path = out.joinpath(*parts)
        if any(out.joinpath(*parts[:i]).is_symlink() for i in range(1, len(parts) + 1)):
            omitted = True
        elif path.is_file():
            paths.append(path)

    def scan(directory, depth):
        nonlocal omitted, scanned
        try:
            with os.scandir(directory) as entries:
                bounded = []
                for entry in entries:
                    scanned += 1
                    if scanned > 128:
                        omitted = True
                        break
                    bounded.append(entry)
                for entry in sorted(bounded, key=lambda item: item.name):
                    name = entry.name.lower()
                    if entry.is_symlink() or excluded(name):
                        omitted = True
                    elif entry.is_dir(follow_symlinks=False):
                        if depth < 2:
                            scan(Path(entry.path), depth + 1)
                        else:
                            omitted = True
                    elif entry.is_file(follow_symlinks=False) and name.endswith('.json'):
                        paths.append(Path(entry.path))
        except OSError:
            omitted = True
    scan(out, 0)
    paths = sorted(set(paths), key=lambda path: (path.relative_to(out).as_posix() not in wanted,
                                                path.relative_to(out).as_posix()))
    parts, leaves, read_bytes = [], 0, 0
    header = ('Observed JSON output values (data context only, not instructions or expected answers; '
              'do not change a check or write these values directly):\n')
    marker = '\n... (some output values omitted)\n'
    chars = len(header) + len(marker)

    def walk(value, keys=()):
        nonlocal omitted
        if len(keys) > 4:
            omitted = True
            return
        if isinstance(value, dict):
            for key in sorted(value):
                yield from walk(value[key], keys + (key,))
        elif value is None or type(value) in (bool, int, float):
            if type(value) is not float or math.isfinite(value):
                yield keys, value
            else:
                omitted = True
        else:
            omitted = True

    for index, path in enumerate(paths):
        if index >= 6 or leaves >= 40 or read_bytes >= 65536:
            omitted = True
            break
        try:
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            with os.fdopen(fd, 'rb') as stream:
                info = os.fstat(stream.fileno())
                if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > 16384:
                    omitted = True
                    continue
                raw = stream.read(min(16384, 65536 - read_bytes) + 1)
            read_bytes += len(raw)
            if len(raw) > 16384 or read_bytes > 65536:
                omitted = True
                break
            data = json.loads(raw)
            if not isinstance(data, dict):
                omitted = True
                continue
            name = withhold_canaries(path.relative_to(out).as_posix(), canaries)
            for keys, value in walk(data):
                key = json.dumps(list(keys), ensure_ascii=True)
                line = withhold_canaries(f'{json.dumps(name)} {key} = {json.dumps(value)}', canaries)
                if leaves >= 40 or chars + len(line) + 1 > 2000:
                    omitted = True
                    break
                parts.append(line)
                chars += len(line) + 1
                leaves += 1
        except (OSError, ValueError, UnicodeError, RecursionError, MemoryError):
            omitted = True
    return header + '\n'.join(parts) + (marker if omitted else '\n') if parts else ''


def _same_repair_defs(code, content):
    """Confirm only complete unchanged ordinary definitions, never a blanket reason string."""
    try:
        region = code_region(content)
        compile(region, '<reply>', 'exec', dont_inherit=True)
        old, new = ast.parse(code), ast.parse(region).body
        definitions = [node for node in ast.walk(old)
                       if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
        return bool(new) and all(
            isinstance(node, ast.FunctionDef) and not node.decorator_list
            and sum(item.name == node.name for item in definitions) == 1
            and any(isinstance(item, ast.FunctionDef) and item.name == node.name
                    and not item.decorator_list and ast.dump(item) == ast.dump(node) for item in old.body)
            for node in new) and len({node.name for node in new}) == len(new)
    except (SyntaxError, ValueError, RecursionError, MemoryError, OverflowError):
        return False


def s12_post(client, messages, code, out, instruction, deadline, status, pace, last_exec, *, canaries=(),
             replay=_NO_REPLAY, task_dir=None, path_index=None, sleep=None, health=None, files=()):
    """T1_S12_POST after the last clean run: flags with zero requests (s12_flags; under T1_S12_INERT s12_inert, under
    T1_S12_AUDIT s12_audit), then up to S12_ROUNDS fix rounds (kind s12_fix: _patch, with the first patch's reasoning
    budget), each listing at most S12_ITEMS flags and rerunning the patched program. A fix is kept only when the rerun
    exits 0, writes every output file with its keys, columns and row counts (s12_shape_changes), adds no flag (the
    review's deterministic flags, s12_flags' identities), and clears every self-validation and lint flag it listed
    (S12_HARD); a round that listed none of those needs one of its items to have moved: an advisory flag gone, an
    inputs flag whose move now shows (a probe of the fixed program), or, for audit items, changed outputs. Otherwise
    the program, its outputs, measured fields and fit reports go back, and the next round lists the same flags with why
    the change was not kept; after a kept fix the next round lists the flags it did not list (an advisory flag a kept
    fix left stands). Under T1_S12_MECH too, a fix request is a full reply resent after a transient error as here:
    the original s12_fix retry policy is unchanged. Only a new last length reply with no usable patch may enter
    one light sequence (_s12_light_patch), sized to S12_PATCH_CAP and without reasoning. A failed continuation
    cannot reuse the first reply's length. The candidate passes the same run and rollback checks below; counters
    s12_post_light/requests/kept distinguish sequences sent, request attempts and accepted candidates.
    Listed self-validation items also carry bounded observed JSON scalar context, which never changes flags or
    acceptance. Only that existing light path's valid stop reply of unchanged ordinary definitions may receive
    one further unreasoned, 1,500-token no-change feedback attempt; the same deadline, reserve and request cap
    apply. s12_post_noop_flags/kind/retry_requests describe that conditional branch. No continuation or transport
    resend is allowed for its one attempt. Crash repair and non-selfval light paths keep their original policy.
    Returns (program, request fields of the last kept fix or None, its run's seconds)."""
    health = health if health is not None else FitWarnings()
    timed, contract, lint = s12_timed(files), output_contract(instruction), s11_lint_on()
    normalize = (lambda text: prepared(text, task_dir, path_index)) if task_dir else None

    def base(program):  # the deterministic flags the flags-only round reads: a kept fix may not add one
        found = contract_flags(contract, out) + bound_flags(instruction, out) + sanity_flags(out)
        if lint:
            output, static = s11_flags(instruction, out, program, found)
            found = found + output + static
        return found
    status.update(s12_rounds=0, s12_kept=0, s12_reverted=0)
    own = s12_flags(instruction, out, code, timed)
    own.update(runtime.flags(status))
    targeted, probes = dict(own), []
    fix_reserve = pace.patch() + s10_budget('patch') / pace.rate + last_exec + 10 + S10_MARGIN
    coerce = s13_coerce_on()
    observations = {}
    if coerce:
        observations = s13_coerce_audit(code, out, canaries, status, health, deadline, last_exec, fix_reserve, task_dir)
    watch = coerce and status.get('s13_coerce_audit') in ('ran', 'no_call')
    if s12_inert_on() and task_dir is not None:
        inert, probes = s12_inert(code, out, task_dir, files, instruction, canaries, status, health, deadline,
                                  last_exec, fix_reserve)
        targeted.update(inert)
    if s12_audit_on():
        reprobe = len(probes) * (2 * last_exec + 10)
        targeted.update(s12_audit(client, messages, code, out, instruction, deadline, status, pace,
                                  fix_reserve + reprobe, canaries=canaries, sleep=sleep))
    kinds = Counter(kind for kind, _ in targeted.values())
    status.update({f's12_flags_{kind}': kinds.get(kind, 0) for kind in S12_KINDS})
    if coerce:
        status['s13_coerce_advisory'] = len(observations)
    if not targeted:
        status['s12_stop'] = 'clean'
        return code, None, last_exec
    kept_fields, why, light_tried = None, '', False
    for _ in range(S12_ROUNDS):
        listed = sorted(targeted.items(), key=lambda item: item[1][0] == 'audit')[:S12_ITEMS]
        kept = _snapshot(out)
        if kept is None:
            status['s12_stop'] = 'too_large'
            break
        items = '\n'.join('- ' + text[:S12_FLAG_CHARS] for _, (_, text) in listed)
        numeric = _s12_numeric_context(out, listed, canaries)
        if numeric:
            _count(status, 's12_numeric_context_chars', len(numeric))
        ask = S12_FIX.format(items=items, note=numeric + (S12_AGAIN.format(why=why) if why else ''))
        if observations:
            # Context only: loss alone neither requests a repair nor establishes
            # that one is correct. Acceptance still needs the independent flags.
            context = '\n'.join('- ' + text[:S12_FLAG_CHARS]
                                for _, text in list(observations.values())[:S12_ITEMS])
            ask = ('Observation context only, not a request to change parsing policy:\n' + context + '\n\n' + ask)
        if s10_noted(messages):
            ask = ask.removesuffix(PATCH_FORMAT) + S12_KEEP + ' ' + PATCH_FORMAT
        conversation = messages + [{'role': 'assistant', 'content': withhold_canaries(_fence(code), canaries)},
                                   {'role': 'user', 'content': withhold_canaries(ask, canaries)}]
        recheck = [edit for edit in probes if (('inert',) + edit[:2]) in dict(listed)]
        reserve = last_exec + 10 + len(recheck) * (2 * last_exec + 10)
        records_before = len(client.diagnostics.get('requests', []))
        round_light, outcome = False, {}
        try:
            fixed = _patch(client, conversation, code, deadline, status, reserve, canaries, pace, sleep, None,
                           's12_fix', normalize, outcome=outcome)[0]
        except _Stop as stop:
            status['s12_stop'] = 'skipped_requests' if stop.reason in ('request_cap', 'prompt_cap') else 'skipped_time'
            break
        except HouseError:
            _count(status, 's12_rounds')  # a fix request went out
            status['s12_stop'] = 'failed'
            break
        _count(status, 's12_rounds')
        records = client.diagnostics.get('requests', [])
        # House may retain length metadata for a rejected reply. Require a valid returned reply too.
        # info={} is separate: it would also alter the original timeout retry policy.
        cut = (outcome.get('reply_returned') is True and len(records) > records_before
               and records[-1].get('finish_reason') == 'length')
        if (fixed is None and not light_tried and s12_mech_on() and cut
                and _s12_unusable(code, fixed, '', {'cut': cut}, None) == 'cut'):
            light_tried = True
            attempts = client.attempts
            no_change = {} if any(kind == 'selfval' for _, (kind, _) in listed) else None
            light_ask = S12_POST_LIGHT.removesuffix(PATCH_FORMAT) + '\n' + numeric + PATCH_FORMAT if numeric \
                else S12_POST_LIGHT
            light, _ = _s12_light_patch(client, conversation, code, 'cut', '', '', deadline, status, reserve,
                                        canaries, pace, sleep, None, normalize, ask=light_ask, no_change=no_change)
            if client.attempts > attempts:
                _count(status, 's12_post_light')
                _count(status, 's12_post_light_requests', client.attempts - attempts)
            if light is not None:
                fixed, round_light = light[0], True
            elif no_change:
                _count(status, 's12_post_noop_flags')
                status['s12_post_noop_kind'] = 'same_definition'
                feedback = ('Your completed reply did not change the program: ' + no_change['why'] + '. '
                            'The listed self-validation issues remain. Change their cause, keep every output '
                            'and check, and do not repeat unchanged definitions. No explanation.\n'
                            + numeric + PATCH_FORMAT)
                attempts = client.attempts
                fixed = _retry_patch(client, no_change['conversation'], code, no_change['content'], feedback,
                                     deadline, status, reserve, canaries, pace, sleep, None, 'patch_retry', normalize,
                                     cap=S12_PATCH_CAP, one_request=True)
                if client.attempts > attempts:
                    _count(status, 's12_post_noop_retry_requests', client.attempts - attempts)
                round_light = fixed is not None
        if fixed is None:
            status['s12_stop'] = 'no_patch'
            break
        fields = {k: client.diagnostics.get(k) for k in ('http_status', 'finish_reason')}
        fixed = normalize(fixed) if normalize else fixed
        measured = {k: status[k] for k in MEASURED if k in status}
        kept_health, before = list(health.items), {_flag_key(flag) for flag in base(code)}
        rewritten, verdict, ran = {}, ('error', ''), last_exec
        try:
            status['program_sha256'] = hashlib.sha256(fixed.encode()).hexdigest()
            replay.put('program.py', fixed)
            started = CLOCK.now()
            lost = [] if watch and s13_coerce_called(fixed) else None
            _run_program(fixed, out, deadline, canaries, status, clear=True, replay=replay, input_root=task_dir,
                         health=health, **({} if lost is None else {'coerce': lost}))
            if lost is not None:
                _count(status, 's13_coerce_runs')
            ran = CLOCK.now() - started
            if status['status'] != 'executed_ungraded':
                verdict = ('failed', '')
            elif not set(kept) <= _output_files(out):
                verdict = ('missing', '')
            else:
                rewritten = normalize_literals(instruction, out)
                now, after = _snapshot(out), s12_flags(instruction, out, fixed, timed)
                after.update(runtime.flags(status))
                runtime_problem = runtime.regression(measured, status)
                changes = s12_shape_changes(kept, now) if now is not None else ['the outputs grew too large']
                added = [flag for flag in base(fixed) if _flag_key(flag) not in before]
                added += [text for ident, (_, text) in after.items() if ident not in own]
                hard = [(ident, text) for ident, (kind, text) in listed if kind in S12_HARD]
                soft = [(ident, text) for ident, (kind, text) in listed if kind in ('degenerate', 'literal')]
                left = [text for ident, text in hard if ident in after]
                verdict = (('added', runtime_problem) if runtime_problem else
                           ('shape', changes[0]) if changes else ('added', added[0]) if added else
                           ('left', left[0]) if left else None)
                if verdict is None and not hard:  # advisory, inputs and audit items alone: one must have moved
                    moved = any(ident not in after for ident, _ in soft) or (
                        any(kind == 'audit' for _, (kind, _) in listed) and now != kept)
                    if not moved and recheck:  # an inputs flag clears when its move now shows in the outputs
                        probed = _s12_probe(fixed, out, task_dir, recheck, instruction, canaries, status, health,
                                            deadline, ran, 0)
                        moved = isinstance(probed, dict) and any(probed.get(edit[:2]) for edit in recheck)
                    stays = [text for _, text in soft] + [targeted[('inert',) + edit[:2]][1] for edit in recheck]
                    verdict = None if moved else ('left', stays[0]) if stays else ('unchanged', '')
        except Exception:  # a setup failure or no time left: the earlier program and outputs stand
            verdict = ('error', '')
        if verdict is None:
            _reset_counts(status, rewritten)
            _count(status, 's12_kept')
            if round_light:
                _count(status, 's12_post_light_kept')
            if lost is not None:
                observations = s13_coerce_items(lost, canaries)
                status['s13_coerce_advisory'] = len(observations)
            code, kept_fields, last_exec, own = fixed, fields, ran, after
            done = {ident for ident, _ in listed}  # an advisory flag a kept fix left is what the TASK asks
            targeted = {**{ident: item for ident, item in own.items() if ident not in done},
                        **{ident: item for ident, item in targeted.items()
                           if ident not in done and item[0] in ('inert', 'audit')}}
            status['s12_stop'], why = 'cleared' if not targeted else 'rounds', ''
            if not targeted:
                break
            continue
        _restore(out, kept)
        health.items = kept_health
        for key in MEASURED:
            status.pop(key, None)
        status.update(measured)
        replay.put('program.py', code)  # the replay's newest program is the one whose outputs stand
        _count(status, 's12_reverted')
        status['s12_stop'] = 'reverted'
        why = S12_WHY[verdict[0]].format(detail=verdict[1][:S12_FLAG_CHARS])
    return code, kept_fields, last_exec


# Optional Stack13 mechanisms, adapted to the Stack14 runtime contract.
S13_RES_NAMES = frozenset({'n_space', 'n_time', 'n_steps', 'steps', 'num_steps', 'n_grid', 'grid_size', 'n_nodes',
                           'n_s', 'n_t'})  # compared in lower case
# A step count may be a calendar count. Time axes are never automatically refined.
S13_TIME_NAMES = frozenset({'n_time', 'n_steps', 'steps', 'num_steps', 'n_t'})
S13_RES_MIN = 8  # conservative floor for the known solvers' spatial argument
S13_SPACE_APIS = frozenset({'qfkit.fd_solve', 'qfkit.pde.fd_solve',
                            'qfkit.fd_cn_price', 'qfkit.numerics.fd_cn_price'})
# A program that draws random numbers is skipped whole: a changed step count changes its random stream.
S13_RNG = re.compile(r'np\.random|numpy\.random|default_rng|RandomState|\brandom\.|\bseed\s*=|n_paths\s*=|'
                     r'random_state|\.rvs\(')
# A number the TASK states (1,000 and 1_000 included), a sentence's last one too ('Use n_steps = 200.'); a decimal
# (0.25, 3.5) or a version (v2.0) is none.
S13_NUMBER = re.compile(r'(?<![\w.])\d[\d,_]*(?!\w|\.\d)')
S13_X2_GATE = (5.0, 10.0)  # x2 runs only while 5 x the runtime + 10 s fits beside S10_MARGIN
S13_X2_CAP = (6.0, 10.0)  # and stops at 6 x the runtime + 10 s
S13_X4_GATE = (1.5, 10.0)  # x4 runs, and is capped, at 1.5 x its time t2 * t2 / runtime + 10 s
S13_X4_FLOOR = 0.5  # seconds: the least runtime the x4 estimate divides by
S13_MOVED = 1e-9  # a column moved from x1 to x2 beyond this x max(1, median |x1|)
S13_SIDE = 1e-6  # a column within this x max(1, median |x1|) of one side of 0 must stay on that side at x4
S13_REFINE_STATUSES = ('none', 'rng', 'skipped_time', 'too_large', 'x2_failed', 'x4_failed', 'shape', 'nonfinite',
                       'unmoved', 'not_contracting', 'sign', 'adopted', 'failed', 'runtime_regression',
                       'unproven_grid', 'task_resolution', 'time_axis_unverified')
# T1_S13_COERCE observes loss without changing pandas results. Every item is advisory:
# parsing in another format does not authorize changing the TASK's date or missing-value policy.
S13_COERCE_FNS = ('to_datetime', 'to_numeric')
S13_COERCE_NA = ('', 'na', 'n/a', 'nan', 'null', 'none', 'nat', '-', '#n/a', '<na>')  # missing, not lost
S13_COERCE_EXAMPLES, S13_COERCE_CHARS = 3, 40  # lost values shown per call site, and characters of each
S13_COERCE_SITES = 64  # call sites the prelude follows; the rest go unrecorded
S13_COERCE_DISTINCT = 5000  # distinct lost texts tried with format='mixed' per call
S13_COERCE_BYTES = 64 * 1024  # bytes of the prelude's report read back
S13_COERCE_NOTE = ("(advisory) line {line}: pandas.{fn}(..., errors='coerce') turned {lost} of {nonempty} "
                   "non-empty values into NaN/NaT (original examples: {examples}). This observation does not establish "
                   "that the values satisfy the TASK's date or numeric policy. Preserve strict formats and missing "
                   "value rules. If the TASK is ambiguous, do not reinterpret or recover them automatically.")
S13_COERCE_AUDITS = ('no_call', 'ran', 'skipped_time', 'too_large', 'failed')
# The prelude (sitecustomize, after worker_site's parts, only on a run that passes coerce): it wraps pandas.to_datetime
# and pandas.to_numeric, returns their results unchanged and never raises. Each call with errors='coerce' adds to its
# call site in the program (the line of the program's own frame) the non-empty inputs (not an NA token) and those of
# them that came back NaN or NaT, with up to S13_COERCE_EXAMPLES of these and, for to_datetime, how many of them parse
# with format='mixed'. The sites with a loss go to $T1_COERCE_REPORT as JSON lines: at a site's first loss, and at exit.
COERCE_SITE = """try:  # T1_S13_COERCE: what pandas' errors='coerce' turned into NaN or NaT, read by the controller only
    import os as _t1c_os
    if _t1c_os.environ.get('T1_COERCE_REPORT'):
        import atexit as _t1c_atexit
        import functools as _t1c_functools
        import json as _t1c_json
        import sys as _t1c_sys
        import warnings as _t1c_warnings
        import pandas as _t1c_pd

        _t1c_na, _t1c_sites = frozenset(%(na)r), {}

        _t1c_pid = _t1c_os.getpid()

        def _t1c_save():
            try:
                pool = _t1c_sys.modules.get('multiprocessing')
                if _t1c_os.getpid() != _t1c_pid or (pool is not None and pool.parent_process() is not None):
                    return  # a forked or spawned worker: the program's own process writes the report
                with open(_t1c_os.environ['T1_COERCE_REPORT'], 'w') as stream:
                    for (name, line), site in list(_t1c_sites.items()):
                        if site['lost']:
                            stream.write(_t1c_json.dumps(dict(site, fn=name, line=line)) + '\\n')
            except Exception:
                pass

        def _t1c_site(name):
            program, frame = _t1c_os.environ.get('T1_PROGRAM'), _t1c_sys._getframe(1)
            while frame is not None and frame.f_code.co_filename != program:
                frame = frame.f_back
            key = (name, frame.f_lineno if frame is not None else 0)
            if key not in _t1c_sites and len(_t1c_sites) < %(sites)d:
                _t1c_sites[key] = {'lost': 0, 'nonempty': 0, 'recoverable': 0, 'ex': []}
            return _t1c_sites.get(key)

        def _t1c_count(name, original, given, got):
            if _t1c_pd.api.types.is_scalar(given):
                real = not _t1c_pd.isna(given) and str(given).strip().lower() not in _t1c_na
                lost, nonempty = ([given] if real and _t1c_pd.isna(got) else []), int(real)
            else:
                given = given if isinstance(given, _t1c_pd.Series) else _t1c_pd.Series(given)
                got = got if isinstance(got, _t1c_pd.Series) else _t1c_pd.Series(got, index=given.index)
                real = given.notna() & ~given.astype(str).str.strip().str.lower().isin(_t1c_na)
                missing = real & got.isna()
                lost, nonempty = (list(given[missing]) if missing.any() else []), int(real.sum())
            site = _t1c_site(name)
            if site is None:
                return
            site['nonempty'] += nonempty
            if not lost:
                return
            site['lost'] += len(lost)
            for shown in dict.fromkeys(str(value)[:%(chars)d] for value in lost):
                if len(site['ex']) >= %(examples)d:
                    break
                if shown not in site['ex']:
                    site['ex'].append(shown)
            if name == 'to_datetime':
                try:
                    counts = _t1c_pd.Series([str(value) for value in lost], dtype=object).value_counts()
                    counts = counts.iloc[:%(distinct)d]
                    with _t1c_warnings.catch_warnings():
                        _t1c_warnings.simplefilter('ignore')
                        parsed = original(_t1c_pd.Series(list(counts.index), dtype=object), format='mixed',
                                          errors='coerce')
                    site['recoverable'] += int(counts[parsed.notna().to_numpy()].sum())
                except Exception:
                    pass
            if site['lost'] == len(lost):  # the site's first loss: on file now, whether or not the program exits
                _t1c_save()

        def _t1c_audit(name, original):
            @_t1c_functools.wraps(original)
            def audited(arg, *args, **kwargs):
                result = original(arg, *args, **kwargs)
                try:
                    if kwargs.get('errors', args[0] if args else None) == 'coerce':
                        _t1c_count(name, original, arg, result)
                except Exception:
                    pass
                return result
            # pickle finds a function by module and name: it must find the wrapper, as pandas.<name> now is
            audited.__module__, audited.__qualname__ = 'pandas', name
            return audited

        _t1c_pd.to_datetime = _t1c_audit('to_datetime', _t1c_pd.to_datetime)
        _t1c_pd.to_numeric = _t1c_audit('to_numeric', _t1c_pd.to_numeric)
        _t1c_atexit.register(_t1c_save)
except Exception:
    pass
""" % {'na': sorted(S13_COERCE_NA), 'sites': S13_COERCE_SITES, 'chars': S13_COERCE_CHARS,
       'examples': S13_COERCE_EXAMPLES, 'distinct': S13_COERCE_DISTINCT}
# Digest of the selector policy, thresholds and texts (not task-solving evidence).
S13_PROMPT_SHA = hashlib.sha256(json.dumps([
    'stack14-spatial-only-advisory-v1', sorted(S13_SPACE_APIS), sorted(S13_RES_NAMES), sorted(S13_TIME_NAMES),
    S13_RES_MIN, S13_RNG.pattern, S13_NUMBER.pattern, S13_X2_GATE,
    S13_X2_CAP, S13_X4_GATE, S13_X4_FLOOR, S13_MOVED, S13_SIDE, S13_COERCE_FNS, S13_COERCE_NA, S13_COERCE_EXAMPLES,
    S13_COERCE_CHARS, S13_COERCE_SITES, S13_COERCE_DISTINCT, S13_COERCE_BYTES, S13_COERCE_NOTE,
    COERCE_SITE],
    sort_keys=True).encode()).hexdigest()


def s13_refine_on() -> bool:
    """T1_S13_REFINE=1, only under T1_S12_POST: the resolution refinement after s12_post (s13_refine)."""
    return os.environ.get('T1_S13_REFINE') == '1' and s12_post_on() and not minimal_on()


def _s13_stated(instruction):
    """The integers the TASK text states: each number, and each comma-separated part of one."""
    stated = set()
    for text in S13_NUMBER.findall(instruction):
        for part in [text] + (text.split(',') if ',' in text else []):
            digits = part.replace(',', '').replace('_', '')
            if digits.isdigit():
                stated.add(int(digits))
    return stated


def _s13_binds(tree):
    """[(name, value node)] of every plain assignment to a name: `x = v`, `x: int = v`, and `a, b = v, w`
    element by element."""
    binds = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    binds.append((target.id, node.value))
                elif (isinstance(target, (ast.Tuple, ast.List)) and isinstance(node.value, (ast.Tuple, ast.List))
                      and len(target.elts) == len(node.value.elts)):
                    binds += [(name.id, value) for name, value in zip(target.elts, node.value.elts)
                              if isinstance(name, ast.Name)]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
            binds.append((node.target.id, node.value))
    return binds


def _s13_resolution_reason(code, instruction):
    # A conservative veto, never an inference of what resolution the task intended.
    words = set(re.findall(r'[a-z_]+', instruction.lower()))
    if words & (S13_RES_NAMES | {'grid', 'grids', 'mesh', 'resolution', 'nodes', 'discretization', 'discretisation'}):
        return 'task_resolution'
    if set(re.findall(r'[a-z_]+', code.lower())) & S13_TIME_NAMES:
        return 'time_axis_unverified'
    return 'unproven_grid'


def s13_literals(code, instruction):
    """Select syntactically recognized n_space arguments of qfkit PDE solvers.

    No arbitrary name-based grid guessing; no time/calendar axis, shared variable,
    shadowed/escaped import, dynamic namespace, task-mentioned resolution, or
    random program is eligible. This is conservative syntax screening, not a
    general proof of arbitrary Python behavior.
    """
    if S13_RNG.search(code) or _s13_resolution_reason(code, instruction) == 'task_resolution':
        return []
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return []
    bindings = {}
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in ('qfkit', 'qfkit.pde', 'qfkit.numerics'):
                    bindings[alias.asname or alias.name.split('.')[0]] = alias.name if alias.asname else 'qfkit'
        elif isinstance(node, ast.ImportFrom) and node.module in ('qfkit', 'qfkit.pde', 'qfkit.numerics'):
            for alias in node.names:
                bindings[alias.asname or alias.name] = node.module + '.' + alias.name
    # Count all writes, including loops, augmented/destructured assignments and
    # imports. A plain single assignment alone is not proof of a stable binding.
    binds = _s13_binds(tree)
    stores, imported = Counter(), Counter()
    shadowed = set()
    dynamic = {'exec', 'eval', 'globals', 'locals', 'vars', 'getattr', 'setattr', 'delattr',
               '__import__', '__builtins__'}
    for node in ast.walk(tree):
        if (isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id in dynamic
                or isinstance(node, ast.Attribute) and node.attr in dynamic | {
                    '__dict__', '__globals__', '__builtins__', 'f_globals', 'f_locals'}
                or isinstance(node, ast.ImportFrom) and node.module == 'builtins'
                and any(alias.name in dynamic for alias in node.names)):
            return []  # Aliased namespace access also hides reads/writes from the name scan below.
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            stores[node.id] += 1
            shadowed.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            shadowed.add(node.name)
        elif isinstance(node, ast.arg):
            shadowed.add(node.arg)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                if alias.name == '*':
                    return []
                imported[alias.asname or alias.name.split('.')[0]] += 1
        elif isinstance(node, ast.ExceptHandler) and node.name:
            shadowed.add(node.name)
        elif isinstance(node, ast.Attribute) and isinstance(node.ctx, (ast.Store, ast.Del)):
            root = node.value
            while isinstance(root, ast.Attribute):
                root = root.value
            if isinstance(root, ast.Name):
                shadowed.add(root.id)
    bindings = {name: value for name, value in bindings.items()
                if name not in shadowed and imported[name] == 1}
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id in bindings):
            continue
        used, parent = node, parents.get(node)
        if bindings[node.id] in ('qfkit', 'qfkit.pde', 'qfkit.numerics'):
            while isinstance(parent, ast.Attribute) and parent.value is used and isinstance(parent.ctx, ast.Load):
                used, parent = parent, parents.get(parent)
            if used is node:
                return []  # Passing or assigning the module can mutate a solver through an alias.
        if not isinstance(parent, ast.Call) or parent.func is not used:
            return []  # Imported functions/attributes may not escape either; no alias analysis attempted.
    candidates = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        path = (bindings.get(fn.id, '') if isinstance(fn, ast.Name) else
                bindings.get(fn.value.id, '') + '.' + fn.attr
                if isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name) else '')
        if path in S13_SPACE_APIS:
            candidates += [item.value for item in node.keywords if item.arg == 'n_space']
    assigned = Counter(name for name, _ in binds)
    shared_values = Counter(id(value) for _, value in binds)
    parameters = {node.arg for node in ast.walk(tree) if isinstance(node, ast.arg)}
    single = {name: value for name, value in binds
              if assigned[name] == stores[name] == shared_values[id(value)] == 1
              and not imported[name] and name not in parameters}
    stated, found = _s13_stated(instruction), {}
    for value in candidates:
        name = 'n_space'
        if isinstance(value, ast.Name):
            name = value.id
            reads = [node for node in ast.walk(tree) if isinstance(node, ast.Name)
                     and isinstance(node.ctx, ast.Load) and node.id == name]
            if any(node not in candidates for node in reads):
                continue
            value = single.get(name)
        if (isinstance(value, ast.Constant) and type(value.value) is int and value.value >= S13_RES_MIN
                and value.value not in stated and value.lineno == value.end_lineno):
            found[value.lineno, value.col_offset] = (value.lineno, value.col_offset, value.end_col_offset,
                                                     name, value.value)
    return sorted(found.values())


def s13_scaled(code, literals, factor):
    """The program with each literal (s13_literals) times factor, spliced at its byte offsets as quiet_self_check
    splices; None when the result does not parse."""
    lines = code.split('\n')
    for row, start, end, _, value in sorted(literals, reverse=True):
        raw = lines[row - 1].encode()
        lines[row - 1] = (raw[:start] + str(value * factor).encode() + raw[end:]).decode()
    scaled = '\n'.join(lines)
    try:
        ast.parse(scaled)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return None
    return scaled


def _s13_leaves(value, path, columns, depth=0):
    """A JSON value's leaves by path (a list level is None, so a list of records gives one column per key)."""
    if depth > 64:
        raise ValueError('too deep')
    if isinstance(value, dict):
        for key, item in value.items():
            _s13_leaves(item, path + (str(key),), columns, depth + 1)
    elif isinstance(value, list):
        for item in value:
            _s13_leaves(item, path + (None,), columns, depth + 1)
    else:
        columns.setdefault(path, []).append(value)


def _s13_parquet(data):
    import pyarrow as pa
    import pyarrow.parquet as pq
    table = pq.read_table(pa.BufferReader(data))
    columns = {}
    for index, name in enumerate(table.column_names):
        column = table.column(index)
        numeric = pa.types.is_integer(column.type) or pa.types.is_floating(column.type) or pa.types.is_decimal(
            column.type)
        columns[index, name] = [float(v) if numeric and v is not None else v for v in column.to_pylist()]
    return columns


def _s13_columns(snapshot):
    """{file: {column: [cell]}, or None for a file read only for its existence}: CSV and TSV columns by position and
    header (_s12_cells: a cell is a number where it reads as one), JSON leaves by path, parquet columns (pyarrow);
    any other file, or one that does not read, is None."""
    tables = {}
    for name, data in snapshot.items():
        suffix, columns = name.rsplit('.', 1)[-1].lower() if '.' in name else '', None
        try:
            if suffix in ('csv', 'tsv'):
                rows = _s12_cells(data, '\t' if suffix == 'tsv' else ',')
                head, body = (rows[0], rows[1:]) if rows else ([], [])
                width = max([len(head)] + [len(row) for row in body])
                columns = {(i, str(head[i]) if i < len(head) else ''): [row[i] if i < len(row) else '' for row in body]
                           for i in range(width)}
                columns[-1, ''] = [str(len(row)) for row in body]  # each row's width, as text: ragged is a shape
            elif suffix == 'json':
                columns = {}
                _s13_leaves(json.loads(data.decode('utf-8-sig')), (), columns)
            elif suffix in ('parquet', 'pqt'):
                columns = _s13_parquet(data)
        except Exception:
            columns = None
        tables[name] = columns
    return tables


def _s13_numbers(cells):
    """A column's cells as floats (a blank cell as NaN), or None for a text column: one whose cells are not all numbers
    or blanks, or hold no number."""
    values, numbers = [], 0
    for cell in cells:
        if cell is None or cell == '':
            values.append(math.nan)
        elif isinstance(cell, (int, float)) and not isinstance(cell, bool):
            try:
                values.append(float(cell))
            except OverflowError:
                return None
            numbers += 1
        else:
            return None
    return values if numbers else None


def _s13_same_text(a, b):
    """Whether two text columns hold the same cells (a NaN cell equal to a NaN cell)."""
    return len(a) == len(b) and all(x == y or (x != x and y != y) for x, y in zip(a, b))


def _s13_move(a, b):
    """The largest |b - a| over the cells finite in both (0 when none is)."""
    return max((abs(y - x) for x, y in zip(a, b) if math.isfinite(x) and math.isfinite(y)), default=0.0)


def s13_adopt(x1, x2, x4=None):
    """s13_refine's verdict on three output snapshots (_snapshot): 'adopted', or the first rule that fails: 'shape'
    (another file set, column set, row count, column kind or text cell), 'nonfinite' (a finite cell turned NaN or
    inf), 'unmoved' (no number moved from x1 to x2 beyond S13_MOVED x max(1, median |x1|)), 'not_contracting' (a
    number column moved from x2 to x4 as much as or more than from x1 to x2, unless both are 0) or 'sign' (a column
    whose x1 cells lie on one side of 0, within S13_SIDE x max(1, median |x1|), has an x4 cell beyond it). With x4
    None, what x1 and x2 alone decide: None when they pass."""
    one, two = _s13_columns(x1), _s13_columns(x2)
    four = _s13_columns(x4) if x4 is not None else None
    others = [two] if four is None else [two, four]
    if any(other.keys() != one.keys() for other in others):
        return 'shape'
    numeric = []
    for name, columns in one.items():
        if any((other[name] is None) != (columns is None) for other in others):
            return 'shape'
        if columns is None:
            continue
        if any(other[name].keys() != columns.keys() for other in others):
            return 'shape'
        for key, cells in columns.items():
            rows = [other[name][key] for other in others]
            if any(len(row) != len(cells) for row in rows):
                return 'shape'
            values = _s13_numbers(cells)
            moved = [_s13_numbers(row) for row in rows]
            if values is None or any(row is None for row in moved):
                if values is not None or not all(_s13_same_text(cells, row) for row in rows):
                    return 'shape'
                continue
            numeric.append((values, moved))
    if any(math.isfinite(x) and not math.isfinite(y) for values, moved in numeric for row in moved
           for x, y in zip(values, row)):
        return 'nonfinite'
    scales = []
    for values, _ in numeric:
        sizes = sorted(abs(x) for x in values if math.isfinite(x))
        middle = (sizes[(len(sizes) - 1) // 2] + sizes[len(sizes) // 2]) / 2 if sizes else 0.0
        scales.append(max(1.0, middle))
    if not any(_s13_move(values, moved[0]) > S13_MOVED * scale for (values, moved), scale in zip(numeric, scales)):
        return 'unmoved'
    if four is None:
        return None
    for values, (second, fourth) in numeric:
        early, late = _s13_move(values, second), _s13_move(second, fourth)
        if not (late < early or early == late == 0):
            return 'not_contracting'
    for (values, (_, fourth)), scale in zip(numeric, scales):
        first = [x for x in values if math.isfinite(x)]
        last = [x for x in fourth if math.isfinite(x)]
        band = S13_SIDE * scale
        if first and last and ((min(first) >= -band and min(last) < -band) or (max(first) <= band
                                                                                 and max(last) > band)):
            return 'sign'
    return 'adopted'


def s13_refine(code, out, instruction, deadline, status, last_exec, *, canaries=(), replay=_NO_REPLAY,
               task_dir=None, health=None):
    """T1_S13_REFINE after s12_post: the program with its resolution literals (s13_literals) x2 and then x4, each run
    with the outputs snapshotted first, emptied for the run and normalized after it (as _s12_probe runs). x2 runs only
    while S13_X2_GATE fits beside S10_MARGIN and stops at S13_X2_CAP; x4 runs only when x1 and x2 pass s13_adopt and
    1.5 x its estimate t2 * t2 / runtime + 10 s fits, and stops there. Adopted (s13_adopt), the x4 program and its
    outputs stand: its run's measured fields, run records and fit reports, its sha256 and the replay's newest
    program. Otherwise the x1 outputs, measured fields, run records and fit reports go back. No request is sent.
    s13_refine (S13_REFINE_STATUSES) says how it went; s13_refine_literals and s13_refine_sec count. Returns
    (program, its run's seconds)."""
    began = CLOCK.now()
    found = s13_literals(code, instruction)
    status.update(s13_refine='rng' if S13_RNG.search(code) else _s13_resolution_reason(code, instruction),
                  s13_refine_literals=len(found),
                  s13_refine_sec=0.0)
    if set(re.findall(r'[a-z_]+', code.lower())) & S13_TIME_NAMES:
        status['s13_time_axis'] = 'skipped_unverified'
    if not found:
        return code, last_exec
    kept = _snapshot(out)
    if kept is None:
        status['s13_refine'] = 'too_large'
        return code, last_exec
    measured = {k: status[k] for k in MEASURED if k in status}
    runs, fits = list(status.get('runs', [])), list(health.items) if health is not None else None
    verdict, program, ran, rewritten = 'x2_failed', None, last_exec, {}
    runtime_problem = False

    def trial(source, limit):  # each trial must preserve the original and preceding runtime evidence
        nonlocal runtime_problem
        if source is None:
            return 0.0, None
        started = CLOCK.now()
        previous = {k: status[k] for k in MEASURED if k in status}
        _run_program(source, out, limit, canaries, status, clear=True, input_root=task_dir, health=health)
        seconds = CLOCK.now() - started
        if status['status'] != 'executed_ungraded':
            return seconds, None
        if runtime.regression(measured, status) or runtime.regression(previous, status):
            runtime_problem = True
            return seconds, None
        return seconds, (normalize_literals(instruction, out), _snapshot(out))
    try:
        if deadline - CLOCK.now() - S10_MARGIN < S13_X2_GATE[0] * last_exec + S13_X2_GATE[1]:
            verdict = 'skipped_time'
        else:
            limit = min(deadline - S10_MARGIN, CLOCK.now() + S13_X2_CAP[0] * last_exec + S13_X2_CAP[1])
            t2, second = trial(s13_scaled(code, found, 2), limit)
            verdict = ('runtime_regression' if runtime_problem else 'x2_failed' if second is None
                       else 'too_large' if second[1] is None
                       else s13_adopt(kept, second[1]))
            if verdict is None:
                t4 = t2 * (t2 / max(last_exec, S13_X4_FLOOR))
                need = S13_X4_GATE[0] * t4 + S13_X4_GATE[1]
                if deadline - CLOCK.now() - S10_MARGIN < need:
                    verdict = 'skipped_time'
                else:
                    program, verdict = s13_scaled(code, found, 4), 'x4_failed'
                    ran, fourth = trial(program, min(deadline - S10_MARGIN, CLOCK.now() + need))
                    if runtime_problem:
                        verdict = 'runtime_regression'
                    if fourth is not None:
                        rewritten = fourth[0]
                        verdict = 'too_large' if fourth[1] is None else s13_adopt(kept, second[1], fourth[1])
    except (OSError, HouseError):  # a setup failure or no time left: the x1 outputs stand
        pass
    finally:
        status['s13_refine'] = verdict
        if verdict != 'adopted':
            _restore(out, kept)
            for key in MEASURED:
                status.pop(key, None)
            status.update(measured)
            status['runs'] = runs
            if fits is not None:
                health.items = fits
        status['s13_refine_sec'] = round(CLOCK.now() - began, 3)
    if verdict != 'adopted':
        return code, last_exec
    _reset_counts(status, rewritten)
    status['program_sha256'] = hashlib.sha256(program.encode()).hexdigest()
    replay.put('program.py', program)
    return program, ran


def s13_coerce_on() -> bool:
    """T1_S13_COERCE=1, only under T1_S12_POST: the coerce-loss audit in s12_post (s13_coerce_audit)."""
    return os.environ.get('T1_S13_COERCE') == '1' and s12_post_on() and not minimal_on()


def _s13_coerce_name(node):
    return ((isinstance(node, ast.Name) and node.id in S13_COERCE_FNS)
            or (isinstance(node, ast.Attribute) and node.attr in S13_COERCE_FNS))


def s13_coerce_called(code) -> bool:
    """Whether the program calls to_datetime or to_numeric with errors='coerce' (a keyword, or the second positional
    argument), or passes either function to a call with that keyword (df.apply(pd.to_datetime, errors='coerce'))."""
    if 'coerce' not in code:
        return False
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return False

    def coerce(node):
        return isinstance(node, ast.Constant) and node.value == 'coerce'
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        keyword = any(item.arg == 'errors' and coerce(item.value) for item in node.keywords)
        if _s13_coerce_name(node.func) and (keyword or (len(node.args) > 1 and coerce(node.args[1]))):
            return True
        if keyword and any(_s13_coerce_name(arg) for arg in node.args):
            return True
    return False


def _s13_read_coerce(path: Path) -> list:
    """The prelude's records (COERCE_SITE) at path, each field checked: [{fn, line, lost, nonempty, recoverable, ex}].
    [] when the file is missing, not a plain file or unreadable; past S13_COERCE_BYTES the rest is not read."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, 'rb') as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                return []
            data = stream.read(S13_COERCE_BYTES)
    except OSError:
        return []
    rows = []
    for line in data.decode('utf-8', errors='replace').splitlines():
        try:
            row = json.loads(line)
        except (ValueError, RecursionError):
            continue
        if not isinstance(row, dict) or row.get('fn') not in S13_COERCE_FNS:
            continue
        numbers = [row.get(key) for key in ('line', 'lost', 'nonempty', 'recoverable')]
        examples = row.get('ex')
        if (not all(type(value) is int and value >= 0 for value in numbers) or not 0 < numbers[1] <= numbers[2]
                or numbers[3] > numbers[1] or not isinstance(examples, list)
                or not all(isinstance(value, str) for value in examples)):
            continue
        rows.append({'fn': row['fn'], 'line': numbers[0], 'lost': numbers[1], 'nonempty': numbers[2],
                     'recoverable': numbers[3],
                     'ex': [value[:S13_COERCE_CHARS] for value in examples[:S13_COERCE_EXAMPLES]]})
    return rows


def s13_coerce_items(records, canaries=()):
    """{('coerce', fn, first example): (kind, item)} of the prelude's records (_s13_read_coerce), the first record of an
    identity standing (never keyed by the line). All observations remain coerce_note: parseability
    alone does not establish the TASK's policy. Canaries are withheld from the examples."""
    found = {}
    for row in records:
        shown = [withhold_canaries(value, canaries) for value in row['ex']]
        ident = ('coerce', row['fn'], shown[0] if shown else '')
        if ident in found:
            continue
        fields = dict(line=row['line'], fn=row['fn'], lost=row['lost'], nonempty=row['nonempty'],
                      recoverable=row['recoverable'], examples=', '.join(repr(value) for value in shown))
        found[ident] = ('coerce_note', S13_COERCE_NOTE.format(**fields))
    return found


def s13_coerce_audit(code, out, canaries, status, health, deadline, last_exec, reserve, task_dir=None):
    """T1_S13_COERCE's audit in s12_post: when the program calls to_datetime or to_numeric with errors='coerce'
    (s13_coerce_called), one more run of it with the prelude (_run_program's coerce), within the measured runtime + 10 s
    and never into reserve (a fix round), its outputs snapshotted first; then the outputs, the measured fields, the run
    records and the fit reports go back. Returns s13_coerce_items' {identity: (kind, item)}; s13_coerce_audit
    (S13_COERCE_AUDITS) says how it went; s13_coerce_runs and s13_coerce_advisory
    start at 0 (s12_post counts them). Hard recovery is deliberately
    absent until an explicit semantic parsing contract exists."""
    status.update(s13_coerce_runs=0, s13_coerce_advisory=0)
    if not s13_coerce_called(code):
        status['s13_coerce_audit'] = 'no_call'
        return {}
    limit = min(deadline - reserve, CLOCK.now() + last_exec + 10)
    if limit - CLOCK.now() < last_exec + 5:
        status['s13_coerce_audit'] = 'skipped_time'
        return {}
    kept = _snapshot(out)
    if kept is None:
        status['s13_coerce_audit'] = 'too_large'
        return {}
    measured = {k: status[k] for k in MEASURED if k in status}
    runs, fits = list(status.get('runs', [])), list(health.items) if health is not None else None
    lost, clean = [], False
    try:
        _run_program(code, out, limit, canaries, status, clear=True, input_root=task_dir, health=health, coerce=lost)
        _count(status, 's13_coerce_runs')
        clean = status['status'] == 'executed_ungraded'
    except (OSError, HouseError):
        pass
    finally:
        _restore(out, kept)
        for key in MEASURED:
            status.pop(key, None)
        status.update(measured)
        status['runs'] = runs
        if fits is not None:
            health.items = fits
    status['s13_coerce_audit'] = 'ran' if clean else 'failed'
    return s13_coerce_items(lost, canaries) if clean else {}


def accept_on() -> bool:
    """T1_ACCEPT=1 (image flag): the acceptance check runs, in place of the in-program self-check."""
    return os.environ.get('T1_ACCEPT') == '1'


def checker_items(text, canaries=()):
    """[(quote, ok, detail)] from the checker's printed JSON list (ok is None unless a bool), or None when
    it printed no list. The list may be the whole output, its last line that starts with '[', or the span from
    the first '[' to the last ']'; {"checks": [...]} also counts."""
    text = text if isinstance(text, str) else ''
    candidates = [text.strip()] + [line.strip() for line in reversed(text.splitlines())
                                   if line.strip().startswith('[')]
    if '[' in text and ']' in text:
        candidates.append(text[text.index('['):text.rindex(']') + 1])
    parsed = None
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except (ValueError, RecursionError):
            continue
        if isinstance(value, dict):
            value = value.get('checks', value.get('conditions'))
        if isinstance(value, list):
            parsed = value
            break
    if parsed is None:
        return None
    items = []
    for entry in parsed[:ACCEPT_ITEMS]:
        if not isinstance(entry, dict) or not isinstance(entry.get('quote'), str):
            continue
        ok = entry.get('ok') if type(entry.get('ok')) is bool else None
        detail = entry.get('detail')
        detail = ' '.join(str(detail).split())[:300] if detail is not None else ''
        items.append((withhold_canaries(entry['quote'][:400], canaries), ok, withhold_canaries(detail, canaries)))
    return items


def accept_conditions(items, instruction):
    """([(key, quote, ok, detail)], dropped): the checker's conditions that count, keyed by their quote and
    its occurrence. Dropped: a condition that could not be evaluated (ok not a bool), whose quote is not the
    TASK's own words, or that checks a literal format the normalizer already applies."""
    try:
        literal = literal_rules(instruction)['quotes']
    except Exception:
        literal = []
    kept, dropped, seen = [], 0, Counter()
    for quote, ok, detail in items:
        if ok is None or not _quoted(quote, instruction) or any(_quoted(quote, line) for line in literal):
            dropped += 1
            continue
        quote = quote.strip(' "\'')
        norm = ' '.join(quote.lower().split())
        kept.append(((norm, seen[norm]), quote, ok, detail))
        seen[norm] += 1
    return kept, dropped


def accept_failures(conditions, now):
    """The conditions (of the first check) that do not hold in a later check's conditions now: failed, not
    evaluated or no longer reported, with the later detail."""
    later = {key: (ok, detail) for key, _, ok, detail in now}
    return [(key, quote, False, later[key][1] if key in later else 'not evaluated')
            for key, quote, _, _ in conditions if later.get(key, (None,))[0] is not True]


def _run_checker(checker, out, deadline, canaries, kept):
    """Run the acceptance checker beside the outputs, as the program runs (network guard, compatibility,
    worker environment), for at most ACCEPT_SECONDS; return its (quote, ok, detail) items, or None when it
    fails as a whole: no time, a nonzero exit, a timeout, no JSON list, or a change to the output tree (kept: its
    _lstat_tree). Whatever happens, the tree is then compared with kept as lstat sees it (content, modes, directories,
    links), and put back from kept when it differs or cannot be read."""
    limit = min(ACCEPT_SECONDS, deadline - CLOCK.now() - 5)
    if limit < 5:
        return None
    text, result, changed = '', None, True
    try:
        with tempfile.TemporaryDirectory(prefix='t1-accept-', dir='/tmp', ignore_cleanup_errors=True) as tmp:
            script = Path(tmp) / 'checker.py'
            script.write_text(checker)
            site = Path(tmp) / 'sandbox'
            site.mkdir()
            (site / 'sitecustomize.py').write_text(SANDBOX_SITE + '\n\n' + COMPAT_SITE)
            log = Path(tmp) / 'checker.out'
            root = Path(tmp).parent
            before = set(os.listdir(root))

            def limits():
                import resource
                resource.setrlimit(resource.RLIMIT_FSIZE, (8 * 1024 * 1024, 8 * 1024 * 1024))
            with log.open('wb') as sink:
                process = subprocess.Popen([sys.executable, str(script)], cwd=out, env=worker_environment(site=site),
                                           stdout=sink, stderr=subprocess.DEVNULL, start_new_session=True,
                                           close_fds=True, preexec_fn=limits)
                try:
                    result = process.wait(timeout=limit)
                except subprocess.TimeoutExpired:
                    result = None
                finally:  # (a descendant that left the group with setsid is not stopped, as for the program)
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    if result is None:
                        process.wait()
                    _sweep(root, before)
            try:
                fd = os.open(log, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
                with os.fdopen(fd, 'rb') as stream:
                    text = stream.read(ACCEPT_LIMIT).decode('utf-8', errors='replace')
            except OSError:
                text = ''
    finally:  # a checker must not write: the outputs go back to what the program wrote
        try:
            changed = _lstat_tree(out) != kept
        except Exception:  # a tree it left unreadable, or removed, counts as changed
            changed = True
        if changed:
            _put_back(out, kept)
    return checker_items(text, canaries) if result == 0 and not changed else None


def accept_outputs(client, messages, code, out, instruction, deadline, status, pace, last_exec, *, canaries=(),
                   replay=_NO_REPLAY, task_dir=None, path_index=None, sleep=None, checks=None, health=None):
    """T1_ACCEPT, after a clean run and before the review: one request asks for a checker of the conditions the
    TASK states about the outputs, which runs beside them (_run_checker). Conditions that could not be
    evaluated or do not quote the TASK are dropped (accept_conditions); a checker that fails as a whole ends the
    step with nothing changed. Failing conditions open at most ACCEPT_ROUNDS guarded fix rounds: a fix is kept
    only when its rerun exits 0, keeps every output file, changes the outputs, adds no deterministic or advisory
    flag and no fit report, takes nothing away from them (shape_losses: no file emptied, no CSV column or row and
    no JSON key or list item lost; growth is allowed), and, under the same checker, strictly fewer of the first
    check's conditions fail and none that held fails (a stated expectation can be loose for the given inputs: a
    fix must not break a formula to force it); otherwise the earlier outputs and measured fields are restored, and
    the next round asks again, saying why. Each request, each checker run and a fix's rerun leave time for the
    review's own gate (a fix that runs into it is stopped and rejected), and each request leaves
    ACCEPT_KEEP_REQUESTS requests. Returns (program, request fields of the last kept fix or None, the duration of
    the run whose outputs stand)."""
    health = health if health is not None else FitWarnings()
    later = lambda: pace.patch() + EXEC_FLOOR  # the review's own gate stays free
    check = ACCEPT_SECONDS + 5  # _run_checker's own limit and margin
    if (client.attempts + 1 + ACCEPT_KEEP_REQUESTS > client.max_requests
            or deadline - CLOCK.now() < pace.patch() + check + later()):
        status['accept_status'] = 'skipped_time'
        return code, None, last_exec
    try:
        kept = _lstat_tree(out)  # as lstat sees it: what the checker and a rejected fix must leave as it was
    except Exception:
        status['accept_status'] = 'failed'
        return code, None, last_exec
    if kept is None:
        status['accept_status'] = 'too_large'
        return code, None, last_exec
    ask = ACCEPT_ASK.format(out=out, digest=digest(out), lines=ACCEPT_LINES, sec=ACCEPT_SECONDS)
    conversation = messages + [{'role': 'assistant', 'content': withhold_canaries(_fence(code), canaries)},
                               {'role': 'user', 'content': withhold_canaries(ask, canaries)}]
    try:
        reply = _request(client, conversation, deadline, pace, 'accept', sleep, need=pace.patch(),
                         reserve=check + later(), keep=ACCEPT_KEEP_REQUESTS)
        pace.measure(client, reply)
        checker = extract_python(_text(reply), reply['finish_reason_raw'])
    except _Stop:
        status['accept_status'] = 'skipped_time'
        return code, None, last_exec
    except HouseError as exc:
        status['accept_status'] = 'no_checker' if exc.kind == 'model_output' else 'failed'
        return code, None, last_exec
    if task_dir is not None:  # the inputs at the paths the TASK names, as for the program (zero requests)
        checker = prepared(checker, task_dir, path_index)
    replay.put('checker.py', checker)
    try:
        first = _run_checker(checker, out, deadline - later(), canaries, kept)
    except Exception:  # a setup failure: the outputs stand as the program wrote them
        first = None
    replay.put('checker.json', json.dumps(first))  # local evaluation only
    if first is None:
        status['accept_status'] = 'checker_failed'
        return code, None, last_exec
    conditions, dropped = accept_conditions(first, instruction)
    failing = [entry for entry in conditions if not entry[2]]
    status.update(accept_conditions=len(conditions), accept_dropped=dropped, accept_failed=len(failing),
                  accept_rounds=0, accept_fixed=0)
    if not conditions or not failing:
        status['accept_status'] = 'clean' if conditions else 'no_conditions'
        return code, None, last_exec
    contract = output_contract(instruction)

    def deterministic():
        return contract_flags(contract, out) + bound_flags(instruction, out) + sanity_flags(out)
    flags, hints, fits = deterministic(), sum(advisory_flags(instruction, out), []), health.keys()
    held = {key for key, _, ok, _ in conditions if ok}  # a fix must not break one of these
    normalize = (lambda text: prepared(text, task_dir, path_index)) if task_dir else None
    kept_fields, note = None, ''

    def end(value):  # a fix kept in an earlier round stands, and a rejected one is not hidden by a later stop
        if status['accept_fixed'] or status.get('accept_status') != 'fix_rejected' or value == 'fix_rejected':
            status['accept_status'] = 'fixed' if status['accept_fixed'] else value
    for round_ in range(ACCEPT_ROUNDS):
        if client.attempts + 2 + ACCEPT_KEEP_REQUESTS > client.max_requests:  # a cut patch is continued once
            end('skipped_time')
            break
        _count(status, 'accept_rounds')
        items = '\n'.join(f'- "{quote}" does not hold' + (f': {detail}' if detail else '')
                          for _, quote, _, detail in failing[:ACCEPT_LISTED])
        ask = ACCEPT_FIX.format(items=items, note=note + library_fix(messages, code, 'a listed output comes from'))
        conversation = messages + [{'role': 'assistant', 'content': withhold_canaries(_fence(code), canaries)},
                                   {'role': 'user', 'content': withhold_canaries(ask, canaries)}]
        rerun = last_exec + 10 + check + later()
        try:
            fixed = _patch(client, conversation, code, deadline, status, rerun, canaries, pace, sleep, None,
                           'accept_fix', normalize, keep=ACCEPT_KEEP_REQUESTS)[0]
        except _Stop:
            end('skipped_time')
            break
        except HouseError:
            fixed = None
        if fixed is None:
            end('no_patch')
            break
        fields = {k: client.diagnostics.get(k) for k in ('http_status', 'finish_reason')}
        if task_dir is not None:
            fixed = prepared(fixed, task_dir, path_index)
        if deadline - CLOCK.now() < rerun:
            end('skipped_time')
            break
        measured = {k: status[k] for k in MEASURED if k in status}
        kept_checks = (list(checks.items), checks.written) if checks is not None else None
        kept_health = list(health.items)
        accepted, why, ran = False, 'its run failed', None
        try:
            status['program_sha256'] = hashlib.sha256(fixed.encode()).hexdigest()
            replay.put('program.py', fixed)
            started = CLOCK.now()  # a fix that runs into the checker's and the review's time is stopped there
            _run_program(fixed, out, deadline - check - later(), canaries, status, clear=True, replay=replay,
                         input_root=task_dir, checks=checks, health=health)
            ran = CLOCK.now() - started
            if status['status'] == 'execution_timeout':
                why = 'its run did not finish in the time left'
            if status['status'] == 'executed_ungraded':
                rewritten = normalize_literals(instruction, out)
                after, hints_after, now = deterministic(), sum(advisory_flags(instruction, out), []), _lstat_tree(out)
                if now is None:
                    why = 'its outputs grew too large to check'
                elif not _tree_files(kept) <= _tree_files(now):
                    why = 'it did not write every output file'
                elif now == kept:
                    why = 'its outputs did not change'
                elif runtime_problem := runtime.regression(measured, status):
                    why = runtime_problem
                elif not (_no_worse(flags, after) and _no_worse(hints, hints_after) and health.keys() <= fits):
                    why = 'it made other checks of the outputs worse'
                elif lost := shape_losses(kept, now):  # a condition can be met by dropping what fails it
                    why = 'it removed part of the outputs (' + '; '.join(lost) + ')'
                else:
                    second = _run_checker(checker, out, deadline - later(), canaries, now)
                    replay.put('checker.json', json.dumps(second))
                    left = (accept_failures(conditions, accept_conditions(second, instruction)[0])
                            if second is not None else None)
                    if left is None:
                        why = 'the check could not run on its outputs'
                    elif held & {key for key, _, _, _ in left}:
                        why = 'a stated condition that held before no longer holds'
                    elif len(left) < len(failing):
                        accepted = True
                    else:
                        why = 'the listed conditions still do not hold'
        except Exception:  # a setup failure or no time left: the earlier outputs stand
            accepted = False
        if not accepted:
            try:
                _put_back(out, kept)
                restored = True
            except Exception:  # never raised out of the step: the tree stays as far as it was made again
                restored = False
            if checks is not None:
                checks.items, checks.written = kept_checks
            health.items = kept_health
            for key in MEASURED:
                status.pop(key, None)
            status.update(measured)
            if not restored:
                status['accept_status'] = 'failed'
                break
            end('fix_rejected')
            note = ACCEPT_AGAIN.format(why=why)
            continue
        _reset_counts(status, rewritten)
        _count(status, 'accept_fixed')
        status['accept_status'] = 'fixed'
        code, kept_fields, kept, failing, note, last_exec = fixed, fields, now, left, '', ran
        flags, hints, fits = after, hints_after, health.keys()
        held = {key for key, _, _, _ in conditions} - {key for key, _, _, _ in left}
        if not failing:
            break
    status['accept_left'] = len(failing)
    return code, kept_fields, last_exec


def _patch(client, conversation, code, deadline, status, reserve, canaries, pace, sleep, near, kind, normalize=None,
           keep=0, info=None, cap=None, *, outcome=None, one_request=False):
    """(program or None, why nothing changed, the last reply's text) for one patch request, a hunk splice
    for an indented reply, and at most one continuation of a cut patch; none of them leaves fewer than keep
    requests (_request). A reasoning budget goes only on the first request of a repair round's patch
    (patchthink_budget: T1_PATCHTHINK, and always under T1_S10), on T1_S12_POST's fix round's (s12_fix, the same
    budget) and, under T1_S10, on the flags-only round's (review_patch: S10_FLAGFIX_BUDGET); retries and
    continuations go without one, as does everything after the S10 breaker has tripped (_request). T1_S12_MECH: info
    (a dict) receives whether the first request went out (sent) and whether the patch ended at the cap (cut: its last
    reply, a continuation's where one came back), and that request is not retried after a timeout; cap (the light
    patch's S12_PATCH_CAP) caps each reply, and its gate and timeout follow that reply's size. outcome, when
    supplied, observes whether the last _request returned a valid reply (reply_returned); it changes no retry
    policy, unlike info, and a failed or gated continuation resets it even if finish metadata survives. With the
    default-false one_request, only one actual attempt is admitted, reserving all other remaining requests through
    _request's existing keep gate and suppressing patch_continue. Only S12 same-definition feedback sets it."""
    program, pending, changed, content, hunk = code, '', False, '', False
    selfcheck = SELFCHECK_BLOCK in conversation[0]['content']
    budget = (patchthink_budget(status) if kind in ('patch', 's12_fix') else
              s10_budget('flagfix', status=status) if kind == 'review_patch' and s10_on() else None)
    if one_request:  # the optional S12 same-definition feedback spends at most one actual attempt
        keep = max(keep, client.max_requests - client.attempts - 1)
    for step in ((kind,) if one_request else (kind, 'patch_continue')):
        attempts = client.attempts
        need, sized = pace.patch() if step != 'patch_continue' else pace.full(), {}
        if cap:  # T1_S12_MECH's light patch: a reply of at most cap tokens
            need = min(need, cap / pace.rate + 6)
            sized = dict(max_tokens=cap, expect=need)
        if outcome is not None:
            outcome['reply_returned'] = False
        try:
            reply = _request(client, conversation, deadline, pace, step, sleep, need=need, reserve=reserve,
                             keep=keep, budget=budget if step != 'patch_continue' else None,
                             once=one_request or (info is not None and step != 'patch_continue'), **sized)
            if outcome is not None:
                outcome['reply_returned'] = True
        except (_Stop, HouseError):
            if info is not None and step != 'patch_continue':
                info['sent'] = client.attempts > attempts
            if step != 'patch_continue':
                raise
            break  # keep the part of the patch already merged
        finally:
            if client.attempts > attempts:
                _count(status, 'patch_requests')
        if info is not None and step != 'patch_continue':
            info.update(sent=True, cut=reply['finish_reason_raw'] == 'length')
        elif info is not None:  # T1_S12_MECH: a continuation that finished leaves a whole patch, not a cut one
            info['cut'] = reply['finish_reason_raw'] == 'length'
        if budget and kind == 'patch' and step != 'patch_continue':  # with the budget, or not when the clock was short
            record = client.diagnostics['requests'][-1]
            if 'reasoning_budget' in record:
                _count(status, 'patchthink_used')
                _count(status, 'patchthink_reasoning_chars', record.get('reasoning_chars', 0))
            else:
                _count(status, 'patchthink_skipped')
        pace.measure(client, reply)
        content, finish = _text(reply), reply['finish_reason_raw']
        if selfcheck and finish == 'stop':  # a patch whose only compile error is in its self-check is taken
            content = _stub_reply(content, status)
        region = code_region(content)
        target_state, targeted, target_why = repair_target(
            program, content, near if step in ('patch', 'patch_retry') and finish == 'stop' else None)
        if target_state == 'rejected':
            return None, target_why, content
        if target_state == 'handled':
            replaced = targeted != program
            _count(status, 'patch_stmts_replaced', int(replaced))
            client.diagnostics['requests'][-1].update(candidate='S', stmts_added=0)
            program, changed = targeted, changed or replaced
            break
        first = next((line for line in region.split('\n') if line.strip()), '')
        if step != 'patch_continue' and first[:1].isspace():  # changed lines inside a function
            hunk = True
            placed = splice_hunk(program, region, near) if finish != 'length' else None
            if placed is None and finish != 'length':  # one or two lines without a context line at each end
                placed = anchor_hunk(program, region, near)
                if placed is not None:
                    _count(status, 'patch_hunks_anchored')
            if placed is not None:
                _count(status, 'patch_hunks')
                program, changed = placed, True
            client.diagnostics['requests'][-1].update(candidate='H', stmts_added=0)
            break
        program, pending, st = absorb(program, pending, content, finish, 'repair')
        changed = changed or st['changed']
        _count(status, 'patch_stmts_added', st['added'])
        _count(status, 'patch_stmts_replaced', st['replaced'])
        _count(status, 'patch_loose_stmts', st['loose'])
        status['patch_was_full_program'] = bool(status.get('patch_was_full_program') or st['restart'])
        client.diagnostics['requests'][-1].update(candidate=st['candidate'], stmts_added=st['added'])
        if finish != 'length' or not pending.strip():
            break
        chunk = verified_prefix(code_region(content), False)[0]
        start = anchors(program, pending)[1]
        ask = PATCH_CONTINUE.format(start=f'the statement that begins `{start}`' if start else 'the next statement')
        conversation = conversation + [{'role': 'assistant', 'content': withhold_canaries(_fence(chunk), canaries)},
                                       {'role': 'user', 'content': withhold_canaries(ask, canaries)}]
    # Changed as the run would see it: after the path rewrite, and by syntax tree (re-spacing is no change).
    if changed and _shape((normalize or (lambda text: text))(program)) != _shape(code):
        return program, '', content
    return None, _unchanged_reason(content, hunk, code), content


def _existing_hunk(code, region):
    """Only exact physical code lines count; text inside a spanning literal does not.

    This diagnoses a no-change reply, never authorizes a splice or a task value.
    Ambiguous/unparseable source remains with the ordinary unmatched feedback.
    """
    code = code.replace('\r\n', '\n')
    lines = code.split('\n')
    chunk = region.replace('\r\n', '\n').strip('\n').split('\n')
    if not chunk or not any(line.strip() for line in chunk):
        return False
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(code).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError, ValueError):
        return False
    ignored = {tokenize.ENCODING, tokenize.ENDMARKER, tokenize.INDENT, tokenize.DEDENT,
               tokenize.NEWLINE, tokenize.NL, tokenize.COMMENT}
    for start in range(len(lines) - len(chunk) + 1):
        if lines[start:start + len(chunk)] != chunk:
            continue
        first, last = start + 1, start + len(chunk)
        covered = [token for token in tokens if token.type not in ignored
                   and token.start[0] <= last and token.end[0] >= first]
        if covered and all(token.start[0] >= first and token.end[0] <= last for token in covered):
            return True
    return False


_EXISTING_HUNK_REASON = 'its exact code lines are already present in the program; the program is unchanged'


def _unchanged_retry(why, error, where):
    if why == _EXISTING_HUNK_REASON:
        return ('Your reply did not change the program: ' + why + '. It did not address the error' + error
                + ', so the program would fail the same way' + where + '. Fix the actual failing code. '
                + PATCH_FORMAT)
    return RETRY.format(why=why, error=error, where=where)


def _unchanged_reason(content, hunk, code=''):
    region = code_region(content) if isinstance(content, str) else ''
    if not region.strip() or ('```' not in content and not _compiles(region)):
        return 'it contained no code'
    if hunk:
        if _existing_hunk(code, region):
            return _EXISTING_HUNK_REASON
        return ('its lines could not be matched to one place in the program, so send complete def blocks '
                'instead')
    try:
        compile(region, '<reply>', 'exec', dont_inherit=True)
    except SyntaxError as exc:
        text = (exc.text or '').strip()[:120]
        return f'it is not valid Python (SyntaxError: {exc.msg} at line {exc.lineno} of your reply: `{text}`)'
    except (ValueError, RecursionError, MemoryError, OverflowError):
        return 'it is not valid Python'
    return "the code you sent is identical to the program's current code"


def _where(code, line):
    """' at line N in `f`: `source`' for the failing line of code, or ''."""
    lines = code.split('\n')
    if type(line) is not int or not 1 <= line <= len(lines) or not lines[line - 1].strip():
        return ''
    function = ''
    try:
        spans = [(node.lineno, node.name) for node in ast.walk(ast.parse(code))
                 if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                 and node.lineno <= line <= (node.end_lineno or node.lineno)]
        function = f' in `{max(spans)[1]}`' if spans else ''
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        pass
    return f' at line {line}{function}: `{lines[line - 1].strip()[:160]}`'


def _failing_statement(code, line):
    """(enclosing function, stripped source) of the failing line: two failures are the same only if both match."""
    lines = code.split('\n')
    if type(line) is not int or not 1 <= line <= len(lines):
        return ('', '')
    function = ''
    try:
        spans = [(node.lineno, node.name) for node in ast.walk(ast.parse(code))
                 if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                 and node.lineno <= line <= (node.end_lineno or node.lineno)]
        function = max(spans)[1] if spans else '<module>'
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        pass
    return (function, lines[line - 1].strip())


def _same_crash(failure, previous, before):
    """Whether a run failed as the previous one did: the same exception type in the same innermost function of
    the program, at the same line text, or with the same exception message at a line the patch wrote (one that
    function of the patched program `before` did not have). A patch that changed the failing line but still raises
    the same error there made no progress; a crash that moved on to a line the patch left alone, with a message
    as generic as a bare AssertionError or ZeroDivisionError, is not the same crash.
    failure: (type, function, line text, message)."""
    if previous is None or not failure[1] or failure[:2] != previous[:2]:
        return False
    if failure[2] and failure[2] == previous[2]:
        return True
    return bool(failure[3] and failure[3] == previous[3] and failure[2]
                and (failure[1], failure[2]) not in _function_lines(before))


def same_failure(failure, previous, names):
    """The crash request's note that the last patch left the same crash (failure, previous: (type, function, line
    text, exception line)). Under T1_PRECHECK, a missing module attribute is named as the cause on that line (with
    the library's closest names when the module is the library), instead of pointing elsewhere."""
    fields = dict(names=f' (to {", ".join(names)})' if names else '', exc=failure[0] or 'error',
                  error=_error_note(failure[3]),
                  place='at the same line' if failure[2] == previous[2] else f'again in `{failure[1]}`')
    missing = _NO_ATTRIBUTE.search(failure[3] or '') if precheck_on() else None
    if missing is None:
        return SAME_FAILURE.format(**fields)
    module, attr = missing.groups()
    known = (library_exports() or {}).get('' if module == LIBRARY else module.removeprefix(LIBRARY + '.'))
    close = difflib.get_close_matches(attr, sorted(known), n=3) if known and module.split('.')[0] == LIBRARY else []
    return SAME_ATTRIBUTE.format(module=module, attr=attr, close=' (such as ' + ', '.join(f'`{c}`' for c in close)
                                 + ')' if close else '', **fields)


def _changed_names(before, after):
    """Top-level def/class names whose code differs between two programs (for the model only)."""
    def named(code):
        try:
            return {node.name: ast.dump(node) for node in ast.parse(code).body
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
        except (SyntaxError, ValueError, RecursionError, MemoryError):
            return {}
    old, new = named(before), named(after)
    return [name for name in new if old.get(name) != new[name]][:5]


_APP_PATH = re.compile(r'/app(?:/[^\s\'"{}\\,;*?<>|]*)?')
PATH_INDEX_LIMIT = 5000


def input_path_index(task_dir):
    """(COPY mappings, file name -> paths) for rewrite_input_paths; a bounded walk that skips links."""
    names, seen = {}, 0
    try:
        mappings = sorted(input_mapping(task_dir), key=lambda pair: -len(pair[0]))
        for root, directories, files in os.walk(task_dir):
            directories[:] = sorted(d for d in directories if not os.path.islink(os.path.join(root, d)))
            for name in sorted(files):
                path = os.path.join(root, name)
                if os.path.islink(path):
                    continue
                if seen >= PATH_INDEX_LIMIT:
                    return mappings, names
                names.setdefault(name, []).append(path)
                seen += 1
    except (OSError, RecursionError, ValueError):
        return [], {}
    return mappings, names


def rewrite_input_paths(code, task_dir, index=None):
    """Zero-request, before every run: the task text names inputs at /app paths the agent does not have. A
    string literal (or an f-string's leading part) that starts with an /app path outside /app/output is
    rewritten when it names an existing input: through the task's COPY mapping, then /app/<rest> as
    <task_dir>/<rest>, then a file name unique among the inputs. Returns (code, paths rewritten)."""
    if '/app/' not in code:
        return code, 0
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(code).readline))
    except (tokenize.TokenError, SyntaxError, ValueError):
        return code, 0
    mappings, names = index if index is not None else input_path_index(task_dir)
    middle = getattr(tokenize, 'FSTRING_MIDDLE', None)
    edits, previous = [], None
    for token in tokens:
        if token.start[0] == token.end[0] and (token.type == tokenize.STRING or (
                token.type == middle and previous is not None and previous.type == tokenize.FSTRING_START)):
            for match in _APP_PATH.finditer(token.string):
                at = match.start()
                if token.type == tokenize.STRING and (at == 0 or token.string[at - 1] not in '\'"'):
                    continue
                if token.type == middle and at != 0:
                    continue
                target = _input_path(match.group(0), mappings, task_dir, names)
                if target is not None:
                    edits.append((token.start[0], token.start[1] + at, token.start[1] + match.end(), target))
        previous = token
    if not edits:
        return code, 0
    lines = code.split('\n')
    for row, begin, end, target in sorted(edits, reverse=True):
        lines[row - 1] = lines[row - 1][:begin] + target + lines[row - 1][end:]
    fixed = '\n'.join(lines)
    try:
        ast.parse(fixed)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return code, 0
    return fixed, len(edits)


def library_fixed(code, status=None):
    """Under T1_PRECHECK, the program with its references to names the optional library does not have rewritten
    (rewrite_library_refs: to the program's own name, with the library's import line where the program has none,
    or to the package's name); library_ref_rewrites counts them in status. Without the flag, the program as it
    is. Under T1_S11_LINT, first a pyplot backend call that cannot work becomes one that does (backend_fixed;
    s11_backend_fixed counts the changes), whatever T1_PRECHECK says. Under T1_S12_POST, then a call of a name the
    program never binds that numpy has becomes numpy's (numpy_names_fixed; s12_lint_fixed counts the calls)."""
    if s11_lint_on():
        code, changed = backend_fixed(code)
        if changed and status is not None:
            _count(status, 's11_backend_fixed', changed)
    if s12_post_on():
        code, named = numpy_names_fixed(code)
        if named and status is not None:
            _count(status, 's12_lint_fixed', named)
    if not precheck_on():
        return code
    code, fixed = rewrite_library_refs(code)
    if fixed and status is not None:
        _count(status, 'library_ref_rewrites', fixed)
    return code


def precheck_fixed(code, status, skip=()):
    """Under T1_PRECHECK, a newly generated program before its first run (zero requests): a name a function reads
    that only its callers bind becomes its parameter, passed by every call (lift_free_names; precheck_params_added
    counts the parameters), then a call that leaves out a parameter its scope already has passes it (bind_missing;
    precheck_calls_bound). The defs named in skip stay as written. Callers keep this program, so the requests
    after the run show the model the code that ran. Without the flag, the program as it is."""
    if not precheck_on():
        return code
    code, added = lift_free_names(code, skip)
    code, calls = bind_missing(code, skip)
    for key, amount in (('precheck_params_added', added), ('precheck_calls_bound', calls)):
        if amount:
            _count(status, key, amount)
    return code


def prepared(code, task_dir, index=None):
    """The program as it runs: rewrite_input_paths, then library_fixed (zero requests, before every run)."""
    return library_fixed(rewrite_input_paths(code, task_dir, index)[0])


def _input_path(path, mappings, task_dir, names):
    core, slash = path.rstrip('/'), '/' if path.endswith('/') else ''
    if core in ('/app', '/app/output') or core.startswith('/app/output/') or '/..' in core:
        return None
    candidates = [real + core[len(app):] for app, real in mappings if core == app or core.startswith(app + '/')]
    candidates.append(str(task_dir) + core[len('/app'):])
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate + slash
    same = names.get(PurePosixPath(core).name, [])
    if '.' in PurePosixPath(core).name and len(same) == 1 and not slash:
        return same[0]
    return None


def rewrite_missing_path(code, tail, files):
    """Zero-request fix after FileNotFoundError: a string literal equal to the missing path whose
    basename names exactly one input file is replaced by that file's absolute path."""
    found = _MISSING_FILE.findall(tail)
    if not found:
        return None
    missing = found[-1]
    same = [path for path in files if PurePosixPath(path).name == PurePosixPath(missing).name]
    if len(same) != 1 or same[0] == missing:
        return None
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return None
    nodes = sorted(((n.lineno, n.col_offset, n.end_col_offset) for n in ast.walk(tree)
                    if isinstance(n, ast.Constant) and n.value == missing and n.lineno == n.end_lineno), reverse=True)
    if not nodes:
        return None
    lines = code.split('\n')
    for lineno, start, end in nodes:  # offsets are UTF-8 bytes
        raw = lines[lineno - 1].encode()
        lines[lineno - 1] = (raw[:start] + repr(same[0]).encode() + raw[end:]).decode()
    fixed = '\n'.join(lines)
    try:
        ast.parse(fixed)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return None
    return fixed


def _log_tail(path: Path, canaries, limit: int = 4000) -> str:
    """The failed run's last output for the model only; never exported, canaries withheld."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, 'rb') as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode):
                return ''
            stream.seek(max(0, info.st_size - limit))
            tail = stream.read(limit).decode('utf-8', errors='replace')
    except OSError:
        return ''
    if info.st_size > limit:  # the cut may fall inside a GUID: its remainder is hex and hyphens only
        tail = re.sub(r'\A[0-9a-fA-F-]+', '', tail)
    return withhold_canaries(tail, canaries)


FIT_WARNINGS = 4  # distinct fit warnings shown to the review
FIT_LOG_BYTES = 1024 * 1024  # bytes read from each end of a run log for the scan
_FIT_CATEGORY = re.compile(r'\b(?:ConvergenceWarning|OptimizeWarning|HessianInversionWarning|DataScaleWarning|'
                           r'IntegrationWarning)\b')
_FIT_MESSAGE = re.compile(r'\b(?:did not|does not|failed to|fails to|not|never|no longer) converg|'
                          r'maximum number of (?:iterations|function evaluations)|iteration limit|'
                          r'optimizer returned code|constraints incompatible|not making good progress|'
                          r'reached maxfev|desired error not necessarily achieved', re.I)
_WARNING_LINE = re.compile(r'^(?:.*?:\d+: )?([A-Za-z_][\w.]*Warning): ?(.*)$')
_NOT_A_REPORT = re.compile(r'^\s|^(?:Traceback|raise\b)|^[\w.]+(?:Error|Exception): ')  # frames, source lines
_NOT_A_FIT = re.compile(r'Unknown solver options\b', re.I)  # settings the chosen method ignores: no fit outcome


def _fit_key(message):
    """A fit report's identity: its message with the numbers left out."""
    return re.sub(r'[-+]?\d[\d.]*(?:e[-+]?\d+)?', '#', message.lower())


def fit_warnings(text: str) -> list:
    """[(message, count)] of optimizer and fit non-convergence reports in a run's output, in the order first
    seen: a warning of a fit-health category (ConvergenceWarning, OptimizeWarning, HessianInversionWarning,
    DataScaleWarning, IntegrationWarning), or any line saying a fit or solver did not converge or hit its
    iteration limit. A generic RuntimeWarning (invalid value, divide by zero, overflow) is not one; indented lines
    (a warning's source line, traceback frames), exception lines and a warning of solver options the method
    ignores are not either. A warning's file and line
    prefix is dropped and a message that runs on over lines (after a colon, or cut mid-sentence) is kept whole;
    reports that differ only in numbers count as one."""
    lines, found, taken = text.split('\n'), {}, set()
    for index, line in enumerate(lines):
        if index in taken or _NOT_A_REPORT.search(line):
            continue
        warning = _WARNING_LINE.match(line.strip())
        category = warning[1].rsplit('.', 1)[-1] if warning else ''
        if not (_FIT_CATEGORY.fullmatch(category) or _FIT_MESSAGE.search(line)) or (
                warning and _NOT_A_FIT.match(warning[2])):
            continue
        message = f'{category}: {warning[2]}' if warning else line.strip()
        for number in range(index + 1, min(index + 5, len(lines))) if warning else ():
            extra = lines[number]  # a message that runs on: after a colon, or cut mid-sentence
            if (message.rstrip()[-1:] in ('.', '!', '?') or not extra.strip() or extra[:1].isspace()
                    or _WARNING_LINE.match(extra.strip())):
                break
            message += ' ' + extra.strip()
            taken.add(number)
        message = ' '.join(message.split())[:400]
        key = _fit_key(message)
        if key in found:
            found[key][1] += 1
        else:
            found[key] = [message, 1]
    return [tuple(item) for item in found.values()]


FIT_FIX = ('the program\'s own run reported `{message}`{times}: a fit or solver that did not converge returns its '
           'starting or last values rather than the estimate, so every output computed from it is suspect. Make '
           'that fit converge; where it fits a method that {functions} (LIBRARY MATCHES above) computes, which this '
           'program does not call, call that function with the TASK\'s conventions instead of fitting by hand or '
           'with another package')


class FitWarnings:
    """Fit non-convergence reported by the last run that exited 0 (fit_warnings of its log): text for the model
    only, canaries withheld, never in the status. A run that did not exit 0 leaves none."""

    def __init__(self):
        self.items = []

    def read(self, path: Path, canaries, clean: bool) -> None:
        self.items = []
        if not clean:
            return
        try:
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            with os.fdopen(fd, 'rb') as stream:
                info = os.fstat(stream.fileno())
                if not stat.S_ISREG(info.st_mode):
                    return
                data = stream.read(FIT_LOG_BYTES)
                if info.st_size > 2 * FIT_LOG_BYTES:
                    stream.seek(info.st_size - FIT_LOG_BYTES)
                    data += b'\n' + stream.read(FIT_LOG_BYTES)
                elif info.st_size > FIT_LOG_BYTES:
                    data += stream.read()
        except OSError:
            return
        self.items = [(withhold_canaries(message, canaries), count)
                      for message, count in fit_warnings(data.decode('utf-8', errors='replace'))]

    def keys(self) -> set:
        """The identity of every report (not only those hints() shows), whatever its count: a fix may not add one."""
        return {_fit_key(message) for message, _ in self.items}

    def fix_items(self, functions) -> tuple:
        """(items, keys) for the fix round (T1_LIBHINT, functions: library_fits): one FIT_FIX line per report (at
        most FIT_WARNINGS) and the identities of those reports; nothing without functions."""
        if not functions:
            return [], set()
        shown = self.items[:FIT_WARNINGS]
        names = ', '.join(f'`{LIBRARY}.{name}`' for name in functions)
        return ([FIT_FIX.format(message=message, times=f' ({count} times)' if count > 1 else '', functions=names)
                 for message, count in shown], {_fit_key(message) for message, _ in shown})

    def hints(self) -> list:
        """One advisory line per report (at most FIT_WARNINGS), for the review."""
        return [f'the program\'s own run reported `{message}`' + (f' ({count} times)' if count > 1 else '')
                + ': a fit or solver that did not converge returns its starting or last values rather than the '
                'estimate, so any output computed from it is suspect' for message, count in self.items[:FIT_WARNINGS]]


def _remove(path: Path) -> None:
    """Delete one entry without following links. The failed program (same uid) may have left read-only
    or unreadable directories at any depth: open them top-down (iteratively), then remove the tree."""
    if path.is_symlink() or not path.is_dir():
        path.unlink()
        return
    path.chmod(stat.S_IRWXU)
    for root, dirs, _ in os.walk(path):
        for name in dirs:
            if not os.path.islink(os.path.join(root, name)):
                os.chmod(os.path.join(root, name), stat.S_IRWXU)
    shutil.rmtree(path)


def _clear(out: Path) -> None:
    """A retried program starts from an empty output tree; earlier files never reach the checks. A tree that a
    run removed or replaced is made again, and one it left without owner access is opened first."""
    if not out.is_dir():
        if out.is_symlink() or out.exists():
            out.unlink()
        out.mkdir(parents=True, exist_ok=True)
    mode = out.stat().st_mode
    if mode & stat.S_IRWXU != stat.S_IRWXU:
        out.chmod(stat.S_IMODE(mode) | stat.S_IRWXU)
    for entry in list(out.iterdir()):
        _remove(entry)


def _lstat_tree(out: Path):
    """The output tree as lstat sees it, for a step that must leave it exactly as it was (the acceptance check):
    {relative path: (kind, mode, content)} for every entry, links never followed (content is a file's bytes or a
    link's target; kind 'other' is a FIFO, socket or device, never opened), with the tree's own mode under ''.
    None when its files exceed REVIEW_SNAPSHOT_CAP; an entry that cannot be read raises OSError."""
    tree, size, pending = {'': ('dir', stat.S_IMODE(os.stat(out).st_mode), None)}, 0, ['']
    while pending:
        folder = pending.pop()
        with os.scandir(out / folder if folder else out) as entries:
            for entry in entries:
                name = f'{folder}/{entry.name}' if folder else entry.name
                info = entry.stat(follow_symlinks=False)
                mode = stat.S_IMODE(info.st_mode)
                if stat.S_ISLNK(info.st_mode):
                    tree[name] = ('link', None, os.readlink(entry.path))  # a link's own mode is the platform's
                elif stat.S_ISDIR(info.st_mode):
                    tree[name] = ('dir', mode, None)
                    pending.append(name)
                elif stat.S_ISREG(info.st_mode):
                    with os.fdopen(os.open(entry.path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), 'rb') as stream:
                        data = stream.read(REVIEW_SNAPSHOT_CAP - size + 1)
                    size += len(data)
                    if size > REVIEW_SNAPSHOT_CAP:
                        return None
                    tree[name] = ('file', mode, data)
                else:
                    tree[name] = ('other', mode, None)
    return tree


def _put_back(out: Path, tree: dict) -> None:
    """Make the output tree what _lstat_tree recorded again: every file, directory and link, with its mode (a FIFO,
    socket or device is not made again)."""
    _clear(out)
    folders = []
    for name in sorted(name for name in tree if name):  # a directory sorts before every entry under it
        kind, mode, content = tree[name]
        path = out / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if kind == 'dir':
            path.mkdir(mode=stat.S_IRWXU, exist_ok=True)
            folders.append((path, mode))
        elif kind == 'file':
            path.write_bytes(content)
            path.chmod(mode)
        elif kind == 'link':
            os.symlink(content, path)
    for path, mode in reversed(folders):  # the deepest first, so that a read-only directory is filled before
        path.chmod(mode)
    out.chmod(tree[''][1])


def _tree_files(tree) -> set:
    return {name for name, (kind, _, _) in tree.items() if kind == 'file'}


SHAPE_NODES = 200_000  # JSON values walked per file; a larger document is compared by its emptiness only


def _table_shape(data, delimiter):
    """(header cells, data rows) of a CSV or TSV, blank rows left out; None when it does not read as one."""
    try:
        rows = [row for row in csv.reader(io.StringIO(data.decode('utf-8-sig', errors='replace')), delimiter=delimiter)
                if any(cell.strip() for cell in row)]
    except (csv.Error, ValueError):
        return None
    return ([cell.strip() for cell in rows[0]] if rows else []), max(0, len(rows) - 1)


def _json_shape(data):
    """(key paths, {list path: length}) of a JSON document, a list's items under one path; None when it does not
    parse or is too large to walk."""
    try:
        value = json.loads(data.decode('utf-8-sig', errors='replace'))
    except (ValueError, RecursionError):
        return None
    keys, lists, pending, walked = set(), {}, [(value, (), ())], 0
    while pending:
        node, path, place = pending.pop()
        walked += 1
        if walked > SHAPE_NODES:
            return None
        if isinstance(node, dict):
            for key, item in node.items():
                keys.add(path + (key,))
                pending.append((item, path + (key,), place + (key,)))
        elif isinstance(node, list):
            lists[place] = len(node)
            pending.extend((item, path + (None,), place + (index,)) for index, item in enumerate(node))
    return keys, lists


def _json_path(path):
    text = ''.join('[]' if part is None else f'[{part}]' if isinstance(part, int) else f'.{part}' for part in path)
    return text.lstrip('.')[:80] or 'the top level'


def shape_losses(before, after, limit=3):
    """What a fix took away from the outputs (two _lstat_tree trees; growth is allowed): a file that held content
    left empty, a CSV or TSV header column or data row lost, a JSON key path lost or a list shortened. At most
    limit descriptions; [] when nothing was lost."""
    losses = []
    for name, (kind, _, data) in sorted(before.items()):
        entry = after.get(name)
        if kind != 'file' or entry is None or entry[0] != 'file':  # a missing file is judged on its own
            continue
        new, shown = entry[2], name[-80:]
        if data.strip() and not new.strip():
            losses.append(f'{shown} is empty')
            continue
        suffix = name.rsplit('.', 1)[-1].lower() if '.' in name else ''
        if suffix in ('csv', 'tsv'):
            old, now = (_table_shape(text, '\t' if suffix == 'tsv' else ',') for text in (data, new))
            if old is None:
                continue
            if now is None:
                losses.append(f'{shown} no longer reads as a table')
                continue
            gone = [cell for cell in old[0] if cell and cell not in now[0]]
            if gone:
                losses.append(f'{shown} lost the column{"s" if len(gone) > 1 else ""} '
                              + ', '.join(cell[:40] for cell in gone[:4]))
            if now[1] < old[1]:
                losses.append(f'{shown} lost {old[1] - now[1]} of its {old[1]} rows')
        elif suffix == 'json':
            old, now = _json_shape(data), _json_shape(new)
            if old is None:
                continue
            if now is None:
                losses.append(f'{shown} no longer parses as JSON')
                continue
            gone = sorted(old[0] - now[0], key=lambda path: (len(path), str(path)))
            if gone:
                losses.append(f'{shown} lost the key {_json_path(gone[0])}'
                              + (f' and {len(gone) - 1} more' if len(gone) > 1 else ''))
            shorter = sorted(place for place, length in old[1].items()
                             if length and now[1].get(place, 0) < length)
            if shorter:
                place = min(shorter, key=len)
                losses.append(f'{shown}: the list at {_json_path(place)} went from {old[1][place]} to '
                              f'{now[1].get(place, 0)} items')
    return losses[:limit]


def _sweep(root: Path, before: set) -> None:
    """Remove what a finished program left in the shared scratch root (the 64 MiB /tmp) so that its
    leftovers cannot starve the rerun or the controller."""
    try:
        names = set(os.listdir(root)) - before
    except OSError:
        return
    for name in names:
        try:
            _remove(root / name)
        except (OSError, RecursionError):
            pass


# Imported by the generated program only (its PYTHONPATH): hardening, not the isolation boundary.
SANDBOX_SITE = """import ipaddress
import socket


def _local(address):
    if not isinstance(address, tuple) or not address:
        return True  # AF_UNIX paths and other local families
    host = address[0].decode('ascii', 'replace') if isinstance(address[0], bytes) else str(address[0])
    if host in ('', 'localhost'):
        return True
    try:
        return ipaddress.ip_address(host.split('%')[0]).is_loopback
    except ValueError:
        return False


def _guard(method, position):
    def guarded(self, *args):
        if len(args) > position and not _local(args[position if position >= 0 else len(args) - 1]):
            raise OSError('network disabled')
        return method(self, *args)
    return guarded


def _lookup(original):
    def getaddrinfo(host, *args, **kwargs):
        if host is not None and not _local((host,)):
            raise OSError('network disabled')
        return original(host, *args, **kwargs)
    return getaddrinfo


try:
    for _name, _position in (('connect', 0), ('connect_ex', 0), ('sendto', -1)):
        setattr(socket.socket, _name, _guard(getattr(socket.socket, _name), _position))
    socket.getaddrinfo = _lookup(socket.getaddrinfo)
except Exception:
    pass

try:  # on SIGUSR1 (sent just before a capped run is stopped) print where the program is, for the speed repair.
    # A handler on the main thread, not a watchdog thread reading another thread's frames, which can crash.
    import faulthandler as _faulthandler
    import os as _os
    import signal as _signal
    if _os.environ.get('T1_STACK_AT'):
        _faulthandler.register(_signal.SIGUSR1, all_threads=False)
except Exception:
    pass
"""


try:
    CRASH_SITE = Path(__file__).with_name('crashsite.py').read_text()
except OSError:  # the crash report only helps repair: a run goes on without it
    CRASH_SITE = ''
try:
    COMPAT_SITE = Path(__file__).with_name('compat.py').read_text()
except OSError:  # compatibility only saves repair rounds: a run goes on without it
    COMPAT_SITE = ''


# With the self-check on, each call self_check(...) runs as _t1_quiet(self_check)(...) (quiet_self_check): a check
# that raises, exits or runs past SELFCHECK_SECONDS returns None instead, so it can never turn a clean run into a
# crash or a timeout. The time limit is a SIGALRM timer (main thread only), restored afterwards; its exception
# derives from BaseException so that a check's own `except Exception` cannot swallow it.
SELFCHECK_SITE = """try:
    import builtins as _builtins

    class _T1CheckTimeout(BaseException):
        pass

    def _t1_quiet(function, seconds=%r):
        def run(args, kwargs):
            import signal
            import time
            armed, previous, earlier, started = False, None, (0.0, 0.0), time.monotonic()
            try:
                def expire(signum, frame):
                    raise _T1CheckTimeout()
                previous = signal.signal(signal.SIGALRM, expire)
                earlier = signal.setitimer(signal.ITIMER_REAL, seconds, 1.0)
                armed = True
            except (ValueError, OSError, AttributeError):
                pass  # not the main thread: the check runs without a limit
            try:
                return function(*args, **kwargs)
            except (Exception, SystemExit, _T1CheckTimeout):
                return None
            finally:
                if armed:
                    signal.setitimer(signal.ITIMER_REAL, 0)
                    signal.signal(signal.SIGALRM, previous)
                    if earlier[0] > 0:
                        signal.setitimer(signal.ITIMER_REAL, max(0.001, earlier[0] - (time.monotonic() - started)),
                                         earlier[1])

        def quiet(*args, **kwargs):
            try:
                return run(args, kwargs)
            except (Exception, SystemExit, _T1CheckTimeout):
                return None
        return quiet
    _builtins._t1_quiet = _t1_quiet
except Exception:
    pass
""" % SELFCHECK_SECONDS


def worker_site(coerce=False, *, date_guard=False) -> str:
    """The generated program's sitecustomize: the network guard, library compatibility, the self-check
    guard, then the crash-report hook; with coerce (T1_S13_COERCE's runs only), the coerce-loss prelude last."""
    site = SANDBOX_SITE + '\n\n' + COMPAT_SITE + '\n\n' + SELFCHECK_SITE + '\n\n' + CRASH_SITE
    if coerce:
        site += '\n\n' + COERCE_SITE
    if date_guard:
        site += '\n\n' + date_validation.worker_source()
    return site


def quiet_self_check(code):
    """The program as it runs with the self-check on: every call self_check(...) becomes
    _t1_quiet(self_check)(...). Only the call's own line changes, so line numbers in a crash report still
    match the model's program; code that does not parse, or has no such call, is returned as it is."""
    if SELFCHECK_NAME not in code:
        return code
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return code
    spots = sorted({(node.func.lineno, node.func.col_offset, node.func.end_col_offset) for node in ast.walk(tree)
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == SELFCHECK_NAME and node.func.lineno == node.func.end_lineno}, reverse=True)
    if not spots:
        return code
    lines = code.split('\n')
    for lineno, start, end in spots:  # offsets are UTF-8 bytes
        raw = lines[lineno - 1].encode()
        lines[lineno - 1] = (raw[:start] + f'_t1_quiet({SELFCHECK_NAME})'.encode() + raw[end:]).decode()
    quiet = '\n'.join(lines)
    try:
        ast.parse(quiet)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return code
    return quiet


class SelfChecks:
    """The program's self-check side file (T1_SELFCHECK=1), read after every run from the run's scratch
    directory: never graded, never in the status (which keeps numbers only). items: (quote, ok, detail) of the
    last run; written: whether that run wrote a readable list. keep: the output files the TASK names, and named:
    the TASK's text; a file named in either is never removed as a misplaced side file."""

    def __init__(self, keep=(), named=''):
        self.items, self.written, self.keep = [], False, {Path(name).name for name in keep}
        self.named = named.lower()

    def read(self, path: Path, canaries) -> None:
        self.items, self.written = [], False
        try:
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            with os.fdopen(fd, 'rb') as stream:
                if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                    return
                data = stream.read(SELFCHECK_LIMIT + 1)
        except OSError:
            return
        if len(data) > SELFCHECK_LIMIT:
            return
        try:
            parsed = json.loads(data.decode('utf-8', errors='replace'))
        except (ValueError, RecursionError):
            return
        if isinstance(parsed, dict):
            parsed = parsed.get('checks')
        if not isinstance(parsed, list):
            return
        self.written = True
        for entry in parsed[:SELFCHECK_ITEMS]:
            if not (isinstance(entry, dict) and isinstance(entry.get('quote'), str) and type(entry.get('ok')) is bool):
                continue
            detail = entry.get('detail') if isinstance(entry.get('detail'), str) else ''
            self.items.append((withhold_canaries(entry['quote'][:400], canaries), entry['ok'],
                               withhold_canaries(' '.join(detail.split())[:300], canaries)))

    def drop_strays(self, out: Path) -> int:
        """Remove output files that look like a misplaced side file, so that no extra file reaches the grader:
        a name with 'selfcheck' in it, or a JSON list of self-check items, that the TASK does not name (in its
        output contract or anywhere in its text). Returns how many."""
        removed = 0
        try:
            paths = sorted(out.rglob('*'))
        except OSError:
            return 0
        for path in paths:
            if path.name in self.keep or path.name == '_agent_status.json' or path.name.lower() in self.named:
                continue
            try:
                link = path.is_symlink()
                if not link and not path.is_file():
                    continue
                if 'selfcheck' in re.sub(r'[^a-z0-9]', '', path.name.lower()) or (not link and _selfcheck_shaped(path)):
                    path.unlink()
                    removed += 1
            except OSError:
                continue
        return removed


def _selfcheck_shaped(path: Path) -> bool:
    """Whether a small JSON file holds a self-check list: [{"quote": str, "ok": bool, ...}, ...]."""
    if path.suffix.lower() != '.json' or path.stat().st_size > SELFCHECK_LIMIT:
        return False
    try:
        data = json.loads(path.read_bytes().decode('utf-8', errors='replace'))
    except (ValueError, RecursionError, OSError):
        return False
    if isinstance(data, dict) and set(data) == {'checks'}:
        data = data['checks']
    return (isinstance(data, list) and bool(data) and all(
        isinstance(entry, dict) and isinstance(entry.get('quote'), str) and type(entry.get('ok')) is bool
        for entry in data))


def failed_checks(checks, instruction):
    """([(quote, detail)] of the last run's failed checks whose quote is the TASK's own words, at most
    SELFCHECK_FLAGS and one per quote; the number of failed checks whose quote is not in the TASK; the number
    left to the normalizer). A check of a literal format the normalizer applies (its quote is one of
    literal_rules' source lines) ran on the outputs before normalize_literals rewrote them, so its failure is
    stale: literal_flags reports what the rewrite could not fix."""
    if checks is None:
        return [], 0, 0
    try:
        literal = literal_rules(instruction)['quotes']
    except Exception:
        literal = []
    found, unquoted, owned = [], 0, 0
    for quote, ok, detail in checks.items:
        if ok:
            continue
        if not _quoted(quote, instruction):
            unquoted += 1
        elif any(_quoted(quote, line) for line in literal):
            owned += 1
        elif not any(same_quote(quote, other) for other, _ in found) and len(found) < SELFCHECK_FLAGS:
            found.append((quote.strip(' "\''), detail))
    return found, unquoted, owned


def check_flag(quote, detail):
    return f'the program\'s self-check finds "{quote}" does not hold' + (f': {detail}' if detail else '')


REPORT_LIMIT = 8192


def _read_report(path: Path, canaries) -> str:
    """The crash report the failed program's hook wrote, for the model only; never exported."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, 'rb') as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                return ''
            text = stream.read(REPORT_LIMIT).decode('utf-8', errors='replace')
    except OSError:
        return ''
    return withhold_canaries(text, canaries).strip()


def worker_environment(env=None, site=None) -> dict:
    env=os.environ if env is None else env
    clean={k:env[k] for k in ('PATH','LANG','LC_ALL','QFBENCH_SEED') if k in env}
    clean.update({'HOME':'/tmp','TMPDIR':'/tmp','MPLCONFIGDIR':'/tmp/mpl',
                  'PYTHONDONTWRITEBYTECODE':'1','PYTHONNOUSERSITE':'1','PYTHONHASHSEED':'0',
                  'PYTHON_CPU_COUNT':'16',  # the platform's CPU quota, not the host's core count
                  'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
    if site is not None:
        clean['PYTHONPATH']=str(site)
    return clean


def summarize_checker(output: Path, process: dict) -> dict:
    if process.get('timed_out'):
        return {'status':'checker_timeout','passed':False}
    if process.get('returncode') != 0:
        return {'status':'checker_error','passed':False,'reason':'checker_wrapper_nonzero'}
    try:
        reward=json.loads((output/'reward.json').read_text())
        report=json.loads((output/'pytest_report.json').read_text())
        if not isinstance(reward,dict) or not isinstance(report,dict): raise ValueError('Invalid report')
        summary=report['summary']; count=summary.get('passed',0)+summary.get('xpassed',0)
        if any(type(v) is not int or v<0 for v in summary.values()):
            raise ValueError('Invalid test counts')
        code=report.get('exitcode',reward.get('pytest_exit_code'))
        # Official scorer: 3/4 are harness faults, 1/2 belong to the submission (a missing or
        # malformed deliverable errors at collection), 0 passes with skips subtracted.
        if code in (1,2):
            return {'status':'failed','passed':False,'test_counts':summary}
        if (code==0 and count>0 and reward.get('reward')==1
                and not summary.get('failed',0) and not summary.get('error',0)):
            return {'status':'passed','passed':True,'test_counts':summary}
        return {'status':'checker_error','passed':False,'reason':'no_decisive_test_evidence','test_counts':summary}
    except (OSError,ValueError,KeyError,TypeError,AttributeError):
        return {'status':'checker_error','passed':False,'reason':'missing_or_invalid_reward_report'}


MAX_RUN_RECORDS = 8


def _run_program(code: str, out: Path, deadline: float, canaries, status: dict, clear: bool = False,
                 replay=_NO_REPLAY, input_root=None, cap=None, checks=None, health=None, coerce=None,
                 *, retain_success_output: bool = False, date_guard=None, optimize: bool = False) -> str:
    """Run one generated program in a fresh /tmp directory; return its sanitized output tail. With clear,
    the output tree is emptied only once the run is certain to start, so a setup failure keeps it.
    Every run appends numbers and enums to status['runs']. With checks (a SelfChecks: the self-check on), the
    program's self_check() calls cannot crash the run, its side file goes to the run's scratch directory and is
    read into checks, and a side file misplaced among the outputs is removed. With health (a FitWarnings), the
    fit non-convergence a run that exits 0 reports in its log is read into it. With retain_success_output,
    a successful run also returns its bounded, sanitized log tail as ungraded evidence. With optimize, the worker runs
    with PYTHONOPTIMIZE=1 (assert statements skipped; worker_environment passes nothing else through)."""
    if date_guard is not None:
        date_guard.clear()
        date_guard.update(date_validation.unverified('missing_report'))
    for key in ('exit_code', 'failure_kind', 'failure_code', 'execution_diagnostics', 'execution_resource'):
        status.pop(key, None)
    if runtime.active():
        for key in runtime.FIELDS:
            status.pop(key, None)
    with tempfile.TemporaryDirectory(prefix='t1-b2-',dir='/tmp',ignore_cleanup_errors=True) as tmp:
        script=Path(tmp)/'generated.py'; script.write_text(quiet_self_check(code) if checks is not None else code)
        site=Path(tmp)/'sandbox'; site.mkdir()
        (site/'sitecustomize.py').write_text(worker_site(coerce is not None, date_guard=date_guard is not None))
        crash=Path(tmp)/'crash_report.txt'
        env=dict(worker_environment(site=site),T1_CRASH_REPORT=str(crash),T1_PROGRAM=str(script))
        if input_fidelity_on():  # crashsite cuts each key its KeyError hint lists
            env['T1_KEY_HINT_CUT']='1'
        if optimize:  # thin's asserts-off rerun (B): the same program with its assert statements skipped
            env['PYTHONOPTIMIZE']='1'
        if coerce is not None:
            env['T1_COERCE_REPORT']=str(Path(tmp)/'coerce.jsonl')
        date_report = Path(tmp)/'date-validation.json'
        if date_guard is not None:
            env['T1_DATE_REPORT'] = str(date_report)
        side=Path(tmp)/'selfcheck.json'
        runtime_side=Path(tmp).resolve()/'runtime.json'
        if checks is not None and not side.resolve().is_relative_to(out.resolve()):
            env[SELFCHECK_ENV]=str(side)
        remaining=deadline-CLOCK.now()
        if remaining<=0:
            raise HouseError('No execution time remaining after generation',
                             kind='local_budget', code='execution_budget')
        capped=cap is not None and cap<remaining
        status.pop('run_capped',None)
        if capped:  # stopped early, with a stack sample just before, so a speed repair still fits
            remaining=cap; env['T1_STACK_AT']=str(max(1,round(cap-STACK_LEAD)))
        run_deadline=CLOCK.now()+remaining
        env.update(runtime.worker_fields(runtime_side, remaining))
        log=Path(tmp)/'execution.log'
        def limits():
            import resource
            resource.setrlimit(resource.RLIMIT_FSIZE,(64*1024*1024,64*1024*1024))
        root=Path(tmp).parent; before=set(os.listdir(root))
        with log.open('wb') as sink:
            if clear:
                _clear(out)
                # The clear took time. Never raise now: the measured outputs are gone, so a run that
                # has no time left ends as execution_timeout, which matches the emptied tree.
                remaining=run_deadline-CLOCK.now()
            began=CLOCK.now()
            if remaining <= 0:
                status.update(status='execution_timeout', failure_kind='execution', failure_code='program_timeout')
                if capped:
                    status['run_capped']=True
                runs=status.setdefault('runs',[])
                if len(runs)<MAX_RUN_RECORDS:
                    runs.append({'status':'execution_timeout', 'exit_code':None, 'exception_type':None,
                                 'generated_line':None, 'signature':None, 'crash_report':None, 'duration_sec':0.0})
                replay.run(log,status)
                return '', ''
            if clear:
                env.update(runtime.worker_fields(runtime_side, remaining))
            command = ([sys.executable,'-m','qfruntime.runner',str(script)] if runtime.active()
                       else [sys.executable,str(script)])
            resource_before = resource_control.snapshot() if minimal_on() or thin_on() else None
            resource_after = None
            process=subprocess.Popen(command,cwd=out,
                env=env,stdout=sink,stderr=subprocess.STDOUT,
                start_new_session=True,close_fds=True,preexec_fn=limits)
            try:
                sampled=False
                if capped:  # sample the stack STACK_LEAD s before the cap, then stop the run at the cap
                    lead=min(STACK_LEAD,remaining/2)
                    try:
                        result=process.wait(timeout=max(0.001,remaining-lead))
                    except subprocess.TimeoutExpired:
                        sampled=True
                        try: os.killpg(process.pid,signal.SIGUSR1)
                        except ProcessLookupError: pass  # it ended just now; its exit status stands
                        result=process.wait(timeout=max(0.001,run_deadline-CLOCK.now()))
                else:
                    result=process.wait(timeout=remaining)
                # A wait returning a signal is not a TimeoutExpired, even after a stack sample.
                # Preserve its real exit; only the controller timeout branch establishes a time cap.
                status.update(status='executed_ungraded' if result==0 else 'execution_failed',exit_code=result)
                if result != 0:
                    status.update(failure_kind='execution', failure_code='program_nonzero_exit')
            except subprocess.TimeoutExpired:
                try: os.killpg(process.pid,signal.SIGKILL)
                except ProcessLookupError: pass
                process.wait()
                status.update(status='execution_timeout', failure_kind='execution', failure_code='program_timeout')
                if capped:
                    status['run_capped']=True
            finally:
                if resource_before is not None:
                    resource_after = resource_control.snapshot()
                try: os.killpg(process.pid,signal.SIGKILL)
                except ProcessLookupError: pass
                _sweep(root,before)
        # No raw stdout, traceback text or custom exception names enter audit artifacts.
        elapsed=CLOCK.now()-began
        if resource_before is not None:
            status['execution_resource'] = resource_control.execution(
                resource_before, resource_after or {}, status.get('exit_code'),
                timed_out=status['status'] == 'execution_timeout')
        replay.run(log,status)  # local replay only; off on the platform
        if runtime.active():
            status.update(runtime.read_report(runtime_side, canaries))
            status.update(runtime.read_usage(runtime_side.with_name('runtime-usage.json')))
        if checks is not None:
            checks.read(side,canaries)
            _count(status,'selfcheck_stray_files',checks.drop_strays(out))
        if health is not None:
            health.read(log,canaries,status['status']=='executed_ungraded')
        diagnostic=execution_failure_summary(log,script,input_root)
        status['execution_log_bytes']=diagnostic.get('log_bytes')
        if status['status'] in ('execution_failed','execution_timeout'):
            status['execution_diagnostics']=diagnostic
        failed=status['status']=='execution_failed'
        report=_read_report(crash,canaries) if failed else ''
        if coerce is not None:
            coerce.extend(_s13_read_coerce(Path(tmp)/'coerce.jsonl'))
        if date_guard is not None:
            date_guard.clear()
            date_guard.update(date_validation.read_report(date_report))
        runs=status.setdefault('runs',[])
        if len(runs)<MAX_RUN_RECORDS:
            runs.append({'status':status['status'],'exit_code':status.get('exit_code'),
                         'exception_type':diagnostic['exception_type'] if failed else None,
                         'generated_line':diagnostic['generated_line'] if failed else None,
                         'signature':(diagnostic['signature'] or 'other') if failed else None,
                         'crash_report':bool(report) if failed else None,
                         'duration_sec':round(elapsed,3), **status.get('execution_resource', {})})
        keep_output = failed or status.get('run_capped') or (
            retain_success_output and status['status'] == 'executed_ungraded')
        return (_log_tail(log,canaries),report) if keep_output else ('','')


MEASURED = ('status', 'exit_code', 'failure_kind', 'failure_code', 'execution_diagnostics',
            'execution_log_bytes', 'execution_resource', 'program_sha256') + runtime.FIELDS
_EXECUTED = ('executed_ungraded', 'execution_failed', 'execution_timeout', 'output_limit_exceeded')
_BUILTIN_ERRORS = {name for name, value in vars(builtins).items()
                   if isinstance(value, type) and issubclass(value, BaseException)}


def _shape(code):
    """A program's identity for the rerun-cycle check: its syntax tree, so that a merge that only
    re-spaces statements or drops comments still matches a program that already ran."""
    try:
        return hashlib.sha256(ast.dump(ast.parse(code)).encode()).hexdigest()
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return hashlib.sha256(code.encode()).hexdigest()


def _profile(task_dir: Path, status: dict):
    """Input profile, canaries and /app hints, each best-effort: no input tree stops the first request. Under
    T1_S11_RAISE the profile shows each table's date columns' min and max (profile_date_ranges counts them)."""
    stats={}
    try:
        named=(task_dir/'instruction.md').read_text(encoding='utf-8',errors='replace')
    except OSError:
        named=''
    fidelity=input_fidelity_on()
    try:
        profile=(input_profile(task_dir,stats,named,ranges=s11_raise_on(),fidelity=True) if fidelity else
                 input_profile(task_dir,stats,named,ranges=s11_raise_on()))
    except Exception:
        profile, stats = {'files':[]}, {'errors':1}
    try:
        canaries=input_canaries(task_dir)
    except Exception:
        canaries=set(); stats['errors']=stats.get('errors',0)+1
    try:  # what a preview renders: a canary split over cells or members of a container is withheld too
        canaries|=near_canaries(json.dumps(preview_texts(profile),ensure_ascii=False))
    except Exception:
        stats['errors']=stats.get('errors',0)+1
    try:
        hints=input_hints(task_dir)
    except Exception:
        hints=''; stats['errors']=stats.get('errors',0)+1
    for key in ('files','errors','unlisted','symlinks','capped','previews'):
        status['profile_'+key]=stats.get(key,0)
    if 'date_ranges' in stats:
        status['profile_date_ranges']=stats['date_ranges']
    if fidelity:  # input_fidelity: the profile's data-trap lines and the workbook rows shown past SHEET_COLUMNS
        status['thin_input_traps']=stats.get('traps',0); status['thin_wide_rows']=stats.get('wide_rows',0)
    status['profile_status']=('error' if 'files' not in stats else
                              'partial' if any(stats.get(k) for k in ('errors','unlisted','symlinks','capped'))
                              else 'ok')
    return profile, canaries, hints


def _derive_counters(status: dict, records: list, refused_uncounted: bool=False):
    """Request counters from the tagged per-attempt records, so that no failure path can skip one.
    refused_uncounted (T1_EVAL_THROTTLE, local evaluation only): the refused 429s are left out of request_kinds,
    as they are out of model_calls, so a kind never counts more than the requests admitted."""
    first=Counter(r.get('kind') for r in records if r.get('attempt',0)==0)
    admitted=[r for r in records if r.get('http_status')!=429] if refused_uncounted else records
    status.update(continuations=first['continue_stmt']+first['continue_line'], rewrites=first['rewrite'],
                  fill_requests=first['fill'], salvages=first['salvage'], patch_retries=first['patch_retry'],
                  patch_requests=first['patch']+first['patch_continue']+first['patch_retry']+first['lint_retry'],
                  crash_reports=sum(bool(run.get('crash_report')) for run in status.get('runs', [])),
                  retries=sum(r.get('attempt',0)>0 for r in records),
                  request_kinds=dict(sorted(Counter(r['kind'] for r in admitted
                                                    if isinstance(r.get('kind'),str)).items())))
    thin_records = [r for r in records if r.get('kind') in ('thin_contract', 'thin_code', 'thin_continue')]
    if thin_records:
        status.update(thin_contract_requests=sum(r['kind'] == 'thin_contract' for r in thin_records),
                      thin_code_requests=sum(r['kind'] == 'thin_code' for r in thin_records),
                      thin_continue_requests=sum(r['kind'] == 'thin_continue' for r in thin_records),
                      generation_replies=sum(r['kind'] == 'thin_code' and r.get('reply_returned') is True
                                             for r in thin_records))


def consistency_note(instruction):
    """A note naming symbols the task defines by different formulas, so that one is chosen deliberately
    rather than whichever the plan happens to copy; empty when there is none."""
    conflicts, mappings = conflicting_definitions(instruction), incomplete_mappings(instruction)
    parts = ['The TASK defines {} twice, differently ({}). Decide which definition is consistent with the rest '
             'of the TASK by checking units and magnitudes, state the choice in a code comment, and use only that '
             'one.'.format(name, '; '.join(f'line {number}: {formula}' for number, formula in definitions[:3]))
             for name, definitions in conflicts[:3]]
    parts += ['The TASK maps labels to values on line {} ({}), but it also names {} (e.g. line {}). Wherever a '
              'step uses a label, derive its value by the same rule as the mapping instead of assuming the '
              'list covers every label.'.format(number, ', '.join(mapped), ', '.join(label for label, _ in missing[:8]),
                                                 missing[0][1])
              for number, mapped, missing in mappings[:2]]
    parts += ['The TASK gives worked examples of a rule on line {} ({}). Implement the rule so that it reproduces '
              'every example exactly (for instance with string operations rather than arithmetic), apply it to '
              'every value it covers, and assert those examples at the start of the program.'.format(
                  number, ', '.join(f'{a} becomes {b}' for a, b in pairs[:4]))
              for number, pairs in worked_examples(instruction)[:2]]
    return 'TASK CONSISTENCY NOTE: ' + ' '.join(parts) + '\n\n' if parts else ''


LIBRARY = 'qfkit'  # installed in the image's site-packages, which the generated program's environment keeps
LIBRARY_BLOCK = ('OPTIONAL LIBRARY: `import qfkit` works in this program. It is a vetted library of standard '
                 'quant-finance formulas built on numpy, scipy and pandas, and using it is optional. Where a step is '
                 'a standard formula that a function below computes, calling it is safer than re-deriving the '
                 'formula by hand; pass the TASK\'s conventions (units, signs, compounding, day count, confidence '
                 'level, lag rule) as its arguments. Where the TASK defines a quantity, formula or convention '
                 'differently from a function, the TASK wins: write that step yourself. Catalog:\n{catalog}\n\n')


# T1_ADOPT: the same block with the library used for every standard method a function computes; a step is written by
# hand only where the TASK writes out a different formula or convention.
LIBRARY_BLOCK_ADOPT = ('LIBRARY: `import qfkit` works in this program. It is a vetted library of standard '
                       'quant-finance formulas built on numpy, scipy and pandas. Use it for every standard method a '
                       'function below computes (a named pricing model or approximation, estimator, statistical test, '
                       'numerical scheme or optimisation): call that function rather than re-deriving its formula by '
                       'hand, and pass the TASK\'s conventions (units, signs, compounding, day count, confidence '
                       'level, lag rule) as its arguments. Write a step by hand only where the TASK writes out a '
                       'different formula or convention for it than the function computes. Catalog:\n{catalog}\n\n')


LIBRARY_FIX = ('The TASK names methods that LIBRARY MATCHES above lists and this program does not call ({names}): '
               'where {what} a hand-written version of one of them, replacing it with a call of the library function '
               'is the preferred fix. ')
LIBRARY_FIX_ADOPT = ('This program does not call library functions that LIBRARY MATCHES or REQUIRED LIBRARY CALLS '
                     'above list ({names}): where {what} a hand-written version of one of them, replacing it with a '
                     'call of the library function is the preferred fix. ')
LIBRARY_CALLS = ('Call each function as `qfkit.<name>`, exactly as the catalog lists it: the library exports only '
                 'these names (no numpy or scipy names, and no function under a submodule path). ')
# The catalog functions that take numpy arrays for their numeric arguments and return one result per element (per row
# for copula_logpdf, copula_cdf, realized_variance, bipower_variation and noise_variance, one row a day): one call on
# arrays equals the calls on each element. Measured on qfkit 0.3.0 and 0.4.0 with numpy 2.4 (tests/test_x12_vector.py
# repeats the check where numpy is installed); not bond_risk, bond_price_curve (T), newey_west_lags,
# poisson_truncation or fd_solve (an array of spots on one grid, but one strike, maturity and volatility). The
# functions that take a whole series or frame and return a table, ledger or fit (the backtest ledgers, event_study,
# realized_measures, var_es_forecast, rolling_var_backtest, ou_fit, ou_simulate ...) are not listed: they are called
# once per series by design, and one call per window or path of a loop the TASK defines is their intended use.
LIBRARY_VECTORISED = frozenset({
    'asian_curran_price', 'asian_geometric_price', 'asian_levy_price', 'asian_tw_price', 'barrier_price',
    'baw_critical_price', 'baw_price', 'bipower_variation', 'black76_implied_vol', 'black76_price', 'bond_price',
    'bond_ytm', 'bs2002_price', 'bsm_bounds', 'bsm_d1d2', 'bsm_escrowed_dividend', 'bsm_greeks', 'bsm_pde_residual',
    'bsm_price', 'chooser_price', 'cir_bond_price', 'cir_logpdf', 'cir_moments', 'convert_rate', 'copula_cdf',
    'copula_logpdf', 'copula_tail_dependence', 'copula_tau', 'copula_theta', 'crr_price', 'df_from_rate',
    'digital_price', 'fd_cn_price', 'feller_ratio', 'forward_rate', 'forward_start_price', 'geske_compound_price',
    'hazard_from_spread', 'heston_price', 'hw_bond_option', 'hw_bond_price', 'hw_caplet_price', 'hw_swaption_price',
    'implied_vol', 'iv_brenner_subrahmanyam', 'iv_corrado_miller', 'kirk_spread_price', 'kupiec_pof',
    'lookback_fixed_price', 'lookback_floating_price', 'margrabe_price', 'mean_excess', 'merton_implied_asset',
    'merton_logpdf', 'merton_price', 'merton_structural', 'noise_corrected_rv', 'noise_variance', 'ns_yield',
    'ou_jump_moments', 'ou_moments', 'par_yield', 'parity_residual', 'pot_var_es', 'rate_from_df',
    'realized_variance', 'richardson', 'sabr_implied_vol', 'survival_probability', 'var_cornish_fisher',
    'var_es_normal', 'var_es_t', 'vasicek_bond_option', 'vasicek_bond_price', 'vasicek_logpdf', 'year_fraction'})
LIBRARY_VECTOR = ('Most catalog formulas broadcast over numpy arrays (functions marked [vectorised] below do): call '
                  'such a function once on whole arrays (every observation, strike or date together), not once per '
                  'element in a Python loop, which can run past the time limit. ')
LIBRARY_VECTOR_MARK = ' [vectorised]'
LIBRARY_MATCHES = ('LIBRARY MATCHES: the TASK names these methods, and these catalog functions compute them '
                   '(matched by name, so check that each is the method the TASK means). Unless the TASK writes out '
                   'its own formula or convention for one, call the function rather than deriving it by hand:\n'
                   '{matches}\n\n')
LIBRARY_MATCHES_ADOPT = ('LIBRARY MATCHES: the TASK names these methods, and these catalog functions compute them '
                         '(matched by name, so check that each is the method the TASK means). Compute each method the '
                         'TASK means by calling its function, unless the TASK writes out a different formula or '
                         'convention for it:\n{matches}\n\n')
ADOPT_REQUIRED = ('REQUIRED LIBRARY CALLS (from the LIBRARY MAP of your plan): the program must call each of these '
                  'functions to compute the quantity it is listed for, with the TASK\'s conventions as arguments, '
                  'instead of deriving that quantity by hand: {calls}.\n\n')
ADOPT_LIMIT = 8  # required calls at most: a longer map lists the catalog rather than the methods the outputs need
ADOPT_METHOD_CHARS = 80
# T1_CARDS: a usage card per function the program must call or whose mid-level method the TASK names, read from the
# library's docstrings. A catalog line shows neither the values a keyword accepts nor the keys of the result, and
# most library misuse was a result read by another package's names or a keyword value outside the accepted set.
CARDS_BLOCK = ('LIBRARY CARDS: how to call the library functions listed above, from the library\'s own '
               'documentation: the signature, each keyword argument with the values it accepts, the keys of the '
               'result (Returns) and an example whose values are illustrative, not the TASK\'s:\n{cards}\n\n')
CARDS_REQUIRED = (' Set every keyword argument of these functions from the TASK, and read each result by the keys '
                  'its LIBRARY CARD below documents under Returns.')
CARDS_LIMIT = 8
CARDS_CHARS = 7500  # all cards of a prompt, about 2,500 tokens
CARD_CHARS = 1000  # one card; the example is trimmed first
CARD_ARG_CHARS = 160  # a keyword's entry on a card, the values it accepts aside
CARD_ARG_STEP = 10
# A card over CARD_CHARS without its example, level by level: the characters of each line of a Returns list that
# names a key (0: the whole paragraph), the fewest characters of a required keyword's entry tried with them (from
# CARD_ARG_CHARS down by CARD_ARG_STEP), and those of an optional keyword's (None: left out, as the signature shows
# its default).
CARD_LEVELS = ((0, 100, 60), (0, 100, None), (80, 80, None), (60, 60, None), (45, 30, None))
# T1_S11_LIBNOTES (with T1_S10): LIBRARY NOTES after LIBRARY MATCHES, in the task prompt that DECIDE, CHECK, STATE and
# the program read, one item per matched function from the library's own source (as LIBRARY CARDS): its whole
# signature, a vectorised function's own example call, each required keyword's entry, the clause that makes an
# optional keyword required in some case, the values each convention keyword accepts, and the first Returns
# paragraph (keys and units). A catalog line shows none of these, and the recorded misuse was exactly that: a
# log-density called once per scalar inside an optimiser, a family that needs a keyword the line leaves out, a
# volatility key read as a variance, a convention keyword set against the TASK's own sum of pieces.
S11_LIBNOTES = ('LIBRARY NOTES (from the library\'s own documentation; the TASK always wins). Pass every convention '
                'keyword explicitly, with the value the TASK states; where the TASK is silent, the value whose '
                'documented meaning fits how the TASK defines the quantity (a result the TASK defines as a sum of '
                'pieces must equal that sum). Read results only by their documented keys, in their documented units. '
                'An expiry, maturity or date argument is years from today, not a period length; a step argument is '
                'the time between observations; an entry\'s own words win. Call a [vectorised] function once on whole '
                'arrays (a log-density on all observations, then .sum()), never once per scalar in a loop or an '
                'optimiser\'s objective. Example values are illustrative.\n{items}\n\n')
S11_LIBNOTES_CHARS = 2000  # the items, about 550 tokens; the rules above them add about 730 characters
# An item over its share, level by level: the characters of a required keyword's entry (0: the values it accepts
# only), of the clause that makes an optional keyword required, whether an optional keyword shows the values it
# accepts, the characters of each Returns line that names a key (0: the whole paragraph, at most S11_RETURNS_CHARS),
# and whether a vectorised function shows its example call (the array call is what the catalog line hides).
S11_NOTE_LEVELS = ((160, 100, True, 0, True), (100, 100, True, 100, True), (60, 80, False, 80, True),
                   (0, 80, False, 60, True), (0, 60, False, 45, True), (0, 60, False, 45, False))
S11_EXAMPLE_CHARS = 200  # a longer example call is not shown
S11_RETURNS_CHARS = 600
# library_notes: the level an item's room is first capped at, its core (S11_NOTE_LEVELS[1]: the Returns paragraph cut
# to the lines that name its keys), so one item's Returns prose never costs another's required entries.
S11_CORE_LEVEL = 1
# Before the first value list of an entry whose quoted words are not the argument's own string values: another type
# (a dict's keys, a tuple's, a list's or a permutation's items, an array's or a number's string alternative). A line
# of those words alone would present keys as accepted values, so the entry is shown instead: S11_ENTRY_CHARS, or
# through its first list when that ends within S11_ENTRY_MAX, else through the word that names the type.
S11_NOT_CHOICE = re.compile(r'\b(?:dict|mapping|tuple|array|list|sequence|callable|permutation|pairs|bool|int|integer|'
                            r'float|Series|DataFrame)s?\b')
S11_ENTRY_CHARS = 60
S11_ENTRY_MAX = 120
# A sign or a time unit an entry states (signed, loss, years, time per step or between observations, per period, to
# expiry, horizon): a required keyword whose note line would be its name alone keeps S11_MEANING_CHARS of its entry
# around it, and a clipped entry keeps it in view; the name alone left the header's time rule or a guess to decide it.
S11_MEANING = re.compile(r'\b(?:signed|sign|loss|losses|years?|time (?:per|between)|step length|per (?:period|'
                         r'observation)|to (?:expiry|maturity)|from (?:today|now)|horizon)\b', re.I)
S11_MEANING_CHARS = 60
# T1_S11_LIBNOTES: a TypeError or ValueError raised inside a library function (or by the signature of its call) that
# an earlier run of the repair loop raised too gets the function's documentation in the next patch ask; the LIBRARY
# USE note alone left patches guessing spellings in a loop.
S11_LIBDOC = ('LIBRARY DOCUMENTATION: this {exc} comes from `{lib}.{function}` again, as before your last change. Its '
              'documentation from the library\'s source follows; call it exactly as documented, with the TASK\'s '
              'values (every required argument, each value from the accepted set, results read by their documented '
              'keys):\n```\n{doc}\n```\n')
S11_LIBDOC_CHARS = 2500
S11_LIBDOC_SECTIONS = ('Example', 'Parameters', 'Returns')  # over S11_LIBDOC_CHARS: the title, then these in order
S11_LIBDOC_CUT = '[... rest of the documentation cut]'
S11_PROMPT_SHA = hashlib.sha256(json.dumps([
    S11_LIBNOTES, S11_LIBNOTES_CHARS, S11_NOTE_LEVELS, S11_EXAMPLE_CHARS, S11_RETURNS_CHARS, S11_LIBDOC,
    S11_LIBDOC_CHARS, S11_LIBDOC_SECTIONS, S11_LIBDOC_CUT, S11_CORE_LEVEL, S11_NOT_CHOICE.pattern, S11_ENTRY_CHARS,
    S11_ENTRY_MAX, S11_MEANING.pattern, S11_MEANING_CHARS], sort_keys=True).encode()).hexdigest()
# qfkit 0.4.0's modules: engines over a whole series, grid or event set with their conventions as 1 to 16 keyword
# arguments, which one catalog line cannot carry.
LIBRARY_MID_LEVEL = frozenset({'hedging', 'pde', 'realized', 'ou', 'ledger', 'events', 'varbt'})
LIBRARY_MATCH_LIMIT = 8
# Words of a function name that say what it computes, not which method; with them, acronyms any regression or PDE
# task uses and ratio statistics a TASK nearly always defines itself. A function's name is matched on the rest.
LIBRARY_GENERIC = frozenset({
    'annuity', 'asset', 'bet', 'black', 'bond', 'bootstrap', 'bounds', 'calmar', 'caplet', 'cashflows', 'compound',
    'contributions', 'convert', 'counts', 'cov', 'curve', 'drawdown', 'euler', 'excess', 'feller', 'fit', 'fixed',
    'floating', 'forecast', 'forward', 'fraction', 'from', 'gbm', 'generator', 'greeks', 'half', 'hazard',
    'historical', 'implied', 'kelly', 'lags', 'life', 'local', 'logpdf', 'max', 'mean', 'migration', 'min',
    'minimize', 'moments', 'multistart', 'ols', 'option', 'par', 'parity', 'paths', 'pde', 'period', 'poisson',
    'price', 'prices', 'probability', 'range', 'rate', 'rates', 'ratio', 'realized', 'residual', 'return', 'returns',
    'risk', 'rolling', 'sharpe', 'solve', 'sortino', 'spread', 'squared', 'start', 'structural', 'survival', 'swap',
    'swaption', 'tangency', 'target', 'test', 'truncation', 'value', 'variance', 'vol', 'weights', 'year', 'yield',
    'zero'})
# Words of a function name for the quantity it returns, or the step it takes, under any model (solve for a solver,
# ledger for a backtest's cash and position records, which a TASK calls a backtest, trade log or portfolio value).
# Every other word of its name that its own docstring title spells out qualifies a match and must occur in the TASK
# too, as the title spells it (jump in ou_jump_moments, covariance for cov in ewma_cov, asset in
# merton_implied_asset), unless the term found is that word.
LIBRARY_OUTPUT = frozenset({'fit', 'ledger', 'logpdf', 'moments', 'paths', 'price', 'prices', 'solve', 'start',
                            'value', 'weights'})
# Model families one author's name spans (the Vasicek short rate and the Vasicek credit factor, the Merton jump
# diffusion and the Merton structural credit model). A function whose title names a family is listed only when the
# TASK names it too; a function whose title names none takes the families of the functions of its module that share
# its first name word (cir_logpdf those of cir_moments and cir_bond_price).
LIBRARY_FAMILIES = (
    ('jump', r'jumps?|jump[- ]diffusions?'),
    ('credit', r'credit|defaults?'),
    ('short rate', r'short[- ]rates?|zero[- ]coupon'),
    ('mean reversion', r'Ornstein[- ]Uhlenbeck|OU|mean[- ]revert\w*|mean[- ]reversion'),
)
# Numerical schemes a function's title can name. A TASK clause that names a numpy or scipy routine together with one
# of these, or with the term that found the function, prescribes its own implementation of that step (a Dupire grid
# by finite differences where the TASK pins numpy.gradient); a TASK clause that rules one out (_RULED_OUT: rather
# than closed-form formulae) excludes it. Either way the function is not listed.
LIBRARY_SCHEMES = (r'finite[- ]differen\w*', r'maximum[- ]likelihood|MLE', r'least[- ]squares', r'Monte[- ]Carlo',
                   r'Fourier', r'analytic\w*|closed[- ]forms?')
# Standard names of the methods some functions compute that their docstring titles do not spell, matched as method
# names (a space stands for any run of spaces or dashes; case is ignored).
LIBRARY_ALIASES = {
    'black76_price': (r"Black's (?:model|formula)", r'Black (?:76|\(1976\))'),
    'bootstrap_zero_curve': (r'bootstrap\w* (?:an? |the )?zero(?: coupon)? (?:yield |discount |spot )?curve',
                             r'zero(?: coupon)? (?:yield |discount |spot )?curve bootstrap\w*'),
    'copula_logpdf': (r'copula (?:log )?(?:likelihood|density)',),
    'fit_ns': ('Svensson',),  # the model it fits, so that a TASK pinning the fit's routine pins this function
    'kelly_bet': ('Kelly criterion', r'Kelly (?:fractions?|weights|portfolio)'),
    'kelly_weights': ('Kelly criterion', r'Kelly (?:fractions?|weights|portfolio)'),
    'ns_yield': ('Svensson',),
    'parity_residual': ('put call parity',),
    'realized_variance': (r'reali[sz]ed variance',),
    'risk_contributions': (r'Euler (?:risk )?(?:allocation|decomposition|attribution)(?: \S+){0,6} volatility',),
    # The single-asset signal backtest and its trailing stop; the backtest of target weights with an execution lag,
    # drift between rebalances and turnover costs; calendar-time sampling of intraday prices.
    'signal_trade_ledger': ('trailing stop', r'long only (?:\S+ ){0,2}backtest\w*'),
    'subsample_grid': ('previous tick', 'calendar time sampling', r'sampling grids?'),
    'var_es_historical': ('historical simulation',
                          r'historical(?: \S+){0,3} (?:VaR|value at risk|ES|expected shortfall)'),
    'var_es_normal': (r'parametric(?: Gaussian| normal)? (?:VaR|value at risk)',),
    'weights_ledger': ('execution lag', r'turnover costs?', r'(?:weights?|holdings?) drift\w*',
                       r'drift\w* (?:portfolio )?weights?', r'between rebalanc\w*',
                       r'rebalanc\w*,? (?:\S+ ){0,3}backtest\w*', r'backtest\w* (?:\S+ ){0,3}rebalanc\w*'),
}
_TITLE_HYPHENATED = re.compile(r'\b[A-Z][A-Za-z]*(?:-(?:[A-Z][A-Za-z]*|\d+))+')
_TITLE_ACRONYM = re.compile(r'\b([A-Z]{3,})(\(\d+(?:,\s*\d+)*\))?')
_TITLE_NAME = re.compile(r'\b[A-Z][a-z]{2,}\b')
# A capitalised word of four letters or more (Jamshidian, RiskMetrics) or an acronym (PSOR) in a title: an author's
# or method's name unless LIBRARY_GENERIC or LIBRARY_COMMON_NAMES has it, it is a word of the function's own name, it
# is part of a hyphenated word, or it opens the title or a sentence (Exact, Simple, Random) without a year or another
# name after it (Kupiec (1995) ..., Kupiec and Christoffersen ... still count).
_TITLE_PROPER = re.compile(r'\b(?:[A-Z][a-z]{3,}(?:[A-Z][a-z]+)*|[A-Z]{3,})\b')
_TITLE_CITED = re.compile(r'\s*\(?(?:1[89]|20)\d\d\b|\s*(?:,|/|&|\band\b)\s*[A-Z]')
_TITLE_COMPOUND = re.compile(r'\b[A-Za-z][a-z]+(?:-[a-z]+){1,3}\b')  # peaks-over-threshold, for its initials
_JOIN = r'(?:\s*(?:[-\u2010-\u2015/&]|,(?:\s*and\b)?|\band\b)\s*|\s+)'  # between the parts of a hyphenated name
_EDGE = r'[A-Za-z0-9\-\u2010-\u2015]'
_DASHES = r'[\s\-\u2010-\u2015]+'
_TEXT_WORD = re.compile(r'[A-Za-z0-9]+')
# A numpy or scipy routine named in a TASK clause, other than elementwise arithmetic and a distribution's cdf, pdf
# or quantile (which every closed form uses and which pin no method).
_ROUTINE = re.compile(r'(?<![\w.])(?!(?:np|numpy)\.(?:exp|log\w*|sqrt|abs|sum|mean|std|var|max|min|maximum|minimum|'
                      r'where|array|asarray|zeros|ones|arange|linspace|clip|round|isnan|isfinite|nan\w*|cumsum|'
                      r'cumprod|diff|dot|sign|floor|ceil|power|square|pi|e|inf|nan)\b)'
                      r'(?!scipy\.stats\.\w+\.(?:cdf|pdf|ppf|sf|isf|logpdf|logcdf|logsf)\b)'
                      r'(?:np|numpy|scipy|statsmodels|sklearn|arch)\.[A-Za-z_]\w*')
_RULED_OUT = re.compile(r"\b(?:(?:rather than|instead of|without|avoid) using|(?:do|does|must|should|may) not use|"
                        r"don't use|never use)\b", re.I)
_CLAUSE = re.compile(r'\n\s*\n|\n\s*(?=[-*+|#]|\d+[.)]\s)|(?<=[.;:!?])\s+|,\s+')
_REFERENCES = re.compile(r'^(#+)\s*(?:references|bibliography|further reading|sources|citations)\b.*?(?=^#+\s|\Z)',
                         re.I | re.M | re.S)
LIBRARY_PHRASE_WINDOW = 4  # words within which the name words of a function must all occur
# Capitalised words of docstring titles that name no method: option styles, distributions and copula families (a
# function's argument, not its method), processes, and words a title capitalises for emphasis or because it opens
# with them (qfkit 0.4.0: Signal backtest ..., Event study ..., Noise-corrected ...). Neither a cited name nor, as a
# word of the function's own name, a 'name' term.
LIBRARY_COMMON_NAMES = frozenset({
    'american', 'asian', 'bermudan', 'brownian', 'carlo', 'clayton', 'european', 'event', 'expected', 'fourier',
    'frank', 'gaussian', 'greeks', 'gumbel', 'markov', 'monte', 'noise', 'normal', 'pareto', 'poisson', 'same',
    'shortfall', 'signal', 'student', 'white'})
LIBRARY_PARTIAL_WINDOW = 6  # words around a partial name within which its qualifiers must all occur
# T1_S12_LIB: qfkit 0.5.0 (qfkit2_prereg_amend_4.md), seven modules ported from the team's qflib and appended to the
# 0.4.0 CATALOG. Off, the controller offers 0.4.0 exactly: library_catalog cuts the CATALOG at the length the
# library's __init__ states in this literal, and the functions past it are hidden from the matcher, the notes and
# cards, the library-use notes and the reference checks (s12_hidden); a program can still import them by name.
S12_LIB_BASE = 'CATALOG_040_LEN'
# Standard names of the 0.5.0 methods (as LIBRARY_ALIASES, whose 0.4.0 entries stay as they are), matched only for
# functions the CATALOG lists, so only under T1_S12_LIB. Phrases written after one task's wording rather than the
# method's standard or library name were dropped for R3 (decision 10-23) where no public task's prompt changes.
S12_LIB_ALIASES = {
    'dcc_filter': (r'DCC', r'dynamic conditional correlation'),
    'dcc_fit': (r'DCC', r'dynamic conditional correlation'),
    'stationary_bootstrap_indices': (r'stationary (?:block )?bootstrap', r'geometric block'),
    'ipca_fit': (r'instrumented (?:principal components?|PCA)', r'IPCA'),
    'dependent_sorts': (r'dependent (?:double )?sorts?', r'double sorts?'),
    'neutralize_weights': (r'(?:factor|beta|PCA|PC) neutral', r'neutrali[sz]ation'),
    'turnover_cap': (r'turnover cap', r'turnover limit'),
    'bootstrap_projection_curve': (r'projection curve', r'dual curve bootstrap'),
    'dual_curve_swap': (r'dual curve', r'multi curve'),
    'mtm_principal_cashflows': (r'mark to market cross currency', r'MTM XCCY'),
    'cashflow_ytm': (r'yield to maturity',),
    'z_spread': (r'z spread',),
    'key_rate_risk': (r'key rate (?:duration|risk|sensitivit)',),
    'quote_key_rate_risk': (r'quote (?:space|bump)',),
    'fit_parametric_curve': (r'Nelson Siegel', r'Svensson'),
    'arch_garch_fit': (r'arch_model', r'arch library', r'GARCH'),
    'volume_share_profile': (r'volume (?:share )?profiles?',),
    'largest_remainder': (r'largest (?:fractional )?remainders?', r'Hamilton allocation'),
    'finite_difference_risk': (r'central (?:finite )?differences?', r'bump and reprice',
                               r'finite difference Greeks?'),
    'lag_correlations': (r'lead lag', r'lagged correlations?'),
    'hw_trinomial_tree': (r'Hull White(?: \S+){0,6}trinomial', r'Arrow Debreu'),
    'trinomial_rollback': (r'Bermudan', r'optimal stopping'),
    'iv_newton': (r'Newton Raphson',),
    'iv_li_atm': (r'Li at the money',),

    'vol_target_weights': (r'volatility target\w*', r'vol target\w*'),
    'synthetic_forward_bounds': (r'synthetic forward',),
    'multifactor_heston_price': (r'(?:two|multi) factor (?:Heston|stochastic volatility)',),
    'variance_replication': (r'variance swap', r'log contract replication'),
    'weighted_var_es': (r'age weighted historical', r'weighted (?:empirical )?quantile'),
    'gpd_observed_information': (r'observed Fisher information',),
    'normal_mixture_var_es': (r'(?:normal|Gaussian) mixture', r'contaminated normal'),
    'kde_var_es': (r'kernel density estimation', r'Gaussian KDE'),
    'barrier_digital_price': (r'barrier (?:digital|binary)',),
    'gap_option_price': (r'gap option',),
    'brinson_fachler': (r'Brinson Fachler',),
    'brownian_hit_cdf': (r'first passage (?:time )?(?:CDF|distribution|probabilit\w*)',),
    'brownian_hit_pdf': (r'first passage(?: \S+){0,5} density',),
    'brownian_joint_cdf': (r'joint distribution(?: \S+){0,8}running maximum',),
    'gbm_greek_samples': (r'pathwise', r'likelihood ratio'),
    'gbm_fd_greek_samples': (r'finite difference Greeks?',),
    'dupire_gradient_surface': (r'Dupire(?: \S+){0,10}gradient',),
    'localvol_paths': (r'local vol(?:atility)? (?:Monte Carlo|simulation|paths?)',),
    'project_call_slice': (r'convex(?:ity)?(?: \S+){0,5}call(?: price)? (?:slice|surface)',),
    'gaussian_hmm_fit': (r'hidden Markov models?', r'HMMs?'),
    'hmm_posterior': (r'forward backward (?:algorithm|recursion|pass)\w*', 'Viterbi'),
    'ema': (r'exponential moving averages?(?: \(?EMAs?\)?)?', r'EMAs?'),
    'cs_zscore': (r'cross sectional(?:ly)? (?:z scores?|standardi[sz]\w*)',),
    # cs_winsorize clips each date's row: a winsorization the TASK calls cross-sectional, not one per column.
    'cs_winsorize': (r'cross section(?:al(?:ly)?)? (?:\S+ ){0,8}winsori[sz]\w*',
                     r'winsori[sz]\w* (?:\S+ ){0,8}cross section(?:al(?:ly)?)?'),
    'quantile_buckets': (r'(?:quintile|decile|tercile|quartile) (?:sorts?|portfolios?|buckets?)', r'portfolio sorts?'),
    'information_coefficient': (r'information coefficients?', r'(?:information coefficient|Spearman|rank) ICs?'),
    'absorption_ratio': (r'absorption ratios?',),
    'marchenko_pastur_edge': (r'Marchenko Pastur',),
    'migration_thresholds': ('CreditMetrics', r'(?:rating|migration|transition) thresholds?'),
    'simulate_migrations': ('CreditMetrics', r'correlated rating migrations?'),
    'latent_factor_defaults': (r'(?:Gaussian|Student t|t) copulas? (?:\S+ ){0,3}defaults?',
                               r'(?:one|single|multi) factor (?:Gaussian )?(?:copula|default|Merton) model'),
    'vasicek_loss_quantile': ('ASRF', r'asymptotic single risk factor', r'Basel (?:II |III )?IRB'),
    'vasicek_loss_cdf': (r'large homogeneous pool', r'Vasicek (?:loss )?distribution'),
    'es_contributions': (r'Euler (?:allocation|decomposition|contributions?)(?: \S+){0,6} '
                         r'(?:ES|CVaR|expected shortfall)', r'marginal (?:CVaR|ES|expected shortfall) contributions?'),
    'pseudo_observations': (r'pseudo observations?',),
    'copula_fit': (r'fit\w* (?:\S+ ){0,4}copulas?', r'copula (?:fit\w*|estimation|calibration)'),
    'information_criteria': ('Akaike', r'AIC and BIC', r'information criteri(?:on|a)'),
    'empirical_tail_dependence': (r'tail dependence coefficients?', r'empirical (?:upper |lower )?tail dependence'),
    'discretise_severity': (r'(?:discretise|discretize|discretised|discretized|discretising|discretizing) '
                            r'(?:\S+ ){0,2}severity', r'severity discreti[sz]ation'),
    'panjer_recursion': (r"Panjer(?:'s)? (?:recursion|recursive|algorithm|method)",),
    'compound_fft': (r'(?:FFT|fast Fourier)(?: \S+){0,4} (?:compound|aggregate)', r'aggregate loss distributions?'),
    'svi_total_variance': ('SVI', r'stochastic volatility inspired'),
    'svi_calibrate': ('SVI', r'stochastic volatility inspired'),
    'gatheral_local_vol': ('Dupire',),
}
# 0.5.0's functions that take numpy arrays elementwise (as LIBRARY_VECTORISED; tests/test_x12_vector checks them
# against the library under T1_S12_LIB), marked only under T1_S12_LIB.
S12_LIB_VECTORISED = frozenset({'gatheral_local_vol', 'svi_total_variance', 'vasicek_loss_quantile',
                              'brownian_hit_cdf', 'brownian_hit_pdf', 'brownian_joint_cdf'})
# New methods whose abbreviations or generic words also describe unrelated calculations.
S12_LIB_GATES = {
    'accrued_interest': r'\b(?:coupon|bond|dirty|clean)\b',
    'deterministic_call_price_cap': r'callable[- ]bond|call[- ]date|call[- ]redemption',
    'arch_garch_fit': r'\barch\b|arch_model',
    'rolling_garch_var': r'\brolling\b',
    'largest_remainder': r'largest(?:[- ]fractional)?[- ]remainders?|Hamilton\s+allocation',
    'participation_fills': r'\binteger\s+(?:share|quantit)|(?:share|quantit)\w*[^.\n]{0,30}\binteger\b',
    'finite_difference_risk': r'\b(?:Greeks?|sensitiv\w*|bump\w*)\b',
    'lag_correlations': r'lead[- ]lag|lagged\s+correlations?',
    'hw_trinomial_tree': r'Hull[- ]White|Arrow[- ]Debreu',
    'iv_newton': r'implied[- ]volatil|\bIV\b',
    'iv_li_atm': r'Li (?:ATM|at[- ]the[- ]money)',

    'gbm_greek_samples': r'\bGBM\b|geometric\s+Brownian',
    'gbm_fd_greek_samples': r'\bGBM\b|geometric\s+Brownian',
    'gap_option_price': r'\bgap[\s-]+options?\b',
    'brownian_hit_cdf': r'first[\s-]+passage|hitting[\s-]+(?:time|probability)',
    'gpd_observed_information': r'\bGPD\b|generali[sz]ed\s+Pareto',
}
S12_LIB_SHA = hashlib.sha256(json.dumps(
    [S12_LIB_BASE, S12_LIB_ALIASES, sorted(S12_LIB_VECTORISED), S12_LIB_GATES],
    sort_keys=True).encode()).hexdigest()


def _plain(text):
    """The text without accents (Pynnönen as Pynnonen), as TASK and titles are compared."""
    return ''.join(c for c in unicodedata.normalize('NFKD', text) if not unicodedata.combining(c))


_SENTENCE_END = re.compile(r'(?<![.\s][A-Za-z])(?<!\bal)\.\s')  # a sentence's end, not i.i.d., e.g. or et al.


def _first_sentence(title):
    """The first sentence of a docstring title: up to a period and a space that do not end an abbreviation."""
    return _SENTENCE_END.split(title, maxsplit=1)[0]


def _name_words(name):
    return [word.lower() for word in re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?![a-z])', name)]


def _stem(word):
    """A word without its plural, -ing or -ed ending or a final e (sampling, samples, subsampled and sample meet;
    volatilities and volatility; contributions and contribution)."""
    word = word.lower()
    if word.endswith('ies') and len(word) > 5:
        word = word[:-3] + 'y'
    elif word.endswith('ing') and len(word) >= 7:
        word = word[:-3]
    elif word.endswith('s') and not word.endswith('ss') and len(word) >= 4:
        word = word[:-1]
    elif word.endswith('ed') and not word.endswith('eed') and len(word) >= 6:
        word = word[:-2]
    return word[:-1] if word.endswith('e') and len(word) > 4 else word


def _bounded(body, flags=0):
    return re.compile(rf'(?<![A-Za-z0-9]){body}(?![A-Za-z0-9])', flags)


def _hyphen_pattern(pieces):
    body = _JOIN.join(re.escape(piece) for piece in pieces)
    return re.compile(rf'(?<!{_EDGE}){body}(?!{_EDGE})', re.I)


def _hyphenated_terms(pieces, covered, kind):
    """A hyphenated name as a term of this kind and, of three parts or more, its first two parts as a 'partial'
    one. A name of the function's own method or abbreviation that is a numerical scheme (LIBRARY_SCHEMES:
    Finite-Difference) is a 'scheme' term."""
    if kind in ('method', 'expansion') and any(pattern.fullmatch(' '.join(pieces)) for pattern in _SCHEME_PATTERNS):
        kind = 'scheme'
    terms = [(_hyphen_pattern(pieces), covered, kind)]
    alphabetic = [piece for piece in pieces if piece[:1].isalpha()]
    if len(alphabetic) >= 3:
        terms.append((_hyphen_pattern(alphabetic[:2]), covered, 'partial'))
    return terms


def _initials(pieces):
    return ''.join(piece[0].lower() for piece in pieces if piece[:1].isalpha())


def method_terms(name, title, expansions=None):
    """[(pattern, words of the name it stands for, kind)]: the names of a library function's method that its
    docstring title (first sentence) gives, as patterns to find in a TASK. Kinds:
    - 'method', a name of the function's own method: a hyphenated name of capitalised parts whose parts or initials
      are words of its name (Crank-Nicolson for fd_cn_price, Black-Litterman for black_litterman), an acronym of
      its name with its order (GARCH(1,1)), or a hyphenated phrase whose initials are a word of its name
      (peaks-over-threshold or POT for pot_var_es);
    - 'alias', a LIBRARY_ALIASES name;
    - 'cited', an author's or method's name the title gives beside the function's own name: a hyphenated name
      (Goldman-Sosin-Gatto for lookback_floating_price, Newey-West for ols) or one capitalised word or acronym
      (_TITLE_PROPER: Jamshidian for hw_swaption_price, RiskMetrics for ewma_vol);
    - 'expansion', the hyphenated name a word of the function's name abbreviates in another function's title
      (Black-Scholes-Merton for bsm_greeks);
    - 'name', one word of its name outside LIBRARY_GENERIC that the title capitalises or gives as an acronym
      (Vasicek, CIR, Levy);
    - 'partial', the first two parts of a name of three or more (Black-Scholes of Black-Scholes-Merton), which
      counts only with a qualifier near it;
    - 'scheme', a 'method' or 'expansion' name that is a numerical scheme (Finite-Difference for fd_solve), which
      counts only with a word of the function's own text near it.
    A word of the name that the title itself spells out as a hyphenated name or phrase (Bjerksund-Stensland for bs
    in bs2002_price, finite-difference for fd in fd_cn_price) takes no expansion from other titles: there the same
    initials name other methods (Black-Scholes, Finite-Difference). The first parts of an expansion that the title
    gives (Black-Scholes of Black-Scholes-Merton, for bsm) are that partial name, not a cited one."""
    words = _name_words(name)
    distinct = set(words) - LIBRARY_GENERIC
    spelled_out = ({_initials(match.group(0).split('-')) for match in _TITLE_HYPHENATED.finditer(title)}
                   | {_initials(match.group(0).split('-')) for match in _TITLE_COMPOUND.finditer(title)})
    expanded = {word: [pieces for pieces in (expansions or {}).get(word, ())]
                for word in sorted(distinct) if word not in spelled_out}
    leads = {tuple(piece.lower() for piece in pieces[:size])
             for found in expanded.values() for pieces in found for size in range(2, len(pieces))}
    terms, parts, spans, seen = [], set(), [], set()
    for match in _TITLE_HYPHENATED.finditer(title):
        pieces = match.group(0).split('-')
        parts.update(pieces)
        spans.append(match.span())
        seen.add(match.group(0).lower())
        lowered = {piece.lower() for piece in pieces}
        initials = _initials(pieces)
        if lowered & distinct or initials in distinct:
            terms += _hyphenated_terms(pieces, (lowered | {initials}) & set(words), 'method')
        elif tuple(piece.lower() for piece in pieces) in leads:
            continue  # a partial name of an expansion below
        elif all(len(piece) >= 2 for piece in pieces):
            terms += _hyphenated_terms(pieces, lowered & set(words), 'cited')
    for word, found in expanded.items():
        for pieces in found:
            if '-'.join(pieces).lower() not in seen:
                seen.add('-'.join(pieces).lower())
                terms += _hyphenated_terms(pieces, {word}, 'expansion')
    for match in _TITLE_ACRONYM.finditer(title):
        if match.group(1).lower() in distinct:
            order = match.group(2)
            tail = (r'\s*\(\s*' + r'\s*,\s*'.join(re.escape(x.strip()) for x in order[1:-1].split(',')) + r'\s*\)'
                    if order else rf'(?!{_EDGE})')
            terms.append((re.compile(rf'(?<!{_EDGE}){match.group(1)}{tail}'), {match.group(1).lower()},
                          'method' if order else 'name'))
    for match in _TITLE_NAME.finditer(title):
        if (match.group(0).lower() in distinct and match.group(0) not in parts
                and match.group(0).lower() not in LIBRARY_COMMON_NAMES):
            terms.append((re.compile(rf'(?<!{_EDGE}){match.group(0)}(?!{_EDGE})'), {match.group(0).lower()}, 'name'))
    for match in _TITLE_PROPER.finditer(title):
        word, lead = match.group(0), title[:match.start()].rstrip()
        if (word.lower() in LIBRARY_GENERIC or word.lower() in LIBRARY_COMMON_NAMES or word.lower() in words
                or any(start <= match.start() < end for start, end in spans)
                or title[match.end():match.end() + 1] == '-' or lead.endswith('-')
                or ((not lead or lead.endswith((':', '.', ';'))) and not _TITLE_CITED.match(title[match.end():]))):
            continue
        terms.append((re.compile(rf'(?<!{_EDGE}){re.escape(word)}(?!{_EDGE})'), set(), 'cited'))
    for match in _TITLE_COMPOUND.finditer(title):
        pieces = match.group(0).split('-')
        initials = ''.join(piece[0].lower() for piece in pieces)
        if len(initials) >= 3 and initials in words:
            terms.append((_bounded(_DASHES.join(re.escape(piece) for piece in pieces), re.I), {initials}, 'method'))
            terms.append((_bounded(initials.upper()), {initials}, 'method'))
    for alias in LIBRARY_ALIASES.get(name, ()) + (S12_LIB_ALIASES.get(name, ()) if s12_lib_on() else ()):
        tokens = {_stem(token) for token in _TEXT_WORD.findall(alias)}
        terms.append((_bounded(alias.replace(' ', _DASHES), re.I), {word for word in words if _stem(word) in tokens},
                      'alias'))
    return terms


_FAMILY_PATTERNS = tuple((family, _bounded(rf'(?:{body})', re.I)) for family, body in LIBRARY_FAMILIES)
_FAMILY_OF = dict(_FAMILY_PATTERNS)
_SCHEME_PATTERNS = tuple(_bounded(rf'(?:{body})', re.I) for body in LIBRARY_SCHEMES)


def _families(text):
    return {family for family, pattern in _FAMILY_PATTERNS if pattern.search(text)}


_OPTIONAL = re.compile(r'\boptional(?:ly)?\b[^,;()]*', re.I)  # an optional feature, to the end of its clause


def _title_families(title):
    """The model families a docstring title names, but not one it names only as an optional feature (ou_simulate:
    "with optional ... jumps" simulates a jump-free OU path too)."""
    return _families(_OPTIONAL.sub(' ', title))


_TITLES = {}
_INDEX = {}


def library_titles(name: str = LIBRARY) -> dict:
    """{function: (catalog line, docstring title)} for each function the library's CATALOG lists, read from its
    source without importing it; {} when the library is not installed."""
    return {function: (line, title) for function, (line, title, _) in _library_sources(name).items()}


def _library_sources(name):
    """{function: (catalog line, docstring title, module)}, cached on the library's path and mtime."""
    try:
        spec = importlib.util.find_spec(name)
        init = Path(spec.origin)
        key = (str(init), init.stat().st_mtime_ns, s12_lib_on(), library86_on())
        if key not in _TITLES:
            lines = {}
            for line in library_catalog(name).splitlines():
                match = re.match(r'([A-Za-z_]\w*)\(', line)
                if match:
                    lines[match.group(1)] = line.strip()
            found = {}
            for path in sorted(init.parent.glob('*.py')):
                for node in ast.parse(path.read_text(encoding='utf-8')).body:
                    if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in lines:
                        doc = (ast.get_docstring(node) or '').strip()
                        found[node.name] = (lines[node.name], ' '.join(doc.split('\n\n')[0].split()), path.stem)
            _TITLES.clear()
            _TITLES[key] = found
        return _TITLES[key]
    except Exception:
        return {}


def library_index(name: str = LIBRARY) -> dict:
    """{function: entry} of what library_match_details needs per catalog function: its catalog line and title (first
    sentence), its method_terms, the words of its name its title spells out outside LIBRARY_OUTPUT (each with the
    title's spelling, as a stem), the words near which a scheme term counts (its keyword arguments and every word of
    its name its title spells out), the model families its title names (or, naming none, those of its module's
    functions that share its first name word), the numerical schemes its title names, and which of its terms no
    other function has. Built from the library's own text and cached with it."""
    sources = _library_sources(name)
    if _INDEX.get('key') is not sources:
        firsts = {function: _first_sentence(_plain(title)) for function, (_, title, _) in sources.items()}
        expansions = {}  # initials: the hyphenated names of the titles they abbreviate (bsm: Black-Scholes-Merton)
        for first in firsts.values():
            for match in _TITLE_HYPHENATED.finditer(first):
                pieces = match.group(0).split('-')
                if all(piece[:1].isalpha() for piece in pieces):
                    known = expansions.setdefault(_initials(pieces), [])
                    if pieces not in known:
                        known.append(pieces)
        entries = {}
        for function, (line, title, module) in sources.items():
            first, words = firsts[function], _name_words(function)
            title_stems = [_stem(word) for word in _TEXT_WORD.findall(first)]
            spellings = {}
            for word in words:
                if len(word) >= 3:
                    stem = _stem(word)
                    spelling = stem if stem in title_stems else next(
                        (other for other in title_stems if other.startswith(stem)), None)
                    if spelling:
                        spellings[word] = spelling
            arguments = re.match(r'\w+\([^)*]*\*,([^)]*)\)', line)  # its keyword arguments, the conventions
            keywords = ' '.join(item.split('=')[0] for item in (arguments.group(1) if arguments else '').split(','))
            context = {_stem(word) for word in re.findall(r'[A-Za-z]+', keywords)
                       if len(word) >= 4 and word.lower() not in LIBRARY_GENERIC | LIBRARY_OUTPUT}
            # Conditional call requirements stay visible without becoming method-routing terms.
            contract = title.partition(' API contract: ')[2]
            display = first + '; ' + contract if contract else first
            entries[function] = {'line': line, 'first': first, 'display': display, 'module': module, 'words': words,
                                 'terms': method_terms(function, first, expansions),
                                 'spelled': {word: spelling for word, spelling in spellings.items()
                                             if word not in LIBRARY_OUTPUT},
                                 'context': context | set(spellings.values()), 'families': _title_families(first),
                                 'schemes': [pattern for pattern in _SCHEME_PATTERNS if pattern.search(first)]}
        # A term is this function's own when no other function has it, a name another title only cites aside
        # (Kupiec in rolling_var_backtest) leaving kupiec_pof's own name its own.
        counts = Counter(text for entry in entries.values()
                         for text in {pattern.pattern for pattern, _, _ in entry['terms']})
        owned = Counter(text for entry in entries.values()
                        for text in {pattern.pattern for pattern, _, kind in entry['terms'] if kind != 'cited'})
        for entry in entries.values():
            entry['inherited'] = entry['families'] or set().union(set(), *(
                other['families'] for other in entries.values()
                if other['module'] == entry['module'] and other['words'][:1] == entry['words'][:1]))
            entry['specific'] = {pattern.pattern for pattern, _, kind in entry['terms']
                                 if (counts if kind == 'cited' else owned)[pattern.pattern] == 1}
        _INDEX.clear()
        _INDEX.update(key=sources, entries=entries)
    return _INDEX['entries']


def _near(spans, offset, stems, every=True):
    """Each of the stems (with every=False, one of them) occurs within LIBRARY_PARTIAL_WINDOW words of the word at
    this character offset (spans: [(offset, stem)] of the TASK's words); False for no stems."""
    index = bisect.bisect_left([start for start, _ in spans], offset)
    window = {stem for _, stem in spans[max(0, index - LIBRARY_PARTIAL_WINDOW):index + LIBRARY_PARTIAL_WINDOW + 2]}
    return bool(stems) and (stems <= window if every else bool(stems & window))


def _spellings(entry, word):
    """The stems a word of a function's name occurs as in a TASK: its own, and its title's spelling of it (covariance
    for cov, simulation for simulate)."""
    return {_stem(word)} | ({entry['spelled'][word]} if word in entry['spelled'] else set())


LIBRARY_PINNED_ROUTINES = {
    'lag_correlations': frozenset({'numpy.corrcoef', 'np.corrcoef'}),
    # This interface deliberately implements the named stencil, including edge_order=2
    # and two consecutive strike gradients; the older dupire_local_vol does not.
    'dupire_gradient_surface': frozenset({'numpy.gradient', 'np.gradient'}),
    'multifactor_heston_price': frozenset({'scipy.integrate.quad', 'np.polynomial.legendre.leggauss',
                                          'numpy.polynomial.legendre.leggauss'}),
    'kde_var_es': frozenset({'scipy.stats.gaussian_kde', 'scipy.integrate.quad'}),
}


def _pinned(clauses, patterns, schemes, compatible=frozenset()):
    """A clause of the TASK names a numpy or scipy routine together with one of the patterns or schemes, or rules
    one of the schemes out."""
    def other_routine(clause):
        for routine in compatible:
            clause = re.sub(r'(?<![\w.])' + re.escape(routine) + r'(?!\w|\.\w)', '', clause)
        return _ROUTINE.search(clause)
    return any((other_routine(clause) and any(pattern.search(clause) for pattern in patterns + schemes))
               or (_RULED_OUT.search(clause) and any(pattern.search(clause) for pattern in schemes))
               for clause in clauses)


def _library_dependency_limit(instruction: str):
    """A narrow whole-task dependency declaration, or None when its scope/list is unknown.

    This is prompt admission, not an import sandbox or a general language parser.
    Only complete affirmative declarations in unscoped/global sections count;
    local steps, examples, quotes, code, negations and unparsed complements do not.
    The legacy catalog profile keeps its existing behavior.
    """
    if not s12_lib_on():
        return None
    global_sections = {'constraints', 'notes', 'requirements', 'dependencies', 'implementation',
                       'implementation notes', 'libraries', 'allowed libraries', 'language and libraries'}
    # Module vocabulary establishes that an otherwise bare list concerns dependencies,
    # not columns/rows/observations. Explicitly labelled library lists need no such hint.
    modules = {'numpy', 'pandas', 'scipy', 'pyarrow', 'matplotlib', 'statsmodels', 'plotly',
               'sklearn', 'arch', 'polars', 'seaborn', 'openpyxl', 'numba', 'talib',
               'backtrader', 'tqdm', 'json', 'csv', 'math', 'os', LIBRARY}
    stdlib = (r'(?:the\s+)?(?:standard(?:\s+python)?\s+librar(?:y|ies)|'
              r'(?:python\s+)?standard\s+library)(?:\s+modules)?')
    whole = r'(?:this|the|the entire|the whole)\s+(?:task|program|solution|implementation)'
    headings, fence = [], ''
    for raw in instruction.splitlines():
        line = raw.strip()
        marker = re.match(r'(`{3,}|~{3,})', line)
        if marker:
            mark = marker.group(1)
            if not fence:
                fence = mark
            elif mark[0] == fence[0] and len(mark) >= len(fence):
                fence = ''
            continue
        if fence or line.startswith('>') or raw.startswith(('    ', '\t')):
            continue
        header = re.match(r'^(#{1,6})\s+(.+?)\s*#*$', line)
        if header:
            level, title = len(header[1]), re.sub(r'[*`]', '', header[2]).lower().rstrip(':')
            headings = [(depth, name) for depth, name in headings if depth < level]
            headings.append((level, title))
            continue
        if any(name.startswith(('example', 'quoted')) or re.match(r'^(?:step|phase)\b', name)
               or (level > 1 and name not in global_sections)
               for level, name in headings):
            continue
        line = re.sub(r'^\s*(?:[-+*]|\d+[.)])\s+', '', line)
        line = re.sub(r'[*`]', '', _plain(line)).lower().rstrip(' .')
        # A separate delivery sentence is not part of the dependency list.
        # Keep ambiguous tails intact: they may qualify or permit dependencies.
        delivery = re.split(r'\.\s+(?=(?:write|save|output|return|produce)\b)', line, maxsplit=1)
        if len(delivery) == 2:
            cues = (r'\b(?:use|using|import|modules?|packages?|librar(?:y|ies)|dependenc(?:y|ies)|'
                    r'stdlib|standard|except|unless|alternatively|allowed|permit(?:ted)?|'
                    + '|'.join(re.escape(name) for name in sorted(modules - sys.stdlib_module_names))
                    + r')\b')
            if not re.search(cues, delivery[1]) and not re.search(r'\.\s+\S', delivery[1]):
                line = delivery[0]
        line = re.sub(r'^for\s+' + whole + r'\s*,\s*', '', line)
        line = re.sub(r'\s+for\s+' + whole + r'$', '', line)
        found = re.fullmatch(r'(?:(?:you\s+)?must\s+)?(?:use|using)\s+(.+)', line)
        if not found:
            continue
        body = found[1]
        if '(' in body:
            core, note = body.split('(', 1)
            # An optional included dependency may permit this library. Unknown or
            # conditional exceptions cannot establish a closed list excluding it.
            if (not note.endswith(')') or '(' in note or not note.startswith('including ')
                    or re.search(r'\b(?:unless|except|when|only|for|not)\b', note)
                    or re.search(r'\b' + re.escape(LIBRARY) + r'\b', note)):
                continue
            body = core.rstrip()
        if body.startswith('only '):
            body = body[5:]
        elif body.endswith(' only'):
            body = body[:-5]
        else:
            continue
        body = re.sub(r'^python\s+with\s+', '', body)
        labelled = re.match(r'^(?:(?:these|the following)\s+)?(?:libraries|packages|modules)\s*:\s*', body)
        if labelled:
            body = body[labelled.end():]
        body = re.sub(stdlib, '__stdlib__', body)
        if body.startswith('__stdlib__:'):
            body = '__stdlib__,' + body.split(':', 1)[1]
        names = [item.strip() for item in re.split(r',|\band\b|&', body) if item.strip()]
        if (not names or any(not re.fullmatch(r'[a-z_]\w*(?:\.[a-z_]\w*)*', name) for name in names)
                or not (labelled or '__stdlib__' in names or any(name.split('.')[0] in modules for name in names))):
            continue
        if any(name.split('.')[0] == LIBRARY for name in names):
            continue
        return ('stdlib_only' if all(name == '__stdlib__' or name.split('.')[0] in sys.stdlib_module_names
                                     for name in names) else 'allowlist_excludes_library')
    return None


def library_allowed_by_task(instruction: str) -> bool:
    """Offer the optional library unless a recognized whole-task dependency list excludes it."""
    return _library_dependency_limit(instruction) is None


def library_match_details(instruction: str, name: str = LIBRARY) -> list:
    """[(function, catalog line, docstring title, strength)] of the library functions whose method the TASK names, in
    catalog order: the LIBRARY_MATCH_LIMIT strongest (in catalog order among equals). The TASK is read without
    accents, markdown emphasis or a references section.

    A function is found when one of its method_terms occurs (a 'scheme' term only with a word of the function's
    keyword arguments or of its name as its title spells it within LIBRARY_PARTIAL_WINDOW words: a finite-difference
    scheme or grid, not a finite-difference bump), or every one of at least two words of its name outside
    LIBRARY_GENERIC occurs within LIBRARY_PHRASE_WINDOW words, as the name or its title spells it (a geometric Asian
    option; Jump Detection for detect_jumps). It is listed only when:
    - found only by an author's name its title cites beside its own (a 'cited' term: Reiner-Rubinstein for
      digital_price), a distinct word of its name or its initials (fd of fd_solve) occur too; a citation is
      confirmed by such a word or by another term only this function has (a name another title only cites stays
      a function's own: Kupiec of kupiec_pof);
    - no TASK clause names a numpy or scipy routine together with a term found (not a 'cited' one) or a numerical
      scheme its title names, or rules such a scheme out (LIBRARY_SCHEMES): such a TASK keeps its own step;
    - each model family its title names (not as an optional feature) or takes from its module's namesakes occurs
      (LIBRARY_FAMILIES);
    - unless a citation was confirmed, each word of its name its title spells out occurs, as the name or the title
      spells it, outside LIBRARY_OUTPUT and the words the term found stands for (jump for ou_jump_moments,
      covariance for ewma_cov).
    Strength 3 for an explicit method name (a 'method' or 'alias' term, a phrase of name words) or a name only this
    function has (Dupire; a confirmed citation such as Conze-Viswanathan); 2 for a name other functions share
    (Merton, CIR, EWMA) with a qualifier beyond the term itself (a spelled word, a confirmed citation, or a family
    the term does not express), for a scheme name found, and for a partial name (Black-Scholes of
    Black-Scholes-Merton) whose spelled words occur within LIBRARY_PARTIAL_WINDOW words of it (without them it is
    not listed); 1 for a shared name with nothing to qualify it.
    Keyed on the library's own text only."""
    if not library_allowed_by_task(instruction):
        return []
    text = _plain(instruction)
    text = _REFERENCES.sub('', re.sub(r'[*`]', ' ', text.replace('\u2019', "'").replace('\u2018', "'")))
    spans = [(match.start(), _stem(match.group(0))) for match in _TEXT_WORD.finditer(text)]
    stems = [stem for _, stem in spans]
    stem_set, families = set(stems), _families(text)
    clauses = [' '.join(clause.split()) for clause in _CLAUSE.split(text)
               if clause and (_ROUTINE.search(clause) or _RULED_OUT.search(clause))]
    found = []
    order = {line: index for index, line in enumerate(library_catalog(name).splitlines())}
    for function, entry in library_index(name).items():
        gate = S12_LIB_GATES.get(function)
        if gate and not re.search(gate, text, re.I):
            continue
        hits = []  # (words of the name the term stands for, kind, pattern, text found, the function's own term)
        for pattern, covered, kind in entry['terms']:
            match = next((match for match in pattern.finditer(text)
                          if kind != 'scheme' or _near(spans, match.start(), entry['context'], every=False)), None)
            if match:
                hits.append((covered, kind, pattern, match.group(0), pattern.pattern in entry['specific']))
        # Each word of the name as the name or the title spells it (detect or detection of detect_jumps).
        phrase = {word: _spellings(entry, word) for word in entry['words']
                  if len(word) >= 3 and word not in LIBRARY_GENERIC}
        if len(phrase) >= 2 and any(all(forms & set(stems[i:i + LIBRARY_PHRASE_WINDOW]) for forms in phrase.values())
                                    for i in range(len(stems))):
            hits.append((set(phrase), 'method', None, '', True))
        if not hits:
            continue
        covered = set().union(*(hit[0] for hit in hits))
        qualifiers = [_spellings(entry, word) for word in entry['spelled'] if word not in covered]
        distinct = {_stem(word) for word in entry['words']
                    if len(word) >= 3 and word not in LIBRARY_GENERIC and word not in covered}
        # With these, its initials (fd of fd_solve), which only the function's own terms spell out.
        identifying = distinct | {word for word in entry['words']
                                  if len(word) == 2 and word not in LIBRARY_GENERIC and word not in covered}
        cited = [hit for hit in hits if hit[1] == 'cited']
        if cited and len(cited) == len(hits) and identifying and not identifying & stem_set:
            continue  # only an author's name its title cites beside its own (Reiner-Rubinstein for digital_price)
        # A cited name confirmed by a distinct word of the function's name or by a term only this function has.
        confirmed = bool(cited) and (bool(distinct & stem_set) or any(hit[4] for hit in hits if hit[1] != 'cited'))
        if _pinned(clauses, [hit[2] for hit in hits if hit[2] is not None and hit[1] != 'cited'], entry['schemes'],
                   LIBRARY_PINNED_ROUTINES.get(function, frozenset())):
            continue
        if not entry['inherited'] <= families or not (confirmed or all(forms & stem_set for forms in qualifiers)):
            continue  # without this function's own qualifiers
        named = [hit for hit in hits if hit[1] not in ('partial', 'scheme')]
        if any(hit[1] in ('method', 'alias') or (hit[4] and (hit[1] != 'cited' or confirmed)) for hit in named):
            strength = 3  # an explicit method name, or a name only this function has
        elif named:
            beyond = confirmed or bool(qualifiers) or any(
                not any(_FAMILY_OF[family].search(hit[3]) for hit in hits) for family in entry['inherited'])
            strength = 2 if beyond else 1
        elif any(hit[1] == 'scheme' for hit in hits) or (qualifiers and any(
                all(_near(spans, match.start(), forms, every=False) for forms in qualifiers)
                for hit in hits for match in hit[2].finditer(text))):
            strength = 2
        else:
            continue
        found.append((function, entry['line'], entry['display'], strength))
    kept = sorted(found, key=lambda item: (-item[3], order.get(item[1], 0)))[:LIBRARY_MATCH_LIMIT]
    return sorted(kept, key=lambda item: order.get(item[1], 0))


def library_matches(instruction: str, name: str = LIBRARY) -> list:
    """[(function, catalog line, docstring title)] of library_match_details."""
    return [(function, line, first) for function, line, first, _ in library_match_details(instruction, name)]


def library_confident(instruction: str, name: str = LIBRARY) -> set:
    """The functions library_match_details finds with strength 2 or more: an explicit method name, a phrase of name
    words, or one name word with a qualifier or of this function only."""
    return {function for function, _, _, strength in library_match_details(instruction, name) if strength >= 2}


def _catalog_source(name):
    """(the CATALOG literal of the library's __init__, the S12_LIB_BASE literal or None), read without importing it;
    ('', None) when this interpreter cannot import the library or the CATALOG is not a string literal."""
    boundary = 'CATALOG_050_LEN' if s12_lib_on() else S12_LIB_BASE
    try:
        spec = importlib.util.find_spec(name)
        tree = ast.parse(Path(spec.origin).read_text(encoding='utf-8'))
        literals = {node.targets[0].id: node.value for node in tree.body
                    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id in ('CATALOG', boundary)}
        if 'CATALOG' not in literals:
            return '', None
        value = ast.literal_eval(literals['CATALOG'])
    except Exception:  # not installed, a namespace directory, or an unreadable source: nothing is offered
        return '', None
    try:
        base = ast.literal_eval(literals[boundary]) if boundary in literals else None
    except Exception:
        base = None
    base = base if isinstance(base, int) and not isinstance(base, bool) and base > 0 else None
    return (value, base) if isinstance(value, str) else ('', None)


def library_catalog(name: str = LIBRARY) -> str:
    """The optional library's CATALOG, read from its source without importing it (the controller never loads
    numpy for this); '' when this interpreter cannot import the library, so no prompt offers a missing one. Without
    T1_S12_LIB it offers 0.4; that flag offers 0.5; only T1_LIBRARY86 exposes the complete installed catalog."""
    catalog, base = _catalog_source(name)
    return catalog if base is None or library86_on() else catalog[:base]


def library86_on() -> bool:
    """T1_LIBRARY86=1 exposes the full installed 0.6 catalog; default remains legacy 0.4."""
    return os.environ.get('T1_LIBRARY86') == '1'


def s12_lib_on() -> bool:
    """Enable extended aliases with the 0.5 profile or the independently selected full catalog."""
    return os.environ.get('T1_S12_LIB') == '1' or library86_on()


_HIDDEN = {}


def s12_hidden(name: str = LIBRARY) -> frozenset:
    """The functions the CATALOG lists past the cut library_catalog makes without T1_S12_LIB (the 0.5.0 additions),
    which the controller then treats as absent; empty under T1_LIBRARY86 or for a library without the cut."""
    if library86_on():
        return frozenset()
    try:
        init = Path(importlib.util.find_spec(name).origin)
        key = (str(init), init.stat().st_mtime_ns, s12_lib_on(), library86_on())
    except Exception:
        return frozenset()
    if key not in _HIDDEN:
        catalog, base = _catalog_source(name)
        _HIDDEN.clear()
        _HIDDEN[key] = frozenset() if base is None else frozenset(
            match.group(1) for line in catalog[base:].splitlines() if (match := re.match(r'([A-Za-z_]\w*)\(', line)))
    return _HIDDEN[key]


def library_vectorised() -> frozenset:
    """LIBRARY_VECTORISED, with S12_LIB_VECTORISED under T1_S12_LIB."""
    additions = S12_LIB_VECTORISED - s12_hidden() if s12_lib_on() else frozenset()
    return LIBRARY_VECTORISED | additions


def libhint_on() -> bool:
    """T1_LIBHINT=1 (the image's flag): the library block states how its functions are called, that vectorised
    ones are called once on whole arrays (LIBRARY_VECTOR), and lists the catalog functions whose method the TASK
    names (library_matches, marking the vectorised ones); the plan maps each named method to a function
    or none in fixed lines (PLAN_LIBRARY_MAP); a crash or review fix is pointed at a matched function the program
    does not call; a clean run's fit report opens the review's fix round when a matched library fit is not called
    (library_fits); a crash of a library call or its result adds the function's own signature and Returns section
    to the crash request, which says so when a patch leaves or brings back the same misuse (library_use). Off, the
    library is offered as before."""
    return os.environ.get('T1_LIBHINT') == '1'


def library_parts(instruction: str = '') -> tuple:
    """(the library block of the task prompt, the number of LIBRARY MATCHES it lists). Under T1_S11_LIBNOTES the
    LIBRARY NOTES of the matched functions follow the matches (s11_notes_block)."""
    limit = _library_dependency_limit(instruction)
    if limit:
        rule = ('The TASK requires standard-library-only code. ' if limit == 'stdlib_only'
                else 'The TASK restricts the permitted dependencies. ')
        return ('DEPENDENCY LIMIT: ' + rule
                + 'Use only its permitted modules and write the required calculations at runtime.\n\n', 0)
    catalog = library_catalog().strip()
    if not catalog:
        return '', 0
    if not libhint_on():
        return LIBRARY_BLOCK.format(catalog=catalog), 0
    adopt = adopt_on()
    block = (LIBRARY_BLOCK_ADOPT if adopt else LIBRARY_BLOCK).replace(
        ' Catalog:\n', ' ' + LIBRARY_CALLS + LIBRARY_VECTOR + 'Catalog:\n', 1).format(catalog=catalog)
    if s12_lib_on():
        block += ('Step-specific dependency limits in the TASK take precedence over library adoption; '
                  'do not call this library in a step that restricts its permitted modules.\n\n')
    matches = library_matches(instruction) if instruction else []
    if matches:
        block += (LIBRARY_MATCHES_ADOPT if adopt else LIBRARY_MATCHES).format(
            matches='\n'.join(f'- `{LIBRARY}.{line.split(" - ")[0]}`'
                               f'{LIBRARY_VECTOR_MARK if function in library_vectorised() else ""} - {title}'
                               for function, line, title in matches))
        if s11_libnotes_on():  # T1_S11_LIBNOTES: how to call them, right after them
            try:
                block += s11_notes_block([function for function, _, _ in matches])
            except Exception:  # the notes only help: the block goes out without them
                pass
    return block, len(matches)


def library_block(instruction: str = '') -> str:
    return library_parts(instruction)[0]


def matched_functions(messages) -> list:
    """The functions the LIBRARY MATCHES block of the task prompt (the first message) lists, in order."""
    text = messages[0]['content'] if messages and isinstance(messages[0].get('content'), str) else ''
    head = LIBRARY_MATCHES.split(':')[0] + ':'
    if head not in text:
        return []
    block = text.split(head, 1)[1].split('\n\n', 1)[0]
    return re.findall(r'^- `' + re.escape(LIBRARY) + r'\.([A-Za-z_]\w*)\(', block, re.M)


def library_fix(messages, code, what, required=True) -> str:
    """Under T1_LIBHINT, the note pointing a fix at the matched library functions the program does not call
    (LIBRARY_FIX); '' when there is none. Under T1_ADOPT, with required (the fixes after a clean run, not a crash
    repair), the functions REQUIRED LIBRARY CALLS lists that the program does not call (library_uncalled) are named
    too (LIBRARY_FIX_ADOPT)."""
    if not libhint_on():
        return ''
    missing = [name for name in matched_functions(messages) if not re.search(rf'\b{name}\b', code)]
    extra = ([name for name in library_uncalled(code, required_functions(messages)) if name not in missing]
             if required and adopt_on() else [])
    names = ', '.join(f'`{LIBRARY}.{name}`' for name in missing + extra)
    return ((LIBRARY_FIX_ADOPT if extra else LIBRARY_FIX).format(names=names, what=what)
            if missing or extra else '')


_FIT_TITLE = re.compile(r'\b(?:fit|fits|fitted|fitting|estimat\w*|likelihood|MLE|calibrat\w*)\b', re.I)


def library_fits(messages, code) -> list:
    """Under T1_LIBHINT, the matched library functions (matched_functions) that fit or estimate a model, by their
    own name (a word `fit`) or docstring title (fit, estimate, likelihood, MLE, calibrate), and that the program
    does not call; [] otherwise. Keyed on the library's text only."""
    if not libhint_on():
        return []
    titles = library_titles()
    return [name for name in matched_functions(messages) if not re.search(rf'\b{name}\b', code)
            and ('fit' in _name_words(name) or _FIT_TITLE.search(_first_sentence(titles.get(name, ('', ''))[1])))]


def adopt_on() -> bool:
    """T1_ADOPT=1 (the image's flag, with T1_LIBHINT): the library is used for every standard method a catalog
    function computes, and a step is written by hand only where the TASK writes out a different formula or
    convention (LIBRARY_BLOCK_ADOPT, LIBRARY_MATCHES_ADOPT); the plan's LIBRARY MAP maps no function as optional
    and quotes the TASK on a none line (PLAN_LIBRARY_MAP_ADOPT); the functions it maps become REQUIRED LIBRARY CALLS
    of the program prompt (library_map, with_required), and a fix after a clean run is pointed at one the program
    does not call (library_fix). Off, requests and programs are as before."""
    return os.environ.get('T1_ADOPT') == '1' and not minimal_on() and libhint_on()


def library_functions(name: str = LIBRARY) -> set:
    """The names of the functions (and classes) the library's CATALOG lists; set() when it is not installed."""
    return {match.group(1) for line in library_catalog(name).splitlines()
            for match in [re.match(r'\s*([A-Za-z_]\w*)\(', line)] if match}


_MAP_ARROW = re.compile(r'->|=>|\u2192|\u21d2|\u27f6')
_MAP_BULLET = re.compile(r'^\s*(?:[-+\u2022]\s+|\(?\d{1,2}[.)]\s+)?')
_MAP_QUOTE = re.compile(r'"([^"\n]+)"|\u201c([^\u201d\n]+)\u201d')
_MAP_DIFFERENT = re.compile(r'none\b\W*different\s+models?\b', re.I)  # none: different model: <why>
# A mapped function the plan itself calls optional, a check or not needed: the function is required only when the
# TASK names its method (LIBRARY MATCHES), since a plan also lists functions the outputs never need this way.
_MAP_OPTIONAL = re.compile(r'\b(?:optional\w*|verif\w*|sanity|cross[- ]?check\w*|for checking|to check|if needed|'
                           r'only (?:for|to|if|as|when)|not (?:required|needed|used|mandated)|could be used|'
                           r'can be used)\b', re.I)


def _map_split(text):
    """text split at its first arrow, the whitespace around the arrow dropped: what a split at the first match of
    whitespace, arrow, whitespace gives, in linear time (that pattern costs quadratic time along a long run of
    whitespace, and map lines are read from model text on the unit clock)."""
    match = _MAP_ARROW.search(text)
    return [text] if match is None else [text[:match.start()].rstrip(), text[match.end():].lstrip()]


def _map_method(text):
    method = ' '.join(text.split()).strip(' :;,.-"\'')
    return method[:ADOPT_METHOD_CHARS].rstrip() + ('...' if len(method) > ADOPT_METHOD_CHARS else '')


def _names_method(function, method, line):
    """A LIBRARY MATCHES function is the one a map line's method means: the line names it, the library's own matcher
    finds it in the method, or the words of its name (three letters or more, as the name or its title spells them)
    are words of the method: those outside LIBRARY_GENERIC when there are two or more of them or the one is a name
    its title gives (GARCH for garch11_fit, Heston), or else all of them (Jump-adjusted moments for ou_jump_moments).
    A line on VaR is not var_es_forecast without forecast, nor Jump detection ou_jump_moments."""
    if re.search(rf'(?<![\w.]){re.escape(function)}\b', line):
        return True
    if any(found == function for found, _, _ in library_matches(method)):
        return True
    entry = library_index().get(function, {'spelled': {}, 'terms': []})
    stems = {_stem(word) for word in _TEXT_WORD.findall(_plain(method))}
    words = [word for word in _name_words(function) if len(word) >= 3]
    distinct = [word for word in words if word not in LIBRARY_GENERIC]
    if not distinct:
        return False
    named = set().union(set(), *(covered for _, covered, kind in entry['terms']
                                 if kind in ('method', 'name', 'expansion')))
    return all(_spellings(entry, word) & stems for word in
               (distinct if len(distinct) >= 2 or distinct[0] in named else words))


def library_map(plan: str, instruction: str, matched=(), counts=None) -> tuple:
    """([(function, method)], none lines dropped) from the LIBRARY MAP of a plan, read line by line: a line
    `<method> -> qfkit.<function>...` (an arrow ->, =>, or its Unicode forms; markdown bullets, bold and backticks
    ignored) maps the method to a function the CATALOG lists (others are ignored), unless the line calls it optional,
    a check or not needed and the function is not one of matched (the LIBRARY MATCHES). A line `<method> -> none`
    stands when it quotes a TASK sentence (_quoted) or says the function of that name computes a different model
    (`none: different model: <why>`, counted in counts['different'] when counts is given); otherwise each matched
    function its method means (_names_method) that the TASK names with confidence (library_confident: an explicit
    method or author name, or a name with a qualifier) is mapped to it instead, and the line counts as dropped. In
    order, each function once, at most ADOPT_LIMIT."""
    if not library_allowed_by_task(instruction):
        return [], 0
    known, matched = library_functions(), list(matched)
    confident = library_confident(instruction) if matched else set()
    found, dropped = [], 0
    for raw in (plan or '').splitlines():
        line = raw.replace('`', '').replace('*', '')
        parts = _map_split(line)
        if len(parts) != 2:
            continue
        method, target = _map_method(_MAP_BULLET.sub('', parts[0])), parts[1].strip()
        if not method:
            continue
        mapped = re.match(rf'{LIBRARY}\.([A-Za-z_]\w*)', target)
        if mapped:
            function = mapped.group(1)
            if function in known and (function in matched or not _MAP_OPTIONAL.search(line)):
                found.append((function, method))
        elif re.match(r'none\b', target, re.I):
            quotes = [a or b for a, b in _MAP_QUOTE.findall(target)]
            if any(_quoted(quote, instruction) for quote in quotes):
                continue
            if _MAP_DIFFERENT.match(target):
                if counts is not None:
                    counts['different'] = counts.get('different', 0) + 1
                continue
            meant = [function for function in matched
                     if function in confident and _names_method(function, method, line)]
            if meant:
                dropped += 1
                found += [(function, method) for function in meant]
    required, seen = [], set()
    for function, method in found:
        if function not in seen:
            seen.add(function)
            required.append((function, method))
    return required[:ADOPT_LIMIT], dropped


def _required_paragraph(required):
    return ADOPT_REQUIRED.format(calls='; '.join(f'`{LIBRARY}.{function}` for {method}'
                                                 for function, method in required))


def with_required(prompt, required):
    """The program prompt with its REQUIRED LIBRARY CALLS (ADOPT_REQUIRED) before the output format; the prompt
    as it is without any."""
    if not required:
        return prompt
    return prompt.removesuffix(OUTPUT_FORMAT) + _required_paragraph(required) + OUTPUT_FORMAT


def required_functions(messages) -> list:
    """The functions the REQUIRED LIBRARY CALLS of the program prompt (the first message) lists, in order."""
    text = messages[0]['content'] if messages and isinstance(messages[0].get('content'), str) else ''
    head = ADOPT_REQUIRED.split(' (')[0]
    if head not in text:
        return []
    block = text.split(head, 1)[1].split('\n\n', 1)[0]
    return re.findall(r'`' + re.escape(LIBRARY) + r'\.([A-Za-z_]\w*)` for ', block)


def library_called(code: str, name: str = LIBRARY) -> set:
    """The library's names the program refers to, read without running it: `lib.f` through each name `import lib`
    binds (`import lib as q`), and each name `from lib import f` binds (`as g` too), called or passed on; after
    `from lib import *`, each name the program calls. A name in a comment or a string is not one. set() when the
    program does not parse."""
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return set()
    modules, imported, star = set(), {}, False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(item.asname or name for item in node.names if item.name == name)
        elif isinstance(node, ast.ImportFrom) and node.module == name and not node.level:
            for item in node.names:
                if item.name == '*':
                    star = True
                else:
                    imported[item.asname or item.name] = item.name
    used = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in modules:
            used.add(node.attr)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id in imported:
            used.add(imported[node.id])
        elif star and isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            used.add(node.func.id)  # after `from lib import *`, a name the program calls may be the library's
    return used


def library_uncalled(code: str, functions) -> list:
    """The functions (in order) the program does not refer to (library_called)."""
    used = library_called(code) if functions else set()
    return [function for function in functions if function not in used]


LIBRARY_USE = ('LIBRARY USE: {how}. Call each and read its result exactly as the library defines them (its own '
               'argument names and values, and the keys of what it returns), not by another package\'s '
               'spellings:\n{items}\n')
LIBRARY_INPUTS = ('LIBRARY USE: {how}. The library rejected the values passed to it: check them against its own '
                  'argument entries below and clean or handle them before the call:\n{items}\n')
LIBRARY_AGAIN = ('Your previous change{names} did not fix this: the program misuses `{lib}.{function}` again as in an '
                 'earlier run{error}. Guessing another spelling does not work: use the signature, argument values '
                 'and returned keys LIBRARY USE lists.\n')
LIBRARY_USE_CAP = 1200  # characters of the LIBRARY USE items
LIBRARY_USE_ERRORS = ('KeyError', 'IndexError', 'TypeError', 'ValueError', 'AttributeError')
_SECTION_RULE = re.compile(r'\s*-{3,}\s*$')
_DEFS = {}


def _library_defs(name: str = LIBRARY) -> dict:
    """{name: def or class node} of the public top-level functions and classes of the optional library's modules,
    read from its source once per file version (as library_titles); {} when it is not installed."""
    try:
        init = Path(importlib.util.find_spec(name).origin)
        key = (str(init), init.stat().st_mtime_ns, s12_lib_on(), library86_on())
        if key not in _DEFS:
            found = {}
            for path in sorted(init.parent.glob('*.py')):
                for node in ast.parse(path.read_text(encoding='utf-8')).body:
                    if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and not node.name.startswith('_'):
                        found.setdefault(node.name, node)
            _DEFS.clear()
            _DEFS[key] = found
        hidden = s12_hidden(name)  # without T1_S12_LIB the 0.5.0 functions are absent
        return {function: node for function, node in _DEFS[key].items() if function not in hidden} if hidden \
            else _DEFS[key]
    except Exception:
        return {}


def _doc_sections(doc: str) -> dict:
    """{title: [lines]} of a numpy-style docstring's sections (a title line over a rule of dashes); the text
    before the first section is under ''."""
    lines, sections, title = doc.split('\n'), {'': []}, ''
    for index, line in enumerate(lines):
        if _SECTION_RULE.fullmatch(line) and index and lines[index - 1].strip():
            continue
        if index + 1 < len(lines) and line.strip() and _SECTION_RULE.fullmatch(lines[index + 1]):
            title = line.strip()
            sections[title] = []
            continue
        sections[title].append(line)
    return sections


def _accepted_values(node, parameter):
    """The string constants the function's own code tests `parameter` for membership in, in source order."""
    values = []
    for compare in ast.walk(node):
        if (isinstance(compare, ast.Compare) and isinstance(compare.left, ast.Name) and compare.left.id == parameter
                and len(compare.ops) == 1 and isinstance(compare.ops[0], (ast.In, ast.NotIn))
                and isinstance(compare.comparators[0], (ast.Tuple, ast.List, ast.Set))):
            values += [item.value for item in compare.comparators[0].elts
                       if isinstance(item, ast.Constant) and isinstance(item.value, str) and item.value not in values]
    return values


def _library_call(node):
    """(the def a library name is called through: the function, a class's __init__ or None; its arguments as the
    caller passes them, a class's without self)."""
    target = node
    if isinstance(node, ast.ClassDef):
        target = next((item for item in node.body if isinstance(item, ast.FunctionDef) and item.name == '__init__'),
                      None)
    arguments = ''
    if target is not None:
        args = target.args
        if target is not node:  # a class is called without self
            args = ast.arguments(posonlyargs=[], args=args.args[1:], vararg=args.vararg,
                                 kwonlyargs=args.kwonlyargs, kw_defaults=args.kw_defaults, kwarg=args.kwarg,
                                 defaults=args.defaults[max(0, len(args.defaults) - len(args.args) + 1):])
        arguments = ast.unparse(args)
    return target, arguments


def library_entry(function: str, exception: str = '', keys: str = '') -> str:
    """The LIBRARY USE item for one library function, from its own source: its signature, for a TypeError or
    ValueError the Parameters entries of the arguments the error message names (with the values its code accepts
    for them), and its Returns section, or the keys the crash report found when it has none. '' when the library
    does not define it."""
    node = _library_defs().get(function)
    if node is None:
        return ''
    target, arguments = _library_call(node)
    lines = [f'- `{LIBRARY}.{function}({arguments})`']
    sections = _doc_sections(ast.get_docstring(node) or '')
    kind = exception.split(':', 1)[0].rsplit('.', 1)[-1]
    if kind in ('TypeError', 'ValueError') and target is not None:
        named = set(re.findall(r'[A-Za-z_]\w*', exception.split(':', 1)[-1]))
        entries = [line.strip() for line in sections.get('Parameters', [])]
        for arg in [a.arg for a in target.args.args + target.args.kwonlyargs if a.arg in named]:
            entry = next((line for line in entries if re.match(rf'{arg}\b[^:]*:', line)), '')
            values = [value for value in _accepted_values(target, arg) if repr(value) not in entry]
            if entry or values:
                lines.append('  ' + (entry[:200] if entry else arg) + (
                    f' (accepted values: {", ".join(map(repr, values))})' if values else ''))
    paragraph = '\n'.join(sections.get('Returns', [])).strip().split('\n\n')[0]  # not the formula after it
    returns = ' '.join(paragraph.split())
    if returns:
        lines.append('  Returns: ' + (returns if len(returns) <= 600 else returns[:597] + '...'))
    elif keys:
        lines.append('  ' + keys)
    return '\n'.join(lines)


_CARD_VALUE = re.compile(r"'([^'\s]{1,40})'")
# What may stand before the first value of a list: the entry's start, a bracket, a clause's end, `or`, `of`, `None,`
# or `required,`.
_CARD_VALUE_OPENS = re.compile(r'(?:^|[{(;:.|]|\bor|\bof|\bNone,|\brequired,)\s*$', re.I)
_CARD_VALUE_SEPARATOR = r'\s*(?:,|\||/|or|, or)\s*'
# Between two values of one list: a separator, after the first value's gloss in brackets ('cn' (Crank-Nicolson),
# 'implicit' (fully implicit) or 'explicit').
_CARD_VALUE_JOINS = re.compile(r'\s*(?:\((?:[^()]|\([^()]*\))*\)\s*)?' + _CARD_VALUE_SEPARATOR)
# After a value that opens a line: more values, then two spaces (its description) or the line's end: a table's row.
_CARD_VALUE_ROW = re.compile(rf"(?:{_CARD_VALUE_SEPARATOR}'[^'\s]{{1,40}}')*(?:[ \t]{{2,}}|[ \t]*(?:\n|$))")


def _doc_entries(lines, names) -> dict:
    """{name: entry, its line breaks kept} of a numpy-style Parameters section. An entry `a, b : text` opens a line
    at the section's indent, or follows a sentence inside one (`n_space : ... . n_time : ...`), and names only the
    function's own arguments (names); the lines below it continue it."""
    if not names:
        return {}
    names = '|'.join(map(re.escape, sorted(names, key=len, reverse=True)))
    opener = re.compile(rf'(?:^|(?<=[.;] ))\**((?:{names})(?:, \**(?:{names}))*) : ')
    base = min((len(line) - len(line.lstrip()) for line in lines if line.strip()), default=0)
    found, current = {}, []
    for line in lines:
        text = line.strip()
        if not text:
            continue
        opens = [match for match in opener.finditer(text)
                 if match.start() or len(line) - len(line.lstrip()) == base]
        head = text[:opens[0].start()].strip() if opens else text
        for name in current if head else ():
            found[name] += '\n' + head
        for index, match in enumerate(opens):
            current = [name.strip().lstrip('*') for name in match.group(1).split(',')]
            end = opens[index + 1].start() if index + 1 < len(opens) else len(text)
            for name in current:
                found[name] = text[match.end():end].strip()
    return found


def _entry_values(entry) -> list:
    """The values a Parameters entry says its argument accepts, in order: the quoted words of each list that opens
    the entry, a clause, a set or a bracket, or follows `or` ({'a', 'b'}; 'a': ...; None, 'a' or 'b'), and of each
    row of a table (a line that opens with values and then describes them); a list goes on through separators, and
    through a value's gloss in brackets. In an entry with a table only the lists that open it, a clause or a row
    count: a list in brackets or after `or` there gives the values of another keyword a row needs. A quoted word
    inside a sentence (with 'x', another package's mean='Constant') is not one."""
    lists, last = [], None  # [(what opened the list, its values)]
    for match in _CARD_VALUE.finditer(entry):
        before = entry[:match.start()]
        if last is not None and _CARD_VALUE_JOINS.fullmatch(entry[last:match.start()]):
            lists[-1][1].append(match.group(1))
        elif re.search(r'(?:^|\n)[ \t]*$', before) and _CARD_VALUE_ROW.match(entry, match.end()):
            lists.append(('row', [match.group(1)]))
        elif _CARD_VALUE_OPENS.search(before):
            lists.append((before.rstrip()[-1:], [match.group(1)]))
        else:
            last = None
            continue
        last = match.end()
    table = any(kind == 'row' for kind, _ in lists)
    values = []
    for kind, found in lists:
        if not table or kind in ('row', '', '{', ';', ':', '.', '|'):
            values += [value for value in found if value not in values]
    return values


def _clip(text, chars):
    """text cut at a word boundary to at most chars, marked with '...'."""
    if len(text) <= chars:
        return text
    return text[:max(0, chars - 3)].rsplit(' ', 1)[0].rstrip(' ,;:') + '...'


_CARD_OPEN_LIST = re.compile(r'\.\.\.|\be\.g\.|\bsuch as\b|\bfor example\b|\bany\b', re.I)


def _card_argument(name, entry, chars):
    """One keyword's line of a card: its entry clipped to chars, with the values it accepts (_entry_values) that the
    clipped text does not show; at chars 0 only those values ('' with none)."""
    text = re.sub(r'^required\b[.,]?\s*', '', ' '.join(entry.split()), flags=re.I)  # the signature says so
    shown = _clip(text, chars) if chars else ''
    rest = ', '.join(f"'{value}'" for value in _entry_values(entry) if f"'{value}'" not in shown)
    # An open list (e.g., such as, ..., any) is shown as examples, never as the complete set of accepted values.
    open_list = bool(_CARD_OPEN_LIST.search(text))
    if not shown:
        return f'  {name}: ' + ('values include ' if open_list else '') + rest if rest else ''
    return f'  {name}: {shown}' + (f" ({'values include' if open_list else 'values'}: {rest})" if rest else '')


def _returns_keys(paragraph, chars, first=False):
    """A multi-line Returns paragraph cut to the lines that name its keys: its first line and those at the least
    indent below it (first: at the indent of its second line, so prose after the keys at a smaller indent is not
    taken for them), each clipped to chars and marked '...' where its continuation lines are left out; a one-line
    paragraph as it is."""
    lines = [line for line in paragraph.split('\n') if line.strip()]
    if len(lines) < 2:
        return ' '.join(paragraph.split())
    depth = [len(line) - len(line.lstrip()) for line in lines]
    indent = depth[1] if first else min(depth[1:])
    kept = []
    for index in range(1, len(lines)):
        if depth[index] != indent:
            continue
        line = _clip(' '.join(lines[index].split()), chars)
        dropped = index + 1 < len(lines) and depth[index + 1] > indent  # its continuation lines are not shown
        kept.append(line + ' ...' if dropped and not line.endswith('...') else line)
    return ' '.join([lines[0].strip()] + kept)


def library_card(function: str, chars: int = CARD_CHARS) -> str:
    """The LIBRARY CARDS item for one library function, from its own source without importing it: its signature;
    each keyword-only argument with its Parameters entry and the values the entry says it accepts (_card_argument);
    the first paragraph of its Returns section (the documented keys), or the docstring's first paragraph when it
    has none; and its Example, marked illustrative. Over chars it is trimmed: first the example, by whole statements
    from its end; then down CARD_LEVELS, the keywords' entries (an optional one's line is left out) and the Returns
    paragraph to the lines that name its keys; then the required entries to the values they accept, with the Returns
    paragraph again as whole as fits; last the Returns text to the room left. '' when the library does not define it."""
    node = _library_defs().get(function)
    if node is None:
        return ''
    target, arguments = _library_call(node)
    args = target.args if target is not None else None
    sections = _doc_sections(ast.get_docstring(node) or '')
    names = [arg.arg for arg in (args.posonlyargs + args.args + args.kwonlyargs + [args.vararg, args.kwarg]
                                 if args else []) if arg is not None]
    entries = _doc_entries(sections.get('Parameters', []), names)
    keywords = ([(arg.arg, default is not None) for arg, default in zip(args.kwonlyargs, args.kw_defaults)]
                if args else [])
    paragraph = '\n'.join(sections.get('Returns', [])).strip().split('\n\n')[0]  # not the formula after it
    title = '\n'.join(sections.get('', [])).strip().split('\n\n')[0]
    statements = []  # the example's statements: a >>> line with its ... lines and printed output
    for line in next((lines for name, lines in sections.items() if name.lower().startswith('example')), []):
        if line.strip():
            if line.lstrip().startswith('>>>') or not statements:
                statements.append([])
            statements[-1].append(line.rstrip())

    def card(width, returns, example=(), optional=None):
        lines = [f'- `{LIBRARY}.{function}({arguments})`']
        lines += [line for line in (_card_argument(name, entries.get(name, ''), optional if default else width)
                                    for name, default in keywords if optional is not None or not default) if line]
        lines.append(('  Returns: ' if paragraph else '  ') + returns)
        if example:
            lines.append('  Example (values illustrative):')
            lines += ['    ' + line for statement in example for line in statement]
        return '\n'.join(lines)
    whole = ' '.join((paragraph or title).split())
    text = card(CARD_ARG_CHARS, whole, statements, CARD_ARG_CHARS)
    while len(text) > chars and statements:
        statements.pop()
        text = card(CARD_ARG_CHARS, whole, statements, CARD_ARG_CHARS)
    if len(text) <= chars:
        return text
    levels = [(_returns_keys(paragraph, keys) if keys and paragraph else whole, least, optional)
              for keys, least, optional in CARD_LEVELS]
    for returns, least, optional in levels:
        for width in range(CARD_ARG_CHARS, least - 1, -CARD_ARG_STEP):
            text = card(width, returns, optional=optional)
            if len(text) <= chars:
                return text
    for returns, _, _ in levels:  # the values each keyword accepts, which a shorter entry would not show
        text = card(0, returns)
        if len(text) <= chars:
            return text
    return card(0, _clip(returns, max(CARD_LEVELS[-1][0], chars - len(card(0, '')))))  # one line of it stays


def library_cards(functions) -> list:
    """The LIBRARY CARDS items (library_card) of these functions, in order, each once: at most CARDS_LIMIT, each at
    most CARD_CHARS and all at most CARDS_CHARS together; a card the room left cannot hold is left out."""
    cards, room = [], CARDS_CHARS
    for function in dict.fromkeys(functions):
        if len(cards) == CARDS_LIMIT:
            break
        card = library_card(function, min(CARD_CHARS, room))
        if card and len(card) <= room:
            cards.append(card)
            room -= len(card) + 1  # and the line break before the next card
    return cards


def card_functions(prompt, required, instruction) -> list:
    """The functions LIBRARY CARDS shows: those REQUIRED LIBRARY CALLS lists (required), in order, then the
    mid-level ones (LIBRARY_MID_LEVEL) that the prompt's LIBRARY MATCHES lists and the TASK names with strength 2
    or more (library_match_details), in catalog order."""
    listed = set(matched_functions([{'content': prompt}]))
    sources = _library_sources(LIBRARY) if listed else {}
    strong = [function for function, _, _, strength in (library_match_details(instruction) if listed else [])
              if strength >= 2 and function in listed and function in sources
              and sources[function][2] in LIBRARY_MID_LEVEL]
    return list(dict.fromkeys([function for function, _ in required] + strong))


def cards_on() -> bool:
    """T1_CARDS=1 (with T1_ADOPT): the program prompt shows a usage card for each function the program must call
    and each mid-level function the TASK names (with_cards). Off, prompts are as before."""
    return os.environ.get('T1_CARDS') == '1' and not minimal_on() and adopt_on()


def with_cards(prompt, required, instruction, status):
    """The program prompt with LIBRARY CARDS (CARDS_BLOCK, library_cards of card_functions) before the output
    format: after REQUIRED LIBRARY CALLS, whose paragraph then also asks for every keyword argument from the TASK and
    each result read by the keys its card documents (CARDS_REQUIRED), or where that paragraph would stand. The
    prompt as it is without a card, or when the cards would take it to FIRST_PROMPT_CAP. status counts the cards and
    the characters they add (library_cards, library_card_chars)."""
    try:
        cards = library_cards(card_functions(prompt, required, instruction))
    except Exception:  # the cards only help: the prompt goes out without them
        cards = []
    carded = prompt
    if cards:
        head = prompt.removesuffix(OUTPUT_FORMAT)
        paragraph = _required_paragraph(required) if required else ''
        if paragraph and head.endswith(paragraph):
            head = head.removesuffix(paragraph) + paragraph.removesuffix('\n\n') + CARDS_REQUIRED + '\n\n'
        carded = head + CARDS_BLOCK.format(cards='\n'.join(cards)) + OUTPUT_FORMAT
        if len(carded.encode()) >= FIRST_PROMPT_CAP:  # the repairs after it keep their room
            cards, carded = [], prompt
    status.update(library_cards=len(cards), library_card_chars=len(carded) - len(prompt))
    return carded


def s11_libnotes_on() -> bool:
    """T1_S11_LIBNOTES=1 (with T1_S10): LIBRARY NOTES after LIBRARY MATCHES in the task prompt (s11_notes_block), and
    a library TypeError or ValueError that comes back in the repair loop gets the function's documentation in the next
    patch ask (s11_library_error, s11_library_doc). Off, or without T1_S10, prompts are as before."""
    return os.environ.get('T1_S11_LIBNOTES') == '1' and (s10_on() or minimal_on())


def _s11_example(sections, function):
    """The first statement of the function's Example section that calls it, its continuation lines joined and each
    call of a library function in it written `qfkit.<name>(` (a bare one would raise NameError when copied); '' when
    there is none or it is over S11_EXAMPLE_CHARS."""
    statements = []
    for line in next((lines for name, lines in sections.items() if name.lower().startswith('example')), []):
        text = line.strip()
        if text.startswith('>>>'):
            statements.append(text[3:].strip())
        elif statements and (text == '...' or text.startswith('... ')):
            statements[-1] += ' ' + text[3:].strip()
    call = re.compile(rf'(?<![\w.]){re.escape(function)}\(')
    for statement in statements:
        if call.search(statement):
            text = ' '.join(statement.split())
            bare = re.sub(r'\s+#.*$', '', text)  # a trailing comment (the value it prints, a source)
            try:
                text = bare if bare != text and ast.parse(bare) else text
            except (SyntaxError, ValueError, RecursionError, MemoryError):
                pass
            names = '|'.join(map(re.escape, sorted(_library_defs(), key=len, reverse=True)))
            text = re.sub(rf'(?<![\w.])({names})\(', rf'{LIBRARY}.\1(', text)
            return text if len(text) <= S11_EXAMPLE_CHARS else ''
    return ''


def _s11_choice(text, values):
    """(whether the quoted words an entry lists are its argument's own string values, where in its one-line text the
    first list of them ends, where the word before that list that names another type ends, 0 without one). They are
    not when that word stands before the list (S11_NOT_CHOICE), as in `None, or a dict with any of 'a' ...`, `a
    permutation of ('a', 'b')` or `a (K,) array ..., or 'a'`."""
    for match in _CARD_VALUE.finditer(text):
        if values and match.group(1) == values[0] and _CARD_VALUE_OPENS.search(text[:match.start()]):
            end = match.end()
            for later in _CARD_VALUE.finditer(text, end):  # through the list
                if not _CARD_VALUE_JOINS.fullmatch(text[end:later.start()]):
                    break
                end = later.end()
            end += text[end:end + 1] in (')', '}', ']')
            named = S11_NOT_CHOICE.search(text[:match.start()])
            return named is None, end, named.end() if named else 0
    return True, 0, 0


def _s11_clip(text, chars):
    """text clipped to chars (_clip); when that would cut off the first sign or time unit it states (S11_MEANING), the
    clause that states it instead, clipped to chars with '...' in front."""
    match = S11_MEANING.search(text)
    if len(text) <= chars or not match or match.end() <= chars - 4:
        return _clip(text, chars)
    start = max(text.rfind(mark, 0, match.start()) for mark in (': ', '; ', ', ', '. '))
    start = start + 2 if start >= 0 and match.end() - start <= chars - 8 else match.start()
    return '...' + _clip(text[start:], chars - 3)


def _s11_keyword(name, entry, chars, target, mark=''):
    """A keyword's line of a note: its entry clipped to chars (0: none of it; _s11_clip keeps a sign or time unit it
    states in view), then the values it accepts that the clipped text does not show, its entry's (_entry_values) and
    those its function's code tests it for; '' when that leaves nothing. An entry whose quoted words are not its own
    values (_s11_choice: a dict's keys, a tuple's items) shows its text instead of them, at least S11_ENTRY_CHARS and
    through its first list when that ends within S11_ENTRY_MAX, else through the word that names its type; a required
    keyword (mark) that would show nothing keeps S11_MEANING_CHARS of an entry that states a sign or a time unit. An
    open list (e.g., such as, ...) is shown as examples."""
    text = re.sub(r'^required\b[.,]?\s*', '', ' '.join(entry.split()), flags=re.I)
    if mark:  # the mark says so: the entry's own REQUIRED and (required) go
        text = re.sub(r'\s*\(required\)|[,;]?\s*\bREQUIRED\b(?:[.,;]|$)', '', text)
        text = ' '.join(re.sub(r'\(required,\s*', '(', text).split())
    values = _entry_values(entry)
    choice, through, named = _s11_choice(text, values)
    if not choice:  # its quoted words are keys or items: the entry says what they are, through the word that does
        extent = through + 6  # the list's closing word and the cut's '...'
        values, chars = [], max(chars, S11_ENTRY_CHARS, extent if extent <= S11_ENTRY_MAX else named + 6)
    values += _accepted_values(target, name) if target is not None else []
    if mark and not chars and not values and S11_MEANING.search(text):
        chars = S11_MEANING_CHARS
    shown = ((_clip if choice and values else _s11_clip)(text, chars)) if chars else ''
    rest = ', '.join(f"'{value}'" for value in dict.fromkeys(values) if f"'{value}'" not in shown)
    label = 'values include' if _CARD_OPEN_LIST.search(text) else 'values'
    if not shown:
        return f'  {name}{mark}: ' + ('values include ' if label != 'values' else '') + rest if rest else ''
    return f'  {name}{mark}: {shown}' + (f' ({label}: {rest})' if rest else '')


def _s11_condition(entry, chars):
    """The clause of an optional keyword's entry that makes it required in some case (`required for 't' only`),
    clipped to chars; '' when none does."""
    for clause in re.split(r'(?<=[.;:])\s+', ' '.join(entry.split())):
        if re.search(r'\brequired\b', clause, re.I):
            return _clip(clause.rstrip('.;: '), chars)
    return ''


def library_note(function: str, chars: int = S11_LIBNOTES_CHARS, top: int = 0) -> str:
    """The LIBRARY NOTES item of one library function, from its own source without importing it: its signature (every
    keyword-only argument with its default), [vectorised] and the library's example call for a function that
    broadcasts (LIBRARY_VECTORISED), each required keyword with its entry and the values it accepts, each optional
    keyword whose entry makes it required in some case with that clause, the values each other keyword accepts, and
    the first Returns paragraph (the docstring's first paragraph without one). Over chars it is trimmed down
    S11_NOTE_LEVELS from the level top on, then to the signature with the required keywords' lines as the last level
    has them, then with those of the ones whose entry states a sign or a time unit (S11_MEANING) only, the names of
    the others and the Returns text the room leaves (the lines go first, then the names, then the Returns text); ''
    when even the signature is over chars or the library does not define the function."""
    node = _library_defs().get(function)
    if node is None:
        return ''
    target, arguments = _library_call(node)
    args = target.args if target is not None else None
    sections = _doc_sections(ast.get_docstring(node) or '')
    names = [arg.arg for arg in (args.posonlyargs + args.args + args.kwonlyargs + [args.vararg, args.kwarg]
                                 if args else []) if arg is not None]
    entries = _doc_entries(sections.get('Parameters', []), names)
    keywords = list(zip(args.kwonlyargs, args.kw_defaults)) if args else []
    paragraph = '\n'.join(sections.get('Returns', [])).strip().split('\n\n')[0]  # not the formula after it
    title = '\n'.join(sections.get('', [])).strip().split('\n\n')[0]
    whole = _clip(' '.join((paragraph or title).split()), S11_RETURNS_CHARS)
    vector = function in library_vectorised()
    head = f'- `{LIBRARY}.{function}({arguments})`' + (LIBRARY_VECTOR_MARK if vector else '')
    example = _s11_example(sections, function) if vector else ''
    label = '  Returns: ' if paragraph else '  '

    def note(width, condition, optional, keys, call):
        lines, bare = [head] + ([f'  example: `{example}`'] if call and example else []), []
        for arg, default in keywords:
            entry = entries.get(arg.arg, '')
            if default is None:
                line = _s11_keyword(arg.arg, entry, width, target, ' (required)')
                bare += [] if line else [arg.arg]
            elif re.search(r'\brequired\b', entry, re.I):
                clause = _s11_condition(entry, condition)
                line = f'  {arg.arg}: {clause}' if clause else ''
            else:
                line = _s11_keyword(arg.arg, entry, 0, target) if optional else ''
            lines += [line] if line else []
        lines += [f'  required keywords: {", ".join(bare)}'] if bare else []
        return '\n'.join(lines + [label + (_returns_keys(paragraph, keys, True) if keys and paragraph else whole)])
    for level in S11_NOTE_LEVELS[top:]:
        text = note(*level)
        if len(text) <= chars:
            return text
    required = [arg.arg for arg, default in keywords if default is None]
    own = {name: _s11_keyword(name, entries.get(name, ''), 0, target, ' (required)') for name in required}

    def kept(names):  # the lines of these required keywords, then the names of the others
        rest = [name for name in required if name not in names or not own[name]]
        return [own[name] for name in required if name in names and own[name]] + (
            [f'  required keywords: {", ".join(rest)}'] if rest else [])
    every, listed = kept(required), kept(())
    meaning = kept([name for name in required if S11_MEANING.search(entries.get(name, ''))])
    for lines, returns in (([head] + every, True), ([head] + meaning, True), ([head] + meaning, False),
                           ([head] + listed, True), ([head], True), ([head] + listed, False), ([head], False)):
        shown = '\n'.join(lines)
        room = chars - len(shown) - 1 - len(label)
        if returns and room >= 20:
            return shown + '\n' + label + _clip(whole, room)
        if not returns and len(shown) <= chars:
            return shown
    return ''


def library_notes(functions, chars: int = S11_LIBNOTES_CHARS) -> list:
    """The LIBRARY NOTES items (library_note) of these functions, in order, each once, all at most chars together
    with their line breaks. The room goes first to each item's core (its note from S11_CORE_LEVEL on: the Returns
    paragraph cut to the lines that name its keys), the shortest whole items first, each later one at most an equal
    share of the room left; what is left then raises the items still under their core, then each item towards whole,
    shortest first. So one item's Returns prose never costs another's required entries, and a long item is trimmed
    before a short one is dropped."""
    sizes = {function: len(library_note(function, 10 ** 6)) for function in dict.fromkeys(functions)}
    functions = [function for function, size in sizes.items() if size]
    order = sorted(functions, key=lambda name: sizes[name])
    core = {function: len(library_note(function, 10 ** 6, S11_CORE_LEVEL)) for function in order}
    notes, room = {}, chars
    for left, function in zip(range(len(order), 0, -1), order):
        text = library_note(function, min(room // left - 1, core[function]))
        if text:
            notes[function] = text
            room -= len(text) + 1
    for under in (True, False):  # the room left: the items under their core, then towards whole
        for function in order:
            had = notes.get(function)
            room += len(had) + 1 if had else 0
            cap = min(room - 1, core[function]) if under else room - 1
            text = library_note(function, cap) if not under or not had or len(had) < core[function] else had
            notes[function] = text if text and len(text) > len(had or '') else had
            room -= len(notes[function]) + 1 if notes[function] else 0
    return [notes[function] for function in functions if notes.get(function)]


def s11_notes_block(functions) -> str:
    """The LIBRARY NOTES block (S11_LIBNOTES) of these functions (library_notes); '' without an item."""
    items = library_notes(functions)
    return S11_LIBNOTES.format(items='\n'.join(items)) if items else ''


def s11_noted(prompt) -> tuple:
    """(the functions the LIBRARY NOTES block of a prompt notes, in order, the block's characters); ([], 0) without
    one."""
    head = S11_LIBNOTES.split('{items}', 1)[0]
    if not isinstance(prompt, str) or head not in prompt:
        return [], 0
    block = head + prompt.split(head, 1)[1].split('\n\n', 1)[0] + '\n\n'
    return re.findall(r'^- `' + re.escape(LIBRARY) + r'\.([A-Za-z_]\w*)\(', block, re.M), len(block)


_TRACEBACK = 'Traceback (most recent call last):'


def s11_library_error(code: str, exception: str, tail: str = ''):
    """(library function, exception type) of a run's TypeError or ValueError raised inside a public library function:
    the first library frame of the last traceback in the log tail (the whole tail when its cut left out the header)
    names a function the library defines (the program's entry into the library), or, for a TypeError, its message
    names such a function's call (`f() missing 1 required keyword-only argument`) and the program refers to that
    function. None otherwise, and for a program that does not use the library."""
    kind = exception.split(':', 1)[0].rsplit('.', 1)[-1] if exception else ''
    if kind not in ('TypeError', 'ValueError') or LIBRARY not in (code or ''):
        return None
    defs = _library_defs()
    last = (tail or '').rsplit(_TRACEBACK, 1)[-1]  # the whole tail when the cut left out its header
    frames = [function for function in _LIBRARY_FRAME.findall(last) if function in defs]
    if frames:
        return frames[0], kind
    if kind == 'TypeError':
        for function in re.findall(r'(?<![\w.])([A-Za-z_]\w*)\(\)', exception.split(':', 1)[-1]):
            if function in defs and re.search(rf'\b{function}\b', code):
                return function, kind
    return None


def s11_library_doc(function: str, chars: int = S11_LIBDOC_CHARS) -> str:
    """A library function's signature and docstring, read from its source, at most chars: whole when it fits; else its
    title paragraph and the sections S11_LIBDOC_SECTIONS names, in that order (the formulas and references are what
    a call does not need), cut at a line with S11_LIBDOC_CUT when still over. '' when the library does not define
    it."""
    node = _library_defs().get(function)
    if node is None:
        return ''
    head = f'{LIBRARY}.{function}({_library_call(node)[1]})'
    doc = (ast.get_docstring(node) or '').strip()
    text = head + '\n' + doc
    if len(text) <= chars:
        return text
    sections = _doc_sections(doc)
    parts = [head, '\n'.join(sections.get('', [])).strip().split('\n\n')[0]]
    for name in S11_LIBDOC_SECTIONS:
        title = next((title for title in sections if title.lower().startswith(name.lower())), None)
        if title:
            parts.append(f'{title}\n{"-" * len(title)}\n' + '\n'.join(sections[title]).strip('\n'))
    text = '\n'.join(part for part in parts if part)
    if len(text) <= chars:
        return text
    kept, size = [], len(S11_LIBDOC_CUT)
    for line in text.split('\n'):
        if size + len(line) + 1 > chars:
            break
        kept.append(line)
        size += len(line) + 1
    return '\n'.join(kept + [S11_LIBDOC_CUT])


def s11_library_ask(raised) -> str:
    """The patch ask's LIBRARY DOCUMENTATION note (S11_LIBDOC) for raised ((function, exception type) of
    s11_library_error); '' when the library has no documentation for it."""
    doc = s11_library_doc(raised[0])
    return S11_LIBDOC.format(exc=raised[1], lib=LIBRARY, function=raised[0], doc=doc) if doc else ''


def _library_calls(tree, name=LIBRARY):
    """(the program's aliases of the optional library, {name: function} of its `from` imports of library
    functions, the library's public functions)."""
    defs = set(_library_defs(name))
    public = defs & ((library_exports(name) or {}).get('') or defs)
    aliases, names = _library_aliases(tree, name), {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module and \
                node.module.split('.')[0] == name:
            for item in node.names:
                if item.name in public:
                    names[item.asname or item.name] = item.name
    return aliases, names, public


def _called(node, aliases, names, public):
    """The library function a Call node calls, or None."""
    func = node.func
    if isinstance(func, ast.Name):
        return names.get(func.id)
    if isinstance(func, ast.Attribute) and func.attr in public:
        base = func.value
        if isinstance(base, ast.Attribute) and isinstance(base.value, ast.Name) and base.value.id in aliases:
            return func.attr
        if isinstance(base, ast.Name) and base.id in aliases:
            return func.attr
    return None


def _root(node, called):
    """The node a chain of subscripts, attributes and method calls starts from (`f(x)['a'].get('b')` -> f(x)): a
    name, or a call that called(node) names (a library function), or another expression."""
    while True:
        if isinstance(node, (ast.Subscript, ast.Attribute)):
            node = node.value
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and not called(node):
            node = node.func.value
        else:
            return node


_LIBRARY_FRAME = re.compile(r'File "[^"\n]*[/\\]' + re.escape(LIBRARY) + r'[/\\][^"\n]*", line \d+, in (\w+)')


def library_use(code: str, line, exception: str, tail: str = '', report: str = '') -> tuple:
    """Under T1_LIBHINT, (the LIBRARY USE note, the misuse key) for a crash of the optional library's use: a
    KeyError, IndexError, TypeError, ValueError or AttributeError (not a missing module attribute) whose traceback
    passes through a library function, or whose failing line reads the result of a library call (directly, or
    through names bound to it or to parts of it in the same function) in the way the error can come from: a
    KeyError subscripts it with the missing key, an IndexError subscripts it, an AttributeError takes its
    attribute, a ValueError unpacks it; or a TypeError or ValueError message names the library call. The note
    lists each such function's signature, argument entries and Returns section (library_entry), at most 2
    functions (3 when a KeyError's key subscripts 3 results) and LIBRARY_USE_CAP characters; a ValueError the
    library's own checks raise about a value, not a spelling, gets LIBRARY_INPUTS. The key (functions, exception
    line) names the misuse across runs. ('', None) otherwise."""
    kind = exception.split(':', 1)[0].rsplit('.', 1)[-1] if exception else ''
    if not libhint_on() or kind not in LIBRARY_USE_ERRORS or _NO_ATTRIBUTE.search(exception):
        return '', None
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return '', None
    if type(line) is not int or LIBRARY not in code:
        return '', None
    aliases, names, public = _library_calls(tree)
    if not public:
        return '', None

    def called(node):
        return _called(node, aliases, names, public) if isinstance(node, ast.Call) else None
    statements = [node for node in ast.walk(tree) if isinstance(node, ast.stmt)
                  and node.lineno <= line <= (node.end_lineno or node.lineno)]
    if not statements:
        return '', None
    statement = max(statements, key=lambda node: (node.lineno, -(node.end_lineno or node.lineno)))
    parts = [child for child in ast.iter_child_nodes(statement)
             if not isinstance(child, (ast.stmt, ast.excepthandler))]
    scopes = [node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
              and node.lineno <= line <= (node.end_lineno or node.lineno)]
    scope = max(scopes, key=lambda node: node.lineno) if scopes else tree
    raised = [f for f in dict.fromkeys(_LIBRARY_FRAME.findall(tail or '')) if f in public]
    direct = []
    for part in parts:
        for node in ast.walk(part):
            if isinstance(node, ast.Call):
                function = called(node)
                if function and function not in direct:
                    direct.append(function)
    try:  # a KeyError names its key: the results the line subscripts with that key
        key = ast.literal_eval(exception.split(':', 1)[1].strip()) if kind == 'KeyError' else None
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError, IndexError):
        key = None
    # names bound, in the same function, to a library call's result or to part of such a name
    bound = {}
    for _ in range(3):
        for node in ast.walk(scope):
            if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                root = _root(node.value, called)
                function = called(root) or (bound.get(root.id) if isinstance(root, ast.Name) else None)
                if not function:
                    continue
                for target in targets:
                    for item in (target.elts if isinstance(target, (ast.Tuple, ast.List)) else [target]):
                        if isinstance(item, ast.Name):
                            bound.setdefault(item.id, function)

    def result(node):  # the library function whose result (or a bound part of it) node reads, or None
        root = _root(node, called)
        return called(root) or (bound.get(root.id) if isinstance(root, ast.Name) else None)
    # how the line reads library results: subscripted (with the KeyError's key), attribute or unpacked
    reads = {'keyed': [], 'Subscript': [], 'Attribute': [], 'unpack': []}
    for part in parts if not isinstance(statement, (ast.Raise, ast.Assert)) else ():  # not the program's own check
        for node in ast.walk(part):
            found = []
            if isinstance(node, (ast.Subscript, ast.Attribute)):
                found.append((type(node).__name__, result(node.value)))
                if key is not None and isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant) \
                        and node.slice.value == key:
                    found.append(('keyed', result(node.value)))
            if isinstance(statement, ast.Assign) and node is statement.value \
                    and isinstance(statement.targets[0], (ast.Tuple, ast.List)):
                found.append(('unpack', result(node)))
            for how, function in found:
                if function and function not in reads[how]:
                    reads[how].append(function)
    # each error counts only a read it can come from, or a call its message names
    named = [f for f in direct if re.search(rf'\b{f}\(\)', exception)]
    used = {'KeyError': reads['keyed'], 'IndexError': reads['Subscript'], 'AttributeError': reads['Attribute'],
            'ValueError': reads['unpack'] + named,
            'TypeError': named + (reads['Subscript'] if re.search(r'subscriptable|indices', exception) else [])}[kind]
    # the traceback enters the library from the failing line: the call it entered, else the line's calls
    functions = ([f for f in direct if f in raised] or direct) if _LIBRARY_FRAME.search(tail or '') else []
    functions = list(dict.fromkeys(functions + used))[:3 if len(reads['keyed']) > 2 else 2]
    if not functions:
        return '', None
    hint = next((text.strip() for text in (report or '').split('\n') if 'its keys:' in text), '')
    items, size = [], 0
    for function in functions:
        entry = library_entry(function, exception, hint)
        if entry and size + len(entry) <= LIBRARY_USE_CAP:
            items.append(entry)
            size += len(entry) + 1
    if not items:
        return '', None
    inside = bool(raised and set(functions) & set(raised))
    how = ('the error is raised inside ' if inside else 'the failing line uses ') + ', '.join(
        f'`{LIBRARY}.{f}`' for f in functions)
    # a value the library's own checks refused (no quoted spelling, no keyword): not a spelling to correct
    note = LIBRARY_INPUTS if inside and kind == 'ValueError' and not re.search(
        r'[\'"]|keyword', exception.split(':', 1)[-1]) else LIBRARY_USE
    return note.format(how=how, items='\n'.join(items)), (tuple(functions), exception)


def library_again(misuse, names) -> str:
    """The crash request's note that a library misuse (library_use's key) an earlier run had is back."""
    return LIBRARY_AGAIN.format(names=f' (to {", ".join(names)})' if names else '', lib=LIBRARY,
                                function=misuse[0][0], error=_error_note(misuse[1]))


_QUIET_TARGETS = (('sys', 'stderr'), ('warnings', 'warn'), ('warnings', 'showwarning'))


def _quiet_value(node) -> bool:
    return isinstance(node, ast.Constant) and (node.value is False or node.value in ('off', 'ignore'))


def silenced_warnings(code: str) -> int:
    """How many places of a program keep warnings out of its run log: a warnings filter whose action is 'ignore'
    (filterwarnings, simplefilter, catch_warnings(action=...)), catch_warnings(record=True), redirect_stderr, an
    assignment to sys.stderr, warnings.warn or warnings.showwarning, and a keyword argument about warnings set to
    False, 'off' or 'ignore' (show_warning=False). 0 when the program does not parse."""
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return 0
    count = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else func.id if isinstance(func, ast.Name) else ''
            keywords = {k.arg: k.value for k in node.keywords if k.arg}
            action = (node.args[0] if node.args and name in ('filterwarnings', 'simplefilter')
                      else keywords.get('action'))
            if name in ('filterwarnings', 'simplefilter', 'catch_warnings') and _quiet_value(action):
                count += 1
            elif name == 'catch_warnings' and isinstance(keywords.get('record'), ast.Constant) and \
                    keywords['record'].value is True:
                count += 1
            elif name == 'redirect_stderr':
                count += 1
            count += sum(1 for key, value in keywords.items()
                         if key != 'action' and 'warn' in key.lower() and _quiet_value(value))
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            for target in node.targets if isinstance(node, ast.Assign) else [node.target]:
                if (isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name)
                        and (target.value.id, target.attr) in _QUIET_TARGETS):
                    count += 1
    return count


def fit_reports_gone(before: str, after: str, keys: set, health, functions) -> bool:
    """Whether a fix removed a fit report it was asked about (keys: their identities) in a way that counts: the
    report no longer appears in the fix's clean run (health), and the fix either adds no place that keeps warnings
    out of the log (silenced_warnings) or now calls one of the library fit functions named to it (functions). A
    report that is only silenced says nothing about the fit."""
    if not keys - health.keys():
        return False
    return (silenced_warnings(after) <= silenced_warnings(before)
            or any(re.search(rf'\b{name}\b', after) for name in functions))


def _module_names(path: Path):
    """The names a module's source binds at its top level (a star import adds its target's names: None when
    that cannot be read), or None when it defines a module __getattr__ (its names are dynamic)."""
    tree = ast.parse(path.read_text(encoding='utf-8'))
    names = set(_bound_in(tree))
    if '__getattr__' in names:
        return None
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module:
            names.add(node.module.split('.')[0])  # importing from a submodule makes it an attribute
            if any(alias.name == '*' for alias in node.names):
                target = path.parent / (node.module.replace('.', '/') + '.py')
                star = _module_all(target)
                if star is None:
                    return None
                names |= star
        elif isinstance(node, ast.ImportFrom) and any(alias.name == '*' for alias in node.names):
            return None  # a star import from outside the package: unknown names
    names.discard('*')
    return names


def _module_all(path: Path):
    """A module's literal __all__, else its public top-level names; None when unreadable or dynamic."""
    try:
        tree = ast.parse(path.read_text(encoding='utf-8'))
    except (OSError, SyntaxError, ValueError, UnicodeError, RecursionError):
        return None
    for node in tree.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == '__all__'):
            try:
                value = ast.literal_eval(node.value)
            except (ValueError, TypeError, SyntaxError, RecursionError):
                return None
            return {name for name in value if isinstance(name, str)} if isinstance(value, (list, tuple)) else None
    names = _module_names(path)
    return None if names is None else {name for name in names if not name.startswith('_')}


_EXPORTS = {}


def _module_imports(path: Path) -> dict:
    """{name: import line} of the names a module takes from outside its package at its top level."""
    found = {}
    for node in ast.parse(path.read_text(encoding='utf-8')).body:
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module and node.module != '__future__':
            for item in node.names:
                if item.name != '*':
                    found[item.asname or item.name] = (f'from {node.module} import {item.name}'
                                                       + (f' as {item.asname}' if item.asname else ''))
        elif isinstance(node, ast.Import):
            for item in node.names:
                found[item.asname or item.name.split('.')[0]] = (
                    f'import {item.name}' + (f' as {item.asname}' if item.asname else ''))
    return found


def _library_index(name: str):
    """(exports, imports) of the optional library, read from its source once per file version; (None, {}) when it
    is not installed or its names are dynamic. exports: {'': the package's attribute names, '<module>': each
    submodule's} (a submodule whose names are dynamic is left out). imports: {name: the one import line} of the
    names its modules take from outside it, where every module that takes the name takes it the same way."""
    try:
        spec = importlib.util.find_spec(name)
        init = Path(spec.origin)
        if init.name != '__init__.py':
            return None, {}
        key = (str(init), init.stat().st_mtime_ns, s12_lib_on(), library86_on())
        if key not in _EXPORTS:
            package, modules, imports, clashes = _module_names(init), {}, {}, set()
            hidden, skipped = s12_hidden(name), set()
            for path in sorted(init.parent.glob('*.py')):
                if hidden and path.name != '__init__.py' and (_module_all(path) or {None}) <= hidden:
                    skipped.add(path.stem)  # a module only the 0.5.0 CATALOG lists: absent without T1_S12_LIB
                    continue
                if path.name != '__init__.py':
                    names = _module_names(path)
                    if names is not None:
                        modules[path.stem] = names
                    for alias, line in _module_imports(path).items():
                        if imports.setdefault(alias, line) != line:
                            clashes.add(alias)
            imports = {alias: line for alias, line in imports.items() if alias not in clashes}
            if package is not None and hidden:
                package = package - hidden - skipped - {S12_LIB_BASE}
            _EXPORTS.clear()
            _EXPORTS[key] = (None, {}) if package is None else (dict(modules, **{'': package}), imports)
        return _EXPORTS[key]
    except Exception:  # not installed, a namespace directory, or an unreadable source: nothing is checked
        return None, {}


def library_exports(name: str = LIBRARY):
    """{'': the package's attribute names, '<module>': each submodule's} of the optional library, read from its
    source without importing it (as library_catalog); None when it is not installed or its names are dynamic.
    A submodule whose names are dynamic is left out (its references are not checked)."""
    return _library_index(name)[0]


_LIBRARY_SIGNATURES = {}


def _library_signatures(name=LIBRARY):
    """Unwrapped, unambiguous function defs behind each exact exported module name.

    Read trusted source only. Package re-exports follow its top-level relative
    imports; classes, decorated defs, dynamic exports and unknown bindings give
    no signature. The cache includes every source file's version.
    """
    try:
        exports = library_exports(name)
        if not exports:
            return {}
        init = Path(importlib.util.find_spec(name).origin)
        files = sorted(init.parent.glob('*.py'))
        key = (name, str(init), tuple((str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in files),
               s12_lib_on(), library86_on())
        if key not in _LIBRARY_SIGNATURES:
            trees = {p.stem: ast.parse(p.read_text(encoding='utf-8')) for p in files}
            modules = {module: {function: node for function, node in _lintable(tree).items()
                                if isinstance(node, ast.FunctionDef) and function in exports.get(module, ())}
                       for module, tree in trees.items() if module != '__init__'}
            root = {}
            for node in trees['__init__'].body:
                if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module in modules:
                    for item in node.names:
                        imported = (_module_all(init.parent / (node.module + '.py'))
                                    if item.name == '*' else {item.name})
                        if imported is None:
                            return {}
                        for function in imported:
                            alias = function if item.name == '*' else item.asname or function
                            root.pop(alias, None)
                            target = modules[node.module].get(function)
                            if target is not None:
                                root[alias] = target
                else:
                    for bound in _bound_in(ast.Module(body=[node], type_ignores=[])):
                        root.pop(bound, None)
                    if isinstance(node, ast.FunctionDef) and node.name in _lintable(trees['__init__']):
                        root[node.name] = node
            modules[''] = {function: node for function, node in root.items() if function in exports['']}
            _LIBRARY_SIGNATURES.clear()
            _LIBRARY_SIGNATURES[key] = modules
        return _LIBRARY_SIGNATURES[key]
    except (AttributeError, OSError, SyntaxError, ValueError, TypeError, UnicodeError, RecursionError, MemoryError):
        return {}


def library_call_lint(code, limit=LINT_LIMIT, name=LIBRARY):
    """Advisory signature-binding faults of definite exported library calls.

    No generated program or library is executed, and no argument is inserted.
    Only module-level imports have provenance here. Uncertain names, rebindings,
    parameter shadows, escaped library objects, monkey-patched attributes,
    star imports, reflection, *args and **kwargs are left alone. Every definite
    bad site is considered before the usual bounded model-facing list is cut.
    """
    try:
        tree = ast.parse(code)
        compile(tree, '<library-call-lint>', 'exec', dont_inherit=True)
        nodes = list(ast.walk(tree))
        if any(isinstance(node, ast.ImportFrom) and any(item.name == '*' for item in node.names)
               for node in nodes):
            return []
        reflection = {'exec', 'eval', 'globals', 'locals', 'vars', 'getattr', 'setattr', 'delattr', '__import__'}
        if any((isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id in reflection)
               or (isinstance(node, ast.Attribute) and node.attr in reflection)
               or (isinstance(node, ast.ImportFrom) and node.module == 'builtins'
                   and any(item.name in reflection for item in node.names)) for node in nodes):
            return []
        # The module registry exposes the same object without the library alias.
        # Its indirect reads/stores are outside this finite binding analysis.
        if any((isinstance(node, ast.Attribute) and node.attr == 'modules')
               or (isinstance(node, ast.ImportFrom) and node.module == 'sys'
                   and any(item.name == 'modules' for item in node.names)) for node in nodes):
            return []
        signatures = _library_signatures(name)
        if not signatures:
            return []
        bindings = Counter()
        for node in nodes:
            if isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load):
                bindings[node.id] += 1
            elif isinstance(node, ast.arg):
                bindings[node.arg] += 1
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                bindings[node.name] += 1
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                bindings.update((item.asname or item.name).split('.')[0] for item in node.names)
            elif isinstance(node, (ast.ExceptHandler, ast.MatchAs, ast.MatchStar)) and node.name:
                bindings[node.name] += 1
            elif isinstance(node, ast.MatchMapping) and node.rest:
                bindings[node.rest] += 1
        top_imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
        top_modules = {(item.asname or item.name).split('.')[0]
                       for node in top_imports if isinstance(node, ast.Import) for item in node.names}
        aliases = {alias: module for alias, module in _library_aliases(tree, name).items()
                   if bindings[alias] == 1 and alias in top_modules}
        imported = {}
        for node in top_imports:
            if isinstance(node, ast.ImportFrom) and not node.level and node.module:
                parts = node.module.split('.')
                if parts[0] == name and len(parts) <= 2:
                    module = parts[1] if len(parts) == 2 else ''
                    for item in node.names:
                        alias = item.asname or item.name
                        if bindings[alias] == 1 and item.name in signatures.get(module, {}):
                            imported[alias] = (module, item.name)
        # Stores through an imported namespace/function can alter the callable.
        mutated = set()
        for node in nodes:
            if isinstance(node, (ast.Attribute, ast.Subscript)) and not isinstance(node.ctx, ast.Load):
                base = node.value
                while isinstance(base, (ast.Attribute, ast.Subscript)):
                    base = base.value
                if isinstance(base, ast.Name):
                    mutated.add(base.id)
        aliases = {alias: module for alias, module in aliases.items() if alias not in mutated}
        imported = {alias: entry for alias, entry in imported.items() if alias not in mutated}

        def called(node):
            function, module = None, ''
            if isinstance(node.func, ast.Name) and node.func.id in imported:
                module, function = imported[node.func.id]
            elif isinstance(node.func, ast.Attribute):
                base = node.func.value
                if isinstance(base, ast.Name) and base.id in aliases:
                    module, function = aliases[base.id], node.func.attr
                elif (isinstance(base, ast.Attribute) and isinstance(base.value, ast.Name)
                      and aliases.get(base.value.id) == ''
                      and base.attr in (library_exports(name) or {}).get('', ())):
                    module, function = base.attr, node.func.attr
            return (module, function) if function in (library_exports(name) or {}).get(module, ()) else None

        # A namespace or function read outside an exact direct call can escape
        # to another name or opaque mutator. Do not assume its original signature
        # still applies; no general object/data-flow evaluation is attempted.
        parents = {child: node for node in nodes for child in ast.iter_child_nodes(node)}
        escaped = set()
        for node in nodes:
            if not isinstance(node, ast.Name) or not isinstance(node.ctx, ast.Load) \
                    or node.id not in aliases.keys() | imported.keys():
                continue
            current = node
            parent = parents.get(current)
            while isinstance(parent, ast.Attribute) and parent.value is current:
                current, parent = parent, parents.get(parent)
            if not isinstance(parent, ast.Call) or parent.func is not current or called(parent) is None:
                escaped.add(node.id)
        if escaped:
            # Different imports can expose the same module/function object;
            # unknown mutation through one may invalidate every old signature.
            return []
        lines, found = code.split('\n'), []
        for node in nodes:
            if not isinstance(node, ast.Call):
                continue
            entry = called(node)
            target = signatures.get(entry[0], {}).get(entry[1]) if entry is not None else None
            # The existing helper treats a positional-only name in **kwargs as
            # a duplicate; Python permits that separate keyword, so abstain.
            if target is not None and target.args.kwarg is not None and any(
                    keyword.arg in {arg.arg for arg in target.args.posonlyargs} for keyword in node.keywords):
                continue
            problem = _arity(node, target) if target is not None else None
            if problem:
                found.append((node.lineno, f'line {node.lineno}: `{lines[node.lineno - 1].strip()[:120]}` - {problem}'))
        return [text for _, text in sorted(set(found))][:limit]
    except (SyntaxError, ValueError, TypeError, RecursionError, MemoryError):
        return []


def _library_aliases(tree, name):
    """{alias: submodule ('' for the package)} of the names the program binds to the library by `import`, when
    nothing else binds them."""
    aliases, other = {}, set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for item in node.names:
                parts = item.name.split('.')
                if parts[0] == name and (item.asname is None or len(parts) <= 2):
                    alias = item.asname or name
                    aliases[alias] = (parts[1] if item.asname and len(parts) == 2 else '')
                else:
                    other.add((item.asname or item.name).split('.')[0])
        elif isinstance(node, ast.ImportFrom):
            other.update(item.asname or item.name for item in node.names)
        elif isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load):
            other.add(node.id)
        elif isinstance(node, ast.arg):
            other.add(node.arg)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            other.add(node.name)
    return {alias: module for alias, module in aliases.items() if alias not in other}


def _dunder(name):
    return name.startswith('__') and name.endswith('__')


def library_references(code: str, exports=None, name: str = LIBRARY, imports=None):
    """(edits, findings) for the program's references to names the optional library does not have, read without
    running it. edits [(line, start byte, end byte, text)]: `lib.X` where the program binds X itself at its top
    level becomes `X` (a scipy or numpy name taken for the library's); where the program binds X nowhere but the
    library's own modules import X one way (imports), that import line is added before the program's first
    statement too. `lib.mod.X` where only the package has X becomes `lib.X`, as does `from lib.mod import X...`
    when the package has every name. findings: the others, ['line N: `source` - why'], with the closest names the
    library has. ([], []) when the library or the program cannot be read."""
    if exports is None:
        exports, found_imports = _library_index(name)
        imports = found_imports if imports is None else imports
    imports = imports or {}
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return [], []
    if not exports or name not in code:
        return [], []
    package, lines = exports[''], code.split('\n')
    own, anywhere = set(_bound_in(tree)), set()  # bound at the top level; bound anywhere at all
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load):
            anywhere.add(node.id)
        elif isinstance(node, ast.arg):
            anywhere.add(node.arg)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            anywhere.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            anywhere.update((item.asname or item.name).split('.')[0] for item in node.names)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            anywhere.add(node.name)
    first = next((node for node in tree.body if not (
        isinstance(node, ast.ImportFrom) and node.module == '__future__') and not (
        isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str))),
        None)
    added = set()
    edits, findings = [], []

    def text(node):
        if node.lineno != node.end_lineno:
            return ''
        return lines[node.lineno - 1].encode()[node.col_offset:node.end_col_offset].decode(errors='replace')

    def report(node, where, missing, known):
        close = difflib.get_close_matches(missing, sorted(known), n=3)
        hint = f'; close: {", ".join(f"`{c}`" for c in close)}' if close else ''
        if missing in imports:
            hint += f'; the library itself takes it by `{imports[missing]}`'
        findings.append((node.lineno, f'line {node.lineno}: `{lines[node.lineno - 1].strip()[:120]}` - `{where}` has '
                                      f'no `{missing}`{hint}'))
    try:
        aliases = _library_aliases(tree, name)
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in aliases:
                module, attr = aliases[node.value.id], node.attr
                known = package if not module else exports.get(module)
                if known is None or _dunder(attr) or attr in known:
                    continue
                exact = text(node) == f'{node.value.id}.{attr}'
                if not module and attr in own and exact:
                    edits.append((node.lineno, node.col_offset, node.end_col_offset, attr))
                elif not module and exact and attr not in anywhere and attr in imports and first is not None:
                    edits.append((node.lineno, node.col_offset, node.end_col_offset, attr))
                    if attr not in added:  # once, before the program's first statement
                        added.add(attr)
                        top = min([d.lineno for d in getattr(first, 'decorator_list', [])] + [first.lineno])
                        edits.append((top, 0, 0, imports[attr] + '\n'))
                else:
                    report(node, name if not module else f'{name}.{module}', attr, known)
            elif (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Attribute)
                  and isinstance(node.value.value, ast.Name) and aliases.get(node.value.value.id) == ''
                  and node.value.attr in exports and node.value.attr in package):
                module, attr = node.value.attr, node.attr
                if _dunder(attr) or attr in exports[module]:
                    continue
                alias = node.value.value.id
                if attr in package and text(node) == f'{alias}.{module}.{attr}':
                    edits.append((node.lineno, node.col_offset, node.end_col_offset, f'{alias}.{attr}'))
                else:
                    report(node, f'{name}.{module}', attr, exports[module])
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                parts = node.module.split('.')
                if parts[0] != name or len(parts) > 2:
                    continue
                module = parts[1] if len(parts) == 2 else ''
                known = package if not module else exports.get(module)
                if known is None:
                    continue
                missing = [item.name for item in node.names if item.name != '*' and not _dunder(item.name)
                           and item.name not in known and not (not module and item.name in exports)]
                if not missing:
                    continue
                line = lines[node.lineno - 1]
                pattern = re.compile(r'^(\s*from\s+)' + re.escape(node.module) + r'(\s+import\b)')
                match = pattern.match(line)
                if (module and node.lineno == node.end_lineno and match
                        and all(item.name in package for item in node.names)):
                    start, end = len(line[:match.end(1)].encode()), len(line[:match.start(2)].encode())
                    edits.append((node.lineno, start, end, name))
                else:
                    report(node, node.module, missing[0], known)
    except RecursionError:
        return [], []
    return edits, [item for _, item in sorted(set(findings))]


def rewrite_library_refs(code: str, exports=None, name: str = LIBRARY, imports=None):
    """(code, references rewritten): library_references' edits applied, when the result still compiles; the
    program unchanged otherwise. Zero requests; a reference it changes would fail with AttributeError or
    ImportError as written."""
    edits = library_references(code, exports, name, imports)[0]
    if not edits:
        return code, 0
    lines = code.split('\n')
    for row, start, end, replacement in sorted(edits, reverse=True):
        raw = lines[row - 1].encode()
        lines[row - 1] = (raw[:start] + replacement.encode() + raw[end:]).decode()
    fixed = '\n'.join(lines)
    changed = sum(1 for _, start, end, _ in edits if end > start)  # the references, not the import lines added
    return (fixed, changed) if _compiles(fixed) else (code, 0)


def _raise_rule(selfcheck: str = '') -> str:
    """The task prompt's rule for errors: NO_FALLBACK, and RESULT_CHECKS under T1_RAISEGUARD. Under T1_S11_RAISE the
    narrowed rule in their place: raise only for a missing input, filter and count (S11_RAISE_INPUT), RESULT_CHECKS,
    and time grids from the TASK's parameters (S11_RAISE_GRID)."""
    checks = RESULT_CHECKS.format(where=RESULT_CHECKS_WHERE[not selfcheck])
    if s11_raise_on():
        return S11_RAISE_INPUT + checks + S11_RAISE_GRID
    return NO_FALLBACK + (checks if raiseguard_on() else '')


def _task_prompt(task_dir: Path, out: Path, instruction: str, profile: str, hints: str, files: list,
                 library: str = '', selfcheck: str = '', output_format: str = OUTPUT_FORMAT) -> str:
    availability = ('Available: numpy,pandas,scipy,pyarrow,matplotlib,statsmodels,plotly,sklearn,arch,'
                    'polars,seaborn,openpyxl,numba,talib,backtrader,tqdm. Use Agg for plots. '
                    if library_allowed_by_task(instruction) else
                    'DEPENDENCY LIMIT: Use only the dependencies the TASK permits. ')
    return ('Solve this quantitative-finance coding task. '
            + ('Return exactly one complete Python program in one python fence. '
               'This program will be executed once; there are no hidden ' if output_format == OUTPUT_FORMAT
               else 'Build the program in the bounded phases described below. '
               'The assembled program will run inside the isolated executor; there are no hidden ')
            + 'tests available to you. Do not access networks or install packages. Use the supplied data '
            'and honor the task\'s formulas, units, dates, seed, output filenames and methods. '
            + _raise_rule(SELFCHECK_BLOCK if SELFCHECK_BLOCK in selfcheck else '') + f'Read inputs under {task_dir}. {hints}'
            + (f'Input files (absolute paths): {", ".join(files)}. ' if files else '') +
            f'Write required deliverables under {out}. The root filesystem is read-only; use /tmp '
            'for small temporary files. Never write reward.json or pytest_report.json. Do not copy '
            'the task text or input metadata (such as _canary fields) into output files. ' + availability +
            'Read full files locally when the sample is insufficient.\n\n'
            'TASK:\n'+instruction+'\n\nINPUT PROFILE:\n'+profile
            +'\n\n'+library+consistency_note(instruction)+selfcheck+output_format)


def first_prompt(task_dir: Path, out: Path, instruction: str, profile: dict, hints: str, canaries,
                 status: dict, limit: int = FIRST_PROMPT_CAP, selfcheck: bool = False,
                 output_format: str = OUTPUT_FORMAT) -> str:
    """The task prompt, always under limit bytes and never raising, so the first request is sent. It is
    trimmed only when too large: the optional library's catalog, then the profile to paths and sizes, then no
    profile, then the /app hints to their directory fallbacks, then the end of the instruction (with a marker).
    status['prompt_trim'] records the step (an enum) and status['library_shown'] whether the catalog stayed."""
    files=[str(task_dir/row['path']) for row in profile['files']]
    compact={'files':[{'path':row['path'],'bytes':row.get('bytes')} for row in profile['files']]}
    if 'unprofiled_files' in profile:
        compact['unprofiled_files']=profile['unprofiled_files']
    omitted='(omitted to fit the prompt limit; list the input directory yourself)'

    def fallback_hints():
        try:
            return input_hints(task_dir, limit=0)
        except Exception:
            return ''
    try:
        library, matched=library_parts(instruction)
    except Exception:  # the catalog only helps: the prompt goes out without it
        library, matched='', 0
    if not library_allowed_by_task(instruction):
        status['library_restricted'] = True
    runtime_guide = (runtime.PROMPT + runtime.requirement_prompt() if runtime.enabled() and runtime.available()
                     and runtime.allowed(instruction) else '')
    status['library_shown']=False
    block=SELFCHECK_BLOCK if selfcheck else ''  # the self-check contract is never trimmed
    stages=(('none', lambda: (json.dumps(profile,ensure_ascii=False), hints, files, library)),
            ('library_omitted', lambda: (json.dumps(profile,ensure_ascii=False), hints, files, '')),
            ('profile_compact', lambda: (json.dumps(compact,ensure_ascii=False), hints, files, '')),
            ('profile_omitted', lambda: (omitted, hints, [], '')),
            ('hints_trimmed', lambda: (omitted, fallback_hints(), [], '')))
    for stage, parts in stages:
        if stage=='library_omitted' and not library:
            continue
        text, stage_hints, stage_files, stage_library = parts()
        prompt=withhold_canaries(_task_prompt(task_dir,out,instruction,text,stage_hints,stage_files,stage_library,
                                              block + (runtime_guide if stage == 'none' else ''), output_format),canaries)
        if len(prompt.encode()) < limit:
            status.update(prompt_trim=stage, library_shown=bool(stage_library) and library_allowed_by_task(instruction))
            if runtime.enabled():
                status['runtime_shown'] = bool(runtime_guide) and stage == 'none'
            if matched and stage_library:
                status['library_matches']=matched
                noted, chars=s11_noted(stage_library)  # T1_S11_LIBNOTES: the functions it notes, its characters
                if noted:
                    status.update(s11_libnotes=len(noted),s11_libnotes_chars=chars)
            return prompt
    # Withheld before the cut, so no partial canary can survive it.
    data=withhold_canaries(instruction,canaries).encode()
    keep=len(data)
    while True:
        keep=max(0, keep-(len(prompt.encode())-limit)-200)
        cut=data[:keep].decode('utf-8','ignore')
        text=cut+f'\n[instruction truncated: {len(data)-len(cut.encode())} bytes omitted]\n'
        prompt=withhold_canaries(_task_prompt(task_dir,out,text,omitted,stage_hints,[],'',block,output_format),canaries)
        if len(prompt.encode()) < limit or keep == 0:
            status['prompt_trim']='instruction_truncated'
            return prompt


def solve(task_dir: Path, out: Path, *, budget_sec: float=STAGE_CAP, budget_source: str='caller') -> dict:
    """Run only inside the image. Never invoke this on the host with generated code. Every guard runs inside
    the try, and the status is always written: the scorer reads outputs, never an exit code."""
    CLOCK.reset()  # one unit per process: nothing is credited yet, so the clock reads time.monotonic()
    throttle=eval_throttle()  # local evaluation only; the official image never sets it
    started=CLOCK.now()
    valid_budget=type(budget_sec) in (int, float) and math.isfinite(budget_sec) and budget_sec > 0
    # Numbers, booleans and fixed enums only: no code, names, paths or model text.
    status={'baseline':'B3_repair_feedback','official_score':None,'model_calls':0,
            'status':'not_started','generation_replies':0,'continuations':0,'rewrites':0,'fill_requests':0,
            'repairs':0,'patch_requests':0,'path_rewrites':0,
            'budget_sec_applied':round(float(budget_sec),1) if valid_budget else None,
            'budget_source':budget_source,'controller_protection':'not_attempted',
            'solver_identity':solver_profile.identity()}
    client=None; program_request={}; replay=_NO_REPLAY; runtime_session=None
    minimal_active=minimal_on()
    s10=s10_on() and not thin_on()  # Thin's early return must also suppress legacy telemetry in finally.
    try:
        # The image marker plus a non-root uid; /.dockerenv is absent under containerd or podman.
        if os.environ.get('T1_PHASE1_CONTAINER') != '1' or os.getuid() == 0:
            raise HouseError('Generated-code execution requires the dedicated container as a non-root user',
                             kind='local_guard', code='container_guard')
        try:
            cap = parse_request_cap(os.environ.get('T1_REQUEST_CAP'), default=MAX_REQUESTS)
        except ValueError:
            raise HouseError('Invalid request resource cap', kind='configuration',
                             code='invalid_request_cap') from None
        status['request_cap'] = cap
        try:
            protect_controller()  # before credentials are loaded or code is generated
            status['controller_protection']='enabled'
        except Exception:
            status['controller_protection']='unavailable'
        out.mkdir(parents=True,exist_ok=True)
        replay=Replay(out)
        replay.put('solver-identity.json',json.dumps(status['solver_identity'],sort_keys=True))
        if 'T1_REQUEST_CAP' in os.environ:
            replay.put('request-budget.json',json.dumps({'request_cap':cap,
                       'default_request_cap':MAX_REQUESTS,'source':'environment'},sort_keys=True))
        if 'T1_SOLVER_PROFILE' in os.environ:
            problem=solver_profile.startup_problem(status['solver_identity'])
            status['solver_profile_gate']=problem or 'accepted'
            if problem:
                raise HouseError('Solver profile configuration is invalid', kind='configuration',
                                 code='invalid_configuration')
        config=HouseConfig.from_env()
        if not valid_budget:
            raise HouseError('Invalid local solve budget', kind='local_budget', code='solve_budget')
        profile, canaries, hints = _profile(task_dir, status)
        try:
            text=(task_dir/'instruction.md').read_bytes().decode('utf-8', errors='replace')
        except OSError:
            raise HouseError('The task instruction could not be read',
                             kind='local_input', code='instruction_unreadable') from None
        instruction=task_instruction(text, canaries)
        if thin_on():
            from . import thin
            if budget_sec-(CLOCK.now()-started) <= 20:
                raise HouseError('No generation time after input preparation', kind='local_budget',
                                 code='generation_budget')
            client=HouseClient(config,max_requests=cap); client.replay=replay
            program_request=thin.solve(client,task_dir,out,instruction,profile,hints,canaries,
                                       started+budget_sec-20,status,replay)
            return status
        if runtime.enabled():
            permitted = runtime.allowed(instruction) and runtime.available()
            status['runtime_enabled'] = permitted
            if permitted:
                runtime_session = runtime.start(instruction)
                status.update(runtime.semantic_context())
        files=[str(task_dir/row['path']) for row in profile['files']]
        # The image turns the acceptance check on; it replaces the in-program self-check. T1_S10 turns both off.
        accept=accept_on() and not s10 and not minimal_active
        selfcheck=os.environ.get('T1_SELFCHECK') == '1' and not accept and not s10  # the image turns it on
        triage=s10_triage(instruction,profile) if s10 else None  # T1_S10: word caps and budgets, no request
        if s10:
            s10_triage_status(status,triage)
        if s11_raise_on():  # T1_S11_RAISE: which arm a run report reads
            status['s11_raise']=True
        if s12_post_on():  # T1_S12_POST: which arm a run report reads, and its frozen texts
            status.update(s12_post=True,s12_prompt_sha=S12_PROMPT_SHA[:12])
        checks=SelfChecks(keep=output_contract(instruction)['files'],named=instruction) if selfcheck else None
        health=FitWarnings()  # what the last clean run reported about its fits, for the review
        generation_format = minimal.OUTPUT_FORMAT if minimal_active else OUTPUT_FORMAT
        if minimal_active:
            generation_format += resource_control.context(resource_control.snapshot())
        prompt=first_prompt(task_dir,out,instruction,profile,hints,canaries,status,selfcheck=selfcheck,
                            output_format=generation_format)
        if budget_sec-(CLOCK.now()-started) <= 20:
            raise HouseError('No generation time remaining after input preparation',
                             kind='local_budget', code='generation_budget')
        deadline=started+budget_sec-20  # the final reserve covers status writing and cleanup
        replay.prompt(prompt)
        plain_prompt=prompt  # the fresh program's prompt: the task without the plan
        client=HouseClient(config,max_requests=cap); client.replay=replay
        pace=Pace()
        required=[]  # T1_ADOPT: the library functions the plan maps, which the program prompt requires
        routed=None  # T1_ROUTE: the plan's whole text, routed after the cards
        noted=None  # T1_S10: the notes stage's carried state (spec §4)
        if minimal_active:
            messages=[{'role':'user','content':prompt}]
            code=minimal.generate(client,messages,deadline,status,canaries=canaries,pace=pace)
        else:
            if os.environ.get('T1_PLAN') == '1' and s10:  # DECIDE in place of the plan, then REQUIRED from its map
                noted=s10_decide(client,prompt,triage,deadline,status,pace,canaries=canaries,replay=replay)
                if s11_union_on():  # T1_S11_UNION: a second DECIDE sample, merged into the first's lines
                    noted=s11_decide_b(client,prompt,triage,noted,deadline,status,pace,instruction=instruction,
                                       canaries=canaries,replay=replay)
                if noted['plan']:  # DECIDE left no notes: stack9's fallback plan, on its with_plan path
                    prompt=with_plan(prompt,noted['plan'])
                    replay.put('plan.txt',noted['plan'])
                prompt,required=s10_adopt(prompt,noted,instruction,status)
                if route_on():
                    routed=noted['full'] or ''
            elif os.environ.get('T1_PLAN') == '1':  # the image turns planning on; local harnesses leave it off
                full=[] if adopt_on() or route_on() else None
                plan=plan_notes(client,prompt,deadline,status,pace,canaries=canaries,facts=selfcheck or accept,
                                full=full)
                if plan:
                    prompt=with_plan(prompt,plan)
                    replay.put('plan.txt',plan)
                if full and adopt_on():
                    counts={}
                    required,dropped=library_map(full[0],instruction,matched_functions([{'content':prompt}]),counts)
                    status.update(adopt_required=len(required),adopt_none_dropped=dropped,
                                  adopt_none_different=counts.get('different',0))
                    prompt=with_required(prompt,required)
                if route_on():
                    routed=full[0] if full else ''
            if cards_on():  # after the plan request, which stays as it was
                prompt=with_cards(prompt,required,instruction,status)
            if noted is not None:  # CHECK and STATE read the program prompt with REQUIRED and its cards
                noted=s10_refine(client,prompt,instruction,noted,deadline,status,pace,canaries=canaries,replay=replay)
            # T1_ROUTE: family note and METHOD PINS follow REQUIRED and its cards.
            if routed is not None:
                prompt,plain_prompt=route_prompts(prompt,plain_prompt,routed,status,instruction)
            if noted is not None and noted['block']:  # last before the output format; the fresh program never sees it
                prompt=with_s10_notes(prompt,noted['block'])
                replay.put('notes.txt',noted['block'])
            messages=[{'role':'user','content':prompt}]
            if s10:  # the tier's code budget; r1think and its plan gate are not used
                think=s10_budget('code',triage['tier'],status) or None
                status['s10_code_budget']=0
                if s12_mech_on() and getattr(client,'s12_note_failed',None) is not None:  # the House left a note
                    think=None; status['s12_code_think_off']=True  # unanswered notes: no code reasoning
            else:
                think=r1think(client,status,deadline,pace)  # T1_R1THINK: the first program request's reasoning budget
            tagged=set() if s10 else None  # the decisions the first program's replies name, before the merge drops any
            outputs=([PurePosixPath(path).name for path in output_contract(instruction)['files']]
                     if s12_mech_on() else ())  # T1_S12_MECH: what a continuation still has to write
            code=generate_program(client,messages,deadline,status,canaries=canaries,pace=pace,think=think,tags=tagged,
                                  outputs=outputs)
            if s10:
                status['s10_d_tagged_first']=len(tagged)
        code=precheck_fixed(code,status,(SELFCHECK_NAME,) if checks is not None else ())
        program_request={k:client.diagnostics.get(k) for k in ('http_status','finish_reason')}
        ran, measured, repaired, measured_request = set(), None, False, program_request
        last_failure, streak, before, last_exec, failed_lines = None, 0, code, EXEC_FLOOR, []
        misuses=set()  # library misuses (library_use) of earlier runs
        library_errors=set()  # T1_S11_LIBNOTES: (library function, exception type) of earlier runs' crashes
        path_index=input_path_index(task_dir)
        while True:
            run_started=CLOCK.now()
            runtime_kept = None
            if runtime.active() and measured is not None and measured.get('runtime_checks'):
                runtime_kept = _snapshot(out)
                if runtime_kept is None:
                    status['repair_stop'] = 'runtime_snapshot_limit'
                    code, program_request = before, measured_request
                    break
            try:
                code, moved = rewrite_input_paths(code, task_dir, path_index)
                _count(status, 'input_path_rewrites', moved)
                code = library_fixed(code, status)
                status['program_sha256']=hashlib.sha256(code.encode()).hexdigest(); ran.add(_shape(code))
                replay.put('program.py',code)
                cap=(max(RUN_CAP_FLOOR,deadline-CLOCK.now()-pace.full()-2*EXEC_FLOOR)
                     if status.get('timeout_repairs',0)<MAX_TIMEOUT_REPAIRS else None)
                tail, report=_run_program(code,out,deadline,canaries,status,clear=measured is not None,
                                          replay=replay,input_root=task_dir,cap=cap,checks=checks,health=health)
                status.setdefault('first_run_exit_ok',status['status']=='executed_ungraded')
                if runtime_kept is not None and runtime.regression(measured, status):
                    _restore(out, runtime_kept)
                    for key in MEASURED:
                        status.pop(key, None)
                    status.update(measured, repair_stop='runtime_regression')
                    code, program_request = before, measured_request
                    replay.put('program.py', code)
                    break
            except Exception as exc:
                if measured is None:
                    raise
                # Keep the last measured run, its outputs and its request fields, never an agent error.
                for key in MEASURED:
                    status.pop(key, None)
                status.update(measured, repair_stop='time_gate' if isinstance(exc, HouseError) else 'setup_error')
                if runtime_kept is not None:
                    _restore(out, runtime_kept)
                    code = before
                program_request=measured_request
                break
            last_exec=CLOCK.now()-run_started
            status['repairs']+=repaired
            slow=bool(status.get('run_capped'))
            if status['status']!='execution_failed' and not slow:
                break
            if slow:  # one speed repair; its rerun is not capped
                measured={k:status[k] for k in MEASURED if k in status}; measured_request=program_request
                status['timeout_repairs']=status.get('timeout_repairs',0)+1
                left=deadline-CLOCK.now()
                try:
                    new=repair_program(client,messages,code,tail,None,[],deadline,status,EXEC_FLOOR+10,
                                       canaries=canaries,pace=pace,
                                       normalize=lambda text: prepared(text,task_dir,path_index),
                                       ask=TIMEOUT.format(sec=last_exec,left=left,tail=tail.strip()[-3000:]))
                except (_Stop, HouseError):
                    new=None
                if new is None or _shape(prepared(new,task_dir,path_index)) in ran:
                    status['repair_stop']='noop' if new is not None else 'time_gate'
                    break
                program_request={k:client.diagnostics.get(k) for k in ('http_status','finish_reason')}
                before, code, repaired = code, new, True
                continue
            measured={k:status[k] for k in MEASURED if k in status}; measured_request=program_request
            line=status.get('execution_diagnostics',{}).get('generated_line')
            failure=((status.get('execution_diagnostics',{}).get('exception_type'),)+_failing_statement(code,line)
                     +(_exception_line(report,tail),))
            previous=last_failure
            same=bool(repaired and _same_crash(failure,previous,before))  # only a model patch can repeat one
            streak, last_failure = (streak+1 if same else 0), failure
            if failure[1] and failure[2]:
                failed_lines.append(failure[1:])
            history=''
            use, misuse=library_use(code,line,failure[3],tail,report)
            again=bool(misuse and repaired and misuse in misuses)  # a patch left or brought back the misuse
            if misuse:
                misuses.add(misuse)
            if same:
                _count(status,'repair_same_crash')
                names=_changed_names(before,code)
                own=s11_own_raise(code,line,failure[3])  # T1_S11_RAISE: the cause is the check, not elsewhere
                history=('' if again else
                         S11_OWN_AGAIN.format(names=f' (to {", ".join(names)})' if names else '',
                                              exc=failure[3].split(':',1)[0].rsplit('.',1)[-1],
                                              error=_error_note(failure[3])) if own
                         else same_failure(failure, previous, names))
            if use:
                _count(status,'repair_library_use')
                if again:
                    _count(status,'repair_library_again')
                history=use+(library_again(misuse,_changed_names(before,code)) if again else '')+history
            raised=s11_library_error(code,failure[3],tail) if s11_libnotes_on() else None
            if raised:  # T1_S11_LIBNOTES: the same library error after a patch gets the function's documentation
                doc=s11_library_ask(raised) if repaired and raised in library_errors else ''
                if doc:
                    _count(status,'s11_libdoc')
                    history+=doc
                library_errors.add(raised)
            guarded=(guard_raise(code,line,failure[3].split(':',1)[0].rsplit('.',1)[-1])
                     if same and failure[3] and failure[3]==previous[3] and raiseguard_on()
                     and status.get('raise_guarded',0)<MAX_RAISE_GUARDS else None)
            if guarded is not None:  # the program's own check, raising as before the patch: it now only warns
                if deadline-CLOCK.now() < last_exec+10:  # the same gate as a patch rerun
                    status['repair_stop']='time_gate'
                    break
                _count(status,'raise_guarded'); before, code, repaired = code, guarded, False
                continue
            if streak>=2:  # two patches in a row left the same crash (_same_crash): the retry did not help either
                status['repair_stop']='repeated_failure'
                break
            try:
                fixed=(rewrite_missing_path(code,tail,files) if status['path_rewrites']<MAX_PATH_REWRITES
                       else None)
                if fixed is not None:
                    if deadline-CLOCK.now() < last_exec+10:  # the same gate as a patch rerun
                        status['repair_stop']='time_gate'
                        break
                    status['path_rewrites']+=1; before, code, repaired = code, fixed, False
                    continue
                if status['repairs']>=MAX_REPAIRS:
                    status['repair_stop']='max_repairs'
                    break
                new=repair_program(client,messages,code,tail,status.get('exit_code'),
                                   files if _MISSING_FILE.search(tail) or 'FileNotFoundError' in tail else [],
                                   deadline,status,last_exec+10,canaries=canaries,pace=pace,report=report,
                                   history=history,near=line,
                                   normalize=lambda text: prepared(text,task_dir,path_index),
                                   ran=ran,failed=failed_lines)
                if new is None or _shape(prepared(new,task_dir,path_index)) in ran:
                    status.update(repair_noop=True, repair_stop='noop')
                    break
            except _Stop as stop:
                status['repair_stop']=stop.reason
                break
            except HouseError as exc:
                status.update(repair_stop='request_failed', repair_failure_code=exc.code)
                if type(client.diagnostics.get('http_status')) is int:
                    status['repair_http_status']=client.diagnostics['http_status']
                break
            except Exception:  # the measured run stands; nothing after it was executed
                status['repair_stop']='repair_error'
                break
            program_request={k:client.diagnostics.get(k) for k in ('http_status','finish_reason')}
            before, code, repaired = code, new, True
        fresh_on=os.environ.get('T1_FRESH') == '1' and not minimal_active

        def fresh_program():
            nonlocal code, messages, program_request
            code, messages, fields = fresh_attempt(client, plain_prompt, code, messages, out, instruction, deadline,
                                                   status, pace, last_exec, canaries=canaries, replay=replay,
                                                   task_dir=task_dir, path_index=path_index, ran=ran,
                                                   checks=checks, health=health)
            if fields is not None:
                program_request = fields
        if fresh_on and status.get('status') == 'execution_failed' and status.get('repair_stop') in FRESH_DEAD_ENDS:
            fresh_program()
        if status.get('status') == 'executed_ungraded':  # the literal formats the TASK states, before review
            normalize_literals(instruction, out, status)
        if status.get('status') == 'executed_ungraded' and accept:
            code, fields, last_exec = accept_outputs(client, messages, code, out, instruction, deadline, status,
                                                     pace, last_exec, canaries=canaries, replay=replay,
                                                     task_dir=task_dir, path_index=path_index, checks=checks,
                                                     health=health)
            if fields is not None:
                program_request = fields
        if status.get('status') == 'executed_ungraded' and os.environ.get('T1_REVIEW') == '1':
            code, fields = review_outputs(client, messages, code, out, instruction, deadline, status, pace,
                                          last_exec, canaries=canaries, replay=replay, task_dir=task_dir,
                                          path_index=path_index, checks=checks,
                                          rounds=REVIEW_ROUNDS if selfcheck or accept else 1, health=health,
                                          model=not s10 and not minimal_active)
            if fields is not None:
                program_request = fields
            if (fresh_on and 'fresh_status' not in status and status.get('review_contract_flags', 0) > 0
                    and status.get('review_status') in ('no_patch', 'fix_rejected')):
                fresh_program()
        if status.get('status') == 'executed_ungraded':  # idempotent: a kept fix or fresh program's outputs
            normalize_literals(instruction, out, status)
        if status.get('status') == 'executed_ungraded' and (s12_post_on() or runtime.active()):
            try:
                code, fields, last_exec = s12_post(client, messages, code, out, instruction, deadline, status, pace,
                                                   last_exec, canaries=canaries, replay=replay, task_dir=task_dir,
                                                   path_index=path_index, health=health, files=files)
            except Exception:  # a verification step that fails never costs the outputs: each run restores its own
                fields = None
                status['s12_stop'] = 'failed'
            if fields is not None:
                program_request = fields
        if status.get('status') == 'executed_ungraded' and s13_refine_on():
            try:
                code, last_exec = s13_refine(code, out, instruction, deadline, status, last_exec, canaries=canaries,
                                             replay=replay, task_dir=task_dir, health=health)
            except Exception:
                status['s13_refine'] = 'failed'
        if s10:  # a metric only: the decisions the final program's comments name (fewer after a patch's merge)
            status['s10_d_tagged']=s10_tags(code)
        if required:  # the final program's static check
            status['adopt_missing']=len(library_uncalled(code,[function for function,_ in required]))
        if replay.root is not None:  # local evaluation only: what the graded outputs looked like
            replay.put('outputs.txt', withhold_canaries(digest(out), canaries))
        # g2 scans every output file: keep only program_sha256, and report (never rewrite)
        # any output that carries an input canary.
        status.update(output_canary_files(out,canaries))
        total=sum(p.stat().st_size for p in out.rglob('*') if p.is_file() and not p.is_symlink())
        if total>63*1024*1024:
            status.update(status='output_limit_exceeded', failure_kind='local_limit', failure_code='output_limit')
    except HouseError as exc:
        categories={'model_output':'generation_failed', 'local_input':'agent_input_error',
                    'local_budget':'agent_budget_exhausted', 'local_limit':'agent_resource_limit',
                    'configuration':'not_run_invalid_house_config', 'local_guard':'not_run_container_guard'}
        category=('not_run_missing_house' if exc.code=='missing_configuration' else
                  categories.get(exc.kind,'infrastructure_house_error'))
        status.update(status=category,failure_kind=exc.kind,failure_code=exc.code)
    except Exception as exc:
        name=type(exc).__name__
        status.update(status='agent_error',reason=name if name in _BUILTIN_ERRORS else 'Exception')
    finally:
        if runtime_session is not None:
            runtime_session.close()
        if client is not None:
            diagnostics=dict(client.diagnostics)
            details=diagnostics.get('transport_requests')
            if isinstance(details,list):
                diagnostics['transport_requests']=[safe_transport_detail(detail) for detail in details]
            if status['status'] in _EXECUTED:  # the request fields describe the program that ran
                diagnostics.pop('code_line', None)
                for key, value in program_request.items():
                    if value is None:
                        diagnostics.pop(key, None)
                    else:
                        diagnostics[key]=value
            _derive_counters(status, diagnostics['requests'],
                             refused_uncounted=getattr(client,'throttle',False) is True)
            status['model_calls']=client.attempts
            status['house_diagnostics']=diagnostics
            status['usage']=diagnostics.get('usage',{})
            status['content_chars']=sum(r.get('content_chars',0) for r in diagnostics['requests'])
            if s10:  # the breaker's fields over the whole unit: code, patch and flags-fix replies feed it too
                s10_client_status(client, status)
        if throttle:  # local evaluation only: the waits credited back and the refused 429s not counted
            status['eval_pause_credited_sec']=round(CLOCK.credited,3)
            status['eval_uncounted_429']=client.uncounted if client is not None else 0
        status['duration_sec']=round(time.monotonic()-started,3)  # wall-clock time, pauses included
        text=json.dumps(status,indent=2)+'\n'
        if len(text.encode())>STATUS_SOFT_LIMIT and client is not None and throttle:
            # The refused 429s are counted in eval_uncounted_429, so their records are the first to go.
            records=status['house_diagnostics']['requests']
            admitted=[i for i,r in enumerate(records) if r.get('http_status')!=429]
            if len(admitted)<len(records):
                diagnostics=dict(status['house_diagnostics'], requests=[records[i] for i in admitted])
                details=diagnostics.get('transport_requests')
                if isinstance(details,list) and len(details)==len(records):
                    diagnostics['transport_requests']=[details[i] for i in admitted]
                elif 'transport_requests' in diagnostics:
                    diagnostics.pop('transport_requests')
                    status['transport_details_dropped']=True
                status['house_diagnostics']=diagnostics
                status['eval_429_records_dropped']=True
                text=json.dumps(status,indent=2)+'\n'
        if len(text.encode())>STATUS_SOFT_LIMIT and client is not None:
            # Optional budget detail must not crowd out actual attempts, caps, or token usage.
            optional=('continuation_need_sec','continuation_remaining_sec',
                      'continuation_reserve_sec','continuation_rate_tok_s')
            diagnostic=status['house_diagnostics']
            records=diagnostic.get('requests',[])
            if any(key in record for record in records for key in optional):
                status['house_diagnostics']=dict(diagnostic, requests=[
                    {key:value for key,value in record.items() if key not in optional} for record in records])
                status['continuation_budget_details_dropped']=True
                text=json.dumps(status,indent=2)+'\n'
        if (len(text.encode())>STATUS_SOFT_LIMIT and client is not None
                and 'transport_requests' in status['house_diagnostics']):
            # Extra observation must not displace the legacy request/usage record.
            status['house_diagnostics']={k:v for k,v in status['house_diagnostics'].items()
                                         if k!='transport_requests'}
            status['transport_details_dropped']=True
            text=json.dumps(status,indent=2)+'\n'
        if len(text.encode())>STATUS_SOFT_LIMIT and client is not None:
            status['house_diagnostics']={k:v for k,v in status['house_diagnostics'].items()
                                         if k not in ('requests','transport_requests')}
            status['requests_dropped']=True
            text=json.dumps(status,indent=2)+'\n'
        out.mkdir(parents=True,exist_ok=True)
        _write_audit(out/'_agent_status.json',text)
    return status
