"""Generic output sanity checks (S3-01 study, S3-02 module): properties any delivered result should have.

The five checks are exactly the pre-registered candidates (FP/triage/SANITY_CANDIDATES.md v1). Each one looks only at
file shapes, numbers and fixed phrases in our own program text: never a task, file, column or key name's meaning,
never the task wording, never an expected value. They read our own outputs, our own program, the output contract the
prompt already states (audit.output_contract) and the task's input files.

    header_only     a contract .csv/.tsv deliverable that is not blank but has no data row after its first row
    json_all_null   a contract JSON deliverable whose template keys all exist and whose leaf keys are all null
    copied_input    a delivered file byte-identical to an input file
    constant_column a contract table deliverable with a numeric column of one value while an input table varies
    unfinished      the program says it is unfinished in a comment or string (placeholder implementation, ...)

A finding is evidence for one repair request, never a verdict and never a veto: the caller keeps the outputs.
Every reader here is bounded and never raises on a bad file; a file it cannot read is not judged.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import tokenize
from pathlib import Path

CHECKS = ('header_only', 'json_all_null', 'copied_input', 'constant_column', 'unfinished')
DELIMITERS = ',;\t|'
SNIFF_CHARS = 64 * 1024  # the head csv.Sniffer reads (whole lines)
TABLE_BYTES = 16 * 1024 * 1024  # a table larger than this is not judged
TABLE_ROWS = 200_000  # nor one with more rows
HASH_BYTES = 256 * 1024 * 1024  # inputs and outputs hashed per call, in total
JSON_BYTES = 16 * 1024 * 1024
PROGRAM_CHARS = 2 * 1024 * 1024
# Candidate 5 (RPL-2, narrow): phrases a program uses to say it is not finished. "dummy" or "placeholder" alone
# do not count.
PHRASES = ('placeholder implementation', 'dummy values', 'dummy data', 'not implemented')  # any case
CASED_PHRASES = ('TODO',)  # this case only
TABLE_SUFFIXES = ('.csv', '.tsv')
HARNESS_FILES = ('_agent_status.json',)  # written by the solver itself, not a delivered result
_TEXT_TOKENS = {tokenize.COMMENT, tokenize.STRING} | {getattr(tokenize, 'FSTRING_MIDDLE', tokenize.STRING),
                                                      getattr(tokenize, 'TSTRING_MIDDLE', tokenize.STRING)}


def read_table(path):
    """(rows, delimiter) of a table read with Python csv (delimiter: csv.Sniffer among , ; tab |, else a comma), or
    None when the file is missing, blank (0 bytes or only whitespace: OS-1 covers those), too large or unreadable."""
    try:
        path = Path(path)
        if not path.is_file() or path.stat().st_size > TABLE_BYTES:
            return None
        text = path.read_bytes().decode('utf-8-sig', errors='replace')
    except OSError:
        return None
    if not text.strip():
        return None
    sample = text[:SNIFF_CHARS]
    if len(text) > SNIFF_CHARS and '\n' in sample:
        sample = sample[:sample.rfind('\n') + 1]
    try:
        delimiter = csv.Sniffer().sniff(sample, delimiters=DELIMITERS).delimiter
    except csv.Error:
        delimiter = ','
    rows = []
    try:
        for row in csv.reader(io.StringIO(text, newline=''), delimiter=delimiter):
            rows.append(row)
            if len(rows) > TABLE_ROWS:
                return None
    except csv.Error:
        return None
    return rows, delimiter


def _filled(row):
    return any(cell.strip() for cell in row)


def data_rows(rows):
    """The rows after the first one that have a non-blank cell."""
    return [row for row in rows[1:] if _filled(row)]


def header_only(rows):
    """Candidate 1: the table has a first row and no non-blank row after it."""
    return bool(rows) and not data_rows(rows)


def _number(cell):
    try:
        value = float(cell.strip())
    except (ValueError, AttributeError):
        return None
    return value if math.isfinite(value) else None


def numeric_columns(rows):
    """{column index: [values]} of the columns where there are at least 2 data rows and every data row's cell
    reads as a finite float."""
    body = data_rows(rows)
    if len(body) < 2:
        return {}
    found = {}
    for index in range(max(len(row) for row in body)):
        values = [_number(row[index]) if index < len(row) else None for row in body]
        if all(value is not None for value in values):
            found[index] = values
    return found


def constant_columns(rows):
    """Candidate 4, output side: indexes of numeric columns (numeric_columns) that take a single value."""
    return sorted(index for index, values in numeric_columns(rows).items() if len(set(values)) == 1)


def varying_columns(rows):
    """Candidate 4, input side: whether some numeric column takes 2 or more values."""
    return any(len(set(values)) >= 2 for values in numeric_columns(rows).values())


def _walk(obj, path):
    """Values at a template key path, as audit._walk reads them ('*' any key, '[]' the first list element);
    None when the path is missing, [] when an empty container stops it."""
    from . import audit
    return audit._walk(obj, path)


def leaf_keys(keys):
    """The key paths that are no other key path's proper prefix (an object's keys, not the object)."""
    keys = {tuple(key) for key in keys}
    return sorted(key for key in keys if not any(len(other) > len(key) and other[:len(key)] == key for other in keys))


def json_all_null(data, keys):
    """Candidate 2: every template key path exists, and every value at every leaf key path is JSON null (at least
    one value). A container key holds its leaves, so only leaves can be null while the keys all exist."""
    keys = [tuple(key) for key in keys]
    if not keys:
        return False
    if any(_walk(data, key) is None for key in keys):
        return False
    for key in leaf_keys(keys):
        values = _walk(data, key)
        if not values or any(value is not None for value in values):
            return False
    return True


def unfinished_phrases(source):
    """Candidate 5: the fixed phrases found in the program's comment and string tokens (f-string text included).
    A program that does not tokenize is not judged."""
    if not isinstance(source, str) or not source or len(source) > PROGRAM_CHARS:
        return []
    found = set()
    try:
        for token in tokenize.generate_tokens(io.StringIO(source).readline):
            if token.type not in _TEXT_TOKENS:
                continue
            lowered = token.string.lower()
            found.update(phrase for phrase in PHRASES if phrase in lowered)
            found.update(phrase for phrase in CASED_PHRASES if phrase in token.string)
    except (tokenize.TokenError, SyntaxError, ValueError):
        return []
    return sorted(found)


class _Hasher:
    """sha256 of files within HASH_BYTES in total; a file past the budget is not hashed (None)."""

    def __init__(self, budget=HASH_BYTES):
        self.left = budget

    def __call__(self, path):
        try:
            size = Path(path).stat().st_size
            if size > self.left:
                return None
            self.left -= size
            digest = hashlib.sha256()
            with open(path, 'rb') as stream:
                for block in iter(lambda: stream.read(1 << 20), b''):
                    digest.update(block)
            return digest.hexdigest()
        except OSError:
            return None


def _files(root):
    try:
        return sorted(p for p in Path(root).rglob('*') if p.is_file() and not p.is_symlink())
    except OSError:
        return []


def findings(out, contract, inputs, program=None, checks=CHECKS):
    """[{'check', 'file', 'detail'}] for the outputs in `out`: `contract` is audit.output_contract's dict (files,
    json), `inputs` the task's input file paths (Path), `program` the executed program's source, `checks` the
    checks to run (CHECKS by default). Ordered by check, then file. Never raises on a bad file."""
    out = Path(out)
    checks = tuple(c for c in CHECKS if c in set(checks))
    found = []
    tables = [name for name in contract.get('files', ()) if Path(name).suffix.lower() in TABLE_SUFFIXES]
    read = {}

    def table(name):
        if name not in read:
            read[name] = read_table(out / name)
        return read[name]

    if 'header_only' in checks:
        for name in tables:
            got = table(name)
            if got and header_only(got[0]):
                found.append({'check': 'header_only', 'file': name, 'detail': ''})
    if 'json_all_null' in checks:
        for name, keys in sorted(contract.get('json', {}).items()):
            path = out / name
            try:
                if not keys or not path.is_file() or path.stat().st_size > JSON_BYTES:
                    continue
                data = json.loads(path.read_bytes().decode('utf-8', errors='replace'))
            except (OSError, ValueError, RecursionError):
                continue
            if json_all_null(data, keys):
                found.append({'check': 'json_all_null', 'file': name, 'detail': ''})
    input_paths = [Path(p) for p in inputs]
    if 'copied_input' in checks:
        digest = _Hasher()
        known = {}
        for path in input_paths:
            value = digest(path)
            if value is not None:
                known.setdefault(value, path)
        for path in _files(out):
            if path.name in HARNESS_FILES:
                continue
            value = digest(path) if known else None
            if value is not None and value in known:
                found.append({'check': 'copied_input', 'file': path.relative_to(out).as_posix(),
                              'detail': known[value].name})
    if 'constant_column' in checks:
        flat = []
        for name in tables:
            got = table(name)
            if got:
                flat += [(name, index) for index in constant_columns(got[0])]
        if flat and any(varying_columns(got[0]) for got in (read_table(p) for p in input_paths
                                                            if p.suffix.lower() in TABLE_SUFFIXES) if got):
            for name, index in flat:
                header = read[name][0][0]
                label = header[index].strip() if index < len(header) and header[index].strip() else f'#{index + 1}'
                found.append({'check': 'constant_column', 'file': name, 'detail': label})
    if 'unfinished' in checks and program:
        phrases = unfinished_phrases(program)
        if phrases:
            found.append({'check': 'unfinished', 'file': '', 'detail': ', '.join(phrases)})
    return found


# The property each check names in its feedback (no expected value, no task wording).
PROPERTY = {
    'header_only': '{file} has no rows after its first row; if an empty result is not what the task asks for, '
                   'compute the rows, otherwise keep it',
    'json_all_null': '{file}: every required key is present but every value is null',
    'copied_input': '{file} is byte-identical to the input file {detail}, so nothing was computed into it',
    'constant_column': '{file}: numeric column {detail} has the same value in every row although the input tables '
                       'vary',
    'unfinished': 'the program marks itself unfinished ({detail} in a comment or string)',
}


def describe(finding):
    """One feedback line naming the generic property a finding violates."""
    return 'sanity: ' + PROPERTY[finding['check']].format(**finding)


# S3-02: the checks that passed the S3-01 gate (FP/s3/SANITY_STUDY.md), each behind its own T1_THIN_FEATURES name so
# gate D judges each one alone. The other three stay here for the study only and are never wired.
FEATURES = {'sanity': ('header_only',), 'sanity_constant': ('constant_column',)}
LABELS = ('header_only', 'constant_column', 'header_only+constant_column')  # thin_sanity_check values
# thin_sanity_outcome: the next passing run is clean or still breaks the property, or no run passed after the
# repair request and the questioned run stands.
OUTCOMES = ('passed_clean', 'passed_flagged', 'kept')
# thin_sanity_repair_stop (outcome kept): how the repair itself ended; public_eval derives the value set from its
# thin_stop values without internal_checks_passed, plus these two.
REPAIR_STOPS = ('repair_run_not_passed', 'request_spent')


def enabled_checks():
    """The checks T1_THIN_FEATURES turns on, read through thin._feature (R2's switch). Fails closed: a name the
    switch does not know (it raises) or any other error means that check is off, never a crash of a unit that
    already passed."""
    from . import thin
    feature = thin._feature
    found = []
    for name, checks in FEATURES.items():
        try:
            on = bool(feature(name))
        except Exception:
            on = False
        if on:
            found += checks
    return tuple(found)


def wired_findings(contract, out, task_dir, profile, program):
    """findings() for the enabled checks on a run's outputs: the inputs are the profiled input files (the files the
    prompt's INPUT PROFILE lists). [] when no check is on or anything fails: a sanity check never stops a unit."""
    checks = enabled_checks()
    if not checks:
        return []
    try:
        inputs = [Path(task_dir) / row['path'] for row in (profile or {}).get('files', ())
                  if isinstance(row, dict) and isinstance(row.get('path'), str)]
        return findings(out, contract, inputs, program, checks=checks)
    except Exception:
        return []


def label(found):
    """The thin_sanity_check value of a non-empty findings() list."""
    return '+'.join(check for check in CHECKS if any(f['check'] == check for f in found))
