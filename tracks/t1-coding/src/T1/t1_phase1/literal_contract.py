"""Output-contract slips the TASK's own literal text exposes (thin switch literal_contract).

findings(instruction, expected, files) reads the TASK and the bytes of a clean run's outputs and returns short lines
naming a column or key the TASK spells out for one output file that the file lacks or spells differently. The host's
internal check (audit.contract_flags) only tests that the named files exist, parse and carry the keys of a fenced JSON
template; these lines add, from the TASK text alone:

    1. a CSV output whose header row lacks a column the TASK lists for that file, or spells it differently only in
       case, underscores, hyphens or spaces ('Date' for `date`); and a first header cell that is empty or 'Unnamed: 0'
       (an index written as a column) where the TASK lists the file's columns. The lists are audit.contract_rules'
       statements (a 'columns'/'header' lead-in with backticked names, a fence or indented block of names, a markdown
       table of columns), plus bulleted lists of backticked names under a 'columns:'/'keys:' lead-in (_bullet_lists).
    2. a JSON output whose object lacks a key of the TASK's fenced template (audit.output_contract) but holds one that
       differs from it only in case, underscores, hyphens or spaces; and a key a bulleted 'keys:' list gives for that
       file that no object anywhere in the file holds (a key the file holds at any depth counts as present).

Only literal names count, each tied to exactly one output file (the file named on the lead-in, just above it in its
paragraph, or in its heading); a list in an input section or about an input file, under a negated, optional, sorting
or index lead-in, or two complete lists of one file that disagree give nothing. Extra columns and keys and their order
are never judged. It never edits or deletes a file and never raises (any internal error gives [])."""
from __future__ import annotations

import csv
import io
import json
from pathlib import PurePosixPath
import re

from . import audit

MAX_LINES = 8
MAX_CHARS = 240
NAMES_SHOWN = 8  # listed names a line shows before '...'
HEADER_BYTES = 64 * 1024  # a CSV header row must end within this many bytes
JSON_BYTES = 32 * 1024 * 1024  # a larger JSON output is not read
JSON_NODES = 1_000_000

_FILE = re.compile(r'\b([A-Za-z0-9_\-]+\.(?:csv|json))\b')
_DATA_FILE = re.compile(r'(?:/app/data|/input|\bdata)/(?:[\w.\-]+/)*([\w.\-]+\.(?:csv|json))\b')
_LIST_LEAD = re.compile(r'\b(?:columns?|headers?|fields?|keys?)\b', re.I)
_CSV_LEAD = re.compile(r'\b(?:columns?|headers?|fields?)\b', re.I)
_JSON_LEAD = re.compile(r'\b(?:keys?|fields?)\b', re.I)
_INPUT_LEAD = re.compile(r'\binputs?\b|\b(?:read|reads|load|loads|loaded|source|raw)\b', re.I)
_BULLET = re.compile(r'^\s*(?:[-*+]|\d{1,2}[.)])\s+(.*\S)\s*$')
_ITEM = re.compile(r'`([^`\n]+)`\**(.*)$')  # an item that starts with a backticked name (bold or not)
_ITEM_END = re.compile(r'\s*(?:$|[:=(,;]|[-–—]+\s)')  # what may follow the name: its description
_OPTIONAL = re.compile(r'\boptional(?:ly)?\b|\bif (?:any|available|present|applicable|needed|requested)\b'
                       r'|\bmay be (?:omitted|empty|absent|left out|dropped|skipped)\b|\bnot required\b|\bomit',
                       re.I)
_UNNAMED = re.compile(r'Unnamed: \d+')
_SPELLING = re.compile(r'[\s_\-]+')


def findings(instruction, expected, files) -> list[str]:
    """Short lines (at most MAX_LINES of at most MAX_CHARS characters, no repeats) naming each column or key the TASK
    spells out for an output file in `files` (relative path -> bytes) that the file lacks or misspells. `expected` is
    audit.output_contract(instruction): only its declared files are judged (every output when it declares none), and
    its JSON template keys. [] when nothing is found or on any internal error."""
    try:
        listed = _listed(instruction, expected.get('files') or ())
        found = []
        for line in _file_findings(listed, expected, files):
            line = line if len(line) <= MAX_CHARS else line[:MAX_CHARS - 3] + '...'
            if line not in found:
                found.append(line)
        return found[:MAX_LINES]
    except Exception:
        return []


def _target(lines, fenced, sections, i, inputs):
    """The one output CSV or JSON file a lead-in on line i is about, as audit._csv_target finds a CSV: named on the
    line, else on the lines just above it in its paragraph, else in its nearest heading. None for none, two, an input
    file or a lead-in in an input section."""
    if any(audit._INPUT_HEADING.search(lines[h]) for h in sections[i]):
        return None
    names = _FILE.findall(lines[i])
    j = i - 1
    while not names and j >= 0 and i - j <= 3 and lines[j].strip() and j not in fenced:
        names = _FILE.findall(lines[j])
        if audit._HEADING.match(lines[j]):
            break
        j -= 1
    if not names and sections[i]:
        names = _FILE.findall(lines[sections[i][-1]])
    names = {PurePosixPath(n).name for n in names}
    if len(names) != 1:
        return None
    name = names.pop()
    return None if name in inputs else name


def _item(text):
    """The names one bullet gives: '`a`, `b`' or a leading '`name`' and its description; None when the bullet is no
    literal name (prose, a pattern such as `<ticker>_beta`, a value word, an optional entry)."""
    text = text.strip().lstrip('*').strip()
    if audit._ONLY_SPANS.fullmatch(text):
        names = audit._names(audit._SPAN.findall(text))
        return names if names and not any(n in audit._ELLIPSIS for n in names) else None
    match = _ITEM.match(text)
    if not match or not _ITEM_END.match(match.group(2)) or _OPTIONAL.search(match.group(2)):
        return None
    name = match.group(1).strip()
    if not audit._COLUMN_NAME.fullmatch(name) or name.lower() in audit._LITERAL:
        return None
    return [name]


def _bullet_lists(lines, fenced, sections, inputs):
    """(target, names, complete) for each bulleted list of backticked names right under a lead-in that ends with ':'
    and speaks of columns, headers, fields or keys (not negated, not about sorting, grouping, an index or an input).
    Items are the bullets at the list's own indentation (deeper lines are their descriptions or values); the list ends
    at a blank line or at any other line. complete: every item was one literal name."""
    found = []
    for i, line in enumerate(lines):
        lead = line.rstrip().rstrip('*').rstrip()
        if (i in fenced or not lead.endswith(':') or not _LIST_LEAD.search(lead) or audit._NEGATION.search(lead)
                or audit._NOT_A_HEADER.search(lead) or _INPUT_LEAD.search(lead)):
            continue
        names, whole, indent = [], True, None
        k = i + 2 if i + 1 < len(lines) and not lines[i + 1].strip() else i + 1  # one blank line may come first
        while k < len(lines) and k not in fenced and lines[k].strip():
            depth = len(lines[k]) - len(lines[k].lstrip())
            bullet = _BULLET.match(lines[k])
            if indent is None and bullet:
                indent = depth
            if indent is None or depth < indent or depth == indent and not bullet:
                break
            if depth == indent:
                item = _item(bullet.group(1))
                whole = whole and item is not None
                names += [n for n in item or () if n not in names]
            k += 1
        target = _target(lines, fenced, sections, i, inputs) if names else None
        kind = _CSV_LEAD if target and target.endswith('.csv') else _JSON_LEAD
        if target and kind.search(lead):
            found.append((target, names, whole))
    return found


def _listed(instruction, declared):
    """{output file name: names the TASK lists for it, in its order}: audit.contract_rules' header statements (with
    the indented blocks T1_CONTRACT2 reads) and _bullet_lists, each counting as complete or partial. Names of all the
    statements about one file add up; a file with a statement that is no clean list (a '...' that is not a counter
    run, a repeated name) or with two complete statements that disagree is left out."""
    lines = instruction.splitlines()
    fenced = audit._fenced_lines(lines)
    sections = audit._sections(lines, fenced)
    # The input files, as contract_rules finds input CSVs: named under a data path, or in an input section and not
    # declared as an output.
    in_sections = {PurePosixPath(n).name for i, line in enumerate(lines) if i not in fenced
                   and any(audit._INPUT_HEADING.search(lines[h]) for h in sections[i]) for n in _FILE.findall(line)}
    inputs = ({PurePosixPath(n).name for n in _DATA_FILE.findall(instruction)}
              | (in_sections - {PurePosixPath(n).name for n in declared}))
    statements = [(target, names, True) for target, names, _, _ in
                  audit._header_statements(lines, fenced, sections, inputs)
                  + audit._indented_statements(lines, fenced, sections, inputs)]
    statements += _bullet_lists(lines, fenced, sections, inputs)
    listed, complete, dropped = {}, {}, set()
    for target, names, whole in statements:
        if names is None or len(set(names)) != len(names) or any(n in audit._ELLIPSIS for n in names):
            dropped.add(target)
            continue
        if whole and complete.setdefault(target, set(names)) != set(names):
            dropped.add(target)
        listed.setdefault(target, [])
        listed[target] += [n for n in names if n not in listed[target]]
    return {target: names for target, names in listed.items() if target not in dropped}


def _spelling(name):
    return _SPELLING.sub('', name).casefold()


def _near(name, candidates):
    """The first candidate that differs from `name` only in case, underscores, hyphens or spaces, or None."""
    want = _spelling(name)
    return next((c for c in candidates if isinstance(c, str) and c != name and _spelling(c) == want), None)


def _shown(names):
    shown = ', '.join(f'`{n}`' for n in names[:NAMES_SHOWN])
    return shown + (f' and {len(names) - NAMES_SHOWN} more' if len(names) > NAMES_SHOWN else '')


def _header(data):
    """The first row of a CSV's bytes, or None when it is no text or does not end within HEADER_BYTES."""
    chunk = data[:HEADER_BYTES]
    try:
        text = chunk.decode('utf-8-sig')
    except UnicodeDecodeError as exc:
        if exc.start < len(chunk) - 4:
            return None
        text = chunk[:exc.start].decode('utf-8-sig')  # a character cut at the chunk's end
    reader = csv.reader(io.StringIO(text, newline=''))
    try:
        row = next(reader, None)
        if len(data) > HEADER_BYTES and next(reader, None) is None:
            return None  # the header row may go on past the chunk
    except csv.Error:
        return None
    return row if row and any(cell.strip() for cell in row) else None


def _csv_findings(name, data, columns):
    header = _header(data)
    if header is None:
        return []
    found = []
    first = header[0].strip()
    if not first or _UNNAMED.fullmatch(first):
        cell = 'empty' if not first else f'`{first}`'
        found.append(f'{name}: the first header cell is {cell}, an index written as a column; the TASK lists the '
                     f'columns as {_shown(columns)}')
    missing = []
    for column in columns:
        if column in header:
            continue
        near = _near(column, header)
        if near is None:
            missing.append(column)
        else:
            found.append(f'{name}: column `{near}` must be spelled `{column}` as the TASK writes it')
    if missing:
        found.append(f'{name}: lacks column{"s" if len(missing) > 1 else ""} the TASK lists for it: {_shown(missing)}')
    return found


def _all_keys(value):
    """Every string key of every object in a JSON value; None past JSON_NODES nodes."""
    keys, stack, seen = [], [value], 0
    while stack:
        item = stack.pop()
        seen += 1
        if seen > JSON_NODES:
            return None
        if isinstance(item, dict):
            keys += [k for k in item if isinstance(k, str)]
            stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(item)
    return keys


def _json_findings(name, data, paths, keys):
    if len(data) > JSON_BYTES:
        return []
    try:
        value = json.loads(data.decode('utf-8-sig'))
    except (UnicodeDecodeError, ValueError, RecursionError):
        return []
    found = []
    for path in sorted(paths):  # template keys a near spelling stands in for (contract_flags names the missing ones)
        last = path[-1]
        if last in ('*', '[]'):
            continue
        for where, node in audit._dicts_at(value, path[:-1]):
            near = None if last in node else _near(last, node)
            if near is not None:
                found.append(f'{name}: key `{audit._dotted(where + (near,))}` must be spelled `{last}` as the '
                             f"TASK's template writes it")
                break
    templated = {segment for path in paths for segment in path}
    listed = [k for k in keys if k not in templated]
    held = _all_keys(value) if listed else []
    if held is None or listed and not held and not isinstance(value, dict):
        return found  # too large, or no object to judge (an empty list is stub_check's)
    present = set(held)
    missing = []
    for key in listed:
        if key in present:
            continue
        near = _near(key, held)
        if near is None:
            missing.append(key)
        else:
            found.append(f'{name}: key `{near}` must be spelled `{key}` as the TASK writes it')
    if missing:
        found.append(f'{name}: lacks key{"s" if len(missing) > 1 else ""} the TASK lists for it: {_shown(missing)}')
    return found


def _file_findings(listed, expected, files):
    """The lines for each output CSV or JSON file whose name no other output shares and that is declared (or every
    output when none is)."""
    declared = {PurePosixPath(n).name for n in expected.get('files') or ()}
    templates = expected.get('json') or {}
    outputs = {}
    for name in files:
        path = PurePosixPath(name)
        if path.suffix.lower() in ('.csv', '.json') and path.name not in audit._NOT_OURS:
            outputs.setdefault(path.name, []).append(name)
    found = []
    for base, names in sorted(outputs.items()):
        if len(names) != 1 or declared and base not in declared:
            continue
        name, data = names[0], files[names[0]]
        if not isinstance(data, (bytes, bytearray)):
            continue
        if base.lower().endswith('.csv'):
            if listed.get(base):
                found += _csv_findings(name, bytes(data), listed[base])
            continue
        paths = templates.get(name)
        if paths is None:
            matches = [p for target, p in templates.items() if PurePosixPath(target).name == base]
            paths = matches[0] if len(matches) == 1 else set()
        keys = listed.get(base, [])
        if paths or keys:
            found += _json_findings(name, bytes(data), paths, keys)
    return found
