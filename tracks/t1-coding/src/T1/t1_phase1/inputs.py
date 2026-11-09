"""Previews of structured input files for the input profile: the sheets of a workbook with their first rows, the
members of a zip archive, and the element outline of an XML or HTML file; plus the decompressed text of an
archive's members for the canary scan. The standard library only (a workbook is a zip of XML parts), every read
bounded in bytes, elements and characters, so a large or odd input costs little. Each preview raises on an
unreadable file; the caller keeps that file's plain row and moves on."""
from __future__ import annotations

from datetime import datetime
from html.parser import HTMLParser
import json
import math
import os
import stat
import re
import xml.etree.ElementTree as ET
import zipfile

SHEETS = 8  # sheets previewed per workbook
SHEET_ROWS = 8  # leading rows (that hold a value) shown per sheet
SHEET_COLUMNS = 12  # leading columns shown per row
SHEET_COLUMNS_WIDE = 80  # input_fidelity: columns scanned per shown row, so a row holding values only past
# SHEET_COLUMNS still shows
WIDE_CELL_CHARS = 16  # input_fidelity: the cut of each cell of a row that holds values past SHEET_COLUMNS
CELL_CHARS = 40
SHARED_STRINGS = 20_000  # workbook strings read; a cell whose string lies past them shows its index
PART_BYTES = 8 * 1024 * 1024  # decompressed bytes of one workbook part read
PREVIEW_CHARS = 3000  # characters of a workbook preview or an archive listing
MEMBERS = 40  # archive members listed
OUTLINE_BYTES = 2 * 1024 * 1024  # bytes of an XML file read for its outline
HTML_BYTES = 1024 * 1024  # bytes of an HTML file read for its outline (a pure-Python parser)
OUTLINE_DEPTH = 6  # element depth kept; deeper elements count toward nothing
OUTLINE_SEEN = 400  # distinct element paths collected before the outline is chosen
OUTLINE_CHARS = 1500
SCAN_MEMBERS = 2000  # archive members read by the canary scan
SCAN_MEMBER_BYTES = 16 * 1024 * 1024
SCAN_TOTAL_BYTES = 64 * 1024 * 1024
_REL = '{http://schemas.openxmlformats.org/officeDocument/2006/relationships}'
_CELL = re.compile(r'([A-Z]{1,3})(\d+)')
_GUID = re.compile(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', re.I)


def cut(text, limit, mark=''):
    """text within limit characters (mark, such as '...', appended when it is cut), the cut moved back before a
    GUID it would split: a partial GUID is never shown, since only whole ones are recognised and withheld."""
    if len(text) <= limit:
        return text
    end = limit - len(mark)
    for match in _GUID.finditer(text, max(0, end - 36), end + 36):
        if match.start() < end < match.end():
            end = match.start()
            break
    return text[:end] + mark


JSON_READ_BYTES = 64 * 1024
JSON_READ_TOTAL = 256 * 1024
JSON_DEPTH = 32
JSON_NODES = 4096
JSON_KEYS = 32


class _JsonPreviewError(ValueError):
    pass


def _json_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise _JsonPreviewError('duplicate_keys')
        result[key] = value
    return result


def _json_nonfinite(_):
    raise _JsonPreviewError('nonfinite')


def _json_descriptor(value):
    if value is None:
        return {'type': 'null'}
    if isinstance(value, bool):
        return {'type': 'boolean'}
    if isinstance(value, (dict, list)):
        return {'type': 'object' if isinstance(value, dict) else 'array', 'length': len(value)}
    return {'type': 'string' if isinstance(value, str) else 'integer' if isinstance(value, int) else 'number'}


def _json_structure(data):
    # Limit syntactic nesting before json.loads, without interpreting anything inside strings.
    text = data.decode('utf-8')
    depth, quoted, escape = 0, False, False
    for char in text:
        if quoted:
            if escape:
                escape = False
            elif char == '\\':
                escape = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char in '[{':
            depth += 1
            if depth > JSON_DEPTH:
                raise _JsonPreviewError('resource_limit')
        elif char in ']}':
            depth -= 1
    value = json.loads(text, object_pairs_hook=_json_pairs, parse_constant=_json_nonfinite)
    pending, count = [value], 0
    while pending:
        item = pending.pop()
        count += 1
        if count > JSON_NODES:
            raise _JsonPreviewError('resource_limit')
        if isinstance(item, float) and not math.isfinite(item):
            raise _JsonPreviewError('nonfinite')
        if isinstance(item, dict):
            for key in item:
                key.encode('utf-8')  # escaped unpaired surrogates cannot safely enter a prompt
            pending.extend(item.values())
        elif isinstance(item, list):
            pending.extend(item)
    fields = ([dict(key=key, **_json_descriptor(item)) for key, item in list(value.items())[:JSON_KEYS]]
              if isinstance(value, dict) else [])
    result = {'status': 'parsed', 'display': 'complete_shallow', 'root': _json_descriptor(value),
              'fields': fields}
    if isinstance(value, dict) and len(value) > len(fields):
        result.update(display='truncated', omitted_fields=len(value) - len(fields))
    return result


def json_preview(path, head, budget):
    """Replacement fields for a known-truncated JSON head, and additional bytes reserved.

    Never larger in serialized characters or UTF-8 bytes than {'head': head}. A shallow description exposes no scalar
    values or inferred element shape. Byte/depth/node bounds limit work, not elapsed wall time. Complete
    small files keep their original rows in the caller and require no additional inspection here.
    """
    used, structure = 0, {'status': 'unreadable', 'display': 'unavailable'}
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, 'rb') as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise OSError('not a regular input')
            if before.st_size > JSON_READ_BYTES:
                structure['status'] = 'byte_cap'
            elif before.st_size > budget:
                structure['status'] = 'unit_budget'
            else:
                used = before.st_size  # charge the attempted read even if I/O fails after some bytes
                data = stream.read(before.st_size)
                after = os.fstat(stream.fileno())
                if len(data) != before.st_size or (after.st_size, after.st_mtime_ns) != (
                        before.st_size, before.st_mtime_ns):
                    raise OSError('input changed while inspecting')
                structure = _json_structure(data)
    except _JsonPreviewError as exc:
        structure = {'status': str(exc), 'display': 'unavailable'}
    except (UnicodeError, ValueError):
        structure = {'status': 'invalid_json', 'display': 'unavailable'}
    except (RecursionError, MemoryError):
        structure = {'status': 'resource_limit', 'display': 'unavailable'}
    except OSError:
        structure = {'status': 'unreadable', 'display': 'unavailable'}
    def size(value):
        text = json.dumps(value, ensure_ascii=False)
        return len(text), len(text.encode('utf-8'))

    limit = size({'head': head})
    fits = lambda value: all(actual <= bound for actual, bound in zip(size(value), limit))
    fields = {'head': '', 'head_truncated': True, 'json_structure': structure}
    while not fits(fields) and structure.get('fields'):
        structure['fields'].pop()
        structure['display'] = 'truncated'
        structure['omitted_fields'] = structure.get('omitted_fields', 0) + 1
    if not fits(fields):  # defensive: caller normally provides a nearly 2,000-character head
        return {'head': head}, used
    low, high = 0, len(head)
    while low < high:
        middle = (low + high + 1) // 2
        fields['head'] = cut(head, middle)
        if fits(fields):
            low = middle
        else:
            high = middle - 1
    fields['head'] = cut(head, low)
    return fields, used


class _Capped:
    """A binary stream that ends after limit bytes (a truncated part then stops parsing early)."""

    def __init__(self, stream, limit):
        self.stream, self.left = stream, limit

    def read(self, size=-1):
        if self.left <= 0:
            return b''
        size = self.left if size is None or size < 0 else min(size, self.left)
        data = self.stream.read(size)
        self.left -= len(data)
        return data


def _local(tag):
    return tag.rsplit('}', 1)[-1] if isinstance(tag, str) else ''


def _events(archive, name, handle):
    """Feed one bounded part through iterparse's end events; a part cut by the bound keeps what it gave."""
    with archive.open(name) as stream:
        try:
            for _, element in ET.iterparse(_Capped(stream, PART_BYTES), events=('end',)):
                if handle(element):
                    break
        except ET.ParseError:
            pass


def _part(archive, name):
    with archive.open(name) as stream:
        return ET.fromstring(_Capped(stream, PART_BYTES).read())


def _column(letters):
    index = 0
    for letter in letters:
        index = index * 26 + ord(letter) - 64
    return index - 1


def _text(element):
    """The text of a shared or inline string: its <t> runs joined."""
    return ''.join(node.text or '' for node in element.iter() if _local(node.tag) == 't')


def _shared_strings(archive, names):
    found = []
    if 'xl/sharedStrings.xml' not in names:
        return found

    def handle(element):
        if _local(element.tag) == 'si':
            found.append(_text(element))
            element.clear()
        return len(found) >= SHARED_STRINGS
    _events(archive, 'xl/sharedStrings.xml', handle)
    return found


def _cell(element, shared):
    kind = element.get('t')
    value = next((node.text or '' for node in element if _local(node.tag) == 'v'), '')
    if kind == 's':
        index = int(value) if value.isdigit() else -1
        value = shared[index] if 0 <= index < len(shared) else f'<string {value}>'
    elif kind == 'inlineStr':
        value = _text(element)
    elif kind == 'b':
        value = 'TRUE' if value == '1' else 'FALSE'
    return cut(' '.join(value.split()), CELL_CHARS, '...')


def _sheet(archive, name, shared, wide=False):
    """(dimension, {row number: [cells]}) of a sheet's first SHEET_ROWS rows that hold a value. wide
    (input_fidelity): SHEET_COLUMNS_WIDE columns are scanned, a row that holds values past SHEET_COLUMNS shows
    them with each cell cut to WIDE_CELL_CHARS, and one holding values only past SHEET_COLUMNS_WIDE still shows,
    as a count of those cells."""
    state = {'dimension': None, 'rows': {}}
    limit = SHEET_COLUMNS_WIDE if wide else SHEET_COLUMNS

    def handle(element):
        tag = _local(element.tag)
        if tag == 'dimension':
            state['dimension'] = element.get('ref')
        elif tag == 'row':
            cells, beyond = {}, 0
            for cell in element:
                match = _CELL.fullmatch(cell.get('r') or '') if _local(cell.tag) == 'c' else None
                if match and _column(match[1]) < limit:
                    text = _cell(cell, shared)
                    if text:
                        cells[_column(match[1])] = text
                elif wide and match and _cell(cell, shared):
                    beyond += 1
            rows = state['rows']
            if wide and (beyond or (cells and max(cells) >= SHEET_COLUMNS)):
                cells = {index: cut(text, WIDE_CELL_CHARS, '...') for index, text in cells.items()}
                shown = [cells.get(i, '') for i in range(max(cells) + 1)] if cells else []
                if beyond:
                    shown.append(f'(+{beyond} cells past column {SHEET_COLUMNS_WIDE})')
                rows[element.get('r') or str(len(rows) + 1)] = shown
            elif cells:
                rows[element.get('r') or str(len(rows) + 1)] = [cells.get(i, '') for i in range(max(cells) + 1)]
            element.clear()
            return len(rows) >= SHEET_ROWS
        return False
    _events(archive, name, handle)
    return state['dimension'], state['rows']


def _size(value):
    return len(json.dumps(value, ensure_ascii=False))


def _fit_sheets(sheets, limit):
    """The leading sheets within limit characters; the last one that fits only in part keeps its leading rows."""
    kept = []
    for sheet in sheets:
        rows = sheet.get('rows') or {}
        numbers = list(rows)
        for count in range(len(numbers), -1, -1):
            entry = dict(sheet, rows={n: rows[n] for n in numbers[:count]}) if 'rows' in sheet else sheet
            if _size(kept + [entry]) <= limit:
                kept.append(entry)
                break
        else:
            return kept
        if count < len(numbers):
            return kept
    return kept


def _fit_wide(sheets, limit):
    """input_fidelity: _fit_sheets's budget shared out, so a sheet of wide rows does not hide the sheets after it:
    the leading sheets' names and dimensions that fit, then their rows in turn (each sheet's first row, then each
    one's second, ...), a sheet keeping its leading rows up to the first that no longer fits."""
    kept = _fit_sheets([dict(sheet, rows={}) if 'rows' in sheet else sheet for sheet in sheets], limit)
    open_ = [index for index, entry in enumerate(kept) if 'rows' in entry]
    depth = 0
    while open_:
        still = []
        for index in open_:
            rows = sheets[index]['rows'] or {}
            numbers = list(rows)
            if depth >= len(numbers):
                continue
            kept[index]['rows'][numbers[depth]] = rows[numbers[depth]]
            if _size(kept) > limit:
                del kept[index]['rows'][numbers[depth]]
                continue
            still.append(index)
        open_, depth = still, depth + 1
    return kept


def workbook_preview(path, wide=False) -> dict:
    """{'sheets': [{'name', 'dimension', 'rows': {sheet row number: [cells]}}], 'sheet_count'}: an .xlsx/.xlsm
    workbook's sheet names and the first rows of each, as the workbook stores the cells (numbers unformatted, a
    date as its serial day number), within PREVIEW_CHARS characters. wide (input_fidelity): _sheet's rows, fitted by
    _fit_wide."""
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        book = _part(archive, 'xl/workbook.xml')
        targets = {}
        if 'xl/_rels/workbook.xml.rels' in names:
            for rel in _part(archive, 'xl/_rels/workbook.xml.rels'):
                target = (rel.get('Target') or '').replace('\\', '/')
                targets[rel.get('Id')] = target.lstrip('/') if target.startswith('/') else 'xl/' + target
        listed = [sheet for sheet in book.iter() if _local(sheet.tag) == 'sheet']
        shared = _shared_strings(archive, names)
        sheets = []
        for sheet in listed[:SHEETS]:
            entry = {'name': cut(str(sheet.get('name')), CELL_CHARS, '...')}
            part = targets.get(sheet.get(_REL + 'id'))
            if part in names:
                dimension, entry['rows'] = _sheet(archive, part, shared, wide)
                if dimension:
                    entry['dimension'] = cut(dimension, CELL_CHARS, '...')
            sheets.append(entry)
    fitted = _fit_wide(sheets, PREVIEW_CHARS) if wide else _fit_sheets(sheets, PREVIEW_CHARS)
    return {'sheets': fitted, 'sheet_count': len(listed)}


def archive_listing(path) -> dict:
    """{'members': [{'name', 'bytes'}], 'member_count'}: a zip archive's first MEMBERS files, within
    PREVIEW_CHARS characters."""
    with zipfile.ZipFile(path) as archive:
        members = [info for info in archive.infolist() if not info.is_dir()]
    listed, kept = [{'name': info.filename, 'bytes': info.file_size} for info in members[:MEMBERS]], []
    for item in listed:
        if _size(kept + [item]) > PREVIEW_CHARS:
            break
        kept.append(item)
    return {'members': kept, 'member_count': len(members)}


def archive_texts(path):
    """(member name, decompressed bytes) of a zip archive's members, bounded per member and in total: what a
    preview of the archive could show, for the canary scan. Raises on an unreadable archive."""
    total = 0
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist()[:SCAN_MEMBERS]:
            if info.is_dir() or total >= SCAN_TOTAL_BYTES:
                continue
            try:
                with archive.open(info) as stream:
                    data = stream.read(min(SCAN_MEMBER_BYTES, SCAN_TOTAL_BYTES - total))
            except Exception:  # an encrypted or damaged member: the others are still read
                continue
            total += len(data)
            yield info.filename, data


def _tree(paths, order):
    tree = {}
    for path in sorted(paths, key=order.get):
        node = tree
        for name in path:
            node = node.setdefault(name, {})
    return tree


def _render(tree, counts, parents, prefix=()):
    parts = []
    for name, children in tree.items():
        path = prefix + (name,)
        text = name + (f' ({counts[path]})' if counts.get(path, 1) > 1 else '')
        if children:
            text += '{' + _render(children, counts, parents, path) + '}'
        elif path in parents:
            text += '{...}'
        parts.append(text)
    return ', '.join(parts)


def _outline(counts):
    """The element tree as `root{child (count){leaf, ...}, ...}`: paths chosen shallowest first, in the order
    first seen, while the rendering fits OUTLINE_CHARS; `{...}` marks children left out."""
    order = {path: index for index, path in enumerate(counts)}
    parents = {path[:-1] for path in counts}
    chosen, text = set(), ''
    for path in sorted(counts, key=lambda p: (len(p), order[p])):
        if path[:-1] and path[:-1] not in chosen:
            continue
        trial = _render(_tree(chosen | {path}, order), counts, parents)
        if len(trial) > OUTLINE_CHARS:
            break
        chosen.add(path)
        text = trial
    return text


def _count(counts, stack):
    if len(stack) <= OUTLINE_DEPTH:
        key = tuple(stack)
        if key in counts or len(counts) < OUTLINE_SEEN:
            counts[key] = counts.get(key, 0) + 1


def xml_outline(path) -> str:
    """The element outline of an XML file (namespaces dropped) from its first OUTLINE_BYTES: each distinct
    element path up to OUTLINE_DEPTH deep, with how often it occurs. A file that stops parsing keeps what was
    found before; one that never parses raises."""
    counts, stack = {}, []
    parser = ET.XMLPullParser(events=('start', 'end'))
    with open(path, 'rb') as stream:
        source = _Capped(stream, OUTLINE_BYTES)
        try:
            while True:
                chunk = source.read(65536)
                if not chunk:
                    break
                parser.feed(chunk)
                for event, element in parser.read_events():
                    if event == 'start':
                        stack.append(_local(element.tag))
                        _count(counts, stack)
                    else:
                        stack.pop()
                        element.clear()
        except ET.ParseError:
            if not counts:
                raise
    if not counts:
        raise ValueError('no elements')
    return _outline(counts)


class _Outline(HTMLParser):
    VOID = frozenset({'area', 'base', 'br', 'col', 'embed', 'hr', 'img', 'input', 'link', 'meta', 'param',
                      'source', 'track', 'wbr'})

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack, self.counts = [], {}

    def handle_starttag(self, tag, attrs):
        _count(self.counts, self.stack + [tag])
        if tag not in self.VOID:
            self.stack.append(tag)

    def handle_endtag(self, tag):
        if tag in self.stack:  # an unclosed element is closed by the end tag of an element around it
            while self.stack and self.stack.pop() != tag:
                pass


def html_outline(path) -> str:
    """xml_outline for HTML, parsed leniently."""
    parser = _Outline()
    with open(path, 'rb') as stream:
        parser.feed(stream.read(HTML_BYTES).decode('utf-8', errors='replace'))
    if not parser.counts:
        raise ValueError('no elements')
    return _outline(parser.counts)


# input_fidelity: data traps of the delimited tables, read whole (each file within TRAP_FILE_BYTES, all of them
# within TRAP_TOTAL_BYTES; a larger file is skipped, never judged from a part): digit codes with leading zeros, the
# strptime formats of date text (and join columns of two files whose formats differ), and the exact values of
# low-cardinality text columns. One line each, at most TRAP_LINES; a column holding a GUID is never listed.
TRAP_LINES = 12
TRAP_LINE_CHARS = 240
TRAP_FILE_BYTES = 32 * 1024 * 1024
TRAP_TOTAL_BYTES = 64 * 1024 * 1024
TRAP_VALUES = 8  # distinct text values listed for a low-cardinality column
TRAP_SHAPES = 12  # distinct date shapes (digits as 9, letters as a) beyond which a column is not date text
TRAP_SHAPE_VALUES = 500  # distinct values per date shape tested against the formats
TRAP_FORMATS = ('%Y-%m-%d', '%Y/%m/%d', '%Y.%m.%d', '%Y%m%d', '%m/%d/%Y', '%d/%m/%Y', '%m-%d-%Y', '%d-%m-%Y',
                '%d.%m.%Y', '%m/%d/%y', '%d/%m/%y', '%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M',
                '%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%dT%H:%M:%S.%f', '%m/%d/%Y %H:%M', '%m/%d/%Y %H:%M:%S',
                '%d %b %Y', '%d-%b-%Y', '%b %d, %Y', '%b %d %Y', '%d %B %Y', '%B %d, %Y', '%Y-%m', '%b-%y', '%b %Y')
_DATE_TEXT = re.compile(r'\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}(?:[ T]\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?)?$|\d{8}$|'
                        r'\d{4}-\d{1,2}$|\d{1,2}[ -][A-Za-z]{3,9}[ -]\d{2,4}$|[A-Za-z]{3,9}[ -]\d{1,2},? \d{4}$|'
                        r'[A-Za-z]{3}[- ]\d{2,4}$')
_ZERO_CODE = re.compile(r'0\d+')


def _date_ok(value, form):
    try:
        return 1900 <= datetime.strptime(value, form).year <= 2100
    except ValueError:
        return False


def _date_formats(shapes):
    """(formats, kind) of a column whose values all look like dates: the formats that parse every kept value
    ('one' or 'ambiguous'), else those of each shape ('mixed'); None when some shape fits no format."""
    values = [value for kept in shapes.values() for value in kept]
    common = [form for form in TRAP_FORMATS if all(_date_ok(value, form) for value in values)]
    if common:
        return common, ('one' if len(common) == 1 else 'ambiguous')
    found = []
    for kept in shapes.values():
        fits = [form for form in TRAP_FORMATS if all(_date_ok(value, form) for value in kept)]
        if not fits:
            return None
        found += [form for form in fits if form not in found]
    return found, 'mixed'


class _Column:
    __slots__ = ('name', 'count', 'numeric', 'digits', 'zero', 'values', 'shapes')

    def __init__(self, name):
        self.name, self.count, self.numeric, self.digits, self.zero = name, 0, True, True, None
        self.values, self.shapes = set(), {}

    def add(self, value):
        """Whether the column still needs its values (some judgement is still open)."""
        self.count += 1
        if self.digits:
            if value.isdigit() and value.isascii():
                if self.zero is None and _ZERO_CODE.fullmatch(value):
                    self.zero = value
            else:
                self.digits = False
        if self.numeric:
            try:
                float(value)
            except ValueError:
                self.numeric = False
        if self.values is not None:
            self.values.add(value)
            if len(self.values) > TRAP_VALUES or _GUID.search(value):
                self.values = None
        if self.shapes is not None:
            if not _DATE_TEXT.match(value):
                self.shapes = None
            else:
                kept = self.shapes.setdefault(re.sub(r'[A-Za-z]', 'a', re.sub(r'\d', '9', value)), set())
                if len(self.shapes) > TRAP_SHAPES:
                    self.shapes = None
                elif len(kept) < TRAP_SHAPE_VALUES:
                    kept.add(value)
        return self.digits or self.values is not None or self.shapes is not None


def _delimiter(path, first):
    if path.lower().endswith('.tsv'):
        return '\t'
    return ';' if first.count(',') == 0 and first.count(';') > 0 else ','


def _table_columns(path):
    """The _Column judgements of one delimited table read whole; raises on an unreadable file."""
    import csv
    with open(path, newline='', encoding='utf-8-sig', errors='replace') as stream:
        first = stream.readline()
        stream.seek(0)
        reader = csv.reader(stream, delimiter=_delimiter(str(path), first))
        header = next(reader, None) or []
        columns = [_Column(name.strip() or f'#{index}') for index, name in enumerate(header)]
        open_ = list(range(len(columns)))
        for row in reader:
            if not open_:
                break
            still = []
            for index in open_:
                value = row[index].strip() if index < len(row) else ''
                if not value or columns[index].add(value):
                    still.append(index)
            open_ = still
    # a column dropped from open_ has no judgement left open, so stopping early changes no line
    return columns


def data_traps(tables) -> list:
    """input_fidelity: the INPUT PROFILE's data-trap lines (at most TRAP_LINES, each within TRAP_LINE_CHARS) of
    tables, (shown path, file path) pairs of .csv/.tsv inputs in path order: identifier columns whose digit codes
    have leading zeros, the strptime formats of date-text columns, join columns (one name, case-blind, in two
    files) whose date formats differ, and the exact values of text columns with at most TRAP_VALUES of them.
    Deterministic; a file over TRAP_FILE_BYTES, past TRAP_TOTAL_BYTES, or unreadable gives nothing."""
    zeros, dates, texts, formats_of, left = [], [], [], {}, TRAP_TOTAL_BYTES
    for shown, path in tables:
        try:
            size = os.stat(path).st_size
            if size > TRAP_FILE_BYTES or size > left:
                continue
            left -= size
            columns = _table_columns(path)
        except Exception:
            continue
        for column in columns:
            if not column.count:
                continue
            if column.digits and column.zero is not None:
                zeros.append(f'{shown} column {column.name!r}: digit codes with leading zeros (e.g. {column.zero!r}); '
                             'read it as str (dtype=str) to keep them')
                continue
            found = _date_formats(column.shapes) if column.shapes else None
            if found:
                forms, kind = found
                text = (f'strptime format {forms[0]}' if kind == 'one' else
                        f'strptime formats {" or ".join(forms)} (each fits every value)' if kind == 'ambiguous' else
                        f'mixed strptime formats {", ".join(forms)} (parse each)')
                dates.append(f'{shown} column {column.name!r}: date text, {text}')
                formats_of.setdefault(column.name.lower(), []).append((shown, column.name, tuple(forms),
                                                                       kind != 'mixed'))
            elif not column.numeric and column.values:
                listed = ', '.join(repr(cut(value, CELL_CHARS, '...')) for value in sorted(column.values))
                texts.append(f'{shown} column {column.name!r}: {len(column.values)} distinct text values {listed}')
    joins = []
    for name, seen in formats_of.items():
        for index, (shown, column, forms, whole) in enumerate(seen):
            # one format that reads both columns whole (each fits every value of its own) is no mismatch
            other = next((item for item in seen[index + 1:] if item[0] != shown and item[2] != forms
                          and not (whole and item[3] and set(forms) & set(item[2]))), None)
            if other is not None:
                joins.append(f'join column {column!r} has different date formats: {shown} {" or ".join(forms)}, '
                             f'{other[0]} {" or ".join(other[2])}; parse each with its own format before joining')
                break
    return [cut(line, TRAP_LINE_CHARS, '...') for line in (zeros + joins + dates + texts)[:TRAP_LINES]]
