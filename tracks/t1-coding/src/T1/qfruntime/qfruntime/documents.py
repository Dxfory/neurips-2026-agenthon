"""Bounded local document extraction with explicit field and quantity semantics.

Text uses labelled ``name: value`` / ``name=value`` lines and blank-line records;
HTML supports labelled text and tables with explicit header cells, not rendered
CSS layout; explicit hidden content needs preprocessing. CSV
uses an explicit comma/quote dialect; JSON preserves duplicate object fields.
This is deterministic extraction, not OCR or semantic interpretation of prose.
``extracted`` means one grounded, specification-compatible scalar candidate, not
independent verification of the document or its assertion. No unit conversion,
date guessing, network access, decompression or first-match conflict resolution
is performed. Unsupported formats and exceeded limits fail explicitly.
"""
from dataclasses import dataclass
from datetime import datetime, timezone
from html import unescape
from html.parser import HTMLParser
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
from .semantics import canonical_unit


class DocumentError(ValueError):
    """Unsupported, malformed, unsafe or over-budget document."""


@dataclass(frozen=True)
class DocumentLimits:
    max_bytes: int = 1048576
    max_rows: int = 4096
    max_fields: int = 16384
    max_field_chars: int = 4096
    max_depth: int = 32
    max_candidates: int = 1024
    max_output_fields: int = 32768

    def __post_init__(self):
        for value in self.__dict__.values():
            if type(value) is not int or value <= 0:
                raise ValueError('Document limits must be positive integers')


@dataclass(frozen=True)
class _Field:
    value: object
    start: int
    end: int
    locator: str


@dataclass(frozen=True)
class _Record:
    locator: str
    fields: tuple

    def get(self, name):
        return tuple(field for key, field in self.fields if key == name)


@dataclass(frozen=True)
class Document:
    source_id: str
    format: str
    text: str
    sha256: str
    records: tuple
    limits: DocumentLimits
    issues: tuple = ()


def _read(root, relative, limits):
    if not isinstance(relative, (str, Path)):
        raise DocumentError('Path must be a local relative path')
    name = str(relative)
    parts = name.split('/')
    if (not name or ':' in name or '\\' in name or name.startswith('/')
            or any(part in ('', '.', '..') for part in parts)):
        raise DocumentError('Network, absolute and traversing paths are forbidden')
    base = Path(os.path.abspath(os.fspath(root)))
    fd = None
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    try:
        fd = os.open(base.anchor, flags)
        for part in (*base.parts[1:], *parts[:-1]):
            child = os.open(part, flags, dir_fd=fd)
            os.close(fd)
            fd = child
        source = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
        with os.fdopen(source, 'rb') as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise DocumentError('Source must be a regular file with exactly one link')
            if info.st_size > limits.max_bytes:
                raise DocumentError('Document exceeds max_bytes')
            data = stream.read(limits.max_bytes + 1)
            if len(data) > limits.max_bytes:
                raise DocumentError('Document exceeds max_bytes')
    except OSError as exc:
        raise DocumentError('Local source cannot be read safely') from exc
    finally:
        if fd is not None:
            os.close(fd)
    if data.startswith((b'PK\x03\x04', b'PK\x05\x06', b'PK\x07\x08', b'\x1f\x8b', b'BZh', b'\xfd7zXZ', b'7z\xbc\xaf')):
        raise DocumentError('Compressed sources are unsupported; no decompression is attempted')
    try:
        return name, data.decode('utf-8')
    except UnicodeDecodeError as exc:
        raise DocumentError('Source must be explicit UTF-8 text') from exc


class _Builder:
    def __init__(self, text, limits):
        self.text, self.limits = text, limits
        self.records, self.count = [], 0

    def field(self, value, start, end, locator):
        if (end - start > self.limits.max_field_chars
                or isinstance(value, str) and len(value) > self.limits.max_field_chars):
            raise DocumentError('Field exceeds max_field_chars')
        self.count += 1
        if self.count > self.limits.max_fields:
            raise DocumentError('Document exceeds max_fields')
        return _Field(value, start, end, locator)

    def add(self, locator, fields):
        if fields:
            self.records.append(_Record(locator, tuple(fields)))
            if len(self.records) > self.limits.max_rows:
                raise DocumentError('Document exceeds max_rows')


def _trim_mapped(items):
    start, end = 0, len(items)
    while start < end and items[start][0].isspace():
        start += 1
    while end > start and items[end-1][0].isspace():
        end -= 1
    return items[start:end]


def _mapped_field(builder, items, locator, empty_at):
    items = _trim_mapped(items)
    if not items:
        return builder.field('', empty_at, empty_at, locator)
    return builder.field(''.join(x[0] for x in items), items[0][1], items[-1][2], locator)


def _labelled(builder, items, prefix='text'):
    fields, line, number, record = [], [], 1, 1

    def consume():
        nonlocal fields, record
        content = ''.join(item[0] for item in line)
        if not content.strip():
            builder.add(f'{prefix}:record[{record}]', fields)
            fields, record = [], record + 1
            return
        match = re.match(r'\s*([^:=\r\n]+?)\s*[:=]\s*(.*?)\s*$', content)
        if match:
            name = match.group(1)
            if len(name) > builder.limits.max_field_chars:
                raise DocumentError('Field name exceeds max_field_chars')
            offset = line[match.start(2)][1] if match.start(2) < len(line) else line[-1][2]
            value = _mapped_field(builder, line[match.start(2):match.end(2)],
                                  f'{prefix}:line[{number}]/{name}', offset)
            fields.append((name, value))
    for item in items:
        if item[0] == '\n':
            consume()
            line, number = [], number + 1
        else:
            line.append(item)
    if line:
        consume()
    builder.add(f'{prefix}:record[{record}]', fields)


def _csv(builder):
    text, index, rows = builder.text, 0, []
    while index < len(text):
        row = []
        while True:
            start, value = index, []
            if index < len(text) and text[index] == '"':
                index += 1
                while True:
                    if index == len(text):
                        raise DocumentError('Unterminated CSV quote')
                    char = text[index]
                    index += 1
                    if char == '"':
                        if index < len(text) and text[index] == '"':
                            value.append('"')
                            index += 1
                        else:
                            break
                    else:
                        value.append(char)
                if index < len(text) and text[index] not in ',\r\n':
                    raise DocumentError('Unexpected data after a CSV quoted field')
            else:
                while index < len(text) and text[index] not in ',\r\n':
                    if text[index] == '"':
                        raise DocumentError('Quote inside an unquoted CSV field')
                    value.append(text[index])
                    index += 1
            row.append(builder.field(''.join(value), start, index, f'csv:row[{len(rows)+1}]/column[{len(row)+1}]'))
            if index == len(text):
                break
            char = text[index]
            index += 1
            if char != ',':
                if char == '\r' and index < len(text) and text[index] == '\n':
                    index += 1
                break
        rows.append(row)
        if len(rows) > builder.limits.max_rows + 1:
            raise DocumentError('CSV exceeds max_rows')
    if not rows:
        return
    headers = [field.value for field in rows[0]]
    if any(not header.strip() for header in headers):
        raise DocumentError('CSV needs nonempty explicit header names')
    for number, row in enumerate(rows[1:], 2):
        if len(row) != len(headers):
            raise DocumentError('CSV row width differs from its header')
        builder.add(f'csv:row[{number}]', zip(headers, row))


class _JSON:
    def __init__(self, builder):
        self.builder, self.text, self.index = builder, builder.text, 0
        self.decoder = json.JSONDecoder(parse_constant=self.invalid_constant)

    @staticmethod
    def invalid_constant(value):
        raise DocumentError('JSON nonfinite constants are forbidden')

    def ws(self):
        while self.index < len(self.text) and self.text[self.index] in ' \t\n\r':
            self.index += 1

    def expect(self, char):
        self.ws()
        if self.index >= len(self.text) or self.text[self.index] != char:
            raise DocumentError('Malformed JSON structure')
        self.index += 1

    def node(self, path='', depth=0):
        if depth > self.builder.limits.max_depth:
            raise DocumentError('JSON exceeds max_depth')
        self.ws()
        start = self.index
        if start == len(self.text):
            raise DocumentError('Unexpected JSON end')
        char = self.text[start]
        if char in '[{':
            self.index += 1
            self.ws()
            close, fields, number = ('}' if char == '{' else ']'), [], 0
            if self.index < len(self.text) and self.text[self.index] == close:
                self.index += 1
                return None
            while True:
                if char == '{':
                    self.ws()
                    if self.index >= len(self.text) or self.text[self.index] != '"':
                        raise DocumentError('JSON field names must be strings')
                    key, self.index = self.decoder.raw_decode(self.text, self.index)
                    if len(key) > self.builder.limits.max_field_chars:
                        raise DocumentError('JSON field name exceeds max_field_chars')
                    self.expect(':')
                else:
                    key = str(number)
                number += 1
                if number > self.builder.limits.max_fields:
                    raise DocumentError('JSON container exceeds max_fields')
                # Occurrence indices at every object level retain repeated
                # ancestor keys, for which plain JSON pointers are not unique.
                child = path + '/' + key.replace('~', '~0').replace('/', '~1')
                if char == '{':
                    child += f'#{number}'
                value = self.node(child, depth + 1)
                if value is not None:
                    if char == '{':
                        fields.append((key, value))
                    else:
                        self.builder.add('json:'+child, [('value', value)])
                self.ws()
                if self.index < len(self.text) and self.text[self.index] == close:
                    self.index += 1
                    break
                self.expect(',')
            if char == '{':
                self.builder.add('json:'+path, fields)
            return None
        value, self.index = self.decoder.raw_decode(self.text, self.index)
        if isinstance(value, (list, dict)) or isinstance(value, float) and not math.isfinite(value):
            raise DocumentError('Invalid JSON scalar')
        if type(value) is float and value == 0 and _nonzero_mantissa(self.text[start:self.index]):
            raise DocumentError('JSON number underflows binary64; explicit preprocessing required')
        return self.builder.field(value, start, self.index, 'json:'+path)


class _HTML(HTMLParser):
    _BLOCKS = {'p', 'div', 'section', 'article', 'li', 'h1', 'h2', 'h3', 'h4', 'br'}
    _VOID = {'area', 'base', 'br', 'col', 'embed', 'hr', 'img', 'input', 'link', 'meta', 'param', 'source', 'track', 'wbr'}

    def __init__(self, builder):
        super().__init__(convert_charrefs=False)
        self.builder, self.visible, self.tables = builder, [], []
        self.offsets, offset = [], 0
        for line in builder.text.splitlines(keepends=True):
            self.offsets.append(offset)
            offset += len(line)
        self.offsets.append(offset)
        self.depth, self.hidden, self.table, self.row, self.cell = 0, [], None, None, None
        self.issues = []

    def newline(self):
        if self.visible and self.visible[-1][0] != '\n':
            self.visible.append(('\n', 0, 0))

    def handle_starttag(self, tag, attrs):
        for name, value in attrs:
            if name == 'hidden' or name == 'aria-hidden' and value and value.lower().strip() == 'true':
                raise DocumentError('Explicit hidden HTML needs preprocessing')
            if name == 'style' and value:
                style = re.sub(r'/\*.*?\*/', '', value, flags=re.S)
                if re.search(r'(?:^|;)\s*(?:display\s*:\s*none|visibility\s*:\s*(?:hidden|collapse))\b', style, re.I):
                    raise DocumentError('Explicit hidden HTML needs preprocessing')
        if tag not in self._VOID:
            self.depth += 1
            if self.depth > self.builder.limits.max_depth:
                raise DocumentError('HTML exceeds max_depth')
        if tag in ('script', 'style', 'template', 'head'):
            self.hidden.append(tag)
        if self.hidden:
            return
        if tag == 'table':
            if self.table is not None:
                raise DocumentError('Nested HTML tables need explicit preprocessing')
            self.table = []
        elif tag == 'tr' and self.table is not None:
            if self.row is not None:
                raise DocumentError('Unclosed HTML table row')
            self.row = []
        elif tag in ('td', 'th') and self.row is not None:
            if self.cell is not None:
                raise DocumentError('Unclosed HTML table cell')
            if any(name in ('colspan', 'rowspan') and value != '1' for name, value in attrs):
                raise DocumentError('Merged HTML cells need explicit preprocessing')
            line, col = self.getpos()
            at = self.offsets[line-1] + col + len(self.get_starttag_text())
            self.cell = (tag, [], at)
        elif tag in self._BLOCKS and self.table is None:
            self.newline()

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag not in self._VOID:
            self.handle_endtag(tag)

    def handle_endtag(self, tag):
        if tag not in self._VOID:
            self.depth = max(0, self.depth - 1)
        if self.hidden:
            if tag == self.hidden[-1]:
                self.hidden.pop()
            return
        if tag in ('td', 'th') and self.cell is not None:
            self.row.append(self.cell)
            self.cell = None
        elif tag == 'tr' and self.row is not None:
            if self.cell is not None:
                raise DocumentError('Unclosed HTML table cell')
            self.table.append(self.row)
            self.row = None
            if len(self.table) > self.builder.limits.max_rows + 1:
                raise DocumentError('HTML table exceeds max_rows')
        elif tag == 'table' and self.table is not None:
            if self.row is not None:
                raise DocumentError('Unclosed HTML table row')
            self.tables.append(self.table)
            self.table = None
        elif tag in self._BLOCKS and self.table is None:
            self.newline()

    def append(self, value, raw_length=None):
        if self.hidden:
            return
        line, col = self.getpos()
        start = self.offsets[line - 1] + col
        target = self.cell[1] if self.cell is not None else self.visible if self.table is None else None
        if target is not None:
            target.extend((char, start + i if raw_length is None else start,
                           start + i + 1 if raw_length is None else start + raw_length)
                          for i, char in enumerate(value))

    def handle_data(self, data):
        self.append(data)

    def handle_entityref(self, name):
        raw = '&'+name+';'
        self.entity(raw)

    def handle_charref(self, name):
        raw = '&#'+name+';'
        self.entity(raw)

    def entity(self, raw):
        line, col = self.getpos()
        at = self.offsets[line-1] + col
        if self.builder.text[at:at+len(raw)] != raw:
            raise DocumentError('Unterminated HTML entities need explicit preprocessing')
        self.append(unescape(raw), len(raw))

    def finish(self):
        if self.table is not None or self.cell is not None or self.row is not None or self.hidden:
            raise DocumentError('Incomplete HTML structure')
        for table_index, table in enumerate(self.tables, 1):
            if not table or not all(tag == 'th' for tag, _, _ in table[0]):
                self.issues.append('html_table_without_explicit_headers')
                continue
            headers = [''.join(c for c, _, _ in items).strip() for _, items, _ in table[0]]
            if any(not header for header in headers):
                raise DocumentError('Empty HTML table header')
            if any(len(header) > self.builder.limits.max_field_chars for header in headers):
                raise DocumentError('HTML header exceeds max_field_chars')
            for row_index, row in enumerate(table[1:], 2):
                if len(row) != len(headers):
                    raise DocumentError('HTML row width differs from its header')
                fields = []
                locator = f'html:table[{table_index}]/tr[{row_index}]'
                for column, (name, (_, items, at)) in enumerate(zip(headers, row), 1):
                    field = _mapped_field(self.builder, items, locator+f'/cell[{column}]', at)
                    fields.append((name, field))
                self.builder.add(locator, fields)
        _labelled(self.builder, self.visible, 'html:text')


def read_document(root, relative_path, *, format=None, limits=None):
    """Read one regular UTF-8 file below root; return immutable grounded records.

    Each retained field includes an exact span of the original source. HTML
    values decode visible text; their raw spans may include inline markup or
    entities. JSON spans include string quotes/escapes; CSV spans include quotes.
    These transformations are recorded by the format-specific locator.
    """
    limits = DocumentLimits() if limits is None else limits
    if not isinstance(limits, DocumentLimits):
        raise TypeError('limits must be DocumentLimits')
    name, text = _read(root, relative_path, limits)
    selected = format or {'.txt': 'text', '.md': 'text', '.csv': 'csv', '.json': 'json',
                          '.html': 'html', '.htm': 'html'}.get(Path(name).suffix.lower())
    if selected not in ('text', 'csv', 'json', 'html'):
        raise DocumentError('Unsupported document format; PDF/OCR and archives are not implemented')
    builder, issues = _Builder(text, limits), ()
    try:
        if selected == 'text':
            _labelled(builder, [(c, i, i+1) for i, c in enumerate(text)])
        elif selected == 'csv':
            _csv(builder)
        elif selected == 'json':
            parser = _JSON(builder)
            scalar = parser.node()
            if scalar is not None:
                builder.add('json:', [('value', scalar)])
            parser.ws()
            if parser.index != len(text):
                raise DocumentError('Trailing JSON content')
        else:
            parser = _HTML(builder)
            parser.feed(text)
            parser.close()
            parser.finish()
            issues = tuple(parser.issues)
    except (RecursionError, json.JSONDecodeError, UnicodeError, OverflowError) as exc:
        raise DocumentError('Malformed or unsupported document content') from exc
    return Document(name, selected, text, hashlib.sha256(text.encode('utf-8')).hexdigest(),
                    tuple(builder.records), limits, issues)


_NUMBER = re.compile(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?\Z', re.ASCII)
_ROLES = {'value', 'unit', 'period', 'name', 'available_at', 'revision_at', 'event_at', 'decision_at', 'period_end'}
_TIME = {'available': 'available_at', 'revision': 'revision_at', 'event': 'event_at',
         'decision': 'decision_at', 'period_end': 'period_end'}


def _nonzero_mantissa(text):
    return any(char in '123456789' for char in re.split('[eE]', text, maxsplit=1)[0])


def _numeric(value):
    if type(value) is str and _NUMBER.fullmatch(value.strip()):
        text = value.strip()
        value = int(text) if re.fullmatch(r'[+-]?\d+', text, re.ASCII) else float(text)
        if type(value) is float and value == 0 and _nonzero_mantissa(text):
            raise ValueError('value_underflows_binary64')
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError('value_is_not_a_finite_plain_number')
    return value


def _timestamp(value):
    if type(value) is not str:
        raise ValueError('time_requires_explicit_timezone')
    stamp = datetime.fromisoformat(value)
    if stamp.tzinfo is None or stamp.utcoffset() is None:
        raise ValueError('time_requires_explicit_timezone')
    return stamp.astimezone(timezone.utc)


def _unit(value):
    return canonical_unit(value)


def extract_quantity(document, quantity_spec, *, selectors, as_of=None):
    """Bind all candidate records to a QuantitySpec.to_dict() fingerprint.

    selectors maps semantic roles to exact field names chosen from this task;
    value is required. name optionally filters an explicit quantity-name field.
    Availability and revision are required for as_of; an explicitly unrevised
    stream uses selectors['revision_at']=None. No latest-vintage selection is
    implicit: multiple eligible candidates, even equal values, stay ambiguous.
    Only scalar extraction is supported. method='literal' explicitly describes
    reading a stated number; other methods remain unverified. Aware timestamps
    are normalised to UTC with source offsets and original spans retained. Only
    timezone='UTC' describes this output; a named source timezone is not inferred
    from an offset. Callers retain candidates/provenance on every outcome.
    """
    if not isinstance(document, Document):
        raise TypeError('document must be a Document')
    if (type(selectors) is not dict or not selectors.get('value') or not set(selectors) <= _ROLES
            or any(not isinstance(value, str) or not value for role, value in selectors.items()
                   if not (role == 'revision_at' and value is None))):
        raise ValueError('selectors must explicitly map supported roles to field names')
    spec = quantity_spec.to_dict()
    if type(spec) is not dict or type(spec.get('name')) is not str or not spec['name']:
        raise ValueError('quantity_spec needs a nonempty name')
    encoded = json.dumps(spec, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    result = {'quantity': spec['name'], 'spec_fingerprint': hashlib.sha256(encoded).hexdigest(),
              'source_sha256': document.sha256, 'status': 'unverified', 'candidates': [], 'issues': list(document.issues)}
    global_issues = []
    if spec.get('shape') not in (None, (), []):
        global_issues.append('non_scalar_quantity_requires_explicit_assembly')
    if spec.get('axes') not in (None, (), []):
        global_issues.append('axes_require_explicit_assembly')
    if spec.get('method') not in (None, 'literal'):
        global_issues.append('method_semantics_require_independent_interpretation')
    if spec.get('unit') is None:
        global_issues.append('quantity_unit_unspecified')
    basis = spec.get('time_basis')
    if basis not in (None, 'static') and basis not in _TIME:
        global_issues.append('unsupported_time_basis')
    requested_times = any(role in selectors and selectors[role] is not None for role in _TIME.values())
    if basis is None and (requested_times or as_of is not None):
        global_issues.append('time_basis_unspecified')
    if spec.get('timezone') not in (None, 'UTC'):
        global_issues.append('named_timezone_requires_explicit_interpretation')
    if spec.get('timezone') is not None and not requested_times:
        global_issues.append('timezone_without_explicit_time_field')
    cutoff = None
    if as_of is not None:
        try:
            cutoff = _timestamp(as_of)
        except (ValueError, TypeError, OverflowError):
            global_issues.append('invalid_as_of')

    output_fields = 0

    def evidence(field):
        nonlocal output_fields
        output_fields += 1
        if output_fields > document.limits.max_output_fields:
            raise DocumentError('Extraction exceeds max_output_fields; no truncation performed')
        return {'source_id': document.source_id, 'sha256': document.sha256,
                'start': field.start, 'end': field.end,
                'quote': document.text[field.start:field.end], 'locator': field.locator}

    for record in document.records:
        amounts = record.get(selectors['value'])
        if not amounts:
            continue
        names = record.get(selectors['name']) if 'name' in selectors else ()
        if names and all(field.value != spec['name'] for field in names):
            continue
        for amount in amounts:
            candidate = {'value': amount.value, 'status': 'unverified', 'locator': record.locator,
                         'provenance': {'value': evidence(amount)}, 'issues': list(global_issues), 'alternatives': {}}
            issues, ambiguous = candidate['issues'], False
            try:
                candidate['value'] = _numeric(amount.value)
            except (ValueError, TypeError, OverflowError):
                issues.append('value_is_not_a_finite_plain_number')
            for role, name in selectors.items():
                if role == 'value' or name is None:
                    continue
                fields = record.get(name)
                if len(fields) > 1:
                    ambiguous = True
                    candidate['alternatives'][role] = [{'value': f.value, 'source': evidence(f)} for f in fields]
                    issues.append('duplicate_'+role)
                elif len(fields) == 1:
                    candidate[role] = fields[0].value
                    candidate['provenance'][role] = evidence(fields[0])
                else:
                    issues.append('missing_'+role)
            if 'name' in selectors and candidate.get('name') != spec['name']:
                issues.append('quantity_name_not_unique')
            try:
                if not isinstance(candidate.get('unit'), str) or _unit(candidate['unit']) != _unit(spec.get('unit')):
                    issues.append('unit_missing_or_mismatch')
            except (TypeError, ValueError):
                issues.append('unit_unrecognized')
            if spec.get('period') is not None and candidate.get('period') != spec['period']:
                issues.append('period_missing_or_mismatch')
            stamps = {}
            candidate['normalized_times'], candidate['source_timezones'] = {}, {}
            for role in _TIME.values():
                if role in candidate:
                    try:
                        stamps[role] = _timestamp(candidate[role])
                        candidate['normalized_times'][role] = stamps[role].isoformat().replace('+00:00', 'Z')
                        original = datetime.fromisoformat(candidate[role])
                        candidate['source_timezones'][role] = original.isoformat()[len(original.replace(tzinfo=None).isoformat()):]
                    except (ValueError, TypeError, OverflowError):
                        issues.append('invalid_'+role)
            if stamps:
                candidate['normalized_timezone'] = 'UTC'
            if basis in _TIME and _TIME[basis] not in stamps:
                issues.append('missing_time_basis_field')
            excluded = False
            if as_of is not None:
                required = ['available_at'] + ([] if selectors.get('revision_at', '') is None else ['revision_at'])
                if cutoff is None or any(role not in stamps for role in required):
                    issues.append('availability_or_revision_unverified')
                elif any(stamps[role] > cutoff for role in required):
                    excluded = True
                    issues.append('not_available_by_cutoff')
            candidate['status'] = ('ambiguous' if ambiguous else 'excluded' if excluded else
                                   'unverified' if issues else 'eligible')
            result['candidates'].append(candidate)
            if len(result['candidates']) > document.limits.max_candidates:
                raise DocumentError('Extraction exceeds max_candidates; no truncation performed')
    candidates = result['candidates']
    eligible = [row for row in candidates if row['status'] == 'eligible']
    if any(row['status'] == 'ambiguous' for row in candidates) or len(eligible) > 1:
        result['status'] = 'ambiguous'
    elif (len(eligible) == 1 and not document.issues
          and all(row['status'] in ('eligible', 'excluded') for row in candidates)):
        result.update(status='extracted', value=eligible[0]['value'], unit=eligible[0]['unit'])
    elif candidates and all(row['status'] == 'excluded' for row in candidates):
        result['status'] = 'excluded'
    if not candidates:
        result['issues'].append('no_explicitly_labelled_candidate')
    return result
