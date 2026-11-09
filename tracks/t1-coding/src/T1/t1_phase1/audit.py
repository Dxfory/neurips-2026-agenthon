"""Checks of a finished program's outputs against the contract its instruction states.

The solver used to stop at the first clean exit, so outputs that plainly contradict the instruction (a
missing file or wrapper key, NaN in a required field, a residual the task says must be below 1e-10) were
graded as they were. These helpers are deterministic and read only the instruction and the output tree:
output_contract() takes the files and JSON template keys the instruction names, contract_flags() lists
what the outputs lack, digest() summarises the outputs for the model's review, and review_items() keeps a
review line only when it quotes the instruction and names an output shown in the digest. literal_rules() reads
the literal output formats the instruction states; normalize_outputs() rewrites only those, and literal_flags()
and degenerate_flags() list what is left for the review as advisory flags. contract_rules() reads the output
contract the instruction states (CSV headers, JSON key names and order, integer columns, slots a template elides
with '...'), and normalize_contract() applies it with normalize_outputs(), file by file, only where that leaves
fewer problems and none new (T1_CONTRACT).
"""
from collections import Counter
import csv
import difflib
import io
import json
import math
import os
from pathlib import Path
import re

_OUTPUT = re.compile(r'/(?:app/)?output/([A-Za-z0-9_][A-Za-z0-9_.\-/]*\.[A-Za-z0-9]+)')
_NOT_OURS = {'reward.json', 'pytest_report.json', '_agent_status.json'}
_FENCE = re.compile(r'```[A-Za-z]*\n(.*?)```', re.S)
_TEMPLATE_JSON = re.compile(r'(?<![A-Za-z0-9_.:/\\-])'
                            r'((?:/?[A-Za-z0-9_.-]+/)*[A-Za-z0-9_][A-Za-z0-9_.-]*\.json)\b')
_TOKEN = re.compile(r'"(?:[^"\\\n]|\\.)*"|[{}\[\]:,]|//[^\n]*|#[^\n]*')
_NAN_ALLOWED = re.compile(r'\b(?:NaN|nan|null|None)\b')
TEMPLATE_WINDOW = 800  # characters before a JSON block searched for the file it describes
DIGEST_LIMIT = 6000
JSON_SHOWN = 1800
CSV_ROWS = 200_000
CSV_COLUMNS = 12


_DELIVERY_HEADING = re.compile(r'\b(?:outputs?|deliverables?)\b', re.I)
_DELIVERY_LIST = re.compile(r'^\s*(?:[-*+] |\d+[.)]\s+)(.*)$')
# A dot in a field name is not a file declaration. Bare names need a known artifact format;
# explicit /output paths retain the existing arbitrary-extension contract.
_DELIVERY_FORMATS = frozenset({'csv', 'tsv', 'json', 'jsonl', 'parquet', 'feather', 'npy', 'npz',
                                'txt', 'md', 'html', 'pdf', 'png', 'jpg', 'jpeg', 'svg', 'webp',
                                'xlsx', 'xls', 'yaml', 'yml'})
_NON_DELIVERY = re.compile(r'\b(?:optional(?:ly)?|examples?|illustrat\w*|may|might|could|if)\b|\be\.g\.'
                           r'|\bnever\b|\b(?:do|does|must|should|is|are)\s+not\b|\bnot\s+(?:required|needed)\b',
                           re.I)
# A heading excludes every section below it, so it needs a narrower test than a sentence: a qualifier, a heading that
# says its files are not outputs, or a negation of writing that refers to the heading's own files (the verb ends the
# heading or a clause of it, or takes only words such as "these", "them", "the files" after it, and then perhaps the
# output directory as its destination: "Do not write these to the output directory", "Never write the files below
# to /app/output", "Do not write to /app/output"; or the passive at the start or after the files: "Must not be
# written: scratch files", "Files that must not be modified"). A bare "never" or "must not" (a rule such as "Never
# use future data", or a banner), or a negation about something else ("do not write into /app/data", "do not create
# any other files", "must not modify the inputs", "inputs must not be modified") does not. Words that describe the
# files as scratch work ("Do not submit these intermediate files") count only in a heading whose main clause names
# no outputs: "## Outputs (do not write intermediate files)" keeps its deliverables.
_WRITING = r'(?:write|create|produce|save|emit|modify|edit|overwrite|change|alter|delete|remove|submit|include|commit)'
_WRITTEN = (r'(?:written|created|produced|saved|emitted|modified|edited|overwritten|changed|altered|deleted|removed'
            r'|submitted|included|committed)')
_HEADING_OBJECT = (r'(?:or|and|any|of|them|it|these|this|those|such|the|following|listed|below|above|here|output|result'
                   r'|files?|outputs?|deliverables?|results?|' + _WRITING + r')')
_SCRATCH_OBJECT = r'(?:intermediate|scratch|temporary|temp|debug(?:ging)?)'
_HEADING_END = r'[\s`*]*(?:[.,;:!?)\]]|\s[-\u2013\u2014]|$)'
# The output directory (a literal, or a masked output path: _delivery_prose blanks it) after the verb and its objects.
_HEADING_DESTINATION = (r'\s+(?:to|into|in|under)(?:\s+(?:the|your))?'
                        r'(?:\s+[`*]*(?:/?(?:app/)?output/?|outputs?\s+(?:dir(?:ectory)?|folder))'
                        r'|(?=' + _HEADING_END + r'))')
_NEGATED_WRITING = r"\b(?:(?:do|does|must|should)(?:\s+not|n['\u2019]?t)|never)\s+" + _WRITING
_PASSIVE_WRITING = (r"(?:^[#\s*`]*|\b(?:files?|these|they|those|this|it)\s+(?:(?:that|which)\s+)?)"
                    r"(?:(?:must|should|need)\s+not|(?:must|should)n['\u2019]?t|(?:is|are)\s+not\s+to|(?:must|should)?"
                    r"\s*never)\s+be\s+" + _WRITTEN + '(?:' + _HEADING_DESTINATION + ')?')


def _negated_writing(objects):
    return _NEGATED_WRITING + r'(?:\s+' + objects + r')*(?:' + _HEADING_DESTINATION + ')?' + _HEADING_END


_NON_DELIVERY_HEADING = re.compile(
    r'\b(?:optional(?:ly)?|examples?|illustrat\w*|may|might|could|if)\b|\be\.g\.|\bnot\s+(?:required|needed)\b'
    r'|\bnot\s+(?:(?:an?|the)\s+)?(?:outputs?|deliverables?|graded|checked|evaluated|scored)\b'
    r'|' + _negated_writing(_HEADING_OBJECT) + r'|' + _PASSIVE_WRITING + _HEADING_END, re.I)
_SCRATCH_HEADING = re.compile(_negated_writing(r'(?:' + _HEADING_OBJECT + r'|' + _SCRATCH_OBJECT + r')'), re.I)
# The part of a heading before a parenthesis, a bracket, a colon or a dash; when it names outputs or deliverables, an
# "inputs" in the rest ("## Outputs (you must not modify the inputs)") does not make the heading an input section.
_HEADING_CLAUSE = re.compile(r'[(\[:]|\s[-\u2013\u2014]\s')
_DELIVERY_INPUT = re.compile(r'\binputs?\b|^\s*(?:read|load)\b', re.I)
_OPTIONAL_FILE = re.compile(r'\b(?:(?:this|the)\s+(?:file|output|deliverable)|it)\s+(?:is\s+)?'
                            r'(?:optional|not\s+(?:required|needed))\b'
                            r'|^\s*(?:(?:optional|optionally)(?=\s*(?:[.:;]|$))|only if\b)',
                            re.I | re.M)
_FILE_TOKEN = re.compile(r'(?:/?[A-Za-z0-9_.-]+/)*[A-Za-z0-9_][A-Za-z0-9_.-]*\.[A-Za-z0-9]+')
_DELIVERY_BOUNDARY = re.compile(
    r'[;!?]|(?<!e\.g)\.(?=\s|$)|(?:,\s*but\s+|\s+(?:and|but)\s+'
    r'|,\s*(?=always|optionally|may|might|could|never|do not|must not|should not))'
    r'(?=(?:(?:always|optionally|may|might|could|never|do not|must not|should not)\s+)*'
    r'(?:write|save|create|produce|emit)\b)', re.I)
_FILE_SUFFIX_EXCLUSION = re.compile(
    r'^(?:(?:is|are)\s+)?(?:optional(?:ly)?(?=\s*(?:[.,;)]|$|(?:file|output|deliverable|only)\b))'
    r'|not\s+(?:required|needed)\b|(?:only\s+)?if\b|examples?(?:\s+only)?(?=\s*(?:[.,;]|$)))'
    r'|^(?:may|might|could)\s+be\s+(?:omitted|skipped)\b', re.I)


def _delivery_prose(text):
    """Mask file literals at equal width, so their words and dots are not prose qualifiers."""
    def mask(match):
        token = match.group(0)
        known = Path(token).suffix.lower().lstrip('.') in _DELIVERY_FORMATS
        return ' ' * len(token) if known or token.startswith(('/app/output/', '/output/')) else token
    return _FILE_TOKEN.sub(mask, text)


def _heading_excludes(heading):
    """Whether a heading that declares no file of its own excludes every line of its section from delivery."""
    prose = _delivery_prose(heading)
    if _NON_DELIVERY_HEADING.search(prose):
        return True
    main = _HEADING_CLAUSE.split(prose, 1)[0]
    if _DELIVERY_HEADING.search(main):
        return bool(_DELIVERY_INPUT.search(main))
    return bool(_SCRATCH_HEADING.search(prose) or _DELIVERY_INPUT.search(prose))


def _file_excluded(text, start, end):
    """Only the file's declaration clause, not a later content constraint, controls delivery.

    Split explicit sentence/semicolon boundaries and coordinated write/save actions.
    A trailing qualifier must describe file creation/omission directly. This is a
    bounded declaration grammar, not inference about arbitrary prose references.
    """
    prose = _delivery_prose(text)
    edges = list(_DELIVERY_BOUNDARY.finditer(prose))
    left = max((m.end() for m in edges if m.end() <= start), default=0)
    right = min((m.start() for m in edges if m.start() >= end), default=len(text))
    before = prose[left:start].strip(' \t`*:#-')
    after = prose[end:right].strip(' \t`*:(\u2014-')
    return bool(_NON_DELIVERY.search(before) or re.match(r'^(?:read|load)\b', before, re.I)
                or _FILE_SUFFIX_EXCLUSION.search(after) or _OPTIONAL_FILE.search(after))


def _required_declared_files(text):
    declared = set(_declared_files(text))
    return {name for match in _FILE_TOKEN.finditer(text)
            if (name := _output_name(match.group(0))) in declared
            and not _file_excluded(text, match.start(), match.end())}


def _output_name(token):
    """A literal path relative to the output root; never normalize away unsafe components."""
    name = token.rstrip('.,:')
    for root in ('/app/output/', '/output/'):
        if name.startswith(root):
            name = name[len(root):]
            break
    if (not re.fullmatch(r'(?:[A-Za-z0-9_][A-Za-z0-9_.-]*/)*'
                         r'[A-Za-z0-9_][A-Za-z0-9_.-]*\.[A-Za-z0-9]+', name)
            or Path(name).name in _NOT_OURS):
        return None
    return name


def _delivery_context(lines, i, fenced):
    """The first prose paragraph after a declaration, stopping before another item or code."""
    parts, started = [], False
    item = _DELIVERY_LIST.match(lines[i])
    for j in range(i + 1, len(lines)):
        line = lines[j]
        if j in fenced or _HEADING.match(line) or _DELIVERY_LIST.match(line):
            break
        if not line.strip():
            if started:
                break
            continue
        if item and len(line) - len(line.lstrip()) <= len(lines[i]) - len(lines[i].lstrip()):
            break
        parts.append(line)
        started = True
    return '\n'.join(parts)


def _declared_files(text):
    """One or more leading literal file names, optionally joined by comma/and; not prose references."""
    text = re.sub(r'^(?:file\s+)?\d+[.):]\s*', '', text.strip(), flags=re.I)
    found = []
    while text:
        first = re.match(r'^[`*]*([^\s`*,]+)[`*]*', text)
        if not first:
            break
        name = _output_name(first.group(1))
        if not name or not (Path(name).suffix.lower().lstrip('.') in _DELIVERY_FORMATS
                            or first.group(1).startswith(('/app/output/', '/output/'))):
            break
        found.append(name)
        rest = text[first.end():].lstrip()
        join = re.match(r'(?:,\s*(?:and\s+)?|and\s+|&\s+)', rest, re.I)
        if not join:
            break
        text = rest[join.end():].lstrip()
    return found


def _files(instruction):
    """Explicit output paths, plus file headings/list items in an unambiguous delivery section.

    Bare prose mentions, ambiguous sections, examples and conditional/optional declarations stay unknown.
    This is deliberately not a general natural-language requirements extractor.
    """
    names, lines = set(), instruction.splitlines()
    fenced = _fenced_lines(lines)
    tilde = False
    for i, line in enumerate(lines):
        if line.lstrip().startswith('~~~') and i not in fenced:
            tilde = not tilde
            fenced.add(i)
        elif tilde:
            fenced.add(i)
    sections = _sections(lines, fenced)
    list_excluded, file_column = False, None
    excluded_body, lead_indent = None, 0
    for i, line in enumerate(lines):
        if i in fenced or line.lstrip().startswith('>'):
            continue
        heading, item = _HEADING.match(line), _DELIVERY_LIST.match(line)
        if heading:
            list_excluded, excluded_body = False, None
        elif list_excluded:
            # Blank lines before the first body/list item do not end a lead-in.
            # A list owns its indented continuations, not a subsequent prose paragraph.
            indent = len(line) - len(line.lstrip())
            if (excluded_body == 'paragraph' and not line.strip()
                    or excluded_body == 'list' and line.strip() and not item and indent <= lead_indent):
                list_excluded, excluded_body = False, None
            elif line.strip():
                excluded_body = excluded_body or ('list' if item else 'paragraph')
        if not heading and not item and line.rstrip().endswith(':'):
            prose = _delivery_prose(line)
            list_excluded = bool(_NON_DELIVERY.search(prose) or _DELIVERY_INPUT.search(prose))
            excluded_body, lead_indent = None, len(line) - len(line.lstrip())
        ancestors = [lines[h] for h in sections[i]]
        output_section = any(_DELIVERY_HEADING.search(h) for h in ancestors)
        cells = [cell.strip(' `*').lower() for cell in line.strip().strip('|').split('|')]
        table = line.lstrip().startswith('|')
        if not table:
            file_column = None
        elif any(cell in {'file', 'filename', 'file name'} for cell in cells):
            columns = [j for j, cell in enumerate(cells) if cell in {'file', 'filename', 'file name'}]
            file_column = columns[0] if len(columns) == 1 else None
        if any((not _required_declared_files(_HEADING.match(h).group(2))
                if _declared_files(_HEADING.match(h).group(2)) else
                _heading_excludes(h))
               for h in ancestors):
            continue
        declaration = heading.group(2) if heading else item.group(1) if item else line
        if list_excluded:
            continue
        if (heading or item) and _OPTIONAL_FILE.search(_delivery_context(lines, i, fenced)):
            continue
        if table:
            if output_section and file_column is not None and file_column < len(cells):
                # Preserve case in file names; only the header matching above is case insensitive.
                cell = line.strip().strip('|').split('|')[file_column].strip()
                description = ' '.join(c for j, c in enumerate(line.strip().strip('|').split('|'))
                                       if j != file_column)
                names.update(_required_declared_files(cell + ': ' + description.strip()))
            continue
        if (heading or item) and output_section:
            if heading:
                declaration = re.sub(r'^(?:required\s+)?(?:outputs?|deliverables?)\s*:\s*', '', declaration,
                                     flags=re.I)
            declared = _declared_files(declaration)
            if declared:
                names.update(_required_declared_files(declaration))
                continue
        for match in _OUTPUT.finditer(line):
            # Do not match the suffix of a URL, a longer absolute path, or an escaped path.
            if match.start() and line[match.start() - 1] not in ' \t`\'"([':
                continue
            name = _output_name(match.group(0))
            if name and not _file_excluded(line, match.start(), match.end()):
                names.add(name)
    return sorted(names)


def _segment(key):
    return '*' if re.search(r'[<>{}]|\.\.\.', key) else key


def template_keys(block):
    """Key paths of a JSON-like template; placeholders (<float>, "<ticker>"), comments and ellipses are
    tolerated. A key in angle brackets matches any key; '[]' stands for the elements of a list."""
    tokens = [m.group(0) for m in _TOKEN.finditer(block) if not m.group(0).startswith(('//', '#'))]
    paths, stack = set(), []  # frames: [kind, current key]
    for i, token in enumerate(tokens):
        if token in ('{', '['):
            stack.append([token, None])
        elif token in ('}', ']'):
            if stack:
                stack.pop()
        elif token.startswith('"') and i + 1 < len(tokens) and tokens[i + 1] == ':' and stack and stack[-1][0] == '{':
            try:
                key = json.loads(token)
            except ValueError:
                continue
            stack[-1][1] = _segment(key)
            prefix = tuple(frame[1] if frame[0] == '{' else '[]' for frame in stack[:-1])
            if None not in prefix:
                paths.add(prefix + (stack[-1][1],))
    return paths


def _json_templates(instruction, files=None):
    """(output file, template block, fence match) for each fenced block starting with { or [ that follows a
    .json declaration within TEMPLATE_WINDOW characters. Use the nearest declaration line, exact paths first;
    a line naming multiple files or a basename shared by multiple outputs does not identify one target."""
    files = _files(instruction) if files is None else files
    for match in _FENCE.finditer(instruction):
        block = match.group(1)
        if not block.lstrip().startswith(('{', '[')):
            continue
        before = instruction[max(0, match.start() - TEMPLATE_WINDOW):match.start()]
        mentioned = list(_TEMPLATE_JSON.finditer(before))
        if not mentioned:
            continue
        start = before.rfind('\n', 0, mentioned[-1].start()) + 1
        references = [m.group(1) for m in mentioned if m.start() >= start]
        named = {_output_name(name) for name in references}
        if len(named) != 1 or None in named:
            continue
        name = named.pop()
        if '/' not in name and not any(ref.startswith(('/app/output/', '/output/')) for ref in references):
            choices = [f for f in files if Path(f).name == name]
            if len(choices) > 1:
                continue
            name = choices[0] if choices else name
        yield name, block, match


# A key the prose says an output reports: "... is reported as the `x` <container>", "record the count as `x` in
# <file>.json". The verb and "as `x`" stand in one sentence; a negation before either leaves it out.
_PROSE_KEY = re.compile(r'\b(report|record|output|write|written|save|store|include|export)\w*\b([^.`\n]{0,60}?)'
                        r'\bas\s+(?:the\s+|an?\s+)?`([A-Za-z_][A-Za-z0-9_]{0,63})`(?:[ \t]*\n?[ \t]*([A-Za-z]{2,24}))?',
                        re.I)
_NEGATION = re.compile(r"\b(?:not|never|no|nor|without)\b|n't\b", re.I)
# The file a sentence-placed key goes into must follow the key at once: "as `x` in /app/output/<file>.json".
_GAP = r'(?:[ \t]*\n[ \t]*|[ \t]+)'  # a space or one line break
_JSON_AFTER = re.compile(rf'`{_GAP}(?:in|to|into|inside|within){_GAP}(?:the{_GAP})?`?'
                         r'(?:/?[A-Za-z0-9_.\-]+/)*([A-Za-z0-9_\-]+\.json)\b', re.I)
_DOTTED = re.compile(r'\b([A-Za-z0-9_\-]+\.[A-Za-z][A-Za-z0-9]{1,7})\b')  # a file name (or any dotted name)
# A value or type word is never a key: "report undefined ratios as `null`", "write numbers as `float`".
_LITERAL = frozenset({'null', 'none', 'nan', 'inf', 'infinity', 'true', 'false', 'float', 'int', 'str', 'bool',
                      'string', 'number', 'integer', 'boolean', 'object', 'array', 'list', 'dict', 'tuple',
                      'float32', 'float64', 'int32', 'int64'})


def _forms(word):
    return {word, word + 's', word + 'es'} | ({word[:-1] + 'ies'} if word.endswith('y') else set())


def _names_container(word, name):
    """Whether a word names a template container (the same word, singular or plural)."""
    word, name = word.lower(), name.lower()
    return name in _forms(word) or word in _forms(name)


def _sentence(text, start, end):
    """The sentence around text[start:end]: from after the previous full stop or blank line to the next one."""
    before = max(text.rfind('. ', 0, start), text.rfind('\n\n', 0, start), text.rfind('.\n', 0, start))
    after = [i for i in (text.find('. ', end), text.find('\n\n', end), text.find('.\n', end)) if i >= 0]
    return text[before + 1:min(after) if after else len(text)]


def prose_keys(instruction, json_keys):
    """{file: {key path}} of output keys the prose names but no template has, placed only where the TASK leaves no
    doubt: "reported as the `x` <word>" where <word> names exactly one object in all the JSON templates (the
    key goes into it), or "... as `x` in <file>.json" where the file follows the key at once, is templated with a
    flat template (a top-level key) and is the only file (or other dotted name) the sentence names. A key some
    template already has anywhere, a value or type word (`null`, `float`) or one after a negation is left out;
    anything else is too."""
    known = {segment for keys in json_keys.values() for path in keys for segment in path}
    containers = {}
    for name, keys in json_keys.items():
        for path in keys:
            for depth in range(1, len(path)):
                if path[depth - 1] not in ('*', '[]') and '[]' not in path[:depth]:
                    containers.setdefault(path[depth - 1].lower(), set()).add((name, path[:depth]))
    found = {}
    for match in _PROSE_KEY.finditer(instruction):
        verb, between, key, word = match.group(1), match.group(2), match.group(3), match.group(4)
        lead = instruction[max(0, match.start() - 40):match.start()].split('\n')[-1]
        if key in known or key.lower() in _LITERAL or _NEGATION.search(lead + ' ' + between):
            continue
        places = set()
        if word:
            places = {place for name, spots in containers.items() if _names_container(word, name) for place in spots}
        if len(places) == 1:
            name, path = places.pop()
            found.setdefault(name, set()).add(path + (key,))
            continue
        after = _JSON_AFTER.match(instruction, match.end(3))
        if not after:
            continue
        sentence = _sentence(instruction, match.start(), match.end())
        named = {m.group(1) for m in _DOTTED.finditer(sentence)}
        files = [name for name in json_keys if Path(name).name == after.group(1)]
        if named == {after.group(1)} and len(files) == 1 and all(len(path) == 1 for path in json_keys[files[0]]):
            found.setdefault(files[0], set()).add((key,))
    return found


def output_contract(instruction):
    """{'files': output paths the instruction names, 'json': {file: template key paths, and the keys only the
    prose names (prose_keys)}, 'nan_allowed'}."""
    files, json_keys = _files(instruction), {}
    for target, block, _ in _json_templates(instruction, files):
        keys = template_keys(block)
        if keys:
            json_keys.setdefault(target, set()).update(keys)
    for target, keys in prose_keys(instruction, json_keys).items():
        json_keys[target] |= keys
    return {'files': files, 'json': json_keys, 'nan_allowed': bool(_NAN_ALLOWED.search(instruction))}


def _child(obj, segment):
    if segment == '[]':
        return obj[:1] if isinstance(obj, list) else None
    if not isinstance(obj, dict):
        return None
    if segment == '*':
        return list(obj.values())
    return [obj[segment]] if segment in obj else None


def _walk(obj, path):
    """Values at path (wildcards expand); [] when an empty container stops the walk; None when missing."""
    values = [obj]
    for segment in path:
        found = []
        for value in values:
            children = _child(value, segment)
            if children is None:
                if segment in ('*', '[]') and isinstance(value, (dict, list)) and not value:
                    continue
                return None
            found += children
        values = found
    return values


def _finite(value):
    return not (isinstance(value, float) and (math.isnan(value) or math.isinf(value)))


def contract_flags(contract, out):
    """Short lines naming what the outputs lack: missing files, the shallowest missing template keys, and
    non-finite numbers at template keys unless the instruction speaks of NaN or null values."""
    flags = []
    for name in contract['files']:
        if not (out / name).exists():
            flags.append(f'missing file {name}')
    for name, keys in sorted(contract['json'].items()):
        path = out / name
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(errors='replace'))
        except ValueError:
            flags.append(f'{name}: not valid JSON')
            continue
        missing = sorted(key for key in keys if _walk(data, key) is None)
        for key in missing:
            if not any(other != key and key[:len(other)] == other for other in missing):
                flags.append(f'{name}: missing key ' + '.'.join(key))
        if contract.get('nan_allowed'):
            continue
        for key in sorted(keys):
            values = _walk(data, key) or []
            if any(not _finite(v) for v in values):
                flags.append(f'{name}: ' + '.'.join(key) + ' is not a finite number')
    return flags


def _shown_json(path):
    try:
        text = json.dumps(json.loads(path.read_text(errors='replace')), separators=(', ', ': '))
    except ValueError:
        text = path.read_text(errors='replace')
    return text if len(text) <= JSON_SHOWN else text[:JSON_SHOWN] + f' ... ({len(text)} chars)'


def _number(text):
    try:
        value = float(text)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def _shown_csv(path):
    with path.open(newline='', errors='replace') as stream:
        reader = csv.reader(stream)
        header = next(reader, None)
        if header is None:
            return 'empty'
        first, last, count = None, None, 0
        low, high = {}, {}
        for row in reader:
            if count >= CSV_ROWS:
                break
            count += 1
            first = first or row
            last = row
            for i, cell in enumerate(row[:CSV_COLUMNS]):
                value = _number(cell)
                if value is not None:
                    low[i] = min(low.get(i, value), value)
                    high[i] = max(high.get(i, value), value)
    lines = [f'{count} rows; columns ' + ', '.join(header[:40]) + (' ...' if len(header) > 40 else '')]
    ranges = [f'{header[i]}: min {low[i]} max {high[i]}' for i in sorted(low) if i < len(header)]
    if ranges:
        lines.append('  ' + '; '.join(ranges))
    if first is not None:
        lines.append('  first ' + ','.join(first)[:300])
        lines.append('  last ' + ','.join(last)[:300])
    return '\n'.join(lines)


def digest(out, limit=DIGEST_LIMIT):
    """The outputs as the model's review sees them: small JSON verbatim, CSV shape, ranges and end rows,
    other files by size. Bounded by limit characters."""
    parts = []
    for path in sorted(p for p in out.rglob('*') if p.is_file() and not p.is_symlink()):
        name = path.relative_to(out).as_posix()
        if path.name in _NOT_OURS or len(path.relative_to(out).parts) > 3:
            continue
        try:
            if path.suffix == '.json':
                parts.append(f'{name}: {_shown_json(path)}')
            elif path.suffix in ('.csv', '.tsv'):
                parts.append(f'{name}: {_shown_csv(path)}')
            else:
                parts.append(f'{name}: {path.stat().st_size} bytes')
        except (OSError, UnicodeError, csv.Error):
            parts.append(f'{name}: unreadable')
    text = '\n'.join(parts)
    return text if len(text) <= limit else text[:limit - 20] + '\n... (digest cut)'


def _norm(text):
    return ' '.join(re.sub(r'[`*_]', '', text).lower().split())


def _quoted(quote, instruction):
    quote, text = _norm(quote.strip(' "\'')), _norm(instruction)
    if len(quote) < 15:
        return False
    if quote in text:
        return True
    match = difflib.SequenceMatcher(None, text, quote, autojunk=False).find_longest_match(0, len(text), 0, len(quote))
    return match.size >= 0.8 * len(quote)


_ITEM = re.compile(r'QUOTE:\s*(.+?)\s*\|\s*OUTPUT:\s*(.+?)\s*\|\s*PROBLEM:\s*(.+)')
_ITEM_START = re.compile(r'\bQUOTE:')
REVIEW_ITEM_LIMIT = 8
# A PROBLEM that only confirms compliance ("which matches the requirement", "no contradiction found") is not an
# item. Dropping a real problem hides it from the fix and lets a fix that left it standing pass the gate, so the
# filter errs toward keeping: an explicit verdict that nothing is wrong decides only when no other scope is named
# ("no contradiction otherwise") and nothing before it contrasts the output with the TASK ("whereas", "vs",
# "requires", "exceeds"); an affirming phrase counts only when no objection or contrast remains once it (and
# "the task does not require ...") is removed, so that "is not a contradiction" is never read as an objection.
_VERDICT = re.compile(r'\b(?:no|not an?) (?:contradictions?|issues?|problems?|violations?|discrepanc(?:y|ies)|'
                      r'errors?|mismatch(?:es)?)\b', re.I)
_VERDICT_SCOPE = re.compile(r'\b(?:otherwise|else|elsewhere|besides|apart from|aside from|other than|beyond|'
                            r'for the rest|remaining|in other respects|in all other)\b', re.I)
_AFFIRM = re.compile(r'\b(?:is|are|looks|appears|seems|was|were|remains?) (?:all )?(?:correct|fine|consistent|right|'
                     r'valid|ok|okay|satisfied|compliant)\b'
                     r'|\b(?:as|which is|which are|is) (?:required|expected|specified|stated|correct)\b'
                     r'|\b(?:has|have|uses?|contains?|shows?|with) the (?:correct|right|required|expected)\b'
                     r'|\bmatch(?:es|ing)? (?:the|exactly|as|what)\b|\bsatisf(?:ies|ied|y|ying)\b'
                     r'|\bcompl(?:ies|iant|y|ying)\b|\bconsistent with\b|\bcorrectly\b', re.I)
_NO_DEMAND = re.compile(r"\b(?:does|do|did)(?: not|n't) (?:require|specify|mandate|say|state|forbid|prohibit|ask)\w*"
                        r'|\bnot (?:required|specified|needed|necessary|mandated|stated|asked)\b', re.I)
# Words that set the output against the TASK or against another value. Comparatives that also confirm a bound
# ("below the bound, as required") and adjectives such as "required keys" are left out.
_CONTRAST = (r'whereas|while|yet|although|though|vs\.?|versus|instead|rather than|only|still|unchanged|'
             r'requires?|expects?|excludes?|implies|demands?|needs?|exceed\w*|far (?:above|below|from)|'
             r'differs?|different|inconsistent|incompatible|implausibl\w*|unreasonabl\w*|impossible|invalid|'
             r'unexpected\w*')
_OBJECTION = re.compile(r'\b(?:should|must|but|however|incorrect(?:ly)?|wrong(?:ly)?|missing|not|no longer|'
                        r"doesn't|don't|isn't|aren't|fail\w*|errone\w*|broken|incomplete|omit\w*|lack\w*|"
                        r'violat(?:es|ed|ing)|contradicts?|mismatch(?:es)?|off by|'
                        r'too (?:high|low|large|small|many|few)|nan|never|absent|' + _CONTRAST + r')(?!\w)', re.I)
_HARD = re.compile(r'\b(?:' + _CONTRAST + r')(?!\w)', re.I)


def _affirms(problem):
    """Whether a PROBLEM only confirms that the output complies."""
    verdicts = [m for m in _VERDICT.finditer(problem)
                if not _VERDICT_SCOPE.search(problem[max(0, m.start() - 24):m.end() + 24])]
    if verdicts:
        before = _NO_DEMAND.sub(' ', _AFFIRM.sub(' ', problem[:verdicts[-1].start()]))
        return not _HARD.search(before) and not _OBJECTION.search(_NO_DEMAND.sub(' ', problem[verdicts[-1].end():]))
    return bool(_AFFIRM.search(problem)) and not _OBJECTION.search(_NO_DEMAND.sub(' ', _AFFIRM.sub(' ', problem)))


def review_parse(reply, instruction, shown):
    """(entries, affirmations): the (quote, output, problem) of each item of the model's review that quotes the
    instruction and names an output key or column the digest shows, and how many such items were dropped
    because their PROBLEM only confirms compliance. A line that packs several QUOTE items gives one entry each;
    the same quote and problem are kept once."""
    entries, seen, affirmations = [], set(), 0
    for line in (reply or '').splitlines():
        starts = [m.start() for m in _ITEM_START.finditer(line)]
        for begin, end in zip(starts, starts[1:] + [len(line)]):
            match = _ITEM.search(line[begin:end].strip().rstrip('|').strip())
            if not match:
                continue
            quote, output, problem = (group.strip() for group in match.groups())
            problem = re.sub(r'\s*\|\s*PROBLEM:\s*', '; ', problem).strip(' |')
            token = re.split(r'[:.\[/]', output.strip('` '))[-1].strip('` "\'')
            if not token or token not in shown or not _quoted(quote, instruction):
                continue
            if _affirms(problem):
                affirmations += 1
                continue
            key = (_norm(quote.strip(' "\'')), _norm(problem))
            if key not in seen:
                seen.add(key)
                entries.append((quote.strip(chr(34)), output, problem))
    return entries[:REVIEW_ITEM_LIMIT], affirmations


def review_entries(reply, instruction, shown):
    """review_parse's entries alone."""
    return review_parse(reply, instruction, shown)[0]


def format_entry(entry):
    quote, output, problem = entry
    return f'"{quote}" -> {output}: {problem}'


def review_items(reply, instruction, shown):
    """review_entries as the lines of the follow-up request."""
    return [format_entry(entry) for entry in review_entries(reply, instruction, shown)]


def same_quote(first, second):
    """Whether two quotes of the instruction cite the same passage: one contains the other, or they share a
    run of at least 80% of the shorter one."""
    a, b = _norm(first.strip(' "\'')), _norm(second.strip(' "\''))
    if not a or not b:
        return False
    if a in b or b in a:
        return True
    shorter = min(len(a), len(b))
    match = difflib.SequenceMatcher(None, a, b, autojunk=False).find_longest_match(0, len(a), 0, len(b))
    return match.size >= 0.8 * shorter


_TOLERANCE_WORDS = ('residual', 'error', 'rmse', 'diff', 'parity', 'deviation', 'discrepanc', 'mismatch')
_BOUND = re.compile(r'(?:<=|<|≤|\bless than\b|\bbelow\b|\bat most\b|\bsmaller than\b|\bwithin\b|\bunder\b)\s*\$?\s*'
                    r'(10\s*\^\s*\{?\s*-\s*\d+\s*\}?|\d*\.?\d+(?:[eE][-+]?\d+)?)', re.I)
_IDENT = re.compile(r'\b[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+\b')
BOUND_MARGIN = 100.0  # a value is flagged only when it exceeds the stated tolerance this many times over


def _bound_value(text):
    power = re.fullmatch(r'10\s*\^\s*\{?\s*-\s*(\d+)\s*\}?', text)
    return 10.0 ** -int(power.group(1)) if power else float(text)


def _tolerances(instruction):
    """(sentence, bound, identifiers, words) for sentences that bound an error-like quantity below 1."""
    found = []
    for line in instruction.splitlines():
        for sentence in re.split(r'(?<=[.!?])\s+(?=[A-Z`*\-])', line.strip()):
            lower = sentence.lower()
            words = [w for w in _TOLERANCE_WORDS if w in lower]
            match = _BOUND.search(sentence)
            if not words or not match:
                continue
            try:
                bound = _bound_value(match.group(1))
            except ValueError:
                continue
            if not 0 < bound < 1:
                continue
            idents = [i.lower() for i in _IDENT.findall(sentence) if any(w in i.lower() for w in _TOLERANCE_WORDS)]
            found.append((sentence.strip(), bound, idents, words))
    return found


def _json_numbers(obj, path=()):
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield from _json_numbers(value, path + (str(key),))
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool) and path:
        yield path, float(obj)


def _csv_extremes(path):
    with path.open(newline='', errors='replace') as stream:
        reader = csv.reader(stream)
        header = next(reader, None) or []
        top = {}
        for count, row in enumerate(reader):
            if count >= CSV_ROWS:
                break
            for i, cell in enumerate(row[:len(header)]):
                value = _number(cell)
                if value is not None:
                    top[i] = max(top.get(i, 0.0), abs(value))
    return [(header[i], value) for i, value in sorted(top.items())]


def bound_flags(instruction, out):
    """Output fields that exceed an error-like tolerance the instruction states (below 1) by BOUND_MARGIN
    times: JSON numbers and CSV columns (their largest absolute value) whose name carries the identifier the
    sentence names, or else one of its tolerance words. Each field is reported once."""
    rules = _tolerances(instruction)
    if not rules:
        return []
    flags, seen = [], set()

    def rule_for(name):
        name = name.lower()
        for sentence, bound, idents, words in rules:
            if (any(i in name for i in idents) if idents else any(w in name for w in words)):
                return sentence, bound
        return None

    for path in sorted(p for p in out.rglob('*') if p.is_file() and not p.is_symlink()):
        name = path.relative_to(out).as_posix()
        if path.name in _NOT_OURS:
            continue
        try:
            if path.suffix == '.json':
                fields = [('.'.join(key), value, False)
                          for key, value in _json_numbers(json.loads(path.read_text(errors='replace')))]
            elif path.suffix in ('.csv', '.tsv'):
                fields = [(column, value, True) for column, value in _csv_extremes(path)]
            else:
                continue
        except (OSError, ValueError, UnicodeError, csv.Error):
            continue
        for field, value, column in fields:
            rule = rule_for(field.rsplit('.', 1)[-1])
            if rule is None or (name, field) in seen or not math.isfinite(value) or abs(value) <= BOUND_MARGIN * rule[1]:
                continue
            seen.add((name, field))
            shown = f'|{field}| reaches {value!r}' if column else f'{field} = {value!r}'
            flags.append(f'{name}: {shown} but the task says: "{rule[0]}"')
    return flags


_DEFINITION = re.compile(r'^\s*(?:[-*]\s+)?`?([A-Za-z][A-Za-z0-9_]*(?:\([^)]*\))?)`?\s*=\s*(.+?)\s*$')
_PIECEWISE = re.compile(r'\b(?:if|otherwise|else|when|interior)\b', re.I)
_PROSE = re.compile(r'`|\s(?:for|where|with|using|e\.g\.|i\.e\.)\s|\.\s+[A-Z]')
_MODEL_TERM = re.compile(r'\b(?:error|epsilon|noise)\b', re.I)


def _formula(rhs):
    """The formula part of a definition's right-hand side, or None when it is prose, piecewise or a model
    equation with an error term rather than a definition."""
    text = rhs.strip().lstrip('`')
    cut = _PROSE.search(text)
    text = (text[:cut.start()] if cut else text).strip().rstrip('.,;:`').strip()
    if _PIECEWISE.search(rhs) or _MODEL_TERM.search(text) or len(text) < 3:
        return None
    if not re.search(r'[*/+\-^]|\w\(', text):
        return None
    return text


def _canonical(formula):
    text = re.sub(r'\s+', '', formula).replace('[', '(').replace(']', ')')
    text = re.sub(r'_\{[^}]*\}', '', text)  # subscripts are notation: r_t and r, K_{i+1} and K
    return re.sub(r'(?<=[A-Za-z0-9])_[A-Za-z0-9](?![A-Za-z0-9])', '', text)


def conflicting_definitions(instruction):
    """Symbols the instruction defines by two or more different formulas: [(symbol, [(line, formula), ...])].
    Bracket style, spacing, subscripts and a trailing explanation do not make a definition different."""
    found = {}
    for number, line in enumerate(instruction.splitlines(), 1):
        match = _DEFINITION.match(line)
        if not match:
            continue
        formula = _formula(match.group(2))
        if formula is not None:
            found.setdefault(match.group(1), []).append((number, formula))
    conflicts = []
    for name, definitions in found.items():
        if len({_canonical(formula) for _, formula in definitions}) > 1:
            conflicts.append((name, definitions))
    return conflicts


_LIKELIHOOD = re.compile(r'likelihood|loglik|\bllf\b|\bnll\b', re.I)
LIKELIHOOD_LIMIT = 1e7  # a fitted model's log-likelihood this large in magnitude is not a real fit
# Likelihood-ratio and chi-square test statistics are >= 0 by construction; rounding of a zero statistic stays
# above LR_FLOOR. A key names one by a word (kupiec_lr, lr_ind, LRuc, chi2, likelihood_ratio_stat), or is a bare
# stat/statistic under a key naming such a test; t and z statistics, which may be negative, never match.
# 'lr' alone also abbreviates a learning rate, a linear regression, a long-run value or a log return, so a key
# whose only such word is 'lr' counts only beside a test: a stat word, a test's name or kind in the key or its
# parent (lr_stat, kupiec_lr, lr_ind, {"kupiec": {"lr": ...}}), never lr_slope, lr_skew or a bare lr beside a p.
LR_FLOOR = -1e-6
_LR_WORDS = {'lr', 'lrt', 'lrstat', 'lruc', 'lrind', 'lrcc', 'lrpof', 'chi2', 'chisq', 'chisquare', 'chisquared',
             'likelihoodratio'}
_LR_TESTS = (_LR_WORDS - {'lr', 'lrstat'}) | {'kupiec', 'christoffersen'}  # a key naming the test, not the value
_LR_KINDS = {'test', 'uc', 'ind', 'cc', 'pof', 'unconditional', 'independence', 'conditional',
             'coverage'}  # lr_test, lr_uc, ...: with 'lr', a key naming the test
_LR_PAIRS = {('likelihood', 'ratio'), ('chi', 'sq'), ('chi', 'square'), ('chi', 'squared'), ('chi', '2')}
_BARE_STAT = {'stat', 'statistic', 'teststat', 'test'}
_LR_CONTEXT = _LR_TESTS | _LR_KINDS | _BARE_STAT  # beside a lone 'lr', words that make it a test's statistic
_SIGNED_STAT = {'t', 'z', 'tstat', 'zstat', 'tvalue', 'zvalue', 'tscore', 'zscore'}
# An estimate or a difference (lr_mean, chi2_diff), or a statistic signed by design (signed root, Vuong), may be
# negative: a key or test key with one of these words never counts.
_NOT_LR = {'mean', 'avg', 'average', 'median', 'slope', 'beta', 'coef', 'coeff', 'coefficient', 'intercept', 'ret',
           'return', 'returns', 'diff', 'difference', 'delta', 'change', 'signed', 'vuong'}
_P_WORDS = {'p', 'pval', 'pvalue', 'pvalues'}


def _words(key):
    return re.findall(r'[a-z0-9]+', re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', str(key)).lower())


def _is_number(value):
    """A finite JSON number; an int of any size (float() of a huge one would overflow)."""
    return (isinstance(value, int) and not isinstance(value, bool)) or (isinstance(value, float)
                                                                         and math.isfinite(value))


def _names_lr(words, tests=_LR_WORDS):
    return bool(set(words) & tests or set(zip(words, words[1:])) & _LR_PAIRS)


def _names_test(words, kind=False):
    """Whether a key (its words) names a likelihood-ratio or chi-square test: kupiec, chi2_test, lr_ind; kind: a
    test kind alone (uc, independence) will do, the key under it saying 'lr'."""
    if set(words) & _NOT_LR:
        return False
    return _names_lr(words, _LR_TESTS) or bool(set(words) & _LR_KINDS) and (kind or 'lr' in words)


def _statistic(words, parent):
    """Whether a key (its words) names a likelihood-ratio or chi-square statistic; parent: the enclosing key's."""
    if set(words) & (_P_WORDS | _SIGNED_STAT | _NOT_LR):
        return False
    if _names_lr(words, _LR_WORDS - {'lr'}):
        return True
    if 'lr' in words:  # alone, 'lr' needs a test beside it
        return bool(set(words) & _LR_CONTEXT) or _names_test(parent, kind=True)
    return bool(words) and set(words) <= _BARE_STAT and _names_test(parent)


def _negative_statistics(data):
    """(key path, value) of each likelihood-ratio or chi-square statistic below LR_FLOOR with its p-value beside
    it in the same object: a key with a p word whose value is a number in [0, 1] and whose other words all
    belong to the statistic's key (kupiec_lr and kupiec_pvalue, lr_ind and p_ind, stat and p). Objects nested in
    objects and lists are searched without recursion."""
    found, stack = [], [((), data)]
    while stack:
        path, obj = stack.pop()
        items = list(obj.items()) if isinstance(obj, dict) else list(enumerate(obj)) if isinstance(obj, list) else []
        p_values = []  # the words, other than the p word, of each p-value key in this object
        for key, value in items if isinstance(obj, dict) else ():
            words = set(_words(key))
            if words & _P_WORDS and _is_number(value) and 0 <= value <= 1:
                p_values.append(words - _P_WORDS - {'value', 'val'})
        if p_values:
            parent = _words(next((p for p in reversed(path) if not p.isdigit()), ''))
            for key, value in items:
                if not (_is_number(value) and value < LR_FLOOR):
                    continue
                words = _words(key)
                if _statistic(words, parent) and any(rest <= set(words) for rest in p_values):
                    found.append((path + (str(key),), value))
        stack.extend(reversed([(path + (str(key),), value) for key, value in items
                               if isinstance(value, (dict, list))]))
    return found


def sanity_flags(out):
    """Fitted log-likelihoods (JSON keys naming a likelihood) too extreme to be real: the density or its
    parameters were computed wrong, whatever the optimizer reported. Likelihood-ratio or chi-square test
    statistics below 0 (a key naming one, with its p-value beside it): the likelihoods or terms it compares
    were computed wrong."""
    flags = []
    for path in sorted(p for p in out.rglob('*.json') if p.is_file() and not p.is_symlink()):
        if path.name in _NOT_OURS:
            continue
        try:
            data = json.loads(path.read_text(errors='replace'))
            numbers = list(_json_numbers(data))
        except (OSError, ValueError, UnicodeError, OverflowError, RecursionError):  # a huge int or deep nesting
            continue
        name = path.relative_to(out).as_posix()
        for key, value in numbers:
            if _LIKELIHOOD.search(key[-1].replace('_', ' ')) and (not math.isfinite(value)
                                                                    or abs(value) > LIKELIHOOD_LIMIT):
                flags.append(f'{name}: {".".join(key)} = {value!r} is too extreme for '
                             'a fitted model; the density or its parameters are almost certainly computed wrong')
        for key, value in _negative_statistics(data):
            flags.append(f'{name}: {".".join(key)} = {value!r} is below 0, but a likelihood-ratio or chi-square '
                         'test statistic is never negative; the log-likelihoods or terms it compares are computed '
                         'wrong')
    return flags


_PAIR = re.compile(r'`([^`\n]{1,20})`\s*(?:→|->|=>)\s*`?[-0-9./]+')
_LABEL = re.compile(r'^\s*\d+(?:\.\d+)?\s*([A-Za-z]+)\s*$')


def incomplete_mappings(instruction):
    """Explicit label mappings (a line with three or more `label`→value pairs) that leave out labels of the
    same kind (number + the same unit word) that the instruction names elsewhere:
    [(line, mapped labels, [(missing label, first line naming it), ...])]. A step that reuses the short
    mapping for more labels fails on the ones it lacks."""
    lines = instruction.splitlines()
    found = []
    for number, line in enumerate(lines, 1):
        mapped = [label.strip() for label in _PAIR.findall(line)]
        if len(mapped) < 3:
            continue
        units = {m.group(1) for m in map(_LABEL.match, mapped) if m}
        if not units:
            continue
        bare = re.compile(r'(?<![\w.])(\d+(?:\.\d+)?\s*(?:%s))\b' % '|'.join(sorted(map(re.escape, units))))
        missing = {}
        for other, text in enumerate(lines, 1):
            if other == number:
                continue
            for token in re.findall(r'`([^`\n]{1,20})`', text) + bare.findall(text):
                token = ' '.join(token.split())
                match = _LABEL.match(token)
                if match and match.group(1) in units and token not in mapped:
                    missing.setdefault(token, other)
        if missing:
            found.append((number, mapped, sorted(missing.items(), key=lambda item: (item[1], item[0]))))
    return found


_EXAMPLE = re.compile(r'`?([\w.%-]{1,24})`?\s{1,3}(?:becomes|is written as|is encoded as)\s{1,3}`?([\w.%-]{1,24}?)`?'
                      r'(?=[\s,;.)]|$)')
_EXAMPLE_WORDS = ('becomes', 'is written as', 'is encoded as')


def worked_examples(instruction):
    """Lines giving two or more worked examples of a naming or formatting rule ("0.99 becomes 99, 0.995 becomes
    995"): [(line, [(from, to), ...])]. A program that implements such a rule by arithmetic often reproduces
    one example and breaks another."""
    found = []
    for number, line in enumerate(instruction.splitlines(), 1):
        if not any(word in line for word in _EXAMPLE_WORDS):
            continue
        pairs = _EXAMPLE.findall(line)
        if len(pairs) >= 2:
            found.append((number, pairs))
    return found


# Literal output formats the instruction states (fixed or rounded decimals, true/false, numbers rather than
# strings), and degenerate outputs. The instruction's own words decide every rewrite: where it is silent,
# nothing is touched. Rewrites keep each value (a number is rounded or re-typed, never recomputed); what
# cannot be rewritten safely, and outputs that look degenerate, only become advisory flags.
_NUMBER_WORDS = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8,
                 'nine': 9, 'ten': 10, 'eleven': 11, 'twelve': 12}
_N = r'(\d{1,2}|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)'
_FIXED = re.compile(r'\b(?:fixed(?:[- ]point)?(?:\s+with)?(?:\s+exactly)?|exactly)\s+' + _N + r'\s+decimal', re.I)
_ROUNDED = re.compile(r'\bround(?:ed|ing)?\s+(?:to\s+)?' + _N + r'\s+decimal', re.I)
_DECIMALS = re.compile(r'\b' + _N + r'\s+decimal', re.I)
_NOT_EXACT = re.compile(r'\b(?:at least|up to|at most|no more than|no fewer than|or more|or fewer|or less|minimum|'
                        r'maximum|min|max)\b', re.I)
_JSON_WORD = re.compile(r'\bjson\b', re.I)
_EXCEPT = re.compile(r'\b(?:except|unless|other than|apart from|aside from|besides)\b', re.I)
# A clause that negates its rule ("do not round", "are not rounded") states no rule; a line that keeps some values
# unrounded ("other values keep full precision") gives no rule for a whole file.
_NEGATION = re.compile(r"\b(?:not|never|no|without|rather than|instead of|avoid)\b|n't\b", re.I)
_KEEP_PRECISION = re.compile(r'\b(?:full|maximum|native|original|machine|double) precision\b|\bunrounded\b'
                             r'|\bnot (?:be )?rounded\b|\bwithout rounding\b|\bno rounding\b'
                             r"|\b(?:do not|don't|never|not) round\b", re.I)
_UNIVERSAL = re.compile(r'\b(?:all|every|each|any)\b', re.I)
_CLAUSE = re.compile(r';|(?<=[.!?])\s+(?=[A-Z`*(\-])')
# Words that leave a clause about every value of its target ("All numeric columns use exactly 6 decimals"). Any
# other word narrows it ("monetary", "adjusted prices", "other values"), and then it sets no whole-file rule.
_GENERIC_WORDS = frozenset('''
    a an the all every each any of in for on at to from as by with and or its their this that these those
    is are be been being must should shall will has have use uses used using written write writes stored
    reported represented serialized expressed given shown displayed kept keep
    json csv file files output outputs result results value values number numbers numeric numerical float floats
    floating point floating-point double doubles real reals column columns field fields cell cells entry entries
    scalar scalars data
    round rounded rounding fixed fixed-point exactly decimal decimals place places digit digits precision
    note notes important rule rules format formats formatting formatted
    one two three four five six seven eight nine ten eleven twelve'''.split())
_PATHLIKE = re.compile(r'(?:/?[\w.\-]+/)*[\w.\-]+\.(?:csv|json|tsv|txt)\b', re.I)
_CSV_NAME = re.compile(r'\b([A-Za-z0-9_\-]+\.csv)\b')
_JSON_NAME = re.compile(r'\b([A-Za-z0-9_\-]+\.json)\b')
_HEADER_ONLY = re.compile(r'\bheader(?: row)? only\b|\bonly (?:the |a )?header\b|\bno data rows\b', re.I)
_NUMERIC_RULE = re.compile(r'\b(?:numeric|numbers?)\b[^.\n]{0,40}?\b(?:scalars?|not (?:as |be )?(?:quoted |a )?'
                           r'strings?)\b', re.I)
_COLUMN_WORDS = re.compile(r'\b(?:column|columns|field|fields|csv|value|values|numeric|float|floats|cell|cells)\b',
                           re.I)
_NAMED_RULE = re.compile(r'^\s*(?:[-*]\s+)?((?:`?[A-Za-z_][A-Za-z0-9_]*`?\s*,\s*)*`?[A-Za-z_][A-Za-z0-9_]*`?)'
                         r'\s*:\s*(.+)$')
_TABLE_RULE = re.compile(r'^\s*\|\s*`?([A-Za-z_][A-Za-z0-9_]*)`?\s*\|(.+)$')
_SLOT_TOKEN = re.compile(r'"(?:[^"\\\n]|\\.)*"|<[^<>\n]*>|[{}\[\]:,]|//[^\n]*|#[^\n]*|-?\d[\w.+-]*'
                         r'|\b(?:true|false|null|True|False|None)\b|[A-Za-z_][A-Za-z0-9_]*')
_NUMERIC_TEXT = re.compile(r'[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?')
_DECIMAL_TEXT = re.compile(r'-?\d+\.\d+')
_FLOAT_CELL = re.compile(r'[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+\.?\d*[eE][-+]?\d+|[-+]?\d+\.')
_NAN_CELL = {'nan', 'inf', '-inf', '+inf', 'infinity', '-infinity'}
NORMALIZE_LIMIT = 32 * 1024 * 1024  # bytes: a larger output file is left as written
DEGENERATE_MIN = 3  # values in a column or container before 'all zero' or 'all NaN' means anything
FLAG_LIMIT = 12


def _count_of(token):
    return int(token) if token.isdigit() else _NUMBER_WORDS[token.lower()]


def _fenced_lines(lines):
    inside, fenced = False, set()
    for i, line in enumerate(lines):
        if line.lstrip().startswith('```'):
            inside = not inside
            fenced.add(i)
        elif inside:
            fenced.add(i)
    return fenced


def _exact(text, match):
    """Whether a decimals count is exact: no 'at least', 'up to', 'minimum' and the like right around it."""
    return not _NOT_EXACT.search(text[max(0, match.start() - 16):match.end() + 12])


def _section_csv(lines, i):
    """CSV file names on line i; else those its section heading (the nearest earlier markdown heading) names.
    A nearer line that merely mentions a CSV does not make the section that file's."""
    names = _CSV_NAME.findall(lines[i])
    if names:
        return [Path(n).name for n in names]
    heading = next((j for j in range(i - 1, -1, -1) if lines[j].lstrip().startswith('#')), -1)
    if heading >= 0 and _CSV_NAME.findall(lines[heading]):
        return [Path(n).name for n in _CSV_NAME.findall(lines[heading])]
    return []


def _clauses(line):
    return [part.strip() for part in _CLAUSE.split(line) if part.strip()]


def _clause_rule(clause):
    """(count, mode, keys, whole) for a clause that states one exact number of decimals, else None. mode is
    'fixed', 'round' or None (a bare 'N decimals'); keys are the backticked names it gives the rule to; whole
    says that it speaks of every value of its target: no word narrows it, and a named key comes with 'all' or
    'every'. A clause with a negation, an exception or two different counts states no rule."""
    if _NEGATION.search(clause) or _EXCEPT.search(clause):
        return None
    found = list(_DECIMALS.finditer(clause))
    counts = {_count_of(m.group(1)) for m in found}
    if len(counts) != 1 or not all(_exact(clause, m) for m in found):
        return None
    mode = 'fixed' if _FIXED.search(clause) else 'round' if _ROUNDED.search(clause) else None
    keys = [k for k in re.findall(r'`([A-Za-z_][A-Za-z0-9_]*)`', clause)
            if k.lower() not in ('true', 'false', 'null', 'none', 'nan')]
    bare = _PATHLIKE.sub(' ', re.sub(r'`[^`]*`', ' ', clause))
    narrowing = [w for w in re.findall(r"[A-Za-z][A-Za-z'-]*", bare) if w.lower() not in _GENERIC_WORDS]
    whole = not narrowing and (not keys or bool(_UNIVERSAL.search(bare)))
    return counts.pop(), mode, keys, whole


def _placeholder(text):
    """(kind, decimals) of a template placeholder such as <float, 4 decimals>, <int>, <bool> or <string>."""
    low = text.lower()
    found = _DECIMALS.search(low)
    digits = _count_of(found.group(1)) if found and _exact(low, found) else None
    if re.search(r'\b(?:object|dict|array|list|map)\b', low):
        return 'other', None
    if re.search(r'\bbool', low):
        return 'bool', None
    if re.search(r'\bint(?:eger)?s?\b', low):
        return 'int', None
    if re.search(r'\b(?:float|number|numeric|double|decimal|real)s?\b', low):
        return 'number', digits
    if re.search(r'\b(?:str|string|text|date|datetime|timestamp|iso)', low):
        return 'string', None
    return 'other', digits


def template_slots(block):
    """{key path: (kind, decimals, example)} for the values of a JSON-like template: kind is number, int,
    bool, string or other; decimals come from a placeholder such as <float, 4 decimals>; example is a
    literal number. The first token of a value decides: a bare type word (`[float x 12]` is a number slot), and a
    literal number only when it is the whole value. Paths use template_keys' segments ('*' for a placeholder
    key, '[]' for list items)."""
    tokens = [t for t in _SLOT_TOKEN.findall(block) if not t.startswith(('//', '#'))]
    slots, stack, expect = {}, [], None

    def prefix(frames):
        return tuple(frame[1] if frame[0] == '{' else '[]' for frame in frames)

    for i, token in enumerate(tokens):
        if token in ('{', '['):
            stack.append([token, None])
            expect = None
            continue
        if token in ('}', ']'):
            if stack:
                stack.pop()
            expect = None
            continue
        if token in (':', ','):
            if token == ',':
                expect = None
            continue
        if token.startswith('"') and i + 1 < len(tokens) and tokens[i + 1] == ':' and stack and stack[-1][0] == '{':
            try:
                stack[-1][1] = _segment(json.loads(token))
            except ValueError:
                stack[-1][1] = None
            expect = prefix(stack) if stack[-1][1] is not None and None not in prefix(stack) else None
            continue
        if stack and stack[-1][0] == '[':
            path = prefix(stack)
        elif expect is not None:
            path = expect
        else:
            continue
        if None in path:
            continue
        example = None
        if token.startswith('<'):
            kind, digits = _placeholder(token[1:-1])
        elif token.startswith('"'):
            kind, digits = 'string', None
        elif token in ('true', 'false', 'True', 'False'):
            kind, digits = 'bool', None
        elif token in ('null', 'None'):
            kind, digits = 'other', None
        elif re.fullmatch(r'[A-Za-z_]\w*', token):
            kind, digits = _placeholder(token)
        else:
            try:
                example = float(token)
            except ValueError:
                continue
            whole = i + 1 == len(tokens) or tokens[i + 1] in (',', '}', ']')
            kind = ('int' if re.fullmatch(r'-?\d+', token) else 'number') if whole else 'other'
            digits = None
        slots.setdefault(path, (kind, digits, example))
        expect = None
    return slots


def literal_rules(instruction):
    """The literal output formats the instruction states, and what it allows, as data for normalize_outputs,
    literal_flags and degenerate_flags. Only explicit statements count, one clause at a time (_clause_rule): a
    'fixed N decimals' clause about every value of a CSV (named in it, or its section's heading), a 'rounded to N
    decimals' clause about every JSON value or every value of a named file, the same about a backticked key or
    column (that key's rule only), a key annotated with N decimals (in a template, a bullet or a table row), a
    rule that JSON numbers are numeric scalars, backticked `true`/`false` (never capitalised, or the reverse),
    and 'header only' sentences. A clause that negates its rule or names an exception gives none, and a line that
    keeps some values unrounded gives no whole-file rule."""
    lines = instruction.splitlines()
    fenced = _fenced_lines(lines)
    rules = {'csv_fixed': {}, 'csv_round': {}, 'csv_columns': {}, 'json_round': {}, 'names': {}, 'scalars': set(),
             'header_only': set(), 'templates': {}, 'zero_keys': set(), 'booleans': None,
             'nan_allowed': bool(_NAN_ALLOWED.search(instruction)), 'quotes': []}

    def cite(line):
        text = ' '.join(line.split()).lstrip('-*| ').strip()
        if text and text not in rules['quotes']:
            rules['quotes'].append(text)

    for target, block, match in _json_templates(instruction):
        slots = rules['templates'].setdefault(target, {})
        for path, slot in template_slots(block).items():
            slots.setdefault(path, slot)
            if slot[2] is not None and abs(slot[2]) <= 1e-3 and path and path[-1] not in ('*', '[]'):
                rules['zero_keys'].add(path[-1])
        # A rounding sentence right after a template, naming no other target, is that template's rule.
        after = [line for line in instruction[match.end():].splitlines()[:4] if line.strip()][:1]
        for line in after:
            if _EXCEPT.search(line) or _KEEP_PRECISION.search(line) or _CSV_NAME.search(line):
                continue
            for clause in _clauses(line):
                rule = _clause_rule(clause)
                if rule and rule[1] == 'round' and rule[3]:
                    for name in _JSON_NAME.findall(line) or [Path(target).name]:
                        rules['json_round'].setdefault(Path(name).name, rule[0])
                    cite(line)
    for line in lines:  # a key the instruction (or its template) shows next to values of about 0 may be 0
        for key in re.findall(r'["`]([A-Za-z_][A-Za-z0-9_]*)["`]', line):
            rest = line.split(key, 1)[1]
            numbers = [float(x) for x in re.findall(r'(?<![\w.])-?\d+\.\d+', rest)]
            if numbers and all(abs(x) <= 1e-3 for x in numbers):
                rules['zero_keys'].add(key)
    for i, line in enumerate(lines):
        if i in fenced:
            continue
        whole_ok = not _EXCEPT.search(line) and not _KEEP_PRECISION.search(line)
        for clause in _clauses(line):
            rule = _clause_rule(clause)
            if rule is None or rule[1] is None:
                continue
            count, mode, keys, whole = rule
            csv_named = [Path(n).name for n in _CSV_NAME.findall(clause)]
            json_named = [Path(n).name for n in _JSON_NAME.findall(clause)]
            json_side = bool(_JSON_WORD.search(clause) or json_named) and not csv_named
            if whole and whole_ok:
                fixed = csv_named or (_section_csv(lines, i) if _COLUMN_WORDS.search(clause) else [])
                if mode == 'fixed' and not json_side and fixed:
                    for name in fixed:
                        rules['csv_fixed'].setdefault(name, count)
                    cite(line)
                elif mode == 'round' and (json_side or json_named or csv_named):
                    for name in (json_named or ['*']) if json_side else json_named:
                        rules['json_round'].setdefault(name, count)
                    for name in csv_named:
                        rules['csv_round'].setdefault(name, count)
                    cite(line)
            elif keys and not whole:
                tables = csv_named or ([] if json_side else _section_csv(lines, i))
                for key in keys:
                    for name in tables:
                        rules['csv_columns'].setdefault(name, {}).setdefault(key, (mode, count))
                    if not tables:
                        rules['names'].setdefault(key, count)
                cite(line)
        if _JSON_WORD.search(line) and _NUMERIC_RULE.search(line):
            rules['scalars'].update([Path(n).name for n in _JSON_NAME.findall(line)] or ['*'])
        if _HEADER_ONLY.search(line):
            rules['header_only'].update(_section_csv(lines, i))
        named = _NAMED_RULE.match(line) or _TABLE_RULE.match(line)
        if named and not _NEGATION.search(named.group(2)) and not _EXCEPT.search(named.group(2)):
            found = list(_DECIMALS.finditer(named.group(2)))
            if len({_count_of(m.group(1)) for m in found}) == 1 and all(_exact(named.group(2), m) for m in found):
                for key in re.findall(r'[A-Za-z_][A-Za-z0-9_]*', named.group(1)):
                    rules['names'].setdefault(key, _count_of(found[0].group(1)))
                cite(line)
    ticks = set(re.findall(r'`(true|false|True|False)`', instruction))
    if ticks & {'true', 'false'} and not ticks & {'True', 'False'}:
        rules['booleans'] = 'lower'
    elif ticks & {'True', 'False'} and not ticks & {'true', 'false'}:
        rules['booleans'] = 'upper'
    return rules


def _template_for(rules, name):
    templates = rules['templates']
    if name in templates:
        return templates[name]
    return next((slots for target, slots in templates.items() if Path(target).name == Path(name).name), {})


class _Slots:
    """Matches output JSON paths (keys and list indices) against one file's template paths."""

    def __init__(self, slots):
        self.slots = slots
        self.prefixes = {path[:n] for path in slots for n in range(len(path) + 1)}

    def step(self, active, key):
        found = []
        for node in active:
            for segment in ((key, '*') if key != '[]' else ('[]',)):
                if node + (segment,) in self.prefixes:
                    found.append(node + (segment,))
        return found

    def slot(self, active):
        best = sorted((path.count('*'), path) for path in active if path in self.slots)
        return self.slots[best[0][1]] if best else None


def _key_decimals(rules, path):
    for key in reversed(path):
        if isinstance(key, str) and key in rules['names']:
            return rules['names'][key]
    return None


def _visit_json(value, rules, name, visit_leaf):
    """Rebuild a JSON value with visit_leaf(leaf, (kind, decimals, example) or None, path, decimals) applied
    to every leaf; decimals are the template's, else a key rule's, else the file's rule."""
    slots = _Slots(_template_for(rules, name))
    base = Path(name).name
    file_digits = rules['json_round'].get(base, rules['json_round'].get('*'))

    def visit(node, path, active):
        if isinstance(node, dict):
            return {key: visit(child, path + (key,), slots.step(active, str(key))) for key, child in node.items()}
        if isinstance(node, list):
            return [visit(child, path + (i,), slots.step(active, '[]')) for i, child in enumerate(node)]
        slot = slots.slot(active)
        digits = slot[1] if slot and slot[1] is not None else _key_decimals(rules, path)
        return visit_leaf(node, slot, path, file_digits if digits is None else digits)
    return visit(value, (), [()])


def _read_text(path):
    """(text, had BOM) of a regular output file that is safe to rewrite, else None."""
    try:
        if path.is_symlink() or not path.is_file():
            return None
        info = path.stat()
        if info.st_nlink != 1 or info.st_size > NORMALIZE_LIMIT:
            return None
        data = path.read_bytes()
        bom = data.startswith(b'\xef\xbb\xbf')
        return data[3 if bom else 0:].decode('utf-8'), bom
    except (OSError, UnicodeError):
        return None


def _write_text(path, text, bom):
    """Replace the file (never through a link) with the rewritten text, keeping its permission bits so that
    whoever could read the program's file can read this one."""
    import stat
    import tempfile
    mode = stat.S_IMODE(os.lstat(path).st_mode)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix='.t1-normalize-')
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write((b'\xef\xbb\xbf' if bom else b'') + text.encode('utf-8'))
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _json_indent(text):
    lines = text.strip().splitlines()
    if len(lines) < 2:
        return None
    second = lines[1]
    if second.startswith('\t'):
        return '\t'
    return (len(second) - len(second.lstrip(' '))) or 2


def _normalize_json_file(path, name, rules, counts):
    read = _read_text(path)
    if read is None:
        return False
    text, bom = read
    try:
        data = json.loads(text)
    except (ValueError, RecursionError):
        return False
    scalars = '*' in rules['scalars'] or Path(name).name in rules['scalars']
    changed = Counter()

    def leaf(value, slot, path, digits):
        kind = slot[0] if slot else None
        elided = slot is not None and len(slot) > 3  # a slot only contract_literal_rules adds (T1_CONTRACT)
        if isinstance(value, str):
            text_value = value.strip()
            if kind == 'bool' and text_value.lower() in ('true', 'false'):
                changed['normalized_json_booleans'] += 1
                if elided:
                    changed['contract_slot_values'] += 1
                return text_value.lower() == 'true'
            if (kind in ('number', 'int') and _NUMERIC_TEXT.fullmatch(text_value)) or (
                    kind not in ('string', 'bool', 'int') and scalars and _DECIMAL_TEXT.fullmatch(text_value)):
                number = float(text_value)
                if not math.isfinite(number):
                    return value
                changed['normalized_json_numbers'] += 1
                if elided and kind in ('number', 'int'):
                    changed['contract_slot_values'] += 1
                if kind == 'int' and number.is_integer():
                    return int(number)
                value = number
            else:
                return value
        if isinstance(value, bool) or not isinstance(value, float) or not math.isfinite(value):
            return value
        if kind == 'int' and value.is_integer() and abs(value) < 2 ** 53:
            changed['normalized_json_integers'] += 1
            if elided:
                changed['contract_slot_values'] += 1
            return int(value)
        if digits is not None and kind not in ('int', 'string', 'bool') and round(value, digits) != value:
            changed['normalized_json_rounded'] += 1
            return round(value, digits)
        return value

    try:
        new = _visit_json(data, rules, name, leaf)
    except RecursionError:
        return False
    if not changed:
        return False
    ascii_only = all(ord(ch) < 128 for ch in text)
    _write_text(path, json.dumps(new, indent=_json_indent(text), ensure_ascii=ascii_only)
                + ('\n' if text.endswith('\n') else ''), bom)
    counts.update(changed)
    return True


def _csv_rows(text):
    return list(csv.reader(io.StringIO(text, newline='')))


def _normalize_csv_file(path, name, rules, counts):
    base = Path(name).name
    fixed, rounded, booleans = rules['csv_fixed'].get(base), rules['csv_round'].get(base), rules['booleans']
    columns = rules['csv_columns'].get(base, {})
    contract = rules.get('contract')  # only contract_literal_rules (T1_CONTRACT) sets it
    guard = rules.get('int_guard')  # only integer_guard_rules (thin's outputs_fix) sets it
    if fixed is None and rounded is None and booleans is None and not columns and contract is None:
        return False
    read = _read_text(path)
    if read is None:
        return False
    text, bom = read
    try:
        rows = _csv_rows(text)
    except csv.Error:
        return False
    if len(rows) < 2:
        return False
    header, body, changed = rows[0], rows[1:], Counter()
    wrong, right = ('True', 'False'), ('true', 'false')
    if booleans == 'upper':
        wrong, right = right, wrong
    for j in range(len(header)):
        cells = [row[j].strip() for row in body if j < len(row)]
        filled = [c for c in cells if c]
        if booleans and filled and all(c in wrong + right for c in filled) and any(c in wrong for c in filled):
            for row in body:
                if j < len(row) and row[j].strip() in wrong:
                    row[j] = right[wrong.index(row[j].strip())]
                    changed['normalized_csv_booleans'] += 1
            continue
        own = columns.get(header[j].strip())  # a rule the instruction gives this column alone
        if contract is not None and own is None and _integer_column(contract, base, header[j].strip(), filled):
            for row in body:  # whole numbers in a column the TASK states as integer: never N decimals
                if j < len(row) and row[j].strip() and _integer_text(row[j].strip()) != row[j].strip():
                    row[j] = _integer_text(row[j].strip())
                    changed['contract_int_cells'] += 1
            continue
        col_fixed, col_rounded = ((own[1], None) if own[0] == 'fixed' else (None, own[1])) if own else (fixed, rounded)
        if (guard is not None and own is None and col_fixed is not None
                and _guarded_integer(guard['contract'], base, header[j].strip(), filled)):
            for row in body:  # whole numbers in an id, count or timestamp column: plain integers, never N decimals
                if j < len(row) and row[j].strip() and _integer_text(row[j].strip()) != row[j].strip():
                    row[j] = _integer_text(row[j].strip())
                    changed['normalized_csv_integers'] += 1
                    guard['cells'].setdefault(name, set()).add(j)
            continue
        if (col_fixed is None and col_rounded is None) or not any(_FLOAT_CELL.fullmatch(c) for c in filled):
            continue
        for row in body:
            if j >= len(row) or not _NUMERIC_TEXT.fullmatch(row[j].strip()):
                continue
            number = float(row[j])
            if not math.isfinite(number):
                continue
            if col_fixed is not None:
                new = f'{number:.{col_fixed}f}'
            elif round(number, col_rounded) != number:
                new = repr(round(number, col_rounded))
            else:
                continue
            if new != row[j]:
                row[j] = new
                changed['normalized_csv_decimals'] += 1
    if not changed:
        return False
    stream = io.StringIO()
    csv.writer(stream, lineterminator='\r\n' if '\r\n' in text[:65536] else '\n').writerows(rows)
    _write_text(path, stream.getvalue(), bom)
    counts.update(changed)
    return True


def _outputs(out, suffixes):
    try:
        paths = sorted(p for p in out.rglob('*') if p.suffix.lower() in suffixes)
    except OSError:
        return []
    return [(p, p.relative_to(out).as_posix()) for p in paths
            if p.name not in _NOT_OURS and not p.is_symlink() and p.is_file()]


def normalize_outputs(rules, out):
    """Rewrite output values only where the instruction states the literal format (see literal_rules):
    fixed or rounded decimals, booleans, and numbers written as strings or ints written as floats at the
    template's numeric slots. Returns counters (normalized_files and one per kind of rewrite)."""
    counts = Counter()
    for suffix, rewrite in (('.json', _normalize_json_file), ('.csv', _normalize_csv_file)):
        for path, name in _outputs(out, (suffix,)):
            try:
                if rewrite(path, name, rules, counts):
                    counts['normalized_files'] += 1
            except (OSError, ValueError, TypeError, OverflowError, csv.Error):
                continue
    return dict(counts)


def _dotted(path):
    return ''.join(f'[{p}]' if isinstance(p, int) else ('.' if i else '') + str(p) for i, p in enumerate(path))


def _load_csv(path):
    with path.open(newline='', errors='replace') as stream:
        reader = csv.reader(stream)
        header = next(reader, None)
        body = [row for _, row in zip(range(CSV_ROWS), reader)]
    return header, body


def _literal_json_flags(rules, path, name, limit=3):
    """literal_flags' lines for one JSON output: up to limit slot shapes (None: all)."""
    try:
        data = json.loads(path.read_text(errors='replace'))
    except (OSError, ValueError, RecursionError):
        return []
    found = {}

    def leaf(value, slot, where, digits):
        kind = slot[0] if slot else None
        wanted = {'number': 'a number', 'int': 'an integer', 'bool': 'true or false'}.get(kind)
        if wanted is None:
            return value
        ok = (isinstance(value, bool) if kind == 'bool' else
              isinstance(value, (int, float)) and not isinstance(value, bool) or value is None)
        if not ok:
            shape = tuple('[]' if isinstance(p, int) else p for p in where)
            found.setdefault(shape, (where, type(value).__name__, wanted))
        return value
    try:
        _visit_json(data, rules, name, leaf)
    except RecursionError:
        return []
    return [f'{name}: {_dotted(where)} is a {got} but the TASK shows {wanted} there'
            for where, got, wanted in list(found.values())[:limit]]


def _literal_csv_flags(rules, path, name):
    """literal_flags' lines for one CSV output; with the contract's rules (T1_CONTRACT) also whole numbers written
    with decimals in a column the TASK states as integer (which no decimals rule then covers)."""
    base = Path(name).name
    whole, own = rules['csv_fixed'].get(base), rules['csv_columns'].get(base, {})
    contract = rules.get('contract')
    if whole is None and not any(rule[0] == 'fixed' for rule in own.values()) and contract is None:
        return []
    try:
        header, body = _load_csv(path)
    except (OSError, csv.Error):
        return []
    flags = []
    for j, column in enumerate(header or []):
        rule = own.get(column.strip())
        cells = [row[j].strip() for row in body if j < len(row) and row[j].strip()]
        if contract is not None and rule is None and _integer_column(contract, base, column.strip(), cells):
            bad = [c for c in cells if _integer_text(c) != c]
            if bad:
                flags.append(f'{name}: column {column} has {len(bad)} whole numbers not written as integers, '
                             f'e.g. {bad[0][:40]}')
            continue
        digits = (rule[1] if rule[0] == 'fixed' else None) if rule else whole
        if digits is None:
            continue
        pattern = re.compile(r'-?\d+\.\d{%d}' % digits)
        if not any(_FLOAT_CELL.fullmatch(c) for c in cells):
            continue
        bad = [c for c in cells if not pattern.fullmatch(c)]
        if bad:
            flags.append(f'{name}: column {column} has {len(bad)} cells not written with exactly {digits} '
                         f'decimals, e.g. {bad[0][:40]}')
    return flags


def literal_flags(rules, out):
    """Literal format problems left after normalize_outputs: a template's numeric or boolean slot holding
    another type, and cells of a fixed-decimals CSV that are still not in that format."""
    flags = []
    for path, name in _outputs(out, ('.json',)):
        flags += _literal_json_flags(rules, path, name)
    for path, name in _outputs(out, ('.csv',)):
        flags += _literal_csv_flags(rules, path, name)
    return flags[:FLAG_LIMIT]


def _zero_allowed(rules, key):
    key = str(key)
    return any(w in key.lower() for w in _TOLERANCE_WORDS) or key in rules['zero_keys']


def _leaves(obj, path=()):
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield from _leaves(value, path + (key,))
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            yield from _leaves(value, path + (i,))
    else:
        yield path, obj


def _containers(obj, path=()):
    if isinstance(obj, dict):
        yield path, list(obj.values())
        for key, value in obj.items():
            yield from _containers(value, path + (key,))
    elif isinstance(obj, list):
        yield path, obj
        for i, value in enumerate(obj):
            yield from _containers(value, path + (i,))


def _degenerate_json_flags(rules, path, name):
    try:
        data = json.loads(path.read_text(errors='replace'))
        containers = list(_containers(data))
    except (OSError, ValueError, RecursionError):
        return []
    flags = []
    bad = [where for where, value in _leaves(data) if isinstance(value, float) and not _finite(value)]
    if bad and not rules['nan_allowed']:
        flags.append(f'{name}: {len(bad)} values are NaN or infinite, e.g. {_dotted(bad[0])}')
    for where, values in containers:
        numbers = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
        key = next((p for p in reversed(where) if isinstance(p, str)), '')
        if (len(numbers) >= DEGENERATE_MIN and len(numbers) == len(values) and all(v == 0 for v in numbers)
                and not _zero_allowed(rules, key)):
            flags.append(f'{name}: {_dotted(where) or "the top level"} holds {len(numbers)} values, all 0')
    return flags


def _degenerate_csv_flags(rules, path, name):
    try:
        header, body = _load_csv(path)
    except (OSError, csv.Error):
        return []
    if header is None:
        return []
    if not body:
        return [] if Path(name).name in rules['header_only'] else [f'{name}: header only, no data rows']
    flags = []
    for j, column in enumerate(header):
        cells = [row[j].strip() for row in body if j < len(row) and row[j].strip()]
        if len(cells) < DEGENERATE_MIN:
            continue
        if not rules['nan_allowed'] and all(c.lower() in _NAN_CELL for c in cells):
            flags.append(f'{name}: column {column} is NaN in every row')
        elif (any(_FLOAT_CELL.fullmatch(c) for c in cells) and all(_NUMERIC_TEXT.fullmatch(c) for c in cells)
              and all(float(c) == 0 for c in cells) and not _zero_allowed(rules, column)):
            flags.append(f'{name}: column {column} is 0 in every row')
    return flags


def degenerate_flags(rules, out):
    """Outputs that look degenerate (advisory: a task may want them): a CSV with a header and no rows unless
    a 'header only' sentence covers it, a CSV column that is all NaN, a CSV float column or a JSON container
    of DEGENERATE_MIN or more numbers that are all zero (not for tolerance-like names or keys the
    instruction shows as about zero), and non-finite JSON numbers unless the instruction speaks of NaN."""
    flags = []
    for path, name in _outputs(out, ('.json',)):
        flags += _degenerate_json_flags(rules, path, name)
    for path, name in _outputs(out, ('.csv',)):
        flags += _degenerate_csv_flags(rules, path, name)
    return flags[:FLAG_LIMIT]


# The output contract the TASK states (T1_CONTRACT): the columns of a CSV and their order, the keys of a JSON
# template and their order, integer cells, and template slots a "..." elides. contract_rules() reads only what the
# TASK states without doubt; normalize_contract() runs normalize_outputs() alone and then the contract's fixes plus
# normalize_outputs(), file by file, and keeps a file's contract version only when the TASK's own statements find
# fewer problems in it and none it did not have before (_file_problems).
_LAYOUT_TOKEN = re.compile(r'"(?:[^"\\\n]|\\.)*"|<[^<>\n]*>|\.\.\.|…|[{}\[\]:,]|//[^\n]*|#[^\n]*|-?\d[\w.+-]*'
                           r'|\b(?:true|false|null|True|False|None)\b|[A-Za-z_][A-Za-z0-9_]*')
_ELLIPSIS = ('...', '…')
_HEADING = re.compile(r'^\s{0,3}(#{1,6})\s+(.*)$')
_INPUT_HEADING = re.compile(r'\binputs?\b', re.I)
_INPUT_CSV = re.compile(r'(?:/app/data|/input|\bdata)/(?:[\w.\-]+/)*([\w.\-]+\.csv)\b')
_COLUMN_LEAD = re.compile(r'\b(?:columns?|headers?)\b', re.I)
# The column word ends the lead-in right before the list ("with columns `a,b`", "Columns (in this order): `a`, `b`").
_COLUMN_LEAD_END = re.compile(r'\b(?:columns?|headers?)\b\s*(?:\([^()\n]*\))?\s*:?\s*$', re.I)
# A lead-in about sorting, grouping or keys names some columns, never all of them.
_NOT_A_HEADER = re.compile(r'\b(?:sort\w*|ordered by|order by|rank\w*|group\w*|merge\w*|join\w*|index\w*|keyed|'
                           r'unique\w*|dedup\w*|identif\w*|primary|drop\w*|exclud\w*|optional\w*)\b', re.I)
_COLUMN_NAME = re.compile(r'[A-Za-z_][\w\-+%]*')
_ONLY_SPANS = re.compile(r'(?:`[^`\n]+`\s*(?:,|\band\b)?\s*)+[.;:]?')
_SPAN = re.compile(r'`([^`\n]+)`')
# What a lead-in says of its list: that the order is the file's ("in this order", "column order", "exact") and that
# the list is all the file has ("exactly these columns", "must be exact", "only these"). Neither is read into a bare
# "columns:", and an order or "only" about rows ("in date order", "write only the rows ...") is neither.
_HEADER_ORDER = re.compile(r'\bin (?:this|the following|the listed|the given|the same|the exact|exact)\s+order\b'
                           r'|\b(?:column|header)s?\s+order\b|\border\s+(?:shown|listed|given)\b|\bexact(?:ly)?\b', re.I)
_HEADER_EXACT = re.compile(r'\bexact(?:ly)?\b|\bonly (?:these|the following|the listed)\b'
                           r'|\b(?:these|listed) columns only\b|\bno (?:other|extra|additional) columns?\b', re.I)
_TABLE_SEPARATOR = re.compile(r'^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)*\|?\s*$')
_TABLE_NAME_CELLS = {'column', 'columns', 'column name', 'field', 'field name'}
_INT_TYPE = re.compile(r'^(?:int|integer|int32|int64|integers)\b', re.I)
_INT_BULLET = re.compile(r'^\s*(?:[-*]\s+)?`([A-Za-z_]\w*)`\s*(?::|—|–|-|\(|\bis\b)\s*(?:an?\s+)?'
                         r'(?:int|integer|int32|int64)s?\b', re.I)
_INT_LITERAL = re.compile(r'`([A-Za-z_]\w*)\s*=\s*-?\d+`')
_TIMESTAMP_WORD = re.compile(r'\b(?:timestamps?|epoch)\b', re.I)
_INTEGER_WORD = re.compile(r'\bintegers?\b', re.I)
_TIMESTAMP_COLUMN = re.compile(r'(?:^|_)(?:ms|timestamp|epoch)$', re.I)
_ID_COLUMN = re.compile(r'(?:^|_)id$', re.I)
_KEY_ORDER = re.compile(r'\b(?:keys?|fields?)\b[^.\n]{0,60}?\border\b|\border\b[^.\n]{0,20}?\bkeys?\b', re.I)
_ORDER_SOURCE = re.compile(r'\b(?:listed|shown|exact|exactly|this|following|given|same|template|above|below)\b',
                           re.I)
_ROW_ORDER = re.compile(r'\b(?:ascending|descending|sorted|sort|any order|chronological\w*|alphabetical\w*)\b', re.I)
CONTRACT_SNAPSHOT_LIMIT = 64 * 1024 * 1024  # bytes of JSON and CSV outputs held in memory to compare two versions
# T1_CONTRACT2 (contract2_on): header fixes kept on a CSV whose header an indented block states, and those that drop
# an extra column the TASK backticks only because it is a column of another output's stated header.
CONTRACT2_COUNTERS = ('contract2_indented_headers', 'contract2_cross_file_drops')
CONTRACT_COUNTERS = ('contract_int_cells', 'contract_slot_values', 'contract_key_reorders', 'contract_key_renames',
                     'contract_header_fixes', 'contract_reverted_files') + CONTRACT2_COUNTERS
_INDENTED = re.compile(r'^(?: {4}|\t)[ \t]*\S')  # a line of an indented (markdown code) block


def template_layout(block):
    """({dict path: [key segments in the template's order]}, {dict paths whose entries a '...' elides}, {dict
    paths shown with two different key lists}) for a JSON-like template; paths are template_keys'."""
    tokens = [t for t in _LAYOUT_TOKEN.findall(block) if not t.startswith(('//', '#'))]
    shown, elided, stack = {}, set(), []  # frames: [kind, current key, dict path or None, keys in order]

    def where():
        return tuple(frame[1] if frame[0] == '{' else '[]' for frame in stack)

    for i, token in enumerate(tokens):
        if token in ('{', '['):
            path = where()
            stack.append([token, None, None if None in path else path, []])
        elif token in ('}', ']'):
            if stack:
                kind, _, path, keys = stack.pop()
                if kind == '{' and path is not None:
                    shown.setdefault(path, []).append(keys)
        elif token in _ELLIPSIS:
            if stack and stack[-1][0] == '{' and stack[-1][2] is not None and (i == 0 or tokens[i - 1] != ':'):
                elided.add(stack[-1][2])
        elif token.startswith('"') and i + 1 < len(tokens) and tokens[i + 1] == ':' and stack and stack[-1][0] == '{':
            try:
                key = _segment(json.loads(token))
            except ValueError:
                key = None
            stack[-1][1] = key
            if key is not None and key not in stack[-1][3]:
                stack[-1][3].append(key)
    layout = {path: lists[0] for path, lists in shown.items()}
    mixed = {path for path, lists in shown.items() if any(keys != lists[0] for keys in lists[1:])}
    return layout, elided, mixed


def _sections(lines, fenced):
    """For each line, the indices of the markdown headings above it, outermost first."""
    stack, found = [], []
    for i, line in enumerate(lines):
        match = None if i in fenced else _HEADING.match(line)
        if match:
            level = len(match.group(1))
            stack = [(lvl, j) for lvl, j in stack if lvl < level] + [(level, i)]
        found.append([j for _, j in stack])
    return found


def _key_order_stated(text):
    """Whether a passage (fences removed) says that keys keep the order the template shows."""
    for sentence in re.split(r'(?<=[.!?])\s+|\n\s*\n|\n\s*[-*]\s+', text):
        sentence = ' '.join(sentence.split())
        if (_KEY_ORDER.search(sentence) and _ORDER_SOURCE.search(sentence) and not _ROW_ORDER.search(sentence)
                and not _NEGATION.search(sentence) and not _CSV_NAME.search(sentence)):
            return sentence
    return None


def _unfenced(text):
    return re.sub(r'```.*?(?:```|$)', ' ', text, flags=re.S)


def _json_contract(instruction):
    """{template target: {'layout', 'elided', 'mixed', 'paths', 'ordered'}} for every JSON template; 'ordered' when
    the template's section (its heading to the next one, fences left out) says keys keep the template's order and
    names no other JSON or CSV file in that sentence."""
    found = {}
    headings = [m.start() for m in re.finditer(r'(?m)^\s{0,3}#{1,6}\s', instruction)]
    for target, block, match in _json_templates(instruction):
        entry = found.setdefault(target, {'layout': {}, 'elided': set(), 'mixed': set(), 'paths': set(),
                                          'ordered': False})
        layout, elided, mixed = template_layout(block)
        for path, keys in layout.items():
            if path in entry['layout'] and entry['layout'][path] != keys:
                entry['mixed'].add(path)
            entry['layout'].setdefault(path, keys)
        entry['elided'] |= elided
        entry['mixed'] |= mixed
        entry['paths'] |= template_keys(block)
        start = max([h for h in headings if h < match.start()], default=max(0, match.start() - TEMPLATE_WINDOW))
        end = min([h for h in headings if h > match.end()], default=len(instruction))
        sentence = _key_order_stated(_unfenced(instruction[start:match.start()] + '\n\n' + instruction[match.end():end]))
        if sentence and {Path(n).name for n in _JSON_NAME.findall(sentence)} <= {Path(target).name}:
            entry['ordered'] = True
    return found


def _expand(names):
    """Column names with each '...' between two runs of a counter (lag_-5, lag_-4, ..., lag_4, lag_5) filled in;
    None when an ellipsis cannot be filled in that way."""
    result = []
    for i, name in enumerate(names):
        if name not in _ELLIPSIS:
            result.append(name)
            continue
        before, after = names[max(0, i - 2):i], names[i + 1:i + 3]
        parts = [re.fullmatch(r'(.*?)(-?\d+)(\D*)', n) for n in before + after]
        if len(before) < 2 or len(after) < 2 or not all(parts) or any(n in _ELLIPSIS for n in before + after):
            return None
        if len({(p.group(1), p.group(3)) for p in parts}) != 1:
            return None
        numbers = [int(p.group(2)) for p in parts]
        step = numbers[1] - numbers[0]
        if step == 0 or numbers[3] - numbers[2] != step or (numbers[2] - numbers[1]) % step or \
                (numbers[2] - numbers[1]) // step < 2:
            return None
        head, tail = parts[0].group(1), parts[0].group(3)
        result += [f'{head}{n}{tail}' for n in range(numbers[1] + step, numbers[2], step)]
    return result


def _names(spans):
    """Column names from backticked spans ('a,b,c' or one name each); None unless every piece is a name or '...'."""
    names = []
    for span in spans:
        for piece in span.split(','):
            piece = piece.strip()
            if not (piece in _ELLIPSIS or _COLUMN_NAME.fullmatch(piece)):
                return None
            names.append(piece)
    return names


def _csv_target(lines, fenced, sections, i, inputs):
    """The output CSV a statement on line i is about: the CSV named on the line, else on the lines just above it
    in the same paragraph, else in the nearest heading. None when that names no CSV or two, when the CSV is an
    input (named under a data path) or the statement sits in an input section."""
    if any(_INPUT_HEADING.search(lines[h]) for h in sections[i]):
        return None
    names = _CSV_NAME.findall(lines[i])
    j = i - 1
    while not names and j >= 0 and i - j <= 3 and lines[j].strip() and j not in fenced:
        names = _CSV_NAME.findall(lines[j])
        if _HEADING.match(lines[j]):
            break
        j -= 1
    if not names and sections[i]:
        names = _CSV_NAME.findall(lines[sections[i][-1]])
    names = {Path(n).name for n in names}
    if len(names) != 1:
        return None
    name = names.pop()
    return None if name in inputs else name


def _lead(lines, fenced, i):
    """The nearest non-blank line above line i (within two lines) outside a fence, or ''."""
    for j in range(i - 1, max(-1, i - 3), -1):
        if lines[j].strip():
            return '' if j in fenced else lines[j]
    return ''


def _header_statements(lines, fenced, sections, inputs):
    """(target CSV, column names or None, type cells or None, lead-in text) for each statement of a CSV's
    columns: a 'columns'/'header' lead-in followed by backticked names only, a fence holding only a comma-separated
    list of names after such a lead-in or right under the CSV's heading, and a markdown table whose first column
    is the column name (its lead-in: the paragraph line above it)."""
    found = []
    for i, line in enumerate(lines):
        if i in fenced or not line.strip():
            continue
        if line.lstrip().startswith('|') and i + 1 < len(lines) and _TABLE_SEPARATOR.match(lines[i + 1]):
            cells = [c.strip().strip('`*').strip().lower() for c in line.strip().strip('|').split('|')]
            if cells and cells[0] in _TABLE_NAME_CELLS:
                kind = next((k for k, c in enumerate(cells) if c in ('type', 'dtype', 'data type')), None)
                names, types, k = [], [], i + 2
                while k < len(lines) and lines[k].lstrip().startswith('|'):
                    row = [c.strip() for c in lines[k].strip().strip('|').split('|')]
                    names.append(row[0].strip('`* '))
                    types.append(row[kind] if kind is not None and kind < len(row) else '')
                    k += 1
                target = _csv_target(lines, fenced, sections, i, inputs)
                valid = len(names) >= 2 and all(_COLUMN_NAME.fullmatch(n) for n in names)
                if target:
                    found.append((target, names if valid else None, types if valid and kind is not None else None,
                                  _lead(lines, fenced, i)))
            continue
        spans = _SPAN.findall(line)
        if not spans:
            continue
        first = line.index('`')
        before, rest = line[:first], line[first:].strip()
        own_lead = (bool(_COLUMN_LEAD_END.search(before)) and not _NEGATION.search(before)
                    and not _NOT_A_HEADER.search(before))
        lead = _lead(lines, fenced, i)
        lead_above = (not before.strip(' -*') and lead.rstrip().endswith(':') and bool(_COLUMN_LEAD.search(lead))
                      and not _NEGATION.search(lead) and not _NOT_A_HEADER.search(lead))
        if not (own_lead or lead_above) or not _ONLY_SPANS.fullmatch(rest):
            continue
        k = i + 1  # a list that wraps onto lines holding only backticked names
        while k < len(lines) and k not in fenced and _ONLY_SPANS.fullmatch(lines[k].strip() or '-'):
            spans += _SPAN.findall(lines[k])
            k += 1
        names = _names(spans)
        target = _csv_target(lines, fenced, sections, i, inputs)
        if target and names is not None and len(names) >= 2:
            found.append((target, _expand(names), None, before if own_lead else lead))
    i = 0
    while i < len(lines):  # fences
        if not (i in fenced and lines[i].lstrip().startswith('```')):
            i += 1
            continue
        info = lines[i].strip()[3:].strip().lower()
        end = i + 1
        while end < len(lines) and not lines[end].lstrip().startswith('```'):
            end += 1
        body = ' '.join(lines[i + 1:end]).strip().rstrip(',')
        lead = _lead(lines, fenced, i)
        under_heading = bool(lead and _HEADING.match(lead) and _CSV_NAME.search(lead))
        about_columns = (lead.rstrip().endswith(':') and bool(_COLUMN_LEAD.search(lead)) and not _NEGATION.search(lead)
                         and not _NOT_A_HEADER.search(lead))
        if info in ('', 'text', 'txt', 'csv', 'plain', 'plaintext') and body and (under_heading or about_columns):
            names = [n.strip() for n in body.split(',')]
            if len(names) >= 2 and all(n in _ELLIPSIS or _COLUMN_NAME.fullmatch(n) for n in names):
                target = _csv_target(lines, fenced, sections, i, inputs)
                if target:
                    found.append((target, _expand(names), None, '' if under_heading else lead))
        i = end + 1
    return found


def contract2_on() -> bool:
    """T1_CONTRACT2=1 (with T1_CONTRACT; off in the image): contract_rules also reads a CSV header stated in an
    indented block (_indented_statements), and an exact header drops an extra column the TASK backticks on its own
    when it is a column of another output CSV's stated header (_fix_csv_structure). Off, rules and outputs are as
    before."""
    return os.environ.get('T1_CONTRACT2') == '1'


def _indented_statements(lines, fenced, sections, inputs):
    """T1_CONTRACT2: _header_statements' statements from indented blocks, read like an untyped fence: a run of lines
    each starting with 4 spaces or a tab, after a blank line and outside any fence, holding only a comma-separated
    list of two or more names, under the same lead-in as a fence (a 'columns' line ending in ':' or the CSV's
    heading). Its CSV is the one the lead-in line is about (_csv_target from that line): fama-french states
    `rolling_betas.csv` there and its header below a blank line, under a heading that names no CSV."""
    found, i = [], 1
    while i < len(lines):
        if i in fenced or not _INDENTED.match(lines[i]) or lines[i - 1].strip():
            i += 1
            continue
        end = i
        while end < len(lines) and end not in fenced and _INDENTED.match(lines[end]):
            end += 1
        body = ' '.join(lines[i:end]).strip().rstrip(',')
        j = next((k for k in range(i - 1, max(-1, i - 3), -1) if lines[k].strip()), None)
        lead = _lead(lines, fenced, i)
        under_heading = bool(lead and _HEADING.match(lead) and _CSV_NAME.search(lead))
        about_columns = (lead.rstrip().endswith(':') and bool(_COLUMN_LEAD.search(lead)) and not _NEGATION.search(lead)
                         and not _NOT_A_HEADER.search(lead))
        names = [n.strip() for n in body.split(',')]
        if (j is not None and lead and (under_heading or about_columns) and len(names) >= 2
                and all(n in _ELLIPSIS or _COLUMN_NAME.fullmatch(n) for n in names)):
            target = _csv_target(lines, fenced, sections, j, inputs)
            if target:
                found.append((target, _expand(names), None, '' if under_heading else lead))
        i = end
    return found


def contract_rules(instruction, contract2=None):
    """The output contract the TASK states, as data for normalize_contract: 'headers' {CSV: columns in order}
    from statements that agree (a CSV with two different statements, or one with a '...' that is not a counter
    run, has none); 'header_order' and 'header_exact', the CSVs whose lead-in says the order is the file's or the
    list is all it has; 'int_columns' {CSV: columns a type cell, a bullet ('`n`: integer'), or a backticked
    `name=<integer>` beside a listed column states as integer}; 'timestamps' (CSV names, or '*') where a line says
    timestamps or epoch values are integers; 'ticked' (every name the TASK backticks on its own); 'json'
    (_json_contract). Under T1_CONTRACT2 (contract2 None reads contract2_on()) the statements include indented
    blocks (_indented_statements), and the rules carry 'contract2' True and 'indented' (the CSVs whose header an
    indented block states)."""
    contract2 = contract2_on() if contract2 is None else contract2
    lines = instruction.splitlines()
    fenced = _fenced_lines(lines)
    sections = _sections(lines, fenced)
    in_sections = {Path(n).name for i, line in enumerate(lines) if i not in fenced
                   and any(_INPUT_HEADING.search(lines[h]) for h in sections[i]) for n in _CSV_NAME.findall(line)}
    inputs = ({Path(n).name for n in _INPUT_CSV.findall(instruction)}
              | (in_sections - {Path(n).name for n in _files(instruction)}))  # a CSV under a data path is an input
    headers, conflicts, ints, ordered, exact = {}, set(), {}, set(), set()
    statements = _header_statements(lines, fenced, sections, inputs)
    indented = _indented_statements(lines, fenced, sections, inputs) if contract2 else []
    statements += indented
    for target, names, types, lead in statements:
        if names is None or len(set(names)) != len(names):
            conflicts.add(target)
        elif headers.setdefault(target, names) != names:
            conflicts.add(target)
        if _HEADER_ORDER.search(lead):
            ordered.add(target)
        if _HEADER_EXACT.search(lead):
            exact.add(target)
    for target in conflicts:
        headers.pop(target, None)
    for target, names, types, _ in statements:
        if names and types:
            ints.setdefault(target, set()).update(n for n, t in zip(names, types) if _INT_TYPE.match(t.strip('`* ')))
    timestamps = set()
    for i, line in enumerate(lines):
        if i in fenced:
            continue
        bullet = _INT_BULLET.match(line)
        literals = _INT_LITERAL.findall(line)
        if (bullet or literals) and sections[i]:
            named = {Path(n).name for n in _CSV_NAME.findall(lines[sections[i][-1]])}  # the innermost heading's CSV
            target = named.pop() if len(named) == 1 else None
            if target and target not in inputs and not any(_INPUT_HEADING.search(lines[h]) for h in sections[i]):
                stated = headers.get(target)
                for name in ([bullet.group(1)] if bullet else []) + literals:
                    if stated is None and literals and name in literals:
                        continue  # a `name=0` counts only beside a stated header
                    if stated is None or name in stated:
                        ints.setdefault(target, set()).add(name)
        if (_TIMESTAMP_WORD.search(line) and _INTEGER_WORD.search(line) and not _NEGATION.search(line)
                and not any(_INPUT_HEADING.search(lines[h]) for h in sections[i])):
            lead = _lead(lines, fenced, i)
            named = {Path(n).name for n in _CSV_NAME.findall(line + ' ' + (lead if lead.rstrip().endswith(':')
                                                                             else ''))}
            timestamps |= (named - inputs) if named else {'*'}
    ticked = set(re.findall(r'`([A-Za-z_][\w\-+%]*)`', instruction))
    rules = {'headers': headers, 'header_order': ordered & set(headers), 'header_exact': exact & set(headers),
             'int_columns': ints, 'timestamps': timestamps, 'ticked': ticked, 'json': _json_contract(instruction)}
    if contract2:
        rules.update(contract2=True, indented={target for target, _, _, _ in indented} & set(headers))
    return rules


def contract_literal_rules(rules, contract):
    """literal_rules' rules with the contract's additions: a '*' slot for the keys a template elides with '...'
    when every key it shows beside them is a leaf of one kind (number, int or bool) and decimals, and the
    contract itself (integer columns). The added slots carry a fourth element so their rewrites are counted."""
    extended = dict(rules)
    templates = {target: dict(slots) for target, slots in rules['templates'].items()}
    for target, entry in contract['json'].items():
        slots = templates.setdefault(target, {})
        for path in entry['elided']:
            keys = entry['layout'].get(path, [])
            shown = [k for k in keys if k not in ('*', '[]')]
            if path in entry['mixed'] or '*' in keys or len(shown) < 2:
                continue
            if any(len(p) > len(path) + 1 and p[:len(path) + 1] == path + (k,) for p in entry['paths'] for k in shown):
                continue  # a shown key holds an object or a list
            kinds = {slots.get(path + (k,), (None, None))[:2] for k in shown}
            if len(kinds) == 1 and next(iter(kinds))[0] in ('number', 'int', 'bool'):
                kind, digits = kinds.pop()
                slots.setdefault(path + ('*',), (kind, digits, None, 'elided'))
    extended['templates'] = templates
    extended['contract'] = contract
    return extended


def _integer_text(cell):
    """A whole number written with a decimal point or an exponent ('12.0', '1.2e3') as an integer; any other
    cell as it is."""
    if not ('.' in cell or 'e' in cell or 'E' in cell) or not _NUMERIC_TEXT.fullmatch(cell):
        return cell
    import decimal
    try:
        value = decimal.Decimal(cell)
    except decimal.InvalidOperation:
        return cell
    if not value.is_finite() or value != value.to_integral_value():
        return cell
    return str(int(value))


def _whole(cell):
    return bool(_NUMERIC_TEXT.fullmatch(cell)) and (_integer_text(cell) != cell or re.fullmatch(r'[-+]?\d+', cell)
                                                   is not None)


def _integer_column(contract, base, column, filled):
    """Whether a CSV column holds integers by the TASK's statement and every filled cell is a whole number: a
    column it states as integer, a timestamp or epoch column (…_ms) where it says timestamps are integers, or an
    identifier column (id, …_id) it names."""
    if not filled or not all(_whole(c) for c in filled):
        return False
    if column in contract['int_columns'].get(base, ()):
        return True
    if _TIMESTAMP_COLUMN.search(column) and ('*' in contract['timestamps'] or base in contract['timestamps']):
        return True
    return bool(_ID_COLUMN.search(column)) and (column in contract['headers'].get(base, ())
                                                 or column in contract['ticked'])


# outputs_fix (thin): a CSV column whose header names an id, count, index or timestamp (id, *_id, *_ms, timestamp,
# epoch, *_index, *_count, n_*); with every filled cell a whole number, fg_normalize writes it as plain integers.
_INTEGER_NAME = re.compile(r'(?:^|_)(?:id|ms|timestamp|epoch|index|count)$|^n_\w', re.I)


def integer_guard_rules(rules, instruction, cells):
    """literal_rules' rules for normalize_outputs under thin's outputs_fix: a stated fixed-decimals rewrite of a CSV
    never gives N decimals to a column whose filled cells are all whole numbers and that holds integers by its name
    (_INTEGER_NAME) or by the TASK's statement (_integer_column, the contract_int_cells rule): those cells are written
    as plain integers ('12.0' as '12'). `cells` collects {output name: indexes of the columns so rewritten}."""
    try:
        contract = contract_rules(instruction)
    except Exception:
        contract = None
    return dict(rules, int_guard={'contract': contract, 'cells': cells})


def _guarded_integer(contract, base, column, filled):
    """Whether integer_guard_rules keeps this CSV column whole: every filled cell a whole number, and an integer name
    or a TASK statement (_integer_column) for it."""
    if not filled or not all(_whole(c) for c in filled):
        return False
    if _INTEGER_NAME.search(column):
        return True
    return contract is not None and _integer_column(contract, base, column, filled)


def _header_problems(name, stated, header, ordered=True):
    problems = [f'{name}: lacks column {c}' for c in stated if c not in header]
    problems += [f'{name}: has column {c} the TASK does not list' for c in header if c not in stated]
    if ordered and [c for c in header if c in stated] != [c for c in stated if c in header]:
        problems.append(f"{name}: columns are not in the TASK's order")
    return problems


def _header_plan(stated, header, ticked, ordered, exact):
    """(new header, source column of each new column) that makes a CSV header the stated one: one column renamed
    where it is the only one that differs, at the same place; when the TASK says the list is exact, the columns it
    does not list dropped (never one it names on its own in backticks, which it may ask for elsewhere); when it
    says the order is the file's, the columns put in that order. None otherwise."""
    if header == stated or len(set(header)) != len(header) or any(c != c.strip() for c in header):
        return None
    if len(header) == len(stated):
        differ = [j for j in range(len(header)) if header[j] != stated[j]]
        if len(differ) == 1 and header[differ[0]] not in stated and stated[differ[0]] not in header:
            return list(stated), list(range(len(header)))
    if not set(stated) <= set(header):
        return None
    extra = [c for c in header if c not in stated]
    if extra and (not exact or any(c in ticked for c in extra)):
        return None
    kept = list(stated) if ordered else [c for c in header if c in stated]
    return (kept, [header.index(c) for c in kept]) if kept != header else None


def _problem_keys(problems):
    return Counter(re.sub(r'[-+]?\d[\d.eE+-]*', '#', problem) for problem in problems)


def _better(before, after):
    """Whether the problems after a fix are fewer and none is new (numbers in a problem do not count)."""
    before, after = _problem_keys(before), _problem_keys(after)
    return all(after[key] <= before[key] for key in after) and sum(after.values()) < sum(before.values())


def _fix_csv_structure(contract, path, name, counts):
    """Give a CSV the header the TASK states (_header_plan) when that leaves fewer header problems. Under
    T1_CONTRACT2 a name the TASK backticks protects an extra column only when it is no column of another output CSV's
    stated header (parity-audit's violations.csv carries parity_audit.csv's columns, two of them backticked alone in
    rule bullets about that file)."""
    stated = contract['headers'].get(Path(name).name)
    read = _read_text(path) if stated else None
    if read is None:
        return False
    text, bom = read
    try:
        rows = _csv_rows(text)
    except csv.Error:
        return False
    if not rows or any(len(row) != len(rows[0]) for row in rows[1:]):
        return False
    base = Path(name).name
    ordered = base in contract['header_order']
    ticked = contract['ticked']
    if contract.get('contract2'):
        ticked = ticked - set().union(*(columns for other, columns in contract['headers'].items() if other != base))
    plan = _header_plan(stated, rows[0], ticked, ordered, base in contract['header_exact'])
    if plan is None or not _better(_header_problems(name, stated, rows[0], ordered),
                                   _header_problems(name, stated, plan[0], ordered)):
        return False
    header, source = plan
    written, rows = rows[0], [header] + [[row[j] for j in source] for row in rows[1:]]
    stream = io.StringIO()
    csv.writer(stream, lineterminator='\r\n' if '\r\n' in text[:65536] else '\n').writerows(rows)
    _write_text(path, stream.getvalue(), bom)
    counts['contract_header_fixes'] += 1
    if contract.get('contract2') and base in contract['indented']:
        counts['contract2_indented_headers'] += 1
    if (contract.get('contract2') and len(header) < len(written)
            and any(c in contract['ticked'] and c not in header for c in written)):  # a drop, not a rename
        counts['contract2_cross_file_drops'] += 1
    return True


def _dicts_at(data, path):
    """(location, dict) for each object of the output at a template dict path ('*' any key, '[]' every item)."""
    nodes = [((), data)]
    for segment in path:
        found = []
        for where, node in nodes:
            if segment == '[]':
                if isinstance(node, list):
                    found += [(where + (i,), value) for i, value in enumerate(node)]
            elif isinstance(node, dict):
                if segment == '*':
                    found += [(where + (key,), value) for key, value in node.items()]
                elif segment in node:
                    found.append((where + (segment,), node[segment]))
        nodes = found
    return [(where, node) for where, node in nodes if isinstance(node, dict)]


def _literal_levels(entry):
    """(path, keys) of the template objects whose keys are all literal and shown one way, outermost first."""
    return sorted(((path, keys) for path, keys in entry['layout'].items()
                   if keys and '*' not in keys and '[]' not in keys and path not in entry['mixed']),
                  key=lambda item: len(item[0]))


def _json_problems(entry, name, data):
    problems = []
    for path, keys in _literal_levels(entry):
        for where, node in _dicts_at(data, path):
            place = _dotted(where) or 'the top level'
            missing = [k for k in keys if k not in node]
            problems += [f'{name}: {place} lacks key {k}' for k in missing]
            problems += [f'{name}: {place} has key {k} the TASK does not show' for k in node if k not in keys]
            if entry['ordered'] and not missing and [k for k in node if k in keys] != keys:
                problems.append(f"{name}: {place} keys are not in the TASK's order")
    return problems


def _container(entry, path):
    return any(len(p) > len(path) and p[:len(path)] == path for p in entry['paths'])


def _renames(entry, path, keys, node):
    """{output key: template key} when the template keys an object lacks are exactly its extra keys with one
    constant prefix or suffix added (two or more of them), each holding the kind of value the template shows."""
    missing = [k for k in keys if k not in node]
    extra = [k for k in node if k not in keys]
    if len(missing) < 2 or len(missing) != len(extra):
        return None
    found = []
    for affix in ('prefix', 'suffix'):
        ends = (lambda m, e: m.endswith(e)) if affix == 'prefix' else (lambda m, e: m.startswith(e))
        candidates = {m[:len(m) - len(extra[0])] if affix == 'prefix' else m[len(extra[0]):]
                      for m in missing if len(m) > len(extra[0]) and ends(m, extra[0])}
        for constant in sorted(candidates):
            mapping = {e: constant + e if affix == 'prefix' else e + constant for e in extra}
            if sorted(mapping.values()) == sorted(missing):
                found.append(mapping)
    if len(found) != 1:
        return None
    mapping = found[0]
    for old, new in mapping.items():
        if isinstance(node[old], (dict, list)) != _container(entry, path + (new,)):
            return None
    return mapping


def _restructured(entry, data, kind, counts):
    """A copy of the output with template objects renamed (kind 'rename') or put in the template's key order
    ('order'), counting each object changed."""
    import copy
    data = copy.deepcopy(data)
    for path, keys in _literal_levels(entry):
        for _, node in _dicts_at(data, path):
            if kind == 'rename':
                mapping = _renames(entry, path, keys, node)
                if not mapping:
                    continue
                items = list(node.items())
                node.clear()
                node.update((mapping.get(key, key), value) for key, value in items)
                counts['contract_key_renames'] += 1
            elif all(k in node for k in keys) and [k for k in node if k in keys] != keys:
                items = dict(node)
                node.clear()
                node.update((k, items[k]) for k in keys)
                node.update((k, v) for k, v in items.items() if k not in keys)
                counts['contract_key_reorders'] += 1
    return data


def _fix_json_structure(contract, path, name, counts):
    """Rename and then reorder the keys of a JSON output's template objects (when the TASK states the order),
    each step kept only when it leaves fewer key problems."""
    entry = next((e for target, e in contract['json'].items() if Path(target).name == Path(name).name), None)
    read = _read_text(path) if entry else None
    if read is None:
        return False
    text, bom = read
    try:
        data = json.loads(text)
    except (ValueError, RecursionError):
        return False
    new, changed = data, Counter()
    for kind in ('rename', 'order') if entry['ordered'] else ('rename',):
        step = Counter()
        candidate = _restructured(entry, new, kind, step)
        if step and _better(_json_problems(entry, name, new), _json_problems(entry, name, candidate)):
            new = candidate
            changed.update(step)
    if not changed:
        return False
    ascii_only = all(ord(ch) < 128 for ch in text)
    _write_text(path, json.dumps(new, indent=_json_indent(text), ensure_ascii=ascii_only)
                + ('\n' if text.endswith('\n') else ''), bom)
    counts.update(changed)
    return True


def _file_problems(rules, contract, json_keys, out, path, name):
    """Every problem the TASK's own statements find in one output, uncapped: its header or template keys against
    the contract, literal formats (with integer columns and elided slots), degenerate values, and missing or
    non-finite template keys (contract_flags, json_keys being the file's template keys)."""
    problems = []
    try:
        if path.suffix.lower() == '.json':
            problems += _literal_json_flags(rules, path, name, None) + _degenerate_json_flags(rules, path, name)
            entry = next((e for target, e in contract['json'].items() if Path(target).name == Path(name).name), None)
            if entry:
                try:
                    problems += _json_problems(entry, name, json.loads(path.read_text(errors='replace')))
                except (ValueError, RecursionError):
                    problems.append(f'{name}: not valid JSON')
            if json_keys:
                problems += contract_flags({'files': [], 'json': {name: json_keys},
                                            'nan_allowed': rules['nan_allowed']}, out)
        else:
            problems += _literal_csv_flags(rules, path, name) + _degenerate_csv_flags(rules, path, name)
            stated = contract['headers'].get(Path(name).name)
            if stated:
                header, _ = _load_csv(path)
                problems += _header_problems(name, stated, header or [],
                                             Path(name).name in contract['header_order'])
    except (OSError, ValueError, UnicodeError, csv.Error, RecursionError, OverflowError):
        problems.append(f'{name}: unreadable')
    return problems


def _write_bytes(path, data):
    """_write_text's atomic replace for bytes already encoded."""
    import stat
    import tempfile
    mode = stat.S_IMODE(os.lstat(path).st_mode)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix='.t1-normalize-')
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data)
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _normalize_one(path, name, rules, contract=None):
    """normalize_outputs for one file, after the contract's fixes of its header or keys when contract is given."""
    counts = Counter()
    json_file = path.suffix.lower() == '.json'
    if contract is not None:
        try:
            (_fix_json_structure if json_file else _fix_csv_structure)(contract, path, name, counts)
        except (OSError, ValueError, TypeError, KeyError, IndexError, OverflowError, RecursionError, csv.Error):
            pass
    try:
        (_normalize_json_file if json_file else _normalize_csv_file)(path, name, rules, counts)
    except (OSError, ValueError, TypeError, OverflowError, csv.Error):
        pass
    if counts:
        counts['normalized_files'] = 1
    return counts


def normalize_contract(instruction, out):
    """normalize_outputs plus the output contract the TASK states (T1_CONTRACT), file by file: every JSON and CSV
    output is first normalized as normalize_outputs alone would (version A), then from the program's own bytes
    with the contract's fixes (headers, key names and order) and its rules (integer columns, elided slots) before
    normalize_outputs (version B). A file keeps B only when _file_problems finds fewer problems in B than in A and
    none that A lacks; otherwise it keeps A (contract_reverted_files). Outputs past CONTRACT_SNAPSHOT_LIMIT, and
    any failure, get normalize_outputs alone. Returns the counters of the versions kept."""
    rules = literal_rules(instruction)
    try:
        contract = contract_rules(instruction)
        extended = contract_literal_rules(rules, contract)
        templated = output_contract(instruction)
    except Exception:
        return normalize_outputs(rules, out)
    files, original, total = [], {}, 0
    for path, name in _outputs(out, ('.json', '.csv')):
        if _read_text(path) is None:
            continue  # neither version rewrites it
        try:
            original[name] = path.read_bytes()
        except OSError:
            continue
        total += len(original[name])
        if total > CONTRACT_SNAPSHOT_LIMIT:  # nothing is rewritten yet
            return normalize_outputs(rules, out)
        files.append((path, name))

    def put_back(versions):
        for path, name in files:
            try:
                if path.read_bytes() != versions[name]:
                    _write_bytes(path, versions[name])
            except OSError:
                pass

    def template_keys_of(name):
        return next((k for t, k in templated['json'].items() if Path(t).name == Path(name).name), None)

    counts_a, version_a, problems_a = {}, {}, {}
    try:
        for path, name in files:
            counts_a[name] = _normalize_one(path, name, rules)
            version_a[name] = path.read_bytes()
            problems_a[name] = _file_problems(extended, contract, template_keys_of(name), out, path, name)
    except Exception:
        put_back(original)
        return normalize_outputs(rules, out)
    result = Counter()
    try:
        put_back(original)
        for path, name in files:
            counts_b = _normalize_one(path, name, extended, contract)
            if path.read_bytes() == version_a[name]:
                result.update(counts_a[name])
            elif _better(problems_a[name], _file_problems(extended, contract, template_keys_of(name), out, path,
                                                          name)):
                result.update(counts_b)
            else:
                _write_bytes(path, version_a[name])
                result.update(counts_a[name])
                result['contract_reverted_files'] += 1
    except Exception:  # every file as normalize_outputs alone left it
        put_back(version_a)
        result = Counter()
        for counts in counts_a.values():
            result.update(counts)
    return {key: value for key, value in result.items() if value}


# T1_S11_LINT: output lints after a clean run, zero requests: a non-finite number in a field the TASK names
# (nonfinite_flags), and one quantity written in two output JSON files with two values (crosskey_flags). Each must
# stay silent on a correct program: a TASK that speaks of NaN or infinite values, or defines the field as a quotient
# (which may be infinite), a NaN inside a list (a window's first values), a field the TASK never names; and a key
# name two files share without evidence that it is one quantity there (two files of one schema for a strategy and
# its benchmark, two models, two samples, a closed form and a Monte Carlo) is left alone.
_INF_WORDS = re.compile(r'\b(?:inf|infinity|infinite|infinitely|non-?finite|unbounded)\b|∞', re.I)
_NAN_WORDS = re.compile(r'\b(?:NaN|nan|null|None|undefined|n/a)\b')
_INF_CELLS = frozenset({'inf', '-inf', '+inf', 'infinity', '-infinity', '+infinity'})
_QUOTIENT = re.compile(r'/|÷|\\frac\b|\b(?:divided|ratio|quotient)\b', re.I)  # a field the TASK defines as one
_ABS_PATH = re.compile(r'(?<![\w)\]])/[\w.\-]+(?:/[\w.\-]*)*')  # /app/output: a path, not a quotient
CROSS_REL, CROSS_ABS = 1e-9, 1.5e-6  # floats this close are one value written twice
_ROUNDING = re.compile(r'\bround(?:s|ed|ing)?\b|\bd\.?p\.?\b|\bsignificant (?:digits|figures)\b', re.I)
# A TASK line that says a key holds the same value in two files; not one that gives two files the same schema
# ("same structure as"), or two values a tolerance ("should match within 3 standard errors").
_SAME = re.compile(r'\b(?:same|identical|equal|equals|match|matches|consistent|cop(?:y|ies|ied)|repeat(?:s|ed)?|'
                   r'restate[sd]?|duplicate[sd]?)\b', re.I)
_SAME_SHAPE = re.compile(r'\b(?:same|identical|matching|consistent)\s+(?:\w+\s+)?(?:structure|format|layout|schema|'
                         r'shape|keys|fields|columns|way|method|form|order|names?)\b', re.I)
_APPROX = re.compile(r'\b(?:within|approximately|roughly|close to|near|tolerance|about)\b|≈|~', re.I)
# Keys that group values without saying which quantity they are; any other key on a path (a model, a sample, a
# level, a stage) does.
_CONTAINERS = frozenset({'intermediates', 'intermediate', 'checkpoints', 'checkpoint', 'results', 'result',
                         'outputs', 'output', 'values', 'solution', 'summary', 'metrics', 'data', 'final'})
LINT_FILE_LIMIT = 4 * 1024 * 1024  # bytes: a larger output file is not linted


def _task_names(instruction, key):
    """Whether the TASK names key as a word (any case)."""
    key = str(key).strip()
    return bool(key) and re.search(rf'(?<![A-Za-z0-9_]){re.escape(key)}(?![A-Za-z0-9_])', instruction, re.I) is not None


def _lint_leaves(path):
    """(key path, value) of every leaf of a JSON output, or None when it is too large or does not parse."""
    try:
        if path.stat().st_size > LINT_FILE_LIMIT:
            return None
        return list(_leaves(json.loads(path.read_text(errors='replace'))))
    except (OSError, ValueError, UnicodeError, RecursionError):
        return None


def _field(path):
    """The key a leaf is written under: the last object key on its path, a 'value' wrapper key read as its owner's."""
    keys = [p for p in path if isinstance(p, str)]
    if len(keys) > 1 and keys[-1] == 'value' and path[-1] == 'value':
        return keys[-2]
    return keys[-1] if keys else None


def _quotient(instruction, key):
    """Whether a TASK line that names key defines it as a quotient ('/', 'divided', 'ratio'), which is infinite where
    its divisor is zero (a profit factor with no losing trade, a Calmar ratio with no drawdown). A file path's
    slashes do not count."""
    return any(_task_names(line, key) and _QUOTIENT.search(_ABS_PATH.sub(' ', _PATHLIKE.sub(' ', line)))
               for line in instruction.splitlines())


def nonfinite_flags(instruction, out, known=()):
    """Non-finite numbers in output fields the TASK names (_task_names): NaN at a JSON object key (not inside a list,
    where a rolling window's first values may be NaN) unless the TASK speaks of NaN, null, None or undefined values,
    and +-inf in a JSON value or a CSV cell unless it speaks of infinite values or defines the field as a quotient
    (_quotient). A JSON field that known (the contract's flags) already reports as not finite is left out."""
    inf_ok, nan_ok = bool(_INF_WORDS.search(instruction)), bool(_NAN_WORDS.search(instruction))
    flags = []
    if inf_ok and nan_ok:
        return flags
    for path, name in _outputs(out, ('.json',)):
        for keys, value in _lint_leaves(path) or ():
            if not isinstance(value, float) or math.isfinite(value):
                continue
            if (inf_ok if math.isinf(value) else nan_ok or any(isinstance(p, int) for p in keys)):
                continue
            dotted, field = _dotted(keys), _field(keys) or ''
            if (_task_names(instruction, field) and not (math.isinf(value) and _quotient(instruction, field))
                    and f'{name}: {dotted} is not a finite number' not in known):
                flags.append(f'{name}: {dotted} = {value!r}, which is not a finite number; compute it as the TASK '
                             'defines it')
    for path, name in () if inf_ok else _outputs(out, ('.csv',)):
        try:
            if path.stat().st_size > LINT_FILE_LIMIT:
                continue
            header, body = _load_csv(path)
        except (OSError, UnicodeError, csv.Error):
            continue
        counts, first = Counter(), {}
        for row, cells in enumerate(body, 1):
            for i, cell in enumerate(cells[:len(header or ())]):
                if cell.strip().lower() in _INF_CELLS:
                    counts[i] += 1
                    first.setdefault(i, row)
        for i in sorted(counts):
            if _task_names(instruction, header[i]) and not _quotient(instruction, header[i]):
                flags.append(f'{name}: column {header[i].strip()} holds {counts[i]} infinite value(s) (first in data '
                             f'row {first[i]}), which are not finite numbers; compute them as the TASK defines them')
    return flags[:FLAG_LIMIT]


def _places(value):
    """Decimal places as written: 0 for an int or a whole float, None in exponent form."""
    if isinstance(value, int):
        return 0
    text = repr(value)
    if 'e' in text or 'E' in text:
        return None
    return 0 if text.endswith('.0') else len(text.split('.', 1)[1]) if '.' in text else 0


def _same_number(a, b, rounded=False):
    """Whether two finite JSON numbers are one value: equal ints, floats within CROSS_REL / CROSS_ABS, or, where
    the TASK states a rounding (rounded), the one with fewer decimals the other rounded to them."""
    if isinstance(a, int) and isinstance(b, int):
        return a == b
    try:
        x, y = float(a), float(b)
    except OverflowError:
        return a == b
    if math.isclose(x, y, rel_tol=CROSS_REL, abs_tol=CROSS_ABS):
        return True
    places = [_places(a), _places(b)]
    return rounded and None not in places and abs(x - y) <= 0.5 * 10.0 ** -min(places) * (1 + 1e-9)


def _rounding_for(instruction, name, key):
    """Whether a TASK line that speaks of decimals or rounding names the output file name, the key, or JSON."""
    words = [Path(name).name, str(key), 'json']
    return any((_DECIMALS.search(line) or _ROUNDING.search(line)) and any(_task_names(line, word) for word in words)
               for line in instruction.splitlines())


def _stated_decimals(instruction):
    """Every number of decimals the TASK states anywhere ('rounded to 4 decimal places', 'six decimals')."""
    return {_count_of(m.group(1)) for m in _DECIMALS.finditer(instruction)}


def _clash(instruction, first, second):
    """Whether two (file, key path, number) entries hold two values: not _same_number, and not a rounding the TASK
    states, either on a line about decimals or rounding that names the file or key of the one with fewer decimals
    (or JSON), or anywhere as n decimals, the less precise one written with at most n and within half a unit in the
    n-th place of the other. Two ints must be equal."""
    (a_name, a_keys, a), (b_name, b_keys, b) = first, second
    if _same_number(a, b):
        return False
    places = [_places(a), _places(b)]
    if None in places or isinstance(a, int) and isinstance(b, int):
        return True
    name, keys = (a_name, a_keys) if places[0] <= places[1] else (b_name, b_keys)
    if _rounding_for(instruction, name, _field(keys)) and _same_number(a, b, rounded=True):
        return False
    try:
        gap = abs(float(a) - float(b))
    except OverflowError:
        return True
    return not any(min(places) <= n and gap <= 0.5 * 10.0 ** -n * (1 + 1e-9) for n in _stated_decimals(instruction))


def _wrapped(keys):
    """Whether a leaf sits in a {"value": ...} wrapper, the TASK's layout for an intermediate in a solution file."""
    return keys[-1] == 'value' and sum(isinstance(p, str) for p in keys) > 1


def _stripped(keys):
    return tuple(keys[:-1]) if _wrapped(keys) else tuple(keys)


def _qualifiers(keys):
    """The keys on a leaf's path that say which quantity its field is: all but the field, its value wrapper and
    containers (_CONTAINERS)."""
    return [str(p).lower() for p in _stripped(keys)[:-1] if str(p).lower() not in _CONTAINERS]


def _said_same(instruction, field, names):
    """Whether a TASK line names the key and one of the two files and says the key holds the same value (_SAME), not
    that the files share a schema (_SAME_SHAPE) or that two values agree within a tolerance (_APPROX)."""
    return any(_SAME.search(_SAME_SHAPE.sub(' ', line)) and not _APPROX.search(line) and _task_names(line, field)
               and any(_task_names(line, Path(name).name) for name in names) for line in instruction.splitlines())


def _parallel(first, second):
    """Whether two files ({field: stripped path} of the numbers each writes once) lay out one schema for two things
    (a strategy and its benchmark, two models, two samples): two or more shared fields at one path in both, or
    every field of one of them at the same path in the other."""
    shared = set(first) & set(second)
    same = [field for field in shared if first[field] == second[field]]
    return len(same) >= 2 or bool(shared) and len(same) == len(shared) and shared in (set(first), set(second))


def _one_quantity(instruction, field, first, second, numbers):
    """Whether one key in two files is evidently one quantity: the same qualifiers on both paths (_qualifiers), a
    one-word key (mean, total) at one path, and either exactly one of the two in a value wrapper (a solution file's
    intermediate of a deliverable's number) or a TASK line that says so (_said_same) for two files that do not run
    parallel (_parallel). A shared key name alone is no evidence: parallel files reuse key names for other things."""
    (a_name, a_keys, _), (b_name, b_keys, _) = first, second
    if a_name == b_name or _qualifiers(a_keys) != _qualifiers(b_keys):
        return False
    if len(_words(field)) < 2 and _stripped(a_keys) != _stripped(b_keys):
        return False
    if _wrapped(a_keys) != _wrapped(b_keys):
        return True
    return _said_same(instruction, field, (a_name, b_name)) and not _parallel(numbers[a_name], numbers[b_name])


def crosskey_flags(instruction, out):
    """One quantity written in two output JSON files with two values (_clash: a rounding the TASK states is
    allowed): the key once in each file, lists not entered (their rows are not one quantity), and only where there
    is evidence that the two are one quantity (_one_quantity)."""
    index, counts = {}, Counter()
    for path, name in _outputs(out, ('.json',)):
        for keys, value in _lint_leaves(path) or ():
            field = _field(keys)
            if field is None or any(isinstance(p, int) for p in keys):
                continue
            counts[name, field] += 1
            index.setdefault(field, []).append((name, keys, value))
    numbers = {}  # file: {field: stripped path} of the numbers it writes once
    for field, entries in index.items():
        for name, keys, value in entries:
            if counts[name, field] == 1 and _is_number(value):
                numbers.setdefault(name, {})[field] = _stripped(keys)
    flags = []
    for field, entries in sorted(index.items()):
        entries = [entry for entry in entries if counts[entry[0], field] == 1 and _is_number(entry[2])]
        for i, first in enumerate(entries):
            clash = next((second for second in entries[i + 1:] if _one_quantity(
                instruction, field, first, second, numbers) and _clash(instruction, first, second)), None)
            if clash is not None:
                flags.append(f'{first[0]}: {_dotted(first[1])} = {first[2]!r}, but {clash[0]}: {_dotted(clash[1])} '
                             f'= {clash[2]!r}; one quantity has two values: compute it once and write that value in '
                             'both files')
                break
    return flags[:FLAG_LIMIT]


# T1_S12_POST: the program's own validation verdicts and comparison errors, read from its JSON outputs with zero
# requests (selfval_flags). A false boolean counts where the TASK says it is true (`parity_holds = True`), or where
# its key names the program's own check (validates, converged, consistent, matches; pass or ok only beside check or
# validation), never a statistical test's outcome on the data (Kupiec, a breach, a rejection), and in both cases
# only where the TASK states no rule for it: under a rule the TASK states (`true if max_error <= 3 * se`) a correct
# program may write false, even where the TASK also says it must be true (all 18 passing local lookback runs did,
# from discrete-monitoring bias; Kirk's did too). Only at an object key: a flag inside a list is a row's data. An
# error key names an error or a discrepancy and none of an ordinary metric's words (tracking, standard, squared, a
# fit's or forecast's): it is judged against the tolerance a TASK sentence that names the key states (a plain number,
# never a multiple of another quantity), or, where none does, only as a relative error (rel, relative, pct) that a
# comparison or validation stands behind (its file, its keys or the TASK), above SELFVAL_RELATIVE (half of what it
# compares with).
SELFVAL_RELATIVE = 0.5
SELFVAL_LIMIT = 6
_VERDICT_STEMS = ('validat', 'verif', 'converg', 'consisten', 'agree', 'match', 'reconcil')
_VERDICT_PASS = frozenset({'pass', 'passed', 'passes', 'ok', 'success', 'succeeded', 'holds'})
_VERDICT_CHECK = frozenset({'check', 'checks', 'checked', 'validation', 'sanity', 'selfcheck', 'verification'})
_NOT_VERDICT = frozenset({'kupiec', 'christoffersen', 'basel', 'traffic', 'backtest', 'reject', 'rejected', 'rejects',
                          'null', 'hypothesis', 'pvalue', 'breach', 'breached', 'breaches', 'violation', 'violated',
                          'violations', 'exceed', 'exceeds', 'exceeded', 'arbitrage', 'excluded', 'missing', 'outlier',
                          'default', 'defaulted', 'stationary', 'significant', 'jump', 'jumps'})
_ERROR_WORDS = frozenset({'error', 'errors', 'err', 'discrepancy', 'discrepancies', 'relerr', 'abserr'})
_NOT_SELFVAL = frozenset({'tracking', 'standard', 'std', 'stderr', 'se', 'squared', 'mse', 'rmse', 'mae', 'mape',
                          'forecast', 'forecasts', 'prediction', 'predicted', 'fit', 'fitted', 'fitting', 'residual',
                          'residuals', 'train', 'training', 'test', 'testing', 'cv', 'oos', 'calibration', 'pricing',
                          'model', 'hedging', 'hedge', 'sampling', 'estimation', 'measurement', 'rounding',
                          'truncation', 'discretization', 'discretisation', 'numerical', 'violation', 'violations'})
_RELATIVE_WORDS = frozenset({'rel', 'relative', 'pct', 'percent', 'percentage', 'frac', 'fraction', 'relerr'})
_PERCENT_WORDS = frozenset({'pct', 'percent', 'percentage'})
_COMPARE_WORDS = frozenset({'comparison', 'compare', 'compared', 'validation', 'validate', 'validates', 'validated',
                            'verify', 'verification', 'verified', 'check', 'checks', 'vs', 'versus', 'mc', 'montecarlo',
                            'monte', 'benchmark', 'reference', 'analytic', 'analytical', 'exact', 'theoretical',
                            'simulated', 'simulation', 'agreement', 'consistency'})
_COMPARE_TASK = re.compile(r'\bvalidat|\bverif|\bcompar|\bversus\b|\bvs\.?(?=\s)|\bagree|\brelative error|'
                           r'\bcheck', re.I)
_SELFVAL_BOUND = re.compile(r'(?:<=|<|≤|\bless than\b|\bbelow\b|\bat most\b|\bsmaller than\b|\bwithin\b|\bunder\b|'
                            r'\bnot exceed(?:ing)?\b|\bexceed(?:s|ing)?\b|\btolerance(?: of)?\b)\s*\$?\s*'
                            r'((?>10\s*\^\s*\{?\s*-\s*\d+\s*\}?|\d*\.?\d+(?:[eE][-+]?\d+)?))\s*'
                            r'(%|percent\b|bps\b|basis points\b)?'
                            r'(?!\s*(?:[*×·/^(]|x\b|times\b|standard\b|std\b|se\b|sigma\b|σ|'
                            r'[A-Za-z]\w*_\w*|[A-Za-z]\w*\())', re.I)  # never a multiple of another quantity
_SELFVAL_RULE = re.compile(r'[<>≤≥]|\bwithin\b|\bless than\b|\bbelow\b|\bat most\b|\bexceed|\btolerance\b|'
                           r'\bstandard errors?\b', re.I)
_PLACEHOLDER = re.compile(r'<[A-Za-z_ |]{1,20}>')  # a template's slot, `<bool>`: not a rule


def _verdict_key(words):
    if set(words) & _NOT_VERDICT:
        return False
    return (any(word.startswith(_VERDICT_STEMS) for word in words)
            or bool(set(words) & _VERDICT_PASS and set(words) & _VERDICT_CHECK))


def _said_true(instruction, key):
    """Whether the TASK says key is true: `key = True`, `key: true`, `key must be true`; not a definition (`key = true
    if ...`)."""
    return re.search(rf'(?<![A-Za-z0-9_]){re.escape(key)}`?\s*(?:={{1,2}}|:|\bis\b|\b(?:must|should) be\b)\s*`?true\b'
                     r'(?!`?\s*(?:if|iff|when|whenever|only|unless|exactly)\b)', instruction, re.I) is not None


def _rule_stated(instruction, key):
    """Whether a TASK line that names key states a rule for it: a comparison, a bound or a tolerance."""
    return any(_task_names(line, key) and _SELFVAL_RULE.search(_PLACEHOLDER.sub(' ', line))
               for line in instruction.splitlines())


def _stated_bound(instruction, key, relative, percent):
    """(threshold in the key's own units, the TASK sentence) from the first TASK sentence that names key and bounds it
    (`< 0.01`, `within 1%`, `at most 5 bps`), read so that only a value above it under every reading of the units
    exceeds it; None when no sentence does."""
    for line in instruction.splitlines():
        for sentence in re.split(r'(?<=[.!?;])\s+', line.strip()):
            if not _task_names(sentence, key):
                continue
            match = _SELFVAL_BOUND.search(sentence)
            if not match:
                continue
            try:
                bound = _bound_value(re.sub(r'\s+', '', match.group(1)))
            except ValueError:
                continue
            unit = (match.group(2) or '').lower()
            if unit.startswith(('bps', 'basis')):
                if not relative or percent:
                    continue
                bound /= 10000
            elif unit:
                if not relative:
                    continue
                bound = bound if percent else bound / 100
            elif percent:
                bound = max(bound, 100 * bound)  # a fraction or a percentage: the looser reading
            if bound > 0:
                return bound, sentence.strip()
    return None


def selfval_flags(instruction, out, limit=SELFVAL_LIMIT, *, stated_only=False):
    """[(identity, flag)]: the program's own validation verdicts that are false, and its own comparison errors past
    the TASK's tolerance (or, without one, a relative error above SELFVAL_RELATIVE), in its JSON outputs. One flag per
    verdict and per file and error key (the largest value, and how many exceed). The identity holds no number or
    position, so a flag that moves within a file is the same flag. Never raises: an output that does not read finds
    nothing. With `stated_only` (thin's S2-05 rule 3) only what the TASK's own words state counts: a false verdict
    the TASK says is true (_said_true), an error past a tolerance a TASK sentence gives (_stated_bound); a key's
    name never makes a verdict and no default tolerance applies."""
    flags, context = [], _COMPARE_TASK.search(instruction or '') is not None
    for path, name in _outputs(out, ('.json',)):
        errors = {}
        for keys, value in _lint_leaves(path) or ():
            key = next((p for p in reversed(keys) if isinstance(p, str)), None)
            if key is None:
                continue
            words = _words(key)
            if value is False and not any(isinstance(p, int) for p in keys) and (
                    not _rule_stated(instruction or '', key)
                    and (_said_true(instruction or '', key)
                         or (not stated_only and _verdict_key(words) and context))):
                flags.append((('selfval', name, _dotted(keys)),
                              f'{name}: {_dotted(keys)} = false: the program\'s own check reports a failure. Either '
                              'what it checks is computed wrong, or the check itself is (its tolerance, the units it '
                              'compares, a simulation\'s sampling error): fix the one that is wrong, and never set '
                              'this value directly'))
            elif (_is_number(value) and set(words) & _ERROR_WORDS and not set(words) & _NOT_SELFVAL):
                errors.setdefault(key, []).append((abs(float(value)), keys))
        stem = _words(Path(name).stem)
        for key, values in errors.items():
            words = set(_words(key))
            relative, percent = bool(words & _RELATIVE_WORDS), bool(words & _PERCENT_WORDS)
            stated = _stated_bound(instruction or '', key, relative, percent)
            if stated is not None:
                threshold, sentence = stated
            elif not stated_only and relative and (context or set(stem) & _COMPARE_WORDS or any(
                    set(_words(p)) & _COMPARE_WORDS for _, keys in values for p in keys if isinstance(p, str))):
                threshold, sentence = (100 * SELFVAL_RELATIVE if percent else SELFVAL_RELATIVE), None
            else:
                continue
            above = [(value, keys) for value, keys in values if value > threshold * (1 + 1e-9)]
            if not above:
                continue
            value, keys = max(above, key=lambda item: item[0])
            told = f'; the TASK says: "{sentence[:300]}"' if sentence else ''
            flags.append((('selfval', name, key),
                          f'{name}: {key} reaches {value:.6g} at {_dotted(keys)} ({len(above)} of {len(values)} '
                          f'values above {threshold:g}){told}: the two computations it compares disagree. Find which '
                          'side is wrong (a formula, a sign, the direction or scaling of a transform, a grid too '
                          'coarse) and fix that, never the comparison itself'))
    return flags[:limit]
