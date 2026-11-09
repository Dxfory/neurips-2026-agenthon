"""Evidence that a program which ran cleanly is still an unfinished draft (a milestone, not the final answer).

findings(program, expected_files, files) reads the program source that ran and the bytes of its output files and
returns short lines naming clear signs that the work is not done, so the host can keep going instead of stopping. It
only reports evidence: it never edits, deletes, rewrites or runs a program or a file, and it never raises (a part it
cannot read is not judged; any internal error gives []).

    1. markers in comments and in the module docstring (other docstrings, string literals and names are not read):
       TODO, FIXME, TBD, stub, not (yet) implemented, to be implemented, implement(ed) later; placeholder only for the
       work itself or for now ("placeholder implementation", "placeholder for the full model", "placeholder for now",
       "until the next milestone": a bare one mostly pre-allocates a column the program fills later); and, only after
       a deferral word ("will be added in the next step", "comes in a later milestone"): milestone, in/for the next
       step, subsequent step(s). A marker after a negation ("no placeholders", "replaces the stub") or one its
       sentence says the program fills ("stub, replaced below") does not count, nor a fixed-income stub ("stub
       period", "front stub"), a scope note ("not implemented since/because ...", "... by/in the task") or a
       pre-allocated container ("placeholder for the model results", "placeholder array"). At most 2 lines.
    2. an unconditional `raise NotImplementedError` (not inside an if/loop/try branch, not an abstract method).
    3. a function whose body (docstring aside) is only pass/... that the module calls and does not define again.
    4. required outputs (every output when none is named): a CSV/TSV with a header and no data row (unless the TASK,
       when given, lets the table be empty), a JSON that is [] or {}, a JSON whose 3+ numeric values (booleans and
       values under count-like or legitimately-zero keys such as count, id, breaches, errors, pnl aside) are all
       exactly 0, a parquet file with no rows.
"""
from __future__ import annotations

import ast
import csv
import io
import json
import os
import re
import tokenize
import warnings
from collections.abc import Mapping
from pathlib import PurePosixPath

PREFIX = 'unfinished: '
MAX_LINES = 8
MAX_CHARS = 160
COMMENT_LINES = 2
STUB_LINES = 3
PROGRAM_CHARS = 2 * 1024 * 1024  # a longer program is not read
SMALL_FILE = 1024 * 1024  # a header-only table or a row-less parquet file is far smaller; a larger one is not read
JSON_BYTES = 32 * 1024 * 1024
# clock_fix (bound_parquet): a parquet file is decoded only when its metadata (the footer, read without decoding a page)
# keeps it within PARQUET_CELLS cells (rows x columns) and PARQUET_BYTES of uncompressed column data. A small, highly
# compressible file (6 million equal values fit in 3 kB) can otherwise decode to more memory than the solver has.
PARQUET_CELLS = 5_000_000
PARQUET_BYTES = 256 * 1024 * 1024
PARQUET_ROW_GROUPS = 100_000
JSON_NODES = 1_000_000
ZERO_VALUES = 3
NAME_CHARS = 80  # a longer output name keeps its end, so each file's line stays distinct
HARNESS_FILES = frozenset({'_agent_status.json'})  # the solver's own file, skipped when every output is judged

_SEP = r'[ \t-]+'
# (pattern, kind): a 'noun' or 'verb' marker counts unless negated, a noun also not when its sentence says the program
# fills it; a 'defer' phrase counts only after a deferral word (the host's own prompts speak of milestones and steps);
# a 'context' marker only with a _WORK noun just after it or a _TEMPORARY phrase in its sentence. Saved T1 programs
# (2,691 distinct, 10-28) carried a bare "placeholder" comment in 246, nearly all a pre-allocated value or frame.
_PHRASES = tuple((re.compile(rf'\b{pattern}\b', re.I), kind) for pattern, kind in (
    ('placeholders?', 'context'), ('to-?dos?', 'noun'), ('fixmes?', 'noun'), ('tbd', 'noun'), ('stubs?', 'stub'),
    (rf'not{_SEP}(?:yet{_SEP})?implemented', 'unimplemented'), (rf'to{_SEP}be{_SEP}implemented', 'verb'),
    (rf'implement(?:ed)?{_SEP}later', 'verb'), ('milestones?', 'defer'),
    (rf'(?:in|for){_SEP}the{_SEP}next{_SEP}step', 'defer'), (rf'subsequent{_SEP}steps?', 'defer')))
WINDOW = 6  # words before a marker, within its clause, read for negation or deferral
_WORD = re.compile(r'[a-z0-9]+')
_NOT = re.compile(r"n['\u2019]t\b")
_CLAUSE_END = re.compile(r'[.!?](?=\s|$)|[;:()\[\]{}\u2013\u2014]|\s-+\s')
_SENTENCE_END = re.compile(r'[.!?](?=\s|$)')
_NEGATION = frozenset('no not non never without instead rather nor none nothing avoid avoids avoided avoiding '
                      'replace replaces replaced replacing remove removes removed removing'.split())
_FILLED = frozenset('filled overwritten overwrite overwrites populated replaced removed below'.split())
_STILL_OPEN = frozenset('later will next future subsequent eventually pending yet todo tbd temporary temporarily '
                        'until'.split())
_DEFERRAL = frozenset('next later future subsequent upcoming further coming comes come left leave leaves deferred '
                      'defer defers postponed postpone until till pending will tbd todo'.split())
_WORK = frozenset('implementation implementations logic model models computation computations calculation '
                  'calculations method methods algorithm version code function'.split())
WORK_WINDOW = 4  # words after a context marker read for a _WORK noun
_TEMPORARY = re.compile(r'\b(?:for now|not yet|temporary[ \t-]+placeholders?|placeholders?[ \t-]+temporarily|'
                        r'(?:next|later|subsequent|future)[ \t-]+(?:steps?|milestones?|drafts?|versions?|iterations?|'
                        r'rounds?))\b', re.I)
_DATA_FLOW = frozenset('used use uses using needed need needs required requires feed feeds fed input inputs reuse '
                       'reused store stored stores cache cached keep kept save saved pass passed prepare prepared '
                       'prepares consumed read reads available ready'.split())
# a fixed-income stub (a short first or last coupon period) is not an unfinished function
_STUB_AFTER = frozenset('period periods coupon coupons date dates accrual accruals schedule schedules interval '
                        'intervals'.split())
_STUB_BEFORE = frozenset('front back short long initial final'.split())
# 'not implemented' with its reason (the task leaves it out) is a scope note, not unfinished work
_SCOPE_REASON = re.compile(r'[\s,:;(\-–—]*(?:since|because|(?:by|in)[ \t]+the[ \t]+task)\b', re.I)
# 'placeholder for the ... results' (or 'placeholder array'): a pre-allocated container the program fills
_CONTAINERS = frozenset('result results output outputs dict dicts dictionary array arrays frame frames dataframe '
                        'dataframes list lists'.split())
_PHRASE_END = re.compile(r'[,;:.!?()\[\]{}–—]')
CONTAINER_WINDOW = 8  # words of 'placeholder for ...' read for a container noun

_NIE = frozenset({'NotImplementedError', 'NotImplemented'})
_ABSTRACT_DECORATORS = frozenset({'abstractmethod', 'abstractproperty', 'abstractclassmethod',
                                  'abstractstaticmethod', 'overload'})
_ABSTRACT_BASES = frozenset({'ABC', 'ABCMeta', 'Protocol'})
_DEFS = (ast.FunctionDef, ast.AsyncFunctionDef)

_CONTROL = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]')
_CELL_SPLIT = re.compile(r'[,;\t|]')
# count-like or legitimately-zero key words: a key holding one exempts every value under it from rule 4's zero count
_COUNT_WORDS = frozenset('count counts cnt n num nums number numbers nobs id ids idx index indices rank seed year '
                         'month day lag lags window version status code flag flags is has rows trades breach breaches '
                         'violation violations exceedance exceedances exception exceptions error errors missing '
                         'dropped removed skipped outlier outliers residual residuals diff diffs difference '
                         'differences drift bias pnl'.split())

# a TASK that lets a table have no data rows: explicitly, or a listing of events that may not occur
_EMPTY_TABLE_OK = re.compile(r'\b(?:header[ \t-]+only|only[ \t]+(?:the|a)[ \t]+header|no[ \t]+data[ \t]+rows?|'
                             r'(?:may|might|can|could)[ \t]+be[ \t]+empty)\b', re.I)
_EVENT_LISTING = re.compile(r'\b(?:if[ \t]+there[ \t]+(?:are|is)[ \t]+no|if[ \t]+none|only[ \t]+(?:the[ \t]+)?rows|'
                            r'list[ \t]+(?:each|every|all)|one[ \t]+row[ \t]+(?:per|for[ \t]+each))\b', re.I)
_EVENT = re.compile(r'\b(?:breach(?:es)?|exceptions?|violations?|exceedances?|events?|trades?|signals?)\b', re.I)
_SENTENCES = re.compile(r'(?<=[.!?])\s+|\n')


def findings(program: str, expected_files, files: dict, instruction: str = '', *, bound_parquet=False) -> list[str]:
    """Short lines ('unfinished: ...', at most MAX_LINES of at most MAX_CHARS characters, no repeats) naming the
    evidence that `program` (the source that ran) is unfinished, read from its text and from `files` (relative output
    path -> bytes) for the outputs in `expected_files` (all of them when empty). `instruction` (the TASK text, optional)
    only exempts a header-only table the TASK lets be empty. `bound_parquet` (clock_fix): a parquet file beyond the
    bound its metadata shows (parquet_bounded) is not decoded, so not judged. [] when nothing is found or on any
    internal error."""
    try:
        found = []
        for line in _code_findings(program) + _output_findings(expected_files, files, instruction, bound_parquet):
            line = _clip(PREFIX + line)
            if line not in found:
                found.append(line)
        return found[:MAX_LINES]
    except Exception:
        return []


def _clip(line, limit=MAX_CHARS):
    return line if len(line) <= limit else line[:limit - 3] + '...'


def _words(text):
    return _WORD.findall(_NOT.sub(' not', text.lower()))


def _marker(text):
    """Position of the first marker that counts in one comment or docstring line (rule 1), or None."""
    for pattern, kind in _PHRASES:
        for match in pattern.finditer(text):
            before = set(_words(_CLAUSE_END.split(text[:match.start()])[-1])[-WINDOW:])
            if before & _NEGATION:
                continue
            if kind == 'stub':
                if (_STUB_AFTER.intersection(_words(text[match.end():])[:1])
                        or _STUB_BEFORE.intersection(_words(text[:match.start()])[-1:])):
                    continue
                kind = 'noun'
            elif kind == 'unimplemented':
                if 'yet' not in match.group().lower() and _SCOPE_REASON.match(text, match.end()):
                    continue
                kind = 'verb'
            if kind == 'defer':
                if before & _DEFERRAL and not before & _DATA_FLOW:
                    return match.start()
                continue
            tail = _SENTENCE_END.split(text[match.end():])[0]
            after = _words(tail)
            if kind == 'context':
                if _TEMPORARY.search(_SENTENCE_END.split(text[:match.start()])[-1] + match.group() + tail):
                    return match.start()
                phrase = _words(_PHRASE_END.split(text[match.end():])[0])[:CONTAINER_WINDOW]
                if phrase and (phrase[0] in _CONTAINERS
                               or (phrase[0] == 'for' and _CONTAINERS.intersection(phrase[1:]))):
                    continue  # pre-allocation: 'placeholder for the model results', 'placeholder array'
                if _WORK.intersection(after[:WORK_WINDOW]) and not _FILLED.intersection(after):
                    return match.start()
                continue
            if kind == 'noun' and _FILLED.intersection(after) and not (
                    _STILL_OPEN.intersection(after) or 'for now' in tail.lower()):
                continue
            return match.start()
    return None


def _excerpt(text, start, room):
    """text cut to at most `room` characters around position `start`, '...' marking each cut."""
    if len(text) <= room:
        return text
    left = max(0, min(start - 20, len(text) - room + 3))
    head = '...' if left else ''
    piece = text[left:left + room - len(head)]
    if left + len(piece) < len(text):
        piece = piece[:room - len(head) - 3] + '...'
    return head + piece


def _parse(program):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # invalid escape sequences in the model's strings
            return ast.parse(program)
    except Exception:
        return None


def _texts(program, tree):
    """(line, where, text) of every comment, and of every module docstring line."""
    found = []
    try:
        for token in tokenize.generate_tokens(io.StringIO(program).readline):
            if token.type == tokenize.COMMENT:
                found.append((token.start[0], 'comment', token.string.lstrip('#').strip()))
    except Exception:
        pass  # a program that does not tokenize: the comments read before the error still count
    doc = ast.get_docstring(tree, clean=False) if tree is not None else None
    if doc:
        first = tree.body[0].lineno
        found += [(first + offset, 'module docstring', text.strip()) for offset, text in enumerate(doc.split('\n'))]
    return sorted(found)


def _marker_lines(program, tree):
    found = []
    for line, where, text in _texts(program, tree):
        start = _marker(text)
        if start is None:
            continue
        head = f'line {line} {where} says '
        found.append(head + "'" + _excerpt(text, start, MAX_CHARS - len(PREFIX) - len(head) - 2) + "'")
        if len(found) == COMMENT_LINES:
            break
    return found


def _name(node):
    if isinstance(node, ast.Call):
        node = node.func
    return node.id if isinstance(node, ast.Name) else node.attr if isinstance(node, ast.Attribute) else None


def _body(fn):
    body = fn.body
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
            and isinstance(body[0].value.value, str):
        body = body[1:]
    return body


def _empty(fn):
    """The body, docstring aside, is only pass and ... (or nothing)."""
    return all(isinstance(s, ast.Pass) or (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant)
                                           and s.value.value is Ellipsis) for s in _body(fn))


def _raises_nie(stmt):
    return isinstance(stmt, ast.Raise) and stmt.exc is not None and _name(stmt.exc) in _NIE


def _real(fn):
    body = _body(fn)
    return not _empty(fn) and not (len(body) == 1 and _raises_nie(body[0]))


class _Module:
    """What rules 2 and 3 need from the parsed program."""

    def __init__(self, tree):
        self.parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
        nodes = list(ast.walk(tree))
        self.defs = [n for n in nodes if isinstance(n, _DEFS)]
        self.real = {}
        for fn in self.defs:
            self.real[fn.name] = self.real.get(fn.name, 0) + _real(fn)
        self.bases = {_name(base) for n in nodes if isinstance(n, ast.ClassDef) for base in n.bases}
        calls = [n.func for n in nodes if isinstance(n, ast.Call)]
        self.called = {c.id for c in calls if isinstance(c, ast.Name)}
        self.called_attrs = {c.attr for c in calls if isinstance(c, ast.Attribute)}

    def abstract(self, fn):
        """An abstract, overloaded or overridden def: another def of its name has a real body, its class is a base
        of another class or an ABC/Protocol, or a decorator says so."""
        if any(_name(d) in _ABSTRACT_DECORATORS for d in fn.decorator_list):
            return True
        if self.real.get(fn.name, 0) - _real(fn) > 0:
            return True
        owner = self.parents.get(fn)
        if isinstance(owner, ast.ClassDef):
            marks = {_name(b) for b in owner.bases} | {_name(k.value) for k in owner.keywords}
            return owner.name in self.bases or bool(marks & _ABSTRACT_BASES)
        return False

    def unconditional_owner(self, node):
        """The def (or the module) whose every run reaches `node`, or None when a branch, loop, try or class body
        holds it."""
        node = self.parents.get(node)
        while isinstance(node, (ast.With, ast.AsyncWith)):
            node = self.parents.get(node)
        return node if isinstance(node, (*_DEFS, ast.Module)) else None


def _nie_line(module, tree):
    raises = sorted((n for n in ast.walk(tree) if _raises_nie(n)), key=lambda n: (n.lineno, n.col_offset))
    for node in raises:
        owner = module.unconditional_owner(node)
        if owner is None or (isinstance(owner, _DEFS) and module.abstract(owner)):
            continue
        where = f' in {owner.name}()' if isinstance(owner, _DEFS) else ''
        return [f'line {node.lineno} raises NotImplementedError{where}']
    return []


def _stub_lines(module):
    found = []
    for fn in sorted(module.defs, key=lambda n: n.lineno):
        dunder = fn.name.startswith('__') and fn.name.endswith('__')
        if dunder or not _empty(fn) or module.abstract(fn):
            continue
        method = isinstance(module.parents.get(fn), ast.ClassDef)
        if fn.name in (module.called_attrs if method else module.called):
            found.append(f"function '{fn.name}' (line {fn.lineno}) has an empty body and is called")
            if len(found) == STUB_LINES:
                break
    return found


def _code_findings(program):
    if not isinstance(program, str) or not program.strip() or len(program) > PROGRAM_CHARS:
        return []
    tree = _parse(program)
    found = _marker_lines(program, tree)
    if tree is not None:
        module = _Module(tree)
        found += _nie_line(module, tree) + _stub_lines(module)
    return found


def _norm(name):
    try:
        name = os.fspath(name)
    except TypeError:
        return None
    if not isinstance(name, str) or not name.strip():
        return None
    return PurePosixPath(name.strip().replace('\\', '/')).as_posix()


def _targets(expected_files, files):
    """(name, bytes) of the outputs to judge: the named ones present in `files`, or every file when none is named."""
    table = {}
    for key, data in (files.items() if isinstance(files, Mapping) else ()):
        name = _norm(key)
        if isinstance(data, (bytearray, memoryview)):
            data = bytes(data)
        if name is not None and isinstance(data, bytes):
            table.setdefault(name, data)
    if isinstance(expected_files, (str, os.PathLike)):
        expected_files = (expected_files,)
    names = [n for n in (_norm(x) for x in (expected_files or ())) if n is not None]
    if names:
        return [(n, table[n]) for n in dict.fromkeys(names) if n in table]
    return [(n, d) for n, d in sorted(table.items()) if PurePosixPath(n).name not in HARNESS_FILES]


def _text(data):
    """UTF-8 text (BOM aside), or None for bytes that are not text."""
    try:
        text = data.decode('utf-8-sig')
    except UnicodeDecodeError:
        return None
    return None if _CONTROL.search(text) else text


def _number(cell):
    try:
        float(cell.strip())
    except ValueError:
        return False
    return True


def _header_only(data, delimiter):
    """One non-blank row that reads as a header (no cell is a number) and no row after it."""
    text = _text(data) if len(data) <= SMALL_FILE else None
    if not text or not text.strip():
        return False
    rows = []
    try:
        for row in csv.reader(io.StringIO(text, newline=''), delimiter=delimiter):
            if any(cell.strip() for cell in row):
                rows.append(row)
                if len(rows) > 1:
                    return False
    except csv.Error:
        return False
    return len(rows) == 1 and not any(_number(part) for cell in rows[0] for part in _CELL_SPLIT.split(cell))


def _count_like(key, cache):
    if not isinstance(key, str):
        return False
    if key not in cache:
        words = re.findall(r'[a-z]+', re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', key).lower())
        cache[key] = bool(_COUNT_WORDS.intersection(words)) or key.lower().startswith(('n_', 'num'))
    return cache[key]


def _zero_values(value):
    """How many numeric values the JSON holds when there are ZERO_VALUES or more and all are exactly 0, else 0.
    Booleans and every value under a key with a count-like or legitimately-zero word (count, id, breaches, errors,
    pnl, ...; at any depth above it) are not counted."""
    stack, cache, seen, count = [(value, False)], {}, 0, 0
    while stack:
        item, exempt = stack.pop()
        seen += 1
        if seen > JSON_NODES:
            return 0
        if isinstance(item, dict):
            stack.extend((v, exempt or _count_like(k, cache)) for k, v in item.items())
        elif isinstance(item, list):
            stack.extend((v, exempt) for v in item)
        elif isinstance(item, (int, float)) and not isinstance(item, bool) and not exempt:
            if item != 0:
                return 0
            count += 1
    return count if count >= ZERO_VALUES else 0


def _json_line(name, data):
    text = _text(data) if len(data) <= JSON_BYTES else None
    if text is None:
        return None
    try:
        value = json.loads(text)
    except (ValueError, RecursionError):
        return None
    if isinstance(value, (list, dict)) and not value:
        return f"{name} is an empty JSON {'list' if isinstance(value, list) else 'object'}"
    zeros = _zero_values(value)
    return f'{name} has all {zeros} numeric values exactly 0' if zeros else None


def parquet_bound(data):
    """clock_fix: True when a parquet file's own metadata keeps its decoded size within PARQUET_CELLS cells (rows x
    columns, and the leaf values of nested columns) and PARQUET_BYTES of uncompressed column data, False when it shows
    more, None when there is no metadata to read (not a parquet file, or a damaged one: a reader fails on its footer
    before decoding a page). Only the footer is read. Never raises."""
    try:
        if not isinstance(data, (bytes, bytearray)) or not data.startswith(b'PAR1'):
            return None
        import pyarrow.parquet as pq
        meta = pq.ParquetFile(io.BytesIO(bytes(data))).metadata
        rows, width, groups = int(meta.num_rows), int(meta.num_columns), int(meta.num_row_groups)
    except Exception:
        return None
    try:
        if min(rows, width, groups) < 0 or groups > PARQUET_ROW_GROUPS or rows * max(width, 1) > PARQUET_CELLS:
            return False
        size = values = 0
        for index in range(groups):
            group = meta.row_group(index)
            size += max(int(group.total_byte_size), 0)
            values += sum(max(int(group.column(column).num_values), 0) for column in range(group.num_columns))
            if size > PARQUET_BYTES or values > PARQUET_CELLS:
                return False
        return True
    except Exception:
        return False


def parquet_bounded(data) -> bool:
    """clock_fix: whether parquet_bound shows the file within the bound (a file without readable metadata is not)."""
    return parquet_bound(data) is True


def _parquet_empty(data, bound=False):
    if len(data) > SMALL_FILE or not data.startswith(b'PAR1'):
        return False
    if bound and not parquet_bounded(data):
        return False
    try:
        import pandas as pd
    except Exception:
        return False
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return len(pd.read_parquet(io.BytesIO(data))) == 0
    except Exception:
        return False


def _empty_table_allowed(instruction):
    """The TASK lets a table have no data rows: it says so ('header only', 'no data rows', 'may be empty'), or one of
    its sentences lists events that may not occur ('list each breach', 'one row per trade', 'if there are no
    signals')."""
    if not isinstance(instruction, str) or not instruction.strip():
        return False
    if _EMPTY_TABLE_OK.search(instruction):
        return True
    return any(_EVENT_LISTING.search(s) and _EVENT.search(s) for s in _SENTENCES.split(instruction))


def _output_findings(expected_files, files, instruction='', bound_parquet=False):
    found = []
    empty_ok = _empty_table_allowed(instruction)
    for path, data in _targets(expected_files, files):
        suffix = PurePosixPath(path).suffix.lower()
        name = path if len(path) <= NAME_CHARS else '...' + path[3 - NAME_CHARS:]
        if suffix in ('.csv', '.tsv'):
            if not empty_ok and _header_only(data, '\t' if suffix == '.tsv' else ','):
                found.append(f'{name} has a header but no data rows')
        elif suffix == '.json':
            line = _json_line(name, data)
            if line:
                found.append(line)
        elif suffix in ('.parquet', '.pq') and _parquet_empty(data, bound_parquet):
            found.append(f'{name} has no rows')
    return found
