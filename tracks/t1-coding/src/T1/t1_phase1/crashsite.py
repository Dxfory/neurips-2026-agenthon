"""Crash report for the generated program: where it failed and what the local variables held.

The source of this module is appended to the worker's sitecustomize (baseline.SANDBOX_SITE), so it runs
inside the generated program only, uses the standard library only and never raises. When the program dies
of an uncaught exception, the report goes to T1_CRASH_REPORT for the controller, which sends it to the
model with the repair request. Values are described by type, shape, dtype, columns, keys and a few leading
items; arbitrary objects are never repr()'d. The report is model input only: it is never exported.
"""
import difflib
import itertools
import linecache
import os
import re
import sys

REPORT_CAP = 3500
LINE_CAP = 240
MAX_LOCALS = 24
HEAD_FRAMES, TAIL_FRAMES = 2, 6
_SKIP_TYPES = ('module', 'function', 'builtin_function_or_method', 'type', 'method', 'ABCMeta')


def _cut(text, cap=LINE_CAP):
    text = ' '.join(str(text).split('\n'))
    return text if len(text) <= cap else text[:cap - 3] + '...'


def _first(values, count=3):
    return list(itertools.islice(values, count))


def _moment(item):
    """A NumPy datetime64 or timedelta64 scalar as ISO-like text (2024-01-31, 2024-01-31T09:30:00,
    1 days 00:00:00); tolist() would give integer nanoseconds for the [ns] unit."""
    text = str(item)
    if item.dtype.kind == 'M':
        return re.sub(r'(\.\d*?)0+$', r'\1', re.sub(r'T00:00:00(\.0*)?$', '', text)).rstrip('.')
    if text == 'NaT' or not re.search(r'\[(W|D|h|m|s|ms|us|ns)\]$', str(item.dtype)):
        return text  # months and years have no fixed length: '3 months'
    days, rest = divmod(int(item.astype('timedelta64[ns]').astype('int64')), 86400 * 10 ** 9)
    hours, rest = divmod(rest, 3600 * 10 ** 9)
    minutes, rest = divmod(rest, 60 * 10 ** 9)
    seconds, nanos = divmod(rest, 10 ** 9)
    return f'{days} days {hours:02}:{minutes:02}:{seconds:02}' + (f'.{nanos:09}'.rstrip('0') if nanos else '')


def brief(value, depth=0):
    """One line describing value, never its full repr; any error falls back to the type name."""
    kind = type(value)
    name, module = kind.__name__, getattr(kind, '__module__', '') or ''
    try:
        if value is None or isinstance(value, (bool, int, float, complex)):
            return _cut(repr(value), 60)
        if isinstance(value, str):  # sliced first: a huge string is never copied whole
            return _cut(repr(value[:80]), 70) + (f' (len {len(value)})' if len(value) > 60 else '')
        if isinstance(value, (bytes, bytearray)):
            return f'{name}(len {len(value)})'
        if module.startswith('numpy'):
            if hasattr(value, 'shape') and hasattr(value, 'dtype') and hasattr(value, 'flat'):
                text = f'{name}(dtype={value.dtype}, shape={tuple(value.shape)})'
                if int(value.size) and getattr(value.dtype, 'kind', None) in ('M', 'm'):
                    text += ' first ' + _cut('[' + ', '.join(_moment(v) for v in value.flat[0:3]) + ']', 90)
                elif int(value.size):
                    head = value.flat[0:3].tolist()
                    if str(value.dtype) == 'object':  # Python objects: described, not repr()'d
                        text += ' first [' + ', '.join(brief(v, 2) for v in head) + ']'
                    else:
                        text += ' first ' + _cut(repr(head), 90)
                return _cut(text, 300)
            return _cut(f'{name} {value!r}', 80)
        if module.startswith('pandas'):
            if name == 'DataFrame':
                columns, dtypes = list(value.columns[:15]), value.dtypes
                listed = ', '.join(f'{c!r}:{dtypes.iloc[i]}' for i, c in enumerate(columns))
                more = f' (+{len(value.columns) - 15} more)' if len(value.columns) > 15 else ''
                text = f'DataFrame(shape={value.shape}) columns [{listed}{more}] index {type(value.index).__name__}'
                if len(value.index):
                    text += f' from {value.index[0]!r}'
                return _cut(text, 700)
            if name == 'Series':
                text = (f'Series(name={value.name!r}, dtype={value.dtype}, len={len(value)}) '
                        f'index {type(value.index).__name__}')
                if len(value):
                    text += ' first ' + _cut(repr(value.iloc[:3].tolist()), 90) + f' at {value.index[0]!r}'
                return _cut(text, 300)
            if 'Index' in name and hasattr(value, '__len__'):
                return _cut(f'{name}(len={len(value)}) ' + repr(list(value[:6])), 300)
            return _cut(f'{name} {value!r}', 80)
        if isinstance(value, dict):
            shown = [f'{k!r}: ' + (brief(v, depth + 1) if depth == 0 else type(v).__name__)
                     for k, v in _first(value.items(), 10)]
            return _cut(f'dict(len {len(value)}) {{' + ', '.join(shown) + (', ...' if len(value) > 10 else '') + '}',
                        600)
        if isinstance(value, (list, tuple, set, frozenset)):
            shown = [brief(v, depth + 1) if depth == 0 else type(v).__name__ for v in _first(value)]
            return _cut(f'{name}(len {len(value)}) [' + ', '.join(shown) + (', ...' if len(value) > 3 else '') + ']',
                        300)
    except Exception:
        pass
    return name


def _wanted(name, value):
    dunder = len(name) > 4 and name.startswith('__') and name.endswith('__')  # a parameter named __ is data
    if dunder or type(value).__name__ in _SKIP_TYPES:
        return False
    return not (callable(value) and not hasattr(value, 'shape') and not hasattr(value, '__len__'))


HINT_ITEMS = 10000  # items of a sequence compared against a mapping


def _module(value):
    return (getattr(type(value), '__module__', '') or '').split('.', 1)[0]


def _keys(value):
    """The keys a subscript looks up: a dict's keys, a DataFrame's columns, a Series' index; else None."""
    if isinstance(value, dict):
        return value
    if _module(value) == 'pandas':
        name = type(value).__name__
        if name == 'DataFrame':
            return value.columns
        if name == 'Series':
            return value.index
    return None


def _has(keys, key):
    try:
        return key in keys
    except Exception:
        return True  # unhashable or incomparable: never reported as missing


KEY_CHARS = 40  # input_fidelity (T1_KEY_HINT_CUT): each key a KeyError hint lists, so one long name cannot fill it
KEY_LIST = 300  # characters of one listed run of keys
KEY_SCAN = 200  # keys considered for the listing


def _short(item):
    text = repr(item)
    return text if len(text) <= KEY_CHARS else text[:KEY_CHARS - 3] + '...'


def _fitted(items, more=False):
    """'[a, b, ...]': items' reprs, each cut to KEY_CHARS, as many as fit in KEY_LIST characters; a trailing
    '...' when any is left out (more: items past those given)."""
    shown, size = [], 2
    for item in items:
        text = _short(item)
        if size + len(text) + 7 > KEY_LIST:
            more = True
            break
        shown.append(text)
        size += len(text) + 2
    return '[' + ', '.join(shown + (['...'] if more else [])) + ']'


def _key_hint(key, on_line):
    cut_keys = os.environ.get('T1_KEY_HINT_CUT') == '1'
    for name, value in on_line:
        keys = _keys(value)
        if keys is None or _has(keys, key):
            continue
        for other, items in on_line:
            if other == name or not (isinstance(items, (list, tuple)) or
                                     (_module(items) == 'pandas' and type(items).__name__ in ('Index', 'Series'))):
                continue
            items = list(itertools.islice(iter(items), HINT_ITEMS))
            if not any(_has([item], key) and item == key for item in items):
                continue
            missing = []
            for item in items:
                if not _has(keys, item) and item not in missing:
                    missing.append(item)
            if cut_keys:
                return (f'Hint: {name} lacks {len(missing)} of the {len(items)} items of {other}: '
                        + _fitted(missing[:KEY_SCAN], len(missing) > KEY_SCAN))
            return (f'Hint: {name} lacks {len(missing)} of the {len(items)} items of {other}: '
                    + _cut(repr(missing[:10]), 300))
        if cut_keys:
            listed = list(itertools.islice(iter(keys), KEY_SCAN))
            close = difflib.get_close_matches(str(key), [str(k) for k in itertools.islice(iter(keys), 500)], n=3)
            return (f'Hint: {name} has no key {_short(key)}; its {len(keys)} keys: '
                    + _fitted(listed, len(keys) > len(listed))
                    + ('; closest: [' + ', '.join(_short(item) for item in close) + ']' if close else ''))
        listed = list(itertools.islice(iter(keys), 12))
        text = f'Hint: {name} has no key {key!r}; its keys: ' + _cut(repr(listed), 300)
        if len(keys) > 12:
            text += ' ...'
        close = difflib.get_close_matches(str(key), [str(k) for k in itertools.islice(iter(keys), 500)], n=3)
        return text + (f'; closest: {close!r}' if close else '')
    return None


def _definition(message, program):
    match = re.match(r'([A-Za-z_][\w.]*)\(\) (?:takes|missing|got)', message)
    if not match:
        return None
    name = match.group(1).rsplit('.', 1)[-1]
    pattern = re.compile(r'\s*(?:async\s+)?def\s+' + re.escape(name) + r'\s*\(')
    for number, line in enumerate(linecache.getlines(program), 1):
        if pattern.match(line):
            return _cut(f'Hint: {name} is defined at line {number}: {line.strip()}')
    return None


def _object_values(name, value):
    if _module(value) == 'numpy' and str(getattr(value, 'dtype', '')) == 'object':
        return f'Hint: {name} holds Python objects (dtype object); convert it with .astype(float) before arithmetic'
    if _module(value) == 'pandas':
        kind = type(value).__name__
        if kind == 'Series' and str(value.dtype) == 'object':
            return f'Hint: {name} holds Python objects (dtype object); convert it with pd.to_numeric before arithmetic'
        if kind == 'DataFrame':
            columns = [c for c, t in itertools.islice(value.dtypes.items(), 200) if str(t) == 'object']
            if columns:
                return (f'Hint: {name} has object columns {_cut(repr(columns[:8]), 200)}; convert them with '
                        'pd.to_numeric or pd.to_datetime before use')
    return None


INDEX_READS = ('index', 'loc', 'truncate', 'asof')  # attributes through which a line compares an index


def _integral(value):
    """An int (not a bool), a NumPy integer scalar or array, a pandas integer Series or Index, or a list or
    tuple whose first item is an int."""
    if isinstance(value, (list, tuple)) and value:
        value = value[0]
    if isinstance(value, int):
        return not isinstance(value, bool)
    return _module(value) in ('numpy', 'pandas') and getattr(getattr(value, 'dtype', None), 'kind', None) in ('i', 'u')


_CHAIN = re.compile(r'[^\W\d]\w*(?:\s*\.\s*[^\W\d]\w*)*')  # name, name.attr, ...


def _unbound(target):
    """An assignment target without what it binds or stores into, which is not compared: `idx`, `i, *rest`,
    the `out` of `out[k]` and `self.out[k]`. The rest is read: what the brackets hold (`out[pos > start]`),
    and a store through .loc and the like, which compares its key with the index (`df.loc[start:, 'c']`)."""
    kept, opened, at, previous = [], [], 0, ''  # opened: True for a subscript or call, False for a group
    while at < len(target):
        char, chain = target[at], _CHAIN.match(target, at)
        if chain:
            text = chain.group()
            if not any(opened) and previous != '.':  # what the target stores into
                attributes = re.split(r'\s*\.\s*', text)
                if not (attributes[-1] in INDEX_READS and target[chain.end():].lstrip().startswith('[')):
                    text = ' '
            kept.append(text)
            at, previous = chain.end(), chain.group()[-1]
            continue
        if char in '\'"':
            end = target.find(char, at + 1)
            end = len(target) if end < 0 else end + 1
        else:
            end = at + 1
            if char in '([{':
                opened.append(previous.isalnum() or previous in ('_', ')', ']'))
            elif char in ')]}' and opened:
                opened.pop()
        kept.append(target[at:end])
        at, previous = end, previous if char.isspace() else char
    return ''.join(kept)


def _operands(line):
    """The failing line without what its assignments bind or store into (see _unbound): the values it reads.
    An `=` with unbalanced brackets or quotes before it is a keyword argument, not an assignment."""
    targets = []
    while True:
        match = re.search(r'(?<![<>!=+\-*/%&|^@:])=(?!=)', line)
        head = line[:match.start()] if match else ''
        if not match or any(head.count(a) != head.count(b) for a, b in ('()', '[]', '{}')) or \
                head.count("'") % 2 or head.count('"') % 2:
            return ' '.join(targets + [line])
        targets.append(_unbound(head))
        line = line[match.end():]


def _attributes(on_line, text):
    """(name.attr, value) for the attribute chains on text that start at a plain object of the line and reach
    a NumPy or pandas value, an int or a list (`self.prices` of `self.prices.index`). Attributes are read from
    instance dictionaries (vars), so no property runs."""
    found = {}
    try:
        for name, value in on_line:
            if _module(value) in ('numpy', 'pandas', 'builtins'):
                continue
            pattern = r'(?<![\w.])' + re.escape(name) + r'((?:\.[^\W\d]\w*)+)'
            for match in itertools.islice(re.finditer(pattern, text), 20):
                chain, current = name, value
                for attribute in match.group(1).split('.')[1:]:
                    try:
                        current = vars(current)[attribute]
                    except Exception:
                        break  # a property, a method, a slot or a missing attribute: not followed
                    chain += '.' + attribute
                    if _module(current) in ('numpy', 'pandas') or isinstance(current, (int, list, tuple)):
                        found.setdefault(chain, current)
                        break
    except Exception:
        pass
    return list(found.items())[:20]


def _reads_integers(name, value, text):
    """Whether text (the failing line's operands) reads integers through name: the value itself, unless it
    only subscripts (`x[name]`); a pandas object's integer index (`name.index`, `name.loc[a:b]`, ...); or an
    integer column the line names (`name.col`, `name['col']`, or 'col' anywhere on the line)."""
    try:
        uses = []  # the text after each use of name
        for match in re.finditer(r'(?<![\w.])' + re.escape(name) + r'\b', text):
            before, after = text[:match.start()], text[match.end():]
            if before.count("'") % 2 or before.count('"') % 2:
                continue  # inside a string
            if before.rstrip().endswith('[') and after.lstrip().startswith(']'):
                continue  # a position that picks an item is not compared
            uses.append(after)
        if not uses:
            return False
        if _integral(value):
            return True
        if _module(value) != 'pandas' or type(value).__name__ not in ('DataFrame', 'Series'):
            return False
        attributes = [m.group(1) for m in (re.match(r'\s*\.\s*([A-Za-z_]\w*)', after) for after in uses) if m]
        if any(attribute in INDEX_READS for attribute in attributes) and _integral(value.index):
            return True
        if type(value).__name__ == 'DataFrame':
            named = attributes + [m.group(2) for m in re.finditer(r'''(['"])(.*?)\1''', text)]
            return any(key in value.columns and _integral(value[key]) for key in named[:20])
    except Exception:
        pass
    return False


DATE_HINT = ('Hint: an integer position or index was compared with a date; parse the date column with '
             'pd.to_datetime, set it as a sorted DatetimeIndex, and compare dates with dates')


def hints(exc_type, exc, message, on_line, program, line=''):
    """Short, rule-based pointers from the exception and the failing line's values; model input only."""
    out = []
    if issubclass(exc_type, KeyError) and getattr(exc, 'args', None):
        text = _key_hint(exc.args[0], on_line)
        if text:
            out.append(text)
    if issubclass(exc_type, TypeError):
        text = _definition(message, program)
        if text:
            out.append(text)
        if "'list'" in message:
            out += [f'Hint: {name} is a list, not an array; arithmetic on it is not elementwise '
                    f'(use np.asarray({name}, dtype=float))' for name, value in on_line if isinstance(value, list)][:2]
        out += [text for text in (_object_values(name, value) for name, value in on_line) if text][:2]
    # Only when the line reads an integer: NumPy's searchsorted on a datetime64[ns] array with a Timestamp
    # compares integer nanoseconds with it, and the message names an int where the program has none.
    if issubclass(exc_type, (TypeError, ValueError)) and re.search(r'Timestamp|datetime', message) and \
            re.search(r"\bint\b|int64|'<'|'>'|comparison", message):
        operands = _operands(line[:1000])  # bounded: the report has TIME_LIMIT in all
        reached = list(on_line) + _attributes(on_line, operands)  # self.prices of self.prices.index, ...
        if any(_reads_integers(name, value, operands) for name, value in reached):
            out.append(DATE_HINT)
    return out[:5]


def report(exc_type, exc, tb, program, lines=None):
    """The report text, or None when no frame of the program is on the traceback. Lines are appended to
    `lines` as they are made, so a report cut short by the time limit keeps what it has."""
    frames = []
    while tb is not None:
        frames.append((tb.tb_frame, tb.tb_lineno))
        tb = tb.tb_next
    own = [i for i, (frame, _) in enumerate(frames) if frame.f_code.co_filename == program]
    if not own:
        return None
    try:
        message = str(exc)
    except Exception:
        message = ''
    lines = [] if lines is None else lines
    lines.append(_cut(f'{getattr(exc_type, "__name__", "Exception")}: {message}', 400))
    inner, number = frames[own[-1]]
    source = linecache.getline(program, number)
    try:
        variables = list(dict(inner.f_locals).items())
    except Exception:
        variables = []
    variables = [(k, v) for k, v in variables if _wanted(k, v)]
    # Names used on the failing line first, then the rest in definition order.
    used = set(re.findall(r'[A-Za-z_][A-Za-z0-9_]*', source))
    variables.sort(key=lambda item: item[0] not in used)
    try:
        lines += hints(exc_type, exc, message, [(k, v) for k, v in variables if k in used], program, source)
    except Exception:
        pass
    lines.append('Program frames (innermost last):')
    shown = own if len(own) <= HEAD_FRAMES + TAIL_FRAMES else own[:HEAD_FRAMES] + [None] + own[-TAIL_FRAMES:]
    for index in shown:
        if index is None:
            lines.append(f'  ... {len(own) - HEAD_FRAMES - TAIL_FRAMES} more program frames ...')
            continue
        frame, number = frames[index]
        lines.append(_cut(f'  line {number}, in {frame.f_code.co_name}: '
                          + linecache.getline(program, number).strip()))
    if own[-1] + 1 < len(frames):
        library = frames[own[-1] + 1][0].f_code
        lines.append(_cut(f'  then inside {os.path.basename(library.co_filename)}, in {library.co_name}'))
    if variables:
        lines.append(f'Local variables in {inner.f_code.co_name} at line {number}:')
        for key, value in variables[:MAX_LOCALS]:
            lines.append(_cut(f'  {key} = {brief(value)}', 700))
        if len(variables) > MAX_LOCALS:
            lines.append(f'  ... {len(variables) - MAX_LOCALS} more')
    text = '\n'.join(lines)
    return text if len(text) <= REPORT_CAP else text[:REPORT_CAP - 4] + '\n...'


TIME_LIMIT = 2.0  # seconds; the program has already failed, so the report must not delay its exit


class _Late(BaseException):
    pass


def _write(path, text):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, 'O_NOFOLLOW', 0), 0o600)
    with os.fdopen(fd, 'w', encoding='utf-8', errors='replace') as stream:
        stream.write(text[:REPORT_CAP])


def install(path, program):
    previous = sys.excepthook

    def hook(exc_type, exc, tb):
        previous(exc_type, exc, tb)  # the traceback reaches the log first, whatever happens below
        lines, alarm = [], None
        try:
            import signal

            def late(signum, frame):
                raise _Late()
            alarm = signal.signal(signal.SIGALRM, late)
            signal.setitimer(signal.ITIMER_REAL, TIME_LIMIT)
        except BaseException:
            alarm = None
        try:
            text = report(exc_type, exc, tb, program, lines)
        except BaseException:  # too slow, or failed: keep the lines already made
            text = '\n'.join(lines + ['  ... (report cut short)']) if len(lines) > 2 else None
        finally:
            if alarm is not None:
                try:
                    import signal
                    signal.setitimer(signal.ITIMER_REAL, 0)
                    signal.signal(signal.SIGALRM, alarm)
                except BaseException:
                    pass
        try:
            if text:
                _write(path, text)
        except BaseException:
            pass

    sys.excepthook = hook


if os.environ.get('T1_CRASH_REPORT') and os.environ.get('T1_PROGRAM'):
    try:
        install(os.environ['T1_CRASH_REPORT'], os.environ['T1_PROGRAM'])
    except Exception:
        pass
