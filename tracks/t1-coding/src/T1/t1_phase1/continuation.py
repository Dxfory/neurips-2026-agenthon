"""Pure helpers for replies cut at the output cap: find the code, keep only verified statements,
and merge the next reply by statement (or by line inside one huge statement). No I/O.

Everything returned as stats is a number, a boolean or a fixed enum; code, names and model text
stay in the returned program and tail, which callers must never export.
"""
from __future__ import annotations

import ast
import builtins
import codeop
import difflib
import io
import re
import symtable
import tokenize
import warnings
import zlib

_FENCE = re.compile(r'^\s*(`{3,}|~{3,})\s*([A-Za-z0-9_.+-]*)\s*$')
_GLUED = re.compile(r'(`{3,}|~{3,})\s*$')
_PY_TAG = re.compile(r'(python3?(\.\d+)*|py3?|.*\.py)?')  # house._python_tag, plus untagged
_THINK = re.compile(r'^\s*<think>.*?(?:</think>|\Z)', re.S | re.I)
_BARE_THINK = re.compile(r'^\s*</think>\s*$', re.M | re.I)
_NOT_START = re.compile(r'^(\)|\]|\}|else\b|elif\b|except\b|finally\b|#)')
_CODEY = re.compile(r'^(\s+\S|import |from |def |class |async |@|if |for |while |with |try:|'
                    r'[A-Za-z_][\w.]*(\[[^\]]*\])?\s*(=|\+=|-=|\(|,|:))')
_COMPOUND = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.If, ast.For, ast.AsyncFor,
             ast.While, ast.With, ast.AsyncWith, ast.Try, ast.TryStar, ast.Match)
_IMPORT = (ast.Import, ast.ImportFrom)


def _compiles(src):
    # compile, not ast.parse: a top-level return or a late __future__ import only fails here.
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            compile(src, '<generated>', 'exec', dont_inherit=True)
        return True
    except (SyntaxError, ValueError, RecursionError, MemoryError, OverflowError):
        return False


def _body(code):
    if not code.strip():
        return []
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return ast.parse(code).body


def statement_count(code):
    try:
        return len(_body(code))
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return 0


def verified_prefix(code, complete):
    """(prefix, tail): the longest run of whole top-level statements that compiles.
    A cut reply (complete=False) never keeps the statement that held the cut."""
    if not code.strip():
        return '', ''
    if complete and _compiles(code):
        return code.rstrip() + '\n', ''
    lines = code.split('\n')
    starts = _statement_starts(lines)
    for cut in reversed(starts):
        src = '\n'.join(lines[:cut])
        if _compiles(src):
            return (src.rstrip() + '\n') if src.strip() else '', '\n'.join(lines[cut:])
    return '', code


def _nl(text):
    return text.replace('\r\n', '\n').replace('\r', '\n')


def _strip_think(text):
    text = _THINK.sub('', _nl(text), count=1)
    close = _BARE_THINK.search(text)
    if close and '<think>' not in text[:close.start()].lower():
        text = text[close.end():]
    return text


def _loose_code(lines):
    """Lines outside fences: skip prose until a code-looking line that starts verified code."""
    codey = [i for i, line in enumerate(lines) if _CODEY.match(line)]
    for i in codey[:10]:
        if statement_count(verified_prefix('\n'.join(lines[i:]), True)[0]):
            return lines[i:]
    return lines[codey[0]:] if codey else []


def _regions(text):
    """(lines, closed, rank) for python/untagged fences and the text before the first fence."""
    lines = text.split('\n')
    i = next((k for k, line in enumerate(lines) if _FENCE.match(line)), len(lines))
    regions = [(_loose_code(lines[:i]), True, 0)]
    while i < len(lines):
        match = _FENCE.match(lines[i])
        if not match:
            i += 1
            continue
        fence, tag, body, closed = match.group(1), match.group(2).lower(), [], False
        i += 1
        while i < len(lines):
            line = lines[i]
            i += 1
            if line.strip().startswith(fence) and _FENCE.match(line):
                closed = True
                break
            glued = _GLUED.search(line)
            if glued and glued.group(1).startswith(fence):
                body.append(line[:glued.start()])
                closed = True
                break
            body.append(line)
        if _PY_TAG.fullmatch(tag):
            regions.append((body, closed, 2 if tag else 1))
    return regions


def _scored(text):
    """[(key, code, closed, rank)] for every region of one reply, in document order."""
    out = []
    for lines, closed, rank in _regions(_strip_think(text or '')):
        code = '\n'.join(lines)
        key = (statement_count(verified_prefix(code, closed)[0]), bool(_CODEY.match(_first_line(code))), rank,
               len(code.strip()))
        out.append((key, code, closed, rank))
    return out


def code_region(text):
    """The code of one reply: the python/untagged fence (an unclosed last one counts) or loose
    code that adds the most verified statements. A stray closer or prose block scores nothing."""
    best, best_key = '', None
    for key, code, _, _ in _scored(text):
        if (key[0] or key[1]) and (best_key is None or key >= best_key):
            best, best_key = code, key
    return best


def _code_regions(text, best, program):
    """A finished reply that splits its code over several fences: every closed fence that compiles
    whole and holds only definitions, constants, known imports or the guard, in document order,
    around the best region. [best] when there is nothing else to add."""
    try:
        known = {ast.dump(n) for n in _body(best) + _body(program) if isinstance(n, _IMPORT)}
        out = []
        for key, code, closed, rank in _scored(text):
            if code == best:
                out += [] if best in out else [best]
            elif closed and rank and key[0] and _compiles(code) and all(
                    _name(n) or _constant(n) or is_main_guard(n) or (isinstance(n, _IMPORT) and ast.dump(n) in known)
                    for n in _body(code)):
                out.append(code)
        return out if best in out else [best]
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return [best]


def is_main_guard(node):
    if not (isinstance(node, ast.If) and isinstance(node.test, ast.Compare) and len(node.test.comparators) == 1):
        return False
    sides = [node.test.left, node.test.comparators[0]]
    return (any(isinstance(s, ast.Name) and s.id == '__name__' for s in sides)
            and any(isinstance(s, ast.Constant) and s.value == '__main__' for s in sides))


def _start(node):
    return min([d.lineno for d in getattr(node, 'decorator_list', [])] + [node.lineno])


def _segment(text, node):
    # Compound statements own whole lines; simple ones may share a line after ';'.
    if isinstance(node, _COMPOUND):
        return '\n'.join(text.split('\n')[_start(node) - 1:node.end_lineno])
    return ast.get_source_segment(text, node)


def _name(node):
    return node.name if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) else None


def _constant(node):
    return (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id.isupper() and node.targets[0].id)


def _import_slot(items):
    slot = 0
    for i, node in enumerate(it[2] for it in items):
        if (isinstance(node, ast.ImportFrom) and node.module == '__future__') or (
                i == 0 and isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)):
            slot = i + 1
        else:
            break
    return slot


_ZERO = {'added': 0, 'replaced': 0, 'overlap': 0, 'import_dup': 0, 'loose': 0, 'stub_kept': 0, 'superseded': 0,
         'restart': False, 'rewrite': False, 'short_restart': False, 'changed': False}


def _dispatch(node):
    return any(isinstance(d, ast.Attribute) and d.attr in ('register', 'overload') or
               isinstance(d, ast.Name) and d.id == 'overload'
               for d in (x.func if isinstance(x, ast.Call) else x for x in getattr(node, 'decorator_list', [])))


def _stub(node):
    """A placeholder body: only pass, ... or a docstring."""
    return all(isinstance(s, ast.Pass) or (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant) and (
        s.value.value is Ellipsis or isinstance(s.value.value, str))) for s in node.body)


def _bound(nodes):
    out = set()
    for n in nodes:
        if _name(n):
            out.add(n.name)
        elif isinstance(n, _IMPORT):
            out.update((a.asname or a.name).split('.')[0] for a in n.names)
        elif isinstance(n, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            out.update(x.id for t in getattr(n, 'targets', [getattr(n, 'target', None)]) for x in ast.walk(t)
                       if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Store))
    return out


def _leading_imports(nodes):
    return next((i for i, n in enumerate(nodes) if not isinstance(n, _IMPORT)), len(nodes))


def merge(program, chunk, mode='continue', complete=False, rest=''):
    """Merge verified statements of the next reply into the verified program.

    The repeated tail of the program (also after a preamble of imports and definitions) and a restart
    from the top (aligned on the first statement after the import block) are skipped; a def/class
    with a known, unique name replaces it unless the new body is a placeholder; a known import is
    dropped; everything else is appended, repeats included. A restart that diverged replaces the
    program only when it is a finished reply in continue mode that binds every name the program binds
    and runs something; otherwise it is aligned with the program by content and only its statements
    past the program's last one are appended. A cut chunk (rest: the statement it was cut in) that
    only restates statements before the program's end adds nothing. mode='repair' also replaces
    UPPER_CASE constants, puts new imports on top and drops loose statements of a patch. The last
    __main__ guard goes last. A statement too deep to analyse leaves the program unchanged."""
    try:
        return _merge(_nl(program), _nl(chunk), mode, complete, _nl(rest or ''))
    except (RecursionError, MemoryError):
        return program, dict(_ZERO)


def _past_program(old, new, a, b):
    """(past, hold) for a diverged restart: new (dumps b) against the program's statements old (dumps
    a), both after their import blocks, aligned by content. new[past:] lies beyond the program's last
    statement; new[hold:] is withheld because which of those statements are new is ambiguous (the
    program's last statements were replaced by more statements, and no def or UPPER constant of the
    same name pins the pairing)."""
    ops = difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes()
    tag, i1, i2, j1, j2 = next(op for op in reversed(ops) if op[2] == len(a) and op[2] > op[1])
    if tag != 'replace':
        return j2, len(b)
    pos = j1 + (i2 - i1)  # positional pairing inside the replaced block
    if j2 <= pos:
        return len(b), len(b)
    key = _key(old[i2 - 1]) or _constant(old[i2 - 1])
    same = next((j for j in range(pos - 1, j2) if key and (_key(new[j]) or _constant(new[j])) == key), None)
    return (same + 1, len(b)) if same is not None else (len(b), pos)


def _preamble(nodes):
    """Leading imports and definitions: what a reply may put before a restated seam."""
    return next((i for i, n in enumerate(nodes) if not (isinstance(n, _IMPORT) or _name(n))), len(nodes))


def _seam(nd, od, start):
    """The longest run of the chunk from start that repeats the program's last statements."""
    return max((k for k in range(1, min(len(nd) - start, len(od)) + 1) if nd[start:start + k] == od[-k:]), default=0)


def _reopens(items, at, rest):
    """The cut statement rest restates items[at] (a partial first line compares as a prefix)."""
    want, header = _first_line(rest).strip(), _first_line(items[at][1]).strip() if at < len(items) else ''
    return bool(want) and (header.startswith(want) if '\n' not in rest.rstrip('\n') else header == want)


_DEFS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def _unfinished_target(text, names):
    """A column-zero def/class header can conflict before its body is parseable. Ignore string tokens."""
    header, asynchronous = False, False
    try:
        for token in tokenize.generate_tokens(io.StringIO(text).readline):
            if header and token.type == tokenize.NAME:
                if token.string in names:
                    return True
            header = token.type == tokenize.NAME and (
                token.start[1] == 0 and token.string in ('def', 'class')
                or asynchronous and token.string == 'def')
            asynchronous = token.type == tokenize.NAME and token.start[1] == 0 and token.string == 'async'
    except (tokenize.TokenError, IndentationError):
        pass
    return False


def repair_target(program, content, near):
    """Three-state guard for bare repair definitions with ambiguous existing names.

    A current failing body line may select one ordinary function in its original lexical scope.
    None as near only detects conflicts (cut replies/patch continuations must not reuse old lines).
    Unique names, whole owners and complete programs remain on the ordinary merge path. Returned
    reasons are fixed text for the existing retry prompt; this does not execute the program.
    """
    rejected = ('rejected', program, 'the bare definition has an ambiguous or unsupported repair scope; '
                'send the complete enclosing def/class block instead')
    try:
        tree = ast.parse(program)
        parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
        definitions = [n for n in ast.walk(tree) if isinstance(n, _DEFS)]
        names = [n.name for n in definitions]
        ambiguous = {name for name in names if names.count(name) > 1}
        regions = []
        for lines, _, _ in _regions(_strip_think(content or '')):
            text = '\n'.join(lines)
            try:
                nodes = ast.parse(text).body
                whole = True
            except (SyntaxError, ValueError):
                if _unfinished_target(text, ambiguous):
                    return rejected
                nodes = _body(verified_prefix(text, False)[0])
                whole = False
            if nodes:
                regions.append((text, nodes, whole))
        conflicts = [n for _, nodes, _ in regions for n in nodes
                     if isinstance(n, _DEFS) and n.name in ambiguous]
        if not conflicts:
            return 'not_applicable', program, ''
        if len(regions) == 1 and regions[0][2]:
            nodes = regions[0][1]
            old_keys = [(type(n), n.name) for n in tree.body if isinstance(n, _DEFS)]
            new_keys = [(type(n), n.name) for n in nodes if isinstance(n, _DEFS)]
            if (old_keys and all(new_keys.count(key) >= old_keys.count(key) for key in old_keys)
                    and _bound(tree.body) <= _bound(nodes) and any(is_main_guard(n) for n in nodes)):
                return 'not_applicable', program, ''
        if (len(regions) != 1 or not regions[0][2] or len(regions[0][1]) != 1
                or type(near) is not int):
            return rejected
        replacement = regions[0][1][0]
        enclosing = [n for n in definitions if n.lineno <= near <= n.end_lineno]
        target = max(enclosing, key=lambda n: n.lineno) if enclosing else None
        if (type(target) is not ast.FunctionDef or type(replacement) is not ast.FunctionDef
                or target.name != replacement.name or not target.lineno < near
                or target.decorator_list or replacement.decorator_list
                or target.body[0].lineno == target.lineno or replacement.body[0].lineno == replacement.lineno
                or any(isinstance(n, ast.Nonlocal) for n in ast.walk(replacement))):
            return rejected

        def scope(node):
            parent = parents[node]
            while parent is not tree and not isinstance(parent, _DEFS):
                parent = parents[parent]
            return parent

        owner, ancestor = scope(target), parents[target]
        while ancestor is not tree:
            if isinstance(ancestor, ast.ClassDef):
                return rejected
            ancestor = parents[ancestor]
        if sum(n.name == target.name and scope(n) is owner for n in definitions) != 1:
            return rejected
        # A source line locates a def, not every later binding of its name. Do not infer through
        # assignments/imports/loop targets or declarations in that same lexical scope. Child
        # function and comprehension locals are separate; a comprehension walrus binds outward.
        for node in ast.walk(owner):
            if node is owner:
                continue
            binds = ((isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del))
                      and node.id == target.name)
                     or (isinstance(node, ast.arg) and node.arg == target.name)
                     or (isinstance(node, ast.alias) and (node.asname or node.name.split('.')[0]) == target.name)
                     or (isinstance(node, ast.ExceptHandler) and node.name == target.name)
                     or (isinstance(node, (ast.Global, ast.Nonlocal)) and target.name in node.names)
                     or (isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name == target.name)
                     or (isinstance(node, ast.MatchMapping) and node.rest == target.name))
            if not binds:
                continue
            child, ancestor = node, parents[node]
            walrus = isinstance(ancestor, ast.NamedExpr)
            while ancestor is not owner:
                if isinstance(ancestor, _DEFS + (ast.Lambda,)):
                    body = ancestor.body if isinstance(ancestor.body, list) else [ancestor.body]
                    # Defaults/decorators/annotations run in the enclosing scope. A nested body
                    # and its ordinary parameters do not; only a header walrus crosses this boundary.
                    if not walrus or child in body:
                        break
                if isinstance(ancestor, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)) and not walrus:
                    break
                child, ancestor = ancestor, parents[ancestor]
            if ancestor is owner:
                return rejected
        if _stub(replacement) or ast.dump(target) == ast.dump(replacement):
            return 'handled', program, ''
        text = _segment(regions[0][0], replacement)
        if not _compiles(text):
            return rejected
        lines = program.split('\n')  # AST physical lines; Unicode separators may occur inside literals.
        prefix = ''.join(line + '\n' for line in lines[:target.lineno - 1])
        suffix = '\n'.join(lines[target.end_lineno:])
        placed = '\n'.join(' ' * target.col_offset + line if line.strip() else line for line in text.split('\n'))
        changed = prefix + placed + ('\n' if target.end_lineno < len(lines) else '') + suffix
        if not _compiles(changed):
            return rejected
        inserted = [n for n in ast.walk(ast.parse(changed)) if isinstance(n, _DEFS)
                    and n.lineno == target.lineno and n.col_offset == target.col_offset]
        if len(inserted) != 1 or ast.dump(inserted[0]) != ast.dump(replacement):
            return rejected
        return 'handled', changed, ''
    except (SyntaxError, ValueError, RecursionError, MemoryError, OverflowError):
        return rejected


def _nested(items, key):
    """(item index, node) for the one def or class named key nested inside a top-level statement of the
    program (a helper inside a function, a method of a class); None when there is none or more than one.
    Nodes are positioned in the item's own text."""
    found = []
    for idx, item in enumerate(items):
        if not isinstance(item[2], _COMPOUND):
            continue
        try:
            tops = ast.parse(item[1]).body
        except (SyntaxError, ValueError):
            continue
        found += [(idx, sub) for top in tops for sub in ast.walk(top)
                  if sub is not top and isinstance(sub, _DEFS) and sub.name == key]
    return found[0] if len(found) == 1 else None


def _replace_nested(owner, node, text):
    """The owner's text with node (and its decorators) replaced by text, indented to node's column."""
    lines = owner.split('\n')
    indent = ' ' * node.col_offset
    body = [indent + line if line.strip() else line for line in text.split('\n')]
    replaced = '\n'.join(lines[:_start(node) - 1] + body + lines[node.end_lineno:])
    try:
        tops = ast.parse(replaced).body
    except (SyntaxError, ValueError):
        return None
    return (replaced, tops[0]) if len(tops) == 1 else None


def _merge(program, chunk, mode, complete, rest=''):
    old, new = _body(program), _body(chunk)
    items = [[ast.dump(n), _segment(program, n), n, True] for n in old]  # True: a slot of the program
    od, nd = [it[0] for it in items], [ast.dump(n) for n in new]
    st = dict(_ZERO)
    k = max((k for k in range(1, min(len(nd), len(od)) + 1) if nd[:k] == od[-k:]), default=0)
    lo, ln = _leading_imports(old), _leading_imports(new)  # import blocks are restated freely
    r = 0
    while lo + r < len(od) and ln + r < len(nd) and nd[ln + r] == od[lo + r]:
        r += 1
    stubs = any(_name(n) and _stub(n) and any(_name(o) == n.name and not _stub(o) for o in old) for n in new)
    restart = bool(r and ln + r >= k and (ln or not lo))
    diverged = restart and lo + r < len(od)
    if (complete and mode == 'continue' and not stubs and len(new) >= len(old) and nd[:len(od)] != od
            and _bound(old) <= _bound(new)
            and any(is_main_guard(n) or isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) for n in new)):
        st.update(added=len(new) - r, overlap=r, restart=True, rewrite=True, changed=True)  # a newer full version
        return chunk.rstrip() + '\n', st
    past = hold = len(nd)  # chunk statements from past on lie beyond the program's end; from hold on, withheld
    if diverged:
        past, hold = (ln + x for x in _past_program(old[lo:], new[ln:], od[lo:], nd[ln:]))
    st['restart'], st['short_restart'] = restart, diverged and (past >= len(nd) or hold < len(nd))
    block = mode == 'continue' and any(d in od for d in nd[:ln])  # a restated import block
    skip = set(range(ln, ln + r) if restart else range(k))
    if mode == 'continue' and not (restart or k):  # the seam overlap, after restated imports or a preamble
        s = _preamble(new)
        for start in dict.fromkeys(([ln] if block else []) + [s]):
            k = _seam(nd, od, start)
            if k:
                skip = set(range(start, start + k))
                break
        j = len(nd) - s
        if not (k or complete) and j and rest.strip():  # cut while restating statements before the seam:
            # two or more statements cut in the program's last one, or a window whose cut statement
            # reopens the next program statement (one statement only when that is the last one).
            def restated(m):
                last, reopen = m + j == len(od) - 1, _reopens(items, m + j, rest)
                return od[m:m + j] == nd[s:] and (j >= 2 and (last or reopen) or last and reopen)
            if any(restated(m) for m in range(len(od) - j)):
                skip, st['short_restart'] = set(range(s, len(nd))), True
    names = [_key(it[2]) for it in items]
    inserted = 0
    for i, (n, d) in enumerate(zip(new, nd)):
        if i in skip:
            st['overlap'] += 1
            continue
        key, text, const = _key(n), _segment(chunk, n), _constant(n)
        idx = next((j for j, it in enumerate(items) if it[3] and _key(it[2]) == key), None) if (
            key and names.count(key) == 1) else None
        nested = (_nested(items, key) if idx is None and key and mode == 'repair' and isinstance(n, _DEFS)
                  and names.count(key) == 0 else None)
        if nested is not None:  # a patch restating a helper or method: replace it where it is
            j, sub = nested
            if _stub(n) and not _stub(sub):
                st['stub_kept'] += 1
                continue
            placed = _replace_nested(items[j][1], sub, text)
            if placed is not None:
                dump = ast.dump(placed[1])
                if dump == items[j][0]:
                    st['overlap'] += 1
                else:
                    items[j] = [dump, placed[0], placed[1], True]
                    st['replaced'] += 1
                continue
        if idx is not None:
            if items[idx][0] == d:
                st['overlap'] += 1
            elif _stub(n) and not _stub(items[idx][2]):
                st['stub_kept'] += 1
            else:
                items[idx] = [d, text, n, True]
                st['replaced'] += 1
            continue
        if isinstance(n, _IMPORT) and d in od:
            st['import_dup'] += 1
            continue
        if const and (mode == 'repair' or diverged):
            idx = next((j for j, it in enumerate(items) if it[3] and _constant(it[2]) == const), None)
            if idx is not None:
                if items[idx][0] != d:
                    items[idx] = [d, text, n, True]
                    st['replaced'] += 1
                continue
        if i >= hold:
            st['superseded'] += 1  # past an ambiguous seam: nothing is taken, what is pending stays
            continue
        if isinstance(n, _IMPORT) and (mode == 'repair' or (block and i < ln)):
            at = _import_slot(items) + inserted if mode == 'repair' else lo + inserted
            items.insert(at, [d, text, n, False])
            inserted += 1
            st['added'] += 1
            continue
        if diverged and not (_name(n) or const or isinstance(n, _IMPORT)) and i < past:
            st['superseded'] += 1  # aligned before the program's last statement: the program holds it
            continue
        if mode == 'repair' and not (st['restart'] or _name(n) or const or is_main_guard(n)):
            st['loose'] += 1
            continue
        items.append([d, text, n, False])
        st['added'] += 1
    guards = [it for it in items if is_main_guard(it[2])]
    items = [it for it in items if not is_main_guard(it[2])] + guards[-1:]
    out = '\n\n'.join(it[1] for it in items).rstrip() + '\n' if items else ''
    if not _compiles(out):  # defensive: never hand back a program worse than the input
        return program, dict(st, added=0, replaced=0, changed=False)
    st['changed'] = [it[0] for it in items] != od
    return (out, st) if st['changed'] else (program, st)


def _key(node):
    """The name a def/class is replaced by; None for '_' and dispatch registrations."""
    name = _name(node)
    return name if name and name != '_' and not _dispatch(node) else None


def _first_line(text):
    return next((line for line in text.split('\n') if line.strip()), '')


def _complete_lines(tail):
    return tail.split('\n')[:-1] if tail else []


def _code_lines(lines):
    return [i for i, line in enumerate(lines) if line.strip() and not line.lstrip().startswith('#')]


def _function_tail_state(tail):
    """Static context of complete physical lines in one unfinished function. No guessed closures.
    A compilable body prefix is evidence of syntax only; the whole function remains pending."""
    source = '\n'.join(_complete_lines(tail)) + '\n'
    if not re.match(r'^(?:async )?def \w+\(', _first_line(source)):
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            codeop.compile_command(source, '<pending>', 'exec')
        ends, brackets, logical_start, indent = [], [], True, 0
        try:
            for token in tokenize.generate_tokens(io.StringIO(source).readline):
                if token.type == tokenize.NEWLINE:
                    ends.append(token.end[0])
                    logical_start = True
                elif token.type not in (tokenize.INDENT, tokenize.DEDENT, tokenize.NL,
                                       tokenize.COMMENT, tokenize.ENDMARKER):
                    if logical_start:
                        indent, logical_start = token.start[1], False
                    if token.type == tokenize.OP:
                        if token.string in '([{':
                            brackets.append(token.string)
                        elif token.string in ')]}':
                            if not brackets or '([{'.index(brackets.pop()) != ')]}'.index(token.string):
                                return None
        except tokenize.TokenError as exc:
            if exc.args[0] not in ('EOF in multi-line statement', 'unexpected EOF in multi-line statement'):
                return None
        lines = source.splitlines(True)
        for end in reversed(ends):
            prefix = ''.join(lines[:end])
            if not _compiles(prefix):
                continue
            tree = ast.parse(prefix)
            if len(tree.body) != 1 or not isinstance(tree.body[0], (ast.FunctionDef, ast.AsyncFunctionDef)):
                return None
            function = tree.body[0]
            operations = {ast.dump(n) for n in ast.walk(function) if isinstance(n, ast.stmt)
                          and not isinstance(n, (*_COMPOUND, ast.Pass))
                          and not (isinstance(n, ast.Expr) and isinstance(n.value, (ast.Constant, ast.JoinedStr)))}
            if operations:
                return {'name': function.name, 'operations': operations, 'block_indent': indent,
                        'open_brackets': len(brackets)}
            return None
    except (SyntaxError, ValueError, TypeError, OverflowError, RecursionError, MemoryError):
        return None
    return None


def function_tail_context(tail):
    """Bounded numeric context, or None when this pending function is not safely identified."""
    state = _function_tail_state(tail)
    return ({'body_statements': len(state['operations']), 'block_indent': state['block_indent'],
             'open_brackets': state['open_brackets']} if state else None)


def _pending_tokens(source):
    """Keep literal tokens byte-exact; physical '#' or whitespace inside strings is data."""
    result = []
    try:
        for token in tokenize.generate_tokens(io.StringIO(source).readline):
            if token.type not in (tokenize.ENCODING, tokenize.ENDMARKER, tokenize.INDENT, tokenize.DEDENT,
                                  tokenize.NEWLINE, tokenize.NL, tokenize.COMMENT):
                result.append((token.type, token.string))
    except tokenize.TokenError as exc:
        if exc.args[0] not in ('EOF in multi-line statement', 'unexpected EOF in multi-line statement'):
            return None
    except (SyntaxError, ValueError):
        return None
    return result


def _tail_extension(program, tail, selected):
    """Only the selected, unchanged pending function may earn syntactic body progress."""
    before = _function_tail_state(tail)
    if not before:
        return True, False
    out, rest = selected
    retained = (program + '\n'.join(_complete_lines(tail))).splitlines()
    combined = (out + rest).splitlines()
    old_lines = [retained[i].rstrip() for i in _code_lines(retained)]
    new_lines = [combined[i].rstrip() for i in _code_lines(combined)]
    if new_lines[:len(old_lines)] != old_lines:
        return False, False
    old_tokens = _pending_tokens(program + '\n'.join(_complete_lines(tail)) + '\n')
    new_tokens = _pending_tokens(out + rest)
    if old_tokens is None or new_tokens is None or new_tokens[:len(old_tokens)] != old_tokens:
        return False, False
    # A top-level restart must not close the old function then silently redefine it.
    nodes = _body(out)[statement_count(program):]
    if any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == before['name'] for n in nodes):
        if _unfinished_target(rest, {before['name']}):
            return False, False
        return True, False  # ordinary completed-statement progress already accounts for this
    after = _function_tail_state(rest)
    if not after or after['name'] != before['name']:
        return False, False
    return True, bool(after['operations'] - before['operations'])


def line_join(program, tail, chunk, complete=False):
    """Continue inside one huge statement: program + the tail's complete lines + chunk, minus the
    longest run the chunk repeats at the seam (compared on code lines: merge re-spaces the program
    and drops comments). A chunk that restates the pending statement from its header replaces it;
    a cut chunk found whole inside what is pending adds nothing. Then the verified prefix."""
    program, tail, chunk = _nl(program), _nl(tail), _nl(chunk)
    head = program.split('\n')[:-1]
    base = head + _complete_lines(tail)
    while base and not base[-1].strip():
        base.pop()
    lines = chunk.split('\n')
    while lines and not lines[0].strip():
        lines.pop(0)
    bi, ci = _code_lines(base), _code_lines(lines)
    if not ci:  # an empty, prose or comment-only reply never vouches for what is pending
        return program, tail, {'overlap_lines': 0, 'tail_lines_dropped': 0}
    top, strip = [base[i].rstrip() for i in bi], [lines[i].rstrip() for i in ci]
    k = max((k for k in range(1, min(len(top), len(strip)) + 1) if strip[:k] == top[-k:]), default=0)
    drop, header = ci[k - 1] + 1 if k else 0, _first_line(tail).rstrip()
    whole = strip if complete else strip[:-1]
    if not k and strip and header[:1].strip() and strip[0] == header:
        base = head
    elif not k and len(whole) >= 2 and any(top[i:i + len(whole)] == whole for i in range(len(top))):
        drop = len(lines)
    prefix, rest = verified_prefix('\n'.join(base + lines[drop:]), complete)
    partial = 1 if tail and not tail.endswith('\n') else 0
    return prefix, rest, {'overlap_lines': k, 'tail_lines_dropped': partial}


def _repeats(program, joined, tail=''):
    """A line join that re-runs code: its first new statements restate the program's last ones, it
    closes a definition and then opens the same one again, it repeats a whole compound statement, or
    it reopens the header of the pending statement after closing it."""
    old, body = _body(program), _body(joined)
    new = body[len(old):]
    od, nd = [ast.dump(n) for n in old], [ast.dump(n) for n in new]
    if any(nd[:j] == od[-j:] for j in range(1, min(len(nd), len(od)) + 1)):
        return True
    keys = [_key(n) for n in old + new]
    if any(_key(n) and keys.count(_key(n)) > 1 for n in new):
        return True
    lines, closed, seen = joined.split('\n'), _first_line(tail).strip(), set(od)
    for i, (n, d) in enumerate(zip(new, nd)):
        if isinstance(n, _COMPOUND) and not _name(n) and (
                d in seen or i and closed and lines[n.lineno - 1].strip() == closed):
            return True
        seen.add(d)
    return False


def absorb(program, tail, content, finish, mode='continue'):
    """Fold one reply into (program, tail). mode: 'continue', 'line' or 'repair'.
    Candidates: S (statement merge) and L (line join, for indented replies or line mode);
    more verified statements wins, ties go to S (to L in line mode). L never wins by re-running
    code. A statement too deep to analyse changes nothing."""
    try:
        return _absorb(_nl(program), _nl(tail), content, finish, mode)
    except (RecursionError, MemoryError):
        count = statement_count(program)
        return program, tail, dict(_ZERO, candidate='S', overlap_lines=0, tail_lines_dropped=0,
                                   verified_stmts_total=count, stmts_added=0, pending_lines=len(_complete_lines(tail)))


def _absorb(program, tail, content, finish, mode):
    complete, count, kind = finish == 'stop', statement_count(program), 'repair' if mode == 'repair' else 'continue'
    code = code_region(content)
    chunk, chunk_tail = verified_prefix(code, complete)
    merged, st = merge(program, chunk, kind, complete, chunk_tail)
    regions = _code_regions(content, code, program) if complete and mode != 'line' and not chunk_tail else [code]
    if len(regions) > 1:  # a finished reply split over several fences: merge each, in order
        merged, st = program, dict(_ZERO)
        for region in regions:
            merged, one = merge(merged, verified_prefix(region, True)[0], kind, complete)
            st = {key: (st[key] + one[key]) if type(st[key]) is int else (st[key] or one[key]) for key in st}
        st['changed'] = merged != program
    stats = dict(st, candidate='S', overlap_lines=0, tail_lines_dropped=0)
    new_tail = chunk_tail
    if ((program.strip() or tail.strip()) and not st['rewrite']
            and (st['short_restart'] or statement_count(merged) == count and not st['replaced'])):
        # no progress, or a restart cut before the program's end: keep what is pending,
        # but retain a first partial statement when there is no prior state to protect.
        restated = chunk_tail and _first_line(chunk_tail) == _first_line(tail)  # unless restated longer
        new_tail = chunk_tail if restated and len(chunk_tail) > len(tail) else tail
    best = (merged, new_tail)
    # A restart from the top has no line overlap at the seam: joining would repeat the program.
    if (mode == 'line' or _first_line(code)[:1].isspace()) and not st['restart']:
        joined, rest, jst = line_join(program, tail, code, complete)
        n, m = statement_count(joined), statement_count(best[0])
        if (n > m or (n == m and mode == 'line' and len(joined + rest) > len(best[0] + best[1]))) and not (
                n > count and _repeats(program, joined, tail)):
            best = (joined, rest)
            stats.update(jst, candidate='L', added=max(0, n - count), overlap=0, replaced=0, restart=False,
                         rewrite=False, changed=joined != program)
    # Only line mode shows the tail to the model, so only there does a bare closer vouch for it.
    closer = _FENCE.match(_first_line(_strip_think(content or '')))
    if (complete and mode == 'line' and tail and closer and not closer.group(2) and not code.strip()
            and not chunk.strip()):
        whole, rest = verified_prefix(program + '\n'.join(_complete_lines(tail)), True)
        if not rest and statement_count(whole) > statement_count(best[0]):
            best = (whole, '')
            stats.update(candidate='S', changed=True)
    tail_progress = False
    if mode == 'line' and _function_tail_state(tail):
        accepted, tail_progress = _tail_extension(program, tail, best)
        if not accepted:
            best = program, tail
            stats = dict(_ZERO, candidate='S', overlap_lines=0, tail_lines_dropped=0)
    total = statement_count(best[0])
    stats.update(verified_stmts_total=total, stmts_added=total - count,
                 pending_lines=len(_complete_lines(best[1])), tail_progress=tail_progress)
    return best[0], best[1], stats


def _numpy_missing_names(program):
    """Legacy program-wide eligibility for automatic NumPy name fixes, not lexical resolution."""
    try:
        tree = ast.parse(program)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return []
    defined, called = set(dir(builtins)), set()
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined.add(n.name)
        elif isinstance(n, ast.arg):
            defined.add(n.arg)
        elif isinstance(n, _IMPORT):
            if any(a.name == '*' for a in n.names):
                return []
            defined.update((a.asname or a.name).split('.')[0] for a in n.names)
        elif isinstance(n, ast.Name) and not isinstance(n.ctx, ast.Load):
            defined.add(n.id)
        elif isinstance(n, ast.ExceptHandler) and n.name:
            defined.add(n.name)
        elif isinstance(n, (ast.Global, ast.Nonlocal)):
            defined.update(n.names)
        elif isinstance(n, (ast.MatchAs, ast.MatchStar)) and n.name:
            defined.add(n.name)
        elif isinstance(n, ast.MatchMapping) and n.rest:
            defined.add(n.rest)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
            called.add(n.func.id)
    return sorted(called - defined)


def _missing_call_sites(program):
    """Bare calls lacking a visible lexical binding; not definite-assignment or runtime proof.

    Stars, unsupported scopes and ambiguous symbol tables produce no findings.
    Lambda/comprehension bodies are conservative: their locals never bind the outside.
    """
    if not isinstance(program, str):
        return []
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            tree = ast.parse(program)
            compile(tree, '<call-scope>', 'exec', dont_inherit=True)
            table = symtable.symtable(program, '<call-scope>', 'exec')
        nodes = list(ast.walk(tree))
        if any(isinstance(n, ast.ImportFrom) and any(a.name == '*' for a in n.names) for n in nodes):
            return []
        deferred_annotations = any(isinstance(n, ast.ImportFrom) and n.module == '__future__'
                                   and any(a.name == 'annotations' for a in n.names) for n in tree.body)
        bound = set(dir(builtins)) | set(_bound_in(tree))
        # Explicit global writers may initialize names at runtime. Keep the old
        # conservative declaration policy, and do not guess whether they ran.
        bound.update(name for n in nodes if isinstance(n, ast.Global) for name in n.names)
        # A walrus in a module-level default/comprehension binds the module,
        # but one in a function body (including its nested defaults) does not.
        todo = list(tree.body)
        while todo:
            node = todo.pop()
            if isinstance(node, _DEFS):
                todo.extend(node.decorator_list)
                if isinstance(node, ast.ClassDef):
                    todo.extend(node.bases + [item.value for item in node.keywords])
                else:
                    todo.extend(node.args.defaults + [d for d in node.args.kw_defaults if d is not None])
                continue
            if isinstance(node, ast.Lambda):
                todo.extend(node.args.defaults + [d for d in node.args.kw_defaults if d is not None])
                continue
            if isinstance(node, ast.AnnAssign) and deferred_annotations:
                todo.extend([node.target] + ([node.value] if node.value is not None else []))
                continue
            if isinstance(node, ast.NamedExpr) and isinstance(node.target, ast.Name):
                bound.add(node.target.id)
            todo.extend(ast.iter_child_nodes(node))
        bound.update(n.rest for n in _own_nodes(tree) if isinstance(n, ast.MatchMapping) and n.rest)
        found = []

        def visit(node, symbols):
            if node is None:
                return
            if isinstance(node, _DEFS):
                outer = list(node.decorator_list)
                if isinstance(node, ast.ClassDef):
                    outer += node.bases + [item.value for item in node.keywords]
                    kind = 'class'
                else:
                    outer += node.args.defaults + [d for d in node.args.kw_defaults if d is not None]
                    kind = 'function'
                for expr in outer:
                    visit(expr, symbols)
                matches = [t for t in symbols.get_children() if t.get_name() == node.name
                           and t.get_lineno() == node.lineno and t.get_type() == kind]
                if len(matches) == 1 and not getattr(node, 'type_params', ()):
                    for stmt in node.body:
                        visit(stmt, matches[0])
                return
            if isinstance(node, ast.Lambda):
                for expr in node.args.defaults + [d for d in node.args.kw_defaults if d is not None]:
                    visit(expr, symbols)
                return
            if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
                # Python 3.13 inlines several comprehension symbol tables. Their
                # first iterable is evaluated outside; do not guess inner scope.
                visit(node.generators[0].iter, symbols)
                return
            if isinstance(node, ast.AnnAssign):
                visit(node.target, symbols)
                visit(node.value, symbols)
                if node.simple and symbols.get_type() in ('module', 'class') and not deferred_annotations:
                    visit(node.annotation, symbols)
                return  # function-local annotations do not execute
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                try:
                    symbol = symbols.lookup(node.func.id)
                except KeyError:
                    symbol = None
                if symbol is not None and symbol.is_global() and node.func.id not in bound:
                    found.append(node)
            for child in ast.iter_child_nodes(node):
                visit(child, symbols)

        visit(tree, table)
        return found
    except (SyntaxError, ValueError, TypeError, RecursionError, MemoryError, OverflowError):
        return []


def missing_calls(program):
    """Sorted bare-call names with no visible lexical binding; [] is not a safety guarantee."""
    return sorted({node.func.id for node in _missing_call_sites(program)})


def ensure_guard(program):
    """(program, appended): call a top-level main() that takes no required argument and is never called."""
    try:
        body = _body(program)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return program, False
    main = next((n for n in body if isinstance(n, ast.FunctionDef) and n.name == 'main'), None)
    if main is None or len(main.args.posonlyargs + main.args.args) > len(main.args.defaults) or any(
            d is None for d in main.args.kw_defaults):
        return program, False
    for n in body:
        if n is not main and any(isinstance(c, ast.Call) and isinstance(c.func, ast.Name) and c.func.id == 'main'
                                 for c in ast.walk(n)):
            return program, False
    return program.rstrip() + '\n\n\nif __name__ == "__main__":\n    main()\n', True


def anchors(program, tail, limit=120):
    """(last_header, anchor, last_line): first line of the last verified statement, first line of
    the dropped tail, and the last complete line before the cut. Model-facing text only."""
    body = _body(program)
    last = program.split('\n')[_start(body[-1]) - 1] if body else ''
    anchor = _first_line(tail)
    lines = [line for line in program.split('\n')[:-1] + _complete_lines(tail) if line.strip()]
    return last.strip()[:limit], anchor.strip()[:limit], (lines[-1] if lines else '').rstrip()[:limit]


def reply_metrics(content, completion_tokens, reasoning_chars=0):
    """Size and shape of one reply, numbers only."""
    content = content if isinstance(content, str) else ''
    answer = _strip_think(content)
    code = code_region(content)
    rest = answer.replace(code, '', 1) if code else answer
    tail = content[-8192:].encode('utf-8', errors='replace')
    tokens = completion_tokens if type(completion_tokens) is int and completion_tokens > 0 else 0
    return {'think_chars': len(_nl(content)) - len(answer),
            'prose_chars': sum(len(line.strip()) for line in rest.split('\n') if not _FENCE.match(line)),
            'code_chars': len(code), 'code_lines': sum(1 for line in code.split('\n') if line.strip()),
            'chars_per_token': round(len(content) / tokens, 3) if tokens else 0.0,
            'repeat_ratio': round(len(tail) / len(zlib.compress(tail)), 3) if tail else 0.0,
            'reasoning_chars': reasoning_chars if type(reasoning_chars) is int and reasoning_chars > 0 else 0}


# A line standing for code the hunk leaves out: `...`, or a comment with an ellipsis or a "kept code" phrase.
_ELIDED = re.compile(r'^\s*(\.\.\.|\u2026|#.*(\.\.\.|\u2026)|#.*\b(unchanged|rest|remain\w*|same|existing|'
                     r'previous|before|above|below|snip\w*|etc|omitted|stays?|kept|keep)\b)', re.I)


def _anchor(text):
    text = text.strip()
    return len(text) >= 4 and any(c.isalnum() for c in text)


def _indent(text):
    return len(text) - len(text.lstrip())


def _resembles(new, old):
    return difflib.SequenceMatcher(None, new.strip(), old.strip()).ratio() >= 0.6


def splice_hunk(program, hunk, near=None):
    """Place an indented repair reply: changed lines framed by unchanged context lines. The hunk's first
    and last lines must match program lines by stripped text and relative indentation (or the next ones in,
    when the edge line resembles the program line it then replaces one for one); they bound the region the
    hunk replaces, re-indented to the program. The region stays inside the hunk's block and its size is the
    one closest to the hunk's; several placements go to the one nearest `near` (the failing line, 1-based).
    A tie, elided code, a region much longer than the hunk, or a result that does not compile gives None."""
    lines = [line.rstrip() for line in _nl(hunk).split('\n')]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    if len(lines) < 2 or any(_ELIDED.match(line) for line in lines):
        return None
    source = _nl(program).split('\n')
    stripped = [line.strip() for line in source]
    size = len(lines)
    shallowest = min(_indent(line) for line in lines if line.strip())
    for a, b in ((0, size - 1), (0, size - 2), (1, size - 1), (1, size - 2)):
        if not (0 <= a < b < size) or not (_anchor(lines[a]) and _anchor(lines[b])):
            continue
        regions = []
        for i, text in enumerate(stripped):
            if text != lines[a].strip():
                continue
            floor = _indent(source[i]) + shallowest - _indent(lines[a])  # the hunk's block, in program indentation
            fits = []
            for k in range(i + 1, min(len(source), i + 2 * (b - a) + 21)):
                if (stripped[k] != lines[b].strip()
                        or _indent(source[k]) - _indent(source[i]) != _indent(lines[b]) - _indent(lines[a])):
                    continue
                start, end = i - a, k + size - 1 - b
                if start < 0 or end >= len(source):
                    continue
                span = end - start + 1
                if span > max(2 * size, size + 10) or size > 3 * span + 10:
                    continue
                if (a and not _resembles(lines[0], source[start])) or (
                        b < size - 1 and not _resembles(lines[-1], source[end])):
                    continue
                if any(source[x].strip() and _indent(source[x]) < floor for x in range(start, end + 1)):
                    continue  # the region would leave the block the hunk is in
                fits.append((abs(span - size), start, end))
            if fits:
                _, start, end = min(fits)
                distance = (0 if near is None or start < near <= end + 1
                            else min(abs(near - start - 1), abs(near - end - 1)))
                regions.append((distance, start, end))
        if not regions:
            continue
        regions.sort()
        if len(regions) > 1 and regions[0][0] == regions[1][0]:
            return None  # ambiguous placement
        _, start, end = regions[0]
        placed = _reindent(lines, lines[a], source[start + a])
        if placed is None:
            return None
        merged = '\n'.join(source[:start] + placed + source[end + 1:])
        return merged if merged != _nl(program) and _compiles(merged) else None
    return None


ANCHOR_REACH = 3  # lines from the failing line within which a context-free hunk is placed
CONTEXT_REACH = 10  # lines from the failing line within which a context line that recurs is taken


def _base_names(target):
    """The names an assignment target binds or mutates: x, x[i], x.a -> x; a tuple gives each."""
    while isinstance(target, (ast.Attribute, ast.Subscript, ast.Starred)):
        target = target.value
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.Tuple, ast.List)):
        return set().union(*(_base_names(item) for item in target.elts)) if target.elts else set()
    return set()


def _assigned(text):
    """The names one line of code sets: what it assigns, or the object whose method a bare call runs (x.sort()
    sets x). Empty for any other line or one that does not parse alone."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            body = ast.parse(text.strip()).body
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return set()
    if len(body) != 1:
        return set()
    node = body[0]
    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute):
        return _base_names(node.value.func.value)
    if not isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
        return set()
    return set().union(*(_base_names(t) for t in (node.targets if isinstance(node, ast.Assign) else [node.target])))


STANDS, AMBIGUOUS = 2, 1  # _correspondence levels


def _correspondence(new, old):
    """How a changed line stands for a program line: STANDS when it differs from it and resembles it; AMBIGUOUS
    when it only assigns the same names and shares some of its text (x = x.a() and x = x.b(), common in a chain
    of steps), which cannot tell a replacement from a new line beside it; 0 otherwise."""
    new, old = new.strip(), old.strip()
    if new == old or not old or old.startswith('#'):
        return 0
    ratio = difflib.SequenceMatcher(None, new, old).ratio()
    if ratio >= 0.6:
        return STANDS
    names = _assigned(new)
    return AMBIGUOUS if names and names == _assigned(old) and ratio >= 0.3 else 0


def _placed(source, start, end, lines, reference, program):
    """The program with source lines start..end replaced by lines, re-indented by reference (a hunk line and the
    program line it takes the place of); None when that fails, changes nothing or does not compile."""
    placed = _reindent(lines, *reference)
    if placed is None:
        return None
    merged = '\n'.join(source[:start] + placed + source[end + 1:])
    return merged if merged != _nl(program) and _compiles(merged) else None


def anchor_hunk(program, hunk, near=None):
    """Place a repair hunk of one or two lines that splice_hunk cannot frame (it needs a context line at each
    end). Two lines, one of them a program line (a context line: the only one with its text, or the one nearest
    the failing line `near`): the other line replaces the program line beside it when it stands for that line
    (_correspondence), is inserted beside the context line when it has nothing in common with it, and is not
    placed when that is ambiguous (only the same names set). Lines that match no program line replace, one for
    one, program lines they stand for within ANCHOR_REACH lines of `near` (the nearest such place; without
    `near` nothing is placed). Relative indentation must agree. A tie, elided code, no change or a result that
    does not compile gives None, and the caller's retry asks for whole defs."""
    lines = [line.rstrip() for line in _nl(hunk).split('\n') if line.strip()]
    if not 1 <= len(lines) <= 2 or any(_ELIDED.match(line) for line in lines):
        return None
    source = _nl(program).split('\n')
    stripped = [line.strip() for line in source]
    matched = [[i for i, text in enumerate(stripped) if text == line.strip()] for line in lines]
    if len(lines) == 2 and bool(matched[0]) != bool(matched[1]):
        return _beside_context(program, source, lines, matched, near)
    if any(matched) or type(near) is not int or not 1 <= near <= len(source):
        return None  # a line that changes nothing, or both lines already in the program (splice_hunk's case)
    at, size, fits = near - 1, len(lines), []
    shape = [_indent(line) - _indent(lines[0]) for line in lines]
    for start in range(max(0, at - ANCHOR_REACH - size + 1), min(len(source) - size, at + ANCHOR_REACH) + 1):
        window = source[start:start + size]
        if ([_indent(line) - _indent(window[0]) for line in window] != shape
                or not all(_correspondence(new, old) == STANDS for new, old in zip(lines, window))):
            continue
        end = start + size - 1
        distance = 0 if start <= at <= end else min(abs(at - start), abs(at - end))
        if distance <= ANCHOR_REACH:
            fits.append((distance, start))
    fits.sort()
    if not fits or len(fits) > 1 and fits[0][0] == fits[1][0]:
        return None
    start = fits[0][1]
    return _placed(source, start, start + size - 1, lines, (lines[0], source[start]), program)


def _beside_context(program, source, lines, matched, near):
    """anchor_hunk for two lines of which exactly one (the context) is a program line."""
    c = 0 if matched[0] else 1
    new = lines[1 - c]
    if not _anchor(lines[c]):
        return None
    spots = matched[c]
    if len(spots) > 1:  # a recurring context line: the one nearest the failing line, if that is clear
        if type(near) is not int:
            return None
        spots = sorted((abs(i + 1 - near), i) for i in spots)
        if spots[0][0] == spots[1][0] or spots[0][0] > CONTEXT_REACH:
            return None
        spots = [spots[0][1]]
    i = spots[0]
    want = _indent(source[i]) + _indent(new) - _indent(lines[c])
    if want < 0 or '\t' in source[i][:_indent(source[i])] + new[:_indent(new)]:
        return None
    step = 1 if c == 0 else -1
    j = i + step
    while 0 <= j < len(source) and not source[j].strip():
        j += step
    line = ' ' * want + new.strip()
    level = _correspondence(new, source[j]) if 0 <= j < len(source) and _indent(source[j]) == want else 0
    if level == AMBIGUOUS:
        return None  # a replacement or a new line: guessing could delete a statement the program needs
    if level == STANDS:
        start, end = j, j  # it replaces the line beside the context line
    else:
        start = end = i + 1 if c == 0 else i  # it is inserted beside the context line
        end -= 1
    merged = '\n'.join(source[:start] + [line] + source[end + 1:])
    return merged if merged != _nl(program) and _compiles(merged) else None


def _reindent(lines, reference, target):
    have = reference[:len(reference) - len(reference.lstrip())]
    want = target[:len(target) - len(target.lstrip())]
    if have == want:
        return list(lines)
    if '\t' in have + want:
        return None
    shift, out = len(want) - len(have), []
    for line in lines:
        indent = len(line) - len(line.lstrip())
        if not line.strip():
            out.append('')
        elif '\t' in line[:indent]:
            return None
        elif shift >= 0:
            out.append(' ' * shift + line)
        elif indent >= -shift:
            out.append(line[-shift:])
        else:
            return None
    return out


LINT_LIMIT = 5  # findings reported by call_lint


def _lintable(tree):
    """{name: def} of the program's top-level functions whose name nothing else binds (no other def, class,
    assignment, parameter, import or handler name) and that no decorator wraps: a call by that name runs that
    def as written."""
    defs, bound = {}, {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            defs.setdefault(node.name, []).append(node)
    for node in ast.walk(tree):
        names = []
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names = [node.name]
        elif isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load):
            names = [node.id]
        elif isinstance(node, ast.arg):
            names = [node.arg]
        elif isinstance(node, _IMPORT):
            names = [(a.asname or a.name).split('.')[0] for a in node.names]
        elif isinstance(node, (ast.ExceptHandler, ast.MatchAs, ast.MatchStar)) and node.name:
            names = [node.name]
        for name in names:
            bound[name] = bound.get(name, 0) + 1
    return {name: nodes[0] for name, nodes in defs.items()
            if len(nodes) == 1 and bound.get(name) == 1 and not nodes[0].decorator_list}


def _return_width(function):
    """The tuple width every return of a function gives, when all of its returns give a tuple display of one
    width (no starred item) and it is no generator; None otherwise."""
    widths, todo = set(), list(function.body)
    while todo:
        node = todo.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        if isinstance(node, (ast.Yield, ast.YieldFrom)):
            return None
        if isinstance(node, ast.Return):
            if not isinstance(node.value, ast.Tuple) or any(isinstance(e, ast.Starred) for e in node.value.elts):
                return None
            widths.add(len(node.value.elts))
        todo.extend(ast.iter_child_nodes(node))
    return widths.pop() if len(widths) == 1 else None


def _arity(call, function):
    """Why a call cannot bind to the function's parameters, or None: too many positional arguments, a keyword
    it does not take or gets twice, or a required parameter left out."""
    if any(isinstance(a, ast.Starred) for a in call.args) or any(k.arg is None for k in call.keywords):
        return None
    spec = function.args
    positional = [a.arg for a in spec.posonlyargs + spec.args]
    given, name = len(call.args), function.name
    if given > len(positional) and spec.vararg is None:
        return f'passes {given} positional arguments, but `{name}` takes {len(positional)}'
    by_keyword = set(positional[len(spec.posonlyargs):]) | {a.arg for a in spec.kwonlyargs}
    keywords = [k.arg for k in call.keywords]
    for key in keywords:
        if key not in by_keyword and spec.kwarg is None:
            return f'passes `{key}=`, which `{name}` does not take'
        if key in positional[:given]:
            return f'passes `{key}` twice'
    required = len(positional) - len(spec.defaults)
    missing = [p for i, p in enumerate(positional[:required]) if i >= given and p not in keywords]
    missing += [a.arg for a, d in zip(spec.kwonlyargs, spec.kw_defaults) if d is None and a.arg not in keywords]
    if missing:
        return f'leaves out {", ".join(f"`{p}`" for p in missing)}, which `{name}` requires'
    return None


_CONDITIONAL = (ast.IfExp, ast.BoolOp, ast.Lambda, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
_SIMPLE = (ast.Expr, ast.Assign, ast.AugAssign, ast.AnnAssign, ast.Return)


def _own_nodes(node):
    """The nodes under node that run as part of it: nested defs, classes and lambdas are left out."""
    todo = list(ast.iter_child_nodes(node))
    while todo:
        item = todo.pop()
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        yield item
        todo.extend(ast.iter_child_nodes(item))


def _unconditional_call(statement, name):
    """The first call `name(...)` a simple statement always makes: not inside a conditional expression, a
    boolean operator, a lambda or a comprehension. None when there is none."""
    todo = [statement]
    while todo:
        item = todo.pop(0)
        if isinstance(item, ast.Call) and isinstance(item.func, ast.Name) and item.func.id == name:
            return item
        if not isinstance(item, _CONDITIONAL):
            todo.extend(ast.iter_child_nodes(item))
    return None


def endless_recursion(function):
    """The call of function to itself that runs on every call before any return, so that the recursion never
    ends: a simple statement at the top level of its body (not inside if/for/while/try/with) that always calls it
    by name, with no earlier statement of the body holding a return or ending in a raise. None for a generator,
    and whenever a base case may come first."""
    if any(isinstance(node, (ast.Yield, ast.YieldFrom)) for node in _own_nodes(function)):
        return None
    for statement in function.body:
        if isinstance(statement, _DEFS):  # defining a helper runs none of its body
            continue
        if isinstance(statement, _SIMPLE):
            call = _unconditional_call(statement, function.name)
            if call is not None:
                return call
        if isinstance(statement, (ast.Raise, ast.Return)) or any(
                isinstance(node, ast.Return) for node in _own_nodes(statement)):
            return None
    return None


def call_lint(program, limit=LINT_LIMIT, recursion=False):
    """Calls of the program's own top-level functions (_lintable) that cannot work as written, found without
    running it: a call whose arguments cannot bind to the def's parameters, and an unpacking `a, b = f(...)` of
    a width that differs from the one tuple width every return of f gives. With recursion, also a function that
    calls itself on every call before any return (endless_recursion). ['line N: `source` - why'], at most
    limit, in line order. Model-facing advice only; [] when the program does not compile (a def whose parameters
    repeat parses, but a finding about its calls would point the repair away from the def)."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            tree = ast.parse(program)
            compile(tree, '<program>', 'exec', dont_inherit=True)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return []
    try:
        functions = _lintable(tree)
        lines, found = program.split('\n'), []
        for node in ast.walk(tree):
            problem = None
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in functions:
                problem = _arity(node, functions[node.func.id])
            elif (isinstance(node, ast.Assign) and len(node.targets) == 1
                  and isinstance(node.targets[0], (ast.Tuple, ast.List))
                  and not any(isinstance(e, ast.Starred) for e in node.targets[0].elts)
                  and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
                  and node.value.func.id in functions):
                function = functions[node.value.func.id]
                width, want = _return_width(function), len(node.targets[0].elts)
                if width is not None and width != want:
                    problem = (f'unpacks {want} values, but every return of `{function.name}` (line '
                               f'{function.lineno}) gives {width}')
            if problem:
                found.append((node.lineno, f'line {node.lineno}: `{lines[node.lineno - 1].strip()[:120]}` - {problem}'))
        for function in functions.values() if recursion else ():
            call = endless_recursion(function)
            if call is not None:
                found.append((call.lineno, f'line {call.lineno}: `{lines[call.lineno - 1].strip()[:120]}` - '
                                           f'`{function.name}` calls itself here on every call, before any return, '
                                           'so the recursion never ends'))
    except RecursionError:
        return []
    return [text for _, text in sorted(set(found))][:limit]


def new_lint(before, after, recursion=False):
    """The call_lint findings of after that before does not have, compared without their line numbers (a patch
    moves lines): what a patch broke. [] when after does not compile."""
    def key(item):
        return item.split(': ', 1)[-1]
    known = {key(item) for item in call_lint(before, limit=None, recursion=recursion)}
    return [item for item in call_lint(after, limit=None, recursion=recursion) if key(item) not in known]


def _global_reads(program):
    """A narrow {(function, name): diagnostic} for straight-line function-body reads.

    This is static advice, not a proof of NameError: module bindings are a superset,
    and external reflection is not generally decidable. Uncertain control flow and
    nested scopes are deliberately not traversed. No generated code is executed.
    """
    if not isinstance(program, str):
        return {}
    try:
        tree = ast.parse(program)
        compile(tree, '<scope-check>', 'exec')
        table = symtable.symtable(program, '<scope-check>', 'exec')
        nodes = list(ast.walk(tree))
        dynamic = set(_DYNAMIC) | {'getattr', 'setattr', 'delattr', '__import__', '__builtins__',
                                 '__globals__', '__dict__', 'f_globals', 'f_locals', 'modules'}
        # References as well as calls matter: ns = globals; ns()[name] = value.
        if any((isinstance(n, ast.Name) and n.id in dynamic)
               or (isinstance(n, ast.Attribute) and n.attr in dynamic)
               or (isinstance(n, ast.ImportFrom)
                   and any(a.name == '*' or a.name in dynamic for a in n.names)) for n in nodes):
            return {}
        bound = {s.get_name() for s in table.get_symbols()
                 if s.is_assigned() or s.is_imported() or s.is_namespace()}
        bound.update(name for n in nodes if isinstance(n, (ast.Global, ast.Nonlocal)) for name in n.names)
        bound.update(dir(builtins))
        children = {(t.get_name(), t.get_lineno()): t for t in table.get_children()}
        found = {}

        def reads(node):
            if node is None or isinstance(node, _CONDITIONAL + (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                return
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                yield node
            for child in ast.iter_child_nodes(node):
                yield from reads(child)

        for name, function in _lintable(tree).items():
            symbols = children.get((name, function.lineno))
            if symbols is None or any(isinstance(n, (ast.Yield, ast.YieldFrom)) for n in _own_nodes(function)):
                continue
            free = {s.get_name() for s in symbols.get_symbols()
                    if s.is_referenced() and s.is_global() and not s.is_declared_global()} - bound
            for stmt in function.body:
                if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
                                     ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal, ast.Pass)):
                    continue
                if not isinstance(stmt, _SIMPLE + (ast.Raise,)):
                    break  # branches, loops, try/except, with/suppress: do not guess reachability
                # Local variable annotations are metadata, not evaluated expressions.
                runtime = stmt.value if isinstance(stmt, ast.AnnAssign) else stmt
                for node in reads(runtime):
                    if node.id in free and not (node.id.startswith('__') and node.id.endswith('__')):
                        key = (name, node.id)
                        found.setdefault(key, f'line {node.lineno}: function `{name}` reads `{node.id}` '
                                             'with no statically visible binding')
                if isinstance(stmt, (ast.Return, ast.Raise)):
                    break
        return found
    except (SyntaxError, ValueError, TypeError, RecursionError, MemoryError):
        return {}


def new_global_reads(before, after):
    """New static global-read risks, keyed by (function, name), independent of line/expression changes.

    Only unique undecorated top-level functions' own straight-line expressions are
    considered. [] means no finding in this limited analysis, not verified scope safety.
    """
    known = _global_reads(before)
    return [text for key, text in _global_reads(after).items() if key not in known]


def unverified_global_reads(before, observed, after):
    """Previously observed risks that a retry merely moves outside this analysis.

    Keep these separate from diagnosed reads. Only deletion of the read, an explicit
    parameter, or a direct module binding clears an observation after coverage is
    lost; this is not a general proof of data-flow correctness.
    """
    known = _global_reads(before)
    tracked = {key: text for key, text in _global_reads(observed).items() if key not in known}
    diagnosed = _global_reads(after)
    try:
        tree = ast.parse(after)
        compile(tree, '<scope-check>', 'exec')
        table = symtable.symtable(after, '<scope-check>', 'exec')
        direct = set()
        for stmt in tree.body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                direct.add(stmt.name)
            elif isinstance(stmt, (ast.Import, ast.ImportFrom)):
                direct.update(a.asname or (a.name.split('.')[0] if isinstance(stmt, ast.Import) else a.name)
                              for a in stmt.names if a.name != '*')
            elif isinstance(stmt, (ast.Assign, ast.AnnAssign)):
                if isinstance(stmt, ast.AnnAssign) and stmt.value is None:
                    continue
                targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
                direct.update(n.id for target in targets for n in ast.walk(target)
                              if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store))
        unknown = []
        for (function, name), text in tracked.items():
            if (function, name) in diagnosed:
                continue
            matches = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                       and n.name == function]
            # A vanished top-level function may have moved to a nested/class scope.
            region = matches[0] if len(matches) == 1 else tree
            present = any(isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id == name
                          for n in ast.walk(region))
            parameter = False
            if len(matches) == 1:
                args = matches[0].args
                parameters = args.posonlyargs + args.args + args.kwonlyargs
                parameters += [a for a in (args.vararg, args.kwarg) if a is not None]
                parameter = any(a.arg == name for a in parameters)
                if parameter:
                    scopes = [t for t in table.get_children()
                              if t.get_name() == function and t.get_lineno() == matches[0].lineno]
                    parameter = bool(scopes)
                    while scopes:
                        child = scopes.pop()
                        if any(s.get_name() == name and s.is_referenced() and s.is_global()
                               for s in child.get_symbols()):
                            parameter = False  # an inner explicit global does not capture the parameter
                            break
                        scopes.extend(child.get_children())
            if present and name not in direct and not parameter:
                unknown.append(text)
        return unknown
    except (SyntaxError, ValueError, TypeError, RecursionError, MemoryError):
        return list(tracked.values())


def _missing_parameters(call, function):
    """The required parameters a call leaves out, when leaving them out is its only problem (_arity); []
    otherwise."""
    problem = _arity(call, function)
    if not problem or not problem.startswith('leaves out'):
        return []
    spec = function.args
    positional = [a.arg for a in spec.posonlyargs + spec.args]
    keywords = {k.arg for k in call.keywords}
    required = len(positional) - len(spec.defaults)
    missing = [p for i, p in enumerate(positional[:required]) if i >= len(call.args) and p not in keywords]
    if any(p in [a.arg for a in spec.posonlyargs] for p in missing):
        return []  # a positional-only parameter cannot be passed by keyword
    return missing + [a.arg for a, d in zip(spec.kwonlyargs, spec.kw_defaults) if d is None and a.arg not in keywords]


def _bound_in(scope):
    """{name: [lines]} of the names a function (or the module) binds in its own code: parameters (line 0),
    assignments, loop and with targets, imports, handlers, and the defs and classes it holds (not what those,
    lambdas or comprehensions bind inside themselves)."""
    found = {}

    def add(name, line):
        found.setdefault(name, []).append(line)
    if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
        spec = scope.args
        for a in spec.posonlyargs + spec.args + spec.kwonlyargs + [x for x in (spec.vararg, spec.kwarg) if x]:
            add(a.arg, 0)
    todo = list(scope.body)
    while todo:
        node = todo.pop()
        if isinstance(node, _DEFS):
            add(node.name, node.lineno)
            continue
        if isinstance(node, (ast.Lambda, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            continue
        if isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load):
            add(node.id, node.lineno)
        elif isinstance(node, _IMPORT):
            for a in node.names:
                add((a.asname or a.name).split('.')[0], node.lineno)
        elif isinstance(node, (ast.ExceptHandler, ast.MatchAs, ast.MatchStar)) and node.name:
            add(node.name, node.lineno)
        todo.extend(ast.iter_child_nodes(node))
    return found


def bind_missing(program, skip=()):
    """(program, calls changed): each call of the program's own function (_lintable) whose only problem is that it
    leaves out required parameters gets `p=p` for each, when the scope of the call already has p: a parameter of
    the enclosing function, a name that function binds on an earlier line (a name it binds only later would be
    unbound), or, when it never binds p, a name the module binds at its top level (on an earlier line for a call
    at module level). A call is changed only when all its missing parameters are there. The top-level defs named
    in skip are left as written: no call of them or in them changes. The program unchanged when it does not parse
    or the result does not compile."""
    try:
        tree = ast.parse(program)
        functions = {name: node for name, node in _lintable(tree).items() if name not in skip}
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return program, 0
    module = _bound_in(tree)
    edits = []

    def visit(node, scope, local):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node is not tree or child.name not in skip:
                    visit(child, child, _bound_in(child))
                continue
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Name) and child.func.id in functions:
                missing = _missing_parameters(child, functions[child.func.id])

                def has(name):
                    if scope is None:
                        return any(line < child.lineno for line in module.get(name, ()))
                    if name in local:
                        return any(line < child.lineno for line in local[name])
                    return name in module
                if missing and all(has(name) for name in missing):
                    edits.append((child, missing))
            visit(child, scope, local)
    try:
        visit(tree, None, {})
    except RecursionError:
        return program, 0
    data, starts = program.encode(), _line_starts(program)
    fixed = _inserted(data, [_passing(data, starts, call, missing) for call, missing in edits])
    if fixed == program or not _compiles(fixed):
        return program, 0
    return fixed, len(edits)


def _line_starts(program):
    """The byte offset where each line of program starts (ast columns count bytes)."""
    starts = [0]
    for line in program.split('\n')[:-1]:
        starts.append(starts[-1] + len(line.encode()) + 1)
    return starts


def _passing(data, starts, call, names):
    """(offset, text) that makes a call pass `name=name` for each of names, the text going before its closing
    parenthesis; None when the call does not end in one."""
    close = starts[call.end_lineno - 1] + call.end_col_offset - 1
    if data[close:close + 1] != b')':
        return None
    text = ', '.join(f'{name}={name}' for name in names)
    if call.args or call.keywords:
        text = (' ' if data[:close].rstrip().endswith(b',') else ', ') + text
    return close, text


def _inserted(data, edits):
    """data (bytes) with each (offset, text) of edits inserted, None edits left out, as text."""
    for at, text in sorted((edit for edit in edits if edit), reverse=True):
        data = data[:at] + text.encode() + data[at:]
    return data.decode()


_DYNAMIC = ('globals', 'locals', 'vars', 'dir', 'exec', 'eval')  # calls that bind or look up names by string
LIFT_PASSES = 3  # a name that callers of callers bind reaches its reader one level per pass


def _binds(statement):
    """The names a statement binds at its own level whenever it completes: an assignment (not an annotation
    alone), an import, a def or class, a `with ... as`. What an if, a loop or a try binds inside may not have run."""
    if isinstance(statement, _DEFS):
        return {statement.name}
    if isinstance(statement, _IMPORT):
        return {(a.asname or a.name).split('.')[0] for a in statement.names}
    if isinstance(statement, ast.Assign):
        targets = statement.targets
    elif isinstance(statement, (ast.AugAssign, ast.AnnAssign)) and statement.value is not None:
        targets = [statement.target]
    elif isinstance(statement, (ast.With, ast.AsyncWith)):
        targets = [item.optional_vars for item in statement.items if item.optional_vars is not None]
    else:
        return set()
    return {node.id for target in targets for node in ast.walk(target)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)}


def _always_bound(scope, name, node, parents):
    """Whether name is bound in a function (scope) whenever node in it runs: a parameter, a name a statement before
    the one holding node binds (_binds), in its block or a block around it, or the target of a `for` or `with`
    whose body holds node."""
    spec = scope.args
    if name in {a.arg for a in spec.posonlyargs + spec.args + spec.kwonlyargs + [spec.vararg, spec.kwarg] if a}:
        return True
    child = node
    while child is not scope:
        parent = parents[id(child)]
        for field in ('body', 'orelse', 'finalbody') if isinstance(child, ast.stmt) else ():
            block = getattr(parent, field, None)
            if isinstance(block, list) and any(item is child for item in block):
                if any(name in _binds(item) for item in block[:[id(i) for i in block].index(id(child))]):
                    return True
                if field == 'body' and isinstance(parent, (ast.For, ast.AsyncFor)) and name in {
                        n.id for n in ast.walk(parent.target) if isinstance(n, ast.Name)}:
                    return True
                if field == 'body' and isinstance(parent, (ast.With, ast.AsyncWith)) and name in _binds(parent):
                    return True
        child = parent
    return False


def _free_reads(table, skip):
    """The names a function's symbol table (and those of the scopes inside it) reads from the module, less skip."""
    found, todo = set(), [table]
    while todo:
        table = todo.pop()
        found |= {s.get_name() for s in table.get_symbols()
                  if s.is_referenced() and s.is_global() and not s.is_declared_global()}
        todo.extend(table.get_children())
    return found - skip


def lift_free_names(program, skip=()):
    """(program, parameters added): a top-level function (_lintable, not named in skip) that is called, and only
    ever by name, and that reads a name nothing gives it (not bound in it, not by the module at any line, not
    declared global anywhere, no builtin or dunder), gets that name as a new parameter, and each of its calls passes
    `name=name`, when every call sits in the function itself or in a function where the name is always bound when
    the call runs (_always_bound). Such a read can only raise NameError. Unlike bind_missing this changes calls
    that work, so a binding that may not have run is not enough, and a call in a def named in skip stops the
    change. A new parameter is keyword-only after defaults or *args. Nothing changes for a function with **kwargs,
    a call of it with ** or a handler of NameError in it; the program is unchanged when it does not compile,
    star-imports or looks names up by string (_DYNAMIC: `'x' in locals()`), or when the result does not compile.
    Repeats up to LIFT_PASSES times, so a name reaches a reader through a caller that got it as a parameter."""
    total = 0
    for _ in range(LIFT_PASSES):
        program, added = _lift_once(program, skip)
        total += added
        if not added:
            break
    return program, total


def _lift_once(program, skip):
    if not _compiles(program):
        return program, 0
    try:
        tree = ast.parse(program)
        table = symtable.symtable(program, '<program>', 'exec')
        nodes = list(ast.walk(tree))
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return program, 0
    if any(isinstance(n, ast.ImportFrom) and any(a.name == '*' for a in n.names) or isinstance(n, ast.Call) and (
            isinstance(n.func, ast.Name) and n.func.id in _DYNAMIC) for n in nodes):
        return program, 0
    parents = {id(child): node for node in nodes for child in ast.iter_child_nodes(node)}
    unavailable = set(_bound_in(tree)) | set(skip) | {name for n in nodes if isinstance(n, (ast.Global, ast.Nonlocal))
                                                      for name in n.names}
    tables = {(t.get_name(), t.get_lineno()): t for t in table.get_children() if t.get_type() == 'function'}
    data, starts = program.encode(), _line_starts(program)
    tokens, edits, added = _tokens(program), [], 0
    for name, function in _lintable(tree).items():
        spec = function.args
        if name in skip or spec.kwarg or (name, function.lineno) not in tables or any(
                isinstance(n, ast.ExceptHandler) and n.type is not None and 'NameError' in {
                    t.id for t in ast.walk(n.type) if isinstance(t, ast.Name)} for n in ast.walk(function)):
            continue  # a handler of NameError expects the name to be missing
        free = {n for n in _free_reads(tables[name, function.lineno], unavailable)
                if not hasattr(builtins, n) and not (n.startswith('__') and n.endswith('__'))}
        uses = [n for n in nodes if isinstance(n, ast.Name) and n.id == name and isinstance(n.ctx, ast.Load)]
        calls = [parents[id(n)] for n in uses if isinstance(parents[id(n)], ast.Call) and parents[id(n)].func is n]
        if not free or not calls or len(calls) != len(uses) or any(k.arg is None or k.arg in free for c in calls
                                                      for k in c.keywords):
            continue
        for call in calls:
            scope = parents[id(call)]
            while not isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Module)):
                scope = parents[id(scope)]
            outer = scope
            while parents.get(id(outer)) is not tree and outer is not tree:
                outer = parents[id(outer)]
            if scope is tree or getattr(outer, 'name', None) in skip:
                free = set()
            elif scope is not function:
                free = {n for n in free if _always_bound(scope, n, call, parents)}
        reads = sorted((n.lineno, n.col_offset, n.id) for n in ast.walk(function)
                       if isinstance(n, ast.Name) and n.id in free)
        names = list(dict.fromkeys(n for _, _, n in reads))
        header = _header_close(tokens, function)
        if not names or header is None:
            continue
        (row, col), previous = header
        text = ', '.join(names)
        if not (spec.vararg or spec.kwonlyargs) and spec.defaults:
            text = '*, ' + text
        if previous != '(':
            text = (' ' if previous == ',' else ', ') + text
        passing = [_passing(data, starts, call, names) for call in calls]
        if None in passing:
            continue
        edits += [(starts[row - 1] + len(program.split('\n')[row - 1][:col].encode()), text)] + passing
        added += len(names)
    fixed = _inserted(data, edits)
    if fixed == program or not _compiles(fixed):
        return program, 0
    return fixed, added


def _header_close(tokens, function):
    """((row, column), the token before it) of the parenthesis that closes a def's parameters, from its tokens
    (_tokens); None when they cannot be matched."""
    for i, token in enumerate(tokens):
        if (token.start[0] >= function.lineno and token.string == 'def' and i + 2 < len(tokens)
                and tokens[i + 1].string == function.name and tokens[i + 2].string == '('):
            depth = 0
            for j in range(i + 2, len(tokens)):
                if tokens[j].type == tokenize.OP and tokens[j].string in ('(', '[', '{'):
                    depth += 1
                elif tokens[j].type == tokenize.OP and tokens[j].string in (')', ']', '}'):
                    depth -= 1
                    if not depth:
                        return tokens[j].start, tokens[j - 1].string
            return None
    return None


def _raises_unwritten(statement):
    """`raise NotImplementedError` or `raise NotImplementedError(...)`."""
    exc = statement.exc if isinstance(statement, ast.Raise) else None
    exc = exc.func if isinstance(exc, ast.Call) else exc
    return isinstance(exc, ast.Name) and exc.id == 'NotImplementedError'


def placeholder(node):
    """A function whose body is only a docstring, `...`, `pass` or `raise NotImplementedError`: a stub."""
    body = list(node.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(
            body[0].value.value, str):
        body = body[1:]
    return len(body) <= 1 and all(
        isinstance(s, ast.Pass) or _raises_unwritten(s)
        or (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant) and s.value.value is Ellipsis) for s in body)


def stub_defs(program):
    """[(name, def line, first docstring line)] of the program's top-level functions that are stubs, in order."""
    try:
        body = _body(program)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return []
    lines, found = program.split('\n'), []
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and placeholder(node):
            doc = (ast.get_docstring(node) or '').strip().split('\n')[0]
            found.append((node.name, lines[node.lineno - 1].strip()[:160], doc[:160]))
    return found


SALVAGE_REACH = 8  # following statements a block that does not compile alone may take in (a decorator, say)


def _statement_starts(lines):
    """Indexes of the lines that may begin a top-level statement (as verified_prefix cuts)."""
    return [i for i, line in enumerate(lines) if line[:1] and not line[:1].isspace()
            and not _NOT_START.match(line) and not (i and lines[i - 1].rstrip().endswith('\\'))]


def salvage_blocks(code, complete):
    """(blocks, dropped): the top-level statements of a region that compile, each on its own (or joined with at
    most SALVAGE_REACH following ones, as a decorator needs its def), in order; a statement that compiles in no
    such way is dropped. Unlike verified_prefix, one invalid statement in the middle does not take the complete
    statements after it. A cut region (complete=False) never keeps its last statement."""
    lines = code.split('\n')
    starts = _statement_starts(lines)
    bounds = starts + [len(lines)]
    limit = len(starts) if complete else max(0, len(starts) - 1)
    blocks, dropped, i = [], 0, 0
    while i < limit:
        for j in range(i + 1, min(limit, i + SALVAGE_REACH) + 1):
            source = '\n'.join(lines[bounds[i]:bounds[j]])
            if _compiles(source):
                blocks.append(source.rstrip() + '\n')
                i = j
                break
        else:
            dropped += 1
            i += 1
    return blocks, dropped


def first_invalid(content):
    """'SyntaxError: <message> in the statement that begins `<first line>`' for the first top-level statement of
    a reply's code region that compiles in no way salvage_blocks tries, or ''. Model-facing text only."""
    code = code_region(content or '')
    lines = code.split('\n')
    starts = _statement_starts(lines)
    bounds = starts + [len(lines)]
    for i in range(len(starts)):
        if any(_compiles('\n'.join(lines[bounds[i]:bounds[j]]))
               for j in range(i + 1, min(len(starts), i + SALVAGE_REACH) + 1)):
            continue
        try:
            compile('\n'.join(lines[bounds[i]:bounds[i + 1]]), '<reply>', 'exec', dont_inherit=True)
        except SyntaxError as exc:
            return f'SyntaxError: {exc.msg} in the statement that begins `{lines[bounds[i]].strip()[:120]}`'
        except (ValueError, RecursionError, MemoryError, OverflowError):
            pass
        return f'invalid Python in the statement that begins `{lines[bounds[i]].strip()[:120]}`'
    return ''


def absorb_defs(program, content, finish):
    """Merge the definitions of one reply into the program as a patch would (merge in repair mode): every python
    or untagged fence in order, a cut last one up to its verified prefix, so each complete def replaces or adds
    its function and a cut one adds nothing. A region that starts indented (a body without its def line) is
    skipped. A region with an invalid statement in the middle keeps every other statement that compiles
    (salvage_blocks) when that keeps more than its verified prefix: stats 'salvaged' counts the statements kept
    past it and 'dropped' those left out. (program, merge stats summed)."""
    total = dict(_ZERO, salvaged=0, dropped=0)
    try:
        regions = [(code, closed) for _, code, closed, rank in _scored(content) if rank]
        if not regions:
            regions = [(code_region(content), finish == 'stop')]
        for index, (code, closed) in enumerate(regions):
            complete = closed or (finish == 'stop' and index == len(regions) - 1)
            chunk, rest = verified_prefix(code, complete)
            if _first_line(code)[:1].isspace():
                continue
            if rest.strip():  # something did not verify: keep the complete statements after an invalid one too
                blocks, dropped = salvage_blocks(code, complete)
                joined = '\n\n'.join(blocks)
                count = statement_count(chunk)
                if blocks and statement_count(joined) > count and _compiles(joined):
                    total['salvaged'] += statement_count(joined) - count
                    total['dropped'] += dropped
                    chunk = joined
            if not chunk.strip():
                continue
            program, st = merge(program, chunk, 'repair', complete)
            total = {key: (total[key] + st.get(key, 0)) if type(total[key]) is int else (total[key] or st[key])
                     for key in total}
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        pass
    return program, total


def keep_complete_defs(before, after):
    """(program, restored): after, with each top-level function or class that before defines completely (not a
    placeholder) and after replaced by different code put back as before had it. A skeleton reply may restate
    verified functions; it never replaces a complete one. after unchanged when there is nothing to restore or
    a restore does not merge."""
    try:
        old = {_name(n): (ast.dump(n), _segment(before, n)) for n in _body(before)
               if isinstance(n, _DEFS) and not placeholder_node(n)}
        new = {_name(n): ast.dump(n) for n in _body(after) if isinstance(n, _DEFS)}
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return after, 0
    restored = 0
    for name, (dump, text) in old.items():
        if name in new and new[name] != dump:
            merged, st = merge(after, text, 'repair', True)
            if st['replaced']:
                after, restored = merged, restored + 1
    return after, restored


def completion_binding_conflicts(before, after):
    """Original module binding statements/order changed by completion, after defs are restored.

    Include direct stores and module-evaluated header/comprehension walruses. A stub
    may be completed. This is a mechanical source contract, not proof that arbitrary
    calls or dynamic mutation leave runtime globals unchanged.
    """
    def bindings(text):
        found, ordered = {}, []
        body = _body(text)
        deferred = any(isinstance(n, ast.ImportFrom) and n.module == '__future__'
                       and any(a.name == 'annotations' for a in n.names) for n in body)
        for node in body:
            names = set(_bound_in(ast.Module(body=[node], type_ignores=[])))
            if placeholder_node(node):
                names.discard(node.name)  # body may be filled, but its defaults still evaluate
            todo = [node]
            while todo:
                item = todo.pop()
                if isinstance(item, _DEFS + (ast.Lambda,)):
                    todo.extend(getattr(item, 'decorator_list', []))
                    if isinstance(item, ast.ClassDef):
                        todo.extend(item.bases + [k.value for k in item.keywords])
                    else:
                        args = item.args
                        todo.extend(args.defaults + [d for d in args.kw_defaults if d is not None])
                        if not deferred:
                            todo.extend(a.annotation for a in args.posonlyargs + args.args + args.kwonlyargs
                                        + [a for a in (args.vararg, args.kwarg) if a] if a.annotation is not None)
                            if getattr(item, 'returns', None) is not None:
                                todo.append(item.returns)
                    continue
                if isinstance(item, ast.AnnAssign) and deferred:
                    todo.extend([item.target] + ([item.value] if item.value is not None else []))
                    continue
                if isinstance(item, ast.NamedExpr) and isinstance(item.target, ast.Name):
                    names.add(item.target.id)
                todo.extend(ast.iter_child_nodes(item))
            dump = ast.dump(node)
            for name in names:
                found.setdefault(name, []).append(dump)
            if names:
                ordered.append((names, dump))
        return found, ordered
    try:
        old, old_order = bindings(before)
        new, new_order = bindings(after)
        conflicts = sorted(name for name in old if new.get(name) != old[name])
        if not conflicts:
            projected = [(names & old.keys(), dump) for names, dump in new_order if names & old.keys()]
            if projected != old_order:
                conflicts = sorted(old)  # unchanged bindings reordered across default/header evaluation
        return conflicts
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return ['<unverified module bindings>']


def placeholder_node(node):
    """A def whose body is a placeholder (placeholder()); a class never is."""
    return isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and placeholder(node)


def raise_stubs(program, message='{name} is not written yet: write its complete body'):
    """The program with each top-level stub's body (after its docstring) replaced by a raise of
    NotImplementedError naming it, so running an unfinished program fails at the stub rather than on the None it
    would return. The program unchanged when it does not parse or the result would not compile."""
    try:
        body = _body(program)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return program
    lines = program.split('\n')
    for node in reversed(body):
        if not (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and placeholder(node)):
            continue
        statement = 'raise NotImplementedError(' + repr(message.format(name=node.name)) + ')'
        rest = node.body[1:] if ast.get_docstring(node) is not None else node.body
        first = node.body[0]
        if first.lineno == node.lineno:  # `def f(): ...` on one line
            if rest:
                line = lines[first.lineno - 1].encode()
                lines[first.lineno - 1] = (line[:first.col_offset] + statement.encode()
                                           + line[node.body[-1].end_col_offset:]).decode()
            continue
        indent = ' ' * first.col_offset
        if rest:
            lines[rest[0].lineno - 1:node.end_lineno] = [indent + statement]
        else:  # a docstring alone
            lines[node.end_lineno:node.end_lineno] = [indent + statement]
    out = '\n'.join(lines)
    return out if _compiles(out) else program


def stub_broken_def(code, name, stub):
    """code that does not compile with each top-level def of name (and its decorators) replaced by stub, when that
    alone makes it compile; code unchanged otherwise. Code that does not compile cannot be parsed, so a def is
    found by its lines: from its column-0 `def name(` line to the line before the next one that may begin a
    top-level statement (_statement_starts), or to the end of the code when it is cut inside the def. A column-0
    line inside a string ends the block early; the compile check then rejects the change."""
    if 'def ' + name not in code or _compiles(code):
        return code
    lines = code.split('\n')
    header = re.compile(r'(async\s+)?def\s+' + re.escape(name) + r'\s*\(')
    starts = _statement_starts(lines)
    blocks = [(i, next((j for j in starts if j > i), len(lines))) for i in starts if header.match(lines[i])]
    if not blocks:
        return code
    stub_lines = stub.rstrip('\n').split('\n')
    for begin, end in reversed(blocks):
        while begin and lines[begin - 1].startswith('@'):
            begin -= 1
        while end > begin + 1 and not lines[end - 1].strip():  # the blank lines before what follows stay
            end -= 1
        lines[begin:end] = stub_lines
    out = '\n'.join(lines)
    return out if _compiles(out) else code


# A raise the program wrote to check its own results (guard_raise): its if-test compares values or tests for NaN,
# and neither the test nor the message is about something missing or about invalid input.
_ORDER = (ast.Lt, ast.Gt, ast.LtE, ast.GtE)
_VALUE_TESTS = ('isnan', 'isfinite', 'isinf', 'isclose', 'allclose')
_EXISTENCE = ('len', 'hasattr', 'isinstance', 'exists', 'isfile', 'isdir', 'is_file', 'is_dir', 'empty', 'columns',
              'keys', 'get')
_MISSING_WORDS = re.compile(r'missing|not (?:be )?found|no such|not exist|does not exist|required|not available|'
                            r'unavailable|cannot find|could not find|not present|empty|no data|no rows', re.I)
_INVALID_WORDS = re.compile(r'invalid|malformed|must (?:not )?be|non-?positive|non-?negative|not positive|'
                            r'\bbefore\b|\bafter\b|out of order', re.I)
WARNING_CALL = "print('warning:', repr({exc}), file=__import__('sys').stderr)"
# in a for loop the warning is printed once (keyed by the raise's line), not on every pass
WARNING_ONCE = ("__import__('sys').__dict__.setdefault('t1_warned', set()).__contains__({line}) or "
                "__import__('sys').t1_warned.add({line}) or " + WARNING_CALL)


def _called_name(node):
    node = node.func if isinstance(node, ast.Call) else node
    return node.attr if isinstance(node, ast.Attribute) else node.id if isinstance(node, ast.Name) else ''


def _value_check(test):
    """Whether an if-test checks computed values: an order comparison or a NaN/finiteness test, and nothing that
    asks whether an input exists (membership, length, emptiness, a path, a key or an attribute)."""
    nodes = list(ast.walk(test))
    if any(isinstance(n, ast.Compare) and any(isinstance(op, (ast.In, ast.NotIn)) for op in n.ops) for n in nodes):
        return False
    if any(isinstance(n, (ast.Call, ast.Attribute)) and _called_name(n) in _EXISTENCE for n in nodes):
        return False
    return any((isinstance(n, ast.Compare) and any(isinstance(op, _ORDER) for op in n.ops))
               or (isinstance(n, ast.Call) and _called_name(n) in _VALUE_TESTS) for n in nodes)


def _read_names(tree):
    """The names the program binds straight from a reading call (_READERS: read_csv, json.load, ...)."""
    return {target.id for node in ast.walk(tree) if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
            and _READERS.fullmatch(_called_name(node.value) or '_') for target in node.targets
            if isinstance(target, ast.Name)}


def _reads_input(test, read):
    """Whether an if-test reads a column or key of loaded input: `df['price']` where df is in read."""
    return any(isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name) and n.value.id in read
               and isinstance(n.slice, ast.Constant) and isinstance(n.slice.value, str) for n in ast.walk(test))


def guard_raise(code, line, exc_name):
    """The program with the raise at line rewritten into a warning on stderr that keeps its exception (type and
    message), or None when that raise is not a check of the program's own results. It must construct the named
    exception class itself (`raise Name(...)`), sit outside any except clause and any while loop of its function
    (a loop that may need the raise to end), directly in the body of an if whose test checks computed values
    (_value_check) and reads no column of loaded input (_reads_input), and its message must not say that something
    is missing or that input is invalid (_INVALID_WORDS): a raise that guards missing or invalid input data, a
    re-raise of a library error and a stub's NotImplementedError are kept. In a for loop the warning is printed
    once (WARNING_ONCE)."""
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return None
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    found = [n for n in ast.walk(tree) if isinstance(n, ast.Raise) and n.lineno <= line <= (n.end_lineno or n.lineno)]
    if len(found) != 1 or not isinstance(found[0].exc, ast.Call) or _called_name(found[0].exc) != exc_name:
        return None
    node = found[0]
    parent = parents.get(node)
    if not isinstance(parent, ast.If) or node not in parent.body or not _value_check(parent.test):
        return None
    if _reads_input(parent.test, _read_names(tree)):
        return None
    scope, looped, child = parent, False, node
    while scope in parents and not isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
        if isinstance(scope, ast.ExceptHandler) or isinstance(scope, ast.While) and child not in scope.orelse:
            return None
        looped = looped or isinstance(scope, (ast.For, ast.AsyncFor)) and child not in scope.orelse
        scope, child = parents[scope], scope
    words = ' '.join(n.value for n in ast.walk(node.exc) if isinstance(n, ast.Constant) and isinstance(n.value, str))
    if _MISSING_WORDS.search(words) or _INVALID_WORDS.search(words):
        return None
    exc = ast.get_source_segment(code, node.exc)
    if exc is None:
        return None
    lines = code.split('\n')
    head, tail = lines[node.lineno - 1].encode(), lines[node.end_lineno - 1].encode()
    warning = (WARNING_ONCE if looped else WARNING_CALL).format(exc=exc, line=node.lineno)
    lines[node.lineno - 1:node.end_lineno] = [(head[:node.col_offset] + warning.encode()
                                               + tail[node.end_col_offset:]).decode()]
    out = '\n'.join(lines)
    return out if _compiles(out) else None


def own_raise(code, line, exc_name):
    """'check' when the exception exc_name at line is one the program raised itself (T1_S11_RAISE's patch note): the
    line holds a `raise Name(...)` or `raise Name` of that class, or an assert failing with AssertionError; '' for a
    raise in an except clause (it stands in for the library's or Python's error it caught, which the patch must fix,
    as guard_raise keeps it), a raise inside a while loop of its function (a loop that may need the raise to end), a
    re-raise (`raise`, `raise e`), a stub's NotImplementedError and a library's or Python's own error."""
    if not exc_name or exc_name == 'NotImplementedError' or type(line) is not int:
        return ''
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return ''
    found = [n for n in ast.walk(tree) if isinstance(n, (ast.Raise, ast.Assert))
             and n.lineno <= line <= (n.end_lineno or n.lineno)]
    if len(found) != 1:
        return ''
    node = found[0]
    if isinstance(node, ast.Assert):
        if exc_name != 'AssertionError':
            return ''
    elif not isinstance(node.exc, (ast.Call, ast.Name)) or _called_name(node.exc) != exc_name:
        return ''
    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    scope, child = parents.get(node), node
    while scope is not None and not isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
        if isinstance(scope, ast.ExceptHandler) or isinstance(scope, ast.While) and child not in scope.orelse:
            return ''
        scope, child = parents.get(scope), scope
    return 'check'


# Output names (staged generation): a file name is a literal path or name ending in a data, text or image extension.
_FILE_NAME = re.compile(r'[^\s\'"]*[\w-]\.(csv|tsv|json|jsonl|txt|parquet|feather|xlsx|xls|png|jpg|svg|pdf|pkl|pickle|'
                        r'npy|npz|md|html|yaml|yml|xml|h5)', re.I)
_READERS = re.compile(r'(read|load)\w*|genfromtxt|fromfile', re.I)  # calls whose file name is read, not written
_WRITE_MODE = re.compile(r'[rbt+]*[wax][rbt+]*')
OUTPUT_NAME_CHARS = 80  # longer literals are text, not names


def _tokens(code):
    """The tokens of code that may be cut, without comments, blank lines or indentation: tokenizing stops where
    the code does. NEWLINE tokens stay, as statement ends."""
    skip = (tokenize.NL, tokenize.COMMENT, tokenize.INDENT, tokenize.DEDENT, tokenize.ENDMARKER)
    found = []
    try:
        for token in tokenize.generate_tokens(io.StringIO(code).readline):
            if token.type not in skip:
                found.append(token)
    except (tokenize.TokenError, SyntaxError, ValueError, RecursionError, MemoryError):
        pass
    return found


def _string_value(token):
    """A plain one-line string literal's value (at most OUTPUT_NAME_CHARS), or None; f-strings and bytes are None."""
    if token.type != tokenize.STRING:
        return None
    try:
        value = ast.literal_eval(token.string)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return None
    ok = isinstance(value, str) and value and '\n' not in value and len(value) <= OUTPUT_NAME_CHARS
    return value if ok else None


def string_values(code):
    """The plain string literals of code that may be cut, and the literal parts of its f-strings, as a set."""
    middle = getattr(tokenize, 'FSTRING_MIDDLE', None)
    return {token.string if token.type == middle else _string_value(token) for token in _tokens(code)} - {None}


def has_name(values, name):
    """Whether values (string_values) still name name: the same literal, or for a file name any literal (an
    f-string part too) ending in its base name, as `os.path.join(out, 'a.csv')` names '/app/output/a.csv'."""
    base = name.rsplit('/', 1)[-1]
    return name in values or (bool(_FILE_NAME.fullmatch(name)) and any(value.endswith(base) for value in values))


def _calls(tokens):
    """{index of each '(' : whether its call reads a file}: a reader (_READERS), or open() without a write mode
    among its string arguments."""
    reads, stack = {}, []
    for i, token in enumerate(tokens):
        if token.type == tokenize.OP and token.string in ('(', '[', '{'):
            stack.append(i)
        elif token.type == tokenize.OP and token.string in (')', ']', '}') and stack:
            start = stack.pop()
            callee = tokens[start - 1].string if start and tokens[start - 1].type == tokenize.NAME else ''
            if tokens[start].string == '(' and callee:
                modes = [value for value in map(_string_value, tokens[start + 1:i]) if value]
                reads[start] = bool(_READERS.fullmatch(callee)) or callee == 'open' and not any(
                    _WRITE_MODE.fullmatch(mode) for mode in modes)
    return reads


def output_names(code, skip=''):
    """The output names code writes, in order and once each: every string literal that is a key of a dict display
    or of a subscript assignment (`d['key'] = ...`), and every file name (_FILE_NAME) that no call reads (_calls):
    not inside a reading call, and when a statement assigns it to a name, that name is passed to a call and never
    to a reading one. code may be cut, a reply's unverified tail included: the tokens before the cut count. The
    body of a top-level def named skip is left out. Staged generation checks that its program still has them
    (has_name)."""
    tokens = _tokens(code)
    reads, stack, found, start, skipping = _calls(tokens), [], [], 0, False
    taken = {True: set(), False: set()}  # the names a reading call takes, and those only other calls take
    for i, token in enumerate(tokens):
        if token.type == tokenize.OP and token.string in ('(', '[', '{'):
            stack.append(reads.get(i))
        elif token.type == tokenize.OP and token.string in (')', ']', '}'):
            stack = stack[:-1]
        elif token.type == tokenize.NAME and any(read is not None for read in stack):
            taken[any(stack)].add(token.string)
    stack = []
    for i, token in enumerate(tokens):
        if token.type == tokenize.NEWLINE:
            start = i + 1
            continue
        if i == start and token.start[1] == 0:  # a top-level statement begins
            skipping = bool(skip) and token.string == 'def' and i + 1 < len(tokens) and tokens[i + 1].string == skip
        if skipping:
            continue
        if token.type == tokenize.OP and token.string in ('(', '[', '{'):
            stack.append((token.string, reads.get(i, False)))
            continue
        if token.type == tokenize.OP and token.string in (')', ']', '}'):
            stack = stack[:-1]
            continue
        value = _string_value(token)
        if value is None or value in found:
            continue
        after = [item.string for item in tokens[i + 1:i + 3]]
        key = ((after[:1] == [':'] and stack and stack[-1][0] == '{')
               or (i and tokens[i - 1].string == '[' and after == [']', '=']))
        target = tokens[start].string if tokens[start].type == tokenize.NAME and tokens[start + 1:start + 2] and (
            tokens[start + 1].string == '=') else ''
        written = (_FILE_NAME.fullmatch(value) and not any(read for _, read in stack)
                   and (not target or target in taken[False] - taken[True]))
        if key or written:
            found.append(value)
    return found


# T1_S11_LINT: library calls whose results a program reads in a way that cannot be right, found without running it
# (api_lint), and a pyplot backend call that cannot work, rewritten before the run (backend_fixed).
_LSTSQ_MODULES = ('numpy.linalg', 'scipy.linalg')
# Calls that need a vector (the second lstsq value is the residual sum of squares, an array of at most one number).
_VECTOR_CALLS = frozenset({'diff', 'std', 'var', 'square', 'cumsum', 'dot', 'corrcoef', 'correlate', 'convolve',
                           'autocorr', 'acf', 'pacf', 'durbin_watson', 'jarque_bera', 'acorr_ljungbox', 'skew',
                           'kurtosis', 'shapiro', 'normaltest', 'het_breuschpagan', 'het_white', 'percentile',
                           'quantile'})
_VECTOR_METHODS = frozenset({'std', 'var', 'cumsum', 'dot', 'autocorr', 'diff', 'skew', 'kurt', 'kurtosis'})
_RNG_MAKERS = ('numpy.random.default_rng', 'numpy.random.RandomState', 'numpy.random.Generator')
_RNG_NAME = re.compile(r'_?(?:\w*_)?rng\w*|random_state|prng', re.I)  # a parameter that carries a numpy generator
# The TASK's own words for a Pareto with a minimum (the symbol x_m near it, or 'minimum' in its sentence), and for
# the shifted Lomax / Pareto II.
_PARETO = re.compile(r'pareto', re.I)
_PARETO_SYMBOL = re.compile(r'\bx_?(?:\{m(?:in)?\}|m(?:in)?\b)', re.I)
_PARETO_MINIMUM = re.compile(r'\bminimum\b|\blower bound\b', re.I)
# 'Run a minimum of 100000 simulations': a count, not the distribution's minimum.
_COUNT_MINIMUM = re.compile(r'\b(?:minimum|lower bound)\s+(?:number\s+)?of\s+[\d,_.]+\s*[km]?\s+(?:[a-z-]+\s+)?'
                            r'(?:simulat|path|trial|sample|draw|iteration|observation|row|scenario|year|day|run|step|'
                            r'point|replicat|record|period|month|week|claim|event|loss)', re.I)
_SENTENCE = re.compile(r'(?<=[.!?;])\s+|\n')
_LOMAX = re.compile(r'lomax|pareto[\s_-]*(?:type[\s_-]*)?(?:ii|2)\b|(?:type|kind)[\s_-]*(?:ii|2)[\s_-]*pareto|'
                    r'generali[sz]ed[\s_-]*pareto', re.I)
# A TASK that writes numpy's pareto call itself, or names numpy's pareto, prescribes the draw: the TASK wins.
_TASK_DRAW = re.compile(r'(?<![A-Za-z_])pareto\(|(?i:\b(?:np|numpy|rng|random|generator)\b[\w.]*\.pareto\b|'
                        r"\bnumpy(?:'s)?\s+(?:random\s+)?pareto\b)")
PARETO_REACH = (150, 250)  # characters before and after a TASK 'Pareto' that its x_m may be named in


def _aliases(tree):
    """{name: dotted module or attribute it is bound to} by the program's imports, wherever they stand."""
    found = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for item in node.names:
                if item.asname:
                    found[item.asname] = item.name
                else:
                    root = item.name.split('.')[0]
                    found[root] = root
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            for item in node.names:
                if item.name != '*':
                    found[item.asname or item.name] = f'{node.module}.{item.name}'
    return found


def _resolved(node, aliases):
    """'a.b.c' for a Name or Attribute chain, its root replaced by what an import binds it to; None otherwise."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    return '.'.join([aliases.get(node.id, node.id)] + parts[::-1])


def _parents(tree):
    return {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}


def _scope(node, parents):
    """The innermost def or lambda around node, else the module."""
    while node in parents:
        node = parents[node]
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.Module)):
            return node
    return node


def _stores(scope, name):
    """How many times scope binds name: assignments, loop and with targets, parameters, imports and handlers."""
    count = 0
    for node in ast.walk(scope):
        if isinstance(node, ast.Name) and node.id == name and isinstance(node.ctx, (ast.Store, ast.Del)):
            count += 1
        elif isinstance(node, ast.arg) and node.arg == name:
            count += 1
        elif isinstance(node, _IMPORT) and any((a.asname or a.name.split('.')[0]) == name for a in node.names):
            count += 1
        elif isinstance(node, ast.ExceptHandler) and node.name == name:
            count += 1
    return count


def _is_name(node, name):
    return isinstance(node, ast.Name) and node.id == name


def _truth_test(node, parents):
    """Whether node (a len() call) only asks whether something is there: the whole test of an if, while or
    conditional expression, an operand of not/and/or or bool(), or compared with 0 or 1."""
    parent = parents.get(node)
    if isinstance(parent, (ast.If, ast.While, ast.IfExp, ast.Assert)) and parent.test is node:
        return True
    if isinstance(parent, (ast.BoolOp, ast.UnaryOp)) or isinstance(parent, ast.Call) and _is_name(parent.func, 'bool'):
        return True
    return isinstance(parent, ast.Compare) and all(
        isinstance(side, ast.Constant) and side.value in (0, 1) and not isinstance(side.value, bool)
        for side in [parent.left] + parent.comparators if side is not node)


def _vector_use(scope, name, parents):
    """The first read of name in scope that needs a vector: a power, a slice, a matrix product, a call that needs
    one (_VECTOR_CALLS, _VECTOR_METHODS), or a len() that is more than a test for emptiness. None otherwise."""
    uses = []
    for node in ast.walk(scope):
        if isinstance(node, ast.BinOp) and (
                (isinstance(node.op, ast.Pow) and _is_name(node.left, name))
                or (isinstance(node.op, ast.MatMult) and (_is_name(node.left, name) or _is_name(node.right, name)))):
            uses.append(node)
        elif isinstance(node, ast.Subscript) and _is_name(node.value, name) and isinstance(node.slice, ast.Slice):
            uses.append(node)
        elif isinstance(node, ast.Call):
            called = _called_name(node)
            if isinstance(node.func, ast.Attribute) and _is_name(node.func.value, name) and called in _VECTOR_METHODS:
                uses.append(node)
            elif called in _VECTOR_CALLS and any(_is_name(arg, name) for arg in node.args):
                uses.append(node)
            elif (_is_name(node.func, 'len') and len(node.args) == 1 and _is_name(node.args[0], name)
                  and not _truth_test(node, parents)):
                uses.append(node)
    return min(uses, key=lambda node: (node.lineno, node.col_offset), default=None)


def _second_return(node, lstsq, holders):
    """The name a statement binds to the second value of an lstsq call (lstsq(node) tells one): `coef, r, ... =
    lstsq(...)`, `r = lstsq(...)[1]`, or `r = res[1]` where res (in holders) holds a whole lstsq result. None
    otherwise."""
    if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
        return None
    target, value = node.targets[0], node.value
    if isinstance(target, (ast.Tuple, ast.List)) and len(target.elts) >= 2 and lstsq(value):
        if not any(isinstance(item, ast.Starred) for item in target.elts[:2]) and isinstance(target.elts[1], ast.Name):
            return target.elts[1].id
    if (isinstance(target, ast.Name) and isinstance(value, ast.Subscript) and isinstance(value.slice, ast.Constant)
            and value.slice.value == 1 and not isinstance(value.slice.value, bool)
            and (lstsq(value.value) or (isinstance(value.value, ast.Name) and value.value.id in holders))):
        return target.id
    return None


def _aliased(scope, name):
    """name and the names scope binds to it alone (`e = r`), each bound once there."""
    names, grown = {name}, True
    while grown:
        grown = False
        for node in ast.walk(scope):
            if (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                    and isinstance(node.value, ast.Name) and node.value.id in names
                    and node.targets[0].id not in names and _stores(scope, node.targets[0].id) == 1):
                names.add(node.targets[0].id)
                grown = True
    return names


def _lstsq_lint(tree, aliases, parents, lines):
    def lstsq(call):
        if not isinstance(call, ast.Call):
            return False
        name = _resolved(call.func, aliases)
        return bool(name) and any(name == module + '.lstsq' for module in _LSTSQ_MODULES)
    holders = {node.targets[0].id for node in ast.walk(tree)
               if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
               and lstsq(node.value) and _stores(_scope(node, parents), node.targets[0].id) == 1}
    found = []
    for node in ast.walk(tree):
        name = _second_return(node, lstsq, holders)
        if not name or name == '_':
            continue
        scope = _scope(node, parents)
        if _stores(scope, name) != 1:  # rebound (say to y - X @ coef): its later reads are not the lstsq value
            continue
        uses = [use for alias in _aliased(scope, name) for use in [_vector_use(scope, alias, parents)] if use]
        if uses:
            use = min(uses, key=lambda item: (item.lineno, item.col_offset))
            found.append((node.lineno, f'line {node.lineno}: `{lines[node.lineno - 1].strip()[:120]}` - `{name}` is '
                                       'the second value lstsq returns: the sum of squared residuals (an array of at '
                                       'most one number), not the residual vector, yet line '
                                       f'{use.lineno} uses it as the residuals; compute them as y - X @ coef'))
    return found


def _base_name(node):
    while isinstance(node, (ast.Subscript, ast.Attribute)):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def _added(scope, names):
    """Whether scope adds to one of names (or to an item or attribute of it): `x + 1`, `1 + x[i]`, `x += 1`."""
    for item in ast.walk(scope):
        if isinstance(item, ast.BinOp) and isinstance(item.op, ast.Add) and (
                _base_name(item.left) in names or _base_name(item.right) in names):
            return True
        if isinstance(item, ast.AugAssign) and isinstance(item.op, ast.Add) and _base_name(item.target) in names:
            return True
    return False


def _shifted(call, parents, aliases, tree, depth=1):
    """Whether a draw (a pareto(...) call, or a call of the program's function that returns one) may be shifted:
    an addition in the expression around it or on the name (or container) the statement around it assigns, an
    argument of a call outside numpy (which may shift it), or, for a draw a function returns, any call of that
    function that may be. Unknown counts as shifted, so the lint only fires where no shift can be."""
    node = call
    while isinstance(parents.get(node), ast.expr):
        parent = parents[node]
        if isinstance(parent, ast.BinOp) and isinstance(parent.op, ast.Add):
            return True
        if isinstance(parent, ast.Call) and node is not parent.func and not (
                _resolved(parent.func, aliases) or '').startswith('numpy.'):
            return True
        node = parent
    statement = parents.get(node)
    if not isinstance(statement, (ast.Assign, ast.AugAssign, ast.AnnAssign, ast.Return, ast.For)):
        return True  # a keyword argument, a comprehension, a test: where the draw goes is not followed
    if isinstance(statement, ast.Return):
        function = _scope(statement, parents)
        if depth <= 0 or not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return True
        calls = [item for item in ast.walk(tree) if isinstance(item, ast.Call) and _is_name(item.func, function.name)]
        return not calls or any(_shifted(item, parents, aliases, tree, depth - 1) for item in calls)
    targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
    names = {item.id for target in targets for item in ast.walk(target) if isinstance(item, ast.Name)}
    names |= {_base_name(target) for target in targets} - {None}
    return _added(_scope(statement, parents), names)


def classical_pareto(instruction):
    """Whether the TASK names a Pareto distribution with a minimum and never the shifted Lomax / Pareto II nor numpy's
    own pareto draw (_TASK_DRAW, which it then prescribes): the symbol x_m near a 'Pareto', or 'minimum' or 'lower
    bound' in a sentence that names one (not 'a minimum of 100000 simulations')."""
    if not instruction or _LOMAX.search(instruction) or _TASK_DRAW.search(instruction):
        return False
    before, after = PARETO_REACH
    if any(_PARETO_SYMBOL.search(instruction[max(0, m.start() - before):m.end() + after])
           for m in _PARETO.finditer(instruction)):
        return True
    return any(_PARETO.search(sentence) and _PARETO_MINIMUM.search(_COUNT_MINIMUM.sub(' ', sentence))
               for sentence in _SENTENCE.split(instruction))


def _pareto_lint(tree, aliases, parents, lines):
    makers = set()  # names bound to a numpy generator: rng = np.random.default_rng(...)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call) and (
                _resolved(node.value.func, aliases) in _RNG_MAKERS):
            makers |= {target.id for target in node.targets if isinstance(target, ast.Name)}
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Attribute) and node.func.attr == 'pareto':
            receiver = node.func.value
            name = _resolved(receiver, aliases) or ''
            if isinstance(receiver, ast.Call):
                numpy = _resolved(receiver.func, aliases) in _RNG_MAKERS
            else:
                last = name.rsplit('.', 1)[-1]
                numpy = (name == 'numpy.random' or (isinstance(receiver, ast.Name) and receiver.id in makers)
                         or (not name.startswith(('scipy', 'jax', 'torch')) and bool(_RNG_NAME.fullmatch(last))))
        else:
            numpy = isinstance(node.func, ast.Name) and _resolved(node.func, aliases) == 'numpy.random.pareto'
        if numpy and not _shifted(node, parents, aliases, tree):
            found.append((node.lineno, f'line {node.lineno}: `{lines[node.lineno - 1].strip()[:120]}` - numpy\'s '
                                       'pareto(a) draws from the Lomax (Pareto II) distribution, which starts at 0; a '
                                       'Pareto variate with minimum x_m, as the TASK describes it, is x_m * (1 + '
                                       'pareto(a))'))
    return found


def api_lint(program, instruction='', pareto=True):
    """Library results the program reads in a way that cannot be right, found without running it: the second value
    of numpy's or scipy's lstsq (the residual sum of squares) bound to a name the program then uses as the residual
    vector (a power, a slice, a matrix product, a vector statistic or a length), and, when the TASK names a Pareto
    distribution with a minimum and never the Lomax / Pareto II (classical_pareto), a numpy pareto() draw with no
    shift (no addition around it or on the name it is assigned to). ['line N: `source` - why'] in line order; []
    when the program does not compile. Model-facing advice only."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            tree = ast.parse(program)
            compile(tree, '<program>', 'exec', dont_inherit=True)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return []
    try:
        aliases, parents, lines = _aliases(tree), _parents(tree), program.split('\n')
        found = _lstsq_lint(tree, aliases, parents, lines)
        if pareto and classical_pareto(instruction):
            found += _pareto_lint(tree, aliases, parents, lines)
    except RecursionError:
        return []
    return [text for _, text in sorted(set(found))]


def _module_binds(tree, name, module):
    """Whether a statement outside every def and class imports module under name."""
    todo = list(tree.body)
    while todo:
        node = todo.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        if isinstance(node, ast.Import) and any(
                (item.asname or item.name.split('.')[0]) == name
                and (item.name if item.asname else item.name.split('.')[0]) == module for item in node.names):
            return True
        todo.extend(ast.iter_child_nodes(node))
    return False


def backend_fixed(program):
    """(program, changes): a pyplot backend call that cannot work made one that does, zero requests. pyplot has no
    use(), so `plt.use(...)` (any name or path bound to matplotlib.pyplot) becomes `matplotlib.use(...)`, and a
    `matplotlib.use(...)` without a module-level `import matplotlib` gets that line before the program's first
    statement (after its docstring and __future__ imports). changes counts the calls rewritten and the import added;
    the program as it is (0) when there is nothing to change, it does not compile, or the result would not."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            tree = ast.parse(program)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return program, 0
    aliases, edits, uses = _aliases(tree), [], False
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'use'):
            continue
        receiver = node.func.value
        name = _resolved(receiver, aliases)
        if name == 'matplotlib.pyplot' and receiver.lineno == receiver.end_lineno:
            edits.append((receiver.lineno, receiver.col_offset, receiver.end_col_offset))
            uses = True
        elif _is_name(receiver, 'matplotlib') and name == 'matplotlib':
            uses = True
    imported = uses and not _module_binds(tree, 'matplotlib', 'matplotlib')
    if not edits and not imported:
        return program, 0
    lines = program.split('\n')
    for row, begin, end in sorted(edits, reverse=True):
        data = lines[row - 1].encode()  # ast offsets count UTF-8 bytes
        lines[row - 1] = (data[:begin] + b'matplotlib' + data[end:]).decode()
    if imported:
        first = next((node for node in tree.body if not (
            (isinstance(node, ast.ImportFrom) and node.module == '__future__')
            or (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)))), None)
        if first is None or first.col_offset != 0:
            return program, 0
        lines.insert(_start(first) - 1, 'import matplotlib')
    fixed = '\n'.join(lines)
    return (fixed, len(edits) + imported) if _compiles(fixed) else (program, 0)


# T1_S12_POST: code lints after a clean run that cost no request (s12_lint), and a call of a name the program never
# binds that numpy has, rewritten to numpy's before every run (numpy_names_fixed). numpy is read for its own names and
# ufunc arities, never listed by hand; without numpy (a local test run) both find nothing.
_NUMPY_NAMES = None
_YEAR_BASES = (360, 365, 365.25)
_YEAR_WORDS = re.compile(r'\bACT/36[05]\b|\bA/36[05]\b|\bactual/36[05]\b|\byear[- _]?fractions?\b|\byearfrac\b|'
                         r'\btime[- _]to[- _](?:expir|maturit)|\bday[- _]count\b', re.I)
_WHOLE_DAYS = re.compile(r'\b(?:whole|calendar|integer|full) days\b|\bdays between (?:the )?dates\b|\bdate[- ]only\b',
                         re.I)
_DAY_WRAPPERS = frozenset({'float', 'int', 'abs', 'asarray', 'array', 'float64', 'to_numpy', 'astype'})
_AGGREGATES = frozenset({'mean', 'nanmean', 'average', 'median', 'nanmedian', 'sum', 'nansum', 'fmean'})
_AGGREGATE_MODULES = ('numpy.', 'statistics.')
_GROWS = frozenset({'append', 'extend', 'insert'})
_SHRINKS = frozenset({'pop', 'popleft', 'remove', 'clear'})
_EXPANDING = re.compile(r'\bexpanding\b|\brunning (?:mean|average|median|sum|total)\b|\bso far\b|\bto date\b|'
                        r'\bcumulative (?:mean|average|median)\b|\bup to (?:and including )?(?:that|the current|each) '
                        r'(?:day|date|period|step|bar|observation|row)\b', re.I)


def numpy_names():
    """{top-level name: input count (ufunc.nin) for a ufunc, else None} of numpy's public callables, read from numpy
    itself once; {} when numpy cannot be imported (never cached then)."""
    global _NUMPY_NAMES
    if _NUMPY_NAMES is None:
        try:
            import numpy
            found = {}
            for name in dir(numpy):
                value = getattr(numpy, name, None)
                if not name.startswith('_') and callable(value):
                    found[name] = value.nin if isinstance(value, numpy.ufunc) else None
        except Exception:
            return {}
        _NUMPY_NAMES = found
    return _NUMPY_NAMES


def _numpy_alias(tree):
    """The name a module-level `import numpy` binds ('np' first), else None."""
    names = [item.asname or 'numpy' for node in tree.body if isinstance(node, ast.Import) for item in node.names
             if item.name == 'numpy']
    return 'np' if 'np' in names else names[0] if names else None


def numpy_names_fixed(program, names=None):
    """(program, changes): a call of a name the program never binds that numpy has, such as `exp(x)`
    after a fill reply that only imported numpy, becomes `<alias>.<name>(...)` with the alias a module-level `import
    numpy` binds; zero requests. names: numpy's callables (numpy_names()). changes counts the calls rewritten; the
    program as it is (0) when numpy is not imported, nothing is to change, or the result would not compile."""
    names = numpy_names() if names is None else names
    # A newly detected inaccessible nested helper is not permission to guess it
    # meant NumPy. Preserve the original program-wide automatic-repair eligibility.
    missing = [name for name in _numpy_missing_names(program) if name in names]
    if not missing:
        return program, 0
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            tree = ast.parse(program)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return program, 0
    alias = _numpy_alias(tree)
    if alias is None:
        return program, 0
    edits = sorted(((node.func.lineno, node.func.col_offset, node.func.end_col_offset, node.func.id)
                    for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id in missing and node.func.lineno == node.func.end_lineno), reverse=True)
    lines = program.split('\n')
    for row, begin, end, name in edits:
        data = lines[row - 1].encode()  # ast offsets count UTF-8 bytes
        lines[row - 1] = (data[:begin] + f'{alias}.{name}'.encode() + data[end:]).decode()
    fixed = '\n'.join(lines)
    return (fixed, len(edits)) if edits and _compiles(fixed) else (program, 0)


def _days_operand(node, days_names):
    """Whether node is a whole-day count: a `.days` attribute (a timedelta's, or a pandas `.dt.days`), through
    float()/int()/.astype()/.values wrappers, or a name a single assignment binds to one."""
    while True:
        if isinstance(node, ast.Call) and len(node.args) <= 1 and _called_name(node) in _DAY_WRAPPERS:
            node = node.func.value if _called_name(node) in ('to_numpy', 'astype') and isinstance(
                node.func, ast.Attribute) else node.args[0] if node.args else None
        elif isinstance(node, ast.Attribute) and node.attr == 'values':
            node = node.value
        else:
            break
    return (isinstance(node, ast.Attribute) and node.attr == 'days') or (
        isinstance(node, ast.Name) and node.id in days_names)


def _year_base(node, constants):
    value = node.value if isinstance(node, ast.Constant) else constants.get(node.id) if isinstance(
        node, ast.Name) else None
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) and value in _YEAR_BASES else None


def _days_lint(tree, parents, lines):
    """A whole-day count divided by 360, 365 or 365.25 (a constant, or a module constant bound to one)."""
    constants, days_names = {}, set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name, scope = node.targets[0].id, _scope(node, parents)
            if _stores(scope, name) != 1:
                continue
            if isinstance(node.value, ast.Constant) and isinstance(scope, ast.Module):
                constants[name] = node.value.value
            elif _days_operand(node.value, ()):
                days_names.add(name)
    found = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div)):
            continue
        base = _year_base(node.right, constants)
        if base is not None and _days_operand(node.left, days_names):
            found.append((node.lineno, f'line {node.lineno}: `{lines[node.lineno - 1].strip()[:120]}` - `.days` keeps '
                                       'whole days and drops the time of day, yet the inputs carry times of day: a '
                                       'year fraction between two timestamps counts the elapsed time, (end - '
                                       f'start).total_seconds() / 86400 / {base:g} (pandas: .dt.total_seconds()), '
                                       'unless the TASK counts whole days'))
    return found


def _loop_nodes(loop):
    """The nodes of a loop's body (not its else), defs and lambdas inside it left out."""
    found, todo = [], list(loop.body)
    while todo:
        node = todo.pop()
        found.append(node)
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            todo.extend(ast.iter_child_nodes(node))
    return found


def _aggregate(node, aliases, bound):
    """The aggregate a call takes of one bare name (np.mean(xs), statistics.median(xs), sum(xs)): (function, name),
    else None."""
    if not (isinstance(node, ast.Call) and len(node.args) >= 1 and isinstance(node.args[0], ast.Name)):
        return None
    name = _resolved(node.func, aliases) or ''
    short = name.rsplit('.', 1)[-1]
    if short not in _AGGREGATES:
        return None
    if name.startswith(_AGGREGATE_MODULES) and name.count('.') == 1 or (name == 'sum' and 'sum' not in bound):
        return short, node.args[0].id
    return None


def _statement(node, parents):
    """The innermost statement that holds node."""
    while not isinstance(node, ast.stmt) and node in parents:
        node = parents[node]
    return node


def _read_later(loop_nodes, statement, names):
    """Whether one of names is read in the loop's body after statement."""
    end = statement.end_lineno or statement.lineno
    return any(isinstance(node, ast.Name) and node.id in names and isinstance(node.ctx, ast.Load)
               and node.lineno > end for node in loop_nodes)


def _running_lint(tree, aliases, parents, lines, instruction):
    """An aggregate of a list taken inside the loop that grows the list, used in that pass, while the program also
    takes it over the whole list after the loop (or the TASK calls it the grand one): a running aggregate where the
    TASK means the one over all items. Silent where the TASK says expanding, running, so far or to date, where the
    loop also shrinks the list (a moving window) or the list is made anew inside the loop."""
    if _EXPANDING.search(instruction or ''):
        return []
    grand = re.search(r'\bgrand\b', instruction or '', re.I) is not None
    bound = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)}
    bound |= {node.name for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    found = []
    for loop in ast.walk(tree):
        if not isinstance(loop, (ast.For, ast.AsyncFor, ast.While)):
            continue
        nodes = _loop_nodes(loop)
        grown, shrunk = {}, set()
        for node in nodes:
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and isinstance(
                    node.func.value, ast.Name):
                if node.func.attr in _GROWS:
                    grown.setdefault(node.func.value.id, node.lineno)
                elif node.func.attr in _SHRINKS:
                    shrunk.add(node.func.value.id)
            elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
                shrunk.add(node.id)
            elif isinstance(node, (ast.Subscript, ast.Attribute)) and isinstance(node.ctx, (ast.Store, ast.Del)):
                shrunk.add(_base_name(node))
        scope = _scope(loop, parents)
        after = {_aggregate(node, aliases, bound) for node in ast.walk(scope)
                 if isinstance(node, ast.Call) and node.lineno > (loop.end_lineno or loop.lineno)} - {None}
        for node in nodes:
            taken = _aggregate(node, aliases, bound) if isinstance(node, ast.Call) else None
            if not taken or taken[1] not in grown or taken[1] in shrunk or node.lineno <= grown[taken[1]]:
                continue
            if not (taken in after or grand):
                continue
            statement = _statement(node, parents)
            targets = [item.id for target in getattr(statement, 'targets', [getattr(statement, 'target', None)])
                       if target is not None for item in ast.walk(target) if isinstance(item, ast.Name)]
            plain = isinstance(statement, (ast.Assign, ast.AnnAssign)) and targets
            if plain and not _read_later(nodes, statement, set(targets)):
                continue  # only the last pass's value is kept: that one is over the whole list
            found.append((node.lineno, f'line {node.lineno}: `{lines[node.lineno - 1].strip()[:120]}` - this '
                                       f'{taken[0]} of `{taken[1]}` is taken inside the loop that appends to it, so '
                                       f'each pass uses only the items appended so far (a running {taken[0]}), '
                                       f'while the program also takes the {taken[0]} over every item after the loop; '
                                       f'where the TASK means the {taken[0]} over all items for every row, compute it '
                                       'in a first pass, before the loop that uses it'))
    return found


def _ufunc_lint(tree, aliases, lines, names):
    """A numpy ufunc called with more positional arguments than it has inputs (ufunc.nin): numpy takes the next ones
    as output arrays, which it overwrites, or raises."""
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or any(isinstance(arg, ast.Starred) for arg in node.args):
            continue
        name = _resolved(node.func, aliases) or ''
        short = name[len('numpy.'):] if name.startswith('numpy.') else ''
        inputs = names.get(short) if short and '.' not in short else None
        if inputs is None or len(node.args) <= inputs:
            continue
        found.append((node.lineno, f'line {node.lineno}: `{lines[node.lineno - 1].strip()[:120]}` - numpy\'s {short} '
                                   f'takes {inputs} input{"s" if inputs > 1 else ""}; a further positional argument is '
                                   'an output array it overwrites, not another input'
                                   + (f'; combine more terms with np.{short}.reduce([...])' if inputs == 2 else '')))
    return found


def _compiled_tree(program):
    """The program's syntax tree when it compiles, else None."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            tree = ast.parse(program)
            compile(tree, '<program>', 'exec', dont_inherit=True)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return None
    return tree


def ufunc_lint(program, names=None, limit=LINT_LIMIT):
    """_ufunc_lint alone, for a crash ask (T1_S12_POST): ['line N: `source` - why'] in line order, at most limit; []
    when the program does not compile or numpy cannot be read."""
    tree = _compiled_tree(program)
    if tree is None:
        return []
    try:
        found = _ufunc_lint(tree, _aliases(tree), program.split('\n'), numpy_names() if names is None else names)
    except RecursionError:
        return []
    return [text for _, text in sorted(set(found))][:limit]


def s12_lint(program, instruction='', timed=False, names=None, limit=LINT_LIMIT):
    """T1_S12_POST's code lints after a clean run, zero requests: a whole-day count (`.days`) divided by 360, 365 or
    365.25 where the inputs carry times of day (timed) and the TASK speaks of a year fraction, an ACT/365 or a time to
    expiry, and not of whole or calendar days (_days_lint); a running aggregate where the TASK means the one over all
    items (_running_lint); a numpy ufunc given more positional arguments than inputs (_ufunc_lint; names: numpy's
    callables, numpy_names()). ['line N: `source` - why'] in line order, at most limit; [] when the program does not
    compile. Model-facing evidence only."""
    tree = _compiled_tree(program)
    if tree is None:
        return []
    try:
        aliases, parents, lines = _aliases(tree), _parents(tree), program.split('\n')
        found = _running_lint(tree, aliases, parents, lines, instruction)
        found += _ufunc_lint(tree, aliases, lines, numpy_names() if names is None else names)
        if timed and _YEAR_WORDS.search(instruction or '') and not _WHOLE_DAYS.search(instruction or ''):
            found += _days_lint(tree, parents, lines)
    except RecursionError:
        return []
    return [text for _, text in sorted(set(found))][:limit]
