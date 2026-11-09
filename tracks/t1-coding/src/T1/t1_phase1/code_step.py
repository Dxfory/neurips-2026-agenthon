"""Static, atomic append to an executed program; no semantic success claims."""
import ast
import hashlib
import io
import tokenize
import unicodedata

from .program_version import ProgramVersion


_IMPORTS = (ast.Import, ast.ImportFrom)


def _tree(source):
    compile(source, '<code-step>', 'exec', dont_inherit=True)
    return ast.parse(source)


def fingerprint(source):
    """Ignore comments and layout when deciding whether code already ran."""
    return hashlib.sha256(ast.dump(_tree(source), include_attributes=False).encode()).hexdigest()


def _dumps(nodes):
    return [ast.dump(node, include_attributes=False) for node in nodes]


def _contains(sequence, group):
    return bool(group) and any(sequence[i:i+len(group)] == group
                               for i in range(len(sequence)-len(group)+1))


def _header(tree):
    nodes = tree.body
    doc = (nodes[0] if nodes and isinstance(nodes[0], ast.Expr)
           and isinstance(nodes[0].value, ast.Constant)
           and isinstance(nodes[0].value.value, str) else None)
    cursor = int(doc is not None)
    start = cursor
    while cursor < len(nodes) and isinstance(nodes[cursor], _IMPORTS):
        cursor += 1
    return doc, nodes[start:cursor], nodes[cursor:]


def _bindings(node):
    return [alias.asname or (alias.name.split('.')[0] if isinstance(node, ast.Import)
                            else alias.name) for alias in node.names]


def _first_line(node):
    return min([node.lineno] + [d.lineno for d in getattr(node, 'decorator_list', ())])


def _physical_lines(source):
    # AST lineno counts Python newlines, not every Unicode separator inside
    # strings. Preserve CRLF bytes, but reject ambiguous bare-CR source.
    if '\r' in source.replace('\r\n', ''):
        raise ValueError('append_invalid_code')
    pieces = source.split('\n')
    return [line + '\n' for line in pieces[:-1]] + [pieces[-1]]


def _whole_lines(source, node, nodes):
    """Extract an import only when no other statement shares its physical lines."""
    start, end = _first_line(node), node.end_lineno
    if node.col_offset or any(other is not node and
            _first_line(other) <= end and other.end_lineno >= start for other in nodes):
        raise ValueError('append_ambiguous_prefix')
    return ''.join(_physical_lines(source)[start-1:end])


def recover_suffix(program, source):
    """Recover a bounded full-prefix reply, otherwise retain a native append.

    Only a complete old body at the beginning can be removed. Leading imports
    may add fresh names, but old imports and the module docstring must survive.
    Added imports execute AFTER the original program: this is the APPEND
    contract, not a claim that arbitrary import reordering is equivalent.
    Replay-shaped partial/modified prefixes are rejected conservatively; it is
    not generally possible to infer whether arbitrary new code intended an edit.
    """
    _physical_lines(program)
    _physical_lines(source)
    old, incoming = _tree(program), _tree(source)
    old_doc, old_imports, old_body = _header(old)
    new_doc, new_imports, new_body = _header(incoming)
    old_keys, new_keys = _dumps(old_body), _dumps(new_body)
    old_import_keys = _dumps(old_imports)
    old_header_bindings = {name for node in old_imports for name in _bindings(node)}
    shared_header = any(ast.dump(node, include_attributes=False) in old_import_keys or
                        set(_bindings(node)) & old_header_bindings for node in new_imports)
    same_doc = (old_doc is not None and new_doc is not None and
                ast.dump(old_doc, include_attributes=False) == ast.dump(new_doc, include_attributes=False))
    replay_shaped = (shared_header or same_doc or
                     bool(old_keys and new_keys and old_keys[0] == new_keys[0]) or
                     _contains(_dumps(incoming.body), old_keys))
    if not replay_shaped:
        return source
    if (_dumps([old_doc]) if old_doc is not None else []) != (
            _dumps([new_doc]) if new_doc is not None else []):
        raise ValueError('append_ambiguous_prefix')
    if not old_keys or new_keys[:len(old_keys)] != old_keys:
        raise ValueError('append_ambiguous_prefix')

    # Lexical names include loads, stores, parameters, definitions, exception
    # targets, attributes and import components, even in dormant nested scopes.
    old_names = {unicodedata.normalize('NFKC', token.string)
                 for token in tokenize.generate_tokens(io.StringIO(program).readline)
                 if token.type == tokenize.NAME}
    old_cursor, added_names, added = 0, set(), []
    for node in new_imports:
        key = ast.dump(node, include_attributes=False)
        if old_cursor < len(old_import_keys) and key == old_import_keys[old_cursor]:
            old_cursor += 1
            continue
        if key in old_import_keys:
            raise ValueError('append_ambiguous_prefix')
        names = _bindings(node)
        if (isinstance(node, ast.ImportFrom) and (node.level or node.module == '__future__')
                or any(alias.name == '*' for alias in node.names)
                or len(names) != len(set(names)) or set(names) & (old_names | added_names)):
            raise ValueError('append_unsafe_import')
        added_names.update(names)
        added.append(_whole_lines(source, node, incoming.body))
    if old_cursor != len(old_import_keys):
        raise ValueError('append_ambiguous_prefix')

    tail = new_body[len(old_body):]
    if not tail:
        raise ValueError('append_no_executable_step')
    end = new_body[len(old_body)-1].end_lineno
    if _first_line(tail[0]) <= end:
        raise ValueError('append_ambiguous_prefix')
    suffix = ''.join(_physical_lines(source)[end:])
    if _contains(_dumps(_tree(suffix).body), old_keys):
        raise ValueError('append_repeated_code')
    return ''.join(added) + suffix


def compose(program, source, *, base_version, executed_version, previous_steps=()):
    """Keep the exact prefix; require a complete, non-replayed executable suffix.

    Variable reuse is allowed. This does not prove which branches run, whether
    results improve, or whether an assignment preserves a financial invariant.
    """
    if (not program or hashlib.sha256(program.encode()).hexdigest() != base_version
            or base_version != executed_version):
        raise ValueError('append_stale_base')
    try:
        suffix = recover_suffix(program, source)
        incoming = _tree(suffix)
        work = [node for node in incoming.body
                if not isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef,
                                         ast.AsyncFunctionDef, ast.ClassDef, ast.Pass))
                and not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant))]
        if not work:
            raise ValueError('append_no_executable_step')
        if fingerprint(suffix) in previous_steps:
            raise ValueError('append_repeated_code')
        candidate = program + ('' if program.endswith('\n') else '\n') + suffix
        probe = ProgramVersion()
        result = probe.apply(candidate, 'stop', 'replace', probe.version)
        if not result.accepted or not probe.ready:
            raise ValueError('append_incomplete_step')
        return candidate
    except (SyntaxError, TypeError, RecursionError, MemoryError, OverflowError, tokenize.TokenError) as exc:
        raise ValueError('append_invalid_code') from exc
