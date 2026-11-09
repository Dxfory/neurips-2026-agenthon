"""Exact declared input relocation, without filesystem aliases or basename guesses.

The controller freezes a map from participant metadata. Source string literals equal
to those file paths are relocated, and so are literals that name a mounted directory
or start with one (a directory constant, the first part of a join, concatenation or
f-string); other computed paths stay unchanged, /app and /app/output never move, and
neither does a directory literal in a statement that climbs a path ('..', dirname, parent).
The existing worker and read-only input mount enforce file access as before.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path, PurePosixPath
import re
import stat

from . import staging

METADATA_BYTES = 512 * 1024
MAX_BINDINGS = 8192
_NEVER_MOVED = ('/app', '/app/output')
_FSTRING_OPEN = re.compile(rb'(?:[rRfFuU]{0,2}(?:\'\'\'|"""|\'|"))?')  # a string prefix and quote
_FSTRING_UNSAFE = re.compile('[{}\\\\\'"\x00-\x1f\x7f]')  # what f-string source would have to escape
_CLIMBING = frozenset({'dirname', 'pardir', 'parent', 'parents'})


class Bindings(tuple):
    """The sorted (original file, runtime file) pairs; directories holds the sorted (original directory, runtime
    directory) pairs of the mounted directories and the directories below them, as exact as the files."""
    directories = ()

    def __new__(cls, files=(), directories=()):
        self = super().__new__(cls, files)
        self.directories = tuple(directories)
        return self


def _never_moved(path):
    return path in _NEVER_MOVED or path.startswith('/app/output/')


def _relative(value):
    if not isinstance(value, str) or not value or '\\' in value or any(ord(c) < 32 for c in value):
        raise ValueError('invalid_input_binding_path')
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value or any(p in ('.', '..', '') for p in value.split('/')):
        raise ValueError('invalid_input_binding_path')
    if not staging.participant_file(path) or value in ('instruction.md', 'card.toml'):
        raise ValueError('input_binding_not_participant_data')
    return path


def _safe(path, root):
    relative = path.relative_to(root)
    for index in range(1, len(relative.parts) + 1):
        if (root / Path(*relative.parts[:index])).is_symlink():
            raise ValueError('input_binding_symlink')
    mode = path.lstat().st_mode
    if not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)) or not path.resolve().is_relative_to(root):
        raise ValueError('input_binding_not_regular')
    return stat.S_ISDIR(mode)


def _metadata(path, root):
    if _safe(path, root) or path.stat().st_size > METADATA_BYTES:
        raise ValueError('input_binding_metadata_limit')


def _unique_object(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError('input_binding_duplicate_metadata_key')
        result[name] = value
    return result


def load(task_dir):
    """Return immutable exact (original file, runtime file) bindings.

    A present manifest is authoritative, including when it is empty or invalid.
    Only when it is absent may literal Docker COPY metadata supply the mappings.
    Missing metadata yields no bindings; invalid metadata raises without fallback.
    Directory declarations enumerate existing allowed files; the declared directory and
    those below it are kept as directory bindings (Bindings.directories) only where every
    file bound under one lies at the same relative path under its runtime directory.
    """
    root = Path(task_dir).resolve()
    manifest = root / 'staging_manifest.json'
    if manifest.exists() or manifest.is_symlink():
        _metadata(manifest, root)
        document = json.loads(manifest.read_text(), object_pairs_hook=_unique_object)
        if not isinstance(document, dict) or not isinstance(document.get('data_mounts'), list):
            raise ValueError('input_binding_invalid_manifest')
        mappings = document['data_mounts']
    else:
        for relative in ('Dockerfile', 'environment/Dockerfile'):
            path = root / relative
            if path.exists() or path.is_symlink():
                _metadata(path, root)
        mappings = staging.data_mounts(root)
    if len(mappings) > MAX_BINDINGS:
        raise ValueError('input_binding_limit')
    result, directories, visited = {}, {}, 0
    for item in mappings:
        if not isinstance(item, dict):
            raise ValueError('input_binding_invalid_mapping')
        relative = _relative(item.get('source_relative'))
        source = root / Path(relative)
        _safe(source, root)
        destination = item.get('destination')
        if (not isinstance(destination, str) or PurePosixPath(destination).as_posix() != destination
                or any(ord(char) < 32 for char in destination)):
            raise ValueError('input_binding_invalid_destination')
        # The stager's exact protected-path rules apply here too; no /app/output.
        staging._destination(destination, source)
        pending = [(source, destination)]
        while pending:
            path, original = pending.pop()
            visited += 1
            if visited > MAX_BINDINGS:
                raise ValueError('input_binding_limit')
            if _safe(path, root):
                directories.setdefault(original, set()).add(str(path))
                pending.extend((child, original + '/' + child.name) for child in sorted(path.iterdir()))
                continue
            _relative(path.relative_to(root).as_posix())
            staging._destination(original, path)
            runtime = str(path)
            if original in result and result[original] != runtime:
                raise ValueError('input_binding_conflict')
            result[original] = runtime
    exact = {}
    for original, runtimes in directories.items():
        runtime = min(runtimes)
        if (len(runtimes) == 1 and not _never_moved(original) and runtime != original
                and not runtime.startswith(original + '/')
                and all(value == runtime + key[len(original):] for key, value in result.items()
                        if key.startswith(original + '/'))):
            exact[original] = runtime
    return Bindings(sorted(result.items()), sorted(exact.items()))


def _climbs(node):
    """A string with a '..' segment, or a dirname / pardir / parent / parents step."""
    if isinstance(node, ast.Constant):
        return isinstance(node.value, str) and '..' in node.value.split('/')
    return (isinstance(node, ast.Attribute) and node.attr in _CLIMBING
            or isinstance(node, ast.Name) and node.id in _CLIMBING)


def _climbing(tree):
    """The ids of the expression nodes of every statement that climbs a path (its own expressions, not those of the
    statements nested in it): relocating a directory there could move a path that leaves the mounted directory,
    such as '/app/data/../output', os.path.join('/app/data', '..', 'output') or Path('/app/data').parent."""
    found = set()
    for statement in ast.walk(tree):
        if not isinstance(statement, ast.stmt):
            continue
        own, stack = [], [child for child in ast.iter_child_nodes(statement) if not isinstance(child, ast.stmt)]
        while stack:
            node = stack.pop()
            own.append(node)
            stack.extend(child for child in ast.iter_child_nodes(node) if not isinstance(child, ast.stmt))
        if any(_climbs(node) for node in own):
            found.update(id(node) for node in own)
    return found


def _relocated(value, directories):
    """value with its longest mounted directory prefix replaced, or None: value names the directory itself (with or
    without a trailing slash) or a path under it."""
    for original, runtime in directories:
        if value == original or value.startswith(original + '/'):
            return runtime + value[len(original):]
    return None


def rewrite(program, bindings):
    """Relocate literal input paths and return source plus edit count.

    A string literal equal to a bound file is replaced whole; one that names a mounted
    directory or a path under it (a directory constant, the first operand of a join or
    a concatenation) has that prefix relocated; of an f-string, only the leading literal
    part may be relocated so. Preserve every other byte, including comments, output
    paths, docstrings, bytes literals, later f-string parts and unknown paths. /app and
    /app/output are never relocated, and no directory is relocated in a statement that
    climbs a path ('..', dirname, parent): that path may leave the mounted directory.
    No source is executed.
    """
    directories = sorted(((original, runtime) for original, runtime in getattr(bindings, 'directories', ())
                          if not _never_moved(original)), key=lambda pair: -len(pair[0]))
    if not (bindings or directories) or '/app/' not in program:
        return program, 0
    try:
        tree = ast.parse(program)
    except (SyntaxError, ValueError, RecursionError):
        return program, 0
    mapped = dict(bindings)
    climbing = _climbing(tree) if directories else set()
    specs = {id(node.format_spec) for node in ast.walk(tree)
             if isinstance(node, ast.FormattedValue) and node.format_spec is not None}
    leading = {id(node.values[0]): len(node.values) for node in ast.walk(tree)
               if isinstance(node, ast.JoinedStr) and id(node) not in specs and node.values
               and isinstance(node.values[0], ast.Constant) and isinstance(node.values[0].value, str)}
    skipped = {id(child) for node in ast.walk(tree) if isinstance(node, ast.JoinedStr)
               for child in ast.walk(node)}
    skipped.update(id(node.value) for node in ast.walk(tree)
                   if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant))
    raw = program.encode()
    lines = raw.splitlines(keepends=True)
    offsets, total = [], 0
    for line in lines:
        offsets.append(total)
        total += len(line)
    edits, spliced = [], []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
            continue
        start = offsets[node.lineno - 1] + node.col_offset
        end = offsets[node.end_lineno - 1] + node.end_col_offset
        if id(node) in leading:
            if id(node) in climbing:
                continue
            # Splice the runtime directory into the f-string's own source text, only where that text spells the
            # directory out literally and the runtime path needs no escaping there. A directory a placeholder
            # continues (f'/app/data{suffix}') is a different name, not a path under it.
            for original, runtime in directories:
                if node.value.startswith(original + '/') or node.value == original and leading[id(node)] == 1:
                    head = _FSTRING_OPEN.match(raw, start, end).end()
                    if raw.startswith(original.encode(), head) and not _FSTRING_UNSAFE.search(runtime):
                        spliced.append((head, head + len(original.encode()), runtime.encode()))
                    break
            continue
        if id(node) in skipped:
            continue
        target = mapped.get(node.value)
        if target is None and id(node) not in climbing:
            target = _relocated(node.value, directories)
        if target is not None:
            edits.append((start, end, repr(target).encode()))

    def apply(changes):
        result = raw
        for start, end, replacement in sorted(changes, reverse=True):
            result = result[:start] + replacement + result[end:]
        return result.decode()
    result = apply(edits + spliced)
    try:
        compile(result, '<input-bound-program>', 'exec', dont_inherit=True)
    except SyntaxError:
        spliced = []  # an f-string splice must never cost the whole relocation
        result = apply(edits)
        compile(result, '<input-bound-program>', 'exec', dont_inherit=True)
    return result, len(edits) + len(spliced)
