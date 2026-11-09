"""Static, version-bound replacement of one controller-selected top-level unit.

A worker line number is an untrusted diagnostic hint, not proof of the cause.
This module limits edit scope; it neither executes code nor validates mathematics.
"""
import ast
from dataclasses import dataclass
import hashlib


_DEFINITIONS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


@dataclass(frozen=True)
class Target:
    version: str
    line: int
    start: int
    end: int
    kind: str
    name: str | None
    source: str


def _tree(source):
    compile(source, '<repair-unit>', 'exec', dont_inherit=True)
    return ast.parse(source)


def _span(node):
    return (min([node.lineno] + [d.lineno for d in getattr(node, 'decorator_list', [])]),
            node.end_lineno)


def _lines(source):
    # Python line numbers count physical newlines, not Unicode string separators.
    parts = source.split('\n')
    return [part + '\n' for part in parts[:-1]] + ([parts[-1]] if parts[-1] else [])


def select(program, *, base_version, executed_version, line):
    """Return one unambiguous complete unit in the exact executed version."""
    if (not isinstance(program, str) or not program or '\r' in program
            or type(line) is not int or line < 1):
        return None
    version = hashlib.sha256(program.encode()).hexdigest()
    if base_version != version or executed_version != version:
        return None
    try:
        nodes = _tree(program).body
        matches = [node for node in nodes if _span(node)[0] <= line <= _span(node)[1]]
        if len(matches) != 1:
            return None
        node = matches[0]
        start, end = _span(node)
        # Replacing complete physical lines must not touch a neighboring statement.
        if node.col_offset or any(other is not node and
                                 _span(other)[0] <= end and _span(other)[1] >= start
                                 for other in nodes):
            return None
        source = ''.join(_lines(program)[start-1:end])
        if len(_tree(source).body) != 1:
            return None
    except (SyntaxError, ValueError, TypeError, RecursionError, MemoryError, OverflowError):
        return None
    return Target(version, line, start, end, type(node).__name__,
                  node.name if isinstance(node, _DEFINITIONS) else None, source)


def apply(program, target, replacement):
    """Compose one complete unit, preserving every byte outside its bound span."""
    if not isinstance(target, Target) or select(
            program, base_version=target.version, executed_version=target.version,
            line=target.line) != target:
        raise ValueError('repair_stale_target')
    if not isinstance(replacement, str) or not replacement.strip() or '\r' in replacement:
        raise ValueError('repair_invalid_code')
    try:
        incoming = _tree(replacement).body
        if len(incoming) != 1:
            raise ValueError('repair_one_unit_required')
        node = incoming[0]
        if type(node).__name__ != target.kind:
            raise ValueError('repair_wrong_unit_kind')
        if isinstance(node, _DEFINITIONS) and node.name != target.name:
            raise ValueError('repair_wrong_definition_name')
        if isinstance(node, ast.Assert) and isinstance(node.test, ast.Constant):
            raise ValueError('repair_empty_assertion')
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            raise ValueError('repair_empty_expression')
        lines = _lines(program)
        source = replacement.rstrip('\n') + '\n'
        # Retain a missing final newline for callers that supply such a source.
        if target.end == len(lines) and not program.endswith('\n'):
            source = source[:-1]
        candidate = ''.join(lines[:target.start-1]) + source + ''.join(lines[target.end:])
        _tree(candidate)
        return candidate
    except (SyntaxError, TypeError, RecursionError, MemoryError, OverflowError) as exc:
        raise ValueError('repair_invalid_code') from exc
