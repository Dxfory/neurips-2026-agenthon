"""Bounded plan/skeleton and complete-component generation; no task-code execution.

READY means the assembled source may enter the isolated executor. Output coverage
and numerical correctness are checked independently after execution.
"""
from __future__ import annotations

import ast
import io
import keyword
import os
import re
import tokenize
from dataclasses import dataclass, field

from . import continuation
from .house import HouseError

PLAN_TOKENS = 2200
COMPLETE_TOKENS = 3200
MAX_REPLIES = 9
MAX_NO_PROGRESS = 2
REPAIR_REQUEST_RESERVE = 1
MAX_COMPONENTS = 3
MAX_COMPONENT_STATEMENTS = 60

OUTPUT_FORMAT = '''
SHORT PLAN AND SKELETON — first response only:
Return one python fence containing a short module docstring (at most 160 words)
with Contract (all deliverables), Semantics (units, signs, dimensions, dates,
seed and numerical conventions), and Plan (small functions, matching qfkit APIs,
major array shapes/dtypes, live-buffer cost and bounded batch size where needed).
Then write imports, shared constants, and signatures
with one-line docstrings and `...` bodies for the implementation functions.
Write a COMPLETE, SHORT main orchestrator and its __main__ guard. Every required
deliverable must have a path through this skeleton. Keep main under 25 lines;
put computation, data parsing, plots and serialization in separate helpers.
Do not put nested functions or long computation in main. Do not implement the
helper bodies yet. The next replies will fill complete functions in small
batches. Existing qfkit calls should use their actual documented signatures.
A truly small task may instead be completed within this first short response.
'''

COMPLETION = '''
Complete only the unfinished components listed below, in one python fence.
Return one or a few COMPLETE top-level functions (including their def lines),
or necessary new imports/helpers. Keep each function small; split long work
into helpers with explicit stub signatures rather than outputting an unfinished
body. You need not fill every stub in this reply. Preserve existing function
signatures, completed functions, constants, module bindings and existing top-level statements. Do not repeat
the task plan or the whole program. Do not emit prose, line continuations,
placeholder return values, or a success claim. Include an explicit entry only
when it is listed as missing. Task semantics and all deliverables still apply.
'''

RECOVERY = '''
The previous reply left an UNFINISHED statement. Its unfinished body will not
execute. Replace that entire unfinished function with a SMALL COMPLETE function
that delegates to helper stubs, and declare those helper signatures now. Keep
all verified complete functions and existing module bindings unchanged. This is
component recovery, not a rewrite of the whole program. Complete any pending
non-function statement in full. Never continue halfway through a line.
'''

STRUCTURE = '''
STRUCTURE REPAIR — do not implement the whole task yet.
The selected outer component has no usable decomposition or is too large.
Preserve its signature and replace it with a COMPLETE delegator of at most
25 lines, calling explicit top-level helper stubs. Declare those helper
signatures with one-line contracts and ... bodies. Do not nest helper functions
or fill helper bodies in this reply. Preserve all verified code and constants.
Only repairing this selected component counts as structural progress.
'''

FORMAT_RECOVERY = '''
FORMAT RECOVERY — the previous reply did not supply usable Python for this unit.
Return one python fence, with no explanatory prose outside it. If no code has
been verified, supply a SHORT skeleton: imports/constants, explicit helper
signatures with ... bodies, and a small complete orchestrator and entry.
Otherwise supply only the selected missing components. Preserve the verified
code below and all genuine unfinished Python statements; do not restart it.
'''


def _clear_prose(text):
    """Reject clear loose prose, not ambiguous or syntactically cut Python.

    Adjacent ordinary words cannot start Python syntax. Assignments, strings,
    and definition headers remain obligations even if their body is malformed.
    Fenced Python bypasses this heuristic entirely.
    """
    first = next((line.strip() for line in text.splitlines()
                  if line.strip() and not line.lstrip().startswith('#')), '')
    if re.match(r'(?i)^(?:[rubf]{0,2})[\'\"]', first) or re.match(
            r'^(?:(?:async\s+)?def|class)\s+\w*(?:\s*\(|\s*:|\s*$)', first):
        return False
    if re.match(r'^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*|\[[^\]]*\])*\s*(?:[+*/%-]?=|:(?!\s*[A-Za-z]+\s+[A-Za-z]+))', first):
        return False
    tokens = _tokens(first)
    ordinary = lambda item: (item[0] == tokenize.NAME and not keyword.iskeyword(item[1])
                             and not keyword.issoftkeyword(item[1]))
    return any(ordinary(left) and ordinary(right) for left, right in zip(tokens, tokens[1:]))


def _reply_parts(content, finish):
    """Use the existing extractor, but never create an obligation from clear prose."""
    chunk, tail, _ = continuation.absorb('', '', content, finish)
    regions = continuation._scored(content)
    chosen = continuation.code_region(content)
    fenced = any(code == chosen and rank > 0 for _, code, _, rank in regions)
    explicit = [code for _, code, _, rank in regions if rank > 0 and code.strip()]
    if tail.strip() and not fenced and _clear_prose(tail) and explicit:
        # Loose prose can outscore a fence beginning with a cut string. Keep
        # any already verified loose prefix and prefer the actual Python tail.
        prefix, tail = continuation.verified_prefix(explicit[-1], finish == 'stop')
        chunk = _merge_components(chunk, prefix) if prefix.strip() else chunk
        fenced = True
    if not chunk.strip() and not tail.strip():
        # The legacy code-looking regex does not recognize a bare cut string or
        # an explicit Python fence whose first statement has invalid syntax.
        raw = continuation._strip_think(content or '').strip()
        if explicit:
            chunk, tail = continuation.verified_prefix(explicit[-1], finish == 'stop')
            fenced = True
        elif re.match(r'(?i)^(?:[rubf]{0,2})[\'\"]', raw):
            chunk, tail = continuation.verified_prefix(raw, finish == 'stop')
    bad_format = bool(tail.strip() and not fenced and _clear_prose(tail))
    if bad_format:
        tail = ''
    return chunk, tail, bad_format or not (chunk.strip() or tail.strip())


def _feedback(items, reason, detail):
    """Bounded internal prompt feedback, never public diagnostics or code progress."""
    if items is not None and len(items) < 8:
        message = reason + ': ' + detail
        if message not in items:
            items.append(message[:700])


def _signature(node):
    header = ((tuple(ast.dump(item) for item in node.bases), tuple(ast.dump(item) for item in node.keywords))
              if isinstance(node, ast.ClassDef) else
              (ast.dump(node.args), ast.dump(node.returns) if node.returns else None))
    return header + (tuple(ast.dump(item) for item in node.decorator_list),
                     tuple(ast.dump(item) for item in getattr(node, 'type_params', [])), type(node).__name__)


def _incomplete(node):
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return any(isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and continuation.placeholder(item)
                   for item in ast.walk(node))
    if isinstance(node, ast.ClassDef):
        return continuation.placeholder(node) or any(_incomplete(item) for item in node.body)
    return False


def _literal_truth(node):
    try:
        return bool(ast.literal_eval(node))
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return None


def _has_entry(program):
    """A visible module call, excluding definitions and trivially dead branches.

    This admits execution; it does not prove dynamic reachability or task work.
    In particular, a straight-line writer may coexist with unused helper defs.
    """
    tree = ast.parse(program)
    aliases = {}
    for node in tree.body:
        if isinstance(node, ast.Import):
            aliases.update((item.asname or item.name.split('.')[0], item.name) for item in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            aliases.update((item.asname or item.name, node.module + '.' + item.name) for item in node.names)
    writes, mutated = {}, set()
    bindings = list(tree.body)
    while bindings:
        node = bindings.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            aliases.pop(node.name, None)
            writes[node.name] = writes.get(node.name, 0) + 1
        elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            aliases.pop(node.id, None)
            writes[node.id] = writes.get(node.id, 0) + 1
        elif isinstance(node, (ast.Attribute, ast.Subscript)) and isinstance(node.ctx, (ast.Store, ast.Del)):
            root = node.value
            while isinstance(root, (ast.Attribute, ast.Subscript)):
                root = root.value
            if isinstance(root, ast.Name):
                aliases.pop(root.id, None)
                mutated.add(root.id)
        else:
            bindings.extend(ast.iter_child_nodes(node))
    paths = set()

    def qualified(node):
        if isinstance(node, ast.Name):
            return 'pathlib.Path()' if node.id in paths else aliases.get(node.id, '')
        if isinstance(node, ast.Attribute):
            parent = qualified(node.value)
            return parent + '.' + node.attr if parent else ''
        if isinstance(node, ast.Call) and qualified(node.func) == 'pathlib.Path':
            return 'pathlib.Path()'
        return ''

    # A single, direct Path construction is enough to identify directory setup.
    # Reassigned or mutated objects stay unknown, as do arbitrary factory calls.
    for node in tree.body:
        targets = (node.targets if isinstance(node, ast.Assign) else
                   [node.target] if isinstance(node, ast.AnnAssign) else [])
        if (len(targets) == 1 and isinstance(targets[0], ast.Name) and
                writes.get(targets[0].id) == 1 and targets[0].id not in mutated and
                isinstance(node.value, ast.Call) and qualified(node.value.func) == 'pathlib.Path'):
            paths.add(targets[0].id)
    setup = {'matplotlib.use', 'numpy.set_printoptions', 'numpy.random.seed',
             'numpy.arange', 'numpy.linspace', 'numpy.array', 'numpy.zeros',
             'numpy.ones', 'numpy.empty', 'numpy.full', 'os.getenv', 'os.environ.get',
             'os.mkdir', 'os.makedirs', 'pathlib.Path', 'pathlib.Path().mkdir'}
    todo = list(tree.body)
    while todo:
        node = todo.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        if isinstance(node, ast.Call):
            if qualified(node.func) in setup:
                # A setup call can still contain real work in its arguments or
                # receiver construction, e.g. Path(emit_output()).mkdir().
                todo.extend(ast.iter_child_nodes(node))
                continue
            return True
        if isinstance(node, (ast.If, ast.While, ast.IfExp)):
            truth = _literal_truth(node.test)
            yes = [node.body] if isinstance(node, ast.IfExp) else node.body
            no = [node.orelse] if isinstance(node, ast.IfExp) else node.orelse
            todo.extend([node.test] + (yes if truth is True else no if truth is False else yes + no))
        elif isinstance(node, ast.BoolOp):
            for item in node.values:
                todo.append(item)
                truth = _literal_truth(item)
                if (isinstance(node.op, ast.And) and truth is False or
                        isinstance(node.op, ast.Or) and truth is True):
                    break
        else:
            todo.extend(ast.iter_child_nodes(node))
    return False


def _binding_view(program, incomplete):
    tree = ast.parse(program)
    for node in tree.body:
        if getattr(node, 'name', None) in incomplete:
            node.body = [ast.Pass()]
    return ast.unparse(ast.fix_missing_locations(tree))


def _tail_header(text):
    """Recover a complete def/class header without parsing or trusting its body."""
    lines = text.lstrip().splitlines(keepends=True)
    source = ''.join(lines)
    depth = 0
    try:
        for token in tokenize.generate_tokens(io.StringIO(source).readline):
            if token.type != tokenize.OP:
                continue
            if token.string in '([{':
                depth += 1
            elif token.string in ')]}':
                depth -= 1
            elif token.string == ':' and depth == 0:
                row, col = token.end
                header = ''.join(lines[:row - 1]) + lines[row - 1][:col]
                node = ast.parse(header + '\n    ...\n').body[0]
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    return header, node
                return None
    except (SyntaxError, ValueError, tokenize.TokenError, IndentationError):
        pass
    return None


def _tail_key(text):
    recovered = _tail_header(text)
    if recovered:
        return ('definition', recovered[1].name)
    first = text.lstrip()
    definition = re.match(r'(?:async\s+)?(?:def|class)\s+([A-Za-z_]\w*)', first)
    if definition:
        return ('definition', definition.group(1))
    # Non-definition recovery must retain the actual statement prefix. A later
    # unrelated call/assignment cannot certify the omitted work.
    return ('prefix', re.sub(r'\s+', '', first)[:300])


def _completed_keys(chunk, accepted):
    """Only components actually retained in accepted source can discharge a tail."""
    actual = {ast.dump(node) for node in ast.parse(accepted).body}
    result = []
    for node in ast.parse(chunk).body:
        if ast.dump(node) not in actual:
            continue
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            result.append(('definition', node.name))
        segment = ast.get_source_segment(chunk, node) or ''
        result.append(('prefix', re.sub(r'\s+', '', segment)))
    return result


def _tokens(source):
    result = []
    try:
        for token in tokenize.generate_tokens(io.StringIO(source).readline):
            if token.type not in (tokenize.ENDMARKER, tokenize.NEWLINE, tokenize.NL,
                                  tokenize.INDENT, tokenize.DEDENT, tokenize.COMMENT):
                result.append((token.type, token.string))
    except (tokenize.TokenError, SyntaxError, ValueError):
        pass
    return result


def _nested_completed(program):
    found = {}
    def visit(node, path):
        nested = isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        current = path + (node.name,) if nested else path
        if nested and len(current) > 1 and not _incomplete(node):
            found[current] = ast.dump(node)
        for child in ast.iter_child_nodes(node):
            visit(child, current)
    visit(ast.parse(program), ())
    return found


def _preserved(before, after):
    """Every verified non-placeholder statement and completed nested def survives."""
    expected = [ast.dump(node) for node in ast.parse(before).body if not _incomplete(node)]
    actual = iter(ast.dump(node) for node in ast.parse(after).body)
    if not all(any(value == wanted for value in actual) for wanted in expected):
        return False
    new_nested = _nested_completed(after)
    return all(new_nested.get(path) == dump for path, dump in _nested_completed(before).items())


def _merge_components(program, chunk):
    if not program.strip():
        return chunk
    # Repair merge has no full-program replacement mode. Complete loose module
    # statements (e.g. a straight-line writer) are appended individually below.
    candidate, _ = continuation.merge(program, chunk, 'repair', True)
    for node in ast.parse(chunk).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom)):
            continue
        if ast.dump(node) not in {ast.dump(item) for item in ast.parse(candidate).body}:
            text = ast.get_source_segment(chunk, node) or ''
            candidate, _ = continuation.merge(candidate, text, 'continue', False)
    return candidate


def _covered(key, complete):
    return any(kind == key[0] and (value == key[1] if kind == 'definition' else value.startswith(key[1]))
               for kind, value in complete)


@dataclass
class ProgramState:
    """Verified statements plus explicit unverified obligations, never just _runs."""
    program: str = ''
    unfinished: dict = field(default_factory=dict)
    format_error: bool = False

    @property
    def pending(self):
        stubs = [node.name for node in ast.parse(self.program).body if _incomplete(node)]
        missing = continuation.missing_calls(self.program)
        return list(dict.fromkeys(stubs + missing))

    @property
    def tail_chars(self):
        return sum(len(text) for text in self.unfinished.values())

    @property
    def phase(self):
        if self.unfinished or self.pending or self.format_error:
            return 'INCOMPLETE'
        if not self.program.strip():
            return 'EMPTY'
        if not _has_entry(self.program):
            return 'DEFINITIONS_ONLY'
        return 'READY'

    @property
    def ready(self):
        return self.phase == 'READY'

    @property
    def complete_functions(self):
        return sum(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and
                   not _incomplete(node) for node in ast.parse(self.program).body)

    def add(self, content, finish, feedback=None):
        """Merge verified blocks, protecting old source; tails never count as progress."""
        chunk, tail, self.format_error = _reply_parts(content, finish)
        if self.format_error:
            _feedback(feedback, 'response_format', 'Return a short Python skeleton or selected components in one fence.')
        old_shape = ast.dump(ast.parse(self.program))
        candidate = self.program
        complete = []
        if chunk.strip():
            old_defs = {node.name: node for node in ast.parse(self.program).body
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
            for text in self.unfinished.values():
                recovered = _tail_header(text)
                if recovered:
                    old_defs.setdefault(recovered[1].name, recovered[1])
            for node in ast.parse(chunk).body:
                old = old_defs.get(getattr(node, 'name', None))
                if (old is not None and _incomplete(old)
                        and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                        and _signature(old) != _signature(node)):
                    _feedback(feedback, 'signature_changed', f'Preserve the existing signature of {node.name}.')
                    return False
            for pending_tail in self.unfinished.values():
                if _tail_header(pending_tail) is None and _tail_key(pending_tail)[0] == 'definition':
                    name = _tail_key(pending_tail)[1]
                    for node in ast.parse(chunk).body:
                        if getattr(node, 'name', None) == name:
                            header = _tail_header(continuation._segment(chunk, node))
                            prefix = _tokens(pending_tail)
                            if header is None or _tokens(header[0])[:len(prefix)] != prefix:
                                _feedback(feedback, 'signature_changed', f'Preserve the unfinished header of {name}.')
                                return False
            candidate = _merge_components(self.program, chunk)
            incomplete = {node.name for node in ast.parse(self.program).body if _incomplete(node)}
            protected = '\n'.join(ast.get_source_segment(self.program, node) or ''
                                  for node in ast.parse(self.program).body if not _incomplete(node))
            candidate, _ = continuation.keep_complete_defs(protected, candidate)
            if continuation.completion_binding_conflicts(_binding_view(self.program, incomplete),
                                                         _binding_view(candidate, incomplete)):
                _feedback(feedback, 'binding_changed', 'Keep verified imports, constants and module bindings unchanged.')
                return False
            if not _preserved(self.program, candidate):
                _feedback(feedback, 'verified_code_changed', 'Keep completed definitions and statements unchanged.')
                return False
            complete = _completed_keys(chunk, candidate)
        changed = ast.dump(ast.parse(candidate)) != old_shape
        previous = set(self.unfinished)
        unresolved = {key: value for key, value in self.unfinished.items() if not _covered(key, complete)}
        if tail.strip():
            key = _tail_key(tail)
            # Keep bounded model context; the obligation itself survives clipping.
            unresolved.setdefault(key, tail[:5000])
        self.program, self.unfinished = candidate, unresolved
        return changed or bool(previous - set(unresolved))

    def instructions(self):
        items = [f'- {header} — {doc}' for _, header, doc in continuation.stub_defs(self.program)]
        stubs = {name for name, _, _ in continuation.stub_defs(self.program)}
        items += [f'- incomplete or missing component: {name}' for name in self.pending if name not in stubs]
        if not _has_entry(self.program):
            items.append('- missing explicit main/orchestrator and entry guard')
        for key, tail in self.unfinished.items():
            recovered = _tail_header(tail)
            name = recovered[0] if recovered else (key[1] if key[0] == 'definition' else tail.splitlines()[0][:120])
            items.append(f'- unfinished statement (preserve this header): {name}')
        return '\n'.join(items) or '- no verified implementation yet: supply a short complete skeleton'


def _calls(node, *, module=False):
    """Direct-name calls in argument/dependency order; do not enter module defs."""
    if module and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
        return []
    result = []
    for child in ast.iter_child_nodes(node):
        result.extend(_calls(child, module=module))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        result.append(node.func.id)
    return result


def _bare_entries(tree):
    """A bare run() is an outer entry; print(calc()) consumes a value leaf."""
    todo, result = list(tree.body), []
    while todo:
        node = todo.pop(0)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            if isinstance(node.value.func, ast.Name):
                result.append(node.value.func.id)
        elif isinstance(node, ast.If):
            truth = _literal_truth(node.test)
            todo[0:0] = node.body if truth is True else node.orelse if truth is False else node.body + node.orelse
    return result


def _structure_target(state, forced):
    definitions = {getattr(node, 'name', ''): node for node in ast.parse(state.program).body
                   if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
    pending = set(state.pending)
    for name in forced:
        if name in pending or ('definition', name) in state.unfinished:
            return name
    for key, tail in state.unfinished.items():
        if key[0] == 'definition' and (len(tail.splitlines()) > 25 or len(tail) > 1600):
            return key[1]
    entries = [name for name in _bare_entries(ast.parse(state.program)) if name in definitions]
    # A completed orchestration entry already provides the task's outer graph.
    if not any(not _incomplete(definitions[name]) for name in entries):
        for name in entries:
            if continuation.placeholder(definitions[name]):
                return name
    return ''


@dataclass(frozen=True)
class ComponentScope:
    names: tuple = ()
    structure: str = ''

    def instructions(self):
        selected = ', '.join(self.names) or '(only the missing entry or unfinished module statement)'
        return ('\nSELECTED COMPONENTS: ' + selected + '\nOnly these existing components may receive bodies. '
                'Other functions may only be new helper stubs. Complete at most three selected components.\n' +
                (STRUCTURE if self.structure else ''))


def _select_scope(state, forced):
    target = _structure_target(state, forced)
    if target:
        return ComponentScope((target,), target)
    tree = ast.parse(state.program)
    definitions = {getattr(node, 'name', ''): node for node in tree.body
                   if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
    pending = set(state.pending)
    pending.update(key[1] for key in state.unfinished if key[0] == 'definition')
    seen, ordered = set(), []

    def visit(name):
        if name in seen:
            return
        seen.add(name)
        if name in definitions:
            for dependency in _calls(definitions[name]):
                visit(dependency)
        if name in pending:
            ordered.append(name)

    for name in _calls(tree, module=True) + list(definitions) + sorted(pending):
        visit(name)
    return ComponentScope(tuple(ordered[:MAX_COMPONENTS]))


def _bounded_function(node, limit, prior=None):
    """Bound executable AST statements, not layout or computational complexity."""
    allowed = {id(node)}
    if isinstance(node, ast.ClassDef):
        allowed.update(id(child) for child in node.body
                       if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)))
    nested = [child for child in ast.walk(node)
              if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and child is not node]
    previous = [] if prior is None else [child for child in ast.walk(prior)
                                        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                                        and child is not prior]
    existing_nesting = (prior is not None and
                        [(child.name, _signature(child)) for child in nested] ==
                        [(child.name, _signature(child)) for child in previous])
    docstrings = {id(owner.body[0]) for owner in ast.walk(node)
                  if isinstance(owner, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                  and owner.body and isinstance(owner.body[0], ast.Expr)
                  and isinstance(owner.body[0].value, ast.Constant) and isinstance(owner.body[0].value.value, str)}
    return (sum(isinstance(child, ast.stmt) and id(child) not in docstrings for child in ast.walk(node)) <= limit and
            (existing_nesting or all(id(child) in allowed for child in nested)))


def _scoped_add(state, content, finish, scope, feedback=None):
    """Keep admissible complete components before a cutoff; reject unsafe scope atomically."""
    chunk, tail, bad_format = _reply_parts(content, finish)
    if bad_format:
        _feedback(feedback, 'response_format', 'Return a short Python skeleton or selected components in one fence.')
    if not chunk.strip() and not tail.strip():
        state.format_error = bad_format
        return False, True, []
    nodes = ast.parse(chunk).body
    existing = {getattr(node, 'name', ''): node for node in ast.parse(state.program).body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
    names, kept, rejected, oversized = set(scope.names), [], False, []
    selected = {}
    old_nodes = {ast.dump(node) for node in ast.parse(state.program).body}
    entry_missing = not _has_entry(state.program)
    skeleton_init = not scope.names and entry_missing
    for node in nodes:
        segment = continuation._segment(chunk, node) or ''
        completes_module_tail = any(key[0] == 'prefix' and
                                   re.sub(r'\s+', '', segment).startswith(key[1]) for key in state.unfinished)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            # Initial declarations become pending work; only complete bodies
            # consume this reply's body allowance. Pending work selects the
            # normal bounded component path on the following request.
            if (skeleton_init and not _incomplete(node) and node.name not in existing
                    and len(names) < MAX_COMPONENTS):
                names.add(node.name)
            if node.name in names:
                if not _incomplete(node) and not _bounded_function(
                        node, 25 if scope.structure else MAX_COMPONENT_STATEMENTS, existing.get(node.name)):
                    rejected = True
                    oversized.append(node.name)
                    _feedback(feedback, 'component_too_large',
                              f'Split {node.name} into a short delegator and top-level helper stubs; keep its signature.')
                    continue
                selected[node.name] = node
            elif node.name in existing or not continuation.placeholder(node):
                rejected = True
                _feedback(feedback, 'unselected_component', f'Only supply selected bodies; {node.name} was not selected.')
                continue
        elif (scope.names and not entry_missing and
              not isinstance(node, (ast.Import, ast.ImportFrom, ast.Assign, ast.AnnAssign)) and
              ast.dump(node) not in old_nodes and not completes_module_tail):
            rejected = True
            _feedback(feedback, 'unselected_statement', 'Do not add module execution outside the selected component.')
            continue
        if scope.names and isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            assigned = {item.id for target in targets for item in ast.walk(target) if isinstance(item, ast.Name)}
            old_assigned = {item.id for old in ast.parse(state.program).body
                            if isinstance(old, (ast.Assign, ast.AnnAssign))
                            for target in (old.targets if isinstance(old, ast.Assign) else [old.target])
                            for item in ast.walk(target) if isinstance(item, ast.Name)}
            if not assigned.intersection(old_assigned) and not completes_module_tail:
                try:
                    ast.literal_eval(node.value)
                except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
                    rejected = True
                    _feedback(feedback, 'unselected_binding', 'Put new computation inside the selected component.')
                    continue
        kept.append(node)
    required_helpers = {name for node in selected.values() for name in _calls(node)}
    related = []
    for node in kept:
        if (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and
                node.name not in names and node.name not in required_helpers and not skeleton_init):
            rejected = True
            _feedback(feedback, 'unrelated_helper', f'Declare {node.name} only if the selected component calls it.')
            continue
        related.append(node)
    kept = related
    if scope.structure:
        target = selected.get(scope.structure)
        helpers = {node.name for node in kept
                   if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                   and node.name != scope.structure and continuation.placeholder(node)}
        helpers.update(name for name, node in existing.items()
                       if name != scope.structure and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)))
        if (target is None or _incomplete(target) or not helpers.intersection(_calls(target))):
            if target is not None and not _incomplete(target):
                missing = set(continuation.missing_calls(state.program + '\n' + chunk))
                required = list(dict.fromkeys(name for name in _calls(target) if name in missing))
                if required:
                    _feedback(feedback, 'missing_helper_declarations', ', '.join(required[:12]) +
                              '. Resend the delegator with explicit top-level signatures and ... bodies for these calls.')
                else:
                    _feedback(feedback, 'missing_decomposition',
                              'Supply a short delegator calling explicit top-level helper stubs, not a whole-task body.')
            else:
                _feedback(feedback, 'incomplete_delegator',
                          'Supply the selected complete short delegator and its top-level helper stub declarations.')
            return False, True, oversized
    # A complete global conflict invalidates the transaction; do not hide it by
    # filtering globals. ProgramState enforces signatures and all old bindings.
    accepted = '\n\n'.join(continuation._segment(chunk, node) or '' for node in kept)
    if tail.strip():
        key = _tail_key(tail)
        if key[0] == 'definition' and key[1] not in names:
            rejected = True
            _feedback(feedback, 'unselected_tail', f'Only recover selected components; {key[1]} was not selected.')
            tail = ''
    old_pending = set(state.pending)
    old_shape = ast.dump(ast.parse(state.program))
    clone = ProgramState(state.program, dict(state.unfinished), bad_format)
    # Verified prefix is already certified by absorb; do not mark its last node
    # as cut a second time. The original unverified tail remains an obligation.
    progress = clone.add(accepted, 'stop', feedback) if accepted.strip() else False
    clone.format_error = bad_format
    if tail.strip():
        key = _tail_key(tail)
        clone.unfinished.setdefault(key, tail[:5000])
    if scope.structure and scope.structure in clone.pending:
        _feedback(feedback, 'unaccepted_delegator', 'The selected delegator was not accepted; preserve its old contract.')
        return False, True, oversized
    # Imports/stub churn without completing a selected component cannot prolong
    # a body-completion loop. Entry-only recovery still uses normal AST progress.
    completed = old_pending - set(clone.pending)
    if names and not completed and not scope.structure and ast.dump(ast.parse(clone.program)) != old_shape:
        if not any(name in selected and not _incomplete(selected[name]) for name in names):
            clone.program = state.program
            progress = False
    state.program, state.unfinished, state.format_error = clone.program, clone.unfinished, clone.format_error
    return progress, rejected, oversized


def _record(state, status):
    status.update(minimal_state=state.phase, minimal_ready=state.ready,
                  minimal_complete_functions=state.complete_functions,
                  minimal_pending_count=len(state.pending) + len(state.unfinished) + int(not _has_entry(state.program)),
                  minimal_tail_chars=state.tail_chars)


def _plan_thinking():
    """Opt-in experiment: a first-plan budget, never a component-generation setting."""
    value = os.environ.get('T1_MINIMAL_PLAN_THINK_BUDGET', '0')
    if not re.fullmatch(r'[0-9]{1,4}', value) or int(value) > 1024:
        raise HouseError('T1_MINIMAL_PLAN_THINK_BUDGET must be an integer from 0 through 1024',
                         kind='local_input', code='invalid_minimal_plan_think_budget')
    return int(value) or False


def generate(client, messages, deadline, status, *, canaries=(), pace=None, sleep=None):
    """One shared plan/skeleton then complete batches, under existing House gates."""
    from . import baseline
    pace = pace or baseline.Pace()
    state = ProgramState()
    no_progress = 0
    forced = []
    feedback = []
    status.update(minimal_plan_requests=0, minimal_completion_requests=0,
                  minimal_recovery_requests=0, minimal_no_progress=0,
                  minimal_structure_requests=0, minimal_scope_rejections=0,
                  minimal_structure_pending=0, minimal_selected_components=0)
    _record(state, status)
    plan_thinking = _plan_thinking()
    stop = 'reply_limit'
    for index in range(MAX_REPLIES):
        scope = _select_scope(state, forced) if index else ComponentScope()
        status['minimal_structure_pending'] = int(bool(scope.structure))
        status['minimal_selected_components'] = len(scope.names)
        recovering = bool(state.unfinished) or bool(scope.structure) or state.format_error or bool(feedback)
        kind = 'minimal_plan' if index == 0 else ('minimal_recover' if recovering else 'minimal_complete')
        counter = {'minimal_plan': 'minimal_plan_requests', 'minimal_recover': 'minimal_recovery_requests',
                   'minimal_complete': 'minimal_completion_requests'}[kind]
        tokens = PLAN_TOKENS if index == 0 else COMPLETE_TOKENS
        ask = list(messages)
        if index:
            ask += [{'role': 'assistant', 'content': baseline._fence(state.program)},
                    {'role': 'user', 'content': baseline.withhold_canaries(
                        COMPLETION + (RECOVERY if state.unfinished or scope.structure else '') +
                        (FORMAT_RECOVERY if state.format_error else '') + '\nREMAINING WORK:\n' +
                        state.instructions() + scope.instructions() +
                        ('\nPREVIOUS REPLY FEEDBACK:\n' + '\n'.join(feedback)[:2400] if feedback else ''), canaries)}]
        # Execution and one local repair retain their own request/time resources.
        # A later completion is useful only if its whole bounded reply can arrive.
        need = tokens / pace.rate + 6
        reserve = baseline.EXEC_FLOOR + pace.patch()
        before_attempts = client.attempts
        try:
            reply = baseline._request(client, ask, deadline, pace, kind, sleep,
                                      need=need, reserve=reserve, keep=REPAIR_REQUEST_RESERVE,
                                      expect=need, max_tokens=tokens,
                                      thinking=plan_thinking if kind == 'minimal_plan' else False, once=True)
        except baseline._Stop as exc:
            stop = exc.reason
            break
        except HouseError:
            if client.attempts > before_attempts:
                baseline._count(status, counter)
                if scope.structure:
                    baseline._count(status, 'minimal_structure_requests')
            status.update(minimal_state='STOPPED', minimal_ready=False, minimal_stop_reason='provider_error',
                          stop_reason='request_failed')
            raise
        baseline._count(status, counter)
        if scope.structure:
            baseline._count(status, 'minimal_structure_requests')
        baseline._count(status, 'generation_replies')
        pace.measure(client, reply)
        content = baseline.withhold_canaries(baseline._text(reply), canaries)
        feedback = []
        if index:
            progress, rejected, oversized = _scoped_add(state, content, reply['finish_reason'], scope, feedback)
            if rejected:
                baseline._count(status, 'minimal_scope_rejections')
            forced.extend(name for name in oversized if name not in forced)
        else:
            progress = state.add(content, reply['finish_reason'], feedback)
        _record(state, status)
        status['minimal_structure_pending'] = int(bool(_structure_target(state, forced)))
        no_progress = 0 if progress else no_progress + 1
        status['minimal_no_progress'] = no_progress
        if state.ready:
            status['stop_reason'] = 'complete'
            return state.program
        if no_progress >= MAX_NO_PROGRESS:
            stop = 'no_progress'
            break
    status.update(minimal_state='STOPPED', minimal_ready=False, minimal_stop_reason=stop,
                  stop_reason={'reply_limit': 'chunk_cap', 'incomplete_code': 'invalid_code'}.get(stop, stop))
    raise HouseError('Program generation stopped with unfinished work', kind='model_output', code='incomplete_code')
