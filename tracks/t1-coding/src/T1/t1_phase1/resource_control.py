"""Bounded cgroup observations and advisory allocation locations; never execute code.

An OOM counter belongs to the memory cgroup, not to a particular Python line.
Static element-storage estimates exclude temporary arrays, aliases and libraries.
Neither is an execution-admission gate or a reason to change task parameters.
"""
from __future__ import annotations

import ast
import math
from collections import Counter
from pathlib import Path, PurePosixPath

MAX_NUMBER = 2 ** 60
MEMORY_FIELDS = ('memory_limit_bytes', 'memory_current_before_bytes', 'memory_current_after_bytes',
                 'memory_oom_kill_delta')
MEMORY_EVENTS = ('cgroup_oom_observed', 'signal_killed_unknown', 'controller_timeout',
                 'no_oom_observed', 'unknown')


def _number(value, positive=False):
    return type(value) is int and int(positive) <= value < MAX_NUMBER


def _read(path, maximum=4096):
    try:
        with path.open('r', encoding='ascii') as stream:
            text = stream.read(maximum + 1)
        return text if len(text) <= maximum else ''
    except (OSError, UnicodeError):
        return ''


def _integer(text, positive=False):
    text = text.strip()
    if not text.isascii() or not text.isdecimal() or len(text) > 19:
        return None
    value = int(text)
    return value if _number(value, positive) else None


def cgroup_directory(membership, mountinfo):
    """Resolve self membership relative to the mount root, without parent fallback."""
    members = []
    for line in membership.splitlines():
        parts = line.split(':', 2)
        if len(parts) == 3 and (parts[1] == '' or 'memory' in parts[1].split(',')):
            members.append((parts[1] == '', PurePosixPath(parts[2])))
    for line in mountinfo.splitlines():
        parts = line.split(' - ', 1)
        if len(parts) != 2:
            continue
        left, right = parts[0].split(), parts[1].split()
        if len(left) < 5 or len(right) < 3:
            continue
        kind, mount_root, mount_at = right[0], PurePosixPath(left[3]), PurePosixPath(left[4])
        if kind not in ('cgroup', 'cgroup2') or (kind == 'cgroup' and 'memory' not in right[2].split(',')):
            continue
        if not mount_at.is_relative_to('/sys/fs/cgroup'):
            continue
        for unified, member in members:
            if unified != (kind == 'cgroup2') or not member.is_absolute():
                continue
            if '..' in member.parts or '..' in mount_root.parts or '..' in mount_at.parts:
                continue
            try:
                relative = member.relative_to(mount_root)
            except ValueError:
                continue
            return Path(str(mount_at / relative))
    return None


def snapshot(root=None):
    """Read the current process's memory cgroup; an explicit root is already scoped."""
    result = {'source': 'unknown', 'limit_bytes': None, 'current_bytes': None, 'oom_kill': None}
    if root is None:
        root = cgroup_directory(_read(Path('/proc/self/cgroup')),
                                _read(Path('/proc/self/mountinfo'), 65536))
    if root is None:
        return result
    root = Path(root)
    if (root / 'memory.max').is_file():
        result['source'] = 'cgroup_v2'
        limit, current, events = 'memory.max', 'memory.current', 'memory.events'
    else:
        if (root / 'memory' / 'memory.limit_in_bytes').is_file():
            root = root / 'memory'
        if not (root / 'memory.limit_in_bytes').is_file():
            return result
        result['source'] = 'cgroup_v1'
        limit, current, events = 'memory.limit_in_bytes', 'memory.usage_in_bytes', 'memory.oom_control'
    result['limit_bytes'] = _integer(_read(root / limit), positive=True)
    result['current_bytes'] = _integer(_read(root / current))
    kills = [line.split()[1:] for line in _read(root / events).splitlines()
             if line.split()[:1] == ['oom_kill']]
    if len(kills) == 1 and len(kills[0]) == 1:
        result['oom_kill'] = _integer(kills[0][0])
    return result


def execution(before, after, exit_code, *, timed_out=False):
    """Record a cgroup event during this run, with no inferred process peak or line."""
    old, new = before.get('oom_kill'), after.get('oom_kill')
    delta = (new - old if before.get('source') == after.get('source') != 'unknown'
             and _number(old) and _number(new) and new >= old else None)
    event = ('controller_timeout' if timed_out else 'cgroup_oom_observed' if delta else
             'signal_killed_unknown' if exit_code in (-9, 137) else
             'no_oom_observed' if delta == 0 else 'unknown')
    limit = after.get('limit_bytes')
    if not _number(limit, True):
        limit = before.get('limit_bytes')
    return {'memory_event': event, 'memory_limit_bytes': limit if _number(limit, True) else None,
            'memory_current_before_bytes': before.get('current_bytes')
            if _number(before.get('current_bytes')) else None,
            'memory_current_after_bytes': after.get('current_bytes')
            if _number(after.get('current_bytes')) else None,
            'memory_oom_kill_delta': delta}


def context(observed):
    limit = observed.get('limit_bytes')
    ceiling = f'{limit} bytes (observed cgroup ceiling)' if _number(limit, True) else 'unknown'
    return ('\nRESOURCE CONTRACT: memory limit = ' + ceiling + '. This is shared with the controller and '
            'libraries, not a free array budget. In the short Plan state major array shapes, dtype, simultaneous '
            'live buffers and batch size; include expression temporaries and views retaining their base. '
            'Prefer batches with small scratch arrays (about 32 MiB where possible) and mergeable statistics, '
            'not whole-dataset allocation followed by slicing. Inspect EVERY computation path and caller. '
            'Keep the required sample count, time grid, method, seed, precision and output schema. Maintain '
            'the original random draw order and common draws for paired comparisons; do not reseed each batch. '
            'For antithetic or dependent samples first form the correct independent paired observations; '
            'do not average batch standard errors. When the TASK permits qfkit, qfkit.batch_slices and '
            'qfkit.OnlineMoments provide generic '
            'batch indexing and mergeable vector moments; follow their documented signatures. Use terminal '
            'or running sufficient state only when the task permits mathematically equivalent computation.\n')


def _literal(node, values):
    """Tiny bounded integer arithmetic, not eval or a general shape interpreter."""
    if isinstance(node, ast.Constant):
        value = node.value
    elif isinstance(node, ast.Name):
        value = values.get(node.id)
    elif isinstance(node, ast.BinOp):
        left, right = _literal(node.left, values), _literal(node.right, values)
        if left is None or right is None:
            return None
        if isinstance(node.op, ast.Add):
            value = left + right
        elif isinstance(node.op, ast.Mult):
            value = left * right
        elif isinstance(node.op, ast.FloorDiv) and right:
            value = left // right
        elif isinstance(node.op, ast.Pow) and 0 <= right <= 8:
            value = left ** right
        else:
            return None
    else:
        return None
    return value if type(value) is int and 0 <= value <= 2 ** 48 else None


def allocation_sites(code, limit=12):
    """Locate possible numeric allocations, including unresolved shapes.

    A method spelling is a candidate, not proof of NumPy dispatch or reachability.
    The estimate assumes that dispatch and its default float64 unless an explicit
    recognized dtype is supplied. All sites remain advisory, never a code gate.
    """
    if not isinstance(code, str) or len(code) > 300000:
        return []
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, RecursionError):
        return []
    bindings = Counter(node.id for node in ast.walk(tree)
                       if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bindings[node.name] += 1
        elif isinstance(node, ast.alias):
            bindings[node.asname or node.name.split('.')[0]] += 1
    constants = {}
    for statement in tree.body:
        if isinstance(statement, ast.Assign):
            for target in statement.targets:
                if isinstance(target, ast.Name) and bindings[target.id] == 1:
                    constants[target.id] = _literal(statement.value, constants)
        elif isinstance(statement, ast.AugAssign) and isinstance(statement.target, ast.Name):
            constants[statement.target.id] = None
    sites = []
    sizes = {'float64': 8, 'float32': 4, 'int64': 8, 'int32': 4, 'bool': 1, 'float': 8}

    class Scan(ast.NodeVisitor):
        function = '<module>'
        values = constants

        def visit_FunctionDef(self, node):
            previous = self.function, self.values
            self.function = node.name
            self.values = dict(self.values)
            for item in ast.walk(node):
                if isinstance(item, ast.arg):
                    self.values.pop(item.arg, None)
                elif isinstance(item, ast.Name) and isinstance(item.ctx, ast.Store):
                    self.values.pop(item.id, None)
            for item in node.body:
                self.visit(item)
            self.function, self.values = previous

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_Call(self, node):
            name = node.func.attr if isinstance(node.func, ast.Attribute) else (
                node.func.id if isinstance(node.func, ast.Name) else '')
            if name in ('zeros', 'ones', 'empty', 'full', 'normal', 'standard_normal'):
                args = {item.arg: item.value for item in node.keywords if item.arg is not None}
                index = 2 if name == 'normal' else 0
                shape = args.get('size' if name in ('normal', 'standard_normal') else 'shape')
                if shape is None and len(node.args) > index:
                    shape = node.args[index]
                if shape is not None:
                    dims = shape.elts if isinstance(shape, (ast.Tuple, ast.List)) else [shape]
                    values = [_literal(item, self.values) for item in dims] if len(dims) <= 8 else [None]
                    dtype = args.get('dtype')
                    if dtype is None and name in ('zeros', 'ones', 'empty') and len(node.args) > 1:
                        dtype = node.args[1]
                    if dtype is None and name == 'full' and len(node.args) > 2:
                        dtype = node.args[2]
                    typename = (dtype.attr if isinstance(dtype, ast.Attribute) else dtype.id
                                if isinstance(dtype, ast.Name) else dtype.value
                                if isinstance(dtype, ast.Constant) else None)
                    size = 8 if dtype is None else sizes.get(typename) if isinstance(typename, str) else None
                    if name == 'full' and dtype is None:  # inferred from fill_value, not default float64
                        size = None
                    estimate = math.prod(values) * size if size and None not in values else None
                    if estimate is not None and estimate >= MAX_NUMBER:
                        estimate = None
                    sites.append({'function': self.function, 'line': node.lineno,
                                  'expression': ast.unparse(node)[:140], 'estimated_bytes': estimate})
            self.generic_visit(node)

    try:
        Scan().visit(tree)
    except (RecursionError, ValueError):
        return []
    return sites[:max(0, min(limit, 20))] if type(limit) is int else []


def repair_context(code, record):
    event = record.get('memory_event')
    if event == 'cgroup_oom_observed':
        cause = ('The memory cgroup recorded an OOM kill during this execution; '
                 'the exact killed line is unknown. ')
    else:
        cause = 'The program was killed, but its cause is unknown; signal 9 alone does not establish OOM. '
    limit = record.get('memory_limit_bytes')
    if _number(limit, True):
        cause += f'Observed memory ceiling: {limit} bytes. '
    lines = []
    for site in allocation_sites(code):
        amount = (f"assumed element storage {site['estimated_bytes']} bytes"
                  if site['estimated_bytes'] is not None else 'size unresolved')
        lines.append(f"- line {site['line']} in {site['function']}: {site['expression']} ({amount})")
    return ('\nRESOURCE FAILURE EVIDENCE: ' + cause + '\n'
            'Allocation candidates across the current program (not proven executed sites; assumes numeric '
            'dispatch/default float64; not a measured peak):\n' + ('\n'.join(lines) or '- none resolved') + '\n'
            'Fix the relevant allocating functions AND their callers. A terminal-column view can retain the '
            'full matrix; changing only a helper does not change separate inline allocations. Use bounded '
            'batches and aggregate sufficient state; release intermediates before later allocations. Preserve '
            'required sample count, time grid, seed/draw order, paired randomness, estimator and precision. '
            'Do not repeat an unchanged helper or weaken task checks.\n')
