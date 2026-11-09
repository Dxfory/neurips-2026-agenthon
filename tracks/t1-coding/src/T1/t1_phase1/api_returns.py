"""Bounded installed return documentation, never a numerical correctness gate."""
import ast
import importlib
import inspect
import re


# Trusted public objects only: input text never chooses an import or attribute.
_APIS = {module + '.' + name: (module, name)
         for module in ('numpy.linalg', 'scipy.linalg')
         for name in ('lstsq', 'svd', 'eigh', 'qr')}
_NAMES = re.compile(r'(?<![\w.])[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*(?![\w.])')
_HEADER = 'Installed API documentation; not a numerical correctness check.\n'


def _aliases(text):
    aliases = {'np': 'numpy', 'numpy': 'numpy', 'scipy': 'scipy'}
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return aliases  # prose supports explicit canonical/common names only
    imported, ambiguous = {}, set()
    for node in tree.body:
        bindings = []
        if isinstance(node, ast.Import):
            bindings = [(item.asname or item.name.split('.')[0],
                         item.name if item.asname else item.name.split('.')[0]) for item in node.names]
        elif isinstance(node, ast.ImportFrom):
            if any(item.name == '*' for item in node.names):
                return {}
            if node.level or not node.module:
                ambiguous.update(item.asname or item.name for item in node.names)
            else:
                bindings = [(item.asname or item.name, node.module + '.' + item.name) for item in node.names]
        for name, target in bindings:
            if name in imported and imported[name] != target:
                ambiguous.add(name)
            imported[name] = target
    aliases.update(imported)
    # Conservative omission under any local rebinding avoids claiming the
    # installed function is the object used by a shadowed program reference.
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            ambiguous.add(node.id)
        elif isinstance(node, ast.arg):
            ambiguous.add(node.arg)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            ambiguous.add(node.name)
        elif isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name:
            ambiguous.add(node.name)
        elif isinstance(node, ast.MatchMapping) and node.rest:
            ambiguous.add(node.rest)
        elif isinstance(node, (ast.Import, ast.ImportFrom)) and node not in tree.body:
            ambiguous.update(item.asname or item.name.split('.')[0] for item in node.names)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            ambiguous.add(node.name)
        elif isinstance(node, ast.Attribute) and isinstance(node.ctx, (ast.Store, ast.Del)):
            root = node.value
            while isinstance(root, ast.Attribute):
                root = root.value
            if isinstance(root, ast.Name):
                ambiguous.add(root.id)
    return {name: target for name, target in aliases.items() if name not in ambiguous}


def _returns(doc):
    match = re.search(r'(?m)^Returns\n-{3,}\n', doc)
    if not match:
        return ''
    section = doc[match.end():]
    end = re.search(r'(?m)^[A-Za-z][^\n]*\n-{3,}\s*$', section)
    return section[:end.start()].strip() if end else section.strip()


def context(*texts):
    """Document at most four explicitly referenced APIs in at most 4000 chars.

    Imports resolve only through the fixed registry above. Never execute input,
    call the documented functions, or truncate a Returns section mid-condition.
    """
    names = []
    for text in texts:
        if not isinstance(text, str):
            continue
        aliases = _aliases(text)
        for match in _NAMES.finditer(text):
            root, *tail = match.group().split('.')
            name = '.'.join([aliases.get(root, root), *tail])
            # A removed standard alias must not regain its interpretation.
            if root in aliases and name in _APIS and name not in names:
                names.append(name)
    parts, size = [], len(_HEADER)
    for name in names:
        module, attr = _APIS[name]
        unavailable = name + ': unavailable (installed signature or Returns documentation could not be read).'
        try:
            obj = getattr(importlib.import_module(module), attr)
            signature = str(inspect.signature(obj))
            returns = _returns(inspect.getdoc(obj) or '')
            entry = name + signature + '\nReturns\n-------\n' + returns if returns else unavailable
        except Exception:
            # Optional documentation must not abort a solver request. Do not
            # expose loader exception text or invent an API return contract.
            entry = unavailable
        if size + len(entry) + 2 > 4000:
            continue
        parts.append(entry)
        size += len(entry) + 2
        if len(parts) == 4:
            break
    return _HEADER + '\n\n'.join(parts) if parts else ''
