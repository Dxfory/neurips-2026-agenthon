"""Small, bounded recipes over fixed numerical components; no Python evaluation.

Validation/rendering is host-safe. Only run_recipe performs numerical work and
writes outputs, and it belongs in the existing isolated worker. This is neither
a Python sandbox nor a financial verifier: the worker's resource/network/file
isolation and the unchanged task checker remain required.
"""
from __future__ import annotations

import ast
import csv
import io
import inspect
import json
import keyword
import math
import operator
import os
from pathlib import Path
import re
import stat


class RecipeError(ValueError):
    """An unsupported, malformed or failed recipe operation."""


_MAX_BYTES = 120_000
_MAX_STEPS = 1000
_MAX_DEPTH = 16
_MAX_EXPR_NODES = 256
_MAX_WORK = 20_000
_MAX_ITEMS = 100_000
_MAX_ARRAY = 8_000_000
_ROOTS = {'INPUT_ROOT', 'OUTPUT_ROOT'}
_NAME = re.compile(r'[A-Za-z][A-Za-z0-9_]{0,63}\Z')
_SIGNATURES = {
    'qfkit.bsm_price': '(S,K,T,r,sigma,*,kind,q=0) -> price; European analytic price',
    'qfkit.bsm_greeks': '(S,K,T,r,sigma,*,kind,q=0,vega_per="unit",theta_per="year",rho_per="unit") -> price/delta/gamma/vega/theta/rho; inputs broadcast; vega/rho unit or percent, theta year/calendar_day/trading_day',
    'qfkit.gbm_greek_samples': '(S,K,T,r,sigma,*,q,kind,averaging,normals,theta) -> price/pathwise/lr arrays; averaging terminal|arithmetic (excludes initial spot), normals (paths,steps); theta calendar=-dV/dT or maturity=dV/dT; unit vol/rate, annual theta; unavailable estimates None',
    'qfkit.gbm_fd_greek_samples': '(S,K,T,r,sigma,*,q,kind,averaging,normals,spot_bump,vol_bump,time_bump,rate_bump,theta) -> five central common-random-number Greek arrays; positive absolute bumps; unit vol/rate, annual theta',
    'qfkit.mc_estimate': '(samples,*,pair_antithetic) -> value,variance,se,n_independent,n_samples; ddof=1; pair_antithetic=True pairs first/second halves',
    'qfkit.batch_slices': '(total,*,batch_size) -> slices covering exactly total; no sampling/reseeding',
    'slice.start': '(part) -> integer start of a bounded slice',
    'slice.stop': '(part) -> integer exclusive stop of a bounded slice',
    'slice.size': '(part) -> number of selected indices; use for the final partial batch',
    'qfkit.OnlineMoments': '() -> mutable moment handle; rows are observations, columns features',
    'rng.RandomState': '(seed) -> persistent numpy RandomState handle',
    'rng.standard_normal': '(rng,size) -> draws; size integer or list of dimensions; advances this handle only',
    'moments.update': '(moments,samples) -> same handle; feed paired differences before aggregation; do not average block SEs',
    'moments.result': '(moments,*,ddof) -> n,mean,variance,covariance,se; array results, explicit ddof',
    'np.array': '(values,dtype=None) -> numeric/bool array; dtype float64|int64|bool only',
    'np.linspace': '(start,stop,num=50,endpoint=True) -> float64 vector',
    'np.arange': '(start,stop=None,step=1) -> numeric vector',
    'np.meshgrid': '(*vectors,indexing="xy") -> coordinate matrices; indexing xy|ij',
    'np.zeros': '(shape) -> float64 array',
    'np.ones': '(shape) -> float64 array',
    'np.mean': '(a,axis=None,keepdims=False) -> mean',
    'np.sum': '(a,axis=None,keepdims=False) -> sum',
    'np.std': '(a,axis=None,ddof=0,keepdims=False) -> standard deviation',
    'np.var': '(a,axis=None,ddof=0,keepdims=False) -> variance',
    'np.sqrt': '(x)', 'np.exp': '(x)', 'np.log': '(x)', 'np.abs': '(x)',
    'np.maximum': '(x,y)', 'np.minimum': '(x,y)', 'np.where': '(condition,x,y)',
    'np.all': '(a,axis=None)', 'np.any': '(a,axis=None)', 'np.isfinite': '(x)',
    'np.allclose': '(a,b,rtol=1e-5,atol=1e-8) -> bool',
    'np.column_stack': '(arrays)', 'np.stack': '(arrays,axis=0)',
    'np.concatenate': '(arrays,axis=0)', 'np.reshape': '(a,shape)',
    'np.shape': '(a) -> tuple of dimensions', 'len': '(value) -> integer length',
}


def allowed_functions(*, allow_library=True):
    if type(allow_library) is not bool:
        raise RecipeError('allow_library must be boolean')
    return tuple(name for name in _SIGNATURES if allow_library or not name.startswith(('qfkit.', 'moments.')))


def catalog(*, allow_library=True):
    return '''Constrained component recipe: one JSON object {"steps":[...],"basis":"optional short source-based explanation"}.
Each step has op:
 let: name,value. call: name,fn,args optional list,kwargs optional object.
 for: var,over,body (list of steps); loop bindings stay local; append/moments mutate shared handles.
 append: target (existing list name),value. assert: condition,message optional string.
 write: format,path,value. json: finite JSON; csv: records plus required ordered columns list.
 text: string. png: value={X,Y,Z} equal-shaped 2D grids OR records with x,y,z field names;
 PNG record grid must be complete with no duplicates; optional title,xlabel,ylabel,zlabel strings.
 print: value (bounded diagnostic only, never a verdict).
Values are JSON literals recursively, {"ref":"name"}, or {"expr":"expression"}.
Only SINGLE-KEY ref/expr objects are special; a business object with that shape is unsupported.
Plain strings are literal. Expressions allow names, constants, indexing/slicing, arithmetic,
comparisons, boolean operators, ternaries, list/tuple/dict. NO calls, attributes, imports or comprehensions.
Array indexing supports integers/slices/None, or one standalone 1D integer/boolean index;
multiple/multidimensional advanced indices are unsupported.
Names cannot be private or contain double underscores. INPUT_ROOT/OUTPUT_ROOT are read-only strings;
there are no input-reading operations in this pilot. Call names below are the complete whitelist.
Use a prior call for every function, then ref/expr its named result. No dotted method expressions.
Output paths must be relative, without dot segments or symlinks. Grading/controller filenames are reserved.
Every output requires actual values. slice.start/stop/size read bounded batch slices; attributes remain forbidden.
No automatic financial convention, task parameter, tolerance, RNG schedule, missing output or method choice.
Limits: 120000 recipe bytes, 1000 static steps, nesting 16, 256 nodes per expression,
20000 total executed steps/loop iterations, 100000 list/serialized items, 8000000 elements per array.
Every run starts a fresh recipe; only handles shared within that run persist.
''' + '\n'.join(name + _SIGNATURES[name] for name in allowed_functions(allow_library=allow_library))


def _name(value):
    if type(value) is not str or not _NAME.fullmatch(value) or '__' in value or keyword.iskeyword(value):
        raise RecipeError('invalid/private variable name')
    return value


_BINARY = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
           ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
           ast.Mod: operator.mod, ast.Pow: operator.pow}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg, ast.Not: operator.not_}
_COMPARE = {ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt,
            ast.LtE: operator.le, ast.Gt: operator.gt, ast.GtE: operator.ge,
            ast.In: lambda a, b: a in b, ast.NotIn: lambda a, b: a not in b}
_AST_ALLOWED = (ast.Expression, ast.Constant, ast.Name, ast.Load, ast.Subscript, ast.Slice,
                ast.List, ast.Tuple, ast.Dict, ast.BinOp, ast.UnaryOp, ast.BoolOp,
                ast.Compare, ast.IfExp, ast.And, ast.Or, *_BINARY, *_UNARY, *_COMPARE)


def _expression(value, names):
    if type(value) is not str or not value or len(value) > 4096:
        raise RecipeError('expression must be a bounded nonempty string')
    try:
        tree = ast.parse(value, mode='eval')
    except (SyntaxError, ValueError, RecursionError) as exc:
        raise RecipeError('invalid expression syntax') from exc
    nodes = list(ast.walk(tree))
    if len(nodes) > _MAX_EXPR_NODES:
        raise RecipeError('expression node limit')
    for node in nodes:
        if not isinstance(node, _AST_ALLOWED):
            raise RecipeError('unsupported expression node: ' + type(node).__name__)
        if isinstance(node, ast.Name) and (_name(node.id) not in names):
            raise RecipeError('unknown/forward reference: ' + node.id)
        if isinstance(node, ast.Constant):
            _literal(node.value)
        if isinstance(node, ast.Dict) and any(key is None for key in node.keys):
            raise RecipeError('dictionary unpacking is unsupported')
    return tree.body


def _literal(value):
    if value is None or type(value) in (bool, str):
        return
    if type(value) is int and abs(value) <= 2 ** 63 - 1:
        return
    if type(value) is float and math.isfinite(value):
        return
    raise RecipeError('only finite JSON scalar values are supported')


def _value(value, names, depth=0):
    if depth > _MAX_DEPTH:
        raise RecipeError('value nesting limit')
    if type(value) is dict:
        if set(value) == {'ref'}:
            if _name(value['ref']) not in names:
                raise RecipeError('unknown/forward reference: ' + value['ref'])
        elif set(value) == {'expr'}:
            _expression(value['expr'], names)
        else:
            if any(type(key) is not str for key in value):
                raise RecipeError('object keys must be strings')
            for child in value.values():
                _value(child, names, depth + 1)
    elif type(value) is list:
        if len(value) > _MAX_ITEMS:
            raise RecipeError('list size limit')
        for child in value:
            _value(child, names, depth + 1)
    else:
        _literal(value)


_FIELDS = {
    'let': ({'op', 'name', 'value'}, set()),
    'call': ({'op', 'name', 'fn'}, {'args', 'kwargs'}),
    'for': ({'op', 'var', 'over', 'body'}, set()),
    'append': ({'op', 'target', 'value'}, set()),
    'assert': ({'op', 'condition'}, {'message'}),
    'write': ({'op', 'format', 'path', 'value'}, {'columns', 'x', 'y', 'z', 'title', 'xlabel', 'ylabel', 'zlabel'}),
    'print': ({'op', 'value'}, set()),
}


def _validate_steps(steps, names, count, allowed, depth=0, prefix='steps'):
    if type(steps) is not list or depth > _MAX_DEPTH:
        raise RecipeError('steps must be a list within nesting limit')
    for index, step in enumerate(steps):
        where = f'{prefix}[{index}]'
        try:
            count[0] += 1
            if count[0] > _MAX_STEPS:
                raise RecipeError('static step limit')
            if type(step) is not dict or type(step.get('op')) is not str or step['op'] not in _FIELDS:
                raise RecipeError('unsupported step operation')
            op = step['op']
            required, optional = _FIELDS[op]
            if not required <= set(step) or set(step) - required - optional:
                raise RecipeError('missing or unknown step fields')
            if op in ('let', 'call'):
                name = _name(step['name'])
                if name in _ROOTS:
                    raise RecipeError('root names are read-only')
            if op in ('let', 'append', 'print', 'write'):
                _value(step['value'], names)
            if op == 'call':
                if type(step['fn']) is not str or step['fn'] not in allowed:
                    raise RecipeError('function is not in the whitelist')
                if type(step.get('args', [])) is not list or type(step.get('kwargs', {})) is not dict:
                    raise RecipeError('call args/kwargs require list/object')
                _value(step.get('args', []), names)
                for key, value in step.get('kwargs', {}).items():
                    _name(key)
                    _value(value, names)
            elif op == 'for':
                variable = _name(step['var'])
                if variable in names:
                    raise RecipeError('loop variable cannot shadow an existing name')
                _value(step['over'], names)
                _validate_steps(step['body'], names | {variable}, count, allowed, depth + 1, where + '.body')
            elif op == 'append':
                if _name(step['target']) not in names:
                    raise RecipeError('unknown append target')
            elif op == 'assert':
                _value(step['condition'], names)
                if 'message' in step and type(step['message']) is not str:
                    raise RecipeError('assert message must be a literal string')
            elif op == 'write':
                fmt = step['format']
                if fmt not in ('json', 'csv', 'text', 'png'):
                    raise RecipeError('unsupported output format')
                _value(step['path'], names)
                if type(step['path']) is str:
                    _relative_path(step['path'])
                extra = set(step) - required
                if fmt == 'csv':
                    columns = step.get('columns')
                    if (type(columns) is not list or not columns or any(type(c) is not str or not c for c in columns)
                            or len(set(columns)) != len(columns) or extra - {'columns'}):
                        raise RecipeError('CSV needs unique ordered columns and no other format options')
                elif fmt == 'png':
                    xyz = {'x', 'y', 'z'} & extra
                    if xyz and xyz != {'x', 'y', 'z'}:
                        raise RecipeError('PNG records need all x/y/z field names')
                    if extra - {'x', 'y', 'z', 'title', 'xlabel', 'ylabel', 'zlabel'}:
                        raise RecipeError('unsupported PNG fields')
                    if any(type(step[key]) is not str for key in extra):
                        raise RecipeError('PNG field names and labels must be strings')
                elif extra:
                    raise RecipeError('unsupported format options')
            if op in ('let', 'call'):
                names.add(name)
        except RecipeError as exc:
            raise RecipeError(where + ': ' + str(exc)) from exc


def validate_recipe(recipe, *, allow_library=True):
    """Reject unsupported syntax and unknown references without executing a call."""
    try:
        if type(recipe) is not dict or set(recipe) - {'steps', 'basis'} or 'steps' not in recipe:
            raise RecipeError('recipe requires steps and optional basis only')
        if 'basis' in recipe and (type(recipe['basis']) is not str or len(recipe['basis']) > 12_000):
            raise RecipeError('basis must be a bounded string')
        if len(json.dumps(recipe, allow_nan=False).encode('utf-8')) > _MAX_BYTES:
            raise RecipeError('recipe byte limit')
        _validate_steps(recipe['steps'], set(_ROOTS), [0], allowed_functions(allow_library=allow_library))
    except (TypeError, ValueError, RecursionError) as exc:
        if isinstance(exc, RecipeError):
            raise
        raise RecipeError('recipe must be bounded finite JSON') from exc


def render_program(recipe, input_root: str, output_root: str, *, allow_library=True):
    """Only embed validated literal data; execution remains with the worker."""
    validate_recipe(recipe, allow_library=allow_library)
    if type(input_root) is not str or type(output_root) is not str:
        raise RecipeError('root paths must be strings')
    return ('from t1_component_recipe import run_recipe\n'
            f'run_recipe({recipe!r}, {input_root!r}, {output_root!r}, allow_library={allow_library!r})\n')


def _relative_path(value):
    if (type(value) is not str or not value or len(value) > 1024 or '\\' in value or '\x00' in value
            or value.startswith('/') or any(p in ('', '.', '..') for p in value.split('/'))):
        raise RecipeError('output path must be a safe relative path')
    if Path(value).name.casefold() in {'reward.json', 'pytest_report.json', 'reward.txt', '_agent_status.json'}:
        raise RecipeError('grading/controller output filename is reserved')
    return Path(value)


def _target(root, value):
    relative = _relative_path(value)
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise RecipeError('output symlinks are forbidden')
        if current.exists():
            node = current.stat()
            if current != root / relative and not stat.S_ISDIR(node.st_mode):
                raise RecipeError('output parent must be a directory')
            if current == root / relative and (not stat.S_ISREG(node.st_mode) or node.st_nlink != 1):
                raise RecipeError('output target must be a regular file with no hard links')
    if not current.resolve().is_relative_to(root):
        raise RecipeError('output path escaped root')
    current.parent.mkdir(parents=True, exist_ok=True)
    return current


def _size(shape):
    dims = [shape] if isinstance(shape, int) else list(shape)
    if len(dims) > 8 or any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in dims):
        raise RecipeError('array shape must contain nonnegative integer dimensions')
    total = math.prod(dims)
    if total > _MAX_ARRAY:
        raise RecipeError('array element limit before allocation')
    return tuple(dims)


def _elements(value, np, depth=0):
    """Count referenced data without coercing nested arrays into a new allocation."""
    if depth > _MAX_DEPTH:
        raise RecipeError('argument nesting limit before allocation')
    if isinstance(value, np.ndarray):
        count = value.size
    elif isinstance(value, (list, tuple, dict)):
        count = 0
        children = value.values() if isinstance(value, dict) else value
        for child in children:
            count += _elements(child, np, depth + 1)
            if count > _MAX_ARRAY:
                break
    else:
        count = 1
    if count > _MAX_ARRAY:
        raise RecipeError('array element limit before allocation')
    return count


def _functions(np, q):
    def random_state(seed):
        if type(seed) is not int or not 0 <= seed < 2 ** 32:
            raise RecipeError('RandomState requires an explicit integer seed in [0,2**32)')
        return np.random.RandomState(seed)

    def normal(rng, size):
        if not isinstance(rng, np.random.RandomState):
            raise RecipeError('rng handle must be RandomState')
        return rng.standard_normal(_size(size))

    def array(values, dtype=None):
        if dtype not in (None, 'float64', 'int64', 'bool'):
            raise RecipeError('unsupported array dtype')
        _elements(values, np)
        result = np.array(values, dtype=dtype)
        if result.dtype.kind not in 'biuf':
            raise RecipeError('array must be real numeric or boolean')
        return result

    def linspace(start, stop, num=50, endpoint=True):
        _size(num)
        if type(endpoint) is not bool:
            raise RecipeError('endpoint must be boolean')
        return np.linspace(start, stop, num=num, endpoint=endpoint)

    def arange(start, stop=None, step=1):
        if stop is None:
            start, stop = 0, start
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in (start, stop, step)) or step == 0:
            raise RecipeError('arange needs finite numeric bounds and nonzero step')
        _size(max(0, math.ceil((stop - start) / step)))
        return np.arange(start, stop, step)

    def meshgrid(*vectors, indexing='xy'):
        if indexing not in ('xy', 'ij') or any(np.ndim(v) != 1 for v in vectors):
            raise RecipeError('meshgrid needs one-dimensional vectors and xy/ij indexing')
        _size([len(v) for v in vectors])
        return np.meshgrid(*vectors, indexing=indexing)

    def moments_update(moments, samples):
        if not isinstance(moments, q.OnlineMoments):
            raise RecipeError('expected OnlineMoments handle')
        shape = np.shape(samples)
        if len(shape) == 2:
            _size([shape[1], shape[1]])
        return moments.update(samples)

    def moments_result(moments, *, ddof):
        if not isinstance(moments, q.OnlineMoments):
            raise RecipeError('expected OnlineMoments handle')
        return moments.result(ddof=ddof)

    def slices(total, *, batch_size):
        if type(total) is not int or type(batch_size) is not int or total < 0 or batch_size <= 0:
            raise RecipeError('batch count/size must be nonnegative/positive integers')
        if (total + batch_size - 1) // batch_size > _MAX_WORK:
            raise RecipeError('batch iteration limit')
        return list(q.batch_slices(total, batch_size=batch_size))

    def slice_bound(part, field):
        if not isinstance(part, slice) or type(part.start) is not int or type(part.stop) is not int:
            raise RecipeError('slice accessor requires bounded integer start/stop')
        if field == 'start':
            return part.start
        if field == 'stop':
            return part.stop
        return len(range(part.start, part.stop, 1 if part.step is None else part.step))

    def broadcast(function, *values):
        _size(np.broadcast_shapes(*(np.shape(v) for v in values)))
        return function(*values)

    def join(function, arrays, axis=0):
        if not isinstance(arrays, (list, tuple)) or not arrays:
            raise RecipeError('stack/concatenate require nonempty list/tuple')
        if sum(np.size(a) for a in arrays) > _MAX_ARRAY:
            raise RecipeError('array element limit before allocation')
        return function(arrays, axis=axis)

    def columns(arrays):
        if not isinstance(arrays, (list, tuple)) or not arrays:
            raise RecipeError('column_stack requires nonempty list/tuple')
        if sum(np.size(a) for a in arrays) > _MAX_ARRAY:
            raise RecipeError('array element limit before allocation')
        return np.column_stack(arrays)

    functions = {
        'rng.RandomState': random_state, 'rng.standard_normal': normal,
        'slice.start': lambda part: slice_bound(part, 'start'),
        'slice.stop': lambda part: slice_bound(part, 'stop'),
        'slice.size': lambda part: slice_bound(part, 'size'),
        'np.array': array, 'np.linspace': linspace, 'np.arange': arange, 'np.meshgrid': meshgrid,
        'np.zeros': lambda shape: np.zeros(_size(shape)), 'np.ones': lambda shape: np.ones(_size(shape)),
        'np.mean': lambda a, axis=None, keepdims=False: np.mean(a, axis=axis, keepdims=keepdims),
        'np.sum': lambda a, axis=None, keepdims=False: np.sum(a, axis=axis, keepdims=keepdims),
        'np.std': lambda a, axis=None, ddof=0, keepdims=False: np.std(a, axis=axis, ddof=ddof, keepdims=keepdims),
        'np.var': lambda a, axis=None, ddof=0, keepdims=False: np.var(a, axis=axis, ddof=ddof, keepdims=keepdims),
        'np.sqrt': lambda x: np.sqrt(x), 'np.exp': lambda x: np.exp(x),
        'np.log': lambda x: np.log(x), 'np.abs': lambda x: np.abs(x),
        'np.maximum': lambda x, y: broadcast(np.maximum, x, y),
        'np.minimum': lambda x, y: broadcast(np.minimum, x, y),
        'np.where': lambda condition, x, y: broadcast(np.where, condition, x, y),
        'np.all': lambda a, axis=None: np.all(a, axis=axis),
        'np.any': lambda a, axis=None: np.any(a, axis=axis),
        'np.isfinite': lambda x: np.isfinite(x),
        'np.allclose': lambda a, b, rtol=1e-5, atol=1e-8: broadcast(lambda x, y: np.allclose(x, y, rtol=rtol, atol=atol), a, b),
        'np.column_stack': columns, 'np.stack': lambda arrays, axis=0: join(np.stack, arrays, axis),
        'np.concatenate': lambda arrays, axis=0: join(np.concatenate, arrays, axis),
        'np.reshape': lambda a, shape: np.reshape(a, shape), 'np.shape': lambda a: np.shape(a), 'len': len,
    }
    if q is not None:
        def bsm_price(S, K, T, r, sigma, *, kind, q=0):
            _size(np.broadcast_shapes(*(np.shape(v) for v in (S, K, T, r, sigma, q))))
            return financial.bsm_price(S, K, T, r, sigma, kind=kind, q=q)

        def bsm_greeks(S, K, T, r, sigma, *, kind, q=0, vega_per='unit', theta_per='year', rho_per='unit'):
            _size(np.broadcast_shapes(*(np.shape(v) for v in (S, K, T, r, sigma, q))))
            return financial.bsm_greeks(S, K, T, r, sigma, kind=kind, q=q,
                vega_per=vega_per, theta_per=theta_per, rho_per=rho_per)

        financial = q
        functions.update({
            'qfkit.bsm_price': bsm_price, 'qfkit.bsm_greeks': bsm_greeks,
            'qfkit.gbm_greek_samples': q.gbm_greek_samples, 'qfkit.gbm_fd_greek_samples': q.gbm_fd_greek_samples,
            'qfkit.mc_estimate': q.mc_estimate, 'qfkit.batch_slices': slices,
            'qfkit.OnlineMoments': q.OnlineMoments, 'moments.update': moments_update, 'moments.result': moments_result,
        })
    return functions


def _bound(value, np):
    if isinstance(value, np.ndarray):
        if value.size > _MAX_ARRAY or value.dtype.kind not in 'biuf' or not np.all(np.isfinite(value)):
            raise RecipeError('array must be bounded finite real numeric/bool')
    elif isinstance(value, (list, tuple, dict)) and len(value) > _MAX_ITEMS:
        raise RecipeError('container item limit')
    elif isinstance(value, str) and len(value) > _MAX_BYTES:
        raise RecipeError('string size limit')
    elif isinstance(value, (float, np.floating)) and not math.isfinite(value):
        raise RecipeError('nonfinite computed value')
    elif isinstance(value, int) and abs(value) > 2 ** 63 - 1:
        raise RecipeError('integer range limit')
    return value


def _eval(node, env, np):
    def walk(child):
        return _eval(child, env, np)
    if isinstance(node, ast.Constant):
        result = node.value
    elif isinstance(node, ast.Name):
        result = env[node.id]
    elif isinstance(node, (ast.List, ast.Tuple)):
        values = [walk(child) for child in node.elts]
        result = tuple(values) if isinstance(node, ast.Tuple) else values
    elif isinstance(node, ast.Dict):
        result = {walk(key): walk(value) for key, value in zip(node.keys, node.values)}
    elif isinstance(node, ast.Slice):
        result = slice(*(walk(v) if v is not None else None for v in (node.lower, node.upper, node.step)))
    elif isinstance(node, ast.Subscript):
        container = walk(node.value)
        if not isinstance(container, (dict, list, tuple, str, np.ndarray)):
            raise RecipeError('indexing requires a data container')
        index = walk(node.slice)
        if isinstance(container, np.ndarray):
            basic = lambda i: i is None or isinstance(i, slice) or (isinstance(i, (int, np.integer)) and not isinstance(i, (bool, np.bool_)))
            if isinstance(index, tuple):
                if not all(basic(i) for i in index):
                    raise RecipeError('multiple advanced index arrays are unsupported')
            elif not basic(index):
                if isinstance(index, list) and all(type(i) in (int, bool) for i in index):
                    index = np.asarray(index)
                if not isinstance(index, np.ndarray) or index.ndim != 1 or index.dtype.kind not in 'biu':
                    raise RecipeError('advanced index must be one standalone 1D integer/boolean array')
                _size((len(index), *container.shape[1:]))
        result = container[index]
    elif isinstance(node, ast.BinOp):
        left, right = walk(node.left), walk(node.right)
        if isinstance(node.op, ast.Pow) and (not np.isscalar(right) or not -64 <= right <= 64):
            raise RecipeError('power exponent must be a scalar in [-64,64]')
        if isinstance(node.op, ast.Mult):
            for sequence, amount in ((left, right), (right, left)):
                if isinstance(sequence, (str, list, tuple)) and isinstance(amount, int) and len(sequence) * max(0, amount) > _MAX_ITEMS:
                    raise RecipeError('sequence expansion limit')
        if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
            _size(np.broadcast_shapes(np.shape(left), np.shape(right)))
        result = _BINARY[type(node.op)](left, right)
    elif isinstance(node, ast.UnaryOp):
        result = _UNARY[type(node.op)](walk(node.operand))
    elif isinstance(node, ast.BoolOp):
        result = walk(node.values[0])
        for child in node.values[1:]:
            if (isinstance(node.op, ast.And) and not bool(result)) or (isinstance(node.op, ast.Or) and bool(result)):
                break
            result = walk(child)
    elif isinstance(node, ast.Compare):
        left = walk(node.left)
        result = True
        for op, child in zip(node.ops, node.comparators):
            right = walk(child)
            if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
                _size(np.broadcast_shapes(np.shape(left), np.shape(right)))
            result = _COMPARE[type(op)](left, right)
            if len(node.ops) > 1 and not bool(result):
                break
            left = right
    elif isinstance(node, ast.IfExp):
        result = walk(node.body if bool(walk(node.test)) else node.orelse)
    else:
        raise RecipeError('unsupported expression')
    return _bound(result, np)


def _resolve(value, env, np):
    if type(value) is dict:
        if set(value) == {'ref'}:
            return env[value['ref']]
        if set(value) == {'expr'}:
            return _eval(_expression(value['expr'], set(env)), env, np)
        return {key: _resolve(child, env, np) for key, child in value.items()}
    if type(value) is list:
        return [_resolve(child, env, np) for child in value]
    return value


def _json_value(value, np, count=None, depth=0):
    if count is None:
        count = [0]
    count[0] += 1
    if count[0] > _MAX_ITEMS or depth > _MAX_DEPTH:
        raise RecipeError('serialization size/nesting limit')
    if isinstance(value, np.ndarray):
        if value.size > _MAX_ITEMS:
            raise RecipeError('serialized array size limit')
        value = value.tolist()
    elif isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, dict):
        if any(type(k) is not str for k in value):
            raise RecipeError('serialized object keys must be strings')
        return {key: _json_value(child, np, count, depth + 1) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(child, np, count, depth + 1) for child in value]
    _literal(value)
    return value


def _png(value, step, np):
    # Explicit Agg canvas, without pyplot, GUI state or task-specific plot defaults.
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    if 'x' in step:
        if type(value) is not list or not value:
            raise RecipeError('PNG records must be a nonempty list')
        x, y, z = step['x'], step['y'], step['z']
        coords = [(float(row[x]), float(row[y]), float(row[z])) for row in value]
        if not np.all(np.isfinite(coords)):
            raise RecipeError('PNG coordinates must be finite')
        xs, ys = sorted({row[0] for row in coords}), sorted({row[1] for row in coords})
        if len(xs) < 2 or len(ys) < 2 or len(xs) * len(ys) != len(coords) or len({(a, b) for a, b, _ in coords}) != len(coords):
            raise RecipeError('PNG records must form a complete unique rectangular grid')
        X, Y = np.meshgrid(xs, ys, indexing='xy')
        positions = {(a, b): c for a, b, c in coords}
        Z = np.array([[positions[(a, b)] for a in xs] for b in ys])
    else:
        if type(value) is not dict or set(value) != {'X', 'Y', 'Z'}:
            raise RecipeError('PNG grids require X,Y,Z')
        X, Y, Z = (np.asarray(value[k], dtype=float) for k in ('X', 'Y', 'Z'))
        if X.ndim != 2 or X.shape != Y.shape or X.shape != Z.shape or min(X.shape) < 2:
            raise RecipeError('PNG grids must have matching two-dimensional shapes')
    if X.size > _MAX_ITEMS or not all(np.all(np.isfinite(a)) for a in (X, Y, Z)):
        raise RecipeError('PNG grid size/finite check failed')
    figure = Figure(figsize=(6, 4))
    FigureCanvasAgg(figure)
    axes = figure.add_subplot(111, projection='3d')
    axes.plot_surface(X, Y, Z)
    axes.set(title=step.get('title', ''), xlabel=step.get('xlabel', ''),
             ylabel=step.get('ylabel', ''), zlabel=step.get('zlabel', ''))
    buffer = io.BytesIO()
    figure.savefig(buffer, format='png')
    figure.clear()
    return buffer.getvalue()


def _write(root, path, value, step, np):
    fmt = step['format']
    if fmt == 'png':
        data = _png(value, step, np)
    elif fmt == 'text':
        if type(value) is not str:
            raise RecipeError('text output requires a string')
        data = value.encode('utf-8')
    else:
        value = _json_value(value, np)
        if fmt == 'json':
            data = (json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + '\n').encode('utf-8')
        else:
            columns = step['columns']
            if type(value) is not list or any(type(row) is not dict or set(row) != set(columns) for row in value):
                raise RecipeError('CSV records must exactly match columns')
            if any(isinstance(cell, (dict, list)) for row in value for cell in row.values()):
                raise RecipeError('CSV cells must be scalar')
            buffer = io.StringIO(newline='')
            writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator='\n')
            writer.writeheader()
            writer.writerows(value)
            data = buffer.getvalue().encode('utf-8')
    if len(data) > 8_000_000:
        raise RecipeError('output byte limit')
    target = _target(root, path)
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(data)


def run_recipe(recipe, input_root, output_root, *, allow_library=True):
    """Execute a validated recipe in the worker; errors identify the current step."""
    validate_recipe(recipe, allow_library=allow_library)
    import numpy as np
    q = None
    if allow_library:
        import qfkit as q
    root = Path(output_root)
    if root.is_symlink() or not root.is_dir():
        raise RecipeError('output root must be an existing directory, not a symlink')
    root = root.resolve()
    env = {'INPUT_ROOT': str(Path(input_root).resolve()), 'OUTPUT_ROOT': str(root)}
    functions = _functions(np, q)
    work, writes = [0], []

    def tick():
        work[0] += 1
        if work[0] > _MAX_WORK:
            raise RecipeError('runtime step/loop iteration limit')

    def execute(steps, scope, prefix='steps'):
        for index, step in enumerate(steps):
            where = f'{prefix}[{index}]'
            try:
                tick()
                resolve = lambda value: _resolve(value, scope, np)
                op = step['op']
                if op == 'let':
                    scope[step['name']] = resolve(step['value'])
                elif op == 'call':
                    args = resolve(step.get('args', []))
                    kwargs = {key: resolve(value) for key, value in step.get('kwargs', {}).items()}
                    for value in [*args, *kwargs.values()]:
                        _elements(value, np)
                    try:
                        inspect.signature(functions[step['fn']]).bind(*args, **kwargs)
                    except TypeError as exc:
                        raise RecipeError('unsupported argument: ' + str(exc)) from exc
                    with np.errstate(over='raise', invalid='raise', divide='raise'):
                        result = functions[step['fn']](*args, **kwargs)
                    scope[step['name']] = _bound(result, np)
                elif op == 'for':
                    values = resolve(step['over'])
                    if not isinstance(values, (list, tuple, np.ndarray)) or len(values) > _MAX_WORK:
                        raise RecipeError('for requires a bounded list/tuple/array')
                    for iteration, value in enumerate(values):
                        tick()
                        local = dict(scope)
                        local[step['var']] = value
                        execute(step['body'], local, f'{where}.body@{iteration}')
                elif op == 'append':
                    target = scope[step['target']]
                    if type(target) is not list or len(target) >= _MAX_ITEMS:
                        raise RecipeError('append target must be a bounded existing list')
                    target.append(resolve(step['value']))
                elif op == 'assert':
                    condition = resolve(step['condition'])
                    if type(condition) is not bool and not isinstance(condition, np.bool_):
                        raise RecipeError('assert condition must be boolean; use np.all for arrays')
                    if not condition:
                        raise RecipeError(step.get('message', 'assertion failed'))
                elif op == 'write':
                    path = resolve(step['path'])
                    _write(root, path, resolve(step['value']), step, np)
                    writes.append(path)
                elif op == 'print':
                    value = _json_value(resolve(step['value']), np)
                    print(json.dumps(value, ensure_ascii=False, allow_nan=False)[:4000], flush=True)
            except Exception as exc:
                raise RecipeError(f'{where}: {type(exc).__name__}: {str(exc)[:1000]}') from exc

    with np.errstate(over='raise', invalid='raise', divide='raise'):
        execute(recipe['steps'], env)
    return {'steps_executed': work[0], 'written': writes}
