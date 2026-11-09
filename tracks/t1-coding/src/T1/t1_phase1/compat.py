"""Runtime compatibility for the generated program: library names that recent releases removed, solver
settings passed to scipy.optimize.minimize outside `options`, and json for NumPy, pandas and datetime values.

The source of this module is appended to the worker's sitecustomize (baseline.worker_site), so it runs
inside the generated program only, uses the standard library only and never raises. The model often writes
names that NumPy 2 and SciPy 1.14 removed (np.math, np.float, np.NaN, np.trapz, scipy.integrate.simps) or
that never existed (np.erf), and json.dump of a NumPy integer or bool; each crash costs a repair round. A
removed name resolves to its documented replacement only when the library lacks it; nothing the library
still has is replaced, and every other missing name fails as before. Likewise minimize(..., ftol=...) is
given its solver settings through options= only when the call as written would raise TypeError.
"""
import datetime as _compat_datetime
import functools as _compat_functools
import json as _compat_json
import math as _compat_math
import sys as _compat_sys
import warnings as _compat_warnings

# Removed name -> how to build its replacement from the imported module.
_COMPAT_NUMPY = {
    'math': lambda np: _compat_math, 'float': lambda np: float, 'int': lambda np: int,
    'bool': lambda np: bool, 'object': lambda np: object, 'str': lambda np: str,
    'complex': lambda np: complex, 'long': lambda np: int, 'unicode': lambda np: str,
    'NaN': lambda np: np.nan, 'NAN': lambda np: np.nan, 'Inf': lambda np: np.inf,
    'Infinity': lambda np: np.inf, 'infty': lambda np: np.inf, 'PINF': lambda np: np.inf,
    'NINF': lambda np: -np.inf, 'PZERO': lambda np: 0.0, 'NZERO': lambda np: -0.0,
    'float_': lambda np: np.float64, 'complex_': lambda np: np.complex128, 'cfloat': lambda np: np.complex128,
    'singlecomplex': lambda np: np.complex64, 'longfloat': lambda np: np.longdouble,
    'longcomplex': lambda np: np.clongdouble, 'clongfloat': lambda np: np.clongdouble,
    'string_': lambda np: np.bytes_, 'unicode_': lambda np: np.str_,
    'product': lambda np: np.prod, 'cumproduct': lambda np: np.cumprod, 'alltrue': lambda np: np.all,
    'sometrue': lambda np: np.any, 'round_': lambda np: np.round, 'trapz': lambda np: np.trapezoid,
    'in1d': lambda np: np.isin, 'row_stack': lambda np: np.vstack,
    'asfarray': lambda np: (lambda a, dtype=np.float64: np.asarray(a, dtype=dtype)),
    'msort': lambda np: (lambda a: np.sort(a, axis=0)),
}
_COMPAT_SCIPY_INTEGRATE = {
    'simps': lambda m: m.simpson, 'trapz': lambda m: m.trapezoid, 'cumtrapz': lambda m: m.cumulative_trapezoid,
}
_COMPAT_SPECIAL = ('erf', 'erfc', 'erfinv')  # np.<name> -> scipy.special.<name>, imported on first use


def _compat_missing(module, name):
    try:
        with _compat_warnings.catch_warnings():
            _compat_warnings.simplefilter('ignore')
            getattr(module, name)
        return False
    except AttributeError:
        return True
    except Exception:
        return False


def _compat_fill(module, table):
    for name, make in table.items():
        try:
            if _compat_missing(module, name):
                setattr(module, name, make(module))
        except Exception:
            pass


def _compat_special(np):
    previous = vars(np).get('__getattr__')

    def __getattr__(name):
        if name in _COMPAT_SPECIAL:
            try:
                import scipy.special
            except ImportError:  # hasattr(np, 'erf') must stay False without SciPy
                raise AttributeError(f"module 'numpy' has no attribute {name!r}") from None
            return getattr(scipy.special, name)
        if previous is not None:
            return previous(name)
        raise AttributeError(f"module 'numpy' has no attribute {name!r}")
    np.__getattr__ = __getattr__


def _compat_numpy(np):
    _compat_fill(np, _COMPAT_NUMPY)
    _compat_special(np)


# Settings scipy.optimize.minimize takes only inside options= (per method); passed as keywords of minimize
# itself they raise TypeError. A keyword outside this set is never moved, so the call fails as before.
_COMPAT_MINIMIZE_OPTIONS = frozenset({
    'ftol', 'gtol', 'xtol', 'xatol', 'fatol', 'xrtol', 'maxiter', 'maxfun', 'maxfev', 'maxcor', 'maxls', 'eps',
    'disp', 'iprint', 'norm', 'adaptive', 'return_all', 'initial_simplex', 'rhobeg', 'catol', 'barrier_tol',
    'finite_diff_rel_step', 'initial_trust_radius', 'max_trust_radius'})
# The wrapper is compiled under minimize's own file name and first line, so a traceback through it (an error in
# the objective, say) shows a frame of scipy's minimize, never this module.
_COMPAT_MINIMIZE_SOURCE = '''def minimize(*args, **kwargs):
    extra = [key for key in kwargs if key not in _accepted]
    options = kwargs.get('options')
    if (extra and len(args) <= _options_at and all(key in _settings for key in extra)
            and (options is None or isinstance(options, dict))):
        kwargs = dict(kwargs)
        options = dict(options or {})
        for key in extra:
            value = kwargs.pop(key)
            options.setdefault(key, value)
        kwargs['options'] = options
    return _original(*args, **kwargs)
'''


def _compat_minimize(original):
    """minimize, or a wrapper that moves known solver settings passed as its own keywords into options: only a
    call that minimize would reject (a keyword it lacks), never one it accepts."""
    code = getattr(original, '__code__', None)
    if code is None or code.co_flags & 0x08:  # not a Python function, or it takes **kwargs itself
        return original
    names = code.co_varnames[:code.co_argcount + code.co_kwonlyargcount]
    if 'options' not in names[:code.co_argcount]:
        return original
    space = {'_original': original, '_accepted': frozenset(names), '_settings': _COMPAT_MINIMIZE_OPTIONS,
             '_options_at': names.index('options')}
    source = '\n' * max(0, code.co_firstlineno - 1) + _COMPAT_MINIMIZE_SOURCE
    exec(compile(source, code.co_filename, 'exec', dont_inherit=True), space)
    return _compat_functools.update_wrapper(space['minimize'], original)


def _compat_optimize(module):
    original = getattr(module, 'minimize', None)
    if original is not None and not hasattr(original, '__wrapped__'):
        wrapper = _compat_minimize(original)
        if wrapper is not original:  # pickle finds a function by module and name: it must find the wrapper
            wrapper.__module__, wrapper.__qualname__ = module.__name__, 'minimize'
        module.minimize = wrapper


# Patched right after their own import finishes; every other import is left to the normal finders.
_COMPAT_AFTER = {'numpy': _compat_numpy,
                 'scipy.integrate': lambda module: _compat_fill(module, _COMPAT_SCIPY_INTEGRATE),
                 'scipy.optimize': _compat_optimize}


class _CompatLoader:
    def __init__(self, loader, name):
        self._compat_loader, self._compat_name = loader, name

    def __getattr__(self, attribute):
        return getattr(self._compat_loader, attribute)

    def create_module(self, spec):
        return self._compat_loader.create_module(spec)

    def exec_module(self, module):
        self._compat_loader.exec_module(module)
        try:
            _COMPAT_AFTER[self._compat_name](module)
        except Exception:
            pass


class _CompatFinder:
    """Finds numpy and scipy.integrate through the other finders and wraps their loader. It returns None
    for every other name, so a failed import's traceback never shows this module."""

    def __init__(self):
        self._busy = set()

    def find_spec(self, name, path=None, target=None):
        if name not in _COMPAT_AFTER or name in self._busy:
            return None
        self._busy.add(name)
        try:
            for finder in list(_compat_sys.meta_path):
                if finder is self or not hasattr(finder, 'find_spec'):
                    continue
                spec = finder.find_spec(name, path, target)
                if spec is not None:
                    break
            else:
                return None
        except Exception:
            return None
        finally:
            self._busy.discard(name)
        if spec.loader is not None and hasattr(spec.loader, 'exec_module'):
            spec.loader = _CompatLoader(spec.loader, name)
        return spec


_compat_default = _compat_json.JSONEncoder.default


def _compat_json_default(self, o):
    root = (getattr(type(o), '__module__', '') or '').split('.', 1)[0]
    if root == 'numpy' and hasattr(o, 'tolist'):
        return o.tolist()
    if root == 'pandas':
        if hasattr(o, 'isoformat'):
            return o.isoformat()
        if hasattr(o, 'tolist'):
            return o.tolist()
    if isinstance(o, (_compat_datetime.date, _compat_datetime.time)):
        return o.isoformat()
    return _compat_default(self, o)


try:
    _compat_sys.meta_path.insert(0, _CompatFinder())
    _compat_json.JSONEncoder.default = _compat_json_default
except Exception:
    pass
