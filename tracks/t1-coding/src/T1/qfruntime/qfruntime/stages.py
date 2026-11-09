"""Explicit JSON stage reuse and a one-shot, caller-defined resource probe.

These tools do not choose an algorithm, alter task precision, or supply task results.
Version a stage whenever its computation or validation meaning changes. File inputs
must be represented by their content hashes, not merely paths or modification times.
"""
from __future__ import annotations

import hashlib
import dis
import inspect
import json
import math
import os
import re
import secrets
import stat
import time
import types
from pathlib import Path

__all__ = ['StageCache', 'probe']
MAX_ENTRY_BYTES = 8 * 1024 * 1024
_NAME = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,99}\Z')
_SCHEMA = 1


def _json_bytes(value):
    """Canonical, finite, bounded JSON without tuple/key coercion or custom objects."""
    def visit(item, depth=0):
        if depth > 64:
            raise ValueError('JSON nesting exceeds 64 levels')
        kind = type(item)
        if kind in (str, int, bool, type(None)):
            return
        if kind is float and math.isfinite(item):
            return
        if kind is list:
            for child in item:
                visit(child, depth + 1)
            return
        if kind is dict and all(type(key) is str for key in item):
            for child in item.values():
                visit(child, depth + 1)
            return
        raise ValueError('values must be finite JSON with string object keys')
    try:
        visit(value)
        result = json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False,
                            separators=(',', ':')).encode('utf-8')
    except (TypeError, OverflowError, RecursionError) as exc:
        raise ValueError('value cannot be represented as finite JSON') from exc
    if len(result) > MAX_ENTRY_BYTES:
        raise ValueError('JSON exceeds the stage cache size limit')
    return result


def _digest(value):
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _code_spec(code):
    def constant(value):
        if isinstance(value, types.CodeType):
            return {'code': _code_spec(value)}
        if isinstance(value, tuple):
            return {'tuple': [constant(item) for item in value]}
        if isinstance(value, bytes):
            return {'bytes': value.hex()}
        if value is Ellipsis:
            return {'constant': 'ellipsis'}
        return value
    return {'bytecode': code.co_code.hex(), 'constants': [constant(item) for item in code.co_consts],
            'names': list(code.co_names), 'variables': list(code.co_varnames),
            'free_variables': list(code.co_freevars), 'flags': code.co_flags,
            'argcount': code.co_argcount, 'kwonly': code.co_kwonlyargcount,
            'posonly': code.co_posonlyargcount}


def _global_names(code):
    names = {instruction.argval for instruction in dis.get_instructions(code)
             if instruction.opname in ('LOAD_GLOBAL', 'LOAD_NAME')}
    for item in code.co_consts:
        if isinstance(item, types.CodeType):
            names.update(_global_names(item))
    return names


def _function_spec(function, seen):
    if not isinstance(function, types.FunctionType) or id(function) in seen:
        raise ValueError('opaque or cyclic function dependency')
    seen = seen | {id(function)}

    def binding(value):
        if isinstance(value, types.FunctionType):
            return {'function': _function_spec(value, seen)}
        if ((isinstance(value, (types.BuiltinFunctionType, type)))
                and getattr(value, '__module__', None) == 'builtins'):
            if value.__name__ in ('globals', 'locals', 'vars', 'eval', 'exec', '__import__'):
                raise ValueError('dynamic namespace access is not a stable cache dependency')
            return {'builtin': value.__name__}
        _json_bytes(value)
        return {'json': value}

    try:
        source = inspect.getsource(function)
    except (OSError, TypeError):
        source = None
    referenced = {}
    for name in sorted(_global_names(function.__code__)):
        if name in function.__globals__:
            value = function.__globals__[name]
        elif name in function.__builtins__:
            value = function.__builtins__[name]
        else:
            raise ValueError('unresolved global dependency')
        referenced[name] = binding(value)
    return {'source': source, 'code': _code_spec(function.__code__),
            'defaults': [binding(value) for value in (function.__defaults__ or ())],
            'keyword_defaults': {name: binding(value) for name, value in (function.__kwdefaults__ or {}).items()},
            'closure': [binding(cell.cell_contents) for cell in (function.__closure__ or ())],
            'globals': referenced}


def _function_fingerprint(function):
    """Fingerprint referenced JSON state and helper functions; opaque state disables hits."""
    try:
        return _digest(_function_spec(function, set()))
    except (ValueError, TypeError, AttributeError, RecursionError):
        return None


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON keys')
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError('non-finite JSON constant')


class StageCache:
    """Reuse validated finite JSON in one explicit input namespace.

    ``run(name, *, version, inputs, parameters, dependencies, compute, validate)``
    calls ``compute()`` only on a miss, then requires ``validate(result) is True``.
    Inputs and parameters are JSON; file inputs must contain content hashes.
    Dependencies are names already successfully run in this instance. Changed
    upstream signatures make downstream stages stale until explicitly rerun.

    A hit re-runs the current validator. Both compute and validate function code,
    source, defaults, JSON closure/global contents and referenced helper functions
    are fingerprinted where possible. Modules, opaque objects, dynamic namespace
    access and recursive helper dependency cycles disable reuse; computation still runs, so conservative
    identification may cost recomputation. Imports performed inside compute are
    allowed but dependency/package changes still require an explicit new version.
    Validation must be deterministic and must not mutate result.
    No result can be trusted merely because an earlier validator passed.

    Entries are bounded JSON, atomically replaced; no pickle or code loading is
    used. Symlinks at the supplied root, namespace or entry are rejected. The root
    should be a controller-owned directory for this solve, not a shared cache.
    Concurrent callers require external coordination; this is not a distributed
    cache. Diagnostics contain names, hashes and statuses, not input/result values.
    """

    def __init__(self, root, *, namespace):
        if not isinstance(namespace, str) or not namespace or len(namespace) > 4096:
            raise ValueError('namespace must be a nonempty string of at most 4096 characters')
        self.root = Path(root).absolute()
        if self.root.is_symlink():
            raise ValueError('cache root must not be a symlink')
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.namespace = hashlib.sha256(namespace.encode('utf-8')).hexdigest()
        self._current = {}
        self._diagnostics = {}
        self._stats = {'hits': 0, 'computed': 0, 'writes': 0, 'invalid': 0}
        fd = self._open_directory()
        os.close(fd)

    def _open_directory(self):
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        try:
            root_fd = os.open(self.root, flags)
        except OSError as exc:
            raise ValueError('cache root is not a regular directory') from exc
        try:
            try:
                os.mkdir(self.namespace, mode=0o700, dir_fd=root_fd)
            except FileExistsError:
                pass
            try:
                return os.open(self.namespace, flags, dir_fd=root_fd)
            except OSError as exc:
                raise ValueError('cache namespace is not a regular directory') from exc
        finally:
            os.close(root_fd)

    @staticmethod
    def _entry_name(name):
        if not isinstance(name, str) or not _NAME.fullmatch(name):
            raise ValueError('stage name must be a simple identifier, not a path')
        return hashlib.sha256(name.encode('utf-8')).hexdigest() + '.json'

    @staticmethod
    def _check_entry(fd, filename):
        try:
            info = os.stat(filename, dir_fd=fd, follow_symlinks=False)
        except FileNotFoundError:
            return None
        if not stat.S_ISREG(info.st_mode):
            raise ValueError('cache entry must be a regular file, never a symlink')
        return info

    def _read(self, filename):
        fd = self._open_directory()
        try:
            info = self._check_entry(fd, filename)
            if info is None:
                return None
            if info.st_size > MAX_ENTRY_BYTES:
                self._stats['invalid'] += 1
                return None
            stream_fd = os.open(filename, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
            with os.fdopen(stream_fd, 'rb') as stream:
                if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                    raise ValueError('cache entry is not a regular file')
                data = stream.read(MAX_ENTRY_BYTES + 1)
            try:
                if len(data) > MAX_ENTRY_BYTES:
                    raise ValueError('oversized entry')
                entry = json.loads(data, object_pairs_hook=_object, parse_constant=_reject_constant)
                _json_bytes(entry)
                if (not isinstance(entry, dict) or entry.get('schema') != _SCHEMA
                        or entry.get('namespace') != self.namespace
                        or 'result' not in entry or entry.get('result_sha256') != _digest(entry['result'])):
                    raise ValueError('invalid stage entry')
                return entry
            except (ValueError, UnicodeError, RecursionError):
                self._stats['invalid'] += 1
                return None
        finally:
            os.close(fd)

    def _write(self, filename, entry):
        data = _json_bytes(entry)
        fd = self._open_directory()
        temporary = '.write-' + secrets.token_hex(12)
        try:
            self._check_entry(fd, filename)
            stream_fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                                0o600, dir_fd=fd)
            with os.fdopen(stream_fd, 'wb') as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            self._check_entry(fd, filename)
            os.replace(temporary, filename, src_dir_fd=fd, dst_dir_fd=fd)
            self._stats['writes'] += 1
        finally:
            try:
                os.unlink(temporary, dir_fd=fd)
            except FileNotFoundError:
                pass
            os.close(fd)

    def _dependency(self, name, seen):
        self._entry_name(name)
        if name in seen:
            raise ValueError('cyclic stage dependency')
        entry = self._current.get(name)
        if entry is None:
            raise ValueError('dependencies must first run successfully in this StageCache instance')
        for dependency, expected in entry['dependencies'].items():
            current = self._dependency(dependency, seen | {name})
            if current != expected:
                raise ValueError('dependency is stale; rerun the upstream stage first')
        return {'signature': entry['signature'], 'result_sha256': entry['result_sha256']}

    def run(self, name, *, version, inputs, parameters, dependencies, compute, validate):
        """Compute or reuse one validated JSON result, returning a detached JSON value."""
        self._entry_name(name)
        self._current.pop(name, None)
        try:
            return self._run(name, version=version, inputs=inputs, parameters=parameters,
                             dependencies=dependencies, compute=compute, validate=validate)
        except Exception:
            self._diagnostics[name] = {'status': 'failed'}
            raise

    def _run(self, name, *, version, inputs, parameters, dependencies, compute, validate):
        filename = self._entry_name(name)
        if not isinstance(version, str) or not version or len(version) > 4096:
            raise ValueError('version must be an explicit nonempty string')
        if not callable(compute) or not callable(validate):
            raise ValueError('compute and validate must be callable')
        if (not isinstance(dependencies, (list, tuple))
                or not all(isinstance(dep, str) for dep in dependencies)
                or len(dependencies) != len(set(dependencies))):
            raise ValueError('dependencies must contain unique stage names')
        dependency_state = {dep: self._dependency(dep, {name}) for dep in dependencies}
        validator = _function_fingerprint(validate)
        computation = _function_fingerprint(compute)
        signature = _digest({'name': name, 'namespace': self.namespace, 'version': version,
                             'inputs': inputs, 'parameters': parameters, 'dependencies': dependency_state,
                             'validator': validator, 'compute': computation})
        entry = self._read(filename)
        result, hit = None, False
        if (validator is not None and computation is not None and entry is not None and entry.get('name') == name
                and entry.get('signature') == signature):
            candidate = entry['result']
            before = _json_bytes(candidate)
            if validate(candidate) is True and _json_bytes(candidate) == before:
                result, hit = candidate, True
            else:
                self._stats['invalid'] += 1
        if not hit:
            result = compute()
            self._stats['computed'] += 1
            before = _json_bytes(result)
            if validate(result) is not True:
                raise ValueError('stage validation must return literal True')
            if _json_bytes(result) != before:
                raise ValueError('validator must not mutate the stage result')
            entry = {'schema': _SCHEMA, 'namespace': self.namespace, 'name': name,
                     'signature': signature, 'result_sha256': _digest(result), 'result': result}
            self._write(filename, entry)
        else:
            self._stats['hits'] += 1
        self._current[name] = {'signature': signature, 'result_sha256': entry['result_sha256'],
                               'dependencies': dependency_state}
        self._diagnostics[name] = {'status': 'hit' if hit else 'computed', 'signature': signature,
                                  'result_sha256': entry['result_sha256'],
                                  'validator_fingerprint': 'available' if validator is not None else 'unavailable',
                                  'compute_fingerprint': 'available' if computation is not None else 'unavailable'}
        return json.loads(_json_bytes(result))

    def diagnostics(self):
        """Return detached JSON diagnostics without cached input or result values."""
        return json.loads(_json_bytes({'namespace': self.namespace, 'stages': self._diagnostics}))

    def stats(self):
        """Return numeric cache hit, computation, invalid entry and write counters."""
        return dict(self._stats)


def probe(compute, *, sample_size, target_size, time_budget, memory_budget=None,
          growth_power=1, fixed_seconds=0):
    """Run ``compute(sample_size)`` once and report an explicit time extrapolation.

    ``fixed_seconds + max(0, elapsed-fixed_seconds)*(target/sample)**growth_power``
    is an estimate, not a guarantee. Budgets are seconds and (optionally) bytes.
    No target computation, batching, seed changes, precision reductions or memory
    measurement occur here. Memory is ``unverified`` when a memory budget is given;
    in that case ``fits_budget`` is None unless the time budget already fails.
    The caller must run the full specified calculation independently. Exceptions
    raised by the small computation propagate; they never produce a passing probe.
    """
    if (type(sample_size) is not int or sample_size <= 0 or type(target_size) is not int
            or target_size < sample_size):
        raise ValueError('sample_size and target_size must be positive integers, target >= sample')
    for name, value, positive in (('time_budget', time_budget, True),
                                  ('growth_power', growth_power, False),
                                  ('fixed_seconds', fixed_seconds, False)):
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or (positive and value == 0):
            raise ValueError(name + ' must be finite and ' + ('positive' if positive else 'nonnegative'))
    if memory_budget is not None and (type(memory_budget) is not int or memory_budget <= 0):
        raise ValueError('memory_budget must be a positive integer byte count')
    if not callable(compute):
        raise ValueError('compute must be callable')
    started = time.perf_counter()
    result = compute(sample_size)
    elapsed = max(0., time.perf_counter() - started)
    try:
        projected = fixed_seconds + max(0., elapsed - fixed_seconds) * (target_size / sample_size) ** growth_power
    except OverflowError:
        projected = math.inf
    fits_time = math.isfinite(projected) and projected <= time_budget
    return {'result': result, 'sample_size': sample_size, 'target_size': target_size,
            'elapsed_seconds': elapsed, 'projected_seconds': projected if math.isfinite(projected) else None,
            'projection_status': 'estimated' if math.isfinite(projected) else 'overflow',
            'growth_power': growth_power, 'fixed_seconds': fixed_seconds, 'time_budget': time_budget,
            'fits_time_budget': fits_time, 'memory_budget': memory_budget, 'fits_memory_budget': None,
            'memory_status': 'unverified' if memory_budget is not None else 'not_requested',
            'fits_budget': False if not fits_time else None if memory_budget is not None else True,
            'target_executed': False}
