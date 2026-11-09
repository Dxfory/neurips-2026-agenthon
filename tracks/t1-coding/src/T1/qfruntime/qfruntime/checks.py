"""Bounded, explicit numerical self-checks using only the Python standard library.

Reports are untrusted diagnostic evidence, not a correctness certificate. A
different declared method does not establish that two computations are independent.
No values, task text, file contents, or full arrays are stored in a report.
"""
from __future__ import annotations

import copy
import json
import math
from numbers import Integral
import os
from pathlib import Path
import stat
import uuid

__all__ = ['Checks']

_MAX_CHECKS = 128
_MAX_BYTES = 262144
_PREFIX = '__checks_'


class _Invalid(ValueError):
    """Internal, data-free reason for a failed check."""


def _label(value, *, optional=False):
    if optional and value is None:
        return None
    if not isinstance(value, str) or not value.strip() or len(value) > 96:
        raise _Invalid('invalid_label')
    return value


def _number(value, *, nonnegative=False):
    if isinstance(value, (str, bytes, bool)):
        raise _Invalid('non_numeric')
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise _Invalid('non_numeric') from exc
    if not math.isfinite(out):
        raise _Invalid('nonfinite')
    if isinstance(value, Integral) and int(out) != int(value):
        raise _Invalid('integer_precision_loss')
    if nonnegative and out < 0:
        raise _Invalid('negative_parameter')
    return out


def _array(value, depth=0):
    """Return rectangular shape and flattened values; never broadcast."""
    if depth > 32:
        raise _Invalid('too_many_dimensions')
    if not isinstance(value, (list, tuple)) and callable(getattr(value, 'tolist', None)):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        return (), [value]
    shapes, flat = [], []
    for child in value:
        shape, items = _array(child, depth + 1)
        shapes.append(shape)
        flat.extend(items)
    if shapes and any(shape != shapes[0] for shape in shapes[1:]):
        raise _Invalid('ragged_array')
    return (len(value),) + (shapes[0] if shapes else ()), flat


def _numbers(value):
    shape, flat = _array(value)
    if not flat:
        raise _Invalid('empty_values')
    return shape, [_number(item) for item in flat]


def _metadata(value, depth=0):
    """Bound all user supplied specification metadata without storing data arrays."""
    if depth > 4:
        return {'invalid': 'metadata_depth'}
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value if len(value) <= 96 else {'invalid': 'label_length'}
    if isinstance(value, (int, float)):
        try:
            return value if math.isfinite(value) else {'invalid': 'nonfinite'}
        except OverflowError:
            return {'invalid': 'numeric_range'}
    if isinstance(value, (list, tuple)):
        if len(value) > 32:
            return {'invalid': 'metadata_length'}
        return [_metadata(item, depth + 1) for item in value]
    # Array scalar metadata is allowed, but no implicit stringification of objects.
    if callable(getattr(value, 'item', None)):
        try:
            return _metadata(value.item(), depth + 1)
        except (TypeError, ValueError):
            pass
    return {'invalid': 'metadata_type'}


def _compare(actual, expected, atol, rtol):
    atol, rtol = _number(atol, nonnegative=True), _number(rtol, nonnegative=True)
    actual_shape, actual = _numbers(actual)
    expected_shape, expected = _numbers(expected)
    if actual_shape != expected_shape:
        raise _Invalid('shape_mismatch')
    differences = [_number(abs(a - b)) for a, b in zip(actual, expected)]
    tolerances = [_number(atol + rtol * abs(b)) for b in expected]
    return all(d <= t for d, t in zip(differences, tolerances)), max(differences), max(tolerances)


class Checks:
    """Record bounded contracts and diagnostics with explicit numerical tolerances.

    ``report_path=None`` uses ``T1_RUNTIME_REPORT`` when set, otherwise reports stay
    in memory. The destination parent must already exist; symlinks in the path are
    rejected. Each recorded result is atomically saved as mode 0600. Only contract
    failures raise ValueError; other failed/invalid checks return False so reporting
    does not erase a task's normal outputs. ``finish()`` returns the current report.

    IDs must be distinct nonempty strings of at most 96 characters and cannot use
    the reserved ``__checks_`` prefix. At most 127 ordinary records are retained;
    further records produce an explicit failure in the 128th slot. A report is a
    deep copy and contains no raw input arrays. Freeze id/kind/parameters/evidence_type
    across repair attempts: deleting checks or weakening tolerances is not a repair.
    Numerical checks use binary64 arithmetic; integers that cannot be represented
    exactly as binary64 are rejected instead of silently rounded to an equal value.
    """

    def __init__(self, report_path=None):
        selected = os.environ.get('T1_RUNTIME_REPORT') if report_path is None else report_path
        self._path = None if selected in (None, '') else Path(selected)
        self._checks = []
        self._ids = set()
        self._internal_counter = 0
        self._overflow_count = 0
        self._report_bytes = 40

    def _append(self, row):
        row_bytes = len(json.dumps(row, ensure_ascii=True, allow_nan=False).encode('utf-8')) + 2
        if (self._overflow_count or len(self._checks) >= _MAX_CHECKS - 1
                or self._report_bytes + row_bytes > _MAX_BYTES - 1024):
            self._overflow_count += 1
            overflow = {'id': _PREFIX + 'overflow', 'kind': 'overflow', 'status': 'failed',
                        'parameters': {'capacity': _MAX_CHECKS, 'max_bytes': _MAX_BYTES},
                        'difference': self._overflow_count, 'tolerance': None,
                        'evidence_type': 'unverified'}
            if self._overflow_count == 1:
                self._checks.append(overflow)
            else:
                self._checks[-1] = overflow
            return False
        self._checks.append(row)
        self._report_bytes += row_bytes
        return row['status'] == 'passed'

    def _internal(self, kind, reason, *, original_id=None):
        self._internal_counter += 1
        parameters = {'reason': reason}
        if original_id is not None:
            parameters['duplicate_of'] = original_id
        return self._append({'id': _PREFIX + kind + '_' + str(self._internal_counter),
                             'kind': kind, 'status': 'failed', 'parameters': parameters,
                             'difference': None, 'tolerance': None, 'evidence_type': 'unverified'})

    def _run(self, name, kind, parameters, compute, evidence_type, *, raise_error=False):
        try:
            _label(name)
            if name.startswith(_PREFIX):
                raise _Invalid('reserved_id')
        except _Invalid:
            self._internal('invalid_id', 'invalid_id')
            self._publish()
            if raise_error:
                raise ValueError('contract failed: invalid_id')
            return False
        if name in self._ids:
            self._internal('duplicate', 'duplicate_id', original_id=name)
            self._publish()
            if raise_error:
                raise ValueError('contract failed: duplicate_id')
            return False
        # Once capped, avoid accumulating an unbounded set of unseen identifiers.
        if not self._overflow_count:
            self._ids.add(name)
        reason = 'comparison_failed'
        valid_input = True
        try:
            passed, difference, tolerance = compute()
        except (ValueError, TypeError, OverflowError, ArithmeticError) as exc:
            passed, difference, tolerance = False, None, None
            valid_input = False
            reason = str(exc) if isinstance(exc, _Invalid) else 'invalid_input'
        status = 'skipped' if kind == 'skip' and valid_input else ('passed' if passed else 'failed')
        recorded = self._append({'id': name, 'kind': kind, 'status': status,
                                 'parameters': {k: _metadata(v) for k, v in parameters.items()},
                                 'difference': difference, 'tolerance': tolerance,
                                 'evidence_type': evidence_type})
        self._publish()
        if raise_error and not recorded:
            raise ValueError('contract failed: ' + reason)
        return recorded

    def contract(self, name, values, *, shape=None, actual_unit=None, expected_unit=None,
                 finite=True, lower=None, upper=None, choices=None, method=None, expected_method=None):
        """Validate only supplied constraints, with no unit inference or conversion.

        Shape is a sequence of nonnegative integers or None (wildcard axes), with
        () indicating a scalar. Strings are permitted only with explicit choices;
        numerical values are finite by default. ``finite=False`` permits missing
        numerical values only when no range is specified. All failures are recorded
        before raising ValueError. Success returns True, without altering values.
        """
        parameters = dict(shape=shape, actual_unit=actual_unit, expected_unit=expected_unit,
                          finite=finite, lower=lower, upper=upper, choices=choices,
                          method=method, expected_method=expected_method)

        def compute():
            observed_shape, flat = _array(values)
            if not isinstance(finite, bool):
                raise _Invalid('invalid_finite_flag')
            if shape is not None:
                if not isinstance(shape, (list, tuple)) or len(shape) > 32:
                    raise _Invalid('invalid_shape')
                for axis in shape:
                    if axis is not None and (isinstance(axis, bool) or not isinstance(axis, int) or axis < 0):
                        raise _Invalid('invalid_shape')
                if len(shape) != len(observed_shape) or any(
                        wanted is not None and wanted != got for wanted, got in zip(shape, observed_shape)):
                    raise _Invalid('shape_mismatch')
            for label in (actual_unit, expected_unit, method, expected_method):
                _label(label, optional=True)
            if expected_unit is not None and actual_unit != expected_unit:
                raise _Invalid('unit_mismatch')
            if expected_method is not None and method != expected_method:
                raise _Invalid('method_mismatch')
            lo = None if lower is None else _number(lower)
            hi = None if upper is None else _number(upper)
            if lo is not None and hi is not None and lo > hi:
                raise _Invalid('inverted_bounds')
            if choices is not None:
                if not isinstance(choices, (list, tuple)) or not 1 <= len(choices) <= 32:
                    raise _Invalid('invalid_choices')
                for choice in choices:
                    _label(choice) if isinstance(choice, str) else _number(choice)
            for item in flat:
                if isinstance(item, str):
                    if choices is None:
                        raise _Invalid('non_numeric')
                    _label(item)
                    if lo is not None or hi is not None:
                        raise _Invalid('non_numeric')
                elif finite or lo is not None or hi is not None:
                    number = _number(item)
                    if (lo is not None and number < lo) or (hi is not None and number > hi):
                        raise _Invalid('outside_bounds')
                else:
                    # Finiteness is explicitly disabled, not type validation.
                    if isinstance(item, (str, bytes, bool)):
                        raise _Invalid('non_numeric')
                    float(item)
                if choices is not None and item not in choices:
                    raise _Invalid('outside_choices')
            return True, None, None

        return self._run(name, 'contract', parameters, compute, 'contract', raise_error=True)

    def close(self, name, actual, expected, *, atol, rtol, computed_method, reference_method):
        """Compare equal-shaped finite values using abs(a-b) <= atol+rtol*abs(b).

        Different method names are only a declaration of cross-method comparison;
        their implementations or independence are not inspected. Arrays are compared
        elementwise, never broadcast. Difference/tolerance show their maxima, while
        status reflects every element, not the maxima alone.
        """
        parameters = dict(atol=atol, rtol=rtol, computed_method=computed_method,
                          reference_method=reference_method)

        def compute():
            _label(computed_method)
            _label(reference_method)
            return _compare(actual, expected, atol, rtol)

        same_method = (isinstance(computed_method, str) and isinstance(reference_method, str)
                       and computed_method == reference_method)
        evidence = 'consistency' if same_method else 'declared_cross_method'
        return self._run(name, 'close', parameters, compute, evidence)

    def balance(self, name, parts, total, *, atol, rtol):
        """Check the compensated sum of finite components against a scalar total."""
        def compute():
            _, components = _numbers(parts)
            shape, expected = _numbers(total)
            if shape:
                raise _Invalid('total_must_be_scalar')
            return _compare(math.fsum(components), expected[0], atol, rtol)

        return self._run(name, 'balance', dict(atol=atol, rtol=rtol), compute, 'identity')

    def bounds(self, name, values, *, lower, upper, atol=0):
        """Check finite values against explicit inclusive scalar bounds plus atol.

        Either bound may be None, but at least one is required. No economic range
        is inferred; for example, negative rates require no special exception.
        """
        def compute():
            tolerance = _number(atol, nonnegative=True)
            lo = None if lower is None else _number(lower)
            hi = None if upper is None else _number(upper)
            if lo is None and hi is None:
                raise _Invalid('missing_bounds')
            if lo is not None and hi is not None and lo > hi:
                raise _Invalid('inverted_bounds')
            _, flat = _numbers(values)
            deviations = [_number(max(0., lo - value if lo is not None else 0.,
                                       value - hi if hi is not None else 0.)) for value in flat]
            return max(deviations) <= tolerance, max(deviations), tolerance

        return self._run(name, 'bounds', dict(lower=lower, upper=upper, atol=atol), compute, 'bounds')

    def convergence(self, name, coarse, fine, *, atol, rtol):
        """Record adjacent-resolution agreement; it does not prove convergence to truth."""
        return self._run(name, 'convergence', dict(atol=atol, rtol=rtol),
                         lambda: _compare(coarse, fine, atol, rtol), 'convergence')

    def estimate(self, name, estimate, reference, standard_error, *, z):
        """Check abs(estimate-reference) <= z*standard_error for equal-shaped arrays.

        The caller supplies nonnegative finite standard errors and z. No normality,
        independence, or correctness of those standard errors is inferred. Scalars
        and arrays must have matching shapes, including standard_error.
        """
        def compute():
            multiplier = _number(z, nonnegative=True)
            shape, actual = _numbers(estimate)
            ref_shape, expected = _numbers(reference)
            se_shape, errors = _numbers(standard_error)
            if shape != ref_shape or shape != se_shape:
                raise _Invalid('shape_mismatch')
            tolerances = [_number(multiplier * _number(se, nonnegative=True)) for se in errors]
            differences = [_number(abs(a - b)) for a, b in zip(actual, expected)]
            return all(d <= t for d, t in zip(differences, tolerances)), max(differences), max(tolerances)

        return self._run(name, 'estimate', dict(z=z), compute, 'statistical_comparison')

    def skip(self, name, reason):
        """Record an explicitly skipped, unverified check; always return False."""
        def compute():
            _label(reason)
            return False, None, None

        return self._run(name, 'skip', dict(reason=reason), compute, 'unverified')

    def report(self):
        """Return a detached, bounded JSON-compatible diagnostic report."""
        return {'schema_version': 1, 'checks': copy.deepcopy(self._checks)}

    def finish(self):
        """Persist and return the report without raising for failed checks."""
        self._publish()
        return self.report()

    def _publish(self):
        if self._path is None:
            return
        try:
            self._write_report()
        except (OSError, ValueError, TypeError):
            if not any(row['kind'] == 'report_write' for row in self._checks):
                self._internal('report_write', 'report_write_failed')

    def _write_report(self):
        """Use anchored directory FDs so symlink races cannot redirect writes."""
        payload = json.dumps(self.report(), ensure_ascii=True, allow_nan=False,
                             separators=(',', ':')).encode('utf-8')
        if len(payload) > _MAX_BYTES:
            raise ValueError('report_too_large')
        path = Path(os.path.abspath(os.fspath(self._path)))
        if not path.name or path.name in ('.', '..'):
            raise ValueError('invalid_report_path')
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        directory_fd = os.open(path.anchor, flags)
        temporary = None
        try:
            for component in path.parts[1:-1]:
                next_fd = os.open(component, flags, dir_fd=directory_fd)
                os.close(directory_fd)
                directory_fd = next_fd
            try:
                mode = os.stat(path.name, dir_fd=directory_fd, follow_symlinks=False).st_mode
                if not stat.S_ISREG(mode):
                    raise ValueError('report_target_not_regular')
            except FileNotFoundError:
                pass
            temporary = '.' + path.name + '.checks-' + uuid.uuid4().hex + '.tmp'
            fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                         0o600, dir_fd=directory_fd)
            with os.fdopen(fd, 'wb') as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path.name, src_dir_fd=directory_fd, dst_dir_fd=directory_fd)
            temporary = None
        finally:
            if temporary is not None:
                try:
                    os.unlink(temporary, dir_fd=directory_fd)
                except OSError:
                    pass
            os.close(directory_fd)
