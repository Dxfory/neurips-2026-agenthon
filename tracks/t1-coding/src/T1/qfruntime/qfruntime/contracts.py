"""Execute contracts defined by the shared semantic layer.

No field is inferred from values or names. Unknown definitions remain unverified;
known conflicts still fail even if another field is unknown. Explicit unit
conversion belongs to semantics.convert_unit and must happen before this check.
A source quote anchors a caller's interpretation; it does not prove that the
interpretation follows from that quote, or that declared units/methods were used.
These are diagnostic contracts, not a trusted scorer.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib

from .checks import Checks, _array, _number
from .semantics import QuantitySpec, TaskSpec, canonical_unit

__all__ = ['enforce_contract', 'enforce_task']

_REQUIRED = ('unit', 'shape', 'axes', 'method', 'time_basis')
_MEANINGS = ('axes', 'method', 'time_basis', 'timezone', 'period')


def _matches_shape(pattern, actual):
    return len(pattern) == len(actual) and all(want is None or want == got
                                              for want, got in zip(pattern, actual))


def _validate(value, expected, observed):
    shape, flat = _array(value)
    for item in flat:
        _number(item)
    if observed is not None:
        if not isinstance(observed, QuantitySpec):
            raise ValueError('observed_spec_required')
        if observed.shape is not None and not _matches_shape(observed.shape, shape):
            raise ValueError('observed_shape_mismatch')
    if expected is None:
        return
    if not isinstance(observed, QuantitySpec):
        raise ValueError('observed_spec_required')
    if expected.name != observed.name:
        raise ValueError('quantity_name_mismatch')
    if expected.shape is not None:
        if observed.shape is None or not _matches_shape(expected.shape, shape):
            raise ValueError('expected_shape_mismatch')
    if expected.unit is not None:
        if observed.unit is None or canonical_unit(expected.unit) != canonical_unit(observed.unit):
            raise ValueError('unit_mismatch')
    for field in _MEANINGS:
        wanted, got = getattr(expected, field), getattr(observed, field)
        if wanted is not None and wanted != got:
            raise ValueError('semantic_mismatch_' + field)


def _record(checks, name, expected, observed, error=None):
    # Reuse the bounded, atomic Checks reporting machinery. The existing contract
    # row schema is unchanged; only definition fingerprints, never source text or
    # actual data arrays, are retained as specification metadata.
    parameters = {'expected_spec': expected.fingerprint if expected is not None else None,
                  'observed_spec': observed.fingerprint if isinstance(observed, QuantitySpec) else None}

    def compute():
        if error:
            raise ValueError(error)
        return True, None, None
    checks._run(name, 'contract', parameters, compute, 'contract', raise_error=True)


def _enforce(checks, name, value, expected, observed, context_missing=()):
    if not isinstance(checks, Checks):
        raise TypeError('checks must be Checks')
    if expected is not None and not isinstance(expected, QuantitySpec):
        raise TypeError('expected must be QuantitySpec or None')
    try:
        _validate(value, expected, observed)
    except (ValueError, TypeError, ArithmeticError) as exc:
        _record(checks, name, expected, observed, 'invalid_value_or_semantic_conflict')
        raise AssertionError('failed contract must raise') from exc
    missing = list(context_missing)
    if expected is None:
        missing.append('expected_definition_missing')
    else:
        missing.extend('undefined_' + field for field in _REQUIRED if getattr(expected, field) is None)
        if expected.source is None:
            missing.append('source_reference_missing')
    result = {'status': 'unverified' if missing else 'passed', 'reasons': missing,
              'expected_fingerprint': expected.fingerprint if expected is not None else None,
              'observed_fingerprint': observed.fingerprint if isinstance(observed, QuantitySpec) else None}
    if missing:
        checks.skip(name, 'semantic_definition_incomplete')
    else:
        _record(checks, name, expected, observed)
    return result


def enforce_contract(checks, name, value, *, expected: QuantitySpec | None,
                     observed: QuantitySpec | None):
    """Check a finite quantity against explicit expected and observed meanings.

    A passing definition supplies unit, shape, axes, method, time_basis and a
    source reference. Scalars use shape=(), axes=(); static quantities explicitly
    use time_basis='static'. Timezone and period are checked whenever specified.
    Unknown expected fields yield unverified, while known requirements absent or
    different in observed metadata fail. Both supplied shapes must fit the actual
    rectangular value; wildcard dimensions never permit axis-name substitution.

    Failed contracts are recorded before ValueError is raised. Tolerances in the
    QuantitySpec belong to numerical verification and cannot excuse unit, axis,
    method, timezone, or period conflicts. No implicit conversion is performed.
    """
    return _enforce(checks, name, value, expected, observed)


def enforce_task(checks, task: TaskSpec, values, observed_specs):
    """Execute all quantity contracts and retain unresolved requirements.

    values and observed_specs map exact quantity names. Missing declared values,
    unexpected values, or semantic conflicts fail; other quantities are still
    checked before raising. A TaskSpec source reference must have a bound content
    hash (normally produced by TaskSpec.from_dict with sources). A bare constructor
    supplies declarations, not evidence. Even with hashes, semantic interpretation
    remains caller-supplied. An empty or unresolved task cannot pass as complete.
    Check identities bind the complete task fingerprint. Distinct TaskSpecs can
    share a Checks report, retaining earlier failures; repeating the same TaskSpec
    still triggers Checks duplicate-ID protection. Use a fresh Checks instance for
    a new execution attempt. Changed definitions cannot evade repair guards, which
    reject removal or replacement of earlier check identities.
    """
    if not isinstance(checks, Checks) or not isinstance(task, TaskSpec):
        raise TypeError('Checks and TaskSpec are required')
    if not isinstance(values, Mapping) or not isinstance(observed_specs, Mapping):
        raise TypeError('values and observed_specs must be mappings')
    task_identity = task.fingerprint
    defined = {q.name for q in task.quantities}
    results, failures = {}, []
    if set(values)-defined or set(observed_specs)-defined:
        try:
            _record(checks, 'task.unexpected.' + task_identity, None, None, 'unexpected_quantity')
        except ValueError:
            failures.append('unexpected_quantity')
    bound_sources = dict(task.source_hashes)
    for quantity in task.quantities:
        name = 'task.quantity.' + hashlib.sha256((task_identity + ':' + quantity.name).encode()).hexdigest()
        observed = observed_specs.get(quantity.name)
        try:
            if quantity.name not in values:
                _record(checks, name, quantity, observed, 'missing_quantity')
            missing = (() if quantity.source is not None and quantity.source.source_id in bound_sources
                       else ('unbound_task_source',))
            results[quantity.name] = _enforce(checks, name, values[quantity.name], quantity, observed, missing)
        except ValueError:
            failures.append(quantity.name)
    if failures:
        raise ValueError('task contract failed')
    reasons = []
    if not task.quantities:
        reasons.append('no_quantity_definitions')
    if task.unresolved:
        reasons.append('unresolved_task_requirements')
    if reasons:
        checks.skip('task.incomplete.' + task_identity, reasons[0])
    complete = not reasons and all(result['status'] == 'passed' for result in results.values())
    return {'status': 'passed' if complete else 'unverified', 'reasons': reasons,
            'task_fingerprint': task.fingerprint, 'quantities': results}
