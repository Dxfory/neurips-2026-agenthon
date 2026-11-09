"""Select bounded numerical diagnostics from operations and current inputs.

This module never selects by task identity and never computes a task deliverable.
``method`` is the caller's declaration of the algorithm actually used; it cannot
be authenticated from a number. Only recognized, different algorithm families
are eligible for a cross-method comparison. Arbitrary new method labels do not
establish independence. Residuals, compensated sums, and probability constraints
are consistency/identity checks, not independent solutions.

Callbacks must be deterministic for fixed inputs. Quadrature and derivative
errors are estimates for sufficiently smooth functions,
not rigorous interval bounds or coverage guarantees. Lack of convergence, an
exhausted evaluation cap, or an ambiguous tolerance boundary stays unverified.
The requested tolerances are never enlarged to accommodate a reference estimate.
"""
from __future__ import annotations

import math
import sys

from .checks import Checks, _number, _numbers
from .semantics import QuantitySpec

__all__ = ['verify', 'verify_quantity']

_INTEGRAL_METHODS = frozenset(('trapezoid', 'trapezoidal', 'composite_trapezoid',
                             'midpoint', 'gauss_legendre', 'gaussian_quadrature',
                             'analytic', 'closed_form', 'symbolic'))
_DERIVATIVE_METHODS = frozenset(('analytic', 'closed_form', 'symbolic', 'autodiff',
                               'automatic_differentiation'))


class _Unavailable(Exception):
    pass


class _Sampler:
    def __init__(self, function, limit):
        if not callable(function):
            raise ValueError('function_required')
        self.function, self.limit, self.values = function, limit, {}
        self.calls = 0

    def __call__(self, point):
        point = _number(point)
        if point not in self.values:
            if self.calls >= self.limit:
                raise _Unavailable('reference_evaluation_limit')
            self.calls += 1
            try:
                self.values[point] = _number(self.function(point))
            except Exception as exc:
                raise _Unavailable('invalid_reference_evaluation') from exc
        return self.values[point]


def _scalar(value):
    shape, numbers = _numbers(value)
    if shape:
        raise ValueError('scalar_required')
    return numbers[0]


def _tolerance(reference, atol, rtol):
    return _number(atol + rtol * abs(reference), nonnegative=True)


def _integral(sample, lower, upper, atol, rtol):
    lower, upper = _scalar(lower), _scalar(upper)
    sign = 1. if lower <= upper else -1.
    a, b = sorted((lower, upper))
    if a == b:
        return 0., 0.
    mid = a / 2 + b / 2
    if mid == a or mid == b:
        raise _Unavailable('interval_below_float_resolution')
    fa, fm, fb = sample(a), sample(mid), sample(b)
    initial = _number((b-a) * (fa+4*fm+fb) / 6)
    budget = _tolerance(initial, atol, rtol) / 16
    if budget <= 0:
        raise _Unavailable('no_reference_error_budget')
    stack = [(a, b, fa, fm, fb, initial, budget, 0)]
    values, errors = [], []
    while stack:
        a, b, fa, fm, fb, whole, local_budget, depth = stack.pop()
        mid = a / 2 + b / 2
        left_mid, right_mid = a / 2 + mid / 2, mid / 2 + b / 2
        if left_mid in (a, mid) or right_mid in (mid, b) or depth >= 28:
            raise _Unavailable('reference_not_converged')
        fl, fr = sample(left_mid), sample(right_mid)
        left = _number((mid-a) * (fa+4*fl+fm) / 6)
        right = _number((b-mid) * (fm+4*fr+fb) / 6)
        combined = _number(left + right)
        delta = _number(combined - whole)
        roundoff = _number(8 * sys.float_info.epsilon * (abs(left)+abs(right)+abs(whole)))
        error = _number(abs(delta)/15 + roundoff)
        if error <= local_budget:
            values.append(_number(combined+delta/15)); errors.append(error)
        else:
            stack.append((mid, b, fm, fr, fb, right, local_budget/2, depth+1))
            stack.append((a, mid, fa, fl, fm, left, local_budget/2, depth+1))
    reference, error = sign * math.fsum(values), math.fsum(errors)
    if error > _tolerance(reference, atol, rtol)/8:
        raise _Unavailable('reference_not_converged')
    return _number(reference), _number(error)


def _derivative(sample, point, atol, rtol):
    point = _scalar(point)
    center = sample(point)
    step = max(1., abs(point)) / 64
    previous, previous_gap = None, None
    for _ in range(32):
        if point+step == point or point-step == point:
            break
        plus, minus = sample(point+step), sample(point-step)
        central = _number((plus-minus)/(2*step))
        gap = _number(abs((plus-center)/step - (center-minus)/step))
        if previous is not None:
            reference = _number((4*central-previous)/3)
            roundoff = _number(8*sys.float_info.epsilon*(abs(plus)+abs(minus)+2*abs(center))/step)
            error = _number(abs(central-previous)/3 + roundoff)
            tolerance = _tolerance(reference, atol, rtol)
            # A symmetric difference at a cusp can be exactly zero. Require the
            # one-sided disagreement to contract, or already be below precision.
            sides_converging = (gap <= tolerance or gap <= .75*previous_gap)
            if tolerance > 0 and error <= tolerance/16 and sides_converging:
                return reference, error
        previous, previous_gap = central, gap
        step /= 2
    raise _Unavailable('reference_not_converged')


def _result(status, strategy, evidence, reason='', error=None, evaluations=0):
    return {'status': status, 'strategy': strategy, 'evidence': evidence,
            'reason': reason, 'estimated_error': error, 'evaluations': evaluations}


def _skip(checks, name, reason, strategy=None, evaluations=0):
    checks.skip(name, reason)
    return _result('unverified', strategy, 'unverified', reason, evaluations=evaluations)


def _invalid(checks, name):
    # Keep invalid inputs distinct from a successful mathematical comparison;
    # the public Checks interface records a bounded failed diagnostic.
    checks.bounds(name, math.nan, lower=0., upper=0.)
    return _result('failed', None, 'unverified', 'invalid_input')


def verify(checks: Checks, name, *, operation, actual, method, inputs,
           atol, rtol, max_evaluations=4097):
    """Verify one declared operation without changing data, precision, or outputs.

    ``inputs`` schemas: integral: function/lower/upper; derivative: function/x;
    linear_system: matrix/rhs; sum or cashflows: parts; probabilities: empty dict.
    Arrays are rectangular, finite, and never broadcast. The linear residual is
    ``A @ actual`` versus the *original* rhs, row by row; it does not establish
    conditioning or small solution error. Probability vectors must be one
    dimensional, lie in [0,1], and sum to 1. Probability bounds use no slack.

    Configuration errors (invalid tolerance/evaluation cap) raise ValueError.
    Invalid data record a failed check. An unavailable alternative records a
    Checks.skip and returns status='unverified'. Callback errors are not exposed
    in reports. A numerical reference near the tolerance boundary is likewise
    unverified; estimated error is never added to the allowed task tolerance.
    """
    if not isinstance(checks, Checks):
        raise TypeError('checks must be Checks')
    atol, rtol = _number(atol, nonnegative=True), _number(rtol, nonnegative=True)
    if (type(max_evaluations) is not int or not 3 <= max_evaluations <= 65537):
        raise ValueError('max_evaluations must be between 3 and 65537')
    if not isinstance(inputs, dict) or not isinstance(method, str) or not method.strip():
        return _invalid(checks, name)
    sample, strategy = None, None
    try:
        if operation in ('integral', 'derivative'):
            actual = _scalar(actual)
            expected_keys = {'function', 'lower', 'upper'} if operation == 'integral' else {'function', 'x'}
            if set(inputs) != expected_keys:
                raise ValueError('invalid_inputs')
            for field in expected_keys - {'function'}:
                _scalar(inputs[field])
            recognized = _INTEGRAL_METHODS if operation == 'integral' else _DERIVATIVE_METHODS
            if method.strip().lower() not in recognized:
                return _skip(checks, name, 'no_distinct_recognized_reference')
            strategy = 'adaptive_simpson' if operation == 'integral' else 'central_difference_richardson'
            sample = _Sampler(inputs['function'], max_evaluations)
            try:
                if operation == 'integral':
                    reference, error = _integral(sample, inputs['lower'], inputs['upper'], atol, rtol)
                else:
                    reference, error = _derivative(sample, inputs['x'], atol, rtol)
            except (ValueError, ArithmeticError) as exc:
                raise _Unavailable('reference_numeric_failure') from exc
            tolerance = _tolerance(reference, atol, rtol)
            difference = _number(abs(actual-reference))
            uncertainty = _number(error*(1+rtol))
            if abs(difference-tolerance) <= uncertainty and uncertainty > 0:
                return _skip(checks, name, 'reference_uncertainty_at_tolerance', strategy, sample.calls)
            ok = checks.close(name, actual, reference, atol=atol, rtol=rtol,
                              computed_method=method.strip().lower(), reference_method=strategy)
            return _result('passed' if ok else 'failed', strategy, 'declared_cross_method',
                           error=error, evaluations=sample.calls)
        if operation == 'linear_system':
            if set(inputs) != {'matrix', 'rhs'}:
                raise ValueError('invalid_inputs')
            matrix_shape, matrix = _numbers(inputs['matrix'])
            x_shape, vector = _numbers(actual)
            rhs_shape, rhs = _numbers(inputs['rhs'])
            if (len(matrix_shape) != 2 or x_shape != (matrix_shape[1],)
                    or rhs_shape != (matrix_shape[0],)):
                raise ValueError('shape_mismatch')
            width = matrix_shape[1]
            residual_side = [math.fsum(_number(matrix[i*width+j]*vector[j]) for j in range(width))
                             for i in range(matrix_shape[0])]
            ok = checks.close(name, residual_side, rhs, atol=atol, rtol=rtol,
                              computed_method='linear_residual', reference_method='linear_residual')
            return _result('passed' if ok else 'failed', 'linear_residual', 'consistency')
        if operation in ('sum', 'cashflows'):
            if set(inputs) != {'parts'}:
                raise ValueError('invalid_inputs')
            actual = _scalar(actual)
            shape, parts = _numbers(inputs['parts'])
            if len(shape) != 1:
                raise ValueError('vector_required')
            reference = _number(math.fsum(parts))
            # Checks.balance scales rtol by its total argument. The reference
            # must therefore be that total, never an erroneous actual result.
            ok = checks.balance(name, [actual], reference, atol=atol, rtol=rtol)
            return _result('passed' if ok else 'failed', 'compensated_sum_identity', 'identity')
        if operation == 'probabilities':
            if inputs:
                raise ValueError('invalid_inputs')
            shape, probabilities = _numbers(actual)
            if len(shape) != 1:
                raise ValueError('vector_required')
            bounded = checks.bounds(name+'.bounds', probabilities, lower=0., upper=1., atol=0)
            normalized = checks.balance(name+'.mass', probabilities, 1., atol=atol, rtol=rtol)
            return _result('passed' if bounded and normalized else 'failed',
                           'probability_constraints', 'consistency')
        return _skip(checks, name, 'no_applicable_operation_rule')
    except _Unavailable as exc:
        return _skip(checks, name, str(exc), strategy, sample.calls if sample else 0)
    except (ValueError, TypeError, OverflowError, ArithmeticError):
        return _invalid(checks, name)


def verify_quantity(checks, name, *, quantity: QuantitySpec, actual, operation, inputs,
                    max_evaluations=4097):
    """Use the shared definition as the sole source of method and task precision.

    Unit, shape, axes and time semantics are enforced by enforce_contract during
    preflight; this adapter does not infer them. A missing method, atol, or rtol is
    unverified. There are deliberately no tolerance or method override arguments.
    Source attachment describes provenance, not proof of semantic interpretation.
    """
    if not isinstance(quantity, QuantitySpec):
        raise TypeError('quantity must be QuantitySpec')
    if any(getattr(quantity, field) is None for field in ('method', 'atol', 'rtol')):
        result = _skip(checks, name, 'quantity_method_or_precision_undefined')
    else:
        result = verify(checks, name, operation=operation, actual=actual, method=quantity.method,
                        inputs=inputs, atol=quantity.atol, rtol=quantity.rtol,
                        max_evaluations=max_evaluations)
    return dict(result, quantity_fingerprint=quantity.fingerprint)
