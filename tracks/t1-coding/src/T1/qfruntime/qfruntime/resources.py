"""Small, explicit resource measurements and cooperative numerical batching.

RSS is the operating system's *process-lifetime high-water mark*, including work
before this helper and native allocations. It is not current RSS, a batch's
incremental allocation, child-process memory, or a hard resource limit. Linux
reports ru_maxrss in KiB; macOS reports bytes. Unknown platforms stay unverified.

Batch callbacks receive a private random.Random. They must themselves respect
the requested count and statistical assumptions. Budgets are checked between
batches, not enforced by preemption; one callback can overrun a budget.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import islice
import math
from numbers import Integral, Real
import random
from statistics import NormalDist
import sys
import time

try:
    import resource
except ImportError:  # Explicitly unavailable on platforms without getrusage.
    resource = None

__all__ = ['ResourceBudget', 'measure_resources', 'run_batches']


def _nonnegative(value, name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(name + ' must be finite and nonnegative')
    try:
        converted = float(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(name + ' must be finite and nonnegative') from exc
    if not math.isfinite(converted) or converted < 0:
        raise ValueError(name + ' must be finite and nonnegative')
    return converted


def _positive_int(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(name + ' must be a positive integer')
    return value


@dataclass(frozen=True)
class ResourceBudget:
    """Cooperative wall-time and absolute process-lifetime RSS ceilings.

    None means no ceiling for that resource; zero stops before any work. RSS
    includes earlier allocations and can conservatively reject a new batch.
    An unavailable RSS reading cannot satisfy a declared memory ceiling.
    """
    wall_seconds: float | None = None
    rss_bytes: int | None = None

    def __post_init__(self):
        if self.wall_seconds is not None:
            object.__setattr__(self, 'wall_seconds', _nonnegative(self.wall_seconds, 'wall_seconds'))
        if self.rss_bytes is not None and (type(self.rss_bytes) is not int or self.rss_bytes < 0):
            raise ValueError('rss_bytes must be a nonnegative integer')


def _rss():
    report = {'rss_peak_bytes': None, 'rss_status': 'unverified',
              'rss_scope': 'process_lifetime_high_water', 'rss_source': 'unavailable'}
    if resource is None or sys.platform not in ('linux', 'darwin'):
        return report
    try:
        raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if not math.isfinite(raw) or raw < 0:
            return report
        multiplier = 1024 if sys.platform == 'linux' else 1
        report.update(rss_peak_bytes=int(raw * multiplier), rss_status='measured',
                      rss_source='getrusage_linux_kib' if multiplier == 1024 else 'getrusage_macos_bytes')
    except (OSError, ValueError, OverflowError):
        pass
    return report


def _measurement(start, cpu_start):
    return {'elapsed_seconds': max(0., time.perf_counter() - start),
            'cpu_seconds': max(0., time.process_time() - cpu_start), **_rss()}


def measure_resources(compute):
    """Run a zero-argument callback once, returning its result and measurements.

    Exceptions propagate. CPU is process CPU, elapsed time is monotonic wall
    time, and RSS is the lifetime peak of this process (not just this callback).
    """
    if not callable(compute):
        raise ValueError('compute must be callable')
    start, cpu_start = time.perf_counter(), time.process_time()
    result = compute()
    return {'result': result, **_measurement(start, cpu_start)}


def _budget_reason(budget, measured):
    if budget.wall_seconds is not None and measured['elapsed_seconds'] >= budget.wall_seconds:
        return 'time_budget'
    if budget.rss_bytes is not None:
        if measured['rss_status'] != 'measured':
            return 'memory_unverified'
        if measured['rss_peak_bytes'] >= budget.rss_bytes:
            return 'memory_budget'
    return None


def run_batches(draw, *, max_samples, batch_size, required_min_samples=None,
                absolute_error=None, confidence=.95, seed=None, budget=None):
    """Aggregate scalar samples using bounded batches and shifted Welford variance.

    ``draw(count, rng)`` must yield exactly count finite real numbers. Invalid
    batches commit no samples. Integers not exactly representable as float and
    variance arithmetic that over/underflows are rejected, never treated as zero
    estimated error. Fixed-count mode requires all max_samples. With
    absolute_error, stop only after required_min_samples and an estimated
    normal-SE half-width <= the unchanged target; default minimum=max_samples.
    Failure to meet the target within max_samples or resource ceilings returns
    incomplete. A single sample cannot verify variance or an error target.

    The reported half-width assumes IID finite-variance observations and a
    normal approximation. Repeated checks/optional stopping do not guarantee
    confidence coverage. No statistical correctness or task pass is certified.
    The caller must handle dependence, heavy tails and model bias separately.
    Budgets include callback and aggregation time but cannot interrupt either.
    """
    if not callable(draw):
        raise ValueError('draw must be callable')
    max_samples = _positive_int(max_samples, 'max_samples')
    batch_size = _positive_int(batch_size, 'batch_size')
    minimum = max_samples if required_min_samples is None else _positive_int(required_min_samples, 'required_min_samples')
    if minimum > max_samples:
        raise ValueError('required_min_samples must not exceed max_samples')
    if absolute_error is not None:
        absolute_error = _nonnegative(absolute_error, 'absolute_error')
    confidence = _nonnegative(confidence, 'confidence')
    probability = (1. + confidence) / 2.
    if not 0 < confidence < 1 or not .5 < probability < 1:
        raise ValueError('confidence must be representably between zero and one')
    z = NormalDist().inv_cdf(probability)
    if seed is not None and type(seed) is not int:
        raise ValueError('seed must be an integer or None')
    if budget is None:
        budget = ResourceBudget()
    if not isinstance(budget, ResourceBudget):
        raise ValueError('budget must be ResourceBudget or None')
    rng = random.Random(seed)
    start, cpu_start = time.perf_counter(), time.process_time()
    count, batches, requested, attempts = 0, 0, 0, 0
    mean, m2, origin = 0., 0., None
    status, reason, error_type = 'incomplete', 'max_samples', None

    def estimates():
        if count < 2:
            return None, None, None
        variance = m2 / (count - 1)
        if variance == 0 and m2 != 0:
            return None, None, None
        se = math.sqrt(variance) / math.sqrt(count)
        width = z * se
        if not all(math.isfinite(x) for x in (variance, se, width)):
            return None, None, None
        return variance, se, width

    while count < max_samples:
        reason = _budget_reason(budget, _measurement(start, cpu_start))
        if reason:
            break
        size = min(batch_size, max_samples - count)
        requested += size
        attempts += 1
        try:
            samples = list(islice(iter(draw(size, rng)), size + 1))
        except Exception as exc:
            reason, error_type = 'batch_error', type(exc).__name__
            break
        if len(samples) != size:
            reason = 'invalid_batch'
            break
        new_count, new_mean, new_m2, new_origin = count, mean, m2, origin
        try:
            for sample in samples:
                if isinstance(sample, bool) or not isinstance(sample, Real):
                    raise ValueError('sample must be real')
                value = float(sample)
                if not math.isfinite(value):
                    raise ValueError('sample must be finite')
                if isinstance(sample, Integral) and int(value) != int(sample):
                    raise ValueError('integer sample loses precision as float')
                if new_count == 0:
                    new_origin = value
                shifted = value - new_origin
                new_count += 1
                delta = shifted - new_mean
                new_mean += delta / new_count
                increment = delta * (shifted - new_mean)
                if delta != 0 and increment == 0:
                    raise ArithmeticError('variance increment lost to roundoff or underflow')
                new_m2 += increment
                if not math.isfinite(new_origin + new_mean) or not math.isfinite(new_m2) or new_m2 < 0:
                    raise ArithmeticError('unrepresentable statistics')
        except ArithmeticError:
            reason = 'unrepresentable_statistics'
            break
        except (ValueError, TypeError):
            reason = 'invalid_batch'
            break
        count, mean, m2, origin = new_count, new_mean, new_m2, new_origin
        batches += 1
        reason = _budget_reason(budget, _measurement(start, cpu_start))
        if reason:
            break
        _, _, half_width = estimates()
        if absolute_error is not None and count >= minimum and half_width is not None and half_width <= absolute_error:
            status, reason = 'completed', 'estimated_error_target'
            break
        if count == max_samples:
            status, reason = ('completed', 'target_samples') if absolute_error is None else ('incomplete', 'max_samples')
    variance, standard_error, half_width = estimates()
    return {'status': status, 'reason': reason, 'samples': count,
            'samples_requested': requested, 'batches': batches, 'attempted_batches': attempts,
            'max_samples': max_samples, 'batch_size': batch_size, 'required_min_samples': minimum,
            'absolute_error': absolute_error, 'confidence': confidence, 'seed': seed,
            'mean': origin + mean if count else None, 'sample_variance': variance,
            'standard_error': standard_error, 'estimated_half_width': half_width,
            'variance_status': 'estimated' if variance is not None else 'unverified',
            'statistical_assumptions': 'iid_finite_variance_normal_approximation',
            'coverage_guaranteed': False, 'budget_enforcement': 'cooperative_between_batches',
            'budget': {'wall_seconds': budget.wall_seconds, 'rss_bytes': budget.rss_bytes},
            'error_type': error_type, 'resources': _measurement(start, cpu_start)}
