"""Fixed-count row batching and shifted, mergeable mean/covariance summaries.

These helpers draw no random numbers and choose no sampling or financial model.
Batch moments use centered products; merging uses the parallel variance identity
M2 = M2_a + M2_b + outer(mean_b-mean_a) * n_a*n_b/(n_a+n_b).
An origin plus a small mean offset avoids subtracting rounded large means.
"""
from __future__ import annotations

from numbers import Integral

import numpy as np

__all__ = ['batch_slices', 'OnlineMoments']


def _integer(value, name, minimum):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return int(value)


def batch_slices(total, *, batch_size):
    """Yield consecutive row slices covering an exact fixed sample count.

    Parameters
    ----------
    total : nonnegative integer number of rows. Zero yields no slices.
    batch_size : positive integer maximum rows in each slice. The final slice
        may be shorter. Booleans and nonintegers are rejected on iteration.

    Returns
    -------
    iterator of slice : [0,total) in original order, using constant state.
        No samples are generated, stored, omitted, shuffled or reseeded. Keep
        the caller's one RNG and its prescribed draw order across iterations.

    Example
    -------
    >>> [(s.start, s.stop) for s in batch_slices(7, batch_size=3)]
    [(0, 3), (3, 6), (6, 7)]
    """
    count = _integer(total, 'total', 0)
    size = _integer(batch_size, 'batch_size', 1)
    for start in range(0, count, size):
        yield slice(start, min(count, start + size))


def _samples(values):
    raw = np.asarray(values)
    if raw.ndim not in (1, 2) or raw.size == 0:
        raise ValueError('samples must be a nonempty vector or (observations, features) matrix')
    if raw.dtype.kind not in 'iuf' or raw.dtype.kind == 'f' and raw.dtype.itemsize > 8:
        raise ValueError('samples require real integer or at-most-float64 floating dtype; no implicit precision reduction')
    values = np.asarray(raw, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError('samples must all be finite')
    if raw.dtype.kind in 'iu':
        try:
            with np.errstate(over='raise', invalid='raise'):
                exact = np.array_equal(values.astype(raw.dtype), raw)
        except FloatingPointError:
            exact = False
        if not exact:
            raise ValueError('integer samples must be exactly representable as float64')
    return values[:, None] if values.ndim == 1 else values


def _finite(*arrays):
    if not all(np.all(np.isfinite(value)) for value in arrays):
        raise FloatingPointError('moment arithmetic produced a nonfinite result')


class OnlineMoments:
    """Accumulate mergeable float64 means, variances and cross-covariances.

    Parameters
    ----------
    None : construct an empty accumulator. Call update(samples) or merge(other).
        A vector means scalar observations; a matrix has rows=observations and
        columns=features. Feature count must be unchanged. Large offsets are
        tracked separately from mean deviations; arithmetic remains float64.

    Returns
    -------
    OnlineMoments : mutable accumulator with count and dimension properties,
        update/merge methods and result(ddof=...) returning n, mean, variance,
        covariance and se. Vector outputs always have shape (features,), and
        covariance has shape (features,features). State costs O(features**2);
        update costs O(batch_rows*features + features**2), retaining no samples.
        Use a small number of reduced output features, not a column per path.

    Notes
    -----
    se is sqrt(sample variance / row count). This assumes independent input
    observations; this helper cannot establish independence or remove bias.
    For common random numbers, form per-row paired differences BEFORE update.
    For antithetic draws, form each complete pair's average BEFORE update;
    the count must be the number of independent pairs, not twice that number.
    Never average batch standard errors or concatenate independently drawn
    bump estimates. No sample count, random order, dtype or method is reduced
    to meet a resource budget. Invalid batches and arithmetic failures leave
    the accumulator unchanged. Floating-point grouping can change final bits.

    Example
    -------
    >>> stats = OnlineMoments().update([1., 2.]).update([3., 4.])
    >>> stats.result(ddof=1)['mean'].tolist()
    [2.5]
    """
    def __init__(self):
        self._count = 0
        self._origin = None
        self._offset = None
        self._m2 = None

    @property
    def count(self):
        """Number of supplied observations (pairs if the caller pre-paired them)."""
        return self._count

    @property
    def dimension(self):
        """Feature count, or None before the first nonempty update/merge."""
        return None if self._origin is None else len(self._origin)

    def _combine(self, count, origin, offset, m2):
        if self._count == 0:
            return count, origin.copy(), offset.copy(), m2.copy()
        if len(origin) != self.dimension:
            raise ValueError('feature dimension must stay unchanged')
        total = self._count + count
        with np.errstate(over='raise', invalid='raise', divide='raise', under='raise'):
            delta = (origin - self._origin) + (offset - self._offset)
            weight = count / total
            new_offset = self._offset + delta * weight
            between = np.outer(delta, delta) * (self._count * weight)
            if np.any((delta != 0) & (np.diag(between) == 0)):
                raise FloatingPointError('between-batch variance underflow')
            new_m2 = self._m2 + m2 + between
            _finite(new_offset, new_m2, self._origin + new_offset)
        return total, self._origin.copy(), new_offset, new_m2

    def update(self, samples):
        """Add one nonempty finite batch atomically and return self.

        Inputs may be real floating arrays up to float64 or integer arrays
        whose values are exactly float64-representable. Object, string, bool,
        complex and higher-precision floating arrays raise ValueError. Empty
        batches, changing feature count and nonfinite values also raise.
        Overflow/underflow in moment arithmetic raises FloatingPointError.
        """
        values = _samples(samples)
        if self.dimension is not None and values.shape[1] != self.dimension:
            raise ValueError('feature dimension must stay unchanged')
        with np.errstate(over='raise', invalid='raise', divide='raise', under='raise'):
            origin = values[0].copy()
            centered = values - origin
            offset = centered.mean(axis=0)
            centered -= offset
            m2 = centered.T @ centered
            if np.any(np.any(centered != 0, axis=0) & (np.diag(m2) == 0)):
                raise FloatingPointError('within-batch variance underflow')
            _finite(origin, offset, m2, origin + offset)
            combined = self._combine(len(values), origin, offset, m2)
        self._count, self._origin, self._offset, self._m2 = combined
        return self

    def merge(self, other):
        """Merge another accumulator atomically, leaving other unchanged.

        Empty accumulators are identities. Dimensions must match. Merging self
        is rejected to prevent an accidental doubling of the same observations.
        Separate overlapping datasets are not detectable; the caller must keep
        the observation sets disjoint and preserve their statistical semantics.
        """
        if not isinstance(other, OnlineMoments):
            raise TypeError('other must be OnlineMoments')
        if other is self:
            raise ValueError('cannot merge an accumulator into itself')
        if other.count == 0:
            return self
        combined = self._combine(other._count, other._origin, other._offset, other._m2)
        self._count, self._origin, self._offset, self._m2 = combined
        return self

    def result(self, *, ddof):
        """Return detached arrays of mean, variance, covariance and mean SE.

        ddof is an explicit nonnegative integer smaller than count. Covariance
        uses denominator count-ddof; se uses sqrt(variance)/sqrt(count). A single
        observation admits ddof=0, but cannot provide a sample-variance SE with
        ddof=1. Returned arrays never alias internal state.
        """
        correction = _integer(ddof, 'ddof', 0)
        if correction >= self.count:
            raise ValueError('need more observations than ddof')
        with np.errstate(over='raise', invalid='raise', divide='raise', under='raise'):
            mean = self._origin + self._offset
            covariance = self._m2 / (self.count - correction)
            if np.any((self._m2 != 0) & (covariance == 0)):
                raise FloatingPointError('covariance underflow')
            variance = np.diag(covariance).copy()
            se = np.sqrt(variance) / np.sqrt(self.count)
            _finite(mean, covariance, variance, se)
        return {'n': self.count, 'mean': mean.copy(), 'variance': variance,
                'covariance': covariance, 'se': se}
