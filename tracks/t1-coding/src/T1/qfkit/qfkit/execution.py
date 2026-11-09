"""Volume profiles and integer execution allocations.

Pure numerical functions: the caller supplies complete, chronological sessions,
selects bars eligible for an order, and aligns market volumes to scheduled bars.
No file access, clock access, market calendar or random generator is used.
"""
from __future__ import annotations

import numbers

import numpy as np

__all__ = ["volume_share_profile", "profile_r2", "largest_remainder", "participation_fills"]

_MAX_QUANTITY = 2**53 - 1


def _quantity(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{name} must be a nonnegative integer")
    if value < 0 or value > _MAX_QUANTITY:
        raise ValueError(f"{name} must lie in [0, 2**53-1]")
    return int(value)


def _finite(value, name):
    a = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be finite; clean missing observations explicitly")
    return a


def volume_share_profile(volume, *, method, half_life=None,
                         winsor_quantiles=(.05, .95), quantile_method="linear", day_axis=0):
    """Estimate one intraday volume-share profile from complete historical days.

    Parameters
    ----------
    volume : array_like, 2-D
        Nonnegative bar volumes. Days are chronological oldest to newest; each
        day must have positive total volume. Daily shares are computed BEFORE
        aggregation, so a high-volume session does not receive more weight.
    method : {'mean', 'median', 'ewma', 'winsorized'}
        Aggregate daily shares per bar, then renormalize the result to sum one.
        Winsorization clips EACH BAR across historical days before averaging.
    half_life : float > 0, optional
        REQUIRED for ewma, in days. Finite-history weights are proportional to
        exp(-log(2) * lag / half_life); the last historical day has lag zero.
    winsor_quantiles : pair of floats, default (.05, .95)
        Lower and upper probabilities with 0 <= lower <= upper <= 1.
    quantile_method : str, default 'linear'
        NumPy quantile interpolation used for winsorization.
    day_axis : {0, 1}, default 0
        Which axis contains historical days. Output always indexes bars.

    Returns
    -------
    ndarray, shape (n_bars,)
        Nonnegative profile with sum one. No incomplete-day filtering occurs.

    Example
    -------
    >>> volume_share_profile([[80, 20], [20, 80]], method='ewma', half_life=1)
    array([0.4, 0.6])
    """
    a = _finite(volume, "volume")
    if a.ndim != 2 or not all(a.shape) or np.any(a < 0):
        raise ValueError("volume must be a nonempty, nonnegative 2-D array")
    if isinstance(day_axis, (bool, np.bool_)) or day_axis not in (0, 1):
        raise ValueError("day_axis must be 0 or 1")
    if day_axis == 1:
        a = a.T
    # Scaling each row first avoids overflow in otherwise valid positive sums.
    row_max = a.max(axis=1, keepdims=True)
    if np.any(row_max == 0):
        raise ValueError("each historical day must have positive total volume")
    scaled = a / row_max
    shares = scaled / scaled.sum(axis=1, keepdims=True)
    if method == "mean":
        profile = shares.mean(axis=0)
    elif method == "median":
        profile = np.median(shares, axis=0)
    elif method == "ewma":
        if half_life is None or not np.isfinite(half_life) or half_life <= 0:
            raise ValueError("ewma requires positive, finite half_life in days")
        lag = np.arange(len(shares) - 1, -1, -1, dtype=float)
        weights = np.exp(-np.log(2.) * (lag / float(half_life)))
        profile = weights @ shares / weights.sum()
    elif method == "winsorized":
        q = np.asarray(winsor_quantiles, dtype=float)
        if q.shape != (2,) or not np.all(np.isfinite(q)) or not 0 <= q[0] <= q[1] <= 1:
            raise ValueError("winsor_quantiles must satisfy 0 <= lower <= upper <= 1")
        cuts = np.quantile(shares, q, axis=0, method=quantile_method)
        profile = np.clip(shares, cuts[0], cuts[1]).mean(axis=0)
    else:
        raise ValueError("method must be mean, median, ewma or winsorized")
    total = float(profile.sum())
    if total <= 0 or not np.isfinite(total):
        raise ValueError("aggregated profile has zero or invalid mass")
    return profile / total


def profile_r2(actual, predicted, *, axis=-1, zero_sst="exact"):
    """Centered R-squared of actual and predicted profiles.

    Parameters
    ----------
    actual, predicted : array_like
        Finite profiles. Broadcasting is allowed, for example one predicted
        (bars,) profile against actual (days, bars) daily shares.
    axis : int, default -1
        Axis over which squared errors and centered actual variation are summed.
    zero_sst : {'exact', 'raise'}, default 'exact'
        For a constant actual profile, exact equality gives 1 and a different
        prediction gives 0. 'raise' rejects any zero denominator instead.

    Returns
    -------
    float or ndarray
        1 - SSE/SST, retaining non-profile axes. Negative scores are possible.

    Example
    -------
    >>> profile_r2([.5, .5], [.4, .6])
    0.0
    """
    a, p = np.broadcast_arrays(_finite(actual, "actual"), _finite(predicted, "predicted"))
    if a.ndim == 0 or a.size == 0:
        raise ValueError("profiles must be nonempty and have at least one axis")
    if not isinstance(axis, numbers.Integral) or isinstance(axis, (bool, np.bool_)):
        raise ValueError("axis must be an integer")
    if axis < -a.ndim or axis >= a.ndim:
        raise ValueError("axis is outside profile dimensions")
    if zero_sst not in ("exact", "raise"):
        raise ValueError("zero_sst must be 'exact' or 'raise'")
    sse = np.sum((a - p)**2, axis=axis)
    sst = np.sum((a - a.mean(axis=axis, keepdims=True))**2, axis=axis)
    if zero_sst == "raise" and np.any(sst == 0):
        raise ValueError("constant actual profile has zero SST")
    ratio = np.divide(sse, sst, out=np.zeros_like(sse, dtype=float), where=sst != 0)
    result = np.where(sst == 0, np.where(sse == 0, 1., 0.), 1. - ratio)
    return float(result) if result.ndim == 0 else result


def largest_remainder(quantity, weights, *, tie_order=None):
    """Hamilton allocation: distribute an integer quantity by nonnegative weights.

    Parameters
    ----------
    quantity : int >= 0
        Exact total quantity, at most 2**53-1 (the exact integer float64 range).
    weights : array_like, shape (n,)
        Finite nonnegative weights; they need not sum to one. Positive quantity
        requires positive total weight. Zero quantity returns zeros even when
        all weights are zero.
    tie_order : array_like, shape (n,), optional
        Ascending secondary keys for equal fractional remainders, e.g. bar start
        timestamps. When omitted, original position breaks ties. Equal keys
        themselves retain original position.

    Returns
    -------
    ndarray of int64, shape (n,)
        Floor allocations plus one for the largest remaining fractions, with
        nonnegative entries and exact sum quantity. Filter eligible bars BEFORE
        calling; renormalization is performed on the supplied weights.

    Example
    -------
    >>> largest_remainder(7, [1, 1, 1], tie_order=[30, 10, 20])
    array([2, 3, 2])
    """
    q = _quantity(quantity, "quantity")
    w = _finite(weights, "weights")
    if w.ndim != 1 or np.any(w < 0):
        raise ValueError("weights must be a nonnegative 1-D vector")
    keys = np.arange(len(w)) if tie_order is None else np.asarray(tie_order)
    if keys.shape != w.shape:
        raise ValueError("tie_order must have one key per weight")
    if np.issubdtype(keys.dtype, np.number) and not np.all(np.isfinite(keys)):
        raise ValueError("numeric tie_order keys must be finite")
    if q == 0:
        return np.zeros(len(w), dtype=np.int64)
    if not len(w) or not np.any(w > 0):
        raise ValueError("positive quantity requires positive total weight")
    precise = w.astype(np.longdouble)
    precise /= precise.max()
    raw = np.longdouble(q) * precise / precise.sum()
    allocated = np.floor(raw).astype(np.int64)
    remaining = q - sum(map(int, allocated))
    if not 0 <= remaining <= len(w):
        raise ArithmeticError("allocation precision exceeded available fractional units")
    remainders = raw - allocated
    try:
        order = sorted(range(len(w)), key=lambda i: (-remainders[i], keys[i], i))
    except TypeError as exc:
        raise ValueError("tie_order keys must be mutually orderable") from exc
    allocated[order[:remaining]] += 1
    return allocated


def participation_fills(target, market_volume, *, participation, carry_unfilled=True):
    """Simulate integer fills limited by per-bar participation capacity.

    Parameters
    ----------
    target : int >= 0 or array_like of nonnegative integers, shape (bars,)
        A scalar is an order released before the first bar. A vector releases
        each scheduled child quantity at its bar. No future quantity is filled
        early. An empty market-volume vector leaves a scalar order unfilled.
    market_volume : array_like, shape (bars,)
        Finite nonnegative observed share volumes, aligned to eligible bars.
    participation : float or array_like, broadcast to (bars,)
        Maximum market-volume fraction in [0, 1]. Integer capacity is
        floor(participation * market_volume), with no fractional-share fills.
    carry_unfilled : bool, default True
        Carry blocked quantities to later bars. False expires each child at its
        own bar; expired quantities still count toward total unfilled quantity.

    Returns
    -------
    dict
        'fills' and 'capacity' are int64 bar vectors; 'scheduled' records releases;
        'remaining' is cumulative releases minus cumulative fills at each bar;
        'unfilled' is the final integer quantity not executed. Fills are unsigned:
        the caller applies a buy/sell sign and its price/fee convention.

    Example
    -------
    >>> participation_fills(9, [20, 30, 40], participation=.1)['fills']
    array([2, 3, 4])
    """
    volume = _finite(market_volume, "market_volume")
    if volume.ndim != 1 or np.any(volume < 0):
        raise ValueError("market_volume must be a nonnegative vector")
    if not isinstance(carry_unfilled, (bool, np.bool_)):
        raise ValueError("carry_unfilled must be a bool")
    rate = _finite(participation, "participation")
    try:
        rate = np.broadcast_to(rate, volume.shape)
    except ValueError as exc:
        raise ValueError("participation must broadcast to market_volume") from exc
    if np.any((rate < 0) | (rate > 1)):
        raise ValueError("participation must lie in [0, 1]")
    raw_capacity = volume * rate
    if np.any(raw_capacity > _MAX_QUANTITY):
        raise ValueError("bar capacity exceeds exact integer range")
    capacity = np.floor(raw_capacity).astype(np.int64)
    scheduled = np.zeros(len(volume), dtype=np.int64)
    if np.ndim(target) == 0:
        total = _quantity(target, "target")
        if len(scheduled):
            scheduled[0] = total
    else:
        original = np.asarray(target)
        if original.shape != volume.shape:
            raise ValueError("target schedule must align with market_volume")
        for i, value in enumerate(original):
            scheduled[i] = _quantity(value, "target quantity")
        total = sum(map(int, scheduled))
        _quantity(total, "total target")
    fills = np.zeros(len(volume), dtype=np.int64)
    pending = 0
    for i, release in enumerate(scheduled):
        available = pending + int(release) if carry_unfilled else int(release)
        fills[i] = min(available, int(capacity[i]))
        pending = available - int(fills[i]) if carry_unfilled else 0
    remaining = np.cumsum(scheduled) - np.cumsum(fills)
    return {"fills": fills, "capacity": capacity, "scheduled": scheduled,
            "remaining": remaining, "unfilled": total - sum(map(int, fills))}
