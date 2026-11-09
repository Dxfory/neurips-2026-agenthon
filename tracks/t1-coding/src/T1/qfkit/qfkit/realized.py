"""Realized measures from intraday prices: anchored grid sampling, realized variance on several
grids, microstructure-noise variance (Bandi-Russell), noise-corrected realized variance
(additive bias correction, Zhang-Mykland-Ait-Sahalia two-scales) and bipower variation.

Conventions for the whole module:
- Prices are LEVELS, time-ordered, with unique timestamps. Clean them first: which duplicate
  quote to keep (first, last, median) and which hours form the session are the task's choice,
  not this module's.
- Returns are LOG returns between consecutive KEPT prices of the same calendar day. A gap
  (missing minutes) gives one longer return; nothing is forward-filled unless
  rule='previous_tick' asks for it; there is no overnight return.
- n always counts RETURNS (kept prices minus one), per day.
- Variances are per day in squared log-return units, not annualised (multiply by e.g. 252 for
  an annual variance). A noise correction can make a variance negative; it is NOT clipped:
  only clip inside the square root when you turn it into a volatility, sqrt(max(v, 0)).
- The grid of step f minutes is {anchor + k f : k integer} on each calendar day (time of day
  of the timestamps' own clock / time zone).

Functions
---------
subsample_grid      keep the prices on an anchored f-minute grid ('on_grid'), or the previous tick
realized_measures   per-day RV, n, bipower variation, noise variance and corrected RV on several grids
noise_variance      i.i.d. noise variance omega^2: Bandi-Russell RV/(2n) or minus the first autocovariance
noise_corrected_rv  RV - 2 n omega^2 (additive) or the two-scales TSRV from subsample-grid RVs

References: Andersen, Bollerslev, Diebold and Labys (2001) JASA 96, (2003) Econometrica 71;
Bandi and Russell (2006) J. Financial Economics 79, (2008) Review of Economic Studies 75;
Zhang, Mykland and Ait-Sahalia (2005) JASA 100; Barndorff-Nielsen and Shephard (2004)
J. Financial Econometrics 2; Hansen and Lunde (2006) J. Business & Economic Statistics 24.
"""

from __future__ import annotations

import datetime as _dt
import math

import numpy as np
import pandas as pd

__all__ = [
    "subsample_grid",
    "realized_measures",
    "noise_variance",
    "noise_corrected_rv",
]

_NS_PER_MIN = 60_000_000_000
_NS_PER_DAY = 24 * 60 * _NS_PER_MIN
_HALF_PI = 0.5 * math.pi          # mu_1^{-2}, mu_1 = E|Z| = sqrt(2/pi)


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _choice(value, name, allowed):
    if not isinstance(value, str) or value not in allowed:
        raise ValueError(f"{name} must be one of {', '.join(repr(a) for a in allowed)}, got {value!r}")
    return value


def _anchor_ns(anchor):
    """Nanoseconds after midnight of the anchor time of day ('HH:MM', 'HH:MM:SS' or datetime.time)."""
    if isinstance(anchor, _dt.time):
        t = anchor
    elif isinstance(anchor, str):
        t = None
        for fmt in ("%H:%M", "%H:%M:%S"):
            try:
                t = _dt.datetime.strptime(anchor.strip(), fmt).time()
                break
            except ValueError:
                continue
        if t is None:
            raise ValueError(f"anchor must be a time of day such as '09:30' or '09:30:00', got {anchor!r}")
    else:
        raise TypeError("anchor must be a time of day: 'HH:MM', 'HH:MM:SS' or datetime.time")
    return ((t.hour * 60 + t.minute) * 60 + t.second) * 1_000_000_000 + t.microsecond * 1000


def _step_ns(every_minutes, name="every_minutes"):
    if isinstance(every_minutes, bool):
        raise TypeError(f"{name} must be a number of minutes")
    f = float(every_minutes)
    if not (math.isfinite(f) and f > 0):
        raise ValueError(f"{name} must be a positive number of minutes")
    step = int(round(f * _NS_PER_MIN))
    if step <= 0 or abs(step - f * _NS_PER_MIN) > 1.0:
        raise ValueError(f"{name} must be a whole number of nanoseconds")
    return step


def _times_prices(timestamps, prices):
    """(DatetimeIndex, float prices) after checking length, finiteness and strict time order."""
    if isinstance(timestamps, (pd.Series, pd.Index)):
        ts = pd.DatetimeIndex(pd.to_datetime(timestamps))
    else:
        ts = pd.DatetimeIndex(pd.to_datetime(np.asarray(timestamps).ravel()))
    p = np.asarray(prices, dtype=float).ravel()
    if p.size != len(ts):
        raise ValueError(f"timestamps ({len(ts)}) and prices ({p.size}) differ in length")
    if ts.hasnans:
        raise ValueError("timestamps contain NaT")
    if not np.all(np.isfinite(p)):
        raise ValueError("prices contain NaN/inf; drop them first")
    ns = _to_ns(ts)
    if ns.size > 1 and np.any(np.diff(ns) <= 0):
        raise ValueError("timestamps must be strictly increasing: sort and de-duplicate first "
                         "(collapsing duplicates is the task's rule, e.g. the median quote)")
    return ts, p


def _to_ns(ts):
    """Wall-clock nanoseconds since the epoch of a DatetimeIndex (its own time zone's clock)."""
    naive = ts.tz_localize(None) if ts.tz is not None else ts
    return naive.to_numpy().astype("datetime64[ns]").astype(np.int64)


def _days(ts):
    """[(day start ns, start index, stop index)] for each calendar day of a sorted index."""
    ns = _to_ns(ts)
    day = np.floor_divide(ns, _NS_PER_DAY) * _NS_PER_DAY
    cut = np.flatnonzero(np.diff(day)) + 1
    starts = np.concatenate(([0], cut))
    stops = np.concatenate((cut, [ns.size]))
    return [(int(day[a]), int(a), int(b)) for a, b in zip(starts, stops)], ns


def _sample_day(tod, p, step, anchor, rule):
    """Grid indices k (time = anchor + k step) and prices of one day; tod = ns after midnight."""
    off = tod - anchor
    if rule == "on_grid":
        keep = np.mod(off, step) == 0
        return np.floor_divide(off[keep], step), p[keep]
    if tod.size == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0)
    k0 = -np.floor_divide(-off[0], step)      # ceil: first grid point at or after the first tick
    k1 = np.floor_divide(off[-1], step)       # last grid point at or before the last tick
    k = np.arange(k0, k1 + 1, dtype=np.int64)
    pos = np.searchsorted(off, k * step, side="right") - 1
    return k, p[pos]


def _rv_bv(logp, scaling):
    """(n returns, RV, BV) of one day's log prices; RV nan if n = 0, BV nan if n < 2."""
    r = np.diff(logp)
    n = r.size
    rv = float(r @ r) if n >= 1 else math.nan
    if n >= 2:
        a = np.abs(r)
        bv = _HALF_PI * float(a[1:] @ a[:-1])
        if scaling == "n_over_n_minus_1":
            bv *= n / (n - 1.0)
    else:
        bv = math.nan
    return n, rv, bv


def _noise_one(r, method):
    n = r.size
    if method == "bandi_russell":
        return float(r @ r) / (2.0 * n) if n >= 1 else math.nan
    return -float(r[1:] @ r[:-1]) / (n - 1.0) if n >= 2 else math.nan


def _label(g):
    return f"{g:g}"


# ----------------------------------------------------------------------------
# sampling
# ----------------------------------------------------------------------------
def subsample_grid(timestamps, prices, *, every_minutes, anchor, rule):
    """Calendar-time sampling of intraday prices on an anchored f-minute grid (Andersen-Bollerslev-Diebold-Labys
    realized-variance sampling; previous-tick rule of Hansen-Lunde).

    Parameters
    ----------
    timestamps : array_like of datetimes (DatetimeIndex, Series, strings), strictly
        increasing (sort and de-duplicate first), tz-naive or tz-aware; may span several days.
    prices : array_like, the price LEVEL at each timestamp (finite).
    every_minutes : REQUIRED. Grid step f in minutes (> 0; 0.5 = 30 seconds).
    anchor : REQUIRED. Time of day the grid is counted from, 'HH:MM', 'HH:MM:SS' or a
        datetime.time, e.g. '09:30' for the US cash open. The grid on each calendar day is
        {anchor + k f, k integer}.
    rule : REQUIRED.
        'on_grid'       keep only the observations that lie exactly ON the grid, i.e. whose
                        minutes after the anchor are a multiple of f: ((h*60 + m) - anchor) % f == 0
                        for minute data. Nothing is filled: a missing grid minute is simply absent.
        'previous_tick' one price per grid point from the first to the last grid point inside the
                        day's observed range: the last observation at or before the grid time of
                        that same day (last-tick interpolation; forward-fills inside the day).

    Returns
    -------
    pd.Series of the sampled prices (name 'price'), indexed by the observation times
    ('on_grid') or the grid times ('previous_tick'), in time order.

    Formula
    -------
    off = time of day - anchor (exact, in nanoseconds); 'on_grid' keeps off mod f == 0.
    'previous_tick': for grid time g = day + anchor + k f with first tick <= g <= last tick,
    price(g) = p(max{t_j <= g}). Returns between consecutive kept prices are the sampled
    log returns of realized variance (n = kept prices - 1 per day).

    References
    ----------
    Andersen, Bollerslev, Diebold and Labys (2001) JASA 96, (2003) Econometrica 71 (calendar-time
    sampled realized variance); Hansen and Lunde (2006) JBES 24 (previous-tick interpolation).

    Example
    -------
    >>> s = subsample_grid(df.timestamp, df.mid, every_minutes=5, anchor='09:30', rule='on_grid')
    >>> r = np.diff(np.log(s.values))          # one day: its 5-minute log returns
    """
    rule = _choice(rule, "rule", ("on_grid", "previous_tick"))
    step = _step_ns(every_minutes)
    anc = _anchor_ns(anchor)
    ts, p = _times_prices(timestamps, prices)
    days, ns = _days(ts)
    out_ns, out_p = [], []
    for day, a, b in days:
        k, pk = _sample_day(ns[a:b] - day, p[a:b], step, anc, rule)
        out_ns.append(day + anc + k * step)
        out_p.append(pk)
    grid_ns = np.concatenate(out_ns) if out_ns else np.zeros(0, dtype=np.int64)
    idx = pd.DatetimeIndex(grid_ns.astype("datetime64[ns]"))
    if ts.tz is not None:
        idx = idx.tz_localize(ts.tz)
    idx.name = ts.name
    return pd.Series(np.concatenate(out_p) if out_p else np.zeros(0), index=idx, name="price")


# ----------------------------------------------------------------------------
# noise variance and corrections
# ----------------------------------------------------------------------------
def noise_variance(returns, *, method):
    """Microstructure-noise variance omega^2 of i.i.d. noise: Bandi-Russell RV/(2n) or the negative first
    autocovariance of returns.

    Parameters
    ----------
    returns : the highest-frequency intraday LOG returns: array_like (n,) for one day; (D, n)
        or a DataFrame with one row per day; or a list of 1-D arrays (days with different n).
    method : REQUIRED.
        'bandi_russell'  omega^2 = RV / (2n) = sum r_j^2 / (2n), n = number of returns
                         (Bandi-Russell 2006 Sec. 3.1; 2008).
        'autocov'        omega^2 = -(1/(n-1)) sum_{j=2..n} r_j r_{j-1} (Roll / Zhang-Mykland-Ait-
                         Sahalia; can be negative, not clipped).

    Returns
    -------
    float for one day; ndarray (D,) otherwise (Series for a DataFrame). nan for a day with
    fewer than 1 ('bandi_russell') or 2 ('autocov') returns.

    Formula
    -------
    Observed log price = efficient price + u, u i.i.d. (0, omega^2): r_j = e_j + u_j - u_{j-1}, so
    E[r_j^2] = IV/n + 2 omega^2 and Cov(r_j, r_{j-1}) = -omega^2. Hence E[RV] = IV + 2 n omega^2:
    Bandi-Russell is biased up by IV/(2n) (negligible at high frequency); 'autocov' is unbiased
    for i.i.d. noise. The grand mean across days is np.mean of the per-day values.

    References
    ----------
    Bandi and Russell (2006), "Separating microstructure noise from volatility", J. Financial
    Economics 79, Sec. 3.1; Bandi and Russell (2008), Review of Economic Studies 75; Zhang,
    Mykland and Ait-Sahalia (2005) JASA 100; Roll (1984) J. Finance 39.

    Example
    -------
    >>> omega2 = noise_variance(np.diff(np.log(prices_1min)), method='bandi_russell')
    """
    method = _choice(method, "method", ("bandi_russell", "autocov"))
    if isinstance(returns, (list, tuple)) and len(returns) and all(np.ndim(x) == 1 for x in returns):
        rows = [np.asarray(x, dtype=float) for x in returns]
        for x in rows:
            if not np.all(np.isfinite(x)):
                raise ValueError("returns contain NaN/inf; drop them first")
        return np.array([_noise_one(x, method) for x in rows])
    index = returns.index if isinstance(returns, pd.DataFrame) else None
    a = np.asarray(returns, dtype=float)
    if a.ndim not in (1, 2):
        raise ValueError("returns must be 1-D (one day), 2-D (days x intraday) or a list of 1-D arrays")
    if not np.all(np.isfinite(a)):
        raise ValueError("returns contain NaN/inf; drop them first")
    if a.ndim == 1:
        return _noise_one(a, method)
    out = np.array([_noise_one(row, method) for row in a])
    return pd.Series(out, index=index, name="noise_var") if index is not None else out


def noise_corrected_rv(rv, n, omega2, *, method):
    """Noise-corrected realized variance: Bandi-Russell additive bias correction RV - 2 n omega^2, or the
    Zhang-Mykland-Ait-Sahalia two-scales estimator (TSRV).

    Parameters
    ----------
    rv : realized variance(s) (sum of squared log returns), per day.
    n : number of RETURNS behind rv (kept prices - 1), same shape as rv.
    omega2 : noise variance omega^2 (from noise_variance; a grand mean across days or each
        day's own value - the task decides), broadcast against rv.
    method : REQUIRED.
        'additive'  RV_n - 2 n omega^2 (rv, n elementwise).
        'tsrv'      rv and n hold the K subsample-grid values on their LAST axis: the RVs of
                    the K offset sparse grids (offsets 0, 1, ..., K-1 fine steps) and their
                    return counts. TSRV = mean_k RV^(k) - 2 nbar omega^2, nbar = mean_k n^(k).
                    With omega^2 = RV_all / (2 n_all) (Bandi-Russell on all fine returns)
                    this is ZMA's RV^(avg) - (nbar / n_all) RV^(all). K = 1 gives 'additive'.

    Returns
    -------
    float for scalar input (or one day of K values for 'tsrv'); otherwise ndarray (a Series
    keeps its index for 'additive'). NOT clipped at zero: a quiet day can be negative.

    Formula
    -------
    E[RV_n] = IV + 2 n omega^2 under i.i.d. noise, so RV_n - 2 n omega^2 is unbiased for IV when
    omega^2 is. ZMA's small-sample adjustment (1 - nbar/n_all)^{-1} is NOT applied.

    References
    ----------
    Bandi and Russell (2006) J. Financial Economics 79; (2008) Review of Economic Studies 75;
    Zhang, Mykland and Ait-Sahalia (2005), "A tale of two time scales", JASA 100.

    Example
    -------
    >>> rv5_corr = noise_corrected_rv(rv5, n5, omega2_grand_mean, method='additive')
    """
    method = _choice(method, "method", ("additive", "tsrv"))
    if method == "additive":
        if isinstance(rv, pd.Series):
            return rv - 2.0 * np.asarray(n, dtype=float) * np.asarray(omega2, dtype=float)
        out = np.asarray(rv, dtype=float) - 2.0 * np.asarray(n, dtype=float) * np.asarray(omega2, dtype=float)
        return float(out) if out.ndim == 0 else out
    rv_a = np.asarray(rv, dtype=float)
    n_a = np.asarray(n, dtype=float)
    if rv_a.ndim == 0 or rv_a.shape != n_a.shape:
        raise ValueError("for 'tsrv', rv and n must have the same shape with the K subsample grids on the last axis")
    out = rv_a.mean(axis=-1) - 2.0 * n_a.mean(axis=-1) * np.asarray(omega2, dtype=float)
    return float(out) if np.ndim(out) == 0 else out


# ----------------------------------------------------------------------------
# the per-day table
# ----------------------------------------------------------------------------
def realized_measures(day_prices, *, grids, anchor, rule, noise_estimator, correction, omega_source,
                      bv_scaling):
    """Per-day realized variance on several sampling grids (Andersen-Bollerslev-Diebold-Labys), Bandi-Russell
    noise variance, noise-corrected RV (additive, or Zhang-Mykland-Ait-Sahalia TSRV) and
    Barndorff-Nielsen-Shephard bipower variation.

    Parameters
    ----------
    day_prices : pd.Series of cleaned price LEVELS indexed by timestamp (DatetimeIndex, strictly
        increasing, finite, > 0), one or many calendar days (a one-column DataFrame is accepted).
    grids : REQUIRED. Grid steps in minutes, e.g. (1, 2, 5, 15, 30). The noise variance is
        estimated on the FINEST grid, min(grids).
    anchor : REQUIRED. Grid origin as a time of day, e.g. '09:30' (see subsample_grid).
    rule : REQUIRED. 'on_grid' (keep observations exactly on the grid, no fill) or
        'previous_tick' (see subsample_grid).
    noise_estimator : REQUIRED. 'bandi_russell' (RV/(2n)) or 'autocov' (see noise_variance),
        applied to each day's finest-grid returns.
    correction : REQUIRED. 'additive' (RV_f - 2 n_f omega^2), 'tsrv' (two-scales: the mean RV of
        the K = f/min(grids) offset sub-grids of the finest grid minus 2 nbar omega^2; f must be a
        multiple of min(grids)) or 'none' (rv_corr = rv).
    omega_source : REQUIRED. 'grand_mean' (one omega^2 for all days: the arithmetic mean of the
        per-day estimates, NaN days skipped) or 'per_day' (each day its own estimate).
    bv_scaling : REQUIRED. 'none' (pi/2 sum_{j>=2} |r_j||r_{j-1}|, BNS 2004 Definition 1) or
        'n_over_n_minus_1' (times n/(n-1)), as riskstats.bipower_variation.

    Returns
    -------
    dict
        table : pd.DataFrame indexed by calendar day ('date', time order) with, for each grid f
                (label f'{f:g}', e.g. 'rv_5'): rv_f (sum of squared log returns), n_f (number of
                returns, int), bv_f, rv_corr_f (NOT clipped), vol_corr_f = sqrt(max(rv_corr_f, 0))
                (daily; times sqrt(252) annualises), n_bar_f (only for 'tsrv'); plus noise_var
                (the day's omega^2 on the finest grid) and omega2_used.
        omega2_grand : float, the mean of noise_var across days.

    Formula
    -------
    Per day and grid: r = diff(log p) over the sampled prices; RV = sum r^2; n = len(r);
    BV = (pi/2) sum_{j=2..n} |r_j| |r_{j-1}|. Under i.i.d. noise E[RV_f] = IV + 2 n_f omega^2, so
    rv_corr_f = RV_f - 2 n_f omega^2. TSRV: RV^(avg) = (1/K) sum_k RV of the sub-grid k, k f_min +
    j K f_min (j integer), nbar = mean sub-grid n.

    References
    ----------
    Andersen, Bollerslev, Diebold and Labys (2003) Econometrica 71; Bandi and Russell (2006)
    JFE 79, Sec. 3.1, (2008) RES 75; Zhang, Mykland and Ait-Sahalia (2005) JASA 100;
    Barndorff-Nielsen and Shephard (2004) J. Financial Econometrics 2, Definition 1.

    Example
    -------
    >>> s = df.groupby('timestamp')['mid'].median()            # the task's own cleaning
    >>> s = s[(s.index.time >= time(9, 30)) & (s.index.time <= time(16, 0))]
    >>> out = realized_measures(s, grids=(1, 2, 5, 15, 30), anchor='09:30', rule='on_grid',
    ...                         noise_estimator='bandi_russell', correction='additive',
    ...                         omega_source='grand_mean', bv_scaling='none')
    >>> out['table'][['rv_5', 'n_5', 'bv_5', 'rv_corr_5']]; out['omega2_grand']
    """
    rule = _choice(rule, "rule", ("on_grid", "previous_tick"))
    noise_estimator = _choice(noise_estimator, "noise_estimator", ("bandi_russell", "autocov"))
    correction = _choice(correction, "correction", ("additive", "tsrv", "none"))
    omega_source = _choice(omega_source, "omega_source", ("grand_mean", "per_day"))
    bv_scaling = _choice(bv_scaling, "bv_scaling", ("none", "n_over_n_minus_1"))
    if isinstance(day_prices, pd.DataFrame):
        if day_prices.shape[1] != 1:
            raise ValueError("day_prices must be a Series (or a one-column DataFrame) of prices")
        day_prices = day_prices.iloc[:, 0]
    if not isinstance(day_prices, pd.Series) or not isinstance(day_prices.index, pd.DatetimeIndex):
        raise TypeError("day_prices must be a pd.Series of prices indexed by timestamp (DatetimeIndex)")
    if isinstance(grids, (str, bytes)) or np.ndim(grids) != 1 or len(grids) == 0:
        raise ValueError("grids must be a non-empty sequence of grid steps in minutes, e.g. (1, 5)")
    grids = list(grids)
    steps = [_step_ns(g, "grids") for g in grids]
    if len(set(steps)) != len(steps):
        raise ValueError("grids contains a step twice")
    step0 = min(steps)
    if correction == "tsrv" and any(s % step0 for s in steps):
        raise ValueError("'tsrv' needs every grid step to be a whole multiple of the finest, min(grids)")
    anc = _anchor_ns(anchor)
    ts, p = _times_prices(day_prices.index, day_prices.to_numpy())
    if np.any(p <= 0):
        raise ValueError("prices must be > 0 (log returns)")
    days, ns = _days(ts)
    labels = [_label(g) for g in grids]
    rows = []
    for day, a, b in days:
        tod, lp_day = ns[a:b] - day, np.log(p[a:b])
        row = {}
        k0, lp0 = _sample_day(tod, lp_day, step0, anc, rule)
        row["noise_var"] = _noise_one(np.diff(lp0), noise_estimator)
        for lab, step in zip(labels, steps):
            _, lp = _sample_day(tod, lp_day, step, anc, rule)
            n, rv, bv = _rv_bv(lp, bv_scaling)
            row[f"rv_{lab}"], row[f"n_{lab}"], row[f"bv_{lab}"] = rv, n, bv
            if correction == "tsrv":
                K = step // step0
                parts = [_rv_bv(lp0[np.mod(k0, K) == j], "none")[:2] for j in range(K)]
                row[f"rvavg_{lab}"] = float(np.mean([0.0 if pn == 0 else prv for pn, prv in parts]))
                row[f"n_bar_{lab}"] = float(np.mean([pn for pn, _ in parts]))
        rows.append(row)
    idx = pd.DatetimeIndex(np.array([d for d, _, _ in days], dtype=np.int64).astype("datetime64[ns]"), name="date")
    if ts.tz is not None:
        idx = idx.tz_localize(ts.tz)
    raw = pd.DataFrame(rows, index=idx)
    noise = raw["noise_var"].to_numpy(dtype=float)
    finite = noise[np.isfinite(noise)]
    grand = float(finite.mean()) if finite.size else math.nan
    used = np.full(noise.shape, grand) if omega_source == "grand_mean" else noise
    table = pd.DataFrame(index=idx)
    for lab in labels:
        rv = raw[f"rv_{lab}"].to_numpy(dtype=float)
        n = raw[f"n_{lab}"].to_numpy(dtype=np.int64)
        table[f"rv_{lab}"] = rv
        table[f"n_{lab}"] = n
        table[f"bv_{lab}"] = raw[f"bv_{lab}"].to_numpy(dtype=float)
        if correction == "additive":
            corr = rv - 2.0 * n * used
        elif correction == "tsrv":
            nbar = raw[f"n_bar_{lab}"].to_numpy(dtype=float)
            corr = raw[f"rvavg_{lab}"].to_numpy(dtype=float) - 2.0 * nbar * used
            table[f"n_bar_{lab}"] = nbar
        else:
            corr = rv.copy()
        table[f"rv_corr_{lab}"] = corr
        with np.errstate(invalid="ignore"):
            table[f"vol_corr_{lab}"] = np.sqrt(np.maximum(corr, 0.0))
    table["noise_var"] = noise
    table["omega2_used"] = used
    return {"table": table, "omega2_grand": grand}
