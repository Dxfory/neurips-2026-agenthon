"""Signal construction: the exponential moving average (EMA) and cross-sectional tools (z-score, winsorizing,
quantile buckets, information coefficient).

Conventions for the whole module:
- A 2-D input is a panel: ROWS are dates, COLUMNS are assets (a DataFrame keeps its index and columns).
  The cross-sectional functions (cs_*, quantile_buckets, information_coefficient) work on each ROW; a 1-D input
  is one cross-section. ema works down each COLUMN; a 1-D input is one time series.
- In the cross-sectional functions NaN marks a missing asset on that date: it is skipped and stays NaN.
- Every choice tasks state differently (ddof, the zero-dispersion rule, the quantile method, the
  bucket rule, the correlation type, the EMA start) is a required keyword.

Functions
---------
ema                      EMA, alpha = 2/(span+1), started at the first value, at the SMA of the first span, or
                         with pandas' adjust=True weights
cs_zscore                per-date (x - mean)/sd with a stated ddof and zero-dispersion rule
cs_winsorize             per-date clip at the date's lower/upper quantiles
quantile_buckets         per-date buckets 1..n (1 = lowest): pd.qcut of the 'first' ranks or of the values
information_coefficient  per-date Spearman (rank IC) or Pearson correlation of a signal with forward returns,
                         with mean, sd, IR and t-statistic

Ported from the team's qflib (``backtest.ema_sma_seed`` and ``factors.cs_zscore``, ``cs_winsorize``,
``quantile_buckets``, ``information_coefficient``), generalised to NaN-aware panels with the conventions required.

References: Grinold and Kahn, Active Portfolio Management (2000) Ch. 6 and 12 (IC, information ratio); Fama and
French (1992) (portfolio sorts); Jegadeesh and Titman (1993); Murphy, Technical Analysis of the Financial Markets
(1999) Ch. 9 (EMA); pandas ``DataFrame.ewm`` and ``qcut`` documentation.
"""

from __future__ import annotations

import math
import numbers

import numpy as np
import pandas as pd
from scipy import signal as _signal
from scipy import stats

__all__ = [
    "ema",
    "cs_zscore",
    "cs_winsorize",
    "quantile_buckets",
    "information_coefficient",
]

_ZERO_STD = ("zero", "nan")
_BUCKET_RULES = ("ordinal", "quantile")
_IC_METHODS = ("spearman", "pearson")
_EMA_INIT = ("first", "sma", "adjusted")
_QUANTILE_METHODS = ("inverted_cdf", "averaged_inverted_cdf", "closest_observation", "interpolated_inverted_cdf",
                     "hazen", "weibull", "linear", "median_unbiased", "normal_unbiased", "lower", "higher",
                     "midpoint", "nearest")


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _choice(value, name, allowed):
    if not isinstance(value, str) or value not in allowed:
        raise ValueError(f"{name} must be one of {', '.join(repr(a) for a in allowed)}, got {value!r}")
    return value


def _is_int(v):
    return isinstance(v, numbers.Integral) and not isinstance(v, bool)


def _panel(x, name):
    """(2-D float array with one row per date, rebuild(values) giving the caller's type and shape)."""
    if isinstance(x, pd.DataFrame):
        a = x.to_numpy(dtype=float)
        return a, lambda v: pd.DataFrame(v, index=x.index, columns=x.columns)
    if isinstance(x, pd.Series):
        a = x.to_numpy(dtype=float)[None, :]
        return a, lambda v: pd.Series(v[0], index=x.index, name=x.name)
    a = np.asarray(x, dtype=float)
    if a.ndim == 1:
        return a[None, :], lambda v: v[0]
    if a.ndim != 2:
        raise ValueError(f"{name} must be 1-D (one cross-section) or 2-D (dates x assets), got shape {a.shape}")
    return a, lambda v: v


def _rows(a, func):
    """Apply func to the finite entries of each row; NaN stays NaN."""
    out = np.full(a.shape, np.nan)
    for i in range(a.shape[0]):
        ok = np.isfinite(a[i])
        if ok.any():
            out[i, ok] = func(a[i, ok])
    return out


# ----------------------------------------------------------------------------
# time series
# ----------------------------------------------------------------------------
def ema(x, *, span, init):
    """Exponential moving average (EMA) with alpha = 2/(span+1), started at the first value, at the simple moving
    average of the first span values, or with pandas' adjusted weights.

    Parameters
    ----------
    x : array_like (T,) or (T, N), Series or DataFrame; each column is one time series (oldest first), finite.
    span : REQUIRED. The EMA span (> 0; an integer >= 1 for init='sma'); alpha = 2/(span + 1).
    init : REQUIRED. How the recursion EMA_t = alpha x_t + (1 - alpha) EMA_{t-1} starts:
        'first'    EMA_0 = x_0 (pandas ``ewm(span=span, adjust=False).mean()``);
        'sma'      NaN before t = span-1, EMA_{span-1} = mean(x_0..x_{span-1}) (SMA-seeded EMA), then the recursion;
        'adjusted' EMA_t = sum_i (1-alpha)^i x_{t-i} / sum_i (1-alpha)^i over i = 0..t (pandas ``adjust=True``,
                   its default).

    Returns
    -------
    The EMA in the shape and type of x (a Series/DataFrame keeps its index).

    References
    ----------
    Murphy (1999) Ch. 9; Hunter (1986) "The exponentially weighted moving average", J. Quality Technology 18;
    pandas ``DataFrame.ewm``.

    Example
    -------
    >>> fast = ema(close, span=12, init='sma'); slow = ema(close, span=26, init='sma')
    >>> ema([1.0, 2.0, 3.0], span=3, init='first')  # alpha = 0.5: 1, 1.5, 2.25
    """
    mode = _choice(init, "init", _EMA_INIT)
    if isinstance(span, bool) or not isinstance(span, numbers.Real) or not span > 0:
        raise ValueError("span must be a number > 0 (alpha = 2/(span + 1))")
    if mode == "sma" and not (_is_int(span) or float(span).is_integer()):
        raise ValueError("init='sma' needs an integer span (the SMA covers the first span values)")
    if isinstance(x, pd.DataFrame):
        a, rebuild = x.to_numpy(dtype=float), (lambda v: pd.DataFrame(v, index=x.index, columns=x.columns))
    elif isinstance(x, pd.Series):
        a, rebuild = x.to_numpy(dtype=float)[:, None], (lambda v: pd.Series(v[:, 0], index=x.index, name=x.name))
    else:
        a = np.asarray(x, dtype=float)
        one = a.ndim == 1
        if a.ndim not in (1, 2):
            raise ValueError(f"x must be 1-D (one series) or 2-D (time x series), got shape {a.shape}")
        a = a[:, None] if one else a
        rebuild = (lambda v: v[:, 0]) if one else (lambda v: v)
    if a.shape[0] == 0:
        raise ValueError("x is empty")
    if not np.all(np.isfinite(a)):
        raise ValueError("x contains NaN/inf; drop or fill missing values first")
    alpha = 2.0 / (float(span) + 1.0)
    keep = 1.0 - alpha
    out = np.full(a.shape, np.nan)
    if mode == "adjusted":
        num = _signal.lfilter([1.0], [1.0, -keep], a, axis=0)
        den = _signal.lfilter([1.0], [1.0, -keep], np.ones(a.shape[0]))
        out = num / den[:, None]
    else:
        start = 0 if mode == "first" else int(span) - 1
        if start >= a.shape[0]:
            return rebuild(out)  # fewer than span values: no SMA seed yet
        seed = a[0] if mode == "first" else a[: start + 1].mean(axis=0)
        out[start] = seed
        if start + 1 < a.shape[0]:
            out[start + 1:] = _signal.lfilter([alpha], [1.0, -keep], a[start + 1:], axis=0,
                                              zi=(keep * seed)[None, :])[0]
    return rebuild(out)


# ----------------------------------------------------------------------------
# cross-section
# ----------------------------------------------------------------------------
def cs_zscore(x, *, ddof, zero_std):
    """Cross-sectional z-score per date: (x - mean)/sd over the assets present, with a stated ddof and rule for a
    date whose dispersion is zero.

    Parameters
    ----------
    x : array_like (N,) one cross-section, or (T, N) / DataFrame with ROWS = dates, COLUMNS = assets. NaN = missing.
    ddof : REQUIRED. 0 = population sd (divide by n), 1 = sample sd (n - 1); n = the date's non-missing count.
    zero_std : REQUIRED. A date whose sd is 0 (all values equal, up to 1e-12 of their size):
        'zero' its z-scores are 0; 'nan' they are NaN. A date with n <= ddof is NaN.

    Returns
    -------
    Same shape and type as x; NaN where x is NaN.

    References
    ----------
    Grinold and Kahn (2000) Ch. 14 (standardised scores).

    Example
    -------
    >>> z = cs_zscore(signal_panel, ddof=0, zero_std='zero')
    """
    if not _is_int(ddof) or ddof < 0:
        raise ValueError("ddof must be an int >= 0 (0 = population, 1 = sample standard deviation)")
    rule = _choice(zero_std, "zero_std", _ZERO_STD)
    a, rebuild = _panel(x, "x")

    def score(v):
        if v.size <= ddof:
            return np.full(v.size, np.nan)
        dev = v - v.mean()
        sd = math.sqrt(float(dev @ dev) / (v.size - ddof))
        if sd <= 1e-12 * max(float(np.max(np.abs(v))), 1e-300):
            return np.zeros(v.size) if rule == "zero" else np.full(v.size, np.nan)
        return dev / sd
    return rebuild(_rows(a, score))


def cs_winsorize(x, *, lower, upper, method):
    """Cross-sectional winsorizing per date: clip each date's values at that date's lower and upper quantiles.

    Parameters
    ----------
    x : array_like (N,) or (T, N) / DataFrame, ROWS = dates, COLUMNS = assets. NaN = missing (skipped).
    lower, upper : REQUIRED. Quantile PROBABILITIES (0.01, 0.99 for the 1st/99th percentiles), 0 <= lower < upper <= 1.
    method : REQUIRED. numpy quantile method of the cut-offs: 'linear' (np.percentile's default), 'lower',
        'higher', 'nearest', 'midpoint', 'inverted_cdf', ...

    Returns
    -------
    Same shape and type as x.

    References
    ----------
    Tukey and McLaughlin (1963) (winsorizing); numpy ``quantile`` methods (Hyndman and Fan 1996).

    Example
    -------
    >>> w = cs_winsorize(chars, lower=0.01, upper=0.99, method='linear')
    """
    m = _choice(method, "method", _QUANTILE_METHODS)
    for name, v in (("lower", lower), ("upper", upper)):
        if isinstance(v, bool) or not isinstance(v, numbers.Real) or not 0.0 <= float(v) <= 1.0:
            raise ValueError(f"{name} must be a probability in [0, 1] (0.01 = 1st percentile), not a percent")
    if not float(lower) < float(upper):
        raise ValueError("lower must be below upper")
    a, rebuild = _panel(x, "x")

    def clip(v):
        lo, hi = np.quantile(v, [float(lower), float(upper)], method=m)
        return np.clip(v, lo, hi)
    return rebuild(_rows(a, clip))


def quantile_buckets(x, *, n_buckets, rule):
    """Per-date portfolio sorts into buckets 1..n_buckets (quintiles, deciles), 1 = lowest values.

    Parameters
    ----------
    x : array_like (N,) or (T, N) / DataFrame, ROWS = dates, COLUMNS = assets. NaN = missing (no bucket).
    n_buckets : REQUIRED. Number of buckets (5 = quintiles, 10 = deciles), int >= 1.
    rule : REQUIRED.
        'ordinal'  rank stably (ties by position, pandas rank(method='first')) and cut the ranks 1..n at their
                   linear quantiles: near-equal counts, = pd.qcut(s.rank(method='first'), n_buckets) + 1;
        'quantile' breakpoints at the date's linear quantiles 0, 1/n_buckets, ..., 1; an asset is in bucket j
                   when edge_{j-1} < x <= edge_j (the lowest edge included), as pd.qcut(x, n_buckets).

    Returns
    -------
    Same shape and type as x: int labels, or float with NaN where x is NaN.

    References
    ----------
    Fama and French (1992); Jegadeesh and Titman (1993); pandas ``qcut``.

    Example
    -------
    >>> q = quantile_buckets(pred, n_buckets=5, rule='ordinal'); spread = ret[q == 5].mean() - ret[q == 1].mean()
    """
    if not _is_int(n_buckets) or n_buckets < 1:
        raise ValueError("n_buckets must be an int >= 1 (5 = quintiles)")
    kind = _choice(rule, "rule", _BUCKET_RULES)
    a, rebuild = _panel(x, "x")
    nb = int(n_buckets)

    def label(row):
        n = row.size
        if kind == "ordinal":
            v = np.empty(n)
            v[np.argsort(row, kind="mergesort")] = np.arange(1.0, n + 1.0)
        else:
            v = row
        edges = np.quantile(v, np.linspace(0.0, 1.0, nb + 1), method="linear")
        return np.digitize(v, edges[1:-1], right=True) + 1
    out = _rows(a, label)
    if np.all(np.isfinite(out)):
        out = out.astype(int)
    return rebuild(out)


def information_coefficient(signal, forward_returns, *, method, min_obs, ddof):
    """Information coefficient (IC) per date: the cross-sectional rank correlation (rank IC) or linear correlation
    of a signal with the returns that follow it, with its mean, sd, information ratio (IC IR) and t-statistic.

    Parameters
    ----------
    signal, forward_returns : array_like (N,) or (T, N) / DataFrames of the same shape, ROWS = dates. Row t of
        forward_returns must be the return earned AFTER signal row t is known (shift it yourself). NaN = missing;
        each date uses the assets where both are present.
    method : REQUIRED. 'spearman' (rank IC: Pearson correlation of average ranks) or 'pearson'.
    min_obs : REQUIRED. Fewest assets for a date's IC (int >= 2); dates below it are NaN and skipped.
    ddof : REQUIRED. ddof of the IC standard deviation across dates (1 = sample sd).

    Returns
    -------
    dict
        ic : per-date IC (Series on the dates of a DataFrame input, else an array); NaN where a date has fewer than
            min_obs pairs or a constant signal or return (the correlation is undefined).
        mean, std : mean and sd (with ddof) of the valid per-date ICs; ir : mean/std (the IC information ratio,
            not annualised); t_stat : mean/(std/sqrt(n)); n : number of valid dates.

    References
    ----------
    Grinold and Kahn (2000) Ch. 6 and 12; Qian, Hua and Sorensen, Quantitative Equity Portfolio Management (2007)
    Ch. 6.

    Example
    -------
    >>> res = information_coefficient(scores, fwd_ret, method='spearman', min_obs=3, ddof=1)
    >>> res['mean'], res['ir'], res['ic']
    """
    kind = _choice(method, "method", _IC_METHODS)
    if not _is_int(min_obs) or min_obs < 2:
        raise ValueError("min_obs must be an int >= 2")
    if not _is_int(ddof) or ddof < 0:
        raise ValueError("ddof must be an int >= 0 (1 = sample standard deviation)")
    s, _ = _panel(signal, "signal")
    r, _ = _panel(forward_returns, "forward_returns")
    if s.shape != r.shape:
        raise ValueError(f"signal and forward_returns must have the same shape, got {s.shape} and {r.shape}")
    ic = np.full(s.shape[0], np.nan)
    for t in range(s.shape[0]):
        ok = np.isfinite(s[t]) & np.isfinite(r[t])
        if ok.sum() < min_obs:
            continue
        u, v = s[t, ok], r[t, ok]
        if kind == "spearman":
            u, v = stats.rankdata(u), stats.rankdata(v)
        u, v = u - u.mean(), v - v.mean()
        den = math.sqrt(float(u @ u) * float(v @ v))
        if den > 0.0:
            ic[t] = max(-1.0, min(1.0, float(u @ v) / den))
    valid = ic[np.isfinite(ic)]
    n = int(valid.size)
    mean = float(valid.mean()) if n else math.nan
    std = float(valid.std(ddof=ddof)) if n > ddof else math.nan
    ir = mean / std if std and math.isfinite(std) else math.nan
    t_stat = ir * math.sqrt(n) if math.isfinite(ir) else math.nan
    out_ic = pd.Series(ic, index=signal.index) if isinstance(signal, pd.DataFrame) else (
        float(ic[0]) if np.ndim(signal) == 1 else ic)
    return {"ic": out_ic, "mean": mean, "std": std, "ir": ir, "t_stat": t_stat, "n": n}
