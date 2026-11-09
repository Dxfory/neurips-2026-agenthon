"""Interest-rate curve building blocks: day counts, compounding, zero curves, bonds, Nelson-Siegel.

Every market convention is an explicit, required keyword argument: day-count
``basis``, compounding ``comp``, coupon ``freq``, curve ``interp``, NS
``lam_form``. Nothing silently picks a convention for you.

Units: rates are decimals (0.05 = 5 %), times are in years, prices per ``face``.

Compounding values accepted wherever ``comp`` appears:
``'continuous'`` | ``'simple'`` | positive int m (compounded m times a year),
with aliases ``'annual'``=1, ``'semiannual'``=2, ``'quarterly'``=4, ``'monthly'``=12.

Vectorised functions return a Python ``float`` when every input is a scalar and
an ``np.ndarray`` (broadcast shape) otherwise.

Functions
---------
year_fraction          day-count year fraction between dates
df_from_rate           discount factor from a zero rate
rate_from_df           zero rate from a discount factor
convert_rate           convert a rate between compounding conventions
forward_rate           forward rate implied by two discount factors
ZeroCurve              interpolated zero curve (df, zero, forward, par_yield)
bootstrap_zero_curve   exact bootstrap from par rates or coupon-bond prices
par_yield              par coupon rate from a ZeroCurve
bond_cashflows         coupon times and amounts of a bullet bond
bond_price             full price from a yield to maturity
bond_price_curve       full price by discounting on a ZeroCurve
bond_ytm               yield to maturity from a full price
bond_risk              price, Macaulay/modified duration, convexity, DV01
ns_yield               Nelson-Siegel / Svensson yield curve
fit_ns                 least-squares Nelson-Siegel / Svensson fit
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import optimize

__all__ = [
    "year_fraction",
    "df_from_rate",
    "rate_from_df",
    "convert_rate",
    "forward_rate",
    "ZeroCurve",
    "bootstrap_zero_curve",
    "par_yield",
    "bond_cashflows",
    "bond_price",
    "bond_price_curve",
    "bond_ytm",
    "bond_risk",
    "ns_yield",
    "fit_ns",
]

_EPS = np.finfo(float).eps


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _ret(x):
    """float for 0-d results, ndarray otherwise."""
    x = np.asarray(x, dtype=float)
    return float(x) if x.ndim == 0 else x


_COMP_ALIASES = {
    "continuous": "continuous",
    "continuously": "continuous",
    "cont": "continuous",
    "simple": "simple",
    "annual": 1,
    "annually": 1,
    "semiannual": 2,
    "semiannually": 2,
    "quarterly": 4,
    "monthly": 12,
}


def _parse_comp(comp, *, allow_simple=True, allow_continuous=True, name="comp"):
    """Return 'continuous', 'simple' or a positive int m."""
    if isinstance(comp, str):
        key = comp.strip().lower().replace("-", "").replace("_", "").replace(" ", "")
        if key not in _COMP_ALIASES:
            raise ValueError(
                f"{name}={comp!r} not understood; use 'continuous', 'simple', a positive int m, "
                "or 'annual'/'semiannual'/'quarterly'/'monthly'"
            )
        c = _COMP_ALIASES[key]
    elif isinstance(comp, (bool, np.bool_)) or comp is None:
        raise ValueError(f"{name}={comp!r} not understood")
    elif isinstance(comp, (int, np.integer)):
        c = int(comp)
    elif isinstance(comp, (float, np.floating)) and float(comp).is_integer():
        c = int(comp)
    else:
        raise ValueError(f"{name}={comp!r} not understood; compounding frequency must be a positive integer")
    if isinstance(c, int) and c <= 0:
        raise ValueError(f"{name} must be a positive integer, got {comp!r}")
    if c == "simple" and not allow_simple:
        raise ValueError(f"{name}='simple' is not allowed here")
    if c == "continuous" and not allow_continuous:
        raise ValueError(f"{name}='continuous' is not allowed here")
    return c


def _parse_freq(freq):
    return _parse_comp(freq, allow_simple=False, allow_continuous=False, name="freq")


# ----------------------------------------------------------------------------
# day counts
# ----------------------------------------------------------------------------
# Keys are normalised by _basis_key (lower case; spaces, '_', '-', '(', ')' removed).
_BASES = {
    "act/365f": "act/365f",
    "act/365fixed": "act/365f",
    "actual/365f": "act/365f",
    "actual/365fixed": "act/365f",
    "a/365f": "act/365f",
    "act/360": "act/360",
    "actual/360": "act/360",
    "a/360": "act/360",
    "act/actisda": "act/act_isda",
    "actual/actualisda": "act/act_isda",
    "30/360us": "30/360_us",
    "30/360bond": "30/360_us",
    "30/360bondbasis": "30/360_us",
    "bondbasis": "30/360_us",
    "30e/360": "30e/360",
    "30/360eurobond": "30e/360",
    "30/360eurobondbasis": "30e/360",
    "eurobondbasis": "30e/360",
}


def _basis_key(basis):
    k = str(basis).strip().lower()
    for ch in " _-()":
        k = k.replace(ch, "")
    return k


def _to_d64(x):
    """Convert date-like input to datetime64[D] (wall-clock date; time of day and tz dropped).

    Returns (array, is_scalar). None / NaN / NaT give NaT; bare numbers are rejected
    (pandas would silently read them as nanoseconds since 1970).
    """
    if x is None or x is pd.NaT:
        return np.array("NaT", dtype="datetime64[D]"), True
    if isinstance(x, (str, _dt.date, np.datetime64)):  # datetime and pd.Timestamp are date subclasses
        ts = pd.Timestamp(x)
        if ts is pd.NaT:
            return np.array("NaT", dtype="datetime64[D]"), True
        if ts.tz is not None:  # keep the local calendar date, as the array path does
            ts = ts.tz_localize(None)
        return np.array(ts.to_datetime64()).astype("datetime64[D]"), True
    arr = np.asarray(x)
    if arr.dtype.kind == "M":
        return arr.astype("datetime64[D]"), arr.ndim == 0
    if arr.dtype.kind in "biufc":
        if arr.dtype.kind == "f" and np.all(np.isnan(arr)):
            return np.full(arr.shape, np.datetime64("NaT", "D")), arr.ndim == 0
        raise TypeError("dates must be date-like (date, datetime64, Timestamp, ISO string), not numbers")
    if arr.ndim == 0:
        item = arr.item()
        if item is None or isinstance(item, (str, _dt.date, np.datetime64, float, int)):
            return _to_d64(item)
        raise TypeError(f"cannot interpret {item!r} as a date")
    conv = pd.to_datetime(arr.ravel())
    if getattr(conv, "tz", None) is not None:
        conv = conv.tz_localize(None)
    return np.asarray(conv.values).astype("datetime64[D]").reshape(arr.shape), False


def _days(a, b):
    """Integer day count b - a."""
    return (b - a).astype("timedelta64[D]").astype(np.int64)


def _ymd(d):
    y = d.astype("datetime64[Y]").astype(np.int64) + 1970
    m = d.astype("datetime64[M]").astype(np.int64) % 12 + 1
    dd = _days(d.astype("datetime64[M]").astype("datetime64[D]"), d) + 1
    return y, m, dd


def _jan1(y):
    return (np.asarray(y, dtype=np.int64) - 1970).astype("datetime64[Y]").astype("datetime64[D]")


def _isleap(y):
    return (y % 4 == 0) & ((y % 100 != 0) | (y % 400 == 0))


def year_fraction(start, end, *, basis):
    """Year fraction between two dates (or arrays of dates) under a named day-count basis.

    Parameters
    ----------
    start, end : date-like or array-like of date-likes
        ``datetime.date``/``datetime``, ``np.datetime64``, ``pd.Timestamp``, ISO
        string, or lists/arrays/Series of these (broadcast together). Times of
        day are dropped (tz-aware values keep their local calendar date).
        None/NaN/NaT give nan; bare numbers raise TypeError. ``end < start``
        gives the negative of the swapped fraction (the 30/360 rules are
        applied to the ordered pair).
    basis : str, required (case, spaces, '_', '-', brackets ignored)
        ``'act/365f'``  actual days / 365 (alias 'act/365 fixed', 'A/365F').
        ``'act/360'``   actual days / 360.
        ``'act/act_isda'`` days falling in leap years / 366 + days in non-leap
        years / 365 (period split at 1 January). Not Act/Act ICMA (coupon-period
        based, needs the coupon schedule).
        ``'30/360_us'`` (alias '30/360_bond', 'bond basis') ISDA 2006 4.16(f)
        "30/360, Bond Basis": D1=31 -> 30; D2=31 -> 30 only if D1 >= 30.
        No end-of-February rule: the SIFMA / QuantLib-USA / Excel DAYS360
        variants that also set D1 = 30 when D1 is the last day of February are
        NOT applied; adjust D1 yourself if a task states that rule.
        ``'30e/360'``   (alias 'eurobond basis') ISDA 2006 4.16(g) "30E/360":
        D1, D2 = min(D, 30). Not 30E/360 (ISDA) 4.16(h), which also maps
        the last day of February to 30.
        Plain '30/360', 'act/365' and 'act/act' are rejected as ambiguous.

    Returns
    -------
    float or ndarray (nan where a date is NaT).

    Formula: 30/360 bases use (360(Y2-Y1) + 30(M2-M1) + (D2-D1)) / 360 after the
    day adjustments above.

    Example
    -------
    >>> year_fraction('2003-11-01', '2004-05-01', basis='act/act_isda')  # 61/365 + 121/366
    0.4977243805674...
    """
    key = _basis_key(basis)
    if key not in _BASES:
        raise ValueError(f"basis={basis!r} not understood; use one of {sorted(set(_BASES.values()))}")
    b = _BASES[key]
    s, s_scalar = _to_d64(start)
    e, e_scalar = _to_d64(end)
    s, e = np.broadcast_arrays(s, e)
    nat = np.isnat(s) | np.isnat(e)
    if np.any(nat):  # compute on a placeholder date (no int64 overflow warnings), nan-ed at the end
        fill = np.datetime64("2000-01-01", "D")
        s = np.where(nat, fill, s)
        e = np.where(nat, fill, e)
    neg = e < s
    lo = np.where(neg, e, s)
    hi = np.where(neg, s, e)
    if b == "act/365f":
        frac = _days(lo, hi) / 365.0
    elif b == "act/360":
        frac = _days(lo, hi) / 360.0
    elif b == "act/act_isda":
        y1 = lo.astype("datetime64[Y]").astype(np.int64) + 1970
        y2 = hi.astype("datetime64[Y]").astype(np.int64) + 1970
        diy1 = np.where(_isleap(y1), 366.0, 365.0)
        diy2 = np.where(_isleap(y2), 366.0, 365.0)
        same = _days(lo, hi) / diy1
        split = _days(lo, _jan1(y1 + 1)) / diy1 + (y2 - y1 - 1) + _days(_jan1(y2), hi) / diy2
        frac = np.where(y1 == y2, same, split)
    else:
        y1, m1, d1 = _ymd(lo)
        y2, m2, d2 = _ymd(hi)
        if b == "30/360_us":
            d1a = np.where(d1 == 31, 30, d1)
            d2a = np.where((d2 == 31) & (d1a >= 30), 30, d2)
        else:
            d1a = np.minimum(d1, 30)
            d2a = np.minimum(d2, 30)
        frac = (360 * (y2 - y1) + 30 * (m2 - m1) + (d2a - d1a)) / 360.0
    out = np.where(neg, -frac, frac).astype(float)
    out = np.where(nat, np.nan, out)
    if s_scalar and e_scalar:
        return float(out)
    return out


# ----------------------------------------------------------------------------
# compounding
# ----------------------------------------------------------------------------
def df_from_rate(rate, t, *, comp):
    """Discount factor for a zero rate over t years under a stated compounding convention.

    Parameters
    ----------
    rate : float or array, decimal (0.05 = 5 %).
    t : float or array, years (>= 0). Broadcasts with ``rate``.
    comp : required. 'continuous' | 'simple' | positive int m | alias
        ('annual'=1, 'semiannual'=2, 'quarterly'=4, 'monthly'=12).

    Returns
    -------
    float or ndarray. Exactly 1 at t=0; nan where the rate is outside the
    domain (1 + r t <= 0 for simple, r <= -m for m-compounding).

    Formula: continuous exp(-r t); simple 1/(1 + r t); m (1 + r/m)^(-m t).

    Example
    -------
    >>> df_from_rate(0.10, 1.0, comp=2)   # 1/1.05**2
    0.90702947845805...
    """
    c = _parse_comp(comp)
    r = np.asarray(rate, dtype=float)
    tt = np.asarray(t, dtype=float)
    with np.errstate(all="ignore"):
        if c == "continuous":
            out = np.exp(-r * tt)
        elif c == "simple":
            base = 1.0 + r * tt
            out = np.where(base > 0, 1.0 / base, np.nan)
        else:
            x = r / c
            out = np.where(x > -1.0, np.exp(-c * tt * np.log1p(x)), np.nan)
        out = np.where(tt == 0, 1.0, out)
    return _ret(out)


def rate_from_df(df, t, *, comp):
    """Zero rate implied by a discount factor over t years under a stated compounding convention.

    Parameters
    ----------
    df : float or array, discount factor (> 0).
    t : float or array, years; must be > 0 (t <= 0 or df <= 0 give nan).
    comp : required, same values as ``df_from_rate``.

    Returns
    -------
    float or ndarray, decimal rate.

    Formula: continuous -ln(df)/t; simple (1/df - 1)/t; m m(df^(-1/(m t)) - 1).

    Example
    -------
    >>> rate_from_df(0.975, 0.25, comp='continuous')   # Hull 3-month T-bill, 10.127 %
    0.10127123193715...
    """
    c = _parse_comp(comp)
    d = np.asarray(df, dtype=float)
    tt = np.asarray(t, dtype=float)
    with np.errstate(all="ignore"):
        if c == "continuous":
            out = -np.log(d) / tt
        elif c == "simple":
            out = (1.0 - d) / (d * tt)
        else:
            out = c * np.expm1(-np.log(d) / (c * tt))
        out = np.where((tt > 0) & (d > 0), out, np.nan)
    return _ret(out)


def convert_rate(rate, *, from_comp, to_comp, t=None):
    """Convert a rate between compounding conventions so that it gives the same discount factor.

    Parameters
    ----------
    rate : float or array, decimal.
    from_comp, to_comp : required compounding values (see ``df_from_rate``).
    t : float or array, years. Required only when either side is 'simple'
        (simple rates depend on the horizon); ignored otherwise.

    Returns
    -------
    float or ndarray.

    Formula: rate_from_df(df_from_rate(rate, t, from_comp), t, to_comp); without a
    simple leg the closed forms m->cont m ln(1 + R/m), cont->m m(e^{R/m} - 1)
    are used (horizon-free).

    Example
    -------
    >>> convert_rate(0.10, from_comp=2, to_comp='continuous')   # Hull: 9.758 %
    0.09758032833886...
    """
    f = _parse_comp(from_comp, name="from_comp")
    g = _parse_comp(to_comp, name="to_comp")
    r = np.asarray(rate, dtype=float)
    if f == g:
        return _ret(r.copy())
    if f == "simple" or g == "simple":
        if t is None:
            raise ValueError("t (years) is required when converting to or from 'simple'")
        return _ret(rate_from_df(df_from_rate(r, t, comp=f), t, comp=g))
    with np.errstate(all="ignore"):
        if f == "continuous":
            cont = r
        else:
            cont = np.where(r / f > -1.0, f * np.log1p(r / f), np.nan)
        out = cont if g == "continuous" else g * np.expm1(cont / g)
    return _ret(out)


def forward_rate(t1, t2, df1, df2, *, comp):
    """Forward rate for the period [t1, t2] implied by the discount factors at t1 and t2.

    Parameters
    ----------
    t1, t2 : float or array, years, t2 > t1 >= 0.
    df1, df2 : float or array, discount factors to t1 and t2.
    comp : required compounding of the forward. 'simple' gives the simply
        compounded (LIBOR-style) forward used in Black-76 caplet pricing.

    Returns
    -------
    float or ndarray.

    Formula: F = rate_from_df(df2/df1, t2 - t1, comp); continuous
    F = (z2 t2 - z1 t1)/(t2 - t1); simple F = (df1/df2 - 1)/(t2 - t1).

    Example
    -------
    >>> import numpy as np
    >>> forward_rate(1, 2, np.exp(-0.03), np.exp(-0.08), comp='continuous')  # Hull Table 4.5: 5 %
    0.04999999999999...
    """
    c = _parse_comp(comp)
    a = np.asarray(t1, dtype=float)
    b = np.asarray(t2, dtype=float)
    if np.any(a < 0) or np.any(b <= a):
        raise ValueError("forward_rate requires t2 > t1 >= 0")
    with np.errstate(all="ignore"):
        ratio = np.asarray(df2, dtype=float) / np.asarray(df1, dtype=float)
    return rate_from_df(ratio, b - a, comp=c)


# ----------------------------------------------------------------------------
# zero curve
# ----------------------------------------------------------------------------
_INTERPS = ("linear_zero", "log_df")
_EXTRAPS = ("flat", "linear", "raise")


@dataclass(frozen=True, eq=False, init=False)
class ZeroCurve:
    """Zero curve on knot times with a selectable interpolation, giving discount factors, zero rates, forwards and par yields.

    Constructor: ``ZeroCurve(t, zero, *, comp, interp, extrapolate='flat')``
    (frozen: attributes cannot be reassigned).

    Parameters
    ----------
    t : array, knot times in years, strictly increasing and > 0 (DF(0) = 1 is implicit).
    zero : array, knot zero rates (decimal), compounded as ``comp``. Stored as
        the attribute ``zeros`` (``zero`` is the method below).
    comp : required, compounding of the knot rates ('continuous' | 'simple' | m).
    interp : required.
        'linear_zero': zero rate (in ``comp``) linear in t between knots.
        'log_df': ln DF linear in t between knots (piecewise-flat continuous forwards).
    extrapolate : 'flat' (default; hold the first/last zero rate, the common
        choice) | 'linear' (extend the end segment: zero line for linear_zero,
        last forward for log_df) | 'raise'. For log_df, 'linear' before the first
        knot uses the segment from DF(0)=1 (constant forward = continuous z1),
        because extending the first ln DF segment would break DF(0)=1. A
        one-knot curve extrapolates flat.

    Methods
    -------
    df(t)                         discount factors (float or array)
    zero(t, *, comp=None)         zero rates, default in the curve's comp; t=0 -> nan
    forward(t1, t2, *, comp)      forward rate over [t1, t2]
    par_yield(T, *, freq)         par coupon rate (see ``par_yield``)
    knots()                       dict(t, zero, df, comp, interp, extrapolate)

    Attributes ``t``, ``zeros`` (read-only arrays), ``comp``, ``interp``,
    ``extrapolate``; ``repricing_error`` is None unless set by
    ``bootstrap_zero_curve``. Knot inputs are returned exactly at knot times.

    Example
    -------
    >>> c = ZeroCurve([1, 2, 5, 10], [0.02, 0.025, 0.035, 0.04], comp='continuous', interp='log_df')
    >>> c.df(3.0), c.zero(3.0), c.forward(2, 5, comp='continuous')
    """

    t: np.ndarray
    zeros: np.ndarray
    comp: Any
    interp: str
    extrapolate: str = "flat"
    repricing_error: Any = field(default=None, repr=False)

    def __init__(self, t, zero, *, comp, interp, extrapolate="flat"):
        t = np.array(t, dtype=float, ndmin=1).ravel()
        z = np.array(zero, dtype=float, ndmin=1).ravel()
        if t.size == 0 or t.size != z.size:
            raise ValueError("t and zero must be non-empty and of equal length")
        if not (np.all(np.isfinite(t)) and np.all(np.isfinite(z))):
            raise ValueError("knot times and zero rates must be finite")
        if t[0] <= 0 or np.any(np.diff(t) <= 0):
            raise ValueError("knot times must be > 0 and strictly increasing")
        c = _parse_comp(comp)
        interp_ = str(interp).strip().lower()
        if interp_ not in _INTERPS:
            raise ValueError(f"interp={interp!r}; use one of {_INTERPS}")
        extrap = str(extrapolate).strip().lower()
        if extrap not in _EXTRAPS:
            raise ValueError(f"extrapolate={extrapolate!r}; use one of {_EXTRAPS}")
        d = np.atleast_1d(np.asarray(df_from_rate(z, t, comp=c), dtype=float))
        if not np.all(np.isfinite(d) & (d > 0)):
            raise ValueError("knot zero rates give non-positive or non-finite discount factors")
        lnd = np.log(d)
        for a in (t, z, d, lnd):
            a.setflags(write=False)
        object.__setattr__(self, "t", t)
        object.__setattr__(self, "zeros", z)
        object.__setattr__(self, "comp", comp)
        object.__setattr__(self, "interp", interp_)
        object.__setattr__(self, "extrapolate", extrap)
        object.__setattr__(self, "repricing_error", None)
        object.__setattr__(self, "_c", c)
        object.__setattr__(self, "_df", d)
        object.__setattr__(self, "_lnd", lnd)

    # -- internals -----------------------------------------------------------
    @staticmethod
    def _as_times(t):
        tt = np.asarray(t, dtype=float)
        scalar = tt.ndim == 0
        tt = np.atleast_1d(tt)
        if not np.all(np.isfinite(tt)) or np.any(tt < 0):
            raise ValueError("times must be finite and >= 0")
        return tt, scalar

    def _check_range(self, tt):
        if self.extrapolate == "raise":
            T = self.t
            bad = ((tt > 0) & (tt < T[0])) | (tt > T[-1])
            if np.any(bad):
                raise ValueError(
                    f"time(s) {tt[bad][:5]} outside knot range [{T[0]}, {T[-1]}] with extrapolate='raise'"
                )

    def _lz_zero(self, tt):
        """linear_zero: zero rate in the curve's comp at times tt (any tt >= 0)."""
        T, Z = self.t, self.zeros
        z = np.interp(tt, T, Z)  # flat outside the knot range
        if self.extrapolate == "linear" and T.size >= 2:
            lo = tt < T[0]
            hi = tt > T[-1]
            z = np.where(lo, Z[0] + (tt - T[0]) * (Z[1] - Z[0]) / (T[1] - T[0]), z)
            z = np.where(hi, Z[-1] + (tt - T[-1]) * (Z[-1] - Z[-2]) / (T[-1] - T[-2]), z)
        return z

    def _df_core(self, tt):
        T, Z, D, Y = self.t, self.zeros, self._df, self._lnd
        n = T.size
        self._check_range(tt)
        if self.interp == "linear_zero":
            out = np.atleast_1d(np.asarray(df_from_rate(self._lz_zero(tt), tt, comp=self._c), dtype=float))
            out = out.copy()
        else:
            out = np.empty(tt.shape)
            lo = tt < T[0]
            hi = tt > T[-1]
            mid = ~lo & ~hi
            tm = tt[mid]
            if n == 1:
                out[mid] = D[0]
            else:
                j = np.clip(np.searchsorted(T, tm, side="right") - 1, 0, n - 2)
                w = (tm - T[j]) / (T[j + 1] - T[j])
                out[mid] = D[j] * np.exp(w * (Y[j + 1] - Y[j]))
            ts = tt[lo]
            if self.extrapolate == "linear":
                out[lo] = np.exp(ts / T[0] * Y[0])
            else:
                out[lo] = df_from_rate(Z[0], ts, comp=self._c)
            tl = tt[hi]
            if self.extrapolate == "linear" and n >= 2:
                out[hi] = D[-1] * np.exp((tl - T[-1]) * (Y[-1] - Y[-2]) / (T[-1] - T[-2]))
            else:
                out[hi] = df_from_rate(Z[-1], tl, comp=self._c)
        k = np.clip(np.searchsorted(T, tt), 0, n - 1)
        hit = T[k] == tt
        out[hit] = D[k[hit]]
        out[tt == 0] = 1.0
        return out

    # -- public ----------------------------------------------------------------
    def df(self, t):
        """Discount factor(s) at time(s) t (years, >= 0); float for scalar t."""
        tt, scalar = self._as_times(t)
        out = self._df_core(tt)
        return float(out[0]) if scalar else out

    def zero(self, t, *, comp=None):
        """Zero rate(s) at t (years, > 0) in ``comp`` (default: the curve's comp); t=0 gives nan."""
        target = self._c if comp is None else _parse_comp(comp)
        tt, scalar = self._as_times(t)
        if target == self._c and self.interp == "linear_zero":
            self._check_range(tt)
            out = np.where(tt > 0, self._lz_zero(tt), np.nan)
        else:
            out = np.atleast_1d(np.asarray(rate_from_df(self._df_core(tt), tt, comp=target), dtype=float)).copy()
            if target == self._c:
                T = self.t
                k = np.clip(np.searchsorted(T, tt), 0, T.size - 1)
                hit = T[k] == tt
                out[hit] = self.zeros[k[hit]]
        return float(out[0]) if scalar else out

    def forward(self, t1, t2, *, comp):
        """Forward rate over [t1, t2] (years, t2 > t1 >= 0) compounded as ``comp`` (required)."""
        return forward_rate(t1, t2, self.df(t1), self.df(t2), comp=comp)

    def par_yield(self, T, *, freq):
        """Par coupon rate for maturity T paying ``freq`` coupons a year (see module ``par_yield``)."""
        return par_yield(self, T, freq=freq)

    def knots(self):
        """Knot data: dict(t, zero, df, comp, interp, extrapolate) (arrays are copies)."""
        return {
            "t": self.t.copy(),
            "zero": self.zeros.copy(),
            "df": self._df.copy(),
            "comp": self.comp,
            "interp": self.interp,
            "extrapolate": self.extrapolate,
        }


# ----------------------------------------------------------------------------
# bonds
# ----------------------------------------------------------------------------
def _coupon_times(T, freq):
    """Coupon times T - j/freq > 0 (ascending); a date within ~1e-9 periods of 0 counts as 0."""
    T = float(T)
    if not (np.isfinite(T) and T > 0):
        raise ValueError(f"maturity T must be > 0, got {T}")
    n = int(np.ceil(T * freq - 1e-9))
    n = max(n, 1)
    j = np.arange(n - 1, -1, -1, dtype=float)
    return T - j / freq


def bond_cashflows(T, coupon_rate, *, freq, face=100.0):
    """Coupon payment times and cash amounts of a fixed-coupon bullet bond.

    Parameters
    ----------
    T : float, maturity in years (> 0).
    coupon_rate : float, annual coupon rate (decimal), paid ``freq`` times a year.
    freq : required, coupons per year (positive int or 'annual'/'semiannual'/...).
    face : float, redemption amount (default 100).

    Returns
    -------
    (times, amounts) : ndarrays. Times T - j/freq > 0 counted back from
    maturity, ascending (a short first period is allowed; a coupon at t=0 is
    excluded). Each coupon is face*coupon_rate/freq (full coupon, no accrual
    adjustment); the last amount includes face.

    Accrued interest (for clean = full - accrued), linear in time on this grid:
    AI = face*coupon_rate/freq * (1 - freq*times[0]). Use a day-count ratio
    instead when the task defines accrual in days.

    Example
    -------
    >>> bond_cashflows(3, 0.10, freq=2)   # Hull Table 4.6
    (array([0.5, 1. , 1.5, 2. , 2.5, 3. ]), array([  5.,   5.,   5.,   5.,   5., 105.]))
    """
    m = _parse_freq(freq)
    times = _coupon_times(T, m)
    amounts = np.full(times.shape, float(face) * float(coupon_rate) / m)
    amounts[-1] += float(face)
    return times, amounts


_SHORT_ENDS = ("simple", "compounded", "full_coupon")


def _short_end_rule(short_end):
    """The validated short_end rule, one of _SHORT_ENDS (case-insensitive); None is the default 'full_coupon'."""
    if short_end is None:
        return "full_coupon"
    rule = str(short_end).strip().lower()
    if rule not in _SHORT_ENDS:
        raise ValueError(f"short_end must be 'full_coupon' (default), 'compounded' or 'simple'; got {short_end!r}")
    return rule


def _par_instrument_flows(T, coupon_rate, freq, face, short_end):
    """Cash flows of a par instrument: bond_cashflows, except that a coupon instrument maturing within one coupon
    period (T < 1/freq) is a single payment at T under the short_end rule ('compounded': face (1 + c/freq)^(freq T);
    'simple': face (1 + c T); 'full_coupon': face (1 + c/freq), bond_cashflows')."""
    T, c = float(T), float(coupon_rate)
    if c == 0 or not T < 1.0 / freq - 1e-12:
        return bond_cashflows(T, c, freq=freq, face=face)
    if short_end == "simple":
        amount = float(face) * (1.0 + c * T)
    elif short_end == "compounded":
        amount = float(face) * (1.0 + c / freq) ** (freq * T)
    else:
        amount = float(face) * (1.0 + c / freq)
    return np.array([T]), np.array([amount])


def bond_price(T, coupon_rate, ytm, *, freq, comp, face=100.0):
    """Full (dirty) bond price from a yield to maturity.

    Parameters
    ----------
    T : float, maturity in years (arrays are handled element-wise).
    coupon_rate : float or array, annual coupon rate (decimal).
    ytm : float or array, yield to maturity (decimal), compounded as ``comp``.
    freq : required, coupons per year.
    comp : required yield compounding: 'continuous' or m. The bond-market
        convention is comp=freq. 'simple' is rejected.
    face : float, default 100.

    Returns
    -------
    float or ndarray, the full price (clean = full - accrued, caller's job;
    see ``bond_cashflows``).

    Formula: P = sum_j CF_j * df_from_rate(ytm, t_j, comp), cash flows from
    ``bond_cashflows``; for comp=m this is CF_j (1 + y/m)^(-m t_j) with a
    fractional exponent in a short first period ("street" convention; the
    Treasury convention of simple interest over the first period is not used).

    Example
    -------
    >>> bond_price(3, 0.10, 0.12, freq=2, comp='continuous')   # Hull Table 4.6: 94.213
    94.21302055...
    """
    m = _parse_freq(freq)
    c = _parse_comp(comp, allow_simple=False)
    if np.ndim(T) > 0:
        f = np.vectorize(lambda TT, cc, yy: bond_price(TT, cc, yy, freq=m, comp=c, face=face), otypes=[float])
        return _ret(f(T, coupon_rate, ytm))
    times = _coupon_times(T, m)
    y = np.asarray(ytm, dtype=float)[..., None]
    dfs = np.asarray(df_from_rate(y, times, comp=c), dtype=float)
    coupon = float(face) * np.asarray(coupon_rate, dtype=float) / m
    return _ret(coupon * dfs.sum(axis=-1) + float(face) * dfs[..., -1])


def bond_price_curve(T, coupon_rate, curve, *, freq, face=100.0):
    """Full bond price by discounting each cash flow on a ZeroCurve.

    Parameters
    ----------
    T : float, maturity in years.
    coupon_rate : float (or array), annual coupon rate (decimal).
    curve : ZeroCurve (its interp/extrapolate rules apply).
    freq : required, coupons per year.
    face : float, default 100.

    Returns
    -------
    float (ndarray if coupon_rate is an array).

    Formula: P = sum_j CF_j * curve.df(t_j).

    Example
    -------
    >>> zc = ZeroCurve([0.5, 1, 1.5, 2], [0.05, 0.058, 0.064, 0.068], comp='continuous', interp='log_df')
    >>> bond_price_curve(2, 0.06, zc, freq=2)   # Hull Table 4.2: 98.39
    98.38506277...
    """
    m = _parse_freq(freq)
    times = _coupon_times(T, m)
    d = np.atleast_1d(np.asarray(curve.df(times), dtype=float))
    coupon = float(face) * np.asarray(coupon_rate, dtype=float) / m
    return _ret(coupon * d.sum() + float(face) * d[-1])


def par_yield(curve, T, *, freq):
    """Par coupon rate for maturity T implied by a ZeroCurve.

    Parameters
    ----------
    curve : ZeroCurve.
    T : float or array, maturity in years; coupon dates counted back from T.
    freq : required, coupons per year. The result is an annual coupon rate
        paid ``freq`` times a year (full coupons, as ``bond_cashflows``).

    Returns
    -------
    float or ndarray.

    Formula: c = freq (1 - DF(T)) / sum_j DF(t_j).

    Example
    -------
    >>> zc = ZeroCurve([0.5, 1, 1.5, 2], [0.05, 0.058, 0.064, 0.068], comp='continuous', interp='log_df')
    >>> par_yield(zc, 2, freq=2)   # Hull: 6.87 %
    0.06872876...
    """
    m = _parse_freq(freq)
    Ts = np.asarray(T, dtype=float)
    out = np.empty(Ts.shape)
    for idx, Ti in np.ndenumerate(Ts):
        d = np.atleast_1d(np.asarray(curve.df(_coupon_times(Ti, m)), dtype=float))
        out[idx] = m * (1.0 - d[-1]) / d.sum()
    return _ret(out)


def bond_ytm(price, T, coupon_rate, *, freq, comp, face=100.0):
    """Yield to maturity that reprices a bond to a given full price.

    Parameters
    ----------
    price : float, full (dirty) price per ``face`` (> 0). Arrays are solved element-wise.
    T : float, maturity in years.
    coupon_rate : float, annual coupon rate (decimal).
    freq : required, coupons per year.
    comp : required yield compounding, 'continuous' or m (not 'simple').
    face : float, default 100.

    Returns
    -------
    float (ndarray for array inputs).

    Method: brentq on bond_price(y) - price, bracket starting at [-0.5, 1]
    and expanded as needed (kept inside y > -m for comp m), xtol 1e-14.
    A nan price gives nan; a non-positive price raises ValueError.

    Example
    -------
    >>> bond_ytm(98.3850627729, 2, 0.06, freq=2, comp='continuous')   # Hull: 6.76 %
    0.06762438...
    """
    m = _parse_freq(freq)
    c = _parse_comp(comp, allow_simple=False)
    if np.ndim(price) > 0 or np.ndim(T) > 0 or np.ndim(coupon_rate) > 0:
        f = np.vectorize(lambda p, TT, cc: bond_ytm(p, TT, cc, freq=m, comp=c, face=face), otypes=[float])
        return _ret(f(price, T, coupon_rate))
    p = float(price)
    if np.isnan(p):
        return float("nan")
    if not (np.isfinite(p) and p > 0):
        raise ValueError("price must be finite and > 0")
    times, amounts = bond_cashflows(T, coupon_rate, freq=m, face=face)

    def g(y):
        with np.errstate(all="ignore"):
            return float(amounts @ np.asarray(df_from_rate(y, times, comp=c))) - p

    y_floor = -np.inf if c == "continuous" else -float(c)
    lo, hi = -0.5, 1.0
    glo, ghi = g(lo), g(hi)
    for _ in range(200):
        if ghi <= 0:
            break
        hi = 2.0 * hi + 1.0
        ghi = g(hi)
    for _ in range(200):
        if glo >= 0:
            break
        lo = lo - 1.0 if c == "continuous" else y_floor + (lo - y_floor) / 4.0
        glo = g(lo)
    if not (glo >= 0 >= ghi):
        raise ValueError("could not bracket the yield; check price and cash flows")
    return float(optimize.brentq(g, lo, hi, xtol=1e-14, rtol=4 * _EPS, maxiter=500))


def bond_risk(T, coupon_rate, ytm, *, freq, comp, face=100.0):
    """Price, Macaulay duration, modified duration, convexity and DV01 of a bond at a given yield.

    Parameters
    ----------
    T : float, maturity in years.
    coupon_rate : float, annual coupon rate (decimal).
    ytm : float, yield (decimal) compounded as ``comp``.
    freq : required, coupons per year.
    comp : required, 'continuous' or m (not 'simple').
    face : float, default 100.

    Returns
    -------
    dict with
        price      full price
        macaulay   sum t_j CF_j DF_j / P (years)
        modified   -(1/P) dP/dy = macaulay (continuous) or macaulay/(1 + y/m)
        convexity  (1/P) d2P/dy2 in the given comp, in years^2: sum t^2 CF DF / P
                   (continuous) or sum CF t (t + 1/m)(1 + y/m)^(-m t - 2) / P.
                   Texts quoting convexity in periods^2 use m^2 times this.
        dv01       modified * P * 1e-4 = -dP/dy * 1e-4, the (first-order) price
                   drop for +1bp per ``face``; scale by notional/face yourself.

    Price change: dP/P ~ -modified*dy + 0.5*convexity*dy^2 (dy in the yield's comp).

    Example
    -------
    >>> bond_risk(3, 0.10, 0.12, freq=2, comp='continuous')['macaulay']   # Hull: 2.653
    2.65301003...
    """
    m = _parse_freq(freq)
    c = _parse_comp(comp, allow_simple=False)
    times, amounts = bond_cashflows(T, coupon_rate, freq=m, face=face)
    y = float(ytm)
    dfs = np.asarray(df_from_rate(y, times, comp=c), dtype=float)
    pv = amounts * dfs
    P = float(pv.sum())
    mac = float((times * pv).sum() / P)
    if c == "continuous":
        mod = mac
        conv = float((times * times * pv).sum() / P)
    else:
        g = 1.0 + y / c
        mod = mac / g
        conv = float((pv * times * (times + 1.0 / c)).sum() / (P * g * g))
    return {"price": P, "macaulay": mac, "modified": mod, "convexity": conv, "dv01": mod * P * 1e-4}


# ----------------------------------------------------------------------------
# bootstrap
# ----------------------------------------------------------------------------
def _bracket_decreasing(f, lo=-0.5, hi=1.0):
    """Bracket the root of a decreasing function f whose domain may be bounded below."""
    flo = f(lo)
    k = 0
    while not np.isfinite(flo) and k < 200:
        lo *= 0.5
        flo = f(lo)
        k += 1
    fhi = f(hi)
    k = 0
    while np.isfinite(fhi) and fhi > 0 and k < 200:
        hi = 2.0 * hi + 1.0
        fhi = f(hi)
        k += 1
    step = 0.5
    k = 0
    while np.isfinite(flo) and flo < 0 and k < 400:
        cand = lo - step
        fc = f(cand)
        if np.isfinite(fc):
            lo, flo = cand, fc
            step *= 2.0
        else:
            step /= 4.0
        k += 1
    if not (np.isfinite(flo) and np.isfinite(fhi) and flo >= 0 >= fhi):
        raise ValueError("could not bracket a zero rate that reprices the instrument")
    return lo, hi


def bootstrap_zero_curve(maturities, coupon_rates, *, freq, comp, interp, prices=None, face=100.0,
                         extrapolate="flat", short_end="full_coupon"):
    """Bootstrap an exact zero curve from par coupon rates (or coupon-bond prices), root-solving each new maturity so every input bond reprices exactly.

    A coupon instrument maturing within one coupon period (T < 1/freq, coupon != 0) is one
    payment at T. By default (short_end='full_coupon', the bond's own cash flows) it pays a full coupon; short_end='compounded' makes it a zero-coupon instrument at the
    quoted rate on the bond's compounding basis, face (1 + c/freq)^(freq T); 'simple' reads
    the rate as a money-market yield, face (1 + c T); 'full_coupon' pays a full coupon,
    face (1 + c/freq) (the bond_cashflows reading). Zero-coupon rows are unaffected; given
    prices stay the PV targets.

    Parameters
    ----------
    maturities : array, years (> 0, distinct); sorted internally. They become the knots.
    coupon_rates : array (or scalar), annual coupon rates (decimal); 0 gives a
        zero-coupon instrument.
    freq : required, coupons per year.
    comp : required, compounding of the output knot zeros.
    interp : required, 'linear_zero' | 'log_df' (used for coupon dates between
        quoted maturities).
    prices : array or None. Full prices per ``face``; None means par bonds
        (price = face). Accrued interest is not modelled.
    face : float, default 100.
    extrapolate : ZeroCurve extrapolation rule (default 'flat'); it sets the DF of
        coupon dates before the first maturity.
    short_end : 'full_coupon' (default), 'compounded' or 'simple'; the rule for a coupon
        instrument maturing within one coupon period (see above).

    Returns
    -------
    ZeroCurve with attribute ``repricing_error`` (model price - input price,
    in knot/maturity order).

    Method: for T_i (ascending) solve price_i = sum face c_i/freq DF(t_j) +
    face DF(T_i) for z_i with brentq (bracket from [-0.5, 1], expanded;
    xtol 1e-15, rtol 4 eps), using knots already solved plus (T_i, z_i).
    Coupon dates follow ``bond_cashflows`` (counted back from maturity). If the
    extrapolation makes an earlier bond depend on a later knot
    (extrapolate='linear' with linear_zero), Gauss-Seidel sweeps re-solve the
    knots until every bond reprices.

    Example
    -------
    >>> zc = bootstrap_zero_curve([0.25, 0.5, 1.0, 1.5, 2.0], [0, 0, 0, 0.08, 0.12], freq=2,
    ...                           comp='continuous', interp='linear_zero',
    ...                           prices=[97.5, 94.9, 90.0, 96.0, 101.6])   # Hull Table 4.3
    >>> zc.zero(2.0)   # 10.808 %
    """
    m = _parse_freq(freq)
    _parse_comp(comp)
    rule = _short_end_rule(short_end)
    Tm = np.array(maturities, dtype=float, ndmin=1).ravel()
    n = Tm.size
    cr = np.broadcast_to(np.asarray(coupon_rates, dtype=float), Tm.shape).astype(float)
    if prices is None:
        px = np.full(n, float(face))
    else:
        px = np.broadcast_to(np.asarray(prices, dtype=float), Tm.shape).astype(float)
    if n == 0 or not (np.all(np.isfinite(Tm)) and np.all(Tm > 0)):
        raise ValueError("maturities must be finite and > 0")
    order = np.argsort(Tm, kind="stable")
    Tm, cr, px = Tm[order], cr[order], px[order]
    if np.any(np.diff(Tm) <= 0):
        raise ValueError("maturities must be distinct")
    flows = []
    for T, c in zip(Tm, cr):  # zero-amount dates (zero-coupon instruments) are dropped
        tm, am = _par_instrument_flows(T, c, m, face, rule)
        keep = am != 0
        flows.append((tm[keep], am[keep]))
    extrap = str(extrapolate).strip().lower()
    if extrap == "raise" and any(tm[0] < Tm[0] for tm, _ in flows):
        raise ValueError("coupon dates fall before the first maturity; use extrapolate='flat' or 'linear'")

    def make(zk, k):
        return ZeroCurve(Tm[:k], zk[:k], comp=comp, interp=interp, extrapolate=extrapolate)

    def solve(i, z, k):
        times, amounts = flows[i]

        def f(zi):
            zz = z.copy()
            zz[i] = zi
            try:
                curve = make(zz, k)
            except ValueError:
                return np.nan
            return float(amounts @ np.atleast_1d(curve.df(times))) - px[i]

        a, b = _bracket_decreasing(f)
        return optimize.brentq(f, a, b, xtol=1e-15, rtol=4 * _EPS, maxiter=500)

    z = np.zeros(n)
    for i in range(n):
        z[i] = solve(i, z, i + 1)

    def errors(zz):
        curve = make(zz, n)
        return curve, np.array([float(a @ np.atleast_1d(curve.df(t))) for t, a in flows]) - px

    curve, err = errors(z)
    if np.max(np.abs(err)) > 1e-12 * float(face):
        for _ in range(200):
            z_old = z.copy()
            for i in range(n):
                z[i] = solve(i, z, n)
            if np.max(np.abs(z - z_old)) <= 1e-15:
                break
        curve, err = errors(z)
    object.__setattr__(curve, "repricing_error", err)
    return curve


# ----------------------------------------------------------------------------
# Nelson-Siegel / Svensson
# ----------------------------------------------------------------------------
def _parse_lam_form(lam_form):
    form = str(lam_form).strip().lower()
    if form not in ("rate", "time"):
        raise ValueError("lam_form must be 'rate' (x = lam*tau, Diebold-Li) or 'time' (x = tau/lam, NS 1987)")
    return form


def _ns_loadings(tau, lams, form, p):
    """Loading matrix (len(tau), p) for NS (p=3) or Svensson (p=4)."""
    tau = np.asarray(tau, dtype=float)

    def xs(lam):
        return lam * tau if form == "rate" else tau / lam

    def L(x):
        with np.errstate(all="ignore"):
            return np.where(x == 0, 1.0, -np.expm1(-x) / np.where(x == 0, 1.0, x))

    x1 = xs(lams[0])
    L1 = L(x1)
    cols = [np.ones_like(tau), L1, L1 - np.exp(-x1)]
    if p == 4:
        x2 = xs(lams[1])
        cols.append(L(x2) - np.exp(-x2))
    return np.stack(cols, axis=-1)


def _parse_lams(lams, nl):
    lam = np.atleast_1d(np.asarray(lams, dtype=float)).ravel()
    if lam.size != nl:
        raise ValueError(f"expected {nl} lambda value(s), got {lam.size}")
    if not np.all(np.isfinite(lam) & (lam > 0)):
        raise ValueError("lambda values must be finite and > 0")
    return lam


def ns_yield(tau, betas, lams, *, lam_form):
    """Nelson-Siegel (3 betas, 1 lambda) or Svensson (4 betas, 2 lambdas) yield curve at maturities tau.

    Parameters
    ----------
    tau : float or array, maturities in years (>= 0; tau=0 uses the limit).
    betas : (b0, b1, b2) for Nelson-Siegel or (b0, b1, b2, b3) for Svensson.
    lams : lambda (NS) or (lam1, lam2) (Svensson), > 0.
    lam_form : required. 'rate': x = lam*tau (Diebold-Li 2006, lam in 1/years).
        'time': x = tau/lam (Nelson-Siegel 1987 / Svensson 1994, lam in years).

    Returns
    -------
    ndarray shaped like tau (float for scalar tau), in the units of betas.

    Formula: L(x) = (1 - e^{-x})/x (L(0) = 1);
    y = b0 + b1 L(x1) + b2 [L(x1) - e^{-x1}] + b3 [L(x2) - e^{-x2}].
    Limits: y(0) = b0 + b1, y(inf) = b0.

    Example
    -------
    >>> ns_yield([1, 5, 10], [0.04, -0.02, 0.01], 0.6, lam_form='rate')
    """
    form = _parse_lam_form(lam_form)
    b = np.asarray(betas, dtype=float).ravel()
    if b.size not in (3, 4):
        raise ValueError("betas must have 3 (Nelson-Siegel) or 4 (Svensson) elements")
    lam = _parse_lams(lams, 1 if b.size == 3 else 2)
    tt = np.asarray(tau, dtype=float)
    if np.any(tt < 0):
        raise ValueError("tau must be >= 0")
    X = _ns_loadings(tt.ravel(), lam, form, b.size)
    return _ret((X @ b).reshape(tt.shape))


def fit_ns(tau, y, *, model, lam_form, lam=None, lam_bounds=None, weights=None):
    """Least-squares Nelson-Siegel or Svensson fit, with lambda fixed (given) or estimated.

    fit_ns runs its own profiled lambda grid plus polish and may return a different,
    lower-SSE optimum than the optimizer and start a TASK prescribes. When the TASK fixes
    x0, method and tolerances, use fit_parametric_curve(gradient='2-point').

    Parameters
    ----------
    tau : array, maturities in years (>= 0).
    y : array, observed yields (same length).
    model : required, 'ns' (3 betas, 1 lambda) | 'svensson' (4 betas, 2 lambdas).
    lam_form : required, 'rate' (x = lam*tau) | 'time' (x = tau/lam); see ``ns_yield``.
    lam : float (ns) or 2-tuple (svensson). If given, lambda is held fixed and
        the betas come from linear least squares; lambda is never re-estimated.
    lam_bounds : required when lam is None. ns: (lo, hi). svensson: (lo, hi)
        applied to both, or ((lo1, hi1), (lo2, hi2)). Disjoint Svensson bounds
        help identification.
    weights : array or None, non-negative weights w_i (weighted least squares).

    Returns
    -------
    dict with betas (ndarray), lams (ndarray, length 1 or 2), fitted, resid
    (y - fitted), sse (= sum w_i resid_i^2), rmse (= sqrt(sse/n), n = number of
    observations including zero-weight ones; with weights this is NOT
    sqrt(mean(resid^2)); use sqrt(sse/(n - p)) for a degrees-of-freedom
    version), n, lam_fixed (bool).

    Method: min over beta, lam of sum w_i (y_i - ns_yield(tau_i))^2. For each
    lambda the betas are profiled out by OLS (np.linalg.lstsq); lambda is then
    found by a coarse log-grid search, bounded Brent (ns) or L-BFGS-B in log
    lambda (svensson), and a final scipy least_squares polish of the profiled
    residuals (kept only if it lowers sse).

    Example
    -------
    >>> res = fit_ns(tau, y, model='ns', lam_form='rate', lam=0.0609 * 12)   # Diebold-Li lambda in 1/years
    >>> res['betas']
    """
    mdl = str(model).strip().lower()
    if mdl in ("ns", "nelson-siegel", "nelson_siegel", "nelsonsiegel"):
        p, nl = 3, 1
    elif mdl in ("svensson", "nss", "nelson-siegel-svensson"):
        p, nl = 4, 2
    else:
        raise ValueError("model must be 'ns' or 'svensson'")
    form = _parse_lam_form(lam_form)
    tt = np.asarray(tau, dtype=float).ravel()
    yy = np.asarray(y, dtype=float).ravel()
    n = tt.size
    if yy.size != n:
        raise ValueError("tau and y must have the same length")
    if n < p:
        raise ValueError(f"need at least {p} observations")
    if not (np.all(np.isfinite(tt)) and np.all(np.isfinite(yy))) or np.any(tt < 0):
        raise ValueError("tau and y must be finite and tau >= 0")
    if weights is None:
        w = np.ones(n)
    else:
        w = np.asarray(weights, dtype=float).ravel()
        if w.size != n or not np.all(np.isfinite(w)) or np.any(w < 0) or not np.any(w > 0):
            raise ValueError("weights must be finite, non-negative, not all zero and match y")
    sw = np.sqrt(w)

    def profile(lams):
        X = _ns_loadings(tt, lams, form, p)
        beta = np.linalg.lstsq(X * sw[:, None], yy * sw, rcond=None)[0]
        r = yy - X @ beta
        return float(np.sum(w * r * r)), beta, X

    if lam is not None:
        lams = _parse_lams(lam, nl)
        fixed = True
    else:
        if lam_bounds is None:
            raise ValueError("lam_bounds is required when lam is None (lambda estimated)")
        lb = np.asarray(lam_bounds, dtype=float)
        if lb.shape == (2,):
            bnds = np.tile(lb, (nl, 1))
        elif lb.shape == (nl, 2):
            bnds = lb
        else:
            raise ValueError("lam_bounds must be (lo, hi) or, for svensson, ((lo1, hi1), (lo2, hi2))")
        if not np.all(np.isfinite(bnds)) or np.any(bnds[:, 0] <= 0) or np.any(bnds[:, 1] <= bnds[:, 0]):
            raise ValueError("lam_bounds need 0 < lo < hi")
        fixed = False

        def sse_of(lv):
            return profile(lv)[0]

        if nl == 1:
            lo, hi = bnds[0]
            grid = np.geomspace(lo, hi, 60)
            vals = np.array([sse_of([g]) for g in grid])
            i = int(np.argmin(vals))
            a, b = grid[max(i - 1, 0)], grid[min(i + 1, grid.size - 1)]
            best = np.array([grid[i]])
            best_sse = vals[i]
            res = optimize.minimize_scalar(lambda L: sse_of([L]), bounds=(a, b), method="bounded",
                                           options={"xatol": 1e-14 * max(1.0, b), "maxiter": 1000})
            if np.isfinite(res.fun) and res.fun <= best_sse:
                best, best_sse = np.array([float(res.x)]), float(res.fun)
        else:
            g1 = np.geomspace(bnds[0, 0], bnds[0, 1], 30)
            g2 = np.geomspace(bnds[1, 0], bnds[1, 1], 30)
            vals = np.array([[sse_of([a, b]) for b in g2] for a in g1])
            i, j = np.unravel_index(int(np.argmin(vals)), vals.shape)
            best = np.array([g1[i], g2[j]])
            best_sse = float(vals[i, j])
            scale = max(best_sse, 1e-300)
            logb = np.log(bnds)
            res = optimize.minimize(lambda u: sse_of(np.exp(u)) / scale, np.log(best), method="L-BFGS-B",
                                    bounds=[tuple(r) for r in logb],
                                    options={"ftol": 1e-15, "gtol": 1e-12, "maxiter": 2000})
            cand = np.exp(np.clip(res.x, logb[:, 0], logb[:, 1]))
            cs = sse_of(cand)
            if np.isfinite(cs) and cs <= best_sse:
                best, best_sse = cand, cs
        # Gauss-Newton polish on the profiled residual vector (log lambda), kept only if better.
        logb = np.log(bnds)
        u0 = np.clip(np.log(best), logb[:, 0], logb[:, 1])

        def rvec(u):
            X = _ns_loadings(tt, np.exp(u), form, p)
            beta = np.linalg.lstsq(X * sw[:, None], yy * sw, rcond=None)[0]
            return sw * (yy - X @ beta)

        try:
            pol = optimize.least_squares(rvec, u0, bounds=(logb[:, 0], logb[:, 1]), jac="3-point",
                                         ftol=1e-15, xtol=1e-15, gtol=1e-15, max_nfev=200 * nl)
            cand = np.exp(pol.x)
            cs = sse_of(cand)
            if np.isfinite(cs) and cs < best_sse:
                best, best_sse = cand, cs
        except (ValueError, np.linalg.LinAlgError):
            pass
        lams = best

    sse, beta, X = profile(lams)
    fitted = X @ beta
    return {
        "betas": beta,
        "lams": np.asarray(lams, dtype=float).copy(),
        "fitted": fitted,
        "resid": yy - fitted,
        "sse": sse,
        "rmse": float(np.sqrt(sse / n)),
        "n": int(n),
        "lam_fixed": fixed,
    }
