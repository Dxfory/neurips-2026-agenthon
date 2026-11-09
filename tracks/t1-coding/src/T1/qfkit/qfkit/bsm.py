"""Black-Scholes-Merton and Black-76 European option pricing, Greeks and implied volatility.

Conventions for the whole module: T is time to expiry in YEARS; r and q are
continuously compounded annual rates as decimals (0.05, not 5); sigma is the
annual volatility as a decimal (0.20, not 20). ``kind`` is always required and
is ONE string, 'call' or 'put', for the whole call (price a mixed book in two
calls, or combine with np.where). N is the standard normal CDF
(scipy.special.ndtr, the same function as scipy.stats.norm.cdf); N(-d) is
evaluated directly, never 1-N(d).
All numeric inputs broadcast. Functions return a numpy float64 (a ``float``
subclass) when every input is a scalar and an ``np.ndarray`` otherwise.

Functions
---------
bsm_d1d2             d1, d2 with dividend yield q
bsm_price            European call/put price (T == 0 -> intrinsic)
bsm_greeks           price, delta, gamma, vega, theta, rho with selectable units
bsm_pde_residual     residual of the BSM PDE for a price and its Greeks
parity_residual      put-call parity residual C - P - (S e^{-qT} - K e^{-rT})
bsm_bounds           no-arbitrage lower/upper bounds of a European price
implied_vol          vectorised bracketed BSM implied volatility (nan if no root)
black76_price        Black-76 price on a forward/futures (caplets, floorlets)
black76_implied_vol  vectorised bracketed Black-76 implied volatility
"""

from __future__ import annotations

import math
import numbers

import numpy as np
from scipy.special import ndtr

__all__ = [
    "bsm_d1d2",
    "bsm_price",
    "bsm_greeks",
    "bsm_pde_residual",
    "parity_residual",
    "bsm_bounds",
    "implied_vol",
    "black76_price",
    "black76_implied_vol",
]

_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)
_THETA_DIVISORS = {"year": 1.0, "calendar_day": 365.0, "trading_day": 252.0}
_UNIT_DIVISORS = {"unit": 1.0, "percent": 100.0}
_FTOL_REL = 1e-14  # relative price tolerance of the implied-vol solvers


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _out(x):
    """0-d result -> numpy float64 scalar; otherwise the ndarray itself."""
    x = np.asarray(x, dtype=float)
    return x[()] if x.ndim == 0 else x


def _arr(x):
    return np.asarray(x, dtype=float)


def _sign(kind):
    """'call' -> +1.0, 'put' -> -1.0 (case-insensitive); anything else raises."""
    if not isinstance(kind, str):
        raise TypeError("kind must be the string 'call' or 'put'")
    k = kind.strip().lower()
    if k == "call":
        return 1.0
    if k == "put":
        return -1.0
    raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")


def _check_nonneg(**named):
    for name, v in named.items():
        if np.any(v < 0):
            raise ValueError(f"{name} must be >= 0")


def _check_pos(**named):
    for name, v in named.items():
        if np.any(v <= 0):
            raise ValueError(f"{name} must be > 0")


def _divisor(value, table, argname):
    if isinstance(value, str):
        if value not in table:
            raise ValueError(f"{argname} must be one of {sorted(table)} or a positive number, got {value!r}")
        return table[value]
    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        v = float(value)
        if math.isfinite(v) and v > 0:
            return v
    raise ValueError(f"{argname} must be one of {sorted(table)} or a positive number, got {value!r}")


def _phi(x):
    with np.errstate(over="ignore"):  # |x| ~ 1e300 (sigma sqrt T tiny): x*x = inf -> phi = 0
        return _INV_SQRT_2PI * np.exp(-0.5 * x * x)


def _core_price(Fd, Kd, lnm, T, sigma, w):
    """Black formula on discounted forward Fd and discounted strike Kd.

    price = w [Fd N(w d1) - Kd N(w d2)], d1 = lnm/(sigma sqrt T) + sigma sqrt T / 2,
    lnm = ln(Fd/Kd). Where sigma sqrt T == 0 the deterministic limit
    max(w (Fd - Kd), 0) is returned (at T == 0 this is the intrinsic value).
    """
    s = sigma * np.sqrt(T)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        d1 = lnm / s + 0.5 * s
        d2 = d1 - s
        p = w * (Fd * ndtr(w * d1) - Kd * ndtr(w * d2))
    det = np.maximum(w * (Fd - Kd), 0.0)
    return np.where(s == 0, det, p)


# ----------------------------------------------------------------------------
# Black-Scholes-Merton
# ----------------------------------------------------------------------------
def bsm_d1d2(S, K, T, r, sigma, *, q=0.0):
    """Black-Scholes-Merton d1 and d2 with a continuous dividend yield q.

    Parameters
    ----------
    S, K : spot and strike (> 0).
    T : time to expiry in years (> 0).
    r, q : continuously compounded annual risk-free rate and dividend yield
        (decimals). q defaults to 0.0 (no dividends).
    sigma : annual volatility as a decimal (0.20, not 20), > 0.

    Returns
    -------
    (d1, d2), each broadcast over the inputs.

    Formula
    -------
    d1 = [ln(S/K) + (r - q + sigma^2/2) T] / (sigma sqrt T);  d2 = d1 - sigma sqrt T.

    Example
    -------
    >>> bsm_d1d2(42, 40, 0.5, 0.10, 0.20)          # Hull Ex. 15.6
    (0.7692626..., 0.6278413...)
    """
    S, K, T, r, sigma, q = map(_arr, (S, K, T, r, sigma, q))
    _check_pos(S=S, K=K, T=T, sigma=sigma)
    s = sigma * np.sqrt(T)
    with np.errstate(over="ignore"):  # sigma sqrt T tiny -> d1 = +-inf, no warning
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / s
    d2 = d1 - s
    return _out(d1), _out(d2)


def bsm_price(S, K, T, r, sigma, *, kind, q=0.0):
    """European call or put price under Black-Scholes-Merton with dividend yield q.

    Parameters
    ----------
    S, K : spot and strike (>= 0).
    T : time to expiry in years (>= 0). T == 0 returns the intrinsic value
        max(S-K, 0) / max(K-S, 0).
    r, q : continuously compounded annual rate and dividend yield (decimals);
        q defaults to 0.0.
    sigma : annual volatility as a decimal (>= 0). sigma == 0 returns the
        deterministic limit max(S e^{-qT} - K e^{-rT}, 0) (call) or
        max(K e^{-rT} - S e^{-qT}, 0) (put).
    kind : 'call' or 'put' (required).

    Returns
    -------
    Option price (same currency units as S and K).

    Formula
    -------
    C = S e^{-qT} N(d1) - K e^{-rT} N(d2);  P = K e^{-rT} N(-d2) - S e^{-qT} N(-d1),
    d1, d2 as in bsm_d1d2.

    Example
    -------
    >>> bsm_price(42, 40, 0.5, 0.10, 0.20, kind='call')   # Hull Ex. 15.6
    4.7594...
    """
    w = _sign(kind)
    S, K, T, r, sigma, q = map(_arr, (S, K, T, r, sigma, q))
    _check_nonneg(S=S, K=K, T=T, sigma=sigma)
    with np.errstate(divide="ignore", invalid="ignore"):
        lnm = np.log(S / K) + (r - q) * T
    Fd = S * np.exp(-q * T)
    Kd = K * np.exp(-r * T)
    return _out(_core_price(Fd, Kd, lnm, T, sigma, w))


def bsm_greeks(S, K, T, r, sigma, *, kind, q=0.0, vega_per="unit", theta_per="year", rho_per="unit"):
    """Analytic BSM price and Greeks (delta, gamma, vega, theta, rho) with selectable units.

    Parameters
    ----------
    S, K, T, r, sigma, q : as in bsm_price, but T > 0, sigma > 0, S > 0, K > 0.
    kind : 'call' or 'put' (required).
    vega_per : 'unit' (dV/dsigma, per 1.00 of vol; default) or 'percent'
        (per 1 vol point = unit/100).
    theta_per : theta is dV/dt in calendar time (= -dV/dT, negative for a long
        vanilla in normal cases). 'year' (default, the analytic derivative),
        'calendar_day' (/365), 'trading_day' (/252), or a number N (/N).
    rho_per : 'unit' (dV/dr per 1.00 of rate; default) or 'percent' (/100,
        i.e. per 1 percentage point = 100 bp). rho is the sensitivity to r
        only; the sensitivity to the dividend yield q is not returned.
    Note: a per-day theta here is the annual derivative divided by 365/252, not
    the one-day revaluation V(T - 1/365) - V(T) (they differ by O(1/365^2)).

    Returns
    -------
    dict with keys 'price', 'delta', 'gamma', 'vega', 'theta', 'rho'.
    delta = dV/dS, gamma = d2V/dS2 (always per 1 unit of S). Gamma and vega are
    the same for calls and puts.

    Formula
    -------
    delta_c = e^{-qT} N(d1);  delta_p = -e^{-qT} N(-d1)  (= e^{-qT}(N(d1) - 1))
    gamma = e^{-qT} phi(d1) / (S sigma sqrt T);  vega = S e^{-qT} phi(d1) sqrt T
    theta_c = -S e^{-qT} phi(d1) sigma/(2 sqrt T) + q S e^{-qT} N(d1) - r K e^{-rT} N(d2)
    theta_p = -S e^{-qT} phi(d1) sigma/(2 sqrt T) - q S e^{-qT} N(-d1) + r K e^{-rT} N(-d2)
    rho_c = K T e^{-rT} N(d2);  rho_p = -K T e^{-rT} N(-d2)

    Example
    -------
    >>> g = bsm_greeks(49, 50, 20/52, 0.05, 0.20, kind='call', theta_per='calendar_day')
    >>> round(float(g['delta']), 3), round(float(g['theta']), 4)   # Hull Ch.19
    (0.522, -0.0118)
    """
    w = _sign(kind)
    vdiv = _divisor(vega_per, _UNIT_DIVISORS, "vega_per")
    tdiv = _divisor(theta_per, _THETA_DIVISORS, "theta_per")
    rdiv = _divisor(rho_per, _UNIT_DIVISORS, "rho_per")
    S, K, T, r, sigma, q = map(_arr, (S, K, T, r, sigma, q))
    _check_pos(S=S, K=K, T=T, sigma=sigma)
    sqT = np.sqrt(T)
    s = sigma * sqT
    with np.errstate(over="ignore"):  # sigma sqrt T tiny -> d1 = +-inf, no warning
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / s
    d2 = d1 - s
    eq = np.exp(-q * T)
    er = np.exp(-r * T)
    Fd = S * eq
    Kd = K * er
    Nw1 = ndtr(w * d1)
    Nw2 = ndtr(w * d2)
    pdf1 = _phi(d1)
    price = w * (Fd * Nw1 - Kd * Nw2)
    delta = w * eq * Nw1
    gamma = eq * pdf1 / (S * s)
    vega = Fd * pdf1 * sqT
    theta = -Fd * pdf1 * sigma / (2.0 * sqT) + w * (q * Fd * Nw1 - r * Kd * Nw2)
    rho = w * K * T * er * Nw2
    return {
        "price": _out(price),
        "delta": _out(delta),
        "gamma": _out(gamma),
        "vega": _out(vega / vdiv),
        "theta": _out(theta / tdiv),
        "rho": _out(rho / rdiv),
    }


def bsm_pde_residual(S, sigma, r, price, delta, gamma, theta, *, q=0.0):
    """Residual of the Black-Scholes-Merton PDE for a given option price and Greeks.

    About 0 when price, delta, gamma and theta are a consistent BSM solution.

    Parameters
    ----------
    S : spot.  sigma : annual volatility (decimal).  r, q : continuously
        compounded annual rate and dividend yield (q defaults to 0.0).
    price : the option value V itself (NOT vega or another Greek).
    delta, gamma : dV/dS and d2V/dS2.
    theta : dV/dt PER YEAR (calendar time). If you scaled theta per day,
        multiply it back (e.g. by 365 or 252) before calling.

    Returns
    -------
    res = theta + (r - q) S delta + 0.5 sigma^2 S^2 gamma - r V.

    Example
    -------
    >>> g = bsm_greeks(100, 95, 0.75, 0.04, 0.30, kind='call', q=0.02)
    >>> abs(bsm_pde_residual(100, 0.30, 0.04, g['price'], g['delta'], g['gamma'], g['theta'], q=0.02)) < 1e-12
    True
    """
    S, sigma, r, price, delta, gamma, theta, q = map(_arr, (S, sigma, r, price, delta, gamma, theta, q))
    return _out(theta + (r - q) * S * delta + 0.5 * sigma * sigma * S * S * gamma - r * price)


def parity_residual(call, put, S, K, T, r, *, q=0.0):
    """Put-call parity residual C - P - (S e^{-qT} - K e^{-rT}) for European options.

    Parameters
    ----------
    call, put : European call and put prices with the same S, K, T.
    S, K : spot and strike.  T : years to expiry.
    r, q : continuously compounded annual rate (discounts the strike) and
        dividend yield (discounts the spot); q defaults to 0.0.

    Returns
    -------
    C - P - (S e^{-qT} - K e^{-rT}); 0 under no arbitrage. For options on a
    forward F with discount factor df pass S = F*df, q = 0, r = -ln(df)/T
    (so that K e^{-rT} = K df); the residual is then C - P - df (F - K).

    Example
    -------
    >>> c = bsm_price(42, 40, 0.5, 0.10, 0.20, kind='call')        # Hull Ex. 15.6
    >>> p = bsm_price(42, 40, 0.5, 0.10, 0.20, kind='put')
    >>> abs(parity_residual(c, p, 42, 40, 0.5, 0.10)) < 1e-12
    True
    """
    call, put, S, K, T, r, q = map(_arr, (call, put, S, K, T, r, q))
    return _out(call - put - (S * np.exp(-q * T) - K * np.exp(-r * T)))


def bsm_bounds(S, K, T, r, *, kind, q=0.0):
    """No-arbitrage lower and upper bounds for a European call or put price.

    Parameters
    ----------
    S, K, T, r, q : as in bsm_price (q defaults to 0.0).
    kind : 'call' or 'put' (required).

    Returns
    -------
    (lower, upper):
    call: (max(S e^{-qT} - K e^{-rT}, 0), S e^{-qT});
    put:  (max(K e^{-rT} - S e^{-qT}, 0), K e^{-rT})   (Hull OFOD Ch.11).

    Example
    -------
    >>> bsm_bounds(42, 40, 0.5, 0.10, kind='call')    # (42 - 40 e^{-0.05}, 42)
    (3.9508..., 42.0)
    """
    w = _sign(kind)
    S, K, T, r, q = map(_arr, (S, K, T, r, q))
    Fd = S * np.exp(-q * T)
    Kd = K * np.exp(-r * T)
    lower = np.maximum(w * (Fd - Kd), 0.0)
    upper = Fd if w > 0 else Kd
    # np.array copies: broadcast_arrays returns stride-0 views that share memory
    # (writing one element would change all of them).
    lower, upper = (np.array(a) for a in np.broadcast_arrays(lower, upper))
    return _out(lower), _out(upper)


# ----------------------------------------------------------------------------
# implied volatility engine (shared by BSM and Black-76)
# ----------------------------------------------------------------------------
def _iv_solve(target, Fd, Kd, lnm, T, w, lo, hi, xtol, maxiter):
    """Safeguarded Newton-bisection for sigma in Fd N(d1) - Kd N(d2) = target.

    Works on the out-of-the-money side (put when Fd > Kd, else call), whose
    price is obtained from the target through put-call parity, and applies
    Newton to ln(price) (same root, better scaled for tiny prices). A Newton
    step is accepted only inside the current bracket [a, b] with
    price(a) <= target <= price(b); otherwise the bracket is bisected.
    """
    lo = float(lo)
    hi = float(hi)
    xtol = float(xtol)
    maxiter = int(maxiter)
    if not (math.isfinite(lo) and math.isfinite(hi) and 0.0 < lo < hi):
        raise ValueError("need 0 < lo < hi (finite)")
    arrs = np.broadcast_arrays(*(_arr(a) for a in (target, Fd, Kd, lnm, T)))
    shape = arrs[0].shape
    target, Fd, Kd, lnm, T = (a.ravel() for a in arrs)
    out = np.full(target.size, np.nan)

    with np.errstate(all="ignore"):
        ok = (
            np.isfinite(target) & np.isfinite(Fd) & np.isfinite(Kd) & np.isfinite(lnm) & np.isfinite(T)
            & (T > 0) & (Fd > 0) & (Kd > 0)
        )
        lower = np.maximum(w * (Fd - Kd), 0.0)
        upper = Fd if w > 0 else Kd
        ok &= (target > lower) & (target < upper)
        wo = np.where(Fd > Kd, -1.0, 1.0)  # out-of-the-money side
        tgt = np.where(wo == w, target, target - w * (Fd - Kd))
        ok &= tgt > 0
    idx = np.flatnonzero(ok)
    if idx.size == 0:
        return _out(out.reshape(shape))

    tgt, Fd, Kd, lnm, sqT, wo = tgt[idx], Fd[idx], Kd[idx], lnm[idx], np.sqrt(T[idx]), wo[idx]
    lnt = np.log(tgt)

    def price_vega(sel, sig):
        s = sig * sqT[sel]
        d1 = lnm[sel] / s + 0.5 * s
        d2 = d1 - s
        ws = wo[sel]
        p = ws * (Fd[sel] * ndtr(ws * d1) - Kd[sel] * ndtr(ws * d2))
        v = Fd[sel] * _phi(d1) * sqT[sel]
        return p, v

    m = idx.size
    allsel = np.arange(m)
    with np.errstate(all="ignore"):
        # root must lie inside [lo, hi]
        p_lo, _ = price_vega(allsel, np.full(m, lo))
        p_hi, _ = price_vega(allsel, np.full(m, hi))
        inside = (p_lo <= tgt) & (p_hi >= tgt)
        a = np.full(m, lo)
        b = np.full(m, hi)
        # Manaster-Koehler start: the vega-maximising sigma, sqrt(2|ln(F/K)|/T)
        x = np.clip(np.sqrt(2.0 * np.abs(lnm)) / sqT, lo, hi)
        res = np.full(m, np.nan)
        act = np.flatnonzero(inside)
        for _ in range(maxiter):
            if act.size == 0:
                break
            xa = x[act]
            p, v = price_vega(act, xa)
            f = np.where(p > 0, np.log(p) - lnt[act], -np.inf)
            aa = np.where(f < 0, xa, a[act])
            bb = np.where(f > 0, xa, b[act])
            a[act] = aa
            b[act] = bb
            xn = xa - f * p / v  # Newton on ln(price): d ln p / d sigma = vega / p
            newton = np.isfinite(xn) & (xn > aa) & (xn < bb)
            done = (np.abs(f) <= _FTOL_REL) | (bb - aa < xtol) | (newton & (np.abs(xn - xa) < xtol))
            # converged: keep the final in-bracket Newton correction (free polish)
            final = np.where(newton, xn, xa)
            x[act] = np.where(newton, xn, 0.5 * (aa + bb))
            res[act[done]] = final[done]
            act = act[~done]
    out[idx] = res
    return _out(out.reshape(shape))


def implied_vol(price, S, K, T, r, *, kind, q=0.0, lo=1e-8, hi=10.0, xtol=1e-12, maxiter=200):
    """Vectorised, bracketed Black-Scholes-Merton implied volatility; nan when no valid root.

    Parameters
    ----------
    price : observed European option price(s).
    S, K, T, r, q : as in bsm_price (T in years, r and q continuously
        compounded decimals; q defaults to 0.0).
    kind : 'call' or 'put' (required).
    lo, hi : scalar search bracket for sigma (annual decimal), 0 < lo < hi.
    xtol : absolute tolerance on sigma.  maxiter : iteration cap (most
        options need 5-20 iterations; keep the default unless timing matters).

    Returns
    -------
    sigma (annual decimal). Never raises for bad data: nan where price <= the
    lower bound or >= the upper bound of bsm_bounds, any input is non-finite,
    T <= 0, S <= 0, K <= 0, the root lies outside [lo, hi], or the solver has
    not converged after maxiter iterations.

    Method
    ------
    Safeguarded Newton-bisection on arrays: keep [a, b] with
    price(a) <= target <= price(b); accept a Newton step only inside the
    bracket, otherwise bisect; stop when b - a < xtol, the Newton step is
    < xtol, or the relative price error is < 1e-14 (the last in-bracket
    Newton correction is kept). Solved on the out-of-the-money side via
    put-call parity, Newton applied to ln(price), started at the
    Manaster-Koehler point sqrt(2|ln(F/K)|/T); typically 5-15 iterations.
    Note: deep in-the-money prices whose time value is below double
    precision cannot identify sigma (they equal the lower bound -> nan).

    Example
    -------
    >>> c = bsm_price(42, 40, 0.5, 0.10, 0.20, kind='call')        # Hull Ex. 15.6
    >>> implied_vol(c, 42, 40, 0.5, 0.10, kind='call')             # 0.2 (to 1e-12)
    >>> implied_vol([4.759422, 0.5], 42, 40, 0.5, 0.10, kind='call')
    array([0.19999996, nan])      # rounded quote; 0.5 < lower bound 3.95 -> nan
    """
    w = _sign(kind)
    price, S, K, T, r, q = map(_arr, (price, S, K, T, r, q))
    with np.errstate(all="ignore"):
        Fd = S * np.exp(-q * T)
        Kd = K * np.exp(-r * T)
        lnm = np.log(S / K) + (r - q) * T
        bad = ~(np.isfinite(S) & np.isfinite(K) & np.isfinite(r) & np.isfinite(q) & (S > 0) & (K > 0))
    Fd = np.where(bad, np.nan, Fd)
    return _iv_solve(price, Fd, Kd, lnm, T, w, lo, hi, xtol, maxiter)


# ----------------------------------------------------------------------------
# Black-76
# ----------------------------------------------------------------------------
def black76_price(F, K, T, sigma, *, df, kind):
    """Black-76 price of a European option on a forward or futures price (also caplets/floorlets).

    Parameters
    ----------
    F : forward or futures price (>= 0) for the option's expiry.
    K : strike (>= 0).
    T : time to expiry in years (>= 0), the time used for the volatility
        (for a caplet: the rate reset/fixing time t_k, not the payment time).
    sigma : annual lognormal volatility of F as a decimal (>= 0).
    df : REQUIRED discount factor to the PAYMENT date: e^{-rT} for a futures
        option; P(0, T + delta) for a caplet/floorlet (then multiply the result
        by notional * delta).
    kind : 'call' or 'put' (required); a caplet is a call on the rate, a
        floorlet a put.

    Returns
    -------
    Option price per unit of the underlying. T == 0 or sigma == 0 gives
    df * max(F - K, 0) (call) / df * max(K - F, 0) (put).

    Formula
    -------
    d1 = [ln(F/K) + sigma^2 T/2]/(sigma sqrt T), d2 = d1 - sigma sqrt T;
    call = df [F N(d1) - K N(d2)];  put = df [K N(-d2) - F N(-d1)].

    Example
    -------
    >>> black76_price(20, 20, 4/12, 0.25, df=math.exp(-0.09*4/12), kind='put')   # Hull Ch.18
    1.1166...
    """
    w = _sign(kind)
    F, K, T, sigma, df = map(_arr, (F, K, T, sigma, df))
    _check_nonneg(F=F, K=K, T=T, sigma=sigma, df=df)
    with np.errstate(divide="ignore", invalid="ignore"):
        lnm = np.log(F / K)
    return _out(_core_price(df * F, df * K, lnm, T, sigma, w))


def black76_implied_vol(price, F, K, T, *, df, kind, lo=1e-8, hi=10.0, xtol=1e-12, maxiter=200):
    """Vectorised bracketed Black-76 implied volatility; nan outside the no-arbitrage bounds.

    Parameters
    ----------
    price : observed option price per unit (for a caplet divide the quoted
        premium by notional * delta first).
    F, K, T : forward, strike, years to expiry.
    df : REQUIRED discount factor to the payment date (as in black76_price).
    kind : 'call' or 'put' (required).
    lo, hi, xtol, maxiter : as in implied_vol.

    Returns
    -------
    sigma (annual decimal). nan (never raises) where the price is outside
    call (df max(F-K, 0), df F) / put (df max(K-F, 0), df K), any input is
    non-finite, T <= 0, F <= 0, K <= 0, df <= 0, the root lies outside
    [lo, hi], or no convergence within maxiter. Same algorithm as implied_vol.

    Formula
    -------
    Solve black76_price(F, K, T, sigma, df=df, kind=kind) = price for sigma.

    Example
    -------
    >>> p = black76_price(0.07, 0.08, 1.0, 0.20, df=math.exp(-0.065*1.25), kind='call')
    >>> black76_implied_vol(p, 0.07, 0.08, 1.0, df=math.exp(-0.065*1.25), kind='call')
    0.2000000...
    """
    w = _sign(kind)
    price, F, K, T, df = map(_arr, (price, F, K, T, df))
    with np.errstate(all="ignore"):
        lnm = np.log(F / K)
        Fd = df * F
        Kd = df * K
        bad = ~(np.isfinite(F) & np.isfinite(K) & np.isfinite(df) & (F > 0) & (K > 0) & (df > 0))
    Fd = np.where(bad, np.nan, Fd)
    return _iv_solve(price, Fd, Kd, lnm, T, w, lo, hi, xtol, maxiter)
