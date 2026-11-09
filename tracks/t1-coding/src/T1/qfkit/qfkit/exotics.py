"""Closed-form exotic options: barriers, binaries, lookbacks, Asians, two-asset, compound, chooser.

Conventions for the whole module (the same as ``qfkit.bsm``): S is spot, K is
the strike, T is time to expiry in YEARS; r and q are continuously compounded
annual rates as decimals and b = r - q is the cost of carry; sigma is the
annual volatility as a decimal (0.20, not 20). ``kind`` is always required and
is ONE string, 'call' or 'put', for the whole call. N is the standard normal
CDF (scipy.special.ndtr) and M(a, b; rho) the standard bivariate normal CDF.
Every option is written on lognormal (Black-Scholes-Merton) underlyings with
constant r, q and sigma; path-dependent payoffs are monitored CONTINUOUSLY
unless a function says otherwise (barrier_price takes monitor_dt for the
Broadie-Glasserman-Kou discrete-monitoring shift).
All numeric inputs broadcast. Functions return a numpy float64 (a ``float``
subclass) when every input is a scalar and an ``np.ndarray`` otherwise.

Functions
---------
barrier_price            the 8 single-barrier knock-in/knock-out options, rebates, BGK shift
digital_price            cash-or-nothing and asset-or-nothing European binaries
lookback_floating_price  floating-strike lookback (Goldman-Sosin-Gatto)
lookback_fixed_price     fixed-strike lookback (Conze-Viswanathan)
asian_geometric_price    exact geometric-average option, continuous or discrete fixings
asian_tw_price           Turnbull-Wakeman arithmetic-average approximation
asian_levy_price         Levy arithmetic-average approximation, continuous or discrete
margrabe_price           option to exchange one asset for another
kirk_spread_price        Kirk spread-option approximation on forwards
geske_compound_price     option on an option (Geske)
chooser_price            simple chooser (Rubinstein)

References: Haug, The Complete Guide to Option Pricing Formulas, 2nd ed. (2007),
Ch. 4-5; Hull, Options, Futures, and Other Derivatives (OFOD), "Exotic Options".
"""

from __future__ import annotations

import math

import numpy as np
from scipy import optimize
from scipy.special import exprel, log_ndtr, ndtr

from .bsm import bsm_price

__all__ = [
    "barrier_price",
    "digital_price",
    "lookback_floating_price",
    "lookback_fixed_price",
    "asian_geometric_price",
    "asian_tw_price",
    "asian_levy_price",
    "margrabe_price",
    "kirk_spread_price",
    "geske_compound_price",
    "chooser_price",
]

_TWO_PI = 2.0 * math.pi
_INV_SQRT_2PI = 1.0 / math.sqrt(_TWO_PI)
# Broadie-Glasserman-Kou (1997) constant beta = -zeta(1/2)/sqrt(2 pi), zeta(1/2) = -1.4603545088095868
_BGK_BETA = 1.4603545088095868 / math.sqrt(_TWO_PI)
_BARRIERS = ("down-in", "down-out", "up-in", "up-out")

# Gauss-Legendre rules: 20 points on [0, 2] for the bivariate normal (Genz 2004),
# 48 points on [0, 1] for the continuous-average second moment.
_GL20_X, _GL20_W = np.polynomial.legendre.leggauss(20)
_BV_X = 1.0 + _GL20_X
_BV_W = _GL20_W
_GL48_X, _GL48_W = np.polynomial.legendre.leggauss(48)
_M2_X = 0.5 * (1.0 + _GL48_X)
_M2_W = 0.5 * _GL48_W

# coefficients of (A, B, C, D) in Haug's standard-barrier table, (strike > barrier, strike < barrier)
_BARRIER_COEF = {
    ("down-in", 1.0): ((0, 0, 1, 0), (1, -1, 0, 1)),
    ("up-in", 1.0): ((1, 0, 0, 0), (0, 1, -1, 1)),
    ("down-in", -1.0): ((0, 1, -1, 1), (1, 0, 0, 0)),
    ("up-in", -1.0): ((1, -1, 0, 1), (0, 0, 1, 0)),
    ("down-out", 1.0): ((1, 0, -1, 0), (0, 1, 0, -1)),
    ("up-out", 1.0): ((0, 0, 0, 0), (1, -1, 1, -1)),
    ("down-out", -1.0): ((1, -1, 1, -1), (0, 0, 0, 0)),
    ("up-out", -1.0): ((0, 1, 0, -1), (1, 0, -1, 0)),
}


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _out(x):
    """0-d result -> numpy float64 scalar; otherwise the ndarray itself."""
    x = np.asarray(x, dtype=float)
    return x[()] if x.ndim == 0 else x


def _arr(x):
    return np.asarray(x, dtype=float)


def _choice(value, allowed, argname):
    """Case-insensitive string option; anything outside ``allowed`` raises."""
    if not isinstance(value, str):
        raise TypeError(f"{argname} must be one of {list(allowed)}, got {value!r}")
    v = value.strip().lower()
    if v not in allowed:
        raise ValueError(f"{argname} must be one of {list(allowed)}, got {value!r}")
    return v


def _sign(kind, argname="kind"):
    """'call' -> +1.0, 'put' -> -1.0 (case-insensitive); anything else raises."""
    return 1.0 if _choice(kind, ("call", "put"), argname) == "call" else -1.0


def _check_pos(**named):
    for name, v in named.items():
        if np.any(v <= 0):  # nan passes and propagates, as in qfkit.bsm
            raise ValueError(f"{name} must be > 0")


def _check_nonneg(**named):
    for name, v in named.items():
        if np.any(v < 0):
            raise ValueError(f"{name} must be >= 0")


def _phi(x):
    with np.errstate(over="ignore"):
        return _INV_SQRT_2PI * np.exp(-0.5 * x * x)


def _black(Fd, Kd, s, w):
    """w [Fd N(w d1) - Kd N(w d2)], d1 = ln(Fd/Kd)/s + s/2, d2 = d1 - s.

    Fd and Kd are the discounted forward and the discounted strike, s the total
    volatility sigma sqrt(T). s == 0 gives the deterministic limit max(w (Fd - Kd), 0).
    """
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        d1 = np.log(Fd / Kd) / s + 0.5 * s
        d2 = d1 - s
        p = w * (Fd * ndtr(w * d1) - Kd * ndtr(w * d2))
    det = np.maximum(w * (Fd - Kd), 0.0)
    return np.where(s == 0, det, p)


def _pow_N(log_ratio, power, x):
    """ratio**power * N(x), evaluated in log space (no overflow of the power)."""
    with np.errstate(over="ignore", invalid="ignore"):
        return np.exp(power * log_ratio + log_ndtr(x))


def _expm1_div(a, eps):
    """expm1(a eps) / eps, equal to a at eps == 0."""
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        v = np.expm1(a * eps) / eps
    return np.where(eps == 0, a, v)


def _dN_div(c, h):
    """(N(c + h) - N(c)) / h, accurate for small |h| (h == 0 gives phi(c))."""
    with np.errstate(divide="ignore", invalid="ignore"):
        diff = np.where(c > 0, ndtr(-c) - ndtr(-c - h), ndtr(c + h) - ndtr(c))
        direct = diff / h
    # Taylor series of N about c: N' = phi, N'' = -c phi, N''' = (c^2-1) phi, N'''' = (3c-c^3) phi
    series = _phi(c) * (1.0 - c * h / 2.0 + (c * c - 1.0) * h * h / 6.0 + (3.0 * c - c**3) * h**3 / 24.0)
    return np.where(np.abs(h) < 1e-3, series, direct)


def _lookback_G(S, e, T, b, sigma, w):
    """(1/eps) [ (S/e)^{-eps} N(c + w eps s) - e^{bT} N(c) ],  eps = 2b/sigma^2, s = sigma sqrt T,

    c = -w a1, a1 = [ln(S/e) + (b + sigma^2/2) T]/s. This is the sigma^2/(2b) term of the
    Goldman-Sosin-Gatto / Conze-Viswanathan formulas, finite and smooth through b = 0.
    """
    s = sigma * np.sqrt(T)
    L = np.log(S / e)
    a1 = (L + (b + 0.5 * sigma * sigma) * T) / s
    c = -w * a1
    eps = 2.0 * b / (sigma * sigma)
    h = w * eps * s
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        # |eps| <= 1: exact rearrangement without the 0/0 at b = 0
        g_small = (ndtr(c + h) * _expm1_div(-L, eps)
                   + w * s * _dN_div(c, h)
                   - ndtr(c) * _expm1_div(0.5 * s * s, eps))
        # |eps| > 1: direct form, powers in log space
        g_big = (np.exp(-eps * L + log_ndtr(c + h)) - np.exp(b * T + log_ndtr(c))) / eps
    return np.where(np.abs(eps) <= 1.0, g_small, g_big)


def _bvn_cdf(a, b, rho):
    """Standard bivariate normal CDF M(a, b; rho) = P(X <= a, Y <= b), corr(X, Y) = rho.

    Port of Genz's BVNU (Genz 2004, Statistics and Computing 14, 251-260; the
    Drezner-Wesolowsky method with 20-point Gauss-Legendre), absolute accuracy
    about 1e-15. Infinite limits are exact; nan propagates.
    """
    a, b, rho = np.broadcast_arrays(_arr(a), _arr(b), _arr(rho))
    shape = a.shape
    h = -a.ravel()
    k = -b.ravel()
    r = rho.ravel()
    out = np.full(h.size, np.nan)
    with np.errstate(all="ignore"):
        # P(X > h, Y > k) with infinite limits
        inf = ~(np.isfinite(h) & np.isfinite(k))
        vi = np.where((h == np.inf) | (k == np.inf), 0.0,
                      np.where(h == -np.inf, np.where(k == -np.inf, 1.0, ndtr(-k)), ndtr(-h)))
        vi = np.where(np.isnan(h) | np.isnan(k), np.nan, vi)
        out[inf] = vi[inf]
        fin = ~inf & np.isfinite(r) & (np.abs(r) <= 1.0)
        m1 = fin & (np.abs(r) < 0.925)
        if m1.any():
            hh, kk, rr = h[m1], k[m1], r[m1]
            hk = hh * kk
            hs = 0.5 * (hh * hh + kk * kk)
            asr = 0.5 * np.arcsin(rr)
            sn = np.sin(asr[:, None] * _BV_X[None, :])
            val = np.exp((sn * hk[:, None] - hs[:, None]) / (1.0 - sn * sn)) @ _BV_W
            out[m1] = val * asr / _TWO_PI + ndtr(-hh) * ndtr(-kk)
        m2 = fin & ~(np.abs(r) < 0.925)
        if m2.any():
            hh, kk, rr = h[m2], k[m2], r[m2]
            kk = np.where(rr < 0, -kk, kk)
            hk = hh * kk
            bvn = np.zeros(hh.size)
            lt1 = np.abs(rr) < 1.0
            if lt1.any():
                h1, k1, hk1, r1 = hh[lt1], kk[lt1], hk[lt1], np.abs(rr[lt1])
                as_ = (1.0 - r1) * (1.0 + r1)
                a_ = np.sqrt(as_)
                bs = (h1 - k1) ** 2
                asr = -0.5 * (bs / as_ + hk1)
                c = (4.0 - hk1) / 8.0
                d = (12.0 - hk1) / 80.0
                t1 = np.where(asr > -100.0,
                              a_ * np.exp(asr) * (1.0 - c * (bs - as_) * (1.0 - d * bs) / 3.0 + c * d * as_ * as_),
                              0.0)
                bb = np.sqrt(bs)
                sp = math.sqrt(_TWO_PI) * ndtr(-bb / a_)
                t2 = np.where(hk1 > -100.0, np.exp(-0.5 * hk1) * sp * bb * (1.0 - c * bs * (1.0 - d * bs) / 3.0), 0.0)
                v = t1 - t2
                a2 = 0.5 * a_
                xs = (a2[:, None] * _BV_X[None, :]) ** 2
                asr2 = -0.5 * (bs[:, None] / xs + hk1[:, None])
                sp2 = 1.0 + c[:, None] * xs * (1.0 + 5.0 * d[:, None] * xs)
                rs = np.sqrt(1.0 - xs)
                ep = np.exp(-(hk1[:, None] / 2.0) * xs / (1.0 + rs) ** 2) / rs
                terms = np.where(asr2 > -100.0, np.exp(asr2) * (sp2 - ep), 0.0)
                bvn[lt1] = (a2 * (terms @ _BV_W) - v) / _TWO_PI
            pos = bvn + ndtr(-np.maximum(hh, kk))
            L = np.where(hh < 0, ndtr(kk) - ndtr(hh), ndtr(-hh) - ndtr(-kk))
            neg = np.where(hh >= kk, -bvn, L - bvn)
            out[m2] = np.where(rr > 0, pos, neg)
    return np.clip(out, 0.0, 1.0).reshape(shape)


# ----------------------------------------------------------------------------
# barriers and binaries
# ----------------------------------------------------------------------------
def barrier_price(S, K, H, T, r, sigma, *, kind, barrier, rebate=0.0, rebate_timing=None, q=0.0,
                  monitor_dt=None):
    """Price of the 8 standard single-barrier options (Merton / Reiner-Rubinstein), with rebate.

    Parameters
    ----------
    S, K, H : spot, strike and barrier level (> 0).
    T : time to expiry in years (> 0).
    r, q : continuously compounded annual rate and dividend yield (decimals);
        q defaults to 0.0 (b = r - q is the cost of carry).
    sigma : annual volatility as a decimal (> 0).
    kind : 'call' or 'put' (required).
    barrier : 'down-in', 'down-out', 'up-in' or 'up-out' (required).
    rebate : cash rebate (>= 0, default 0.0). A knock-in rebate is paid AT EXPIRY
        if the barrier was never hit. A knock-out rebate is paid when the barrier
        is hit, at the time given by rebate_timing.
    rebate_timing : 'hit' (paid at the hitting time, Haug's F term) or 'expiry'
        (paid at T). REQUIRED for a knock-out option with rebate != 0; for a
        knock-in only None or 'expiry' is accepted.
    monitor_dt : None (default) = continuous monitoring. A float dt (years between
        monitoring dates) applies the Broadie-Glasserman-Kou correction: the
        continuous formula is evaluated at H e^{+beta sigma sqrt(dt)} for an up
        barrier and H e^{-beta sigma sqrt(dt)} for a down barrier,
        beta = -zeta(1/2)/sqrt(2 pi) = 0.5826.

    Returns
    -------
    Option price. If S is already on the far side of H (S <= H for a down barrier,
    S >= H for an up barrier, judged on the unshifted H) the barrier counts as hit:
    a knock-in is then the vanilla BSM option and a knock-out is worth its rebate
    (rebate now for 'hit', rebate e^{-rT} for 'expiry').

    Formula
    -------
    Haug (2007) Ch. 4, "Standard Barrier Options", with phi = +1 call / -1 put, eta = +1 down / -1 up,
    mu = (b - sigma^2/2)/sigma^2, lambda = sqrt(mu^2 + 2r/sigma^2), s = sigma sqrt T:
    x1 = ln(S/K)/s + (1+mu)s, x2 = ln(S/H)/s + (1+mu)s,
    y1 = ln(H^2/(SK))/s + (1+mu)s, y2 = ln(H/S)/s + (1+mu)s, z = ln(H/S)/s + lambda s,
    A = phi S e^{-qT} N(phi x1) - phi K e^{-rT} N(phi x1 - phi s)   (B: x1 -> x2)
    C = phi S e^{-qT} (H/S)^{2(mu+1)} N(eta y1) - phi K e^{-rT} (H/S)^{2mu} N(eta y1 - eta s)  (D: y1 -> y2)
    E = R e^{-rT} [N(eta x2 - eta s) - (H/S)^{2mu} N(eta y2 - eta s)]   (R x Q(no hit), paid at T)
    F = R [(H/S)^{mu+lambda} N(eta z) + (H/S)^{mu-lambda} N(eta z - 2 eta lambda s)]  (paid at hit)
    K > H: DI call C+E, UI call A+E, DI put B-C+D+E, UI put A-B+D+E,
           DO call A-C+F, UO call F, DO put A-B+C-D+F, UO put B-D+F;
    K < H: DI call A-B+D+E, UI call B-C+D+E, DI put A+E, UI put C+E,
           DO call B-D+F, UO call A-B+C-D+F, DO put F, UO put A-C+F.
    A knock-out rebate paid at expiry replaces F by R e^{-rT} - E.
    In-out parity: in + out = vanilla (+ R e^{-rT} when both rebates are paid at expiry).
    'hit' needs mu^2 + 2r/sigma^2 >= 0 (always true for r >= 0).

    References
    ----------
    Merton (1973); Reiner and Rubinstein (1991), "Breaking down the barriers";
    Haug (2007) Ch. 4, "Standard Barrier Options" and its value table; Hull OFOD
    "Exotic Options" (barrier options); Broadie, Glasserman and Kou (1997), Mathematical Finance 7, 325-349.

    Example
    -------
    >>> barrier_price(100, 90, 95, 0.5, 0.08, 0.25, q=0.04, kind='call', barrier='down-out',
    ...               rebate=3, rebate_timing='hit')
    9.02456...         # Haug's standard-barrier table prints 9.0246
    """
    phi = _sign(kind)
    btype = _choice(barrier, _BARRIERS, "barrier")
    down = btype.startswith("down")
    knock_in = btype.endswith("in")
    eta = 1.0 if down else -1.0
    S, K, H, T, r, sigma, rebate, q = map(_arr, (S, K, H, T, r, sigma, rebate, q))
    _check_pos(S=S, K=K, H=H, T=T, sigma=sigma)
    _check_nonneg(rebate=rebate)
    timing = None if rebate_timing is None else _choice(rebate_timing, ("hit", "expiry"), "rebate_timing")
    if knock_in:
        if timing == "hit":
            raise ValueError("a knock-in rebate is paid at expiry (barrier never hit); use rebate_timing='expiry' or None")
        timing = "expiry"
    elif timing is None:
        if np.any(rebate != 0):
            raise ValueError("rebate_timing ('hit' or 'expiry') is required for a knock-out option with a rebate")
        timing = "expiry"  # the rebate is zero, so the timing is irrelevant
    if monitor_dt is None:
        Hs = H
    else:
        dt = _arr(monitor_dt)
        _check_pos(monitor_dt=dt)
        Hs = H * np.exp(-eta * _BGK_BETA * sigma * np.sqrt(dt))

    b = r - q
    s = sigma * np.sqrt(T)
    v2 = sigma * sigma
    mu = (b - 0.5 * v2) / v2
    Fd = S * np.exp(-q * T)
    er = np.exp(-r * T)
    Kd = K * er
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        lnHS = np.log(Hs / S)
        lnSK = np.log(S / K)
        x1 = lnSK / s + (1.0 + mu) * s
        x2 = -lnHS / s + (1.0 + mu) * s
        y1 = (2.0 * lnHS + lnSK) / s + (1.0 + mu) * s
        y2 = lnHS / s + (1.0 + mu) * s
        A = phi * (Fd * ndtr(phi * x1) - Kd * ndtr(phi * (x1 - s)))
        B = phi * (Fd * ndtr(phi * x2) - Kd * ndtr(phi * (x2 - s)))
        p1 = 2.0 * (mu + 1.0)
        p0 = 2.0 * mu
        C = phi * (Fd * _pow_N(lnHS, p1, eta * y1) - Kd * _pow_N(lnHS, p0, eta * (y1 - s)))
        D = phi * (Fd * _pow_N(lnHS, p1, eta * y2) - Kd * _pow_N(lnHS, p0, eta * (y2 - s)))
        hi, lo = _BARRIER_COEF[(btype, phi)]
        core_hi = hi[0] * A + hi[1] * B + hi[2] * C + hi[3] * D
        core_lo = lo[0] * A + lo[1] * B + lo[2] * C + lo[3] * D
        core = np.where(K > Hs, core_hi, core_lo)
        p_nohit = ndtr(eta * (x2 - s)) - _pow_N(lnHS, p0, eta * (y2 - s))
        if knock_in:
            reb = rebate * er * p_nohit
        elif timing == "hit":
            lam = np.sqrt(mu * mu + 2.0 * r / v2)
            z = lnHS / s + lam * s
            reb = rebate * (_pow_N(lnHS, mu + lam, eta * z) + _pow_N(lnHS, mu - lam, eta * (z - 2.0 * lam * s)))
        else:
            reb = rebate * er * (1.0 - p_nohit)
        reb = np.where(rebate == 0, 0.0, reb)
        alive = core + reb
    hit_now = (S <= H) if down else (S >= H)
    if knock_in:
        dead = bsm_price(S, K, T, r, sigma, kind=kind, q=q)
    else:
        dead = rebate if timing == "hit" else rebate * er
    return _out(np.where(hit_now, dead, alive))


def digital_price(S, K, T, r, sigma, *, kind, payoff, cash=1.0, q=0.0):
    """European binary option: cash-or-nothing or asset-or-nothing (Reiner-Rubinstein).

    Parameters
    ----------
    S : spot (>= 0).  K : strike (> 0).
    T : time to expiry in years (>= 0).
    r, q : continuously compounded annual rate and dividend yield (decimals);
        q defaults to 0.0.
    sigma : annual volatility as a decimal (>= 0).
    kind : 'call' (pays if S_T > K) or 'put' (pays if S_T < K) (required).
    payoff : 'cash' (cash-or-nothing: pays ``cash``) or 'asset' (asset-or-nothing:
        pays S_T) (required).
    cash : the cash amount of a cash-or-nothing option (default 1.0); ignored
        for payoff='asset'.

    Returns
    -------
    Option price. Where sigma sqrt(T) == 0 the deterministic limit is returned:
    the discounted payoff if the forward S e^{(r-q)T} is above (call) or below
    (put) K, and half of it when the forward equals K (the sigma -> 0 limit).

    Formula
    -------
    d1 = [ln(S/K) + (r - q + sigma^2/2) T]/(sigma sqrt T), d2 = d1 - sigma sqrt T;
    cash-or-nothing: call cash e^{-rT} N(d2), put cash e^{-rT} N(-d2);
    asset-or-nothing: call S e^{-qT} N(d1), put S e^{-qT} N(-d1).
    Asset call - K x (cash call with cash=1) = vanilla BSM call.

    References
    ----------
    Reiner and Rubinstein (1991), "Unscrambling the binary code"; Haug (2007)
    Ch. 4, "Cash-or-Nothing Options" and "Asset-or-Nothing Options"; Hull OFOD
    "Exotic Options" (binary options).

    Example
    -------
    >>> digital_price(100, 80, 0.75, 0.06, 0.35, q=0.06, kind='put', payoff='cash', cash=10)   # Haug cash-or-nothing
    2.6710...
    >>> digital_price(70, 65, 0.5, 0.07, 0.27, q=0.05, kind='put', payoff='asset')   # Haug asset-or-nothing
    20.2069...
    """
    w = _sign(kind)
    pay = _choice(payoff, ("cash", "asset"), "payoff")
    S, K, T, r, sigma, cash, q = map(_arr, (S, K, T, r, sigma, cash, q))
    _check_nonneg(S=S, T=T, sigma=sigma)
    _check_pos(K=K)
    s = sigma * np.sqrt(T)
    Fd = S * np.exp(-q * T)
    er = np.exp(-r * T)
    Kd = K * er
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        d1 = np.log(Fd / Kd) / s + 0.5 * s
        d2 = d1 - s
    ind = 0.5 * (1.0 + w * np.sign(Fd - Kd))  # sigma -> 0 limit of N(w d)
    if pay == "cash":
        p = np.where(s == 0, cash * er * ind, cash * er * ndtr(w * d2))
    else:
        p = np.where(s == 0, Fd * ind, Fd * ndtr(w * d1))
    return _out(p)


# ----------------------------------------------------------------------------
# lookbacks
# ----------------------------------------------------------------------------
def lookback_floating_price(S, s_ext, T, r, sigma, *, kind, q=0.0):
    """Floating-strike lookback option, continuous monitoring (Goldman-Sosin-Gatto).

    Parameters
    ----------
    S : spot (> 0).
    s_ext : running extreme observed so far (> 0): the MINIMUM for a call
        (s_ext <= S) or the MAXIMUM for a put (s_ext >= S). s_ext = S for a newly
        issued option.
    T : time to expiry in years (> 0).
    r, q : continuously compounded annual rate and dividend yield (decimals);
        q defaults to 0.0 (b = r - q; b = 0 is handled as the exact limit).
    sigma : annual volatility as a decimal (> 0).
    kind : 'call' (pays S_T - min) or 'put' (pays max - S_T) (required).

    Returns
    -------
    Option price.

    Formula
    -------
    Haug (2007) Ch. 4, "Floating-Strike Lookback Options"; e = s_ext, a1 = [ln(S/e) + (b + sigma^2/2)T]/(sigma sqrt T),
    a2 = a1 - sigma sqrt T:
    call = S e^{-qT} N(a1) - e e^{-rT} N(a2)
           + S e^{-rT} (sigma^2/2b) [(S/e)^{-2b/sigma^2} N(-a1 + 2b sqrt(T)/sigma) - e^{bT} N(-a1)]
    put  = e e^{-rT} N(-a2) - S e^{-qT} N(-a1)
           + S e^{-rT} (sigma^2/2b) [-(S/e)^{-2b/sigma^2} N(a1 - 2b sqrt(T)/sigma) + e^{bT} N(a1)].
    The sigma^2/(2b) term is evaluated in a rearranged form that is exact and
    continuous through b = 0.

    References
    ----------
    Goldman, Sosin and Gatto (1979), Journal of Finance 34, 1111-1127; Haug
    (2007) Ch. 4, "Floating-Strike Lookback Options"; Hull OFOD "Exotic Options"
    (lookback options, worked example).

    Example
    -------
    >>> lookback_floating_price(50, 50, 0.25, 0.10, 0.40, kind='put')    # Hull lookback example
    7.79...
    >>> lookback_floating_price(120, 100, 0.5, 0.10, 0.30, q=0.06, kind='call')   # Haug floating-strike example
    25.3533...
    """
    w = _sign(kind)
    S, e, T, r, sigma, q = map(_arr, (S, s_ext, T, r, sigma, q))
    _check_pos(S=S, s_ext=e, T=T, sigma=sigma)
    if w > 0 and np.any(e > S):
        raise ValueError("for a call s_ext is the running minimum, so s_ext <= S")
    if w < 0 and np.any(e < S):
        raise ValueError("for a put s_ext is the running maximum, so s_ext >= S")
    b = r - q
    s = sigma * np.sqrt(T)
    a1 = (np.log(S / e) + (b + 0.5 * sigma * sigma) * T) / s
    base = w * (S * np.exp(-q * T) * ndtr(w * a1) - e * np.exp(-r * T) * ndtr(w * (a1 - s)))
    return _out(base + w * S * np.exp(-r * T) * _lookback_G(S, e, T, b, sigma, w))


def lookback_fixed_price(S, K, s_ext, T, r, sigma, *, kind, q=0.0):
    """Fixed-strike lookback option, continuous monitoring (Conze-Viswanathan).

    Parameters
    ----------
    S : spot (> 0).  K : strike (> 0).
    s_ext : running extreme observed so far (> 0): the MAXIMUM for a call
        (s_ext >= S) or the MINIMUM for a put (s_ext <= S); s_ext = S for a newly
        issued option.
    T : time to expiry in years (> 0).
    r, q : continuously compounded annual rate and dividend yield (decimals);
        q defaults to 0.0 (b = r - q; b = 0 is handled as the exact limit).
    sigma : annual volatility as a decimal (> 0).
    kind : 'call' (pays max(M_T - K, 0), M_T the maximum) or 'put' (pays
        max(K - m_T, 0), m_T the minimum) (required).

    Returns
    -------
    Option price.

    Formula
    -------
    Haug (2007) Ch. 4, "Fixed-Strike Lookback Options". Call, with E = max(K, s_ext),
    d1 = [ln(S/E) + (b + sigma^2/2)T]/(sigma sqrt T), d2 = d1 - sigma sqrt T:
    c = e^{-rT}(E - K) + S e^{-qT} N(d1) - E e^{-rT} N(d2)
        + S e^{-rT} (sigma^2/2b) [-(S/E)^{-2b/sigma^2} N(d1 - 2b sqrt(T)/sigma) + e^{bT} N(d1)].
    Put, with E = min(K, s_ext):
    p = e^{-rT}(K - E) - S e^{-qT} N(-d1) + E e^{-rT} N(-d2)
        + S e^{-rT} (sigma^2/2b) [(S/E)^{-2b/sigma^2} N(-d1 + 2b sqrt(T)/sigma) - e^{bT} N(-d1)].
    Equivalently fixed call(K, s_ext) = floating put(s_ext -> E) + S e^{-qT} - K e^{-rT}.

    References
    ----------
    Conze and Viswanathan (1991), Journal of Finance 46, 1893-1907; Haug (2007)
    Ch. 4, "Fixed-Strike Lookback Options" and its value table; Hull OFOD
    "Exotic Options" (lookback options).

    Example
    -------
    >>> lookback_fixed_price(100, 100, 100, 0.5, 0.10, 0.30, kind='call')   # Haug fixed-strike table
    20.2296...
    """
    w = _sign(kind)
    S, K, e, T, r, sigma, q = map(_arr, (S, K, s_ext, T, r, sigma, q))
    _check_pos(S=S, K=K, s_ext=e, T=T, sigma=sigma)
    if w > 0 and np.any(e < S):
        raise ValueError("for a call s_ext is the running maximum, so s_ext >= S")
    if w < 0 and np.any(e > S):
        raise ValueError("for a put s_ext is the running minimum, so s_ext <= S")
    E = np.maximum(K, e) if w > 0 else np.minimum(K, e)
    b = r - q
    s = sigma * np.sqrt(T)
    er = np.exp(-r * T)
    d1 = (np.log(S / E) + (b + 0.5 * sigma * sigma) * T) / s
    base = w * er * (E - K) + w * (S * np.exp(-q * T) * ndtr(w * d1) - E * er * ndtr(w * (d1 - s)))
    return _out(base - w * S * er * _lookback_G(S, E, T, b, sigma, -w))


# ----------------------------------------------------------------------------
# Asian options
# ----------------------------------------------------------------------------
def _fixing_times(fixings, T):
    """Sorted 1-D array of fixing times in [0, T]; raises on bad input."""
    t = np.sort(np.asarray(fixings, dtype=float).ravel())
    if t.size == 0:
        raise ValueError("fixings must contain at least one fixing time")
    if not np.all(np.isfinite(t)) or t[0] < 0:
        raise ValueError("fixing times must be finite and >= 0 (years from now)")
    if np.any(t[-1] > T):
        raise ValueError("fixing times must be <= T")
    return t


def _moments_continuous(b, sigma, tau0, T):
    """M1 = E[A]/S, M2 = E[A^2]/S^2 for the continuous average A of S over [tau0, T].

    M1 = e^{b tau0} (e^{bL} - 1)/(bL), L = T - tau0;
    M2 = e^{(2b+sigma^2) tau0} 2 int_0^1 x e^{bLx} (e^{cLx} - 1)/(cLx) dx, c = b + sigma^2,
    which equals Haug's closed form 2e^{(2b+s2)T}/((b+s2)(2b+s2)L^2) + ... but has no
    singularity at b = 0, b = -sigma^2 or 2b = -sigma^2. The integrand is entire; the
    48-point Gauss-Legendre rule is exact to rounding for |bL|, |cL| up to ~50.
    """
    L = T - tau0
    v2 = sigma * sigma
    M1 = np.exp(b * tau0) * exprel(b * L)
    bL = (b * L)[..., None]
    cL = ((b + v2) * L)[..., None]
    m2 = 2.0 * np.sum(_M2_W * _M2_X * np.exp(bL * _M2_X) * exprel(cL * _M2_X), axis=-1)
    M2 = np.exp((2.0 * b + v2) * tau0) * m2
    return M1, M2


def _moments_discrete(b, sigma, t):
    """M1 = E[A]/S, M2 = E[A^2]/S^2 for A = (1/n) sum S_{t_i}, t sorted ascending.

    E[S_ti S_tj]/S^2 = e^{b(t_i + t_j) + sigma^2 min(t_i, t_j)}; the double sum is
    evaluated in O(n) with suffix sums.
    """
    n = t.size
    eb = np.exp(b[..., None] * t)
    ec = np.exp((b + sigma * sigma)[..., None] * t)
    M1 = eb.mean(axis=-1)
    suffix = np.cumsum(eb[..., ::-1], axis=-1)[..., ::-1] - eb  # sum_{j>i} e^{b t_j}
    M2 = (np.sum(ec * eb, axis=-1) + 2.0 * np.sum(ec * suffix, axis=-1)) / (n * n)
    return M1, M2


def _asian_lognormal(S, K, T, r, w, M1, M2, realized_avg, past_weight):
    """Two-moment lognormal price of max(w (pw Abar + (1-pw) A_f - K), 0) paid at T."""
    pw = _arr(past_weight)
    if np.any((pw < 0) | (pw > 1)):
        raise ValueError("past_weight must be in [0, 1]")
    if realized_avg is None:
        if np.any(pw > 0):
            raise ValueError("realized_avg is required when past_weight > 0")
        abar = _arr(0.0)
    else:
        abar = _arr(realized_avg)
        _check_nonneg(realized_avg=abar)
    F = S * M1
    with np.errstate(divide="ignore", invalid="ignore"):
        v = np.maximum(np.log(M2) - 2.0 * np.log(M1), 0.0)  # rounding can give -1e-17
    Fd = (1.0 - pw) * F           # (1-pw) E[A_f]
    Kn = K - pw * abar            # (1-pw) K*, K* the strike on the future average
    er = np.exp(-r * T)
    with np.errstate(divide="ignore", invalid="ignore"):
        opt = _black(Fd, Kn, np.sqrt(v), w)
    det = np.maximum(w * (Fd - Kn), 0.0)  # K* <= 0: the call is certainly exercised
    return _out(er * np.where(Kn > 0, opt, det))


def asian_geometric_price(S, K, T, r, sigma, *, kind, fixings, q=0.0):
    """Exact price of a geometric-average-price option (Kemna-Vorst; discrete geometric).

    Parameters
    ----------
    S, K : spot and strike (> 0).
    T : time to expiry = payment date in years (> 0).
    r, q : continuously compounded annual rate and dividend yield (decimals);
        q defaults to 0.0 (b = r - q).
    sigma : annual volatility as a decimal (> 0).
    kind : 'call' (pays max(G - K, 0)) or 'put' (pays max(K - G, 0)) (required).
    fixings : 'continuous' (G = exp of the time-average of ln S over [0, T],
        Kemna-Vorst) or an array of fixing times in years from now, each in
        [0, T] (a time-0 fixing is today's spot S), G = (prod S_{t_i})^{1/n}
        (required).

    Returns
    -------
    Option price.

    Formula
    -------
    ln G is normal with mean ln S + (b - sigma^2/2) tbar and variance
    V = sigma^2 (1/n^2) sum_i sum_j min(t_i, t_j) (tbar = mean fixing time);
    price = e^{-rT} [G_F N(d1) - K N(d2)] (call), G_F = S exp((b - sigma^2/2) tbar + V/2),
    d1 = [ln(G_F/K) + V/2]/sqrt(V), d2 = d1 - sqrt(V).
    Continuous (Kemna-Vorst): tbar = T/2, V = sigma^2 T/3, i.e. a BSM option with
    sigma_A = sigma/sqrt(3) and cost of carry b_A = (b - sigma^2/6)/2.

    References
    ----------
    Kemna and Vorst (1990), Journal of Banking and Finance 14, 113-129; Haug
    (2007) Ch. 4, "Geometric Average-Rate Options"; Hull OFOD "Exotic Options"
    (Asian options).

    Example
    -------
    >>> asian_geometric_price(80, 85, 0.25, 0.05, 0.20, q=-0.03, kind='put', fixings='continuous')  # Haug
    4.6922...
    """
    w = _sign(kind)
    S, K, T, r, sigma, q = map(_arr, (S, K, T, r, sigma, q))
    _check_pos(S=S, K=K, T=T, sigma=sigma)
    b = r - q
    if isinstance(fixings, str):
        _choice(fixings, ("continuous",), "fixings")
        tbar = T / 2.0
        vfac = T / 3.0
    else:
        t = _fixing_times(fixings, T)
        n = t.size
        tbar = t.mean()
        vfac = np.sum(t * (2.0 * (n - 1 - np.arange(n)) + 1.0)) / (n * n)  # (1/n^2) sum min(t_i, t_j)
    V = sigma * sigma * vfac
    GF = S * np.exp((b - 0.5 * sigma * sigma) * tbar + 0.5 * V)
    er = np.exp(-r * T)
    return _out(er * _black(GF, K, np.sqrt(V), w))


def asian_tw_price(S, K, T, r, sigma, *, kind, q=0.0, avg_start=0.0, realized_avg=None, past_weight=0.0):
    """Turnbull-Wakeman approximation for a continuous arithmetic-average-price option.

    Parameters
    ----------
    S, K : spot and strike (> 0).
    T : time to expiry = payment date in years (> 0).
    r, q : continuously compounded annual rate and dividend yield (decimals);
        q defaults to 0.0 (b = r - q).
    sigma : annual volatility as a decimal (> 0).
    kind : 'call' (pays max(A - K, 0)) or 'put' (pays max(K - A, 0)) (required).
    avg_start : years from now to the start of the (remaining) averaging period,
        0 <= avg_start <= T (default 0.0: averaging runs from now to T).
    realized_avg : average of the fixings already observed (seasoned option);
        required when past_weight > 0.
    past_weight : weight of the realised part in the final average, in [0, 1]
        (elapsed averaging time / total averaging time; default 0.0 = new option).
        The final average is A = past_weight * realized_avg + (1 - past_weight) * A_f,
        A_f the continuous average of S over [avg_start, T].

    Returns
    -------
    Option price. When the realised part alone guarantees exercise
    (K - past_weight * realized_avg <= 0) the call is the forward value
    e^{-rT} (E[A] - K) and the put is 0.

    Formula
    -------
    A_f is taken lognormal with its exact first two moments (tau = avg_start, L = T - tau):
    M1 = E[A_f]/S = (e^{bT} - e^{b tau})/(bL),
    M2 = E[A_f^2]/S^2 = 2e^{(2b+sigma^2)T}/((b+sigma^2)(2b+sigma^2)L^2)
         + 2e^{(2b+sigma^2)tau}/(bL^2) [1/(2b+sigma^2) - e^{bL}/(b+sigma^2)];
    sigma_A^2 T = ln(M2) - 2 ln(M1); K* = (K - w_p Abar)/(1 - w_p);
    price = (1 - w_p) e^{-rT} [S M1 N(d1) - K* N(d2)] (call),
    d1 = [ln(S M1/K*) + sigma_A^2 T/2]/(sigma_A sqrt T). M2 is evaluated as an
    equivalent integral that is regular at b = 0 and b = -sigma^2.

    References
    ----------
    Turnbull and Wakeman (1991), JFQA 26, 377-389; Haug (2007) Ch. 4,
    "Arithmetic Average-Rate Options" (Turnbull-Wakeman);
    Hull OFOD "Exotic Options" (Asian options: moment matching and the
    adjusted strike of a seasoned option).

    Example
    -------
    >>> asian_tw_price(50, 50, 1.0, 0.10, 0.40, kind='call')
    5.6167...          # Hull's example: M1 = 52.59, M2 = 2922.76, sigma = 23.54%, price 5.62
    """
    w = _sign(kind)
    S, K, T, r, sigma, q, tau0 = map(_arr, (S, K, T, r, sigma, q, avg_start))
    _check_pos(S=S, K=K, T=T, sigma=sigma)
    if np.any((tau0 < 0) | (tau0 > T)):
        raise ValueError("avg_start must be in [0, T]")
    M1, M2 = _moments_continuous(r - q, sigma, tau0, T)
    return _asian_lognormal(S, K, T, r, w, M1, M2, realized_avg, past_weight)


def asian_levy_price(S, K, T, r, sigma, *, kind, fixings, q=0.0, realized_avg=None, past_weight=0.0):
    """Levy two-moment lognormal approximation for an arithmetic-average-price option.

    Parameters
    ----------
    S, K : spot and strike (> 0).
    T : time to expiry = payment date in years (> 0).
    r, q : continuously compounded annual rate and dividend yield (decimals);
        q defaults to 0.0 (b = r - q).
    sigma : annual volatility as a decimal (> 0).
    kind : 'call' (pays max(A - K, 0)) or 'put' (pays max(K - A, 0)) (required).
    fixings : 'continuous' (the remaining average is the continuous average of S
        over [0, T]) or an array of the REMAINING fixing times in years from now,
        each in [0, T] (a time-0 fixing is today's spot S), equally weighted
        (required).
    realized_avg : average of the fixings already observed; required when
        past_weight > 0.
    past_weight : weight of the realised fixings in the final average, in [0, 1]
        (e.g. n_past / n_total; default 0.0 = new option). The final average is
        A = past_weight * realized_avg + (1 - past_weight) * A_f.

    Returns
    -------
    Option price. When K - past_weight * realized_avg <= 0 the call is the
    forward value e^{-rT} (E[A] - K) and the put is 0.

    Formula
    -------
    A_f is taken lognormal with its exact first two moments. Discrete fixings:
    M1 = (1/n) sum_i e^{b t_i}, M2 = (1/n^2) sum_i sum_j e^{b(t_i+t_j) + sigma^2 min(t_i, t_j)}
    (Hull's F_i, sigma_i moment formulas with a flat volatility); continuous: the
    Turnbull-Wakeman M1, M2 with avg_start = 0. Then V = ln(M2) - 2 ln(M1),
    X* = K - w_p Abar, S_E = (1 - w_p) S M1 e^{-rT},
    price = S_E N(d1) - X* e^{-rT} N(d2) (call), d1 = [ln(S_E/(X* e^{-rT})) + V/2]/sqrt(V).

    References
    ----------
    Levy (1992), Journal of International Money and Finance 11, 474-491; Haug
    (2007) Ch. 4, "Arithmetic Average-Rate Options" (Levy); Hull OFOD "Exotic Options" (Asian options, moment
    matching for continuous and discrete averages).

    Example
    -------
    >>> asian_levy_price(50, 50, 1.0, 0.10, 0.40, kind='call', fixings='continuous')
    5.6167...          # Hull's Asian example prints 5.62
    >>> asian_levy_price(50, 50, 1.0, 0.10, 0.40, kind='call', fixings=np.arange(1, 13) / 12)   # monthly
    5.9957...
    """
    w = _sign(kind)
    S, K, T, r, sigma, q = map(_arr, (S, K, T, r, sigma, q))
    _check_pos(S=S, K=K, T=T, sigma=sigma)
    b = r - q
    if isinstance(fixings, str):
        _choice(fixings, ("continuous",), "fixings")
        M1, M2 = _moments_continuous(b, sigma, np.zeros_like(T), T)
    else:
        t = _fixing_times(fixings, T)
        M1, M2 = _moments_discrete(b, sigma, t)
    return _asian_lognormal(S, K, T, r, w, M1, M2, realized_avg, past_weight)


# ----------------------------------------------------------------------------
# two-asset options
# ----------------------------------------------------------------------------
def margrabe_price(S1, S2, T, sigma1, sigma2, rho, *, q1=0.0, q2=0.0):
    """Margrabe option to exchange asset 2 for asset 1, payoff max(S1_T - S2_T, 0).

    Parameters
    ----------
    S1, S2 : current prices of the two assets (> 0), one unit of each.
    T : time to expiry in years (>= 0).
    sigma1, sigma2 : annual volatilities of the two assets (decimals, >= 0).
    rho : correlation of the two log-returns, in [-1, 1].
    q1, q2 : continuous dividend yields of the two assets (decimals, default 0.0).
        The risk-free rate cancels and is not an input.

    Returns
    -------
    Option price (in the currency of S1 and S2). Where sigma sqrt(T) == 0 the
    value is max(S1 e^{-q1 T} - S2 e^{-q2 T}, 0).

    Formula
    -------
    sigma = sqrt(sigma1^2 + sigma2^2 - 2 rho sigma1 sigma2),
    d1 = [ln(S1/S2) + (q2 - q1 + sigma^2/2) T]/(sigma sqrt T), d2 = d1 - sigma sqrt T,
    price = S1 e^{-q1 T} N(d1) - S2 e^{-q2 T} N(d2).
    Exchange parity: price(1 for 2) - price(2 for 1) = S1 e^{-q1 T} - S2 e^{-q2 T}.

    References
    ----------
    Margrabe (1978), Journal of Finance 33, 177-186; Haug (2007) Ch. 5,
    "Exchange-One-Asset-for-Another Options";
    Hull OFOD "Exotic Options" (options to exchange one asset for another).

    Example
    -------
    >>> margrabe_price(42, 40 * math.exp(-0.05), 0.5, 0.20, 0.0, 0.0)   # asset 2 = bond paying 40 at T
    4.7594...                                                          # = BSM call, Hull Ex. 15.6
    """
    S1, S2, T, s1, s2, rho, q1, q2 = map(_arr, (S1, S2, T, sigma1, sigma2, rho, q1, q2))
    _check_pos(S1=S1, S2=S2)
    _check_nonneg(T=T, sigma1=s1, sigma2=s2)
    if np.any((rho < -1) | (rho > 1)):
        raise ValueError("rho must be in [-1, 1]")
    sig = np.sqrt(np.maximum(s1 * s1 + s2 * s2 - 2.0 * rho * s1 * s2, 0.0))
    return _out(_black(S1 * np.exp(-q1 * T), S2 * np.exp(-q2 * T), sig * np.sqrt(T), 1.0))


def kirk_spread_price(F1, F2, K, T, sigma1, sigma2, rho, *, df, kind):
    """Kirk approximation for a European spread option on two forwards/futures.

    Parameters
    ----------
    F1, F2 : forward (or futures) prices of the two legs for the option expiry (> 0).
    K : strike of the spread (F2 + K must be > 0; K = 0 gives Margrabe exactly).
    T : time to expiry in years (>= 0).
    sigma1, sigma2 : annual lognormal volatilities of F1 and F2 (decimals, >= 0).
    rho : correlation of the two log-forwards, in [-1, 1].
    df : REQUIRED discount factor to the payment date (e^{-rT} if paid at T).
    kind : 'call' (pays max(F1 - F2 - K, 0)) or 'put' (pays max(K - F1 + F2, 0))
        (required).

    Returns
    -------
    Option price.

    Formula
    -------
    f = F2/(F2 + K), F = F1/(F2 + K),
    sigma = sqrt(sigma1^2 + (sigma2 f)^2 - 2 rho sigma1 sigma2 f),
    d1 = [ln F + sigma^2 T/2]/(sigma sqrt T), d2 = d1 - sigma sqrt T,
    call = df (F2 + K) [F N(d1) - N(d2)], put = df (F2 + K) [N(-d2) - F N(-d1)].
    Parity: call - put = df (F1 - F2 - K).

    References
    ----------
    Kirk (1995), "Correlation in the energy markets", in Managing Energy Price
    Risk; Haug (2007) Ch. 5, spread-option approximation (Kirk) and its example.

    Example
    -------
    >>> kirk_spread_price(28, 20, 7, 0.25, 0.29, 0.36, 0.42, df=math.exp(-0.05*0.25), kind='call')
    2.1670...
    """
    w = _sign(kind)
    F1, F2, K, T, s1, s2, rho, df = map(_arr, (F1, F2, K, T, sigma1, sigma2, rho, df))
    _check_pos(F1=F1, F2=F2)
    _check_nonneg(T=T, sigma1=s1, sigma2=s2, df=df)
    if np.any((rho < -1) | (rho > 1)):
        raise ValueError("rho must be in [-1, 1]")
    F2K = F2 + K
    if np.any(F2K <= 0):
        raise ValueError("F2 + K must be > 0 for the Kirk approximation")
    f = F2 / F2K
    sig = np.sqrt(np.maximum(s1 * s1 + (s2 * f) ** 2 - 2.0 * rho * s1 * s2 * f, 0.0))
    return _out(df * _black(F1, F2K, sig * np.sqrt(T), w))


# ----------------------------------------------------------------------------
# compound and chooser
# ----------------------------------------------------------------------------
def _critical_price(K1, K2, tau, r, sigma, q, underlying):
    """I* with bsm_price(I*, K2, tau, r, sigma, q, underlying) = K1 (0 when no root exists)."""
    u = _sign(underlying, "underlying")
    arrs = np.broadcast_arrays(K1, K2, tau, r, sigma, q)
    shape = arrs[0].shape
    flat = [a.ravel() for a in arrs]
    out = np.empty(flat[0].size)
    for i in range(out.size):
        k1, k2, t, rr, sg, qq = (float(a[i]) for a in flat)
        if not all(map(math.isfinite, (k1, k2, t, rr, sg, qq))):
            out[i] = np.nan
            continue
        if u < 0 and k1 >= k2 * math.exp(-rr * t):
            out[i] = 0.0  # the put never reaches K1: the compound region is empty or everything
            continue

        def f(x, k1=k1, k2=k2, t=t, rr=rr, sg=sg, qq=qq):
            return float(bsm_price(x, k2, t, rr, sg, kind=underlying, q=qq)) - k1

        lo, hi = k2 / math.e, k2 * math.e
        for _ in range(2000):  # expand geometrically until the root is bracketed
            flo, fhi = u * f(lo), u * f(hi)
            if flo < 0 < fhi:
                break
            if flo >= 0:
                lo /= 2.0
            if fhi <= 0:
                hi *= 2.0
        else:
            out[i] = np.nan
            continue
        out[i] = optimize.brentq(f, lo, hi, xtol=1e-300, rtol=4 * np.finfo(float).eps, maxiter=500)
    return out.reshape(shape)


def geske_compound_price(S, K1, K2, T1, T2, r, sigma, *, kind, underlying, q=0.0):
    """Compound option (option on a European option), Geske / Rubinstein closed form.

    Parameters
    ----------
    S : spot of the stock (> 0).
    K1 : strike of the compound option, paid at T1 for the underlying option (> 0).
    K2 : strike of the underlying European option (> 0).
    T1 : expiry of the compound option in years (> 0).
    T2 : expiry of the underlying option in years (T2 > T1).
    r, q : continuously compounded annual rate and dividend yield (decimals);
        q defaults to 0.0 (b = r - q).
    sigma : annual volatility of the stock (decimal, > 0).
    kind : 'call' or 'put', the compound option (required).
    underlying : 'call' or 'put', the underlying European option (required).

    Returns
    -------
    Price of the compound option.

    Formula
    -------
    I* solves BSM_underlying(I*, K2, T2 - T1) = K1 (I* = 0 when a put underlying
    can never be worth K1). rho = sqrt(T1/T2),
    y1 = [ln(S/I*) + (b + sigma^2/2)T1]/(sigma sqrt T1), y2 = y1 - sigma sqrt T1,
    z1 = [ln(S/K2) + (b + sigma^2/2)T2]/(sigma sqrt T2), z2 = z1 - sigma sqrt T2;
    call on call = S e^{-qT2} M(z1, y1; rho) - K2 e^{-rT2} M(z2, y2; rho) - K1 e^{-rT1} N(y2)
    put on call  = K2 e^{-rT2} M(z2, -y2; -rho) - S e^{-qT2} M(z1, -y1; -rho) + K1 e^{-rT1} N(-y2)
    call on put  = K2 e^{-rT2} M(-z2, -y2; rho) - S e^{-qT2} M(-z1, -y1; rho) - K1 e^{-rT1} N(-y2)
    put on put   = S e^{-qT2} M(-z1, y1; -rho) - K2 e^{-rT2} M(-z2, y2; -rho) + K1 e^{-rT1} N(y2).
    Parity: call on X - put on X = X(S, K2, T2) - K1 e^{-rT1}.
    M is evaluated with Genz's (2004) bivariate normal algorithm (~1e-15).

    References
    ----------
    Geske (1979), Journal of Financial Economics 7, 63-81; Rubinstein (1991),
    "Double trouble", Risk 5; Haug (2007) Ch. 4, "Options on Options"; Hull
    OFOD "Exotic Options" (compound options).

    Example
    -------
    >>> geske_compound_price(500, 50, 520, 0.25, 0.5, 0.08, 0.35, q=0.03, kind='put', underlying='call')
    21.19635...        # Haug's put-on-call example prints 21.1965
    """
    v = _sign(kind)
    u = _sign(underlying, "underlying")
    S, K1, K2, T1, T2, r, sigma, q = map(_arr, (S, K1, K2, T1, T2, r, sigma, q))
    _check_pos(S=S, K1=K1, K2=K2, T1=T1, T2=T2, sigma=sigma)
    if np.any(T2 <= T1):
        raise ValueError("T2 must be > T1")
    I = _critical_price(K1, K2, T2 - T1, r, sigma, q, underlying)
    b = r - q
    s1 = sigma * np.sqrt(T1)
    s2 = sigma * np.sqrt(T2)
    with np.errstate(divide="ignore"):
        y1 = (np.log(S / I) + (b + 0.5 * sigma * sigma) * T1) / s1
    y2 = y1 - s1
    z1 = (np.log(S / K2) + (b + 0.5 * sigma * sigma) * T2) / s2
    z2 = z1 - s2
    rho = np.sqrt(T1 / T2)
    sg = u * v
    inner = u * (S * np.exp(-q * T2) * _bvn_cdf(u * z1, sg * y1, v * rho)
                 - K2 * np.exp(-r * T2) * _bvn_cdf(u * z2, sg * y2, v * rho))
    return _out(v * (inner - K1 * np.exp(-r * T1) * ndtr(sg * y2)))


def chooser_price(S, K, t_choose, T, r, sigma, *, q=0.0):
    """Simple chooser option: at t_choose the holder picks a European call or put (same K, T).

    Parameters
    ----------
    S, K : spot and common strike (> 0).
    t_choose : time of the choice in years, 0 <= t_choose <= T (0 gives
        max(call, put) today, T gives the straddle call + put).
    T : common expiry of the call and the put in years (> 0).
    r, q : continuously compounded annual rate and dividend yield (decimals);
        q defaults to 0.0 (b = r - q).
    sigma : annual volatility as a decimal (> 0).

    Returns
    -------
    Option price.

    Formula
    -------
    d = [ln(S/K) + (b + sigma^2/2)T]/(sigma sqrt T), y = [ln(S/K) + bT + sigma^2 t/2]/(sigma sqrt t),
    price = S e^{-qT} N(d) - K e^{-rT} N(d - sigma sqrt T) - S e^{-qT} N(-y) + K e^{-rT} N(-y + sigma sqrt t),
    i.e. call(S, K, T) + e^{-q(T-t)} put(S, K e^{-b(T-t)}, t) (Hull).

    References
    ----------
    Rubinstein (1991), "Options for the undecided", Risk 4; Haug (2007) Ch. 4,
    "Simple Chooser Options"; Hull OFOD "Exotic Options" (chooser options).

    Example
    -------
    >>> chooser_price(50, 50, 0.25, 0.5, 0.08, 0.25)
    6.10707...         # Haug's simple-chooser example prints 6.1071
    """
    S, K, t, T, r, sigma, q = map(_arr, (S, K, t_choose, T, r, sigma, q))
    _check_pos(S=S, K=K, T=T, sigma=sigma)
    if np.any((t < 0) | (t > T)):
        raise ValueError("t_choose must be in [0, T]")
    Fd = S * np.exp(-q * T)
    Kd = K * np.exp(-r * T)
    return _out(_black(Fd, Kd, sigma * np.sqrt(T), 1.0) + _black(Fd, Kd, sigma * np.sqrt(t), -1.0))
