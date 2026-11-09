"""Supplements: the BAW exercise boundary, forward-start and cliquet options, mean-reverting jump moments,
the Curran discrete arithmetic-average (Asian) approximation.

Conventions for the whole module (the same as ``qfkit.bsm``): S is spot, K the
strike, T and every other time are in YEARS from today; r and q are
continuously compounded annual rates as decimals and b = r - q is the cost of
carry; sigma is the annual volatility as a decimal (0.20, not 20). ``kind`` is
ONE string, 'call' or 'put'. Options are on lognormal (Black-Scholes-Merton)
underlyings with constant r and q. Functions return a numpy float64 (a
``float`` subclass) when every input is a scalar and an ``np.ndarray``
otherwise; cliquet_price and ou_jump_moments return dicts.

Functions
---------
baw_critical_price   Barone-Adesi-Whaley critical price S* (early-exercise boundary), as used by baw_price
forward_start_price  forward-start option, strike alpha * S(t_start) (Rubinstein 1990)
cliquet_price        uncapped cliquet (ratchet) as a sum of forward-start options
ou_jump_moments      moments of an Ornstein-Uhlenbeck process with compound-Poisson normal jumps
asian_curran_price   Curran (1994) conditioning approximation, discrete arithmetic-average option

References: Barone-Adesi and Whaley (1987); Haug, The Complete Guide to Option
Pricing Formulas, 2nd ed. (2007), Ch. 3-4; Hull, Options, Futures, and Other
Derivatives (OFOD), "Exotic Options"; Rubinstein (1990); Barndorff-Nielsen and
Shephard (2001); Cont and Tankov (2004); Curran (1994).
"""

from __future__ import annotations

import math

import numpy as np
from scipy import optimize
from scipy.special import ndtr

from .bsm import bsm_d1d2, bsm_price
from .exotics import _fixing_times
from .numerics import _BAND_MSG, _exercise_regime, _m_over_k

__all__ = [
    "baw_critical_price",
    "forward_start_price",
    "cliquet_price",
    "ou_jump_moments",
    "asian_curran_price",
]

# 64-point Gauss-Legendre rule on [-1, 1] for the jump part of E[e^X]
_GL64_X, _GL64_W = np.polynomial.legendre.leggauss(64)


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


def _choice(value, name, allowed):
    if not isinstance(value, str) or value.strip().lower() not in allowed:
        raise ValueError(f"{name} must be one of {', '.join(repr(a) for a in allowed)}, got {value!r}")
    return value.strip().lower()


def _check_pos(**named):
    for name, v in named.items():
        if not np.all(np.isfinite(v)) or np.any(v <= 0):
            raise ValueError(f"{name} must be finite and > 0")


def _check_nonneg(**named):
    for name, v in named.items():
        if not np.all(np.isfinite(v)) or np.any(v < 0):
            raise ValueError(f"{name} must be finite and >= 0")


def _check_finite(**named):
    for name, v in named.items():
        if not np.all(np.isfinite(v)):
            raise ValueError(f"{name} must be finite")


def _scalar(x, name):
    a = np.asarray(x, dtype=float)
    if a.ndim != 0 or not math.isfinite(float(a)):
        raise ValueError(f"{name} must be a finite scalar")
    return float(a)


# ----------------------------------------------------------------------------
# Barone-Adesi-Whaley critical price
# ----------------------------------------------------------------------------
def _baw_critical_one(K, T, r, sigma, q, w):
    """The critical price that numerics.baw_price solves for (same equation, bracket and tolerances)."""
    regime = _exercise_regime(w, r, q)
    if regime == "european":        # early exercise is never optimal
        return math.inf if w > 0 else 0.0
    if regime == "band":
        raise ValueError(_BAND_MSG.format(cond="r < q (call)" if w > 0 else "q < r (put)"))
    b = r - q
    s2 = sigma * sigma
    n = 2.0 * b / s2
    mk = _m_over_k(r, sigma, T)
    root = math.sqrt((n - 1.0) ** 2 + 4.0 * mk)
    carry = math.exp((b - r) * T)
    kind = "call" if w > 0 else "put"

    def euro(x):
        return float(bsm_price(x, K, T, r, sigma, kind=kind, q=q))

    def nd1(x):
        d1, _ = bsm_d1d2(x, K, T, r, sigma, q=q)
        return float(ndtr(w * d1))

    if w > 0:
        q2 = 0.5 * (-(n - 1.0) + root)

        def f(x):
            return x - K - euro(x) - (1.0 - carry * nd1(x)) * x / q2

        lo, hi = K, 2.0 * K
        while f(hi) < 0:
            hi *= 2.0
            if hi > 1e12 * K:
                raise RuntimeError("BAW: no critical price found for the call")
        return optimize.brentq(f, lo, hi, xtol=1e-14 * K, rtol=4 * np.finfo(float).eps, maxiter=500)
    q1 = 0.5 * (-(n - 1.0) - root)

    def g(x):
        return K - x - euro(x) + (1.0 - carry * nd1(x)) * x / q1

    return optimize.brentq(g, 1e-12 * K, K, xtol=1e-14 * K, rtol=4 * np.finfo(float).eps, maxiter=500)


def baw_critical_price(K, T, r, sigma, *, kind, q=0.0):
    """Barone-Adesi-Whaley critical price S* of an American call or put: the early-exercise boundary.

    Parameters
    ----------
    K : strike (> 0).
    T : years to expiry (> 0).
    r, q : continuously compounded annual rate and dividend yield (q defaults
        to 0.0); cost of carry b = r - q (options on futures: q = r).
    sigma : annual volatility (decimal, > 0).
    kind : 'call' or 'put' (required).

    Returns
    -------
    The critical spot price, the same value that ``baw_price`` solves for
    internally: the call is exercised when S >= S*, the put when S <= S**, and
    there baw_price equals the intrinsic value. A call with q <= 0 and q <= r is
    never exercised early (returns inf); a put with r <= 0 and r <= q likewise
    (returns 0.0). r < 0 and q < 0 with r < q (call) or q < r (put) raises
    ValueError, as in baw_price (the exercise region is a band).

    Formula
    -------
    N = 2b/sigma^2, M = 2r/sigma^2, K(T) = 1 - e^{-rT}, d1 the BSM d1 at spot S*.
    Call: q2 = [-(N-1) + sqrt((N-1)^2 + 4M/K(T))]/2 and S* solves
          S* - K = c(S*) + [1 - e^{(b-r)T} N(d1(S*))] S*/q2.
    Put:  q1 = [-(N-1) - sqrt((N-1)^2 + 4M/K(T))]/2 and S** solves
          K - S** = p(S**) - [1 - e^{(b-r)T} N(-d1(S**))] S**/q1.
    These are value matching and smooth pasting of c + A2 (S/S*)^q2 against
    S - K. Solved by Brent's method to about 1e-14 K. As T -> infinity S*
    tends to the perpetual boundary K q2/(q2 - 1) (Merton 1973). Like the BAW
    price, S* is an approximation of the true boundary (about 2% away at
    T = 0.5-1); for very long maturities (decades) it is not monotone in T and
    overshoots the perpetual boundary before converging to it.

    References
    ----------
    Barone-Adesi and Whaley (1987), J. Finance 42; Haug (2007) Ch. 3 (with a seed
    value for the iteration); Merton (1973) (perpetual limit).

    Example
    -------
    >>> s = baw_critical_price(100, 0.5, 0.10, 0.25, kind='call', q=0.10)
    >>> baw_price(s, 100, 0.5, 0.10, 0.25, kind='call', q=0.10) - (s - 100)   # value matching
    0.0
    """
    w = _sign(kind)
    K, T, r, sigma, q = map(_arr, (K, T, r, sigma, q))
    _check_pos(K=K, T=T, sigma=sigma)
    _check_finite(r=r, q=q)
    shape = np.broadcast_shapes(K.shape, T.shape, r.shape, sigma.shape, q.shape)
    flat = [np.broadcast_to(a, shape).ravel() for a in (K, T, r, sigma, q)]
    vals = np.array([_baw_critical_one(*(float(a[i]) for a in flat), w) for i in range(flat[0].size)])
    return _out(vals.reshape(shape))


# ----------------------------------------------------------------------------
# forward start and cliquet
# ----------------------------------------------------------------------------
def forward_start_price(S, t_start, T, r, sigma, *, kind, alpha, q=0.0):
    """Forward-start option: a European call or put whose strike alpha * S(t_start) is set at t_start.

    Parameters
    ----------
    S : spot today (>= 0).
    t_start : years from today to the strike-setting date (>= 0).
    T : years from today to expiry (>= t_start).
    r, q : continuously compounded annual rate and dividend yield (q defaults to 0.0).
    sigma : annual volatility (decimal, >= 0).
    kind : 'call' or 'put' (required).
    alpha : REQUIRED. Strike as a fraction of the spot at t_start (1.0 = at the
        money, 1.1 = 10% out of the money for a call).

    Returns
    -------
    Price today. Calls pay max(S_T - alpha S_{t_start}, 0) at T; puts pay
    max(alpha S_{t_start} - S_T, 0).

    Formula
    -------
    By homogeneity of the BSM price in (S, K):
    V = S e^{-q t_start} * BSM(1, alpha, T - t_start, r, sigma, q), i.e.
    call = S e^{-q t} [e^{-q tau} N(d1) - alpha e^{-r tau} N(d2)], tau = T - t,
    d1 = [ln(1/alpha) + (r - q + sigma^2/2) tau]/(sigma sqrt(tau)), d2 = d1 - sigma sqrt(tau).
    t_start = 0 gives the BSM price with K = alpha S.

    References
    ----------
    Rubinstein (1990), "Pay now, choose later"; Haug (2007) Ch. 4 (forward start
    options; example S=60, alpha=1.1, t=0.25, T=1, r=8%, b=4%, sigma=30%: 4.4064);
    Hull OFOD "Exotic Options" (forward start options).

    Example
    -------
    >>> forward_start_price(60, 0.25, 1.0, 0.08, 0.30, kind='call', alpha=1.1, q=0.04)
    4.4064...
    """
    w = _sign(kind)
    S, t_start, T, r, sigma, q, alpha = map(_arr, (S, t_start, T, r, sigma, q, alpha))
    _check_nonneg(S=S, t_start=t_start, sigma=sigma)
    _check_pos(alpha=alpha)
    _check_finite(T=T, r=r, q=q)
    if np.any(T < t_start):
        raise ValueError("T (expiry) must be >= t_start (strike-setting date)")
    unit = bsm_price(1.0, alpha, T - t_start, r, sigma, kind="call" if w > 0 else "put", q=q)
    return _out(S * np.exp(-q * t_start) * unit)


def cliquet_price(S, reset_times, r, sigma, *, kind, payoff, payment, alpha=1.0, q=0.0, notional=1.0):
    """Cliquet (ratchet) option without a cap or floor, priced as a sum of forward-start options.

    Parameters
    ----------
    S : spot today (> 0, scalar).
    reset_times : increasing period end dates t_1 < ... < t_n in years from today.
        Period i runs from t_{i-1} to t_i with t_0 = 0 (today); its strike is set
        at t_{i-1}. A leading 0 is accepted and read as t_0.
    r, q : continuously compounded annual rate and dividend yield (scalars).
    sigma : annual volatility, a scalar or one value per period (length n).
    kind : 'call' or 'put' (required).
    payoff : REQUIRED. 'price': period i pays max(S_i - alpha S_{i-1}, 0)
        (call; put max(alpha S_{i-1} - S_i, 0)) in units of S, the textbook sum of
        forward-start options. 'return': period i pays
        notional * max(S_i / S_{i-1} - alpha, 0) (call), a return-based ratchet;
        alpha = 1 + k for a floor-strike k on the period return.
    payment : REQUIRED. 'period_end': each period's payoff is paid at t_i;
        'maturity': all payoffs are paid at t_n.
    alpha : strike as a fraction of the period-start spot (default 1.0, at the money).
    notional : scale of the 'return' payoff (default 1.0; ignored for 'price').

    Returns
    -------
    dict with keys 'price' (float, the sum) and 'legs' (ndarray of the n
    period values, in period order).

    Formula
    -------
    With tau_i = t_i - t_{i-1} and B_i = BSM(1, alpha, tau_i, r, sigma_i, q)
    (unit spot, strike alpha):
    'price':  leg_i = S e^{-q t_{i-1}} B_i        (forward-start option, Rubinstein 1990)
    'return': leg_i = notional e^{-r t_{i-1}} B_i  (S_i/S_{i-1} is independent of the past)
    'maturity' payment multiplies leg_i by e^{-r (t_n - t_i)}. price = sum_i leg_i.
    No cap or floor on the sum (globally or locally capped cliquets need Monte
    Carlo or a transform method).

    References
    ----------
    Hull OFOD "Exotic Options" (cliquet options as a series of forward-start
    options); Haug (2007) Ch. 4 (forward start options); Rubinstein (1990);
    Wilmott, Paul Wilmott on Quantitative Finance (2006) (cliquets).

    Example
    -------
    >>> cliquet_price(100, [1, 2, 3], 0.05, 0.2, kind='call', payoff='price', payment='period_end')['price']
    """
    w = _sign(kind)
    payoff = _choice(payoff, "payoff", ("price", "return"))
    payment = _choice(payment, "payment", ("period_end", "maturity"))
    S = _scalar(S, "S")
    r = _scalar(r, "r")
    q = _scalar(q, "q")
    alpha = _scalar(alpha, "alpha")
    notional = _scalar(notional, "notional")
    if S <= 0 or alpha <= 0:
        raise ValueError("S and alpha must be > 0")
    t = np.asarray(reset_times, dtype=float).ravel()
    if t.size and t[0] == 0.0:
        t = t[1:]
    if t.size == 0 or not np.all(np.isfinite(t)) or t[0] <= 0 or np.any(np.diff(t) <= 0):
        raise ValueError("reset_times must be increasing period end dates > 0")
    starts = np.concatenate(([0.0], t[:-1]))
    tau = t - starts
    sig = np.asarray(sigma, dtype=float)
    if sig.ndim == 0:
        sig = np.full(t.size, float(sig))
    elif sig.shape != (t.size,):
        raise ValueError(f"sigma must be a scalar or have one value per period ({t.size})")
    _check_nonneg(sigma=sig)
    unit = np.asarray(bsm_price(1.0, alpha, tau, r, sig, kind="call" if w > 0 else "put", q=q), dtype=float)
    if payoff == "price":
        legs = S * np.exp(-q * starts) * unit
    else:
        legs = notional * np.exp(-r * starts) * unit
    if payment == "maturity":
        legs = legs * np.exp(-r * (t[-1] - t))
    return {"price": float(np.sum(legs)), "legs": legs}


# ----------------------------------------------------------------------------
# mean-reverting jump process
# ----------------------------------------------------------------------------
def ou_jump_moments(x_prev, kappa, theta, sigma, lam, mu_j, sigma_j, *, dt, compensated):
    """Ornstein-Uhlenbeck (mean-reverting) jump-diffusion moments of X after dt, and E[e^X].

    Model: dX = kappa (theta - X) dt + sigma dW + J dN, N a Poisson process with
    intensity lam, jump sizes J ~ Normal(mu_j, sigma_j^2) iid, independent of W.
    compensated=True uses J dN - lam mu_j dt instead (the jumps then have zero
    mean effect and theta is the long-run mean).

    Parameters
    ----------
    x_prev : float or array_like
        Current value X_t (for a log-price model X = log S).
    kappa : mean-reversion speed per year (> 0).
    theta : long-run level of the drift.
    sigma : diffusion volatility per sqrt(year) (>= 0).
    lam : jump intensity per year (>= 0).
    mu_j, sigma_j : mean and standard deviation of the jump size (sigma_j >= 0;
        0 gives fixed jumps of size mu_j).
    dt : REQUIRED. Horizon in years (> 0); np.inf gives the stationary distribution.
    compensated : REQUIRED. bool, see Model.

    Returns
    -------
    dict with keys 'mean', 'var', 'skew', 'ex_kurt' (excess kurtosis) of
    X_{t+dt} given X_t, and 'mean_exp' = E[exp(X_{t+dt})] (the expected price
    when X = log S). Floats or arrays (broadcast).

    Formula
    -------
    e_n = e^{-n kappa dt}; jump raw moments m1 = mu_j, m2 = mu_j^2 + sigma_j^2,
    m3 = mu_j^3 + 3 mu_j sigma_j^2, m4 = mu_j^4 + 6 mu_j^2 sigma_j^2 + 3 sigma_j^4.
    mean    = theta + (x - theta) e_1 + (lam m1 / kappa)(1 - e_1)   [compensated: no last term]
    var     = (sigma^2 + lam m2)(1 - e_2) / (2 kappa)
    k3      = lam m3 (1 - e_3) / (3 kappa),  k4 = lam m4 (1 - e_4) / (4 kappa)
    skew    = k3 / var^{3/2},  ex_kurt = k4 / var^2
    (cumulants of a Levy-driven OU process: k_n(X) = k_n(L_1)(1 - e_n)/(n kappa), n >= 2).
    log mean_exp = theta + (x - theta) e_1 + sigma^2 (1 - e_2)/(4 kappa)
                   + (lam/kappa) int_{e_1}^{1} [exp(y mu_j + y^2 sigma_j^2 / 2) - 1] / y dy
                   [compensated: - lam mu_j (1 - e_1)/kappa]
    (64-point Gauss-Legendre rule; the integrand is smooth on [0, 1]).

    References
    ----------
    Barndorff-Nielsen and Shephard (2001), JRSS B 63 (cumulants of OU processes
    driven by Levy processes); Cont and Tankov (2004) (Levy-driven OU processes);
    Clewlow and Strickland (2000), Energy Derivatives (mean-reverting jump diffusion);
    Cartea and Figueroa (2005).

    Example
    -------
    >>> m = ou_jump_moments(0.0, 2.0, 0.0, 0.3, 5.0, 0.1, 0.05, dt=1.0, compensated=False)
    >>> m['var']     # (0.09 + 5 * 0.0125)(1 - e^-4)/4
    0.0374...
    """
    if not isinstance(compensated, (bool, np.bool_)):
        raise TypeError("compensated must be True or False")
    x, kappa, theta, sigma, lam, mu_j, sigma_j, dt = map(
        _arr, (x_prev, kappa, theta, sigma, lam, mu_j, sigma_j, dt))
    _check_pos(kappa=kappa)
    _check_nonneg(sigma=sigma, lam=lam, sigma_j=sigma_j)
    _check_finite(x_prev=x, theta=theta, mu_j=mu_j)
    if np.any(np.isnan(dt)) or np.any(dt <= 0):
        raise ValueError("dt must be > 0 (np.inf for the stationary distribution)")
    one_e1 = -np.expm1(-kappa * dt)
    one_e2 = -np.expm1(-2.0 * kappa * dt)
    one_e3 = -np.expm1(-3.0 * kappa * dt)
    one_e4 = -np.expm1(-4.0 * kappa * dt)
    e1 = 1.0 - one_e1
    s2j = sigma_j * sigma_j
    m1 = mu_j
    m2 = mu_j * mu_j + s2j
    m3 = mu_j ** 3 + 3.0 * mu_j * s2j
    m4 = mu_j ** 4 + 6.0 * mu_j * mu_j * s2j + 3.0 * s2j * s2j
    base = theta + (x - theta) * e1
    drift_jump = 0.0 if compensated else lam * m1 / kappa * one_e1
    mean = base + drift_jump
    var = (sigma * sigma + lam * m2) * one_e2 / (2.0 * kappa)
    k3 = lam * m3 * one_e3 / (3.0 * kappa)
    k4 = lam * m4 * one_e4 / (4.0 * kappa)
    with np.errstate(divide="ignore", invalid="ignore"):
        skew = np.where(var > 0, k3 / np.where(var > 0, var, 1.0) ** 1.5, np.nan)
        ex_kurt = np.where(var > 0, k4 / np.where(var > 0, var, 1.0) ** 2, np.nan)
    # jump part of log E[e^X]: (lam/kappa) int_{e1}^{1} expm1(y mu + y^2 s^2/2)/y dy
    lo = np.asarray(e1, dtype=float)[..., None]
    y = lo + (1.0 - lo) * 0.5 * (_GL64_X + 1.0)
    wts = (1.0 - lo) * 0.5 * _GL64_W
    mu_b = np.asarray(mu_j, dtype=float)[..., None]
    s2_b = np.asarray(s2j, dtype=float)[..., None]
    integral = np.sum(wts * np.expm1(y * mu_b + 0.5 * y * y * s2_b) / y, axis=-1)
    log_mean_exp = (base + sigma * sigma * one_e2 / (4.0 * kappa) + lam / kappa * integral
                    - (lam * m1 / kappa * one_e1 if compensated else 0.0))
    return {"mean": _out(mean), "var": _out(var), "skew": _out(skew), "ex_kurt": _out(ex_kurt),
            "mean_exp": _out(np.exp(log_mean_exp))}


# ----------------------------------------------------------------------------
# Curran's discrete arithmetic-average approximation
# ----------------------------------------------------------------------------
def asian_curran_price(S, K, T, r, sigma, *, kind, fixings, q=0.0):
    """Curran (1994) conditioning approximation for a discrete arithmetic-average-price option.

    Parameters
    ----------
    S, K : spot and strike (> 0).
    T : time to expiry = payment date in years (> 0).
    r, q : continuously compounded annual rate and dividend yield (decimals);
        q defaults to 0.0 (b = r - q).
    sigma : annual volatility as a decimal (> 0).
    kind : 'call' (pays max(A - K, 0)) or 'put' (pays max(K - A, 0)) (required).
    fixings : REQUIRED. Array of the n fixing times in years from now, each in
        [0, T] (a time-0 fixing is today's spot S), equally weighted:
        A = (1/n) sum_i S_{t_i}. E.g. np.arange(1, n + 1) * T / n for n equally
        spaced fixings ending at T (as asian_levy_price / asian_geometric_price).

    Returns
    -------
    Option price (numpy float64, or ndarray when S, K, T, r, sigma or q is an
    array; fixings are shared). The put is the call minus the forward of the
    average, e^{-rT} (E[A] - K) (put-call parity for average-price options).

    Formula
    -------
    G = (prod S_{t_i})^{1/n} is lognormal: ln G ~ N(mu, sx^2) with
    mu = ln S + (b - sigma^2/2) tbar and sx^2 = sigma^2 (1/n^2) sum_i sum_j min(t_i, t_j);
    mu_i = ln S + (b - sigma^2/2) t_i, s_i^2 = sigma^2 t_i (moments of ln S_{t_i}),
    s_xi = Cov(ln S_{t_i}, ln G) = sigma^2 (1/n) sum_j min(t_i, t_j).
    Conditioning on G, E[S_{t_i} | ln G = g] = exp(mu_i + s_xi (g - mu)/sx^2 + (s_i^2 - s_xi^2/sx^2)/2).
    Curran's exercise point: K^ = 2K - (1/n) sum_i E[S_{t_i} | G = K]  (a first-order
    expansion of E[A | G = K^] = K), and
    call = e^{-rT} [ (1/n) sum_i S e^{b t_i} N(d + s_xi/sx) - K N(d) ],  d = (mu - ln K^)/sx,
    which is exactly e^{-rT} E[(A - K) 1{G > K^}]. It is therefore a lower bound of
    the true call and lies below the Rogers-Shi bound E[(E[A|G] - K)^+], within
    about 1e-6 S of it; it exceeds the geometric-average call. K^ <= 0 gives the
    forward e^{-rT} (E[A] - K). One fixing at T gives the BSM price exactly.

    References
    ----------
    Curran (1994), "Valuing Asian and portfolio options by conditioning on the
    geometric mean price", Management Science 40, 1705-1711; Haug (2007) Ch. 4,
    "Arithmetic Average Rate Options" (Curran's approximation); Rogers and Shi
    (1995), J. Applied Probability 32 (lower bound).

    Example
    -------
    >>> asian_curran_price(100, 100, 1.0, 0.05, 0.2, kind='call', fixings=np.arange(1, 13) / 12, q=0.02)
    5.5197...   # Levy 5.5346, geometric 5.3277, control-variate Monte Carlo 5.5201 (SE 0.0001)
    """
    w = _sign(kind)
    S, K, T, r, sigma, q = map(_arr, (S, K, T, r, sigma, q))
    _check_pos(S=S, K=K, T=T, sigma=sigma)
    _check_finite(r=r, q=q)
    if isinstance(fixings, str):
        raise ValueError("fixings must be an array of fixing times (Curran's method is for discrete averages; "
                         "use asian_levy_price or asian_tw_price for a continuous average)")
    t = _fixing_times(fixings, T)
    n = t.size
    b = r - q
    er = np.exp(-r * T)
    # m_i = (1/n) sum_j min(t_i, t_j) for sorted t: (sum_{j<=i} t_j + (n - 1 - i) t_i) / n
    m = (np.cumsum(t) + (n - 1 - np.arange(n)) * t) / n
    mbar = float(m.mean())                       # (1/n^2) sum_i sum_j min(t_i, t_j)
    fwd = S[..., None] * np.exp(b[..., None] * t)  # E[S_{t_i}] = S e^{b t_i}
    mean_a = fwd.mean(axis=-1)                   # E[A]
    if mbar == 0.0:                              # every fixing today: A = S
        call = er * np.maximum(S - K, 0.0)
    else:
        s2 = (sigma * sigma)[..., None]
        sx = sigma * math.sqrt(mbar)
        mu = np.log(S) + (b - 0.5 * sigma * sigma) * t.mean()
        mu_i = np.log(S)[..., None] + (b - 0.5 * sigma * sigma)[..., None] * t
        beta = m / mbar                          # s_xi / sx^2
        cond = np.exp(mu_i + beta * (np.log(K) - mu)[..., None] + 0.5 * s2 * (t - m * m / mbar))
        k_hat = 2.0 * K - cond.mean(axis=-1)
        with np.errstate(divide="ignore", invalid="ignore"):
            d = np.where(k_hat > 0, (mu - np.log(np.where(k_hat > 0, k_hat, 1.0))) / sx, np.inf)
        s_xi_over_sx = (sigma * sigma)[..., None] * m / sx[..., None]
        call = er * (np.mean(fwd * ndtr(d[..., None] + s_xi_over_sx), axis=-1) - K * ndtr(d))
    if w > 0:
        return _out(call)
    return _out(call - er * (mean_a - K))
