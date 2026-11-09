"""Credit risk: Merton structural model, hazard rates, CDS valuation and bootstrap, rating migration.

Conventions for the whole module: times are in YEARS; r is a continuously
compounded annual decimal; volatilities are annual decimals. Hazard rates are
per year and PIECEWISE CONSTANT: ``hazard_times`` are the RIGHT ends of the
intervals (hazards[k] applies on (hazard_times[k-1], hazard_times[k]], with
hazard_times[-1] = 0), and the last hazard stays flat after the last knot.
CDS spreads are annual decimals (0.0124 = 124 bp) paid on the premium dates;
``recovery`` R is a fraction of notional (0.40). A ``curve`` argument is a
qfkit ``ZeroCurve`` or any callable t -> P(0, t) (risk-free discounting).
Transition matrices are row-stochastic numpy arrays in DECIMALS (not
percent): rows are the rating now, columns the rating at the end of the
period, and the last state is usually the absorbing default state.
Vectorised functions return a numpy float64 when every input is a scalar and
an ``np.ndarray`` otherwise.

Functions
---------
merton_structural         Merton (1974) equity, debt, PD, expected loss, recovery, spread
merton_implied_asset      asset value and volatility implied by equity value and volatility
hazard_from_spread        credit-triangle hazard rate s / (1 - R)
survival_probability      survival exp(-int lambda) for a piecewise-constant hazard
cds_value                 CDS protection / premium legs, risky annuity, par spread, value
bootstrap_hazard          piecewise-constant hazards that reprice par CDS spreads exactly
migration_n_period        n-period transition matrix P^n
migration_generator       generator matrix Q of a one-period matrix (log, IRW, JLT)
migration_from_generator  transition matrix expm(Q t) for any horizon t
"""

from __future__ import annotations

import numpy as np
from scipy import linalg, optimize
from scipy.special import ndtr

from .curves import ZeroCurve, _coupon_times, _parse_freq
from .rates import _discount

__all__ = [
    "merton_structural",
    "merton_implied_asset",
    "hazard_from_spread",
    "survival_probability",
    "cds_value",
    "bootstrap_hazard",
    "migration_n_period",
    "migration_generator",
    "migration_from_generator",
]

_TIMINGS = ("mid_period", "continuous")
_GRID_DT = 1.0 / 365.0  # max sub-step (years) of the 'continuous' CDS integration grid
_STOCH_TOL = 1e-8  # tolerance of the row-stochastic and generator checks


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _out(x):
    """0-d result -> numpy float64 scalar; otherwise the ndarray itself."""
    x = np.asarray(x, dtype=float)
    return x[()] if x.ndim == 0 else x


def _arr(x):
    return np.asarray(x, dtype=float)


def _check(cond, msg):
    if not np.all(cond):
        raise ValueError(msg)


def _hazard_curve(hazard_times, hazards):
    """Validated knots (strictly increasing, > 0) and hazards (>= 0), both 1-D float arrays."""
    knots = np.array(hazard_times, dtype=float, ndmin=1).ravel()
    lam = np.array(hazards, dtype=float, ndmin=1).ravel()
    if knots.size == 0 or knots.shape != lam.shape:
        raise ValueError("hazard_times and hazards must be non-empty and of equal length")
    if not (np.all(np.isfinite(knots)) and np.all(np.isfinite(lam))):
        raise ValueError("hazard_times and hazards must be finite")
    if knots[0] <= 0 or np.any(np.diff(knots) <= 0):
        raise ValueError("hazard_times must be > 0 and strictly increasing")
    if np.any(lam < 0):
        raise ValueError("hazards must be >= 0 (per year)")
    return knots, lam


def _cum_hazard(t, knots, lam):
    """Integral of the piecewise-constant hazard from 0 to t (t >= 0, any shape)."""
    t = _arr(t)
    starts = np.concatenate(([0.0], knots[:-1]))
    cum = np.concatenate(([0.0], np.cumsum(lam * (knots - starts))))  # H at 0, t_1, ..., t_n
    k = np.minimum(np.searchsorted(knots, t, side="left"), knots.size - 1)
    return cum[k] + lam[k] * (t - starts[k])


def _hazard_at(t, knots, lam):
    """Hazard in force on the interval containing t (right-continuous knots, flat after the last)."""
    k = np.minimum(np.searchsorted(knots, _arr(t), side="left"), knots.size - 1)
    return lam[k]


def _phi1(x):
    """(1 - e^{-x}) / x, 1 at x = 0."""
    x = _arr(x)
    nz = x != 0
    xx = np.where(nz, x, 1.0)
    return np.where(nz, -np.expm1(-xx) / xx, 1.0)


def _phi2(x):
    """(1 - e^{-x}(1 + x)) / x^2, 1/2 at x = 0 (series for |x| < 1e-3)."""
    x = _arr(x)
    small = np.abs(x) < 1e-3
    xs = np.where(small, x, 0.0)
    series = 0.5 - xs / 3.0 + xs * xs / 8.0 - xs**3 / 30.0
    xl = np.where(small, 1.0, x)
    with np.errstate(over="ignore", invalid="ignore"):
        direct = (-np.expm1(-xl) - xl * np.exp(-xl)) / (xl * xl)
    return np.where(small, series, direct)


def _parse_timing(default_timing):
    if not isinstance(default_timing, str) or default_timing.strip().lower() not in _TIMINGS:
        raise ValueError(f"default_timing must be one of {_TIMINGS}, got {default_timing!r}")
    return default_timing.strip().lower()


def _pay_schedule(pay_times):
    pay = np.array(pay_times, dtype=float, ndmin=1).ravel()
    if pay.size == 0 or not np.all(np.isfinite(pay)):
        raise ValueError("pay_times must be a non-empty array of finite times (years)")
    if pay[0] <= 0 or np.any(np.diff(pay) <= 0):
        raise ValueError("pay_times must be > 0 and strictly increasing")
    return pay


def _cds_legs(knots, lam, pay, recovery, curve, timing, accrual_on_default):
    """(annuity, accrual_annuity, protection) per unit notional (and per unit spread for the annuities)."""
    edges = np.concatenate(([0.0], pay))
    tau = np.diff(edges)
    Q_pay = np.exp(-_cum_hazard(pay, knots, lam))
    annuity = float(np.dot(tau, _discount(curve, pay) * Q_pay))
    lgd = 1.0 - recovery
    if timing == "mid_period":
        Q_edges = np.concatenate(([1.0], Q_pay))
        dQ = Q_edges[:-1] - Q_edges[1:]
        P_mid = _discount(curve, 0.5 * (edges[:-1] + edges[1:]))
        protection = lgd * float(np.dot(dQ, P_mid))
        accrual = float(np.dot(dQ * 0.5 * tau, P_mid)) if accrual_on_default else 0.0
        return annuity, accrual, protection
    # 'continuous': exact integrals on sub-intervals where the hazard and the (log-linear)
    # forward rate are constant; the grid holds every premium date, every hazard knot and
    # every ZeroCurve knot inside (0, t_n), and a uniform sub-grid of step <= 1/365 year.
    pieces = []
    for a, b in zip(edges[:-1], edges[1:]):
        n = max(int(np.ceil((b - a) / _GRID_DT - 1e-9)), 1)
        pieces.append(np.linspace(a, b, n + 1)[:-1])
    grid = np.concatenate(pieces + [pay[-1:]])
    inner = knots
    if isinstance(curve, ZeroCurve):
        inner = np.concatenate((inner, curve.t))
    inner = np.unique(inner[(inner > 0) & (inner < pay[-1])])
    if inner.size:
        near = np.abs(inner[:, None] - grid[None, :]).min(axis=1) <= 1e-10
        grid = np.sort(np.concatenate((grid, inner[~near])))
    g0, g1 = grid[:-1], grid[1:]
    h = g1 - g0
    P = _discount(curve, grid)
    Q = np.exp(-_cum_hazard(grid, knots, lam))
    lam_j = _hazard_at(0.5 * (g0 + g1), knots, lam)
    fwd = np.log(P[:-1] / P[1:]) / h
    x = (lam_j + fwd) * h
    w = P[:-1] * Q[:-1] * lam_j
    protection = lgd * float(np.sum(w * h * _phi1(x)))
    if accrual_on_default:
        period_start = edges[np.searchsorted(pay, g0, side="right")]
        d = g0 - period_start
        accrual = float(np.sum(w * (d * h * _phi1(x) + h * h * _phi2(x))))
    else:
        accrual = 0.0
    return annuity, accrual, protection


# ----------------------------------------------------------------------------
# Merton structural model
# ----------------------------------------------------------------------------
def merton_structural(V, D, T, r, sigma_V, *, mu=None):
    """Merton (1974) structural credit model: equity as a call on the firm's assets, debt, PD and spread.

    Parameters
    ----------
    V : market value of the firm's assets today (> 0).
    D : face value of zero-coupon debt due at T (> 0), in the units of V.
    T : debt maturity in years (> 0).
    r : continuously compounded risk-free rate (decimal).
    sigma_V : annual asset volatility (decimal, > 0).
    mu : optional real-world (physical) asset drift, continuously compounded
        annual decimal. When given, 'dd' and 'pd_physical' are added.

    Returns
    -------
    dict (numpy float64 or arrays, broadcast over the inputs):
    'equity'  E = V N(d1) - D e^{-rT} N(d2)
    'debt'    market value of debt V - E = V N(-d1) + D e^{-rT} N(d2)
    'd1', 'd2' d1 = [ln(V/D) + (r + sigma_V^2/2) T] / (sigma_V sqrt T), d2 = d1 - sigma_V sqrt T
    'pd_rn'   risk-neutral default probability N(-d2)
    'expected_loss' risk-neutral expected loss as a fraction of the riskless
              value D e^{-rT}: 1 - debt / (D e^{-rT}) = N(-d2) - N(-d1) V / (D e^{-rT})
    'recovery' risk-neutral expected recovery rate given default,
              E[V_T | V_T < D] / D = N(-d1) V e^{rT} / (N(-d2) D) (1.0 in the limit pd_rn -> 0);
              expected_loss = pd_rn (1 - recovery)
    'spread'  continuously compounded credit spread -ln(debt / (D e^{-rT})) / T
    'sigma_E' equity volatility N(d1) sigma_V V / E
    with mu: 'dd' distance to default [ln(V/D) + (mu - sigma_V^2/2) T] / (sigma_V sqrt T)
    and 'pd_physical' N(-dd).

    Formula
    -------
    As listed (Merton 1974; Hull OFOD "Credit Risk", Merton's model).

    Example
    -------
    >>> m = merton_structural(12.3954, 10.0, 1.0, 0.05, 0.21230)   # Hull OFOD Credit Risk example
    >>> round(float(m['equity']), 3), round(float(m['pd_rn']), 3)
    (3.0, 0.127)
    """
    V, D, T, r, s = map(_arr, (V, D, T, r, sigma_V))
    for name, v in (("V", V), ("D", D), ("T", T), ("sigma_V", s)):
        _check(np.isfinite(v) & (v > 0), f"{name} must be finite and > 0")
    sq = s * np.sqrt(T)
    lnVD = np.log(V / D)
    d1 = (lnVD + (r + 0.5 * s * s) * T) / sq
    d2 = d1 - sq
    Dd = D * np.exp(-r * T)
    equity = V * ndtr(d1) - Dd * ndtr(d2)
    debt = V * ndtr(-d1) + Dd * ndtr(d2)
    pd = ndtr(-d2)
    recovered = V / Dd * ndtr(-d1)  # E[V_T 1{V_T < D}] / D under Q, as a fraction of D
    el = pd - recovered
    with np.errstate(divide="ignore", invalid="ignore"):
        rec = np.where(pd > 0, recovered / pd, 1.0)
        sigma_E = ndtr(d1) * s * V / equity
    out = {
        "equity": _out(equity),
        "debt": _out(debt),
        "d1": _out(d1),
        "d2": _out(d2),
        "pd_rn": _out(pd),
        "expected_loss": _out(el),
        "recovery": _out(rec),
        "spread": _out(-np.log1p(-el) / T),
        "sigma_E": _out(sigma_E),
    }
    if mu is not None:
        dd = (lnVD + (_arr(mu) - 0.5 * s * s) * T) / sq
        out["dd"] = _out(dd)
        out["pd_physical"] = _out(ndtr(-dd))
    return out


def _implied_asset_one(E, sE, D, T, r):
    Dd = D * np.exp(-r * T)
    sq = np.sqrt(T)

    def d12(V, s):
        d1 = (np.log(V / D) + (r + 0.5 * s * s) * T) / (s * sq)
        return d1, d1 - s * sq

    def V_of(s):
        def f(V):
            d1, d2 = d12(V, s)
            return V * ndtr(d1) - Dd * ndtr(d2) - E

        lo, hi = E, E + Dd  # V - D e^{-rT} <= call <= V brackets the root
        if f(hi) <= 0:
            return hi
        return optimize.brentq(f, lo, hi, xtol=1e-14 * hi, rtol=1e-15, maxiter=500)

    def g(s):
        V = V_of(s)
        return ndtr(d12(V, s)[0]) * s * V - sE * E

    s_lo = 1e-6 * sE * E / (E + Dd)
    s_hi = sE  # sigma_V V N(d1) = sigma_E E and V N(d1) >= E imply sigma_V <= sigma_E
    if g(s_hi) <= 0:  # N(d2) ~ 0: the debt is worthless and sigma_V = sigma_E to rounding
        return V_of(s_hi), s_hi
    s = optimize.brentq(g, s_lo, s_hi, xtol=1e-15, rtol=1e-15, maxiter=500)
    return V_of(s), s


def merton_implied_asset(E, sigma_E, D, T, r):
    """Asset value and asset volatility implied by equity value and equity volatility (Merton 1974).

    Parameters
    ----------
    E : market value of equity today (> 0).
    sigma_E : annual equity volatility (decimal, > 0).
    D : face value of zero-coupon debt due at T (> 0), in the units of E.
    T : debt maturity in years (> 0).
    r : continuously compounded risk-free rate (decimal).

    Returns
    -------
    dict: 'V' (asset value), 'sigma_V' (asset volatility) and every key of
    ``merton_structural(V, D, T, r, sigma_V)`` ('equity' reproduces E and
    'sigma_E' reproduces sigma_E). Arrays broadcast; each element is solved
    separately.

    Formula
    -------
    Solves the two equations E = V N(d1) - D e^{-rT} N(d2) and
    sigma_E E = N(d1) sigma_V V (Hull OFOD "Credit Risk", Jones-Mason-Rosenfeld
    1984). For each trial sigma_V the equity equation is solved for V in
    [E, E + D e^{-rT}]; the volatility equation is then solved for sigma_V in
    (0, sigma_E] (both by Brent's method, to ~1e-14 relative).

    Example
    -------
    >>> m = merton_implied_asset(3.0, 0.80, 10.0, 1.0, 0.05)   # Hull OFOD Credit Risk example
    >>> round(float(m['V']), 2), round(float(m['sigma_V']), 4), round(float(m['pd_rn']), 3)
    (12.4, 0.2123, 0.127)
    """
    E, sE, D, T, r = np.broadcast_arrays(*map(_arr, (E, sigma_E, D, T, r)))
    for name, v in (("E", E), ("sigma_E", sE), ("D", D), ("T", T)):
        _check(np.isfinite(v) & (v > 0), f"{name} must be finite and > 0")
    _check(np.isfinite(r), "r must be finite")
    V = np.empty(E.shape)
    sV = np.empty(E.shape)
    for idx in np.ndindex(E.shape):
        V[idx], sV[idx] = _implied_asset_one(float(E[idx]), float(sE[idx]), float(D[idx]),
                                             float(T[idx]), float(r[idx]))
    out = {"V": _out(V), "sigma_V": _out(sV)}
    out.update(merton_structural(V, D, T, r, sV))
    return out


# ----------------------------------------------------------------------------
# hazard rates and survival
# ----------------------------------------------------------------------------
def hazard_from_spread(spread, *, recovery):
    """Credit-triangle hazard rate lambda = s / (1 - R) from a credit spread.

    Parameters
    ----------
    spread : credit spread (bond yield spread or CDS spread), annual decimal
        (0.015 = 150 bp). A spread to maturity T gives the AVERAGE hazard over [0, T].
    recovery : required, recovery rate R as a fraction of par, 0 <= R < 1.

    Returns
    -------
    Hazard rate per year (decimal), broadcast over the inputs.

    Formula
    -------
    lambda_bar(T) = s(T) / (1 - R) (Hull OFOD "Credit Risk", hazard rates from
    bond yield spreads; exact for a CDS with continuously paid premium and a
    flat hazard rate). Forward hazards between maturities follow from
    lambda_bar: (T2 lambda_bar(T2) - T1 lambda_bar(T1)) / (T2 - T1).

    Example
    -------
    >>> hazard_from_spread([0.015, 0.018, 0.0195], recovery=0.40)   # Hull: 2.50 %, 3.00 %, 3.25 %
    array([0.025 , 0.03  , 0.0325])
    """
    s, R = _arr(spread), _arr(recovery)
    _check(np.isfinite(R) & (R >= 0) & (R < 1), "recovery must satisfy 0 <= R < 1")
    return _out(s / (1.0 - R))


def survival_probability(t, hazard_times, hazards):
    """Survival probability Q(t) = exp(-int_0^t lambda(u) du) for a piecewise-constant hazard rate.

    Parameters
    ----------
    t : time(s) in years (>= 0), any shape.
    hazard_times : 1-D knots t_1 < ... < t_n (years, > 0), the RIGHT ends of the
        hazard intervals.
    hazards : 1-D hazard rates per year (>= 0); hazards[k] applies on
        (t_{k-1}, t_k] (t_0 = 0) and the last one stays flat after t_n.

    Returns
    -------
    Q(t) (1.0 at t = 0), shape of t. The default probability by t is 1 - Q(t).

    Formula
    -------
    Q(t) = exp(-[sum_{j<k} lambda_j (t_j - t_{j-1}) + lambda_k (t - t_{k-1})]) for
    t in (t_{k-1}, t_k] (O'Kane 2008 Ch. 3; Hull OFOD "Credit Derivatives").

    Example
    -------
    >>> survival_probability([1, 2, 3], [1, 2, 3], [0.025, 0.035, 0.0375])
    array([0.97530991, 0.94176453, 0.90710234])
    """
    knots, lam = _hazard_curve(hazard_times, hazards)
    t = _arr(t)
    _check(np.isfinite(t) & (t >= 0), "t must be finite and >= 0 (years)")
    return _out(np.exp(-_cum_hazard(t, knots, lam)))


# ----------------------------------------------------------------------------
# CDS
# ----------------------------------------------------------------------------
def cds_value(spread, hazard_times, hazards, *, pay_times, recovery, curve, default_timing,
              accrual_on_default, notional=1.0):
    """Credit default swap legs, risky annuity, par spread and value to the protection buyer.

    Parameters
    ----------
    spread : contractual CDS spread, annual decimal (0.015 = 150 bp); an array
        of spreads broadcasts in 'premium_pv' and 'value'.
    hazard_times, hazards : piecewise-constant risk-neutral hazard curve (see
        ``survival_probability``).
    pay_times : required, 1-D premium payment times in years (> 0, increasing).
        The CDS starts today: premium accruals are tau_i = t_i - t_{i-1}, t_0 = 0,
        and protection runs to the last payment time.
    recovery : required, recovery rate R (fraction of notional, 0 <= R <= 1).
    curve : required, risk-free discount curve (ZeroCurve or t -> P(0, t)).
    default_timing : required.
        'mid_period' (Hull): a default in (t_{i-1}, t_i] happens at the middle of
        the period; the payoff (1 - R) and the accrued premium (tau_i / 2) are
        discounted from there.
        'continuous' (O'Kane-Turnbull 2003): default at any time; the legs are
        integrated exactly on a grid holding every premium date, every hazard
        knot, every knot of a ZeroCurve and a sub-step <= 1/365 year, with the
        forward rate constant between grid points (exact for a 'log_df'
        ZeroCurve; otherwise the error is O(1e-9) relative).
    accrual_on_default : required bool. True: the buyer pays the premium accrued
        from the last premium date to default (standard contract). False: none.
    notional : principal (default 1.0).

    Returns
    -------
    dict, all in units of ``notional`` except 'par_spread':
    'protection_pv'   (1 - R) E[P(0, tau) 1{tau <= t_n}]
    'annuity'         sum_i tau_i P(0, t_i) Q(t_i) (scheduled premiums per unit spread)
    'accrual_annuity' PV of the accrued premium paid at default per unit spread (0 if not accrued)
    'risky_annuity'   annuity + accrual_annuity (RPV01 per unit spread)
    'premium_pv'      spread * risky_annuity
    'par_spread'      protection_pv / risky_annuity (the spread that makes the value 0)
    'value'           protection_pv - premium_pv (value to the protection BUYER)
    'survival'        Q(t_i) at pay_times (array)

    Formula
    -------
    mid_period: protection = (1 - R) sum_i [Q(t_{i-1}) - Q(t_i)] P(0, m_i),
    accrual = sum_i [Q(t_{i-1}) - Q(t_i)] (tau_i / 2) P(0, m_i), m_i = (t_{i-1} + t_i)/2
    (Hull OFOD "Credit Derivatives", CDS valuation example).
    continuous: on each sub-interval [u, u + h] with hazard lambda and forward f,
    c = lambda + f, the protection integral is P Q lambda (1 - e^{-c h}) / c and
    the accrual integral P Q lambda [d (1 - e^{-c h}) / c + (1 - e^{-c h}(1 + c h)) / c^2],
    d = u - (start of the premium period) (O'Kane-Turnbull 2003).

    Example
    -------
    Hull's example: conditional default probability 2 % a year (lambda = -ln 0.98),
    R = 0.4, flat 5 % continuous rate, 5-year CDS with annual premiums.
    >>> lam = -np.log(0.98)
    >>> v = cds_value(0.015, [5.0], [lam], pay_times=[1, 2, 3, 4, 5], recovery=0.4,
    ...               curve=lambda t: np.exp(-0.05 * np.asarray(t)), default_timing='mid_period',
    ...               accrual_on_default=True)
    >>> round(v['annuity'], 4), round(v['accrual_annuity'], 4), round(v['protection_pv'], 4)
    (4.0704, 0.0426, 0.0511)
    >>> round(v['par_spread'], 4), round(v['value'], 4)
    (0.0124, -0.0106)
    """
    knots, lam = _hazard_curve(hazard_times, hazards)
    pay = _pay_schedule(pay_times)
    timing = _parse_timing(default_timing)
    if not isinstance(accrual_on_default, (bool, np.bool_)):
        raise TypeError("accrual_on_default must be True or False")
    R = float(recovery)
    if not (0.0 <= R <= 1.0):
        raise ValueError("recovery must satisfy 0 <= R <= 1")
    notional = float(notional)
    ann, acc, prot = _cds_legs(knots, lam, pay, R, curve, timing, bool(accrual_on_default))
    risky = ann + acc
    s = _arr(spread)
    premium = s * risky
    return {
        "protection_pv": _out(notional * prot),
        "annuity": _out(notional * ann),
        "accrual_annuity": _out(notional * acc),
        "risky_annuity": _out(notional * risky),
        "premium_pv": _out(notional * premium),
        "par_spread": _out(prot / risky),
        "value": _out(notional * (prot - premium)),
        "survival": np.exp(-_cum_hazard(pay, knots, lam)),
    }


def bootstrap_hazard(maturities, spreads, *, pay_freq, recovery, curve, default_timing,
                     accrual_on_default):
    """Piecewise-constant hazard rates that reprice a term structure of par CDS spreads exactly.

    Parameters
    ----------
    maturities : 1-D CDS maturities in years (> 0, strictly increasing); they
        become the hazard knots (right ends of the hazard intervals).
    spreads : 1-D par CDS spreads, annual decimals (> 0), one per maturity.
    pay_freq : required, premium payments per year (positive int or
        'annual'/'semiannual'/'quarterly'/'monthly'). Premium dates are counted
        back from each maturity, T - j/pay_freq > 0 (a short first period
        when T * pay_freq is not an integer), accruing from t = 0.
    recovery : required, recovery rate R (0 <= R < 1).
    curve : required, risk-free discount curve (ZeroCurve or t -> P(0, t)).
    default_timing, accrual_on_default : required, as in ``cds_value``.

    Returns
    -------
    dict: 'hazard_times' (= maturities), 'hazards' (per year, one per
    interval) and 'survival' (Q at each maturity).

    Formula
    -------
    Sequentially for k = 1..n, lambda_k solves value_k(lambda_1..lambda_k) = 0,
    where value_k is the ``cds_value`` of the k-th CDS at its par spread
    (Brent's method, to ~1e-14; O'Kane-Turnbull 2003; Hull OFOD "Credit
    Derivatives"). Raises ValueError when a spread would need a negative
    hazard (a spread curve that falls too steeply is an arbitrage).

    Example
    -------
    >>> b = bootstrap_hazard([5.0], [0.0100], pay_freq=1, recovery=0.4,
    ...                      curve=lambda t: np.exp(-0.05 * np.asarray(t)),
    ...                      default_timing='mid_period', accrual_on_default=True)
    >>> round(float(1 - np.exp(-b['hazards'][0])), 4)   # Hull: 1.61 % a year
    0.0161
    """
    mats = np.array(maturities, dtype=float, ndmin=1).ravel()
    sp = np.array(spreads, dtype=float, ndmin=1).ravel()
    if mats.size == 0 or mats.shape != sp.shape:
        raise ValueError("maturities and spreads must be non-empty and of equal length")
    if not (np.all(np.isfinite(mats)) and mats[0] > 0 and np.all(np.diff(mats) > 0)):
        raise ValueError("maturities must be finite, > 0 and strictly increasing")
    if not np.all(np.isfinite(sp) & (sp > 0)):
        raise ValueError("spreads must be finite and > 0")
    m = _parse_freq(pay_freq)
    timing = _parse_timing(default_timing)
    if not isinstance(accrual_on_default, (bool, np.bool_)):
        raise TypeError("accrual_on_default must be True or False")
    aod = bool(accrual_on_default)
    R = float(recovery)
    if not (0.0 <= R < 1.0):
        raise ValueError("recovery must satisfy 0 <= R < 1")
    lam = np.zeros(mats.size)
    for k in range(mats.size):
        pay = _coupon_times(mats[k], m)
        knots = mats[: k + 1]

        def value(x, k=k, pay=pay, knots=knots):
            trial = lam[: k + 1].copy()
            trial[k] = x
            ann, acc, prot = _cds_legs(knots, trial, pay, R, curve, timing, aod)
            return prot - sp[k] * (ann + acc)

        if value(0.0) > 0:
            raise ValueError(
                f"spread {sp[k]} at maturity {mats[k]} needs a negative hazard rate; "
                "the spread curve falls too steeply (arbitrage)")
        hi = 1.0
        while value(hi) < 0:
            hi *= 2.0
            if hi > 1e4:
                raise ValueError(f"no hazard rate reprices spread {sp[k]} at maturity {mats[k]}")
        lam[k] = optimize.brentq(value, 0.0, hi, xtol=1e-15, rtol=1e-14, maxiter=500)
    return {
        "hazard_times": mats,
        "hazards": lam,
        "survival": np.exp(-_cum_hazard(mats, mats, lam)),
    }


# ----------------------------------------------------------------------------
# rating migration
# ----------------------------------------------------------------------------
def _check_stochastic(P, name="P"):
    P = np.array(P, dtype=float)
    if P.ndim != 2 or P.shape[0] != P.shape[1] or P.shape[0] == 0:
        raise ValueError(f"{name} must be a square 2-D matrix")
    if not np.all(np.isfinite(P)):
        raise ValueError(f"{name} must be finite")
    if np.any(P < -_STOCH_TOL):
        raise ValueError(f"{name} has negative entries")
    dev = np.abs(P.sum(axis=1) - 1.0)
    if np.any(dev > _STOCH_TOL):
        raise ValueError(
            f"rows of {name} must sum to 1 (max deviation {dev.max():.3g}); pass decimals, not "
            "percent, and renormalise rounded published tables, e.g. P / P.sum(1, keepdims=True)")
    return P


def migration_n_period(P, n):
    """n-period rating transition matrix P^n of a time-homogeneous Markov chain.

    Parameters
    ----------
    P : (k, k) one-period transition matrix, row-stochastic, in decimals
        (rows sum to 1 within 1e-8; entries >= 0).
    n : number of periods, a non-negative integer (0 gives the identity).
        For a fractional horizon use migration_generator + migration_from_generator.

    Returns
    -------
    (k, k) ndarray P^n; entry (i, j) is the probability of moving from rating i
    to rating j over n periods (the default column gives cumulative default
    probabilities when default is absorbing).

    Formula
    -------
    P(n) = P^n by the Chapman-Kolmogorov equations (Hull OFOD "Credit Risk",
    rating transition matrices; Jarrow-Lando-Turnbull 1997).

    Example
    -------
    >>> migration_n_period([[0.9, 0.08, 0.02], [0.1, 0.8, 0.1], [0, 0, 1]], 2)
    array([[0.818, 0.136, 0.046],
           [0.17 , 0.648, 0.182],
           [0.   , 0.   , 1.   ]])
    """
    P = _check_stochastic(P)
    if isinstance(n, (bool, np.bool_)):
        raise TypeError("n must be a non-negative integer")
    if isinstance(n, (float, np.floating)) and float(n).is_integer():
        n = int(n)
    if not isinstance(n, (int, np.integer)) or n < 0:
        raise ValueError(f"n must be a non-negative integer, got {n!r}")
    return np.linalg.matrix_power(P, int(n))


_GEN_METHODS = ("log", "diagonal_adjust", "weighted_adjust", "jlt")


def migration_generator(P, *, method):
    """Generator (intensity) matrix Q of a one-period rating transition matrix, with expm(Q) ~ P.

    Parameters
    ----------
    P : (k, k) one-period transition matrix, row-stochastic, in decimals.
    method : required.
        'log': the principal matrix logarithm L = logm(P), returned as is. It
            is exact (expm(L) = P) but may have negative off-diagonal entries,
            i.e. not be a valid generator. Raises if logm(P) is not real.
        'diagonal_adjust' (IRW DA): L with negative off-diagonal entries set to
            0 and each diagonal reset to minus its row's off-diagonal sum.
        'weighted_adjust' (IRW WA): negative off-diagonal entries set to 0 and
            their total B_i removed from the entries of the correct sign in
            proportion to their absolute values: q_ij -= B_i |q_ij| / G_i,
            G_i = |q_ii| + sum_{j != i} max(q_ij, 0).
        'jlt' (Jarrow-Lando-Turnbull 1997, at most one move a period):
            q_ii = ln p_ii, q_ij = p_ij ln(p_ii) / (p_ii - 1); a row with p_ii = 1
            (absorbing) is 0. Needs every p_ii > 0.

    Returns
    -------
    (k, k) ndarray Q with rows summing to 0 (to rounding); the two IRW methods
    and 'jlt' give non-negative off-diagonal entries.

    Formula
    -------
    Q = logm(P) and the adjustments above (Israel-Rosenthal-Wei 2001;
    Jarrow-Lando-Turnbull 1997). P(t) = expm(Q t) (``migration_from_generator``).

    Example
    -------
    >>> migration_generator([[0.95, 0.05], [0.0, 1.0]], method='jlt')   # exact for 2 states
    array([[-0.05129329,  0.05129329],
           [ 0.        ,  0.        ]])
    """
    P = _check_stochastic(P)
    if not isinstance(method, str) or method.strip().lower() not in _GEN_METHODS:
        raise ValueError(f"method must be one of {_GEN_METHODS}, got {method!r}")
    method = method.strip().lower()
    k = P.shape[0]
    off = ~np.eye(k, dtype=bool)
    if method == "jlt":
        p = np.diag(P)
        if np.any(p <= 0):
            raise ValueError("method='jlt' needs every diagonal entry p_ii > 0")
        move = p < 1.0
        lp = np.log(p)
        ratio = np.where(move, lp / np.where(move, p - 1.0, 1.0), 0.0)
        Q = P * ratio[:, None]
        Q[~move, :] = 0.0
        np.fill_diagonal(Q, np.where(move, lp, 0.0))
        return Q
    L = linalg.logm(P)
    if np.iscomplexobj(L):
        if np.max(np.abs(L.imag)) > 1e-10 * max(1.0, np.max(np.abs(L.real))):
            raise ValueError("P has no real principal logarithm (e.g. negative eigenvalues)")
        L = L.real
    L = np.array(L, dtype=float)
    if method == "log":
        return L
    if method == "diagonal_adjust":
        Q = np.where(off, np.maximum(L, 0.0), 0.0)
        np.fill_diagonal(Q, -Q.sum(axis=1))
        return Q
    # weighted_adjust
    Q = L.copy()
    neg = off & (L < 0)
    B = np.where(neg, -L, 0.0).sum(axis=1)
    G = np.abs(np.diag(L)) + np.where(off & (L > 0), L, 0.0).sum(axis=1)
    Q[neg] = 0.0
    keep = ~neg
    with np.errstate(divide="ignore", invalid="ignore"):
        adj = np.where(G[:, None] > 0, B[:, None] * np.abs(L) / G[:, None], 0.0)
    Q = np.where(keep, Q - adj, Q)
    return Q


def migration_from_generator(Q, t):
    """Transition matrix P(t) = expm(Q t) of a continuous-time rating Markov chain.

    Parameters
    ----------
    Q : (k, k) generator matrix, rows summing to 0 (within 1e-8 of the row scale).
        Off-diagonal entries should be >= 0; a matrix with negative ones (e.g.
        from migration_generator(method='log')) is accepted, but P(t) may then
        have negative entries.
    t : horizon in years (units of the period of Q), a scalar >= 0 or a 1-D array.

    Returns
    -------
    (k, k) ndarray for scalar t, or (len(t), k, k) stacked for an array t.

    Formula
    -------
    P(t) = expm(Q t) = sum_n (Q t)^n / n! (Jarrow-Lando-Turnbull 1997;
    Israel-Rosenthal-Wei 2001), via scipy.linalg.expm (Pade approximation).

    Example
    -------
    >>> migration_from_generator([[-0.1, 0.1], [0.0, 0.0]], 2.0)
    array([[0.81873075, 0.18126925],
           [0.        , 1.        ]])
    """
    Q = np.array(Q, dtype=float)
    if Q.ndim != 2 or Q.shape[0] != Q.shape[1] or Q.shape[0] == 0:
        raise ValueError("Q must be a square 2-D matrix")
    if not np.all(np.isfinite(Q)):
        raise ValueError("Q must be finite")
    scale = np.maximum(np.abs(Q).max(axis=1), 1.0)
    if np.any(np.abs(Q.sum(axis=1)) > _STOCH_TOL * scale):
        raise ValueError("rows of a generator Q must sum to 0")
    tt = np.asarray(t, dtype=float)
    if tt.ndim > 1 or not np.all(np.isfinite(tt)) or np.any(tt < 0):
        raise ValueError("t must be a scalar or 1-D array of finite times >= 0")
    if tt.ndim == 0:
        return linalg.expm(Q * float(tt))
    return np.stack([linalg.expm(Q * float(x)) for x in tt]) if tt.size else np.empty((0,) + Q.shape)
