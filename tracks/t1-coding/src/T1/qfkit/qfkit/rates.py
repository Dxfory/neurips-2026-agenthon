"""Short-rate option pricing (Vasicek, Hull-White one-factor) and swap annuities / par swap rates.

Conventions for the whole module: times are in YEARS from today (t = 0);
the short rate r and forward rates are continuously compounded annual
decimals; ``sigma`` is the ABSOLUTE (normal) short-rate volatility per
sqrt(year) (0.01 = 100 bp a year), and ``a`` / ``kappa`` is the mean-reversion
speed per year (a = 0 is the Ho-Lee / Merton limit). Models:
Vasicek dr = kappa (theta - r) dt + sigma dW with a RISK-NEUTRAL theta, and
Hull-White dr = [theta(t) - a r] dt + sigma dW with theta(t) fitted to
today's discount curve.

A ``curve`` argument is today's discount curve: a qfkit ``ZeroCurve`` (its
``df`` method is used, with the curve's own interpolation) or any callable
t -> P(0, t) (called with an array first, element-wise if that fails).
Bond options are on a zero-coupon bond paying ``face`` at T_bond, and the
strike K is in the same units (K = 63 with face = 100 is 63 % of par).
``kind`` is always required. Rate strikes K of caplets and swaptions are
simple (money-market) annual rates per unit of accrual year fraction.
Vectorised functions return a numpy float64 (a ``float`` subclass) when every
input is a scalar and an ``np.ndarray`` (broadcast shape) otherwise.

Functions
---------
vasicek_bond_option  European option on a zero-coupon bond under Vasicek (Jamshidian 1989)
hw_bond_price        Hull-White P(t, T) given the short rate r(t), fitted to today's curve
hw_bond_option       Hull-White European option on a zero-coupon bond
hw_caplet_price      Hull-White caplet / floorlet, priced as a zero-bond put / call
hw_swaption_price    Hull-White payer / receiver swaption by Jamshidian decomposition
swap_annuity         fixed-leg annuity (PV01 per unit rate) sum tau_i P(0, t_i)
par_swap_rate        single-curve (forward-start) par swap rate
"""

from __future__ import annotations

import numpy as np
from scipy import optimize
from scipy.special import ndtr

from .processes import vasicek_bond_price

__all__ = [
    "vasicek_bond_option",
    "hw_bond_price",
    "hw_bond_option",
    "hw_caplet_price",
    "hw_swaption_price",
    "swap_annuity",
    "par_swap_rate",
]

_FWD_H = 1e-4  # years (~53 minutes): step of the finite difference that gives f(0, t)


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _out(x):
    """0-d result -> numpy float64 scalar; otherwise the ndarray itself."""
    x = np.asarray(x, dtype=float)
    return x[()] if x.ndim == 0 else x


def _arr(x):
    return np.asarray(x, dtype=float)


def _kind(kind, table):
    """Map a case-insensitive kind string through table {name: value}; anything else raises."""
    names = " or ".join(repr(k) for k in table)
    if not isinstance(kind, str):
        raise TypeError(f"kind must be the string {names}")
    k = kind.strip().lower()
    if k not in table:
        raise ValueError(f"kind must be {names}, got {kind!r}")
    return table[k]


def _check(cond, msg):
    if not np.all(cond):
        raise ValueError(msg)


def _discount(curve, t):
    """P(0, t) from a ZeroCurve (``.df``) or a callable t -> P(0, t); same shape as t."""
    t = _arr(t)
    fn = getattr(curve, "df", None)
    if fn is None:
        if not callable(curve):
            raise TypeError("curve must be a qfkit ZeroCurve or a callable t -> P(0, t)")
        fn = curve
    try:
        p = _arr(fn(t))
        if p.shape != t.shape:
            raise ValueError("shape")
    except (TypeError, ValueError):
        p = np.array([float(fn(float(x))) for x in t.ravel()], dtype=float).reshape(t.shape)
    if not np.all(np.isfinite(p) & (p > 0)):
        raise ValueError("curve returned a non-positive or non-finite discount factor")
    return p


def _inst_forward(curve, t):
    """Instantaneous forward f(0, t) = -d ln P(0, t)/dt by finite differences of ln P.

    Central difference with step h = 1e-4 years where t >= h (error O(h^2));
    the second-order one-sided difference [3 ln P(t) - 4 ln P(t+h) + ln P(t+2h)]/(2h)
    near t = 0. At a kink of a piecewise curve the central difference returns
    the average of the left and right forwards.
    """
    t = _arr(t)
    h = _FWD_H
    central = t >= h
    lp = lambda s: np.log(_discount(curve, s))  # noqa: E731
    fc = (lp(np.where(central, t - h, t)) - lp(t + h)) / (2.0 * h)
    fo = (3.0 * lp(t) - 4.0 * lp(t + h) + lp(t + 2.0 * h)) / (2.0 * h)
    return np.where(central, fc, fo)


def _B(a, tau):
    """(1 - e^{-a tau}) / a, with the limit tau at a = 0 (expm1: no cancellation for small a)."""
    a, tau = _arr(a), _arr(tau)
    nz = a != 0
    aa = np.where(nz, a, 1.0)
    return np.where(nz, -np.expm1(-aa * tau) / aa, tau)


def _zb_price(Fd, Kd, s, w):
    """Black formula for an option on a zero-coupon bond.

    Fd = face P(0, T_bond) (discounted bond forward), Kd = K P(0, T_opt)
    (discounted strike), s = sigma_P (price volatility over the option life),
    w = +1 call / -1 put. price = w [Fd N(w h) - Kd N(w (h - s))],
    h = ln(Fd/Kd)/s + s/2. s == 0 gives the deterministic limit max(w (Fd - Kd), 0).
    """
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        h = np.log(Fd / Kd) / s + 0.5 * s
        p = w * (Fd * ndtr(w * h) - Kd * ndtr(w * (h - s)))
    det = np.maximum(w * (Fd - Kd), 0.0)
    return np.where(s == 0, det, p)


def _schedule(start, pay_times, accruals, name="start"):
    """Validated (start, pay_times, accruals) as floats / 1-D arrays."""
    start = float(start)
    pay = np.array(pay_times, dtype=float, ndmin=1).ravel()
    if pay.size == 0 or not np.all(np.isfinite(pay)):
        raise ValueError("pay_times must be a non-empty array of finite times (years)")
    if not np.isfinite(start) or start < 0:
        raise ValueError(f"{name} must be a finite time >= 0 (years)")
    if pay[0] <= start or np.any(np.diff(pay) <= 0):
        raise ValueError(f"pay_times must be strictly increasing and > {name}")
    if accruals is None:
        tau = np.diff(np.concatenate(([start], pay)))
    else:
        tau = np.array(accruals, dtype=float, ndmin=1).ravel()
        if tau.shape != pay.shape:
            raise ValueError("accruals must have one entry per payment time")
    if not np.all(np.isfinite(tau) & (tau > 0)):
        raise ValueError("accruals must be finite and > 0 (year fractions)")
    return start, pay, tau


# ----------------------------------------------------------------------------
# Vasicek
# ----------------------------------------------------------------------------
def vasicek_bond_option(r, T_opt, T_bond, K, kappa, theta, sigma, *, kind, face=1.0):
    """European call or put on a zero-coupon bond under the Vasicek model (closed form).

    Parameters
    ----------
    r : current short rate (decimal, continuously compounded).
    T_opt : option expiry in years (>= 0). T_opt == 0 gives the intrinsic value
        max(w (face P(0, T_bond) - K), 0).
    T_bond : maturity of the underlying zero-coupon bond in years (>= T_opt).
    K : strike price, in the units of ``face`` (>= 0).
    kappa : mean-reversion speed per year (>= 0; 0 is the driftless limit).
    theta : RISK-NEUTRAL long-run mean of r (decimal). If a source gives a
        real-world theta and a market price of risk lambda, pass
        theta - lambda sigma / kappa.
    sigma : absolute short-rate volatility per sqrt(year) (>= 0).
    kind : 'call' or 'put' (required).
    face : bond principal paid at T_bond (default 1.0; use 100 for % of par).

    Returns
    -------
    Option price today, in the units of ``face``.

    Formula
    -------
    P(0, T) = vasicek_bond_price(r, T, kappa, theta, sigma) (qfkit.processes).
    sigma_P = sigma B(T_bond - T_opt) sqrt((1 - e^{-2 kappa T_opt}) / (2 kappa)),
    B(tau) = (1 - e^{-kappa tau}) / kappa (limits tau and sqrt(T_opt) at kappa = 0),
    h = ln(face P(0, T_bond) / (K P(0, T_opt))) / sigma_P + sigma_P / 2,
    call = face P(0, T_bond) N(h) - K P(0, T_opt) N(h - sigma_P),
    put  = K P(0, T_opt) N(-h + sigma_P) - face P(0, T_bond) N(-h)
    (Jamshidian 1989; Hull OFOD "Interest Rate Derivatives: Models of the
    Short Rate"). sigma_P == 0 gives max(w (face P(0, T_bond) - K P(0, T_opt)), 0).

    Example
    -------
    >>> vasicek_bond_option(0.05, 1.0, 3.0, 87.0, 0.1, 0.08, 0.015, kind='call', face=100)
    2.5929830...   # Hull OFOD short-rate chapter problem: 2.59
    """
    w = _kind(kind, {"call": 1.0, "put": -1.0})
    r, T_opt, T_bond, K, kappa, theta, sigma, face = map(
        _arr, (r, T_opt, T_bond, K, kappa, theta, sigma, face))
    _check(T_opt >= 0, "T_opt must be >= 0 (years)")
    _check(T_bond >= T_opt, "T_bond must be >= T_opt")
    _check(K >= 0, "K must be >= 0")
    _check(face > 0, "face must be > 0")
    _check(kappa >= 0, "kappa must be >= 0")
    _check(sigma >= 0, "sigma must be >= 0")
    P_opt = _arr(vasicek_bond_price(r, T_opt, kappa, theta, sigma))
    P_bond = _arr(vasicek_bond_price(r, T_bond, kappa, theta, sigma))
    sP = sigma * _B(kappa, T_bond - T_opt) * np.sqrt(_B(2.0 * kappa, T_opt))
    return _out(_zb_price(face * P_bond, K * P_opt, sP, w))


# ----------------------------------------------------------------------------
# Hull-White one-factor
# ----------------------------------------------------------------------------
def hw_bond_price(r_t, t, T, a, sigma, *, curve):
    """Hull-White one-factor zero-coupon bond price P(t, T) given the short rate r(t).

    Parameters
    ----------
    r_t : short rate at time t (decimal, continuously compounded).
    t : future time in years (>= 0) at which the bond is valued.
    T : bond maturity in years (>= t).
    a : mean-reversion speed per year (>= 0; a = 0 is Ho-Lee).
    sigma : absolute short-rate volatility per sqrt(year) (>= 0).
    curve : required, today's discount curve (ZeroCurve or t -> P(0, t)) to
        which theta(t) is fitted.

    Returns
    -------
    P(t, T) per unit face.

    Formula
    -------
    P(t, T) = A(t, T) e^{-B(t, T) r(t)},  B(t, T) = (1 - e^{-a (T - t)}) / a,
    ln A(t, T) = ln(P(0, T) / P(0, t)) + B(t, T) f(0, t)
                 - sigma^2 / (4 a) (1 - e^{-2 a t}) B(t, T)^2
    (Hull-White 1990; Hull OFOD "Models of the Short Rate"; Brigo-Mercurio
    2006 eq. 3.39). a = 0 gives B = T - t and sigma^2 t (T - t)^2 / 2.
    f(0, t) = -d ln P(0, t)/dt is taken from ``curve`` by a central difference
    of ln P with step 1e-4 years (one-sided, second order, for t < 1e-4); at a
    kink of a piecewise curve it is the average of the two one-sided forwards.
    With t = 0 and r_t = f(0, 0) the result is P(0, T) exactly.

    Example
    -------
    >>> zc = ZeroCurve([1, 5, 10], [0.03, 0.035, 0.04], comp='continuous', interp='linear_zero')
    >>> hw_bond_price(0.04, 2.0, 7.0, 0.1, 0.01, curve=zc)
    0.8006208...
    """
    r_t, t, T, a, sigma = map(_arr, (r_t, t, T, a, sigma))
    _check(t >= 0, "t must be >= 0 (years)")
    _check(T >= t, "T must be >= t")
    _check(a >= 0, "a must be >= 0")
    _check(sigma >= 0, "sigma must be >= 0")
    tb, Tb = np.broadcast_arrays(t, T)
    lnP_t = np.log(_discount(curve, tb))
    lnP_T = np.log(_discount(curve, Tb))
    f = _inst_forward(curve, t)
    B = _B(a, T - t)
    lnA = lnP_T - lnP_t + B * f - 0.5 * sigma**2 * _B(2.0 * a, t) * B**2
    return _out(np.exp(lnA - B * r_t))


def hw_bond_option(K, T_opt, T_bond, a, sigma, *, curve, kind, face=1.0):
    """European call or put on a zero-coupon bond under Hull-White one-factor (closed form).

    Parameters
    ----------
    K : strike price in the units of ``face`` (>= 0).
    T_opt : option expiry in years (>= 0; 0 gives the intrinsic value).
    T_bond : bond maturity in years (>= T_opt).
    a : mean-reversion speed per year (>= 0; a = 0 is Ho-Lee).
    sigma : absolute short-rate volatility per sqrt(year) (>= 0).
    curve : required, today's discount curve (ZeroCurve or t -> P(0, t)).
    kind : 'call' or 'put' (required).
    face : bond principal paid at T_bond (default 1.0; 100 for % of par).

    Returns
    -------
    Option price today, in the units of ``face``.

    Formula
    -------
    sigma_P = (sigma / a) (1 - e^{-a (T_bond - T_opt)}) sqrt((1 - e^{-2 a T_opt}) / (2 a)),
    call = face P(0, T_bond) N(h) - K P(0, T_opt) N(h - sigma_P),
    put  = K P(0, T_opt) N(-h + sigma_P) - face P(0, T_bond) N(-h),
    h = ln(face P(0, T_bond) / (K P(0, T_opt))) / sigma_P + sigma_P / 2,
    with P(0, .) read from ``curve`` (Hull-White 1990; Hull OFOD "Models of the
    Short Rate"). sigma_P == 0 gives max(w (face P(0, T_bond) - K P(0, T_opt)), 0).

    Example
    -------
    Hull OFOD (Models of the Short Rate): 3-year put on a 9-year zero, K = 63,
    face 100, a = 0.1, sigma = 0.01, Hull's zero curve -> 1.8093.
    >>> zc = ZeroCurve([1, 5, 10], [0.03, 0.035, 0.04], comp='continuous', interp='linear_zero')
    >>> hw_bond_option(0.87, 2.0, 5.0, 0.1, 0.01, curve=zc, kind='put')
    0.0032883...
    """
    w = _kind(kind, {"call": 1.0, "put": -1.0})
    K, T_opt, T_bond, a, sigma, face = map(_arr, (K, T_opt, T_bond, a, sigma, face))
    _check(T_opt >= 0, "T_opt must be >= 0 (years)")
    _check(T_bond >= T_opt, "T_bond must be >= T_opt")
    _check(K >= 0, "K must be >= 0")
    _check(face > 0, "face must be > 0")
    _check(a >= 0, "a must be >= 0")
    _check(sigma >= 0, "sigma must be >= 0")
    P_opt = _discount(curve, T_opt)
    P_bond = _discount(curve, T_bond)
    sP = sigma * _B(a, T_bond - T_opt) * np.sqrt(_B(2.0 * a, T_opt))
    return _out(_zb_price(face * P_bond, K * P_opt, sP, w))


def hw_caplet_price(K, t_reset, t_pay, a, sigma, *, curve, kind, notional=1.0, accrual=None):
    """Hull-White caplet or floorlet, priced as a put or call on a zero-coupon bond.

    Parameters
    ----------
    K : cap / floor rate, a simple annual rate (decimal) with 1 + accrual K > 0.
    t_reset : fixing (reset) time of the rate, in years (>= 0).
    t_pay : payment time in years (> t_reset); the rate runs over [t_reset, t_pay].
    a, sigma : Hull-White mean reversion (>= 0) and absolute volatility (>= 0).
    curve : required, today's discount curve (ZeroCurve or t -> P(0, t)).
    kind : 'caplet' or 'floorlet' (required).
    notional : principal (default 1.0).
    accrual : year fraction tau of the period (day count of the rate).
        None (default) uses t_pay - t_reset.

    Returns
    -------
    Price today, in units of ``notional``.

    Formula
    -------
    The payoff notional tau max(L - K, 0) at t_pay, with 1 + tau L = 1 / P(t_reset, t_pay),
    equals notional (1 + tau K) max(X - P(t_reset, t_pay), 0) at t_reset, X = 1/(1 + tau K):
    caplet   = notional (1 + tau K) ZBP(0, t_reset, t_pay, X),
    floorlet = notional (1 + tau K) ZBC(0, t_reset, t_pay, X)
    with the Hull-White zero-bond options of ``hw_bond_option`` (Hull OFOD
    "Interest Rate Derivatives: Models of the Short Rate"; Brigo-Mercurio 2006
    eq. 3.40). caplet - floorlet = notional tau P(0, t_pay) (F - K), F the simple
    forward rate (P(0, t_reset)/P(0, t_pay) - 1)/tau.

    Example
    -------
    >>> zc = ZeroCurve([1, 5, 10], [0.03, 0.035, 0.04], comp='continuous', interp='linear_zero')
    >>> hw_caplet_price(0.04, 2.0, 2.5, 0.1, 0.01, curve=zc, kind='caplet', notional=1e6)
    1321.8085...
    """
    w = _kind(kind, {"floorlet": 1.0, "caplet": -1.0})
    K, t_reset, t_pay, a, sigma, notional = map(_arr, (K, t_reset, t_pay, a, sigma, notional))
    tau = t_pay - t_reset if accrual is None else _arr(accrual)
    _check(t_reset >= 0, "t_reset must be >= 0 (years)")
    _check(t_pay > t_reset, "t_pay must be > t_reset")
    _check(tau > 0, "accrual must be > 0")
    _check(1.0 + tau * K > 0, "1 + accrual * K must be > 0")
    _check(a >= 0, "a must be >= 0")
    _check(sigma >= 0, "sigma must be >= 0")
    X = 1.0 / (1.0 + tau * K)
    P_reset = _discount(curve, t_reset)
    P_pay = _discount(curve, t_pay)
    sP = sigma * _B(a, t_pay - t_reset) * np.sqrt(_B(2.0 * a, t_reset))
    return _out(notional * (1.0 + tau * K) * _zb_price(P_pay, X * P_reset, sP, w))


def hw_swaption_price(K, T_opt, pay_times, a, sigma, *, curve, kind, notional=1.0, accruals=None):
    """Hull-White European payer or receiver swaption by Jamshidian's decomposition.

    Parameters
    ----------
    K : fixed rate of the underlying swap, simple annual decimal (>= 0).
        A 1-D array of strikes is priced strike by strike.
    T_opt : option expiry = swap start, in years (>= 0).
    pay_times : 1-D array of fixed-leg payment times (years, > T_opt, increasing).
    a, sigma : Hull-White mean reversion (>= 0) and absolute volatility (>= 0), scalars.
    curve : required, today's discount curve (ZeroCurve or t -> P(0, t)). The
        floating leg is valued single-curve (worth P(T_opt) - P(t_n) per unit).
    kind : 'payer' (right to pay fixed) or 'receiver' (right to receive fixed), required.
    notional : principal (default 1.0).
    accruals : fixed-leg year fractions tau_i; None (default) uses
        diff([T_opt, *pay_times]).

    Returns
    -------
    Price today in units of ``notional`` (float, or an array for an array K).

    Formula
    -------
    A payer swaption is a put with strike 1 on the coupon bond paying
    c_i = K tau_i at t_i (plus 1 at t_n); a receiver is the call. With
    P(T_opt, t_i) = A_i e^{-B_i r}, the rate r* solving sum_i c_i A_i e^{-B_i r*} = 1
    gives strikes X_i = A_i e^{-B_i r*}, and
    payer    = notional sum_i c_i ZBP(0, T_opt, t_i, X_i),
    receiver = notional sum_i c_i ZBC(0, T_opt, t_i, X_i)
    (Jamshidian 1989; Hull OFOD "Models of the Short Rate"; Brigo-Mercurio 2006
    eq. 3.44-3.45). r* is found by bracketed root search (Brent). Parity:
    payer - receiver = notional [P(0, T_opt) - P(0, t_n) - K sum_i tau_i P(0, t_i)].

    Example
    -------
    >>> zc = ZeroCurve([1, 5, 10], [0.03, 0.035, 0.04], comp='continuous', interp='linear_zero')
    >>> hw_swaption_price(0.04, 2.0, [3, 4, 5, 6, 7], 0.1, 0.01, curve=zc, kind='payer', notional=100)
    1.7502334...
    """
    w = _kind(kind, {"receiver": 1.0, "payer": -1.0})
    T_opt, a, sigma, notional = float(T_opt), float(a), float(sigma), float(notional)
    if not (np.isfinite(a) and a >= 0):
        raise ValueError("a must be >= 0")
    if not (np.isfinite(sigma) and sigma >= 0):
        raise ValueError("sigma must be >= 0")
    T_opt, pay, tau = _schedule(T_opt, pay_times, accruals, name="T_opt")
    P_opt = float(_discount(curve, T_opt))
    P_pay = _discount(curve, pay)
    B = _B(a, pay - T_opt)
    vT = float(_B(2.0 * a, T_opt))
    sP = sigma * B * np.sqrt(vT)
    # P(T_opt, t_i) = base_i exp(-B_i y), y = r(T_opt) - f(0, T_opt) (no forward-rate derivative needed)
    base = P_pay / P_opt * np.exp(-0.5 * sigma**2 * vT * B**2)

    def one(k):
        if not (np.isfinite(k) and k >= 0):
            raise ValueError("K must be finite and >= 0 (Jamshidian needs non-negative coupons)")
        c = k * tau
        c[-1] += 1.0

        def g(y):
            with np.errstate(over="ignore"):
                return float(np.dot(c, base * np.exp(-B * y))) - 1.0

        lo, hi = -0.5, 0.5
        for _ in range(200):
            if g(lo) > 0:
                break
            lo *= 2.0
        for _ in range(200):
            if g(hi) < 0:
                break
            hi *= 2.0
        y = optimize.brentq(g, lo, hi, xtol=1e-15, rtol=1e-15, maxiter=500)
        X = base * np.exp(-B * y)
        return notional * float(np.dot(c, _zb_price(P_pay, X * P_opt, sP, w)))

    Ks = _arr(K)
    out = np.empty(Ks.shape)
    for idx, k in np.ndenumerate(Ks):
        out[idx] = one(float(k))
    return _out(out)


# ----------------------------------------------------------------------------
# swaps
# ----------------------------------------------------------------------------
def swap_annuity(start, pay_times, *, curve, accruals=None):
    """Fixed-leg annuity (PV01 per unit of rate) of a swap: sum_i tau_i P(0, t_i).

    Parameters
    ----------
    start : accrual start of the first period in years (>= 0; 0 = spot start).
    pay_times : 1-D array of fixed payment times (years, > start, increasing).
    curve : required, discount curve (ZeroCurve or t -> P(0, t)).
    accruals : year fractions tau_i (day count of the fixed leg); None
        (default) uses diff([start, *pay_times]).

    Returns
    -------
    float, the annuity per unit notional per unit rate (multiply by
    notional * 1e-4 for a DV01 per basis point).

    Formula
    -------
    A = sum_i tau_i P(0, t_i) (Hull OFOD "Swaps"; Brigo-Mercurio 2006 Ch. 1).
    A fixed leg paying rate c is worth c A, plus P(0, t_n) with the principal.

    Example
    -------
    >>> zc = ZeroCurve([0.25, 0.75, 1.25], [0.10, 0.105, 0.11], comp='continuous', interp='linear_zero')
    >>> 100 * (0.08 * swap_annuity(0, [0.25, 0.75, 1.25], curve=zc, accruals=[0.5] * 3) + zc.df(1.25))
    98.237...   # Hull OFOD "Swaps": B_fix = 98.24
    """
    _, pay, tau = _schedule(start, pay_times, accruals)
    return _out(np.dot(tau, _discount(curve, pay)))


def par_swap_rate(start, pay_times, *, curve, accruals=None):
    """Single-curve par swap rate of a spot- or forward-starting swap.

    Parameters
    ----------
    start : swap start in years (>= 0; 0 = spot start).
    pay_times : 1-D array of fixed payment times (years, > start, increasing).
    curve : required, discount curve (ZeroCurve or t -> P(0, t)); the floating
        leg is valued on the same curve (worth P(0, start) - P(0, t_n)).
    accruals : fixed-leg year fractions; None (default) uses diff([start, *pay_times]).

    Returns
    -------
    float, the fixed rate (simple, per accrual year fraction) that makes the
    swap worth zero.

    Formula
    -------
    S = (P(0, start) - P(0, t_n)) / sum_i tau_i P(0, t_i) (Hull OFOD "Swaps";
    Brigo-Mercurio 2006 eq. 1.25). A single period gives the simple forward
    rate (P(0, start)/P(0, t_1) - 1)/tau_1; start = 0 with a regular schedule
    equals ``curves.par_yield``.

    Example
    -------
    >>> zc = ZeroCurve([0.5, 1, 1.5, 2], [0.04, 0.045, 0.048, 0.04953], comp='continuous', interp='log_df')
    >>> par_swap_rate(0, [0.5, 1, 1.5, 2], curve=zc)
    0.0500006...   # Hull OFOD "Swaps": 5 % (the 2-year zero 4.953 % is rounded)
    """
    start, pay, tau = _schedule(start, pay_times, accruals)
    ends = _discount(curve, np.array([start, pay[-1]]))
    return _out((ends[0] - ends[1]) / np.dot(tau, _discount(curve, pay)))
