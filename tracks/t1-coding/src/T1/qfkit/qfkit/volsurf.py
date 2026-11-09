"""Stochastic and local volatility: Heston prices, SABR implied vols, implied-vol approximations, Dupire.

Conventions for the whole module (the same as ``bsm``): S is spot, F a forward,
K strike, T time to expiry in YEARS; r and q are continuously compounded annual
rates as decimals; volatilities are annual decimals (0.20, not 20). Heston's v0
and theta are VARIANCES (0.04 = 20% vol). ``kind`` is ONE string, 'call' or
'put'. Every convention on which sources differ (SABR vol type, the Dupire
input type, what to do with a negative Corrado-Miller radicand) is a REQUIRED
keyword. Vectorised functions return a numpy float64 (a ``float`` subclass)
when every input is a scalar and an ``np.ndarray`` otherwise.

Functions
---------
heston_price              Heston (1993) European price, little-trap CF + Lewis Fourier inversion
sabr_implied_vol          Hagan et al. (2002) SABR lognormal (Black) or normal (Bachelier) vol
iv_brenner_subrahmanyam   at-the-money-forward implied-vol approximation
iv_corrado_miller         Corrado-Miller (1996) quadratic implied-vol approximation
dupire_local_vol          Dupire local-vol grid from a call-price or implied-vol surface
"""

from __future__ import annotations

import math

import numpy as np
from scipy import integrate

from .bsm import bsm_price

__all__ = [
    "heston_price",
    "sabr_implied_vol",
    "iv_brenner_subrahmanyam",
    "iv_corrado_miller",
    "dupire_local_vol",
]

_SQRT_2PI = math.sqrt(2.0 * math.pi)


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


def _choice(value, name, options):
    if not isinstance(value, str) or value.strip().lower() not in options:
        raise ValueError(f"{name} must be one of {options}, got {value!r}")
    return value.strip().lower()


# ----------------------------------------------------------------------------
# Heston
# ----------------------------------------------------------------------------
def _heston_cf(u, T, v0, kappa, theta, sigma_v, rho):
    """E[exp(i u X)] with X = ln(S_T / F_T), Albrecher et al. (2007) "little trap" form (complex u).

    Written without the 1/sigma_v^2 cancellation: b - d = -sigma_v^2 (iu + u^2)/(b + d),
    so D and C stay accurate as sigma_v -> 0 (where they tend to the BSM exponent).
    """
    iu = 1j * u
    a = iu + u * u
    b = kappa - rho * sigma_v * iu
    s2 = sigma_v * sigma_v
    d = np.sqrt(b * b + s2 * a)
    bd = b + d
    g = -s2 * a / (bd * bd)                      # (b - d)/(b + d)
    e = np.exp(-d * T)
    D = -a / bd * (1.0 - e) / (1.0 - g * e)      # (b - d)/sigma_v^2 (1 - e)/(1 - g e)
    zs = -a / (bd * bd) * (1.0 - e) / (1.0 - g)  # z / sigma_v^2, z = g (1 - e)/(1 - g)
    z = s2 * zs
    small = np.abs(z) < 1e-8
    zsafe = np.where(small, 1.0, z)
    ratio = np.where(small, 1.0 - 0.5 * z, np.log1p(zsafe) / zsafe)   # ln(1 + z)/z
    # C = kappa theta/sigma_v^2 [(b - d) T - 2 ln((1 - g e)/(1 - g))], ln(...) = ln(1 + z)
    C = kappa * theta * (-a * T / bd - 2.0 * zs * ratio)
    return np.exp(C + D * v0)


def _heston_call_one(S, K, T, r, q, v0, kappa, theta, sigma_v, rho):
    F = S * math.exp((r - q) * T)
    x = math.log(F / K)

    def integrand(u):
        z = _heston_cf(u - 0.5j, T, v0, kappa, theta, sigma_v, rho)
        return (np.exp(1j * u * x) * z).real / (u * u + 0.25)

    val, _ = integrate.quad(integrand, 0.0, np.inf, limit=2000, epsabs=1e-13, epsrel=1e-12)
    price = S * math.exp(-q * T) - math.sqrt(F * K) * math.exp(-r * T) * val / math.pi
    # quadrature noise can leave a far out-of-the-money price a few 1e-13 outside the bounds
    return min(max(price, max(S * math.exp(-q * T) - K * math.exp(-r * T), 0.0)), S * math.exp(-q * T))


def heston_price(S, K, T, r, v0, kappa, theta, sigma_v, rho, *, kind, q=0.0):
    """Heston (1993) European call or put price by Fourier inversion of the "little trap" characteristic function.

    Parameters
    ----------
    S, K : spot and strike (> 0).
    T : years to expiry (>= 0; T == 0 gives the intrinsic value).
    r, q : continuously compounded annual rate and dividend yield (q defaults to 0.0).
    v0 : initial instantaneous VARIANCE (>= 0; 0.04 means 20% vol).
    kappa : mean-reversion speed of the variance (per year, >= 0).
    theta : long-run VARIANCE (>= 0).
    sigma_v : volatility of variance ("vol of vol", >= 0). sigma_v == 0 gives
        the BSM price with the integrated deterministic variance.
    rho : correlation of the spot and variance Brownian motions, in [-1, 1].
    kind : 'call' or 'put' (required).
    All parameters are RISK-NEUTRAL. Numeric inputs broadcast.

    Returns
    -------
    Option price. Puts come from put-call parity P = C - S e^{-qT} + K e^{-rT}.

    Formula
    -------
    dS/S = (r - q) dt + sqrt(v) dW1, dv = kappa (theta - v) dt + sigma_v sqrt(v) dW2,
    dW1 dW2 = rho dt. With X = ln(S_T/F), F = S e^{(r-q)T}, the characteristic
    function (Albrecher et al. 2007) is phi(u) = exp(C + D v0),
    b = kappa - rho sigma_v i u, d = sqrt(b^2 + sigma_v^2 (i u + u^2)),
    g = (b - d)/(b + d), C = kappa theta/sigma_v^2 [(b - d) T - 2 ln((1 - g e^{-dT})/(1 - g))],
    D = (b - d)/sigma_v^2 (1 - e^{-dT})/(1 - g e^{-dT}).
    Lewis (2001) single-integral inversion, x = ln(F/K):
    C = S e^{-qT} - sqrt(F K) e^{-rT}/pi * int_0^inf Re[e^{i u x} phi(u - i/2)]/(u^2 + 1/4) du,
    integrated by adaptive quadrature (scipy quad, tolerances 1e-13 absolute,
    1e-12 relative); the call is clipped into its no-arbitrage bounds
    [max(S e^{-qT} - K e^{-rT}, 0), S e^{-qT}] (removes quadrature noise of ~1e-13).
    sigma_v == 0: BSM with sigma^2 T = theta T + (v0 - theta)(1 - e^{-kappa T})/kappa.

    References
    ----------
    Heston (1993), Rev. Financial Studies 6; Albrecher, Mayer, Schoutens and
    Tistaert (2007), "The little Heston trap"; Lewis (2000/2001); Gatheral
    (2006) Ch. 2.

    Example
    -------
    >>> heston_price(100, 100, 1.0, 0.01, 0.04, 4.0, 0.25, 1.0, -0.5, kind='call', q=0.02)
    16.0701549...                                        # Lewis (2000) reference value
    """
    w = _sign(kind)
    S, K, T, r, q, v0, kappa, theta, sigma_v, rho = map(
        _arr, (S, K, T, r, q, v0, kappa, theta, sigma_v, rho))
    for name, v in (("S", S), ("K", K)):
        if not np.all(np.isfinite(v)) or np.any(v <= 0):
            raise ValueError(f"{name} must be finite and > 0")
    for name, v in (("T", T), ("v0", v0), ("kappa", kappa), ("theta", theta), ("sigma_v", sigma_v)):
        if not np.all(np.isfinite(v)) or np.any(v < 0):
            raise ValueError(f"{name} must be finite and >= 0")
    if not np.all(np.isfinite(r)) or not np.all(np.isfinite(q)):
        raise ValueError("r and q must be finite")
    if not np.all(np.isfinite(rho)) or np.any(np.abs(rho) > 1):
        raise ValueError("rho must lie in [-1, 1]")
    arrs = np.broadcast_arrays(S, K, T, r, q, v0, kappa, theta, sigma_v, rho)
    shape = arrs[0].shape
    S, K, T, r, q, v0, kappa, theta, sigma_v, rho = (a.ravel() for a in arrs)
    call = np.empty(S.size)
    for i in range(S.size):
        if T[i] == 0.0:
            call[i] = max(S[i] - K[i], 0.0)
        elif sigma_v[i] == 0.0:
            kT = kappa[i] * T[i]
            frac = -math.expm1(-kT) / kT if kT > 0 else 1.0
            var = theta[i] * T[i] + (v0[i] - theta[i]) * T[i] * frac
            vol = math.sqrt(max(var, 0.0) / T[i])
            call[i] = float(bsm_price(S[i], K[i], T[i], r[i], vol, kind="call", q=q[i]))
        else:
            call[i] = _heston_call_one(S[i], K[i], T[i], r[i], q[i], v0[i], kappa[i], theta[i],
                                       sigma_v[i], rho[i])
    if w > 0:
        price = call
    else:
        price = call - S * np.exp(-q * T) + K * np.exp(-r * T)
    return _out(price.reshape(shape))


# ----------------------------------------------------------------------------
# SABR
# ----------------------------------------------------------------------------
def _z_over_x(z, rho):
    """z / x(z), x(z) = ln[(sqrt(1 - 2 rho z + z^2) + z - rho)/(1 - rho)]; series near z = 0."""
    small = np.abs(z) < 1e-5
    zs = np.where(small, 1.0, z)  # placeholder away from 0 to avoid 0/0 in the unused branch
    with np.errstate(divide="ignore", invalid="ignore"):
        xz = np.log((np.sqrt(1.0 - 2.0 * rho * zs + zs * zs) + zs - rho) / (1.0 - rho))
        exact = zs / xz
    series = 1.0 - 0.5 * rho * z + (2.0 - 3.0 * rho * rho) * z * z / 12.0
    return np.where(small, series, exact)


def sabr_implied_vol(F, K, T, alpha, beta, rho, nu, *, vol_type):
    """Hagan-Kumar-Lesniewski-Woodward (2002) SABR implied volatility, lognormal (Black) or normal (Bachelier).

    Parameters
    ----------
    F, K : forward and strike (> 0).
    T : years to expiry (>= 0).
    alpha : initial volatility level (> 0), in units of F^{1-beta} per sqrt(year).
    beta : CEV exponent in [0, 1] (0 normal, 1 lognormal backbone).
    rho : forward/vol correlation in (-1, 1).
    nu : volatility of volatility (>= 0, per sqrt(year)).
    vol_type : required, 'lognormal' (Black-76 vol, Hagan eq. 2.17a) or
        'normal' (Bachelier / basis-point vol in units of F per sqrt(year),
        Hagan eq. A.69a).
    Numeric inputs broadcast.

    Returns
    -------
    Implied volatility (decimal). The at-the-money limit K -> F is taken
    analytically (z/x(z) -> 1 by its Taylor series for |z| < 1e-5).

    Formula
    -------
    Lognormal: sigma_B = alpha / {(FK)^{(1-b)/2} [1 + (1-b)^2/24 L^2 + (1-b)^4/1920 L^4]}
      * z/x(z) * {1 + [(1-b)^2 alpha^2/(24 (FK)^{1-b}) + rho b nu alpha/(4 (FK)^{(1-b)/2})
      + (2 - 3 rho^2) nu^2/24] T},
      L = ln(F/K), z = (nu/alpha)(FK)^{(1-b)/2} L,
      x(z) = ln[(sqrt(1 - 2 rho z + z^2) + z - rho)/(1 - rho)].
    ATM (eq. 2.18): alpha/F^{1-b} {1 + [(1-b)^2 alpha^2/(24 F^{2-2b}) + rho b nu alpha/(4 F^{1-b})
      + (2 - 3 rho^2) nu^2/24] T}.
    Normal: sigma_N = alpha (1-b)(F - K)/(F^{1-b} - K^{1-b}) * zeta/x(zeta)
      * {1 + [-b(2-b) alpha^2/(24 f_av^{2-2b}) + rho alpha nu b/(4 f_av^{1-b})
      + (2 - 3 rho^2) nu^2/24] T},  f_av = sqrt(FK), zeta = (nu/alpha)(F - K)/f_av^b;
      the factor (1-b)(F-K)/(F^{1-b} - K^{1-b}) is (F-K)/ln(F/K) at b = 1 and F^b at K = F.

    References
    ----------
    Hagan, Kumar, Lesniewski and Woodward (2002), "Managing smile risk",
    Wilmott Magazine, eqs. (2.17a), (2.18), (A.69a).

    Example
    -------
    >>> sabr_implied_vol(0.03, 0.03, 1.0, 0.2, 1.0, 0.0, 0.0, vol_type='lognormal')   # nu=0, beta=1: Black
    0.2
    """
    vt = _choice(vol_type, "vol_type", ("lognormal", "normal"))
    F, K, T, alpha, beta, rho, nu = map(_arr, (F, K, T, alpha, beta, rho, nu))
    for name, v in (("F", F), ("K", K), ("alpha", alpha)):
        if not np.all(np.isfinite(v)) or np.any(v <= 0):
            raise ValueError(f"{name} must be finite and > 0")
    if not np.all(np.isfinite(T)) or np.any(T < 0):
        raise ValueError("T must be finite and >= 0")
    if not np.all(np.isfinite(nu)) or np.any(nu < 0):
        raise ValueError("nu must be finite and >= 0")
    if not np.all(np.isfinite(beta)) or np.any((beta < 0) | (beta > 1)):
        raise ValueError("beta must lie in [0, 1]")
    if not np.all(np.isfinite(rho)) or np.any(np.abs(rho) >= 1):
        raise ValueError("rho must lie in (-1, 1)")
    omb = 1.0 - beta
    L = np.log(F / K)
    fk = F * K
    three = (2.0 - 3.0 * rho * rho) * nu * nu / 24.0
    if vt == "lognormal":
        fkb = fk ** (0.5 * omb)
        denom = fkb * (1.0 + omb ** 2 / 24.0 * L ** 2 + omb ** 4 / 1920.0 * L ** 4)
        z = nu / alpha * fkb * L
        corr = 1.0 + (omb ** 2 * alpha ** 2 / (24.0 * fkb ** 2) + rho * beta * nu * alpha / (4.0 * fkb)
                      + three) * T
        vol = alpha / denom * _z_over_x(z, rho) * corr
    else:
        fav = np.sqrt(fk)
        near = np.abs(L) < 1e-8
        with np.errstate(divide="ignore", invalid="ignore"):
            gen = omb * (F - K) / (F ** omb - K ** omb)
            lim1 = (F - K) / L
        ratio = np.where(beta == 1.0, lim1, gen)
        ratio = np.where(near, fav ** beta, ratio)
        zeta = nu / alpha * (F - K) / fav ** beta
        corr = 1.0 + (-beta * (2.0 - beta) * alpha ** 2 / (24.0 * fav ** (2.0 - 2.0 * beta))
                      + rho * alpha * nu * beta / (4.0 * fav ** omb) + three) * T
        vol = alpha * ratio * _z_over_x(zeta, rho) * corr
    return _out(vol)


# ----------------------------------------------------------------------------
# implied-volatility approximations
# ----------------------------------------------------------------------------
def iv_brenner_subrahmanyam(price, S, T, *, q=0.0):
    """Brenner-Subrahmanyam (1988) at-the-money-forward implied-volatility approximation.

    Parameters
    ----------
    price : option price (a call or a put; they are equal at the money forward).
    S : spot (> 0).  T : years to expiry (> 0).
    q : continuous dividend yield (decimal), default 0.0.

    Returns
    -------
    sigma (annual decimal) ~ sqrt(2 pi / T) * price / (S e^{-qT}).

    Formula
    -------
    For K = S e^{(r-q)T}: C = S e^{-qT} [2 N(sigma sqrt(T)/2) - 1]
    ~ S e^{-qT} sigma sqrt(T) / sqrt(2 pi), so sigma ~ sqrt(2 pi/T) C/(S e^{-qT}).
    The relative error is about -sigma^2 T/24 exactly at the money forward
    and grows with |ln(S e^{(r-q)T}/K)|; the strike is not an input.

    References
    ----------
    Brenner and Subrahmanyam (1988), "A simple formula to compute the implied
    standard deviation", Financial Analysts Journal 44(5).

    Example
    -------
    >>> iv_brenner_subrahmanyam(2.40, 49, 20/52)    # Hull Ch. 19 option (true vol 0.20)
    0.1979...
    """
    price, S, T, q = map(_arr, (price, S, T, q))
    if not np.all(np.isfinite(S)) or np.any(S <= 0):
        raise ValueError("S must be finite and > 0")
    if not np.all(np.isfinite(T)) or np.any(T <= 0):
        raise ValueError("T must be finite and > 0")
    return _out(np.sqrt(2.0 * np.pi / T) * price / (S * np.exp(-q * T)))


def iv_corrado_miller(price, S, K, T, r, *, kind, on_negative, q=0.0):
    """Corrado-Miller (1996) quadratic approximation of the Black-Scholes-Merton implied volatility.

    Parameters
    ----------
    price : option price.
    S, K : spot and strike (> 0).  T : years to expiry (> 0).
    r, q : continuously compounded annual rate and dividend yield (q defaults to 0.0).
    kind : 'call' or 'put' (required); a put is first converted to the call
        price C = P + S e^{-qT} - K e^{-rT} (put-call parity).
    on_negative : required, what to do when the radicand below is negative
        (far from the money or a price below intrinsic):
        'nan'  -> return nan;
        'zero' -> set the radicand to 0 (drop the square-root term).

    Returns
    -------
    sigma (annual decimal); nan where the result is not positive or finite.

    Formula
    -------
    With S' = S e^{-qT} and X = K e^{-rT}:
    sigma sqrt(T) = sqrt(2 pi)/(S' + X) * [C - (S' - X)/2
                    + sqrt((C - (S' - X)/2)^2 - (S' - X)^2/pi)].

    References
    ----------
    Corrado and Miller (1996), "A note on a simple, accurate formula to
    compute implied standard deviations", J. Banking & Finance 20.

    Example
    -------
    >>> iv_corrado_miller(1.875, 21, 20, 0.25, 0.10, kind='call', on_negative='nan')
    0.2324...                                         # Hull: exact implied vol 0.235
    """
    w = _sign(kind)
    mode = _choice(on_negative, "on_negative", ("nan", "zero"))
    price, S, K, T, r, q = map(_arr, (price, S, K, T, r, q))
    for name, v in (("S", S), ("K", K), ("T", T)):
        if not np.all(np.isfinite(v)) or np.any(v <= 0):
            raise ValueError(f"{name} must be finite and > 0")
    Sd = S * np.exp(-q * T)
    X = K * np.exp(-r * T)
    C = price if w > 0 else price + Sd - X
    A = C - 0.5 * (Sd - X)
    rad = A * A - (Sd - X) ** 2 / np.pi
    if mode == "zero":
        rad = np.maximum(rad, 0.0)
    with np.errstate(invalid="ignore"):
        root = np.sqrt(rad)
    sig = _SQRT_2PI / (Sd + X) * (A + root) / np.sqrt(T)
    sig = np.where(np.isfinite(sig) & (sig > 0), sig, np.nan)
    return _out(sig)


# ----------------------------------------------------------------------------
# Dupire local volatility
# ----------------------------------------------------------------------------
def _second_diff(f, x):
    """Three-point second derivative along the last axis on a non-uniform grid; nan at both edges."""
    out = np.full(f.shape, np.nan)
    h1 = x[1:-1] - x[:-2]
    h2 = x[2:] - x[1:-1]
    out[..., 1:-1] = 2.0 * (f[..., :-2] / (h1 * (h1 + h2)) - f[..., 1:-1] / (h1 * h2)
                            + f[..., 2:] / (h2 * (h1 + h2)))
    return out


def dupire_local_vol(strikes, maturities, surface, *, S, r, surface_type, q=0.0):
    """Dupire (1994) local-volatility grid from a call-price or implied-volatility surface by finite differences.

    Parameters
    ----------
    strikes : 1-D strictly increasing strikes (> 0), at least 3.
    maturities : 1-D strictly increasing times in years (> 0), at least 2.
    surface : array of shape (len(maturities), len(strikes)); row i is maturity i.
        surface_type='price': European CALL prices as present values (discounted,
        not forward/undiscounted prices).
        surface_type='iv': Black-Scholes implied volatilities (annual decimals).
    S : spot (> 0).  r, q : continuously compounded annual rate and dividend
        yield (scalars; q defaults to 0.0).
    surface_type : required, 'price' or 'iv'.

    Returns
    -------
    ndarray (len(maturities), len(strikes)) of local VOLATILITIES sigma_loc(K, T)
    (annual decimals). nan in the first and last strike columns (no central
    second derivative) and wherever the finite-difference numerator or
    denominator shows calendar or butterfly arbitrage (numerator < 0 or
    denominator <= 0) or is not finite.

    Formula
    -------
    'price' (Dupire): sigma_loc^2 = [dC/dT + (r - q) K dC/dK + q C] / [K^2/2 d2C/dK2].
    'iv' (Gatheral 2006 eq. 1.10, total variance w = sigma_imp^2 T, y = ln(K/F_T),
    F_T = S e^{(r-q)T}):
    sigma_loc^2 = (dw/dT)|_y / [1 - (y/w) w_y + (1/4)(-1/4 - 1/w + y^2/w^2) w_y^2 + w_yy/2],
    with the grid derivatives at fixed K converted by (dw/dT)|_y = (dw/dT)|_K + (r - q) K dw/dK,
    w_y = K dw/dK, w_yy = K^2 d2w/dK2 + K dw/dK.
    Derivatives: first derivatives by numpy.gradient on the (possibly
    non-uniform) grid, second-order central inside and second-order one-sided
    at the edges (maturity edges included); second strike derivatives by the
    three-point non-uniform central formula (interior strikes only).

    References
    ----------
    Dupire (1994), "Pricing with a smile", Risk 7; Gatheral (2006) Ch. 1.

    Example
    -------
    >>> Ks = np.linspace(60, 140, 41); Ts = np.array([0.25, 0.5, 1.0])
    >>> iv = np.full((3, 41), 0.2)                         # flat surface -> local vol 0.2
    >>> dupire_local_vol(Ks, Ts, iv, S=100, r=0.03, surface_type='iv')[1, 20]
    0.2
    """
    st = _choice(surface_type, "surface_type", ("price", "iv"))
    K = np.asarray(strikes, dtype=float)
    Tm = np.asarray(maturities, dtype=float)
    V = np.asarray(surface, dtype=float)
    if K.ndim != 1 or K.size < 3 or np.any(~np.isfinite(K)) or np.any(K <= 0) or np.any(np.diff(K) <= 0):
        raise ValueError("strikes must be 1-D, finite, > 0, strictly increasing, with at least 3 values")
    if Tm.ndim != 1 or Tm.size < 2 or np.any(~np.isfinite(Tm)) or np.any(Tm <= 0) or np.any(np.diff(Tm) <= 0):
        raise ValueError("maturities must be 1-D, finite, > 0, strictly increasing, with at least 2 values")
    if V.shape != (Tm.size, K.size):
        raise ValueError(f"surface must have shape (len(maturities), len(strikes)) = {(Tm.size, K.size)}, "
                         f"got {V.shape}")
    S = float(S)
    r = float(r)
    q = float(q)
    if not (math.isfinite(S) and S > 0 and math.isfinite(r) and math.isfinite(q)):
        raise ValueError("S must be > 0 and r, q finite")
    edge_T = 2 if Tm.size >= 3 else 1
    Kg = K[None, :]
    Tg = Tm[:, None]
    with np.errstate(all="ignore"):
        if st == "price":
            dT = np.gradient(V, Tm, axis=0, edge_order=edge_T)
            dK = np.gradient(V, K, axis=1, edge_order=2)
            dKK = _second_diff(V, K)
            num = dT + (r - q) * Kg * dK + q * V
            den = 0.5 * Kg * Kg * dKK
        else:
            wv = V * V * Tg
            dT = np.gradient(wv, Tm, axis=0, edge_order=edge_T)
            dK = np.gradient(wv, K, axis=1, edge_order=2)
            dKK = _second_diff(wv, K)
            y = np.log(Kg / (S * np.exp((r - q) * Tg)))
            num = dT + (r - q) * Kg * dK
            wy = Kg * dK
            wyy = Kg * Kg * dKK + Kg * dK
            den = (1.0 - y / wv * wy + 0.25 * (-0.25 - 1.0 / wv + y * y / (wv * wv)) * wy * wy
                   + 0.5 * wyy)
        loc = num / den
        ok = np.isfinite(num) & np.isfinite(den) & (num >= 0) & (den > 0)
        if st == "iv":
            ok &= wv > 0
        return np.where(ok, np.sqrt(np.where(ok, loc, 0.0)), np.nan)
