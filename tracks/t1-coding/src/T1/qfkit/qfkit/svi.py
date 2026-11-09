"""SVI (stochastic volatility inspired) implied-variance smiles and local volatility from total implied variance:
the raw SVI parameterisation with its analytic strike derivatives, a bounded least-squares SVI calibration per
expiry, and Gatheral's form of the Dupire local volatility.

Conventions for the whole module:
- k is LOG-MONEYNESS ln(K/F_T) against the FORWARD F_T = S e^{(r-q)T} (not the spot), per expiry.
- w is TOTAL implied variance sigma_imp^2 T (not volatility); sigma_imp = sqrt(w/T).
- Raw SVI parameters (a, b, rho, m, sigma): level a, slope b >= 0, correlation |rho| < 1, centre m, width sigma > 0
  (an SVI width, not a Black-Scholes volatility).

Functions
---------
svi_total_variance  w(k) = a + b (rho (k - m) + sqrt((k - m)^2 + sigma^2)) and its analytic dw/dk, d2w/dk2
svi_calibrate       weighted least squares of raw SVI on one expiry's total variances, within bounds
gatheral_local_vol  Dupire local volatility from w and its derivatives (Gatheral 2006 eq. 1.10)

Ported from the team's qflib (``localvol.svi_total_variance``, ``svi_calibrate`` and the Gatheral denominator of
``dupire_from_total_variance``), with the derivatives exposed and supplied by the caller, so the stencil a task pins
(analytic in k, forward difference in T, ...) can be followed.

References: Gatheral (2004), "A parsimonious arbitrage-free implied volatility parameterization" (Global
Derivatives); Gatheral, The Volatility Surface (2006) Ch. 1 (eq. 1.10) and Ch. 3; Gatheral and Jacquier (2014),
Quantitative Finance 14 (arbitrage-free SVI); Dupire (1994), Risk 7.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import optimize

__all__ = [
    "svi_total_variance",
    "svi_calibrate",
    "gatheral_local_vol",
]


def _arrays(*values):
    return [np.asarray(v, dtype=float) for v in values]


def svi_total_variance(k, a, b, rho, m, sigma, *, deriv=0):
    """Raw SVI total implied variance w(k) = a + b (rho (k - m) + sqrt((k - m)^2 + sigma^2)) (Gatheral 2004), or its
    analytic first or second derivative in log-moneyness.

    Parameters
    ----------
    k : float or array. Log-moneyness ln(K/F_T).
    a, b, rho, m, sigma : floats or arrays (broadcast). Raw SVI parameters: b >= 0, |rho| < 1, sigma > 0.
    deriv : 0 (w), 1 (dw/dk = b (rho + (k-m)/sqrt((k-m)^2 + sigma^2))) or 2
        (d2w/dk2 = b sigma^2 / ((k-m)^2 + sigma^2)^{3/2}).

    Returns
    -------
    float or ndarray. Implied vol = sqrt(w / T).

    References
    ----------
    Gatheral (2004); Gatheral and Jacquier (2014) eq. (3.1).

    Example
    -------
    >>> w = svi_total_variance(np.log(K / F), *params); iv = np.sqrt(w / T)
    >>> svi_total_variance(0.0, 0.04, 0.4, -0.4, 0.0, 0.1)  # 0.04 + 0.4 * 0.1 = 0.08
    """
    kk, aa, bb, rr, mm, ss = _arrays(k, a, b, rho, m, sigma)
    if np.any(bb < 0) or np.any(np.abs(rr) >= 1) or np.any(ss <= 0):
        raise ValueError("raw SVI needs b >= 0, |rho| < 1 and sigma > 0")
    if deriv not in (0, 1, 2) or isinstance(deriv, bool):
        raise ValueError("deriv must be 0 (w), 1 (dw/dk) or 2 (d2w/dk2)")
    y = kk - mm
    root = np.sqrt(y * y + ss * ss)
    if deriv == 0:
        out = aa + bb * (rr * y + root)
    elif deriv == 1:
        out = bb * (rr + y / root)
    else:
        out = bb * ss * ss / root ** 3
    return float(out) if np.ndim(out) == 0 else out


def svi_calibrate(k, w, *, lower, upper, weights=None):
    """Calibrate a raw SVI smile to one expiry: weighted least squares on total implied variance within box bounds,
    from deterministic starts, keeping the minimum total variance a + b sigma sqrt(1 - rho^2) >= 0.

    Parameters
    ----------
    k : array_like (n,). Log-moneyness ln(K/F_T) of the quotes, n >= 5.
    w : array_like (n,). Market total implied variance sigma_imp^2 T (> 0), not volatility.
    lower, upper : REQUIRED. Length-5 bounds in the order (a, b, rho, m, sigma), lower < upper, with b >= 0,
        -1 < rho < 1 and sigma > 0 inside them; e.g. lower=(-max(w), 1e-4, -0.999, 2*min(k), 1e-3),
        upper=(max(w), 5.0, 0.999, 2*max(k), 2.0).
    weights : optional (n,) weights >= 0 (e.g. vega); None = equal.

    Returns
    -------
    dict
        a, b, rho, m, sigma : fitted parameters; rmse : sqrt(sum wt (w_svi - w)^2 / sum wt) in total variance;
        success : the optimiser converged and the minimum total variance is >= 0.

    Formula
    -------
    min over the box of sum_i wt_i (w_svi(k_i) - w_i)^2 + (100 min(0, a + b sigma sqrt(1 - rho^2)))^2 by
    scipy.optimize.least_squares from 10 starts (the box centre and a 3 x 3 grid of rho and m); the best is kept.

    References
    ----------
    Gatheral (2004); Gatheral and Jacquier (2014); Zeliade Systems (2009), "Quasi-explicit calibration of
    Gatheral's SVI model".

    Example
    -------
    >>> fit = svi_calibrate(np.log(K / F), iv**2 * T, lower=(-0.5, 1e-4, -0.999, -1.0, 1e-3),
    ...                     upper=(0.5, 5.0, 0.999, 1.0, 2.0))
    >>> iv_svi = np.sqrt(svi_total_variance(np.log(K / F), fit['a'], fit['b'], fit['rho'], fit['m'], fit['sigma']) / T)
    """
    kk, ww = _arrays(k, w)
    if kk.ndim != 1 or kk.shape != ww.shape or kk.size < 5:
        raise ValueError("k and w must be 1-D arrays of the same length >= 5 (five parameters)")
    if np.any(~np.isfinite(kk)) or np.any(~np.isfinite(ww)) or np.any(ww <= 0):
        raise ValueError("k must be finite and w finite total variances > 0 (iv**2 * T, not volatilities)")
    wt = np.ones(kk.size) if weights is None else np.asarray(weights, dtype=float)
    if wt.shape != kk.shape or np.any(~np.isfinite(wt)) or np.any(wt < 0) or wt.sum() <= 0:
        raise ValueError("weights must be None or n finite numbers >= 0, not all 0")
    lo, hi = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    if lo.shape != (5,) or hi.shape != (5,) or np.any(~(hi > lo)):
        raise ValueError("lower and upper must have length 5, ordered (a, b, rho, m, sigma), with lower < upper")
    if lo[1] < 0 or lo[2] <= -1 or hi[2] >= 1 or lo[4] <= 0:
        raise ValueError("the bounds must keep b >= 0, -1 < rho < 1 and sigma > 0")
    root_wt = np.sqrt(wt)

    def residual(p):
        a, b, rho, m, sigma = p
        y = kk - m
        model = a + b * (rho * y + np.sqrt(y * y + sigma * sigma))
        gap = a + b * sigma * math.sqrt(max(0.0, 1.0 - rho * rho))
        return np.append(root_wt * (model - ww), 100.0 * min(0.0, gap))
    mid = 0.5 * (lo + hi)
    starts = [mid]
    for rho in np.linspace(lo[2] + 0.05 * (hi[2] - lo[2]), hi[2] - 0.05 * (hi[2] - lo[2]), 3):
        for m in np.linspace(lo[3], hi[3], 3):
            guess = mid.copy()
            guess[2], guess[3] = rho, m
            starts.append(guess)
    best = None
    for guess in starts:
        res = optimize.least_squares(residual, guess, bounds=(lo, hi), xtol=1e-14, ftol=1e-14, gtol=1e-14,
                                     max_nfev=4000)
        if best is None or res.cost < best.cost:
            best = res
    a, b, rho, m, sigma = (float(v) for v in best.x)
    err = svi_total_variance(kk, a, b, rho, m, sigma) - ww
    rmse = math.sqrt(float(np.sum(wt * err * err) / wt.sum()))
    feasible = a + b * sigma * math.sqrt(max(0.0, 1.0 - rho * rho)) >= -1e-10
    return {"a": a, "b": b, "rho": rho, "m": m, "sigma": sigma, "rmse": rmse,
            "success": bool(best.status > 0 and feasible)}


def gatheral_local_vol(k, w, w_k, w_kk, w_T):
    """Dupire local volatility from total implied variance and its derivatives (analytic, e.g. from SVI, or by finite
    differences): sigma_loc^2 = w_T / (1 - (k/w) w_k + (1/4)(-1/4 - 1/w + k^2/w^2) w_k^2 + (1/2) w_kk).

    Gatheral (2006) eq. (1.10), the Dupire formula in log-moneyness and total variance.

    Parameters
    ----------
    k : float or array. Log-moneyness ln(K/F_T) against the forward of the same expiry.
    w : float or array. Total implied variance at (k, T), > 0.
    w_k, w_kk : first and second derivatives of w in k at fixed T (analytic from
        svi_total_variance(..., deriv=1 / 2), or finite differences).
    w_T : derivative of w in T at fixed k (e.g. the forward difference (w(k, T2) - w(k, T1))/(T2 - T1)).

    Returns
    -------
    float or ndarray. The local volatility sigma_loc(K, T) (annualised); NaN where the numerator w_T or the
    denominator is not positive (calendar or butterfly arbitrage).

    Formula
    -------
    Equivalent to Dupire's sigma^2 = (C_T + (r - q) K C_K + q C) / (0.5 K^2 C_KK) on the Black-Scholes call prices
    of w (forward-moneyness form, any r and q). A flat surface w = sigma^2 T returns sigma.

    References
    ----------
    Dupire (1994); Gatheral (2006) eq. (1.10); Gatheral and Jacquier (2014) eq. (2.2) (the denominator g(k)).

    Example
    -------
    >>> p = svi_params[T]; k = np.log(K / F_T)
    >>> w, wk, wkk = (svi_total_variance(k, *p, deriv=d) for d in (0, 1, 2))
    >>> lv = gatheral_local_vol(k, w, wk, wkk, (w_next_T - w) / (T_next - T))
    """
    kk, ww, wk, wkk, wt = np.broadcast_arrays(*_arrays(k, w, w_k, w_kk, w_T))
    if np.any(~(ww > 0)):
        raise ValueError("w must be total implied variance > 0")
    den = 1.0 - kk / ww * wk + 0.25 * (-0.25 - 1.0 / ww + kk * kk / (ww * ww)) * wk * wk + 0.5 * wkk
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where((den > 0) & (wt > 0), np.sqrt(np.where(den > 0, wt / den, np.nan)), np.nan)
    return float(out) if out.ndim == 0 else out
