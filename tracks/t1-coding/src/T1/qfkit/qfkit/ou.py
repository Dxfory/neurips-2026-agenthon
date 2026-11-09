"""Ornstein-Uhlenbeck toolkit: OLS fit (Euler-differenced or exact AR(1)), path simulation
(exact or Euler, optional Bernoulli or Poisson normal jumps) and k-sigma jump detection.

Model: dX = kappa (theta - X) dt + sigma dW (+ J dN), X a level, a log price, a spread or a
de-seasonalised residual. Conventions for the whole module:
- ``dt`` is the time between consecutive observations or steps IN THE UNIT kappa and sigma are
  quoted in (1/252 for daily data with annual parameters; 1 for parameters per period).
  kappa is per unit time, sigma per sqrt(unit time).
- Half-lives are given both in time units, ln 2 / kappa, and in PERIODS (steps),
  ln 2 / (kappa dt).
- Series are time-ordered and equally spaced; ``pair_mask`` drops the pairs (x_{t-1}, x_t) that
  straddle a break (e.g. keep within-year pairs only).
- Simulation draws are documented exactly (order and shapes), so a seeded numpy generator
  reproduces a hand-written loop.

Functions
---------
ou_fit        kappa, theta, sigma, half-life and stationary sd from OLS on (x_{t-1}, x_t) pairs
ou_simulate   OU paths: exact transition or Euler steps; record before or after each advance; jumps
detect_jumps  flag residuals beyond k standard deviations; jump mean, sd and the diffusion sd

References: Uhlenbeck and Ornstein (1930), Physical Review 36; Vasicek (1977), J. Financial
Economics 5; Brigo and Mercurio, Interest Rate Models (2006) Ch. 3; Glasserman (2004) Sec. 3.3;
Cont and Tankov, Financial Modelling with Jump Processes (2004); Tsay (2010) Ch. 2.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

__all__ = [
    "ou_fit",
    "ou_simulate",
    "detect_jumps",
]

_LN2 = math.log(2.0)


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _choice(value, name, allowed):
    if not isinstance(value, str) or value not in allowed:
        raise ValueError(f"{name} must be one of {', '.join(repr(a) for a in allowed)}, got {value!r}")
    return value


def _pos(value, name):
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a number")
    v = float(value)
    if not (math.isfinite(v) and v > 0):
        raise ValueError(f"{name} must be finite and > 0")
    return v


def _count(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be an integer >= 1")
    return int(value)


def _ddof(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError(f"{name} must be an integer >= 0 (0 = divide by n, 1 = n - 1, 2 = n - 2)")
    return int(value)


def _var_factor(kappa, dt):
    """(1 - e^{-2 kappa dt}) / (2 kappa): the exact one-step variance per unit sigma^2 (dt at kappa = 0)."""
    if kappa == 0.0:
        return dt
    return -math.expm1(-2.0 * kappa * dt) / (2.0 * kappa)


def _generator(rng):
    if isinstance(rng, (bool, np.bool_)):
        raise TypeError("rng must be a numpy Generator, a RandomState, numpy.random itself or an int seed")
    if isinstance(rng, (int, np.integer)):
        return np.random.default_rng(int(rng))
    for attr in ("standard_normal", "random", "poisson"):
        if not callable(getattr(rng, attr, None)):
            raise TypeError("rng must be a numpy Generator (np.random.default_rng(seed)), a RandomState, "
                            "numpy.random itself (after np.random.seed) or an int seed")
    return rng


# ----------------------------------------------------------------------------
# fit
# ----------------------------------------------------------------------------
def ou_fit(x, *, dt, method, sigma_method, resid_ddof, pair_mask=None):
    """Ornstein-Uhlenbeck (Vasicek) parameters from an OLS fit on consecutive pairs: Euler-differenced or
    exact AR(1) form (Uhlenbeck-Ornstein; Brigo-Mercurio).

    Parameters
    ----------
    x : array_like (n,) or Series, equally spaced observations (finite where used), n >= 4.
    dt : REQUIRED. Time between observations in the unit of kappa (1/252: daily data, annual
        kappa; 1/365 calendar days; 1: per period).
    method : REQUIRED.
        'euler_diff' OLS  dx_t = alpha + beta x_{t-1} + e  (dx_t = x_t - x_{t-1});
                     kappa = -beta/dt, theta = -alpha/beta (the plug-in Euler convention).
        'exact_ar1'  OLS  x_t = a + b x_{t-1} + e;
                     kappa = -ln(b)/dt, theta = a/(1 - b) (exact discretisation, b = e^{-kappa dt};
                     kappa = nan when b <= 0).
        Both regressions have the same residuals and theta; b = 1 + beta.
    sigma_method : REQUIRED.
        'euler' sigma = s / sqrt(dt).
        'exact' sigma = s sqrt(2 kappa / (1 - e^{-2 kappa dt})), the sigma whose exact one-step
                variance equals s^2 (= s sqrt(2 kappa/(1 - b^2)) for 'exact_ar1').
    resid_ddof : REQUIRED. s^2 = SSR / (m - resid_ddof), m = pairs used: 0 = MLE (SSR/m),
        1 = np.std(resid, ddof=1), 2 = the unbiased OLS s^2 with two coefficients (statsmodels scale).
    pair_mask : optional boolean array (n-1,): True keeps the pair (x_{t-1}, x_t), t = 1..n-1,
        e.g. ``years[1:] == years[:-1]`` for within-year pairs only. None uses every pair.

    Returns
    -------
    dict
        kappa, theta, sigma : floats.
        half_life : ln 2 / kappa (time units of dt); half_life_periods = ln 2 / (kappa dt) (steps).
        stationary_std : sigma / sqrt(2 kappa) (sd of the stationary law).
            half_life* and stationary_std are nan when kappa <= 0 (no mean reversion).
        intercept, slope : the OLS coefficients of the regression run (alpha, beta or a, b).
        resid_std : s; r_squared : of the regression run; nobs : pairs used m.
        resid : the m residuals (a Series on the later date of each pair when x is a Series).

    Formula
    -------
    OU sampled every dt: x_t = theta + (x_{t-1} - theta) e^{-kappa dt} + eps,
    Var eps = sigma^2 (1 - e^{-2 kappa dt}) / (2 kappa). 'exact_ar1' + 'exact' + resid_ddof=0 is
    the conditional MLE (= processes.ou_fit_ar1). 'euler_diff' is its first-order version
    (beta ~ -kappa dt, Var eps ~ sigma^2 dt); the two agree as dt -> 0.
    half_life_periods equals riskstats.ar1_half_life(x, method='ar1' | 'diff', dt=1).

    References
    ----------
    Uhlenbeck and Ornstein (1930); Vasicek (1977) J. Financial Economics 5; Brigo and Mercurio
    (2006) Sec. 3.2; Tsay (2010) Sec. 2.4 (AR(1) half-life); Chan (2013) Ch. 2.

    Example
    -------
    >>> f = ou_fit(rates, dt=1/252, method='exact_ar1', sigma_method='exact', resid_ddof=0)
    >>> f['kappa'], f['theta'], f['sigma'], f['half_life_periods']
    """
    method = _choice(method, "method", ("euler_diff", "exact_ar1"))
    sigma_method = _choice(sigma_method, "sigma_method", ("euler", "exact"))
    dt = _pos(dt, "dt")
    resid_ddof = _ddof(resid_ddof, "resid_ddof")
    index = x.index if isinstance(x, pd.Series) else None
    a = np.asarray(x, dtype=float)
    if a.ndim != 1:
        raise ValueError(f"x must be 1-D (got shape {a.shape})")
    if a.size < 3:
        raise ValueError("need at least 3 observations")
    if pair_mask is None:
        mask = np.ones(a.size - 1, dtype=bool)
    else:
        mask = np.asarray(pair_mask)
        if mask.dtype != bool or mask.shape != (a.size - 1,):
            raise ValueError(f"pair_mask must be a boolean array of length n-1 = {a.size - 1} (one per pair)")
    x0, x1 = a[:-1][mask], a[1:][mask]
    m = x0.size
    if not (np.all(np.isfinite(x0)) and np.all(np.isfinite(x1))):
        raise ValueError("x has NaN/inf in a pair that is used; drop it or mask the pair out")
    if m < 3 or m <= resid_ddof:
        raise ValueError("too few pairs for the fit")
    y = x1 - x0 if method == "euler_diff" else x1
    d0 = x0 - x0.mean()
    sxx = float(d0 @ d0)
    if sxx == 0.0:
        raise ValueError("x is constant over the pairs used; the fit is undefined")
    yc = y - y.mean()
    slope = float(d0 @ yc) / sxx
    intercept = float(y.mean() - slope * x0.mean())
    resid = y - intercept - slope * x0
    ssr = float(resid @ resid)
    s = math.sqrt(ssr / (m - resid_ddof))
    syy = float(yc @ yc)
    r2 = 1.0 - ssr / syy if syy > 0 else math.nan
    if method == "euler_diff":
        kappa = -slope / dt
        theta = -intercept / slope if slope != 0.0 else math.nan
    else:
        kappa = -math.log(slope) / dt if slope > 0.0 else math.nan
        theta = intercept / (1.0 - slope) if slope != 1.0 else math.nan
    if not math.isfinite(kappa):
        sigma = math.nan
    elif sigma_method == "euler":
        sigma = s / math.sqrt(dt)
    else:
        sigma = s / math.sqrt(_var_factor(kappa, dt))
    if math.isfinite(kappa) and kappa > 0:
        half_life = _LN2 / kappa
        stat_sd = sigma / math.sqrt(2.0 * kappa)
    else:
        half_life = stat_sd = math.nan
    if index is not None:
        resid = pd.Series(resid, index=index[1:][mask], name="resid")
    return {"kappa": float(kappa), "theta": float(theta), "sigma": float(sigma), "half_life": float(half_life),
            "half_life_periods": float(half_life / dt), "stationary_std": float(stat_sd),
            "intercept": intercept, "slope": slope, "resid_std": s, "r_squared": float(r2), "nobs": int(m),
            "resid": resid}


# ----------------------------------------------------------------------------
# simulation
# ----------------------------------------------------------------------------
def ou_simulate(x0, kappa, theta, sigma, *, dt, n_steps, n_paths, rng, scheme, record, jumps=None,
                jump_scheme=None):
    """Ornstein-Uhlenbeck path simulation, exact transition or Euler, with optional Bernoulli or
    compound-Poisson normal jumps (Glasserman; Cont-Tankov).

    Parameters
    ----------
    x0 : start value, scalar or one per path (n_paths,).
    kappa : mean-reversion speed (>= 0, per unit time); theta : long-run level; sigma : >= 0.
    dt : REQUIRED. Step length in the time unit of kappa (> 0).
    n_steps, n_paths : REQUIRED. Integers >= 1.
    rng : REQUIRED. np.random.default_rng(seed) (preferred), a np.random.RandomState, the module
        numpy.random itself (the global stream after np.random.seed(s)), or an int seed
        (= default_rng(seed)).
    scheme : REQUIRED.
        'exact'  x' = theta + (x - theta) e^{-kappa dt} + sigma sqrt((1 - e^{-2 kappa dt})/(2 kappa)) z
        'euler'  x' = x + kappa (theta - x) dt + sigma sqrt(dt) z (Euler-Maruyama)
    record : REQUIRED. Which states the result holds, per path:
        'before_advance' x_0, ..., x_{n-1}: the value USED on step t, recorded before advancing
                         (day 1 uses x_0) -> (n_paths, n_steps)
        'after_advance'  x_1, ..., x_n: the value generated by each step (x_0 excluded)
                         -> (n_paths, n_steps)
        'full'           x_0, ..., x_n -> (n_paths, n_steps + 1)
    jumps : optional dict {'lam': intensity per unit time, 'mu_j': mean, 'sigma_j': sd} (or the
        tuple (lam, mu_j, sigma_j)) of normal jump sizes J ~ N(mu_j, sigma_j^2), added to x at the
        end of each step, undamped (as in Euler); not compensated (to compensate, pass theta -
        lam mu_j / kappa).
    jump_scheme : required with jumps. 'bernoulli': one jump with probability lam dt per step
        (lam dt <= 1); 'poisson': N ~ Poisson(lam dt) jumps per step, total N mu_j + sqrt(N) sigma_j z'.

    Returns
    -------
    ndarray of float, shape (n_paths, n_steps) or (n_paths, n_steps + 1) ('full').

    Formula
    -------
    Draw order (time-major): for each step t = 0..n_steps-1, z = rng.standard_normal(n_paths);
    with jumps, then u = rng.random(n_paths) ('bernoulli', jump if u < lam dt) or
    N = rng.poisson(lam dt, n_paths) ('poisson'), then z' = rng.standard_normal(n_paths) for the
    jump sizes of every path. Without jumps this equals Z = rng.standard_normal((n_steps,
    n_paths)), row t for step t. The exact scheme has the OU law at every step: mean
    theta + (x0 - theta) e^{-kappa t}, variance sigma^2 (1 - e^{-2 kappa t})/(2 kappa) (ou_moments).

    References
    ----------
    Glasserman (2004) Sec. 3.3 (exact OU simulation); Uhlenbeck and Ornstein (1930); Cont and
    Tankov (2004) Ch. 6 (compound Poisson jumps); Clewlow and Strickland (2000).

    Example
    -------
    >>> X = ou_simulate(0.0, 2.0, 0.0, 0.3, dt=1/365, n_steps=31, n_paths=10000,
    ...                 rng=np.random.default_rng(42), scheme='exact', record='before_advance')
    """
    scheme = _choice(scheme, "scheme", ("exact", "euler"))
    record = _choice(record, "record", ("before_advance", "after_advance", "full"))
    dt = _pos(dt, "dt")
    n_steps = _count(n_steps, "n_steps")
    n_paths = _count(n_paths, "n_paths")
    kappa, theta, sigma = (float(v) for v in (kappa, theta, sigma))
    if not (math.isfinite(kappa) and kappa >= 0):
        raise ValueError("kappa must be finite and >= 0")
    if not (math.isfinite(sigma) and sigma >= 0):
        raise ValueError("sigma must be finite and >= 0")
    if not math.isfinite(theta):
        raise ValueError("theta must be finite")
    start = np.asarray(x0, dtype=float)
    if start.ndim > 1 or (start.ndim == 1 and start.shape != (n_paths,)):
        raise ValueError("x0 must be a scalar or have one value per path")
    if not np.all(np.isfinite(start)):
        raise ValueError("x0 must be finite")
    if jumps is None:
        if jump_scheme is not None:
            raise ValueError("jump_scheme given without jumps")
    else:
        jump_scheme = _choice(jump_scheme, "jump_scheme", ("bernoulli", "poisson"))
        if isinstance(jumps, dict):
            missing = {"lam", "mu_j", "sigma_j"} - set(jumps)
            if missing:
                raise ValueError(f"jumps needs the keys lam, mu_j, sigma_j (missing {sorted(missing)})")
            lam, mu_j, sigma_j = (float(jumps[k]) for k in ("lam", "mu_j", "sigma_j"))
        else:
            lam, mu_j, sigma_j = (float(v) for v in jumps)
        if not (math.isfinite(lam) and lam >= 0 and math.isfinite(mu_j) and math.isfinite(sigma_j)
                and sigma_j >= 0):
            raise ValueError("jumps: lam >= 0, mu_j finite and sigma_j >= 0 required")
        p_jump = lam * dt
        if jump_scheme == "bernoulli" and p_jump > 1.0:
            raise ValueError("'bernoulli' needs lam * dt <= 1")
    gen = _generator(rng)
    exact = scheme == "exact"
    if exact:
        decay = math.exp(-kappa * dt)
        sd = sigma * math.sqrt(_var_factor(kappa, dt))
    else:
        sd = sigma * math.sqrt(dt)
    out = np.empty((n_paths, n_steps + 1))
    out[:, 0] = start
    x = out[:, 0].copy()
    for t in range(n_steps):
        z = np.asarray(gen.standard_normal(n_paths), dtype=float)
        if exact:
            x = theta + (x - theta) * decay + sd * z
        else:
            x = x + kappa * (theta - x) * dt + sd * z
        if jumps is not None:
            if jump_scheme == "bernoulli":
                hit = np.asarray(gen.random(n_paths)) < p_jump
                zj = np.asarray(gen.standard_normal(n_paths), dtype=float)
                x = x + np.where(hit, mu_j + sigma_j * zj, 0.0)
            else:
                cnt = np.asarray(gen.poisson(p_jump, n_paths), dtype=float)
                zj = np.asarray(gen.standard_normal(n_paths), dtype=float)
                x = x + cnt * mu_j + np.sqrt(cnt) * sigma_j * zj
        out[:, t + 1] = x
    if record == "before_advance":
        return out[:, :-1]
    if record == "after_advance":
        return out[:, 1:]
    return out


# ----------------------------------------------------------------------------
# jumps
# ----------------------------------------------------------------------------
def detect_jumps(resid, *, k_sigma, std_ddof, reestimate_sigma):
    """Jump detection by a k-sigma threshold on regression residuals, with the jump-size moments and the
    diffusion residual sd (threshold jump calibration of a mean-reverting jump diffusion; Cont-Tankov).

    Parameters
    ----------
    resid : array_like (m,) or Series of residuals (e.g. ou_fit(...)['resid']), finite.
    k_sigma : REQUIRED. Threshold in standard deviations (e.g. 3.0); a jump is |resid| > k s
        (strict), s = np.std(resid, ddof=std_ddof) over ALL residuals.
    std_ddof : REQUIRED. ddof for s, for sigma_j and for sigma_resid (0 or 1).
    reestimate_sigma : REQUIRED bool. True: sigma_resid is the sd of the NON-jump residuals
        (diffusion re-estimated without jumps); False: sigma_resid = s.

    Returns
    -------
    dict
        mask : bool array (Series for Series input), True at jumps.
        n_jumps : int. jumps : the residuals flagged (array).
        mu_j, sigma_j : mean and sd (ddof=std_ddof) of the flagged residuals (nan if too few).
        resid_std : s; threshold : k s; sigma_resid : see reestimate_sigma.
    The intensity is lam = n_jumps / (m dt); the OU sigma from sigma_resid is
    sigma_resid / sqrt(dt) (Euler) or sigma_resid sqrt(2 kappa / (1 - e^{-2 kappa dt})) (exact).

    Formula
    -------
    z_t = resid_t / s; jump_t = |z_t| > k. The residual is not re-centred (OLS residuals with an
    intercept have mean 0); s itself is the usual de-meaned sd.

    References
    ----------
    Cont and Tankov (2004) Ch. 7 (threshold estimators); Clewlow and Strickland (2000) Ch. 2
    (mean-reverting jump diffusion calibration); Ait-Sahalia and Jacod (2009) (truncation).

    Example
    -------
    >>> fit = ou_fit(x, dt=1/252, method='exact_ar1', sigma_method='exact', resid_ddof=1)
    >>> j = detect_jumps(fit['resid'], k_sigma=3.0, std_ddof=1, reestimate_sigma=True)
    >>> lam = j['n_jumps'] / (fit['nobs'] / 252)
    """
    k = _pos(k_sigma, "k_sigma")
    std_ddof = _ddof(std_ddof, "std_ddof")
    if not isinstance(reestimate_sigma, (bool, np.bool_)):
        raise TypeError("reestimate_sigma must be True or False")
    index = resid.index if isinstance(resid, pd.Series) else None
    e = np.asarray(resid, dtype=float)
    if e.ndim != 1 or e.size <= std_ddof + 1:
        raise ValueError("resid must be 1-D with more than std_ddof + 1 values")
    if not np.all(np.isfinite(e)):
        raise ValueError("resid contains NaN/inf")
    s = float(np.std(e, ddof=std_ddof))
    thr = k * s
    mask = np.abs(e) > thr
    jumps = e[mask]
    nj = int(mask.sum())
    mu_j = float(jumps.mean()) if nj >= 1 else math.nan
    sigma_j = float(np.std(jumps, ddof=std_ddof)) if nj > std_ddof else math.nan
    clean = e[~mask]
    if reestimate_sigma:
        sigma_resid = float(np.std(clean, ddof=std_ddof)) if clean.size > std_ddof else math.nan
    else:
        sigma_resid = s
    return {"mask": pd.Series(mask, index=index, name="jump") if index is not None else mask,
            "n_jumps": nj, "jumps": jumps, "mu_j": mu_j, "sigma_j": sigma_j, "resid_std": s, "threshold": thr,
            "sigma_resid": sigma_resid}
