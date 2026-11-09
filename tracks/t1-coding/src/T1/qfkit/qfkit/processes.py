"""Short-rate and jump-diffusion process building blocks.

Exact transition densities, conditional moments and bond prices for the
CIR (Cox-Ingersoll-Ross 1985) and Vasicek / Ornstein-Uhlenbeck short-rate
models; the Merton (1976) jump-diffusion return density and option price;
and a bounded multistart minimiser for maximum-likelihood fitting.

Time is in years throughout. Every function that works on sampled data takes
the sampling step ``dt`` (years per step) as a REQUIRED keyword: 1/252 for
trading days, 1/365 for calendar days, 1/12 for months, and so on.
"""

from __future__ import annotations

import numpy as np
from scipy import optimize
from scipy.special import ndtr
from scipy.stats import ncx2, norm, poisson

__all__ = [
    "cir_logpdf",
    "cir_moments",
    "cir_bond_price",
    "cir_euler_start",
    "feller_ratio",
    "ou_moments",
    "ou_fit_ar1",
    "vasicek_logpdf",
    "vasicek_bond_price",
    "poisson_truncation",
    "merton_logpdf",
    "merton_price",
    "multistart_minimize",
]

_LOG_2PI = float(np.log(2.0 * np.pi))


def _out(a):
    """Return a numpy scalar for 0-d results, the array otherwise."""
    a = np.asarray(a)
    return a[()] if a.ndim == 0 else a


def _check_dt(dt):
    dt = np.asarray(dt, dtype=float)
    if not np.all(np.isfinite(dt)) or np.any(dt <= 0):
        raise ValueError("dt must be a positive number of years per step")
    return dt


# ---------------------------------------------------------------------------
# CIR
# ---------------------------------------------------------------------------

def cir_logpdf(r_next, r_prev, kappa, theta, sigma, *, dt, density_of="rate"):
    """Exact CIR transition log-density log p(r_{t+dt} | r_t) (noncentral chi-square).

    Model: dr = kappa (theta - r) dt + sigma sqrt(r) dW.

    Parameters
    ----------
    r_next, r_prev : array_like
        Rate after and before the step (decimals, e.g. 0.03). Vectorised:
        pass ``r[1:], r[:-1]`` for a sampled path.
    kappa, theta, sigma : float or array_like
        Mean-reversion speed (1/year), long-run mean (decimal), volatility
        (per sqrt(year)). Any of them <= 0 gives -inf (convenient in MLE).
    dt : float
        REQUIRED. Years per step (1/252 trading days, 1/365 calendar, ...).
    density_of : {'rate', 'scaled'}
        'rate' (default, the textbook density of r itself, Glasserman 2003
        Sec. 3.4, Brigo-Mercurio) includes the Jacobian ln(2c).
        'scaled' is the density of 2c r_{t+dt}, i.e. without ln(2c). The two
        likelihoods differ by a parameter-dependent term, so their MLEs
        differ: never mix them.

    Returns
    -------
    ndarray
        Log-density per observation: -inf for r_next < 0, r_prev < 0 or
        invalid parameters; nan where r_next or r_prev is nan (missing data;
        drop it or use np.nansum). At r_next = 0 exactly the density is 0
        (-inf) if the Feller ratio 2 kappa theta/sigma^2 > 1, finite if = 1 and
        +inf if < 1 (the likelihood is then unbounded): drop or floor exact
        zero rates before an MLE.

    Formula
    -------
    c = 2 kappa / (sigma^2 (1 - e^{-kappa dt})), df = 4 kappa theta / sigma^2,
    nc = 2 c r_t e^{-kappa dt},
    log p = ln(2c) + ncx2.logpdf(2c r_{t+dt}, df, nc)
          = ln c - u - v + (q/2) ln(v/u) + ln I_q(2 sqrt(u v)),
    u = c r_t e^{-kappa dt}, v = c r_{t+dt}, q = df/2 - 1 (CIR 1985).
    Uses ncx2.logpdf (Bessel ive form), never log(pdf), to avoid underflow;
    where even ive underflows (extreme parameters) ln I_q is taken from its
    Debye expansion, so the value stays finite for optimizers.

    Example
    -------
    >>> ll = cir_logpdf(r[1:], r[:-1], 0.8, 0.05, 0.12, dt=1/252).sum()
    """
    if density_of not in ("rate", "scaled"):
        raise ValueError("density_of must be 'rate' or 'scaled'")
    dt = _check_dt(dt)
    r_next = np.asarray(r_next, dtype=float)
    r_prev = np.asarray(r_prev, dtype=float)
    kappa = np.asarray(kappa, dtype=float)
    theta = np.asarray(theta, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    valid = (kappa > 0) & (theta > 0) & (sigma > 0) & (r_prev >= 0)
    # Substitute harmless values where invalid, then overwrite with -inf.
    k = np.where(kappa > 0, kappa, 1.0)
    th = np.where(theta > 0, theta, 1.0)
    s = np.where(sigma > 0, sigma, 1.0)
    rp = np.where(r_prev >= 0, r_prev, 0.0)

    e = np.exp(-k * dt)
    c = 2.0 * k / (s * s * (-np.expm1(-k * dt)))
    df = 4.0 * k * th / (s * s)
    nc = 2.0 * c * rp * e
    x = 2.0 * c * r_next
    with np.errstate(divide="ignore", invalid="ignore"):
        lp = ncx2.logpdf(x, df, nc)
        # ncx2.logpdf is -inf when ive(q, sqrt(nc x)) underflows (large order q,
        # i.e. extreme parameters far from the data). Replace those entries by
        # the same formula with the Debye expansion of ln I_q, so optimizers
        # see a finite, very negative value instead of -inf.
        q = 0.5 * df - 1.0
        fix = np.isneginf(lp) & (x > 0) & (nc > 0) & (q > 0)
        if np.any(fix):
            xb, ncb, qb = np.broadcast_arrays(x, nc, q)
            xf, ncf, qf = xb[fix], ncb[fix], qb[fix]
            alt = (-np.log(2.0) - 0.5 * (xf + ncf) + 0.5 * qf * np.log(xf / ncf)
                   + _log_iv_debye(qf, np.sqrt(xf * ncf)))
            lp = np.array(np.broadcast_to(lp, fix.shape), dtype=float)
            lp[fix] = alt
        # r_{t+dt} = 0 is the boundary of the support. As x -> 0 the ncx2
        # density ~ e^{-nc/2} x^q / (2^{q+1} Gamma(q+1)): 0 for q > 0,
        # e^{-nc/2}/2 for q = 0, +inf for q < 0. scipy returns -inf for q <= 0.
        edge = np.where(q > 0, -np.inf, np.where(q < 0, np.inf, -np.log(2.0) - 0.5 * nc))
        lp = np.where(x == 0, edge, lp)
        if density_of == "rate":
            lp = lp + np.log(2.0 * c)
    lp = np.where(valid & ~np.isnan(lp), lp, -np.inf)
    lp = np.where(np.isnan(r_next) | np.isnan(r_prev), np.nan, lp)
    return _out(lp)


def _log_iv_debye(nu, z):
    """ln I_nu(z) for nu > 0 by the uniform (Debye) expansion, A&S 9.7.7.

    Relative error of the bracket is O(nu^-4); used only where ive underflows.
    """
    t = z / nu
    s = np.sqrt(1.0 + t * t)
    p = 1.0 / s
    eta = s + np.log(t / (1.0 + s))
    p2 = p * p
    u1 = p * (3.0 - 5.0 * p2) / 24.0
    u2 = p2 * (81.0 - 462.0 * p2 + 385.0 * p2 * p2) / 1152.0
    u3 = p * p2 * (30375.0 - 369603.0 * p2 + 765765.0 * p2 * p2
                   - 425425.0 * p2 * p2 * p2) / 414720.0
    return (nu * eta - 0.5 * np.log(2.0 * np.pi * nu) - 0.5 * np.log(s)
            + np.log1p(u1 / nu + u2 / nu**2 + u3 / nu**3))


def cir_moments(r_prev, kappa, theta, sigma, *, dt):
    """Conditional mean and variance of the CIR short rate after dt.

    Parameters
    ----------
    r_prev : array_like
        Current rate r_t (decimal).
    kappa, theta, sigma : float or array_like
        Mean-reversion speed (1/year), long-run mean, volatility (per sqrt(year)).
        kappa = 0 gives the limit mean = r, var = r sigma^2 dt.
    dt : float or array_like
        REQUIRED. Horizon in years.

    Returns
    -------
    (mean, var) : tuple of ndarray

    Formula
    -------
    mean = theta + (r - theta) e^{-kappa dt}
    var  = r sigma^2/kappa (e^{-kappa dt} - e^{-2 kappa dt})
           + theta sigma^2/(2 kappa) (1 - e^{-kappa dt})^2
    (evaluated as sigma^2 [r e^{-kappa dt} b + theta kappa b^2/2] with
    b = (1 - e^{-kappa dt})/kappa, which is exact and stable as kappa -> 0).

    Example
    -------
    >>> m, v = cir_moments(0.03, 0.8, 0.05, 0.12, dt=1/252)
    """
    dt = _check_dt(dt)
    r = np.asarray(r_prev, dtype=float)
    kappa = np.asarray(kappa, dtype=float)
    theta = np.asarray(theta, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    e = np.exp(-kappa * dt)
    nz = kappa != 0
    k = np.where(nz, kappa, 1.0)
    b = np.where(nz, -np.expm1(-k * dt) / k, dt)       # (1 - e^{-kappa dt})/kappa
    mean = theta + (r - theta) * e
    var = sigma**2 * (r * e * b + 0.5 * theta * kappa * b * b)
    return _out(mean), _out(var)


def cir_bond_price(r, tau, kappa, theta, sigma, *, lam=0.0, return_ab=False):
    """CIR zero-coupon bond price P = A(tau) exp(-B(tau) r).

    Parameters
    ----------
    r : array_like
        Current short rate (decimal, continuously compounded).
    tau : array_like
        Time to maturity in years (tau = 0 gives P = 1).
    kappa, theta, sigma : float
        CIR parameters (kappa > 0, theta > 0, sigma >= 0).
    lam : float, default 0.0
        CIR (1985) market-price-of-risk parameter, sign convention of CIR eq.
        (23): risk-neutral dynamics dr = [kappa theta - (kappa+lam) r] dt
        + sigma sqrt(r) dW, i.e. risk-neutral speed kappa+lam and risk-neutral
        mean kappa theta/(kappa+lam). Default 0: kappa, theta are already
        risk-neutral (pass kappa*, theta* from a source that quotes them).
    return_ab : bool, default False
        If True return (P, A, B) instead of P.

    Returns
    -------
    ndarray, or (P, A, B)
        Continuously compounded zero yield = -ln(P)/tau
        (curves.rate_from_df(P, tau, comp='continuous')).

    Formula
    -------
    h = sqrt((kappa+lam)^2 + 2 sigma^2),
    B = 2(e^{h tau} - 1) / [2h + (kappa+lam+h)(e^{h tau} - 1)],
    A = [2h e^{(kappa+lam+h) tau/2} / (2h + (kappa+lam+h)(e^{h tau} - 1))]^{2 kappa theta/sigma^2}.
    Evaluated in an overflow-free form (e^{-h tau}) and, for kappa+lam >= 0,
    a cancellation-free form of ln A that stays accurate as sigma -> 0;
    sigma = 0 uses the deterministic limit B = (1 - e^{-(kappa+lam) tau})/(kappa+lam),
    ln A = -kappa theta (tau - B)/(kappa+lam).

    Example
    -------
    >>> P = cir_bond_price(0.03, 5.0, 0.8, 0.05, 0.12)   # 0.79937
    """
    r = np.asarray(r, dtype=float)
    tau = np.asarray(tau, dtype=float)
    kappa = np.asarray(kappa, dtype=float)
    theta = np.asarray(theta, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    lam = np.asarray(lam, dtype=float)
    if np.any(tau < 0):
        raise ValueError("tau must be >= 0 (years)")

    kl = kappa + lam
    zero_sig = sigma == 0
    s = np.where(zero_sig, 1.0, sigma)
    h = np.sqrt(kl * kl + 2.0 * s * s)
    g = kl + h
    em = -np.expm1(-h * tau)                     # 1 - e^{-h tau}
    D = 2.0 * h * np.exp(-h * tau) + g * em      # (2h + g(e^{h tau}-1)) e^{-h tau}
    B = 2.0 * em / D
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        # Direct form: the bracket is O(sigma^2), so dividing by sigma^2 loses
        # ~1e-16/sigma^2 absolute accuracy (0.1 at sigma = 1e-8).
        lnA_direct = (2.0 * kappa * theta / (s * s)) * (
            np.log(2.0 * h) + 0.5 * (kl - h) * tau - np.log(D))
        # Same quantity without cancellation (valid for any sigma > 0; used for
        # kl >= 0): with h - kl = 2 sigma^2/g and D/(2h) = 1 - w,
        # w = em sigma^2/(g h) in [0, 1/2],
        # ln A = 2 kappa theta [-tau/g + em phi(w)/(g h)], phi(w) = -log1p(-w)/w.
        w = em * s * s / (g * h)
        phi = np.where(w > 0, -np.log1p(-w) / np.where(w > 0, w, 1.0), 1.0)
        lnA_stable = 2.0 * kappa * theta * (-tau / g + em * phi / (g * h))
    lnA = np.where(kl >= 0, lnA_stable, lnA_direct)

    # sigma = 0: deterministic path limit.
    with np.errstate(divide="ignore", invalid="ignore"):
        kl_safe = np.where(kl != 0, kl, 1.0)
        B0 = np.where(kl != 0, -np.expm1(-kl_safe * tau) / kl_safe, tau)
        lnA0 = np.where(kl != 0, -kappa * theta * (tau - B0) / kl_safe,
                        -0.5 * kappa * theta * tau * tau)
    B = np.where(zero_sig, B0, B)
    lnA = np.where(zero_sig, lnA0, lnA)

    P = np.exp(lnA - B * r)
    if return_ab:
        return _out(P), _out(np.exp(lnA)), _out(B)
    return _out(P)


def cir_euler_start(r, *, dt):
    """Rough CIR parameters from the Euler-discretisation regression (optimizer x0 only).

    Parameters
    ----------
    r : array_like, 1-D
        Observed short-rate path (decimals, all > 0), equally spaced.
    dt : float
        REQUIRED. Years per step.

    Returns
    -------
    dict with keys 'kappa', 'theta', 'sigma' (floats).
        Biased (Euler scheme); use only as x0 for a cir_logpdf MLE, e.g.
        with multistart_minimize (which clips x0 into the bounds, so a
        theta <= 0 from a short noisy sample is harmless there).
        Raises ValueError if the implied kappa <= 0.

    Formula
    -------
    OLS without intercept of dr/sqrt(r_t) on [1/sqrt(r_t), sqrt(r_t)] -> (a, b);
    kappa = -b/dt, theta = a/(kappa dt), sigma = std(resid, ddof=0)/sqrt(dt).

    Example
    -------
    >>> p0 = cir_euler_start(r, dt=1/252)
    >>> x0 = [p0['kappa'], p0['theta'], p0['sigma']]
    """
    dt = float(_check_dt(dt))
    r = np.asarray(r, dtype=float).ravel()
    if r.size < 3:
        raise ValueError("need at least 3 observations")
    if np.any(r[:-1] <= 0) or not np.all(np.isfinite(r)):
        raise ValueError("rates must be finite and > 0 for the Euler regression")
    sq = np.sqrt(r[:-1])
    y = np.diff(r) / sq
    X = np.column_stack([1.0 / sq, sq])
    (a, b), *_ = np.linalg.lstsq(X, y, rcond=None)
    kappa = -b / dt
    if not kappa > 0:
        raise ValueError(f"Euler regression implies kappa = {kappa:.6g} <= 0 "
                         "(no mean reversion detected)")
    theta = a / (kappa * dt)
    resid = y - X @ np.array([a, b])
    sigma = float(np.std(resid, ddof=0)) / np.sqrt(dt)
    return {"kappa": float(kappa), "theta": float(theta), "sigma": float(sigma)}


def feller_ratio(kappa, theta, sigma):
    """Feller ratio 2 kappa theta / sigma^2 (the CIR rate stays positive if >= 1).

    Parameters
    ----------
    kappa, theta, sigma : float or array_like
        CIR mean-reversion speed, long-run mean, volatility.

    Returns
    -------
    float or ndarray

    Formula
    -------
    2 kappa theta / sigma^2

    Example
    -------
    >>> feller_ratio(0.8, 0.05, 0.12)   # 5.5556
    """
    kappa = np.asarray(kappa, dtype=float)
    theta = np.asarray(theta, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    return _out(2.0 * kappa * theta / sigma**2)


# ---------------------------------------------------------------------------
# Ornstein-Uhlenbeck / Vasicek
# ---------------------------------------------------------------------------

def ou_moments(x_prev, kappa, theta, sigma, *, dt):
    """Exact Ornstein-Uhlenbeck (Vasicek) conditional mean and variance after dt.

    Model: dX = kappa (theta - X) dt + sigma dW.

    Parameters
    ----------
    x_prev : array_like
        Current value X_t.
    kappa, theta, sigma : float or array_like
        Mean-reversion speed (1/year), long-run mean, volatility (per
        sqrt(year)). kappa = 0 gives the Brownian limit var = sigma^2 dt.
    dt : float or array_like
        REQUIRED. Horizon in years.

    Returns
    -------
    (mean, var) : tuple of ndarray

    Formula
    -------
    mean = theta + (x - theta) e^{-kappa dt}
    var  = sigma^2 (1 - e^{-2 kappa dt}) / (2 kappa)

    Example
    -------
    >>> m, v = ou_moments(0.04, 2.0, 0.05, 0.02, dt=1/252)
    """
    dt = _check_dt(dt)
    x = np.asarray(x_prev, dtype=float)
    kappa = np.asarray(kappa, dtype=float)
    theta = np.asarray(theta, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    mean = theta + (x - theta) * np.exp(-kappa * dt)
    nz = kappa != 0
    k = np.where(nz, kappa, 1.0)
    var = np.where(nz, sigma**2 * (-np.expm1(-2.0 * k * dt)) / (2.0 * k),
                   sigma**2 * dt)
    return _out(mean), _out(var)


def ou_fit_ar1(x, *, dt):
    """Exact (conditional Gaussian) MLE of OU/Vasicek parameters via the AR(1) form.

    Parameters
    ----------
    x : array_like, 1-D
        Equally spaced observations of the process (e.g. short rates).
    dt : float
        REQUIRED. Years per step.

    Returns
    -------
    dict with keys 'kappa', 'theta', 'sigma', 'a', 'b', 'resid_sd' (floats).
        resid_sd uses SSR/n (MLE, n = number of transitions), not n-2.
        Raises ValueError('no mean reversion') unless 0 < b < 1.

    Formula
    -------
    x_{t+1} = a + b x_t + eps (OLS); b = e^{-kappa dt}, a = theta (1 - b),
    Var eps = sigma^2 (1 - b^2)/(2 kappa) =>
    kappa = -ln(b)/dt, theta = a/(1 - b), sigma = resid_sd sqrt(2 kappa/(1 - b^2)).
    Conditional on x_0 this is the exact MLE (maximises sum(vasicek_logpdf)).

    Example
    -------
    >>> p = ou_fit_ar1(r, dt=1/252); p['kappa'], p['theta'], p['sigma']
    """
    dt = float(_check_dt(dt))
    x = np.asarray(x, dtype=float).ravel()
    if x.size < 3:
        raise ValueError("need at least 3 observations")
    x0, x1 = x[:-1], x[1:]
    n = x0.size
    m0, m1 = x0.mean(), x1.mean()
    d0 = x0 - m0
    sxx = d0 @ d0
    if sxx == 0:
        raise ValueError("no mean reversion")
    b = (d0 @ (x1 - m1)) / sxx
    a = m1 - b * m0
    if not (0.0 < b < 1.0):
        raise ValueError("no mean reversion")
    resid = x1 - a - b * x0
    resid_sd = float(np.sqrt(resid @ resid / n))
    kappa = -np.log(b) / dt
    theta = a / (1.0 - b)
    sigma = resid_sd * np.sqrt(2.0 * kappa / (1.0 - b * b))
    return {"kappa": float(kappa), "theta": float(theta), "sigma": float(sigma),
            "a": float(a), "b": float(b), "resid_sd": resid_sd}


def vasicek_logpdf(r_next, r_prev, kappa, theta, sigma, *, dt):
    """Exact Vasicek/OU transition log-density (Gaussian with the OU moments).

    Parameters
    ----------
    r_next, r_prev : array_like
        Value after and before the step; pass ``r[1:], r[:-1]`` for a path.
    kappa, theta, sigma : float or array_like
        OU parameters (see ou_moments; kappa = 0 is the Brownian limit).
        sigma <= 0 gives -inf.
    dt : float
        REQUIRED. Years per step.

    Returns
    -------
    ndarray
        Log-density per observation; nan where r_next or r_prev is nan
        (missing data; drop it or use np.nansum).

    Formula
    -------
    norm.logpdf(r_next, mean, sqrt(var)) with (mean, var) = ou_moments(r_prev, ...).

    Example
    -------
    >>> ll = vasicek_logpdf(r[1:], r[:-1], 2.0, 0.05, 0.02, dt=1/252).sum()
    """
    sigma = np.asarray(sigma, dtype=float)
    r_next = np.asarray(r_next, dtype=float)
    r_prev = np.asarray(r_prev, dtype=float)
    ok = sigma > 0
    s = np.where(ok, sigma, 1.0)
    mean, var = ou_moments(r_prev, kappa, theta, s, dt=dt)
    lp = norm.logpdf(r_next, mean, np.sqrt(var))
    lp = np.where(ok & ~np.isnan(lp), lp, -np.inf)
    lp = np.where(np.isnan(r_next) | np.isnan(r_prev), np.nan, lp)
    return _out(lp)


def vasicek_bond_price(r, tau, kappa, theta, sigma):
    """Vasicek zero-coupon bond price P = A(tau) exp(-B(tau) r).

    Parameters
    ----------
    r : array_like
        Current short rate (decimal, continuously compounded).
    tau : array_like
        Time to maturity in years (>= 0; tau = 0 gives P = 1).
    kappa, theta, sigma : float or array_like
        kappa >= 0 (kappa = 0 gives the driftless limit B = tau,
        ln A = sigma^2 tau^3/6); theta is the RISK-NEUTRAL long-run mean
        (if a source gives a market price of risk, convert theta first);
        sigma >= 0.

    Returns
    -------
    ndarray
        Continuously compounded zero yield = -ln(P)/tau.

    Formula
    -------
    B = (1 - e^{-kappa tau})/kappa,
    ln A = (theta - sigma^2/(2 kappa^2))(B - tau) - sigma^2 B^2/(4 kappa),
    P = A e^{-B r}  (Vasicek 1977; Hull OFOD Ch. 31). Evaluated as
    ln A = theta (B - tau) + (sigma^2/2) tau^3 f(kappa tau),
    f(x) = [x - 3/2 + 2e^{-x} - e^{-2x}/2]/x^3 (f(0) = 1/3, series for small x),
    which is identical but free of the 1/kappa^2 cancellation as kappa -> 0.

    Example
    -------
    >>> vasicek_bond_price(0.1, 10.0, 0.1, 0.1, 0.02)   # 0.38046
    """
    r = np.asarray(r, dtype=float)
    tau = np.asarray(tau, dtype=float)
    kappa = np.asarray(kappa, dtype=float)
    theta = np.asarray(theta, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    if np.any(tau < 0):
        raise ValueError("tau must be >= 0 (years)")
    nz = kappa != 0
    k = np.where(nz, kappa, 1.0)
    B = np.where(nz, -np.expm1(-k * tau) / k, tau)
    lnA = theta * (B - tau) + 0.5 * sigma**2 * tau**3 * _vasicek_f(kappa * tau)
    return _out(np.exp(lnA - B * r))


# f(x) = sum_{n>=3} (-1)^n (2 - 2^{n-1}) x^{n-3} / n!  (Taylor series of _vasicek_f)
_VAS_F_COEF = np.array([(-1.0) ** n * (2.0 - 2.0 ** (n - 1)) / np.prod(np.arange(1.0, n + 1))
                        for n in range(3, 28)])


def _vasicek_f(x):
    """[x - 3/2 + 2e^{-x} - e^{-2x}/2] / x^3 = Var(int_0^tau r ds) / (sigma^2 tau^3)."""
    x = np.asarray(x, dtype=float)
    small = np.abs(x) < 0.5
    xs = np.where(small, x, 0.0)
    series = np.polynomial.polynomial.polyval(xs, _VAS_F_COEF)
    xl = np.where(small, 1.0, x)
    with np.errstate(over="ignore", invalid="ignore"):
        direct = (xl - 1.5 + 2.0 * np.exp(-xl) - 0.5 * np.exp(-2.0 * xl)) / xl**3
    return np.where(small, series, direct)


# ---------------------------------------------------------------------------
# Merton jump-diffusion
# ---------------------------------------------------------------------------

def poisson_truncation(mean, *, tol=1e-12):
    """Smallest N with P(Poisson(mean) > N) < tol, for truncating jump-count sums.

    Parameters
    ----------
    mean : float
        Poisson mean (e.g. lam * dt), >= 0.
    tol : float, default 1e-12
        Tail mass allowed (numerical accuracy, not a modelling convention).

    Returns
    -------
    int

    Formula
    -------
    min N such that poisson.sf(N, mean) < tol.

    Example
    -------
    >>> poisson_truncation(5/252)   # 5
    """
    mean = float(mean)
    tol = float(tol)
    if not (np.isfinite(mean) and mean >= 0):
        raise ValueError("mean must be finite and >= 0")
    if not (0.0 < tol < 1.0):
        raise ValueError("tol must be in (0, 1)")
    if mean == 0.0:
        return 0
    guess = poisson.isf(tol, mean)
    N = int(max(0.0, guess - 2.0)) if np.isfinite(guess) else 0
    while poisson.sf(N, mean) >= tol:
        N += 1
    while N > 0 and poisson.sf(N - 1, mean) < tol:
        N -= 1
    return N


def merton_logpdf(x, mu, sigma, lam, mu_j, sigma_j, *, dt, drift="price",
                  n_max=None, tol=1e-12):
    """Merton jump-diffusion log-return log-density over dt (truncated Poisson mixture).

    Model: log-return over dt = diffusion + sum of N ~ Poisson(lam dt) jumps,
    each jump in log price ~ N(mu_j, sigma_j^2).

    Parameters
    ----------
    x : array_like
        Log returns ln(S_{t+dt}/S_t).
    mu, sigma : float
        Drift (per year, meaning set by ``drift``) and diffusion vol (per
        sqrt(year), > 0).
    lam : float
        Jump intensity (jumps per year, >= 0). lam = 0 reduces to the normal.
    mu_j, sigma_j : float
        Mean and std of each log-price jump (sigma_j >= 0).
    dt : float
        REQUIRED. Years per observation.
    drift : {'price', 'log'}
        'price' (default, Merton 1976): mu is the expected instantaneous return
        of S, so the log drift is mu - lam kbar - sigma^2/2.
        'log': mu is the drift of log S (m_n = mu dt + n mu_j).
    n_max : int, optional
        Largest jump count kept; default poisson_truncation(lam dt, tol=tol).
    tol : float, default 1e-12
        Poisson tail mass dropped when n_max is None.

    Returns
    -------
    ndarray
        Log-density per observation; invalid parameters give -inf; nan where
        x is nan (missing data; drop it or use np.nansum).

    Formula
    -------
    kbar = e^{mu_j + sigma_j^2/2} - 1;
    m_n = (mu - lam kbar - sigma^2/2) dt + n mu_j ('price') or mu dt + n mu_j ('log');
    v_n = sigma^2 dt + n sigma_j^2;
    log p(x) = logsumexp_{n=0..N} [ln Pois(n; lam dt) + ln N(x; m_n, v_n)].
    Computed on an (n_obs x N+1) array.

    Example
    -------
    >>> nll = -merton_logpdf(ret, 0.08, 0.2, 5, -0.03, 0.06, dt=1/252).sum()
    """
    if drift not in ("price", "log"):
        raise ValueError("drift must be 'price' or 'log'")
    dt = float(_check_dt(dt))
    x = np.asarray(x, dtype=float)
    mu, sigma, lam, mu_j, sigma_j = (float(v) for v in (mu, sigma, lam, mu_j, sigma_j))
    if not (sigma > 0 and lam >= 0 and sigma_j >= 0
            and np.isfinite([mu, sigma, lam, mu_j, sigma_j]).all()):
        return _out(np.where(np.isnan(x), np.nan, -np.inf))

    lam_dt = lam * dt
    if n_max is None:
        N = poisson_truncation(lam_dt, tol=tol)
    else:
        N = int(n_max)
        if N < 0:
            raise ValueError("n_max must be >= 0")
    n = np.arange(N + 1, dtype=float)
    if lam_dt > 0:
        logw = poisson.logpmf(n, lam_dt)
    else:
        logw = np.where(n == 0, 0.0, -np.inf)

    kbar = np.expm1(mu_j + 0.5 * sigma_j**2)
    if drift == "price":
        base = (mu - lam * kbar - 0.5 * sigma**2) * dt
    else:
        base = mu * dt
    m = base + n * mu_j
    v = sigma**2 * dt + n * sigma_j**2
    coef = logw - 0.5 * (_LOG_2PI + np.log(v))      # (N+1,)

    xf = x.reshape(-1, 1)
    z = coef - (xf - m) ** 2 / (2.0 * v)            # (n_obs, N+1)
    zmax = z.max(axis=1, keepdims=True)
    safe = np.where(np.isfinite(zmax), zmax, 0.0)
    with np.errstate(divide="ignore"):
        lp = safe[:, 0] + np.log(np.exp(z - safe).sum(axis=1))
    lp = np.where(np.isfinite(zmax[:, 0]), lp, -np.inf)
    lp = np.where(np.isnan(xf[:, 0]), np.nan, lp)
    return _out(lp.reshape(x.shape))


def _bsm_price(S, K, T, r, sigma, q, is_call):
    """Black-Scholes-Merton European price (continuous r and q), broadcasting."""
    sqT = np.sqrt(T)
    vs = sigma * sqT
    d1 = (np.log(S / K) + (r - q) * T) / vs + 0.5 * vs
    d2 = d1 - vs
    dfq = np.exp(-q * T)
    dfr = np.exp(-r * T)
    call = S * dfq * ndtr(d1) - K * dfr * ndtr(d2)
    put = K * dfr * ndtr(-d2) - S * dfq * ndtr(-d1)
    return np.where(is_call, call, put)


def merton_price(S, K, T, r, sigma, lam, mu_j, sigma_j, *, kind, q=0.0,
                 n_max=None, tol=1e-12):
    """Merton (1976) jump-diffusion European option price (Poisson-weighted BSM series).

    Parameters
    ----------
    S, K : array_like
        Spot and strike.
    T : array_like
        Time to expiry in years (> 0).
    r : array_like
        Continuously compounded risk-free rate (decimal per year).
    sigma : array_like
        Diffusion volatility (per sqrt(year), > 0).
    lam, mu_j, sigma_j : float
        RISK-NEUTRAL jump intensity (per year), mean and std of the log jump.
    kind : {'call', 'put'}
        REQUIRED.
    q : array_like, default 0.0
        Continuous dividend yield.
    n_max : int, optional
        Largest jump count kept; default poisson_truncation(max(lam' T, lam T),
        tol=tol), so the dropped tail is < tol * max(S, K).
    tol : float, default 1e-12
        Poisson tail mass dropped when n_max is None.

    Returns
    -------
    ndarray

    Formula
    -------
    kbar = e^{mu_j + sigma_j^2/2} - 1; lam' = lam (1 + kbar);
    sigma_n^2 = sigma^2 + n sigma_j^2 / T; r_n = r - lam kbar + n ln(1 + kbar)/T;
    price = sum_n e^{-lam' T} (lam' T)^n / n! * BSM(S, K, T, r_n, sigma_n; q)
    (Merton 1976; Hull OFOD Ch. 27).

    Example
    -------
    >>> merton_price(100, 100, 1, 0.05, 0.2, 1, -0.1, 0.15, kind='call')  # 12.7613
    """
    k = str(kind).lower()
    if k not in ("call", "put"):
        raise ValueError("kind must be 'call' or 'put'")
    S, K, T, r, sigma, q = (np.asarray(v, dtype=float) for v in (S, K, T, r, sigma, q))
    if np.any(T <= 0):
        raise ValueError("T must be > 0 (years)")
    lam, mu_j, sigma_j = float(lam), float(mu_j), float(sigma_j)
    if lam < 0 or sigma_j < 0:
        raise ValueError("lam and sigma_j must be >= 0")

    kbar = np.expm1(mu_j + 0.5 * sigma_j**2)
    lamp = lam * (1.0 + kbar)
    lamT = lamp * T
    if n_max is None:
        # Call terms are <= S e^{-qT} with Poisson(lam' T) weights; put terms are
        # <= K e^{-r_n T}, whose weighted tail is K e^{-rT} P(Poisson(lam T) > N).
        # Truncating at the larger mean keeps both tails below tol.
        N = poisson_truncation(float(np.max(np.maximum(lamT, lam * T))), tol=tol)
    else:
        N = int(n_max)
        if N < 0:
            raise ValueError("n_max must be >= 0")

    shape = np.broadcast_shapes(S.shape, K.shape, T.shape, r.shape, sigma.shape, q.shape)
    n = np.arange(N + 1, dtype=float).reshape((N + 1,) + (1,) * len(shape))
    sig_n = np.sqrt(sigma**2 + n * sigma_j**2 / T)
    r_n = r - lam * kbar + n * np.log1p(kbar) / T
    if lam > 0:
        w = poisson.pmf(n, lamT)
    else:
        w = np.where(n == 0, 1.0, 0.0) * np.ones_like(lamT)
    prices = _bsm_price(S, K, T, r_n, sig_n, q, k == "call")
    return _out(np.sum(w * prices, axis=0).reshape(shape))


# ---------------------------------------------------------------------------
# Optimisation helper
# ---------------------------------------------------------------------------

def multistart_minimize(fun, bounds, *, n_starts=8, x0=None, seed=0,
                        method="L-BFGS-B", require_success=False, args=(),
                        options=None):
    """Run a bounded local minimiser from several starting points; return the best.

    Parameters
    ----------
    fun : callable
        Objective fun(x, *args) -> float (e.g. a negative log-likelihood).
    bounds : sequence of (low, high) or scipy.optimize.Bounds
        Box for each parameter; None or +-inf means unbounded on that side.
    n_starts : int, default 8
        Total number of starting points, x0 (if given) included.
    x0 : array_like, optional
        First start (clipped into the bounds), e.g. from cir_euler_start.
    seed : int, default 0
        Seed for np.random.default_rng; results are deterministic per seed.
    method : str, default 'L-BFGS-B'
        Passed to scipy.optimize.minimize (with bounds).
    require_success : bool, default False
        If True a start only counts when res.success is True as well.
    args : tuple
        Extra arguments for fun.
    options : dict, optional
        Passed to scipy.optimize.minimize.

    Returns
    -------
    scipy.optimize.OptimizeResult
        The best qualifying result, plus ``.all_results`` (every start's
        result, in start order), ``.n_finite`` (number of results with finite
        fun) and ``.starts`` (array of starting points). Raises RuntimeError
        if no start qualifies (never returns None).

    Formula
    -------
    Starts: x0, then uniform draws in the bounds (an infinite side is replaced
    by x0 -+ 1, or finite-bound -+ 1 / [-1, 1] without x0). Only finite
    res.fun values count. Returns argmin over starts of
    scipy.optimize.minimize(fun, start, method=method, bounds=bounds).fun.

    Example
    -------
    >>> nll = lambda p: -cir_logpdf(r[1:], r[:-1], *p, dt=1/252).sum()
    >>> res = multistart_minimize(nll, [(1e-3, 10), (1e-4, 0.5), (1e-3, 1)], x0=x0)
    >>> kappa, theta, sigma = res.x
    """
    if isinstance(bounds, optimize.Bounds):
        lb, ub = np.broadcast_arrays(np.asarray(bounds.lb, dtype=float),
                                     np.asarray(bounds.ub, dtype=float))
        pairs = list(zip(np.ravel(lb), np.ravel(ub)))
    else:
        pairs = [tuple(b) for b in bounds]
    dim = len(pairs)
    if dim == 0:
        raise ValueError("bounds must be non-empty")
    lo = np.array([-np.inf if b[0] is None else float(b[0]) for b in pairs])
    hi = np.array([np.inf if b[1] is None else float(b[1]) for b in pairs])
    if np.any(lo > hi):
        raise ValueError("each bound must satisfy low <= high")
    n_starts = int(n_starts)
    if n_starts < 1:
        raise ValueError("n_starts must be >= 1")

    if x0 is not None:
        x0 = np.asarray(x0, dtype=float).ravel()
        if x0.size != dim:       # check before clip, which would broadcast a length-1 x0
            raise ValueError(f"x0 has {x0.size} entries but bounds has {dim}")
        x0 = np.clip(x0, lo, hi)
        ref = x0
    else:
        ref = np.where(np.isfinite(lo), lo, np.where(np.isfinite(hi), hi, 0.0))
    dlo = np.where(np.isfinite(lo), lo, ref - 1.0)
    dhi = np.where(np.isfinite(hi), hi, ref + 1.0)
    dhi = np.maximum(dhi, dlo)

    rng = np.random.default_rng(seed)
    n_rand = n_starts - (1 if x0 is not None else 0)
    draws = rng.uniform(dlo, dhi, size=(n_rand, dim))
    starts = draws if x0 is None else np.vstack([x0[None, :], draws])

    scipy_bounds = list(zip(lo, hi))
    results = []
    best = None
    n_finite = 0
    first_err = None
    for st in starts:
        try:
            res = optimize.minimize(fun, st, args=args, method=method,
                                    bounds=scipy_bounds, options=options)
        except Exception as exc:  # a failing start must not kill the search
            if first_err is None:
                first_err = exc
            res = optimize.OptimizeResult(x=np.array(st), fun=np.nan, success=False,
                                          message=f"{type(exc).__name__}: {exc}")
        results.append(res)
        fval = float(np.asarray(res.fun).ravel()[0]) if np.size(res.fun) else np.nan
        if not np.isfinite(fval):
            continue
        n_finite += 1
        if require_success and not res.success:
            continue
        if best is None or fval < float(np.asarray(best.fun).ravel()[0]):
            best = res

    if best is None:
        msg = (f"multistart_minimize: no qualifying result from {n_starts} starts "
               f"({n_finite} finite, require_success={require_success})")
        raise RuntimeError(msg) from first_err
    out = optimize.OptimizeResult(dict(best))
    out.all_results = results
    out.n_finite = n_finite
    out.starts = starts
    return out
