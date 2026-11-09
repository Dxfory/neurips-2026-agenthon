"""Volatility and risk-statistics building blocks: GARCH(1,1), EWMA, range and realised
volatility, Engle-Granger cointegration, AR(1) half-life and Fama-MacBeth regressions.

Conventions for the whole module:
- Returns are PER PERIOD in the units you pass (decimals or percent). Nothing is
  annualised unless a ``periods_per_year`` keyword asks for it; variances are per
  period in squared units (a percent return gives a variance in percent^2).
- Rows are observations in TIME ORDER (oldest first).
- Inputs must be finite unless a function says otherwise (``fama_macbeth`` drops
  missing asset rows period by period).
- pandas in -> labelled pandas out; numpy in -> numpy out. Two pandas inputs that
  are paired row by row must carry identical indexes (raises instead of pairing
  rows by position).

Functions
---------
garch11_fit         GARCH(1,1) maximum likelihood with the arch package's conventions
garch11_forecast    h-step GARCH(1,1) variance forecasts and their cumulative sum
ewma_vol            RiskMetrics EWMA volatility (forecast or filter timing)
ewma_cov            RiskMetrics EWMA covariance matrices
range_vol           Parkinson / Garman-Klass / Rogers-Satchell / Yang-Zhang volatility
realized_variance   sum of squared intraday returns (per day for 2-D input)
bipower_variation   Barndorff-Nielsen-Shephard jump-robust bipower variation
engle_granger       two-step Engle-Granger cointegration test (= statsmodels coint)
ar1_half_life       mean-reversion half-life from an AR(1) or a Dickey-Fuller regression
fama_macbeth        Fama-MacBeth two-pass cross-sectional regressions (FM or Newey-West se)
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import optimize, special, stats
from scipy.signal import lfilter

__all__ = [
    "garch11_fit",
    "garch11_forecast",
    "ewma_vol",
    "ewma_cov",
    "range_vol",
    "realized_variance",
    "bipower_variation",
    "engle_granger",
    "ar1_half_life",
    "fama_macbeth",
]

_LOG_2PI = math.log(2.0 * math.pi)
_LN2 = math.log(2.0)
_ARCH_LAM = 0.94     # arch: EWMA weight used by GARCH.backcast and variance_bounds
_ARCH_TAU = 75       # arch: backcast uses the first min(75, n) residuals
_MU1 = math.sqrt(2.0 / math.pi)  # E|Z| for Z ~ N(0, 1)


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _index(*objs):
    """Index shared by the pandas inputs (None if none is pandas); raises if two differ."""
    idx = None
    for o in objs:
        if isinstance(o, (pd.Series, pd.DataFrame)):
            if idx is None:
                idx = o.index
            elif not idx.equals(o.index):
                raise ValueError("pandas inputs have different indexes; align them first")
    return idx


def _vector(x, name):
    a = np.asarray(x, dtype=float)
    if a.ndim == 2 and a.shape[1] == 1:
        a = a[:, 0]
    if a.ndim != 1:
        raise ValueError(f"{name} must be 1-D (got shape {a.shape})")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} contains NaN/inf; drop or fill missing rows first")
    return a


def _panel(returns, name):
    """(2-D float array (T, N), was_1d, index, columns) of a 1-D or 2-D return input."""
    index = returns.index if isinstance(returns, (pd.Series, pd.DataFrame)) else None
    columns = returns.columns if isinstance(returns, pd.DataFrame) else None
    a = np.asarray(returns, dtype=float)
    was_1d = a.ndim == 1
    if was_1d:
        a = a[:, None]
    if a.ndim != 2:
        raise ValueError(f"{name} must be 1-D or 2-D (T, N) (got shape {a.shape})")
    if a.shape[0] == 0:
        raise ValueError(f"{name} is empty")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} contains NaN/inf; drop or fill missing rows first")
    return a, was_1d, index, columns


def _check_lam(lam):
    if isinstance(lam, bool) or not (0.0 < float(lam) < 1.0):
        raise ValueError("lam is the EWMA decay and must lie in (0, 1), e.g. 0.94 (RiskMetrics daily)")
    return float(lam)


def _check_timing(timing):
    if timing not in ("forecast", "filter"):
        raise ValueError("timing must be 'forecast' (sigma_t uses returns up to t-1) or "
                         "'filter' (sigma_t includes r_t)")
    return timing


def _is_int(v):
    return isinstance(v, (int, np.integer)) and not isinstance(v, bool)


# ----------------------------------------------------------------------------
# GARCH(1,1)
# ----------------------------------------------------------------------------
def _arch_backcast(e):
    """arch GARCH.backcast: sum_j w_j e_j^2, w_j proportional to 0.94^j, j < min(75, n)."""
    tau = min(_ARCH_TAU, e.size)
    w = _ARCH_LAM ** np.arange(tau)
    w = w / w.sum()
    return float(w @ (e[:tau] ** 2))


def _arch_var_bounds(e):
    """arch variance_bounds: per-period (lower, upper) limits on sigma^2_t.

    lower = max(EWMA_t / 1e6, var(e) / 1e8); upper = min(max(EWMA_t * 1e6, 1 + max e^2),
    1e7 (1 + max e^2)), EWMA_t the lambda = 0.94 recursion started at the backcast.
    They bind only for absurd parameter values (never at a sensible optimum)."""
    n = e.size
    e2 = e * e
    init = _arch_backcast(e)
    ew = np.empty(n)
    ew[0] = init
    if n > 1:
        ew[1:] = lfilter([1.0 - _ARCH_LAM], [1.0, -_ARCH_LAM], e2[:-1], zi=[_ARCH_LAM * init])[0]
    lower = ew / 1e6
    upper = ew * 1e6
    lower = np.where(lower < e.var() / 1e8, e.var() / 1e8, lower)
    top = 1.0 + float(e2.max())
    upper = np.where(upper < top, top, upper)
    upper = np.where(upper > 1e7 * top, 1e7 * top, upper)
    return lower, upper


def _garch11_filter(e2, omega, alpha, beta, s2_first):
    """sigma^2_0 = s2_first; sigma^2_t = omega + alpha e2_{t-1} + beta sigma^2_{t-1} (no bounds)."""
    u = np.empty(e2.size)
    u[0] = s2_first
    u[1:] = omega + alpha * e2[:-1]
    return lfilter([1.0], [1.0, -beta], u)


def _normal_ll(e2, s2):
    """Per-observation Gaussian log-likelihood -0.5 (ln 2pi + ln s2 + e^2/s2)."""
    return -0.5 * (_LOG_2PI + np.log(s2) + e2 / s2)


def _garch_loop(e, e2, omega, alpha, beta, bc, lower, upper):
    """Slow path: the recursion with arch's bounds_check and d sigma2 / d(mu, omega, alpha, beta)."""
    n = e.size
    s2 = np.empty(n)
    D = np.zeros((n, 4))
    for t in range(n):
        if t == 0:
            val = omega + (alpha + beta) * bc
            d = np.array([0.0, 1.0, bc, bc])
        else:
            val = omega + alpha * e2[t - 1] + beta * s2[t - 1]
            d = np.array([-2.0 * alpha * e[t - 1], 1.0, e2[t - 1], s2[t - 1]]) + beta * D[t - 1]
        if val < lower[t]:
            val, d = lower[t], np.zeros(4)
        elif val > upper[t]:
            if np.isinf(val):
                val, d = upper[t] + 1000.0, np.zeros(4)
            else:
                d = d / val
                val = upper[t] + math.log(val / upper[t])
        s2[t] = val
        D[t] = d
    return s2, D


class _GarchProblem:
    """GARCH(1,1) negative log-likelihood (with analytic gradient) on internally scaled data.

    theta = [mu?] + [omega, alpha, beta] + [nu?]. The backcast and the variance bounds are
    fixed from the starting residuals, as in arch."""

    def __init__(self, x, mean_const, is_t, bc, lower, upper):
        self.x, self.mean_const, self.is_t = x, mean_const, is_t
        self.bc, self.lower, self.upper = bc, lower, upper
        self.io = 1 if mean_const else 0

    def split(self, th):
        mu = th[0] if self.mean_const else 0.0
        o = self.io
        nu = th[o + 3] if self.is_t else None
        return mu, th[o], th[o + 1], th[o + 2], nu

    def sigma2(self, th, grad):
        mu, omega, alpha, beta, _ = self.split(th)
        e = self.x - mu
        e2 = e * e
        s2 = _garch11_filter(e2, omega, alpha, beta, omega + (alpha + beta) * self.bc)
        if np.any(~(s2 >= self.lower)) or np.any(~(s2 <= self.upper)):
            s2, D = _garch_loop(e, e2, omega, alpha, beta, self.bc, self.lower, self.upper)
            return e, e2, s2, D
        if not grad:
            return e, e2, s2, None
        n = e.size
        D = np.empty((n, 4))
        v = np.zeros(n)
        v[1:] = -2.0 * alpha * e[:-1]
        D[:, 0] = lfilter([1.0], [1.0, -beta], v)
        D[:, 1] = lfilter([1.0], [1.0, -beta], np.ones(n))
        v = np.empty(n)
        v[0] = self.bc
        v[1:] = e2[:-1]
        D[:, 2] = lfilter([1.0], [1.0, -beta], v)
        v = np.empty(n)
        v[0] = self.bc
        v[1:] = s2[:-1]
        D[:, 3] = lfilter([1.0], [1.0, -beta], v)
        return e, e2, s2, D

    def loglik(self, th, grad=True):
        """(per-observation log-likelihood (n,), per-observation gradient (n, k) or None, s2)."""
        _, _, _, _, nu = self.split(th)
        e, e2, s2, D = self.sigma2(th, grad)
        if self.is_t:
            if not nu > 2.0:
                return np.full(e.size, -np.inf), None, s2
            q = e2 / (s2 * (nu - 2.0))
            const = (special.gammaln(0.5 * (nu + 1.0)) - special.gammaln(0.5 * nu)
                     - 0.5 * math.log(math.pi * (nu - 2.0)))
            ll = const - 0.5 * np.log(s2) - 0.5 * (nu + 1.0) * np.log1p(q)
        else:
            ll = _normal_ll(e2, s2)
        if not grad:
            return ll, None, s2
        if self.is_t:
            dl_ds2 = -0.5 / s2 + 0.5 * (nu + 1.0) * q / ((1.0 + q) * s2)
            dl_dmu = (nu + 1.0) * e / (s2 * (nu - 2.0) * (1.0 + q))
        else:
            dl_ds2 = -0.5 * (1.0 / s2 - e2 / (s2 * s2))
            dl_dmu = e / s2
        cols = []
        if self.mean_const:
            cols.append(dl_ds2 * D[:, 0] + dl_dmu)
        cols += [dl_ds2 * D[:, 1], dl_ds2 * D[:, 2], dl_ds2 * D[:, 3]]
        if self.is_t:
            cols.append(0.5 * special.digamma(0.5 * (nu + 1.0)) - 0.5 * special.digamma(0.5 * nu)
                        - 0.5 / (nu - 2.0) - 0.5 * np.log1p(q)
                        + 0.5 * (nu + 1.0) * q / ((1.0 + q) * (nu - 2.0)))
        return ll, np.column_stack(cols), s2

    def objective(self, th):
        """Average negative log-likelihood and its gradient (for the optimiser)."""
        ll, G, _ = self.loglik(th, grad=True)
        n = ll.size
        f = -float(ll.sum()) / n
        if not np.isfinite(f) or G is None:
            return 1e10, np.zeros(th.size)
        return f, -G.sum(axis=0) / n


def garch11_fit(returns, *, mean, dist, cov_type):
    """GARCH(1,1) maximum likelihood with the arch package's conventions (Sheppard's arch_model).

    arch on decimal returns with input_scale=1 and rescale=False stops at its starting values
    and still reports convergence_flag 0. garch11_fit accepts any units and reports native
    units (same arch likelihood, fitted on internally rescaled data). With arch_garch_fit, use
    input_scale=100 and read params_native.

    Parameters
    ----------
    returns : array_like or Series, 1-D, time-ordered, finite (at least 10 values).
        Per-period returns in any unit (decimals or percent). Estimates are scale
        equivariant: the fit is done on internally rescaled data, so percent returns give
        omega x 1e4 and mu x 100 of the decimal fit, with the same alpha, beta and nu.
    mean : {'constant', 'zero'}, REQUIRED.
        'constant': r_t = mu + e_t (arch mean='Constant'); 'zero': r_t = e_t (arch 'Zero').
    dist : {'normal', 't'}, REQUIRED.
        Distribution of z_t = e_t / sigma_t: standard normal, or STANDARDISED Student t
        (unit variance, nu > 2 degrees of freedom, arch dist='t'; nu bounded to [2.05, 500]).
    cov_type : {'robust', 'classic'}, REQUIRED.
        'robust': Bollerslev-Wooldridge sandwich H^-1 J H^-1 / n (arch's default), with H the
        Hessian of the average negative log-likelihood and J = np.cov of the per-period
        scores; 'classic': H^-1 / n (inverse information).

    Returns
    -------
    dict
        params : Series ['mu' (constant mean only), 'omega', 'alpha', 'beta', 'nu' (t only)].
        se, tvalues, pvalues : Series with the same labels (p two-sided, standard normal, as arch).
        cov : DataFrame, parameter covariance.
        loglik : float, log-likelihood in the units passed (including all constants).
        aic, bic : -2 loglik + 2k, -2 loglik + k ln n.
        cond_vol : sigma_t (n,) in return units; std_resid : e_t / sigma_t; resid : e_t
            (Series with the input's index when returns is a Series).
        var_next : sigma^2_{n+1} = omega + alpha e_n^2 + beta sigma_n^2, the one-step
            forecast (feed it to garch11_forecast).
        persistence : alpha + beta; long_run_var : omega / (1 - alpha - beta) (inf if >= 1).
        backcast : the starting variance proxy; nobs : n; converged : bool.

    Model and conventions
    ---------------------
    e_t = sigma_t z_t;  sigma^2_t = omega + alpha e^2_{t-1} + beta sigma^2_{t-1};
    sigma^2_1 = omega + (alpha + beta) * backcast, backcast = sum_{j<tau} w_j e_j^2 with
    w_j proportional to 0.94^j, tau = min(75, n), e from the sample-mean (or zero) mean; the
    backcast is held fixed during estimation (arch GARCH.backcast). arch's variance bounds are
    applied. Constraints: omega > 0, alpha >= 0, beta >= 0, alpha + beta <= 1.
    Log-likelihood per period: normal -0.5 (ln 2pi + ln sigma^2_t + e_t^2 / sigma^2_t);
    t: ln G((nu+1)/2) - ln G(nu/2) - 0.5 ln(pi (nu-2)) - 0.5 ln sigma^2_t
       - (nu+1)/2 ln(1 + e_t^2 / (sigma^2_t (nu-2))).
    NOTE: Hull's textbook likelihood sets the first variance to the first squared return and
    drops the mean; its estimates differ from this arch convention in small samples.

    References
    ----------
    Bollerslev (1986) J. Econometrics 31; Bollerslev-Wooldridge (1992) Econometric Reviews 11;
    Tsay (2010) Analysis of Financial Time Series 3e, Ch. 3; Hull OFOD, "Estimating Volatilities
    and Correlations" (maximum likelihood GARCH); arch documentation (K. Sheppard).

    Example
    -------
    >>> fit = garch11_fit(returns, mean='constant', dist='t', cov_type='robust')
    >>> fit['params']['alpha'], fit['var_next']
    """
    mean = mean.lower() if isinstance(mean, str) else mean
    dist = dist.lower() if isinstance(dist, str) else dist
    if mean not in ("constant", "zero"):
        raise ValueError("mean must be 'constant' or 'zero'")
    if dist not in ("normal", "t"):
        raise ValueError("dist must be 'normal' or 't' (standardised Student t)")
    if cov_type not in ("robust", "classic"):
        raise ValueError("cov_type must be 'robust' (Bollerslev-Wooldridge) or 'classic'")
    index = returns.index if isinstance(returns, pd.Series) else None
    r = _vector(returns, "returns")
    n = r.size
    if n < 10:
        raise ValueError(f"need at least 10 returns (got {n})")
    mean_const = mean == "constant"
    is_t = dist == "t"
    mu0 = float(r.mean()) if mean_const else 0.0
    scale = math.sqrt(float(np.mean((r - mu0) ** 2)))
    if not scale > 0:
        raise ValueError("returns have zero variance around the mean")
    x = r / scale
    e0 = x - mu0 / scale
    bc = _arch_backcast(e0)
    lower, upper = _arch_var_bounds(e0)
    prob = _GarchProblem(x, mean_const, is_t, bc, lower, upper)

    # arch GARCH.starting_values: grid over alpha and alpha+beta, best Gaussian likelihood
    target = float(np.mean(e0 ** 2))
    grid = []
    for agb in (0.5, 0.7, 0.9, 0.98):
        for a in (0.01, 0.05, 0.1, 0.2):
            sv = np.array([(1.0 - agb) * target, a, agb - a])
            s2 = _garch11_filter(e0 * e0, sv[0], sv[1], sv[2], sv[0] + (sv[1] + sv[2]) * bc)
            s2 = np.clip(s2, lower, upper)
            grid.append((float(_normal_ll(e0 * e0, s2).sum()), sv, s2))
    grid.sort(key=lambda g: -g[0])
    starts = []
    for _, sv, s2 in grid[:3]:
        th = []
        if mean_const:
            th.append(mu0 / scale)
        th += list(sv)
        if is_t:
            k = stats.kurtosis(e0 / np.sqrt(s2), fisher=False)
            th.append(max((4.0 * k - 6.0) / (k - 3.0) if k > 3.75 else 12.0, 4.0))
        starts.append(np.array(th))

    io = prob.io
    bounds = ([(None, None)] if mean_const else []) + [(1e-8 * target, 10.0 * target), (0.0, 1.0),
                                                       (0.0, 1.0)]
    if is_t:
        bounds.append((2.05, 500.0))
    kpar = len(bounds)
    cjac = np.zeros(kpar)
    cjac[io + 1] = cjac[io + 2] = -1.0
    cons = [{"type": "ineq", "fun": lambda th: 1.0 - th[io + 1] - th[io + 2], "jac": lambda th: cjac}]
    best = None
    for th0 in starts:
        res = optimize.minimize(prob.objective, th0, jac=True, method="SLSQP", bounds=bounds,
                                constraints=cons, options={"ftol": 1e-13, "maxiter": 2000})
        if best is None or res.fun < best.fun:
            best = res
    th = np.asarray(best.x, dtype=float)
    ll, G, s2 = prob.loglik(th, grad=True)

    # covariance: numerical Hessian of the average negative log-likelihood (central
    # differences of the analytic gradient), scores = per-period gradients (arch compute_param_cov)
    H = np.empty((kpar, kpar))
    for i in range(kpar):
        h = 6e-6 * max(abs(th[i]), 0.1)
        tp, tm = th.copy(), th.copy()
        tp[i] += h
        tm[i] -= h
        H[:, i] = (prob.objective(tp)[1] - prob.objective(tm)[1]) / (2.0 * h)
    H = 0.5 * (H + H.T)
    Hinv = np.linalg.inv(H)
    if cov_type == "robust":
        cov = Hinv @ np.cov(G.T) @ Hinv / n
    else:
        cov = Hinv / n
    J = np.ones(kpar)
    if mean_const:
        J[0] = scale
    J[io] = scale * scale
    params = th * J
    cov = cov * np.outer(J, J)

    names = (["mu"] if mean_const else []) + ["omega", "alpha", "beta"] + (["nu"] if is_t else [])
    se = np.sqrt(np.diag(cov))
    tv = params / se
    mu = params[0] if mean_const else 0.0
    omega, alpha, beta = params[io], params[io + 1], params[io + 2]
    resid = r - mu
    sig2 = s2 * scale * scale
    cond_vol = np.sqrt(sig2)
    loglik = float(ll.sum()) - n * math.log(scale)
    pers = float(alpha + beta)
    out = {
        "params": pd.Series(params, index=names),
        "se": pd.Series(se, index=names),
        "tvalues": pd.Series(tv, index=names),
        "pvalues": pd.Series(2.0 * stats.norm.sf(np.abs(tv)), index=names),
        "cov": pd.DataFrame(cov, index=names, columns=names),
        "loglik": loglik,
        "aic": -2.0 * loglik + 2.0 * kpar,
        "bic": -2.0 * loglik + kpar * math.log(n),
        "cond_vol": cond_vol,
        "std_resid": resid / cond_vol,
        "resid": resid,
        "var_next": float(omega + alpha * resid[-1] ** 2 + beta * sig2[-1]),
        "persistence": pers,
        "long_run_var": float(omega / (1.0 - pers)) if pers < 1.0 else math.inf,
        "backcast": bc * scale * scale,
        "nobs": n,
        "converged": bool(best.success),
    }
    if index is not None:
        for key in ("cond_vol", "std_resid", "resid"):
            out[key] = pd.Series(out[key], index=index, name=key)
    return out


def garch11_forecast(omega, alpha, beta, *, var_next, horizon):
    """h-step-ahead GARCH(1,1) conditional variance forecasts and their running sum.

    Parameters
    ----------
    omega, alpha, beta : float >= 0
        GARCH(1,1) parameters (omega in squared return units per period).
    var_next : float >= 0, REQUIRED.
        sigma^2_{t+1}, the variance of the NEXT period already known at t (e.g.
        garch11_fit(...)['var_next'], or Hull's "current" daily variance estimate).
    horizon : int >= 1, REQUIRED.
        Number of periods H.

    Returns
    -------
    dict
        variance : ndarray (H,), E_t[sigma^2_{t+h}] for h = 1..H (variance[0] = var_next).
        cumulative : ndarray (H,), running sum = variance of the h-period summed return
            (zero-mean, serially uncorrelated returns); sqrt(cumulative[h-1]) is the
            h-period volatility; sqrt(cumulative[h-1] * P / h) annualises with P periods/year.
        long_run_var : omega / (1 - alpha - beta) (inf when alpha + beta >= 1).
        persistence : alpha + beta.

    Formula
    -------
    E_t[sigma^2_{t+h}] = V_L + (alpha + beta)^(h-1) (sigma^2_{t+1} - V_L),  V_L = omega/(1-alpha-beta);
    computed as (a+b)^(h-1) var_next + omega sum_{j<h-1} (a+b)^j, which also covers
    alpha + beta >= 1 (IGARCH: var_next + (h-1) omega).

    References
    ----------
    Hull OFOD, "Estimating Volatilities and Correlations" (using GARCH(1,1) to forecast
    future volatility); Tsay (2010) Sec. 3.5.

    Example
    -------
    >>> f = garch11_forecast(1.3465e-6, 0.083394, 0.910116, var_next=0.0003, horizon=10)
    >>> f['variance'][-1], np.sqrt(252 * f['cumulative'][-1] / 10)
    """
    omega, alpha, beta, v1 = (float(v) for v in (omega, alpha, beta, var_next))
    for nm, v in (("omega", omega), ("alpha", alpha), ("beta", beta), ("var_next", v1)):
        if not (math.isfinite(v) and v >= 0.0):
            raise ValueError(f"{nm} must be a finite number >= 0")
    if not _is_int(horizon) or horizon < 1:
        raise ValueError("horizon must be an int >= 1")
    p = alpha + beta
    j = np.arange(horizon, dtype=float)          # j = h - 1
    pj = p ** j
    geo = j if p == 1.0 else (1.0 - pj) / (1.0 - p)
    var = pj * v1 + omega * geo
    return {
        "variance": var,
        "cumulative": np.cumsum(var),
        "long_run_var": omega / (1.0 - p) if p < 1.0 else math.inf,
        "persistence": p,
    }


# ----------------------------------------------------------------------------
# EWMA
# ----------------------------------------------------------------------------
def _ewma_inputs(returns, lam, init, timing, demean, name="returns"):
    a, was_1d, index, columns = _panel(returns, name)
    lam = _check_lam(lam)
    _check_timing(timing)
    if demean:
        a = a - a.mean(axis=0)
    return a, was_1d, index, columns, lam


def ewma_vol(returns, *, lam, init, timing, demean=False):
    """RiskMetrics exponentially weighted (EWMA) volatility series.

    Parameters
    ----------
    returns : array_like (T,) or (T, N), Series or DataFrame; time-ordered, finite.
        Per-period returns; columns of a 2-D input are treated separately.
    lam : float in (0, 1), REQUIRED. Decay factor (RiskMetrics: 0.94 daily, 0.97 monthly).
    init : {'first_sq', 'sample_var'} or float / array (N,) >= 0, REQUIRED.
        The variance sigma^2_{-1} that precedes the first return (a VARIANCE, not a vol):
        'first_sq'   r_0^2 (Hull's start: the first estimate equals the first squared return);
        'sample_var' mean of r_t^2 over the whole sample (of the demeaned r_t if demean=True,
                     i.e. np.var(r, ddof=0)); uses the full sample (look-ahead);
        float        a given variance, e.g. 0.01**2 for a 1% volatility.
    timing : {'forecast', 'filter'}, REQUIRED.
        'forecast': sigma_t uses returns up to t-1: sigma^2_0 = init,
                    sigma^2_t = lam sigma^2_{t-1} + (1-lam) r_{t-1}^2 (Hull / RiskMetrics
                    forecast: the estimate for day t made at the end of day t-1).
        'filter':   sigma_t includes r_t: sigma^2_t = lam sigma^2_{t-1} + (1-lam) r_t^2 with
                    sigma^2_{-1} = init; filter[t] = forecast[t+1], and filter[-1] is the
                    forecast for the period AFTER the sample.
    demean : bool, default False.
        True subtracts the full-sample mean (per column) first; RiskMetrics assumes a zero mean.

    Returns
    -------
    ndarray (T,) or (T, N) of volatilities in return units per period (Series / DataFrame
    with the input's labels for pandas input). Square for variances; multiply by
    sqrt(periods per year) to annualise.

    Formula
    -------
    sigma^2_t = (1-lam) sum_{k>=0} lam^k r_{t-1-k}^2 + lam^t init  ('forecast');
    the weights of the last K returns sum to 1 - lam^K.

    References
    ----------
    J.P. Morgan/Reuters RiskMetrics Technical Document, 4th ed. (1996), Sec. 5.2;
    Hull OFOD, "Estimating Volatilities and Correlations" (EWMA); Jorion (2007) Value at Risk.

    Example
    -------
    >>> ewma_vol([0.02], lam=0.90, init=0.01**2, timing='filter')   # Hull: 1% -> 1.14%
    array([0.01140175])
    """
    a, was_1d, index, columns, lam = _ewma_inputs(returns, lam, init, timing, demean)
    x2 = a * a
    if isinstance(init, str):
        if init == "first_sq":
            v0 = x2[0].copy()
        elif init == "sample_var":
            v0 = x2.mean(axis=0)
        else:
            raise ValueError("init must be 'first_sq', 'sample_var' or a variance >= 0")
    else:
        v0 = np.broadcast_to(np.asarray(init, dtype=float), (a.shape[1],)).copy()
        if not np.all(np.isfinite(v0)) or np.any(v0 < 0):
            raise ValueError("a numeric init is a VARIANCE and must be finite and >= 0")
    filt = lfilter([1.0 - lam], [1.0, -lam], x2, axis=0, zi=(lam * v0)[None, :])[0]
    if timing == "filter":
        var = filt
    else:
        var = np.vstack([v0[None, :], filt[:-1]])
    vol = np.sqrt(var)
    if was_1d:
        vol = vol[:, 0]
        return pd.Series(vol, index=index, name=getattr(returns, "name", None)) if index is not None else vol
    if index is not None:
        return pd.DataFrame(vol, index=index, columns=columns)
    return vol


def ewma_cov(returns, *, lam, init, timing, demean=False):
    """RiskMetrics EWMA covariance matrices, one (N, N) matrix per period.

    Parameters
    ----------
    returns : array_like (T, N) or DataFrame; time-ordered, finite. Per-period returns.
    lam : float in (0, 1), REQUIRED. Decay factor (RiskMetrics 0.94 daily).
    init : {'first_sq', 'sample_var'} or array (N, N), REQUIRED.
        The covariance Sigma_{-1} before the first return: 'first_sq' r_0 r_0';
        'sample_var' mean of r_t r_t' over the sample (np.cov(..., ddof=0) when demean=True);
        an (N, N) symmetric matrix (a scalar only when N = 1).
    timing : {'forecast', 'filter'}, REQUIRED. As ewma_vol: 'forecast' Sigma_t uses returns up
        to t-1 (Sigma_0 = init); 'filter' Sigma_t includes r_t r_t'.
    demean : bool, default False. Subtract full-sample column means first.

    Returns
    -------
    ndarray (T, N, N) of covariance matrices (squared return units per period); the diagonal
    equals ewma_vol(...)**2 with the same arguments. Correlations: C / sqrt(d d').

    Formula
    -------
    Sigma_t = lam Sigma_{t-1} + (1-lam) r r'  (r = r_{t-1} for 'forecast', r_t for 'filter').

    References
    ----------
    RiskMetrics Technical Document (1996) Sec. 5.3; Hull OFOD, "Estimating Volatilities and
    Correlations" (EWMA covariance example, lam = 0.95).

    Example
    -------
    >>> S = ewma_cov(R, lam=0.94, init='first_sq', timing='forecast')
    >>> corr_last = S[-1, 0, 1] / np.sqrt(S[-1, 0, 0] * S[-1, 1, 1])
    """
    a, _, _, _, lam = _ewma_inputs(returns, lam, init, timing, demean)
    T, N = a.shape
    outer = (a[:, :, None] * a[:, None, :]).reshape(T, N * N)
    if isinstance(init, str):
        if init == "first_sq":
            v0 = outer[0].copy()
        elif init == "sample_var":
            v0 = outer.mean(axis=0)
        else:
            raise ValueError("init must be 'first_sq', 'sample_var' or an (N, N) matrix")
    else:
        m = np.asarray(init, dtype=float)
        if m.ndim == 0 and N == 1:
            m = m.reshape(1, 1)
        if m.shape != (N, N):
            raise ValueError(f"a matrix init must have shape ({N}, {N}) (got {m.shape})")
        if not np.all(np.isfinite(m)) or not np.allclose(m, m.T, rtol=0, atol=1e-15 + 1e-12 * np.abs(m).max()):
            raise ValueError("a matrix init must be finite and symmetric")
        v0 = m.reshape(N * N).copy()
    filt = lfilter([1.0 - lam], [1.0, -lam], outer, axis=0, zi=(lam * v0)[None, :])[0]
    if timing == "forecast":
        filt = np.vstack([v0[None, :], filt[:-1]])
    return filt.reshape(T, N, N)


# ----------------------------------------------------------------------------
# range and realised measures
# ----------------------------------------------------------------------------
_RANGE_ESTIMATORS = ("parkinson", "garman_klass", "rogers_satchell", "yang_zhang")


def _yz_k(n):
    return 0.34 / (1.34 + (n + 1.0) / (n - 1.0))


def range_vol(open, high, low, close, *, estimator, periods_per_year, window=None):  # noqa: A002
    """Range-based (OHLC) volatility: Parkinson, Garman-Klass, Rogers-Satchell or Yang-Zhang.

    Parameters
    ----------
    open, high, low, close : array_like (T,) or Series with identical indexes; prices > 0,
        time-ordered, with low <= min(open, close) and high >= max(open, close) in every bar.
    estimator : {'parkinson', 'garman_klass', 'rogers_satchell', 'yang_zhang'}, REQUIRED.
        With u = ln(H/O), d = ln(L/O), c = ln(C/O) per bar:
        'parkinson'       (u - d)^2 / (4 ln 2)                  (zero drift, no opening jump)
        'garman_klass'    0.5 (u - d)^2 - (2 ln 2 - 1) c^2       (GK's practical estimator;
                          ignores the overnight return)
        'rogers_satchell' u (u - c) + d (d - c)                  (unbiased under any drift)
        'yang_zhang'      s_o^2 + k s_c^2 + (1 - k) mean(RS), k = 0.34 / (1.34 + (n+1)/(n-1)),
                          s_o^2 = sample variance (ddof 1) of overnight o_t = ln(O_t / C_{t-1}),
                          s_c^2 = sample variance (ddof 1) of c_t; includes the overnight jump.
        The first three average the per-bar value over the bars used.
    periods_per_year : float > 0, REQUIRED. Annualisation factor (252 for daily bars).
    window : int or None, default None.
        None: one estimate from the whole sample. For 'yang_zhang' bar 0 only supplies C_0,
        so n = T - 1 periods are used (Yang-Zhang's day 0).
        int w: rolling estimate over the last w periods, NaN until a window is complete
        (first value at row w-1; at row w for 'yang_zhang', which needs the close before
        the window).

    Returns
    -------
    float (window=None), else ndarray (T,) (Series with the input's index for pandas input):
    ANNUALISED volatility sqrt(periods_per_year * variance per period).

    References
    ----------
    Parkinson (1980) J. Business 53; Garman-Klass (1980) J. Business 53; Rogers-Satchell
    (1991) Ann. Appl. Prob. 1; Yang-Zhang (2000) J. Business 73.

    Example
    -------
    >>> range_vol(df.Open, df.High, df.Low, df.Close, estimator='yang_zhang',
    ...           periods_per_year=252, window=21)
    """
    if estimator not in _RANGE_ESTIMATORS:
        raise ValueError(f"estimator must be one of {_RANGE_ESTIMATORS}")
    ppy = float(periods_per_year)
    if not (math.isfinite(ppy) and ppy > 0):
        raise ValueError("periods_per_year must be > 0 (e.g. 252 for daily bars)")
    index = _index(open, high, low, close)
    O, H, L, C = (_vector(v, nm) for v, nm in ((open, "open"), (high, "high"), (low, "low"),
                                               (close, "close")))
    T = O.size
    if not (H.size == T and L.size == T and C.size == T):
        raise ValueError("open, high, low and close must have the same length")
    if np.any(O <= 0) or np.any(H <= 0) or np.any(L <= 0) or np.any(C <= 0):
        raise ValueError("prices must be > 0")
    if np.any(H < np.maximum(O, C)) or np.any(L > np.minimum(O, C)):
        raise ValueError("inconsistent bars: need low <= min(open, close) and high >= max(open, close)")
    u, d, c = np.log(H / O), np.log(L / O), np.log(C / O)
    yz = estimator == "yang_zhang"
    if estimator == "parkinson":
        per = (u - d) ** 2 / (4.0 * _LN2)
    elif estimator == "garman_klass":
        per = 0.5 * (u - d) ** 2 - (2.0 * _LN2 - 1.0) * c * c
    else:
        per = u * (u - c) + d * (d - c)
    if yz:
        o = np.log(O[1:] / C[:-1])
        cc, rs = c[1:], per[1:]

    if window is None:
        if yz:
            n = T - 1
            if n < 2:
                raise ValueError("yang_zhang needs at least 3 bars (2 periods after the first close)")
            k = _yz_k(n)
            var = o.var(ddof=1) + k * cc.var(ddof=1) + (1.0 - k) * rs.mean()
        else:
            var = per.mean()
        return float(math.sqrt(ppy * var))

    if not _is_int(window) or window < (2 if yz else 1):
        raise ValueError("window must be an int >= 1 (>= 2 for yang_zhang)")
    out = np.full(T, np.nan)
    if yz:
        if T - 1 >= window:
            k = _yz_k(window)
            wv = np.lib.stride_tricks.sliding_window_view
            var = (wv(o, window).var(axis=1, ddof=1) + k * wv(cc, window).var(axis=1, ddof=1)
                   + (1.0 - k) * wv(rs, window).mean(axis=1))
            out[window:] = np.sqrt(ppy * var)
    elif T >= window:
        var = np.lib.stride_tricks.sliding_window_view(per, window).mean(axis=1)
        out[window - 1:] = np.sqrt(ppy * var)
    if index is not None:
        return pd.Series(out, index=index, name=f"{estimator}_vol")
    return out


def _intraday(returns, name, min_n):
    index = returns.index if isinstance(returns, pd.DataFrame) else None
    a = np.asarray(returns, dtype=float)
    if a.ndim not in (1, 2):
        raise ValueError(f"{name} must be 1-D (one day) or 2-D (days x intraday)")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} contains NaN/inf; drop or fill them first")
    if a.shape[-1] < min_n:
        raise ValueError(f"need at least {min_n} intraday returns per day")
    return a, index


def realized_variance(returns):
    """Realised variance: the sum of squared intraday returns.

    Parameters
    ----------
    returns : array_like (n,) for one day, or (D, n) / DataFrame with one row per day and
        n intraday returns (log returns, time-ordered) per row; finite.

    Returns
    -------
    float for 1-D input; ndarray (D,) (Series with the DataFrame's index) for 2-D input.
    Units: squared return units per day (not annualised; sqrt gives the daily realised vol).

    Formula
    -------
    RV = sum_j r_j^2. For r_j iid N(0, sigma^2/n): E[RV] = sigma^2 and Var[RV] = 2 sigma^4 / n;
    RV -> integrated variance (plus squared jumps) as n -> infinity.

    References
    ----------
    Andersen-Bollerslev-Diebold-Labys (2001) JASA 96, (2003) Econometrica 71;
    Barndorff-Nielsen-Shephard (2002) JRSS-B 64.

    Example
    -------
    >>> realized_variance(np.diff(np.log(prices_5min)))
    """
    a, index = _intraday(returns, "returns", 1)
    rv = np.sum(a * a, axis=-1)
    if a.ndim == 1:
        return float(rv)
    return pd.Series(rv, index=index, name="rv") if index is not None else rv


def bipower_variation(returns, *, scaling):
    """Barndorff-Nielsen-Shephard realised bipower variation (jump-robust integrated variance).

    Parameters
    ----------
    returns : array_like (n,) for one day, or (D, n) / DataFrame, one row per day; n >= 2
        intraday log returns per row, time-ordered, finite.
    scaling : {'none', 'n_over_n_minus_1'}, REQUIRED.
        'none': BNS (2004) definition mu1^-2 sum_{j=2..n} |r_j| |r_{j-1}| (n-1 products).
        'n_over_n_minus_1': multiply by n/(n-1) (the finite-sample correction for the missing
        product, used e.g. by Huang-Tauchen 2005); unbiased for constant volatility.

    Returns
    -------
    float for 1-D input; ndarray (D,) (Series for a DataFrame) for 2-D input; squared return
    units per day.

    Formula
    -------
    mu1 = E|Z| = sqrt(2/pi), so mu1^-2 = pi/2. With no jumps BV -> integrated variance; with
    jumps RV - BV -> sum of squared jumps. For iid N(0, s^2) returns
    E[BV_none] = (n-1) s^2 and n Var[BV] -> (pi^2/4 + pi - 3) (n s^2)^2 = 2.609 IQ.

    References
    ----------
    Barndorff-Nielsen-Shephard (2004) J. Financial Econometrics 2; (2006) J. Financial
    Econometrics 4; Huang-Tauchen (2005) J. Financial Econometrics 3.

    Example
    -------
    >>> jump_part = realized_variance(r) - bipower_variation(r, scaling='none')
    """
    if scaling not in ("none", "n_over_n_minus_1"):
        raise ValueError("scaling must be 'none' or 'n_over_n_minus_1'")
    a, index = _intraday(returns, "returns", 2)
    ab = np.abs(a)
    bv = np.sum(ab[..., 1:] * ab[..., :-1], axis=-1) / (_MU1 * _MU1)
    if scaling == "n_over_n_minus_1":
        n = a.shape[-1]
        bv = bv * n / (n - 1.0)
    if a.ndim == 1:
        return float(bv)
    return pd.Series(bv, index=index, name="bv") if index is not None else bv


# ----------------------------------------------------------------------------
# econometrics
# ----------------------------------------------------------------------------
def engle_granger(y, x, *, trend, maxlag, autolag):
    """Two-step Engle-Granger cointegration test (null: NO cointegration); matches statsmodels coint.

    Parameters
    ----------
    y : array_like (n,) or Series. The dependent I(1) series.
    x : array_like (n,) / (n, m), Series or DataFrame. The other I(1) series (m <= 11).
        If y and x are pandas their indexes must be identical.
    trend : {'c', 'ct', 'ctt', 'n'}, REQUIRED. Deterministic terms in the cointegrating
        regression: constant; constant + linear trend; + quadratic trend; none.
    maxlag : int >= 0 or None, REQUIRED keyword (None allowed). Largest ADF lag; None uses
        statsmodels' 12 (n/100)^(1/4) rule.
    autolag : {'aic', 'bic', 't-stat'} or None, REQUIRED keyword. Lag choice for the ADF
        regression on the residuals; None uses exactly maxlag lags.

    Returns
    -------
    dict
        stat : ADF t-statistic on the residuals (reject no-cointegration when stat < crit).
        pvalue : MacKinnon (1994) asymptotic p-value for N = m + 1 series.
        crit : {'1%', '5%', '10%'} MacKinnon (2010) finite-sample critical values at n - 1
               (as statsmodels; NaN for trend='n').
        usedlag : ADF lags used; nobs : observations in the ADF regression.
        beta : Series, cointegrating regression coefficients (x columns, then 'const',
               'trend', 'trend_squared'); resid : the spread y - fitted (Series for pandas).
        n_series : N = m + 1.

    Formula
    -------
    Step 1: OLS y_t = x_t' b + deterministic terms + u_t. Step 2: ADF without constant on u_t:
    du_t = rho u_{t-1} + sum_{i<=p} phi_i du_{t-i} + e_t; stat = t(rho). Critical values
    use Engle-Granger (not Dickey-Fuller) tables. Nearly collinear series (R^2 >= 1 - 100
    sqrt(eps)) give stat = -inf and p = 0, as in statsmodels.

    References
    ----------
    Engle-Granger (1987) Econometrica 55; MacKinnon (1994) JBES 12; MacKinnon (2010) Queen's
    Economics WP 1227, Table 2; Tsay (2010) Ch. 8.

    Example
    -------
    >>> eg = engle_granger(log_p1, log_p2, trend='c', maxlag=None, autolag='aic')
    >>> eg['stat'], eg['pvalue'], eg['crit']['5%']
    """
    from statsmodels.tsa.adfvalues import mackinnoncrit, mackinnonp
    from statsmodels.tsa.stattools import adfuller

    from .econ import ols

    if trend not in ("c", "ct", "ctt", "n"):
        raise ValueError("trend must be 'c', 'ct', 'ctt' or 'n'")
    if autolag is not None:
        if not isinstance(autolag, str) or autolag.lower() not in ("aic", "bic", "t-stat"):
            raise ValueError("autolag must be 'aic', 'bic', 't-stat' or None")
        autolag = autolag.lower()
    if maxlag is not None and (not _is_int(maxlag) or maxlag < 0):
        raise ValueError("maxlag must be an int >= 0 or None")
    index = _index(y, x)
    yv = _vector(y, "y")
    X = np.asarray(x, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    if X.ndim != 2 or X.shape[0] != yv.size:
        raise ValueError("x must be (n,) or (n, m) with the same n as y")
    if not np.all(np.isfinite(X)):
        raise ValueError("x contains NaN/inf; drop or fill missing rows first")
    n, m = X.shape
    if isinstance(x, pd.DataFrame):
        names = [str(c) for c in x.columns]
    elif isinstance(x, pd.Series) and x.name is not None:
        names = [str(x.name)]
    else:
        names = [f"x{i + 1}" for i in range(m)]
    cols, labels = [X], list(names)
    if trend != "n":
        cols.append(np.ones((n, 1)))
        labels.append("const")
    if trend in ("ct", "ctt"):
        tt = np.arange(1.0, n + 1.0)
        cols.append(tt[:, None])
        labels.append("trend")
        if trend == "ctt":
            cols.append((tt * tt)[:, None])
            labels.append("trend_squared")
    res = ols(yv, np.hstack(cols), add_const=False)
    resid = res["resid"]
    if res["r2"] < 1.0 - 100.0 * math.sqrt(np.finfo(float).eps):
        adf = adfuller(resid, maxlag=maxlag, autolag=autolag, regression="n")
        stat, usedlag, nobs = float(adf[0]), int(adf[2]), int(adf[3])
    else:
        stat, usedlag, nobs = -math.inf, None, None
    N = m + 1
    pvalue = float(mackinnonp(stat, regression=trend, N=N))
    if trend == "n":
        crit = [math.nan] * 3
    else:
        crit = [float(v) for v in mackinnoncrit(N=N, regression=trend, nobs=n - 1)]
    return {
        "stat": stat,
        "pvalue": pvalue,
        "crit": {"1%": crit[0], "5%": crit[1], "10%": crit[2]},
        "usedlag": usedlag,
        "nobs": nobs,
        "beta": pd.Series(np.asarray(res["params"], dtype=float), index=labels),
        "resid": pd.Series(resid, index=index, name="resid") if index is not None else resid,
        "n_series": N,
    }


def ar1_half_life(x, *, method, dt):
    """Mean-reversion half-life of a series from an AR(1) or a Dickey-Fuller-style regression.

    Parameters
    ----------
    x : array_like (n,) or Series, n >= 3, equally spaced, time-ordered, finite (e.g. a spread).
    method : {'ar1', 'diff'}, REQUIRED.
        'ar1'  OLS x_t = a + b x_{t-1} + e; half-life = dt ln(0.5) / ln|b| (exact for a
               discrete AR(1): |b|^h = 1/2; for -1 < b < 0 the deviation alternates in sign
               and its size halves after this time).
        'diff' OLS dx_t = a + lambda x_{t-1} + e (lambda = b - 1); half-life = -dt ln 2 / lambda
               (the continuous-time OU approximation popular in pairs trading; larger than
               'ar1' by the factor ln(1/b) / (1 - b)).
    dt : float > 0, REQUIRED. Time per step; the half-life is returned in the same unit
        (dt=1: periods; dt=1/252: years of trading days).

    Returns
    -------
    float. inf when there is no mean reversion (|b| >= 1 for 'ar1', lambda >= 0 for 'diff');
    0.0 when b = 0 ('ar1').

    Formula
    -------
    b = sum (x_{t-1} - m0)(x_t - m1) / sum (x_{t-1} - m0)^2 over t = 1..n-1. For an OU process
    sampled every dt, b = exp(-kappa dt) and 'ar1' gives exactly ln 2 / kappa.

    References
    ----------
    Tsay (2010) Sec. 2.4 (AR(1) half-life ln 0.5 / ln|phi_1|); Uhlenbeck-Ornstein (1930);
    Chan (2013) Algorithmic Trading Ch. 2 (lambda regression).

    Example
    -------
    >>> ar1_half_life(spread, method='ar1', dt=1)    # in trading days
    """
    if method not in ("ar1", "diff"):
        raise ValueError("method must be 'ar1' or 'diff'")
    dt = float(dt)
    if not (math.isfinite(dt) and dt > 0):
        raise ValueError("dt must be a positive time per step")
    a = _vector(x, "x")
    if a.size < 3:
        raise ValueError("need at least 3 observations")
    x0, x1 = a[:-1], a[1:]
    d0 = x0 - x0.mean()
    sxx = float(d0 @ d0)
    if sxx == 0.0:
        raise ValueError("x is constant; the half-life is undefined")
    if method == "ar1":
        b = float(d0 @ (x1 - x1.mean())) / sxx
        ab = abs(b)
        if ab >= 1.0:
            return math.inf
        if ab == 0.0:
            return 0.0
        return float(dt * math.log(0.5) / math.log(ab))
    dx = x1 - x0
    lam = float(d0 @ (dx - dx.mean())) / sxx
    if lam >= 0.0:
        return math.inf
    return float(-dt * _LN2 / lam)


def _fm_panel(y, X):
    """(Y (T, N) with NaN, Xs (T, N, K), period labels or None, regressor names or None)."""
    if isinstance(y, pd.Series) and isinstance(y.index, pd.MultiIndex):
        if y.index.nlevels != 2:
            raise ValueError("a long-format y needs a 2-level (period, asset) MultiIndex")
        if not isinstance(X, (pd.Series, pd.DataFrame)) or not X.index.equals(y.index):
            raise ValueError("with a long-format y, X must be a Series/DataFrame with the same index")
        if y.index.has_duplicates:
            raise ValueError("duplicate (period, asset) pairs in the index")
        Xf = X.to_frame() if isinstance(X, pd.Series) else X
        names = [str(c) if c is not None else f"x{i + 1}" for i, c in enumerate(Xf.columns)]
        Yw = y.astype(float).unstack(level=1).sort_index()
        mats = [Xf.iloc[:, j].astype(float).unstack(level=1).reindex(index=Yw.index,
                                                                      columns=Yw.columns)
                for j in range(Xf.shape[1])]
        return Yw.to_numpy(), np.stack([mm.to_numpy() for mm in mats], axis=2), Yw.index, names
    periods = y.index if isinstance(y, pd.DataFrame) else None
    Y = np.asarray(y, dtype=float)
    if Y.ndim != 2:
        raise ValueError("y must be (T, N) (periods x assets) or a long Series with a "
                         "(period, asset) MultiIndex")
    names = None
    if isinstance(X, dict):
        names = [str(k) for k in X]
        for v in X.values():
            if isinstance(y, pd.DataFrame) and isinstance(v, pd.DataFrame) and not (
                    v.index.equals(y.index) and v.columns.equals(y.columns)):
                raise ValueError("each X DataFrame must have y's index and columns")
        Xs = np.stack([np.asarray(v, dtype=float) for v in X.values()], axis=2)
    else:
        if isinstance(y, pd.DataFrame) and isinstance(X, pd.DataFrame):
            if not (X.index.equals(y.index) and X.columns.equals(y.columns)):
                raise ValueError("X DataFrame must have y's index and columns")
        Xs = np.asarray(X, dtype=float)
        if Xs.ndim == 2:
            Xs = Xs[:, :, None]
    if Xs.ndim != 3 or Xs.shape[:2] != Y.shape:
        raise ValueError(f"X must be (T, N) or (T, N, K) matching y {Y.shape} (got {Xs.shape})")
    return Y, Xs, periods, names


def fama_macbeth(y, X, *, add_const, se, lags=None):
    """Fama-MacBeth: one cross-sectional OLS per period, then time-series averages of the slopes.

    Parameters
    ----------
    y : returns, one of
        (T, N) array or DataFrame (periods x assets), or a long Series with a 2-level
        (period, asset) MultiIndex. NaN = asset missing that period.
    X : regressors (betas or characteristics, known when y is realised) matching y:
        (T, N) or (T, N, K) array; for a wide y a DataFrame (K = 1) or a dict {name: DataFrame};
        for a long y a Series/DataFrame with y's index. Rows with any NaN are dropped per period.
    add_const : bool, REQUIRED. True puts an intercept first in every cross-section ('const').
    se : {'fm', 'newey_west'}, REQUIRED.
        'fm': s / sqrt(T) with s the SAMPLE standard deviation (ddof = 1) of lambda_t, i.e. the
        one-sample t-test of Fama-MacBeth (1973). (Cochrane's (2005) eq. 12.25 divides by T
        instead of T - 1: multiply se by sqrt((T-1)/T).)
        'newey_west': HAC variance of the mean, [G_0 + sum_{j=1..L} (1 - j/(L+1)) (G_j + G_j')] / T
        with G_j = (1/T) sum_t d_t d_{t-j}', d_t = lambda_t - mean (econ.cov_ols kind='HAC';
        no small-sample factor). With lags=0 it equals 'fm' times sqrt((T-1)/T).
    lags : int >= 0 or {'nw1994', 'sw'}, required for 'newey_west' only (strings apply
        econ.newey_west_lags to T).

    Returns
    -------
    dict
        lambda, se, t, pvalue : mean slopes, standard errors, t = lambda/se and two-sided
            p-values from Student t with T - 1 d.o.f. (Series labelled ['const', names...]
            when regressor names are known, else ndarrays).
        cov : (k, k) covariance of the mean slopes under `se`.
        lambdas : (T_used, k) per-period slopes (DataFrame indexed by period when labelled).
        n_periods : T_used; n_obs : assets used per period; r2 : per-period R^2 (centered when
            add_const, else uncentered); r2_mean : their average.
        skipped : periods dropped because the cross-section had fewer than k usable assets or
            rank-deficient regressors; lags : Newey-West lag used (None for 'fm').

    References
    ----------
    Fama-MacBeth (1973) JPE 81; Cochrane (2005) Asset Pricing rev. ed., Ch. 12 (Sec. 12.3:
    with constant betas the FM estimate and standard errors equal the cross-sectional OLS
    ones corrected for cross-correlation); Newey-West (1987); Petersen (2009) RFS 22.

    Example
    -------
    >>> fm = fama_macbeth(ret_wide, {'beta': beta_wide, 'size': logme_wide},
    ...                   add_const=True, se='newey_west', lags=6)
    >>> fm['lambda'], fm['t']
    """
    from .econ import newey_west_lags

    if se not in ("fm", "newey_west"):
        raise ValueError("se must be 'fm' or 'newey_west'")
    if se == "fm" and lags is not None:
        raise ValueError("lags is only used with se='newey_west'")
    Y, Xs, periods, names = _fm_panel(y, X)
    T, N, K = Xs.shape
    k = K + (1 if add_const else 0)
    lam = np.full((T, k), np.nan)
    nobs = np.zeros(T, dtype=int)
    r2 = np.full(T, np.nan)
    for t in range(T):
        ok = np.isfinite(Y[t]) & np.all(np.isfinite(Xs[t]), axis=1)
        yt, Xt = Y[t, ok], Xs[t, ok]
        if add_const:
            Xt = np.column_stack([np.ones(yt.size), Xt])
        if yt.size < k or np.linalg.matrix_rank(Xt) < k:
            continue
        b = np.linalg.lstsq(Xt, yt, rcond=None)[0]
        e = yt - Xt @ b
        sst = float(((yt - yt.mean()) ** 2).sum()) if add_const else float(yt @ yt)
        lam[t] = b
        nobs[t] = yt.size
        r2[t] = 1.0 - float(e @ e) / sst if sst > 0 else np.nan
    used = np.isfinite(lam[:, 0])
    L = lam[used]
    Tu = L.shape[0]
    if Tu < 2:
        raise ValueError("fewer than 2 periods with an identified cross-sectional regression")
    mean = L.mean(axis=0)
    D = L - mean
    if se == "fm":
        cov = (D.T @ D) / (Tu - 1) / Tu
        nl = None
    else:
        if lags is None:
            raise ValueError("se='newey_west' needs lags: an int >= 0, or 'nw1994' / 'sw'")
        if isinstance(lags, str):
            nl = newey_west_lags(Tu, rule=lags)
        elif _is_int(lags) and 0 <= lags < Tu:
            nl = int(lags)
        else:
            raise ValueError(f"lags must be an int with 0 <= lags < T={Tu}, or 'nw1994' / 'sw'")
        S = D.T @ D
        for j in range(1, nl + 1):
            g = D[j:].T @ D[:-j]
            S = S + (1.0 - j / (nl + 1.0)) * (g + g.T)
        cov = S / Tu / Tu
    cov = 0.5 * (cov + cov.T)
    sev = np.sqrt(np.diag(cov))
    tv = mean / sev
    pv = 2.0 * stats.t.sf(np.abs(tv), Tu - 1)
    labels = None
    if names is not None:
        labels = (["const"] if add_const else []) + names
    per_idx = periods[used] if periods is not None else None
    if labels is not None:
        out = {"lambda": pd.Series(mean, index=labels), "se": pd.Series(sev, index=labels),
               "t": pd.Series(tv, index=labels), "pvalue": pd.Series(pv, index=labels),
               "cov": pd.DataFrame(cov, index=labels, columns=labels),
               "lambdas": pd.DataFrame(L, index=per_idx, columns=labels)}
    else:
        out = {"lambda": mean, "se": sev, "t": tv, "pvalue": pv, "cov": cov,
               "lambdas": pd.DataFrame(L, index=per_idx) if per_idx is not None else L}
    skipped = np.flatnonzero(~used)
    out.update(
        n_periods=Tu,
        n_obs=nobs[used],
        r2=r2[used],
        r2_mean=float(np.nanmean(r2[used])) if np.any(np.isfinite(r2[used])) else math.nan,
        skipped=list(periods[skipped]) if periods is not None else skipped.tolist(),
        lags=nl,
    )
    return out
