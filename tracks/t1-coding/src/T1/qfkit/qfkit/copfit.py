"""Copula fitting and selection: rank pseudo-observations, maximum likelihood (canonical MLE) or Kendall-tau
inversion fits of a bivariate Gaussian, Student-t, Clayton, Gumbel or Frank copula, AIC/BIC, and the empirical
upper and lower tail-dependence coefficients.

Conventions for the whole module:
- Pseudo-observations are ranks / (n + 1) per margin (strictly inside (0, 1)); fits take them as u of shape (n, 2).
- Families, parameters and densities are qfkit.copulas' (theta = rho for 'gaussian' and 't'); a fit maximises the
  sum of copulas.copula_logpdf, whose Frank density is evaluated stably at every theta.
- The log-likelihood is the copula's alone (margins are not modelled), so AIC/BIC count copula parameters only.
- Gumbel holds tau >= 0 only: at a negative sample tau its 'tau' fit is the independence edge theta = 1 (flagged
  boundary) and its 'mle' fit the maximiser over theta >= 1, so a loop over the families does not stop on a pair.

Functions
---------
pseudo_observations        ranks / (n + 1) per column, scipy rankdata ties
copula_fit                 'mle' (exact maximum likelihood) or 'tau' (Kendall-tau inversion; t: profile nu)
information_criteria       AIC = 2k - 2 log L, BIC = k log n - 2 log L
empirical_tail_dependence  upper P(U>q, V>q)/P(U>q) and lower P(U<=1-q, V<=1-q)/P(U<=1-q)

Ported from the team's qflib (``copulas.pseudo_observations``, ``fit_copula``, ``information_criteria``), with the
Gaussian fit made the exact maximiser (qflib returned the normal-scores correlation) and the Frank fit run on a
density that does not vanish numerically near theta = 49 (qflib's raised there).

References: Genest, Ghoudi and Rivest (1995), Biometrika 82 (pseudo-likelihood); Genest and MacKay (1986) and
Genest and Rivest (1993) (tau inversion); McNeil, Frey and Embrechts (2015) Ch. 7; Nelsen (2006) Ch. 5;
Akaike (1974), IEEE Trans. Automatic Control 19; Schwarz (1978), Annals of Statistics 6; Joe (2014) Ch. 2
(empirical tail dependence).
"""

from __future__ import annotations

import math
import numbers

import numpy as np
import pandas as pd
from scipy import optimize, stats

from .copulas import _family, copula_logpdf, copula_theta

__all__ = [
    "pseudo_observations",
    "copula_fit",
    "information_criteria",
    "empirical_tail_dependence",
]

_TIES = ("average", "min", "max", "dense", "ordinal")
_METHODS = ("mle", "tau")
_NEGATIVE_TAU = ("keep", "independence", "raise")
_NU_BOUNDS = (2.0 + 1e-6, 100.0)
_RHO_MAX = 1.0 - 1e-6


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _choice(value, name, allowed):
    if not isinstance(value, str) or value not in allowed:
        raise ValueError(f"{name} must be one of {', '.join(repr(a) for a in allowed)}, got {value!r}")
    return value


def _pairs(u):
    a = u.to_numpy(dtype=float) if isinstance(u, pd.DataFrame) else np.asarray(u, dtype=float)
    if a.ndim != 2 or a.shape[1] != 2:
        raise ValueError("u must be an (n, 2) array of paired pseudo-observations (see pseudo_observations)")
    if a.shape[0] < 3:
        raise ValueError("u needs at least 3 pairs")
    if not np.all((a > 0.0) & (a < 1.0)):
        raise ValueError("u must lie strictly inside (0, 1): pass pseudo-observations ranks/(n+1), not raw data")
    return a


def _maximise(objective, lo, hi, start=None):
    """Maximise a smooth function of one variable on [lo, hi]: a 41-point scan (plus the start), then bounded Brent
    between the neighbours of the best scan point."""
    grid = np.linspace(lo, hi, 41)
    if start is not None and lo < start < hi:
        grid = np.sort(np.append(grid, start))
    values = np.array([objective(g) for g in grid])
    values[~np.isfinite(values)] = -np.inf
    i = int(np.argmax(values))
    if not np.isfinite(values[i]):
        raise ValueError("the copula log-likelihood is not finite anywhere on the parameter range")
    a, b = grid[max(i - 1, 0)], grid[min(i + 1, grid.size - 1)]
    res = optimize.minimize_scalar(lambda t: -objective(t), bounds=(a, b), method="bounded",
                                   options={"xatol": 1e-10 * max(1.0, abs(b)), "maxiter": 500})
    if np.isfinite(res.fun) and -res.fun >= values[i]:
        return float(res.x), float(-res.fun)
    return float(grid[i]), float(values[i])


def _loglik(u, theta, fam, nu=None):
    return float(np.sum(copula_logpdf(u, theta, family=fam, nu=nu)))


def _theta_range(fam, tau):
    if fam in ("gaussian", "t"):
        return -_RHO_MAX, _RHO_MAX
    if fam == "clayton":  # theta > 0: at tau <= 0 the maximiser is at or near the independence edge 0
        return 1e-6, 100.0
    if fam == "gumbel":  # theta >= 1 at any sample tau: a negative one puts the maximiser at or near the edge 1
        return 1.0, 100.0
    return (1e-6, 100.0) if tau >= 0 else (-100.0, -1e-6)  # frank


# ----------------------------------------------------------------------------
# public
# ----------------------------------------------------------------------------
def pseudo_observations(x, *, ties):
    """Rank pseudo-observations ranks / (n + 1) of each column (the empirical-CDF transform used before fitting a
    copula), with scipy rankdata tie handling.

    Parameters
    ----------
    x : array_like (n,) or (n, d), or a DataFrame; one column per margin (e.g. each stock's returns), finite.
    ties : REQUIRED. 'average', 'min', 'max', 'dense' or 'ordinal' (ties by position), as scipy.stats.rankdata.

    Returns
    -------
    Same shape and type as x, values rank/(n + 1) strictly inside (0, 1).

    References
    ----------
    Genest, Ghoudi and Rivest (1995); McNeil, Frey and Embrechts (2015) Sec. 7.5.1.

    Example
    -------
    >>> u = pseudo_observations(returns[['AAPL', 'JPM']].dropna(), ties='average')
    """
    kind = _choice(ties, "ties", _TIES)
    a = x.to_numpy(dtype=float) if isinstance(x, (pd.DataFrame, pd.Series)) else np.asarray(x, dtype=float)
    if a.ndim not in (1, 2) or a.shape[0] < 1:
        raise ValueError("x must be 1-D or 2-D (n, d) with n >= 1")
    if not np.all(np.isfinite(a)):
        raise ValueError("x contains NaN/inf; drop incomplete rows first")
    out = stats.rankdata(a, method=kind, axis=0) / (a.shape[0] + 1.0)
    if isinstance(x, pd.DataFrame):
        return pd.DataFrame(out, index=x.index, columns=x.columns)
    if isinstance(x, pd.Series):
        return pd.Series(out, index=x.index, name=x.name)
    return out


def information_criteria(loglik, *, n, n_params):
    """Akaike (AIC) and Schwarz Bayesian (BIC) information criteria from a maximised log-likelihood; the model with
    the lowest value is preferred.

    Parameters
    ----------
    loglik : float. The maximised log-likelihood log L (a sum over observations); -inf (a fit
        that puts observations outside the model's support) gives aic = bic = +inf.
    n : REQUIRED. Number of observations, int >= 1.
    n_params : REQUIRED. Number of estimated parameters k, int >= 0.

    Returns
    -------
    dict
        aic : 2 k - 2 log L; bic : k ln(n) - 2 log L (both +inf when log L = -inf).

    References
    ----------
    Akaike (1974); Schwarz (1978).

    Example
    -------
    >>> information_criteria(fit['loglik'], n=len(u), n_params=2)
    """
    if isinstance(loglik, bool) or not isinstance(loglik, numbers.Real) or not (
            math.isfinite(float(loglik)) or float(loglik) == -math.inf):
        raise ValueError("loglik must be a finite number (or -inf)")
    if isinstance(n, bool) or not isinstance(n, numbers.Integral) or n < 1:
        raise ValueError("n must be an int >= 1")
    if isinstance(n_params, bool) or not isinstance(n_params, numbers.Integral) or n_params < 0:
        raise ValueError("n_params must be an int >= 0")
    ll = float(loglik)
    if ll == -math.inf:
        return {"aic": math.inf, "bic": math.inf}
    return {"aic": 2.0 * n_params - 2.0 * ll, "bic": n_params * math.log(n) - 2.0 * ll}


def copula_fit(u, *, family, method, nu=None, negative_tau="keep"):
    """Fit a bivariate copula (Gaussian, Student-t, Clayton, Gumbel or Frank) to pseudo-observations by exact
    maximum likelihood or by tau inversion, with its log-likelihood and information criteria.

    A family that cannot represent the sample's dependence sign never raises: the fit is returned with
    infeasible=True so that a loop over families finishes; skip such a fit when choosing a family. Gumbel (and
    Clayton under 'mle') at a negative sample tau is the independence edge (theta 1, or 0 for Clayton: loglik 0,
    boundary True). A Clayton 'tau' fit theta = 2 tau/(1 - tau) < 0 that puts pairs outside the Clayton support
    (u^-theta + v^-theta <= 1) reports loglik -inf, aic = bic = +inf and their count n_outside_support.

    Parameters
    ----------
    u : (n, 2) array or DataFrame of pseudo-observations in (0, 1) (pseudo_observations), n >= 3.
    family : REQUIRED. 'gaussian', 't', 'clayton', 'gumbel' or 'frank' (as qfkit.copula_logpdf).
    method : REQUIRED.
        'mle' theta (and nu for 't' unless fixed) maximise sum log c(u_i; theta), the canonical (pseudo) MLE;
        'tau' theta = copula_theta(sample Kendall tau-b): rho = sin(pi tau/2), Clayton 2 tau/(1 - tau),
              Gumbel 1/(1 - tau), Frank by the Debye equation.
        At a negative sample tau, which Gumbel cannot hold, a Gumbel fit is theta = 1 (independence) under 'tau'
        and the maximiser over theta >= 1 under 'mle'; a Clayton fit under 'mle' is the maximiser over theta > 0,
        theta = 0 (loglik 0) when no theta > 0 beats independence.
    nu : only for family t (None for the others). A number fixes the degrees of freedom; None estimates nu > 2
        (jointly with rho under 'mle'; by profile likelihood with rho held at its tau value under 'tau'), searched
        on (2, 100]; a sequence of values (e.g. range(3, 31)) is the grid of the profile search.
    negative_tau : method 'tau' at a negative sample tau, Clayton and Gumbel only. 'keep' (default): Clayton
        2 tau/(1 - tau), Gumbel the edge theta = 1; 'independence': Clayton theta = 0 and Gumbel theta = 1
        (loglik 0, boundary True); 'raise': ValueError.

    Returns
    -------
    dict
        family, method; theta (rho for 'gaussian'/'t'); nu (None unless 't'); loglik (copula log-likelihood at the
        fit); n_params (1, or 2 for 't' with nu estimated); aic, bic (information_criteria); n; tau (sample);
        boundary (True when a Gumbel fit is the edge theta = 1 at a negative sample tau: loglik 0, finite AIC/BIC;
        also a Clayton independence fit); n_outside_support (pairs outside the fitted copula's support, 0 unless a
        Clayton theta < 0: then loglik -inf, aic = bic = +inf); infeasible (True when the family cannot represent
        the sample's dependence sign or the fit puts pairs outside its support: skip it when choosing a family).

    Formula
    -------
    loglik = sum_i log c(u_i1, u_i2; theta); the scalar maximisations scan 41 points of the parameter range and
    polish with bounded Brent. An estimate of nu at 100 means the data are close to Gaussian.

    References
    ----------
    Genest, Ghoudi and Rivest (1995); Genest and Rivest (1993); McNeil, Frey and Embrechts (2015) Sec. 7.5.

    Example
    -------
    >>> fits = {f: copula_fit(u, family=f, method='tau') for f in ('gaussian', 't', 'clayton', 'gumbel')}
    >>> best = min(fits, key=lambda f: fits[f]['aic'])
    """
    fam = _family(family)
    how = _choice(method, "method", _METHODS)
    negative = _choice(negative_tau, "negative_tau", _NEGATIVE_TAU)
    a = _pairs(u)
    n = a.shape[0]
    tau = float(stats.kendalltau(a[:, 0], a[:, 1]).statistic)
    if not math.isfinite(tau):
        raise ValueError("Kendall's tau is undefined (a constant margin)")
    grid = None
    if fam != "t":
        if nu is not None:
            raise ValueError("nu applies only to family 't'")
    elif nu is not None and not isinstance(nu, numbers.Real):
        grid = np.asarray(list(nu), dtype=float)
        if grid.ndim != 1 or grid.size == 0 or np.any(~np.isfinite(grid)) or np.any(grid <= 0):
            raise ValueError("nu must be a number > 0, None, or a sequence of candidate values > 0")
    elif nu is not None and (isinstance(nu, bool) or not float(nu) > 0):
        raise ValueError("nu must be > 0")
    nu_free = fam == "t" and (nu is None or grid is not None)
    edge = how == "tau" and tau < 0 and fam in ("clayton", "gumbel")  # families without negative dependence
    if edge and negative == "raise":
        raise ValueError(f"sample Kendall tau {tau:.6g} < 0: a {fam} copula cannot hold negative dependence "
                         "(negative_tau='raise'); use 'keep' or 'independence', or another family")
    independence = edge and negative == "independence"
    if independence:
        theta, nu_hat, ll = (0.0 if fam == "clayton" else 1.0), None, 0.0
    elif how == "tau":
        theta = 1.0 if fam == "gumbel" and tau < 0 else float(copula_theta(tau, family=fam))  # Gumbel: tau >= 0 only
        if fam in ("gaussian", "t"):
            theta = max(-_RHO_MAX, min(_RHO_MAX, theta))
        if fam == "t" and nu_free:
            if grid is None:
                nu_hat, ll = _maximise(lambda v: _loglik(a, theta, fam, v), *_NU_BOUNDS)
            else:
                values = [_loglik(a, theta, fam, v) for v in grid]
                nu_hat, ll = float(grid[int(np.argmax(values))]), float(np.max(values))
        else:
            nu_hat = None if fam != "t" else float(nu)
            ll = _loglik(a, theta, fam, nu_hat)
    else:
        lo, hi = _theta_range(fam, tau)
        start = float(copula_theta(tau, family=fam)) if fam not in ("clayton", "gumbel") or tau > 0 else None
        if fam != "t" or not nu_free:
            nu_hat = None if fam != "t" else float(nu)
            theta, ll = _maximise(lambda t: _loglik(a, t, fam, nu_hat), lo, hi, start)
            if fam == "clayton" and tau <= 0 and ll <= 0.0:  # no theta > 0 beats independence: the edge theta = 0
                theta, ll = 0.0, 0.0
        else:
            def profile(v):
                return _maximise(lambda t: _loglik(a, t, fam, v), lo, hi, start)
            if grid is None:
                nu_hat, ll = _maximise(lambda v: profile(v)[1], *_NU_BOUNDS)
            else:
                values = [profile(v)[1] for v in grid]
                nu_hat = float(grid[int(np.argmax(values))])
            theta, ll = profile(nu_hat)
    k = 2 if nu_free else 1
    outside = 0
    if not math.isfinite(ll) and fam == "clayton" and float(theta) < 0:  # pairs outside the Clayton support
        outside = int(np.sum(a[:, 0] ** -theta + a[:, 1] ** -theta - 1.0 <= 0.0))
        ll = -math.inf if outside else ll
    crit = information_criteria(ll, n=n, n_params=k)
    wrong_sign = tau < 0 and (fam == "gumbel" or fam == "clayton" and (how == "mle" or independence))
    return {"family": fam, "method": how, "theta": float(theta), "nu": None if fam != "t" else float(nu_hat),
            "loglik": float(ll), "n_params": k, "aic": crit["aic"], "bic": crit["bic"], "n": n, "tau": tau,
            "boundary": independence or (fam == "gumbel" and tau < 0 and float(theta) == 1.0)
                        or (fam == "clayton" and how == "mle" and float(theta) == 0.0),
            "n_outside_support": outside, "infeasible": bool(wrong_sign or not math.isfinite(ll))}


def empirical_tail_dependence(u, *, q):
    """Empirical upper and lower tail-dependence coefficients of a bivariate sample at the threshold q:
    upper = P(U > q, V > q)/P(U > q), lower = P(U <= 1-q, V <= 1-q)/P(U <= 1-q) (conditioning on the first margin).

    Parameters
    ----------
    u : (n, 2) array or DataFrame of pseudo-observations (pseudo_observations) in (0, 1).
    q : REQUIRED. Threshold probability in (0.5, 1), e.g. 0.95 (lower tail at 1 - q).

    Returns
    -------
    dict
        upper, lower : the two ratios (NaN when no observation passes the first margin's threshold);
        n_upper, n_lower : the number of observations in the conditioning event.

    References
    ----------
    Joe (2014) Sec. 2.13; Frahm, Junker and Schmidt (2005), Insurance: Mathematics and Economics 37; Nelsen (2006)
    Sec. 5.4 (the limits lambda_U, lambda_L these estimate as q -> 1).

    Example
    -------
    >>> td = empirical_tail_dependence(u, q=0.95); td['upper'], td['lower']
    """
    a = _pairs(u)
    if isinstance(q, bool) or not isinstance(q, numbers.Real) or not 0.5 < float(q) < 1.0:
        raise ValueError("q must be a probability in (0.5, 1), e.g. 0.95")
    qq = float(q)
    up1 = a[:, 0] > qq
    lo1 = a[:, 0] <= 1.0 - qq
    n_up, n_lo = int(up1.sum()), int(lo1.sum())
    upper = float(np.sum(up1 & (a[:, 1] > qq)) / n_up) if n_up else math.nan
    lower = float(np.sum(lo1 & (a[:, 1] <= 1.0 - qq)) / n_lo) if n_lo else math.nan
    return {"upper": upper, "lower": lower, "n_upper": n_up, "n_lower": n_lo}
