"""Market-risk building blocks: VaR / Expected Shortfall, VaR backtests, extreme-value tails.

Sign convention for the whole module: every input and output is a LOSS
(positive number = money lost). For a return series r pass ``-r``; the loss
mean is ``-mean(r)``, the loss skewness is ``-skew(r)``, kurtosis is unchanged.
``alpha`` is always the confidence level (e.g. 0.99), never the tail
probability. Horizon scaling (e.g. sigma*sqrt(h)) is the caller's job.

Vectorised functions return a numpy float64 (a ``float`` subclass) when every
input is a scalar and an ``np.ndarray`` (broadcast shape) otherwise.

Functions
---------
var_es_historical     empirical VaR/ES with explicit quantile method and ES rule
var_es_normal         Gaussian VaR/ES
var_es_t              Student-t VaR/ES (t scale or standard deviation given)
var_cornish_fisher    Cornish-Fisher (modified) VaR
kupiec_pof            Kupiec unconditional-coverage LR test
christoffersen_counts transition counts of a 0/1 hit series
christoffersen_ind    Christoffersen independence LR test
christoffersen_cc     Christoffersen conditional-coverage LR test
gpd_fit               GPD maximum-likelihood fit to threshold excesses
pot_var_es            peaks-over-threshold VaR/ES from a GPD tail
hill_estimator        Hill tail-index estimator (xi = 1/tail index)
mean_excess           empirical mean-excess function
"""

from __future__ import annotations

import math
import warnings

import numpy as np
from scipy import optimize, special, stats

__all__ = [
    "var_es_historical",
    "var_es_normal",
    "var_es_t",
    "var_cornish_fisher",
    "kupiec_pof",
    "christoffersen_counts",
    "christoffersen_ind",
    "christoffersen_cc",
    "gpd_fit",
    "pot_var_es",
    "hill_estimator",
    "mean_excess",
]

_QUANTILE_METHODS = (
    "inverted_cdf",
    "averaged_inverted_cdf",
    "closest_observation",
    "interpolated_inverted_cdf",
    "hazen",
    "weibull",
    "linear",
    "median_unbiased",
    "normal_unbiased",
    "lower",
    "higher",
    "midpoint",
    "nearest",
)
_ES_RULES = ("ge", "gt", "k_worst")


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _out(x):
    """0-d result -> numpy float64 scalar; otherwise the ndarray itself."""
    x = np.asarray(x, dtype=float)
    return x[()] if x.ndim == 0 else x


def _sample(x, name):
    a = np.asarray(x, dtype=float).ravel()
    if a.size == 0:
        raise ValueError(f"{name} is empty")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} contains NaN/inf; drop or fill them first")
    return a


def _check_alpha(alpha):
    a = np.asarray(alpha, dtype=float)
    if np.any(~(a > 0.0) | ~(a < 1.0)):
        raise ValueError("alpha is the confidence level and must lie in (0, 1), e.g. 0.99")
    return a


def _hits01(hits):
    h = np.asarray(hits).ravel()
    if h.dtype == bool:
        return h.astype(np.int64)
    hf = h.astype(float)
    if not np.all((hf == 0.0) | (hf == 1.0)):
        raise ValueError("hits must be a 0/1 (or bool) VaR-violation indicator series")
    return hf.astype(np.int64)


# ----------------------------------------------------------------------------
# VaR / ES
# ----------------------------------------------------------------------------
def var_es_historical(losses, alpha, *, method, es_rule):
    """Historical-simulation (empirical) VaR and Expected Shortfall of a loss sample.

    Parameters
    ----------
    losses : array_like, 1-D. LOSSES, positive = loss (for returns pass ``-returns``).
        Must be finite.
    alpha : float in (0, 1). Confidence level, e.g. 0.99.
    method : str, REQUIRED. ``numpy.quantile`` method that defines VaR:
        'linear'       numpy default (Hyndman-Fan type 7), interpolates;
        'higher'       Hull's "k-th worst loss" when k = n(1-alpha) is an integer
                       (5th worst of 500 at 99%); in general the floor(n(1-alpha)+alpha)-th worst;
        'inverted_cdf' type 1, smallest l with F_n(l) >= alpha, i.e. the
                       (floor(n(1-alpha))+1)-th worst (6th worst of 500 at 99%);
        'lower', 'nearest', 'hazen', 'weibull', 'median_unbiased', ... (any numpy method).
    es_rule : str, REQUIRED. How ES averages the tail:
        'ge'      mean of losses >= VaR;
        'gt'      mean of losses >  VaR (returns VaR if there are none);
        'k_worst' mean of the k = floor(n(1-alpha)) largest losses, whatever VaR is (Hull's
                  historical ES; equals the Acerbi-Tasche 2002 estimator only when n(1-alpha)
                  is an integer, since the fractional (k+1)-th term is dropped);
                  raises ValueError if k = 0.

    Returns
    -------
    (var, es) : tuple of Python floats, both in loss units.

    Formula: VaR = np.quantile(L, alpha, method=method); ES per es_rule.

    Example
    -------
    >>> var, es = var_es_historical(-returns, 0.99, method='higher', es_rule='k_worst')
    """
    L = _sample(losses, "losses")
    a = float(_check_alpha(alpha))
    if method not in _QUANTILE_METHODS:
        raise ValueError(f"method must be one of {_QUANTILE_METHODS}")
    if es_rule not in _ES_RULES:
        raise ValueError(f"es_rule must be one of {_ES_RULES}")
    var = float(np.quantile(L, a, method=method))
    if es_rule == "ge":
        es = float(L[L >= var].mean())
    elif es_rule == "gt":
        tail = L[L > var]
        es = float(tail.mean()) if tail.size else var
    else:
        n = L.size
        # small tolerance so that e.g. 100*(1-0.95)=5.000000000000004 or 4.9999999999 -> 5
        k = int(math.floor(n * (1.0 - a) + 1e-9))
        if k < 1:
            raise ValueError(f"es_rule='k_worst' needs n(1-alpha) >= 1; got n={n}, alpha={a}")
        es = float(np.sort(L)[-k:].mean())
    return var, es


def var_es_normal(alpha, *, sigma, mu=0.0):
    """Parametric Gaussian VaR and Expected Shortfall of a loss L ~ N(mu, sigma^2).

    Parameters
    ----------
    alpha : float or array in (0, 1). Confidence level(s), e.g. 0.99.
    sigma : float or array, REQUIRED. Standard deviation of the LOSS over the horizon
        (scale it yourself, e.g. sigma_daily*sqrt(h) under iid).
    mu : float or array. Mean of the LOSS (= -mean return), default 0.

    Returns
    -------
    (var, es) : numpy float64 for scalar inputs, else arrays (broadcast).

    Formula: z = Phi^{-1}(alpha); VaR = mu + sigma z; ES = mu + sigma phi(z)/(1-alpha).
    Standard-normal values: alpha 0.99 -> (2.326348, 2.665214); 0.975 -> (1.959964, 2.337803).

    Example
    -------
    >>> var, es = var_es_normal(0.99, sigma=0.02 * np.sqrt(10), mu=-0.0005 * 10)
    """
    a = _check_alpha(alpha)
    s = np.asarray(sigma, dtype=float)
    if np.any(s < 0):
        raise ValueError("sigma must be >= 0")
    m = np.asarray(mu, dtype=float)
    z = stats.norm.ppf(a)
    var = m + s * z
    es = m + s * stats.norm.pdf(z) / (1.0 - a)
    return _out(var), _out(es)


def var_es_t(alpha, nu, *, mu=0.0, scale=None, sd=None):
    """Student-t VaR and Expected Shortfall of a location-scale loss L = mu + s*T_nu.

    Exactly one of ``scale`` or ``sd`` must be given (ValueError otherwise):
    this makes "standardised t or not" explicit.

    Parameters
    ----------
    alpha : float or array in (0, 1). Confidence level(s).
    nu : float or array > 0. Degrees of freedom (np.inf gives the normal limit).
    mu : float or array. Location (= mean if nu > 1) of the LOSS, default 0.
    scale : t scale parameter s (L = mu + s*T; sd = s*sqrt(nu/(nu-2))).
    sd : standard deviation of the loss; then s = sd*sqrt((nu-2)/nu), requires nu > 2.

    Returns
    -------
    (var, es) : numpy float64 for scalar inputs, else arrays (broadcast).
        ES is nan where nu <= 1 (tail mean infinite).

    Formula (McNeil-Frey-Embrechts QRM 2e, Sec. 2.3): q = t_nu^{-1}(alpha);
    VaR = mu + s q; ES = mu + s * g_nu(q)/(1-alpha) * (nu + q^2)/(nu - 1), g_nu = t density.

    Example
    -------
    >>> var, es = var_es_t(0.99, 4, sd=0.02)        # t with sd 2%
    >>> var, es = var_es_t(0.99, 4, scale=0.015)    # t with scale 1.5%
    """
    if (scale is None) == (sd is None):
        raise ValueError("give exactly one of scale= (t scale) or sd= (standard deviation)")
    a = _check_alpha(alpha)
    v = np.asarray(nu, dtype=float)
    if np.any(~(v > 0)):
        raise ValueError("nu must be > 0")
    if sd is not None:
        if np.any(~(v > 2)):
            raise ValueError("sd= requires nu > 2 (variance must exist); use scale= otherwise")
        # sqrt(1 - 2/nu) == sqrt((nu-2)/nu) but stays finite (-> 1) at nu = inf
        s = np.asarray(sd, dtype=float) * np.sqrt(1.0 - 2.0 / v)
    else:
        s = np.asarray(scale, dtype=float)
    if np.any(s < 0):
        raise ValueError("scale/sd must be >= 0")
    m = np.asarray(mu, dtype=float)
    q = stats.t.ppf(a, v)
    var = m + s * q
    with np.errstate(divide="ignore", invalid="ignore"):
        # (nu + q^2)/(nu - 1) written as (1 + q^2/nu)/(1 - 1/nu): -> 1 at nu = inf (normal ES)
        es_std = stats.t.pdf(q, v) / (1.0 - a) * (1.0 + q * q / v) / (1.0 - 1.0 / v)
    es = np.where(v > 1.0, m + s * es_std, np.nan)
    return _out(var), _out(es)


def var_cornish_fisher(alpha, *, sigma, skew, excess_kurt, mu=0.0):
    """Cornish-Fisher (modified) VaR: normal quantile corrected for skewness and excess kurtosis.

    Parameters
    ----------
    alpha : float or array in (0, 1). Confidence level(s).
    sigma : float or array, REQUIRED. Standard deviation of the LOSS.
    skew : float or array, REQUIRED. Skewness of the LOSS (= -skewness of returns!).
    excess_kurt : float or array, REQUIRED. EXCESS kurtosis (raw kurtosis - 3); same for
        losses and returns.
    mu : float or array. Mean of the LOSS (= -mean return), default 0.

    Returns
    -------
    var : numpy float64 for scalar inputs, else array (broadcast).

    Formula (Zangari 1996): z = Phi^{-1}(alpha), S = skew, K = excess_kurt,
    z_cf = z + (z^2-1)S/6 + (z^3-3z)K/24 - (2z^3-5z)S^2/36; VaR = mu + sigma z_cf.
    Only reliable for moderate S, K (the expansion is not monotone for large ones).

    Example
    -------
    >>> r = returns; from scipy.stats import skew, kurtosis
    >>> var = var_cornish_fisher(0.99, sigma=r.std(ddof=1), skew=-skew(r),
    ...                          excess_kurt=kurtosis(r), mu=-r.mean())
    """
    a = _check_alpha(alpha)
    z = stats.norm.ppf(a)
    S = np.asarray(skew, dtype=float)
    K = np.asarray(excess_kurt, dtype=float)
    z_cf = (z + (z * z - 1.0) * S / 6.0 + (z**3 - 3.0 * z) * K / 24.0
            - (2.0 * z**3 - 5.0 * z) * S * S / 36.0)
    var = np.asarray(mu, dtype=float) + np.asarray(sigma, dtype=float) * z_cf
    return _out(var)


# ----------------------------------------------------------------------------
# VaR backtests
# ----------------------------------------------------------------------------
def kupiec_pof(n_exceed, n_obs, p):
    """Kupiec (1995) proportion-of-failures (unconditional coverage) LR test, chi2(1).

    p is the violation probability 1 - alpha (0.01 for 99% VaR), not the confidence level.
    A p above 0.5 is read as a confidence level passed by mistake: 1 - p is used, p_used
    reports it and a RuntimeWarning says so. Only p <= 0 or p >= 1 raises ValueError.

    Parameters
    ----------
    n_exceed : int or int array. Number of VaR violations x (0 <= x <= n_obs).
    n_obs : int. Number of observations n.
    p : float in (0, 1). Expected violation probability = 1 - alpha (0.01 for 99% VaR);
        a value above 0.5 is taken as the confidence level and 1 - p is used.

    Returns
    -------
    dict: lr_uc (LR statistic), pvalue (chi2(1) survival function), rate (x/n).
        Floats for scalar input, arrays otherwise.

    Also p_used: the violation probability used (p, or 1 - p for p > 0.5).

    Formula: LR_uc = -2[(n-x)ln(1-p) + x ln p - (n-x)ln(1-x/n) - x ln(x/n)], with
    0*ln 0 = 0 (so x = 0 and x = n are valid); pvalue = chi2.sf(LR_uc, 1).
    Reject at level 5% when LR_uc > 3.841459.

    Example
    -------
    >>> kupiec_pof(5, 250, 0.01)   # {'lr_uc': 1.9568, 'pvalue': 0.1619, 'rate': 0.02}
    """
    used = _violation_probability(p, "kupiec_pof")
    return dict(_kupiec_pof(n_exceed, n_obs, used), p_used=used)


def _violation_probability(p, where):
    """The violation probability a public backtest uses: p in (0, 0.5], or 1 - p for p in (0.5, 1) (a confidence
    level passed by mistake, with a RuntimeWarning); p <= 0 or p >= 1 raises."""
    if not 0.0 < p < 1.0:
        raise ValueError(f"p is the violation probability 1 - alpha (0.01 for 99% VaR), in (0, 1); got {p}")
    if p <= 0.5:
        return float(p)
    used = float(f"{1.0 - float(p):.15g}")  # 1 - 0.99 is 0.01, not 0.010000000000000009
    try:  # a note for the run log; never an exception, even under warnings-as-errors
        warnings.warn(f"{where}: interpreted confidence level {p:g} as violation probability {used:g}",
                      RuntimeWarning, stacklevel=3)
    except Warning:
        pass
    return used


def _kupiec_pof(n_exceed, n_obs, p):
    """kupiec_pof for any p in (0, 1) (rolling_var_backtest passes 1 - alpha for every alpha it accepts)."""
    x = np.asarray(n_exceed, dtype=float)
    n = float(n_obs)
    if n <= 0 or n != int(n):
        raise ValueError("n_obs must be a positive integer")
    if np.any((x < 0) | (x > n) | (x != np.round(x))):
        raise ValueError("n_exceed must be integers in [0, n_obs]")
    if not 0.0 < p < 1.0:
        raise ValueError("p is the violation probability 1-alpha, in (0, 1)")
    rate = x / n
    ll0 = special.xlogy(n - x, 1.0 - p) + special.xlogy(x, p)
    ll1 = special.xlogy(n - x, 1.0 - rate) + special.xlogy(x, rate)
    lr = np.maximum(-2.0 * (ll0 - ll1), 0.0)
    return {"lr_uc": _out(lr), "pvalue": _out(stats.chi2.sf(lr, 1)), "rate": _out(rate)}


def christoffersen_counts(hits):
    """Transition counts n00, n01, n10, n11 of a 0/1 VaR-violation series.

    Parameters
    ----------
    hits : array_like of 0/1 or bool, length T. 1 = VaR violated at t.

    Returns
    -------
    dict of Python ints n00, n01, n10, n11 with n_ij = #{t >= 2: I_{t-1} = i, I_t = j};
    they sum to T - 1.

    Example
    -------
    >>> christoffersen_counts((losses > var_forecasts).astype(int))
    """
    h = _hits01(hits)
    prev, cur = h[:-1], h[1:]
    n11 = int(np.sum(prev & cur))
    n10 = int(np.sum(prev & (1 - cur)))
    n01 = int(np.sum((1 - prev) & cur))
    n00 = int(prev.size - n11 - n10 - n01)
    return {"n00": n00, "n01": n01, "n10": n10, "n11": n11}


def _counts_from(hits, counts):
    if (hits is None) == (counts is None):
        raise ValueError("give exactly one of hits or counts")
    if hits is not None:
        c = christoffersen_counts(hits)
        return c["n00"], c["n01"], c["n10"], c["n11"]
    if isinstance(counts, dict):
        vals = (counts["n00"], counts["n01"], counts["n10"], counts["n11"])
    else:
        vals = tuple(counts)
        if len(vals) != 4:
            raise ValueError("counts must be (n00, n01, n10, n11)")
    if not all(float(v) == int(v) for v in vals):       # int() would silently truncate 12.7
        raise ValueError("counts must be integers")
    vals = tuple(int(v) for v in vals)
    if min(vals) < 0:
        raise ValueError("counts must be >= 0")
    return vals


def _ind_stats(n00, n01, n10, n11):
    tot = n00 + n01 + n10 + n11
    if tot == 0:
        raise ValueError("need at least 2 observations (one transition)")
    pi0 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0.0
    pi1 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0
    pi = (n01 + n11) / tot
    xl = special.xlogy
    ll0 = xl(n00 + n10, 1.0 - pi) + xl(n01 + n11, pi)
    ll1 = xl(n00, 1.0 - pi0) + xl(n01, pi0) + xl(n10, 1.0 - pi1) + xl(n11, pi1)
    return float(ll0), float(ll1), pi0, pi1, pi


def christoffersen_ind(hits=None, *, counts=None):
    """Christoffersen (1998) independence LR test of a VaR-violation series, chi2(1).

    Tests H0: P(I_t=1 | I_{t-1}=0) = P(I_t=1 | I_{t-1}=1) against a first-order Markov chain.

    Parameters
    ----------
    hits : array_like of 0/1 or bool, optional. Violation series.
    counts : dict {'n00','n01','n10','n11'} or 4-tuple (n00, n01, n10, n11), optional.
        Give exactly one of hits / counts.

    Returns
    -------
    dict: lr_ind, pvalue (chi2(1) sf), pi0, pi1, pi, n00, n01, n10, n11.

    Formula: pi0 = n01/(n00+n01), pi1 = n11/(n10+n11) (0 if that row is empty),
    pi = (n01+n11)/(n00+n01+n10+n11);
    lnL0 = (n00+n10)ln(1-pi) + (n01+n11)ln pi;
    lnL1 = n00 ln(1-pi0) + n01 ln pi0 + n10 ln(1-pi1) + n11 ln pi1  (0*ln0 = 0);
    LR_ind = -2(lnL0 - lnL1).

    Example
    -------
    >>> christoffersen_ind(hits)['pvalue']
    >>> christoffersen_ind(counts=(12, 3, 3, 1))['lr_ind']   # 0.046066
    """
    n00, n01, n10, n11 = _counts_from(hits, counts)
    ll0, ll1, pi0, pi1, pi = _ind_stats(n00, n01, n10, n11)
    lr = max(-2.0 * (ll0 - ll1), 0.0)
    return {"lr_ind": lr, "pvalue": float(stats.chi2.sf(lr, 1)), "pi0": pi0, "pi1": pi1,
            "pi": pi, "n00": n00, "n01": n01, "n10": n10, "n11": n11}


def christoffersen_cc(hits, p, *, uc_sample="full"):
    """Christoffersen (1998) conditional-coverage test LR_cc = LR_uc + LR_ind, chi2(2).

    p is the violation probability 1 - alpha (0.01 for 99% VaR), not the confidence level.
    A p above 0.5 is read as a confidence level passed by mistake: 1 - p is used, p_used
    reports it and a RuntimeWarning says so. Only p <= 0 or p >= 1 raises ValueError.

    Parameters
    ----------
    hits : array_like of 0/1 or bool, length T. Violation series.
    p : float in (0, 1). Expected violation probability = 1 - alpha; a value above 0.5 is
        taken as the confidence level and 1 - p is used.
    uc_sample : 'full' (default; Kupiec LR_uc on all T hits, the common practice) or
        'transitions' (Kupiec on hits[1:], T-1 obs; then LR_cc equals exactly the direct
        statistic -2[lnL(p) - lnL(pi0, pi1)] on the transition counts).

    Returns
    -------
    dict: lr_uc, lr_ind, lr_cc, p_uc (chi2(1)), p_ind (chi2(1)), p_cc (chi2(2)).

    Also p_used: the violation probability used (p, or 1 - p for p > 0.5).

    Formula: LR_cc = LR_uc + LR_ind ~ chi2(2) under correct conditional coverage.

    Example
    -------
    >>> christoffersen_cc(losses > var_fc, 0.01)['p_cc']
    """
    used = _violation_probability(p, "christoffersen_cc")
    return dict(_christoffersen_cc(hits, used, uc_sample=uc_sample), p_used=used)


def _christoffersen_cc(hits, p, *, uc_sample="full"):
    """christoffersen_cc for any p in (0, 1) (rolling_var_backtest passes 1 - alpha for every alpha it accepts)."""
    if uc_sample not in ("full", "transitions"):
        raise ValueError("uc_sample must be 'full' or 'transitions'")
    h = _hits01(hits)
    if h.size < 2:
        raise ValueError("need at least 2 observations")
    uc_h = h if uc_sample == "full" else h[1:]
    lr_uc = float(_kupiec_pof(int(uc_h.sum()), uc_h.size, p)["lr_uc"])
    lr_ind = christoffersen_ind(h)["lr_ind"]
    lr_cc = lr_uc + lr_ind
    return {"lr_uc": lr_uc, "lr_ind": lr_ind, "lr_cc": lr_cc,
            "p_uc": float(stats.chi2.sf(lr_uc, 1)), "p_ind": float(stats.chi2.sf(lr_ind, 1)),
            "p_cc": float(stats.chi2.sf(lr_cc, 2))}


# ----------------------------------------------------------------------------
# Extreme value theory
# ----------------------------------------------------------------------------
def _gpd_nll(xi, beta, y):
    if not (beta > 0) or not np.isfinite(xi):
        return np.inf
    z = y / beta
    w = 1.0 + xi * z
    if np.any(w <= 0):
        return np.inf
    n = y.size
    if abs(xi) < 1e-12:
        return float(n * math.log(beta) + z.sum())
    return float(n * math.log(beta) + (1.0 + 1.0 / xi) * np.log1p(xi * z).sum())


def gpd_fit(excesses, *, polish=True):
    """Maximum-likelihood fit of a generalised Pareto distribution GPD(xi, beta) to excesses.

    Parameters
    ----------
    excesses : array_like, 1-D. Threshold excesses y = L - u for the losses with L > u
        (strict), so every y > 0. Build them as ``y = L[L > u] - u``.
    polish : bool, default True. After ``scipy.stats.genpareto.fit(y, floc=0)`` (scipy
        shape c = xi, scale = beta), re-minimise the NLL with L-BFGS-B over (xi, ln beta)
        from that point and keep whichever NLL is lower.

    Returns
    -------
    dict: xi (shape; > 0 heavy tail, 0 exponential, < 0 bounded), beta (scale, > 0),
        nll (minimised negative log-likelihood), n (number of excesses).

    Formula: NLL = n ln beta + (1 + 1/xi) sum ln(1 + xi y/beta)  (xi != 0);
    NLL = n ln beta + sum y/beta  (xi = 0). Regular MLE needs xi > -1/2; for xi <= -1
    the likelihood is unbounded (no MLE) and the returned fit sits at the support edge
    beta = -xi*max(y), so do not trust a result with xi < -1/2.

    Example
    -------
    >>> u = np.quantile(L, 0.9); fit = gpd_fit(L[L > u] - u)
    >>> var, es = pot_var_es(0.99, u=u, xi=fit['xi'], beta=fit['beta'], n=L.size, n_u=fit['n'])
    """
    y = _sample(excesses, "excesses")
    if np.any(y <= 0):
        raise ValueError("excesses must be strictly positive: use y = L[L > u] - u")
    if y.size < 2:
        raise ValueError("need at least 2 excesses")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        c, _, sc = stats.genpareto.fit(y, floc=0)
    xi, beta = float(c), float(sc)
    nll = _gpd_nll(xi, beta, y)
    if polish:
        def f(theta):
            val = _gpd_nll(theta[0], math.exp(theta[1]), y)
            return val if np.isfinite(val) else 1e300

        x0 = np.array([xi, math.log(beta)])
        try:
            with np.errstate(all="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = optimize.minimize(f, x0, method="L-BFGS-B")
            xi2, beta2 = float(res.x[0]), float(math.exp(res.x[1]))
            nll2 = _gpd_nll(xi2, beta2, y)
            if np.isfinite(nll2) and nll2 < nll:
                xi, beta, nll = xi2, beta2, nll2
        except (ValueError, OverflowError, FloatingPointError):
            pass
    return {"xi": xi, "beta": beta, "nll": float(nll), "n": int(y.size)}


def pot_var_es(alpha, *, u, xi, beta, n, n_u):
    """Peaks-over-threshold VaR and ES from a GPD tail fitted above threshold u (McNeil-Frey 2000).

    Parameters
    ----------
    alpha : float or array in (0, 1). Confidence level(s); need 1-alpha <= n_u/n
        (the quantile must lie in the fitted tail).
    u : float. Threshold (loss units).
    xi, beta : floats (or arrays). GPD shape and scale fitted to the excesses L - u.
    n : int. Total number of observations. n_u : int. Number of exceedances L > u.

    Returns
    -------
    (var, es) : numpy float64 for scalar inputs, else arrays. ES = nan where xi >= 1.

    Formula (MFE QRM Sec. 5.2): r = (n/n_u)(1-alpha);
    VaR = u + (beta/xi)(r^{-xi} - 1);  ES = VaR/(1-xi) + (beta - xi u)/(1-xi);
    xi = 0 (|xi| < 1e-12): VaR = u - beta ln r, ES = VaR + beta.

    Example
    -------
    >>> pot_var_es(0.995, u=2.0, xi=0.25, beta=0.6, n=1000, n_u=80)   # (4.4, 6.0)
    """
    a = _check_alpha(alpha)
    xi_a = np.asarray(xi, dtype=float)
    b = np.asarray(beta, dtype=float)
    if np.any(~(b > 0)):
        raise ValueError("beta must be > 0")
    if not (0 < n_u <= n):
        raise ValueError("need 0 < n_u <= n")
    r = (n / n_u) * (1.0 - a)
    if np.any(r > 1.0 + 1e-12):
        raise ValueError("1-alpha must be <= n_u/n: the quantile lies below the threshold u")
    small = np.abs(xi_a) < 1e-12
    xi_safe = np.where(small, 1.0, xi_a)
    with np.errstate(divide="ignore", invalid="ignore"):
        var_gpd = u + (b / xi_safe) * (r ** (-xi_safe) - 1.0)
        var_exp = u - b * np.log(r)
        var = np.where(small, var_exp, var_gpd)
        es_gpd = var / (1.0 - xi_a) + (b - xi_a * u) / (1.0 - xi_a)
        es = np.where(small, var + b, np.where(xi_a < 1.0, es_gpd, np.nan))
    return _out(var), _out(es)


def hill_estimator(losses, k, *, ref):
    """Hill estimate of the tail shape xi (= 1/tail index) from the k largest losses.

    Parameters
    ----------
    losses : array_like, 1-D. LOSSES; only the strictly positive ones are used.
    k : int or int array. Number of upper order statistics (array -> Hill plot).
    ref : str, REQUIRED. Reference order statistic (threshold):
        'k+1' Hill (1975): threshold is the (k+1)-th largest, 1 <= k <= m-1;
        'k'   Embrechts-Klueppelberg-Mikosch / MFE form: threshold is the k-th largest,
              2 <= k <= m  (m = number of positive losses).

    Returns
    -------
    xi_hat : float (scalar k) or ndarray (array k). Tail index estimate = 1/xi_hat.

    Formula: X(1) >= X(2) >= ... ;
    'k+1': xi = (1/k) sum_{i<=k} ln X(i) - ln X(k+1);  'k': (1/k) sum_{i<=k} ln X(i) - ln X(k).
    Asymptotic standard error ~ xi/sqrt(k).

    Example
    -------
    >>> ks = np.arange(20, 500); xi_path = hill_estimator(L, ks, ref='k+1')
    """
    if ref not in ("k+1", "k"):
        raise ValueError("ref must be 'k+1' (Hill 1975) or 'k' (EKM/MFE)")
    L = _sample(losses, "losses")
    X = np.sort(L[L > 0])[::-1]
    m = X.size
    k_arr = np.asarray(k)
    if not np.all(k_arr == np.round(k_arr)):
        raise ValueError("k must be integer(s)")
    k_int = k_arr.astype(np.int64)
    lo, hi = (1, m - 1) if ref == "k+1" else (2, m)
    if np.any((k_int < lo) | (k_int > hi)):
        raise ValueError(f"k must lie in [{lo}, {hi}] for ref={ref!r} with {m} positive losses")
    lx = np.log(X)
    csum = np.cumsum(lx)
    mean_top = csum[k_int - 1] / k_int
    xi_hat = mean_top - (lx[k_int] if ref == "k+1" else lx[k_int - 1])
    return float(xi_hat) if xi_hat.ndim == 0 else xi_hat


def mean_excess(losses, u):
    """Empirical mean-excess function e(u) = mean(L - u | L > u) at one or many thresholds.

    Parameters
    ----------
    losses : array_like, 1-D. LOSSES (positive = loss). Must be finite.
    u : float or array. Threshold(s); exceedance is strict (L > u).

    Returns
    -------
    e : numpy float64 for scalar u, else array; nan where no loss exceeds u.

    Formula: e(u) = sum_{L>u}(L - u) / #{L > u}. For L ~ GPD(xi, beta) with xi < 1,
    e(u) = (beta + xi u)/(1 - xi) is linear in u; for excesses over a threshold u0 that
    follow GPD(xi, beta), e(v) = (beta + xi (v - u0))/(1 - xi) for v >= u0; for
    exponential(beta) it is constant beta. Identity: VaR + e(VaR) = historical ES with
    es_rule='gt' whenever some loss exceeds VaR.

    Example
    -------
    >>> us = np.quantile(L, np.linspace(0.8, 0.99, 40)); e = mean_excess(L, us)
    """
    Ls = np.sort(_sample(losses, "losses"))
    uu = np.asarray(u, dtype=float)
    idx = np.searchsorted(Ls, uu, side="right")        # first index with L > u
    suffix = np.concatenate([np.cumsum(Ls[::-1])[::-1], [0.0]])
    cnt = Ls.size - idx
    with np.errstate(invalid="ignore", divide="ignore"):
        e = np.where(cnt > 0, (suffix[idx] - uu * cnt) / np.maximum(cnt, 1), np.nan)
    return _out(e)
