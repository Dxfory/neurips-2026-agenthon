"""Credit-portfolio simulation and the Vasicek ASRF model: CreditMetrics migration thresholds and correlated
rating migrations, Gaussian/Student-t latent-factor (Merton-style) defaults, the Vasicek large-pool loss
distribution, and the Euler allocation of a simulated expected shortfall.

Conventions for the whole module:
- A transition matrix P has one ROW per initial rating and one COLUMN per final rating ordered from the BEST
  rating to the WORST, default LAST (the CreditMetrics table layout); probabilities, not percents.
- Latent variable of obligor i: A_i = sqrt(rho_i) M + sqrt(1 - rho_i) eps_i with standard normal factor(s) M and
  idiosyncratic eps_i; LOW values are bad (default when A_i is below its threshold).
- alpha is a CONFIDENCE level (0.99); losses are positive numbers.
- rng: a numpy Generator (np.random.default_rng(seed)), a legacy RandomState, ``numpy.random`` itself (after
  np.random.seed) or an int seed. The order of the draws is documented in each function, so a hand-written loop
  with the same generator reproduces it exactly; simulations are processed in row blocks, which leaves the stream
  unchanged.

Functions
---------
migration_thresholds    CreditMetrics N^-1 thresholds of the cumulative transition probabilities, worst first
simulate_migrations     one-factor Gaussian-copula rating migrations (draw order 'blocks' or 'per_path')
latent_factor_defaults  Gaussian or Student-t copula defaults, one factor or correlated sector factors
vasicek_loss_cdf        Vasicek large homogeneous pool loss-fraction CDF
vasicek_loss_quantile   ASRF (Basel IRB) loss quantile, homogeneous or sum over obligors
es_contributions        Euler allocation of a simulated ES to components (sums to ES)

Ported from the team's qflib (``credit.transition_cumulatives``, ``rating_thresholds``, ``assign_ratings``,
``simulate_rating_migrations``, ``latent_factor_defaults``, ``vasicek_loss_cdf``, ``vasicek_loss_quantile``,
``vasicek_portfolio_loss`` and ``risk.component_historical_es``), extended to sector factors, legacy generators and
block-wise simulation, with the order-statistic counts made integer-safe.

References: Gupton, Finger and Bhatia, CreditMetrics Technical Document (1997) Ch. 8 and 10; Vasicek (2002), Risk
15(12); Gordy (2003), J. Financial Intermediation 12; BCBS, An Explanatory Note on the Basel II IRB Risk Weight
Functions (2005); Glasserman (2004) Sec. 9.4; McNeil, Frey and Embrechts (2015) Ch. 8 and 11 (t copula by normal
variance mixture); Tasche (1999) "Risk contributions and performance measurement"; Tasche (2000).
"""

from __future__ import annotations

import math
import numbers

import numpy as np
import pandas
from scipy import stats

__all__ = [
    "migration_thresholds",
    "simulate_migrations",
    "latent_factor_defaults",
    "vasicek_loss_cdf",
    "vasicek_loss_quantile",
    "es_contributions",
]

_ORDERS = ("best_to_worst", "worst_to_best")
_DRAWS = ("blocks", "per_path")
_ES_RULES = ("ge", "gt", "k_worst")
_QUANTILE_METHODS = ("inverted_cdf", "averaged_inverted_cdf", "closest_observation", "interpolated_inverted_cdf",
                     "hazen", "weibull", "linear", "median_unbiased", "normal_unbiased", "lower", "higher",
                     "midpoint", "nearest")
_BLOCK = 2_000_000  # simulated values per row block


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _choice(value, name, allowed):
    if not isinstance(value, str) or value not in allowed:
        raise ValueError(f"{name} must be one of {', '.join(repr(a) for a in allowed)}, got {value!r}")
    return value


def _count(value, name):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 1:
        raise ValueError(f"{name} must be an int >= 1")
    return int(value)


def _generator(rng):
    if isinstance(rng, (bool, np.bool_)):
        raise TypeError("rng must be a numpy Generator, a RandomState, numpy.random itself or an int seed")
    if isinstance(rng, numbers.Integral):
        return np.random.default_rng(int(rng))
    for attr in ("standard_normal", "chisquare"):
        if not callable(getattr(rng, attr, None)):
            raise TypeError("rng must be a numpy Generator (np.random.default_rng(seed)), a RandomState, "
                            "numpy.random itself (after np.random.seed) or an int seed")
    return rng


def _state(rng):
    # A Generator by type: numpy.random itself also has a bit_generator attribute (a submodule, without state).
    return rng.bit_generator.state if isinstance(rng, np.random.Generator) else rng.get_state()


def _restore(rng, state):
    if isinstance(rng, np.random.Generator):
        rng.bit_generator.state = state
    else:  # a RandomState or numpy.random: the legacy state, with its cached normal
        rng.set_state(state)


def _rows_per_block(n_cols):
    return max(1, _BLOCK // max(1, n_cols))


def _matrix(P):
    a = np.asarray(P, dtype=float)
    if a.ndim != 2 or a.shape[1] < 2 or a.shape[0] < 1:
        raise ValueError("P must be a 2-D transition matrix (initial ratings x final ratings)")
    if np.any(~np.isfinite(a)) or np.any(a < -1e-12):
        raise ValueError("P must hold finite probabilities >= 0")
    sums = a.sum(axis=1)
    if np.any(np.abs(sums - 100.0) < 1e-6) and not np.any(np.abs(sums - 1.0) < 1e-6):
        raise ValueError("P looks like percents (rows sum to 100): divide by 100")
    if np.any(np.abs(sums - 1.0) > 1e-6):
        raise ValueError("each row of P must sum to 1")
    return np.clip(a, 0.0, None)


def _thresholds(a):
    """(n_from, n_to-1) increasing N^-1 of cumulative probabilities from the worst column (the last)."""
    cum = np.cumsum(a[:, ::-1], axis=1)[:, :-1]
    with np.errstate(divide="ignore"):
        return stats.norm.ppf(np.clip(cum, 0.0, 1.0))


def _rho(rho, n):
    r = np.asarray(rho, dtype=float)
    if r.ndim == 0:
        r = np.full(n, float(r))
    if r.shape != (n,) or np.any(~np.isfinite(r)) or np.any((r < 0.0) | (r >= 1.0)):
        raise ValueError(f"rho must be a correlation in [0, 1), a scalar or one per obligor ({n})")
    return r


def _alpha(alpha):
    a = np.asarray(alpha, dtype=float)
    if np.any(~(a > 0.0)) or np.any(~(a < 1.0)):
        raise ValueError("alpha is a confidence level in (0, 1), e.g. 0.999")
    return a


def _ceil(x):
    """ceil for a count formed in floating point: within 1e-9 of an integer counts as that integer."""
    return int(math.ceil(x - 1e-9))


def _floor(x):
    return int(math.floor(x + 1e-9))


# ----------------------------------------------------------------------------
# migrations
# ----------------------------------------------------------------------------
def migration_thresholds(P, *, order):
    """CreditMetrics rating-migration thresholds: the standard normal quantiles N^-1 of the cumulative transition
    probabilities, accumulated from the worst state (default), one increasing row per initial rating.

    Parameters
    ----------
    P : array_like (n_from, n_to). Transition probabilities, row = initial rating, rows sum to 1.
    order : REQUIRED. The column order of P: 'best_to_worst' (AAA ... default, the usual table) or
        'worst_to_best' (default ... AAA).

    Returns
    -------
    ndarray (n_from, n_to - 1), each row increasing. Row i: z < t[i, 0] is the worst state (default),
    t[i, j-1] <= z < t[i, j] the (j+1)-th worst, z >= t[i, -1] the best. A zero-probability state gives equal
    neighbouring thresholds (or -inf/+inf at the ends).

    Formula
    -------
    t[i, j] = N^-1(p_i,worst + ... + p_i,(j+1)-th worst), j = 0 .. n_to - 2 (CreditMetrics TD Table 8.4).

    References
    ----------
    Gupton, Finger and Bhatia (1997) Sec. 8.4.

    Example
    -------
    >>> thr = migration_thresholds(P, order='best_to_worst')  # thr[i, 0] = N^-1(PD of rating i)
    """
    how = _choice(order, "order", _ORDERS)
    a = _matrix(P)
    return _thresholds(a if how == "best_to_worst" else a[:, ::-1])


def _ratings(z, thr, n_to):
    """Column index (best = 0, default = n_to - 1) of latent z under increasing thresholds thr."""
    return (n_to - 1) - np.searchsorted(thr, z, side="right")


def simulate_migrations(P, start, *, rho, n_paths, rng, draws):
    """CreditMetrics correlated rating migrations under a one-factor Gaussian copula:
    Z_i = sqrt(rho) M + sqrt(1 - rho) eps_i mapped to a final rating through the thresholds of its initial rating.

    Parameters
    ----------
    P : array_like (n_from, n_to). Transition probabilities with columns BEST to WORST, default LAST; rows sum to 1.
    start : array_like (n_obligors,) of ints. Each obligor's initial rating = its row index in P.
    rho : REQUIRED. Asset correlation in [0, 1), a scalar or one per obligor.
    n_paths : REQUIRED. Number of simulations, int >= 1.
    rng : REQUIRED. numpy Generator, RandomState, ``numpy.random`` (after np.random.seed) or an int seed.
    draws : REQUIRED. The order of the normal draws:
        'blocks'   M = rng.standard_normal(n_paths), then eps = rng.standard_normal((n_paths, n_obligors));
        'per_path' for each simulation in turn, its M and then its n_obligors eps
                   (= rng.standard_normal((n_paths, 1 + n_obligors)), column 0 = M).

    Returns
    -------
    dict
        ratings : (n_paths, n_obligors) int final ratings as COLUMN indices of P (0 = best, n_to - 1 = default).
        default : (n_paths, n_obligors) bool, ratings == n_to - 1.

    Formula
    -------
    Thresholds as migration_thresholds; z below the first threshold defaults; a tie goes to the better rating.
    Migration frequencies converge to the rows of P for any rho. When row i of P is the rating of column i (the
    CreditMetrics layout), ratings < start are upgrades and ratings > start downgrades, default included; a TASK's
    downgrade without default is (ratings > start) & ~default.

    References
    ----------
    Gupton, Finger and Bhatia (1997) Ch. 8 and 10 (Monte Carlo, Sec. 10.2); Glasserman (2004) Sec. 9.4.

    Example
    -------
    >>> sim = simulate_migrations(P, start, rho=0.2, n_paths=200_000, rng=np.random.default_rng(42),
    ...                           draws='blocks')
    >>> loss = sim['default'] @ (lgd * face)  # one total per simulation
    """
    a = _matrix(P)
    n_to = a.shape[1]
    s = np.asarray(start)
    if s.ndim != 1 or s.size == 0 or not np.issubdtype(s.dtype, np.integer) or s.min() < 0 or s.max() >= a.shape[0]:
        raise ValueError(f"start must be a 1-D array of row indices of P (ints in 0..{a.shape[0] - 1})")
    n = s.size
    r = _rho(rho, n)
    paths = _count(n_paths, "n_paths")
    order = _choice(draws, "draws", _DRAWS)
    gen = _generator(rng)
    thr = _thresholds(a)
    dtype = np.int8 if n_to <= 127 else np.int32
    ratings = np.empty((paths, n), dtype=dtype)
    a_sys, a_idio = np.sqrt(r), np.sqrt(1.0 - r)
    groups = [(row, np.flatnonzero(s == row)) for row in np.unique(s)]
    m_all = gen.standard_normal(paths) if order == "blocks" else None
    step = _rows_per_block(n + 1)
    for lo in range(0, paths, step):
        hi = min(paths, lo + step)
        if order == "blocks":
            m, eps = m_all[lo:hi], gen.standard_normal((hi - lo, n))
        else:
            block = gen.standard_normal((hi - lo, n + 1))
            m, eps = block[:, 0], block[:, 1:]
        z = a_sys * m[:, None] + a_idio * eps
        for row, cols in groups:
            ratings[lo:hi, cols] = _ratings(z[:, cols], thr[row], n_to)
    return {"ratings": ratings, "default": ratings == n_to - 1}


def latent_factor_defaults(pd, *, rho, n_paths, rng, nu, factor_corr=None, factor=None):
    """Merton-style latent-factor defaults under a Gaussian or Student-t copula, with one systematic factor or
    correlated sector factors: A_i = sqrt(rho_i) M_f(i) + sqrt(1 - rho_i) eps_i, default when A_i < threshold_i.

    Parameters
    ----------
    pd : array_like (n_obligors,). Default probabilities over the horizon, in [0, 1) (0 never defaults).
    rho : REQUIRED. Asset correlation (factor loading squared) in [0, 1), a scalar or one per obligor (e.g. the
        Basel IRB correlation).
    n_paths : REQUIRED. Number of simulations, int >= 1.
    rng : REQUIRED. numpy Generator, RandomState, ``numpy.random`` (after np.random.seed) or an int seed.
    nu : REQUIRED. None = Gaussian copula, thresholds N^-1(pd). A number nu > 0 = Student-t copula by the normal
        variance mixture: W = chi2(nu)/nu per simulation, BOTH M and eps divided by sqrt(W), thresholds
        t_nu^-1(pd) so every marginal PD is kept.
    factor_corr : optional (K, K) correlation matrix of K systematic (sector) factors; None = one factor.
    factor : optional (n_obligors,) ints in 0..K-1, each obligor's factor; required with factor_corr.

    Returns
    -------
    dict
        default : (n_paths, n_obligors) bool default indicators;
        threshold : (n_obligors,) the default thresholds used.

    Formula
    -------
    Draw order (fixed): Z = rng.standard_normal((n_paths, K)) and M = Z @ cholesky(factor_corr).T (M = Z for one
    factor), then eps = rng.standard_normal((n_paths, n_obligors)), then (t only) W = rng.chisquare(nu, n_paths)/nu.
    One factor: P(i and j default) = N2(t_i, t_j; sqrt(rho_i rho_j)).

    References
    ----------
    Vasicek (2002); Gordy (2000) J. Banking and Finance 24; Glasserman (2004) Sec. 9.4; McNeil, Frey and Embrechts
    (2015) Sec. 11.1 (Gaussian and t threshold models).

    Example
    -------
    >>> sim = latent_factor_defaults(pd_1y, rho=basel_rho, n_paths=100_000, rng=np.random.default_rng(42), nu=None,
    ...                              factor_corr=sector_corr, factor=sector_index)
    >>> loss = sim['default'] @ (ead * lgd)
    """
    p = np.asarray(pd, dtype=float)
    if p.ndim != 1 or p.size == 0 or np.any(~np.isfinite(p)) or np.any((p < 0.0) | (p >= 1.0)):
        raise ValueError("pd must be a non-empty 1-D array of default probabilities in [0, 1)")
    n = p.size
    r = _rho(rho, n)
    paths = _count(n_paths, "n_paths")
    gen = _generator(rng)
    if nu is not None and (isinstance(nu, bool) or not isinstance(nu, numbers.Real) or not float(nu) > 0.0):
        raise ValueError("nu must be None (Gaussian copula) or the Student-t degrees of freedom > 0")
    if factor_corr is None:
        if factor is not None:
            raise ValueError("factor needs factor_corr (the factors' correlation matrix)")
        chol, idx = np.ones((1, 1)), np.zeros(n, dtype=int)
    else:
        c = np.asarray(factor_corr, dtype=float)
        if c.ndim != 2 or c.shape[0] != c.shape[1] or not np.allclose(c, c.T, atol=1e-10) \
                or not np.allclose(np.diag(c), 1.0, atol=1e-8):
            raise ValueError("factor_corr must be a symmetric correlation matrix with unit diagonal")
        try:
            chol = np.linalg.cholesky(c)
        except np.linalg.LinAlgError as exc:
            raise ValueError("factor_corr must be positive definite") from exc
        idx = np.asarray(factor)
        if factor is None or idx.shape != (n,) or not np.issubdtype(idx.dtype, np.integer) \
                or idx.min() < 0 or idx.max() >= c.shape[0]:
            raise ValueError(f"factor must give each obligor's factor index, ints in 0..{c.shape[0] - 1}")
    with np.errstate(divide="ignore"):
        thr = stats.norm.ppf(p) if nu is None else stats.t.ppf(p, float(nu))
    m = gen.standard_normal((paths, chol.shape[0])) @ chol.T
    step = _rows_per_block(n)
    scale = None
    if nu is not None:  # W comes after every eps: pass over them once, draw W, then replay the eps
        before = _state(gen)
        for lo in range(0, paths, step):
            gen.standard_normal((min(paths, lo + step) - lo, n))
        scale = np.sqrt(gen.chisquare(float(nu), paths) / float(nu))
        after = _state(gen)
        _restore(gen, before)
    default = np.empty((paths, n), dtype=bool)
    a_sys, a_idio = np.sqrt(r), np.sqrt(1.0 - r)
    for lo in range(0, paths, step):
        hi = min(paths, lo + step)
        asset = a_sys * m[lo:hi][:, idx] + a_idio * gen.standard_normal((hi - lo, n))
        if scale is not None:
            asset /= scale[lo:hi, None]
        default[lo:hi] = asset < thr
    if nu is not None:
        _restore(gen, after)
    return {"default": default, "threshold": thr}


# ----------------------------------------------------------------------------
# Vasicek ASRF
# ----------------------------------------------------------------------------
def vasicek_loss_cdf(x, pd, *, lgd, rho):
    """Vasicek large homogeneous pool: the distribution function P(L <= x) of the portfolio loss fraction
    L = LGD N((N^-1(PD) - sqrt(rho) M)/sqrt(1 - rho)) of an infinitely granular one-factor portfolio.

    Parameters
    ----------
    x : float or array_like. Loss fraction(s) of exposure (0.05 = 5%).
    pd : float in (0, 1). Default probability of every obligor.
    lgd : REQUIRED. Loss given default in (0, 1].
    rho : REQUIRED. Asset correlation in [0, 1); 0 puts all mass at LGD PD.

    Returns
    -------
    float or ndarray in [0, 1] (0 for x <= 0, 1 for x >= LGD).

    Formula
    -------
    F(x) = N((sqrt(1 - rho) N^-1(x/LGD) - N^-1(PD)) / sqrt(rho)), 0 < x < LGD; its inverse is vasicek_loss_quantile.

    References
    ----------
    Vasicek (2002), Risk 15(12), eq. (2).

    Example
    -------
    >>> vasicek_loss_cdf(0.02, 0.01, lgd=0.45, rho=0.12)
    """
    xx = np.asarray(x, dtype=float)
    p = float(pd)
    if not 0.0 < p < 1.0:
        raise ValueError("pd must lie in (0, 1)")
    if isinstance(lgd, bool) or not 0.0 < float(lgd) <= 1.0:
        raise ValueError("lgd must lie in (0, 1]")
    if isinstance(rho, bool) or not 0.0 <= float(rho) < 1.0:
        raise ValueError("rho must lie in [0, 1)")
    g, c = float(lgd), float(rho)
    if c == 0.0:
        out = (xx >= g * p).astype(float)
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            u = stats.norm.ppf(np.clip(xx / g, 0.0, 1.0))
            out = stats.norm.cdf((math.sqrt(1.0 - c) * u - stats.norm.ppf(p)) / math.sqrt(c))
        out = np.where(xx <= 0.0, 0.0, np.where(xx >= g, 1.0, out))
    return float(out) if out.ndim == 0 else out


def vasicek_loss_quantile(alpha, pd, *, lgd, rho, weight=None):
    """Vasicek asymptotic single risk factor (ASRF, Basel IRB) loss quantile: LGD times the conditional default
    probability at the alpha-quantile of the systematic factor, for one homogeneous pool or summed over obligors.

    Parameters
    ----------
    alpha : float or array. Confidence level(s), e.g. 0.999.
    pd : float or array (n,). Default probabilities in [0, 1) (0 contributes nothing).
    lgd : REQUIRED. Loss given default in [0, 1], a scalar or one per obligor.
    rho : REQUIRED. Asset correlation in [0, 1), a scalar or one per obligor.
    weight : optional (n,) exposures (EAD) or weights. None: the homogeneous loss FRACTION LGD N(h) (pd, lgd, rho
        broadcast elementwise); given: the portfolio loss sum_i weight_i LGD_i N(h_i) in the units of weight.

    Returns
    -------
    float (or an array over alpha / elementwise for weight=None).

    Formula
    -------
    h_i(alpha) = (N^-1(PD_i) + sqrt(rho_i) N^-1(alpha)) / sqrt(1 - rho_i); the IRB capital charge is
    K = LGD N(h) - LGD PD (times the maturity adjustment).

    References
    ----------
    Vasicek (2002); Gordy (2003); BCBS (2005), An Explanatory Note on the Basel II IRB Risk Weight Functions, eq. (1).

    Example
    -------
    >>> var_999 = vasicek_loss_quantile(0.999, pd_1y, lgd=1 - recovery, rho=basel_rho, weight=ead)
    """
    a = _alpha(alpha)
    p = np.asarray(pd, dtype=float)
    g = np.asarray(lgd, dtype=float)
    c = np.asarray(rho, dtype=float)
    if np.any(~np.isfinite(p)) or np.any((p < 0.0) | (p >= 1.0)):
        raise ValueError("pd must lie in [0, 1)")
    if np.any(~np.isfinite(g)) or np.any((g < 0.0) | (g > 1.0)):
        raise ValueError("lgd must lie in [0, 1] (a fraction, 1 - recovery)")
    if np.any(~np.isfinite(c)) or np.any((c < 0.0) | (c >= 1.0)):
        raise ValueError("rho must lie in [0, 1)")
    with np.errstate(divide="ignore"):
        if weight is None:
            h = (stats.norm.ppf(p) + np.sqrt(c) * stats.norm.ppf(a)) / np.sqrt(1.0 - c)
            out = g * stats.norm.cdf(h)
            return float(out) if np.ndim(out) == 0 else out
        w = np.asarray(weight, dtype=float)
        if w.ndim != 1 or np.any(~np.isfinite(w)):
            raise ValueError("weight must be a finite 1-D array, one per obligor")
        p, g, c, w = np.broadcast_arrays(p, g, c, w)
        h = (stats.norm.ppf(p)[None, :] + np.sqrt(c)[None, :] * stats.norm.ppf(np.atleast_1d(a))[:, None]) \
            / np.sqrt(1.0 - c)[None, :]
        out = (stats.norm.cdf(h) * (w * g)[None, :]).sum(axis=1)
    return float(out[0]) if a.ndim == 0 else out


# ----------------------------------------------------------------------------
# Euler allocation of ES
# ----------------------------------------------------------------------------
def es_contributions(X, alpha, *, weights, var_rule, es_rule):
    """Euler allocation of a simulated expected shortfall (CVaR) to components: contribution_i = weight_i times the
    mean of X_i over the tail scenarios of the total loss L = X @ weights; the contributions sum to ES.

    Parameters
    ----------
    X : array_like (n_paths, n) or DataFrame. Per scenario and component: a loss per unit weight (e.g. the default
        indicators of latent_factor_defaults), or the component losses themselves with weights=None.
    alpha : float in (0, 1). Confidence level, e.g. 0.99.
    weights : REQUIRED. (n,) multipliers (e.g. EAD * LGD), or None when X already holds the component losses.
    var_rule : REQUIRED. VaR of the total loss L over the n_paths scenarios:
        'k_worst'    the k-th largest L, k = n_paths (1 - alpha) (Hull: the 5th worst of 500 at 99%);
        'ceil_index' the smallest L with empirical CDF >= alpha = sorted(L)[ceil(n_paths alpha) - 1]
                     (the 6th worst of 500 at 99%; numpy 'inverted_cdf');
        or any numpy quantile method ('linear', 'higher', 'lower', ...).
    es_rule : REQUIRED. Tail scenarios: 'ge' L >= VaR; 'gt' L > VaR; 'k_worst' the k largest L
        (k = floor(n_paths (1 - alpha)), ties by position).

    Returns
    -------
    dict
        var, es : of L; contrib : (n,) Euler contributions (a Series on X's columns for a DataFrame), summing to es;
        pct : contrib / es; n_tail : tail scenarios; loss : (n_paths,) total losses L.

    Formula
    -------
    ES = E[L | tail] = sum_i w_i E[X_i | tail] (Tasche 1999): the derivative of ES with respect to w_i, estimated on
    the simulated tail. Counts n alpha and n (1 - alpha) within 1e-9 of an integer are that integer.

    References
    ----------
    Tasche (1999), "Risk contributions and performance measurement"; Tasche (2000); McNeil, Frey and Embrechts
    (2015) Sec. 8.5; Hull, Risk Management and Financial Institutions (2018) Ch. 13 (historical VaR order).

    Example
    -------
    >>> eu = es_contributions(sim['default'], 0.99, weights=ead * lgd, var_rule='ceil_index', es_rule='ge')
    >>> eu['es'], eu['contrib'].sum(), eu['contrib']
    """
    columns = X.columns if isinstance(X, pandas.DataFrame) else None
    x = X.to_numpy() if isinstance(X, pandas.DataFrame) else np.asarray(X)
    if x.ndim != 2 or x.shape[0] < 1 or x.shape[1] < 1:
        raise ValueError("X must be 2-D: one row per scenario, one column per component")
    if x.dtype != bool and not np.issubdtype(x.dtype, np.number):
        raise ValueError("X must be numeric or boolean")
    n_paths, n = x.shape
    if alpha is None or isinstance(alpha, bool) or not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha is a confidence level in (0, 1), e.g. 0.99")
    a = float(alpha)
    rule = _choice(es_rule, "es_rule", _ES_RULES)
    if var_rule not in ("k_worst", "ceil_index") and var_rule not in _QUANTILE_METHODS:
        raise ValueError(f"var_rule must be 'k_worst', 'ceil_index' or a numpy quantile method {_QUANTILE_METHODS}")
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    if w.shape != (n,) or np.any(~np.isfinite(w)):
        raise ValueError(f"weights must be None or {n} finite numbers, one per column of X")
    loss = np.empty(n_paths)
    step = _rows_per_block(n)
    for lo in range(0, n_paths, step):
        block = x[lo:lo + step].astype(float)
        if not np.all(np.isfinite(block)):
            raise ValueError("X contains NaN/inf")
        loss[lo:lo + step] = block @ w
    ordered = np.sort(loss)
    k = _floor(n_paths * (1.0 - a))
    if var_rule == "k_worst" or rule == "k_worst":
        if k < 1:
            raise ValueError(f"'k_worst' needs n_paths (1 - alpha) >= 1; got n_paths={n_paths}, alpha={a}")
    if var_rule == "k_worst":
        var = float(ordered[n_paths - k])
    elif var_rule == "ceil_index":
        var = float(ordered[min(max(_ceil(n_paths * a) - 1, 0), n_paths - 1)])
    else:
        var = float(np.quantile(loss, a, method=var_rule))
    if rule == "k_worst":
        tail = np.zeros(n_paths, dtype=bool)
        tail[np.argsort(loss, kind="stable")[n_paths - k:]] = True
    else:
        tail = loss >= var if rule == "ge" else loss > var
    n_tail = int(tail.sum())
    if n_tail == 0:
        raise ValueError("no scenario lies above VaR: use es_rule='ge'")
    contrib = w * x[tail].astype(float).mean(axis=0)
    es = float(loss[tail].mean())
    pct = contrib / es if es != 0.0 else np.full(n, np.nan)
    if columns is not None:
        contrib, pct = pandas.Series(contrib, index=columns), pandas.Series(pct, index=columns)
    return {"var": var, "es": es, "contrib": contrib, "pct": pct, "n_tail": n_tail, "loss": loss}
