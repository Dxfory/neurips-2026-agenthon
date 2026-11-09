"""Portfolio construction and performance: mean-variance, risk parity, Black-Litterman, Kelly, ratios.

Conventions for the whole module:
- ``mu``, ``cov``, ``rf``, ``target``, ``pi`` and ``Q`` share ONE period and unit (e.g. all annual
  decimals: mu = 0.08, cov in decimal^2 per year, rf = 0.03). The weight functions never
  annualise anything.
- Weights are fractions of capital. A fully-invested portfolio sums to 1 and a negative weight
  is a short. ``long_only`` is REQUIRED wherever it appears: False allows shorts (closed form);
  True adds w_i >= 0 and solves the quadratic programme (Lawson-Hanson least-distance programme
  via NNLS, Goldfarb-Idnani fallback; the answer must pass a KKT check and solves the KKT system
  exactly on its support).
- ``cov`` must be square, finite and symmetric. The closed forms, risk parity and Kelly need it
  positive definite (ValueError otherwise). The long-only QPs accept a singular cov (e.g. fewer
  observations than assets): a 1e-10 relative ridge then picks one of the equal-variance optima.
- The performance ratios take PER-PERIOD SIMPLE returns in decimals (0.01 = 1%) and annualise
  with an explicit ``periods_per_year`` (252 daily, 52 weekly, 12 monthly; 1 = no annualisation).
  Means scale by periods_per_year and standard deviations by sqrt(periods_per_year) (the iid
  rule; Lo 2002), so Sharpe and Sortino ratios scale by sqrt(periods_per_year).
- pandas in -> labelled pandas out; numpy in -> numpy out. Inputs must be finite (drop or fill
  NaN first). Two pandas inputs must carry identical labels (no silent re-alignment).

Functions
---------
min_variance_weights   global minimum-variance portfolio (shorts allowed or long-only)
tangency_weights       maximum-Sharpe (tangency) portfolio of the risky assets
target_return_weights  minimum-variance frontier portfolio for a target mean
risk_parity_weights    long-only equal-risk-contribution / risk-budgeting portfolio
risk_contributions     marginal, component and percent volatility contributions
bl_implied_returns     reverse-optimised equilibrium excess returns delta * cov @ w_mkt
black_litterman        Black-Litterman posterior mean and covariances, He-Litterman weights
kelly_bet              Kelly fraction for a binary bet
kelly_weights          growth-optimal (continuous-time Kelly) weights
sharpe_ratio           annualised Sharpe ratio
sortino_ratio          annualised Sortino ratio
max_drawdown           maximum drawdown with its peak, trough and drawdown series
calmar_ratio           annualised return divided by maximum drawdown
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import linalg, optimize

__all__ = [
    "min_variance_weights",
    "tangency_weights",
    "target_return_weights",
    "risk_parity_weights",
    "risk_contributions",
    "bl_implied_returns",
    "black_litterman",
    "kelly_bet",
    "kelly_weights",
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown",
    "calmar_ratio",
]

_TINY = np.finfo(float).tiny
_EPS = np.finfo(float).eps


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _out(x):
    """0-d result -> numpy float64 scalar; otherwise the ndarray itself."""
    x = np.asarray(x, dtype=float)
    return x[()] if x.ndim == 0 else x


def _labels_of(obj):
    if isinstance(obj, pd.DataFrame):
        return list(obj.columns)
    if isinstance(obj, pd.Series):
        return list(obj.index)
    return None


def _common_labels(*named):
    """Asset labels shared by the pandas inputs (DataFrame columns / Series index), else None.

    Raises instead of re-aligning when two pandas inputs disagree."""
    labels, first = None, None
    for name, obj in named:
        if isinstance(obj, pd.DataFrame) and name in ("cov",):
            if list(obj.index) != list(obj.columns):
                raise ValueError("cov DataFrame must have identical index and columns (same order)")
        lab = _labels_of(obj)
        if lab is None:
            continue
        if labels is None:
            labels, first = lab, name
        elif lab != labels:
            raise ValueError(
                f"{name} and {first} carry different asset labels (or order); align them first, "
                f"e.g. {name} = {name}.reindex(...)")
    return labels


def _vec(x, name, n=None):
    a = np.asarray(x, dtype=float)
    if a.ndim == 2 and 1 in a.shape:
        a = a.ravel()
    if a.ndim != 1:
        raise ValueError(f"{name} must be 1-D (got shape {a.shape})")
    if n is not None and a.shape[0] != n:
        raise ValueError(f"{name} has length {a.shape[0]} but cov is {n} x {n}")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} contains NaN/inf")
    return a


def _cov(cov):
    S = np.asarray(cov, dtype=float)
    if S.ndim != 2 or S.shape[0] != S.shape[1] or S.shape[0] == 0:
        raise ValueError(f"cov must be a non-empty square matrix (got shape {S.shape})")
    if not np.all(np.isfinite(S)):
        raise ValueError("cov contains NaN/inf")
    scale = max(float(np.abs(S).max()), _TINY)
    if float(np.abs(S - S.T).max()) > 1e-10 * scale:
        raise ValueError("cov must be symmetric")
    S = 0.5 * (S + S.T)
    if np.any(np.diag(S) < 0):
        raise ValueError("cov has a negative variance on its diagonal")
    return S


def _chol(S, what):
    """Cholesky factor of a positive-definite S; ValueError when S is singular to working precision
    (judged scale-free, on the correlation matrix, so assets in different units are fine)."""
    msg = (f"{what} needs a positive-definite cov (this one is singular or indefinite: remove "
           "redundant assets or shrink the estimate)")
    sd = np.sqrt(np.diag(S))
    if np.any(sd == 0):
        raise ValueError(msg)
    try:
        piv = np.diag(linalg.cholesky(S / np.outer(sd, sd), lower=True, check_finite=False))
        if float(piv.min()) ** 2 < 1e-13:
            raise linalg.LinAlgError
        return linalg.cho_factor(S, lower=True, check_finite=False)
    except linalg.LinAlgError:
        raise ValueError(msg) from None


def _wrap(v, labels, name=None):
    return pd.Series(v, index=labels, name=name) if labels is not None else v


def _bool(x, name):
    if not isinstance(x, (bool, np.bool_)):
        raise TypeError(f"{name} must be True or False")
    return bool(x)


def _eq_qp(S, A, b, free):
    """argmin 0.5 w'Sw s.t. A w = b with w_j = 0 off ``free``; returns (w, lam), grad Sw = A'lam."""
    n = S.shape[0]
    idx = np.flatnonzero(free)
    k, m = idx.size, A.shape[0]
    K = np.zeros((k + m, k + m))
    K[:k, :k] = S[np.ix_(idx, idx)]
    K[:k, k:] = -A[:, idx].T
    K[k:, :k] = A[:, idx]
    rhs = np.concatenate([np.zeros(k), b])
    try:
        sol = np.linalg.solve(K, rhs)
        if not np.all(np.isfinite(sol)):
            raise np.linalg.LinAlgError
    except np.linalg.LinAlgError:
        sol = np.linalg.lstsq(K, rhs, rcond=None)[0]
    w = np.zeros(n)
    w[idx] = sol[:k]
    return w, sol[k:]


def _qp_ldp(S, R, A, b):
    """Lawson & Hanson, Solving Least Squares Problems (1974), Ch. 20-23: eliminate the
    equalities (w = w_p + Z v), turn the least-squares problem with inequalities into a
    least-distance programme and solve that by NNLS (Alg. 23.27). S = R'R."""
    U, sv, Vt = np.linalg.svd(A, full_matrices=True)
    r = int(np.sum(sv > sv.max() * max(A.shape) * _EPS))
    w_p = Vt[:r].T @ ((U[:, :r].T @ b) / sv[:r])
    if float(np.abs(A @ w_p - b).max()) > 1e-9 * max(1.0, float(np.abs(b).max())):
        raise ValueError("the equality constraints are inconsistent")
    Z = Vt[r:].T
    if Z.shape[1] == 0:
        return w_p
    Q1, Rr = np.linalg.qr(R @ Z)
    g = Q1.T @ (-R @ w_p)
    G = linalg.solve_triangular(Rr, Z.T, trans="T").T      # Z Rr^{-1}
    h = -w_p - G @ g
    k = G.shape[1]
    cn = np.sqrt(np.einsum("ij,ij->i", G, G))              # unit-norm constraint rows: same
    cn[cn == 0] = 1.0                                       # feasible set, better-conditioned NNLS
    G, h = G / cn[:, None], h / cn
    E = np.vstack([G.T, h[None, :]])
    f = np.zeros(k + 1)
    f[-1] = 1.0
    z = optimize.nnls(E, f, maxiter=50 * (S.shape[0] + k + 1))[0]
    res = E @ z - f
    if not abs(res[-1]) > 1e-14:
        raise ValueError("the long-only constraints are infeasible")
    v = linalg.solve_triangular(Rr, -res[:k] / res[-1] + g)
    return w_p + Z @ v


def _qp_gi(S, A, b):
    """Goldfarb & Idnani (1983) dual active-set method for min 0.5 w'Sw s.t. A w = b, w >= 0
    (S positive definite). Factorisations are recomputed each step (small N)."""
    n, m = S.shape[0], A.shape[0]
    Linv = linalg.solve_triangular(np.linalg.cholesky(S), np.eye(n), lower=True)
    C = np.vstack([A, np.eye(n)])
    rhs = np.concatenate([b, np.zeros(n)])
    sign = np.ones(m + n)
    x = np.zeros(n)                        # unconstrained minimiser of 0.5 w'Sw
    active, u = [], np.zeros(0)
    cap = 20 * (n + m) + 20
    for _ in range(cap):
        pending = [i for i in range(m) if i not in active]
        if pending:                        # equalities first, oriented so they are "violated"
            p = pending[0]
            sign[p] = -1.0 if C[p] @ x - rhs[p] > 0 else 1.0
        else:
            s = x.copy()
            s[[a - m for a in active if a >= m]] = np.inf
            j = int(np.argmin(s))
            if s[j] >= -1e-13:
                return x
            p = m + j
        npv, bp, uplus = sign[p] * C[p], sign[p] * rhs[p], 0.0
        for _ in range(cap):
            q = len(active)
            if q:
                N = (sign[active][:, None] * C[active]).T
                Qf, Rf = np.linalg.qr(Linv @ N, mode="complete")
                J1, J2 = Linv.T @ Qf[:, :q], Linv.T @ Qf[:, q:]
                r = linalg.solve_triangular(Rf[:q], J1.T @ npv)
            else:
                J2, r = Linv.T, np.zeros(0)
            z = J2 @ (J2.T @ npv)
            t1, k = np.inf, -1
            for jj in range(q):
                if active[jj] >= m and r[jj] > 0 and u[jj] / r[jj] < t1:
                    t1, k = u[jj] / r[jj], jj
            zn = float(z @ npv)
            dependent = not (np.linalg.norm(z) > 1e-14 * np.linalg.norm(npv) and zn > 0)
            t2 = np.inf if dependent else -float(npv @ x - bp) / zn
            t = min(t1, t2)
            if not np.isfinite(t):
                raise ValueError("the long-only constraints are infeasible")
            if not dependent:
                x = x + t * z
            u = u - t * r
            uplus += t
            if not dependent and t == t2:
                active.append(p)
                u = np.append(u, uplus)
                break
            del active[k]
            u = np.delete(u, k)
        else:
            raise RuntimeError("long-only QP (Goldfarb-Idnani) did not converge")
    raise RuntimeError("long-only QP (Goldfarb-Idnani) did not converge")


def _polish(S, A, b, w):
    """Exact equality-constrained re-solve on the support of w (drops weights pushed below 0)."""
    free = w > 1e-10 * max(float(np.abs(w).max()), _TINY)
    for _ in range(S.shape[0]):
        x, _lam = _eq_qp(S, A, b, free)
        if np.all(x[free] >= 0.0):
            return x
        free &= x > 0.0
    return np.clip(w, 0.0, None)


def _kkt_ok(S, A, b, w, tol=1e-9):
    """KKT certificate for min 0.5 w'Sw, A w = b, w >= 0: feasibility, stationarity on the
    support and non-negative bound multipliers (an LP picks lambda when it is not unique)."""
    if np.any(w < 0) or float(np.abs(A @ w - b).max()) > 1e-9 * max(1.0, float(np.abs(b).max())):
        return False
    g = S @ w
    scale = max(float(np.abs(g).max()), _TINY)
    free = w > 0
    AF = A[:, free].T
    lam0 = np.linalg.lstsq(AF, g[free], rcond=None)[0]
    if float(np.abs(AF @ lam0 - g[free]).max()) > tol * scale:
        return False
    fixed = ~free
    if not fixed.any():
        return True
    nu0 = g[fixed] - A[:, fixed].T @ lam0
    _, sv, Vt = np.linalg.svd(AF, full_matrices=True)
    rank = int(np.sum(sv > (sv.max() if sv.size else 0.0) * max(AF.shape) * _EPS))
    Nn = Vt[rank:].T                       # directions along which lambda is not identified
    if Nn.shape[1] == 0:
        return float(nu0.min()) >= -tol * scale
    M = A[:, fixed].T @ Nn                 # nu(s) = nu0 - M s; maximise min_j nu_j(s)
    q = Nn.shape[1]
    lp = optimize.linprog(np.r_[np.zeros(q), -1.0], A_ub=np.hstack([M, np.ones((M.shape[0], 1))]),
                          b_ub=nu0, bounds=[(None, None)] * q + [(None, 0.0)], method="highs")
    return lp.status == 0 and -lp.fun >= -tol * scale


def _qp_nonneg(S, A, b):
    """min 0.5 w'Sw s.t. A w = b, w >= 0 (S symmetric PSD, A (m, n) of full row rank).

    Lawson-Hanson LDP/NNLS first (fast and exact to rounding for cond(S) up to ~1e8), then an
    exact KKT re-solve on the support; if the KKT certificate fails (near-singular S), the
    Goldfarb-Idnani dual active-set method takes over. A singular S gets a 1e-10 relative ridge,
    which makes the minimiser unique (the tie-break among equal-variance portfolios)."""
    n = S.shape[0]
    d = float(np.mean(np.diag(S)))
    S = S / (d if d > 0 else 1.0)          # unit-free: the objective scale does not move the argmin
    try:
        R = np.linalg.cholesky(S).T        # S = R'R
        dg = np.diag(R)
        if (float(dg.min()) / float(dg.max())) ** 2 < 1e-12:
            raise np.linalg.LinAlgError    # numerically singular: regularise as below
    except np.linalg.LinAlgError:
        S = S + 1e-10 * np.eye(n)
        try:
            R = np.linalg.cholesky(S).T
        except np.linalg.LinAlgError:
            raise ValueError("cov is not positive semidefinite") from None
    rs = np.abs(A).max(axis=1)
    rs[rs == 0] = 1.0
    A, b = A / rs[:, None], b / rs         # row-equilibrate the equality constraints
    w = None
    try:
        w = _polish(S, A, b, _qp_ldp(S, R, A, b))
        if _kkt_ok(S, A, b, w):
            return w
    except ValueError:
        pass
    w2 = _polish(S, A, b, _qp_gi(S, A, b))
    if w is None or _kkt_ok(S, A, b, w2) or float(w2 @ S @ w2) <= float(w @ S @ w):
        return w2
    return w


def _returns_2d(returns, name="returns"):
    idx = returns.index if isinstance(returns, (pd.Series, pd.DataFrame)) else None
    cols = list(returns.columns) if isinstance(returns, pd.DataFrame) else None
    a = np.asarray(returns, dtype=float)
    if a.ndim == 1:
        one_d, a = True, a[:, None]
    elif a.ndim == 2:
        one_d = False
    else:
        raise ValueError(f"{name} must be 1-D (T,) or 2-D (T, N) (got shape {a.shape})")
    if a.shape[0] == 0:
        raise ValueError(f"{name} is empty")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} contains NaN/inf; drop or fill them first")
    return a, one_d, cols, idx


def _per_period(x, a, idx, name):
    """A scalar, (T,) or (T, N) per-period rate aligned with the rows of the returns ``a``."""
    if isinstance(x, (pd.Series, pd.DataFrame)) and idx is not None and not x.index.equals(idx):
        raise ValueError(f"{name} and returns have different indexes; align them first")
    v = np.asarray(x, dtype=float)
    if v.ndim == 1:
        if v.shape[0] != a.shape[0]:
            raise ValueError(f"{name} has {v.shape[0]} rows but returns has {a.shape[0]}")
        v = v[:, None]
    elif v.ndim == 2 and v.shape != a.shape:
        raise ValueError(f"{name} shape {v.shape} does not match returns {a.shape}")
    elif v.ndim > 2:
        raise ValueError(f"{name} must be a scalar, (T,) or (T, N)")
    if not np.all(np.isfinite(v)):
        raise ValueError(f"{name} contains NaN/inf")
    return v


def _ppy(periods_per_year):
    if isinstance(periods_per_year, bool):
        raise TypeError("periods_per_year must be a positive number (252, 52, 12, 1, ...)")
    p = float(periods_per_year)
    if not (math.isfinite(p) and p > 0):
        raise ValueError("periods_per_year must be a positive number (252, 52, 12, 1, ...)")
    return p


def _ratio(num, den):
    """num/den column-wise; den == 0 gives sign(num)*inf, or nan when num is 0 too."""
    with np.errstate(divide="ignore", invalid="ignore"):
        r = num / den
    zero = den == 0
    return np.where(zero, np.where(num > 0, np.inf, np.where(num < 0, -np.inf, np.nan)), r)


def _ratio_out(vals, one_d, cols):
    if one_d:
        return float(vals[0])
    if cols is not None:
        return pd.Series(vals, index=cols)
    return vals


def _log_wealth(a, input_type):
    """Log wealth path; for return inputs a starting row 0 (wealth 1 before the first return) is added."""
    if input_type == "prices":
        if np.any(a <= 0):
            raise ValueError("prices must be > 0 (for a P&L or NAV that can go negative, use returns)")
        return np.log(a)
    if input_type == "returns":
        if np.any(a <= -1.0):
            raise ValueError("a simple return <= -1 wipes out wealth; drawdown is then 100%")
        body = np.cumsum(np.log1p(a), axis=0)
    elif input_type == "log_returns":
        body = np.cumsum(a, axis=0)
    else:
        raise ValueError("input_type must be 'prices', 'returns' or 'log_returns'")
    return np.concatenate([np.zeros((1,) + a.shape[1:]), body], axis=0)


# ----------------------------------------------------------------------------
# mean-variance
# ----------------------------------------------------------------------------
def min_variance_weights(cov, *, long_only):
    """Global minimum-variance (GMV) portfolio weights, fully invested (sum to 1).

    Parameters
    ----------
    cov : array (N, N) or DataFrame. Covariance of asset returns (any single period/unit).
        Positive definite when long_only=False.
    long_only : bool, REQUIRED. False: shorts allowed (closed form). True: w_i >= 0 (exact QP).

    Returns
    -------
    w : ndarray (N,), or Series labelled like cov. Sums to 1.

    Formula (Merton 1972): w = Sigma^{-1} 1 / (1' Sigma^{-1} 1); its variance is 1/C with
    C = 1' Sigma^{-1} 1 and its mean A/C with A = 1' Sigma^{-1} mu. Two assets:
    w_1 = (s2^2 - s12) / (s1^2 + s2^2 - 2 s12).
    long_only=True: min w'Sigma w s.t. 1'w = 1, w >= 0.

    References: Markowitz (1952); Merton (1972) JFQA 7(4); Bodie-Kane-Marcus, Investments,
    Ch. 7 (two-fund example: w_D = 0.82, sigma = 11.45%).

    Example
    -------
    >>> w = min_variance_weights(np.array([[0.0144, 0.0072], [0.0072, 0.04]]), long_only=False)
    >>> # array([0.82, 0.18])
    """
    long_only = _bool(long_only, "long_only")
    labels = _common_labels(("cov", cov))
    S = _cov(cov)
    n = S.shape[0]
    if long_only:
        w = _qp_nonneg(S, np.ones((1, n)), np.array([1.0]))
        w = w / w.sum()
    else:
        cf = _chol(S, "min_variance_weights(long_only=False)")
        z = linalg.cho_solve(cf, np.ones(n), check_finite=False)
        w = z / z.sum()
    return _wrap(w, labels)


def tangency_weights(mu, cov, *, rf, long_only):
    """Maximum-Sharpe (tangency) portfolio of the risky assets, weights summing to 1.

    Parameters
    ----------
    mu : array (N,) or Series. Expected (arithmetic) returns, same period as cov and rf.
    cov : array (N, N) or DataFrame. Return covariance (positive definite if long_only=False).
    rf : float, REQUIRED. Risk-free rate for the same period (e.g. annual 0.05 with annual mu).
    long_only : bool, REQUIRED. False: closed form with shorts. True: w_i >= 0 (exact QP).

    Returns
    -------
    w : ndarray (N,), or Series labelled like mu/cov. Sums to 1. Its Sharpe ratio is the
        maximum (mu_p - rf)/sigma_p over all fully-invested (long-only if asked) portfolios.

    Formula (Merton 1972; Campbell-Lo-MacKinlay 1997 Sec. 5.2):
    w = Sigma^{-1}(mu - rf 1) / 1'Sigma^{-1}(mu - rf 1); squared Sharpe = (mu-rf)'Sigma^{-1}(mu-rf).
    The denominator equals C (A/C - rf): a tangency portfolio exists only if rf < A/C, the GMV
    mean; otherwise the Sharpe ratio has no maximum on the efficient branch and ValueError is
    raised. long_only=True solves min y'Sigma y s.t. (mu - rf)'y = 1, y >= 0 and returns
    y / sum(y) (Sharpe is scale-free); it needs at least one mu_i > rf.

    References: Merton (1972); Campbell, Lo & MacKinlay (1997) Ch. 5; Bodie-Kane-Marcus,
    Investments, Ch. 7 (rf 5%: w_D = 0.40, w_E = 0.60, Sharpe 0.42).

    Example
    -------
    >>> cov = np.array([[0.0144, 0.0072], [0.0072, 0.04]])
    >>> tangency_weights([0.08, 0.13], cov, rf=0.05, long_only=False)   # array([0.4, 0.6])
    """
    long_only = _bool(long_only, "long_only")
    labels = _common_labels(("cov", cov), ("mu", mu))
    S = _cov(cov)
    n = S.shape[0]
    m = _vec(mu, "mu", n)
    rf = float(rf)
    if not math.isfinite(rf):
        raise ValueError("rf must be finite")
    e = m - rf
    if long_only:
        if not np.any(e > 0):
            raise ValueError("long-only tangency needs at least one asset with mu > rf")
        y = _qp_nonneg(S, e[None, :], np.array([1.0]))
        w = y / y.sum()
    else:
        cf = _chol(S, "tangency_weights(long_only=False)")
        z = linalg.cho_solve(cf, e, check_finite=False)
        ones = linalg.cho_solve(cf, np.ones(n), check_finite=False)
        denom = z.sum()
        if not denom > 1e-12 * float(np.abs(z).sum()) or not denom > 0:
            gmv_mean = float(m @ ones / ones.sum())
            raise ValueError(
                f"rf={rf:g} is not below the global minimum-variance mean {gmv_mean:g}: no "
                "tangency portfolio exists on the efficient frontier (the Sharpe ratio has no max)")
        w = z / denom
    return _wrap(w, labels)


def target_return_weights(mu, cov, target, *, long_only):
    """Minimum-variance (frontier) portfolio with expected return ``target``, weights summing to 1.

    Parameters
    ----------
    mu : array (N,) or Series. Expected returns, same period as cov and target.
    cov : array (N, N) or DataFrame. Return covariance (positive definite if long_only=False).
    target : float or 1-D array. Target portfolio mean(s), same period and unit as mu.
    long_only : bool, REQUIRED. False: closed form with shorts. True: w_i >= 0 (exact QP);
        then every target must lie in [min(mu), max(mu)], else ValueError.

    Returns
    -------
    w : ndarray (N,) for a scalar target (Series if labelled); for an array of M targets an
        ndarray (M, N) (DataFrame indexed by target if labelled). Each row sums to 1 and has
        w'mu = target.

    Formula (Merton 1972): A = 1'S^{-1}mu, B = mu'S^{-1}mu, C = 1'S^{-1}1, D = BC - A^2;
    w(t) = [(B - A t) S^{-1}1 + (C t - A) S^{-1}mu] / D  (affine in t: two-fund separation);
    variance (C t^2 - 2 A t + B) / D. Needs D > 0 (not all mu equal).
    long_only=True: min w'Sigma w s.t. 1'w = 1, mu'w = t, w >= 0.

    References: Markowitz (1952); Merton (1972) JFQA 7(4); Hull, Risk Management and
    Financial Institutions, Table 1.2 (two-asset risk/return trade-off).

    Example
    -------
    >>> cov = np.array([[0.0256, 0.00768], [0.00768, 0.0576]])
    >>> target_return_weights([0.10, 0.15], cov, 0.12, long_only=False)   # array([0.6, 0.4])
    """
    long_only = _bool(long_only, "long_only")
    labels = _common_labels(("cov", cov), ("mu", mu))
    S = _cov(cov)
    n = S.shape[0]
    m = _vec(mu, "mu", n)
    t = np.asarray(target, dtype=float)
    if t.ndim > 1:
        raise ValueError("target must be a scalar or a 1-D array of targets")
    if not np.all(np.isfinite(t)):
        raise ValueError("target must be finite")
    ts = np.atleast_1d(t)
    if long_only:
        lo, hi = float(m.min()), float(m.max())
        span = max(hi - lo, abs(hi), abs(lo), _TINY) * 1e-12
        if np.any(ts < lo - span) or np.any(ts > hi + span):
            raise ValueError(f"long-only target must lie in [min(mu), max(mu)] = [{lo:g}, {hi:g}]")
        A = np.vstack([np.ones(n), m])
        tol = 1e-12 * max(hi - lo, abs(hi), abs(lo), _TINY)
        rows = []
        for tk in ts:
            tk = float(tk)
            if tk >= hi - tol or tk <= lo + tol:
                # an end of the range: only the assets with that extreme mean can be held, so the
                # problem is the long-only minimum variance over them (no interior point exists)
                J = np.flatnonzero(m >= hi - tol) if tk >= hi - tol else np.flatnonzero(m <= lo + tol)
                w = np.zeros(n)
                w[J] = _qp_nonneg(S[np.ix_(J, J)], np.ones((1, J.size)), np.array([1.0]))
                rows.append(w)
            else:
                rows.append(_qp_nonneg(S, A, np.array([1.0, tk])))
        W = np.array(rows)
    else:
        cf = _chol(S, "target_return_weights(long_only=False)")
        s1 = linalg.cho_solve(cf, np.ones(n), check_finite=False)
        sm = linalg.cho_solve(cf, m, check_finite=False)
        a_, b_, c_ = float(m @ s1), float(m @ sm), float(s1.sum())
        d_ = b_ * c_ - a_ * a_
        if not d_ > 1e-12 * b_ * c_:
            raise ValueError("all expected returns are (nearly) equal: the target pins nothing; "
                             "use min_variance_weights")
        g = (b_ * s1 - a_ * sm) / d_
        h = (c_ * sm - a_ * s1) / d_
        W = g[None, :] + ts[:, None] * h[None, :]
    if t.ndim == 0:
        return _wrap(W[0], labels)
    if labels is not None:
        return pd.DataFrame(W, index=pd.Index(ts, name="target"), columns=labels)
    return W


def risk_parity_weights(cov, *, budgets=None):
    """Long-only risk-budgeting weights: each asset's share of portfolio volatility equals its budget.

    Parameters
    ----------
    cov : array (N, N) or DataFrame. Positive-definite return covariance.
    budgets : array (N,) or Series, optional. Risk budgets b_i > 0 summing to 1 (ValueError
        otherwise). None = equal risk contribution (ERC), b_i = 1/N.

    Returns
    -------
    w : ndarray (N,), or Series labelled like cov. w_i > 0, sums to 1, and
        w_i (Sigma w)_i / (w'Sigma w) = b_i (percent risk contributions equal the budgets).

    Formula (Maillard-Roncalli-Teiletche 2010; Spinu 2013): y* = argmin_{y>0}
    0.5 y'Sigma y - sum_i b_i ln y_i (strictly convex), then w = y*/sum(y*). The first-order
    condition y_i (Sigma y)_i = b_i gives the budgets. Solved by damped Newton to machine
    precision. Special cases (MRT 2010): N = 2 or a constant correlation gives ERC weights
    proportional to 1/sigma_i; sigma(minimum variance) <= sigma(ERC) <= sigma(1/N).

    References: Maillard, Roncalli & Teiletche (2010) J. Portfolio Mgmt 36(4); Spinu (2013)
    "An algorithm for computing risk parity weights".

    Example
    -------
    >>> vol = np.array([0.1, 0.2, 0.3, 0.4]); corr = np.full((4, 4), 0.5) + 0.5 * np.eye(4)
    >>> risk_parity_weights(corr * np.outer(vol, vol))   # array([0.48, 0.24, 0.16, 0.12])
    """
    labels = _common_labels(("cov", cov), ("budgets", budgets))
    S = _cov(cov)
    n = S.shape[0]
    if budgets is None:
        b = np.full(n, 1.0 / n)
    else:
        b = _vec(budgets, "budgets", n)
        if np.any(b <= 0):
            raise ValueError("budgets must be > 0 (drop an asset instead of giving it budget 0)")
        if abs(float(b.sum()) - 1.0) > 1e-9:
            raise ValueError(f"budgets must sum to 1 (got {float(b.sum()):.12g})")
    _chol(S, "risk_parity_weights")
    d = float(np.mean(np.diag(S)))
    Sn = S / d

    def f(y):
        return 0.5 * float(y @ Sn @ y) - float(b @ np.log(y))

    y = np.full(n, 1.0 / math.sqrt(float(Sn.sum())))
    for _ in range(200):
        g = Sn @ y - b / y
        H = Sn + np.diag(b / (y * y))
        dy = np.linalg.solve(H, g)
        dec2 = float(g @ dy)                       # squared Newton decrement
        if dec2 < 1e-8 and np.all(y - dy > 0):
            # quadratic-convergence region: full Newton steps (a line search on f would only
            # see rounding noise here)
            y = y - dy
            if dec2 < 1e-26:
                break
            continue
        step, f0 = 1.0, f(y)
        while True:
            y_new = y - step * dy
            if np.all(y_new > 0) and f(y_new) <= f0 - 0.25 * step * dec2:
                break
            step *= 0.5
            if step < 1e-30:
                raise RuntimeError("risk_parity_weights: Newton line search failed")
        y = y_new
    else:
        raise RuntimeError("risk_parity_weights did not converge")
    return _wrap(y / y.sum(), labels)


def risk_contributions(w, cov, *, kind):
    """Euler decomposition of portfolio volatility into per-asset contributions.

    Parameters
    ----------
    w : array (N,) or Series. Weights (fractions) or money positions; any scale.
    cov : array (N, N) or DataFrame. Return covariance for the horizon of interest.
    kind : str, REQUIRED.
        'marginal'  d sigma_p / d w_i = (Sigma w)_i / sigma_p;
        'component' w_i (Sigma w)_i / sigma_p, sums to sigma_p (same units as w * return);
        'percent'   component / sigma_p, sums to 1.

    Returns
    -------
    ndarray (N,), or Series labelled like w/cov.

    Formula: sigma_p = sqrt(w'Sigma w) (Euler: sum_i w_i d sigma_p/d w_i = sigma_p).
    Component VaR (Jorion) = z_alpha * component (normal, zero mean).

    References: Jorion, Value at Risk 3e (2007) Ch. 7 (marginal / component VaR);
    Maillard, Roncalli & Teiletche (2010).

    Example
    -------
    >>> risk_contributions([2.0, 1.0], np.diag([0.05**2, 0.12**2]), kind='component') * 1.65
    >>> # array([0.10563, 0.15211])  ($M: Jorion's CAD/EUR component VaR)
    """
    if kind not in ("marginal", "component", "percent"):
        raise ValueError("kind must be 'marginal', 'component' or 'percent'")
    labels = _common_labels(("cov", cov), ("w", w))
    S = _cov(cov)
    x = _vec(w, "w", S.shape[0])
    Sx = S @ x
    var = float(x @ Sx)
    if not var > 0:
        raise ValueError("portfolio variance is zero: contributions are undefined")
    sp = math.sqrt(var)
    marginal = Sx / sp
    if kind == "marginal":
        out = marginal
    elif kind == "component":
        out = x * marginal
    else:
        out = x * Sx / var
    return _wrap(out, labels)


# ----------------------------------------------------------------------------
# Black-Litterman
# ----------------------------------------------------------------------------
def bl_implied_returns(cov, w_mkt, *, delta):
    """Reverse-optimised equilibrium EXCESS returns implied by market weights (Black-Litterman prior).

    Parameters
    ----------
    cov : array (N, N) or DataFrame. Excess-return covariance (e.g. annual).
    w_mkt : array (N,) or Series. Market-capitalisation weights (normally summing to 1).
    delta : float, REQUIRED. Risk-aversion coefficient (He-Litterman use 2.5; market-implied
        delta = (E[r_m] - rf) / sigma_m^2).

    Returns
    -------
    pi : ndarray (N,), or Series. Equilibrium excess returns, same period as cov.

    Formula: pi = delta * Sigma @ w_mkt (the market is the mean-variance optimum under pi).

    References: Black & Litterman (1992) FAJ; He & Litterman (1999) Goldman Sachs; Idzorek
    (2005).

    Example
    -------
    >>> pi = bl_implied_returns(cov, w_mkt, delta=2.5)
    """
    labels = _common_labels(("cov", cov), ("w_mkt", w_mkt))
    S = _cov(cov)
    w = _vec(w_mkt, "w_mkt", S.shape[0])
    delta = float(delta)
    if not (math.isfinite(delta) and delta > 0):
        raise ValueError("delta must be a positive finite number")
    return _wrap(delta * (S @ w), labels)


def black_litterman(cov, pi, P, Q, *, tau, omega, delta=None):
    """Black-Litterman posterior: mean, covariance of the mean, predictive covariance, optimal weights.

    Parameters
    ----------
    cov : array (N, N) or DataFrame. Excess-return covariance Sigma.
    pi : array (N,) or Series. Prior (equilibrium) excess returns, e.g. bl_implied_returns.
    P : array (K, N) (or (N,) for one view) or DataFrame with asset columns. View portfolios.
    Q : array (K,) or float. View returns (absolute or relative), same period as pi.
    tau : float > 0, REQUIRED. Uncertainty scale of the prior mean (prior pi ~ N(mu, tau Sigma)).
    omega : REQUIRED. View-error covariance: a (K, K) array, a (K,) array = its diagonal (view
        variances), a float when K = 1, or 'he_litterman' = diag(P tau Sigma P').
    delta : float, optional. Risk aversion; if given, 'weights' holds the He-Litterman
        unconstrained optimum w* = (delta (Sigma + M))^{-1} mu_BL (else None). Idzorek (2005)
        uses Sigma instead of Sigma + M: solve(delta * cov, mean) gives that variant.

    Returns
    -------
    dict with
        'mean'      mu_BL (N,), posterior expected excess returns;
        'cov_mean'  M (N, N), posterior covariance of the mean;
        'cov_pred'  Sigma + M (N, N), predictive covariance of returns;
        'omega'     Omega (K, K) actually used;
        'weights'   (N,) or None.
    Series/DataFrames labelled by asset when cov/pi/P carry labels.

    Formula (He-Litterman 1999; Meucci 2010), with Omega possibly singular:
    mu_BL = pi + tau Sigma P' (P tau Sigma P' + Omega)^{-1} (Q - P pi)
    M = tau Sigma - tau Sigma P' (P tau Sigma P' + Omega)^{-1} P tau Sigma
      (= ((tau Sigma)^{-1} + P' Omega^{-1} P)^{-1} when Omega is invertible).
    With pi = delta Sigma w_eq: w* = (w_eq + P' Lambda) / (1 + tau) (He-Litterman), so assets
    outside every view keep w_eq / (1 + tau).

    References: Black & Litterman (1992); He & Litterman (1999) "The Intuition Behind
    Black-Litterman Model Portfolios"; Idzorek (2005); Meucci (2010) "The Black-Litterman
    approach: original model and user specifications".

    Example
    -------
    >>> P = [[0, 0, -0.295, 1, 0, -0.705, 0]]          # Germany vs rest of Europe
    >>> bl = black_litterman(cov, pi, P, [0.05], tau=0.05, omega='he_litterman', delta=2.5)
    >>> bl['mean'], bl['weights']
    """
    labels = _common_labels(("cov", cov), ("pi", pi))
    S = _cov(cov)
    n = S.shape[0]
    p = _vec(pi, "pi", n)
    if isinstance(P, pd.DataFrame):
        if labels is not None and list(P.columns) != labels:
            raise ValueError("P columns and the asset labels of cov/pi differ; align them first")
        if labels is None:
            labels = list(P.columns)
    view_labels = list(P.index) if isinstance(P, pd.DataFrame) else None
    Pm = np.asarray(P, dtype=float)
    if Pm.ndim == 1:
        Pm = Pm[None, :]
    if Pm.ndim != 2 or Pm.shape[1] != n or Pm.shape[0] == 0:
        raise ValueError(f"P must be (K, {n}) with K >= 1 views (got shape {Pm.shape})")
    k = Pm.shape[0]
    q = np.atleast_1d(np.asarray(Q, dtype=float)).ravel()
    if q.shape != (k,):
        raise ValueError(f"Q must have one entry per view (K={k}); got shape {q.shape}")
    if not (np.all(np.isfinite(Pm)) and np.all(np.isfinite(q))):
        raise ValueError("P and Q must be finite")
    tau = float(tau)
    if not (math.isfinite(tau) and tau > 0):
        raise ValueError("tau must be a positive finite number")
    tS = tau * S
    PtS = Pm @ tS
    V = PtS @ Pm.T
    V = 0.5 * (V + V.T)
    if isinstance(omega, str):
        if omega != "he_litterman":
            raise ValueError("omega must be an array, a float (K=1) or 'he_litterman'")
        Om = np.diag(np.diag(V))
    else:
        o = np.asarray(omega, dtype=float)
        if o.ndim == 0 and k == 1:
            Om = o.reshape(1, 1)
        elif o.ndim == 1 and o.shape == (k,):
            Om = np.diag(o)
        elif o.shape == (k, k):
            if float(np.abs(o - o.T).max()) > 1e-10 * max(float(np.abs(o).max()), _TINY):
                raise ValueError("omega must be symmetric")
            Om = 0.5 * (o + o.T)
        else:
            raise ValueError(f"omega must be (K, K), (K,) or 'he_litterman' (K={k}); got {o.shape}")
        if not np.all(np.isfinite(Om)) or np.any(np.diag(Om) < 0):
            raise ValueError("omega must be finite with non-negative view variances")
    try:
        cf = linalg.cho_factor(V + Om, lower=True, check_finite=False)
    except linalg.LinAlgError:
        raise ValueError("P tau Sigma P' + Omega is singular: the views are linearly dependent "
                         "and certain (Omega = 0); drop a redundant view") from None
    mean = p + PtS.T @ linalg.cho_solve(cf, q - Pm @ p, check_finite=False)
    M = tS - PtS.T @ linalg.cho_solve(cf, PtS, check_finite=False)
    M = 0.5 * (M + M.T)
    cov_pred = S + M
    weights = None
    if delta is not None:
        delta = float(delta)
        if not (math.isfinite(delta) and delta > 0):
            raise ValueError("delta must be a positive finite number")
        weights = linalg.solve(delta * cov_pred, mean, assume_a="pos")
    if labels is not None:
        mean = pd.Series(mean, index=labels)
        M = pd.DataFrame(M, index=labels, columns=labels)
        cov_pred = pd.DataFrame(cov_pred, index=labels, columns=labels)
        if weights is not None:
            weights = pd.Series(weights, index=labels)
    if view_labels is not None:
        Om = pd.DataFrame(Om, index=view_labels, columns=view_labels)
    return {"mean": mean, "cov_mean": M, "cov_pred": cov_pred, "omega": Om, "weights": weights}


# ----------------------------------------------------------------------------
# Kelly
# ----------------------------------------------------------------------------
def kelly_bet(p, b, *, a=1.0):
    """Kelly (growth-optimal) fraction of wealth to stake on a binary bet.

    Parameters
    ----------
    p : float or array in [0, 1]. Probability of winning.
    b : float or array > 0. Net gain per unit staked when the bet wins (odds b-to-1).
    a : float or array > 0. Loss per unit staked when it loses (1.0 = the whole stake).

    Returns
    -------
    f : numpy float64 (scalar inputs) or ndarray. The formula's value is returned UNCLIPPED:
        f <= 0 means no favourable bet (do not bet, or take the other side).

    Formula: maximise g(f) = p ln(1 + b f) + (1 - p) ln(1 - a f)  ->  f* = p/a - (1 - p)/b.
    Even money (a = b = 1): f* = p - q (Kelly 1956).

    References: Kelly (1956) Bell System Technical Journal 35; Thorp (2006) "The Kelly
    criterion in blackjack, sports betting and the stock market".

    Example
    -------
    >>> kelly_bet(0.6, 1.0)          # 0.2 of wealth on a 60% even-money bet
    >>> kelly_bet(0.4, 2.0)          # 0.1 on a 40% bet paying 2-to-1
    """
    p = np.asarray(p, dtype=float)
    b = np.asarray(b, dtype=float)
    a = np.asarray(a, dtype=float)
    if np.any(~((p >= 0) & (p <= 1))):
        raise ValueError("p must be a probability in [0, 1]")
    if np.any(~(b > 0)) or np.any(~np.isfinite(b)):
        raise ValueError("b (net win per unit staked) must be > 0 and finite")
    if np.any(~(a > 0)) or np.any(~np.isfinite(a)):
        raise ValueError("a (loss per unit staked) must be > 0 and finite")
    return _out(p / a - (1.0 - p) / b)


def kelly_weights(mu, cov, *, rf):
    """Growth-optimal (continuous-time Kelly) portfolio weights of lognormal risky assets.

    Parameters
    ----------
    mu : float, or array (N,) / Series. Expected ARITHMETIC return rates (the drift of dS/S),
        same period as cov and rf. From a mean log return m use mu = m + sigma^2/2.
    cov : float (the VARIANCE sigma^2 when mu is a float), or array (N, N) / DataFrame.
    rf : float, REQUIRED. Risk-free rate for the same period.

    Returns
    -------
    f : float for scalar inputs, else ndarray (N,) / Series. Fractions of wealth in each risky
        asset; 1 - sum(f) is in the risk-free asset (sum > 1 = leverage). Not normalised.

    Formula: maximise g(f) = rf + f'(mu - rf) - f'Sigma f / 2  ->  f* = Sigma^{-1}(mu - rf)
    (scalar: (mu - rf)/sigma^2), with growth g* = rf + (mu-rf)'Sigma^{-1}(mu-rf)/2. f*/sum(f*)
    is the tangency portfolio. Merton (1969) with log utility.

    References: Merton (1969) REStat 51(3); Thorp (2006) Sec. 7; MacLean, Thorp & Ziemba (2011).

    Example
    -------
    >>> kelly_weights(0.11, 0.15**2, rf=0.06)           # 2.22: lever 2.22x into the index
    >>> kelly_weights(mu_vec, cov_mat, rf=0.03)
    """
    rf = float(rf)
    if not math.isfinite(rf):
        raise ValueError("rf must be finite")
    pandas_in = isinstance(mu, (pd.Series, pd.DataFrame)) or isinstance(cov, pd.DataFrame)
    if np.ndim(mu) == 0 and np.ndim(cov) == 0 and not pandas_in:
        var = float(cov)
        m = float(mu)
        if not (math.isfinite(var) and var > 0) or not math.isfinite(m):
            raise ValueError("scalar kelly_weights needs a finite mu and a variance cov > 0")
        return (m - rf) / var
    labels = _common_labels(("cov", cov), ("mu", mu))
    S = _cov(cov)
    m = _vec(mu, "mu", S.shape[0])
    cf = _chol(S, "kelly_weights")
    return _wrap(linalg.cho_solve(cf, m - rf, check_finite=False), labels)


# ----------------------------------------------------------------------------
# performance ratios
# ----------------------------------------------------------------------------
def sharpe_ratio(returns, *, rf, periods_per_year, ddof):
    """Annualised Sharpe ratio of per-period excess returns.

    Parameters
    ----------
    returns : array (T,) / (T, N), Series or DataFrame. Per-period simple returns (decimals).
    rf : REQUIRED. Risk-free rate PER PERIOD (e.g. annual 3% on daily data: 0.03/252 or
        (1.03)**(1/252) - 1, your choice): a float, or (T,) / (T, N) aligned with returns.
        Pass 0.0 for returns that are already excess returns.
    periods_per_year : float, REQUIRED. 252, 52, 12, ... ; 1 returns the per-period ratio.
    ddof : int, REQUIRED. Standard-deviation divisor T - ddof: 1 = sample sd (Sharpe 1994,
        pandas), 0 = population sd (numpy default).

    Returns
    -------
    float for 1-D input; ndarray (N,) (Series for a DataFrame) for 2-D input.
    A constant excess series gives +-inf (nan if it is identically zero).

    Formula (Sharpe 1994): d_t = r_t - rf_t;  SR = mean(d) / sd(d; ddof) * sqrt(periods_per_year).
    The sqrt scaling assumes iid returns (Lo 2002 gives the autocorrelation correction).

    References: Sharpe (1966, 1994) J. Portfolio Mgmt; Lo (2002) FAJ 58(4).

    Example
    -------
    >>> sharpe_ratio(daily_returns, rf=0.0, periods_per_year=252, ddof=1)
    """
    a, one_d, cols, idx = _returns_2d(returns)
    r = _per_period(rf, a, idx, "rf")
    ppy = _ppy(periods_per_year)
    if isinstance(ddof, bool) or not isinstance(ddof, (int, np.integer)) or ddof < 0:
        raise ValueError("ddof must be an integer >= 0 (1 = sample sd, 0 = population sd)")
    T = a.shape[0]
    if T - ddof <= 0:
        raise ValueError(f"need more than ddof={ddof} observations (got T={T})")
    x = a - r
    m = x.mean(axis=0)
    sd = np.sqrt(((x - m) ** 2).sum(axis=0) / (T - ddof))
    # a constant series leaves rounding noise in sd; treat that as exactly zero
    sd = np.where(sd <= 64 * _EPS * np.abs(x).max(axis=0), 0.0, sd)
    return _ratio_out(_ratio(m, sd) * math.sqrt(ppy), one_d, cols)


def sortino_ratio(returns, *, target, periods_per_year, downside):
    """Annualised Sortino ratio: mean return in excess of a target over downside deviation.

    Parameters
    ----------
    returns : array (T,) / (T, N), Series or DataFrame. Per-period simple returns (decimals).
    target : REQUIRED. Minimum acceptable return (MAR) PER PERIOD: a float (0.0 or rf per
        period) or (T,) / (T, N) aligned with returns.
    periods_per_year : float, REQUIRED. 252, 52, 12, ... ; 1 returns the per-period ratio.
    downside : str, REQUIRED. Divisor of the downside semi-variance:
        'full'         all T periods (Sortino-Price 1994; empyrical, PerformanceAnalytics);
        'below_target' only the periods with r_t < target.

    Returns
    -------
    float for 1-D input; ndarray (N,) (Series for a DataFrame) for 2-D input.
    +inf when no period is below target and the mean excess is positive (nan if it is zero).

    Formula: d_t = r_t - target_t; DD = sqrt(sum_t min(d_t, 0)^2 / n) with n = T ('full') or
    #{d_t < 0} ('below_target'); Sortino = mean(d) / DD * sqrt(periods_per_year).

    References: Sortino & Price (1994) J. Investing 3(3); Sortino & van der Meer (1991).

    Example
    -------
    >>> sortino_ratio(daily_returns, target=0.0, periods_per_year=252, downside='full')
    """
    if downside not in ("full", "below_target"):
        raise ValueError("downside must be 'full' or 'below_target'")
    a, one_d, cols, idx = _returns_2d(returns)
    tg = _per_period(target, a, idx, "target")
    ppy = _ppy(periods_per_year)
    x = a - tg
    neg = np.minimum(x, 0.0)
    ss = (neg * neg).sum(axis=0)
    if downside == "full":
        n = np.full(x.shape[1], float(x.shape[0]))
    else:
        n = (x < 0).sum(axis=0).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = np.where(n > 0, np.sqrt(ss / np.where(n > 0, n, 1.0)), 0.0)
    m = x.mean(axis=0)
    return _ratio_out(_ratio(m, dd) * math.sqrt(ppy), one_d, cols)


def max_drawdown(x, *, input_type):
    """Maximum drawdown of a wealth path, with its peak, trough and the drawdown series.

    Parameters
    ----------
    x : array (T,) or Series. One path only (loop over columns yourself).
    input_type : str, REQUIRED.
        'prices'      x is a price / NAV level (> 0); the path is x itself;
        'returns'     x are per-period simple returns (> -1); wealth starts at 1 BEFORE the
                      first return, so a first-period loss is a drawdown (empyrical convention;
                      to start at the first observation instead pass prices=cumprod(1+r));
        'log_returns' x are per-period log returns; wealth exp(cumsum), starting at 1.

    Returns
    -------
    dict with
        'max_drawdown' float >= 0, the largest fall from a running peak: 0.25 = -25%;
        'peak'   label (Series) or position (ndarray) of the LAST running-peak observation before
                 the trough; None if the peak is the starting wealth of a return input, or if
                 there is no drawdown;
        'trough' label/position of the FIRST observation at the maximum drawdown (None if none);
        'drawdown' ndarray (T,) / Series of 1 - W_t / max_{s<=t} W_s >= 0, aligned with x.

    Formula: DD_t = 1 - W_t / max_{s<=t} W_s;  MDD = max_t DD_t (computed in logs, no overflow).
    For driftless Brownian log wealth, E[-ln(1 - MDD)] = sqrt(pi/2) sigma sqrt(T)
    (Magdon-Ismail et al. 2004).

    References: Magdon-Ismail, Atiya, Pratap & Abu-Mostafa (2004) J. Appl. Prob. 41(1);
    Young (1991).

    Example
    -------
    >>> max_drawdown([100, 120, 90, 110, 80, 130], input_type='prices')['max_drawdown']  # 1/3
    """
    idx = x.index if isinstance(x, pd.Series) else None
    if isinstance(x, pd.DataFrame):
        raise ValueError("max_drawdown takes one series; loop over the DataFrame columns")
    a = np.asarray(x, dtype=float)
    if a.ndim != 1 or a.shape[0] == 0:
        raise ValueError(f"x must be a non-empty 1-D series (got shape {a.shape})")
    if not np.all(np.isfinite(a)):
        raise ValueError("x contains NaN/inf; drop or fill them first")
    lw = _log_wealth(a, input_type)
    under = lw - np.maximum.accumulate(lw)          # <= 0, exactly 0 at running peaks
    dd_path = 0.0 - np.expm1(under)                  # 0.0 - x: no -0.0 entries
    offset = 0 if input_type == "prices" else 1     # path position of x[0]
    dd = dd_path[offset:]
    t_path = int(np.argmax(dd_path))
    mdd = float(dd_path[t_path])
    peak = trough = None
    if mdd > 0:
        p_path = int(np.flatnonzero(under[: t_path + 1] == 0.0)[-1])
        trough = t_path - offset
        peak = None if p_path - offset < 0 else p_path - offset
        if idx is not None:
            trough = idx[trough]
            peak = None if peak is None else idx[peak]
    if idx is not None:
        dd = pd.Series(dd, index=idx, name="drawdown")
    return {"max_drawdown": mdd, "peak": peak, "trough": trough, "drawdown": dd}


def calmar_ratio(returns, *, periods_per_year, ann_return):
    """Calmar ratio: annualised return over the maximum drawdown of the SAME supplied sample.

    Parameters
    ----------
    returns : array (T,) / (T, N), Series or DataFrame. Per-period simple returns (> -1).
        Young (1991) uses the trailing 36 months: slice the sample yourself.
    periods_per_year : float, REQUIRED. 252, 52, 12, ...
    ann_return : str, REQUIRED.
        'cagr'       compound annual growth (prod(1 + r))^(periods_per_year / T) - 1;
        'arithmetic' mean(r) * periods_per_year.

    Returns
    -------
    float for 1-D input; ndarray (N,) (Series for a DataFrame) for 2-D input.
    +inf if there is no drawdown and the annual return is positive (nan if it is zero).

    Formula: Calmar = annual return / MDD, with MDD from max_drawdown(r, input_type='returns')
    (wealth starts at 1 before the first return).

    References: Young (1991) "Calmar ratio: a smoother tool", Futures 20(1);
    Magdon-Ismail & Atiya (2004) Risk 17(10).

    Example
    -------
    >>> calmar_ratio(monthly_returns[-36:], periods_per_year=12, ann_return='cagr')
    """
    if ann_return not in ("cagr", "arithmetic"):
        raise ValueError("ann_return must be 'cagr' or 'arithmetic'")
    a, one_d, cols, _ = _returns_2d(returns)
    ppy = _ppy(periods_per_year)
    lw = _log_wealth(a, "returns")
    mdd = (0.0 - np.expm1(lw - np.maximum.accumulate(lw, axis=0))).max(axis=0)
    T = a.shape[0]
    if ann_return == "cagr":
        ann = np.expm1(lw[-1] * (ppy / T))
    else:
        ann = a.mean(axis=0) * ppy
    return _ratio_out(_ratio(ann, mdd), one_d, cols)
