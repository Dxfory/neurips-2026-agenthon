"""Linear-regression econometrics: OLS with robust covariances, diagnostics, GRS, rolling OLS.

Conventions for the whole module:
- Rows are observations in TIME ORDER (HAC and Durbin-Watson depend on it).
- ``k`` always counts every estimated coefficient INCLUDING the intercept.
- ``add_const=True`` puts the intercept in the FIRST column (params[0]).
- Inputs must be finite; drop missing rows first (``rolling_ols`` is the one
  exception: it drops missing rows inside each window).
- pandas in -> labelled pandas out; numpy in -> numpy out. When two inputs are
  both pandas their indexes must be identical (raises instead of pairing rows by
  position); align/join them first.

Functions
---------
ols              OLS with classical / HC0 / HC1 / Newey-West HAC covariance
cov_ols          OLS coefficient covariance matrix (classical, HC0, HC1, HAC)
newey_west_lags  automatic HAC lag length ('nw1994' or 'sw' rule of thumb)
t_test           standard errors, t-statistics, two-sided p-values
r_squared        R^2 and adjusted R^2 (centered or uncentered)
durbin_watson    Durbin-Watson statistic of residuals
grs_test         Gibbons-Ross-Shanken test that N alphas are jointly zero
vif              variance inflation factors
rolling_ols      rolling-window OLS coefficients
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy import stats

__all__ = [
    "ols",
    "cov_ols",
    "newey_west_lags",
    "t_test",
    "r_squared",
    "durbin_watson",
    "grs_test",
    "vif",
    "rolling_ols",
]

_EPS = np.finfo(float).eps
_KINDS = {"classical": "classical", "hc0": "HC0", "hc1": "HC1", "hac": "HAC"}


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _as_vector(x, name):
    a = np.asarray(x, dtype=float)
    if a.ndim == 2 and a.shape[1] == 1:
        a = a[:, 0]
    if a.ndim != 1:
        raise ValueError(f"{name} must be 1-D (got shape {a.shape})")
    return a


def _as_matrix(x, name):
    a = np.asarray(x, dtype=float)
    if a.ndim == 1:
        a = a[:, None]
    if a.ndim != 2:
        raise ValueError(f"{name} must be 1-D or 2-D (got shape {a.shape})")
    return a


def _labels(X):
    """Column labels of a pandas X, else None."""
    if isinstance(X, pd.DataFrame):
        return list(X.columns)
    if isinstance(X, pd.Series):
        return [X.name if X.name is not None else "x1"]
    return None


def _index(*objs):
    """Index of the pandas inputs; raises if two pandas inputs carry different indexes
    (numpy conversion would otherwise pair rows by position, silently misaligning them)."""
    idx = None
    for o in objs:
        if isinstance(o, (pd.Series, pd.DataFrame)):
            if idx is None:
                idx = o.index
            elif not idx.equals(o.index):
                raise ValueError(
                    "pandas inputs have different indexes; align them first "
                    "(e.g. df = pd.concat([y, X], axis=1, join='inner').dropna())"
                )
    return idx


def _require_finite(a, name):
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} contains NaN/inf; drop or fill missing rows first")


def _const_columns(X):
    """Boolean mask of columns that are constant and non-zero."""
    if X.shape[0] == 0:
        return np.zeros(X.shape[1], dtype=bool)
    return (np.ptp(X, axis=0) == 0) & (X[0] != 0)


def _unit_columns(X):
    """X with every non-zero column scaled to unit Euclidean norm (rank is scale-free)."""
    d = np.sqrt(np.einsum("ij,ij->j", X, X))
    return X / np.where(d > 0, d, 1.0), d


def _has_constant(X):
    """True if X has an explicit constant column or spans one (statsmodels k_constant rule)."""
    if _const_columns(X).any():
        return True
    Xn = _unit_columns(X)[0]
    aug = np.column_stack([np.full(X.shape[0], 1.0 / math.sqrt(X.shape[0])), Xn])
    return np.linalg.matrix_rank(aug) == np.linalg.matrix_rank(Xn)


def _xtx_inv(X):
    """(X'X)^{-1} from the SVD of the column-equilibrated X; raises if X is rank deficient.

    Columns are scaled to unit norm first, so the rank test does not depend on units
    (e.g. volume in shares next to returns in decimals)."""
    n, k = X.shape
    Xs, d = _unit_columns(X)
    _, s, vt = np.linalg.svd(Xs, full_matrices=False)
    if (d == 0).any() or s.size < k or s[-1] <= s[0] * max(n, k) * _EPS:
        raise ValueError(
            "X is rank deficient (perfectly collinear columns, e.g. a constant "
            "column plus add_const=True, or a full set of dummies plus a constant)"
        )
    v = vt.T / s
    return (v @ v.T) / np.outer(d, d)


def _norm_kind(kind):
    key = str(kind).lower()
    if key not in _KINDS:
        raise ValueError("cov kind must be 'classical', 'HC0', 'HC1' or 'HAC'")
    return _KINDS[key]


def _resolve_lags(lags, n):
    if lags is None:
        raise ValueError("HAC needs lags: an int >= 0, or 'nw1994' / 'sw' (see newey_west_lags)")
    if isinstance(lags, str):
        return newey_west_lags(n, rule=lags)
    if isinstance(lags, bool) or not isinstance(lags, (int, np.integer, float, np.floating)):
        raise TypeError("lags must be an int >= 0 or 'nw1994' / 'sw'")
    if lags != int(lags):
        raise ValueError(f"lags must be a whole number (got {lags})")
    lags = int(lags)
    if lags < 0 or lags >= n:
        raise ValueError(f"lags must satisfy 0 <= lags < n (got {lags}, n={n})")
    return lags


def _hac_meat(X, e, lags):
    """Newey-West S = Gamma_0 + sum_j (1 - j/(L+1)) (Gamma_j + Gamma_j'), unscaled (no 1/n)."""
    k = X.shape[1]
    xe = X * e[:, None]
    S = xe.T @ xe
    for j in range(1, lags + 1):
        g = (X[j:] * (e[j:] * e[:-j])[:, None]).T @ X[:-j]
        assert g.shape == (k, k)
        S += (1.0 - j / (lags + 1.0)) * (g + g.T)
    return S


# ----------------------------------------------------------------------------
# public API
# ----------------------------------------------------------------------------
def newey_west_lags(n, *, rule, rounding="floor"):
    """Automatic Newey-West (Bartlett kernel) lag length from a named rule of thumb.

    Parameters
    ----------
    n : int
        Number of observations (>= 1).
    rule : {'nw1994', 'sw'}  (required)
        'nw1994': L = 4 (n/100)^(2/9)  (Newey-West 1994 rule of thumb; what
                  statsmodels cov_hac / S_hac_simple use when nlags=None).
        'sw':     L = m = 0.75 n^(1/3) (Stock-Watson rule of thumb, returned as the
                  lag count, which is how it is usually passed as maxlags).
                  CAUTION off-by-one: Stock-Watson's own estimator uses weights
                  (m-j)/m for j = 1..m-1, which equals lags = m - 1 in this module's
                  1 - j/(L+1) convention; subtract 1 to reproduce the textbook exactly.
    rounding : {'floor', 'round', 'ceil'}, default 'floor'
        How the real-valued rule is turned into an integer ('round' = half up).
        'floor' matches statsmodels; Stock-Watson say "rounded to an integer".
        A 1e-9 guard stops float noise from moving exact integers or halves
        (0.75*64^(1/3) = 3 -> 3; 0.75*1000^(1/3) = 7.5 -> 'round' gives 8).

    Returns
    -------
    int, the lag count L for cov_ols / ols (Bartlett weights 1 - j/(L+1), j = 1..L).

    Example
    -------
    >>> newey_west_lags(600, rule='nw1994')     # 4*6^(2/9) = 5.957 -> 5
    5
    """
    if isinstance(n, bool) or int(n) != n or n < 1:
        raise ValueError("n must be a positive integer")
    n = int(n)
    if rule == "nw1994":
        x = 4.0 * (n / 100.0) ** (2.0 / 9.0)
    elif rule == "sw":
        x = 0.75 * math.cbrt(n)
    else:
        raise ValueError("rule must be 'nw1994' or 'sw'")
    if rounding == "floor":
        return int(math.floor(x + 1e-9))
    if rounding == "ceil":
        return int(math.ceil(x - 1e-9))
    if rounding == "round":
        return int(math.floor(x + 0.5 + 1e-9))
    raise ValueError("rounding must be 'floor', 'round' or 'ceil'")


def cov_ols(X, resid, *, kind, lags=None, small_sample=False):
    """OLS coefficient covariance matrix: classical, White HC0/HC1 or Newey-West HAC.

    Parameters
    ----------
    X : array (n, k)
        The design matrix exactly as estimated, INCLUDING the constant column if any.
    resid : array (n,)
        OLS residuals e = y - X b (time-ordered for HAC).
    kind : {'classical', 'HC0', 'HC1', 'HAC'}  (required, case-insensitive)
        classical: sigma2 (X'X)^-1 with sigma2 = e'e/(n-k).
        HC0: White (X'X)^-1 X' diag(e^2) X (X'X)^-1.   HC1: HC0 * n/(n-k).
        HAC: Newey-West (X'X)^-1 S (X'X)^-1 with Bartlett weights 1 - j/(L+1).
    lags : int >= 0 or {'nw1994', 'sw'}, required for HAC only
        Bartlett truncation lag L (a string applies newey_west_lags with floor).
        lags=0 gives HC0.
    small_sample : bool, default False  (HAC only)
        True multiplies the HAC matrix by n/(n-k) (statsmodels use_correction=True).
        Default False = Newey-West (1987) = statsmodels fit(cov_type='HAC').

    Returns
    -------
    ndarray (k, k), symmetric positive semi-definite (DataFrame labelled by
    X's columns when X is a DataFrame).

    Formula
    -------
    S = Gamma_0 + sum_{j=1..L} (1 - j/(L+1)) (Gamma_j + Gamma_j'),
    Gamma_j = sum_{t=j+1..n} e_t e_{t-j} x_t x_{t-j}'   (no 1/n; it cancels).

    Example
    -------
    >>> V = cov_ols(Xc, e, kind='HAC', lags=5); se = np.sqrt(np.diag(V))
    """
    _index(X, resid)
    labels = list(X.columns) if isinstance(X, pd.DataFrame) else None
    Xa = _as_matrix(X, "X")
    e = _as_vector(resid, "resid")
    n, k = Xa.shape
    if e.shape[0] != n:
        raise ValueError(f"resid has {e.shape[0]} rows but X has {n}")
    if n <= k:
        raise ValueError(f"need n > k (n={n}, k={k})")
    _require_finite(Xa, "X")
    _require_finite(e, "resid")
    kind = _norm_kind(kind)
    if kind != "HAC":
        if lags is not None:
            raise ValueError("lags is only used with kind='HAC'")
        if small_sample:
            raise ValueError("small_sample applies to HAC only (use kind='HC1' for the n/(n-k) White estimator)")
    xtxi = _xtx_inv(Xa)
    if kind == "classical":
        cov = xtxi * (e @ e / (n - k))
    else:
        if kind == "HAC":
            meat = _hac_meat(Xa, e, _resolve_lags(lags, n))
        else:
            xe = Xa * e[:, None]
            meat = xe.T @ xe
        cov = xtxi @ meat @ xtxi
        if kind == "HC1" or (kind == "HAC" and small_sample):
            cov = cov * (n / (n - k))
    cov = 0.5 * (cov + cov.T)
    if labels is not None:
        return pd.DataFrame(cov, index=labels, columns=labels)
    return cov


def t_test(params, cov, *, df=None, null=0.0):
    """Standard errors, t-statistics and two-sided p-values from coefficients and their covariance.

    Parameters
    ----------
    params : array (k,) or Series
        Estimated coefficients b.
    cov : array (k, k) or DataFrame
        Covariance matrix of b (e.g. from cov_ols). A DataFrame is reordered to
        params' labels when params is a Series (raises if the labels differ).
    df : int or None, default None
        None -> standard normal reference (NOTE: ols defaults to t(n-k));
        a number -> Student t(df) (df = n - k for OLS).
    null : float or array (k,), default 0.0
        Hypothesised value(s) b0.

    Returns
    -------
    dict with 'se', 't', 'pvalue' (Series if params is a Series).

    Formula
    -------
    se = sqrt(diag cov); t = (b - b0)/se; p = 2 * P(Z > |t|) (Z ~ N(0,1) or t(df)).

    Example
    -------
    >>> t_test(b, V, df=n - k)['pvalue']
    """
    if isinstance(params, pd.Series) and isinstance(cov, pd.DataFrame):
        lab = params.index
        if not (set(cov.index) == set(lab) and set(cov.columns) == set(lab)):
            raise ValueError("cov labels do not match params labels")
        cov = cov.loc[lab, lab]
    b = _as_vector(params, "params")
    C = np.asarray(cov, dtype=float)
    if C.ndim == 0 and b.size == 1:
        C = C.reshape(1, 1)
    if C.shape != (b.size, b.size):
        raise ValueError(f"cov must be ({b.size}, {b.size}), got {C.shape}")
    se = np.sqrt(np.diag(C))
    t = (b - np.asarray(null, dtype=float)) / se
    if df is None:
        p = 2.0 * stats.norm.sf(np.abs(t))
    else:
        if df <= 0:
            raise ValueError("df must be positive")
        p = 2.0 * stats.t.sf(np.abs(t), df)
    if isinstance(params, pd.Series):
        idx = params.index
        return {"se": pd.Series(se, index=idx), "t": pd.Series(t, index=idx),
                "pvalue": pd.Series(p, index=idx)}
    return {"se": se, "t": t, "pvalue": p}


def r_squared(y, fitted, *, k, centered=True):
    """R^2 and adjusted R^2 of a fitted regression.

    Parameters
    ----------
    y, fitted : array (n,)
        Observed and fitted values.
    k : int  (required)
        Number of estimated coefficients INCLUDING the intercept.
    centered : bool, default True
        True: SST = sum (y - ybar)^2 (model with an intercept).
        False: uncentered SST = sum y^2 (no-intercept model; statsmodels' rule
        when there is no constant).

    Returns
    -------
    (r2, adj_r2) : tuple of float

    Formula
    -------
    R2 = 1 - SSR/SST;  adj = 1 - (1 - R2)(n - 1)/(n - k) when centered,
    and 1 - (1 - R2) n/(n - k) when uncentered (statsmodels).

    Example
    -------
    >>> r2, adj = r_squared(y, X @ b, k=X.shape[1])
    """
    _index(y, fitted)
    yv = _as_vector(y, "y")
    f = _as_vector(fitted, "fitted")
    if yv.shape != f.shape:
        raise ValueError("y and fitted must have the same length")
    n = yv.size
    if n <= k:
        raise ValueError(f"need n > k (n={n}, k={k})")
    resid = yv - f
    ssr = float(resid @ resid)
    if centered:
        d = yv - yv.mean()
        sst = float(d @ d)
        m = n - 1
    else:
        sst = float(yv @ yv)
        m = n
    r2 = 1.0 - ssr / sst
    adj = 1.0 - (1.0 - r2) * m / (n - k)
    return r2, adj


def durbin_watson(resid):
    """Durbin-Watson statistic of regression residuals (about 2 = no AR(1), < 2 = positive autocorrelation).

    Parameters
    ----------
    resid : array (n,), time-ordered residuals.

    Returns
    -------
    float

    Formula
    -------
    DW = sum_{t>=2} (e_t - e_{t-1})^2 / sum_t e_t^2  (approximately 2(1 - rho_1)).

    Example
    -------
    >>> durbin_watson([1, -1, 1, -1])
    3.0
    """
    e = _as_vector(resid, "resid")
    d = np.diff(e)
    return float((d @ d) / (e @ e))


def ols(y, X, *, add_const, cov="classical", lags=None, small_sample=False, dist="t"):
    """Ordinary least squares with classical, White HC0/HC1 or Newey-West HAC inference.

    Parameters
    ----------
    y : array (n,) or Series
        Dependent variable (time-ordered if cov='HAC').
    X : array (n,) / (n, p), Series or DataFrame
        Regressors. If y and X are both pandas their indexes must be identical
        (raises otherwise; rows are never matched by position across different indexes).
    add_const : bool  (required)
        True prepends a column of ones as the FIRST column (params[0] = intercept);
        raises if X already has a constant column. False: X is used as given
        (pass False when X already contains the constant, or for a no-intercept model).
    cov : {'classical', 'HC0', 'HC1', 'HAC'}, default 'classical'
        Coefficient covariance, see cov_ols.
    lags : int >= 0 or {'nw1994', 'sw'}, required for 'HAC' only
        Bartlett truncation lag (strings use newey_west_lags with floor).
    small_sample : bool, default False  (HAC only)
        True multiplies the HAC covariance by n/(n-k) (statsmodels use_correction=True).
    dist : {'t', 'normal'}, default 't'
        Reference distribution for p-values: Student t(n-k) or N(0,1).
        statsmodels uses 'normal' for robust covariances (use_t=False).

    Returns
    -------
    dict
        params, bse, tvalues, pvalues : (k,) arrays; Series labelled
            ['const', *columns] when X is pandas (so unpacking order cannot go wrong).
        cov : (k, k) coefficient covariance (DataFrame when X is pandas).
        resid, fitted : (n,) (Series when y or X is pandas).
        r2, adj_r2 : R^2 (centered if the design has a constant, else uncentered).
        ssr : e'e;  sigma2 : e'e/(n-k);  dw : Durbin-Watson.
        n, k, df_resid (= n-k) : int;  lags : int used by HAC, else None.

    Formula
    -------
    b = lstsq(X, y); e = y - X b; sigma2 = e'e/(n-k); V = cov_ols(X, e, kind=cov);
    t = b/se; p = 2 sf(|t|).

    Example
    -------
    >>> res = ols(r_asset, factors_df, add_const=True, cov='HAC', lags=5)
    >>> alpha, t_alpha = res['params']['const'], res['tvalues']['const']
    """
    labels = _labels(X)
    index = _index(y, X)
    yv = _as_vector(y, "y")
    A = _as_matrix(X, "X")
    if A.shape[0] != yv.shape[0]:
        raise ValueError(f"y has {yv.shape[0]} rows but X has {A.shape[0]}")
    _require_finite(yv, "y")
    _require_finite(A, "X")
    if dist not in ("t", "normal"):
        raise ValueError("dist must be 't' or 'normal'")
    if add_const:
        if _const_columns(A).any():
            raise ValueError("X already contains a constant column; use add_const=False")
        Xd = np.column_stack([np.ones(A.shape[0]), A])
        names = ["const", *labels] if labels is not None else None
    else:
        Xd = A
        names = labels
    n, k = Xd.shape
    if n <= k:
        raise ValueError(f"need n > k (n={n}, k={k})")
    _xtx_inv(Xd)  # rank check before estimating
    kind = _norm_kind(cov)
    if kind != "HAC" and lags is not None:
        raise ValueError("lags is only used with cov='HAC'")
    L = _resolve_lags(lags, n) if kind == "HAC" else None

    Xs, d = _unit_columns(Xd)  # equilibrate so the SVD cutoff is unit-free
    beta = np.linalg.lstsq(Xs, yv, rcond=None)[0] / d
    fitted = Xd @ beta
    e = yv - fitted
    ssr = float(e @ e)
    df_resid = n - k
    V = cov_ols(Xd, e, kind=kind, lags=L, small_sample=small_sample)
    tt = t_test(beta, V, df=df_resid if dist == "t" else None)
    r2, adj = r_squared(yv, fitted, k=k, centered=bool(add_const) or _has_constant(Xd))

    out = {
        "params": beta, "bse": tt["se"], "tvalues": tt["t"], "pvalues": tt["pvalue"],
        "cov": V, "resid": e, "fitted": fitted,
    }
    if names is not None:
        for key in ("params", "bse", "tvalues", "pvalues"):
            out[key] = pd.Series(out[key], index=names)
        out["cov"] = pd.DataFrame(V, index=names, columns=names)
    if index is not None:
        out["resid"] = pd.Series(e, index=index, name="resid")
        out["fitted"] = pd.Series(fitted, index=index, name="fitted")
    out.update(r2=r2, adj_r2=adj, ssr=ssr, sigma2=ssr / df_resid, dw=durbin_watson(e),
               n=n, k=k, df_resid=df_resid, lags=L)
    return out


def grs_test(excess_returns, factors, *, variant="mle"):
    """Gibbons-Ross-Shanken test that the intercepts of N test assets on K traded factors are jointly zero.

    Parameters
    ----------
    excess_returns : array (T, N) or DataFrame
        Test-asset EXCESS returns (same periodicity as the factors).
    factors : array (T,) / (T, K) or DataFrame
        Traded-factor EXCESS returns (e.g. Mkt-RF, SMB, HML).
    variant : {'mle', 'sample_omega'}, default 'mle'
        'mle': Sigma = E'E/T and Omega = factor covariance with 1/T; the textbook
            GRS statistic, exactly F(N, T-N-K) under the null when residuals are
            i.i.d. multivariate normal (Cochrane, Asset Pricing, ch. 12; equal to
            GRS 1989's T/N (T-N-K)/(T-K-1) a' Sigma_u^{-1} a / (1 + fbar' Omega^{-1} fbar)
            with Sigma_u = E'E/(T-K-1)).
        'sample_omega': Omega divided by T-1 (np.cov default); Sigma unchanged
            (E'E/T). Not exactly F. (Code that uses Sigma = E'E/d, e.g. np.cov with
            d = T-1, gets stat_other = stat * d / T for the same Omega.)

    Returns
    -------
    dict with stat, pvalue, df1 (= N), df2 (= T-N-K), alphas (N,),
    resid_cov (Sigma = E'E/T, N x N), factor_cov (Omega, K x K).

    Formula
    -------
    Per asset: r_i = a_i + F b_i + e_i (OLS with intercept, same T rows for all).
    GRS = (T-N-K)/N * a' Sigma^{-1} a / (1 + fbar' Omega^{-1} fbar) ~ F(N, T-N-K).

    Example
    -------
    >>> g = grs_test(portfolios_excess, ff3_factors); g['stat'], g['pvalue']
    """
    _index(excess_returns, factors)
    asset_names = _labels(excess_returns)
    factor_names = _labels(factors)
    R = _as_matrix(excess_returns, "excess_returns")
    F = _as_matrix(factors, "factors")
    T, N = R.shape
    K = F.shape[1]
    if F.shape[0] != T:
        raise ValueError(f"excess_returns has {T} rows but factors has {F.shape[0]}")
    _require_finite(R, "excess_returns")
    _require_finite(F, "factors")
    if T - N - K < 1:
        raise ValueError(f"need T > N + K (T={T}, N={N}, K={K})")
    if variant not in ("mle", "sample_omega"):
        raise ValueError("variant must be 'mle' or 'sample_omega'")
    Z = np.column_stack([np.ones(T), F])
    B = np.linalg.lstsq(Z, R, rcond=None)[0]
    alphas = B[0]
    E = R - Z @ B
    sigma = E.T @ E / T
    fbar = F.mean(axis=0)
    fc = F - fbar
    omega = fc.T @ fc / (T if variant == "mle" else T - 1)
    quad_a = float(alphas @ np.linalg.solve(sigma, alphas))
    quad_f = float(fbar @ np.linalg.solve(omega, fbar))
    df1, df2 = N, T - N - K
    stat = df2 / N * quad_a / (1.0 + quad_f)
    pvalue = float(stats.f.sf(stat, df1, df2))
    if asset_names is not None:
        alphas = pd.Series(alphas, index=asset_names)
        sigma = pd.DataFrame(sigma, index=asset_names, columns=asset_names)
    if factor_names is not None:
        omega = pd.DataFrame(omega, index=factor_names, columns=factor_names)
    return {"stat": float(stat), "pvalue": pvalue, "df1": df1, "df2": df2,
            "alphas": alphas, "resid_cov": sigma, "factor_cov": omega}


def vif(X, *, names=None):
    """Variance inflation factor of each regressor.

    Parameters
    ----------
    X : array (n, p) or DataFrame
        Regressors WITHOUT the constant column. Each column j is regressed on the
        other columns PLUS an intercept (statsmodels variance_inflation_factor
        applied to add_constant(X)).
    names : list of str, optional
        Labels for an ndarray input (returns a Series); DataFrame input uses its columns.

    Returns
    -------
    ndarray (p,), or Series indexed by column name. 1 = orthogonal; > 10 is the usual alarm.

    Formula
    -------
    VIF_j = 1 / (1 - R_j^2),  R_j^2 = centered R^2 of x_j on [1, x_{-j}]
    (equivalently diag of the inverse correlation matrix).

    Example
    -------
    >>> vif(factors_df)
    """
    labels = names if names is not None else _labels(X)
    A = _as_matrix(X, "X")
    _require_finite(A, "X")
    n, p = A.shape
    if labels is not None and len(labels) != p:
        raise ValueError("names must have one entry per column")
    if (np.ptp(A, axis=0) == 0).any():
        raise ValueError("X has a constant column; pass the regressors WITHOUT the constant")
    if n <= p:
        raise ValueError(f"need n > p (n={n}, p={p})")
    Ac = A - A.mean(axis=0)
    Ac = Ac / np.sqrt(np.einsum("ij,ij->j", Ac, Ac))  # VIF is unit-free; equilibrate
    out = np.ones(p)
    if p > 1:
        for j in range(p):
            xj = Ac[:, j]
            others = np.delete(Ac, j, axis=1)
            b = np.linalg.lstsq(others, xj, rcond=None)[0]
            r = xj - others @ b
            r2 = 1.0 - (r @ r) / (xj @ xj)
            out[j] = np.inf if r2 >= 1.0 else 1.0 / (1.0 - r2)
    if labels is not None:
        return pd.Series(out, index=labels, name="VIF")
    return out


def rolling_ols(y, X, window, *, add_const, min_nobs=None):
    """Rolling-window OLS coefficients; row t uses observations t-window+1 .. t.

    Parameters
    ----------
    y : array (T,) or Series
    X : array (T,) / (T, p), Series or DataFrame
    window : int
        Number of rows in each window; the window ENDS at row t inclusive, so
        the first window-1 rows are NaN (no partial leading windows).
    add_const : bool  (required)
        True prepends the intercept as the FIRST column (raises if X already has a
        constant column); False uses X as given.
    min_nobs : int, optional
        Rows with NaN in y or X are dropped inside each window; a window is
        estimated only if it keeps >= min_nobs complete rows. Default None = k
        (number of coefficients; statsmodels RollingOLS default). With no NaNs
        every full window qualifies.

    Returns
    -------
    ndarray (T, k), or DataFrame (same index, columns ['const', *names]) when
    y or X is pandas. Rank-deficient windows (e.g. a regressor constant inside
    the window) are NaN.

    Formula
    -------
    beta_t = (X_w' X_w)^{-1} X_w' y_w over the window ending at t (normal
    equations on globally demeaned regressors when add_const=True, and with each
    window's X'X scaled to unit diagonal, so units such as volume in shares next
    to returns in decimals do not matter). Cost O(T window k^2).

    Example
    -------
    >>> betas = rolling_ols(r_asset, mkt_excess, 252, add_const=True)  # col 1 = rolling beta
    """
    labels = _labels(X)
    index = _index(y, X)
    yv = _as_vector(y, "y")
    A = _as_matrix(X, "X")
    T = yv.shape[0]
    if A.shape[0] != T:
        raise ValueError(f"y has {T} rows but X has {A.shape[0]}")
    valid = np.isfinite(yv) & np.isfinite(A).all(axis=1)
    if add_const and _const_columns(A[valid]).any():
        raise ValueError("X already contains a constant column; use add_const=False")
    k = A.shape[1] + (1 if add_const else 0)
    if isinstance(window, bool) or int(window) != window:
        raise ValueError("window must be an integer")
    window = int(window)
    if window < k or window > T:
        raise ValueError(f"need k <= window <= T (k={k}, window={window}, T={T})")
    min_nobs = k if min_nobs is None else int(min_nobs)
    if min_nobs < k or min_nobs > window:
        raise ValueError(f"need k <= min_nobs <= window (k={k}, min_nobs={min_nobs})")

    out = np.full((T, k), np.nan)
    if valid.any():
        if add_const:
            mx = A[valid].mean(axis=0)
            my = yv[valid].mean()
            Xw = np.column_stack([np.ones(T), A - mx])
            yw = yv - my
        else:
            mx = my = None
            Xw = A.copy()
            yw = yv.copy()
        Xw[~valid] = 0.0
        yw[~valid] = 0.0
        xv = sliding_window_view(Xw, window, axis=0)            # (m, k, window)
        xtx = np.matmul(xv, np.swapaxes(xv, 1, 2))              # (m, k, k)
        xty = np.matmul(xv, sliding_window_view(yw, window)[..., None])[..., 0]
        nobs = sliding_window_view(valid.astype(np.int64), window).sum(axis=1)
        # Jacobi-scale each window (unit diagonal) so the rank test is unit-free.
        dg = np.sqrt(np.einsum("mii->mi", xtx))                 # (m, k) column norms
        nonzero = (dg > 0).all(axis=1)
        dg = np.where(dg > 0, dg, 1.0)
        xtx = xtx / (dg[:, :, None] * dg[:, None, :])
        xty = xty / dg
        eig = np.linalg.eigvalsh(xtx)                           # ascending
        lam_max = eig[:, -1]
        ok = (nobs >= min_nobs) & nonzero & (eig[:, 0] > lam_max * max(window, k) * _EPS)
        if ok.any():
            beta = np.linalg.solve(xtx[ok], xty[ok][..., None])[..., 0] / dg[ok]
            if add_const:
                beta[:, 0] += my - beta[:, 1:] @ mx
            out[window - 1:][ok] = beta
    if index is not None or labels is not None:
        cols = labels if labels is not None else [f"x{i + 1}" for i in range(A.shape[1])]
        if add_const:
            cols = ["const", *cols]
        return pd.DataFrame(out, index=index, columns=cols)
    return out
