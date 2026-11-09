"""Eigenvalue (spectral) regime measures: the Kritzman-Page absorption ratio and the Marchenko-Pastur noise edge.

Conventions: a covariance or correlation matrix is symmetric and decomposed with numpy.linalg.eigvalsh; for a
correlation matrix the eigenvalues sum to the number of assets. The Marchenko-Pastur edges are those of a sample
covariance of n_obs observations of n_vars iid variables with variance sigma2 (sigma2 = 1 for a correlation
matrix).

Functions
---------
absorption_ratio       share of total variance in the k largest eigenvalues (Kritzman, Li, Page and Rigobon 2011)
marchenko_pastur_edge  random-matrix bulk edges sigma2 (1 -+ sqrt(n_vars/n_obs))^2 (Marchenko-Pastur 1967)

Ported from the team's qflib (``pca.absorption_ratio``, ``pca.marchenko_pastur_edge``), taking a matrix as well as
eigenvalues and the noise VARIANCE rather than its square root.

References: Kritzman, Li, Page and Rigobon (2011), Journal of Portfolio Management 37(4); Marchenko and Pastur
(1967), Mathematics of the USSR-Sbornik 1; Laloux, Cizeau, Bouchaud and Potters (1999), Physical Review Letters 83;
Bouchaud and Potters, Theory of Financial Risk and Derivative Pricing (2003) Ch. 2.
"""

from __future__ import annotations

import math
import numbers

import numpy as np

__all__ = [
    "absorption_ratio",
    "marchenko_pastur_edge",
]


def absorption_ratio(x, *, k):
    """Absorption ratio (Kritzman-Page): the share of total variance the k largest eigenvalues absorb,
    sum of the k largest eigenvalues / sum of all eigenvalues.

    Parameters
    ----------
    x : array_like. Either a 1-D array of eigenvalues (any order, >= 0), or a symmetric (N, N) covariance or
        correlation matrix (its eigenvalues by numpy.linalg.eigvalsh; for a correlation matrix the denominator
        is N). Pass a rolling window's matrix (e.g. ``returns[t-w:t].corr()``) once per window.
    k : REQUIRED. Number of leading eigenvalues (eigenvectors), 1 <= k <= N.

    Returns
    -------
    float in (0, 1].

    References
    ----------
    Kritzman, Li, Page and Rigobon (2011), eq. (1).

    Example
    -------
    >>> absorption_ratio(returns.iloc[-60:].corr().to_numpy(), k=2)
    >>> absorption_ratio([5.0, 2.0, 1.0, 0.5, 0.5], k=2)  # 7/9
    """
    a = np.asarray(x, dtype=float)
    if a.ndim == 2:
        if a.shape[0] != a.shape[1] or not np.allclose(a, a.T, rtol=1e-8, atol=1e-12):
            raise ValueError("a 2-D x must be a symmetric covariance or correlation matrix")
        a = np.linalg.eigvalsh(a)
    elif a.ndim != 1:
        raise ValueError("x must be 1-D eigenvalues or a 2-D symmetric matrix")
    if a.size == 0 or not np.all(np.isfinite(a)):
        raise ValueError("x must be non-empty and finite")
    if np.any(a < -1e-10 * max(1.0, float(np.max(np.abs(a))))):
        raise ValueError("eigenvalues must be >= 0 (the matrix must be positive semi-definite)")
    if isinstance(k, bool) or not isinstance(k, numbers.Integral) or not 1 <= k <= a.size:
        raise ValueError(f"k must be an int in [1, {a.size}]")
    ordered = np.sort(np.clip(a, 0.0, None))[::-1]
    total = float(ordered.sum())
    if total <= 0.0:
        raise ValueError("all eigenvalues are 0: the ratio is undefined")
    return float(ordered[:k].sum() / total)


def marchenko_pastur_edge(n_obs, n_vars, *, sigma2):
    """Marchenko-Pastur noise bounds: the eigenvalue range sigma2 (1 -+ sqrt(q))^2, q = n_vars/n_obs, of a sample
    covariance (correlation) matrix of pure noise; eigenvalues above lambda_max carry signal (Laloux et al. 1999).

    Parameters
    ----------
    n_obs : int >= 1. Observations T in the window (rows of the return matrix).
    n_vars : int >= 1. Variables N (assets, columns).
    sigma2 : REQUIRED. Noise variance (>= 0): 1 for a correlation matrix, the average asset variance for a
        covariance matrix.

    Returns
    -------
    dict
        lambda_min, lambda_max : the bulk edges; q : n_vars/n_obs.

    Formula
    -------
    lambda_+- = sigma2 (1 +- sqrt(q))^2 (for q > 1 the spectrum also has n_vars - n_obs zero eigenvalues).

    References
    ----------
    Marchenko and Pastur (1967); Laloux, Cizeau, Bouchaud and Potters (1999); Bouchaud and Potters (2003) Sec. 2.3.

    Example
    -------
    >>> mp = marchenko_pastur_edge(252, 50, sigma2=1.0); n_signal = (eig > mp['lambda_max']).sum()
    """
    for name, v in (("n_obs", n_obs), ("n_vars", n_vars)):
        if isinstance(v, bool) or not isinstance(v, numbers.Integral) or v < 1:
            raise ValueError(f"{name} must be an int >= 1")
    if isinstance(sigma2, bool) or not isinstance(sigma2, numbers.Real) or not float(sigma2) >= 0.0:
        raise ValueError("sigma2 must be a variance >= 0 (1 for a correlation matrix)")
    q = n_vars / n_obs
    root = math.sqrt(q)
    s2 = float(sigma2)
    return {"lambda_min": s2 * (1.0 - root) ** 2, "lambda_max": s2 * (1.0 + root) ** 2, "q": q}
