"""Copulas: Gaussian, Student-t and the Archimedean Clayton, Gumbel and Frank families.

Conventions for the whole module: ``u`` holds points of the unit cube, one
point per row, shape (n, d) (a single point may be a 1-D array of length d);
every coordinate must lie strictly inside (0, 1). ``family`` is ONE string:
'gaussian' (alias 'normal'), 't' (aliases 'student', 'student_t', 'student-t'),
'clayton', 'gumbel' or 'frank' (case-insensitive). ``theta`` is the family
parameter: for 'gaussian' and 't' it is the correlation rho, either a scalar
(an equicorrelation matrix in d dimensions) or a full d x d correlation matrix;
't' also needs the degrees of freedom ``nu`` (> 0, need not be an integer).
Archimedean parameter ranges (Nelsen 2006, Table 4.1):
Clayton theta >= -1 (theta < 0 only in two dimensions; theta = 0 is the
independence limit), Gumbel theta >= 1 (1 = independence), Frank theta real
(theta < 0 only in two dimensions; 0 = independence).
Archimedean copulas are written with their generator inverse psi
(C(u) = psi(sum_i psi^{-1}(u_i))): Clayton psi(t) = (1 + t)^{-1/theta},
Gumbel psi(t) = exp(-t^{1/theta}), Frank psi(t) = -log(1 - (1 - e^{-theta}) e^{-t})/theta.
Results are numpy float64 (a ``float`` subclass) for scalar results and
``np.ndarray`` otherwise.

Functions
---------
copula_tau              Kendall's tau from the parameter (Frank through the Debye function D_1)
copula_theta            the parameter from Kendall's tau (inverse of copula_tau)
copula_sample           random draws: elliptical by Cholesky, Archimedean by Marshall-Olkin frailty
copula_logpdf           log copula density (Archimedean: bivariate)
copula_cdf              copula C(u) (elliptical: bivariate)
copula_tail_dependence  lower and upper tail-dependence coefficients

References: Nelsen, An Introduction to Copulas, 2nd ed. (2006), Ch. 4-5;
Joe, Dependence Modeling with Copulas (2014), Ch. 4; McNeil, Frey and
Embrechts, Quantitative Risk Management (2005), Ch. 5; Marshall and Olkin
(1988); Embrechts, McNeil and Straumann (2002); Hofert (2008).
"""

from __future__ import annotations

import math
import numbers
import warnings

import numpy as np
from scipy import integrate, optimize
from scipy.special import gammaln, ndtr, ndtri, spence, stdtr, stdtrit

from .numerics import _bvn_cdf

__all__ = [
    "copula_tau",
    "copula_theta",
    "copula_sample",
    "copula_logpdf",
    "copula_cdf",
    "copula_tail_dependence",
]

_ALIASES = {
    "gaussian": "gaussian", "normal": "gaussian",
    "t": "t", "student": "t", "student_t": "t", "student-t": "t", "studentt": "t",
    "clayton": "clayton", "gumbel": "gumbel", "frank": "frank",
}
_ELLIPTICAL = ("gaussian", "t")
_TINY = np.finfo(float).tiny
_ONE_MINUS = np.nextafter(1.0, 0.0)
_PI2_6 = math.pi ** 2 / 6.0
# Frank tau = sum_k c_k theta^(2k-1), c_k = 4 B_2k / ((2k+1)(2k)!) (Debye series, Abramowitz-Stegun 27.1.1),
# B_2k the Bernoulli numbers; 12 terms leave a truncation error below 1e-16 for |theta| < 1.
_FRANK_SERIES = tuple(4.0 * float(b) / ((2 * k + 1) * math.factorial(2 * k)) for k, b in enumerate(
    (1 / 6, -1 / 30, 1 / 42, -1 / 30, 5 / 66, -691 / 2730, 7 / 6, -3617 / 510, 43867 / 798, -174611 / 330,
     854513 / 138, -236364091 / 2730), start=1))


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _out(x):
    """0-d result -> numpy float64 scalar; otherwise the ndarray itself."""
    x = np.asarray(x, dtype=float)
    return x[()] if x.ndim == 0 else x


def _family(family):
    if not isinstance(family, str):
        raise TypeError("family must be one of 'gaussian', 't', 'clayton', 'gumbel', 'frank'")
    key = family.strip().lower()
    if key not in _ALIASES:
        raise ValueError(f"family must be one of 'gaussian', 't', 'clayton', 'gumbel', 'frank', got {family!r}")
    return _ALIASES[key]


def _nu(nu, fam):
    if fam != "t":
        if nu is not None:
            raise ValueError("nu applies only to family 't'")
        return None
    if nu is None:
        raise ValueError("family 't' needs nu (degrees of freedom > 0)")
    nu = float(nu)
    if not (math.isfinite(nu) and nu > 0):
        raise ValueError("nu must be finite and > 0")
    return nu


def _int_arg(value, name, minimum):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be an integer, got {value!r}")
    v = int(value)
    if v < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {v}")
    return v


def _check_archimedean(fam, th, d):
    """Validate an Archimedean parameter (scalar or array) for dimension d."""
    if not np.all(np.isfinite(th)):
        raise ValueError("theta must be finite")
    if fam == "clayton":
        if np.any(th < -1.0):
            raise ValueError("Clayton theta must be >= -1")
        if d > 2 and np.any(th < 0.0):
            raise ValueError("Clayton theta < 0 is a copula only in two dimensions")
    elif fam == "gumbel":
        if np.any(th < 1.0):
            raise ValueError("Gumbel theta must be >= 1")
    elif fam == "frank":
        if d > 2 and np.any(th < 0.0):
            raise ValueError("Frank theta < 0 is a copula only in two dimensions")


def _check_rho(rho):
    if not np.all(np.isfinite(rho)) or np.any(np.abs(rho) > 1.0):
        raise ValueError("rho must lie in [-1, 1]")


def _corr(theta, d):
    """Correlation matrix (d x d) and its Cholesky factor from a scalar rho or a matrix."""
    th = np.asarray(theta, dtype=float)
    if th.ndim == 0:
        rho = float(th)
        R = np.full((d, d), rho)
        np.fill_diagonal(R, 1.0)
    elif th.ndim == 2 and th.shape[0] == th.shape[1]:
        if th.shape[0] != d:
            raise ValueError(f"correlation matrix is {th.shape[0]} x {th.shape[0]} but the points have {d} columns")
        R = th.copy()
        if not np.allclose(R, R.T, rtol=0, atol=1e-12) or not np.allclose(np.diag(R), 1.0, rtol=0, atol=1e-12):
            raise ValueError("theta must be a symmetric correlation matrix with unit diagonal")
    else:
        raise ValueError("for 'gaussian' and 't', theta is a scalar rho or a square correlation matrix")
    try:
        L = np.linalg.cholesky(R)
    except np.linalg.LinAlgError:
        raise ValueError("correlation matrix is not positive definite") from None
    return R, L


def _points(u):
    u = np.asarray(u, dtype=float)
    if u.ndim == 0:
        raise ValueError("u must hold points of the unit cube, shape (n, d) or (d,)")
    if u.shape[-1] < 2:
        raise ValueError("a copula needs at least 2 coordinates (last axis of u)")
    if not np.all((u > 0.0) & (u < 1.0)):
        raise ValueError("every coordinate of u must lie strictly inside (0, 1)")
    return u


def _scalar_theta(theta, fam):
    th = np.asarray(theta, dtype=float)
    if th.ndim != 0:
        raise ValueError(f"{fam} theta must be a scalar")
    return float(th)


def _log_abs_expm1(x):
    """log|e^x - 1| without overflow or cancellation (x != 0)."""
    x = np.asarray(x, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        pos = x + np.log(-np.expm1(-np.abs(x)))
        neg = np.log(-np.expm1(np.minimum(x, 0.0)))
    return np.where(x > 0, pos, neg)


def _frank_log_d(theta, u, v):
    """log|D| with D = (1 - e^-th) - (1 - e^-th u)(1 - e^-th v), the Frank denominator.

    D = -e^{-th u} expm1(-th v) - e^{-th v} expm1(-th (1 - v)): both terms share the
    sign of theta, so the sum has no cancellation.
    """
    a = -theta * u + _log_abs_expm1(-theta * v)
    b = -theta * v + _log_abs_expm1(-theta * (1.0 - v))
    return np.logaddexp(a, b)


# ----------------------------------------------------------------------------
# Kendall's tau <-> parameter
# ----------------------------------------------------------------------------
def _frank_tau_pos(x):
    """Frank tau for x > 0: 1 - 4/x (1 - D_1(x)), D_1 the Debye function."""
    x = np.asarray(x, dtype=float)
    small = x < 1.0
    xs = np.where(small, x, 0.0)
    # tau = x/9 - x^3/900 + x^5/52920 - ... (Bernoulli series of D_1); the dilogarithm form below
    # loses about log10(4/x^2) digits to cancellation, so the series is used for x < 1
    series = np.zeros_like(xs)
    for c in reversed(_FRANK_SERIES):
        series = series * xs * xs + c
    series = series * xs
    xl = np.where(small, 1.0, x)
    with np.errstate(over="ignore", under="ignore"):
        e = np.exp(-xl)
        # int_0^x t/(e^t - 1) dt = pi^2/6 + x log(1 - e^-x) - Li2(e^-x); Li2(z) = spence(1 - z)
        integral = _PI2_6 + xl * np.log1p(-e) - spence(-np.expm1(-xl))
    big = 1.0 + 4.0 * (integral / xl - 1.0) / xl
    return np.where(small, series, big)


def copula_tau(theta, *, family):
    """Kendall's tau of a Gaussian, Student-t, Clayton, Gumbel or Frank copula from its parameter.

    Parameters
    ----------
    theta : float or array_like
        The family parameter (rho for 'gaussian' and 't'; evaluated elementwise,
        so a correlation matrix gives the matrix of pairwise taus).
    family : str
        REQUIRED. 'gaussian', 't', 'clayton', 'gumbel' or 'frank'.

    Returns
    -------
    Kendall's tau (float or ndarray, elementwise).

    Formula
    -------
    gaussian, t: tau = (2/pi) arcsin(rho)  (any nu)
    clayton:     tau = theta / (theta + 2)
    gumbel:      tau = 1 - 1/theta
    frank:       tau = 1 - (4/theta) [1 - D_1(theta)],  D_1(x) = (1/x) int_0^x t/(e^t - 1) dt
                 (odd in theta; 0 at theta = 0)

    References
    ----------
    Nelsen (2006) Ch. 5 (Kendall's tau of Archimedean copulas); Genest (1987) (Frank);
    Lindskog, McNeil and Schmock (2003) (elliptical); McNeil-Frey-Embrechts (2005) Ch. 5.

    Example
    -------
    >>> copula_tau(2.0, family='clayton')
    0.5
    >>> copula_tau(5.736282707, family='frank')
    0.5000...
    """
    fam = _family(family)
    th = np.asarray(theta, dtype=float)
    if fam in _ELLIPTICAL:
        _check_rho(th)
        return _out(2.0 / math.pi * np.arcsin(th))
    _check_archimedean(fam, th, 2)
    if fam == "clayton":
        return _out(th / (th + 2.0))
    if fam == "gumbel":
        return _out(1.0 - 1.0 / th)
    a = np.abs(th)
    tau = np.where(a == 0.0, 0.0, _frank_tau_pos(np.where(a == 0.0, 1.0, a)))
    return _out(np.sign(th) * tau)


def copula_theta(tau, *, family):
    """Parameter of a Gaussian, Student-t, Clayton, Gumbel or Frank copula from Kendall's tau.

    Parameters
    ----------
    tau : float or array_like
        Kendall's tau in (-1, 1) (elementwise).
    family : str
        REQUIRED. 'gaussian', 't', 'clayton', 'gumbel' or 'frank'.

    Returns
    -------
    The parameter (rho for 'gaussian' and 't'), the exact inverse of copula_tau.
    Gumbel cannot represent negative dependence: tau < 0 gives the independence edge
    theta = 1 with a RuntimeWarning (an infeasible family for that sample; copula_tau(1) = 0).
    Clayton tau < 0 gives theta in [-1, 0), a copula in two dimensions only.
    tau = 0 gives the independence value (0 for clayton, frank, rho; 1 for gumbel).

    Formula
    -------
    gaussian, t: rho = sin(pi tau / 2)
    clayton:     theta = 2 tau / (1 - tau)
    gumbel:      theta = 1 / (1 - tau)
    frank:       root of 1 - (4/theta)[1 - D_1(theta)] = tau (Brent, |error| < 1e-12 theta);
                 theta(-tau) = -theta(tau)

    References
    ----------
    Nelsen (2006) Ch. 5; Genest (1987); Genest and Rivest (1993)
    (method-of-moments estimation by inversion of Kendall's tau).

    Example
    -------
    >>> copula_theta(0.5, family='gumbel')
    2.0
    >>> copula_theta(0.5, family='frank')
    5.7362...
    """
    fam = _family(family)
    t = np.asarray(tau, dtype=float)
    lowest = -1.0 if fam == "clayton" else np.nextafter(-1.0, 0.0)  # Clayton tau = -1 is theta = -1
    if not np.all(np.isfinite(t)) or np.any(t < lowest) or np.any(t >= 1.0):
        raise ValueError("tau must lie in (-1, 1)")
    if fam in _ELLIPTICAL:
        return _out(np.sin(0.5 * math.pi * t))
    if fam == "clayton":
        return _out(2.0 * t / (1.0 - t))
    if fam == "gumbel":
        if np.any(t < 0.0):  # no Gumbel parameter: the edge theta = 1, so a loop over families goes on
            try:  # a note for the run log; never an exception, even under warnings-as-errors
                warnings.warn("copula_theta: a Gumbel copula cannot represent tau < 0; returning the independence "
                              "edge theta = 1 (an infeasible family for this sample)", RuntimeWarning, stacklevel=2)
            except Warning:
                pass
        return _out(np.where(t < 0.0, 1.0, 1.0 / (1.0 - t)))
    flat = np.abs(t).ravel()
    res = np.empty_like(flat)
    for i, target in enumerate(flat):
        if target == 0.0:
            res[i] = 0.0
            continue
        if target < 1e-3:  # tau = x/9 - x^3/900 + ...: Newton from 9 tau converges in a few steps
            x = 9.0 * target
            for _ in range(8):
                x -= (float(_frank_tau_pos(x)) - target) / (1.0 / 9.0 - x * x / 300.0 + x ** 4 / 10584.0)
            res[i] = x
            continue
        hi = 10.0
        while float(_frank_tau_pos(hi)) < target:
            hi *= 2.0
            if hi > 1e8:
                raise ValueError("tau too close to 1 for a Frank parameter")
        res[i] = optimize.brentq(lambda x: float(_frank_tau_pos(x)) - target, 1e-8, hi,
                                 xtol=1e-14, rtol=4 * np.finfo(float).eps, maxiter=500)
    return _out(np.sign(t) * res.reshape(t.shape))


# ----------------------------------------------------------------------------
# sampling
# ----------------------------------------------------------------------------
def _positive_stable(rng, alpha, n):
    """Positive stable variates with Laplace transform exp(-s^alpha), 0 < alpha <= 1 (Kanter 1975)."""
    if alpha == 1.0:
        return np.ones(n)
    th = math.pi * (rng.random(n) + 2.0 ** -54)       # uniform on the open interval (0, pi)
    w = rng.standard_exponential(n)
    return (np.sin(alpha * th) / np.sin(th) ** (1.0 / alpha)
            * (np.sin((1.0 - alpha) * th) / w) ** ((1.0 - alpha) / alpha))


def _log_series(rng, theta, n):
    """Logarithmic-series variates P(V = k) = p^k / (-k log(1 - p)), p = 1 - e^{-theta} (Kemp 1981, LK).

    Works from log(1 - p) = -theta and log q = log(1 - e^{-theta U}), so it stays exact
    when p or q round to 1 in floating point (large theta): numpy's logseries would
    need p < 1 and loses the large values of V there.
    """
    p = -math.expm1(-theta)
    v_u = rng.random(n)
    u1 = rng.random(n)
    q = -np.expm1(-theta * u1)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        log_q = np.log1p(-np.exp(-theta * u1))
        k = np.floor(1.0 + np.log(np.maximum(v_u, _TINY)) / log_q)
    k = np.where(np.isnan(k) | (k < 1.0), 1.0, k)       # u1 = 0 (q = 0): V = 1
    out = np.where(v_u < q, 2.0, 1.0)                  # q^2 < V < q: 2;  V >= q: 1
    out = np.where(v_u <= q * q, k, out)               # V <= q^2: floor(1 + log V / log q)
    out = np.where(v_u >= p, 1.0, out)
    return out


def _conditional_bivariate(fam, theta, u, w):
    """Second coordinate by inverting the conditional cdf dC(u, v)/du = w (Clayton, Frank)."""
    if fam == "clayton":
        if theta == -1.0:
            return 1.0 - u
        # v = (u^-theta (w^{-theta/(1+theta)} - 1) + 1)^{-1/theta}
        s = u ** (-theta) * np.expm1(-theta / (1.0 + theta) * np.log(w)) + 1.0
        return s ** (-1.0 / theta)
    # Frank: v = -(1/theta) log(1 + w (e^-theta - 1) / (w + (1 - w) e^{-theta u}))
    return -np.log1p(w * np.expm1(-theta) / (w + (1.0 - w) * np.exp(-theta * u))) / theta


def copula_sample(n, theta, *, family, seed, dim=2, nu=None, method=None):
    """Random draws from a Gaussian, Student-t, Clayton, Gumbel or Frank copula (uniform margins). API contract: nu required for family='t', otherwise omit or None.

    Parameters
    ----------
    n : int
        Number of draws (rows).
    theta : float or array_like
        Family parameter. 'gaussian' / 't': scalar rho (equicorrelation in dim
        dimensions) or a d x d correlation matrix (then dim is its size).
        Archimedean: a scalar (exchangeable in dim dimensions).
    family : str
        REQUIRED. 'gaussian', 't', 'clayton', 'gumbel' or 'frank'.
    seed : int, None or numpy Generator
        REQUIRED. Passed to numpy.random.default_rng.
    dim : int
        Dimension d >= 2 (default 2).
    nu : float
        Degrees of freedom, required for 't' only.
    method : None, 'frailty' or 'conditional'
        Archimedean algorithm. None picks 'frailty' (Marshall-Olkin) when theta > 0
        and 'conditional' (bivariate conditional inversion) when theta < 0.
        'conditional' is available for Clayton and Frank in two dimensions only.

    Returns
    -------
    ndarray of shape (n, d) with every coordinate in the open interval (0, 1)
    (a draw that rounds to 0 or 1 in floating point is moved to the nearest
    representable interior value).

    Formula
    -------
    gaussian: U = Phi(L Z), L L' = R, Z ~ N(0, I).
    t:        U = t_nu(L Z / sqrt(W / nu)), one W ~ chi2(nu) per draw shared by all coordinates.
    Archimedean, Marshall-Olkin (1988): V ~ frailty with Laplace transform psi,
    E_i ~ Exp(1) independent, U_i = psi(E_i / V):
      clayton V ~ Gamma(1/theta, 1);  gumbel V positive stable with E e^{-sV} = e^{-s^{1/theta}}
      (Kanter 1975 / Chambers-Mallows-Stuck); frank V logarithmic series with p = 1 - e^{-theta}.
    Conditional inversion (d = 2): U ~ U(0,1), W ~ U(0,1), V solves dC(U, V)/dU = W:
      clayton V = (U^-theta (W^{-theta/(1+theta)} - 1) + 1)^{-1/theta};
      frank   V = -log(1 + W (e^-theta - 1) / (W + (1 - W) e^{-theta U})) / theta.
    theta = 0 (Clayton, Frank) and theta = 1 (Gumbel) give independent uniforms.

    References
    ----------
    Marshall and Olkin (1988), JASA 83; Hofert (2008), CSDA 52; McNeil (2008);
    Nelsen (2006) Ch. 2 and 4; McNeil-Frey-Embrechts (2005) Ch. 5 (simulation of Gaussian, t and
    exchangeable Archimedean copulas); Kanter (1975); Kemp (1981) (logarithmic series).

    Example
    -------
    >>> u = copula_sample(100000, 2.0, family='gumbel', seed=1)
    >>> scipy.stats.kendalltau(u[:, 0], u[:, 1])[0]      # theory 1 - 1/2
    0.50...
    """
    fam = _family(family)
    nu = _nu(nu, fam)
    n = _int_arg(n, "n", 1)
    dim = _int_arg(dim, "dim", 2)
    rng = np.random.default_rng(seed)
    if fam in _ELLIPTICAL:
        if method is not None:
            raise ValueError("method applies to the Archimedean families only")
        th = np.asarray(theta, dtype=float)
        if th.ndim == 2:
            if dim not in (2, th.shape[0]):
                raise ValueError("dim disagrees with the size of the correlation matrix")
            dim = th.shape[0]
        _, L = _corr(th, dim)
        x = rng.standard_normal((n, dim)) @ L.T
        if fam == "gaussian":
            u = ndtr(x)
        else:
            w = rng.chisquare(nu, n)
            u = stdtr(nu, x / np.sqrt(w / nu)[:, None])
        return np.clip(u, _TINY, _ONE_MINUS)
    th = _scalar_theta(theta, fam)
    _check_archimedean(fam, th, dim)
    independent = (fam == "gumbel" and th == 1.0) or (fam != "gumbel" and th == 0.0)
    if method is None:
        method = "conditional" if th < 0 else "frailty"
    if method not in ("frailty", "conditional"):
        raise ValueError("method must be None, 'frailty' or 'conditional'")
    if independent:
        return np.clip(rng.random((n, dim)), _TINY, _ONE_MINUS)
    if method == "conditional":
        if fam == "gumbel" or dim != 2:
            raise ValueError("'conditional' sampling is available for Clayton and Frank in two dimensions")
        u = rng.random(n)
        w = rng.random(n)
        u = np.clip(u, _TINY, _ONE_MINUS)
        w = np.clip(w, _TINY, _ONE_MINUS)
        v = _conditional_bivariate(fam, th, u, w)
        return np.clip(np.column_stack([u, v]), _TINY, _ONE_MINUS)
    if th < 0:
        raise ValueError("'frailty' sampling needs theta > 0")
    if fam == "clayton":
        v = rng.gamma(1.0 / th, 1.0, n)
    elif fam == "gumbel":
        v = _positive_stable(rng, 1.0 / th, n)
    else:
        v = _log_series(rng, th, n)
    e = rng.standard_exponential((n, dim))
    t = e / v[:, None]
    with np.errstate(over="ignore", under="ignore", divide="ignore"):
        if fam == "clayton":
            u = np.exp(-np.log1p(t) / th)
        elif fam == "gumbel":
            u = np.exp(-t ** (1.0 / th))
        else:
            # psi(t) = -log(1 - (1 - e^-th) e^-t)/th, with 1 - (1 - e^-th) e^-t = (1 - e^-t) + e^{-th-t}
            u = -np.logaddexp(np.log(-np.expm1(-t)), -th - t) / th
    return np.clip(u, _TINY, _ONE_MINUS)


# ----------------------------------------------------------------------------
# density and distribution function
# ----------------------------------------------------------------------------
def _elliptical_logpdf(fam, u, theta, nu):
    d = u.shape[-1]
    R, L = _corr(theta, d)
    logdet = 2.0 * np.sum(np.log(np.diag(L)))
    x = ndtri(u) if fam == "gaussian" else stdtrit(nu, u)
    flat = x.reshape(-1, d)
    y = np.linalg.solve(L, flat.T)              # L y = x  =>  x' R^-1 x = |y|^2
    quad = np.sum(y * y, axis=0).reshape(x.shape[:-1])
    if fam == "gaussian":
        return -0.5 * logdet - 0.5 * (quad - np.sum(x * x, axis=-1))
    const = gammaln(0.5 * (nu + d)) + (d - 1) * gammaln(0.5 * nu) - d * gammaln(0.5 * (nu + 1.0))
    return (const - 0.5 * logdet - 0.5 * (nu + d) * np.log1p(quad / nu)
            + 0.5 * (nu + 1.0) * np.sum(np.log1p(x * x / nu), axis=-1))


def copula_logpdf(u, theta, *, family, nu=None):
    """Log-density of a Gaussian, Student-t, Clayton, Gumbel or Frank copula at points of the unit cube. API contract: nu required for family='t', otherwise omit or None.

    Parameters
    ----------
    u : array_like, shape (n, d) or (d,)
        Points with coordinates strictly inside (0, 1) (e.g. pseudo-observations
        rank / (n + 1)). Archimedean families: d = 2.
    theta : float or array_like
        Family parameter ('gaussian' / 't': scalar rho or d x d correlation
        matrix; Archimedean: scalar).
    family : str
        REQUIRED. 'gaussian', 't', 'clayton', 'gumbel' or 'frank'.
    nu : float
        Degrees of freedom, required for 't' only.

    Returns
    -------
    log c(u), shape u.shape[:-1]. Summing gives the copula log-likelihood (the
    marginal densities are NOT included). -inf outside the support (Clayton
    theta < 0 where u^-theta + v^-theta - 1 <= 0).

    Formula
    -------
    gaussian: log c = -1/2 log|R| - 1/2 z'(R^-1 - I) z,  z = Phi^-1(u).
    t:        log c = log G((nu+d)/2) + (d-1) log G(nu/2) - d log G((nu+1)/2) - 1/2 log|R|
                      - (nu+d)/2 log(1 + x'R^-1 x/nu) + (nu+1)/2 sum_i log(1 + x_i^2/nu),
              x = t_nu^-1(u), G the gamma function.
    clayton:  c = (1+theta) (uv)^{-1-theta} (u^-theta + v^-theta - 1)^{-2-1/theta}  (theta > -1).
    gumbel:   x = -log u, y = -log v, s = x^theta + y^theta, A = s^{1/theta}:
              c = e^{-A} (uv)^-1 (xy)^{theta-1} s^{1/theta - 2} (A + theta - 1).
    frank:    c = theta (1 - e^-theta) e^{-theta(u+v)} / [(1 - e^-theta) - (1 - e^{-theta u})(1 - e^{-theta v})]^2.
    theta = 0 (Clayton, Frank) and theta = 1 (Gumbel) give log c = 0.

    References
    ----------
    Nelsen (2006) Ch. 4; Joe (2014) Ch. 4; McNeil-Frey-Embrechts (2005) Ch. 5
    (Gaussian and t densities); Demarta and McNeil (2005) (t copula).

    Example
    -------
    >>> copula_logpdf([[0.3, 0.4], [0.9, 0.8]], 0.5, family='t', nu=4)
    array([...])
    """
    fam = _family(family)
    nu = _nu(nu, fam)
    u = _points(u)
    if fam in _ELLIPTICAL:
        return _out(_elliptical_logpdf(fam, u, theta, nu))
    if u.shape[-1] != 2:
        raise ValueError("the Archimedean densities are bivariate: u must have 2 columns")
    th = _scalar_theta(theta, fam)
    _check_archimedean(fam, th, 2)
    a, b = u[..., 0], u[..., 1]
    la, lb = np.log(a), np.log(b)
    if (fam == "gumbel" and th == 1.0) or (fam != "gumbel" and th == 0.0):
        return _out(np.zeros(a.shape))
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        if fam == "clayton":
            if th == -1.0:
                raise ValueError("Clayton theta = -1 is the countermonotonic copula, which has no density")
            pa, pb = -th * la, -th * lb               # log u^-theta, log v^-theta
            if th > 0:
                m = np.maximum(pa, pb)
                log_s = m + np.log(np.exp(pa - m) + np.exp(pb - m) - np.exp(-m))
            else:
                s = np.exp(pa) + np.exp(pb) - 1.0
                log_s = np.where(s > 0, np.log(np.where(s > 0, s, 1.0)), -np.inf)
            val = math.log1p(th) - (1.0 + th) * (la + lb) - (2.0 + 1.0 / th) * log_s
            if th < 0:
                val = np.where(np.isneginf(log_s), -np.inf, val)
            return _out(val)
        if fam == "gumbel":
            x, y = -la, -lb
            lx, ly = np.log(x), np.log(y)
            log_s = np.logaddexp(th * lx, th * ly)
            A = np.exp(log_s / th)
            return _out(-A + x + y + (th - 1.0) * (lx + ly) + (1.0 / th - 2.0) * log_s + np.log(A + th - 1.0))
        # Frank
        log_num = math.log(abs(th)) + float(_log_abs_expm1(-th))
        return _out(log_num - th * (a + b) - 2.0 * _frank_log_d(th, a, b))


def _t_copula_cdf(u1, u2, rho, nu):
    """Bivariate t copula cdf by its conditional form, accurate in relative terms deep in the tails.

    With m = min(u1, u2), M = max(u1, u2), x_m, x_M their t_nu quantiles and
    h(x) = P(X2 <= x_M | X1 = x) = t_{nu+1}((x_M - rho x) / sqrt((1 - rho^2)(nu + x^2)/(nu + 1))):
    C = int_{-inf}^{x_m} f_nu(x) h(x) dx. The map x = x_m + (1 - 1/y) s, s = max(1, |x_m|), y in (0, 1],
    and the division by m make every point's integrand O(1) on one common interval, so a single
    vector-valued adaptive Gauss-Kronrod rule serves all points.
    """
    u1 = np.ravel(u1)
    u2 = np.ravel(u2)
    m = np.minimum(u1, u2)
    big = np.maximum(u1, u2)
    xm = stdtrit(nu, m)
    x_big = stdtrit(nu, big)
    scale = np.maximum(1.0, np.abs(xm))
    c = math.sqrt((1.0 - rho * rho) / (nu + 1.0))
    log_k = gammaln(0.5 * (nu + 1.0)) - gammaln(0.5 * nu) - 0.5 * math.log(nu * math.pi)

    def g(y):
        if y <= 0.0:
            return np.zeros_like(m)
        x = xm + (1.0 - 1.0 / y) * scale
        log_f = log_k - 0.5 * (nu + 1.0) * np.log1p(x * x / nu)
        h = stdtr(nu + 1.0, (x_big - rho * x) / (c * np.sqrt(nu + x * x)))
        return np.exp(log_f) * h * scale / (y * y) / m

    val, _ = integrate.quad_vec(g, 0.0, 1.0, epsabs=1e-13, epsrel=1e-11, norm="max", limit=4000)
    return m * val


def copula_cdf(u, theta, *, family, nu=None):
    """Distribution function C(u) of a Gaussian, Student-t, Clayton, Gumbel or Frank copula. API contract: nu required for family='t', otherwise omit or None.

    Parameters
    ----------
    u : array_like, shape (n, d) or (d,)
        Points with coordinates strictly inside (0, 1). 'gaussian' and 't':
        d = 2. Archimedean: any d (Clayton and Frank need theta > 0 when d > 2).
    theta : float or array_like
        Family parameter ('gaussian' / 't': scalar rho or 2 x 2 correlation
        matrix; Archimedean: scalar).
    family : str
        REQUIRED. 'gaussian', 't', 'clayton', 'gumbel' or 'frank'.
    nu : float
        Degrees of freedom, required for 't' only.

    Returns
    -------
    C(u) = P(U_1 <= u_1, ..., U_d <= u_d), shape u.shape[:-1].

    Formula
    -------
    gaussian: Phi_2(Phi^-1(u), Phi^-1(v); rho) (Owen's T function, exact to rounding).
    t:        t_{2,nu}(x, y; rho), x = t_nu^-1(u), y = t_nu^-1(v), in its conditional form
              int_{-inf}^{x} f_nu(s) t_{nu+1}((y - rho s) / sqrt((1 - rho^2)(nu + s^2)/(nu + 1))) ds
              (u <= v; adaptive quadrature, error about 1e-11 relative to min(u, v), so
              deep-tail values keep their digits).
    clayton:  (sum_i u_i^-theta - d + 1)^{-1/theta}  (d = 2, theta < 0: max(., 0)^{-1/theta}).
    gumbel:   exp(-(sum_i (-log u_i)^theta)^{1/theta}).
    frank:    -(1/theta) log(1 + prod_i (e^{-theta u_i} - 1) / (e^-theta - 1)^{d-1}).
    theta = 0 (Clayton, Frank) and theta = 1 (Gumbel) give prod_i u_i.

    References
    ----------
    Nelsen (2006) Ch. 4; McNeil-Frey-Embrechts (2005) Ch. 5; Owen (1956);
    Genz (2004) (bivariate normal and t probabilities); Li (2000) (Gaussian copula
    joint default probability).

    Example
    -------
    >>> copula_cdf([0.5, 0.5], 0.5, family='gaussian')    # Sheppard: 1/4 + asin(rho)/(2 pi)
    0.3333...
    """
    fam = _family(family)
    nu = _nu(nu, fam)
    u = _points(u)
    d = u.shape[-1]
    if fam in _ELLIPTICAL:
        if d != 2:
            raise ValueError("the elliptical copula cdf here is bivariate; use scipy.stats.multivariate_normal "
                             "or multivariate_t for d > 2")
        R, _ = _corr(theta, 2)
        rho = float(R[0, 1])
        a, b = u[..., 0], u[..., 1]
        if fam == "gaussian":
            return _out(_bvn_cdf(ndtri(a), ndtri(b), rho))
        val = _t_copula_cdf(a, b, rho, nu)
        return _out(np.clip(val, 0.0, 1.0).reshape(a.shape))
    th = _scalar_theta(theta, fam)
    _check_archimedean(fam, th, d)
    lu = np.log(u)
    if (fam == "gumbel" and th == 1.0) or (fam != "gumbel" and th == 0.0):
        return _out(np.exp(np.sum(lu, axis=-1)))
    with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
        if fam == "clayton":
            p = -th * lu                                # log u_i^-theta
            if th > 0:
                m = np.max(p, axis=-1)
                log_s = m + np.log(np.sum(np.exp(p - m[..., None]), axis=-1) - (d - 1) * np.exp(-m))
                return _out(np.exp(-log_s / th))
            s = np.sum(np.exp(p), axis=-1) - 1.0         # d = 2, theta in [-1, 0)
            return _out(np.where(s > 0, np.maximum(s, 0.0) ** (-1.0 / th), 0.0))
        if fam == "gumbel":
            x = -lu
            log_s = np.logaddexp.reduce(th * np.log(x), axis=-1)
            return _out(np.exp(-np.exp(log_s / th)))
        if d == 2:
            a, b = u[..., 0], u[..., 1]
            # 1 + (e^-tu - 1)(e^-tv - 1)/(e^-t - 1) = D / (1 - e^-t), D as in the density
            log_ratio = _frank_log_d(th, a, b) - _log_abs_expm1(-th)
            return _out(-log_ratio / th)
        prod = np.prod(np.expm1(-th * u), axis=-1)
        return _out(-np.log1p(prod / np.expm1(-th) ** (d - 1)) / th)


def copula_tail_dependence(theta, *, family, nu=None):
    """Lower and upper tail-dependence coefficients of a Gaussian, Student-t, Clayton, Gumbel or Frank copula. API contract: nu required for family='t', otherwise omit or None; these are asymptotic limits, not finite-threshold empirical frequencies.

    Parameters
    ----------
    theta : float or array_like
        Family parameter (rho for 'gaussian' and 't'; elementwise).
    family : str
        REQUIRED. 'gaussian', 't', 'clayton', 'gumbel' or 'frank'.
    nu : float
        Degrees of freedom, required for 't' only.

    Returns
    -------
    dict with keys 'lower' = lim_{q->0} P(U_2 <= q | U_1 <= q) and
    'upper' = lim_{q->0} P(U_2 > 1-q | U_1 > 1-q) (floats or arrays).

    Formula
    -------
    gaussian: 0 and 0 for rho < 1 (1 and 1 at rho = 1).
    t:        lower = upper = 2 t_{nu+1}(-sqrt((nu + 1)(1 - rho)/(1 + rho))).
    clayton:  lower = 2^{-1/theta} (theta > 0, else 0), upper = 0.
    gumbel:   lower = 0, upper = 2 - 2^{1/theta}.
    frank:    0 and 0.

    References
    ----------
    Joe (1997) Ch. 2; Nelsen (2006) Ch. 5; Embrechts, McNeil and Straumann (2002)
    (t copula table); McNeil-Frey-Embrechts (2005) Ch. 5.

    Example
    -------
    >>> copula_tail_dependence(0.5, family='t', nu=4)    # printed 0.25 (Embrechts et al. 2002)
    {'lower': 0.2532..., 'upper': 0.2532...}
    """
    fam = _family(family)
    nu = _nu(nu, fam)
    th = np.asarray(theta, dtype=float)
    if fam in _ELLIPTICAL:
        _check_rho(th)
        if fam == "gaussian":
            lam = np.where(th >= 1.0, 1.0, 0.0)
        else:
            with np.errstate(divide="ignore"):
                arg = -np.sqrt((nu + 1.0) * (1.0 - th) / (1.0 + th))
            lam = 2.0 * stdtr(nu + 1.0, arg)
        return {"lower": _out(lam), "upper": _out(lam)}
    _check_archimedean(fam, th, 2)
    zero = np.zeros(th.shape)
    if fam == "clayton":
        pos = th > 0
        lower = np.where(pos, np.exp2(-1.0 / np.where(pos, th, 1.0)), 0.0)
        return {"lower": _out(lower), "upper": _out(zero)}
    if fam == "gumbel":
        return {"lower": _out(zero), "upper": _out(2.0 - np.exp2(1.0 / th))}
    return {"lower": _out(zero), "upper": _out(zero)}
