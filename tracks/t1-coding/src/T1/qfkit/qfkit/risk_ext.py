"""Scalar DCC(1,1) correlation filtering and stationary row resampling.

These are numerical primitives, without asset names, input files or task
pipelines. DCC follows Engle (2002), J. Business & Economic Statistics 20(3),
339--350. Circular stationary blocks follow Politis and Romano (1994),
J. American Statistical Association 89(428), 1303--1313.
"""
from __future__ import annotations

import numpy as np

__all__ = ['dcc_filter', 'dcc_fit', 'stationary_bootstrap_indices']


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f'{name} must be a positive integer')
    return int(value)


def _positive_definite(value, n, name):
    out = np.asarray(value, dtype=float)
    if out.shape != (n, n) or not np.all(np.isfinite(out)):
        raise ValueError(f'{name} must be a finite ({n}, {n}) matrix')
    if not np.allclose(out, out.T, rtol=1e-12, atol=1e-14):
        raise ValueError(f'{name} must be symmetric')
    out = (out + out.T) * .5
    try:
        np.linalg.cholesky(out)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f'{name} must be positive definite; specify a valid target explicitly') from exc
    return out.copy()


def _prepare(std_resid, qbar, q0, vol):
    z = np.asarray(std_resid, dtype=float)
    if z.ndim != 2 or z.shape[0] < 2 or z.shape[1] < 2 or not np.all(np.isfinite(z)):
        raise ValueError('std_resid must be a finite time-by-asset matrix with at least two rows and assets')
    n = z.shape[1]
    if qbar is None:
        qbar = z.T @ z / len(z)
    target = _positive_definite(qbar, n, 'qbar')
    initial = target.copy() if q0 is None else _positive_definite(q0, n, 'q0')
    if vol is not None:
        vol = np.asarray(vol, dtype=float)
        if vol.shape != z.shape or not np.all(np.isfinite(vol)) or np.any(vol <= 0):
            raise ValueError('vol must have std_resid shape and contain positive finite standard deviations')
    return z, target, initial, vol


def _parameters(a, b, *, strictly_positive=False):
    try:
        a, b = float(a), float(b)
    except (TypeError, ValueError) as exc:
        raise ValueError('a and b must be finite scalars') from exc
    if not np.isfinite(a) or not np.isfinite(b) or a < 0 or b < 0 or a+b >= 1:
        raise ValueError('DCC requires a >= 0, b >= 0 and a+b < 1')
    if strictly_positive and (a == 0 or b == 0):
        raise ValueError('DCC fitting start requires strictly positive a and b')
    return a, b


def _filter(z, a, b, target, initial, vol):
    t, n = z.shape
    q = np.empty((t, n, n))
    q[0] = initial
    intercept = (1-a-b) * target
    for j in range(1, t):
        q[j] = intercept + a*np.outer(z[j-1], z[j-1]) + b*q[j-1]
    sd = np.sqrt(np.diagonal(q, axis1=1, axis2=2))
    corr = q / (sd[:, :, None] * sd[:, None, :])
    sign, logdet = np.linalg.slogdet(corr)
    if np.any(sign <= 0) or not np.all(np.isfinite(logdet)):
        raise ValueError('DCC correlation matrices are numerically nonpositive definite')
    try:
        solved = np.linalg.solve(corr, z[:, :, None])[:, :, 0]
    except np.linalg.LinAlgError as exc:
        raise ValueError('DCC correlation matrices are numerically singular') from exc
    quadratic = np.einsum('ti,ti->t', z, solved)
    ll_obs = -.5 * (logdet + quadratic)
    loglik = float(np.sum(ll_obs))
    return {'a': a, 'b': b, 'persistence': a+b, 'qbar': target.copy(),
            'q0': initial.copy(), 'Q': q, 'R': corr,
            'H': None if vol is None else corr*vol[:, :, None]*vol[:, None, :],
            'loglik_obs': ll_obs, 'loglik': loglik,
            'correlation_loglik': loglik+.5*float(np.sum(z*z)),
            'gaussian_loglik': loglik-.5*t*n*np.log(2*np.pi),
            'nobs': t, 'n_assets': n}


def dcc_filter(std_resid, *, a, b, qbar=None, q0=None, vol=None):
    """Filter scalar DCC(1,1), with explicit lag and likelihood conventions.

    Parameters
    ----------
    std_resid : array_like, shape (time, assets)
        Finite standardized innovations, at least two rows and two assets.
        No centering or marginal volatility fitting is performed.
    a, b : float
        Nonnegative innovation and persistence coefficients, a+b strictly < 1.
        Zero is permitted here for constant-correlation limiting cases.
    qbar : array_like, shape (assets, assets), optional
        Positive definite unconditional target. Default is the UNCENTERED
        second moment Z.T @ Z / T. To use sample covariance or correlation,
        calculate that matrix explicitly and supply it. A singular target
        raises rather than silently adding a ridge.
    q0 : array_like, shape (assets, assets), optional
        Positive definite initial Q[0], default qbar. For t >= 1 the recursion
        is Q[t]=(1-a-b)*qbar+a*outer(Z[t-1],Z[t-1])+b*Q[t-1].
    vol : array_like, shape (time, assets), optional
        Positive conditional STANDARD DEVIATIONS, aligned with Z. Supply in
        any declared return unit; resulting H is in its squared unit. This
        function neither rescales nor annualizes values.

    Returns
    -------
    dict
        Q and R have shape (time, assets, assets); R has unit diagonal.
        H = diag(vol[t]) @ R[t] @ diag(vol[t]), or None when vol is absent.
        loglik_obs[t] = -0.5*(logdet(R[t])+Z[t].T@inv(R[t])@Z[t]); loglik
        sums ALL rows including t=0 and omits the normal density constant.
        correlation_loglik adds 0.5*sum(Z**2), the parameter-independent term
        used in Engle's correlation likelihood. gaussian_loglik instead adds
        the full normal constant -0.5*T*N*log(2*pi). Also returns a, b,
        persistence, qbar, q0, nobs and n_assets. The three likelihoods have
        identical optimizers but different reported values.

    Example
    -------
    >>> z = [[1., 2.], [2., -1.], [-1., .5]]
    >>> out = dcc_filter(z, a=.05, b=.93, qbar=np.eye(2))
    >>> out['R'].shape
    (3, 2, 2)
    """
    a, b = _parameters(a, b)
    z, target, initial, vol = _prepare(std_resid, qbar, q0, vol)
    return _filter(z, a, b, target, initial, vol)


def _from_logits(x):
    # Include the intercept weight in a three-way stable softmax. Bounding
    # logits to +/-30 keeps all three weights positive in float64.
    v = np.r_[0., x]
    w = np.exp(v-np.max(v))
    w /= np.sum(w)
    return float(w[1]), float(w[2])


def dcc_fit(std_resid, *, start=(.05, .93), qbar=None, q0=None, vol=None,
            method='L-BFGS-B', tol=1e-8, max_iter=1000):
    """Estimate positive stationary DCC coefficients by correlation QML.

    Parameters
    ----------
    std_resid, qbar, q0, vol : array_like
        As in dcc_filter, including the uncentered default qbar and t=0
        likelihood contribution. Target and initial matrices remain fixed
        during optimization; this is second-stage DCC, not joint GARCH MLE.
    start : pair of float, optional
        Initial ACTUAL (a,b), default (.05,.93), positive with sum < 1.
    method : {'L-BFGS-B'}, optional
        scipy.optimize.minimize method. The objective is evaluated in LOGIT
        coordinates u=log(a/(1-a-b)), v=log(b/(1-a-b)), bounded to [-30,30].
        A three-way softmax guarantees positive a,b and a+b<1. start must map
        inside those bounds. Initial actual coefficients equal start; the
        optimization trajectory is NOT direct bounded optimization in a,b.
        If a task specifies a direct-parameter trajectory, use dcc_filter to
        form that objective and run its specified optimizer yourself.
    tol : float, optional
        Positive tolerance passed to minimize (default 1e-8).
    max_iter : int, optional
        Positive iteration limit (default 1000).

    Returns
    -------
    dict
        All dcc_filter outputs at the final coefficients, plus converged,
        status, message, n_iter, n_eval, method, parameterization='logit',
        start, optimizer_x (logits), optimizer_fun, and optimizer_bounds.
        A failed optimizer is reported explicitly via converged=False;
        returned finite numerical values alone do not certify convergence.

    Example
    -------
    >>> z = np.random.default_rng(1).standard_normal((100, 2))
    >>> out = dcc_fit(z, start=(.05,.93), max_iter=200)
    >>> out['parameterization']
    'logit'
    """
    from scipy.optimize import minimize

    if method != 'L-BFGS-B':
        raise ValueError("method must be 'L-BFGS-B'")
    if not np.isscalar(tol) or not np.isfinite(tol) or tol <= 0:
        raise ValueError('tol must be positive and finite')
    max_iter = _positive_integer(max_iter, 'max_iter')
    try:
        a, b = start
    except (TypeError, ValueError) as exc:
        raise ValueError('start must contain exactly two coefficients') from exc
    a, b = _parameters(a, b, strictly_positive=True)
    x0 = np.log(np.array([a, b]) / (1-a-b))
    bounds = [(-30., 30.), (-30., 30.)]
    if np.any(np.abs(x0) > 30):
        raise ValueError('start logit coordinates must lie within [-30,30]')
    z, target, initial, vol = _prepare(std_resid, qbar, q0, vol)

    def objective(x):
        ca, cb = _from_logits(x)
        try:
            return -_filter(z, ca, cb, target, initial, None)['loglik']
        except ValueError:
            return float('inf')

    result = minimize(objective, x0, method=method, bounds=bounds, tol=float(tol),
                      options={'maxiter': max_iter})
    a, b = _from_logits(result.x)
    out = _filter(z, a, b, target, initial, vol)
    out.update(converged=bool(result.success), status=int(result.status),
               message=str(result.message), n_iter=int(result.nit),
               n_eval=int(result.nfev), method=method, parameterization='logit',
               start=tuple(float(v) for v in start), optimizer_x=result.x.copy(),
               optimizer_fun=float(result.fun), optimizer_bounds=bounds)
    return out


def stationary_bootstrap_indices(n_obs, *, n_samples, length, p, seed):
    """Generate circular stationary-block resampling indices for joint rows.

    Parameters
    ----------
    n_obs : int
        Positive number of time-ordered source observations.
    n_samples : int
        Positive number of independent bootstrap paths.
    length : int
        Positive number of observations per bootstrap path. For one sequence
        of B joint residual rows, use n_samples=1 and length=B.
    p : float
        Geometric restart probability in (0,1]; expected block length 1/p.
        Initial indices are uniform. Thereafter each path restarts uniformly
        with probability p, or advances one index modulo n_obs. p=1 is IID.
    seed : int, numpy.random.Generator, or None
        Passed to default_rng. A supplied Generator advances its existing
        stream. Draw order: all initial indices, then for each subsequent
        column all restart uniforms followed by new indices for restarting
        paths only, in path order. At p=1 uniforms are still drawn.

    Returns
    -------
    ndarray of int, shape (n_samples, length)
        Each element is in [0,n_obs). Index a whole source row with this
        array, rather than independently resampling each asset column, to
        preserve contemporaneous cross-sectional dependence. No observations
        or asset-specific distributions are transformed by this function.

    Example
    -------
    >>> z = np.arange(20.).reshape(10, 2)
    >>> idx = stationary_bootstrap_indices(10, n_samples=1, length=1000,
    ...                                    p=1/21, seed=42)
    >>> z[idx[0]].shape
    (1000, 2)
    """
    n_obs = _positive_integer(n_obs, 'n_obs')
    n_samples = _positive_integer(n_samples, 'n_samples')
    length = _positive_integer(length, 'length')
    if not np.isscalar(p) or not np.isfinite(p) or not 0 < p <= 1:
        raise ValueError('p must be finite and in (0,1]')
    rng = np.random.default_rng(seed)
    indices = np.empty((n_samples, length), dtype=np.int64)
    indices[:, 0] = rng.integers(0, n_obs, size=n_samples)
    for j in range(1, length):
        restart = rng.random(n_samples) < p
        indices[:, j] = (indices[:, j-1]+1) % n_obs
        indices[restart, j] = rng.integers(0, n_obs, size=int(np.sum(restart)))
    return indices
