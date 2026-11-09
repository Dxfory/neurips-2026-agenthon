"""Characteristic-based latent factors and explicit portfolio constraints.

IPCA is ordinary alternating least squares of r[t,i] = z[t,i]' Gamma f[t].
The caller provides lagged characteristics, a finite cleaned panel and initial
factors. No missing-value imputation, time shifting or random initialization is
hidden in the estimator. Sorting and weight constraints similarly operate on
already aligned numerical arrays, without task-specific file handling.
"""
from __future__ import annotations

import numbers

import numpy as np

__all__ = ["ipca_fit", "dependent_sorts", "neutralize_weights", "turnover_cap"]


def _finite(value, name):
    a = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be finite; apply an explicit missing-data policy first")
    return a


def _count(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _r2_uncentered(actual, fitted, axis=None):
    sse = np.sum((actual - fitted)**2, axis=axis)
    denom = np.sum(actual**2, axis=axis)
    ratio = np.divide(sse, denom, out=np.zeros_like(sse, dtype=float), where=denom != 0)
    return np.where(denom == 0, np.where(sse == 0, 1., 0.), 1. - ratio)


def ipca_fit(returns, Z_lagged, *, n_factors, F0, tol, max_iter, convergence,
             rotation, sign_rule):
    """Fit instrumented PCA by alternating ordinary least squares.

    Parameters
    ----------
    returns : array_like, shape (T, N)
        Finite excess returns, rows time and columns assets. Units are retained.
    Z_lagged : array_like, shape (T, N, L)
        Finite characteristics available BEFORE the corresponding return period.
        They must already be time-lagged and aligned; no standardization, NaN
        replacement, intercept or date shift is performed internally.
    n_factors : int >= 1
        Number K of latent factors, no larger than min(T, N, L).
    F0 : array_like, shape (T, K)
        Explicit finite initial factor matrix with full column rank. It is copied
        and not modified. Use the caller's chosen RNG/seed to construct it.
    tol : float >= 0
        Absolute convergence threshold in the chosen metric.
    max_iter : int >= 1
        Maximum complete Gamma/F alternating updates. Reaching this limit is
        reported with converged=False unless the threshold was actually met.
    convergence : {'factors', 'parameters', 'loss'}
        'factors': max absolute change in F; 'parameters': the maximum change
        across F and Gamma (needs at least two iterations); 'loss': absolute
        change in residual sum of squares (needs at least two iterations).
    rotation : {'none', 'covariance'}
        'covariance' diagonalizes the UNCENTERED factor second moment F'F/T by
        an orthogonal eigenvector rotation, ordered by descending eigenvalue.
        The SAME rotation is applied to Gamma. There is NO whitening or scale
        normalization; factor standard deviations are not made equal to one.
    sign_rule : {'none', 'mean'}
        'mean' flips both corresponding columns of Gamma and F when factor mean
        is negative. An exactly zero mean is left unchanged. Rotations/signs are
        applied only after ALS convergence/termination.

    Returns
    -------
    dict
        Gamma (L,K), factors (T,K), fitted/residuals (T,N), orthogonal rotation
        (K,K), factor_covariance=F'F/T, factor_variances=diag(F'F/T), total_r2
        (uncentered), cross_sectional_r2 (uncentered per time), loss_history
        (sum of squared errors after each full update), n_iter, converged and
        convergence_value. Nonconverged estimates remain explicit usable results.
        Return decomposition has no intercept: r_hat[t] = Z_lagged[t]@Gamma@f[t].

    Example
    -------
    >>> fit = ipca_fit(r, z_lag, n_factors=2, F0=initial_factors,
    ...                tol=1e-8, max_iter=1000, convergence='factors',
    ...                rotation='covariance', sign_rule='mean')
    >>> fitted_returns = fit['fitted']
    """
    r = _finite(returns, "returns")
    z = _finite(Z_lagged, "Z_lagged")
    k = _count(n_factors, "n_factors")
    limit = _count(max_iter, "max_iter")
    if r.ndim != 2 or not all(r.shape) or z.ndim != 3 or z.shape[:2] != r.shape or z.shape[2] < 1:
        raise ValueError("returns (T,N) and Z_lagged (T,N,L) must be nonempty and aligned")
    t, n = r.shape
    l = z.shape[2]
    if k > min(t, n, l):
        raise ValueError("n_factors cannot exceed min(T,N,L)")
    f = _finite(F0, "F0").copy()
    if f.shape != (t, k) or np.linalg.matrix_rank(f) != k:
        raise ValueError("F0 must have shape (T,K) and full column rank")
    if not isinstance(tol, numbers.Real) or not np.isfinite(tol) or tol < 0:
        raise ValueError("tol must be finite and nonnegative")
    if convergence not in ("factors", "parameters", "loss"):
        raise ValueError("convergence must be factors, parameters or loss")
    if rotation not in ("none", "covariance") or sign_rule not in ("none", "mean"):
        raise ValueError("rotation must be none/covariance and sign_rule none/mean")
    gamma_old = None
    previous_loss = np.inf
    losses = []
    converged = False
    value = np.inf
    for iteration in range(1, limit + 1):
        # Row (t,i) is vec(z[t,i] f[t]') in C order, matching Gamma.ravel().
        design = (z[..., :, None] * f[:, None, None, :]).reshape(t*n, l*k)
        gamma = np.linalg.lstsq(design, r.reshape(-1), rcond=None)[0].reshape(l, k)
        f_new = np.empty_like(f)
        for time in range(t):
            f_new[time] = np.linalg.lstsq(z[time] @ gamma, r[time], rcond=None)[0]
        fitted = np.einsum("tnl,lk,tk->tn", z, gamma, f_new)
        loss = float(np.sum((r - fitted)**2))
        losses.append(loss)
        factor_change = float(np.max(np.abs(f_new - f)))
        if convergence == "factors":
            value = factor_change
        elif convergence == "parameters":
            gamma_change = np.inf if gamma_old is None else float(np.max(np.abs(gamma - gamma_old)))
            value = max(factor_change, gamma_change)
        else:
            value = abs(previous_loss - loss)
        f = f_new
        if value <= tol:
            converged = True
            break
        previous_loss = loss
        gamma_old = gamma
    transform = np.eye(k)
    if rotation == "covariance":
        _, vectors = np.linalg.eigh(f.T @ f / t)
        transform = vectors[:, ::-1]
    rotated_f = f @ transform
    if sign_rule == "mean":
        signs = np.where(rotated_f.mean(axis=0) < 0, -1., 1.)
        transform = transform * signs
    f = f @ transform
    gamma = gamma @ transform
    fitted = np.einsum("tnl,lk,tk->tn", z, gamma, f)
    moment = f.T @ f / t
    return {"Gamma": gamma, "factors": f, "fitted": fitted,
            "residuals": r - fitted, "rotation": transform,
            "factor_covariance": moment, "factor_variances": np.diag(moment).copy(),
            "total_r2": float(_r2_uncentered(r, fitted)),
            "cross_sectional_r2": _r2_uncentered(r, fitted, axis=1),
            "loss_history": np.asarray(losses), "n_iter": iteration,
            "converged": converged, "convergence_value": float(value)}


def dependent_sorts(first, second, ids, *, n_first, n_second):
    """Dependent rank buckets with numeric/string ID ties and floor-rank cuts.

    Parameters
    ----------
    first, second : array_like, shape (N,) or (T,N)
        Signals to sort ascending. First-stage buckets are formed over jointly
        finite observations. Second-stage sorting occurs independently INSIDE
        each first-stage bucket. NaN observations receive NaN in both outputs;
        infinity is rejected. Rows, when present, are independent formation dates.
    ids : array_like, shape (N,)
        Unique, mutually orderable asset IDs, used ascending for equal signals.
        Integer IDs retain integer ordering; they are never converted to strings.
    n_first, n_second : int >= 1
        Number of buckets in each stage. A zero-based rank j among n assets gets
        min(B, floor(B*j/n)+1). Empty buckets are allowed, notably when n < B.
        This convention deliberately differs from linear-quantile qcut ranks.

    Returns
    -------
    dict
        first_bucket and second_bucket, float arrays with the input shape and
        labels starting at one. NaN marks joint ineligibility. No portfolio return,
        eligibility threshold, forced selection or skipped-date state is imposed.

    Example
    -------
    >>> dependent_sorts([0, 0, 0, 0], [1, 2, 3, 4], [1, 2, 3, 4],
    ...                 n_first=1, n_second=5)['second_bucket']
    array([1., 2., 3., 4.])
    """
    a, b = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    if a.shape != b.shape or a.ndim not in (1, 2) or not all(a.shape):
        raise ValueError("first and second must be aligned nonempty vectors or (T,N) panels")
    if np.any(np.isinf(a)) or np.any(np.isinf(b)):
        raise ValueError("signals may contain NaN but not infinity")
    nf, ns = _count(n_first, "n_first"), _count(n_second, "n_second")
    labels = np.asarray(ids)
    if labels.shape != (a.shape[-1],):
        raise ValueError("ids must have one value per asset column")
    if np.issubdtype(labels.dtype, np.number) and not np.all(np.isfinite(labels)):
        raise ValueError("numeric IDs must be finite")
    try:
        if len(set(labels.tolist())) != len(labels):
            raise ValueError("ids must be unique")
        sorted(labels.tolist())
    except TypeError as exc:
        raise ValueError("ids must be mutually orderable scalar values") from exc
    one_d = a.ndim == 1
    if one_d:
        a, b = a[None, :], b[None, :]
    first_bucket = np.full(a.shape, np.nan)
    second_bucket = np.full(a.shape, np.nan)
    for time in range(len(a)):
        valid = np.flatnonzero(np.isfinite(a[time]) & np.isfinite(b[time]))
        if not len(valid):
            continue
        ordered = valid[np.lexsort((labels[valid], a[time, valid]))]
        first_bucket[time, ordered] = np.minimum(nf, nf*np.arange(len(ordered)) // len(ordered) + 1)
        for bucket in range(1, nf + 1):
            members = np.flatnonzero(first_bucket[time] == bucket)
            if len(members):
                order = members[np.lexsort((labels[members], b[time, members]))]
                second_bucket[time, order] = np.minimum(ns, ns*np.arange(len(order)) // len(order) + 1)
    return {"first_bucket": first_bucket[0] if one_d else first_bucket,
            "second_bucket": second_bucket[0] if one_d else second_bucket}


def neutralize_weights(weights, exposures, *, include_intercept,
                       target_exposure=None, net_target=None):
    """Find a minimum-norm hedge satisfying explicit exposure constraints.

    Parameters
    ----------
    weights : array_like, shape (N,)
        Initial finite portfolio weights, already aligned to the exposure rows.
    exposures : array_like, shape (N,K) or (N,)
        Asset factor exposures. A vector denotes one factor; no standardization
        or intercept column is added unless requested below.
    include_intercept : bool
        Include a net-investment constraint. With net_target=None, the ORIGINAL
        sum of weights is preserved, so the HEDGE is dollar neutral. Set
        net_target=0 explicitly for a dollar-neutral FINAL portfolio instead.
    target_exposure : array_like, shape (K,), optional
        Final factor exposure targets, default zero. Redundant feasible constraints
        use the minimum-norm solution; inconsistent constraints raise ValueError.
    net_target : float, optional
        Desired FINAL sum of weights when include_intercept=True. None preserves
        the original sum. Supplying it without an intercept constraint is an error.

    Returns
    -------
    ndarray, shape (N,)
        Initial weights plus the minimum Euclidean-norm correction from lstsq of
        the augmented constraint matrix. No gross rescaling, bounds, turnover cap
        or asset-universe expansion is applied afterward.

    Example
    -------
    >>> neutralize_weights([.5, .3, .2], [[1], [0], [-1]], include_intercept=True)
    array([0.35, 0.3 , 0.35])
    """
    w = _finite(weights, "weights")
    e = _finite(exposures, "exposures")
    if e.ndim == 1:
        e = e[:, None]
    if w.ndim != 1 or not len(w) or e.ndim != 2 or e.shape[0] != len(w):
        raise ValueError("weights (N,) and exposures (N,K) must be nonempty and aligned")
    if not isinstance(include_intercept, (bool, np.bool_)):
        raise ValueError("include_intercept must be a bool")
    target = np.zeros(e.shape[1]) if target_exposure is None else _finite(target_exposure, "target_exposure")
    if target.shape != (e.shape[1],):
        raise ValueError("target_exposure must have one value per factor")
    if include_intercept:
        net = w.sum() if net_target is None else _finite(net_target, "net_target")
        if np.ndim(net) != 0:
            raise ValueError("net_target must be a scalar")
        e = np.column_stack((np.ones(len(w)), e))
        target = np.concatenate(([float(net)], target))
    elif net_target is not None:
        raise ValueError("net_target requires include_intercept=True")
    if not e.shape[1]:
        return w.copy()
    correction = np.linalg.lstsq(e.T, target - e.T @ w, rcond=None)[0]
    result = w + correction
    mismatch = e.T @ result - target
    scale = max(1., float(np.linalg.norm(target, ord=np.inf)),
                float(np.linalg.norm(e.T, ord=np.inf) * np.linalg.norm(result, ord=np.inf)))
    if np.max(np.abs(mismatch)) > 1e-10 * scale:
        raise ValueError("requested exposure constraints are inconsistent")
    return result


def turnover_cap(target, previous, *, max_turnover, convention):
    """Shrink a target toward previous weights to cap aligned L1 turnover.

    Parameters
    ----------
    target, previous : array_like, shape (N,)
        Finite weight vectors ALREADY aligned on the union of asset IDs, with
        missing holdings filled with zero. Previous means the last executed or
        tradable portfolio, not an automatically shifted calendar observation.
    max_turnover : float >= 0
        Maximum turnover under the specified convention.
    convention : {'one_way', 'l1'}
        'one_way' = .5*sum(abs(target-previous)); 'l1' uses the full sum.

    Returns
    -------
    ndarray, shape (N,)
        previous + min(1, cap/turnover)*(target-previous), or unchanged target at
        zero turnover. NO renormalization or reprojection follows the shrink.
        If exposure reprojection is needed, the caller performs it explicitly and
        calls this function again for the final cap; gross leverage may fall.

    Example
    -------
    >>> turnover_cap([.5, -.5], [0, 0], max_turnover=.3, convention='one_way')
    array([ 0.3, -0.3])
    """
    target = _finite(target, "target")
    previous = _finite(previous, "previous")
    if target.ndim != 1 or target.shape != previous.shape:
        raise ValueError("target and previous must be aligned 1-D vectors")
    if not isinstance(max_turnover, numbers.Real) or not np.isfinite(max_turnover) or max_turnover < 0:
        raise ValueError("max_turnover must be finite and nonnegative")
    if convention not in ("one_way", "l1"):
        raise ValueError("convention must be one_way or l1")
    change = target - previous
    measure = float(np.abs(change).sum()) * (.5 if convention == "one_way" else 1.)
    scale = 1. if measure == 0 else min(1., float(max_turnover) / measure)
    return previous + scale*change
