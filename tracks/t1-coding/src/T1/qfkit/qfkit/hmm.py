"""Gaussian hidden Markov models: Baum-Welch EM fit and scaled forward-backward / Viterbi posteriors.

Model: a K-state Markov chain s_t with transition matrix trans[i, j] = P(s_t = j | s_{t-1} = i) and initial
distribution pi; x_t | s_t = k ~ N(mu_k, var_k) (univariate x of shape (T,)) or N(mu_k, Sigma_k) (multivariate x
of shape (T, D), mu (K, D), var (K, D, D)). Conventions for the whole module:
- var are VARIANCES (covariance matrices), never standard deviations; results also give sd = sqrt(var).
- Recursions are Rabiner's scaled forward-backward (log space where a scaled step would underflow):
  loglik = sum_t log c_t.
- States keep the order of the initial parameters until gaussian_hmm_fit's state_order relabels them, once,
  after the last iteration (so a stop rule on the means compares like with like).
- Every initial parameter is the caller's; nothing is random.

Functions
---------
gaussian_hmm_fit  Baum-Welch EM on one sequence; stop rule 'mu' (max |mu_new - mu_old| < tol) or 'loglik'
hmm_posterior     filtered and smoothed state probabilities, pairwise xi, Viterbi path and log-likelihood

Ported from the team's qflib (``hmm.py``: the log-space recursions and the M-step), with the stop rule made a
required keyword and the result keys renamed to qfkit's (var, sd).

References: Baum, Petrie, Soules and Weiss (1970), Annals of Mathematical Statistics 41; Rabiner (1989),
Proceedings of the IEEE 77; Hamilton (1994) Ch. 22; Zucchini, MacDonald and Langrock (2016) Ch. 4.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import logsumexp

__all__ = [
    "gaussian_hmm_fit",
    "hmm_posterior",
]

_LN_2PI = math.log(2.0 * math.pi)
_CONVERGE = ("mu", "loglik")
_ORDERS = ("as_given", "mean", "variance")


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _choice(value, name, allowed):
    if not isinstance(value, str) or value not in allowed:
        raise ValueError(f"{name} must be one of {', '.join(repr(a) for a in allowed)}, got {value!r}")
    return value


def _series(x, min_length):
    data = np.asarray(x, dtype=float)
    if data.ndim == 2 and data.shape[1] == 1:
        data = data[:, 0]
    if data.ndim not in (1, 2):
        raise ValueError(f"x must be 1-D (T,) or 2-D (T, D) (got shape {data.shape})")
    if data.shape[0] < min_length:
        raise ValueError(f"x needs at least {min_length} observations")
    if not np.all(np.isfinite(data)):
        raise ValueError("x contains NaN/inf; drop or fill missing rows first")
    return data, data.ndim == 2


def _params(mu, var, multivariate, n_dim, names=("mu", "var")):
    m = np.array(mu, dtype=float)
    v = np.array(var, dtype=float)
    if m.ndim == 0 or m.shape[0] < 1:
        raise ValueError(f"{names[0]} must hold one mean per state")
    k = m.shape[0]
    if not multivariate:
        if m.shape != (k,) or v.shape != (k,):
            raise ValueError(f"univariate {names[0]} and {names[1]} must both have shape (K,) = ({k},)")
        if np.any(~np.isfinite(m)) or np.any(~(v > 0)):
            raise ValueError(f"{names[1]} must be finite variances > 0 (variances, not standard deviations)")
        return m, v
    if m.shape != (k, n_dim) or v.shape != (k, n_dim, n_dim):
        raise ValueError(f"multivariate {names[0]} must have shape (K, D) and {names[1]} (K, D, D) "
                         f"with D = {n_dim}")
    for j in range(k):
        if not np.allclose(v[j], v[j].T, rtol=1e-8, atol=1e-12) or np.any(np.linalg.eigvalsh(v[j]) <= 0):
            raise ValueError(f"each {names[1]}[k] must be a symmetric positive definite covariance matrix")
    return m, v


def _stochastic(trans, k, name):
    a = np.array(trans, dtype=float)
    if a.shape != (k, k):
        raise ValueError(f"{name} must have shape (K, K) = ({k}, {k})")
    if np.any(a < 0) or np.any(np.abs(a.sum(axis=1) - 1.0) > 1e-8):
        raise ValueError(f"each row of {name} must be non-negative and sum to 1 (row i = from state i)")
    return a


def _distribution(pi, k, name):
    p = np.array(pi, dtype=float)
    if p.shape != (k,) or np.any(p < 0) or abs(float(p.sum()) - 1.0) > 1e-8:
        raise ValueError(f"{name} must be K = {k} non-negative probabilities summing to 1")
    return p


def _log_emission(data, mu, var, multivariate):
    if not multivariate:
        resid = data[:, None] - mu[None, :]
        return -0.5 * (_LN_2PI + np.log(var) + resid * resid / var)
    n_obs, n_dim = data.shape
    out = np.empty((n_obs, mu.shape[0]))
    for k in range(mu.shape[0]):
        _, logdet = np.linalg.slogdet(var[k])
        resid = data - mu[k]
        quad = np.sum(resid * np.linalg.solve(var[k], resid.T).T, axis=1)
        out[:, k] = -0.5 * (n_dim * _LN_2PI + logdet + quad)
    return out


def _forward_backward(log_b, log_a, log_pi):
    """(filtered, smoothed, xi, loglik) by Rabiner's scaled recursions on emissions rescaled per date (each row's
    largest density set to 1, its log added back to loglik); in log space when a scaled step underflows."""
    shift = log_b.max(axis=1, keepdims=True)
    if not np.all(np.isfinite(shift)):
        raise ValueError("an observation has zero density under every state: check the initial parameters")
    b = np.exp(log_b - shift)
    a, p0 = np.exp(log_a), np.exp(log_pi)
    n_obs, k = b.shape
    alpha = np.empty((n_obs, k))
    c = np.empty(n_obs)
    step = p0 * b[0]
    for t in range(n_obs):
        if t:
            step = (alpha[t - 1] @ a) * b[t]
        c[t] = step.sum()
        if not c[t] > 1e-290:
            return _forward_backward_log(log_b, log_a, log_pi)
        alpha[t] = step / c[t]
    beta = np.ones((n_obs, k))
    for t in range(n_obs - 2, -1, -1):
        beta[t] = a @ (b[t + 1] * beta[t + 1]) / c[t + 1]
    gamma = alpha * beta
    gamma /= gamma.sum(axis=1, keepdims=True)
    xi = alpha[:-1, :, None] * a[None, :, :] * (b[1:] * beta[1:])[:, None, :] / c[1:, None, None]
    xi /= xi.sum(axis=(1, 2), keepdims=True)
    return alpha, gamma, xi, float(np.log(c).sum() + shift.sum())


def _forward_backward_log(log_b, log_a, log_pi):
    """The same recursions in log space (slower, for emissions a scaled step cannot hold)."""
    n_obs, k = log_b.shape
    log_alpha = np.empty((n_obs, k))
    log_c = np.empty(n_obs)
    step = log_pi + log_b[0]
    log_c[0] = logsumexp(step)
    log_alpha[0] = step - log_c[0]
    for t in range(1, n_obs):
        step = log_b[t] + logsumexp(log_alpha[t - 1][:, None] + log_a, axis=0)
        log_c[t] = logsumexp(step)
        log_alpha[t] = step - log_c[t]
    if not np.all(np.isfinite(log_c)):
        raise ValueError("the forward likelihood is zero or not finite: check the initial parameters")
    log_beta = np.zeros((n_obs, k))
    for t in range(n_obs - 2, -1, -1):
        log_beta[t] = logsumexp(log_a + (log_b[t + 1] + log_beta[t + 1])[None, :], axis=1) - log_c[t + 1]
    log_gamma = log_alpha + log_beta
    gamma = np.exp(log_gamma - logsumexp(log_gamma, axis=1, keepdims=True))
    log_xi = (log_alpha[:-1, :, None] + log_a[None, :, :] + (log_b[1:] + log_beta[1:])[:, None, :]
              - log_c[1:, None, None])
    xi = np.exp(log_xi - logsumexp(log_xi, axis=(1, 2), keepdims=True)) if n_obs > 1 else np.empty((0, k, k))
    return np.exp(log_alpha), gamma, xi, float(log_c.sum())


def _viterbi(log_b, log_a, log_pi):
    n_obs, k = log_b.shape
    delta = log_pi + log_b[0]
    back = np.zeros((n_obs, k), dtype=int)
    for t in range(1, n_obs):
        scores = delta[:, None] + log_a
        back[t] = np.argmax(scores, axis=0)
        delta = log_b[t] + scores[back[t], np.arange(k)]
    path = np.empty(n_obs, dtype=int)
    path[-1] = int(np.argmax(delta))
    for t in range(n_obs - 2, -1, -1):
        path[t] = back[t + 1, path[t + 1]]
    return path


def _logs(trans, pi):
    with np.errstate(divide="ignore"):
        return np.log(trans), np.log(pi)


def _m_step(data, gamma, xi, multivariate):
    mass = gamma.sum(axis=0)
    occupied = gamma[:-1].sum(axis=0)
    if np.any(mass <= 1e-300) or np.any(occupied <= 1e-300):
        raise ValueError("a state has lost all posterior mass; start from other initial parameters")
    trans = xi.sum(axis=0) / occupied[:, None]
    pi = gamma[0].copy()
    if not multivariate:
        mu = gamma.T @ data / mass
        var = np.sum(gamma * (data[:, None] - mu[None, :]) ** 2, axis=0) / mass
        if np.any(~(var > 0)):
            raise ValueError("a state variance collapsed to 0; start from other initial parameters")
        return mu, var, trans, pi
    mu = gamma.T @ data / mass[:, None]
    var = np.empty((mu.shape[0], data.shape[1], data.shape[1]))
    for k in range(mu.shape[0]):
        resid = data - mu[k]
        block = (resid * gamma[:, k, None]).T @ resid / mass[k]
        var[k] = 0.5 * (block + block.T)
        if np.any(np.linalg.eigvalsh(var[k]) <= 0):
            raise ValueError("a state covariance is no longer positive definite; start from other parameters")
    return mu, var, trans, pi


# ----------------------------------------------------------------------------
# public
# ----------------------------------------------------------------------------
def hmm_posterior(x, *, mu, var, trans, pi):
    """Gaussian hidden Markov model posteriors (Rabiner scaled forward-backward) and the Viterbi path.

    Parameters
    ----------
    x : array_like (T,) univariate, or (T, D) multivariate observations, time-ordered, finite, T >= 1.
    mu : REQUIRED. State means: (K,) univariate, (K, D) multivariate.
    var : REQUIRED. State VARIANCES (K,) (not standard deviations), or covariance matrices (K, D, D).
    trans : REQUIRED. (K, K) transition matrix, row i = from state i, rows sum to 1.
    pi : REQUIRED. (K,) distribution of the first state s_0 (e.g. the stationary distribution).

    Returns
    -------
    dict
        filtered : (T, K) P(s_t = k | x_0..x_t).
        smoothed : (T, K) P(s_t = k | x_0..x_{T-1}) (the gamma of Baum-Welch).
        xi : (T-1, K, K) P(s_t = i, s_{t+1} = j | all x).
        viterbi : (T,) int, the most likely state path (ties to the lower state index).
        loglik : float, log p(x_0..x_{T-1}) = sum_t log c_t.

    Formula
    -------
    alpha_t(j) = b_j(x_t) sum_i alpha_{t-1}(i) a_ij / c_t (scaled forward), beta_t(i) = sum_j a_ij b_j(x_{t+1})
    beta_{t+1}(j) / c_{t+1}, gamma_t = alpha_t beta_t (in log space where a scaled step would underflow).

    References
    ----------
    Rabiner (1989) Sec. V.A (scaling) and III (Viterbi); Zucchini, MacDonald and Langrock (2016) Ch. 5.

    Example
    -------
    >>> post = hmm_posterior(r, mu=[-0.001, 0.001], var=[4e-4, 1e-4], trans=[[0.95, 0.05], [0.05, 0.95]],
    ...                      pi=[0.5, 0.5])
    >>> post['smoothed'][-1], post['viterbi'][-1], post['loglik']
    """
    data, multivariate = _series(x, 1)
    n_dim = data.shape[1] if multivariate else 1
    m, v = _params(mu, var, multivariate, n_dim)
    k = m.shape[0]
    a = _stochastic(trans, k, "trans")
    p = _distribution(pi, k, "pi")
    log_b = _log_emission(data, m, v, multivariate)
    log_a, log_pi = _logs(a, p)
    filtered, smoothed, xi, loglik = _forward_backward(log_b, log_a, log_pi)
    return {"filtered": filtered, "smoothed": smoothed, "xi": xi, "viterbi": _viterbi(log_b, log_a, log_pi),
            "loglik": loglik}


def gaussian_hmm_fit(x, *, mu0, var0, trans0, pi0, tol, max_iter, converge, state_order):
    """Baum-Welch (EM) fit of a Gaussian hidden Markov model on one sequence, with a stated stop rule.

    Parameters
    ----------
    x : array_like (T,) univariate, or (T, D) multivariate observations, time-ordered, finite, T >= 2.
    mu0 : REQUIRED. Initial state means (K,) or (K, D) (e.g. the means of the first and second half).
    var0 : REQUIRED. Initial state VARIANCES (K,) (square a standard deviation first) or covariances (K, D, D).
    trans0 : REQUIRED. Initial (K, K) transition matrix, row i = from state i.
    pi0 : REQUIRED. Initial (K,) distribution of the first state.
    tol : REQUIRED. Tolerance of the stop rule (>= 0), in the units of x for 'mu', of log-likelihood for 'loglik'.
    max_iter : REQUIRED. Maximum number of EM iterations (M-steps), >= 1.
    converge : REQUIRED. The stop rule, checked after each M-step:
        'mu'      stop when max_k |mu_new - mu_old| < tol (element-wise over all states and dimensions);
        'loglik'  stop when the log-likelihood gain loglik(new) - loglik(old) < tol (hmmlearn's rule).
    state_order : REQUIRED. Labels of the fitted states, applied once after the last iteration:
        'as_given' the order of mu0; 'mean' ascending mean (multivariate: mean of the mean vector), so the
        last state has the highest mean; 'variance' ascending variance (multivariate: trace).

    Returns
    -------
    dict
        mu, var, sd : fitted means, variances (covariances) and standard deviations sqrt(var) (univariate)
            or sqrt of the diagonals (multivariate), in state_order.
        trans, pi : fitted transition matrix and first-state distribution (pi = smoothed P(s_0)).
        gamma : (T, K) smoothed state probabilities at the fitted parameters; filtered : (T, K);
        viterbi : (T,) most likely path; loglik : log-likelihood at the fitted parameters.
        n_iter : int, the number of EM iterations (M-steps) performed; converged : bool, whether the stop
            rule fired before max_iter ran out; loglik_initial : at the initial parameters.

    Formula
    -------
    E-step: gamma_t(k), xi_t(i, j) by hmm_posterior. M-step: mu_k = sum_t gamma_t(k) x_t / sum_t gamma_t(k);
    var_k = sum_t gamma_t(k) (x_t - mu_k)^2 / sum_t gamma_t(k) (maximum likelihood, no ddof);
    a_ij = sum_{t<T-1} xi_t(i, j) / sum_{t<T-1} gamma_t(i); pi = gamma_0. The log-likelihood never decreases.

    References
    ----------
    Baum, Petrie, Soules and Weiss (1970); Rabiner (1989) Sec. III.C; Bilmes (1998) "A gentle tutorial of the
    EM algorithm"; Zucchini, MacDonald and Langrock (2016) Sec. 4.2.

    Example
    -------
    >>> fit = gaussian_hmm_fit(r, mu0=[r[:n // 2].mean(), r[n // 2:].mean()],
    ...                        var0=[r[:n // 2].var(ddof=1), r[n // 2:].var(ddof=1)],
    ...                        trans0=[[0.95, 0.05], [0.05, 0.95]], pi0=[0.5, 0.5], tol=1e-6, max_iter=200,
    ...                        converge='mu', state_order='mean')
    >>> bull = 1  # with state_order='mean' the last state has the higher mean
    >>> fit['gamma'][:, bull] > 0.5, fit['sd'], fit['n_iter']
    """
    rule = _choice(converge, "converge", _CONVERGE)
    order_by = _choice(state_order, "state_order", _ORDERS)
    data, multivariate = _series(x, 2)
    n_dim = data.shape[1] if multivariate else 1
    mu, var = _params(mu0, var0, multivariate, n_dim, ("mu0", "var0"))
    k = mu.shape[0]
    trans = _stochastic(trans0, k, "trans0")
    pi = _distribution(pi0, k, "pi0")
    if isinstance(tol, bool) or not isinstance(tol, (int, float, np.integer, np.floating)) or not tol >= 0:
        raise ValueError("tol must be a number >= 0")
    if isinstance(max_iter, bool) or not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
        raise ValueError("max_iter must be an integer >= 1")
    log_a, log_pi = _logs(trans, pi)
    _, gamma, xi, loglik = _forward_backward(_log_emission(data, mu, var, multivariate), log_a, log_pi)
    loglik_initial = loglik
    converged = False
    n_iter = 0
    for _ in range(int(max_iter)):
        mu_new, var_new, trans_new, pi_new = _m_step(data, gamma, xi, multivariate)
        n_iter += 1
        log_a, log_pi = _logs(trans_new, pi_new)
        _, gamma, xi, loglik_new = _forward_backward(_log_emission(data, mu_new, var_new, multivariate), log_a,
                                                     log_pi)
        change = float(np.max(np.abs(mu_new - mu))) if rule == "mu" else loglik_new - loglik
        mu, var, trans, pi, loglik = mu_new, var_new, trans_new, pi_new, loglik_new
        if change < tol:
            converged = True
            break
    if order_by == "as_given":
        order = np.arange(k)
    else:
        key = (mu.mean(axis=1) if multivariate else mu) if order_by == "mean" else (
            np.trace(var, axis1=1, axis2=2) if multivariate else var)
        order = np.argsort(key, kind="mergesort")
    mu, var, pi, trans = mu[order], var[order], pi[order], trans[np.ix_(order, order)]
    post = hmm_posterior(data, mu=mu, var=var, trans=trans, pi=pi)
    sd = np.sqrt(np.diagonal(var, axis1=1, axis2=2)) if multivariate else np.sqrt(var)
    return {"mu": mu, "var": var, "sd": sd, "trans": trans, "pi": pi, "gamma": post["smoothed"],
            "filtered": post["filtered"], "viterbi": post["viterbi"], "loglik": post["loglik"], "n_iter": n_iter,
            "converged": converged, "loglik_initial": float(loglik_initial)}
