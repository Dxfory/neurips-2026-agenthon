"""Compound (aggregate) loss distributions S = X_1 + ... + X_N: severity discretisation, the Panjer recursion and
the FFT (Fourier) method for Poisson, negative binomial and binomial claim counts, and VaR/ES of a discrete loss
distribution.

Conventions for the whole module:
- A severity pmf f lives on the grid 0, h, 2h, ..., (m-1)h (f[j] = P(X = j h)); the aggregate pmf g is returned
  on the same grid (g[s] = P(S = s h)), so loss amounts are grid * h, not bin indices.
- freq is the claim-count distribution: ('poisson', lam), ('negative_binomial', r, beta) with mean r beta and
  variance r beta (1 + beta), or ('binomial', n, p).
- Mass beyond the grid is not returned (the Panjer output stops at the last bin; the FFT folds mass beyond its
  buffer back onto the start, which ``pad`` controls).

Functions
---------
discretise_severity  severity pmf on 0, h, ..., (n-1)h by 'rounding', 'lower' or 'upper' (actuar's names)
panjer_recursion     aggregate pmf by the (a, b, 0) recursion of Panjer (1981)
compound_fft         aggregate pmf by FFT: g = ifft(PGF(fft(f))) on a buffer of pad x len(f)
pmf_var_es           VaR = min{x: F(x) >= alpha} and ES = mean of S >= VaR or S > VaR of a discrete distribution

Ported from the team's qflib (``aggregate_loss.discretise_severity``, ``panjer_recursion``, ``panjer_binomial``,
``compound_fft``, ``pmf_var_es``) without its moment-matching discretisation (which cancels and raises on long
grids) and with the lower/upper names, which qflib had swapped, set as actuar uses them.

References: Panjer (1981), ASTIN Bulletin 12; Klugman, Panjer and Willmot, Loss Models (2019) Ch. 9 (Secs. 9.6
recursion and 9.7 discretisation); Embrechts and Frei (2009), Mathematical Methods of Operations Research 69 (FFT
and aliasing); Dutang, Goulet and Pigeon (2008), J. Statistical Software 25(7) (actuar ``discretize``).
"""

from __future__ import annotations

import math
import numbers

import numpy as np

__all__ = [
    "discretise_severity",
    "panjer_recursion",
    "compound_fft",
    "pmf_var_es",
]

_DISCRETE = ("rounding", "lower", "upper")
_FREQ = ("poisson", "negative_binomial", "binomial")
_ES_RULES = ("ge", "gt")


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _choice(value, name, allowed):
    if not isinstance(value, str) or value not in allowed:
        raise ValueError(f"{name} must be one of {', '.join(repr(a) for a in allowed)}, got {value!r}")
    return value


def _real(v):
    return not isinstance(v, bool) and isinstance(v, numbers.Real) and math.isfinite(float(v))


def _freq(freq):
    """(kind, a, b, pgf) of a claim-count distribution given as a tuple."""
    usage = ("freq must be ('poisson', lam), ('negative_binomial', r, beta) (mean r*beta) or ('binomial', n, p)")
    if not isinstance(freq, (tuple, list)) or not freq or freq[0] not in _FREQ:
        raise ValueError(usage)
    kind = freq[0]
    if kind == "poisson":
        if len(freq) != 2 or not _real(freq[1]) or not freq[1] > 0:
            raise ValueError(usage + "; lam > 0")
        lam = float(freq[1])
        return kind, 0.0, lam, lambda z: np.exp(lam * (z - 1.0))
    if kind == "negative_binomial":
        if len(freq) != 3 or not (_real(freq[1]) and _real(freq[2])) or not (freq[1] > 0 and freq[2] > 0):
            raise ValueError(usage + "; r > 0 and beta > 0")
        r, beta = float(freq[1]), float(freq[2])
        a = beta / (1.0 + beta)
        return kind, a, (r - 1.0) * a, lambda z: (1.0 - beta * (z - 1.0)) ** (-r)
    if len(freq) != 3 or isinstance(freq[1], bool) or not isinstance(freq[1], numbers.Integral) or freq[1] < 0 \
            or not _real(freq[2]) or not 0.0 <= freq[2] <= 1.0:
        raise ValueError(usage + "; n an int >= 0 and p in [0, 1]")
    n, p = int(freq[1]), float(freq[2])
    a = -p / (1.0 - p) if p < 1.0 else -math.inf
    return kind, a, (n + 1.0) * p / (1.0 - p) if p < 1.0 else math.inf, lambda z: (1.0 + p * (z - 1.0)) ** n


def _severity(f):
    a = np.asarray(f, dtype=float)
    if a.ndim != 1 or a.size == 0 or np.any(~np.isfinite(a)) or np.any(a < 0.0):
        raise ValueError("f must be a non-empty 1-D severity pmf with masses >= 0 on 0, h, 2h, ...")
    if a.sum() > 1.0 + 1e-8:
        raise ValueError("the severity masses must sum to at most 1")
    return a


# ----------------------------------------------------------------------------
# public
# ----------------------------------------------------------------------------
def discretise_severity(cdf, *, h, n_points, method):
    """Discretise a continuous severity distribution onto the grid 0, h, ..., (n_points-1) h (Klugman-Panjer-Willmot
    method of rounding, or the lower/upper bound discretisations).

    Parameters
    ----------
    cdf : callable. The severity CDF F, applied to a numpy array (e.g. ``lambda x: stats.gamma.cdf(x, 2,
        scale=3)``); the severity is >= 0 (F(0-) = 0).
    h : REQUIRED. Span of the grid (> 0), in loss units.
    n_points : REQUIRED. Number of grid points m (int >= 1); choose m h beyond the aggregate's tail.
    method : REQUIRED.
        'rounding' mass of ((j-1/2)h, (j+1/2)h] at jh (f_0 = F(h/2)): keeps the mean to O(h^2);
        'lower'    mass of ((j-1)h, jh] at jh (f_0 = F(0)): its CDF is a lower bound of F (losses rounded up);
        'upper'    mass of [jh, (j+1)h) at jh (f_0 = F(h)): its CDF is an upper bound of F (losses rounded down).

    Returns
    -------
    dict
        pmf : (m,) masses; grid : (m,) the loss values j h; truncated_mass : 1 - pmf.sum(), the mass beyond the grid.

    References
    ----------
    Klugman, Panjer and Willmot (2019) Sec. 9.7; Dutang, Goulet and Pigeon (2008) (actuar methods 'rounding',
    'lower', 'upper').

    Example
    -------
    >>> sev = discretise_severity(lambda x: stats.lognorm.cdf(x, 1.2, scale=1000), h=50.0, n_points=2**14,
    ...                           method='rounding')
    """
    how = _choice(method, "method", _DISCRETE)
    if not _real(h) or not h > 0:
        raise ValueError("h must be a number > 0")
    if isinstance(n_points, bool) or not isinstance(n_points, numbers.Integral) or n_points < 1:
        raise ValueError("n_points must be an int >= 1")
    if not callable(cdf):
        raise ValueError("cdf must be a callable F(x) that accepts a numpy array")
    m, step = int(n_points), float(h)
    if how == "rounding":
        edges = (np.arange(m + 1) - 0.5) * step
    elif how == "lower":
        edges = (np.arange(m + 1) - 1.0) * step
    else:
        edges = np.arange(m + 1) * step
    values = np.asarray(cdf(np.clip(edges, 0.0, None)), dtype=float)
    if values.shape != edges.shape or not np.all(np.isfinite(values)):
        raise ValueError("cdf must return one finite value per point of a numpy array")
    if np.any(values < -1e-12) or np.any(values > 1.0 + 1e-12) or np.any(np.diff(values) < -1e-12):
        raise ValueError("cdf must be a distribution function: non-decreasing, within [0, 1]")
    values = np.clip(values, 0.0, 1.0)
    values[edges < 0.0] = 0.0
    values[0] = 0.0  # the first edge is 0-: F(0-) = 0, so the atom at 0 goes to the first bin
    pmf = np.clip(np.diff(values), 0.0, None)
    return {"pmf": pmf, "grid": np.arange(m) * step, "truncated_mass": max(0.0, 1.0 - float(pmf.sum()))}


def panjer_recursion(f, *, freq):
    """Aggregate (compound) loss pmf by the Panjer (a, b, 0) recursion for Poisson, negative binomial or binomial
    claim counts.

    Parameters
    ----------
    f : array_like (m,). Severity pmf on 0, h, ..., (m-1)h (e.g. discretise_severity(...)['pmf']).
    freq : REQUIRED. ('poisson', lam), ('negative_binomial', r, beta) (mean r beta, variance r beta (1 + beta)) or
        ('binomial', n, p).

    Returns
    -------
    ndarray (m,). g[s] = P(S = s h); mass beyond the last bin is not returned (g.sum() < 1 by that mass).

    Formula
    -------
    g_0 = P_N(f_0) (the claim-count pgf); g_s = sum_{j=1..s} (a + b j/s) f_j g_{s-j} / (1 - a f_0), with
    (a, b) = (0, lam), (beta/(1+beta), (r-1) beta/(1+beta)), (-p/(1-p), (n+1) p/(1-p)). O(m^2): for very long grids
    use compound_fft.

    References
    ----------
    Panjer (1981); Klugman, Panjer and Willmot (2019) Theorem 9.8.

    Example
    -------
    >>> g = panjer_recursion(sev['pmf'], freq=('poisson', 50.0))
    """
    fs = _severity(f)
    kind, a, b, pgf = _freq(freq)
    m = fs.size
    if kind == "binomial" and freq[2] == 1.0:  # every trial a claim: the n-fold convolution
        g = np.zeros(m)
        g[0] = 1.0
        for _ in range(int(freq[1])):
            g = np.convolve(g, fs)[:m]
        return g
    g = np.zeros(m)
    g[0] = float(pgf(fs[0]))
    scale = 1.0 - a * fs[0]
    j = np.arange(1, m)
    for s in range(1, m):
        jj = j[:s]
        g[s] = float(np.dot((a + b * jj / s) * fs[1:s + 1], g[s - 1::-1][:s])) / scale
    return np.clip(g, 0.0, None)


def compound_fft(f, *, freq, pad):
    """Aggregate (compound) loss pmf by the fast Fourier transform: g = ifft(PGF_N(fft(f))) on a zero-padded buffer.

    Parameters
    ----------
    f : array_like (m,). Severity pmf on 0, h, ..., (m-1)h.
    freq : REQUIRED. ('poisson', lam), ('negative_binomial', r, beta) or ('binomial', n, p), as panjer_recursion.
    pad : REQUIRED. Buffer length as a multiple of m (int >= 1): the FFT works modulo pad*m bins, so aggregate mass
        beyond pad*m*h wraps onto the low bins; pad = 1 only when the grid already covers the aggregate's tail.

    Returns
    -------
    ndarray (m,). g[s] = P(S = s h) on the severity grid (equals panjer_recursion when the buffer covers the tail).

    Formula
    -------
    phi = numpy.fft.fft(f padded with zeros to pad*m) (the characteristic function on the grid);
    g = real(numpy.fft.ifft(P_N(phi)))[:m] with P_N(z) = exp(lam (z - 1)), (1 - beta (z - 1))^-r, (1 + p (z - 1))^n.
    The pairing fft then ifft is what inverts it (ifft twice reflects the pmf).

    References
    ----------
    Embrechts and Frei (2009); Klugman, Panjer and Willmot (2019) Sec. 9.8.

    Example
    -------
    >>> g = compound_fft(sev['pmf'], freq=('poisson', 50.0), pad=4)
    >>> pmf_var_es(g, sev['grid'], 0.99, es_rule='gt')
    """
    fs = _severity(f)
    _, _, _, pgf = _freq(freq)
    if isinstance(pad, bool) or not isinstance(pad, numbers.Integral) or pad < 1:
        raise ValueError("pad must be an int >= 1 (the buffer is pad * len(f) bins)")
    m = fs.size
    buffer = np.zeros(int(pad) * m)
    buffer[:m] = fs
    g = np.real(np.fft.ifft(pgf(np.fft.fft(buffer))))[:m]
    return np.where(g < 0.0, 0.0, g)  # FFT round-off of order 1e-17 below zero


def pmf_var_es(pmf, grid, alpha, *, es_rule):
    """Value-at-Risk and expected shortfall of a discrete loss distribution (an aggregate-loss pmf):
    VaR = min{x on the grid : F(x) >= alpha}; ES = the mean loss over grid >= VaR ('ge') or grid > VaR ('gt').

    Parameters
    ----------
    pmf : array_like (m,). Masses (>= 0) on the grid; their sum may be below 1 (mass beyond the grid).
    grid : array_like (m,). Increasing loss values (e.g. j h).
    alpha : float in (0, 1). Confidence level, e.g. 0.99.
    es_rule : REQUIRED. 'ge' E[S | S >= VaR] or 'gt' E[S | S > VaR], on the grid.

    Returns
    -------
    dict
        var, es : in the units of grid; cdf_at_var : F(VaR); tail_mass : the mass ES averages over.

    Formula
    -------
    F(x_k) = sum_{j<=k} pmf_j, compared with alpha to within 1e-12; raises when the grid's mass never reaches alpha
    (extend the grid).

    References
    ----------
    McNeil, Frey and Embrechts (2015) Sec. 2.3 (generalised inverse, expected shortfall); Klugman, Panjer and
    Willmot (2019) Ch. 3.

    Example
    -------
    >>> pmf_var_es(g, h * np.arange(len(g)), 0.995, es_rule='gt')
    """
    p = np.asarray(pmf, dtype=float)
    x = np.asarray(grid, dtype=float)
    if p.ndim != 1 or p.shape != x.shape or p.size == 0:
        raise ValueError("pmf and grid must be 1-D arrays of the same length")
    if np.any(~np.isfinite(p)) or np.any(p < -1e-12) or np.any(~np.isfinite(x)) or np.any(np.diff(x) <= 0):
        raise ValueError("pmf must be finite masses >= 0 and grid strictly increasing")
    if not _real(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha is a confidence level in (0, 1), e.g. 0.99")
    rule = _choice(es_rule, "es_rule", _ES_RULES)
    p = np.clip(p, 0.0, None)
    cum = np.cumsum(p)
    hit = np.flatnonzero(cum >= float(alpha) - 1e-12)
    if hit.size == 0:
        raise ValueError(f"the pmf's mass ({cum[-1]:.6g}) never reaches alpha = {alpha}: extend the grid")
    k = int(hit[0])
    tail = slice(k, None) if rule == "ge" else slice(k + 1, None)
    mass = float(p[tail].sum())
    if mass <= 0.0:
        raise ValueError("no mass lies above VaR on the grid: use es_rule='ge' or extend the grid")
    return {"var": float(x[k]), "es": float(np.dot(p[tail], x[tail]) / mass), "cdf_at_var": float(cum[k]),
            "tail_mass": mass}
