"""Numerical option pricing: binomial tree, Crank-Nicolson PDE, American approximations, GBM Monte Carlo.

Conventions for the whole module (the same as ``bsm``): S is spot, K strike,
T time to expiry in YEARS; r and q are continuously compounded annual rates as
decimals (0.05, not 5); sigma is the annual volatility as a decimal (0.20, not
20). ``kind`` is ONE string, 'call' or 'put'. ``exercise`` is 'european' or
'american'. Every method-specific choice (tree steps, grid size and upper
boundary, number of paths, seed, estimator) is a REQUIRED keyword: there is no
hidden default grid, step count or random seed.
The pricers (all but gbm_paths / gbm_mc_price, which take scalars) broadcast
their numeric inputs and return a numpy float64 (a ``float`` subclass) when
every input is a scalar and an ``np.ndarray`` otherwise.

Functions
---------
crr_price      Cox-Ross-Rubinstein binomial tree, European/American, optional tree Greeks
fd_cn_price    Crank-Nicolson finite differences on a uniform S grid (PSOR for American)
baw_price      Barone-Adesi-Whaley (1987) quadratic approximation of an American option
bs2002_price   Bjerksund-Stensland (2002) two-step-boundary American approximation
gbm_paths      exact lognormal GBM paths from numpy default_rng(seed), optional antithetics
gbm_mc_price   Monte Carlo price of a path payoff: plain, antithetic or control variate
"""

from __future__ import annotations

import math
import numbers

import numpy as np
from scipy import optimize
from scipy.interpolate import CubicSpline
from scipy.linalg import solve_banded
from scipy.special import ndtr, owens_t

from .bsm import bsm_d1d2, bsm_price

__all__ = [
    "crr_price",
    "fd_cn_price",
    "baw_price",
    "bs2002_price",
    "gbm_paths",
    "gbm_mc_price",
]

_Z975 = 1.959963984540054  # standard normal 97.5% quantile (two-sided 95% interval)


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _out(x):
    """0-d result -> numpy float64 scalar; otherwise the ndarray itself."""
    x = np.asarray(x, dtype=float)
    return x[()] if x.ndim == 0 else x


def _arr(x):
    return np.asarray(x, dtype=float)


def _sign(kind):
    """'call' -> +1.0, 'put' -> -1.0 (case-insensitive); anything else raises."""
    if not isinstance(kind, str):
        raise TypeError("kind must be the string 'call' or 'put'")
    k = kind.strip().lower()
    if k == "call":
        return 1.0
    if k == "put":
        return -1.0
    raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")


def _american(exercise):
    """'american' -> True, 'european' -> False (case-insensitive); anything else raises."""
    if not isinstance(exercise, str):
        raise TypeError("exercise must be the string 'european' or 'american'")
    e = exercise.strip().lower()
    if e == "american":
        return True
    if e == "european":
        return False
    raise ValueError(f"exercise must be 'european' or 'american', got {exercise!r}")


def _int_arg(value, name, minimum):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be an integer, got {value!r}")
    v = int(value)
    if v < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {v}")
    return v


def _check_pos(**named):
    for name, v in named.items():
        if not np.all(np.isfinite(v)) or np.any(v <= 0):
            raise ValueError(f"{name} must be finite and > 0")


def _check_finite(**named):
    for name, v in named.items():
        if not np.all(np.isfinite(v)):
            raise ValueError(f"{name} must be finite")


def _broadcast_flat(*arrays):
    shape = np.broadcast_shapes(*(a.shape for a in arrays))
    return shape, [np.broadcast_to(a, shape).ravel().copy() for a in arrays]


# ----------------------------------------------------------------------------
# Cox-Ross-Rubinstein binomial tree
# ----------------------------------------------------------------------------
def crr_price(S, K, T, r, sigma, *, kind, exercise, steps, q=0.0, greeks=False):
    """Cox-Ross-Rubinstein binomial-tree price of a European or American option, with tree Greeks.

    Parameters
    ----------
    S, K : spot and strike (> 0).
    T : time to expiry in years (> 0).
    r, q : continuously compounded annual rate and dividend yield (decimals);
        q defaults to 0.0. For a futures option pass q = r.
    sigma : annual volatility (decimal, > 0).
    kind : 'call' or 'put' (required).
    exercise : 'european' or 'american' (required). American exercise is
        checked at every tree node, including t = 0.
    steps : number of time steps N (required, >= 1; >= 2 when greeks=True).
    greeks : False (default) returns the price only; True returns a dict.

    Returns
    -------
    Price (float or array), or with greeks=True a dict with keys
    'price', 'delta', 'gamma', 'theta' read off the tree as in Hull:
    delta = (f_11 - f_10)/(S u - S d);
    gamma = [(f_22 - f_21)/(S u^2 - S) - (f_21 - f_20)/(S - S d^2)] / (0.5 (S u^2 - S d^2));
    theta = (f_21 - f_00)/(2 dt), PER YEAR (dV/dt in calendar time; divide by
    365 for a per-calendar-day theta). f_ij is the value at step i after j up-moves.

    Formula
    -------
    dt = T/N, u = e^{sigma sqrt(dt)}, d = 1/u, p = (e^{(r-q) dt} - d)/(u - d);
    f_ij = e^{-r dt} [p f_{i+1,j+1} + (1-p) f_{i+1,j}], then (American)
    f_ij = max(f_ij, payoff(S u^j d^{i-j})). Raises ValueError if p is outside
    [0, 1] (increase steps). Error is O(1/N) and oscillates with N.

    References
    ----------
    Cox, Ross and Rubinstein (1979); Hull OFOD "Basic Numerical Procedures"
    (American put tree example); Haug (2007) Ch. 7.

    Example
    -------
    >>> crr_price(50, 50, 5/12, 0.10, 0.40, kind='put', exercise='american', steps=5)   # Hull: 4.49
    4.4884...
    """
    w = _sign(kind)
    american = _american(exercise)
    N = _int_arg(steps, "steps", 2 if greeks else 1)
    S, K, T, r, sigma, q = map(_arr, (S, K, T, r, sigma, q))
    _check_pos(S=S, K=K, T=T, sigma=sigma)
    _check_finite(r=r, q=q)
    shape, (S, K, T, r, sigma, q) = _broadcast_flat(S, K, T, r, sigma, q)

    dt = T / N
    s = sigma * np.sqrt(dt)
    u = np.exp(s)
    d = np.exp(-s)
    a = np.exp((r - q) * dt)
    p = (a - d) / (u - d)
    if np.any((p < 0) | (p > 1)):
        raise ValueError("CRR probability outside [0, 1]: sigma sqrt(dt) too small for |r - q| dt; "
                         "increase steps")
    pm = (u - a) / (u - d)  # 1 - p without cancellation
    disc = np.exp(-r * dt)
    Sc, Kc, sc = S[:, None], K[:, None], s[:, None]
    pc, pmc, dc = p[:, None], pm[:, None], disc[:, None]

    j = np.arange(N + 1)
    V = np.maximum(w * (Sc * np.exp(sc * (2 * j - N)) - Kc), 0.0)
    kept = {}
    for i in range(N - 1, -1, -1):
        V = dc * (pc * V[:, 1:] + pmc * V[:, :-1])
        if american:
            Si = Sc * np.exp(sc * (2 * np.arange(i + 1) - i))
            V = np.maximum(V, w * (Si - Kc))
        if greeks and i <= 2:
            kept[i] = V
    price = V[:, 0]
    if not greeks:
        return _out(price.reshape(shape))

    f10, f11 = kept[1][:, 0], kept[1][:, 1]
    f20, f21, f22 = kept[2][:, 0], kept[2][:, 1], kept[2][:, 2]
    delta = (f11 - f10) / (S * u - S * d)
    Suu, Sdd = S * u * u, S * d * d
    gamma = ((f22 - f21) / (Suu - S) - (f21 - f20) / (S - Sdd)) / (0.5 * (Suu - Sdd))
    theta = (f21 - price) / (2.0 * dt)
    return {
        "price": _out(price.reshape(shape)),
        "delta": _out(delta.reshape(shape)),
        "gamma": _out(gamma.reshape(shape)),
        "theta": _out(theta.reshape(shape)),
    }


# ----------------------------------------------------------------------------
# Crank-Nicolson finite differences
# ----------------------------------------------------------------------------
def _psor(lo, di, up, rhs, g, x0, omega, tol, max_iter):
    """Projected SOR for A x >= rhs, x >= g, (A x - rhs)'(x - g) = 0, A tridiagonal.

    A has sub-diagonal lo, diagonal di and super-diagonal up (lo[0], up[-1]
    unused). Red-black ordering: the even nodes are relaxed together, then the
    odd nodes with the new even values (a consistently ordered SOR sweep, so it
    vectorises). Stops when the largest change in a sweep is <= tol.
    """
    n = x0.size
    xp = np.zeros(n + 2)
    xp[1:-1] = np.maximum(x0, g)
    groups = [np.arange(0, n, 2), np.arange(1, n, 2)]
    for _ in range(max_iter):
        change = 0.0
        for idx in groups:
            if idx.size == 0:
                continue
            gs = (rhs[idx] - lo[idx] * xp[idx] - up[idx] * xp[idx + 2]) / di[idx]
            old = xp[idx + 1]
            new = np.maximum(g[idx], old + omega * (gs - old))
            xp[idx + 1] = new
            change = max(change, float(np.max(np.abs(new - old))))
        if change <= tol:
            return xp[1:-1].copy()
    raise RuntimeError(f"PSOR did not converge in {max_iter} sweeps (last change {change:.3e})")


def _lcp_policy(lo, di, up, rhs, g, x0, max_iter=100):
    """Howard policy iteration for min(A x - rhs, x - g) = 0, A tridiagonal (lo, di, up).

    Each iteration solves the linear system whose row j is x_j = g_j where the
    exercise branch attains the min and (A x)_j = rhs_j elsewhere; it stops when
    that choice no longer changes (finite for an M-matrix). Used as the warm
    start of _psor, which then usually stops after one sweep.
    """
    n = rhs.size
    act = x0 <= g
    x = x0
    for _ in range(max_iter):
        ab = np.zeros((3, n))
        ab[0, 1:] = np.where(act, 0.0, up)[:-1]
        ab[1] = np.where(act, 1.0, di)
        ab[2, :-1] = np.where(act, 0.0, lo)[1:]
        x = solve_banded((1, 1), ab, np.where(act, g, rhs), check_finite=False)
        res = di * x - rhs
        res[1:] += lo[1:] * x[:-1]
        res[:-1] += up[:-1] * x[1:]
        new = (x - g) < res
        if np.array_equal(new, act):
            break
        act = new
    return x


def _fd_theta_grid(K, T, r, sigma, q, w, american, s_max, M, N, theta=0.5,
                   projection="psor", psor_tol=None, psor_max_iter=100000):
    """Theta-scheme (theta=0.5 Crank-Nicolson, 1 fully implicit) on S_j = j s_max/M.

    Returns (S grid, option values at t = 0). American: projection='psor'
    solves the LCP of each step by projected SOR warm-started from Howard
    policy iteration (the public method);
    'after_solve' solves the linear system and then takes the max with the
    payoff (Hull's textbook shortcut, kept to reproduce Hull's grid tables).
    Boundaries: V(0) = 0 (call) / K e^{-r tau}
    (European put) / K (American put); V(s_max) = s_max e^{-q tau} - K e^{-r tau}
    (European call; American: max with s_max - K) / 0 (put).
    """
    Sg = np.linspace(0.0, s_max, M + 1)
    j = np.arange(1, M, dtype=float)
    dt = T / N
    sig2 = sigma * sigma
    alpha = 0.5 * sig2 * j * j - 0.5 * (r - q) * j
    beta = -sig2 * j * j - r
    gamma = 0.5 * sig2 * j * j + 0.5 * (r - q) * j
    lo = -theta * dt * alpha
    di = 1.0 - theta * dt * beta
    up = -theta * dt * gamma
    ab = np.zeros((3, M - 1))
    ab[0, 1:] = up[:-1]
    ab[1, :] = di
    ab[2, :-1] = lo[1:]
    payoff = np.maximum(w * (Sg - K), 0.0)
    g = payoff[1:-1]
    if american:
        rows = (np.abs(lo) + np.abs(up)) / di
        mu = float(np.max(rows))
        omega = 2.0 / (1.0 + math.sqrt(1.0 - mu * mu)) if mu < 1.0 else 1.0
        omega = min(omega, 1.9)
        tol = psor_tol if psor_tol is not None else 1e-13 * max(K, 1.0)
    V = payoff.copy()
    ex = 1.0 - theta
    for n in range(1, N + 1):
        tau = n * dt
        if w > 0:
            v0 = 0.0
            vM = s_max * math.exp(-q * tau) - K * math.exp(-r * tau)
            if american:
                vM = max(vM, s_max - K)
        else:
            v0 = K if american else K * math.exp(-r * tau)
            vM = 0.0
        rhs = V[1:-1] + ex * dt * (alpha * V[:-2] + beta * V[1:-1] + gamma * V[2:])
        rhs[0] -= lo[0] * v0
        rhs[-1] -= up[-1] * vM
        if american and projection == "psor":
            x0 = _lcp_policy(lo, di, up, rhs, g, V[1:-1])
            x = _psor(lo, di, up, rhs, g, x0, omega, tol, psor_max_iter)
        else:
            x = solve_banded((1, 1), ab, rhs, check_finite=False)
            if american:
                x = np.maximum(x, g)
        V = np.concatenate(([v0], x, [vM]))
    return Sg, V


def fd_cn_price(S, K, T, r, sigma, *, kind, exercise, s_max, n_space, n_time, q=0.0):
    """Crank-Nicolson finite-difference price of a European or American option on a uniform S grid.

    Parameters
    ----------
    S : spot(s) to value, 0 <= S <= s_max (scalar or array; all spots are read
        off one grid when K, T, r, sigma, q are scalars).
    K, T : strike (> 0) and years to expiry (> 0).
    r, q : continuously compounded annual rate and dividend yield (q defaults to 0.0).
    sigma : annual volatility (decimal, > 0).
    kind : 'call' or 'put' (required).
    exercise : 'european' or 'american' (required).
    s_max : upper edge of the grid (required, > K; 3-4 times K is usual).
    n_space : number of S intervals M (required, >= 3); dS = s_max / M.
    n_time : number of time steps N (required, >= 1); dt = T / N.

    Returns
    -------
    Option value(s) at S. Grid values are interpolated at S by a not-a-knot
    cubic spline through the M+1 grid values (exact at grid nodes).

    Formula
    -------
    In time to expiry tau: V_tau = 0.5 sigma^2 S^2 V_SS + (r - q) S V_S - r V,
    central differences at S_j = j dS (j = 1..M-1), Crank-Nicolson in time:
    (I - dt/2 L) V^{n+1} = (I + dt/2 L) V^n, with
    L V_j = a_j V_{j-1} + b_j V_j + c_j V_{j+1}, a_j = (sigma^2 j^2 - (r-q) j)/2,
    b_j = -sigma^2 j^2 - r, c_j = (sigma^2 j^2 + (r-q) j)/2.
    Dirichlet boundaries: V(0) = 0 (call), K e^{-r tau} (European put), K
    (American put); V(s_max) = s_max e^{-q tau} - K e^{-r tau} (European call),
    max(that, s_max - K) (American call), 0 (put).
    American: each step solves the linear complementarity problem
    V >= payoff, A V >= rhs, (A V - rhs)(V - payoff) = 0 by projected SOR
    (red-black ordering, over-relaxation from the Jacobi bound, stop when the
    largest update is <= 1e-13 max(K, 1)), warm-started from a Howard
    policy-iteration solution of the same problem, so PSOR usually confirms
    convergence in one sweep even on fine grids. Errors are O(dS^2 + dt^2) for smooth
    data; pure CN (no implicit start-up steps) may show small oscillations near
    K when dt is much larger than dS^2 / (sigma S)^2.

    References
    ----------
    Crank and Nicolson (1947); Wilmott, Howison and Dewynne (1995) Ch. 8-9
    (Crank-Nicolson, projected SOR); Hull OFOD "Basic Numerical Procedures"
    (finite difference methods, American put grid example).

    Example
    -------
    >>> fd_cn_price(42, 40, 0.5, 0.10, 0.20, kind='call', exercise='european',
    ...             s_max=160, n_space=800, n_time=400)      # BSM (Hull Ex. 15.6): 4.7594
    4.7591...                                               # error O(dS^2): 3e-4 at dS = 0.2
    """
    w = _sign(kind)
    american = _american(exercise)
    M = _int_arg(n_space, "n_space", 3)
    N = _int_arg(n_time, "n_time", 1)
    s_max = float(s_max)
    S, K, T, r, sigma, q = map(_arr, (S, K, T, r, sigma, q))
    _check_pos(K=K, T=T, sigma=sigma)
    _check_finite(S=S, r=r, q=q)
    if not (math.isfinite(s_max) and np.all(s_max > K)):
        raise ValueError("s_max must be finite and > K")
    if np.any(S < 0) or np.any(S > s_max):
        raise ValueError("S must lie in [0, s_max]")
    shape = np.broadcast_shapes(S.shape, K.shape, T.shape, r.shape, sigma.shape, q.shape)
    if all(a.ndim == 0 for a in (K, T, r, sigma, q)):
        Sg, V = _fd_theta_grid(float(K), float(T), float(r), float(sigma), float(q), w, american,
                               s_max, M, N)
        vals = CubicSpline(Sg, V)(np.broadcast_to(S, shape))
        return _out(vals)
    _, (Sf, Kf, Tf, rf, sf, qf) = _broadcast_flat(S, K, T, r, sigma, q)
    vals = np.empty(Sf.size)
    for i in range(Sf.size):
        Sg, V = _fd_theta_grid(Kf[i], Tf[i], rf[i], sf[i], qf[i], w, american, s_max, M, N)
        vals[i] = CubicSpline(Sg, V)(Sf[i])
    return _out(vals.reshape(shape))


# ----------------------------------------------------------------------------
# Barone-Adesi-Whaley
# ----------------------------------------------------------------------------
def _m_over_k(r, sigma, T):
    """(2 r / sigma^2) / (1 - e^{-rT}), continuous at r = 0 (limit 2 / (sigma^2 T))."""
    x = r * T
    ratio = x / -math.expm1(-x) if x != 0.0 else 1.0
    return 2.0 / (sigma * sigma * T) * ratio


def _exercise_regime(w, r, q):
    """Shape of the early-exercise region of an American call (w = 1) or put (w = -1).

    In the money the discounted payoff e^{-rt} (w (S_t - K))^+ has drift
    e^{-rt} w (r K - q S_t). Early exercise is never optimal when that drift is
    >= 0 on the whole in-the-money region: call q <= 0 and q <= r, put r <= 0 and
    r <= q ('european'; Merton 1973 for r >= 0). Otherwise the region is one-sided
    (call S >= S*, put S <= S**: 'one_sided'), except when r < 0 and q < 0 with
    r < q (call) or q < r (put), where it is a band bounded on both sides ('band';
    Battauz, De Donno and Sbuelz 2015), which a single critical price cannot represent.
    """
    if w > 0:
        if q > 0.0:
            return "one_sided"
        if r >= q:
            return "european"
        return "one_sided" if q == 0.0 else "band"
    if r > 0.0:
        return "one_sided"
    if q >= r:
        return "european"
    return "one_sided" if r == 0.0 else "band"


_BAND_MSG = ("r < 0 and q < 0 with {cond}: the early-exercise region is a band bounded on both "
             "sides, outside this approximation; use crr_price or fd_cn_price")


def _baw_one(S, K, T, r, sigma, q, w):
    b = r - q
    regime = _exercise_regime(w, r, q)
    if regime == "european":        # early exercise is never optimal
        return float(bsm_price(S, K, T, r, sigma, kind="call" if w > 0 else "put", q=q))
    if regime == "band":
        raise ValueError(_BAND_MSG.format(cond="r < q (call)" if w > 0 else "q < r (put)"))
    s2 = sigma * sigma
    n = 2.0 * b / s2
    mk = _m_over_k(r, sigma, T)
    root = math.sqrt((n - 1.0) ** 2 + 4.0 * mk)
    carry = math.exp((b - r) * T)
    kind = "call" if w > 0 else "put"

    def euro(x):
        return float(bsm_price(x, K, T, r, sigma, kind=kind, q=q))

    def nd1(x):
        d1, _ = bsm_d1d2(x, K, T, r, sigma, q=q)
        return float(ndtr(w * d1))

    if w > 0:
        q2 = 0.5 * (-(n - 1.0) + root)

        def f(x):
            return x - K - euro(x) - (1.0 - carry * nd1(x)) * x / q2

        lo, hi = K, 2.0 * K
        while f(hi) < 0:
            hi *= 2.0
            if hi > 1e12 * K:
                raise RuntimeError("BAW: no critical price found for the call")
        s_crit = optimize.brentq(f, lo, hi, xtol=1e-14 * K, rtol=4 * np.finfo(float).eps, maxiter=500)
        if S >= s_crit:
            return S - K
        A2 = s_crit / q2 * (1.0 - carry * nd1(s_crit))
        return euro(S) + A2 * (S / s_crit) ** q2
    q1 = 0.5 * (-(n - 1.0) - root)

    def g(x):
        return K - x - euro(x) + (1.0 - carry * nd1(x)) * x / q1

    s_crit = optimize.brentq(g, 1e-12 * K, K, xtol=1e-14 * K, rtol=4 * np.finfo(float).eps, maxiter=500)
    if S <= s_crit:
        return K - S
    A1 = -s_crit / q1 * (1.0 - carry * nd1(s_crit))
    return euro(S) + A1 * (S / s_crit) ** q1


def baw_price(S, K, T, r, sigma, *, kind, q=0.0):
    """Barone-Adesi-Whaley (1987) quadratic approximation of an American call or put price.

    Parameters
    ----------
    S, K : spot and strike (> 0).
    T : years to expiry (> 0).
    r, q : continuously compounded annual rate and dividend yield (q defaults
        to 0.0); cost of carry b = r - q (options on futures: q = r, b = 0).
    sigma : annual volatility (decimal, > 0).
    kind : 'call' or 'put' (required).

    Returns
    -------
    Approximate American price. A call with q <= 0 and q <= r, and a put with
    r <= 0 and r <= q, are never exercised early, so their European BSM price is
    returned. Negative rates are supported where the exercise region is one-sided
    (a call with q > 0, or q = 0 and r < 0; a put with r > 0, or r = 0 and q < 0).
    When r < 0 and q < 0 with r < q (call) or q < r (put) the exercise region is a
    band bounded on both sides (Battauz, De Donno and Sbuelz 2015) and ValueError
    is raised: use crr_price or fd_cn_price there.

    Formula
    -------
    N = 2b/sigma^2, M = 2r/sigma^2, K(T) = 1 - e^{-rT}.
    Call: q2 = [-(N-1) + sqrt((N-1)^2 + 4M/K(T))]/2; the critical price S* solves
    S* - K = c(S*) + [1 - e^{(b-r)T} N(d1(S*))] S*/q2;
    C = c(S) + A2 (S/S*)^q2 with A2 = (S*/q2)[1 - e^{(b-r)T} N(d1(S*))] if S < S*,
    else S - K.
    Put: q1 = [-(N-1) - sqrt((N-1)^2 + 4M/K(T))]/2; S** solves
    K - S** = p(S**) - [1 - e^{(b-r)T} N(-d1(S**))] S**/q1;
    P = p(S) + A1 (S/S**)^q1 with A1 = -(S**/q1)[1 - e^{(b-r)T} N(-d1(S**))] if S > S**,
    else K - S. c, p, d1 are the BSM values with dividend yield q. The critical
    price is solved by Brent's method to about 1e-14 K (Haug uses a Newton
    iteration to 1e-6; the fourth printed decimal is unaffected).

    References
    ----------
    Barone-Adesi and Whaley (1987), J. Finance 42, tables; Haug (2007) Ch. 3;
    Battauz, De Donno and Sbuelz (2015), Management Science 61 (negative rates,
    double continuation region).

    Example
    -------
    >>> baw_price(100, 100, 0.1, 0.10, 0.15, kind='call', q=0.10)   # Haug's table prints 1.8771
    1.8769...
    """
    w = _sign(kind)
    S, K, T, r, sigma, q = map(_arr, (S, K, T, r, sigma, q))
    _check_pos(S=S, K=K, T=T, sigma=sigma)
    _check_finite(r=r, q=q)
    shape, flat = _broadcast_flat(S, K, T, r, sigma, q)
    vals = np.array([_baw_one(*(float(a[i]) for a in flat), w) for i in range(flat[0].size)])
    return _out(vals.reshape(shape))


# ----------------------------------------------------------------------------
# Bjerksund-Stensland 2002
# ----------------------------------------------------------------------------
def _bvn_cdf(h, k, rho):
    """Bivariate standard normal CDF P(X <= h, Y <= k), corr(X, Y) = rho, |rho| < 1.

    Owen (1956): Phi2 = [N(h) + N(k)]/2 - T(h, a_h) - T(k, a_k) - beta, with
    a_h = (k - rho h)/(h sqrt(1-rho^2)), a_k = (h - rho k)/(k sqrt(1-rho^2)),
    beta = 1/2 if h k < 0 or (h k = 0 and h + k < 0) else 0; T is Owen's T.
    h = k = 0 gives 1/4 + arcsin(rho)/(2 pi).
    """
    h, k, rho = np.broadcast_arrays(_arr(h), _arr(k), _arr(rho))
    sq = np.sqrt((1.0 - rho) * (1.0 + rho))
    with np.errstate(divide="ignore", invalid="ignore"):
        ah = (k - rho * h) / (h * sq)
        ak = (h - rho * k) / (k * sq)
    # h == 0 (k != 0): T(0, +-inf) = +-1/4 is the limit of the formula
    ah = np.where(h == 0, np.copysign(np.inf, k - rho * h), ah)
    ak = np.where(k == 0, np.copysign(np.inf, h - rho * k), ak)
    ah = np.where(np.isnan(ah), 0.0, ah)
    ak = np.where(np.isnan(ak), 0.0, ak)
    tail = np.where((h * k < 0) | ((h * k == 0) & (h + k < 0)), 0.5, 0.0)
    val = 0.5 * (ndtr(h) + ndtr(k)) - owens_t(h, ah) - owens_t(k, ak) - tail
    both0 = (h == 0) & (k == 0)
    val = np.where(both0, 0.25 + np.arcsin(rho) / (2.0 * np.pi), val)
    return np.clip(val, 0.0, 1.0)


def _bs_phi(S, T, gamma, H, I, r, b, sigma):
    """Bjerksund-Stensland phi: E[e^{-rT} S_T^gamma 1{S_T <= H} 1{no hit of I before T}]."""
    sT = sigma * np.sqrt(T)
    lam = (-r + gamma * b + 0.5 * gamma * (gamma - 1.0) * sigma * sigma) * T
    d = (np.log(S / H) + (b + (gamma - 0.5) * sigma * sigma) * T) / sT
    kappa = 2.0 * b / (sigma * sigma) + (2.0 * gamma - 1.0)
    return np.exp(lam) * S ** gamma * (ndtr(-d) - (I / S) ** kappa * ndtr(-d - 2.0 * np.log(I / S) / sT))


def _bs_psi(S, T2, gamma, H, I2, I1, t1, r, b, sigma):
    """Bjerksund-Stensland (2002) psi: two-period version of phi (barrier I2 on [0, t1], I1 on [t1, T2])."""
    s2 = sigma * sigma
    mu = b + (gamma - 0.5) * s2
    st1 = sigma * np.sqrt(t1)
    st2 = sigma * np.sqrt(T2)
    d1 = -(np.log(S / I1) + mu * t1) / st1
    d2 = -(np.log(I2 * I2 / (S * I1)) + mu * t1) / st1
    d3 = -(np.log(S / I1) - mu * t1) / st1
    d4 = -(np.log(I2 * I2 / (S * I1)) - mu * t1) / st1
    e1 = -(np.log(S / H) + mu * T2) / st2
    e2 = -(np.log(I2 * I2 / (S * H)) + mu * T2) / st2
    e3 = -(np.log(I1 * I1 / (S * H)) + mu * T2) / st2
    e4 = -(np.log(S * I1 * I1 / (H * I2 * I2)) + mu * T2) / st2
    rho = np.sqrt(t1 / T2)
    lam = -r + gamma * b + 0.5 * gamma * (gamma - 1.0) * s2
    kappa = 2.0 * b / s2 + (2.0 * gamma - 1.0)
    return np.exp(lam * T2) * S ** gamma * (
        _bvn_cdf(d1, e1, rho)
        - (I2 / S) ** kappa * _bvn_cdf(d2, e2, rho)
        - (I1 / S) ** kappa * _bvn_cdf(d3, e3, -rho)
        + (I1 / I2) ** kappa * _bvn_cdf(d4, e4, -rho)
    )


def _bs2002_call(S, K, T, r, b, sigma):
    """Bjerksund-Stensland 2002 American call, cost of carry b, arrays of equal shape, b < r."""
    s2 = sigma * sigma
    beta = (0.5 - b / s2) + np.sqrt((b / s2 - 0.5) ** 2 + 2.0 * r / s2)
    b_inf = beta / (beta - 1.0) * K
    with np.errstate(divide="ignore", invalid="ignore"):
        b0 = np.maximum(K, np.where(r - b != 0, r / (r - b) * K, K))
    t1 = 0.5 * (math.sqrt(5.0) - 1.0) * T
    den = (b_inf - b0) * b0
    h1 = -(b * t1 + 2.0 * sigma * np.sqrt(t1)) * K * K / den
    h2 = -(b * T + 2.0 * sigma * np.sqrt(T)) * K * K / den
    I1 = b0 + (b_inf - b0) * (1.0 - np.exp(h1))
    I2 = b0 + (b_inf - b0) * (1.0 - np.exp(h2))
    a1 = (I1 - K) * I1 ** (-beta)
    a2 = (I2 - K) * I2 ** (-beta)
    with np.errstate(all="ignore"):
        Se = np.minimum(S, I2)  # keep the formula finite where S >= I2 (result replaced below)
        val = (
            a2 * Se ** beta
            - a2 * _bs_phi(Se, t1, beta, I2, I2, r, b, sigma)
            + _bs_phi(Se, t1, 1.0, I2, I2, r, b, sigma)
            - _bs_phi(Se, t1, 1.0, I1, I2, r, b, sigma)
            - K * _bs_phi(Se, t1, 0.0, I2, I2, r, b, sigma)
            + K * _bs_phi(Se, t1, 0.0, I1, I2, r, b, sigma)
            + a1 * _bs_phi(Se, t1, beta, I1, I2, r, b, sigma)
            - a1 * _bs_psi(Se, T, beta, I1, I2, I1, t1, r, b, sigma)
            + _bs_psi(Se, T, 1.0, I1, I2, I1, t1, r, b, sigma)
            - _bs_psi(Se, T, 1.0, K, I2, I1, t1, r, b, sigma)
            - K * _bs_psi(Se, T, 0.0, I1, I2, I1, t1, r, b, sigma)
            + K * _bs_psi(Se, T, 0.0, K, I2, I1, t1, r, b, sigma)
        )
    return np.where(S >= I2, S - K, val)


def bs2002_price(S, K, T, r, sigma, *, kind, q=0.0):
    """Bjerksund-Stensland (2002) closed-form approximation of an American call or put price.

    Parameters
    ----------
    S, K : spot and strike (> 0).
    T : years to expiry (> 0).
    r, q : continuously compounded annual rate and dividend yield (q defaults
        to 0.0); cost of carry b = r - q (options on futures: q = r).
    sigma : annual volatility (decimal, > 0).
    kind : 'call' or 'put' (required).

    Returns
    -------
    Approximate American price, a lower bound on the true value (it prices
    the flat two-step exercise boundary below). A call with q <= 0 and q <= r,
    and a put with r <= 0 and r <= q, are never exercised early and return the
    European BSM price. Otherwise the formula needs r >= 0 and q > 0 (call), or
    q >= 0 and r > 0 (put): q > 0 keeps beta > 1 and B_inf finite, and
    B0 = max(K, r K/(r - b)) is the boundary at expiry only when r >= 0 (with
    r < 0 the formula can return values below intrinsic, even negative). Other
    cases that can be exercised early raise ValueError (use baw_price or
    crr_price); r < 0, q < 0 with r < q (call) or q < r (put) has a band-shaped
    exercise region (Battauz, De Donno and Sbuelz 2015).

    Formula
    -------
    Call, b < r: t1 = (sqrt 5 - 1) T / 2,
    beta = (1/2 - b/sigma^2) + sqrt((b/sigma^2 - 1/2)^2 + 2r/sigma^2),
    B_inf = beta K/(beta - 1), B0 = max(K, r K/(r - b)),
    h(t) = -(b t + 2 sigma sqrt t) K^2 / ((B_inf - B0) B0),
    I1 = B0 + (B_inf - B0)(1 - e^{h(t1)}) (boundary on [t1, T]),
    I2 = B0 + (B_inf - B0)(1 - e^{h(T)}) (boundary on [0, t1]),
    alpha_i = (I_i - K) I_i^{-beta}. If S >= I2 the value is S - K, otherwise
    C = alpha2 S^beta - alpha2 phi(S,t1,beta,I2,I2) + phi(S,t1,1,I2,I2) - phi(S,t1,1,I1,I2)
        - K phi(S,t1,0,I2,I2) + K phi(S,t1,0,I1,I2) + alpha1 phi(S,t1,beta,I1,I2)
        - alpha1 psi(S,T,beta,I1,I2,I1,t1) + psi(S,T,1,I1,I2,I1,t1) - psi(S,T,1,K,I2,I1,t1)
        - K psi(S,T,0,I1,I2,I1,t1) + K psi(S,T,0,K,I2,I1,t1),
    with phi and psi as in Haug (2007) Sec. 3.1.4 (psi uses the bivariate
    normal with correlation sqrt(t1/T), evaluated here through Owen's T).
    Put by the put-call transformation P(S, K, T, r, b) = C(K, S, T, r - b, -b),
    i.e. a call with spot K, strike S, rate q and dividend yield r.

    References
    ----------
    Bjerksund and Stensland (2002), "Closed form valuation of American
    options", NHH discussion paper; Haug (2007) Ch. 3; Battauz, De Donno and
    Sbuelz (2015), Management Science 61 (negative rates).

    Example
    -------
    >>> bs2002_price(100, 100, 0.5, 0.10, 0.25, kind='call', q=0.10)   # Haug (2007) table: 6.7661
    6.7661...
    """
    w = _sign(kind)
    S, K, T, r, sigma, q = map(_arr, (S, K, T, r, sigma, q))
    _check_pos(S=S, K=K, T=T, sigma=sigma)
    _check_finite(r=r, q=q)
    shape, (S, K, T, r, sigma, q) = _broadcast_flat(S, K, T, r, sigma, q)
    if w > 0:
        spot, strike, rate, carry = S, K, r, r - q
        euro = (q <= 0.0) & (q <= r)
    else:
        spot, strike, rate, carry = K, S, q, q - r
        euro = (r <= 0.0) & (r <= q)
    # The (transformed) call's formula needs a positive yield (beta > 1, finite B_inf) and a
    # non-negative rate (B0 = max(K, r K / (r - b)) is the boundary at expiry only for r >= 0).
    am = (carry < rate) & (rate >= 0.0)
    bad = ~(euro | am)
    if np.any(bad):
        if np.any(bad & (r < 0.0) & (q < 0.0)):
            raise ValueError(_BAND_MSG.format(cond="r < q (call)" if w > 0 else "q < r (put)"))
        raise ValueError("Bjerksund-Stensland needs r >= 0 and q > 0 for a call (q >= 0 and r > 0 "
                         "for a put) that can be exercised early; use baw_price or crr_price")
    out = np.empty(S.size)
    if np.any(euro):
        out[euro] = bsm_price(S[euro], K[euro], T[euro], r[euro], sigma[euro],
                              kind="call" if w > 0 else "put", q=q[euro])
    if np.any(am):
        out[am] = _bs2002_call(spot[am], strike[am], T[am], rate[am], carry[am], sigma[am])
    return _out(out.reshape(shape))


# ----------------------------------------------------------------------------
# GBM Monte Carlo
# ----------------------------------------------------------------------------
def _scalar(x, name, positive=False, nonneg=False):
    if isinstance(x, bool) or not isinstance(x, numbers.Real):
        raise TypeError(f"{name} must be a real number")
    v = float(x)
    if not math.isfinite(v):
        raise ValueError(f"{name} must be finite")
    if positive and v <= 0:
        raise ValueError(f"{name} must be > 0")
    if nonneg and v < 0:
        raise ValueError(f"{name} must be >= 0")
    return v


def gbm_paths(S, T, mu, sigma, *, n_paths, n_steps, seed, antithetic=False):
    """Exact lognormal geometric Brownian motion paths on a uniform time grid.

    Parameters
    ----------
    S : initial price (> 0).  T : horizon in years (> 0).
    mu : annual drift of dS/S (continuously compounded decimal); under the
        risk-neutral measure mu = r - q, under the real-world measure the
        expected return.
    sigma : annual volatility (decimal, >= 0).
    n_paths : number of paths (required, >= 1; even when antithetic=True).
    n_steps : number of time steps (required, >= 1); t_k = k T / n_steps.
    seed : required seed passed to numpy.random.default_rng (an int gives
        reproducible paths; None draws fresh entropy; a Generator is used as is).
    antithetic : False (default) or True. True draws n_paths/2 normal rows Z
        and uses [Z; -Z]: path i and path i + n_paths/2 are antithetic partners.

    Returns
    -------
    ndarray of shape (n_paths, n_steps + 1); column 0 is S and column k is
    S_{t_k}. Draw order: rng.standard_normal((rows, n_steps)), one row per path.

    Formula
    -------
    S_{t+dt} = S_t exp((mu - sigma^2/2) dt + sigma sqrt(dt) Z), Z ~ N(0, 1),
    dt = T / n_steps; exact in distribution for any dt (no discretisation bias).
    E[S_T] = S e^{mu T}, Var[S_T] = S^2 e^{2 mu T}(e^{sigma^2 T} - 1).

    References
    ----------
    Glasserman (2004) Ch. 3.2 (exact GBM simulation) and Ch. 4.2 (antithetic
    variates); Hull OFOD "Wiener Processes and Ito's Lemma" / "The BSM Model".

    Example
    -------
    >>> p = gbm_paths(100, 1.0, 0.05, 0.2, n_paths=4, n_steps=12, seed=7)
    >>> p.shape
    (4, 13)
    """
    S = _scalar(S, "S", positive=True)
    T = _scalar(T, "T", positive=True)
    mu = _scalar(mu, "mu")
    sigma = _scalar(sigma, "sigma", nonneg=True)
    n_paths = _int_arg(n_paths, "n_paths", 1)
    n_steps = _int_arg(n_steps, "n_steps", 1)
    if antithetic not in (True, False):
        raise TypeError("antithetic must be True or False")
    if antithetic and n_paths % 2:
        raise ValueError("n_paths must be even when antithetic=True")
    rng = np.random.default_rng(seed)
    rows = n_paths // 2 if antithetic else n_paths
    Z = rng.standard_normal((rows, n_steps))
    if antithetic:
        Z = np.concatenate([Z, -Z], axis=0)
    dt = T / n_steps
    incr = (mu - 0.5 * sigma * sigma) * dt + sigma * math.sqrt(dt) * Z
    logp = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(incr, axis=1)], axis=1)
    return S * np.exp(logp)


_ESTIMATORS = ("plain", "antithetic", "control_variate")


def gbm_mc_price(S, T, r, sigma, *, payoff, n_paths, n_steps, seed, estimator, q=0.0, control=None):
    """Risk-neutral Monte Carlo price of a path-dependent payoff on GBM: plain, antithetic or control variate.

    Parameters
    ----------
    S, T, r, sigma, q : spot (> 0), years (> 0), continuously compounded rate,
        volatility and dividend yield (q defaults to 0.0). Paths are drawn by
        gbm_paths with mu = r - q.
    payoff : required callable, payoff(paths) -> array of n_paths payoffs paid
        at T (undiscounted); paths has shape (n_paths, n_steps + 1), column 0 = S,
        column -1 = S_T. Example: lambda p: np.maximum(p[:, -1] - K, 0).
    n_paths : number of paths (required; even for 'antithetic').
    n_steps : time steps per path (required; 1 is enough for a European payoff).
    seed : required, passed to numpy.random.default_rng (see gbm_paths).
    estimator : required, one of
        'plain'           Y_i = e^{-rT} payoff_i, se = sd(Y) / sqrt(n)  (sd with ddof=1);
        'antithetic'      pairs (Z, -Z): Y_i = (Y(Z_i) + Y(-Z_i))/2 over n/2 pairs,
                          se = sd(pair means) / sqrt(n/2);
        'control_variate' Y_i - b (X_i - E[X]) with b = cov(Y, X)/var(X)
                          estimated from the same paths, se = sd(Y - b X) / sqrt(n).
    control : only for 'control_variate'. None (default) uses X = e^{-rT} S_T
        with E[X] = S e^{-qT}; otherwise a pair (func, mean) where func(paths)
        returns the n_paths control values and mean is their exact expectation.

    Returns
    -------
    dict with 'price' (float), 'se' (standard error), 'ci95' (tuple lo, hi =
    price -+ 1.959964 se), 'n_paths', and 'beta' (the control-variate
    coefficient b; nan for the other estimators).

    Formula
    -------
    price = mean of the (paired / controlled) discounted payoffs; the 95%
    interval is the normal-approximation interval price +- z_0.975 se.

    References
    ----------
    Boyle (1977); Glasserman (2004) Ch. 4.1 (control variates) and 4.2
    (antithetic variates); Hull OFOD "Basic Numerical Procedures" (Monte Carlo).

    Example
    -------
    >>> res = gbm_mc_price(42, 0.5, 0.10, 0.20, payoff=lambda p: np.maximum(p[:, -1] - 40, 0),
    ...                    n_paths=200_000, n_steps=1, seed=1, estimator='control_variate')
    >>> res['price'], res['se']                       # BSM (Hull Ex. 15.6): 4.7594
    (4.7565..., 0.0030...)
    """
    S = _scalar(S, "S", positive=True)
    T = _scalar(T, "T", positive=True)
    r = _scalar(r, "r")
    q = _scalar(q, "q")
    sigma = _scalar(sigma, "sigma", nonneg=True)
    if not callable(payoff):
        raise TypeError("payoff must be a callable payoff(paths) -> n_paths values")
    est = str(estimator).strip().lower() if isinstance(estimator, str) else None
    if est not in _ESTIMATORS:
        raise ValueError(f"estimator must be one of {_ESTIMATORS}, got {estimator!r}")
    if control is not None and est != "control_variate":
        raise ValueError("control is only used with estimator='control_variate'")
    n_paths = _int_arg(n_paths, "n_paths", 2)
    paths = gbm_paths(S, T, r - q, sigma, n_paths=n_paths, n_steps=n_steps, seed=seed,
                      antithetic=(est == "antithetic"))
    disc = math.exp(-r * T)
    pay = np.asarray(payoff(paths), dtype=float).reshape(-1)
    if pay.size != n_paths:
        raise ValueError(f"payoff returned {pay.size} values for {n_paths} paths")
    Y = disc * pay
    beta = float("nan")
    if est == "plain":
        sample = Y
    elif est == "antithetic":
        h = n_paths // 2
        sample = 0.5 * (Y[:h] + Y[h:])
    else:
        if control is None:
            X = disc * paths[:, -1]
            mean_x = S * math.exp(-q * T)
        else:
            try:
                func, mean_x = control
            except (TypeError, ValueError):
                raise TypeError("control must be None or a pair (func, mean)") from None
            X = np.asarray(func(paths), dtype=float).reshape(-1)
            mean_x = float(mean_x)
            if X.size != n_paths:
                raise ValueError(f"control returned {X.size} values for {n_paths} paths")
        xc = X - X.mean()
        vx = float(np.dot(xc, xc))
        beta = float(np.dot(xc, Y - Y.mean()) / vx) if vx > 0 else 0.0
        sample = Y - beta * (X - mean_x)
    m = sample.size
    price = float(sample.mean())
    se = float(sample.std(ddof=1) / math.sqrt(m)) if m > 1 else float("nan")
    return {
        "price": price,
        "se": se,
        "ci95": (price - _Z975 * se, price + _Z975 * se),
        "n_paths": n_paths,
        "beta": beta,
    }
