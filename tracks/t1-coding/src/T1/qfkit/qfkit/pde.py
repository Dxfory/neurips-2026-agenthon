"""Finite-difference PDE engine: Crank-Nicolson / implicit / explicit grids, American exercise, Richardson.

Conventions for the whole module (the same as ``qfkit.bsm``): S is spot, K the
strike, T the time to expiry in YEARS and every time is calendar time in years
from today; r and q are continuously compounded annual rates as decimals; sigma
is the annual volatility as a decimal (0.20, not 20). ``kind`` is ONE string,
'call' or 'put'; ``exercise`` is 'european' or 'american'. The grid is uniform
in S on [0, s_max] with n_space intervals and uniform in time with n_time
steps. Every numerical choice (scheme, grid, boundary condition, American
method and tolerances, dividend interpolation) is a REQUIRED keyword: there is
no hidden default grid or method.

Functions
---------
fd_solve    theta-scheme solver of the Black-Scholes PDE: price, grid delta/gamma/theta, grid, exercise boundary
richardson  Richardson extrapolation of a fine and a coarse grid result

References: Wilmott, Paul Wilmott on Quantitative Finance, 2nd ed. (2006),
Ch. 77-78 (finite-difference methods, explicit and Crank-Nicolson code, early
exercise); Crank and Nicolson (1947); Brennan and Schwartz (1977), J. Finance
32; Cryer (1971), SIAM J. Control 9 (projected SOR); Rannacher (1984); Giles
and Carter (2006), J. Comput. Finance 9 (Rannacher start-up); Richardson (1911).
"""

from __future__ import annotations

import math
import numbers

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.linalg import solve_banded

__all__ = [
    "fd_solve",
    "richardson",
]

_SCHEMES = {"cn": 0.5, "implicit": 1.0, "explicit": 0.0}
_PSOR_MAX_SWEEPS = 100_000


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _out(x):
    """0-d result -> numpy float64 scalar; otherwise the ndarray itself."""
    x = np.asarray(x, dtype=float)
    return x[()] if x.ndim == 0 else x


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


def _choice(value, name, allowed):
    if not isinstance(value, str) or value.strip().lower() not in allowed:
        raise ValueError(f"{name} must be one of {', '.join(repr(a) for a in allowed)}, got {value!r}")
    return value.strip().lower()


def _int_arg(value, name, minimum):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be an integer, got {value!r}")
    v = int(value)
    if v < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {v}")
    return v


def _real(x, name, *, positive=False, nonneg=False):
    a = np.asarray(x, dtype=float) if not isinstance(x, bool) else None
    if a is None or a.ndim != 0:
        raise TypeError(f"{name} must be a real scalar, got {x!r}")
    v = float(a)
    if not math.isfinite(v):
        raise ValueError(f"{name} must be finite")
    if positive and v <= 0:
        raise ValueError(f"{name} must be > 0")
    if nonneg and v < 0:
        raise ValueError(f"{name} must be >= 0")
    return v


def _psor(lo, di, up, rhs, g, x0, omega, tol):
    """Projected SOR (Cryer 1971) for A x >= rhs, x >= g, (A x - rhs)'(x - g) = 0, A tridiagonal.

    Row i: lo[i] x[i-1] + di[i] x[i] + up[i] x[i+1]; lo[0] and up[-1] are
    ignored (their boundary terms are in rhs). Red-black ordering: the even
    rows are relaxed together, then the odd rows with the new even values (a
    consistently ordered SOR sweep, so each half-sweep is one vector
    operation). Starts from max(x0, g); stops when the largest absolute change
    in a sweep is <= tol.
    """
    n = rhs.size
    xp = np.zeros(n + 2)
    xp[1:-1] = np.maximum(x0, g)
    lo = lo.copy()
    up = up.copy()
    lo[0] = 0.0
    up[-1] = 0.0
    ev, od = slice(0, n, 2), slice(1, n, 2)
    halves = (
        (rhs[ev], lo[ev], up[ev], di[ev], g[ev], slice(1, n + 1, 2), slice(0, n, 2), slice(2, n + 2, 2)),
        (rhs[od], lo[od], up[od], di[od], g[od], slice(2, n + 1, 2), slice(1, n, 2), slice(3, n + 2, 2)),
    )
    change = math.inf
    for _ in range(_PSOR_MAX_SWEEPS):
        change = 0.0
        for b, a_lo, a_up, a_di, gg, centre, left, right in halves:
            if gg.size == 0:
                continue
            old = xp[centre]
            gs = (b - a_lo * xp[left] - a_up * xp[right]) / a_di
            new = np.maximum(gg, old + omega * (gs - old))
            change = max(change, float(np.max(np.abs(new - old))))
            xp[centre] = new
        if change <= tol:
            return xp[1:-1].copy()
    raise RuntimeError(f"PSOR did not converge in {_PSOR_MAX_SWEEPS} sweeps (last change {change:.3e}); "
                       "check omega (0 < omega < 2) and tol")


class _BrennanSchwartz:
    """Brennan-Schwartz (1977) direct solver of the LCP for an exercise region at the LOW-index end.

    The system (lo, di, up) is reduced to lower bidiagonal form by eliminating
    the super-diagonal from the last row upwards (factorisation kept for every
    time step); the projected substitution V_i = max(g_i, (b'_i - lo_i V_{i-1})/d'_i)
    then runs from row 0 upwards. Vectorised: the substitution is taken as one
    exercise block followed by one linear block (exact when the projection
    never binds after the first continuation row, the vanilla case); otherwise
    the plain sequential loop is used.
    """

    def __init__(self, lo, di, up):
        n = di.size
        dp = np.empty(n)
        m = np.zeros(n)
        dp[-1] = di[-1]
        for i in range(n - 2, -1, -1):
            m[i] = up[i] / dp[i + 1]
            dp[i] = di[i] - m[i] * lo[i + 1]
        self.n = n
        self.m = m
        self.dp = dp
        e = lo / dp
        e[0] = 0.0
        self.e = e
        ub = np.zeros((2, n))
        ub[0, 1:] = m[:-1]
        ub[1] = 1.0
        self.ub = ub

    def solve(self, rhs, g):
        n = self.n
        bp = solve_banded((0, 1), self.ub, rhs, check_finite=False)
        c = bp / self.dp
        e = self.e
        cand = c.copy()
        cand[1:] -= e[1:] * g[:-1]
        cont = np.flatnonzero(cand > g)
        if cont.size == 0:
            return g.copy()
        k = int(cont[0])
        x = g.copy()
        x[k] = cand[k]
        if k + 1 < n:
            sub = c[k + 1:].copy()
            sub[0] -= e[k + 1] * x[k]
            lb = np.zeros((2, n - k - 1))
            lb[0] = 1.0
            lb[1, :-1] = e[k + 2:]
            x[k + 1:] = solve_banded((1, 0), lb, sub, check_finite=False)
            if np.any(x[k + 1:] < g[k + 1:]):
                x = np.empty(n)
                prev = 0.0
                cl, el, gl = c.tolist(), e.tolist(), g.tolist()
                for i in range(n):
                    v = cl[i] - el[i] * prev
                    prev = v if v > gl[i] else gl[i]
                    x[i] = prev
        return x


def _boundary_value(w, american, side, tau, t_now, K, r, q, s_max, divs):
    """Dirichlet values: V(0) = 0 (call) | K e^{-r tau} (European put) | K (American put);
    V(s_max) = s_max e^{-q tau} - PV(cash dividends in (t, T]) - K e^{-r tau} (call; American: max with
    s_max - K) | 0 (put)."""
    if side == 0:
        if w > 0:
            return 0.0
        return K if american else K * math.exp(-r * tau)
    if w < 0:
        return 0.0
    pv = sum(d * math.exp(-r * (td - t_now)) for td, d in divs if td > t_now + 1e-12)
    v = s_max * math.exp(-q * tau) - pv - K * math.exp(-r * tau)
    return max(v, s_max - K) if american else v


def fd_solve(S, K, T, r, sigma, *, kind, exercise, scheme, s_max, n_space, n_time, boundary, american_method,
             tol, omega, exercise_tol, return_grid, q=0.0, dividends=None, dividend_interp=None,
             rannacher_steps=0):
    """Finite-Difference solver of the Black-Scholes PDE (Crank-Nicolson, implicit or explicit; Wilmott) with American exercise by PSOR (Cryer), Brennan-Schwartz or projection. API contract: dividend_interp 'linear'|'cubic' required when a dividend is applied.

    Parameters
    ----------
    S : spot(s) at which to report price and Greeks, 0 <= S <= s_max (scalar or array).
    K, T : strike (> 0) and years to expiry (> 0), scalars.
    r, q : continuously compounded annual rate and dividend yield (q defaults to 0.0).
    sigma : annual volatility (decimal, > 0).
    kind : 'call' or 'put' (required).
    exercise : 'european' or 'american' (required).
    scheme : 'cn' (Crank-Nicolson, theta = 1/2), 'implicit' (fully implicit,
        theta = 1) or 'explicit' (theta = 0; raises ValueError unless
        dt (sigma^2 (M-1)^2 + r) <= 1, the positivity/stability condition).
    s_max : upper edge of the S grid (> K); the nodes are S_i = i s_max / n_space.
    n_space : number of S intervals M (>= 3). n_time : number of time steps N (>= 1), dt = T / N.
    boundary : 'dirichlet' (discounted asymptotic values): V(0) = 0 (call),
        K e^{-r tau} (European put), K (American put); V(s_max) = s_max e^{-q tau} - PV(remaining cash
        dividends) - K e^{-r tau} (call; American: max with s_max - K), 0 (put).
        'linear': V_SS = 0 at s_max (V_M = 2 V_{M-1} - V_{M-2}) and the PDE
        itself at S = 0 (dV/dtau = -r V there), as in Wilmott's code.
    american_method : for exercise='american' (ignored for 'european': pass None):
        'psor' projected SOR on each time step's linear complementarity problem,
        warm-started from the previous time level (Cryer 1971);
        'brennan_schwartz' the direct Brennan-Schwartz (1977) elimination with
        projection during back-substitution (exact for one exercise region at
        one end of the grid: puts low, calls high);
        'projection' solve the linear system, then V = max(V, payoff)
        (the explicit-projection shortcut; O(dt) off the LCP solution).
        With scheme='explicit' all three are the same projection.
    tol : PSOR stopping rule, the largest absolute change in a sweep (price
        units, e.g. 1e-8); None unless american_method='psor'.
    omega : PSOR over-relaxation factor, 0 < omega < 2 (e.g. 1.2-1.6); None unless 'psor'.
    exercise_tol : American only (None for European): a node is in the
        exercise region when payoff > 0 and V - payoff <= exercise_tol.
    return_grid : True also returns the whole value grid.
    dividends : None or pairs (t_d, D) of discrete cash dividends, t_d in years
        from today (> 0; dividends after T are ignored). The jump condition
        V(S, t_d-) = V(max(S - D, 0), t_d+) is applied at the time node nearest
        t_d (never t = 0), then (American) V = max(V, payoff).
    dividend_interp : 'linear' or 'cubic' (not-a-knot spline) interpolation of
        V(., t_d+) at S - D; required when at least one dividend is applied.
    rannacher_steps : m >= 0 (scheme 'cn' only): the first m time steps from
        expiry are each taken as two fully implicit half-steps (Rannacher 1984;
        Giles-Carter 2006 use m = 2), damping the oscillations CN shows near
        the payoff kink; the grid rows stay at t_n = n dt.

    Returns
    -------
    dict with
    'price', 'delta', 'gamma', 'theta' at S (floats, or arrays for an array S):
        node values V_i, central differences (V_{i+1} - V_{i-1}) / (2 dS) and
        (V_{i+1} - 2 V_i + V_{i-1}) / dS^2 on the t = 0 row (one-sided at the
        edges), read at S by a not-a-knot cubic spline through the node values
        (exact at a node); theta = (V(S, t_1) - V(S, t_0)) / dt, dV/dt PER YEAR
        in calendar time (negative for a long vanilla; divide by 365 for a day);
    's' the S nodes (M+1); 't' the time nodes t_n = n dt (N+1);
    'exercise_boundary' (American; None for European) an array of N+1 values
        aligned with 't': the largest exercise node S_i (put; 0.0 if none) or
        the smallest (call; nan if none), exercise as defined by exercise_tol;
        at a dividend node it is that of the cum-dividend (t_d-) values;
    'grid' (N+1, M+1) array with grid[n, i] = V(S_i, t_n) (row N = payoff;
        a dividend node holds the cum-dividend values) when return_grid=True, else None;
    'dividend_times' the grid times at which the dividend jumps were applied.

    Formula
    -------
    In time to expiry tau: V_tau = L V = 0.5 sigma^2 S^2 V_SS + (r - q) S V_S - r V,
    central differences at S_j = j dS: L V_j = a_j V_{j-1} + b_j V_j + c_j V_{j+1},
    a_j = (sigma^2 j^2 - (r-q) j)/2, b_j = -sigma^2 j^2 - r, c_j = (sigma^2 j^2 + (r-q) j)/2.
    Theta scheme: (I - theta dt L) V^{n+1} = (I + (1 - theta) dt L) V^n.
    American: V^{n+1} >= payoff, (I - theta dt L) V^{n+1} >= rhs, with equality in one of the two.
    CN is O(dS^2 + dt^2), implicit and explicit O(dS^2 + dt).

    References
    ----------
    Wilmott, Paul Wilmott on Quantitative Finance (2006) Ch. 77-78; Crank and
    Nicolson (1947); Brennan and Schwartz (1977); Cryer (1971); Rannacher (1984);
    Giles and Carter (2006); Hull OFOD "Basic Numerical Procedures" (finite
    differences; cash dividends as a jump in S).

    Example
    -------
    >>> res = fd_solve(100, 100, 1.0, 0.05, 0.2, kind='put', exercise='american', scheme='cn',
    ...                s_max=400, n_space=400, n_time=400, boundary='dirichlet',
    ...                american_method='psor', tol=1e-8, omega=1.5, exercise_tol=1e-6,
    ...                return_grid=False, rannacher_steps=2)
    >>> round(float(res['price']), 4)   # CRR, 40,000 steps: 6.0904 (grid error O(dS^2 + dt^2))
    6.0874
    """
    w = _sign(kind)
    american = _choice(exercise, "exercise", ("european", "american")) == "american"
    theta = _SCHEMES[_choice(scheme, "scheme", tuple(_SCHEMES))]
    bc = _choice(boundary, "boundary", ("dirichlet", "linear"))
    M = _int_arg(n_space, "n_space", 3)
    N = _int_arg(n_time, "n_time", 1)
    K = _real(K, "K", positive=True)
    T = _real(T, "T", positive=True)
    r = _real(r, "r")
    q = _real(q, "q")
    sigma = _real(sigma, "sigma", positive=True)
    s_max = _real(s_max, "s_max", positive=True)
    if s_max <= K:
        raise ValueError("s_max must be > K")
    if return_grid not in (True, False):
        raise TypeError("return_grid must be True or False")
    m_ran = _int_arg(rannacher_steps, "rannacher_steps", 0)
    if m_ran and theta != 0.5:
        raise ValueError("rannacher_steps applies to scheme='cn' only")
    m_ran = min(m_ran, N)
    S_arr = np.asarray(S, dtype=float)
    if not np.all(np.isfinite(S_arr)) or np.any(S_arr < 0) or np.any(S_arr > s_max):
        raise ValueError("S must lie in [0, s_max]")
    method = None
    if american:
        method = _choice(american_method, "american_method", ("psor", "brennan_schwartz", "projection"))
        exercise_tol = _real(exercise_tol, "exercise_tol", nonneg=True)
        if theta == 0.0:
            method = "projection"
        if method == "psor":
            tol = _real(tol, "tol", positive=True)
            omega = _real(omega, "omega")
            if not 0.0 < omega < 2.0:
                raise ValueError("omega must lie in (0, 2)")

    dt = T / N
    divs = []
    if dividends is not None:
        items = dividends.items() if isinstance(dividends, dict) else dividends
        for item in items:
            try:
                td, amount = item
            except (TypeError, ValueError):
                raise TypeError("dividends must be pairs (t_d, D)") from None
            td = _real(td, "dividend time")
            amount = _real(amount, "dividend amount", nonneg=True)
            if td <= 0:
                raise ValueError("dividend times must be > 0 (years from today)")
            if td <= T + 1e-12:
                divs.append((max(1, min(N, int(round(td / dt)))), amount))
    interp = None
    if divs:
        interp = _choice(dividend_interp, "dividend_interp", ("linear", "cubic"))
    div_rows = {}
    for n_d, amount in divs:
        div_rows.setdefault(n_d, []).append(amount)
    div_times = [(n_d * dt, amount) for n_d, amount in divs]

    s = np.linspace(0.0, s_max, M + 1)
    dS = s_max / M
    j = np.arange(M + 1, dtype=float)
    sig2 = sigma * sigma
    a = 0.5 * sig2 * j * j - 0.5 * (r - q) * j
    b = -sig2 * j * j - r
    c = 0.5 * sig2 * j * j + 0.5 * (r - q) * j
    if bc == "dirichlet":
        lo_i, hi_i = 1, M            # unknown rows 1..M-1
    else:
        lo_i, hi_i = 0, M            # unknown rows 0..M-1; V_M by linear extrapolation
    La, Lb, Lc = a[lo_i:hi_i].copy(), b[lo_i:hi_i].copy(), c[lo_i:hi_i].copy()
    if bc == "linear":
        La[-1] -= Lc[-1]
        Lb[-1] += 2.0 * Lc[-1]
        Lc[-1] = 0.0
    if theta == 0.0:
        worst = dt * (sig2 * (M - 1) ** 2 + r)
        if worst > 1.0:
            need = math.ceil(T * (sig2 * (M - 1) ** 2 + r))
            raise ValueError(f"explicit scheme unstable: dt (sigma^2 (M-1)^2 + r) = {worst:.3g} > 1; "
                             f"use n_time >= {need}, fewer S steps, or scheme='implicit'/'cn'")
    payoff = np.maximum(w * (s - K), 0.0)
    g = payoff[lo_i:hi_i]

    systems = {}

    def system(th, h):
        key = (th, h)
        if key not in systems:
            lo, di, up = -th * h * La, 1.0 - th * h * Lb, -th * h * Lc
            ab = np.zeros((3, di.size))
            ab[0, 1:] = up[:-1]
            ab[1] = di
            ab[2, :-1] = lo[1:]
            bs = None
            if american and method == "brennan_schwartz":
                bs = (_BrennanSchwartz(lo, di, up) if w < 0
                      else _BrennanSchwartz(up[::-1].copy(), di[::-1].copy(), lo[::-1].copy()))
            systems[key] = (lo, di, up, ab, bs)
        return systems[key]

    def step(V, th, h, tau_new, t_new):
        """One theta-scheme step of size h from V (at tau_new - h) to tau_new."""
        Vr = V[lo_i:hi_i]
        Vm = V[lo_i - 1:hi_i - 1] if lo_i else np.concatenate(([0.0], V[:hi_i - 1]))
        Vp = V[lo_i + 1:hi_i + 1]
        rhs = Vr + (1.0 - th) * h * (La * Vm + Lb * Vr + Lc * Vp)
        new = np.empty(M + 1)
        if bc == "dirichlet":
            new[0] = _boundary_value(w, american, 0, tau_new, t_new, K, r, q, s_max, div_times)
            new[M] = _boundary_value(w, american, M, tau_new, t_new, K, r, q, s_max, div_times)
        if th == 0.0:
            x = rhs
            if american:
                x = np.maximum(x, g)
        else:
            lo, di, up, ab, bs = system(th, h)
            if bc == "dirichlet":
                rhs = rhs.copy()
                rhs[0] -= lo[0] * new[0]
                rhs[-1] -= up[-1] * new[M]
            if not american or method == "projection":
                x = solve_banded((1, 1), ab, rhs, check_finite=False)
                if american:
                    x = np.maximum(x, g)
            elif method == "psor":
                x = _psor(lo, di, up, rhs, g, Vr, omega, tol)
            elif w < 0:
                x = bs.solve(rhs, g)
            else:
                x = bs.solve(rhs[::-1].copy(), g[::-1].copy())[::-1]
        new[lo_i:hi_i] = x
        if bc == "linear":
            new[M] = 2.0 * new[M - 1] - new[M - 2]
            if american:
                new[M] = max(new[M], payoff[M])
        return new

    def jump(V, row):
        for amount in div_rows.get(row, ()):
            x = np.maximum(s - amount, 0.0)
            V = np.interp(x, s, V) if interp == "linear" else CubicSpline(s, V)(x)
            if american:
                V = np.maximum(V, payoff)
        return V

    def boundary_of(V):
        ex = (payoff > 0) & (V - payoff <= exercise_tol)
        idx = np.flatnonzero(ex)
        if w < 0:
            return float(s[idx[-1]]) if idx.size else 0.0
        return float(s[idx[0]]) if idx.size else math.nan

    grid = np.empty((N + 1, M + 1)) if return_grid else None
    ex_bound = np.empty(N + 1) if american else None
    V = jump(payoff.copy(), N)
    if return_grid:
        grid[N] = V
    if american:
        ex_bound[N] = boundary_of(V)
    row1 = V if N == 1 else None
    for n in range(N - 1, -1, -1):       # calendar row n, tau = (N - n) dt
        k = N - n                         # step number from expiry
        tau_new, t_new = k * dt, n * dt
        if k <= m_ran:
            V = step(V, 1.0, 0.5 * dt, tau_new - 0.5 * dt, t_new + 0.5 * dt)
            V = step(V, 1.0, 0.5 * dt, tau_new, t_new)
        else:
            V = step(V, theta, dt, tau_new, t_new)
        if n > 0:
            V = jump(V, n)
        if return_grid:
            grid[n] = V
        if american:
            ex_bound[n] = boundary_of(V)
        if n == 1:
            row1 = V

    V0 = V
    delta_n = np.empty(M + 1)
    gamma_n = np.empty(M + 1)
    delta_n[1:-1] = (V0[2:] - V0[:-2]) / (2.0 * dS)
    delta_n[0] = (V0[1] - V0[0]) / dS
    delta_n[-1] = (V0[-1] - V0[-2]) / dS
    gamma_n[1:-1] = (V0[2:] - 2.0 * V0[1:-1] + V0[:-2]) / (dS * dS)
    gamma_n[0], gamma_n[-1] = gamma_n[1], gamma_n[-2]

    def at(values):
        return _out(CubicSpline(s, values)(S_arr))

    return {
        "price": at(V0),
        "delta": at(delta_n),
        "gamma": at(gamma_n),
        "theta": _out((CubicSpline(s, row1)(S_arr) - CubicSpline(s, V0)(S_arr)) / dt),
        "s": s,
        "t": np.arange(N + 1) * dt,
        "exercise_boundary": ex_bound,
        "grid": grid,
        "dividend_times": [n_d * dt for n_d, _ in divs],
    }


def richardson(fine, coarse, *, order, ratio=2):
    """Richardson extrapolation of a numerical result from a fine and a coarse grid.

    Parameters
    ----------
    fine, coarse : the results on the fine grid (step h) and on the coarse grid
        (step ratio * h); scalars or arrays that broadcast.
    order : the order p of the leading error term, error ~ C h^p (required;
        2 for Crank-Nicolson refined in S and t together, 1 for an implicit
        scheme refined in t, ...).
    ratio : the refinement ratio between the two grids (default 2: the coarse
        grid has twice the step of the fine one in every refined direction).

    Returns
    -------
    The extrapolated value (float, or an array for array input).

    Formula
    -------
    R = fine + (fine - coarse) / (ratio^p - 1) = (ratio^p fine - coarse) / (ratio^p - 1);
    for ratio 2, p 2: (4 fine - coarse) / 3. Then |coarse - R| / |fine - R| = ratio^p.

    References
    ----------
    Richardson (1911); Wilmott, Paul Wilmott on Quantitative Finance, Ch. 77
    (extrapolating finite-difference prices); Hull OFOD "Basic Numerical Procedures".

    Example
    -------
    >>> float(richardson(10.4477, 10.4398, order=2))      # (4*10.4477 - 10.4398)/3
    10.45033...
    """
    p = _real(order, "order", positive=True)
    k = _real(ratio, "ratio")
    if k <= 1:
        raise ValueError("ratio must be > 1")
    fine = np.asarray(fine, dtype=float)
    coarse = np.asarray(coarse, dtype=float)
    f = k ** p
    return _out((f * fine - coarse) / (f - 1.0))
