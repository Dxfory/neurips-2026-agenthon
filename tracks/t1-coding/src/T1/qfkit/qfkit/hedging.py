"""Delta hedging: a discrete Delta-Hedging cash ledger and escrowed-dividend Black-Scholes.

Conventions for the whole module: prices in currency units; times in YEARS; r
is an annual rate as a decimal (0.05, not 5); sigma is the annual volatility as
a decimal; transaction costs are in BASIS POINTS of the traded notional
|delta shares| * S (10 bps = 0.001). ``kind`` is ONE string, 'call' or 'put'.
A price path is indexed by integer steps t = 0..N (closes; on an ex-dividend
step the close is already the ex-dividend price). Every rule on which hedging
exercises differ (the order of interest, dividend and rebalance within a step,
whose shares earn a dividend, the rebalance schedule, costs on each side,
rebalancing at expiry, the time-to-expiry clock) is a REQUIRED keyword: there is
no hidden default.

Functions
---------
delta_hedge_pnl        cash ledger of an option position hedged in the underlying on a price path
bsm_escrowed_dividend  Black-Scholes price and delta with known cash dividends (escrowed model)

References: Hull, Options, Futures, and Other Derivatives (OFOD), "The Greek
Letters" (delta hedging, the weekly-rebalancing simulation tables) and "The
Black-Scholes-Merton Model" (known cash dividends); Haug, The Complete Guide to
Option Pricing Formulas, 2nd ed. (2007), Ch. 1 and Ch. 9 (escrowed dividend).
"""

from __future__ import annotations

import math
import numbers

import numpy as np
from scipy.special import ndtr

from .bsm import bsm_price

__all__ = [
    "delta_hedge_pnl",
    "bsm_escrowed_dividend",
]

_STEP_NAMES = ("interest", "dividend", "rebalance")
_LEDGER_COLUMNS = ("step", "S", "tau", "hedge_ratio", "trade_shares", "trade_value", "tc", "dividend",
                   "interest", "option_cash", "shares", "cash", "option_value", "book_value")


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


def _real(x, name, *, nonneg=False, positive=False):
    if isinstance(x, bool) or not isinstance(x, numbers.Real):
        raise TypeError(f"{name} must be a real number, got {x!r}")
    v = float(x)
    if not math.isfinite(v):
        raise ValueError(f"{name} must be finite")
    if positive and v <= 0:
        raise ValueError(f"{name} must be > 0")
    if nonneg and v < 0:
        raise ValueError(f"{name} must be >= 0")
    return v


def _step(x, name):
    if isinstance(x, bool) or not isinstance(x, numbers.Integral):
        raise TypeError(f"{name} must be an integer step, got {x!r}")
    return int(x)


def _dividend_pairs(dividends, name):
    """[(when, amount)] from a mapping {when: amount} or an iterable of pairs (or an (n, 2) array)."""
    if dividends is None:
        raise TypeError(f"{name} is required: pass [] (or {{}}) when there are none")
    items = dividends.items() if isinstance(dividends, dict) else dividends
    pairs = []
    for item in items:
        try:
            when, amount = item
        except (TypeError, ValueError):
            raise TypeError(f"{name} must be a mapping {{when: amount}} or pairs (when, amount)") from None
        pairs.append((when, amount))
    return pairs


# ----------------------------------------------------------------------------
# escrowed-dividend Black-Scholes
# ----------------------------------------------------------------------------
def bsm_escrowed_dividend(S, K, t_rem, r, sigma, *, kind, dividends):
    """Escrowed-Dividend Black-Scholes price and delta of a European option on a stock paying known cash dividends (Hull; Haug).

    Parameters
    ----------
    S : observed (cum-dividend) spot price(s), > 0.
    K : strike (> 0).
    t_rem : time to expiry in years (>= 0); t_rem == 0 returns the intrinsic value.
    r : continuously compounded annual rate (decimal), used both to discount
        the dividends and in the Black-Scholes formula.
    sigma : annual volatility of the escrowed (ex-dividend) stock price (decimal, > 0).
    kind : 'call' or 'put' (required).
    dividends : required sequence of pairs (tau_i, D_i): tau_i is the time in
        YEARS from now to the i-th ex-dividend date and D_i the cash amount per
        share (a mapping {tau_i: D_i} or an (n, 2) array also works). Pass []
        when there are none. A dividend counts iff 0 < tau_i <= t_rem: one with
        tau_i <= 0 has gone ex already (S is quoted ex that dividend), one after
        expiry does not affect the option.

    Returns
    -------
    dict with
    'price'   the option price;
    'delta'   dV/dS with respect to the OBSERVED spot: N(d1) (call) or N(d1) - 1
              (put), d1 evaluated at S_eff; the dividend PV is a deterministic
              offset, so dS_eff/dS = 1. At t_rem == 0: 1{S > K} (call), -1{S < K} (put);
    'S_eff'   S - pv_divs;
    'pv_divs' sum of D_i e^{-r tau_i} over the counted dividends.
    Each value is a float, or an array when S, K, t_rem, r or sigma is an array.

    Formula
    -------
    pv = sum_{0 < tau_i <= t_rem} D_i e^{-r tau_i};  S_eff = S - pv (must be > 0);
    price = BSM(S_eff, K, t_rem, r, sigma) with no dividend yield:
    C = S_eff N(d1) - K e^{-r t_rem} N(d2),  P = K e^{-r t_rem} N(-d2) - S_eff N(-d1),
    d1 = [ln(S_eff/K) + (r + sigma^2/2) t_rem] / (sigma sqrt(t_rem)), d2 = d1 - sigma sqrt(t_rem).

    References
    ----------
    Hull OFOD "The Black-Scholes-Merton Model" (European options on stocks paying
    known cash dividends: subtract the PV of the dividends paid during the life
    of the option); Haug (2007) Ch. 1 and Ch. 9 (escrowed dividend model).

    Example
    -------
    >>> res = bsm_escrowed_dividend(40, 40, 0.5, 0.09, 0.30, kind='call',
    ...                             dividends=[(2/12, 0.5), (5/12, 0.5)])   # Hull: 3.67
    >>> round(float(res['price']), 4), round(float(res['S_eff']), 4)
    (3.6712, 39.0258)
    """
    w = _sign(kind)
    S, K, t_rem, r, sigma = (np.asarray(a, dtype=float) for a in (S, K, t_rem, r, sigma))
    for name, v in (("S", S), ("K", K), ("sigma", sigma)):
        if not np.all(np.isfinite(v)) or np.any(v <= 0):
            raise ValueError(f"{name} must be finite and > 0")
    if not np.all(np.isfinite(t_rem)) or np.any(t_rem < 0):
        raise ValueError("t_rem must be finite and >= 0")
    if not np.all(np.isfinite(r)):
        raise ValueError("r must be finite")
    pv = np.zeros(np.broadcast_shapes(t_rem.shape, r.shape))
    for tau_i, d_i in _dividend_pairs(dividends, "dividends"):
        tau_i = _real(tau_i, "dividend time")
        d_i = _real(d_i, "dividend amount", nonneg=True)
        pv = pv + np.where((tau_i > 0.0) & (tau_i <= t_rem), d_i * np.exp(-r * tau_i), 0.0)
    s_eff = S - pv
    if np.any(s_eff <= 0):
        raise ValueError("S minus the PV of the remaining dividends must be > 0")
    kind_s = "call" if w > 0 else "put"
    price = bsm_price(s_eff, K, t_rem, r, sigma, kind=kind_s)
    s_eff_b, K_b, t_b, r_b, sig_b = np.broadcast_arrays(s_eff, K, t_rem, r, sigma)
    live = t_b > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        st = sig_b * np.sqrt(t_b)
        d1 = (np.log(s_eff_b / K_b) + (r_b + 0.5 * sig_b * sig_b) * t_b) / st
    expiry = np.where(w * (s_eff_b - K_b) > 0, w, 0.0)
    delta = np.where(live, ndtr(d1) - (0.0 if w > 0 else 1.0), expiry)
    shape = np.broadcast_shapes(S.shape, K.shape, t_rem.shape, r.shape, sigma.shape)
    return {
        "price": _out(np.broadcast_to(price, shape)),
        "delta": _out(np.broadcast_to(delta, shape)),
        "S_eff": _out(np.broadcast_to(s_eff, shape)),
        "pv_divs": _out(np.broadcast_to(pv, shape)),
    }


# ----------------------------------------------------------------------------
# delta-hedge cash ledger
# ----------------------------------------------------------------------------
def _per_step(source, name, n_steps, n_paths, tau, S):
    """Turn a callable f(t, S_t, tau_t) or an array of per-step values into a getter t -> (n_paths,)."""
    if callable(source):
        def get(t):
            v = np.asarray(source(t, S[:, t] if n_paths > 1 else S[0, t], tau[t]), dtype=float)
            v = np.broadcast_to(v, (n_paths,)) if v.ndim == 0 or v.shape == (n_paths,) else None
            if v is None:
                raise ValueError(f"{name}(t, S_t, tau_t) must return a scalar or one value per path")
            if not np.all(np.isfinite(v)):
                raise ValueError(f"{name} returned a non-finite value at step {t}")
            return v
        return get
    arr = np.asarray(source, dtype=float)
    if arr.ndim == 1 and arr.shape == (n_steps + 1,):
        arr = np.broadcast_to(arr, (n_paths, n_steps + 1))
    if arr.shape != (n_paths, n_steps + 1):
        raise ValueError(f"{name} must be a callable f(t, S_t, tau_t) or an array of one value per step "
                         f"(shape ({n_steps + 1},) or ({n_paths}, {n_steps + 1}))")

    def get(t):
        v = arr[:, t]
        if not np.all(np.isfinite(v)):
            raise ValueError(f"{name} has a non-finite value at step {t}")
        return v
    return get


def delta_hedge_pnl(S_path, *, option_value, hedge_ratio, payoff, position, rebalance_every, dt, r,
                    step_order, interest_comp, dividends, dividend_timing, tc_buy_bps, tc_sell_bps,
                    rebalance_at_expiry, t_remaining):
    """Delta-Hedging simulation (Hull): the cash ledger and P&L of a Delta-Hedge of an option position along a price path.

    Parameters
    ----------
    S_path : prices at steps t = 0..N (N >= 1), shape (N+1,) for one path or
        (n_paths, N+1) for many (Monte Carlo: all paths run at once, without a ledger).
        On an ex-dividend step the price is the ex-dividend (post-drop) close.
    option_value : per-option value used for the premium at step 0 and the
        ledger's mark-to-market column; a callable option_value(t, S_t, tau_t)
        (S_t a float for one path, an array of n_paths values for many) or an
        array of one value per step (shape (N+1,) or (n_paths, N+1)). Called
        at step 0 and, for one path, at steps 1..N-1 (never at N: the payoff is used).
    hedge_ratio : per-option dV/dS (the delta); a callable hedge_ratio(t, S_t,
        tau_t) or an array of one value per step. The hedge holds
        -position * hedge_ratio shares after each rebalance. Called at step 0,
        at every rebalance step, and at step N (with tau_N, usually 0) only
        when rebalance_at_expiry is True.
    payoff : callable payoff(S_N) -> per-option payoff at expiry (vectorised
        for many paths), e.g. lambda s: np.maximum(s - K, 0.0).
    position : signed number of options held: -1 = short (written) one
        option, +100 = long 100 options. A short call is hedged by buying shares.
    rebalance_every : int k >= 1: rebalance at the steps t with t % k == 0 and
        0 < t < N; or an explicit collection of steps in 1..N-1. The initial
        trade (t = 0) and the liquidation (t = N) are always made.
    dt : years per step, used for interest (1/252 for trading days, 1/52 weekly).
    r : annual rate on the cash account (decimal), for positive and negative balances alike.
    step_order : the order of the three actions inside each step t = 1..N, a
        permutation of ('interest', 'dividend', 'rebalance'). The usual one is
        ('interest', 'dividend', 'rebalance'): the cash carried from t-1 earns
        interest first, then the dividend is credited, then the hedge is traded.
    interest_comp : 'continuous' (cash *= e^{r dt}) or 'simple' (cash *= 1 + r dt) per step.
    dividends : cash dividends per share as a mapping {step: amount} or pairs
        (step, amount), steps in 1..N; pass [] when there are none. The amount is
        credited (or, for a short stock position, debited) on that step.
    dividend_timing : whose shares earn a dividend: 'prior' (the shares
        carried from the close of step t-1, i.e. the holder of record before
        the ex-date) or 'current' (the shares held when the dividend action
        runs in step_order; differs only when 'rebalance' comes before 'dividend').
    tc_buy_bps, tc_sell_bps : proportional costs in basis points of |traded
        shares| * S_t for purchases and for sales, charged on every trade: the
        initial hedge, each rebalance and the final liquidation.
    rebalance_at_expiry : True rebalances at step N (to hedge_ratio(N, S_N,
        tau_N)) before the liquidation and counts it in n_rebalances; False
        (the usual rule) does not rebalance at expiry.
    t_remaining : the clock passed to option_value and hedge_ratio: a callable
        t -> years to expiry, or an array of N+1 values; e.g. lambda t: (N - t) / 252.

    Returns
    -------
    dict with
    'pnl'             final cash after the liquidation of the shares (and its
                      cost) and the option settlement position * payoff(S_N);
                      it includes the premium received (or paid) at step 0 and
                      all interest, so it is the realised P&L of the hedged book
                      at expiry (positive = the book made money);
    'total_tc'        transaction costs paid, initial trade and liquidation included;
    'total_dividends' dividend cash credited to the hedge (negative if short stock);
    'total_interest'  interest credited (negative when the cash balance is negative);
    'n_rebalances'    rebalance actions at steps 1..N-1 (plus the expiry one when
                      rebalance_at_expiry is True); never the initial trade or the
                      liquidation; a scheduled rebalance counts even if it trades 0 shares;
    'premium'         option cash at step 0, -position * option_value(0, S_0, tau_0);
    'ledger'          one path: a pandas DataFrame, one row per step 0..N
                      (None for many paths). Columns: step, S,
                      tau, hedge_ratio (nan where not rebalanced), trade_shares,
                      trade_value (cash paid for shares, signed), tc, dividend,
                      interest, option_cash (premium at 0, settlement at N),
                      shares and cash after the step, option_value (per option;
                      payoff at N) and book_value = cash + shares*S + position*option_value,
                      the mark-to-market P&L of the hedged book (at N, after the
                      liquidation and settlement, book_value = cash = pnl).
    pnl, total_tc, total_dividends and total_interest are floats for one path
    and arrays of n_paths values for many.

    Formula
    -------
    Step 0: shares = -position * h_0; cash = -position * V_0 - shares * S_0 - tc(shares).
    Step t = 1..N, in step_order: interest: cash *= g (g = e^{r dt} or 1 + r dt);
    dividend: cash += D_t * shares_base; rebalance (scheduled t, or t = N if
    rebalance_at_expiry): trade = -position * h_t - shares, cash -= trade * S_t + tc(trade),
    shares += trade. Then at N: cash += shares * S_N - tc(-shares), shares = 0,
    cash += position * payoff(S_N). tc(q) = bps/1e4 * |q| * S_t, with the buy
    rate for q > 0 and the sell rate for q < 0.
    Self-financing: with order interest, dividend, rebalance, the book
    B_t = cash_t + shares_t S_t satisfies
    B_t = g B_{t-1} + shares_{t-1} (S_t + D_t - g S_{t-1}) - tc_t.

    References
    ----------
    Hull OFOD "The Greek Letters" (delta hedging; the weekly-rebalancing
    simulation of 100,000 written calls, S = 49, K = 50, r = 5%, sigma = 20%,
    20 weeks, whose hedging cost is $263,300 in the table where the option
    closes in the money); Wilmott, Paul Wilmott on Quantitative Finance, Ch. 10
    (hedging error of discrete rebalancing).

    Example
    -------
    >>> import numpy as np
    >>> from qfkit import bsm_greeks
    >>> K, r, sig, N = 50.0, 0.05, 0.20, 20
    >>> path = 49.0 * np.exp(np.linspace(0.0, 0.1, N + 1))
    >>> res = delta_hedge_pnl(path, position=-100_000, payoff=lambda s: max(s - K, 0.0),
    ...     option_value=lambda t, s, tau: bsm_greeks(s, K, tau, r, sig, kind='call')['price'],
    ...     hedge_ratio=lambda t, s, tau: bsm_greeks(s, K, tau, r, sig, kind='call')['delta'],
    ...     t_remaining=lambda t: (N - t) / 52, rebalance_every=1, dt=1 / 52, r=r,
    ...     step_order=('interest', 'dividend', 'rebalance'), interest_comp='continuous',
    ...     dividends=[], dividend_timing='prior', tc_buy_bps=0.0, tc_sell_bps=0.0,
    ...     rebalance_at_expiry=False)
    >>> res['n_rebalances'], sorted(res)
    (19, ['ledger', 'n_rebalances', 'pnl', 'premium', 'total_dividends', 'total_interest', 'total_tc'])
    """
    S = np.asarray(S_path, dtype=float)
    single = S.ndim == 1
    if S.ndim not in (1, 2):
        raise ValueError("S_path must have shape (N+1,) or (n_paths, N+1)")
    S = S.reshape(1, -1) if single else S
    n_paths, n1 = S.shape
    N = n1 - 1
    if N < 1 or n_paths < 1:
        raise ValueError("S_path needs at least two steps (t = 0 and t = N >= 1)")
    if not np.all(np.isfinite(S)) or np.any(S <= 0):
        raise ValueError("S_path must be finite and > 0")
    if not callable(payoff):
        raise TypeError("payoff must be a callable payoff(S_N) -> per-option payoff")
    position = _real(position, "position")
    if position == 0:
        raise ValueError("position must be non-zero (signed number of options; -1 = short one option)")
    dt = _real(dt, "dt", positive=True)
    r = _real(r, "r")
    tc_buy = _real(tc_buy_bps, "tc_buy_bps", nonneg=True) / 1e4
    tc_sell = _real(tc_sell_bps, "tc_sell_bps", nonneg=True) / 1e4
    if rebalance_at_expiry not in (True, False):
        raise TypeError("rebalance_at_expiry must be True or False")
    comp = _choice(interest_comp, "interest_comp", ("continuous", "simple"))
    growth = math.exp(r * dt) if comp == "continuous" else 1.0 + r * dt
    timing = _choice(dividend_timing, "dividend_timing", ("prior", "current"))
    if isinstance(step_order, str) or not hasattr(step_order, "__iter__"):
        raise TypeError("step_order must be a sequence such as ('interest', 'dividend', 'rebalance')")
    order = tuple(str(s).strip().lower() for s in step_order)
    if sorted(order) != sorted(_STEP_NAMES):
        raise ValueError(f"step_order must be a permutation of {_STEP_NAMES}, got {tuple(step_order)!r}")

    if isinstance(rebalance_every, numbers.Integral) and not isinstance(rebalance_every, bool):
        k = int(rebalance_every)
        if k < 1:
            raise ValueError("rebalance_every must be >= 1")
        rebal = {t for t in range(1, N) if t % k == 0}
    else:
        try:
            steps = [_step(t, "rebalance step") for t in rebalance_every]
        except TypeError:
            raise TypeError("rebalance_every must be an int k or a collection of steps in 1..N-1") from None
        if any(t < 1 or t > N - 1 for t in steps):
            raise ValueError("explicit rebalance steps must lie in 1..N-1 (use rebalance_at_expiry for step N)")
        rebal = set(steps)

    divs = {}
    for t, amount in _dividend_pairs(dividends, "dividends"):
        t = _step(t, "dividend step")
        if t < 1 or t > N:
            raise ValueError(f"dividend steps must lie in 1..N = 1..{N}, got {t}")
        divs[t] = divs.get(t, 0.0) + _real(amount, "dividend amount")

    if callable(t_remaining):
        tau = np.array([_real(t_remaining(t), "t_remaining(t)") for t in range(N + 1)])
    else:
        tau = np.asarray(t_remaining, dtype=float)
        if tau.shape != (N + 1,) or not np.all(np.isfinite(tau)):
            raise ValueError(f"t_remaining must be a callable t -> years or {N + 1} finite values")
    get_value = _per_step(option_value, "option_value", N, n_paths, tau, S)
    get_ratio = _per_step(hedge_ratio, "hedge_ratio", N, n_paths, tau, S)
    x = -position  # shares per unit of hedge ratio

    # the per-step ledger is kept for one path only; many paths keep running totals
    cols = {c: np.zeros(N + 1) for c in _LEDGER_COLUMNS} if single else None
    if single:
        cols["step"][:] = np.arange(N + 1)
        cols["S"][:] = S[0]
        cols["tau"][:] = tau
        cols["hedge_ratio"][:] = np.nan
    totals = {"tc": np.zeros(n_paths), "dividend": np.zeros(n_paths), "interest": np.zeros(n_paths)}

    def record(name, t, value, add=False):
        if single:
            if add:
                cols[name][t] += value[0]
            else:
                cols[name][t] = value[0]

    def trade_to(target, t, shares, cash):
        q = target - shares
        cost = np.where(q > 0, tc_buy, tc_sell) * np.abs(q) * S[:, t]
        totals["tc"] += cost
        record("trade_shares", t, q, add=True)
        record("trade_value", t, q * S[:, t], add=True)
        record("tc", t, cost, add=True)
        return target, cash - q * S[:, t] - cost

    v0 = get_value(0)
    premium = -position * v0
    h0 = get_ratio(0)
    record("hedge_ratio", 0, h0)
    record("option_cash", 0, premium)
    shares, cash = trade_to(x * h0, 0, np.zeros(n_paths), premium.copy())
    record("shares", 0, shares)
    record("cash", 0, cash)
    record("option_value", 0, v0)
    n_rebalances = 0
    for t in range(1, N + 1):
        prior = shares
        for action in order:
            if action == "interest":
                interest = cash * (growth - 1.0)
                cash = cash + interest
                totals["interest"] += interest
                record("interest", t, interest)
            elif action == "dividend":
                if t in divs:
                    base = prior if timing == "prior" else shares
                    paid = divs[t] * base
                    cash = cash + paid
                    totals["dividend"] += paid
                    record("dividend", t, paid)
            elif t in rebal or (t == N and rebalance_at_expiry):
                h = get_ratio(t)
                record("hedge_ratio", t, h)
                shares, cash = trade_to(x * h, t, shares, cash)
                n_rebalances += 1
        if t < N:
            if single:
                record("option_value", t, get_value(t))
        else:
            shares, cash = trade_to(np.zeros(n_paths), t, shares, cash)
            pay = np.asarray(payoff(S[0, N] if single else S[:, N]), dtype=float)
            if pay.shape not in ((), (n_paths,)):
                raise ValueError("payoff must return a scalar or one value per path")
            pay = np.broadcast_to(pay, (n_paths,))
            if not np.all(np.isfinite(pay)):
                raise ValueError("payoff returned a non-finite value")
            settle = position * pay
            cash = cash + settle
            record("option_cash", t, settle)
            record("option_value", t, pay)
        record("shares", t, shares)
        record("cash", t, cash)

    def fin(a):
        return float(a[0]) if single else a

    ledger = None
    if single:
        import pandas as pd
        cols["book_value"] = cols["cash"] + cols["shares"] * cols["S"] + position * cols["option_value"]
        cols["book_value"][N] = cols["cash"][N]
        ledger = pd.DataFrame({c: cols[c] for c in _LEDGER_COLUMNS})
        ledger["step"] = ledger["step"].astype(int)
    return {
        "pnl": fin(cash),
        "total_tc": fin(totals["tc"]),
        "total_dividends": fin(totals["dividend"]),
        "total_interest": fin(totals["interest"]),
        "n_rebalances": n_rebalances,
        "premium": fin(premium),
        "ledger": ledger,
    }
