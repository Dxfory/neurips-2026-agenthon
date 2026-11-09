"""Backtest ledgers: a long-only share ledger driven by entry and exit signals, and a portfolio-weights ledger.

Conventions for the whole module:
- Time runs over the rows of the inputs, oldest first, one row per period (a trading day, a
  month, ...). A signal or a target weight is formed at the CLOSE of its row. When it trades
  and which return it earns are REQUIRED keywords, never defaults.
- Prices are positive and finite where they are used. Returns are per-period SIMPLE returns in
  decimals (0.01 = 1%).
- Fees: ``fee_fixed`` currency units per fill plus ``fee_pct`` (a decimal, 0.001 = 0.1%) of
  the traded value, charged on buys and on sells. Costs on weights are in basis points of
  portfolio value (``cost_bps`` = 10 means 0.10% per unit of |dw|).
- Inputs are aligned by POSITION (the same number of rows); results are pandas objects
  indexed by the input dates.

Functions
---------
signal_trade_ledger  long-only single-asset share ledger: next-open, next-close or same-close
                     fills, fixed plus percentage fees, sizing rules, trailing stop, forced close
weights_ledger       portfolio-weights ledger: execution lag, rebalance schedule, drift between
                     rebalances, turnover |dw| and its costs

References: double-entry accounting (equity = cash + shares x mark price at every close);
Grinold and Kahn, Active Portfolio Management, 2nd ed. (2000), Ch. 16 (turnover and
transaction costs).
"""

from __future__ import annotations

import math
import numbers
import warnings

import numpy as np
import pandas as pd

__all__ = [
    "signal_trade_ledger",
    "weights_ledger",
]

_EXECUTIONS = ("next_open", "next_close", "same_close")
_FORCED = ("last_close", "none")
_SIZING = ("all_cash_floor", "all_cash")
_PEAK_INIT = ("signal_close", "exec_price")
_PRIORITY = ("signal", "stop")
_SCHEDULES = ("every_period", "target_dates", "month_start", "month_end", "quarter_start", "quarter_end",
              "year_start", "year_end")
_COST_BASES = ("abs_dw", "half_abs_dw", "abs_dw_ex_initial", "half_abs_dw_ex_initial")
_COST_TIMINGS = ("effective", "decision")


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _choice(value, name, allowed):
    if not isinstance(value, str) or value.strip().lower() not in allowed:
        raise ValueError(f"{name} must be one of {', '.join(repr(a) for a in allowed)}, got {value!r}")
    return value.strip().lower()


def _flag(value, name):
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be True or False, got {value!r}")
    return bool(value)


def _nonneg(value, name):
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise TypeError(f"{name} must be a number") from None
    if not math.isfinite(v) or v < 0:
        raise ValueError(f"{name} must be finite and >= 0")
    return v


def _vector(x, n, name):
    """1-D float array of length n (positional alignment); a pandas object gives its values."""
    if isinstance(x, (pd.Series, pd.Index)):
        a = x.to_numpy(dtype=float, na_value=np.nan)
    else:
        a = np.asarray(x, dtype=float)
    a = a.ravel()
    if a.size != n:
        raise ValueError(f"{name} has {a.size} values; dates has {n}")
    return a


def _signal(x, n, name):
    """Boolean signal of length n; NaN / None / pd.NA count as False."""
    if isinstance(x, (pd.Series, pd.Index)):
        a = x.to_numpy(dtype=float, na_value=np.nan)
    else:
        a = pd.Series(np.asarray(x, dtype=object).ravel()).to_numpy(dtype=float, na_value=np.nan)
    if a.size != n:
        raise ValueError(f"{name} has {a.size} values; dates has {n}")
    return np.where(np.isnan(a), 0.0, a) != 0.0


# ----------------------------------------------------------------------------
# signal-driven share ledger
# ----------------------------------------------------------------------------
def signal_trade_ledger(dates, mark_price, exec_price, *, entry, exit, execution, sizing, fee_fixed, fee_pct,
                        initial_cash, forced_close, block_while_pending, stop=None, exit_priority=None):
    """Signal backtest share ledger: long-only, one position at a time, fixed + % fees, trailing stop.

    A single asset is bought on an entry signal and sold on an exit signal or a stop. Signals are
    read at the CLOSE of a row and filled by the ``execution`` rule. The ledger keeps cash and
    whole (or fractional) shares and marks the position at ``mark_price`` every close.

    Parameters
    ----------
    dates : sequence of T row labels (dates), oldest first.
    mark_price : T prices used to value the position at each close, to fill 'next_close' and
        'same_close' orders, for the stops and for the forced close (e.g. adj_close).
    exec_price : T prices used to fill 'next_open' orders (e.g. adj_open); may be None for the
        other two execution rules. Only the rows where a fill happens must be finite and > 0.
    entry, exit : T booleans (NaN = False) evaluated at the close of each row. ``entry`` is read
        only while flat, ``exit`` only while long. Compute crossings yourself
        (e.g. fast[t] > slow[t] and fast[t-1] <= slow[t-1]).
    execution : REQUIRED. When an order signalled at the close of row t fills:
        'next_open'  at exec_price[t+1] (signal at close, trade at the next open);
        'next_close' at mark_price[t+1];
        'same_close' at mark_price[t].
        A signal on the last row with no next row is not filled.
    sizing : REQUIRED. Shares bought at a fill price p with cash C:
        'all_cash_floor' floor((C - fee_fixed) / (p (1 + fee_pct))), the most whole shares
                         whose cost including fees fits in cash (floor(C/p) without fees);
        'all_cash'       the same without the floor (fractional shares);
        a number         that fixed number of shares (cash may go negative);
        a callable       f(ctx) -> shares, ctx a dict with cash, equity (= cash: the ledger is
                         flat at an entry fill), price (the fill price), signal_index,
                         exec_index, signal_date, exec_date, signal_close (mark at the signal
                         row), max_affordable (the 'all_cash_floor' count), fee_fixed, fee_pct.
                         Risk-based sizing: min(floor(risk_cash / stop_distance), max_affordable).
        A result <= 0 skips the entry (listed in skipped_entries).
    fee_fixed : REQUIRED. Flat fee per fill in currency units (0 for none).
    fee_pct : REQUIRED. Proportional fee as a decimal of traded value (0.001 = 0.1%; 0 for none).
        Buy cost = n p (1 + fee_pct) + fee_fixed; sell proceeds = n p (1 - fee_pct) - fee_fixed.
    initial_cash : REQUIRED. Starting cash (> 0).
    forced_close : REQUIRED. 'last_close' sells an open position at the last row's mark_price
        (with fees; exit_reason 'forced_close'; a pending exit on the last row also ends here);
        'none' leaves it open (its trade row has NaN exit fields and exit_reason 'open').
    block_while_pending : REQUIRED. True: on a row whose close fires an exit, no entry is read
        (the position is still open until the exit fills), so a new buy can come at the
        earliest from the next row's close. False: on that row the entry rule is also read and
        a new buy fills right after the sell, at the same price. In both cases entries are read
        again on the row where the sell filled.
    stop : None, or a dict with any of
        'trailing_pct' (decimal) with 'peak_init' 'signal_close' (mark at the entry signal row)
            or 'exec_price' (the entry fill price): the peak is raised to mark_price at every
            close while long, and the stop fires when mark_price[t] < peak (1 - trailing_pct);
        'loss_pct' (decimal): fires when mark_price[t] < entry price (1 - loss_pct).
        A stop fires at a close and fills by ``execution`` like an exit signal.
    exit_priority : 'signal' or 'stop' (required when stop is given): the exit_reason when the
        exit signal and a stop fire on the same close.

    Returns
    -------
    dict with
    - trades: DataFrame, one row per round trip: trade_num (1, 2, ...), entry_signal_date,
      entry_date, entry_price, shares, buy_cost, exit_signal_date, exit_date, exit_price,
      sell_proceeds, pnl (= sell_proceeds - buy_cost), ret (= pnl / buy_cost), exit_reason
      ('signal', 'stop', 'forced_close' or 'open'), entry_index, exit_index (row positions).
    - fills: DataFrame, one row per fill: trade_num, side ('BUY'/'SELL'), signal_date,
      exec_date, price, shares, fees, cash_after, pnl (SELL rows only), reason.
    - equity: Series, cash + shares * mark_price at every close (after that row's fills).
    - returns: Series, equity.pct_change() with the first value 0.
    - cash, shares: Series at every close.
    - skipped_entries: list of signal dates whose sizing gave no shares.

    Formula
    -------
    equity_t = cash_t + shares_t * mark_t. A round trip's pnl is n p_exit (1 - f) - F -
    (n p_entry (1 + f) + F). When flat at the end, sum(pnl) = equity_T - initial_cash.

    References
    ----------
    Double-entry accounting of a cash and share position; Grinold and Kahn (2000) Ch. 16 for
    proportional costs.

    Example
    -------
    >>> px = [10, 11, 12, 11, 13]
    >>> out = signal_trade_ledger(range(5), px, px, entry=[1, 0, 0, 0, 0], exit=[0, 0, 1, 0, 0],
    ...     execution='next_open', sizing='all_cash_floor', fee_fixed=0, fee_pct=0, initial_cash=100,
    ...     forced_close='last_close', block_while_pending=True)
    >>> out['trades'][['trade_num', 'entry_price', 'shares', 'exit_price', 'pnl']].values.tolist()
    [[1.0, 11.0, 9.0, 11.0, 0.0]]
    """
    idx = pd.Index(dates)
    n = len(idx)
    if n < 1:
        raise ValueError("dates is empty")
    execution = _choice(execution, "execution", _EXECUTIONS)
    forced_close = _choice(forced_close, "forced_close", _FORCED)
    block = _flag(block_while_pending, "block_while_pending")
    F = _nonneg(fee_fixed, "fee_fixed")
    f = _nonneg(fee_pct, "fee_pct")
    if f >= 1:
        raise ValueError("fee_pct is a decimal (0.001 = 0.1%) and must be < 1")
    cash0 = _nonneg(initial_cash, "initial_cash")
    if cash0 <= 0:
        raise ValueError("initial_cash must be > 0")
    mark = _vector(mark_price, n, "mark_price")
    if not np.all(np.isfinite(mark)) or np.any(mark <= 0):
        raise ValueError("mark_price must be finite and > 0 on every row")
    if execution == "next_open":
        if exec_price is None:
            raise ValueError("execution='next_open' fills at exec_price; pass it (e.g. the open prices)")
        ep = _vector(exec_price, n, "exec_price")
    else:
        ep = None
    ent = _signal(entry, n, "entry")
    ext = _signal(exit, n, "exit")
    if isinstance(sizing, str):
        sizing_rule = _choice(sizing, "sizing", _SIZING)
    elif callable(sizing):
        sizing_rule = "callable"
    elif isinstance(sizing, numbers.Real) and not isinstance(sizing, bool):
        if not math.isfinite(float(sizing)) or float(sizing) <= 0:
            raise ValueError("a fixed sizing must be a positive number of shares")
        sizing_rule = "fixed"
    else:
        raise TypeError("sizing must be 'all_cash_floor', 'all_cash', a number of shares or a callable f(ctx)")
    trail = loss = None
    peak_init = None
    if stop is not None:
        if not isinstance(stop, dict) or not stop:
            raise TypeError("stop must be None or a dict with 'trailing_pct' (+ 'peak_init') and/or 'loss_pct'")
        unknown = set(stop) - {"trailing_pct", "peak_init", "loss_pct"}
        if unknown:
            raise ValueError(f"unknown stop keys {sorted(unknown)}; use 'trailing_pct', 'peak_init', 'loss_pct'")
        if "trailing_pct" in stop:
            trail = _nonneg(stop["trailing_pct"], "stop['trailing_pct']")
            if "peak_init" not in stop:
                raise ValueError("a trailing stop needs stop['peak_init']: 'signal_close' or 'exec_price'")
            peak_init = _choice(stop["peak_init"], "stop['peak_init']", _PEAK_INIT)
        elif "peak_init" in stop:
            raise ValueError("stop['peak_init'] is only used with stop['trailing_pct']")
        if "loss_pct" in stop:
            loss = _nonneg(stop["loss_pct"], "stop['loss_pct']")
        if trail is None and loss is None:
            raise ValueError("stop needs 'trailing_pct' and/or 'loss_pct'")
        if exit_priority is None:
            raise ValueError("with a stop, exit_priority is required: 'signal' or 'stop' (the reason when both fire)")
        priority = _choice(exit_priority, "exit_priority", _PRIORITY)
    else:
        if exit_priority is not None:
            raise ValueError("exit_priority is only used with a stop")
        priority = None

    state = {"cash": cash0, "shares": 0.0, "trade": None, "peak": math.nan}
    trades, fills, skipped = [], [], []

    def fill_price(t):
        if execution == "next_open":
            p = ep[t]
            if not (math.isfinite(p) and p > 0):
                raise ValueError(f"exec_price at row {t} ({idx[t]!r}) must be finite and > 0: an order fills there")
            return float(p)
        return float(mark[t])

    def buy(i, j, p):
        cash = state["cash"]
        n_max = max(math.floor((cash - F) / (p * (1.0 + f))), 0)
        if sizing_rule == "all_cash_floor":
            q = float(n_max)
        elif sizing_rule == "all_cash":
            q = max((cash - F) / (p * (1.0 + f)), 0.0)
        elif sizing_rule == "fixed":
            q = float(sizing)
        else:
            ctx = {"cash": cash, "equity": cash, "price": p, "signal_index": i, "exec_index": j,
                   "signal_date": idx[i], "exec_date": idx[j], "signal_close": float(mark[i]),
                   "max_affordable": n_max, "fee_fixed": F, "fee_pct": f}
            q = float(sizing(ctx))
            if not math.isfinite(q):
                raise ValueError(f"sizing returned {q!r} at row {j}")
        if not q > 0:
            skipped.append(idx[i])
            return
        cost = q * p * (1.0 + f) + F
        state["cash"] = cash - cost
        state["shares"] = q
        num = len(trades) + 1
        state["trade"] = {"trade_num": num, "entry_signal_date": idx[i], "entry_date": idx[j], "entry_price": p,
                          "shares": q, "buy_cost": cost, "exit_signal_date": None, "exit_date": None,
                          "exit_price": math.nan, "sell_proceeds": math.nan, "pnl": math.nan, "ret": math.nan,
                          "exit_reason": "open", "entry_index": j, "exit_index": -1, "_signal_index": i}
        trades.append(state["trade"])
        state["peak"] = float(mark[i]) if peak_init == "signal_close" else p
        fills.append({"trade_num": num, "side": "BUY", "signal_date": idx[i], "exec_date": idx[j], "price": p,
                      "shares": q, "fees": q * p * f + F, "cash_after": state["cash"], "pnl": math.nan,
                      "reason": "entry"})

    def sell(i, j, p, reason):
        tr = state["trade"]
        q = state["shares"]
        proceeds = q * p * (1.0 - f) - F
        state["cash"] += proceeds
        state["shares"] = 0.0
        state["trade"] = None
        pnl = proceeds - tr["buy_cost"]
        tr.update({"exit_signal_date": idx[i], "exit_date": idx[j], "exit_price": p, "sell_proceeds": proceeds,
                   "pnl": pnl, "ret": pnl / tr["buy_cost"], "exit_reason": reason, "exit_index": j})
        fills.append({"trade_num": tr["trade_num"], "side": "SELL", "signal_date": idx[i], "exec_date": idx[j],
                      "price": p, "shares": q, "fees": q * p * f + F, "cash_after": state["cash"], "pnl": pnl,
                      "reason": reason})

    equity = np.empty(n)
    cash_s = np.empty(n)
    shares_s = np.empty(n)
    pending_buy = None          # signal row of an entry waiting for its fill
    pending_sell = None         # (signal row, reason) of an exit waiting for its fill
    for t in range(n):
        # 1) orders signalled on the previous row fill on this row (sell first, then buy)
        if execution != "same_close":
            if pending_sell is not None:
                sell(pending_sell[0], t, fill_price(t), pending_sell[1])
                pending_sell = None
            if pending_buy is not None:
                buy(pending_buy, t, fill_price(t))
                pending_buy = None
        # 2) exit rules at this close, while long (not on the entry's own signal row)
        exit_fired = False
        tr = state["trade"]
        if tr is not None and t > tr["_signal_index"]:
            m = float(mark[t])
            stopped = False
            if trail is not None:
                state["peak"] = max(state["peak"], m)
                stopped = m < state["peak"] * (1.0 - trail)
            if loss is not None and m < tr["entry_price"] * (1.0 - loss):
                stopped = True
            if ext[t] or stopped:
                reason = "signal" if ext[t] and (not stopped or priority == "signal") else "stop"
                exit_fired = True
                if execution == "same_close":
                    sell(t, t, m, reason)
                else:
                    pending_sell = (t, reason)
        # 3) entry rule at this close
        if exit_fired:
            can_enter = not block
        else:
            can_enter = state["trade"] is None and pending_buy is None
        if can_enter and ent[t]:
            if execution == "same_close":
                buy(t, t, float(mark[t]))
            elif t + 1 < n:
                pending_buy = t
        # 4) forced close on the last row
        if t == n - 1 and state["trade"] is not None and forced_close == "last_close":
            sell(t, t, float(mark[t]), "forced_close")
            pending_sell = None
        cash_s[t] = state["cash"]
        shares_s[t] = state["shares"]
        equity[t] = state["cash"] + state["shares"] * mark[t]

    cols = ["trade_num", "entry_signal_date", "entry_date", "entry_price", "shares", "buy_cost", "exit_signal_date",
            "exit_date", "exit_price", "sell_proceeds", "pnl", "ret", "exit_reason", "entry_index", "exit_index"]
    trades_df = pd.DataFrame([{k: tr[k] for k in cols} for tr in trades], columns=cols)
    fills_df = pd.DataFrame(fills, columns=["trade_num", "side", "signal_date", "exec_date", "price", "shares", "fees",
                                            "cash_after", "pnl", "reason"])
    eq = pd.Series(equity, index=idx, name="equity")
    rets = eq.pct_change()
    rets.iloc[0] = 0.0
    return {"trades": trades_df, "fills": fills_df, "equity": eq, "returns": rets.rename("returns"),
            "cash": pd.Series(cash_s, index=idx, name="cash"), "shares": pd.Series(shares_s, index=idx, name="shares"),
            "skipped_entries": skipped}


# ----------------------------------------------------------------------------
# weights ledger
# ----------------------------------------------------------------------------
def _as_dates(index):
    """A PeriodIndex as the start timestamps of its periods (so calendar rules and date alignment apply); any other
    index unchanged."""
    return index.to_timestamp(how="start") if isinstance(index, pd.PeriodIndex) else index


def _calendar_mask(idx, schedule):
    idx = _as_dates(idx)
    if isinstance(idx, pd.DatetimeIndex):
        d = idx
    else:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                d = pd.DatetimeIndex(pd.to_datetime(idx))
        except (TypeError, ValueError):
            raise ValueError(f"rebalance={schedule!r} needs dates as the asset_returns index") from None
    unit, edge = schedule.split("_")
    if unit == "month":
        key = d.year * 12 + d.month
    elif unit == "quarter":
        key = d.year * 4 + d.quarter
    else:
        key = d.year
    key = np.asarray(key)
    if len(key) > 1 and np.any(np.diff(d.asi8) <= 0):
        raise ValueError("asset_returns index must be strictly increasing dates")
    out = np.ones(len(key), dtype=bool)
    if edge == "start":
        out[1:] = key[1:] != key[:-1]
    else:
        out[:-1] = key[:-1] != key[1:]
    return out


def _targets(target_weights, idx, cols):
    """(stamped target matrix T x N with NaN rows before the first target row, row-present mask)."""
    T, N = len(idx), len(cols)
    if isinstance(target_weights, pd.DataFrame):
        tw = target_weights
        if tw.index.has_duplicates:
            raise ValueError("target_weights has duplicate index labels")
        if isinstance(tw.index, pd.PeriodIndex):
            tw = tw.set_axis(_as_dates(tw.index), axis=0)
        if isinstance(idx, pd.DatetimeIndex) and not isinstance(tw.index, pd.DatetimeIndex):
            tw = tw.set_axis(pd.to_datetime(tw.index), axis=0)
        missing = tw.index.difference(idx)
        if len(missing):
            raise ValueError(f"target_weights rows must be dates of asset_returns; not found: {list(missing[:3])}")
        unknown = tw.columns.difference(cols)
        if len(unknown):
            raise ValueError(f"target_weights columns not in asset_returns: {list(unknown[:5])}")
        tw = tw.reindex(columns=cols).astype(float).fillna(0.0)
        present = idx.isin(tw.index)
        W = tw.reindex(idx).ffill().to_numpy(dtype=float)
        return W, present
    if isinstance(target_weights, (pd.Series, dict)):
        s = pd.Series(target_weights, dtype=float)
        unknown = s.index.difference(cols)
        if len(unknown):
            raise ValueError(f"target_weights labels not in asset_returns columns: {list(unknown[:5])}")
        row = s.reindex(cols).fillna(0.0).to_numpy(dtype=float)
        return np.tile(row, (T, 1)), np.ones(T, dtype=bool)
    a = np.asarray(target_weights, dtype=float)
    if a.ndim == 1:
        if a.size != N:
            raise ValueError(f"a constant target needs {N} weights, got {a.size}")
        return np.tile(np.nan_to_num(a, nan=0.0), (T, 1)), np.ones(T, dtype=bool)
    if a.shape != (T, N):
        raise ValueError(f"target_weights array must have shape {(T, N)}, got {a.shape}")
    return np.nan_to_num(a, nan=0.0), np.ones(T, dtype=bool)


def weights_ledger(asset_returns, target_weights, *, lag, rebalance, drift, cost_bps, cost_base, cost_timing,
                   cash_return=None, gross_normalize=None):
    """Portfolio weights backtest ledger: execution lag, rebalance schedule, drift, turnover costs (Grinold-Kahn).

    Target weights formed at the close of a row are held from ``lag`` rows later. On the
    rebalance rows the holdings are reset to the target in effect; in between they either drift
    with the asset returns (buy and hold) or are kept at the last target (constant mix). The
    uninvested part 1 - sum(w) is cash.

    Parameters
    ----------
    asset_returns : DataFrame (T periods x N assets) of per-period SIMPLE returns, index = dates
        (needed for the calendar schedules), or a 2-D array. NaN is allowed only where the
        weight held is 0.
    target_weights : the targets, as
        - a DataFrame indexed by dates of asset_returns (all rows, or only some: each row stays
          the target until the next one), columns a subset of the assets (missing or NaN = 0);
        - a Series / dict by asset, or a 1-D array of N weights: one constant target;
        - a 2-D array of shape (T, N).
    lag : REQUIRED int >= 0. The target stamped on row t first earns the return of row t + lag:
        lag=1 is "weights decided at the close of t earn r_{t+1}"; lag=0 means each row is
        already the beginning-of-period weight of its own row's return (e.g. a month's
        start-of-month weights next to that month's return).
    rebalance : REQUIRED. The stamp rows at which the holdings are reset to the target (taking
        effect lag rows later):
        'every_period'   every row (to the latest target row);
        'target_dates'   only rows where target_weights has a row;
        'month_end', 'quarter_end', 'year_end'       the last row of each calendar month /
                         quarter / year present in the index;
        'month_start', 'quarter_start', 'year_start' the first such row (e.g. a monthly series
                         reset each January, April, July and October: 'quarter_start');
        or a boolean array of T stamps, or a list of stamp dates.
        Before the first reset the portfolio is all cash.
    drift : REQUIRED bool. True: between resets the weights drift,
        w_i <- w_i (1 + r_i) / (1 + R_p) (buy and hold). False: the weights stay at the last
        target (constant mix; the re-trading that keeps them there is not counted as turnover).
    cost_bps : REQUIRED. Cost in basis points per unit of the cost base (0 for none).
    cost_base : REQUIRED. 'abs_dw' cost = bps/1e4 * sum_i |w_new_i - w_old_i|; 'half_abs_dw'
        uses half that sum (one-way turnover). Suffix '_ex_initial' ('abs_dw_ex_initial',
        'half_abs_dw_ex_initial') makes the first reset, the build from all cash, free. w_old is
        the drifted weight (drift=True) or the previous weight (drift=False).
    cost_timing : REQUIRED. 'effective': the cost is deducted from the return of the row whose
        beginning-of-period weights changed (stamp + lag); 'decision': from the return of the
        stamp row itself. A reset trades at the close before the row it takes effect on, so
        resets that would take effect after the last row are not traded ('decision' still
        charges one that trades at the last close). With drift=False, 'every_period' and a
        first target row that is not all zero, 'decision' with 'abs_dw_ex_initial' is the pandas
        (w.shift(lag) * r).sum(1) - c * w.diff().abs().sum(1) (for lag <= 1; the last lag - 1
        rows are not charged when lag > 1), and 'effective' is that cost shifted by lag rows.
    cash_return : None (cash earns 0), a per-period rate, or T rates (array or Series on the index).
    gross_normalize : None, or g > 0 to scale every target row to sum|w| = g (all-zero rows stay 0).

    Returns
    -------
    dict with
    - returns: Series of net returns R_p - cost.
    - gross_returns: Series R_p = sum_i w_i r_i + (1 - sum_i w_i) r_cash.
    - weights_bop: DataFrame of the beginning-of-period weights that earn each row's return.
    - turnover: Series sum_i |dw_i| of each reset (initial build included), on the row its cost
      is deducted from (cost_timing).
    - costs: Series of the costs on the rows they are deducted from.
    - rebalanced: boolean Series, True on rows that start with a reset.
    - equity: Series cumprod(1 + returns), growth of 1.

    Formula
    -------
    R_p,t = w_t . r_t + (1 - sum w_t) rc_t, w_t the beginning-of-period weights. With drift the
    end-of-period weights are w_t (1 + r_t) / (1 + R_p,t). Turnover at a reset
    sum_i |target_i - w_pre_i|. Net return R_p,t - cost_t. Costs are deducted from the return;
    the weights drift with the gross return.

    References
    ----------
    Grinold and Kahn, Active Portfolio Management, 2nd ed. (2000), Ch. 16 (turnover and
    transaction costs).

    Example
    -------
    >>> r = pd.DataFrame({'A': [0.10, 0.0], 'B': [0.0, 0.10]}, index=pd.to_datetime(['2024-01-31', '2024-02-29']))
    >>> out = weights_ledger(r, {'A': 0.5, 'B': 0.5}, lag=0, rebalance='every_period', drift=False,
    ...                      cost_bps=10, cost_base='abs_dw', cost_timing='effective')
    >>> out['returns'].round(6).tolist()   # 0.05 less 10 bps of the build sum|w| = 1, then 0.05
    [0.049, 0.05]
    """
    if isinstance(asset_returns, pd.DataFrame):
        R = asset_returns
    else:
        a = np.asarray(asset_returns, dtype=float)
        if a.ndim == 1:
            a = a[:, None]
        if a.ndim != 2:
            raise ValueError("asset_returns must be a 2-D table (T periods x N assets)")
        R = pd.DataFrame(a)
    idx, cols = R.index, R.columns
    T, N = R.shape
    if T < 1 or N < 1:
        raise ValueError("asset_returns is empty")
    if cols.has_duplicates:
        raise ValueError("asset_returns has duplicate columns")
    if isinstance(lag, (bool, np.bool_)) or not isinstance(lag, numbers.Integral) or lag < 0:
        raise ValueError(f"lag must be an int >= 0, got {lag!r}")
    lag = int(lag)
    drift = _flag(drift, "drift")
    bps = _nonneg(cost_bps, "cost_bps")
    cost_base = _choice(cost_base, "cost_base", _COST_BASES)
    cost_timing = _choice(cost_timing, "cost_timing", _COST_TIMINGS)
    half = cost_base.startswith("half")
    ex_initial = cost_base.endswith("_ex_initial")
    rv = R.to_numpy(dtype=float)

    dates = _as_dates(idx)  # a PeriodIndex aligns and schedules by its period starts; outputs keep idx
    W, present = _targets(target_weights, dates, cols)
    if gross_normalize is not None:
        g = _nonneg(gross_normalize, "gross_normalize")
        if g <= 0:
            raise ValueError("gross_normalize must be > 0")
        gross = np.nansum(np.abs(W), axis=1, keepdims=True)
        W = np.where(gross > 0, W * g / np.where(gross > 0, gross, 1.0), W)
    defined = ~np.isnan(W).any(axis=1)

    if isinstance(rebalance, str):
        rb = _choice(rebalance, "rebalance", _SCHEDULES)
        if rb == "every_period":
            stamps = np.ones(T, dtype=bool)
        elif rb == "target_dates":
            stamps = present.copy()
        else:
            stamps = _calendar_mask(dates, rb)
    else:
        arr = np.asarray(rebalance)
        if arr.dtype == bool:
            if arr.shape != (T,):
                raise ValueError(f"a boolean rebalance mask needs {T} values")
            stamps = arr.copy()
        else:
            wanted = _as_dates(pd.Index(rebalance))
            if isinstance(dates, pd.DatetimeIndex):
                wanted = pd.DatetimeIndex(pd.to_datetime(wanted))
            missing = wanted.difference(dates)
            if len(missing):
                raise ValueError(f"rebalance dates not in asset_returns index: {list(missing[:3])}")
            stamps = dates.isin(wanted)
    stamps = stamps & defined

    if cash_return is None:
        rc = np.zeros(T)
    elif isinstance(cash_return, pd.Series):
        rc_s = cash_return.reindex(idx)
        if rc_s.isna().any():
            raise ValueError("cash_return Series must cover every row of asset_returns")
        rc = rc_s.to_numpy(dtype=float)
    else:
        rc = np.broadcast_to(np.asarray(cash_return, dtype=float), (T,)).astype(float)
    if not np.all(np.isfinite(rc)):
        raise ValueError("cash_return must be finite")

    bop = np.zeros((T, N))
    gross_r = np.zeros(T)
    turnover = np.zeros(T)
    costs = np.zeros(T)
    rebal = np.zeros(T, dtype=bool)
    w_carry = np.zeros(N)          # weights entering the row: drifted (drift) or last held (no drift)
    build_pending = ex_initial     # the first reset that trades (the build from all cash) is free
    # Row T is virtual: a reset that would take effect there trades at the last close, so with
    # cost_timing='decision' its cost is charged on its stamp row; no return is earned.
    for t in range(T + 1):
        s = t - lag
        reset = 0 <= s < T and stamps[s] and (t < T or cost_timing == "decision")
        if reset:
            target = W[s]
            dw = float(np.abs(target - w_carry).sum())
            w = target.copy()
            if build_pending and dw > 0:
                c = 0.0
                build_pending = False
            else:
                c = bps / 1e4 * (0.5 * dw if half else dw)
            at = t if cost_timing == "effective" else s
            turnover[at] += dw
            costs[at] += c
        else:
            w = w_carry
        if t == T:
            break
        rebal[t] = reset
        r = rv[t]
        bad = (w != 0) & ~np.isfinite(r)
        if bad.any():
            raise ValueError(f"asset_returns is not finite at row {idx[t]!r} for held asset(s) "
                             f"{list(cols[bad][:5])}")
        r = np.where(w != 0, r, 0.0)
        Rp = float(w @ r + (1.0 - w.sum()) * rc[t])
        bop[t] = w
        gross_r[t] = Rp
        if drift:
            if 1.0 + Rp <= 0:
                raise ValueError(f"portfolio value is wiped out at row {idx[t]!r} (return {Rp})")
            w_carry = w * (1.0 + r) / (1.0 + Rp)
        else:
            w_carry = w
    net = gross_r - costs
    out = {"returns": pd.Series(net, index=idx, name="returns"),
           "gross_returns": pd.Series(gross_r, index=idx, name="gross_returns"),
           "weights_bop": pd.DataFrame(bop, index=idx, columns=cols),
           "turnover": pd.Series(turnover, index=idx, name="turnover"),
           "costs": pd.Series(costs, index=idx, name="costs"),
           "rebalanced": pd.Series(rebal, index=idx, name="rebalanced"),
           "equity": pd.Series(np.cumprod(1.0 + net), index=idx, name="equity")}
    return out
