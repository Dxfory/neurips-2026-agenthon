"""Event studies: market-model abnormal returns and the Patell, BMP, Kolari-Pynnonen and rank tests.

Conventions for the whole module:
- ``returns`` holds per-period returns (simple or log, as given; nothing is converted), one
  column per security, indexed by date, oldest first. ``market`` is the market (or benchmark)
  return on the same dates.
- Relative days count the TRADING days of each security: the dates where both its return and
  the market return are finite. Day 0 is the event day; windows are (first, last) relative
  days, both included: estimation_window=(-130, -11) and event_window=(-5, 5) are the 120 and
  11 days of a classic earnings study.
- Each event is fitted on its own estimation window. Abnormal returns (AR) keep their calendar
  date; CAR is the sum of an event's ARs, AAR the cross-event mean AR of a relative day and
  CAAR the running sum of AAR.
- Every convention on which textbooks and tasks differ (day-0 rule, Patell correction,
  multi-day standardisation, Kolari-Pynnonen formula and alignment, rank scaling, ...) is a
  REQUIRED keyword.

Functions
---------
event_study        per-event market model on an estimation window; AR, CAR, AAR, CAAR
event_study_tests  Patell (1976), Boehmer-Musumeci-Poulsen (1991), Kolari-Pynnonen (2010)
                   adjusted BMP, cross-sectional t, Corrado (1989) / Cowan (1992) rank test and
                   Kolari-Pynnonen (2011) GRANK

References: MacKinlay (1997), J. Econ. Lit. 35; Campbell, Lo and MacKinlay, The Econometrics of
Financial Markets (1997), Ch. 4; Patell (1976), J. Acc. Res. 14; Boehmer, Musumeci and Poulsen
(1991), J. Fin. Econ. 30; Kolari and Pynnonen (2010), Rev. Fin. Stud. 23; Kolari and Pynnonen
(2011), J. Empirical Finance 18; Corrado (1989), J. Fin. Econ. 23; Corrado and Zivney (1992),
JFQA 27; Cowan (1992), Rev. Quant. Fin. Acc. 2.
"""

from __future__ import annotations

import math
import numbers

import numpy as np
import pandas as pd
from scipy import stats

__all__ = [
    "event_study",
    "event_study_tests",
]

_MODELS = ("market", "mean_adjusted", "market_adjusted")
_DAY0 = ("exact", "next", "previous")
_TESTS = ("patell", "bmp", "kp", "csect", "rank")
_CAR_STD = ("sum_sar", "car_var")
_KP_FORMULAS = ("kp2010", "inflation")
_KP_ALIGN = ("calendar", "event_time")
_RANK_METHODS = ("average", "min", "max", "ordinal")
_RANK_SCALING = ("raw", "uniform")
_RANK_AGG = ("cowan", "grank")


def _choice(value, name, allowed):
    if not isinstance(value, str) or value.strip().lower() not in allowed:
        raise ValueError(f"{name} must be one of {', '.join(repr(a) for a in allowed)}, got {value!r}")
    return value.strip().lower()


def _flag(value, name):
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be True or False, got {value!r}")
    return bool(value)


def _int(value, name, low):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral) or value < low:
        raise ValueError(f"{name} must be an int >= {low}, got {value!r}")
    return int(value)


def _window(w, name):
    try:
        a, b = w
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a pair (first, last) of relative trading days") from None
    a, b = _int(a, f"{name}[0]", -10**9), _int(b, f"{name}[1]", -10**9)
    if a > b:
        raise ValueError(f"{name} must have first <= last, got {w!r}")
    return a, b


def _events_frame(events):
    if isinstance(events, pd.DataFrame):
        if not {"ticker", "event_date"} <= set(events.columns):
            raise ValueError("an events DataFrame needs the columns 'ticker' and 'event_date'")
        return events[["ticker", "event_date"]].reset_index(drop=True)
    rows = [tuple(e) for e in events]
    if any(len(r) != 2 for r in rows):
        raise ValueError("events must be (ticker, event_date) pairs or a DataFrame with those columns")
    return pd.DataFrame(rows, columns=["ticker", "event_date"])


# ----------------------------------------------------------------------------
# event study
# ----------------------------------------------------------------------------
def event_study(returns, market, events, *, estimation_window, event_window, model, min_est_obs,
                require_full_event_window, event_day_rule):
    """Event study (MacKinlay 1997): per-event market model on an estimation window, AR, CAR, AAR and CAAR by calendar date.

    Parameters
    ----------
    returns : DataFrame of per-period security returns, index = dates (ascending, unique),
        one column per ticker (a Series is one ticker named by its .name).
    market : Series of market returns on the same kind of index, or a DataFrame with one
        benchmark column per ticker.
    events : DataFrame with columns 'ticker' and 'event_date' (other columns are ignored), or a
        list of (ticker, event_date) pairs. event_id is the row position in this input.
    estimation_window : REQUIRED (first, last) relative trading days of the estimation window,
        e.g. (-130, -11); it must not overlap the event window. Days before the first trading
        day of the security are simply missing (see min_est_obs).
    event_window : REQUIRED (first, last) relative trading days, e.g. (-5, 5).
    model : REQUIRED. 'market' R = alpha + beta R_m + e by OLS (k = 2 parameters);
        'mean_adjusted' R = mu + e (k = 1); 'market_adjusted' AR = R - R_m (k = 0).
    min_est_obs : REQUIRED int. An event needs at least this many estimation-window days
        (>= k + 3), else it is skipped.
    require_full_event_window : REQUIRED bool. True skips an event unless every event-window day
        exists; False keeps the days that exist (the others are NaN, never zero).
    event_day_rule : REQUIRED. Day 0 when event_date is not a trading day of the security:
        'exact' skips the event; 'next' uses the first trading day on or after event_date
        (announcements on holidays or after the close); 'previous' the last one on or before.

    Returns
    -------
    dict with
    - params: DataFrame by event_id: ticker, event_date, day0_date, alpha, beta, sigma
      (residual sd, SSE/(L - k)), r_squared, n_obs (L), dof (L - k), mkt_mean and mkt_ss (mean
      and sum of squared deviations of R_m in the estimation window, for Patell's C).
    - ar: DataFrame of ARs, event_id x relative day (NaN where missing).
    - ar_long: DataFrame, one row per event day: event_id, ticker, event_date, rel_day, date,
      ret, mkt, ar.
    - car: Series by event_id, the sum of the event-window ARs.
    - aar: Series by relative day, the mean AR across events; caar: its running sum from the
      first event-window day.
    - est_resid: DataFrame, one row per estimation day: event_id, ticker, rel_day, date, resid,
      mkt (what event_study_tests uses for the Patell variance, ranks and rho_bar).
    - skipped: DataFrame event_id, ticker, event_date, reason.
    - windows, model: the settings, for event_study_tests.

    Formula
    -------
    Market model: (alpha, beta) by OLS on the L estimation days; e = R - alpha - beta R_m;
    sigma^2 = sum e^2 / (L - 2). AR_t = R_t - alpha - beta R_m,t on the event days.
    CAR_i = sum_t AR_i,t; AAR_t = mean_i AR_i,t; CAAR(t) = sum_{s <= t} AAR_s.

    References
    ----------
    MacKinlay (1997); Campbell, Lo and MacKinlay (1997) Ch. 4; Brown and Warner (1985).

    Example
    -------
    >>> res = event_study(rets, mkt, [('AAPL', '2020-07-30')], estimation_window=(-130, -11),
    ...     event_window=(-5, 5), model='market', min_est_obs=60, require_full_event_window=True,
    ...     event_day_rule='exact')                                    # doctest: +SKIP
    >>> res['car'], res['ar'].loc[:, -1:1].sum(axis=1)                 # CAR(-5,5), CAR(-1,1)
    """
    if isinstance(returns, pd.Series):
        returns = returns.to_frame(name=returns.name if returns.name is not None else "asset")
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a pandas DataFrame (dates x tickers)")
    if not isinstance(market, (pd.Series, pd.DataFrame)):
        raise TypeError("market must be a pandas Series (or a DataFrame with a column per ticker)")
    e1, e2 = _window(estimation_window, "estimation_window")
    w1, w2 = _window(event_window, "event_window")
    if not (e2 < w1 or e1 > w2):
        raise ValueError("estimation_window and event_window overlap")
    model = _choice(model, "model", _MODELS)
    k = {"market": 2, "mean_adjusted": 1, "market_adjusted": 0}[model]
    min_est_obs = _int(min_est_obs, "min_est_obs", k + 3)
    require_full = _flag(require_full_event_window, "require_full_event_window")
    day0_rule = _choice(event_day_rule, "event_day_rule", _DAY0)
    ev = _events_frame(events)
    is_dt = isinstance(returns.index, pd.DatetimeIndex)

    cache = {}

    def calendar(tk):
        if tk not in cache:
            if tk not in returns.columns:
                cache[tk] = None
            else:
                m = market if isinstance(market, pd.Series) else market.get(tk)
                if m is None:
                    raise ValueError(f"market has no column for ticker {tk!r}")
                j = pd.concat([returns[tk].rename("r"), m.rename("m")], axis=1, join="inner").astype(float)
                j = j[np.isfinite(j["r"].to_numpy()) & np.isfinite(j["m"].to_numpy())]
                if not j.index.is_monotonic_increasing or j.index.has_duplicates:
                    raise ValueError("returns and market must be indexed by unique dates in ascending order")
                cache[tk] = (j.index, j["r"].to_numpy(), j["m"].to_numpy())
        return cache[tk]

    prm, skipped, ar_parts, est_parts = [], [], [], []
    for eid, (tk, edate) in enumerate(zip(ev["ticker"], ev["event_date"])):
        cal = calendar(tk)
        if cal is None:
            skipped.append((eid, tk, edate, "ticker not in returns"))
            continue
        dates, R, M = cal
        n = len(dates)
        d = pd.Timestamp(edate) if is_dt else edate
        if day0_rule == "exact":
            p0 = int(dates.get_indexer([d])[0])
        elif day0_rule == "next":
            p0 = int(dates.searchsorted(d, side="left"))
            p0 = p0 if p0 < n else -1
        else:
            p0 = int(dates.searchsorted(d, side="right")) - 1
        if p0 < 0:
            skipped.append((eid, tk, edate, "no day 0 (event_day_rule)"))
            continue
        lo, hi = max(p0 + e1, 0), min(p0 + e2, n - 1)
        est = np.arange(lo, hi + 1) if lo <= hi else np.arange(0)
        if est.size < min_est_obs:
            skipped.append((eid, tk, edate, f"{est.size} estimation days < min_est_obs"))
            continue
        a, b = p0 + w1, p0 + w2
        if require_full and (a < 0 or b > n - 1):
            skipped.append((eid, tk, edate, "event window incomplete"))
            continue
        evp = np.arange(max(a, 0), min(b, n - 1) + 1)
        if evp.size == 0:
            skipped.append((eid, tk, edate, "no event-window day"))
            continue
        y, x = R[est], M[est]
        mbar = float(x.mean())
        mss = float(((x - mbar) ** 2).sum())
        if model == "market":
            if mss <= 0:
                skipped.append((eid, tk, edate, "market return constant in the estimation window"))
                continue
            X = np.column_stack([np.ones(est.size), x])
            alpha, beta = np.linalg.lstsq(X, y, rcond=None)[0]
        elif model == "mean_adjusted":
            alpha, beta = float(y.mean()), 0.0
        else:
            alpha, beta = 0.0, 1.0
        resid = y - alpha - beta * x
        L = est.size
        sse = float(resid @ resid)
        sst = float(((y - y.mean()) ** 2).sum())
        r2 = (1.0 - sse / sst if sst > 0 else math.nan) if model == "market" else (
            0.0 if model == "mean_adjusted" else math.nan)
        prm.append({"event_id": eid, "ticker": tk, "event_date": edate, "day0_date": dates[p0],
                    "alpha": float(alpha), "beta": float(beta), "sigma": math.sqrt(sse / (L - k)), "r_squared": r2,
                    "n_obs": L, "dof": L - k, "mkt_mean": mbar, "mkt_ss": mss})
        ar = R[evp] - alpha - beta * M[evp]
        ar_parts.append(pd.DataFrame({"event_id": eid, "ticker": tk, "event_date": edate, "rel_day": evp - p0,
                                      "date": dates[evp], "ret": R[evp], "mkt": M[evp], "ar": ar}))
        est_parts.append(pd.DataFrame({"event_id": eid, "ticker": tk, "rel_day": est - p0, "date": dates[est],
                                       "resid": resid, "mkt": x}))

    pcols = ["ticker", "event_date", "day0_date", "alpha", "beta", "sigma", "r_squared", "n_obs", "dof", "mkt_mean",
             "mkt_ss"]
    params = pd.DataFrame(prm, columns=["event_id"] + pcols).set_index("event_id")
    ar_cols = ["event_id", "ticker", "event_date", "rel_day", "date", "ret", "mkt", "ar"]
    ar_long = pd.concat(ar_parts, ignore_index=True) if ar_parts else pd.DataFrame(columns=ar_cols)
    est_cols = ["event_id", "ticker", "rel_day", "date", "resid", "mkt"]
    est_resid = pd.concat(est_parts, ignore_index=True) if est_parts else pd.DataFrame(columns=est_cols)
    rel = pd.Index(range(w1, w2 + 1), name="rel_day")
    if ar_parts:
        ar_wide = ar_long.pivot(index="event_id", columns="rel_day", values="ar").reindex(
            index=params.index, columns=rel)
    else:
        ar_wide = pd.DataFrame(index=params.index, columns=rel, dtype=float)
    ar_wide.columns.name = "rel_day"
    car = ar_wide.sum(axis=1, min_count=1).rename("car")
    aar = ar_wide.mean(axis=0).rename("aar")
    return {"params": params, "ar": ar_wide, "ar_long": ar_long, "car": car, "aar": aar,
            "caar": aar.cumsum().rename("caar"), "est_resid": est_resid,
            "skipped": pd.DataFrame(skipped, columns=["event_id", "ticker", "event_date", "reason"]),
            "windows": {"estimation": (e1, e2), "event": (w1, w2)}, "model": model}


# ----------------------------------------------------------------------------
# test statistics
# ----------------------------------------------------------------------------
def _kp_factor(rho, n, formula):
    """Kolari-Pynnonen multiplier of the BMP t: kp2010 sqrt((1-r)/(1+(n-1)r)); inflation 1/sqrt(1+(n-1)r)."""
    den = 1.0 + (n - 1) * rho
    if not den > 0:
        return math.nan
    if formula == "kp2010":
        return math.sqrt((1.0 - rho) / den) if rho <= 1 else math.nan
    return 1.0 / math.sqrt(den)


def _rho_bar(est, alignment, min_overlap):
    """Mean pairwise Pearson correlation of estimation residuals over common dates (or relative days)."""
    key = "date" if alignment == "calendar" else "rel_day"
    if est.empty:
        return 0.0, 0
    mat = est.pivot(index=key, columns="event_id", values="resid")
    if mat.shape[1] < 2:
        return 0.0, 0
    c = mat.corr(method="pearson", min_periods=min_overlap).to_numpy()
    iu = np.triu_indices(c.shape[0], k=1)
    vals = c[iu]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0, 0
    return float(vals.mean()), int(vals.size)


def _rank_dev(values, method, scaling):
    """Rank deviations of one event's combined series: K - (T+1)/2 ('raw') or K/(T+1) - 1/2 ('uniform')."""
    K = stats.rankdata(values, method=method)
    T = values.size
    return K - (T + 1) / 2.0 if scaling == "raw" else K / (T + 1.0) - 0.5


def _grid_stat(dev, n_events):
    """Per slot (column) of an events x slots array of rank deviations: the mean over the events present
    dbar_t, their number N_t, and S = sqrt((1/T) sum_t (N_t/N) dbar_t^2) over the T slots with N_t > 0."""
    nt = np.sum(~np.isnan(dev), axis=0)
    used = nt > 0
    dbar = np.full(dev.shape[1], np.nan)
    dbar[used] = np.nansum(dev[:, used], axis=0) / nt[used]
    T = int(used.sum())
    S = math.sqrt(np.sum(nt[used] / n_events * dbar[used] ** 2) / T) if T else math.nan
    return dbar, nt, S, T


def event_study_tests(res, *, tests, patell_correction, car_standardization, kp_formula, kp_alignment,
                      kp_min_overlap, rank_method, rank_scaling, rank_aggregation, windows=None):
    """Event study tests: Patell, Boehmer-Musumeci-Poulsen (BMP), Kolari-Pynnonen, Corrado-Cowan rank and GRANK.

    Parameters
    ----------
    res : the dict returned by event_study.
    tests : REQUIRED list of 'patell', 'bmp', 'kp', 'csect', 'rank'.
    patell_correction : REQUIRED bool. True: Patell's prediction-error factor
        C_i,t = 1 + 1/L_i + (R_m,t - Rbar_m)^2 / sum_tau (R_m,tau - Rbar_m)^2 (market model;
        1 + 1/L_i for 'mean_adjusted'; 1 for 'market_adjusted'); False: C = 1.
        SAR_i,t = AR_i,t / (sigma_i sqrt(C_i,t)).
    car_standardization : REQUIRED. How a multi-day window [a, b] (L_W days) is standardised:
        'sum_sar' CSAR_i = sum_t SAR_i,t / sqrt(L_W) (Patell 1976);
        'car_var' SCAR_i = CAR_i / sqrt(Var CAR_i), Var = sigma_i^2 [L_W + L_W^2/L_i +
        (sum_t (R_m,t - Rbar_m))^2 / sum_tau (R_m,tau - Rbar_m)^2] (the exact market-model
        prediction variance; sigma_i^2 L_W without the correction). One day: both are SAR.
    kp_formula : REQUIRED. Kolari-Pynnonen (2010) adjustment of the BMP t with rho_bar:
        'kp2010' t_BMP sqrt((1 - rho_bar) / (1 + (N - 1) rho_bar)); 'inflation'
        t_BMP / sqrt(1 + (N - 1) rho_bar) (variance inflation only). rho_bar = 0 gives BMP.
    kp_alignment : REQUIRED. 'calendar': rho_bar is the mean pairwise correlation of the
        estimation residuals on their COMMON CALENDAR DATES; 'event_time': on common relative
        days.
    kp_min_overlap : REQUIRED int >= 2. Pairs with fewer common observations are skipped
        (rho_bar = 0 when no pair is left: no calendar overlap, no cross-correlation).
    rank_method : REQUIRED tie rule of the ranks: 'average', 'min', 'max' or 'ordinal'.
    rank_scaling : REQUIRED. 'raw' ranks K with mean (T_i+1)/2, or 'uniform' K/(T_i+1) with
        mean 1/2 (Corrado-Zivney 1992; equal to 'raw' when every T_i is the same).
    rank_aggregation : REQUIRED. 'cowan': Corrado (1989) ranks of each event's estimation
        residuals and event-window ARs together (T_i = L_i + event days); day statistic
        dbar_0 / S and multi-day sum_{t in W} dbar_t / (sqrt(L_W) S) (Cowan 1992), with
        S^2 = (1/T) sum_t (N_t/N) dbar_t^2 over every relative day. 'grank': Kolari-Pynnonen
        (2011) generalized rank test: the window's standardised CAR, divided by its
        cross-sectional sd, is ranked as one observation among the event's estimation
        SAR = e/sigma (T_i = L_i + 1); Z = ubar_0 / S, t = Z sqrt((T-2)/(T-1-Z^2)).
    windows : None (only the full event window, label 'full') or a dict label -> (a, b) of
        relative days inside the event window, e.g. {'full': (-5, 5), 'day0': (0, 0)}.

    Returns
    -------
    dict with
    - stats: DataFrame, one row per window, columns n_events, mean_car and for each test asked:
      patell_z, patell_p; bmp_t, bmp_p; kp_t, kp_p; csect_t, csect_p (t-test of the CARs);
      rank_z, rank_p ('cowan') or grank_z, grank_t, grank_p ('grank'). p-values are two-sided:
      normal for z, Student t(N-1) for bmp/kp/csect, t(T-2) for grank_t.
    - by_day: the same columns for every single event day (index rel_day).
    - rho_bar, n_pairs: the Kolari-Pynnonen average correlation and the pairs it averages
      (NaN, 0 when 'kp' is not asked).

    Formula
    -------
    Patell Z = sum_i S_i / sqrt(sum_i d_i/(d_i - 2)), d_i = L_i - k. BMP t = mean(S) / (sd(S)
    / sqrt(N)), sd with ddof 1. S_i is SAR (one day), CSAR or SCAR (car_standardization).
    Kolari-Pynnonen: as kp_formula. Rank tests: as rank_aggregation.

    References
    ----------
    Patell (1976); Boehmer, Musumeci and Poulsen (1991); Kolari and Pynnonen (2010, 2011);
    Corrado (1989); Corrado and Zivney (1992); Cowan (1992); Campbell, Lo and MacKinlay (1997)
    Ch. 4.

    Example
    -------
    >>> out = event_study_tests(res, tests=['patell', 'bmp', 'kp', 'rank'], patell_correction=True,
    ...     car_standardization='car_var', kp_formula='kp2010', kp_alignment='calendar',
    ...     kp_min_overlap=10, rank_method='average', rank_scaling='uniform',
    ...     rank_aggregation='cowan', windows={'full': (-5, 5), 'day0': (0, 0)})  # doctest: +SKIP
    >>> out['stats'].loc['full', 'kp_t'], out['by_day'].loc[0, 'rank_z'], out['rho_bar']
    """
    if not isinstance(res, dict) or not {"params", "ar", "ar_long", "est_resid", "windows", "model"} <= set(res):
        raise TypeError("res must be the dict returned by event_study")
    if isinstance(tests, str):
        tests = [tests]
    tests = [_choice(t, "tests", _TESTS) for t in tests]
    if not tests:
        raise ValueError(f"tests must name at least one of {_TESTS}")
    corr = _flag(patell_correction, "patell_correction")
    car_std = _choice(car_standardization, "car_standardization", _CAR_STD)
    kp_formula = _choice(kp_formula, "kp_formula", _KP_FORMULAS)
    kp_alignment = _choice(kp_alignment, "kp_alignment", _KP_ALIGN)
    kp_min_overlap = _int(kp_min_overlap, "kp_min_overlap", 2)
    rank_method = _choice(rank_method, "rank_method", _RANK_METHODS)
    rank_scaling = _choice(rank_scaling, "rank_scaling", _RANK_SCALING)
    rank_agg = _choice(rank_aggregation, "rank_aggregation", _RANK_AGG)
    w1, w2 = res["windows"]["event"]
    e1, e2 = res["windows"]["estimation"]
    model = res["model"]
    if windows is None:
        windows = {"full": (w1, w2)}
    if not isinstance(windows, dict) or not windows:
        raise TypeError("windows must be None or a dict label -> (first, last)")
    wins = {}
    for lab, w in windows.items():
        a, b = _window(w, f"windows[{lab!r}]")
        if a < w1 or b > w2:
            raise ValueError(f"window {lab!r} {w!r} is outside the event window {(w1, w2)}")
        wins[lab] = (a, b)

    params = res["params"]
    ids = params.index.to_numpy()
    N = ids.size
    A = res["ar"].reindex(index=params.index, columns=range(w1, w2 + 1)).to_numpy(dtype=float)
    arl = res["ar_long"]
    Mk = arl.pivot(index="event_id", columns="rel_day", values="mkt").reindex(
        index=params.index, columns=range(w1, w2 + 1)).to_numpy(dtype=float) if N else A.copy()
    s = params["sigma"].to_numpy(dtype=float)[:, None]
    L = params["n_obs"].to_numpy(dtype=float)[:, None]
    dof = params["dof"].to_numpy(dtype=float)
    mbar = params["mkt_mean"].to_numpy(dtype=float)[:, None]
    mss = params["mkt_ss"].to_numpy(dtype=float)[:, None]
    if corr and model == "market":
        C = 1.0 + 1.0 / L + (Mk - mbar) ** 2 / mss
    elif corr and model == "mean_adjusted":
        C = np.broadcast_to(1.0 + 1.0 / L, A.shape)
    else:
        C = np.ones_like(A)
    SAR = A / (s * np.sqrt(C))

    rho_bar, n_pairs = math.nan, 0
    if "kp" in tests:
        rho_bar, n_pairs = _rho_bar(res["est_resid"], kp_alignment, kp_min_overlap)

    # Corrado/Cowan rank deviations on the grid (estimation days + event days), events x slots
    est = res["est_resid"]
    est_slots = np.arange(e1, e2 + 1)
    by_event = {eid: g for eid, g in est.groupby("event_id")} if len(est) else {}
    E = np.full((N, est_slots.size), np.nan)          # estimation residuals on relative days
    for i, eid in enumerate(ids):
        g = by_event[eid]
        E[i, g["rel_day"].to_numpy() - e1] = g["resid"].to_numpy(dtype=float)
    if "rank" in tests and rank_agg == "cowan":
        comb = np.concatenate([E, A], axis=1)
        dev = np.full(comb.shape, np.nan)
        for i in range(N):
            ok = ~np.isnan(comb[i])
            dev[i, ok] = _rank_dev(comb[i, ok], rank_method, rank_scaling)
        dbar, nt, S_rank, _ = _grid_stat(dev, N)
        dbar_ev, nt_ev = dbar[est_slots.size:], nt[est_slots.size:]
    sar_est = E / s

    def window_stats(a, b):
        cols = np.arange(a - w1, b - w1 + 1)
        Aw, SARw, Mw = A[:, cols], SAR[:, cols], Mk[:, cols]
        have = ~np.isnan(Aw)
        LW = have.sum(axis=1).astype(float)
        keep = LW > 0
        car = np.where(keep, np.nansum(Aw, axis=1), np.nan)
        if car_std == "sum_sar":
            S = np.where(keep, np.nansum(SARw, axis=1) / np.sqrt(np.where(keep, LW, 1.0)), np.nan)
        else:
            V = LW.copy()
            if corr and model in ("market", "mean_adjusted"):
                V = V + LW ** 2 / L[:, 0]
            if corr and model == "market":
                V = V + np.nansum(np.where(have, Mw - mbar, 0.0), axis=1) ** 2 / mss[:, 0]
            S = np.where(keep, car / (s[:, 0] * np.sqrt(np.where(keep, V, 1.0))), np.nan)
        out = {"n_events": int(keep.sum()), "mean_car": float(np.nanmean(car)) if keep.any() else math.nan}
        Sk, n = S[keep], int(keep.sum())
        tdist = stats.t(n - 1) if n > 1 else None

        def tstat(x):
            if n < 2:
                return math.nan
            sd = float(np.std(x, ddof=1))
            return float(np.mean(x) / (sd / math.sqrt(n))) if sd > 0 else math.nan

        def tp(t):
            return float(2 * tdist.sf(abs(t))) if tdist is not None and np.isfinite(t) else math.nan

        if "patell" in tests:
            d = dof[keep]
            z = float(Sk.sum() / math.sqrt(np.sum(d / (d - 2.0)))) if n else math.nan
            out.update(patell_z=z, patell_p=float(2 * stats.norm.sf(abs(z))) if np.isfinite(z) else math.nan)
        t_bmp = tstat(Sk)
        if "bmp" in tests:
            out.update(bmp_t=t_bmp, bmp_p=tp(t_bmp))
        if "kp" in tests:
            t_kp = t_bmp * _kp_factor(rho_bar, n, kp_formula) if np.isfinite(t_bmp) else math.nan
            out.update(kp_t=t_kp, kp_p=tp(t_kp))
        if "csect" in tests:
            t_cs = tstat(car[keep])
            out.update(csect_t=t_cs, csect_p=tp(t_cs))
        if "rank" in tests and rank_agg == "cowan":
            used = nt_ev[cols] > 0
            m = int(used.sum())
            z = float(np.sum(dbar_ev[cols][used]) / (math.sqrt(m) * S_rank)) if m and S_rank > 0 else math.nan
            out.update(rank_z=z, rank_p=float(2 * stats.norm.sf(abs(z))) if np.isfinite(z) else math.nan)
        if "rank" in tests and rank_agg == "grank":
            z = tg = pg = math.nan
            if n > 1:
                sd = float(np.std(Sk, ddof=1))
                gev = np.full(N, np.nan)
                gev[keep] = Sk / sd if sd > 0 else np.nan
                G = np.concatenate([sar_est, gev[:, None]], axis=1)
                dev = np.full(G.shape, np.nan)
                for i in np.flatnonzero(keep):
                    ok = ~np.isnan(G[i])
                    dev[i, ok] = _rank_dev(G[i, ok], rank_method, rank_scaling)
                dev = dev[keep]
                ub, _, S_g, T = _grid_stat(dev, n)
                if S_g > 0:
                    z = float(ub[-1] / S_g)
                    if T - 1 - z * z > 0:
                        tg = z * math.sqrt((T - 2.0) / (T - 1.0 - z * z))
                        pg = float(2 * stats.t(T - 2).sf(abs(tg)))
            out.update(grank_z=z, grank_t=tg, grank_p=pg)
        return out

    st = pd.DataFrame({lab: window_stats(a, b) for lab, (a, b) in wins.items()}).T
    by_day = pd.DataFrame({t: window_stats(t, t) for t in range(w1, w2 + 1)}).T
    by_day.index.name = "rel_day"
    for df in (st, by_day):
        df["n_events"] = df["n_events"].astype(int)
        for c in df.columns.drop("n_events"):
            df[c] = df[c].astype(float)
    return {"stats": st, "by_day": by_day, "rho_bar": rho_bar, "n_pairs": n_pairs}
