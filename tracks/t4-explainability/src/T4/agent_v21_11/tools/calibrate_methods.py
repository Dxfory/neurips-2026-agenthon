"""Which method should each statistical baseline use? Replays on public data published by 2021 (development only).

Every number printed here is scorer 5.2.2's predictive quality measured on pre-2022 public data (../calib_data, sources
and SHA-256 in its README): regression naive_MAE / (naive_MAE + MAE) per cross-section against the naive rule (the last
value for a next-value target, zero change for a change target); ranking (Spearman + 1) / 2. 0.5 = as good as the
naive rule. No task outcome, corpus or Development score is read.

v19 (review of v17): every cross-section now holds the series' values at ONE date or reference month -- v17 grouped
daily series by array position, and the series start in different years (the euro in 1999, Treasury yields in 1990),
so a 'cross-section' mixed years. A first-print target is replayed in real time: the window is the history as known
the day before the release (ALFRED vintages) and the outcome is the first print -- v17 read the 2021 vintage, revised
values the agent never sees.

  first   next first print of a monthly m/m rate (CPI components; twelve macro series), real time:
          v16's 6-period median vs v17's AR(1) reversion, by half-sample
  rate    next daily return (FX, oil), by date
  level   next value of a level series: v16 vs v19 must agree (AR(1) is kept out of the level candidates)
  change  h-step change: v16's switching rule vs v17's stricter one, on random walks (yields, FX, oil; by date) and on
          trending monthly levels (macro levels, CPI indexes; by month)
  cot     ranking ten futures markets by the change in net positioning as a share of open interest over 2, 5 and 8
          weeks (CFTC 2011-2021): v16 (switching), v18 (a fixed 0.15 reversion), v19 (reversion growing with the
          horizon; drift only when it halves the backtest error)

    ../.venv/bin/python tools/calibrate_methods.py
"""
from __future__ import annotations

import csv
import importlib.util
import json
import math
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
AGENT = HERE.parent
ROOT = AGENT.parent
DATA = ROOT / "calib_data"
sys.path[:0] = [str(HERE), str(AGENT)]
import calibrate_scales as C  # noqa: E402
from t4agent import stats as S  # noqa: E402

MACRO = ("PAYEMS", "INDPRO", "RSAFS", "HOUST", "PCE", "PI", "DGORDER", "TCU", "MANEMP", "PERMIT", "BUSINV", "DSPIC96")


def load_stats(version: str):
    """Another version's stats module, for side-by-side replays (v16 and v18 sit next to this agent)."""
    folder = ROOT / f"agent_{version}" / "t4agent"
    pkg = importlib.util.spec_from_file_location(f"t4m_{version}", folder / "__init__.py", submodule_search_locations=[str(folder)])
    pm = importlib.util.module_from_spec(pkg)
    sys.modules[f"t4m_{version}"] = pm
    pkg.loader.exec_module(pm)
    spec = importlib.util.spec_from_file_location(f"t4m_{version}.stats", folder / "stats.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules[f"t4m_{version}.stats"] = m
    spec.loader.exec_module(m)
    return m


def pq(preds, naive, acts) -> float:
    n = st.mean(abs(a - b) for a, b in zip(naive, acts))
    e = st.mean(abs(a - b) for a, b in zip(preds, acts))
    return n / (n + e) if n + e > 0 else 0.5


def _rt(series: str):
    d = json.loads((DATA / f"alfred_rt_{series}.json").read_text())
    by: dict[str, list[tuple[str, str, float]]] = defaultdict(list)
    for o in d["observations"]:
        try:
            by[o["date"][:7]].append((o["realtime_start"], o["realtime_end"], float(o["value"])))
        except ValueError:
            continue  # '.'
    vint = sorted({o["realtime_start"] for o in d["observations"]} - {d["realtime_start"]})  # the window start is no release
    return by, vint


def _at(by, ref: str, day: str):
    for a, b, v in by.get(ref, ()):
        if a <= day <= b:
            return v
    return None


def _prev(ref: str) -> str:
    y, m = int(ref[:4]), int(ref[5:7]) - 1
    return f"{y - (m == 0):04d}-{m or 12:02d}"


def latest_levels(series: str) -> dict[str, float]:
    """Reference month -> level in the vintage of 2021-12-31 (revised: for level and change replays only)."""
    by, _ = _rt(series)
    return {ref: v for ref in by if (v := _at(by, ref, "2021-12-31")) is not None}


def first_prints(series: str, window: int = 9) -> dict[str, tuple[list[float], float]]:
    """Reference month M -> (the last `window` m/m % changes as known the day before M's first release, M's first-print
    m/m %), for every release that added exactly one reference month (ALFRED real-time periods, 2002-2021)."""
    by, vint = _rt(series)
    out = {}
    for before, day in zip(vint, vint[1:]):
        new = [r for r in by if _at(by, r, day) is not None and _at(by, r, before) is None]
        if len(new) != 1:
            continue
        m = new[0]
        p = _prev(m)
        lv_m, lv_p = _at(by, m, day), _at(by, p, day)
        if lv_p is None or not lv_p:
            continue
        refs = [p]
        for _ in range(window):
            refs.append(_prev(refs[-1]))
        lv = [_at(by, r, before) for r in reversed(refs)]
        if any(v is None or v == 0 for v in lv):
            continue
        out[m] = ([round((b / a - 1) * 100, 3) for a, b in zip(lv, lv[1:])], round((lv_m / lv_p - 1) * 100, 3))
    return out


def next_value(Sm, target: str, unit: str, period: float):
    def f(w):
        ser = Sm.Series("x", "value", list(w), [str(j) for j in range(len(w))], period, True)
        try:
            return Sm.forecast(ser, target, 30, {"unit": unit}).value
        except (ValueError, ZeroDivisionError, IndexError):
            return w[-1]
    return f


def compare(cross: dict, label: str, target: str, unit: str, period: float, versions, min_n: int = 3) -> dict:
    keys = [k for k in sorted(cross) if len(cross[k]) >= min_n]
    half = len(keys) // 2
    out = {}
    for name, Sm in versions:
        f = next_value(Sm, target, unit, period)
        vals = [pq([f(w) for w, _ in cross[k]], [w[-1] for w, _ in cross[k]], [a for _, a in cross[k]]) for k in keys]
        out[name] = (st.mean(vals), st.mean(vals[:half]), st.mean(vals[half:]))
    print(f"  {label:50s} " + "   ".join(f"{n} {m:.4f} ({a:.4f}/{b:.4f})" for n, (m, a, b) in out.items())
          + f"  [{len(keys)} x-sections {keys[0]}..{keys[-1]}]")
    return out


def _daily(names) -> dict[str, list[tuple[str, float]]]:
    return {s: C.read(DATA / f"{s}.csv") for s in names}


def first_print(versions) -> None:
    print("NEXT FIRST PRINT, real time (mean predictive quality, first/second half; naive = the last m/m as known)")
    for names, label, target, min_n in ((C.MONTHLY, "CPI components m/m, 9-month window", "cpi_component_mom_first_print", 4),
                                        (MACRO, "twelve macro series m/m, 9-month window", "series_mom_first_print", 6)):
        cross = defaultdict(list)
        for s in names:
            for m, case in first_prints(s).items():
                cross[m].append(case)
        compare(cross, label, target, "mom_pct_change_sa", 30.0, versions, min_n)


def rate_and_level(versions) -> None:
    print("NEXT VALUE, by date (mean predictive quality, first/second half; naive = last value)")
    data = _daily(C.DAILY["fx"] + C.DAILY["oil"])
    rets = {s: [(d, round((b / a - 1) * 100, 3)) for (_, a), (d, b) in zip(rows, rows[1:]) if a] for s, rows in data.items()}
    grid = sorted({d for r in rets.values() for d, _ in r})[20::10]
    idx = {s: {d: i for i, (d, _) in enumerate(r)} for s, r in rets.items()}
    cross = defaultdict(list)
    for d in grid:
        for s, r in rets.items():
            i = idx[s].get(d)
            if i is not None and i >= 20:
                cross[d].append(([x for _, x in r[i - 20:i]], r[i][1]))
    compare(cross, "rate: daily FX / oil returns, 20-day window", "daily_return_pct", "return_pct", 1.0, versions)
    for names, label in ((C.DAILY["rates"], "level: yields, 10-day window"), (C.DAILY["fx"] + C.DAILY["oil"], "level: FX / oil, 10-day window")):
        data = _daily(names)
        dates = sorted({d for rows in data.values() for d, _ in rows})[::10]
        idx = {s: {d: i for i, (d, _) in enumerate(rows)} for s, rows in data.items()}
        cross = defaultdict(list)
        for d in dates:
            for s, rows in data.items():
                i = idx[s].get(d)
                if i is not None and 10 <= i < len(rows) - 1:
                    cross[d].append(([x for _, x in rows[i - 10:i + 1]], rows[i + 1][1]))
        compare(cross, label, "yield_level", "pct", 1.0, versions)


def change(versions) -> None:
    print("CHANGE TARGETS (mean predictive quality against the zero-change rule)")

    def run(label, cases, period):
        line = []
        for name, Sm in versions:
            vals = []
            for cs in cases:
                preds = []
                for w, hd, _ in cs:
                    ser = Sm.Series("x", "lvl", list(w), [str(i) for i in range(len(w))], period, True)
                    preds.append(Sm.forecast(ser, "value_change", hd, {}).value)
                vals.append(pq(preds, [0.0] * len(cs), [a for *_, a in cs]))
            line.append(f"{name} {st.mean(vals):.4f}")
        print(f"  {label:50s} " + "   ".join(line) + f"  [{len(cases)} x-sections]")

    for names, label in ((C.DAILY["rates"], "random walk: yields, 20-day change"), (C.DAILY["fx"] + C.DAILY["oil"], "random walk: FX / oil, 20-day change")):
        data = _daily(names)
        idx = {s: {d: i for i, (d, _) in enumerate(rows)} for s, rows in data.items()}
        cases = []
        for d in sorted({d for rows in data.values() for d, _ in rows})[80::50]:
            cs = []
            for s, rows in data.items():
                i = idx[s].get(d)
                if i is not None and i >= 60 and i + 20 < len(rows):
                    cs.append(([x for _, x in rows[i - 60:i + 1]], 28, rows[i + 20][1] - rows[i][1]))
            if len(cs) >= 3:
                cases.append(cs)
        run(label, cases, 1.0)
    lv = {s: latest_levels(s) for s in ("PAYEMS", "MANEMP", "PCE", "PI", "RSAFS", "DSPIC96", "BUSINV", "TTLCONS")}
    cpi = {name: {d[:7]: v for d, v in C.read(DATA / f"{name}_v20211231.csv")} for name in C.MONTHLY}
    for data, label in ((lv, "trend: macro levels, 1-month change"), (cpi, "trend: CPI indexes, 1-month change")):
        months = sorted({m for v in data.values() for m in v})
        cases = []
        for t in range(30, len(months) - 1, 3):
            cs = []
            for v in data.values():
                w = [v.get(m) for m in months[t - 24:t + 2]]
                if all(x is not None for x in w):
                    cs.append((w[:-1], 30, w[-1] - w[-2]))
            if len(cs) >= 3:
                cases.append(cs)
        run(label, cases, 30.0)


def spearman(a, b) -> float:
    def ranks(x):
        o = sorted(range(len(x)), key=lambda i: x[i])
        r = [0.0] * len(x)
        i = 0
        while i < len(o):
            j = i
            while j + 1 < len(o) and x[o[j + 1]] == x[o[i]]:
                j += 1
            for k in range(i, j + 1):
                r[o[k]] = (i + j) / 2
            i = j + 1
        return r
    ra, rb = ranks(a), ranks(b)
    ma, mb = st.mean(ra), st.mean(rb)
    va, vb = math.sqrt(sum((x - ma) ** 2 for x in ra)), math.sqrt(sum((y - mb) ** 2 for y in rb))
    return sum((x - ma) * (y - mb) for x, y in zip(ra, rb)) / (va * vb) if va and vb else 0.0


def cot(versions, horizons=(2, 5, 8)) -> None:
    rows = defaultdict(list)
    for r in csv.DictReader(open(DATA / "cftc" / "cot_10_markets_2010_2021.csv")):
        rows[r["market"]].append((r["report_date"], int(r["open_interest"]), int(r["noncomm_long"]) - int(r["noncomm_short"])))
    dates = sorted({d for v in rows.values() for d, *_ in v})
    idx = {m: {d: i for i, (d, *_) in enumerate(v)} for m, v in rows.items()}
    W = 26
    cuts = [d for d in dates if "2011-01-01" <= d <= "2021-10-31"][::2]
    print("POSITIONING (ten CFTC markets, 26-week window, change as % of starting open interest; ranking quality with "
          "first/second half, regression quality against zero change, share of forecasts not reverting)")
    for H in horizons:
        for name, Sm in versions:
            rq, ma, mz, halves, moved = [], [], [], ([], []), []
            for ci, d in enumerate(cuts):
                preds, acts, ok = [], [], True
                for m, v in rows.items():
                    i = idx[m].get(d)
                    if i is None or i < W or i + H >= len(v):
                        ok = False
                        break
                    win = v[i - W + 1:i + 1]
                    net, oi = [x[2] for x in win], [x[1] for x in win]
                    acts.append((v[i + H][2] - v[i][2]) / v[i][1] * 100)
                    fc = Sm._ratio_change_forecast(Sm.Series("x", "net_%OI", net, [x[0] for x in win], 7.0, True, oi, 100.0), 7 * H)
                    preds.append(fc.value)
                    moved.append(not fc.method.startswith("level"))
                if not ok:
                    continue
                q = (spearman(preds, acts) + 1) / 2 if len(set(preds)) > 1 else 0.5
                rq.append(q)
                halves[ci >= len(cuts) // 2].append(q)
                ma.append(st.mean(abs(a - b) for a, b in zip(preds, acts)))
                mz.append(st.mean(abs(a) for a in acts))
            n, e = st.mean(mz), st.mean(ma)
            print(f"  {H} weeks {name}: ranking {st.mean(rq):.4f} ({st.mean(halves[0]):.4f}/{st.mean(halves[1]):.4f}), "
                  f"regression {n / (n + e):.4f}, not reverting {st.mean(moved):.3f}  [{len(rq)} cross-sections]")


if __name__ == "__main__":
    v16, v18 = load_stats("v16"), load_stats("v18")
    versions = [("v16", v16), ("v19", S)]
    first_print(versions)
    rate_and_level(versions)
    change(versions)
    cot([("v16", v16), ("v18", v18), ("v19", S)])
