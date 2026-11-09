"""Calibrate the pump-price nowcast of CPI gasoline, energy and all items (`t4agent/nowcast.py`) on public data
published by 2021 (development only).

Data (../calib_data, sources and SHA-256 in its README): GASREGW_1990_2021.json, the EIA's weekly U.S. regular
gasoline retail price via FRED (market data, not revised), and the ALFRED real-time vintages of the three CPI series
(alfred_rt_<SERIES>.json). The target is each month's FIRST PRINT of the seasonally adjusted month-over-month change,
as in the Development CPI unit; its driver is the change in the average weekly pump price from the previous calendar
month to the target month (not seasonally adjusted), from the weeks of the target month available (2, 3 or all).

    first print = BETA x driver + GAP[calendar month]

GAP absorbs the seasonal factors (pump prices rise every spring and fall every autumn before adjustment) and, for
energy and all items, the components the pump price does not move. Every forecast below is out of sample: BETA and GAP
are fitted on reference months before the one forecast (at least 36, with three of the same calendar month). The
printed constants are fitted on every month and go into t4agent/nowcast.py; the 90% half-widths are the
interval-score-optimal quantiles of the out-of-sample errors.

    ../.venv/bin/python tools/calibrate_nowcast.py
"""
from __future__ import annotations

import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
AGENT = HERE.parent
DATA = AGENT.parent / "calib_data"
sys.path[:0] = [str(HERE), str(AGENT)]
import calibrate_methods as M  # noqa: E402
from t4agent import stats as S  # noqa: E402

SERIES = {"CUSR0000SETB01": "gasoline", "CPIENGSL": "energy", "CPIAUCSL": "all items"}
WEEKS = (2, 3, None)  # weeks of the target month seen; None = all


def weekly() -> dict[str, list[float]]:
    """Calendar month -> weekly prices in date order."""
    d = json.loads((DATA / "GASREGW_1990_2021.json").read_text())
    out: dict[str, list[float]] = defaultdict(list)
    for o in sorted(d["observations"], key=lambda o: o["date"]):
        if o["value"] != ".":
            out[o["date"][:7]].append(float(o["value"]))
    return out


def driver(w: dict[str, list[float]], month: str, weeks: int | None) -> float | None:
    cur, prev = w.get(month, []), w.get(M._prev(month), [])
    if weeks is not None:
        if len(cur) < weeks:
            return None
        cur = cur[:weeks]
    if len(cur) < 2 or len(prev) < 3:
        return None
    return (st.fmean(cur) / st.fmean(prev) - 1) * 100


def fit(rows: list[tuple[str, float, float]], beta: float | None) -> tuple[float, dict[str, float]] | None:
    """(BETA, GAP by calendar month) by least squares with calendar-month intercepts; BETA given or fitted."""
    by = defaultdict(list)
    for m, d, y in rows:
        by[m[5:]].append((d, y))
    if len(by) < 12 or min(len(v) for v in by.values()) < 3:
        return None
    if beta is None:
        num = sum((d - st.fmean(x for x, _ in v)) * (y - st.fmean(z for _, z in v)) for v in by.values() for d, y in v)
        den = sum((d - st.fmean(x for x, _ in v)) ** 2 for v in by.values() for d, _ in v)
        beta = num / den
    return beta, {k: st.fmean(y - beta * d for d, y in v) for k, v in sorted(by.items())}


def replay(series: str, beta: float | None, weeks: int | None, w=None):
    """Out-of-sample nowcasts: (month, first print, last first print, nowcast, AR(1))."""
    w = w or weekly()
    fp = M.first_prints(series)
    full = [(m, driver(w, m, None), fp[m][1]) for m in sorted(fp)]
    full = [r for r in full if r[1] is not None]
    out = []
    for m in sorted(fp):
        d = driver(w, m, weeks)
        past = [r for r in full if r[0] < m]
        if d is None or len(past) < 36:
            continue
        f = fit(past, beta)
        if f is None:
            continue
        b, gap = f
        window, y = fp[m]
        ser = S.Series("x", "value", list(window), [str(j) for j in range(len(window))], 30.0, True)
        ar1 = S.forecast(ser, "cpi_component_mom_first_print", 30, {"unit": "mom_pct_change_sa"}).value
        out.append((m, y, window[-1], b * d + gap[m[5:]], ar1))
    return out


def quality(rows, col: int) -> float:
    n = st.fmean(abs(r[1] - r[2]) for r in rows)
    e = st.fmean(abs(r[1] - r[col]) for r in rows)
    return n / (n + e)


def q90(errors: list[float]) -> float:
    s = sorted(abs(x) for x in errors)
    k = (len(s) - 1) * 0.9
    lo = int(k)
    return s[lo] + (s[min(lo + 1, len(s) - 1)] - s[lo]) * (k - lo)


def main() -> dict:
    w = weekly()
    consts = {}
    for series, name in SERIES.items():
        beta0 = 1.0 if name == "gasoline" else None  # the CPI gasoline index is the pump price: its slope is fixed at 1
        rows = replay(series, beta0, None, w)
        half = len(rows) // 2
        print(f"{name} ({series}): {len(rows)} out-of-sample months {rows[0][0]}..{rows[-1][0]}; quality vs the last first "
              f"print -- nowcast {quality(rows, 3):.3f} ({quality(rows[:half], 3):.3f}/{quality(rows[half:], 3):.3f}), "
              f"AR(1) {quality(rows, 4):.3f} ({quality(rows[:half], 4):.3f}/{quality(rows[half:], 4):.3f}); "
              f"MAE nowcast {st.fmean(abs(r[1] - r[3]) for r in rows):.3f}, AR(1) {st.fmean(abs(r[1] - r[4]) for r in rows):.3f}")
        for f_ in (0.75, 0.5):
            mix = [(m, y, last, f_ * nc + (1 - f_) * ar, ar) for m, y, last, nc, ar in rows]
            print(f"    {f_:.2f} nowcast + {1 - f_:.2f} AR(1): {quality(mix, 3):.3f}")
        errs = {}
        for wk in WEEKS:
            rs = rows if wk is None else replay(series, beta0, wk, w)
            errs[wk or 4] = [r[1] - r[3] for r in rs]
            print(f"    weeks of the target month seen: {wk or 'all'}: {len(rs)} months, quality {quality(rs, 3):.3f}, "
                  f"MAE {st.fmean(abs(r[1] - r[3]) for r in rs):.3f}, 90% half-width {q90(errs[wk or 4]):.3f}")
        widths = {2: q90(errs[2]), 3: q90(errs[3] + errs[4])}  # two weeks; three or more (pooled)
        print(f"    90% half-widths used: two weeks {widths[2]:.3f}, three or more {widths[3]:.3f}")
        fp = M.first_prints(series)
        full = [(m, driver(w, m, None), fp[m][1]) for m in sorted(fp)]
        full = [r for r in full if r[1] is not None]
        b, gap = fit(full, beta0)
        # time split of the fitted constants: fit on one half of the months, score the other half
        h = len(full) // 2
        for a_, b_, lab in ((full[:h], full[h:], "first half -> second"), (full[h:], full[:h], "second half -> first")):
            fb, fg = fit(a_, beta0)
            sc = [(m, y, fp[m][0][-1], fb * d + fg[m[5:]], 0.0) for m, d, y in b_]
            print(f"    fitted on the {lab}: BETA {fb:.3f}, quality {quality(sc, 3):.3f}")
        consts[series] = {"beta": round(b, 3), "gap": {k: round(v, 2) for k, v in gap.items()},
                          "q90": {k: round(v, 2) for k, v in widths.items()}, "mae": round(st.fmean(abs(r[1] - r[3]) for r in rows), 3),
                          "months": f"{full[0][0]}..{full[-1][0]}", "n": len(full)}
        print(f"    constants (all {len(full)} months {full[0][0]}..{full[-1][0]}): BETA {b:.3f}, GAP "
              + " ".join(f"{k}:{v:+.2f}" for k, v in gap.items()))
    print(json.dumps(consts))
    return consts


if __name__ == "__main__":
    main()
