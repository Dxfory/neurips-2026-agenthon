"""Calibrate the agent's interval scales on public observations through 2021 (complete daily releases by early January 2022) (development only).

The Track 4 training policy permits interval calibration on public, licensed data whose features and labels were
available before every task cutoff (the earliest Development cutoff is 2022-07-28). The data are U.S. government
series downloaded from FRED / ALFRED into ../calib_data (sources, vintages and SHA-256 in its README.md); nothing
here reads a task outcome or a corpus.

Under the scorer's interval score (width + 20 x miss distance, 90% level) the best symmetric half-width for a scale
family half = k * s minimises the mean of 2 k s + 20 max(0, |e| - k s): k solves sum(s over misses) = 0.1 x sum(s), a
SCALE-WEIGHTED 90% quantile of |e| / s (v16; v15 used the unweighted quantile -- the coverage criterion -- which
overstates k when scales vary). Each series is weighted equally (its scales divided by their mean). Measured on the
agent's own scale estimators replayed over history:

  daily  -- a short window of n daily closes, extrapolated as a random walk to an h-trading-day change:
            s = rms(last n daily changes) * sqrt(h); the agent used k = 1.645 (normal theory with the window's rms
            taken as exact), times 1.25 in one of its two estimates.
  monthly -- a 9-month window of a seasonally adjusted month-over-month percent change (CPI components), next value:
            the agent's statistical baseline (`stats.forecast`) and its two scales (the backtest's error quantile with
            its small-sample inflation, and 1.645 x the window's step rms), combined geometrically as
            `calibrate.quantity_range` does.

    ../.venv/bin/python tools/calibrate_scales.py
"""
from __future__ import annotations

import csv
import datetime as dt
import math
import statistics as st
import sys
from pathlib import Path

AGENT = Path(__file__).resolve().parents[1]
DATA = AGENT.parent / "calib_data"
sys.path.insert(0, str(AGENT))
from t4agent.stats import Series, forecast, series_half_width  # noqa: E402

DAILY = {"rates": ["DGS3MO", "DGS1", "DGS2", "DGS3", "DGS5", "DGS7", "DGS10", "DGS20", "DGS30"],
         "fx": ["DEXUSEU", "DEXJPUS", "DEXUSUK", "DEXCAUS"], "oil": ["DCOILWTICO"]}
MONTHLY = ["CPIAUCSL", "CPILFESL", "CUSR0000SAH1", "CPIENGSL", "CUSR0000SETB01", "CPIUFDSL", "CUSR0000SETA01",
           "CUSR0000SETA02", "CPIAPPSL", "CPIMEDSL", "CUSR0000SAS4"]
NS = (5, 10, 20, 40, 60)
HS = (1, 5, 10, 20, 40, 60)


def read(path: Path) -> list[tuple[str, float]]:
    out = []
    with open(path) as fh:
        r = csv.reader(fh)
        next(r)
        for row in r:
            try:
                out.append((row[0], float(row[1])))
            except (ValueError, IndexError):
                continue  # FRED marks holidays with '.'
    return out


def q(xs: list[float], p: float) -> float:
    s = sorted(xs)
    k = (len(s) - 1) * p
    lo, hi = math.floor(k), math.ceil(k)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def daily_ratios(values: list[float], n: int, h: int, step: int = 5) -> list[float]:
    d = [b - a for a, b in zip(values, values[1:])]
    out = []
    for t in range(n, len(values) - h, step):  # origin index t: the window is the n changes ending at values[t]
        w = d[t - n:t]
        rms = math.sqrt(sum(x * x for x in w) / n)
        if rms > 0:
            out.append(abs(values[t + h] - values[t]) / (rms * math.sqrt(h)))
    return out


def daily_samples(values: list[float], n: int, h: int, step: int = 5) -> tuple[list[float], list[float]]:
    """(|h-step change| / s, s) with s = rms of the n changes before the origin x sqrt(h), every `step` origins."""
    d = [b - a for a, b in zip(values, values[1:])]
    rr, ss = [], []
    for t in range(n, len(values) - h, step):
        w = d[t - n:t]
        rms = math.sqrt(sum(x * x for x in w) / n)
        if rms > 0:
            sc = rms * math.sqrt(h)
            rr.append(abs(values[t + h] - values[t]) / sc)
            ss.append(sc)
    return rr, ss


def weighted_quantile(pairs: list[tuple[float, float]], p: float = 0.9) -> float:
    pairs = sorted(pairs)
    tot, acc = sum(w for _, w in pairs), 0.0
    for x, w in pairs:
        acc += w
        if acc >= p * tot:
            return x
    return pairs[-1][0]


def k_is(series: list[list[float]], n: int, h: int) -> float:
    """The interval-score-optimal multiplier: scale-weighted 90% quantile, each series weighted equally."""
    pairs = []
    for v in series:
        rr, ss = daily_samples(v, n, h)
        if ss:
            m = sum(ss) / len(ss)
            pairs += [(r, x / m) for r, x in zip(rr, ss)]
    return weighted_quantile(pairs)


def daily_is() -> dict:
    """v16: the table in t4agent/stats.py (_RW_K)."""
    series = [[v for _, v in read(DATA / f"{s}.csv")] for names in DAILY.values() for s in names]
    print("DAILY (v16, interval-score optimal k):")
    table = {}
    for n in NS:
        table.update({(n, h): round(k_is(series, n, h), 2) for h in HS + (90, 120)})
        print(f"  n={n:2d} " + " ".join(f"h={h}: {table[(n, h)]:.2f}" for h in HS + (90, 120)))
    return table


def daily() -> dict:
    table = {}
    print("DAILY (v15 diagnostic, coverage): 90% quantile of |h-day change| / (rms of the last n daily changes x sqrt(h))")
    print(f"{'n':>3s} " + " ".join(f"{'h=' + str(h):>16s}" for h in HS))
    series = {c: [[v for _, v in read(DATA / f"{s}.csv")] for s in names] for c, names in DAILY.items()}
    for n in NS:
        row = []
        for h in HS:
            pooled, by_class = [], {}
            for c, lists in series.items():
                rs = [r for v in lists for r in daily_ratios(v, n, h)]
                by_class[c] = q(rs, 0.9)
                pooled += rs
            table[(n, h)] = q(pooled, 0.9)
            row.append(f"{table[(n, h)]:5.2f} ({'/'.join(f'{by_class[c]:.2f}' for c in series)})")
        print(f"{n:3d} " + " ".join(f"{x:>16s}" for x in row))
    print("   (pooled, and rates/fx/oil)")
    for label, (a, b) in {"1990-2005": ("1990", "2005-12-31"), "2006-2021": ("2006", "2021-12-31")}.items():
        rs = []
        for names in DAILY.values():
            for s in names:
                vals = [v for d, v in read(DATA / f"{s}.csv") if a <= d <= b]
                rs += daily_ratios(vals, 10, 34)
        print(f"   n=10, h=34 trading days, {label}: {q(rs, 0.9):.2f}")
    return table


def mom(path: Path) -> list[float]:
    lv = [v for _, v in read(path)]
    return [round((b / a - 1.0) * 100.0, 2) for a, b in zip(lv, lv[1:])]


def monthly(window: int = 9) -> dict:
    """Replays stats.forecast and series_half_width on rolling windows (first origin 2000) of each component."""
    r_bt, r_tab, r_own, cover = [], [], [], []
    for s in MONTHLY:
        m = mom(DATA / f"{s}_v20211231.csv")
        start = 120  # 2000-02: ten years of history before the first origin, none of it used but the window
        for t in range(max(start, window), len(m)):
            w = m[t - window:t]
            ser = Series(s, "m/m %", w, [f"{i}" for i in range(window)], 30.0, True)
            ent = {"entity_id": s, "unit": "mom_pct_change_sa"}
            try:
                f = forecast(ser, "cpi_component_mom_first_print", 30, ent)
            except (ValueError, ZeroDivisionError):
                continue
            # Development calibration deliberately inspects the admitted backend. This is
            # retrospective calibration, not an as-of-origin performance claim.
            from t4agent.artifact_policy import ArtifactPolicy
            tab = series_half_width(ser, "cpi_component_mom_first_print", 30, ent,
                                    policy=ArtifactPolicy(dt.date(2022, 2, 1)))
            err = abs(m[t] - f.value)
            scales = [x for x in (f.q90, tab) if x and x > 0]
            if not scales:
                continue
            own = math.sqrt(scales[0] * scales[1]) if len(scales) == 2 else scales[0]
            if f.q90:
                r_bt.append(err / f.q90)
            if tab:
                r_tab.append(err / tab)
            r_own.append(err / own)
            cover.append(err <= own)
    qq = lambda xs: q(xs, 0.9) if xs else float("nan")  # noqa: E731
    print(f"MONTHLY ({window}-month window, CPI components, {len(r_own)} forecasts, {len(r_bt)} with a backtest scale): "
          f"coverage of the agent's 90% half-width {sum(cover) / len(cover):.1%}; 90% quantile of |error| / scale: "
          f"backtest q90 {qq(r_bt):.2f}, step scale {qq(r_tab):.2f}, combined {qq(r_own):.2f}")
    return {"bt": qq(r_bt), "tab": qq(r_tab), "own": qq(r_own), "coverage": sum(cover) / len(cover)}


if __name__ == "__main__":
    daily_is()
    for w in (6, 9, 12, 24):
        monthly(w)
