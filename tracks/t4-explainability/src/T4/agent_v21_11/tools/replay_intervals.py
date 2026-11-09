"""Replay the code path's point and interval on public FRED series published before 2022 (development only; never in
the image).

For each series, many origins and the horizons a unit of its frequency would ask (next print, a week, a month, a
quarter), the series up to the origin is written as the 'date | value' table an entity's own document would carry, and
the agent's own code produces the statistical baseline (`stats.entity_forecast`), the table scale
(`stats.series_half_width`) and the submitted range (`calibrate.quantity_range`), exactly as on a unit without a model
reply. Each is scored by scorer 5.2.2's formulas against the naive rule a unit would declare (last value for a level,
zero for a change over the horizon, the last period's change for a next-period percent change):
  PQ = naive_MAE / (naive_MAE + MAE) and IQ = naive_IS / (naive_IS + IS), IS = width + 20 x miss distance,
with the naive interval sized like the stress units' (+/- 1.645 x the SD of the naive's own errors over the visible
window). Variants scale the submitted half-width, or shrink the point towards the naive, to show where the rules are
off. Every outcome resolved before 2022 (TRAINING-POLICY).

    ../.venv/bin/python tools/replay_intervals.py                # fetch (cached under ../units_pre2022/raw) and replay
"""
from __future__ import annotations

import datetime as dt
import math
import random
import statistics as st
import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import build_pre2022_units as B  # noqa: E402  (FRED access, cached)
from t4agent.artifact_policy import ArtifactPolicy  # noqa: E402
from t4agent.calibrate import quantity_range  # noqa: E402
from t4agent.predict import Forecast  # noqa: E402
from t4agent.stats import entity_forecast, find_series, series_half_width  # noqa: E402

DAILY = ["DGS3MO", "DGS2", "DGS5", "DGS10", "DGS30", "T10Y2Y", "T10Y3M", "BAA10Y", "AAA10Y", "DEXUSEU", "DEXJPUS",
         "DEXUSUK", "DEXCAUS", "DEXUSAL", "DEXSZUS", "DEXMXUS", "DEXKOUS", "DEXINUS", "DEXBZUS", "DCOILWTICO",
         "DCOILBRENTEU", "DHHNGSP", "VIXCLS", "NASDAQCOM", "DFF", "TEDRATE", "DTB3", "DAAA", "DBAA", "T10YIE"]
WEEKLY = ["ICSA", "CCSA", "IURSA", "GASREGW", "GASDESW", "MORTGAGE30US", "MORTGAGE15US", "WALCL", "TOTBKCR", "NFCI",
          "WM2NS", "ICNSA"]
MONTHLY = ["UNRATE", "U6RATE", "PAYEMS", "MANEMP", "USCONS", "CPIAUCSL", "CPILFESL", "CPIENGSL", "PCEPI", "PCEPILFE",
           "INDPRO", "IPMAN", "TCU", "RSAFS", "RSXFS", "HOUST", "PERMIT", "HSN1F", "UMCSENT", "PPIACO", "CES0500000003",
           "AWHAETP", "CIVPART", "EMRATIO", "DSPIC96", "PSAVERT", "TOTALSA", "BUSINV", "ISRATIO", "DGORDER", "NEWORDER",
           "M2SL", "CSUSHPINSA", "CAUR", "TXUR", "NYUR", "FLUR", "ILUR", "PAUR", "OHUR", "MIUR", "GAUR", "NCUR"]
WINDOW = {"D": 60, "W": 26, "M": 24}  # observations a unit's own table typically shows
HORIZONS = {"D": [(1, 1), (7, 5), (30, 21), (91, 63)], "W": [(1, 1), (28, 4)], "M": [(2, 1), (91, 3)]}
ORIGINS = 40
Z90 = 1.645


def load(sid: str) -> tuple[str, list[tuple[str, float]]]:
    info = B.series_info(sid)
    xs = B.obs(sid, "1995-01-01", "2021-12-31")
    return info["frequency_short"], xs


def scenarios(freq: str, xs: list[tuple[str, float]], rnd: random.Random):
    """(window, truth index, horizon days, steps) for origins spread over the history."""
    n, win = len(xs), WINDOW[freq]
    for days, steps in HORIZONS[freq]:
        hi = n - steps - 1
        if hi <= win + 30:
            continue
        for t in sorted(rnd.sample(range(win + 30, hi), min(ORIGINS, hi - win - 30))):
            yield xs[t - win + 1:t + 1], xs[:t + 1], t + steps, days, steps


def table(window) -> SimpleNamespace:
    text = "date | value\n" + "\n".join(f"{d} | {v:g}" for d, v in window)
    return SimpleNamespace(docs={"d": SimpleNamespace(text=text)})


def score(point, lo, hi, y, npoint, nlo, nhi):
    is_ = (hi - lo) + 20 * (max(0.0, lo - y) + max(0.0, y - hi))
    nis = (nhi - nlo) + 20 * (max(0.0, nlo - y) + max(0.0, y - nhi))
    return abs(point - y), abs(npoint - y), is_, nis


def run_case(kind: str, window, full, y_index, days, steps, xs):
    """kind: 'level' (value at the horizon), 'change' (difference over the horizon), 'pct' (next period's percent change)."""
    vals = [v for _, v in full]
    last = vals[-1]
    if kind == "pct":
        if min(v for _, v in window) <= 0 or steps != 1:
            return None
        target, unit = "series_mom_pct_first_print", "percent change"
        y = 100 * (xs[y_index][1] / last - 1)
        npoint = 100 * (vals[-1] / vals[-2] - 1)
        hist = [100 * (b / a - 1) for a, b in zip(vals[-25:-1], vals[-24:])]
        nerr = [b - a for a, b in zip(hist, hist[1:])]
    elif kind == "change":
        target, unit = "series_change_over_horizon", "difference"
        y = xs[y_index][1] - last
        npoint = 0.0
        nerr = [vals[i] - vals[i - steps] for i in range(max(steps, len(vals) - 60), len(vals))]
    else:
        target, unit = "series_level_on_target_date", "level"
        y = xs[y_index][1]
        npoint = last
        nerr = [vals[i] - vals[i - steps] for i in range(max(steps, len(vals) - 60), len(vals))]
    if len(nerr) < 5:
        return None
    nhalf = Z90 * st.pstdev(nerr) or 1e-9
    ent = {"entity_id": "X", "units": unit}
    if kind == "level":
        ent["latest_value"] = last  # a unit's row usually carries the latest value (the naive)
    elif kind == "pct":
        ent["latest_mom_pct"] = round(npoint, 4)  # and the latest period's change
    corpus = table(window)
    # v21.6: derive eligibility per origin; never borrow later calibrated methods.
    # Observation dates remain proxies for release dates in this development tool,
    # so its replay is not vintage-correct acceptance evidence.
    policy = ArtifactPolicy(dt.date.fromisoformat(window[-1][0]))
    sf = entity_forecast(corpus, ["d"], ent, target, days, {"d"}, allow_own=True, policy=policy)
    if sf is None:
        return None
    s = find_series(corpus, ["d"], ent, target, {"d"}, True)
    th = series_half_width(s, target, days, ent, policy=policy) if s else None
    spec = SimpleNamespace(target_name=target, target_type="regression", prompt="", entities=[ent], labels=[],
                           horizon_days=days)
    fc = Forecast(eid="X", source="stat", median=sf.value, stat_q90=sf.q90, stat_reliable=sf.mae is not None)
    rng = quantity_range(spec, fc, None, ent=ent, table_half=th)
    if rng is None:
        return None
    point, lo, hi = rng
    return {"y": y, "point": point, "lo": lo, "hi": hi, "npoint": npoint, "nlo": npoint - nhalf, "nhi": npoint + nhalf,
            "method": sf.method}


def main() -> int:
    rnd = random.Random(2021)
    rows = []
    for group, ids in (("D", DAILY), ("W", WEEKLY), ("M", MONTHLY)):
        for sid in ids:
            try:
                freq, xs = load(sid)
            except Exception as exc:  # noqa: BLE001 - a missing series is skipped
                print(f"skip {sid}: {type(exc).__name__}")
                continue
            if freq not in WINDOW or len(xs) < 200:
                continue
            for window, full, yi, days, steps in scenarios(freq, xs, rnd):
                for kind in ("level", "change", "pct"):
                    r = run_case(kind, window, full, yi, days, steps, xs)
                    if r:
                        rows.append({**r, "sid": sid, "freq": freq, "kind": kind, "steps": steps})
    print(f"{len(rows)} cases from {len({r['sid'] for r in rows})} series")

    def summary(sel, k=1.0, w=1.0):
        pq, iq = [], []
        for r in sel:
            p = r["npoint"] + w * (r["point"] - r["npoint"])
            lo = p - k * (r["point"] - r["lo"])
            hi = p + k * (r["hi"] - r["point"])
            e, ne, is_, nis = score(p, lo, hi, r["y"], r["npoint"], r["nlo"], r["nhi"])
            pq.append((e, ne))
            iq.append((is_, nis))
        # unit-like aggregation: groups of 8 cases of one series and horizon, scored like a unit, then averaged
        PQ = sum(ne for _, ne in pq) / max(1e-12, sum(ne for _, ne in pq) + sum(e for e, _ in pq))
        IQ = sum(n for _, n in iq) / max(1e-12, sum(n for _, n in iq) + sum(i for i, _ in iq))
        cov = st.mean(r["lo"] <= r["y"] <= r["hi"] for r in sel) if k == 1.0 and w == 1.0 else None
        return PQ, IQ, cov

    def unitwise(sel, k=1.0, w=1.0):
        """Mean over (series, kind, steps) groups of the scorer's per-unit ratios (rows within a group pooled)."""
        groups: dict = {}
        for r in sel:
            groups.setdefault((r["sid"], r["kind"], r["steps"]), []).append(r)
        pqs, iqs = [], []
        for g in groups.values():
            PQ, IQ, _ = summary(g, k, w)
            pqs.append(PQ)
            iqs.append(min(IQ, max(0.5, PQ)))  # the scorer's cap
        return st.mean(pqs), st.mean(iqs), len(groups)

    print(f"\n{'subset':16s} {'groups':>6s} {'PQ':>6s} {'IQ':>6s} {'cover':>6s} | IQ at half-width x 0.7 0.85 1.0 1.2 1.5 "
          f"| PQ with point shrunk to naive w=0 .25 .5 .75 1")
    for name, sel in [("all", rows)] + [(f"{f}-{k}", [r for r in rows if r["freq"] == f and r["kind"] == k])
                                        for f in ("D", "W", "M") for k in ("level", "change", "pct")]:
        if len(sel) < 20:
            continue
        PQ, IQ, n = unitwise(sel)
        cov = st.mean(r["lo"] <= r["y"] <= r["hi"] for r in sel)
        iqk = [unitwise(sel, k=k)[1] for k in (0.7, 0.85, 1.0, 1.2, 1.5)]
        pqw = [unitwise(sel, w=w)[0] for w in (0.0, 0.25, 0.5, 0.75, 1.0)]
        print(f"{name:16s} {n:6d} {PQ:6.3f} {IQ:6.3f} {cov:6.2f} | " + " ".join(f"{x:.3f}" for x in iqk)
              + " | " + " ".join(f"{x:.3f}" for x in pqw))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
