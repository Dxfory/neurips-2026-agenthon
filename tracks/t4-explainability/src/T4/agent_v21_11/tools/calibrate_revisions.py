"""Replay the agent's revision base rate (`t4agent/revisions.py`) on real-time data published by 2021-12-31
(development only; never in the image).

Data: ../calib_data/alfred_rt_<SERIES>.json, the FRED API's real-time periods (ALFRED) for fourteen monthly U.S. series,
vintages 2003-2021 (sources and SHA-256 in ../calib_data/README.md). At every release from 2006 on, the vintage table as
known that day is rebuilt in the shape of the Development unit's tables (the last 5 vintages x the last 8 reference
months), and the agent's `_prior` is asked about the next estimate of the latest and second-latest reference month,
one and two releases ahead. Every 'next estimate' is itself a release on or before 2021-12-23: no task outcome, no
corpus, nothing published after 2021 is read.

Under the interval score the best half-width in a scale family k x s solves sum(s over misses) = 0.1 x sum(s): a
scale-weighted quantile of |error| / s, here with each series weighted equally (v16; v15 used the unweighted quantile,
i.e. 90% coverage, and raised the mean interval score by 8%). The direction rate is checked by its Brier score.

    ../.venv/bin/python tools/calibrate_revisions.py
"""
from __future__ import annotations

import bisect
import datetime as dt
import json
import math
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

AGENT = Path(__file__).resolve().parents[1]
DATA = AGENT.parent / "calib_data"
sys.path.insert(0, str(AGENT))
from t4agent import revisions as R  # noqa: E402

SERIES = ["PAYEMS", "INDPRO", "RSAFS", "HOUST", "DGORDER", "PI", "PCE", "PERMIT", "TTLCONS", "BUSINV", "TCU", "MANEMP",
          "NEWORDER", "DSPIC96"]
K, ROWS = 5, 8


def load(series: str):
    d = json.loads((DATA / f"alfred_rt_{series}.json").read_text())
    by_ref: dict[str, list[tuple[str, str, float]]] = defaultdict(list)
    for o in d["observations"]:
        try:
            by_ref[o["date"][:7]].append((o["realtime_start"], o["realtime_end"], float(o["value"])))
        except ValueError:
            continue  # '.'
    vint = sorted({o["realtime_start"] for o in d["observations"]} - {d["realtime_start"]})  # the window start is no release
    return by_ref, vint


def value_at(rows, d: str):
    for a, b, v in rows:
        if a <= d <= b:
            return v
    return None


def samples(series: str):
    by_ref, vint = load(series)
    refs = sorted(by_ref)
    out = []
    for k in range(K, len(vint) - 2):
        if vint[k] < "2006-01-01":
            continue
        dates = vint[k - K + 1:k + 1]
        known = [r for r in refs if value_at(by_ref[r], vint[k]) is not None]
        window = known[-ROWS:]
        if len(window) < ROWS:
            continue
        rows = {r: [value_at(by_ref[r], d) for d in dates] for r in window}
        ddates = [dt.date.fromisoformat(d) for d in dates]
        for ref in window[-2:]:
            latest = rows[ref][-1]
            for steps in (1, 2):
                nxt = vint[k + steps]
                actual = value_at(by_ref[ref], nxt)
                if actual is None or latest is None:
                    continue
                pr = R._prior(f"ALFRED_{series}_X", ddates, rows, ref, ddates[-1], latest, dt.date.fromisoformat(nxt))
                if pr is None:
                    continue
                out.append({"series": series, "t": vint[k], "steps": pr.steps, "q90": pr.q90, "value": pr.value, "latest": latest,
                            "actual": actual, "p_up": pr.p_up, "n_scale": pr.n_scale, "updates": bool(pr.updates)})
    return out


def q(xs, p):
    s = sorted(xs)
    k = (len(s) - 1) * p
    lo, hi = math.floor(k), math.ceil(k)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def report(rs: list[dict], label: str) -> None:
    r = [abs(x["actual"] - x["value"]) / x["q90"] for x in rs if x["q90"] > 0]
    mae_v = st.mean(abs(x["actual"] - x["value"]) for x in rs) / st.mean(abs(x["latest"]) for x in rs)
    mae_l = st.mean(abs(x["actual"] - x["latest"]) for x in rs) / st.mean(abs(x["latest"]) for x in rs)
    nz = [x for x in rs if x["actual"] != x["latest"]]
    brier = st.mean((x["p_up"] - (x["actual"] > x["latest"])) ** 2 for x in nz) if nz else float("nan")
    base = st.mean(x["actual"] > x["latest"] for x in nz) if nz else float("nan")
    acc = st.mean((x["p_up"] > 0.5) == (x["actual"] > x["latest"]) for x in nz) if nz else float("nan")
    print(f"{label:34s} n={len(rs):5d}  coverage of q90 {st.mean(v <= 1 for v in r):5.1%}  90% quantile of "
          f"|err|/q90 {q(r, 0.9):4.2f}  rel MAE value {mae_v * 1e4:6.1f}bp vs latest {mae_l * 1e4:6.1f}bp  "
          f"direction: up share {base:4.2f}, accuracy {acc:4.2f}, Brier {brier:5.3f} (0.5 -> 0.250)")


def main() -> list[dict]:
    allx = []
    for s in SERIES:
        xs = samples(s)
        allx += xs
        report(xs, s)
    print()
    report(allx, "ALL")
    report([x for x in allx if x["n_scale"] >= R.MIN_SCALE], "  measured scale (n_scale >= 4)")
    report([x for x in allx if x["n_scale"] < R.MIN_SCALE], "  estimated scale (n_scale < 4)")
    for s in (1, 2):
        report([x for x in allx if x["steps"] == s], f"  {s} release(s) ahead")
    report([x for x in allx if x["updates"]], "  a benchmark update left out")
    bins = defaultdict(list)
    for x in allx:
        if x["actual"] != x["latest"]:
            bins[min(9, int(x["p_up"] * 10))].append(x["actual"] > x["latest"])
    print("reliability (p_up bin: share revised up, n):",
          {f"{b / 10:.1f}": (round(st.mean(v), 2), len(v)) for b, v in sorted(bins.items())})
    return allx


def is_factors(allx: list[dict], fit=lambda x: True) -> dict[int, float]:
    """v16: interval-score-optimal envelope factors (one release / more), each series weighted equally, the centre being
    the latest estimate; the envelope here is the one `_prior` returns divided by its current factor."""
    out = {}
    for steps in (1, 2):
        sub = [x for x in allx if fit(x) and min(2, x["steps"]) == steps and x["q90"] > 0]
        f = R.SCALE_ONE_RELEASE if steps == 1 else R.SCALE_MORE_RELEASES
        by = defaultdict(list)
        for x in sub:
            by[x["series"]].append(x)
        pairs = []
        for xs in by.values():
            m = st.mean(x["q90"] / f for x in xs)
            pairs += [(abs(x["actual"] - x["latest"]) / (x["q90"] / f), x["q90"] / f / m) for x in xs]
        pairs.sort()
        tot, acc = sum(w for _, w in pairs), 0.0
        for r, w in pairs:
            acc += w
            if acc >= 0.9 * tot:
                out[steps] = r
                break
    return out


if __name__ == "__main__":
    allx = main()
    print("interval-score-optimal envelope factors (one release, more):", {k: round(v, 2) for k, v in is_factors(allx).items()})
    for lab, fit in (("2006-2013", lambda x: x["t"] < "2014"), ("2014-2021", lambda x: x["t"] >= "2014")):
        print(f"  fitted on {lab}:", {k: round(v, 2) for k, v in is_factors(allx, fit).items()})
