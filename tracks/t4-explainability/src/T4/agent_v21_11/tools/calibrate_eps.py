"""Calibrate the EPS anchor (`t4agent/eps.py`) on SEC EDGAR filings made before 2022 (development only).

Data: ../calib_data/sec/eps_<TICKER>.json, the SEC XBRL `companyconcept` API's us-gaap EarningsPerShareDiluted for ~145
large U.S. companies (public data; sources and SHA-256 in ../calib_data/README.md). Only facts FILED before 2022-01-01
are read. For every quarter T reported 2010-2021 the Development units' setting is rebuilt: the base is T's prior-year
quarter as first reported, the latest information is the most recent 10-Q filed before T's own report (its quarter and
the prior-year quarter printed beside it), k = quarters from that 10-Q to T, and the outcome is T's first-reported EPS.

v20 (review of v18): a pseudo-forecast is kept only when its base and its outcome are on one share basis. v18 paired
the prior-year quarter as first reported with T as first reported, so a split or restatement in between compared
numbers on different bases (AAPL, quarter ended 2014-09-27: 1.42 against 8.26 reported before the 7:1 split; 1.18
after it). T's own filing prints the prior-year quarter beside T: when that comparative differs from the base by more
than 20% of the larger of the two (at least 0.05) the case is dropped, and so is a case carrying an XBRL scale error
(a figure more than 50 times the company's median |EPS|, e.g. 120,000,000 for 1.20). The comparative only decides
which cases are comparable; it is never a feature, since it was published with the outcome.

  point      scorer 5.2.2 predictive quality against the seasonal random walk (the prior-year quarter), per calendar
             quarter cross-section, for carries of the latest year-over-year change: by k, by half-period, and in both
             target units (the EPS level; the percent change against a positive base, as the bank unit asks)
  direction  P(T above its prior-year quarter | the latest change rose / fell), by half-period
  interval   the interval-score-optimal factor on the anchor's rule-based 90% half-width (scale-weighted quantile,
             companies weighted equally), by k and half-period, and the mean interval score of each half's factor on
             the other half against v18-v19's 1.8

    ../.venv/bin/python tools/calibrate_eps.py
"""
from __future__ import annotations

import datetime as dt
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
AGENT = HERE.parent
SEC = AGENT.parent / "calib_data" / "sec"
sys.path.insert(0, str(AGENT))
from t4agent import eps as E  # noqa: E402

BASIS_TOL = 0.20  # base vs the comparative in T's own filing
SCALE_ERROR = 50.0  # a figure this many times the company's median |EPS| is an XBRL scale error


def _d(s: str) -> dt.date:
    return dt.date.fromisoformat(s)


def _quarterly(f: Path) -> list[dict]:
    d = json.loads(f.read_text())
    facts = [x for x in (d.get("units", {}).get("USD/shares") or []) if x.get("filed", "9999") < "2022-01-01" and x.get("start")]
    return [x for x in facts if 80 <= (_d(x["end"]) - _d(x["start"])).days <= 100]


def cases(clean: bool = True) -> list[dict]:
    """The pseudo-forecasts; `clean` drops those whose base and outcome are on different bases, and scale errors."""
    out = []
    for f in sorted(SEC.glob("eps_*.json")):
        q = _quarterly(f)
        first: dict[str, dict] = {}
        for x in q:
            if x["end"] not in first or x["filed"] < first[x["end"]]["filed"]:
                first[x["end"]] = x
        by_accn = defaultdict(list)
        for x in q:
            by_accn[x["accn"]].append(x)
        filings = []
        for xs in by_accn.values():
            if not xs[0].get("form", "").startswith("10-Q"):
                continue
            cur_end = max(x["end"] for x in xs)
            cur = [x for x in xs if x["end"] == cur_end]
            pri = [x for x in xs if abs((_d(cur_end) - _d(x["end"])).days - 365) <= 10]
            if cur and pri:
                filings.append((min(x["filed"] for x in cur), cur_end, float(cur[0]["val"]), float(pri[0]["val"])))
        filings.sort()
        rows = []
        for tend, tx in first.items():
            t = _d(tend)
            if not "2010-01-01" <= tend <= "2021-09-30":
                continue
            base_end = [e for e in first if abs((t - _d(e)).days - 365) <= 10]
            prior = [fl for fl in filings if fl[0] < tx["filed"] and _d(fl[1]) < t]
            if not base_end or not prior:
                continue
            fl = prior[-1]
            gap = (t.year * 12 + t.month) - (_d(fl[1]).year * 12 + _d(fl[1]).month)
            k = round(gap / 3)
            if not (1 <= k <= 3 and abs(gap - 3 * k) <= 1):
                continue
            comp = [y for y in by_accn[tx["accn"]] if abs((t - _d(y["end"])).days - 365) <= 10]
            rows.append({"t": f.stem[4:], "end": tend, "k": k, "base": float(first[base_end[0]]["val"]), "cur": fl[2],
                         "prior": fl[3], "actual": float(tx["val"]), "comp": float(comp[0]["val"]) if comp else None})
        if not rows:
            continue
        med = max(st.median(abs(r["actual"]) for r in rows), 0.05)
        for r in rows:
            c, b = r["comp"], r["base"]
            r["basis_change"] = c is not None and abs(c - b) > BASIS_TOL * max(abs(c), abs(b), 0.05)
            r["scale_error"] = any(abs(r[x]) > SCALE_ERROR * med for x in ("base", "cur", "prior", "actual"))
        out += [r for r in rows if not clean or not (r["basis_change"] or r["scale_error"])]
    return out


def _cross(cs):
    by = defaultdict(list)
    for c in cs:
        by[(c["end"][:4], (int(c["end"][5:7]) - 1) // 3)].append(c)
    return [v for v in by.values() if len(v) >= 5]


def point_quality(cs, carry, unit: str = "level") -> float:
    """Mean per-cross-section quality of base + carry x (cur - prior) against the base, in EPS ('level') or in
    percent of a positive base ('pct')."""
    if unit == "pct":
        cs = [c for c in cs if c["base"] > 0]
    vals = []
    for g in _cross(cs):
        s = (lambda c: 1.0 / c["base"]) if unit == "pct" else (lambda c: 1.0)
        n = st.mean(abs(c["actual"] - c["base"]) * s(c) for c in g)
        e = st.mean(abs(c["actual"] - (c["base"] + carry(c) * (c["cur"] - c["prior"]))) * s(c) for c in g)
        vals.append(n / (n + e) if n + e > 0 else 0.5)
    return st.mean(vals)


def anchor_half(c) -> tuple[float, float]:
    """(level, 90% half-width) of the anchor with its rule-based scale (EPS_SCALE_* = 1), as t4agent/eps.py builds it."""
    carry = E.PHI if c["k"] == 1 else E.CARRY_FAR
    level = c["base"] + carry * (c["cur"] - c["prior"])
    sd = max(E.REL_SD * max(abs(level), abs(c["base"])), 0.5 * abs(c["cur"] - c["prior"]) * E.PHI ** (c["k"] - 1), 0.02)
    return level, 1.645 * sd


def _pairs(cs):
    """(|error| / half-width, half-width / the company's mean half-width): companies weighted equally."""
    by = defaultdict(list)
    for c in cs:
        by[c["t"]].append(c)
    out = []
    for xs in by.values():
        hs = [anchor_half(c) for c in xs]
        m = st.mean(h for _, h in hs)
        out += [(abs(c["actual"] - lv) / h, h / m) for c, (lv, h) in zip(xs, hs)]
    return out


def interval_factor(cs) -> float:
    pairs = sorted(_pairs(cs))
    tot, acc = sum(w for _, w in pairs), 0.0
    for r, w in pairs:
        acc += w
        if acc >= 0.9 * tot:
            return r
    return pairs[-1][0]


def mean_interval_score(cs, factor: float) -> float:
    """Mean interval score of the half-width factor x h, in units of the company's mean half-width."""
    return st.mean(w * (2 * factor + 20 * max(0.0, r - factor)) for r, w in _pairs(cs))


HALVES = (("2010-2015", lambda c: c["end"] < "2016"), ("2016-2021", lambda c: c["end"] >= "2016"), ("all", lambda c: True))


def main() -> list[dict]:
    raw = cases(clean=False)
    cs = [c for c in raw if not (c["basis_change"] or c["scale_error"])]
    bc = [c for c in raw if c["basis_change"]]
    print(f"{len(raw)} pseudo-forecasts from {len({c['t'] for c in raw})} companies; dropped {len(bc)} with a basis change "
          f"({sum((c['actual'] > c['base']) != (c['actual'] > c['comp']) for c in bc)} with the direction flipped) and "
          f"{sum(c['scale_error'] and not c['basis_change'] for c in raw)} more with a scale error; kept {len(cs)}")
    print("POINT, carry of the latest year-over-year change (0 = seasonal random walk = 0.500), level | percent:")
    for lab, f in HALVES:
        one = [c for c in cs if c["k"] == 1 and f(c)]
        far = [c for c in cs if c["k"] >= 2 and f(c)]
        print(f"  {lab:10s} k=1  " + " ".join(f"{p:.1f}:{point_quality(one, lambda c, p=p: p):.4f}" for p in (0.0, 0.2, 0.3, 0.4, 0.6))
              + "  | " + " ".join(f"{p:.1f}:{point_quality(one, lambda c, p=p: p, 'pct'):.4f}" for p in (0.0, 0.2, 0.3, 0.4, 0.6)))
        print(f"  {lab:10s} k>=2 " + " ".join(f"{p:.2f}:{point_quality(far, lambda c, p=p: p):.4f}" for p in (0.0, 0.08, 0.16, 0.24))
              + "  | " + " ".join(f"{p:.2f}:{point_quality(far, lambda c, p=p: p, 'pct'):.4f}" for p in (0.0, 0.08, 0.16, 0.24)))
    print("DIRECTION, P(above the prior-year quarter | latest change):")
    nz = [c for c in cs if c["actual"] != c["base"] and c["cur"] != c["prior"]]
    for lab, f in HALVES:
        up = [c["actual"] > c["base"] for c in nz if f(c) and c["cur"] > c["prior"]]
        dn = [c["actual"] > c["base"] for c in nz if f(c) and c["cur"] < c["prior"]]
        print(f"  {lab:10s} after a rise {st.mean(up):.4f} (n={len(up)})   after a fall {st.mean(dn):.4f} (n={len(dn)})")
    print(f"INTERVAL, loss-minimising factor on the rule-based 90% half-width (carries PHI {E.PHI}, CARRY_FAR {E.CARRY_FAR}):")
    for lab, f in HALVES:
        print(f"  {lab:10s} k=1: {interval_factor([c for c in cs if c['k'] == 1 and f(c)]):.3f}   "
              f"k>=2: {interval_factor([c for c in cs if c['k'] >= 2 and f(c)]):.3f}")
    print("  out of sample, mean interval score of the other half's factor / of v18-v19's 1.8:")
    for kname, kf in (("k=1", lambda c: c["k"] == 1), ("k>=2", lambda c: c["k"] >= 2)):
        a = [c for c in cs if kf(c) and HALVES[0][1](c)]
        b = [c for c in cs if kf(c) and HALVES[1][1](c)]
        fa, fb = interval_factor(a), interval_factor(b)
        print(f"    {kname}: fitted on 2010-2015 ({fa:.2f}) scored on 2016-2021 {mean_interval_score(b, fa) / mean_interval_score(b, 1.8):.3f}; "
              f"fitted on 2016-2021 ({fb:.2f}) scored on 2010-2015 {mean_interval_score(a, fb) / mean_interval_score(a, 1.8):.3f}; "
              f"the full-sample factor on each half {mean_interval_score(b, interval_factor([c for c in cs if kf(c)])) / mean_interval_score(b, 1.8):.3f} / "
              f"{mean_interval_score(a, interval_factor([c for c in cs if kf(c)])) / mean_interval_score(a, 1.8):.3f}")
    return cs


if __name__ == "__main__":
    main()
