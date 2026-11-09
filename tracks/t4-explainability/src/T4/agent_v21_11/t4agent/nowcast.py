"""Pump-price nowcast of the first print of CPI gasoline, energy and all items (v20).

When the corpus holds a weekly retail gasoline price that reaches into the target month, the statistical baseline of
those three seasonally adjusted month-over-month changes is

    first print = BETA x (average weekly price in the target month / average in the month before - 1) x 100
                  + GAP[calendar month]

instead of the series' own AR(1). GAP absorbs the seasonal factors (pump prices rise every spring and fall every
autumn before adjustment) and, for energy and all items, what the pump price does not move. BETA, GAP and the 90%
half-widths were fitted on the EIA's weekly U.S. regular gasoline price and ALFRED real-time first prints published by
2021 (tools/calibrate_nowcast.py; ../calib_data/README.md). Out of sample, against the last first print (scorer 5.2.2
predictive quality, 0.5 = as good as it), with the whole target month seen:

    gasoline   0.754 (0.753 / 0.755 by half; 2014-2021)   AR(1) 0.525   two weeks seen 0.710
    energy     0.745 (0.757 / 0.728; 2005-2021)            AR(1) 0.525   two weeks seen 0.723
    all items  0.662 (0.703 / 0.608; 2005-2021)            AR(1) 0.520   two weeks seen 0.664

v17-v19 forecast these from their own history only, and the analysts were shown the raw price change without the
target month's seasonal gap (the corpus covers one year, so the gap of the month ahead is not in it).
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import statistics
from dataclasses import dataclass

from .artifact_policy import ArtifactPolicy
from .corpus import Corpus
from .stats import StatForecast
from .tables import parse_tables, to_number

# fitted constants: tools/calibrate_nowcast.py (first prints 2011-04..2021-11 for gasoline, 2002-01..2021-11 otherwise)
NOWCAST: dict[str, dict] = {
    "CUSR0000SETB01": {"name": "gasoline", "beta": 1.0, "mae": 1.645, "ar1_mae": 4.55, "months": "2011-2021", "n": 92,
                       "q90": {2: 4.03, 3: 3.22},
                       "gap": {1: -0.49, 2: -1.47, 3: -5.50, 4: -2.89, 5: -3.14, 6: 1.13, 7: 1.77, 8: 1.58, 9: 1.54,
                               10: 2.67, 11: 2.47, 12: 2.26}},
    "CPIENGSL": {"name": "energy", "beta": 0.537, "mae": 1.009, "ar1_mae": 2.66, "months": "2002-2021", "n": 203,
                 "q90": {2: 2.29, 3: 2.00},
                 "gap": {1: 0.03, 2: -0.33, 3: -2.32, 4: -2.00, 5: -1.49, 6: 0.37, 7: 0.98, 8: 0.86, 9: 0.86,
                         10: 1.32, 11: 0.83, 12: 1.44}},
    "CPIAUCSL": {"name": "all items", "beta": 0.049, "mae": 0.135, "ar1_mae": 0.245, "months": "2002-2021", "n": 203,
                 "q90": {2: 0.29, 3: 0.25},
                 "gap": {1: 0.16, 2: 0.13, 3: -0.09, 4: -0.01, 5: 0.04, 6: 0.22, 7: 0.26, 8: 0.21, 9: 0.21,
                         10: 0.27, 11: 0.20, 12: 0.25}},
}
_DAY = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")
_MONTH = re.compile(r"^(\d{4})-(\d{2})$")
_SA_MOM = re.compile(r"(?:mom|month[\s_-]*over[\s_-]*month|m/m)", re.I)
_SA = re.compile(r"(?:^|[^a-z])sa(?:$|[^a-z])|seasonally[\s_-]*adjusted", re.I)
_NSA = re.compile(r"(?:^|[^a-z])nsa(?:$|[^a-z])|not[\s_-]*seasonally", re.I)
_MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
           "November", "December")


@dataclass
class PumpNowcast(StatForecast):
    code: str = ""
    month: str = ""  # target month, YYYY-MM
    driver: float = 0.0  # change in the average weekly pump price, %, not seasonally adjusted
    weeks: int = 0  # weekly readings of the target month in the corpus
    avg_cur: float = 0.0
    avg_prev: float = 0.0

    def describe(self) -> str:
        c = NOWCAST[self.code]
        m = int(self.month[5:])
        return (f"statistical baseline (code): pump-price nowcast {self.value:+.2f} = {c['beta']:g} x the weekly U.S. "
                f"gasoline price change ({self.avg_cur:.3f} in {self.weeks} weeks of {self.month} vs {self.avg_prev:.3f} the "
                f"month before: {self.driver:+.2f}%, not seasonally adjusted) + {_MONTHS[m - 1]} seasonal gap "
                f"{c['gap'][m]:+.2f} (fitted on {c['name']} first prints {c['months']}; out-of-sample MAE {c['mae']:.2f} vs "
                f"{c['ar1_mae']:.2f} for the series' own AR(1); 90% error {self.q90:.2f})")


def coefficients_digest() -> str:
    """SHA-256 of the fitted coefficients (BETA, GAP, 90% half-widths): the learned model's revision, as disclosed in
    the submission's models[] row (t4agent.LOCAL_MODELS)."""
    table = {code: {"beta": c["beta"], "gap": {str(m): g for m, g in c["gap"].items()},
                    "q90": {str(k): v for k, v in c["q90"].items()}} for code, c in NOWCAST.items()}
    return hashlib.sha256(json.dumps(table, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _code(ent: dict) -> str | None:
    for k, v in ent.items():
        if isinstance(v, str) and re.search(r"series|fred|code", k, re.I) and v.strip().upper() in NOWCAST:
            return v.strip().upper()
    return None


def _target_month(ent: dict) -> str | None:
    for k, v in ent.items():
        if isinstance(v, str) and "month" in k.lower() and "latest" not in k.lower() and _MONTH.match(v.strip()):
            return v.strip()
    return None


def _prev(month: str) -> str:
    y, m = int(month[:4]), int(month[5:]) - 1
    return f"{y - (m == 0):04d}-{m or 12:02d}"


def pump_prices(corpus: Corpus, doc_ids: list[str]) -> dict[str, list[float]] | None:
    """Month -> weekly retail gasoline prices, from a weekly table in a document about gasoline prices."""
    for did in doc_ids:
        doc = corpus.docs.get(did)
        if doc is None or not re.search(r"gasoline", f"{doc.title} {doc.text[:300]}", re.I):
            continue
        for t in parse_tables(did, doc.text):
            rows = [(r[0].strip(), to_number(r[1])) for r in t.rows if len(r) >= 2 and _DAY.match(r[0].strip())]
            rows = [(d, x) for d, x in rows if x is not None and 0.5 < x < 20]  # dollars per gallon
            if len(rows) < 8:
                continue
            days = [dt.date.fromisoformat(d) for d, _ in rows]
            if not 5 <= statistics.median((b - a).days for a, b in zip(days, days[1:])) <= 9:
                continue  # not weekly
            out: dict[str, list[float]] = {}
            for d, x in sorted(rows):
                out.setdefault(d[:7], []).append(x)
            return out
    return None


def pump_nowcast(corpus: Corpus, doc_ids: list[str], ent: dict, target_name: str,
                 base: StatForecast | None, *, policy: ArtifactPolicy | None = None) -> PumpNowcast | None:
    """The nowcast for a CPI gasoline / energy / all-items first print, or None (the univariate baseline stands)."""
    if policy is None or not policy.permits("nowcast"):
        return None
    code = _code(ent)
    month = _target_month(ent)
    unit = f"{target_name} {ent.get('unit', '')} {ent.get('units', '')}"
    if code is None or month is None or not _SA_MOM.search(unit) or not _SA.search(unit) or _NSA.search(unit):
        return None
    prices = pump_prices(corpus, doc_ids)
    if not prices:
        return None
    cur, prev = prices.get(month, []), prices.get(_prev(month), [])
    if len(cur) < 2 or len(prev) < 3:
        return None
    c = NOWCAST[code]
    a_cur, a_prev = statistics.fmean(cur), statistics.fmean(prev)
    driver = (a_cur / a_prev - 1) * 100
    value = c["beta"] * driver + c["gap"][int(month[5:])]
    q90 = c["q90"][2 if len(cur) == 2 else 3]
    # n_backtest: the out-of-sample months behind MAE and q90 (a measured scale: reliable_scale holds)
    return PumpNowcast("pump-price nowcast", value, c["mae"], q90, c["n"], base.last if base is not None else float("nan"),
                       base.series if base is not None else None, code, month, driver, len(cur), a_cur, a_prev)
