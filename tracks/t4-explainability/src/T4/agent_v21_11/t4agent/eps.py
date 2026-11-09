"""Statistical EPS anchor: a seasonal random walk with a partial carry of the latest year-over-year change.

    E[EPS_t] = EPS_{t-4} + phi^k * (EPS_{t-k} - EPS_{t-k-4})

EPS_{t-4} is the same quarter a year earlier (a column of the entity table); EPS_{t-k} and EPS_{t-k-4} are
the latest reported quarter and the same quarter a year before it, read by code from the filing's own table
(facts.EpsRecord), k quarters before the target. Quarterly EPS changes against the same quarter a year
earlier persist but decay, so a fixed share of the latest change is carried forward (Foster 1977; Brown and
Rozeff 1979). PHI is taken from that literature, not fitted to any outcome.

The anchor is computed from pre-cutoff numbers only. It is shown to the analysts as the starting point they
must argue away from, and combined with their forecast like the other statistical baselines.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .artifact_policy import ArtifactPolicy
from .corpus import Corpus
from .facts import EpsRecord, eps_records

PHI = 0.4
REL_SD = 0.25  # typical error of a statistical quarterly EPS forecast, relative to the EPS level
# v18: checked on SEC EDGAR XBRL diluted EPS of 134 large U.S. companies, filings 2010-2021 (tools/calibrate_eps.py,
# ../calib_data/sec), in the Development units' setting; scorer-5.2.2 predictive quality against the prior-year quarter
# (the seasonal random walk).
# v20 (review of v18): v18's sample paired bases and outcomes on different share bases (splits, restatements, XBRL scale
# errors: 156 of 5,318 cases, 82 with the direction flipped), and its interval factor leaned on an assumed model-blend
# error. On the 5,162 comparable cases --
# - one quarter ahead, PHI = 0.4 stays best in EPS (0.520; in percent of the base 0.512 against 0.513 for 0.3),
#   above 0.5 in both half-periods and both target units;
# - two or more quarters ahead a carry of PHI ** 2 = 0.16 beats none (0.508; percent 0.502): CARRY_FAR. v18 set 0 on the
#   mixed-basis sample, where every carry lost;
# - the anchor's rule-based 90% half-width needs x1.97 one quarter ahead and x2.22 further out to minimise the interval
#   score (1.75-2.13 and 1.82-2.53 by half-period). The model's forecasts were measured to sit on the anchor (median
#   departure 0-8% of the anchor's error in five live runs of the Development EPS units) and the submitted interval
#   is the anchor's, so the blend's error is the anchor's: the factors apply as measured. v18's 1.8 assumed a blend
#   error 0.71-0.87 of the anchor's;
# - direction: the next quarter rose against its prior-year quarter in 75% of cases after a latest quarter that rose
#   (0.761 / 0.733 by half-period), in 50% after one that fell (0.504 / 0.493): EPS_P_UP_* below.
CARRY_FAR = 0.16
EPS_SCALE_NEXT = 1.97  # k = 1
EPS_SCALE_FAR = 2.22  # k >= 2
EPS_P_UP_AFTER_UP = 0.75
EPS_P_UP_AFTER_DOWN = 0.50
_DATE = re.compile(r"(\d{4})-(\d{2})-\d{2}")
_BASE_KEY = re.compile(r"prior.*(eps|earn)|(eps|earn).*prior", re.I)
_PERIOD_KEY = re.compile(r"quarter|period", re.I)


@dataclass
class EpsAnchor:
    method: str
    value: float  # in target units (EPS level, or % change / difference against the base)
    q90: float  # 90% half-width of the anchor's error, target units
    level: float  # anchor EPS level
    level_sd: float
    base: float
    latest: EpsRecord
    k: int
    units: str  # "level", "pct_change" or "difference"

    @property
    def p_up(self) -> float:
        """v18: calibrated probability that the target quarter's EPS is above the prior-year quarter's, from the sign
        of the latest reported quarter's change against its own prior-year quarter (see EPS_P_UP_*)."""
        d = self.latest.cur - self.latest.prior
        return EPS_P_UP_AFTER_UP if d > 0 else EPS_P_UP_AFTER_DOWN if d < 0 else 0.5 * (EPS_P_UP_AFTER_UP + EPS_P_UP_AFTER_DOWN)

    def describe(self) -> str:
        r = self.latest
        carry = f"{PHI:g}" if self.k == 1 else f"{CARRY_FAR:g}"
        target = {"level": f"{self.level:.2f} per share",
                  "pct_change": f"{self.level:.2f} per share, i.e. {self.value:+.1f}% against {self.base:.2f}",
                  "difference": f"{self.level:.2f} per share, i.e. {self.value:+.2f} against {self.base:.2f}"}[self.units]
        return (f"statistical anchor (seasonal random walk on quarterly EPS, computed by code): same quarter a year "
                f"earlier {self.base:.2f} + {carry} x the latest reported quarter's change against its own prior-year "
                f"quarter ({r.cur:.2f} - {r.prior:.2f} = {r.cur - r.prior:+.2f}, quarter ending {r.year}-{r.month:02d}) = "
                f"{target}; typical error about +/-{self.level_sd:.2f} per share. Historically (large U.S. companies, "
                f"2010-2021) EPS rose against the prior-year quarter {EPS_P_UP_AFTER_UP:.0%} of the time after a latest "
                f"quarter that rose and {EPS_P_UP_AFTER_DOWN:.0%} after one that fell: P(up) = {self.p_up:.2f} here. One-off "
                f"items in the quarters used are not removed: adjust the anchor for any you find, and depart from it only "
                f"with specific evidence.")


def _months(y: int, m: int) -> int:
    return y * 12 + m


def target_period(ent: dict) -> tuple[int, int] | None:
    """(year, month) of the target quarter's end, from an entity column such as 'three months ended 2024-09-30'."""
    for key, val in ent.items():
        if isinstance(val, str) and _PERIOD_KEY.search(key) and "prior" not in key.lower():
            m = _DATE.search(val)
            if m:
                return int(m.group(1)), int(m.group(2))
    return None


def base_value(ent: dict, base_field: str | None = None) -> float | None:
    """The same quarter a year earlier: the planner's base column when given, else a 'prior ... eps' column."""
    keys = [base_field] if base_field else []
    keys += [k for k in ent if _BASE_KEY.search(k)]
    for k in keys:
        v = ent.get(k) if k else None
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return float(v)
    return None


def eps_anchor(corpus: Corpus, docs: set[str], ent: dict, units: str,
               base_field: str | None = None, *, policy: ArtifactPolicy | None = None) -> EpsAnchor | None:
    """The anchor for one entity, or None when the target period, the base or a recent quarter is missing."""
    if policy is None or not policy.permits("eps"):
        return None
    tp = target_period(ent)
    base = base_value(ent, base_field)
    if tp is None or base is None or units not in ("level", "pct_change", "difference"):
        return None
    if units == "pct_change" and base <= 0:
        return None
    quarters = [r for r in eps_records(corpus, docs) if r.kind == "10-Q"]
    best = None
    for r in quarters:
        gap = _months(*tp) - _months(r.year, r.month)
        k = round(gap / 3)
        if 1 <= k <= 3 and abs(gap - 3 * k) <= 1 and (best is None or k < best[1]):
            best = (r, k)
    if best is None:
        return None
    r, k = best
    carry = PHI if k == 1 else CARRY_FAR  # v20: PHI ** 2 further out (v18: none)
    level = base + carry * (r.cur - r.prior)
    sd = (EPS_SCALE_NEXT if k == 1 else EPS_SCALE_FAR) * max(REL_SD * max(abs(level), abs(base)), 0.5 * abs(r.cur - r.prior) * PHI ** (k - 1), 0.02)
    if units == "pct_change":
        value, q90 = (level / base - 1.0) * 100.0, 1.645 * sd / base * 100.0
    elif units == "difference":
        value, q90 = level - base, 1.645 * sd
    else:
        value, q90 = level, 1.645 * sd
    return EpsAnchor("seasonal random walk (EPS)", value, q90, level, sd, base, r, k, units)
