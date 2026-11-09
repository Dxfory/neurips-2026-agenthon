"""Per-task eligibility for constants fitted or selected using historical outcomes.

The dates below are conservative eligibility bounds, not claimed publication dates.
They use the latest availability of fitting AND selection data. Development practice
permissions and a task's name never relax this policy for an earlier cutoff. Pure
corpus calculations and independently justified generic rules need no permission.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping


@dataclass(frozen=True, slots=True)
class Artifact:
    capability: str
    eligible_from: dt.date
    data_available: str
    conservative_bound: str
    evidence: tuple[str, ...]
    includes: tuple[str, ...]


ARTIFACTS: Mapping[str, Artifact] = MappingProxyType({
    "statistics": Artifact(
        capability="statistics",
        eligible_from=dt.date(2022, 2, 1),
        data_available=(
            "Daily Treasury, H.10 FX and WTI observations through 2021-12-31 were public by early January 2022; "
            "CFTC 2010-2021 annual files were complete in the first days of 2022. Monthly calibration also uses "
            "ALFRED vintages and first prints through 2021. Exact final publication days are not established."
        ),
        conservative_bound=(
            "The first day after January 2022 conservatively covers the documented early-January availability; "
            "2022-02-01 is a policy bound, not an inferred exact release date."
        ),
        evidence=(
            "ARTIFACT_PROVENANCE.md: random-walk interval table; backtest inflation; v17 rate-series and "
            "change-target defaults; v17/v19 positioning reversion",
            "../calib_data/README.md: daily market series; real-time CPI vintages; CFTC Commitments of Traders",
            "tools/calibrate_scales.py; tools/calibrate_methods.py",
        ),
        includes=(
            "stats._RW_K interpolation and all short-history/table-scale consumers",
            "stats._inflation retention after interval-score comparison",
            "stats AR(1) rate default and outcome-selected candidate restrictions",
            "stats.CHANGE_SWITCH and both-half switching condition",
            "stats.LEVEL_REVERT_DAILY, RATIO_SWITCH and ratio-reversion method selection",
        ),
    ),
    "nowcast": Artifact(
        capability="nowcast",
        eligible_from=dt.date(2022, 1, 1),
        data_available=(
            "CPI first prints used by the fit end at reference month 2021-11, with the last CPI release on "
            "2021-12-10. The disclosed EIA weekly-price data extend through 2021-12-27; provenance records "
            "all fitting values as public by 2021-12-31."
        ),
        conservative_bound=(
            "The day after the disclosed year-end availability bound is used instead of claiming an exact "
            "earlier date for the complete fitting and selection dependency set."
        ),
        evidence=(
            "ARTIFACT_PROVENANCE.md: t4agent/nowcast.py NOWCAST fitted model and selection",
            "../calib_data/README.md: real-time CPI vintages; weekly retail gasoline price",
            "tools/calibrate_nowcast.py",
        ),
        includes=(
            "NOWCAST beta, monthly gaps and two/three-week q90 scales",
            "NOWCAST mae, ar1_mae, sample counts and fitted-period descriptions",
            "PumpNowcast numerical output, reliability, request allocation and all describe/prompt paths",
        ),
    ),
    "revision": Artifact(
        capability="revision",
        eligible_from=dt.date(2022, 1, 1),
        data_available=(
            "Fourteen ALFRED monthly series have fitting and resolving releases through 2021-12-23."
        ),
        conservative_bound=(
            "Use the day after the complete disclosed calibration year as a conservative bound, even though "
            "the recorded last release is 2021-12-23."
        ),
        evidence=(
            "ARTIFACT_PROVENANCE.md: revision-rate calibration and expected-next-estimate selection",
            "../calib_data/README.md: real-time vintages (v15 revision calibration)",
            "tools/calibrate_revisions.py",
        ),
        includes=(
            "DIRECTION_WEIGHT, SCALE_ONE_RELEASE and SCALE_MORE_RELEASES",
            "Outcome-selected expected next estimate equal to the latest estimate",
            "RevisionPrior historical calibration statements, direction blending and interval scales",
        ),
    ),
    "eps": Artifact(
        capability="eps",
        eligible_from=dt.date(2022, 1, 1),
        data_available=(
            "SEC diluted-EPS calibration reads only facts with filed dates before 2022-01-01, including "
            "outcomes and comparative figures used to select comparable share bases."
        ),
        conservative_bound=(
            "2022-01-01 follows every allowed filing date; the target quarter alone is not an availability date. "
            "The original literature rationale for PHI does not erase its later outcome-based selection."
        ),
        evidence=(
            "ARTIFACT_PROVENANCE.md: EPS anchor v18/recalibrated v20",
            "../calib_data/README.md: SEC EDGAR diluted EPS",
            "tools/calibrate_eps.py: filed-date filtering and comparable-case selection",
        ),
        includes=(
            "PHI retention/selection, CARRY_FAR, EPS_SCALE_NEXT and EPS_SCALE_FAR",
            "EPS_P_UP_AFTER_UP and EPS_P_UP_AFTER_DOWN",
            "EpsAnchor values, intervals, direction blending and historical-frequency descriptions",
        ),
    ),
    "blend": Artifact(
        capability="blend",
        eligible_from=dt.date(2020, 1, 1),
        data_available=(
            "TreasuryDirect pseudo-auction fitting/selection comparisons use results published on auction "
            "days in 2016-2019, before 2020."
        ),
        conservative_bound=(
            "Use the day after the disclosed selection period. The equal-weight value originally had an "
            "a-priori rationale, but the later decision to retain it after comparing outcome scores remains "
            "part of its provenance."
        ),
        evidence=(
            "ARTIFACT_PROVENANCE.md: STAT_WEIGHT and v16 pseudo-auction comparison",
            "../calib_data/README.md: Treasury auction results",
            "tools/pseudo_auctions.py: preregistered decision rule",
        ),
        includes=("STAT_WEIGHT retention after the pseudo-auction outcome-score comparison",),
    ),
})


@dataclass(frozen=True, slots=True)
class ArtifactPolicy:
    """An immutable task-local cutoff; missing or non-exact dates permit no artifacts.

    Datetime is deliberately rejected even though it subclasses date. Date strings
    must be validated by the task loader, not interpreted by this policy.
    """

    cutoff: dt.date | None

    def __post_init__(self) -> None:
        if type(self.cutoff) is not dt.date:
            object.__setattr__(self, "cutoff", None)

    def permits(self, capability: str) -> bool:
        if self.cutoff is None or type(capability) is not str:
            return False
        artifact = ARTIFACTS.get(capability)
        return artifact is not None and self.cutoff >= artifact.eligible_from
