"""Revision priors from real-time vintage tables in the corpus.

A vintage table gives, for each reference period, the value as published on each pre-cutoff release date.
For an entity asking whether the estimate published on its latest vintage will be revised up or down by
a later release, the analogous past transitions are read off the same table: other reference periods'
estimates, from the same revision stage, the same number of releases later. Their direction frequency,
shrunk towards the series' frequency over all stages, is a reference-class prior computed from pre-cutoff
data only; the analysts see it and the label probabilities are combined with it.

Only reference periods whose whole estimate history is visible in the table are counted: a period whose
first visible value came long after its usual first-release lag may have earlier estimates the table does
not show, and its stage numbering would be wrong.

v13: a benchmark or annual update (a release that revises reference periods the regular monthly revisions had
left behind, e.g. a NIPA annual update moving every month's personal income by about +3%) is a scheduled one-off,
not a sample of the next regular revision. Transitions across such a release are left out of the base rates and
the scale whenever regular transitions remain; v12 sized the July personal-income interval at +/-5% from it.
"""
from __future__ import annotations

import datetime as dt
import math
import re
import statistics
from dataclasses import dataclass

from .artifact_policy import ArtifactPolicy
from .corpus import Corpus

SHRINK = 2.0  # prior strength of the all-stage frequency when estimating a stage's frequency
MIN_SCALE = 4  # revisions needed before their envelope is a measured scale (stats.MIN_BACKTEST)
# v15/v16: calibrated by replaying this module on real-time (ALFRED) vintages of fourteen U.S. monthly series published
# 2003-2021 (tools/calibrate_revisions.py; 12,758 next-estimate forecasts, no task outcome; ARTIFACT_PROVENANCE.md):
# - a past-revision direction rate called the next revision right 53% of the time, and taken at face value it scored
#   worse than a coin flip (Brier 0.291 vs 0.250): only 15% of its departure from 0.5 is kept (Brier 0.248);
# - adding the mean past revision to the latest estimate made the expected next estimate worse (relative MAE 46 bp vs
#   36 bp for the latest itself): the expected next estimate is the latest estimate;
# - v16: the envelope's factor minimises the mean interval score (the scorer's loss), not the miss rate. v15 widened it
#   to cover 90% (x1.22 / x1.51), which raised the mean interval score by 8% (each series weighted equally); the
#   loss-minimising factors are below 1 -- wide envelopes (large past revisions) over-cover and their width dominates
#   the loss -- and cut it by 4.6-4.8% against v14 out of sample (time split 2006-2013 / 2014-2021).
DIRECTION_WEIGHT = 0.15
SCALE_ONE_RELEASE = 0.76
SCALE_MORE_RELEASES = 0.76

_ASOF = re.compile(r"as_of_(\d{4}-\d{2}-\d{2})")
_PERIOD = re.compile(r"^(\d{4})-(\d{2})$")


@dataclass
class RevisionPrior:
    method: str
    series: str
    stage: int | None  # estimates of the reference period published up to the latest vintage
    steps: int  # releases between the latest vintage and the resolving release
    up: int  # analogous past transitions revised up / with a non-zero revision
    n: int
    up_all: int
    n_all: int
    p_up: float
    mean_rel: float  # mean relative revision over the analogous transitions (fraction)
    latest: float
    value: float  # expected next estimate
    q90: float
    n_scale: int = 0  # revisions behind q90 (the same-stage ones, else every stage's)
    updates: str = ""  # v13: the benchmark or annual updates left out of the base rates, described

    def describe(self) -> str:
        stage = f"from estimate #{self.stage} to #{self.stage + self.steps}" if self.stage else "over all stages"
        left_out = f" {self.updates}" if self.updates else ""
        return (f"revision prior (computed by code from the {self.series} vintage table): past revisions {stage}: "
                f"{self.up} up of {self.n} non-zero; all stages: {self.up_all} up of {self.n_all}; mean revision "
                f"{self.mean_rel * 100:+.2f}%; P(next estimate above the latest) = {self.p_up:.2f} -- calibrated: in "
                f"2006-2021 real-time data such counts called the next revision's direction right only about 53% of the "
                f"time.{left_out} A weak base rate, not a verdict: weigh it against scheduled annual or benchmark "
                f"updates and other evidence.")


def _date(s: str) -> dt.date | None:
    try:
        return dt.date.fromisoformat(s[:10])
    except (TypeError, ValueError):
        return None


def _period_end(ref: str) -> dt.date | None:
    m = _PERIOD.match(ref)
    if not m:
        return None
    y, mo = int(m.group(1)), int(m.group(2))
    nxt = dt.date(y + (mo == 12), mo % 12 + 1, 1)
    return nxt - dt.timedelta(days=1)


def vintage_table(text: str) -> tuple[list[dt.date], dict[str, list[float | None]]] | None:
    """(vintage dates, {reference period: value per vintage or None}) from a pipe table with as_of_ columns."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        cells = [c.strip() for c in line.split("|")]
        dates = [_date(m.group(1)) for c in cells[1:] for m in [_ASOF.search(c)] if m]
        if len(cells) < 4 or len(dates) != len(cells) - 1 or None in dates:
            continue
        rows: dict[str, list[float | None]] = {}
        for row in lines[i + 1:]:
            parts = [c.strip() for c in row.split("|")]
            if len(parts) != len(cells) or not _PERIOD.match(parts[0]):
                if rows:
                    break
                continue
            vals = []
            for c in parts[1:]:
                try:
                    vals.append(float(c.replace(",", "")))
                except ValueError:
                    vals.append(None)
            rows[parts[0]] = vals
        if rows:
            return dates, rows
    return None


def _field(ent: dict, *need: str, exclude: str = "") -> object:
    for k, v in ent.items():
        kl = k.lower()
        if all(n in kl for n in need) and not (exclude and exclude in kl):
            return v
    return None


def revision_prior(corpus: Corpus, docs: set[str], ent: dict,
                   *, policy: ArtifactPolicy | None = None) -> RevisionPrior | None:
    if policy is None or not policy.permits("revision"):
        return None
    ref = _field(ent, "ref")
    vintage = _date(str(_field(ent, "vintage") or ""))
    latest = _field(ent, "latest", exclude="vintage")
    resolving = _date(str(_field(ent, "resolv") or ""))
    if not isinstance(ref, str) or vintage is None or not isinstance(latest, (int, float)) or isinstance(latest, bool):
        return None
    for did in sorted(docs):
        doc = corpus.docs.get(did)
        table = vintage_table(doc.text) if doc else None
        if not table or ref not in table[1]:
            continue
        dates, rows = table
        if vintage not in dates:
            continue
        return _prior(did, dates, rows, ref, vintage, float(latest), resolving)
    return None


def level_updates(dates: list[dt.date], rows: dict[str, list[float | None]], settled: int = 2) -> dict[int, tuple[int, float]]:
    """Vintage indices of benchmark or annual updates -> (periods it revised that had settled, their median relative
    revision). A release is one when it revises at least two reference periods whose estimate had stood unchanged
    over the `settled` releases before it: the regular monthly revisions had left those periods behind."""
    out: dict[int, tuple[int, float]] = {}
    for k in range(settled + 1, len(dates)):
        moves = []
        for vals in rows.values():
            before, now = vals[k - 1 - settled:k], vals[k]
            if now is None or any(v is None for v in before) or len(set(before)) != 1 or before[-1] == 0:
                continue
            if now != before[-1]:
                moves.append((now - before[-1]) / abs(before[-1]))
        if len(moves) >= 2:
            out[k] = (len(moves), statistics.median(moves))
    return out


def _prior(did: str, dates: list[dt.date], rows: dict[str, list[float | None]], ref: str, vintage: dt.date,
           latest: float, resolving: dt.date | None) -> RevisionPrior | None:
    # release cadence and usual first-estimate lag, from the table itself
    gaps = [(b - a).days for a, b in zip(dates, dates[1:]) if (b - a).days > 0]
    cadence = statistics.median(gaps) if gaps else 30.0
    steps = max(1, round((resolving - vintage).days / cadence)) if resolving and resolving > vintage else 1
    first_lags = []
    for r, vals in rows.items():
        end = _period_end(r)
        idx = next((i for i, v in enumerate(vals) if v is not None), None)
        if end is not None and idx is not None and idx > 0:
            first_lags.append((dates[idx] - end).days)
    usual_lag = statistics.median(first_lags) if first_lags else None

    def history(r: str) -> list[tuple[int, float]] | None:
        """All estimates of period r as (vintage index, value), oldest first, or None when the first ones may be
        missing."""
        vals = rows[r]
        idx = next((i for i, v in enumerate(vals) if v is not None), None)
        end = _period_end(r)
        if idx is None or end is None:
            return None
        if idx == 0 and (usual_lag is None or (dates[0] - end).days > usual_lag + 7):
            return None
        return [(i, v) for i, v in enumerate(vals) if i >= idx and v is not None]

    target = rows[ref][: dates.index(vintage) + 1]
    own = history(ref)
    stage = len([v for v in target if v is not None]) if own is not None else None
    updates = level_updates(dates, rows)

    def transitions(skip: set[int]) -> tuple[list[float], list[float]]:
        moves, all_moves = [], []
        for r in rows:  # every row is pre-cutoff data; the target row cannot reach the stage it asks about
            h = history(r)
            if not h:
                continue
            for j in range(len(h) - steps):
                (i0, a), (i1, b) = h[j], h[j + steps]
                if a == 0 or any(i0 < k <= i1 for k in skip):
                    continue
                all_moves.append((b - a) / abs(a))
                if stage is not None and j == stage - 1:
                    moves.append((b - a) / abs(a))
        return moves, all_moves

    moves, all_moves = transitions(set(updates))
    note = ""
    if updates and any(x != 0 for x in all_moves):
        note = "Left out of these rates as one-off benchmark/annual updates: " + "; ".join(
            f"the {dates[k].isoformat()} release, which revised {n} settled earlier months by a median "
            f"{m * 100:+.2f}%" for k, (n, m) in sorted(updates.items())) + "."
    else:  # no regular transition left: every one counts, as before v13
        moves, all_moves = transitions(set())
    up_all = sum(1 for x in all_moves if x > 0)
    n_all = sum(1 for x in all_moves if x != 0)
    if n_all == 0:
        return None
    up = sum(1 for x in moves if x > 0)
    n = sum(1 for x in moves if x != 0)
    p_all = (up_all + 2.0) / (n_all + 4.0)  # Beta(2, 2): a handful of revisions is weak evidence
    p_up = 0.5 + DIRECTION_WEIGHT * ((up + SHRINK * p_all) / (n + SHRINK) - 0.5)  # v15: calibrated (see above)
    sample = moves if len(moves) >= 2 else all_moves
    mean_rel = statistics.fmean(sample)

    def envelope(xs: list[float]) -> float:
        return max(abs(latest) * max((abs(x) for x in xs), default=0.0) * 1.5, abs(latest) * 1e-4)

    # The scale of the next revision: the same-stage revisions when there are enough of them. Two or three are not
    # (one of HOUST's two was zero, giving +/-0.1%), so they are met on the log scale by every stage's revisions of
    # the series, a larger sample that runs wide because early revisions are larger.
    if len(moves) >= MIN_SCALE or len(moves) < 2:
        q90, n_scale = envelope(sample), len(sample)
    else:
        q90, n_scale = math.sqrt(envelope(moves) * envelope(all_moves)), len(all_moves)
    series = did.split("_")[1] if did.count("_") >= 2 else did
    # v15: the expected next estimate is the latest one (adding the mean past revision made it worse in the replay;
    # v14 had fixed its sign for negative series), and the envelope is widened by its measured 90% factor
    q90 *= SCALE_ONE_RELEASE if steps == 1 else SCALE_MORE_RELEASES
    return RevisionPrior("vintage revision base rate", series, stage, steps, up, n, up_all, n_all, p_up, mean_rel,
                         latest, latest, q90, n_scale, note)
