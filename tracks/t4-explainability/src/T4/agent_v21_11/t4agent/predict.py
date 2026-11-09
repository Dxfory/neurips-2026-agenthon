"""Forecasting: planner -> grouped analyst calls with self-consistency -> per-entity aggregate.

Every entity always ends up with a forecast: LLM samples when they arrive in time, otherwise a
deterministic fallback. Nothing here raises to the caller.
"""
from __future__ import annotations

import json
import logging
import math
import re
import statistics
import time
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field

from . import config
from .artifact_policy import ArtifactPolicy
from .corpus import Corpus
from .eps import EpsAnchor, eps_anchor
from .facts import key_fact_chunks, like_for_like, risk_profile
from .features import entity_features
from .grounding import sanitize_plan
from .labels import (assertion_rules, directional_pair, directional_rules, implied_probs, rule_label, rules_parse,
                     table_rules)
from .llm import BudgetExhausted, HouseClient, LLMUnavailable, extract_json, margin, parse_reply
from .nowcast import pump_nowcast
from .prompts import analyst_messages, planner_messages, recheck_messages, repair_messages, review_messages
from .retrieval import BM25, entity_query, keyline_chunks
from .revisions import RevisionPrior, revision_prior
from .stats import MIN_BACKTEST, StatForecast, entity_forecast
from .tables import summarize
from .task import TaskSpec

log = logging.getLogger("t4agent.predict")


@dataclass
class Forecast:
    eid: str
    label: str | None = None
    probs: dict[str, float] | None = None
    model_probs: dict[str, float] | None = None  # v11: the model's own label probabilities, before any blending
    median: float | None = None
    p05: float | None = None
    p95: float | None = None
    support: list[int] = field(default_factory=list)
    why: dict[int, str] = field(default_factory=dict)
    rationale: str = ""
    source: str = "fallback"
    n_samples: int = 0
    stat_q90: float | None = None  # backtest 90% absolute error of the statistical baseline
    stat_reliable: bool = False  # stat_q90 was measured on the entity's own history (see reliable_scale)
    rechecked: bool = False


class Truncated(ValueError):
    """A group's reply hit the output cap and could not be repaired: retry it as two smaller groups."""


def _f(x) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def unit_deviation(spec: TaskSpec, plan: dict | None) -> float | None:
    """The planner's typical deviation over the horizon: one number for the whole unit, so it means nothing
    when the rows are quantities in different units (payrolls in thousands of persons next to an index)."""
    dev = _f((plan or {}).get("typical_abs_deviation"))
    if dev is None or dev <= 0:
        return None
    for key in ("unit", "units"):
        if len({str(e.get(key)).strip().lower() for e in spec.entities if e.get(key) is not None}) > 1:
            return None
    return dev


_EVENT_RETURN = re.compile(r"abnormal return|announcement(?:-day)? return|event[- ]day return|(?:one|two|three|1|2|3)"
                           r"[- ]day (?:stock |abnormal |price )?return|(?:stock|share)[- ]price reaction"
                           r"|earnings reaction|price reaction", re.I)


def event_return_sd(spec: TaskSpec) -> float | None:
    """A priori scale (one standard deviation, percentage points) when the unit's quantity is a stock's return over
    a short window around a scheduled announcement, stated in percent; None otherwise. Without it the scale came
    from the plan's single typical deviation (1.5 in one run, 3.0 in the next), and a narrow one made a +/-1%
    'flat' band look like the likeliest label of an earnings-day return."""
    raw = getattr(spec, "raw", None) or {}
    target = raw.get("target") if isinstance(raw.get("target"), dict) else {}
    assertions = target.get("label_assertions") if isinstance(target.get("label_assertions"), dict) else {}
    text = " ".join([str(getattr(spec, "prompt", "") or ""), str(getattr(spec, "target_name", "") or "").replace("_", " ")]
                    + [str(v) for v in assertions.values()])
    if not _EVENT_RETURN.search(text) or not re.search(r"%|percent", text, re.I):
        return None
    return config.EVENT_RETURN_SD


def reliable_scale(sf) -> bool:
    """Whether a baseline's error scale was measured on the entity's own history: an empirical error quantile
    over the backtest origins stats.py accepts (inflated there for a small sample), or the envelope of as many
    of the series' past revisions in its vintage table. A random-walk extrapolation from a short history, an
    envelope of one or two revisions and the EPS anchor's rule-based scale are estimates, not measurements."""
    if isinstance(sf, RevisionPrior):  # an envelope of two past revisions is not yet a measurement
        return sf.q90 > 0 and sf.n_scale >= MIN_BACKTEST
    return isinstance(sf, StatForecast) and sf.q90 is not None and sf.n_backtest >= MIN_BACKTEST


# ---------------------------------------------------------------------------------------------- #
# Evidence                                                                                        #
# ---------------------------------------------------------------------------------------------- #
class Evidence:
    def __init__(self, spec: TaskSpec, corpus: Corpus, specific: dict[str, list[str]], shared: list[str]):
        self.spec, self.corpus = spec, corpus
        self.specific, self.shared = specific, shared
        self.bm = BM25(corpus.chunks)
        total = sum(len(d.text) for d in corpus.docs.values())
        self.small = total <= config.SMALL_CORPUS_CHARS
        self.packs: dict[str, list[int]] = {}
        self.facts: dict[str, dict[int, str]] = {}  # eid -> {chunk id: fact category}
        self.lfl: dict[str, list[str]] = {}  # eid -> like-for-like lines computed from its filings
        self.features: dict[str, list[str]] = {}  # eid -> series, driver, cross-section, going-concern lines

    def entity_docs(self, eid: str) -> list[str]:
        """Documents an entity may cite: its own, else the shared ones, else everything."""
        return self.specific.get(eid) or self.shared or sorted(self.corpus.docs)

    def allowed_support_docs(self, eid: str) -> set[str]:
        return set(self.specific.get(eid) or []) | set(self.shared) | (set(self.corpus.docs) if self.small else set())

    def build_packs(self, plan_terms: list[str]) -> None:
        for ent in self.spec.entities:
            eid = ent["entity_id"]
            if self.small:
                self.packs[eid] = [c.cid for c in self.corpus.chunks]
                continue
            docs = set(self.specific.get(eid) or []) | {d for d in self.shared if len(self.corpus.docs[d].text) < 60000}
            if not docs:
                docs = set(self.entity_docs(eid))
            q = entity_query(self.spec, ent, plan_terms)
            own = set(self.specific.get(eid) or []) or docs
            if eid not in self.lfl:
                self.lfl[eid] = like_for_like(self.corpus, set(self.specific.get(eid) or []))
            facts = key_fact_chunks(self.corpus, self.bm, own, q)
            self.facts[eid] = {c.cid: cat for cat, c in facts}
            picked = [c.cid for _, c in facts]
            for _, c in self.bm.search(q, config.EVIDENCE_CHUNKS_PER_ENTITY, docs):
                if c.cid not in picked:
                    picked.append(c.cid)
            for c in keyline_chunks(self.corpus, docs, self.bm, q, config.KEYLINE_CHUNKS_PER_ENTITY):
                if c.cid not in picked:
                    picked.append(c.cid)
            if plan_terms:
                for _, c in self.bm.search(" ".join(plan_terms) + " " + str(ent.get("name", eid)), 8, docs):
                    if c.cid not in picked:
                        picked.append(c.cid)
            self.packs[eid] = picked

    def render(self, eids: list[str], char_cap: int = config.PROMPT_EVIDENCE_CHARS) -> str:
        """Snippet block for a group: shared corpus once, or per-entity packs in document order."""
        corpus = self.corpus
        lines: list[str] = []
        if self.small:
            for eid in eids:  # the computed facts first, per entity; then the whole (small) corpus
                facts = (self.lfl.get(eid) or []) + (self.features.get(eid) or [])
                if facts:
                    lines.append(f"=== computed facts for {eid}:")
                    lines.extend(facts)
            for did in sorted(corpus.docs):
                d = corpus.docs[did]
                lines.append(f"--- {did} ({d.doc_date}) {d.title[:120]}")
                for c in corpus.doc_chunks(did):
                    lines.append(f"[E{c.cid}] {c.text}")
            return "\n".join(lines)[:char_cap * 2]
        budget = char_cap
        per = max(2000, budget // max(1, len(eids)))
        seen: set[int] = set()
        for eid in eids:
            lines.append(f"=== snippets for {eid}:")
            own = set(self.specific.get(eid) or [])
            profile = risk_profile(corpus, own) if own else ""
            if profile:
                lines.append(f"RISK INDICATOR COUNTS in {eid}'s own filings: {profile}")
            lines.extend(self.lfl.get(eid) or [])
            lines.extend(self.features.get(eid) or [])
            used = 0
            facts = self.facts.get(eid, {})
            rest = sorted((i for i in self.packs.get(eid, []) if i not in facts),
                          key=lambda i: (corpus.chunks[i].doc_id, corpus.chunks[i].start))
            for cid in list(facts) + rest:  # key facts first, so the budget never cuts them
                c = corpus.chunks[cid]
                if cid in seen:
                    continue
                tag = f"KEY FACT ({facts[cid]}) " if cid in facts else ""
                s = f"[E{c.cid}] {tag}({c.doc_id}, {corpus.docs[c.doc_id].doc_date}) {c.text}"
                if used + len(s) > per:
                    break
                lines.append(s)
                used += len(s)
                seen.add(cid)
        return "\n".join(lines)


    def digest(self, char_cap: int = config.PLAN_DIGEST_CHARS) -> str:
        """What the planner reads besides the inventory, so that it plans from the evidence rather than from
        memory: the documents themselves for a small corpus; otherwise each entity's computed facts and its
        first key passages."""
        corpus = self.corpus
        if self.small:
            lines: list[str] = []
            for did in sorted(corpus.docs):
                d = corpus.docs[did]
                lines += [f"--- {did} ({d.doc_date}) {d.title[:120]}", d.text]
            return "\n".join(lines)[:char_cap]
        per = max(600, char_cap // max(1, len(self.spec.entity_ids)))
        blocks = []
        for eid in self.spec.entity_ids:
            block = [f"=== {eid}:"] + (self.lfl.get(eid) or []) + (self.features.get(eid) or [])
            for cid in list(self.facts.get(eid, {}))[:4]:
                c = corpus.chunks[cid]
                block.append(f"({c.doc_id}, {corpus.docs[c.doc_id].doc_date}) {c.text[:400]}")
            blocks.append("\n".join(block)[:per])
        return "\n".join(blocks)[:char_cap]


def inventory(corpus: Corpus, max_docs: int = 60) -> str:
    out = []
    for did in sorted(corpus.docs)[:max_docs]:
        d = corpus.docs[did]
        head = (d.title or d.text[:160]).replace("\n", " ")[:160]
        out.append(f"{did} | {d.doc_date} | {len(d.text)} chars | {head}")
    return "\n".join(out)


# ---------------------------------------------------------------------------------------------- #
# Fallback                                                                                        #
# ---------------------------------------------------------------------------------------------- #
_INFORMED_PRIORS = ("no_event", "no", "none", "no_default", "beat", "up", "positive_reaction", "positive", "increase")
_PRIOR_ORDER = ("no_event", "no", "none", "no_default", "beat", "up", "positive_reaction", "positive", "increase",
                # v21.1: the status quo of a label set none of the above names (hold / stable / unchanged ...); v21 took
                # the first label, 'cut' for a policy decision and 'appreciate' for a currency
                "hold", "unchanged", "no_change", "stable", "flat", "maintain", "neutral", "inline", "same",
                "steady")  # v21.2


def prior_label(labels: list[str]) -> str | None:
    if not labels:
        return None
    low = {lab.lower(): lab for lab in labels}
    for key in _PRIOR_ORDER:
        if key in low:
            return low[key]
    return labels[0]


_CHANGEY = re.compile(r"change|growth|revision|delta|bps|return|shift|surprise"
                      r"|(?:^|[\W_])(?:mom|qoq|yoy|m/m|q/q|y/y)(?:$|[\W_])", re.I)  # v21.1: and mom/qoq/yoy
# v21.1: a row's latest published value of a change-type target ('latest_published_mom_pct', 'latest_change_2019_04')
_LATEST_CHANGE_FIELD = re.compile(r"(?=.*(?:latest|last|prior|previous))(?=.*(?:change|chg|mom|qoq|yoy|growth|delta))",
                                  re.I)
_LEVEL_FIELD = re.compile(r"latest|last|current|prior|previous|consensus|estimate|expected|eps\b|_eps|value", re.I)
_NOT_QUANTITY = re.compile(r"threshold|cik|maturity|count|size_announced|mktcap|market_cap|open_interest"
                           r"|offering|amount|close_|_close|price", re.I)
_DATE_PERIOD_TOKENS = frozenset({
    "year", "years", "yr", "fy", "fiscal", "fiscalyear", "qtr", "quarter", "month", "period", "vintage",
    "date", "dates", "datetime", "timestamp", "week", "weeks",
})
_DATE_PERIOD_MODIFIERS = frozenset({
    "latest", "last", "current", "prior", "previous", "fiscal", "target", "start", "end", "ending",
})
_DATE_VALUE = re.compile(
    r"(?:^|[^0-9])(?:19|20)\d{2}[-/](?:0?[1-9]|1[0-2])(?:[-/](?:0?[1-9]|[12]\d|3[01]))?(?:$|[^0-9])"
    r"|(?:^|[^a-z0-9])(?:fy\s*)?(?:19|20)\d{2}(?:\s*[-/]?\s*q[1-4])?(?:$|[^a-z0-9])"
    r"|(?:^|[^a-z0-9])q[1-4]\s*[-/]?\s*(?:19|20)\d{2}(?:$|[^a-z0-9])",
    re.I,
)


def _field_tokens(name) -> tuple[str, ...]:
    """Field-name tokens after underscore/punctuation and camelCase boundaries."""
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", str(name))
    text = re.sub(r"(?<=[A-Za-z])(?=\d)|(?<=\d)(?=[A-Za-z])", "_", text)
    return tuple(t for t in re.split(r"[^a-z0-9]+", text.lower()) if t)


def _date_period_field(name, value) -> bool:
    """A calendar/period field whose value has the corresponding date-like shape.

    Both halves are required: every name token must have a calendar role, and
    the value must have the matching calendar shape. Thus
    ``prior_quarter_gdp`` and ``latest_rate_at_date`` are quantities without
    maintaining an open-ended allow-list. This helper is consulted only when
    the planner explicitly marks an interval as probability.
    """
    tokens = set(_field_tokens(name))
    calendar = tokens & _DATE_PERIOD_TOKENS
    if not calendar or tokens - _DATE_PERIOD_TOKENS - _DATE_PERIOD_MODIFIERS or isinstance(value, bool):
        return False
    if isinstance(value, str) and _DATE_VALUE.search(value.strip()):
        return True
    number = _f(value)
    if number is None or not number.is_integer():
        return False
    n = int(number)
    if 1900 <= n <= 2100:
        return True
    if 190001 <= n <= 210012 and 1 <= n % 100 <= 12:
        return True
    if 19000101 <= n <= 21001231:
        month, day = (n // 100) % 100, n % 100
        return 1 <= month <= 12 and 1 <= day <= 31
    if calendar & {"year", "years", "yr", "fy", "fiscalyear"} and 0 <= n <= 99:
        return True
    if calendar & {"qtr", "quarter"} and 19001 <= n <= 21004:
        year, quarter = divmod(n, 10)
        if 1900 <= year <= 2100 and 1 <= quarter <= 4:
            return True
    if calendar & {"week", "weeks"} and 1 <= n <= 53:
        return True
    if calendar & {"date", "dates", "datetime"} and 1 <= n <= 73415:
        return True  # Excel's 1900 date system through 2100-12-31
    if calendar & {"date", "dates", "datetime", "timestamp"} and 0 <= n <= 4_133_980_799:
        return n >= 315_532_800  # Unix seconds from 1980-01-01 through 2100-12-31
    if calendar & {"month", "period"} and 1 <= n <= 12:
        return True
    return bool(calendar & {"qtr", "quarter"} and 1 <= n <= 4)


def _numeric_fields(ent: dict) -> list[tuple[str, float]]:
    out = []
    for k, v in ent.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            continue
        fv = _f(v)
        if fv is not None:
            out.append((k, fv))
    return out


def _rules_compare_base(rules, base: float, pair: tuple[str, str], ent: dict) -> bool:
    """Whether label rules say 'up: Q > X, down: otherwise / Q < X' with X evaluating to `base` on this row."""
    if not isinstance(rules, dict):
        return False
    from .labels import _COND, _eval

    m = _COND.match(str(rules.get(pair[0]) or ""))
    if not m or m.group(1) not in (">", ">="):
        return False
    x = _eval(m.group(2), ent)
    return x is not None and abs(x - base) <= 1e-9 * max(1.0, abs(base))


def baseline_value(spec: TaskSpec, ent: dict, plan: dict | None, *, skip_date_period: bool = False) -> float | None:
    """Naive forecast of the TARGET (regression / ranking).

    A change-type target (growth, change, revision, bps, return) starts from zero unless the plan
    names a column that is itself a change of the same kind; a level column (e.g. last year's EPS
    for an EPS-growth target) is never mistaken for it.
    """
    changey = bool(_CHANGEY.search(spec.target_name))
    field_name = (plan or {}).get("baseline_field")
    if isinstance(field_name, str) and field_name in ent:
        v = _f(ent.get(field_name))
        if v is not None and not (skip_date_period and _date_period_field(field_name, ent.get(field_name))) \
                and (not changey or _CHANGEY.search(field_name) or "mom" in field_name.lower()):
            return v
    if changey:
        # v21.1: the latest published value of the same change ('latest_published_mom_pct' for a month-over-month
        # target) carried forward, as v21 did through the level-field rule before 'mom' made a target change-type;
        # a level column ('latest_index_2019_03') is never the base of a change (v21 forecast +110% for an index)
        # Keep the planner-normal tokenizer byte-for-byte compatible with
        # v21.10/6739240. Calendar detection may split camelCase and digits,
        # but target matching historically did neither.
        tgt = set(re.split(r"[^a-z0-9]+", spec.target_name.lower())) - {""}
        for k, v in _numeric_fields(ent):
            ktok = set(re.split(r"[^a-z0-9]+", k.lower())) - {""}
            if _LATEST_CHANGE_FIELD.search(k) and not _NOT_QUANTITY.search(k) and ktok & tgt \
                    and not (skip_date_period and _date_period_field(k, ent.get(k))):
                return v
        return 0.0
    for k, v in _numeric_fields(ent):
        if _LEVEL_FIELD.search(k) and not _NOT_QUANTITY.search(k) \
                and not (skip_date_period and _date_period_field(k, ent.get(k))):
            return v
    return None


# v21.2: a quantity stated as a change ('percent change of the value of 1 EUR', 'bps_change', 'one-day abnormal return');
# 'revision' or 'surprise' alone is not read as one (a revised estimate is a level)
_CHANGE_QUANTITY = re.compile(r"percent(?:age)?[\s_-]+change|pct[\s_-]+change|%[\s_-]*change|\bchange[\s_]+(?:in|of)\b|"
                              r"\bbps[\s_-]+change|basis[\s_-]+points?[\s_-]+change|\breturns?\b", re.I)


def quantity_baseline(spec: TaskSpec, ent: dict, plan: dict | None, *, skip_date_period: bool = False) -> float | None:
    """Latest known value of the numeric quantity a classification interval sits on (e.g. consensus
    EPS for beat/miss, the latest estimate for a revision direction), or None. v21.2: when that quantity is a change
    (as the plan's interval quantity or the row's own unit text states it -- not the target's name, which a level such
    as 'eps_yoy_direction' carries too), a level field is never its base: a currency's quote (111.5 yen) stretched the
    interval of its percent change to [-2.6, 57] through the naive hedge."""
    qtext = f"{(plan or {}).get('interval_quantity') or ''} {ent.get('unit') or ''} {ent.get('units') or ''}"
    change_q = bool(_CHANGE_QUANTITY.search(qtext))
    field_name = (plan or {}).get("baseline_field")
    if isinstance(field_name, str) and field_name in ent:
        v = _f(ent.get(field_name))
        if v is not None and not (skip_date_period and _date_period_field(field_name, ent.get(field_name))) \
                and (not change_q or _CHANGEY.search(field_name) or _LATEST_CHANGE_FIELD.search(field_name)):
            return v
    if change_q:
        # the row's latest published value of the same change ('latest_mom_pct'), as baseline_value reads it
        for k, v in _numeric_fields(ent):
            if _LATEST_CHANGE_FIELD.search(k) and not _NOT_QUANTITY.search(k) \
                    and not (skip_date_period and _date_period_field(k, ent.get(k))):
                return v
        return None
    for k, v in _numeric_fields(ent):
        if _LEVEL_FIELD.search(k) and not _NOT_QUANTITY.search(k) \
                and not (skip_date_period and _date_period_field(k, ent.get(k))):
            return v
    return None


def fallback_forecast(spec: TaskSpec, ent: dict, plan: dict | None) -> Forecast:
    """Rule-based forecast; the interval is sized later by calibrate.finalize (never narrow)."""
    eid = ent["entity_id"]
    fc = Forecast(eid=eid)
    skip_date_period = bool(plan and plan.get("interval_is_probability") is True)
    if spec.target_type == "classification":
        fc.label = prior_label(spec.labels)
        fc.median = quantity_baseline(spec, ent, plan, skip_date_period=skip_date_period)
    elif spec.target_type == "regression":
        base = baseline_value(spec, ent, plan, skip_date_period=skip_date_period)
        fc.median = base if base is not None else 0.0
    else:  # ranking: a constant carries no ordering and scores the neutral 0.5
        fc.median = 0.0
    return fc


_BPS_UNIT = re.compile(r"(?<![a-z])bps?(?![a-z])|basis[\s_-]*points?", re.I)
_PCT_UNIT = re.compile(r"(?<![a-z])(?:pct|percent(?:age)?)(?![a-z])|%", re.I)
_UNIT_COLUMN = re.compile(r"(?:.*_)?units?", re.I)


def _unit_of(*texts) -> str | None:
    """'bps' or 'pct' when the texts name exactly one of the two units, else None."""
    text = " ".join(t for t in texts if isinstance(t, str))
    bps, pct = bool(_BPS_UNIT.search(text)), bool(_PCT_UNIT.search(text))
    return ("bps" if bps else "pct") if bps != pct else None


def target_unit(spec: TaskSpec, plan: dict | None) -> str | None:
    """The target's unit ('bps', 'pct' or None), read off the task's own target name and unit columns; the
    plan's description of the target is used only when the task names no unit."""
    cols = [str(v) for e in spec.entities for k, v in e.items() if _UNIT_COLUMN.fullmatch(str(k))]
    unit = _unit_of(spec.target_name, *cols)
    if unit is None and plan:
        unit = _unit_of(plan.get("target_quantity"), plan.get("interval_quantity"))
    return unit


def level_scale(spec: TaskSpec, plan: dict | None, base_field: str, level_quantity: str,
                formula: str) -> float | None:
    """Factor that puts a level's percent change (in percent) or difference (in the level's units) into the
    target's units: 100 for a yield in percent against a change in basis points (v8 submitted that change in
    percentage points, 100x too small). None when the two units cannot be reconciled, or when the formula does not
    fit the level (a percent change of a level that is itself a percent); the analysts then forecast the target
    itself, as they do when the plan gives no level rule."""
    tu = target_unit(spec, plan)
    lu = _unit_of(base_field, level_quantity)
    if formula == "pct_change":
        # v11: a change of a level that is itself a percent (a yield, a rate, a share) is a difference in points or
        # basis points, never a percent change of the percent: v9/v10 turned 2.85% -> 3.05% into 702 bp, not 20.
        if lu == "pct":
            return None
        return 100.0 if tu == "bps" else 1.0  # a price's percent change, stated in basis points
    if tu is None or tu == lu:
        return 1.0
    if lu is None:
        return None  # e.g. a basis-point or percent target against a level of unstated units
    return 100.0 if (lu, tu) == ("pct", "bps") else 0.01


# ---------------------------------------------------------------------------------------------- #
# Orchestration                                                                                   #
# ---------------------------------------------------------------------------------------------- #
class Forecaster:
    def __init__(self, spec: TaskSpec, corpus: Corpus, evidence: Evidence, client: HouseClient, t0: float):
        self.spec, self.corpus, self.ev, self.client, self.t0 = spec, corpus, evidence, client, t0
        self.policy = ArtifactPolicy(spec.cutoff)
        self.stat_weight = config.STAT_WEIGHT if self.policy.permits("blend") else 0.0
        self.plan: dict | None = None
        self.stats = ""
        self.stat: dict[str, StatForecast | None] = {}
        self.budget_scale = 1.0  # shrinks after a reply hits the output cap
        self.outlier_refs: dict[str, float] = {}  # flagged entity -> the value that flagged it
        self.group_size = config.GROUP_SIZE
        self.analyst_budget = config.REASONING_BUDGET_ANALYST
        self.anchor: dict[str, object] = {}  # classification entity -> code-computed quantity anchor
        self._halves: dict[str, float | None] | None = None  # table step scales, computed once (_submitted_range)
        for ent in spec.entities:
            eid = ent["entity_id"]
            own = list(evidence.specific.get(eid) or [])
            try:
                self.stat[eid] = entity_forecast(corpus, own + list(evidence.shared), ent, spec.target_name,
                                                 spec.horizon_days, set(own),
                                                 allow_own=spec.target_type != "classification", policy=self.policy)
            except Exception:  # noqa: BLE001 - a baseline must never break the unit
                self.stat[eid] = None
            try:  # v20: a CPI gasoline / energy / all-items first print is nowcast from weekly pump prices
                self.stat[eid] = pump_nowcast(corpus, own + list(evidence.shared), ent, spec.target_name,
                                              self.stat[eid], policy=self.policy) or self.stat[eid]
            except Exception:  # noqa: BLE001
                pass
        try:  # code-computed features of each entity (series, co-movement, driver, cross-section, going concern)
            evidence.features = entity_features(spec, corpus, evidence.specific, self.stat)
        except Exception:  # noqa: BLE001 - features must never break the unit
            evidence.features = {}
        self.prior: dict[str, object] = {}  # directional classification (v13: or regression) entity -> revision base rate
        if (spec.target_type == "classification" and directional_pair(spec.labels)) or spec.target_type == "regression":
            for ent in spec.entities:
                eid = ent["entity_id"]
                try:
                    pr = revision_prior(corpus, set(evidence.specific.get(eid) or []) | set(evidence.shared), ent, policy=self.policy)
                except Exception:  # noqa: BLE001 - a prior must never break the unit
                    pr = None
                if pr is not None:
                    self.prior[eid] = pr

    def baselines(self, group: list[str]) -> str:
        lines = []
        for eid in group:
            for sf in (self.stat.get(eid) or self.anchor.get(eid), self.prior.get(eid)):
                if sf is not None:
                    lines.append(f"{eid}: {sf.describe()}")
        return "\n".join(lines)

    # ---------------------------------------------------------------- plan
    def run_plan(self) -> dict | None:
        try:
            docs_for_stats = sorted(self.corpus.docs, key=lambda d: len(self.corpus.docs[d].text))
            self.stats = summarize(self.corpus, docs_for_stats)
        except Exception as exc:  # noqa: BLE001 - v21.5: column statistics are a hint; the analysts run without them
            log.warning("table statistics unavailable: %r", exc)
        try:
            msgs = planner_messages(self.spec, inventory(self.corpus), self.stats, self.ev.digest())
            rep = self.client.chat(msgs, reasoning_budget=config.REASONING_BUDGET_PLAN,
                                   temperature=config.PLAN_TEMPERATURE, seed=config.SEED, max_attempts=4,
                                   tag="plan")
            try:
                plan = parse_reply(rep, "interval_quantity")
            except ValueError:
                # Usually a reply that thought past the output cap. The plan is mostly structure (quantity,
                # units, label rules), so one thinking-free retry is cheap and far better than no plan.
                if (self.client.remaining <= config.REQUEST_BUDGET // 2
                        or time.time() > self.client.deadline - margin(self.client, 120)):
                    raise
                log.warning("plan reply unparseable (finish=%s); retrying without thinking", rep.finish)
                rep = self.client.chat(msgs, reasoning_budget=None, thinking=False,
                                       temperature=config.PLAN_TEMPERATURE, seed=config.SEED, max_attempts=3,
                                       tag="plan-nothink")
                plan = parse_reply(rep, "interval_quantity")
            if isinstance(plan, dict):
                # notes written from memory must not reach the analysts as facts
                for note in sanitize_plan(plan, self.spec, self.corpus):
                    log.info("plan note dropped: %s", note)
                from .calibrate import numeric_target_plan  # v21.11 (T4-002); calibrate imports this module

                if numeric_target_plan(plan, self.spec):
                    log.info("plan: probability reading dropped for a %s target in measured units", self.spec.target_type)
                self.plan = plan
                log.info("plan: %s", json.dumps(plan, ensure_ascii=False)[:1500])
                if isinstance(plan.get("target_from_level"), dict):
                    rule = self.level_rule()
                    log.info("level rule: %s", f"target = {rule['formula']} against {rule['base_field']} x "
                                               f"{rule['scale']:g}" if rule else "not used (does not apply)")
        except (LLMUnavailable, BudgetExhausted, ValueError) as exc:
            log.warning("planner unavailable: %s", exc)
        except Exception as exc:  # noqa: BLE001 - v21.5: a crashed plan (an odd plan or unit shape in the code around the
            # call) must not cost the analysts, who run without a plan as after a 403; v21.3 lost the whole model stage
            log.warning("planner crashed: %r", exc)
            self.plan = None  # a plan the code could not digest is not used further down either
        try:
            self.build_anchors()
        except Exception as exc:  # noqa: BLE001 - an anchor must never break the unit
            log.warning("anchors unavailable: %r", exc)
        return self.plan

    # ---------------------------------------------------------------- anchors
    def _eps_units(self) -> tuple[str | None, str | None]:
        """How an EPS anchor is expressed for this target: 'level', 'pct_change' or 'difference' (+ base column)."""
        plan = self.plan or {}
        text = f"{self.spec.target_name} {plan.get('interval_quantity') or ''}".lower()
        # v21.5: an 'earnings reaction' or announcement return is a stock's return, never an EPS-dollar anchor; any other
        # earnings target keeps v21.3's match, also without a plan (the fallback stage, a failed planner)
        if event_return_sd(self.spec) is not None or re.search(r"\b(?:reaction|returns?)\b",
                                                                 self.spec.target_name.lower().replace("_", " ")):
            return None, None
        if not re.search(r"\beps\b|eps_|_eps|earnings per share|earnings", text):
            return None, None
        if self.spec.target_type == "classification":
            if plan.get("interval_is_probability") is True:
                return None, None
            return "level", plan.get("baseline_field") if isinstance(plan.get("baseline_field"), str) else None
        rule = self.level_rule()
        if rule and rule["scale"] == 1.0:  # an EPS anchor is built in the level's own units
            return rule["formula"], rule["base_field"]
        name = self.spec.target_name.lower()
        if re.search(r"growth|pct|percent", name):
            return "pct_change", None
        return ("difference", None) if re.search(r"change|diff", name) else ("level", None)

    def build_anchors(self) -> None:
        """Code-computed statistical anchors (after the plan, which says how the target is expressed)."""
        units, base_field = self._eps_units()
        if units is None:
            return
        for ent in self.spec.entities:
            eid = ent["entity_id"]
            a = eps_anchor(self.corpus, set(self.ev.specific.get(eid) or []), ent, units, base_field, policy=self.policy)
            if a is None:
                continue
            if self.spec.target_type == "classification":
                self.anchor[eid] = a
            elif self.stat.get(eid) is None or isinstance(self.stat[eid], EpsAnchor):
                self.stat[eid] = a  # a series baseline stands; an earlier anchor gives way to the plan's reading
            log.info("anchor %s: %s", eid, a.describe()[:160])

    def _blend_anchor(self, fc: Forecast, anchor) -> None:
        """Equal-weight combination of the model's quantity with the code-computed anchor (level units)."""
        a, sd = anchor.level, anchor.level_sd
        fc.stat_q90, fc.stat_reliable = anchor.q90, False
        if fc.median is None:
            fc.median, fc.p05, fc.p95 = a, a - config.Z90 * sd, a + config.Z90 * sd
            return
        if abs(fc.median - a) > 10 * max(abs(a), abs(anchor.base), 0.1):
            return  # the model's quantity is on another scale: leave both alone
        new = (1.0 - self.stat_weight) * fc.median + self.stat_weight * a
        delta = new - fc.median
        fc.median = new
        if fc.p05 is not None:
            fc.p05 += delta
        if fc.p95 is not None:
            fc.p95 += delta

    def _center_on_prior(self, fc: Forecast) -> None:
        """Equal-weight combination of the model's quantity with the revision base rate's expected next estimate
        (the rule every other baseline follows), so the submitted interval sits where both point. A quantity more
        than half the latest estimate away from it is on another scale and is left alone."""
        pr = self.prior.get(fc.eid)
        if pr is None or fc.median is None or abs(fc.median - pr.value) > 0.5 * abs(pr.latest):
            return
        new = (1.0 - self.stat_weight) * fc.median + self.stat_weight * pr.value
        delta = new - fc.median
        fc.median = new
        if fc.p05 is not None:
            fc.p05 += delta
        if fc.p95 is not None:
            fc.p95 += delta

    def _rule_sources(self, eid: str) -> list[dict]:
        """Candidate label rules: the conditions the entity table states for each label, the numeric thresholds or
        bands the task's own label definitions state, the planner's, and 'higher / lower than the base' for
        directional labels with an anchor. Each must read on this row (v13: a five-band growth label could not be
        written in the planner's one-sided forms, and four of eight banks were labelled against their own submitted
        growth). v21 (review of v19): the task's own definitions come first and the planner's rules only fill in where
        the task states none -- a planner that misread the example unit's beat threshold as Q > 5 (the table says 1.575)
        turned a forecast of 1.80 into 'inline'."""
        ent = self.spec.entity(eid)
        labels = self.spec.labels
        out: list[dict] = []
        table = table_rules(labels, ent)
        if table:
            out.append(table)
        target = (getattr(self.spec, "raw", None) or {}).get("target")
        stated = assertion_rules(labels, target.get("label_assertions") if isinstance(target, dict) else None)
        if stated and rules_parse(stated, labels, ent):
            out.append(stated)
        rules = (self.plan or {}).get("label_rules")
        if isinstance(rules, dict) and rules and rules_parse(rules, labels, ent):
            out.append(rules)
        anchor = self.anchor.get(eid)
        if anchor is not None:
            directional = directional_rules(labels, anchor.base)
            if directional:
                out.append(directional)
        return out

    def _rules(self, eid: str) -> dict | None:
        """The first candidate label rules that read on this row (see `_rule_sources`), or None."""
        sources = self._rule_sources(eid)
        return sources[0] if sources else None

    def _prior_probs(self, eid: str) -> dict[str, float] | None:
        pr = self.prior.get(eid)
        pair = directional_pair(self.spec.labels)
        if pr is None or pair is None:
            return None
        return {pair[0]: pr.p_up, pair[1]: 1.0 - pr.p_up}

    def _blend_eps_direction(self, fc: Forecast) -> None:
        """v18: equal-weight combination of 'higher / lower than the prior-year quarter' label probabilities with the
        calibrated EPS direction rate (eps.EpsAnchor.p_up), when the labels compare the target EPS with the anchor's
        base. In the SEC replay a fall in the latest quarter left the next one a coin flip (49% up), where the anchor's
        own distribution called it down with 0.6-0.8 confidence."""
        anchor = self.anchor.get(fc.eid)
        pair = directional_pair(self.spec.labels)
        if not isinstance(anchor, EpsAnchor) or pair is None or not fc.probs:
            return
        rules = self._rules(fc.eid)
        if rules != directional_rules(self.spec.labels, anchor.base) and not _rules_compare_base(rules, anchor.base, pair, self.spec.entity(fc.eid)):
            return
        w = self.stat_weight
        prior = {pair[0]: anchor.p_up, pair[1]: 1.0 - anchor.p_up}
        fc.probs = {lab: (1 - w) * fc.probs.get(lab, 0.0) + w * prior[lab] for lab in self.spec.labels}

    def _blend_prior(self, fc: Forecast) -> None:
        """Equal-weight combination of the label probabilities with the entity's revision base rate."""
        prior = self._prior_probs(fc.eid)
        if prior is None:
            return
        w = self.stat_weight
        fc.probs = {lab: (1 - w) * (fc.probs or {}).get(lab, 0.0) + w * prior[lab] for lab in self.spec.labels}
        pr = self.prior[fc.eid]
        if fc.median is None:
            fc.median = pr.value
        if fc.stat_q90 is None:  # the series' own revision scale sizes the interval on the revised value
            fc.stat_q90, fc.stat_reliable = pr.q90, reliable_scale(pr)

    def _quantity_from_series(self, fc: Forecast) -> None:
        """A numeric classification with no row value takes the forecast of its own series.

        The series is the one `find_series` already names from the target or the entity id. A probability
        (schema or planner flag, not the word in the question) is left alone so its interval stays [0, 1]."""
        if fc.median is not None:
            return
        from .calibrate import is_probability

        if is_probability(self.plan, self.spec):
            return
        sf = self.stat.get(fc.eid)
        value = None if sf is None else getattr(sf, "value", None)
        if value is None or not math.isfinite(value):
            return
        fc.median = float(value)
        q90 = getattr(sf, "q90", None)
        if q90 and q90 > 0 and math.isfinite(q90):
            fc.stat_q90 = float(q90)
            fc.stat_reliable = reliable_scale(sf)

    def _label_from_rules(self, fc: Forecast, ent: dict) -> None:
        """v21.2: a row with no model reply and no code anchor takes the label its rules give to the quantity's naive
        value -- the latest value the row states, or no change for a change -- as the declared naive rule would, instead
        of the first label of the list ('rise' for a three-way yield move, 'above' for a CPI threshold)."""
        if any(lab.lower() in _INFORMED_PRIORS for lab in self.spec.labels):
            return  # a base-rate prior chosen for this label family (beat, up, no_event ...) stands
        q = fc.median
        if q is None:
            qtext = f"{(self.plan or {}).get('interval_quantity') or ''} {ent.get('unit') or ''} {ent.get('units') or ''}"
            q = 0.0 if _CHANGE_QUANTITY.search(qtext) else None
        if q is None:
            return
        for rules in self._rule_sources(fc.eid):
            lab = rule_label(rules, self.spec.labels, ent, q)
            if lab:
                if lab != fc.label:
                    log.info("fallback label %s: %s -> %s (its rules at the naive value %g)", fc.eid, fc.label, lab, q)
                fc.label = lab
                return

    def _anchor_only(self, ent: dict) -> Forecast | None:
        """A classification forecast from the code-computed anchor or prior alone (no model reply)."""
        eid = ent["entity_id"]
        prior = self._prior_probs(eid)
        if prior is not None and eid not in self.anchor:
            pr = self.prior[eid]
            fc = Forecast(eid=eid, source="stat", probs=prior, median=pr.value, stat_q90=pr.q90,
                          stat_reliable=reliable_scale(pr))
            fc.label = max(self.spec.labels, key=lambda lab: (fc.probs[lab], -self.spec.labels.index(lab)))
            return fc
        anchor = self.anchor.get(eid)
        rules = self._rules(eid)
        if anchor is None or not rules:
            return None
        fc = Forecast(eid=eid, source="stat")
        self._blend_anchor(fc, anchor)
        self._settle_label(fc, ent)
        return fc if fc.label is not None else None

    def plan_terms(self, label: str | None = None) -> list[str]:
        """Planner search terms plus label cues: every label's cues for balanced analyst evidence,
        only the predicted label's cues when retrieving that entity's citations."""
        terms = list((self.plan or {}).get("search_terms") or [])
        cues = (self.plan or {}).get("label_cues") or {}
        if isinstance(cues, dict):
            for lab, v in cues.items():
                if isinstance(v, list) and (label is None or lab == label):
                    terms += v
        return [t for t in terms if isinstance(t, str) and t.strip()][:30]

    def level_rule(self) -> dict | None:
        """The planner's 'target = f(level, base column)' rule, validated, with the factor into the target's
        units; None when it does not apply."""
        if self.spec.target_type not in ("regression", "ranking"):
            return None
        r = (self.plan or {}).get("target_from_level")
        if not isinstance(r, dict):
            return None
        base, formula, lq = r.get("base_field"), r.get("formula"), r.get("level_quantity")
        if formula not in ("pct_change", "difference") or not isinstance(lq, str) or not lq.strip():
            return None
        vals = [_f(e.get(base)) if isinstance(base, str) else None for e in self.spec.entities]
        if not vals or any(v is None for v in vals):
            return None
        if formula == "pct_change" and any(v <= 0 for v in vals):
            return None
        scale = level_scale(self.spec, self.plan, base, lq, formula)
        if scale is None:
            return None
        return {"level_quantity": lq.strip(), "base_field": base, "formula": formula, "scale": scale}

    def _to_target(self, eid: str, x: float | None, rule: dict) -> float | None:
        if x is None:
            return None
        base = _f(self.spec.entity(eid).get(rule["base_field"]))
        if base is None:
            return None
        return ((x / base - 1.0) * 100.0 if rule["formula"] == "pct_change" else x - base) * rule["scale"]

    def interval_quantity(self) -> str:
        q = (self.plan or {}).get("interval_quantity")
        if isinstance(q, str) and q.strip():
            return q.strip()
        return f"the realized {self.spec.target_name.replace('_', ' ')}"

    # ---------------------------------------------------------------- analysts
    def groups(self) -> list[list[str]]:
        ids = self.spec.entity_ids
        n = len(ids)
        if n <= 8 and (self.ev.small or n <= 4):
            return [ids]
        if self.spec.target_type == "ranking" and n <= 8:
            return [ids]  # a ranking is a relative judgement: one analyst sees the whole cross-section
        # (larger rosters are split: a 10-entity reply plus its reasoning overran the 4,000-token cap;
        # every group still sees the full entity table and all statistical baselines)
        g = max(1, min(self.group_size, 8))
        k = math.ceil(n / g)
        size = math.ceil(n / k)
        return [ids[i : i + size] for i in range(0, n, size)]

    def _one_call(self, group: list[str], sample: int) -> dict[str, dict]:
        evidence = self.ev.render(group)
        msgs = analyst_messages(self.spec, self.plan, group, self.stats, evidence, self.interval_quantity(),
                                baselines=self.baselines(self.spec.entity_ids), sample=sample,
                                level_rule=self.level_rule())
        # thinking and the reply share the 4,000-token cap: a shorter reply leaves more room to think
        room = config.MAX_TOKENS - 200 - config.ANSWER_TOKENS_PER_ENTITY * len(group)
        budget = int(max(800, min(self.analyst_budget, room)) * self.budget_scale)
        rep = self.client.chat(msgs, reasoning_budget=budget,
                               temperature=config.TEMPERATURE, seed=config.SEED * 1000 + sample * 17 + len(group),
                               max_attempts=1, tag=f"analyst g{group[0]} s{sample}")
        if rep.finish == "length":
            # The reply ran into the 4,000-token cap (thinking included): think less from now on.
            self.budget_scale = max(0.5, self.budget_scale * 0.75)
        try:
            obj = parse_reply(rep, "forecasts")
        except ValueError:
            try:
                if (self.client.remaining > self.end_reserve()
                        and time.time() < self.client.deadline - margin(self.client, 30)):
                    iq = (self.level_rule() or {}).get("level_quantity") or self.interval_quantity()
                    rep2 = self.client.chat(repair_messages(rep.content + "\n" + rep.reasoning[-6000:], group,
                                                            self.spec, iq),
                                            reasoning_budget=None, thinking=False, temperature=0.0, tag="repair")
                    obj = extract_json(rep2.content, key="forecasts")
                else:
                    raise ValueError("no budget for a repair")
            except (ValueError, LLMUnavailable) as exc:
                if rep.finish == "length" and len(group) > 1:
                    raise Truncated(f"reply truncated and unrepairable for {len(group)} entities") from exc
                raise
        out: dict[str, dict] = {}
        items = obj.get("forecasts") if isinstance(obj, dict) else None
        for it in items or []:
            if isinstance(it, dict) and it.get("entity_id") in group and self._usable(it):
                out[it["entity_id"]] = it
        if not out:
            raise ValueError("reply parsed but carried no usable forecast")
        return out

    def _usable(self, it: dict) -> bool:
        """A forecast item the aggregator can use: label probabilities (or a label) for a
        classification unit, a finite median otherwise."""
        if self.spec.target_type == "classification":
            probs = it.get("probs")
            if isinstance(probs, dict) and any(_f(v) for v in probs.values()):
                return True
            return isinstance(it.get("label"), str) and it["label"].strip().lower() in {l.lower() for l in self.spec.labels}
        return _f(it.get("median")) is not None

    def choose_layout(self) -> None:
        """The deepest (group size, thinking budget) that still samples every group at least twice before the
        analyst deadline, at the speed the route has shown so far (a cautious guess before any call)."""
        speed = self.client.speed or 40.0
        end = self.client.deadline - self.reserve() - margin(self.client, 25)
        window = max(0.0, end - time.time())
        n = max(1, len(self.spec.entity_ids))
        for size, budget in config.LAYOUTS:
            calls = math.ceil(n / size)
            seconds = config.LATENCY_MARGIN * (budget + config.ANSWER_TOKENS_PER_ENTITY * min(size, n) + 200) / speed
            if max(1, config.CONCURRENCY) * window / max(seconds, 1.0) >= 2 * calls:
                break
        self.group_size, self.analyst_budget = size, budget
        log.info("layout: groups of %d, thinking %d (route %.0f tok/s, %.0f s left)", size, budget, speed, window)

    def end_reserve(self) -> int:
        """Requests the end of the unit keeps: the outlier re-check, the review when on, and two for the reasons (a call
        and its retry or follow-up). v21: v20's review slot had doubled as the reasons' spare; with the review off and
        JSON repairs eating into the reserve, three v21 units on a route that needs many repairs (deepinfra) were left
        without a request for the reasons' retry and fell back to code-built reasons."""
        return 1 + (1 if config.REVIEW else 0) + (2 if config.REASONS else 0)

    def reserve(self) -> float:
        """Seconds kept free after the analysts: the review, then the reasons (v21.5: shrunk with the unit's clock)."""
        return margin(self.client, (config.REVIEW_RESERVE_SEC if config.REVIEW else 0.0)
                      + (config.REASONS_RESERVE_SEC if config.REASONS else 0.0))

    def run_analysts(self) -> dict[str, list[dict]]:
        """Run (group, sample) jobs concurrently. A failed job is re-queued with exponential
        backoff (the route answers 503 immediately when overloaded) while budget and time last;
        every group's first sample outranks any group's extra samples."""
        self.choose_layout()
        groups = self.groups()
        reserve = self.end_reserve() + 1  # and one for a JSON repair of the last analyst replies
        if self.spec.target_type == "classification":
            reserve += config.EXTRA_VOTES  # v21: kept for focused votes on undecided labels (extra_votes)
        end = self.client.deadline - self.reserve()
        avail = max(0, self.client.remaining - reserve)
        k = max(1, min(config.MAX_SAMPLES, avail // max(1, len(groups))))
        if self.anchored():
            k = min(k, config.ANCHORED_SAMPLES)  # v21: the requests left go to the reasons
        samples: dict[str, list[dict]] = defaultdict(list)
        if not self.client.enabled or avail <= 0:
            return samples
        log.info("analysts: %d groups x %d samples (budget left %d)", len(groups), k, self.client.remaining)
        # job = [sample, group_index, attempts, not_before]
        queue: list[list] = [[s, gi, 0, 0.0] for s in range(k) for gi in range(len(groups))]
        pool = ThreadPoolExecutor(max_workers=max(1, config.CONCURRENCY))
        pending: dict = {}
        try:
            while queue or pending:
                now = time.time()
                # keep two requests back for the outlier re-check and a JSON repair
                stop_new = now > end - margin(self.client, 25) or self.client.remaining <= reserve
                if not stop_new:
                    queue.sort(key=lambda j: (j[0], j[3]))
                    for job in list(queue):
                        if len(pending) >= config.CONCURRENCY:
                            break
                        if job[3] <= now:
                            queue.remove(job)
                            fut = pool.submit(self._one_call, groups[job[1]], job[0])
                            pending[fut] = job
                elif not pending:
                    break
                if not pending:
                    wake = min((j[3] for j in queue), default=now + 1)
                    time.sleep(max(0.2, min(5.0, wake - now)))
                    continue
                timeout = max(0.5, min(5.0, end - time.time()))
                done, _ = wait(list(pending), timeout=timeout, return_when=FIRST_COMPLETED)
                if not done and time.time() > end:
                    log.warning("analyst deadline reached with %d calls in flight", len(pending))
                    self._late_replies(pending, groups, samples)
                    break
                for fut in done:
                    job = pending.pop(fut)
                    s, gi, attempts = job[0], job[1], job[2] + 1
                    try:
                        for eid, it in fut.result().items():
                            samples[eid].append(it)
                    except BudgetExhausted:
                        queue.clear()
                    except Truncated as exc:
                        # split the group: two smaller replies fit where one did not
                        log.warning("analyst g%d s%d: %s; splitting the group", gi, s, exc)
                        g = groups[gi]
                        for half in (g[: len(g) // 2], g[len(g) // 2:]):
                            groups.append(half)
                            queue.append([s, len(groups) - 1, attempts, time.time()])
                    except (LLMUnavailable, ValueError) as exc:
                        log.warning("analyst g%d s%d attempt %d failed: %s", gi, s, attempts, exc)
                        if re.search(r"http (?:429|50[234])", str(exc)):
                            # throttled, not wrong: wait for capacity (the budget and the deadline still bound it)
                            queue.append([s, gi, attempts - 1, max(time.time() + 1.0, self.client.cool_until)])
                            continue
                        limit = 4 if s == 0 else 2
                        if attempts < limit and "refused" not in str(exc):
                            queue.append([s, gi, attempts, time.time() + min(20.0, 3.0 * 2 ** (attempts - 1))])
                    except Exception as exc:  # noqa: BLE001 - never let one call kill the unit
                        log.warning("analyst g%d s%d crashed: %r", gi, s, exc)
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
        return samples

    def _late_replies(self, pending: dict, groups: list[list[str]], samples: dict[str, list[dict]]) -> None:
        """v21.5: calls still in flight at the analyst deadline whose group has an entity without any sample are waited
        for until the reasons gate, and their replies kept: a model forecast outweighs a model-written reason (code
        reasons stand in). On a 300 s card the analyst deadline (98 s) fell inside one House round trip (18-52 s, sent
        at about 49 s): the replies came 0.1-2 s late and were dropped, and the unit went out on code forecasts."""
        late_end = self.client.deadline - margin(self.client, 25)
        need = [f for f, job in pending.items() if any(not samples.get(e) for e in groups[job[1]])]
        if not need or time.time() >= late_end:
            return
        done, _ = wait(need, timeout=max(0.0, late_end - time.time()))
        kept = 0
        for fut in done:
            pending.pop(fut, None)
            try:
                for eid, it in fut.result().items():
                    samples[eid].append(it)
                kept += 1
            except Exception as exc:  # noqa: BLE001 - a late call that failed changes nothing
                log.warning("late analyst reply failed: %r", exc)
        log.info("late analyst replies kept: %d of %d waited for", kept, len(need))

    def anchored(self) -> bool:
        """v21: a regression unit whose every entity has a code baseline with a measured or calibrated scale (a
        backtested statistical forecast, the pump-price nowcast, the EPS anchor, a vintage-table prior). There the
        analysts moved v20's answers by 0.01-0.25 standard deviations, so a few samples suffice."""
        if self.spec.target_type != "regression" or not self.spec.entity_ids:
            return False
        return all(isinstance(self.stat.get(e), EpsAnchor) or reliable_scale(self.stat.get(e))
                   or reliable_scale(self.prior.get(e)) for e in self.spec.entity_ids)

    @staticmethod
    def _vote(it: dict) -> str | None:
        probs = it.get("probs")
        if isinstance(probs, dict) and probs:
            scored = [(v, k) for k, v in probs.items() if _f(v) is not None]
            if scored:
                return str(max(scored)[1]).strip().lower()
        lab = it.get("label")
        return lab.strip().lower() if isinstance(lab, str) else None

    def undecided(self, samples: dict[str, list[dict]]) -> list[str]:
        """Classification entities whose samples split, most split first: the top label holds under 75% of the votes."""
        out = []
        for eid in self.spec.entity_ids:
            votes = [v for v in (self._vote(it) for it in samples.get(eid) or []) if v]
            if len(votes) >= 2:
                share = max(votes.count(v) for v in set(votes)) / len(votes)
                if share < 0.75:
                    out.append((share, eid))
        return [e for _, e in sorted(out)]

    def extra_votes(self, samples: dict[str, list[dict]]) -> dict[str, list[dict]]:
        """v21: focused single-entity calls for the classification labels the samples left undecided, with the
        requests run_analysts kept back (EXTRA_VOTES). Same-code runs of v12-v20 flipped such labels between runs
        (post-earnings AAPL took three labels in four runs, a credit-event entity flipped every other run): more votes,
        each looking at that entity alone, make the decision rest on the model's leaning rather than on one draw."""
        if self.spec.target_type != "classification" or not self.client.enabled:
            return samples
        budget = min(config.EXTRA_VOTES, self.client.remaining - self.end_reserve())
        end = self.client.deadline - self.reserve()
        targets = self.undecided(samples)[:max(0, budget)]
        if budget <= 0 or not targets or time.time() > end - margin(self.client, 30):
            return samples
        per = max(1, budget // len(targets))
        jobs = [(eid, s) for s in range(per) for eid in targets][:budget]
        log.info("extra votes: %d single-entity calls on %s", len(jobs), ", ".join(targets))
        pool = ThreadPoolExecutor(max_workers=max(1, config.CONCURRENCY))
        try:
            futs = {pool.submit(self._one_call, [eid], 50 + s): eid for eid, s in jobs}
            done, _ = wait(list(futs), timeout=max(1.0, end - margin(self.client, 30) - time.time()))
            for fut in done:
                try:
                    for eid, it in fut.result().items():
                        samples.setdefault(eid, []).append(it)
                except Exception as exc:  # noqa: BLE001 - an extra vote is optional
                    log.warning("extra vote on %s failed: %s", futs[fut], exc)
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
        return samples

    # ---------------------------------------------------------------- aggregate
    def aggregate(self, samples: dict[str, list[dict]]) -> dict[str, Forecast]:
        out: dict[str, Forecast] = {}
        for ent in self.spec.entities:
            eid = ent["entity_id"]
            items = samples.get(eid) or []
            fc = self._aggregate_one(ent, items) if items else None
            if fc is None and self.spec.target_type == "classification" and (eid in self.anchor or eid in self.prior):
                fc = self._anchor_only(ent)
            if fc is None:
                fc = fallback_forecast(self.spec, ent, self.plan)
                if self.spec.target_type == "classification":
                    self._quantity_from_series(fc)
                    self._label_from_rules(fc, ent)
                    anchor = self.anchor.get(eid)
                    if anchor is not None:
                        # v21.3: without label rules the anchor was ignored here, and a growth-band row's EPS went out at
                        # last year's value with a +/-20 default; its level and calibrated scale size the quantity
                        self._blend_anchor(fc, anchor)
            elif self.spec.target_type == "classification" and fc.median is None:
                # A label-only model reply has a label and an empty median. Copy the series forecast when the
                # target already names one, so a numeric classification is not submitted as [0, 1].
                self._quantity_from_series(fc)
            out[eid] = fc
        try:
            self.recheck(out)
        except Exception as exc:  # noqa: BLE001 - a second look is optional
            log.warning("recheck failed: %r", exc)
        if self.spec.target_type == "ranking":
            self._rank_by_baseline = self._baselines_rank(out)
        for fc in out.values():
            self._combine_with_stats(fc)
        if self.spec.target_type == "ranking":
            self._place_uninformed(out)
        try:
            self.review(out)
        except Exception as exc:  # noqa: BLE001 - a review is optional
            log.warning("review failed: %r", exc)
        return out

    def _describe_forecast(self, eid: str, fc: Forecast, rule: dict | None) -> str:
        def num(x):
            return "?" if x is None else f"{x:.4g}"
        if self.spec.target_type == "classification":
            probs = ", ".join(f"{k} {v:.2f}" for k, v in (fc.probs or {}).items())
            return f"label {fc.label} ({probs}); quantity {num(fc.median)} [{num(fc.p05)}, {num(fc.p95)}]"
        text = f"{num(fc.median)} [{num(fc.p05)}, {num(fc.p95)}] in target units"
        base = _f(self.spec.entity(eid).get(rule["base_field"])) if rule else None
        if rule and base is not None and fc.median is not None:
            x = fc.median / rule["scale"]
            level = base * (1 + x / 100.0) if rule["formula"] == "pct_change" else base + x
            text += f" (implied level {level:.4g} against base {base:.4g})"
        return text

    def review(self, forecasts: dict[str, Forecast]) -> None:
        """One reviewing pass over the whole cross-section. An entity whose forecast contradicts its code-computed
        facts or the cross-section gets a second opinion, blended equal-weight with the current forecast."""
        if (not config.REVIEW or self.client.remaining < 1
                or time.time() > self.client.deadline - margin(self.client, 60)):
            return
        if not any(fc.source == "llm" for fc in forecasts.values()):
            return  # nothing from the analysts to review (the early fallback, or every call failed)
        rule = self.level_rule()
        rows = []
        for ent in self.spec.entities:
            eid = ent["entity_id"]
            facts = [sf.describe()[:320] for sf in (self.stat.get(eid) or self.anchor.get(eid), self.prior.get(eid))
                     if sf is not None]
            facts += [line[:320] for line in (self.ev.lfl.get(eid) or [])[:2]]
            facts += [line[:320] for line in (self.ev.features.get(eid) or [])[:3]]
            rows.append(f"{eid}: FACTS: {' / '.join(facts) or '(none)'} | FORECAST: "
                        f"{self._describe_forecast(eid, forecasts[eid], rule)}")
        iq = rule["level_quantity"] + " (the LEVEL; the target is derived from it)" if rule else self.interval_quantity()
        expected = (config.REVIEW_REASONING + 150 * len(rows) + 300) / (self.client.speed or 40.0)
        keep = margin(self.client, 30.0) if config.REASONS else 0.0  # leave the reasons call a chance
        if time.time() + expected > self.client.deadline - margin(self.client, 10) - keep:
            log.info("review skipped: about %.0f s needed, too close to the deadline", expected)
            return
        rep = self.client.chat(review_messages(self.spec, self.plan, "\n".join(rows), iq),
                               reasoning_budget=config.REVIEW_REASONING, temperature=0.2,
                               seed=config.SEED + 11, max_attempts=2, tag="review")
        obj = parse_reply(rep, "revisions")
        for it in (obj.get("revisions") if isinstance(obj, dict) else None) or []:
            eid = it.get("entity_id") if isinstance(it, dict) else None
            if eid not in forecasts or not self._usable(it):
                continue
            new = self._aggregate_one(self.spec.entity(eid), [it])
            old = forecasts[eid]
            if new is None:
                continue
            changed = False
            if self.spec.target_type == "classification" and new.model_probs and (old.model_probs or old.probs):
                base = old.model_probs or old.probs
                old.model_probs = {lab: 0.5 * base[lab] + 0.5 * new.model_probs[lab] for lab in self.spec.labels}
                changed = True
            if new.median is not None and old.median is not None and not self._plausible_move(old, new.median):
                # v13: a second opinion on another scale is no opinion: a reviewer that answered in the target's units
                # where the level rule expects the level turned INDPRO's -0.40% into a level of -0.40, a -100% revision,
                # and the 50/50 blend submitted -50%
                log.info("review %s: quantity %g ignored, off the scale of %g", eid, new.median, old.median)
            elif new.median is not None and old.median is not None:  # the quantity moves with the label
                lo = min(x for x in (old.p05, new.p05, old.median, new.median) if x is not None)
                hi = max(x for x in (old.p95, new.p95, old.median, new.median) if x is not None)
                old.median, old.p05, old.p95 = 0.5 * (old.median + new.median), lo, hi
                changed = True
            if not changed:
                continue
            if self.spec.target_type == "classification":
                self._settle_label(old, self.spec.entity(eid))  # v11: v10 re-read the label off blended probabilities
            log.info("review %s: %s (%s)", eid, self._describe_forecast(eid, old, rule)[:120],
                     str(it.get("reason") or "")[:120])

    def outliers(self, forecasts: dict[str, Forecast]) -> dict[str, str]:
        """Model forecasts far outside the cross-section (robust z > 4) or their own baseline (> 3x its
        90% backtest error). Only numeric targets that are comparable across entities are screened."""
        if self.spec.target_type not in ("regression", "ranking"):
            return {}
        vals = {e: f.median for e, f in forecasts.items() if f.source == "llm" and f.median is not None}
        flagged: dict[str, str] = {}
        if len(vals) >= 4:
            xs = list(vals.values())
            med = statistics.median(xs)
            mad = statistics.median([abs(x - med) for x in xs])
            if mad > 0:
                for eid, x in vals.items():
                    # scale-aware: an entity with its own (backtested) volatility is judged against it,
                    # so a genuinely volatile member (energy among CPI components) is not an outlier
                    sf = self.stat.get(eid)
                    scale = max(1.4826 * mad, (sf.q90 or 0.0) if sf is not None else 0.0)
                    z = abs(x - med) / scale
                    if z > 4.0:
                        flagged[eid] = (f"first pass {x:g} vs cross-section median {med:g} (robust z {z:.1f})")
                        self.outlier_refs[eid] = med
        for eid, x in vals.items():
            sf = self.stat.get(eid)
            if sf is not None and sf.q90 and abs(x - sf.value) > 3.0 * sf.q90:
                scale_name = "90% backtest error" if sf.mae is not None else "theoretical 90% half-width"
                flagged[eid] = (flagged.get(eid, f"first pass {x:g}") +
                                f"; statistical baseline {sf.value:.4g} with {scale_name} {sf.q90:.3g}")
                self.outlier_refs[eid] = sf.value  # the entity's own baseline is the more specific reference
        return dict(list(flagged.items())[:4])

    def recheck(self, forecasts: dict[str, Forecast]) -> None:
        flagged = self.outliers(forecasts)
        if not flagged:
            return
        log.info("recheck: %s", flagged)
        try:
            if self.client.remaining < 2 or time.time() > self.client.deadline - margin(self.client, 60):
                return
            group = list(flagged)
            msgs = recheck_messages(self.spec, self.plan, flagged, self.stats, self.ev.render(group),
                                    self.interval_quantity(), self.baselines(group), level_rule=self.level_rule())
            rep = self.client.chat(msgs, reasoning_budget=config.REASONING_BUDGET_ANALYST, temperature=0.3,
                                   seed=config.SEED + 7, max_attempts=2, tag="recheck")
            obj = parse_reply(rep, "forecasts")
            for it in (obj.get("forecasts") if isinstance(obj, dict) else None) or []:
                eid = it.get("entity_id") if isinstance(it, dict) else None
                if eid not in flagged:
                    continue
                new = self._aggregate_one(self.spec.entity(eid), [it])
                if new is None or new.median is None:
                    continue
                old = forecasts[eid]
                ref = self.outlier_refs.get(eid)
                if ref is not None and old.median is not None and abs(new.median - ref) > abs(old.median - ref) \
                        and not self._plausible_move(old, new.median):
                    log.info("recheck %s: %g ignored, further off than the first pass %g", eid, new.median, old.median)
                    continue
                log.info("recheck %s: %s -> %s", eid, old.median, new.median)
                new.support = new.support or old.support
                new.why = {**old.why, **new.why}
                new.n_samples = old.n_samples + 1
                new.rechecked = True
                forecasts[eid] = new
        finally:
            for eid in flagged:
                if not forecasts[eid].rechecked:
                    self._shrink_unconfirmed(forecasts[eid])

    @staticmethod
    def _plausible_move(fc: Forecast, x: float) -> bool:
        """v13: whether a second opinion's quantity x is on the scale of the current forecast: within ten times the
        larger of its measured 90% scale and its own half-ranges. A legitimate correction moves a forecast by a few of
        its spreads; a quantity a hundred spreads away answered in other units."""
        spreads = [v for v in (fc.stat_q90, None if fc.p95 is None else fc.p95 - fc.median,
                               None if fc.p05 is None else fc.median - fc.p05) if v is not None and v > 0]
        if fc.median is None or not spreads or not math.isfinite(x):
            return math.isfinite(x)
        return abs(x - fc.median) <= 10.0 * max(spreads)

    def _shrink_unconfirmed(self, fc: Forecast) -> None:
        """A flagged outlier that no second look confirmed moves halfway to the value that flagged it.

        In development such outliers were mostly period or base errors, and under absolute error the
        midpoint is the safer centre while either value could be right; the range is stretched so it
        still covers both.
        """
        ref = self.outlier_refs.get(fc.eid)
        if ref is None or fc.median is None or not math.isfinite(ref):
            return
        first = fc.median
        spread_lo = first - fc.p05 if fc.p05 is not None and fc.p05 < first else 0.0
        spread_hi = fc.p95 - first if fc.p95 is not None and fc.p95 > first else 0.0
        fc.median = 0.5 * (first + ref)
        fc.p05 = min(first, ref) - spread_lo
        fc.p95 = max(first, ref) + spread_hi
        log.info("unconfirmed outlier %s: %g -> %g", fc.eid, first, fc.median)

    def _combine_with_stats(self, fc: Forecast) -> None:
        """Equal-weight combination of the model's forecast with the backtested statistical baseline.

        Combining a judgmental forecast with a simple statistical one is the most robust default in
        the forecasting literature (equal weights, no fitted weights). The model's own range is kept
        around the combined centre; the baseline's backtest error becomes the interval's scale.
        Without a baseline from the entity's own history, the naive rule (zero change, or the latest value
        carried forward) is the simplest statistical forecast there is, and a regression forecast is
        combined with it the same way: a model forecast is never submitted unshrunk.
        """
        sf = self.stat.get(fc.eid)
        if self.spec.target_type not in ("regression", "ranking") or fc.median is None:
            return
        # v21.1: a vintage table's measured revision scale outranks a series read off an unnamed table
        own = getattr(getattr(sf, "series", None), "kind", "")  # an EPS anchor has no series
        rev = self._revision_in_target(fc.eid) if sf is None or own else None
        if rev is not None:
            # v13: a regression on the next estimate of a vintage series (its level, or its revision against the latest
            # estimate) has a measured baseline and scale in the vintage table, as the direction labels do; v12 sized
            # such rows from the planner's single deviation (a 2.5-point percent revision for payrolls: +/-5%)
            value, fc.stat_q90, fc.stat_reliable = rev
            new = (1.0 - self.stat_weight) * fc.median + self.stat_weight * value if fc.source == "llm" else value
        elif sf is None:
            if fc.source != "llm" or self.spec.target_type != "regression":
                return  # a ranking needs no anchor: shifting every entity towards one value keeps the order
            skip_date_period = bool(self.plan and self.plan.get("interval_is_probability") is True)
            ref = baseline_value(
                self.spec, self.spec.entity(fc.eid), self.plan, skip_date_period=skip_date_period
            )
            if ref is None or not math.isfinite(ref):
                return
            new = (1.0 - self.stat_weight) * fc.median + self.stat_weight * ref
        else:
            fc.stat_q90, fc.stat_reliable = sf.q90, reliable_scale(sf)
            if self.spec.target_type == "ranking" and own:
                # v21.1: a series read off an unnamed table picks its method entity by entity (zero change for most,
                # momentum for a few); blended into a ranking it reorders the rows by method, not by evidence
                # (51 state unemployment changes: rank correlation 0.58 for the model alone, 0.24 blended). It sizes
                # the interval only (v21.5: with or without a model reply).
                return
            if fc.source == "llm":
                new = (1.0 - self.stat_weight) * fc.median + self.stat_weight * sf.value
            else:
                # no model forecast: the backtested baseline beats the rule-based one. v21.5: a ranking row too -- v21.3
                # left it at a constant 0, so a full fallback carried no ordering (ranking 0.5) where the baseline ranks
                # positioning at 0.565 (CFTC replay, v19); it is in the target's units, as the model's values are. Only
                # when the rows' baselines come from one rule (_baselines_rank)
                if self.spec.target_type == "ranking" and not getattr(self, "_rank_by_baseline", True):
                    return
                new = sf.value
        delta = new - fc.median
        fc.median = new
        if fc.p05 is not None:
            fc.p05 += delta
        if fc.p95 is not None:
            fc.p95 += delta

    def _baselines_rank(self, forecasts: dict[str, Forecast]) -> bool:
        """v21.5: whether the ranking rows without a model reply may take their code baselines' values: only when those
        baselines come from one rule -- the ratio-change reversion (stats._ratio_change_forecast, with its drift switch:
        the rule the CFTC replay ranks at 0.565), or a single method throughout. Methods picked entity by entity order
        the rows by method, not by evidence (v17: switching among the methods lost on both measures; momentum 0.479
        against the constant's 0.500), so a mix keeps v21.3's constant."""
        sfs = [self.stat.get(f.eid) for f in forecasts.values() if f.source != "llm"]
        sfs = [sf for sf in sfs if sf is not None and not getattr(getattr(sf, "series", None), "kind", "")]
        if all(getattr(getattr(sf, "series", None), "denom", None) is not None for sf in sfs):
            return True
        return len({getattr(sf, "method", None) for sf in sfs}) <= 1

    def _place_uninformed(self, forecasts: dict[str, Forecast]) -> None:
        """v21.5: a ranking row with neither a model reply nor a code baseline that sets its value, among rows that have
        one, ranks in the middle of them rather than at 0: on a level-valued target (a yield, a ratio) 0 sorts it to the
        bottom. On a change-valued target 0 is no change, the naive value, and stays."""
        if _CHANGEY.search(self.spec.target_name):
            return

        def informed(f: Forecast) -> bool:  # a model reply, or a baseline that set the value (_combine_with_stats)
            sf = self.stat.get(f.eid)
            return f.source == "llm" or (getattr(self, "_rank_by_baseline", True) and sf is not None
                                         and not getattr(getattr(sf, "series", None), "kind", ""))

        known = [f.median for f in forecasts.values()
                 if informed(f) and f.median is not None and math.isfinite(f.median)]
        if not known:
            return  # no row carries an ordering: the constant stands (neutral)
        mid = statistics.median(known)
        for fc in forecasts.values():
            if not informed(fc):
                delta = mid - (fc.median or 0.0)
                fc.median = mid
                if fc.p05 is not None:
                    fc.p05 += delta
                if fc.p95 is not None:
                    fc.p95 += delta

    def _revision_in_target(self, eid: str) -> tuple[float, float, bool] | None:
        """(expected value, 90% scale, measured?) of a regression row's revision base rate in the target's units: the
        level itself when the target is the next estimate, its percent change or difference against the latest estimate
        through the plan's level rule; None when the target is neither."""
        pr = self.prior.get(eid)
        if pr is None or self.spec.target_type != "regression":
            return None
        ent = self.spec.entity(eid)
        rule = self.level_rule()
        if rule:
            base = _f(ent.get(rule["base_field"]))
            if base is None or base == 0 or abs(base - pr.latest) > 1e-9 * max(1.0, abs(base)):
                return None
            if rule["formula"] == "pct_change":
                k = 100.0 * rule["scale"] / abs(base)
                return (pr.value / base - 1.0) * 100.0 * rule["scale"], pr.q90 * k, reliable_scale(pr)
            return (pr.value - base) * rule["scale"], pr.q90 * rule["scale"], reliable_scale(pr)
        skip_date_period = bool(self.plan and self.plan.get("interval_is_probability") is True)
        naive = baseline_value(self.spec, ent, self.plan, skip_date_period=skip_date_period)
        if naive is not None and abs(naive - pr.latest) <= 1e-9 * max(1.0, abs(naive)):
            return pr.value, pr.q90, reliable_scale(pr)
        return None

    def _support_ids(self, eid: str, items: list[dict]) -> tuple[list[int], dict[int, str]]:
        allowed = self.ev.allowed_support_docs(eid)
        votes: Counter = Counter()
        why: dict[int, str] = {}
        for it in items:
            for sp in it.get("support") or []:
                if isinstance(sp, dict):
                    ref, reason = sp.get("id"), sp.get("why")
                else:
                    ref, reason = sp, None
                m = re.search(r"(\d+)", str(ref or ""))
                if not m:
                    continue
                cid = int(m.group(1))
                c = self.corpus.chunk(cid)
                if c is None or c.doc_id not in allowed:
                    continue
                votes[cid] += 1
                if isinstance(reason, str) and reason.strip() and cid not in why:
                    why[cid] = reason.strip()[:200]
        return [cid for cid, _ in votes.most_common()], why

    def _submitted_range(self, fc: Forecast, ent: dict) -> tuple[float, float, float] | None:
        """(point, lo, hi) as the answer will submit this row's quantity (calibrate.quantity_range)."""
        from .calibrate import quantity_range, table_halves  # calibrate imports this module

        if self._halves is None:
            try:
                self._halves = table_halves(self.spec, self.corpus, self.ev)
            except Exception:  # noqa: BLE001 - a scale hint must never break the unit
                self._halves = {}
        return quantity_range(self.spec, fc, self.plan, ent=ent, table_half=self._halves.get(fc.eid))

    def _settle_label(self, fc: Forecast, ent: dict) -> None:
        """v11: the label of a threshold-defined classification from the distribution the answer submits. The model's
        own probabilities are blended with those the label rules imply on the submitted range (a half normal on each
        side of the point, so the point stays the median), then with the revision base rate; and when the submitted
        range lies wholly inside one label's region (that label holds with probability >= 0.95), it is the label."""
        labels = self.spec.labels
        probs = dict(fc.model_probs or fc.probs or {})
        sources = self._rule_sources(fc.eid) if fc.median is not None else []
        rng = self._submitted_range(fc, ent) if sources else None
        implied = None
        for rules in sources if rng else []:
            # v14: a source whose rules fail on the submitted distribution (regions that overlap or do not partition,
            # a threshold on another scale) gives way to the next one instead of leaving the label to the model's odds
            implied = implied_probs(rules, labels, ent, rng[0], rng[1], rng[2])
            if implied:
                break
        if implied:
            w = config.IMPLIED_LABEL_WEIGHT if probs else 1.0
            probs = {lab: (1 - w) * probs.get(lab, 0.0) + w * implied[lab] for lab in labels}
        if not probs:
            return
        fc.probs = {lab: probs.get(lab, 0.0) for lab in labels}
        self._blend_prior(fc)
        self._blend_eps_direction(fc)
        fc.label = max(labels, key=lambda lab: (fc.probs[lab], -labels.index(lab)))
        sure = [lab for lab in labels if implied and implied[lab] >= 0.95 - 1e-9]
        if sure and sure[0] != fc.label:
            log.info("label %s: %s -> %s (the submitted range lies wholly in its region)", fc.eid, fc.label, sure[0])
            fc.label = sure[0]

    def _aggregate_one(self, ent: dict, items: list[dict]) -> Forecast | None:
        eid = ent["entity_id"]
        fc = Forecast(eid=eid, source="llm", n_samples=len(items))
        if self.spec.target_type == "classification":
            acc: dict[str, float] = defaultdict(float)
            n_ok = 0
            for it in items:
                probs = it.get("probs")
                if not isinstance(probs, dict):
                    lab = str(it.get("label") or "").strip().lower()
                    probs = {l: 1.0 for l in self.spec.labels if l.lower() == lab}
                norm = {str(k).strip().lower(): v for k, v in probs.items()}
                vals = {lab: max(0.0, _f(norm.get(lab.lower())) or 0.0) for lab in self.spec.labels}
                tot = sum(vals.values())
                if tot <= 0:
                    continue
                for lab, v in vals.items():
                    acc[lab] += v / tot
                n_ok += 1
            if n_ok == 0:
                return None
            fc.probs = {lab: acc[lab] / n_ok for lab in self.spec.labels}
            fc.model_probs = dict(fc.probs)
            qs = [it.get("quantity") for it in items if isinstance(it.get("quantity"), dict)]
            meds = [_f(q.get("median")) for q in qs]
            meds = [m for m in meds if m is not None]
            if meds:
                fc.median = statistics.median(meds)
                lows = [v for v in (_f(q.get("p05")) for q in qs) if v is not None]
                highs = [v for v in (_f(q.get("p95")) for q in qs) if v is not None]
                fc.p05 = statistics.median(lows) if lows else None
                fc.p95 = statistics.median(highs) if highs else None
            anchor = self.anchor.get(eid)
            if anchor is not None:
                self._blend_anchor(fc, anchor)
            pr = self.prior.get(eid)
            if pr is not None and fc.stat_q90 is None:  # the revision scale sizes the label range as well
                fc.stat_q90, fc.stat_reliable = pr.q90, reliable_scale(pr)
            self._center_on_prior(fc)  # before the label is read off the quantity, so the two agree
            self._settle_label(fc, ent)
        else:
            rule = self.level_rule()
            conv = (lambda x: self._to_target(eid, x, rule)) if rule else (lambda x: x)
            meds = [v for v in (conv(_f(it.get("median"))) for it in items) if v is not None]
            if not meds:
                return None
            fc.median = statistics.median(meds)
            lows = [v for v in (conv(_f(it.get("p05"))) for it in items) if v is not None]
            highs = [v for v in (conv(_f(it.get("p95"))) for it in items) if v is not None]
            fc.p05 = statistics.median(lows) if lows else None
            fc.p95 = statistics.median(highs) if highs else None
        fc.support, fc.why = self._support_ids(eid, items)
        rats = [it.get("rationale") for it in items if isinstance(it.get("rationale"), str)]
        fc.rationale = rats[0][:300] if rats else ""
        return fc
