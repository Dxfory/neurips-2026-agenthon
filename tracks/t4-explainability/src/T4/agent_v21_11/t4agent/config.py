"""Runtime knobs. Defaults are the production settings; env vars exist for local testing only."""
from __future__ import annotations

import math
import os


def _num(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


# Wall clock. The unit ceiling is 600 s and includes container creation and image pull, so the
# agent finishes well inside it. LLM work stops at LLM_DEADLINE; the answer is final by TOTAL_BUDGET.
TOTAL_BUDGET_SEC = _num("T4_TIME_BUDGET", 470.0)
LLM_DEADLINE_SEC = _num("T4_LLM_DEADLINE", 400.0)
# H5: production caps are lower-only and disabled by default. The existing
# T4_TIME_BUDGET / T4_LLM_DEADLINE variables remain local-testing overrides.
CAP_TIME_BUDGET_SEC = _num("T4_CAP_TIME_BUDGET", 0.0)
CAP_LLM_DEADLINE_SEC = _num("T4_CAP_LLM_DEADLINE", 0.0)
STAGE_GRACE_SEC = 20.0  # v21.13: the main thread waits this long past the LLM deadline for the model stage
UNIT_TIMEOUT_SEC = 600.0  # the Development unit ceiling, used when the unit card states none
CLOCK_REF_SEC = 400.0  # the LLM deadline every absolute margin of the model path was set for


def derive_clock(unit_timeout: float | None) -> tuple[float, float, float]:
    """v21.5: (hard stop, LLM deadline, margin scale) from the unit's own time limit, the card's [agent].timeout_sec
    (600 s when absent). The hard stop keeps 90 s for image pull, container start-up and slack; the LLM deadline keeps
    70 s for the final CPU work and a call still in flight (v21.13: it is given up at the LLM deadline itself; was the deadline + 15 s). Every absolute margin
    of the model path (reasons reserve, analyst stop, re-check and reasons gates) shrinks with the LLM deadline: v21.3
    kept them in seconds, and an LLM deadline under about 185 s left the analysts no window at all. 600 s gives v21.3's
    470 / 400 s and scale 1.0 exactly. A local-testing clock (T4_TIME_BUDGET set) is taken as set."""
    t = UNIT_TIMEOUT_SEC
    if unit_timeout is not None and math.isfinite(unit_timeout) and unit_timeout > 0:
        t = unit_timeout
    if "T4_TIME_BUDGET" in os.environ:
        t = max(t, TOTAL_BUDGET_SEC + 90.0)
    hard = max(30.0, min(TOTAL_BUDGET_SEC, t - 90.0))
    if CAP_TIME_BUDGET_SEC > 0:
        hard = max(30.0, min(hard, CAP_TIME_BUDGET_SEC))
    llm = max(0.0, min(LLM_DEADLINE_SEC, hard - 70.0))
    if CAP_LLM_DEADLINE_SEC > 0:
        llm = max(0.0, min(llm, CAP_LLM_DEADLINE_SEC))
    return hard, llm, min(1.0, llm / CLOCK_REF_SEC)


# House allowance: 25 admitted requests per unit, 4,000 output tokens per request (thinking included).
REQUEST_BUDGET = int(_num("T4_REQUEST_BUDGET", 23))
MAX_TOKENS = 4000
REASONING_BUDGET_PLAN = int(_num("T4_REASONING_PLAN", 1400))
REASONING_BUDGET_ANALYST = int(_num("T4_REASONING_ANALYST", 2800))
ANSWER_TOKENS_PER_ENTITY = 230  # room kept for each entity's JSON under the 4,000-token cap
CONCURRENCY = int(_num("T4_CONCURRENCY", 3))  # House route concurrency is unpublished; stay moderate
MAX_RPM = _num("T4_MAX_RPM", 0.0)  # 0 = no client-side rate limit (local testing sets it)
REQUEST_TIMEOUT_SEC = _num("T4_REQUEST_TIMEOUT", 170.0)
TEMPERATURE = _num("T4_TEMPERATURE", 0.6)
PLAN_TEMPERATURE = 0.3

# Analyst batching and self-consistency.
GROUP_SIZE = int(_num("T4_GROUP_SIZE", 3))  # small groups: shorter replies leave more room to think
# Samples per group: the queue runs every group's first sample before any extra one, and stops launching calls
# well before the analyst deadline, so a higher cap only spends budget and time a small unit would leave unused
# (v9: 8 of 23 requests on a six-entity unit). v10: 9, from 5.
MAX_SAMPLES = int(_num("T4_MAX_SAMPLES", 9))
# v21: in v20's sweep the model moved regression answers that have a code baseline by 0.01-0.25 standard deviations
# (bank EPS 0.01, CPI 0.05, auctions 0.09): a few samples suffice there, and the requests go to the reasons
ANCHORED_SAMPLES = int(_num("T4_ANCHORED_SAMPLES", 3))
# v21: requests kept back for focused single-entity votes on classification labels the samples left undecided
EXTRA_VOTES = int(_num("T4_EXTRA_VOTES", 4))
# One reviewing call over the whole cross-section after the analysts; they stop early enough to leave it time.
# v21: off. In v20's sweep the review call changed 1 answer in 11 units (and failed in another), while an accepted
# second opinion entered at half weight on top of the statistical blend (0.25 model + 0.25 baseline + 0.5 review).
# Units, periods and scales are checked in code (plausible-move guard, outlier re-check); its request goes to the reasons.
REVIEW = int(_num("T4_REVIEW", 0))
REVIEW_RESERVE_SEC = 100.0
REVIEW_REASONING = 1500  # a consistency check needs less thinking
# (group size, thinking budget) from deepest to broadest: the deepest layout that still samples every group at least
# twice in the time left, at the route's measured speed, is used.
LAYOUTS = ((3, 2800), (4, 2400), (6, 2000), (8, 1500))
LATENCY_MARGIN = 1.25  # calls on a shared route vary in duration: plan with a margin over the measured speed
# Equal-weight combination with the backtested baseline: the literature's untuned default, not validated on this agent's
# forecasts. v14: replayed on pre-cutoff history, the baselines are about as good as the naive rule (scorer-5.2.2 quality
# 0.49-0.57) and pick the naive rule itself for 71-81% of auction and positioning forecasts, so there this in effect
# shrinks the model halfway towards the naive rule. v16: tested on the model's own forecasts (tools/pseudo_auctions.py:
# 12 pseudo-units of 2016-2019 Treasury auctions, 72 forecasts, blend off): the model alone scored 0.505 against the
# last-auction naive rule and the baseline alone 0.515, so more baseline weight helped (0.75: +0.002, 7 of 12 units).
# Below the pre-registered bar for a change (+0.01 and two thirds of the units): 0.5 stays.
STAT_WEIGHT = _num("T4_STAT_WEIGHT", 0.5)

# Uncertainty. The model's own 90% ranges were measured far too narrow: they are widened before an interval is
# built and before label probabilities are read off them, so label and interval describe one distribution.
WIDEN = 2.0
Z90 = 1.645
REL_FLOOR = 0.03  # relative half-width used only when nothing else is known about a quantity's scale
# v10: one standard deviation (percentage points) of a large U.S. stock's return over a short window around a
# scheduled company announcement (earnings), from the event-study and announcement-volatility literature (roughly
# 4-6 points; the low end). A floor for such a quantity's scale when the entity has no measured one.
EVENT_RETURN_SD = 4.0
# Threshold-defined labels: weight of the probabilities implied by that widened distribution. The rest
# stays on the model's own label probabilities, a hedge against a misread rule.
IMPLIED_LABEL_WEIGHT = 0.75

# Evidence packing (characters).
SMALL_CORPUS_CHARS = 36000  # below this every document goes to the analyst in full
PLAN_DIGEST_CHARS = 40000  # what the planner reads: a small corpus in full, else computed facts and key passages
CHUNK_MAX = 420
EVIDENCE_CHUNKS_PER_ENTITY = 22
KEYLINE_CHUNKS_PER_ENTITY = 10
PROMPT_EVIDENCE_CHARS = 90000  # hard cap on snippet text in one analyst prompt

# Claims (scorer 5.2.x). Each claim is a verbatim quote of the span it cites: the figure check passes whole and
# the judge is never asked. A claim must stay under 400 judge tokens; 600 characters of filing or table text is
# well inside that.
MAX_CLAIMS_PER_ENTITY = 4
CLAIM_MAX_CHARS = 600
SMALL_DOC_HEAD_MAX = 40000

# Reasons (submitted_reasons, graded in the Final only): one call after the analysts, with a code-built fallback.
REASONS = int(_num("T4_REASONS", 1))
REASONS_ROUNDS = 2  # v21 (1): follow-up rounds for held-back, missing or ungrounded reasons
# v10 (60 s): three reasons over up to eight drivers are about twice as long; v13 (75): room for one follow-up call that
# replaces reasons held back or dropped; v20 (90); v21 (120): up to REASONS_ROUNDS follow-up rounds
REASONS_RESERVE_SEC = 120.0
REASONS_FOLLOWUP_SEC = 45.0  # v13: the follow-up starts only this long before the LLM deadline or earlier
REASONS_REASONING = 1100  # v10 (900): the call now plans which drivers each reason covers

# v21.5: a constant (v21.3 read QFBENCH_SEED, which the organizers' rerun sets afresh): the derived per-call seeds
# (SEED * 1000 + sample * 17 + group size, SEED + 7 ...) stay the magnitudes every validation run used
SEED = 0
