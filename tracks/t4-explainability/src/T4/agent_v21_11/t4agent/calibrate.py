"""Turn aggregated forecasts into the submitted numbers: label, point, and a calibrated interval.

Scorer 5.2.x scores an interval with the interval score: its width plus 20 times the distance by which
the outcome falls outside it, against the unit's naive interval. Width costs as much as a miss does in
expectation at the true 5th/95th percentiles, so the aim is a calibrated 90% range, not maximal coverage
(the v3-v6 coverage-first widening is gone). The half-width comes from the entity's own error scale when
there is one: measured on its history (a backtest over enough origins, a revision envelope) it stands
alone; estimated (a random-walk extrapolation, the EPS anchor's rule) a wider model range moves it to their
geometric mean. Without one, the model's range is widened (its ranges were measured too narrow) and floored
by the planner's deviation. An interval sits on a numeric quantity whenever one exists -- [0, 1] only for
probabilities or when nothing numeric is known.
"""
from __future__ import annotations

import dataclasses
import math
import re

from .artifact_policy import ArtifactPolicy
from .config import REL_FLOOR, WIDEN, Z90
from .corpus import Corpus
from .predict import (_CHANGEY, Forecast, _f, _field_tokens, baseline_value, event_return_sd, quantity_baseline,
                      unit_deviation)
from .stats import find_series, relative_quantity, series_half_width
from .task import TaskSpec



def _decimals(width: float, plan_dec) -> int:
    try:
        d = int(plan_dec)
        if 0 <= d <= 6:
            return max(d, 0 if width >= 10 else 1 if width >= 1 else 2 if width >= 0.1 else 3)
    except (TypeError, ValueError):
        pass
    if width <= 0 or not math.isfinite(width):
        return 2
    return max(0, min(4, 1 - math.floor(math.log10(width))))


def _round(x: float, d: int) -> float:
    v = round(x, d)
    return 0.0 if v == 0 else v


# A unit or type string that is itself a probability (optionally with a 0-1 domain). The bare word inside a
# longer question or title is not this: "probability the ratio exceeds 2.5" is not a probability unit.
_PROB_EXACT = re.compile(
    r"^(?:a[\s_-]+)?(?:probabilit(?:y|ies)|probs?|chances?|likelihoods?|binary|bernoulli|indicators?|"
    r"label[\s_-]*codes?)"
    r"(?:[\s_-]*(?:in[\s_-]*)?\[?\s*0(?:\.0+)?\s*(?:to|[-–,])\s*1(?:\.0+)?\s*\]?)?$",
    re.I,
)
_PROB_TYPE = re.compile(r"\b(?:probability|binary|bernoulli|label[\s_-]*code)\b", re.I)
# A measured unit. Used only when no probability signal has already won, so "percentage points" and
# "index points" stay numeric without hiding a real 0-1 schema.
_NUMERIC_UNIT = re.compile(
    r"\b(?:percent(?:age)?(?:[\s_-]*points?)?|pct|bps|bp|basis[\s_-]*points?|dollars?|usd|eur|gbp|"
    r"index|shares?|ratios?|yields?|prices?|returns?|growth|revisions?|levels?|counts?|"
    r"thousands?|millions?|billions?|points?|changes?)\b",
    re.I,
)
# G1 F3: extra percent spellings used only to reject a planner's probability
# flag. Keeping them off `_text_numeric` preserves the planner-normal
# classification path byte for byte.
_PCT_FORMS = re.compile(r"%|％|(?<![a-z])pp(?![a-z])|(?<![a-z])per(?:\s+|-)+cent(?![a-z])", re.I)


def _norm_schema_text(text: str) -> str:
    return re.sub(r"[\s_]+", " ", text.strip().replace("_", " "))


def _bounds_01(lo, hi) -> bool:
    try:
        return float(lo) == 0.0 and float(hi) == 1.0
    except (TypeError, ValueError):
        return False


def _enum_01(enum) -> bool:
    if not isinstance(enum, (list, tuple)) or not enum:
        return False
    allowed = {0, 1, 0.0, 1.0, "0", "1", False, True}
    return all(v in allowed for v in enum)


def _schema_nodes(spec: TaskSpec) -> list:
    raw = spec.raw if isinstance(getattr(spec, "raw", None), dict) else {}
    target = raw.get("target") if isinstance(raw.get("target"), dict) else {}
    nodes = []
    for src, keys in (
        (target, ("value", "value_type", "value_field", "unit", "units", "quantity", "dtype", "domain",
                  "answer_schema", "schema")),
        (raw, ("answer_schema", "schema")),
    ):
        for key in keys:
            if key in src:
                nodes.append(src[key])
    return nodes


def _shared_unit(spec: TaskSpec) -> str | None:
    vals = []
    for ent in getattr(spec, "entities", None) or []:
        if not isinstance(ent, dict):
            continue
        for key in ("unit", "units"):
            val = ent.get(key)
            if isinstance(val, str) and val.strip():
                vals.append(val.strip())
    if not vals or len({v.lower() for v in vals}) != 1:
        return None
    return vals[0]


def _walk_schema(node):
    """This answer-schema node and the value objects nested inside it."""
    yield node
    if not isinstance(node, dict):
        return
    for key in ("value", "items", "answer_schema", "schema"):
        if key in node:
            yield from _walk_schema(node[key])
    props = node.get("properties")
    if isinstance(props, dict):
        for key in ("value", "point_forecast", "outcome"):
            if key in props:
                yield from _walk_schema(props[key])


def _text_probability(text: str) -> bool:
    return bool(_PROB_EXACT.match(_norm_schema_text(text)))


def _text_numeric(text: str) -> bool:
    return bool(_NUMERIC_UNIT.search(_norm_schema_text(text)))


def _text_measured(text: str) -> bool:
    text = _norm_schema_text(text)
    return bool(_NUMERIC_UNIT.search(text) or _PCT_FORMS.search(text))


def _node_probability(node) -> bool:
    """True for a probability type, an exact probability unit, a 0/1 enum, or numeric bounds of 0 and 1.

    Bounds are read on their own, so {"type": "number", "minimum": 0, "maximum": 1} counts. A title that
    only contains the word does not, and a sibling measured unit does not cancel any of these.
    """
    if isinstance(node, str):
        return _text_probability(node)
    if not isinstance(node, dict):
        return False
    for key in ("type", "format", "kind", "dtype"):
        val = node.get(key)
        if isinstance(val, str) and (_text_probability(val) or _PROB_TYPE.search(_norm_schema_text(val))):
            return True
    for key in ("unit", "units"):
        val = node.get(key)
        if isinstance(val, str) and _text_probability(val):
            return True
    if _enum_01(node.get("enum", node.get("values"))):
        return True
    # Reachable for a numeric type: a probability label is not required before the 0-1 bounds count.
    return _bounds_01(node.get("minimum", node.get("min")), node.get("maximum", node.get("max")))


def _node_numeric(node) -> bool:
    if isinstance(node, str):
        return _text_numeric(node)
    if not isinstance(node, dict):
        return False
    parts = [node.get(k) for k in ("type", "format", "kind", "dtype", "unit", "units", "name", "title")]
    return _text_numeric(" ".join(p for p in parts if isinstance(p, str)))


def _schema_probability(spec: TaskSpec) -> bool:
    for node in _schema_nodes(spec):
        if any(_node_probability(sub) for sub in _walk_schema(node)):
            return True
    unit = _shared_unit(spec)
    return bool(unit and _text_probability(unit))


def _schema_numeric(spec: TaskSpec) -> bool:
    """A measured unit (ratio, bps, percent, level, yield, ...) on the schema or on every row."""
    for node in _schema_nodes(spec):
        if any(_node_numeric(sub) for sub in _walk_schema(node)):
            return True
    unit = _shared_unit(spec)
    return bool(unit and _text_numeric(unit))


def _schema_measured(spec: TaskSpec) -> bool:
    """G1 F3/F4: measured schema/row units on the flag-true path.

    Percent spellings count, and non-empty row unit strings may differ so long
    as every one is measured (for example, ``bps`` and ``basis points``).
    """
    if _schema_numeric(spec):
        return True
    for node in _schema_nodes(spec):
        for sub in _walk_schema(node):
            if isinstance(sub, str) and _text_measured(sub):
                return True
            if isinstance(sub, dict):
                parts = [sub.get(k) for k in ("type", "format", "kind", "dtype", "unit", "units", "name", "title")]
                if _text_measured(" ".join(p for p in parts if isinstance(p, str))):
                    return True
    units = []
    for ent in getattr(spec, "entities", None) or []:
        if not isinstance(ent, dict):
            continue
        for key in ("unit", "units"):
            val = ent.get(key)
            if isinstance(val, str) and val.strip():
                units.append(val)
    return bool(units) and all(_text_measured(unit) for unit in units)


def _binary_value_column(spec: TaskSpec) -> bool:
    """A column literally named value whose every entry is 0 or 1 is a label code.

    A latest/consensus/value level reading is not consulted here, so it cannot hide this column or a
    planner flag. A measured unit is decided before this and still keeps a scored ratio numeric.
    """
    entities = [e for e in (getattr(spec, "entities", None) or []) if isinstance(e, dict)]
    if not entities:
        return False
    vals: list[float] = []
    for ent in entities:
        if "value" not in ent:
            return False
        val = _f(ent.get("value"))
        if val is None:
            return False
        vals.append(val)
    return all(v in (0.0, 1.0) for v in vals)


def is_probability(plan: dict | None, spec: TaskSpec) -> bool:
    """Whether the submitted interval is a probability in [0, 1].

    Planner `interval_is_probability: true`, a probability type or unit, a 0/1 enum, numeric bounds of
    0 and 1, and `interval_leg` false all win, including over a latest, consensus, or value column.
    The bare word "probability" in the question is not a signal. With none of those signals, a measured
    unit (ratio, bps, pct, percent, level, yield, ...) stays off [0, 1].

    v21.10: only the planner flag applies to a regression or ranking unit, as in v21.8.
    v21.11 (T4-002): not the flag alone. It clamped a bid-to-cover of 2.5 to 1.0 in [0, 1] and tied ranking scores at 0
    or 1. A ranking is never [0, 1]; a regression only when the task names no measured unit (`_planner_probability`),
    and then row by row in `finalize` (a forecast outside [0, 1] is no probability).
    """
    if getattr(spec, "target_type", None) != "classification":
        return _planner_probability(plan, spec)
    if plan and plan.get("interval_is_probability") is True:
        return True
    # v21.10: the signals below decide classification units only. A label-code truth (`interval_leg = false`) is a
    # classification declaration; a regression or ranking unit is scored on its numeric value, and a schema hint or
    # a row column that happens to look binary (a `value` column of zeros, say) would send a basis-point forecast
    # out as 0 in [0, 1] there.
    if getattr(spec, "target_type", None) != "classification":
        return False
    if _schema_probability(spec):
        return True
    card = getattr(spec, "card_params", None) or {}
    if card.get("interval_leg") is False:
        return True
    if _schema_numeric(spec):
        return False
    return _binary_value_column(spec)


_PROB_WORDS = re.compile(r"probabilit|likelihood|chance|\b0\s*(?:-|to|–)\s*1\b", re.I)
_PROB_TOKENS = frozenset({
    "prob", "probs", "probability", "probabilities", "likelihood", "likelihoods", "chance", "chances", "odds",
})
_GLUED_PROB_COLUMN = re.compile(
    r"(?:(?:implied|consensus|estimated|predicted|forecast|model)(?:prob|probability)(?:value|score|level|pct|percent)?"
    r"|(?:prob|probability)(?:value|score|level|pct|percent))",
    re.I,
)


def _prob_column(name) -> bool:
    """G1 F5: a probability column by complete token or known glued form."""
    tokens = _field_tokens(name)
    if any(token in _PROB_TOKENS for token in tokens):
        return True
    return bool(_GLUED_PROB_COLUMN.fullmatch("".join(tokens)))


def _planner_probability(plan: dict | None, spec: TaskSpec) -> bool:
    """v21.11 (T4-002): a regression unit takes the planner's `interval_is_probability: true` only when neither the
    target's name nor its schema or shared row unit names a measured unit (bps, percent, ratio, level, ...); a ranking
    unit never does (its points are the ordering)."""
    if getattr(spec, "target_type", None) != "regression" or not plan or plan.get("interval_is_probability") is not True:
        return False
    name = str(getattr(spec, "target_name", "") or "").replace("_", " ")
    if _text_measured(name) or _schema_measured(spec):
        return False
    # T4-002/C1: keep the pre-analyst guard from c83a768, but make its
    # unit-level decision unanimous. A mixed unit stays on the probability
    # prompt and is resolved row by row in finalize; an unnamed regression
    # whose every row has a non-calendar naive level outside [0, 1] does not.
    seen = False
    for ent in getattr(spec, "entities", None) or []:
        if not isinstance(ent, dict):
            return True
        base = baseline_value(spec, ent, plan, skip_date_period=True)
        if base is None or not math.isfinite(base):
            continue
        if 0.0 <= base <= 1.0:
            return True
        seen = True
    if seen:
        return False
    return True


def numeric_target_plan(plan: dict | None, spec: TaskSpec) -> bool:
    """v21.11 (T4-002): in place, drop a planner's probability reading of a ranking target, or of a regression target
    the task itself states in measured units -- the flag, and a 'probability 0-1' interval quantity that would ask the
    analysts for probabilities instead of the target. True when something was dropped."""
    if not isinstance(plan, dict) or plan.get("interval_is_probability") is not True:
        return False
    if getattr(spec, "target_type", None) not in ("regression", "ranking") or _planner_probability(plan, spec):
        return False
    plan["interval_is_probability"] = False
    if isinstance(plan.get("interval_quantity"), str) and _PROB_WORDS.search(plan["interval_quantity"]):
        plan.pop("interval_quantity", None)
    return True


def _planner_only(plan: dict | None, spec: TaskSpec) -> bool:
    """v21.11 (T4-002): whether a unit's [0, 1] decision rests on the planner's flag alone (no schema, card or row
    signal); `finalize` then checks it row by row."""
    if not plan or plan.get("interval_is_probability") is not True:
        return False
    tt = getattr(spec, "target_type", None)
    if tt == "regression":
        return _planner_probability(plan, spec)
    if tt == "classification":
        card = getattr(spec, "card_params", None) or {}
        return card.get("interval_leg") is not False and not is_probability(None, spec)
    return False


def _numeric_view(spec: TaskSpec, fc: Forecast, plan: dict | None, ent: dict | None) -> Forecast | None:
    """v21.11 (T4-002): for a classification row whose [0, 1] rests on the planner alone, the forecast to size a numeric
    interval with when the row's own quantity (consensus, latest estimate ...) lies outside [0, 1] -- that quantity is
    no probability and no 0/1 code, and the scored truth is on its scale (README: an interval on a probability never
    covers a dollar y). A model quantity on the probability scale is set aside; None when the row stays [0, 1]."""
    if ent is None or any(_prob_column(k) for k, v in ent.items()
                          if isinstance(v, (int, float)) and not isinstance(v, bool)):
        return None  # the row's own table defines a probability quantity: the planner's reading stands
    base = quantity_baseline(spec, ent, plan, skip_date_period=True)
    if event_return_sd(spec) is not None:
        base = None  # an announcement return (in percent, centred on 0) is no probability whatever its sign
    elif base is None or not math.isfinite(base) or 0.0 <= base <= 1.0:
        return None
    if fc.median is not None and not 0.0 <= fc.median <= 1.0:
        return fc
    return dataclasses.replace(fc, median=base, p05=None, p95=None)


def _row_numeric_baseline(spec: TaskSpec, ent: dict | None, plan: dict | None) -> float | None:
    """This row's non-calendar baseline when it proves the quantity is not a probability.

    On a true planner flag, ``baseline_value`` has already skipped tokenised
    date/period fields whose values have calendar shapes.
    """
    if not isinstance(ent, dict):
        return None
    base = baseline_value(spec, ent, plan, skip_date_period=True)
    if base is None or not math.isfinite(base) or 0.0 <= base <= 1.0:
        return None
    return base


def _quantity_text(spec: TaskSpec, plan: dict | None) -> str:
    return (spec.target_name.replace("_", " ") + " " + str((plan or {}).get("interval_quantity") or "")).lower()


def _default_half(qtext: str, point: float) -> float:
    """Scale when nothing but the quantity's name is known (no history, no model range, no plan)."""
    if re.search(r"\bbps\b|basis point", qtext):
        return 60.0
    if re.search(r"pct|percent|%|growth|return|change|revision", qtext):
        return max(20.0, 0.5 * abs(point))  # only when nothing else is known: err wide
    return max(1.0, 0.5 * abs(point))


def table_halves(spec: TaskSpec, corpus: Corpus, ev) -> dict[str, float | None]:
    """Each entity's step-volatility scale from its own corpus table (or None): one input of the interval, shared
    by the submitted interval and the label probabilities read off it."""
    out = {}
    policy = ArtifactPolicy(getattr(spec, "cutoff", None))
    for ent in spec.entities:
        eid = ent["entity_id"]
        own = list(ev.specific.get(eid) or [])
        try:
            # v12: the statistical baseline's own series, so both read one column with the same unit guards
            # v21.1: also an unnamed date-indexed table of the entity's own (regression, ranking, or a classification
            # whose row quantity is a relative change)
            allow = spec.target_type != "classification" or relative_quantity(spec.target_name, ent)
            series = find_series(corpus, own + list(ev.shared), ent, spec.target_name, set(own), allow)
            out[eid] = series_half_width(series, spec.target_name, spec.horizon_days, ent, policy=policy) if series else None
        except Exception:  # noqa: BLE001 - a scale hint must never break the answer
            out[eid] = None
    return out


def quantity_range(spec: TaskSpec, fc: Forecast, plan: dict | None, *, ent: dict | None = None,
                   table_half: float | None = None, numeric: bool = False) -> tuple[float, float, float] | None:
    """(point, lo, hi) of the numeric quantity, before rounding, exactly as `finalize` submits it; None when the
    row has no numeric quantity or the quantity is a probability (its interval is [0, 1]). v11: the label is read
    off this same range, so label and interval describe one distribution (v10 read labels off the model's range
    widened 2x, while the interval used the entity's own scale: IBM could be submitted at 2.0 against last year's
    1.53, interval [1.0, 2.7], labelled 'down')."""
    if not numeric and is_probability(plan, spec):  # v21.11: `numeric` -- finalize has decided the row is numeric
        return None
    dev = unit_deviation(spec, plan)
    qtext = _quantity_text(spec, plan)
    if ent is not None:  # v21.1: the row's own unit text ('bps_change ...', 'percent change of ...')
        qtext += f" {ent.get('unit') or ''} {ent.get('units') or ''}".lower().replace("_", " ")
    ev = event_return_sd(spec)
    if spec.target_type == "classification":
        point = fc.median
        if point is None and ent is not None:
            skip_date_period = bool(plan and plan.get("interval_is_probability") is True)
            point = quantity_baseline(spec, ent, plan, skip_date_period=skip_date_period)
        if point is None and _CHANGEY.search(qtext):
            point = 0.0
        if point is None and ev is not None:
            point = 0.0  # v21.5: an announcement-window return; v21.3 sent [0, 1] in its full fallback
        if point is None:  # nothing numeric known about this row
            return None
    else:
        point = fc.median if fc.median is not None else 0.0

    half_lo = max(0.0, point - fc.p05) if fc.p05 is not None else None
    half_hi = max(0.0, fc.p95 - point) if fc.p95 is not None else None
    if half_lo and not half_hi:  # one-sided model range: mirror the known side
        half_hi = half_lo
    if half_hi and not half_lo:
        half_lo = half_hi
    # The entity's own scale beats the planner's unit-wide deviation, which is one number for the whole
    # cross-section and would give a stable member (core CPI) a volatile member's (gasoline) width. Two own
    # estimates may exist -- the backtested error of the statistical baseline and the series' step volatility --
    # and they are combined geometrically: the larger one overstates a smooth series, the smaller one a
    # trending one, and under the interval score neither erring wide nor erring narrow is free.
    scales = [x for x in (fc.stat_q90, table_half) if x and x > 0 and math.isfinite(x)]
    own = math.sqrt(scales[0] * scales[1]) if len(scales) == 2 else (scales[0] if scales else 0.0)

    def side(model_half: float | None) -> float:
        widened = WIDEN * model_half if model_half else 0.0
        if own > 0:
            # A scale measured on the entity's own history (a backtest over enough origins, a revision envelope)
            # stands: the model's ranges are off by large factors in both directions (+/-4bp on a seven-week rate
            # move; ten times the revision scale on revised payrolls), so they carry little information next to
            # a measurement. An extrapolated or rule-based own scale is itself an estimate: a wider model range
            # moves it to their geometric mean, the midpoint on the log scale on which the interval score's
            # cost of a wrong width is roughly symmetric.
            raw = model_half or 0.0
            if fc.stat_reliable or raw <= own:
                return own
            return math.sqrt(own * raw)
        if widened > 0:
            half = max(widened, Z90 * dev) if dev else widened
        elif dev:
            half = Z90 * dev
        else:
            half = max(_default_half(qtext, point), REL_FLOOR * abs(point))
        return max(half, Z90 * ev) if ev else half  # v10: an announcement-window return is never narrower

    lo_half, hi_half = side(half_lo), side(half_hi)
    lo, hi = point - lo_half, point + hi_half
    # The naive baseline (latest value / zero change / consensus): when the forecast has moved away from it and it
    # falls outside the range, the range reaches halfway towards it -- a hedge on the forecast's departure that
    # costs half the gap in width instead of twenty times the distance if the naive value turns out right.
    base = None
    if ent is not None:
        skip_date_period = bool(plan and plan.get("interval_is_probability") is True)
        base = (quantity_baseline(spec, ent, plan, skip_date_period=skip_date_period)
                if spec.target_type == "classification"
                else baseline_value(spec, ent, plan, skip_date_period=skip_date_period))
    if base is not None and math.isfinite(base):
        if base < lo:
            lo = 0.5 * (lo + base)
        elif base > hi:
            hi = 0.5 * (hi + base)
    return point, lo, hi


def finalize(spec: TaskSpec, fc: Forecast, plan: dict | None, *, ent: dict | None = None,
             table_half: float | None = None) -> dict:
    """-> {label?, point_forecast?, lo, hi}, all finite, lo <= hi."""
    out: dict = {}
    prob = is_probability(plan, spec)
    view = fc
    if prob and _planner_only(plan, spec):
        # v21.11 (T4-002): a [0, 1] resting on the planner alone holds only for a row that can be a probability
        if spec.target_type == "classification":
            numeric_fc = _numeric_view(spec, fc, plan, ent)
            if numeric_fc is not None:
                prob, view = False, numeric_fc
        elif fc.median is not None and not 0.0 <= fc.median <= 1.0:
            prob = False
        else:
            base = _row_numeric_baseline(spec, ent, plan)
            if base is not None:
                # C2: the analysts answered on a probability scale for this
                # mixed unit. Set that model quantity aside and centre this
                # numeric row on its own naive level.
                prob = False
                view = dataclasses.replace(fc, median=base, p05=None, p95=None)
    if spec.target_type == "classification":
        out["label"] = fc.label if fc.label in spec.labels else (spec.labels[0] if spec.labels else "unknown")
        if prob:
            # A probability (or a 0/1 event indicator): [0, 1] always covers. The optional point is the
            # prompt's own quantity when the model gave it; guessing which label it refers to would misstate it.
            if fc.median is not None and 0.0 <= fc.median <= 1.0:
                out["point_forecast"] = _round(fc.median, 2)
            out["lo"], out["hi"] = 0.0, 1.0
            return out
    rng = None if prob else quantity_range(spec, view, plan, ent=ent, table_half=table_half, numeric=True)
    if rng is None:
        if spec.target_type == "classification":  # nothing numeric known about this row
            out["lo"], out["hi"] = 0.0, 1.0
            return out
        point = min(1.0, max(0.0, fc.median if fc.median is not None else 0.0))  # a probability target
        lo, hi = 0.0, 1.0
    else:
        point, lo, hi = rng
    d = _decimals(hi - lo, (plan or {}).get("decimals"))
    if spec.target_type == "ranking":
        d = min(6, d + 2)  # the scorer ranks by the point: rounding must not manufacture ties
    scale = 10 ** d
    # bounds round OUTWARD so rounding can never shrink the interval past the forecast
    point = _round(point, d)
    lo = _round(math.floor(lo * scale + 1e-9) / scale, d)
    hi = _round(math.ceil(hi * scale - 1e-9) / scale, d)
    lo, hi = min(lo, point), max(hi, point)
    if hi <= lo:
        step = 10 ** (-d)
        lo, hi = _round(lo - step, d), _round(hi + step, d)
    out["point_forecast"] = point
    out["lo"], out["hi"] = lo, hi
    return out
