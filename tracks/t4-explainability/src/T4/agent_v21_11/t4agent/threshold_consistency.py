"""Threshold parsing and output consistency, split out of the C freeze layer.

This module is the piece that can run on a v21.13 row without replacing the forecast.
It parses the task's own threshold with Decimal, then checks two gates:

- the point belongs to the submitted label's region, for a threshold classification
- the interval contains the point

When the label and the point disagree, the label is kept and the point moves onto the
publication grid inside that label, but only when the submitted point and the
threshold are the same quantity. A probability or a growth rate is not moved onto an
EPS level. When the submitted interval already meets the label region on a legal grid
point, the point stays inside both and the interval is left alone. The interval is
widened only when it does not already contain the point.
Regression rows, ranking rows, and the 360/290 clock are not rewritten. Family
policies from the freeze layer (bank growth blend, vintage rebasing, FOMC sigma,
COT last-week points, conditional medians, roster deviation) are not implemented here.

Rule order follows the v21.13 task-first order. The freeze layer's build_card reads
the planner's label_rules before the entity table; that order is not copied.
"""
from __future__ import annotations

import ast
import math
import operator
import re
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_UP

from .task import TaskSpec

# Publication constants the parser and the gate need. Not a score fit.
PERCENT_SCALE = 100
DECIMAL_CAP = 6
EPS_LEVEL_DECIMALS = 2
GROWTH_PERCENT_DECIMALS = 2
BPS_DECIMALS = 1

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}
_COND = re.compile(r"^\s*Q\s*(>=|<=|>|<)\s*(.+)$")
_BAND = re.compile(r"^\s*(.+?)\s*(<=|<)\s*Q\s*(<=|<)\s*(.+)$")
_REL_PLUS = re.compile(
    r"([A-Za-z_][A-Za-z0-9_]*)\s*\*\s*\(\s*1\s*\+\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)"
)
_REL_MINUS = re.compile(
    r"([A-Za-z_][A-Za-z0-9_]*)\s*\*\s*\(\s*1\s*-\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)"
)
_REL_DENOM = re.compile(r"abs\(\s*Q\s*-\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)\s*/\s*\1\b")
_THRESHOLD_TEXT = re.compile(
    r"(?P<dollar>\$\s*(?P<d>\d+(?:\.\d+)?))"
    r"|(?P<cents>(?P<c>\d+(?:\.\d+)?)\s*cents?)"
    r"|(?P<bp>(?P<b>\d+(?:\.\d+)?)\s*(?:bps?|basis\s+points?))"
    r"|(?P<pct>(?P<p>\d+(?:\.\d+)?)\s*(?:%|percent|pct))",
    re.I,
)
_OTHERWISE = frozenset({"otherwise", "else"})
_UP_NAME = re.compile(r"^(?:up|higher|increase|rise|above|positive|beat|gain)$", re.I)
_DOWN_NAME = re.compile(r"^(?:down|lower|decrease|fall|below|negative|miss|decline)$", re.I)
_BASE_FIELD = re.compile(
    r"prior_year_q_eps|latest_precutoff_estimate|consensus_eps|consensus|latest_estimate",
    re.I,
)
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_BPS_WORD = re.compile(r"\bbps?\b|basis\s+points?", re.I)
_DILUTED_LEVEL = re.compile(
    r"diluted\s+EPS|per\s+diluted\s+share|earnings\s+per\s+(?:diluted\s+)?share", re.I
)
_PERCENT_WORD = re.compile(r"%|percent|\bpct\b", re.I)
_LEVEL_WORD = re.compile(r"\blevel\b|index|per share|dollar|(?:usd|us\$)\s*/\s*share", re.I)


class ThresholdConsistencyError(RuntimeError):
    """The point could not be placed in the submitted label."""


def D(x) -> Decimal | None:
    """A Decimal that keeps a short decimal's printed digits."""
    if isinstance(x, Decimal):
        return x
    if isinstance(x, bool) or x is None:
        return None
    if isinstance(x, (int, float)):
        if isinstance(x, float) and not math.isfinite(x):
            return None
        return Decimal(str(x))
    if isinstance(x, str):
        try:
            return Decimal(x.strip())
        except Exception:
            return None
    return None


def _dec_eval(expr: str, names: dict[str, Decimal]) -> Decimal | None:
    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError:
        return None

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return D(node.value)
        if isinstance(node, ast.Name):
            if node.id not in names:
                raise ValueError(node.id)
            return names[node.id]
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -ev(node.operand)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "abs" and len(node.args) == 1:
            return abs(ev(node.args[0]))
        raise ValueError("unsupported")

    try:
        return ev(tree)
    except (ValueError, ZeroDivisionError, TypeError, ArithmeticError):
        return None


def rewrite_relative(expr: str) -> str:
    """``ref * (1 ± p)`` uses |ref|, so a negative consensus does not flip the band."""
    expr = _REL_PLUS.sub(r"(\1 + abs(\1) * \2)", expr)
    expr = _REL_MINUS.sub(r"(\1 - abs(\1) * \2)", expr)
    expr = _REL_DENOM.sub(r"abs(Q - \1) / abs(\1)", expr)
    return expr


@dataclass(frozen=True)
class ThresholdSpec:
    kind: str
    magnitude: Decimal


def parse_threshold_text(text: str) -> ThresholdSpec | None:
    """Read one threshold out of ``5%``, ``$0.05``, ``5 cents`` or ``5bp``."""
    if not isinstance(text, str):
        return None
    match = _THRESHOLD_TEXT.search(text)
    if not match:
        return None
    if match.group("dollar"):
        return ThresholdSpec("absolute_dollars", D(match.group("d")))
    if match.group("cents"):
        return ThresholdSpec("absolute_dollars", D(match.group("c")) / Decimal(PERCENT_SCALE))
    if match.group("bp"):
        return ThresholdSpec("absolute_bp", D(match.group("b")))
    if match.group("pct"):
        return ThresholdSpec("relative_pct", D(match.group("p")) / Decimal(PERCENT_SCALE))
    return None


@dataclass
class Region:
    lo: Decimal | None
    hi: Decimal | None
    closed_lo: bool = False
    closed_hi: bool = False

    def contains(self, value: Decimal) -> bool:
        if self.lo is not None and (value < self.lo or (value == self.lo and not self.closed_lo)):
            return False
        if self.hi is not None and (value > self.hi or (value == self.hi and not self.closed_hi)):
            return False
        return True

    def pieces(self) -> list[Region]:
        return [self]


@dataclass
class RegionUnion:
    """A label such as 'outside the band' can occupy more than one interval."""

    parts: list[Region]

    def contains(self, value: Decimal) -> bool:
        return any(part.contains(value) for part in self.parts)

    def pieces(self) -> list[Region]:
        return list(self.parts)


RegionLike = Region | RegionUnion


@dataclass
class ThresholdCard:
    labels: list[str]
    rules: dict[str, str]
    fields: dict[str, Decimal]
    kind: str = "absolute"
    reference: Decimal | None = None
    reference_name: str = ""
    decimals: int = 2
    regions: dict[str, RegionLike] = field(default_factory=dict)

    def prepared_rules(self) -> dict[str, str]:
        return {
            lab: rewrite_relative(cond) if cond.strip().lower() not in _OTHERWISE else cond
            for lab, cond in self.rules.items()
        }


def _region_from_condition(cond: str, names: dict[str, Decimal]) -> Region | None:
    cond = rewrite_relative(cond)
    match = _COND.match(cond)
    band = None if match else _BAND.match(cond)
    if match:
        thr = _dec_eval(match.group(2), names)
        if thr is None:
            return None
        op = match.group(1)
        if op == ">":
            return Region(thr, None, False, False)
        if op == ">=":
            return Region(thr, None, True, False)
        if op == "<":
            return Region(None, thr, False, False)
        return Region(None, thr, False, True)
    if band:
        lo, hi = _dec_eval(band.group(1), names), _dec_eval(band.group(4), names)
        if lo is None or hi is None or lo > hi:
            return None
        return Region(lo, hi, band.group(2) == "<=", band.group(3) == "<=")
    return None


def _complement(regions: list[Region]) -> RegionLike | None:
    """Complement the union, preserving open endpoints and any gaps between pieces."""
    merged: list[Region] = []
    for region in sorted(regions, key=lambda r: (r.lo is not None, r.lo or Decimal(0), not r.closed_lo)):
        part = Region(region.lo, region.hi, region.closed_lo, region.closed_hi)
        if not merged:
            merged.append(part)
            continue
        prev = merged[-1]
        touches = prev.hi is None or part.lo is None or part.lo < prev.hi or (
            part.lo == prev.hi and (prev.closed_hi or part.closed_lo)
        )
        if not touches:
            merged.append(part)
        elif prev.hi is not None:
            if part.hi is None or part.hi > prev.hi:
                prev.hi, prev.closed_hi = part.hi, part.closed_hi
            elif part.hi == prev.hi:
                prev.closed_hi = prev.closed_hi or part.closed_hi
    if not merged:
        return None
    gaps: list[Region] = []
    if merged[0].lo is not None:
        gaps.append(Region(None, merged[0].lo, False, not merged[0].closed_lo))
    for left, right in zip(merged, merged[1:]):
        gaps.append(Region(left.hi, right.lo, not left.closed_hi, not right.closed_lo))
    if merged[-1].hi is not None:
        gaps.append(Region(merged[-1].hi, None, not merged[-1].closed_hi, False))
    return gaps[0] if len(gaps) == 1 else RegionUnion(gaps) if gaps else None


def _fill_regions(card: ThresholdCard) -> None:
    rules = card.prepared_rules()
    regions: dict[str, RegionLike] = {}
    otherwise = None
    for lab in card.labels:
        cond = rules.get(lab)
        if not isinstance(cond, str):
            card.regions = {}
            return
        if cond.strip().lower() in _OTHERWISE:
            otherwise = lab
            continue
        region = _region_from_condition(cond, card.fields)
        if region is None:
            card.regions = {}
            return
        regions[lab] = region
    if otherwise is not None:
        gap = _complement(list(regions.values()))
        if gap is None:
            card.regions = {}
            return
        regions[otherwise] = gap
    card.regions = regions if len(regions) == len(card.labels) else {}


def _published_decimals(value: Decimal | None, floor: int = 0) -> int:
    if value is None:
        return floor
    exp = -value.normalize().as_tuple().exponent
    if exp < 0:
        exp = 0
    return min(DECIMAL_CAP, max(floor, exp))


def card_from_rules(labels: list[str], rules: dict[str, str], fields: dict, *,
                    kind: str = "absolute", reference: Decimal | None = None,
                    reference_name: str = "", decimals: int | None = None) -> ThresholdCard | None:
    if not labels or not isinstance(rules, dict):
        return None
    names = {str(k): d for k, v in fields.items() if (d := D(v)) is not None}
    dec = decimals if decimals is not None else _published_decimals(reference, 0)
    card = ThresholdCard(list(labels), dict(rules), names, kind, reference, reference_name, dec)
    _fill_regions(card)
    if len(card.regions) != len(labels):
        return None
    return card


def label_of(value, card: ThresholdCard | None) -> str | None:
    """The single label whose region contains ``value``."""
    if card is None or not card.regions:
        return None
    number = D(value)
    if number is None:
        return None
    hold = [lab for lab in card.labels if card.regions[lab].contains(number)]
    return hold[0] if len(hold) == 1 else None


def _step(decimals: int) -> Decimal:
    return Decimal(10) ** (-int(decimals))


def round_to_decimals(value, decimals: int) -> Decimal:
    return D(value).quantize(_step(decimals), rounding=ROUND_HALF_UP)


def _grid_at(bound: Decimal, decimals: int, *, upward: bool, strict: bool) -> Decimal:
    """The publication-grid value on the legal side of ``bound``.

    ``strict`` excludes the bound itself (``Q > 1.575`` at two decimals is 1.58, not 1.57).
    A closed bound that already sits on the grid is kept (``Q <= 102.3058`` at four decimals
    is 102.3058). The step count is not walked.
    """
    step = _step(decimals)
    quot = (bound / step).to_integral_value(rounding=ROUND_CEILING if upward else ROUND_FLOOR)
    candidate = (quot * step).quantize(step, rounding=ROUND_HALF_UP)
    if upward and (candidate < bound or (strict and candidate <= bound)):
        candidate += step
    elif not upward and (candidate > bound or (strict and candidate >= bound)):
        candidate -= step
    return candidate.quantize(step, rounding=ROUND_HALF_UP)


def nudge_into(value: Decimal, region: Region, decimals: int) -> Decimal | None:
    """Nearest publication-grid value inside ``region``. None when the grid misses it."""
    if region.contains(value):
        return value
    choices: list[Decimal] = []
    if region.lo is not None and (value < region.lo or (value == region.lo and not region.closed_lo)):
        choices.append(_grid_at(region.lo, decimals, upward=True, strict=not region.closed_lo))
    if region.hi is not None and (value > region.hi or (value == region.hi and not region.closed_hi)):
        choices.append(_grid_at(region.hi, decimals, upward=False, strict=not region.closed_hi))
    legal = [item for item in choices if region.contains(item)]
    if not legal:
        return None
    return min(legal, key=lambda item: (abs(item - value), item))


def _table_rules(labels: list[str], ent: dict) -> dict[str, str] | None:
    rules: dict[str, str] = {}
    for lab in labels:
        cond = ent.get(f"{lab}_condition")
        if not isinstance(cond, str):
            continue
        match = re.match(r"^\s*[A-Za-z_][A-Za-z0-9_]*\s*(>=|<=|>|<)\s*(.+)$", cond)
        if match:
            rules[lab] = f"Q {match.group(1)} {match.group(2).strip()}"
    rest = [lab for lab in labels if lab not in rules]
    if not rules or len(rest) > 1:
        return None
    if rest:
        rules[rest[0]] = "otherwise"
    return rules


_NUMBER = r"[-+]?\d+(?:\.\d+)?"
_CLOSED_LOW = r"not more than|no more than|at most|(?:less|lower) than or equal(?: to)?|<=|≤"
_CLOSED_HIGH = r"at least|(?:greater|higher) than or equal(?: to)?|>=|≥"
_CLOSED_PREFIX = re.compile(rf"(?P<op>{_CLOSED_LOW}|{_CLOSED_HIGH})\s*(?P<n>{_NUMBER})", re.I)
_CLOSED_SUFFIX = re.compile(
    rf"(?P<n>{_NUMBER})\s*(?:%|percent|pct|bps?|basis points?)?\s+or\s+"
    r"(?P<op>more|greater|less|fewer)(?!\s+than)\b", re.I,
)
_OFFSET_CUT = re.compile(
    rf"(?P<op>not more than|no more than|at most|at least|more than|greater than|less than)\s+"
    rf"(?P<dollar>\$)?(?P<n>{_NUMBER})\s*"
    r"(?P<unit>cents?|dollars?|usd|%|percent|pct|bps?|basis points?)?\s+"
    r"(?P<side>above|below|higher than|lower than)\s+(?P<ref>.+)", re.I,
)
_BOUND_TEXT = re.compile(
    rf"(?:at least|at most|more than|less than|between|within|[<>≤≥])\s*\$?{_NUMBER}"
    rf"|{_NUMBER}\s+(?:index\s+)?points?\b", re.I,
)


def _has_quantity_bound(text) -> bool:
    """A numeric condition must not disappear into a zero-offset direction fallback.

    Dates in the public macro/EPS direction descriptions are not quantity bounds.
    """
    return isinstance(text, str) and bool(parse_threshold_text(text) or _BOUND_TEXT.search(text))


def _reference_offset(text: str, ent: dict) -> str | None:
    """A supported minimum offset from one unambiguous row reference.

    Cents/dollars on EPS and relative percent offsets keep their magnitude. Other
    reference comparisons abstain; in particular a bp amount is not added to a level
    whose unit is unknown. Neither the planner nor an entity's key order chooses it.
    """
    match = _OFFSET_CUT.search(text)
    if not match or match.group("op").lower() not in {"at least", "more than", "greater than"}:
        return None
    if re.search(r"\b(?:no|not)\s*$", text[:match.start()], re.I):
        return None
    tail = match.group("ref")
    if _has_quantity_bound(tail):
        return None  # an additional bound must not be dropped from a compound assertion
    candidates = [k for k in ent if _BASE_FIELD.search(k) and D(ent[k]) is not None]
    named = [k for k in candidates if re.search(rf"\b{re.escape(k)}\b", tail, re.I)]
    if named:
        candidates = named
    else:
        families = []
        if re.search(r"\bconsensus\b", tail, re.I):
            families.append("consensus")
        if re.search(r"\b(?:prior|previous)\b", tail, re.I):
            families.append("prior")
        if re.search(r"\blatest\b", tail, re.I):
            families.append("latest")
        if len(families) != 1:
            return None
        candidates = [k for k in candidates if families[0] in k.lower()]
    if len(candidates) != 1:
        return None
    ref = candidates[0]
    amount = D(match.group("n"))
    if amount is None or amount < 0:
        return None
    unit = (match.group("unit") or "").lower()
    if unit in {"%", "percent", "pct"} and not match.group("dollar"):
        delta = f"abs({ref}) * {amount / Decimal(PERCENT_SCALE)}"
    elif match.group("dollar") or unit in {"cent", "cents", "dollar", "dollars", "usd"}:
        declared = f"{ref} {ent.get('unit', '')} {ent.get('units', '')} {ent.get('currency', '')}"
        if not re.search(r"eps|dollar|(?:usd|us\$)\s*/\s*share|per share", declared, re.I):
            return None
        delta = str(amount / Decimal(PERCENT_SCALE) if unit in {"cent", "cents"} else amount)
    else:
        return None
    above = match.group("side").lower() in {"above", "higher than"}
    closed = match.group("op").lower() == "at least"
    op = (">=" if closed else ">") if above else ("<=" if closed else "<")
    return f"Q {op} ({ref} {'+' if above else '-'} {delta})"


def _assertion_condition(text, ent: dict) -> str | None:
    from .labels import _FALL, _REFERENCE, _RISE, _assertion_cond

    if not isinstance(text, str):
        return None
    text = text.replace("\u2212", "-").replace("\u2013", "-").replace(",", "")
    relative = _reference_offset(text, ent)
    if relative is not None:
        return relative
    parsed = _assertion_cond(text)
    band = _BAND.match(parsed or "")
    # A symmetric band is a complete condition, even when it contains "or less".
    if band:
        left = "<" if re.search(r"\b(?:greater than|more than|above)\s+" + _NUMBER, text, re.I) else "<="
        right = "<" if re.search(r"\b(?:below|under|but below|and below)\s+" + _NUMBER, text, re.I) else "<="
        return f"{band.group(1)} {left} Q {right} {band.group(4)}"
    if _REFERENCE.search(text):
        return None
    matches = list(_CLOSED_PREFIX.finditer(text)) + list(_CLOSED_SUFFIX.finditer(text))
    if len(matches) != 1:
        return parsed
    match = matches[0]
    if len(re.findall(_NUMBER, text)) != 1 or re.search(r"\b(?:no|not)\s*$", text[:match.start()], re.I):
        return None  # negated/compound conditions need more than one one-sided cut
    op_text, shown = match.group("op").lower(), match.group("n")
    lower = bool(re.fullmatch(_CLOSED_LOW, op_text, re.I)) or op_text in {"less", "fewer"}
    prefix = text[:match.start()]
    fall = bool(re.search(rf"\b{_FALL}\s+(?:by\s+)?$", prefix, re.I))
    rise = bool(re.search(rf"\b{_RISE}\s+(?:by\s+)?$", prefix, re.I))
    if fall or rise:
        amount = abs(D(shown))
        if lower:
            # A fall/rise of at most X stays on its named side of zero.
            return f"{-amount} <= Q <= 0" if fall else f"0 <= Q <= {amount}"
        return f"Q <= {-amount}" if fall else f"Q >= {amount}"
    # "falls TO at most 10" and an explicit "Q <= -10" are absolute cuts.
    return f"Q {'<=' if lower else '>='} {shown.lstrip('+')}"


def _assertion_rules(labels: list[str], assertions, ent: dict) -> dict[str, str] | None:
    if not isinstance(assertions, dict):
        return None
    rules: dict[str, str] = {}
    rest = []
    for lab in labels:
        text = assertions.get(lab)
        cond = _assertion_condition(text, ent)
        if cond is not None:
            rules[lab] = cond
        else:
            # Do not turn another unparsed quantitative assertion into 'otherwise'.
            if _has_quantity_bound(text):
                return None
            rest.append(lab)
    if not rules or len(rest) > 1:
        return None
    if rest:
        rules[rest[0]] = "otherwise"
    return rules


def _direction_name(labels: list[str], higher: bool) -> str | None:
    for lab in labels:
        token = str(lab).replace("_", " ").split()[0]
        if higher and (_UP_NAME.match(lab) or _UP_NAME.match(token)):
            return lab
        if not higher and (_DOWN_NAME.match(lab) or _DOWN_NAME.match(token)):
            return lab
    return None


def _directional_rules(labels: list[str], base_name: str) -> dict[str, str] | None:
    ups = [lab for lab in labels if _direction_name([lab], True)]
    downs = [lab for lab in labels if _direction_name([lab], False)]
    if len(labels) != 2 or len(ups) != 1 or len(downs) != 1:
        return None
    # The cut belongs to the lower label, so the line is partitioned.
    return {ups[0]: f"Q > {base_name}", downs[0]: "otherwise"}


def _rule_identifiers(rules: dict | None) -> set[str]:
    """Names cited by a threshold condition. ``Q`` and ``otherwise`` are not fields."""
    names: set[str] = set()
    if not isinstance(rules, dict):
        return names
    for cond in rules.values():
        if not isinstance(cond, str) or cond.strip().lower() in _OTHERWISE:
            continue
        names.update(_IDENT.findall(cond))
    names.discard("Q")
    return names


def _reference_field(ent: dict, rules: dict | None = None) -> str | None:
    """Base field the threshold names. Entity key order does not choose it.

    ``consensus_eps`` is selected when the condition cites it, even if
    ``prior_year_q_eps`` is stored first. A key is a whole identifier, so
    ``consensus`` does not match ``consensus_eps``. With no cited base field the
    choice is the longest name, then alphabetical.
    """
    bases = [str(key) for key in ent if _BASE_FIELD.search(str(key)) and D(ent.get(key)) is not None]
    if not bases:
        return None
    mentioned = [name for name in bases if name in _rule_identifiers(rules)]
    pool = mentioned or bases
    pool.sort(key=lambda name: (-len(name), name))
    return pool[0]


def _declared_kind(ent: dict) -> str | None:
    from .predict import _unit_of

    unit = _unit_of(ent.get("unit"), ent.get("units"))
    if unit is not None:
        return "bps" if unit == "bps" else "percent"
    text = f"{ent.get('unit') or ''} {ent.get('units') or ''} {ent.get('currency') or ''}"
    return "level" if _LEVEL_WORD.search(text) else None


def _decimals_for(spec: TaskSpec, ent: dict, reference: Decimal | None, kind: str,
                  plan: dict | None = None, reference_name: str = "") -> int:
    quantity = _numeric_kind(spec, ent, plan)
    if kind == "absolute_bp" or quantity == "bps":
        return BPS_DECIMALS
    if quantity == "percent":
        return GROWTH_PERCENT_DECIMALS
    text = f"{spec.target_name} {reference_name} {ent.get('unit') or ''} {ent.get('units') or ''} {ent.get('currency') or ''}"
    if _DILUTED_LEVEL.search(text) or re.search(r"eps|per share|dollar|(?:usd|us\$)\s*/\s*share", text, re.I):
        return max(EPS_LEVEL_DECIMALS, _published_decimals(reference, 0))
    return _published_decimals(reference, 0)


def build_card(spec: TaskSpec, ent: dict, plan: dict | None = None) -> ThresholdCard | None:
    """The task's threshold card. Planner rules are not allowed to outrank the task."""
    labels = list(getattr(spec, "labels", None) or [])
    if not labels or getattr(spec, "target_type", "") != "classification":
        return None
    raw = getattr(spec, "raw", None) or {}
    target = raw.get("target") if isinstance(raw.get("target"), dict) else {}
    assertions = target.get("label_assertions") if isinstance(target.get("label_assertions"), dict) else {}
    rules = _table_rules(labels, ent)
    if rules is None:
        rules = _assertion_rules(labels, assertions, ent)
    ref_name = _reference_field(ent, rules)
    if rules is None and ref_name and not any(_has_quantity_bound(v) for v in assertions.values()):
        rules = _directional_rules(labels, ref_name)
    if rules is None:
        return None
    fields = {k: v for k, v in ent.items() if D(v) is not None}
    kind = "absolute"
    if any("threshold_pct" in str(v) for v in rules.values()):
        kind = "relative_pct"
    reference = D(fields.get(ref_name)) if ref_name else None
    decimals = _decimals_for(spec, ent, reference, kind, plan, ref_name or "")
    return card_from_rules(
        labels, rules, fields, kind=kind, reference=reference,
        reference_name=ref_name or "", decimals=decimals,
    )


def _as_float(value) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _submitted_probability(spec: TaskSpec, ent: dict, plan: dict | None, row: dict) -> bool:
    """Use finalize's per-row decision, including its planner-only numeric correction.

    A shared planner flag can stay true after finalize restores one row's EPS or other
    numeric level. Reuse its semantic check with the submitted point instead of
    reclassifying that row from the flag or from an interval that happens to be [0, 1].
    Schema, card and genuine row probability signals retain their original precedence.
    """
    from .calibrate import _numeric_view, _planner_only, is_probability
    from .predict import Forecast

    if not is_probability(plan, spec):
        return False
    if not _planner_only(plan, spec):
        return True
    view = Forecast(eid=str(ent.get("entity_id", "")), median=_as_float(row.get("point_forecast")))
    return _numeric_view(spec, view, plan, ent) is None


def _field_kind(name: str) -> str | None:
    """Unit of one threshold field, using the same level / rate split as the baselines."""
    if _BPS_WORD.search(name):
        return "bps"
    if _PERCENT_WORD.search(name) or re.search(r"growth|percent|\bpct\b", name, re.I):
        return "percent"
    if _BASE_FIELD.search(name):
        return "level"
    return None


def _assertion_text(spec: TaskSpec) -> str:
    raw = getattr(spec, "raw", None) or {}
    target = raw.get("target") if isinstance(raw.get("target"), dict) else {}
    assertions = target.get("label_assertions") if isinstance(target.get("label_assertions"), dict) else {}
    return " ".join(str(v) for v in assertions.values())


def _threshold_kind(spec: TaskSpec, card: ThresholdCard) -> str | None:
    """Unit of Q on this card. A percent band around an EPS level is still that level."""
    rules = card.prepared_rules()
    names = _rule_identifiers(rules)
    ref = card.reference_name or ""
    # Whole identifier: a reference named consensus must not match consensus_eps.
    if ref and ref in names:
        return _field_kind(ref)
    mentioned = [name for name in names if _BASE_FIELD.search(name)]
    if mentioned:
        mentioned.sort(key=lambda name: (-len(name), name))
        return _field_kind(mentioned[0])
    if card.kind == "absolute_bp":
        return "bps"
    text = _assertion_text(spec)
    bps, pct = bool(_BPS_WORD.search(text)), bool(_PERCENT_WORD.search(text))
    if bps and not pct:
        return "bps"
    if pct and not bps:
        return "percent"
    return None


def _submitted_kind(spec: TaskSpec, ent: dict, plan: dict | None, row: dict) -> str | None:
    """Unit of the submitted point, from the row checks v21.13 already uses.

    Order matches those checks: a probability unit is not a level, even when a noisy
    bound or the point leaves [0, 1]. A growth rate or a return is not the level column
    it is measured against, and only then is a baseline a level.
    """
    if _submitted_probability(spec, ent, plan, row):
        return "probability"
    return _numeric_kind(spec, ent, plan)


def _numeric_kind(spec: TaskSpec, ent: dict, plan: dict | None) -> str | None:
    from .predict import _CHANGE_QUANTITY, _unit_of, event_return_sd, quantity_baseline
    from .stats import relative_quantity

    # The current row's own measured unit precedes shared target/planner wording.
    declared = _declared_kind(ent)
    if declared is not None:
        return declared
    unit = _unit_of(spec.target_name)
    if unit is None and plan:
        unit = _unit_of(plan.get("target_quantity"), plan.get("interval_quantity"))
    if unit == "bps":
        return "bps"
    if unit == "pct":
        return "percent"
    if event_return_sd(spec) is not None or relative_quantity(spec.target_name, ent):
        return "percent"
    qtext = f"{(plan or {}).get('interval_quantity') or ''} {ent.get('unit') or ''} {ent.get('units') or ''}"
    if _CHANGE_QUANTITY.search(qtext):
        return "percent"
    if quantity_baseline(spec, ent, plan) is not None:
        return "level"
    return None


def _same_quantity(spec: TaskSpec, ent: dict, plan: dict | None, row: dict, card: ThresholdCard) -> bool:
    """True only when the point and the threshold name one known, shared unit."""
    submitted, threshold = _submitted_kind(spec, ent, plan, row), _threshold_kind(spec, card)
    return submitted is not None and submitted == threshold


def _submitted_window(row: dict) -> Region | None:
    """Closed submitted interval, ordered. Missing endpoints are not a window."""
    left, right = _as_float(row.get("lo")), _as_float(row.get("hi"))
    if left is None or right is None:
        return None
    if right < left:
        left, right = right, left
    lo, hi = D(left), D(right)
    if lo is None or hi is None:
        return None
    return Region(lo, hi, True, True)


def _intersect(region: Region, window: Region) -> Region | None:
    """Overlap of the label region and the submitted interval. None when it is empty."""
    lo, closed_lo = region.lo, region.closed_lo
    if window.lo is not None and (lo is None or window.lo > lo or (window.lo == lo and not window.closed_lo)):
        lo, closed_lo = window.lo, window.closed_lo
    elif window.lo is not None and window.lo == lo:
        closed_lo = closed_lo and window.closed_lo
    hi, closed_hi = region.hi, region.closed_hi
    if window.hi is not None and (hi is None or window.hi < hi or (window.hi == hi and not window.closed_hi)):
        hi, closed_hi = window.hi, window.closed_hi
    elif window.hi is not None and window.hi == hi:
        closed_hi = closed_hi and window.closed_hi
    if lo is not None and hi is not None and (lo > hi or (lo == hi and not (closed_lo and closed_hi))):
        return None
    return Region(lo, hi, closed_lo, closed_hi)


def _snap(point: Decimal, region: RegionLike, window: Region | None, decimals: int) -> Decimal | None:
    """Nearest published grid value inside the label.

    The submitted interval wins when it overlaps the label on a legal grid point
    (down versus 102.3058 with [100, 102] is 102.0000, not the cut). No such point
    means there is nothing to clamp to, and the region snap stands alone so the
    interval can widen.
    """
    rounded = round_to_decimals(point, decimals)
    held: list[Decimal] = []
    free: list[Decimal] = []
    for part in region.pieces():
        if window is not None:
            overlap = _intersect(part, window)
            if overlap is not None:
                candidate = nudge_into(rounded, overlap, decimals)
                if candidate is not None and overlap.contains(candidate):
                    held.append(candidate)
                    continue
        candidate = nudge_into(rounded, part, decimals)
        if candidate is not None:
            free.append(candidate)
    # Compare against the original point so 0.004 chooses the positive outside ray.
    return min(held or free, key=lambda x: (abs(x - point), x), default=None)


def _cover(lo, hi, point: float) -> tuple[float | None, float | None, bool]:
    """Expand an endpoint only when the point lies outside. Missing endpoints stay missing."""
    left, right = _as_float(lo), _as_float(hi)
    if left is None or right is None:
        return lo, hi, False
    changed = right < left
    if right < left:
        left, right = right, left
    if point < left:
        left, right, changed = point, right, True
    if point > right:
        right, changed = point, True
    return left, right, changed


def apply_threshold_consistency(spec: TaskSpec, ent: dict, plan: dict | None, row: dict) -> tuple[dict, list[str]]:
    """Return the row and the codes for any consistency repair. The forecast model is not replaced."""
    out = dict(row or {})
    codes: list[str] = []
    point = _as_float(out.get("point_forecast"))
    if point is None:
        return out, codes
    card = build_card(spec, ent, plan) if getattr(spec, "target_type", "") == "classification" else None
    label = out.get("label") if isinstance(out.get("label"), str) else None
    # A probability or a growth rate does not share a unit with an EPS-level cut. Leave it.
    same = card is not None and _same_quantity(spec, ent, plan, out, card)
    region = card.regions.get(label) if same and label is not None else None
    # A point can belong to the submitted label even at a shared closed endpoint.
    if region is not None and not region.contains(D(point)):
        nudged = _snap(D(point), region, _submitted_window(out), card.decimals)
        if nudged is None or not region.contains(nudged):
            raise ThresholdConsistencyError(
                f"point={point} label={label}"
            )
        if float(nudged) != float(point):
            point = float(nudged)
            out["point_forecast"] = point
            codes.append("THRESHOLD_FIX")
    lo, hi, covered = _cover(out.get("lo"), out.get("hi"), point)
    if covered:
        out["lo"], out["hi"] = lo, hi
        codes.append("INTERVAL_COVER")
    if region is not None and not region.contains(D(out.get("point_forecast"))):
        raise ThresholdConsistencyError(
            f"point={out.get('point_forecast')} label={label}"
        )
    if _as_float(out.get("lo")) is None or _as_float(out.get("hi")) is None:
        return out, codes
    if float(out["lo"]) > float(out["point_forecast"]) or float(out["hi"]) < float(out["point_forecast"]):
        raise ThresholdConsistencyError("interval misses point")
    return out, codes
