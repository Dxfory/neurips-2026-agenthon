"""Label coherence: labels defined by thresholds on a numeric quantity follow that quantity.

The planner states each label as a condition on the quantity Q in terms of entity-table fields,
e.g. {"beat": "Q > consensus_eps * (1 + threshold_pct)", "miss": "Q < consensus_eps * (1 - threshold_pct)",
"inline": "otherwise"}. The condition's arithmetic is evaluated safely (numbers, entity fields,
+ - * /, abs) -- never with eval. Given the aggregated forecast distribution of Q (median and
5th/95th percentiles, read as a two-piece normal), each label gets the probability mass of its
region; that is blended with the model's own label probabilities, so a label can no longer
contradict the quantity it is defined by.
"""
from __future__ import annotations

import ast
import math
import operator
import re

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}
_COND = re.compile(r"^\s*Q\s*(>=|<=|>|<)\s*(.+)$")
_BAND = re.compile(r"^\s*(.+?)\s*(?:<=|<)\s*Q\s*(?:<=|<)\s*(.+)$")  # v13: 'lo < Q < hi', a label that is a band


def _eval(expr: str, ent: dict) -> float | None:
    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError:
        return None

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return float(node.value)
        if isinstance(node, ast.Name):
            v = ent.get(node.id)
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise ValueError(node.id)
            return float(v)
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -ev(node.operand)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "abs" and len(node.args) == 1:
            return abs(ev(node.args[0]))
        raise ValueError("unsupported expression")

    try:
        v = ev(tree)
    except (ValueError, ZeroDivisionError, TypeError):
        return None
    return v if math.isfinite(v) else None


def _cdf(x: float, med: float, s_lo: float, s_hi: float) -> float:
    """CDF of the distribution whose 5th, 50th and 95th percentiles are med - 1.645 s_lo, med and med + 1.645 s_hi:
    a half normal of scale s_lo below the median and one of scale s_hi above it, each holding half the mass.
    v11: v10 used a two-piece normal with `med` as its MODE, which moves the median towards the narrow side
    (median 0, p05 -1, p95 9 gave P(Q > 0) = 0.90 and P(Q > 1) = 0.77, against the stated median)."""
    s = max(s_lo if x < med else s_hi, 1e-9)
    return 0.5 * (1 + math.erf((x - med) / s / math.sqrt(2)))


def rule_label(rules: dict, labels: list[str], ent: dict, q: float) -> str | None:
    """v21.2: the one label whose condition holds at Q = q ('otherwise' when none of the others does), or None when the
    rules do not read on this row or more than one label holds."""
    if not isinstance(rules, dict) or not labels:
        return None
    hold, otherwise = [], None
    for lab in labels:
        cond = rules.get(lab)
        if not isinstance(cond, str):
            return None
        if cond.strip().lower() in ("otherwise", "else"):
            otherwise = lab
            continue
        m = _COND.match(cond)
        band = None if m else _BAND.match(cond)
        if m:
            thr = _eval(m.group(2), ent)
            if thr is None:
                return None
            ok = {">": q > thr, ">=": q >= thr, "<": q < thr, "<=": q <= thr}[m.group(1)]
        elif band:
            lo, hi = _eval(band.group(1), ent), _eval(band.group(2), ent)
            if lo is None or hi is None:
                return None
            ok = lo <= q <= hi
        else:
            return None
        if ok:
            hold.append(lab)
    if len(hold) == 1:
        return hold[0]
    return otherwise if not hold else None


def implied_probs(rules: dict, labels: list[str], ent: dict, med: float, p05: float | None,
                  p95: float | None) -> dict[str, float] | None:
    """Label probabilities implied by the quantity distribution, or None when the rules don't apply."""
    if not isinstance(rules, dict) or not labels:
        return None
    s_lo = (med - p05) / 1.645 if p05 is not None and p05 < med else None
    s_hi = (p95 - med) / 1.645 if p95 is not None and p95 > med else None
    if s_lo is None and s_hi is None:
        return None
    s_lo, s_hi = s_lo or s_hi, s_hi or s_lo
    probs: dict[str, float] = {}
    otherwise = None
    for lab in labels:
        cond = rules.get(lab)
        if not isinstance(cond, str):
            return None
        if cond.strip().lower() in ("otherwise", "else"):
            if otherwise is not None:
                return None
            otherwise = lab
            continue
        m, band = _COND.match(cond), None if _COND.match(cond) else _BAND.match(cond)
        if m:
            thr = _eval(m.group(2), ent)
            bounds = [thr]
        elif band:
            bounds = [_eval(band.group(1), ent), _eval(band.group(2), ent)]
        else:
            return None
        if any(b is None for b in bounds) or (band and not bounds[0] < bounds[1]):
            return None
        if any(abs(b - med) > 50 * max(s_lo, s_hi) for b in bounds):  # dozens of spreads away: another scale
            return None
        if band:
            probs[lab] = max(0.0, _cdf(bounds[1], med, s_lo, s_hi) - _cdf(bounds[0], med, s_lo, s_hi))
            continue
        below = _cdf(bounds[0], med, s_lo, s_hi)
        probs[lab] = (1.0 - below) if m.group(1) in (">", ">=") else below
    if otherwise is not None:
        probs[otherwise] = max(0.0, 1.0 - sum(probs.values()))
    total = sum(probs.values())
    if total <= 0 or total > 1.2 or len(probs) != len(labels):  # overlapping regions: rules are not a partition
        return None
    return {lab: p / total for lab, p in probs.items()}


_UP = re.compile(r"^(?:up|higher|increase[sd]?|rise[sn]?|above|positive|gain)$", re.I)
_DOWN = re.compile(r"^(?:down|lower|decrease[sd]?|fall|falls|below|negative|decline[sd]?)$", re.I)


def directional_pair(labels: list[str]) -> tuple[str, str] | None:
    """(the 'higher' label, the 'lower' label) of a two-label directional target, else None."""
    if len(labels) != 2:
        return None
    ups = [lab for lab in labels if _UP.match(lab.strip())]
    downs = [lab for lab in labels if _DOWN.match(lab.strip())]
    return (ups[0], downs[0]) if len(ups) == 1 and len(downs) == 1 else None


def rules_parse(rules, labels: list[str], ent: dict) -> bool:
    """Whether every label has a condition `implied_probs` can read on this row (at most one 'otherwise')."""
    if not isinstance(rules, dict) or not labels:
        return False
    other = 0
    for lab in labels:
        cond = rules.get(lab)
        if not isinstance(cond, str):
            return False
        if cond.strip().lower() in ("otherwise", "else"):
            other += 1
            continue
        m = _COND.match(cond)
        band = None if m else _BAND.match(cond)
        exprs = [m.group(2)] if m else [band.group(1), band.group(2)] if band else []
        vals = [_eval(x, ent) for x in exprs]
        if not exprs or any(v is None for v in vals) or (band and not vals[0] < vals[1]):
            return False  # v14: a band written backwards ('2.6 < Q < 2.3') does not read either
    return other <= 1


# v13: the task's own label definitions. A label asserted as a numeric threshold or band on the quantity ("bid-to-cover
# ratio above 2.50", "growth from -20% to below -5%", "between -1% and 1%") is a rule on Q.
# v14: only an ABSOLUTE threshold is read. A definition that measures the quantity against another one -- the consensus,
# an estimate, an earlier period, an average, a multiple -- is not (v13 read "more than 5 percent above the consensus"
# as Q > 5: the unit word could be given back to the number, and the reference was never looked for).
_REFERENCE = re.compile(r"\b(?:consensus|estimates?|expectations?|expected|forecasts?|guidance|projections?|prior|"
                        r"previous|preceding|last|earlier|ago|before|average|mean|median|baseline|benchmark|"
                        r"same (?:quarter|month|period|week|year)|times|fold|multiple|relative|versus|vs\.?|compared|"
                        r"respect to)\b", re.I)
_UNIT = r"(?:%|percent(?:age points?)?|per cent|pct|bps|bp|basis points?)"
_NUM = rf"([-]?\d+(?:\.\d+)?)(?>\s*{_UNIT})?"  # atomic: a unit read with its number is never given back
_TAIL = r"(?![\d.])(?!\s+(?:above|below|over|under|of|than|from|higher|lower|more|less|greater|smaller|up|down|in excess)\b)"
_GT = r"(?:above|greater than|more than|higher than|exceeds?|exceeding|over|in excess of|at least)"
_LT = r"(?:below|less than|lower than|under|at most|no more than)"
_A_BAND = [re.compile(rf"\b(?:between|from)\s+{_NUM}\s+(?:and|to|up to)\s+(?:below\s+|under\s+)?{_NUM}{_TAIL}", re.I),
           re.compile(rf"\b{_GT}\s+{_NUM}\s+(?:and\s+)?(?:up to|to|and at most|and below|but below)\s+{_NUM}{_TAIL}", re.I)]
_A_GT = [re.compile(rf"\b{_GT}\s+{_NUM}{_TAIL}", re.I), re.compile(rf"{_NUM}\s+or\s+(?:more|above|higher|greater){_TAIL}", re.I),
         re.compile(rf"(?:>|≥)\s*{_NUM}{_TAIL}")]
_A_LT = [re.compile(rf"\b{_LT}\s+{_NUM}{_TAIL}", re.I), re.compile(rf"{_NUM}\s+or\s+(?:less|below|lower|fewer){_TAIL}", re.I),
         re.compile(rf"(?:<|≤)\s*{_NUM}{_TAIL}")]


# v21.2: a move stated with its direction ("falls by more than 1.0%", "widens by more than 10 bp") and a symmetric band
# ("by 1.0% or less in either direction", "within +/-1%"). v21 read "falls by more than 1.0%" as Q > 1.0 -- the same
# region as "rises by more than 1.0%" -- and "1.0% or less in either direction" as Q < 1.0; such rules overlap, and
# when they still summed to about one they labelled a -3% forecast 'stable'.
_FALL = (r"(?:falls?|fell|falling|declines?|declined|declining|drops?|dropped|dropping|decreases?|decreased|decreasing|"
         r"weakens?|weakened|depreciates?|depreciated|loses|lost|shrinks?|shrank|narrows?|narrowed|tightens?|tightened|"
         r"contracts?|contracted|slides?|slid|slips?|slipped|plunges?|plunged|sinks?|sank)")
_RISE = (r"(?:rises?|rose|rising|increases?|increased|increasing|gains?|gained|climbs?|climbed|grows?|grew|"
         r"strengthens?|strengthened|appreciates?|appreciated|widens?|widened|jumps?|jumped|advances?|advanced|"
         r"expands?|expanded|surges?|surged)")
_MORE = r"(?:by\s+)?(?:more than|over|in excess of|at least|greater than)"
_A_DIR = [(re.compile(rf"\b{_FALL}\s+{_MORE}\s+{_NUM}{_TAIL}", re.I), "<"),
          (re.compile(rf"\b{_RISE}\s+{_MORE}\s+{_NUM}{_TAIL}", re.I), ">")]
_A_EITHER = [re.compile(rf"\b(?:by\s+)?{_NUM}\s+or\s+less\s+in\s+(?:either|both|any)\s+directions?", re.I),
             re.compile(rf"\bwithin\s+(?:±|plus or minus)?\s*{_NUM}", re.I),
             re.compile(rf"(?:±|plus or minus)\s*{_NUM}", re.I)]


def _assertion_cond(text) -> str | None:
    if not isinstance(text, str):
        return None
    t = text.replace("\u2212", "-").replace("\u2013", "-").replace("+/-", "±").replace("+-", "±").replace("+", "")
    if _REFERENCE.search(t):
        return None  # measured against another quantity: not an absolute threshold
    found = []
    for pat in _A_EITHER:
        found += [f"{-abs(float(m.group(1)))!r} < Q < {abs(float(m.group(1)))!r}" for m in pat.finditer(t)
                  if float(m.group(1)) != 0]
    if not found:
        for pat, op in _A_DIR:
            found += [f"Q < {-abs(float(m.group(1)))!r}" if op == "<" else f"Q > {abs(float(m.group(1)))!r}"
                      for m in pat.finditer(t)]
    if found:
        return found[0] if len(set(found)) == 1 else None
    for pat in _A_BAND:
        for m in pat.finditer(t):
            lo, hi = float(m.group(1)), float(m.group(2))
            if lo < hi:
                found.append(f"{lo!r} < Q < {hi!r}")
    if not found:
        found += [f"Q > {float(m.group(1))!r}" for pat in _A_GT for m in pat.finditer(t)]
        found += [f"Q < {float(m.group(1))!r}" for pat in _A_LT for m in pat.finditer(t)]
    return found[0] if len(set(found)) == 1 else None  # two different conditions in one sentence: not read


def assertion_rules(labels: list[str], assertions) -> dict[str, str] | None:
    """Label rules read off the task's label_assertions when every label but at most one states a numeric threshold or
    band on the quantity (the remaining label is 'otherwise'); None otherwise."""
    if not isinstance(assertions, dict) or not labels:
        return None
    rules, rest = {}, []
    for lab in labels:
        cond = _assertion_cond(assertions.get(lab))
        if cond:
            rules[lab] = cond
        else:
            rest.append(lab)
    if not rules or len(rest) > 1:
        return None
    if rest:
        rules[rest[0]] = "otherwise"
    return rules


_TABLE_COND = re.compile(r"^\s*[A-Za-z_][A-Za-z0-9_]*\s*(>=|<=|>|<)\s*(.+)$")


def table_rules(labels: list[str], ent: dict) -> dict[str, str] | None:
    """Label rules read off the entity table's own '<label>_condition' columns, when the table states them
    (e.g. beat_condition 'reported_eps > consensus_eps * (1 + threshold_pct)'): the quantity on the left becomes
    Q, and the one label without a one-sided condition (the middle band) is 'otherwise'. None unless every
    other label has a one-sided condition whose right-hand side evaluates on this row."""
    rules: dict[str, str] = {}
    for lab in labels:
        cond = ent.get(f"{lab}_condition")
        m = _TABLE_COND.match(cond) if isinstance(cond, str) else None
        if m and _eval(m.group(2), ent) is not None:
            rules[lab] = f"Q {m.group(1)} {m.group(2).strip()}"
    rest = [lab for lab in labels if lab not in rules]
    if not rules or len(rest) > 1:
        return None
    if rest:
        rules[rest[0]] = "otherwise"
    return rules


def directional_rules(labels: list[str], threshold: float) -> dict[str, str] | None:
    """{'up': 'Q > threshold', 'down': 'otherwise'} for a two-label directional target, else None."""
    pair = directional_pair(labels)
    return {pair[0]: f"Q > {float(threshold)!r}", pair[1]: "otherwise"} if pair else None
