"""Runtime statistical baselines, backtested on each entity's own pre-cutoff history.

Runtime series come from the unit's frozen corpus, but several method choices and interval
constants were selected offline on later history. Public entry points require an eligible
task policy; otherwise they use the corpus-only rule in runtime_naive. Raw numerical helpers
are for the admitted backend and development diagnostics, not independent runtime entry points.

For an entity whose own series appears in a corpus table (v12: the same series also sizes the interval, see
series_half_width; a ratio column's change is measured on its numerator in its starting denominator; a growth
column serving a growth target is forecast at its next value):
  * next-value targets (a bid-to-cover, a monthly print): last value, recent means, median,
    exponential smoothing and drift are backtested one step ahead on the series itself;
  * change-over-horizon targets (a yield change, a positioning change): zero change, drift,
    momentum and reversal are backtested at the horizon when the history is long enough,
    otherwise zero change with a random-walk error scale.
The method with the lowest backtest MAE supplies the baseline forecast; its empirical error
quantile supplies an interval floor.
"""
from __future__ import annotations

import math
import re
import statistics
from dataclasses import dataclass

from .artifact_policy import ArtifactPolicy
from .corpus import Corpus
from .tables import _period_days, _tokens, parse_tables, to_number

_CHANGE_TARGET = re.compile(r"change|growth|delta|shift", re.I)
# A next-value target whose values are themselves period changes (a monthly % change, a return): the
# changes of a random-walk level are close to independent, so the last change is a noisy forecast of
# the next and the recent median is the default instead.
_RATE_SERIES = re.compile(r"(?:^|[\W_])mom(?:$|[\W_])|pct_change|percent[\s_]change|month[\s_-]over[\s_-]month|return", re.I)
MIN_BACKTEST = 4
MIN_SWITCH = 8  # backtest origins needed before a method may replace the default
# v11: a target that is a RELATIVE change (growth, a percent change) is not the difference of a level series: the
# change methods below difference the series in its own units, so a column of EPS levels gave a 'growth' forecast in
# dollars (scale the dollars by 10 and the 'growth' moved by 10). Such a target only takes a column that is itself
# a change or a growth rate; EPS growth is left to the EPS anchor, which reads the plan's percent-change rule.
_RELATIVE_TARGET = re.compile(r"growth|pct[\s_]change|percent[\s_]change|yoy[\s_]pct", re.I)
# v12: a column that is itself a period change or rate (a bare '%' is not: 'net % of OI' is a share)
_RATE_COLUMN = re.compile(r"growth|change|chg|yoy|y/y|mom|m/m|return|pct[\s_]change|percent[\s_]change", re.I)
_NOT_LEVEL = re.compile(r"%|pct|percent|share|ratio|change|chg|wow|delta|diff|growth|yoy|mom|return", re.I)


def _level_column_for_relative_target(target_name: str, column: str) -> bool:
    return bool(_RELATIVE_TARGET.search(target_name)) and not _RATE_COLUMN.search(column)


def _same_quantity(target_name: str, column: str) -> bool:
    """v12: a growth or rate column serving a growth target is the target quantity itself, so its NEXT value is the
    forecast. v11 differenced it like a level: growth held at 10% every period forecast 0%."""
    return bool(_RELATIVE_TARGET.search(target_name)) and bool(_RATE_COLUMN.search(column))


# v21.1: a relative quantity named by the target or by the row's own unit text ('percent change of the value of 1 EUR
# in U.S. dollars', 'index 2012=100 (target: percent change)'). A percentage-POINT change and a level quoted in percent
# are not relative: their changes are differences.
_REL_ANY = re.compile(r"growth|pct[\s_]change|percent(?:age)?[\s_]+change|%[\s_]change|(?:^|[\W_])(?:mom|qoq|m/m|q/q)(?:$|[\W_])"
                      r"|month[\s_-]over[\s_-]month|quarter[\s_-]over[\s_-]quarter", re.I)
_YOY = re.compile(r"(?:^|[\W_])(?:yoy|y/y)(?:$|[\W_])|year[\s_-]over[\s_-]year|annual", re.I)
_VINTAGE_COL = re.compile(r"as[\s_-]*of|vintage", re.I)
# v21.1: a target measured against something other than the series' own past (a revision of a given month's estimate,
# a surprise against consensus, an abnormal return): the series' next value or change is not the target, so an
# unnamed table gives it no baseline (the macro revision variant read 158,779 payrolls as its percent revision)
_OTHER_REFERENCE = re.compile(r"revision|revised|surprise|beat|miss|abnormal|excess|consensus|relative[\s_]to", re.I)


def relative_quantity(target_name: str, ent: dict) -> bool:
    text = f"{target_name} {ent.get('unit') or ''} {ent.get('units') or ''}"
    return bool(_REL_ANY.search(text))


def _own_table_series(corpus: Corpus, doc_ids: list[str], ent: dict, target_name: str,
                      entity_docs: set[str]) -> Series | None:
    """v21.1: the series of a date-indexed table in one of the entity's OWN documents, whatever its column is called:
    a two-column 'date | value' table, or a vintage table read off its latest vintage (every period as last published
    before the cutoff). The public units name the column after the series or the target; a held-out family need not,
    and without a series v21 had no baseline and no scale (a housing-starts table without its column name gave 0, an
    unnamed unemployment rate an interval of +/-20 points). For a relative target (a percent change) the level is
    taken as 100 x ln(level), whose changes are percent changes; a year-over-year rate is not derived (its base
    period is twelve steps back) and such a target gets no series here."""
    text = f"{target_name} {ent.get('unit') or ''} {ent.get('units') or ''}"
    if _OTHER_REFERENCE.search(text):
        return None
    rel = relative_quantity(target_name, ent)
    if rel and _YOY.search(text):
        return None
    for did in doc_ids:
        if did not in entity_docs:
            continue
        doc = corpus.docs.get(did)
        if doc is None:
            continue
        for t in parse_tables(did, doc.text):
            if len(t.header) < 2 or _period_days([r[0] for r in t.rows]) is None:
                continue
            cols = list(range(1, len(t.header)))
            if len(cols) == 1:
                j = 1
            elif all(_VINTAGE_COL.search(t.header[c]) for c in cols):
                j = cols[-1]
            else:
                continue
            pairs = [(r[0], to_number(r[j])) for r in t.rows if j < len(r)]
            pairs = [(d, v) for d, v in pairs if v is not None]
            if len(pairs) < 6:
                continue
            dates, vals = [d for d, _ in pairs], [float(v) for _, v in pairs]
            header = t.header[j].strip()
            if rel:
                if _RATE_COLUMN.search(header) or min(vals) <= 0:
                    continue  # a column of rates is not a level to take logs of
                return Series(did, header + " (100 x ln level)", [100.0 * math.log(v) for v in vals], dates,
                              _period_days(dates), False, kind="own-rel")
            return Series(did, header, vals, dates, _period_days(dates), False, kind="own")
    return None


def _resolution(values: list[float]) -> float:
    """v21.1: the series' own resolution -- its smallest nonzero step, else one unit of its last printed decimal -- a
    floor for a measured scale: a series flat over its window (an insured unemployment rate at 1.2 for 26 weeks) has
    a zero backtest error, and v21 then fell back to a +/-20 default."""
    steps = [abs(b - a) for a, b in zip(values, values[1:]) if abs(b - a) > 1e-12]
    if steps:
        return min(steps)
    dec = max((len(repr(float(v)).split(".")[1].rstrip("0")) for v in values if math.isfinite(v)), default=0)
    return 10.0 ** -min(dec, 6) if dec else 1.0


def _ratio_parts(header: str) -> tuple[set[str], set[str]] | None:
    """('net', 'oi') for a header such as 'net_%OI' or 'net % of OI'; None when the header is not a ratio."""
    m = re.match(r"(?i)^(.*?[a-z0-9])[\s_]*(?:%|pct[\s_]+of|percent[\s_]+of|share[\s_]+of)[\s_]*(?:of[\s_]+)?(.+)$", header.strip())
    if not m:
        return None
    num, den = _tokens(m.group(1)) - {"of"}, _tokens(m.group(2)) - {"of"}
    return (num, den) if num and den else None


def _ratio_columns(t, j: int) -> tuple[list[float], list[float], float] | None:
    """For a ratio column j of table t ('X % of Y'): the numerator X and denominator Y columns of the same table and
    the scale (100 for a percent, 1 for a fraction), when the table itself confirms column j = X / Y * scale on every
    row; else None."""
    parts = _ratio_parts(t.header[j])
    if parts is None:
        return None
    num_t, den_t = parts

    def initials(toks: list[str]) -> str:
        return "".join(x[0] for x in toks)

    num_j = den_j = None
    for k, name in enumerate(t.header):
        if k in (0, j):
            continue
        toks = [x for x in re.split(r"[^a-z0-9]+", name.lower()) if x]
        if num_j is None and num_t <= set(toks) and not _NOT_LEVEL.search(name):
            num_j = k
        if den_j is None and (den_t <= set(toks) or initials(toks) == "".join(sorted(den_t))
                              or initials(toks) in den_t) and not _NOT_LEVEL.search(name):
            den_j = k
    if num_j is None or den_j is None:
        return None
    num = [to_number(r[num_j]) if num_j < len(r) else None for r in t.rows]
    den = [to_number(r[den_j]) if den_j < len(r) else None for r in t.rows]
    rat = [to_number(r[j]) if j < len(r) else None for r in t.rows]
    if any(v is None for v in num + den + rat) or any(d == 0 for d in den):
        return None
    for scale in (100.0, 1.0):
        errs = [abs(r - n / d * scale) for n, d, r in zip(num, den, rat)]
        tol = [0.02 * max(abs(r), 1e-9) + 0.051 for r in rat]  # rounding of the printed ratio
        if all(e <= t_ for e, t_ in zip(errs, tol)):
            return [float(x) for x in num], [float(x) for x in den], scale
    return None


@dataclass
class Series:
    doc_id: str
    header: str
    values: list[float]
    dates: list[str]
    period_days: float | None
    exact: bool  # header named by the entity's own code/name (vs. a target-name match in its own doc)
    # v12: a change of a ratio column 'X % of Y' is (X[t+h] - X[t]) / Y[t] * scale -- the numerator's change measured
    # in the starting denominator, as a task such as 'the change in net position as a percent of the starting open
    # interest' defines it. v11 differenced the printed ratio, X[t+h]/Y[t+h] - X[t]/Y[t], which differs whenever Y
    # moves (gold, 2024-06-18 to 07-23: +6.75 points by the task's definition, -7.00 as differenced).
    denom: list[float] | None = None
    denom_scale: float = 1.0
    # v21.1: '' -- a column named by the entity or the target (v12-v21); 'own' -- the single series of a date-indexed
    # table in the entity's own document, whatever its column is called; 'own-rel' -- such a level series turned into
    # 100 x ln(level), so its changes are percent changes, for a relative target (see _own_table_series)
    kind: str = ""


@dataclass
class StatForecast:
    method: str
    value: float  # forecast of the TARGET, in target units
    mae: float | None
    q90: float | None  # 90% absolute-error quantile, target units (inflated for small samples)
    n_backtest: int
    last: float  # last observed series value, series units
    series: Series

    def describe(self) -> str:
        bt = (f"backtest MAE {self.mae:.3g} over {self.n_backtest} origins, 90% abs error {self.q90:.3g}"
              if self.mae is not None and self.q90 is not None else "not backtestable (short history)")
        return (f"series '{self.series.header}' in {self.series.doc_id} ({len(self.series.values)} obs, last "
                f"{self.last:g}); best simple method '{self.method}' -> {self.value:.4g} ({bt})")


def find_series(corpus: Corpus, doc_ids: list[str], ent: dict, target_name: str,
                entity_docs: set[str], allow_own: bool = False) -> Series | None:
    """The entity's own numeric series in a corpus table, or None. v21.1: with `allow_own`, when no column is named by
    the entity or the target, the series of a date-indexed table in the entity's own document (_own_table_series)."""
    ids = {str(ent.get(k)).strip().lower() for k in ("series_fred", "series_id", "name", "series_name", "entity_id")
           if isinstance(ent.get(k), str) and ent.get(k)}
    tgt = _tokens(target_name)
    for did in doc_ids:
        doc = corpus.docs.get(did)
        if doc is None:
            continue
        for t in parse_tables(did, doc.text):
            for j, name in enumerate(t.header):
                if j == 0:
                    continue
                h = name.strip().lower()
                if _level_column_for_relative_target(target_name, h):
                    continue
                ht = _tokens(h)
                exact = h in ids or any(h.startswith(i + " ") or h.startswith(i + " (") for i in ids if len(i) >= 4)
                own = did in entity_docs and bool(ht) and ht <= tgt | {"pct"} and len(ht & tgt) >= 1
                if not (exact or own):
                    continue
                dates = [r[0] for r in t.rows]
                if _CHANGE_TARGET.search(target_name) and _ratio_parts(h) is not None:
                    # a change of a ratio: measured on its numerator, in its starting denominator (see Series.denom)
                    parts = _ratio_columns(t, j)
                    if parts is None or len(parts[0]) < 6:
                        continue  # the table does not confirm the ratio: no baseline from it
                    num, den, scale = parts
                    return Series(did, name.strip(), num, dates, _period_days(dates), exact, den, scale)
                vals = [to_number(r[j]) for r in t.rows if j < len(r)]
                if len(vals) < 6 or any(v is None for v in vals):
                    continue
                return Series(did, name.strip(), [float(v) for v in vals], dates, _period_days(dates), exact)
    return _own_table_series(corpus, doc_ids, ent, target_name, entity_docs) if allow_own else None


def _is_change(series: Series, target_name: str) -> bool:
    """Whether the target is a change over the horizon of this series (else the series' next value is the target)."""
    if series.kind == "own-rel":
        return True  # v21.1: a percent change is the change of 100 x ln(level)
    return bool(_CHANGE_TARGET.search(target_name)) and not _same_quantity(target_name, series.header)


def _horizon_steps(series: Series, horizon_days: int | None) -> float:
    """Periods of the series in the horizon (trading days for a daily series)."""
    period = series.period_days or 30.0
    return (horizon_days or 30) * (5 / 7) if period <= 1.5 else (horizon_days or 30) / period


def _steps(series: Series) -> list[float]:
    """One-period changes of the series in target units (a ratio's numerator change over its starting denominator)."""
    v, d = series.values, series.denom
    if d is None:
        return [b - a for a, b in zip(v, v[1:])]
    return [(b - a) / d[i] * series.denom_scale for i, (a, b) in enumerate(zip(v, v[1:]))]


# v16: the 90% multiplier of a random-walk extrapolation from a short window, calibrated on public U.S. government
# data through 2021 (Treasury yields, FX and WTI; complete releases by early January 2022; see
# tools/calibrate_scales.py and ARTIFACT_PROVENANCE.md) for the scorer's own loss: the k minimising the mean interval
# score of the half-width k x rms(last n steps) x sqrt(h), each series weighted equally. That k solves
# sum(s over misses) = 0.1 x sum(s), a scale-weighted quantile of |change| / s. v15 used the unweighted 90% quantile
# (coverage), which overstates k when scales vary (n = 10, h = 40: 2.11 vs 1.89) and raised the interval score by up to
# 3.4% out of sample; this table is within -1.1% / +0.5% of v14's constant out of sample (time split 1990-2005 /
# 2006-2021) and lower at its edges.
_RW_N = (5, 10, 20, 40, 60)
_RW_H = (1, 5, 10, 20, 40, 60, 90, 120)
_RW_K = (
    (1.76, 1.83, 1.80, 1.85, 1.98, 2.03, 2.13, 2.20),
    (1.66, 1.74, 1.72, 1.76, 1.89, 1.96, 2.08, 2.12),
    (1.63, 1.69, 1.65, 1.72, 1.83, 1.93, 2.05, 2.08),
    (1.60, 1.66, 1.62, 1.67, 1.78, 1.89, 2.00, 2.06),
    (1.60, 1.65, 1.61, 1.66, 1.78, 1.88, 1.98, 2.02),
)


def rw_k(n_steps: int, h_steps: float) -> float:
    """Calibrated 90% multiplier for n observed steps and an h-step horizon: bilinear in log n and log h, held at the
    edges of the measured grid (n 5-60, h 1-120)."""
    def pos(grid, x):
        x = min(max(x, grid[0]), grid[-1])
        for i in range(len(grid) - 1):
            if x <= grid[i + 1]:
                a, b = math.log(grid[i]), math.log(grid[i + 1])
                return i, (math.log(x) - a) / (b - a)
        return len(grid) - 2, 1.0

    i, u = pos(_RW_N, max(1, n_steps))
    j, w = pos(_RW_H, max(1.0, h_steps))
    k = _RW_K
    return ((1 - u) * (1 - w) * k[i][j] + (1 - u) * w * k[i][j + 1] + u * (1 - w) * k[i + 1][j] + u * w * k[i + 1][j + 1])


def _inflation(n_errors: int, h_steps: int) -> float:
    """Small-sample inflation of a backtest's empirical 90% error quantile. v15 counted overlapping h-step backtest
    errors as n / h independent ones (a coverage argument); judged by the mean interval score on the same public daily
    series (windows of 60-250 days, horizons of 7-56) that rule was no better than counting all n (0.997-1.003 of it),
    so v16 keeps v14's rule. `h_steps` stays in the signature for the callers."""
    return 1.0 + 0.75 / math.sqrt(max(1, n_errors))


def series_half_width(series: Series, target_name: str, horizon_days: int | None, ent: dict,
                      *, policy: ArtifactPolicy | None = None) -> float | None:
    """90% half-width from the series' step volatility: one step for a next-value target, random-walk scaled by
    sqrt(periods in the horizon) for a change target; a level quoted in percent is converted for a basis-point
    target. v12: computed on the series the statistical baseline uses (the same columns, the same unit guards),
    where v11 chose its own column in tables.entity_half_width and could read EPS dollars for a percent-growth
    target (scale the dollars by 10 and the interval moved by the square root of 10)."""
    if policy is None or not policy.permits("statistics"):
        from .runtime_naive import series_half_width as runtime_half_width
        return runtime_half_width(series, target_name, horizon_days, ent)
    diffs = _steps(series)
    if not diffs:
        return None
    rms = math.sqrt(sum(x * x for x in diffs) / len(diffs))
    if rms <= 0:
        rms = _resolution(series.values)  # v21.1: a flat series still has its printed resolution
    sd, steps = rms, 1.0
    if (_is_change(series, target_name) or series.kind == "own") and horizon_days:
        # v21.1: a level read off the entity's own table several steps ahead (a daily price a week out) spreads like
        # a random walk too; v12-v21 named columns keep one step
        steps = max(1.0, _horizon_steps(series, horizon_days))
        sd = rms * math.sqrt(steps)
    bps = "bps" in target_name.lower() or str(ent.get("unit", "")).lower().startswith("bps")
    if bps and series.denom is None and max(abs(v) for v in series.values) < 30:
        sd *= 100.0
    return rw_k(len(diffs), steps) * sd  # v15: calibrated, was 1.645


CHANGE_SWITCH = 0.75  # v17: a change target leaves its default only for a method this much better (MAE ratio)


def _change_switch_ok(errs: dict[str, list[float]], best: str, default: str, ratio: float = CHANGE_SWITCH) -> bool:
    """v17: whether a change target may leave its default method for `best`. Its backtest errors are h-step changes
    that overlap, so a method can look better by chance: v16's rule (10% better over 8 origins) left zero change for 63%
    of 20-day changes of daily yields, FX and oil (60-day windows) and scored 0.47 against the zero-change rule's 0.50.
    Now the method must be 25% better and better in both halves of the backtest: 0.48 there, while a real trend is still
    found (monthly changes of trending macro levels: 0.569 against 0.576 for v16's rule and 0.500 for zero change;
    tools/calibrate_methods.py)."""
    a, b = errs.get(best) or [], errs.get(default) or []
    n = min(len(a), len(b))
    if best == default or n < MIN_SWITCH or sum(a) > ratio * sum(b):
        return False
    k = n // 2
    return sum(a[:k]) < sum(b[:k]) and sum(a[k:n]) < sum(b[k:n])


def _quantile(xs: list[float], q: float) -> float:
    s = sorted(xs)
    if not s:
        return float("nan")
    k = (len(s) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def _ewma(v: list[float], a: float = 0.5) -> float:
    m = v[0]
    for x in v[1:]:
        m = a * x + (1 - a) * m
    return m


def _ar1_phi(v: list[float]) -> float:
    """Share of the way from the last value to the window mean that an AR(1) around that mean predicts for the next
    value: 1 - the lag-1 autocorrelation, clipped to [0, 1] (0 for fewer than four values)."""
    if len(v) < 4:
        return 0.0
    m = sum(v) / len(v)
    den = sum((x - m) ** 2 for x in v)
    if den <= 0:
        return 0.0
    rho = sum((a - m) * (b - m) for a, b in zip(v, v[1:])) / den
    return max(0.0, min(1.0, 1.0 - rho))


_NEXT_METHODS = {
    "last": lambda v: v[-1],
    # v17: the next value of a stationary rate pulled towards the window mean by the window's own persistence
    "ar1": lambda v: v[-1] + _ar1_phi(v) * (sum(v) / len(v) - v[-1]),
    "mean3": lambda v: sum(v[-3:]) / len(v[-3:]),
    "mean6": lambda v: sum(v[-6:]) / len(v[-6:]),
    "median6": lambda v: statistics.median(v[-6:]),
    "ewma": _ewma,
    "drift": lambda v: v[-1] + (v[-1] - v[max(0, len(v) - 7)]) / max(1, min(6, len(v) - 1)),
}


def _change_methods(h: int):
    def hchanges(v):
        return [v[i + h] - v[i] for i in range(len(v) - h)]

    return {
        "zero": lambda v: 0.0,
        "drift": lambda v: (sum(hchanges(v)) / len(hchanges(v))) if hchanges(v) else 0.0,
        "momentum": lambda v: 0.5 * (v[-1] - v[-1 - h]) if len(v) > h else 0.0,
        "reversal": lambda v: -0.5 * (v[-1] - v[-1 - h]) if len(v) > h else 0.0,
    }


def forecast(series: Series, target_name: str, horizon_days: int | None, ent: dict) -> StatForecast:
    if _level_column_for_relative_target(target_name, series.header):
        raise ValueError(f"a level series ({series.header}) cannot forecast a relative change ({target_name})")
    v = series.values
    bps = "bps" in target_name.lower() or str(ent.get("unit", "")).lower().startswith("bps")
    scale = 100.0 if (bps and series.denom is None and max(abs(x) for x in v) < 30) else 1.0
    change_type = _is_change(series, target_name)
    if change_type and series.denom is not None:
        return _ratio_change_forecast(series, horizon_days)
    rate_series = not change_type and bool(_RATE_SERIES.search(f"{target_name} {ent.get('unit', '')}"))
    if not change_type:
        # v17: AR(1) is a rate series' method only; a level series keeps v16's candidates (on yield, FX and oil levels a
        # short window's AR(1), picked by the backtest now and then, lost 0.002 against the random walk)
        cands = {m: f for m, f in _NEXT_METHODS.items() if rate_series or m != "ar1"}
        errs: dict[str, list[float]] = {m: [] for m in cands}
        # v21.1: a level from the entity's own table is backtested at the horizon (a daily price a week out is five
        # steps ahead); named columns keep v12-v21's one step
        hl = max(1, int(round(_horizon_steps(series, horizon_days)))) if series.kind == "own" else 1
        for t in range(3, len(v) - hl + 1):
            for m, f in cands.items():
                errs[m].append(abs(f(v[:t]) - v[t + hl - 1]))
    else:
        period = series.period_days or 30.0
        periods = (horizon_days or 30) * (5 / 7) if period <= 1.5 else (horizon_days or 30) / period
        h = max(1, int(round(periods)))
        methods = _change_methods(h)
        errs = {m: [] for m in methods}
        for t in range(h + 2, len(v) - h):
            hist, actual = v[: t + 1], v[t + h] - v[t]
            for m, f in methods.items():
                errs[m].append(abs(f(hist) - actual))
        if len(errs["zero"]) < MIN_BACKTEST:
            diffs = [b - a for a, b in zip(v, v[1:])]
            rms = math.sqrt(sum(d * d for d in diffs) / len(diffs)) if diffs else 0.0
            rw = rw_k(len(diffs), h) * rms * math.sqrt(h)  # v15: calibrated, was 1.645 x 1.25
            return StatForecast("zero (random walk)", 0.0, None, rw * scale if rw else None, 0, v[-1], series)
    usable = {m: e for m, e in errs.items() if len(e) >= MIN_BACKTEST}
    # v17: a rate series (month-over-month change, return) defaults to its own AR(1) mean reversion. Replayed on public
    # data published by 2021 (tools/calibrate_methods.py), the 6-period median scored 0.516 / 0.584 / 0.562 against the
    # naive last value (scorer 5.2.2 predictive quality) on CPI components, twelve macro series and daily FX / oil
    # returns; AR(1) scored 0.525 / 0.592 / 0.576, and won in 5 of the 6 half-samples.
    # v19 (review of v17: those m/m replays read the revised 2021 vintage, and the daily cross-sections mixed dates):
    # replayed in real time -- the window as known the day before each release, the outcome its first print (ALFRED,
    # 2002-2021) -- the median scored 0.508 / 0.592 and AR(1) 0.524 / 0.597; on daily returns grouped by date 0.562 and
    # 0.578. AR(1) won in all six half-samples.
    default = "zero" if change_type else "ar1" if rate_series else "last"
    if not usable:
        f0 = 0.0 if change_type else _NEXT_METHODS[default](v)
        return StatForecast(default, f0 * scale, None, None, 0, v[-1], series)
    # Parsimony: with a handful of backtest origins, choosing among several methods overfits, so
    # the default stands unless another method beats it by at least 10% over enough origins.
    maes = {m: sum(e) / len(e) for m, e in usable.items()}
    best = min(maes, key=lambda m: (maes[m], m))
    if change_type:
        if default in maes and not _change_switch_ok(usable, best, default):
            best = default
    elif default in maes and (maes[best] > 0.9 * maes[default] or len(usable[default]) < MIN_SWITCH):
        best = default
    e = usable[best]
    mae = sum(e) / len(e)
    # small-sample inflation of the empirical quantile: the interval score charges width, so it is mild
    small = _inflation(len(e), h if change_type else hl)
    q90 = _quantile(e, 0.9) * small
    if q90 <= 0:
        q90 = _resolution(v) * scale  # v21.1: a series flat over its window
    method_fn = _NEXT_METHODS[best] if not change_type else _change_methods(h)[best]
    return StatForecast(best, method_fn(v) * scale, mae * scale, q90 * scale, len(e), v[-1], series)


LEVEL_REVERT_DAILY = 0.003  # v19: per calendar day; the share of the gap closed over h days is 1 - exp(-0.003 h)
RATIO_SWITCH = 0.5  # v19: a ratio change leaves reversion only for a drift that halves its backtest error (see below)


def revert_share(horizon_days: int | None) -> float:
    """Share of a ratio's gap to its window mean that the change over the horizon closes (v19). Fitted on CFTC
    positioning of ten futures markets, 2011-2021, at horizons of 1-26 weeks (tools/calibrate_methods.py): regression
    quality against zero change, averaged over the horizons, is flat for daily rates 0.003-0.004 (0.5028-0.5030; 0.5000
    at no reversion); the lower end is kept. v17 closed a fixed 0.15 whatever the horizon."""
    return 1.0 - math.exp(-LEVEL_REVERT_DAILY * max(1, horizon_days or 30))


def _ratio_change_forecast(series: Series, horizon_days: int | None) -> StatForecast:
    """The change-over-horizon methods on a ratio's numerator, each forecast measured in the denominator at the
    forecast origin: the target is (X[t+h] - X[t]) / Y[t] * scale.

    v17: the default is reversion of the ratio towards its window mean -- ranking ten futures markets by it scored
    0.567 at five weeks on CFTC 2011-2021 (zero change carries no ordering; v16's switching rule ranked them at 0.500).
    v19: the share closed grows with the horizon (revert_share), and a clear trend is followed: the in-window drift
    replaces the reversion when it halves the backtest error and is better in both halves -- v17 never switched and
    forecast -1.9 for a steady +5 trend at every horizon. Only drift may replace it, and only on that stricter evidence:
    on the CFTC replay the change rule's 25% (CHANGE_SWITCH) took drift for 4% of markets at five weeks and 10% at eight,
    and ranked worse by 0.006 and 0.024; at 50% it switches for 1% and 4% (cost 0.002 and 0.007, about one standard
    error), while still following a +1/week trend under weekly noise of sd 0.5 (99% of windows) or 1.0 (79%)."""
    v, d, k = series.values, series.denom, series.denom_scale
    period = series.period_days or 30.0
    periods = (horizon_days or 30) * (5 / 7) if period <= 1.5 else (horizon_days or 30) / period
    h = max(1, int(round(periods)))
    # the span of the change itself: h reports of a weekly series (the unit's horizon counts from its cutoff, which can
    # fall days after the last report: 32 days for the 35 from one report to the target one)
    share = revert_share(horizon_days if period <= 1.5 else h * period)
    methods = _change_methods(h)

    def level_revert(vh: list[float]) -> float:  # a numerator change, like the other methods
        dh = d[:len(vh)]
        ratios = [a / b for a, b in zip(vh, dh) if b]
        if len(ratios) < 4 or not dh[-1]:
            return 0.0
        return -share * (ratios[-1] - sum(ratios) / len(ratios)) * dh[-1]

    methods["level_revert"] = level_revert
    default = "level_revert"
    errs: dict[str, list[float]] = {m: [] for m in methods}
    for t in range(h + 2, len(v) - h):
        actual = (v[t + h] - v[t]) / d[t] * k
        for m, f in methods.items():
            errs[m].append(abs(f(v[: t + 1]) / d[t] * k - actual))
    if len(errs["zero"]) < MIN_BACKTEST:
        steps = _steps(series)
        rms = math.sqrt(sum(x * x for x in steps) / len(steps)) if steps else 0.0
        rw = rw_k(len(steps), h) * rms * math.sqrt(h)  # v15/v16: calibrated
        return StatForecast("level reversion (short history)", level_revert(v) / d[-1] * k if d[-1] else 0.0, None,
                            rw or None, 0, v[-1], series)
    usable = {m: e for m, e in errs.items() if len(e) >= MIN_BACKTEST}
    best = "drift" if _change_switch_ok(usable, "drift", default, RATIO_SWITCH) else default
    e = usable[best]
    q90 = _quantile(e, 0.9) * _inflation(len(e), h)
    return StatForecast(best, methods[best](v) / d[-1] * k, sum(e) / len(e), q90, len(e), v[-1], series)


def entity_forecast(corpus: Corpus, doc_ids: list[str], ent: dict, target_name: str,
                    horizon_days: int | None, entity_docs: set[str], allow_own: bool = False,
                    *, policy: ArtifactPolicy | None = None) -> StatForecast | None:
    if policy is None or not policy.permits("statistics"):
        from .runtime_naive import entity_forecast as runtime_forecast
        return runtime_forecast(corpus, doc_ids, ent, target_name, horizon_days, entity_docs, allow_own)
    s = find_series(corpus, doc_ids, ent, target_name, entity_docs, allow_own)
    if s is None:
        return None
    try:
        return forecast(s, target_name, horizon_days, ent)
    except (ValueError, ZeroDivisionError, IndexError):
        return None
