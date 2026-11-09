"""Features computed by code from the unit's own tables and filings, so the model reads numbers instead of
deriving them.

Every line here is pre-cutoff by construction (it is computed from the frozen corpus and the entity table)
and is written into the analyst prompt next to the entity's evidence. A feature that cannot be computed
unambiguously is left out rather than guessed.

  * series features: where the entity's latest value sits in its own history, recent changes, and the
    same-calendar-month pattern when there are at least two years of monthly data;
  * co-movement: slope and correlation against peer series of the same table (other entities' columns),
    e.g. how many points an index moves per point of one of its components;
  * driver: a higher-frequency series in the corpus that shares the entity's name (weekly prices for a
    monthly index), averaged over the target month, with the published-minus-driver gap by month so a
    seasonally adjusted target is not forecast from an unadjusted driver's raw change;
  * cross-section: the entity's rank on each numeric column of the entity table;
  * going concern: whether a filing concludes substantial doubt, says the doubt was alleviated by
    management's plans, or only uses procedural or conditional going-concern language.
"""
from __future__ import annotations

import datetime as dt
import math
import re
import statistics

from .corpus import Corpus
from .tables import parse_tables, to_number
from .task import TaskSpec

_MONTH = re.compile(r"^(\d{4})-(\d{2})$")
_DAY = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")
_WORD = re.compile(r"[A-Za-z]{5,}")
_GENERIC = {"index", "price", "prices", "total", "items", "level", "rate", "rates", "change", "percent", "series",
            "weekly", "monthly", "annual", "united", "states", "retail", "average", "consumer", "other", "value"}


def _g(x: float) -> str:
    return f"{x:.4g}"


# ---------------------------------------------------------------------------------------------- #
# Series features                                                                                 #
# ---------------------------------------------------------------------------------------------- #
def _changes_like(header: str, values: list[float]) -> bool:
    """A series of period changes (it takes both signs, or its name says so) rather than a level."""
    if re.search(r"change|chg|pct|%|mom|yoy|return|growth", header, re.I):
        return True
    return min(values) < 0 < max(values)


def series_line(series, target_month: int | None = None) -> str | None:
    v, d = list(series.values), list(series.dates)
    n = len(v)
    if n < 6:
        return None
    last = v[-1]
    parts = [f"last {_g(last)} ({d[-1]})", f"change vs previous {v[-1] - v[-2]:+.4g}"]
    if n >= 4:
        parts.append(f"vs 3 periods earlier {v[-1] - v[-4]:+.4g}")
    if n >= 13:
        parts.append(f"vs 12 periods earlier {v[-1] - v[-13]:+.4g}")
    below = sum(1 for x in v if x < last) + 0.5 * (sum(1 for x in v if x == last) - 1)
    pct = 100.0 * below / max(1, n - 1)
    sd = statistics.pstdev(v)
    z = (last - statistics.fmean(v)) / sd if sd > 0 else 0.0
    text = (f"SERIES FEATURES (computed by code from '{series.header}' in {series.doc_id}): " + ", ".join(parts) +
            f"; the last value is at percentile {pct:.0f} of its {n}-period history (z {z:+.1f})")
    months = [_MONTH.match(x) for x in d]
    if target_month and all(months) and n >= 24:
        changes = v if _changes_like(series.header, v) else [None] + [b - a for a, b in zip(v, v[1:])]
        same = [c for m, c in zip(months, changes) if c is not None and int(m.group(2)) == target_month]
        if len(same) >= 2:
            text += (f"; in the same calendar month of earlier years the {'value' if changes is v else 'change'} "
                     f"averaged {statistics.fmean(same):+.4g} ({len(same)} years)")
    return text + "."


# ---------------------------------------------------------------------------------------------- #
# Co-movement with peer series                                                                   #
# ---------------------------------------------------------------------------------------------- #
def _ols(x: list[float], y: list[float]) -> tuple[float, float] | None:
    """(slope of y on x, correlation), or None when either side is flat."""
    mx, my = statistics.fmean(x), statistics.fmean(y)
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    if sxx <= 0 or syy <= 0:
        return None
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    return sxy / sxx, sxy / math.sqrt(sxx * syy)


def comovement_line(corpus: Corpus, series, peer_headers: set[str]) -> str | None:
    """How the entity's series moves with peer series (other entities' columns) of the same table."""
    doc = corpus.docs.get(series.doc_id)
    if doc is None:
        return None
    for t in parse_tables(series.doc_id, doc.text):
        if series.header not in t.header:
            continue
        j = t.header.index(series.header)
        rels = []
        for k, name in enumerate(t.header):
            if k in (0, j) or name not in peer_headers:
                continue
            pairs = [(to_number(r[k]), to_number(r[j])) for r in t.rows if max(j, k) < len(r)]
            pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
            if len(pairs) < 6:
                continue
            fit = _ols([a for a, _ in pairs], [b for _, b in pairs])
            if fit and abs(fit[1]) >= 0.85:  # over a handful of periods, weaker correlations arise by chance
                rels.append((abs(fit[1]), name, fit[0], fit[1], len(pairs)))
        if rels:
            rels.sort(reverse=True)
            shown = "; ".join(f"'{name}': slope {b:+.2f} per unit, correlation {r:+.2f} over {m} periods"
                              for _, name, b, r, m in rels[:1])
            return f"CO-MOVEMENT (computed by code from {series.doc_id}): '{series.header}' moves with {shown}."
    return None


# ---------------------------------------------------------------------------------------------- #
# Higher-frequency driver for a monthly target                                                    #
# ---------------------------------------------------------------------------------------------- #
def _name_words(*texts: str) -> set[str]:
    return {w.lower() for t in texts for w in _WORD.findall(t or "")} - _GENERIC


def _target_month(ent: dict) -> tuple[int, int] | None:
    for k, v in ent.items():
        if isinstance(v, str) and "month" in k.lower() and "latest" not in k.lower():
            m = _MONTH.match(v.strip())
            if m:
                return int(m.group(1)), int(m.group(2))
    return None


def driver_line(spec: TaskSpec, corpus: Corpus, ent: dict, series) -> str | None:
    """A weekly or daily series sharing the entity's name, averaged over the target month, against the
    entity's monthly seasonally adjusted series."""
    sa = re.search(r"seasonally\s+adjusted", spec.prompt, re.I) or re.search(r"(?:^|_)sa(?:$|_)", str(ent.get("unit", "")), re.I)
    tm = _target_month(ent)
    if not sa or tm is None or not all(_MONTH.match(x) for x in series.dates):
        return None
    own = _name_words(str(ent.get("name", "")), series.header)
    if not own:
        return None
    published = {x: y for x, y in zip(series.dates, series.values)}
    for did in sorted(corpus.docs):
        if did == series.doc_id:
            continue
        doc = corpus.docs[did]
        title_words = _name_words(doc.title)
        for t in parse_tables(did, doc.text):
            rows = [(r[0].strip(), r[1]) for r in t.rows if len(r) >= 2 and _DAY.match(r[0].strip())]
            if len(rows) < 8 or not (own & (title_words | _name_words(" ".join(t.header)))):
                continue
            dates = [dt.date.fromisoformat(a) for a, _ in rows]
            if statistics.median((b - a).days for a, b in zip(dates, dates[1:])) > 8:
                continue  # not a weekly or daily series
            by_month: dict[str, list[float]] = {}
            for (a, raw), day in zip(rows, dates):
                x = to_number(raw)
                if x is not None:
                    by_month.setdefault(f"{day.year}-{day.month:02d}", []).append(x)
            target = f"{tm[0]}-{tm[1]:02d}"
            prev_y, prev_m = (tm[0], tm[1] - 1) if tm[1] > 1 else (tm[0] - 1, 12)
            prev = f"{prev_y}-{prev_m:02d}"
            if len(by_month.get(target, [])) < 2 or prev not in by_month:
                continue
            avg = {k: statistics.fmean(vs) for k, vs in by_month.items()}
            keys = sorted(avg)
            change = {b: (avg[b] / avg[a] - 1) * 100 for a, b in zip(keys, keys[1:]) if avg[a]}
            gaps = {k: published[k] - change[k] for k in keys if k in published and k in change}
            if len(gaps) < 3:
                continue
            name = t.header[1] if len(t.header) > 1 else did
            gap_text = ", ".join(f"{k[5:]} {g:+.1f}" for k, g in sorted(gaps.items()))
            text = (f"DRIVER (computed by code from '{name}' in {did}): averaged {avg[target]:.4g} over "
                    f"{len(by_month[target])} observations in {target} vs {avg[prev]:.4g} in {prev} "
                    f"({change[target]:+.1f}%, not seasonally adjusted). In the overlapping months the published "
                    f"seasonally adjusted change minus the driver's change was ({target[:4]}): {gap_text}. The gap "
                    f"varies with the calendar month (seasonal factors): estimate {target}'s seasonal effect "
                    f"before applying the driver's raw change.")
            last_year = f"{tm[0] - 1}-{tm[1]:02d}"
            if last_year in gaps:
                text += f" The same month a year earlier had a gap of {gaps[last_year]:+.1f}."
            return text
    return None


# ---------------------------------------------------------------------------------------------- #
# Cross-section of the entity table                                                               #
# ---------------------------------------------------------------------------------------------- #
def cross_section_lines(spec: TaskSpec) -> dict[str, str]:
    """Each entity's rank on every numeric column of the entity table (when the rows share one unit)."""
    ents = spec.entities
    unit_keys = [k for k in (ents[0].keys() if ents else []) if re.fullmatch(r"units?|currency", k, re.I)]
    if any(len({str(e.get(k)) for e in ents}) > 1 for k in unit_keys) or len(ents) < 3:
        return {}
    cols = []
    for k in ents[0].keys():
        if re.search(r"cik|id$|_id|code", k, re.I):
            continue
        vals = [e.get(k) for e in ents]
        if all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in vals) and len(set(vals)) > 1:
            cols.append(k)
    out: dict[str, str] = {}
    for e in ents:
        parts = []
        for k in cols[:6]:
            vals = sorted((float(x[k]) for x in ents), reverse=True)
            rank = vals.index(float(e[k])) + 1
            parts.append(f"{k} {e[k]:g} (rank {rank} of {len(ents)}, median {statistics.median(vals):.4g})")
        if parts:
            out[e["entity_id"]] = "CROSS-SECTION (computed by code from the entity table): " + "; ".join(parts) + "."
    return out


# ---------------------------------------------------------------------------------------------- #
# Going-concern language                                                                          #
# ---------------------------------------------------------------------------------------------- #
_DOUBT = re.compile(r"substantial\s+doubt", re.I)
_RELIEVED = re.compile(r"alleviat|mitigat", re.I)
_CONDITIONAL = re.compile(r"\b(?:whether|could|may|might|would|if|unless|evaluate[sd]?|must\s+evaluate)\b", re.I)
_CONCLUDED = re.compile(r"concluded|there\s+is\s+substantial|substantial\s+doubt\s+(?:exists|remains)|raises?\s+substantial", re.I)


def _sentence(text: str, pos: int) -> tuple[int, int]:
    s = max(text.rfind(". ", 0, pos), text.rfind("\n", 0, pos))
    e = text.find(". ", pos)
    return (s + 2 if s >= 0 else 0), (e + 1 if e >= 0 else len(text))


def going_concern_line(corpus: Corpus, docs: set[str]) -> str | None:
    """Whether the entity's own filings conclude substantial doubt about continuing as a going concern."""
    found = {"concluded": None, "alleviated": None, "procedural": None}
    for did in sorted(docs):
        doc = corpus.docs.get(did)
        if doc is None or len(doc.text) < 20000:
            continue
        t = doc.text
        for m in _DOUBT.finditer(t):
            s, e = _sentence(t, m.start())
            sent = t[s:e]
            after = t[e:e + 500]
            if _RELIEVED.search(sent) or (re.search(r"initially", sent, re.I) and _RELIEVED.search(after)):
                kind = "alleviated"
            elif _CONDITIONAL.search(sent):
                kind = "procedural"
            elif _CONCLUDED.search(sent):
                kind = "concluded"
            else:
                kind = "procedural"
            if found[kind] is None:
                found[kind] = (did, " ".join(sent.split())[:220])
    if found["concluded"]:
        did, quote = found["concluded"]
        return (f"GOING CONCERN (read by code from {did}): management concludes that substantial doubt exists about "
                f"continuing as a going concern: \"{quote}\"")
    if found["alleviated"]:
        did, quote = found["alleviated"]
        return (f"GOING CONCERN (read by code from {did}): substantial doubt was raised, and management states that "
                f"its plans alleviate it (plans that must still succeed): \"{quote}\"")
    if found["procedural"]:
        return ("GOING CONCERN (read by code): the filings mention going concern only in procedural or conditional "
                "language; no substantial doubt is concluded.")
    return None


# ---------------------------------------------------------------------------------------------- #
# All features for a unit                                                                         #
# ---------------------------------------------------------------------------------------------- #
def entity_features(spec: TaskSpec, corpus: Corpus, specific: dict[str, list[str]],
                    stat: dict[str, object]) -> dict[str, list[str]]:
    """{entity id: feature lines}, every computation guarded: a failure drops the line, never the unit."""
    peers = {getattr(s, "series").header for s in stat.values() if getattr(s, "series", None) is not None}
    cross = {}
    try:
        cross = cross_section_lines(spec)
    except (ValueError, TypeError, KeyError):
        pass
    out: dict[str, list[str]] = {}
    for ent in spec.entities:
        eid = ent["entity_id"]
        lines: list[str] = []
        series = getattr(stat.get(eid), "series", None)
        tm = _target_month(ent)
        for make in (
            (lambda: series_line(series, tm[1] if tm else None)) if series is not None else None,
            (lambda: comovement_line(corpus, series, peers - {series.header})) if series is not None else None,
            (lambda: driver_line(spec, corpus, ent, series)) if series is not None else None,
            lambda: cross.get(eid),
            lambda: going_concern_line(corpus, set(specific.get(eid) or [])),
        ):
            if make is None:
                continue
            try:
                line = make()
            except (ValueError, TypeError, KeyError, IndexError, ZeroDivisionError, statistics.StatisticsError):
                line = None
            if line:
                lines.append(line)
        out[eid] = lines
    return out
