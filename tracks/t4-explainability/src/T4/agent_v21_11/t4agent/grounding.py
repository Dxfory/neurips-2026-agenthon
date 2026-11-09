"""Plan grounding: the planner's free-text notes are checked against the task and the corpus before use.

The planner writes before any analyst reads the evidence, and its notes (key drivers, label cues, the naive
baseline) go into every analyst, re-check and review prompt. A note written from memory can contradict the
corpus -- in a development run it described the September 2024 FOMC meeting as holding the policy rate at
5.25-5.50% when the statement in the corpus announces a cut to 4-3/4 to 5 percent -- or describe how an
event after the cutoff turned out ("outcome of the 2024-11-05 election" for a unit cut off on 2024-10-31).

Two exact checks, no model involved:
  * every figure a note states must appear in the task, the entity table or the corpus;
  * a note may name a date after the cutoff only when the task or the corpus names that date too (a scheduled
    meeting, an election day) and the note does not speak of how it turned out.
A note that fails either is dropped. The rest are still only hints, and the analyst prompt says so.
"""
from __future__ import annotations

import datetime as dt
import re

from .corpus import Corpus
from .task import TaskSpec

_MONTHS = ("january|february|march|april|may|june|july|august|september|october|november|december|"
           "jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec")
_MONTH_NUM = {name: i for i, names in enumerate(
    (("jan", "january"), ("feb", "february"), ("mar", "march"), ("apr", "april"), ("may",), ("jun", "june"),
     ("jul", "july"), ("aug", "august"), ("sep", "sept", "september"), ("oct", "october"), ("nov", "november"),
     ("dec", "december")), start=1) for name in names}
# ISO dates (2024-11-05, 2024/11/05), month-day-year (November 5, 2024), month-year (November 2024), year-month
# (2024-11). Each is read as the earliest day it can mean, so a month that starts after the cutoff is after it.
_DATE_PATTERNS = (
    (re.compile(r"\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})\b"), lambda m: (m[1], m[2], m[3])),
    (re.compile(rf"\b({_MONTHS})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})\b", re.I), lambda m: (m[3], m[1], m[2])),
    (re.compile(rf"\b({_MONTHS})\.?\s+(\d{{4}})\b", re.I), lambda m: (m[2], m[1], "1")),
    (re.compile(r"\b(\d{4})-(\d{2})\b(?![-/]\d)"), lambda m: (m[1], m[2], "1")),
)
_NUMBER = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?|\.\d+")
# models write non-breaking hyphens, en dashes and fraction slashes inside dates and ranges ("2024‑11‑05")
_ASCII = str.maketrans({**{c: "-" for c in "‐‑‒–—―−﹘﹣－"},
                        **{c: "/" for c in "⁄∕／"}})
# a note about a dated event that speaks of its result describes what happened after the cutoff
_OUTCOME = re.compile(r"\b(?:outcomes?|results?|won|wins?|winners?|victory|defeat|aftermath|reaction to|"
                      r"response to|after the)\b", re.I)


def _norm(tok: str) -> str:
    t = tok.replace(",", "")
    if "." in t:
        t = t.rstrip("0").rstrip(".")
    t = t.lstrip("0")
    return t if t and t != "." else "0"


def _dates(text: str) -> tuple[list[dt.date], str]:
    """Dates named in `text`, and the text with them blanked out (their digits are not figures)."""
    text = text.translate(_ASCII)
    found: list[dt.date] = []
    for rx, parts in _DATE_PATTERNS:
        def take(m: re.Match) -> str:
            y, mo, d = parts(m)
            month = int(mo) if str(mo).isdigit() else _MONTH_NUM.get(str(mo).lower().rstrip("."))
            try:
                found.append(dt.date(int(y), int(month), int(d)))
            except (TypeError, ValueError):
                return m.group(0)
            return " "
        text = rx.sub(take, text)
    return found, text


def figures(text: str) -> set[str]:
    """Normalised numbers stated in `text`, dates excluded ('5.50' and '5.5' are one figure)."""
    _, rest = _dates(text)  # (dashes already made ASCII)
    return {_norm(t) for t in _NUMBER.findall(rest)}


def grounded_figures(spec: TaskSpec, corpus: Corpus) -> set[str]:
    """Every figure the planner may legitimately state: any number in the task text, the entity table or the
    corpus (the parts of a date included: a note may quote 'the 18th' of a dated statement)."""
    parts = [spec.prompt, " ".join(str(v) for e in spec.entities for v in e.values())]
    parts += [d.text for d in corpus.docs.values()]
    return {_norm(t) for p in parts for t in _NUMBER.findall(p)}


def problems(text: str, ground: set[str], cutoff: dt.date | None, known_dates: set[dt.date] = frozenset()) -> list[str]:
    """Why a note cannot be passed on: figures found in neither the task nor the corpus; a date after the cutoff
    that the evidence does not name, or that the note speaks of the result of."""
    out = []
    found, _ = _dates(text)
    if cutoff is not None:
        for d in found:
            if d > cutoff and (d not in known_dates or _OUTCOME.search(text)):
                out.append(f"date {d.isoformat()} after the cutoff")
    out += [f"figure {f} not in the task or corpus" for f in sorted(figures(text) - ground)]
    return out


def sanitize_plan(plan: dict, spec: TaskSpec, corpus: Corpus) -> list[str]:
    """Drop the planner's notes that state an ungrounded figure or a date after the cutoff, in place.
    Returns a description of each dropped note (for the log)."""
    ground = grounded_figures(spec, corpus)
    cutoff = spec.cutoff
    known = {d for p in [spec.prompt, *(doc.text for doc in corpus.docs.values())] for d in _dates(p)[0]}
    dropped: list[str] = []

    def keep(text: object, field: str) -> bool:
        if not isinstance(text, str):
            return False
        why = problems(text, ground, cutoff, known)
        if why:
            dropped.append(f"{field}: {text[:120]!r} ({'; '.join(why)})")
        return not why

    drivers = plan.get("key_drivers")
    if isinstance(drivers, list):
        plan["key_drivers"] = [d for d in drivers if keep(d, "key_drivers")]
    cues = plan.get("label_cues")
    if isinstance(cues, dict):
        plan["label_cues"] = {lab: [c for c in v if keep(c, f"label_cues[{lab}]")] if isinstance(v, list) else v
                              for lab, v in cues.items()}
    nb = plan.get("naive_baseline")
    if isinstance(nb, str) and not keep(nb, "naive_baseline"):
        plan.pop("naive_baseline", None)
    return dropped
