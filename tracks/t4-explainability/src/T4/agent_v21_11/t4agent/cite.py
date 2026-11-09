"""Claims for scorer 5.2.x: every claim is a verbatim quote of the span it cites.

The scorer reads each claim as a hypothesis. A claim is false when it states a figure its cited span does not
carry, cites a document the manifest does not label for its entity (nor marks shared), cites a span over
8,000 characters, is content-free filler, or is malformed (over 4,000 characters or 400 judge tokens). A
claim that is word for word a piece of its span passes the figure check whole and is never put to the
judge, so a quote of evidence the analysts relied on is both faithful and safe. Claims earn nothing beyond
that: evidence is credited through `submitted_reasons`, so a few good quotes per entity are enough.
"""
from __future__ import annotations

import re

from .config import CLAIM_MAX_CHARS, MAX_CLAIMS_PER_ENTITY
from .corpus import Corpus
from .predict import Evidence, Forecast
from .retrieval import entity_query
from .task import TaskSpec

_WORD = re.compile(r"[A-Za-z]{4,}")
_BULLET = re.compile(r"\s*[-*•]\s+\S")
_BANNER = re.compile(r"^[A-Z][A-Z0-9]+(?:[ \t]+[A-Z][A-Z0-9]+){0,8}[ \t]+[—–-][ \t]+\S")
_MONTH = re.compile(r"^\d{4}-\d{2}$")
_DAY = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_TENOR_PART = re.compile(r"^(\d+)Y$", re.I)
_TENOR_PREFIX = re.compile(r"^([A-Za-z]+?)(\d+)Y$", re.I)
_TENOR_PHRASE = re.compile(r"\d+[ \t-]*Year", re.I)
_PAREN = re.compile(r"\(([^)]*)\)")
_SUFFIX = re.compile(
    r"(?:,|\s)+(?:inc|incorporated|corp|corporation|co|company|companies|ltd|plc|llc|l\.p|lp|group|futures)\.?\s*$",
    re.I,
)
# Name words that do not identify one entity. The full name phrase is still kept.
_NAME_STOP = frozenset({
    "a", "an", "the", "of", "and", "or", "for", "to", "in", "on", "by", "at", "from", "with", "its",
    "inc", "incorporated", "corp", "corporation", "co", "company", "companies", "group", "ltd", "plc", "llc",
    "futures", "future", "auction", "auctions", "note", "notes", "bond", "bonds", "week", "index", "indexes",
    "indices", "total", "all", "new", "u", "s", "us", "treasury", "constant", "maturity", "yield", "yields",
    "percent", "percentage", "rate", "rates", "market", "markets", "bank", "banks", "america", "financial",
    "financials", "international", "services", "service", "care", "items", "less", "types", "type", "physical",
    "delivery", "close", "level", "value", "values", "estimate", "estimates", "first", "print", "change",
    "changes", "price", "prices", "report", "reports", "united", "states",
})
# Two-letter fragments that are ordinary words, not series codes.
_SHORT_DENY = frozenset({
    "we", "us", "of", "in", "on", "at", "to", "or", "an", "as", "be", "by", "do", "go", "if", "is", "it",
    "me", "my", "no", "so", "up", "am", "he", "re", "un", "ex", "ok",
})
_ABBR = frozenset({
    "u.s", "mr", "mrs", "ms", "dr", "inc", "ltd", "corp", "co", "jr", "sr", "st", "no", "vs", "fig", "eq",
    "etc", "eg", "ie", "cf", "al", "vol", "pp", "approx", "jan", "feb", "mar", "apr", "jun", "jul", "aug",
    "sep", "sept", "oct", "nov", "dec",
})
_CLOSERS = "\"')]}”’"
_CODE_KEYS = frozenset({"tenor", "series_id", "series_fred"})
_NAME_KEYS = frozenset({"name", "series_name"})


def quote_span(text: str, start: int, end: int, max_chars: int = CLAIM_MAX_CHARS) -> tuple[int, int] | None:
    """[s, e) inside [start, end) whose text is a self-contained quote of at most `max_chars` characters:
    cut at a line or sentence end when the span is longer, with surrounding whitespace trimmed."""
    s, e = max(0, start), min(len(text), end)
    while s < e and text[s].isspace():
        s += 1
    if e - s > max_chars:
        seg = text[s : s + max_chars]
        cut = max(seg.rfind("\n"), seg.rfind(". ") + 1, seg.rfind("; ") + 1)
        if cut < max_chars // 3:
            cut = seg.rfind(" ")
        e = s + (cut if cut > 0 else max_chars)
    while e > s and text[e - 1].isspace():
        e -= 1
    return (s, e) if e > s else None


def unit_names(spec: TaskSpec) -> list[str]:
    """The names the scorer blanks before its content-free test: every row's name, entity_id and *ticker strings
    (scoring.unit_entity_names), longest first."""
    names = {str(v).strip() for e in spec.entities for k, v in e.items()
             if isinstance(v, str) and (k in ("name", "entity_id") or k.endswith("ticker")) and str(v).strip()}
    return sorted(names, key=len, reverse=True)


def contentful(quote: str, names: list[str] | tuple[str, ...] = ()) -> bool:
    """A quote that asserts something: a figure, or several real words (filler is false from scorer 5.2.2). v21.1: the
    unit's own entity names are blanked first, as the scorer does: a document's title line that is just the entity's
    name ('DCOILWTICO - Crude Oil Prices: West Texas Intermediate (WTI) - Cushing, Oklahoma') is content-free."""
    for n in names:
        quote = re.sub(r"(?<![\w])" + re.escape(n) + r"(?![\w])", " ", quote, flags=re.IGNORECASE)
    return bool(re.search(r"\d", quote)) or len(_WORD.findall(quote)) >= 5


def is_header(text: str) -> bool:
    """A section title rather than a sentence or table row: it ends with ':', is a short line with no digit,
    is an all-caps title, or is an all-caps banner ('VINTAGE TABLE — ...') with no digit."""
    s = " ".join(text.split())
    if not s:
        return True
    if s.endswith(":"):
        return True
    letters = [c for c in s if c.isalpha()]
    if letters and len(s) <= 140 and all(c.isupper() for c in letters):
        return True
    if any(c.isdigit() for c in s):
        return False
    if len(s) <= 40:
        return True
    return _BANNER.match(s) is not None


def entity_aliases(spec: TaskSpec) -> dict[str, tuple[str, ...]]:
    """Phrases that name one entity and not the whole roster: id fragments, tenor words ('10-Year'),
    series codes, the name, and a reference month ('2024-07') when rows do not all share it."""
    raw: dict[str, set[str]] = {}
    for ent in spec.entities:
        eid = ent.get("entity_id")
        if not isinstance(eid, str):
            continue
        found: set[str] = set()
        _add_id_fragments(found, eid)
        for key, value in ent.items():
            if isinstance(value, str):
                if key in _CODE_KEYS or key.endswith("ticker"):
                    _add_alias(found, value.strip())
                elif key in _NAME_KEYS:
                    _add_name(found, value)
                elif key == "ref_month" and _MONTH.fullmatch(value.strip()):
                    _add_alias(found, value.strip())
                elif key == "auction_date" and _DAY.fullmatch(value.strip()):
                    _add_alias(found, value.strip())
        raw[eid] = found
    if not raw:
        return {}
    shared = set.intersection(*(set(a.casefold() for a in als) for als in raw.values())) if len(raw) > 1 else set()
    out: dict[str, tuple[str, ...]] = {}
    for eid, als in raw.items():
        kept = [a for a in als if a.casefold() not in shared]
        out[eid] = tuple(sorted(set(kept), key=lambda a: (-len(a), a.casefold())))
    return out


def claim_score(text: str, own: tuple[str, ...] | list[str], other: tuple[str, ...] | list[str]) -> int:
    """3*own + has_number - 4*foreign - 3*header. Foreign: another entity's alias is present and this entity's is not."""
    owned = _has_alias(text, own)
    foreign = not owned and _has_alias(text, other)
    number = any(c.isdigit() for c in text)
    return 3 * int(owned) + int(number) - 4 * int(foreign) - 3 * int(is_header(text))


def quote_units(text: str) -> list[tuple[int, int]]:
    """Full sentences and table rows as [start, end) in `text`. A wrapped prose line stays one sentence.
    A table row (two or more '|') or a bullet is one row. The span is a verbatim slice, trimmed of outer whitespace."""
    n = len(text)
    spans: list[tuple[int, int]] = []
    i = 0
    while i < n:
        while i < n and text[i] == "\n":
            i += 1
        if i >= n:
            break
        line_end = text.find("\n", i)
        if line_end < 0:
            line_end = n
        line = text[i:line_end]
        if _is_table_line(line) or _BULLET.match(line):
            s, e = _trim_bounds(text, i, line_end)
            if e > s:
                spans.append((s, e))
            i = line_end + 1 if line_end < n else n
            continue
        para_end = line_end
        j = line_end + 1 if line_end < n else n
        while j < n and text[j] != "\n":
            nxt = text.find("\n", j)
            if nxt < 0:
                nxt = n
            nxt_line = text[j:nxt]
            if _is_table_line(nxt_line) or _BULLET.match(nxt_line):
                break
            para_end = nxt
            j = nxt + 1 if nxt < n else n
        spans.extend(_sentences(text, i, para_end))
        i = para_end + 1 if para_end < n else n
    return spans


def build_citations(spec: TaskSpec, corpus: Corpus, ev: Evidence, fc: Forecast, plan_terms: list[str]) -> list[dict]:
    """Up to MAX_CLAIMS_PER_ENTITY verbatim sentences or table rows about this entity.

    Each candidate passage (analyst support, code facts, then retrieved chunks, then the entity's admitted
    documents) is split into full sentences and table rows. A row scores 3*own + has_number - 4*foreign - 3*header.
    Headers are dropped. A score of 0 or less is skipped for the next passage. When any remaining row names this
    entity, only those are kept (a shared document that names no tenor falls through to its best non-foreign row).
    Equal scores prefer the row that names fewer other entities, then the earlier passage. Nothing scoring above
    0 leaves the claims empty so the task-row citation path can fill the row.
    """
    eid = fc.eid
    ent = spec.entity(eid)
    names = unit_names(spec)
    aliases = entity_aliases(spec)
    own = aliases.get(eid, ())
    own_cf = {a.casefold() for a in own}
    other = tuple(a for k, als in aliases.items() if k != eid for a in als if a.casefold() not in own_cf)
    own_re = _boundary_re(own)
    other_re = _boundary_re(other)
    best: dict[tuple[str, int, int], tuple[int, int, int, bool]] = {}
    cache: dict[str, list[tuple[int, int]]] = {}
    seq = 0

    def units_of(doc_id: str) -> list[tuple[int, int]]:
        got = cache.get(doc_id)
        if got is None:
            got = quote_units(corpus.docs[doc_id].text)
            cache[doc_id] = got
        return got

    def consider(doc_id: str, us: int, ue: int, order: int) -> None:
        if doc_id not in corpus.docs or not corpus.admits(eid, doc_id):
            return
        text = corpus.docs[doc_id].text
        # quote_span caps a long sentence at a line or sentence boundary and keeps the sentence start.
        span = quote_span(text, us, ue)
        if span is None:
            return
        quote = text[span[0]:span[1]]
        if is_header(quote) or not contentful(quote, names):
            return
        owned = own_re is not None and own_re.search(quote) is not None
        foreign = not owned and other_re is not None and other_re.search(quote) is not None
        score = 3 * int(owned) + int(any(c.isdigit() for c in quote)) - 4 * int(foreign)
        if score <= 0:
            return
        hits = len(other_re.findall(quote)) if other_re is not None else 0
        key = (doc_id, span[0], span[1])
        prev = best.get(key)
        if prev is None or (score, -hits, -order) > (prev[0], -prev[1], -prev[2]):
            best[key] = (score, hits, order, owned)

    def consider_region(doc_id: str, start: int, end: int) -> None:
        nonlocal seq
        for us, ue in units_of(doc_id):
            if us < end and start < ue:
                seq += 1
                consider(doc_id, us, ue, seq)

    for cid in fc.support:
        c = corpus.chunk(cid)
        if c is not None:
            consider_region(c.doc_id, c.start, c.end)
    for cid in ev.facts.get(eid, {}):
        c = corpus.chunk(cid)
        if c is not None:
            consider_region(c.doc_id, c.start, c.end)
    docs = {d for d in corpus.docs if corpus.admits(eid, d)}
    if docs:
        q = entity_query(spec, ent, plan_terms)
        for _, c in ev.bm.search(q, MAX_CLAIMS_PER_ENTITY * 6, docs):
            consider_region(c.doc_id, c.start, c.end)
    # Admitted documents, after the passages above, so a sentence the chunker cut in half is still a whole row
    # and a passage that never names this entity cannot crowd it out.
    specific = set(ev.specific.get(eid) or [])
    for did in sorted(docs, key=lambda d: (d not in specific, d)):
        consider_region(did, 0, len(corpus.docs[did].text))

    pool = [(sc, hits, order, owned, doc, s, e) for (doc, s, e), (sc, hits, order, owned) in best.items()]
    owned_pool = [row for row in pool if row[3]]
    use = owned_pool if owned_pool else pool
    use.sort(key=lambda row: (-row[0], row[1], row[2], row[5]))
    cites: list[tuple[str, int, int]] = []
    for _sc, _hits, _order, _owned, doc, s, e in use:
        if len(cites) >= MAX_CLAIMS_PER_ENTITY:
            break
        if any(d0 == doc and s0 < e and s < e0 for d0, s0, e0 in cites):
            continue
        cites.append((doc, s, e))
    return [{"doc_id": d, "span_start": s, "span_end": e, "claim": corpus.docs[d].text[s:e]} for d, s, e in cites]


def _add_alias(found: set[str], text: str) -> None:
    text = " ".join(text.split())
    if len(text) < 2 or text.casefold() in _NAME_STOP or text.casefold() in _SHORT_DENY:
        return
    if text.isdigit():
        return
    if len(text) == 2 and not (text.isalpha() and text.isupper()):
        return
    found.add(text)


def _add_tenor(found: set[str], number: str) -> None:
    _add_alias(found, f"{number}-Year")
    _add_alias(found, f"{number} Year")
    _add_alias(found, f"{number}Y")


def _add_id_fragments(found: set[str], entity_id: str) -> None:
    for part in re.split(r"[_\s]+", entity_id):
        if not part or _DAY.fullmatch(part) or re.fullmatch(r"\d{8}", part):
            continue
        if _MONTH.fullmatch(part):
            _add_alias(found, part)
            continue
        tenor = _TENOR_PART.fullmatch(part)
        if tenor:
            _add_tenor(found, tenor.group(1))
            continue
        prefixed = _TENOR_PREFIX.fullmatch(part)
        if prefixed:
            _add_alias(found, prefixed.group(1))
            _add_tenor(found, prefixed.group(2))
            continue
        if len(part) >= 3 or (len(part) == 2 and part.isalpha() and part.isupper()):
            _add_alias(found, part)


def _add_name(found: set[str], name: str) -> None:
    def add_phrase(phrase: str, keep_name: bool) -> None:
        phrase = " ".join(phrase.split()).strip(" ,;:-")
        while True:
            nxt = _SUFFIX.sub("", phrase).strip(" ,")
            if nxt == phrase:
                break
            phrase = nxt
        if phrase.casefold().startswith("the "):
            phrase = phrase[4:].strip()
        if not phrase:
            return
        for match in _TENOR_PHRASE.finditer(phrase):
            token = re.sub(r"\s+", "-", match.group(0).strip())
            _add_alias(found, token)
            _add_tenor(found, re.match(r"\d+", token).group(0))
        words = re.findall(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)?", phrase)
        content = [w for w in words if w.casefold() not in _NAME_STOP and len(w) >= 3]
        if content or (keep_name and len(words) >= 2 and len(phrase) >= 8):
            _add_alias(found, phrase)

    add_phrase(_PAREN.sub(" ", name), True)
    for match in _PAREN.finditer(name):
        add_phrase(match.group(1), False)


def _boundary_re(aliases: tuple[str, ...] | list[str]) -> re.Pattern[str] | None:
    months: list[str] = []
    rest: list[str] = []
    seen: set[str] = set()
    for alias in aliases:
        key = alias.casefold()
        if not alias or key in seen:
            continue
        seen.add(key)
        if _MONTH.fullmatch(alias):
            months.append(re.escape(alias))
        else:
            rest.append(re.escape(alias))
    parts: list[str] = []
    if rest:
        parts.append(r"(?<![\w])(?:" + "|".join(rest) + r")(?![\w])")
    if months:
        # '2024-07' is the reference month, not the vintage date '2024-07-26'.
        parts.append(r"(?<![\w])(?:" + "|".join(months) + r")(?![\w-])")
    if not parts:
        return None
    return re.compile("|".join(parts), re.IGNORECASE)


def _has_alias(text: str, aliases: tuple[str, ...] | list[str]) -> bool:
    pattern = _boundary_re(aliases)
    return pattern is not None and pattern.search(text) is not None


def _is_table_line(line: str) -> bool:
    return len(line) <= 2000 and line.count("|") >= 2


def _trim_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _is_sentence_end(text: str, i: int, end: int) -> bool:
    ch = text[i]
    # "2.50" is a decimal; "2024. Source" is the end of a sentence.
    if ch == "." and i > 0 and text[i - 1].isdigit() and i + 1 < end and text[i + 1].isdigit():
        return False
    if ch == ".":
        j = i - 1
        while j >= 0 and text[j].isalpha():
            j -= 1
        word = text[j + 1:i]
        if word.casefold() in _ABBR or (len(word) == 1 and word.isalpha()):
            return False
    k = i + 1
    while k < end and text[k] in " \t" + _CLOSERS:
        k += 1
    while k < end and text[k].isspace():
        k += 1
    if k >= end:
        return True
    return text[k].isupper() or text[k] in "(\"'"


def _sentences(text: str, start: int, end: int) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    sent: int | None = None
    i = start
    while i < end:
        ch = text[i]
        if sent is None:
            if ch.isspace():
                i += 1
                continue
            sent = i
        if ch in ".!?" and _is_sentence_end(text, i, end):
            stop = i + 1
            while stop < end and text[stop] in _CLOSERS:
                stop += 1
            s, e = _trim_bounds(text, sent, stop)
            if e > s:
                spans.append((s, e))
            sent = None
            i = stop
            continue
        i += 1
    if sent is not None:
        s, e = _trim_bounds(text, sent, end)
        if e > s:
            spans.append((s, e))
    return spans
