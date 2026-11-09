"""`submitted_reasons`: one to three unit-level reasons for the reasoning grader (Final, keyed units).

A grader panel compares each reason's premise, mechanism and answer implication with the unit's hidden target
reasons (a target reason that none of ours covers scores 0), checks that the premise is grounded in the cited
corpus text and that the implication agrees with the submitted answer. So:
- every premise is a verbatim quote of a corpus passage, cited with the span it was cut from (a verbatim premise
  of at least three words is also exempt from the deny list);
- v10: the reasons are planned against the drivers the task statement names ("Reason from ...: a, b, c"), or
  the plan's key drivers when it names none. Each reason takes one to three drivers, quotes the strongest
  passage for its first one and cites one more passage per further driver, so three reasons can cover six or
  more drivers (v9 covered one driver per reason, about 55-65% of the listed drivers);
- v10: each implication is checked in code against the submitted answers (a number written next to an entity
  id, a label named next to an entity, an up/down word next to an entity whose answer moves the other way);
  one that disagrees is replaced by an implication built from the answers themselves;
- the model writes the reasons in one call after the review; when there is no time or no reply, code builds
  them from the same drivers and passages, with implications built from the answers (v9 grouped entities by
  the kind of computed fact, so one implication listed credit_event and no_event entities together).
A malformed block would make the whole answer invalid, so every reason is checked against the published shape,
caps and deny list, and anything doubtful is dropped (no reasons at all costs nothing).
"""
from __future__ import annotations

import json
import logging
import math
import re
import time
import unicodedata

from . import config
from .corpus import Corpus, para_start
from .llm import BudgetExhausted, HouseClient, LLMUnavailable, extract_json, margin, parse_reply, salvage_array
from .prompts import reasons_messages
from .retrieval import BM25
from .task import TaskSpec

log = logging.getLogger("t4agent.reasons")

# The grader's deny list (case-insensitive substrings of the participant's own text).
DENY = ("leaderboard", "canary", "/home/", "units/", "reference/", "outcome.json", "team_id", "team name",
        "participant_id", "participant name", "submission_id", "other submission")
_URL = re.compile(r"(?i)(?:\b[a-z][a-z0-9+.\-]*:/{2}|\bwww\.|(?<![\w/])/{2}(?=\w))\S*")
CITATION_MAX = 8000  # characters per citation
CAP_REASONS = 6000  # bytes of the judge-visible reason fields (the published cap is 6,500)
CAP_EVIDENCE = 42000  # bytes of cited passages as the judge reads them (published 46,500)
CAP_ANSWER = 3000  # bytes of the per-entity answer as the judge reads it
PREMISE_MAX = 600
TEXT_MAX = 420
MECH_MAX = 650  # v10 (v9: 420 for every field): room for two or three drivers; three reasons stay under CAP_REASONS
IMPL_MAX = 450
WINDOW = 1500  # context cited beside a premise
SUPPORT_MAX = 2  # passages cited beside the premise, one per further driver
PASSAGES_MAX = 22
DRIVERS_MAX = 8
# typographic dashes, quotes and spaces cost two or three bytes each under the byte caps: own text uses ASCII
_ASCII = str.maketrans({"\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-", "\u2015": "-",
                        "\u2212": "-", "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u00a0": " ",
                        "\u2009": " ", "\u202f": " "})


def _compact(obj) -> int:
    return len(json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Cf")
    return re.sub(r"\s+", " ", text).strip().casefold()


# v21.5: the grader's folded view (citation_rail.fold_for_detection): look-alikes NFKC keeps apart read as '/' and ':'
_SLASH_LIKE = frozenset("\u2044\u2215\u29f8\u2571\u27cb\u3033\u1735\u2cfa\ufe68\u2e4a\u0338")
_COLON_LIKE = frozenset("\u2236\ua789\u02d0")
_FILLERS = frozenset("\u115f\u1160\u3164\uffa0")  # Hangul fillers (category Lo: clean() keeps them)


def fold(text: str) -> str:
    """The text as the grader scans it for the deny list: each character NFKC-normalised (fullwidth and mathematical
    letters become ASCII), slash and colon look-alikes read as '/' and ':', format characters and Hangul fillers
    dropped. v21.3 matched the raw text, so a fullwidth 'units/' passed here and refused the unit's reasons there."""
    if text.isascii():
        return text
    out = []
    for ch in text:
        if ch in _FILLERS or unicodedata.category(ch) == "Cf":
            continue
        out.append("/" if ch in _SLASH_LIKE else ":" if ch in _COLON_LIKE else unicodedata.normalize("NFKC", ch))
    return "".join(out)


def denied(text: str) -> bool:
    """On the grader's deny list, or a '://' with no scheme letters before it (refused, not masked); v21.5: read on the
    folded text, as the grader reads it."""
    low = fold(text).lower()
    if any(tok in low for tok in DENY):
        return True
    return any(not re.search(r"[a-z]$", low[:m.start()]) for m in re.finditer(r":/{2}", low))


def clean(text, limit: int = TEXT_MAX) -> str | None:
    """Own prose for a reason field: one line, no URLs, within `limit` characters (cut at a sentence end),
    and nothing on the deny list; None when unusable."""
    if not isinstance(text, str):
        return None
    t = unicodedata.normalize("NFC", text).translate(_ASCII)
    t = "".join(ch for ch in t if unicodedata.category(ch)[0] != "C" or ch in "\n\t")
    t = _URL.sub("", t)
    t = re.sub(r"\s+", " ", t).strip()
    if len(t) > limit:
        cut = t.rfind(". ", 0, limit)
        t = t[: cut + 1] if cut > limit // 2 else t[:limit].rsplit(" ", 1)[0]
    if len(t.split()) < 3 or denied(t):
        return None
    return t


def declared_rows(spec: TaskSpec, preds: list[dict]) -> list[dict]:
    """The per-entity answer as the grader's judge reads it (entity_id plus the declared answer fields)."""
    fields = ("label", "interval") if spec.target_type == "classification" else ("point_forecast", "interval")
    return [{"entity_id": p["entity_id"], **{k: p[k] for k in fields if k in p}} for p in preds]


def _premise_ok(corpus: Corpus, cite: dict, premise: str) -> bool:
    doc = corpus.docs.get(cite.get("doc_id"))
    s, e = cite.get("span_start"), cite.get("span_end")
    if doc is None or not isinstance(s, int) or not isinstance(e, int) or not 0 <= s < e <= len(doc.text):
        return False
    return e - s <= CITATION_MAX and premise in doc.text and len(premise.split()) >= 3


def validate(spec: TaskSpec, corpus: Corpus, reasons: list[dict]) -> list[dict]:
    """The reasons that fit the published contract, in order: shape, citations that resolve on or before the
    cutoff, verbatim premises, no duplicates, the deny list and the cumulative byte caps."""
    out: list[dict] = []
    seen: set[tuple[str, str, str]] = set()
    used_text = used_evidence = 0
    for r in reasons:
        if len(out) >= 3:
            break
        if not isinstance(r, dict):
            continue
        if not all(isinstance(r.get(k), str) and r[k].strip() for k in ("reason_id", "premise", "mechanism",
                                                                         "answer_implication")):
            continue
        cites = r.get("citations") or []
        if not isinstance(cites, list) or not cites or not all(
                isinstance(c, dict) and set(c) == {"doc_id", "span_start", "span_end"} and isinstance(c["doc_id"], str)
                and all(isinstance(c[k], int) and not isinstance(c[k], bool) and c[k] >= 0
                        for k in ("span_start", "span_end")) for c in cites):
            continue
        ok = True
        for c in cites:
            doc = corpus.docs.get(c["doc_id"])
            if (doc is None or spec.cutoff is None or doc.doc_date > spec.cutoff
                    or not 0 <= c["span_start"] < c["span_end"] <= len(doc.text)
                    or c["span_end"] - c["span_start"] > CITATION_MAX):
                ok = False
        if not ok or not _premise_ok(corpus, cites[0], r["premise"]):
            continue
        if denied(r["mechanism"]) or denied(r["answer_implication"]):
            continue
        key = (_norm(r["premise"]), _norm(r["mechanism"]), _norm(r["answer_implication"]))
        if key in seen:
            continue
        text_bytes = _compact({k: r[k] for k in ("reason_id", "premise", "mechanism", "answer_implication")})
        ev_bytes = sum(75 + len(c["doc_id"]) + _compact(corpus.docs[c["doc_id"]].text[c["span_start"]:c["span_end"]])
                       for c in cites)
        if used_text + text_bytes > CAP_REASONS or used_evidence + ev_bytes > CAP_EVIDENCE:
            continue
        scope = r.get("scope")
        clean_r = {k: r[k] for k in ("reason_id", "premise", "mechanism", "answer_implication")}
        if isinstance(scope, dict) and isinstance(scope.get("entities"), list):
            ents = [e for e in scope["entities"] if e in spec.entity_ids]
            if ents:
                clean_r["scope"] = {"entities": list(dict.fromkeys(ents))}
        clean_r["citations"] = [dict(c) for c in cites]
        out.append(clean_r)
        seen.add(key)
        used_text += text_bytes
        used_evidence += ev_bytes
    return out


# ---------------------------------------------------------------------------------------------- #
# Drivers the reasons should cover                                                                #
# ---------------------------------------------------------------------------------------------- #
_LEAD = re.compile(r"(?is)\breason(?:ing)?\b[^.:;]{0,60}?\bfrom\b\s*(.{10,800}?)(?:\.\s|\.$|$)")
_SEP = re.compile(r"\s*(?::|\s[-–—]\s|—)\s*")
_FILLER = re.compile(r"(?i)in the corpus|see below|given|absent by design")
_STOP = frozenset("the a an of to in on for and or with by from at as its it is are be been this that these those "
                  "how what which into over per vs versus each own their there than about whether".split())


def _split_top(text: str) -> list[str]:
    """Split on commas and semicolons outside parentheses, and after a closing parenthesis followed by 'and'."""
    out, depth, cur = [], 0, []
    for ch in text:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        if depth == 0 and ch in ",;":
            out.append("".join(cur))
            cur = []
            continue
        cur.append(ch)
    out.append("".join(cur))
    parts: list[str] = []
    for p in out:
        parts += re.split(r"(?<=\))\s+and\s+(?:from\s+)?", p)
    return parts


def task_drivers(prompt: str) -> list[str]:
    """The drivers the task statement asks to reason from ("Reason from X: a, b (c, d) and e"), as a list; a
    parenthesis that lists short items (revision direction, revision size, ...) contributes them one by one."""
    m = _LEAD.search(prompt or "")
    if not m:
        return []
    items: list[str] = []
    for raw in _split_top(_SEP.split(m.group(1).strip(), maxsplit=1)[-1]):
        t = re.sub(r"^\s*(?:and|or|while|as well as)\s+", "", raw.strip(), flags=re.I).strip(" .")
        t = re.sub(r"^from\s+", "", t, flags=re.I)
        if not t:
            continue
        sub = re.search(r"\(([^()]*)\)", t)
        if sub:
            subs = [x.strip() for x in re.split(r"[;,]", sub.group(1)) if x.strip()]
            if len(subs) >= 2 and all(2 <= len(x.split()) <= 8 for x in subs):
                items += subs
                continue
            if len(subs) == 1 and _FILLER.fullmatch(subs[0]):
                t = (t[:sub.start()] + t[sub.end():]).strip()
        items.append(re.sub(r"\s+", " ", t))
    return items[:DRIVERS_MAX]


def unit_drivers(spec: TaskSpec, plan: dict | None) -> list[str]:
    """What the reasons should cover: the task statement's own list of drivers, else the plan's key drivers
    (already grounded: notes with figures or dates the documents do not carry were dropped)."""
    drivers = task_drivers(spec.prompt)
    if drivers:
        return drivers
    kd = (plan or {}).get("key_drivers")
    out = []
    for d in kd if isinstance(kd, list) else []:
        if isinstance(d, str) and len(d.split()) >= 2 and not denied(d):
            out.append(re.sub(r"\s+", " ", d).strip()[:160])
    return out[:5]


def _words(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z]{3,}", text.lower()) if w not in _STOP]


def _stems(text: str) -> set[str]:
    return {w[:5] for w in _words(text)}


def driver_coverage(drivers: list[str], reasons: list[dict]) -> list[bool]:
    """Per driver: whether some reason's own text names at least half of the driver's content words (crude
    stems). A log line and a test, not a filter."""
    texts = [_stems(" ".join(str(r.get(k, "")) for k in ("premise", "mechanism", "answer_implication")))
             for r in reasons]
    out = []
    for d in drivers:
        ds = _stems(d)
        out.append(bool(ds) and any(len(ds & t) >= max(1, math.ceil(len(ds) / 2)) for t in texts))
    return out


# ---------------------------------------------------------------------------------------------- #
# Candidate premises                                                                              #
# ---------------------------------------------------------------------------------------------- #
def _driver_query(spec: TaskSpec) -> str:
    """The drivers the task statement asks to reason from ("Reason from ..."), else the whole prompt."""
    m = re.search(r"(?is)\breason(?:ing)? from\b(.{20,600}?)(?:\.\s|$)", spec.prompt)
    return m.group(1) if m else spec.prompt


def _owner(ev, doc_id: str) -> str | None:
    """The one entity a document belongs to, if any."""
    owners = [e for e, docs in (getattr(ev, "specific", None) or {}).items() if doc_id in (docs or [])]
    return owners[0] if len(owners) == 1 else None


def _prose(text: str) -> bool:
    """A sentence-like passage rather than a table row, a header or a list fragment."""
    t = text.strip()
    if len(t.split()) < 8 or t.count("|") > 2:
        return False
    chars = [ch for ch in t if not ch.isspace()]
    return sum(ch.isalpha() for ch in chars) >= 0.4 * len(chars)


def _evidence_pool(ev, forecasts: dict) -> set[int]:
    """Passages the analysts relied on and the key facts code found: the preferred evidence for a driver."""
    pool = {cid for fc in forecasts.values() if fc is not None for cid in (fc.support or [])[:4]}
    return pool | {cid for facts in (getattr(ev, "facts", None) or {}).values() for cid in facts}


def driver_passages(spec: TaskSpec, corpus: Corpus, ev, pool: set[int], driver: str, exclude=(),
                    k: int = 2) -> list[int]:
    """The best passages for one driver, from different documents: a prose passage that carries two of the
    driver's content words (the one word of a one-word driver), ranked by the share of them it carries, its word
    pairs found side by side ('going concern') and whether the analysts relied on it, then by BM25. A rare single word must not decide it (v10: 'going-concern
    language' once matched a passage on cultural and language differences)."""
    stems = _stems(driver)
    if not stems:
        return []
    words = _words(driver)
    pairs = list(zip(words, words[1:]))
    # hyphens split ('going-concern' is two words in most filings), plus the target's own words
    query = " ".join(words + [w for w in spec.target_name.replace("_", " ").split() if w.isalpha()])
    need = 1 if len(stems) == 1 else 2
    scored = []
    for score, c in ev.bm.search(query, 40, set(corpus.docs)):
        if c.cid in exclude or not _prose(c.text):
            continue
        hit = len(stems & _stems(c.text))
        if hit < need:
            continue
        cov = hit / len(stems)
        low = c.text.lower()
        together = sum(1 for a, b in pairs if re.search(rf"{re.escape(a)}[\s\-]+{re.escape(b)}", low))
        scored.append((cov + 0.5 * together + (0.5 if c.cid in pool else 0.0), score, c))
    scored.sort(key=lambda t: (-t[0], -t[1]))
    out: list[int] = []
    docs: set[str] = set()
    for _, _, c in scored:
        if c.doc_id not in docs:
            out.append(c.cid)
            docs.add(c.doc_id)
        if len(out) >= k:
            break
    return out


def candidates(spec: TaskSpec, corpus: Corpus, ev, forecasts: dict, drivers=(),
               limit: int = PASSAGES_MAX) -> list[tuple[int, int | None]]:
    """(chunk id, driver index or None) worth quoting: for each driver its best passages (driver_passages), then
    the analysts' most-cited evidence per entity and the code-found key facts."""
    picked: dict[int, int | None] = {}

    def add(cid: int, d: int | None = None) -> bool:
        c = corpus.chunk(cid)
        if c is None or cid in picked or len(c.text.split()) < 6 or len(picked) >= limit:
            return False
        picked[cid] = d
        return True

    bm: BM25 = ev.bm
    alldocs = set(corpus.docs)
    pool = _evidence_pool(ev, forecasts)
    for i, d in enumerate(drivers):
        for cid in driver_passages(spec, corpus, ev, pool, d, exclude=set(picked)):
            add(cid, i)
    for eid in spec.entity_ids:
        fc = forecasts.get(eid)
        for cid in (fc.support[:2] if fc is not None else []):
            add(cid)
    for eid in spec.entity_ids:
        for cid in list(ev.facts.get(eid, {}))[:1]:
            add(cid)
    if not drivers:
        for _, c in bm.search(_driver_query(spec), 12, alldocs):
            add(c.cid)
    return list(picked.items())


def _window(corpus: Corpus, cid: int) -> dict:
    """The premise chunk's paragraph context (within the citation cap), as a citation."""
    c = corpus.chunk(cid)
    text = corpus.docs[c.doc_id].text
    s = para_start(text, c.start)
    e = min(len(text), max(c.end, s + WINDOW))
    if e - s > WINDOW:
        s, e = c.start, min(len(text), c.start + WINDOW)
    e = max(e, c.end)
    return {"doc_id": c.doc_id, "span_start": s, "span_end": min(e, s + CITATION_MAX)}


def _premise(corpus: Corpus, cid: int) -> tuple[str, dict] | None:
    c = corpus.chunk(cid)
    if c is None:
        return None
    s, e = c.start, c.end
    text = corpus.docs[c.doc_id].text
    while s < e and text[s].isspace():
        s += 1
    while e > s and text[e - 1].isspace():
        e -= 1
    if e - s > PREMISE_MAX:
        cut = text.rfind(". ", s, s + PREMISE_MAX)
        e = cut + 1 if cut > s + PREMISE_MAX // 3 else s + PREMISE_MAX
    premise = text[s:e]
    if len(premise.split()) < 3 or "\n" in premise.strip() and len(premise) < 30:
        return None
    if denied(premise):  # a verbatim premise is exempt, but nothing is gained by testing the exemption
        return None
    return premise, {"doc_id": c.doc_id, "span_start": s, "span_end": e}


# ---------------------------------------------------------------------------------------------- #
# Answers in words, and whether an implication agrees with them                                   #
# ---------------------------------------------------------------------------------------------- #
_CHANGE_TARGET = re.compile(r"change|growth|revision|delta|bps|return|shift|surprise", re.I)
_UPW = re.compile(r"(?i)\b(?:positive|increase[sd]?|increasing|rise[sn]?|rising|rose|higher|gains?|improve[sd]?"
                  r"|improving|improvement|up|upward|beat)\b")
_DOWNW = re.compile(r"(?i)\b(?:negative|decline[sd]?|declining|decrease[sd]?|decreasing|falls?|falling|fell|lower"
                    r"|drops?|dropping|dropped|contraction|down|downward|miss)\b")
# a negation or a comparison makes a direction word say something else ('no maturity is expected to decline',
# 'the long end rises less than the front end'): such a clause is not read for direction
_HEDGE = re.compile(r"(?i)\b(?:not|no|never|without|neither|nor|than)\b|n't\b")
NEAR = 80  # characters between an entity id and the direction word that governs it
_LAB_FLAT = re.compile(r"(?i)flat|inline|in_line|unchanged|neutral|no_change")
_LAB_UP = re.compile(r"(?i)positive|^up$|beat|higher|increase|rise|above")
_LAB_DOWN = re.compile(r"(?i)negative|^down$|miss|lower|decrease|fall|decline|below")


def _num(x) -> str:
    return f"{x:g}" if isinstance(x, (int, float)) else str(x)


def answer_line(spec: TaskSpec, pred: dict, plan: dict | None = None) -> str:
    iv = pred.get("interval") or {}
    rng = f"90% interval [{_num(iv.get('lo'))}, {_num(iv.get('hi'))}]"
    if spec.target_type == "classification":
        return f"{pred['entity_id']}: {pred.get('label')} ({rng})"
    rel = versus_latest(spec, pred, plan)
    return f"{pred['entity_id']}: {_num(pred.get('point_forecast'))} ({rng}{'; ' + rel if rel else ''})"


def versus_latest(spec: TaskSpec, pred: dict, plan: dict | None = None) -> str | None:
    """v20: for a level target, where the answer sits against the naive baseline (the latest value), the reference of
    the reasons' 'argues' field. The reasons prompt listed the answers alone, and v20's CPI reasons declared 'higher'
    for components forecast below their latest prints; those reasons were held back (argued_against)."""
    v = pred.get("point_forecast")
    if _CHANGE_TARGET.search(spec.target_name) or not isinstance(v, (int, float)):
        return None
    from .predict import baseline_value

    base = baseline_value(spec, spec.entity(pred["entity_id"]), plan)
    if not isinstance(base, (int, float)):
        return None
    iv = pred.get("interval") if isinstance(pred.get("interval"), dict) else {}
    lo, hi = iv.get("lo"), iv.get("hi")
    tol = NEUTRAL_SHARE * (hi - lo) / 2 if isinstance(lo, (int, float)) and isinstance(hi, (int, float)) else 0.0
    word = "about the same as" if abs(v - base) <= tol else "higher than" if v > base else "lower than"
    return f"{word} the latest value {_num(base)}"


def _label_dir(label: str) -> str | None:
    if _LAB_FLAT.search(label):
        return "flat"
    if _LAB_UP.search(label):
        return "up"
    if _LAB_DOWN.search(label):
        return "down"
    return None


def _answer_dirs(spec: TaskSpec, preds: list[dict]) -> dict[str, str]:
    """Direction of each submitted answer, where it has one: a directional label (up, beat, positive_reaction,
    flat ...), or the sign of a change-type value (growth, change in bps, revision)."""
    change_like = bool(_CHANGE_TARGET.search(spec.target_name))
    out: dict[str, str] = {}
    for p in preds:
        d = None
        if spec.target_type == "classification":
            d = _label_dir(str(p.get("label") or ""))
        elif change_like and isinstance(p.get("point_forecast"), (int, float)):
            v = p["point_forecast"]
            d = "up" if v > 0 else "down" if v < 0 else None
        if d:
            out[p["entity_id"]] = d
    return out


def _clauses(text: str) -> list[str]:
    """Sentences and semicolon parts, also cut at a contrast ('X rises while Y declines' holds two directions)."""
    return [c for c in re.split(r";\s+|\.\s+(?=[A-Z(])|\n+|,?\s+(?:while|whereas|but|although|though|versus|vs\.?"
                                r"|compared with|in contrast to)\s+", text) if c and c.strip()]


def _id_pattern(eid: str) -> re.Pattern:
    return re.compile(rf"(?<![\w-]){re.escape(eid)}(?![\w-])")


def _ids_in(text: str, ids) -> list[str]:
    return [e for e in sorted(ids, key=len, reverse=True) if _id_pattern(e).search(text)]


def implication_issues(spec: TaskSpec, preds: list[dict], text: str) -> list[str]:
    """Ways an implication disagrees with the submitted answers, clause by clause: a number written right after
    an entity id that is neither its submitted value nor an interval bound; a label named in a clause next to an
    entity submitted with another label; an up (down) word in a clause next to an entity whose answer goes down
    (up). Empty when nothing disagrees."""
    by_id = {p["entity_id"]: p for p in preds}
    dirs = _answer_dirs(spec, preds)
    labels = sorted(spec.labels or [], key=len, reverse=True)
    issues: list[str] = []
    for clause in _clauses(text):
        ids = _ids_in(clause, by_id)
        if not ids:
            continue
        for eid in ids:
            for m in re.finditer(rf"(?<![\w-]){re.escape(eid)}(?![\w-])[\s:(=]{{0,3}}([+-]?\d+(?:\.\d+)?)", clause):
                p = by_id[eid]
                vals = [p.get("point_forecast")] + [(p.get("interval") or {}).get(k) for k in ("lo", "hi")]
                vals = [v for v in vals if isinstance(v, (int, float))]
                dec = len(m.group(1).split(".")[1]) if "." in m.group(1) else 0
                tol = 0.5 * 10 ** (-dec) + 1e-9
                if vals and not any(abs(float(m.group(1)) - v) <= tol for v in vals):
                    issues.append(f"{eid} written as {m.group(1)}")
        named, rest = [], clause
        for lab in labels:
            pat = re.compile(r"(?i)(?<![\w-])" + re.escape(lab).replace("_", "[_ ]") + r"(?![\w-])")
            if pat.search(rest):
                named.append(lab)
                rest = pat.sub(" ", rest)
        if len(named) == 1:
            issues += [f"{e} named {named[0]}" for e in ids if by_id[e].get("label") not in (None, named[0])]
        elif not named:
            bare = re.sub(r"\([^()]*\)", " ", clause)  # '(-0.5% - offsets a negative provision effect)' qualifies
            if _HEDGE.search(bare):
                continue
            words = [(m.start(), "up") for m in _UPW.finditer(bare)] + [(m.start(), "down") for m in _DOWNW.finditer(bare)]
            for e in ids:
                if not dirs.get(e) or (dirs[e] == "flat" and spec.target_type != "classification"):
                    continue
                at = [m.start() for m in _id_pattern(e).finditer(bare)]
                near = sorted((abs(w - a), d) for w, d in words for a in at if abs(w - a) <= NEAR)
                if near and near[0][1] != dirs[e]:
                    issues.append(f"{e} called {near[0][1]}")
    return issues


# v12: only an explicit contrast with the submitted answer; 'for all other entities the reason does not apply' is
# a reason stating its scope (v12's first sweep dropped two sound credit reasons on it). A reason arguing the other
# way is caught by the direction it declares (argued_against).
_SELF_DENIAL = re.compile(r"(?i)not assigned to|(?:contradict|conflict)s? (?:with )?(?:the |its |our )?submitted"
                          r"|against (?:the |its |our )?submitted (?:answer|label|value)|but the submitted (?:answer|label|value)")


def self_denying(text: str) -> bool:
    """An implication that argues against the answers it is meant to explain ('points to a lower EPS, but the
    submitted answer is up; this reason does not apply'): v10 drops such a reason rather than submit it."""
    return bool(_SELF_DENIAL.search(text or ""))


_CORP_WORDS = re.compile(r"(?i)\b(?:the|inc|incorporated|corp|corporation|company|co|group|holdings?|plc|ltd|limited|llc"
                         r"|futures?|contracts?)\b\.?")
_ALIAS_STOP = frozenset("the and for with from all less new total index rate rates note notes bond bonds auction "
                        "week year month quarter services items type types".split())


def entity_aliases(spec: TaskSpec) -> dict[str, list[str]]:
    """How a reason may name each entity: its id, its series code, its name without corporate words, and any word
    of its name (3+ letters, letters in it) that no other entity's name carries ('goldman', 'yen', '10-year')."""
    names, full = {}, {}
    for e in spec.entities:
        nm = str(e.get("name") or e.get("series_name") or "")
        full[e["entity_id"]] = nm
        names[e["entity_id"]] = re.sub(r"\s+", " ", _CORP_WORDS.sub(" ", re.sub(r"\(.*?\)", " ", nm))).strip(" ,.&")
    words = {eid: {w for w in _WORD.findall(nm.lower()) if len(w) >= 3 and w not in _ALIAS_STOP and re.search("[a-z]", w)}
             for eid, nm in full.items()}
    out = {}
    for e in spec.entities:
        eid = e["entity_id"]
        others = set().union(*[w for k, w in words.items() if k != eid]) if len(words) > 1 else set()
        al = {eid} | {str(e[k]) for k in ("series_fred", "series_id", "ticker") if isinstance(e.get(k), str) and e[k]}
        if len(names[eid]) >= 3:
            al.add(names[eid])
        al |= {w for w in words[eid] if w not in others}
        out[eid] = sorted(al, key=len, reverse=True)
    return out


_WORD = re.compile(r"[a-z0-9][a-z0-9&\-]*[a-z0-9]")


def _names(text: str, eid: str, aliases: dict[str, list[str]]) -> bool:
    """Whether the text names the entity: its id, a multi-word alias, an alias word, or a 4+ letter abbreviation of
    a long alias word ('Citi' for 'citigroup'). Generous on purpose: a missed name holds back a sound reason."""
    words = set(_WORD.findall(text.lower()))
    for a in aliases.get(eid, [eid]):
        if a == eid:
            if _id_pattern(eid).search(text):
                return True
        elif " " in a or not a.isalnum():
            if re.search(r"(?i)(?<![\w-])" + re.escape(a) + r"(?![\w-])", text):
                return True
        else:
            lo = a.lower()
            if lo in words or (len(lo) >= 6 and any(len(w) >= 4 and lo.startswith(w) for w in words)):
                return True
    return False


_YEAR = re.compile(r"^(?:19|20)\d\d$")


def _figures(text: str) -> set[str]:
    return {f.replace(",", "") for f in re.findall(r"\d[\d,]*(?:\.\d+)?", text)
            if len(f.replace(",", "").replace(".", "")) >= 2 and not _YEAR.match(f)}


def borrows_from(mechanism: str, source: str, own: str) -> bool:
    """v12: whether a mechanism uses what only an unnamed source says: a figure of that source (not a year, not in
    the reason's own sources), or two of its words of 6+ letters found nowhere in the reason's own sources."""
    figs = _figures(mechanism) & _figures(source) - _figures(own)
    mech = {w for w in re.findall(r"[a-z]{6,}", mechanism.lower())}
    words = mech & set(re.findall(r"[a-z]{6,}", source.lower())) - set(re.findall(r"[a-z]{6,}", own.lower())) - _STOP
    return bool(figs) or len(words) >= 2


def names_strongly(text: str, eid: str, aliases: dict[str, list[str]]) -> bool:
    """Whether the text names the entity unmistakably: its id, its series code, its full cleaned name, or a word of
    5+ letters only its name carries (or a 4+ letter abbreviation of a 6+ letter one) -- not a short common word
    such as 'bank' or 'best'."""
    words = set(_WORD.findall(text.lower()))
    for a in aliases.get(eid, [eid]):
        if a == eid:
            if _id_pattern(eid).search(text):
                return True
        elif " " in a or not a.isalnum() or a.isupper():
            if re.search(r"(?i)(?<![\w-])" + re.escape(a) + r"(?![\w-])", text):
                return True
        elif len(a) >= 5:
            lo = a.lower()
            if lo in words or (len(lo) >= 6 and any(len(w) >= 4 and lo.startswith(w) for w in words)):
                return True
    return False


def unnamed_sources(ev, citations: list[dict], text: str, aliases: dict[str, list[str]]) -> list[str]:
    """v12: the entities whose own documents a reason cites but never names (by id, code or name). A reason may
    use a peer's filing only by saying whose it is: v11's GS reason quoted Citi's Visa gain and Bank of America's
    investment-banking fees as if they were Goldman's."""
    owners: dict[str, set[str]] = {}
    for e, docs in (getattr(ev, "specific", None) or {}).items():
        for d in docs or []:
            owners.setdefault(d, set()).add(e)
    shared = set(getattr(ev, "shared", None) or [])
    missing: list[str] = []
    for c in citations:
        own = owners.get(c.get("doc_id"))
        if c.get("doc_id") in shared or not own:
            continue
        if not any(_names(text, e, aliases) for e in own):
            missing += sorted(own - set(missing))
    return missing


NEUTRAL_SHARE = 0.1  # v20: of the 90% half-width; a closer answer argues no direction (argued_against)
_UP_WORDS = {"up", "higher", "positive", "increase", "rise", "long", "more long", "beat"}
_DOWN_WORDS = {"down", "lower", "negative", "decrease", "fall", "decline", "short", "more short", "miss"}


def argued_against(spec: TaskSpec, pred: dict, argued: str, plan: dict | None = None) -> bool:
    """v12: whether the direction a reason declares it argues for an entity contradicts the submitted answer: a
    label other than the submitted one; or up/down against the sign of a change target's value (or against the
    point's side of the naive baseline for a level target). Unknown declarations contradict nothing."""
    a = re.sub(r"[_\s]+", " ", str(argued or "").strip().lower())
    if spec.target_type == "classification":
        labels = {re.sub(r"[_\s]+", " ", lab.lower()): lab for lab in spec.labels}
        return a in labels and labels[a] != pred.get("label")
    v = pred.get("point_forecast")
    if not isinstance(v, (int, float)) or a not in _UP_WORDS | _DOWN_WORDS:
        return False
    ref = 0.0
    if not _CHANGE_TARGET.search(spec.target_name):
        from .predict import baseline_value

        base = baseline_value(spec, spec.entity(pred["entity_id"]), plan)
        if base is None:
            return False
        ref = base
        # v20: against the latest value, an answer within a tenth of its 90% half-width makes no directional call, so
        # no declared direction contradicts it. v20's first CPI sweep held back both reasons on the October gasoline
        # rebound because headline's 0.17 sat 0.01 below its latest 0.18 (half-width 0.25), leaving gasoline and
        # energy unexplained. A change target keeps the sign test: -3% read against 'up' is a contradiction.
        iv = pred.get("interval") if isinstance(pred.get("interval"), dict) else {}
        lo, hi = iv.get("lo"), iv.get("hi")
        if isinstance(lo, (int, float)) and isinstance(hi, (int, float)) and abs(v - ref) <= NEUTRAL_SHARE * (hi - lo) / 2:
            return False
    if v == ref:
        return False
    return (a in _UP_WORDS) != (v > ref)


_FIG = re.compile(r"(?<![\w.])[-+\u2212]?\$?\d[\d,]*(?:\.\d+)?%?")


def _fig_value(raw: str) -> float | None:
    t = raw.replace("\u2212", "-").replace("$", "").replace(",", "").rstrip("%").lstrip("+")
    try:
        return abs(float(t))
    except ValueError:
        return None


def ungrounded_figures(text: str, sources: list[str], preds: list[dict]) -> list[str]:
    """v21: figures a reason states that neither its cited passages nor the submitted answers contain. The grader checks
    the premise and mechanism against the cited text, and v20's post-earnings reasons put invented reactions ('roughly
    -2% to -3%') in theirs. Small integers (counts, days, quarters) and years are ignored; a figure matches one in the
    sources to its own rounding and within 5% (12% matches 12.3)."""
    have = [v for src in sources for v in (_fig_value(m.group(0)) for m in _FIG.finditer(src or "")) if v is not None]
    for p in preds:
        iv = p.get("interval") if isinstance(p.get("interval"), dict) else {}
        have += [abs(x) for x in (p.get("point_forecast"), iv.get("lo"), iv.get("hi")) if isinstance(x, (int, float))]
    out = []
    for m in _FIG.finditer(text or ""):
        raw = m.group(0).strip()
        v = _fig_value(raw)
        if v is None or (v.is_integer() and "%" not in raw and "$" not in raw and (v <= 31 or 1900 <= v <= 2100)):
            continue  # a count, a day, a quarter or a year; '2%' and '$3' are figures
        dec = len(raw.rstrip("%").split(".")[1]) if "." in raw else 0
        # the figure as rounded from a source number, and within 5% of it (2% is no rounding of an EPS of 1.82)
        if not any(abs(v - h) <= min(0.5 * 10 ** -dec, 0.05 * max(v, h)) + 1e-9 for h in have):
            out.append(raw)
    return out


def _as_text(x) -> str | None:
    """A field the model returned as an object or a list ({"JPY_CME": "..."}) as one line of text; v11 rebuilt every
    such implication from the answers unchecked, which hid a mechanism arguing the other way."""
    if isinstance(x, str) or x is None:
        return x
    if isinstance(x, dict):
        return "; ".join(f"{k}: {_as_text(v)}" for k, v in x.items() if _as_text(v))
    if isinstance(x, list):
        return "; ".join(t for t in (_as_text(v) for v in x) if t)
    return str(x)


def evidence_cover(ev, citations: list[dict], entity_ids) -> set[str]:
    """The entities a reason's citations speak for: the entity whose own document is cited, or every entity once a
    cited document is shared by all (or belongs to no single entity)."""
    owners: dict[str, set[str]] = {}
    for e, docs in (getattr(ev, "specific", None) or {}).items():
        for d in docs or []:
            owners.setdefault(d, set()).add(e)
    shared = set(getattr(ev, "shared", None) or [])
    out: set[str] = set()
    for c in citations:
        d = c.get("doc_id")
        if d in shared or d not in owners:
            return set(entity_ids)
        out |= owners[d]
    return out


def argues_hint(spec: TaskSpec) -> str:
    """What 'argues' means for this unit's target, for the reasons prompt."""
    if spec.target_type == "classification":
        return f"the label the reason supports for it (one of {', '.join(spec.labels)})"
    if _CHANGE_TARGET.search(spec.target_name):
        return "'up' if the reason implies a positive value of the target for it, 'down' if a negative one"
    return ("'higher' or 'lower' than the naive baseline (the latest value) for it, as listed next to each answer "
            "(leave out 'argues' for an answer listed as about the same as its latest value)")


def focus_entities(spec: TaskSpec, preds: list[dict], plan: dict | None, k: int = 4) -> list[str]:
    """The answers the reasons must explain first: on a numeric target the ones furthest from the naive baseline
    (no change, or the latest value), on a classification the labels that differ from the most common one. v10's
    bank reasons explained Citi, U.S. Bancorp and PNC while GS (+40.5%) and MS (+16.7%) went unexplained."""
    if spec.target_type == "classification":
        labels = [p.get("label") for p in preds]
        counts = sorted(((labels.count(lab), lab) for lab in set(labels)), key=lambda t: -t[0])
        if len(counts) < 2 or counts[0][0] == counts[1][0]:
            return []  # no most common label: nothing stands out
        return [p["entity_id"] for p in preds if p.get("label") != counts[0][1]][:k]
    from .predict import baseline_value

    scored = []
    for p in preds:
        v = p.get("point_forecast")
        if isinstance(v, (int, float)):
            base = baseline_value(spec, spec.entity(p["entity_id"]), plan)
            scored.append((abs(v - (base if base is not None else 0.0)), p["entity_id"]))
    scored.sort(key=lambda t: -t[0])
    return [e for d, e in scored[:k] if d > 0]


def code_implication(spec: TaskSpec, preds: list[dict], eids) -> str | None:
    """An implication built from the submitted answers themselves, one group per label or direction."""
    by_id = {p["entity_id"]: p for p in preds}
    eids = [e for e in dict.fromkeys(eids or []) if e in by_id] or list(by_id)
    if spec.target_type == "classification":
        groups: dict[str, list[str]] = {}
        for e in eids:
            groups.setdefault(str(by_id[e].get("label")), []).append(e)
        text = "Consistent with the submitted labels: " + "; ".join(
            f"{lab} for {', '.join(es)}" for lab, es in groups.items()) + "."
    elif _CHANGE_TARGET.search(spec.target_name):
        groups = {"increases": [], "decreases": [], "no change": []}
        for e in eids:
            v = by_id[e].get("point_forecast")
            key = "no change" if not isinstance(v, (int, float)) or v == 0 else "increases" if v > 0 else "decreases"
            groups[key].append(f"{e} ({_num(v)})")
        text = "Consistent with the submitted values: " + "; ".join(
            f"{k} for {', '.join(v)}" for k, v in groups.items() if v) + "."
    else:
        text = "Consistent with the submitted values: " + "; ".join(
            f"{e} {_num(by_id[e].get('point_forecast'))}" for e in eids) + "."
    return clean(_fit_list(text, IMPL_MAX), IMPL_MAX)


def _fit_list(text: str, limit: int) -> str:
    """v21.5: a listing over `limit` characters cut after its last whole item that fits (at '; ' or ', ') and closed
    with a period. clean() cut it at a space, which left long rosters ending '...; down for' with no period."""
    if len(text) <= limit:
        return text
    cut = max(text.rfind("; ", 0, limit), text.rfind(", ", 0, limit))
    if cut <= 0:
        return text
    return text[:cut].rstrip(" ,;:") + "."


# ---------------------------------------------------------------------------------------------- #
# Code-built reasons                                                                              #
# ---------------------------------------------------------------------------------------------- #
def _fact(forecaster, eid: str):
    """The code-computed fact behind an entity's forecast: its statistical baseline, anchor or base rate."""
    if forecaster is None:
        return None
    for table in ("stat", "anchor", "prior"):
        f = (getattr(forecaster, table, None) or {}).get(eid)
        if f is not None and hasattr(f, "describe"):
            return f
    return None


def _join(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


# v21.5: the plain words of a statistical baseline's method (stats.py), for a code-built mechanism
_METHOD_WORDS = {
    "last": "the latest value carried forward", "zero": "no change", "zero (random walk)": "no change",
    "ar1": "the latest value pulled towards its recent mean by the series' own persistence",
    "mean3": "the mean of its last 3 values", "mean6": "the mean of its last 6 values",
    "median6": "the median of its last 6 values", "ewma": "an exponentially weighted average of its recent values",
    "drift": "its recent trend extended", "momentum": "half its latest change repeated",
    "reversal": "half its latest change reversed", "level_revert": "a partial reversion towards its recent average",
    "level reversion (short history)": "a partial reversion towards its recent average",
}
# backtest bookkeeping and the notes describe() addresses to the analysts, never mechanism text
_BOOKKEEPING = re.compile(r"\s*\((?:[^()]|\([^()]*\))*\b(?:backtest|backtestable|origins|out-of-sample|fitted on)\b"
                          r"(?:[^()]|\([^()]*\))*\)|(?:, )?computed by code ?|\s*\(code\)"
                          r"|\s*(?:One-off items in the quarters used|A weak base rate, not a verdict)[^.]*\.")
_SENTENCE_END = re.compile(r"(?<=[a-z0-9%)])\.(?=\s+[A-Z(]|$)")  # not 'i.e. +4.7%' nor 'U.S. gasoline'


def _plain_fact(text: str, corpus: Corpus, limit: int) -> str | None:
    """v21.5: a code-computed fact (`describe()`) as a code-built mechanism states it: no internal document ids (the
    panel reads the cited passages, not file names), no backtest bookkeeping, the method in plain words, and whole
    sentences only -- cut at the last sentence end that fits `limit` (v21.3 cut at 170 characters mid-word: '...against
    its ow'), else at a clause end; None when not even one clause fits."""
    if limit <= 0:  # no room left (a long lead): str.rfind would read a negative end from the string's end
        return None
    for did in sorted(set(corpus.docs) | set(corpus.excluded), key=len, reverse=True):
        if did in text:
            text = re.sub(rf"\s+(?:in|from|of)\s+{re.escape(did)}(?![\w.-])", "", text)
            text = re.sub(rf"(?<![\w.-]){re.escape(did)}(?![\w.-])", "a corpus table", text)
    text = re.sub(r"best simple method '([^']+)' -> (\S+)",
                  lambda m: f"{_METHOD_WORDS.get(m.group(1), 'a simple rule')} gives {m.group(2)}", text)
    text = re.sub(r"\s+", " ", _BOOKKEEPING.sub("", text)).strip().rstrip(";,: ")
    text += "" if text.endswith(".") else "."
    if len(text) <= limit:
        return text
    ends = [m.end() for m in _SENTENCE_END.finditer(text) if m.end() <= limit]
    if ends:
        return text[:ends[-1]]
    cut = text.rfind("; ", 0, limit)
    return text[:cut] + "." if cut > 0 else None


def _code_reason(spec, corpus, ev, forecaster, preds, cids: list[int], names: list[str], eids: list[str],
                 bare: list[str] = ()) -> dict | None:
    prem = _premise(corpus, cids[0])
    if prem is None:
        return None
    premise, cite = prem
    eids = list(dict.fromkeys(eids))  # v21.5: an entity once, however many of its passages are cited
    if names:
        lead = (f"The task statement names {_join(names)} as {'drivers' if len(names) > 1 else 'a driver'} of the "
                f"outcome; the cited passages are the pre-cutoff evidence on {'them' if len(names) > 1 else 'it'}, "
                "and the forecasts weigh that evidence against each entity's own history in the corpus.")
    else:
        lead = "The cited passages are the pre-cutoff evidence the forecasts for these entities start from."
    if bare:
        lead += (f" The forecasts also reflect {_join(list(bare))}, on which the corpus offers no passage beyond the "
                 "computed history.")
    # v21.5: the first two entities' facts, each in whole sentences within the room the lead leaves
    facts, room = [], MECH_MAX - len(lead) - len(" Computed from the corpus: ")
    for e in eids[:2]:
        sf = _fact(forecaster, e)
        fact = _plain_fact(sf.describe(), corpus, room - len(e) - 3) if sf is not None else None
        if fact:
            facts.append(f"{e}: {fact}")
            room -= len(facts[-1]) + 1
    mechanism = clean(lead + (" Computed from the corpus: " + " ".join(facts) if facts else ""), MECH_MAX)
    implication = code_implication(spec, preds, eids)
    if not (mechanism and implication):
        return None
    citations = [cite]
    for cid in cids:
        w = _window(corpus, cid)
        if w not in citations:
            citations.append(w)
    return {"reason_id": "r0", "premise": premise, "mechanism": mechanism, "answer_implication": implication,
            "scope": {"entities": eids[:12]}, "citations": citations[:1 + 1 + SUPPORT_MAX]}


def code_reasons(spec: TaskSpec, corpus: Corpus, ev, forecaster, forecasts: dict, preds: list[dict],
                 drivers=()) -> list[dict]:
    """Reasons from the drivers and the evidence when the model cannot write them: the drivers are dealt into
    three bundles; each reason quotes the best passage for its first driver, cites one passage per further
    driver, and its implication is built from the submitted answers. Without drivers, one reason per label or
    direction of the answers, each quoting what the analysts relied on for those entities."""
    by_id = {p["entity_id"]: p for p in preds}
    cands = candidates(spec, corpus, ev, forecasts, drivers)
    out: list[dict] = []
    used: set[int] = set()
    if drivers:
        pool = _evidence_pool(ev, forecasts)
        for bundle in [list(range(len(drivers)))[i::3] for i in range(3)]:
            cids, names, bare = [], [], []
            for d in bundle:
                hits = [c for c in driver_passages(spec, corpus, ev, pool, drivers[d], exclude=used, k=3)
                        if _premise(corpus, c)]
                cid = hits[0] if hits else None
                if cid is not None:
                    cids.append(cid)
                    names.append(drivers[d])
                    used.add(cid)
                else:
                    bare.append(drivers[d])
            if not cids:
                continue
            owners = list(dict.fromkeys(o for o in (_owner(ev, corpus.chunk(c).doc_id) for c in cids) if o))
            r = _code_reason(spec, corpus, ev, forecaster, preds, cids, names, owners or list(by_id), bare)
            if r is not None:
                out.append(r)
    if not out:
        dirs = _answer_dirs(spec, preds)
        groups: dict[str, list[str]] = {}
        for e in spec.entity_ids:
            if e in by_id:
                key = str(by_id[e].get("label")) if spec.target_type == "classification" else dirs.get(e, "other")
                groups.setdefault(key, []).append(e)
        for _, eids in sorted(groups.items(), key=lambda kv: -len(kv[1]))[:3]:
            cids = []
            for e in eids:
                fc = forecasts.get(e)
                for c in (fc.support if fc is not None else []) + list(ev.facts.get(e, {})):
                    if c not in used and _premise(corpus, c):
                        cids.append(c)
                        used.add(c)
                        break
                if len(cids) >= 1 + SUPPORT_MAX:
                    break
            if not cids:
                cids = [c for c, _ in cands if c not in used and _premise(corpus, c)][:1]
                used.update(cids)
            if cids:
                r = _code_reason(spec, corpus, ev, forecaster, preds, cids, [], eids)
                if r is not None:
                    out.append(r)
    for i, r in enumerate(out, 1):
        r["reason_id"] = f"r{i}"
    return out


# ---------------------------------------------------------------------------------------------- #
# Model-written reasons                                                                           #
# ---------------------------------------------------------------------------------------------- #
def _unlabel(text, drivers) -> str | None:
    """The model's working labels out of its prose: 'D3' becomes the driver it stands for and 'P7' a cited passage
    (the panel sees the cited text, never these numbers)."""
    if not isinstance(text, str):
        return text
    text = re.sub(r"\bD(\d{1,2})\b", lambda m: (f"'{drivers[int(m.group(1)) - 1]}'" if 0 < int(m.group(1)) <= len(drivers)
                                               else "this driver"), text)
    text = re.sub(r"\(\s*(?:and\s+)?P\d{1,2}(?:\s*(?:,|and)\s*P\d{1,2})*\s*\)", "", text)
    text = re.sub(r"\bP\d{1,2}(?:\s*(?:,|and)\s*P\d{1,2})*\b", "a cited passage", text)
    return re.sub(r"\s{2,}", " ", text)


def model_reasons(spec: TaskSpec, corpus: Corpus, ev, forecaster, forecasts: dict, preds: list[dict],
                  client: HouseClient, drivers=()) -> list[dict]:
    cands = candidates(spec, corpus, ev, forecasts, drivers)
    passages, keyed = [], {}
    for i, (cid, d) in enumerate(cands, 1):
        prem = _premise(corpus, cid)
        if prem is None:
            continue
        pid = f"P{i}"
        keyed[pid] = (cid, prem)
        c = corpus.chunk(cid)
        tags = [f"{c.doc_id}, {corpus.docs[c.doc_id].doc_date}"]
        owner = _owner(ev, c.doc_id)
        if owner:
            tags.append(f"entity {owner}")
        if d is not None:
            tags.append(f"found for D{d + 1}")
        passages.append(f"[{pid}] ({'; '.join(tags)}) {prem[0]}")
    if not passages:
        return []
    facts = []
    for eid in spec.entity_ids:
        lines = [sf.describe()[:260] for sf in (forecaster.stat.get(eid) or forecaster.anchor.get(eid),
                                                forecaster.prior.get(eid)) if sf is not None and hasattr(sf, "describe")]
        lines += [x[:260] for x in (ev.features.get(eid) or [])[:2]]
        if lines:
            facts.append(f"{eid}: " + " / ".join(lines))
    answers = "\n".join(answer_line(spec, p, forecaster.plan) for p in preds)
    by_id = {p["entity_id"]: p for p in preds}
    aliases = entity_aliases(spec)
    focus = [answer_line(spec, by_id[e]).split(" (90%")[0] for e in focus_entities(spec, preds, forecaster.plan)]
    msgs = reasons_messages(spec, answers, "\n".join(facts), "\n".join(passages), list(drivers), focus,
                            argues_hint(spec))
    rep = client.chat(msgs, reasoning_budget=config.REASONS_REASONING, temperature=0.3, seed=config.SEED + 23,
                      max_attempts=4, tag="reasons")
    try:
        obj = parse_reply(rep, "reasons")
    except ValueError:
        # v21.13: a reply cut off by the output cap keeps its complete reasons
        obj = salvage_array(rep.content, "reasons") or salvage_array(rep.reasoning, "reasons")
        if obj is not None:
            log.warning("reasons reply cut off (finish=%s): %d complete reasons salvaged", rep.finish,
                        len(obj["reasons"]))
        # v12: a reply that thought past the output cap (or lost its JSON) gets one thinking-free retry, as the plan
        # does; without a request or the time for it, the salvaged or code-built reasons stand. v21.13: a salvaged
        # part does not give the retry up; it stands when the retry fails or brings fewer reasons
        if client.remaining >= 1 and time.time() <= client.deadline - margin(client, 30):
            log.warning("reasons reply unparseable (finish=%s); retrying without thinking", rep.finish)
            try:
                rep2 = client.chat(msgs, reasoning_budget=None, thinking=False, temperature=0.3,
                                   seed=config.SEED + 29, max_attempts=2, tag="reasons-nothink")
                obj2 = parse_reply(rep2, "reasons")
                if obj is None or len(obj2.get("reasons") or []) >= len(obj.get("reasons") or []):
                    obj = obj2
            except (ValueError, LLMUnavailable, BudgetExhausted):
                if obj is None:
                    raise
        if obj is None:
            raise ValueError("no JSON object found")
    out: list[dict] = []
    held: list[dict] = []
    unused: list[str] = []  # v13: why a reason of the reply could not be used, for the follow-up

    def consider(it, n: int) -> None:
        if not isinstance(it, dict):
            return
        pid = str(it.get("premise_id") or "").strip().strip("[]")
        if pid not in keyed:
            unused.append(f"reason {n} quoted no listed passage as its premise")
            return
        cid, (premise, cite) = keyed[pid]
        mechanism = clean(_unlabel(_as_text(it.get("mechanism")), drivers), MECH_MAX)
        if not mechanism:
            unused.append(f"reason {n} had no usable mechanism")
            return
        ents, argued = [], {}
        for x in it.get("entities") if isinstance(it.get("entities"), list) else []:
            eid = x if isinstance(x, str) else (x.get("id") or x.get("entity_id")) if isinstance(x, dict) else None
            if eid in spec.entity_ids:
                ents.append(eid)
                if isinstance(x, dict) and isinstance(x.get("argues"), str):
                    argued[eid] = x["argues"]
        implication = clean(_unlabel(_as_text(it.get("answer_implication")), drivers), IMPL_MAX)
        if self_denying(implication) or self_denying(mechanism):
            log.info("reason dropped: it argues against the submitted answers (%s)", (implication or "")[:160])
            unused.append(f"reason {n} argued against the submitted answers")
            return
        citations = [cite]
        win = _window(corpus, cid)
        if win != cite:
            citations.append(win)
        sids = it.get("support_ids") if isinstance(it.get("support_ids"), list) else []
        for sid in sids[:SUPPORT_MAX]:
            sid = str(sid).strip().strip("[]")
            if sid in keyed and sid != pid:
                w = _window(corpus, keyed[sid][0])
                if w not in citations:
                    citations.append(w)
        # v11: a reason speaks only for the entities its citations cover (their own documents, or any document
        # shared by all): v10 once carried JPM's, Citi's and PNC's filings over to GS and MS
        covered = evidence_cover(ev, citations, spec.entity_ids)
        named = _ids_in(implication, spec.entity_ids) if implication else []
        ents = [e for e in ents if e in covered]
        issues = implication_issues(spec, preds, implication) if implication else ["no usable implication"]
        issues += [f"{e} without its own cited evidence" for e in named if e not in covered]
        # v12: the direction the model declares it argues for each entity, against the answer; and whose documents
        # it quotes, against whom it names
        issues += [f"{e} argued {argued[e]}" for e in argued if e in by_id and argued_against(spec, by_id[e], argued[e],
                                                                                              forecaster.plan)]
        text = f"{premise} {mechanism} {implication or ''}"
        unnamed = set(unnamed_sources(ev, citations, text, aliases))
        if unnamed:
            # a citation of an entity's document the reason never names: when the mechanism uses what only that
            # document says, the reason puts a peer's facts on another entity and is held back (the premise's own
            # document counts as used); otherwise the citation is extraneous and is dropped
            owners = {d: {e for e, docs in ev.specific.items() if d in (docs or [])} for d in {c["doc_id"] for c in citations}}
            named_docs = [c for c in citations if not (owners.get(c["doc_id"], set()) & unnamed)]
            own_text = " ".join(corpus.docs[c["doc_id"]].text[c["span_start"]:c["span_end"]] for c in named_docs)
            kept = []
            for i, c in enumerate(citations):
                bad_owner = owners.get(c["doc_id"], set()) & unnamed
                src = corpus.docs[c["doc_id"]].text[c["span_start"]:c["span_end"]]
                if bad_owner and (i < 2 or borrows_from(mechanism, src, own_text)):  # i < 2: the premise and its context
                    issues += [f"{e} quoted unnamed" for e in sorted(bad_owner)]
                elif not bad_owner:
                    kept.append(c)
            if not any(" quoted unnamed" in i for i in issues):
                log.info("reason %d: dropped %d citation(s) of unnamed entities' documents", len(out) + len(held) + 1,
                         len(citations) - len(kept))
                citations = kept
        # a label or direction set against an entity's answer means the reasoning itself points the other way for
        # it, and a peer's filing quoted without its name puts facts on the wrong entity: the reason is held back
        # (and used only if fewer than two others survive); a misquoted number, a missing implication or an entity
        # without evidence only needs the implication rebuilt from the answers
        # v12: an entity the mechanism discusses by name must have its own (or a shared) document among the
        # citations: v12's first sweeps credited Citi's Visa gain and PNC's guidance in reasons citing neither
        issues += [f"{e} discussed without its evidence" for e in spec.entity_ids
                   if e not in covered and names_strongly(mechanism, e, aliases)]
        conflicted = [i for i in issues if " named " in i or " called " in i or " argued " in i or " quoted " in i
                      or " discussed " in i]
        if conflicted:
            unused.append(f"reason {n} was held back: " + "; ".join(conflicted)[:240])
        if issues:
            log.info("reason %d: %s (%s)", len(out) + len(held) + 1,
                     "held back" if conflicted else "implication rebuilt from the answers", "; ".join(issues)[:200])
            bad = ({i.split(" ")[0] for i in conflicted if " quoted " not in i and " discussed " not in i}
                   | {e for e in named if e not in covered})
            keep = [e for e in (ents or named or [e for e in spec.entity_ids if e in covered]) if e not in bad]
            implication = code_implication(spec, preds, keep) if keep else None
        if not implication:
            if not conflicted:
                unused.append(f"reason {n} applied to no entity its cited evidence covers")
            return
        r = {"reason_id": f"r{len(out) + 1}", "premise": premise, "mechanism": mechanism,
             "answer_implication": implication, "citations": citations}
        if ents:
            r["scope"] = {"entities": ents}
        # v21: a mechanism stating figures that neither its citations nor the answers contain is kept only until a
        # follow-up round writes a replacement from the cited figures
        sources = ([premise] + [corpus.docs[c["doc_id"]].text[c["span_start"]:c["span_end"]] for c in citations]
                   + [spec.prompt, json.dumps(spec.entities)])  # the task's own statement and table are visible too
        figs = ungrounded_figures(mechanism, sources, preds)
        if len(figs) >= 2 and not conflicted:  # one figure may be the reason's own arithmetic on cited numbers
            r["_ungrounded"] = figs[:4]
            unused.append(f"reason {n} states figures its cited passages do not contain ({', '.join(figs[:4])}): "
                          "write it again with only the figures the cited passages give")
            log.info("reason %d: figures not in its citations: %s", len(out) + len(held) + 1, ", ".join(figs[:4]))
        (held if conflicted else out).append(r)

    items = [it for it in ((obj.get("reasons") if isinstance(obj, dict) else None) or [])]
    for n, it in enumerate(items, 1):
        consider(it, n)
    # v13: three usable reasons cover more of the hidden target reasons than two. A reply whose reasons were held
    # back or dropped gets a follow-up call, told why, for replacements on other passages (v12 submitted two reasons in
    # four of eleven units after holding one back). v21: up to REASONS_ROUNDS rounds, and a reason with figures its
    # citations lack is replaced when a round writes a clean one (it may reuse that reason's passage).
    def grounded() -> list[dict]:
        return [r for r in out if not r.get("_ungrounded")]

    rounds = 0
    while (len(grounded()) < 3 and items and rounds < config.REASONS_ROUNDS and client.remaining >= 1
           and time.time() < client.deadline - margin(client, config.REASONS_FOLLOWUP_SEC)):
        rounds += 1
        try:
            kept = grounded()
            taken = {k for k, (_, prem) in keyed.items() if prem[0] in {r["premise"] for r in kept}}
            more = follow_up(msgs, rep.content, unused, 3 - len(kept), drivers, kept, client, taken)
        except (LLMUnavailable, BudgetExhausted, ValueError) as exc:
            log.warning("reasons follow-up failed: %s", exc)
            break
        unused.clear()  # the next round is told about this round's items only
        start = len(items) + 1 + 10 * (rounds - 1)
        for n, it in enumerate(more, start):
            if len(grounded()) >= 3:
                break
            pid = str(it.get("premise_id") or "").strip().strip("[]") if isinstance(it, dict) else ""
            used = {r["premise"] for r in grounded() + held}
            if pid in keyed and keyed[pid][1][0] in used:
                log.info("follow-up reason %d skipped: it repeats a premise already used", n)
                continue
            consider(it, n)
        log.info("reasons follow-up round %d: %d clean of %d usable", rounds, len(grounded()), len(out))
    good = grounded()
    soft = [r for r in out if r.get("_ungrounded") and r["premise"] not in {g["premise"] for g in good}]
    chosen = (good + soft)[:3]
    chosen += held[:max(0, 2 - len(chosen))]
    for r in chosen:
        r.pop("_ungrounded", None)
    return chosen


def follow_up(msgs: list[dict], first: str, unused: list[str], need: int, drivers, kept: list[dict],
              client: HouseClient, taken=()) -> list:
    """v13: one more reasons call, after the first reply, for `need` replacements of the reasons that could not be used:
    told why each failed and which drivers the usable ones leave uncovered. Returns the reply's reason items."""
    cov = driver_coverage(list(drivers), kept)
    missing = [f"D{i}: {d}" for i, (d, c) in enumerate(zip(drivers, cov), 1) if not c]
    note = "\n".join(x for x in [
        ("These of your reasons cannot be submitted: " + "; ".join(unused[:4]) + ".") if unused else "",
        f"Write {need} NEW reason{'s' if need > 1 else ''} in the same JSON format to replace them"
        + (f", covering the drivers your usable reasons leave out: {'; '.join(missing)}" if missing else "") + ".",
        "Quote a different passage as each premise than the reasons that remain"
        + (f" (not {', '.join(sorted(taken))})" if taken else "") + "; every reason argues for the submitted "
        "answers, cites each entity's own documents and names every entity whose document it quotes.",
    ] if x)
    convo = msgs + [{"role": "assistant", "content": (first or "")[:8000]}, {"role": "user", "content": note}]
    rep = client.chat(convo, reasoning_budget=config.REASONS_REASONING, temperature=0.3, seed=config.SEED + 31,
                      max_attempts=2, tag="reasons-more")
    obj = parse_reply(rep, "reasons")
    items = obj.get("reasons") if isinstance(obj, dict) else None
    return [it for it in items or [] if isinstance(it, dict)][: need + 1]


def build_reasons(spec: TaskSpec, corpus: Corpus, ev, forecaster, forecasts: dict, preds: list[dict],
                  client: HouseClient | None) -> list[dict]:
    """Validated reasons (possibly empty). Model-written when time and budget allow, else code-built."""
    drivers = unit_drivers(spec, getattr(forecaster, "plan", None))
    reasons: list[dict] = []
    # v21.5: gated on requests and time; v21.3 also required a model forecast, and built code reasons when every
    # analyst failed although House still answered. Without a model forecast the call goes only when the route's
    # typical round trip still fits before the deadline: on a slow route it waited out a certain read timeout
    # (115 s per unit at T=600) for nothing
    typical = getattr(client, "typical_seconds", None) if client is not None else None
    fits = any(f.source == "llm" for f in forecasts.values()) or (
        typical is not None and client.deadline - time.time() - 5 >= typical)
    if (client is not None and config.REASONS and client.enabled and client.remaining >= 1
            and time.time() < client.deadline - margin(client, 25) and fits):
        try:
            reasons = model_reasons(spec, corpus, ev, forecaster, forecasts, preds, client, drivers)
        except (LLMUnavailable, BudgetExhausted, ValueError) as exc:
            log.warning("reasons call failed: %s", exc)
        except Exception as exc:  # noqa: BLE001 - reasons are optional
            log.warning("reasons call crashed: %r", exc)
    ok = validate(spec, corpus, reasons)
    if not ok:
        try:
            ok = validate(spec, corpus, code_reasons(spec, corpus, ev, forecaster, forecasts, preds, drivers))
        except Exception as exc:  # noqa: BLE001 - reasons are optional
            log.warning("code reasons failed: %r", exc)
            ok = []
    for i, r in enumerate(ok, 1):
        r["reason_id"] = f"r{i}"
    if ok and _compact(declared_rows(spec, preds)) > CAP_ANSWER:
        log.info("per-entity answer is over the grader's 3,000-byte cap; reasons would not be judged")
    cov = driver_coverage(drivers, ok)
    focus = focus_entities(spec, preds, getattr(forecaster, "plan", None))
    named = {e for r in ok for e in _ids_in(r["answer_implication"], spec.entity_ids)}
    if focus:
        log.info("reasons explain %d/%d of the answers furthest from the naive baseline%s", sum(e in named for e in focus),
                 len(focus), "" if all(e in named for e in focus) else " (not: " + ", ".join(e for e in focus if e not in named) + ")")
    log.info("reasons: %d (%s); drivers covered %d/%d%s", len(ok),
             "model" if reasons and ok and ok[0]["premise"] == reasons[0]["premise"] else "code", sum(cov), len(cov),
             "" if all(cov) else " (not: " + "; ".join(d for d, c in zip(drivers, cov) if not c)[:160] + ")")
    return ok
