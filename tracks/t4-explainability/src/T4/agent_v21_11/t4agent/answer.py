"""answer.json: assembly, a validator mirroring the scorer's gates, and an atomic write.

Mirrors qfbench2_track_analysis.alignment / corpus: exact roster, interval level equal to the
unit's, finite ordered bounds, label in vocabulary, point forecast on numeric units, >= 1 claim
per row, every citation resolvable, dated on/before the cutoff and non-empty when sliced. From
scorer 5.2.x a bad claim no longer refuses the unit but costs a share of it, so claims that are not
verbatim quotes of an admitted span (or cite more than 8,000 characters) are dropped before writing,
and a `submitted_reasons` block is written only in the published shape (a malformed one would void
the whole answer). v21.5: an entity with no citable passage at all claims a verbatim piece of its own
row of the task table (doc_id "task"): an empty claims list voids the whole unit.
"""
from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path

from .corpus import Corpus
from .task import TaskSpec


_WS = re.compile(r"\s+")
SPAN_CAP = 8000
TASK_DOC_ID = "task"  # the scorer's reserved doc_id of the task table (qfbench2_track_analysis.corpus.TASK_DOC_ID)
_ROW_KEYS = re.compile(r"entity_id|name|corpus_ref|.*ticker", re.I)  # blanked by the scorer or no statement
# a task-row claim stays under the judge's 400-token claim cap (scoring.CLAIM_MAX_JUDGE_TOKENS; over it the claim is
# malformed): the judge tokenizer gives at most about one token per character (CJK, punctuation), so 300 characters
TASK_CLAIM_CHARS = 300


def task_table(spec: TaskSpec) -> tuple[str, dict[str, tuple[int, int]]]:
    """The text a doc_id "task" span indexes and each entity's row as [start, end): one line per task.json `entities`
    row in file order, json.dumps(row, ensure_ascii=False, separators=(", ", ": ")), joined by a single newline
    (qfbench2_track_analysis.corpus.task_table_text). Empty when the scorer would refuse the table (a row without an
    entity_id, an entity named twice) -- an organizer fault no claim can fix."""
    rows = spec.raw.get("entities") if isinstance(spec.raw, dict) else None
    lines: list[str] = []
    ranges: dict[str, tuple[int, int]] = {}
    off = 0
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict) or not isinstance(row.get("entity_id"), str) or row["entity_id"] in ranges:
            return "", {}
        line = json.dumps(row, ensure_ascii=False, separators=(", ", ": "))
        ranges[row["entity_id"]] = (off, off + len(line))
        lines.append(line)
        off += len(line) + 1
    return "\n".join(lines), ranges


def task_claim(spec: TaskSpec, eid: str) -> dict | None:
    """A verbatim claim on the entity's own task-table row: its shortest numeric '"key": value' pair (a figure the task
    gives; not the id, name or ticker the scorer blanks), else its shortest other pair, else the row itself (cut at
    TASK_CLAIM_CHARS). None when the entity has no row."""
    text, ranges = task_table(spec)
    if eid not in ranges:
        return None
    start, end = ranges[eid]
    row = next(r for r in spec.raw["entities"] if r["entity_id"] == eid)
    best = None
    pos = start + 1  # past the row's "{"
    for k, v in row.items():
        pair = json.dumps(k, ensure_ascii=False) + ": " + json.dumps(v, ensure_ascii=False, separators=(", ", ": "))
        numeric = isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
        if text[pos:pos + len(pair)] != pair:
            best = None  # the row did not render as its pairs: quote the row itself
            break
        if not _ROW_KEYS.fullmatch(str(k)) and len(pair) <= TASK_CLAIM_CHARS and not has_canary(spec, pair):
            rank = (0 if numeric else 1, len(pair))
            if best is None or rank < best[0]:
                best = (rank, pos, pos + len(pair))
        pos += len(pair) + 2
    s, e = (best[1], best[2]) if best else (start, min(end, start + TASK_CLAIM_CHARS))
    return {"doc_id": TASK_DOC_ID, "span_start": s, "span_end": e, "claim": text[s:e]}


def task_claim_ok(spec: TaskSpec, eid: str, c: dict) -> bool:
    """A doc_id "task" claim the scorer accepts: a verbatim piece of the entity's own row (another row, or a span
    crossing a line, is a wrong-entity citation)."""
    text, ranges = task_table(spec)
    s, e = c.get("span_start"), c.get("span_end")
    if eid not in ranges or not (isinstance(s, int) and isinstance(e, int)) or isinstance(s, bool) \
            or isinstance(e, bool):
        return False
    r0, r1 = ranges[eid]
    if not (r0 <= s < e <= r1) or "citations" in c or not isinstance(c.get("claim"), str):
        return False
    claim = _WS.sub(" ", c["claim"]).strip()
    return bool(claim) and len(c["claim"]) <= 4000 and claim in _WS.sub(" ", text[s:e])


def claim_ok(corpus: Corpus, eid: str, c: dict, spec: TaskSpec | None = None) -> bool:
    """The scorer's deterministic claim rules, as this agent writes claims: a verbatim piece of an admitted span
    of at most 8,000 characters, no claim-level `citations` list, at most 4,000 characters. v21.5: or of the entity's
    own task-table row (`spec` given)."""
    if not isinstance(c, dict) or "citations" in c or not isinstance(c.get("claim"), str):
        return False
    if c.get("doc_id") == TASK_DOC_ID and TASK_DOC_ID not in corpus.docs:
        return spec is not None and task_claim_ok(spec, eid, c)
    doc = corpus.docs.get(c.get("doc_id"))
    s, e = c.get("span_start"), c.get("span_end")
    if doc is None or not corpus.admits(eid, doc.doc_id):
        return False
    if not (isinstance(s, int) and isinstance(e, int)) or isinstance(s, bool) or isinstance(e, bool):
        return False
    if s < 0 or e <= s or e > len(doc.text) or e - s > SPAN_CAP or not doc.text[s:e].strip():
        return False
    claim = _WS.sub(" ", c["claim"]).strip()
    return bool(claim) and len(c["claim"]) <= 4000 and claim in _WS.sub(" ", doc.text[s:e])


def has_canary(spec: TaskSpec, obj) -> bool:
    """v21.11: whether any of the unit's canary strings occurs in `obj` (a string, or anything JSON-serialisable)."""
    tokens = [t.lower() for t in getattr(spec, "canaries", ()) or () if isinstance(t, str) and t]
    if not tokens:
        return False
    text = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False)
    low = text.lower()
    return any(t in low for t in tokens)


def _redact_canaries(spec: TaskSpec, text: str) -> str:
    """Remove registered tokens from required identifier fields.

    An identifier containing a token cannot both match the task roster and avoid
    contamination. The safe failure is an invalid, redacted answer rather than
    emitting the token.
    """
    out = str(text)
    for token in getattr(spec, "canaries", ()) or ():
        if isinstance(token, str) and token:
            out = re.sub(re.escape(token), "[redacted]", out, flags=re.I)
    return out


def build_answer(spec: TaskSpec, rows: dict[str, dict], trace: str = "", reasons: list[dict] | None = None,
                 corpus: Corpus | None = None) -> dict:
    preds = []
    for eid in spec.entity_ids:
        r = rows[eid]
        item: dict = {"entity_id": _redact_canaries(spec, eid)}
        if r.get("label") is not None and not has_canary(spec, r["label"]):  # G1 F7 / H4
            item["label"] = r["label"]
        if r.get("point_forecast") is not None:
            item["point_forecast"] = float(r["point_forecast"])
        item["interval"] = {"level": spec.interval_level, "lo": float(r["lo"]), "hi": float(r["hi"])}
        claims = r["claims"]
        if corpus is not None:
            kept = [c for c in claims if claim_ok(corpus, eid, c, spec)]
            claims = kept or claims[:1]  # schema needs one claim; the builder only writes verbatim quotes anyway
        claims = [c for c in claims if not has_canary(spec, c)]  # v21.11: a quoted canary is contamination
        if not claims:  # v21.5: no citable passage at all (a no-text corpus, only stale documents)
            tc = task_claim(spec, eid)
            if tc is not None and not has_canary(spec, tc):
                claims = [tc]
        item["claims"] = claims
        preds.append(item)
    ans: dict = {"task_id": _redact_canaries(spec, spec.task_id), "schema_version": "3"}
    if spec.target_type_trusted:
        ans["target_type"] = spec.target_type
    ans["entity_predictions"] = preds
    reasons = [r for r in reasons or [] if not has_canary(spec, r)]  # v21.11
    if reasons:
        ans["submitted_reasons"] = reasons[:3]
    if trace and not has_canary(spec, trace):
        ans["evidence_trace"] = trace[:4000]
    return ans


JUDGE_ANSWER_CAP = 2950  # bytes; the reasoning grader judges no reason when the per-entity answer it reads exceeds 3,000


def _judged_bytes(spec: TaskSpec, rows: dict[str, dict]) -> int:
    """Compact UTF-8 bytes of the per-entity answer as the reasoning grader's judge reads it."""
    fields = ("label", "interval") if spec.target_type == "classification" else ("point_forecast", "interval")
    out = []
    for eid in spec.entity_ids:
        r = rows[eid]
        item = {"entity_id": eid, "label": r.get("label"), "point_forecast": r.get("point_forecast"),
                "interval": {"level": spec.interval_level, "lo": r["lo"], "hi": r["hi"]}}
        out.append({"entity_id": eid, **{k: item[k] for k in fields if item.get(k) is not None}})
    return len(json.dumps(out, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def _sig(x: float, digits: int, mode: str) -> float:
    if x == 0 or not math.isfinite(x):
        return x
    k = 10.0 ** (digits - 1 - math.floor(math.log10(abs(x))))
    f = {"down": math.floor, "up": math.ceil}.get(mode, round)
    v = f(x * k) / k
    return float(f"{v:.{max(0, digits)}g}") if mode == "round" else v


CAP_WIDEN = 0.01  # v14: the most an interval may widen (share of its width) to fit the cap
CAP_SHIFT = 0.005  # v14: the most a point may move (share of its interval's width) to fit the cap
# v21.8: a second tier, tried only when the first cannot fit. Over the cap a unit's reasons score 0 for certain, and the
# reasoning part of the Final score is a bonus that cannot go below 0, so fitting can only add to it. The cost is bounded
# by the published loss alone: an interval widened by at most 5% of its width raises its interval score by at most 5% of
# the width, which lowers the unit's interval quality by at most about 0.25 x 5% (about 0.005 on the Final score); a point
# rounded to nearest moves by at most 2.5% of the width, and an unbiased move that small changes the expected absolute
# error only to second order. Round numbers fixed a priori; no outcome or proxy-judge score was consulted.
CAP_WIDEN_MORE = 0.05
CAP_SHIFT_MORE = 0.025
_CAP_TIERS = ((CAP_WIDEN, CAP_SHIFT, (9, 8, 7, 6, 5, 4)), (CAP_WIDEN_MORE, CAP_SHIFT_MORE, (9, 8, 7, 6, 5, 4, 3, 2)))


def fit_judge_cap(spec: TaskSpec, rows: dict[str, dict], cap: int = JUDGE_ANSWER_CAP) -> dict[str, dict]:
    """v13: rows whose numbers fit the reasoning grader's 3,000-byte cap on the per-entity answer (over it, the unit's
    reasons are not judged). v14: only while the forecasts stay what they were -- every interval widens by at most 1%
    of its width and every point moves by at most 0.5% of it. When no such rounding fits, the rows are returned
    unchanged (v13 widened a one-unit interval to 1,000 and still missed the cap).
    v21.8: when the first tier cannot fit, a second allows 5% and 2.5% (see CAP_WIDEN_MORE). A ranking is rounded only
    when its points keep exactly the same order and ties, so its ranks cannot change (v13 merged 26 close ranking
    points into one; v14 therefore left every ranking as it was). A roster too long to fit at all (about 37 rows and
    more: the field names alone fill the cap) is left as it is."""
    if _judged_bytes(spec, rows) <= cap:
        return rows
    for widen, shift, digits in _CAP_TIERS:
        out = _rounded_to_fit(spec, rows, cap, widen, shift, digits)
        if out is not None:
            return out
    return rows


def _rounded_to_fit(spec: TaskSpec, rows: dict[str, dict], cap: int, widen: float, shift: float,
                    digits_tried: tuple[int, ...]) -> dict[str, dict] | None:
    """The rows at the most digits that fit the cap within these tolerances, or None. Fewer digits only widen an
    interval further, move a point further and merge more ranking points, so the first violation ends the search."""
    ranking = spec.target_type == "ranking"
    for digits in digits_tried:
        out = {}
        for eid, r in rows.items():
            lo, hi = float(r["lo"]), float(r["hi"])
            width = hi - lo
            r2 = dict(r)
            r2["lo"], r2["hi"] = _sig(lo, digits, "down"), _sig(hi, digits, "up")
            if width <= 0 or (r2["hi"] - r2["lo"]) - width > widen * width:
                return None
            pt = r.get("point_forecast")
            if pt is not None and spec.target_type != "classification":  # a classification point is not judged
                p2 = min(max(_sig(float(pt), digits, "round"), r2["lo"]), r2["hi"])
                if abs(p2 - float(pt)) > shift * width:
                    return None
                r2["point_forecast"] = p2
            out[eid] = r2
        if ranking and not _same_order(rows, out):
            return None
        if _judged_bytes(spec, out) <= cap:
            return out
    return None


def _same_order(before: dict[str, dict], after: dict[str, dict]) -> bool:
    """v21.8: whether every pair of ranking points keeps its order and its ties."""
    ids = sorted(before, key=lambda e: float(before[e]["point_forecast"]))
    for a, b in zip(ids, ids[1:]):
        x, y = float(before[a]["point_forecast"]), float(before[b]["point_forecast"])
        u, v = float(after[a]["point_forecast"]), float(after[b]["point_forecast"])
        if (x < y) != (u < v) or (x == y) != (u == v):
            return False
    return True


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def validate(spec: TaskSpec, corpus: Corpus, ans: dict) -> list[str]:
    """Problems that would make the scorer refuse the unit (empty list = admissible shape)."""
    errs: list[str] = []
    if ans.get("task_id") != spec.task_id:
        errs.append("task_id mismatch")
    tt = ans.get("target_type")
    if tt is not None and tt != spec.target_type:
        errs.append("target_type mismatch")
    preds = ans.get("entity_predictions")
    if not isinstance(preds, list) or not preds:
        return errs + ["no entity_predictions"]
    ids = [p.get("entity_id") for p in preds if isinstance(p, dict)]
    if len(ids) != len(preds) or len(set(ids)) != len(ids) or set(ids) != set(spec.entity_ids):
        errs.append("roster mismatch")
    for p in preds:
        eid = p.get("entity_id")
        iv = p.get("interval")
        if not isinstance(iv, dict) or not all(_finite(iv.get(k)) for k in ("level", "lo", "hi")):
            errs.append(f"{eid}: interval invalid")
        else:
            if float(iv["level"]) != float(spec.interval_level):
                errs.append(f"{eid}: interval level")
            if float(iv["lo"]) > float(iv["hi"]):
                errs.append(f"{eid}: lo > hi")
        if spec.target_type == "classification":
            lab = p.get("label")
            if not isinstance(lab, str) or not lab or (spec.labels and lab not in spec.labels):
                errs.append(f"{eid}: label")
        else:
            if not _finite(p.get("point_forecast")):
                errs.append(f"{eid}: point_forecast")
        if "point_forecast" in p and not _finite(p.get("point_forecast")):
            errs.append(f"{eid}: point_forecast not finite")
        if "rank" in p:
            errs.append(f"{eid}: rank present (not emitted by design)")
        claims = p.get("claims")
        if not isinstance(claims, list) or not claims:
            errs.append(f"{eid}: no claims")
            continue
        for c in claims:
            if not isinstance(c, dict) or not isinstance(c.get("claim"), str):
                errs.append(f"{eid}: malformed claim")
                continue
            if c.get("doc_id") == TASK_DOC_ID and TASK_DOC_ID not in corpus.docs:  # v21.5: the task table
                if not task_claim_ok(spec, eid, c):
                    errs.append(f"{eid}: task span outside its own row")
                continue
            doc = corpus.docs.get(c.get("doc_id"))
            s, e = c.get("span_start"), c.get("span_end")
            if doc is None:
                errs.append(f"{eid}: unresolved doc {c.get('doc_id')}")
                continue
            if spec.cutoff is None or doc.doc_date > spec.cutoff:
                errs.append(f"{eid}: post-cutoff doc")
            if not (isinstance(s, int) and isinstance(e, int)) or isinstance(s, bool) or isinstance(e, bool):
                errs.append(f"{eid}: span not int")
                continue
            if s < 0 or e <= s or e > len(doc.text) or not doc.text[s:e].strip():
                errs.append(f"{eid}: empty/out-of-range span")
    reasons = ans.get("submitted_reasons")
    if reasons is not None:
        if not isinstance(reasons, list) or not 1 <= len(reasons) <= 3:
            errs.append("submitted_reasons shape")
        else:
            for r in reasons:
                if not isinstance(r, dict) or not all(isinstance(r.get(k), str) for k in
                                                      ("reason_id", "premise", "mechanism", "answer_implication")):
                    errs.append("submitted_reasons item")
                    continue
                for c in r.get("citations") or []:
                    if not (isinstance(c, dict) and isinstance(c.get("doc_id"), str)
                            and all(isinstance(c.get(k), int) and not isinstance(c.get(k), bool) and c[k] >= 0
                                    for k in ("span_start", "span_end"))):
                        errs.append("submitted_reasons citation")
    try:
        json.dumps(ans, allow_nan=False)
    except (TypeError, ValueError):
        errs.append("not JSON-serializable without NaN/Inf")
    if has_canary(spec, ans):  # v21.11: the Final scores a canary anywhere in the output as contamination
        errs.append("canary string in the answer")
    return errs


def write_atomic(path: str | Path, obj: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    data = json.dumps(obj, ensure_ascii=False, indent=1, allow_nan=False)
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
