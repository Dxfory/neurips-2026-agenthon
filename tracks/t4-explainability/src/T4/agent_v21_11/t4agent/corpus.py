"""Corpus: citable documents only, strictly on or before the cutoff, with offset-exact chunks.

A citation resolves only to a document the unit manifest declares with role "corpus"; its text is
the flat `text` field, else `spans[].text` joined with one space (the scorer's `_doc_text`). Any
other document shape renders as an empty premise and is never cited. v21.5: the scorer slices a
citation out of the document's `text` field only, so a document without a non-empty string `text`
(spans only) is read as evidence but never cited.
"""
from __future__ import annotations

import datetime as dt
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .config import CHUNK_MAX
from .task import TaskSpec, parse_date

DOC_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
MAX_DOC_BYTES = 8 * 1024 * 1024


@dataclass
class Doc:
    doc_id: str
    doc_date: dt.date
    text: str
    meta: dict

    @property
    def title(self) -> str:
        t = self.meta.get("title")
        return t if isinstance(t, str) else ""


@dataclass
class Chunk:
    cid: int
    doc_id: str
    start: int
    end: int
    text: str


@dataclass
class Corpus:
    docs: dict[str, Doc]
    chunks: list[Chunk]
    excluded: dict[str, str] = field(default_factory=dict)  # doc_id -> reason (post_cutoff, undated, ...)
    # The manifest's entity labels (scorer 5.2.x): doc_id -> (entity_ids or None, shared). A claim may cite a
    # document only when it is shared or lists the claim's entity; empty when the manifest carries no labels.
    labels: dict[str, tuple[tuple[str, ...] | None, bool]] = field(default_factory=dict)
    # v21.5: documents without a non-empty string `text` field: every claim span of one is out of range for the scorer
    # (scoring._span_text reads `text` only). Claims only: the reasons rail resolves the spans joined by a space
    # (citation_rail._doc_text, as _doc_text here), so a reason may still cite one
    nocite: set[str] = field(default_factory=set)

    def admits(self, eid: str, doc_id: str) -> bool:
        """May entity `eid` cite `doc_id`? By the manifest's labels when there are any (the scorer's rule)."""
        if doc_id in self.nocite:
            return False
        if not self.labels:
            return doc_id in self.docs
        ids, shared = self.labels.get(doc_id, (None, False))
        return doc_id in self.docs and (shared or (ids is not None and eid in ids))

    def chunk(self, cid: int) -> Chunk | None:
        if 0 <= cid < len(self.chunks):
            return self.chunks[cid]
        return None

    def doc_chunks(self, doc_id: str) -> list[Chunk]:
        return [c for c in self.chunks if c.doc_id == doc_id]


def _doc_text(raw: dict) -> str:
    if isinstance(raw.get("text"), str):
        return raw["text"]
    if isinstance(raw.get("spans"), list):
        return " ".join(sp.get("text", "") for sp in raw["spans"] if isinstance(sp, dict))
    return ""


def _manifest_entries(cands) -> list[dict] | None:
    for cand in cands:
        try:
            if cand.is_symlink() or not cand.is_file():
                continue
            man = json.loads(cand.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            continue
        files = man.get("files") if isinstance(man, dict) else None
        if isinstance(files, list):
            return [f for f in files if isinstance(f, dict)]
    return None


def manifest_labels(unit_dir: Path | None, corpus_dir: Path | None) -> dict[str, tuple[tuple[str, ...] | None, bool]]:
    """doc_id -> (entity_ids, shared) from the manifest's corpus entries (corpus/manifest.json is mounted
    with the corpus; the unit's root manifest is read when present). Unlabelled entries are left out. v21.5: in the
    scorer's order (`CorpusIndex._read_manifest`): the root manifest first, then corpus/manifest.json."""
    cands = []
    if unit_dir is not None:
        cands += [unit_dir / "manifest.json", unit_dir / "corpus" / "manifest.json"]
    if corpus_dir is not None:
        cands.append(corpus_dir / "manifest.json")
    out: dict[str, tuple[tuple[str, ...] | None, bool]] = {}
    for entry in _manifest_entries(cands) or []:
        if entry.get("role") != "corpus":
            continue
        path = entry.get("path")
        if not isinstance(path, str) or not path.endswith(".json"):
            continue
        did = path.split("/")[-1][: -len(".json")]
        raw_ids, shared = entry.get("entity_ids"), entry.get("shared") is True
        ids = tuple(x for x in raw_ids if isinstance(x, str)) if isinstance(raw_ids, list) else None
        if ids is None and not shared:
            continue
        out[did] = (ids, shared)
    return out


def declared_corpus_ids(unit_dir: Path | None) -> set[str] | None:
    """doc_ids the unit manifest declares (same lookup order as the scorer); None if unknown."""
    if unit_dir is None:
        return None
    for cand in (unit_dir / "manifest.json", unit_dir / "corpus" / "manifest.json"):
        try:
            if cand.is_symlink() or not cand.is_file():
                continue
            man = json.loads(cand.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            continue
        files = man.get("files") if isinstance(man, dict) else None
        if not isinstance(files, list):
            return None
        ids: set[str] = set()
        for entry in files:
            if not isinstance(entry, dict) or entry.get("role") != "corpus":
                continue
            path = entry.get("path")
            if not isinstance(path, str) or path == "corpus/manifest.json":
                continue
            parts = path.split("/")
            if len(parts) == 2 and parts[0] == "corpus" and parts[1].endswith(".json"):
                did = parts[1][: -len(".json")]
                if DOC_ID_RE.match(did):
                    ids.add(did)
        return ids
    return None


_BOUND = re.compile(r"\n+|(?<=[.!?])\s+(?=[A-Z(\"'\[])|\s{3,}")


def chunk_text(text: str, max_len: int = CHUNK_MAX, min_len: int = 12) -> list[tuple[int, int]]:
    """Line / sentence pieces with exact offsets; over-long pieces are cut at whitespace."""
    pieces = []
    pos = 0
    for m in _BOUND.finditer(text):
        pieces.append((pos, m.start()))
        pos = m.end()
    pieces.append((pos, len(text)))
    out: list[tuple[int, int]] = []
    for s, e in pieces:
        while s < e and text[s].isspace():
            s += 1
        while e > s and text[e - 1].isspace():
            e -= 1
        while e - s > max_len:
            cut = text.rfind(" ", s + max_len // 2, s + max_len)
            if cut <= s:
                cut = s + max_len
            out.append((s, cut))
            s = cut
            while s < e and text[s].isspace():
                s += 1
        if e - s >= min_len:
            out.append((s, e))
    return out


def para_start(text: str, pos: int, max_back: int = 1200) -> int:
    """Start of the paragraph (or line) containing `pos`; `pos` itself if none is close."""
    k = text.rfind("\n\n", 0, pos)
    if k >= 0 and pos - k <= max_back:
        s = k + 2
    else:
        k = text.rfind("\n", 0, pos)
        if k < 0 or pos - k > max_back:
            return pos
        s = k + 1
    while s < pos and text[s].isspace():
        s += 1
    return s


def load_corpus(corpus_dir: str | Path, spec: TaskSpec) -> Corpus:
    corpus_dir = Path(corpus_dir)
    declared = declared_corpus_ids(spec.unit_dir)
    docs: dict[str, Doc] = {}
    excluded: dict[str, str] = {}
    nocite: set[str] = set()
    try:
        files = sorted(p for p in corpus_dir.iterdir() if p.suffix == ".json")
    except OSError:
        files = []
    for path in files:
        did = path.stem
        if path.name == "manifest.json" or not DOC_ID_RE.match(did):
            continue
        if declared is not None and did not in declared:
            excluded[did] = "undeclared"
            continue
        try:
            if path.is_symlink() or path.stat().st_size > MAX_DOC_BYTES:
                excluded[did] = "unreadable"
                continue
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            excluded[did] = "unreadable"
            continue
        if not isinstance(raw, dict):
            excluded[did] = "not_object"
            continue
        ddate = parse_date(raw.get("doc_date"))
        if ddate is None:
            excluded[did] = "undated"
            continue
        if spec.cutoff is None or ddate > spec.cutoff:
            excluded[did] = "post_cutoff" if spec.cutoff else "no_cutoff"
            continue
        text = _doc_text(raw)
        if not text.strip():
            excluded[did] = "empty"
            continue
        docs[did] = Doc(did, ddate, text, {k: v for k, v in raw.items() if k not in ("text", "spans")})
        if not (isinstance(raw.get("text"), str) and raw["text"].strip()):
            nocite.add(did)
    chunks: list[Chunk] = []
    for did in sorted(docs):
        t = docs[did].text
        for s, e in chunk_text(t):
            chunks.append(Chunk(len(chunks), did, s, e, t[s:e]))
    try:
        labels = manifest_labels(spec.unit_dir, corpus_dir)
    except Exception:  # noqa: BLE001 - labels refine citations; they must never break loading
        labels = {}
    # labels count only when they cover every loaded document: with a partial labelling the unlabelled
    # documents would become uncitable, so the name-matching map is used instead
    labels = {d: labels[d] for d in docs} if labels and all(d in labels for d in docs) else {}
    return Corpus(docs, chunks, excluded, labels, nocite)


# --------------------------------------------------------------------------------------------- #
# Entity -> document mapping                                                                     #
# --------------------------------------------------------------------------------------------- #
_GENERIC = set("""the of and inc corp corporation company co group holdings plc ltd llc sa nv u.s us usa
treasury constant maturity yield futures index total all items less food energy note bond auction week
year month day rate new series level usd share shares per eps change pct percent""".split())


def _identifiers(ent: dict) -> list[tuple[str, int]]:
    """(identifier, strength): full entity id 3; ticker / series code / name 2; id fragment 1."""
    out: list[tuple[str, int]] = []
    eid = ent.get("entity_id", "")
    if isinstance(eid, str) and eid:
        out.append((eid.lower(), 3))
        for part in re.split(r"[_\-\s/]+", eid):
            p = part.lower()
            mixed = len(p) >= 2 and any(ch.isdigit() for ch in p) and any(ch.isalpha() for ch in p)
            if (len(p) >= 3 or mixed) and p not in _GENERIC and not p.isdigit():
                out.append((p, 1))
    for key in ("ticker", "series_id", "series_fred", "symbol", "cusip"):
        v = ent.get(key)
        if isinstance(v, str) and len(v) >= 2:
            out.append((v.lower(), 2))
    for key in ("name", "series_name"):
        v = ent.get(key)
        if isinstance(v, str) and v:
            out.append((v.lower(), 2))
    best: dict[str, int] = {}
    for ident, s in out:
        best[ident] = max(s, best.get(ident, 0))
    return list(best.items())


def _doc_header(doc: Doc) -> str:
    meta = " ".join(str(doc.meta.get(k, "")) for k in ("title", "ticker", "series_id", "form_type", "doc_type"))
    return f"{doc.doc_id} {meta} {doc.text[:2000]}".lower()


def _contains(hay: str, needle: str) -> bool:
    if len(needle) <= 4:
        return re.search(r"(?<![a-z0-9])" + re.escape(needle) + r"(?![a-z0-9])", hay) is not None
    return needle in hay


def map_entities(spec: TaskSpec, corpus: Corpus) -> tuple[dict[str, list[str]], list[str]]:
    """Per entity: its own documents; plus the shared documents (nobody's own).

    Match strength per (entity, document): same CIK 4; full entity id 3; ticker / series code /
    name 2; an id fragment 1 -- found in the doc id, metadata or opening text. Each entity keeps
    only its strongest matches (a fragment such as "nymex" must not hand the natural-gas table to
    the crude-oil row), and a document that is the best match of more than half the roster is
    shared context rather than one entity's document.
    """
    if corpus.labels:
        # the organizer's labels decide which documents are an entity's own (and so citable for it)
        lab_specific: dict[str, list[str]] = {e["entity_id"]: [] for e in spec.entities}
        lab_shared: list[str] = []
        for did in sorted(corpus.docs):
            ids, is_shared = corpus.labels[did]
            if is_shared:
                lab_shared.append(did)
            for eid in ids or ():
                if eid in lab_specific:
                    lab_specific[eid].append(did)
        return lab_specific, lab_shared
    n = max(1, len(spec.entities))
    strength: dict[str, dict[str, int]] = {}
    for ent in spec.entities:
        eid = ent["entity_id"]
        cik = str(ent.get("cik") or "").lstrip("0")
        ids = _identifiers(ent)
        row: dict[str, int] = {}
        for did, doc in corpus.docs.items():
            dcik = str(doc.meta.get("cik") or "").lstrip("0")
            m = re.match(r"EDGAR_0*(\d+)_", did)
            if m and not dcik:
                dcik = m.group(1)
            if cik and dcik:
                if cik == dcik:
                    row[did] = 4
                continue
            hdr = _doc_header(doc)
            s = max((st for ident, st in ids if _contains(hdr, ident)), default=0)
            if s:
                row[did] = s
        strength[eid] = row
    owners: dict[str, set[str]] = {d: set() for d in corpus.docs}
    for eid, row in strength.items():
        if row:
            top = max(row.values())
            for did, s in row.items():
                if s == top:
                    owners[did].add(eid)
    specific: dict[str, list[str]] = {e["entity_id"]: [] for e in spec.entities}
    shared: list[str] = []
    for did, who in owners.items():
        if who and (len(who) <= max(1, n // 2) or n <= 2):
            for eid in who:
                specific[eid].append(did)
        else:
            shared.append(did)
    return specific, sorted(shared)
