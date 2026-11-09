"""Lexical retrieval (BM25) over offset-exact chunks, plus finance key-line cues for filings."""
from __future__ import annotations

import math
import re
from collections import Counter

from .corpus import Chunk, Corpus
from .task import TaskSpec

TOKEN = re.compile(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*")
STOP = set("""a an and are as at be by for from has have in is it its of on or that the to was were will with this
these those which who what when where why how than then there their they we our us you your per into over under
about after before during such other any all each more most less not no nor only own same so too very can may
might must should would could using use predict prediction predicted corpus frozen evidence cutoff support
citations citation passages provide point forecast interval entity table row rows given do rely absent design
reason""".split())


def tokenize(s: str) -> list[str]:
    return [t for t in TOKEN.findall(s.lower()) if t not in STOP and len(t) > 1]


class BM25:
    def __init__(self, chunks: list[Chunk], k1: float = 1.4, b: float = 0.7):
        self.chunks = chunks
        self.tf = [Counter(tokenize(c.text)) for c in chunks]
        self.lens = [sum(t.values()) or 1 for t in self.tf]
        self.avg = (sum(self.lens) / len(self.lens)) if self.lens else 1.0
        df: Counter = Counter()
        for t in self.tf:
            df.update(t.keys())
        n = len(chunks)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}
        self.k1, self.b = k1, b

    def search(self, query: str, k: int, docs: set[str] | None = None) -> list[tuple[float, Chunk]]:
        q = Counter(tokenize(query))
        if not q:
            return []
        scored: list[tuple[float, Chunk]] = []
        for c, tf, ln in zip(self.chunks, self.tf, self.lens):
            if docs is not None and c.doc_id not in docs:
                continue
            s = 0.0
            for w, _ in q.items():
                f = tf.get(w)
                if f:
                    s += self.idf.get(w, 0.0) * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * ln / self.avg))
            if s > 0:
                scored.append((s, c))
        scored.sort(key=lambda x: -x[0])
        return scored[:k]


# Generic finance cues. They steer evidence toward the lines that move common targets; the task
# prompt and the planner's search terms carry everything family-specific.
KEYLINE = re.compile(
    r"diluted (?:net )?(?:earnings|income|loss) per|earnings per (?:common |diluted )?share|per diluted share|"
    r"net (?:income|earnings|loss)(?: applicable| attributable| available)?|total (?:net )?revenues?|net sales|"
    r"net interest income|provision for credit losses|going concern|substantial doubt|liquidity|covenant|"
    r"default|bankruptcy|chapter 11|restructuring|guidance|outlook|we expect|expects? to|anticipate|"
    r"maturit(?:y|ies)|refinanc|credit facility|revised (?:up|down)|increased|decreased|rose|fell|declined",
    re.IGNORECASE,
)


def entity_strings(ent: dict) -> str:
    return " ".join(str(v) for k, v in ent.items() if isinstance(v, str) and k not in ("corpus_ref",))


def entity_query(spec: TaskSpec, ent: dict, extra_terms: list[str] | None = None) -> str:
    parts = [spec.prompt, spec.target_name.replace("_", " "), entity_strings(ent)]
    if extra_terms:
        parts.append(" ".join(extra_terms))
    return " ".join(parts)


def keyline_chunks(corpus: Corpus, docs: set[str], bm: BM25, query: str, k: int) -> list[Chunk]:
    """Chunks in `docs` that contain a finance cue, ranked by relevance to `query`."""
    ranked = bm.search(query, 4000, docs)
    out = [c for _, c in ranked if KEYLINE.search(c.text)]
    return out[:k]
