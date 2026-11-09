"""Deterministic key-fact retrieval for company filings.

Lexical retrieval ranks by the words in the task prompt, so a prompt that says "EPS" can miss the
filing line that says "Diluted earnings per common share ... $ 8.62". This module always surfaces,
from the entity's own documents, the few lines that decide most filing-based targets: per-share
earnings, net income, revenue, forward-looking guidance and distress/liquidity signals. The chunks
it returns are ordinary corpus chunks (citable, offset-exact), ranked by relevance to the entity
and by how much quantitative content they carry.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .corpus import Chunk, Corpus
from .retrieval import BM25

CATEGORIES: list[tuple[str, re.Pattern, int]] = [
    ("per-share earnings", re.compile(
        r"(?:diluted|basic)?\s*(?:net\s+)?(?:earnings|income|loss)\s+(?:\(loss\)\s+)?per\s+(?:common\s+|diluted\s+)?share"
        r"|per\s+diluted\s+share|diluted\s+EPS|\bEPS\b", re.I), 5),
    ("net income", re.compile(r"net\s+(?:income|earnings|loss)(?:\s+(?:applicable|attributable|available))?", re.I), 3),
    ("revenue", re.compile(r"(?:total\s+)?(?:net\s+)?revenues?\b|net\s+sales|net\s+interest\s+income", re.I), 3),
    ("guidance / outlook", re.compile(
        r"\bguidance\b|\boutlook\b|we\s+expect|expects?\s+to|anticipat|\bforecast|for\s+the\s+(?:full\s+)?(?:fiscal\s+)?year", re.I), 3),
    ("distress / liquidity", re.compile(
        r"going\s+concern|substantial\s+doubt|forbearance|waiver|event\s+of\s+default|chapter\s+11|bankruptcy|"
        r"restructuring\s+support|covenant|liquidity|maturit(?:y|ies)|refinanc", re.I), 4),
]
_NUM = re.compile(r"\$\s?\(?\d|\d+(?:\.\d+)?\s?%|\d{1,3}(?:,\d{3})+|\d+\.\d+")

RISK_INDICATORS: list[tuple[str, re.Pattern]] = [
    ("going-concern / substantial doubt", re.compile(r"going\s+concern|substantial\s+doubt", re.I)),
    ("covenant waiver / amendment / forbearance", re.compile(
        r"(?:covenant|credit\s+agreement)[^.]{0,80}(?:waiver|amend)|forbearance", re.I)),
    ("default / event of default", re.compile(r"event\s+of\s+default|payment\s+default|in\s+default|defaulted", re.I)),
    ("restructuring / bankruptcy / chapter 11", re.compile(
        r"chapter\s+11|bankruptcy|restructuring\s+(?:support|advisor|plan)|debt\s+exchange", re.I)),
    ("delisting / listing standards", re.compile(r"delist|continued\s+listing\s+standard", re.I)),
    ("net loss", re.compile(r"\bnet\s+loss(?:es)?\b", re.I)),
    ("impairment / write-down", re.compile(r"impairment|write-?down", re.I)),
]


def risk_profile(corpus: Corpus, docs: set[str]) -> str:
    """Phrase counts of hard distress indicators in the entity's own documents (a feature, not a verdict)."""
    text = " ".join(corpus.docs[d].text for d in docs if d in corpus.docs)
    if len(text) < 20000:  # data tables and short notes: not a filing
        return ""
    parts = [f"{name}: {len(pat.findall(text))}" for name, pat in RISK_INDICATORS]
    return "; ".join(parts)


def _numeric_density(text: str) -> float:
    return min(1.0, len(_NUM.findall(text)) / 6.0)


def key_fact_chunks(corpus: Corpus, bm: BM25, docs: set[str], query: str) -> list[tuple[str, Chunk]]:
    """(category, chunk) pairs from `docs`, best first within each category, categories in order."""
    if not docs:
        return []
    rel = {c.cid: s for s, c in bm.search(query, 6000, docs)}
    out: list[tuple[str, Chunk]] = []
    seen: set[int] = set()
    for name, pat, k in CATEGORIES:
        cands = [c for c in corpus.chunks if c.doc_id in docs and c.cid not in seen and pat.search(c.text)]
        if not cands:
            continue
        top = max(rel.values(), default=1.0) or 1.0
        cands.sort(key=lambda c: -(0.6 * rel.get(c.cid, 0.0) / top + 0.4 * _numeric_density(c.text)))
        for c in cands[:k]:
            seen.add(c.cid)
            out.append((name, c))
    return out


# ---------------------------------------------------------------------------------------------- #
# Like-for-like per-share earnings, computed from a filing's own income-statement tables          #
# ---------------------------------------------------------------------------------------------- #
# A model handed "Q1 EPS 1.02" and "prior-year Q2 EPS 1.53" compares the two and concludes earnings
# fell; the filing's own table says the same quarter rose 39.7%. These lines state, per filing, the
# reported period against the same period a year earlier, with periods named. Columns are mapped by
# the table's own header dates (filers differ: some list the prior year first, some show quarterly
# trend or per-share-class columns); when the mapping is not unambiguous nothing is stated.
_MONTHS = ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec")
_SEP = r"[\s​\xa0$]*"
_VAL = r"\(\s*\$?\s*\d{0,3}(?:,\d{3})*\.\d{2,3}\s*\)|-?\d{0,3}(?:,\d{3})*\.\d{2,3}"
_ROW = re.compile(rf"(?:{_SEP}(?:{_VAL})(?![\d%]|\.\d)){{2,8}}")
_EPS_LABEL = re.compile(
    r"diluted\s+EPS|assuming\s+dilution|per\s+diluted\s+share"
    r"|(?:earnings|income|loss)[^$\d]{0,40}?per\s+(?:common\s+)?share[^$\d]{0,30}?dilut\w*"
    r"|dilut\w*[^$\d]{0,40}?(?:earnings|income|loss)[^$\d]{0,25}?per\s+(?:common\s+)?share"
    r"|(?<=[\s​\xa0$\d)])diluted\b", re.I)  # last form: "Earnings per share: Basic $ .. Diluted $ .."
SUBLABEL_SPAN = 400
# v21.13: a hyphen right before a digit or a point is the figure's minus sign, not a separator ("-0.40" was read +0.40)
_SUBLABEL = re.compile(r"(?:[\s​\xa0:—–]|-(?![\d.]))*(?:\(\d\)[\s​\xa0]*)?(?:[A-Za-z][A-Za-z ,'’()-]{0,45}?)?[\s​\xa0:]*(?=[$(\d.-])")
_DATE = re.compile(r"(?:\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+(?:\d{1,2},?\s+)?)?\b((?:19|20)\d{2})\b", re.I)
_PERIOD_IN_TEXT = re.compile(r"(?:quarterly\s+period|fiscal\s+year|three\s+months|quarter)\s+ended\s+"
                             r"([A-Za-z]+\.?\s+\d{1,2},\s+\d{4})", re.I)
_YTD = re.compile(r"(six|nine|twenty-six|thirty-nine)\s+(months|weeks)", re.I)


def _num(tok: str) -> float:
    t = tok.strip()
    v = float(re.sub(r"[^\d.]", "", t))
    return -v if t.startswith("(") or t.startswith("-") else v


def _period(doc) -> tuple[int, int] | None:
    raw = str(doc.meta.get("period_of_report") or "")
    m = re.match(r"(\d{4})-(\d{2})-\d{2}$", raw)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = _PERIOD_IN_TEXT.search(doc.text[:8000])
    if m:
        d = _DATE.search(m.group(1))
        if d and d.group(1):
            return int(d.group(2)), _MONTHS.index(d.group(1).lower()[:3]) + 1
    return None


def _header(text: str, end: int, year: int) -> tuple[list[tuple[int | None, int]], int] | None:
    """The last run of >= 2 column dates before `end` (plausible years only), as (month, year) pairs."""
    window_start = max(0, end - 6000)
    toks = [(m.start(), m.end(), m.group(1), int(m.group(2))) for m in _DATE.finditer(text, window_start, end)]
    runs: list[list[tuple]] = []
    for t in toks:
        if runs and t[0] - runs[-1][-1][1] <= 45 and not re.search(r"\d", text[runs[-1][-1][1]:t[0]]):
            runs[-1].append(t)
        else:
            runs.append([t])
    for run in reversed(runs):
        if len(run) >= 2 and all(year - 3 <= t[3] <= year + 1 for t in run):
            cols = [(_MONTHS.index(t[2].lower()[:3]) + 1 if t[2] else None, t[3]) for t in run]
            return cols, run[0][0]
        if len(run) >= 2:
            return None  # the nearest date run is not this table's header: do not guess
    return None


def _map_columns(cols: list[tuple[int | None, int]], year: int, month: int,
                 kind: str) -> tuple[int, int, tuple[int, int] | None, int | None] | None:
    """(current, same period a year earlier, (ytd current, ytd prior) or None, a previous quarter or None).

    With full header dates columns are matched by date (the prior year may end in a neighbouring month
    under a 52/53-week calendar). With years only, only the standard layouts are accepted: a quarter
    pair, optionally followed by a year-to-date pair, in either year order; for a 10-K, fiscal years.
    """
    if all(mo is not None for mo, _ in cols):
        near = [i for i, (mo, y) in enumerate(cols) if y == year - 1 and min((mo - month) % 12, (month - mo) % 12) <= 1]
        cur = [i for i, (mo, y) in enumerate(cols) if y == year and mo == month]
        if not cur or not near:
            return None
        ytd = (cur[1], near[1]) if len(cur) > 1 and len(near) > 1 else None
        prev = next((i for i, (mo, y) in enumerate(cols) if i not in cur + near and y >= year - 1), None)
        return cur[0], near[0], ytd, prev
    y = year
    layouts: dict[tuple, tuple] = {(y, y - 1): (0, 1, None), (y - 1, y): (1, 0, None),
                                   (y, y - 1, y, y - 1): (0, 1, (2, 3)), (y - 1, y, y - 1, y): (1, 0, (3, 2))}
    if kind == "10-K":
        layouts.update({(y, y - 1, y - 2): (0, 1, None), (y - 2, y - 1, y): (2, 1, None)})
    hit = layouts.get(tuple(yr for _, yr in cols))
    return (hit[0], hit[1], hit[2], None) if hit else None


def _pct(a: float, b: float) -> str:
    return f"{(a / b - 1) * 100:+.1f}%" if b > 0 else "change not meaningful: base <= 0"


@dataclass
class EpsRecord:
    """Diluted EPS from one filing's own table: the reported period and the same period a year earlier."""
    doc_id: str
    kind: str  # "10-Q" (quarter) or "10-K" (fiscal year)
    year: int
    month: int
    cur: float
    prior: float
    ytd: tuple[float, float] | None  # (this year, a year earlier) for a 10-Q with year-to-date columns
    ytd_name: str
    prev: tuple[int, int, float] | None  # (year, month, value) of a previous quarter shown in the same table
    label: str

    def line(self) -> str:
        end = f"{self.year}-{self.month:02d}"
        span = "quarter" if self.kind == "10-Q" else "fiscal year"
        parts = [f"{span} ending {end}: {self.cur:.2f} vs {self.prior:.2f} in the same {span} a year earlier "
                 f"({_pct(self.cur, self.prior)})"]
        if self.ytd:
            a, b = self.ytd
            parts.append(f"{self.ytd_name} ending {end}: {a:.2f} vs {b:.2f} ({_pct(a, b)}), so the earlier quarter(s) "
                         f"of this fiscal year total about {a - self.cur:.2f} vs {b - self.prior:.2f} a year before")
        if self.prev:
            y, mo, v = self.prev
            parts.append(f"quarter ending {y}-{mo:02d}: {v:.2f}")
        return (f"LIKE-FOR-LIKE (computed by code from the row '{self.label[:60]}' in {self.doc_id}, a {self.kind} for "
                f"the period ending {end}): " + "; ".join(parts) + ". These are reported periods; compare the target "
                "period with the same period a year earlier, never with a different quarter.")


def eps_record(doc_id: str, doc) -> EpsRecord | None:
    """The filing's diluted-EPS row, its columns mapped by the table's own header dates, or None."""
    form = str(doc.meta.get("form_type") or doc.meta.get("title") or "")
    kind = "10-K" if "10-K" in form else "10-Q" if "10-Q" in form else None
    per = _period(doc)
    if kind is None or per is None:
        return None
    year, month = per
    t = doc.text
    for m in _EPS_LABEL.finditer(t):
        if m.group(0).lower() == "diluted" and not re.search(r"per\s+(?:common\s+)?share|EPS", t[max(0, m.start() - 250):m.start()], re.I):
            continue
        # v21.5: the sub-label is read within SUBLABEL_SPAN characters: its two whitespace runs overlap, so on a long
        # run of separators the match backtracks quadratically (1.1 s on 16,000 spaces); the longest match in the public
        # filings is 66 characters
        sub = _SUBLABEL.match(t, m.end(), m.end() + SUBLABEL_SPAN)
        pos = sub.end() if sub else m.end()
        row = _ROW.match(t, pos)
        if not row:
            continue
        vals = [_num(v) for v in re.findall(_VAL, row.group(0))]
        hdr = _header(t, m.start(), year)
        if hdr is None:
            continue
        cols, hstart = hdr
        if len(vals) == 2 * len(cols) and all(vals[2 * i] == vals[2 * i + 1] for i in range(len(cols))):
            vals = vals[::2]  # one column per share class with identical EPS
        if len(vals) > len(cols):
            continue
        mapped = _map_columns(cols[:len(vals)], year, month, kind)
        if mapped is None or abs(vals[mapped[0]]) > 1000:
            continue
        ci, pi, ytd_cols, prev = mapped
        ytd, ytd_name, prev_q = None, "year to date", None
        if kind == "10-Q" and ytd_cols:
            ytd = (vals[ytd_cols[0]], vals[ytd_cols[1]])
            found = _YTD.search(t, max(0, hstart - 400), m.start())
            if found:
                ytd_name = f"{found.group(1).lower()} {found.group(2).lower()}"
        if kind == "10-Q" and prev is not None:
            mo, y = cols[prev]
            prev_q = (y, mo, vals[prev])
        label = re.sub(r"[\s\u200b\xa0]+", " ", t[m.start():pos]).strip(" :—–-")
        return EpsRecord(doc_id, kind, year, month, vals[ci], vals[pi], ytd, ytd_name, prev_q, label)
    return None


def eps_like_for_like(doc_id: str, doc) -> str | None:
    """One line: the filing's reported period vs the same period a year earlier (diluted EPS)."""
    rec = eps_record(doc_id, doc)
    return rec.line() if rec else None


def eps_records(corpus: Corpus, docs: set[str]) -> list[EpsRecord]:
    """EPS records from an entity's own filings (10-Q/10-K with a parseable table), oldest first."""
    out = []
    for did in sorted(docs, key=lambda d: (corpus.docs[d].doc_date, d) if d in corpus.docs else (None, d)):
        doc = corpus.docs.get(did)
        if doc is None or len(doc.text) < 20000:
            continue
        try:
            rec = eps_record(did, doc)
        except (ValueError, IndexError):
            rec = None
        if rec:
            out.append(rec)
    return out


def like_for_like(corpus: Corpus, docs: set[str]) -> list[str]:
    """Like-for-like lines (diluted EPS, then revenue) for an entity's own filings, oldest first."""
    lines = [r.line() for r in eps_records(corpus, docs)]
    for did in sorted(docs, key=lambda d: (corpus.docs[d].doc_date, d) if d in corpus.docs else (None, d)):
        doc = corpus.docs.get(did)
        if doc is None or len(doc.text) < 20000:
            continue
        try:
            line = statement_line(did, doc)
        except (ValueError, IndexError):
            line = None
        if line:
            lines.append(line)
    return lines


# ---------------------------------------------------------------------------------------------- #
# Like-for-like revenue from the same income statement                                          #
# ---------------------------------------------------------------------------------------------- #
_STATEMENT = re.compile(r"consolidated\s+(?:and\s+combined\s+)?(?:condensed\s+)?statements?\s+of\s+(?:consolidated\s+)?"
                        r"(?:earnings|income|operations)", re.I)
_AMOUNT = r"\(\s*\$?\s*\d{1,3}(?:,\d{3})*(?:\.\d+)?\s*\)|-?\d{1,3}(?:,\d{3})*(?:\.\d+)?"
_AMOUNT_ROW = re.compile(rf"(?:{_SEP}(?:{_AMOUNT})(?![\d%]|\.\d)){{2,8}}")
# Revenue only: a "net income" label also matches discontinued-operations and noncontrolling-interest rows,
# which surveyed wrong on several filings, and EPS already carries the earnings comparison.
_LINES = (
    ("revenue", re.compile(r"total\s+net\s+revenues?|total\s+revenues?(?:,\s+net)?|total\s+net\s+sales|net\s+revenues?|"
                           r"revenues?,\s+net|net\s+sales", re.I)),
)


def _amounts(text: str) -> list[float]:
    return [_num(v) for v in re.findall(_AMOUNT, text)]


def statement_line(doc_id: str, doc) -> str | None:
    """Revenue for the filing's reported period against the same period a year earlier."""
    form = str(doc.meta.get("form_type") or doc.meta.get("title") or "")
    kind = "10-K" if "10-K" in form else "10-Q" if "10-Q" in form else None
    per = _period(doc)
    if kind is None or per is None:
        return None
    year, month = per
    t = doc.text
    heads = [m.start() for m in _STATEMENT.finditer(t)]
    parts = []
    for name, pat in _LINES:
        found = None
        for h in heads:
            for m in pat.finditer(t, h, min(len(t), h + 12000)):
                row = _AMOUNT_ROW.match(t, m.end())
                if not row:
                    continue
                vals = _amounts(row.group(0))
                hdr = _header(t, m.start(), year)
                if hdr is None:
                    continue
                cols, _ = hdr
                if len(vals) > len(cols) or len(vals) < 2:
                    continue
                mapped = _map_columns(cols[:len(vals)], year, month, kind)
                if mapped is None:
                    continue
                found = (vals[mapped[0]], vals[mapped[1]])
                break
            if found:
                break
        if found:
            a, b = found
            parts.append(f"{name} {a:,.0f} vs {b:,.0f} ({_pct(a, b)})")
    if not parts:
        return None
    span = "quarter" if kind == "10-Q" else "fiscal year"
    return (f"LIKE-FOR-LIKE (computed by code from the income statement in {doc_id}, {span} ending {year}-{month:02d} "
            f"vs the same {span} a year earlier, in the statement's own units): " + "; ".join(parts) + ".")
