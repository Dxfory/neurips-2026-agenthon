"""Pipe-table parsing and compact per-column statistics (strong priors and interval scales)."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

from .corpus import Corpus

_NUM = re.compile(r"^[+\-]?\$?\(?[+\-]?\d[\d,]*(?:\.\d+)?\)?%?$")


def to_number(cell: str) -> float | None:
    s = cell.strip().replace("−", "-")
    if not s or s in ("--", "-", "—", "n/a", "NA", "N/A"):
        return None
    if not _NUM.match(s):
        return None
    neg = s.startswith("(") and s.endswith(")")
    s = s.strip("()").replace("$", "").replace(",", "").replace("%", "").replace("+", "")
    try:
        v = float(s)
    except ValueError:
        return None
    return -v if neg else v


@dataclass
class Table:
    doc_id: str
    header: list[str]
    rows: list[list[str]]


def parse_tables(doc_id: str, text: str) -> list[Table]:
    tables: list[Table] = []
    block: list[list[str]] = []
    for line in text.split("\n") + [""]:
        if line.count("|") >= 1:  # a two-column table ("date | value") has a single separator per line
            block.append([c.strip() for c in line.split("|")])
            continue
        if len(block) >= 4:
            width = max(set(len(r) for r in block), key=[len(r) for r in block].count)
            rows = [r for r in block if len(r) == width]
            if len(rows) >= 4:
                tables.append(Table(doc_id, rows[0], rows[1:]))
        block = []
    return tables


def _stats(vals: list[float]) -> dict[str, float]:
    n = len(vals)
    mean = sum(vals) / n
    sd = math.sqrt(sum((v - mean) ** 2 for v in vals) / (n - 1)) if n > 1 else 0.0
    diffs = [b - a for a, b in zip(vals, vals[1:])]
    sdd = math.sqrt(sum(d * d for d in diffs) / len(diffs)) if diffs else 0.0
    return {
        "n": n,
        "last": vals[-1],
        "prev": vals[-2] if n > 1 else vals[-1],
        "mean3": sum(vals[-3:]) / min(3, n),
        "mean6": sum(vals[-6:]) / min(6, n),
        "mean": mean,
        "sd": sd,
        "rms_change": sdd,
        "min": min(vals),
        "max": max(vals),
    }


def _g(x: float) -> str:
    if x == 0:
        return "0"
    mag = abs(x)
    if mag >= 1000:
        return f"{x:,.0f}"
    if mag >= 100:
        return f"{x:.1f}"
    if mag >= 1:
        return f"{x:.3g}" if mag < 10 else f"{x:.2f}".rstrip("0").rstrip(".")
    return f"{x:.3f}".rstrip("0").rstrip(".")


_DATE_CELL = re.compile(r"^(\d{4})-(\d{2})(?:-(\d{2}))?$")
_CHANGE_TARGET = re.compile(r"change|growth|delta|shift", re.I)


def _period_days(cells: list[str]) -> float | None:
    """Median spacing, in days, of a date-like first column (YYYY-MM-DD or YYYY-MM)."""
    import datetime as dt

    days = []
    for c in cells:
        m = _DATE_CELL.match(c.strip())
        if not m:
            return None
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3) or 15)
        try:
            days.append(dt.date(y, mo, d).toordinal())
        except ValueError:
            return None
    gaps = sorted(b - a for a, b in zip(days, days[1:]) if b > a)
    return float(gaps[len(gaps) // 2]) if gaps else None


def _tokens(s: str) -> set[str]:
    return {t for t in re.split(r"[^a-z0-9]+", s.lower()) if t}


def entity_half_width(corpus: Corpus, doc_ids: list[str], ent: dict, target_name: str,
                      horizon_days: int | None, entity_docs: set[str]) -> float | None:
    """90% half-width implied by the entity's own series in a corpus table, or None.

    A column is the entity's series when its header equals the entity's series code / name, or
    (inside the entity's own document) when every header token appears in the target name. The
    scale is the RMS of per-row changes: one step for a next-value target, random-walk scaled by
    sqrt(periods in the horizon) for a change-over-horizon target. Levels quoted in percent are
    converted when the target is in basis points.
    """
    ids = {str(ent.get(k)).strip().lower() for k in ("series_fred", "series_id", "name", "series_name", "entity_id")
           if isinstance(ent.get(k), str) and ent.get(k)}
    tgt = _tokens(target_name)
    change_type = bool(_CHANGE_TARGET.search(target_name))
    bps = "bps" in target_name.lower() or str(ent.get("unit", "")).lower().startswith("bps")
    for did in doc_ids:
        doc = corpus.docs.get(did)
        if doc is None:
            continue
        for t in parse_tables(did, doc.text):
            for j, name in enumerate(t.header):
                if j == 0:
                    continue
                h = name.strip().lower()
                ht = _tokens(h)
                exact = h in ids or any(h.startswith(i + " ") or h.startswith(i + " (") for i in ids if len(i) >= 4)
                own = did in entity_docs and ht and ht <= tgt | {"pct"} and len(ht & tgt) >= 1
                if not (exact or own):
                    continue
                vals = [to_number(r[j]) for r in t.rows if j < len(r)]
                if len(vals) < 6 or any(v is None for v in vals):
                    continue
                diffs = [b - a for a, b in zip(vals, vals[1:])]
                rms = math.sqrt(sum(d * d for d in diffs) / len(diffs))
                if rms <= 0:
                    continue
                sd = rms
                if change_type and horizon_days:
                    period = _period_days([r[0] for r in t.rows]) or 30.0
                    periods = horizon_days * (5 / 7) if period <= 1.5 else horizon_days / period
                    sd = rms * math.sqrt(max(1.0, periods))
                if bps and max(abs(v) for v in vals) < 30:
                    sd *= 100.0
                return 1.645 * sd
    return None


def summarize(corpus: Corpus, doc_ids: list[str], max_chars: int = 7000) -> str:
    """Compact column statistics for every numeric column of every pipe table in `doc_ids`."""
    out: list[str] = []
    used = 0
    for did in doc_ids:
        doc = corpus.docs.get(did)
        if doc is None:
            continue
        for t in parse_tables(did, doc.text):
            first = t.header[0] if t.header else ""
            span = f"{t.rows[0][0]}..{t.rows[-1][0]}" if t.rows and t.rows[0] else ""
            lines = [f"[{did}] table rows by '{first}' ({span}, {len(t.rows)} rows):"]
            for j, name in enumerate(t.header):
                if j == 0:
                    continue
                vals = [to_number(r[j]) for r in t.rows if j < len(r)]
                vals = [v for v in vals if v is not None and math.isfinite(v)]
                if len(vals) < 4:
                    continue
                s = _stats(vals)
                lines.append(
                    f"  {name[:48]}: last={_g(s['last'])} prev={_g(s['prev'])} mean3={_g(s['mean3'])} "
                    f"mean6={_g(s['mean6'])} mean={_g(s['mean'])} sd={_g(s['sd'])} "
                    f"rms_change={_g(s['rms_change'])} min={_g(s['min'])} max={_g(s['max'])}"
                )
            if len(lines) > 1:
                block = "\n".join(lines)
                if used + len(block) > max_chars:
                    return "\n".join(out)
                out.append(block)
                used += len(block)
    return "\n".join(out)
