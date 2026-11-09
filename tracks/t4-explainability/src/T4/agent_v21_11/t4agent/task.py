"""Task loading: the trusted roster, the target spec, and the scorer's hypothesis template."""
from __future__ import annotations

import datetime as dt
import json
import math
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

TARGET_TYPES = ("classification", "regression", "ranking")
ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def parse_date(value: Any) -> dt.date | None:
    if isinstance(value, str) and ISO_DATE.match(value):
        try:
            return dt.date.fromisoformat(value)
        except ValueError:
            return None
    return None


@dataclass
class TaskSpec:
    raw: dict
    task_id: str
    target_type: str  # always one of TARGET_TYPES (best guess when the task is malformed)
    target_type_trusted: bool  # emit target_type in the answer only when this is True
    target_name: str
    labels: list[str]
    prompt: str
    cutoff: dt.date | None
    cutoff_str: str
    resolution_str: str
    interval_level: float
    entities: list[dict]
    entity_ids: list[str]
    unit_dir: Path | None = None
    card_params: dict = field(default_factory=dict)
    canaries: tuple[str, ...] = ()  # v21.11: the unit's canary strings (card [contamination]); never written out

    @property
    def horizon_days(self) -> int | None:
        r = parse_date(self.resolution_str)
        if r and self.cutoff:
            return (r - self.cutoff).days
        return None

    def entity(self, eid: str) -> dict:
        for e in self.entities:
            if e.get("entity_id") == eid:
                return e
        return {"entity_id": eid}

    def display_name(self, eid: str) -> str:
        """Subject string exactly as the scorer's hypothesis builds it."""
        name = self.entity(eid).get("name")
        if isinstance(name, str) and name:
            return f"{name} ({eid})"
        return eid


def _read_card(unit_dir: Path | None) -> dict:
    if unit_dir is None:
        return {}
    card = unit_dir / "card.toml"
    try:
        data = tomllib.loads(card.read_text(encoding="utf-8"))
        return ((data.get("scoring") or {}).get("params")) or {}
    except (OSError, tomllib.TOMLDecodeError, UnicodeDecodeError):
        return {}


_CANARY_MIN = 8  # a canary is a long token (a GUID); short card values are not treated as one


def card_canaries(unit_dir: Path | None) -> tuple[str, ...]:
    """v21.11: the string values of the card's [contamination] table (the unit's canary GUID). README: in the Final, a
    canary string anywhere in the output is scored as contamination. Empty when there is no card or no such table."""
    if unit_dir is None:
        return ()
    try:
        data = tomllib.loads((unit_dir / "card.toml").read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError, UnicodeDecodeError):
        return ()
    table = data.get("contamination") if isinstance(data.get("contamination"), dict) else {}
    vals = {v.strip() for v in table.values() if isinstance(v, str) and len(v.strip()) >= _CANARY_MIN}
    return tuple(sorted(vals))


def card_timeout(unit_dir: Path | None) -> float | None:
    """v21.5: the unit's time limit in seconds, the card's [agent].timeout_sec (the ceiling the platform runs the unit
    under), or None when there is no card or no usable value. Nothing else of the card is read here."""
    if unit_dir is None:
        return None
    try:
        data = tomllib.loads((unit_dir / "card.toml").read_text(encoding="utf-8"))
        v = (data.get("agent") or {}).get("timeout_sec")
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return None
        v = float(v)  # an integer too large for a float raises OverflowError here, inside the guard
    except (OSError, tomllib.TOMLDecodeError, UnicodeDecodeError, AttributeError, OverflowError):
        return None
    return v if math.isfinite(v) and v > 0 else None


def load_task(task_path: str | Path) -> TaskSpec:
    task_path = Path(task_path)
    raw = json.loads(task_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("task.json is not an object")
    unit_dir = task_path.parent
    card = _read_card(unit_dir)

    target = raw.get("target") if isinstance(raw.get("target"), dict) else {}
    ttype = target.get("type")
    card_type = card.get("target_type")
    trusted = ttype in TARGET_TYPES and (card_type is None or card_type == ttype)
    if ttype not in TARGET_TYPES:
        ttype = card_type if card_type in TARGET_TYPES else ("classification" if target.get("labels") else "regression")
    labels = [x for x in (target.get("labels") or []) if isinstance(x, str) and x]
    tname = target.get("name") if isinstance(target.get("name"), str) and target.get("name") else "outcome"

    level = None
    for src in (card.get("interval_level"), raw.get("interval_level")):
        try:
            v = float(src)
            if 0.0 < v < 1.0 and math.isfinite(v):
                level = v
                break
        except (TypeError, ValueError):
            continue
    if level is None:
        level = 0.90

    entities = [e for e in (raw.get("entities") or []) if isinstance(e, dict) and isinstance(e.get("entity_id"), str)]
    seen: set[str] = set()
    ids: list[str] = []
    for e in entities:
        if e["entity_id"] not in seen:
            seen.add(e["entity_id"])
            ids.append(e["entity_id"])
    cutoff_str = raw.get("cutoff_date") if isinstance(raw.get("cutoff_date"), str) else ""
    return TaskSpec(
        raw=raw,
        task_id=str(raw.get("task_id", "")),
        target_type=ttype,
        target_type_trusted=trusted,
        target_name=tname,
        labels=labels,
        prompt=str(raw.get("prompt", "")),
        cutoff=parse_date(cutoff_str),
        cutoff_str=cutoff_str,
        resolution_str=str(raw.get("resolution_date") or ""),
        interval_level=level,
        entities=entities,
        entity_ids=ids,
        unit_dir=unit_dir,
        card_params=card,
        canaries=card_canaries(unit_dir),
    )


def fmt_number(value: float) -> str:
    """The scorer's `_fmt`: the exact rendering the judge sees."""
    if value == int(value) and abs(value) < 1e15:
        return str(int(value))
    return f"{value:.6g}"


def canonical_hypothesis(spec: TaskSpec, eid: str, *, label: str = "", point: float = 0.0,
                         rank: int | None = None, lo: float = 0.0, hi: float = 0.0) -> str:
    """Replica of qfbench2_track_analysis.hypothesis.canonical_hypothesis (for logging/diagnostics)."""
    subject = spec.display_name(eid)
    tname = spec.target_name.replace("_", " ")
    level_pct = fmt_number(spec.interval_level * 100.0)
    tail = f" The {level_pct}% prediction interval for the {tname} of {subject} is {fmt_number(lo)} to {fmt_number(hi)}."
    if spec.target_type == "classification":
        head = f"The {tname} of {subject} is {label}."
    elif spec.target_type == "regression":
        head = f"The {tname} of {subject} is {fmt_number(point)}."
    else:
        pos = f"ranked {rank}" if rank is not None else "ranked"
        head = f"By {tname}, {subject} is {pos} among the entities in this task, with a score of {fmt_number(point)}."
    return head + tail
