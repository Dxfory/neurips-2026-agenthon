"""Apply only the threshold gate to v21.13 offline answers.

Reads answer JSON and public task cards. Does not score, and does not open truth,
outcome, or score files. Does not call a model endpoint.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
AGENT = HERE.parent
T4 = AGENT.parent
sys.path.insert(0, str(AGENT))

from t4agent.task import load_task  # noqa: E402
from t4agent.threshold_consistency import apply_threshold_consistency, build_card, label_of  # noqa: E402

A1 = Path("/workspace/t4a1/T4/sub/verification-b/offline-A1")
UNITS = Path("/workspace/t4c/T4/track4-analysis-public/units")


def _preds(path: Path) -> dict[str, dict]:
    ans = json.loads(path.read_text())
    out = {}
    for row in ans.get("entity_predictions") or []:
        eid = row.get("entity_id")
        if not eid:
            continue
        interval = row.get("interval") or {}
        out[eid] = {
            "label": row.get("label"),
            "point_forecast": row.get("point_forecast"),
            "lo": interval.get("lo"),
            "hi": interval.get("hi"),
        }
    return out


def main() -> int:
    if not UNITS.is_dir():
        print("public units missing")
        return 1
    summary = {"rows": 0, "changed": 0, "failures": [], "units": {}}
    for unit in sorted(p for p in UNITS.iterdir() if p.is_dir()):
        spec = load_task(unit / "task.json")
        old = _preds(A1 / f"{spec.task_id}.json")
        unit_changes = []
        failures = 0
        for eid in spec.entity_ids:
            row = dict(old.get(eid) or {})
            summary["rows"] += 1
            try:
                new, codes = apply_threshold_consistency(spec, spec.entity(eid), None, row)
            except Exception as exc:  # noqa: BLE001 - record the gate failure, do not repair further
                failures += 1
                summary["failures"].append({"unit": spec.task_id, "entity": eid, "error": type(exc).__name__})
                continue
            card = build_card(spec, spec.entity(eid), None)
            point, label = new.get("point_forecast"), new.get("label")
            problems = []
            if point is not None and new.get("lo") is not None and new.get("hi") is not None:
                if not (float(new["lo"]) <= float(point) <= float(new["hi"])):
                    problems.append("interval misses point")
            if card is not None and label is not None and point is not None:
                got = label_of(point, card)
                if got != label:
                    problems.append(f"label_of={got}")
            if problems:
                failures += 1
                summary["failures"].append({
                    "unit": spec.task_id, "entity": eid, "problems": problems, "row": new,
                })
            changed = []
            if row.get("label") != new.get("label"):
                changed.append("label")
            if row.get("point_forecast") != new.get("point_forecast"):
                changed.append("point")
            if row.get("lo") != new.get("lo") or row.get("hi") != new.get("hi"):
                changed.append("interval")
            if changed:
                summary["changed"] += 1
                unit_changes.append({
                    "entity": eid,
                    "fields": changed,
                    "codes": codes,
                    "before": row,
                    "after": {
                        "label": new.get("label"),
                        "point_forecast": new.get("point_forecast"),
                        "lo": new.get("lo"),
                        "hi": new.get("hi"),
                    },
                })
        summary["units"][spec.task_id] = {
            "n": len(spec.entity_ids),
            "changed": len(unit_changes),
            "failures": failures,
            "changes": unit_changes,
        }
    out = Path("/tmp/hybrid-gate.json")
    out.write_text(json.dumps(summary, indent=1))
    print(
        f"rows={summary['rows']} changed={summary['changed']} "
        f"failures={len(summary['failures'])} out={out}"
    )
    for name, unit in summary["units"].items():
        if not unit["changed"] and not unit["failures"]:
            continue
        print(f"{name}: changed={unit['changed']} failures={unit['failures']}")
        for change in unit["changes"]:
            before, after = change["before"], change["after"]
            print(
                f"  {change['entity']} {change['fields']} {change['codes']} "
                f"point {before.get('point_forecast')} -> {after.get('point_forecast')} "
                f"label {before.get('label')} -> {after.get('label')} "
                f"interval [{before.get('lo')},{before.get('hi')}] -> [{after.get('lo')},{after.get('hi')}]"
            )
    return 1 if summary["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
