"""Score saved answers on the pre-2022 stress units (tools/build_pre2022_units.py) with scorer 5.2.2's own code.

For each <run>/<unit>/answer.json of a unit in ../units_pre2022/units: the answer is aligned and scored by the official
`_composite` against the unit's pre-2022 outcome and declared naive answer (copied with the unit into a temporary
directory, so the agent never reads them), times the deterministic claim factor (`citation_rail.claim_penalty_preview`,
the scorer's factor whenever no claim is contradicted). A run that fails alignment scores the worst case. Development
only: these outcomes may be used to compare versions because they resolved before 2022 (TRAINING-POLICY); the House
model has probably seen them, so the numbers measure robustness and relative change, not honest skill.

    ../.venv/bin/python tools/score_pre2022.py <run> [<run> ...]       # per-unit table and means, runs side by side
"""
from __future__ import annotations

import json
import shutil
import statistics as stx
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRACK = ROOT / "track4-analysis-public"
NEW = ROOT / "units_pre2022"
sys.path.insert(0, str(TRACK))
sys.path.insert(0, str(TRACK / "baselines" / "guardrails_example"))

import citation_rail as R  # noqa: E402
from qfbench2_track_analysis import scoring as S  # noqa: E402
from qfbench2_track_analysis.alignment import align_predictions  # noqa: E402


def score(unit_id: str, ans_path: Path, counter=None) -> dict:
    unit, ref = NEW / "units" / unit_id, NEW / "reference" / unit_id
    if not ans_path.exists():
        return {"error": "no answer"}
    ans = json.loads(ans_path.read_text())
    with tempfile.TemporaryDirectory(prefix="t4s-") as tmp:
        t = Path(tmp) / unit_id
        shutil.copytree(unit, t)
        shutil.copytree(ref, t / "reference")
        ctx: dict = {"unit_dir": str(t)}
        S.hydrate(ctx)
        params = ctx["_params"]
        try:
            aligned = align_predictions(ans, ctx["_roster"], target_type=params.target_type,
                                        interval_level=params.interval_level)
        except Exception as exc:  # noqa: BLE001 - a participant failure scores the worst case
            return {"error": f"alignment: {type(exc).__name__}: {str(exc)[:120]}", "analysis": params.worst_case}
        parts = S._composite(aligned, ctx["realized"], params, ctx["_roster"], naive_aligned=ctx["_naive_aligned"])
    pen = R.claim_penalty_preview(ans, unit, token_counter=counter)
    factor = pen.get("factor")
    factor = 0.0 if factor is None else float(factor)
    return {"pq": parts["predictive_quality"], "iq": parts.get("interval_quality"),
            "coverage": parts.get("interval_coverage"), "composite": parts["composite"], "factor": factor,
            "analysis": factor * parts["composite"]}


def main(argv: list[str]) -> int:
    runs = [Path(a) for a in argv]
    units = sorted(p.name for p in (NEW / "units").iterdir() if p.is_dir())
    try:
        counter = R.judge_token_counter()
    except Exception:  # noqa: BLE001 - the 400-token cap is then not checked
        counter = None
    table = {r: {u: score(u, r / u / "answer.json", counter) for u in units} for r in runs}
    head = f"{'unit':20s}" + "".join(f" | {r.parent.name + '/' + r.name:>26s}" for r in runs)
    print(head + "\n" + " " * 20 + "".join(f" | {'PQ    IQ    fac   anal':>26s}" for _ in runs))
    for u in units:
        cells = []
        for r in runs:
            s = table[r][u]
            if "error" in s:
                cells.append(f" | {s['error'][:26]:>26s}")
            else:
                iq = "  -  " if s["iq"] is None else f"{s['iq']:.3f}"
                cells.append(f" | {s['pq']:.3f} {iq} {s['factor']:.2f} {s['analysis']:.3f}")
        print(f"{u:20s}" + "".join(cells))
    means = []
    for r in runs:
        vals = [table[r][u].get("analysis", 0.0) for u in units]
        means.append(stx.mean(vals))
    print(f"{'MEAN analysis':20s}" + "".join(f" | {m:26.4f}" for m in means))
    if len(runs) >= 2:
        a, b = runs[0], runs[-1]
        d = [table[b][u].get("analysis", 0.0) - table[a][u].get("analysis", 0.0) for u in units]
        print(f"{b.name} - {a.name}: {stx.mean(d):+.4f}, better in {sum(x > 1e-9 for x in d)}/{len(d)}, "
              f"worse in {sum(x < -1e-9 for x in d)}/{len(d)}")
    out = runs[0] / "pre2022_scores.json" if runs else None
    if out:
        out.write_text(json.dumps({str(r): table[r] for r in runs}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
