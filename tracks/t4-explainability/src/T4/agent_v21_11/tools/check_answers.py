"""Check saved answers against the published Track 4 rules (scorer 5.2.2) with the organizers' own code.

For each <run>/<unit>/answer.json: the toolkit's answer schema, the smoke gates g0-g3, the deterministic claim
rules (with the judge's tokenizer when it is cached) and the faithfulness factor they imply, and the
`submitted_reasons` checks with the grader's per-reason plan.

    ../.venv/bin/python tools/check_answers.py tools/.runs/<run> [...]
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRACK = ROOT / "track4-analysis-public"
UNITS = TRACK / "units"
sys.path.insert(0, str(TRACK))
sys.path.insert(0, str(TRACK / "baselines" / "guardrails_example"))

import citation_rail as R  # noqa: E402
import jsonschema  # noqa: E402
import qfbench2_common  # noqa: E402
from qfbench2_common.smoke import run_smoke  # noqa: E402
from qfbench2_track_analysis.scoring import build_smoke_verifier  # noqa: E402

SCHEMA = json.loads((Path(qfbench2_common.__file__).parent / "schemas" / "analysis.schema.json").read_text())


def check(unit_dir: Path, answer_path: Path, counter) -> dict:
    ans = json.loads(answer_path.read_text(encoding="utf-8"))
    task = json.loads((unit_dir / "task.json").read_text(encoding="utf-8"))
    corpus = R.load_corpus(unit_dir / "corpus")
    rec: dict = {"unit": unit_dir.name}
    rec["schema_errors"] = [e.message[:120] for e in jsonschema.Draft202012Validator(SCHEMA).iter_errors(ans)][:3]
    rails = R.check_answer(ans, corpus, task["cutoff_date"], task=task)
    rec["rail_findings"] = dict(Counter(f.code for f in rails))
    claims = R.check_claim_rules(ans, unit_dir, token_counter=counter)
    rec["false_claims"] = dict(Counter(f.code for f in claims if f.claim_index >= 0))
    rec["unit_findings"] = sorted({f.code for f in claims if f.claim_index < 0})
    pen = R.claim_penalty_preview(ans, unit_dir, token_counter=counter)
    rec["claims"], rec["factor"] = pen.get("claims"), pen.get("factor")
    rec["reason_findings"] = dict(Counter(f.code for f in R.check_submitted_reasons(ans, corpus, task["cutoff_date"])))
    plan = R.reasons_judged(ans, corpus, task["cutoff_date"])
    rec["reasons"] = f"{sum(1 for p in plan if p['judged'])}/{len(ans.get('submitted_reasons') or [])} judged"
    with tempfile.TemporaryDirectory(prefix="t4c-") as tmp:
        inp, out = Path(tmp, "input"), Path(tmp, "output")
        shutil.copytree(unit_dir, inp)
        out.mkdir()
        shutil.copy(answer_path, out / "answer.json")
        verdict = run_smoke(inp, out, build_smoke_verifier)
    rec["admissible"] = bool(verdict.admissible)
    return rec


def main(argv: list[str]) -> int:
    try:
        counter = R.judge_token_counter()
    except Exception as exc:  # noqa: BLE001 - the token cap is then reported as unchecked
        print(f"(judge tokenizer unavailable: {exc!r}; the 400-token cap is not checked)")
        counter = None
    bad = 0
    for run in argv:
        for ans in sorted(Path(run).glob("*/answer.json")):
            unit = UNITS / ans.parent.name
            if not unit.is_dir():
                continue
            rec = check(unit, ans, counter)
            ok = (rec["admissible"] and not rec["schema_errors"] and not rec["false_claims"]
                  and not rec["reason_findings"] and rec["factor"] == 1.0)
            bad += not ok
            print(("OK  " if ok else "BAD ") + json.dumps(rec, ensure_ascii=False))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
