"""Stress the agent on unit shapes no public unit has (development only; never in the image).

The Final's held-out set "spans many more task families, across all three target types", most with no published
counterpart. This builds variants of the public units on their own frozen corpora -- another target type, other units,
a level instead of a change, label names outside the usual vocabulary, more labels, a missing column -- runs the agent
on them and checks what can be checked without any outcome: the official answer schema, smoke gates, claim rules and
reasons checks, plus a priori sanity bounds on every number (a forecast 100x off its quantity's scale is a unit error
whatever the outcome). Nothing here reads or needs an outcome.

    ../.venv/bin/python tools/stress_units.py --offline            # fallback path only, seconds
    ../.venv/bin/python tools/stress_units.py --parallel 2         # live, OpenRouter's free endpoint, prod clocks
    ../.venv/bin/python tools/stress_units.py --only v-fomc-pp,v-cot-level
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_units as RU  # noqa: E402

UNITS = RU.UNITS


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ------------------------------------------------------------------------------------------------ variants
# Each variant: (source unit, transform(task) -> task, target type, check(entity, row) -> problem or None).
def _ent(task: dict) -> dict[str, dict]:
    return {e["entity_id"]: e for e in task["entities"]}


def fomc_pp(t):
    t["target"] = {"name": "yield_change_pp_intermeeting", "type": "regression"}
    t["prompt"] = re.sub(r"basis points?|bps?\b", "percentage points", t["prompt"], flags=re.I)
    t["prompt"] += " State every change in PERCENTAGE POINTS (e.g. +0.10 for ten basis points), not basis points."
    for e in t["entities"]:
        for k in list(e):
            if k in ("unit", "units"):
                e[k] = "percentage_points"
    return t


def fomc_rank(t):
    t["target"] = {"name": "yield_change_bps_rank", "type": "ranking"}
    t["prompt"] += (" Rank the maturities from the largest to the smallest yield change: put each maturity's predicted "
                    "change in basis points in point_forecast (the ordering is read from it).")
    return t


def cot_level(t):
    t["target"] = {"name": "net_noncommercial_pct_oi_20241126", "type": "regression"}
    t["prompt"] = ("Using ONLY the frozen evidence corpus (CFTC Commitments of Traders history as of the cutoff), predict "
                   "for each futures market its noncommercial NET POSITION AS A PERCENT OF OPEN INTEREST in the "
                   "2024-11-26 report (a level, e.g. -18.2, not a change). Give a point_forecast and a 90% interval, and "
                   "support every claim with citations to the frozen corpus.")
    return t


def cpi_accel(t):
    t["target"] = {"name": "mom_change_vs_prior_month", "type": "classification",
                   "labels": ["accelerates", "decelerates"],
                   "label_assertions": {
                       "accelerates": "the October 2024 seasonally adjusted month-over-month percent change, as first "
                                      "published, is higher than the September 2024 change given in the table",
                       "decelerates": "the October 2024 seasonally adjusted month-over-month percent change, as first "
                                      "published, is equal to or lower than the September 2024 change given in the table"}}
    t["prompt"] += (" Instead of a number, answer for each component whether its October change ACCELERATES (is higher "
                    "than September's) or DECELERATES; still give an interval on the October change in percent.")
    return t


def auction_above(t):
    t["target"] = {"name": "btc_vs_2_5", "type": "classification", "labels": ["strong_demand", "soft_demand"],
                   "label_assertions": {"strong_demand": "the auction's bid-to-cover ratio is above 2.50",
                                        "soft_demand": "the auction's bid-to-cover ratio is 2.50 or below"}}
    t["prompt"] += " Classify each auction as strong_demand (bid-to-cover above 2.50) or soft_demand."
    for e in t["entities"]:
        e["threshold_btc"] = 2.5
    return t


def eps_level(t):
    t["target"] = {"name": "diluted_eps_usd", "type": "regression"}
    t["prompt"] = ("Using ONLY the frozen evidence corpus, predict for each company its GAAP diluted EPS in US dollars for "
                   "the fiscal quarter reported after the cutoff (the same quarter as prior_year_q_eps, one year later). "
                   "Give a point_forecast in dollars per share and a 90% interval, with citations to the corpus.")
    return t


def banks_bins(t):
    t["target"] = {"name": "eps_growth_band", "type": "classification",
                   "labels": ["sharp_decline", "decline", "flat", "growth", "sharp_growth"],
                   "label_assertions": {"sharp_decline": "year-over-year GAAP diluted EPS growth below -20%",
                                        "decline": "growth from -20% to below -5%",
                                        "flat": "growth from -5% to +5%",
                                        "growth": "growth above +5% up to +20%",
                                        "sharp_growth": "growth above +20%"}}
    t["prompt"] += " Answer with the growth BAND label for each bank and an interval on the growth in percent."
    return t


def macrorev_pct(t):
    t["target"] = {"name": "next_estimate_revision_pct", "type": "regression"}
    t["prompt"] = ("Using ONLY the frozen evidence corpus (ALFRED vintage tables and the BLS benchmark announcement), "
                   "predict for each row the PERCENT revision of the agency's next published estimate relative to "
                   "latest_precutoff_estimate ((next - latest) / latest x 100). Give a point_forecast in percent and a 90% "
                   "interval, with citations to the corpus.")
    return t


def postearn_ret(t):
    t["target"] = {"name": "abnormal_return_pct", "type": "regression"}
    t["prompt"] += (" Instead of a label, predict each company's market-adjusted one-day abnormal return in PERCENT "
                    "(point_forecast and a 90% interval).")
    return t


def credit_three(t):
    t["target"] = {"name": "credit_outcome_12m", "type": "classification",
                   "labels": ["bankruptcy", "other_default", "no_event"],
                   "label_assertions": {"bankruptcy": "the issuer files for bankruptcy within twelve months of the cutoff",
                                        "other_default": "a payment default or rating-based default without a bankruptcy "
                                                         "filing within twelve months",
                                        "no_event": "neither within twelve months"}}
    return t


def cpi_nolatest(t):
    for e in t["entities"]:
        e.pop("latest_published_mom_pct", None)
        e.pop("latest_published_ref_month", None)
    t["prompt"] = t["prompt"].replace("The latest published (September) month-over-month change for each component is "
                                      "given. ", "")
    return t


VARIANTS = {
    "v-fomc-pp": ("t4-fomc-curve-20240918", fomc_pp, "regression"),
    "v-fomc-rank": ("t4-fomc-curve-20240918", fomc_rank, "ranking"),
    "v-cot-level": ("t4-cotpos-202411-us10", cot_level, "regression"),
    "v-cpi-accel": ("t4-cpicomp-202410-us11", cpi_accel, "classification"),
    "v-auction-above": ("t4-auction-btc-202411-us7", auction_above, "classification"),
    "v-eps-level": ("t4-eps-yoy-2023Q2-mixed", eps_level, "regression"),
    "v-banks-bins": ("t4-eps-growth-2024Q3-banks", banks_bins, "classification"),
    "v-macrorev-pct": ("t4-macrorev-20240930-us6", macrorev_pct, "regression"),
    "v-postearn-ret": ("t4-postearn-20240201-megacap", postearn_ret, "regression"),
    "v-credit-3": ("t4-credit-event-2023", credit_three, "classification"),
    "v-cpi-nolatest": ("t4-cpicomp-202410-us11", cpi_nolatest, "regression"),
}


NO_NUMERIC_TRUTH = {"v-credit-3"}  # classification variants whose outcome has no numeric quantity (interval [0, 1])


# A priori bounds on each variant's numbers: (low, high) for the point, max interval width -- generous, set from the
# quantity's definition and the pre-cutoff tables, never from an outcome. A breach is a unit or scale error.
def bounds(variant: str, ent: dict) -> tuple[float, float, float]:
    if variant == "v-fomc-pp":
        return -1.5, 1.5, 4.0
    if variant == "v-fomc-rank":
        return -150.0, 150.0, 400.0
    if variant == "v-cot-level":
        x = float(ent["net_pct_oi_20241022"])
        return x - 40.0, x + 40.0, 120.0
    if variant in ("v-cpi-accel", "v-cpi-nolatest"):
        return -12.0, 12.0, 25.0
    if variant == "v-auction-above":
        return 1.5, 4.0, 2.0
    if variant == "v-eps-level":
        x = float(ent["prior_year_q_eps"])
        return x - 3 * max(1.0, abs(x)), x + 3 * max(1.0, abs(x)), 8 * max(1.0, abs(x))
    if variant == "v-banks-bins":
        return -80.0, 150.0, 250.0
    if variant == "v-macrorev-pct":
        return -3.0, 3.0, 8.0
    if variant == "v-postearn-ret":
        return -25.0, 25.0, 60.0
    return -math.inf, math.inf, math.inf


def build(variant: str, root: Path) -> Path:
    src_name, fn, ttype = VARIANTS[variant]
    src, dst = UNITS / src_name, root / variant
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    task = fn(copy.deepcopy(json.loads((src / "task.json").read_text())))
    task["task_id"] = f"{src_name}-{variant}"
    (dst / "task.json").write_text(json.dumps(task, indent=1))
    card = (dst / "card.toml").read_text()
    card = re.sub(r'(target_type\s*=\s*)"[a-z]+"', rf'\1"{ttype}"', card)
    card = re.sub(r'(\nid\s*=\s*)"[^"]+"', rf'\1"{task["task_id"]}"', card)
    if ttype != "classification":
        card = re.sub(r"\ninterval_leg\s*=\s*false", "", card)
    (dst / "card.toml").write_text(card)
    man = json.loads((dst / "manifest.json").read_text())
    man["unit_id"] = task["task_id"]
    for f in man.get("files", []):
        if f.get("path") in ("task.json", "card.toml"):
            f["sha256"], f["bytes"] = _sha(dst / f["path"]), (dst / f["path"]).stat().st_size
    (dst / "manifest.json").write_text(json.dumps(man, indent=1))
    return dst


def run(unit: Path, out: Path, env: dict) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    ans = out / "answer.json"
    if ans.exists():
        ans.unlink()
    t0 = time.time()
    import subprocess

    proc = subprocess.run([str(RU.PY), "-m", "t4agent", "analyze", "--task", str(unit / "task.json"), "--corpus",
                           str(unit / "corpus"), "--out", str(ans)], cwd=RU.AGENT, env=env, capture_output=True,
                          text=True, timeout=2400)
    (out / "agent.log").write_text(proc.stderr)
    m = re.search(r"requests used (\d+), ok (\d+), llm entities (\d+)/(\d+)", proc.stderr)
    errors = [l for l in proc.stderr.splitlines() if " ERROR " in l or "Traceback" in l or "unexpected failure" in l]
    return {"exit": proc.returncode, "sec": round(time.time() - t0, 1), "requests": int(m.group(1)) if m else 0,
            "llm": f"{m.group(3)}/{m.group(4)}" if m else "0/?", "log_errors": errors[:3]}


def sanity(variant: str, unit: Path, ans_path: Path) -> list[str]:
    task = json.loads((unit / "task.json").read_text())
    ents = _ent(task)
    ttype = task["target"]["type"]
    labels = task["target"].get("labels") or []
    probs: list[str] = []
    if not ans_path.exists():
        return ["no answer written"]
    ans = json.loads(ans_path.read_text())
    if ans.get("target_type") != ttype:
        probs.append(f"target_type {ans.get('target_type')} != {ttype}")
    rows = {p["entity_id"]: p for p in ans.get("entity_predictions", [])}
    if set(rows) != set(ents):
        probs.append("roster mismatch")
    points = []
    for eid, p in rows.items():
        iv = p.get("interval") or {}
        lo, hi, pt = iv.get("lo"), iv.get("hi"), p.get("point_forecast")
        if ttype == "classification" and p.get("label") not in labels:
            probs.append(f"{eid}: label {p.get('label')!r} not in {labels}")
        if lo is None or hi is None or not (lo <= hi):
            probs.append(f"{eid}: bad interval {iv}")
            continue
        if pt is not None and not (lo - 1e-9 <= pt <= hi + 1e-9):
            probs.append(f"{eid}: point {pt} outside [{lo}, {hi}]")
        blo, bhi, bw = bounds(variant, ents[eid])
        if ttype == "classification" and (lo, hi) == (0.0, 1.0) and variant not in NO_NUMERIC_TRUTH:
            probs.append(f"{eid}: a probability interval [0, 1] on a numeric quantity")
        elif ttype != "classification" or (lo, hi) != (0.0, 1.0):
            if pt is not None and not (blo <= pt <= bhi):
                probs.append(f"{eid}: point {pt} outside a priori [{blo:g}, {bhi:g}]")
            if hi - lo > bw:
                probs.append(f"{eid}: interval width {hi - lo:.4g} > a priori {bw:g}")
        if pt is not None:
            points.append(pt)
    if ttype == "ranking" and len(set(points)) < max(2, len(points) // 2):
        probs.append(f"ranking points mostly tied: {points}")
    return probs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--parallel", type=int, default=2)
    ap.add_argument("--root", default="", help="where variant units and runs go (default: tools/.runs/stress)")
    args = ap.parse_args()
    root = Path(args.root) if args.root else HERE / ".runs" / "stress"
    names = [v for v in args.only.split(",") if v] or list(VARIANTS)
    units = {v: build(v, root / "units") for v in names}
    if args.offline:
        env = dict(RU.os.environ)
        for k in ("MODEL_ENDPOINT", "MODEL_NAME", "MODEL_TOKEN"):
            env.pop(k, None)
    else:
        env = RU.openrouter_env(23, args.parallel, prod_clocks=True)
    out_root = root / ("offline" if args.offline else "live")

    def one(v):
        r = run(units[v], out_root / v, env)
        ans = out_root / v / "answer.json"
        try:
            c = RU.check_units.__globals__["sys"]  # noqa: F841 - keep run_units' path setup
            sys.path.insert(0, str(HERE))
            import check_answers as C

            chk = C.check(units[v], ans, None) if ans.exists() else {"check": "NO ANSWER"}
        except Exception as exc:  # noqa: BLE001 - report, do not stop the sweep
            chk = {"check": f"ERROR {type(exc).__name__}: {str(exc)[:160]}"}
        r["admissible"], r["factor"] = chk.get("admissible"), chk.get("factor")
        r["schema"] = chk.get("schema_errors") or chk.get("check")
        r["false"] = chk.get("false_claims")
        r["reasons"] = chk.get("reasons")
        r["reason_findings"] = chk.get("reason_findings")
        r["sanity"] = sanity(v, units[v], ans)
        return v, r

    rows = {}
    with ThreadPoolExecutor(max(1, args.parallel)) as pool:
        for v, r in pool.map(one, names):
            rows[v] = r
            ok = r["admissible"] and not r["sanity"] and not r["schema"] and not r["log_errors"]
            print(f"{'OK  ' if ok else 'FAIL'} {v:16s} exit={r['exit']} {r['sec']:6.1f}s req={r['requests']:2d} "
                  f"llm={r['llm']:>5s} admissible={r['admissible']} factor={r['factor']} reasons={r['reasons']} "
                  f"findings={r['reason_findings']}", flush=True)
            for line in (r["schema"] or [])[:3] if isinstance(r["schema"], list) else [r["schema"]] if r["schema"] else []:
                print(f"       schema: {line}")
            for line in r["sanity"][:8]:
                print(f"       sanity: {line}")
            for line in r["log_errors"]:
                print(f"       log: {line[:200]}")
    (out_root / "summary.json").write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
