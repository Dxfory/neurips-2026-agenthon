"""Run the agent on public units and check every answer the way scorer 5.2.2 would, minus outcomes.

    python tools/run_units.py --parallel 2 --prod-clocks   # all units, live model on OpenRouter's free endpoint
    python tools/run_units.py --offline                    # no model: exercises the fallback path only
    python tools/run_units.py --units t4-fomc-curve-20240918 --no-check

Per unit: exit code, wall time, House requests used, then (tools/check_answers.py, the organizers' own
code) the answer schema, the smoke gates, the deterministic claim rules with the faithfulness factor they
imply, and the submitted_reasons checks. The developer key is read from ../.openrouter.env (--backend nvidia:
../.nvidia.env) at run time and passed only to the child process environment; it is never printed or written.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
AGENT = HERE.parent
ROOT = AGENT.parent
TRACK = ROOT / "track4-analysis-public"
UNITS = TRACK / "units"
PY = ROOT / ".venv" / "bin" / "python"
SMOKE = ROOT / ".venv" / "bin" / "qfbench2-smoke"


def live_env(budget: int, parallel: int = 1, prod_clocks: bool = False) -> dict:
    """NVIDIA's hosted API (integrate.api.nvidia.com): the same NIM serving stack as the House route, so the
    top-level reasoning_budget and chat_template_kwargs behave as in production. The developer's account allows
    38 requests per minute in total, shared by the units run in parallel. The public endpoint is shared and
    slower than the House route, so local clocks are longer unless --prod-clocks is given. The key is read from
    ../.nvidia.env at run time and passed only to the child process."""
    key = re.search(r"nvapi-[A-Za-z0-9_-]+", (ROOT / ".nvidia.env").read_text()).group(0)
    env = dict(os.environ)
    env.pop("T4_EXTRA_BODY", None)
    env.update({
        "MODEL_ENDPOINT": "https://integrate.api.nvidia.com",
        "MODEL_NAME": "nvidia/nemotron-3-super-120b-a12b",
        "MODEL_TOKEN": key,
        "T4_MAX_RPM": str(max(1, 38 // max(1, parallel))),
        "T4_CONCURRENCY": "3",  # as in production; the cool-down absorbs 503 bursts
        "T4_REQUEST_BUDGET": str(budget),
    })
    if not prod_clocks:
        env.update({"T4_REQUEST_TIMEOUT": "300", "T4_LLM_DEADLINE": "900", "T4_TIME_BUDGET": "1000"})
    return env


def openrouter_env(budget: int, parallel: int = 1, prod_clocks: bool = False, free: bool = True, rpm: int = 18) -> dict:
    """OpenRouter. Default: the user's choice since 2026-10-03, the free endpoint nvidia/nemotron-3-super-120b-a12b:free
    (served by Nvidia; it caps thinking with reasoning.max_tokens, so the agent's per-call budgets are translated by
    T4_REASONING_FIELD; about 20 requests a minute and 1,000 a day on this account, shared by the units run in
    parallel). free=False: the paid DeepInfra bf16 route (AGENTS.md), which honours neither budget, on local clocks.
    The NVIDIA hosted API answers HTTP 410 for this model since 2026-10-03 evening."""
    keyfile = ROOT / os.environ.get("T4_OPENROUTER_KEYFILE", ".openrouter.env")  # another account's key file
    key = re.search(r"sk-or-v1-[A-Za-z0-9]+", keyfile.read_text()).group(0)
    env = dict(os.environ)
    env.pop("T4_EXTRA_BODY", None)
    env.pop("T4_OPENROUTER_KEYFILE", None)
    env.update({"MODEL_ENDPOINT": "https://openrouter.ai/api", "MODEL_TOKEN": key, "T4_REQUEST_BUDGET": str(budget)})
    if free:
        env.update({"MODEL_NAME": "nvidia/nemotron-3-super-120b-a12b:free", "T4_REASONING_FIELD": "openrouter",
                    "T4_MAX_RPM": str(max(1, rpm // max(1, parallel))), "T4_CONCURRENCY": "3"})
        if not prod_clocks:
            env.update({"T4_REQUEST_TIMEOUT": "300", "T4_LLM_DEADLINE": "900", "T4_TIME_BUDGET": "1000"})
    else:
        env.update({"MODEL_NAME": "nvidia/nemotron-3-super-120b-a12b", "T4_CONCURRENCY": "2",
                    "T4_REQUEST_TIMEOUT": "600", "T4_LLM_DEADLINE": "1800", "T4_TIME_BUDGET": "2000",
                    "T4_EXTRA_BODY": json.dumps({"provider": {"order": ["deepinfra/bf16"], "allow_fallbacks": False},
                                                 "reasoning": {"effort": "low"}})})
    return env


def run_agent(unit: str, out_dir: Path, env: dict) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    ans = out_dir / "answer.json"
    if ans.exists():
        ans.unlink()
    t0 = time.time()
    proc = subprocess.run(
        [str(PY), "-m", "t4agent", "analyze", "--task", str(UNITS / unit / "task.json"),
         "--corpus", str(UNITS / unit / "corpus"), "--out", str(ans)],
        cwd=AGENT, env=env, capture_output=True, text=True, timeout=2400)
    log = proc.stderr
    (out_dir / "agent.log").write_text(log)
    m = re.search(r"requests used (\d+), ok (\d+), llm entities (\d+)/(\d+)", log)
    return {"exit": proc.returncode, "sec": round(time.time() - t0, 1),
            "requests": int(m.group(1)) if m else 0, "ok_calls": int(m.group(2)) if m else 0,
            "llm_entities": f"{m.group(3)}/{m.group(4)}" if m else "0/?"}


def check_units(units: list[str], out_root: Path) -> dict:
    sys.path.insert(0, str(HERE))
    import check_answers as C

    try:
        counter = C.R.judge_token_counter()
    except Exception:  # noqa: BLE001 - the token cap is then reported as unchecked
        counter = None
    res = {}
    for unit in units:
        ans = out_root / unit / "answer.json"
        if not ans.exists():
            res[unit] = {"check": "NO ANSWER"}
            continue
        try:
            res[unit] = C.check(UNITS / unit, ans, counter)
        except Exception as exc:  # noqa: BLE001 - report, do not stop the sweep
            res[unit] = {"check": f"ERROR {type(exc).__name__}: {str(exc)[:120]}"}
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", default="", help="comma list; default all public units")
    ap.add_argument("--unit-root", default="", help="directory holding the units (default: the public units); "
                    "e.g. ../units_pre2022/units")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--no-check", action="store_true")
    ap.add_argument("--check-only", action="store_true", help="check existing answers")
    ap.add_argument("--budget", type=int, default=23)
    ap.add_argument("--parallel", type=int, default=1, help="units run concurrently")
    ap.add_argument("--prod-clocks", action="store_true", help="production LLM deadline and time budget")
    ap.add_argument("--rpm", type=int, default=18, help="free endpoint: requests a minute for this run, split across "
                    "--parallel (lower it when several runs share one key)")
    ap.add_argument("--backend", default="openrouter", choices=["openrouter", "openrouter-deepinfra", "nvidia"],
                    help="openrouter = the free Nemotron endpoint (default); nvidia answers HTTP 410 since 2026-10-03")
    ap.add_argument("--out", default=str(HERE / ".runs" / "latest"))
    args = ap.parse_args()
    global UNITS
    if args.unit_root:
        UNITS = Path(args.unit_root).resolve()
    units = [u for u in args.units.split(",") if u] or sorted(p.name for p in UNITS.iterdir() if p.is_dir())
    out_root = Path(args.out)
    if args.offline:
        env = dict(os.environ)
    elif args.backend.startswith("openrouter"):
        env = openrouter_env(args.budget, args.parallel, args.prod_clocks, free=args.backend == "openrouter", rpm=args.rpm)
    else:
        env = live_env(args.budget, args.parallel, args.prod_clocks)
    if args.offline:
        for k in ("MODEL_ENDPOINT", "MODEL_NAME", "MODEL_TOKEN"):
            env.pop(k, None)
    rows = {}
    if not args.check_only:
        from concurrent.futures import ThreadPoolExecutor

        def one(unit):
            r = run_agent(unit, out_root / unit, env)
            print(f"{unit:32s} exit={r['exit']} {r['sec']:6.1f}s req={r['requests']:2d} ok={r['ok_calls']:2d} "
                  f"llm={r['llm_entities']:>6s}", flush=True)
            return unit, r

        with ThreadPoolExecutor(max(1, args.parallel)) as pool:
            for unit, r in pool.map(one, units):
                rows[unit] = r
    if not args.no_check:
        cr = check_units(units, out_root)
        for unit in units:
            c = cr.get(unit, {})
            rows.setdefault(unit, {}).update(c)
            print(f"{unit:32s} admissible={c.get('admissible')} factor={c.get('factor')} false={c.get('false_claims')} "
                  f"schema={len(c.get('schema_errors') or [])} reasons={c.get('reasons')} "
                  f"reason_findings={c.get('reason_findings')} {c.get('check', '')}", flush=True)
    (out_root / "summary.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
