"""Proxy grader for `submitted_reasons` (development only; never in the image).

The Final's reasoning grader compares each unit's reasons with hidden target reasons and grades coverage,
evidence grounding, inferential link and answer consistency (plus a contradiction flag), with LLM judge votes.
The target reasons are not published, so this proxy uses the drivers the task statement names (the list the
agent plans against); a statement that names none gets three or four target reasons written once by the judge
from the statement alone, cached and reused for every version compared. No outcome is involved anywhere.

    ../.venv/bin/python tools/judge_reasons.py --label v9 <run>/<unit>/answer.json ... --label v10 <...>

v21: --judge-model grades with another, stronger model (any OpenRouter model id, e.g. anthropic/claude-sonnet-5.5): it
also writes the target reasons of EVERY unit once from the task statement alone (cached per judge model, shared by all
versions compared), casts 5 votes per unit, counts the contradiction flag and reports each component with its spread
across votes. Nemotron judging Nemotron's reasons against the driver list the reasons are planned on was lenient and
circular. The official grader's target reasons and its way of combining the components are not published.

Backend: OpenRouter's free Nemotron endpoint (the developer's key from ../.openrouter.env, read at run time and
never printed; paced under its 20 requests a minute), --backend openrouter-deepinfra (the paid bf16 route, which
graded every proxy number up to v12), or --backend nvidia (../.nvidia.env). Compare versions on one backend only.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
AGENT = HERE.parent
ROOT = AGENT.parent
UNITS = ROOT / "track4-analysis-public" / "units"
sys.path.insert(0, str(AGENT))

from t4agent.corpus import load_corpus  # noqa: E402
from t4agent.reasons import task_drivers  # noqa: E402
from t4agent.task import load_task  # noqa: E402

CACHE = HERE / ".runs" / "judge_targets.json"
JUDGE_MODEL: str | None = None  # v21: set by --judge-model
PROVIDER: str | None = None  # v22: --provider pins one OpenRouter provider (e.g. deepinfra/bf16), no fallbacks
MAX_TOKENS = 6000
WHY = False  # v21: --why asks the judge for a one-line reason per score (diagnosis only; not for comparing versions)
DUMP: dict = {}  # v21: --dump keeps every vote's parsed reply with the targets and the reasons it judged
MODEL = "nvidia/nemotron-3-super-120b-a12b"


def backend(name: str, judge_model: str | None = None) -> tuple[str, dict, dict]:
    if judge_model and judge_model.startswith("deepseek-"):  # v24: DeepSeek's own API (the user's key), e.g. deepseek-flash
        global JUDGE_MODEL
        JUDGE_MODEL = judge_model
        key = re.search(r"sk-[A-Za-z0-9]+", (ROOT / ".deepseek.env").read_text()).group(0)
        return "https://api.deepseek.com/chat/completions", {"Authorization": f"Bearer {key}"}, {"model": judge_model}
    if judge_model:  # any OpenRouter model; JSON is read from the reply (not every free model takes response_format)
        JUDGE_MODEL = judge_model
        key = re.search(r"sk-or-v1-[A-Za-z0-9]+", (ROOT / ".openrouter.env").read_text()).group(0)
        extra = {"model": judge_model}
        if PROVIDER:
            extra["provider"] = {"order": [PROVIDER], "allow_fallbacks": False}
        return ("https://openrouter.ai/api/v1/chat/completions", {"Authorization": f"Bearer {key}"}, extra)
    if name == "nvidia":
        key = re.search(r"nvapi-[A-Za-z0-9_-]+", (ROOT / ".nvidia.env").read_text()).group(0)
        return "https://integrate.api.nvidia.com/v1/chat/completions", {"Authorization": f"Bearer {key}"}, {}
    key = re.search(r"sk-or-v1-[A-Za-z0-9]+", (ROOT / ".openrouter.env").read_text()).group(0)
    if name == "openrouter-deepinfra":
        return ("https://openrouter.ai/api/v1/chat/completions", {"Authorization": f"Bearer {key}"},
                {"provider": {"order": ["deepinfra/bf16"], "allow_fallbacks": False}, "reasoning": {"effort": "low"}})
    return ("https://openrouter.ai/api/v1/chat/completions", {"Authorization": f"Bearer {key}"},
            {"model": MODEL + ":free", "reasoning": {"max_tokens": 3000}})


_PACE = threading.Lock()
_last = [0.0]


def _pace(extra: dict) -> None:
    """The free endpoint allows about 20 requests a minute: start one at most every 3.5 s."""
    if not str(extra.get("model", "")).endswith(":free"):
        return
    with _PACE:
        wait = _last[0] + 3.5 - time.time()
        if wait > 0:
            time.sleep(wait)
        _last[0] = time.time()


def chat(be, messages: list[dict], temperature: float = 0.3) -> str:
    url, headers, extra = be
    body = {"model": MODEL, "messages": messages, "temperature": temperature, "max_tokens": MAX_TOKENS, **extra}
    if JUDGE_MODEL:  # another judge model: no Nemotron reasoning fields
        body.pop("reasoning", None)
    free = str(body.get("model", "")).endswith(":free")
    for attempt in range(15 if free else 6):
        try:
            _pace(extra)
            r = requests.post(url, headers=headers, json=body, timeout=300)
            if r.status_code == 200:
                try:  # v24: a 200 can carry an upstream error instead of choices (Nemotron Ultra, overloaded)
                    text = r.json()["choices"][0]["message"].get("content") or ""
                except (KeyError, IndexError, TypeError, ValueError):
                    text = ""
                if text.strip() or not free:
                    return text
            # v21: a free endpoint throttles bursts (429 upstream); v24: overloaded providers (503) for longer, up to five
            # minutes a try
            time.sleep(min(300, 10 * 2 ** attempt) if free else 5 * (attempt + 1))
        except requests.RequestException:
            time.sleep(min(300, 10 * 2 ** attempt) if free else 5 * (attempt + 1))
    return ""


def extract(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def targets(be, unit: str, spec) -> list[str]:
    judge_model = JUDGE_MODEL
    cache_file = CACHE if not judge_model else HERE / ".runs" / f"judge_targets_{re.sub(r'[^A-Za-z0-9]+', '_', judge_model)}.json"
    drivers = task_drivers(spec.prompt)
    if drivers and not judge_model:
        return drivers
    cache = json.loads(cache_file.read_text()) if cache_file.exists() else {}
    if unit in cache:
        return cache[unit]
    msg = [{"role": "user", "content": (
        "You write the answer key for a forecasting exercise. From the task statement below ALONE (do not use any "
        "knowledge of what happened after its cutoff), list the 3 or 4 reasons an expert analyst would give for "
        "the forecasts: each is a driver to check in pre-cutoff documents, in under 15 words (3 to 5 when the task is "
        "rich enough).\n\nTASK:\n"
        f"{spec.prompt}\n\nReturn JSON: {{\"targets\": [\"...\"]}}")}]
    out: list[str] = []
    for _ in range(3):  # an empty or unparseable reply is retried, never cached
        reply = chat(be, msg, temperature=0.0)
        obj = extract(reply) or {}
        items = obj.get("targets") if isinstance(obj.get("targets"), list) else []
        out = [t if isinstance(t, str) else " ".join(str(v) for v in t.values() if isinstance(v, str))
               for t in items if isinstance(t, (str, dict))]
        out = [t.strip() for t in out if t and t.strip()][:5]
        if not out:  # a bulleted list instead of JSON
            out = [re.sub(r"^\s*(?:[-*\u2022]|\d+[.)])\s*", "", ln).strip() for ln in (reply or "").splitlines()
                   if re.match(r"^\s*(?:[-*\u2022]|\d+[.)])\s+\S", ln)][:4]
        if out:
            break
    if not out:
        return []
    cache[unit] = out
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(cache, indent=1))
    return out


def judge_prompt(spec, corpus, answer: dict, tgts: list[str]) -> list[dict]:
    fields = ("label", "interval") if spec.target_type == "classification" else ("point_forecast", "interval")
    rows = [{"entity_id": p["entity_id"], **{k: p[k] for k in fields if k in p}} for p in answer["entity_predictions"]]
    blocks = []
    for i, r in enumerate(answer.get("submitted_reasons") or [], 1):
        cited = []
        for c in r.get("citations") or []:
            doc = corpus.docs.get(c["doc_id"])
            if doc is not None:
                cited.append(doc.text[c["span_start"]:c["span_end"]][:8000])
        blocks.append(f"r{i}:\n  premise: {r['premise']}\n  mechanism: {r['mechanism']}\n"
                      f"  answer_implication: {r['answer_implication']}\n  cited text:\n    "
                      + "\n    ---\n    ".join(cited))
    names = "\n".join(f"{e['entity_id']}: {e.get('name') or e.get('series_name') or ''}" for e in spec.entities)
    user = "\n".join([
        "You grade the reasons a forecaster submitted, the way a review panel would. You see the task, the "
        "forecaster's answers, its reasons with the corpus text they cite, and the panel's TARGET REASONS (the "
        "drivers a good explanation must cover). Judge only what is written; do not use knowledge of outcomes.",
        "", "TASK:", spec.prompt, "", "ENTITIES:", names, "", "SUBMITTED ANSWERS:", json.dumps(rows), "",
        "TARGET REASONS:", "\n".join(f"T{i}: {t}" for i, t in enumerate(tgts, 1)), "",
        "SUBMITTED REASONS:", "\n\n".join(blocks) or "(none)", "",
        "For each target reason: which submitted reason covers it (or null) and how well, 0 to 1 (1 = the reason "
        "explicitly addresses this driver with evidence and draws its consequence for the answers).",
        "For each submitted reason, 0 to 1: evidence_grounding (the premise and mechanism are supported by the "
        "cited text), inferential_link (the mechanism validly connects the evidence to the answer), "
        "answer_consistency (the implication agrees with the submitted answers), and contradiction (true if it "
        "contradicts the cited text, another reason or the answers).",
        'Return JSON only: {"targets": [{"target": "T1", "covered_by": "r1", "coverage": 0.8}], '
        '"reasons": [{"reason": "r1", "evidence_grounding": 0.9, "inferential_link": 0.7, '
        '"answer_consistency": 1.0, "contradiction": false}]}',
    ] + ([
        "Add to every target and every reason a field \"why\": one sentence (at most 30 words) saying what earned or cost "
        "the score -- what is missing, unsupported or not connected, concretely.",
    ] if WHY else []))
    return [{"role": "user", "content": user}]


def grade(be, path: Path, votes: int) -> dict | None:
    unit = path.parent.name
    spec = load_task(UNITS / unit / "task.json")
    corpus = load_corpus(UNITS / unit / "corpus", spec)
    answer = json.loads(path.read_text())
    tgts = targets(be, unit, spec)
    if not tgts:
        return None
    msgs = judge_prompt(spec, corpus, answer, tgts)
    workers = 1 if str(be[2].get("model", "")).endswith(":free") else votes  # v21: one vote at a time on a free endpoint
    with ThreadPoolExecutor(max_workers=workers) as pool:
        objs = [extract(t) for t in pool.map(lambda _: chat(be, msgs), range(votes))]
    objs = [o for o in objs if isinstance(o, dict)]
    if not objs:
        return None
    if True:  # v21: kept for --dump (cheap)
        DUMP.setdefault(str(path), {"unit": unit, "targets": tgts,
                                    "reasons": [{k: r.get(k) for k in ("premise", "mechanism", "answer_implication")}
                                                for r in answer.get("submitted_reasons") or []], "votes": []})["votes"] += objs

    def f(x) -> float:
        try:
            return max(0.0, min(1.0, float(x)))
        except (TypeError, ValueError):
            return 0.0

    cov, ground, link, cons, contra, per_vote = [], [], [], [], [], []
    for o in objs:
        # v21: one score per target the panel gave (T1..Tn), the best if a vote lists a target twice; a vote that listed
        # extra or repeated targets had pushed coverage above 1 (FOMC 2024: 1.05)
        best: dict[str, float] = {}
        for x in o.get("targets") or []:
            if isinstance(x, dict):
                tid = str(x.get("target") or "").strip().upper()
                if tid in {f"T{i}" for i in range(1, len(tgts) + 1)}:
                    best[tid] = max(best.get(tid, 0.0), f(x.get("coverage")) if x.get("covered_by") else 0.0)
        cov.append(sum(best.values()) / len(tgts))
        rs = [x for x in o.get("reasons") or [] if isinstance(x, dict)]
        if rs:
            ground.append(statistics.mean(f(x.get("evidence_grounding")) for x in rs))
            link.append(statistics.mean(f(x.get("inferential_link")) for x in rs))
            cons.append(statistics.mean(f(x.get("answer_consistency")) for x in rs))
            contra.append(statistics.mean(1.0 if x.get("contradiction") is True else 0.0 for x in rs))
            # v21: one vote's proxy; a reason flagged contradictory earns nothing in it
            ok = [x for x in rs if x.get("contradiction") is not True]
            quality = (statistics.mean(statistics.mean([f(x.get("evidence_grounding")), f(x.get("inferential_link")),
                                                        f(x.get("answer_consistency"))]) for x in ok) * len(ok) / len(rs)
                       if ok else 0.0)
            per_vote.append(cov[-1] * quality)
    m = lambda xs: statistics.mean(xs) if xs else 0.0  # noqa: E731
    out = {"unit": unit, "targets": len(tgts), "votes": len(objs), "coverage": m(cov), "grounding": m(ground),
           "link": m(link), "consistency": m(cons), "contradiction": m(contra)}
    out["proxy"] = out["coverage"] * statistics.mean([out["grounding"], out["link"], out["consistency"]])
    if per_vote:  # v21: with the contradiction flag counted, and its spread across the votes
        out["proxy_c"] = statistics.mean(per_vote)
        out["vote_sd"] = statistics.pstdev(per_vote) if len(per_vote) > 1 else 0.0
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="openrouter", choices=["openrouter", "openrouter-deepinfra", "nvidia"])
    ap.add_argument("--votes", type=int, default=5)  # v21: the official grader's 5 votes per cell (v10-v20: 3)
    ap.add_argument("--judge-model", default=None, help="another OpenRouter model as the judge (v21)")
    ap.add_argument("--why", action="store_true", help="ask for a one-line reason per score (diagnosis only)")
    ap.add_argument("--provider", default=None, help="pin one OpenRouter provider, e.g. deepinfra/bf16 (v22)")
    ap.add_argument("--max-tokens", type=int, default=6000, help="output cap per judge call, thinking included (v22)")
    ap.add_argument("--resume", action="store_true", help="skip units already in this run's progress file (v24)")
    ap.add_argument("--dump", default=None, help="write every vote's parsed reply here (JSON)")
    ap.add_argument("--unit-root", default="", help="directory holding the units (default: the public units), e.g. "
                                                     "../units_pre2022/units (v21.7)")
    args, tokens = ap.parse_known_args(argv)  # the rest, in order: --label NAME answer.json ... (repeatable)
    global WHY, PROVIDER, MAX_TOKENS, UNITS
    WHY, PROVIDER, MAX_TOKENS = args.why, args.provider, args.max_tokens
    if args.unit_root:
        UNITS = Path(args.unit_root).resolve()
    be = backend(args.backend, args.judge_model)
    groups: dict[str, list[Path]] = {}
    label = "default"
    i = 0
    while i < len(tokens):
        if tokens[i] == "--label":
            label = tokens[i + 1]
            i += 2
            continue
        groups.setdefault(label, []).append(Path(tokens[i]))
        i += 1
    results: dict[str, dict[str, dict]] = {}
    jobs = [(lab, p) for lab, ps in groups.items() for p in ps]
    # v24: every graded unit is appended to a progress file as it comes, and --resume skips what it holds: a free
    # judge's provider may stall for hours, and a stopped run had kept nothing
    tag0 = re.sub(r"[^A-Za-z0-9]+", "_", args.judge_model) + "_" if args.judge_model else ""
    progress = HERE / ".runs" / f"progress_{tag0}{'_'.join(groups)}.jsonl"
    if args.resume and progress.exists():
        for line in progress.read_text().splitlines():
            try:
                row = json.loads(line)
                results.setdefault(row["label"], {})[row["unit"]] = row["result"]
            except (ValueError, KeyError):
                continue
        jobs = [(lab, p) for lab, p in jobs if p.parent.name not in results.get(lab, {})]
        print(f"resuming: {sum(len(r) for r in results.values())} graded, {len(jobs)} to go", flush=True)
    for unit in sorted({p.parent.name for _, p in jobs}):  # target reasons first, one unit at a time (one cache file)
        targets(be, unit, load_task(UNITS / unit / "task.json"))
    with ThreadPoolExecutor(max_workers=1 if str(be[2].get("model", "")).endswith(":free") else 4) as pool:
        for (lab, p), res in zip(jobs, pool.map(lambda j: grade(be, j[1], args.votes), jobs)):
            if res:
                results.setdefault(lab, {})[res["unit"]] = res
                with progress.open("a") as fh:
                    fh.write(json.dumps({"label": lab, "unit": res["unit"], "result": res}) + "\n")
    units = sorted({u for r in results.values() for u in r})
    keys = ("coverage", "grounding", "link", "consistency", "contradiction", "proxy")
    if any("proxy_c" in x for r in results.values() for x in r.values()):
        keys += ("proxy_c", "vote_sd")
    print(f"{'unit':32s} " + " ".join(f"{lab:>34s}" for lab in results))
    print(f"{'':32s} " + " ".join(" ".join(f"{k[:5]:>5s}" for k in keys) for _ in results))
    for u in units:
        print(f"{u:32s} " + " ".join(" ".join(f"{results[lab].get(u, {}).get(k, float('nan')):5.2f}" for k in keys)
                                     for lab in results))
    for lab, r in results.items():
        print(f"MEAN {lab:27s} " + " ".join(f"{k}={statistics.mean(x.get(k, float('nan')) for x in r.values()):.3f}" for k in keys))
    tag = re.sub(r"[^A-Za-z0-9]+", "_", args.judge_model) + "_" if args.judge_model else ""
    out = HERE / ".runs" / f"judge_{tag}{'_'.join(results)}.json"
    out.write_text(json.dumps(results, indent=1))
    if args.dump:
        Path(args.dump).write_text(json.dumps(DUMP, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
