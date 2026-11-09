"""`analyze --task /input/task.json --corpus /input/corpus/ --out /output/answer.json`

The harness passes `analyze` as the first argument. Whatever happens, this process writes a
schema-valid answer early, improves it atomically, and exits 0 before the unit clock.
"""
from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import threading
import time
from pathlib import Path

from . import __version__, config
from .artifact_policy import ARTIFACTS, ArtifactPolicy
from .answer import build_answer, fit_judge_cap, validate, write_atomic
from .calibrate import _default_half, _quantity_text, finalize, table_halves
from .cite import build_citations
from .corpus import Corpus, load_corpus, map_entities
from .llm import HouseClient
from .predict import Evidence, Forecaster, baseline_value, fallback_forecast, prior_label
from .reasons import build_reasons
from .task import TaskSpec, card_timeout, load_task
from .threshold_consistency import apply_threshold_consistency

log = logging.getLogger("t4agent")

# the paths of the harness's argument line (SUBMISSION_CLI.md)
DEFAULTS = {"task": "/input/task.json", "corpus": "/input/corpus", "out": "/output/answer.json"}


class _Parser(argparse.ArgumentParser):
    def error(self, message: str):  # v21.5: a malformed argument line must not exit 2 before an answer is written
        raise ValueError(message)


def _parse(argv: list[str]) -> argparse.Namespace:
    """v21.5: unknown arguments are ignored and missing paths take the contract's defaults (v21.3 exited 2 with no
    answer on an unexpected extra argument)."""
    p = _Parser(prog="analyze", description="Track 4 evidence-grounded tabular prediction")
    p.add_argument("verb", nargs="?", default="analyze")
    p.add_argument("--task", default=DEFAULTS["task"])
    p.add_argument("--corpus", default=DEFAULTS["corpus"])
    p.add_argument("--out", default=DEFAULTS["out"])
    try:
        args, extra = p.parse_known_args(argv)
    except ValueError as exc:
        log.warning("argument line not understood (%s): using the contract's paths", exc)
        return argparse.Namespace(verb="analyze", **DEFAULTS)
    ignored = ([args.verb] if args.verb != "analyze" else []) + extra
    if ignored:
        log.warning("ignoring unexpected arguments: %s", " ".join(ignored))
    return args


def _table_halves(spec: TaskSpec, corpus: Corpus, ev: Evidence) -> dict[str, float | None]:
    return table_halves(spec, corpus, ev)  # v11: shared with the label probabilities (calibrate.table_halves)


def emergency_row(spec: TaskSpec, eid: str) -> dict:
    """v21.5: a row from the task alone, for an entity whose code path failed: the status-quo label with the row's
    quantity as finalize sizes it from the task (pure code, no corpus: an announcement return centred on 0, a revised
    level around its latest estimate; [0, 1] only when nothing numeric is known), or the naive value (the latest value,
    or no change) with the default half-width for the quantity's name. Its claims are left empty, so build_answer
    quotes the entity's own task-table row."""
    if spec.target_type == "classification":
        try:
            ent = spec.entity(eid)
            row = finalize(spec, fallback_forecast(spec, ent, None), None, ent=ent)
            nums = [row.get(k) for k in ("point_forecast", "lo", "hi") if row.get(k) is not None]
            if "lo" in row and "hi" in row and all(isinstance(x, (int, float)) and math.isfinite(x) for x in nums) \
                    and row["lo"] <= row["hi"]:
                return {**row, "claims": []}
        except Exception:  # noqa: BLE001 - the last line of defence must not raise: the cruder row below
            pass
        return {"label": prior_label(spec.labels) or "unknown", "lo": 0.0, "hi": 1.0, "claims": []}
    try:
        point = baseline_value(spec, spec.entity(eid), None)
    except Exception:  # noqa: BLE001 - the last line of defence must not raise
        point = None
    point = point if point is not None and math.isfinite(point) else 0.0
    half = _default_half(_quantity_text(spec, None), point)
    return {"point_forecast": point, "lo": point - half, "hi": point + half, "claims": []}


def _rows(spec: TaskSpec, corpus: Corpus, ev: Evidence, forecasts: dict, plan: dict | None,
          terms_for=lambda label: [], fb_rows: dict | None = None) -> dict:
    """Each entity's submitted numbers and claims. v21.5: an entity whose finalize or citations raise keeps its
    fallback row (`fb_rows`), else an emergency row; the others are unaffected (v21.3 lost the whole answer)."""
    rows = {}
    try:
        halves = _table_halves(spec, corpus, ev)
    except Exception as exc:  # noqa: BLE001 - a scale hint must never cost the answer
        log.warning("table scales unavailable: %r", exc)
        halves = {}
    for eid in spec.entity_ids:
        fc = forecasts.get(eid)
        try:
            nums = finalize(spec, fc, plan, ent=spec.entity(eid), table_half=halves.get(eid))
            try:
                nums, _codes = apply_threshold_consistency(spec, spec.entity(eid), plan, nums)
            except Exception as exc:  # noqa: BLE001 - a gate failure must not replace the forecast
                log.warning("threshold consistency left %s unchanged: %r", eid, exc)
            nums["claims"] = build_citations(spec, corpus, ev, fc, terms_for(nums.get("label")))
        except Exception as exc:  # noqa: BLE001
            kept = fb_rows is not None and eid in fb_rows
            log.warning("row %s failed (%r): %s row kept", eid, exc, "fallback" if kept else "emergency")
            rows[eid] = dict(fb_rows[eid]) if kept else emergency_row(spec, eid)
            continue
        rows[eid] = nums
        if fc.source == "llm":
            probs = {k: round(v, 2) for k, v in (fc.probs or {}).items()}
            log.info("  %s: %s point=%s [%s, %s] samples=%d probs=%s supports=%d cites=%d", eid,
                     nums.get("label", ""), nums.get("point_forecast"), nums["lo"], nums["hi"], fc.n_samples,
                     probs or "", len(fc.support), len(nums["claims"]))
    return rows


def _answer(spec: TaskSpec, corpus: Corpus, ev: Evidence, fc, forecasts: dict, rows: dict, trace: str,
            client: HouseClient | None, plan: dict | None = None) -> dict:
    """The answer with its reasons: written once the numbers are final, so the implications agree with them."""
    rows = fit_judge_cap(spec, rows)  # v13: within the reasoning grader's 3,000-byte answer cap
    repaired = {}
    for eid, row in rows.items():
        try:
            repaired[eid], _codes = apply_threshold_consistency(spec, spec.entity(eid), plan, row)
        except Exception as exc:  # noqa: BLE001 - the cap result stays the v21.13 row if the gate cannot place it
            log.warning("post-cap consistency left %s unchanged: %r", eid, exc)
            repaired[eid] = row
    rows = repaired
    ans = build_answer(spec, rows, trace, None, corpus)
    try:
        reasons = build_reasons(spec, corpus, ev, fc, forecasts, ans["entity_predictions"], client)
    except Exception as exc:  # noqa: BLE001 - reasons are optional; leaving them out costs nothing
        log.warning("reasons unavailable: %r", exc)
        reasons = []
    ans = build_answer(spec, rows, trace, reasons, corpus)
    if reasons and validate(spec, corpus, ans):
        ans = build_answer(spec, rows, trace, None, corpus)  # never let reasons spoil an admissible answer
    return ans


def _trace(spec: TaskSpec, corpus: Corpus, forecasts: dict, client: HouseClient | None, t0: float) -> str:
    n_llm = sum(1 for f in forecasts.values() if f.source == "llm")
    parts = [
        f"t4agent {__version__}.",
        "Eligible historical capabilities: "
        + (", ".join(k for k in ARTIFACTS if ArtifactPolicy(spec.cutoff).permits(k)) or "none; corpus-only fallback")
        + ".",
        f"Corpus: {len(corpus.docs)} documents on/before the {spec.cutoff_str} cutoff were indexed"
        + (f"; {len(corpus.excluded)} excluded ({', '.join(sorted(set(corpus.excluded.values())))})" if corpus.excluded else "")
        + ".",
        f"Forecasts: {n_llm}/{len(forecasts)} entities from model analysis (self-consistency), the rest from a "
        "rule-based fallback.",
        "Claims: verbatim quotes of the passages the analysts relied on, from documents the manifest assigns to the "
        "entity or marks shared; every cited document is dated on or before the cutoff. Embargo: PASS.",
    ]
    if client is not None:
        parts.append(f"House requests used: {client.used}.")
    return " ".join(parts)


def emergency_answer(spec: TaskSpec, t0: float) -> dict:
    """v21.5: a schema-valid answer from the task alone (emergency rows, task-row claims), for a fallback stage that
    failed before its answer was written."""
    rows = {eid: emergency_row(spec, eid) for eid in spec.entity_ids}
    trace = (f"t4agent {__version__}. Emergency answer after {time.time() - t0:.0f} s: the code path failed, so every "
             "row is the naive value or the status-quo label; claims quote the entity's own row of the task table.")
    return build_answer(spec, rows, trace, None, None)


def run(argv: list[str]) -> int:
    t0 = time.time()
    args = _parse(argv)
    # v21.5: the clocks follow the unit's own time limit (card [agent].timeout_sec; 600 s gives v21.3's 470 / 400 s)
    try:
        hard, llm_deadline, scale = config.derive_clock(card_timeout(Path(args.task).parent))
    except Exception as exc:  # noqa: BLE001 - runs before any answer is written: a card must never cost it
        log.warning("unit time limit unreadable (%r): using the default clock", exc)
        hard, llm_deadline, scale = config.derive_clock(None)
    # Hard stop: never let the process outlive its budget (the fallback answer is already on disk). v21.5: at the
    # hard stop itself (v21.3: 30 s after it)
    watchdog = threading.Timer(hard, lambda: os._exit(0))
    watchdog.daemon = True
    watchdog.start()
    log.info("clock: hard stop %.0f s, LLM deadline %.0f s, margins x%.2f", hard, llm_deadline, scale)

    spec = load_task(args.task)
    client = HouseClient(deadline=t0 + llm_deadline, clock_scale=scale)
    corpus = ev = fc = None
    rows = None
    # 1. fallback answer on disk within seconds: the code-computed anchors, priors and statistical baselines. v21.5:
    # whatever fails here, an answer is written (v21.3 could end with none: whole-unit zero)
    try:
        corpus = load_corpus(args.corpus, spec)
        specific, shared = map_entities(spec, corpus)
        ev = Evidence(spec, corpus, specific, shared)
        log.info("task %s: %d entities, %s, %d docs (%d excluded), small=%s", spec.task_id, len(spec.entity_ids),
                 spec.target_type, len(corpus.docs), len(corpus.excluded), ev.small)
        ev.build_packs([])
        try:
            fc = Forecaster(spec, corpus, ev, client, t0)
            fc.build_anchors()
            fallback = fc.aggregate({})
        except Exception as exc:  # noqa: BLE001 - the rule-based fallback is the last line of defence
            log.warning("statistical fallback unavailable: %r", exc)
            fallback = {e["entity_id"]: fallback_forecast(spec, e, None) for e in spec.entities}
        rows = _rows(spec, corpus, ev, fallback, None)
        ans = _answer(spec, corpus, ev, fc, fallback, rows, _trace(spec, corpus, fallback, None, t0), None)
        errs = validate(spec, corpus, ans)
        if errs:
            log.error("fallback answer has problems: %s", errs[:5])
    except Exception as exc:  # noqa: BLE001
        log.exception("fallback answer failed (%r): writing an emergency answer", exc)
        ans = emergency_answer(spec, t0)
    write_atomic(args.out, ans)
    log.info("fallback written at %.1fs", time.time() - t0)

    # 2. model-based forecasts. v21.13: the model stage runs in a worker thread and the main thread waits for it only
    # until the LLM deadline plus a short grace, then keeps the last answer written: R1 (2026-10-21) lost a whole
    # fomc unit to a stage that never returned (no log between 220 s and the 360 s hard stop, fallback only).
    if not client.enabled:
        log.warning("House endpoint not configured; keeping the fallback answer")
        return 0
    if corpus is None or ev is None:
        log.warning("no corpus or evidence to forecast from; keeping the emergency answer")
        return 0
    state = {"rc": 0, "stage": "start", "written": "fallback"}

    def stage(name: str) -> None:
        state["stage"] = name
        log.info("stage %s: start at %.1fs", name, time.time() - t0)

    def done(name: str) -> None:
        log.info("stage %s: end at %.1fs", name, time.time() - t0)

    def model_stage() -> None:
        nonlocal fc
        try:
            if fc is None:
                fc = Forecaster(spec, corpus, ev, client, t0)
            stage("plan"); plan = fc.run_plan(); terms = fc.plan_terms(); done("plan")
            if terms:
                ev.build_packs(terms)
            stage("analysts"); samples = fc.run_analysts(); done("analysts")
            stage("extra-votes"); samples = fc.extra_votes(samples); done("extra-votes")
            stage("aggregate"); forecasts = fc.aggregate(samples)
            rows2 = _rows(spec, corpus, ev, forecasts, plan, lambda label: fc.plan_terms(label), fb_rows=rows)
            done("aggregate")
            trace = _trace(spec, corpus, forecasts, client, t0)
            # the model forecasts reach the disk before the reasons call: a reasons call that outlives the deadline
            # can no longer cost them (code reasons stand in)
            stage("forecast-answer")
            pre = _answer(spec, corpus, ev, fc, forecasts, rows2, trace, None, plan)
            if not validate(spec, corpus, pre):
                write_atomic(args.out, pre)
                state["written"] = "forecasts"
                log.info("forecast answer written at %.1fs (code reasons)", time.time() - t0)
            done("forecast-answer")
            stage("reasons")
            ans = _answer(spec, corpus, ev, fc, forecasts, rows2, _trace(spec, corpus, forecasts, client, t0), client, plan)
            done("reasons")
            errs = validate(spec, corpus, ans)
            if errs:
                log.error("final answer failed validation, keeping the %s answer: %s", state["written"], errs[:5])
                return
            write_atomic(args.out, ans)
            state["written"] = "final"
            log.info("final answer written at %.1fs (requests used %d, ok %d, llm entities %d/%d)",
                     time.time() - t0, client.used, client.ok, sum(f.source == "llm" for f in forecasts.values()),
                     len(forecasts))
        except Exception:  # noqa: BLE001 - the answer written so far stands
            log.exception("model stage failed at stage %s; the %s answer stands", state["stage"], state["written"])

    worker = threading.Thread(target=model_stage, name="model-stage", daemon=True)
    worker.start()
    guard = max(1.0, min(llm_deadline + config.STAGE_GRACE_SEC, hard - 20.0))
    worker.join(max(0.0, t0 + guard - time.time()))
    if worker.is_alive():
        client.close()
        log.warning("model stage %s still running at %.1fs (guard %.0f s): the %s answer stands", state["stage"],
                    time.time() - t0, guard, state["written"])
        logging.shutdown()
        os._exit(0)  # in-flight request threads (and pool workers joined at exit) must not hold the process
    return state["rc"]

def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=os.environ.get("T4_LOG", "INFO"), stream=sys.stderr,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    argv = sys.argv[1:] if argv is None else argv
    try:
        return run(argv)
    except SystemExit as exc:  # argparse --help or usage error
        return int(exc.code or 0)
    except Exception:  # noqa: BLE001 - the answer written so far stands; never exit non-zero
        log.exception("unexpected failure; the last answer written stands")
        return 0


if __name__ == "__main__":
    sys.exit(main())
