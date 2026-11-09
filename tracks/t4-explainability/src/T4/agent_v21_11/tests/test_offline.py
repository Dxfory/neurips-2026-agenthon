"""Offline (no model) end-to-end tests against the OFFICIAL alignment and embargo code.

Run from agent_v1/:  ../.venv/bin/python -m pytest -q tests
Needs the dev venv (qfbench2-common) and the track repo next to this folder.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

AGENT = Path(__file__).resolve().parents[1]
ROOT = AGENT.parent
TRACK = ROOT / "track4-analysis-public"
UNITS = TRACK / "units"
sys.path.insert(0, str(TRACK))

from qfbench2_track_analysis.alignment import align_predictions  # noqa: E402
from qfbench2_track_analysis.scoring import hydrate  # noqa: E402

ALL_UNITS = sorted(p.name for p in UNITS.iterdir() if p.is_dir())


def run_agent(unit_dir: Path, out: Path, extra_env: dict | None = None) -> subprocess.CompletedProcess:
    # v21.5: a developer's T4_* settings (local clocks, budgets) never leak into these runs; tests pass their own
    env = {k: v for k, v in os.environ.items() if not k.startswith(("MODEL_", "T4_"))}
    env.update(extra_env or {})
    return subprocess.run(
        [sys.executable, "-m", "t4agent", "analyze", "--task", str(unit_dir / "task.json"),
         "--corpus", str(unit_dir / "corpus"), "--out", str(out)],
        cwd=AGENT, env=env, capture_output=True, text=True, timeout=300)


def official_check(unit_dir: Path, answer: dict) -> None:
    """Raise if the official gates g1-g3 (minus the NLI judge) would refuse the answer."""
    import jsonschema
    from qfbench2_common.taskcard import schema_path

    schema = json.loads(Path(schema_path("analysis.schema.json")).read_text())
    jsonschema.Draft202012Validator(schema).validate(answer)
    ctx = {"unit_dir": unit_dir}
    hydrate(ctx)
    assert answer["task_id"] == ctx["_task"]["task_id"]
    p = ctx["_params"]
    aligned = align_predictions(answer, ctx["_roster"], target_type=p.target_type, interval_level=p.interval_level)
    report = ctx["_corpus"].embargo_report(aligned.all_citations(), ctx["_cutoff"])
    assert report.clean, report


@pytest.mark.parametrize("unit", ALL_UNITS)
def test_public_unit_offline(unit, tmp_path):
    out = tmp_path / "answer.json"
    proc = run_agent(UNITS / unit, out)
    assert proc.returncode == 0, proc.stderr[-2000:]
    answer = json.loads(out.read_text())
    official_check(UNITS / unit, answer)
    for row in answer["entity_predictions"]:
        assert 1 <= len(row["claims"]) <= 16
    # v21.5 (B9d): the code-built reasons name no internal document id and carry no backtest bookkeeping
    ids = [p.stem for p in (UNITS / unit / "corpus").glob("*.json") if p.name != "manifest.json"]
    for r in answer.get("submitted_reasons") or []:
        text = r["mechanism"] + " " + r["answer_implication"]
        assert not [d for d in ids if d in text], text
        assert not re.search(r"backtest|out-of-sample|origins|computed by code", text, re.I), text


# ------------------------------------------------------------------------------------ variants
def _clone(unit: str, dst: Path) -> Path:
    shutil.copytree(UNITS / unit, dst)
    return dst


def _rehash(unit_dir: Path) -> None:
    """Declare new corpus files and rewrite digests after a test edit (paths are unit-relative).

    The corpus manifest is rewritten before the root one, which may hash it.
    """
    for man_path in (unit_dir / "corpus" / "manifest.json", unit_dir / "manifest.json"):
        if not man_path.exists():
            continue
        man = json.loads(man_path.read_text())
        known = {e["path"] for e in man["files"]}
        template = next((e for e in man["files"] if e.get("role") == "corpus"), None)
        for f in sorted((unit_dir / "corpus").glob("*.json")):
            rel = f"corpus/{f.name}"
            if f.name != "manifest.json" and rel not in known and template is not None:
                man["files"].append(dict(template, path=rel))
        for e in man["files"]:
            fp = unit_dir / e["path"]
            if fp.exists():
                data = fp.read_bytes()
                e["sha256"] = hashlib.sha256(data).hexdigest()
                if "bytes" in e:
                    e["bytes"] = len(data)
        man_path.write_text(json.dumps(man, indent=1))


def test_post_cutoff_document_is_never_cited(tmp_path):
    unit = _clone("t4-EXAMPLE-eps-beat", tmp_path / "u")
    leak = {"doc_id": "EDGAR_0000320193_8K_20990101", "doc_date": "2099-01-01", "ticker": "AAPL",
            "title": "Apple results (post-cutoff)", "text": "Apple reported diluted EPS of $9.99, a massive beat."}
    (unit / "corpus" / "EDGAR_0000320193_8K_20990101.json").write_text(json.dumps(leak))
    _rehash(unit)
    out = tmp_path / "answer.json"
    assert run_agent(unit, out).returncode == 0
    answer = json.loads(out.read_text())
    cited = {c["doc_id"] for r in answer["entity_predictions"] for c in r["claims"]}
    assert "EDGAR_0000320193_8K_20990101" not in cited
    official_check(unit, answer)


def test_spans_format_documents(tmp_path):
    unit = _clone("t4-fomc-curve-20240918", tmp_path / "u")
    for f in (unit / "corpus").glob("*.json"):
        if f.name == "manifest.json":
            continue
        d = json.loads(f.read_text())
        words = d.pop("text").split(" ")
        d["spans"] = [{"text": " ".join(words[i:i + 40])} for i in range(0, len(words), 40)]
        f.write_text(json.dumps(d))
    _rehash(unit)
    out = tmp_path / "answer.json"
    assert run_agent(unit, out).returncode == 0
    official_check(unit, json.loads(out.read_text()))


def test_entities_without_names_and_30_rows(tmp_path):
    unit = _clone("t4-auction-btc-202411-us7", tmp_path / "u")
    task = json.loads((unit / "task.json").read_text())
    rows = []
    for i in range(30):
        e = dict(task["entities"][i % len(task["entities"])])
        e.pop("name", None)
        e["entity_id"] = f"{e['entity_id']}_{i}"
        rows.append(e)
    task["entities"] = rows
    (unit / "task.json").write_text(json.dumps(task))
    _rehash(unit)
    out = tmp_path / "answer.json"
    assert run_agent(unit, out).returncode == 0
    answer = json.loads(out.read_text())
    assert len(answer["entity_predictions"]) == 30
    official_check(unit, answer)


def test_all_documents_after_cutoff_still_exits_cleanly(tmp_path):
    unit = _clone("t4-EXAMPLE-eps-beat", tmp_path / "u")
    task = json.loads((unit / "task.json").read_text())
    task["cutoff_date"] = "2000-01-01"
    (unit / "task.json").write_text(json.dumps(task))
    _rehash(unit)
    out = tmp_path / "answer.json"
    proc = run_agent(unit, out)
    assert proc.returncode == 0
    assert out.exists()  # nothing is citable; the unit cannot pass, but the process never crashes


def test_unreachable_model_endpoint_falls_back(tmp_path):
    out = tmp_path / "answer.json"
    proc = run_agent(UNITS / "t4-cpicomp-202410-us11", out,
                     {"MODEL_ENDPOINT": "http://127.0.0.1:9", "MODEL_NAME": "x", "MODEL_TOKEN": "t",
                      "T4_LLM_DEADLINE": "60"})
    assert proc.returncode == 0
    official_check(UNITS / "t4-cpicomp-202410-us11", json.loads(out.read_text()))


# ------------------------------------------------------------------ calibration regressions
@pytest.mark.parametrize("unit,eid,plan,expect_numeric", [
    ("t4-EXAMPLE-eps-beat", "AAPL", {"interval_quantity": "label_only"}, True),
    ("t4-macrorev-20240930-us6", "HOUST_2024-07_20241018", {"interval_quantity": "label_only"}, True),
    ("t4-credit-event-2023", "BBY", {"interval_quantity": "probability 0-1", "interval_is_probability": True}, False),
])
def test_interval_lands_on_the_numeric_quantity(unit, eid, plan, expect_numeric):
    sys.path.insert(0, str(AGENT))
    from t4agent.calibrate import finalize
    from t4agent.predict import Forecast
    from t4agent.task import load_task

    spec = load_task(UNITS / unit / "task.json")
    ent = spec.entity(eid)
    out = finalize(spec, Forecast(eid, label=spec.labels[0]), plan, ent=ent)
    if expect_numeric:
        base = next(v for k, v in ent.items() if k in ("consensus_eps", "latest_precutoff_estimate"))
        assert out["lo"] < base < out["hi"] and (out["lo"], out["hi"]) != (0.0, 1.0)
    else:
        assert (out["lo"], out["hi"]) == (0.0, 1.0)


def _bare_spec(**kwargs):
    from t4agent.task import TaskSpec

    base = dict(raw={}, task_id="t", target_type="classification", target_type_trusted=True, target_name="outcome",
                labels=["yes", "no"], prompt="", cutoff=None, cutoff_str="", resolution_str="", interval_level=0.9,
                entities=[], entity_ids=[], unit_dir=None, card_params={})
    base.update(kwargs)
    return TaskSpec(**base)


def test_probability_interval_follows_the_value_schema_not_the_word():
    """A numeric quantity stays numeric when the question merely says "probability". A value field, unit, or
    answer schema that is itself a binary or 0-1 probability still gets [0, 1], with or without that word."""
    sys.path.insert(0, str(AGENT))
    from t4agent.calibrate import finalize, is_probability
    from t4agent.predict import Forecast
    from t4agent.task import load_task

    credit = load_task(UNITS / "t4-credit-event-2023" / "task.json")
    assert "probability" in credit.prompt.lower()
    assert is_probability(None, credit) is True
    out = finalize(credit, Forecast("BBY", label="no_event", median=0.05), None, ent=credit.entity("BBY"))
    assert (out["lo"], out["hi"]) == (0.0, 1.0)

    # The schema says probability; the question does not use the word.
    schema_prob = _bare_spec(
        raw={"target": {"name": "p_default", "type": "classification",
                        "value": {"type": "probability", "minimum": 0, "maximum": 1}}},
        target_name="p_default",
        prompt="Will the issuer default within twelve months? Give a point forecast.",
        entities=[{"entity_id": "X", "value": 1}], entity_ids=["X"],
    )
    assert "probability" not in schema_prob.prompt.lower()
    assert is_probability(None, schema_prob) is True
    out = finalize(schema_prob, Forecast("X", label="yes", median=0.4), None, ent=schema_prob.entity("X"))
    assert (out["lo"], out["hi"]) == (0.0, 1.0)

    # Binary 0/1 `value` column, no probability word.
    binary = _bare_spec(
        target_name="event_indicator",
        prompt="Did the event resolve in the window?",
        entities=[{"entity_id": "X", "value": 0}, {"entity_id": "Y", "value": 1}],
        entity_ids=["X", "Y"],
    )
    assert is_probability(None, binary) is True
    out = finalize(binary, Forecast("X", label="yes", median=0), None, ent=binary.entity("X"))
    assert (out["lo"], out["hi"]) == (0.0, 1.0)

    # The word is in the question and the unit is a ratio. That word is not a signal, so the interval stays
    # on the ratio. A planner flag set true is a signal and wins, even over the ratio unit.
    numeric = _bare_spec(
        target_name="btc_vs_threshold",
        prompt="Classify demand. Mention a probability only as background; also give a probability interval "
               "in words. The quantity to submit is the bid-to-cover ratio.",
        entities=[{"entity_id": "X", "unit": "bid_to_cover_ratio", "latest_ratio": 2.6}],
        entity_ids=["X"],
    )
    assert "probability" in numeric.prompt.lower()
    assert is_probability(None, numeric) is False
    out = finalize(numeric, Forecast("X", label="yes"), None, ent=numeric.entity("X"))
    assert (out["lo"], out["hi"]) != (0.0, 1.0)
    assert out["lo"] < 2.6 < out["hi"]
    assert is_probability({"interval_is_probability": True}, numeric) is True
    out = finalize(numeric, Forecast("X", label="yes"), {"interval_is_probability": True}, ent=numeric.entity("X"))
    # v21.11 (T4-002): the flag alone no longer wins over a row quantity outside [0, 1] (a 2.6 ratio is no probability)
    assert (out["lo"], out["hi"]) != (0.0, 1.0) and out["lo"] < 2.6 < out["hi"]

    # A numeric type bounded to [0, 1] is a probability. The bounds clause does not need the word, and a
    # latest spread on the row does not override it. A title that only says "probability" next to a ratio does not.
    bounded = _bare_spec(
        raw={"target": {"value": {"type": "number", "minimum": 0, "maximum": 1}}},
        prompt="Did the event resolve?",
        entities=[{"entity_id": "X", "latest_spread": 150}], entity_ids=["X"],
    )
    assert is_probability(None, bounded) is True
    out = finalize(bounded, Forecast("X", label="yes", median=0.4), None, ent=bounded.entity("X"))
    assert (out["lo"], out["hi"]) == (0.0, 1.0)
    titled = _bare_spec(
        raw={"target": {"title": "Probability the ratio exceeds 2.5", "unit": "bid_to_cover_ratio"}},
        entities=[{"entity_id": "X", "unit": "bid_to_cover_ratio", "latest_ratio": 2.6}], entity_ids=["X"],
    )
    assert is_probability(None, titled) is False
    enum_ratio = _bare_spec(
        raw={"target": {"value": {"enum": [0, 1], "unit": "ratio"}}},
        entities=[{"entity_id": "X", "latest_ratio": 2.6}], entity_ids=["X"],
    )
    assert is_probability(None, enum_ratio) is True
    typed_ratio = _bare_spec(
        raw={"target": {"value": {"type": "probability", "unit": "ratio"}}},
        entities=[{"entity_id": "X"}], entity_ids=["X"],
    )
    assert is_probability(None, typed_ratio) is True

    # latest_spread is the level column quantity_baseline would read. It must not veto a real probability
    # signal: the planner flag, or interval_leg false. The bare word in the question is not that signal.
    spread_rows = [
        {"entity_id": "A", "latest_spread": 150, "consensus_probability": 0.04},
        {"entity_id": "B", "latest_spread": 80, "consensus_probability": 0.2},
    ]
    spread = _bare_spec(
        prompt="Give a probability the spread widens, and a probability interval.",
        entities=spread_rows, entity_ids=["A", "B"],
    )
    assert "probability" in spread.prompt.lower()
    assert is_probability(None, spread) is False
    assert is_probability({"interval_is_probability": True}, spread) is True
    out = finalize(spread, Forecast("A", label="yes", median=0.04), {"interval_is_probability": True},
                   ent=spread.entity("A"))
    assert (out["lo"], out["hi"]) == (0.0, 1.0)
    leg = _bare_spec(
        card_params={"interval_leg": False},
        prompt="Give a probability the spread widens.",
        entities=spread_rows, entity_ids=["A", "B"],
    )
    assert is_probability(None, leg) is True
    out = finalize(leg, Forecast("A", label="yes", median=0.04), None, ent=leg.entity("A"))
    assert (out["lo"], out["hi"]) == (0.0, 1.0)

    # v21.10: a regression takes [0, 1] only from the planner flag, as in v21.8, even when its schema names a
    # probability (see test_v2110_*). A basis-point unit stays numeric when the question merely says
    # "probability". The planner flag true still wins over that unit.
    prob_reg = _bare_spec(
        raw={"target": {"name": "p_event", "type": "regression", "unit": "probability"}},
        target_type="regression", target_name="p_event", labels=[],
        prompt="Predict the outcome.",
        entities=[{"entity_id": "X"}], entity_ids=["X"],
    )
    assert is_probability(None, prob_reg) is False
    out = finalize(prob_reg, Forecast("X", median=0.3), {"interval_is_probability": True}, ent=prob_reg.entity("X"))
    # v21.11 (T4-002): a regression whose task names no measured unit still takes the planner's probability
    assert (out["lo"], out["hi"]) == (0.0, 1.0) and abs(out["point_forecast"] - 0.3) <= 0.05
    num_reg = _bare_spec(
        target_type="regression", target_name="yield_change_bps", labels=[],
        prompt="State the probability of a move and a probability interval, but submit the change in basis points.",
        entities=[{"entity_id": "X", "unit": "bps_change", "start_yield_pct": 3.7}],
        entity_ids=["X"],
    )
    assert is_probability(None, num_reg) is False
    assert is_probability({"interval_quantity": "probability 0-1"}, num_reg) is False
    out = finalize(num_reg, Forecast("X", median=10.0), {"interval_quantity": "probability 0-1"}, ent=num_reg.entity("X"))
    assert (out["lo"], out["hi"]) != (0.0, 1.0)
    assert out["hi"] - out["lo"] > 2
    # v21.11 (T4-002): the planner flag alone no longer clamps a basis-point regression into [0, 1]
    assert is_probability({"interval_is_probability": True}, num_reg) is False
    out = finalize(num_reg, Forecast("X", median=10.0), {"interval_is_probability": True}, ent=num_reg.entity("X"))
    assert abs(out["point_forecast"] - 10.0) <= 0.5 and out["hi"] - out["lo"] > 2


def test_regression_interval_never_degenerate_without_model_range():
    sys.path.insert(0, str(AGENT))
    from t4agent.calibrate import finalize
    from t4agent.predict import Forecast
    from t4agent.task import load_task

    spec = load_task(UNITS / "t4-fomc-curve-20220728" / "task.json")
    out = finalize(spec, Forecast("UST2Y"), None, ent=spec.entity("UST2Y"))
    assert out["hi"] - out["lo"] >= 100  # bps: never a +/-1 band
    out = finalize(spec, Forecast("UST2Y", median=20, p05=20, p95=20), None, ent=spec.entity("UST2Y"))
    assert out["hi"] - out["lo"] >= 100


# ------------------------------------------------------------------ v2 components (no model needed)
def _v2():
    sys.path.insert(0, str(AGENT))


def test_stat_baselines_use_only_the_entitys_own_series():
    _v2()
    from t4agent.corpus import load_corpus, map_entities
    from t4agent.stats import entity_forecast
    from t4agent.task import load_task

    spec = load_task(UNITS / "t4-auction-btc-202411-us7" / "task.json")
    corpus = load_corpus(UNITS / "t4-auction-btc-202411-us7" / "corpus", spec)
    specific, shared = map_entities(spec, corpus)
    for ent in spec.entities:
        own = specific[ent["entity_id"]]
        sf = entity_forecast(corpus, own + shared, ent, spec.target_name, spec.horizon_days, set(own), policy=_eligible_policy())
        assert sf is not None and sf.series.doc_id in own  # its own tenor's table, never another's
        assert min(sf.series.values) <= sf.value <= max(sf.series.values)
        assert sf.q90 and sf.q90 > 0


def test_label_rules_are_evaluated_safely_and_coherently():
    _v2()
    from t4agent.labels import implied_probs

    rules = {"beat": "Q > consensus_eps * (1 + threshold_pct)", "miss": "Q < consensus_eps * (1 - threshold_pct)",
             "inline": "otherwise"}
    ent = {"consensus_eps": 1.5, "threshold_pct": 0.05}
    p = implied_probs(rules, ["beat", "miss", "inline"], ent, 1.54, 1.40, 1.70)
    assert max(p, key=p.get) == "inline" and abs(sum(p.values()) - 1) < 1e-9
    assert implied_probs({"beat": "Q > __import__('os').getcwd()"}, ["beat"], ent, 1.5, 1.4, 1.6) is None


def test_label_probabilities_are_read_off_the_widened_range():
    """v11: off the range the answer submits (calibrate.quantity_range), which widens a model range that has no
    own scale behind it."""
    _v2()
    from types import SimpleNamespace

    import pytest
    from t4agent.calibrate import quantity_range
    from t4agent.labels import implied_probs
    from t4agent.predict import Forecast

    rules = {"positive_reaction": "Q > 1", "negative_reaction": "Q < -1", "flat": "otherwise"}
    labels = ["positive_reaction", "negative_reaction", "flat"]
    spec = SimpleNamespace(target_type="classification", target_name="reaction", prompt="", entities=[], raw={})
    plan = {"typical_abs_deviation": 2, "interval_is_probability": False}
    fc = Forecast(eid="X", source="llm", n_samples=1)
    fc.median, fc.p05, fc.p95 = 0.0, -2.5, 2.5  # a too-narrow model range around "no move"
    raw = implied_probs(rules, labels, {}, fc.median, fc.p05, fc.p95)
    point, lo, hi = quantity_range(spec, fc, plan)
    wide = implied_probs(rules, labels, {}, point, lo, hi)
    assert (lo, hi) == (-5.0, 5.0)
    assert max(raw, key=raw.get) == "flat" and wide["flat"] < wide["positive_reaction"]
    fc.p05, fc.p95 = None, 1.0  # one-sided range: mirrored, then floored by the planner's typical deviation
    assert quantity_range(spec, fc, plan)[1:] == pytest.approx((-3.29, 3.29))


def test_key_facts_surface_per_share_earnings_lines():
    _v2()
    from t4agent.corpus import load_corpus, map_entities
    from t4agent.facts import key_fact_chunks
    from t4agent.retrieval import BM25, entity_query
    from t4agent.task import load_task

    spec = load_task(UNITS / "t4-eps-growth-2024Q3-banks" / "task.json")
    corpus = load_corpus(UNITS / "t4-eps-growth-2024Q3-banks" / "corpus", spec)
    specific, _ = map_entities(spec, corpus)
    facts = key_fact_chunks(corpus, BM25(corpus.chunks), set(specific["GS"]), entity_query(spec, spec.entity("GS")))
    eps = [c.text for cat, c in facts if cat == "per-share earnings"]
    assert any("8.62" in t for t in eps)  # "Diluted earnings per common share (EPS) was $8.62 ..."


# ------------------------------------------------------------------ v7: interval score, not coverage first
def test_numeric_intervals_are_calibrated_not_coverage_hedged():
    """Scorer 5.2.x charges width (interval score), so the v3-v6 x4 coverage hedge is gone: the model's range is
    widened once for its measured narrowness and nothing more; probabilities stay [0, 1]."""
    _v2()
    from t4agent.calibrate import finalize
    from t4agent.config import WIDEN
    from t4agent.predict import Forecast
    from t4agent.task import load_task

    spec = load_task(UNITS / "t4-fomc-curve-20240918" / "task.json")
    out = finalize(spec, Forecast("UST2Y", median=10.0, p05=-20.0, p95=40.0), None, ent=spec.entity("UST2Y"))
    assert abs((out["hi"] - out["point_forecast"]) - WIDEN * 30) <= 1  # widened model range only
    assert out["lo"] <= 0.0  # zero change (the naive rule) stays inside
    spec = load_task(UNITS / "t4-credit-event-2023" / "task.json")
    out = finalize(spec, Forecast("BBY", label="no_event", median=0.05), {"interval_is_probability": True},
                   ent=spec.entity("BBY"))
    assert (out["lo"], out["hi"]) == (0.0, 1.0)


def test_a_far_naive_value_pulls_the_range_halfway():
    """A forecast that departs from the naive value hedges with half the gap, not the whole of it."""
    _v2()
    from t4agent.calibrate import finalize
    from t4agent.predict import Forecast
    from t4agent.task import load_task

    spec = load_task(UNITS / "t4-fomc-curve-20240918" / "task.json")
    out = finalize(spec, Forecast("UST2Y", median=-60.0, p05=-70.0, p95=-50.0), None, ent=spec.entity("UST2Y"))
    assert -60 + 20 < out["hi"] < 0  # [-80, -40] reaches halfway towards zero change, to about -20
    assert abs(out["hi"] - (-20.0)) <= 1


def test_repair_prompt_spells_out_the_schema_and_unusable_replies_fail():
    """A reply cut off while reasoning has no JSON to copy field names from: the repair prompt must
    carry the schema, and a parsed reply without usable forecasts must count as a failure."""
    _v2()
    from t4agent.corpus import load_corpus, map_entities
    from t4agent.llm import HouseClient
    from t4agent.predict import Evidence, Forecaster
    from t4agent.prompts import repair_messages
    from t4agent.task import load_task

    spec = load_task(UNITS / "t4-eps-yoy-2023Q2-mixed" / "task.json")
    msg = repair_messages("notes without any JSON", ["AMD", "AMGN"], spec, "EPS in USD")[1]["content"]
    assert '"probs"' in msg and '"up": p' in msg and '"down": p' in msg and '"quantity"' in msg
    # each entity's own row is shown, and only the group's rows (no copying numbers across entities)
    assert '"entity_id":"AMD"' in msg and '"entity_id":"AMGN"' in msg and '"entity_id":"IBM"' not in msg
    corpus = load_corpus(UNITS / "t4-eps-yoy-2023Q2-mixed" / "corpus", spec)
    fc = Forecaster(spec, corpus, Evidence(spec, corpus, *map_entities(spec, corpus)), HouseClient(deadline=0), 0)
    assert fc._usable({"entity_id": "AMD", "probs": {"up": 0.3, "down": 0.7}})
    assert fc._usable({"entity_id": "AMD", "label": "down"})
    assert not fc._usable({"entity_id": "AMD", "eps": 0.12})


def test_unconfirmed_outlier_moves_halfway_and_its_range_covers_both():
    _v2()
    import pytest
    from t4agent.corpus import load_corpus, map_entities
    from t4agent.llm import HouseClient
    from t4agent.predict import Evidence, Forecast, Forecaster
    from t4agent.task import load_task

    unit = UNITS / "t4-eps-growth-2024Q3-banks"
    spec = load_task(unit / "task.json")
    corpus = load_corpus(unit / "corpus", spec)
    f = Forecaster(spec, corpus, Evidence(spec, corpus, *map_entities(spec, corpus)), HouseClient(deadline=0), 0)
    f.stat = {e["entity_id"]: None for e in spec.entities}
    forecasts = {}
    for i, ent in enumerate(spec.entities):
        fc = Forecast(eid=ent["entity_id"], source="llm", n_samples=1)
        fc.median = 118.6 if fc.eid == "GS" else float(i)
        fc.p05, fc.p95 = fc.median - 10, fc.median + 10
        forecasts[fc.eid] = fc
    f.recheck(forecasts)  # no time left for a second look
    gs, ref = forecasts["GS"], f.outlier_refs["GS"]
    assert not gs.rechecked and gs.median == pytest.approx((118.6 + ref) / 2)
    assert gs.p05 <= ref - 10 and gs.p95 >= 128.6
    others = [fc for fc in forecasts.values() if fc.eid != "GS"]
    assert all(fc.p95 - fc.p05 == 20 for fc in others) and set(f.outlier_refs) == {"GS"}


def test_label_follows_the_calibrated_quantity_and_ignores_rules_on_another_scale():
    _v2()
    from t4agent.corpus import load_corpus, map_entities
    from t4agent.labels import implied_probs
    from t4agent.llm import HouseClient
    from t4agent.predict import Evidence, Forecaster
    from t4agent.task import load_task

    unit = UNITS / "t4-postearn-20240201-megacap"
    spec = load_task(unit / "task.json")
    corpus = load_corpus(unit / "corpus", spec)
    f = Forecaster(spec, corpus, Evidence(spec, corpus, *map_entities(spec, corpus)), HouseClient(deadline=0), 0)
    f.plan = {"typical_abs_deviation": 2, "label_rules": {"positive_reaction": "Q > 1", "negative_reaction": "Q < -1",
                                                          "flat": "otherwise"}}
    # the model leans on the middle band while giving a narrow range around "no move"
    item = {"entity_id": "AAPL", "probs": {"positive_reaction": 0.29, "negative_reaction": 0.29, "flat": 0.42},
            "quantity": {"median": 0.0, "p05": -2.5, "p95": 2.5}}
    fc = f._aggregate_one(spec.entity("AAPL"), [item])
    assert fc.label != "flat" and fc.probs["flat"] < 0.33
    # a quantity on another scale than the rule (a revision amount against a level threshold) is ignored
    assert implied_probs({"up": "Q > 710409", "down": "otherwise"}, ["up", "down"], {}, 540.0, 300.0, 800.0) is None


def test_ranking_points_keep_their_order_after_rounding():
    _v2()
    from t4agent.calibrate import finalize
    from t4agent.predict import Forecast
    from t4agent.task import load_task

    spec = load_task(UNITS / "t4-cotpos-202411-us10" / "task.json")
    ids = [e["entity_id"] for e in spec.entities][:3]
    pts = [finalize(spec, Forecast(eid, median=m, p05=m - 12.0, p95=m + 12.0), None, ent=spec.entity(eid))["point_forecast"]
           for eid, m in zip(ids, (0.31, -0.24, 0.12))]
    assert len(set(pts)) == 3 and pts[0] > pts[2] > pts[1]  # no ties manufactured by the wide coverage hedge


def test_truncated_plan_is_retried_once_without_thinking():
    _v2()
    import time

    from t4agent.corpus import load_corpus, map_entities
    from t4agent.llm import HouseClient, Reply
    from t4agent.predict import Evidence, Forecaster
    from t4agent.task import load_task

    unit = UNITS / "t4-macrorev-20240930-us6"
    spec = load_task(unit / "task.json")
    corpus = load_corpus(unit / "corpus", spec)
    f = Forecaster(spec, corpus, Evidence(spec, corpus, *map_entities(spec, corpus)), HouseClient(deadline=0), 0)
    calls: list[dict] = []
    replies = [Reply("", "reasoning that ran into the output cap", "length", 4000, 1.0),
               Reply('{"interval_quantity": "revised value in series units", "label_rules": null}', "", "stop", 40, 1.0)]

    class Stub:
        remaining, deadline = 23, time.time() + 400

        def chat(self, msgs, **kw):
            calls.append(kw)
            return replies.pop(0)

    f.client = Stub()
    plan = f.run_plan()
    assert plan and plan["interval_quantity"] == "revised value in series units"
    assert len(calls) == 2 and calls[0].get("thinking", True) and calls[1]["thinking"] is False


def test_like_for_like_eps_maps_columns_by_the_tables_own_header():
    """Current period vs the same period a year earlier, whatever the column order or table shape."""
    _v2()
    import datetime as dt
    import json

    from t4agent.corpus import Doc
    from t4agent.facts import eps_like_for_like

    def line(unit, name):
        raw = json.loads((UNITS / unit / "corpus" / name).read_text())
        doc = Doc(name, dt.date.fromisoformat(raw["doc_date"]), raw["text"], {k: v for k, v in raw.items() if k != "text"})
        return eps_like_for_like(name, doc) or ""

    ibm = line("t4-eps-yoy-2023Q2-mixed", "EDGAR_0000051143_10Q_20230425.json")
    assert "quarter ending 2023-03: 1.02 vs 0.73" in ibm and "+39.7%" in ibm
    gs = line("t4-eps-growth-2024Q3-banks", "EDGAR_0000886982_10Q_20240802.json")
    assert "8.62 vs 3.08" in gs and "six months ending 2024-06: 20.21 vs 11.91" in gs and "11.59 vs 8.83" in gs
    amzn = line("t4-postearn-20240201-megacap", "EDGAR_0001018724_10Q_20231027.json")  # prior year listed first
    assert "quarter ending 2023-09: 0.94 vs 0.28" in amzn
    wfc = line("t4-eps-growth-2024Q3-banks", "EDGAR_0000072971_10Q_20240801.json")  # quarterly trend table
    assert "1.33 vs 1.25" in wfc and "quarter ending 2024-03: 1.20" in wfc
    pnc = line("t4-eps-growth-2024Q3-banks", "EDGAR_0000713676_10Q_20240802.json")  # 5-column summary is skipped
    assert "six months ending 2024-06: 6.48 vs 7.34" in pnc
    amd = line("t4-eps-yoy-2023Q2-mixed", "EDGAR_0000002488_10Q_20230503.json")  # 52/53-week calendar
    assert "-0.09 vs 0.56" in amd
    meta = line("t4-postearn-20240201-megacap", "EDGAR_0001326801_10Q_20231026.json")  # one column per share class
    assert "4.39 vs 1.64" in meta and "9.56 vs 6.82" in meta


def test_rate_series_baseline_defaults_to_the_recent_median_and_facts_reach_the_prompt():
    _v2()
    import time

    from t4agent.corpus import load_corpus, map_entities
    from t4agent.llm import HouseClient
    from t4agent.predict import Evidence, Forecaster
    from t4agent.task import load_task

    unit = UNITS / "t4-cpicomp-202410-us11"
    spec = load_task(unit / "task.json")
    corpus = load_corpus(unit / "corpus", spec)
    f = Forecaster(spec, corpus, Evidence(spec, corpus, *map_entities(spec, corpus)), HouseClient(deadline=0), time.time())
    # v17: a rate series defaults to its own AR(1) reversion (v3-v16: the 6-month median); not the last print.
    # v20: gasoline, energy and all items take the pump-price nowcast instead (see the v20 tests); core keeps AR(1)
    from t4agent.stats import _ar1_phi

    core = f.stat["CPI_CORE"]
    v = core.series.values
    assert core.method == "ar1" and abs(core.value - (v[-1] + _ar1_phi(v) * (sum(v) / len(v) - v[-1]))) < 1e-9
    assert core.value != v[-1] and min(v) < core.value < max(v)
    assert f.stat["CPI_GASOLINE"].series.values[-1] == -4.10  # the nowcast keeps the entity's own series for features
    unit = UNITS / "t4-eps-yoy-2023Q2-mixed"
    spec = load_task(unit / "task.json")
    corpus = load_corpus(unit / "corpus", spec)
    ev = Evidence(spec, corpus, *map_entities(spec, corpus))
    ev.build_packs([])
    text = ev.render(["IBM"])
    assert "LIKE-FOR-LIKE" in text and "1.02 vs 0.73" in text


def test_an_entitys_own_scale_beats_the_unit_wide_deviation_floor():
    _v2()
    from t4agent.calibrate import finalize
    from t4agent.predict import Forecast
    from t4agent.task import load_task

    spec = load_task(UNITS / "t4-cpicomp-202410-us11" / "task.json")
    plan = {"typical_abs_deviation": 3.06, "decimals": 2}  # one number for gasoline and core CPI alike
    own = finalize(spec, Forecast("CPI_CORE", median=0.26, p05=0.16, p95=0.36, stat_q90=0.3), plan,
                   ent=spec.entity("CPI_CORE"))
    assert own["hi"] - own["lo"] <= 2 * 0.3 + 0.05  # its own backtest scale
    unit = finalize(spec, Forecast("CPI_CORE", median=0.26, p05=0.16, p95=0.36), plan, ent=spec.entity("CPI_CORE"))
    assert unit["hi"] - unit["lo"] >= 2 * 1.645 * 3.06 - 0.1  # no own scale: the unit-wide floor


def test_eps_anchor_is_a_seasonal_random_walk_from_the_filings_own_numbers():
    _v2()
    import time

    import pytest
    from t4agent.corpus import load_corpus, map_entities
    from t4agent.eps import PHI, eps_anchor
    from t4agent.labels import directional_rules
    from t4agent.llm import HouseClient
    from t4agent.predict import Evidence, Forecaster
    from t4agent.task import load_task

    def unit(name):
        spec = load_task(UNITS / name / "task.json")
        corpus = load_corpus(UNITS / name / "corpus", spec)
        return spec, corpus, *map_entities(spec, corpus)

    spec, corpus, specific, shared = unit("t4-eps-growth-2024Q3-banks")
    gs = eps_anchor(corpus, set(specific["GS"]), spec.entity("GS"), "pct_change", policy=_eligible_policy())
    assert gs.level == pytest.approx(5.47 + PHI * (8.62 - 3.08)) and gs.value == pytest.approx((gs.level / 5.47 - 1) * 100)
    spec, corpus, specific, shared = unit("t4-postearn-20240201-megacap")
    assert eps_anchor(corpus, set(specific["AAPL"]), spec.entity("AAPL"), "level", policy=_eligible_policy()) is None  # no EPS target period
    assert directional_rules(["up", "down"], 1.53) == {"up": "Q > 1.53", "down": "otherwise"}
    assert directional_rules(["beat", "miss", "inline"], 1.5) is None

    spec, corpus, specific, shared = unit("t4-eps-yoy-2023Q2-mixed")
    ev = Evidence(spec, corpus, specific, shared)
    f = Forecaster(spec, corpus, ev, HouseClient(deadline=0), time.time())
    f.plan = {"interval_quantity": "GAAP diluted EPS in USD/share", "label_rules": None}
    f.build_anchors()
    assert f.anchor["IBM"].level == pytest.approx(1.53 + PHI * (1.02 - 0.73))
    assert "statistical anchor" in f.baselines(["IBM"])
    only = f.aggregate({})  # no model reply at all: labels come from the anchors, coherently
    assert only["IBM"].label == "up" and only["DOW"].label == "down" and only["AMD"].label == "down"


def test_revision_prior_reads_the_same_stage_off_the_vintage_table():
    _v2()
    import time

    from t4agent.corpus import load_corpus, map_entities
    from t4agent.llm import HouseClient
    from t4agent.predict import Evidence, Forecaster
    from t4agent.revisions import revision_prior, vintage_table
    from t4agent.task import load_task

    unit = UNITS / "t4-macrorev-20240930-us6"
    spec = load_task(unit / "task.json")
    corpus = load_corpus(unit / "corpus", spec)
    specific, shared = map_entities(spec, corpus)
    dates, rows = vintage_table(corpus.docs["ALFRED_DGORDER_VINTAGES_20240930"].text)
    assert len(dates) == 8 and rows["2024-07"][-1] == 289587.0 and rows["2024-08"][:7] == [None] * 7
    dg = revision_prior(corpus, set(specific["DGORDER_2024-07_20241025"]), spec.entity("DGORDER_2024-07_20241025"), policy=_eligible_policy())
    assert (dg.stage, dg.steps, dg.up, dg.n) == (3, 1, 0, 2) and dg.p_up < 0.5  # 2024-04 has hidden early estimates
    pay = revision_prior(corpus, set(specific["PAYEMS_2024-08_20241101"]), spec.entity("PAYEMS_2024-08_20241101"), policy=_eligible_policy())
    assert (pay.stage, pay.steps) == (1, 2)  # resolved two monthly releases after the latest vintage
    f = Forecaster(spec, corpus, Evidence(spec, corpus, specific, shared), HouseClient(deadline=0), time.time())
    assert set(f.prior) == set(spec.entity_ids) and "revision prior" in f.baselines(["PAYEMS_2024-07_20241004"])
    only = f.aggregate({})  # no model reply: labels follow the base rates
    assert only["PAYEMS_2024-07_20241004"].label == "down" and only["DGORDER_2024-07_20241025"].label == "down"


def test_a_rate_limit_pauses_the_client_and_the_retry_succeeds(monkeypatch):
    _v2()
    import time

    from t4agent.llm import HouseClient

    for k, v in {"MODEL_ENDPOINT": "http://house.invalid", "MODEL_NAME": "house", "MODEL_TOKEN": "t"}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("T4_EXTRA_BODY", raising=False)

    class Resp:
        def __init__(self, code, body=None):
            self.status_code, self._body, self.text = code, body or {}, ""

        def json(self):
            return self._body

    ok = {"choices": [{"message": {"content": '{"a": 1}'}, "finish_reason": "stop"}], "usage": {"completion_tokens": 3}}
    replies = [Resp(429), Resp(200, ok)]
    c = HouseClient(deadline=time.time() + 27)  # a cool-down is capped by the deadline margin: 2 s here
    monkeypatch.setattr(c._session, "post", lambda *a, **k: replies.pop(0))
    t0 = time.time()
    rep = c.chat([{"role": "user", "content": "x"}], reasoning_budget=None, max_attempts=2)
    assert rep.content == '{"a": 1}' and time.time() - t0 >= 1.5 and c._streak == 0 and c.used == 2


def test_revenue_like_for_like_reaches_the_prompt_next_to_eps():
    _v2()
    from t4agent.corpus import load_corpus, map_entities
    from t4agent.predict import Evidence
    from t4agent.task import load_task

    unit = UNITS / "t4-eps-yoy-2023Q2-mixed"
    spec = load_task(unit / "task.json")
    corpus = load_corpus(unit / "corpus", spec)
    ev = Evidence(spec, corpus, *map_entities(spec, corpus))
    ev.build_packs([])
    text = ev.render(["DOW"])
    assert "revenue 11,851 vs 15,264 (-22.4%)" in text and "-0.13 vs 2.11" in text


def test_review_blends_a_second_opinion_only_for_reported_entities(monkeypatch):
    """v21: the review call is off by default (T4_REVIEW=1 turns it on); its code path keeps its tests."""
    _v2()
    import time

    from t4agent import config

    monkeypatch.setattr(config, "REVIEW", 1)

    from t4agent.corpus import load_corpus, map_entities
    from t4agent.llm import HouseClient, Reply
    from t4agent.predict import Evidence, Forecast, Forecaster
    from t4agent.task import load_task

    unit = UNITS / "t4-eps-yoy-2023Q2-mixed"
    spec = load_task(unit / "task.json")
    corpus = load_corpus(unit / "corpus", spec)
    ev = Evidence(spec, corpus, *map_entities(spec, corpus))
    ev.build_packs([])  # computes the like-for-like lines, as the agent does before any model call
    f = Forecaster(spec, corpus, ev, HouseClient(deadline=0), 0)
    f.plan = {"interval_quantity": "GAAP diluted EPS in USD/share", "label_rules": {"up": "Q > prior_year_q_eps",
                                                                                    "down": "otherwise"}}
    forecasts = {e: Forecast(eid=e, source="llm", label="down", probs={"up": 0.3, "down": 0.7}, median=1.4, p05=1.2,
                             p95=1.6) for e in spec.entity_ids}
    calls = []
    reply = ('{"revisions": [{"entity_id": "IBM", "probs": {"up": 0.9, "down": 0.1}, '
             '"quantity": {"median": 1.8, "p05": 1.6, "p95": 2.0}, "reason": "Q1 same-quarter EPS rose 39.7%"}]}')

    class Stub:
        remaining, deadline, speed = 10, time.time() + 300, 100.0

        def chat(self, msgs, **kw):
            calls.append(msgs)
            return Reply(reply, "", "stop", 50, 1.0)

    f.client = Stub()
    before = dict(forecasts["AMD"].probs)
    f.review(forecasts)
    ibm = forecasts["IBM"]
    assert len(calls) == 1 and "LIKE-FOR-LIKE" in calls[0][1]["content"]
    assert ibm.probs["up"] > 0.5 and ibm.label == "up" and ibm.p95 >= 2.0  # range now covers both opinions
    assert forecasts["AMD"].probs == before  # not reported, not touched


def test_layout_trades_depth_for_coverage_on_a_slow_route():
    _v2()
    import time

    from t4agent import config
    from t4agent.corpus import load_corpus, map_entities
    from t4agent.llm import HouseClient
    from t4agent.predict import Evidence, Forecaster
    from t4agent.task import load_task

    unit = UNITS / "t4-macrorev-20240930-us6"
    spec = load_task(unit / "task.json")
    corpus = load_corpus(unit / "corpus", spec)
    f = Forecaster(spec, corpus, Evidence(spec, corpus, *map_entities(spec, corpus)), HouseClient(deadline=0), 0)

    class Route:
        def __init__(self, speed):
            self.speed, self.deadline = speed, time.time() + 400

    f.client = Route(150.0)  # fast route: small groups, deep thinking
    f.choose_layout()
    assert (f.group_size, f.analyst_budget) == config.LAYOUTS[0]
    f.client = Route(15.0)  # slow route: broader groups, shallower thinking, every group still sampled twice
    f.choose_layout()
    assert f.group_size > config.LAYOUTS[0][0] and f.analyst_budget < config.LAYOUTS[0][1]


def test_features_are_computed_from_the_units_own_tables_and_filings():
    _v2()
    import time

    from t4agent.corpus import load_corpus, map_entities
    from t4agent.features import entity_features, going_concern_line
    from t4agent.llm import HouseClient
    from t4agent.predict import Evidence, Forecaster
    from t4agent.tables import parse_tables
    from t4agent.task import load_task

    def unit(name):
        spec = load_task(UNITS / name / "task.json")
        corpus = load_corpus(UNITS / name / "corpus", spec)
        specific, shared = map_entities(spec, corpus)
        f = Forecaster(spec, corpus, Evidence(spec, corpus, specific, shared), HouseClient(deadline=0), time.time())
        return spec, corpus, specific, entity_features(spec, corpus, specific, f.stat)

    spec, corpus, specific, feats = unit("t4-cpicomp-202410-us11")
    assert len(parse_tables("EIA_GASREGW_20241028", corpus.docs["EIA_GASREGW_20241028"].text)) == 1  # 2-column table
    energy = " ".join(feats["CPI_ENERGY"])
    assert "moves with 'Gasoline (all types)': slope +0.50 per unit, correlation +0.97" in energy
    gas = " ".join(feats["CPI_GASOLINE"])
    assert "DRIVER" in gas and "(-2.4%, not seasonally adjusted)" in gas and "03 -4.9" in gas and "08 +2.1" in gas
    assert "Food" not in gas  # a chance correlation over nine months is not reported

    spec, corpus, specific, feats = unit("t4-credit-event-2023")
    assert "concludes that substantial doubt exists" in " ".join(feats["BBBY"])
    assert "plans alleviate it" in " ".join(feats["WE"])  # doubt raised, then said to be alleviated
    assert going_concern_line(corpus, set(specific["RAD"])) is None

    spec, corpus, specific, feats = unit("t4-eps-growth-2024Q3-banks")
    assert "prior_year_q_eps 5.47 (rank 1 of 8" in " ".join(feats["GS"])


# ------------------------------------------------------------------ v7: scorer 5.2.2 claims and reasons
def _rail():
    sys.path.insert(0, str(TRACK / "baselines" / "guardrails_example"))
    import citation_rail
    return citation_rail


@pytest.mark.parametrize("unit", ALL_UNITS)
def test_every_claim_passes_the_official_claim_rules_and_reasons_are_judged(unit, tmp_path):
    """Fallback answers on every public unit: no false claim by the scorer's own deterministic rules (factor 1),
    and a submitted_reasons block the grader would judge in full."""
    R = _rail()
    out = tmp_path / "answer.json"
    proc = run_agent(UNITS / unit, out)
    assert proc.returncode == 0, proc.stderr[-2000:]
    ans = json.loads(out.read_text())
    assert R.check_claim_rules(ans, UNITS / unit, token_counter=None) == [] or all(
        f.code == "claim_tokens_unchecked" for f in R.check_claim_rules(ans, UNITS / unit, token_counter=None))
    assert R.claim_penalty_preview(ans, UNITS / unit, token_counter=None)["factor"] == 1.0
    task = json.loads((UNITS / unit / "task.json").read_text())
    corpus = R.load_corpus(UNITS / unit / "corpus")
    assert 1 <= len(ans.get("submitted_reasons") or []) <= 3
    assert R.check_submitted_reasons(ans, corpus, task["cutoff_date"]) == []
    assert all(p["judged"] for p in R.reasons_judged(ans, corpus, task["cutoff_date"]))
    for p in ans["entity_predictions"]:
        for c in p["claims"]:
            text = json.loads((UNITS / unit / "corpus" / f"{c['doc_id']}.json").read_text())["text"]
            assert c["claim"] == text[c["span_start"]:c["span_end"]]  # verbatim quote of the cited span


def test_manifest_labels_decide_which_documents_an_entity_may_cite():
    _v2()
    from t4agent.corpus import load_corpus, map_entities
    from t4agent.task import load_task

    spec = load_task(UNITS / "t4-credit-event-2023" / "task.json")
    corpus = load_corpus(UNITS / "t4-credit-event-2023" / "corpus", spec)
    specific, shared = map_entities(spec, corpus)
    assert specific["BBBY"] == ["EDGAR_0000886158_10Q_20230126"] and shared == []
    assert corpus.admits("BBBY", "EDGAR_0000886158_10Q_20230126")
    assert not corpus.admits("BBY", "EDGAR_0000886158_10Q_20230126")
    spec = load_task(UNITS / "t4-cpicomp-202410-us11" / "task.json")
    corpus = load_corpus(UNITS / "t4-cpicomp-202410-us11" / "corpus", spec)
    assert all(corpus.admits("CPI_ENERGY", d) for d in corpus.docs)  # every CPI document is marked shared


def test_reason_validation_drops_what_the_grader_would_refuse():
    _v2()
    from t4agent.corpus import load_corpus
    from t4agent.reasons import validate
    from t4agent.task import load_task

    spec = load_task(UNITS / "t4-credit-event-2023" / "task.json")
    corpus = load_corpus(UNITS / "t4-credit-event-2023" / "corpus", spec)
    did = "EDGAR_0000886158_10Q_20230126"
    text = corpus.docs[did].text
    s = text.index("substantial doubt")
    s = text.rfind(" ", 0, s - 40) + 1
    e = text.find(".", s) + 1
    good = {"reason_id": "x", "premise": text[s:e], "mechanism": "Going-concern doubt signals a likely default.",
            "answer_implication": "Supports credit_event for BBBY.", "scope": {"entities": ["BBBY", "NOPE"]},
            "citations": [{"doc_id": did, "span_start": s, "span_end": e}]}
    denied = dict(good, mechanism="As the leaderboard shows, default follows.")
    not_verbatim = dict(good, premise=text[s:e] + " (our words)")
    bad_cite = dict(good, citations=[{"doc_id": did, "span_start": s, "span_end": s + 9000}])
    dup = dict(good, reason_id="y")
    out = validate(spec, corpus, [denied, not_verbatim, bad_cite, good, dup])
    assert len(out) == 1 and out[0]["premise"] == text[s:e]
    assert out[0]["scope"] == {"entities": ["BBBY"]}


def test_two_own_scales_combine_geometrically():
    """The backtested error of the baseline and the series' step volatility, when both exist, are combined
    geometrically (each overstates a different kind of series)."""
    _v2()
    from t4agent.calibrate import finalize
    from t4agent.predict import Forecast
    from t4agent.task import load_task

    spec = load_task(UNITS / "t4-cpicomp-202410-us11" / "task.json")
    out = finalize(spec, Forecast("CPI_ALLITEMS", median=0.2, p05=0.15, p95=0.25, stat_q90=0.46), {"decimals": 2},
                   ent=spec.entity("CPI_ALLITEMS"), table_half=0.237)
    half = (0.46 * 0.237) ** 0.5
    assert abs((out["hi"] - out["lo"]) - 2 * half) <= 0.03


# ------------------------------------------------------------------ v8: plans grounded in the corpus, measured scales
def test_plan_notes_with_remembered_figures_or_later_dates_are_dropped():
    """The planner wrote 'maintain 5.25-5.50%' for a meeting whose statement announces a cut, and 'outcome of the
    2024-11-05 election' for a unit cut off on 2024-10-31: neither may reach an analyst."""
    _v2()
    from t4agent.corpus import load_corpus
    from t4agent.grounding import sanitize_plan
    from t4agent.task import load_task

    spec = load_task(UNITS / "t4-fomc-curve-20240918" / "task.json")
    scheduled = ["Scheduled FOMC meeting concluding November 7, 2024",  # the corpus names both dates
                 "U.S. general election on 2024\u201111\u201105 inside the window"]  # non-breaking hyphens
    plan = {"key_drivers": ["FOMC policy rate decision and forward guidance (maintain 5.25\u20115.50% range)",
                            "SEP median of 4.4 percent for end-2024 versus market pricing", *scheduled,
                            "Result of the 2024\u201111\u201105 election for the curve",
                            "Market pricing after the December 18, 2024 meeting"],
            "label_cues": {"up": ["yields rose 35 bps"]}, "naive_baseline": "Zero change (persistence of the yield)"}
    dropped = sanitize_plan(plan, spec, load_corpus(UNITS / "t4-fomc-curve-20240918" / "corpus", spec))
    assert plan["key_drivers"] == ["SEP median of 4.4 percent for end-2024 versus market pricing", *scheduled]
    assert plan["label_cues"] == {"up": []} and plan["naive_baseline"].startswith("Zero change")
    assert len(dropped) == 4 and "5.25" in dropped[0] and "2024-11-05" in dropped[1] and "2024-12-18" in dropped[2]

    spec = load_task(UNITS / "t4-cotpos-202411-us10" / "task.json")
    plan = {"key_drivers": ["Outcome of the 2024-11-05 U.S. general election (policy expectations for rates)",
                            "Current positioning extremes in gold and silver"]}
    sanitize_plan(plan, spec, load_corpus(UNITS / "t4-cotpos-202411-us10" / "corpus", spec))
    assert plan["key_drivers"] == ["Current positioning extremes in gold and silver"]


def test_the_planner_reads_the_evidence_and_analysts_get_its_notes_as_hints():
    _v2()
    from t4agent.corpus import load_corpus, map_entities
    from t4agent.predict import Evidence, inventory
    from t4agent.prompts import analyst_messages, planner_messages
    from t4agent.task import load_task

    for unit, needle in (("t4-fomc-curve-20240918", "4-3/4 to 5 percent"),  # small corpus: the documents in full
                         ("t4-eps-yoy-2023Q2-mixed", "LIKE-FOR-LIKE")):  # large: computed facts per entity
        spec = load_task(UNITS / unit / "task.json")
        corpus = load_corpus(UNITS / unit / "corpus", spec)
        ev = Evidence(spec, corpus, *map_entities(spec, corpus))
        ev.build_packs([])
        user = planner_messages(spec, inventory(corpus), "", ev.digest())[1]["content"]
        assert needle in user and len(user) < 60000
    msgs = analyst_messages(spec, {"key_drivers": ["x"]}, spec.entity_ids[:1], "", "", "EPS")
    assert "PLANNER NOTES (written before the evidence was read" in msgs[1]["content"]
    assert "checklist" in msgs[0]["content"]


def test_a_measured_own_scale_stands_and_an_estimated_one_meets_the_model_halfway_on_the_log_scale():
    """Macro revisions: the model gave ten times the revision scale. A scale measured on the entity's own history
    is not widened by such a range; an extrapolated one moves to the geometric mean; neither is ever narrowed."""
    _v2()
    from t4agent.calibrate import finalize
    from t4agent.predict import Forecast
    from t4agent.task import load_task

    spec = load_task(UNITS / "t4-fomc-curve-20240918" / "task.json")
    ent = spec.entity("UST10Y")
    measured = finalize(spec, Forecast("UST10Y", median=0.0, p05=-90.0, p95=90.0, stat_q90=30.0, stat_reliable=True),
                        None, ent=ent)
    assert measured["hi"] == 30 and measured["lo"] == -30
    estimated = finalize(spec, Forecast("UST10Y", median=0.0, p05=-90.0, p95=90.0, stat_q90=30.0), None, ent=ent)
    assert abs(estimated["hi"] - (30.0 * 90.0) ** 0.5) <= 1
    narrow = finalize(spec, Forecast("UST10Y", median=0.0, p05=-5.0, p95=5.0, stat_q90=30.0), None, ent=ent)
    assert narrow["hi"] == 30


def test_scales_are_marked_measured_only_when_backtested():
    _v2()
    from t4agent.corpus import load_corpus, map_entities
    from t4agent.predict import Evidence, Forecaster, reliable_scale
    from t4agent.llm import HouseClient
    from t4agent.task import load_task

    seen = {}
    for unit in ("t4-cpicomp-202410-us11", "t4-fomc-curve-20240918", "t4-macrorev-20240930-us6"):
        spec = load_task(UNITS / unit / "task.json")
        corpus = load_corpus(UNITS / unit / "corpus", spec)
        ev = Evidence(spec, corpus, *map_entities(spec, corpus))
        f = Forecaster(spec, corpus, ev, HouseClient(deadline=0), 0)
        seen[unit] = {e: reliable_scale(f.stat.get(e) or f.prior.get(e)) for e in spec.entity_ids}
        if unit.startswith("t4-macrorev"):
            houst = f.prior["HOUST_2024-07_20241018"]
    assert all(seen["t4-cpicomp-202410-us11"].values())  # nine monthly prints: an error quantile over six origins
    assert not any(seen["t4-fomc-curve-20240918"].values())  # ten daily closes extrapolated to seven weeks
    macro = seen["t4-macrorev-20240930-us6"]
    # two same-stage revisions (one of them zero) meet the series' six all-stage revisions on the log scale
    assert macro["HOUST_2024-07_20241018"] and houst.n_scale == 6 and 5 < houst.q90 < 15
    assert not macro["RSAFS_2024-08_20241115"]  # three revisions in all: still an estimate


def test_a_regression_forecast_without_a_baseline_is_combined_with_the_naive_rule():
    """FOMC 2022 has no yield history in its corpus: v7 submitted the model's medians unshrunk."""
    _v2()
    from t4agent.corpus import load_corpus, map_entities
    from t4agent.llm import HouseClient
    from t4agent.predict import Evidence, Forecast, Forecaster
    from t4agent.task import load_task

    unit = UNITS / "t4-fomc-curve-20220728"
    spec = load_task(unit / "task.json")
    corpus = load_corpus(unit / "corpus", spec)
    f = Forecaster(spec, corpus, Evidence(spec, corpus, *map_entities(spec, corpus)), HouseClient(deadline=0), 0)
    assert f.stat.get("UST2Y") is None
    fc = Forecast("UST2Y", source="llm", median=100.0, p05=40.0, p95=160.0)
    f._combine_with_stats(fc)
    assert (fc.median, fc.p05, fc.p95) == (50.0, -10.0, 110.0)  # halfway to zero change, range moved with it


def test_a_unit_wide_deviation_is_ignored_when_rows_are_in_different_units():
    _v2()
    from t4agent.predict import unit_deviation
    from t4agent.task import load_task

    plan = {"typical_abs_deviation": "17506"}  # what the planner gave for payrolls, durable goods and an index
    assert unit_deviation(load_task(UNITS / "t4-macrorev-20240930-us6" / "task.json"), plan) is None
    assert unit_deviation(load_task(UNITS / "t4-fomc-curve-20240918" / "task.json"), {"typical_abs_deviation": 30}) == 30


def test_a_revision_quantity_is_combined_with_the_base_rate_before_the_label_is_read():
    """The interval of a revision unit is centred on the model's revised value: like every other baseline, the
    vintage table's expected next estimate gets equal weight, unless the model's number is on another scale."""
    _v2()
    from t4agent.corpus import load_corpus, map_entities
    from t4agent.llm import HouseClient
    from t4agent.predict import Evidence, Forecaster
    from t4agent.task import load_task

    unit = UNITS / "t4-macrorev-20240930-us6"
    spec = load_task(unit / "task.json")
    corpus = load_corpus(unit / "corpus", spec)
    f = Forecaster(spec, corpus, Evidence(spec, corpus, *map_entities(spec, corpus)), HouseClient(deadline=0), 0)
    eid = "PAYEMS_2024-07_20241004"
    prior = f.prior[eid].value
    item = {"entity_id": eid, "probs": {"up": 0.2, "down": 0.8},
            "quantity": {"median": 158500.0, "p05": 158400.0, "p95": 158600.0}}
    fc = f._aggregate_one(spec.entity(eid), [item])
    assert abs(fc.median - 0.5 * (158500.0 + prior)) < 1e-6 and abs((fc.p95 - fc.p05) - 200.0) < 1e-6
    item["quantity"] = {"median": 0.09, "p05": -0.1, "p95": 0.3}  # a percent revision, not the level
    assert f._aggregate_one(spec.entity(eid), [item]).median == 0.09


def test_a_request_after_a_cool_down_never_outlives_the_deadline(monkeypatch):
    """The read timeout is measured after the cool-down, not before it: a call made after a long pause once came
    back 94 s past the deadline and pushed the final answer to 494 s."""
    _v2()
    import time

    from t4agent.llm import HouseClient

    for k, v in (("MODEL_ENDPOINT", "http://127.0.0.1:9"), ("MODEL_NAME", "m"), ("MODEL_TOKEN", "t")):
        monkeypatch.setenv(k, v)
    client = HouseClient(deadline=time.time() + 30)
    client._cool_until = time.time() + 3.0  # within the cool-down cap (deadline - 25 s)
    seen = {}

    class Ok:
        status_code = 200

        @staticmethod
        def json():
            return {"choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}], "usage": {}}

    def post(url, headers, data, timeout):
        seen["read"] = timeout[1]
        return Ok()

    monkeypatch.setattr(client._session, "post", post)
    client.chat([{"role": "user", "content": "x"}], reasoning_budget=None, max_attempts=1)
    assert seen["read"] <= 22.5  # 30 s left, 3 s of cool-down, 5 s margin (measured before the wait: 25 s)


def test_local_reasoning_field_translation_is_inert_in_production(monkeypatch):
    """Local testing on OpenRouter's free Nemotron endpoint caps thinking with reasoning.max_tokens (the top-level
    reasoning_budget is ignored there); T4_REASONING_FIELD translates it. Unset (as in the image), the request
    body is the House body: reasoning_budget, no 'reasoning' field."""
    _v2()
    import time

    from t4agent.llm import HouseClient

    for k, v in (("MODEL_ENDPOINT", "http://127.0.0.1:9"), ("MODEL_NAME", "m"), ("MODEL_TOKEN", "t")):
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("T4_EXTRA_BODY", raising=False)
    bodies = []

    class Ok:
        status_code = 200

        @staticmethod
        def json():
            return {"choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}], "usage": {}}

    def post(url, headers, data, timeout):
        bodies.append(json.loads(data))
        return Ok()

    def send(field, thinking):
        if field:
            monkeypatch.setenv("T4_REASONING_FIELD", field)
        else:
            monkeypatch.delenv("T4_REASONING_FIELD", raising=False)
        client = HouseClient(deadline=time.time() + 60)
        monkeypatch.setattr(client._session, "post", post)
        client.chat([{"role": "user", "content": "x"}], reasoning_budget=2800, thinking=thinking, max_attempts=1)
        return bodies[-1]

    house = send("", True)
    assert house["reasoning_budget"] == 2800 and "reasoning" not in house
    assert house["chat_template_kwargs"] == {"enable_thinking": True}
    local = send("openrouter", True)
    assert local["reasoning"] == {"max_tokens": 2800} and "reasoning_budget" not in local
    assert send("openrouter", False)["reasoning"] == {"enabled": False}
    assert "T4_REASONING_FIELD" not in (AGENT / "Dockerfile").read_text()


def test_label_rules_fall_back_to_the_conditions_the_entity_table_states():
    """A plan without label rules (or no plan at all) left an EPS-beat label to the model's own probabilities,
    which favour the middle band; the table's beat/miss conditions define the labels on the quantity."""
    _v2()
    from t4agent.labels import implied_probs, table_rules
    from t4agent.task import load_task

    spec = load_task(UNITS / "t4-EXAMPLE-eps-beat" / "task.json")
    ent = spec.entity("AAPL")
    rules = table_rules(spec.labels, ent)
    assert rules == {"beat": "Q > consensus_eps * (1 + threshold_pct)",
                     "miss": "Q < consensus_eps * (1 - threshold_pct)", "inline": "otherwise"}
    for med, label in ((1.70, "beat"), (1.52, "inline"), (1.30, "miss")):  # the label follows the quantity
        probs = implied_probs(rules, spec.labels, ent, med, med - 0.22, med + 0.22)
        assert max(probs, key=probs.get) == label and abs(sum(probs.values()) - 1) < 1e-9
    assert table_rules(["up", "down"], {"entity_id": "X"}) is None  # no conditions in the table


# ------------------------------------------------------------------ v9: a level rule in the target's units
def _forecaster(unit_name: str, plan: dict):
    _v2()
    from t4agent.corpus import load_corpus, map_entities
    from t4agent.llm import HouseClient
    from t4agent.predict import Evidence, Forecaster
    from t4agent.task import load_task

    unit = UNITS / unit_name
    spec = load_task(unit / "task.json")
    corpus = load_corpus(unit / "corpus", spec)
    f = Forecaster(spec, corpus, Evidence(spec, corpus, *map_entities(spec, corpus)), HouseClient(deadline=0), 0)
    f.plan = plan
    return f


def test_a_yield_level_in_percent_becomes_a_change_in_basis_points():
    """v8 subtracted start_yield_pct from a yield level in percent and submitted the result as basis points:
    3.05 against 2.85 became 0.2 bp instead of 20 bp, and every FOMC 2022 forecast ended at 0."""
    f = _forecaster("t4-fomc-curve-20220728", {
        "interval_quantity": "Yield change in basis points",
        "target_from_level": {"level_quantity": "U.S. Treasury constant-maturity yield level (in percent)",
                              "base_field": "start_yield_pct", "formula": "difference"}})
    rule = f.level_rule()
    assert rule["scale"] == 100.0
    fc = f._aggregate_one(f.spec.entity("UST2Y"), [{"median": 3.05, "p05": 2.55, "p95": 3.55}])
    assert [round(x, 6) for x in (fc.median, fc.p05, fc.p95)] == [20.0, -30.0, 70.0]
    assert "implied level 3.05 against base 2.85" in f._describe_forecast("UST2Y", fc, rule)


def test_level_rules_already_in_the_targets_units_are_unchanged():
    f = _forecaster("t4-eps-growth-2024Q3-banks", {"target_from_level": {
        "level_quantity": "GAAP diluted EPS (USD/share)", "base_field": "prior_year_q_eps", "formula": "pct_change"}})
    rule = f.level_rule()
    base = f.spec.entity("JPM")["prior_year_q_eps"]
    assert rule["scale"] == 1.0 and round(f._to_target("JPM", base * 1.1, rule), 9) == 10.0
    f = _forecaster("t4-cotpos-202411-us10", {"target_from_level": {
        "level_quantity": "net noncommercial position as percent of open interest",
        "base_field": "net_pct_oi_20241022", "formula": "difference"}})
    assert f.level_rule()["scale"] == 1.0


def test_a_level_rule_whose_units_cannot_be_reconciled_is_not_used():
    """A percent target against a level in dollars or contracts has no fixed factor: the analysts then
    forecast the target itself, as they do without a level rule."""
    f = _forecaster("t4-eps-growth-2024Q3-banks", {"target_from_level": {
        "level_quantity": "GAAP diluted EPS (USD/share)", "base_field": "prior_year_q_eps", "formula": "difference"}})
    assert f.level_rule() is None
    f = _forecaster("t4-cotpos-202411-us10", {"target_from_level": {
        "level_quantity": "net noncommercial position in contracts", "base_field": "net_noncommercial_20241022",
        "formula": "difference"}})
    assert f.level_rule() is None


def test_the_target_unit_is_read_off_the_task_before_the_plan():
    _v2()
    from t4agent.predict import target_unit
    from t4agent.task import load_task

    units = {u: target_unit(load_task(UNITS / u / "task.json"), {"interval_quantity": "basis points"})
             for u in ("t4-fomc-curve-20240918", "t4-cotpos-202411-us10", "t4-cpicomp-202410-us11",
                       "t4-eps-growth-2024Q3-banks", "t4-auction-btc-202411-us7")}
    assert units == {"t4-fomc-curve-20240918": "bps", "t4-cotpos-202411-us10": "pct",
                     "t4-cpicomp-202410-us11": "pct", "t4-eps-growth-2024Q3-banks": "pct",
                     "t4-auction-btc-202411-us7": "bps"}  # the task names no unit: the plan's description decides
    assert target_unit(load_task(UNITS / "t4-auction-btc-202411-us7" / "task.json"), None) is None


# ------------------------------------------------------------------ v10: reasons planned against the drivers
def test_the_drivers_the_task_statement_names_are_split_into_a_list():
    _v2()
    from t4agent.reasons import task_drivers, unit_drivers
    from t4agent.task import load_task

    def drivers(unit):
        return task_drivers(load_task(UNITS / unit / "task.json").prompt)

    assert drivers("t4-eps-growth-2024Q3-banks") == ["net interest income trends", "credit provisions",
                                                      "investment-banking/markets revenue", "fees", "expense trends"]
    assert drivers("t4-macrorev-20240930-us6") == ["persistent revision direction", "typical revision size",
                                                    "annual-update schedules", "the BLS preliminary benchmark announcement"]
    assert len(drivers("t4-credit-event-2023")) == 6 and "covenant headroom" in drivers("t4-credit-event-2023")
    spec = load_task(UNITS / "t4-eps-yoy-2023Q2-mixed" / "task.json")  # the statement lists none: the plan's
    assert unit_drivers(spec, {"key_drivers": ["one-off items in the base quarter", "x"]}) == [
        "one-off items in the base quarter"]


def test_an_implication_that_disagrees_with_the_answers_is_caught():
    """v9's bank reason called every listed bank 'positive growth', one of them submitted at -2.8; a credit reason
    listed credit_event and no_event entities together."""
    _v2()
    from t4agent.reasons import code_implication, implication_issues
    from t4agent.task import load_task

    banks = load_task(UNITS / "t4-eps-growth-2024Q3-banks" / "task.json")
    preds = [{"entity_id": "JPM", "point_forecast": 10.6, "interval": {"lo": -30.0, "hi": 50.0}},
             {"entity_id": "BAC", "point_forecast": -2.8, "interval": {"lo": -40.0, "hi": 35.0}}]
    assert implication_issues(banks, preds, "Supports positive YoY EPS growth for JPM (10.6%) and BAC (-2.8%).")
    assert implication_issues(banks, preds, "Supports higher EPS for JPM (12.0%).")  # not the submitted value
    assert not implication_issues(banks, preds, "Lifts JPM (10.6%) while BAC (-2.8%) declines.")
    assert code_implication(banks, preds, ["JPM", "BAC"]) == (
        "Consistent with the submitted values: increases for JPM (10.6); decreases for BAC (-2.8).")
    credit = load_task(UNITS / "t4-credit-event-2023" / "task.json")
    labels = [{"entity_id": "BBBY", "label": "credit_event"}, {"entity_id": "RAD", "label": "no_event"}]
    assert implication_issues(credit, labels, "Supports credit_event for BBBY and RAD.")
    assert not implication_issues(credit, labels, "Supports credit_event for BBBY; no_event for RAD.")
    revs = load_task(UNITS / "t4-macrorev-20240930-us6" / "task.json")
    row = [{"entity_id": "PAYEMS_2024-07_20241004", "label": "down"}]
    assert implication_issues(revs, row, "Revised up for PAYEMS_2024-07_20241004.")


def test_a_driver_passage_needs_two_of_its_words_not_one_rare_one():
    """'going-concern language' once matched a passage on cultural and language differences."""
    _v2()
    from t4agent.corpus import load_corpus, map_entities
    from t4agent.predict import Evidence
    from t4agent.reasons import driver_passages
    from t4agent.task import load_task

    unit = UNITS / "t4-credit-event-2023"
    spec = load_task(unit / "task.json")
    corpus = load_corpus(unit / "corpus", spec)
    ev = Evidence(spec, corpus, *map_entities(spec, corpus))
    hits = driver_passages(spec, corpus, ev, set(), "going-concern language")
    assert hits and all("going concern" in corpus.chunk(c).text.lower() for c in hits)


def _bank_reasons_setup():
    _v2()
    from t4agent.corpus import load_corpus, map_entities
    from t4agent.llm import HouseClient
    from t4agent.predict import Evidence, Forecast, Forecaster
    from t4agent.reasons import _owner, _premise, candidates, unit_drivers
    from t4agent.task import load_task

    unit = UNITS / "t4-eps-growth-2024Q3-banks"
    spec = load_task(unit / "task.json")
    corpus = load_corpus(unit / "corpus", spec)
    ev = Evidence(spec, corpus, *map_entities(spec, corpus))
    ev.build_packs([])
    f = Forecaster(spec, corpus, ev, HouseClient(deadline=0), 0)
    forecasts = {e: Forecast(eid=e, source="llm", median=5.0 if e != "BAC" else -3.0) for e in spec.entity_ids}
    preds = [{"entity_id": e, "point_forecast": forecasts[e].median, "interval": {"lo": -40.0, "hi": 45.0}}
             for e in spec.entity_ids]
    pid: dict[str, str] = {}
    for i, (cid, _) in enumerate(candidates(spec, corpus, ev, forecasts, unit_drivers(spec, None)), 1):
        owner = _owner(ev, corpus.chunk(cid).doc_id)
        if owner and owner not in pid and _premise(corpus, cid) is not None:
            pid[owner] = f"P{i}"
    return spec, corpus, ev, f, forecasts, preds, pid


def _stub(reply: str):
    import time

    from t4agent.llm import Reply

    seen = []

    class Stub:
        enabled, remaining, deadline, speed = True, 10, time.time() + 300, 100.0

        def chat(self, msgs, **kw):
            seen.append(msgs[1]["content"])
            return Reply(reply, "", "stop", 50, 1.0)

    return Stub(), seen


def test_model_reasons_cite_a_passage_per_driver_and_rebuild_a_wrong_implication():
    from t4agent.reasons import build_reasons, validate

    spec, corpus, ev, f, forecasts, preds, pid = _bank_reasons_setup()
    reply = ('{"reasons": [{"drivers": [1, 2], "premise_id": "%s", "support_ids": ["%s"], '
             '"mechanism": "Net interest income and provisions both move the comparison against last year.", '
             '"answer_implication": "Supports positive EPS growth for JPM (5) and BAC (-3).", "entities": ["JPM", "BAC"]}]}'
             % (pid["JPM"], pid["BAC"]))
    stub, seen = _stub(reply)
    reasons = build_reasons(spec, corpus, ev, f, forecasts, preds, stub)
    assert "D5: expense trends" in seen[0] and "support_ids" in seen[0]
    assert "ANSWERS TO EXPLAIN FIRST" in seen[0] and "BAC: -3" in seen[0]
    r = reasons[0]
    assert len(r["citations"]) >= 3 and validate(spec, corpus, reasons) == reasons
    # 'positive growth' over BAC (-3): the reasoning points the wrong way for BAC; as the only reason it is kept,
    # its implication rebuilt from the answers without BAC
    assert r["answer_implication"] == "Consistent with the submitted values: increases for JPM (5)."


def test_a_reason_speaks_only_for_the_entities_whose_evidence_it_cites():
    """v10 once carried JPM's, Citi's and PNC's filings over to GS and MS."""
    from t4agent.reasons import model_reasons, unit_drivers

    spec, corpus, ev, f, forecasts, preds, pid = _bank_reasons_setup()
    reply = ('{"reasons": [{"premise_id": "%s", "mechanism": "Fee income rose across the large banks in the latest '
             'quarter.", "answer_implication": "Lifts C (5) and GS (5).", "entities": ["C", "GS"]}]}' % pid["C"])
    stub, _ = _stub(reply)
    out = model_reasons(spec, corpus, ev, f, forecasts, preds, stub, unit_drivers(spec, None))
    assert [r["answer_implication"] for r in out] == ["Consistent with the submitted values: increases for C (5)."]
    assert out[0]["scope"] == {"entities": ["C"]}


def test_the_answers_furthest_from_the_naive_baseline_come_first():
    _v2()
    from t4agent.reasons import focus_entities
    from t4agent.task import load_task

    banks = load_task(UNITS / "t4-eps-growth-2024Q3-banks" / "task.json")
    preds = [{"entity_id": e, "point_forecast": v} for e, v in
             (("BAC", -2.2), ("C", 4.8), ("GS", 40.5), ("JPM", 12.7), ("MS", 16.7), ("PNC", 0.3))]
    assert focus_entities(banks, preds, None) == ["GS", "MS", "JPM", "C"]  # growth: the naive rule is no change
    credit = load_task(UNITS / "t4-credit-event-2023" / "task.json")
    labels = [{"entity_id": e, "label": lab} for e, lab in
              (("BBBY", "credit_event"), ("BBY", "no_event"), ("M", "no_event"), ("RAD", "no_event"))]
    assert focus_entities(credit, labels, None) == ["BBBY"]  # the calls against the most common label
    labels[3]["label"] = "credit_event"
    assert focus_entities(credit, labels, None) == []  # two labels as common as each other: nothing stands out


def test_an_announcement_return_is_never_given_a_narrow_scale():
    """The plan's single typical deviation (1.5 in one run) made a +/-1% 'flat' band the likeliest label of an
    earnings-day return; the a priori scale of such a return is about 4 points."""
    _v2()
    from t4agent import config
    from t4agent.calibrate import finalize
    from t4agent.labels import implied_probs
    from t4agent.predict import Forecast, event_return_sd
    from t4agent.task import load_task

    spec = load_task(UNITS / "t4-postearn-20240201-megacap" / "task.json")
    assert event_return_sd(spec) == config.EVENT_RETURN_SD
    assert event_return_sd(load_task(UNITS / "t4-cpicomp-202410-us11" / "task.json")) is None
    plan = {"typical_abs_deviation": 1.5, "interval_quantity": "one-day abnormal return in percent"}
    fc = Forecast("AAPL", label="flat", probs={"positive_reaction": 0.3, "negative_reaction": 0.3, "flat": 0.4},
                  median=0.2, p05=-1.0, p95=1.4)
    out = finalize(spec, fc, plan, ent=spec.entity("AAPL"))
    assert out["hi"] - out["lo"] >= 2 * config.Z90 * config.EVENT_RETURN_SD - 0.1
    rules = {"positive_reaction": "Q > 1", "negative_reaction": "Q < -1", "flat": "otherwise"}
    probs = implied_probs(rules, spec.labels, spec.entity("AAPL"), 0.2, out["lo"], out["hi"])
    assert probs["flat"] < max(probs["positive_reaction"], probs["negative_reaction"])


def test_small_units_take_more_samples():
    _v2()
    from t4agent import config

    assert config.MAX_SAMPLES == 9  # v9: 5, which left a six-entity unit at 8 of 23 requests


def test_passage_and_driver_numbers_never_reach_the_panel_and_qualifiers_do_not_hide_a_direction():
    """The model works with labels P1..Pn and D1..Dn that the panel never sees; a parenthetical qualifier
    ('PNC (-0.5% - offsets a negative provision effect)') must not hide 'positive forecasts' over a negative row."""
    _v2()
    from t4agent.reasons import _unlabel, implication_issues
    from t4agent.task import load_task

    text = _unlabel("For D1, the going-concern language in P2 (and P11) states doubt; P3 notes losses.",
                    ["going-concern language", "net losses"])
    assert "P2" not in text and "P11" not in text and "D1" not in text and "'going-concern language'" in text
    banks = load_task(UNITS / "t4-eps-growth-2024Q3-banks" / "task.json")
    preds = [{"entity_id": "JPM", "point_forecast": 12.9, "interval": {"lo": -30.0, "hi": 50.0}},
             {"entity_id": "PNC", "point_forecast": -0.5, "interval": {"lo": -40.0, "hi": 40.0}}]
    assert implication_issues(banks, preds, "Supports positive forecasts for JPM (12.9%), PNC (-0.5% - fee growth "
                                            "modestly offsets a negative provision effect).") == ["PNC called up"]


def test_a_reason_that_argues_against_the_answers_is_dropped():
    """Asked to cover every driver, the model once wrote 'the expense increase points to a lower EPS, but the
    submitted answer is up; therefore this reason does not apply to Amgen'."""
    _v2()
    from t4agent.reasons import self_denying

    assert self_denying("For Amgen the expense increase points to a lower EPS, but the submitted answer is 'up'; "
                        "therefore this reason does not apply to Amgen.")
    assert self_denying("This is not assigned to IBM.")
    assert not self_denying("Supports up for AMGN (3.49) and down for DOW (1.18).")
    # a driver that does not apply to an entity is an argument, not a denial (v10 once dropped two such reasons)
    assert not self_denying("Going-concern language does not apply to BBY or ODFL, which supports no_event for them.")
    # a reason stating its scope (v12's first sweep dropped two credit reasons on this)
    assert not self_denying("For BBBY this supports the submitted label credit_event. For all other entities the reason "
                            "does not apply because the evidence is specific to BBBY.")


def test_negations_and_comparisons_are_not_read_as_directions():
    """'No maturity is expected to decline' and 'the long end rises less than the front end' once held back a
    reason whose answers all rose."""
    _v2()
    from t4agent.reasons import implication_issues
    from t4agent.task import load_task

    fomc = load_task(UNITS / "t4-fomc-curve-20220728" / "task.json")
    preds = [{"entity_id": e, "point_forecast": v, "interval": {"lo": -80.0, "hi": 110.0}}
             for e, v in (("UST2Y", 15.0), ("UST10Y", 5.0), ("UST30Y", 2.0))]
    assert not implication_issues(fomc, preds, "Supports the submitted rises for UST2Y (15), UST10Y (5) and UST30Y "
                                               "(2); no maturity is expected to decline.")
    assert not implication_issues(fomc, preds, "UST30Y (2) rises less than UST2Y (15).")
    assert implication_issues(fomc, preds, "Points to lower yields for UST2Y (15).") == ["UST2Y called down"]


def test_a_reason_pointing_the_wrong_way_for_an_entity_is_held_back_when_others_remain():
    from t4agent.reasons import model_reasons, unit_drivers

    spec, corpus, ev, f, forecasts, preds, pid = _bank_reasons_setup()
    one = ('{"premise_id": "%s", "mechanism": "Fee income rose across the large banks in the latest quarter.", '
           '"answer_implication": "%s", "entities": %s}')
    reply = '{"reasons": [' + ", ".join([
        one % (pid["BAC"], "Supports positive growth for BAC (-3).", '["BAC"]'),
        one % (pid["JPM"], "Lifts JPM (5).", '["JPM"]'),
        one % (pid["GS"], "Lifts GS (5).", '["GS"]')]) + "]}"
    stub, _ = _stub(reply)
    out = model_reasons(spec, corpus, ev, f, forecasts, preds, stub, unit_drivers(spec, None))
    assert [r["answer_implication"] for r in out] == ["Lifts JPM (5).", "Lifts GS (5)."]


# ------------------------------------------------------------------ v11: one distribution for label and interval
def test_an_asymmetric_range_keeps_its_median():
    """v10 read a two-piece normal with the median as its mode: median 0, p05 -1, p95 9 gave P(Q > 1) = 0.77."""
    _v2()
    import pytest
    from t4agent.labels import _cdf, implied_probs

    s_lo, s_hi = 1 / 1.645, 9 / 1.645
    assert _cdf(0.0, 0.0, s_lo, s_hi) == pytest.approx(0.5)
    assert (_cdf(-1.0, 0.0, s_lo, s_hi), _cdf(9.0, 0.0, s_lo, s_hi)) == pytest.approx((0.05, 0.95), abs=1e-3)
    probs = implied_probs({"above": "Q > 1", "below": "otherwise"}, ["above", "below"], {}, 0.0, -1.0, 9.0)
    assert probs["above"] < 0.5


def test_the_label_agrees_with_the_quantity_the_answer_submits():
    """Built on IBM's public row: v10 submitted 2.0 against last year's 1.53, interval [1.0, 2.7], labelled 'down',
    because the label was read off the model's lower bound widened 2x and a mode-centred distribution."""
    _v2()
    import logging

    from t4agent import cli
    from t4agent.corpus import load_corpus, map_entities
    from t4agent.llm import HouseClient
    from t4agent.predict import Evidence, Forecaster
    from t4agent.task import load_task

    logging.disable(logging.CRITICAL)
    unit = UNITS / "t4-eps-yoy-2023Q2-mixed"
    spec = load_task(unit / "task.json")
    corpus = load_corpus(unit / "corpus", spec)
    ev = Evidence(spec, corpus, *map_entities(spec, corpus))
    ev.build_packs([])
    f = Forecaster(spec, corpus, ev, HouseClient(deadline=0), 0)
    f.plan = {"interval_quantity": "GAAP diluted EPS for the June-2023 quarter (USD/share)",
              "baseline_field": "prior_year_q_eps"}
    f.build_anchors()
    for p05, probs in ((0.4, {"up": 0.3, "down": 0.7}), (1.0, {"up": 0.02, "down": 0.98})):
        items = [{"probs": probs, "quantity": {"median": 2.257, "p05": p05, "p95": 2.6}}] * 5
        fcs = {e: f._aggregate_one(spec.entity(e), items) for e in spec.entity_ids}
        row = cli._rows(spec, corpus, ev, fcs, f.plan)["IBM"]
        assert row["point_forecast"] > spec.entity("IBM")["prior_year_q_eps"] and row["label"] == "up"
    logging.disable(logging.NOTSET)


def test_a_range_wholly_inside_one_label_decides_it():
    _v2()
    from types import SimpleNamespace

    from t4agent.predict import Forecast, Forecaster

    spec = SimpleNamespace(labels=["up", "down"], target_type="classification", target_name="eps_yoy_direction",
                           prompt="", entities=[{"entity_id": "X", "prior_year_q_eps": 1.0}], raw={},
                           entity=lambda eid: {"entity_id": "X", "prior_year_q_eps": 1.0})
    owner = SimpleNamespace(spec=spec, plan={"label_rules": {"up": "Q > prior_year_q_eps", "down": "otherwise"},
                                            "interval_is_probability": False},
                            _halves={}, prior={}, anchor={}, corpus=None, ev=None)
    owner._rule_sources = lambda eid: Forecaster._rule_sources(owner, eid)
    owner._rules = lambda eid: Forecaster._rules(owner, eid)
    owner._submitted_range = lambda fc, ent: Forecaster._submitted_range(owner, fc, ent)
    owner._blend_prior = lambda fc: None
    owner._blend_eps_direction = lambda fc: None
    fc = Forecast("X", model_probs={"up": 0.0, "down": 1.0}, median=1.5, p05=1.3, p95=1.7)
    Forecaster._settle_label(owner, fc, spec.entities[0])
    assert fc.label == "up"  # the whole submitted range is above last year's EPS, whatever the model's own odds


def test_a_level_column_never_forecasts_a_growth_rate():
    """A growth target matched an 'EPS' column and differenced it in dollars: scaled by 10, the 'growth' moved by 10."""
    _v2()
    import pytest
    from t4agent.stats import Series, forecast

    vals = [1.10, 1.20, 1.05, 1.30, 1.25, 1.40, 1.35, 1.50, 1.45, 1.60, 1.55, 1.70]
    dates = [f"20{18 + i // 4}-Q{i % 4 + 1}" for i in range(len(vals))]
    with pytest.raises(ValueError):
        forecast(Series("doc", "EPS", vals, dates, 91.0, False), "eps_yoy_growth_pct", 90, {"entity_id": "X"})
    f1 = forecast(Series("doc", "EPS growth %", vals, dates, 91.0, False), "eps_yoy_growth_pct", 90, {"entity_id": "X"})
    assert f1 is not None  # a growth column still serves a growth target


def test_a_percent_change_of_a_yield_is_refused():
    """A planner choosing pct_change on start_yield_pct turned 2.85% -> 3.05% into 702 bp in v9/v10."""
    _v2()
    from t4agent.predict import level_scale
    from t4agent.task import load_task

    fomc = load_task(UNITS / "t4-fomc-curve-20220728" / "task.json")
    level = "U.S. Treasury constant-maturity yield level (in percent)"
    assert level_scale(fomc, None, "start_yield_pct", level, "pct_change") is None
    assert level_scale(fomc, None, "start_yield_pct", level, "difference") == 100.0


# ------------------------------------------------------------------ v12: series in the target's own definition
def _table_corpus(header: str, rows: list[str]):
    from types import SimpleNamespace

    return SimpleNamespace(docs={"d": SimpleNamespace(text=header + "\n" + "\n".join(rows))})


def test_a_ratio_change_is_measured_on_its_numerator_in_the_starting_denominator():
    """'Change in net position as a percent of the starting open interest' is not the difference of the printed
    net-%-of-OI ratio: when open interest grows, the two can have opposite signs (gold, 2024-06-18 to 07-23)."""
    _v2()
    from t4agent.stats import _steps, find_series

    oi = [1000, 1000, 1000, 1000, 1000, 1000, 1000, 2000]
    net = [100, 110, 120, 130, 140, 150, 160, 240]
    rows = [f"2024-0{1 + i // 4}-{10 + i:02d} | {o} | {n} | {n / o * 100:.2f}" for i, (o, n) in enumerate(zip(oi, net))]
    corpus = _table_corpus("report_date | open_interest | noncomm_net | net_%OI", rows)
    s = find_series(corpus, ["d"], {"entity_id": "X"}, "net_positioning_change_pct_oi", {"d"})
    assert s is not None and s.denom == [float(x) for x in oi] and s.denom_scale == 100.0
    assert _steps(s)[-1] == 8.0  # +80 contracts over the starting 1,000: +8 points (the printed ratio fell 4 points)
    bad = [r.rsplit("|", 1)[0] + "| 99.00" for r in rows]  # a 'ratio' the table does not confirm: no baseline
    assert find_series(_table_corpus("report_date | open_interest | noncomm_net | net_%OI", bad), ["d"],
                       {"entity_id": "X"}, "net_positioning_change_pct_oi", {"d"}) is None


def test_a_growth_column_serving_a_growth_target_is_forecast_at_its_next_value():
    """v11 differenced a growth-rate column like a level: growth held at 10% forecast 0%."""
    _v2()
    from t4agent.stats import Series, forecast

    dates = [f"20{18 + i // 4}-Q{i % 4 + 1}" for i in range(12)]
    f = forecast(Series("doc", "EPS growth %", [10.0] * 12, dates, 91.0, False), "eps_yoy_growth_pct", 90,
                 {"entity_id": "X"})
    assert f.value == 10.0


def test_the_interval_scale_never_reads_a_level_column_for_a_growth_target():
    """v11's table scale read EPS dollars for a percent-growth target: scaled by 10, the half-width moved by 10."""
    _v2()
    from types import SimpleNamespace

    from t4agent.calibrate import table_halves

    vals = [1.10, 1.20, 1.05, 1.30, 1.25, 1.40, 1.35, 1.50, 1.45, 1.60, 1.55, 1.70]
    for k in (1, 10):
        corpus = _table_corpus("quarter | EPS", [f"20{18 + i // 4}-Q{i % 4 + 1} | {v * k:.2f}" for i, v in enumerate(vals)])
        spec = SimpleNamespace(entities=[{"entity_id": "X"}], target_name="eps_yoy_growth_pct", horizon_days=90)
        ev = SimpleNamespace(specific={"X": ["d"]}, shared=[])
        assert table_halves(spec, corpus, ev) == {"X": None}


def test_a_reason_arguing_the_other_way_or_quoting_an_unnamed_peer_is_held_back():
    """v11: a positioning reason argued 'unwinding yen shorts turns the net position less negative' over an answer of
    -9.3 (its implication was rebuilt from the answers, hiding it); a GS reason quoted Citi's and Bank of America's
    figures as Goldman's. A peer's passage cited but not used is only dropped."""
    import re

    from t4agent.reasons import _figures, _premise, candidates, model_reasons, unit_drivers

    spec, corpus, ev, f, forecasts, preds, pid = _bank_reasons_setup()
    cands = candidates(spec, corpus, ev, forecasts, unit_drivers(spec, None))
    citi = _premise(corpus, cands[int(pid["C"][1:]) - 1][0])[0]
    figure = sorted(_figures(citi))[0]  # a number only Citi's passage carries
    one = ('{"premise_id": "%s", "support_ids": %s, "mechanism": "%s", "answer_implication": %s, '
           '"entities": [{"id": "%s", "argues": "%s"}]}')
    reply = '{"reasons": [' + ", ".join([
        one % (pid["GS"], '["%s"]' % pid["C"], f"Markets revenue reached {figure} in the quarter.", '"Supports GS (5)."',
               "GS", "up"),  # Citi's figure, Goldman's name: held back
        one % (pid["BAC"], "[]", "Expenses rose faster than revenue.", '{"BAC": "a decline of 3"}', "BAC", "up"),
        one % (pid["JPM"], '["%s"]' % pid["BAC"], "JPMorgan net interest income rose.", '"Lifts JPM (5)."', "JPM", "up"),
        one % (pid["C"], "[]", "Citi provisions fell.", '"Lifts C (5)."', "C", "up")]) + "]}"
    stub, seen = _stub(reply)
    out = model_reasons(spec, corpus, ev, f, forecasts, preds, stub, unit_drivers(spec, None))
    assert [r["answer_implication"] for r in out] == ["Lifts JPM (5).", "Lifts C (5)."]
    assert len(out[0]["citations"]) == 2  # the unused Bank of America passage was dropped, the reason kept
    assert '"argues"' in seen[0] and "'up' if the reason implies a positive value" in seen[0]


def test_declared_directions_are_read_against_the_answer():
    _v2()
    from t4agent.reasons import argued_against
    from t4agent.task import load_task

    banks = load_task(UNITS / "t4-eps-growth-2024Q3-banks" / "task.json")
    assert argued_against(banks, {"entity_id": "BAC", "point_forecast": -3.0}, "up")
    assert not argued_against(banks, {"entity_id": "BAC", "point_forecast": -3.0}, "down")
    assert not argued_against(banks, {"entity_id": "BAC", "point_forecast": -3.0}, "unclear")
    credit = load_task(UNITS / "t4-credit-event-2023" / "task.json")
    assert argued_against(credit, {"entity_id": "RAD", "label": "no_event"}, "credit_event")
    cpi = load_task(UNITS / "t4-cpicomp-202410-us11" / "task.json")  # a level: read against the naive value (0.22)
    assert argued_against(cpi, {"entity_id": "CPI_SHELTER", "point_forecast": 0.34}, "lower")
    assert not argued_against(cpi, {"entity_id": "CPI_SHELTER", "point_forecast": 0.34}, "higher")
    auction = load_task(UNITS / "t4-auction-btc-202411-us7" / "task.json")  # no naive value in the table: unread
    assert not argued_against(auction, {"entity_id": "AUC_10Y_20241105", "point_forecast": 2.6}, "lower")


def test_a_mechanism_discussing_an_entity_without_its_evidence_is_held_back():
    """v12's first sweeps: a reason citing Bank of America's and JPMorgan's filings also credited Citi's Visa gain."""
    from t4agent.reasons import model_reasons, unit_drivers

    spec, corpus, ev, f, forecasts, preds, pid = _bank_reasons_setup()
    one = ('{"premise_id": "%s", "mechanism": "%s", "answer_implication": "%s", "entities": [{"id": "%s", "argues": "up"}]}')
    reply = '{"reasons": [' + ", ".join([
        one % (pid["JPM"], "JPMorgan fees rose, and Citigroup's markets revenue also jumped.", "Lifts JPM (5).", "JPM"),
        one % (pid["GS"], "Goldman Sachs fees rose.", "Lifts GS (5).", "GS"),
        one % (pid["C"], "Citi provisions fell.", "Lifts C (5).", "C")]) + "]}"
    stub, _ = _stub(reply)
    out = model_reasons(spec, corpus, ev, f, forecasts, preds, stub, unit_drivers(spec, None))
    assert [r["answer_implication"] for r in out] == ["Lifts GS (5).", "Lifts C (5)."]


def test_an_unparseable_reasons_reply_is_retried_once_without_thinking():
    """On a route that ignores the thinking budget the reasons reply ran into the output cap and lost its JSON."""
    import time

    from t4agent.llm import Reply
    from t4agent.reasons import model_reasons, unit_drivers

    spec, corpus, ev, f, forecasts, preds, pid = _bank_reasons_setup()
    good = ('{"reasons": [{"premise_id": "%s", "mechanism": "JPMorgan fees rose.", "answer_implication": '
            '"Lifts JPM (5).", "entities": [{"id": "JPM", "argues": "up"}]}]}' % pid["JPM"])
    calls = []

    class Stub:
        enabled, remaining, deadline, speed = True, 3, time.time() + 300, 100.0

        def chat(self, msgs, **kw):
            calls.append(kw.get("thinking", True))
            return Reply("" if len(calls) == 1 else good, "thinking...", "length" if len(calls) == 1 else "stop", 50, 1.0)

    out = model_reasons(spec, corpus, ev, f, forecasts, preds, Stub(), unit_drivers(spec, None))
    # v13: one usable reason of three asks for replacements; a replacement repeating the premise is not used.
    # v21: up to two follow-up rounds (REASONS_ROUNDS) while fewer than three clean reasons stand
    assert calls == [True, False, True, True] and [r["answer_implication"] for r in out] == ["Lifts JPM (5)."]


# ------------------------------------------------------------------ v13: benchmark/annual updates in vintage tables
def test_a_benchmark_update_is_left_out_of_the_revision_base_rate():
    """The 2024-09-27 NIPA annual update moved every month of personal income by about +3%; v12 read it as the size
    and direction of the next regular revision (interval +/-5% of the level, mean revision +1.65%)."""
    _v2()
    import datetime as dt

    from t4agent.corpus import load_corpus, map_entities
    from t4agent.revisions import level_updates, revision_prior
    from t4agent.task import load_task

    d = [dt.date(2024, m, 28) for m in (6, 7, 8, 9)]
    rows = {"2024-01": [100.0, 100.0, 100.0, 103.0], "2024-02": [101.0, 101.0, 101.0, 104.0],
            "2024-05": [104.0, 103.9, 103.8, 107.0]}
    assert set(level_updates(d, rows)) == {3}  # two settled months revised at once
    rows["2024-02"] = [101.0, 101.2, 101.0, 104.0]  # one settled month alone is a regular revision
    assert level_updates(d, rows) == {}

    unit = UNITS / "t4-macrorev-20240930-us6"
    spec = load_task(unit / "task.json")
    corpus = load_corpus(unit / "corpus", spec)
    specific, shared = map_entities(spec, corpus)
    pi = revision_prior(corpus, set(specific["PI_2024-07_20241031"]) | set(shared), spec.entity("PI_2024-07_20241031"), policy=_eligible_policy())
    assert pi.q90 < 0.005 * pi.latest and abs(pi.mean_rel) < 0.005 and "2024-09-27" in pi.describe()
    for eid in ("DGORDER_2024-07_20241025", "PAYEMS_2024-08_20241101", "INDPRO_2024-08_20241017"):  # no update in these
        pr = revision_prior(corpus, set(specific[eid]) | set(shared), spec.entity(eid), policy=_eligible_policy())
        assert pr.updates == ""


# ------------------------------------------------------------------ v13: label bands from the task's own definitions
def test_label_bands_are_read_off_the_task_label_definitions():
    """A five-band growth label cannot be written in the planner's one-sided forms: v12 left the labels to the model's
    odds, and four of eight banks were labelled against their own submitted growth (-2.4% labelled 'decline')."""
    _v2()
    from t4agent.labels import assertion_rules, implied_probs, rules_parse

    bands = ["sharp_decline", "decline", "flat", "growth", "sharp_growth"]
    stated = {"sharp_decline": "growth below -20%", "decline": "growth from -20% to below -5%",
              "flat": "growth from -5% to +5%", "growth": "growth above +5% up to +20%", "sharp_growth": "growth above +20%"}
    rules = assertion_rules(bands, stated)
    assert rules["flat"] == "-5.0 < Q < 5.0" and rules["growth"] == "5.0 < Q < 20.0" and rules_parse(rules, bands, {})
    probs = implied_probs(rules, bands, {}, -2.4, -5.6, 0.9)
    assert max(probs, key=probs.get) == "flat" and abs(sum(probs.values()) - 1) < 1e-9
    two = assertion_rules(["strong", "soft"], {"strong": "bid-to-cover above 2.50", "soft": "2.50 or below"})
    assert two == {"strong": "Q > 2.5", "soft": "Q < 2.5"}
    # comparisons with another quantity and time spans are not thresholds
    assert assertion_rules(["beat", "miss", "inline"], {"beat": "more than 5% above the analyst consensus estimate",
                                                        "miss": "more than 5% below the analyst consensus estimate",
                                                        "inline": "within 5% of the analyst consensus estimate"}) is None
    assert assertion_rules(["credit_event", "no_event"], {"credit_event": "a credit event in the 12 months after 2023-03-31",
                                                          "no_event": "no credit event over the 12 months"}) is None


def test_planner_rules_that_do_not_read_give_way_to_the_stated_bands():
    _v2()
    from types import SimpleNamespace

    from t4agent.predict import Forecaster

    bands = ["low", "mid", "high"]
    ent = {"entity_id": "X"}
    spec = SimpleNamespace(labels=bands, entity=lambda eid: ent, raw={"target": {"label_assertions": {
        "low": "ratio below 2.3", "mid": "ratio between 2.3 and 2.6", "high": "ratio above 2.6"}}})
    owner = SimpleNamespace(spec=spec, plan={"label_rules": {"low": "Q <= 2.3 or Q is low", "mid": "?", "high": "Q > 2.6"}},
                            anchor={})
    owner._rule_sources = lambda eid: Forecaster._rule_sources(owner, eid)
    assert Forecaster._rules(owner, "X") == {"low": "Q < 2.3", "mid": "2.3 < Q < 2.6", "high": "Q > 2.6"}
    owner.plan = {"label_rules": {"low": "Q < 2.5", "mid": "2.5 < Q < 2.8", "high": "otherwise"}}
    # v21: the task's stated bands come first even when the plan's rules read; the plan fills in only without them
    assert Forecaster._rules(owner, "X") == {"low": "Q < 2.3", "mid": "2.3 < Q < 2.6", "high": "Q > 2.6"}
    owner.spec = SimpleNamespace(labels=bands, entity=lambda eid: ent, raw={"target": {}})
    assert Forecaster._rules(owner, "X") == owner.plan["label_rules"]


# ------------------------------------------------------------------ v13: regression on a vintage series
def test_a_regression_on_the_next_estimate_takes_the_vintage_scale():
    """A percent-revision target had no measured scale: the planner's single deviation (2.5 points) gave payrolls a
    +/-5% interval where the series' own revisions are under 0.1%."""
    _v2()
    import copy
    import json

    from t4agent.corpus import load_corpus, map_entities
    from t4agent.llm import HouseClient
    from t4agent.predict import Evidence, Forecast, Forecaster
    from t4agent.task import load_task

    unit = UNITS / "t4-macrorev-20240930-us6"
    raw = json.loads((unit / "task.json").read_text())
    task = copy.deepcopy(raw)
    task["target"] = {"name": "next_estimate_revision_pct", "type": "regression"}
    path = Path(__import__("tempfile").mkdtemp()) / "task.json"
    path.write_text(json.dumps(task))
    spec = load_task(path)
    corpus = load_corpus(unit / "corpus", spec)
    ev = Evidence(spec, corpus, *map_entities(spec, corpus))
    f = Forecaster(spec, corpus, ev, HouseClient(deadline=0), 0)
    f.plan = {"target_from_level": {"level_quantity": "next published estimate", "base_field": "latest_precutoff_estimate",
                                    "formula": "pct_change"}}
    value, q90, measured = f._revision_in_target("PAYEMS_2024-07_20241004")
    assert abs(value) < 0.2 and 0.0 < q90 < 0.2 and measured
    fc = Forecast(eid="PAYEMS_2024-07_20241004", source="llm", median=0.4, p05=-4.0, p95=4.8)
    f._combine_with_stats(fc)
    assert abs(fc.median - 0.5 * (0.4 + value)) < 1e-9 and fc.stat_q90 == q90 and fc.stat_reliable


# ------------------------------------------------------------------ v13: replacements for unusable reasons
def test_a_held_back_reason_is_replaced_by_a_follow_up():
    import time

    from t4agent.llm import Reply
    from t4agent.reasons import model_reasons, unit_drivers

    spec, corpus, ev, f, forecasts, preds, pid = _bank_reasons_setup()
    first = ('{"reasons": [{"premise_id": "%s", "mechanism": "JPMorgan fees rose.", "answer_implication": "Lifts JPM (5).",'
             ' "entities": [{"id": "JPM", "argues": "up"}]}, {"premise_id": "%s", "mechanism": "Bank of America costs fell.",'
             ' "answer_implication": "Lifts BAC (-3).", "entities": [{"id": "BAC", "argues": "up"}]}]}' % (pid["JPM"], pid["BAC"]))
    second = ('{"reasons": [{"premise_id": "%s", "mechanism": "Bank of America provisions rose.", "answer_implication": '
              '"Lowers BAC (-3).", "entities": [{"id": "BAC", "argues": "down"}]}]}' % pid["BAC"])
    seen = []

    class Stub:
        enabled, remaining, deadline, speed = True, 5, time.time() + 300, 100.0

        def chat(self, msgs, **kw):
            seen.append(msgs)
            return Reply(first if len(seen) == 1 else second, "", "stop", 50, 1.0)

    out = model_reasons(spec, corpus, ev, f, forecasts, preds, Stub(), unit_drivers(spec, None))
    # told why the BAC reason could not be used; v21: a second round follows while fewer than three clean reasons stand
    assert len(seen) == 3 and "BAC argued up" in seen[1][-1]["content"]
    assert [r["answer_implication"] for r in out][:1] == ["Lifts JPM (5)."]
    assert any("BAC" in r["answer_implication"] and "Lowers" in r["answer_implication"] for r in out)


def test_a_review_on_another_scale_is_ignored(monkeypatch):
    """The reviewer answered INDPRO's revision in percent (-0.40) where the level rule expects the level: read as a
    level that is a -100% revision, and the 50/50 blend submitted -50%."""
    from t4agent import config

    monkeypatch.setattr(config, "REVIEW", 1)  # v21: off by default
    import time

    from t4agent.llm import Reply
    from t4agent.predict import Forecast

    f = _forecaster("t4-fomc-curve-20240918", {"target_from_level": {"level_quantity": "10-year yield in percent",
                                                                     "base_field": "start_yield_pct", "formula": "difference"}})
    assert f.level_rule() is not None
    eid = "UST10Y"
    fc = Forecast(eid=eid, source="llm", median=3.0, p05=-30.0, p95=36.0, stat_q90=None)
    good, bad = '{"revisions": [{"entity_id": "UST10Y", "median": 3.78, "p05": 3.5, "p95": 4.1}]}', \
        '{"revisions": [{"entity_id": "UST10Y", "median": 0.05, "p05": -0.3, "p95": 0.4}]}'

    class Stub:
        enabled, remaining, deadline, speed = True, 5, time.time() + 300, 100.0
        reply = bad

        def chat(self, msgs, **kw):
            return Reply(self.reply, "", "stop", 50, 1.0)

    f.client = Stub()
    others = {e: Forecast(eid=e, source="llm", median=0.0, p05=-30.0, p95=30.0) for e in f.spec.entity_ids if e != eid}
    f.review({**others, eid: fc})
    assert fc.median == 3.0  # 0.05 read as a level is a -368 bp move: ignored
    Stub.reply = good
    f.review({**others, eid: fc})
    start = f.spec.entity(eid)["start_yield_pct"]
    assert abs(fc.median - 0.5 * (3.0 + (3.78 - start) * 100)) < 1e-6  # a second opinion on the scale is blended


def test_a_long_roster_fits_the_reasoning_graders_answer_cap():
    """The grader judges no reason when the per-entity answer it reads exceeds 3,000 bytes: a 26-row roster with
    nine-digit levels would have scored 0 for reasoning however good its reasons."""
    _v2()
    from types import SimpleNamespace

    from t4agent.answer import _judged_bytes, fit_judge_cap

    ids = [f"PAYEMS_{i:02d}_2024-08" for i in range(26)]
    spec = SimpleNamespace(entity_ids=ids, target_type="regression", interval_level=0.9)
    rows = {e: {"point_forecast": 158688.688 + i, "lo": 158367.954 + i, "hi": 158884.556 + i, "claims": []}
            for i, e in enumerate(ids)}
    assert _judged_bytes(spec, rows) > 3000
    fit = fit_judge_cap(spec, rows)
    assert _judged_bytes(spec, fit) <= 3000
    for e in ids:  # the interval only widens and keeps the point inside it
        assert fit[e]["lo"] <= rows[e]["lo"] and fit[e]["hi"] >= rows[e]["hi"]
        assert fit[e]["lo"] <= fit[e]["point_forecast"] <= fit[e]["hi"]
    small = {e: rows[e] for e in ids[:5]}
    assert fit_judge_cap(SimpleNamespace(entity_ids=ids[:5], target_type="regression", interval_level=0.9), small) is small



# ------------------------------------------------------------------ v14: fixes from a review of v13
def test_the_answer_cap_never_changes_a_forecast():
    """v13 merged 26 close ranking points into one value and widened a one-unit interval to 1,000 -- and the second
    roster still missed the cap. A forecast is never traded for a chance at the reasoning bonus."""
    _v2()
    from types import SimpleNamespace

    from t4agent.answer import _judged_bytes, fit_judge_cap

    ids = [f"FUTURES_MKT_{i:02d}" for i in range(26)]
    rank = SimpleNamespace(entity_ids=ids, target_type="ranking", interval_level=0.9)
    rows = {e: {"point_forecast": 158688.688 + 0.013 * i, "lo": 158600.123 + 0.013 * i, "hi": 158777.456 + 0.013 * i}
            for i, e in enumerate(ids)}
    assert _judged_bytes(rank, rows) > 3000
    fit = fit_judge_cap(rank, rows)  # v21.8: rounded only because every point keeps its place
    assert _judged_bytes(rank, fit) <= 3000
    assert sorted(ids, key=lambda e: rows[e]["point_forecast"]) == sorted(ids, key=lambda e: fit[e]["point_forecast"])
    assert len({fit[e]["point_forecast"] for e in ids}) == len(ids)
    ids = [f"SERIES_ID_LONG_NAME_{i:02d}" for i in range(30)]
    reg = SimpleNamespace(entity_ids=ids, target_type="regression", interval_level=0.9)
    narrow = {e: {"point_forecast": 158688.5 + i, "lo": 158688.0 + i, "hi": 158689.0 + i} for i, e in enumerate(ids)}
    assert fit_judge_cap(reg, narrow) is narrow  # rounding to fit would widen a 1-unit interval: left as it is
    ids = [f"PAYEMS_{i:02d}_2024-08" for i in range(26)]
    reg = SimpleNamespace(entity_ids=ids, target_type="regression", interval_level=0.9)
    rows = {e: {"point_forecast": 158688.68812 + i, "lo": 158367.95431 + i, "hi": 158884.55617 + i} for i, e in enumerate(ids)}
    fit = fit_judge_cap(reg, rows)
    assert _judged_bytes(reg, fit) <= 3000
    for e in ids:  # within 1% wider and 0.5% of the width away
        w = rows[e]["hi"] - rows[e]["lo"]
        assert (fit[e]["hi"] - fit[e]["lo"]) - w <= 0.01 * w + 1e-9
        assert abs(fit[e]["point_forecast"] - rows[e]["point_forecast"]) <= 0.005 * w


def test_a_threshold_relative_to_another_quantity_is_not_read():
    """v13 read 'more than 5 percent above the consensus' as Q > 5 (the unit word was given back to the number): with
    consensus 3, a forecast of 4 in [3.2, 4.8] -- a beat -- was settled 'inline'."""
    _v2()
    from types import SimpleNamespace

    from t4agent.labels import _assertion_cond
    from t4agent.predict import Forecast, Forecaster

    for text in ("diluted EPS more than 5 percent above the consensus estimate", "EPS 5 pct above consensus",
                 "more than 5 per cent below the consensus", "a change of more than 5 percent from the prior month",
                 "at least 3 percent higher than last year", "up more than 2 percentage points versus the prior quarter",
                 "exceeds 2.5 times the prior auction", "more than 5 percent above 3.0"):
        assert _assertion_cond(text) is None, text
    assert _assertion_cond("return over 5 percent") == "Q > 5.0"  # an absolute threshold still reads
    assert _assertion_cond("a yield change of more than 25 bps") == "Q > 25.0"

    labs = ["beat", "miss", "inline"]
    ent = {"entity_id": "X", "consensus_eps": 3.0}
    spec = SimpleNamespace(labels=labs, target_type="classification", target_name="eps_outcome", prompt="",
                           entities=[ent], entity=lambda e: ent, raw={"target": {"label_assertions": {
                               "beat": "reported diluted EPS more than 5 percent above the consensus estimate",
                               "miss": "reported diluted EPS more than 5 percent below the consensus estimate",
                               "inline": "reported diluted EPS within 5 percent of the consensus estimate"}}})
    owner = SimpleNamespace(spec=spec, plan={"interval_is_probability": False}, _halves={}, prior={}, anchor={},
                            corpus=None, ev=None)
    owner._rule_sources = lambda eid: Forecaster._rule_sources(owner, eid)
    owner._submitted_range = lambda fc, e: Forecaster._submitted_range(owner, fc, e)
    owner._blend_prior = lambda fc: None
    owner._blend_eps_direction = lambda fc: None
    assert Forecaster._rule_sources(owner, "X") == []
    fc = Forecast("X", model_probs={"beat": 0.6, "miss": 0.05, "inline": 0.35}, median=4.0, p05=3.2, p95=4.8)
    Forecaster._settle_label(owner, fc, ent)
    assert fc.label == "beat"


def test_a_negative_series_revised_up_is_expected_higher():
    """Relative revisions are measured on |estimate|; v13 applied them to the signed value: P(up) 0.96, expected -101."""
    _v2()
    import datetime as dt

    from t4agent.revisions import _prior

    dates = [dt.date(2024, m, 5) for m in (3, 4, 5, 6, 7, 8)]
    rows = {"2024-01": [-100.0, -99.0, -98.0, -98.0, -98.0, -98.0], "2024-02": [None, -102.0, -101.0, -100.0, -100.0, -100.0],
            "2024-03": [None, None, -105.0, -104.0, -103.0, -103.0], "2024-04": [None, None, None, -101.0, -100.0, -99.5],
            "2024-05": [None, None, None, None, -110.0, -109.0], "2024-06": [None, None, None, None, None, -100.0]}
    pr = _prior("ALFRED_TB_X", dates, rows, "2024-06", dates[-1], -100.0, dt.date(2024, 9, 5))
    # v15: the expected next estimate is the latest itself (the replay found the mean shift harmful), so it can no longer
    # move against the direction rate; the rate still leans the right way for a series revised up every time
    assert pr.p_up > 0.5 and pr.value == pr.latest and pr.mean_rel > 0
    pos = _prior("ALFRED_TB_X", dates, {k: [None if v is None else -v for v in vals] for k, vals in rows.items()},
                 "2024-06", dates[-1], 100.0, dt.date(2024, 9, 5))
    assert pos.p_up < 0.5 and pos.value == pos.latest  # the mirror image: revised down


def test_rules_that_fail_on_the_distribution_give_way_to_the_next_source():
    """A band written backwards ('2.6 < Q < 2.3') passed the syntax check, failed in the probabilities and kept the task's
    own (correct) bands from being tried: a range wholly inside 'mid' was labelled 'low' from the model's odds."""
    _v2()
    from types import SimpleNamespace

    from t4agent.labels import rules_parse
    from t4agent.predict import Forecast, Forecaster

    labs = ["low", "mid", "high"]
    ent = {"entity_id": "X"}
    assert not rules_parse({"low": "Q < 2.3", "mid": "2.6 < Q < 2.3", "high": "Q > 2.6"}, labs, ent)
    spec = SimpleNamespace(labels=labs, target_type="classification", target_name="btc_band", prompt="", entities=[ent],
                           entity=lambda e: ent, raw={"target": {"label_assertions": {
                               "low": "ratio below 2.3", "mid": "ratio between 2.3 and 2.6", "high": "ratio above 2.6"}}})
    for planned in ({"low": "Q < 2.3", "mid": "2.6 < Q < 2.3", "high": "Q > 2.6"},  # backwards band
                    {"low": "Q < 2.6", "mid": "Q > 2.3", "high": "otherwise"}):  # parses, but the regions overlap
        owner = SimpleNamespace(spec=spec, plan={"label_rules": planned, "interval_is_probability": False}, _halves={},
                                prior={}, anchor={}, corpus=None, ev=None)
        owner._rule_sources = lambda eid, o=owner: Forecaster._rule_sources(o, eid)
        owner._submitted_range = lambda fc, e, o=owner: Forecaster._submitted_range(o, fc, e)
        owner._blend_prior = lambda fc: None
        owner._blend_eps_direction = lambda fc: None
        fc = Forecast("X", model_probs={"low": 0.5, "mid": 0.2, "high": 0.3}, median=2.45, p05=2.35, p95=2.55)
        Forecaster._settle_label(owner, fc, ent)
        assert fc.label == "mid", planned


# ------------------------------------------------------------------ v15: interval scales calibrated on pre-2022 data
def test_the_random_walk_multiplier_is_the_calibrated_table():
    _v2()
    import math

    from t4agent.stats import _RW_H, _RW_K, _RW_N, _inflation, rw_k

    for i, n in enumerate(_RW_N):  # the grid nodes are the measured values
        for j, h in enumerate(_RW_H):
            assert abs(rw_k(n, h) - _RW_K[i][j]) < 1e-9
    assert rw_k(10, 34) > rw_k(20, 34) > rw_k(60, 34) > 1.645  # a short window is less sure of its volatility
    assert rw_k(10, 1) < rw_k(10, 40) < rw_k(10, 120)  # changes over many steps have fatter tails
    assert rw_k(2, 1) == rw_k(5, 1) and rw_k(10, 500) == rw_k(10, 120)  # held at the edges of the grid
    # v16: judged by the interval score, counting overlapping errors as n / h independent ones did not help: n again
    assert _inflation(78, 20) == 1 + 0.75 / math.sqrt(78) and _inflation(6, 1) == 1 + 0.75 / math.sqrt(6)


def test_the_step_scale_uses_the_calibrated_multiplier():
    _v2()
    import math

    from t4agent.stats import Series, forecast, rw_k, series_half_width

    vals = [3.60, 3.55, 3.62, 3.58, 3.50, 3.47, 3.52, 3.49, 3.45, 3.44, 3.47]  # 11 daily closes, percent
    s = Series("snap", "DGS10", vals, [f"2024-09-{5 + i:02d}" for i in range(11)], 1.0, True)
    d = [b - a for a, b in zip(vals, vals[1:])]
    rms = math.sqrt(sum(x * x for x in d) / len(d))
    steps = 48 * 5 / 7
    half = series_half_width(s, "yield_change_bps", 48, {"unit": "bps_change"}, policy=_eligible_policy())
    assert abs(half - rw_k(10, steps) * rms * math.sqrt(steps) * 100) < 1e-9  # v14: 1.645 x
    f = forecast(s, "yield_change_bps", 48, {"unit": "bps_change"})
    h = round(steps)
    assert f.method == "zero (random walk)" and abs(f.q90 - rw_k(10, h) * rms * math.sqrt(h) * 100) < 1e-9  # v14: 2.06 x


def test_the_calibration_table_reproduces_from_the_public_data():
    """Provenance: the table is the 90% quantile measured by tools/calibrate_scales.py on ../calib_data (skipped when
    the data are not next to the agent)."""
    _v2()
    data = AGENT.parent / "calib_data"
    if not (data / "DGS10.csv").exists():
        pytest.skip("calibration data not present")
    sys.path.insert(0, str(AGENT / "tools"))
    import calibrate_scales as C

    from t4agent.stats import rw_k

    series = [[v for _, v in C.read(data / f"{s}.csv")] for names in C.DAILY.values() for s in names]
    for n, h in ((10, 40), (20, 5)):  # v16: the interval-score optimum (scale-weighted quantile), not 90% coverage
        assert abs(C.k_is(series, n, h) - rw_k(n, h)) < 0.006



def test_the_revision_base_rate_is_calibrated():
    """v15: replayed on real-time vintages of fourteen U.S. series published 2003-2021, a past-revision direction rate
    taken at face value scored worse than a coin flip, the mean shift made the expected value worse, and the envelope
    covered 84% of next estimates. Kept: 15% of the rate's departure from 0.5, the latest estimate, a wider envelope."""
    _v2()
    import datetime as dt

    from t4agent import revisions as R

    dates = [dt.date(2024, m, 5) for m in (3, 4, 5, 6, 7, 8)]
    rows = {"2024-01": [100.0, 101.0, 102.0, 102.0, 102.0, 102.0], "2024-02": [None, 98.0, 99.0, 100.0, 100.0, 100.0],
            "2024-03": [None, None, 97.0, 98.0, 99.0, 99.0], "2024-04": [None, None, None, 99.0, 100.0, 100.5],
            "2024-05": [None, None, None, None, 90.0, 91.0], "2024-06": [None, None, None, None, None, 100.0]}
    one = R._prior("X_SERIES_V", dates, rows, "2024-06", dates[-1], 100.0, dt.date(2024, 9, 5))
    two = R._prior("X_SERIES_V", dates, rows, "2024-06", dates[-1], 100.0, dt.date(2024, 10, 5))
    assert one.up == one.n and abs(one.p_up - 0.5) <= 0.5 * R.DIRECTION_WEIGHT + 1e-9  # every revision up: still near 0.5
    assert one.value == one.latest == 100.0
    assert one.steps == 1 and two.steps == 2 and two.q90 > one.q90 > 0


def test_the_revision_calibration_reproduces_from_the_real_time_data():
    """Provenance: the replay of tools/calibrate_revisions.py on ../calib_data covers 90% (skipped without the data)."""
    _v2()
    import contextlib
    import io
    import statistics as st

    data = AGENT.parent / "calib_data"
    if not (data / "alfred_rt_PAYEMS.json").exists():
        pytest.skip("calibration data not present")
    sys.path.insert(0, str(AGENT / "tools"))
    import calibrate_revisions as C

    from qfbench2_common.scoring.faithfulness import mean_interval_score

    from t4agent import revisions as R

    with contextlib.redirect_stdout(io.StringIO()):
        allx = C.main()
    nz = [x for x in allx if x["actual"] != x["latest"]]
    assert len(allx) > 12000
    assert st.mean((x["p_up"] - (x["actual"] > x["latest"])) ** 2 for x in nz) < 0.25  # better than a coin flip
    f = C.is_factors(allx)  # v16: the envelope factor is the interval-score optimum
    assert abs(f[1] - R.SCALE_ONE_RELEASE) < 0.015 and abs(f[2] - R.SCALE_MORE_RELEASES) < 0.015

    def loss(factor):  # mean official interval score, each series normalised by its loss at the current factor
        out = []
        for s in {x["series"] for x in allx}:
            xs = [x for x in allx if x["series"] == s and x["q90"] > 0]
            def one(k):
                return mean_interval_score([x["latest"] - k * x["q90"] / R.SCALE_ONE_RELEASE for x in xs],
                                           [x["latest"] + k * x["q90"] / R.SCALE_ONE_RELEASE for x in xs],
                                           [x["actual"] for x in xs], 0.9)
            out.append(one(factor) / one(R.SCALE_ONE_RELEASE))
        return st.mean(out)

    assert loss(1.22) > 1.0 and loss(1.0) > 1.0 and loss(0.6) > 1.0  # v15's and v14's factors, and a narrower one



# ------------------------------------------------------------------ v17: statistical baselines chosen on pre-2022 replays
def test_a_rate_series_reverts_by_its_own_persistence():
    """A stationary rate (m/m change, return) is pulled towards its window mean by 1 - its lag-1 autocorrelation; a
    level series keeps v16's candidates (no AR(1): on random walks it lost 0.002)."""
    _v2()
    from t4agent.stats import Series, _ar1_phi, forecast

    noisy = [0.3, -0.1, 0.4, -0.2, 0.35, -0.15, 0.3, -0.05, 0.6]  # alternating: strong reversion
    persistent = [0.1, 0.15, 0.2, 0.26, 0.3, 0.34, 0.4, 0.45, 0.5]  # trending: little
    assert _ar1_phi(noisy) > 0.9 and _ar1_phi(persistent) < 0.5 and _ar1_phi([1.0, 2.0, 3.0]) == 0.0
    s = Series("t", "m/m %", noisy, [str(i) for i in range(9)], 30.0, True)
    f = forecast(s, "component_mom_first_print", 30, {"unit": "mom_pct_change_sa"})
    assert f.method == "ar1" and abs(f.value - (0.6 + _ar1_phi(noisy) * (sum(noisy) / 9 - 0.6))) < 1e-9
    lv = Series("t", "level", [100, 101, 99, 102, 103, 101, 104, 105, 103, 106, 107], [str(i) for i in range(11)], 30.0, True)
    assert forecast(lv, "index_level", 30, {"unit": "index"}).method != "ar1"


def test_a_change_target_leaves_zero_change_only_for_a_clear_and_steady_edge():
    """Overlapping h-step backtest errors made v16's rule leave zero change for 63% of daily-market windows (and lose to
    it); a method must now be 25% better and better in both halves."""
    _v2()
    from t4agent.stats import _change_switch_ok

    zero = [1.0] * 10
    assert not _change_switch_ok({"zero": zero, "drift": [0.85] * 10}, "drift", "zero")  # 15% better: not enough
    assert _change_switch_ok({"zero": zero, "drift": [0.5] * 10}, "drift", "zero")
    assert not _change_switch_ok({"zero": zero, "drift": [0.1] * 5 + [1.2] * 5}, "drift", "zero")  # only early on
    assert not _change_switch_ok({"zero": [1.0] * 6, "drift": [0.1] * 6}, "drift", "zero")  # too few origins


def test_positioning_changes_revert_to_their_window_level():
    """A ratio change ('X % of Y', e.g. net position as a share of open interest) forecasts a share of the gap between
    the ratio and its window mean, closing it: on CFTC 2011-2021 this ranked the ten markets at 0.568 at five weeks where
    zero change carries no ordering. v19: the share grows with the horizon (1 - exp(-0.003 days))."""
    _v2()
    from t4agent.stats import Series, _ratio_change_forecast, revert_share

    oi = [1000.0] * 26
    long_hi = Series("x", "net_%OI", [100.0] * 20 + [300.0] * 6, [str(i) for i in range(26)], 7.0, True, oi, 100.0)
    long_lo = Series("x", "net_%OI", [100.0] * 20 + [0.0] * 6, [str(i) for i in range(26)], 7.0, True, oi, 100.0)
    up, down = _ratio_change_forecast(long_lo, 35), _ratio_change_forecast(long_hi, 35)
    assert down.value < 0 < up.value and down.method == up.method == "level_revert"
    ratios = [x / 10 for x in long_hi.values]
    assert abs(down.value - (-revert_share(35) * (ratios[-1] - sum(ratios) / 26))) < 1e-9
    assert abs(revert_share(35) - 0.0995) < 1e-3 and abs(revert_share(91) - 0.239) < 1e-3  # 5 and 13 weeks
    longer = [abs(_ratio_change_forecast(long_hi, d).value) for d in (7, 35, 91)]
    assert longer[0] < longer[1] < longer[2]  # v17/v18: one fixed share of 0.15 at every horizon
    # the Development unit's horizon counts 32 days from its cutoff; the change spans five weekly reports (35 days)
    assert _ratio_change_forecast(long_hi, 32).value == down.value


def test_a_clear_positioning_trend_is_followed():
    """v19 (review of v17): reversion was forced on every ratio change, so a steady +5 points per five weeks was
    forecast at -1.9. Drift now replaces it when it halves the backtest error and wins both halves -- the CFTC replay
    makes the bar high (at the 25% change rule switching ranked the markets worse by 0.006 at five weeks)."""
    _v2()
    from t4agent.stats import Series, _ratio_change_forecast

    oi = [1000.0] * 26
    trend = Series("x", "net_%OI", [10.0 * i for i in range(26)], [str(i) for i in range(26)], 7.0, True, oi, 100.0)
    for days, want in ((7, 1.0), (35, 5.0)):
        fc = _ratio_change_forecast(trend, days)
        assert fc.method == "drift" and abs(fc.value - want) < 1e-9
    # thirteen weeks in a 26-week window leave no backtest origins: the default stands (no evidence either way)
    assert _ratio_change_forecast(trend, 91).method.startswith("level reversion")
    rng = __import__("random").Random(7)
    x = [0.0]
    for _ in range(25):
        x.append(x[-1] + rng.gauss(0, 10.0))  # a driftless walk: reversion stands
    assert _ratio_change_forecast(Series("x", "net_%OI", x, [str(i) for i in range(26)], 7.0, True, oi, 100.0), 35).method == "level_revert"


def test_first_prints_are_replayed_in_real_time():
    """v19 (review of v17): the month-over-month replays behind the AR(1) default read the revised 2021 vintage. The
    replay now rebuilds each release from ALFRED: the window as known the day before, the outcome the first print."""
    if not (ROOT / "calib_data" / "alfred_rt_CPIAUCSL.json").exists():
        pytest.skip("calibration data not present")
    sys.path.insert(0, str(AGENT / "tools"))
    import calibrate_methods as M

    w, first = M.first_prints("CPIAUCSL")["2021-10"]
    assert len(w) == 9 and abs(first - 0.9) < 0.05  # BLS, 2021-11-10: CPI-U +0.9% in October, seasonally adjusted
    assert abs(w[-1] - 0.4) < 0.05  # September as published on 2021-10-13, before any revision: +0.4%


# ------------------------------------------------------------------ v20: CPI first prints nowcast from pump prices
def test_cpi_gasoline_energy_and_headline_are_nowcast_from_weekly_pump_prices():
    """v20: the corpus's weekly gasoline price reaches into the target month, so the statistical baseline of the three
    gasoline-driven first prints is BETA x its change + the calendar month's seasonal gap (fitted on first prints
    published by 2021). v19 used the series' own AR(1): gasoline -0.99, i.e. the October seasonal rise was missed."""
    _v2()
    from t4agent.corpus import load_corpus, map_entities
    from t4agent.llm import HouseClient
    from t4agent.nowcast import NOWCAST
    from t4agent.predict import Evidence, Forecaster
    from t4agent.task import load_task

    unit = UNITS / "t4-cpicomp-202410-us11"
    spec = load_task(unit / "task.json")
    corpus = load_corpus(unit / "corpus", spec)
    f = Forecaster(spec, corpus, Evidence(spec, corpus, *map_entities(spec, corpus)), HouseClient(deadline=0), 0)
    now = {e: sf for e, sf in f.stat.items() if sf is not None and sf.method == "pump-price nowcast"}
    assert set(now) == {"CPI_GASOLINE", "CPI_ENERGY", "CPI_ALLITEMS"}
    g = now["CPI_GASOLINE"]
    assert g.weeks == 4 and abs(g.driver - (3.137 / 3.214 - 1) * 100) < 0.02  # four October readings vs September
    for e, code in (("CPI_GASOLINE", "CUSR0000SETB01"), ("CPI_ENERGY", "CPIENGSL"), ("CPI_ALLITEMS", "CPIAUCSL")):
        c = NOWCAST[code]
        assert abs(now[e].value - (c["beta"] * g.driver + c["gap"][10])) < 1e-9 and now[e].q90 == c["q90"][3]
    assert abs(g.value - 0.28) < 0.01 and "October seasonal gap +2.67" in g.describe()
    assert f.stat["CPI_CORE"].method == "ar1"  # no driver: the univariate baseline stands


def test_no_pump_price_nowcast_without_the_target_month_or_for_other_targets():
    _v2()
    from t4agent.corpus import load_corpus, map_entities
    from t4agent.nowcast import pump_nowcast
    from t4agent.task import load_task

    unit = UNITS / "t4-cpicomp-202410-us11"
    spec = load_task(unit / "task.json")
    corpus = load_corpus(unit / "corpus", spec)
    docs = sorted(corpus.docs)
    ent = dict(spec.entity("CPI_GASOLINE"))
    assert pump_nowcast(corpus, docs, ent, spec.target_name, None, policy=_eligible_policy()) is not None
    assert pump_nowcast(corpus, docs, {**ent, "ref_month": "2024-11"}, spec.target_name, None, policy=_eligible_policy()) is None  # no week yet
    assert pump_nowcast(corpus, docs, {**ent, "unit": "mom_pct_change_nsa"}, "cpi_mom_nsa", None, policy=_eligible_policy()) is None
    assert pump_nowcast(corpus, docs, {**ent, "series_fred": "CPILFESL"}, spec.target_name, None, policy=_eligible_policy()) is None
    assert pump_nowcast(corpus, [d for d in docs if "GASREG" not in d], ent, spec.target_name, None, policy=_eligible_policy()) is None


def test_the_pump_price_nowcast_reproduces_from_the_public_data():
    """Provenance: tools/calibrate_nowcast.py on ../calib_data (skipped without the data)."""
    _v2()
    if not (AGENT.parent / "calib_data" / "GASREGW_1990_2021.json").exists():
        pytest.skip("calibration data not present")
    sys.path.insert(0, str(AGENT / "tools"))
    import calibrate_nowcast as CN

    from t4agent.nowcast import NOWCAST

    got = CN.main()
    for code, c in NOWCAST.items():
        g = got[code]
        assert abs(g["beta"] - c["beta"]) < 1e-3 and abs(g["mae"] - c["mae"]) < 1e-3
        assert all(abs(g["gap"][f"{m:02d}"] - c["gap"][m]) < 0.006 for m in range(1, 13))
        assert all(abs(g["q90"][k] - c["q90"][k]) < 0.006 for k in (2, 3))


def test_the_nowcast_is_disclosed_as_a_local_model_with_its_coefficients_digest():
    """ARTIFACT-POLICY.md: a fitted linear model is permitted with disclosure, and every learned local model gets a
    models[] row with access "local". The row's revision must name the coefficients actually shipped."""
    _v2()
    import t4agent
    from qfbench2_common.contracts.descriptor import ModelDisclosure
    from t4agent.nowcast import coefficients_digest

    (row,) = t4agent.LOCAL_MODELS
    ModelDisclosure.from_mapping(row, path="models[1]")  # the official descriptor schema accepts the row
    assert row["access"] == "local" and row["version"] == t4agent.__version__
    assert row["revision"] == coefficients_digest() and row["training_cutoff"] == "2021-12"


def test_a_reason_is_not_held_back_by_an_answer_that_sits_on_its_naive_value():
    """v20: CPI headline at 0.17 against its latest 0.18 (90% half-width 0.25) argues no direction, so the reasons on
    the gasoline rebound that also lift headline are no longer held back; a clear departure still conflicts."""
    _v2()
    from t4agent.reasons import argued_against
    from t4agent.task import load_task

    spec = load_task(UNITS / "t4-cpicomp-202410-us11" / "task.json")
    near = {"entity_id": "CPI_ALLITEMS", "point_forecast": 0.17, "interval": {"level": 0.9, "lo": -0.08, "hi": 0.42}}
    assert not argued_against(spec, near, "higher") and not argued_against(spec, near, "lower")
    far = dict(near, point_forecast=0.10)  # 0.08 below the latest, more than a tenth of the half-width
    assert argued_against(spec, far, "higher") and not argued_against(spec, far, "lower")
    gas = {"entity_id": "CPI_GASOLINE", "point_forecast": 0.35, "interval": {"level": 0.9, "lo": -4.28, "hi": 4.98}}
    assert argued_against(spec, gas, "lower") and not argued_against(spec, gas, "higher")  # latest -4.10
    # the reasons prompt now shows where each answer sits against the latest value, the reference of 'argues'
    from t4agent.reasons import answer_line

    assert answer_line(spec, near).endswith("; about the same as the latest value 0.18)")
    assert answer_line(spec, far).endswith("; lower than the latest value 0.18)")
    assert answer_line(spec, gas).endswith("; higher than the latest value -4.1)")


# ------------------------------------------------------------------ v21: harness reallocation
def test_reply_json_is_recovered_without_a_repair_call():
    """v21: v20's sweep spent 16 of 194 requests on JSON repair calls. Trailing commas, NaN and Python literals are fixed in
    code, and a reply whose JSON sits only in its reasoning is read from there; the repair call is the last resort."""
    _v2()
    import time

    from t4agent.corpus import load_corpus, map_entities
    from t4agent.llm import HouseClient, Reply, parse_reply
    from t4agent.predict import Evidence, Forecaster
    from t4agent.task import load_task

    assert parse_reply(Reply('{"forecasts": [{"entity_id": "A", "median": 1.2,}],}', "", "stop", 1, 1.0), "forecasts")
    rep = Reply("Prose only.", 'draft {"forecasts": [{"entity_id": "A", "median": 1.0}]} final '
                '{"forecasts": [{"entity_id": "A", "median": 1.3}]}', "stop", 1, 1.0)
    assert parse_reply(rep, "forecasts")["forecasts"][0]["median"] == 1.3  # the last draft
    unit = UNITS / "t4-cpicomp-202410-us11"
    spec = load_task(unit / "task.json")
    corpus = load_corpus(unit / "corpus", spec)
    f = Forecaster(spec, corpus, Evidence(spec, corpus, *map_entities(spec, corpus)), HouseClient(deadline=0), 0)
    tags = []

    class Stub:
        enabled, remaining, deadline, speed, cool_until = True, 10, time.time() + 300, 100.0, 0.0

        def chat(self, msgs, **kw):
            tags.append(kw.get("tag"))
            return Reply("I think the answer is below.", '{"forecasts": [{"entity_id": "CPI_FOOD", "median": 0.2, '
                         '"p05": 0.0, "p95": 0.4}]}', "stop", 50, 1.0)

    f.client = Stub()
    assert f._one_call(["CPI_FOOD"], 0)["CPI_FOOD"]["median"] == 0.2 and tags == ["analyst gCPI_FOOD s0"]  # no repair


def test_the_task_label_conditions_outrank_the_planners_rules():
    """v21 (P1 of the v19 audit): the example unit's table gives beat above 1.50 x 1.05 = 1.575; a planner that misread it
    as Q > 5 turned a forecast of 1.80 (90% 1.65-1.95) into 'inline'. The task's own conditions now come first."""
    _v2()
    from t4agent.corpus import load_corpus, map_entities
    from t4agent.llm import HouseClient
    from t4agent.predict import Evidence, Forecast, Forecaster
    from t4agent.task import load_task

    unit = UNITS / "t4-EXAMPLE-eps-beat"
    spec = load_task(unit / "task.json")
    corpus = load_corpus(unit / "corpus", spec)
    f = Forecaster(spec, corpus, Evidence(spec, corpus, *map_entities(spec, corpus)), HouseClient(deadline=0), 0)
    f.plan = {"label_rules": {"beat": "Q > 5", "miss": "Q < -5", "inline": "otherwise"}}
    fc = Forecast(eid="AAPL", source="llm", model_probs={"beat": 0.95, "miss": 0.01, "inline": 0.04},
                  median=1.80, p05=1.65, p95=1.95)
    f._settle_label(fc, spec.entity("AAPL"))
    assert fc.label == "beat" and fc.probs["beat"] > 0.8


def test_anchored_regression_units_take_few_analyst_samples_and_keep_the_rest(monkeypatch):
    """v21: where every entity has a code baseline, the analysts moved v20's answers by 0.01-0.25 standard deviations;
    such units take ANCHORED_SAMPLES per group and leave their other requests to the reasons."""
    _v2()
    import time

    from t4agent import config
    from t4agent.corpus import load_corpus, map_entities
    from t4agent.llm import HouseClient, Reply
    from t4agent.predict import Evidence, Forecaster
    from t4agent.task import load_task

    def forecaster(name):
        unit = UNITS / name
        spec = load_task(unit / "task.json")
        corpus = load_corpus(unit / "corpus", spec)
        return Forecaster(spec, corpus, Evidence(spec, corpus, *map_entities(spec, corpus)), HouseClient(deadline=0), 0)

    cpi = forecaster("t4-cpicomp-202410-us11")
    assert cpi.anchored() and not forecaster("t4-credit-event-2023").anchored()
    calls = []

    class Stub:
        enabled, remaining, deadline, speed, cool_until = True, 22, time.time() + 300, 100.0, 0.0

        def chat(self, msgs, **kw):
            calls.append(kw.get("tag"))
            self.remaining -= 1
            rows = [{"entity_id": e, "median": 0.2, "p05": 0.0, "p95": 0.4} for e in cpi.spec.entity_ids]
            return Reply(__import__("json").dumps({"forecasts": rows}), "", "stop", 50, 1.0)

    monkeypatch.setattr(config, "CONCURRENCY", 1)
    cpi.client = Stub()
    cpi.run_analysts()
    groups = len(cpi.groups())
    assert 0 < len(calls) <= groups * config.ANCHORED_SAMPLES and cpi.client.remaining >= 22 - groups * 3


def test_extra_votes_go_to_the_labels_the_samples_left_undecided(monkeypatch):
    """v21: a label whose samples split (top label under 75% of the votes) gets focused single-entity calls."""
    _v2()
    import time

    from t4agent import config
    from t4agent.corpus import load_corpus, map_entities
    from t4agent.llm import HouseClient, Reply
    from t4agent.predict import Evidence, Forecaster
    from t4agent.task import load_task

    unit = UNITS / "t4-credit-event-2023"
    spec = load_task(unit / "task.json")
    corpus = load_corpus(unit / "corpus", spec)
    f = Forecaster(spec, corpus, Evidence(spec, corpus, *map_entities(spec, corpus)), HouseClient(deadline=0), 0)
    ev, no = ({"probs": {"credit_event": 0.7, "no_event": 0.3}}, {"probs": {"credit_event": 0.2, "no_event": 0.8}})
    samples = {"RAD": [ev, no, ev, no], "BBY": [no, no, no, no]}
    assert f.undecided(samples) == ["RAD"]
    asked = []

    class Stub:
        enabled, remaining, deadline, speed, cool_until = True, 12, time.time() + 300, 100.0, 0.0

        def chat(self, msgs, **kw):
            asked.append(kw.get("tag"))
            self.remaining -= 1
            return Reply('{"forecasts": [{"entity_id": "RAD", "probs": {"credit_event": 0.8, "no_event": 0.2}}]}',
                         "", "stop", 50, 1.0)

    monkeypatch.setattr(config, "CONCURRENCY", 1)
    f.client = Stub()
    out = f.extra_votes(samples)
    assert asked and all(t.startswith("analyst gRAD") for t in asked) and len(asked) <= config.EXTRA_VOTES
    assert len(out["RAD"]) == 4 + len(asked) and len(out["BBY"]) == 4


def test_a_reason_with_figures_its_citations_lack_is_replaced_by_a_clean_rewrite():
    """v21: v20's post-earnings reasons put invented reactions ('roughly -2% to -3%') in their mechanisms; a reason with
    two or more figures that neither its citations, the task nor the answers contain stays only until a follow-up round
    writes a clean one, which may reuse its passage."""
    _v2()
    import time

    from t4agent.llm import Reply
    from t4agent.reasons import model_reasons, ungrounded_figures, unit_drivers

    assert ungrounded_figures("down by roughly -2% to -3.5% in 3 days of 2024", ["EPS rose to 1.82"], []) == ["-2%", "-3.5%"]
    assert ungrounded_figures("EPS rose to 1.8 (12%)", ["diluted EPS $1.82, up 12.3%"], []) == []
    spec, corpus, ev, f, forecasts, preds, pid = _bank_reasons_setup()
    first = ('{"reasons": [{"premise_id": "%s", "mechanism": "JPMorgan fees rose, worth about 7.7%% to 9.1%% more EPS.", '
             '"answer_implication": "Lifts JPM (5).", "entities": [{"id": "JPM", "argues": "up"}]}]}' % pid["JPM"])
    second = ('{"reasons": [{"premise_id": "%s", "mechanism": "JPMorgan fees rose.", "answer_implication": "Lifts JPM (5).",'
              ' "entities": [{"id": "JPM", "argues": "up"}]}]}' % pid["JPM"])
    seen = []

    class Stub:
        enabled, remaining, deadline, speed = True, 5, time.time() + 300, 100.0

        def chat(self, msgs, **kw):
            seen.append(msgs)
            return Reply(first if len(seen) == 1 else second, "", "stop", 50, 1.0)

    out = model_reasons(spec, corpus, ev, f, forecasts, preds, Stub(), unit_drivers(spec, None))
    assert "7.7%, 9.1%" in seen[1][-1]["content"]  # the follow-up is told which figures lack a source (two or more)
    assert [r["mechanism"] for r in out] == ["JPMorgan fees rose."] and all("_ungrounded" not in r for r in out)


def test_the_reasons_keep_two_requests_that_json_repairs_cannot_take():
    """v21: with the review off, repairs on a route that needs many of them left three units without a request for the
    reasons' retry; the end of the unit now keeps the re-check plus two for the reasons, and a repair never takes them."""
    _v2()
    import time

    from t4agent.corpus import load_corpus, map_entities
    from t4agent.llm import HouseClient, Reply
    from t4agent.predict import Evidence, Forecaster
    from t4agent.task import load_task

    unit = UNITS / "t4-cpicomp-202410-us11"
    spec = load_task(unit / "task.json")
    corpus = load_corpus(unit / "corpus", spec)
    f = Forecaster(spec, corpus, Evidence(spec, corpus, *map_entities(spec, corpus)), HouseClient(deadline=0), 0)
    assert f.end_reserve() == 3
    tags = []

    class Stub:
        enabled, remaining, deadline, speed, cool_until = True, 3, time.time() + 300, 100.0, 0.0

        def chat(self, msgs, **kw):
            tags.append(kw.get("tag"))
            return Reply("no JSON here", "", "stop", 50, 1.0)

    f.client = Stub()
    with pytest.raises(ValueError):
        f._one_call(["CPI_FOOD"], 0)
    assert tags == ["analyst gCPI_FOOD s0"]  # no repair call: the three left belong to the end of the unit


def test_the_review_call_is_off_by_default():
    _v2()
    from t4agent import config

    assert config.REVIEW == 0 and config.REASONS_ROUNDS == 2


# ------------------------------------------------------------------ v18: the EPS anchor calibrated on SEC filings
def test_the_eps_anchor_carries_one_quarter_and_widens():
    _v2()
    from t4agent import eps as E
    from t4agent.facts import EpsRecord

    rec = EpsRecord.__new__(EpsRecord)
    rec.cur, rec.prior, rec.year, rec.month, rec.kind = 1.20, 1.00, 2023, 3, "10-Q"
    a1 = E.EpsAnchor("x", 0, 0, 1.0 + E.PHI * 0.2, 0, 1.0, rec, 1, "level")
    assert a1.p_up == E.EPS_P_UP_AFTER_UP == 0.75
    rec2 = EpsRecord.__new__(EpsRecord)
    rec2.cur, rec2.prior, rec2.year, rec2.month, rec2.kind = 0.80, 1.00, 2023, 3, "10-Q"
    assert E.EpsAnchor("x", 0, 0, 1.0, 0, 1.0, rec2, 2, "level").p_up == E.EPS_P_UP_AFTER_DOWN == 0.50
    assert E.CARRY_FAR == 0.16 and (E.EPS_SCALE_NEXT, E.EPS_SCALE_FAR) == (1.97, 2.22)  # v20 (v18: 0, 1.8 for both)


def test_an_eps_fall_no_longer_reads_as_a_confident_down():
    """In SEC pseudo-forecasts a fall in the latest quarter left the next one a coin flip (50% up on the 5,162
    comparable cases), while the anchor's own distribution called it down with 0.6-0.8 confidence: the direction labels
    blend in the calibrated rate."""
    _v2()
    from t4agent.corpus import load_corpus, map_entities
    from t4agent.llm import HouseClient
    from t4agent.predict import Evidence, Forecast, Forecaster
    from t4agent.task import load_task

    unit = UNITS / "t4-eps-yoy-2023Q2-mixed"
    spec = load_task(unit / "task.json")
    corpus = load_corpus(unit / "corpus", spec)
    f = Forecaster(spec, corpus, Evidence(spec, corpus, *map_entities(spec, corpus)), HouseClient(deadline=0), 0)
    f.build_anchors()
    fell = [e for e, a in f.anchor.items() if a.latest.cur < a.latest.prior]
    assert fell and all(f.anchor[e].p_up == 0.50 for e in fell)
    eid = fell[0]
    a = f.anchor[eid]
    fc = Forecast(eid=eid, source="llm", model_probs={"up": 0.1, "down": 0.9}, median=a.level, p05=a.level - 0.1, p95=a.level + 0.1)
    f._settle_label(fc, spec.entity(eid))
    with_prior = fc.probs["up"]
    f._blend_eps_direction = lambda fc: None
    fc2 = Forecast(eid=eid, source="llm", model_probs={"up": 0.1, "down": 0.9}, median=a.level, p05=a.level - 0.1, p95=a.level + 0.1)
    Forecaster._settle_label(f, fc2, spec.entity(eid))
    assert with_prior > fc2.probs["up"] and abs(with_prior - (0.5 * fc2.probs["up"] + 0.5 * 0.50)) < 1e-9


def test_the_eps_calibration_reproduces_from_the_sec_filings():
    """Provenance: tools/calibrate_eps.py on ../calib_data/sec (skipped without the data)."""
    _v2()
    import statistics as st

    if not (AGENT.parent / "calib_data" / "sec" / "eps_AAPL.json").exists():
        pytest.skip("SEC data not present")
    sys.path.insert(0, str(AGENT / "tools"))
    import calibrate_eps as CE

    from t4agent import eps as E

    raw = CE.cases(clean=False)
    cs = CE.cases()
    # v20: the AAPL quarter of 2014-09-27 compared 1.42 with 8.26, its prior-year quarter before the 7:1 split
    aapl = [c for c in raw if c["t"] == "AAPL" and c["end"] == "2014-09-27"][0]
    assert aapl["basis_change"] and abs(aapl["base"] - 8.26) < 0.01 and abs(aapl["comp"] - 1.18) < 0.01
    assert len(raw) - len(cs) == 156 and len(cs) == 5162
    nz = [c for c in cs if c["actual"] != c["base"] and c["cur"] != c["prior"]]
    up = st.mean(c["actual"] > c["base"] for c in nz if c["cur"] > c["prior"])
    dn = st.mean(c["actual"] > c["base"] for c in nz if c["cur"] < c["prior"])
    assert abs(up - E.EPS_P_UP_AFTER_UP) < 0.005 and abs(dn - E.EPS_P_UP_AFTER_DOWN) < 0.005
    one = [c for c in cs if c["k"] == 1]
    assert CE.point_quality(one, lambda c: E.PHI) > max(CE.point_quality(one, lambda c: p) for p in (0.0, 0.2, 0.6, 1.0))
    far = [c for c in cs if c["k"] >= 2]
    assert CE.point_quality(far, lambda c: E.CARRY_FAR) > CE.point_quality(far, lambda c: 0.0)
    assert abs(CE.interval_factor(one) - E.EPS_SCALE_NEXT) < 0.01 and abs(CE.interval_factor(far) - E.EPS_SCALE_FAR) < 0.01


# ------------------------------------------------------------------ v21.1: unseen table shapes, fallback rules
def _dated(values, start_year=2018, monthly=True):
    if monthly:
        return [f"{start_year + i // 12}-{i % 12 + 1:02d} | {v}" for i, v in enumerate(values)]
    import datetime as dt
    d0 = dt.date(2017, 1, 2)
    return [f"{(d0 + dt.timedelta(days=i)).isoformat()} | {v}" for i, v in enumerate(values)]


def test_an_unnamed_date_table_in_the_entitys_own_document_is_its_series():
    """A held-out family need not name its column after the series: v21 found no series in 'date | value' and gave a
    housing-starts row 0 and an unemployment-rate row +/-20 points."""
    _v2()
    from t4agent.stats import entity_forecast, find_series

    corpus = _table_corpus("date | value", _dated([1200, 1250, 1180, 1230, 1260, 1240, 1270, 1282]))
    ent = {"entity_id": "HOUST", "units": "thousands of units, SAAR"}
    assert find_series(corpus, ["d"], ent, "housing_starts_thousands_saar_first_print", {"d"}) is None  # v21 rule
    # the unit text does not rename the column: an unnamed `value` column still needs allow_own
    assert find_series(corpus, ["d"], ent, "btc_vs_threshold", {"d"}) is None
    s = find_series(corpus, ["d"], ent, "housing_starts_thousands_saar_first_print", {"d"}, allow_own=True)
    assert s is not None and s.kind == "own" and s.values[-1] == 1282.0
    f = entity_forecast(corpus, ["d"], ent, "housing_starts_thousands_saar_first_print", 2, {"d"}, allow_own=True, policy=_eligible_policy())
    assert 1100 < f.value < 1400 and f.q90 > 0
    # only in the entity's own documents
    assert find_series(corpus, ["d"], ent, "housing_starts_thousands_saar_first_print", set(), allow_own=True) is None


def test_label_only_reply_uses_the_series_the_target_already_names():
    """A classification reply that carries only a label leaves median empty. When the target name already
    matches a column, the numeric interval sits on that series. A renamed target does not borrow the unit."""
    _v2()
    import datetime as dt

    from t4agent.calibrate import finalize
    from t4agent.corpus import Corpus, Doc
    from t4agent.llm import HouseClient
    from t4agent.predict import Evidence, Forecaster

    rows = [f"2024-{m:02d}-01 | 2.{40 + m}" for m in range(1, 9)]
    text = "auction_date | bid_to_cover\n" + "\n".join(rows)
    corpus = Corpus(docs={"d": Doc("d", dt.date(2024, 10, 1), text, {})}, chunks=[])
    spec = _bare_spec(
        target_name="bid_to_cover_ratio", labels=["strong_demand", "soft_demand"],
        prompt="Classify each auction as strong_demand or soft_demand.",
        cutoff=dt.date(2024, 10, 31), cutoff_str="2024-10-31", resolution_str="2024-11-15",
        entities=[{"entity_id": "AUC", "unit": "bid_to_cover_ratio"}], entity_ids=["AUC"],
    )
    forecaster = Forecaster(spec, corpus, Evidence(spec, corpus, {"AUC": ["d"]}, []), HouseClient(deadline=0), 0)
    fc = forecaster.aggregate({"AUC": [{"entity_id": "AUC", "label": "strong_demand"}]})["AUC"]
    assert fc.median is not None and fc.median > 1.5
    out = finalize(spec, fc, None, ent=spec.entity("AUC"))
    assert (out["lo"], out["hi"]) != (0.0, 1.0)
    assert out["lo"] < 2.48 < out["hi"]

    renamed = _bare_spec(
        target_name="btc_vs_2_5", labels=["strong_demand", "soft_demand"],
        prompt="Classify each auction as strong_demand or soft_demand.",
        cutoff=dt.date(2024, 10, 31), cutoff_str="2024-10-31", resolution_str="2024-11-15",
        entities=[{"entity_id": "AUC", "unit": "bid_to_cover_ratio"}], entity_ids=["AUC"],
    )
    renamed_fc = Forecaster(renamed, corpus, Evidence(renamed, corpus, {"AUC": ["d"]}, []), HouseClient(deadline=0), 0)
    missed = renamed_fc.aggregate({"AUC": [{"entity_id": "AUC", "label": "strong_demand"}]})["AUC"]
    assert missed.median is None


def test_a_unit_word_does_not_select_a_ratio_column():
    """Unit text is not a column name. `bid_to_cover_ratio` must not select a header named `ratio`
    when the target itself (`btc_vs_threshold`) names neither column."""
    _v2()
    from t4agent.stats import find_series

    rows = [f"2024-{m:02d}-01 | 1.9 | 2.{40 + m}" for m in range(1, 9)]
    corpus = _table_corpus("auction_date | ratio | bid_to_cover", rows)
    ent = {"entity_id": "AUC", "unit": "bid_to_cover_ratio"}
    assert find_series(corpus, ["d"], ent, "btc_vs_threshold", {"d"}) is None
    named = find_series(corpus, ["d"], ent, "bid_to_cover", {"d"})
    assert named is not None and named.header == "bid_to_cover" and named.values[-1] == 2.48


def test_a_vintage_table_is_read_off_its_latest_vintage():
    _v2()
    from t4agent.stats import find_series

    rows = [f"2018-{m:02d} | {100 + m} | {100.5 + m} | {101 + m}" for m in range(1, 9)]
    corpus = _table_corpus("reference_period | as_of_2018-08-03 | as_of_2018-09-07 | as_of_2018-10-05", rows)
    s = find_series(corpus, ["d"], {"entity_id": "X"}, "level_first_print", {"d"}, allow_own=True)
    assert s is not None and s.values[-1] == 109.0 and s.header == "as_of_2018-10-05"


def test_a_percent_change_target_reads_a_level_as_its_log_and_never_its_level():
    """v21 submitted an industrial-production index level (110) as its month-over-month percent change."""
    _v2()
    import math

    from t4agent.stats import entity_forecast

    levels = [100 * 1.002 ** i for i in range(24)]
    corpus = _table_corpus("date | value", _dated([f"{x:.3f}" for x in levels]))
    ent = {"entity_id": "INDPRO", "units": "index 2012=100 (target: percent change)"}
    f = entity_forecast(corpus, ["d"], ent, "ip_index_mom_pct_first_print", 2, {"d"}, allow_own=True, policy=_eligible_policy())
    assert f.series.kind == "own-rel" and abs(f.value) < 1.0 and f.q90 < 1.0
    assert abs(f.series.values[-1] - 100 * math.log(round(levels[-1], 3))) < 1e-6
    # a year-over-year rate is not derived from a monthly level, and a revision is measured against another estimate
    assert entity_forecast(corpus, ["d"], ent, "cpi_yoy_pct_change", 2, {"d"}, allow_own=True, policy=_eligible_policy()) is None
    assert entity_forecast(corpus, ["d"], {"entity_id": "X"}, "next_estimate_revision_pct", 2, {"d"},
                           allow_own=True, policy=_eligible_policy()) is None


def test_a_level_several_steps_ahead_is_backtested_at_its_horizon():
    """A daily price a week out: v21's one-step error would size a five-step interval."""
    _v2()
    import random

    from t4agent.stats import entity_forecast

    rnd = random.Random(7)
    x, vals = 50.0, []
    for _ in range(300):
        x += rnd.gauss(0, 1.0)
        vals.append(round(x, 2))
    corpus = _table_corpus("date | value", _dated(vals, monthly=False))
    ent = {"entity_id": "WTI"}
    one = entity_forecast(corpus, ["d"], ent, "price_level_on_target_date", 1, {"d"}, allow_own=True, policy=_eligible_policy())
    week = entity_forecast(corpus, ["d"], ent, "price_level_on_target_date", 7, {"d"}, allow_own=True, policy=_eligible_policy())
    assert week.q90 > 1.6 * one.q90


def test_a_flat_series_keeps_its_printed_resolution_as_its_scale():
    _v2()
    from t4agent.stats import Series, _resolution, forecast, series_half_width

    s = Series("d", "value", [1.2] * 26, [f"2019-{i:02d}" for i in range(26)], 7.0, False, kind="own")
    assert _resolution(s.values) == 0.1 or abs(_resolution(s.values) - 0.1) < 1e-12
    assert forecast(s, "claims_value_first_print", 1, {"entity_id": "IURSA"}).q90 > 0
    assert series_half_width(s, "claims_value_first_print", 1, {"entity_id": "IURSA"}, policy=_eligible_policy()) > 0


def test_a_change_target_never_takes_a_level_column_as_its_base():
    _v2()
    from types import SimpleNamespace

    from t4agent.predict import baseline_value

    spec = SimpleNamespace(target_name="ip_index_mom_pct_first_print", target_type="regression")
    ent = {"entity_id": "INDPRO", "latest_index_2019_03": 110.2, "latest_mom_pct_2019_03": -0.105}
    assert baseline_value(spec, ent, None) == -0.105
    assert baseline_value(spec, {"entity_id": "INDPRO", "latest_index_2019_03": 110.2}, None) == 0.0
    # the public CPI row keeps v21's carried-forward first print; positioning keeps zero (a trailing change is no base)
    cpi = SimpleNamespace(target_name="cpi_component_mom_first_print", target_type="regression")
    assert baseline_value(cpi, {"entity_id": "C", "latest_published_mom_pct": 0.18}, None) == 0.18
    cot = SimpleNamespace(target_name="net_positioning_change_pct_oi_rank", target_type="ranking")
    assert baseline_value(cot, {"entity_id": "K", "trailing_4wk_net_change_pct_oi": 4.99, "net_pct_oi_20241022": 1.0},
                          None) == 0.0


def test_an_unknown_label_set_falls_back_to_its_status_quo():
    from t4agent.predict import prior_label

    assert prior_label(["cut", "hold", "hike"]) == "hold"
    assert prior_label(["appreciate", "stable", "depreciate"]) == "stable"
    assert prior_label(["beat", "miss", "inline"]) == "beat"
    assert prior_label(["positive_reaction", "negative_reaction", "flat"]) == "positive_reaction"


def test_a_quote_that_is_only_an_entity_name_is_content_free():
    from types import SimpleNamespace

    from t4agent.cite import contentful, unit_names

    spec = SimpleNamespace(entities=[{"entity_id": "DCOILWTICO",
                                      "name": "Crude Oil Prices: West Texas Intermediate (WTI) - Cushing, Oklahoma"}])
    title = "DCOILWTICO - Crude Oil Prices: West Texas Intermediate (WTI) - Cushing, Oklahoma"
    assert contentful(title) and not contentful(title, unit_names(spec))
    assert contentful("- Latest observation: 2017-06-05 = 47.4 Dollars per Barrel.", unit_names(spec))


def test_a_ranking_keeps_the_models_order_against_an_unnamed_table_series():
    _v2()
    from types import SimpleNamespace

    from t4agent.predict import Forecast, Forecaster
    from t4agent.stats import Series, StatForecast

    f = Forecaster.__new__(Forecaster)
    f.spec = SimpleNamespace(target_type="ranking", target_name="unemployment_rate_change_pp_rank",
                             entity=lambda eid: {"entity_id": eid})
    s = Series("d", "value", [4.0] * 12, [f"2017-{i:02d}" for i in range(12)], 30.0, False, kind="own")
    f.stat = {"A": StatForecast("momentum", 0.3, 0.1, 0.2, 10, 4.0, s)}
    f.plan, f.prior = None, {}
    fc = Forecast(eid="A", source="llm", median=-0.1, p05=-0.3, p95=0.1)
    f._combine_with_stats(fc)
    assert fc.median == -0.1 and fc.stat_q90 == 0.2


# ------------------------------------------------------------------ v21.2: label definitions stated as moves
def test_a_move_with_its_direction_and_a_symmetric_band_are_read_as_rules():
    """v21 read 'falls by more than 1.0%' as Q > 1.0 (the region of 'rises by more than 1.0%') and '1.0% or less in
    either direction' as Q < 1.0: overlapping rules that, when they still summed to about one, labelled -3% 'stable'."""
    from t4agent.labels import _assertion_cond, assertion_rules, implied_probs

    labels = ["appreciate", "stable", "depreciate"]
    rules = assertion_rules(labels, {
        "appreciate": "the currency's value in U.S. dollars rises by more than 1.0% from the 2019-05-01 rate",
        "stable": "the currency's value changes by 1.0% or less in either direction between the two dates",
        "depreciate": "the currency's value falls by more than 1.0% from the 2019-05-01 rate"})
    assert rules == {"appreciate": "Q > 1.0", "stable": "-1.0 < Q < 1.0", "depreciate": "Q < -1.0"}
    p = implied_probs(rules, labels, {}, -3.0, -5.7, -0.3)
    assert max(p, key=p.get) == "depreciate"
    assert _assertion_cond("the spread widens by more than 10 bp") == "Q > 10.0"
    assert _assertion_cond("the price falls by more than 5%") == "Q < -5.0"
    assert _assertion_cond("the yield moves within +/-25 bp") == "-25.0 < Q < 25.0"
    assert _assertion_cond("more than 5 percent above the consensus") is None  # a relative threshold stays unread
    assert _assertion_cond("the bid-to-cover ratio is above 2.50") == "Q > 2.5"  # v13-v21 forms unchanged


def test_a_change_quantity_of_a_classification_row_never_takes_a_level_base():
    """A currency's percent change took its quote (111.5 yen) as the naive base: the hedge stretched [-2.6, 2.8] to
    [-2.6, 57]."""
    from types import SimpleNamespace

    from t4agent.predict import quantity_baseline

    spec = SimpleNamespace(target_name="currency_move_vs_usd_1m", target_type="classification")
    ent = {"entity_id": "JPYUSD", "rate_at_cutoff": 111.5, "unit": "percent change of the value of 1 JPY in U.S. dollars"}
    assert quantity_baseline(spec, ent, {"baseline_field": "rate_at_cutoff"}) is None
    # levels keep their base: EPS against consensus or last year's quarter, a revised estimate against the latest one
    eps = SimpleNamespace(target_name="eps_yoy_direction", target_type="classification")
    assert quantity_baseline(eps, {"entity_id": "AMD", "prior_year_q_eps": 0.27, "currency": "USD/share"},
                             {"interval_quantity": "GAAP diluted EPS for the three months ended 2023-06-30"}) == 0.27
    rev = SimpleNamespace(target_name="next_estimate_revision_direction", target_type="classification")
    assert quantity_baseline(rev, {"entity_id": "P", "latest_precutoff_estimate": 158779.0, "units": "thousands"},
                             {"baseline_field": "latest_precutoff_estimate",
                              "interval_quantity": "the next published estimate (the revised value)"}) == 158779.0


def test_a_fallback_label_follows_its_rules_at_the_naive_value():
    """No model reply and no anchor: v21 took the first label ('rise' for a three-way yield move, 'above' for a CPI
    threshold); the rules at the naive value (no change, or the latest print the row states) give the naive's label.
    A base-rate prior chosen for a label family (beat, up, no_event) stands."""
    from t4agent.labels import rule_label

    yield_rules = {"rise": "Q > 10.0", "steady": "-10.0 < Q < 10.0", "fall": "Q < -10.0"}
    assert rule_label(yield_rules, ["rise", "steady", "fall"], {}, 0.0) == "steady"
    assert rule_label(yield_rules, ["rise", "steady", "fall"], {}, -12.0) == "fall"
    band = {"strong": "Q > 0.5", "weak": "Q < -0.5", "steady": "otherwise"}
    assert rule_label(band, ["strong", "steady", "weak"], {}, 0.7) == "strong"
    assert rule_label(band, ["strong", "steady", "weak"], {}, 0.1) == "steady"
    assert rule_label({"a": "Q > 1.0", "b": "Q > 0.0"}, ["a", "b"], {}, 2.0) is None  # overlapping rules: no label
    _v2()
    import time
    from pathlib import Path as P

    from t4agent.corpus import load_corpus, map_entities
    from t4agent.llm import HouseClient
    from t4agent.predict import Evidence, Forecaster
    from t4agent.task import load_task

    unit = ROOT / "units_pre2022" / "units" / "t4-x-cpithr-201906"
    if not unit.exists():
        pytest.skip("stress units not built")
    spec = load_task(unit / "task.json")
    corpus = load_corpus(unit / "corpus", spec)
    ev = Evidence(spec, corpus, *map_entities(spec, corpus))
    ev.build_packs([])
    f = Forecaster(spec, corpus, ev, HouseClient(deadline=0), time.time())
    out = f.aggregate({})
    for ent in spec.entities:
        want = "above" if ent["latest_published_mom_pct"] > 0.2 else "at_or_below"
        assert out[ent["entity_id"]].label == want


def test_a_rule_less_classification_fallback_sizes_its_quantity_from_the_eps_anchor():
    """A growth-band unit whose bands are stated only in words (no rules the code reads): v21.2 at first sent each EPS
    out at last year's value with a +/-20 default, ignoring the anchor it had read off the filings."""
    _v2()
    import time

    from t4agent.corpus import load_corpus, map_entities
    from t4agent.llm import HouseClient
    from t4agent.predict import Evidence, Forecaster
    from t4agent.task import load_task

    unit = ROOT / "units_pre2022" / "units" / "t4-x-epsgrowth3-2021Q1"
    if not unit.exists():
        pytest.skip("stress units not built")
    spec = load_task(unit / "task.json")
    corpus = load_corpus(unit / "corpus", spec)
    ev = Evidence(spec, corpus, *map_entities(spec, corpus))
    ev.build_packs([])
    f = Forecaster(spec, corpus, ev, HouseClient(deadline=0), time.time())
    f.build_anchors()
    out = f.aggregate({})
    for ent in spec.entities:
        fc = out[ent["entity_id"]]
        if ent["entity_id"] in f.anchor:
            assert fc.median is not None and fc.stat_q90 is not None and fc.stat_q90 < 10 * max(1.0, abs(fc.median))


# ------------------------------------------------------------------ v21.5: robustness fixes from the v21.3 audit
def _run_patched(unit_dir: Path, out: Path, patch: str = "", argv: list[str] | None = None,
                 extra_env: dict | None = None) -> subprocess.CompletedProcess:
    """The CLI in a fresh process after `patch` (Python run against `cli`), e.g. to inject a failure."""
    code = "import os, sys\nfrom t4agent import cli\n" + patch + "\nos._exit(cli.main(sys.argv[1:]))\n"
    env = {k: v for k, v in os.environ.items() if not k.startswith(("MODEL_", "T4_"))}
    env.update(extra_env or {})
    argv = argv if argv is not None else ["analyze", "--task", str(unit_dir / "task.json"), "--corpus",
                                          str(unit_dir / "corpus"), "--out", str(out)]
    return subprocess.run([sys.executable, "-c", code, *argv], cwd=AGENT, env=env, capture_output=True, text=True,
                          timeout=300)


def _task_rows(unit_dir: Path) -> tuple[str, dict]:
    from qfbench2_track_analysis.corpus import task_table_text

    return task_table_text(json.loads((unit_dir / "task.json").read_text()))


def _no_false_claims(unit_dir: Path, answer: dict) -> None:
    R = _rail()
    found = R.check_claim_rules(answer, unit_dir, token_counter=None)
    assert all(f.code == "claim_tokens_unchecked" for f in found), found
    assert R.claim_penalty_preview(answer, unit_dir, token_counter=None)["factor"] == 1.0


def test_v215_an_entity_without_a_citable_document_claims_its_own_task_row(tmp_path):
    """v21.3 wrote claims [] for an entity whose only document is dated after the cutoff (a stale filing): the schema
    refuses the whole unit. v21.5 quotes a figure of the entity's own task-table row (doc_id "task")."""
    _v2()
    from t4agent.answer import claim_ok, task_claim, validate
    from t4agent.corpus import load_corpus
    from t4agent.task import load_task

    unit = _clone("t4-credit-event-2023", tmp_path / "u")
    doc = unit / "corpus" / "EDGAR_0000084129_10Q_20230104.json"  # RAD's only document
    d = json.loads(doc.read_text())
    d["doc_date"] = "2023-06-30"  # after the 2023-03-31 cutoff
    doc.write_text(json.dumps(d))
    _rehash(unit)
    out = tmp_path / "answer.json"
    proc = run_agent(unit, out)
    assert proc.returncode == 0, proc.stderr[-2000:]
    answer = json.loads(out.read_text())
    official_check(unit, answer)
    _no_false_claims(unit, answer)
    text, ranges = _task_rows(unit)
    row = next(p for p in answer["entity_predictions"] if p["entity_id"] == "RAD")
    assert [c["doc_id"] for c in row["claims"]] == ["task"]
    c = row["claims"][0]
    assert ranges["RAD"][0] <= c["span_start"] < c["span_end"] <= ranges["RAD"][1]
    assert c["claim"] == text[c["span_start"]:c["span_end"]] and any(ch.isdigit() for ch in c["claim"])
    # the model path keeps such an answer: validate() and claim_ok() accept the row's own span, and no other row's
    spec = load_task(unit / "task.json")
    corpus = load_corpus(unit / "corpus", spec)
    assert validate(spec, corpus, answer) == [] and claim_ok(corpus, "RAD", c, spec) and not claim_ok(corpus, "RAD", c)
    other = dict(c, span_start=ranges["YELL"][0] + 1, span_end=ranges["YELL"][0] + 9)
    other["claim"] = text[other["span_start"]:other["span_end"]]
    bad = json.loads(json.dumps(answer))
    next(p for p in bad["entity_predictions"] if p["entity_id"] == "RAD")["claims"] = [other]
    assert any("RAD" in e for e in validate(spec, corpus, bad)) and not claim_ok(corpus, "RAD", other, spec)
    assert task_claim(spec, "NOPE") is None


def test_v215_a_failing_row_or_fallback_stage_still_leaves_an_answer_on_disk(tmp_path):
    """v21.3: an exception in one entity's finalize or citations left no answer.json at all (whole-unit zero)."""
    unit = UNITS / "t4-credit-event-2023"
    one = tmp_path / "one.json"
    proc = _run_patched(unit, one, (
        "orig = cli.build_citations\n"
        "def boom(spec, corpus, ev, fc, terms):\n"
        "    if fc.eid == spec.entity_ids[1]:\n"
        "        raise KeyError('injected')\n"
        "    return orig(spec, corpus, ev, fc, terms)\n"
        "cli.build_citations = boom\n"))
    assert proc.returncode == 0 and "row BBY failed" in proc.stderr, proc.stderr[-2000:]
    answer = json.loads(one.read_text())
    official_check(unit, answer)
    _no_false_claims(unit, answer)
    rows = {p["entity_id"]: p for p in answer["entity_predictions"]}
    assert list(rows) == [e["entity_id"] for e in json.loads((unit / "task.json").read_text())["entities"]]
    assert [c["doc_id"] for c in rows["BBY"]["claims"]] == ["task"]
    assert all(c["doc_id"] != "task" for e, p in rows.items() if e != "BBY" for c in p["claims"])
    stage = tmp_path / "stage.json"
    proc = _run_patched(unit, stage, "def boom(*a, **k):\n    raise RuntimeError('injected')\ncli._answer = boom\n")
    assert proc.returncode == 0 and "emergency answer" in proc.stderr, proc.stderr[-2000:]
    answer = json.loads(stage.read_text())
    official_check(unit, answer)
    _no_false_claims(unit, answer)
    assert len(answer["entity_predictions"]) == 8
    assert all(c["doc_id"] == "task" for p in answer["entity_predictions"] for c in p["claims"])


def test_v215_a_document_without_a_text_field_is_read_but_never_cited(tmp_path):
    """The scorer slices a claim's span out of a document's `text` field only: a spans-only document's every claim is
    out of range (factor 0 in v21.3). Its entity falls back to its own task-table row. A reason may still cite it: the
    reasons rail resolves the spans joined by a space (citation_rail._doc_text), and dropping the reason only loses."""
    _v2()
    from t4agent.corpus import load_corpus
    from t4agent.task import load_task

    unit = _clone("t4-eps-yoy-2023Q2-mixed", tmp_path / "u")
    did = "EDGAR_0000002488_10Q_20230503"  # AMD's only document
    f = unit / "corpus" / f"{did}.json"
    d = json.loads(f.read_text())
    words = d.pop("text").split(" ")
    d["spans"] = [{"text": " ".join(words[i:i + 40])} for i in range(0, len(words), 40)]
    f.write_text(json.dumps(d))
    _rehash(unit)
    spec = load_task(unit / "task.json")
    corpus = load_corpus(unit / "corpus", spec)
    assert did in corpus.docs and did in corpus.nocite and not corpus.admits("AMD", did)
    out = tmp_path / "answer.json"
    assert run_agent(unit, out).returncode == 0
    answer = json.loads(out.read_text())
    official_check(unit, answer)
    _no_false_claims(unit, answer)
    cited = {c["doc_id"] for p in answer["entity_predictions"] for c in p["claims"]}
    assert did not in cited
    R = _rail()
    cutoff = json.loads((unit / "task.json").read_text())["cutoff_date"]
    assert answer.get("submitted_reasons") and not R.check_submitted_reasons(answer, R.load_corpus(unit / "corpus"),
                                                                             cutoff)


def test_v215_manifest_labels_are_read_from_the_root_manifest_first(tmp_path):
    """The scorer reads the unit's root manifest first (CorpusIndex._read_manifest); v21.3 read corpus/manifest.json
    first, so a pair that disagreed made it cite documents the scorer labels for another entity."""
    _v2()
    from t4agent.corpus import manifest_labels

    unit = _clone("t4-credit-event-2023", tmp_path / "u")
    man_path = unit / "corpus" / "manifest.json"
    man = json.loads(man_path.read_text())
    for e in man["files"]:
        if e["path"] == "corpus/EDGAR_0000084129_10Q_20230104.json":
            e["entity_ids"] = ["YELL"]
    man_path.write_text(json.dumps(man))
    labels = manifest_labels(unit, unit / "corpus")
    assert labels["EDGAR_0000084129_10Q_20230104"] == (("RAD",), False)


def test_v215_a_hung_request_returns_by_the_deadline_plus_15_s(monkeypatch):
    """The read timeout applies per socket read: a reply that trickles in outlived it and held the final answer until
    the watchdog. Every call now returns by the deadline + 15 s, also on a shortened (card-derived) clock."""
    _v2()
    import threading
    import time

    import requests

    from t4agent import config
    from t4agent.llm import HouseClient, LLMUnavailable

    for k, v in (("MODEL_ENDPOINT", "http://house.invalid"), ("MODEL_NAME", "m"), ("MODEL_TOKEN", "t")):
        monkeypatch.setenv(k, v)
    hard, llm_deadline, scale = config.derive_clock(300.0)
    client = HouseClient(deadline=time.time() - 14.0, clock_scale=scale)  # its scaled deadline passed 14 s ago
    release = threading.Event()

    def trickle(url, **kw):
        release.wait(60)
        raise requests.ReadTimeout("never")

    monkeypatch.setattr(client._session, "post", trickle)
    start = time.time()
    with pytest.raises(LLMUnavailable):
        client._post("http://house.invalid/v1/chat/completions", headers={}, data="{}", timeout=(10, 170))
    assert time.time() <= client.deadline + 15.5 and time.time() - start < 5
    release.set()
    # in chat(): the abandoned request stays spent, and no second attempt follows
    client = HouseClient(deadline=time.time() + 60, clock_scale=scale)
    calls = []

    def abandoned(url, **kw):
        calls.append(url)
        raise LLMUnavailable("request outlived the deadline")

    monkeypatch.setattr(client, "_post", abandoned)
    with pytest.raises(LLMUnavailable):
        client.chat([{"role": "user", "content": "x"}], reasoning_budget=None, max_attempts=2)
    assert len(calls) == 1 and client.used == 1


def test_v215_odd_reply_shapes_are_transport_or_malformed_never_a_crash(monkeypatch):
    _v2()
    import time

    from t4agent.llm import HouseClient, LLMUnavailable

    for k, v in (("MODEL_ENDPOINT", "http://house.invalid"), ("MODEL_NAME", "m"), ("MODEL_TOKEN", "t")):
        monkeypatch.setenv(k, v)

    class Resp:
        status_code = 200

        def __init__(self, body):
            self.body = body

        def json(self):
            return self.body

    client = HouseClient(deadline=time.time() + 60)
    ok = {"choices": [{"message": {"content": [{"type": "text", "text": '{"a": 1}'}]}, "finish_reason": "stop"}],
          "usage": [3]}
    monkeypatch.setattr(client._session, "post", lambda *a, **k: Resp(ok))
    rep = client.chat([{"role": "user", "content": "x"}], reasoning_budget=None, max_attempts=1)
    assert rep.content == '{"a": 1}' and rep.completion_tokens == 0  # an odd usage block spoils nothing
    for bad in ({"choices": ["text"]}, [1, 2]):
        client = HouseClient(deadline=time.time() + 60)
        monkeypatch.setattr(client._session, "post", lambda *a, b=bad, **k: Resp(b))
        with pytest.raises(LLMUnavailable, match="malformed"):
            client.chat([{"role": "user", "content": "x"}], reasoning_budget=None, max_attempts=1)

    def odd(*a, **k):
        raise OSError("socket closed")  # not a requests exception

    client = HouseClient(deadline=time.time() + 60)
    monkeypatch.setattr(client._session, "post", odd)
    with pytest.raises(LLMUnavailable, match="OSError"):
        client.chat([{"role": "user", "content": "x"}], reasoning_budget=None, max_attempts=1)


def test_v215_the_clock_follows_the_cards_unit_time_limit(tmp_path, monkeypatch):
    """Card [agent].timeout_sec 600 gives v21.3's clocks exactly (470 s hard stop, 400 s LLM deadline, margins x1);
    300 gives 210 / 140 s with every absolute margin of the model path x0.35 (v21.3 kept them in seconds, and below a
    185 s LLM deadline the analysts had no window at all)."""
    _v2()
    import re
    import tomllib

    from t4agent import config
    from t4agent.llm import HouseClient, margin
    from t4agent.task import card_timeout

    monkeypatch.delenv("T4_TIME_BUDGET", raising=False)
    assert config.derive_clock(600.0) == (470.0, 400.0, 1.0) == config.derive_clock(None)
    assert config.derive_clock(300.0) == (210.0, 140.0, 0.35)
    assert config.derive_clock(900.0) == (470.0, 400.0, 1.0)  # a longer limit never extends the clock
    unit = _clone("t4-credit-event-2023", tmp_path / "u")
    assert card_timeout(unit) == 600.0 and card_timeout(tmp_path) is None
    card = unit / "card.toml"
    card.write_text(re.sub(r"(\[agent\][^\[]*?timeout_sec\s*=\s*)[0-9.]+", r"\g<1>300.0", card.read_text(), count=1))
    assert tomllib.loads(card.read_text())["agent"]["timeout_sec"] == 300.0 and card_timeout(unit) == 300.0
    _rehash(unit)
    out = tmp_path / "answer.json"
    # B6: the watchdog is the Timer the CLI starts; it fires at the hard stop itself (v21.3: 30 s after it)
    timer = ("import threading\n_Timer = threading.Timer\n"
             "class Timer(_Timer):\n"
             "    def __init__(self, interval, fn, *a, **k):\n"
             "        print(f'WATCHDOG {interval:.1f}', file=sys.stderr)\n"
             "        super().__init__(interval, fn, *a, **k)\n"
             "threading.Timer = Timer\n")
    proc = _run_patched(unit, out, timer)
    assert proc.returncode == 0 and "clock: hard stop 210 s, LLM deadline 140 s, margins x0.35" in proc.stderr
    assert "WATCHDOG 210.0" in proc.stderr, proc.stderr[-2000:]
    official_check(unit, json.loads(out.read_text()))
    proc = _run_patched(UNITS / "t4-credit-event-2023", tmp_path / "a600.json", timer)
    assert "clock: hard stop 470 s, LLM deadline 400 s, margins x1.00" in proc.stderr
    assert "WATCHDOG 470.0" in proc.stderr
    f = _forecaster("t4-credit-event-2023", None)
    f.client = HouseClient(deadline=0, clock_scale=0.35)
    assert abs(f.reserve() - 0.35 * config.REASONS_RESERVE_SEC) < 1e-9 and abs(margin(f.client, 25) - 8.75) < 1e-9
    f.client = HouseClient(deadline=0)
    assert f.reserve() == config.REASONS_RESERVE_SEC and margin(f.client, 25) == 25


def test_v215_disguised_deny_list_words_are_caught_as_the_grader_folds_them():
    """The grader scans a folded view (NFKC, slash and colon look-alikes, no format characters or Hangul fillers):
    one hit refuses the unit's whole reasoning. v21.3 matched the raw text and let these through."""
    _v2()
    from t4agent.reasons import clean, denied

    R = _rail()
    disguised = ["ｕｎｉｔｓ／ data", "see units∕ here", "see units⁄ here",
                 "the leadㅤerboard says", "the can​ary row", "\U0001d41c\U0001d41a\U0001d427\U0001d41a"
                 "\U0001d42b\U0001d432 token", "ｌｅａｄｅｒｂｏａｒｄ"]
    for text in disguised:
        official = any(p in R.fold_for_detection(text)[0].lower() for p in R.DENY_LIST)
        assert official and denied(text) and clean(text + " and more words") is None, text
    assert not denied("The cited units of output rose") and not denied("Gains in the core index")


def test_v215_a_ranking_without_model_replies_ranks_by_its_code_baselines():
    """v21.3 left every ranking row without a model reply at 0: a full fallback carried no ordering (0.5) although the
    positioning baseline ranks at 0.565 on the CFTC replay. A row without a baseline either stays at 0 on a change
    target and, on a level-valued target, takes the middle of the others (0 would rank it last)."""
    _v2()
    from types import SimpleNamespace

    from t4agent.predict import Forecast, Forecaster
    from t4agent.stats import Series, StatForecast

    f = _forecaster("t4-cotpos-202411-us10", None)
    out = f.aggregate({})
    vals = {e: fc.median for e, fc in out.items()}
    assert len(set(vals.values())) > 5
    assert all(abs(vals[e] - f.stat[e].value) < 1e-12 for e in vals if f.stat.get(e) is not None)
    named = Series("d", "value", [4.0] * 12, [f"2017-{i:02d}" for i in range(12)], 30.0, False)
    own = Series("d", "value", [4.0] * 12, [f"2017-{i:02d}" for i in range(12)], 30.0, False, kind="own")
    for target, middle in (("net_long_contracts_level_rank", 6.0), ("net_positioning_change_pct_oi_rank", 0.0)):
        g = Forecaster.__new__(Forecaster)
        g.spec = SimpleNamespace(target_type="ranking", target_name=target, entity=lambda eid: {"entity_id": eid})
        g.stat = {"A": StatForecast("last", 5.0, 0.1, 0.2, 10, 5.0, named),
                  "B": StatForecast("last", 7.0, 0.1, 0.2, 10, 7.0, named), "C": None,
                  "D": StatForecast("momentum", 9.0, 0.1, 0.2, 10, 9.0, own)}  # v21.1: sizes the interval only
        g.plan, g.prior = None, {}
        fcs = {e: Forecast(eid=e, median=0.0) for e in "ABCD"}
        for fc in fcs.values():
            g._combine_with_stats(fc)
        g._place_uninformed(fcs)
        assert [fcs[e].median for e in "ABCD"] == [5.0, 7.0, middle, middle] and fcs["D"].stat_q90 == 0.2


def test_v215_an_unexpected_argument_never_costs_the_answer(tmp_path):
    """v21.3 exited 2 with no answer on an argument it did not know."""
    _v2()
    from t4agent.cli import DEFAULTS, _parse

    unit = UNITS / "t4-EXAMPLE-eps-beat"
    out = tmp_path / "answer.json"
    proc = _run_patched(unit, out, argv=["analyze", "--task", str(unit / "task.json"), "--corpus", str(unit / "corpus"),
                                         "--out", str(out), "--seed", "7", "extra"])
    assert proc.returncode == 0 and out.exists(), proc.stderr[-2000:]
    official_check(unit, json.loads(out.read_text()))
    args = _parse(["analyze", "--task"])  # a missing value: the contract's paths, no exit
    assert (args.task, args.corpus, args.out) == (DEFAULTS["task"], DEFAULTS["corpus"], DEFAULTS["out"])


def test_v215_an_announcement_return_without_a_model_reply_sits_around_zero():
    """In the full fallback v21.3 submitted [0, 1] for a post-earnings abnormal return in percent."""
    _v2()
    from t4agent.calibrate import finalize
    from t4agent.predict import Forecast
    from t4agent.task import load_task

    spec = load_task(UNITS / "t4-postearn-20240201-megacap" / "task.json")
    row = finalize(spec, Forecast(eid="AAPL", label="positive_reaction"), None, ent=spec.entity("AAPL"))
    assert row["lo"] < -6 and row["hi"] > 6 and row["lo"] + row["hi"] == 0


def test_v215_earnings_alone_is_not_an_eps_target():
    """'earnings_reaction' (a stock's return) took the EPS-dollar anchor in v21.3; EPS targets keep it, also when no
    plan names EPS (the fallback stage, a failed planner) and the name is underscored."""
    from types import SimpleNamespace

    from t4agent.predict import Forecaster

    f = _forecaster("t4-postearn-20240201-megacap", {"interval_quantity": "one-day abnormal return in %"})
    assert f._eps_units() == (None, None)
    f.plan = None
    assert f._eps_units() == (None, None)
    f = _forecaster("t4-eps-yoy-2023Q2-mixed", {"interval_quantity": "GAAP diluted EPS for the quarter"})
    assert f._eps_units()[0] == "level"
    expected = {"net_earnings_per_share_level": "level", "earnings_growth_pct": "pct_change",
                "quarterly_earnings_yoy_change": "difference", "earnings_beat_miss": "level",
                "earnings_reaction": None, "earnings_announcement_return": None, "bid_to_cover_ratio": None}
    for name, units in expected.items():
        g = SimpleNamespace(plan=None, level_rule=lambda: None,
                            spec=SimpleNamespace(target_name=name, target_type="regression", prompt="", raw={}))
        assert Forecaster._eps_units(g)[0] == units, name


def test_v215_a_long_implication_is_cut_after_a_whole_item():
    _v2()
    from types import SimpleNamespace

    from t4agent.reasons import IMPL_MAX, code_implication

    spec = SimpleNamespace(target_type="classification", target_name="next_estimate_revision_direction")
    preds = [{"entity_id": f"SERIES_{i:02d}_2024-08_20241025", "label": "down" if i % 3 else "up"} for i in range(30)]
    text = code_implication(spec, preds, [p["entity_id"] for p in preds])
    assert len(text) <= IMPL_MAX and text.endswith(".") and not text.rstrip(".").endswith(" for")
    assert text.rstrip(".").split(", ")[-1].split("; ")[-1].split(" for ")[-1] in {p["entity_id"] for p in preds}


# ------------------------------------------------------------------ v21.5 fix round: the verifiers' findings on the build
def test_v215_an_unreadable_time_limit_never_costs_the_answer(tmp_path):
    """The clock is read before any answer is written: a 401-digit [agent].timeout_sec raised OverflowError there and
    the process exited 0 with no answer.json. Any unusable value now gives the default clock."""
    _v2()
    from t4agent.task import card_timeout

    unit = _clone("t4-credit-event-2023", tmp_path / "u")
    card = unit / "card.toml"
    text = card.read_text()
    for bad in ("9" * 401, "inf", "nan", "-300.0", "true", '"300"'):
        card.write_text(re.sub(r"(\[agent\][^\[]*?timeout_sec\s*=\s*)[^\n]+", lambda m: m.group(1) + bad, text, count=1))
        assert card_timeout(unit) is None, bad
    card.write_text(re.sub(r"(\[agent\][^\[]*?timeout_sec\s*=\s*)[^\n]+", lambda m: m.group(1) + "9" * 401, text,
                           count=1))
    _rehash(unit)
    out = tmp_path / "answer.json"
    proc = run_agent(unit, out)
    assert proc.returncode == 0 and "clock: hard stop 470 s, LLM deadline 400 s" in proc.stderr, proc.stderr[-2000:]
    official_check(unit, json.loads(out.read_text()))


def test_v215_a_task_row_claim_stays_under_the_judges_claim_cap():
    """A row with no short non-id field was quoted whole up to 600 characters: a CJK profile came to 405 judge tokens,
    over the 400-token claim cap (malformed, false). The judge tokenizer gives at most about one token a character."""
    _v2()
    from types import SimpleNamespace

    from t4agent.answer import task_claim, task_table

    row = {"entity_id": "A", "name": "Alpha", "profile_zh": "零售药房连锁企业，负债沉重，面临诉讼与流动性压力，" * 15}
    spec = SimpleNamespace(raw={"entities": [{"entity_id": "Z", "name": "Zed", "x": 1}, row]})
    c = task_claim(spec, "A")
    text, ranges = task_table(spec)
    assert len(c["claim"]) <= 300  # at most about 300 judge tokens
    assert c["claim"] == text[c["span_start"]:c["span_end"]] and ranges["A"][0] <= c["span_start"] < c["span_end"]


def test_v215_a_crashing_planner_leaves_the_analysts_their_run(monkeypatch, caplog):
    """run_plan caught only LLMUnavailable / BudgetExhausted / ValueError: anything else from the code around the call
    (an odd plan or unit shape) left run_plan and cost the whole model stage. The analysts now run without a plan."""
    import logging

    from t4agent import predict

    f = _forecaster("t4-credit-event-2023", None)

    def boom(*a, **k):
        raise RuntimeError("injected")

    monkeypatch.setattr(f.client, "chat", boom)
    monkeypatch.setattr(predict, "summarize", boom)
    with caplog.at_level(logging.WARNING, logger="t4agent"):
        assert f.run_plan() is None and f.plan is None
    assert "planner crashed" in caplog.text and "table statistics unavailable" in caplog.text


def test_v215_analyst_replies_in_flight_at_the_deadline_are_kept_for_unsampled_entities():
    """On a 300 s card the analyst deadline (98 s) fell inside one House round trip (18-52 s, sent at about 49 s): the
    first replies came 0.1-2 s late and were dropped, and the unit went out on code forecasts. Calls whose group has an
    entity without a sample are now waited for until the reasons gate; the others are still left."""
    _v2()
    import time
    from collections import defaultdict
    from concurrent.futures import ThreadPoolExecutor
    from types import SimpleNamespace

    f = _forecaster("t4-credit-event-2023", None)
    ids = f.spec.entity_ids
    groups = [ids[:2], ids[2:4]]
    samples = defaultdict(list, {ids[2]: [{"median": 0.0}], ids[3]: [{"median": 0.0}]})
    pool = ThreadPoolExecutor(3)
    try:
        f.client = SimpleNamespace(deadline=time.time() + 3.0, clock_scale=0.01)  # reasons gate 0.25 s before it
        late = pool.submit(lambda: (time.sleep(0.5), {ids[0]: {"median": 1.0}, ids[1]: {"median": 2.0}})[1])
        slow = pool.submit(lambda: (time.sleep(5.0), {ids[2]: {"median": 9.0}})[1])  # its group is sampled already
        pending = {late: [0, 0, 0, 0.0], slow: [0, 1, 0, 0.0]}
        start = time.time()
        f._late_replies(pending, groups, samples)
        assert time.time() - start < 2.0 and late not in pending and slow in pending
        assert samples[ids[0]] == [{"median": 1.0}] and samples[ids[1]] == [{"median": 2.0}] and len(samples[ids[2]]) == 1
        f.client = SimpleNamespace(deadline=time.time(), clock_scale=0.01)  # the reasons gate has passed: no wait
        stuck = pool.submit(time.sleep, 5.0)
        start = time.time()
        f._late_replies({stuck: [0, 0, 0, 0.0]}, [[ids[4]]], samples)
        assert time.time() - start < 0.5
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def test_v215_reasons_are_asked_for_without_a_model_forecast_only_when_a_reply_fits(monkeypatch):
    """B9b: with no model forecast but House answering, v21.3 never asked for model reasons. On a slow route that call
    waited out a certain read timeout (115 s per unit at T=600): it now goes only when the route's typical round trip
    fits before the deadline."""
    _v2()
    import time

    from t4agent import reasons as RS
    from t4agent.answer import build_answer, fit_judge_cap
    from t4agent.cli import _rows
    from t4agent.llm import HouseClient, Reply

    f = _forecaster("t4-eps-growth-2024Q3-banks", None)
    f.ev.build_packs([])
    f.build_anchors()
    fallback = f.aggregate({})
    rows = fit_judge_cap(f.spec, _rows(f.spec, f.corpus, f.ev, fallback, None))
    preds = build_answer(f.spec, rows, "", None, f.corpus)["entity_predictions"]

    class Fake:
        enabled, budget, used, ok, clock_scale = True, 6, 0, 0, 1.0

        def __init__(self, typical):
            self.deadline, self.typical_seconds = time.time() + 300, typical

        @property
        def remaining(self):
            return self.budget - self.used

        def chat(self, messages, **kw):
            self.used += 1
            return Reply(content='{"reasons": []}', reasoning="", finish="stop", completion_tokens=10, seconds=0.1)

    for typical, called in ((None, False), (400.0, False), (30.0, True)):
        client = Fake(typical)
        RS.build_reasons(f.spec, f.corpus, f.ev, f, fallback, preds, client)
        assert (client.used > 0) == called, typical
    # HouseClient keeps the round trip of its successful calls
    for k, v in (("MODEL_ENDPOINT", "http://house.invalid"), ("MODEL_NAME", "m"), ("MODEL_TOKEN", "t")):
        monkeypatch.setenv(k, v)

    class Resp:
        status_code = 200

        def json(self):
            return {"choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}]}

    client = HouseClient(deadline=time.time() + 60)
    monkeypatch.setattr(client._session, "post", lambda *a, **k: Resp())
    assert client.typical_seconds is None
    client.chat([{"role": "user", "content": "x"}], reasoning_budget=None, max_attempts=1)
    assert client.typical_seconds is not None and client.typical_seconds < 5


def test_v215_a_trickling_request_is_given_up_after_its_own_timeouts(monkeypatch):
    """Every call was bounded by the deadline + 15 s only: a hung plan held the main thread until 415 s at T=600 and the
    analysts never ran. A call is now also given up once its connect + read timeouts and 5 s have passed, as a timeout
    (a transport error, so chat() may try again while time is left)."""
    _v2()
    import threading
    import time

    import requests

    from t4agent.llm import HouseClient

    for k, v in (("MODEL_ENDPOINT", "http://house.invalid"), ("MODEL_NAME", "m"), ("MODEL_TOKEN", "t")):
        monkeypatch.setenv(k, v)
    client = HouseClient(deadline=time.time() + 300)
    release = threading.Event()

    def trickle(url, **kw):
        release.wait(60)
        raise requests.ReadTimeout("never")

    monkeypatch.setattr(client._session, "post", trickle)
    start = time.time()
    with pytest.raises(requests.Timeout):
        client._post("http://house.invalid/v1/chat/completions", headers={}, data="{}", timeout=(0.2, 0.3))
    assert 5.0 <= time.time() - start < 7.0
    release.set()


def test_v215_an_emergency_row_keeps_the_code_fallbacks_quantity_logic():
    """The emergency row (a fallback stage that failed) skipped finalize: an announcement return came out [0, 1] and a
    quantity near 289,587 [0, 1]. A classification row's quantity is now sized by finalize from the task alone (no
    corpus needed); a numeric row keeps the naive value with the default half-width."""
    _v2()
    from t4agent.cli import emergency_row
    from t4agent.task import load_task

    spec = load_task(UNITS / "t4-postearn-20240201-megacap" / "task.json")
    row = emergency_row(spec, spec.entity_ids[0])
    assert row["lo"] < -6 and row["hi"] > 6 and row["claims"] == [] and row["label"] in spec.labels
    spec = load_task(UNITS / "t4-macrorev-20240930-us6" / "task.json")
    row = emergency_row(spec, spec.entity_ids[0])
    assert row["hi"] > 1000 and row["lo"] <= row.get("point_forecast", row["lo"]) <= row["hi"]
    spec = load_task(UNITS / "t4-cpicomp-202410-us11" / "task.json")
    row = emergency_row(spec, "CPI_APPAREL")  # the latest print 1.14, +/-1 for a quantity named without a unit
    assert abs(row["point_forecast"] - 1.14) < 1e-9 and abs(row["hi"] - row["lo"] - 2.0) < 1e-9


def test_v215_a_mechanism_never_runs_past_its_room():
    """With a long lead the room left for a fact went negative, and str.rfind read the negative end from the string's
    end: a whole fact came back and clean() cut the mechanism mid-sentence."""
    _v2()
    from types import SimpleNamespace

    from t4agent.reasons import _plain_fact

    corpus = SimpleNamespace(docs={}, excluded={})
    fact = "series 'x' (26 obs, last 1.09); the level is stable; a second clause here; and a third one."
    assert _plain_fact(fact, corpus, 400) == fact
    assert all(_plain_fact(fact, corpus, lim) is None for lim in (0, -5, -60))


def test_v215_ranking_baselines_order_rows_only_when_they_come_from_one_rule():
    """Baselines whose method was picked entity by entity order the rows by method, not by evidence (v17: switching
    among the methods lost; momentum 0.479 against the constant's 0.500): such a mix keeps v21.3's constant. One method
    throughout, or the ratio-change reversion rule (the CFTC replay's 0.565), still ranks."""
    _v2()
    from types import SimpleNamespace

    from t4agent.predict import Forecast, Forecaster
    from t4agent.stats import Series, StatForecast

    named = Series("d", "value", [4.0] * 12, [f"2017-{i:02d}" for i in range(12)], 30.0, False)
    ratio = Series("d", "value", [4.0] * 12, [f"2017-{i:02d}" for i in range(12)], 30.0, False, denom=[10.0] * 12)
    for series, methods, ranked in ((named, ("last", "momentum"), False), (named, ("drift", "drift"), True),
                                    (ratio, ("level_revert", "drift"), True)):
        g = Forecaster.__new__(Forecaster)
        g.spec = SimpleNamespace(target_type="ranking", target_name="net_long_contracts_level_rank",
                                 entity=lambda eid: {"entity_id": eid})
        g.stat = {"A": StatForecast(methods[0], 5.0, 0.1, 0.2, 10, 5.0, series),
                  "B": StatForecast(methods[1], 7.0, 0.1, 0.2, 10, 7.0, series), "C": None}
        g.plan, g.prior = None, {}
        fcs = {e: Forecast(eid=e, median=0.0) for e in "ABC"}
        g._rank_by_baseline = g._baselines_rank(fcs)
        for fc in fcs.values():
            g._combine_with_stats(fc)
        g._place_uninformed(fcs)
        got = [fcs[e].median for e in "ABC"]
        assert got == ([5.0, 7.0, 6.0] if ranked else [0.0, 0.0, 0.0]), (methods, got)
        assert fcs["A"].stat_q90 == 0.2  # the baseline still sizes the interval


def test_v215_a_long_separator_run_after_an_eps_label_reads_in_linear_time():
    """B14: the sub-label's two whitespace runs overlap, and 16,000 spaces after an EPS label took 1.1 s to reject."""
    _v2()
    import datetime
    import time

    from t4agent.corpus import Doc
    from t4agent.facts import eps_record

    doc = Doc("EDGAR_X_10Q", datetime.date(2023, 8, 1), "Earnings per share: Diluted" + " " * 16000 + "x",
              {"form_type": "10-Q", "period_of_report": "2023-06-30"})
    start = time.perf_counter()
    assert eps_record("EDGAR_X_10Q", doc) is None
    assert time.perf_counter() - start < 0.1


def test_v215_the_seed_is_a_constant_and_the_analysts_have_a_window_on_a_300_s_card(caplog):
    """B12: QFBENCH_SEED no longer moves the seed. B7: on a 300 s card (LLM deadline 140 s) the analysts still get a
    window after a 30 s plan; with v21.3's absolute margins (120 + 25 s) there was none."""
    _v2()
    import logging
    import time

    from t4agent import config
    from t4agent.llm import HouseClient

    env = {k: v for k, v in os.environ.items() if not k.startswith(("MODEL_", "T4_"))}
    env["QFBENCH_SEED"] = "5"
    proc = subprocess.run([sys.executable, "-c", "from t4agent import config; print(config.SEED)"], cwd=AGENT, env=env,
                          capture_output=True, text=True, timeout=60)
    assert proc.stdout.strip() == "0", proc.stderr
    f = _forecaster("t4-credit-event-2023", None)
    hard, llm, scale = config.derive_clock(300.0)
    f.client = HouseClient(deadline=time.time() + llm - 30, clock_scale=scale)
    with caplog.at_level(logging.INFO, logger="t4agent"):
        f.choose_layout()
    left = float(re.search(r"tok/s, (-?\d+) s left\)", caplog.text).group(1))
    assert 50 < left < llm - 30 - 8.75 - 0.35 * config.REASONS_RESERVE_SEC + 1


def _eligible_policy():
    """Explicit mature policy for legacy backend math tests; cutoff tests live separately."""
    import datetime as dt
    from t4agent.artifact_policy import ArtifactPolicy
    return ArtifactPolicy(dt.date(2022, 2, 1))


# ------------------------------------------------------------------ v21.7: Python's hash seed in the image
def test_v217_the_image_pins_pythons_hash_seed():
    """Set iteration order, and so the prompts, is the same on every run; the House seed itself is a constant (v21.5)."""
    text = (AGENT / "Dockerfile").read_text()
    assert "PYTHONHASHSEED=0" in text and "VOLUME" not in text


# ------------------------------------------------------------------ v21.8: a second rounding tier for the answer cap
def test_v218_a_second_tier_fits_a_roster_the_first_cannot():
    """Over the 3,000-byte cap a unit's reasons score 0 for certain. When v14's 1% / 0.5% cannot fit, 5% / 2.5% may:
    the interval only widens (by at most 5% of its width) and the point moves by at most 2.5% of it, inside the interval."""
    _v2()
    from types import SimpleNamespace

    from t4agent.answer import (CAP_SHIFT, CAP_SHIFT_MORE, CAP_WIDEN, CAP_WIDEN_MORE, JUDGE_ANSWER_CAP, _judged_bytes,
                                _rounded_to_fit, fit_judge_cap)

    ids = [f"STATE_{i:02d}" for i in range(32)]
    spec = SimpleNamespace(entity_ids=ids, target_type="regression", interval_level=0.9)
    rows = {e: {"point_forecast": 3.14159265 + 0.0123 * i, "lo": 2.71828182 + 0.0123 * i, "hi": 3.56789012 + 0.0123 * i}
            for i, e in enumerate(ids)}
    assert _judged_bytes(spec, rows) > 3000
    assert _rounded_to_fit(spec, rows, JUDGE_ANSWER_CAP, CAP_WIDEN, CAP_SHIFT, (9, 8, 7, 6, 5, 4)) is None
    fit = fit_judge_cap(spec, rows)
    assert _judged_bytes(spec, fit) <= 3000
    for e in ids:
        w = rows[e]["hi"] - rows[e]["lo"]
        assert fit[e]["lo"] <= rows[e]["lo"] and fit[e]["hi"] >= rows[e]["hi"]
        assert (fit[e]["hi"] - fit[e]["lo"]) - w <= CAP_WIDEN_MORE * w + 1e-12
        assert abs(fit[e]["point_forecast"] - rows[e]["point_forecast"]) <= CAP_SHIFT_MORE * w + 1e-12
        assert fit[e]["lo"] <= fit[e]["point_forecast"] <= fit[e]["hi"]


def test_v218_a_ranking_is_never_reordered_and_a_long_roster_is_left_alone():
    """A ranking is rounded only when its points keep their order and ties (v13 merged 26 points into one); a roster
    whose field names alone fill the cap cannot fit at any precision and is returned as it is."""
    _v2()
    from types import SimpleNamespace

    from t4agent.answer import _judged_bytes, fit_judge_cap

    ids = [f"FUTURES_MKT_{i:02d}" for i in range(26)]
    rank = SimpleNamespace(entity_ids=ids, target_type="ranking", interval_level=0.9)
    close = {e: {"point_forecast": 158688.6881 + 0.0001 * i, "lo": 158600.1234 + 0.0001 * i,
                 "hi": 158777.4561 + 0.0001 * i} for i, e in enumerate(ids)}
    assert _judged_bytes(rank, close) > 3000 and fit_judge_cap(rank, close) is close  # any rounding would merge them
    ids = [f"ST{i:02d}" for i in range(51)]  # a state-level roster: about 80 bytes a row before any number
    spec = SimpleNamespace(entity_ids=ids, target_type="regression", interval_level=0.9)
    rows = {e: {"point_forecast": -0.25 + 0.01 * i, "lo": -0.6046 + 0.01 * i, "hi": 0.1046 + 0.01 * i}
            for i, e in enumerate(ids)}
    floor = {e: {"point_forecast": 1, "lo": 0, "hi": 2} for e in ids}
    assert _judged_bytes(spec, floor) > 3000 and fit_judge_cap(spec, rows) is rows



# ------------------------------------------------------------------ v21.10: [0, 1] signals decide classification units only
def test_v2110_a_regression_or_ranking_takes_0_1_only_from_the_planner():
    """A row column that happens to read 0 (or 0 and 1), or a 0-1 schema hint, once sent a regression out at 0 in [0, 1]
    with the planner saying it was no probability. Those signals now decide classification units only; a regression or
    ranking unit takes [0, 1] from the planner flag alone, as in v21.8."""
    _v2()
    from t4agent.calibrate import finalize, is_probability
    from t4agent.predict import Forecast

    zeros = [{"entity_id": "X", "value": 0}, {"entity_id": "Y", "value": 0}]
    reg = _bare_spec(target_type="regression", target_name="yield_change_bps", labels=[],
                     prompt="Predict the change in the 2-year yield in basis points.", entities=zeros, entity_ids=["X", "Y"])
    fc = Forecast("X", median=-20.0, p05=-35.0, p95=-5.0)
    for plan in (None, {"interval_is_probability": False}):
        assert is_probability(plan, reg) is False
        out = finalize(reg, fc, plan, ent=reg.entity("X"))
        assert out["point_forecast"] == -20.0 and out["lo"] <= -35.0 and out["hi"] >= -5.0
    # v21.11 (T4-002): the planner flag no longer turns a regression into [0, 1] either
    assert is_probability({"interval_is_probability": True}, reg) is False
    out = finalize(reg, fc, {"interval_is_probability": True}, ent=reg.entity("X"))
    assert out["point_forecast"] == -20.0 and out["lo"] <= -35.0 and out["hi"] >= -5.0

    mixed = [{"entity_id": "X", "value": 1}, {"entity_id": "Y", "value": 0}]
    bounded = {"target": {"value": {"type": "number", "minimum": 0, "maximum": 1}}}
    for ttype in ("regression", "ranking"):
        spec = _bare_spec(raw=bounded, target_type=ttype, labels=[], card_params={"interval_leg": False},
                          prompt="Give a probability interval.", entities=mixed, entity_ids=["X", "Y"])
        assert is_probability(None, spec) is False
    # the same signals still decide a classification unit
    cls = _bare_spec(prompt="Did the event resolve?", entities=mixed, entity_ids=["X", "Y"])
    assert is_probability(None, cls) is True
    assert is_probability(None, _bare_spec(raw=bounded, entities=[{"entity_id": "X"}], entity_ids=["X"])) is True


# ------------------------------------------------------------------ v21.11: T4-002, the planner flag never clamps a number
def test_v2111_planner_probability_flag_never_clamps_a_numeric_regression_or_ranking():
    """T4-002: a planner that calls a bid-to-cover ratio or a ranking score a probability once sent 2.5 out as 1.0 in
    [0, 1] and tied every ranking score at 0 or 1. The flag now only trims a range whose forecast already lies in [0, 1]."""
    _v2()
    from t4agent.calibrate import finalize, is_probability
    from t4agent.predict import Forecast

    flag = {"interval_is_probability": True}
    btc = _bare_spec(target_type="regression", target_name="bid_to_cover_ratio", labels=[],
                     prompt="Predict the bid-to-cover ratio and the probability it exceeds 2.5.",
                     entities=[{"entity_id": "A", "unit": "ratio"}], entity_ids=["A"])
    out = finalize(btc, Forecast("A", median=2.54, p05=2.3, p95=2.8), flag, ent=btc.entity("A"))
    assert abs(out["point_forecast"] - 2.54) <= 0.05 and out["lo"] < out["point_forecast"] < out["hi"] and out["hi"] > 1.0
    ids = ["A", "B", "C", "D"]
    rank = _bare_spec(target_type="ranking", target_name="net_positioning_change_pct_oi", labels=[],
                      prompt="Rank the markets.", entities=[{"entity_id": e} for e in ids], entity_ids=ids)
    pts = [finalize(rank, Forecast(e, median=m, p05=m - 3, p95=m + 3), flag, ent=rank.entity(e))["point_forecast"]
           for e, m in zip(ids, (-4.2, -0.3, 0.6, 7.5))]
    assert pts == sorted(pts) and len(set(pts)) == 4 and pts[0] < 0 and pts[-1] > 1
    # a genuine probability regression (no measured unit anywhere) keeps the v21.8 [0, 1] path ...
    preg = _bare_spec(target_type="regression", target_name="p_default", labels=[], prompt="Probability of default.",
                      entities=[{"entity_id": "X"}, {"entity_id": "Y"}], entity_ids=["X", "Y"])
    assert is_probability(flag, preg) is True
    out = finalize(preg, Forecast("X", median=0.12, p05=0.02, p95=0.3), flag, ent=preg.entity("X"))
    assert (out["lo"], out["hi"]) == (0.0, 1.0) and abs(out["point_forecast"] - 0.12) <= 0.05
    # ... but not for a row whose forecast lies outside [0, 1]: that is no probability, it stays numeric
    out = finalize(preg, Forecast("Y", median=3.4, p05=2.0, p95=5.0), flag, ent=preg.entity("Y"))
    assert out["point_forecast"] > 3 and out["hi"] > 1.0
    # and a ranking never takes [0, 1]
    prank = _bare_spec(target_type="ranking", target_name="p_default", labels=[], prompt="Probability of default.",
                       entities=[{"entity_id": "X"}], entity_ids=["X"])
    assert is_probability(flag, prank) is False


def test_v2111_a_classification_row_outside_0_1_keeps_its_numeric_interval_against_a_planner_flag():
    """T4-002 on a classification unit with an interval leg: the planner's flag alone once sent consensus-EPS rows out
    as [0, 1] although the scored truth is the EPS in dollars. A row whose quantity lies outside [0, 1] stays numeric;
    `interval_leg = false` (credit-event) and a schema probability keep [0, 1]."""
    _v2()
    from t4agent.calibrate import finalize, is_probability
    from t4agent.predict import Forecast

    flag = {"interval_is_probability": True}
    rows = [{"entity_id": "AAPL", "consensus_eps": 2.10}, {"entity_id": "SMALL", "consensus_eps": 0.4}]
    eps = _bare_spec(target_name="eps_outcome", labels=["beat", "miss", "inline"],
                     prompt="Label beat/miss/inline and give the probability of a beat.", entities=rows,
                     entity_ids=["AAPL", "SMALL"])
    assert is_probability(flag, eps) is True and is_probability(None, eps) is False
    out = finalize(eps, Forecast("AAPL", label="beat", median=2.2, p05=2.0, p95=2.4), flag, ent=eps.entity("AAPL"))
    assert out["lo"] > 1.0 and out["hi"] > 2.2 and out["label"] == "beat"
    out = finalize(eps, Forecast("SMALL", label="beat", median=0.42, p05=0.35, p95=0.5), flag, ent=eps.entity("SMALL"))
    assert (out["lo"], out["hi"]) == (0.0, 1.0)  # inside [0, 1]: the planner's reading stands, as in v21.10
    leg = _bare_spec(target_name="eps_outcome", labels=["beat", "miss", "inline"], card_params={"interval_leg": False},
                     prompt="Label beat/miss/inline.", entities=rows, entity_ids=["AAPL", "SMALL"])
    out = finalize(leg, Forecast("AAPL", label="beat", median=2.2, p05=2.0, p95=2.4), flag, ent=leg.entity("AAPL"))
    assert (out["lo"], out["hi"]) == (0.0, 1.0)


def test_v2111_a_canary_planted_in_every_document_never_reaches_the_answer(tmp_path):
    """README: in the Final, a canary string anywhere in the output is scored as contamination. Plant the card's canary
    GUID at the head of every corpus document (where claims and premises are quoted from): the answer must carry it
    nowhere, and must stay admissible."""
    import tomllib
    unit = _clone("t4-fomc-curve-20240918", tmp_path / "u")
    guid = tomllib.loads((unit / "card.toml").read_text())["contamination"]["canary_guid"]
    for f in (unit / "corpus").glob("*.json"):
        if f.name == "manifest.json":
            continue
        d = json.loads(f.read_text())
        d["text"] = f"Canary {guid}. " + d["text"].replace(". ", f". {guid} ", 3)
        f.write_text(json.dumps(d))
    _rehash(unit)
    out = tmp_path / "answer.json"
    assert run_agent(unit, out).returncode == 0
    text = out.read_text()
    assert guid not in text.lower() and guid.upper() not in text
    answer = json.loads(text)
    official_check(unit, answer)
    assert all(row["claims"] for row in answer["entity_predictions"])


def test_v2111_an_announcement_return_row_keeps_its_numeric_interval_against_a_planner_flag():
    """T4-002: a post-earnings reaction (a one-day abnormal return in percent) is no probability; a planner flag alone
    once sent it out as [0, 1], which cannot cover a negative return."""
    _v2()
    from t4agent.calibrate import finalize
    from t4agent.predict import Forecast

    rows = [{"entity_id": "AAPL", "flat_threshold_abn_pct": 1.0, "pre_earnings_close_2024_01_31": 184.4}]
    spec = _bare_spec(target_name="earnings_reaction", labels=["positive_reaction", "negative_reaction", "flat"],
                      prompt="Classify the one-day abnormal return (in percent) after the report.",
                      raw={"target": {"label_assertions": {"flat": "one-day abnormal return within +/-1%"}}},
                      entities=rows, entity_ids=["AAPL"])
    out = finalize(spec, Forecast("AAPL", label="flat", median=0.4, p05=0.1, p95=0.7),
                   {"interval_is_probability": True}, ent=spec.entity("AAPL"))
    assert out["lo"] < 0.0 < out["hi"] and (out["lo"], out["hi"]) != (0.0, 1.0)
