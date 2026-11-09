"""v21.11 harness items H1-H5 (G1 F3/F4/F5/F7 + clock cap) and two contract tests for the F1/F2 fix.

Place in T4/agent_v21_11/tests/ and run with the suite:  ../.venv/bin/python -m pytest -q -p no:cacheprovider tests
Outside the suite:  T4_AGENT=<agent dir> python -m pytest -q -p no:cacheprovider test_v2111_harness.py

No test reads an outcome. Every H test fails on the bundled v21.11 (commit c83a768 tree) and passes after the fix; the
two contract tests pass on that v21.11 and must keep passing after F1/F2 land.
"""
from __future__ import annotations

import copy
import importlib
import os
import sys
from pathlib import Path

import pytest

AGENT = Path(os.environ.get("T4_AGENT") or Path(__file__).resolve().parents[1]).resolve()
sys.path.insert(0, str(AGENT))

from t4agent.answer import build_answer, has_canary  # noqa: E402
from t4agent.calibrate import finalize, is_probability, numeric_target_plan  # noqa: E402
from t4agent.predict import Forecast  # noqa: E402
from t4agent.task import TaskSpec, card_timeout  # noqa: E402

FLAG = {"interval_is_probability": True}
FLAG_Q = {"interval_is_probability": True, "interval_quantity": "probability 0-1"}
GUID = "11111111-2222-4333-8444-555555555555"


def _spec(**kw) -> TaskSpec:
    base = dict(raw={}, task_id="t", target_type="regression", target_type_trusted=True, target_name="p_default",
                labels=[], prompt="", cutoff=None, cutoff_str="", resolution_str="", interval_level=0.9,
                entities=[], entity_ids=[], unit_dir=None, card_params={})
    base.update(kw)
    if base["entities"] and not base["entity_ids"]:
        base["entity_ids"] = [e["entity_id"] for e in base["entities"]]
    return TaskSpec(**base)


def _is01(out: dict) -> bool:
    return (out["lo"], out["hi"]) == (0.0, 1.0)


# ------------------------------------------------------------------ H1 (G1 F3): percent forms are measured units
@pytest.mark.parametrize("unit,name", [("%", "p_default"), ("％", "p_default"), ("pp", "p_default"),
                                       ("per cent", "p_default"), ("per-cent", "p_default"), (None, "cpi_%"),
                                       (None, "move_%")])
def test_v2111_h1_percent_forms_are_measured_units_on_the_flag_path(unit, name):
    rows = [{"entity_id": "A", "latest_value": 0.4}, {"entity_id": "B", "latest_value": 0.3}]
    if unit:
        for r in rows:
            r["unit"] = unit
    s = _spec(target_name=name, entities=rows)
    assert is_probability(FLAG, s) is False  # v21.11: True, so [0, 1]
    out = finalize(s, Forecast("A", median=0.4, p05=0.3, p95=0.5), FLAG, ent=s.entity("A"))
    assert not _is01(out) and abs(out["point_forecast"] - 0.4) <= 0.05
    plan = copy.deepcopy(FLAG_Q)
    assert numeric_target_plan(plan, s) is True  # the analysts are asked for the target, not a probability
    assert plan["interval_is_probability"] is False and "interval_quantity" not in plan


def test_v2111_h1_planner_normal_classification_is_decided_as_before():
    """I1: the percent forms enter the planner-flag path only. A classification unit with a 0/1 `value` column and a
    '%' unit, and no planner flag, stays [0, 1] as in v21.11."""
    rows = [{"entity_id": "X", "value": 0, "unit": "%"}, {"entity_id": "Y", "value": 1, "unit": "%"}]
    s = _spec(target_type="classification", target_name="event", labels=["yes", "no"], entities=rows)
    assert is_probability(None, s) is True
    assert _is01(finalize(s, Forecast("X", label="yes"), None, ent=s.entity("X")))


# ------------------------------------------------------------------ H2 (G1 F4): measured unit strings that differ
def test_v2111_h2_heterogeneous_measured_units_count_as_measured():
    rows = [{"entity_id": "A", "unit": "bps", "latest_value": 0.4},
            {"entity_id": "B", "unit": "basis points", "latest_value": 0.2}]
    s = _spec(entities=rows)
    assert is_probability(FLAG, s) is False  # v21.11: True, so [0, 1]
    assert not _is01(finalize(s, Forecast("A", median=0.4, p05=0.3, p95=0.5), FLAG, ent=s.entity("A")))


def test_v2111_h2_one_unmeasured_unit_string_keeps_the_planner_reading():
    rows = [{"entity_id": "A", "unit": "bps", "latest_value": 0.4},
            {"entity_id": "B", "unit": "probability", "latest_value": 0.2}]
    s = _spec(entities=rows)
    assert is_probability(FLAG, s) is True
    assert _is01(finalize(s, Forecast("A", median=0.4), FLAG, ent=s.entity("A")))


# ------------------------------------------------------------------ H3 (G1 F5): a probability column is a whole token
def test_v2111_h3_problem_count_does_not_block_the_numeric_rescue():
    s = _spec(target_type="classification", target_name="eps_outcome", labels=["beat", "miss", "inline"],
              entities=[{"entity_id": "X", "consensus_eps": 2.5, "problem_count": 3}])
    out = finalize(s, Forecast("X", label="beat", median=2.2, p05=2.0, p95=2.4), FLAG, ent=s.entity("X"))
    assert not _is01(out) and out["lo"] < 2.2 < out["hi"]  # v21.11: {'label': 'beat', 'lo': 0.0, 'hi': 1.0}


@pytest.mark.parametrize("col", ["implied_prob", "impliedProb", "consensus_probability", "beat_odds",
                                 "default_likelihood", "probs_beat", "chance_of_cut"])
def test_v2111_h3_real_probability_columns_keep_the_planner_reading(col):
    s = _spec(target_type="classification", target_name="eps_outcome", labels=["beat", "miss", "inline"],
              entities=[{"entity_id": "X", "consensus_eps": 2.5, col: 0.6}])
    assert _is01(finalize(s, Forecast("X", label="beat", median=0.6), FLAG, ent=s.entity("X")))


# ------------------------------------------------------------------ H4 (G1 F7): a label never carries the canary
def test_v2111_h4_a_label_that_is_the_canary_is_not_copied():
    row = {"entity_id": "X", "latest_value": 2.0}
    s = _spec(target_type="classification", target_name="event", labels=["yes", "no"], raw={"entities": [row]},
              entities=[row], canaries=(GUID,))
    rows = {"X": {"label": GUID, "lo": 0.0, "hi": 1.0,
                  "claims": [{"doc_id": "task", "span_start": 1, "span_end": 20, "claim": "\"latest_value\": 2.0"}]}}
    assert not has_canary(s, build_answer(s, rows))


# ------------------------------------------------------------------ H5: a production clock cap only lowers the clock
def _config(monkeypatch, **env):
    for k in ("T4_TIME_BUDGET", "T4_LLM_DEADLINE", "T4_CAP_TIME_BUDGET", "T4_CAP_LLM_DEADLINE"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    import t4agent.config as config
    return importlib.reload(config)


def test_v2111_h5_the_cap_lowers_the_clock_and_never_lifts_it_past_the_card(monkeypatch):
    config = _config(monkeypatch, T4_CAP_TIME_BUDGET="360", T4_CAP_LLM_DEADLINE="290")
    try:
        assert config.derive_clock(600.0) == (360.0, 290.0, 0.725)
        assert config.derive_clock(None) == (360.0, 290.0, 0.725)
        assert config.derive_clock(450.0) == (360.0, 290.0, 0.725)
        assert config.derive_clock(300.0) == (210.0, 140.0, 0.35)  # the card's own limit still wins
    finally:
        _config(monkeypatch)


def test_v2111_h5_without_a_cap_the_clock_is_v21_11s(monkeypatch):
    config = _config(monkeypatch)
    assert config.derive_clock(600.0) == (470.0, 400.0, 1.0) == config.derive_clock(None)
    assert config.derive_clock(300.0) == (210.0, 140.0, 0.35)


def test_v2111_h5_the_local_testing_override_is_unchanged(monkeypatch):
    """tools/run_units.py lifts the clock with T4_TIME_BUDGET for slow local runs; that keeps working."""
    config = _config(monkeypatch, T4_TIME_BUDGET="1000", T4_LLM_DEADLINE="900")
    try:
        assert config.derive_clock(600.0) == (1000.0, 900.0, 1.0)
    finally:
        _config(monkeypatch)


def test_v2111_h5_the_image_never_sets_the_local_testing_clock():
    """The Dockerfile may set only the T4_CAP_* clock: T4_TIME_BUDGET there would lift a short card's clock past it."""
    text = (AGENT / "Dockerfile").read_text(encoding="utf-8")
    assert "T4_TIME_BUDGET" not in text and "T4_LLM_DEADLINE" not in text


def _image_env() -> dict[str, str]:
    """The KEY=value pairs of the Dockerfile's ENV instructions: the environment the image runs the agent with."""
    text = (AGENT / "Dockerfile").read_text(encoding="utf-8").replace("\\\n", " ")
    env: dict[str, str] = {}
    for line in text.splitlines():
        words = line.split()
        if words[:1] == ["ENV"]:
            env.update(w.split("=", 1) for w in words[1:])
    return env


def test_v2111_h5_the_image_caps_a_600_s_card_at_360_290(monkeypatch, tmp_path):
    """The Final image sets only the caps, 360 / 290: a card with timeout_sec = 600 gets a 360 s hard stop and a 290 s
    LLM deadline. Without them the same card keeps the 470 / 400 s clock."""
    caps = {k: v for k, v in _image_env().items() if k.startswith("T4_")}
    assert caps == {"T4_CAP_TIME_BUDGET": "360", "T4_CAP_LLM_DEADLINE": "290"}
    (tmp_path / "card.toml").write_text("[agent]\ntimeout_sec = 600\n", encoding="utf-8")
    assert card_timeout(tmp_path) == 600.0
    config = _config(monkeypatch, **caps)
    try:
        assert config.derive_clock(card_timeout(tmp_path)) == (360.0, 290.0, 0.725)
    finally:
        config = _config(monkeypatch)
    assert config.derive_clock(card_timeout(tmp_path)) == (470.0, 400.0, 1.0)


# ------------------------------------------------------------------ contract for the F1/F2 fix (other executor)
def test_v2111_contract_c1_an_unnamed_numeric_regression_is_asked_for_in_its_own_units():
    """T4-002 core case: no measured unit named anywhere, every row's naive value outside [0, 1]. v21.11 drops the
    planner's flag before the analysts are asked, so they answer in the target's units. An F2 fix that leaves the flag
    on here (deleting the row loop in _planner_probability outright) sends 0.6 out as a bid-to-cover of 2.4."""
    rows = [{"entity_id": "A", "latest_value": 2.4}, {"entity_id": "B", "latest_value": 2.6}]
    s = _spec(target_name="auction_btc", entities=rows)
    plan = copy.deepcopy(FLAG_Q)
    assert numeric_target_plan(plan, s) is True
    assert plan["interval_is_probability"] is False and "interval_quantity" not in plan


def test_v2111_contract_c2_a_row_never_submits_a_probability_scale_answer_as_its_value():
    """F2's mixed case: one row 0.2, one row 1.5. Whatever F2 decides for the unit, the analysts answer in the unit the
    plan asks for; row B (naive value 1.5) must never go out with that probability-scale answer as its value."""
    rows = [{"entity_id": "A", "latest_value": 0.2}, {"entity_id": "B", "latest_value": 1.5}]
    s = _spec(entities=rows)
    plan = copy.deepcopy(FLAG_Q)
    numeric_target_plan(plan, s)
    asked_prob = plan.get("interval_is_probability") is True
    med = 0.3 if asked_prob else 1.6
    out = finalize(s, Forecast("B", median=med, p05=med - 0.1, p95=med + 0.1), plan, ent=s.entity("B"))
    if not _is01(out):
        assert not (asked_prob and abs(out["point_forecast"] - med) < 1e-9), out


@pytest.mark.parametrize("col,vals", [("latest_vintage_value", (159000.0, 2100000.0)),
                                      ("latest_rate_at_date", (4.35, 5.10)),
                                      ("latest_value", (2024.0, 2101.5))])
def test_v2111_contract_c3_a_level_column_whose_name_mentions_a_date_is_still_a_level(col, vals):
    """F1 must treat a calendar FIELD (a year such as latest_year = 2024) as no naive value, not every column whose
    name mentions year/date/vintage: a vintage level (payrolls 159,000) or an index at 2024 is a level, and on an
    unnamed numeric regression it is what drops the planner's flag (C1)."""
    rows = [{"entity_id": "A", col: vals[0]}, {"entity_id": "B", col: vals[1]}]
    s = _spec(target_name="next_estimate", entities=rows)
    plan = copy.deepcopy(FLAG_Q)
    assert numeric_target_plan(plan, s) is True and plan["interval_is_probability"] is False
