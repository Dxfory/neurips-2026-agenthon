"""Adjacent cases for the v21.11 G1 re-review of commit 6739240.

Passing tests lock fixes that hold. Failing tests are counterexamples the
re-review found: a date-like level column outside the calendar regex, a
substring exemption that hides an out-of-range baseline, a hyphenated
'per-cent', the fallback median copied from latest_year, a classification
row whose level field is latest_year, and a canary copied through entity_id
or task_id.
"""
from __future__ import annotations

import json

import pytest

from t4agent.answer import build_answer, validate
from t4agent.calibrate import finalize, is_probability, numeric_target_plan
from t4agent.predict import Forecast, baseline_value, fallback_forecast
from t4agent.task import TaskSpec

GUID = "11111111-2222-4333-8444-555555555555"


def _spec(rows, *, tt="regression", name="p_default", leg=True, raw=None, canaries=(), task_id="t", labels=None):
    if labels is None:
        labels = ["yes", "no"] if tt == "classification" else []
    return TaskSpec(
        raw=raw if raw is not None else {},
        task_id=task_id,
        target_type=tt,
        target_type_trusted=True,
        target_name=name,
        labels=labels,
        prompt="",
        cutoff=None,
        cutoff_str="",
        resolution_str="",
        interval_level=0.9,
        entities=rows,
        entity_ids=[r["entity_id"] for r in rows],
        unit_dir=None,
        card_params={"interval_leg": leg},
        canaries=canaries,
    )


def _reg(rows, name="p_default", plan=None, med=0.2):
    plan = {"interval_is_probability": True} if plan is None else plan
    spec = _spec(rows, name=name)
    ent = spec.entity(rows[0]["entity_id"])
    out = finalize(spec, Forecast(rows[0]["entity_id"], median=med), plan, ent=ent)
    return spec, out, is_probability(plan, spec)


def test_latest_year_and_vintage_keep_a_median_inside_unit_interval():
    for key in ("latest_year", "latest_vintage"):
        spec, out, ip = _reg([{"entity_id": "X", key: 2024}], med=0.2)
        assert ip is True
        assert baseline_value(spec, spec.entity("X"), {"interval_is_probability": True}) == 2024
        assert (out["lo"], out["hi"]) == (0.0, 1.0)
        assert out["point_forecast"] == pytest.approx(0.2)


@pytest.mark.xfail(
    strict=True,
    reason="C2 supersedes the old expectation: a mixed row cannot submit the analyst's probability-scale median",
)
def test_mixed_rows_do_not_flip_an_in_range_sibling():
    rows = [
        {"entity_id": "A", "latest_value": 0.2},
        {"entity_id": "B", "latest_value": 1.5},
    ]
    spec = _spec(rows)
    plan = {"interval_is_probability": True}
    assert is_probability(plan, spec) is True
    a = finalize(spec, Forecast("A", median=0.2), plan, ent=spec.entity("A"))
    b = finalize(spec, Forecast("B", median=0.4), plan, ent=spec.entity("B"))
    assert (a["lo"], a["hi"]) == (0.0, 1.0) and a["point_forecast"] == pytest.approx(0.2)
    assert (b["lo"], b["hi"]) != (0.0, 1.0)
    assert b["point_forecast"] == pytest.approx(0.4)


@pytest.mark.xfail(
    strict=True,
    reason="C1 supersedes the old unit boolean: an all-rows-outside unnamed regression drops the flag pre-analyst",
)
def test_a_real_baseline_is_numeric_when_the_year_value_differs():
    """latest_value 2.5 is the baseline; latest_year 2024 must not exempt it."""
    spec, out, ip = _reg(
        [{"entity_id": "X", "latest_value": 2.5, "latest_year": 2024}], med=0.4
    )
    assert ip is True  # no measured unit on the task; the row check is inside finalize
    assert baseline_value(spec, spec.entity("X"), {"interval_is_probability": True}) == 2.5
    assert (out["lo"], out["hi"]) != (0.0, 1.0)


def test_percent_spellings_on_a_true_flag_leave_the_unit_interval():
    plan = {"interval_is_probability": True, "interval_quantity": "probability 0-1"}
    samples = [
        ("p_default", [{"entity_id": "X", "unit": "%", "latest_value": 0.4}]),
        ("p_default", [{"entity_id": "X", "unit": "％", "latest_value": 0.4}]),
        ("p_default", [{"entity_id": "X", "unit": "pp", "latest_value": 0.4}]),
        ("p_default", [{"entity_id": "X", "unit": "per cent", "latest_value": 0.4}]),
        ("p_default", [{"entity_id": "X", "unit": "PER CENT", "latest_value": 0.4}]),
        ("p_default", [{"entity_id": "X", "unit": "per  cent", "latest_value": 0.4}]),
        ("cpi_%", [{"entity_id": "X", "latest_value": 0.4}]),
        ("move_%", [{"entity_id": "X", "latest_value": 0.4}]),
        ("Per_Cent", [{"entity_id": "X", "latest_value": 0.4}]),
    ]
    for name, rows in samples:
        spec = _spec(rows, name=name)
        snap = dict(plan)
        dropped = numeric_target_plan(snap, spec)
        out = finalize(spec, Forecast("X", median=0.4), plan, ent=spec.entity("X"))
        assert dropped is True, name
        assert is_probability(plan, spec) is False, name
        assert (out["lo"], out["hi"]) != (0.0, 1.0), (name, out)


def test_pp_token_does_not_eat_ordinary_words():
    plan = {"interval_is_probability": True}
    for name in ("supply", "happy", "opportunity", "appendix", "apple", "PPP", "support"):
        spec, out, ip = _reg([{"entity_id": "X", "latest_value": 0.4}], name=name, plan=plan, med=0.4)
        assert ip is True, name
        assert (out["lo"], out["hi"]) == (0.0, 1.0), (name, out)


def test_heterogeneous_measured_units_drop_the_flag_and_one_other_unit_does_not():
    plan = {"interval_is_probability": True}
    measured = _spec([
        {"entity_id": "A", "unit": "bps", "latest_value": 0.4},
        {"entity_id": "B", "unit": "basis points", "latest_value": 0.5},
    ], name="outcome")
    out = finalize(measured, Forecast("A", median=0.4), plan, ent=measured.entity("A"))
    assert is_probability(plan, measured) is False
    assert (out["lo"], out["hi"]) != (0.0, 1.0)
    mixed = _spec([
        {"entity_id": "A", "unit": "bps", "latest_value": 0.4},
        {"entity_id": "B", "unit": "score", "latest_value": 0.5},
    ], name="outcome")
    kept = finalize(mixed, Forecast("A", median=0.4), plan, ent=mixed.entity("A"))
    assert is_probability(plan, mixed) is True
    assert (kept["lo"], kept["hi"]) == (0.0, 1.0)


def test_prob_column_tokens():
    plan = {"interval_is_probability": True}

    def run(col, val):
        row = {"entity_id": "A", "consensus_eps": 2.5, col: val}
        spec = _spec([row], tt="classification", name="eps_outcome", labels=["beat", "miss"])
        return finalize(
            spec,
            Forecast("A", label="beat", median=2.2, p05=2.0, p95=2.4),
            plan,
            ent=spec.entity("A"),
        )

    for col in ("problem_count", "probe", "probable", "ProblemCount"):
        out = run(col, 3)
        assert (out["lo"], out["hi"]) != (0.0, 1.0), (col, out)
        assert out["point_forecast"] == pytest.approx(2.2)
    for col in ("implied_prob", "consensus_probability"):
        out = run(col, 0.3)
        assert (out["lo"], out["hi"]) == (0.0, 1.0), (col, out)


def test_planner_normal_percent_matches_an_absent_flag():
    rows = [{"entity_id": "X", "unit": "%", "latest_value": 0.4}]
    spec = _spec(rows)
    flagged = {"interval_is_probability": False}
    a = finalize(spec, Forecast("X", median=0.4, p05=0.3, p95=0.5), None, ent=spec.entity("X"))
    b = finalize(spec, Forecast("X", median=0.4, p05=0.3, p95=0.5), flagged, ent=spec.entity("X"))
    assert a == b
    assert is_probability(None, spec) is False
    assert is_probability(flagged, spec) is False
    assert numeric_target_plan(dict(flagged), spec) is False


def _answer(raw, rows, **kw):
    spec = _spec(raw, tt="classification", name="outcome", canaries=(GUID,), raw={"entities": raw}, **kw)
    built = build_answer(spec, rows, kw.get("trace", ""), kw.get("reasons"), None)
    text = json.dumps(built, ensure_ascii=False)
    errs = validate(spec, type("C", (), {"docs": {}})(), built)
    return built, text, errs


def test_canary_case_variants_and_empty_claims():
    bad = {"doc_id": "d1", "span_start": 0, "span_end": 8, "claim": GUID}
    clean = {"doc_id": "d1", "span_start": 0, "span_end": 5, "claim": "hello"}
    row = {"lo": 0.0, "hi": 1.0, "point_forecast": 0.4, "label": "yes"}
    built, text, errs = _answer(
        [{"entity_id": "X", "commentary": f"note {GUID.upper()}"}],
        {"X": {**row, "claims": [bad]}},
    )
    assert GUID.lower() not in text.lower()
    assert built["entity_predictions"][0]["claims"] == []
    assert "X: no claims" in errs

    built, text, errs = _answer(
        [{"entity_id": "X", "latest_value": 1.5, "commentary": GUID}],
        {"X": {**row, "claims": [{"doc_id": "d1", "span_start": 0, "span_end": 4, "claim": GUID.swapcase()}]}},
    )
    assert GUID.lower() not in text.lower()
    assert built["entity_predictions"][0]["claims"]
    assert "latest_value" in text
    assert not any("canary" in e for e in errs)

    built, text, errs = _answer(
        [{"entity_id": "X", "latest_value": 1.5}],
        {"X": {**row, "label": GUID.upper(), "claims": [clean]}},
    )
    assert GUID.lower() not in text.lower()
    assert "label" not in built["entity_predictions"][0]
    assert "X: label" in errs


def test_other_date_like_level_columns_keep_a_probability():
    """Columns _LEVEL_FIELD already selects, whose names are dates but not year|date|vintage."""
    bad = []
    plan = {"interval_is_probability": True}
    for key, val in (
        ("latest_yr", 2024),
        ("latest_fy", 2024),
        ("current_fy", 2024),
        ("prior_period", 2024),
        ("previous_month", 10),
        ("last_quarter", 4),
        ("latest_qtr", 2024),
    ):
        rows = [{"entity_id": "X", key: val, "latest_value": 0.2}]
        spec = _spec(rows)
        ent = spec.entity("X")
        out = finalize(spec, Forecast("X", median=0.2), plan, ent=ent)
        base = baseline_value(spec, ent, plan)
        if (out["lo"], out["hi"]) != (0.0, 1.0) or out["point_forecast"] != pytest.approx(0.2):
            bad.append(f"{key}={val} baseline={base} out={out}")
    assert not bad, "\n".join(bad)


def test_calendar_substring_does_not_hide_an_out_of_range_baseline():
    bad = []
    plan = {"interval_is_probability": True}
    rows_in = [
        {"entity_id": "X", "updated_value": 2.5},
        {"entity_id": "X", "candidate_value": 2.5},
        {"entity_id": "X", "dated_value": 2.5},
        {"entity_id": "X", "prior_year_value": 4.5},
        {"entity_id": "X", "estimate_year": 2.5},
        {"entity_id": "X", "latest_year_eps": 4.5},
        {"entity_id": "X", "latest_value": 2.5, "update": 2.5},
        {"entity_id": "X", "latest_value": 2.5, "candidate": 2.5},
        {"entity_id": "X", "latest_value": 2024, "latest_year": 2024},
    ]
    for row in rows_in:
        spec = _spec([row])
        out = finalize(spec, Forecast("X", median=0.4), plan, ent=spec.entity("X"))
        base = baseline_value(spec, spec.entity("X"), plan)
        if (out["lo"], out["hi"]) == (0.0, 1.0):
            bad.append(f"baseline={base} row={row} out={out}")
    assert not bad, "\n".join(bad)


def test_hyphenated_per_cent_is_a_measured_unit():
    spec, out, ip = _reg([{"entity_id": "X", "unit": "per-cent", "latest_value": 0.4}], med=0.4)
    assert ip is False
    assert (out["lo"], out["hi"]) != (0.0, 1.0), out


def test_fallback_forecast_on_latest_year_stays_a_probability():
    rows = [{"entity_id": "X", "latest_year": 2024}]
    spec = _spec(rows)
    plan = {"interval_is_probability": True}
    fb = fallback_forecast(spec, spec.entity("X"), plan)
    out = finalize(spec, fb, plan, ent=spec.entity("X"))
    assert fb.median == pytest.approx(0.2) or (out["lo"], out["hi"]) == (0.0, 1.0), (fb.median, out)


def test_classification_latest_year_is_not_the_submitted_quantity():
    rows = [{"entity_id": "X", "latest_year": 2024}]
    spec = _spec(rows, tt="classification", name="eps_outcome", labels=["beat", "miss"])
    plan = {"interval_is_probability": True}
    out = finalize(spec, Forecast("X", label="beat", median=0.2), plan, ent=spec.entity("X"))
    assert out["point_forecast"] == pytest.approx(0.2)
    assert (out["lo"], out["hi"]) == (0.0, 1.0), out


def test_canary_in_entity_id_or_task_id_is_not_emitted():
    clean = {"doc_id": "d1", "span_start": 0, "span_end": 5, "claim": "hello"}
    row = {"lo": 0.0, "hi": 1.0, "point_forecast": 0.4, "label": "yes", "claims": [clean]}
    _built, text, errs = _answer(
        [{"entity_id": GUID, "latest_value": 1.5}],
        {GUID: row},
    )
    assert GUID.lower() not in text.lower(), (text, errs)
    _built, text, errs = _answer(
        [{"entity_id": "X", "latest_value": 1.5}],
        {"X": row},
        task_id=f"unit-{GUID}",
    )
    assert GUID.lower() not in text.lower(), (text, errs)
