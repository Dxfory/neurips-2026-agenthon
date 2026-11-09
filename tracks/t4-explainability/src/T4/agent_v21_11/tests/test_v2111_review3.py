"""Counterexamples for the v21.11 G1 review of d6cdf9c.

Passing tests lock N1–N8 cases that hold. Failing tests are the planner-normal
tokenizer change and the date-rule holes measured against 6739240.
"""
from __future__ import annotations

import json

import pytest

from t4agent.answer import build_answer, validate
from t4agent.calibrate import finalize, is_probability, numeric_target_plan
from t4agent.predict import Forecast, Forecaster, baseline_value, fallback_forecast
from t4agent.task import TaskSpec

GUID = "11111111-2222-4333-8444-555555555555"


def _spec(rows, *, tt="regression", name="p_default", leg=True, canaries=(), task_id="t", labels=None):
    if labels is None:
        labels = ["beat", "miss"] if tt == "classification" else []
    return TaskSpec(
        raw={"entities": rows},
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


def _out(rows, *, name="p_default", plan=None, med=0.2, tt="regression", band=False, labels=None):
    if plan is None:
        plan = {"interval_is_probability": True}
    spec = _spec(rows, tt=tt, name=name, labels=labels)
    kw = {"median": med}
    if tt == "classification":
        kw["label"] = "beat"
    if band:
        kw["p05"] = med - 0.1
        kw["p95"] = med + 0.1
    ent = spec.entity(rows[0]["entity_id"])
    out = finalize(spec, Forecast(rows[0]["entity_id"], **kw), plan, ent=ent)
    return spec, out, is_probability(plan, spec)


def test_named_date_columns_with_a_later_level_stay_on_the_unit_interval():
    plan = {"interval_is_probability": True}
    for key, val in (
        ("latest_year", 2024),
        ("latest_vintage", 2024),
        ("latest_yr", 2024),
        ("latest_fy", 2024),
        ("current_fy", 2024),
        ("prior_period", 2024),
        ("previous_month", 10),
        ("last_quarter", 4),
        ("latest_qtr", 2024),
        ("latestFiscalYear", 2024),
    ):
        _spec_i, out, ip = _out([{"entity_id": "X", key: val, "latest_value": 0.2}], plan=plan, med=0.2)
        assert ip is True
        assert out["point_forecast"] == pytest.approx(0.2)
        assert (out["lo"], out["hi"]) == (0.0, 1.0), key


@pytest.mark.xfail(
    strict=True,
    reason="C1 supersedes the old unit boolean: a one-row out-of-range regression drops the flag pre-analyst",
)
def test_n2_non_year_values_leave_the_unit_interval():
    for row in (
        {"entity_id": "X", "updated_value": 2.5},
        {"entity_id": "X", "candidate_value": 2.5},
        {"entity_id": "X", "dated_value": 2.5},
        {"entity_id": "X", "prior_year_value": 4.5},
        {"entity_id": "X", "estimate_year": 2.5},
        {"entity_id": "X", "latest_year_eps": 4.5},
        {"entity_id": "X", "prior_quarter_estimate": 3.5},
    ):
        _s, out, ip = _out([row], med=0.4)
        assert ip is True
        assert (out["lo"], out["hi"]) != (0.0, 1.0), row


def test_per_cent_hyphen_leaves_the_unit_interval_and_ordinary_words_do_not():
    plan = {"interval_is_probability": True, "interval_quantity": "probability 0-1"}
    spec = _spec([{"entity_id": "X", "unit": "per-cent", "latest_value": 0.4}])
    snap = dict(plan)
    assert numeric_target_plan(snap, spec) is True
    _s, out, ip = _out([{"entity_id": "X", "unit": "per-cent", "latest_value": 0.4}], med=0.4)
    assert ip is False
    assert (out["lo"], out["hi"]) != (0.0, 1.0)
    for name in ("supply", "happy", "opportunity", "appendix", "per-centage", "per-century", "PPP"):
        _s, kept, ip = _out([{"entity_id": "X", "latest_value": 0.4}], name=name, med=0.4)
        assert ip is True and (kept["lo"], kept["hi"]) == (0.0, 1.0), name


def test_implied_prob_camel_and_glued_keep_the_classification_interval():
    plan = {"interval_is_probability": True}
    for col in ("impliedprob", "impliedProb", "probValue", "likelihoodValue", "chanceScore", "oddsRatio"):
        row = [{"entity_id": "A", "consensus_eps": 2.5, col: 0.3}]
        spec = _spec(row, tt="classification", name="eps_outcome")
        out = finalize(
            spec,
            Forecast("A", label="beat", median=2.2, p05=2.0, p95=2.4),
            plan,
            ent=spec.entity("A"),
        )
        assert (out["lo"], out["hi"]) == (0.0, 1.0), col
    for col in ("problem_count", "probe", "probable", "ProblemCount", "problem", "approbation"):
        row = [{"entity_id": "A", "consensus_eps": 2.5, col: 3}]
        spec = _spec(row, tt="classification", name="eps_outcome")
        out = finalize(
            spec,
            Forecast("A", label="beat", median=2.2, p05=2.0, p95=2.4),
            plan,
            ent=spec.entity("A"),
        )
        assert (out["lo"], out["hi"]) != (0.0, 1.0), col


def test_planner_normal_percent_sign_matches_v21_10_shape():
    """Flag absent. The '%' guard must not move this row. Bands match the property test."""
    rows = [{"entity_id": "X", "unit": "%", "latest_value": 0.4}]
    spec = _spec(rows)
    flagged = {"interval_is_probability": False}
    bare = finalize(spec, Forecast("X", median=0.4, p05=0.3, p95=0.5), None, ent=spec.entity("X"))
    marked = finalize(spec, Forecast("X", median=0.4, p05=0.3, p95=0.5), flagged, ent=spec.entity("X"))
    assert bare == marked
    assert bare == {"point_forecast": 0.4, "lo": 0.2, "hi": 0.6}
    assert is_probability(None, spec) is False
    assert is_probability(flagged, spec) is False


def test_task_claim_skips_a_shorter_canary_pair():
    desc = "revenue rose in the quarter and the year before"
    entities = [{"entity_id": "X", "note": GUID, "description": desc}]
    spec = _spec(entities, tt="classification", name="outcome", canaries=(GUID,), labels=["yes", "no"])
    row = {"lo": 0.0, "hi": 1.0, "point_forecast": 0.4, "label": "yes", "claims": [
        {"doc_id": "d1", "span_start": 0, "span_end": 8, "claim": GUID},
    ]}
    built = build_answer(spec, {"X": row})
    text = json.dumps(built)
    assert GUID.lower() not in text.lower()
    claim = built["entity_predictions"][0]["claims"][0]["claim"]
    assert claim == f'"description": "{desc}"'
    errs = validate(spec, type("C", (), {"docs": {}})(), built)
    assert errs == []


def test_clean_identifier_is_not_redacted():
    entities = [{"entity_id": "AAPL", "latest_value": 1.5}]
    spec = _spec(
        entities, tt="classification", name="outcome",
        canaries=("7a2e9c4b-1d6f-4a8e-b3c2-5e9d1f7a6b40",),
        task_id="t4-eps-growth-2024Q3-banks",
        labels=["yes", "no"],
    )
    row = {"lo": 0.0, "hi": 1.0, "point_forecast": 0.4, "label": "yes", "claims": []}
    built = build_answer(spec, {"AAPL": row})
    text = json.dumps(built)
    assert built["task_id"] == "t4-eps-growth-2024Q3-banks"
    assert built["entity_predictions"][0]["entity_id"] == "AAPL"
    assert "7a2e9c4b" not in text
    assert built["entity_predictions"][0]["claims"]
    errs = validate(spec, type("C", (), {"docs": {}})(), built)
    assert errs == []


def test_canary_identifier_is_redacted():
    entities = [{"entity_id": GUID.upper(), "latest_value": 1.5}]
    spec = _spec(entities, tt="classification", name="outcome", canaries=(GUID,), labels=["yes", "no"])
    row = {"lo": 0.0, "hi": 1.0, "point_forecast": 0.4, "label": "yes", "claims": [
        {"doc_id": "d1", "span_start": 0, "span_end": 5, "claim": "hello"},
    ]}
    built = build_answer(spec, {GUID.upper(): row})
    text = json.dumps(built)
    assert GUID.lower() not in text.lower()
    assert built["entity_predictions"][0]["entity_id"] == "[redacted]"
    errs = validate(spec, type("C", (), {"docs": {}})(), built)
    assert "roster mismatch" in errs


def test_planner_normal_change_field_tokens_match_6739240():
    """CamelCase and letter-digit splits must not move a flag-absent regression.

    6739240 / v21.10 baseline for these rows is 0 (the glued token does not
    intersect the target) or the field value (``q3`` / ``yoy2`` / ``mom1``
    still intersect). HEAD's ``_field_tokens`` flips both directions.
    """
    plan = None
    expected = [
        ("mom_change", {"entity_id": "X", "latestMom": 1.5}, 0.0, 0.0, (-20.0, 20.0)),
        ("eps_change", {"entity_id": "X", "latest_change2": 4.0}, 0.0, 0.0, (-20.0, 20.0)),
        ("q3_revision", {"entity_id": "X", "prior_q3_change": 1.1}, 1.1, 1.0, (-19.0, 22.0)),
        ("yoy2_revision", {"entity_id": "X", "latest_yoy2": 2.5}, 2.5, 2.0, (-18.0, 23.0)),
        ("mom1_change", {"entity_id": "X", "latest_mom1": 0.7}, 0.7, 1.0, (-20.0, 21.0)),
    ]
    bad = []
    for name, row, base, point, bounds in expected:
        spec = _spec([row], name=name)
        got_base = baseline_value(spec, spec.entity("X"), plan)
        fb = fallback_forecast(spec, spec.entity("X"), plan)
        out = finalize(spec, fb, plan, ent=spec.entity("X"))
        if got_base != pytest.approx(base) or fb.median != pytest.approx(base) \
                or out["point_forecast"] != pytest.approx(point) or (out["lo"], out["hi"]) != bounds:
            bad.append(f"{name} base={got_base} med={fb.median} out={out}")
    assert not bad, "\n".join(bad)


def test_period_quantities_are_not_calendar_indexes():
    """A 1–12 or 1–4 integer on a quantity field is the measurement.

    6739240 leaves these rows off [0, 1]. HEAD treats month/period/quarter
    tokens plus that integer as a calendar index and clamps them.
    """
    bad = []
    for row in (
        {"entity_id": "X", "previous_month_return": 10},
        {"entity_id": "X", "prior_quarter_estimate": 3},
        {"entity_id": "X", "latest_quarter_eps": 4},
        {"entity_id": "X", "current_period_value": 8},
        {"entity_id": "X", "prior_period_eps": 5},
        {"entity_id": "X", "latest_month_change": 6},
    ):
        _s, out, _ip = _out([row], med=0.4)
        if (out["lo"], out["hi"]) == (0.0, 1.0):
            bad.append(f"{row} {out}")
    spec = _spec([{"entity_id": "X", "prior_quarter_estimate": 3}])
    plan = {"interval_is_probability": True}
    fb = fallback_forecast(spec, spec.entity("X"), plan)
    fout = finalize(spec, fb, plan, ent=spec.entity("X"))
    if (fout["lo"], fout["hi"]) == (0.0, 1.0):
        bad.append(f"fallback med={fb.median} out={fout}")
    assert not bad, "\n".join(bad)


def test_substring_date_names_and_excel_serials_stay_on_the_unit_interval():
    """6739240 keeps these on [0, 1] because the field name contains year or date.

    HEAD's token set misses the plural and the glued form, and the value
    check misses an Excel serial, so the F1 interval comes back.
    """
    bad = []
    for row in (
        {"entity_id": "X", "latest_years": 2024, "latest_value": 0.2},
        {"entity_id": "X", "latest_fiscalyear": 2024, "latest_value": 0.2},
        {"entity_id": "X", "latest_dates": 20240701, "latest_value": 0.2},
        {"entity_id": "X", "latest_date": 45292, "latest_value": 0.2},
    ):
        _s, out, _ip = _out([row], med=0.2)
        if (out["lo"], out["hi"]) != (0.0, 1.0) or out["point_forecast"] != pytest.approx(0.2):
            bad.append(f"{row} {out}")
    assert not bad, "\n".join(bad)


def test_llm_blend_skips_a_calendar_level_on_a_true_flag():
    rows = [{"entity_id": "X", "latest_year": 2024, "latest_value": 0.2}]
    spec = _spec(rows)
    plan = {"interval_is_probability": True}

    class Dummy:
        def __init__(self):
            self.spec = spec
            self.plan = plan
            self.stat = {}
            self.stat_weight = 0.5

        def _revision_in_target(self, eid):
            return None

    fc = Forecast("X", median=0.2, source="llm")
    Forecaster._combine_with_stats(Dummy(), fc)
    out = finalize(spec, fc, plan, ent=spec.entity("X"))
    assert fc.median == pytest.approx(0.2)
    assert (out["lo"], out["hi"]) == (0.0, 1.0), out


def test_classification_point_refill_skips_the_year():
    rows = [{"entity_id": "X", "latest_year": 2024, "consensus_eps": 2.5}]
    _s, out, _ip = _out(rows, tt="classification", name="eps_outcome", med=0.2, labels=["beat", "miss"])
    assert out["point_forecast"] == pytest.approx(2.5)
    assert (out["lo"], out["hi"]) != (0.0, 1.0)
    assert out["point_forecast"] < 100
