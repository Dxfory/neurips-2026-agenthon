"""Review-4 checks for HEAD 4bfed24.

Passing tests lock the B1 revert, the C1/C2 contract, and the H1/H3/M1/H5
fixes. Failing tests are counterexamples still open on the flag-true path:
period quantities outside the allow-list, plural/glued calendar names, and a
calendar-only row that vetoes the pre-analyst drop.
"""
from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from t4agent.calibrate import finalize, is_probability, numeric_target_plan
from t4agent.predict import Forecast, Forecaster, fallback_forecast
from t4agent.task import TaskSpec

ROOT = Path(__file__).resolve().parents[1]


def _spec(rows, *, tt="regression", name="p_default", labels=None):
    if labels is None:
        labels = ["beat", "miss"] if tt == "classification" else []
    return TaskSpec(
        raw={"entities": rows},
        task_id="t",
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
        card_params={"interval_leg": True},
    )


def _finish(rows, *, med=0.4, name="p_default", plan=None, tt="regression", which=None, band=False, labels=None):
    if plan is None:
        plan = {"interval_is_probability": True, "interval_quantity": "probability 0-1"}
    spec = _spec(rows, tt=tt, name=name, labels=labels)
    eid = which or rows[0]["entity_id"]
    kw = {"median": med}
    if tt == "classification":
        kw["label"] = "beat"
    if band and med is not None:
        kw["p05"] = med - 0.1
        kw["p95"] = med + 0.1
    out = finalize(spec, Forecast(eid, **kw), plan, ent=spec.entity(eid))
    return spec, out, is_probability(plan, spec)


def test_planner_normal_camel_change_baselines_match_the_locked_tree():
    """B1: flag absent. CamelCase and letter-digit fields use the historical splitter."""
    expected = [
        ("mom_change", {"entity_id": "X", "latestMom": 1.5}, 0.0, 0.0, (-20.0, 20.0)),
        ("eps_change", {"entity_id": "X", "latest_change2": 4.0}, 0.0, 0.0, (-20.0, 20.0)),
        ("q3_revision", {"entity_id": "X", "prior_q3_change": 1.1}, 1.1, 1.0, (-19.0, 22.0)),
        ("yoy2_revision", {"entity_id": "X", "latest_yoy2": 2.5}, 2.5, 2.0, (-18.0, 23.0)),
        ("mom1_change", {"entity_id": "X", "latest_mom1": 0.7}, 0.7, 1.0, (-20.0, 21.0)),
    ]
    from t4agent.predict import baseline_value

    bad = []
    for name, row, base, point, bounds in expected:
        spec = _spec([row], name=name)
        got = baseline_value(spec, spec.entity("X"), None)
        fb = fallback_forecast(spec, spec.entity("X"), None)
        out = finalize(spec, fb, None, ent=spec.entity("X"))
        if got != pytest.approx(base) or fb.median != pytest.approx(base) \
                or out["point_forecast"] != pytest.approx(point) or (out["lo"], out["hi"]) != bounds:
            bad.append(f"{name} base={got} med={fb.median} out={out}")
    assert not bad, "\n".join(bad)


def test_c2_mixed_row_keeps_the_sibling_and_centres_on_the_naive_level():
    rows = [
        {"entity_id": "A", "latest_value": 0.2},
        {"entity_id": "B", "latest_value": 1.5},
    ]
    plan = {"interval_is_probability": True, "interval_quantity": "probability 0-1"}
    spec = _spec(rows)
    assert is_probability(plan, spec) is True
    snap = dict(plan)
    assert numeric_target_plan(snap, spec) is False
    assert snap == plan
    a = finalize(spec, Forecast("A", median=0.2), plan, ent=spec.entity("A"))
    b = finalize(spec, Forecast("B", median=0.4), plan, ent=spec.entity("B"))
    assert (a["lo"], a["hi"]) == (0.0, 1.0)
    assert a["point_forecast"] == pytest.approx(0.2)
    assert (b["lo"], b["hi"]) == (0.5, 2.5)
    assert b["point_forecast"] == pytest.approx(1.5)


def test_c1_one_row_outside_drops_the_flag_and_the_interval_leaves_unit():
    samples = [
        {"entity_id": "X", "latest_value": 2.5, "latest_year": 2024},
        {"entity_id": "X", "updated_value": 2.5},
        {"entity_id": "X", "dated_value": 2.5},
        {"entity_id": "X", "prior_year_value": 4.5},
        {"entity_id": "X", "estimate_year": 2.5},
        {"entity_id": "X", "latest_year_eps": 4.5},
        {"entity_id": "X", "prior_quarter_estimate": 3.5},
    ]
    plan = {"interval_is_probability": True, "interval_quantity": "probability 0-1"}
    for row in samples:
        spec, out, ip = _finish([row], med=0.4, plan=plan)
        assert ip is False, row
        snap = dict(plan)
        assert numeric_target_plan(snap, spec) is True
        assert snap["interval_is_probability"] is False
        assert "interval_quantity" not in snap
        assert (out["lo"], out["hi"]) != (0.0, 1.0), (row, out)


def test_allowlisted_period_quantities_leave_the_unit_interval():
    plan = {"interval_is_probability": True}
    for row in (
        {"entity_id": "X", "previous_month_return": 10},
        {"entity_id": "X", "prior_quarter_estimate": 3},
        {"entity_id": "X", "latest_quarter_eps": 4},
        {"entity_id": "X", "current_period_value": 8},
        {"entity_id": "X", "prior_period_eps": 5},
        {"entity_id": "X", "latest_month_change": 6},
    ):
        _s, out, _ip = _finish([row], med=0.4, plan=plan)
        assert (out["lo"], out["hi"]) != (0.0, 1.0), (row, out)
    spec = _spec([{"entity_id": "X", "prior_quarter_estimate": 3}])
    fb = fallback_forecast(spec, spec.entity("X"), plan)
    out = finalize(spec, fb, plan, ent=spec.entity("X"))
    assert fb.median == pytest.approx(3.0)
    assert out["point_forecast"] == pytest.approx(3.0)
    assert (out["lo"], out["hi"]) == (1.5, 4.5)


def test_blend_skips_the_year_and_the_classification_refill_uses_eps():
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
    assert (out["lo"], out["hi"]) == (0.0, 1.0)
    assert out["point_forecast"] == pytest.approx(0.2)

    _s, filled, _ip = _finish(
        [{"entity_id": "X", "latest_year": 2024, "consensus_eps": 2.5}],
        tt="classification", name="eps_outcome", med=0.2,
    )
    assert filled["point_forecast"] == pytest.approx(2.5)
    assert (filled["lo"], filled["hi"]) != (0.0, 1.0)


def test_h5_caps_only_lower_and_the_image_sets_no_local_clock(monkeypatch):
    def load(**env):
        for key in ("T4_TIME_BUDGET", "T4_LLM_DEADLINE", "T4_CAP_TIME_BUDGET", "T4_CAP_LLM_DEADLINE"):
            monkeypatch.delenv(key, raising=False)
        for key, val in env.items():
            monkeypatch.setenv(key, val)
        import t4agent.config as config
        return importlib.reload(config)

    bare = load()
    try:
        assert bare.derive_clock(None) == bare.derive_clock(0) == bare.derive_clock(float("nan"))
        assert bare.derive_clock(None) == (470.0, 400.0, 1.0)
        assert bare.derive_clock(300.0) == (210.0, 140.0, 0.35)
        high = load(T4_CAP_TIME_BUDGET="5000", T4_CAP_LLM_DEADLINE="4000")
        assert high.derive_clock(600.0) == (470.0, 400.0, 1.0)
        assert high.derive_clock(300.0) == (210.0, 140.0, 0.35)
        low = load(T4_CAP_TIME_BUDGET="360", T4_CAP_LLM_DEADLINE="290")
        assert low.derive_clock(600.0) == (360.0, 290.0, 0.725)
        assert low.derive_clock(300.0) == (210.0, 140.0, 0.35)
        invalid = load(T4_CAP_TIME_BUDGET="abc", T4_CAP_LLM_DEADLINE="-5")
        assert invalid.derive_clock(600.0) == (470.0, 400.0, 1.0)
        local = load(T4_TIME_BUDGET="1000", T4_LLM_DEADLINE="900")
        assert local.derive_clock(600.0) == (1000.0, 900.0, 1.0)
        # The Final image sets the caps (360 / 290, test_v2111_harness.py), never the local-testing clock.
        text = (ROOT / "Dockerfile").read_text(encoding="utf-8")
        for name in ("T4_TIME_BUDGET", "T4_LLM_DEADLINE"):
            assert name not in text
    finally:
        load()


def test_unlisted_period_quantities_are_not_submitted_as_a_unit_interval():
    """month/quarter/period plus a 1-12 or 1-4 integer is still a quantity.

    Words outside _QUANTITY_FIELD_TOKENS are skipped as calendar indexes, so
    the row is submitted as [0, 1]. c83a768 and 6739240 submit the quantity.
    """
    plan = {"interval_is_probability": True}
    bad = []
    for row in (
        {"entity_id": "X", "previous_month_index": 10},
        {"entity_id": "X", "prior_quarter_gdp": 3},
        {"entity_id": "X", "current_period_sales": 8},
        {"entity_id": "X", "latest_month_revenue": 6},
        {"entity_id": "X", "prior_period_spread": 5},
        {"entity_id": "X", "latest_quarter_ratio": 4},
        {"entity_id": "X", "latest_month_unemployment": 7},
        {"entity_id": "X", "prior_quarter_payroll": 2},
    ):
        _s, out, ip = _finish([row], med=0.4, plan=plan)
        if (out["lo"], out["hi"]) == (0.0, 1.0) or ip is not False:
            bad.append(f"{row} ip={ip} {out}")
    spec = _spec([{"entity_id": "X", "prior_quarter_gdp": 3}])
    fb = fallback_forecast(spec, spec.entity("X"), plan)
    fout = finalize(spec, fb, plan, ent=spec.entity("X"))
    if fb.median != pytest.approx(3.0) or (fout["lo"], fout["hi"]) == (0.0, 1.0):
        bad.append(f"fallback med={fb.median} out={fout}")
    assert not bad, "\n".join(bad)


def test_plural_and_glued_calendar_names_stay_on_the_unit_interval():
    """C3 keeps level columns numeric. It does not require these to widen.

    latest_years / latest_fiscalyear / latest_dates / an Excel serial are
    calendar-shaped. 6739240 keeps the in-range sibling on [0, 1].
    """
    bad = []
    for row in (
        {"entity_id": "X", "latest_years": 2024, "latest_value": 0.2},
        {"entity_id": "X", "latest_fiscalyear": 2024, "latest_value": 0.2},
        {"entity_id": "X", "latest_dates": 20240701, "latest_value": 0.2},
        {"entity_id": "X", "latest_date": 45292, "latest_value": 0.2},
    ):
        _s, out, _ip = _finish([row], med=0.2)
        if (out["lo"], out["hi"]) != (0.0, 1.0) or out["point_forecast"] != pytest.approx(0.2):
            bad.append(f"{row} {out}")
    assert not bad, "\n".join(bad)


def test_a_calendar_only_row_does_not_veto_the_pre_analyst_drop():
    """Every row that has a non-calendar naive level is outside [0, 1].

    The year-only row has no naive level. The unit should drop the flag
    before the analysts, as C1 does when every row has an outside level.
    """
    rows = [
        {"entity_id": "A", "latest_year": 2024},
        {"entity_id": "B", "latest_value": 2.5},
    ]
    plan = {"interval_is_probability": True, "interval_quantity": "probability 0-1"}
    spec = _spec(rows)
    snap = dict(plan)
    dropped = numeric_target_plan(snap, spec)
    assert dropped is True
    assert snap["interval_is_probability"] is False
    assert "interval_quantity" not in snap
