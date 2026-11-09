"""Review-5 checks for the 97c4b26 calendar-guard commit.

Passing tests lock H1b, H2, M2, the M-veto drop, the C3 level columns, and
the calendar-shape boundaries. Residual tests pin current behavior this
review leaves open. They record the measured interval. They are not a claim
that the residual interval is the desired end state.
"""
from __future__ import annotations

import copy

import pytest

from t4agent.calibrate import finalize, is_probability, numeric_target_plan
from t4agent.predict import Forecast, baseline_value, fallback_forecast
from t4agent.task import TaskSpec


def _spec(rows, *, tt="regression", name="p_default"):
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


def _plan():
    return {"interval_is_probability": True, "interval_quantity": "probability 0-1"}


def _finish(rows, *, med=0.4, plan=None, band=False, name="p_default"):
    if plan is None:
        plan = _plan()
    spec = _spec(rows, name=name)
    eid = rows[0]["entity_id"]
    kw = {"median": med}
    if band:
        kw["p05"] = med - 0.1
        kw["p95"] = med + 0.1
    out = finalize(spec, Forecast(eid, **kw), plan, ent=spec.entity(eid))
    return spec, out, is_probability(plan, spec)


def _bounds(out):
    return out["point_forecast"], out["lo"], out["hi"]


def test_h1b_period_quantities_leave_the_unit_interval_and_keep_the_fallback():
    """A calendar token plus a quantity token is a level, including 1-12 and 1-4."""
    plan = _plan()
    expected = {
        "previous_month_index": (10, 0.4, -0.6, 5.7, 10.0, 5.0, 15.0),
        "prior_quarter_gdp": (3, 0.4, -0.6, 2.2, 3.0, 1.5, 4.5),
        "current_period_sales": (8, 0.4, -0.6, 4.7, 8.0, 4.0, 12.0),
        "latest_month_revenue": (6, 0.4, -0.6, 3.7, 6.0, 3.0, 9.0),
        "prior_period_spread": (5, 0.4, -0.6, 3.2, 5.0, 2.5, 7.5),
        "latest_quarter_ratio": (4, 0.4, -0.6, 2.7, 4.0, 2.0, 6.0),
        "latest_month_unemployment": (7, 0.4, -0.6, 4.2, 7.0, 3.5, 10.5),
        "prior_quarter_payroll": (2, 0.4, -0.6, 1.7, 2.0, 1.0, 3.0),
    }
    bad = []
    for field, (level, point, lo, hi, fb_med, fb_lo, fb_hi) in expected.items():
        row = {"entity_id": "X", field: level}
        spec, out, ip = _finish([row], med=0.4, plan=plan)
        fb = fallback_forecast(spec, spec.entity("X"), plan)
        fout = finalize(spec, fb, plan, ent=spec.entity("X"))
        if ip is not False or _bounds(out) != (point, lo, hi) or fb.median != pytest.approx(fb_med) \
                or _bounds(fout) != (fb_med, fb_lo, fb_hi):
            bad.append(f"{field} ip={ip} hand={out} fb={fb.median} {fout}")
    assert not bad, "\n".join(bad)


def test_h2_plural_and_excel_siblings_stay_on_the_unit_interval():
    plan = _plan()
    rows = [
        {"latest_years": 2024},
        {"latest_fiscalyear": 2024},
        {"latestFiscalYear": 2024},
        {"latest_dates": 20240701},
        {"latest_date": 45292},
    ]
    bad = []
    for extra in rows:
        row = {"entity_id": "X", "latest_value": 0.2, **extra}
        _s, out, ip = _finish([row], med=0.2, plan=plan)
        if ip is not True or _bounds(out) != (0.2, 0.0, 1.0):
            bad.append(f"{extra} ip={ip} {out}")
    assert not bad, "\n".join(bad)


def test_m2_added_calendar_tokens_keep_the_in_range_sibling():
    plan = _plan()
    bare = [
        {"latest_timestamp": 1719792000},
        {"latest_week": 42},
        {"latest_weeks": 42},
        {"latest_yr": 24},
        {"latest_qtr": 20241},
    ]
    bad = []
    for extra in bare:
        row = {"entity_id": "X", "latest_value": 0.2, **extra}
        for band in (False, True):
            _s, out, ip = _finish([row], med=0.2, plan=plan, band=band)
            if ip is not True or _bounds(out) != (0.2, 0.0, 1.0):
                bad.append(f"{extra} band={band} ip={ip} {out}")
    assert not bad, "\n".join(bad)


def test_m_veto_drops_only_when_every_non_calendar_level_is_outside():
    """A year-only, empty, or non-finite row is neutral. An outside level drops."""
    samples = [
        [{"entity_id": "A", "latest_year": 2024}, {"entity_id": "B", "latest_value": 2.5}],
        [{"entity_id": "A", "note": "x"}, {"entity_id": "B", "latest_value": 2.5}],
        [{"entity_id": "A", "latest_value": float("nan")}, {"entity_id": "B", "latest_value": 2.5}],
    ]
    for rows in samples:
        spec = _spec(rows)
        snap = _plan()
        assert numeric_target_plan(snap, spec) is True
        assert snap["interval_is_probability"] is False
        assert "interval_quantity" not in snap
        assert is_probability(_plan(), spec) is False

    kept = [
        [{"entity_id": "A", "latest_year": 2024}, {"entity_id": "B", "latest_value": 0.2}],
        [{"entity_id": "A", "latest_year": 2024}],
        [{"entity_id": "A", "latest_value": 0.0}, {"entity_id": "B", "latest_value": 2.5}],
        [{"entity_id": "A", "latest_value": 1.0}, {"entity_id": "B", "latest_value": 2.5}],
    ]
    for rows in kept:
        spec = _spec(rows)
        snap = _plan()
        assert numeric_target_plan(snap, spec) is False, rows
        assert snap["interval_is_probability"] is True
        assert is_probability(_plan(), spec) is True


def test_c3_level_columns_that_mention_a_date_still_drop():
    plan = _plan()
    rows = [
        {"entity_id": "A", "latest_vintage_value": 159000},
        {"entity_id": "B", "latest_rate_at_date": 4.35},
        {"entity_id": "C", "latest_value": 2024},
    ]
    for row in rows:
        spec = _spec([row])
        snap = _plan()
        assert numeric_target_plan(snap, spec) is True, row
        _s, out, ip = _finish([row], med=0.4, plan=plan)
        assert ip is False, row
        assert (out["lo"], out["hi"]) != (0.0, 1.0), (row, out)


def test_calendar_shape_boundaries_hold():
    """Week 1-53, two-digit year 0-99, and Excel through 73415 are calendar. The next integer is a level."""
    plan = _plan()
    stays = [
        {"latest_week": 53, "latest_value": 0.2},
        {"latest_year": 99, "latest_value": 0.2},
        {"latest_date": 73415, "latest_value": 0.2},
        {"latest_timestamp": 315532800, "latest_value": 0.2},
    ]
    leaves = [
        {"latest_week": 54, "latest_value": 0.2},
        {"latest_year": 100, "latest_value": 0.2},
        {"latest_date": 73416, "latest_value": 0.2},
        {"latest_date": 4.35, "latest_value": 0.2},
        {"latest_timestamp": 315532799, "latest_value": 0.2},
    ]
    for fields in stays:
        row = {"entity_id": "X", **fields}
        _s, out, ip = _finish([row], med=0.2, plan=plan)
        assert ip is True and _bounds(out) == (0.2, 0.0, 1.0), (fields, ip, out)
    for fields in leaves:
        row = {"entity_id": "X", **fields}
        _s, out, ip = _finish([row], med=0.2, plan=plan)
        assert ip is False and (out["lo"], out["hi"]) != (0.0, 1.0), (fields, ip, out)


def test_analyst_band_on_the_outside_row_survives_the_drop():
    """After the veto, a real outside level keeps the analyst band. Empty and NaN rows do too."""
    plan = _plan()
    cases = [
        [{"entity_id": "A", "latest_year": 2024}, {"entity_id": "B", "latest_value": 2.5}],
        [{"entity_id": "A", "note": "x"}, {"entity_id": "B", "latest_value": 2.5}],
        [{"entity_id": "A", "latest_value": float("nan")}, {"entity_id": "B", "latest_value": 2.5}],
    ]
    for rows in cases:
        spec = _spec(rows)
        snap = copy.deepcopy(plan)
        assert numeric_target_plan(snap, spec) is True
        out = finalize(
            spec,
            Forecast("B", median=2.5, p05=2.3, p95=2.7),
            snap,
            ent=spec.entity("B"),
        )
        assert _bounds(out) == (2.5, 2.1, 2.9), (rows, out)


def test_residual_calendar_only_row_hedges_to_the_year_after_the_drop():
    """Medium residual. The veto drops the flag, and the hedge then stops skipping dates.

    Desired later: the analyst band 2.5 [2.1, 2.9] on the year-only row as well.
    Today the year is the numeric baseline, banker's rounding takes the point
    to 2.0, and the upper bound is pulled to 1014.
    """
    rows = [
        {"entity_id": "A", "latest_year": 2024},
        {"entity_id": "B", "latest_value": 2.5},
    ]
    spec = _spec(rows)
    snap = _plan()
    assert numeric_target_plan(snap, spec) is True
    out = finalize(
        spec,
        Forecast("A", median=2.5, p05=2.3, p95=2.7),
        snap,
        ent=spec.entity("A"),
    )
    assert _bounds(out) == (2.0, 2.0, 1014.0)
    assert baseline_value(spec, spec.entity("A"), snap, skip_date_period=False) == pytest.approx(2024.0)


def test_residual_generic_year_treats_week_2024_as_a_calendar():
    """Medium residual. The 1900-2100 check runs before the week-of-year check.

    latest_week=2024 is skipped. Parent 4bfed24, c83a768, and 6739240 treat it
    as a level and widen. Week 54 still leaves, and week 42 still stays.
    """
    plan = _plan()
    spec, out, ip = _finish([{"entity_id": "X", "latest_week": 2024}], med=0.4, plan=plan)
    fb = fallback_forecast(spec, spec.entity("X"), plan)
    assert ip is True
    assert _bounds(out) == (0.4, 0.0, 1.0)
    assert fb.median == pytest.approx(0.0)
    _s, wide, dropped_ip = _finish(
        [{"entity_id": "X", "latest_week": 2024, "latest_value": 0.2}], med=0.2, plan=plan,
    )
    assert dropped_ip is True and _bounds(wide) == (0.2, 0.0, 1.0)


def test_residual_excel_window_treats_a_small_serial_as_a_date():
    """Medium residual. Excel 1..73415 skips latest_date=100. 73416 stays a level."""
    plan = _plan()
    _s, kept, ip = _finish(
        [{"entity_id": "X", "latest_date": 100, "latest_value": 0.2}], med=0.2, plan=plan,
    )
    assert ip is True and _bounds(kept) == (0.2, 0.0, 1.0)
    spec, lone, lone_ip = _finish([{"entity_id": "X", "latest_date": 100}], med=0.4, plan=plan)
    fb = fallback_forecast(spec, spec.entity("X"), plan)
    assert lone_ip is True and _bounds(lone) == (0.4, 0.0, 1.0)
    assert fb.median == pytest.approx(0.0)


def test_residual_unix_gate_does_not_fall_through_to_a_period_index():
    """Medium residual. A timestamp below 1980-01-01 returns False and does not fall through.

    latest_period_timestamp=6 is a level on this tree. Parent 4bfed24 skips it
    as a period index and keeps [0, 1]. The in-range latest_value on the same
    row is hidden because the first level field wins.
    """
    plan = _plan()
    row = {"entity_id": "X", "latest_period_timestamp": 6, "latest_value": 0.2}
    spec, out, ip = _finish([row], med=0.2, plan=plan)
    assert ip is False
    assert _bounds(out) == (0.2, -0.8, 3.6)
    assert baseline_value(spec, spec.entity("X"), plan, skip_date_period=True) == pytest.approx(6.0)
    snap = _plan()
    assert numeric_target_plan(snap, spec) is True
    prod = finalize(
        spec,
        Forecast("X", median=2.5, p05=2.3, p95=2.7),
        snap,
        ent=spec.entity("X"),
    )
    assert _bounds(prod) == (2.5, 2.1, 4.5)


def test_planner_normal_change_targets_still_match_the_locked_rows():
    """B1: flag absent. The historical splitter is unchanged on these five rows."""
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
        got = baseline_value(spec, spec.entity("X"), None)
        fb = fallback_forecast(spec, spec.entity("X"), None)
        out = finalize(spec, fb, None, ent=spec.entity("X"))
        if got != pytest.approx(base) or fb.median != pytest.approx(base) \
                or out["point_forecast"] != pytest.approx(point) or (out["lo"], out["hi"]) != bounds:
            bad.append(f"{name} base={got} med={fb.median} out={out}")
    assert not bad, "\n".join(bad)
