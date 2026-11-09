"""Adversarial checks for the label/point consistency gate.

These tests are independent of tests/test_threshold_consistency.py. They assert the
contract the gate claims: adjust only when the submitted point and the threshold are
the same known quantity, and otherwise leave the row alone; a large move lands on the
nearest published value inside the label region, the publication grid, the strictness
of the bound, and the submitted interval.
"""
from decimal import Decimal

import pytest

from t4agent.calibrate import finalize
from t4agent.predict import Forecast
from t4agent.task import TaskSpec
from t4agent.threshold_consistency import (
    apply_threshold_consistency,
    build_card,
    label_of,
)


def _spec(target_type, name, labels, prompt, entities, assertions=None):
    raw = {
        "task_id": "xc",
        "target": {
            "name": name,
            "type": target_type,
            "labels": labels,
            "label_assertions": assertions or {},
        },
        "prompt": prompt,
        "entities": entities,
    }
    return TaskSpec(
        raw=raw,
        task_id="xc",
        target_type=target_type,
        target_type_trusted=True,
        target_name=name,
        labels=list(labels),
        prompt=prompt,
        cutoff=None,
        cutoff_str="",
        resolution_str="",
        interval_level=0.9,
        entities=list(entities),
        entity_ids=[e["entity_id"] for e in entities],
    )


def _eps_band(consensus="1.50", pct="0.05", **extra):
    ent = {
        "entity_id": "AAPL",
        "consensus_eps": float(consensus),
        "threshold_pct": float(pct),
        "beat_condition": "reported_eps > consensus_eps * (1 + threshold_pct)",
        "miss_condition": "reported_eps < consensus_eps * (1 - threshold_pct)",
    }
    ent.update(extra)
    spec = _spec(
        "classification",
        "reported diluted EPS",
        ["beat", "miss", "inline"],
        "Will diluted EPS beat the consensus by 5%?",
        [ent],
    )
    return spec, ent


def _direction(reference, field="latest_precutoff_estimate", **extra):
    ent = {"entity_id": "INDPRO", field: reference}
    ent.update(extra)
    spec = _spec(
        "classification",
        "next_estimate_revision_direction",
        ["up", "down"],
        "Will the next published estimate be revised up or down versus the latest estimate?",
        [ent],
    )
    return spec, ent


def test_xc_exact_probability_interval_is_not_an_eps_level():
    ent = {"entity_id": "IBM", "prior_year_q_eps": 1.53, "unit": "probability"}
    spec = _spec(
        "classification",
        "eps_up_probability",
        ["up", "down"],
        "Probability that diluted EPS is higher than the prior-year quarter.",
        [ent],
    )
    original = {"label": "up", "point_forecast": 0.8, "lo": 0.0, "hi": 1.0}
    row, codes = apply_threshold_consistency(spec, ent, None, original)
    assert row == original and codes == []


def test_xc_probability_unit_with_noisy_upper_bound_is_not_an_eps_level():
    # A probability unit is not an EPS level. A hi of 1+eps must not pull 0.8 up to 1.54.
    ent = {"entity_id": "IBM", "prior_year_q_eps": 1.53, "unit": "probability"}
    spec = _spec(
        "classification",
        "eps_up_probability",
        ["up", "down"],
        "Probability that diluted EPS is higher than the prior-year quarter.",
        [ent],
    )
    original = {"label": "up", "point_forecast": 0.8, "lo": 0.0, "hi": 1.0000001}
    row, codes = apply_threshold_consistency(spec, ent, None, dict(original))
    assert row == original and codes == []


def test_xc_probability_unit_interval_below_zero_is_not_an_eps_level():
    ent = {"entity_id": "IBM", "prior_year_q_eps": 1.53, "unit": "probability"}
    spec = _spec(
        "classification",
        "eps_up_probability",
        ["up", "down"],
        "Probability that diluted EPS is higher than the prior-year quarter.",
        [ent],
    )
    original = {"label": "up", "point_forecast": 0.8, "lo": -0.1, "hi": 1.0}
    row, codes = apply_threshold_consistency(spec, ent, None, dict(original))
    assert row["point_forecast"] == 0.8
    assert row["lo"] == -0.1 and row["hi"] == 1.0
    assert codes == []


def test_xc_probability_unit_point_above_one_is_not_an_eps_level():
    ent = {"entity_id": "IBM", "prior_year_q_eps": 1.53, "unit": "probability"}
    spec = _spec(
        "classification",
        "eps_up_probability",
        ["up", "down"],
        "Probability that diluted EPS is higher than the prior-year quarter.",
        [ent],
    )
    original = {"label": "up", "point_forecast": 1.2, "lo": 0.0, "hi": 1.5}
    row, codes = apply_threshold_consistency(spec, ent, None, dict(original))
    assert row == original and codes == []


def test_xc_growth_name_and_percent_change_unit_are_not_eps_levels():
    ent = {"entity_id": "IBM", "prior_year_q_eps": 1.53, "unit": "percent"}
    spec = _spec(
        "classification",
        "eps_growth",
        ["up", "down"],
        "YoY diluted EPS growth in percent versus the prior-year quarter.",
        [ent],
    )
    original = {"label": "up", "point_forecast": 1.0, "lo": -10.0, "hi": 10.0}
    row, codes = apply_threshold_consistency(spec, ent, None, dict(original))
    assert row == original and codes == []

    ent = {"entity_id": "IBM", "prior_year_q_eps": 1.53, "unit": "percent change"}
    spec = _spec(
        "classification",
        "eps_yoy_direction",
        ["up", "down"],
        "Higher or lower than the prior-year quarter?",
        [ent],
    )
    original = {"label": "up", "point_forecast": 1.0, "lo": -10.0, "hi": 10.0}
    row, codes = apply_threshold_consistency(spec, ent, None, dict(original))
    assert row == original and codes == []


def test_xc_same_unit_eps_repair_when_an_earlier_base_field_is_present():
    # Both the point and consensus_eps are EPS levels. An earlier prior-year level
    # must not hide that and leave a beat of 1.50 inside the inline band.
    ent = {
        "entity_id": "AAPL",
        "prior_year_q_eps": 1.10,
        "consensus_eps": 1.50,
        "threshold_pct": 0.05,
        "beat_condition": "reported_eps > consensus_eps * (1 + threshold_pct)",
        "miss_condition": "reported_eps < consensus_eps * (1 - threshold_pct)",
    }
    spec = _spec(
        "classification",
        "reported diluted EPS",
        ["beat", "miss", "inline"],
        "Will diluted EPS beat the consensus by 5%?",
        [ent],
    )
    row, codes = apply_threshold_consistency(
        spec, ent, None, {"label": "beat", "point_forecast": 1.50, "lo": 0.5, "hi": 2.5},
    )
    assert row["label"] == "beat"
    assert row["point_forecast"] == pytest.approx(1.58)
    assert row["lo"] == 0.5 and row["hi"] == 2.5
    assert codes == ["THRESHOLD_FIX"]
    assert label_of(row["point_forecast"], build_card(spec, ent, None)) == "beat"


def test_xc_same_unit_eps_repair_when_the_threshold_field_comes_first():
    ent = {
        "entity_id": "AAPL",
        "consensus_eps": 1.50,
        "prior_year_q_eps": 1.10,
        "threshold_pct": 0.05,
        "beat_condition": "reported_eps > consensus_eps * (1 + threshold_pct)",
        "miss_condition": "reported_eps < consensus_eps * (1 - threshold_pct)",
    }
    spec = _spec(
        "classification",
        "reported diluted EPS",
        ["beat", "miss", "inline"],
        "Will diluted EPS beat the consensus by 5%?",
        [ent],
    )
    row, codes = apply_threshold_consistency(
        spec, ent, None, {"label": "beat", "point_forecast": 1.50, "lo": 0.5, "hi": 2.5},
    )
    assert row["point_forecast"] == pytest.approx(1.58)
    assert codes == ["THRESHOLD_FIX"]


def test_xc_unknown_unit_without_a_threshold_is_a_noop():
    ent = {"entity_id": "Q", "mystery_metric": 3.2}
    spec = _spec("classification", "mystery", ["up", "down"], "Up or down, with no cut.", [ent])
    original = {"label": "up", "point_forecast": 1.0, "lo": 0.0, "hi": 2.0}
    row, codes = apply_threshold_consistency(spec, ent, None, dict(original))
    assert row == original and codes == []
    assert build_card(spec, ent, None) is None


def test_xc_bps_change_is_not_rewritten_onto_a_level():
    ent = {"entity_id": "UST10Y", "unit": "bps", "latest_precutoff_estimate": 4.25}
    spec = _spec(
        "classification",
        "yield_change",
        ["up", "down"],
        "Yield change in bps versus the latest estimate.",
        [ent],
    )
    original = {"label": "down", "point_forecast": 12.0, "lo": -20.0, "hi": 20.0}
    row, codes = apply_threshold_consistency(spec, ent, None, dict(original))
    assert row == original and codes == []


def test_xc_large_down_move_on_a_wide_interval_lands_on_the_closed_cut():
    spec, ent = _direction(102.3058, units="index 2017=100, SA", currency="index level")
    row, codes = apply_threshold_consistency(
        spec, ent, None, {"label": "down", "point_forecast": 1e9, "lo": 0.0, "hi": 1e9},
    )
    assert row["point_forecast"] == pytest.approx(102.3058)
    assert row["lo"] == 0.0 and row["hi"] == 1e9
    assert codes == ["THRESHOLD_FIX"]
    assert label_of(row["point_forecast"], build_card(spec, ent, None)) == "down"


def test_xc_up_is_strict_and_down_is_inclusive_on_the_cut():
    spec, ent = _direction(102.3058, units="index 2017=100, SA", currency="index level")
    card = build_card(spec, ent, None)
    assert card.regions["up"].closed_lo is False
    assert card.regions["down"].closed_hi is True
    assert label_of(Decimal("102.3058"), card) == "down"
    assert label_of(Decimal("102.3059"), card) == "up"

    on_cut = {"label": "down", "point_forecast": 102.3058, "lo": 100.0, "hi": 110.0}
    row, codes = apply_threshold_consistency(spec, ent, None, dict(on_cut))
    assert row == on_cut and codes == []

    up, codes = apply_threshold_consistency(
        spec, ent, None, {"label": "up", "point_forecast": 102.3058, "lo": 100.0, "hi": 110.0},
    )
    assert up["point_forecast"] == pytest.approx(102.3059)
    assert up["lo"] == 100.0 and up["hi"] == 110.0
    assert codes == ["THRESHOLD_FIX"]
    assert label_of(up["point_forecast"], card) == "up"


def test_xc_large_down_move_stays_inside_an_overlapping_interval():
    # Down is Q <= 102.3058. [100, 102] lies entirely in that region, so the nearest
    # published point to 104 that respects the interval is 102.0000, not the cut.
    spec, ent = _direction(102.3058, units="index 2017=100, SA", currency="index level")
    row, codes = apply_threshold_consistency(
        spec, ent, None, {"label": "down", "point_forecast": 104.0, "lo": 100.0, "hi": 102.0},
    )
    assert row["label"] == "down"
    assert row["point_forecast"] == pytest.approx(102.0)
    assert row["lo"] == 100.0 and row["hi"] == 102.0
    assert codes == ["THRESHOLD_FIX"]
    assert label_of(row["point_forecast"], build_card(spec, ent, None)) == "down"


def test_xc_negative_down_move_stays_inside_an_overlapping_interval():
    # Prior EPS -2.00, down is Q <= -2. The interval [-5, -3] is inside that region.
    # The nearest published point to -1 is -3.00, not the cut.
    ent = {"entity_id": "X", "prior_year_q_eps": -2.0}
    spec = _spec(
        "classification",
        "eps_yoy_direction",
        ["up", "down"],
        "Higher or lower than the prior-year diluted EPS?",
        [ent],
    )
    row, codes = apply_threshold_consistency(
        spec, ent, None, {"label": "down", "point_forecast": -1.0, "lo": -5.0, "hi": -3.0},
    )
    assert row["point_forecast"] == pytest.approx(-3.0)
    assert row["lo"] == -5.0 and row["hi"] == -3.0
    assert codes == ["THRESHOLD_FIX"]
    assert label_of(row["point_forecast"], build_card(spec, ent, None)) == "down"


def test_xc_negative_band_and_strict_up_use_one_grid_step():
    spec, ent = _eps_band("-2.0")
    beat, codes = apply_threshold_consistency(
        spec, ent, None, {"label": "beat", "point_forecast": -4.0, "lo": -6.0, "hi": 1.0},
    )
    assert beat["point_forecast"] == pytest.approx(-1.89)
    assert codes == ["THRESHOLD_FIX"]
    assert label_of(beat["point_forecast"], build_card(spec, ent, None)) == "beat"

    miss, codes = apply_threshold_consistency(
        spec, ent, None, {"label": "miss", "point_forecast": 0.0, "lo": -6.0, "hi": 1.0},
    )
    assert miss["point_forecast"] == pytest.approx(-2.11)
    assert label_of(miss["point_forecast"], build_card(spec, ent, None)) == "miss"

    ent = {"entity_id": "Z", "prior_year_q_eps": -2.0}
    spec = _spec(
        "classification",
        "eps_yoy_direction",
        ["up", "down"],
        "Higher or lower than the prior-year diluted EPS?",
        [ent],
    )
    up, codes = apply_threshold_consistency(
        spec, ent, None, {"label": "up", "point_forecast": -2.0, "lo": -5.0, "hi": 1.0},
    )
    assert up["point_forecast"] == pytest.approx(-1.99)
    assert codes == ["THRESHOLD_FIX"]
    down, codes = apply_threshold_consistency(
        spec, ent, None, {"label": "down", "point_forecast": -2.0, "lo": -5.0, "hi": 1.0},
    )
    assert down["point_forecast"] == pytest.approx(-2.0) and codes == []


def test_xc_zero_threshold_and_zero_consensus_keep_a_one_cent_step():
    spec, ent = _eps_band("1.50", "0")
    beat, codes = apply_threshold_consistency(
        spec, ent, None, {"label": "beat", "point_forecast": 1.5, "lo": 0.0, "hi": 3.0},
    )
    assert beat["point_forecast"] == pytest.approx(1.51) and codes == ["THRESHOLD_FIX"]
    miss, codes = apply_threshold_consistency(
        spec, ent, None, {"label": "miss", "point_forecast": 1.5, "lo": 0.0, "hi": 3.0},
    )
    assert miss["point_forecast"] == pytest.approx(1.49) and codes == ["THRESHOLD_FIX"]
    inline, codes = apply_threshold_consistency(
        spec, ent, None, {"label": "inline", "point_forecast": 1.5, "lo": 0.0, "hi": 3.0},
    )
    assert inline["point_forecast"] == pytest.approx(1.5) and codes == []

    spec, ent = _eps_band("0", "0.05")
    beat, codes = apply_threshold_consistency(
        spec, ent, None, {"label": "beat", "point_forecast": 0.0, "lo": -1.0, "hi": 1.0},
    )
    assert beat["point_forecast"] == pytest.approx(0.01) and codes == ["THRESHOLD_FIX"]
    miss, codes = apply_threshold_consistency(
        spec, ent, None, {"label": "miss", "point_forecast": 0.0, "lo": -1.0, "hi": 1.0},
    )
    assert miss["point_forecast"] == pytest.approx(-0.01)
    assert label_of(miss["point_forecast"], build_card(spec, ent, None)) == "miss"
    held, codes = apply_threshold_consistency(
        spec, ent, None, {"label": "inline", "point_forecast": 0.0, "lo": -1.0, "hi": 1.0},
    )
    assert held["point_forecast"] == pytest.approx(0.0) and codes == []


def test_xc_percent_point_boundary_is_strict_above_one_percent():
    # "greater than 1%" is the number 1 on a percent-point scale, and it is strict.
    assertions = {
        "positive_reaction": "one-day abnormal return greater than 1%",
        "negative_reaction": "one-day abnormal return less than -1%",
        "flat": "one-day abnormal return between -1% and 1%",
    }
    ent = {"entity_id": "AAPL", "flat_threshold_abn_pct": 1.0}
    spec = _spec(
        "classification",
        "earnings_reaction",
        ["positive_reaction", "negative_reaction", "flat"],
        "Predict the one-day abnormal return in percent.",
        [ent],
        assertions,
    )
    flat, codes = apply_threshold_consistency(
        spec, ent, None, {"label": "flat", "point_forecast": 1.0, "lo": -7.0, "hi": 7.0},
    )
    assert flat["point_forecast"] == pytest.approx(1.0) and codes == []
    up, codes = apply_threshold_consistency(
        spec, ent, None, {"label": "positive_reaction", "point_forecast": 1.0, "lo": -7.0, "hi": 7.0},
    )
    assert up["point_forecast"] == pytest.approx(1.01)
    assert codes == ["THRESHOLD_FIX"]
    assert label_of(up["point_forecast"], build_card(spec, ent, None)) == "positive_reaction"
    down, codes = apply_threshold_consistency(
        spec, ent, None, {"label": "negative_reaction", "point_forecast": -1.0, "lo": -7.0, "hi": 7.0},
    )
    assert down["point_forecast"] == pytest.approx(-1.01)
    assert label_of(down["point_forecast"], build_card(spec, ent, None)) == "negative_reaction"


def test_xc_half_up_pre_round_still_lands_on_the_legal_side_of_a_strict_cut():
    # 1.575 is inline. A beat must leave it. HALF_UP of a point just under the cut
    # is 1.57, which is still inline, so the snap has to take the next cent, 1.58.
    spec, ent = _eps_band("1.50")
    row, codes = apply_threshold_consistency(
        spec, ent, None, {"label": "beat", "point_forecast": 1.574, "lo": 0.5, "hi": 2.5},
    )
    assert row["point_forecast"] == pytest.approx(1.58)
    assert codes == ["THRESHOLD_FIX"]
    assert label_of(row["point_forecast"], build_card(spec, ent, None)) == "beat"


@pytest.mark.parametrize("unit", ["percent", "pct", "%", "percentage"])
@pytest.mark.parametrize("quantity", [None, "year-over-year EPS growth rate in percent"])
def test_xc_percent_unit_without_growth_in_target_name_is_not_an_eps_level(unit, quantity):
    ent = {"entity_id": "X", "prior_year_q_eps": 1.53, "unit": unit}
    spec = _spec(
        "classification", "eps_yoy_direction", ["up", "down"],
        "Predict year-over-year EPS growth in percent; up means positive growth.", [ent],
        {"up": "positive year-over-year EPS growth", "down": "nonpositive year-over-year EPS growth"},
    )
    plan = {"interval_quantity": quantity} if quantity else None
    original = {"label": "up", "point_forecast": 1.0, "lo": 0.5, "hi": 1.2}
    row, codes = apply_threshold_consistency(spec, ent, plan, original)
    assert row == original and codes == []


def test_xc_numeric_row_corrected_by_finalize_does_not_keep_the_planner_probability_type():
    ent = {"entity_id": "X", "prior_year_q_eps": 1.53, "currency": "USD/share"}
    spec = _spec(
        "classification", "eps_yoy_direction", ["up", "down"],
        "Predict the diluted EPS level, higher or lower than the prior-year quarter.", [ent],
    )
    plan = {"interval_is_probability": True, "interval_quantity": "probability 0-1"}
    numeric = finalize(spec, Forecast("X", label="up", median=0.8), plan, ent=ent)
    assert numeric == {"label": "up", "point_forecast": 1.5, "lo": 0.5, "hi": 2.6}
    row, codes = apply_threshold_consistency(spec, ent, plan, numeric)
    assert row == {**numeric, "point_forecast": 1.54}
    assert codes == ["THRESHOLD_FIX"]
    assert plan == {"interval_is_probability": True, "interval_quantity": "probability 0-1"}


def test_xc_mixed_roster_keeps_its_probability_row_and_repairs_its_numeric_row():
    numeric = {"entity_id": "N", "prior_year_q_eps": 1.53, "currency": "USD/share"}
    probability = {"entity_id": "P", "prior_year_q_eps": 1.53, "consensus_probability": 0.8}
    spec = _spec(
        "classification", "outcome", ["up", "down"],
        "Each row declares its own forecast quantity.", [numeric, probability],
    )
    plan = {"interval_is_probability": True, "interval_quantity": "probability 0-1"}
    for ent, expected in [(numeric, 1.54), (probability, 0.8)]:
        original = finalize(spec, Forecast(ent["entity_id"], label="up", median=0.8), plan, ent=ent)
        row, codes = apply_threshold_consistency(spec, ent, plan, original)
        assert row["point_forecast"] == expected
        assert (row["lo"], row["hi"]) == (original["lo"], original["hi"])
        assert codes == (["THRESHOLD_FIX"] if ent is numeric else [])


@pytest.mark.parametrize("signal", ["unit", "schema", "card"])
def test_xc_explicit_probability_signals_still_win_over_an_eps_reference(signal):
    ent = {"entity_id": "X", "prior_year_q_eps": 1.53}
    spec = _spec(
        "classification", "outcome", ["up", "down"],
        "Probability that EPS rises above the prior-year value.", [ent],
    )
    if signal == "unit":
        ent["unit"] = "probability"
    elif signal == "schema":
        spec.raw["target"]["value"] = {"type": "number", "minimum": 0, "maximum": 1}
    else:
        spec.card_params = {"interval_leg": False}
    original = finalize(spec, Forecast("X", label="up", median=0.8), None, ent=ent)
    assert original == {"label": "up", "point_forecast": 0.8, "lo": 0.0, "hi": 1.0}
    row, codes = apply_threshold_consistency(spec, ent, None, original)
    assert row == original and codes == []


def test_xc_numeric_eps_interval_that_happens_to_equal_zero_one_still_has_eps_units():
    spec, ent = _eps_band("0.60")
    original = {"label": "beat", "point_forecast": 0.6, "lo": 0.0, "hi": 1.0}
    row, codes = apply_threshold_consistency(spec, ent, None, original)
    assert row == {**original, "point_forecast": 0.64}
    assert codes == ["THRESHOLD_FIX"]


@pytest.mark.parametrize("unit", ["bps change", "bps_change", "basis points change", "basis_points_change", "bps", "bp"])
def test_xc_bps_threshold_and_bps_point_remain_the_same_unit_when_named_a_change(unit):
    ent = {
        "entity_id": "X", "unit": unit,
        "up_condition": "yield_change > 10", "down_condition": "yield_change <= 10",
    }
    spec = _spec(
        "classification", "yield_change_bps", ["up", "down"],
        "Predict the yield change in bp, up when it is greater than 10 bp.", [ent],
        {"up": "yield change greater than 10 bp", "down": "otherwise"},
    )
    original = {"label": "up", "point_forecast": 5.0, "lo": -20.0, "hi": 20.0}
    row, codes = apply_threshold_consistency(spec, ent, None, original)
    assert row == {**original, "point_forecast": 10.1}
    assert codes == ["THRESHOLD_FIX"]


def test_xc_a_planner_bps_quantity_is_not_classified_as_a_percent_change():
    ent = {"entity_id": "X", "up_condition": "yield_change > 10", "down_condition": "yield_change <= 10"}
    spec = _spec(
        "classification", "outcome", ["up", "down"],
        "Predict the yield change, up when it is greater than 10 bp.", [ent],
        {"up": "yield change greater than 10 bp", "down": "otherwise"},
    )
    plan = {"interval_quantity": "change in the yield, in basis points"}
    original = {"label": "up", "point_forecast": 5.0, "lo": -20.0, "hi": 20.0}
    row, codes = apply_threshold_consistency(spec, ent, plan, original)
    assert row == {**original, "point_forecast": 10.1}
    assert codes == ["THRESHOLD_FIX"]


def test_xc_a_percent_sibling_does_not_hide_an_index_level_repair():
    index = {"entity_id": "I", "unit": "index level", "latest_precutoff_estimate": 102.3058}
    rate = {"entity_id": "R", "unit": "percent", "latest_precutoff_estimate": 4.25}
    spec = _spec(
        "classification", "next_estimate_revision_direction", ["up", "down"],
        "Predict each revised value in its own row's units, up or down versus the latest estimate.",
        [index, rate],
    )
    original = {"label": "up", "point_forecast": 100.0, "lo": 90.0, "hi": 110.0}
    row, codes = apply_threshold_consistency(spec, index, None, original)
    assert row == {**original, "point_forecast": 102.3059}
    assert codes == ["THRESHOLD_FIX"]
