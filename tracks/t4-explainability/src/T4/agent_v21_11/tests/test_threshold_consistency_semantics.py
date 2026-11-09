"""Quantity and assertion regressions found after 07fde14 / PR #50 review."""
from decimal import Decimal
from pathlib import Path

import pytest

from t4agent.task import load_task
from t4agent.threshold_consistency import apply_threshold_consistency, build_card, card_from_rules
from test_threshold_consistency_xc import _spec, _eps_band


def _apply(spec, ent, label, point, plan=None, lo=-30.0, hi=200.0):
    row = {"label": label, "point_forecast": point, "lo": lo, "hi": hi}
    out, codes = apply_threshold_consistency(spec, ent, plan, row)
    assert out["label"] == label
    assert out["lo"] <= out["point_forecast"] <= out["hi"]
    return out, codes


@pytest.mark.parametrize("aside", ["Nearby yields are quoted in basis points.", "Do not use basis points."])
def test_eps_grid_does_not_come_from_an_incidental_prompt_unit(aside):
    spec, ent = _eps_band()
    spec.prompt += " " + aside
    row, _ = _apply(spec, ent, "beat", 1.5)
    assert row["point_forecast"] == 1.58


@pytest.mark.parametrize("unit", ["USD/share", "dollars"])
def test_monetary_level_keeps_cents_even_with_a_generic_target_name(unit):
    spec, ent = _eps_band(unit=unit)
    spec.target_name = "outcome"
    row, _ = _apply(spec, ent, "beat", 1.5)
    assert row["point_forecast"] == 1.58


@pytest.mark.parametrize("target,plan", [
    ("cpi_mom_pct", None),
    ("yield_bps", None),
    ("growth_revision", None),
    ("next_estimate_revision_direction", {"interval_quantity": "change in the published estimate"}),
    ("next_estimate_revision_direction", {"interval_quantity": "return of the estimate to its latest level"}),
])
def test_row_level_wins_over_shared_target_and_planner_wording(target, plan):
    ent = {"entity_id": "I", "unit": "index level", "latest_precutoff_estimate": 102.3058}
    spec = _spec("classification", target, ["up", "down"],
                 "Forecast the revised index level. A 5% change is large; nearby yields use bp.", [ent])
    row, codes = _apply(spec, ent, "up", 100.0, plan)
    assert row["point_forecast"] == 102.3059
    assert codes == ["THRESHOLD_FIX"]


def test_currency_alone_does_not_make_a_percent_target_a_level():
    ent = {"entity_id": "X", "currency": "USD", "latest_precutoff_estimate": 102.3058}
    spec = _spec("classification", "cpi_mom_pct", ["up", "down"], "Predict percent change.", [ent])
    row, codes = _apply(spec, ent, "up", 0.2)
    assert row["point_forecast"] == 0.2 and codes == []


@pytest.mark.parametrize("phrase,point,expected", [
    ("at least 10 bp", 10.0, 10.0),
    ("at least 10 bp", 9.0, 10.0),
    ("10 bp or more", 10.0, 10.0),
    ("greater than or equal to 10 bp", 10.0, 10.0),
    ("Q >= 10 bp", 10.0, 10.0),
    ("Q ≥ 10 bp", 10.0, 10.0),
    ("no more than 10 bp", 11.0, 10.0),
    ("not more than 10 bp", 11.0, 10.0),
    ("at most 10 bp", 10.0, 10.0),
    ("10 bp or less", 10.0, 10.0),
    ("Q ≤ 10 bp", 10.0, 10.0),
    ("at least -2 bp", -3.0, -2.0),
    ("or more than 10 bp", 10.0, 10.1),
])
def test_closed_bounds_keep_the_endpoint(phrase, point, expected):
    ent = {"entity_id": "B", "unit": "bp"}
    spec = _spec("classification", "yield_change_bps", ["yes", "no"], "Predict the yield change in bp.",
                 [ent], {"yes": phrase, "no": "otherwise"})
    row, _ = _apply(spec, ent, "yes", point)
    assert row["point_forecast"] == expected


@pytest.mark.parametrize("phrase,point,expected", [
    ("spread falls to at most 10 bp", 15.0, 10.0),
    ("spread decreases to at least 10 bp", 5.0, 10.0),
    ("the spread falls: Q <= -10 bp", -20.0, -20.0),
    ("falls by at least 10 bp", -5.0, -10.0),
    ("rises by at least 10 bp", 5.0, 10.0),
])
def test_directional_amount_is_distinct_from_an_absolute_level(phrase, point, expected):
    ent = {"entity_id": "B", "unit": "bp"}
    spec = _spec("classification", "spread_bps", ["yes", "no"], "Forecast the spread in bp.",
                 [ent], {"yes": phrase, "no": "otherwise"})
    row, _ = _apply(spec, ent, "yes", point)
    assert row["point_forecast"] == expected


@pytest.mark.parametrize("flat", ["by 1% or less in either direction", "within +/-1%", "between -1% and 1%"])
def test_symmetric_band_is_not_overwritten_by_a_one_sided_closed_cut(flat):
    ent = {"entity_id": "X", "unit": "percent"}
    spec = _spec("classification", "event_return", ["up", "down", "flat"], "Predict return in percent.", [ent],
                 {"up": "rises by more than 1%", "down": "falls by more than 1%", "flat": flat})
    row, _ = _apply(spec, ent, "flat", -3.0)
    assert row["point_forecast"] == -1.0


@pytest.mark.parametrize("phrase,bound", [
    ("reported EPS at least 5 cents above the consensus", 1.55),
    ("reported EPS more than 5 cents above the consensus", 1.56),
    ("reported EPS at least $0.05 above the consensus", 1.55),
    ("reported EPS at least 5 percent above the consensus", 1.58),
    ("reported EPS at least 5 cents below the consensus", 1.45),
])
def test_reference_offset_keeps_its_magnitude_and_reference(phrase, bound):
    ent = {"entity_id": "E", "currency": "USD/share", "consensus_eps": 1.5, "prior_year_q_eps": 2.0}
    spec = _spec("classification", "diluted EPS", ["up", "down"], "Forecast reported diluted EPS.", [ent],
                 {"up": phrase, "down": "otherwise"})
    row, _ = _apply(spec, ent, "up", 1.5)
    assert row["point_forecast"] == bound
    assert build_card(spec, ent).reference_name == "consensus_eps"


def test_unresolved_offset_is_not_replaced_by_a_zero_offset_direction():
    ent = {"entity_id": "X", "unit": "bp", "consensus": 4.25}
    spec = _spec("classification", "yield_change_bps", ["up", "down"], "Forecast the yield change in bp.", [ent],
                 {"up": "at least 10 bp above the consensus", "down": "otherwise"})
    assert build_card(spec, ent) is None
    row, codes = _apply(spec, ent, "up", 0.0)
    assert row["point_forecast"] == 0.0 and codes == []


@pytest.mark.parametrize("assertion", [
    "at least 5 cents above the consensus and at most 10 cents above it",
    "at least 5 cents above the consensus or the prior-year EPS",
])
def test_compound_reference_assertion_is_not_reduced_to_one_cut(assertion):
    ent = {"entity_id": "E", "currency": "USD/share", "consensus_eps": 1.5, "prior_year_q_eps": 2.0}
    spec = _spec("classification", "diluted EPS", ["up", "down"], "Forecast reported diluted EPS.", [ent],
                 {"up": assertion, "down": "otherwise"})
    assert build_card(spec, ent) is None


@pytest.mark.parametrize("assertion", ["not at least 10 bp", "no more than 10 bp and above 5 bp"])
def test_unresolved_absolute_condition_is_not_reduced_to_one_closed_cut(assertion):
    ent = {"entity_id": "X", "unit": "bp"}
    spec = _spec("classification", "yield_change_bps", ["yes", "no"], "Predict change in bp.", [ent],
                 {"yes": assertion, "no": "otherwise"})
    assert build_card(spec, ent) is None


def test_negated_reference_offset_does_not_become_a_positive_minimum():
    ent = {"entity_id": "E", "currency": "USD/share", "consensus_eps": 1.5}
    spec = _spec("classification", "diluted EPS", ["up", "down"], "Forecast reported diluted EPS.", [ent],
                 {"up": "no greater than 5 cents above the consensus", "down": "otherwise"})
    assert build_card(spec, ent) is None


def test_an_explicitly_excluded_band_endpoint_stays_excluded():
    ent = {"entity_id": "X", "unit": "percent"}
    spec = _spec("classification", "growth", ["band", "other"], "Predict growth in percent.", [ent],
                 {"band": "from -20% to below -5%", "other": "otherwise"})
    row, _ = _apply(spec, ent, "band", -5.0)
    assert row["point_forecast"] == -5.01


def test_at_most_a_fall_does_not_include_a_positive_rise():
    ent = {"entity_id": "X", "unit": "bp"}
    spec = _spec("classification", "yield_change_bps", ["small_fall", "other"], "Predict change in bp.", [ent],
                 {"small_fall": "falls by at most 10 bp", "other": "otherwise"})
    row, _ = _apply(spec, ent, "small_fall", 5.0)
    assert row["point_forecast"] == 0.0
    assert build_card(spec, ent).regions["other"].contains(Decimal("5"))


def test_shared_table_endpoint_does_not_reject_a_valid_submitted_label():
    ent = {"entity_id": "B", "unit": "bp", "up_condition": "value >= 10", "down_condition": "value <= 10"}
    spec = _spec("classification", "yield_bps", ["up", "down"], "Forecast the value in bp.", [ent],
                 {"up": "at least 10 bp", "down": "at most 10 bp"})
    row, codes = _apply(spec, ent, "up", 10.0)
    assert row["point_forecast"] == 10.0 and codes == []


def test_band_with_otherwise_keeps_both_rays_and_chooses_the_nearest():
    ent = {"entity_id": "X", "unit": "percent"}
    spec = _spec("classification", "event_return", ["flat", "move"], "Predict return in percent.", [ent],
                 {"flat": "between -1% and 1%", "move": "otherwise"})
    row, _ = _apply(spec, ent, "move", 0.004)
    assert row["point_forecast"] == 1.01
    card = build_card(spec, ent)
    assert card.regions["move"].contains(Decimal("-1.01"))
    assert not card.regions["move"].contains(Decimal("0"))


@pytest.mark.parametrize("a,b", [
    ("-1 <= Q <= 1", "Q >= -2"),
    ("-1 <= Q < 1", "Q > 1"),
    ("-3 <= Q <= -2", "2 <= Q <= 3"),
    ("Q < -1", "Q > 1"),
])
def test_otherwise_is_exactly_the_complement_even_with_overlaps_or_gaps(a, b):
    card = card_from_rules(["a", "b", "other"], {"a": a, "b": b, "other": "otherwise"}, {}, decimals=2)
    assert card is not None
    for point in map(Decimal, ["-4", "-3", "-2.5", "-2", "-1", "0", "1", "2", "2.5", "3", "4"]):
        explicit = card.regions["a"].contains(point) or card.regions["b"].contains(point)
        assert card.regions["other"].contains(point) is not explicit


def test_official_macro_direction_is_still_checked():
    public = Path(__file__).resolve().parents[2] / "track4-analysis-public/units"
    spec = load_task(public / "t4-macrorev-20240930-us6/task.json")
    ent = next(e for e in spec.entities if e["entity_id"] == "INDPRO_2024-07_20241017")
    row, _ = _apply(spec, ent, "down", 102.31)
    assert row["point_forecast"] == 102.3058
