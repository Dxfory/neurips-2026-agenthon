"""Offline checks for the split threshold parser and consistency gate."""
from decimal import Decimal

import pytest

from t4agent.task import TaskSpec
from t4agent.threshold_consistency import (
    apply_threshold_consistency,
    build_card,
    card_from_rules,
    label_of,
    parse_threshold_text,
)


def _card_pct(consensus: str, pct: str = "0.05"):
    return card_from_rules(
        ["beat", "miss", "inline"],
        {
            "beat": "Q > consensus_eps * (1 + threshold_pct)",
            "miss": "Q < consensus_eps * (1 - threshold_pct)",
            "inline": "otherwise",
        },
        {"consensus_eps": Decimal(consensus), "threshold_pct": Decimal(pct)},
        kind="relative_pct",
        reference=Decimal(consensus),
        reference_name="consensus_eps",
        decimals=2,
    )


def _spec(target_type, name, labels, prompt, entities, assertions=None):
    raw = {
        "task_id": "synthetic",
        "target": {"name": name, "type": target_type, "labels": labels,
                   "label_assertions": assertions or {}},
        "prompt": prompt,
        "entities": entities,
    }
    return TaskSpec(
        raw=raw,
        task_id="synthetic",
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


def test_five_percent_band_matches_the_published_boundary():
    card = _card_pct("1.50")
    assert label_of(Decimal("1.575"), card) == "inline"
    assert label_of(Decimal("1.425"), card) == "inline"
    assert label_of(Decimal("1.58"), card) == "beat"
    assert label_of(Decimal("1.42"), card) == "miss"
    assert label_of(1.575, card) == "inline"
    assert label_of(1.425, card) == "inline"
    # The binary product sits just above the published boundary. Decimal parsing does not.
    assert Decimal("1.50") * Decimal("1.05") == Decimal("1.575")
    assert (1.50 * 1.05) > 1.575


def test_negative_consensus_does_not_flip_the_band():
    card = _card_pct("-2.0")
    assert label_of(Decimal("-1.89"), card) == "beat"
    assert label_of(Decimal("-2.11"), card) == "miss"
    assert label_of(Decimal("-2.00"), card) == "inline"
    # The float product points the other way.
    assert (-2.0 * 1.05) == pytest.approx(-2.1)
    assert (-2.0 + abs(-2.0) * 0.05) == pytest.approx(-1.9)


def test_threshold_text_units():
    assert parse_threshold_text("more than 5% above").magnitude == Decimal("0.05")
    assert parse_threshold_text("beat by $0.05").kind == "absolute_dollars"
    assert parse_threshold_text("5 cents").magnitude == Decimal("0.05")
    assert parse_threshold_text("a move of 5bp").kind == "absolute_bp"


def test_gate_keeps_the_label_and_nudges_one_grid_step_past_the_boundary():
    ent = {
        "entity_id": "AAPL",
        "consensus_eps": 1.50,
        "threshold_pct": 0.05,
        "beat_condition": "reported_eps > consensus_eps * (1 + threshold_pct)",
        "miss_condition": "reported_eps < consensus_eps * (1 - threshold_pct)",
    }
    spec = _spec(
        "classification", "reported diluted EPS", ["beat", "miss", "inline"],
        "Will diluted EPS beat the consensus by 5%?", [ent],
    )
    row, codes = apply_threshold_consistency(
        spec, ent, {"label_rules": {"beat": "Q > 5", "miss": "Q < -5", "inline": "otherwise"}},
        {"label": "beat", "point_forecast": 1.50, "lo": 0.5, "hi": 2.5},
    )
    # A planner rule of Q > 5 must not define the band. The point moves to the next cent past 1.575.
    assert row["label"] == "beat"
    assert row["point_forecast"] == pytest.approx(1.58)
    assert row["lo"] == 0.5 and row["hi"] == 2.5
    assert codes == ["THRESHOLD_FIX"]
    assert label_of(row["point_forecast"], build_card(spec, ent, None)) == "beat"


def test_consistent_classification_and_regression_rows_stay_put():
    ent = {"entity_id": "IBM", "prior_year_q_eps": 1.53}
    spec = _spec(
        "classification", "diluted EPS", ["up", "down"],
        "Higher or lower than the prior-year quarter?", [ent],
    )
    original = {"label": "up", "point_forecast": 1.60, "lo": 0.3, "hi": 3.0}
    row, codes = apply_threshold_consistency(spec, ent, None, original)
    assert row == original and codes == []

    reg = _spec("regression", "yield change", [], "FOMC treasury yield change in bp", [
        {"entity_id": "UST10Y", "unit": "bp"},
    ])
    baseline = {"point_forecast": 0.0, "lo": -33.0, "hi": 33.0}
    kept, codes = apply_threshold_consistency(reg, reg.entities[0], None, baseline)
    assert kept == baseline and codes == []
    # A non-zero regression point is not forced back to zero.
    moved = {"point_forecast": 4.0, "lo": -10.0, "hi": 10.0}
    still, codes = apply_threshold_consistency(reg, reg.entities[0], None, moved)
    assert still["point_forecast"] == 4.0 and codes == []


def test_interval_cover_does_not_rebuild_a_width():
    ent = {"entity_id": "X", "consensus_eps": 1.50, "threshold_pct": 0.05,
           "beat_condition": "reported_eps > consensus_eps * (1 + threshold_pct)",
           "miss_condition": "reported_eps < consensus_eps * (1 - threshold_pct)"}
    spec = _spec("classification", "diluted EPS", ["beat", "miss", "inline"], "5% consensus", [ent])
    row, codes = apply_threshold_consistency(
        spec, ent, None, {"label": "inline", "point_forecast": 1.50, "lo": 1.60, "hi": 1.70},
    )
    assert row["point_forecast"] == pytest.approx(1.50)
    assert row["label"] == "inline"
    assert row["lo"] == pytest.approx(1.50)
    assert row["hi"] == pytest.approx(1.70)
    assert codes == ["INTERVAL_COVER"]


def test_probability_submission_is_not_rewritten_as_an_eps_level():
    # prior-year EPS 1.53 would otherwise pull a probability 0.8 up to 1.54 and widen [0, 1].
    ent = {"entity_id": "IBM", "prior_year_q_eps": 1.53, "unit": "probability"}
    spec = _spec(
        "classification", "eps_up_probability", ["up", "down"],
        "Probability that diluted EPS is higher than the prior-year quarter.", [ent],
    )
    original = {"label": "up", "point_forecast": 0.8, "lo": 0.0, "hi": 1.0}
    row, codes = apply_threshold_consistency(spec, ent, None, original)
    assert row == original and codes == []


def test_growth_rate_submission_is_not_rewritten_as_an_eps_level():
    # A 1% EPS growth rate is not the dollar level in prior_year_q_eps, so it does not become 1.54%.
    ent = {"entity_id": "IBM", "prior_year_q_eps": 1.53, "unit": "percent"}
    spec = _spec(
        "classification", "eps_growth", ["up", "down"],
        "YoY diluted EPS growth in percent versus the prior-year quarter.", [ent],
    )
    original = {"label": "up", "point_forecast": 1.0, "lo": -10.0, "hi": 10.0}
    row, codes = apply_threshold_consistency(spec, ent, None, original)
    assert row == original and codes == []


def test_large_move_lands_on_the_published_boundary():
    ent = {
        "entity_id": "INDPRO",
        "units": "index 2017=100, SA",
        "currency": "index level",
        "latest_precutoff_estimate": 102.3058,
    }
    spec = _spec(
        "classification", "next_estimate_revision_direction", ["up", "down"],
        "Will the next published estimate be revised up or down versus the latest estimate?", [ent],
    )
    # Down includes the cut. A point of 104, far past the step budget, is the cut itself.
    down, codes = apply_threshold_consistency(
        spec, ent, None, {"label": "down", "point_forecast": 104.0, "lo": 100.0, "hi": 110.0},
    )
    assert down["label"] == "down"
    assert down["point_forecast"] == pytest.approx(102.3058)
    assert down["lo"] == 100.0 and down["hi"] == 110.0
    assert codes == ["THRESHOLD_FIX"]
    assert label_of(down["point_forecast"], build_card(spec, ent, None)) == "down"
    # Up excludes the cut, so the same distance lands one published step above it.
    up, codes = apply_threshold_consistency(
        spec, ent, None, {"label": "up", "point_forecast": 100.0, "lo": 90.0, "hi": 110.0},
    )
    assert up["point_forecast"] == pytest.approx(102.3059)
    assert up["label"] == "up" and codes == ["THRESHOLD_FIX"]
    assert label_of(up["point_forecast"], build_card(spec, ent, None)) == "up"


def test_already_consistent_level_and_percent_rows_stay_put():
    level = {
        "entity_id": "INDPRO",
        "units": "index 2017=100, SA",
        "currency": "index level",
        "latest_precutoff_estimate": 102.3058,
    }
    spec = _spec(
        "classification", "next_estimate_revision_direction", ["up", "down"],
        "Revised up or down versus the latest estimate?", [level],
    )
    on_cut = {"label": "down", "point_forecast": 102.3058, "lo": 100.0, "hi": 110.0}
    row, codes = apply_threshold_consistency(spec, level, None, on_cut)
    assert row == on_cut and codes == []
    below = {"label": "down", "point_forecast": 100.0, "lo": 90.0, "hi": 110.0}
    row, codes = apply_threshold_consistency(spec, level, None, below)
    assert row == below and codes == []

    ent = {"entity_id": "AAPL", "pre_earnings_close_2024_01_31": 184.4, "flat_threshold_abn_pct": 1.0}
    assertions = {
        "positive_reaction": "one-day abnormal return greater than 1%",
        "negative_reaction": "one-day abnormal return less than -1%",
        "flat": "one-day abnormal return between -1% and 1%",
    }
    reaction = _spec(
        "classification", "earnings_reaction",
        ["positive_reaction", "negative_reaction", "flat"],
        "Predict the one-day abnormal return in percent.", [ent], assertions,
    )
    flat = {"label": "flat", "point_forecast": 0.0, "lo": -7.0, "hi": 7.0}
    row, codes = apply_threshold_consistency(reaction, ent, None, flat)
    assert row == flat and codes == []
    already_up = {"label": "positive_reaction", "point_forecast": 2.0, "lo": -7.0, "hi": 7.0}
    row, codes = apply_threshold_consistency(reaction, ent, None, already_up)
    assert row == already_up and codes == []
