"""Analytical checks for corpus-only, uncalibrated fallback forecasts."""
from __future__ import annotations

import datetime as dt
import importlib
import importlib.util
import math
import sys
from decimal import Decimal
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from t4agent.corpus import Corpus, Doc
from t4agent.stats import Series, StatForecast


def runtime():
    assert importlib.util.find_spec("t4agent.runtime_naive") is not None, "corpus-only fallback is missing"
    return importlib.import_module("t4agent.runtime_naive")


def corpus_for(values, header="amount", *, period=7):
    start = dt.date(2020, 1, 1)
    text = "date | " + header + "\n" + "\n".join(
        f"{start + dt.timedelta(days=i * period)} | {format(Decimal(str(v)), 'f')}" for i, v in enumerate(values)
    )
    return Corpus({"history": Doc("history", dt.date(2020, 12, 31), text, {})}, [])


def predict(values, target="amount", *, header="amount", horizon=28, period=7, ent=None, allow_own=False):
    return runtime().entity_forecast(
        corpus_for(values, header, period=period), ["history"], ent or {"entity_id": "amount"},
        target, horizon, {"history"}, allow_own=allow_own,
    )


@pytest.mark.parametrize("values,last", [
    ([2, 4, 3, 6, 4, 8], 8),
    ([-2, -4, -3, -6, -4, -8], -8),
])
def test_next_value_is_latest_with_analytical_one_step_width(values, last):
    result = predict(values)
    assert isinstance(result, StatForecast)
    assert result.value == last
    assert result.method == "runtime last"
    # The five squared observed steps sum to 34; central normal 90% z is 1.6448536269514722.
    assert result.q90 == pytest.approx(1.6448536269514722 * math.sqrt(34 / 5))
    assert result.n_backtest == 0 and result.mae is None
    assert result.q90 == runtime().series_half_width(result.series, "amount", 28, {"entity_id": "amount"})
    description = result.describe().lower()
    assert "theoretical" in description
    assert "best" not in description and "backtest" not in description


@pytest.mark.parametrize("period,horizon,steps", [(1, 7, 5), (7, 28, 4), (30, 60, 2)])
def test_change_forecast_stays_zero_with_calendar_horizon_conversion(period, horizon, steps):
    result = predict([10, 20, 30, 40, 50, 60], "amount_change", period=period, horizon=horizon)
    assert result.value == 0
    assert result.method == "runtime zero"
    assert result.q90 == pytest.approx(1.6448536269514722 * 10 * math.sqrt(steps))


def test_named_next_level_keeps_one_step_but_own_level_uses_horizon():
    named = predict([10, 20, 30, 40, 50, 60], period=1, horizon=28)
    own = predict([10, 20, 30, 40, 50, 60], header="value", period=1, horizon=28, allow_own=True)
    assert named.value == own.value == 60
    assert named.q90 == pytest.approx(1.6448536269514722 * 10)
    assert own.q90 == pytest.approx(1.6448536269514722 * 10 * math.sqrt(20))


@pytest.mark.parametrize("value,resolution", [(0, 1), (10, 1), (1.25, .01), (-1.25, .01), (.0000001, .0000001)])
def test_flat_series_uses_numeric_precision_not_a_calibrated_default(value, resolution):
    result = predict([value] * 6)
    assert result.value == value
    assert result.q90 == pytest.approx(1.6448536269514722 * resolution, rel=1e-12, abs=0)
    assert result.q90 > 0


@pytest.mark.parametrize("header,values,expected", [
    ("yield (%)", [1, 1.01, 1.02, 1.03, 1.04, 1.05], 1.6448536269514722),
    ("yield (bps)", [1, 2, 3, 4, 5, 6], 1.6448536269514722),
    ("yield (%)", [40, 41, 42, 43, 44, 45], 164.48536269514722),
])
def test_basis_point_scale_respects_explicit_source_units(header, values, expected):
    result = predict(values, "yield_change_bps", header=header, horizon=7, ent={"entity_id": "yield", "unit": "bps"})
    assert result.value == 0
    assert result.q90 == pytest.approx(expected)


def test_basis_point_next_level_converts_forecast_as_well_as_width():
    result = predict([1, 1.01, 1.02, 1.03, 1.04, 1.05], "yield_bps", header="yield (%)", ent={"entity_id": "yield"})
    assert result.value == pytest.approx(105)
    assert result.q90 == pytest.approx(1.6448536269514722)


def test_growth_column_is_next_growth_value_not_change_of_growth():
    result = predict([2, 3, 4, 5, 6, 7], "amount_growth", header="amount growth")
    assert result.value == 7
    assert result.method == "runtime last"
    assert result.q90 == pytest.approx(1.6448536269514722)


def test_ratio_change_uses_numerator_steps_at_starting_denominators():
    text = "date | net | OI | net_%OI\n" + "\n".join(
        f"2020-01-{1 + 7*i:02} | {10 * (i+1)} | {100 * (i+1)} | 10" for i in range(4)
    ) + "\n2020-01-29 | 50 | 500 | 10\n2020-02-05 | 60 | 600 | 10"
    corpus = Corpus({"history": Doc("history", dt.date(2020, 12, 31), text, {})}, [])
    result = runtime().entity_forecast(corpus, ["history"], {"entity_id": "net_%OI"}, "position change", 7, {"history"})
    assert result.value == 0
    assert result.method == "runtime zero"
    # Printed ratios are flat; target changes are 10, 5, 10/3, 2.5, 2 percentage points.
    assert result.q90 == pytest.approx(1.6448536269514722 * math.sqrt((100 + 25 + 100 / 9 + 6.25 + 4) / 5))


def test_flat_ratio_precision_is_converted_to_target_units():
    series = Series("history", "net_%OI", [10.0] * 6, [], 7, True, [1000.0] * 6, 100.0)
    width = runtime().series_half_width(series, "net change", 7, {})
    assert width == pytest.approx(.16448536269514722)


def test_unavailable_or_invalid_series_does_not_invent_a_scale():
    assert predict([1, 2, 3]) is None
    assert predict([1, 2, 3, 4, 5, 6], header="unrelated") is None
    for values, denom in [([], None), ([1], None), ([1, float("nan")], None), ([1, float("inf")], None), ([1, 2], [0, 1])]:
        series = Series("history", "amount", values, [], 7, True, denom)
        assert runtime().series_half_width(series, "amount change", 7, {}) is None


def test_positive_own_level_for_relative_target_uses_zero_percent_change():
    result = predict([100, 110, 121, 133.1, 146.41, 161.051], "percent_change", header="value", horizon=7, allow_own=True)
    assert result.value == 0
    assert result.series.kind == "own-rel"
    assert result.q90 == pytest.approx(1.6448536269514722 * 100 * math.log(1.1))


def test_changing_external_fitted_constants_and_selected_methods_cannot_change_results(monkeypatch):
    from t4agent import stats

    before = predict([2, 4, 3, 6, 4, 8])
    change_before = predict([10, 20, 30, 40, 50, 60], "amount_change")

    def forbidden(*args, **kwargs):
        raise AssertionError("runtime fallback called calibrated or selected statistics")

    for name in ("forecast", "rw_k", "_inflation", "_change_methods", "_ratio_change_forecast", "revert_share", "_resolution"):
        monkeypatch.setattr(stats, name, forbidden)
    for name in ("_RW_K", "_RW_N", "_RW_H", "MIN_SWITCH", "MIN_BACKTEST", "CHANGE_SWITCH", "RATIO_SWITCH", "LEVEL_REVERT_DAILY", "_NEXT_METHODS"):
        monkeypatch.setattr(stats, name, object())
    assert predict([2, 4, 3, 6, 4, 8]) == before
    assert predict([10, 20, 30, 40, 50, 60], "amount_change") == change_before


def test_translating_or_rescaling_task_values_has_analytical_effect():
    base = predict([2, 4, 3, 6, 4, 8])
    shifted = predict([102, 104, 103, 106, 104, 108])
    scaled = predict([20, 40, 30, 60, 40, 80])
    assert shifted.value == base.value + 100
    assert shifted.q90 == pytest.approx(base.q90)
    assert scaled.value == base.value * 10
    assert scaled.q90 == pytest.approx(base.q90 * 10)
