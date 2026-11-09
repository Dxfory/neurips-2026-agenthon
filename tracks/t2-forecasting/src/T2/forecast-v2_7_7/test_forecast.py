"""Behavioral controls for the v2.7.4 numerical forecaster. No text or API access."""
import importlib.util
import json
import shutil
import sys
import tempfile
import tomllib
import unittest
import warnings
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
UNITS = HERE.parent / "official_repo" / "track2-forecasting-public" / "units"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


fc = load(HERE / "forecast.py", "numeric_v3_test")


def legacy(**overrides):
    """Settings of the pre-v2.6 engine (v2.4 width, v2.5 post-processing) with overrides; later keys win."""
    return {**fc.V2_6_SETTINGS, **fc.V2_5_SETTINGS, **fc.V2_4_SETTINGS, **overrides}


def v25(**overrides):
    """Settings reproducing v2.5.0 with overrides."""
    return {**fc.V2_6_SETTINGS, **fc.V2_5_SETTINGS, **overrides}


class Controls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mixed = fc.load_task(UNITS / "t2-F3-divergence-2014")
        cls.settings = fc.Settings(draws=256)

    def test_zero_means_disabled(self):
        s = fc.Settings(draws=256, vol_sd_rates=0.0)
        self.assertEqual(s.overlay_for("rates_daily"), 0.0)

    def test_overlay_does_not_advance_base_noise_stream(self):
        off, _ = fc.predict(self.mixed, fc.Settings(draws=256, vol_sd=0.0))
        partial, _ = fc.predict(
            self.mixed, fc.Settings(draws=256, vol_sd_rates=0.0, vol_sd_fx=0.35)
        )
        np.testing.assert_array_equal(off[:, 0], partial[:, 0])

    def test_daily_horizon_key_is_the_step_when_last_print_is_asof(self):
        task = fc.load_task(UNITS / "t2-F1-eur-range-2017")
        self.assertEqual(task.cadence, "daily")
        self.assertTrue(np.all(task.anchors == 0))
        self.assertTrue(np.all(task.steps == np.asarray(task.horizons)))

    def test_monthly_steps_are_not_the_horizon_key(self):
        task = fc.load_task(UNITS / "t2-F1-cpi-glidepath-2023")
        self.assertEqual(task.cadence, "monthly")
        self.assertTrue(np.all(task.steps > 0))
        self.assertLess(int(task.steps.max()), 24)
        self.assertFalse(np.array_equal(task.steps.ravel(), np.asarray(task.horizons)))

    def test_log_return_anchor_is_zero(self):
        task = fc.load_task(UNITS / "t2-F1-ai-mom-2024")
        _, model = fc.predict(task, fc.Settings(draws=200))
        self.assertEqual(model.transforms[0].kind, "log_return")
        self.assertEqual(float(model.spots[0]), 0.0)
        self.assertTrue(np.all(task.anchors == 0))
        self.assertTrue(np.all(task.steps == np.asarray(task.horizons)))

    def test_fx_level_is_log_level_without_relying_on_a_crash(self):
        self.assertEqual(fc.transform_for(self.mixed, "EUR").kind, "log_level")
        self.assertEqual(fc.transform_for(self.mixed, "UST_10Y").kind, "level")

    def test_transfer_floor_is_not_narrower_than_v1_control(self):
        task = fc.load_task(UNITS / "t2-F2-brl-transfer-2013")
        _, wide = fc.predict(task, fc.Settings(draws=200))
        _, old = fc.predict(task, fc.v1_settings(draws=200))
        self.assertTrue(wide.audit["assets"]["BRL"]["transfer"])
        self.assertGreaterEqual(float(wide.scales[0]), float(old.scales[0]) - 1e-12)
        self.assertEqual(wide.audit["bootstrap_scope"], "wiped; no fresh target")

    def test_fresh_card_keeps_its_residual_library(self):
        _, model = fc.predict(self.mixed, self.settings)
        self.assertEqual(model.audit["bootstrap_scope"], "all fresh targets")
        self.assertGreater(model.audit["bootstrap_rows"], 20)
        self.assertIsNone(model.bootstrap_columns)

    def test_fit_failure_still_returns_a_finite_grid(self):
        real = fc.fit_base

        def boom(*args, **kwargs):
            raise RuntimeError("forced")

        fc.fit_base = boom
        try:
            samples, model = fc.predict(self.mixed, fc.Settings(draws=200))
        finally:
            fc.fit_base = real
        self.assertEqual(samples.shape[0], 200)
        self.assertEqual(samples.shape[1:], (len(self.mixed.assets), len(self.mixed.horizons)))
        self.assertTrue(np.isfinite(samples).all())
        self.assertTrue(model.audit["fallback"])

    def test_loader_failure_still_writes_three_files_and_exits_zero(self):
        unit = UNITS / "t2-F1-eur-range-2017"
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "forecast.parquet"
            real = fc.load_task

            def boom(*args, **kwargs):
                raise fc.ContractError("forced loader failure")

            fc.load_task = boom
            try:
                code = fc.main(
                    ["forecast", "--panels", str(unit), "--text", str(unit / "text"), "--out", str(out), "--draws", "200"]
                )
            finally:
                fc.load_task = real
            self.assertEqual(code, 0)
            self.assertTrue(fc.files_ready(out))

    def test_monthly_publication_lag_keeps_a_positive_trend(self):
        task = fc.load_task(UNITS / "t2-F1-cpi-glidepath-2023")
        _, model = fc.predict(task, self.settings)
        self.assertFalse(model.audit["assets"]["CPI_ALL"]["transfer"])
        self.assertGreater(model.trend[0, 0], 0.0)
        self.assertEqual(model.audit["settings"]["block_length"], 7)
        self.assertEqual(model.audit["settings"]["vol_sd"], 0.0)


def write_unit(root: Path, rows: list[tuple], *, target_type: str, asof: str, n_draws_min=200) -> None:
    assets = []
    for asset, _, _ in rows:
        text = "" if asset is None else str(asset)
        if text.strip() and text not in assets:
            assets.append(text)
    if not assets:
        assets = ["AAA"]
    horizons = [1, 5]
    (root / "card.toml").write_text(
        "\n".join(
            [
                "[task]",
                'id = "synthetic"',
                "",
                "[provenance]",
                f'data_cutoff = "{asof}"',
                "",
                "[targets]",
                f"asset_ids = [{', '.join(repr(a) for a in assets)}]",
                f"horizons = [{', '.join(str(h) for h in horizons)}]",
                f'target_type = "{target_type}"',
                'value_unit = "native"',
                "",
                "[metadata]",
                'category = "synthetic"',
                "",
            ]
        )
    )
    (root / "forecast_spec.json").write_text(
        json.dumps(
            {
                "n_draws_min": n_draws_min,
                "targets": {"asset_ids": assets, "horizons": horizons, "target_type": target_type},
            }
        )
    )
    pd.DataFrame(rows, columns=["asset", "date", "value"]).to_parquet(root / "panel.parquet", index=False)


class ContractRepairs(unittest.TestCase):
    def test_lagged_log_return_keeps_the_horizon_key(self):
        dates = pd.bdate_range("2023-01-03", "2024-01-03")
        rows = [("AAA", d.strftime("%Y-%m-%d"), 0.001) for d in dates]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_unit(root, rows, target_type="log_return", asof="2024-02-02")
            task = fc.load_task(root)
        self.assertEqual(task.cadence, "daily")
        self.assertTrue(np.all(task.anchors == 0))
        self.assertTrue(np.array_equal(task.steps, np.array([[1, 5]])))

    def test_lagged_level_uses_the_declared_horizon(self):
        dates = pd.bdate_range("2023-01-03", "2024-01-03")
        rows = [("AAA", d.strftime("%Y-%m-%d"), 1.2) for d in dates]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_unit(root, rows, target_type="level", asof="2024-02-02")
            task = fc.load_task(root)
        self.assertTrue(np.all(task.anchors == 0))
        self.assertTrue(np.array_equal(task.steps, np.array([[1, 5]])))

    def test_missing_duplicate_does_not_erase_the_observation(self):
        rows = [
            ("AAA", "2024-01-02", None),
            ("AAA", "2024-01-02", 1.5),
            ("AAA", "2024-01-03", 1.6),
            ("AAA", "2024-01-04", 1.7),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_unit(root, rows, target_type="level", asof="2024-01-04")
            task = fc.load_task(root)
        self.assertEqual(float(task.histories["AAA"].loc["2024-01-02"]), 1.5)

    def test_blank_asset_and_missing_date_are_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_unit(
                root,
                [("AAA", "2024-01-03", 1.0), ("", "2024-01-04", 1.1)],
                target_type="level",
                asof="2024-01-04",
            )
            with self.assertRaises(fc.ContractError):
                fc.load_task(root)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_unit(root, [("AAA", None, 1.0)], target_type="level", asof="2024-01-04")
            with self.assertRaises(fc.ContractError):
                fc.load_task(root)

    def test_fractional_draw_floor_is_refused(self):
        rows = [("AAA", "2024-01-04", 1.0)]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_unit(root, rows, target_type="level", asof="2024-01-04", n_draws_min=200.5)
            with self.assertRaises(fc.ContractError):
                fc.load_task(root)

    def test_controls_reject_bool_and_nonfinite_values(self):
        with self.assertRaises(ValueError):
            fc.Settings(draws=256, center_guard=1)
        with self.assertRaises(ValueError):
            fc.Settings(draws=256, seed=True)
        with self.assertRaises(ValueError):
            fc.Settings(draws=256, vol_sd=float("nan"))

    def test_correlation_shrinkage_uses_only_the_trailing_window(self):
        rng = np.random.default_rng(1)
        old = pd.DataFrame(rng.normal(size=(1500, 2)), columns=["a", "b"])
        gap = pd.DataFrame(np.nan, index=range(656), columns=["a", "b"])
        recent = pd.DataFrame(rng.normal(size=(100, 2)), columns=["a", "b"])
        frame = pd.concat([old, gap, recent], ignore_index=True)
        tail = pd.concat([gap, recent], ignore_index=True)
        np.testing.assert_allclose(fc.regularized_correlation(frame), fc.regularized_correlation(tail))

    def test_atomic_output_is_readable_by_another_user(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "forecast.parquet"
            fc._replace(path, b"abc")
            if sys.platform == "win32":
                self.assertTrue(path.is_file() and path.stat().st_size == 3)
            else:
                self.assertEqual(path.stat().st_mode & 0o777, 0o644)

    def test_sahm_walks_authored_periods_and_records_the_m0_table(self):
        task = fc.load_task(UNITS / "t2-F1-sahm-watch-2024")
        self.assertEqual(tuple(int(x) for x in task.steps.ravel()), (7, 8))
        self.assertTrue(any("published M0 monthly steps" in w for w in task.warnings))

    def test_v33_daily_defaults_and_monthly_width_are_separate(self):
        s = fc.Settings(draws=256)
        self.assertEqual(s.width_scale, 0.9)
        self.assertEqual((s.horizon_scale_power, s.horizon_scale_anchor, s.horizon_scale_widen_only), (0.3, 63.0, True))
        self.assertEqual((s.vol_blend, s.vol_beta_base, s.vol_beta_slope), (1.0, 1.0, 0.0))
        self.assertEqual((s.monthly_vol_blend, s.transfer_vol_blend), (0.8, 0.8))
        self.assertEqual(s.f3_width_profile, "v2.5")
        self.assertEqual(fc.V2_4_SETTINGS["horizon_scale_power"], 0.05)
        self.assertEqual(s.drift_cap_horizon, 63)
        self.assertEqual(s.for_cadence("monthly").width_scale, 1.15)
        self.assertTrue(s.use_trailing_drift)
        self.assertEqual(fc.v1_settings(draws=256).width_scale, 1.0)
        self.assertEqual(fc.v1_settings(draws=256).horizon_scale_power, 0.0)
        self.assertEqual(fc.v1_settings(draws=256).drift_cap_horizon, 0)
        self.assertFalse(fc.v1_settings(draws=256).use_trailing_drift)

    def test_daily_center_is_one_slope_across_horizons(self):
        task = fc.load_task(UNITS / "t2-F3-divergence-2014")
        model = fc.fit_base(task, fc.Settings(draws=256))
        slopes = (model.center - model.spots[:, None]) / task.steps
        np.testing.assert_allclose(slopes, np.repeat(slopes[:, :1], len(task.horizons), axis=1), atol=1e-14, rtol=0)
        for i, asset in enumerate(task.assets):
            self.assertLessEqual(abs(slopes[i, 0]), model.audit["assets"][asset]["daily_drift_limit"] + 1e-14)

    def test_disabling_trailing_drift_keeps_the_center_guard(self):
        task = fc.load_task(UNITS / "t2-F3-divergence-2014")
        model = fc.fit_base(task, fc.Settings(draws=256, use_trailing_drift=False, center_cap=0.0))
        self.assertTrue(np.any(model.trend + model.reversion != 0))
        np.testing.assert_array_equal(model.center, np.repeat(model.spots[:, None], len(task.horizons), axis=1))

    def test_horizon_width_changes_spread_once_and_preserves_center(self):
        # Engine property: the M0-t mixture share does not depend on the engine's width.
        task = fc.load_task(UNITS / "t2-F3-divergence-2014")
        # An F3 card: f3_width_profile "v2.5" so the explicit width settings below are the ones the engine runs.
        # An F3 card: no sharpening scale and the "v2.5" F3 profile, so the explicit width settings are what runs.
        # A Treasury card: the v2.7 forward shift (horizon-dependent) is pinned off so that only the width moves.
        plain, base = fc.predict(task, fc.Settings(draws=512, horizon_scale_power=0.0, mix_weight=0.0, sharpen=False,
                                                   horizon_scale_anchor=21.0, horizon_scale_widen_only=False,
                                                   f3_width_profile="v2.5", rates_forward_add=0.0))
        scaled, model = fc.predict(task, fc.Settings(draws=512, horizon_scale_power=0.1, mix_weight=0.0, sharpen=False,
                                                     horizon_scale_anchor=21.0, horizon_scale_widen_only=False,
                                                     f3_width_profile="v2.5", rates_forward_add=0.0))
        np.testing.assert_array_equal(model.center, base.center)
        for i, trans in enumerate(model.transforms):
            left, right = plain[:, i], scaled[:, i]
            if trans.kind == "log_level":
                left, right = np.log(left), np.log(right)
            actual = right - model.center[i]
            expected = (left - base.center[i]) * (task.steps[i] / 21.0)**0.1
            np.testing.assert_allclose(actual, expected, atol=1e-14, rtol=0)

    def test_daily_width_controls_do_not_change_monthly_predictions(self):
        task = fc.load_task(UNITS / "t2-F1-cpi-glidepath-2023")
        expected, _ = fc.predict(task, fc.Settings(draws=256))
        actual, _ = fc.predict(task, fc.Settings(draws=256, width_scale=0.2,
                              horizon_scale_power=-0.5, drift_cap_horizon=1000))
        np.testing.assert_array_equal(actual, expected)

    def test_new_controls_reject_invalid_values(self):
        for overrides in ({"drift_cap_horizon": True}, {"drift_cap_horizon": -1},
                          {"horizon_scale_power": float("nan")}, {"monthly_width_scale": 0.1},
                          {"mix_weight": 0.95}, {"mix_df": 2.0}, {"f4_width_scale": 0.1}, {"pair_match": 1}):
            with self.assertRaises(ValueError):
                fc.Settings(**overrides)


# ---------------------------------------------------------------- v1.1 post-processing
# References outside the image: the backtest's M0 reproduction and the previous release.
M0_REFERENCE = load(HERE.parent / "backtest" / "m0.py", "backtest_m0_reference")
V1_REFERENCE = load(HERE.parent / "forecast-v1" / "forecast.py", "forecast_v1_reference")
CARDS = sorted(p.parent for p in UNITS.glob("*/card.toml"))
# The official commit of 2026-10-18 (0ddb889) adds 22 short-horizon practice units, ids ending in -ps1. Some are
# FOMC-week Monday cards, where v2.7.3's profile fires by design. Tests that pin which of the original practice cards
# a release changes keep to the 104 cards they were written for (ORIGINAL_CARDS). ShortHorizonPracticeSet covers
# the 22.
ORIGINAL_CARDS = [c for c in CARDS if not c.name.endswith("-ps1")]
PS1_CARDS = [c for c in CARDS if c.name.endswith("-ps1")]


def pair_statistic(samples):
    flat = samples.reshape(len(samples), -1)
    return float(np.mean(np.abs(flat[:, 0] - flat[:, 1]) ** 0.5))


class PostProcessing(unittest.TestCase):
    def test_embedded_m0_is_bit_exact_with_the_backtest_reproduction(self):
        for unit in CARDS:
            expected, _ = M0_REFERENCE.predict(unit)
            got, _ = fc.m0_reference(unit, tomllib.loads((unit / "card.toml").read_text()))
            np.testing.assert_array_equal(got, expected, err_msg=unit.name)

    def test_every_switch_off_reproduces_forecast_v1(self):
        for unit in CARDS:
            old, _ = V1_REFERENCE.predict(V1_REFERENCE.load_task(unit), V1_REFERENCE.Settings(draws=512))
            new, _ = fc.predict(fc.load_task(unit), fc.Settings(draws=512, **legacy(**fc.POST_PROCESSING_OFF)))
            np.testing.assert_array_equal(new, old, err_msg=unit.name)

    def test_two_cell_grids_match_the_m0_pair_statistic(self):
        checked = 0
        for unit in CARDS:
            task = fc.load_task(unit)
            if task.cell_count != 2:
                continue
            # The v2.7 forward shift comes after the pairing and moves a Treasury pair statistic by the difference
            # of its two cells' shifts (ForwardCenter checks that it is small); the matching itself is exact.
            samples, model = fc.predict(task, fc.Settings(draws=512, rates_forward_add=0.0, pair_match=True, fomc_mean_shift=False))
            reference, _ = fc.m0_reference(unit, task.card)
            self.assertTrue(model.audit["post_processing"]["pair_match"]["reached"], unit.name)
            self.assertLess(abs(pair_statistic(samples) / pair_statistic(reference) - 1), 1e-9, unit.name)
            checked += 1
        self.assertGreaterEqual(checked, 10)

    def test_pairing_keeps_each_cell_marginal_up_to_the_polish(self):
        task = fc.load_task(UNITS / "t2-F1-eur-range-2017")
        before, _ = fc.predict(task, fc.Settings(draws=512, pair_match=False))
        after, model = fc.predict(task, fc.Settings(draws=512, pair_match=True))
        info = model.audit["post_processing"]["pair_match"]
        self.assertEqual(info["shift_fraction"], 0.0)
        for cell in range(2):
            old, new = np.sort(before.reshape(512, -1)[:, cell]), np.sort(after.reshape(512, -1)[:, cell])
            np.testing.assert_allclose(new, old, rtol=0, atol=1e-3 * old.std())

    def test_shock_width_applies_to_daily_f4_only(self):
        for name, expected in (("t2-F4-aud-gfc-2008", 1.1), ("t2-F1-eur-range-2017", None),
                               ("t2-F4-covid-nfp-2020", None)):
            _, model = fc.predict(fc.load_task(UNITS / name), fc.Settings(draws=512))
            self.assertEqual(model.audit["post_processing"]["shock_width"], expected, name)

    def test_mixture_skips_transfer_cards_and_uses_monthly_values_on_monthly_cards(self):
        _, model = fc.predict(fc.load_task(UNITS / "t2-F2-brl-transfer-2013"), fc.Settings(draws=512))
        self.assertEqual(model.audit["post_processing"]["mixture"], "not applied")
        _, model = fc.predict(fc.load_task(UNITS / "t2-F1-cpi-glidepath-2023"), fc.Settings(draws=512))
        self.assertEqual(model.audit["post_processing"]["mixture"],
                         {"engine_draws": 256, "m0t_draws": 256, "df": 5.0, "drift": 1.0, "scale": 0.8})
        _, model = fc.predict(fc.load_task(UNITS / "t2-F1-cpi-glidepath-2023"),
                              fc.Settings(draws=512, monthly_m0_path=False))
        self.assertEqual(model.audit["post_processing"]["mixture"], "not applied")
        _, model = fc.predict(fc.load_task(UNITS / "t2-F3-front-loaded-hikes-2022"), fc.Settings(draws=512))
        self.assertEqual(model.audit["post_processing"]["mixture"]["m0t_draws"], 154)  # v2.6: 0.3 of 512 draws

    def test_covariance_root_is_unique_and_sign_stable(self):
        cov = np.array([[2.0, 0.6], [0.6, 1.0]])
        np.testing.assert_allclose(fc._covariance_root(cov), np.linalg.cholesky(cov))
        singular = np.array([[1.0, 1.0], [1.0, 1.0]])
        root = fc._covariance_root(singular)
        np.testing.assert_allclose(root @ root.T, singular, atol=1e-12)
        self.assertTrue(np.all(root[np.argmax(np.abs(root), axis=0), [0, 1]] >= 0))

    def test_settings_recorded_like_the_backtest_harness_expects(self):
        task = fc.load_task(UNITS / "t2-F4-aud-gfc-2008")
        requested = fc.Settings(draws=512)
        _, model = fc.predict(task, requested)
        self.assertEqual(model.audit["requested_settings"], fc.asdict(requested))
        self.assertEqual(model.audit["settings"], fc.asdict(requested.for_cadence(task.cadence)))
        self.assertFalse(model.audit["fallback"])


class Robustness(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="v11-test-"))

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def copy(self, name):
        target = self.tmp / name
        shutil.copytree(UNITS / name, target)
        return target

    def edit_spec(self, unit, change):
        spec = json.loads((unit / "forecast_spec.json").read_text())
        change(spec)
        (unit / "forecast_spec.json").write_text(json.dumps(spec))

    def run_cli(self, unit, *extra):
        out = self.tmp / "out"
        code = fc.main(["forecast", "--panels", str(unit), "--out", str(out / "forecast.parquet"), "--draws", "512", *extra])
        return code, out, pd.read_parquet(out / "forecast.parquet"), json.loads((out / "forecast_meta.json").read_text())

    def test_monthly_card_in_panel_steps_without_mapping_is_forecast_normally(self):
        unit = self.copy("t2-F1-cpi-glidepath-2023")
        self.edit_spec(unit, lambda s: (s["targets"].pop("observation_periods"), s["targets"].update(horizons=[8, 9])))
        card = unit / "card.toml"
        card.write_text("\n".join(line.replace("[140, 160]", "[8, 9]") if line.strip().startswith("horizons") else line
                                  for line in card.read_text().splitlines()) + "\n")
        task = fc.load_task(unit)
        self.assertEqual(task.steps.tolist(), [[8, 9]])
        samples, model = fc.predict(task, fc.Settings(draws=512))
        self.assertFalse(model.audit["fallback"])
        self.assertLess(abs(float(np.median(samples)) / float(task.histories["CPI_ALL"].iloc[-1]) - 1), 0.05)

    def test_business_day_monthly_horizons_without_mapping_follow_the_m0_table(self):
        unit = self.copy("t2-F1-cpi-glidepath-2023")
        self.edit_spec(unit, lambda s: s["targets"].pop("observation_periods"))
        self.assertEqual(fc.load_task(unit).steps.tolist(), [[8, 9]])

    def test_card_wins_when_the_spec_disagrees(self):
        unit = self.copy("t2-F1-eur-range-2017")
        self.edit_spec(unit, lambda s: s["targets"].update(horizons=[5, 10]))
        code, out, frame, meta = self.run_cli(unit)
        self.assertEqual(code, 0)
        self.assertEqual(meta["horizons"], [126, 189])
        self.assertEqual(sorted(frame.horizon.unique().tolist()), [126, 189])
        self.assertTrue(all(fc.validate_output(unit, out).values()))

    def test_asof_disagreement_binds_to_the_card(self):
        unit = self.copy("t2-F1-eur-range-2017")
        code, _, _, meta = self.run_cli(unit, "--asof", "2017-09-06")
        self.assertEqual(code, 0)
        self.assertEqual(meta["asof"], "2017-09-07")

    def test_loader_failure_writes_an_m0_centred_forecast_that_passes_the_gates(self):
        unit = self.copy("t2-F2-some-pain-2022")
        self.edit_spec(unit, lambda s: s.update(n_draws_min=7))
        code, out, frame, _ = self.run_cli(unit)
        self.assertEqual(code, 0)
        reference, _ = fc.m0_reference(unit, tomllib.loads((unit / "card.toml").read_text()))
        self.assertLess(abs(frame.value.median() - float(np.median(reference))), 0.25)
        self.assertTrue(all(fc.validate_output(unit, out).values()))

    def test_engine_failure_falls_back_to_m0_not_a_wide_grid(self):
        task = fc.load_task(UNITS / "t2-F4-cpi-friday-2022")
        with mock.patch.object(fc, "fit_base", side_effect=RuntimeError("forced")):
            samples, model = fc.predict(task, fc.Settings(draws=512))
        self.assertTrue(model.audit["fallback"])
        self.assertEqual(model.audit["engine"], "m0-fallback")
        reference, _ = fc.m0_reference(task.root, task.card)
        self.assertLess(abs(samples.std() / reference.std() - 1), 0.1)
        self.assertLess(abs(float(samples.mean()) - float(reference.mean())), 0.05)


# ---------------------------------------------------------------- v1.2 review fixes
def renamed_unit(root: Path, name: str, new_id: str) -> Path:
    unit = root / name
    shutil.copytree(UNITS / name, unit)
    card = unit / "card.toml"
    card.write_text(card.read_text().replace(f'"{name}"', f'"{new_id}"', 1))
    return unit


def column_correlation(samples):
    return float(np.corrcoef(samples.reshape(len(samples), -1).T)[0, 1])


class MonthlyM0Steps(unittest.TestCase):
    """Review P1: a monthly card outside the four published ones must not walk its key."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="v12-p1-"))

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_unknown_monthly_card_counts_months_and_keeps_the_pair_coherent(self):
        unit = renamed_unit(self.tmp, "t2-F1-cpi-glidepath-2023", "t2-F1-cpi-unpublished")
        task = fc.load_task(unit)
        _, ref = fc.m0_reference(unit, task.card)
        self.assertEqual(ref["steps"][1] - ref["steps"][0], 1.0)
        self.assertTrue(np.all(ref["steps"] < 21))
        month_steps = {("CPI_ALL", h): float(s) for h, s in zip(task.horizons, task.steps[0])}
        _, walked = fc.m0_reference(unit, task.card, month_steps=month_steps)
        self.assertEqual(walked["steps"].tolist(), task.steps[0].astype(float).tolist())
        samples, model = fc.predict(task, fc.Settings(draws=512, pair_match=True))
        record = model.audit["post_processing"]["pair_match"]
        # v2.2: the embedded M0 walks the card's observation months and a fresh monthly card may move
        # centres or blend, as a fresh daily card does, so M0's pair value is reached.
        self.assertTrue(record["reached"])
        self.assertIn(record["method"], {"repair", "shift", "m0-blend"})
        self.assertGreater(column_correlation(samples), 0.9)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "forecast.parquet"
            fc.write_output(task, samples, out, model, 0)
            self.assertTrue(all(fc.validate_output(unit, out.parent).values()))

    def test_monthly_keys_stated_in_panel_steps_are_walked_as_stated(self):
        unit = renamed_unit(self.tmp, "t2-F1-cpi-glidepath-2023", "t2-F1-cpi-steps")
        spec = json.loads((unit / "forecast_spec.json").read_text())
        spec["targets"].update(horizons=[8, 9])
        spec["targets"].pop("observation_periods")
        (unit / "forecast_spec.json").write_text(json.dumps(spec))
        card = unit / "card.toml"
        card.write_text("\n".join(line.replace("[140, 160]", "[8, 9]") if line.strip().startswith("horizons") else line
                                  for line in card.read_text().splitlines()) + "\n")
        task = fc.load_task(unit)
        _, ref = fc.m0_reference(unit, task.card)
        self.assertEqual(ref["steps"].tolist(), [8.0, 9.0])
        self.assertEqual(task.steps.tolist(), [[8, 9]])

    def test_pairing_is_skipped_when_m0_steps_disagree_with_the_task(self):
        unit = renamed_unit(self.tmp, "t2-F1-cpi-glidepath-2023", "t2-F1-cpi-disagree")
        task = fc.load_task(unit)
        declared = lambda h, last, spacing, asof: int(h)  # noqa: E731 - the v1.1 behaviour
        # With the v2.2 monthly path the observation months fix M0's steps; the refusal is checked on
        # the path that still estimates them.
        with mock.patch.object(fc, "_m0_step_count", side_effect=declared):
            samples, model = fc.predict(task, fc.Settings(draws=512, monthly_m0_path=False, pair_match=True))
            untouched, _ = fc.predict(task, fc.Settings(draws=512, pair_match=False, monthly_m0_path=False))
        record = model.audit["post_processing"]["pair_match"]
        self.assertEqual(record["method"], "skipped: M0 steps disagree with the task")
        np.testing.assert_array_equal(samples, untouched)


class PairFailurePath(unittest.TestCase):
    """Review P2a: an unreachable target is met by a minimal M0 blend or left alone."""

    def synthetic(self):
        rng = np.random.default_rng(3)
        ours = np.column_stack([rng.normal(0.0, 0.1, 512), rng.normal(0.0, 0.1, 512)])
        m0 = rng.standard_normal((500, 2)) * 2.0  # far more pair dispersion than ours can reach
        ref = {"anchor": np.zeros(2), "steps": np.ones(2), "mu": np.zeros(2), "cov": np.eye(2) * 4.0}
        return SimpleNamespace(fingerprint="synthetic"), ours.reshape(512, 1, 2), m0.reshape(500, 1, 2), ref

    def test_fresh_card_blends_in_the_fewest_m0_draws_and_matches_exactly(self):
        task, ours, m0, ref = self.synthetic()
        matched, record = fc._pair_match(task, ours, m0, ref, 0, fresh=True)
        self.assertTrue(record["reached"])
        self.assertEqual(record["method"], "m0-blend")
        self.assertTrue(0.0 < record["m0_blend"] <= 1.0)
        self.assertLess(abs(pair_statistic(matched) / pair_statistic(m0) - 1), 1e-9)
        self.assertEqual(matched.shape, ours.shape)

    def test_monthly_or_transfer_card_keeps_its_draws_when_unreachable(self):
        task, ours, m0, ref = self.synthetic()
        matched, record = fc._pair_match(task, ours, m0, ref, 0, fresh=False)
        self.assertFalse(record["reached"])
        self.assertEqual(record["method"], "unmatched")
        np.testing.assert_array_equal(matched, ours)

    def test_match_pair_returns_its_input_when_nothing_reaches(self):
        _, ours, m0, _ = self.synthetic()
        flat = ours.reshape(512, 2)
        out, record = fc.match_pair(flat, pair_statistic(m0), 0.0, np.random.default_rng(1), allow_shift=True)
        self.assertFalse(record["reached"])
        self.assertIs(out, flat)

    def test_public_two_cell_cards_record_how_they_matched(self):
        for unit in CARDS:
            task = fc.load_task(unit)
            if task.cell_count != 2:
                continue
            _, model = fc.predict(task, fc.Settings(draws=512, pair_match=True))
            self.assertIn(model.audit["post_processing"]["pair_match"]["method"], {"repair", "shift", "m0-blend"},
                          unit.name)


class MonthlyMappingConflicts(unittest.TestCase):
    """Review P2b: conflicting or invalid month mappings are resolved or refused, not guessed."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="v12-p2b-"))
        self.unit = self.tmp / "unit"
        shutil.copytree(UNITS / "t2-F1-cpi-glidepath-2023", self.unit)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def card_periods(self, periods):
        card = self.unit / "card.toml"
        card.write_text(card.read_text().replace(
            "leakage_checked  = true", "leakage_checked  = true\nobservation_periods = " + json.dumps(periods), 1))

    def spec_periods(self, periods):
        spec = json.loads((self.unit / "forecast_spec.json").read_text())
        spec["targets"]["observation_periods"] = periods
        (self.unit / "forecast_spec.json").write_text(json.dumps(spec))

    def test_conflicting_mappings_use_the_cards_own(self):
        self.card_periods(["2024-03", "2024-04"])
        task = fc.load_task(self.unit)
        self.assertEqual(task.steps.tolist(), [[10, 11]])
        self.assertTrue(any("the card's own mapping is used" in w for w in task.warnings))

    def test_invalid_spec_mapping_without_a_card_mapping_is_refused(self):
        self.spec_periods(["2024-13", "2024-02"])
        with self.assertRaises(fc.ContractError):
            fc.load_task(self.unit)
        out = self.tmp / "out"
        code = fc.main(["forecast", "--panels", str(self.unit), "--out", str(out / "forecast.parquet"), "--draws", "512"])
        self.assertEqual(code, 0)
        last = float(pd.read_parquet(self.unit / "macro_monthly.parquet").query("asset == 'CPI_ALL'").value.iloc[-1])
        self.assertLess(abs(pd.read_parquet(out / "forecast.parquet").value.median() / last - 1), 0.05)
        self.assertTrue(all(fc.validate_output(self.unit, out).values()))

    def test_invalid_card_mapping_is_refused(self):
        self.card_periods(["2024-13", "2024-02"])
        with self.assertRaises(fc.ContractError):
            fc.load_task(self.unit)


# ---------------------------------------------------------------- v2.1 sharpening
V1_2_REFERENCE = load(HERE.parent / "forecast-v1_2" / "forecast.py", "forecast_v1_2_reference")


def m0_z(samples, ref):
    flat = samples.reshape(len(samples), -1)
    return (flat - (ref["anchor"] + ref["steps"] * ref["mu"])) / np.sqrt(np.diag(ref["cov"]))


class Sharpening(unittest.TestCase):
    def test_sharpening_off_reproduces_v1_2(self):
        for unit in CARDS:
            old, _ = V1_2_REFERENCE.predict(V1_2_REFERENCE.load_task(unit), V1_2_REFERENCE.Settings(draws=512))
            new, model = fc.predict(fc.load_task(unit), fc.Settings(draws=512, **legacy(sharpen=False, monthly_m0_path=False,
                                                                                       transfer_m0="off")))
            np.testing.assert_array_equal(new, old, err_msg=unit.name)
            self.assertEqual(model.audit["post_processing"]["sharpen"], "not applied", unit.name)

    def test_default_settings_are_the_final_profile(self):
        defaults = fc.Settings()
        for name, pair in fc.SHARPEN_PROFILES["final"].items():
            self.assertEqual(getattr(defaults, name), pair, name)
        self.assertTrue(defaults.sharpen)
        self.assertEqual((defaults.sharpen_f1, defaults.sharpen_f4), ((0.0, 1.1), (0.0, 1.0)))
        self.assertEqual((defaults.mix_weight, defaults.vol_sd), (0.3, 0.0))
        self.assertEqual(fc.SHARPEN_PROFILES["prospective"]["sharpen_f4"], (0.0, 1.0))
        self.assertEqual((defaults.sharpen_f2, defaults.sharpen_f3), ((0.0, 1.2), (0.0, 1.2)))
        # v2.7.1: the engine's shock-family width is back to v2.6.0's 1.1; k applies after it.
        self.assertEqual((defaults.f4_widen, defaults.f4_width_scale), (True, 1.1))
        for name in ("prospective", "balanced", "dev-leaning", "random-leaning"):
            self.assertEqual(fc.SHARPEN_PROFILES[name]["f4_width_scale"], 1.3, name)
        self.assertEqual((defaults.transfer_m0, defaults.transfer_m0_scale), ("geo", 0.85))
        balanced = fc.SHARPEN_PROFILES["balanced"]
        self.assertEqual({k: v for k, v in balanced.items() if k not in ("sharpen_f4", "f4_width_scale")},
                         {k: v for k, v in fc.SHARPEN_PROFILES["final"].items() if k not in ("sharpen_f4", "f4_width_scale")})
        self.assertEqual(balanced["sharpen_f4"], (0.0, 1.25))

    def test_every_fresh_daily_card_is_scaled_by_its_family_k_and_nothing_is_reweighted(self):
        """v2.6.1/v2.7.1: alpha 0 everywhere (a scale, no resampling toward M0's centre): F1 k 1.1, F2 and F3 k 1.2,
        F4 k 1.0 (v2.7.1); monthly cards keep their own (0.5, 1.0); transfer cards are not sharpened."""
        applied = 0
        for unit in ORIGINAL_CARDS:
            task = fc.load_task(unit)
            _, model = fc.predict(task, fc.Settings(draws=256, monthly_price_vol_switch=False))  # E30 is RegimeSwitches'
            record = model.audit["post_processing"]["sharpen"]
            if fc._fresh_daily(task, model) and fc.sharpen_for(fc.Settings(), task.family) == (0.0, 1.0):
                self.assertEqual(record, "not applied", unit.name)  # F4 in v2.7.1: the identity pair is skipped
                applied += 1
            elif fc._fresh_daily(task, model):
                self.assertEqual((record["alpha"], record["k"]), fc.sharpen_for(fc.Settings(), task.family), unit.name)
                self.assertEqual(record["alpha"], 0.0, unit.name)
                self.assertNotIn("reweight", record, unit.name)
                applied += 1
            elif task.cadence == "monthly":
                self.assertEqual((record["alpha"], record["k"]), (0.5, 1.0), unit.name)
            else:
                self.assertEqual(record, "not applied", unit.name)
        self.assertGreaterEqual(applied, 90)

    def test_family_pairs_never_reach_transfer_or_monthly_cards(self):
        strong = fc.Settings(draws=256, sharpen_f1=(1.0, 0.8), sharpen_f2=(1.0, 0.8), sharpen_f4=(1.0, 0.8))
        for name in ("t2-F2-brl-transfer-2013", "t2-F1-cpi-glidepath-2023", "t2-F4-covid-nfp-2020"):
            task = fc.load_task(UNITS / name)
            samples, model = fc.predict(task, strong)
            baseline, _ = fc.predict(task, replace(strong, sharpen=False))
            np.testing.assert_array_equal(samples, baseline, err_msg=name)
        _, model = fc.predict(fc.load_task(UNITS / "t2-F2-brl-transfer-2013"), strong)
        self.assertEqual(model.audit["post_processing"]["sharpen"], "not applied")

    def test_m0_sharpen_without_alpha_or_k_is_the_identity(self):
        task = fc.load_task(UNITS / "t2-F4-jpy-carry-2007")
        samples, _ = fc.predict(task, fc.Settings(draws=512, sharpen=False))
        _, ref = fc.m0_reference(task.root, task.card)
        out, info = fc.m0_sharpen(samples, ref, 0.0, 1.0, np.random.default_rng(0))
        np.testing.assert_array_equal(out, samples)
        self.assertNotIn("reweight", info)

    def test_reweighting_keeps_whole_draws_and_moves_mass_toward_m0(self):
        task = fc.load_task(UNITS / "t2-F3-dollar-squeeze-2020")  # 20 cells: rows must survive intact
        samples, _ = fc.predict(task, fc.Settings(draws=512, sharpen=False))
        _, ref = fc.m0_reference(task.root, task.card)
        out, info = fc.m0_sharpen(samples, ref, 1.0, 1.0, np.random.default_rng(0))
        rows = {row.tobytes() for row in samples.reshape(512, -1)}
        self.assertTrue(all(row.tobytes() in rows for row in out.reshape(512, -1)))
        self.assertLess(np.mean(np.abs(m0_z(out, ref))), np.mean(np.abs(m0_z(samples, ref))))
        self.assertTrue(0.0 < info["effective_draws"] <= 512.0)
        again, _ = fc.m0_sharpen(samples, ref, 1.0, 1.0, np.random.default_rng(0))
        np.testing.assert_array_equal(out, again)

    def test_k_scales_deviations_about_each_cell_median(self):
        task = fc.load_task(UNITS / "t2-F1-eur-range-2017")
        samples, _ = fc.predict(task, fc.Settings(draws=512, sharpen=False))
        _, ref = fc.m0_reference(task.root, task.card)
        out, _ = fc.m0_sharpen(samples, ref, 0.0, 1.5, np.random.default_rng(0))
        flat, new = samples.reshape(512, -1), out.reshape(512, -1)
        np.testing.assert_allclose(np.median(new, axis=0), np.median(flat, axis=0), rtol=0, atol=1e-12)
        np.testing.assert_allclose(new - np.median(new, axis=0), 1.5 * (flat - np.median(flat, axis=0)),
                                   rtol=1e-12, atol=1e-12)

    def test_unusable_m0_moments_skip_the_reweight(self):
        ref = {"anchor": np.zeros(2), "steps": np.ones(2), "mu": np.zeros(2), "cov": np.diag([1.0, 0.0])}
        samples = np.random.default_rng(1).standard_normal((256, 1, 2))
        out, info = fc.m0_sharpen(samples, ref, 1.0, 1.0, np.random.default_rng(0))
        self.assertEqual(info["reweight"], "skipped: unusable M0 moments")
        np.testing.assert_array_equal(out, samples)

    def test_sharpen_settings_are_validated_and_json_lists_become_tuples(self):
        self.assertEqual(fc.Settings(sharpen_f2=[0.5, 1.2]).sharpen_f2, (0.5, 1.2))
        for bad in ((-0.1, 1.0), (0.5, 0.1), (0.5, 3.5), (0.5,), (True, 1.0), (float("nan"), 1.0), "0.5,1.0"):
            with self.assertRaises(ValueError, msg=repr(bad)):
                fc.Settings(sharpen_f4=bad)
        with self.assertRaises(ValueError):
            fc.Settings(sharpen=1)

    def test_cli_profiles_select_the_documented_settings(self):
        unit = UNITS / "t2-F4-jpy-carry-2007"
        profiles = (("default", fc.Settings(draws=512)),
                    ("v1.2", fc.Settings(draws=512, **legacy(sharpen=False, monthly_m0_path=False, transfer_m0="off"))),
                    ("v2.1", fc.Settings(draws=512, **legacy(monthly_m0_path=False, sharpen_f4=(1.0, 1.1), transfer_m0="off"))),
                    ("v2.2", fc.Settings(draws=512, **legacy(sharpen_f4=(1.0, 1.1), transfer_m0="off"))),
                    ("v2.3", fc.Settings(draws=512, **legacy(transfer_m0="off"))),
                    ("v2.4", fc.Settings(draws=512, **legacy())),
                    ("v2.5", fc.Settings(draws=512, **v25())),
                    ("v2.6", fc.Settings(draws=512, **fc.V2_6_SETTINGS)),
                    ("v2.6.1", fc.Settings(draws=512, **fc.V2_6_1_SETTINGS)),
                    ("v2.7", fc.Settings(draws=512, **fc.V2_7_SETTINGS)),
                    ("prospective", fc.Settings(draws=512, **fc.SHARPEN_PROFILES["prospective"])),
                    ("balanced", fc.Settings(draws=512, **fc.SHARPEN_PROFILES["balanced"])),
                    ("dev-leaning", fc.Settings(draws=512, **fc.SHARPEN_PROFILES["dev-leaning"])),
                    ("random-leaning", fc.Settings(draws=512, **fc.SHARPEN_PROFILES["random-leaning"])))
        with tempfile.TemporaryDirectory() as tmp:
            for profile, settings in profiles:
                out = Path(tmp) / profile / "forecast.parquet"
                code = fc.main(["forecast", "--panels", str(unit), "--out", str(out), "--draws", "512",
                                "--seed", "0", "--profile", profile])
                self.assertEqual(code, 0, profile)
                expected, _ = fc.predict(fc.load_task(unit), settings)
                np.testing.assert_array_equal(pd.read_parquet(out).value.to_numpy(), expected.reshape(-1),
                                              err_msg=profile)


# ---------------------------------------------------------------- v2.2 monthly path
V2_1_REFERENCE = load(HERE.parent / "forecast-v2_1" / "forecast.py", "forecast_v2_1_reference")
MONTHLY = ("t2-F1-cpi-glidepath-2023", "t2-F1-sahm-watch-2024", "t2-F4-covid-nfp-2020", "t2-F4-cpi-vintage-2022")


class MonthlyPath(unittest.TestCase):
    def test_monthly_path_off_reproduces_v2_1(self):
        for unit in CARDS:
            old, _ = V2_1_REFERENCE.predict(V2_1_REFERENCE.load_task(unit), V2_1_REFERENCE.Settings(draws=512))
            new, _ = fc.predict(fc.load_task(unit), fc.Settings(draws=512, **legacy(monthly_m0_path=False, sharpen_f4=(1.0, 1.1),
                                                                                   transfer_m0="off")))
            np.testing.assert_array_equal(new, old, err_msg=unit.name)

    def test_only_monthly_cards_change_against_v2_1(self):
        changed = []
        for unit in CARDS:
            old, _ = V2_1_REFERENCE.predict(V2_1_REFERENCE.load_task(unit), V2_1_REFERENCE.Settings(draws=512))
            new, _ = fc.predict(fc.load_task(unit), fc.Settings(draws=512, **legacy(sharpen_f4=(1.0, 1.1), transfer_m0="off")))
            if not np.array_equal(new, old):
                changed.append(unit.name)
        self.assertEqual(sorted(changed), sorted(MONTHLY))

    def test_monthly_cards_record_mixture_sharpening_and_pairing(self):
        for name in MONTHLY:
            task = fc.load_task(UNITS / name)
            samples, model = fc.predict(task, fc.Settings(draws=512, pair_match=True, monthly_price_vol_switch=False))
            post = model.audit["post_processing"]
            self.assertEqual(post["mixture"]["m0t_draws"], 256, name)
            self.assertEqual(post["sharpen"]["reweight"], "applied", name)
            if task.cell_count == 2:
                self.assertTrue(post["pair_match"]["reached"], name)
                reference, _ = fc.m0_reference(task.root, task.card)
                self.assertLess(abs(pair_statistic(samples) / pair_statistic(reference) - 1), 1e-9, name)
            with tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp) / "forecast.parquet"
                fc.write_output(task, samples, out, model, 0)
                self.assertTrue(all(fc.validate_output(task.root, out.parent).values()), name)

    def test_published_step_table_still_wins_over_observation_months(self):
        task = fc.load_task(UNITS / "t2-F1-sahm-watch-2024")
        month_steps = {("UNRATE", h): float(s) for h, s in zip(task.horizons, task.steps[0])}
        _, ref = fc.m0_reference(task.root, task.card, month_steps=month_steps)
        self.assertEqual(ref["steps"].tolist(), [8.0, 9.0])

    def test_stale_monthly_card_is_left_alone(self):
        with tempfile.TemporaryDirectory() as tmp:
            unit = Path(tmp) / "unit"
            shutil.copytree(UNITS / "t2-F4-cpi-vintage-2022", unit)
            panel = pd.read_parquet(unit / "macro_monthly.parquet")
            panel = panel[pd.to_datetime(panel["date"]) < pd.Timestamp("2022-01-01")]
            panel.to_parquet(unit / "macro_monthly.parquet", index=False)
            spec = json.loads((unit / "forecast_spec.json").read_text())
            spec["targets"]["observation_periods"] = ["2022-03"]
            (unit / "forecast_spec.json").write_text(json.dumps(spec))
            task = fc.load_task(unit)
            samples, model = fc.predict(task, fc.Settings(draws=512))
            self.assertTrue(model.audit["assets"]["CPI_ALL"]["transfer"])
            self.assertEqual(model.audit["post_processing"]["mixture"], "not applied")
            self.assertEqual(model.audit["post_processing"]["sharpen"], "not applied")
            self.assertNotIn("transfer_m0", model.audit["post_processing"])

    def test_monthly_settings_are_validated(self):
        for field, bad in (("monthly_mix_weight", 0.95), ("monthly_mix_df", 2.0), ("monthly_mix_drift", -0.1),
                           ("monthly_mix_scale", 0.1), ("monthly_sharpen_alpha", 2.5)):
            with self.assertRaises(ValueError, msg=field):
                fc.Settings(**{field: bad})
        with self.assertRaises(ValueError):
            fc.Settings(monthly_m0_path="yes")


# ---------------------------------------------------------------- v2.3 F4 width
V2_2_REFERENCE = load(HERE.parent / "forecast-v2_2" / "forecast.py", "forecast_v2_2_reference")


class F4Width(unittest.TestCase):
    def test_f4_pair_at_1_1_reproduces_v2_2(self):
        for unit in CARDS:
            old, _ = V2_2_REFERENCE.predict(V2_2_REFERENCE.load_task(unit), V2_2_REFERENCE.Settings(draws=512))
            new, _ = fc.predict(fc.load_task(unit), fc.Settings(draws=512, **legacy(sharpen_f4=(1.0, 1.1), transfer_m0="off")))
            np.testing.assert_array_equal(new, old, err_msg=unit.name)

    def test_only_fresh_daily_f4_cards_change_against_v2_2(self):
        changed, expected = [], []
        for unit in ORIGINAL_CARDS:
            task = fc.load_task(unit)
            old, _ = V2_2_REFERENCE.predict(V2_2_REFERENCE.load_task(unit), V2_2_REFERENCE.Settings(draws=512))
            new, model = fc.predict(task, fc.Settings(draws=512, **legacy(transfer_m0="off")))
            if not np.array_equal(new, old):
                changed.append(unit.name)
            if fc._fresh_daily(task, model) and task.family.endswith("F4"):
                expected.append(unit.name)
                self.assertEqual((model.audit["post_processing"]["sharpen"]["alpha"],
                                  model.audit["post_processing"]["sharpen"]["k"]), (1.0, 1.25), unit.name)
        self.assertEqual(sorted(changed), sorted(expected))
        self.assertEqual(len(expected), 29)


# ---------------------------------------------------------------- v2.4 transfer cards
V2_3_REFERENCE = load(HERE.parent / "forecast-v2_3" / "forecast.py", "forecast_v2_3_reference")
TRANSFER = ("t2-F2-cnh-stress-2015", "t2-F2-cnh-tradewar-2018", "t2-F2-brl-transfer-2013",
            "t2-F2-inr-transfer-2013", "t2-F3-em-transfer-joint-2013")


def proxy_sd(task, model):
    """Per cell: the G10 proxies' median current scale times spot and sqrt(steps), computed independently."""
    out = []
    for a in task.assets:
        scales = []
        for name in model.audit["assets"][a]["scale_proxy"]:
            trans = fc.transform_for(task, name)
            values = fc.innovations(trans.state(task.context_histories[name]), trans.kind, task.cadence)
            scales.append(fc.scale_estimate(values.to_numpy(), task.cadence, fc.Settings().transfer_vol_blend)[0])
        out.append(float(np.median(scales)) * float(task.histories[a].iloc[-1]))
    return np.repeat(out, len(task.horizons))


class TransferM0(unittest.TestCase):
    def test_rule_off_reproduces_v2_3_and_geo_changes_only_transfer_cards(self):
        changed = []
        for unit in CARDS:
            task = fc.load_task(unit)
            old_samples, _ = V2_3_REFERENCE.predict(V2_3_REFERENCE.load_task(unit), V2_3_REFERENCE.Settings(draws=512))
            off, _ = fc.predict(task, fc.Settings(draws=512, **legacy(transfer_m0="off")))
            np.testing.assert_array_equal(off, old_samples, err_msg=unit.name)
            new, model = fc.predict(task, fc.Settings(draws=512, **legacy()))
            if np.array_equal(new, old_samples):
                self.assertNotIn("transfer_m0", model.audit["post_processing"], unit.name)
            else:
                changed.append(unit.name)
                self.assertEqual(model.audit["post_processing"]["transfer_m0"]["rule"], "geo", unit.name)
        self.assertEqual(sorted(changed), sorted(TRANSFER))

    def test_geo_draws_sit_on_m0_moments_with_the_geometric_mean_sd(self):
        for name in TRANSFER:
            task = fc.load_task(UNITS / name)
            samples, model = fc.predict(task, fc.Settings(draws=4096))
            _, ref = fc.m0_reference(task.root, task.card)
            flat = samples.reshape(len(samples), -1)
            m0_sd = np.sqrt(np.diag(ref["cov"]))
            expected = np.sqrt(m0_sd * proxy_sd(task, model) * np.sqrt(ref["steps"]))
            np.testing.assert_allclose(flat.mean(axis=0), ref["anchor"] + ref["steps"] * ref["mu"], rtol=1e-12,
                                       atol=1e-9, err_msg=name)
            np.testing.assert_allclose(flat.std(axis=0, ddof=1), expected, rtol=1e-9, err_msg=name)
            np.testing.assert_allclose(model.audit["post_processing"]["transfer_m0"]["sd_factor_vs_m0"],
                                       expected / m0_sd, rtol=1e-12, err_msg=name)

    def test_m0_rule_scales_m0_sd(self):
        task = fc.load_task(UNITS / "t2-F2-inr-transfer-2013")
        samples, model = fc.predict(task, fc.Settings(draws=4096, transfer_m0="m0", transfer_m0_scale=0.85))
        _, ref = fc.m0_reference(task.root, task.card)
        np.testing.assert_allclose(samples.reshape(4096, -1).std(axis=0, ddof=1), 0.85 * np.sqrt(np.diag(ref["cov"])),
                                   rtol=1e-9)
        self.assertEqual(model.audit["post_processing"]["transfer_m0"]["rule"], "m0")

    def test_joint_transfer_card_keeps_m0_correlation(self):
        task = fc.load_task(UNITS / "t2-F3-em-transfer-joint-2013")
        samples, _ = fc.predict(task, fc.Settings(draws=4096))
        _, ref = fc.m0_reference(task.root, task.card)
        sd = np.sqrt(np.diag(ref["cov"]))
        np.testing.assert_allclose(np.corrcoef(samples.reshape(4096, -1), rowvar=False), ref["cov"] / np.outer(sd, sd),
                                   atol=0.05)

    def test_cell_without_a_fresh_proxy_keeps_m0_sd(self):
        with tempfile.TemporaryDirectory() as tmp:
            unit = Path(tmp) / "unit"
            shutil.copytree(UNITS / "t2-F2-inr-transfer-2013", unit)
            panel = pd.read_parquet(unit / "g10_fx_daily.parquet")
            panel[pd.to_datetime(panel["date"]) < pd.Timestamp("2013-04-01")].to_parquet(unit / "g10_fx_daily.parquet",
                                                                                          index=False)
            task = fc.load_task(unit)
            samples, model = fc.predict(task, fc.Settings(draws=512))
            self.assertEqual(model.audit["assets"]["INR"]["scale_proxy"], [])
            self.assertEqual(model.audit["post_processing"]["transfer_m0"]["sd_factor_vs_m0"], [1.0])
            self.assertTrue(np.isfinite(samples).all())

    def test_transfer_output_passes_the_gates(self):
        for name in TRANSFER:
            task = fc.load_task(UNITS / name)
            samples, model = fc.predict(task, fc.Settings(draws=512))
            with tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp) / "forecast.parquet"
                fc.write_output(task, samples, out, model, 0)
                self.assertTrue(all(fc.validate_output(task.root, out.parent).values()), name)

    def test_cli_profile_v2_3_reproduces_v2_3_on_a_transfer_card(self):
        unit = UNITS / "t2-F2-brl-transfer-2013"
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "forecast.parquet"
            code = fc.main(["forecast", "--panels", str(unit), "--out", str(out), "--draws", "512", "--seed", "0",
                            "--profile", "v2.3"])
            self.assertEqual(code, 0)
            old, _ = V2_3_REFERENCE.predict(V2_3_REFERENCE.load_task(unit), V2_3_REFERENCE.Settings(draws=512))
            np.testing.assert_array_equal(pd.read_parquet(out).value.to_numpy(), old.reshape(-1))

    def test_transfer_settings_are_validated(self):
        for bad in ("on", "", 1, None):
            with self.assertRaises(ValueError, msg=repr(bad)):
                fc.Settings(transfer_m0=bad)
        for bad in (0.4, 2.1, float("nan")):
            with self.assertRaises(ValueError, msg=repr(bad)):
                fc.Settings(transfer_m0_scale=bad)


# ---------------------------------------------------------------- v2.5 width-by-horizon and volatility settings
V2_4_REFERENCE = load(HERE.parent / "forecast-v2_4" / "forecast.py", "forecast_v2_4_reference")
V2_5_REFERENCE = load(HERE.parent / "forecast-v2_5" / "forecast.py", "forecast_v2_5_reference")
V2_6_REFERENCE = load(HERE.parent / "forecast-v2_6" / "forecast.py", "forecast_v2_6_reference")
V2_6_1_REFERENCE = load(HERE.parent / "forecast-v2_6_1" / "forecast.py", "forecast_v2_6_1_reference")


class V25Settings(unittest.TestCase):
    def test_v2_4_settings_reproduce_v2_4_on_every_card(self):
        for unit in CARDS:
            old_samples, _ = V2_4_REFERENCE.predict(V2_4_REFERENCE.load_task(unit), V2_4_REFERENCE.Settings(draws=512))
            new, _ = fc.predict(fc.load_task(unit), fc.Settings(draws=512, **legacy()))
            np.testing.assert_array_equal(new, old_samples, err_msg=unit.name)

    def test_cli_profile_v2_4_reproduces_v2_4(self):
        unit = UNITS / "t2-F1-eur-range-2017"
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "forecast.parquet"
            code = fc.main(["forecast", "--panels", str(unit), "--out", str(out), "--draws", "512", "--seed", "0",
                            "--profile", "v2.4"])
            self.assertEqual(code, 0)
            old, _ = V2_4_REFERENCE.predict(V2_4_REFERENCE.load_task(unit), V2_4_REFERENCE.Settings(draws=512))
            np.testing.assert_array_equal(pd.read_parquet(out).value.to_numpy(), old.reshape(-1))

    def test_earlier_profiles_still_reproduce_their_releases(self):
        unit = UNITS / "t2-F4-aud-gfc-2008"
        for profile, reference, kwargs in (("v2.3", V2_3_REFERENCE, {}), ("v2.2", V2_2_REFERENCE, {}),
                                           ("v1.2", V1_2_REFERENCE, {})):
            with tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp) / "forecast.parquet"
                code = fc.main(["forecast", "--panels", str(unit), "--out", str(out), "--draws", "512", "--seed", "0",
                                "--profile", profile])
                self.assertEqual(code, 0, profile)
                old, _ = reference.predict(reference.load_task(unit), reference.Settings(draws=512, **kwargs))
                np.testing.assert_array_equal(pd.read_parquet(out).value.to_numpy(), old.reshape(-1), err_msg=profile)

    def test_widen_only_anchor_leaves_cells_at_or_under_the_anchor_unchanged(self):
        plain = fc.Settings(draws=512, horizon_scale_anchor=63.0, horizon_scale_power=0.0)
        widen = fc.Settings(draws=512, horizon_scale_anchor=63.0, horizon_scale_power=0.3,
                            horizon_scale_widen_only=True)
        both = fc.Settings(draws=512, horizon_scale_anchor=63.0, horizon_scale_power=0.3,
                           horizon_scale_widen_only=False)
        for name, expect in (("t2-F4-aud-gfc-2008", {21}), ("t2-F3-dots-vs-oil-2014", {63, 126})):
            task = fc.load_task(UNITS / name)
            self.assertEqual(set(int(s) for s in np.asarray(task.steps).ravel()), expect, name)
            a, b, c = (fc.fit_base(task, s) for s in (plain, widen, both))
            steps = np.asarray(task.steps, float)
            # widen-only: cells at or under 63 BD unchanged, longer cells widened by (steps / 63) ** 0.3
            np.testing.assert_allclose(b.spread, a.spread * np.maximum(steps / 63.0, 1.0) ** 0.3, rtol=1e-12,
                                       err_msg=name)
            # two-sided: cells under 63 BD narrowed, the 63-BD cell unchanged, longer cells widened the same way
            np.testing.assert_allclose(c.spread, a.spread * (steps / 63.0) ** 0.3, rtol=1e-12, err_msg=name)
            self.assertTrue(np.all(c.spread[steps < 63] < a.spread[steps < 63]), name)

    def test_widen_only_anchor_widens_longer_cells_by_the_power(self):
        task = fc.load_task(UNITS / "t2-F1-eur-range-2017")
        a = fc.fit_base(task, fc.Settings(draws=512, horizon_scale_anchor=63.0, horizon_scale_power=0.0))
        b = fc.fit_base(task, fc.Settings(draws=512, horizon_scale_anchor=63.0, horizon_scale_power=0.3,
                                          horizon_scale_widen_only=True))
        factor = np.maximum(np.asarray(task.steps, float) / 63.0, 1.0) ** 0.3
        np.testing.assert_allclose(b.spread, a.spread * factor, rtol=1e-12)

    def test_new_settings_are_validated(self):
        for bad in (0.5, 1001.0, float("nan")):
            with self.assertRaises(ValueError, msg=repr(bad)):
                fc.Settings(horizon_scale_anchor=bad)
        for bad in (1, "yes", None):
            with self.assertRaises(ValueError, msg=repr(bad)):
                fc.Settings(horizon_scale_widen_only=bad)
        for name in ("monthly_vol_blend", "transfer_vol_blend"):
            for bad in (-0.1, 1.5, float("nan")):
                with self.assertRaises(ValueError, msg=f"{name}={bad!r}"):
                    fc.Settings(**{name: bad})

    def test_defaults_change_only_the_fresh_daily_cards(self):
        """E11 and E13 evaluated daily fresh cards only: monthly cards and all-transfer cards must equal v2.4.0."""
        changed, expected = [], []
        for unit in ORIGINAL_CARDS:
            task = fc.load_task(unit)
            old_samples, _ = V2_4_REFERENCE.predict(V2_4_REFERENCE.load_task(unit), V2_4_REFERENCE.Settings(draws=512))
            new, model = fc.predict(task, fc.Settings(draws=512, pair_match=True, monthly_price_vol_switch=False))  # v2.7.3's pair_match default (off) is FomcWeekProfile's
            if not np.array_equal(new, old_samples):
                changed.append(unit.name)
            transfer = all(model.audit["assets"][a]["transfer"] for a in task.assets)
            if task.cadence == "daily" and not transfer:
                expected.append(unit.name)
                if task.family.strip().upper().endswith("F4"):
                    # v2.7.1, as v2.6.0: (0.0, 1.0) is the identity, so the sharpening step is skipped; engine width 1.1
                    self.assertEqual(model.audit["post_processing"]["sharpen"], "not applied", unit.name)
                    self.assertEqual(model.audit["post_processing"]["shock_width"], 1.1, unit.name)
        self.assertEqual(sorted(changed), sorted(expected))
        self.assertEqual(len(expected), 95)

    def test_v2_5_settings_reproduce_v2_5_0_on_every_card(self):
        for unit in CARDS:
            old_samples, _ = V2_5_REFERENCE.predict(V2_5_REFERENCE.load_task(unit), V2_5_REFERENCE.Settings(draws=512))
            new, _ = fc.predict(fc.load_task(unit), fc.Settings(draws=512, **v25()))
            np.testing.assert_array_equal(new, old_samples, err_msg=unit.name)

    def test_v2_6_settings_and_profile_reproduce_v2_6_0(self):
        for unit in CARDS:
            old_samples, _ = V2_6_REFERENCE.predict(V2_6_REFERENCE.load_task(unit), V2_6_REFERENCE.Settings(draws=512))
            new, _ = fc.predict(fc.load_task(unit), fc.Settings(draws=512, **fc.V2_6_SETTINGS))
            np.testing.assert_array_equal(new, old_samples, err_msg=unit.name)
        unit = UNITS / "t2-F4-aud-gfc-2008"
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "forecast.parquet"
            self.assertEqual(fc.main(["forecast", "--panels", str(unit), "--out", str(out), "--draws", "512", "--seed", "0",
                                      "--profile", "v2.6"]), 0)
            old_samples, _ = V2_6_REFERENCE.predict(V2_6_REFERENCE.load_task(unit), V2_6_REFERENCE.Settings(draws=512))
            np.testing.assert_array_equal(pd.read_parquet(out).value.to_numpy(), old_samples.reshape(-1))

    def test_cli_profile_v2_5_reproduces_v2_5_0(self):
        unit = UNITS / "t2-F3-dots-vs-oil-2014"
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "forecast.parquet"
            code = fc.main(["forecast", "--panels", str(unit), "--out", str(out), "--draws", "512", "--seed", "0",
                            "--profile", "v2.5"])
            self.assertEqual(code, 0)
            old_samples, _ = V2_5_REFERENCE.predict(V2_5_REFERENCE.load_task(unit), V2_5_REFERENCE.Settings(draws=512))
            np.testing.assert_array_equal(pd.read_parquet(out).value.to_numpy(), old_samples.reshape(-1))

    def test_f3_width_profile_changes_only_f3_cards_and_is_validated(self):
        for unit in CARDS:
            task = fc.load_task(unit)
            a, _ = fc.predict(task, fc.Settings(draws=256))
            b, _ = fc.predict(task, fc.Settings(draws=256, f3_width_profile="v2.4"))
            if not task.family.strip().upper().endswith("F3"):
                np.testing.assert_array_equal(a, b, err_msg=unit.name)
        for bad in ("v2.3", "", None, 1):
            with self.assertRaises(ValueError, msg=repr(bad)):
                fc.Settings(f3_width_profile=bad)

    def test_monthly_and_transfer_paths_keep_the_v2_4_blend(self):
        monthly = fc.Settings(draws=512).for_cadence("monthly")
        self.assertEqual(monthly.vol_blend, 0.8)
        self.assertEqual(fc.Settings(draws=512).for_cadence("daily").vol_blend, 1.0)
        task = fc.load_task(UNITS / "t2-F2-brl-transfer-2013")
        _, model = fc.predict(task, fc.Settings(draws=512))
        self.assertEqual(model.audit["post_processing"]["transfer_m0"]["rule"], "geo")


class ForwardCenter(unittest.TestCase):
    """v2.7 (E21): half of the curve-implied forward increment on Treasury cells; off reproduces v2.6.1."""

    @staticmethod
    def expected_shifts(unit, task):
        """Independent reading of the unit's rates panel: 0.5 * (f(h, T) - y(T)) per Treasury cell, else 0."""
        rates = pd.read_parquet(unit / "rates_daily.parquet")
        rates["date"] = pd.to_datetime(rates["date"])
        curve = {}
        for asset, g in rates[rates.date <= pd.Timestamp(task.asof)].groupby("asset"):
            g = g.dropna(subset=["value"]).sort_values("date")
            tenor = fc.tenor_years(asset)
            if tenor is not None and len(g):
                curve[tenor] = float(g.value.iloc[-1])
        shifts = np.zeros((len(task.assets), len(task.horizons)))
        for ai, asset in enumerate(task.assets):
            tenor = fc.tenor_years(asset)
            if tenor is None or task.panel_for_asset.get(asset) != "rates_daily":
                continue
            for hi in range(len(task.horizons)):
                h = int(np.asarray(task.steps)[ai, hi]) / 252.0
                shifts[ai, hi] = 0.5 * (fc.forward_yield(curve, h, tenor) - curve[tenor])
        return shifts

    @staticmethod
    def treasury(unit, task):
        return (task.cadence == "daily" and task.target_type == "level" and (unit / "rates_daily.parquet").is_file()
                and any(fc.tenor_years(a) is not None and task.panel_for_asset.get(a) == "rates_daily" for a in task.assets))

    def test_off_reproduces_v2_6_1_on_every_card(self):
        for unit in CARDS:
            old_samples, _ = V2_6_1_REFERENCE.predict(V2_6_1_REFERENCE.load_task(unit), V2_6_1_REFERENCE.Settings(draws=512))
            new, _ = fc.predict(fc.load_task(unit), fc.Settings(draws=512, **fc.V2_6_1_SETTINGS))
            np.testing.assert_array_equal(new, old_samples, err_msg=unit.name)

    def test_cli_profile_v2_6_1_reproduces_v2_6_1(self):
        unit = UNITS / "t2-F1-conflicting-texts-2024"
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "forecast.parquet"
            self.assertEqual(fc.main(["forecast", "--panels", str(unit), "--out", str(out), "--draws", "512", "--seed", "0",
                                      "--profile", "v2.6.1"]), 0)
            old_samples, _ = V2_6_1_REFERENCE.predict(V2_6_1_REFERENCE.load_task(unit), V2_6_1_REFERENCE.Settings(draws=512))
            np.testing.assert_array_equal(pd.read_parquet(out).value.to_numpy(), old_samples.reshape(-1))

    def test_v2_7_shifts_treasury_cells_by_half_the_forward_increment_and_nothing_else(self):
        """v2.7.0 against v2.6.1 (V2_7_SETTINGS keeps v2.7.0's F4 width, which v2.7.1's default changes)."""
        changed = []
        for unit in CARDS:
            task = fc.load_task(unit)
            old_samples, _ = V2_6_1_REFERENCE.predict(V2_6_1_REFERENCE.load_task(unit), V2_6_1_REFERENCE.Settings(draws=512))
            new, model = fc.predict(task, fc.Settings(draws=512, **fc.V2_7_SETTINGS))
            if not self.treasury(unit, task):
                np.testing.assert_array_equal(new, old_samples, err_msg=unit.name)
                continue
            changed.append(unit.name)
            expected = self.expected_shifts(unit, task)  # exactly zero where the curve is flat over the tenor span
            for ai in range(len(task.assets)):
                for hi in range(len(task.horizons)):
                    np.testing.assert_allclose(new[:, ai, hi] - old_samples[:, ai, hi], expected[ai, hi], rtol=0, atol=1e-12,
                                               err_msg=f"{unit.name} {task.assets[ai]} {task.horizons[hi]}")
            # width, shape and dependence untouched: only the location moved
            np.testing.assert_allclose(new.std(axis=0), old_samples.std(axis=0), rtol=1e-12, err_msg=unit.name)
            info = model.audit["post_processing"]["rates_forward"]
            self.assertEqual(info["strength"], 0.5)
            self.assertEqual(len(info["cells"]),
                             sum(fc.tenor_years(a) is not None for a in task.assets) * len(task.horizons), unit.name)
            for cell in info["cells"]:
                self.assertAlmostEqual(cell["shift"], 0.5 * (cell["forward"] - cell["spot"]), places=12)
            self.assertNotIn("skipped", info, unit.name)
            if task.cell_count == 2:
                # The pairing came first, so the pair statistic moves by the difference of the two cells' shifts:
                # at most 2.5% on the public cards (t2-F1-liftoff-telegraph-2015), far inside what the matcher corrects.
                moved = abs(pair_statistic(new) / pair_statistic(old_samples) - 1)
                self.assertLess(moved, 0.05, unit.name)
        self.assertGreaterEqual(len(changed), 40)
        self.assertIn("t2-F1-conflicting-texts-2024", changed)

    def test_forward_yield_matches_hand_computation(self):
        curve = {2.0: 4.0, 5.0: 4.6, 10.0: 5.0}
        # y(2.5) = 4.1 by interpolation, y(0.5) flat at the 2Y yield: (2.5 * 4.1 - 0.5 * 4.0) / 2
        self.assertAlmostEqual(fc.forward_yield(curve, 0.5, 2.0), 4.125, places=12)
        # y(10.75) flat beyond the longest tenor: (10.75 * 5.0 - 0.75 * 4.0) / 10
        self.assertAlmostEqual(fc.forward_yield(curve, 0.75, 10.0), 5.075, places=12)
        self.assertEqual(fc.forward_yield({2.0: 3.0, 10.0: 3.0}, 0.5, 2.0), 3.0)  # a flat curve implies no change
        self.assertIsNone(fc.tenor_years("EUR"))
        self.assertEqual(fc.tenor_years("UST_30Y"), 30.0)

    def test_setting_is_validated(self):
        for bad in (-0.1, 1.5, float("nan"), "0.5", True, None):
            with self.assertRaises(ValueError, msg=repr(bad)):
                fc.Settings(rates_forward_add=bad)
        self.assertEqual(fc.Settings(rates_forward_add=1.0).rates_forward_add, 1.0)

    def test_without_a_curve_or_on_an_error_the_draws_stand(self):
        unit = UNITS / "t2-F1-conflicting-texts-2024"
        task = fc.load_task(unit)
        bare = replace(task, context_histories={})  # the target tenor alone: no curve to read
        a, model = fc.predict(bare, fc.Settings(draws=256))
        b, _ = fc.predict(bare, fc.Settings(draws=256, rates_forward_add=0.0))
        np.testing.assert_array_equal(a, b)
        self.assertEqual(model.audit["post_processing"]["rates_forward"]["skipped"], "1 Treasury tenors at the as-of date")
        with mock.patch.object(fc, "curve_at_asof", side_effect=RuntimeError("boom")):
            c, model = fc.predict(task, fc.Settings(draws=256))
        d, _ = fc.predict(task, fc.Settings(draws=256, rates_forward_add=0.0))
        np.testing.assert_array_equal(c, d)
        self.assertEqual(model.audit["post_processing"]["rates_forward"]["skipped"], "RuntimeError: boom")
        monthly = fc.load_task(UNITS / "t2-F4-cpi-vintage-2022")
        _, model = fc.predict(monthly, fc.Settings(draws=256))
        self.assertEqual(model.audit["post_processing"]["rates_forward"]["skipped"], "monthly level task")

    def test_rationale_describes_the_forward_increment(self):
        unit = UNITS / "t2-F1-conflicting-texts-2024"
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "forecast.parquet"
            self.assertEqual(fc.main(["forecast", "--panels", str(unit), "--out", str(out), "--draws", "256", "--seed", "0"]), 0)
            text = (Path(tmp) / "forecast_rationale.md").read_text()
            self.assertIn("forward increment", text)
            self.assertIn("v2.7.3", text)


V2_7_REFERENCE = load(HERE.parent / "forecast-v2_7" / "forecast.py", "forecast_v2_7_reference")


class F4ShortHorizon(unittest.TestCase):
    """v2.7.1 (E23): F4 back to v2.6.0's engine width 1.1 and k 1.0; V2_7_SETTINGS reproduces v2.7.0."""

    def test_v2_7_settings_reproduce_v2_7_0_on_every_card(self):
        for unit in CARDS:
            old, _ = V2_7_REFERENCE.predict(V2_7_REFERENCE.load_task(unit), V2_7_REFERENCE.Settings(draws=512))
            new, _ = fc.predict(fc.load_task(unit), fc.Settings(draws=512, **fc.V2_7_SETTINGS))
            np.testing.assert_array_equal(new, old, err_msg=unit.name)

    def test_cli_profile_v2_7_reproduces_v2_7_0(self):
        for name in ("t2-F4-jpy-carry-2007", "t2-F1-conflicting-texts-2024"):
            unit = UNITS / name
            with tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp) / "forecast.parquet"
                self.assertEqual(fc.main(["forecast", "--panels", str(unit), "--out", str(out), "--draws", "512",
                                          "--seed", "0", "--profile", "v2.7"]), 0)
                old, _ = V2_7_REFERENCE.predict(V2_7_REFERENCE.load_task(unit), V2_7_REFERENCE.Settings(draws=512))
                np.testing.assert_array_equal(pd.read_parquet(out).value.to_numpy(), old.reshape(-1), err_msg=name)

    def test_default_changes_only_fresh_daily_f4_cards_and_narrows_them(self):
        changed = []
        for unit in ORIGINAL_CARDS:
            task = fc.load_task(unit)
            old, _ = V2_7_REFERENCE.predict(V2_7_REFERENCE.load_task(unit), V2_7_REFERENCE.Settings(draws=512))
            new, model = fc.predict(task, fc.Settings(draws=512, pair_match=True, monthly_price_vol_switch=False))  # v2.7.3's pair_match default (off) is FomcWeekProfile's
            if fc.is_shock_family(task.family) and fc._fresh_daily(task, model):
                changed.append(unit.name)
                self.assertTrue((new.std(axis=0) < old.std(axis=0)).all(), unit.name)
                self.assertEqual(model.audit["post_processing"]["shock_width"], 1.1, unit.name)
            else:
                np.testing.assert_array_equal(new, old, err_msg=unit.name)
        self.assertGreaterEqual(len(changed), 20)

    def test_default_f4_equals_v2_6_0_and_other_families_keep_v2_7_0(self):
        defaults, v26, v27 = fc.Settings(), fc.Settings(**fc.V2_6_SETTINGS), fc.Settings(**fc.V2_7_SETTINGS)
        self.assertEqual((defaults.sharpen_f4, defaults.f4_width_scale), (v26.sharpen_f4, v26.f4_width_scale))
        self.assertEqual((v27.sharpen_f4, v27.f4_width_scale), ((0.0, 1.1), 1.3))
        for name in ("sharpen_f1", "sharpen_f2", "sharpen_f3", "rates_forward_add", "width_scale", "mix_weight"):
            self.assertEqual(getattr(defaults, name), getattr(v27, name), name)


# ---------------------------------------------------------------- v2.7.2 ranked-Final card shapes
V2_7_1_REFERENCE = load(HERE.parent / "forecast-v2_7_1" / "forecast.py", "forecast_v2_7_1_reference")


def sparse_unit(root: Path, dates: pd.DatetimeIndex, horizons: list[int], *, asof: str, asset: str, sd: float,
                periods: list[str] | None = None, seed: int = 3) -> Path:
    """A synthetic F1 card on one non-daily random-walk series (step sd `sd`), its panel ending before `asof`."""
    rng = np.random.default_rng(seed)
    root.mkdir(parents=True)
    pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "asset": asset,
                  "value": 4.0 + np.cumsum(rng.normal(0.0, sd, len(dates)))}).to_parquet(root / "macro.parquet",
                                                                                         index=False)
    (root / "card.toml").write_text("\n".join([
        "[task]", f'id = "t2-F1-{asset.lower()}-synthetic"', "", "[metadata]", 'category = "T2-F1"', "",
        "[provenance]", f'data_cutoff = "{asof}"', "", "[scoring.params]", "n_draws_min = 200", "",
        "[targets]", f'asset_ids = ["{asset}"]', f"horizons = [{', '.join(map(str, horizons))}]",
        'target_type = "level"', ""]))
    if periods:
        (root / "forecast_spec.json").write_text(json.dumps({"targets": {
            "asset_ids": [asset], "horizons": horizons, "target_type": "level", "observation_periods": periods}}))
    return root


def weekly_unit(root: Path, horizons: list[int], *, weeks: int = 520) -> Path:
    """Friday closes of a rate-like series (0.08 per week), as-of the following Monday, 2024-10-21."""
    return sparse_unit(root, pd.date_range(end="2024-10-18", periods=weeks, freq="W-FRI"), horizons,
                       asof="2024-10-21", asset="WEEKLY_RATE", sd=0.08)


def edit_card_horizons(unit: Path, horizons: list[int], *, drop_periods: bool) -> None:
    card = unit / "card.toml"
    card.write_text("\n".join(f"horizons = {horizons}" if line.strip().startswith("horizons") else line
                              for line in card.read_text().splitlines()) + "\n")
    spec = json.loads((unit / "forecast_spec.json").read_text())
    spec["targets"]["horizons"] = horizons
    if drop_periods:
        spec["targets"].pop("observation_periods", None)
    (unit / "forecast_spec.json").write_text(json.dumps(spec))


def rationale_audit(out_dir: Path) -> dict:
    text = (out_dir / "forecast_rationale.md").read_text()
    return json.loads(text.split("```json\n", 1)[1].split("\n```", 1)[0])


class FinalCardShapes(unittest.TestCase):
    """v2.7.2: the ranked Final's data-release, weekly and [forecast].asof cards; v2.7.1 elsewhere."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="v272-"))

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def gap_two_card(self) -> Path:
        """A next-figure card whose panel ends two months before the as-of month (the US PCE pattern)."""
        unit = renamed_unit(self.tmp, "t2-F4-cpi-vintage-2022", "t2-F4-cpi-gap-two")
        panel = pd.read_parquet(unit / "macro_monthly.parquet")
        panel[pd.to_datetime(panel["date"]) < pd.Timestamp("2022-04-01")].to_parquet(unit / "macro_monthly.parquet",
                                                                                       index=False)
        spec = json.loads((unit / "forecast_spec.json").read_text())
        spec["targets"]["observation_periods"] = ["2022-04"]
        (unit / "forecast_spec.json").write_text(json.dumps(spec))
        return unit

    def forecast_asof_card(self, *, keep_cutoff: str | None = None) -> Path:
        unit = self.tmp / "t2-F1-eur-range-2017"
        shutil.copytree(UNITS / unit.name, unit)
        card = unit / "card.toml"
        text = card.read_text().replace('data_cutoff         = "2017-09-07"\n',
                                        "" if keep_cutoff is None else f'data_cutoff         = "{keep_cutoff}"\n')
        card.write_text(text + '\n[forecast]\nasof = "2017-09-07"\n')
        return unit

    def test_default_reproduces_v2_7_1_on_every_card(self):
        for unit in ORIGINAL_CARDS:
            old, _ = V2_7_1_REFERENCE.predict(V2_7_1_REFERENCE.load_task(unit), V2_7_1_REFERENCE.Settings(draws=512))
            new, _ = fc.predict(fc.load_task(unit), fc.Settings(draws=512, pair_match=True, monthly_price_vol_switch=False))  # v2.7.3's pair_match default (off) is FomcWeekProfile's
            np.testing.assert_array_equal(new, old, err_msg=unit.name)

    def test_monthly_allowance_is_120_days_and_older_profiles_keep_45(self):
        self.assertEqual(fc.Settings().monthly_publication_lag_days, 120)
        for profile in (fc.V2_7_1_SETTINGS, fc.V2_7_SETTINGS, fc.V2_6_1_SETTINGS, fc.V2_6_SETTINGS, legacy(), v25()):
            self.assertEqual(fc.Settings(**profile).monthly_publication_lag_days, 45)

    def test_gap_two_data_release_card_is_fresh(self):
        task = fc.load_task(self.gap_two_card())
        self.assertEqual(task.steps.tolist(), [[1]])
        samples, model = fc.predict(task, fc.Settings(draws=512))
        record = model.audit["assets"]["CPI_ALL"]
        self.assertFalse(record["transfer"])
        self.assertEqual(record["freshness"]["expected_latest_period"], "2022-01")
        post = model.audit["post_processing"]
        self.assertIsInstance(post["mixture"], dict)
        self.assertIsInstance(post["sharpen"], dict)
        old, old_model = V2_7_1_REFERENCE.predict(V2_7_1_REFERENCE.load_task(task.root),
                                                  V2_7_1_REFERENCE.Settings(draws=512))
        self.assertTrue(old_model.audit["assets"]["CPI_ALL"]["transfer"])
        self.assertEqual(old_model.audit["post_processing"]["mixture"], "not applied")
        pinned, _ = fc.predict(task, fc.Settings(draws=512, **fc.V2_7_1_SETTINGS))
        np.testing.assert_array_equal(pinned, old)
        self.assertFalse(np.array_equal(samples, old))

    def test_cli_profile_v2_7_1_reproduces_v2_7_1(self):
        for unit in (self.gap_two_card(), UNITS / "t2-F4-jpy-carry-2007"):
            out = self.tmp / "cli" / unit.name / "forecast.parquet"
            self.assertEqual(fc.main(["forecast", "--panels", str(unit), "--out", str(out), "--draws", "512",
                                      "--seed", "0", "--profile", "v2.7.1"]), 0)
            old, _ = V2_7_1_REFERENCE.predict(V2_7_1_REFERENCE.load_task(unit), V2_7_1_REFERENCE.Settings(draws=512))
            np.testing.assert_array_equal(pd.read_parquet(out).value.to_numpy(), old.reshape(-1), err_msg=unit.name)

    def test_gap_limit_is_five_days_on_daily_panels_and_scales_with_sparser_ones(self):
        self.assertEqual(fc.gap_limit_days(pd.bdate_range("2024-01-01", periods=300)), 5.0)
        self.assertEqual(fc.gap_limit_days(pd.date_range("2024-01-05", periods=300, freq="W-FRI")), 10.5)
        self.assertEqual(fc.gap_limit_days(pd.DatetimeIndex(["2024-01-05", "2024-03-05"])), 5.0)
        task = fc.load_task(UNITS / "t2-F1-eur-range-2017")
        self.assertEqual(fc.gap_limit_days(task.histories["EUR"].index), 5.0)

    def test_weekly_panel_keeps_its_innovations_and_width(self):
        unit = weekly_unit(self.tmp / "weekly", [1])
        task = fc.load_task(unit)
        self.assertEqual(task.cadence, "daily")
        samples, model = fc.predict(task, fc.Settings(draws=1024))
        record = model.audit["assets"]["WEEKLY_RATE"]
        self.assertEqual(record["valid_innovations"], 519)
        self.assertFalse(record["transfer"])
        self.assertLess(abs(record["current_step_sd"] / 0.08 - 1), 0.25)
        _, ref = fc.m0_reference(unit, task.card)
        self.assertLess(abs(samples.std() / np.sqrt(ref["cov"][0, 0]) - 1), 0.4)
        _, old_model = V2_7_1_REFERENCE.predict(V2_7_1_REFERENCE.load_task(unit), V2_7_1_REFERENCE.Settings(draws=1024))
        self.assertEqual(old_model.audit["assets"]["WEEKLY_RATE"]["valid_innovations"], 0)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "forecast.parquet"
            fc.write_output(task, samples, out, model, 0)
            self.assertTrue(all(fc.validate_output(unit, out.parent).values()))

    def test_weekly_steps_count_prints_as_m0_does(self):
        for key in (1, 4):
            unit = weekly_unit(self.tmp / f"key{key}", [key])
            task = fc.load_task(unit)
            self.assertEqual(task.steps.tolist(), [[1]], key)
            _, ref = fc.m0_reference(unit, task.card)
            self.assertEqual(ref["steps"].tolist(), [1.0], key)
            _, model = fc.predict(task, fc.Settings(draws=512))
            self.assertEqual(model.audit["post_processing"]["sharpen"]["k"], 1.1, key)  # F1: same walk as M0
        friday, monday = pd.Timestamp("2024-10-18"), pd.Timestamp("2024-10-21")
        self.assertEqual(fc._m0_step_count(4, friday, 7.0, monday), 1)
        self.assertEqual(fc._m0_step_count(1, friday, 7.0, monday), 1)
        # Claims dated by week-ending Saturday: two prints to Thursday against a key of 3 stay within 2x.
        self.assertEqual(fc._m0_step_count(3, pd.Timestamp("2024-10-19"), 7.0, pd.Timestamp("2024-10-28")), 3)
        self.assertEqual(fc._m0_step_count(1, monday, 7.0, monday), 1)  # a count of 0 keeps the key (§3.7.4)
        self.assertEqual(fc._m0_step_count(4, friday, 1.4, monday), 4)  # daily panels keep the key
        self.assertEqual(fc._m0_step_count(4, pd.Timestamp("2024-08-01"), 30.4, monday), 4)  # monthly, panel steps
        self.assertEqual(fc._m0_step_count(21, pd.Timestamp("2020-02-01"), 30.4, pd.Timestamp("2020-03-31")), 2)

    def test_monthly_card_with_one_key_below_21_and_no_period_walks_the_next_figure(self):
        unit = renamed_unit(self.tmp, "t2-F4-cpi-vintage-2022", "t2-F4-cpi-no-period")
        for key in (1, 4):
            edit_card_horizons(unit, [key], drop_periods=True)
            task = fc.load_task(unit)
            self.assertEqual(task.steps.tolist(), [[1]], key)
            _, ref = fc.m0_reference(unit, task.card)
            self.assertEqual(ref["steps"].tolist(), [1.0], key)
            _, model = fc.predict(task, fc.Settings(draws=512))
            self.assertIsInstance(model.audit["post_processing"]["sharpen"], dict, key)
        self.assertEqual(V2_7_1_REFERENCE.load_task(unit).steps.tolist(), [[4]])
        edit_card_horizons(unit, [3, 4], drop_periods=True)  # several keys: no single next figure, panel steps
        self.assertEqual(fc.load_task(unit).steps.tolist(), [[3, 4]])
        _, ref = fc.m0_reference(unit, tomllib.loads((unit / "card.toml").read_text()))
        self.assertEqual(ref["steps"].tolist(), [3.0, 4.0])
        self.assertTrue(fc._next_figure_card((4,), True))
        self.assertFalse(fc._next_figure_card((21,), True))
        self.assertFalse(fc._next_figure_card((4,), False))

    def test_quarterly_card_keeps_its_m0_walk_and_v2_7_1_draws(self):
        """A quarterly GDP-type target is a stale daily-cadence card whose draws are M0's; nothing here changes."""
        unit = sparse_unit(self.tmp / "quarterly", pd.date_range("2000-01-01", "2024-04-01", freq="QS"), [2],
                           asof="2024-10-28", asset="GDP_QOQ", sd=1.6, periods=["2024-07"])
        task = fc.load_task(unit)
        self.assertEqual((task.cadence, task.steps.tolist()), ("daily", [[2]]))
        _, ref = fc.m0_reference(unit, task.card)
        self.assertEqual(ref["steps"].tolist(), [2.0])
        samples, model = fc.predict(task, fc.Settings(draws=512))
        self.assertIsInstance(model.audit["post_processing"]["transfer_m0"], dict)
        old, _ = V2_7_1_REFERENCE.predict(V2_7_1_REFERENCE.load_task(unit), V2_7_1_REFERENCE.Settings(draws=512))
        np.testing.assert_array_equal(samples, old)

    def test_forecast_asof_is_read_and_m0_falls_back_to_the_task_asof(self):
        original = fc.load_task(UNITS / "t2-F1-eur-range-2017")
        expected, expected_ref = fc.m0_reference(original.root, original.card)
        bare = self.tmp / "no-asof-key"
        shutil.copytree(UNITS / "t2-F1-eur-range-2017", bare)
        card = bare / "card.toml"
        card.write_text(card.read_text().replace('data_cutoff         = "2017-09-07"\n', ""))
        task = fc.load_task(bare)  # the card states no as-of key; [text].cutoff gives the task's
        self.assertEqual(task.asof, "2017-09-07")
        with self.assertRaises(ValueError):
            fc.m0_reference(bare, task.card)
        draws, ref = fc.m0_reference(bare, task.card, asof=task.asof)
        np.testing.assert_array_equal(draws, expected)
        np.testing.assert_array_equal(ref["cov"], expected_ref["cov"])
        _, model = fc.predict(task, fc.Settings(draws=512))
        self.assertIsInstance(model.audit["post_processing"]["m0_reference"], dict)
        unit = self.forecast_asof_card()
        task = fc.load_task(unit)
        self.assertEqual(task.asof, "2017-09-07")
        self.assertNotIn("data_cutoff", task.card["provenance"])
        np.testing.assert_array_equal(fc.m0_reference(unit, task.card)[0], expected)
        _, model = fc.predict(task, fc.Settings(draws=512, pair_match=True))
        post = model.audit["post_processing"]
        self.assertIsInstance(post["m0_reference"], dict)
        self.assertTrue(post["pair_match"]["reached"])
        _, old_model = V2_7_1_REFERENCE.predict(V2_7_1_REFERENCE.load_task(unit), V2_7_1_REFERENCE.Settings(draws=512))
        self.assertTrue(str(old_model.audit["post_processing"]["m0_reference"]).startswith("unavailable"))
        out = self.tmp / "out" / "forecast.parquet"
        self.assertEqual(fc.main(["forecast", "--panels", str(unit), "--out", str(out), "--draws", "512"]), 0)
        self.assertEqual(json.loads((out.parent / "forecast_meta.json").read_text())["asof"], "2017-09-07")
        self.assertTrue(all(fc.validate_output(unit, out.parent).values()))

    def test_forecast_asof_wins_over_data_cutoff_in_both_loaders(self):
        unit = self.forecast_asof_card(keep_cutoff="2017-09-06")
        self.assertEqual(fc.declared_asof(tomllib.loads((unit / "card.toml").read_text())), "2017-09-07")
        self.assertEqual(fc.load_task(unit).asof, "2017-09-07")
        self.assertEqual(fc.declared_asof({"provenance": {"data_cutoff": "2017-09-06"}}), "2017-09-06")
        self.assertIsNone(fc.declared_asof({"forecast": {"asof": ""}, "provenance": "malformed"}))
        for name, card_unit in (("both", unit), ("forecast-only", self.forecast_asof_card_copy())):
            out = self.tmp / f"emergency-{name}" / "forecast.parquet"
            result = fc.emergency_write(card_unit, None, out, 256, 0, "forced")
            self.assertEqual(result["status"], "degraded", name)
            self.assertEqual(json.loads((out.parent / "forecast_meta.json").read_text())["asof"], "2017-09-07", name)
            self.assertEqual(rationale_audit(out.parent)["engine"], "m0-emergency", name)
            self.assertTrue(all(fc.validate_output(card_unit, out.parent).values()), name)

    def forecast_asof_card_copy(self) -> Path:
        unit = self.tmp / "forecast-only"
        shutil.copytree(UNITS / "t2-F1-eur-range-2017", unit)
        card = unit / "card.toml"
        card.write_text(card.read_text().replace('data_cutoff         = "2017-09-07"\n', "")
                        + '\n[forecast]\nasof = "2017-09-07"\n')
        return unit



# ---------------------------------------------------------------- v2.7.3 FOMC-week variance profile (E24)
V2_7_2_REFERENCE = load(HERE.parent / "forecast-v2_7_2" / "forecast.py", "forecast_v2_7_2_reference")
RATES_SOURCE = UNITS / "t2-F4-ust2y-december-fomc-2021" / "rates_daily.parquet"  # to 2021-12-31; FOMC 2021-12-15
FX_SOURCE = UNITS / "t2-F4-nok-covid-2020" / "g10_fx_daily.parquet"  # to 2020-02-28; FOMC 2020-01-29


def fomc_unit(root: Path, source: Path, assets: list[str], horizons: list[int], *, asof: str, family: str = "T2-F2",
              rename: dict[str, str] | None = None, panel_id: str | None = None, panel_end: str | None = None) -> Path:
    """A daily card on a practice panel cut at `panel_end` (default the as-of); `rename` maps the panel's asset ids."""
    root.mkdir(parents=True)
    panel = pd.read_parquet(source)
    panel = panel[pd.to_datetime(panel["date"]) <= pd.Timestamp(panel_end or asof)].copy()
    if rename:
        panel["asset"] = panel["asset"].replace(rename)
    name = panel_id or source.stem
    panel.assign(panel_id=name).to_parquet(root / f"{name}.parquet", index=False)
    (root / "card.toml").write_text("\n".join([
        "[task]", f'id = "t2-{family[-2:]}-fomc-week-{root.name}"', "", "[metadata]", f'category = "{family}"', "",
        "[provenance]", f'data_cutoff = "{asof}"', "", "[scoring.params]", "n_draws_min = 200", "",
        "[targets]", f"asset_ids = {json.dumps(assets)}", f"horizons = {json.dumps(horizons)}",
        'target_type = "level"', ""]))
    return root


class FomcWeekProfile(unittest.TestCase):
    """v2.7.3 (E24): the FOMC-week step profile on Monday as-of cards with steps 1-4; v2.7.2 everywhere else."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="v273-"))

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def curve(self, name: str = "curve", **kwargs) -> Path:
        return fomc_unit(self.tmp / name, RATES_SOURCE, ["UST_2Y", "UST_10Y", "UST_30Y"], [1, 2, 3, 4],
                         asof=kwargs.pop("asof", "2021-12-13"), **kwargs)

    def test_calendar_table_and_switch(self):
        self.assertEqual(len(fc.FOMC_WEDNESDAYS), 128)  # E23's 127 Wednesday statements and 28 October 2026
        self.assertTrue(all(day.dayofweek == 2 for day in fc.FOMC_WEDNESDAYS))
        self.assertIn(pd.Timestamp("2026-10-28"), fc.FOMC_WEDNESDAYS)
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # every ranked card has a Monday as-of; the check must stay silent there
            self.assertTrue(fc.fomc_week("2026-10-26"))
            self.assertFalse(fc.fomc_week("2026-10-19"))
        self.assertFalse(fc.fomc_week("2026-10-27"))
        self.assertFalse(fc.fomc_week("2024-11-04"))  # that week's decision came on the Thursday
        self.assertEqual(set(fc.EVENT_PROFILE), {"ust", "bills", "g10", "em"})
        for m in fc.EVENT_PROFILE.values():
            self.assertEqual(len(m), 4)
            self.assertTrue(all(0.5 <= v <= 2.5 for v in m))
        self.assertTrue(fc.Settings().event_profile)
        for profile in (fc.V2_7_2_SETTINGS, fc.V2_7_1_SETTINGS, fc.V2_7_SETTINGS, fc.V2_6_1_SETTINGS,
                        fc.V2_6_SETTINGS, legacy(), v25(), fc.POST_PROCESSING_OFF):
            self.assertFalse(fc.Settings(**profile).event_profile)
        self.assertFalse(fc.v1_settings().event_profile)
        with self.assertRaises(ValueError):
            fc.Settings(event_profile=1)

    def test_asset_names(self):
        cases = {"UST_3M": "bills", "UST_6M": "bills", "UST_1Y": "bills", "UST_12M": "bills", "ust_3m": "bills",
                 "UST_2Y": "ust", "UST_30Y": "ust", "UST_24M": "ust",
                 "EUR": "g10", "JPY": "g10", "DKK": "g10", "MXN": "em", "INR": "em", "KRW": "em",
                 "USD_MXN": "em", "USDMXN": "em", "MXN/USD": "em", "EURUSD": "g10", "EUR_USD": "g10",
                 "usd-jpy": "g10", "USD_EUR": "g10",
                 "USD": None, "EURJPY": None, "USDCNY": None, "CNY": None, "XAUUSD": None, "UST_0M": None,
                 "UST10Y": None, "DGS10": None, "UST_10": None, "MKT": None, "USD_MXN_X": None, "USDUSD": None}
        for name, expected in cases.items():
            self.assertEqual(fc.event_asset_class(name), expected, name)

    def test_trigger(self):
        task = fc.load_task(self.curve())
        self.assertIsNone(fc.event_profile_reason(task))
        cases = {
            "as-of is not the Monday of an FOMC week": [self.curve("tuesday", asof="2021-12-14"),
                                                        self.curve("quiet-week", asof="2021-12-06")],
            "a step outside 1-4": [fomc_unit(self.tmp / "h5", RATES_SOURCE, ["UST_2Y"], [1, 5], asof="2021-12-13")],
            "a target outside the profile's classes": [fomc_unit(self.tmp / "dgs", RATES_SOURCE, ["DGS2"], [1, 2],
                                                                 asof="2021-12-13", rename={"UST_2Y": "DGS2"})],
            "a target without a fresh history": [self.curve("stale", panel_end="2021-12-03")],
        }
        for reason, units in cases.items():
            for unit in units:
                self.assertEqual(fc.event_profile_reason(fc.load_task(unit)), reason, unit.name)
        for unit in ORIGINAL_CARDS:  # no practice card has a Monday FOMC-week as-of with steps 1-4
            self.assertIsNotNone(fc.event_profile_reason(fc.load_task(unit)), unit.name)
        _, model = fc.predict(task, fc.Settings(draws=512))
        record = model.audit["post_processing"]["event_profile"]
        self.assertEqual(sorted(record["assets"]), ["UST_10Y", "UST_2Y", "UST_30Y"])
        self.assertNotIn("skipped", record)
        self.assertEqual(record["assets"]["UST_2Y"]["steps"], [1, 2, 3, 4])
        _, off = fc.predict(task, fc.Settings(draws=512, event_profile=False))
        self.assertEqual(off.audit["post_processing"]["event_profile"], "not applied: off")

    def test_profile_is_the_family_reset_then_the_path_transform(self):
        for family in ("T2-F1", "T2-F2", "T2-F3", "T2-F4"):
            task = fc.load_task(self.curve(family.lower(), family=family))
            new, model = fc.predict(task, fc.Settings(draws=1024, fomc_mean_shift=False))  # v2.7.6's E37 has its own test
            plain, _ = fc.predict(task, fc.Settings(draws=1024, event_profile=False, **fc.FAMILY_NEUTRAL))
            expected, _ = fc.apply_event_profile(task, plain)
            np.testing.assert_array_equal(new, expected, err_msg=family)
            self.assertIsNone(model.audit["post_processing"]["shock_width"], family)
            self.assertEqual(model.audit["post_processing"]["sharpen"], "not applied", family)
            # The audit keeps the run's own settings; the per-card reset is recorded under event_profile.
            self.assertEqual(model.audit["settings"], asdict(fc.Settings(draws=1024, fomc_mean_shift=False)), family)
            v272, _ = V2_7_2_REFERENCE.predict(V2_7_2_REFERENCE.load_task(task.root),
                                               V2_7_2_REFERENCE.Settings(draws=1024))
            self.assertFalse(np.array_equal(new, v272), family)

    def test_path_transform_scales_each_step_keeps_medians_correlation_and_quote_direction(self):
        rng = np.random.default_rng(0)
        n = 20000
        rate = 1.5 + np.cumsum(rng.normal(0.0, 0.05, (n, 4)), axis=1)
        m = np.asarray(fc.EVENT_PROFILE["ust"])
        path = SimpleNamespace(assets=("UST_2Y",), steps=np.array([[1, 2, 3, 4]]))
        out, info = fc.apply_event_profile(path, rate[:, None, :])
        self.assertNotIn("skipped", info)
        np.testing.assert_allclose(np.median(out[:, 0], axis=0), np.median(rate, axis=0), atol=1e-12)
        ratios = np.r_[out[:, 0, 0].std() / rate[:, 0].std(),
                       np.diff(out[:, 0], axis=1).std(axis=0) / np.diff(rate, axis=1).std(axis=0)]
        np.testing.assert_allclose(ratios, np.sqrt(m), rtol=1e-9)
        shuffled = SimpleNamespace(assets=("UST_2Y",), steps=np.array([[3, 1, 4, 2]]))  # card order is kept
        reordered, _ = fc.apply_event_profile(shuffled, rate[:, None, [2, 0, 3, 1]])
        np.testing.assert_array_equal(reordered[:, 0], out[:, 0, [2, 0, 3, 1]])
        single = SimpleNamespace(assets=("UST_2Y",), steps=np.array([[3]]))
        cell, _ = fc.apply_event_profile(single, rate[:, None, 2:3])
        self.assertAlmostEqual(cell.std() / rate[:, 2].std(), float(np.sqrt(m[:3].mean())), places=9)
        fx = 1.1 * np.exp(0.02 * (rate - 1.5) + np.cumsum(rng.normal(0.0, 0.006, (n, 4)), axis=1))
        eur, _ = fc.apply_event_profile(SimpleNamespace(assets=("EUR",), steps=np.array([[1, 2, 3, 4]])),
                                        fx[:, None, :])
        inverse, _ = fc.apply_event_profile(SimpleNamespace(assets=("USD_EUR",), steps=np.array([[1, 2, 3, 4]])),
                                            1.0 / fx[:, None, :])
        np.testing.assert_allclose(eur * inverse, 1.0, rtol=1e-12)
        pair = SimpleNamespace(assets=("UST_2Y", "EUR"), steps=np.array([[2], [2]]))
        joint, _ = fc.apply_event_profile(pair, np.stack([rate[:, 1], fx[:, 1]], axis=1)[:, :, None])
        self.assertAlmostEqual(np.corrcoef(joint[:, 0, 0], np.log(joint[:, 1, 0]))[0, 1],
                               np.corrcoef(rate[:, 1], np.log(fx[:, 1]))[0, 1], places=12)
        negative, info = fc.apply_event_profile(SimpleNamespace(assets=("EUR",), steps=np.array([[1]])),
                                                -fx[:, None, :1])
        np.testing.assert_array_equal(negative, -fx[:, None, :1])
        self.assertIn("skipped", info)

    def test_cross_class_correlation_depends_on_the_requested_horizons(self):
        """Known and kept (v2.7.4, the 17 October review's item 1): the transform scales the increments between
        the horizons a card requests. Within one asset, or between assets of one class, the marginals and the
        cross-horizon covariances do not depend on which horizons are requested. Between classes with different
        step profiles the correlation at a horizon does: a dense request (every step to h) scales each daily
        increment by its own multiplier, a sparse one (h alone) scales the whole change by the mean multiplier.
        With independent steps and a constant cross-asset correlation rho, the dense request gives
        rho * sum(sqrt(m_a * m_b)) / sqrt(sum(m_a) * sum(m_b)) at h (0.941 rho for UST notes against G10 at h = 3);
        the sparse one keeps rho. E24's evidence was produced with this transform. In the ranked window the ECB
        fixing takes the FOMC decision one step after the Treasury close, a lead-lag that a per-step model with a
        contemporaneous correlation does not describe either, so neither request is known to be closer. The
        behaviour is documented here, not changed."""
        rng = np.random.default_rng(7)
        n, rho = 200_000, 0.8
        steps = rng.multivariate_normal([0.0, 0.0], [[1.0, rho], [rho, 1.0]], size=(n, 4))  # (n, step, asset)
        rate = 1.5 + 0.05 * np.cumsum(steps[:, :, 0], axis=1)
        fx = 1.1 * np.exp(0.006 * np.cumsum(steps[:, :, 1], axis=1))
        jpy = 150.0 * np.exp(0.007 * np.cumsum(steps[:, :, 0], axis=1))  # same class as EUR, correlation rho

        def corr_at_3(assets, a, b, request):
            idx = [h - 1 for h in request]
            grid = np.stack([a[:, idx], b[:, idx]], axis=1)
            task = SimpleNamespace(assets=assets, steps=np.array([request, request]))
            out, info = fc.apply_event_profile(task, grid)
            self.assertNotIn("skipped", info)
            col = request.index(3)
            x = out[:, 0, col] if assets[0].startswith("UST") else np.log(out[:, 0, col])
            return np.corrcoef(x, np.log(out[:, 1, col]))[0, 1], out[:, 0, col]

        sparse, cell_sparse = corr_at_3(("UST_2Y", "EUR"), rate, fx, [3])
        dense, cell_dense = corr_at_3(("UST_2Y", "EUR"), rate, fx, [1, 2, 3, 4])
        m_ust, m_g10 = (np.asarray(fc.EVENT_PROFILE[c][:3]) for c in ("ust", "g10"))
        factor = np.sqrt(m_ust * m_g10).sum() / np.sqrt(m_ust.sum() * m_g10.sum())
        self.assertAlmostEqual(factor, 0.941, places=3)
        self.assertAlmostEqual(sparse, rho, delta=0.004)
        self.assertAlmostEqual(dense, rho * factor, delta=0.004)
        # The marginal at h = 3 is the same either way: sum(m) over the steps against 3 * mean(m).
        self.assertAlmostEqual(cell_dense.std() / cell_sparse.std(), 1.0, delta=0.01)
        same_sparse, _ = corr_at_3(("EUR", "JPY"), fx, jpy, [3])  # one class: one profile, no change
        same_dense, _ = corr_at_3(("EUR", "JPY"), fx, jpy, [1, 2, 3, 4])
        self.assertAlmostEqual(same_sparse, rho, delta=0.004)
        self.assertAlmostEqual(same_dense, rho, delta=0.004)

    def test_two_cell_card_takes_the_profile_before_pair_matching(self):
        unit = fomc_unit(self.tmp / "pair", FX_SOURCE, ["EUR"], [1, 4], asof="2020-01-27")
        task = fc.load_task(unit)
        self.assertIsNone(fc.event_profile_reason(task))
        samples, model = fc.predict(task, fc.Settings(draws=1024, pair_match=True))
        post = model.audit["post_processing"]
        self.assertTrue(post["pair_match"]["reached"])
        self.assertIsInstance(post["event_profile"], dict)
        reference, _ = fc.m0_reference(unit, task.card)
        self.assertLess(abs(pair_statistic(samples) / pair_statistic(reference) - 1), 1e-9)
        moved, _ = fc.apply_event_profile(task, samples)  # applied after the pairing, it would move the statistic
        self.assertGreater(abs(pair_statistic(moved) / pair_statistic(reference) - 1), 1e-3)
        unpaired, off = fc.predict(task, fc.Settings(draws=1024, pair_match=False))
        self.assertNotIn("pair_match", off.audit["post_processing"])
        self.assertIsInstance(off.audit["post_processing"]["event_profile"], dict)
        self.assertTrue(np.isfinite(unpaired).all())
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "forecast.parquet"
            fc.write_output(task, unpaired, out, off, 0)
            self.assertTrue(all(fc.validate_output(unit, out.parent).values()))

    def test_unfamiliar_names_take_the_profile_in_log_units(self):
        unit = fomc_unit(self.tmp / "names", FX_SOURCE, ["EURUSD", "USD_JPY"], [1, 2, 3, 4], asof="2020-01-27",
                         rename={"EUR": "EURUSD", "JPY": "USD_JPY"}, panel_id="ecb_reference_rates")
        task = fc.load_task(unit)
        self.assertIsNone(fc.event_profile_reason(task))
        samples, model = fc.predict(task, fc.Settings(draws=512))
        record = model.audit["post_processing"]["event_profile"]
        self.assertEqual({a: r["class"] for a, r in record["assets"].items()}, {"EURUSD": "g10", "USD_JPY": "g10"})
        self.assertEqual([t.kind for t in model.transforms], ["level", "level"])  # the engine reads these as levels
        self.assertTrue((samples > 0).all())
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "forecast.parquet"
            fc.write_output(task, samples, out, model, 0)
            self.assertTrue(all(fc.validate_output(unit, out.parent).values()))

    def test_v2_7_2_profile_reproduces_v2_7_2(self):
        units = [self.curve(), fomc_unit(self.tmp / "pair2", FX_SOURCE, ["EUR"], [1, 4], asof="2020-01-27"),
                 UNITS / "t2-F4-jpy-carry-2007"]
        for unit in units:
            old, _ = V2_7_2_REFERENCE.predict(V2_7_2_REFERENCE.load_task(unit), V2_7_2_REFERENCE.Settings(draws=512))
            pinned, _ = fc.predict(fc.load_task(unit), fc.Settings(draws=512, **fc.V2_7_2_SETTINGS))
            np.testing.assert_array_equal(pinned, old, err_msg=unit.name)
            out = self.tmp / "cli" / unit.name / "forecast.parquet"
            self.assertEqual(fc.main(["forecast", "--panels", str(unit), "--out", str(out), "--draws", "512",
                                      "--seed", "0", "--profile", "v2.7.2"]), 0)
            np.testing.assert_array_equal(pd.read_parquet(out).value.to_numpy(), old.reshape(-1), err_msg=unit.name)
            default, _ = fc.predict(fc.load_task(unit), fc.Settings(draws=512))
            self.assertEqual(np.array_equal(default, old), unit.name == "t2-F4-jpy-carry-2007", unit.name)

    def test_practice_cards_change_only_through_the_pairing(self):
        """No practice card takes the profile, so v2.7.3 is v2.7.2 there up to E25's pairing default."""
        changed = []
        for unit in ORIGINAL_CARDS:
            task = fc.load_task(unit)
            old, _ = V2_7_2_REFERENCE.predict(V2_7_2_REFERENCE.load_task(unit), V2_7_2_REFERENCE.Settings(draws=512))
            paired, model = fc.predict(task, fc.Settings(draws=512, pair_match=True, monthly_price_vol_switch=False))
            np.testing.assert_array_equal(paired, old, err_msg=unit.name)
            self.assertTrue(str(model.audit["post_processing"]["event_profile"]).startswith("not applied"), unit.name)
            new, _ = fc.predict(task, fc.Settings(draws=512, monthly_price_vol_switch=False))
            unpaired, _ = V2_7_2_REFERENCE.predict(V2_7_2_REFERENCE.load_task(unit),
                                                   V2_7_2_REFERENCE.Settings(draws=512, pair_match=False))
            np.testing.assert_array_equal(new, unpaired, err_msg=unit.name)
            if not np.array_equal(new, old):
                changed.append(unit.name)
                self.assertEqual(task.cell_count, 2, unit.name)
        self.assertGreaterEqual(len(changed), 10)

    def test_pair_match_is_off_by_default_and_on_in_the_reproduction_profiles(self):
        """v2.7.3 (E25): two-cell cards are not pair-matched by default; v2.7.2 and every earlier profile pair them."""
        self.assertFalse(fc.Settings().pair_match)
        for profile in (fc.V2_7_2_SETTINGS, fc.V2_7_1_SETTINGS, fc.V2_7_SETTINGS, fc.V2_6_1_SETTINGS,
                        fc.V2_6_SETTINGS, legacy(), v25()):
            self.assertTrue(fc.Settings(**profile).pair_match)
        self.assertFalse(fc.Settings(**fc.POST_PROCESSING_OFF).pair_match)
        unit = UNITS / "t2-F1-conflicting-texts-2024"  # one asset at two horizons
        task = fc.load_task(unit)
        self.assertEqual(task.cell_count, 2)
        old, _ = V2_7_2_REFERENCE.predict(V2_7_2_REFERENCE.load_task(unit), V2_7_2_REFERENCE.Settings(draws=512))
        default, model = fc.predict(task, fc.Settings(draws=512))
        self.assertNotIn("pair_match", model.audit["post_processing"])
        self.assertFalse(np.array_equal(default, old))
        out = self.tmp / "cli-pair" / "forecast.parquet"
        self.assertEqual(fc.main(["forecast", "--panels", str(unit), "--out", str(out), "--draws", "512", "--seed", "0",
                                  "--profile", "v2.7.2"]), 0)
        np.testing.assert_array_equal(pd.read_parquet(out).value.to_numpy(), old.reshape(-1))
        self.assertTrue(rationale_audit(out.parent)["post_processing"]["pair_match"]["reached"])


V2_7_3_REFERENCE = load(HERE.parent / "forecast-v2_7_3" / "forecast.py", "forecast_v2_7_3_reference")


def shifted_unit(unit: Path, offset: float) -> Path:
    """Add `offset` percentage points to every Treasury value of a fomc_unit card (moves it off the zero bound)."""
    for path in unit.glob("*.parquet"):
        frame = pd.read_parquet(path)
        frame["value"] = frame["value"] + offset
        frame.to_parquet(path, index=False)
    return unit


class RegimeSwitches(unittest.TestCase):
    """v2.7.4: E29's UST family widths away from the zero bound, E30's price-index switch, M0's strict step rule."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="v274-"))

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_defaults_profiles_and_validation(self):
        s = fc.Settings()
        self.assertTrue(s.ust_family_widths_active)
        self.assertEqual(s.ust_active_floor, 1.0)
        self.assertTrue(s.monthly_price_vol_switch)  # E30 is post hoc: on by the user's call of 18 October
        for profile in (fc.V2_7_3_SETTINGS, fc.V2_7_2_SETTINGS, fc.V2_7_1_SETTINGS, fc.V2_7_SETTINGS,
                        fc.V2_6_1_SETTINGS, fc.V2_6_SETTINGS, legacy(), v25(), fc.POST_PROCESSING_OFF):
            settings = fc.Settings(**profile)
            self.assertFalse(settings.ust_family_widths_active)
            self.assertFalse(settings.monthly_price_vol_switch)
        self.assertFalse(fc.v1_settings().ust_family_widths_active)
        for bad in ({"ust_family_widths_active": 1}, {"ust_active_floor": float("nan")}, {"ust_active_floor": 30.0},
                    {"monthly_price_vol_ratio": 0.1}, {"monthly_price_vol_switch": "yes"}):
            with self.assertRaises(ValueError, msg=str(bad)):
                fc.Settings(**bad)

    def test_ust_notes_keep_family_widths_away_from_the_zero_bound(self):
        rename = {"UST_2Y": "UST_3M"}  # a bill and two notes on one card
        zlb = fomc_unit(self.tmp / "zlb", RATES_SOURCE, ["UST_3M", "UST_10Y", "UST_30Y"], [1, 2, 3, 4],
                        asof="2021-12-13", rename=rename)
        task = fc.load_task(zlb)
        self.assertLess(fc.ust_floor_level(task), 1.0)  # December 2021: the front end near zero
        default, model = fc.predict(task, fc.Settings(draws=512, fomc_mean_shift=False))
        v273, _ = V2_7_3_REFERENCE.predict(V2_7_3_REFERENCE.load_task(zlb), V2_7_3_REFERENCE.Settings(draws=512))
        np.testing.assert_array_equal(default, v273)
        self.assertTrue(model.audit["post_processing"]["ust_family_widths"].startswith("not applied: lowest UST"))
        active = shifted_unit(fomc_unit(self.tmp / "active", RATES_SOURCE, ["UST_3M", "UST_10Y", "UST_30Y"],
                                        [1, 2, 3, 4], asof="2021-12-13", rename=rename), 2.0)
        task = fc.load_task(active)
        self.assertGreaterEqual(fc.ust_floor_level(task), 1.0)
        got, model = fc.predict(task, fc.Settings(draws=512, fomc_mean_shift=False))
        ev, _ = fc.predict(task, fc.Settings(draws=512, ust_family_widths_active=False, fomc_mean_shift=False))
        plain, _ = fc.predict(task, fc.Settings(draws=512, event_profile=False, ust_family_widths_active=False))
        kept, _ = fc.apply_event_profile(task, plain)
        np.testing.assert_array_equal(got[:, 0], ev[:, 0])  # the bill keeps v2.7.3's draws
        np.testing.assert_array_equal(got[:, 1:], kept[:, 1:])  # the notes take the profile on their family widths
        self.assertEqual(model.audit["post_processing"]["ust_family_widths"]["applied_to"], ["UST_10Y", "UST_30Y"])
        self.assertTrue((got[:, 1:].std(axis=0) > ev[:, 1:].std(axis=0)).all())  # F2's k 1.2 is back on the notes
        fx = fomc_unit(self.tmp / "fx", FX_SOURCE, ["EUR", "JPY"], [1, 2, 3, 4], asof="2020-01-27")
        _, model = fc.predict(fc.load_task(fx), fc.Settings(draws=512))
        self.assertEqual(model.audit["post_processing"]["ust_family_widths"], "not applied: no UST-notes target")

    def test_price_switch_skips_the_monthly_reweighting(self):
        cpi = UNITS / "t2-F1-cpi-glidepath-2023"
        task = fc.load_task(cpi)
        off, _ = fc.predict(task, fc.Settings(draws=512, monthly_price_vol_switch=False))
        v273, _ = V2_7_3_REFERENCE.predict(V2_7_3_REFERENCE.load_task(cpi), V2_7_3_REFERENCE.Settings(draws=512))
        np.testing.assert_array_equal(off, v273)
        ratio = fc.monthly_regime_ratio(task)
        on, model = fc.predict(task, fc.Settings(draws=512))  # the default switches it on
        alpha0, _ = fc.predict(task, fc.Settings(draws=512, monthly_sharpen_alpha=0.0))
        if ratio >= 1.0:
            np.testing.assert_array_equal(on, alpha0)
            self.assertIn("skipped", model.audit["post_processing"]["sharpen"])
        else:
            np.testing.assert_array_equal(on, off)
        low, model = fc.predict(task, fc.Settings(draws=512, monthly_price_vol_ratio=3.0))
        np.testing.assert_array_equal(low, off)
        self.assertIn("below", model.audit["post_processing"]["sharpen"]["e30"])
        unrate = UNITS / "t2-F1-sahm-watch-2024"  # a labour series never switches
        task = fc.load_task(unrate)
        base, _ = fc.predict(task, fc.Settings(draws=512, monthly_price_vol_switch=False))
        on, model = fc.predict(task, fc.Settings(draws=512, monthly_price_vol_ratio=0.5))
        np.testing.assert_array_equal(on, base)
        self.assertEqual(model.audit["post_processing"]["sharpen"]["e30"], "a target that does not read as a price index")
        for name, price in (("PCE_CORE", True), ("HICP_EA", True), ("cpi_tokyo", True), ("NFP", False),
                            ("UNRATE", False), ("JOBLESS_CLAIMS", False)):
            self.assertEqual(bool(fc.PRICE_NAMES.search(name)), price, name)

    def test_e30_changes_exactly_the_two_cpi_practice_cards(self):
        """On the public cards E30 fires only on the two CPI cards, whose recent monthly moves exceed their long-run
        scale (ratios about 1.12 and 1.17); every other card is v2.7.3 with E29 (which no public card triggers)."""
        changed = []
        for unit in CARDS:
            task = fc.load_task(unit)
            new, model = fc.predict(task, fc.Settings(draws=512))
            off, _ = fc.predict(task, fc.Settings(draws=512, monthly_price_vol_switch=False))
            if not np.array_equal(new, off):
                changed.append(unit.name)
                self.assertIn("skipped", model.audit["post_processing"]["sharpen"], unit.name)
                alpha0, _ = fc.predict(task, fc.Settings(draws=512, monthly_sharpen_alpha=0.0))
                np.testing.assert_array_equal(new, alpha0, err_msg=unit.name)
        self.assertEqual(sorted(changed), ["t2-F1-cpi-glidepath-2023", "t2-F4-cpi-vintage-2022"])

    def test_m0_step_count_keeps_the_key_at_a_ratio_of_exactly_two(self):
        asof, last = pd.Timestamp("2024-10-21"), pd.Timestamp("2024-10-18")  # Monday, the Friday print before it
        self.assertEqual(fc._m0_step_count(1, last, 7.0, asof), 1)
        self.assertEqual(fc._m0_step_count(2, last, 7.0, asof), 2)  # one print, ratio exactly 2: the key stays
        self.assertEqual(fc._m0_step_count(4, last, 7.0, asof), 1)  # ratio 4: the print count wins
        self.assertEqual(V2_7_3_REFERENCE._m0_step_count(2, last, 7.0, asof), 1)  # v2.7.3's >= 2 reading
        self.assertEqual(fc._m0_step_count(2, last, 1.4, asof), 2)  # a daily panel always keeps its key


class DateUnits(unittest.TestCase):
    """v2.7.4: day counts do not depend on the stored datetime resolution. The pinned pandas 2.3.3 reads these
    dates in nanoseconds; pandas 3 can read them in microseconds or seconds, where `asi8 / 86400e9` was 1,000x or
    more too small and a weekly card lost every innovation (the 17 October review's item 2)."""

    def test_gap_rule_and_digest_are_unit_free(self):
        weekly = pd.date_range(end="2024-10-18", periods=60, freq="W-FRI")
        daily = pd.bdate_range(end="2024-10-18", periods=60)
        for index, expected in ((weekly, 10.5), (daily, 5.0)):
            for unit in ("s", "ms", "us", "ns"):
                converted = index.as_unit(unit)
                self.assertEqual(fc.gap_limit_days(converted), expected, unit)
                np.testing.assert_array_equal(fc.epoch_ns(converted), index.as_unit("ns").asi8)

    def test_weekly_card_keeps_its_innovations_at_any_resolution(self):
        with tempfile.TemporaryDirectory() as tmp:
            task = fc.load_task(weekly_unit(Path(tmp) / "weekly", [1]))
            micro = {name: {a: s.set_axis(s.index.as_unit("us")) for a, s in getattr(task, name).items()}
                     for name in ("histories", "context_histories")}
            ref, _ = fc.predict(task, fc.Settings(draws=512))
            got, model = fc.predict(replace(task, **micro), fc.Settings(draws=512))
            self.assertEqual(model.audit["assets"]["WEEKLY_RATE"]["valid_innovations"], 519)
            np.testing.assert_array_equal(got, ref)



V2_7_4_REFERENCE = load(HERE.parent / "forecast-v2_7_4" / "forecast.py", "forecast_v2_7_4_reference")
# A calendar pattern in log points, January to December; it sums to zero over the year.
SEASON = np.array([-0.6, 0.4, 0.5, 0.3, 0.2, 0.3, -0.5, 0.0, 0.3, 0.1, -0.4, -0.6]) / 100


def seasonal_unit(target: Path) -> Path:
    """t2-F4-cpi-vintage-2022 with CPI_ALL put on SEASON's calendar: an index published without seasonal adjustment."""
    shutil.copytree(UNITS / "t2-F4-cpi-vintage-2022", target)
    path = target / "macro_monthly.parquet"
    frame = pd.read_parquet(path)
    rows = frame.asset.eq("CPI_ALL").to_numpy()
    months = pd.to_datetime(frame.loc[rows, "date"]).dt.month.to_numpy()
    frame.loc[rows, "value"] = frame.loc[rows, "value"].to_numpy() * np.exp(np.cumsum(SEASON)[months - 1])
    frame.to_parquet(path, index=False)
    return target


def monthly_series(n: int, season: np.ndarray, drift: float = 0.002, noise: float = 0.001, seed: int = 5) -> pd.Series:
    rng = np.random.default_rng(seed)
    index = pd.date_range("1999-01-01", periods=n, freq="MS")
    steps = drift + season[index.month.to_numpy() - 1] + noise * rng.standard_normal(n)
    return pd.Series(100.0 * np.exp(np.cumsum(steps)), index=index)


class Seasonal(unittest.TestCase):
    """v2.7.5 (E35, E36): monthly targets with calendar-month effects are forecast seasonally adjusted."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="v275-"))

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_defaults_profiles_and_validation(self):
        self.assertTrue(fc.Settings().monthly_seasonal)
        for profile in (fc.V2_7_4_SETTINGS, fc.V2_7_3_SETTINGS, fc.V2_7_2_SETTINGS, fc.V2_7_1_SETTINGS, fc.V2_7_SETTINGS,
                        fc.V2_6_1_SETTINGS, fc.V2_6_SETTINGS, legacy(), v25(), fc.POST_PROCESSING_OFF):
            self.assertFalse(fc.Settings(**profile).monthly_seasonal)
        self.assertFalse(fc.v1_settings().monthly_seasonal)
        with self.assertRaises(ValueError):
            fc.Settings(monthly_seasonal=1)

    def test_detector_reads_calendar_effects_and_nothing_else(self):
        nsa = monthly_series(300, SEASON)
        found = fc.seasonal_effects(nsa)
        self.assertIsNotNone(found)
        self.assertTrue(found["log"])
        self.assertLess(found["p"], 1e-3)
        np.testing.assert_allclose(found["effects"], SEASON, atol=0.0015)
        self.assertAlmostEqual(float(found["effects"].sum()), 0.0, places=12)
        np.testing.assert_allclose(found["running"], np.cumsum(found["effects"]))
        self.assertIsNone(fc.seasonal_effects(monthly_series(300, np.zeros(12))))  # adjusted: nothing to find
        yoy = (100 * (nsa / nsa.shift(12) - 1)).dropna()
        self.assertIsNone(fc.seasonal_effects(yoy))  # a year-on-year rate has no calendar effect
        self.assertIsNone(fc.seasonal_effects(nsa.iloc[:90]))  # fewer than 96 changes: nothing is done
        rate = pd.Series(np.cumsum(SEASON[nsa.index.month - 1] * 100) - 0.2 + 0.01 * np.arange(len(nsa)) % 0.3,
                         index=nsa.index)
        additive = fc.seasonal_effects(rate)  # a series that crosses zero is adjusted additively
        self.assertIsNotNone(additive)
        self.assertFalse(additive["log"])
        self.assertAlmostEqual(fc.seasonal_offset(found, pd.Timestamp("2022-04-01"), 2),
                               float(found["running"][5] - found["running"][3]))
        self.assertAlmostEqual(fc.seasonal_offset(found, pd.Timestamp("2022-11-01"), 14),
                               float(found["running"][0] - found["running"][10]))  # a full year adds nothing

    def test_no_public_card_changes(self):
        for unit in CARDS:
            task = fc.load_task(unit)
            if task.cadence == "monthly":
                self.assertEqual(fc.seasonal_plan(task), {}, unit.name)
            new, model = fc.predict(task, fc.Settings(draws=512, fomc_mean_shift=False))  # E37 changes four practice-set cards
            old, _ = V2_7_4_REFERENCE.predict(V2_7_4_REFERENCE.load_task(unit), V2_7_4_REFERENCE.Settings(draws=512))
            np.testing.assert_array_equal(new, old, err_msg=unit.name)
            self.assertNotIn("seasonal", model.audit, unit.name)

    def test_a_seasonal_card_is_forecast_adjusted_and_put_back(self):
        unit = seasonal_unit(self.tmp / "nsa")
        task = fc.load_task(unit)
        plan = fc.seasonal_plan(task)
        self.assertEqual(list(plan), ["CPI_ALL"])
        np.testing.assert_allclose(plan["CPI_ALL"]["effects"], SEASON, atol=0.003)
        got, model = fc.predict(task, fc.Settings(draws=512))
        x, effects = task.histories["CPI_ALL"], plan["CPI_ALL"]
        offset = effects["running"][x.index.month.to_numpy() - 1] - effects["running"][x.index[-1].month - 1]
        adjusted = x * np.exp(-offset)
        inner = replace(task, histories={**task.histories, "CPI_ALL": adjusted}, panel_overrides={"CPI_ALL": adjusted})
        plain, _ = fc.predict(inner, fc.Settings(draws=512, monthly_seasonal=False))
        shift = fc.seasonal_offset(effects, x.index[-1], task.steps[0, 0])
        np.testing.assert_array_equal(got, plain * np.exp(shift))
        self.assertEqual(model.audit["seasonal"]["CPI_ALL"]["offsets"], [shift])
        self.assertTrue(model.audit["requested_settings"]["monthly_seasonal"])
        off, _ = fc.predict(task, fc.Settings(draws=512, **fc.V2_7_4_SETTINGS))
        v274, _ = V2_7_4_REFERENCE.predict(V2_7_4_REFERENCE.load_task(unit), V2_7_4_REFERENCE.Settings(draws=512))
        np.testing.assert_array_equal(off, v274)
        self.assertFalse(np.array_equal(got, off))

    def test_failures_fall_back_to_the_plain_forecast(self):
        task = fc.load_task(seasonal_unit(self.tmp / "nsa"))
        plain, _ = fc.predict(task, fc.Settings(draws=512, monthly_seasonal=False))
        with mock.patch.object(fc, "seasonal_offset", side_effect=RuntimeError("boom")):
            got, model = fc.predict(task, fc.Settings(draws=512))
        np.testing.assert_array_equal(got, plain)
        self.assertTrue(model.audit["seasonal"].startswith("not applied: RuntimeError"))

    def test_cli_profile_v2_7_4_reproduces_v2_7_4(self):
        unit = seasonal_unit(self.tmp / "nsa")
        new, old = self.tmp / "new" / "forecast.parquet", self.tmp / "old" / "forecast.parquet"
        args = ["forecast", "--panels", str(unit), "--draws", "512", "--seed", "0"]
        self.assertEqual(fc.main([*args, "--out", str(new), "--profile", "v2.7.4"]), 0)
        self.assertEqual(V2_7_4_REFERENCE.main([*args, "--out", str(old)]), 0)
        np.testing.assert_array_equal(pd.read_parquet(new).value.to_numpy(), pd.read_parquet(old).value.to_numpy())



class ShortHorizonPracticeSet(unittest.TestCase):
    """The 22 official short-horizon practice units (2026-10-18). They run without fallback, v2.7.5 equals v2.7.4
    on each, the FOMC-week profile fires on exactly the FOMC-week Monday cards, and the v2.7.1 profile reproduces
    v2.7.1 there."""

    FOMC_WEEK = {"t2-F2-curve-ust1y-003-ps1", "t2-F2-curve-ust6m-004-ps1", "t2-F2-fx-brl-012-ps1",
                 "t2-F2-fx-krw-015-ps1", "t2-F2-fx-zar-014-ps1", "t2-F3-curve-ust3m-ust6m-ust1y-006-ps1",
                 "t2-F3-curve-ust3m-ust6m-ust1y-ust2y-005-ps1", "t2-F3-fx-krw-inr-zar-017-ps1",
                 "t2-F3-fx-mxn-brl-zar-016-ps1"}

    def test_the_sets_are_the_official_ones(self):
        self.assertEqual(len(PS1_CARDS), 22)
        self.assertEqual(len(ORIGINAL_CARDS), 104)

    def test_the_profile_fires_on_exactly_the_fomc_week_mondays(self):
        fired = {u.name for u in PS1_CARDS if fc.event_profile_reason(fc.load_task(u)) is None}
        self.assertEqual(fired, self.FOMC_WEEK)

    def test_every_card_runs_clean_and_older_profiles_reproduce_their_releases(self):
        for unit in PS1_CARDS:
            task = fc.load_task(unit)
            new, model = fc.predict(task, fc.Settings(draws=512))
            self.assertFalse(model.audit.get("fallback"), unit.name)
            self.assertTrue(np.isfinite(new).all(), unit.name)
            new, _ = fc.predict(task, fc.Settings(draws=512, **fc.V2_7_5_SETTINGS))
            v274, _ = V2_7_4_REFERENCE.predict(V2_7_4_REFERENCE.load_task(unit), V2_7_4_REFERENCE.Settings(draws=512))
            np.testing.assert_array_equal(new, v274, err_msg=unit.name)
            old, _ = V2_7_1_REFERENCE.predict(V2_7_1_REFERENCE.load_task(unit), V2_7_1_REFERENCE.Settings(draws=512))
            pinned, _ = fc.predict(task, fc.Settings(draws=512, **fc.V2_7_1_SETTINGS))
            np.testing.assert_array_equal(pinned, old, err_msg=unit.name)



V2_7_5_REFERENCE = load(HERE.parent / "forecast-v2_7_5" / "forecast.py", "forecast_v2_7_5_reference")
FOMC_RATES_PS1 = ["t2-F2-curve-ust1y-003-ps1", "t2-F2-curve-ust6m-004-ps1", "t2-F3-curve-ust3m-ust6m-ust1y-006-ps1",
                  "t2-F3-curve-ust3m-ust6m-ust1y-ust2y-005-ps1"]


def monthly_unit(target: Path, target_values: pd.Series, siblings: dict[str, pd.Series], period: str) -> Path:
    """A one-target monthly card (panel macro_monthly) whose panel also carries `siblings`."""
    target.mkdir(parents=True)
    rows = [pd.DataFrame({"date": s.index.strftime("%Y-%m-%d"), "asset": name, "value": s.to_numpy(),
                          "panel_id": "macro_monthly"}) for name, s in {"TARGET": target_values, **siblings}.items()]
    pd.concat(rows, ignore_index=True).to_parquet(target / "macro_monthly.parquet", index=False)
    asof = (pd.Period(period, freq="M").to_timestamp() + pd.offsets.MonthEnd(1) + pd.Timedelta(days=26)).strftime("%Y-%m-%d")
    (target / "card.toml").write_text(
        '[task]\nid = "t2-F1-nowcast-test"\n[metadata]\ncategory = "T2-F1"\n'
        f'[provenance]\ndata_cutoff = "{asof}"\n[scoring.params]\nn_draws_min = 200\n'
        '[targets]\nasset_ids = ["TARGET"]\nhorizons = [3]\ntarget_type = "level"\n')
    (target / "forecast_spec.json").write_text(json.dumps({"targets": {
        "asset_ids": ["TARGET"], "horizons": [3], "target_type": "level", "observation_periods": [period]}}))
    return target


class FomcMeanAndNowcast(unittest.TestCase):
    """v2.7.6: E37 (FOMC-week rates centre) and E40b (monthly nowcast from a stable sibling series)."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="v276-"))
        rng = np.random.default_rng(76)
        self.index = pd.date_range("2006-01-01", periods=200, freq="MS")
        self.sib_change = rng.normal(0.25, 0.30, 200)
        self.tgt_change = 0.05 + 0.7 * self.sib_change + rng.normal(0.0, 0.08, 200)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def series(self, changes: np.ndarray, n: int) -> pd.Series:
        return pd.Series(100.0 + np.cumsum(changes[:n]), index=self.index[:n])

    def test_defaults_profiles_and_validation(self):
        s = fc.Settings()
        self.assertTrue(s.fomc_mean_shift)
        self.assertTrue(s.monthly_nowcast)
        for profile in (fc.V2_7_5_SETTINGS, fc.V2_7_4_SETTINGS, fc.V2_7_3_SETTINGS, fc.V2_7_2_SETTINGS, fc.V2_7_SETTINGS,
                        fc.V2_6_1_SETTINGS, legacy(), v25(), fc.POST_PROCESSING_OFF):
            settings = fc.Settings(**profile)
            self.assertFalse(settings.fomc_mean_shift)
            self.assertFalse(settings.monthly_nowcast)
        for bad in ({"fomc_mean_shift": 1}, {"monthly_nowcast": "yes"}):
            with self.assertRaises(ValueError, msg=str(bad)):
                fc.Settings(**bad)

    def test_public_cards_change_only_on_the_fomc_week_rates_cards(self):
        changed = []
        for unit in CARDS:
            task = fc.load_task(unit)
            new, model = fc.predict(task, fc.Settings(draws=512))
            old, _ = V2_7_5_REFERENCE.predict(V2_7_5_REFERENCE.load_task(unit), V2_7_5_REFERENCE.Settings(draws=512))
            pinned, _ = fc.predict(task, fc.Settings(draws=512, **fc.V2_7_5_SETTINGS))
            np.testing.assert_array_equal(pinned, old, err_msg=unit.name)
            if not np.array_equal(new, old):
                changed.append(unit.name)
                self.assertIn("kappa", model.audit["post_processing"]["fomc_mean"], unit.name)
        self.assertEqual(sorted(changed), sorted(FOMC_RATES_PS1))

    def test_rates_cells_shift_by_kappa_times_their_sd(self):
        unit = UNITS / "t2-F3-curve-ust3m-ust6m-ust1y-ust2y-005-ps1"  # bills and the 2Y in an FOMC week
        task = fc.load_task(unit)
        self.assertIsNone(fc.event_profile_reason(task))
        new, model = fc.predict(task, fc.Settings(draws=512))
        plain, _ = fc.predict(task, fc.Settings(draws=512, fomc_mean_shift=False))
        for ai, asset in enumerate(task.assets):
            kappa = fc.FOMC_MEAN_KAPPA[fc.event_asset_class(asset)]
            shift = np.broadcast_to(kappa * plain[:, ai, :].std(axis=0), plain[:, ai, :].shape)
            np.testing.assert_allclose(new[:, ai, :] - plain[:, ai, :], shift, rtol=0, atol=1e-12)
        self.assertEqual(set(model.audit["post_processing"]["fomc_mean"]["kappa"]), set(task.assets))
        fx = fc.load_task(UNITS / "t2-F3-fx-krw-inr-zar-017-ps1")  # an FX card in an FOMC week: no shift
        a, model = fc.predict(fx, fc.Settings(draws=512))
        b, _ = fc.predict(fx, fc.Settings(draws=512, fomc_mean_shift=False))
        np.testing.assert_array_equal(a, b)
        self.assertTrue(model.audit["post_processing"]["fomc_mean"].startswith("not applied"))

    def test_nowcast_fires_on_a_stable_sibling_that_has_the_target_month(self):
        target = self.series(self.tgt_change, 199)  # ends one month before the sibling
        sibling = self.series(self.sib_change, 200)
        period = str(self.index[199].to_period("M"))
        task = fc.load_task(monthly_unit(self.tmp / "fires", target, {"SIBLING": sibling}, period))
        new, model = fc.predict(task, fc.Settings(draws=512, **fc.V2_7_6_SETTINGS))  # E40b's width; E41 has its own test
        info = model.audit["post_processing"]["nowcast"]["TARGET"]
        self.assertEqual(info["sibling"], "SIBLING")
        x, y = np.diff(sibling.to_numpy())[-120 - 1:-1], np.diff(target.to_numpy())[-120:]
        b, a = np.polyfit(x, y, 1)
        centre = target.iloc[-1] + a + b * (sibling.iloc[-1] - sibling.iloc[-2])
        self.assertAlmostEqual(info["cells"][0]["centre"], centre, places=9)
        self.assertAlmostEqual(float(np.median(new)), centre, places=6)
        sigma = float((y - (a + b * x)).std(ddof=2))
        self.assertAlmostEqual(info["cells"][0]["spread"], sigma * 1.1, places=9)
        self.assertAlmostEqual(float(new.std()), sigma * 1.1, places=6)
        core, _ = fc.predict_core(task, fc.Settings(draws=512))
        self.assertFalse(np.array_equal(new, core))

    def test_nowcast_needs_the_target_month_and_a_stable_correlation(self):
        target = self.series(self.tgt_change, 199)
        period = str(self.index[199].to_period("M"))
        same_end = self.series(self.sib_change, 199)  # no month beyond the target's last
        task = fc.load_task(monthly_unit(self.tmp / "same", target, {"SIBLING": same_end}, period))
        new, model = fc.predict(task, fc.Settings(draws=512))
        core, _ = fc.predict_core(task, fc.Settings(draws=512))
        np.testing.assert_array_equal(new, core)
        self.assertEqual(model.audit["post_processing"]["nowcast"], {})
        unstable = self.sib_change.copy()
        unstable[:140] = np.random.default_rng(7).normal(0.25, 0.30, 140)  # unrelated in the window's first half
        task = fc.load_task(monthly_unit(self.tmp / "unstable", target, {"SIBLING": self.series(unstable, 200)}, period))
        new, model = fc.predict(task, fc.Settings(draws=512))
        core, _ = fc.predict_core(task, fc.Settings(draws=512))
        np.testing.assert_array_equal(new, core)

    def test_nowcast_works_in_level_space_on_a_rate_that_crosses_zero(self):
        rate = pd.Series(0.1 + 0.7 * self.sib_change[:199] - 0.2 + np.random.default_rng(3).normal(0, 0.05, 199),
                         index=self.index[:199])
        sibling = pd.Series(self.sib_change - 0.25, index=self.index)
        self.assertTrue((rate < 0).any() and (rate > 0).any())
        period = str(self.index[199].to_period("M"))
        task = fc.load_task(monthly_unit(self.tmp / "rate", rate, {"SIBLING": sibling}, period))
        new, model = fc.predict(task, fc.Settings(draws=512))
        self.assertTrue(np.isfinite(new).all())
        self.assertEqual(model.audit["post_processing"]["nowcast"]["TARGET"]["sibling"], "SIBLING")

    def test_cli_profile_v2_7_5_reproduces_v2_7_5(self):
        unit = UNITS / FOMC_RATES_PS1[0]
        new, old = self.tmp / "new" / "forecast.parquet", self.tmp / "old" / "forecast.parquet"
        args = ["forecast", "--panels", str(unit), "--draws", "512", "--seed", "0"]
        self.assertEqual(fc.main([*args, "--out", str(new), "--profile", "v2.7.5"]), 0)
        self.assertEqual(V2_7_5_REFERENCE.main([*args, "--out", str(old)]), 0)
        np.testing.assert_array_equal(pd.read_parquet(new).value.to_numpy(), pd.read_parquet(old).value.to_numpy())



V2_7_6_REFERENCE = load(HERE.parent / "forecast-v2_7_6" / "forecast.py", "forecast_v2_7_6_reference")


class NowcastWidth(unittest.TestCase):
    """v2.7.7: E41, the nowcast's spread with prediction error, recent residual volatility and a constant per kind."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="v277-"))
        rng = np.random.default_rng(77)
        self.index = pd.date_range("2006-01-01", periods=200, freq="MS")
        self.sib_change = rng.normal(0.25, 0.30, 200)
        self.tgt_change = 0.05 + 0.7 * self.sib_change + rng.normal(0.0, 0.08, 200)
        self.tgt_change[-30:] += rng.normal(0.0, 0.2, 30)  # a noisier recent spell
        self.sib_change[-1] = 2.0  # the sibling's target-month change lies outside the window's range

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def card(self) -> tuple[pd.Series, pd.Series, Path]:
        target = pd.Series(100.0 + np.cumsum(self.tgt_change[:199]), index=self.index[:199])
        sibling = pd.Series(100.0 + np.cumsum(self.sib_change), index=self.index)
        unit = monthly_unit(self.tmp / "fires", target, {"SIBLING": sibling}, str(self.index[199].to_period("M")))
        return target, sibling, unit

    def test_defaults_profiles_and_validation(self):
        self.assertTrue(fc.Settings().nowcast_width)
        for profile in (fc.V2_7_6_SETTINGS, fc.V2_7_5_SETTINGS, fc.V2_7_4_SETTINGS, fc.V2_7_3_SETTINGS, fc.V2_7_2_SETTINGS,
                        fc.V2_7_SETTINGS, fc.V2_6_1_SETTINGS, legacy(), v25(), fc.POST_PROCESSING_OFF):
            self.assertFalse(fc.Settings(**profile).nowcast_width)
        with self.assertRaises(ValueError):
            fc.Settings(nowcast_width=1)

    def test_target_kind(self):
        idx = pd.period_range("2010-01", periods=150, freq="M")
        steps = np.random.default_rng(1).normal(0.2, 0.1, 150)
        self.assertEqual(fc.nowcast_target_kind(pd.Series(100 + np.cumsum(steps), index=idx)), "index")
        self.assertEqual(fc.nowcast_target_kind(pd.Series(2 + np.sin(np.arange(150) / 10), index=idx)), "persistent")
        self.assertEqual(fc.nowcast_target_kind(pd.Series(steps, index=idx)), "change")
        # a smooth level that goes below zero is not an index
        self.assertEqual(fc.nowcast_target_kind(pd.Series(np.cumsum(steps) - 5, index=idx)), "persistent")

    def test_spread_formula_and_the_unchanged_centre(self):
        target, sibling, unit = self.card()
        task = fc.load_task(unit)
        new, model = fc.predict(task, fc.Settings(draws=512))
        old, old_model = fc.predict(task, fc.Settings(draws=512, **fc.V2_7_6_SETTINGS))
        info = model.audit["post_processing"]["nowcast"]["TARGET"]
        self.assertEqual((info["kind"], info["width"]), ("index", "E41"))
        x, y = np.diff(sibling.to_numpy())[-120 - 1:-1], np.diff(target.to_numpy())[-120:]
        b, a = np.polyfit(x, y, 1)
        resid = y - (a + b * x)
        sigma, s24 = resid.std(ddof=2), np.sqrt(np.mean(resid[-24:] ** 2))
        lev = 1 + 1 / 120 + (2.0 - x.mean()) ** 2 / ((x - x.mean()) ** 2).sum()
        spread = np.sqrt(sigma ** 2 / 2 + s24 ** 2 / 2) * np.sqrt(lev) * fc.NOWCAST_KIND_MULT["index"]
        self.assertAlmostEqual(info["cells"][0]["spread"], spread, places=9)
        self.assertAlmostEqual(float(new.std()), spread, places=6)
        self.assertGreater(spread, sigma * 1.1)  # the noisy recent spell and the out-of-range move widen it
        before = old_model.audit["post_processing"]["nowcast"]["TARGET"]["cells"][0]
        self.assertEqual(info["cells"][0]["centre"], before["centre"])
        self.assertAlmostEqual(float(np.median(new)), float(np.median(old)), places=9)
        reference, _ = V2_7_6_REFERENCE.predict(V2_7_6_REFERENCE.load_task(unit), V2_7_6_REFERENCE.Settings(draws=512))
        np.testing.assert_array_equal(old, reference)

    def test_cli_profile_v2_7_6_reproduces_v2_7_6(self):
        _, _, unit = self.card()
        new, old = self.tmp / "new" / "forecast.parquet", self.tmp / "old" / "forecast.parquet"
        args = ["forecast", "--panels", str(unit), "--draws", "512", "--seed", "0"]
        self.assertEqual(fc.main([*args, "--out", str(new), "--profile", "v2.7.6"]), 0)
        self.assertEqual(V2_7_6_REFERENCE.main([*args, "--out", str(old)]), 0)
        np.testing.assert_array_equal(pd.read_parquet(new).value.to_numpy(), pd.read_parquet(old).value.to_numpy())

    def test_every_public_card_equals_v2_7_6(self):
        for unit in CARDS:
            new, _ = fc.predict(fc.load_task(unit), fc.Settings(draws=256))
            old, _ = V2_7_6_REFERENCE.predict(V2_7_6_REFERENCE.load_task(unit), V2_7_6_REFERENCE.Settings(draws=256))
            np.testing.assert_array_equal(new, old, err_msg=unit.name)


if __name__ == "__main__":
    unittest.main()
