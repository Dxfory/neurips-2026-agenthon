#!/usr/bin/env python
"""Numerical forecaster v2.7.6: the V3.3 engine plus M0-aware post-processing.

Changes against forecast-v1 (each has a Settings switch; all off reproduces V3.3.1):
- pair_match: on two-cell grids, match the one variogram pair to M0's (see _pair_match).
- mix_weight: part of the draws come from M0's mean/covariance with Student-t shocks.
- f4_widen: shock-family (T2-F4) daily cards use a wider width scale.
- robust_fallback: failures fall back to the published M0 procedure, never a zero grid.
- sharpen (v2.1): per family, resample draws by 1 / CRPS_M0^alpha, then scale by k (see m0_sharpen).
- monthly_m0_path (v2.2): fresh monthly cards get an M0-t mixture and sharpening, and the embedded M0
  walks the card's observation months. monthly_m0_path=False reproduces v2.1.0; with sharpen=False too, v1.2.0.
- v2.3: the default F4 pair widens from (1.0, 1.1) to (1.0, 1.25); sharpen_f4=(1.0, 1.1) reproduces v2.2.0.
- transfer_m0 (v2.4): cards whose every target is a transfer asset take Gaussian draws on M0's moments,
  each cell's sd the geometric mean of M0's and the G10 proxies' (see transfer_m0_draws).
  transfer_m0="off" reproduces v2.3.0.
- v2.5 daily width (exp/README.md, E11): the current step scale is the 60-step EWMA alone (vol_blend 1.0) and
  the current/long ratio is applied with exponent 1 at every horizon (vol_beta_base 1.0, vol_beta_slope 0.0),
  so a cell's width is the current scale times sqrt(steps); cells longer than 63 BD are widened by
  (steps / 63) ** 0.3 (horizon_scale_anchor, horizon_scale_power, horizon_scale_widen_only). V2_4_SETTINGS
  (and --profile v2.4) reproduce v2.4.0 bit for bit; the earlier profiles carry those settings too.
- v2.6 (exp/README.md, E13) is calibrated under the scoring rule of 6 October 2026, which divides by M0's expected
  error and makes the score proper: F1's draws are no longer reweighted toward M0's centre (sharpen_f1 (0.0, 1.1)),
  F4's are not either and F4's engine width comes down to 1.1 with k 1.0 (f4_width_scale, sharpen_f4), the M0-t
  mixture share is 0.3 (mix_weight) and the stochastic-volatility overlay is off (vol_sd 0). V2_5_SETTINGS (and
  --profile v2.5) reproduce v2.5.0 bit for bit; v2.5.1's F3 choice (f3_width_profile "v2.4") is not taken.
- v2.6.1 (exp/README.md, E14, the team's shock-share choice after the v2.6.0 Development result): F2 and F3 are
  scaled by k 1.2 (sharpen_f2, sharpen_f3), F4 by k 1.1 at engine width 1.3 (sharpen_f4, f4_width_scale), all
  with alpha 0. V2_6_SETTINGS (and --profile v2.6) reproduce v2.6.0 bit for bit.
- v2.7 (exp/README.md, E21): on daily level cells whose target is a Treasury tenor (UST_<n>Y on rates_daily) the
  final draws are shifted by half of the expectations-hypothesis forward increment f(h, T) - y(T) read from the
  unit's own as-of curve (rates_forward_add 0.5; see rates_forward_center). V2_6_1_SETTINGS (and --profile
  v2.6.1) reproduce v2.6.1 bit for bit; the earlier profiles carry the switch off too.
- v2.7.1 (exp/README.md, E23): the ranked Final is UST and G10 FX at 2 to 12 business days, where v2.6.1's F4
  widening is too wide; F4 returns to v2.6.0's engine width 1.1 with k 1.0 (f4_width_scale, sharpen_f4). F2/F3
  keep k 1.2. V2_7_SETTINGS (and --profile v2.7) reproduce v2.7.0 bit for bit.
- v2.7.2 (the ranked Final's card shapes of 12 October 2026): monthly series count as fresh when their last
  period is no earlier than the month 120 days before the as-of (monthly_publication_lag_days 120; at the Final's
  as-of, 45 marked a data-release card stale when its panel ended two months before the as-of month). Panels spaced more than
  daily (weekly jobless claims) no longer lose every innovation to the 5-day hole rule (gap_limit_days), and
  their steps count prints to the as-of plus the key in US business days, as M0 counts them (_m0_step_count).
  A monthly card with one key below 21 and no observation month walks one step, its next figure
  (_next_figure_card). The card's as-of is read where the scorer reads it, [forecast].asof first (declared_asof),
  and the embedded M0 falls back to the task's as-of. V2_7_1_SETTINGS (and --profile v2.7.1) reproduce v2.7.1 on
  every daily and monthly card; the cadence and as-of rules have no switch.
- v2.7.3 (exp/README.md, E24): on a daily level card whose as-of is the Monday of an FOMC week (decision on the
  Wednesday), whose steps all lie in 1-4 and whose every target is a Treasury tenor, a bill or one of the ranked
  Final's 15 exchange rates, every family width is reset (k 1.0, F4 at the default width) and a step-variance
  profile estimated on 2005-2018 FOMC weeks is applied to the draws as a path transform (EVENT_PROFILE,
  event_profile_reason, apply_event_profile): calm on the Tuesday, the decision at the Treasury close on the
  Wednesday and at the ECB fixing on the Thursday. Two-cell cards are no longer pair-matched to M0's variogram
  value (pair_match False; exp/README.md, E25: a default flipped, no new data). V2_7_2_SETTINGS (and
  --profile v2.7.2) reproduce v2.7.2.
- v2.7.4 (the review of 17 October): day counts no longer depend on the stored datetime resolution (epoch_ns).
  The pinned pandas 2.3.3 stores these dates in nanoseconds, so every card is unchanged there. Under pandas 3,
  which can store microseconds, `asi8 / 86400e9` had silently undone the weekly repair, and the seed digest
  differed. The draws on all 104 public cards are now the same under pandas 2.3.3 and 3.0.5. The FOMC-week
  profile's cross-class correlation depends on the horizons a card requests; that is documented and kept
  (FomcWeekProfile test, README).
  - E29: where the profile applies, UST-notes cells keep their family widths when the card's lowest UST yield
    is at least 1% (ust_family_widths_active, ust_active_floor; keep_ust_family_widths).
  - E30 (post hoc, on by the user's call of 18 October): monthly_price_vol_switch skips the monthly reweighting on price-index cards
    in a high-volatility regime (PRICE_NAMES, monthly_regime_ratio).
  - M0's step count keeps the declared key at a ratio of exactly 2 (official commit 24582e9).
  - V2_7_3_SETTINGS (and --profile v2.7.3) reproduce v2.7.3.
- v2.7.5 (exp/README.md, E35 and E36): a monthly target whose last 96 changes differ by calendar month
  (Kruskal-Wallis p < 0.001; seasonal_effects) is forecast seasonally adjusted, and its draws are put back on the
  calendar (monthly_seasonal, seasonal_plan, predict_seasonal). Every other step reads the adjusted series, the
  embedded M0 through an in-memory override (TaskContext.panel_overrides), and the seed fingerprint is the card's
  own. Seasonally adjusted series and year-on-year rates do not trigger it: no public card changes, and neither
  does any task of the adjusted backtest rosters. V2_7_4_SETTINGS (and --profile v2.7.4) reproduce v2.7.4.
- v2.7.6 (exp/README.md, E37 and E40b), applied after everything above (predict_core, then predict):
  - E37 (fomc_mean_shift): on a card the FOMC-week profile reaches, each Treasury-note and bill cell shifts by
    FOMC_MEAN_KAPPA x its SD. Yields fall in FOMC weeks on average (estimated on 2005-18 FOMC weeks).
  - E40b (monthly_nowcast): a monthly target whose panel holds a sibling series that already has the target month,
    with a stable correlation of monthly changes, takes the regression nowcast (monthly_nowcast_fit).
  - V2_7_5_SETTINGS (and --profile v2.7.5) reproduce v2.7.5.
- v2.7.7 (exp/README.md, E41): E40b's width only (nowcast_width). The centre, the sibling and the draws' shape stay;
  the spread sigma*sqrt(k)*1.1 becomes sqrt(sigma^2/2 + s24^2/2)*sqrt(k + k^2/n + (X - k*xbar)^2/Sxx)*c:
  recent residual volatility (the window's last 24 months) shrunk halfway to the window's, the OLS prediction
  variance of the k-month sum, and a constant per kind of target (nowcast_target_kind: index level 1.2, persistent
  rate 1.1, monthly change 0.9), fitted on 2010-18. V2_7_6_SETTINGS (and --profile v2.7.6) reproduce v2.7.6.

No text or language-model calls. Evidence and data boundaries: ARTIFACT_PROVENANCE.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
import tomllib
import zlib
from dataclasses import asdict, dataclass, field, replace
from datetime import date, timedelta
from pathlib import Path
from time import monotonic
from typing import Any, Literal

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from numpy.typing import NDArray
from pandas.tseries.holiday import USFederalHolidayCalendar
from qfbench2_track_forecasting.horizons import HorizonMetadataError, monthly_horizon_steps
from qfbench2_track_forecasting.scoring import build_verifier
from qfbench2_track_forecasting.targets import log_return_steps
from scipy.special import ndtr


ENGINE_VERSION = "v2.7.7"

# Published M0 panel steps for the four monthly cards (M0-BASELINE.md §3.7, measured
# 2026-09-18). The submission still walks authored observation periods. A mismatch is
# recorded; the sealed target date is not invented.
_M0_PUBLISHED_MONTHLY_STEPS = {
    "t2-F1-cpi-glidepath-2023": (8, 9),
    "t2-F1-sahm-watch-2024": (8, 9),
    "t2-F4-covid-nfp-2020": (2,),
    "t2-F4-cpi-vintage-2022": (2,),
}


def require_finite_number(name: str, value: Any) -> None:
    """Reject booleans, strings and non-finite values before they enter a range check."""
    if type(value) not in (int, float):
        raise ValueError(f"{name} must be a finite number")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        raise ValueError(f"{name} must be a finite number")


def draw_floor(value: Any) -> int:
    """Keep a declared minimum draw count from being silently truncated."""
    if type(value) is not int or not 200 <= value <= 20_000:
        raise ContractError("n_draws_min must be an integer between 200 and 20000")
    return value


@dataclass(frozen=True)
class Settings:
    """Frozen numerical defaults plus explicit experimental controls.

    Daily center and width defaults follow the recorded V3 optimization experiments.
    Innovation-distribution controls remain inherited from V2.
    Standardization fixes latent sample variance, not quantile interval widths.
    Monthly settings are separate; the daily volatility overlay is not assumed validated
    at monthly cadence. None inherits an overlay amplitude; zero explicitly disables it.
    """
    draws: int = 4096
    seed: int = 0
    block_length: int = 21
    bootstrap_weight: float = 0.5
    student_df: float = 5.0
    empirical_width: float = 0.35
    center_guard: bool = True
    vol_sd: float = 0.0  # v2.6: the overlay is off (E13); v2.5 used 0.35
    vol_persist: float = 0.99
    vol_sd_rates: float | None = None
    vol_sd_fx: float | None = None
    vol_sd_factors: float | None = None
    vol_decay: float = 63.0
    vol_blend: float = 1.0
    # E11 evaluated vol_blend on daily fresh cards only; the monthly path and the transfer proxies keep v2.4's 0.8.
    monthly_vol_blend: float = 0.8
    transfer_vol_blend: float = 0.8
    vol_beta_base: float = 1.0
    vol_beta_slope: float = 0.0
    width_rates: float = 0.35
    width_fx: float = 0.35
    width_factors: float = 0.35
    # Daily width is scaled once by width_scale * (steps / 21)**power.
    # The daily experiment roster does not justify changing monthly widths.
    width_scale: float = 0.9
    # v2.5: the daily width factor is (steps / anchor) ** power, and with widen_only it is
    # max(1, steps / anchor) ** power, so cells at or under the anchor keep their width and only longer
    # cells widen. v2.4 was anchor 21, power 0.05, widen_only off (V2_4_SETTINGS).
    horizon_scale_power: float = 0.3
    horizon_scale_anchor: float = 63.0
    horizon_scale_widen_only: bool = True
    # The daily width settings a T2-F3 card runs with: "v2.5" (the default) the settings above; "v2.4" replaces
    # them by V2_4_SETTINGS for F3 cards only (v2.5.1's choice under the pre-6-October scoring rule).
    f3_width_profile: str = "v2.5"
    monthly_width_scale: float = 1.15
    monthly_block_length: int = 7
    monthly_student_df: float = 7.0
    monthly_vol_sd: float = 0.0
    monthly_vol_persist: float = 0.9
    # A freshness allowance, not an inferred publication date or permission to add missing inputs.
    # v2.7.2: 120 days. The practice macro panels document a 45-day staging lag, but a Final data-release
    # card's panel ends at the previous print (US PCE for September: August, as-of 26 October), which 45
    # marked stale: no drift, no M0-t share, no sharpening. v2.7.1 used 45 (V2_7_1_SETTINGS).
    monthly_publication_lag_days: int = 120
    # Structural guards, not fitted knobs. V1 control restores the previous caps.
    # 0.40 lets a trailing-window mean exist. M0's level drift is why a driftless
    # walk scored about 1.29 against the denominator. 0.15 clipped that drift away.
    center_cap: float = 0.40
    # Use one clipped daily drift across all horizons. Zero restores V3.2's
    # separate per-horizon clipping; V1 control also sets this to zero.
    drift_cap_horizon: int = 63
    use_trailing_drift: bool = True
    suppress_countertrend_reversion: bool = True
    transfer_floor: float = 2.0
    transfer_allow_narrow: bool = False
    monthly_deceleration: bool = True
    # v1.1 post-processing. On a two-cell grid the variogram has one pair and M0's
    # value can land near zero, so any other pairwise statistic makes the joint ratio
    # a lottery. Matching M0's own 500-draw value makes that ratio exactly one.
    # v2.7.3 (exp/README.md, E25): off. Since the rule of 6 October the joint divisor is M0's expected variogram
    # score, not its realised pair value, and the engine's own dependence scored better on two-cell cards at
    # 1-4 BD (FOMC weeks test -0.022, ordinary weeks -0.033 per two-cell task). V2_7_2_SETTINGS turns it on.
    pair_match: bool = False
    # Share of draws taken from M0's mean/covariance with Student-t shocks and a
    # damped drift. Daily cards with fresh histories only; 0 disables it.
    mix_weight: float = 0.3  # v2.6 (E13); v2.5 used 0.5
    mix_df: float = 5.0
    mix_drift: float = 0.5
    # Shock-family cards are chosen because their window holds a large move. On daily
    # T2-F4 cards the engine width scale and the M0-t scale both become this value.
    f4_widen: bool = True
    f4_width_scale: float = 1.1  # v2.7.1 (E23): v2.6.0's 1.1 for the 2-12 BD ranked Final; v2.6.1/v2.7.0 used 1.3
    # A failure falls back to the published M0 procedure, then to a local random walk.
    robust_fallback: bool = True
    # v2.1 M0-aware sharpening. The board averages per-card ratios to M0, so the forecast that
    # minimizes it is sharper than a calibrated one around M0's centre. Fresh daily draws are
    # resampled with weight 1 / CRPS_M0(x)^alpha, then each cell is scaled by k about its
    # median. One (alpha, k) pair per card family; (0.0, 1.0) leaves a family unchanged.
    # The defaults are the "final" entry of SHARPEN_PROFILES. On F4 the engine width stays
    # f4_width_scale (1.3); k applies after it. v2.3 widened F4's k from 1.1 to 1.25.
    sharpen: bool = True
    sharpen_f1: tuple[float, float] = (0.0, 1.1)  # v2.6: alpha off (E13); v2.5 used (0.75, 1.1)
    sharpen_f2: tuple[float, float] = (0.0, 1.2)  # v2.6.1 (E14); v2.6.0 (0.0, 1.0)
    sharpen_f3: tuple[float, float] = (0.0, 1.2)  # v2.6.1 (E14); v2.6.0 (0.0, 1.0)
    sharpen_f4: tuple[float, float] = (0.0, 1.0)  # v2.7.1 (E23): k 1.0 at engine width 1.1, as v2.6.0; v2.6.1/v2.7.0 (0.0, 1.1) at 1.3
    # v2.2 monthly path. v2.1 scored about 1.3 against M0 on a monthly macro backtest: its monthly
    # width ran 1.2-1.5x M0's while outcomes sat near M0's centre, and the embedded M0 guessed its
    # month counts from the business-day key, so two-cell pairing missed. Fresh monthly cards now
    # get this M0-t mixture and sharpening, the embedded M0 walks the card's observation months,
    # and pairing may move centres or blend as on fresh daily cards. False reproduces v2.1.0.
    monthly_m0_path: bool = True
    monthly_mix_weight: float = 0.5
    monthly_mix_df: float = 5.0
    monthly_mix_drift: float = 1.0
    monthly_mix_scale: float = 0.8
    monthly_sharpen_alpha: float = 0.5
    # v2.4 transfer cards. A transfer target ships only an early window ten years before the as-of
    # and the as-of row; the engine floors its width at twice the G10 proxies' current scale, which on
    # real EM targets scored about 1.9 against M0 (managed currencies move far less than G10 FX).
    # Cards whose every target is a transfer asset now take Gaussian draws on M0's mean and
    # covariance with each cell's sd set by the rule: "geo" the geometric mean of M0's and the G10
    # proxies' current scale, "m0" M0's times transfer_m0_scale, "off" the engine's draws (v2.3.0).
    transfer_m0: str = "geo"
    transfer_m0_scale: float = 0.85
    # v2.7 (E21): fraction of the curve-implied forward increment f(h, T) - y(T) added to the engine's centre on
    # Treasury cells (rates_forward_center). 0 with v2.7.0's F4 (V2_6_1_SETTINGS) reproduces v2.6.1; 1 is E21's full arm.
    rates_forward_add: float = 0.5
    # v2.7.3 (E24): on a daily level card whose as-of is the Monday of an FOMC week (decision on the Wednesday), whose
    # steps all lie in 1-4 and whose every target is a Treasury tenor, a bill or one of the ranked Final's 15 exchange
    # rates, the family widths are reset (k 1.0, F4 at the default width) and EVENT_PROFILE's step variances are
    # applied to the draws (event_profile_reason, apply_event_profile). False (V2_7_2_SETTINGS) reproduces v2.7.2.
    event_profile: bool = True
    # v2.7.4 (exp/README.md, E29): where the profile applies, UST-notes cells (2Y-30Y) keep their family widths
    # when the card's lowest UST yield at the as-of is at least ust_active_floor percent, i.e. away from the zero
    # lower bound. For those cells the profile goes on top of the family widths (E24's ev_k); every other cell is
    # as v2.7.3 has it. False (V2_7_3_SETTINGS) reproduces v2.7.3.
    ust_family_widths_active: bool = True
    ust_active_floor: float = 1.0
    # v2.7.4 (exp/README.md, E30, post hoc; switched on by the user's call of 18 October 2026): on a fresh monthly card
    # whose every target reads as a price index (PRICE_NAMES) and whose recent-to-long volatility ratio
    # (monthly_regime_ratio) is at least monthly_price_vol_ratio, the M0-centred reweighting (monthly_sharpen_alpha) is
    # skipped. False (V2_7_3_SETTINGS) reproduces v2.7.3.
    monthly_price_vol_switch: bool = True
    monthly_price_vol_ratio: float = 1.0
    # v2.7.5 (exp/README.md, E35 and E36): on a monthly card, a target whose last 96 changes differ by calendar month
    # (Kruskal-Wallis p < 0.001) is forecast seasonally adjusted, and its draws are put back on the calendar
    # (seasonal_plan, predict_seasonal). False (V2_7_4_SETTINGS) reproduces v2.7.4.
    monthly_seasonal: bool = True
    # v2.7.6 (exp/README.md, E37): on a card the FOMC-week profile reaches, Treasury-note and bill cells shift by
    # FOMC_MEAN_KAPPA times their SD (fomc_mean_shift). False (V2_7_5_SETTINGS) reproduces v2.7.5.
    fomc_mean_shift: bool = True
    # v2.7.6 (exp/README.md, E40 and E40b): a monthly target whose panel holds a stable sibling that already has its
    # target month is nowcast from it (monthly_nowcast_fit). False (V2_7_5_SETTINGS) reproduces v2.7.5.
    monthly_nowcast: bool = True
    # v2.7.7 (exp/README.md, E41): the nowcast's spread adds the regression's prediction error and recent residual
    # volatility and takes a constant per kind of target (monthly_nowcast). False (V2_7_6_SETTINGS) reproduces v2.7.6.
    nowcast_width: bool = True

    def __post_init__(self) -> None:
        if type(self.seed) is not int:
            raise ValueError("seed must be an integer")
        flags = {
            "center_guard": self.center_guard,
            "use_trailing_drift": self.use_trailing_drift,
            "suppress_countertrend_reversion": self.suppress_countertrend_reversion,
            "transfer_allow_narrow": self.transfer_allow_narrow,
            "monthly_deceleration": self.monthly_deceleration,
            "pair_match": self.pair_match,
            "f4_widen": self.f4_widen,
            "robust_fallback": self.robust_fallback,
            "sharpen": self.sharpen,
            "monthly_m0_path": self.monthly_m0_path,
            "horizon_scale_widen_only": self.horizon_scale_widen_only,
            "event_profile": self.event_profile,
            "ust_family_widths_active": self.ust_family_widths_active,
            "monthly_price_vol_switch": self.monthly_price_vol_switch,
            "monthly_seasonal": self.monthly_seasonal,
            "fomc_mean_shift": self.fomc_mean_shift,
            "monthly_nowcast": self.monthly_nowcast,
            "nowcast_width": self.nowcast_width,
        }
        for name, value in flags.items():
            if type(value) is not bool:
                raise ValueError(f"{name} must be a boolean")
        for name, low, high in (("ust_active_floor", -5.0, 20.0), ("monthly_price_vol_ratio", 0.5, 3.0)):
            value = getattr(self, name)
            require_finite_number(name, value)
            if not low <= value <= high:
                raise ValueError(f"invalid {name}")
        if self.transfer_m0 not in ("off", "m0", "geo"):
            raise ValueError("transfer_m0 must be off, m0 or geo")
        if self.f3_width_profile not in ("v2.5", "v2.4"):
            raise ValueError("f3_width_profile must be v2.5 or v2.4")
        pairs = ("sharpen_f1", "sharpen_f2", "sharpen_f3", "sharpen_f4")
        for name in pairs:
            value = getattr(self, name)
            if type(value) is list:
                # JSON overrides arrive as lists; the frozen record keeps tuples.
                value = tuple(value)
                object.__setattr__(self, name, value)
            if type(value) is not tuple or len(value) != 2:
                raise ValueError(f"{name} must be an (alpha, k) pair")
            for part in value:
                require_finite_number(name, part)
            if not 0.0 <= value[0] <= 2.0 or not 0.3 <= value[1] <= 3.0:
                raise ValueError(f"invalid {name}")
        optional = {"vol_sd_rates", "vol_sd_fx", "vol_sd_factors"}
        for name, value in asdict(self).items():
            if (name in flags or name in pairs or name in ("seed", "transfer_m0", "f3_width_profile")
                    or (name in optional and value is None)):
                continue
            require_finite_number(name, value)
        if not 0.0 <= self.rates_forward_add <= 1.0:
            raise ValueError("rates_forward_add must lie in [0, 1]")
        if type(self.draws) is not int or not 200 <= self.draws <= 20_000:
            raise ValueError("draws must be between 200 and 20000")
        if type(self.block_length) is not int or self.block_length < 1 or not 0 <= self.bootstrap_weight <= 1 or not self.student_df > 2:
            raise ValueError("invalid innovation distribution")
        if not 0 <= self.empirical_width <= 0.6:
            raise ValueError("invalid empirical width")
        if not 0 <= self.vol_sd <= 1.5 or not 0 <= self.vol_persist < 1:
            raise ValueError("invalid volatility overlay")
        if not all(v is None or 0 <= v <= 1.5 for v in (self.vol_sd_rates, self.vol_sd_fx, self.vol_sd_factors)):
            raise ValueError("invalid per-panel volatility overlay")
        if not self.vol_decay > 0 or not 0 <= self.vol_blend <= 1:
            raise ValueError("invalid volatility scale estimate")
        if not 0 <= self.monthly_vol_blend <= 1 or not 0 <= self.transfer_vol_blend <= 1:
            raise ValueError("invalid monthly or transfer volatility scale estimate")
        if not all(0 <= w <= 0.6 for w in (self.width_rates, self.width_fx, self.width_factors)):
            raise ValueError("invalid per-panel empirical width")
        if not 0.2 <= self.width_scale <= 3.0 or not 0.2 <= self.monthly_width_scale <= 3.0:
            raise ValueError("invalid width scale")
        if not -0.5 <= self.horizon_scale_power <= 0.5:
            raise ValueError("invalid horizon scale power")
        if not 1.0 <= self.horizon_scale_anchor <= 1000.0:
            raise ValueError("invalid horizon scale anchor")
        if type(self.drift_cap_horizon) is not int or not 0 <= self.drift_cap_horizon <= 1000:
            raise ValueError("invalid coherent drift cap")
        if not 0 <= self.vol_beta_base <= 1.5 or not -0.5 <= self.vol_beta_slope <= 0.5:
            raise ValueError("invalid volatility extrapolation exponent")
        if type(self.monthly_block_length) is not int or self.monthly_block_length < 1 or not self.monthly_student_df > 2:
            raise ValueError("invalid monthly innovations")
        if not 0 <= self.monthly_vol_sd <= 1.5 or not 0 <= self.monthly_vol_persist < 1:
            raise ValueError("invalid monthly overlay")
        if type(self.monthly_publication_lag_days) is not int or not 0 <= self.monthly_publication_lag_days <= 180:
            raise ValueError("invalid monthly freshness allowance")
        if not 0 <= self.center_cap <= 2.0:
            raise ValueError("invalid center cap")
        if not 1.0 <= self.transfer_floor <= 4.0:
            raise ValueError("invalid transfer width floor")
        if not 0.5 <= self.transfer_m0_scale <= 2.0:
            raise ValueError("invalid transfer M0 scale")
        if not 0.0 <= self.mix_weight <= 0.9 or not self.mix_df > 2 or not 0.0 <= self.mix_drift <= 1.5:
            raise ValueError("invalid M0-t mixture")
        if (not 0.0 <= self.monthly_mix_weight <= 0.9 or not self.monthly_mix_df > 2
                or not 0.0 <= self.monthly_mix_drift <= 1.5 or not 0.3 <= self.monthly_mix_scale <= 3.0
                or not 0.0 <= self.monthly_sharpen_alpha <= 2.0):
            raise ValueError("invalid monthly path")
        if not 0.2 <= self.f4_width_scale <= 3.0:
            raise ValueError("invalid shock-family width scale")

    def for_cadence(self, cadence: str) -> Settings:
        if cadence != "monthly":
            return self
        return replace(self, block_length=self.monthly_block_length, student_df=self.monthly_student_df,
                       vol_sd=self.monthly_vol_sd, vol_persist=self.monthly_vol_persist,
                       vol_sd_rates=None, vol_sd_fx=None, vol_sd_factors=None,
                       width_scale=self.monthly_width_scale, vol_blend=self.monthly_vol_blend)

    def overlay_for(self, panel: str) -> float:
        override = (self.vol_sd_rates if "rates" in panel else
                    self.vol_sd_fx if "fx" in panel or "transfer" in panel else
                    self.vol_sd_factors if "factors" in panel else None)
        return self.vol_sd if override is None else override

    def width_for(self, panel: str) -> float:
        if "rates" in panel:
            return self.width_rates
        if "fx" in panel or "transfer" in panel:
            return self.width_fx
        if "factors" in panel:
            return self.width_factors
        return self.empirical_width


def v1_settings(*, draws: int = 4096, seed: int = 0) -> Settings:
    """Full V1 numerical control on daily tasks; monthly freshness remains repaired.

    Center cap, countertrend reversion, transfer narrowing and monthly slope
    handling are restored so this profile can still match V1 samples. Fail-open
    writing is not part of the numerical profile and always stays on.
    """
    return Settings(
        draws=draws,
        seed=seed,
        vol_sd=0.0,
        vol_blend=0.8,
        vol_beta_base=0.0,
        block_length=7,
        student_df=7.0,
        width_scale=1.0,
        monthly_width_scale=1.0,
        horizon_scale_power=0.0,
        drift_cap_horizon=0,
        center_cap=0.75,
        use_trailing_drift=False,
        suppress_countertrend_reversion=False,
        transfer_floor=1.4,
        transfer_allow_narrow=True,
        monthly_deceleration=False,
        **POST_PROCESSING_OFF,
    )


# Every v1.1, v2.1, v2.2 and v2.4 switch off: the V3.3.1 numerical path.
POST_PROCESSING_OFF = {"pair_match": False, "mix_weight": 0.0, "f4_widen": False, "robust_fallback": False,
                       "sharpen": False, "monthly_m0_path": False, "transfer_m0": "off", "event_profile": False,
                       "ust_family_widths_active": False, "monthly_price_vol_switch": False,
                       "monthly_seasonal": False, "fomc_mean_shift": False, "monthly_nowcast": False,
                       "nowcast_width": False}

# Named sharpening profiles, (alpha, k) per family; they differ mainly on F4. "final" is the default. The other
# profiles are kept for comparison runs (`--profile`) and behave exactly as they did in v2.7.0 on daily cards, so
# they pin the v2.6.1 F4 engine width 1.3 that v2.7.1's default no longer uses; they take v2.7.2's monthly allowance,
# v2.7.3's FOMC-week profile, which replaces every family width where it applies, and v2.7.3's pair_match (off).
# The v2.4.0 daily width settings; Settings(**V2_4_SETTINGS) reproduces v2.4.0 bit for bit.
V2_4_SETTINGS = {"vol_blend": 0.8, "vol_beta_base": 0.77, "vol_beta_slope": 0.0735,
                 "horizon_scale_power": 0.05, "horizon_scale_anchor": 21.0, "horizon_scale_widen_only": False}

# The v2.7.6 value of the setting v2.7.7 adds; Settings(**V2_7_6_SETTINGS) reproduces v2.7.6 bit for bit on every card,
# and every earlier profile inherits it.
V2_7_6_SETTINGS = {"nowcast_width": False}
# The v2.7.5 values of the settings v2.7.6 adds; Settings(**V2_7_5_SETTINGS) reproduces v2.7.5 bit for bit on every card,
# and every earlier profile inherits them.
V2_7_5_SETTINGS = {**V2_7_6_SETTINGS, "fomc_mean_shift": False, "monthly_nowcast": False}
# The v2.7.4 value of the setting v2.7.5 adds; Settings(**V2_7_4_SETTINGS) reproduces v2.7.4 bit for bit on every card,
# and every earlier profile inherits it.
V2_7_4_SETTINGS = {**V2_7_5_SETTINGS, "monthly_seasonal": False}
# The v2.7.3 values of the settings v2.7.4 adds; Settings(**V2_7_3_SETTINGS) reproduces v2.7.3 bit for bit on every card
# (v2.7.4's date-unit repair changes no value under the pinned pandas), and every earlier profile inherits them.
V2_7_3_SETTINGS = {**V2_7_4_SETTINGS, "ust_family_widths_active": False, "monthly_price_vol_switch": False}
# The v2.7.2 values of the settings v2.7.3 adds or changes; Settings(**V2_7_2_SETTINGS) reproduces v2.7.2 bit for bit on
# every card, and every earlier profile below inherits them through V2_7_1_SETTINGS.
V2_7_2_SETTINGS = {**V2_7_3_SETTINGS, "event_profile": False, "pair_match": True}
# The v2.7.1 value of the setting v2.7.2 changes; Settings(**V2_7_1_SETTINGS) reproduces v2.7.1 bit for bit on every
# card whose panels are daily or monthly (all 104 public cards). Weekly-spaced panels take v2.7.2's gap and step rules.
V2_7_1_SETTINGS = {**V2_7_2_SETTINGS, "monthly_publication_lag_days": 45}
# The v2.7.0 values of the settings v2.7.1 changes; Settings(**V2_7_SETTINGS) reproduces v2.7.0 bit for bit on the
# same cards, and every earlier profile below inherits the 45-day allowance through it.
V2_7_SETTINGS = {**V2_7_1_SETTINGS, "sharpen_f4": (0.0, 1.1), "f4_width_scale": 1.3}
# With v2.7's forward switch off as well; Settings(**V2_6_1_SETTINGS) reproduces v2.6.1 bit for bit.
V2_6_1_SETTINGS = {**V2_7_SETTINGS, "rates_forward_add": 0.0}
# The v2.6.0 values of the settings v2.6.1 changed, with v2.7's switch off; Settings(**V2_6_SETTINGS) reproduces
# v2.6.0 bit for bit.
V2_6_SETTINGS = {**V2_6_1_SETTINGS, "sharpen_f2": (0.0, 1.0), "sharpen_f3": (0.0, 1.0), "sharpen_f4": (0.0, 1.0),
                 "f4_width_scale": 1.1}
# The v2.5.0 values of the settings v2.6 changed; Settings(**V2_6_SETTINGS, **V2_5_SETTINGS) (later keys winning)
# reproduces v2.5.0, and with V2_4_SETTINGS on top v2.4.0.
V2_5_SETTINGS = {"vol_sd": 0.35, "mix_weight": 0.5, "f4_width_scale": 1.3, "sharpen_f1": (0.75, 1.1),
                 "sharpen_f4": (1.0, 1.25), "f3_width_profile": "v2.5"}

SHARPEN_PROFILES = {
    "final": {"sharpen_f1": (0.0, 1.1), "sharpen_f2": (0.0, 1.2), "sharpen_f3": (0.0, 1.2),
              "sharpen_f4": (0.0, 1.0), "f4_width_scale": 1.1},
    # F4 variants on the v2.6.1 engine (F4 engine width 1.3, alpha 0 everywhere): narrower, wider, and the
    # random-date optimum of E13 for F4.
    "prospective": {"sharpen_f1": (0.0, 1.1), "sharpen_f2": (0.0, 1.2), "sharpen_f3": (0.0, 1.2),
                    "sharpen_f4": (0.0, 1.0), "f4_width_scale": 1.3},
    "balanced": {"sharpen_f1": (0.0, 1.1), "sharpen_f2": (0.0, 1.2), "sharpen_f3": (0.0, 1.2),
                 "sharpen_f4": (0.0, 1.25), "f4_width_scale": 1.3},
    "dev-leaning": {"sharpen_f1": (1.0, 0.9), "sharpen_f2": (0.0, 1.1), "sharpen_f3": (0.0, 1.0),
                    "sharpen_f4": (1.0, 2.0), "f4_width_scale": 1.3},
    "random-leaning": {"sharpen_f1": (0.0, 1.1), "sharpen_f2": (0.5, 1.0), "sharpen_f3": (0.5, 1.0),
                       "sharpen_f4": (0.0, 1.0), "f4_width_scale": 1.3},
}


class ContractError(ValueError):
    """The task cannot be interpreted without inventing information."""


@dataclass(frozen=True)
class TaskContext:
    root: Path
    unit_id: str
    asof: str
    assets: tuple[str, ...]
    horizons: tuple[int, ...]
    target_type: str
    units: str
    family: str
    card: dict[str, Any]
    spec: dict[str, Any]
    histories: dict[str, pd.Series]
    context_histories: dict[str, pd.Series]
    panel_for_asset: dict[str, str]
    cadence: Literal["daily", "monthly"]
    steps: NDArray[np.int64]
    anchors: NDArray[np.int64]
    min_draws: int
    fingerprint: str
    warnings: tuple[str, ...] = ()
    # v2.7.5: adjusted target histories that M0's reference reads in place of the panel (empty: the panel as given).
    panel_overrides: dict[str, pd.Series] = field(default_factory=dict, hash=False, compare=False)

    @property
    def cell_count(self) -> int:
        return len(self.assets) * len(self.horizons)


def _root(panels: Path) -> Path:
    for root in (panels, panels.parent):
        if (root / "card.toml").is_file():
            return root.resolve()
    raise ContractError("card.toml must be beside the panels directory or inside it")


def declared_asof(card: dict) -> Any:
    """The card's as-of where the scorer reads it (cutoff.trusted_asof): [forecast].asof, then
    [provenance].data_cutoff. None when the card states neither; issue #20 guarantees only [task],
    [targets] and n_draws_min."""
    for table, key in (("forecast", "asof"), ("provenance", "data_cutoff")):
        section = card.get(table)
        if isinstance(section, dict) and section.get(key):
            return section[key]
    return None


def monthly_index(index: pd.DatetimeIndex) -> bool:
    """Whether a series' dates are monthly: three or more, one per month, a median step of one month."""
    periods = index.to_period("M")
    return len(index) >= 3 and not periods.has_duplicates and np.median(np.diff(periods.asi8)) == 1


def load_task(panels: Path, asof: str | None = None) -> TaskContext:
    root = _root(panels)
    raw_card = (root / "card.toml").read_bytes()
    card = tomllib.loads(raw_card.decode())
    spec_path = root / "forecast_spec.json"
    spec = json.loads(spec_path.read_text()) if spec_path.exists() else {}
    warnings: list[str] = []
    cutoff = str(declared_asof(card) or card.get("text", {}).get("cutoff", ""))[:10]
    if asof is None:
        asof = cutoff
    date.fromisoformat(asof)
    if cutoff and asof != cutoff:
        # Gate g2 binds the declared as-of to the trusted card, so the card wins.
        warnings.append(f"--asof {asof} disagrees with card cutoff {cutoff}; using the card")
        asof = cutoff
    targets = card["targets"]
    assets, horizons = tuple(targets["asset_ids"]), tuple(targets["horizons"])
    if not assets or len(set(assets)) != len(assets) or not all(isinstance(a, str) for a in assets):
        raise ContractError("asset_ids must be unique strings")
    if not horizons or len(set(horizons)) != len(horizons):
        raise ContractError("horizons must be nonempty and unique")
    if any(type(h) is not int or h <= 0 for h in horizons):
        raise ContractError("horizon keys must be positive integers")
    if len(assets) * len(horizons) > 1000:
        raise ContractError("grid exceeds this agent's bounded memory budget")
    for key in ("asset_ids", "horizons", "target_type"):
        if key in spec.get("targets", {}) and spec["targets"][key] != targets.get(key):
            # The scorer reads the card; a spec that contradicts it is not used at all.
            warnings.append(f"card and spec disagree on {key}; the spec is ignored")
            spec = {}
            break
    paths = sorted(panels.glob("*.parquet"))
    if not paths:
        paths = sorted((root / "panels").glob("*.parquet")) or sorted(root.glob("*.parquet"))
    if not paths:
        raise ContractError("no input parquet panels found")
    histories: dict[str, pd.Series] = {}
    provenance: dict[str, str] = {}
    digest = hashlib.sha256(raw_card)
    for path in paths:
        if path.stat().st_size > 128 * 1024 * 1024:
            raise ContractError("input panel exceeds the bounded reader limit")
        table = pd.read_parquet(path)
        col = "asset" if "asset" in table else "asset_id"
        if not {col, "date", "value"} <= set(table.columns):
            raise ContractError(f"panel {path.name} has no date/asset/value columns")
        table = table[[col, "date", "value"]].copy()
        table["date"] = (
            pd.to_datetime(table["date"], errors="raise", utc=True)
            .dt.tz_localize(None)
            .dt.normalize()
        )
        if table["date"].isna().any():
            raise ContractError(f"panel {path.name} contains a missing date")
        after = table["date"] > pd.Timestamp(asof)
        if after.any():
            warnings.append(f"filtered {int(after.sum())} post-cutoff rows from {path.name}")
        table = table.loc[~after]
        identifiers = table[col]
        if identifiers.isna().any() or identifiers.astype(str).str.strip().eq("").any():
            raise ContractError(f"panel {path.name} contains a missing asset identifier")
        table["value"] = pd.to_numeric(table["value"], errors="raise")
        for asset, part in table.groupby(col, sort=False):
            asset = str(asset)
            # A missing row must not hide a real observation on the same date.
            part = part.dropna(subset=["value"]).sort_values("date", kind="stable")
            if part.groupby("date")["value"].nunique().gt(1).any():
                warnings.append(f"conflicting observations for {asset} in {path.name}; first row kept")
            part = part.drop_duplicates("date")
            s = part.set_index("date")["value"].astype(float)
            if not np.isfinite(s.to_numpy()).all():
                raise ContractError(f"non-finite observations for {asset}")
            if s.empty:
                continue
            if asset in histories:
                overlap = histories[asset].index.intersection(s.index)
                if not np.allclose(histories[asset].loc[overlap], s.loc[overlap], rtol=0, atol=0):
                    # M0 reads the first panel in sorted order; keep that one's values.
                    warnings.append(f"panels disagree about {asset}; {provenance[asset]} kept")
                histories[asset] = histories[asset].combine_first(s).sort_index()
            else:
                histories[asset] = s
                provenance[asset] = path.stem
    for asset in assets:
        if asset not in histories:
            anchor = spec.get("targets", {}).get("as_of_level", {}).get(asset)
            if anchor is None or not np.isfinite(float(anchor)):
                raise ContractError(f"target {asset} has neither a panel nor an explicit anchor")
            histories[asset] = pd.Series([float(anchor)], index=pd.DatetimeIndex([asof]))
            provenance[asset] = "transfer_anchor"
            warnings.append(f"{asset} has an explicit anchor but no history")
    # Cutoff-filtered canonical observations determine the seed, never withheld future rows
    # or parquet writer metadata. Input order also cannot change the random stream.
    for asset in sorted(histories):
        digest.update(asset.encode())
        digest.update(epoch_ns(histories[asset].index).tobytes())
        digest.update(histories[asset].to_numpy(dtype=np.float64).tobytes())
    selected = {a: histories[a] for a in assets}
    target_type = targets.get("target_type", "level")
    if target_type not in {"level", "yield", "log_return"}:
        raise ContractError("unsupported target_type")
    monthly = [monthly_index(s.index) for s in selected.values()]
    has_periods = any("observation_periods" in source.get("targets", {}) for source in (card, spec))
    if any(monthly) and not all(monthly):
        raise ContractError("mixed target cadence requires an explicit mixed-frequency contract")
    if all(monthly):
        cadence = "monthly"
        steps = resolve_monthly_steps(card["task"]["id"], assets, horizons, selected, asof, card, spec, warnings)
        anchors = np.array(
            [s.index[-1].to_period("M").ordinal for s in selected.values()], dtype=np.int64
        )
    else:
        if has_periods:
            warnings.append("monthly observation periods on daily target data are ignored")
        cadence = "daily"
        # M0 starts at the last observation and uses the declared horizon unless a
        # counted target date is at least 2x away. Released daily cards do not
        # publish that date, and on them the declared horizon wins. Do not add the
        # calendar gap between the last print and the as-of into the future path.
        anchors = np.zeros(len(assets), dtype=np.int64)
        steps = np.broadcast_to(np.asarray(horizons, dtype=np.int64), (len(assets), len(horizons))).copy()
        # v2.7.2: a weekly-type target (spacing over 5 and at most 20 days, e.g. jobless claims) walks the
        # prints M0 counts to the as-of plus the key in US business days (_m0_step_count), not the key.
        for ai, s in enumerate(selected.values()):
            window = s.index[-300:]
            spacing = _spacing(window)
            if spacing is not None and 5 < spacing <= 20:
                steps[ai] = [_m0_step_count(int(h), window[-1], spacing, pd.Timestamp(asof)) for h in horizons]
    if cadence == "monthly" and target_type == "log_return":
        raise ContractError("monthly cumulative-return semantics are not declared")
    published = _M0_PUBLISHED_MONTHLY_STEPS.get(card["task"]["id"])
    if published is not None:
        got = tuple(int(x) for x in np.asarray(steps).ravel())
        if got != tuple(published):
            warnings.append(
                "published M0 monthly steps "
                + str(tuple(published))
                + " differ from authored observation-period steps "
                + str(got)
                + "; walking the authored periods, not the sealed-date table"
            )
    if spec_path.exists():
        digest.update(spec_path.read_bytes())
    minimum = max(
        draw_floor(spec.get("n_draws_min", 200)),
        draw_floor(card.get("scoring", {}).get("params", {}).get("n_draws_min", 200)),
    )
    return TaskContext(
        root=root,
        unit_id=card["task"]["id"],
        asof=asof,
        assets=assets,
        horizons=horizons,
        target_type=target_type,
        units=targets.get("value_unit", "native panel units"),
        family=card.get("metadata", {}).get("category", "unknown"),
        card=card,
        spec=spec,
        histories=selected,
        context_histories=histories,
        panel_for_asset=provenance,
        cadence=cadence,
        steps=steps,
        anchors=anchors,
        min_draws=minimum,
        fingerprint=digest.hexdigest(),
        warnings=tuple(warnings),
    )


USD_PER_CCY = {"EUR", "GBP", "AUD", "NZD"}


CCY_PER_USD = {"JPY", "CHF", "CAD", "SEK", "NOK", "DKK", "BRL", "INR", "CNY", "CNH", "MXN", "ZAR", "TRY", "KRW"}


FX_NAMES = USD_PER_CCY | CCY_PER_USD


@dataclass(frozen=True)
class Transform:
    kind: str
    asset: str
    quote: str

    def state(self, series: pd.Series) -> pd.Series:
        if self.kind == "log_return":
            return pd.Series(log_return_steps(series.to_numpy()), index=series.index)
        if self.kind == "log_level":
            if (series <= 0).any():
                raise ContractError(f"{self.asset} must have positive values for a log-level model")
            return np.log(series)
        return series.copy()

    def invert(self, values: np.ndarray) -> np.ndarray:
        if self.kind == "log_level":
            if np.max(values) > 700:
                raise ContractError("simulated log level would overflow")
            return np.exp(values)
        return values


def transform_for(task: TaskContext, asset: str) -> Transform:
    """Choose a state transform from the contract, then from the asset identity.

    ``log_return`` targets are cumulative log(1+r) anchored at zero. Named FX
    quotes are log-levels. A panel stem is only a fallback for a context series
    whose name is not in the currency set. An unrecognized level series stays a
    level; it is not a reason to refuse the card. Factor panels are not forced
    through log1p unless the card's target_type says so.
    """
    if task.target_type == "log_return" and asset in task.assets:
        return Transform("log_return", asset, "cumulative log(1 + simple return), anchored at zero")
    if asset in USD_PER_CCY:
        return Transform("log_level", asset, "USD per currency")
    if asset in CCY_PER_USD:
        return Transform("log_level", asset, "currency per USD")
    panel = task.panel_for_asset.get(asset, "")
    if "fx" in panel or "transfer" in panel:
        return Transform("log_level", asset, "panel role fallback; native quote")
    return Transform("level", asset, task.units or "native level")


def trailing_step_mean(inc: pd.Series, window: int = 300, shrink: float = 30.0) -> float:
    """Shrunk mean of the last 300 gap-free steps. M0 uses this window with no shrink.

    The shrink is only a short-sample guard. At 300 observations it keeps about 90%
    of the mean, so a hiking-cycle drift is not rounded down to zero.
    """
    x = inc.dropna().to_numpy(dtype=float)
    x = x[np.isfinite(x)][-int(window) :]
    n = int(x.size)
    if n < 5:
        return 0.0
    return float(np.mean(x)) * (n / (n + float(shrink)))


def epoch_ns(index: pd.DatetimeIndex) -> NDArray[np.int64]:
    """Nanoseconds since the epoch, whatever the index's stored resolution.

    `asi8` is in the index's own unit. The pinned pandas 2.3.3 stores these dates in nanoseconds, so this is
    `asi8` there. pandas 3 can store them in microseconds or seconds, which made every `/ 86400e9` day count
    1,000x or more too small (a weekly panel lost every innovation again).
    """
    return index.as_unit("ns").asi8


def gap_limit_days(index: pd.DatetimeIndex) -> float:
    """Longest step, in days, that is not a hole on a non-monthly panel: max(5, 1.5 x median spacing).

    Daily panels keep the 5-day rule. On a weekly panel (jobless claims) every step was a hole under it,
    so the engine had no innovations and fell back to a scale of 0.1% of the level. M0's own hole rule
    is max(10 x median, 5) (§3.3).
    """
    if len(index) < 3:
        return 5.0
    return max(5.0, 1.5 * float(np.median(np.diff(epoch_ns(index)))) / 86400e9)


def innovations(state: pd.Series, kind: str, cadence: str) -> pd.Series:
    if kind == "log_return":
        return state.copy()
    diff = state.diff()
    if len(state) < 2:
        return diff
    if cadence == "monthly":
        gap = np.r_[False, np.diff(state.index.to_period("M").asi8) == 1]
    else:
        delta = state.index.to_series().diff().dt.total_seconds().div(86400)
        gap = (delta <= gap_limit_days(state.index)).to_numpy()
    return diff.where(gap)


def scale_estimate(values: np.ndarray, cadence: str, blend: float = 0.8) -> tuple[float, float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return 0.0, 0.0
    recent_n = 24 if cadence == "monthly" else 60
    recent = x[-recent_n:]
    lam = 0.90 if cadence == "monthly" else 0.97
    weights = lam ** np.arange(len(recent) - 1, -1, -1)
    weights /= weights.sum()
    center = float(np.sum(weights * recent))
    recent_var = float(np.sum(weights * (recent - center) ** 2)) / max(0.5, 1 - np.sum(weights**2))
    # A finite long-run window avoids treating every distant episode as the current regime.
    long = x[-(120 if cadence == "monthly" else 756) :]
    long_var = float(np.var(long, ddof=1))
    current = np.sqrt(blend * recent_var + (1.0 - blend) * long_var)
    return float(current), float(np.sqrt(long_var))


def regularized_correlation(frame: pd.DataFrame) -> np.ndarray:
    d = frame.shape[1]
    if d == 1:
        return np.ones((1, 1))
    # Correlation and its reliability have to describe the same rows.
    window = frame.tail(756)
    corr = window.corr(min_periods=20).to_numpy()
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 1.0)
    valid = window.notna().astype(float).to_numpy()
    overlaps = valid.T @ valid
    shrink = np.clip(20.0 / np.maximum(overlaps, 20.0), 0.05, 1.0)
    corr *= 1 - shrink
    np.fill_diagonal(corr, 1.0)
    eig, vectors = np.linalg.eigh((corr + corr.T) / 2)
    corr = (vectors * np.maximum(eig, 1e-6)) @ vectors.T
    diag = np.sqrt(np.diag(corr))
    return corr / np.outer(diag, diag)


def standardized_residuals(frame: pd.DataFrame, cadence: str) -> pd.DataFrame:
    # No filling holes: the sampler must restart a block at a missing observation.
    window = 24 if cadence == "monthly" else 60
    scale = frame.ewm(span=window, min_periods=8, adjust=False).std().shift(1)
    fallback = frame.std().replace(0, np.nan)
    scale = scale.fillna(fallback).clip(lower=fallback * 0.15, axis=1)
    center = frame.ewm(span=window, min_periods=8, adjust=False).mean().shift(1)
    return (frame - center.fillna(frame.mean())) / scale


@dataclass(frozen=True)
class ExpertParameters:
    drift: np.ndarray
    reversion: np.ndarray
    equilibrium: np.ndarray
    decay: np.ndarray


def parameters(state: np.ndarray, inc: np.ndarray, kind: str, cadence: str) -> ExpertParameters:
    x = inc[np.isfinite(inc)]
    n = 12 if cadence == "monthly" else 60
    x = x[-n:]
    spot = 0.0 if kind == "log_return" else float(state[-1])
    sd = float(np.std(x, ddof=1)) if len(x) > 1 else 0.0
    mean = float(np.mean(x)) if len(x) else 0.0
    strength = 0.8 if cadence == "monthly" else 0.2 if kind == "log_return" else 0.35
    trend = float(np.clip(strength * mean, -sd * 0.35, sd * 0.35))
    drift = np.array([0.0, trend, mean * 0.15 if kind == "log_return" else 0.0])
    phi = 0.0
    equilibrium = spot
    if kind != "log_return" and len(state) > 24:
        # Gap-safe regression: missing innovations have no role in the estimate.
        levels = np.asarray(state[-(n + 1) :], dtype=float)
        delta = np.asarray(inc[-(n + 1) :], dtype=float)[1:]
        lag = levels[:-1]
        good = np.isfinite(delta)
        equilibrium = float(np.median(lag))
        centered = lag - equilibrium
        denom = float(np.sum(centered[good] ** 2)) + len(lag) * sd**2
        slope = float(np.sum(centered[good] * delta[good]) / denom) if denom > 0 else 0.0
        phi = float(np.clip(-slope, 0, 0.12 if cadence == "monthly" else 0.04))
    return ExpertParameters(
        drift,
        np.array([0.0, 0.0, phi]),
        np.array([spot, spot, equilibrium]),
        np.array([1.0, 24.0 if cadence == "monthly" else 126.0, 1.0]),
    )


def expected_change(params: ExpertParameters, spot: float, horizon: int) -> np.ndarray:
    k = np.arange(horizon, dtype=float)
    total = np.sum(params.drift[:, None] * np.exp(-k[None, :] / params.decay[:, None]), axis=1)
    total += (params.equilibrium - spot) * (1 - (1 - params.reversion) ** horizon)
    return total


def horizon_shape(
    increments: pd.Series, horizons: np.ndarray, long_sd: float, cadence: str = "daily"
) -> tuple[np.ndarray, list]:
    ratios, audit = [], []
    history = increments.iloc[-2520:].copy()
    if len(history) > 1:
        gaps = (
            np.diff(history.index.to_period("M").asi8) != 1
            if cadence == "monthly"
            else np.diff(epoch_ns(history.index)) / 86400e9 > gap_limit_days(history.index)
        )
        history.iloc[np.flatnonzero(gaps) + 1] = np.nan
    for horizon in horizons:
        h = int(horizon)
        # A NaN at a gap invalidates every window crossing that gap.
        sums = history.rolling(h, min_periods=h).sum().dropna()
        effective = len(sums) // h
        if effective < 4 or long_sd <= 1e-12:
            ratio, trust = 1.0, 0.0
        else:
            ratio = float(np.clip(sums.std(ddof=1) / (long_sd * np.sqrt(h)), 0.65, 1.8))
            trust = effective / (effective + 20.0)
        ratios.append(np.log(ratio) * trust)
        audit.append(
            {
                "steps": h,
                "overlapping_windows": len(sums),
                "effective_blocks": effective,
                "trust": trust,
                "raw_ratio": ratio,
            }
        )
    return np.asarray(ratios), audit


@dataclass(frozen=True)
class Parameters:
    trend: float = 1.0
    reversion: float = 1.0
    empirical: float | None = None
    scale: float = 1.0

    def __post_init__(self):
        for name, value in asdict(self).items():
            if name == "empirical" and value is None:
                continue
            require_finite_number(name, value)
        if self.empirical is not None and not 0 <= self.empirical <= 0.6:
            raise ValueError("invalid explicit empirical exponent")
        if not 0.2 <= self.scale <= 3.0:
            raise ValueError("invalid explicit width scale")


@dataclass
class NumericModel:
    transforms: list[Transform]
    spots: np.ndarray
    scales: np.ndarray
    long_scales: np.ndarray
    correlation: np.ndarray
    residuals: np.ndarray
    residual_next: np.ndarray
    transfer_uncertainty: np.ndarray
    trend: np.ndarray
    reversion: np.ndarray
    base_spread: np.ndarray
    log_width_ratio: np.ndarray
    width_exponent: np.ndarray
    center: np.ndarray
    spread: np.ndarray
    audit: dict = field(default_factory=dict)
    # None means every residual column is used. A index vector keeps bootstrap
    # shocks on those assets only; the others keep the Student-t draw.
    bootstrap_columns: np.ndarray | None = None


def distribution_parameters(
    model: NumericModel,
    task: TaskContext,
    p: Parameters,
    guard: bool = True,
    cap: float = 0.75,
) -> tuple[np.ndarray, np.ndarray]:
    change = model.trend * p.trend + model.reversion * p.reversion
    if guard and task.cadence == "daily":
        change = np.clip(change, -cap * model.base_spread, cap * model.base_spread)
    exponent = model.width_exponent[:, None] if p.empirical is None else p.empirical
    spread = model.base_spread * np.exp(exponent * model.log_width_ratio) * p.scale
    return model.spots[:, None] + change, spread


def set_parameters(
    model: NumericModel,
    task: TaskContext,
    p: Parameters,
    guard: bool,
    cap: float = 0.75,
) -> None:
    model.center, model.spread = distribution_parameters(model, task, p, guard, cap)
    model.audit["center_cap"] = cap
    model.audit["parameters"] = asdict(p)
    model.audit["effective_width_exponents"] = (
        model.width_exponent.tolist() if p.empirical is None else [p.empirical] * len(task.assets)
    )
    model.audit["center_guard"] = guard
    model.audit["width_application"] = "one final affine scale of the shared innovation paths"


def fit_base(task: TaskContext, settings: Settings, *, fallback: bool = False) -> NumericModel:
    if np.any(task.steps <= 0) or task.steps.max() > 5000:
        raise ContractError("invalid or unbounded forecast duration")
    cutoff = pd.Timestamp(task.asof)
    if any(s.empty or s.index.max() > cutoff for s in task.histories.values()):
        raise ContractError("numerical inputs must be nonempty cutoff-filtered prefixes")
    transforms = [transform_for(task, a) for a in task.assets]
    degraded: list[str] = []
    for i, asset in enumerate(task.assets):
        series = task.histories[asset]
        if transforms[i].kind == "log_level" and (series.dropna() <= 0).any():
            transforms[i] = Transform("level", asset, "nonpositive quote degraded to level")
            degraded.append(asset)
    states = {a: t.state(task.histories[a]) for a, t in zip(task.assets, transforms, strict=True)}
    changes = {
        a: innovations(states[a], t.kind, task.cadence)
        for a, t in zip(task.assets, transforms, strict=True)
    }
    scales, longs, spots, uncertainty, fitted = [], [], [], [], []
    assets = {}
    for a, trans in zip(task.assets, transforms, strict=True):
        state, inc = states[a], changes[a]
        sd, long_sd = scale_estimate(inc.to_numpy(), task.cadence, settings.vol_blend)
        spot = 0.0 if trans.kind == "log_return" else float(state.iloc[-1])
        age = (cutoff - state.index[-1]).days
        if task.cadence == "monthly":
            periods = state.index.to_period("M")
            # Date arithmetic avoids the Period/Timedelta deprecation on the pinned pandas.
            expected = pd.Period(
                cutoff.date() - timedelta(days=settings.monthly_publication_lag_days),
                freq="M",
            )
            stale = (len(state) < 30 or periods[-1] < expected
                     or (len(periods) > 1 and periods[-1].ordinal - periods[-2].ordinal > 1))
            freshness = {"rule": "monthly periods with staging-lag allowance",
                         "lag_allowance_days": settings.monthly_publication_lag_days,
                         "expected_latest_period": str(expected), "observed_period": str(periods[-1])}
        else:
            stale = (len(state) < 30 or age > 20
                     or (len(state) > 1 and (state.index[-1] - state.index[-2]).days > 45))
            freshness = {"rule": "daily age and consecutive-observation gap", "age_days": age}
        proxies = []
        if stale and trans.kind == "log_level" and not fallback:
            for name, hist in task.context_histories.items():
                if name in task.assets or len(hist) < 60 or (cutoff - hist.index[-1]).days > 5:
                    continue
                proxy = transform_for(task, name)
                if proxy.kind == trans.kind:
                    vals = innovations(proxy.state(hist), proxy.kind, task.cadence)
                    current, long = scale_estimate(vals.to_numpy(), task.cadence, settings.transfer_vol_blend)
                    if current > 0:
                        proxies.append((name, current, long))
            if proxies:
                ps, pl = np.median([p[1:] for p in proxies], axis=0)
                floor = float(settings.transfer_floor)
                if settings.transfer_allow_narrow:
                    sd = max(ps * floor, min(sd, ps * 2.0))
                    long_sd = max(pl * floor, min(long_sd, pl * 2.0))
                else:
                    # Floor only. A later beta fit may widen this further, not tighten it.
                    sd = max(sd, ps * floor)
                    long_sd = max(long_sd, pl * floor)
        if sd <= 1e-12:
            sd = 0.005 if trans.kind != "level" else max(abs(spot) * 0.001, 0.01)
            long_sd = sd
        if fallback:
            sd *= 1.25
            long_sd *= 1.25
        scales.append(sd)
        longs.append(max(long_sd, sd * 0.3))
        spots.append(spot)
        uncertainty.append(0.25 if stale else 0.0)
        fitted.append(
            None
            if stale or fallback
            else parameters(state.to_numpy(), inc.to_numpy(), trans.kind, task.cadence)
        )
        assets[a] = {
            "transform": trans.kind,
            "quote": trans.quote,
            "anchor": float(task.histories[a].iloc[-1]),
            "forecast_state_anchor": spot,
            "forecast_steps": task.steps[len(spots) - 1].tolist(),
            "history_rows": len(state),
            "last_observation": str(state.index[-1].date()),
            "valid_innovations": int(inc.notna().sum()),
            "transfer": bool(stale),
            "freshness": freshness,
            "current_step_sd": float(sd),
            "long_step_sd": float(longs[-1]),
            "scale_proxy": [p[0] for p in proxies],
        }
    frame = pd.DataFrame(changes)
    corr = np.eye(len(task.assets)) if fallback else regularized_correlation(frame)
    residuals, next_rows = np.empty((0, len(task.assets))), np.empty(0, dtype=int)
    if not fallback:
        standardized = standardized_residuals(frame, task.cadence)
        complete = standardized.notna().all(axis=1)
        values = standardized.loc[complete].to_numpy()
        if len(values) >= 20:
            residuals = np.clip(values, -12, 12)
            residuals -= residuals.mean(axis=0)
            residuals /= np.maximum(residuals.std(axis=0, ddof=1), 1e-6)
            positions = np.flatnonzero(complete.to_numpy())
            dates = standardized.index[complete]
            adjacent = np.diff(positions) == 1
            adjacent &= (
                np.diff(dates.to_period("M").asi8) == 1
                if task.cadence == "monthly"
                else np.diff(epoch_ns(dates)) / 86400e9 <= gap_limit_days(dates)
            )
            next_rows = np.r_[np.where(adjacent, np.arange(1, len(values)), -1), -1]
    shape = task.steps.shape
    model = NumericModel(
        transforms,
        np.asarray(spots),
        np.asarray(scales),
        np.asarray(longs),
        corr,
        residuals,
        next_rows,
        np.asarray(uncertainty),
        *[np.zeros(shape) for _ in range(4)],
        np.array([settings.width_for(task.panel_for_asset.get(a, "")) for a in task.assets]),
        *[np.zeros(shape) for _ in range(2)],
        audit={
            "engine": "unified-v3",
            "assets": assets,
            "bootstrap_rows": len(residuals),
            "fallback": fallback,
            "degraded_transforms": degraded,
        },
    )
    if not fallback:
        fit_stable_transfer(task, model, settings)
    _retain_fresh_residuals(task, model, frame, assets)
    prior = (
        np.array([0.40, 0.50, 0.10]) if task.cadence == "monthly" else np.array([0.65, 0.20, 0.15])
    )
    decay = 12.0 if task.cadence == "monthly" else settings.vol_decay
    for ai, a in enumerate(task.assets):
        for hi, h in enumerate(task.steps[ai]):
            variance = model.long_scales[ai] ** 2 + (
                model.scales[ai] ** 2 - model.long_scales[ai] ** 2
            ) * np.exp(-np.arange(h) / decay)
            model.base_spread[ai, hi] = np.sqrt(variance.sum())
            if settings.vol_beta_base > 0 and task.cadence == "daily":
                # Marginal width from the measured extrapolation exponent. The path profile
                # above still uses the decay, which is what sets the cross-horizon
                # correlation; the per-cell standardization in `standardized_paths` means
                # that profile never reaches the marginal width, so the two do not fight.
                long_sd = max(float(model.long_scales[ai]), 1e-12)
                exponent = float(
                    np.clip(
                        settings.vol_beta_base
                        - settings.vol_beta_slope * np.log2(max(int(h), 1) / 21.0),
                        0.0,
                        1.0,
                    )
                )
                ratio = max(float(model.scales[ai]) / long_sd, 1e-6)
                model.base_spread[ai, hi] = long_sd * np.sqrt(float(h)) * ratio**exponent
            p = fitted[ai]
            if p is not None:
                contributions = expected_change(p, model.spots[ai], int(h))
                model.trend[ai, hi] = prior[1] * contributions[1]
                if transforms[ai].kind == "log_return":
                    model.trend[ai, hi] += prior[2] * contributions[2]
                else:
                    model.reversion[ai, hi] = prior[2] * contributions[2]
                if (
                    settings.suppress_countertrend_reversion
                    and model.trend[ai, hi] * model.reversion[ai, hi] < 0.0
                ):
                    model.reversion[ai, hi] = 0.0
                if task.cadence == "monthly":
                    # Preserve the submission's distinction between a persistent level
                    # slope and innovation noise. A CPI/payroll trend can be strong even
                    # when first-difference volatility is nearly zero; clipping it to
                    # that volatility destroys the level signal.
                    recent = changes[a].dropna().iloc[-12:].to_numpy()
                    slope = float(np.median(recent)) if len(recent) else 0.0
                    decel = 0.0
                    if settings.monthly_deceleration and len(recent) >= 8:
                        half = len(recent) // 2
                        earlier = float(np.median(recent[:half]))
                        later = float(np.median(recent[half:]))
                        if earlier * later > 0.0 and abs(later) < abs(earlier):
                            decel = abs(earlier - later)
                            slope = later
                    duration = (1 - 0.98 ** int(h)) / 0.02
                    signal = abs(slope) / max(model.long_scales[ai], 1e-12)
                    reliability = (
                        (
                            len(recent)
                            * slope**2
                            / max(len(recent) * slope**2 + model.long_scales[ai] ** 2, 1e-18)
                        )
                        if signal >= 0.25
                        else 0.0
                    )
                    model.trend[ai, hi] = reliability * slope * duration
                    model.reversion[ai, hi] = 0.0
                    model.base_spread[ai, hi] = np.sqrt(
                        variance.sum()
                        + model.scales[ai] ** 2 * duration**2 / max(6, len(recent))
                        + decel**2 * duration**2
                    )
                    assets[a]["monthly_slope"] = {
                        "value": slope,
                        "reliability": reliability,
                        "slope_to_sd": signal,
                        "deceleration": decel,
                        "guard": "weak slope shrinks to zero; a slowing slope is not extrapolated at the older pace",
                    }
        if not assets[a]["transfer"] and not fallback:
            model.log_width_ratio[ai], assets[a]["horizon_width"] = horizon_shape(
                changes[a], task.steps[ai], model.long_scales[ai], task.cadence
            )
    if settings.use_trailing_drift and task.cadence == "daily":
        for ai, a in enumerate(task.assets):
            mu = 0.0 if assets[a]["transfer"] else trailing_step_mean(changes[a])
            assets[a]["unclipped_trailing_step_mean"] = float(mu)
            if settings.center_guard and settings.drift_cap_horizon:
                reference = float(settings.drift_cap_horizon)
                exponent = np.clip(
                    settings.vol_beta_base - settings.vol_beta_slope * np.log2(reference / 21), 0, 1
                )
                limit = (settings.center_cap * model.long_scales[ai]
                         * (model.scales[ai] / model.long_scales[ai])**exponent / np.sqrt(reference))
                mu = float(np.clip(mu, -limit, limit))
                assets[a]["daily_drift_limit"] = float(limit)
            for hi, h in enumerate(task.steps[ai]):
                model.trend[ai, hi] = mu * float(h)
                model.reversion[ai, hi] = 0.0
            assets[a]["trailing_step_mean"] = float(mu)
    set_parameters(
        model,
        task,
        Parameters(scale=settings.width_scale),
        settings.center_guard and not (task.cadence == "daily" and settings.use_trailing_drift
                                       and settings.drift_cap_horizon > 0),
        settings.center_cap,
    )
    if task.cadence == "daily":
        relative = task.steps / settings.horizon_scale_anchor
        if settings.horizon_scale_widen_only:
            relative = np.maximum(relative, 1.0)
        model.spread *= relative**settings.horizon_scale_power
        model.audit["daily_width"] = {"scale": settings.width_scale, "anchor_steps": settings.horizon_scale_anchor,
                                       "horizon_power": settings.horizon_scale_power,
                                       "widen_only": settings.horizon_scale_widen_only}
        model.audit["drift_cap_horizon"] = settings.drift_cap_horizon
    return model


def _retain_fresh_residuals(
    task: TaskContext,
    model: NumericModel,
    frame: pd.DataFrame,
    assets: dict,
) -> None:
    """Drop only stale targets from the bootstrap library.

    A card whose every target is a transfer still has no current residual
    regime, so the library is emptied. A mixed card keeps the fresh assets'
    empirical shocks and lets the stale names draw Student-t instead.
    All-fresh cards do not enter this function's mutation path.
    """
    flags = [bool(assets[a]["transfer"]) for a in task.assets]
    if not any(flags):
        model.audit["bootstrap_scope"] = "all fresh targets"
        return
    if all(flags) or frame.empty:
        model.residuals = np.empty((0, len(task.assets)))
        model.residual_next = np.empty(0, dtype=int)
        model.bootstrap_columns = None
        model.audit["bootstrap_rows"] = 0
        model.audit["bootstrap_scope"] = "wiped; no fresh target"
        return
    fresh = [i for i, flag in enumerate(flags) if not flag]
    standardized = standardized_residuals(frame, task.cadence)
    names = [task.assets[i] for i in fresh]
    if any(name not in standardized.columns for name in names):
        model.residuals = np.empty((0, len(task.assets)))
        model.residual_next = np.empty(0, dtype=int)
        model.audit["bootstrap_rows"] = 0
        model.audit["bootstrap_scope"] = "fresh residual columns unavailable"
        return
    complete = standardized[names].notna().all(axis=1)
    block = standardized.loc[complete]
    if len(block) < 20:
        model.residuals = np.empty((0, len(task.assets)))
        model.residual_next = np.empty(0, dtype=int)
        model.audit["bootstrap_rows"] = 0
        model.audit["bootstrap_scope"] = "too few fresh complete rows"
        return
    values = np.nan_to_num(block.reindex(columns=list(task.assets)).to_numpy(dtype=float), nan=0.0)
    values = np.clip(values, -12.0, 12.0)
    values -= values.mean(axis=0)
    scale = values.std(axis=0, ddof=1)
    for i, flag in enumerate(flags):
        if flag:
            values[:, i] = 0.0
            scale[i] = 1.0
    values /= np.maximum(scale, 1e-6)
    positions = np.flatnonzero(complete.to_numpy())
    dates = standardized.index[complete]
    adjacent = np.diff(positions) == 1
    adjacent &= (
        np.diff(dates.to_period("M").asi8) == 1
        if task.cadence == "monthly"
        else np.diff(epoch_ns(dates)) / 86400e9 <= gap_limit_days(dates)
    )
    model.residuals = values
    model.residual_next = np.r_[np.where(adjacent, np.arange(1, len(values)), -1), -1]
    model.bootstrap_columns = np.asarray(fresh, dtype=int)
    model.audit["bootstrap_rows"] = int(len(values))
    model.audit["bootstrap_scope"] = "fresh targets only"


def fit_stable_transfer(task: TaskContext, model: NumericModel, settings: Settings) -> None:
    """Shrink proxy scale toward the legacy anchor as old exposures lose support.

    Proxy selection uses the first historical block. A later disjoint block checks
    coefficient stability. No monotone volatility inflation is inferred from gap age.
    This is an explicit conservative heuristic, not identifiable current-regime beta.
    """
    asof = pd.Timestamp(task.asof)
    proxies = {}
    for asset, hist in task.context_histories.items():
        hist = hist.loc[hist.index <= asof]
        trans = transform_for(task, asset)
        if asset in task.assets or trans.kind != "log_level" or len(hist) < 100:
            continue
        if (asof - hist.index[-1]).days <= 5:
            proxies[asset] = innovations(trans.state(hist), trans.kind, task.cadence)
    for ai, asset in enumerate(task.assets):
        if not model.audit["assets"][asset]["transfer"] or model.transforms[ai].kind != "log_level":
            continue
        hist = task.histories[asset].loc[task.histories[asset].index <= asof]
        target = (
            innovations(model.transforms[ai].state(hist), "log_level", task.cadence)
            .dropna()
            .tail(1260)
        )
        if len(target) < 160:
            continue
        split_date = target.index[int(len(target) * 0.6)]
        choices = []
        for name, proxy in proxies.items():
            pair = pd.concat([target.rename("target"), proxy.rename("proxy")], axis=1).dropna()
            first = pair.loc[pair.index < split_date]
            if len(first) >= 80 and min(first.std()) > 1e-10:
                choices.append((abs(float(first.corr().iloc[0, 1])), name))
        names = [name for _, name in sorted(choices, reverse=True)[:3]]
        if not names:
            continue
        frame = pd.concat(
            [target.rename("target"), *[proxies[n].rename(n) for n in names]], axis=1
        ).dropna()
        first, second = frame.loc[frame.index < split_date], frame.loc[frame.index >= split_date]
        if min(len(first), len(second)) < 60:
            continue

        def fit(block, selected=tuple(names)):
            x, y = block[list(selected)].to_numpy(), block["target"].to_numpy()
            xs = np.maximum(np.std(x, axis=0), 1e-10)
            z, centered = (x - x.mean(axis=0)) / xs, y - y.mean()
            beta = (
                np.linalg.solve(z.T @ z + np.eye(len(selected)) * len(z) * 0.2, z.T @ centered) / xs
            )
            residual = centered - (x - x.mean(axis=0)) @ beta
            return beta, max(float(np.var(residual, ddof=1)), 0.04 * float(np.var(y, ddof=1)))

        before, _ = fit(first)
        after, _ = fit(second)
        beta, residual_variance = fit(frame)
        norm = np.linalg.norm(before) * np.linalg.norm(after)
        alignment = max(0.0, float(np.dot(before, after) / max(norm, 1e-12)))
        drift = np.linalg.norm(before - after) / max(
            np.linalg.norm(before) + np.linalg.norm(after), 1e-12
        )
        stability = alignment * (1 - min(1.0, drift))
        gap_years = max(0.0, (asof - target.index[-1]).days / 365.25)
        trust = min(0.6, stability * len(frame) / (len(frame) + 252) * np.exp(-gap_years / 5))
        now = pd.DataFrame({n: proxies[n] for n in names}).tail(756).dropna()
        if len(now) < 80 or (asof - now.index[-1]).days > 5:
            continue
        current, long = scale_estimate(now.to_numpy() @ beta, "daily")
        estimates = np.sqrt(np.array([current**2, long**2]) + residual_variance)
        original = np.array([model.scales[ai], model.long_scales[ai]])
        if settings.transfer_allow_narrow:
            estimates = np.clip(estimates, original * 0.67, original * 1.5)
        else:
            estimates = np.minimum(np.maximum(estimates, original), original * 1.5)
        blended = np.exp((1 - trust) * np.log(original) + trust * np.log(estimates))
        if not settings.transfer_allow_narrow:
            blended = np.maximum(blended, original)
        model.scales[ai], model.long_scales[ai] = blended
        model.audit["assets"][asset].update(
            {
                "current_step_sd": float(blended[0]),
                "long_step_sd": float(blended[1]),
                "transfer_fit": {
                    "method": "historically stable proxy scale shrunk toward legacy",
                    "proxy_names": names,
                    "betas": dict(zip(names, beta.tolist(), strict=True)),
                    "selection_end": str(first.index[-1].date()),
                    "stability_start": str(second.index[0].date()),
                    "training_end": str(frame.index[-1].date()),
                    "training_rows": len(frame),
                    "stability": stability,
                    "proxy_weight": float(trust),
                    "missing_years": gap_years,
                    "legacy_scales": original.tolist(),
                    "candidate_scales": estimates.tolist(),
                    "regime_change_identifiable": False,
                },
            }
        )


def derive_seed(seed: int, *parts: str) -> int:
    data = "\0".join([str(seed), *parts]).encode()
    return int.from_bytes(hashlib.sha256(data).digest()[:8], "little")


def generator(seed: int, *parts: str) -> np.random.Generator:
    return np.random.default_rng(derive_seed(seed, *parts))


class InnovationStream:
    def __init__(
        self,
        model: NumericModel,
        rng: np.random.Generator,
        draws: int,
        block: int,
        weight: float,
        df: float,
    ):
        self.model, self.rng, self.draws, self.block, self.df = model, rng, draws, block, df
        self.chol = np.linalg.cholesky(model.correlation)
        self.bootstrap = rng.random(draws) < (weight if len(model.residuals) >= 20 else 0)
        self.positions = rng.integers(max(1, len(model.residuals)), size=draws)
        self.first = True

    def draw(self) -> np.ndarray:
        n, d = self.draws, len(self.model.spots)
        normal = self.rng.standard_normal((n, d)) @ self.chol.T
        z = normal * np.sqrt((self.df - 2) / self.rng.chisquare(self.df, n))[:, None]
        if self.bootstrap.any():
            if not self.first:
                nxt = self.model.residual_next[self.positions]
                restart = (self.rng.random(n) < 1.0 / self.block) | (nxt < 0)
                self.positions = np.where(
                    restart, self.rng.integers(len(self.model.residuals), size=n), nxt
                )
            drawn = z.copy()
            z[self.bootstrap] = self.model.residuals[self.positions[self.bootstrap]]
            columns = self.model.bootstrap_columns
            if columns is not None:
                keep = np.ones(d, dtype=bool)
                keep[np.asarray(columns, dtype=int)] = False
                if keep.any():
                    z[np.ix_(self.bootstrap, np.flatnonzero(keep))] = drawn[
                        np.ix_(self.bootstrap, np.flatnonzero(keep))
                    ]
        self.first = False
        return z


def standardized_paths(task: TaskContext, model: NumericModel, settings: Settings) -> np.ndarray:
    n, a, h = settings.draws, len(task.assets), len(task.horizons)
    if n * task.cell_count > 5_000_000:
        raise ContractError("requested sample grid exceeds the row budget")
    rng = generator(settings.seed, task.fingerprint, "unified-paths")
    stream = InnovationStream(
        model, rng, n, settings.block_length, settings.bootstrap_weight, settings.student_df
    )
    beta = np.exp(
        rng.standard_normal((n, a)) * model.transfer_uncertainty - model.transfer_uncertainty**2
    )
    x, out = np.zeros((n, a)), np.empty((n, a, h))
    endpoints = task.anchors[:, None] + task.steps
    decay = 12.0 if task.cadence == "monthly" else settings.vol_decay
    # One shared log-volatility state per draw: a calm path stays calm for months and a
    # stressed one stays stressed, which is what leaves a fat tail in the accumulated
    # change. Sharing it across assets also makes the extremes co-move, which is the
    # dependence the variogram term scores on a joint card.
    panels = [task.panel_for_asset.get(a, "") for a in task.assets]
    s_v = np.asarray([settings.overlay_for(p) for p in panels])
    phi_v = settings.vol_persist
    # Independent stream: turning the overlay on cannot replace the base shocks.
    vol_rng = generator(settings.seed, task.fingerprint, "volatility-overlay")
    log_vol = vol_rng.standard_normal(n) if np.any(s_v > 0) else None
    step_sd = np.sqrt(max(1.0 - phi_v**2, 0.0))
    for time in range(int(task.anchors.min()) + 1, int(endpoints.max()) + 1):
        age = time - task.anchors
        variance = model.long_scales**2 + (model.scales**2 - model.long_scales**2) * np.exp(
            -np.maximum(age - 1, 0) / decay
        )
        if log_vol is None:
            vol = 1.0
        else:
            log_vol = phi_v * log_vol + step_sd * vol_rng.standard_normal(n)
            vol = np.exp(log_vol[:, None] * s_v[None, :] - 0.5 * s_v[None, :]**2)
        x += stream.draw() * vol * np.sqrt(variance) * beta * (age > 0)
        for ai, hi in np.argwhere(endpoints == time):
            out[:, ai, hi] = x[:, ai]
    # Positive per-cell affine transforms preserve the sampled cross-asset/horizon ranks.
    out -= out.mean(axis=0)
    sd = out.std(axis=0, ddof=1)
    if not np.isfinite(sd).all() or np.any(sd <= 1e-12):
        raise ContractError("degenerate joint innovations")
    return out / sd


def simulate(task: TaskContext, model: NumericModel, settings: Settings) -> np.ndarray:
    if settings.draws < task.min_draws:
        raise ContractError("draw count is below the card minimum")
    values = model.center + standardized_paths(task, model, settings) * model.spread
    for ai, trans in enumerate(model.transforms):
        values[:, ai, :] = trans.invert(values[:, ai, :])
    if not np.isfinite(values).all():
        raise ContractError("non-finite forecast")
    return values


def _shell_model(task: TaskContext, reason: str) -> NumericModel:
    n_a, n_h = len(task.assets), len(task.horizons)
    shape = (n_a, n_h)
    return NumericModel(
        [Transform("level", a, "wide fallback") for a in task.assets],
        np.zeros(n_a),
        np.ones(n_a),
        np.ones(n_a),
        np.eye(n_a),
        np.empty((0, n_a)),
        np.empty(0, dtype=int),
        np.zeros(n_a),
        np.zeros(shape),
        np.zeros(shape),
        np.ones(shape),
        np.zeros(shape),
        np.ones(n_a),
        np.zeros(shape),
        np.ones(shape),
        audit={"engine": "wide-fallback", "fallback": True, "fallback_reason": reason, "assets": {}},
    )


def wide_samples(task: TaskContext, settings: Settings) -> np.ndarray:
    """Finite grid-shaped draws used only when the fitted engine cannot run."""
    n = min(20_000, max(int(settings.draws), int(task.min_draws), 200))
    rng = generator(settings.seed, task.fingerprint or "emergency", "wide-fallback")
    out = np.empty((n, len(task.assets), len(task.horizons)), dtype=float)
    steps = np.asarray(task.steps)
    if steps.ndim == 1:
        steps = np.broadcast_to(steps, (len(task.assets), len(task.horizons)))
    for i, asset in enumerate(task.assets):
        hist = task.histories.get(asset)
        last = 0.0
        if task.target_type != "log_return" and hist is not None and len(hist):
            raw = float(hist.iloc[-1])
            last = raw if np.isfinite(raw) else 0.0
        if task.target_type != "log_return" and last == 0.0 and asset in FX_NAMES:
            last = 1.0
        width = max(abs(last) * 0.05, 1.0)
        for hi, h in enumerate(steps[i]):
            out[:, i, hi] = last + rng.standard_t(5.0, size=n) * width * np.sqrt(max(int(h), 1)) * 1.5
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _stamp(model: NumericModel, task: TaskContext, settings: Settings, requested: dict) -> None:
    model.audit.update(version=ENGINE_VERSION, requested_settings=requested, settings=asdict(settings))
    model.audit["volatility_overlay"] = {
        "persistence": settings.vol_persist,
        "independent_rng": True,
        "per_asset_sd": {a: settings.overlay_for(task.panel_for_asset.get(a, "")) for a in task.assets},
    }
    model.audit["width_exponent"] = dict(zip(task.assets, model.width_exponent.tolist(), strict=True))
    model.audit.setdefault("fallback", False)


# ---------------------------------------------------------------- published M0
# A line-for-line port of backtest/m0.py, i.e. docs/M0-BASELINE.md §3. It reads only the
# card and the supplied panels. Its draws must equal the reference generator's, because
# pair matching targets M0's own 500-draw variogram value.
M0_DRAWS = 500
M0_MONTHLY_STEPS = {uid: dict(zip(hz, steps, strict=True)) for uid, hz, steps in (
    ("t2-F1-cpi-glidepath-2023", (140, 160), (8, 9)),
    ("t2-F1-sahm-watch-2024", (145, 165), (8, 9)),
    ("t2-F4-covid-nfp-2020", (21,), (2,)),
    ("t2-F4-cpi-vintage-2022", (21,), (2,)),
)}


def _m0_timestamp(value: Any) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        raise ValueError("M0: missing date")
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert("UTC").tz_localize(None)
    return stamp.as_unit("ns")


_US_BUSINESS_DAY = pd.offsets.CustomBusinessDay(calendar=USFederalHolidayCalendar())


def _spacing(dates: pd.DatetimeIndex) -> float | None:
    """Mean interval in days over steps that are not holes (§3.3, §3.7.2); None below three dates."""
    if len(dates) < 3:
        return None
    intervals = np.asarray((dates[1:] - dates[:-1]).total_seconds()) / 86400.0
    good = intervals[intervals <= max(10.0 * float(np.median(intervals)), 5.0)]
    return float(np.mean(good)) if len(good) else None


def _m0_step_count(h: int, last: pd.Timestamp, spacing: float | None, asof: pd.Timestamp) -> int:
    """M0 §3.7 without the sealed target date, which is estimated as the as-of plus h US business days.

    §3.7 keeps the declared horizon on every daily panel, so a mean spacing of 5 days or less keeps
    it here. A weekly-type panel (more than 5 and at most 20 days) counts prints to the estimated
    date (§3.7.3): a next-print target keeps a key of 1 and turns a key of 4 into 1 print, as M0
    does with the sealed date. On a monthly panel a key below 21 is already in panel steps (the
    restatement M0 §8 announced); a larger key counts months from the last observation. A count
    replaces the key only when the two are more than 2x apart (§3.7.5; strict since the official
    commit 24582e9 of 17 October 2026, which keeps the key at a ratio of exactly 2), and never when it
    is 0 or less (§3.7.4).
    """
    if spacing is None or spacing <= 5 or (spacing > 20 and h < 21):
        return int(h)
    target = pd.Timestamp(asof) + int(h) * _US_BUSINESS_DAY
    if spacing <= 20:
        count = int(round((target - last).days / spacing))
    else:
        count = 12 * (target.year - last.year) + target.month - last.month
    if count <= 0:
        return int(h)
    return int(count if max(count, h) / max(min(count, h), 1) > 2 else h)


def _next_figure_card(horizons: tuple[int, ...], monthly: bool) -> bool:
    """Whether a monthly card with no observation month forecasts its series' next figure (one step).

    The ranked Final's data-release cards each forecast "the next figure of a monthly official
    statistic" (README), released 1-4 business days after the as-of. Read as panel steps, a key of 4
    would walk four months; M0 counts about one month to the sealed date, and its 2x rule (§3.7.5)
    takes that count for every key above 2 (strict since 17 October 2026: at a key of exactly 2 the
    official M0 keeps two months, while this card still walks its one next figure). One key below 21
    on a monthly panel is read so. A grid of several keys has no single next figure and keeps the
    panel-step reading; quarterly and other sparse panels keep §3.7 as before.
    """
    return monthly and len(horizons) == 1 and int(horizons[0]) < 21


def _m0_history_steps(frame: pd.DataFrame, asset: str, asof: pd.Timestamp,
                      target_type: str) -> tuple[pd.Series, float, pd.DatetimeIndex]:
    """Last 300 observations; gap-aware steps; anchor per target type (§3.1-3.3, §3.6)."""
    key = "asset" if "asset" in frame else "asset_id"
    rows = frame.loc[frame[key].eq(asset), ["date", "value"]].copy()
    dates = pd.to_datetime(rows.date, utc=True, errors="raise").dt.tz_localize(None)
    if dates.isna().any():
        raise ValueError(f"M0: missing history date for {asset}")
    rows["date"] = pd.DatetimeIndex(dates).as_unit("ns")
    rows = rows[rows.date <= asof].sort_values("date").tail(300)
    if len(rows) < 3 or rows.date.duplicated().any():
        raise ValueError(f"M0: insufficient or duplicate history dates for {asset}")
    values = rows.value.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"M0: nonfinite trailing history for {asset}")
    dates = pd.DatetimeIndex(rows.date)
    intervals = np.asarray((dates[1:] - dates[:-1]).total_seconds()) / 86400.0
    valid = intervals <= max(10.0 * float(np.median(intervals)), 5.0)
    if target_type == "level":
        increments, anchor = np.diff(values), float(values[-1])
    elif target_type == "log_return":
        if np.median(np.abs(values)) >= 0.2:
            raise ValueError(f"M0: return-scale tripwire for {asset}")
        increments, anchor = values[1:], 0.0
    else:
        raise ValueError(f"M0: unsupported target_type {target_type}")
    return pd.Series(increments[valid], index=dates[1:][valid], name=asset), anchor, dates


def m0_reference(root: Path, card: dict, month_steps: dict | None = None,
                 asof: str | None = None, overrides: dict[str, pd.Series] | None = None) -> tuple[np.ndarray, dict]:
    """M0's 500 draws as (draws, assets, horizons) in card order, plus its moments.

    Step counts follow §3.7. The four published monthly cards use the counts §3.7 tabulates.
    Other cards have no public target date: daily panels keep the declared horizon, as §3.7
    does on every daily card, weekly-type and monthly panels count prints or months to an
    estimated target date (`_m0_step_count`), and a monthly card with one key below 21 and no
    observation month walks its next figure (`_next_figure_card`). The as-of is the card's
    (`declared_asof`), else `asof`, the task's. On all 104 public cards this equals
    backtest/m0.py bit for bit.
    """
    uid = card["task"]["id"]
    targets = card["targets"]
    assets, horizons = tuple(targets["asset_ids"]), tuple(targets["horizons"])
    cutoff = _m0_timestamp(declared_asof(card) or asof)
    panel_root = root / "panels" if (root / "panels").is_dir() else root
    frames = [pd.read_parquet(p) for p in sorted(panel_root.glob("*.parquet"))]
    sorted_assets = sorted(assets)
    aligned, anchors, windows, monthly = [], [], {}, {}
    for asset in sorted_assets:
        if overrides and asset in overrides:  # v2.7.5: a seasonally adjusted target replaces its panel rows
            own = overrides[asset]
            frame = pd.DataFrame({"date": own.index, "asset": asset, "value": own.to_numpy(dtype=float)})
            series, anchor, dates = _m0_history_steps(frame, asset, cutoff, targets["target_type"])
            aligned.append(series)
            anchors.append(anchor)
            windows[asset] = (dates[-1], _spacing(dates))
            monthly[asset] = monthly_index(dates)
            continue
        for frame in frames:
            key = "asset" if "asset" in frame else "asset_id"
            if key in frame and frame[key].eq(asset).any():
                series, anchor, dates = _m0_history_steps(frame, asset, cutoff, targets["target_type"])
                aligned.append(series)
                anchors.append(anchor)
                windows[asset] = (dates[-1], _spacing(dates))
                monthly[asset] = monthly_index(dates)
                break
        else:
            raise ValueError(f"M0: missing panel for {asset}")
    steps = pd.concat(aligned, axis=1, join="inner").dropna().sort_index()
    if len(steps) < 2 or not np.isfinite(steps.to_numpy()).all():
        raise ValueError("M0: insufficient valid jointly aligned steps")
    mu = steps.mean(axis=0).to_numpy()
    sigma = np.atleast_2d(np.cov(steps.to_numpy(), rowvar=False, ddof=1))
    cells = [(a, h) for a in sorted_assets for h in sorted(horizons)]
    table = M0_MONTHLY_STEPS.get(uid, {})
    # month_steps maps (asset, horizon) to a monthly card's observation-month count; the published
    # table still wins, and without either the §3.7 estimate is used.
    month_steps = month_steps or {}
    next_figure = not _has_monthly_mapping(card)
    s = np.asarray([table[h] if h in table else month_steps[(a, h)] if (a, h) in month_steps
                    else 1 if next_figure and _next_figure_card(horizons, monthly[a])
                    else _m0_step_count(h, *windows[a], cutoff) for a, h in cells],
                   dtype=float)
    ai = np.array([sorted_assets.index(a) for a, _ in cells])
    mean = np.asarray(anchors)[ai] + s * mu[ai]
    cov = np.minimum(s[:, None], s[None, :]) * sigma[ai[:, None], ai[None, :]]
    cov = cov + np.eye(len(cells)) * 1e-10
    cov = cov + np.eye(len(cells)) * 1e-9
    try:
        factor = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        factor = np.linalg.cholesky(np.diag(np.diag(cov)))
    seed = zlib.crc32(uid.encode("utf-8")) & 0x7FFFFFFF
    samples = mean + np.random.default_rng(seed).standard_normal((M0_DRAWS, len(cells))) @ factor.T
    order = [cells.index((a, h)) for a in assets for h in horizons]
    ref = {"anchor": np.asarray(anchors)[ai][order], "steps": s[order], "mu": mu[ai][order],
           "cov": cov[np.ix_(order, order)], "seed": seed, "aligned_rows": len(steps)}
    return samples[:, order].reshape(M0_DRAWS, len(assets), len(horizons)), ref


def _has_monthly_mapping(source: dict) -> bool:
    """Whether a card or spec supplies any field the official monthly helper reads."""
    targets = source.get("targets", {})
    if isinstance(targets, dict) and ("observation_periods" in targets or "target_dates" in targets):
        return True
    questions = source.get("questions", [])
    return isinstance(questions, list) and any(
        isinstance(row, dict) and ("observation_period" in row or "target_date" in row) for row in questions)


def resolve_monthly_steps(uid: str, assets: tuple[str, ...], horizons: tuple[int, ...],
                          selected: dict[str, pd.Series], asof: str, card: dict, spec: dict,
                          warnings: list[str]) -> NDArray[np.int64]:
    """Monthly step counts from the official observation-period helper.

    No mapping anywhere: the M0 §3.7 rule. A mapping the helper refuses (conflict, invalid
    month, misaligned grid): the card's own mapping when it is valid, since the scorer
    trusts the card; otherwise the card is refused with the helper's reason.
    """
    last = {a: str(s.index[-1].date()) for a, s in selected.items()}
    card_has, spec_has = _has_monthly_mapping(card), _has_monthly_mapping(spec)
    if not card_has and not spec_has:
        steps = m0_rule_monthly_steps(uid, selected, horizons, asof)
        warnings.append(f"no monthly observation-period mapping; M0 §3.7 step rule used: {steps.tolist()}")
        return steps
    try:
        return monthly_horizon_steps(assets, horizons, last, asof=asof, card=card,
                                     forecast_spec=spec).astype(np.int64)
    except HorizonMetadataError as exc:
        if not card_has:
            raise ContractError(f"the spec's monthly mapping is invalid and the card has none: {exc}") from exc
        try:
            steps = monthly_horizon_steps(assets, horizons, last, asof=asof, card=card,
                                          forecast_spec=None).astype(np.int64)
        except HorizonMetadataError as inner:
            raise ContractError(f"the card's monthly mapping is invalid: {inner}") from inner
        warnings.append(f"monthly mapping refused ({exc}); the card's own mapping is used, the spec's is ignored")
        return steps


def m0_rule_monthly_steps(uid: str, selected: dict[str, pd.Series], horizons: tuple[int, ...], asof: str) -> NDArray[np.int64]:
    """Monthly steps by the M0 §3.7 rule, for a card that supplies no observation-period mapping."""
    table = M0_MONTHLY_STEPS.get(uid, {})
    next_figure = _next_figure_card(horizons, monthly=True)  # every series here is monthly, the card's cadence
    out = np.empty((len(selected), len(horizons)), dtype=np.int64)
    for ai, series in enumerate(selected.values()):
        window = series.index[-300:]
        for hi, h in enumerate(horizons):
            out[ai, hi] = (table[h] if h in table else 1 if next_figure
                           else _m0_step_count(h, window[-1], _spacing(window), pd.Timestamp(asof)))
    return out


def _covariance_root(cov: np.ndarray) -> np.ndarray:
    """A square root that does not depend on the LAPACK build.

    Cholesky is unique for a positive-definite matrix. Eigenvector signs are arbitrary and
    differ between builds, so the eigen fallback fixes each vector's largest entry positive.
    """
    try:
        return np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        eig, vectors = np.linalg.eigh(cov)
        signs = np.sign(vectors[np.argmax(np.abs(vectors), axis=0), np.arange(vectors.shape[1])])
        return vectors * np.where(signs == 0, 1.0, signs) * np.sqrt(np.maximum(eig, 0.0))


def m0t_draws(ref: dict, n: int, rng: np.random.Generator, *, df: float | None, drift: float,
              scale: float) -> NDArray[np.float64]:
    """Draws (n, cells) from M0's mean (drift scaled) and covariance, card order.

    Student-t shocks when df is given. Each cell is standardized over the draws, which
    removes Monte Carlo error from its centre and scale and leaves correlations intact.
    """
    mean = ref["anchor"] + drift * ref["steps"] * ref["mu"]
    z = rng.standard_normal((n, len(mean))) @ _covariance_root(ref["cov"]).T
    if df is not None:
        z *= np.sqrt((df - 2.0) / rng.chisquare(df, n))[:, None]
    z -= z.mean(axis=0)
    sd = z.std(axis=0, ddof=1)
    z *= np.divide(np.sqrt(np.diag(ref["cov"])), sd, out=np.zeros_like(sd), where=sd > 0)
    return mean + scale * z


def _proxy_scale(task: TaskContext, model: NumericModel, asset: str, settings: Settings) -> float:
    """Median current daily scale of the fresh G10 proxies the engine chose for a transfer target."""
    out = []
    for name in model.audit["assets"][asset].get("scale_proxy", []):
        proxy = transform_for(task, name)
        values = innovations(proxy.state(task.context_histories[name]), proxy.kind, task.cadence)
        out.append(scale_estimate(values.to_numpy(), task.cadence, settings.transfer_vol_blend)[0])
    return float(np.median(out)) if out else float("nan")


def transfer_m0_draws(task: TaskContext, model: NumericModel, ref: dict, settings: Settings,
                      n: int) -> tuple[NDArray[np.float64], dict]:
    """Gaussian draws on M0's mean and covariance for a card whose every target is a transfer asset.

    Each cell's sd is the geometric mean of M0's and the G10 proxies' current scale ("geo"), or M0's
    times transfer_m0_scale ("m0"); a cell without a proxy scale keeps M0's. Chosen on transfer-shaped
    tasks with real EM targets (FRED H.10) whose outcomes end before 2014-12-18: the engine's width
    scored about 1.9 against M0 there, these rules about 1.0-1.08. ARTIFACT_PROVENANCE.md has the record.
    """
    z = m0t_draws(ref, n, generator(settings.seed, task.fingerprint, "transfer-m0"), df=None, drift=1.0,
                  scale=1.0)
    mean = ref["anchor"] + ref["steps"] * ref["mu"]
    m0_sd = np.sqrt(np.diag(ref["cov"]))
    if settings.transfer_m0 == "geo":
        nh = len(task.horizons)
        spot = np.repeat([float(task.histories[a].iloc[-1]) for a in task.assets], nh)
        proxy = np.repeat([_proxy_scale(task, model, a, settings) for a in task.assets], nh)
        target = np.sqrt(m0_sd * proxy * spot * np.sqrt(ref["steps"]))
        usable = np.isfinite(target) & (target > 0) & (m0_sd > 0)
        factor = np.where(usable, target / np.where(m0_sd > 0, m0_sd, 1.0), 1.0)
    else:
        factor = np.full(len(mean), float(settings.transfer_m0_scale))
    draws = mean + (z - mean) * factor
    return draws.reshape(n, len(task.assets), len(task.horizons)), {
        "rule": settings.transfer_m0, "sd_factor_vs_m0": [float(v) for v in factor]}


def _pair_statistic(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b) ** 0.5))


def match_pair(flat: np.ndarray, target: float, target_gap: float, rng: np.random.Generator, *,
               allow_shift: bool) -> tuple[np.ndarray, dict]:
    """Re-pair two columns so mean|X1-X2|^0.5 equals target; each column keeps its values.

    Pairings come from a Gaussian copula with correlation rho, and a final stretch of each
    pair difference by 1 + eps (|eps| <= 1%) closes the last rank-swap gap exactly. When no
    rho reaches the target and allow_shift is set, both centres move symmetrically toward
    M0's gap by the smallest fraction that makes it reachable (bisected to 0.05/4096).
    When nothing reaches it, the input is returned unchanged with reached=False.
    """
    n = len(flat)
    s1, s2 = np.sort(flat[:, 0]), np.sort(flat[:, 1])
    z1, z2 = rng.standard_normal(n), rng.standard_normal(n)
    r1 = np.argsort(np.argsort(z1))
    own_gap = float(s2.mean() - s1.mean())

    def pairing(rho: float, t: float) -> tuple[np.ndarray, np.ndarray]:
        w = rho * z1 + math.sqrt(max(1.0 - rho * rho, 0.0)) * z2
        shift = t * (target_gap - own_gap) / 2.0
        return s1[r1] - shift, s2[np.argsort(np.argsort(w))] + shift

    grid = np.linspace(-1.0, 1.0, 41)

    def bracket(t: float) -> tuple[float, float] | None:
        values = np.array([_pair_statistic(*pairing(r, t)) for r in grid]) - target
        crossing = np.flatnonzero(np.sign(values[:-1]) * np.sign(values[1:]) <= 0)
        return (grid[crossing[0]], grid[crossing[0] + 1]) if len(crossing) else None

    t, found = 0.0, bracket(0.0)
    if found is None and allow_shift:
        previous = 0.0
        for step in np.linspace(0.05, 1.0, 20):
            candidate = bracket(float(step))
            if candidate is not None:
                low, high = previous, float(step)
                for _ in range(12):
                    middle = (low + high) / 2.0
                    inner = bracket(middle)
                    if inner is None:
                        low = middle
                    else:
                        high, candidate = middle, inner
                t, found = high, candidate
                break
            previous = float(step)
    if found is None:
        return flat, {"reached": False, "shift_fraction": 0.0}
    lo, hi = found
    sign_lo = np.sign(_pair_statistic(*pairing(lo, t)) - target)
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if np.sign(_pair_statistic(*pairing(mid, t)) - target) == sign_lo:
            lo = mid
        else:
            hi = mid
    candidates = [pairing(lo, t), pairing(hi, t)]
    a, b = min(candidates, key=lambda ab: abs(_pair_statistic(*ab) - target))
    a, b, stretch = _polish(a, b, target)
    return np.column_stack([a, b]), {"reached": True, "rho": float((lo + hi) / 2.0), "shift_fraction": float(t),
                                     "polish_stretch": stretch,
                                     "relative_residual": abs(_pair_statistic(a, b) - target) / max(target, 1e-300)}


def _polish(a: np.ndarray, b: np.ndarray, target: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Close the last rank-swap gap exactly by stretching every pair difference by 1 + eps.

    mean|X1-X2|^0.5 scales as (1 + eps)^0.5, so eps = (target / current)^2 - 1. Each value
    moves by eps/2 of its own pair difference; eps is of order 1/draws and is capped at 1%.
    """
    current = _pair_statistic(a, b)
    if current <= 0 or target <= 0:
        return a, b, 0.0
    eps = (target / current) ** 2 - 1.0
    if abs(eps) > 0.01:
        return a, b, 0.0
    half = 0.5 * eps * (a - b)
    return a + half, b - half, float(eps)


def sharpen_for(settings: Settings, family: str) -> tuple[float, float]:
    """The (alpha, k) pair for a card family; (0.0, 1.0), no change, for an unknown family."""
    fam = str(family).strip().upper()
    for suffix, pair in (("F1", settings.sharpen_f1), ("F2", settings.sharpen_f2),
                         ("F3", settings.sharpen_f3), ("F4", settings.sharpen_f4)):
        if fam.endswith(suffix):
            return pair
    return (0.0, 1.0)


def m0_sharpen(samples: np.ndarray, ref: dict, alpha: float, k: float,
               rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """Resample joint draws by 1 / CRPS_M0^alpha, then scale each cell by k about its median.

    The board divides each card's score by M0's on the same outcome. Minimizing the mean of
    those ratios is minimizing the score against the outcome distribution reweighted by
    1 / M0's score, which is largest near M0's centre; for CRPS the optimum is exactly the
    predictive distribution times 1 / CRPS_M0. The weight of a draw x is 1 / mean_c CRPS_c(x_c)
    ^ alpha, with CRPS_c the closed-form CRPS of M0's Gaussian marginal (mean anchor + s*mu,
    sd from its covariance) in units of that sd. Systematic resampling keeps whole draws, so
    the cross-cell dependence is reweighted with them, not rebuilt.
    """
    n = len(samples)
    flat = samples.reshape(n, -1)
    info: dict[str, Any] = {"alpha": float(alpha), "k": float(k)}
    if alpha > 0:
        mean0 = np.asarray(ref["anchor"], float) + np.asarray(ref["steps"], float) * np.asarray(ref["mu"], float)
        sd0 = np.sqrt(np.maximum(np.diag(np.asarray(ref["cov"], float)), 0.0))
        if mean0.shape != (flat.shape[1],) or not np.all(np.isfinite(mean0)) or not np.all(np.isfinite(sd0)) \
                or np.any(sd0 <= 1e-12):
            info["reweight"] = "skipped: unusable M0 moments"
        else:
            z = (flat - mean0) / sd0
            crps = z * (2.0 * ndtr(z) - 1.0) + 2.0 * np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi) \
                - 1.0 / math.sqrt(math.pi)
            weight = np.mean(crps, axis=1) ** -float(alpha)
            weight /= weight.sum()
            positions = (rng.random() + np.arange(n)) / n
            flat = flat[np.minimum(np.searchsorted(np.cumsum(weight), positions), n - 1)]
            info["reweight"] = "applied"
            info["effective_draws"] = float(1.0 / np.sum(weight * weight))
    if k != 1.0:
        median = np.median(flat, axis=0)
        flat = median + (flat - median) * float(k)
    return flat.reshape(samples.shape), info


def is_shock_family(family: str) -> bool:
    return str(family).strip().upper().endswith("F4")


def _fresh_daily(task: TaskContext, model: NumericModel) -> bool:
    assets = model.audit.get("assets", {})
    return task.cadence == "daily" and all(not assets.get(a, {}).get("transfer", True) for a in task.assets)


def _pair_match(task: TaskContext, samples: np.ndarray, m0_samples: np.ndarray, ref: dict, seed: int, *,
                fresh: bool) -> tuple[np.ndarray, dict]:
    """Match the single variogram pair to M0's own 500-draw value, or leave the draws alone.

    `method` in the returned record says what moved: `repair` (pairing only), `shift` (the
    smallest centre move toward M0's gap), `m0-blend` (the smallest share of draws, in steps
    of 5%, replaced by draws from M0's distribution), or `unmatched` (nothing changed).
    Only daily cards with fresh histories may move centres or blend; on monthly and
    transfer cards an unreachable target leaves the engine's pairing as it is.
    """
    n = len(samples)
    flat = samples.reshape(n, -1)
    reference = m0_samples.reshape(len(m0_samples), -1)
    target = _pair_statistic(reference[:, 0], reference[:, 1])
    gap = float(np.mean(reference[:, 1] - reference[:, 0]))
    rng = generator(seed, task.fingerprint, "pair-copula")
    matched, info = match_pair(flat, target, gap, rng, allow_shift=fresh)
    info.update(m0_statistic=target, m0_blend=0.0)
    if info["reached"]:
        info["method"] = "shift" if info["shift_fraction"] > 0 else "repair"
        return matched.reshape(samples.shape), info
    if fresh:
        order = rng.permutation(n)
        replacement = m0t_draws(ref, n, rng, df=None, drift=1.0, scale=1.0)
        for share in np.linspace(0.05, 1.0, 20):
            m = int(round(n * share))
            mixed = np.vstack([flat[np.sort(order[: n - m])], replacement[:m]])
            blended, record = match_pair(mixed, target, gap, rng, allow_shift=True)
            if record["reached"]:
                record.update(m0_statistic=target, m0_blend=m / n, method="m0-blend")
                return blended.reshape(samples.shape), record
    info["method"] = "unmatched"
    return samples, info


def m0_fallback(task: TaskContext, settings: Settings, requested: dict, reason: str) -> tuple[np.ndarray, NumericModel]:
    """Failure path: M0's distribution, then a local random walk, then the wide grid."""
    n = min(20_000, max(int(settings.draws), int(task.min_draws), 200))
    model = _shell_model(task, reason)
    try:
        m0_samples, ref = m0_reference(task.root, task.card, asof=task.asof, overrides=task.panel_overrides)
        flat = m0t_draws(ref, n, generator(settings.seed, task.fingerprint, "m0-fallback"), df=None, drift=1.0, scale=1.0)
        samples = flat.reshape(n, len(task.assets), len(task.horizons))
        if task.cell_count == 2:
            samples, info = _pair_match(task, samples, m0_samples, ref, settings.seed, fresh=True)
            model.audit["pair_match"] = info
        model.audit["engine"] = "m0-fallback"
    except Exception as exc:
        model.audit["m0_fallback_error"] = f"{type(exc).__name__}: {exc}"
        try:
            samples = random_walk_samples(task, n, generator(settings.seed, task.fingerprint, "rw-fallback"))
            model.audit["engine"] = "random-walk-fallback"
        except Exception as exc2:
            model.audit["random_walk_error"] = f"{type(exc2).__name__}: {exc2}"
            samples = wide_samples(task, settings)
    if not np.isfinite(samples).all():
        samples = wide_samples(task, settings)
        model.audit["engine"] = "wide-fallback"
    _stamp(model, task, settings, requested)
    return samples, model


def random_walk_samples(task: TaskContext, n: int, rng: np.random.Generator) -> np.ndarray:
    """Last value plus a Student-t walk scaled by a robust step deviation of each target."""
    steps = np.asarray(task.steps)
    if steps.ndim == 1:
        steps = np.broadcast_to(steps, (len(task.assets), len(task.horizons)))
    out = np.empty((n, len(task.assets), len(task.horizons)))
    shocks = rng.standard_t(5.0, size=(n, len(task.assets))) / math.sqrt(5.0 / 3.0)
    for i, asset in enumerate(task.assets):
        hist = task.histories[asset].dropna().astype(float).iloc[-301:]
        if len(hist) < 6:
            raise ContractError(f"too little history for a random walk on {asset}")
        if task.target_type == "log_return":
            moves, last = np.log1p(hist.to_numpy()), 0.0
        else:
            moves, last = np.diff(hist.to_numpy()), float(hist.iloc[-1])
        moves = moves[np.isfinite(moves)]
        if len(moves) < 5:
            raise ContractError(f"too little history for a random walk on {asset}")
        sd = 1.4826 * float(np.median(np.abs(moves - np.median(moves))))
        sd = sd if sd > 0 else float(np.std(moves))
        for hi, h in enumerate(steps[i]):
            out[:, i, hi] = last + shocks[:, i] * sd * math.sqrt(max(int(h), 1))
    return out


RATES_PANEL = "rates_daily"
YEAR_STEPS = 252.0
_TENOR = re.compile(r"UST_(\d+)Y")


def tenor_years(asset: str) -> float | None:
    match = _TENOR.fullmatch(asset)
    return float(match.group(1)) if match else None


def curve_at_asof(task: TaskContext) -> dict[float, float]:
    """Last observation at or before the as-of date of every UST_<n>Y series the unit carries."""
    asof = pd.Timestamp(task.asof)
    curve: dict[float, float] = {}
    for asset, series in {**task.context_histories, **task.histories}.items():
        tenor = tenor_years(asset)
        if tenor is None:
            continue
        s = series.dropna()
        s = s[s.index <= asof]
        if len(s):
            curve[tenor] = float(s.iloc[-1])
    return curve


def interpolated_yield(curve: dict[float, float], maturity: float) -> float:
    tenors = sorted(curve)
    return float(np.interp(maturity, tenors, [curve[t] for t in tenors]))  # flat beyond both ends


def forward_yield(curve: dict[float, float], h: float, tenor: float) -> float:
    """Expectations-hypothesis forward yield of a `tenor`-year bond `h` years ahead (zero-coupon approximation)."""
    return ((tenor + h) * interpolated_yield(curve, tenor + h) - h * interpolated_yield(curve, h)) / tenor


def rates_forward_center(task: TaskContext, samples: np.ndarray, settings: Settings) -> tuple[np.ndarray, dict]:
    """v2.7 (E21): shift every Treasury cell by `rates_forward_add` times the forward increment f(h, T) - y(T).

    T is the target tenor in years, h = steps / 252, y(.) the as-of curve read from the unit's own UST_<n>Y
    series, linear in maturity and flat beyond the shortest and longest tenor. Shape, width and dependence of
    the draws are untouched; monthly, log-return, transfer and non-Treasury cells never change. Never raises:
    anything unexpected leaves the draws unchanged and records why.
    """
    info: dict[str, Any] = {"strength": float(settings.rates_forward_add), "cells": []}
    try:
        if task.cadence != "daily" or task.target_type != "level":
            info["skipped"] = f"{task.cadence} {task.target_type} task"
            return samples, info
        curve = curve_at_asof(task)
        if len(curve) < 2:
            info["skipped"] = f"{len(curve)} Treasury tenors at the as-of date"
            return samples, info
        out = np.array(samples, dtype=np.float64, copy=True)
        steps = np.asarray(task.steps)
        for ai, asset in enumerate(task.assets):
            tenor = tenor_years(asset)
            if tenor is None or task.panel_for_asset.get(asset) != RATES_PANEL or tenor not in curve:
                continue
            spot = curve[tenor]
            for hi in range(len(task.horizons)):
                h = int(steps[ai, hi]) / YEAR_STEPS
                fwd = forward_yield(curve, h, tenor)
                cell = out[:, ai, hi]
                median = float(np.median(cell))
                shift = (fwd - spot) * float(settings.rates_forward_add)
                cell += shift
                info["cells"].append({"asset": asset, "horizon": int(task.horizons[hi]), "steps": int(steps[ai, hi]),
                                      "spot": spot, "forward": fwd, "engine_median": median, "shift": shift})
        info["tenors"] = sorted(curve)
        if not np.isfinite(out).all():
            info["skipped"] = "non-finite shift"
            return samples, info
        return out, info
    except Exception as exc:  # the engine's draws stand; the reason is recorded
        info["skipped"] = f"{type(exc).__name__}: {exc}"
        return samples, info


# ---------------------------------------------------------------- v2.7.3 FOMC-week variance profile (E24)
# m(class, s) for steps s = 1-4 after a Monday as-of in an FOMC week: the variance of each step relative to an average
# Tuesday-Friday step, the Gaussian-CRPS scale of one-step changes over v2.7.1's current step sd, estimated on
# 2005-2018 only (exp/README.md, E24; FRED H.15 Treasury yields and ECB reference rates cross-calculated to USD) and
# clipped to [0.5, 2.5]. Every class is calm on the Tuesday; the Treasury close takes the decision at step 2 and
# the ECB fixing (13:10 UTC, before the 18:00 UTC statement) at step 3.
EVENT_PROFILE: dict[str, tuple[float, float, float, float]] = {
    "ust": (0.6377, 1.6493, 1.0908, 1.1167),
    "bills": (0.7790, 1.6234, 1.4336, 1.0154),
    "g10": (0.8338, 0.7571, 2.5, 1.0216),
    "em": (0.9210, 0.6542, 2.5, 1.0860),
}
# FOMC statements on a Wednesday, 2005-2024 (scheduled and unscheduled, the list of exp/e23_short.py), and the ranked
# window's decision of 28 October 2026 from the Federal Reserve's published 2026 meeting schedule.
FOMC_WEDNESDAYS = frozenset(pd.to_datetime("""
2005-02-02 2006-05-10 2006-09-20 2006-10-25 2007-01-31 2007-03-21 2007-05-09 2007-10-31
2008-01-30 2008-04-30 2008-06-25 2008-10-08 2008-10-29
2009-01-28 2009-03-18 2009-04-29 2009-06-24 2009-08-12 2009-09-23 2009-11-04 2009-12-16
2010-01-27 2010-04-28 2010-06-23 2010-11-03 2011-01-26 2011-04-27 2011-06-22 2011-09-21 2011-11-02
2012-01-25 2012-04-25 2012-06-20 2012-08-01 2012-10-24 2012-12-12
2013-01-30 2013-03-20 2013-05-01 2013-06-19 2013-07-31 2013-09-18 2013-10-30 2013-12-18
2014-01-29 2014-03-19 2014-04-30 2014-06-18 2014-07-30 2014-09-17 2014-10-29 2014-12-17
2015-01-28 2015-03-18 2015-04-29 2015-06-17 2015-07-29 2015-10-28 2015-12-16
2016-01-27 2016-03-16 2016-04-27 2016-06-15 2016-07-27 2016-09-21 2016-11-02 2016-12-14
2017-02-01 2017-03-15 2017-05-03 2017-06-14 2017-07-26 2017-09-20 2017-11-01 2017-12-13
2018-01-31 2018-03-21 2018-05-02 2018-06-13 2018-08-01 2018-09-26 2018-12-19
2019-01-30 2019-03-20 2019-05-01 2019-06-19 2019-07-31 2019-09-18 2019-10-30 2019-12-11
2020-01-29 2020-04-29 2020-06-10 2020-07-29 2020-09-16 2020-12-16
2021-01-27 2021-03-17 2021-04-28 2021-06-16 2021-07-28 2021-09-22 2021-11-03 2021-12-15
2022-01-26 2022-03-16 2022-05-04 2022-06-15 2022-07-27 2022-09-21 2022-11-02 2022-12-14
2023-02-01 2023-03-22 2023-05-03 2023-06-14 2023-07-26 2023-09-20 2023-11-01 2023-12-13
2024-01-31 2024-03-20 2024-05-01 2024-06-12 2024-07-31 2024-09-18 2024-12-18
2026-10-28
""".split()))
EVENT_G10 = frozenset({"EUR", "GBP", "AUD", "NZD", "JPY", "CHF", "CAD", "SEK", "NOK", "DKK"})
EVENT_EM = frozenset({"MXN", "BRL", "ZAR", "KRW", "INR"})
# Where the profile applies it replaces every family width: k 1.0 for every family and F4 at the default engine width.
FAMILY_NEUTRAL = {"sharpen_f1": (0.0, 1.0), "sharpen_f2": (0.0, 1.0), "sharpen_f3": (0.0, 1.0),
                  "sharpen_f4": (0.0, 1.0), "f4_widen": False}
_EVENT_TENOR = re.compile(r"UST_(\d+)([MY])")


def event_fx_code(name: str) -> str | None:
    """The non-USD currency of an FX target name: a bare code (MXN) or an unambiguous pair with USD (USD_MXN, USDMXN,
    EURUSD, EUR/USD), case and separators ignored. None for USD itself, a cross without USD, or anything else."""
    letters = re.sub(r"[\s_/.\-]", "", str(name).upper())
    if re.fullmatch(r"[A-Z]{3}", letters):
        return None if letters == "USD" else letters
    if re.fullmatch(r"[A-Z]{6}", letters):
        base, quote = letters[:3], letters[3:]
        if (base == "USD") != (quote == "USD"):
            return quote if base == "USD" else base
    return None


def event_asset_class(asset: str) -> str | None:
    """The profile class of a target: "bills" (UST_<n>M, or UST_<n>Y below 2 years), "ust" (UST_<n>Y from 2 years),
    "g10" or "em" (the ranked Final's 15 exchange rates, as a code or a pair with USD). None means no profile."""
    match = _EVENT_TENOR.fullmatch(str(asset).strip().upper())
    if match:
        years = int(match.group(1)) / (12.0 if match.group(2) == "M" else 1.0)
        if years <= 0:
            return None
        return "bills" if years < 2.0 else "ust"
    code = event_fx_code(asset)
    return "g10" if code in EVENT_G10 else "em" if code in EVENT_EM else None


def fomc_week(asof: str) -> bool:
    """Whether the as-of is the Monday of a week whose FOMC decision falls on the Wednesday (FOMC_WEDNESDAYS)."""
    day = pd.Timestamp(asof)
    # Date arithmetic, not Timestamp + Timedelta, which warns under the pinned numpy (a generic timedelta unit).
    return day.dayofweek == 0 and pd.Timestamp(day.date() + timedelta(days=2)) in FOMC_WEDNESDAYS


def event_profile_reason(task: TaskContext) -> str | None:
    """None when the FOMC-week profile applies to the card, else the reason it does not."""
    if task.cadence != "daily" or task.target_type != "level":
        return f"{task.cadence} {task.target_type} card"
    if not fomc_week(task.asof):
        return "as-of is not the Monday of an FOMC week"
    steps = np.asarray(task.steps)
    if steps.min() < 1 or steps.max() > 4:
        return "a step outside 1-4"
    if any(event_asset_class(a) is None for a in task.assets):
        return "a target outside the profile's classes"
    asof = pd.Timestamp(task.asof)
    if any((asof - task.histories[a].index[-1]).days > 5 for a in task.assets):
        return "a target without a fresh history"
    return None


def apply_event_profile(task: TaskContext, samples: np.ndarray) -> tuple[np.ndarray, dict]:
    """E24's path transform, per asset in yield units for Treasuries and log units for FX (so the quote direction does
    not matter): the increments of each cell's deviation from its median, taken in horizon order, are scaled by the
    square root of the mean of EVENT_PROFILE over the steps they span, cumulated back and re-centred on each cell's
    median. A single cell at h is scaled by sqrt(mean of m over s <= h), any two horizons of one asset see the
    profile's step variances, and cross-asset correlations are kept. Never raises: anything unexpected leaves the
    draws unchanged and records why."""
    info: dict[str, Any] = {"family_widths": "reset: k 1.0 for every family, F4 at the default engine width",
                            "assets": {}}
    try:
        out = np.array(samples, dtype=np.float64, copy=True)
        steps = np.asarray(task.steps)
        for ai, asset in enumerate(task.assets):
            cls = event_asset_class(asset)
            m = np.asarray(EVENT_PROFILE[cls], dtype=float)
            order = np.argsort(steps[ai], kind="stable")
            h = steps[ai][order]
            edges = np.r_[0, h]
            factors = np.array([np.sqrt(m[edges[k]:edges[k + 1]].mean()) for k in range(len(h))])
            log = cls in ("g10", "em")
            values = out[:, ai, order]
            if log and not np.all(values > 0):
                info["skipped"] = f"non-positive draws for {asset}"
                return samples, info
            state = np.log(values) if log else values
            center = np.median(state, axis=0)
            increments = np.diff(state - center, axis=1, prepend=0.0)
            path = np.cumsum(increments * factors, axis=1)
            path -= np.median(path, axis=0)
            state = center + path
            out[:, ai, order] = np.exp(state) if log else state
            info["assets"][asset] = {"class": cls, "steps": h.tolist(), "increment_factors": factors.round(6).tolist()}
        if not np.isfinite(out).all():
            info["skipped"] = "non-finite profiled draws"
            return samples, info
        return out, info
    except Exception as exc:  # the draws stand; the reason is recorded
        info["skipped"] = f"{type(exc).__name__}: {exc}"
        return samples, info


# ---------------------------------------------------------------- v2.7.4 regime switches (E29, E30)

# E30: a target name reads as a price index (case ignored). An unmatched name keeps v2.7.3's monthly path.
PRICE_NAMES = re.compile(r"CPI|PCE|HICP|PPI|PRICE|INFL|DEFL", re.IGNORECASE)


def ust_floor_level(task: TaskContext) -> float | None:
    """E29: the lowest last value at or before the as-of over the card's UST series (bills included), in percent."""
    levels = [float(s.iloc[-1]) for name, s in task.context_histories.items()
              if event_asset_class(name) in ("ust", "bills") and len(s)]
    return min(levels) if levels else None


def monthly_regime_ratio(task: TaskContext) -> float:
    """E28/E30: the mean over targets of scale_estimate's current / long step scale (monthly, blend 1.0)."""
    ratios = []
    for asset in task.assets:
        transform = transform_for(task, asset)
        inc = innovations(transform.state(task.histories[asset]), transform.kind, "monthly")
        current, long = scale_estimate(inc.to_numpy(), "monthly", 1.0)
        ratios.append(current / long if long > 0 else 1.0)
    return float(np.mean(ratios))


def keep_ust_family_widths(task: TaskContext, samples: np.ndarray, settings: Settings) -> tuple[np.ndarray, Any]:
    """E29: on a profiled card away from the zero lower bound, UST-notes cells take the profile on top of their
    family widths. The family-width draws come from the same card, seed and draw order with the profile off
    (no step reorders rows: pairing off, daily alphas 0), so swapping those cells keeps the rows' dependence.
    Never raises: anything unexpected leaves the draws as they are and records why."""
    notes = [ai for ai, a in enumerate(task.assets) if event_asset_class(a) == "ust"]
    if not notes:
        return samples, "not applied: no UST-notes target"
    floor = ust_floor_level(task)
    if floor is None or floor < settings.ust_active_floor:
        return samples, f"not applied: lowest UST yield {floor} below {settings.ust_active_floor}"
    try:
        plain, _ = predict_core(task, replace(settings, event_profile=False, ust_family_widths_active=False))
        kept, info = apply_event_profile(task, plain)
        if "skipped" in info or kept.shape != samples.shape or not np.isfinite(kept).all():
            return samples, f"not applied: {info.get('skipped', 'unusable family-width draws')}"
        out = np.array(samples, dtype=np.float64, copy=True)
        out[:, notes, :] = kept[:, notes, :]
        return out, {"applied_to": [task.assets[i] for i in notes], "lowest_ust_yield": floor}
    except Exception as exc:  # the profiled draws stand; the reason is recorded
        return samples, f"not applied: {type(exc).__name__}: {exc}"


def monthly_price_vol_reason(task: TaskContext, settings: Settings) -> str | None:
    """None when E30's switch fires on this fresh monthly card (the reweighting is skipped), else why not."""
    if not settings.monthly_price_vol_switch:
        return "off"
    if not all(PRICE_NAMES.search(a) for a in task.assets):
        return "a target that does not read as a price index"
    ratio = monthly_regime_ratio(task)
    if ratio < settings.monthly_price_vol_ratio:
        return f"recent/long volatility {ratio:.3f} below {settings.monthly_price_vol_ratio}"
    return None


# ---------------------------------------------------------------- v2.7.5 seasonal adjustment (E35, E36)

SEASONAL_CHANGES = 96
SEASONAL_P = 0.001


def seasonal_effects(history: pd.Series) -> dict | None:
    """v2.7.5: calendar-month effects of a monthly target, or None when its last 96 changes show none.

    Changes are log changes when every value is positive, else differences, over consecutive months. Detection is a
    Kruskal-Wallis test across the 12 calendar months (p < 0.001). A month's effect is the mean of its changes
    without the highest and the lowest, centred to sum to zero; `running` is their sum from January to each month.
    """
    x = history.dropna()
    if len(x) < SEASONAL_CHANGES + 1:
        return None
    log = bool((x > 0).all())
    state = np.log(x) if log else x
    consecutive = np.r_[False, np.diff(x.index.to_period("M").asi8) == 1]
    changes = state.diff()[consecutive].dropna()
    if len(changes) < SEASONAL_CHANGES:
        return None
    changes = changes.iloc[-SEASONAL_CHANGES:]
    groups = {m: np.sort(g.to_numpy()) for m, g in changes.groupby(changes.index.month)}
    if len(groups) != 12 or min(len(g) for g in groups.values()) < 3:
        return None
    from scipy.stats import kruskal

    p = float(kruskal(*[groups[m] for m in range(1, 13)]).pvalue)
    if not p < SEASONAL_P:
        return None
    trimmed = np.array([g[1:-1].mean() if len(g) >= 5 else g.mean() for g in (groups[m] for m in range(1, 13))])
    effects = trimmed - trimmed.mean()
    return {"log": log, "p": p, "effects": effects, "running": np.cumsum(effects)}


def seasonal_plan(task: TaskContext) -> dict[str, dict]:
    """v2.7.5: the targets of a monthly card that carry calendar-month effects, with those effects."""
    plan = {}
    for asset in task.assets:
        effects = seasonal_effects(task.histories[asset]) if asset in task.histories else None
        if effects is not None:
            plan[asset] = effects
    return plan


def seasonal_offset(effects: dict, last: pd.Timestamp, months: int) -> float:
    """S(c_T) - S(c_L): the calendar effect from the last observed month to `months` months later."""
    target = (last.month - 1 + int(months)) % 12
    return float(effects["running"][target] - effects["running"][last.month - 1])


def predict_seasonal(task: TaskContext, settings: Settings, plan: dict[str, dict]) -> tuple[np.ndarray, NumericModel]:
    """v2.7.5: forecast the adjusted targets with every other step unchanged, then put the draws back on the calendar.
    The seed fingerprint is the card's own. Anything unexpected falls back to the plain forecast."""
    try:
        histories, overrides = dict(task.histories), {}
        for asset, effects in plan.items():
            x = task.histories[asset]
            months = x.index.month.to_numpy()
            offset = effects["running"][months - 1] - effects["running"][x.index[-1].month - 1]
            adjusted = x * np.exp(-offset) if effects["log"] else x - offset
            histories[asset] = overrides[asset] = adjusted
        samples, model = predict_core(replace(task, histories=histories, panel_overrides=overrides),
                                 replace(settings, monthly_seasonal=False))
        out = np.array(samples, dtype=np.float64, copy=True)
        info = {}
        for ai, asset in enumerate(task.assets):
            if asset not in plan:
                continue
            effects, last = plan[asset], task.histories[asset].index[-1]
            offsets = [seasonal_offset(effects, last, task.steps[ai, hi]) for hi in range(len(task.horizons))]
            for hi, off in enumerate(offsets):
                out[:, ai, hi] = out[:, ai, hi] * np.exp(off) if effects["log"] else out[:, ai, hi] + off
            info[asset] = {"p": effects["p"], "log": effects["log"], "effects": np.round(effects["effects"], 6).tolist(),
                           "offsets": offsets}
        if not np.isfinite(out).all():
            raise ValueError("non-finite seasonal draws")
        model.audit["seasonal"] = info
        for key in ("requested_settings", "settings"):
            if isinstance(model.audit.get(key), dict):
                model.audit[key]["monthly_seasonal"] = True
        return out, model
    except Exception as exc:  # the plain forecast stands; the reason is recorded
        samples, model = predict_core(task, replace(settings, monthly_seasonal=False))
        model.audit["seasonal"] = f"not applied: {type(exc).__name__}: {exc}"
        return samples, model


def predict_core(task: TaskContext, settings: Settings) -> tuple[np.ndarray, NumericModel]:
    """v2.7.5's forecast. predict applies v2.7.6's steps on top of it."""
    if settings.monthly_seasonal and task.cadence == "monthly" and not task.panel_overrides:
        plan = seasonal_plan(task)
        if plan:
            return predict_seasonal(task, settings, plan)
    requested = asdict(settings)
    settings = settings.for_cadence(task.cadence)
    if settings.draws < task.min_draws:
        settings = replace(settings, draws=task.min_draws)
    stamped = settings  # the effective settings the audit records; the profile's reset below is per card
    event_reason = event_profile_reason(task) if settings.event_profile else "off"
    if event_reason is None:
        settings = replace(settings, **FAMILY_NEUTRAL)

    def degraded(reason: str) -> tuple[np.ndarray, NumericModel]:
        if stamped.robust_fallback:
            return m0_fallback(task, stamped, requested, reason)
        samples = wide_samples(task, stamped)
        model = _shell_model(task, reason)
        _stamp(model, task, stamped, requested)
        return samples, model

    shock = settings.f4_widen and task.cadence == "daily" and is_shock_family(task.family)
    engine = replace(settings, width_scale=settings.f4_width_scale) if shock else settings
    f3_v24 = (settings.f3_width_profile == "v2.4" and task.cadence == "daily"
              and str(task.family).strip().upper().endswith("F3"))
    if f3_v24:
        engine = replace(engine, **V2_4_SETTINGS)
    try:
        model = fit_base(task, engine)
    except Exception as exc:
        if settings.robust_fallback:
            return degraded(type(exc).__name__)
        try:
            model = fit_base(task, engine, fallback=True)
            model.audit.update(fallback_reason=type(exc).__name__)
        except Exception as exc2:
            return degraded(type(exc2).__name__)
    post: dict[str, Any] = {"family": task.family, "shock_width": engine.width_scale if shock else None,
                            "f3_width_profile": "v2.4" if f3_v24 else None}
    m0_samples = None
    wants_mix = settings.mix_weight > 0 and _fresh_daily(task, model) and not model.audit.get("fallback")
    alpha, k = sharpen_for(settings, task.family)
    # The family pairs apply to fresh daily cards only. Transfer cards are left alone (M0's moments
    # come from a stale early window); monthly cards have their own path below.
    wants_sharpen =(settings.sharpen and (alpha > 0 or k != 1.0) and _fresh_daily(task, model)
                     and not model.audit.get("fallback"))
    fresh_monthly = (settings.monthly_m0_path and task.cadence == "monthly" and not model.audit.get("fallback")
                     and all(not model.audit.get("assets", {}).get(a, {}).get("transfer", True) for a in task.assets))
    # On a monthly card M0 counts months to a sealed target date. The observation month the card
    # names is the best public stand-in: it reproduces three of the four published step tables.
    month_steps = ({(a, int(h)): float(task.steps[ai, hi]) for ai, a in enumerate(task.assets)
                    for hi, h in enumerate(task.horizons)}
                   if task.cadence == "monthly" and settings.monthly_m0_path else None)
    transfer_card = (settings.transfer_m0 != "off" and task.cadence == "daily" and not model.audit.get("fallback")
                     and all(model.audit.get("assets", {}).get(a, {}).get("transfer", False) for a in task.assets))
    if settings.pair_match and task.cell_count == 2 or wants_mix or wants_sharpen or fresh_monthly or transfer_card:
        try:
            m0_samples, ref = m0_reference(task.root, task.card, month_steps=month_steps, asof=task.asof,
                                            overrides=task.panel_overrides)
            post["m0_reference"] = {"seed": ref["seed"], "aligned_rows": ref["aligned_rows"]}
        except Exception as exc:
            post["m0_reference"] = f"unavailable: {type(exc).__name__}: {exc}"
    extra = int(round(settings.draws * settings.mix_weight)) if wants_mix and m0_samples is not None else 0
    if fresh_monthly and m0_samples is not None:
        extra = int(round(settings.draws * settings.monthly_mix_weight))
    if settings.draws - extra < task.min_draws:
        extra = 0
    try:
        samples = simulate(task, model, replace(engine, draws=settings.draws - extra))
    except Exception as exc:
        if settings.robust_fallback:
            return degraded(type(exc).__name__)
        try:
            model = fit_base(task, engine, fallback=True)
            model.audit.update(fallback_reason=type(exc).__name__)
            samples = simulate(task, model, engine)
            extra = 0
        except Exception as exc2:
            return degraded(type(exc2).__name__)
    if extra:
        if fresh_monthly:
            df, drift, scale = settings.monthly_mix_df, settings.monthly_mix_drift, settings.monthly_mix_scale
        else:
            df, drift, scale = settings.mix_df, settings.mix_drift, settings.f4_width_scale if shock else 1.0
        mix = m0t_draws(ref, extra, generator(settings.seed, task.fingerprint, "m0t-mixture"),
                        df=df, drift=drift, scale=scale)
        samples = np.concatenate([samples, mix.reshape(extra, len(task.assets), len(task.horizons))])
        post["mixture"] = {"engine_draws": settings.draws - extra, "m0t_draws": extra, "df": df,
                           "drift": drift, "scale": scale}
    else:
        post["mixture"] = "not applied"
    # The reproduction must describe the same walk as the task before its moments or its pair
    # value are used: step counts that differ by 2x (the §3.7.5 threshold) disqualify it.
    same_walk = False
    if m0_samples is not None:
        own, theirs = np.asarray(task.steps, dtype=float).ravel(), np.asarray(ref["steps"], dtype=float)
        same_walk = bool(np.all(np.maximum(own, theirs) / np.maximum(np.minimum(own, theirs), 1.0) < 2.0))
    monthly_sharpen = fresh_monthly and m0_samples is not None and same_walk and settings.monthly_sharpen_alpha > 0
    price_reason = monthly_price_vol_reason(task, settings) if monthly_sharpen else None
    if monthly_sharpen and price_reason is None:
        # v2.7.4 (E30): a price-index card in a high-volatility regime keeps the engine's draws (alpha 0).
        post["sharpen"] = {"skipped": "price-index card, recent volatility at or above its long-run level (E30)",
                           "ratio": monthly_regime_ratio(task)}
    elif monthly_sharpen:
        samples, post["sharpen"] = m0_sharpen(samples, ref, settings.monthly_sharpen_alpha, 1.0,
                                              generator(settings.seed, task.fingerprint, "m0-sharpen"))
        if settings.monthly_price_vol_switch:
            post["sharpen"]["e30"] = price_reason
    elif not wants_sharpen:
        post["sharpen"] = "not applied"
    elif m0_samples is None:
        post["sharpen"] = "not applied: M0 reference unavailable"
    elif not same_walk:
        post["sharpen"] = "not applied: M0 steps disagree with the task"
    else:
        samples, post["sharpen"] = m0_sharpen(samples, ref, alpha, k,
                                              generator(settings.seed, task.fingerprint, "m0-sharpen"))
    if transfer_card and m0_samples is not None and same_walk:
        samples, post["transfer_m0"] = transfer_m0_draws(task, model, ref, settings, len(samples))
    elif transfer_card:
        post["transfer_m0"] = ("not applied: M0 reference unavailable" if m0_samples is None
                               else "not applied: M0 steps disagree with the task")
    # v2.7.3: on a two-cell card the profile comes before the pairing, so the pair statistic is still matched to M0's.
    if event_reason is None and task.cell_count == 2:
        samples, post["event_profile"] = apply_event_profile(task, samples)
    if settings.pair_match and task.cell_count == 2 and m0_samples is not None:
        if same_walk:
            samples, post["pair_match"] = _pair_match(task, samples, m0_samples, ref, settings.seed,
                                                     fresh=_fresh_daily(task, model) or fresh_monthly)
        else:
            post["pair_match"] = {"reached": False, "method": "skipped: M0 steps disagree with the task",
                                  "m0_steps": theirs.tolist(), "task_steps": own.tolist()}
    if settings.rates_forward_add > 0:
        samples, post["rates_forward"] = rates_forward_center(task, samples, settings)
    # Elsewhere it comes last, where E24 applied it: the forward shift is a constant per cell.
    if event_reason is None and task.cell_count != 2:
        samples, post["event_profile"] = apply_event_profile(task, samples)
    elif event_reason is not None:
        post["event_profile"] = f"not applied: {event_reason}"
    if event_reason is None and stamped.ust_family_widths_active:
        samples, post["ust_family_widths"] = keep_ust_family_widths(task, samples, stamped)
    if not np.isfinite(samples).all():
        return degraded("non-finite post-processed samples")
    model.audit["post_processing"] = post
    _stamp(model, task, stamped, requested)
    return samples, model


FILES = {"forecast.parquet", "forecast_meta.json", "forecast_rationale.md"}


def _replace(path: Path, data: bytes) -> None:
    fd, name = tempfile.mkstemp(prefix=".forecast-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            # mkstemp is mode 0600. The scorer reads these files as a different user.
            os.fchmod(f.fileno(), 0o644)
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def write_output(
    task: TaskContext,
    samples: np.ndarray,
    out: Path,
    model: NumericModel,
    seed: int,
) -> None:
    out = out.resolve()
    if out.name != "forecast.parquet":
        raise ContractError("--out must name forecast.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    extra = {p.name for p in out.parent.iterdir()} - FILES
    if extra:
        raise ContractError(f"output directory contains unrelated entries: {sorted(extra)}")
    n, a, h = samples.shape
    if (a, h) != (len(task.assets), len(task.horizons)) or n < task.min_draws or n > 20_000:
        raise ContractError("sample tensor disagrees with the target grid")
    if not np.isfinite(samples).all() or samples.size > 5_000_000:
        raise ContractError("invalid values or too many rows")
    table = pa.table(
        {
            "draw": pa.array(np.repeat(np.arange(n, dtype=np.int32), a * h)),
            "asset": pa.array(np.tile(np.repeat(task.assets, h), n)),
            "horizon": pa.array(np.tile(task.horizons, n * a), type=pa.int32()),
            "value": pa.array(samples.reshape(-1), type=pa.float64()),
        }
    )
    buffer = pa.BufferOutputStream()
    pq.write_table(
        table, buffer, compression="zstd", row_group_size=max(1, (samples.size + 31) // 32)
    )
    parquet = buffer.getvalue().to_pybytes()
    meta = {
        "unit_id": task.unit_id,
        "asof": task.asof,
        "representation": "samples",
        "asset_ids": list(task.assets),
        "horizons": list(task.horizons),
        "n_draws": n,
        "target": task.target_type,
        "units": task.units,
        "rationale": {
            "file": "forecast_rationale.md",
            "method": f"{ENGINE_VERSION} numerical engine; no text",
        },
    }
    metadata = (json.dumps(meta, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
    rationale = render_rationale(task, model, n, seed).encode()
    if (
        len(rationale) > 1024 * 1024
        or len(metadata) > 256 * 1024
        or sum(map(len, (parquet, metadata, rationale))) > 64 * 1024 * 1024
    ):
        raise ContractError("serialized artifacts exceed the official output limits")
    _replace(out.parent / "forecast_meta.json", metadata)
    _replace(out.parent / "forecast_rationale.md", rationale)
    _replace(out, parquet)


# ---------------------------------------------------------------- v2.7.6 (E37, E40b)

# E37: the mean move of a rates cell over an FOMC week, in units of the cell's forecast SD, estimated on the 2005-18 FOMC
# weeks of the official card shapes (exp/README.md, E37; output/runs/2026-10-19-e37/coefficients.json).
FOMC_MEAN_KAPPA = {"ust": -0.136134, "bills": -0.248081}
# E40/E40b: regression window, minimum overlap, minimum |corr| (whole window and each half) and spread inflation.
NOWCAST_WINDOW, NOWCAST_MIN_N, NOWCAST_MIN_CORR, NOWCAST_INFLATE = 120, 60, 0.5, 1.1
# E41: months of recent residuals, and the spread constant per kind of target, each the multiple with the lowest mean
# score on the 2010-18 tasks of its type (index: level; persistent: yoy; change: mom). A target is an index when it is
# positive and its median absolute monthly change is under NOWCAST_KIND_RATIO[0] of its median absolute value over
# the window, a persistent rate under NOWCAST_KIND_RATIO[1], else a monthly change (output/runs/2026-10-19-e41).
NOWCAST_RECENT = 24
NOWCAST_KIND_MULT = {"index": 1.2, "persistent": 1.1, "change": 0.9}
NOWCAST_KIND_RATIO = (0.01, 0.2)


def fomc_mean_shift(task: TaskContext, samples: np.ndarray) -> tuple[np.ndarray, Any]:
    """E37: on a card the FOMC-week profile reaches, each Treasury-note and bill cell shifts by kappa x its SD."""
    out = np.array(samples, dtype=np.float64, copy=True)
    applied = {}
    for ai, asset in enumerate(task.assets):
        k = FOMC_MEAN_KAPPA.get(event_asset_class(asset) or "")
        if k is not None:
            out[:, ai, :] += k * out[:, ai, :].std(axis=0)
            applied[asset] = k
    if not applied:
        return samples, "not applied: no Treasury-note or bill target"
    if not np.isfinite(out).all():
        return samples, "not applied: non-finite draws"
    return out, {"kappa": applied}


def _monthly_values(series: pd.Series) -> pd.Series:
    s = series.dropna()
    return s.groupby(s.index.to_period("M")).last().sort_index()


def _monthly_changes(s: pd.Series) -> pd.Series:
    consecutive = np.r_[False, np.diff(s.index.asi8) == 1]
    return s.diff()[consecutive].dropna()


def _stable_corr(x: np.ndarray, y: np.ndarray, corr: float) -> bool:
    """E40b: |corr| >= NOWCAST_MIN_CORR with the same sign in each half of the window."""
    half = len(x) // 2
    for part in (slice(0, half), slice(half, None)):
        if x[part].std() <= 0 or y[part].std() <= 0:
            return False
        c = float(np.corrcoef(x[part], y[part])[0, 1])
        if abs(c) < NOWCAST_MIN_CORR or np.sign(c) != np.sign(corr):
            return False
    return True


def monthly_nowcast_fit(task: TaskContext, target: str, last: pd.Period, through: pd.Period) -> dict | None:
    """E40b: the panel series with the highest stable |corr| of monthly changes against the target that already has
    every month from the target's last observation to `through`, with its OLS fit; None when there is none."""
    t = _monthly_values(task.histories[target])
    dt = _monthly_changes(t[t.index <= last])
    best = None
    for name, series in {**task.context_histories, **task.histories}.items():
        if name == target:
            continue
        ds = _monthly_changes(_monthly_values(series))
        if not set(pd.period_range(last + 1, through, freq="M")).issubset(set(ds.index)):
            continue
        both = pd.concat([dt, ds[ds.index <= last]], axis=1, join="inner").dropna().iloc[-NOWCAST_WINDOW:]
        if len(both) < NOWCAST_MIN_N:
            continue
        x, y = both.iloc[:, 1].to_numpy(), both.iloc[:, 0].to_numpy()
        if x.std() <= 0 or y.std() <= 0:
            continue
        corr = float(np.corrcoef(x, y)[0, 1])
        if abs(corr) < NOWCAST_MIN_CORR or (best and abs(corr) <= abs(best["corr"])):
            continue
        if not _stable_corr(x, y, corr):
            continue
        b, a = np.polyfit(x, y, 1)
        resid = y - (a + b * x)
        best = {"sibling": name, "corr": corr, "n": int(len(both)), "a": float(a), "b": float(b),
                "sigma": float(resid.std(ddof=2)), "xbar": float(x.mean()), "sxx": float(((x - x.mean()) ** 2).sum()),
                "s24": float(np.sqrt(np.mean(resid[-NOWCAST_RECENT:] ** 2))), "ds": ds}
    return best


def nowcast_target_kind(values: pd.Series) -> str:
    """E41: 'index' (a positive level whose monthly changes are small against it, like a price index), 'persistent'
    (a slowly moving rate, like a year-on-year rate) or 'change' (a monthly rate of change), by the median absolute
    change over the median absolute value in the last NOWCAST_WINDOW months."""
    v = values.dropna().iloc[-NOWCAST_WINDOW:]
    scale = float(v.abs().median())
    ratio = float(v.diff().dropna().abs().median()) / scale if scale > 0 else math.inf
    if bool((values.dropna() > 0).all()) and ratio < NOWCAST_KIND_RATIO[0]:
        return "index"
    return "persistent" if ratio < NOWCAST_KIND_RATIO[1] else "change"


def monthly_nowcast(task: TaskContext, samples: np.ndarray, width: bool = False) -> tuple[np.ndarray, dict]:
    """E40b: recentre and rescale each nowcast target's draws (level space) on the sibling's fitted changes. With
    `width` (v2.7.7, E41) the spread adds prediction error and recent volatility and takes its kind's constant."""
    out = np.array(samples, dtype=np.float64, copy=True)
    info = {}
    steps = np.asarray(task.steps)
    for ai, asset in enumerate(task.assets):
        t = _monthly_values(task.histories[asset])
        last = t.index[-1]
        fit = monthly_nowcast_fit(task, asset, last, last + int(steps[ai].max()))
        if fit is None:
            continue
        kind = nowcast_target_kind(t)
        cells = []
        for hi in range(len(task.horizons)):
            k = int(steps[ai, hi])
            months = pd.period_range(last + 1, last + k, freq="M")
            centre = float(t.iloc[-1]) + sum(fit["a"] + fit["b"] * float(fit["ds"][m]) for m in months)
            spread = fit["sigma"] * np.sqrt(k) * NOWCAST_INFLATE
            if width:
                big_x = float(sum(float(fit["ds"][m]) for m in months))
                lev = k + k * k / fit["n"] + (big_x - k * fit["xbar"]) ** 2 / fit["sxx"]
                blend = math.sqrt(0.5 * fit["sigma"] ** 2 + 0.5 * fit["s24"] ** 2)
                spread = blend * math.sqrt(lev) * NOWCAST_KIND_MULT[kind]
            x = out[:, ai, hi]
            sd = float(x.std())
            if not np.isfinite(sd) or sd <= 0 or not np.isfinite(spread) or spread <= 0:
                continue
            out[:, ai, hi] = centre + (x - np.median(x)) * (spread / sd)
            cells.append({"steps": k, "centre": centre, "spread": float(spread), "engine_sd": sd})
        info[asset] = {k: v for k, v in fit.items() if k != "ds"} | {"kind": kind, "width": "E41" if width else "E40b",
                                                                     "cells": cells}
    return out, info


def predict(task: TaskContext, settings: Settings) -> tuple[np.ndarray, NumericModel]:
    """v2.7.5's forecast (predict_core), then v2.7.6's E37 and E40b steps (E40b's width per v2.7.7's E41). Each never
    raises."""
    samples, model = predict_core(task, settings)
    post = model.audit.setdefault("post_processing", {})
    # E37 acts where the FOMC-week profile acts: the profile switched on and the card one it reaches (the experiments
    # ran with the profile on, so this is their arm exactly).
    if (settings.fomc_mean_shift and settings.event_profile and task.cadence == "daily"
            and event_profile_reason(task) is None):
        samples, post["fomc_mean"] = fomc_mean_shift(task, samples)
    if settings.monthly_nowcast and task.cadence == "monthly":
        try:
            new, info = monthly_nowcast(task, samples, settings.nowcast_width)
            if np.isfinite(new).all():
                samples = new
            else:
                info = "not applied: non-finite draws"
        except Exception as exc:  # v2.7.5's draws stand; the reason is recorded
            info = f"not applied: {type(exc).__name__}: {exc}"
        post["nowcast"] = info
    return samples, model


def render_rationale(task: TaskContext, model: NumericModel, draws: int, seed: int) -> str:
    lines = [
        f"# {ENGINE_VERSION} forecast — {task.unit_id}",
        f"As of {task.asof}; {draws} draws; seed {seed}.",
        "",
        "1. Inputs and anchors. Only supplied panel observations on or before the cutoff enter the fit. "
        "The source dates and last observed values are listed below. No text is read and no model API is called.",
        "2. Center. The daily trailing-300 mean is shrunk and clipped once using the configured reference "
        "horizon, then accumulated over each duration. Monthly slope and deceleration use separate rules. "
        "These are numerical estimates; no document-backed directional adjustment is asserted.",
        "3. Scale. Base horizon volatility receives the empirical horizon correction and configured final "
        "width scaling. Daily scaling is width_scale * (steps / horizon_scale_anchor)**horizon_scale_power. "
        "Monthly scaling uses monthly_width_scale without the daily power.",
        "4. Shape and dependence. Shared multivariate Student-t / stationary-bootstrap paths carry "
        "cross-asset and cross-horizon dependence, with the recorded volatility overlay. "
        "Log-level states are exponentiated only after the final affine transformation.",
        "5. Post-processing (audit key `post_processing`). On daily cards with fresh histories part of the draws "
        "come from the published M0 mean/covariance with Student-t shocks and a damped drift. "
        "Shock-family (T2-F4) daily cards use a wider width scale. On two-cell grids, when `pair_match` is on "
        "(off by default from v2.7.3), the single variogram pair is matched to M0's value when M0's step counts "
        "agree with the task's; "
        "`pair_match.method` records whether the draws were only re-paired, their centres moved, "
        "part of them replaced by M0 draws, or nothing was changed. "
        "Before that pairing, fresh daily draws are resampled with weight 1 / CRPS_M0^alpha and each "
        "cell is scaled by k about its median, with (alpha, k) set per card family; "
        "`post_processing.sharpen` records the pair used or why nothing was changed. "
        "Fresh monthly cards take their own M0-t share and sharpening, and their M0 walks the card's "
        "observation months. "
        "Cards whose every target is a transfer asset (an early window and an as-of anchor only) take "
        "Gaussian draws on the published M0 mean/covariance; `post_processing.transfer_m0` records the rule "
        "and each cell's spread against M0's. "
        "Last, each Treasury cell (a UST tenor of the rates panel) is shifted by half of the forward increment "
        "the as-of curve implies for its tenor at its target date (expectations hypothesis, zero-coupon "
        "approximation, flat below 2Y); `post_processing.rates_forward` lists each cell's spot, forward and shift. "
        "On a card whose as-of is the Monday of an FOMC week (the decision on the Wednesday), with every step in "
        "1-4 and every target a Treasury tenor, a bill or one of the 15 exchange rates of the ranked Final, the "
        "family widths above are replaced by a step profile: k 1.0 for every family, F4 at the default width, "
        "and each step's variance multiplied by a fixed factor per class (calm on the Tuesday; the decision "
        "reaches the Treasury close on the Wednesday and the ECB fixing on the Thursday), applied to each asset's "
        "path in yield or log-FX units before the pairing on two-cell cards and last elsewhere; "
        "`post_processing.event_profile` records the factors or why it did not apply. "
        "The ledger below describes the engine component only.",
        "",
        "## Input evidence",
        "",
        "| Asset | First input date | Last input date | Last observed value | Transform |",
        "|---|---|---|---:|---|",
    ]
    def label(value: str) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")
    for ai, asset in enumerate(task.assets):
        history = task.histories.get(asset)
        if history is None or history.empty:
            lines.append(f"| {label(asset)} | unavailable | unavailable | unavailable | {model.transforms[ai].kind} |")
            continue
        lines.append(f"| {label(asset)} | {history.index[0].date()} | {history.index[-1].date()} | "
                     f"{float(history.iloc[-1]):.12g} | {model.transforms[ai].kind} |")
    if model.audit.get("engine") == "wide-fallback":
        lines.extend(["", "The fitted engine could not run. A generic finite fallback grid was written. "
                      "No fitted center/scale ledger is asserted for that grid; the error is in the audit."])
    else:
        lines.extend([
            "", "## Adjustment ledger", "",
            "Centers and spreads below are in forecast-state units: native units for levels, "
            "log units for FX log levels, and cumulative log-return units for return targets. "
            "Center = state anchor + trend + reversion + guard adjustment. "
            "Final SD = base SD * total width multiplier; this is not an interval-coverage claim.", "",
            "| Asset | Horizon key | Steps | State anchor | Trend | Reversion | Guard adjustment | Center | Base SD | Width multiplier | Final SD |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for ai, asset in enumerate(task.assets):
            for hi, horizon in enumerate(task.horizons):
                center, base, spread = model.center[ai, hi], model.base_spread[ai, hi], model.spread[ai, hi]
                adjustment = center - model.spots[ai] - model.trend[ai, hi] - model.reversion[ai, hi]
                multiplier = spread / base if base > 0 else 0.0
                values = [model.spots[ai], model.trend[ai, hi], model.reversion[ai, hi],
                          adjustment, center, base, multiplier, spread]
                lines.append(f"| {label(asset)} | {horizon} | {int(task.steps[ai, hi])} | "
                             + " | ".join(f"{float(value):.12g}" for value in values) + " |")
    lines.extend([
        "", "## Limitations and effective settings", "",
        "Global defaults, including the v1.1 post-processing and v2.1-v2.3 sharpening settings, were selected on "
        "public development data whose latest outcome is dated 2024-12-17; the v2.4 transfer rule on "
        "transfer-shaped tasks whose outcomes end before 2014-12-18; the v2.7.3 FOMC-week profile is a static "
        "table estimated on 2005-2018 FOMC weeks from panels that end by 2024-12-18, with the Federal Reserve's "
        "published 28 October 2026 decision added to its calendar. No task answer is stored. "
        "Point-in-time eligibility is claimed only for cutoffs after that date. "
        "Historical inputs are fixed-vintage; interval coverage is not guaranteed.",
        "", "```json", json.dumps(model.audit, ensure_ascii=False, indent=2, allow_nan=False), "```",
        *["- " + warning for warning in task.warnings], "",
    ])
    return "\n".join(lines)


def validate_output(unit_dir: Path, output_dir: Path) -> dict[str, bool]:
    ctx = {"unit_dir": unit_dir, "output_dir": output_dir}
    verifier = build_verifier(ctx)
    verdict = verifier.run(ctx)
    gates = {name: result.passed for name, result in verdict.gate_results.items()}
    if not verdict.admissible:
        failed = [name for name, passed in gates.items() if not passed]
        raise ContractError(f"official output gates refused: {failed}; detail={verdict.detail}")
    if sum(p.stat().st_size for p in output_dir.iterdir()) > 64 * 1024 * 1024:
        raise ContractError("complete output tree exceeds 64 MiB")
    return gates


def run_task(task: TaskContext, out: Path, settings: Settings) -> dict:
    started = monotonic()
    samples, model = predict(task, settings)
    write_output(task, samples, out, model, settings.seed)
    gates: dict[str, bool] = {}
    gate_error = None
    try:
        gates = validate_output(task.root, out.parent.resolve())
    except Exception as exc:
        # Files are already the submission. A local self-check must not turn a
        # written card into a non-zero exit.
        gate_error = f"{type(exc).__name__}: {exc}"
        gates = {"self_check": False}
    result = {
        "unit_id": task.unit_id,
        "version": ENGINE_VERSION,
        "n_draws": len(samples),
        "seed": settings.seed,
        "numeric_fallback": bool(model.audit.get("fallback")),
        "elapsed_seconds": monotonic() - started,
        "gates": gates,
        "output": str(out.resolve()),
    }
    if gate_error:
        result["gate_error"] = gate_error
    return result


def files_ready(out: Path) -> bool:
    parent = out.resolve().parent
    return all((parent / name).is_file() and (parent / name).stat().st_size > 0 for name in FILES)


def _lenient_histories(root: Path, assets: tuple[str, ...], asof: str) -> dict[str, pd.Series]:
    """Best-effort target histories for the emergency path; unreadable rows are skipped."""
    out: dict[str, pd.Series] = {}
    for path in sorted((root / "panels").glob("*.parquet")) or sorted(root.glob("*.parquet")):
        try:
            frame = pd.read_parquet(path)
        except Exception:
            continue
        key = "asset" if "asset" in frame else "asset_id" if "asset_id" in frame else None
        if key is None or not {"date", "value"} <= set(frame.columns):
            continue
        for asset in assets:
            if asset in out:
                continue
            rows = frame.loc[frame[key].astype(str) == asset]
            dates = pd.to_datetime(rows["date"], errors="coerce", utc=True).dt.tz_localize(None)
            series = pd.Series(pd.to_numeric(rows["value"], errors="coerce").to_numpy(), index=pd.DatetimeIndex(dates))
            series = series[series.index.notna() & (series.index <= pd.Timestamp(asof))].dropna().sort_index()
            series = series[~series.index.duplicated(keep="first")]
            if len(series):
                out[asset] = series.astype(float)
    return out


def emergency_write(panels: Path, asof: str | None, out: Path, draws: int, seed: int, reason: str) -> dict:
    """Write a forecast when the normal loader cannot build a task.

    M0's distribution first, then a robust random walk on whatever history is readable;
    the zero-centred grid remains only for a card with no usable panel at all.
    """
    root = _root(panels)
    card = tomllib.loads((root / "card.toml").read_bytes().decode())
    targets = card["targets"]
    assets = tuple(str(a) for a in targets["asset_ids"])
    horizons = tuple(int(h) for h in targets["horizons"])
    cutoff = str(declared_asof(card) or "")[:10]
    asof = cutoff or asof or "1970-01-01"
    target = str(targets.get("target_type", "level"))
    n = min(20_000, max(int(draws), 200))
    rng = np.random.default_rng(seed)
    task = TaskContext(
        root=root,
        unit_id=str(card["task"]["id"]),
        asof=asof,
        assets=assets,
        horizons=horizons,
        target_type=target if target in {"level", "yield", "log_return"} else "level",
        units=str(targets.get("value_unit", "native panel units")),
        family=str(card.get("metadata", {}).get("category", "unknown")),
        card=card,
        spec={},
        histories={a: pd.Series(dtype=float) for a in assets},
        context_histories={},
        panel_for_asset={a: "emergency" for a in assets},
        cadence="daily",
        steps=np.broadcast_to(np.asarray(horizons, dtype=np.int64), (len(assets), len(horizons))).copy(),
        anchors=np.zeros(len(assets), dtype=np.int64),
        min_draws=200,
        fingerprint="emergency",
        warnings=(f"emergency write: {reason}",),
    )
    model = _shell_model(task, reason)
    try:
        m0_samples, ref = m0_reference(root, card, asof=asof)
        samples = m0t_draws(ref, n, rng, df=None, drift=1.0, scale=1.0).reshape(n, len(assets), len(horizons))
        if len(assets) * len(horizons) == 2:
            samples, model.audit["pair_match"] = _pair_match(task, samples, m0_samples, ref, seed, fresh=True)
        model.audit["engine"] = "m0-emergency"
    except Exception as exc:
        model.audit["m0_error"] = f"{type(exc).__name__}: {exc}"
        try:
            found = _lenient_histories(root, assets, asof)
            task = replace(task, histories={a: found.get(a, pd.Series(dtype=float)) for a in assets})
            samples = random_walk_samples(task, n, rng)
            model.audit["engine"] = "random-walk-emergency"
        except Exception as exc2:
            model.audit["random_walk_error"] = f"{type(exc2).__name__}: {exc2}"
            samples = rng.standard_t(5.0, size=(n, len(assets), len(horizons))) * 2.0
    samples = np.nan_to_num(samples)
    model.audit.update(version=ENGINE_VERSION, settings={}, requested_settings={})
    write_output(task, samples, out, model, seed)
    return {
        "unit_id": task.unit_id,
        "version": ENGINE_VERSION,
        "status": "degraded",
        "n_draws": n,
        "seed": seed,
        "numeric_fallback": True,
        "reason": reason,
        "output": str(out.resolve()),
    }


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "forecast":
        args.pop(0)
    parser = argparse.ArgumentParser(description=f"{ENGINE_VERSION} numerical forecaster")
    parser.add_argument("--panels", type=Path, required=True)
    parser.add_argument(
        "--text", type=Path, help="Accepted for the official interface; no text is read"
    )
    parser.add_argument("--asof")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=4096)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--profile",
                        choices=["default", "prospective", "balanced", "dev-leaning", "random-leaning", "v2.7.6", "v2.7.5", "v2.7.4", "v2.7.3",
                                 "v2.7.2", "v2.7.1", "v2.7", "v2.6.1", "v2.6", "v2.5", "v2.4", "v2.3", "v2.2", "v2.1",
                                 "v1.2", "v1-control"],
                        default="default",
                        help="default is the final profile; v2.7.6, v2.7.5, v2.7.4, v2.7.3 and v2.7.2 reproduce those releases; v2.7.1/v2.7/"
                             "v2.6.1/v2.6/v2.5/v2.4/v2.3/v2.2/v2.1/v1.2 reproduce theirs on daily and monthly cards")
    parser.add_argument("--settings", default="{}", help="JSON object of validated numerical overrides")
    opts = parser.parse_args(args)
    try:
        seed = opts.seed if opts.seed is not None else int(os.environ.get("QFBENCH_SEED", "0"))
        if opts.profile == "v1-control":
            base = v1_settings(draws=opts.draws, seed=seed)
        elif opts.profile == "v1.2":
            base = Settings(draws=opts.draws, seed=seed, **{**V2_6_SETTINGS, **V2_5_SETTINGS, **V2_4_SETTINGS, "sharpen": False,
                                                            "monthly_m0_path": False, "transfer_m0": "off"})
        elif opts.profile == "v2.1":
            base = Settings(draws=opts.draws, seed=seed, **{**V2_6_SETTINGS, **V2_5_SETTINGS, **V2_4_SETTINGS,
                                                            "monthly_m0_path": False, "sharpen_f4": (1.0, 1.1), "transfer_m0": "off"})
        elif opts.profile == "v2.2":
            base = Settings(draws=opts.draws, seed=seed, **{**V2_6_SETTINGS, **V2_5_SETTINGS, **V2_4_SETTINGS,
                                                            "sharpen_f4": (1.0, 1.1), "transfer_m0": "off"})
        elif opts.profile == "v2.3":
            base = Settings(draws=opts.draws, seed=seed, **{**V2_6_SETTINGS, **V2_5_SETTINGS, **V2_4_SETTINGS, "transfer_m0": "off"})
        elif opts.profile == "v2.4":
            base = Settings(draws=opts.draws, seed=seed, **{**V2_6_SETTINGS, **V2_5_SETTINGS, **V2_4_SETTINGS})
        elif opts.profile == "v2.5":
            base = Settings(draws=opts.draws, seed=seed, **{**V2_6_SETTINGS, **V2_5_SETTINGS})
        elif opts.profile == "v2.7.6":
            base = Settings(draws=opts.draws, seed=seed, **V2_7_6_SETTINGS)
        elif opts.profile == "v2.7.5":
            base = Settings(draws=opts.draws, seed=seed, **V2_7_5_SETTINGS)
        elif opts.profile == "v2.7.4":
            base = Settings(draws=opts.draws, seed=seed, **V2_7_4_SETTINGS)
        elif opts.profile == "v2.7.3":
            base = Settings(draws=opts.draws, seed=seed, **V2_7_3_SETTINGS)
        elif opts.profile == "v2.7.2":
            base = Settings(draws=opts.draws, seed=seed, **V2_7_2_SETTINGS)
        elif opts.profile == "v2.7.1":
            base = Settings(draws=opts.draws, seed=seed, **V2_7_1_SETTINGS)
        elif opts.profile == "v2.7":
            base = Settings(draws=opts.draws, seed=seed, **V2_7_SETTINGS)
        elif opts.profile == "v2.6.1":
            base = Settings(draws=opts.draws, seed=seed, **V2_6_1_SETTINGS)
        elif opts.profile == "v2.6":
            base = Settings(draws=opts.draws, seed=seed, **V2_6_SETTINGS)
        elif opts.profile in SHARPEN_PROFILES:
            base = Settings(draws=opts.draws, seed=seed, **SHARPEN_PROFILES[opts.profile])
        else:
            base = Settings(draws=opts.draws, seed=seed)
        overrides = json.loads(opts.settings)
        if not isinstance(overrides, dict) or {"draws", "seed"} & overrides.keys():
            raise ValueError("--settings must be an object; use --draws and --seed for those controls")
        unknown = set(overrides) - set(asdict(base))
        if unknown:
            raise ValueError(f"unknown settings: {sorted(unknown)}")
        settings = replace(base, **overrides)
        task = load_task(opts.panels, opts.asof)
        result = run_task(task, opts.out.resolve(), settings)
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
        return 0
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}" if isinstance(exc, (ValueError, FileNotFoundError, ContractError)) else type(exc).__name__
        try:
            seed = opts.seed if opts.seed is not None else int(os.environ.get("QFBENCH_SEED", "0"))
            result = emergency_write(opts.panels, opts.asof, opts.out.resolve(), opts.draws, seed, reason)
            print(json.dumps(result, ensure_ascii=False, allow_nan=False))
            return 0 if files_ready(opts.out) else 1
        except Exception as final:
            print(
                json.dumps(
                    {
                        "status": "failed",
                        "error_type": type(final).__name__,
                        "message": str(final) if isinstance(final, (ValueError, FileNotFoundError, ContractError)) else "forecast failed",
                    }
                ),
                file=sys.stderr,
            )
            ready = False
            try:
                ready = files_ready(opts.out)
            except Exception:
                ready = False
            return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
