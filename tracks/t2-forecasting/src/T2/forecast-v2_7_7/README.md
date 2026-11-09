# forecast-v2_7_7

Numerical forecaster v2.7.7 for Agenthon 2026 Track 2, a Final candidate. This directory is the Docker build context.
It is `forecast-v2_7_6` (v2.7.6) with one change from `../exp/README.md` (E41): the width of v2.7.6's monthly sibling
nowcast (E40b). Its centre, its sibling choice and the draws' shape stay.

| Change | v2.7.6 | v2.7.7 | Cards it reaches |
|---|---|---|---|
| Nowcast width (`nowcast_width`, E41) | residual SD σ × √k × 1.1 | √(σ²/2 + s24²/2) × √(k + k²/n + (X − k·x̄)²/Sxx) × c. That is recent residual volatility (the window's last 24 months) shrunk halfway to the window's, the OLS prediction variance of the k-month sum, and a constant per kind of target (index level 1.2, persistent rate 1.1, monthly change 0.9; fitted on 2010–18) | cards where E40b fires (a monthly target whose panel holds a stable sibling that already has its target month); no practice card |

`--profile v2.7.6` (`Settings(**V2_7_6_SETTINGS)`) reproduces v2.7.6 on every card, and every earlier profile
inherits it. The default equals v2.7.6 on all 126 public cards.

## Why (v2.7.7)

- **An outside review** measured E40b's standardized errors: SD 1.04 at train, 1.50 at test (2019–24) and 0.91 in
  2024. 90 % coverage at test was 0.766.
- **E41**, pre-registered, on E40b's 825 tasks where it fires (seeds 0–2), judged on score.
  - The primary arm (`full`), at test: −0.0362 [−0.0664, −0.0110]. The mean goes 0.785 → 0.748, and 90 % coverage
    0.766 → 0.841.
  - Train −0.0053, covid −0.085. By type at test: level −0.043, yoy −0.059, mom +0.000.
  - It costs a little where the width was already loose: 2024 +0.005, 2018 +0.010.
  - Secondary arms: `pred_recent` −0.0365, `recent` −0.0321 and `pred` −0.0069 pass. `typecal`, a constant per type
    alone, fails at +0.0235.
- **The target kind** is read from the series itself: the median absolute monthly change over the median absolute
  value, last 120 months.
  - Under 0.01, with every value positive: an index. Under 0.2: a persistent rate. Otherwise: a monthly change.
  - On all 2,309 price and payroll tasks of the roster it equals the task's type.

## Checks (sub/verification-v2_7_7/, 2026-10-19)

| Check | Result |
|---|---|
| Tests | 149 source, 14 packaging, 26 contract: OK. Fuzz 70 cards: clean |
| Image `agenthon-v2_7_7:2.7.7` (`sha256:49d6bbf2422e…`) | 126/126 public cards in the constrained container, 0 failures, 0 fallbacks, host within 2.9e-11 |
| Identity | Equals E41's `full` arm on all 2,475 forecasts where E40b fires. The target kind equals the type on every nowcast40 price and payroll task. The default and `--profile v2.7.6` equal v2.7.6 on all 126 public cards |
| 40 Final-shaped cards | 39 clean; the audit's no-as-of card is refused as before. E40b, with E41's width, on the 2 PCE-with-CPI cards (spreads 21 % and 18 % wider, same centres). Everything else equal to v2.7.6 |
| Expected scores | Development about 2.46. Ranked Final: daily cards as v2.7.6 (1.073 per card on the official shapes); a PCE card with that month's CPI in its panel about 0.748 (v2.7.6 0.785, M0 1.503) |

Published 2026-10-21 as `docker.io/greedyshark/forecast-v3:2.7.7` (linux/amd64 manifest `sha256:95ab15eb…`, anonymous pull checked). The Final ZIP is packed and not yet uploaded; see `../sub/PIPELINE.md`.

## v2.7.6, carried unchanged

v2.7.6 is `forecast-v2_7_5` (v2.7.5) with two changes from `../exp/README.md` (E37 and E40b). Both run after v2.7.5's
forecast (`predict_core`, then `predict`):

| Change | v2.7.5 | v2.7.6 | Cards it reaches |
|---|---|---|---|
| FOMC-week rates centre (`fomc_mean_shift`, E37) | rates cells centred at the engine's median | Treasury-note and bill cells shift by κ × their SD (κ: notes −0.136, bills −0.248; estimated on 2005–18 FOMC weeks) | cards the FOMC-week profile reaches (the profile on, an FOMC-week Monday, steps 1–4, ranked series), the ranked week among them; never FX |
| Monthly sibling nowcast (`monthly_nowcast`, E40b) | context series unused on monthly cards | a monthly target whose panel holds a series that already has the target month is nowcast from it. That series must have a stable |corr| ≥ 0.5 of monthly changes over 120 months and in each half. The OLS fit sets the centre, and the residual SD × 1.1 the spread, in level space | data-release cards whose panel carries such a sibling (for example PCE with that month's CPI); no practice card |

`--profile v2.7.5` (`Settings(**V2_7_5_SETTINGS)`) reproduces v2.7.5 on every card, and every earlier profile
inherits it. Among the 126 public cards the default changes only the four FOMC-week rates cards of the
short-horizon practice set.

### Why (v2.7.6)

- **E37**, pre-registered, on the official card shapes over every FOMC week (`ps1shape`). κ was estimated on 2005–18
  only.
  - Rates tasks at test (2019–24): −0.0411 [−0.0727, −0.0105], every class better (notes −0.032, bills −0.047,
    mixed −0.070).
  - Train −0.012.
  - The same shift on non-FOMC Mondays does nothing (+0.007 [−0.031, +0.045]), so the effect belongs to FOMC
    weeks.
  - It is a regime bet: it costs in rising-yield FOMC weeks (2013 +0.025, 2017 +0.023, 2021 +0.015).
- **E40b**, pre-registered after E40 failed its guard. E40 had worked in log space, which fails on a rate's negative
  draws, and had picked a covid-made sibling (UNRATE).
  - On real-time ALFRED panels with first-release targets, where it fires (PCE with CPI): test −0.495
    [−0.759, −0.268] (mean 1.28 → 0.78; M0 1.50), train −0.444.
  - Every type and every year is better. It never fires on CPI, labour, or a task without such a sibling.
- E38 (two-cell increments) and E39 (rates tails) failed their rules and are not taken.

### Checks (sub/verification-v2_7_6/, 2026-10-19)

| Check | Result |
|---|---|
| Tests | 144 source, 14 packaging, 26 contract: OK. Fuzz 70 cards: clean |
| Image `agenthon-v2_7_6:2.7.6` (`sha256:da81b4e8ca4f…`) | 126/126 public cards in the constrained container, 0 failures, 0 fallbacks, host within 2.9e-11 |
| Identity | `--profile v2.7.5` equals v2.7.5 on all 126 cards. The default differs on the four FOMC-week rates practice cards only. v2.7.6 equals the E37 and E40b arms on 80 sampled tasks |
| 40 Final-shaped cards | 39 clean; the audit's no-as-of card is refused as before. E37 on the 6 FOMC-week Treasury cards, E40b on the 2 PCE-with-CPI cards, everything else equal to v2.7.5 |
| Expected scores | Development about 2.46. Ranked Final on the official shapes about 1.073 per card (M0 1.156) |


## v2.7.5, carried unchanged

### What the seasonal step does

- **Detection** (`seasonal_effects`). The step reads the target's own history: log changes if every value is
  positive, else differences, over consecutive months.
  - With fewer than 96 changes, nothing is done.
  - Otherwise a Kruskal–Wallis test across the 12 calendar months runs on the last 96 changes. The step fires at
    p < 0.001.
- **Effects.** Each calendar month's effect is the mean of its 8 changes without the highest and the lowest,
  centred to sum to zero over the year. S(c) is their running sum from January.
- **Adjustment** (`predict_seasonal`). The history becomes x·exp(−(S(c_t) − S(c_L))), or x − (S(c_t) − S(c_L)),
  anchored at the last month L.
  - Every other step of v2.7.4 then runs on it unchanged: the engine, the M0-t mixture, the reweighting and E30's
    switch. The embedded M0 reads the adjusted series through an in-memory override (`panel_overrides`), and the
    seed fingerprint stays the card's own.
  - The draws at a target month T are then multiplied by exp(S(c_T) − S(c_L)), or shifted by it
    (`seasonal_offset`).
- **Safety.** Anything unexpected falls back to v2.7.4's forecast, with the reason in the audit (`seasonal`). No file
  is written.

## Why

The ranked Final has "a few" cards that forecast the next figure of a monthly official statistic first released on
27–30 October. Many such statistics are published without seasonal adjustment, among them the euro-area and
national flash HICP/CPI and Tokyo CPI. On those, a random walk, M0's included, misses the calendar swing.

| Experiment (real-time ALFRED vintages to 2024-12-18, first-release targets) | Test, from 2019 (95%) | Train, to 2018 | Rules |
|---|---|---|---|
| E35, 16 NSA series (US, euro area, DE/FR/IT/ES, UK, CA, JP) | −0.199 [−0.339, −0.066] (2,100 tasks) | −0.346 | PP-P: DE +0.098, JP +0.024 |
| E36 holdout, 20 series E35 never used (11 HICPs, CH/NO/KR/MX CPI, US gasoline and food CPI, US vehicle sales, employment and labour force) | −0.188 [−0.271, −0.112] (3,251 tasks) | −0.190 | PPPP: 19 of 20 series gain; worst FI +0.017 |

- **Guard.** The step fires on no seasonally adjusted backtest task: 0 of 2,488 in `realtime33` and 0 of 2,750 in
  `monthly26`. It never fires on a year-on-year rate.
- **By type** (E36, test): level −0.283, month-on-month −0.265, year-on-year 0.
- **Risk.** A statistic hit by a calendar-specific policy change loses: Germany in January 2021, when the 2020 VAT
  cut was reversed. So does one with weak seasonality (Japan). Covid targets were about neutral in E36 (+0.004) and
  worse in E35 (+0.16).
- **Experiments that left v2.7.4 as it was.**
  - E33: E30 on first-release targets, test −0.104, not confirmed (train +0.006).
  - E34: a wider M0-t mixture (`monthly_mix_scale` 1.0) failed.

## Checks (sub/verification-v2_7_5/, 2026-10-18)

| Check | Result |
|---|---|
| Tests | 137 source, 14 packaging, 26 contract: OK |
| Fuzz (70 cards, seed 5) | 0 not ok, 0 fallbacks |
| Image `agenthon-v2_7_5:2.7.5` (`sha256:f375a4d0e2b2…`) | All 126/126 public cards in the constrained container, including the 22 short-horizon practice units (2026-10-19): 0 failures, 0 fallbacks, host within 2.9e-11 |
| Identity | Equal to v2.7.4 on all 126 public cards. Equal to the E35 research copy on 60 sampled NSA tasks |
| 38 Final-shaped cards (as-of 2026-10-26) | 37 clean. The audit's no-as-of card is refused by the verifier, as before. The step fires on the 5 NSA cards only |
| Expected scores | Development about 2.46 (as v2.7.4). Ranked Final about 1.05 per card (M0 about 1.1; one week 0.65–1.65). Official short-horizon practice set: 1.165 against M0's 1.142 (1.040 against 1.061 without the 2016 election-week BRL card). Official card shapes on every FOMC week of 2005–24: 1.083 against M0's 1.156, ahead in 79 % of weeks. See `../sub/PIPELINE.md` |

No image has been pushed, and nothing has been packed or uploaded.

## v2.7.4, carried unchanged

It is `forecast-v2_7_3` (v2.7.3) with the follow-ups to the review of 17 October:

| Change | v2.7.3 | v2.7.4 | Cards it reaches |
|---|---|---|---|
| Day counts on non-monthly panels (`epoch_ns`) | `asi8 / 86400e9`, nanoseconds assumed | the index converted to nanoseconds first | none under the pinned pandas 2.3.3. Under pandas 3, where dates can be stored in microseconds, the weekly repair and the seed digest were silently wrong. |
| UST family widths away from the zero bound (`ust_family_widths_active`, E29) | the profile resets every family width | UST-notes cells keep their family widths under the profile when the card's lowest UST yield is at least 1% (`ust_active_floor`) | FOMC-week cards with a UST note away from the zero lower bound, the ranked set among them (bills about 4%) |
| Price-index monthly switch (`monthly_price_vol_switch`, E30, post hoc) | M0-centred reweighting (alpha 0.5) on every fresh monthly card | skipped on a card whose every target reads as a price index and whose recent/long volatility ratio is at least 1; **on**, by the user's call of 18 October (post hoc); on first-release targets (E33) test −0.104, train +0.006 | monthly price-index cards in a high-volatility regime |
| M0 step count at a ratio of exactly 2 (`_m0_step_count`) | the print count (`>= 2`) | the declared key (`> 2`, official commit 24582e9) | weekly-type panels whose key is exactly twice the print count |

`--profile v2.7.3` (`Settings(**V2_7_3_SETTINGS)`) reproduces v2.7.3 draw for draw on all 104 public cards, as does
the default except on the two CPI cards where E30 fires (`t2-F1-cpi-glidepath-2023`, `t2-F4-cpi-vintage-2022`,
recent/long volatility ratios 1.12 and 1.17; there the draws equal v2.7.3 with `monthly_sharpen_alpha` 0). No public
card has a Monday FOMC-week as-of, and none has a weekly panel.

Under the pinned pandas 2.3.3, v2.7.4 draws the same values as v2.7.3 on all 104 public cards, with the same card
fingerprints. Under pandas 3.0.5 it draws the same values as under 2.3.3; v2.7.3 did not. The reproduction tests
against earlier releases need the pinned pandas, because those releases' own seed digests change under pandas 3.

### Known behaviour, kept: cross-class correlation under the FOMC-week profile

The profile scales the increments between the horizons a card requests.
- Within one asset, or between assets of one class, the marginals and cross-horizon covariances do not depend on
  which horizons are requested.
- Between classes with different step profiles, the correlation at a horizon does. With independent steps, a card
  that requests every step 1..3 gets UST notes against G10 at h = 3 with correlation 0.941 ρ. A card that requests
  h = 3 alone keeps ρ. The marginals agree either way.
- E24's evidence was produced with this transform.
- In the ranked window the ECB fixing takes the FOMC decision one step after the Treasury close. A per-step model
  with a contemporaneous correlation does not describe that lead-lag either, so neither request is known to be
  closer.
- `FomcWeekProfile.test_cross_class_correlation_depends_on_the_requested_horizons` pins the behaviour.

## v2.7.3, carried unchanged

v2.7.3 is `forecast-v2_7_2` (v2.7.2) with two changes from the experiments in `../exp/README.md`: E24's FOMC-week
variance profile (arm `ev`) built into the forecaster, and E25's pairing default:

| Change | v2.7.2 | v2.7.3 | Cards it reaches |
|---|---|---|---|
| FOMC-week profile (`event_profile`) | none | the family widths reset to k 1.0 (F4 at the default width), then E24's step profile applied to each asset's path | daily level cards whose as-of is the Monday of an FOMC week (decision on the Wednesday), with every step in 1–4 and every target a UST tenor, a bill or one of the ranked Final's 15 exchange rates |
| Pairing on two-cell cards (`pair_match`) | on: the single variogram pair matched to M0's 500-draw value | off (a default flipped by E25; no new data) | cards with exactly two cells (19 of the 104 public cards) |

The ranked set is such a card set: as-of Monday 2026-10-26, FOMC decision Wednesday 2026-10-28, targets 27–30 October.
Every other card is v2.7.2 bit for bit except cards with exactly two cells, which are no longer pair-matched. No text
is read and no model API is called. `--profile v2.7.2` (`Settings(**V2_7_2_SETTINGS)`: `event_profile` False,
`pair_match` True) reproduces v2.7.2 on every card; `v2.7.1` and the earlier profiles carry both and reproduce their
releases as before.

## What it does

- **Trigger** (`event_profile_reason`). The card must meet all of these:
  - daily, `level`;
  - as-of a Monday whose Wednesday is in `FOMC_WEDNESDAYS`;
  - every step in 1–4;
  - every target in a profile class;
  - every target's history within 5 days of the as-of.
- **Asset classes** (`event_asset_class`). Asset ids on the ranked cards are not published, so names are read
  broadly:
  - `UST_<n>M`, and `UST_<n>Y` below 2 years, are bills; `UST_<n>Y` from 2 years are notes.
  - The ten G10 codes are G10, and MXN, BRL, ZAR, KRW and INR are EM. Each counts as a bare code or as an
    unambiguous pair with USD (`USD_MXN`, `USDMXN`, `MXN/USD`, `EURUSD`, `EUR_USD`), ignoring case and separators.
  - Any other or ambiguous name (`EURJPY`, `USDCNY`, `DGS10`, `UST10Y`) means no profile, so the card is v2.7.2.
- **Reset** (`FAMILY_NEUTRAL`). k is 1.0 for every family and F4 runs at the default engine width. The
  family label does not change the forecast. The audit's `settings` keep the run's own values;
  `post_processing.event_profile` records what was applied.
- **Profile** (`EVENT_PROFILE`, `apply_event_profile`). The table holds variance multipliers per step from the
  Monday as-of, estimated on 2005–2018 FOMC weeks only:

  | Class | Tue | Wed | Thu | Fri |
  |---|---:|---:|---:|---:|
  | UST notes | 0.638 | 1.649 | 1.091 | 1.117 |
  | bills | 0.779 | 1.623 | 1.434 | 1.015 |
  | G10 FX | 0.834 | 0.757 | 2.5 | 1.022 |
  | EM FX | 0.921 | 0.654 | 2.5 | 1.086 |

  - Every class is calm on the Tuesday. The decision reaches the Treasury close on the Wednesday and the ECB
    fixing (13:10 UTC, before the 18:00 UTC statement) on the Thursday.
  - Per asset, in yield units for Treasuries and log units for FX (so the quote direction does not matter), the
    increments of each cell's deviation from its median are scaled by √(mean m over the steps they span),
    cumulated back and re-centred on each cell's median.
  - A single cell at h is scaled by √(mean of m over s ≤ h).
- **Order.** On two-cell cards the profile comes before pair matching, so when pairing is on (`--profile v2.7.2`,
  or `pair_match` true) the pair statistic is still M0's. Elsewhere it comes last, after the forward increment,
  which is where E24 applied it and keeps the draws identical to the E24 wrapper.
- **Pairing off** (`pair_match` False, E25). A two-cell card keeps the engine's own dependence instead of having its
  one variogram pair matched to M0's 500-draw value (by re-pairing, shifting centres or blending in M0 draws). The
  matching was built for the rule before 6 October, when the joint divisor was M0's realised pair value. The failure
  paths (`m0_fallback`, `emergency_write`) still pair a two-cell card's M0 draws, as before.
- **Safety.** If anything goes wrong in the profile (a non-positive FX draw, a non-finite value), the draws are
  left unchanged and the reason is recorded. The profile never raises.

## Why

E24 tested this on the `fomcweek` roster (6,233 tasks: the Monday as-of of every Wednesday FOMC week, 2005–2024,
h = 1–4, UST with bills and ECB FX). It was pre-registered, and `ev` passed all three rules:

| Set | M0 | v2.7.2 | v2.7.3 | Change (95% interval) |
|---|---:|---:|---:|---|
| Test, as-of from 2019 (2,290 tasks, 45 weeks) | 1.129 | 1.087 | 1.072 | −0.0140 [−0.0226, −0.0049] |
| Train, as-of to 2018 (3,943 tasks; in-sample for the table) | 1.125 | 1.078 | 1.065 | −0.0134 [−0.0209, −0.0039] |

- At test, the gain is FX's: G10 −0.031, EM −0.019, the 15-currency basket −0.027.
- UST notes are +0.008 at test, at h = 3–4, where v2.7.2 is already narrow and loses F2/F3's k 1.2.
- It is a calm-week bet: 76% of test weeks improve, while shock weeks lose 0.10–0.15 per task (2008-10-27,
  2011-10-31).
- Two alternatives failed E24's rules: bills without the M0-t mixture (`bills0`), and the profile on top of the
  family widths (`ev_k`).

E25 then tested pairing on the `pair2` roster (7,647 two-cell tasks on the same data, FOMC and ordinary weeks), was
pre-registered, and passed all four rules with pairing off:
- FOMC weeks: test −0.0224 [−0.0375, −0.0091], train −0.0252.
- Ordinary weeks: test −0.0334, train −0.0124.
- No group is worse, and bills improve most (−0.13). The gain is mostly in the joint term.
- With 19 of 104 cards having two cells, as in the practice set, that is about −0.004 per card overall.

E25's own check on this release (pairing on against off with the profile active, FOMC-week two-cell tasks) is run
by the experiment's owner from `exp/e25_pair.py`.

## Checks (sub/verification-v2_7_3/)

| Check | Result |
|---|---|
| 104 public cards against v2.7.2 (4,096 draws, seed 0) | With `pair_match` on: draws identical on all 104, `forecast_meta.json` identical apart from the version, and the profile fires on none. At the default: exactly the 19 two-cell cards change |
| E23 short roster (1,600 tasks) against v2.7.2 | With `pair_match` on: 1,593 identical; the profile fires on 7 tasks (FOMC-week Monday as-of, h = 2), and 6 of them improve (mean −0.060). At the default: the 280 two-cell tasks change on top of that, and nothing else |
| `fomcweek` (6,233 tasks) against the E24 wrapper's `ev` | draws identical on all 6,233 (seed 0); harness scores equal the stored arm exactly on all 3 seeds |
| Source tests (`sub/tools/verify_source.py`) | 120 tests OK |
| Fuzz (70 synthetic cards, seed 5) | 0 not ok, 0 fallbacks |
| Packaging tests / backtest contract tests | 14 / 26 OK |
| Local linux/amd64 image `agenthon-v2_7_3:2.7.3` | label `qfbench2.interface_version` 2.0, no `VOLUME`, user 65534 |
| 104 public cards in the constrained container (`verify_local.py`; image `sha256:899c24447228…`) | 104/104 passed: 0 gate failures, 0 fallbacks; host and container agree within 2.9e-11; at most 9.9 s and 816 KB per card; 3 same-seed repeats byte-identical |
| 31 Final-shaped synthetic cards in the container, `--network none` (`final_cards.py`): the audit's 20, and 11 FOMC-week cards as-of 2026-10-26 | All exit 0 with the three files and no fallback. All pass the official gates except the audit's `syn-F1-noprov`, which declares no as-of; the verifier refuses it as an organizer fault, as under v2.7.2. The profile fires on the 10 FOMC-week cards built to trigger it (including `USD_MXN` / `EUR_USD` / `USDJPY` names and a `[forecast].asof`-only card) and not on the step-5 card. Where it does not fire, the cards equal v2.7.2 draw for draw with `pair_match` on; at the default only the five two-cell cards differ (`syn-F1-ecbnames`, `syn-F4-fomc-2cell`, `syn-F4-ust1y-2h`, `syn-F1-noprov`, `fomc-F2-step5`). No warning on stderr |

## Profiles (`--profile`, local comparisons only)

| Profile | Meaning |
|---|---|
| `default` (`final`) | v2.7.7 |
| `v2.7.6`, `v2.7.5`, `v2.7.4`, `v2.7.3`, `v2.7.2` | those releases, bit for bit on every card |
| `v2.7.1`, `v2.7`, `v2.6.1`, `v2.6`, `v2.5`, `v2.4`, `v2.3`, `v2.2`, `v2.1`, `v1.2` | those releases, bit for bit on daily and monthly cards |
| `prospective`, `balanced`, `dev-leaning`, `random-leaning` | as in v2.7.0 on daily cards, except that they take v2.7.3's two defaults: on a profiled card the profile replaces every family width (so they equal the default there), and two-cell cards are not paired. They keep the default monthly switches (E30, the seasonal step) |

## Interface

```text
forecast --panels /input/panels/ --text /input/text/ --asof YYYY-MM-DD --out /output/forecast.parquet
```

It writes `forecast.parquet`, `forecast_meta.json` and `forecast_rationale.md` next to `--out`, and runs the
official gates G0–G3 as a self-check. Defaults: 4,096 draws; the seed is `--seed`, else `QFBENCH_SEED`, else 0.

## Files

| File | Role |
|---|---|
| `forecast.py` | The forecaster (one file) |
| `Dockerfile`, `requirements.txt`, `.dockerignore` | Unchanged from forecast-v1_2 |
| `ARTIFACT_PROVENANCE.md` | Provenance record shipped inside the image |
| `test_forecast.py` | 149 tests. Uses `../backtest/m0.py` and `../forecast-v1` … `../forecast-v2_7_6`; not part of the image |

## Run, test, build

Nothing has been pushed or uploaded. `sub/tools/pack.py` has a disclosure entry for `forecast-v2_7_7`.

```bash
OPENBLAS_NUM_THREADS=1 python -I forecast-v2_7_7/test_forecast.py
OPENBLAS_NUM_THREADS=1 python sub/tools/fuzz_cards.py --build forecast-v2_7_7 --cases 70 --seed 5
bash sub/tools/build_local.sh forecast-v2_7_7 agenthon-v2_7_7:2.7.7 verification-v2_7_7
OPENBLAS_NUM_THREADS=1 python sub/tools/verify_local.py --build forecast-v2_7_7 --verification verification-v2_7_7 --image agenthon-v2_7_7:2.7.7 --workers 3
python -I sub/verification-v2_7_7/identity_checks.py experiments   # also: practice
python -I sub/verification-v2_7_7/final_cards.py run CARDS
```

## Status

See `../sub/PIPELINE.md`.
