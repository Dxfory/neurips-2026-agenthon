# Artifact provenance — t4agent v21.14

## Executive summary (read this first)

This record is kept with the image source for review against `docs/TRAINING-POLICY.md` and
`docs/ARTIFACT-POLICY.md`. The submission calls the House model through `$MODEL_ENDPOINT` and
ships no model weights, adapters, embeddings, external dataset files or stored task answers.
It does ship a fitted CPI nowcast, calibrated interval/revision/EPS values, and forecasting choices
selected using historical outcomes. The detailed data, coefficients and historical results below
are retained; v21.6 adds eligibility checks, not a new fit.

**A retained value is still outcome-selected.** An original literature or a-priori rationale does
not erase a later decision to keep it after comparing realized scores. This applies to
`STAT_WEIGHT`, `PHI`, `_inflation`, AR(1) defaults, switching thresholds and positioning/ranking
choices. Historical validation tables are not a claim of an untouched end-to-end holdout when
the results also informed retention or version choice.

### v21.6 per-task capability bounds

`t4agent/artifact_policy.py` is the registry. A capability is available only when the task's
parsed, exact `cutoff_date` is on or after its bound; missing/invalid dates and unknown capabilities
permit none. The policy is immutable and task-local. Practice permissions, target names and a
previous task cannot relax it.

| Capability | Eligible from (inclusive) | Disclosed availability basis and covered choices |
|---|---|---|
| `statistics` | **2022-02-01** | Daily observations through 2021-12-31 and complete 2021 CFTC files became available in early January 2022; exact final days are not established. The following month is a conservative bound. Covers `_RW_K`, `_inflation` retention, AR(1)/candidate restrictions, `CHANGE_SWITCH`, `LEVEL_REVERT_DAILY`, `RATIO_SWITCH` and the historical positioning choice. |
| `nowcast` | **2022-01-01** | CPI first prints end at the 2021-12-10 release; disclosed EIA readings extend through 2021-12-27 and fitting values were available by year-end. Covers coefficients, seasonal gaps, scales and fitted-performance summaries. |
| `revision` | **2022-01-01** | Calibration and resolving ALFRED releases end on 2021-12-23. Covers direction weight, scale factors and the outcome-selected latest-estimate centre. |
| `eps` | **2022-01-01** | Numerical fitting, comparison and selection use SEC facts filed before 2022-01-01, including share-base comparability checks. Covers retained `PHI`, carry, scales and direction frequencies; company-universe metadata limitations are stated below. |
| `blend` | **2020-01-01** | Retention of `STAT_WEIGHT` after TreasuryDirect pseudo-auction outcome comparisons for 2016-2019. The numeric weight staying at 0.5 does not make this selection-free. |

These are **conservative implementation bounds**, not exact publication dates, organizer approval,
or a guarantee of Final compliance or predictive performance. They cover the disclosed numerical
fitting/selection dependencies; the historical practice exception and later company-universe
metadata remain separately disclosed. The 2026-10-09 local audit reported a 2026-retrieved SEC ticker-to-CIK map and a large-company
universe without a documented as-of selection rule (`research/t4_analysis_2026-10-09/raw/v-comply.md`,
§1 F8, and `raw/a6-comply.md`, §1; indexed as E05/E04 in
[`T4/error_book/evidence.json`](../error_book/evidence.json)). Retrieval date alone does not establish
post-cutoff information use. Whether universe selection used information unavailable at a task
cutoff remains an unresolved provenance limitation; the filed-date filter governs numerical
facts and does not by itself prove contemporaneous universe selection.

If `statistics` is unavailable, `runtime_naive.py` uses an admitted current-task series only:
last observation for a level, zero for a change, and the standard-normal 90% quantile times observed
step RMS with the existing horizon/unit conversion. This is a **theoretical interval, not fitted or
empirically guaranteed 90% coverage**. It reports no backtest accuracy. Ineligible nowcast, EPS and
revision objects are not constructed, so their historical-frequency/performance descriptions do
not enter prompts. Before the `blend` bound the statistical blend weight is zero; corpus-based
fallbacks can still provide an answer when the model is unavailable.

**In-sample exception (Track 4 discussion #24; organizer rulings of 2026-10-05 and 2026-10-07).** The ten
Development practice units -- `t4-auction-btc-202411-us7`, `t4-cotpos-202411-us10`, `t4-cpicomp-202410-us11`,
`t4-credit-event-2023`, `t4-eps-growth-2024Q3-banks`, `t4-eps-yoy-2023Q2-mixed`, `t4-fomc-curve-20220728`,
`t4-fomc-curve-20240918`, `t4-macrorev-20240930-us6`, `t4-postearn-20240201-megacap` -- are in-sample for this
agent, and its scores on them are not out-of-sample evidence:
- version choices from v11 to v21.x referred to the Development leaderboard scores of our own uploads, which the
  organizers compute from these units' outcomes (source: the Agenthon Development leaderboard; each score read on the
  day it was posted, 2026-09-29 to 2026-10-08);
- on 2026-10-05, to diagnose a Development score, the October 2024 first prints of CPI gasoline, energy and all items
  (the outcomes of `t4-cpicomp-202410-us11`) were read once (source: ALFRED, Federal Reserve Bank of St. Louis,
  first-published vintages; retrieved 2026-10-05). The diagnosis found the CPI unit was not the cause of the score
  change and the pump-price nowcast was kept: an outcome-informed version choice. The numerical constants were
  unchanged, but retaining the method is still selection. This practice-outcome use is the disclosed in-sample
  exception, distinct from the pre-2022 historical calibrations and not a bypass of v21.6 capability bounds;
- v3's coverage-first widening (intervals x4) followed v2's per-unit Development results (2026-09-29); v7 removed it,
  and nothing of it remains in the code.

## Models

| Model | Access | Use |
|---|---|---|
| `nvidia/nemotron-3-super-120b-a12b`, snapshot `rl-030326-fp8`, training cutoff unpublished | `api` (House route) | planning, per-entity analysis, JSON repair, review, writing `submitted_reasons` |

| `t4agent-cpi-pump-price-nowcast` 21.6.0 (the version follows the agent's, as in `t4agent.LOCAL_MODELS` and the descriptor's `models[]`; the coefficients and their revision are unchanged since 20.0.0), revision `55f9abb9f5c28d9f8a6821860a86b9303fc4ffb95e466293e9d46f96a557f2c7` (SHA-256 of its coefficients, `t4agent/nowcast.py` `coefficients_digest`), training cutoff 2021-12 | `local` (v20; a fixed linear formula evaluated in Python, no weights file) | statistical baseline of the first print of CPI gasoline, energy and all items when the corpus has weekly pump prices for the target month (see "Learned artifacts") |

No other model -- neural or otherwise -- is loaded or called.

## Learned artifacts

The tables record historical fitting and validation; runtime eligibility is governed by the five
capabilities above, including values or methods that were retained after outcome comparisons.

Two calibrations (v15): a prediction-interval table and the revision base rate's calibration parameters, under
`ARTIFACT-POLICY.md` ("calibration parameters ... prediction-interval calibration ... numerical lookup tables"). v20 adds
one fitted linear model ("Fitted linear models ... permitted with disclosure of the actual learned models and their
fitting/selection data"), listed in full below and in `models[]`. No tree, neural or retrieval model is trained; BM25
statistics are computed at run time from the unit's own frozen corpus (documents dated on or before the cutoff), as
`ARTIFACT-POLICY.md` permits.

| Artifact | `t4agent/nowcast.py` `NOWCAST` (v20; `models[]` row `t4agent-cpi-pump-price-nowcast`, revision = SHA-256 of the coefficients `55f9abb9f5c28d9f8a6821860a86b9303fc4ffb95e466293e9d46f96a557f2c7`) |
|---|---|
| What it is | a linear nowcast of the first print of a seasonally adjusted month-over-month CPI change: BETA x (average weekly retail gasoline price in the target month / average in the month before - 1) x 100 + GAP[calendar month], for CPI gasoline (CUSR0000SETB01: BETA fixed at 1, 12 GAP values), energy (CPIENGSL: BETA 0.537, 12 GAP) and all items (CPIAUCSL: BETA 0.049, 12 GAP), with a 90% half-width for two weeks of the target month seen and for three or more (6 values): 45 numbers in all, printed in the module. It is used only when the entity names one of these series, the target is a seasonally adjusted month-over-month change, and the unit's frozen corpus holds a weekly gasoline price table with at least two readings in the target month and three in the month before; otherwise the series' own AR(1) stands. The forecast enters like every statistical baseline (equal weight with the model) |
| Fitting data | EIA weekly U.S. regular gasoline retail price (FRED `GASREGW`, 1990-08 to 2021-12; market data, not revised) and the ALFRED real-time periods of the three CPI series (BLS), from which each month's FIRST print and the history as known the day before its release are rebuilt: reference months 2011-04 to 2021-11 for gasoline (its real-time record starts in 2011), 2002-01 to 2021-11 for energy and all items. Public domain |
| Fitting / selection | least squares with calendar-month intercepts (`tools/calibrate_nowcast.py`); the gasoline slope is fixed at 1 (the index is the pump price). Selection, out of sample (each month forecast from earlier months only): the static intercepts beat intercepts from the last five years and intercepts shifted by the last nine months' residuals on all three series; the 90% half-widths are the interval-score-optimal quantiles of the out-of-sample errors |
| Historical validation | out of sample, scorer-5.2.2 predictive quality against the last first print: gasoline 0.754 (0.753 / 0.755 by half; AR(1) 0.525), energy 0.745 (0.757 / 0.728; AR(1) 0.525), all items 0.662 (0.703 / 0.608; AR(1) 0.520); with only two weeks of the target month seen 0.710 / 0.723 / 0.664. Time split of the fitted constants (fit on one half of the months, score the other): 0.726 / 0.725, 0.730 / 0.736, 0.631 / 0.685 |
| Cutoff | every fitting value was disclosed as public by 2021-12-31 (last CPI release 2021-12-10; EIA observations through 2021-12-27); v21.6 requires `nowcast` eligibility from 2022-01-01. No assumption is made about every possible task cutoff. |
| Data checksums | `calib_data/README.md`: `GASREGW_1990_2021.json` SHA-256 `c65aba6b9246f525ff50da02c583da6a1312ab5e975cca58e3f531be9bc3e7d7`; the CPI real-time files (SHA-256 over the 11 files' digests in name order `cd7fc41dc9eebb9c8b5809f53557cfce8c5f59097705abd166793f8dbc437e4d`, CPIENGSL and CPIAUCSL among them, CUSR0000SETB01 too) |
| Reproduction | `tools/calibrate_nowcast.py` (not in the image) prints the constants; `tests/test_offline.py::test_the_pump_price_nowcast_reproduces_from_the_public_data` checks them, and `test_the_nowcast_is_disclosed_as_a_local_model_with_its_coefficients_digest` the `models[]` row |

| Artifact | `t4agent/stats.py` `_RW_N`, `_RW_H`, `_RW_K` (SHA-256 of the `_RW_K` literal: `a430f97863ad43e6d40b2ed6d8f3944e39352bb74f2b5808009792d83620eb98`) |
|---|---|
| What it is | the 90% multiplier k(n, h) of a random-walk extrapolation from a short window, on a fixed grid n = 5, 10, 20, 40, 60 and h = 1, 5, 10, 20, 40, 60, 90, 120 steps, read bilinearly in log n and log h. v16: the k minimising the mean interval score (the scorer's loss) of the half-width k x rms(previous n steps) x sqrt(h) -- the scale-weighted 90% quantile of abs(h-step change) / scale, each series weighted equally. v15 used the unweighted quantile (90% coverage), which overstated k by 5-25% and raised the mean interval score by up to 3.4% out of sample |
| Calibration data | FRED daily series, observations 1990-01-02 to 2021-12-31: Treasury constant-maturity yields DGS3MO, DGS1, DGS2, DGS3, DGS5, DGS7, DGS10, DGS20, DGS30 (U.S. Treasury); exchange rates DEXUSEU, DEXJPUS, DEXUSUK, DEXCAUS (Federal Reserve H.10); WTI spot DCOILWTICO (EIA). Public domain; published within days and not revised, so every value was public by early January 2022 |
| Historical validation | time split 1990-2005 / 2006-2021, mean interval score relative to v14's constant 1.839: the fitted table scores 0.989-1.005 on the other period (v15: 0.999-1.034). ALFRED 2021-12-31 vintage of eleven CPI-U components (BLS): the agent's 9-month-window scale scores 1.000 of v14's mean interval score |
| Fitting / selection | quantiles are fitted to realized historical changes and interval-score losses; the grid was fixed before measuring and origins are every 5th trading day. Retaining the table/inflation after score comparisons is selection. No Development competition-unit outcome was used for this fit. |
| Cutoff | the documented latest availability is early January 2022, not 2021-12-31. v21.6 requires `statistics` eligibility from the conservative 2022-02-01 bound; exact final publication days are not established. |
| Data checksums | per-file SHA-256 in `calib_data/README.md` next to the agent folders (data kept outside the image); SHA-256 over the 25 files' SHA-256 digests in name order: `7ac12c74fb7acda04595e890773bc1fa33923edd2b9c40981ff2804586be7903` |
| Reproduction | `tools/calibrate_scales.py` (not in the image) prints the table (`daily_is`); `tests/test_offline.py::test_the_calibration_table_reproduces_from_the_public_data` checks two cells |

| Artifact | `t4agent/revisions.py` `DIRECTION_WEIGHT` 0.15, `SCALE_ONE_RELEASE` 0.76, `SCALE_MORE_RELEASES` 0.76, expected next estimate = the latest estimate (v15; factors v16) |
|---|---|
| What it is | the calibration of the vintage-table revision base rate: the share of a past-revision direction rate's departure from 0.5 that is kept, and the factors that widen its envelope one and two (or more) releases ahead |
| Calibration data | FRED API real-time periods (ALFRED) of fourteen monthly U.S. series -- PAYEMS, MANEMP (BLS); INDPRO, TCU (Federal Reserve); RSAFS, HOUST, DGORDER, NEWORDER, PERMIT, TTLCONS, BUSINV (Census); PI, PCE, DSPIC96 (BEA) -- vintages 2003-01 to 2021-12-23, public domain. At every release from 2006 the vintage table known that day is rebuilt in the Development unit's shape (5 vintages x 8 reference months) and `_prior` is asked about the next estimate of the latest two reference months, one and two releases ahead: 12,758 forecasts, every outcome itself a release published by 2021-12-23 |
| Fitting / selection | v16: the envelope factors minimise the mean interval score (scale-weighted 90% quantile of abs(next - latest) / envelope, each series weighted equally): 0.76 one release and more ahead. v15 used the unweighted quantile (90% coverage: x1.22 / x1.51), which raised the mean interval score by 8%: wide envelopes (large past revisions) over-cover and their width dominates the loss. The direction weight minimises the Brier score over a grid 0.00-1.00 with the centre fixed at 0.5 (the fitted centre 0.53 reflects 2006-2021 growth and was not kept). Before (v14): Brier 0.291 (a coin flip scores 0.250), relative MAE of the expected value 46 bp vs 36 bp for the latest estimate. After: Brier 0.2485, 35.9 bp, mean interval score 0.966 of v14's |
| Validation | time split: envelope factors fitted on 2006-2013 (0.81 / 0.75) score 0.952 of v14's mean interval score on 2014-2021, and the reverse (0.73 / 0.78) 0.954 (v15's factors: 1.069 and 1.069); series split for the direction weight: fitted on eight series, the other six score Brier 0.2456 (coin flip 0.2500), and the reverse 0.2512 (0.2500) -- the direction signal is weak either way, which is why only 15% of it is kept |
| Cutoff | fitting and resolving releases were published by 2021-12-23; v21.6 uses the conservative `revision` bound 2022-01-01. |
| Data checksums | per-file SHA-256 in `calib_data/README.md`; SHA-256 over the 14 files' SHA-256 digests in name order: `bf3e1b7fe5cb25f84b5e8b649eb8b97fb1807892830b4332ac11a213a77e2a2f` |
| Reproduction | `tools/calibrate_revisions.py` (not in the image) prints the factors; `tests/test_offline.py::test_the_revision_calibration_reproduces_from_the_real_time_data` checks them against the constants, that v14's and v15's factors score worse, and the Brier score |

## Constants and method choices: original rationale and later selection

| Constant (`t4agent/config.py`, `t4agent/calibrate.py`) | Value | Basis |
|---|---|---|
| `MAX_CLAIMS_PER_ENTITY`, `CLAIM_MAX_CHARS` (v7) | 4, 600 | scorer 5.2.2 claim rules: a claim is a verbatim quote of its cited span (passes the figure check, never judged); 600 characters stays far below the 400-judge-token cap; checked with the organizers' `check_claim_rules` on the public units' corpora; no labels |
| `WIDEN`, `Z90` | 2.0, 1.645 | observation that the model's own 90% ranges are narrow; standard normal quantile |
| `REQUEST_BUDGET`, `MAX_TOKENS`, reasoning budgets, time budgets | 23, 4000, 1400/2000, 400/470 s | published House allowance and unit clock |
| `STAT_WEIGHT` (model vs statistical baseline) | 0.5 | equal-weight forecast combination, the standard untuned default. v14 note: NOT validated on this agent's forecasts. An out-of-sample replay of the statistical baselines on pre-cutoff history only (`tools/historical_skill.py`, scorer 5.2.2's predictive quality, naive rule = 0.5) gives auction 0.51, positioning 0.49 and CPI components 0.57 (3 origins, its six-month median); the naive rule itself is selected for 71% of auction and 81% of positioning forecasts, so there the blend mostly shrinks the model's forecast halfway towards the naive rule. Validating the weight would need model forecasts at past pseudo-cutoffs, which the model may remember; the default is kept rather than fitted v16: tested on the model's own forecasts with `tools/pseudo_auctions.py` -- 12 pseudo-units in the Development auction unit's format built from TreasuryDirect results of 2016-2019 (public domain, `calib_data/README.md`), 72 forecasts with the blend off: predictive quality against the last-auction naive rule 0.505 for the model alone, 0.512 at 0.5, 0.513 at 0.75, 0.515 for the baseline alone (six-auction mean as naive rule: same order). The pre-registered bar for a change (+0.01 and wins in two thirds of the units) was not met, so the weight is unchanged; retention is nevertheless outcome-based selection under the v21.6 `blend` gate. No Development competition-unit outcome was used in that historical experiment |
| statistical-baseline parsimony margin, minimum backtest origins, small-sample inflation | 10%, 4, 1 + 0.75/sqrt(n) (v6: 1 + 1.5/sqrt(n)) | original v7 a-priori rationale: inflation was halved because scorer 5.2.2 charges interval width. Later retention of `_inflation` after the v15/v16 historical interval-score comparison below counts as selection and is subject to `statistics` eligibility |
| baseline default for a series of period changes; origins needed to replace a default (v3) | recent median (6), 8 | fixed a priori: changes of a random-walk level are close to independent, so the last change is a noisy forecast; a 6-origin backtest cannot separate methods. Prompted by the model's development-time outputs, not by outcomes |
| label-probability blend (model vs quantity-implied), `IMPLIED_LABEL_WEIGHT` | 0.25 / 0.75 (v2: 0.5 / 0.5) | fixed a priori: the implied side is read off the `WIDEN`-widened range, the same distribution the interval is built from, so it carries the calibration correction the model's own label probabilities lack; the 0.25 hedges a misread rule. Motivated by the model's development-time outputs (a middle band chosen from a too-narrow range), not by outcomes |
| outlier re-check thresholds | robust z > 4, > 3x backtest error | fixed a priori |
| EPS anchor carry `PHI` and relative error `REL_SD` (v4, `t4agent/eps.py`) | 0.4, 0.25 | the seasonal random walk with a partial carry of the latest same-quarter change (Foster 1977; Brown and Rozeff 1979): original v4 PHI from that literature, REL_SD an a priori error scale; runtime inputs are the entity table's prior-year EPS and filing table. Later v18/v20 outcome-based selection/retention of PHI and the other EPS values supersedes a blanket no-selection claim; the full anchor is gated by `eps` eligibility |
| revision-prior smoothing (v4, `t4agent/revisions.py`) | Beta(2, 2) on the all-stage rate, stage rate shrunk towards it with strength 2 | fixed a priori; the frequencies are counted from the unit's own pre-cutoff vintage table |
| anchor / prior weight against the model | `STAT_WEIGHT` 0.5 | the same equal-weight combination as the v2 statistical baselines |
| analyst group size, thinking budget, reply room per entity (v5) | 3, 2,800 tokens, 230 tokens | fixed a priori from the published 4,000-token cap (thinking included) and the measured length of a forecast entry; not fitted to outcomes |
| review call and its weight (v5) | 1 request, 80 s reserved, equal-weight blend; off from v21 (`REVIEW` 0, `T4_REVIEW=1` turns it on) | fixed a priori; the same equal-weight combination as the other second opinions. v21: off, see the v21 request-allocation row |
| v6 feature thresholds (`t4agent/features.py`) | co-movement shown only at abs(correlation) >= 0.85 over >= 6 periods, strongest one only; same-month seasonality only with >= 24 monthly observations; driver needs >= 2 observations in the target month | fixed a priori; the features are deterministic transformations of the unit's own pre-cutoff tables and filings; nothing is fitted |
| unconfirmed-outlier shrink (v3) | halfway to the flagging reference | fixed a priori (midpoint of the two candidate centres); motivated by period/base errors seen in the model's own development-time outputs, not by outcomes |
| `COVER_FACTOR` (v3-v6) | removed in v7 | it targeted scorer 3.1.0's coverage penalty; scorer 5.2.2 scores intervals with the interval score (width plus 20 times the miss), whose optimum is the true 5th/95th percentiles |
| v7 interval rule (`t4agent/calibrate.py`) | own scale (geometric mean of the backtest error and the table step scale) as the floor; a wider raw model range moves it halfway (replaced in v8, next row); the model range is widened x2 only without an own scale; plan deviation x 1.645 only without an own scale; `REL_FLOOR` 3% only when nothing else is known; a naive value outside the range pulls the near bound halfway towards it | derived from the interval score's properties (proper for central intervals; width and 20x misses), fixed a priori; not fitted to any outcome |
| v8 interval rule (`t4agent/calibrate.py`) | an own scale measured on the entity's history (an error quantile over at least 4 backtest origins, or an envelope of at least 4 vintage-table revisions: the same-stage ones, or with fewer than 4 of those their log-scale midpoint with every stage's revisions) stands alone; an extrapolated or rule-based own scale (random walk from a short history, the EPS anchor's `REL_SD`) moves to the geometric mean with a wider model range; the plan's unit-wide deviation is ignored when the rows carry different `unit`/`units` | the model's ranges were seen off by large factors in both directions in development runs (+/-4 bp on a seven-week rate move; ten times the revision scale on revised payrolls), and the interval score's cost of a wrong width is roughly symmetric on the log scale (normal errors: 2x too wide +60%, 2x too narrow +52%); fixed a priori, the 4-origin threshold is `stats.MIN_BACKTEST`; not fitted to any outcome |
| v8 naive combination (`t4agent/predict.py`) | a regression forecast with no statistical baseline from the entity's own history is combined with the naive rule (zero change, or the latest value carried forward) at `STAT_WEIGHT` 0.5 | the same equal-weight combination as the other baselines; the naive rule is the simplest statistical forecast; fixed a priori |
| v8 revision centring (`t4agent/predict.py`) | the model's revised value is combined with the vintage base rate's expected next estimate at `STAT_WEIGHT` 0.5 before the label is read off it, unless the two are more than half the latest estimate apart (another scale) | the same equal-weight combination as every other baseline; fixed a priori |
| v8 label rules from the table (`t4agent/labels.py`) | when the plan gives no label rules, the entity table's own `<label>_condition` columns supply them (the left-hand quantity becomes Q; the label without a one-sided condition is the middle band) | exact parsing of the organizers' stated conditions; no constant |
| v8 plan grounding (`t4agent/grounding.py`) | planner reads the documents (small corpus, up to `PLAN_DIGEST_CHARS` 40,000) or each entity's computed facts and first 4 key passages; a key driver, label cue or naive-baseline note is dropped when it states a figure found nowhere in the task, the entity table or the corpus, or names a date after the cutoff that the task and corpus do not name or whose result it speaks of | exact string checks, no model; prompted by planner notes seen in development runs (a policy rate the corpus shows was changed; the outcome of an event after the cutoff); fixed a priori |
| v9 level-rule unit factor (`t4agent/predict.py`) | a target derived from a level (`target_from_level`) is multiplied by 100 when the level is a percent and the target is in basis points, by 0.01 the other way round, by 1 when the units agree or the target names none; the rule is not used when the target names a unit and the level's unit cannot be read; units are read from the task's target name and `unit` columns first, then the plan's description of the target | definitional (1 percentage point = 100 basis points); fixes v8 submitting a yield change in percentage points as basis points; not fitted to outcomes |
| v10 reasons planned against drivers (`t4agent/reasons.py`) | drivers = the task statement's 'Reason from ...' list (a parenthesised list of short items expanded), else the plan's grounded key drivers, at most 8; per driver up to 2 prose passages from different documents carrying at least 2 of its content words (1 for a one-word driver), ranked by that share, adjacent word pairs and whether the analysts cited it; at most 22 candidate passages; each reason covers 1-3 drivers, with up to 2 supporting citations; mechanism <= 650 characters, implication <= 450; implications checked against the submitted answers and rebuilt from them on a mismatch; 1,100 thinking tokens, 75 s reserved | fixed a priori from the published grading contract (uncovered target reasons score 0; the caps of 6,500 reason bytes and 46,500 evidence bytes) and from the task statements' own driver lists; checked with a proxy judge (`tools/judge_reasons.py`) that never sees outcomes |
| v20 reasons direction tolerance (`t4agent/reasons.py` `NEUTRAL_SHARE` 0.1, `versus_latest`) | against the latest value (a level target), an answer within a tenth of its 90% half-width argues no direction, so a reason declaring up or down for it is not held back; the reasons prompt lists each level answer as higher than, lower than or about the same as its latest value (the reference of the reasons' declared direction). Change targets keep the sign test | fixed a priori (a tenth of the half-width is about 0.16 standard deviations). Prompted by the developer's proxy reasons judge (`tools/judge_reasons.py`, no outcomes) on v20's first sweep: the CPI reasons on the gasoline rebound were held back because headline's 0.17 sat 0.01 below its latest 0.18, and other reasons declared 'higher' for components forecast below their latest prints. Checked live: in two CPI runs all three reasons were kept and the task's four drivers covered |
| v21 request allocation (`t4agent/config.py` `REVIEW` 0, `ANCHORED_SAMPLES` 3, `EXTRA_VOTES` 4, `REASONS_ROUNDS` 2; `t4agent/predict.py` `anchored`, `undecided`, `extra_votes`; `t4agent/llm.py` `parse_reply`) | the review call is off; a regression unit whose every entity has a code baseline with a measured or calibrated scale takes at most 3 analyst samples per group; a classification unit keeps 4 requests back and spends them on single-entity calls for the entities whose sampled labels split (top label under 75% of the votes), most split first; a reply's JSON is recovered in code (trailing commas, NaN, Python literals; the last object in its reasoning) before a repair call; the reasons get up to two follow-up rounds; the end of the unit keeps the outlier re-check plus two requests for the reasons, which no JSON repair may take (`Forecaster.end_reserve`) | fixed a priori from v20's own sweep logs (no outcomes): 72% of the 194 requests went to analyst samples that moved regression answers with a code baseline by 0.01-0.25 standard deviations of the submitted interval (bank EPS 0.01, CPI 0.05, auctions 0.09); 8% to JSON repair calls; the review changed one answer in eleven units; the model changed 5 of 30 classification labels from the code's, the same labels that flipped between same-code runs. The thresholds (3 samples, 4 votes, 75%) are round numbers, not fitted. Checked on the paid deepinfra route against v20 (no outcomes of any task): labels differing between same-code runs of the five classification units 1.07 -> 0.27 per pair of runs; 2016-2019 pseudo-auction units (TreasuryDirect results, pre-2022) predictive quality 0.5098 -> 0.5106 at about 40% fewer requests; proxy reasons judge 0.672 -> about 0.685 |
| v21 label-rule order (`t4agent/predict.py` `_rule_sources`) | the conditions the task's entity table states, then the task's own label assertions, then the planner's rules (only where the task states none), then the directional rule | correctness, not a tuned constant: the v19 audit reproduced a planner misreading the example unit's beat threshold (Q > 5 for 1.575) and turning a forecast of 1.80 into 'inline' |
| v21 reasons figure check (`t4agent/reasons.py` `ungrounded_figures`) | a reason whose mechanism states two or more figures that neither its cited passages, the task statement and table nor the submitted answers contain (a figure matches a source number to its own rounding and within 5%; integers up to 31 and years without % or $ are not figures) is kept only until a follow-up round writes a clean one | fixed a priori; prompted by the developer's proxy reasons judge on v20 (post-earnings reasons stated invented reactions such as 'roughly -2% to -3%'); two figures so that one figure of the reason's own arithmetic on cited numbers passes |
| v21.1 a series in an unnamed date table (`t4agent/stats.py` `_own_table_series`, `find_series(allow_own)`, `relative_quantity`, `_OTHER_REFERENCE`; `t4agent/calibrate.py` `table_halves`; `t4agent/predict.py` `Forecaster.__init__`) | when no table column is named by the entity or the target, the single series of a date-indexed table in the entity's own document (a two-column table, or a vintage table read off its latest vintage) is the entity's series, for regression and ranking rows and for a classification row whose quantity is a relative change; a relative target (percent change, growth, m/m, q/q) reads the level as 100 x ln(level) and its change over the horizon; a year-over-year rate is not derived; a target measured against another reference (a revision, a surprise, consensus, an abnormal or excess return) takes no such series | fixed a priori from the unit format: a column's name carries no meaning the agent can rely on (CATEGORIES.md: the held-out families are mostly unpublished). Found with stress units built from public data resolved before 2022 (`tools/build_pre2022_units.py`; no Development outcome read), where v21 had no series: a housing-starts row submitted 0 and an industrial-production row its index level (110) as a percent change |
| v21.1 a level several steps ahead (`t4agent/stats.py` `forecast`, `series_half_width`, `_horizon_steps`) | a level read off the entity's own table is backtested at the horizon in the series' periods (trading days for a daily series), and its step-volatility half-width scales with the square root of the steps; columns named as in v12-v21 keep one step | random-walk scaling (no fitting); a daily price one week out is five steps ahead |
| v21.1 resolution floor (`t4agent/stats.py` `_resolution`) | a measured scale of zero (a series flat over its window) is floored at the series' smallest step, else one unit of its last printed decimal | definitional: a printed 1.2 is known to about 0.1; v21 fell back to a +/-20 default |
| v21.1 change-type base (`t4agent/predict.py` `_CHANGEY`, `_LATEST_CHANGE_FIELD`, `baseline_value`) | m/m, q/q and y/y targets are change-type; the base of a change-type target is the row's latest published value of the same change (a field naming latest/last/prior and a change word, sharing a word with the target), else zero; never a level column | definitional (units); the public CPI rows keep v21's carried-forward first print, positioning keeps zero |
| v21.1 status-quo label (`t4agent/predict.py` `_PRIOR_ORDER`) | after v21's order: hold, unchanged, no_change, stable, flat, maintain, neutral, inline, same | the no-change constant of a decision or move label set, used only without any model reply or code anchor; v21 took the first label ('cut', 'appreciate') |
| v21.1 content check (`t4agent/cite.py` `unit_names`, `contentful`) | the unit's entity names, ids and `*ticker` values are blanked before a quote is tested for content | mirrors the scorer (`scoring.unit_entity_names`, `content_free`): a title line that is only the entity's name is content-free and its claim false |
| v21.2 label definitions stated as moves (`t4agent/labels.py` `_A_DIR`, `_A_EITHER`, `_assertion_cond`) | 'rises / increases / widens ... by more than X' is Q > X; 'falls / declines / narrows ... by more than X' is Q < -X; 'X or less in either direction', 'within +/-X' and '+/-X' are -X < Q < X; v13-v21's forms (above, below, between, at least ...) and the refusal of relative thresholds are unchanged | definitional (the label's own wording). v21 read 'falls by more than 1.0%' as Q > 1.0, the region of 'rises by more than 1.0%', and the symmetric band as Q < 1.0: overlapping rules that, when they still summed to about one, labelled a -3% forecast 'stable'. Found on a stress unit built from FX rates published before 2022; every public unit's rules read as before |
| v21.2 a change quantity's base (`t4agent/predict.py` `quantity_baseline`, `_CHANGE_QUANTITY`) | when a classification row's quantity is a change (the plan's interval quantity or the row's own unit text says percent change, change in, bps change or return), a level field is never its base; the planner's field is kept only when it names a change | definitional (units): a currency's quote (111.5 yen) was the naive base of its percent change, and the naive hedge stretched the interval to [-2.6, 57] and tilted the label probabilities read off it |
| v21.2 a fallback label from its rules (`t4agent/labels.py` `rule_label`; `t4agent/predict.py` `_label_from_rules`, `_INFORMED_PRIORS`, `quantity_baseline`) | a classification row with neither a model reply nor a code anchor takes the label its rules (entity-table conditions, the task's label definitions) give at the quantity's naive value -- the row's latest published value of the quantity or of the same change, else zero for a change -- unless the label set has a base-rate prior chosen for its family (beat, up, no_event ...); 'steady' joins the status-quo labels | the naive rule itself (no fitting): v21 took the first label of the list ('rise' for a three-way yield move, 'above' for a CPI threshold, 'strong' for a retail-sales band). Every public unit's fallback answer is unchanged |
| v21.3 the EPS anchor in a rule-less fallback (`t4agent/predict.py` `aggregate`) | a classification row with no model reply and no label rules the code reads still takes its code-computed anchor's level and calibrated scale for its quantity; the label stays the fallback's | the anchor's own calibration (v18/v20, SEC filings before 2022); a growth-band unit stated only in words had each EPS sent out at last year's value with a +/-20 default. Every public unit's fallback answer is unchanged |
| v10 announcement-return scale (`t4agent/config.py` EVENT_RETURN_SD, `predict.event_return_sd`) | 4.0 percentage points (one standard deviation), a floor for the scale of a stock's return over a short window around a scheduled announcement when the quantity is in percent and the entity has no measured scale; used for the submitted interval and for reading threshold labels | a priori from the event-study and earnings-announcement volatility literature (one-day announcement returns of large U.S. stocks: roughly 4-6 points); the low end; not fitted to any outcome |
| v10 samples per group (`t4agent/config.py` MAX_SAMPLES) | 9 (v9: 5) | the queue runs every group's first sample before extra ones and stops launching calls 25 s before the analyst deadline, keeping 4 requests for the re-check, repair, review and reasons; a higher cap only spends budget a small unit left idle |
| v11 label distribution (`t4agent/labels.py` `_cdf`, `t4agent/predict.py` `_settle_label`, `t4agent/calibrate.py` `quantity_range`) | label probabilities are read off the range the answer submits (the same function and inputs as the interval), with a half normal of each side's scale on each side of the point, each holding half the mass, so the point is the median and the bounds are the 5th/95th percentiles; blended 0.75 : 0.25 with the model's own probabilities, then the revision base rate; a label whose region holds the whole submitted range (probability >= 0.95) is the label; settled again after the review | definitional (the submitted interval is the 90% range of the submitted distribution); replaces v10's mode-centred two-piece normal and its separate label range; not fitted to outcomes |
| v11 relative-change targets (`t4agent/stats.py`) | a growth or percent-change target only takes a corpus column that is itself a change or a rate (its name carries growth, change, %, yoy, m/m or return) | definitional: differences of a level are in the level's units, not percent |
| v11 level-rule formula check (`t4agent/predict.py` `level_scale`) | a percent change of a level that is itself a percent is refused | definitional: a yield's or a rate's change is a difference in points or basis points |
| v11 reasons evidence and focus (`t4agent/reasons.py`) | a reason's implication names only entities whose own documents (or a shared one) it cites, otherwise it is rebuilt without them; the prompt lists up to 4 answers furthest from the naive baseline (classification: the labels that differ from the most common one) and asks how the latest evidence carries into the target period | fixed a priori from the grading contract (evidence grounding, answer consistency, coverage); not fitted to outcomes |
| v12 ratio-change series (`t4agent/stats.py` `_ratio_columns`) | a change target on a column 'X % of Y' is measured as (X[t+h] - X[t]) / Y[t] x scale, with X and Y the table's own numerator and denominator columns and scale 100 or 1, accepted only when every row confirms the printed ratio within its rounding (2% + 0.051); otherwise the column is not used | definitional (the task statement's own target: a change expressed in the starting base); not fitted to outcomes |
| v12 same-quantity rate columns and table scale (`t4agent/stats.py` `_same_quantity`, `series_half_width`) | a growth or change column serving a growth target is forecast at its next value; the table step scale is computed on the statistical baseline's own series | definitional |
| v12 reason checks (`t4agent/reasons.py` `argued_against`, `unnamed_sources`, `entity_aliases`) | each entity's declared direction is read against its submitted answer (label; sign of a change; side of the naive value for a level); a reason quoting an entity's own document must name it (id, series code, cleaned name, a word only its name carries, or a 4+ letter abbreviation of a 6+ letter one), and when it does not, the reason is held back if that document is its premise or its mechanism uses a figure (not a year) or two 6+ letter words found only there, else the citation is dropped; a reason whose mechanism names an entity unmistakably (id, series code, full name, or a 5+ letter word only its name carries) without citing that entity's own or a shared document is held back; a self-denial is only an explicit contrast with the submitted answer | fixed a priori from the grading contract (answer consistency, evidence grounding); not fitted to outcomes or to the proxy judge |
| v12 reasons retry (`t4agent/reasons.py` `model_reasons`) | an unparseable reasons reply is retried once without thinking when a request and 30 s remain | the plan's own retry rule (v3); a route that ignores the thinking budget filled the 4,000-token cap |
| v13 benchmark updates in vintage tables (`t4agent/revisions.py` `level_updates`) | a release that revises at least two reference periods whose estimate had stood unchanged over the two releases before it is a benchmark or annual update; transitions across it are left out of the revision base rates and scale whenever regular transitions remain | definitional (statistical agencies schedule annual and benchmark updates as one-offs); not fitted to outcomes |
| v13 regression on a vintage series (`t4agent/predict.py` `_revision_in_target`) | a regression row asking for the next estimate (its level, or through the plan's level rule its change against the latest estimate) takes the revision base rate's expected value and 90% scale in the target's units, as the direction labels do | definitional: the same pre-cutoff vintage table, converted by the target's own definition |
| v13 label bands (`t4agent/labels.py` `assertion_rules`, `_BAND`; `t4agent/predict.py` `_rules`) | a label whose task definition states a numeric threshold or band on the quantity ('above 2.50', 'from -20% to below -5%', 'between -1% and 1%') is a rule on that quantity; comparisons with another quantity and time spans are not read; the planner may write bands 'lo < Q < hi'; a rule source that does not read on a row gives way to the next | definitional: the task's own label definitions |
| v13 second opinions on another scale (`t4agent/predict.py` `_plausible_move`) | a review or recheck quantity more than 10 times the larger of the forecast's measured 90% scale and its own half-ranges from the current forecast is ignored | a priori: a correction moves a forecast by a few spreads; a quantity a hundred spreads away is in other units |
| v13 reasons follow-up (`t4agent/reasons.py` `follow_up`; `t4agent/config.py` REASONS_RESERVE_SEC 90, REASONS_FOLLOWUP_SEC 45) | when fewer than three of the model's reasons are usable, one more call, told why each failed and which drivers remain uncovered, asks for replacements on other premises; a replacement repeating a used premise is skipped | from the grading contract (coverage counts every target reason; fewer reasons never score higher); not fitted to the proxy judge |
| v14 absolute thresholds only (`t4agent/labels.py` `_REFERENCE`, `_NUM`, `_TAIL`) | a label definition that measures the quantity against another one (consensus, estimate, an earlier period, an average, a benchmark, a multiple, versus, relative) is not read as a threshold; a number is matched with its unit atomically and is not read when a relational word follows it | definitional: only the task's own absolute thresholds are rules on the quantity |
| v14 rule sources (`t4agent/predict.py` `_rule_sources`, `_settle_label`; `labels.rules_parse`) | a band must have its lower bound below its upper bound; the label probabilities come from the first rule source that yields a partition on the submitted distribution | definitional |
| v14 negative series (`t4agent/revisions.py`) | the expected next estimate is latest + abs(latest) x the mean relative revision, the measure the revisions are taken in | definitional |
| v14 answer cap (`t4agent/answer.py` `fit_judge_cap`, CAP_WIDEN 0.01, CAP_SHIFT 0.005) | to fit the reasoning grader's 3,000-byte answer cap, numbers lose significant digits only while every interval widens by at most 1% of its width and every point moves by at most 0.5% of it; a ranking is never changed; otherwise the rows stay as they are. v21.8: a second tier (CAP_WIDEN_MORE 0.05, CAP_SHIFT_MORE 0.025) is tried when the first cannot fit, and a ranking is rounded when its points keep their exact order and ties | a priori: a forecast is not traded for a chance at the reasoning bonus. v21.8 bounds the trade instead, from the published rules alone (see "v21.8 changes") |
| backtest inflation (`t4agent/stats.py` `_inflation`) | 1 + 0.75 / sqrt(n) over the n backtest errors (v15 counted n / h for overlapping h-step errors; v16 reverts) | judged by the mean interval score on the public daily series above (windows 60-250 days, horizons 7-56), counting n / h was 0.997-1.003 of counting n: no better, so v14's rule stands |
| v17 rate-series default (`t4agent/stats.py` `_ar1_phi`, `_NEXT_METHODS['ar1']`) | the next value of a stationary rate (month-over-month change, return) defaults to last + (1 - lag-1 autocorrelation of the window) x (window mean - last); level series keep v16's candidates | model selection on public data with observations through 2021 and latest documented availability in early January 2022 (`tools/calibrate_methods.py`, `calib_data/README.md`): scorer-5.2.2 predictive quality against the last value 0.516 / 0.584 / 0.560 for v16's 6-period median and 0.525 / 0.592 / 0.574 for AR(1) on CPI components, twelve macro series' m/m and daily FX / oil returns; better in 5 of 6 half-samples; level series unchanged. v19 (review of v17: those monthly replays read the revised 2021 vintage and the daily cross-sections grouped series by array position, mixing dates): replayed in real time -- the window as known the day before each release, the outcome its first print (ALFRED 2002-2021; SHA-256 over the 11 CPI files' digests `cd7fc41dc9eebb9c8b5809f53557cfce8c5f59097705abd166793f8dbc437e4d`) -- 0.508 / 0.592 for the median and 0.524 / 0.597 for AR(1); daily returns grouped by date 0.562 / 0.578; AR(1) better in all 6 half-samples, so it is kept |
| v17 change-target switching (`t4agent/stats.py` `CHANGE_SWITCH` 0.75, `_change_switch_ok`) | an h-step change target leaves zero change only for a method whose backtest absolute error is at most 0.75 of zero change's and lower in both halves of the backtest (v16: 0.9 over the whole backtest) | model selection on the same public data: v16's rule left zero change for 63% of 20-day changes of daily yields, FX and oil and scored 0.473 / 0.472 against the zero-change rule; v17 0.483 / 0.482; trending monthly macro levels and CPI indexes 0.576 / 0.549 (v16) and 0.569 / 0.543 (v17). v19, every cross-section at one date or month: 0.472 / 0.476 (v16) and 0.484 / 0.492 on the random walks; 0.576 / 0.550 and 0.569 / 0.543 on the trends; unchanged |
| v17 positioning reversion; v19 horizon-aware (`t4agent/stats.py` `LEVEL_REVERT_DAILY` 0.003, `revert_share`, `RATIO_SWITCH` 0.5, `_ratio_change_forecast`) | a ratio change ('X % of Y', confirmed against the table) forecasts the share 1 - exp(-0.003 x horizon in days) of the gap between the ratio and its window mean, closing it (0.10 at five weeks, 0.24 at thirteen; v17: 0.15 at every horizon); the in-window drift replaces it only when its backtest absolute error is at most half the reversion's and lower in both halves (v17: no switching at all) | calibration and selection on CFTC legacy futures-only reports 2010-2021 for the ten markets of the Development positioning unit (public domain; files and SHA-256 in `calib_data/README.md`; SHA-256 over the 13 files' digests in name order `f659d847e79f600e30329d3e76d61ca322153ad215ccd3040dceae1f6c103be0`), 26-week windows, cross-sections every other week 2011-2021 (`tools/calibrate_methods.py`). v17: 283 five-week cross-sections, ranking quality (Spearman + 1) / 2 0.567 (0.583 / 0.551 by half) against 0.500 for v16 (mostly zero change, no ordering) and 0.479 for momentum; switching among all methods lost on both measures. v19 (review of v17: a forced reversion, the same at every horizon): the regression quality against zero change, averaged over horizons of 1-13 weeks, is flat for daily rates 0.003-0.004 (0.5028-0.5030; none 0.5000; the lower end kept); at 2 / 5 / 8 weeks ranking 0.532 / 0.565 / 0.575 (v17-v18 0.532 / 0.567 / 0.581, v16 0.525 / 0.500 / 0.496) and regression 0.501 / 0.501 / 0.498 (v17-v18 0.495 / 0.502 / 0.504). The drift rule costs 0.002 / 0.007 of ranking at 5 / 8 weeks (about one standard error; the change rule's 25% bar would cost 0.006 / 0.024) and follows a +1/week trend under weekly noise of sd 0.5 in 99% of synthetic windows; kept because the held-out set spans families with no published counterpart (README), where a ratio may trend. Latest documented data availability is early January 2022; v21.6 applies the conservative `statistics` bound 2022-02-01 |
| EPS anchor, v18; recalibrated v20 (`t4agent/eps.py` `PHI` 0.4, `CARRY_FAR` 0.16, `EPS_SCALE_NEXT` 1.97, `EPS_SCALE_FAR` 2.22, `EPS_P_UP_AFTER_UP` 0.75, `EPS_P_UP_AFTER_DOWN` 0.50; `t4agent/predict.py` `_blend_eps_direction`) | one quarter ahead the anchor carries 0.4 of the latest year-over-year change, further out 0.4 ** 2; its rule-based 90% half-width is multiplied by 1.97 one quarter ahead and 2.22 further out; direction labels that compare the target EPS with the prior-year quarter are blended equal-weight with P(up) = 0.75 after a latest quarter that rose and 0.50 after one that fell | calibration and selection on SEC EDGAR XBRL diluted EPS of large U.S. companies, facts filed before 2022-01-01: data for 145 companies were downloaded (146 files with the ticker-to-CIK map) and 134 of them contribute pseudo-forecasts (public data; files in `calib_data/README.md`, SHA-256 over the 146 files' digests `79929438d95c402139f8161d44c7b8b2ce64b5bd535342f88a77382556e0814c`), `tools/calibrate_eps.py`: pseudo-forecasts in the Development units' setting (base = prior-year quarter as first reported, latest information = the last 10-Q filed before the target's report). v20 (review of v18): of 5,318 cases, 151 compared a base and an outcome on different share bases (splits, restatements: the target's own filing prints a prior-year figure more than 20% away; 82 with the direction flipped) and 5 more carried XBRL scale errors; they are dropped (the comparative only decides comparability, it is never a feature). On the 5,162 kept: one quarter ahead carry 0.4 0.520 (0.525 / 0.515 by half; percent of the base 0.512), two or more ahead 0.16 0.508 against 0.500 for none (v18 set none on the mixed sample); direction 0.761 / 0.733 after a rise and 0.504 / 0.493 after a fall. Interval: the loss-minimising factors on the anchor's half-width, 1.97 and 2.22 (1.75-2.13 and 1.82-2.53 by half; each half's factor scores the other half within 2% of 1.8 except the later half's far factor, 6% worse). They apply as measured because the submitted EPS forecast is the anchor's: in five live runs of the Development EPS units the model departed from the anchor by a median 0-8% of the anchor's error and the submitted interval was the anchor's (v18's 1.8 assumed a blend erring 0.71-0.87 as much). Numerical fitting/selection facts were filed by the end of 2021; v21.6 applies the `eps` bound 2022-01-01. Company-universe metadata and the practice-based anchor-retention rationale above are separate limitations, not proved pre-cutoff by that date |
| v7 reasons (`t4agent/reasons.py`) | 1 request after the review, 900 thinking tokens, 60 s reserved; up to 14 candidate passages; premise <= 600 characters, cited context <= 1,500; own text <= 420 characters; byte caps 6,000 / 42,000 under the published 6,500 / 46,500 | the published `submitted_reasons` contract (shape, caps, deny list) with margins; fixed a priori |

## v21.5 changes (no new learned parameters)

Robustness fixes from an audit of v21.3 introduced no new numerical calibration. The ranking fallback choice
(B10) explicitly relied on the earlier outcome-scored CFTC replay; this is historical method selection, even
though it added no fitted coefficient. Its reversion inputs therefore remain under v21.6 `statistics` eligibility.
This is distinct from the Development practice-outcome exception above. Other rows state their own basis;
public-unit numbers changed in the listed cases (positioning ranks and post-earnings intervals in full fallback).

| Change (`t4agent/...`) | Constant | Basis |
|---|---|---|
| a claim on the entity's own task-table row when no document is citable (`answer.py` `task_table`, `task_claim`; `validate`, `claim_ok` accept it) | the shortest numeric `"key": value` pair of the row; at most `TASK_CLAIM_CHARS` 300 characters | the scorer's published task-table citation (`qfbench2_track_analysis.corpus.task_table_text`); an empty `claims` list refuses the whole unit; 300 characters stay under the judge's 400-token claim cap (`scoring.CLAIM_MAX_JUDGE_TOKENS`) at the judge tokenizer's measured maximum of about one token a character (CJK, punctuation) |
| a failing row or fallback stage still leaves an answer (`cli.py` `_rows`, `emergency_row`, `emergency_answer`) | none (the status-quo label with the row's quantity as `finalize` sizes it from the task, [0, 1] only when nothing numeric is known; or the naive value with the default half-width) | the existing rule-based fallback, applied per row |
| documents without a string `text` field are evidence and never cited by a claim (a reason may cite one); manifest labels read root manifest first (`corpus.py`) | none | the scorer slices claims from `text` only and reads the root manifest first; the reasons rail resolves the spans joined by a space (`citation_rail._doc_text`) |
| reply parsing and transport errors (`llm.py`); a planner that crashes leaves the analysts their run (`predict.py` `run_plan`) | none | robustness |
| every House call returns by the LLM deadline + 15 s, and is given up as a timeout once its own connect + read timeouts and 5 s have passed (`llm.py` `HouseClient._post`) | 15 s; 5 s | the per-read timeout does not bound a trickling reply |
| analyst calls in flight at the analyst deadline are waited for until the reasons gate when their group has an entity without a sample (`predict.py` `_late_replies`) | none (the existing reasons gate) | on a 300 s card the analyst deadline fell inside one House round trip; a model forecast outweighs a model-written reason |
| an unusable card time limit (huge, non-finite, non-positive, not a number) gives the default clock (`task.py` `card_timeout`, `cli.py` `run`) | none | robustness |
| clocks from the card's `[agent].timeout_sec` (`config.py` `derive_clock`); watchdog at the hard stop | hard stop = min(470, T - 90) s; LLM deadline = min(400, hard - 70) s; every absolute margin of the model path x LLM deadline / 400 | 90 s for image pull, start-up and slack; 70 s for final CPU work and a call in flight; T = 600 gives v21.3's 470 / 400 s and margins exactly; the watchdog fires at the hard stop (v21.3: 30 s later) |
| an announcement-window return with no numeric quantity sits around 0 (`calibrate.py` `quantity_range`) | none | the existing `EVENT_RETURN_SD` floor (v10) |
| deny-list check on the grader's folded text (`reasons.py` `fold`, `denied`) | the grader's look-alike lists | `citation_rail.fold_for_detection` |
| model reasons no longer need a model forecast: with none, the call goes only when the median wall time of the route's recent successful calls fits before the deadline (`llm.py` `typical_seconds`); code reasons: whole items, plain words, no file names (`reasons.py`) | median of the last 5 calls; 5 s slack | hygiene; the code template itself is unchanged |
| ranking rows without a model reply take their code baseline when those baselines come from one rule (the ratio-change reversion, or one method throughout; a series read off an unnamed table still sizes the interval only, as in v21.1); with neither, the middle of the others on a level-valued target (`predict.py` `_combine_with_stats`, `_baselines_rank`, `_place_uninformed`) | none | outcome-informed choice from the team's historical CFTC replay (positioning ranking 0.565 for the reversion rule, v19; v17: switching among methods lost, momentum 0.479 against the constant's 0.500); reversion inputs are gated by `statistics`. This is historical selection, not the Development practice-outcome exception. The level fallback placement also uses the definitional fact that 0 would rank a positive level last. |
| no EPS anchor for a return target (an announcement-window return, or 'reaction' / 'return' in the target name); other earnings targets as in v21.3 (`predict.py` `_eps_units`) | none | 'earnings reaction' is a stock return |
| constant House seed (`config.py` `SEED`) | 0 | the organizers' rerun sets `QFBENCH_SEED` afresh; 0 keeps every validated seed |
| sub-label match bounded (`facts.py` `SUBLABEL_SPAN`) | 400 characters | the longest match in the public filings is 66; outputs on the public units are byte-identical |

## v21.6 changes (eligibility policy; no refit)

`t4agent/artifact_policy.py` supplies the five immutable task-local capability decisions above.
Callers in prediction, calibration and historical-artifact constructors carry the policy; the
uncalibrated `runtime_naive.py` path replaces ineligible historical statistics. The calibrated
numbers, data checksums and local-model coefficient digest below/above are unchanged. Missing or
invalid task cutoffs do not activate artifacts. This section describes implementation intent and
disclosure, not a claim that all tests, container verification, real House evaluation or Final
acceptance have completed; those require separate versioned receipts.

## v21.10 changes (no refit)

Two changes from development branch `dx/t4-r2-e7a01ca` (commits `5cdb85b` and `e7a01ca`), plus one further restriction.

1. `t4agent/calibrate.py` `is_probability` decides whether a row's interval is submitted as a probability in [0, 1].
   Before, the planner's flag decided, and without one the word "probability" anywhere in the question did on a
   classification unit. The word in the question is no longer a signal. On a classification unit, [0, 1] now follows
   any of: the planner's flag set true; a probability type or unit, a 0/1 enum or numeric bounds of 0 and 1 in the
   target's value schema, or a probability unit shared by every row; the card's `interval_leg = false`; a `value`
   column whose every entry is 0 or 1. Otherwise a measured unit (ratio, bps, percent, level, ...) keeps the interval
   numeric.
2. `t4agent/predict.py` `_quantity_from_series`: a classification row whose model reply gives a label but no number,
   or that has no reply, takes the forecast of the series `find_series` already names from the target or the entity
   id, so a numeric quantity is not submitted as [0, 1].
3. Further restriction: the new signals in (1) decide classification units only. A regression or ranking unit takes [0, 1] only from
   the planner's flag, as in v21.8. Without this, a regression whose rows carry a column named `value` that reads 0 on
   every row (a level or change that happens to be 0) went out at 0 in [0, 1] even with the planner saying it was no
   probability (a constructed basis-point case: -20 [-50, 10] in v21.8, 0.0 [0, 1] on the branch).

The basis is the published rules, with no outcome and no proxy-judge score consulted. The track README says that when
a prompt asks for a numeric quantity, that number goes in `point_forecast` and `interval` in the prompt's units, and
that an interval on a probability never covers a dollar outcome; `docs/CONCEPTS.md` says a classification unit whose
numeric truth is only a 0/1 label code may declare `interval_leg = false` and is then scored on the label alone. So the
interval belongs on the number unless the unit's own schema or card makes its truth a probability or a label code, and
that case is a classification one.

Checks (no outcomes involved): on the 11 public and 16 pre-2022 stress units the offline answers equal v21.8's apart
from the version string. With each planner setting (no flag, false, true) the decision differs from v21.8's only on
`t4-credit-event-2023` when the planner says false; that unit's card declares `interval_leg = false`, so its interval
is not scored. Across 76 live unit runs of v21.7-v21.9 (the same decision code as v21.8), the only rows submitted as
[0, 1] were that unit's, and no classification row lacked a number. So the change guards formats not seen in these
units.

## v21.14 changes (no refit or parameter selection)

v21.14 is v21.13 plus the threshold consistency check described in "Hybrid threshold consistency on v21.13" below
(branch `review/t4-hybrid-dc` at `dcbaa91`). Its source stays in `agent_v21_11/`; `__version__` and the nowcast's
`models[]` row read 21.14.0. No fitted value, selected constant, prompt or clock default changed, and no outcome or
score was used. On the 11 public units, offline, 7 of 78 rows change their point forecast, all on classification
units, where the scorer does not score the point; labels, intervals, claims and reasons are unchanged. The image
keeps the 360/290 clock caps. Verification is recorded in `sub/verification-v21_14/`.

## v21.13 changes (no refit or parameter selection)

v21.13 is v21.11 with five robustness and citation fixes. Its source stays in `agent_v21_11/`; `__version__` and the
nowcast's `models[]` row read 21.13.0. No fitted value, selected constant, prompt or clock default changed. On the 11
public units the predictions (points, intervals and labels) are identical to v21.11's, both offline and against a fake
House (a local stand-in that returns fixed-shape JSON); the code-built reasons are identical too. The fixes answer
failures seen in live runs and the published claim rules; no outcome or score was used.

- Claims (`cite.py`). A claim is a whole sentence or table row of a document the entity may cite, ranked by a fixed
  rule: 3 if it names this entity, plus 1 if it carries a figure, minus 4 if it names another entity and not this
  one. Section titles are dropped. When nothing qualifies, the existing task-row citation fills the row. On the
  public units the claims fall from 312 to 255; the organizers' `check_claim_rules` finds no problem with either
  version, offline or against the fake House.
- Model-stage guard (`cli.py`). The model stage runs in a worker thread. The main thread waits for it until the LLM
  deadline + 20 s (at most the hard stop - 20 s), then exits keeping the last answer written. The model forecasts
  are written before the reasons call, so a reasons call that never returns no longer costs them. In a live run on
  2026-10-21 a stage that never returned left a whole unit on its fallback answer.
- In-flight requests (`llm.py`) are given up at the LLM deadline instead of 15 s after it, and no request starts after
  the deadline.
- Reasons (`llm.py` `salvage_array`, `reasons.py`). A reasons reply cut off by the 4,000-token output cap keeps its
  complete reasons. The thinking-free retry still runs when there is time, and replaces them only if it returns at
  least as many.
- EPS figures (`facts.py`). A hyphen directly before a digit or a decimal point is the figure's minus sign: "-0.40"
  had been read as +0.40. The EPS records read from the public units' corpora are unchanged.

The image keeps the v21.11 clock caps (H5 below). On a 600 s card the guard falls at 310 s, 50 s before the hard stop
at 360 s.

## Hybrid threshold consistency on v21.13 (no fit or parameter selection)

The `review/t4-hybrid-dc` branch adds a task-defined threshold consistency check to the
v21.13 pipeline, before writing predictions and again after answer-size rounding. It
keeps the chosen label and moves an inconsistent numeric point to the nearest legal
publication-grid value, preferring the existing interval's overlap with that label.
An interval expands only if it must contain the repaired point.

Quantity checks reuse `finalize`'s per-row probability decision, including its correction
of a planner-only probability flag on a numeric row. An interval equal to `[0, 1]` is not
itself evidence of probability units. Explicit percent and basis-point units take
precedence over an unrelated level baseline or generic change wording; another row's
units do not choose this row's units. Task thresholds, not planner label rules, define
the label regions.

This guard adds no fitted coefficients, forecast prior, family-specific point rule,
interval calibration, or model call. The v21.13 clock caps remain 360/290. Its tests use
synthetic tasks and public practice inputs without reading outcomes or scores. Local
verification is recorded in `sub/verification-hybrid-dc-unit-fix/report.md`.

### Quantity and assertion repair after 07fde14 (2026-10-23)

The guard now reads the current row's declared level, percent, or bp unit before shared
target/planner wording. Probability precedence still reuses `finalize`. Publication
precision follows that numeric quantity and the named reference; incidental units in
the prompt no longer choose it. Existing integer reference grids are retained.

Supported EPS offsets in cents/dollars and relative-percent offsets retain one
unambiguous task-row reference and their stated magnitude. Unresolved quantitative
assertions abstain instead of becoming zero-offset directions or `otherwise`.
Plain up/down reference comparisons, including the public macro revision tasks,
retain the direction fallback. Closed comparison words include their endpoint;
symmetric bands are parsed before one-sided cuts, and `falls to` is distinguished
from `falls by`. A fall/rise of at most X is bounded by zero on its named side.

Interval complements can contain multiple pieces. Repairs prefer the submitted
interval and then the nearest legal value measured from the original point.
At shared endpoints the point must belong to its submitted label's own region.
These are parsing/consistency changes only: no forecast prior, calibration parameter,
model call, score fit, or clock change is introduced. Regressions and public offline
verification are recorded in `sub/verification-hybrid-dc-semantics/report.md`.

## v21.11 changes (no refit or parameter selection)

No fitted value, selected constant or code clock default changed; the image sets lower-only clock caps (H5 below).
The numeric-target guard now decides ambiguous planner-true regressions per row, and the output filter also covers
task-row fallbacks and labels; both changes rely only on the official README:130, README:483-485 and README:202 text.
Round 2 generalises the same planner-true guard with field-name tokens plus date-value shapes and extends the existing
canary and measured-unit edge handling. It adds no fit, parameter selection or constant change.
Round 3 restores the c83a768/T4-002 pre-analyst rule for an unnamed regression only when every row has a
non-calendar naive value outside [0, 1], while mixed units are decided per row. Calendar rejection uses field tokens,
supported date shapes and quantity-field suffixes; level columns such as `latest_vintage_value` and
`latest_rate_at_date` remain levels. H1-H4 extend only the planner-error and output-sanitisation paths: percent forms
and heterogeneous measured units are recognised, probability columns require complete tokens or known glued forms,
and labels carrying a registered canary are omitted. These guards derive from the official README/CATEGORIES rules
for numeric units and contamination; no outcome, fitted value or selected constant was used.

H5 adds `T4_CAP_TIME_BUDGET` and `T4_CAP_LLM_DEADLINE` as lower-only production caps. Both default to `0` (off)
in code; the existing local-test clock overrides retain their semantics, and the image does not set them. The image
sets the caps in its Dockerfile ENV block, `T4_CAP_TIME_BUDGET=360` and `T4_CAP_LLM_DEADLINE=290`. On a card with
`timeout_sec = 600` the hard stop is 360 s (470 s uncapped) and the LLM deadline 290 s (400 s), and every absolute
margin of the model path scales by 290 / 400 = 0.725. A card with a shorter limit keeps its own, shorter clock. The
reason is the run's time limit, not forecast quality: the ingestion stage runs every unit in series within one
43,200 s clock (`SUBMISSION_CLI.md`), a unit it does not reach scores 0, and the Final's unit count is not published.
The values come from the team's clock table (hard stop = 43,200 s / unit count - an assumed 10 s per-unit overhead;
LLM deadline = hard stop - 70 s, the code's own margin), read at about 101-115 units. Only wall-clock arithmetic and
timings were used: no outcome, score or fitted value.

Round 4 requires both a pure calendar-role field name and a matching calendar value shape, including plural/glued
roles, Excel serials and common compact forms; missing/calendar-only rows no longer veto an otherwise unanimous
numeric-target decision. These flag-true guards add no fit, selected constant, Dockerfile change or P1-1 behavior.

## v21.8 changes (no refit)

One change, in `t4agent/answer.py` `fit_judge_cap`. The reasoning grader judges no reason when the per-entity answer it reads
exceeds 3,000 bytes; v14 rounded the numbers to fit only within 1% of an interval's width and 0.5% for a point, and never
touched a ranking. v21.8 keeps that tier first. When it cannot fit, a second tier allows an interval to widen by up to
5% of its width (outward rounding only) and a point to move by up to 2.5% of it (rounded to nearest, kept inside its
interval). A ranking is now rounded too, but only when every pair of its points keeps its order and its ties, so its
ranks cannot change. A roster too long to fit at any precision (about 37 rows or more: the field names alone fill the
cap) is left unchanged, as before.

The basis is the published scoring rules, with no outcome and no proxy-judge score consulted:
- over the cap the unit's reasons score 0 with certainty;
- the reasoning part of the Final score is a bonus that cannot go below 0, so fitting under the cap can only raise its
  expected value;
- the forecast cost is bounded. Widening an interval by at most 5% of its width raises its interval score (width plus
  20 times the miss) by at most 5% of that width, which lowers the unit's interval quality by at most about 0.25 x 5%,
  or about 0.005 on the Final score at worst;
- a point rounded to nearest by at most 2.5% of the width is an unbiased move, so it changes the expected absolute
  error only to second order.

The tolerances are round numbers fixed a priori. On the public and stress units every answer is under the cap or
cannot fit at all (a 51-row roster), so no submitted number changes there.

## v21.7 changes (no refit)

v21.7 is v21.6 with one build change. The image sets `PYTHONHASHSEED=0`: Python randomises set iteration order in
every process, and fixing it keeps the prompts the same when the organizers rerun the Final on a fresh `QFBENCH_SEED`.
This complements the constant House seed (v21.5). No constant, capability bound or code path changes, and the public
units' answers are v21.6's. The parallel working-folder builds pushed as Docker Hub tags `21.5.0` and `21.6.0`
(a calibration gate with v14's rules as the fallback) are superseded by this policy and are not this source.

## Runtime statistics (v2; qualified by v21.6 eligibility)

`t4agent/stats.py` backtests simple forecasters (last value, recent means, median, exponential
smoothing, drift; zero change, drift, momentum, reversal for change targets) on each entity's own
series in the unit's admitted corpus tables at run time when `statistics` is eligible. The series
values are task-local; the candidate restrictions, defaults, corrections and scale table include
the historical fitting/selection described above. No runtime state is carried between units.
When that capability is unavailable, only the separately described generic corpus rule is used. `t4agent/facts.py` and the risk-indicator counts are fixed regular expressions; the v3 like-for-like lines read numbers from the entity's own filings in the frozen corpus and map columns by the table's own header dates.

## Data

No external dataset files are included in the image or fetched at evaluation time. Historical
external data were used offline for the disclosed fitting/selection, and eligible constants are
embedded in source. At evaluation time the agent reads only
`/input/task.json`, `/input/card.toml` (scoring parameters; from v21.5 also the unit time limit
`[agent].timeout_sec`, nothing else of the card) and the manifest-declared corpus
documents dated on or before `cutoff_date`. Prompts contain only the task, the dated corpus and
statistics computed from it, plus descriptions of eligible historical calibrations; they contain
no stored example outcomes.

`T4_EXTRA_BODY` (extra request fields) and `T4_REASONING_FIELD` (a local route's thinking-budget field), like every
other `T4_*` variable apart from the v21.11 clock caps `T4_CAP_TIME_BUDGET` / `T4_CAP_LLM_DEADLINE`, are
local-testing hooks: the harness never sets them, the image does not set them, and with
them unset the request body is the House contract's (`tests/test_offline.py::test_local_reasoning_field_translation_is_inert_in_production`).

## Build

`Dockerfile`: `python:3.13-slim`, pinned `requirements.txt` (requests, urllib3, certifi, idna,
charset-normalizer), `t4agent/` source. `linux/amd64`, `LABEL qfbench2.interface_version="2.0"`, no `VOLUME`;
v21.7 sets `PYTHONHASHSEED=0`; v21.11 sets the clock caps `T4_CAP_TIME_BUDGET=360` and `T4_CAP_LLM_DEADLINE=290`.
