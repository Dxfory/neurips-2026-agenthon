# Numerical source and calibration provenance

## Release v2.7.7 (`forecast-v2_7_7`)

v2.7.7 is v2.7.6 with one change: the spread of the monthly sibling nowcast (E40b). Its centre, its sibling choice
and the draws' shape stay. No text is read and no model API is called. `--profile v2.7.6`
(`Settings(**V2_7_6_SETTINGS)`) reproduces v2.7.6 draw for draw on every card.

| Change | Setting (default) | Cards it reaches | Evidence |
|---|---|---|---|
| The nowcast's spread σ × √steps × 1.1 becomes √(σ²/2 + s24²/2) × √(k + k²/n + (X − k·x̄)²/Sxx) × c. s24 is the RMS of the window's last 24 residuals; the second root is the OLS prediction variance of the k-month sum; c is per target kind: index level 1.2, persistent rate 1.1, monthly change 0.9 | `nowcast_width` (on), `NOWCAST_RECENT`, `NOWCAST_KIND_MULT`, `NOWCAST_KIND_RATIO`, `nowcast_target_kind` | monthly cards where E40b fires | `exp/README.md` E41 |

### Selection data and procedure

- **E41** (pre-registered 2026-10-19). An outside review had measured E40b's standardized errors: SD 1.04 at train,
  1.50 at test, 0.91 in 2024.
  - The design saw train diagnostics, and the test era only overall.
  - Tasks: E40's roster `nowcast40` where E40b fires (825). Data: ALFRED vintages to 2024-12-18; targets are first
    releases.
  - c: for each type, the multiple on a 0.8–2.5 grid with the lowest mean score on the 482 tasks of 2010–18.
  - Test (2019–24, no covid, 296 tasks): −0.0362 [−0.0664, −0.0110]. No type worse than +0.001.
- **Target kind.** It is read from the series itself: the median absolute monthly change over the median absolute
  value, last 120 months.
  - Under 0.01, with every value positive: an index level.
  - Under 0.2: a persistent rate. Otherwise: a monthly change.
  - On all 2,309 price and payroll tasks of the roster it equals the task's type.
- **Reproduction.** The release equals E41's `full` arm on all 2,475 forecasts (|score difference| ≤ 2.4e-13).

### Cutoff and rule boundaries

- c uses outcomes from 2010–18.
- The nowcast's regression, residuals and target kind are computed on each card's own panel at run time.

## Release v2.7.6 (`forecast-v2_7_6`)

v2.7.6 is v2.7.5 with two steps applied after its forecast. No text is read and no model API is called.
`--profile v2.7.5` (`Settings(**V2_7_5_SETTINGS)`) reproduces v2.7.5 draw for draw on every card.

| Change | Setting (default) | Cards it reaches | Evidence |
|---|---|---|---|
| Treasury-note and bill cells shift by κ × their SD on cards the FOMC-week profile reaches (κ notes −0.136134, bills −0.248081) | `fomc_mean_shift` (on), `FOMC_MEAN_KAPPA` | FOMC-week Monday cards with rates targets, steps 1–4 | `exp/README.md` E37 |
| A monthly target is nowcast from a panel series that already has its target month, with a stable correlation of monthly changes (OLS over 120 months; \|corr\| ≥ 0.5 overall and in each half; spread = residual SD × √steps × 1.1; level space) | `monthly_nowcast` (on) | monthly cards whose panel carries such a sibling | `exp/README.md` E40, E40b |

### Selection data and procedure

- **E37** (pre-registered 2026-10-19).
  - Roster `ps1shape`: the official short-horizon shapes on every FOMC Monday, 2005–24 (FRED UST, ECB reference
    rates crossed to USD).
  - κ is the 2005–18 mean of (outcome − median)/SD over rates cells.
  - Test (2019–24), rates tasks: −0.0411 [−0.0727, −0.0105]. Train −0.0122. No class worse.
  - The placebo on non-FOMC Mondays: +0.0074 [−0.0305, +0.0451].
  - The diagnostic that motivated E37 had seen 2019–24, which is why κ uses 2005–18 only.
- **E40** (pre-registered) failed its guard.
  - Its log-space step skipped some seeds when a YoY rate's draws crossed zero.
  - In covid windows it could pick a spurious sibling.
- **E40b** (pre-registered after E40, with exactly those two corrections) passed all rules on E40's roster.
  - Data: ALFRED vintages to 2024-12-18, real-time panels, first-release targets.
  - Test (2019–24) where it fires: −0.4949 [−0.7593, −0.2677]. Train −0.4437.
  - It never fires without a sibling holding the target month.
- **Reproduction.** On sampled tasks, the release equals the E37 wrapper's `fomc_mean` arm and the E40 wrapper's
  `nowcast` 2 arm (40 + 40 tasks).

### Cutoff and rule boundaries

- κ uses outcomes from 2005–18. Every other number comes from data at or before 2024-12-18.
- E40b fits nothing in advance: its regression runs on each card's own panel at run time.

## Release v2.7.5 (`forecast-v2_7_5`)

v2.7.5 is v2.7.4 with one change. No text is read and no model API is called. `--profile v2.7.4`
(`Settings(**V2_7_4_SETTINGS)`) reproduces v2.7.4 draw for draw on every card, and every earlier profile inherits it.

| Change | Setting (default) | Cards it reaches | Evidence |
|---|---|---|---|
| A monthly target whose last 96 changes differ by calendar month (Kruskal–Wallis p < 0.001) is forecast seasonally adjusted, and its draws are put back on the calendar | `monthly_seasonal` (on) | monthly cards whose target is published without seasonal adjustment (index levels, month-on-month rates); never a seasonally adjusted series, a year-on-year rate or a daily card; no public card | `exp/README.md` E35 and E36 |

### Selection data and procedure

- **Data.** ALFRED (FRED API), every vintage up to 2024-12-18 and nothing after it.
  - Each task's panel is the vintage in force at its as-of: the Monday on or before the 26th of each month,
    2010-01 to 2024-11.
  - Its target is the first release of the next figure.
  - Scores are divided by M0's expected error from the unadjusted panel, as the board's are.
- **E35** (pre-registered; 16 NSA series: US CPI and core CPI, the euro area, DE/FR/IT/ES HICP, UK/CA/JP CPI, and
  US unemployment, payrolls, housing starts, retail sales and industrial production).
  - Result: test (2019–24) −0.199 [−0.339, −0.066], train (2010–18) −0.346.
  - Rule 3 (no series worse than +0.010) failed on Germany (+0.098, January policy shocks) and Japan (+0.024), so
    the step was not taken on E35 alone.
- **E36** (pre-registered before its data were downloaded; the step unchanged; 20 series E35 never used: HICP for
  NL, AT, BE, PT, IE, FI, GR, SE, DK, PL and CZ; CPI for CH, NO, KR and MX; US gasoline and food-at-home CPI; US
  vehicle sales, employment and labour force).
  - Result: test −0.188 [−0.271, −0.112], train −0.190; 19 of 20 series gain, the worst is FI +0.017.
  - All rules pass, and the step was taken.
- **Guard.** The step fires on none of the 2,488 `realtime33` tasks or the 2,750 `monthly26` tasks (all seasonally
  adjusted), and on no year-on-year target.
- **Detector and effects.** The test (Kruskal–Wallis at p < 0.001 on the last 96 changes) and the effect estimate
  (trimmed calendar-month means, centred) were fixed before any E35 forecast. They were never tuned on outcomes.
- **Reproduction.** The release's draws equal E35's research copy (`output/runs/2026-10-18-e35/forecast-v2_7_5rc.py`,
  sha256 `76dbbebe…`, step on) on sampled tasks. With the step off they equal v2.7.4.

### Cutoff and rule boundaries

- No setting of v2.7.5 is fitted. The 96-change window and the p < 0.001 threshold were fixed in advance.
- Every number used to decide on the step comes from data at or before 2024-12-18 (project rule, AGENTS.md).

## Release v2.7.4 (`forecast-v2_7_4`)

v2.7.4 is v2.7.3 with the follow-ups to the review of 17 October 2026. No text is read and no model API is called.
`--profile v2.7.3` (`Settings(**V2_7_3_SETTINGS)`) reproduces v2.7.3 draw for draw on every card. Every earlier
profile inherits that.

| Change | Setting (default) | Cards it reaches | Evidence |
|---|---|---|---|
| Day counts on non-monthly panels are taken in nanoseconds whatever the stored resolution | none (`epoch_ns`) | none under the pinned pandas 2.3.3; weekly cards and seed digests under pandas 3 | the 104 public cards draw the same under pandas 2.3.3 and 3.0.5 |
| UST-notes cells keep their family widths under the FOMC-week profile when the card's lowest UST yield is at least 1% | `ust_family_widths_active` (on), `ust_active_floor` 1.0 | FOMC-week cards with a UST-notes target away from the zero lower bound, the ranked set among them | `exp/README.md` E27 and E29 |
| On a fresh monthly card whose every target reads as a price index and whose recent-to-long volatility ratio is at least 1, the M0-centred reweighting is skipped | `monthly_price_vol_switch` (on, the user's call), `monthly_price_vol_ratio` 1.0 | monthly price-index cards in a high-volatility regime | `exp/README.md` E26, E28 and E30 (post hoc) |
| M0's step count overrides the declared key only at a ratio strictly above 2 | none (`_m0_step_count`) | weekly-type panels whose key is exactly twice the print count | official commit `24582e9` (M0-BASELINE §3.7 step 5, 2026-10-17) |

### Selection data and procedure

- **E29** chooses between two arms already tested in E27: `ev` and `ev_ustk`, the profile on the family widths
  for UST-notes cells.
  - The signal (the card's lowest UST yield at the as-of) and the threshold (1%) were fixed from the mechanism
    before any result was split by them. Decisions at the zero lower bound barely moved Treasuries.
  - Data:
    - FRED DGS3MO–DGS30 and the E24 pool, everything at or before 2024-12-18;
    - the primary set is `pair2`'s FOMC-week tasks with a UST note (E25 roster);
    - the secondary set is `fomcweek` (E24).
  - Result on the primary set: test (2019–24) −0.0287 [−0.0534, −0.0066], train (2005–18) +0.0012. Rules 1–3
    pass.
- **E30** restricts E28's volatility switch to price-index targets. That restriction was chosen after E26's and
  E28's series tables were read, so it is post hoc.
  - Data: the practice monthly panel of `t2-F1-sahm-watch-2024` (ALFRED vintage 2024-08-02, months to 2024-06),
    roster `monthly26`.
  - Result: primary (2019–24, no covid) −0.1370 [−0.2893, −0.0089], train +0.0042. Rules 1–3 pass; post hoc, it
    ships on the user's call of 18 October 2026.
  - Real-time check (E33, pre-registered).
    - Data: ALFRED vintages to 2024-12-18. Each panel is as published at its as-of, and each target is the first
      release of the next figure.
    - Targets: index levels, year-on-year rates and month-on-month rates.
    - Result: primary (2019–24, price-index tasks) −0.1044 [−0.2095, −0.0138], train +0.0060. The rules give P-P,
      so the switch is not confirmed; it stays on by the user's call.
- Every switch evaluation reuses saved forecasts. v2.7.4's own draws equal the evaluated arm on sampled tasks:
  - E29: 60 of 60, against the E27 wrapper;
  - E30 switched on: 40 of 40, against v2.7.3 with `monthly_sharpen_alpha` 0 where it fires.

### Cutoff and rule boundaries

- Every number used to choose a v2.7.4 setting comes from data at or before 2024-12-18 (project rule, AGENTS.md).
- The switches read only the card's own inputs at run time: the UST levels at the as-of, and the target's
  monthly history up to its last observation.
- The 2026-10-28 FOMC date in `FOMC_WEDNESDAYS` is the Federal Reserve's published schedule, not an outcome.

## Release v2.7.3 (`forecast-v2_7_3`)

This release makes two changes to v2.7.2 (parent source `forecast-v2_7_2/forecast.py`, SHA-256
`299a2706eacb0676200c808ff0ab1cbcdfc06ba80fd37efb3087050751d5791f`; this source SHA-256
`754434d45afef0947e7714f0c2ffd9b534be79800a1fdffdce8703bb8213c42f`):
- the FOMC-week variance profile of experiment E24 (`../exp/README.md`, arm `ev`). The ranked Final's cards share
  the as-of Monday 2026-10-26, two days before the FOMC decision of Wednesday 2026-10-28;
- `pair_match` off by default (experiment E25): a default flipped, with no new data or fitted value.

- **Trigger** (`event_profile_reason`). All of these must hold, or the card is v2.7.2 unchanged:
  - a daily `level` card;
  - its as-of is the Monday of a week whose FOMC decision falls on the Wednesday (`FOMC_WEDNESDAYS`);
  - every step lies in 1–4;
  - every target is a Treasury tenor, a bill or one of the ranked Final's 15 exchange rates
    (`event_asset_class`). `UST_<n>M` and `UST_<n>Y` below 2 years are bills; from 2 years they are notes. The
    ten G10 codes and MXN, BRL, ZAR, KRW and INR count as a bare code or as an unambiguous pair with USD
    (`USD_MXN`, `USDMXN`, `EURUSD`, `EUR/USD`). Any other or ambiguous name means no profile;
  - every target's history reaches within 5 days of the as-of.
- **Family widths reset** (`FAMILY_NEUTRAL`). Where the profile applies, k is 1.0 for every family and F4 runs at
  the default engine width (no `f4_widen`), so the card's family label does not change its forecast.
- **Profile** (`EVENT_PROFILE`, `apply_event_profile`). Per asset, in yield units for Treasuries and log units
  for FX, so the quote direction does not matter:
  1. each cell's deviations from its median are differenced into increments, in horizon order;
  2. the increment over steps (h_{k−1}, h_k] is scaled by √(mean of m over those steps);
  3. the increments are cumulated back and each cell is re-centred on its median.
  A single cell at h is scaled by √(mean of m over s ≤ h), and cross-asset correlations are kept. On two-cell
  cards the profile comes before pair matching, so the pair statistic is still matched to M0's. Elsewhere it
  comes last, after the forward increment (a constant per cell), where E24 applied it. (Pairing runs only when
  `pair_match` is on; from this release it is off by default.) If anything is unexpected
  (a non-positive FX draw, a non-finite value) the draws are left unchanged and the reason is recorded.
- **Pairing off** (`pair_match` False). A two-cell card's single variogram pair is no longer matched to M0's own
  500-draw value. The matching was built for the rule before 6 October 2026, whose joint divisor was M0's
  realised pair statistic. Since then the divisor is M0's expected variogram score, and E25 (pre-registered, roster
  `pair2`: 7,647 two-cell tasks on the E24 pool, data to 2024-12-18) found the engine's own dependence better:
  - FOMC weeks: test −0.0224 [−0.0375, −0.0091], train −0.0252;
  - ordinary weeks: test −0.0334, train −0.0124;
  - no group worse (bills −0.13); the gain is mostly in the joint term.
  The failure paths (`m0_fallback`, `emergency_write`) still pair a two-cell card's M0 draws, as before.
- **Reproduction.** `V2_7_2_SETTINGS` (`event_profile` False, `pair_match` True; CLI `--profile v2.7.2`)
  reproduces v2.7.2 bit for bit. `V2_7_1_SETTINGS` and every earlier profile carry both through it;
  `POST_PROCESSING_OFF` turns both off. The four comparison profiles (`prospective` … `random-leaning`) take
  v2.7.3's two defaults, so on a profiled card they equal the default, and two-cell cards are not paired.

Nothing is bundled beyond the 16-number table and the calendar: no data, labels, learned weights, task
answers or per-card values. No text is read and no language model is called.

### Why

v2.7.2 has no calendar. Apart from √h its width is flat over h = 1–4, set per family (F1 k 1.1, F2/F3 k 1.2, F4
engine width 1.1). The ranked window holds the FOMC decision at 18:00 UTC on Wednesday 28 October:
- the UST close (about 3:30 pm ET) sees it at h = 2;
- the ECB fixing (13:10 UTC) sees it at h = 3.
On 2019–2024 FOMC weeks, single cells are too wide early in the week and too narrow late in it: 90% coverage is
0.95 / 0.94 / 0.86 / 0.88 at h = 1–4.

### Selection data and procedure

- **Estimation sets.** The profile is estimated on 2005–2018 only:
  - 82 FOMC weeks with a Wednesday decision, with the as-of on the Monday;
  - reference weeks: every other Monday of 2005–2018 whose Monday–Friday holds no FOMC statement.
- **Estimator.** For each asset and week, z = the one-step change of the state (yield level; log FX) divided by
  v2.7.1's own current step sd at the as-of (`scale_estimate` with blend 1.0, the 0.97-EWMA of the last 60
  innovations). ĉ solves mean[2φ(z/ĉ) − 1/√π] = 0, the scale at which a Gaussian forecast minimises the mean CRPS.
  Then m(c, s) = (ĉ(FOMC weeks, class c, step s) / ĉ(reference weeks, class c, steps 1–4 pooled))², clipped to
  [0.5, 2.5]. The classes are UST notes (2Y–30Y), bills (3M, 6M, 1Y), G10 and EM (MXN, BRL, ZAR, KRW, INR).
- **Data.** FRED DGS3MO … DGS30 (H.15, the Treasury par curve; the 2Y–30Y series equal the practice panels on all
  6,245 common dates) and the ECB's euro reference rates cross-calculated to USD. Nothing after 2024-12-18 enters.
- **Calendar.** The 127 FOMC statements on a Wednesday in E23's list (2005–2024, scheduled and unscheduled), plus
  2026-10-28 from the Federal Reserve's published 2026 meeting schedule (federalreserve.gov, "October 27-28"; read
  2026-10-17). The schedule's other 2026 meetings are not in the calendar.
- **Selection.** Reset plus profile is E24's pre-registered arm `ev`, which passed all three rules: test change ≤
  −0.005 with the 189-business-day block-bootstrap interval below zero; train change ≤ +0.005; no group worse than
  +0.010 at test. Two other arms failed and are not taken: `bills0` (no M0-t mixture on bills) and the stage-2
  `ev_k` (the profile on the family widths).

| Class | step 1 (Tue) | step 2 (Wed) | step 3 (Thu) | step 4 (Fri) |
|---|---:|---:|---:|---:|
| UST notes | 0.6377 | 1.6493 | 1.0908 | 1.1167 |
| bills | 0.7790 | 1.6234 | 1.4336 | 1.0154 |
| G10 FX | 0.8338 | 0.7571 | 2.5 (2.589 before the clip) | 1.0216 |
| EM FX | 0.9210 | 0.6542 | 2.5 (2.729 before the clip) | 1.0860 |

### Results of the released source

E24's `fomcweek` roster: 6,233 tasks, the Monday as-of of every Wednesday FOMC week from 2005 to 2024, h = 1–4,
24 assets in single-cell, path, curve and FX-basket shapes. Seeds 0–2, 4,096 draws, local M0. Scores use the
6 October 2026 rule: each component ÷ M0's §5 expected error, clipped at 8. v2.7.2 equals v2.7.1 on every one of
these daily tasks.

| Set | M0 | v2.7.2 | v2.7.3 | v2.7.3 − v2.7.2 (95% interval) |
|---|---:|---:|---:|---|
| Test, as-of from 2019 (2,290 tasks, 45 weeks) | 1.129 | 1.087 | 1.072 | −0.0140 [−0.0226, −0.0049] |
| Train, as-of to 2018 (3,943 tasks, 82 weeks; in-sample for the table) | 1.125 | 1.078 | 1.065 | −0.0134 [−0.0209, −0.0039] |

Notes on the table:
- By group at test: UST notes +0.008, bills −0.002, G10 −0.031, EM −0.019, full curve +0.006, 15-currency
  basket −0.027.
- By horizon, on single cells at test: −0.048 / −0.034 / +0.011 / +0.004 at h = 1–4.
- 76% of test weeks improve. Shock weeks lose 0.10–0.15 per task.
- **Identity checks** (seed 0, 4,096 draws):
  - With `pair_match` on, which isolates the profile: all 104 public cards equal v2.7.2's draws, and
    `forecast_meta.json` apart from the version. On the E23 short roster 1,593 of 1,600 tasks do; the profile
    fires on the other 7 (an FOMC-week Monday as-of with h = 2), and 6 of them improve, by −0.060 on average.
  - At the default (`pair_match` off), exactly the 19 two-cell public cards and the 280 two-cell short-roster
    tasks change on top of that. Nothing else does.
  - On `fomcweek`, which has no two-cell task, the draws equal the E24 wrapper's `ev` arm on all 6,233 tasks,
    and the harness scores equal the stored arm exactly over 3 seeds.
- E25's own check on this release (pairing on against off with the profile active, on FOMC-week two-cell
  tasks) is run by the experiment (`exp/e25_pair.py`).

### Cutoff and rule boundaries

- Every outcome used to estimate or check the profile is dated on or before 2024-12-18 (repository `AGENTS.md`),
  and the table uses only 2005–2018. The ranked Final's as-of (2026-10-26) is after all of them.
- The only date after 2024-12-18 in the source is the published 2026-10-28 FOMC decision. It is a public schedule
  announced before the Final opened, not an outcome.
- The trigger reads only the card's as-of, steps and target names and the unit's own panel dates.

## Release v2.7.2 (`forecast-v2_7_2`)

This release repairs four input-shape gaps of v2.7.1 (parent source `forecast-v2_7_1/forecast.py`, SHA-256
`a3dca5aa8985c16189ccca10c59392d258691558273a76b208dde23f1271c504`; this source SHA-256
`299a2706eacb0676200c808ff0ab1cbcdfc06ba80fd37efb3087050751d5791f`) for the ranked Final cards announced on
2026-10-12 (track2-forecasting-public issue #4):

- **Monthly freshness allowance.** `monthly_publication_lag_days` goes from 45 to 120 days. A monthly series
  is stale when its last period is earlier than the month 120 days before the as-of (2026-06 at the Final
  as-of).
- **Hole rule.** On non-monthly panels a hole is a step longer than max(5, 1.5 × median spacing) days
  (`gap_limit_days`), in the innovations, the horizon shape and both residual-adjacency tests. Daily panels keep
  5 days.
- **Weekly step count.** On a target spaced more than 5 and at most 20 days, the walk counts prints from the
  last observation to the as-of plus the key in US business days, and takes that count when it is 2× or more
  away from the key (M0 §3.7 steps 3–5). This applies in the engine (`load_task`) and in the embedded M0
  (`_m0_step_count`).
- **Next figure.** A monthly card with one key below 21 and no observation month walks one step
  (`_next_figure_card`).
- **As-of.** The card's as-of is read as the scorer reads it, `[forecast].asof` before
  `[provenance].data_cutoff` (`declared_asof`), in `load_task` and `emergency_write`. The embedded M0 uses it,
  else the task's as-of.

`V2_7_1_SETTINGS` (CLI `--profile v2.7.1`) reproduces v2.7.1 bit for bit on every card whose panels are daily or
monthly, and `V2_7_SETTINGS` and the earlier profiles now pin the 45-day allowance too. The cadence and as-of
rules have no switch. All 104 public cards give the same `forecast.parquet` and `forecast_meta.json` as v2.7.1,
apart from the version string. No data, labels, learned weights, task answers or per-card values are bundled, no
text is read, and no language model is called.

### Why

The ranked cards share one as-of, Monday 2026-10-26: UST yields and ECB-derived FX at 1–4 business days, plus a
few cards on the next figure of a monthly official statistic released 27–30 October. Issue #4 names GDP,
inflation and jobless claims. A team audit of 2026-10-17 ran v2.7.0 and v2.7.1 on 21 synthetic cards in the
staged Final layout (issue #20) and on backtests. It found three gaps that no public card exercises:

- **Stale monthly cards.** With the 45-day allowance the expected latest period at the Final as-of is 2026-09.
  A data-release panel ending in August (US PCE for September, Canada's August GDP) was therefore a stale
  "transfer" series. It got no drift, a 0.25 log-sd scale noise, no M0-t share and no sharpening, and scored
  about 2× M0. The public monthly panels end within 45 days of their as-of, so this never fired on them.
- **Weekly targets.** The 5-day hole rule made every step of a weekly panel a hole. The engine then had no
  innovations and fell back to an sd of 0.1 % of the level, with no exception to trigger the fallback.
  Separately, the embedded M0 kept the declared key on any panel spaced 20 days or less. Official M0 counts
  prints to its sealed target date, which is one print for a next-week target.
- **The as-of key.** The scorer reads `[forecast].asof` first (`cutoff.trusted_asof`). The embedded M0 required
  `[provenance].data_cutoff`, so on a card without it the M0-t share, sharpening, pair match and M0 failure
  fallback were all lost.

### Selection data and procedure

Nothing is fitted:
- 120 days is the round allowance that covers releases two to three months behind the as-of month; the
  validated range of the setting is 0–180.
- 1.5 × the median spacing separates one missing print from a regular step. M0's own hole rule is
  max(10 × median, 5).
- The step count and the 2× rule are M0's (§3.7).
- The next-figure reading follows the README's description of the data-release cards. M0-BASELINE §3.7 says
  the Final's monthly cards name their observation month, so this is a fallback only.

The audit's scripts (`build_cards.py`, `run_cards.py`, `bt_short.py`, `bt_cadence_p.py`; kept with the
2026-10-17 audit note outside the repository) were re-run on this source: in-process for the backtests, and by
the CLI on a staged copy of each synthetic card's `/input` for the card run. Data are dated on or before
2024-12-18: the rates and G10 panel cache, FRED H.10 EM rates and the public macro panel. Seed 0, 4,096 draws.
Scores use the 6 October 2026 rule: each component ÷ M0's §5 expected error, clipped at 8. Every arm the audit
also ran gives its numbers task for task. That includes the weekly prototype, which this source equals on all
690 cadence tasks.

### Results of the released source

| Set | M0 | v2.7.1 | v2.7.2 | v2.7.2 − v2.7.1 (95% interval) |
|---|---:|---:|---:|---|
| Daily, 1–4 BD (900 tasks: UST, G10, EM FX; as-of 2010–2024) | 1.073 | 1.037 | 1.037 | 0 on every task |
| Monthly next print (1,032 tasks: CPI gap 1, PCE/NFP gap 2, UNRATE gap 3; as-of the Monday before the 26th, 2010–2024) | 1.238 | 1.464 | 1.055 | −0.409 [−0.463, −0.360] |
| Weekly key 1 (153 tasks: Friday closes of UST 2Y/10Y, EUR, JPY, MXN, INR; as-of the next Monday) | 1.151 | 1.291 | 1.101 | −0.189 [−0.277, −0.110] |
| Weekly key 4 (147 tasks) | 0.925 | 0.993 | 0.883 | −0.110 [−0.133, −0.086] |
| Quarterly, named quarter (390 tasks) | 0.780 | 0.712 | 0.712 | 0 on every task |

Notes on the table:
- Monthly by shape: PCE_CORE 2.747 → 1.261, PCE_ALL 1.448 → 0.978, NFP 1.077 → 0.679, UNRATE 1.038 → 0.937;
  gap 1 is unchanged. Cards at the clip fall from 24 to 14; M0 has 25.
- Weekly against M0: −0.049 [−0.097, −0.010] at key 1 and −0.042 [−0.073, −0.015] at key 4.
- The intervals are a task bootstrap of 2,000 resamples.

**Synthetic Final-shaped cards.** All 21 runs exit 0 with the three files written and no numeric fallback, at
most 12 s wall each. All are admissible under the current HEAD scorer except `syn-F1-noprov`, which states no
as-of; the scorer refuses it as an organizer fault under both releases. Fourteen cards are byte-identical to
v2.7.1. The others, normalised as above:

| Card | M0 | v2.7.1 | v2.7.2 |
|---|---:|---:|---:|
| `syn-F1-pce-gap2` | 1.106 | 1.963 | 1.169 |
| `syn-F4-pce-gap2` | 2.085 | 3.813 | 1.828 |
| `syn-F1-unrate-gap3` | 0.579 | 0.455 | 0.398 |
| `syn-F1-claims-weekly` | 0.643 | 0.681 | 0.637 |
| `syn-F1-forecast-asof` | 0.861 | 0.860 | 0.880 |

`syn-F1-pce-noperiod` has no outcome. It now walks one month and is fresh, where v2.7.1 walked four and was
stale.

### Cutoff and rule boundaries

- Every outcome used to check the release is dated on or before 2024-12-18 (repository `AGENTS.md`). The
  ranked Final's as-of (2026-10-26) is after all of them.
- The step count reads only the unit's own panel dates, its as-of and the US federal holiday calendar.
- The as-of comes from the card. `[metadata].category` is read as before for the per-family pairs (issue #20).

## Release v2.7.1 (`forecast-v2_7_1`)

This release changes two defaults of v2.7.0 (parent source `forecast-v2_7/forecast.py`, SHA-256
`48b1c3a24ca02c473a05f64589a05c95e257639d2b90fa7dae904c9c87e59bb1`; this source SHA-256
`a3dca5aa8985c16189ccca10c59392d258691558273a76b208dde23f1271c504`): fresh daily shock-family (T2-F4) cards go
back to v2.6.0's engine width 1.1 (`f4_width_scale`) and k 1.0 (`sharpen_f4` (0.0, 1.0), the identity, so the
sharpening step is skipped on them). F2 and F3 keep k 1.2, F1 keeps 1.1, and the v2.7 forward increment stays.
`V2_7_SETTINGS` (CLI `--profile v2.7`) reproduces v2.7.0 on all 104 public cards; `V2_6_1_SETTINGS`,
`V2_6_SETTINGS` and the earlier profiles reproduce their releases. 29 of the 104 public cards differ from v2.7.0
(all of them fresh daily F4 cards, each narrower); the other 75 equal it. No data, labels, learned weights, task
answers or per-card values are bundled, no text is read, and no language model is called.

### Why

The organizers' notice of 2026-10-11 (track2-forecasting-public issue #4 and the answer on issue #26) ranks the
Final on cards built after the close from data up to 2026-10-23: UST and G10 FX series, horizons of 2 to 12
business days, families and wording by fixed rules with no window chosen on outcomes, and an F4 card defined as
one whose text carries a live risk channel or a scheduled event inside the window. Every public card and every
earlier local roster task had a horizon of 21 business days or more, and v2.6.1's F4 widening (engine width 1.3,
k 1.1) had been chosen for hindsight-picked shock cards. At the ranked horizons v2.7.0's F4 forecasts are too
wide (50% / 90% coverage 0.68 / 0.96 on 2019+ cutoffs, 0.73 / 0.99 before) and score worse than M0.

### Selection data and procedure

Experiment E23 (`exp/README.md`; `exp/e23_short.py`; runs under `output/runs/2026-10-11-e23/`), designed and
written down before any run. The `short` roster takes every public card shape on `rates_daily` and
`g10_fx_daily` in fixed short-horizon variants (2, 5, 8, 12; (2, 7), (3, 10), (5, 12) business days), 800 tasks
per era (train = cutoffs 2005–2018, test = 2019 on; latest outcome 2024-12-18), with the daily roster's panel
pool, label policy and quota logic. Rules against v2.7.0: test change ≤ −0.005 per task with the
189-business-day block-bootstrap interval below zero; train not worse than +0.005; no family worse than +0.010 at
test; windows that contain an FOMC statement or the close after a US general election (both eras) not worse than
+0.010. An `analog` roster (the last Friday on or before 23 October, 2005–2024) is reported only. Seeds 0–2,
4,096 draws, the strict harness.

| Arm | Test (95%) | Train | Event windows (95%) | Verdict |
|---|---|---:|---|---|
| F2/F3 and F4 back to v2.6.0 | −0.0213 [−0.0319, −0.0097] | −0.0312 | −0.0064 [−0.0235, +0.0090] | passes |
| **F4 back to v2.6.0 (this release)** | **−0.0175 [−0.0250, −0.0098]** | **−0.0271** | **−0.0135 [−0.0255, −0.0003]** | **passes** |
| F2/F3 back to v2.6.0 | −0.0037 [−0.0110, +0.0039] | −0.0042 | +0.0071 | fails rule 1 |
| width_scale 0.8 / 1.0 (default 0.9) | −0.0018 / +0.0067 | −0.0042 / +0.0091 | +0.0013 / +0.0044 | fail |
| mix_weight 0 (default 0.3) | +0.0022 | −0.0005 | +0.0017 | fails |

Two arms pass. The stage-2 rule as written combines them, which gives the first arm; its part beyond this release
is the F2/F3 revert, which fails rule 1 alone, loses on event windows (+0.011 per F2 task, +0.018 per F3 task) and
in the shock windows of the analog set, while the ranked window holds the 27–28 October FOMC meeting and the
3 November midterm. The team therefore shipped the F4-only arm (decision of 2026-10-11). It improves F4 in all 20
years of the short roster, by −0.094 per F4 task without an event and −0.059 with one; F4's 90% coverage becomes
0.95 and 0.91. The values are v2.6.0's; nothing is fitted.

### Results of the released source

Seeds 0–2, 4,096 draws, strict `backtest/benchmark.py` under the 6 October 2026 rule (0 failures),
`output/runs/2026-10-11-v2_7_1-*`; on the short roster the release equals the E23 arm on every task:

| Set | v2.7.0 | v2.7.1 | Change (95% interval) |
|---|---:|---:|---|
| Short roster (2–12 BD), cutoffs from 2019 (800 tasks; local M0 1.110) | 1.080 | 1.062 | −0.0175 [−0.0250, −0.0098] |
| Short roster, cutoffs to 2018 (800; local M0 0.963) | 0.972 | 0.945 | −0.0271 [−0.0317, −0.0223] |
| Holdout roster, cutoffs from 2019 (653) | 1.356 | 1.348 | −0.0077 [−0.0209, +0.0102] |
| Holdout roster, cutoffs to 2018 (746) | 1.062 | 1.036 | −0.0261 [−0.0352, −0.0162] |
| Daily roster, cutoffs from 2019 (800) | 1.383 | 1.370 | −0.0136 [−0.0231, −0.0028] |
| Daily roster, cutoffs to 2018 (800) | 1.060 | 1.035 | −0.0252 [−0.0323, −0.0173] |
| Transfer roster (500), monthly cards | | | identical |
| 91 practice cards (shock-picked), per card | | | +0.117 (F4 +0.367 per card; 65 validation cards +0.112) |

The practice cards were picked around realized shocks, which the ranked Final rules out; their cost is the price
of the narrower F4 on such windows and is why v2.6.1 had widened it.

### Cutoff and rule boundaries

- Every outcome used to choose the setting is dated on or before 2024-12-18. The FOMC statement and US election
  dates used to flag event windows are public calendars; they are not read at run time.
- The ranked Final's cutoff (2026-10-23) is after every one of them.
- `[metadata].category` is read as before for the per-family pairs (issue #20).

## Release v2.7.0 (`forecast-v2_7`)

This release adds one step to v2.6.1 (parent source `forecast-v2_6_1/forecast.py`, SHA-256 `134936948c1e1c54980f9f1b102b296a308d13aa2c096b6a6ba67b67ea8ae1a9`; this
source SHA-256 `48b1c3a24ca02c473a05f64589a05c95e257639d2b90fa7dae904c9c87e59bb1`): on daily level cells whose target is a Treasury tenor (`UST_<n>Y` on the
`rates_daily` panel) the final draws are shifted by half of the forward increment f(h, T) − y(T) that the unit's
own as-of curve implies for the target tenor at the target date (`rates_forward_add` 0.5; `rates_forward_center`).
f(h, T) = ((T + h) y(T + h) − h y(h)) / T, the expectations-hypothesis forward in the zero-coupon approximation,
with y(.) linear in maturity over the tenors the unit carries and flat beyond both ends (so flat below 2Y). The
shift is applied last; shape, width and dependence are untouched, and monthly, log-return, transfer and
non-Treasury cells never change. `V2_6_1_SETTINGS` (CLI `--profile v2.6.1`) reproduces v2.6.1 on all 104 public
cards; `V2_6_SETTINGS` reproduces v2.6.0, and the earlier profiles reproduce their releases. 48 of the
104 public cards differ from v2.6.1 (1 EXAMPLE, 12 F1, 12 F2, 13 F3, 10 F4); the other 56 equal it. No data, labels, learned
weights, task answers or per-card values are bundled, no text is read, and no language model is called.

### Why

Under the 6 October 2026 rule the largest remaining loss block of v2.6.1 is F1 on the rates panel (holdout
2019–2024: 94 tasks at 2.08 against the roster's 1.37; 90% coverage 77%, twice as many misses above the band as
below). The team's eight-arm replay of 2026-10-09 had shown that moving the centre to M0's mean or to the as-of
anchor hurts, that widening does not help after clipping and that the horizon dependence is useful. The rates
panel carries no policy rate, bill or futures series, so the curve itself is the only "policy path" available at
run time: E21 tested the expectations-hypothesis forward, the one centre not yet tried.

### Selection data and procedure

Experiment E21 (`exp/README.md`; wrapper `exp/e21_forward.py` around the v2.6.1 source; stage `e21` of
`exp/e11_numeric.py`; runs under `output/runs/2026-10-10-e21/`), designed and written down before any run, with
the E13 rules: holdout roster (test = cutoffs 2019–2024, 653 tasks, latest outcome 2024-12-02; train = 2005–2018,
746 tasks), change of at most −0.005 per task at test with the 189-business-day block-bootstrap interval below
zero, train not worse than +0.005, no family worse than +0.010 at test, the 91 practice cards with a local
outcome (latest 2024-12-09) not worse than +0.010 per card, and the 1,600-task daily roster's 2019+ interval
below zero. Seeds 0–2, 4,096 draws, the strict harness; the wrapper's off mode reproduced the formal v2.6.1 runs
exactly.

| Arm (stage 1, pre-registered) | Holdout 2019+ (95%) | 2005–2018 | Practice per card | Daily 2019+ (95%) | Verdict |
|---|---|---:|---:|---|---|
| median moved onto the forward | +0.0138 [−0.0535, +0.0955] | +0.0041 | −0.024 | +0.0083 [−0.0582, +0.0791] | fails rule 1 |
| halfway onto the forward | −0.0025 [−0.0363, +0.0387] | −0.0062 | −0.019 | −0.0057 [−0.0385, +0.0289] | fails rule 1 |
| forward increment added to the engine's centre, strength 1 | −0.0221 [−0.0474, −0.0027] | +0.0080 | −0.010 | −0.0215 [−0.0463, −0.0025] | fails rule 2 only |
| **stage 2: the same at strength 0.5 (this release)** | **−0.0118 [−0.0254, −0.0015]** | **−0.0001** | **−0.006** | **−0.0108 [−0.0226, −0.0013]** | passes all five |

Stage 2 was designed after stage 1's tables and before its run, as E13's stage 2 was: the full-strength
increment passed every rule but the 2005–2018 guard (+0.008 against +0.005, the zero-bound years when steep
curves priced hikes that never came), and half strength was the one follow-up. It is a second look at the same
development rosters; the per-task counts (212 holdout test Treasury tasks better against 117 worse; 261 against
130 on the daily roster), the unchanged capped-task counts and the near-zero train-era change are the reasons it
ships. Per Treasury task at test it is −0.022 (holdout) and −0.020 (daily); F1 −0.031 / −0.029, F2 −0.023 /
−0.013. The strength 0.5 is a grid value; nothing is fitted.

### Results of the released source

Seeds 0–2, 4,096 draws, strict `backtest/benchmark.py` under the new rule (0 failures),
`output/runs/2026-10-10-v2_7-*`:

| Set | M0 | v2.6.1 | v2.7.0 | v2.6.1 → v2.7.0 (95% interval) |
|---|---:|---:|---:|---|
| 1,399-task holdout roster, cutoffs from 2019 | 1.526 | 1.368 | 1.356 | −0.0118 [−0.0254, −0.0015] |
| of which cutoffs to 2018 | 1.088 | 1.062 | 1.062 | −0.0001 |
| 1,600-task daily roster, cutoffs from 2019 | | 1.394 | 1.383 | −0.0108 [−0.0226, −0.0013] |
| 500-task transfer roster | | | identical | 0 on every task |
| 91 practice cards, per card | | | | −0.006 (65 validation cards −0.003) |

### Cutoff and rule boundaries

- Every outcome used to choose the setting is dated on or before 2024-12-17. The shift reads only the unit's own
  Treasury series at or before the as-of date; nothing external, nothing from other units, no text, no model.
- The team's reading is that every Final-phase cutoff is after 2024-12-18 (repository `AGENTS.md`).
- `[metadata].category` is read as before for the per-family pairs (issue #20); the forward step does not read
  the category.

## Release v2.6.1 (`forecast-v2_6_1`)

This release widens three families of v2.6.0 (parent source `forecast-v2_6/forecast.py`, SHA-256 `d243221b037402d2977f8ea79a6ebef1e6cac674d3ecd7ee8443b53ad3ae92af`; this
source SHA-256 `134936948c1e1c54980f9f1b102b296a308d13aa2c096b6a6ba67b67ea8ae1a9`): F2 and F3 draws are scaled by k 1.2 about each cell's median (`sharpen_f2`,
`sharpen_f3`, were 1.0), and F4 runs at engine width 1.3 with k 1.1 (`f4_width_scale`, `sharpen_f4`; v2.6.0 used
1.1 with k 1.0), all with alpha 0, i.e. no resampling toward M0's centre anywhere. Everything else is v2.6.0.
`V2_6_SETTINGS` (CLI `--profile v2.6`) reproduces v2.6.0 on all 104 public cards, `V2_6_SETTINGS` with
`V2_5_SETTINGS` reproduces v2.5.0, and the earlier profiles reproduce their releases. The 74 fresh daily F2, F3
and F4 cards differ from v2.6.0; the 21 F1 cards, the 4 monthly cards and the 5 transfer cards equal it. No data, labels, learned weights, task answers
or per-card values are bundled, no text is read, and no language model is called.

### Why

v2.6.0's widths are the random-date optimum under the 6 October 2026 scoring rule (E13). On the re-scored
Development board (shock-picked practice cards) it scored 2.4893 against M0's 2.64, with 8 of 71 cards at the
cap of 8; the local replica tracks that board per card at correlation 0.998, so the gap to the top of the board
is the width on shock cards, not scoring. Under the new rule a too-narrow shock card costs up to 8 while extra
width costs linearly at calm dates, so the team's shock-share judgement for the Final's F2/F3/F4 cards, which
are placed where the text warns of risk, favours more width than the random-date rule alone (E14).

### Selection data and procedure

Experiment E14 (`exp/README.md`, `exp/e11_numeric.py` stage `e14`; runs under `output/runs/2026-10-08-e14/`),
designed after the v2.6.0 Development result and before any run; the same sets, seeds, draws, harness and new
rule as E13 (holdout roster, test = cutoffs 2019–2024, latest outcome 2024-12-02; daily roster; the 91 practice
cards with a local outcome, latest 2024-12-09). Five profiles were run against v2.6.0; this release is "fp4":

| Set | fp4 against v2.6.0 |
|---|---|
| Holdout, cutoffs 2019–2024 (653 tasks) | −0.0040 [−0.0374, +0.0232]; F2 −0.023 and F3 −0.026 per task of the family, F4 +0.025 per F4 task |
| Holdout, cutoffs 2005–2018 (746) | +0.0300 |
| Daily, cutoffs from 2019 (800) | +0.0007 [−0.0259, +0.0230] |
| 91 practice cards, per card | −0.176 (F2 −0.21, F3 −0.03, F4 −0.36 per card of the family) |

- F2 and F3 at k 1.2 improve every set (random dates and shock cards alike); only the holdout interval's upper
  end had kept them out of E13's rule-based choice.
- F4 at width 1.3, k 1.1, alpha 0 is the shock-share bet: against v2.6.0's F4 it costs +0.025 (holdout) to
  +0.047 (daily) per F4 task at random 2019+ dates and about +0.10 per F4 task on 2005–2018, and saves 0.36 per
  shock-picked practice F4 card; the break-even share of practice-like F4 cards is 0.07 (holdout), 0.12 (daily)
  or 0.22 (train era). The team judged the Final's share to lie between 0.15 and 0.3 and chose this setting
  over the neutral alternative (k 1.25, break-even 0.11–0.25) and the aggressive one (k 1.6, 0.20–0.31).
- Width 1.1 on top of these was rejected (worse at random dates on both rosters).

### Results of the released source

Seeds 0–2, 4,096 draws, strict `backtest/benchmark.py` under the new rule (0 failures),
`output/runs/2026-10-08-v2_6_1-*`:

| Set | M0 | v2.6.0 | v2.6.1 | v2.6.0 → v2.6.1 (95% interval) |
|---|---:|---:|---:|---|
| 1,399-task holdout roster, cutoffs from 2019 | 1.526 | 1.372 | 1.368 | −0.004 [−0.037, +0.023] |
| of which cutoffs to 2018 | 1.088 | 1.032 | 1.062 | +0.030 |
| 1,600-task daily roster, cutoffs from 2019 | | 1.393 | 1.394 | +0.001 [−0.026, +0.023] |
| 500-task transfer roster | | | identical | 0 on every task |
| 91 practice cards, per card | | | | −0.176 (65 validation cards −0.140) |

Implied Development board (local replica, 65 validation cards): v2.6.0 2.29 → v2.6.1 2.15; the three capped
transfer cards and the five capped F4 cards do not move.

### Cutoff and rule boundaries

- Every outcome used to choose the setting is dated on or before 2024-12-17. The Development board was used only
  to confirm the local replica (per-card correlation 0.998) and to prompt E14; the choice rests on the local
  sets and the team's judgement of the Final's shock share, stated above. The team's reading is that every
  Final-phase cutoff is after 2024-12-18 (repository `AGENTS.md`).
- The chosen values are grid values of existing settings (k 1.2, 1.2, 1.1; width 1.3; alpha 0); nothing is
  fitted. `[metadata].category` is read as before for the per-family pairs (issue #20).

### Hardening and tests

`sub/tools/fuzz_cards.py` (fourteen synthetic card shapes) and `test_forecast.py` (the v2.6.0 suite with its
reproduction tests re-pointed: `V2_6_SETTINGS` and the `v2.6` profile reproduce v2.6.0 on all 104 cards, v2.5.0
and the earlier releases through their profiles; every fresh daily card is scaled by its family's k with alpha 0
and nothing is reweighted; F4 cards carry engine width 1.3 and k 1.1). Runs are recorded in `sub/PIPELINE.md`.

## Parent record: release v2.6.0 (`forecast-v2_6`)

This release re-calibrates v2.5.0 (parent source `forecast-v2_5/forecast.py`, SHA-256 `2692a1c8ec6a4cbf75c8878f8f0daa620c453dab0f3cb2119f781f17566b55eb`; this source
SHA-256 `d243221b037402d2977f8ea79a6ebef1e6cac674d3ecd7ee8443b53ad3ae92af`) under the Track 2 scoring rule of 6 October 2026, which divides each component by the error M0
expects of itself on the card (M0-BASELINE.md §5, from the card's inputs) with cap and failure value 8.0. Five
defaults change: F1's draws are no longer resampled toward M0's centre (`sharpen_f1` (0.0, 1.1)), nor are F4's
(`sharpen_f4` (0.0, 1.0)) whose engine width comes down from 1.3 to 1.1 (`f4_width_scale`), the M0-t mixture
share is 0.3 (`mix_weight`, was 0.5) and the stochastic-volatility overlay is off (`vol_sd` 0, was 0.35).
`V2_5_SETTINGS` (CLI `--profile v2.5`) reproduces v2.5.0 on all 104 public cards; the 95 fresh daily cards
differ from v2.4.0, the 4 monthly and 5 transfer cards equal it (`test_forecast.py`). The source also carries
v2.5.1's `f3_width_profile` switch, default "v2.5" (its "v2.4" choice was made under the old rule and is not
taken). No data, labels, learned weights, task answers or per-card values are bundled, no text is read, and no
language model is called.

### Why

Under the previous rule (division by M0's realized error) the forecast that minimized the mean of ratios was
sharper than a calibrated one around M0's centre, and v2.1–v2.5's M0-aware resampling was built for it. The new
rule is proper: the honest distribution has the lowest expected score and a too-narrow card can cost up to 8.
Re-scoring the saved runs under the new rule (`exp/README.md`, E13) showed the v2.x reweighting losing to the
un-sharpened v1.2 at random 2019+ dates (1.442 against 1.411), the engine under-dispersed on every family at
those dates, the alpha reweighting on F4 costing 0.73 per shock-picked practice F4 card, and every widening
helping. The old rule's F4 width hedge and E12's F3 reversal were both products of that rule.

### Selection data and procedure

Experiment E13 (`exp/README.md`, `exp/e11_numeric.py` stages `e13`, `e13b`, `e13c`; runs under
`output/runs/2026-10-08-e13/`), decision rules fixed before each stage:

- Scoring: the local harness implements the new rule (`backtest/m0.py` `expected_scale`, `backtest/compare.py`,
  cap 8, failure 8; `backtest/test_contracts.py` checks the §5 worked example and the audit round-trip). The
  rosters were re-frozen against official commit 60509df on 2026-10-08 with every label recomputed and found
  identical. Practice cards are scored under the same rule.
- Sets: the 1,399-task holdout roster (test = cutoffs 2019-01-07 to 2024-09-26, 653 tasks, latest outcome
  2024-12-02; train = 2005–2018, 746), the 1,600-task daily roster as confirmation, the 91 practice cards with a
  local outcome (latest 2024-12-09). Seeds 0–2, 4,096 draws, the strict harness, local M0.
- Rules: test-era change ≤ −0.005 per task with the 189-business-day block interval below zero; train-era change
  ≤ +0.005; no family worse than +0.010; practice cards not worse than +0.010 per card; the final combination
  must beat the base on the daily roster's 2019+ era with the interval below zero.
- Stage 1 (34 single-knob settings from v2.5.0): five pass (F1 (0, 1.2) −0.022, width 1.1 −0.016, mix_weight 0.3
  −0.014, two-sided anchor 21 power 0.2 −0.014, vol_sd 0 −0.006). Stage 2: their combination gains −0.040 but
  costs the calm era (+0.015) with its interval touching zero, the anchor is redundant with width 1.1, and F4
  with alpha 0 wants a narrower engine width; "F4 width 1.1, (0, 1.0)" passes alone (−0.0125 [−0.031, −0.001],
  train 0.000). Stage 2b de-stacks the combination in the design's drop order: the passer with the lowest
  test-era mean is F1 (0, 1.1) + mix 0.3 + vol_sd 0 + F4 width 1.1 (0, 1.0): −0.0461 [−0.0854, −0.0146], train
  +0.0019, every family negative, practice −0.046 per card. Daily roster: −0.0397 [−0.0739, −0.0107].
- Not taken, with the reason: width 1.1 in any combination (train +0.010 to +0.017); F1 k 1.2 in the combination
  (train at the +0.005 bound); wider F4 under alpha 0 (shock cards −0.37 to −0.52 per card, calm era +0.010 to
  +0.025: the shock-share judgement, left to the team); the anchor 21 (redundant); mix_weight 0 (train +0.012).

### Results of the released source

Seeds 0–2, 4,096 draws, strict `backtest/benchmark.py` under the new rule (0 failures),
`output/runs/2026-10-08-v2_6-*`:

| Set | M0 | v2.5.0 | v2.6.0 | v2.5.0 → v2.6.0 (95% interval) |
|---|---:|---:|---:|---|
| 1,399-task holdout roster, cutoffs from 2019 | 1.526 | 1.418 | 1.372 | −0.046 [−0.085, −0.015] |
| of which cutoffs to 2018 | 1.088 | 1.030 | 1.032 | +0.002 |
| 1,600-task daily roster, cutoffs from 2019 | | 1.433 | 1.393 | −0.040 [−0.074, −0.011] |
| of which cutoffs to 2018 | | 1.024 | 1.030 | +0.006 |
| 500-task transfer roster | | 0.927 / 1.237 | identical | 0 on every task |
| 91 practice cards, per card | | | | −0.046 (65 validation cards −0.036; F4 −0.18; F1 +0.05) |

By family (holdout 2019+): F1 −0.109, F2 −0.008, F3 −0.009, F4 −0.058 per task. Under the new rule M0 itself
scores 2.46 on the 65 validation cards and 2.64 on the 71-card Development board; the local replica of the board
tracked the board at correlation 0.996 under the old rule.

### Cutoff and rule boundaries

- Every outcome used to choose any setting is dated on or before 2024-12-17 (rosters' latest outcome 2024-12-02,
  practice replica's 2024-12-09). No Development-board score entered any decision; the board's family split under
  the old rule only prompted E12, whose choice this release does not take. The team's reading is that every
  Final-phase cutoff is after 2024-12-18 (repository `AGENTS.md`).
- The chosen values are grid values of existing settings (alpha 0, k 1.1 and 1.0, width 1.1, mixture 0.3,
  overlay 0); nothing is fitted. `[metadata].category` is read as before for the per-family pairs (issue #20).

### Hardening

`sub/tools/fuzz_cards.py` runs the build's CLI on synthetic cards of fourteen shapes (odd horizons, up to 64
cells, as-of dates in 2026 beyond the published panels, stale and gappy panels, monthly cards with and without a
period mapping, transfer and mixed-transfer targets, minimal cards without optional keys, 20,000 draws). Every
case must exit 0 with an admissible output; the runs on this source are recorded in `sub/PIPELINE.md`.

### Tests

`test_forecast.py`: the v2.5 suite with its reproduction tests re-pointed (`V2_5_SETTINGS` and the `v2.5` profile
reproduce v2.5.0 on all 104 cards; `V2_4_SETTINGS` with `V2_5_SETTINGS` reproduce v2.4.0; the earlier profiles
reproduce their releases), the rule that only the 95 fresh daily cards differ from v2.4.0 and that F4 cards carry
alpha 0, k 1.0 and engine width 1.1, the widen-only anchor, the monthly and transfer blends, the F3 switch, and
validation of every new setting.

## Parent record: release v2.5.0 (`forecast-v2_5`)

This release changes the daily width of v2.4.0 (parent source `forecast-v2_4/forecast.py`, SHA-256 `53f313f45fd90c60a3d46aa7257d9b6b0410beef819d01dbf7777fd7026eaaea`;
this source SHA-256 `2692a1c8ec6a4cbf75c8878f8f0daa620c453dab0f3cb2119f781f17566b55eb`). On a fresh daily card the current step scale is the 60-step EWMA alone
(`vol_blend` 1.0 instead of a 0.8/0.2 blend with the 756-step scale), the current/long ratio applies with exponent 1
at every horizon (`vol_beta_base` 1.0 and `vol_beta_slope` 0.0 instead of 0.77 − 0.0735·log2(steps/21)), and a
cell longer than 63 business days is widened by (steps / 63)^0.3 (`horizon_scale_anchor` 63, `horizon_scale_power`
0.3, `horizon_scale_widen_only`). Every other default is v2.4.0's. The monthly path and the transfer proxies keep
the 0.8 blend (`monthly_vol_blend`, `transfer_vol_blend`), because the experiment scored daily fresh cards only.
Of the 104 public cards the 95 fresh daily cards change; the 4 monthly and 5 transfer cards are bit for bit
v2.4.0, and `Settings(**V2_4_SETTINGS)` (CLI `--profile v2.4`) reproduces v2.4.0 on all 104 (`test_forecast.py`).
No data, labels, learned weights, task answers or per-card values are bundled, no text is read, and no language
model is called.

### Why

At random cutoffs from 2019 the v2.4 engine scored 0.947 against M0 on the 1,399-task holdout roster while its
tail ratio sat at 0.993 and its 90% intervals covered 63–68% on cells of 64 business days and more, the same
under-coverage as M0. The engine's width at long horizons reverted toward the 756-step scale; on the 2019–2024
history, in which both trending and shock years are common, that reversion cost about 0.01 per task, and it cost
in 2005–2018 as well. With the width taken from current volatility, widening the long cells paid too.

### Selection data and procedure

Experiment E11 (`exp/README.md`, `exp/e11_numeric.py`; runs under `output/runs/2026-10-06-e11/`), with the
decision rules fixed before each stage:

- Sets: the holdout roster (1,399 random-date tasks shaped like the practice cards; `test` = cutoffs 2019-01-07 to
  2024-09-26, 653 tasks, latest outcome 2024-12-02; `train` = cutoffs 2005–2018, 746 tasks), the 1,600-task daily
  roster as the confirmation set (`test` = cutoffs from 2019, 800 tasks), and the 91 practice cards with a local
  outcome (shock-picked; latest outcome 2024-12-09) for the break-even check. Seeds 0–2, 4,096 draws, the strict
  harness (every forecast through G0–G3, no fallback), local M0, the official composite.
- Stage 1: 52 single-knob settings over width, tails, centre and the M0-aware post-processing. None reached the
  bar (−0.005 per test task with the 189-business-day block interval below zero). The three distinguishable from
  zero all raised the weight of current volatility (`vol_blend` 1.0 −0.0043, `vol_beta_slope` 0.0 −0.0028,
  `vol_beta_base` 0.9 −0.0026); every global widening helped cells of 64 BD and more and hurt 22–63 BD by about
  the same; tail-shape and centre knobs were flat or undecidable; F1/F2 per-family widths gained in the 2019+ era
  only.
- Stage 1b: the three volatility knobs combined (−0.0107 [−0.0221, −0.0007], train −0.0086), the F1/F2 grids
  extended (F1 fails the train-era rule, F2 the interval), and the horizon anchor introduced (63, two-sided:
  −0.0074 at power 0.3, but 28% narrower 21-BD cells).
- Stage 2: the volatility combination with the anchor in two forms. Two-sided: −0.0177 [−0.0294, −0.0054], train
  −0.0176, practice cards +0.0225 per card (F4 +0.050). Widen-only, power 0.3: −0.0144 [−0.0279, −0.0016], train
  −0.0083, practice cards +0.0048 (validation +0.0089; F4 +0.001), break-even share 0.75. Daily roster: two-sided
  −0.0112 [−0.0173, −0.0045]; widen-only −0.0065 [−0.0133, +0.0010]; both rosters' test eras pooled (1,453 tasks,
  252-day blocks): −0.0141 [−0.0217, −0.0056] and −0.0101 [−0.0190, −0.0004].
- Choice: the widen-only form. The two-sided anchor's extra gain is the F4 narrowing at random dates (its ≤21 BD
  bucket), which pays only if fewer than about 18% of Final F4 cards behave like the practice cards, and it is the
  F4 width decision that waits for the organizers' answer on issue #26 (how Final F2/F4 windows were chosen); F4's
  k stays 1.25 and its 21-BD cells are untouched. The daily roster alone misses the confirmation bar by 0.001; the
  pooled interval is below zero.

### Results of the released source

Seeds 0–2, 4,096 draws, strict `backtest/benchmark.py` (0 failures), `output/runs/2026-10-06-v2_5-*`:

| Set | v2.4.0 | v2.5.0 | v2.4.0 → v2.5.0 (95% interval) |
|---|---:|---:|---|
| 1,399-task holdout roster, cutoffs from 2019 | 0.9470 | 0.9326 | −0.0144 [−0.0279, −0.0016] |
| of which cutoffs to 2018 | 0.9435 | 0.9353 | −0.0083 |
| all eras | | | −0.0111 [−0.0183, −0.0039] |
| 1,600-task daily roster, cutoffs from 2019 | 0.9558 | 0.9493 | −0.0065 [−0.0133, +0.0010] |
| of which cutoffs to 2018 | 0.9647 | 0.9606 | −0.0041 |
| 500-task transfer roster | 0.9176 / 1.0137 | identical | 0 on every task |

By family (holdout, 2019+): F1 0.973 → 0.948, F2 0.910 → 0.887, F3 0.948 → 0.944, F4 0.953 → 0.946; F1's big-move
tasks 1.026 → 0.970 (−0.045 [−0.075, −0.013] over all eras). Local replica of the Development scoring: the 65
validation cards move from 0.9637 to about 0.973 (+0.0089 per card), the known cost of random-date width on
shock-picked cards; the 4 monthly and 5 transfer practice cards are unchanged.

### Cutoff and rule boundaries

- Every outcome used to choose any setting is dated on or before 2024-12-17: the rosters' latest outcome is
  2024-12-02 and the practice replica's 2024-12-09. No Development-board score was used. The team's reading is
  that every Final-phase cutoff is after 2024-12-18 (repository `AGENTS.md`), so the value is cutoff-clean there;
  for earlier cutoffs point-in-time eligibility is not claimed.
- The chosen setting has no fitted coefficient: three switches at their natural ends (an EWMA used alone; an
  exponent of 1) and one grid value (power 0.3 at anchor 63, the best of 0.2–0.5 under the rule).
- `[metadata].category` (the card family) is read as before for the sharpening pairs; the organizers confirmed
  on issue #20 that the family and the declared horizons may inform the forecast. No other card field is read.

### Tests

`test_forecast.py`: the v2.4.0 suite plus bit-for-bit reproduction of v2.4.0 under `V2_4_SETTINGS` and the `v2.4`
profile on all 104 cards, reproduction of v2.3.0/v2.2.0/v1.2.0 by their profiles, the rule that only the 95
fresh daily cards change, the widen-only factor on 21-, 63- and 126-BD cells, the monthly and transfer blends,
and validation of the new settings.

## Parent record: release v2.4.0 (`forecast-v2_4`)

This release adds one default to v2.3.0: on a card whose every target is a transfer asset (an early window ten
years before the as-of plus the as-of row), the draws are Gaussian on M0's mean and covariance, M0 computed by the
published procedure from the card's own panels, and each cell's sd is the geometric mean of M0's sd and the G10
proxies' current scale times spot and the square root of the steps (`transfer_m0="geo"`). The rule is a fixed
formula: it estimates no coefficient. The parent source is `forecast-v2_3/forecast.py`, SHA-256
`e196295fc6460c10b0f1ff442101eaeddf1518e306ce75d337a28904f7ef6ea1`. Only the 5 transfer cards among the 104 public
cards change; with `transfer_m0="off"` (CLI `--profile v2.3`) the samples equal v2.3.0 bit for bit on all 104
(`test_forecast.py`). No data, labels, learned weights, task answers or per-card values are bundled, and no
language model is called.

### Why

v2.3.0 sized a transfer target at the larger of its early-window scale and twice the G10 proxies' current scale,
with a ±25% lognormal scale on top. The local transfer roster could not test that on EM targets: it puts G10
currencies in the target slot because the published EM panels carry no recorded source. They are FRED H.10: the
published CNY, INR and BRL early windows equal DEXCHUS, DEXINUS and DEXBZUS on every one of their 9,664 rows. On
those series the floor runs about four times wider than managed currencies move (v2.3.0's sd about 10% of spot at
64 business days against M0's 2.5–2.9%), and every currency scored worse than M0.

### Selection data and procedure

- Data: the FRED H.10 daily series of ten EM currencies (CNY, INR, BRL, KRW, MXN, ZAR, THB, TWD, SGD, MYR; public
  domain; downloaded 2026-10-02 for local evaluation only, not packaged) and the published G10 panel.
- Tasks mirror the published transfer cards: the G10 panel to the as-of; the target's [as-of − 20 years, as-of − 10
  years] rows and its as-of row; the outcome on the exact business-day endpoint (64 business days for one target,
  63/126 for two). 1,000 tasks: era `fit`, outcomes before 2014-12-18 (as-of 2003–2014), and era `check`, as-of
  2015 to 2024-10. The five public practice transfer cards were scored with their H.10 outcomes.
- Seven candidate rules were fixed before the run (v2.3.0 with a floor of 1.0, with and without narrowing; M0's sd
  × 0.85, 1.0, 1.2; the geometric mean; the whole early window's sd), with a pre-registered rule: beat v2.3.0 by at
  least 0.05 on the fit era with the month-block interval below zero and no currency worse by 0.05. All but the
  early-window rule passed; the rule's pick was M0 × 0.85 (lowest fit-era mean). The team chose the geometric
  mean instead as a hedge for shock-picked Final cards, from the fit-era scores and the five practice cards.

| Width | Fit era (500) | Check era (500) | Practice transfer cards (5) |
|---|---:|---:|---:|
| v2.3.0 | 1.964 | 1.791 | 0.828 |
| M0's sd × 0.85 | 0.995 | 0.998 | 1.030 |
| M0's sd × 1.0 | 1.001 | 1.001 | 0.999 |
| **geometric mean (v2.4.0)** | 1.080 | 1.030 | 0.856 |

### Results of the released source

- 500 EM transfer tasks per era with this source, seed 0, 4,096 draws: v2.3.0 → v2.4.0 −0.884 [−0.975, −0.796] on
  the fit era and −0.761 [−0.848, −0.673] on the check era (month-block 95% intervals).
- 500-task transfer roster (G10 targets), strict `backtest/benchmark.py`, seeds 0–2: cutoffs from 2019 1.2712 →
  0.9176, paired −0.354 [−0.507, −0.205], 77% of tasks better; earlier cutoffs 1.3064 → 1.0137. 0 failures.
- Daily and holdout rosters: 300 sampled tasks give samples identical to v2.3.0 (seed 0, 4,096 draws); none of their
  targets is a transfer asset, so the v2.3.0 results below stand. The local Development replica excludes transfer
  cards; on the board the five practice transfer cards average 0.856 instead of 0.828.

### Cutoff and rule boundaries

- No choice used an outcome dated after 2024-12-17. The choice among the seven rules rested on outcomes before
  2014-12-18, before the withheld decade of any card with an as-of after 2024-12-18. The practice cards' outcomes
  (2013–2018) and the check era were seen when the hedge was chosen; the geometric mean has no fitted coefficient,
  and no EM value is packaged.
- A card that mixes a transfer target with a fresh one keeps v2.3.0's path; a cell without a fresh G10 proxy keeps
  M0's sd.

### Tests

`test_forecast.py` has 76 tests: the 68 of v2.3.0, eight of them adjusted (the reproductions of v1.2.0, v2.1.0 and
v2.2.0 and the v2.1.0 and v2.2.0 change sets now pass `transfer_m0="off"`; the default profile, the CLI profiles and the stale
monthly card also check the new setting), and eight new ones (the rule off reproduces v2.3.0 on all 104 cards and
only the 5 transfer cards change; the geometric-mean moments; the M0-scale rule; the joint card's correlation; a
card without a fresh proxy; the gates; `--profile v2.3`; settings validation).

---

## Parent record: release v2.3.0 (`forecast-v2_3`)

This release changes one default of v2.2.0: the F4 sharpening pair, from (1.0, 1.1) to (1.0, 1.25). The engine's
shock-family width (`f4_width_scale` 1.3) is unchanged. The parent source is `forecast-v2_2/forecast.py`, SHA-256
`8860dad2c7515939d224a7f0257371010b0f37f1b510b5979543d126e39cbf2c`. Only the 29 fresh daily F4 cards among the 104
public cards change; with `sharpen_f4=(1.0, 1.1)` (CLI `--profile v2.2`) the samples equal v2.2.0 bit for bit on all
104 (`test_forecast.py`). No data, labels, learned weights, task answers or per-card values are bundled, and no
language model is called.

### Why

v2.2.0 was uploaded to the Development board on 2026-09-29 and scored 0.9737, against 0.9439 for v1.2.0. Per card
the board matched the local replica (correlation 0.995 on the 65 cards with a local label; +0.030 predicted and
+0.030 observed over the 71 cards). The whole change came from F4: 0.979 → 1.134 on its 22 cards, while F1 improved
from 0.949 to 0.874. Those cards were chosen around realized shocks, and a narrow F4 loses exactly there.

Most sealed Final units resolved after they were written, so their F4 windows cannot have been chosen for a realized
shock; that is why v2.2.0 took k = 1.1. But F4 cards are still placed where the text warns of risk, so their moves
can run larger than at a random cutoff. A slightly wider F4 costs little at random cutoffs and saves a lot on
shock-picked cards. F4 cards only, alpha 1.0, seeds 0–2, 4,096 draws (lower is better, M0 = 1):

| F4 k | holdout F4 tasks (427) | of which cutoffs from 2019 (244) | validation F4 cards (22) | public-dev F4 cards (7) |
|---|---:|---:|---:|---:|
| 1.0 | 0.915 | 0.953 | 1.180 | 1.105 |
| 1.1 (v2.2.0) | 0.925 | 0.947 | 1.121 | 1.045 |
| **1.25 (v2.3.0)** | 0.950 | 0.953 | 1.056 | 0.964 |
| 1.4 | 0.987 | 0.976 | 1.008 | 0.924 |
| 1.6 (`balanced`) | 1.051 | 1.027 | 0.967 | 0.883 |
| v1.2.0 | 1.082 | 1.042 | 0.979 | 0.842 |

On cutoffs from 2019, k = 1.25 is 0.006 per F4 card behind k = 1.1, and on the shock-picked validation cards it is
0.065 ahead. It is the better choice as soon as more than about 9% of Final F4 cards behave like the practice cards;
k = 1.4 would need more than about 32%.

### Results of the released source

Seeds 0–2, 4,096 draws, strict `backtest/benchmark.py` (every forecast through G0–G3, 0 failures):

| Set | v1.2.0 | v2.3.0 | v1.2.0 → v2.3.0 (95% interval) | v1.2.0 → v2.2.0 |
|---|---:|---:|---|---|
| 1,600-task daily roster | 1.0043 | 0.9602 | −0.044 [−0.055, −0.034] | −0.051 |
| of which cutoffs from 2019 | 0.9841 | 0.9558 | −0.028 [−0.042, −0.016] | −0.033 |
| 1,399-task holdout roster | 0.9887 | 0.9452 | −0.044 [−0.054, −0.034] | −0.051 |
| of which cutoffs from 2019 | 0.9729 | 0.9470 | −0.026 [−0.037, −0.013] | −0.028 |
| 500-task transfer roster | 1.2888 | 1.2888 | identical | identical |

Local replica of the Development scoring (practice cards with a local label): 65 validation cards 0.9556 (v1.2.0),
0.9859 (v2.2.0), 0.9637 (v2.3.0); 26 public-dev cards 0.9742, 1.0280, 1.0062. Monthly cards are untouched: the
family pairs never reach them, so the v2.2.0 monthly backtest stands.

### Cutoff and rule boundaries

- Every outcome used to choose any setting is dated on or before 2024-12-17: the holdout roster's latest outcome is
  2024-12-02 and the local replica's 2024-12-09. The Development board's per-card scores were used only for the F4
  cards, whose outcomes all fall in 2023 or earlier. The team's reading is that every Final-phase cutoff is after
  2024-12-18 (repository `AGENTS.md`), so the value is cutoff-clean there; for earlier cutoffs point-in-time
  eligibility is not claimed.
- The share of resolved sealed units is the organizers' published description of the sealed set, not outcome data.
- Nothing card-specific is packaged.

### Tests

`test_forecast.py` has 68 tests: the 66 of v2.2.0, four of them adjusted to the new default (the default profile,
the v2.1.0 reproductions, the CLI profiles), and two new ones (the F4 pair at 1.1 reproduces v2.2.0; only the 29
fresh daily F4 cards change against v2.2.0).

---

## Parent record: release v2.2.0 (`forecast-v2_2`)

This release adds a monthly path to v2.1.0 and changes nothing on daily or transfer cards. The parent source is
`forecast-v2_1/forecast.py`, SHA-256 `9e422361460d0c3bf4dd571253dc3a5b073272a9187fb3211d7ddeb231f829de`. On all 104
public cards the samples equal v2.1.0 bit for bit except on the four monthly practice cards; with
`monthly_m0_path=False` (CLI `--profile v2.1`) they equal v2.1.0 on all 104 (`test_forecast.py`). No data, labels,
learned weights, task answers or per-card values are bundled, and no language model is called.

### Why

A monthly macro backtest (below) put v2.1.0 at about 1.3 against M0 on monthly cards, with single tasks at the 4.0
clip. Two causes. The monthly engine's width ran 1.2–1.5 times M0's while outcomes sat close to M0's centre, which
the per-card ratio to M0 punishes. And on two-cell monthly cards the embedded M0 estimated its month counts from the
business-day key: pairing then aimed at the wrong value, could not move centres on a monthly card, and left the
joint ratio to chance.

### What changed

1. **M0 steps.** The embedded M0 of a monthly card walks the card's observation months
   (`targets.observation_periods`), counted from each target's last observation. The published step table
   (M0-BASELINE §3.7) still wins for the four cards it lists; without a mapping the §3.7 estimate is kept. The
   observation-month count reproduces three of the four published tables (cpi-glidepath, covid-nfp, cpi-vintage;
   sahm-watch's is one month later).
2. **Mixture and sharpening.** On fresh monthly cards (no fallback, freshness rule met) half of the draws come from
   M0's mean and covariance with Student-t(5) shocks, full drift, scaled by 0.8 (`monthly_mix_*`). The joint draws
   are then resampled by 1 / CRPS_M0^0.5 (`monthly_sharpen_alpha`), as `m0_sharpen` does on daily cards.
3. **Pairing.** On fresh monthly two-cell cards pairing may move centres or blend M0 draws, as it already could on
   fresh daily cards.

Stale monthly cards, daily cards and transfer cards follow v2.1.0 exactly. The audit keys `post_processing.mixture`,
`.sharpen` and `.pair_match` record what was applied.

### Selection data and procedure

Monthly backtest: synthetic tasks from the macro panel of `t2-F1-sahm-watch-2024` (the vintage published on
2024-08-02, observation months 2000-01 to 2024-06) for CPI_ALL, CPI_CORE, PCE_ALL, PCE_CORE, UNRATE and NFP, in the two
shapes of the practice cards: targets 8 and 9 months after the last observation (keys 140/160, family F1) and 2
months after it (key 21, family F4). As-of dates are drawn at random from 2005 onward, and each task sees the panel
up to its as-of minus 45 days, as the practice cards do. M0 is scored with the exact month counts a sealed target
date in the observation month gives; the forecaster has to work them out as in the Final. The latest outcome used is
the 2024-06 observation.

The mixture and sharpening values were chosen among 15 variants on one sample (360 tasks, seed 20260929) and
confirmed on a fresh sample (360 tasks, seed 777). Results of the released source (seed 0, 4,096 draws; lower is
better, M0 = 1):

| Monthly backtest | v2.1.0 | v2.2.0 |
|---|---:|---:|
| first sample, 8/9 months ahead (180 tasks) | 1.399 | 0.994 |
| first sample, 2 months ahead (180 tasks) | 1.133 | 0.912 |
| fresh sample, 8/9 months ahead (180 tasks) | 1.501 | 0.994 |
| fresh sample, 2 months ahead (180 tasks) | 1.188 | 0.932 |
| tasks at or above 2.0 (first / fresh sample) | 47 / 66 | 0 / 1 |

NFP is the one series where v2.1.0's engine already beat M0; it stays below 1 (0.78–0.92 by shape and sample).

Daily cards are unchanged. On a new holdout roster frozen on 2026-09-29 (`backtest/rosters/holdout-roster.json`:
1,399 tasks from generator seed 20260929, dropping tasks within 21 days of a tuning-roster task of the same source
card; latest outcome 2024-12-02), the strict benchmark (`backtest/benchmark.py`, seeds 0–2, 4,096 draws, 0 failures)
gives v1.2.0 → v2.2.0: −0.051 [−0.064, −0.039] over all tasks and −0.028 [−0.041, −0.010] on cutoffs from 2019.

Checked on 2026-09-29 and not changed: F2/F3 sharpening (−0.010 on the tuning roster, −0.003 [−0.009, +0.003] on the
holdout and +0.003 from 2019); single upstream settings (mixture weight and drift, width scale, horizon power,
volatility blend, centre cap, Student-t degrees of freedom, F1 pair), all within ±0.005 on the tuning roster. A
sweep of 200 synthetic grids (up to 30 cells, horizons 1–252) ran with no failure, fallback or gate refusal.

### Cutoff and rule boundaries

- Every outcome used to choose any setting is dated on or before 2024-12-17; the new monthly values used outcomes up
  to the 2024-06 observation, published by 2024-08-02. The team's reading is that every Final-phase cutoff is after
  2024-12-18 (repository `AGENTS.md`), so the values are cutoff-clean there; for earlier cutoffs point-in-time
  eligibility is not claimed.
- Nothing card-specific is packaged. The monthly path reads the card's cadence, targets, observation periods and
  panels, and the four step counts M0-BASELINE §3.7 publishes for M0 itself.

### Tests

`test_forecast.py` has 66 tests: the 60 of v2.1.0, seven of them adjusted to the monthly path, and six new ones
(monthly path off reproduces v2.1.0; only the four monthly cards change; the path's records and gates; the published
table wins; a stale monthly card is left alone; settings validation).

---

## Parent record: release v2.1.0 (`forecast-v2_1`)

This release adds one post-processing step to v1.2.0, M0-aware sharpening, and makes the
"final" profile the default. The parent source is `forecast-v1_2/forecast.py`, SHA-256
`bd54e61cd8419f010abfb9c533994cb545a5137e9c1757bbdd194d34d972e0e9`. With `sharpen=False`
(CLI `--profile v1.2`) the samples equal v1.2.0 bit for bit on all 104 public cards
(`test_forecast.py`). No data, labels, learned weights, task answers or per-card values are
bundled, and no language model is called.

### What changed

1. **Why.** Each card's score is divided by M0's score on the same outcome, and the board
   averages those ratios. Minimizing an expected ratio is minimizing the expected score under the
   outcome distribution reweighted by 1 / M0's score, and that weight is largest near M0's centre,
   where M0 scores best. For CRPS the minimizer is the predictive distribution multiplied by
   1 / CRPS_M0, which is sharper than the predictive distribution itself. Worked analytically for a
   single cell: if M0 were exactly calibrated, a Gaussian 0.8 times as wide as M0 would average
   0.961 against M0's 1.0.
2. **What.** On daily cards whose targets all have fresh histories, after the M0-t mixture and
   before pair matching, the joint draws are resampled (systematic resampling of whole draws) with
   weight 1 / mean_c CRPS_c(x_c)^alpha. CRPS_c is the closed-form CRPS of M0's Gaussian marginal
   for cell c (mean anchor + s·mu, standard deviation from M0's covariance), in units of that
   standard deviation. Each cell is then scaled by k about its median. One (alpha, k) pair per
   family, read from `card.toml [metadata] category`, which is part of the task specification;
   no text is read.
3. **Default profile "final":** F1 (0.75, 1.1), F2 (0, 1), F3 (0, 1), F4 (1.0, 1.1). F2 and F3 draws
   are unchanged from v1.2.0. On F4 the engine keeps its shock-family width (`f4_width_scale` 1.3);
   the draws then get a sharper core around M0's centre and are widened by 1.1 about the median.
4. **Not applied** to transfer or monthly cards, whose M0 moments come from a stale early window or
   converted step counts; to fallback forecasts; or when the embedded M0 cannot be built or its
   step counts differ from the task's by 2x. The audit key `post_processing.sharpen` records the
   pair used and the effective number of draws, or why nothing changed.
5. **Comparison profiles**, selectable only through the CLI `--profile` (the harness passes none):
   `balanced` = "final" with F4 (1.0, 1.6); `dev-leaning` F1 (1.0, 0.9), F2 (0, 1.1), F3 (0, 1),
   F4 (1.0, 2.0); `random-leaning` F1 (0.75, 1.1), F2 (0.5, 1.0), F3 (0.5, 1.0), F4 (1.0, 1.0).

### Selection data and procedure

The (alpha, k) values were chosen after viewing exploratory results. They are development
evidence, not an untouched holdout. A grid alpha ∈ {0, 0.25, 0.5, 0.75, 1} × k ∈ {0.5, …, 2.4} was
applied to v1.2.0 draws (seed 0, 4,096 draws) and scored per family, with the official composite,
component normalization by the local M0 reproduction, single-cell weights and [0, 4] clipping, on:

- the 91 published practice cards with a local label (65 `validation`, 26 `public-dev`; transfer
  and monthly cards excluded). Each outcome is the h-th panel row after the as-of in other
  published practice panels; the latest outcome used is 2024-12-09. On the 65 validation cards
  this replica matches v1.2.0's per-card Development scores with correlation 0.999;
- `backtest/rosters/daily-roster.json`: 1,600 tasks at random cutoffs from 2005-01-28 to
  2024-11-05, latest outcome 2024-12-17.

The two sets disagree. The published cards were authored around realized events: 42% of
single-cell validation cards end more than two M0 standard deviations from M0's mean, against 9%
at random cutoffs. They prefer wider F2 and F4 forecasts; random cutoffs prefer sharper forecasts
in every family. F1 (0.75, 1.1) was no worse than v1.2.0 on both sets; F2 and F3 are left unchanged.

F4 is where the choice matters, and it was made on the organizers' own description of the sealed
set, not on any outcome. At the participant release (2026-08-29) the README stated that only 12 of
the 123 sealed units had a resolved outcome and "the rest resolve in the future", and the scorer
calls such units prospective. A Final F4 card is therefore chosen before its outcome exists, and
for it the random-cutoff roster is the relevant evidence. With alpha 1.0, F4 scored by k:

| F4 k (alpha 1.0), seed 0 | 1.0 | 1.1 | 1.25 | 1.6 | v1.2.0 |
|---|---:|---:|---:|---:|---:|
| random-cutoff F4 tasks, 2019–2024 (244) | 0.936 | 0.936 | 0.951 | 1.030 | 1.050 |
| practice validation F4 cards (22) | 1.178 | 1.125 | 1.060 | 0.965 | 0.984 |

k = 1.1 rather than 1.0 keeps a small margin for as-of dates that sit before a scheduled event.

Results of the released source (seeds 0, 1, 2; 4,096 draws; lower is better, M0 = 1):

| Set | v1.2.0 | v2.1.0 "final" | Paired difference (95% interval) | `balanced` |
|---|---:|---:|---|---:|
| 65 validation cards (official units as of 2026-09-27) | 0.9556 | 0.9859 | +0.030 [−0.005, +0.069] | 0.9338 |
| 26 public-dev cards | 0.9742 | 1.0280 | +0.054 [+0.017, +0.095] | 0.9844 |
| 1,600-task daily roster (`backtest/benchmark.py`, strict) | 1.0043 | 0.9534 | −0.051 [−0.063, −0.040] | 0.9888 |
| of which cutoffs to 2018 | 1.0245 | 0.9556 | −0.069 | 0.9980 |
| of which cutoffs from 2019 | 0.9841 | 0.9512 | −0.033 [−0.049, −0.018] | 0.9796 |
| 500-task transfer roster (strict) | 1.2888 | 1.2888 | identical on every task | 1.2888 |

By family on the daily roster: F4 −0.157 [−0.189, −0.130] (1.095 → 0.938); F1 −0.013
[−0.036, +0.007], which is −0.035 on cutoffs to 2018 and +0.008 on cutoffs from 2019; F2 and F3
identical. On the 385 tasks whose outcome left M0's own 90% interval, v2.1.0 is worse by +0.066
[+0.056, +0.077], and on the 60 such F4 tasks by +0.264: that is the cost of sharpening when a large
move does happen, and it is why the practice cards, chosen for such moves, score worse. The strict
benchmark ran 4,800 forecasts per version with 0 failures, each through the official gates G0–G3
and a check that the written file equals the scored samples.

The other comparison profiles, evaluated the same way before release (validation / public-dev /
daily roster): `dev-leaning` 0.911 / 0.965 / 1.051, `random-leaning` 1.022 / 1.079 / 0.942.

### Cutoff and rule boundaries

- Every outcome used to choose any setting is dated on or before 2024-12-17. The team's reading is
  that every Final-phase cutoff is after 2024-12-18 (repository `AGENTS.md`), so the new values are
  cutoff-clean there. For tasks with earlier cutoffs, including the Development cards,
  point-in-time eligibility is not claimed, as for forecast-v1.
- The F4 width used the organizers' published description of the sealed set (its share of
  resolved units), which is not outcome data.
- The Development board's per-card scores for v1.2.0 were read only to check that the local
  replica reproduces them; no setting was fitted to them. Two of those cards resolve in 2025; they
  have no local label and entered no selection.
- Outcomes of published practice cards were used only to score global settings on the
  development machine. No table, value or per-card parameter derived from them is packaged. At
  runtime the step reads the card's family label, targets and panels, and nothing about its
  identity beyond what v1.2.0 already read.

### Tests

`test_forecast.py` has 60 tests: 50 inherited and 10 for the sharpening step. One inherited test,
the renamed monthly CPI card of review P1, now accepts either a `repair` pairing or `unmatched`
with unchanged draws: on the CPI vintage the organizers published on 2026-09-26, re-pairing no
longer reaches the embedded M0's value on that renamed card, and a monthly card then keeps its
draws by design. The same test fails unchanged under `forecast-v1_2` on the updated units.

---

## Parent record: release v1.2.0 (`forecast-v1_2`)

This release fixes three findings from a review of v1.1.0 and changes nothing else. The parent
source is `forecast-v1_1/forecast.py`, SHA-256
`b8f39472a8adeaca7857af752b7ffc875f076159d6215932643e4c7b0cc02892`. No numerical setting was
re-selected. On all 104 public cards the samples equal v1.1.0 bit for bit (seed 0, 4,096 draws).
No data, labels, learned weights, task answers or per-card values are bundled, and no language
model is called.

### Fixes

1. **M0 step counts on monthly cards outside the published four (review P1).** v1.1.0 walked the
   declared horizon key in its embedded M0 for any monthly card whose id was not in the
   M0-BASELINE.md §3.7 table. A card declaring 140/160 business days therefore became a
   140/160-month walk. Pair matching then pushed the two horizons to a correlation near −1.
   Measured on a renamed copy of `t2-F1-cpi-glidepath-2023` against realized CPI, that scored
   4.0. The embedded M0 now follows §3.7 without the sealed target date. Daily panels keep the
   declared horizon, as §3.7 does on every daily card. On monthly panels a key below 21 is
   already in panel steps; a larger key is converted by counting months from the last
   observation to the as-of plus that many US business days, and the 2x rule decides. The same
   renamed card now counts 9/10 months and scores 1.106. Pair matching also refuses any M0
   reference whose step counts differ from the task's by 2x or more.
2. **Pair-matching failure path (review P2a).** When re-pairing could not reach M0's pair
   statistic, v1.1.0 returned the closest pairing after the full centre shift. That happened on
   6 of 891 two-cell roster forecasts, with joint ratios up to 3.66. Reverting to the unshifted
   draws would have been worse: two of those tasks clip at 4.0 without the shift. v1.2.0
   bisects the smallest centre shift, then, if needed, replaces the smallest share of draws
   (in 5% steps) with draws from M0's distribution. The blended sample always reaches the
   target exactly, since the statistic is a draw-weighted mean. Monthly and transfer cards
   never move centres or blend; an unreachable target leaves their draws unchanged. The audit
   records `pair_match.method` as `repair`, `shift`, `m0-blend`, `unmatched` or `skipped`.
3. **Monthly mapping conflicts (review P2b).** v1.1.0 treated every error of the official
   observation-period helper as a missing mapping and guessed steps. v1.2.0 uses the M0 §3.7
   rule only when no mapping is supplied anywhere. When the helper refuses the combined
   mapping, the card's own mapping is used if it is valid, since the scorer trusts the card,
   and the rationale says the spec was ignored. Otherwise the card is refused with the
   helper's reason; the loader failure then writes the M0-based emergency forecast.

### Corrections to the v1.1.0 record

The v1.1.0 record below is kept as shipped. Two of its statements were too strong:
- Item 2 says the pair statistic equals M0's to better than 1e-9 relative. That held for 868 of
  891 two-cell roster forecasts. Another 17 needed a centre shift, and 6 did not reach M0's
  value at all (fixed above).
- Item 5 describes the monthly step rule as applying to a card without a mapping. It also
  applied to conflicting and invalid mappings (fixed above).

### Evidence

`test_forecast.py` has 50 tests: 40 inherited plus 10 regression tests for the three fixes.
Evaluation against the local M0 reproduction (seeds 0–2, 4,096 draws; lower is better):

| Set | v1.1.0 | v1.2.0 |
|---|---:|---:|
| 91 published cards (65 validation + 26 public-dev) | 0.9623 | 0.9623 (identical) |
| 1,600-task daily roster | 1.0045 | 1.0043 (paired −0.0002, 95% [−0.0008, +0.0001]) |
| Daily roster re-weighted to the published cards' surprise sizes | 0.9563 | 0.9560 |
| 500-task transfer roster (seed 0) | 1.2891 | 1.2891 (identical) |

On the roster, two-cell forecasts were matched by `repair` 868 times, `shift` 17 and
`m0-blend` 6, all to M0's value exactly. v1.1.0 scored 0.9439 on the 71-card Development board
(v1: 1.0126). v1.2.0 produces the same samples on every public card.

Cutoff and rule boundaries are unchanged from v1.1.0: every outcome used to select any setting is
dated on or before 2024-12-17, and no per-card value is packaged.

---

## Parent record: release v1.1.0

This release keeps the V3.3.1 numerical engine of `forecast-v1` unchanged and adds
post-processing. The parent source is `forecast-v1/forecast.py`, SHA-256
`cb11e8f4d3900793da48cc1a191f27e614ad5da2a7e099c62f44732316e2f8f9`. Every addition has a
`Settings` switch; with all of them off (`POST_PROCESSING_OFF`) the samples equal
`forecast-v1` bit for bit on all 104 public cards (`test_forecast.py`). Only runtime
source, pinned dependencies and this record enter the image. No data, labels, learned
weights, task answers or per-card values are bundled, and no language model is called.

### What changed

1. **Published M0 procedure, embedded.** A line-for-line port of `backtest/m0.py`, which
   implements `docs/M0-BASELINE.md` §3. It reads only the card and the supplied panels.
   Its 500 draws equal the backtest reproduction bit for bit on all 104 public cards.
2. **`pair_match`.** On a two-cell grid the variogram has a single pair. M0's value for it can
   land near zero, so any different pairwise statistic turns the joint ratio into a lottery.
   The draws are re-paired through a Gaussian copula, each cell keeping its values, and a
   final stretch of each pair difference by 1 + eps (|eps| ≤ 1%) closes the last gap. The
   pair statistic then equals M0's own 500-draw value to better than 1e-9 relative.
3. **`mix_weight` 0.5, `mix_df` 5, `mix_drift` 0.5.** On daily cards with fresh histories
   half of the draws come from M0's mean and covariance with multivariate Student-t shocks
   and half of M0's drift. Transfer and monthly cards are unchanged.
4. **`f4_widen`, `f4_width_scale` 1.3.** Daily cards whose `card.toml [metadata] category`
   is T2-F4 use width scale 1.3 instead of 0.9 in the engine, and scale 1.3 in the M0-t share.
   The family is part of the task specification; no text corpus is read.
5. **`robust_fallback`.** An engine failure falls back to M0's distribution, then a robust
   random walk, then the previous wide grid. A loader failure writes the same M0-based
   forecast instead of a zero-centred grid. Contract repairs: a spec that contradicts the card
   is ignored, a `--asof` that contradicts the card is replaced by the card's cutoff, a monthly
   card without an observation-period mapping counts steps with the M0 §3.7 rule (a key below
   21 is already in panel steps), and conflicting duplicate rows keep the first row.
6. **Build-independent covariance root.** The M0-t share uses a Cholesky factor, which is
   unique; the eigen fallback fixes eigenvector signs. Draws do not depend on the LAPACK build.

### Selection data and procedure

The four new numerical values (`mix_weight`, `mix_df`, `mix_drift`, `f4_width_scale`) were
chosen after viewing exploratory results. They are development evidence, not an untouched
holdout. Three evaluation sets were used, all scored against the local M0 reproduction with
the official composite, component normalization, single-cell weights and [0, 4] clipping:

- `backtest/rosters/daily-roster.json`: 1,600 tasks, cutoffs 2005-01-28 to 2024-11-05,
  latest outcome 2024-12-17.
- 91 published practice cards (65 `validation`, 26 `public-dev`; transfer and monthly cards
  excluded). Each outcome is the h-th panel row after the as-of, read from other published
  practice panels; the latest outcome used is 2024-12-09. With this rule the parent model's
  local scores match its per-card Development scores with correlation 0.9955.
- The daily roster re-weighted per family so that the size of the realized outcome relative
  to M0's spread matches the published cards, then combined in the validation family mix.

Results for seeds 0, 1, 2, draws 4,096 (lower is better, M0 = 1):

| Set | forecast-v1 | v1.1.0 | Paired difference (95% interval) |
|---|---:|---:|---|
| 91 published cards | 1.0130 | 0.9623 | −0.051 [−0.085, −0.020] |
| of which 65 validation cards | 1.0266 | 0.9566 | |
| 1,600-task daily roster | 1.0146 | 1.0045 | −0.010 [−0.032, +0.012] |
| Re-weighted daily roster | 1.0286 | 0.9563 | |
| 500-task transfer roster | 1.2891 | 1.2891 | identical |

The roster had 51 task-seeds clipped at 4.0 under forecast-v1 and none under v1.1.0. The
shock-family width costs +0.13 on random-date F4-shaped tasks and gains −0.21 on those
whose outcome left M0's 90% interval (`backtest/benchmark.py`, strict mode, 0 failures in
4,800 forecasts). Sensitivity: the mixture values are flat within ±0.005; the F4 width trades
published cards (prefer 1.5) against random dates (prefer 1.2), and 1.3 sits between them.

### Cutoff and rule boundaries

- Every outcome used for selection is dated on or before 2024-12-17. The team's reading is
  that every Final-phase cutoff is after 2024-12-18 (repository `AGENTS.md`), so the new
  values are cutoff-clean there. For tasks with earlier cutoffs, including the Development
  cards, point-in-time eligibility is not claimed, as for forecast-v1.
- Outcomes of published practice cards were used only to score global settings on the
  development machine. No table, value or per-card parameter derived from them is packaged,
  and nothing at runtime depends on the card's identity except the four monthly step counts
  that M0-BASELINE.md §3.7 publishes for M0 itself.
- Inherited V3.3 settings and their provenance are unchanged; see the parent record below.

---

## Parent record: forecast-v1 (V3.3.1-sub)


### Submission packaging revision 3.3.1-sub

This candidate preserves V3.3 numerical settings and predictions. Packaging changes
are a corrected runtime version in the metadata, a numbered rationale with input
dates and an arithmetic adjustment ledger, and bounded numerical thread pools.
The source parent is `forecast/forecast-v3/forecast.py`, SHA-256
`e02d9b06cda509a375c04f8f2b253c8d296a8115846fee43f38624be1a412d73`.
Only runtime source, pinned dependencies, and this provenance record are copied
into the image. Team secrets, historical labels, practice units and experiment
results are excluded from the build context.

This is a locally validated submission candidate, not a certification of historical
calibration eligibility. The unresolved data-selection issue below remains a
submission-readiness blocker. Declaring the numerical calibration as a local model
does not waive the task cutoff. Image publication and the platform upload are left
to the participant.

### V3.3: current local development default

V3.3 adopts three daily controls from the recorded `v3优化` experiments:
`width_scale=0.9`, `drift_cap_horizon=63`, and `horizon_scale_power=0.05`.
One clipped daily mean is shared across horizons; marginal width receives
`0.9 * (steps / 21)**0.05`. Monthly width stays at 1.15 through its separate
setting. The innovation distribution, bootstrap and volatility overlay are not
re-estimated. No data, labels, learned weights, or answer lookup table are bundled.

The parent V3.2 source SHA-256 is
`55f7aa60d2f15f21326c08fed0c297cfcc1b2c4b29f7f996d1d8548f485c707b`.
An exact parent copy and its original documentation/tests are retained in
`v3优化/baseline/`. The experiment source, all settings, per-run source hashes,
frozen task identities, and original score components are under `v3优化/`.

Selection data and procedure:

- The fixed local roster has digest
  `d935ddb4eaeddf3c98e6cbb28acf8124e6cc00064e681a74b237975e0e756b1c`.
  It reconstructs 95 daily public card shapes from the public rates, G10 FX,
  and factors panels verified by `backtest/data.py`. Source manifests, licenses,
  and accepted-file hashes remain in the official repository provenance and the
  backtest cache/roster metadata. Historical vintages are not certified.
- Initial selection uses 800 tasks with cutoffs 2005-01-28 through 2018-11-02;
  their last label is 2018-12-28. There are 28 configurations in two screens,
  using 4,096 draws and seed 0, with three temporal slices reported.
- Eight finalists are evaluated with 4,096 draws and seeds 0/1/2 on all 1,600
  tasks. The 800 late-period tasks have cutoffs 2019-01-04 through 2024-11-05,
  with labels through 2024-12-17. These tasks were already viewed previously
  and are explicitly used for adoption, not represented as untouched validation.
- A further three horizon-scaling configurations are designed after seeing
  multi-horizon regression in that confirmation phase. They are adaptive
  development experiments on the same data. `plan-joint-followup.json` records
  the reason and bounded candidate set; the adoption rule remains unchanged.
- `selection.json` records every finalist, eligibility and decision. The rule
  requires zero failures, better training mean, late-period mean no more than
  0.002 worse than V3.2 and no late temporal-block regression above 0.05.
  Among eligible candidates, minimize the all-task mean; prefer fewer changed
  controls within 0.003. `linear63_h005` has the best all-task mean.

This is selected numerical calibration. Runtime cutoff filtering does not make
selection labels through 2024-12-17 available to earlier historical cards.
V3.3 therefore does **not** claim point-in-time eligibility for those cards,
nor does it repair the inherited V2/V3 parameter provenance below. The panel
first-availability/vintage issue also remains unresolved. Do not interpret local
gate passing as certification of those properties. No submission descriptor is
generated by these experiments; an actual submitted fitted artifact still needs
the disclosures and cutoff eligibility required by the official artifact policy.

The four monthly cards have no new accuracy experiment. Their original outputs
are preserved. The five transfer cards receive public contract checks only.

### Earlier versions

This directory contains numerical code, not bundled language-model weights or an API client.
V3.0 changes failure handling, transform selection, transfer width, daily center guards,
monthly deceleration, and per-asset residual retention. It does not re-estimate the
inherited daily shock hyperparameters.

V3.1 does not change those shock defaults, the center cap, the transfer floor, residual
retention, or fail-open writing. It anchors cumulative log returns at the cutoff, drops
missing values before duplicate-date selection, counts correlation overlap on the same
trailing 756 rows, rejects non-finite or wrongly typed controls, and writes each output
file as mode 0644. Public daily cards already end on the as-of, so the return-anchor
change does not move those samples. This is not a calibration and not an accuracy claim.

V3.2 does not re-estimate shock hyperparameters and does not implement M0. It changes
three submission defaults because published evidence made the previous ones the wrong
side of the denominator: declared daily horizons instead of a calendar bridge, a
trailing-300 drift with a 0.40 cap instead of a 0.15 cap, and width scale 1.15 instead
of 1.0. The coverage figures used for that scale were already viewed. They are not a
fresh holdout and they are not official M0 scales. Monthly walks still follow authored
observation periods. On `t2-F1-sahm-watch-2024` that is 7 and 8 months; the published
M0 table says 8 and 9. The sealed target date is not inferred, so the authored periods
are walked and the disagreement is recorded.

### Inputs and immutable identities

- V3.0 is a descendant of V2.1 in this repository (`forecast-v2/forecast.py`).
- V2 source SHA-256 recorded by that repair: `b1e1e5bcc49331d967d79ce6b9c41f3d126bad07e48c8596d39db4384906267c`.
- V1 source SHA-256: `7f56ec2a1484f051f220870bce3d478db4eeced0f8a655de5912e54c41da13b7`.
- Runtime dependencies and official toolkit revisions are pinned in `requirements.txt`;
  the base and build-tool images are pinned by digest in `Dockerfile`.
- The image uses only the unit's supplied panels, card and spec at runtime, filtered to its cutoff.
  No reconstructed backtest history, stored task labels, API key or task-answer lookup is included.

### What V3.0 did not do

The V2 repair already recorded that its daily defaults were globally selected on pooled
public practice panels covering history through 2024, without a complete per-task cutoff
lineage. V3.0 inherits those shock defaults (`vol_beta_base`, `vol_beta_slope`, block
length, Student-t degrees of freedom, volatility overlay, empirical width exponent).
Runtime prefix filtering does not make that inheritance point-in-time valid for an
early card.

The new transfer floor (2x), center cap (0.15), countertrend-reversion switch, and
monthly deceleration rule are structural guards. They were not selected by searching
the public outcomes. The adoption backtest referenced by V2 is not in this commit and
was not re-run. Local tests check admissibility, V1 sample equivalence under
`--profile v1-control`, and the guards themselves. They are not a leaderboard claim.

### Submission boundary

No descriptor is produced here. Do not infer `models: []` eligibility solely from the
absence of a language-model call if a packaged artifact later includes a fitted model.
These inherited numerical defaults are not, by themselves, a cutoff-clean fitted model.
Disclose any actual packaged fitted model and its training cutoff under the official
artifact policy. House model calls remain absent.
