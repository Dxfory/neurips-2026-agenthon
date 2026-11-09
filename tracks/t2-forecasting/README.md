# Track 2 — Forecasting

Numerical forecaster **v2.7.7** for Agenthon 2026 Track 2. The official score is in [`results/RESULTS.md`](../../results/RESULTS.md). The code under `src/T2/forecast-v2_7_7/` is the Docker build context for the numeric forecaster.

## What it does

Each card produces a probabilistic forecast. Changes are pre-registered experiments with a profile switch, so an earlier version can be reproduced on the same card:

| Version | What changed | Profile that reproduces the previous version |
|---|---|---|
| v2.7.5 | Seasonal adjustment when a monthly series' own history shows a calendar effect | `--profile v2.7.4` |
| v2.7.6 | FOMC-week shift of Treasury note and bill centres (E37); monthly sibling nowcast when a stable series already has the target month (E40b) | `--profile v2.7.5` |
| v2.7.7 | Width of that nowcast (E41): recent residual volatility, OLS prediction variance, and a constant per target kind | `--profile v2.7.6` |

E38, E39, and an earlier E40 stayed off the adopted line. The source README records a later review (E42, E43) that left v2.7.7 unchanged. Outcome-selected choices are described in `src/T2/forecast-v2_7_7/ARTIFACT_PROVENANCE.md`. `src/T2/README.md` and `FORECAST_PIPELINE.md` are the Chinese engineering record; several sections point at contracts, research notes, and backtests that stay upstream of this export.

The source README records the published image `docker.io/greedyshark/forecast-v3:2.7.7` (linux/amd64, anonymous pull checked on 2026-10-21).

## How it was verified

`src/T2/forecast-v2_7_7/README.md` records the 2026-10-19 pre-submission checks: 149 source tests, packaging and contract tests, a 70-card fuzz, 126/126 public cards in a constrained container with a fallback count of 0, identity against the E41 arm, and 40 Final-shaped cards. Those counts are the team's development verification. The official score is the Final table in [`results/RESULTS.md`](../../results/RESULTS.md).

## How to run the offline tests

`test_forecast.py` loads public cards from an upstream checkout:

`src/T2/official_repo/track2-forecasting-public/units`

Clone the official track from [track2-forecasting-public](https://github.com/Agenthon-2026/track2-forecasting-public) into that path and keep the checkout upstream of any redistributed bundle. Install the pins in `src/T2/forecast-v2_7_7/requirements.txt` from the public toolkit, then:

```sh
cd tracks/t2-forecasting/src/T2/forecast-v2_7_7
python3 -m unittest test_forecast.py
```

The module's own header says the tests use the numeric path, with the text model and the API left off. The suite collects its fixtures from the official units. This export carries the forecaster source.

## Scope of this directory

This page describes the v2.7.7 forecaster export. Official units, the contracts and research trees, backtests, experiment logs, submission zips, and credentials stay upstream. A private-repo link in `FORECAST_PIPELINE.md` is already replaced with the placeholder `<team-private-repo>`.
