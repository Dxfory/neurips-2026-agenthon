# Track 2 — Forecasting

Numerical forecaster **v2.7.7** for Agenthon 2026 Track 2. The official score is in [`results/RESULTS.md`](../../results/RESULTS.md). The code under `src/T2/forecast-v2_7_7/` is the Docker build context. It does not call a House model.

## What it does

Each card produces a probabilistic forecast. Changes are pre-registered experiments with a profile switch, so an earlier version can be reproduced on the same card:

| Version | What changed | Profile that reproduces the previous version |
|---|---|---|
| v2.7.5 | Seasonal adjustment when a monthly series' own history shows a calendar effect | `--profile v2.7.4` |
| v2.7.6 | FOMC-week shift of Treasury note and bill centres (E37); monthly sibling nowcast when a stable series already has the target month (E40b) | `--profile v2.7.5` |
| v2.7.7 | Width only of that nowcast (E41): recent residual volatility, OLS prediction variance, and a constant per target kind | `--profile v2.7.6` |

E38, E39, and an earlier E40 were not adopted. The source README records a later review (E42, E43) that left v2.7.7 unchanged. Outcome-selected choices are described in `src/T2/forecast-v2_7_7/ARTIFACT_PROVENANCE.md`. `src/T2/README.md` and `FORECAST_PIPELINE.md` are the Chinese engineering record; several sections point at contracts, research notes, and backtests that are not in this export.

The source README records the published image `docker.io/greedyshark/forecast-v3:2.7.7` (linux/amd64, anonymous pull checked on 2026-10-21).

## How it was verified

`src/T2/forecast-v2_7_7/README.md` records the 2026-10-19 pre-submission checks: 149 source tests, packaging and contract tests, a 70-card fuzz, 126/126 public cards in a constrained container with no fallback, identity against the E41 arm, and 40 Final-shaped cards. Those counts are the team's development verification. They are not the official score.

## How to run the offline tests

`test_forecast.py` loads public cards from a checkout that is intentionally absent:

`src/T2/official_repo/track2-forecasting-public/units`

Clone the official track (do not copy it into a published bundle) from [track2-forecasting-public](https://github.com/Agenthon-2026/track2-forecasting-public) into that path, install the pins in `src/T2/forecast-v2_7_7/requirements.txt` from the public toolkit, then:

```sh
cd tracks/t2-forecasting/src/T2/forecast-v2_7_7
python3 -m unittest test_forecast.py
```

The module's own header says the tests use no text model and no API. Without the official units the suite cannot collect its fixtures. This export has no card data of its own.

## What is excluded

Official units, the contracts and research trees, backtests, experiment logs, submission zips, and credentials. A private-repo link in `FORECAST_PIPELINE.md` is already replaced with the placeholder `<team-private-repo>`.
