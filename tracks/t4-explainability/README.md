# Track 4 — Explainability

`t4agent` **v21.14** for Agenthon 2026 Track 4 (analysis score plus a post-hoc reasoning term). The official score is in [`results/RESULTS.md`](../../results/RESULTS.md). Source is `src/T4/agent_v21_11/` (`t4agent` 21.14.0). The Chinese READMEs under `src/T4/` are the version log; this page is the overview.

## What it does

Code computes the facts. The House model handles the bounded judgements.

1. Read documents the manifest assigns to that entity, at or before the task cutoff. Retrieval is BM25 plus finance key sentences.
2. Code baselines: statement comparisons, an EPS seasonal anchor, a macro-revision base rate, a series baseline, a local CPI nowcast from weekly oil prices in the corpus, and a volatility scale.
3. One planning call, then analyst samples. Classification rows with a split label get a further single-entity vote. The source README records a cap of 23 requests on a unit whose platform cap is 25.
4. The submitted point blends the model with the code baseline. The label and the interval come from the same distribution. Fitted artefacts are gated by `t4agent/artifact_policy.py`: a capability is admitted when the task's exact cutoff is on or after that capability's bound. Missing or invalid dates fail closed. Outcome-selected constants are disclosed in `ARTIFACT_PROVENANCE.md`.
5. Every claim is a verbatim span from a document the manifest assigns to that entity.
6. After the forecast is fixed, the model writes one to three `submitted_reasons`. Code checks that their meaning matches the answer, and writes a fallback if time runs out.

v21.14 adds one consistency check on top of v21.13. When a classification label and the point disagree, the label stays and the point moves to the nearest legal published value inside the label region. The interval widens enough to contain the new point. v21.13 added sentence-level claim selection, a model-stage timeout guard, salvage of a truncated reasons array, and a fix that kept a leading minus on an EPS figure. The image clock defaults recorded in the source README are a 360 s hard stop and a 290 s model deadline, both lower than the card timeout, because Final units share one stage clock.

## How it was verified

`src/T4/README.md` records, for the versions that led to v21.14, offline pytest, 11 public units in a restricted linux/amd64 container (non-root user, root filesystem mounted for reads, offline), and the clock values above. Development leaderboard figures stay in that Chinese log as a development record. The public lead is the Final table in [`results/RESULTS.md`](../../results/RESULTS.md).

On 2026-11-09 the three suites below passed in this workspace: **79 passed**.

```sh
cd tracks/t4-explainability/src/T4/agent_v21_11
python3 -m pytest -q -p no:cacheprovider \
  tests/test_artifact_policy.py \
  tests/test_threshold_consistency.py \
  tests/test_threshold_consistency_xc.py
```

They use the standard library, pytest, and `t4agent`. The three suites run inside this export.

## How to run the rest of the offline tests

The broader suite (`tests/test_offline.py` and several v21.11 files) imports `qfbench2-common` and expects the official track checkout at `src/T4/track4-analysis-public`. Install it from the public repos and keep that checkout upstream of any redistributed bundle:

- <https://github.com/Agenthon-2026/Agenthon2026-public>
- <https://github.com/Agenthon-2026/track4-analysis-public>

`src/T4/README.md` shows the venv and pytest invocation used for earlier agent directories. Point it at `agent_v21_11`. Tests that read `calib_data/` run when that directory is present. Leave the team key unset, and point the tests at the public units. Container verification uses Docker and the scripts under `sub/`, which live with the build tree upstream of this export.

## Scope of this directory

This page is the v21.14 overview. Official corpora and units, `calib_data`, the error book, submission zips, run logs, credentials, model weights, and the earlier `agent_v21_*` directories stay upstream. Published image namespaces that appear in the Chinese logs include `greedyshark/t4-agent` (Docker Hub).
