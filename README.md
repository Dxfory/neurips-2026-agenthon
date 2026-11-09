# Agenthon 2026 — team abc123

Public showcase of **Team abc123** at [Agenthon 2026](https://www.agenthon.net/) (NeurIPS 2026 Competition Track). We built verifiable AI agents for quantitative finance and **financial-semantics processing** across all four tracks. People are named only as team **abc123**. The member list is [`TEAM.md`](TEAM.md).

[中文简版](README.zh-CN.md)

## Results

Official Final leaderboard, team ranks, read 2026-11-09. Evidence: [`results/RESULTS.md`](results/RESULTS.md).

| Track | Rank | Score |
|---|---|---|
| T1 Coding | 48 / 61 | 0.0357 |
| T2 Forecasting | 8 / 87 | −0.9689 |
| **T3 Simulation** | **3 / 55** (top-3 prize) | **3,249,452.2547** |
| T4 Explainability | 26 / 63 | 0.3779 |

Team abc123 competed in **all four tracks**. T3's third place is a prize position under the published top-3 rule. T4 line: **T4 Explainability track 26/63**.

## Poster

Team poster *The Clock Starts Before Your Code Does* — handle-only byline (`@Dxfory · @Yiping-Yin · @EscapedShark`) — **to be presented** on **12 December 2026** at the NeurIPS 2026 Agenthon Workshop in Atlanta.

![Team abc123 poster for the NeurIPS 2026 Agenthon Workshop](docs/assets/agenthon-2026-t3-poster-team.png)

## The competition

Agenthon 2026 is a four-track NeurIPS 2026 Competition Track on verifiable AI for quantitative finance. Each submission is a Docker agent. The sandbox has no open internet. Before a score is computed, the run has to pass four gates: g0 integrity, g1 schema, g2 cutoff and resource limits, g3 domain semantics. The score itself is computed on sealed held-out units.

Official tasks, starter packs, and datasets are not in this tree. They are CC BY-NC 4.0 and stay upstream:

- <https://github.com/Agenthon-2026/Agenthon2026-public>
- <https://www.agenthon.net/>

## What we built

Agent building and financial-semantics processing were the common thread: each track is a sandboxed agent that computes, cites, or simulates under hard resource and cutoff gates.

| Track | In this tree | In one paragraph |
|---|---|---|
| T1 Coding | [`tracks/t1-coding`](tracks/t1-coding/README.md) | A Python write–run–repair agent (`stack14-thin-v1`) with a hard request cap, atomic admission of model code, and an isolated rerun. `qfkit` and `qfruntime` are libraries with explicit arguments. They do not route a task id to an answer. |
| T2 Forecasting | [`tracks/t2-forecasting`](tracks/t2-forecasting/README.md) | Numeric forecaster v2.7.7. Each change is a pre-registered profile (FOMC-week rate shift, a monthly sibling nowcast, then that nowcast's width). Earlier profiles still reproduce the previous draws. |
| T3 Simulation | [`tracks/t3-simulation`](tracks/t3-simulation/README.md) | A link to our public Rust rewrite, `fastsim`. This tree does not copy it. |
| T4 Explainability | [`tracks/t4-explainability`](tracks/t4-explainability/README.md) | `t4agent` v21.14. Code computes the facts; the House model judges inside a request cap; claims are verbatim spans; fitted artefacts are allowed only when the task cutoff is late enough. |

The internal notes under each `src/` directory are often in Chinese. The track READMEs above are the English overview: what the code does, which checks were recorded, how to run an offline test, and what was left out.

## How we checked the work

The official gates were the specification. Local checks were written to mirror them, and a candidate image was exercised in a restricted container before anyone treated it as submittable.

- **T1.** The exported tree can check that image defaults match the profile (`scripts/thin_defaults.py check`) and can import `qfkit` and `qfruntime`. The full unit-test package and the official tasks are not in the export. During development, a separate fix loop used an independent review pass on each round, on a clean archive, with hashed probes. That record is the case study below. Its pytest counts are path checks.
- **T2.** The bundled README records a pre-submission checklist (source tests, a fuzz, public cards in a constrained container, identity against the registered arm). Re-running `test_forecast.py` needs the official public units, cloned from the track repository and kept out of this tree.
- **T3.** Row-level comparison with the Python reference, as recorded in [`results/RESULTS.md`](results/RESULTS.md). Those engineering figures are the team's, and the file says they are not organiser-certified.
- **T4.** Cutoff policy and the label/point consistency gate have offline tests that do not need a unit checkout (79 passed in this workspace on 2026-11-09). Broader tests need the public track repository. Outcome-selected constants are disclosed in `ARTIFACT_PROVENANCE.md` on T2 and T4, including the practice-unit exception, so a later reader can see which choices were kept after looking at scores.

Private units, hidden labels, checker oracles, and the canary registry are not in this repository.

## How the team used several models

Plans, implementation, and review were different jobs. One model drafted a specification and the tests that should fail. Another implemented on its own branch. A third re-ran the probes on an archive and wrote what the probes had failed to see. The team decided which preset, if any, was allowed onto a real model endpoint. Regressions and abandoned arms stayed in the notes.

The short version of that arrangement is [`docs/workflow.md`](docs/workflow.md). The diagram of the sandbox and the four agents is [`docs/architecture.md`](docs/architecture.md). The long version, for the Track 1 development loop from audit v3 through v8, is [`docs/case-study-t1-fix-loop.md`](docs/case-study-t1-fix-loop.md): what the probes caught, what was fitted to the probes, what remained out of reach, and why a later clock constant stopped both lanes.

## Layout

```
README.md                 this file
README.zh-CN.md           short Chinese version
TEAM.md                   member list (handles only)
LICENSE                   Apache License 2.0
NOTICE                    copyright and third-party notes
results/RESULTS.md        official scores and the development snapshot
docs/architecture.md      sandbox diagram
docs/workflow.md          how we used several models
docs/case-study-t1-fix-loop.md
docs/case-study-t1/       findings by version
docs/assets/              team poster (handle-only byline PNG)
tracks/t1-coding          Final-line Python agent, qfkit, qfruntime, build
tracks/t2-forecasting     v2.7.7 forecaster and pipeline notes (no card data)
tracks/t3-simulation      link to the public fastsim repository
tracks/t4-explainability  t4agent v21.14 source and tests
scripts/export_candidate.sh
scripts/leak_scan.sh
```

`scripts/export_candidate.sh` rebuilds the three source trees from an allowlist. It does not push. `scripts/leak_scan.sh` reports credential-shaped strings and disallowed identifiers.

## Links

- Competition site / Final leaderboard: <https://www.agenthon.net/>
- Public Track 3 (`fastsim`): <https://github.com/EscapedShark/agenthon2026-t3-neurips>
- Team poster: [`docs/assets/agenthon-2026-t3-poster-team.png`](docs/assets/agenthon-2026-t3-poster-team.png)

## License

Apache License 2.0. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).

- Track 3 (`fastsim`) is linked, not merged — its own Apache-2.0 NOTICE stays upstream.
- Official Agenthon materials are CC BY-NC 4.0 and stay link-only.
- Official packages such as `qfbench2-common` keep their upstream non-commercial terms; they are not redistributed here.
- `backtrader` (GPL-3.0) may appear in Track 1 container environments only; this tree does not import or ship it.
