# Track 3 — Simulation

Team **abc123** placed third on Track 3 and took the T3 Prize. The official score is **3,249,452.2547**, recorded in [`results/RESULTS.md`](../../results/RESULTS.md).

## What it does

The submission is `fastsim`, a Rust rewrite of the ABIDES market-simulation loop, in the public repository:

<https://github.com/EscapedShark/agenthon2026-t3-neurips>

That repository is the canonical copy (Apache-2.0). This showcase links to it. The source stays in that repository.

## How it was verified

[`results/RESULTS.md`](../../results/RESULTS.md) records the team's engineering notes from the T3 poster and the public repo: 2,969 consistency scenarios and 0 row mismatches at release; 142/142 container validations through gates g0–g3; image about 1.8 MB; **45** Rust tests recorded on the team portfolio page. **880×** is the same-machine median of the ABIDES simulation loop against the fastsim whole process, 65 scenarios, range about 66–1,765×. The official Final score is **3,249,452.2547**.

## How to run it

Follow the README in the public repository above. Build and test there. The canonical tree stays in that repository.

## Scope of this directory

This directory carries the link, the result pointers, and the track overview. The fastsim source, posters, and benchmarks live in the public fastsim repository. Development-phase scores for this track sit under the development snapshot in [`results/RESULTS.md`](../../results/RESULTS.md), after the Final table.
