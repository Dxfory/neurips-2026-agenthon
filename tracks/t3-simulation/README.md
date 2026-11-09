# Track 3 — Simulation

Team **abc123** placed third on Track 3. The official score is in [`results/RESULTS.md`](../../results/RESULTS.md).

## What it does

The submission is `fastsim`, a Rust rewrite of the ABIDES market-simulation loop, in our public repository:

<https://github.com/EscapedShark/agenthon2026-t3-neurips>

That repository is the canonical copy (Apache-2.0). This showcase links to it and does not vendor the source.

## How it was verified

[`results/RESULTS.md`](../../results/RESULTS.md) records the team's own engineering notes from the T3 poster and the public repo. They are not organiser-certified: 2,969 consistency scenarios and 0 row mismatches at release; 142/142 container validations through gates g0–g3; image about 1.8 MB. The same file records a same-machine median of 880× on the simulation loop versus the Python reference (65 scenarios, range about 66–1,765×), and states that this is not an end-to-end container speed-up and not a multiple of the leaderboard score.

## How to run it

Follow the README in the public repository above. Build and test there. Do not copy that tree into this candidate.

## What is excluded

The fastsim source, posters, benchmarks, and any private unit or scorer material. Development-phase scores for this track are provisional and live only in [`results/RESULTS.md`](../../results/RESULTS.md).
