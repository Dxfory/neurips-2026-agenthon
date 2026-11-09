# Track 1 — Coding

Final-line Python agent for Agenthon 2026 Track 1. The official score is in [`results/RESULTS.md`](../../results/RESULTS.md). This page describes the exported tree.

## What it does

The image verb is `solve`. Profile `stack14-thin-v1` (`t1_phase1/thin.py`) runs a bounded write–run–repair loop:

1. A short contract request, then a code request. The controller names the action (replace, bundle, repair one unit, or append). The House model returns code.
2. The reply is admitted atomically. A cut reply, a broken prefix, or a program split across fences is refused and the reason is recorded.
3. An admitted program runs in a fresh isolated process. The run that delivered more files stays when a later run delivers fewer.
4. Retries and continuations count toward the request cap. The source README records a Final cap of 25. Transport failures back off at 2, 4, 8, 15, 25, and 40 seconds and honour `Retry-After`.

`T1_THIN_DIRECT_STEP` and `T1_RUNTIME` default off. Image thinking defaults are `low_effort` on the contract request and `off` on code requests.

`qfkit` is a numerical library (panels, moments, pricing, risk, rates). Callers pass axes, units, and methods explicitly. Task selection and stored answers stay with the controller. `qfruntime` adds contract checks, bounded numerical probes, and provenance helpers. The controller offers it when a profile turns it on. Task identifiers and reference answers stay with the controller, outside these two packages.

The internal notes under `src/T1/` are partly in Chinese. They record a phase-1 baseline (one model call, then an isolated run) and a later decision, dated 2026-10-22 in `src/T1/README.md`, that this Python thin loop is the Final candidate and that a native C++ line was retired. Local pass counts in those notes are development measurements.

## How it was verified

`src/T1/README.md` records a native AMD64 run of the phase-1 baseline: unit tests, nine integration scenarios, acceptance checks, isolation probes, and an outer timeout fixture, with the House call left off. The same file records a GHCR publish target, `ghcr.io/yiping-yin/agenthon-t1`. An anonymous pull is a separate check from a successful push. The workflow file named there lives upstream of this export.

In this tree, `python3 scripts/thin_defaults.py check` (run from `src/T1`) confirms that the image ENV defaults in both Dockerfiles match `t1_phase1/solver_profile.py`. That check passed in this workspace on 2026-11-09.

## How to run the offline checks

From `tracks/t1-coding/src/T1`, offline, with model credentials left unset:

```sh
python3 scripts/thin_defaults.py check
PYTHONPATH=qfkit:qfruntime python3 -c "import qfkit, qfruntime"
```

The `unittest discover -s tests` command in `src/T1/README.md` uses a `tests/` directory that lives with the development tree, upstream of this export.

Building an image and running `scripts/integration_fixture.py` or `scripts/timeout_fixture.py` uses Docker, the locked AMD64 wheelhouse, and a new output directory. Those fixtures talk to a synthetic server on container loopback. They are the offline container checks. `scripts/fetch_sources.py` points at the public starter repository, where the official tasks stay: [Agenthon2026-public](https://github.com/Agenthon-2026/Agenthon2026-public) and [track1-coding-public](https://github.com/Agenthon-2026/track1-coding-public).

## Scope of this directory

This page describes the exported Final-line Python agent. Official task copies, corpora, checker materials, the `tests/` suite, evidence bundles, wheel binaries, run logs, credentials, and the retired C++ line stay upstream. The development-period switch loop (v3–v8) is a separate story: [`docs/case-study-t1-fix-loop.md`](../../docs/case-study-t1-fix-loop.md). The Final-line agent on this page is the Python thin loop.
