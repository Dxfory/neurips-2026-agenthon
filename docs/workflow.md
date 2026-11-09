# Development workflow (draft)

The Track 1 development loop from review v3 through v8 is written up in
[case-study-t1-fix-loop.md](case-study-t1-fix-loop.md).

- We split the model jobs: one model writes a plan/spec, another implements on its own
  branch, a third reviews independently; the team decides.
- Tests first; local gates mirror the official g0–g3 gates; every candidate image is validated in
  a restricted container before upload.
- Experiment ledgers record regressions and abandoned ideas, not only wins.
- The concrete run of that loop on Track 1, from audit v3 through v8, is written up in `docs/case-study-t1-fix-loop.md`.
