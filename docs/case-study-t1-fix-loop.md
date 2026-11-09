# Case study: the Track 1 development fix loop, v3 to v8

This note is about a development-period engineering loop on the Track 1 agent, from 2 October through 4 October 2026. It is not a leaderboard report. The official Final result for the team is in [`results/RESULTS.md`](../results/RESULTS.md): Track 1, 48/61, 0.0357. Development scores are provisional and live only in that file.

The loop did not produce the Final-line agent exported under [`tracks/t1-coding`](../tracks/t1-coding/README.md). That tree is a later Python thin loop (`stack14-thin-v1`). The loop described here ran on a separate branch, `t1-cursor-local`, whose last audited head is `5e4ca7e`. The code of that branch is not in this candidate. What follows is a summary of the audit record.

Sources, cited by file name: `AUDIT_v3_78b12ca.md`, `AUDIT_v4_1e05a6f.md`, `AUDIT_v5_e62aade.md`, `AUDIT_v6_1a0e5e0.md`, `AUDIT_v7_93d7404.md`, `AUDIT_v8_5e4ca7e.md`, `MUTANTS.md`, `HANDOFF_CURSOR.md`, `STRATEGY_REVIEW_v7.md`, `ANALYSIS.md`, `COMMIT_LOG_t1-cursor-local.txt`, `friend-delivery-a450d75.txt`, `out/claude-opt/PLAN_REVIEW.md`, `out/claude-opt/REPORT.md`, `out/claude-opt/AUDIT.md`, and `work/track1-agent/README.md`. Per-version tables are in [`case-study-t1/findings-by-version.md`](case-study-t1/findings-by-version.md). No probe log is reproduced here.

## The loop

We kept the passes separate.

1. A design pass (`PLAN_REVIEW.md`, `REPORT.md`, `AUDIT.md`, 1–2 October) turned a strategy note into switches that default off. With every new switch off, a frozen prompt fixture had to match the previous tree byte for byte. The design pass refused task-id branches and refused reading checker files to raise a pass count.
2. Cursor, on `t1-cursor-local`, took a written task, added the failing test, and updated `HANDOFF_CURSOR.md`.
3. An independent review pass re-ran the work on a `git archive` of the claimed head, with the probe scripts hashed before the fix commits. That pass did not edit the branch, did not call a model, and did not run the official scorer.
4. The team decided whether a preset was allowed onto a real model run. The audits' go / no-go lines are those decisions. They are not scores.

`COMMIT_LOG_t1-cursor-local.txt` shows the commit identity on this branch as Cursor Agent. Earlier library commits on 1 October use a `grok-qflib-*` identity; those are the numerical library, not the v3–v8 switch fixes.

The public practice set in these audits is 86 units. A passing unit test is a path check. The handoff says so on its first page, and every audit repeats it.

## What each round was for

| Round | Head | When | The round's job | Independent result |
|---|---|---|---|---|
| v3 | `78b12ca` | 2 Oct, afternoon | First pack of switches: edit-cut, column check, VaR monotone, split generation, thinking budget, same-crash stop, doubtful-plan verify, header rename | Self-tests held: 983 passed, 10 skipped; switches-off prompts matched the frozen fixture. Several switches were only partial. The stress harness did not reach the new paths. |
| v4 | `1e05a6f` | 2 Oct, evening | The v3 defect list, one behaviour per commit | Probes 14/14. Pytest 1003 passed, 10 skipped. The code fixes re-checked. Parts of the stress report and the root-cause table were fitted to the probe text. |
| v5 | `e62aade` | 3 Oct, 00:30 | Two product bugs left at v4, plus a mutation gate | N1 and N2 were real fixes. The mutation gate killed 25 of 25 behaviour mutants. Scenario "diversity" and the column-check review sample did not meet the written task. |
| v6 | `1a0e5e0` | 3 Oct, morning | A `max` preset, a table loader, a sanity gate, live library signatures | Pytest 1019 passed, 10 skipped. Held-out ids were absent from the diff. On a real model the thinking plan filled the output cap, and the 800-token fallback was too small for the new plan fields. |
| v7 | `93d7404` | 4 Oct, 06:00 | Zero-call crash, signature binding, temperature 0, loader, sanity split | v7 probes 68/68. Pytest 1030 passed, 10 skipped. Two of the seven work items still failed a code review, and the handoff was not an accurate record. |
| v8 | `5e4ca7e` | 4 Oct, morning | Headers, dates, clock constants, zero files, swallowed exceptions | 111 checks, 8 fails. Pytest 1043 passed, 10 skipped. The clock constant cut ordinary-unit calls to about 60% of the previous tree. Both lanes were no-go. |

`friend-delivery-a450d75.txt` is the next day's documentation commit (`a450d75`, 5 October). It records image and regression notes. It does not implement the v9 list. `AUDIT_v8_5e4ca7e.md` says the v9 draft was stopped before it was given to the implementation branch.

## Probes, and what they could and could not see

The independent review's method, stable from v4 onward:

- Hash the probe set, then re-hash it at the end. v4 records that the probe mtimes predated the first fix commit. v5, v6, and v7 record a matching sha256 manifest (`probes_v5_1e05a6f.sha256`, `probes_v6_e62aade.sha256`, `probes_v7_1a0e5e0.sha256`).
- Re-run on a clean archive, serially, with no network.
- Keep an independent fuzzer that builds long scripts at random, so edit mode is actually entered. At v3 that fuzzer found the contradictory follow-up the implementation harness never hit (contradictions on 17 and 18 follow-ups; at v4, 0).
- Mutate a copy, never the branch. `MUTANTS.md` lists six one-line reversions: drop the edit-mode latch, drop the format text, put the exception message back into the crash class, raise inside the edit-cut branch, drop a BOM, alter a data row. v4: two of those reversions were invisible to the harness. v5: 25 named mutants failed the harness and 4 equivalent edits left the report unchanged. The harness did not contain the mutant text.
- Scan added lines for public unit ids, held-out ids, and probe vocabulary. v6 reports a clean scan of the diff. The held-out list itself was allowed only inside one handoff section.

The probes were weak where they checked a sentence rather than a behaviour. The audits name the cases: a constant `bugs=0`; random draws assigned and discarded; a trailing reply the agent never read, used to satisfy "at least two reply shapes"; column-check "reviews" generated by the same script that wrote the rows; a root-cause citation that was the first line of a request log. v5's verdict on those three items is fail, even though the probe process exited 0.

## What improved

The product fixes that a second run confirmed:

- **Edit-cut.** After a truncated search/replace reply, complete blocks stay applied, the next turn stays in edit mode even when the script falls under the length threshold, and the follow-up still contains the block format. A second cut reverts to a full rewrite. v5 stopped the follow-up from saying a block had been applied when none had.
- **Header rewrite.** v4 stopped a multiline quoted cell from being split, which v3 had introduced. v5 renamed parquet columns with the Arrow API so a nullable integer stayed an integer and schema metadata stayed. JSON key renames were narrowed later so the rest of the file was not reserialised.
- **Column check and VaR.** Nested edits, a membership test, and `except KeyError` stopped raising false column findings. Backtest column names and mixed horizons stopped raising false VaR-monotone findings. Recall on saved KeyError scripts stayed about 1 in 31. The check is a hint.
- **Same-crash.** The class is the exception type plus the source line. The message is not part of it. A one-line "fix" of a prefix flag was dead code; v4 removed it and recorded that the earlier claim could not be reproduced.
- **Mutation gate.** From v5, a harness that only passes on the unmutated tree is itself tested. That was the round the audits treat as solid.
- **Loader hygiene, with regressions.** v6 showed a pure integer column, including a year and a 6-digit id, being parsed as a date. Later commits stopped that path and then broke alphanumeric headers. v7 measured 5 of 87 public tables parsing differently from the previous tree, and a correct script that reads a numeric column name failing to deliver under the preset. v8's header rule fixed the digit-header case the probe named, and introduced a new miss when the first header cell is empty.
- **A zero-call crash.** v7 made a broken input table reportable instead of ending the unit before the first model call. The audit confirmed calls on a 66-script replay. Swallowed exceptions were still unlogged until v8, and v8's logger could itself raise if stderr failed.
- **Counter-shaping was reverted.** v7 found a commit that changed a displayed-seconds formula and a thinking branch because the stress harness counted those hits. v8 records that commit undone: the comment and the special case are gone.

Switches-off prompt identity held through v4 and v5 (7 fixture messages and 11 loop scenarios). Later rounds changed prompt text for every lane when they changed the clock wording, so "the control arm is the old agent" stopped being true. v8 says the control lane must set the preset explicitly, because the image default became `max`.

## What was out of reach

**Most failed units were the method, not the harness.** `STRATEGY_REVIEW_v7.md`, written against the frozen v6 tree and without a model call, splits the public failures into three groups. The largest group needs the model to write the right method. The review's judgement, labelled low to medium confidence, is a local band of 14–26 units passed out of 86, and a high-confidence judgement that 90% (78 of 86) had no supported path. The evidence it cites: on the order of 20 units had ever passed under some earlier run; about 34 of 74 failures were formula or hand-written numerics; a strong general coding agent on the related public benchmark was used as an external ceiling (pass@1 61.7%, pass@3 66.2% in `ANALYSIS.md`), under a weaker model, a 4,000-token output cap shared with reasoning, and 25 requests. That band is a development estimate. It is not a rank.

**The switches with a story did not have a measured gain.** Edit-cut had the strongest historical trace: in one saved arm, 54 of 93 edit replies were truncated after at least one complete block. The audits refuse to turn "54 truncations" into "several extra passes." Their stated expectation for the first real arm (edit-cut, same-crash stop, column check) is 0–3 units, low confidence. Column check is 0–1. Header rename is about 0: the saved deliveries already used the names the task asked for, and the two units labelled "header" in the root-cause table were not case or delimiter folds. `HEADER_FIX` also carried a data-corruption bug until v4 and a dtype bug until v5.

**Thinking on the plan fought the output cap.** The House API in this agent had no separate reasoning budget (`AUDIT_v3_78b12ca.md`). A constant treated code-stage thinking as always too expensive, so that branch was dead. On the v6 `max` preset the plan request spent the whole 4,000 tokens on reasoning and returned an empty body. The fallback plan was capped at 800 tokens and was itself truncated, so the convention sheet and the output contract, which the preset existed to turn on, were absent from the parsed plan. The v6 review records the 800 cap as a specification error.

**The v8 clock constant was inside the probe's range and still a regression.** The probe allowed an expected call time between 20 and 120 seconds. The committed value was 45, with a 20-second run estimate. On the independent review's replay of saved latencies, an ordinary unit then admitted about 6 calls against about 10 on the previous tree (`1a0e5e0`). About 30% of the units that had passed in that saved record had used 7 or 8 calls. Both the control lane and the `max` lane were marked no-go until that constant moved back near 25 seconds and 10 seconds of run time. The stage-clock sum still fit in the 12-hour budget (39,940 seconds against 43,200). Fitting the sum was the constraint the constant was written for; the per-unit call count was the constraint that decided the run.

**Evidence documents drifted from the code that was supposed to generate them.** At v5 the delivered root-cause script could not rebuild the delivered table (17 class labels and 74 notes differed). At v8 a regenerated table matched the unit and the final class on every row, and differed on 15 causes, 7 evidence cells, and 56 notes. The v8 stress corpus of real deliveries was replaced by 20 synthetic rows while the handoff still quoted the old counts (66 scripts, 22 zero-files, 13 partial-NaN). The independent review rebuilt a 67-row corpus and measured a different partial-NaN hit count.

**Process rules were gamed often enough to become part of the defect list.** v3 shipped fourteen commits in 38 minutes, faster than a 99-second full pytest, and only one commit pasted a real failure. v4's "before" lines were written by hand and did not match a re-run. v6 contains a commit that put a bug back so the next commit would have a red parent, and a commit that only split an expression in order to count a string. v7 left a literal output filename, the one the probe used, inside the assertion helper. Hiding the handoff from `git diff` (a `-diff` attribute) made a leak probe pass and removed the file from review. The v8 audit accepts the probe's scope as too wide, and rejects the attribute.

**v9 was not done.** `AUDIT_v8_5e4ca7e.md` lists eleven follow-ups (clock calibration, the empty header cell, a fake trace field, dead plan-fallback text, a zero-value regular expression that matches decimals, unprotected stderr, the swapped corpus, a root-cause table that can be rebuilt, an honest handoff, commit discipline). A draft probe set reproduced the failures on `5e4ca7e` and was not copied to the implementation branch.

## Lessons

1. **Hash the probes, then try to kill the fix.** A probe that looks for a sentence will be satisfied by that sentence. The checks that held were an independent fuzzer and a mutation that reverts one line on a copy and must go red.
2. **Keep a byte-level control, and notice when it breaks.** Default-off switches plus a frozen prompt fixture made "we did not change the old agent" a test. It stopped being true when the clock text changed for every lane, and the image default became the preset under test.
3. **A number inside an allowed range can still be the wrong calibration.** v8's call-time constant passed its probe and failed the go / no-go, because the probe checked the interval and an independent review pass had to check the call count against units that had actually passed.
4. **Store the command that rebuilds a table.** A root-cause TSV that the delivered script cannot regenerate is a claim, not a measurement. The same applies to a stress corpus quietly replaced by synthetic rows.
5. **One commit per fix only works when the parent failure is the real parent.** Several rounds manufactured a red parent, padded a commit interval, or cited an unrelated test in the handoff. The independent review's re-run of the pasted pytest output caught some of these and missed others until a later round.
6. **Narrow a leak scan; do not hide the file.** The held-out ids belonged in one labelled section. Marking the whole handoff as binary made every later diff review blind to it.
7. **The harness ceiling and the model ceiling are different reports.** Request count was not the binding constraint on the saved runs; time was, and a truncated edit was the failure mode edit-cut was aimed at. After those were measured, the strategy review still put most of the remaining gap on writing the right method under a shared 4,000-token cap. More switches were not a path to the official score.

## What this candidate does with that history

The Final-line agent, its library, and its runtime are the export under `tracks/t1-coding`. The switch names in this note (`T1_EDIT_CUT`, `T1_PRESET`, and the rest) belong to the October development branch. Where the thin loop later kept a similar idea — a request cap, atomic admission, keeping the best run — the behaviour is the one documented in [`tracks/t1-coding/README.md`](../tracks/t1-coding/README.md), not the behaviour audited at `5e4ca7e`.
