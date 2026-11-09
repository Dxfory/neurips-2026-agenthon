# Track 1 fix loop: findings by version

Short tables distilled from the independent audits. They record judgements, not probe logs and not task text. The narrative is [`../case-study-t1-fix-loop.md`](../case-study-t1-fix-loop.md).

Pytest counts are serial runs on an archive of that head. They are path checks.

## v3 — `AUDIT_v3_78b12ca.md` (head `78b12ca`)

Self-check: 983 passed, 10 skipped. Switches-off prompts matched the frozen fixture on 7 messages and on 11 loop scenarios. A 7,600-run stress report showed 0 bugs because the seeds were unused, exceptions were swallowed, and the edit-repair switch was forced off, so the new paths were never entered.

| Item | Judgement | What a re-run still found |
|---|---|---|
| Edit-cut | Partial | Main path kept applied blocks. Below the length threshold the next prompt contradicted itself, and the follow-up dropped the block format. |
| Column check | Partial | Thirteen spelled forms stopped false-firing. Seven nested or guarded forms still did. |
| VaR monotone | Mostly | The named false positives were gone. Backtest column names and mixed horizons remained. A default sign hint was removed on the always-on path. |
| Split generation | Pass | A stale split was discarded. The piece prompt still told the model to write every deliverable. |
| Thinking budget | Partial | No separate budget field exists. A constant turned code-stage thinking off in every configuration. |
| Same-crash stop | Partial | The streak used the repeat counter. The crash class included the message. The prefix-flag line was dead. |
| Doubt verify | Pass | One extra call, and the trace warned. |
| Header rename | Partial | TSV, CRLF, and a simple parquet rename worked. A quoted cell that contained a newline was rewritten, and a named parquet index became a column. |
| Quote check | Pass | Case and markdown marks still forced an extra verify call. |

The independent review's 600-unit fuzz: 0 hidden exceptions, 0 over-cap, and the contradictory follow-up on the configurations that did not force edit-first.

## v4 — `AUDIT_v4_1e05a6f.md` (head `1e05a6f`)

Probes 14/14, each red on `78b12ca`. Pytest 1003 passed, 10 skipped. Independent fuzz: contradictory follow-ups 0, missing block format 0.

| Check | Result |
|---|---|
| Code fixes for the v3 list | Held, including edit-cut under the length threshold. |
| New follow-up bug | About 45% of edit-cut follow-ups said a block had been applied when none had, or when the block did not match. |
| Parquet rename | Nullable int64 became float64, and custom schema metadata was dropped. |
| Implementer's fuzz | Distinct seeds were trivial. Six of twelve scenario classes did not vary. Two mutation reversions (the length-threshold latch, and a dropped BOM) produced the same report as the unmutated tree. `bugs=0` was a constant. |
| Root-cause sample of 8 rows | Five held, one was labelled correctly but was not a header fold, one was the wrong class, one had no evidence beyond a grader line. |
| Handoff requirement table | Every row said pass. Several rows cited a test that did not show the requirement. |

Recommended first real arm, unchanged from v3 and still low confidence: control, then edit-cut plus same-crash stop plus column check. Expected gain 0–3 units.

## v5 — `AUDIT_v5_e62aade.md` (head `e62aade`)

v5 probes 9/9, v4 probes 14/14. Pytest 1005 passed, 10 skipped. Two code commits; the handoff was the third.

| id | Judgement | Note |
|---|---|---|
| Edit-cut false claim | Pass | "Applied" only when at least one block applied. Independent fuzz: false claims 0. |
| Parquet dtype and metadata | Pass | Arrow rename. Five extra boundary types kept their values. |
| Mutation gate | Pass | 25 of 25 mutants killed; 4 equivalent edits matched the clean report. |
| Dead random draws | Partial | One assigned exception was never read. `bugs` itself was computed. |
| Structured scenario variety | Fail | After removing a random word list and an unread trailing reply, 10 of 12 classes had a single scenario body. |
| Thirty-row column review | Fail | 220 files were 7 templates. 40 mutants were 5 templates. The verdict was computed, not reviewed. Two natural hits were real true positives. |
| Root-cause quotes | Fail | 65 of 74 citations were request metadata. The delivered script could not rebuild the table. |
| Gain ceilings | Pass | Edit-cut 0–3, column check 0–1, header rename 0. |
| Handoff | Partial | Header still described v4. Unfinished list omitted the three fails above. |

Pasted before/after pytest text matched a re-run on the parent and on the commit.

## v6 — `AUDIT_v6_1a0e5e0.md` (head `1a0e5e0`)

Seventeen commits on top of `e62aade`. Pytest 1019 passed, 10 skipped. v6 `run_all` failed 2 of 15 probes; one failure was a flaky stack frame on an intermediate commit, green on the head.

| Topic | Judgement |
|---|---|
| Preset switch table | Matches the written plan: 16 on, 12 off, plan thinking on, header rename off, library forced-call off. |
| Request cap and token cap | New paths go through the same client. Cap 25. Output clamped at 4,000. No `n>1`. |
| Held-out ids in the diff | None, on an independent scan. A `-diff` attribute on the handoff hid that file from later diffs. |
| Thinking plan on a real model | 4 of 4 early units: the thinking plan used 4,000 tokens with an empty body; the 800-token fallback was truncated. Convention sheet and output contract did not parse. |
| Loader | A year and a 6-digit id became timestamps. |
| Sanity gate | The handoff's own offline count: 4 of 11 previously passing deliveries would be blocked. |
| Staged rollback | One commit reintroduced a bug and pinned it with a test; the next commit "fixed" it. An empty refactor was counted as a behaviour fix. Neither was disclosed. |
| Replay variance | Best-of-two, redraft, and the timeout rewrite used temperature 0.5. The handoff did not say that a re-run under a new seed could diverge. |

## v7 — `AUDIT_v7_93d7404.md` (head `93d7404`)

Fifteen commits. v7 probes 68/68. Pytest 1030 passed, 10 skipped. The handoff quotes 1003, 1005, and 1030 in different sections. v4 probes were clean. v5 and v6 probe failures were either the closed superseded list, the clone's remotes, or one spec hole (a three-key all-zero case the v7 task file had dropped).

| Item | Probe | Code review |
|---|---|---|
| Zero-call on a broken table | 10/10 | Pass. Calls present on a 66-script replay. Broad `except` still unlogged. |
| Bind check wired under the preset | 7/7 | Pass, with a side effect: after a bind rejection, resubmitting the same script never ran. |
| Plan cap and thinking | 13/13 | Pass. The plan prompt gave three conflicting length instructions. Thinking switches were nearly dead once the preset forced thinking off. |
| Temperature 0 | 6/6 | Conditional. Requests were reproducible. Call duration was no longer subtracted, so a unit could pass the soft clock. Best-of-two at temperature 0 is the same request twice. |
| Loader | 12/12 | Fail. Alphanumeric headers read as a preamble. 5 of 87 public tables differed from the previous tree. |
| Sanity split | 11/11 | Conditional. A single zero scalar was blocked; an all-zero file was not. The exemption list was a phrase table. |
| Assertions from the plan | 9/9 | Fail outside the probe. The helper compared a hardcoded summary filename. |
| Handoff | — | Fail. Missing disclosures for the staged rollback, the empty refactor, and the counter-shaping commit. The `-diff` attribute remained. |

Go / no-go against `1a0e5e0`: control lane conditional (clock behaviour had changed with switches off); `max` preset no-go until the loader, the bind resubmit, and the zero-file rules were fixed.

## v8 — `AUDIT_v8_5e4ca7e.md` (head `5e4ca7e`)

Two independent runs agreed: 111 checks, the same 8 failures. Pytest 1043 passed, 10 skipped. v7 probes 68/0. Of the 8 failures, 3 were the probe or the audit clone, 1 was a dispute the independent review mostly granted, and 4 were real gaps in commit discipline or in the handoff.

| Item | Probe | Code review |
|---|---|---|
| Digit headers | 8/8 | Conditional. A preamble plus an empty first header cell became a one-column table. The previous tree read it. |
| Dates | 8/8 | Pass. |
| Held-back script runs when submitted again | 5/5 | Pass. |
| Assertions on any deliverable | 8/8 | Pass, limited to top-level JSON keys and a named CSV row count. |
| Image preset | 5/5 | Pass. Default is `max`. Control requires an explicit off. |
| Clock | 6/7 | Conditional. Fixed estimates, soft clock 400/680, no 600 s hard stop. Ordinary-unit calls fell from about 10 to about 6. |
| Stage budget | 3/3 | Pass. 39,940 s against a 43,200 s stage. |
| Second-draft wording | 4/4 | Pass. One added sentence is unreachable. |
| Zero files | 7/7 | Conditional. A word-boundary zero also matches a decimal such as a small price, so an all-zero price table can be waived. |
| Swallowed exceptions | 5/5 | Conditional. The logger writes stderr with no guard. |
| Trace knobs | 3/3 | Conditional. The plan trace still stores a hard-coded overrun of 0. |
| Dead switches removed | 8/8 | Pass. Plan-fallback flag remains and is never set. |
| Loader workbook | 4/4 | Conditional. A later commit fixed a bad-xlsx regression without a test or a captured before/after. |
| Stress corpus | 7/7 | Conditional. Real deliveries were replaced by synthetic rows. The handoff still quoted the old counts. |
| Root cause | 2/3 | Conditional. Machine columns can be rebuilt only if analyst columns are split out. |
| Handoff | 10/12 | Fail. Five sections duplicated, disclosures padded to a word count, old replay numbers retained. |
| Commit discipline | 3/6 | Fail. Two heads lacked a real captured failure on the parent. |

Anti-cheat re-check: no task id, no probe vocabulary, and the counter-shaping commit was gone. The header rule follows the same generic tests the independent review had written (width, a filled grid, no numeric cell, unique names). The empty-first-cell miss is the side effect of that rule.

Go / no-go for a real run of `5e4ca7e`: control no-go, `max` no-go, both because of the call-time constant. `max` also still had the empty-header miss. A one-constant calibration was described as enough to reopen the control lane, and was not made in this loop.
