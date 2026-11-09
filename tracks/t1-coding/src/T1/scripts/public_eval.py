#!/usr/bin/env python3
"""Evaluate the baseline on the pinned public units, one shard at a time, or merge shard results.

Each unit runs the solver (B2) in the candidate image and, when the program ran,
the unit's own checks in a separate offline container. Only bounded metadata is
exported: no task text, prompts, model output, program source or credentials.
This measures a hosted model on public units; it is not an official House score.

With --replay-pubkey, each unit's raw replay (prompt, replies, programs, run logs, the checker's reward files)
is kept for offline replay and leaves the shard only as ciphertext for that key (replay_decrypt.py).
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from fetch_sources import fetch_sources  # noqa: E402
from t1_phase1.baseline import (EVAL_PAUSE_CAP, SIGNATURES, THIRD_PARTY_ERRORS, _BUILTIN_ERRORS,  # noqa: E402
                                card_budget, card_timeout, eval_throttle, summarize_checker)
from t1_phase1.docker_runner import build_agent_command, build_checker_command, run_container  # noqa: E402
from t1_phase1.evaluator import agent_outcome, copy_for_grading, read_status, verify_dataset  # noqa: E402
from t1_phase1.solver_profile import THIN_MAX_CONTINUATIONS_UNIT_CHOICES, public_thin_features, safe_identity
from t1_phase1.house import safe_transport_detail  # noqa: E402
from t1_phase1.staging import stage_unit  # noqa: E402
from replay_decrypt import seal  # noqa: E402

NOT_MEASURED = ('not_run', 'infrastructure_', 'checker_')
FINISH_REASONS = ('stop', 'length', 'content_filter', 'tool_calls', 'function_call', 'unknown')
# Every stop enum baseline.py can emit; test_public_eval collects them from its source and fails on a new one.
STOP_REASONS = ('complete', 'no_progress', 'time_gate', 'chunk_cap', 'request_cap', 'prompt_cap', 'empty_code',
                'invalid_code', 'finish_other', 'request_failed', 'no_entry', 'staged_partial', 'stage_cap',
                'think_gate')
REPAIR_STOPS = ('noop', 'max_repairs', 'time_gate', 'request_cap', 'prompt_cap', 'request_failed', 'setup_error',
                'repair_error', 'repeated_failure', 'think_gate', 'runtime_snapshot_limit', 'runtime_regression')
COUNTERS = ('output_canary_files', 'output_unscanned_files', 'generation_replies', 'continuations', 'rewrites',
            'fill_requests', 'repairs', 'patch_requests', 'path_rewrites', 'patch_loose_stmts', 'content_chars',
            'repair_http_status', 'retries', 'profile_files', 'profile_errors', 'profile_unlisted',
            'profile_symlinks', 'profile_capped', 'salvages', 'patch_retries', 'patch_hunks', 'crash_reports',
            'input_path_rewrites', 'plan_chars', 'review_items', 'review_contract_flags',
            'timeout_repairs', 'review_repairs', 'normalized_files', 'normalized_csv_decimals',
            'normalized_csv_booleans', 'normalized_json_rounded', 'normalized_json_numbers', 'normalized_json_booleans',
            'normalized_json_integers', 'literal_flags', 'degenerate_flags', 'review_rounds', 'selfcheck_total',
            'selfcheck_failed', 'selfcheck_unquoted', 'selfcheck_rounds', 'selfcheck_accepted', 'selfcheck_stray_files',
            'review_affirmations', 'review_unchanged', 'review_retry_patched', 'selfcheck_literal',
            'patch_hunks_anchored', 'repair_same_crash', 'repair_reverts_rejected',
            'repair_lint_flags', 'review_failed_fix', 'profile_previews', 'stage_replies', 'stage_splits',
            'stage_stubs_left', 'fit_warnings', 'stage_salvaged', 'stage_defs_kept', 'repair_lint_new',
            'repair_lint_fixed', 'repair_lint_bound', 'repair_scope_new', 'repair_lint_unresolved',
            'repair_scope_unresolved', 'repair_scope_unknown', 'library_ref_rewrites', 'library_matches',
            'fit_fix_flags', 'selfcheck_syntax_stubs', 'stage_outputs_missing', 'precheck_params_added',
            'precheck_calls_bound', 'contract_int_cells', 'contract_slot_values', 'contract_key_reorders',
            'contract_key_renames', 'contract_header_fixes', 'contract_reverted_files', 'accept_conditions',
            'accept_dropped', 'accept_failed', 'accept_rounds', 'accept_fixed', 'accept_left', 'adopt_required',
            'adopt_missing', 'adopt_none_dropped', 'adopt_none_different', 'raise_guarded',
            'repair_library_use', 'repair_library_again', 'r1think_reasoning_chars', 'plan_code_blocks_cut',
            'patchthink_used', 'patchthink_skipped', 'patchthink_reasoning_chars', 'route_pins', 'route_pins_free',
            'route_pins_quoted', 'library_cards', 'library_card_chars', 'contract2_indented_headers',
            'contract2_cross_file_drops', 's10_chars', 's10_files', 's10_state_hits', 's10_d_lines', 's10_d_quoted',
            's10_d_free', 's10_d_unsure', 's10_d_changed', 's10_d_added', 's10_open', 's10_state_lines',
            's10_notes_chars', 's10_d_tagged', 's10_decide_budget', 's10_check_budget', 's10_state_budget',
            's10_code_budget', 's10_budget_clamped', 's10_reasoning_tokens', 's10_d_tagged_first',
            's10_distill_spill', 's11_d_default', 's11_libnotes', 's11_libnotes_chars', 's11_libdoc', 's11_own_raise',
            'profile_date_ranges', 's11_b_budget', 's11_b_lines', 's11_lines_added', 's11_conflicts',
            's11_binding_alternatives', 's11_binding_skipped')
COUNTERS += ('s11_lint_output', 's11_lint_code', 's11_backend_fixed', 's11_lint_repair', 's11_oom_hints')  # T1_S11_LINT
COUNTERS += ('s12_post_light', 's12_post_light_requests', 's12_post_light_kept')
COUNTERS += ('s12_numeric_context_chars', 's12_post_noop_flags', 's12_post_noop_retry_requests')
COUNTERS += ('minimal_plan_requests', 'minimal_completion_requests', 'minimal_recovery_requests',
             'minimal_complete_functions', 'minimal_pending_count', 'minimal_tail_chars', 'minimal_no_progress')
MINIMAL_STRUCTURE_LIMITS = {'minimal_structure_requests': 9, 'minimal_scope_rejections': 9,
                            'minimal_structure_pending': 1, 'minimal_selected_components': 3}
COUNTERS += tuple(MINIMAL_STRUCTURE_LIMITS)
THIN_COUNTER_LIMITS = {'thin_contract_requests': 25, 'thin_code_requests': 25, 'thin_continue_requests': 25,
                       'thin_continued_complete': 25, 'thin_continued_unparsed': 25, 'thin_headerless_accepted': 25,
                       'thin_late_headers': 25, 'thin_empty_contracts': 25,
                       'thin_tolerant_extractions': 25, 'thin_format_rejections': 25,
                       'thin_code_versions': 25, 'thin_executions': 25, 'thin_pending_count': 1_000_000,
                       'thin_successful_advances': 25, 'thin_no_progress': 25,
                       'thin_missing_files': 1_000_000, 'thin_deliverables_present': 1_000_000,
                       'thin_expected_deliverables': 1_000_000,
                       'thin_date_guard_calls': 1_000_000, 'thin_date_guard_unresolved': 1_000_000_000,
                       'thin_input_bound_files': 8192, 'thin_input_path_rewrites': 1_000_000,
                       'thin_context_chars': 1_000_000, 'thin_time_gate_skips_code': 25,
                       'thin_time_gate_skips_continue': 25, 'thin_time_gate_skips_retry': 25,
                       'thin_continue_skipped': 25, 'thin_bundle_as_replace': 25,
                       'thin_max_continuations': 3, 'thin_stitch_short_echo_dropped': 25,
                       'thin_preexec_blocks': 25, 'thin_continue_unit_capped': 25}
# R2a (T1_THIN_FEATURES): S2-06 final_delivery and S2-08 loop, one per answered code request at most.
THIN_COUNTER_LIMITS.update(thin_final_delivery_replies=25, thin_final_delivery_replace_only=25,
                           thin_loop_revisits=25, thin_loop_executed_resends=25, thin_loop_ast_skips=25,
                           thin_loop_duplicate_bodies=25)
THIN_COUNTER_LIMITS['thin_sanity_at'] = 1_000  # S3-02: requests used when the sanity check held a passing run
THIN_COUNTER_LIMITS['thin_sanity_runs'] = 1_000  # S3-02: runs made when it held it
COUNTERS += tuple(THIN_COUNTER_LIMITS)
COUNTERS += ('s12_rounds', 's12_kept', 's12_reverted', 's12_inert', 's12_inert_runs', 's12_lint_fixed',  # T1_S12_POST
             's12_lint_repair', 's12_flags_degenerate', 's12_flags_literal', 's12_flags_selfval', 's12_flags_lint',
             's12_flags_inert', 's12_flags_audit')
FLAGS = ('continuation_budget_details_dropped', 'transport_details_dropped', 'repair_noop', 'guard_appended', 'first_run_exit_ok', 'library_shown', 'selfcheck_written',
         'eval_429_records_dropped', 'staged_generation', 'stage_skeleton_skipped', 'selfcheck_stubbed',
         'staged_after_continuation', 'route_note', 'route_trimmed', 's10', 's10_template', 's11_conv', 's11_raise',
         's12_post', 'minimal_ready', 'thin_internal_checks_passed', 'thin_direct_step', 'thin_best_restored',
         'thin_no_contract', 'thin_date_guard_blocking', 'thin_preexec_fallback_run')
# S2-04 repair_time, S2-05 false_green and S2-10 (canary, always on): thin's run-region counters and flags.
THIN_EXEC_LIMITS = {'thin_repair_room_waived': 25, 'thin_repair_room_fit': 25, 'thin_canary_side_runs': 25,
                    'thin_false_green_blocks': 25, 'thin_false_green_findings': 1_000_000,
                    'thin_false_green_stop_findings': 1_000_000, 'thin_canary_runs': 25,
                    'thin_canary_files_removed': 1_000_000,
                    # R2 false-green rules (thin.FG_KINDS, held version), rule 4's own counters (fg_bool_false) and
                    # rule 3's host rewrite (runs and files it rewrote, summed over the unit's runs)
                    'thin_fg_zero_or_empty': 1_000_000, 'thin_fg_task_stated': 1_000_000,
                    'thin_fg_stated_sign': 1_000_000, 'thin_fg_log_warning': 1_000_000,
                    'thin_fg_bool_false_findings': 1_000_000, 'thin_fg_bool_false_stop_findings': 1_000_000,
                    'thin_fg_normalized_runs': 25, 'thin_fg_normalized_files': 1_000_000,
                    # the rewrites put back (past OUTPUT_CAP); the requests and runs spent when S2-05 held
                    'thin_fg_normalize_reverted': 25, 'thin_false_green_at': 1_000, 'thin_false_green_runs': 25,
                    # run-through fixes: A1 executed-version notes (switch executed_note), A2 resample (its switch:
                    # requests sent sampled, requests used before the first, triggers the complete best run kept
                    # greedy), B asserts-off reruns (switch assert_rerun since R3, always on before; trigger: started;
                    # exit 0, became the best, not started for time), C imports the host removed
                    'thin_executed_resends': 25, 'thin_resample_requests': 25, 'thin_resample_at': 1_000,
                    'thin_resample_guarded': 25, 'thin_assert_reruns': 25, 'thin_assert_rerun_exit0': 25,
                    'thin_assert_rerun_best': 25, 'thin_assert_rerun_skipped': 25,
                    'thin_shadow_imports_removed': 1_000_000,
                    # R2 trigger counters (r2-final 10-22): repair_time's runs past the old rule's cap, a false-green
                    # hold's rule-4 and other findings, complete_first on the first code prompt (0 or 1), and the
                    # fg_normalize rewrites made inside asserts-off reruns (kept out of thin_fg_normalized_*)
                    'thin_repair_room_overran': 25, 'thin_fg_hold_rule4': 1_000_000,
                    'thin_fg_hold_other': 1_000_000, 'thin_complete_first': 1,
                    'thin_assert_rerun_fg_normalized': 25,
                    # second_draft (10-28): drafts started from the empty program after the loop's passing run
                    'thin_second_drafts': 2,
                    # stub_check and output_review (10-28): runs held as unfinished; the review hold (0 or 1), the
                    # requests spent when it held, and the generic issues it listed
                    'thin_stub_runs': 25, 'thin_review_held': 1, 'thin_review_at': 1_000, 'thin_review_issues': 25,
                    # progress_time (10-28): seconds to the soft and hard deadlines at solve start; runs that improved
                    # the best run past the soft deadline
                    'thin_soft_budget_left': 10_000, 'thin_hard_budget_left': 10_000, 'thin_extended_runs': 25,
                    # crash_redraft (10-28): repeated failures seen (the same one on consecutive runs, or a resend of a
                    # crashed version); the fresh-program request (0 or 1) and the requests spent when it was sent
                    'thin_crash_repeats': 25, 'thin_crash_redrafts': 1, 'thin_crash_redraft_at': 1_000,
                    # literal_contract (10-28): runs held for a column or key the TASK spells out, lacked or misspelt
                    'thin_literal_contract_runs': 25,
                    # outputs_fix (10-28): clean runs held for NaN/inf or |value| > 1e15 in a required output
                    'thin_outputs_blocked_runs': 25,
                    # redraft_fix (10-28): the sampled repairs sent for a repeated failure in place of a fresh program
                    # and the requests spent when the first was sent; the repeats that were a stuck output
                    'thin_crash_repairs': 25, 'thin_crash_repair_at': 1_000, 'thin_crash_stuck_outputs': 25,
                    # loop_fix (10-28): continuation joints re-indented to parse, ready versions held for having no
                    # entry point, cut copies of an executed version ended early, requests sampled after a copy stall,
                    # review replies that resent the reviewed program whole (kept at once), and review keeps (KEEP,
                    # no code, or the reviewed program again) after a fix that was held back before its run
                    'thin_stitch_reindents': 25, 'thin_no_entry_holds': 25, 'thin_copy_stops': 25,
                    'thin_copy_resamples': 25, 'thin_review_copies': 1, 'thin_review_late_keeps': 1,
                    # clock_fix (10-28): continuation requests and runs not started past progress_time's window
                    'thin_clock_continue_stops': 25, 'thin_clock_run_stops': 25,
                    # input_fidelity (10-28): the INPUT PROFILE's data-trap lines, the workbook rows it shows past
                    # column 12 (40 files x 8 sheets x 8 rows at most), and code replies whose kept prefix stopped at
                    # a syntax error the feedback quotes
                    'thin_input_traps': 12, 'thin_wide_rows': 2_560, 'thin_syntax_notes': 25,
                    # lib_review (10-28): the review hold's LIBRARY CROSS-CHECK (0 or 1), and the failed runs whose
                    # repair request got a LIBRARY HINT
                    'thin_lib_review_cross_checks': 1, 'thin_lib_repair_hints': 25}
THIN_COUNTER_LIMITS.update(THIN_EXEC_LIMITS)
COUNTERS += tuple(THIN_EXEC_LIMITS)
FLAGS += ('thin_false_green_unfinished', 'thin_false_green_restored', 'thin_false_green_kept',
          'thin_false_green_sanity_kept', 'thin_assert_rerun_delivered', 'thin_second_draft_replaced',
          'thin_review_kept', 'thin_crash_redraft_accepted', 'thin_crash_redraft_restored')
# T1_S10 (stack 10): its decimal fields, each with its upper bound, and its prompt hash (12 hex digits).
S10_FLOATS = (('s10_overrun', 100.0), ('s10_notes_sec', 100000.0), ('s10_rate_after_decide', 1000.0))
S10_SHA = re.compile(r'[0-9a-f]{12}')
S10_SPILL = ('none', 'used', 'fewer', 'no_lines', 'failed', 'skipped_time', 'skipped_requests')  # spill-over DISTILL
# T1_EVAL_THROTTLE (local evaluation only): the throttled waits credited back to the unit (seconds) and the refused
# 429s the solver did not count, each with its types and upper bound.
EVAL_THROTTLE_FIELDS = (('eval_pause_credited_sec', (int, float), EVAL_PAUSE_CAP),
                        ('eval_uncounted_429', (int,), 10 ** 4))
EVAL_REFUSED_RECORDS = 100  # refused 429 records exported beyond the 25 admitted requests
ENUMS = {
    'thin_date_guard_status': ('complete', 'unverified'),
    'thin_thinking': ('on', 'off', 'low_effort', 'per_request'),
    'thin_contract_thinking': ('on', 'low_effort', 'off', 'budget'),
    'thin_code_thinking': ('on', 'low_effort', 'off', 'budget'),
    # thin.THIN_STOPS (the image has no public_eval); test_thin_s0_export keeps the two equal.
    'thin_stop': ('internal_checks_passed', 'no_progress', 'request_limit', 'execution_setup_error',
                  'generation_budget', 'http_error', 'transport_failure', 'provider_error', 'response_limit',
                  'response_invalid_json', 'response_invalid_shape', 'invalid_configuration',
                  'invalid_prompt', 'prompt_limit', 'choice_unsupported', 'choice_dependencies_unsupported',
                  'thin_output_special', 'output_limit', 'thin_output_changed', 'soft_budget'),
    # thin.FALLBACK_OUTCOMES: how the last-chance run of a version held back before its run ended (S0-11).
    'thin_preexec_fallback_outcome': ('ran', 'passed', 'setup_error', 'output_rejected'),
    # second_draft (thin.SECOND_DRAFT_OUTCOMES, SECOND_DRAFT_DIFFERS, SECOND_DRAFT_ERRORS): how the vote ended, the
    # first way a draft differed from the delivered run, and an error that ended the drafts.
    'thin_second_draft_outcome': ('no_room', 'unfinished', 'disagreed', 'agreed', 'replaced'),
    'thin_second_draft_differs': ('missing', 'differs', 'no_outputs'),
    'thin_second_draft_error': ('house_error', 'other'),
    # sanity.LABELS and sanity.OUTCOMES (S3-02); test_sanity keeps them equal.
    'thin_sanity_check': ('header_only', 'constant_column', 'header_only+constant_column'),
    'thin_sanity_outcome': ('passed_clean', 'passed_flagged', 'kept'),
    'thin_interface': ('free', 'choice'),
    'thin_max_continuations_unit': THIN_MAX_CONTINUATIONS_UNIT_CHOICES,  # T1_THIN_MAX_CONTINUATIONS_UNIT (S2-01)
    # S2-06 final_delivery: thin.FINAL_TRIGGERS candidates, and thin.FINAL_WHY for the first FINAL DELIVERY ask.
    'thin_final_delivery_trigger': ('A', 'B'),
    'thin_final_delivery_first': ('requests', 'seconds'),
    'minimal_state': ('EMPTY', 'INCOMPLETE', 'DEFINITIONS_ONLY', 'READY', 'STOPPED'),
    'minimal_stop_reason': ('request_cap', 'time_gate', 'prompt_cap', 'no_progress', 'reply_limit',
                            'incomplete_code', 'provider_error'),
    'solver_profile_gate': ('accepted','unsupported_solver_profile','full_library_required',
                            'minimal_controller_required', 'thin_controller_required'),
    'budget_source': ('explicit', 'image', 'default', 'caller'),
         'controller_protection': ('enabled', 'unavailable', 'not_attempted'),
         'profile_status': ('ok', 'partial', 'error'),
         'prompt_trim': ('none', 'library_omitted', 'profile_compact', 'profile_omitted', 'hints_trimmed',
                         'instruction_truncated'),
         'plan_status': ('ok', 'reasoning_only', 'empty', 'failed', 'skipped_time'),
         'review_status': ('clean', 'fixed', 'fix_rejected', 'no_patch', 'failed', 'skipped_time', 'too_large'),
         'fresh_status': ('kept', 'rejected', 'duplicate', 'skipped_time', 'failed'),
         'accept_status': ('clean', 'fixed', 'fix_rejected', 'no_patch', 'no_checker', 'checker_failed',
                           'no_conditions', 'skipped_time', 'failed', 'too_large'),
         'r1think': ('used', 'skipped_no_plan', 'skipped_plan_long', 'skipped_time'),
         'plan_map': ('recovered', 'no_map', 'failed', 'skipped'),
         'route_family': ('closed_form', 'numerical_engine', 'process', 'risk', 'rates', 'ledger', 'econometrics',
                          'execution', 'wrangling', 'other', 'none'),
         's10_tier': ('F', 'D', 'L'),
         's10_decide': ('ok', 'cut_kept', 'no_lines', 'distilled', 'retried', 'failed', 'fallback_plan'),
         's10_check': ('ok', 'cut_kept', 'no_lines', 'failed', 'skipped_time', 'skipped_requests', 'skipped_breaker',
                       'not_triggered', 'off'),
         's10_check_trigger': ('cut', 'open', 'unsure', 'free', 'budget', 'always', 'none'),
         's10_state': ('ok', 'cut_kept', 'no_lines', 'failed', 'skipped_time', 'skipped_requests', 'not_stateful',
                       'off'),
         's10_stateful': ('hits', 'reply', 'both', 'none'),
         's10_breaker': ('none', 'overrun', 'reasoning_only', 'overrun_tokens'),
         's10_decide_spill': S10_SPILL, 's10_check_spill': S10_SPILL, 's10_state_spill': S10_SPILL,
         's11_state_rescue': ('none', 'floor', 'think_off'),
         's11_decide_b': ('ok', 'cut_kept', 'no_lines', 'spilled', 'failed', 'skipped_time', 'skipped_requests',
                          'skipped_breaker', 'no_notes'),
         's12_stop': ('clean', 'cleared', 'rounds', 'no_patch', 'reverted', 'skipped_time', 'skipped_requests',
                      'failed', 'too_large'),
         's12_post_noop_kind': ('same_definition',),
         's12_audit': ('items', 'none', 'unparsed', 'no_notes', 'skipped_time', 'skipped_requests', 'skipped_breaker',
                       'failed'),
         's12_inert_status': ('flagged', 'none', 'unverified', 'no_params', 'no_paths', 'skipped_time', 'too_large',
                              'failed')}
# T1_S12_MECH: its counters, flags and enums (baseline's S12_STATE_FROM, and the note request kinds or none).
COUNTERS += ('s12_note_timeouts', 's12_notes_skipped', 's12_no_fence', 's12_cont_sized', 's12_cont_line',
             's12_patch_unusable', 's12_patch_light', 's12_patch_light_used', 's12_patch_retry', 's12_crash_site')
FLAGS += ('s12_mech', 's12_code_think_off', 's12_conv_more')
FLAGS += ('runtime_enabled', 'runtime_shown')
# S3-03b: thin's repair notes (switch repair_feedback; at most one of each per run) and its always-on robustness
# (CMP-4 context steps 1-3, OS-4 early restores per run, OS-6 restore before an exception ends solve).
THIN_REPAIR_LIMITS = {'thin_repair_notes_own_check': 25, 'thin_repair_notes_same_exception': 25,
                      'thin_repair_notes_mounted': 25, 'thin_context_degraded': 3, 'thin_best_restored_early': 25,
                      'thin_early_restore_failures': 25}
THIN_COUNTER_LIMITS.update(THIN_REPAIR_LIMITS)
COUNTERS += tuple(THIN_REPAIR_LIMITS)
FLAGS += ('thin_best_restored_on_error',)
# Optional S13 observations: fixed enums and bounded aggregates only; never raw samples.
S13_COUNTERS = ('s13_refine_literals', 's13_coerce_runs', 's13_coerce_advisory')
S10_FLOATS += (('s13_refine_sec', 100000.0),)
ENUMS.update(s13_refine=('none', 'rng', 'skipped_time', 'too_large', 'x2_failed', 'x4_failed', 'shape',
                       'nonfinite', 'unmoved', 'not_contracting', 'sign', 'adopted', 'failed',
                       'runtime_regression', 'unproven_grid', 'task_resolution', 'time_axis_unverified'),
             s13_coerce_audit=('no_call', 'ran', 'skipped_time', 'too_large', 'failed'),
             s13_time_axis=('skipped_unverified',))
ENUMS['runtime_state'] = ('valid', 'missing', 'invalid', 'empty')
# S3-02 (sanity.REPAIR_STOPS): how the one sanity repair ended when no run passed after it: thin_stop's own code before
# the questioned run was restored as the stop, else repair_run_not_passed (its reply ran), request_limit (the cap) or
# request_spent (the request was used and its reply never ran).
ENUMS['thin_sanity_repair_stop'] = tuple(s for s in ENUMS['thin_stop'] if s != 'internal_checks_passed') + (
    'repair_run_not_passed', 'request_spent')
ENUMS.update(s12_state=('reply', 'strong', 'hits', 'no_honoured', 'none'),
             s12_note_failed=('none', 's10_decide', 's10_distill', 's10_decide_retry', 's10_check', 's10_state',
                              's11_decide_b'))
REQUEST_KINDS = ('plan', 'review', 'review_patch', 'initial', 'continue_stmt', 'continue_line', 'rewrite', 'fill', 'salvage', 'patch',
                 'patch_continue', 'patch_retry', 'review_retry', 'skeleton', 'stage', 'split', 'lint_retry',
                 'accept', 'accept_fix', 'plan_map', 's10_decide', 's10_distill', 's10_decide_retry', 's10_check',
                 's10_state', 's11_decide_b', 's12_audit', 's12_fix',
                 'minimal_plan', 'minimal_complete', 'minimal_recover', 'thin_contract', 'thin_code', 'thin_continue')
RUN_STATUSES = ('executed_ungraded', 'execution_failed', 'execution_timeout')
EXCEPTION_TYPES = _BUILTIN_ERRORS | {name.rsplit('.', 1)[-1].lstrip('_') for name in THIRD_PARTY_ERRORS}
GRADED_AGENT = ('executed_ungraded', 'execution_failed', 'execution_timeout')
FAILED_CHECK_LIMIT = 40
CHECK_NAME = re.compile(r'[A-Za-z_][A-Za-z0-9_]{0,99}')
REPLAY_FILE_LIMIT = 4 * 1024 * 1024
REPLAY_UNIT_LIMIT = 32 * 1024 * 1024


def load_classes(path=None):
    path = path or ROOT / 'selection/classes.json'
    return json.loads(path.read_text())['classes'] if path.exists() else {}


def select_units(units, spec, classes):
    """The roster, or the units and task-shape classes (selection/classes.json) named in spec, separated by
    commas or spaces. An unknown name stops the run before any unit is evaluated."""
    if not spec or not spec.strip():
        return list(units)
    chosen = set()
    for token in re.split(r'[\s,]+', spec.strip()):
        if token in classes:
            chosen.update(classes[token]['units'])
        elif token in units:
            chosen.add(token)
        else:
            raise SystemExit(f'Unknown unit or class in the selection: {token!r}')
    unknown = chosen - set(units)
    if unknown:
        raise SystemExit(f'Selected units missing from the roster: {sorted(unknown)}')
    return sorted(chosen)


MAX_REPEATS = 5


def attempts(units, repeats):
    """(unit, k) for k < repeats: each attempt is evaluated on its own, in its own directories."""
    if not (type(repeats) is int and 1 <= repeats <= MAX_REPEATS):
        raise SystemExit(f'repeats must be 1-{MAX_REPEATS}')
    return [(unit, k) for unit in sorted(units) for k in range(repeats)]


def attempt_name(unit, k, repeats):
    return unit if repeats == 1 else f'{unit}__r{k}'


def shard_units(units, count, index):
    if not (type(count) is int and count >= 1 and type(index) is int and 0 <= index < count):
        raise ValueError('Invalid shard selection')
    return sorted(units)[index::count]


def stage_units(datasets, participant_root, ids):
    """Stage each unit on its own; a unit the local stager rejects is recorded, not fatal."""
    staged, errors = {}, {}
    for unit in ids:
        try:
            staged[unit] = stage_unit(Path(datasets) / unit, Path(participant_root) / unit)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors[unit] = type(exc).__name__
    return staged, errors


SCOPES = {None: 'NVIDIA-hosted model on public units; not an official House score',
          'openrouter': 'OpenRouter FP8 (the House weights) on public units; not an official House score',
          'openrouter-bf16': 'OpenRouter BF16 (not the House FP8 weights; confirm on openrouter) on public units; '
                             'not an official House score',
          'openrouter-free': 'OpenRouter local-free (NVIDIA; quantization unknown; official House weights unverified) '
                             'on public units; not an official House score',
          'mixed': 'Shards answered by different backends; not comparable to a single-backend run'}


def eval_backend():
    """The T1_EVAL_BACKEND this shard ran with (None for the endpoint's own route); anything else is refused."""
    backend = os.environ.get('T1_EVAL_BACKEND') or None
    if backend not in SCOPES or backend == 'mixed':
        raise SystemExit('Unknown T1_EVAL_BACKEND; no unit was run')
    return backend


def summarize(rows, backend=None):
    counts = Counter(row['status'] for row in rows)
    not_measured = sum(row['status'].startswith(NOT_MEASURED) for row in rows)
    measured = len(rows) - not_measured
    passed = counts.get('passed', 0)
    # The hosted model's 429/503 answers, so a comparison can set aside the units throttling touched.
    statuses = [[r.get('http_status') for r in row.get('requests') or [] if isinstance(r, dict)] for row in rows]
    throttled = [sum(s in (429, 503) for s in unit) for unit in statuses]
    return {'units': len(rows), 'passed': passed, 'not_measured': not_measured,
            'pass_rate_all_units': passed / len(rows) if rows and measured else None,
            'pass_rate_measured_units': passed / measured if measured else None,
            'requests': sum(map(len, statuses)), 'throttled_requests': sum(throttled),
            'throttled_units': sum(n > 0 for n in throttled),
            # Units whose solver reported T1_EVAL_THROTTLE applied. The shard's eval_throttle is only the host's
            # switch: an image built before the switch existed ignores it and reports none here.
            'eval_throttle_units': sum('eval_pause_credited_sec' in row for row in rows),
            'status_counts': dict(sorted(counts.items())),
            'scope': SCOPES.get(backend, SCOPES['mixed'])}


def _number(value, high=10 ** 7):
    return type(value) in (int, float) and 0 <= value <= high


def _requests(records, refused=0, transport=None):
    """Per-request numbers and enums (I1), each value re-validated; at most the 25 admitted requests, plus the
    refused 429s that T1_EVAL_THROTTLE left uncounted (at most EVAL_REFUSED_RECORDS)."""
    rows = []
    limit = 25 + min(refused, EVAL_REFUSED_RECORDS)
    aligned = isinstance(records, list) and isinstance(transport, list) and len(records) == len(transport)
    for index, record in enumerate(records[:limit] if isinstance(records, list) else []):
        if not isinstance(record, dict):
            continue
        row = {k: record[k] for k in ('attempt', 'completion_tokens', 'prompt_tokens', 'reasoning_chars',
                                      'reasoning_budget')
               if type(record.get(k)) is int and 0 <= record[k] <= 10 ** 7}
        row.update({k: record[k] for k in ('duration_sec', 'timeout_sec') if _number(record.get(k), 100000)})
        if type(record.get('max_tokens')) is int and 1 <= record['max_tokens'] <= 4000:
            row['max_tokens'] = record['max_tokens']
        for key in ('continuation_need_sec', 'continuation_remaining_sec', 'continuation_reserve_sec'):
            if _number(record.get(key), 100000):
                row[key] = record[key]
        if _number(record.get('continuation_rate_tok_s'), 400) and record['continuation_rate_tok_s'] > 0:
            row['continuation_rate_tok_s'] = record['continuation_rate_tok_s']
        if record.get('kind') in REQUEST_KINDS:
            row['kind'] = record['kind']
        if type(record.get('temperature')) in (int, float) and 0 < record['temperature'] <= 2:
            row['temperature'] = record['temperature']  # a sampled request (house records only non-greedy ones)
        if record.get('finish_reason') in FINISH_REASONS:
            row['finish_reason'] = record['finish_reason']
        if type(record.get('http_status')) is int and 100 <= record['http_status'] <= 599:
            row['http_status'] = record['http_status']
        if isinstance(record.get('provider'), str) and re.fullmatch(r'[A-Za-z0-9._-]{1,40}', record['provider']):
            row['provider'] = record['provider']
        if aligned:
            detail = safe_transport_detail(transport[index])
            if detail:
                row['transport_detail'] = detail
        rows.append(row)
    return rows


def _runs(runs):
    """Per-run exit, exception class and failure signature (I2): enums and numbers only."""
    rows = []
    for run in runs[:8] if isinstance(runs, list) else []:
        if not isinstance(run, dict):
            continue
        row = {k: run[k] for k in ('exit_code', 'generated_line')
               if type(run.get(k)) is int and -1000 <= run[k] <= 1_000_000}
        if type(run.get('crash_report')) is bool:
            row['crash_report'] = run['crash_report']
        if _number(run.get('duration_sec'), 100000):
            row['duration_sec'] = run['duration_sec']
        if _number(run.get('old_cap_sec'), 100000):
            row['old_cap_sec'] = run['old_cap_sec']  # S2-04: the cap the pre-repair_time rule would have set
        for key, allowed in (('status', RUN_STATUSES), ('signature', SIGNATURES), ('exception_type', EXCEPTION_TYPES)):
            if isinstance(run.get(key), str) and run[key] in allowed:
                row[key] = run[key]
        for key in ('memory_limit_bytes', 'memory_current_before_bytes', 'memory_current_after_bytes',
                    'memory_oom_kill_delta'):
            if type(run.get(key)) is int and 0 <= run[key] < 2 ** 60:
                row[key] = run[key]
        if run.get('memory_event') in ('cgroup_oom_observed', 'signal_killed_unknown', 'controller_timeout',
                                      'no_oom_observed', 'unknown'):
            row['memory_event'] = run['memory_event']
        rows.append(row)
    return rows


def bounded_row(unit, status, process):
    """Copy only fixed-schema fields from the controller status."""
    diagnostic = status.get('house_diagnostics') if isinstance(status.get('house_diagnostics'), dict) else {}
    usage = status.get('usage') if isinstance(status.get('usage'), dict) else {}
    code = status.get('failure_code')
    row = {'id': unit, 'model_calls': status.get('model_calls') if type(status.get('model_calls')) is int else 0,
           'usage': {k: v for k, v in usage.items()
                     if k in ('prompt_tokens', 'completion_tokens', 'total_tokens') and type(v) is int},
           'failure_code': code if isinstance(code, str) and re.fullmatch(r'[a-z_]{1,40}', code) else None,
           'finish_reason': diagnostic.get('finish_reason') if diagnostic.get('finish_reason') in FINISH_REASONS else None,
           'agent_returncode': process.get('returncode') if type(process.get('returncode')) is int else None,
           'agent_timed_out': bool(process.get('timed_out'))}
    identity = safe_identity(status.get('solver_identity'))
    if identity:
        row['solver_identity'] = identity
    cap = status.get('request_cap')
    if type(cap) is int and 1 <= cap <= 25:
        row['request_cap'] = cap
    for key in ('thin_reasoning_budget', 'thin_contract_reasoning_budget', 'thin_code_reasoning_budget'):
        thin_budget = status.get(key)
        if type(thin_budget) is int and 1 <= thin_budget <= 2048:
            row[key] = thin_budget
    if public_thin_features(status.get('thin_features')):  # T1_THIN_FEATURES as the loop parsed it (S2-01)
        row['thin_features'] = list(status['thin_features'])
    http_status = diagnostic.get('http_status')
    house_failure = (row['failure_code'] in ('http_error', 'transport_failure')
                     or str(status.get('status')).startswith(('infrastructure_', 'not_run')))
    if house_failure and type(http_status) is int and 100 <= http_status <= 599:
        row['http_status'] = http_status  # 401/429/5xx tell a bad token from an outage
    for key in COUNTERS:
        if type(status.get(key)) is int:
            if key in MINIMAL_STRUCTURE_LIMITS and not 0 <= status[key] <= MINIMAL_STRUCTURE_LIMITS[key]:
                continue
            if key in THIN_COUNTER_LIMITS and not 0 <= status[key] <= THIN_COUNTER_LIMITS[key]:
                continue
            if key.startswith('minimal_') and not 0 <= status[key] <= 1_000_000:
                continue
            if (key in ('repair_scope_new', 'repair_lint_unresolved', 'repair_scope_unresolved', 'repair_scope_unknown')
                    and not 0 <= status[key] <= 1_000_000):
                continue
            row[key] = status[key]
    for key in ('runtime_passed', 'runtime_failed', 'runtime_skipped', 'runtime_unverified'):
        if type(status.get(key)) is int and 0 <= status[key] <= 128:
            row[key] = status[key]
    for key in S13_COUNTERS:
        if type(status.get(key)) is int and 0 <= status[key] <= 1000000:
            row[key] = status[key]
    for key in FLAGS:
        if type(status.get(key)) is bool:
            row[key] = status[key]
    for key, allowed in (('stop_reason', STOP_REASONS), ('repair_stop', REPAIR_STOPS)):
        if status.get(key) in allowed:
            row[key] = status[key]
    if type(status.get('rate_tok_s')) in (int, float) and 0 < status['rate_tok_s'] <= 1000:
        row['rate_tok_s'] = status['rate_tok_s']
    if _number(diagnostic.get('backoff_sec'), 100000):
        row['backoff_sec'] = diagnostic['backoff_sec']
    if type(diagnostic.get('reasoning_chars')) is int:  # 0 means enable_thinking=false held
        row['reasoning_chars'] = diagnostic['reasoning_chars']
    execution = status.get('execution_diagnostics') if isinstance(status.get('execution_diagnostics'), dict) else {}
    if isinstance(execution.get('exception_type'), str) and re.fullmatch(r'[A-Za-z]{1,64}', execution['exception_type']):
        row['exception_type'] = execution['exception_type']
    for key, allowed in ENUMS.items():
        if isinstance(status.get(key), str) and status[key] in allowed:
            row[key] = status[key]
    for key, allowed in (('runtime_usage_state', ('valid','missing','invalid')),
                         ('requirement_state', ('covered','incomplete','unverified','missing')),
                         ('semantic_state', ('not_supplied','unresolved','partial','explicit'))):
        if isinstance(status.get(key),str) and status[key] in allowed:
            row[key] = status[key]
    for key in ('semantic_quantities','semantic_unresolved','requirement_checks') + tuple(
            'runtime_use_'+k for k in ('checks','semantics','contracts','verification','resources',
                                      'workflow','documents','evidence','cache')):
        high = 128 if key.startswith(('semantic_','requirement_')) else 1000000
        if type(status.get(key)) is int and 0 <= status[key] <= high:
            row[key] = status[key]
    if _number(status.get('budget_sec_applied'), 100000):
        row['budget_sec_applied'] = status['budget_sec_applied']
    for key, high in S10_FLOATS:
        if _number(status.get(key), high):
            row[key] = status[key]
    if isinstance(status.get('s10_prompt_sha'), str) and S10_SHA.fullmatch(status['s10_prompt_sha']):
        row['s10_prompt_sha'] = status['s10_prompt_sha']
    if isinstance(status.get('s12_prompt_sha'), str) and S10_SHA.fullmatch(status['s12_prompt_sha']):
        row['s12_prompt_sha'] = status['s12_prompt_sha']
    for key in ('program_sha256', 'thin_output_version'):
        if isinstance(status.get(key), str) and re.fullmatch(r'[0-9a-f]{64}', status[key]):
            row[key] = status[key]
    for key, types, high in EVAL_THROTTLE_FIELDS:
        if type(status.get(key)) in types and 0 <= status[key] <= high:
            row[key] = status[key]
    if _number(status.get('duration_sec'), 100000):
        row['agent_duration_sec'] = status['duration_sec']
    kinds = status.get('request_kinds')
    if isinstance(kinds, dict):
        row['request_kinds'] = {k: v for k, v in kinds.items() if k in REQUEST_KINDS and type(v) is int and 0 <= v <= 25}
    if isinstance(diagnostic.get('requests'), list):
        row['requests'] = _requests(diagnostic['requests'], row.get('eval_uncounted_429', 0),
                                    diagnostic.get('transport_requests'))
    if isinstance(status.get('runs'), list):
        row['runs'] = _runs(status['runs'])
    return row


def evaluate_unit(unit, image, work, args, attempt=None):
    started = time.monotonic()
    base = work / 'results' / (attempt or unit)
    output, verifier = base / 'agent-output', base / 'verifier'
    for directory in (output, verifier):
        directory.mkdir(parents=True)
        directory.chmod(0o777)
    # As on the platform: the card's [agent].timeout_sec is only the container ceiling, and solve takes the
    # image's UNIT_BUDGET_SEC under it. --explicit-budget passes the harness's budget as T1_TIMEOUT_SEC instead.
    source = work / 'datasets' / unit
    timeout = card_timeout(source) or float(args.unit_timeout)
    budget = card_budget(source, args.unit_timeout)
    replay = work / 'replay' / (attempt or unit) if getattr(args, 'replay_pubkey', None) else None
    if replay is not None:  # the agent (uid 65534) records into agent/; grading/ is the host's alone
        (replay / 'agent').mkdir(parents=True)
        (replay / 'agent').chmod(0o777)
    command = build_agent_command(image, work / 'participant' / unit, output, context=args.context,
                                  platform='linux/amd64', timeout_sec=budget, model_env=True, cpus=args.cpus,
                                  memory=args.memory, replay_dir=replay / 'agent' if replay else None,
                                  budget_env=bool(getattr(args, 'explicit_budget', False)))
    # Under T1_EVAL_THROTTLE the solver may also sleep up to EVAL_PAUSE_CAP s of throttled waits it does not charge
    # to its budget, so the container ceiling grows by the same amount and those waits can never reach it.
    ceiling = timeout + 10 + (EVAL_PAUSE_CAP if eval_throttle() else 0)
    process = run_container(command, timeout_sec=ceiling)
    try:
        status = read_status(output / '_agent_status.json')
    except (OSError, ValueError, TypeError):
        status = {'status': 'missing_status', 'model_calls': 0}
    row = bounded_row(unit, status, process)
    row.update(solve_budget_sec=budget, container_timeout_sec=ceiling)
    row['status'] = row['agent_status'] = agent_outcome(status, process)
    # The official scorer grades whatever the agent left, so a crashed or timed-out program's non-empty
    # output is graded too; its agent status stays beside the verdict. Its g2 gate fails first an output
    # carrying an input canary and a run past the card timeout, which a container that overran its
    # ceiling (card timeout + 10 s, plus the credited throttle waits locally) already is: those are never graded.
    ran = row['status'] == 'executed_ungraded'
    crashed = (row['status'] in GRADED_AGENT and not row['agent_timed_out']
               and not row.get('output_canary_files') and has_deliverables(output))
    if ran or crashed:
        grade_output(row, unit, image, work, args, output, base / 'grading-output', verifier, ran)
    if replay is not None:
        keep_grading(replay / 'grading', base / 'grading-output', verifier)
    row['duration_sec'] = round(time.monotonic() - started, 3)
    return row


def has_deliverables(output):
    """Any entry besides the controller's own _agent_* files and the reserved report names."""
    try:
        return any(not (p.name.startswith('_agent_') or p.name in ('reward.json', 'pytest_report.json', 'reward.txt'))
                   for p in output.iterdir())
    except OSError:
        return False


def grade_output(row, unit, image, work, args, output, grade, verifier, ran):
    """For a program that exited 0 every outcome is recorded; for a crashed one only a decisive verdict
    replaces its agent status (a harness fault keeps it measured as the agent's failure)."""
    try:
        copy_for_grading(output, grade)
    except (OSError, ValueError):
        if ran:
            row['status'] = 'invalid_output_tree'
        return
    try:
        command = build_checker_command(image, work / 'datasets' / unit, grade, verifier, context=args.context,
                                        platform='linux/amd64', cpus=args.cpus, memory=args.memory)
    except (OSError, ValueError):
        if ran:
            row['status'] = 'checker_error'  # the harness cannot present this unit's checks
        return
    started = time.monotonic()
    verdict = summarize_checker(grade, run_container(command, timeout_sec=args.checker_timeout))
    row.update(verdict=verdict['status'], checker_sec=round(time.monotonic() - started, 3))
    if ran or verdict['status'] in ('passed', 'failed'):
        row['status'] = verdict['status']
    if isinstance(verdict.get('test_counts'), dict):
        row['test_counts'] = verdict['test_counts']
    row.update(failed_checks(grade))


def failed_checks(grade):
    """Names of the checks that failed or errored, from the checker's pytest_report.json: the public test
    function (and class) only, never its parameters or failure message. A name outside the safe pattern is
    only counted; at most FAILED_CHECK_LIMIT distinct names are kept."""
    try:
        report = json.loads((grade / 'pytest_report.json').read_text())
        tests = report.get('tests') if isinstance(report, dict) else None
    except (OSError, ValueError):
        return {}
    if not isinstance(tests, list):
        return {}
    counts, unnamed = {}, 0
    for test in tests:
        if not isinstance(test, dict) or test.get('outcome') not in ('failed', 'error'):
            continue
        nodeid = test.get('nodeid')
        parts = nodeid.split('::')[1:] if isinstance(nodeid, str) else []
        if parts:
            parts[-1] = parts[-1].split('[', 1)[0]
        if not parts or not all(CHECK_NAME.fullmatch(part) for part in parts):
            unnamed += 1
            continue
        key = ('.'.join(parts), test['outcome'])
        counts[key] = counts.get(key, 0) + 1
    result = {}
    if counts:
        result['failed_checks'] = [{'name': name, 'outcome': outcome, 'count': count}
                                   for (name, outcome), count in list(counts.items())[:FAILED_CHECK_LIMIT]]
    if len(counts) > FAILED_CHECK_LIMIT:
        result['failed_checks_dropped'] = len(counts) - FAILED_CHECK_LIMIT
    if unnamed:
        result['failed_checks_unnamed'] = unnamed
    return result


def _copy_plain(source, target):
    """Copy one regular file without following a link; False when it is not one or too large."""
    try:
        fd = os.open(source, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError:
        return False
    with os.fdopen(fd, 'rb') as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > REPLAY_FILE_LIMIT:
            return False
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644), 'wb') as sink:
            shutil.copyfileobj(stream, sink)
    return True


REPLAY_GRADING_FILES = ('reward.json', 'reward.txt')


def keep_grading(target, grade, verifier):
    """Local replay only: the checker's reward files (REPLAY_GRADING_FILES, from the graded output and the
    verifier directory) beside the agent's recording. The pytest report and the verifier logs stay out: the
    sealed replay is read offline, and the checks are never read. Best-effort; the verdict never depends on it."""
    try:
        items = [(grade / name, name) for name in REPLAY_GRADING_FILES]
        items += [(verifier / name, 'verifier/' + name) for name in REPLAY_GRADING_FILES]
        for path, name in items:
            try:
                _copy_plain(path, target / name)
            except OSError:
                continue
    except Exception:  # noqa: BLE001 - replay capture must not change the measurement
        pass


def replay_key_ok(pubkey):
    """A PEM public key openssl can read. A private key is refused: openssl would accept it, but it must never
    reach the runner."""
    try:
        text = Path(pubkey).read_bytes()
        if b'-----BEGIN PUBLIC KEY-----' not in text or b'PRIVATE' in text:
            return False
        return subprocess.run(['openssl', 'pkey', '-pubin', '-in', str(pubkey), '-noout'],
                              capture_output=True, timeout=30).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _archive(archive, unit_dir, name):
    """Regular files only (no links or devices a program may have left), each and in total bounded."""
    total = 0
    for root, dirs, names in os.walk(unit_dir):
        dirs.sort()
        for entry in sorted(names):
            path = Path(root) / entry
            info = path.lstat()
            if not stat.S_ISREG(info.st_mode) or info.st_size > REPLAY_FILE_LIMIT:
                continue
            if total + info.st_size > REPLAY_UNIT_LIMIT:
                return
            try:  # tarfile opens the file before writing its header, so a refused open leaves no entry
                archive.add(path, arcname=f'{name}/{path.relative_to(unit_dir).as_posix()}', recursive=False)
            except OSError:
                continue
            total += info.st_size


def seal_replay(root, out, shard, pubkey, work):
    """Tar every unit's replay directory, encrypt it for the developer's public key only, and remove all
    plaintext: the tar and the replay tree. Returns enums and counts for the shard header."""
    root, out, work = Path(root), Path(out), Path(work)
    units = sorted(p.name for p in root.iterdir() if p.is_dir() and not p.is_symlink()) if root.is_dir() else []
    if not units:
        return {'replay': 'empty'}
    plain = work / f'replay-shard-{shard}.tar.gz'
    sealed = (out / f'replay-shard-{shard}.tar.gz.enc', out / f'replay-shard-{shard}.key.enc')
    try:
        with tarfile.open(plain, 'w:gz') as archive:
            for unit in units:
                _archive(archive, root / unit, unit)
        seal(plain, pubkey, *sealed)
        result = {'replay': 'sealed', 'replay_units': len(units)}
    except Exception:  # noqa: BLE001 - a lost replay never fails the shard
        for path in sealed:
            path.unlink(missing_ok=True)
        result = {'replay': 'seal_failed'}
    finally:
        plain.unlink(missing_ok=True)
        shutil.rmtree(root, ignore_errors=True)
    # A directory the program (another uid) made under /replay can keep its plaintext from the host runner.
    if root.exists() or plain.exists():
        result['replay'] = 'cleanup_incomplete'
    return result


def gate_passed(receipt, image):
    """The integration fixture proved, for this image, that generated code cannot read the token."""
    checks = receipt.get('checks') if isinstance(receipt, dict) else None
    return (isinstance(checks, dict) and bool(checks) and all(value is True for value in checks.values())
            and receipt.get('status') == 'passed' and receipt.get('fixture_only') is True
            and receipt.get('image_id') == image and receipt.get('platform') == 'linux/amd64')


def exit_code(rows):
    """Fail the merged run when nothing was measured or a shard's results never arrived. Shards themselves
    succeed once their file is written: with one attempt per shard, a single throttled request would
    otherwise mark the commit red."""
    summary = summarize(rows)
    lost = any(row['status'] == 'not_run_shard_missing' for row in rows)
    return 1 if lost or not rows or summary['not_measured'] == summary['units'] else 0


def run_shard(args):
    if any(not os.environ.get(name) for name in ('MODEL_ENDPOINT', 'MODEL_NAME', 'MODEL_TOKEN')):
        raise SystemExit('Model configuration is incomplete; no unit was run')
    if not args.gate_receipt or not gate_passed(json.loads(args.gate_receipt.read_text()), args.image):
        raise SystemExit('No passing isolation receipt for this image; the model token is not exposed')
    if args.replay_pubkey and not replay_key_ok(args.replay_pubkey):
        raise SystemExit('--replay-pubkey is not a readable public key; no unit was run')
    backend = eval_backend()
    index = json.loads(args.index.read_text())
    units = select_units(sorted({PurePosixPath(entry['path']).parts[1] for entry in index['files']}),
                         getattr(args, 'units', None), load_classes())
    repeats = getattr(args, 'repeats', 1) or 1
    mine_attempts = shard_units(attempts(units, repeats), args.shards, args.shard)
    mine = sorted({unit for unit, _ in mine_attempts})
    work = args.work.resolve()
    work.mkdir(parents=True, exist_ok=False)
    (work / 'evidence').mkdir()
    sub_index = work / 'evidence/source_files.json'
    sub_index.write_text(json.dumps(dict(index, files=[e for e in index['files']
                                                       if PurePosixPath(e['path']).parts[1] in mine])))
    acquired = fetch_sources(sub_index, work / 'datasets', work / 'evidence/dataset_manifest.json',
                             workers=args.workers)
    verify_dataset(work / 'datasets', json.loads((work / 'evidence/dataset_manifest.json').read_text()),
                   index['commit'])
    _, errors = stage_units(work / 'datasets', work / 'participant', mine)
    args.out.mkdir(parents=True, exist_ok=True)
    result = {'schema_version': 1, 'source_commit': index['commit'], 'shard': args.shard, 'shards': args.shards,
              'run_attempt': int(os.environ.get('GITHUB_RUN_ATTEMPT', '1')),
              'image_id': args.image, 'model_name': os.environ['MODEL_NAME'], 'eval_backend': backend,
              'eval_throttle': eval_throttle(),  # the host's switch; summary.eval_throttle_units, what the solver did
              'unit_budget_sec': args.unit_timeout,
              'budget_mode': 'explicit' if getattr(args, 'explicit_budget', False) else 'image',
              'checker_budget_sec': args.checker_timeout, 'resource_profile': f'local-{args.cpus:g}cpu-{args.memory}',
              'platform_equivalent': False, 'acquisition': acquired, 'units': []}
    path = args.out / f'shard-{args.shard}.json'

    def record(row):
        result['units'].append(row)
        result['summary'] = summarize(result['units'], backend)
        path.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps({k: row.get(k) for k in ('id', 'status', 'failure_code', 'duration_sec')}), flush=True)

    evaluate_units(mine_attempts, errors, args.image, work, args, on_row=record, repeats=repeats)
    if args.replay_pubkey:
        result.update(seal_replay(work / 'replay', args.out, args.shard, args.replay_pubkey, work))
        path.write_text(json.dumps(result, indent=2) + '\n')
    return result


def evaluate_units(units, errors, image, work, args, on_row=None, repeats=1):
    """A harness failure on one unit is recorded as unmeasured; the shard carries on. Items are unit ids or
    (unit, k) attempts; with repeats > 1 each row carries its attempt number."""
    rows = []
    for item in units:
        unit, k = item if isinstance(item, tuple) else (item, 0)
        if unit in errors:
            row = {'id': unit, 'status': 'not_run_staging_error', 'error_type': errors[unit], 'model_calls': 0}
        else:
            try:
                row = evaluate_unit(unit, image, work, args,
                                    **({'attempt': attempt_name(unit, k, repeats)} if repeats > 1 else {}))
            except Exception as exc:  # noqa: BLE001 - keep measuring the remaining units
                row = {'id': unit, 'status': 'infrastructure_eval_error',
                       'error_type': type(exc).__name__, 'model_calls': 0}
        if repeats > 1:
            row['repeat'] = k
        rows.append(row)
        if on_row is not None:
            on_row(row)
    return rows


def merge(args):
    index = json.loads(args.index.read_text())
    selection = getattr(args, 'units', None) or ''
    repeats = getattr(args, 'repeats', 1) or 1
    expected = attempts(select_units(sorted({PurePosixPath(entry['path']).parts[1] for entry in index['files']}),
                                     selection, load_classes()), repeats)
    latest = {}
    for path in sorted(args.merge.rglob('shard-*.json')):
        shard = json.loads(path.read_text())
        attempt = shard.get('run_attempt') or 0
        if shard['shard'] not in latest or attempt > latest[shard['shard']].get('run_attempt', 0):
            latest[shard['shard']] = shard  # a re-run replaces that shard's earlier attempt
    rows, shards = {}, []
    for _, shard in sorted(latest.items()):
        shards.append({k: shard.get(k) for k in ('shard', 'run_attempt', 'image_id', 'model_name', 'eval_backend',
                                                 'eval_throttle', 'unit_budget_sec', 'budget_mode', 'replay')})
        for row in shard['units']:
            key = (row['id'], row.get('repeat', 0))
            if key in rows:
                raise ValueError('Unit reported by two shards')
            if key not in expected:
                raise ValueError('A shard reported a unit outside the selection')
            rows[key] = row
    for unit, k in expected:
        rows.setdefault((unit, k), dict({'id': unit, 'status': 'not_run_shard_missing', 'model_calls': 0},
                                        **({'repeat': k} if repeats > 1 else {})))
    ordered = [rows[key] for key in expected]
    backends = {shard.get('eval_backend') for shard in shards}
    backend = backends.pop() if len(backends) == 1 else 'mixed' if backends else None
    result = {'schema_version': 1, 'source_commit': index['commit'], 'eval_backend': backend, 'shards': shards,
              'summary': summarize(ordered, backend), 'units': ordered}
    if selection.strip():
        result['selection'] = selection.strip()
    if repeats > 1:
        result['repeats'] = repeats
        passes = {}
        for row in ordered:
            passes[row['id']] = passes.get(row['id'], 0) + (row['status'] == 'passed')
        result['passes_by_unit'] = passes
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'public-eval.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def markdown(result):
    summary = result['summary']
    lines = [f"**Passed {summary['passed']} / {summary['units']} public units** "
             f"({summary['not_measured']} not measured). {summary['scope']}.", '',
             '| Unit | Status | Failure | Tests (passed/failed/error) | Tokens | Continued | Repaired | Seconds |',
             '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for row in result['units']:
        counts = row.get('test_counts') or {}
        tests = (f"{counts.get('passed', 0)}/{counts.get('failed', 0)}/{counts.get('error', 0)}"
                 if counts else '')
        failure = (row.get('failure_code') or '') + (f"({row['http_status']})" if row.get('http_status') else '')
        # A suffix only when it adds something: the agent's failure behind a decisive verdict, or the
        # non-decisive verdict behind a crashed unit's own status.
        agent, verdict = row.get('agent_status'), row.get('verdict')
        status = row['status']
        if verdict and agent not in (None, 'executed_ungraded', row['status']):
            status += f' ({agent})'
        if verdict and verdict != row['status'] and agent != 'executed_ungraded':
            status += f' [{verdict}]'
        lines.append(f"| {row['id']} | {status} | {failure} | {tests} | "
                     f"{(row.get('usage') or {}).get('completion_tokens', '')} | {row.get('continuations', '')} | "
                     f"{row.get('repairs', '')} | {row.get('duration_sec', '')} |")
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=Path, default=ROOT / 'evidence/public_units.json')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--merge', type=Path, help='directory holding shard-*.json files to combine')
    parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--units', default='', help='unit ids or classes from selection/classes.json; all when empty')
    parser.add_argument('--repeats', type=int, default=1, help=f'attempts per unit (1-{MAX_REPEATS}), each on its own')
    parser.add_argument('--image')
    parser.add_argument('--work', type=Path)
    parser.add_argument('--gate-receipt', type=Path, help='integration fixture receipt for --image')
    parser.add_argument('--context', default='default')
    parser.add_argument('--unit-timeout', type=int, default=420)
    parser.add_argument('--explicit-budget', action='store_true',
                        help='pass T1_TIMEOUT_SEC (min of --unit-timeout and the card) instead of the image budget')
    parser.add_argument('--checker-timeout', type=int, default=300)
    parser.add_argument('--cpus', type=float, default=2)
    parser.add_argument('--memory', default='6g')
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--replay-pubkey', type=Path,
                        help='keep each unit\'s raw replay, sealed for this public key only (local evaluation)')
    args = parser.parse_args()
    if args.merge:
        result = merge(args)
        summary_path = os.environ.get('GITHUB_STEP_SUMMARY')
        if summary_path:
            with open(summary_path, 'a') as stream:
                stream.write(markdown(result))
        print(json.dumps(result['summary']))
        return exit_code(result['units'])
    if not args.image or not args.work:
        parser.error('--image and --work are required to run a shard')
    run_shard(args)  # a shard that wrote its file has done its job; the merge judges the whole run
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
