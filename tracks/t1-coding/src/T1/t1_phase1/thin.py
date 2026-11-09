"""One contract, versioned code, and one isolated execution/repair loop.

Readiness and internal checks are operational evidence, never an official score.
"""
from __future__ import annotations

import ast
import hashlib
import copy
import importlib
import importlib.util
import inspect
import json
import keyword
import math
import os
import re
import shutil
import stat
import tempfile
import time
from pathlib import Path

from . import (api_returns, audit, code_step, date_validation, draft_compare, input_binding, literal_contract,
               local_repair, output_report, resource_control, stub_check)
from .house import HouseError, _is_program, strip_reasoning, temperature_for
from .program_version import (FORMAT_REASONS, READY, SYNTAX_NOTE, ApplyResult, FormatError, ProgramVersion,
                              _bindings, _cutoff, _parse, executed_note, extract_code, module_names as _module_names)
from .solver_profile import THIN_MAX_CONTINUATIONS, parse_thin_max_continuations
from .solver_profile import (THIN_FEATURES, THIN_MAX_CONTINUATIONS_UNIT, parse_thin_features,
                             parse_thin_max_continuations_unit)
from .solver_profile import (FINAL_DELIVERY_TRIGGERS, THIN_FINAL_DELIVERY_TRIGGER,
                             parse_thin_final_delivery_trigger)

OUTPUT_CAP = 63 * 1024 * 1024
MAX_TOKENS = 4000
# Transient House failures (429/408/409/425/5xx, transport) are retried after these waits while the unit's
# time and request cap allow. Every attempt is one counted request.
RETRY_WAITS = (2, 4, 8, 15, 25, 40)
RETRY_FLOOR = 30  # request seconds that must remain after a wait, beyond the caller's reserve
# A free-mode code reply cut at the output limit is continued from its last complete line.
MAX_CONTINUATIONS = THIN_MAX_CONTINUATIONS  # default requests per code reply; T1_THIN_MAX_CONTINUATIONS (0-3)
CONTINUE_FLOOR = 30  # request seconds a continuation needs beyond the execution reserve
CONVERSATION_CAP = 170_000  # bytes per continuation conversation; House admits at most 180,000
NO_PROGRESS_LIMIT = 4  # consecutive free-mode replies that leave the program and its pending work unchanged
# S2-06 (switch final_delivery). While a required deliverable is still missing near the end of the unit's requests or
# time, the ask is FINAL DELIVERY (BUNDLE or REPLACE). The trigger is exactly the pre-registered OS-2 definition
# (R2A_PARAMS.md): REMAINING REQUESTS <= r or REMAINING SECONDS <= s as the prompt shows them, with a deliverable
# missing (as the last run left them); no other threshold. The complete first version is its own switch,
# complete_first (COMPLETE_FIRST).
FINAL_TRIGGERS = FINAL_DELIVERY_TRIGGERS  # T1_THIN_FINAL_DELIVERY_TRIGGER (solver_profile: forwarded, identity)
FINAL_TRIGGER = THIN_FINAL_DELIVERY_TRIGGER  # default: the candidate that intervenes less
# The asks that allow BUNDLE or REPLACE. A prompt the trigger holds for whose ask is REPLACE-only (required, module,
# first) keeps that ask and is counted apart: thin_final_delivery_replace_only.
FINAL_ASKS = ('recover', 'default')
FINAL_WHY = {'requests': 'few requests remain', 'seconds': 'little time remains'}
FINAL_NAMES_SHOWN = 8
# S2-08 (switch loop): a reply that returns the program and its pending work to a state the loop already held is no
# progress, like an unchanged one.
REVISIT = ('\nREVISIT: this reply returned to version {version}, which was admitted before; going back to an '
           'earlier version counts as no progress.')
# A2 (switch resample): after RESAMPLE_AFTER consecutive free-loop code replies that left no new version to run (the
# program and its pending work unchanged, or back at a version that already ran), the next code request is sampled at
# temperature_for(RESAMPLE_KIND) instead of greedily, while no run so far has left every required deliverable (any
# file when the contract names none). Requests are counted and NO_PROGRESS_LIMIT stops the unit exactly as before;
# the request is a thin_code request (its record carries the temperature), RESAMPLE_KIND only names its temperature.
RESAMPLE_AFTER = 2
RESAMPLE_KIND = 'thin_code_resample'
# B (switch assert_rerun; always on through R2, off by default since R3): a free-loop run that stopped at an `assert`
# statement of the program itself (_own_assert_line) runs once more, the same version with PYTHONOPTIMIZE=1, capped at
# max(ASSERT_RERUN_FLOOR, ASSERT_RERUN_FACTOR x that run) and only while the next code request still fits after it (or
# none can follow). The rerun only competes for the delivered outputs, and only when it leaves more required
# deliverables than the best run so far (_assert_rerun); it never passes the internal checks or ends the loop, and the
# model is still told about the assertion failure. With the switch off such a run is handled like any other failure.
ASSERT_RERUN_FACTOR = 2
ASSERT_RERUN_FLOOR = 20
# second_draft (switch, 10-28): once the free loop ends with a run that passed the internal checks, up to SECOND_DRAFTS
# more programs are written from the empty program (the unit's first code prompt, sampled at
# temperature_for(RESAMPLE_KIND)) and run on their own. Their deliverables vote (draft_compare.outputs_agree): a draft
# that states the first run's answer ends the vote with the first run kept; two drafts that state the same answer
# against it replace it; anything else, a failed or unfinished draft included, keeps it. A draft starts only with
# SECOND_DRAFT_ROOM seconds left (one 4,000-token reply at House's 60 tok/s, its run and the restore) and a request left.
SECOND_DRAFTS = 2
SECOND_DRAFT_ROOM = 150
SECOND_DRAFT_MARGIN = 30  # seconds kept after a draft's run for the vote and the restore
SECOND_DRAFT_OUTCOMES = ('no_room', 'unfinished', 'disagreed', 'agreed', 'replaced')  # public_eval's enums
SECOND_DRAFT_DIFFERS = ('missing', 'differs', 'no_outputs')
SECOND_DRAFT_ERRORS = ('house_error', 'other')
SECOND_DRAFT_NOTE = ('\n\nINDEPENDENT CHECK: write the complete program from scratch. Read the TASK again line by '
                     'line: every stated formula, convention, unit, date rule, rounding and output field.')
CODE_FENCES =('```', '```python', '```py', '```python3')
RESULT_FIELDS = ('thin_output_version', 'thin_missing_files', 'thin_deliverables_present',
                 'thin_expected_deliverables', 'thin_internal_checks_passed', 'thin_date_guard_status',
                 'thin_date_guard_calls', 'thin_date_guard_unresolved', 'thin_date_guard_blocking')
CONTINUE = ('Your previous reply reached the output limit. The assistant turn above keeps its complete lines; '
            'the cut partial line was removed. Continue the same program from the line after `{line}`, keeping '
            'its indentation. Output only the remaining lines in one ```python fence and close the fence. Do not '
            'repeat earlier lines, restart the program or add explanation. Keep the rest compact and finish with '
            'the program entrypoint.')
STALL = ('\nNO PROGRESS: {n} consecutive replies left the program and its pending work unchanged (limit {limit}). '
         'Do not resend the same code. Make a different, compact change that resolves the actual feedback above; '
         'a complete REPLACE against CURRENT VERSION is allowed.')
# The same notice on a CURRENT PROGRAM over PROGRAM_BUDGET_CHARS (S0-16 B5): too close to the reply limit to resend.
STALL_LARGE = ('\nNO PROGRESS: {n} consecutive replies left the program and its pending work unchanged '
               '(limit {limit}). Do not resend the same code. Make a different, compact change that resolves the '
               'actual feedback above. CURRENT PROGRAM is {size:,} characters, too close to the reply limit of '
               'about 10,000 characters to resend whole, so a REPLACE must be a structurally smaller complete file '
               '(see OUTPUT FORMAT).')
# A CURRENT PROGRAM longer than this is too close to the reply limit (MAX_TOKENS, about 10,000 characters) to be
# resent whole with its action line, fence and any reasoning: a budget, not a hard bound (s14c: 8-10k resends cut).
PROGRAM_BUDGET_CHARS = 8000
# The free loop's first ACTUAL FEEDBACK starts with this; first_feedback adds the first line (S0-16 B1).
FIRST_FEEDBACK = 'Implement the contract; there has been no execution yet.'
NO_CONTRACT = ('None: the planning replies were empty. Derive the required outputs, conventions and one first '
               'runnable milestone directly from the TASK.')
HEADER = r'^\s*(REPLACE|BUNDLE)\s+([0-9a-f]{64})\s*\n'
LATE_HEADER = r'^[ \t]*(REPLACE|BUNDLE)[ \t]+([0-9a-f]{64})[ \t]*\n'  # on its own line after leading prose


def _direct_payload(payload, finish_reason):
    """Remove one redundant final fence without changing any code bytes."""
    if finish_reason != 'stop':
        return payload
    lines = payload.splitlines(keepends=True)
    fences = [i for i, line in enumerate(lines) if line.strip().startswith('```')]
    if len(fences) != 3:
        return payload
    opening, closing, duplicate = fences
    if (lines[opening].strip() not in ('```python', '```py', '```python3')
            or lines[closing].strip() != '```' or lines[duplicate].strip() != '```'
            or any(line.strip() for line in lines[:opening])
            or any(line.strip() for line in lines[closing+1:duplicate])
            or any(line.strip() for line in lines[duplicate+1:])):
        return payload
    source = ''.join(lines[opening+1:closing])
    if not source.strip():
        return payload
    try:
        compile(source, '<direct-reply>', 'exec')
    except (SyntaxError, ValueError, RecursionError):
        return payload
    return ''.join(lines[:duplicate] + lines[duplicate+1:])


def _direct_action(state, status, *, replacement_required=False):
    """Choose the operation before requesting code; a reply cannot override it."""
    if not state.program or replacement_required:
        return 'replace'
    named = False
    for pending in state.pending:
        if pending == 'entry':
            continue
        kind, _, name = pending.partition(':') if isinstance(pending, str) else ('', '', '')
        if (kind not in ('incomplete', 'stub') or not name.isidentifier()
                or keyword.iskeyword(name) or name == 'module'):
            return 'replace'
        named = True
    if named:
        return 'bundle'
    if not state.pending and status.get('thin_executions', 0) and status.get('exit_code') != 0:
        return 'replace'
    if status.get('thin_date_guard_blocking'):
        return 'replace'
    if (state.ready and status.get('thin_executions', 0) and status.get('exit_code') == 0
            and status.get('thin_output_version') == state.version):
        return 'append_step'
    return 'replace'


def _code_payload(raw, finish):
    """Validate the whole direct reply; never search prose for usable code."""
    if finish not in ('stop', 'length'):
        raise ValueError('unconfirmed_finish')
    if not isinstance(raw, str):
        raise ValueError('nontext_code')
    text = _direct_payload(raw, finish).replace('\r\n', '\n').replace('\r', '\n').strip()
    enclosed = text.startswith('```')
    if enclosed:
        lines = text.splitlines()
        if lines[0].lower() not in ('```python', '```py', '```python3'):
            raise ValueError('invalid_code_fence')
        closed = len(lines) > 1 and lines[-1].strip() == '```'
        body = lines[1:-1] if closed else lines[1:]
        if any(line.strip().startswith('```') for line in body) or not closed and finish != 'length':
            raise ValueError('ambiguous_or_unclosed_code_fence')
        text = '\n'.join(body)
    if not text.strip():
        raise ValueError('empty_code')
    if re.match(r'^\s*(?:REPLACE|BUNDLE|APPEND_STEP)\s+[0-9a-fA-F]{64}(?:\s|$)', text):
        raise ValueError('reply_action_header_forbidden')
    source = text.rstrip() + '\n'
    try:
        compile(source, '<direct-code>', 'exec', dont_inherit=True)
    except SyntaxError:
        if not enclosed and any(line.strip().startswith('```') for line in source.splitlines()):
            raise ValueError('ambiguous_code_fence') from None
        # ProgramVersion alone decides whether a length reply has a genuine EOF tail.
        if finish != 'length':
            raise ValueError('invalid_syntax') from None
    except (ValueError, RecursionError, MemoryError, OverflowError):
        raise ValueError('invalid_syntax') from None
    else:
        if finish == 'length':
            raise ValueError('truncated_reply_not_complete')
    return source


def _local_repair_target(state, status, *, replacement_required=False):
    """Use a worker location only as a scope hint for the exact failed version."""
    diagnostic = status.get('execution_diagnostics')
    if (replacement_required or not state.ready or not status.get('thin_executions')
            or status.get('status') != 'execution_failed' or status.get('exit_code') != 1
            or not isinstance(diagnostic, dict)
            or diagnostic.get('source') != 'untrusted_execution_log'
            or diagnostic.get('log_readable') is not True):
        return None
    return local_repair.select(state.program, base_version=state.version,
                               executed_version=status.get('thin_output_version'),
                               line=diagnostic.get('generated_line'))


CONTEXT_BUDGET = 130_000  # bytes of task facts; House admits at most 180,000 per request
CONTEXT_CUT = ('\n[INPUT PROFILE cut here to fit the context budget; read the input files themselves for the rest.]')
# S3-03b OB-B12 (switch repair_feedback): what holds in every run, the platform's too. Programs run with the output
# directory as their working directory (baseline._run_program), INPUT PROFILE paths are relative to the input
# root (baseline.input_profile), and every written file has the 64 MiB RLIMIT_FSIZE; the read-only root and the
# 64 MiB /tmp of the default sentence are the local runner's (docker_runner).
RUN_PLACES = ('Programs run with {out} as the working directory, and INPUT PROFILE paths are relative to '
              '{task_dir}. Each file a program writes is limited to 64 MiB, separate from the process memory limit '
              'below. ')
# lib_expose (switch, 10-28; from the library-use review: 62 of 77 failed units had a strength-3 library match, yet no
# failing program called a numeric qfkit function, and a few long docstrings lost their Returns section to the cut).
# The task facts show each confident match (baseline.library_confident: strength 2 or more), at most
# LIB_EXPOSE_CONTEXT, as its signature and docstring title instead of the bare name; the unit's first code prompts (an
# empty program) document the first LIB_EXPOSE_CODE of them under CONFIRMED REQUESTED APIs, named in the contract or
# not; a docstring over API_DOC_CHARS keeps its head and its Returns section (_doc_cut); and the code prompt's
# guidance gains LIBRARY_FIRST. Off, every prompt is as before.
LIB_EXPOSE_CONTEXT = 8
LIB_EXPOSE_CODE = 6
LIB_EXPOSE_HEAD = ('\nLibrary functions that compute methods the TASK names (optional; signature - docstring title; '
                   'name the ones you use in APIS):\n')
LIB_NAMES_HEAD = '\nPotential relevant library names (optional; request signatures in APIS):\n'
# lib_expose: the task facts' library sentence, which read as a ban (the unit doctors, 10-28: plans wrote 'APIS: None
# (qfkit not permitted; implement it manually)' on TASKs that never restrict libraries). (old, new), found once.
LIB_PERMISSION = (('You may use installed numpy, scipy, pandas, pyarrow, matplotlib (Agg), statsmodels and qfkit '
                   'only where the TASK permits them. qfkit is a generic numerical library, not a task solution. '
                   'Use an API only after its signature is confirmed; otherwise implement it. ',
                   'You may use installed numpy, scipy, pandas, pyarrow, matplotlib (Agg), statsmodels and qfkit '
                   'unless the TASK forbids a library. qfkit is a generic, verified numerical library, not a task '
                   'solution: for a standard method the TASK names without prescribing how to compute it, prefer its '
                   'documented function (signatures and documentation are shown) over re-deriving the formula, and '
                   'pass the TASK\'s conventions through its parameters. '),)
LIBRARY_FIRST = ('LIBRARY FIRST: when a documented qfkit function shown above implements a method the TASK names and '
                 'the TASK does not prescribe how to compute it, call it instead of re-deriving the formula; pass the '
                 "TASK's conventions through its parameters and check its Returns section for units and sign before "
                 'writing outputs. When the TASK prescribes the computation itself (an explicit formula, a named '
                 'numpy/scipy/pandas call, an estimator option such as ddof or a quantile method), implement exactly '
                 'that and use qfkit only to cross-check it.\n')
API_DOC_CHARS = 3500  # each function's documentation in CONFIRMED REQUESTED APIs / CONFIRMED APIs ALREADY USED
API_DOC_CUT = '\n[... documentation cut here to fit; its Returns section follows ...]\n'
API_RETURNS = re.compile(r'(?m)^Returns\n-{3,}\n')  # a numpy-style Returns heading (as api_returns._returns)
API_SECTION = re.compile(r'(?m)^[A-Za-z][^\n]*\n-{3,}\s*$')  # the next numpy-style section heading


BASE_LEAD = ('No networks or package installation. Use the supplied data and preserve the exact methods, '
             'sample sizes, units, seeds, precision and filenames. Do not write reward.json or pytest_report.json '
             'or copy input metadata into outputs. Read full input files where the profile is insufficient. '
             'You may use installed numpy, scipy, pandas, pyarrow, matplotlib (Agg), statsmodels and qfkit '
             'only where the TASK permits them. qfkit is a generic numerical library, not a task solution. '
             'Use an API only after its signature is confirmed; otherwise implement it. '
             'Include assertions for visible requirements and independent numerical consistency. '
             'Batch large arrays while preserving the specified estimator and random sampling contract.\n')


def _qfkit_function(name):
    """The installed public qfkit function `name`; ImportError or AttributeError when there is none."""
    spec = importlib.util.find_spec('qfkit')
    if spec is None or spec.origin is None:
        raise ImportError('qfkit is not an installed regular package')
    obj = getattr(importlib.import_module('qfkit'), name)
    if not callable(obj):
        raise AttributeError(name)
    return obj


def _library_notes(details):
    """lib_expose: the task facts' library notes for library_match_details' `details`: each confident match (strength
    2 or more), at most LIB_EXPOSE_CONTEXT, as qfkit.name(signature) - docstring title, one line each; the other
    matches, and any whose signature cannot be read, by bare name under the old head (alone, it is the old note)."""
    shown, bare = [], []
    # confident matches first, then strength-1 ones (a method the matcher found only through a cited term, such as
    # Crank-Nicolson or PSOR for fd_solve) while there is room; each keeps its matcher order
    details = sorted(details, key=lambda row: row[3] < 2)
    for function, _, title, strength in details:
        line = None
        if strength >= 1 and len(shown) < LIB_EXPOSE_CONTEXT:
            try:
                line = ('qfkit.' + function + str(inspect.signature(_qfkit_function(function)))
                        + (' - ' + ' '.join(title.split()) if isinstance(title, str) and title.strip() else ''))
            except (ImportError, AttributeError, TypeError, ValueError):
                line = None
        if line is None:
            bare.append(function)
        else:
            shown.append(line)
    return ((LIB_EXPOSE_HEAD + '\n'.join(shown) if shown else '')
            + (LIB_NAMES_HEAD + ', '.join('qfkit.' + name for name in bare) if bare else ''))


def _exposed(instruction):
    """lib_expose: the confident matches (strength 2 or more) of the TASK, in matcher order, that the unit's first code
    prompts document unasked: at most LIB_EXPOSE_CODE; () when there are none or the matcher fails."""
    from . import baseline
    try:
        details = baseline.library_match_details(instruction)
        return tuple([function for function, _, _, strength in details if strength >= 2]
                     + [function for function, _, _, strength in details if strength == 1])[:LIB_EXPOSE_CODE]
    except Exception:
        return ()


def _doc_cut(doc, limit=API_DOC_CHARS):
    """`doc` within `limit` characters (lib_expose): over it, its head, API_DOC_CUT and its numpy-style Returns
    section (the section's own tail cut when it alone would leave the head under a third of the room), so the units
    and shapes a function returns are never what the cut drops; without a Returns section, or with one the head
    keeps whole anyway, the head alone as before."""
    if len(doc) <= limit:
        return doc
    match = API_RETURNS.search(doc)
    if not match:
        return doc[:limit]
    following = API_SECTION.search(doc, match.end())
    stop = following.start() if following else len(doc)
    if stop <= limit:
        return doc[:limit]
    section = doc[match.start():stop].rstrip()
    room = limit - len(API_DOC_CUT)
    section = section[:max(0, room - limit // 3)] if len(section) > room - limit // 3 else section
    head = doc[:min(match.start(), room - len(section))]
    line = head.rfind('\n')
    head = (head[:line] if line > len(head) // 2 else head).rstrip()
    return head + API_DOC_CUT + section


# lib_review (switch, 10-28; the 86 unit doctors: qfkit reproduces the correct numbers for 47 of the 75 failing units,
# yet their programs hand-wrote the methods and got them wrong). While the TASK permits the library
# (baseline.library_allowed_by_task) and it has library matches (_library_matched: the confident ones, strength 2 or
# more, else the weak ones, strength 1), a program that uses none of them (_library_used) gets, at the OUTPUT REVIEW
# hold, LIB_CROSS_CHECK naming at most LIB_REVIEW_SHOWN of them, and after a failed run (a crash or failed checks)
# LIB_REPAIR_HINT naming those not yet hinted (each function once per unit). The code prompt that carries either
# documents the named functions first under CONFIRMED REQUESTED APIs (api_context's `first`, with their Returns
# section kept, _doc_cut). Off, every prompt is as before.
LIB_REVIEW_SHOWN = 3
LIB_CROSS_CHECK_MARK = '\nLIBRARY CROSS-CHECK: '
LIB_CROSS_CHECK = (LIB_CROSS_CHECK_MARK + 'the program calls none of the qfkit functions that implement methods the TASK '
                   'names (signature - docstring title; each is documented first under CONFIRMED REQUESTED APIs):'
                   '\n{functions}\nRecompute the main deliverable values with them in the program and compare '
                   "them with the program's own values. Where they differ, decide by the TASK's stated convention: the "
                   "TASK wins over a library default, so pass the TASK's conventions through the function's "
                   'parameters. Then either reply KEEP with no code, when the outputs already hold the values the TASK '
                   'defines, or send the fix against CURRENT VERSION.')
LIB_REPAIR_HINT_MARK = '\nLIBRARY HINT: '
LIB_REPAIR_HINT = (LIB_REPAIR_HINT_MARK + 'qfkit implements methods the TASK names that this program re-derives: '
                   '{functions} (documented first under CONFIRMED REQUESTED APIs). If the failing computation is one '
                   "of them, call that function instead of re-deriving it, passing the TASK's conventions through its "
                   'parameters.')


def _library_matched(instruction):
    """lib_review: (function, docstring title) of the TASK's library matches a LIBRARY CROSS-CHECK or LIBRARY HINT may
    name, in matcher order: the confident ones (strength 2 or more), else the weak ones (strength 1); () when the TASK
    restricts its dependencies (baseline.library_allowed_by_task), has no match, or the matcher fails."""
    from . import baseline
    try:
        if not baseline.library_allowed_by_task(instruction):
            return ()
        details = baseline.library_match_details(instruction)
        confident = tuple((function, title) for function, _, title, strength in details if strength >= 2)
        return confident or tuple((function, title) for function, _, title, strength in details if strength >= 1)
    except Exception:
        return ()


def _library_used(program):
    """lib_review: the qfkit names the program refers to, read without running it (as baseline.library_called, also
    through a qfkit submodule): `q.f` or `q.mod.f` through each name `import qfkit[.mod] [as q]` binds, `mod.f` after
    `from qfkit import mod`, and each name `from qfkit[.mod] import f [as g]` binds, called or passed on; after a star
    import from qfkit, each name the program calls. A name in a comment or a string is not one. set() when the program
    does not parse."""
    try:
        tree = ast.parse(program)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return set()
    modules, imported, star = set(), {}, False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(item.asname or 'qfkit' for item in node.names
                           if item.name == 'qfkit' or item.name.startswith('qfkit.'))
        elif (isinstance(node, ast.ImportFrom) and not node.level and node.module
              and (node.module == 'qfkit' or node.module.startswith('qfkit.'))):
            for item in node.names:
                if item.name == '*':
                    star = True
                else:
                    imported[item.asname or item.name] = item.name
    used = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            root = node.value
            while isinstance(root, ast.Attribute):
                root = root.value
            if isinstance(root, ast.Name) and (root.id in modules or root.id in imported):
                used.add(node.attr)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id in imported:
            used.add(imported[node.id])
        elif star and isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            used.add(node.func.id)
    return used


def _library_unused(matches, program, hinted=()):
    """lib_review: [(function, qfkit.name(signature) - docstring title)] of at most LIB_REVIEW_SHOWN of `matches`
    (_library_matched), in its order and outside `hinted`, when the program uses none of them (_library_used); []
    when it uses one, or when none can be read (qfkit not installed)."""
    if not matches or _library_used(program) & {function for function, _ in matches}:
        return []
    shown = []
    for function, title in matches:
        if function in hinted or len(shown) >= LIB_REVIEW_SHOWN:
            continue
        try:
            line = 'qfkit.' + function + str(inspect.signature(_qfkit_function(function)))
        except Exception:  # not installed, gone, or unreadable: never named (the switch must not end a unit)
            continue
        shown.append((function, line + (' - ' + ' '.join(title.split())
                                        if isinstance(title, str) and title.strip() else '')))
    return shown


def task_context(task_dir, out, instruction, profile, hints, canaries, status):
    """The task facts every request starts with. Over CONTEXT_BUDGET bytes (S3-03b CMP-4) the least needed parts go
    one step at a time instead of ending the unit with no request: the optional library notes, then the profile's
    layout whitespace, then the profile's tail down to none. The TASK and the input-location hints (every
    original -> runtime mapping) are never cut; when they do not fit together the unit still ends with
    thin_context_limit. status thin_context_degraded counts the steps."""
    from . import baseline
    # The profile is already bounded by baseline's PROFILE_CAP; the whole context keeps its byte check below.
    text = json.dumps(profile, ensure_ascii=False)
    room = (RUN_PLACES.format(out=out, task_dir=task_dir) if _feature('repair_feedback') else
            'The root is read-only; /tmp has 64 MiB of filesystem space, separate from the process memory limit '
            'below. ')
    resources = resource_control.context(resource_control.snapshot()).replace('In the short Plan state', 'Account for')

    def build(text, hints, optional):
        lead = BASE_LEAD
        if _feature('lib_expose') and baseline.library_allowed_by_task(instruction):
            for old, new in LIB_PERMISSION:
                lead = lead.replace(old, new, 1)
        return (f'Solve the quantitative-finance coding task below. Read inputs under {task_dir}; '
                f'write deliverables under {out}. ' + room + lead + resources + '\nRUNTIME INPUT LOCATIONS (authoritative for file access):\n'
                + f'Actual input root: {task_dir}. Original TASK paths such as /app are not mounted here. '
                'Use the mappings below and INPUT PROFILE paths; inspect this input root if a mapping is absent. '
                'This relocation changes paths only, not the task methods or data.\n' + hints
                + '\nTASK (authoritative for methods, definitions and required outputs):\n'
                + instruction + '\nINPUT PROFILE (bounded observations):\n' + text + optional)
    optional = ''
    if baseline.library_allowed_by_task(instruction):
        optional += ('\nOptional input checks: qfkit.parse_calendar_dates(values, *, formats, missing="raise") '
                     'returns (date_series, report), preserving index/order/duplicates. Supply explicit strptime '
                     'formats from the input contract; invalid or ambiguous dates raise. missing="preserve" '
                     'allows original nulls only. qfkit.filter_rows(frame, keep, *, reason) returns '
                     '(filtered_frame, report) with per-stage row counts and current-input positions. '
                     'Choose each parsing and filtering policy from the TASK. Assert parsing coverage before '
                     'filters; validate each removal count against the actual mask.\n')
        try:
            if _feature('lib_expose'):  # the confident matches with their signatures and titles (_library_notes)
                optional += _library_notes(baseline.library_match_details(instruction)[:16])
            else:
                names = [name for name, _, _ in baseline.library_matches(instruction)][:16]
                if names:
                    optional += '\nPotential relevant library names (optional; request signatures in APIS):\n' + ', '.join('qfkit.' + name for name in names)
        except Exception:
            pass
    hints = hints[:6000]
    context = baseline.withhold_canaries(build(text, hints, optional), canaries)
    if len(context.encode()) > CONTEXT_BUDGET:
        # Each cut part is withheld before it is cut, so no cut leaves part of a canary behind.
        compact = baseline.withhold_canaries(json.dumps(profile, ensure_ascii=False, separators=(',', ':')), canaries)
        shown = baseline.withhold_canaries(hints, canaries)

        def whole(profile_text, hint_text):
            return baseline.withhold_canaries(build(profile_text, hint_text, ''), canaries)

        def over(candidate, marker):  # bytes the part must lose for candidate, with marker added, to fit
            return len(candidate.encode()) + len(marker.encode()) - CONTEXT_BUDGET
        for steps in range(1, 4):
            if steps == 1:  # the optional library notes
                context = whole(text, hints)
            elif steps == 2:  # the profile's layout whitespace
                context = whole(compact, shown)
            else:  # the profile's tail, down to none; the hints stay whole, so no mapping a program needs is hidden
                context = whole(_cut_to(compact, over(whole(compact, shown), CONTEXT_CUT)) + CONTEXT_CUT, shown)
            if len(context.encode()) <= CONTEXT_BUDGET:
                status['thin_context_degraded'] = steps
                break
    if len(context.encode()) > CONTEXT_BUDGET:
        raise HouseError('Task facts exceed context budget', kind='local_input', code='thin_context_limit')
    status['thin_context_chars'] = len(context)
    return context


def _cut_to(text, drop):
    """text without its last `drop` UTF-8 bytes (more when a character straddles the cut); '' when that is all."""
    data = text.encode()
    return data[:max(0, len(data) - max(0, drop))].decode('utf-8', 'ignore') if drop > 0 else text


PLAN = '''\nACTION: Produce a short implementation contract, at most 250 words, no code.
Record the required outputs and shared quantities/randomness, then order the executable milestones.
Choose ONE first computation that can run now and resolve a concrete uncertainty. State its formula
and units with task evidence, plus a small independent numerical check and expected relation.
Flag conflicting statements instead of silently mixing conventions; use explicit formulas and visible
identities to investigate them. Leave later derivations for their implementation step. Estimate actual
array shapes/live buffers; filesystem space is not RAM. Include APIS: followed by exact qfkit names
needed for the next step, or none. Do not repeat the task. This is the only planning request.
'''
# complete_first (audit 2.7): the contract asks for each deliverable's method chain instead of ONE first computation
# and no longer defers later derivations; (old, new) replacements of PLAN, each found exactly once (a test checks).
PLAN_COMPLETE_FIRST = (
    ('Choose ONE first computation that can run now and resolve a concrete uncertainty. State its formula\n'
     'and units with task evidence, plus a small independent numerical check and expected relation.',
     'For each required deliverable, state its method chain: the inputs it reads, each formula with its\n'
     'units and task evidence, and the file it writes, plus a small independent numerical check and\n'
     'expected relation.'),
    (' Leave later derivations for their implementation step.', ''),
    ('needed for the next step, or none.', 'needed for these methods, or none.'))


def plan_prompt(context, *, complete_first=False, fidelity=False):
    """The free loop's one contract request; `complete_first` (switch complete_first) asks for every deliverable's
    method chain (PLAN_COMPLETE_FIRST); `fidelity` (switch task_fidelity) also asks for the TASK requirement list
    (FIDELITY_PLAN). Off, the text is PLAN as it always was."""
    text = PLAN
    if complete_first:
        for old, new in PLAN_COMPLETE_FIRST:
            text = text.replace(old, new, 1)
    if fidelity:
        for old, new in FIDELITY_PLAN:
            text = text.replace(old, new, 1)
    return context + text


def _qfkit_imported(text):
    """lib_expose: the qfkit names `text` reaches without spelling `qfkit.X`: each imported name of
    `from qfkit import a, b as c` (a and b) and each `q.name` after `import qfkit as q`, in that order."""
    names = []
    for group in re.findall(r'^[ \t]*from[ \t]+qfkit[ \t]+import[ \t]+(\([^)]*\)|[^\n#;]*)', text, re.M):
        for item in group.strip().strip('()').split(','):
            words = item.split()
            if words and re.fullmatch(r'[A-Za-z]\w*', words[0]):
                names.append(words[0])
    for alias in re.findall(r'^[ \t]*import[ \t]+qfkit[ \t]+as[ \t]+([A-Za-z_]\w*)', text, re.M):
        if alias != 'qfkit':
            names += re.findall(r'(?<![\w.])' + re.escape(alias) + r'\.([A-Za-z]\w*)', text)
    return names


def api_context(contract, extra=(), first=()):
    """Read only installed public qfkit functions explicitly named in current work. `extra` (lib_expose: the confident
    library matches of a first code prompt, _exposed) follows the named ones, without repeating one; under lib_expose
    a docstring over API_DOC_CHARS keeps its Returns section (_doc_cut). `first` (lib_review: the functions a LIBRARY
    CROSS-CHECK or LIBRARY HINT names) precedes them all, so the cap never cuts those, and keeps its Returns section
    too. The block's cap is unchanged."""
    expose = _feature('lib_expose')
    found = re.findall(r'\bqfkit\.([A-Za-z]\w*)', contract)
    if expose:  # also `from qfkit import a, b as c` and `import qfkit as q` ... `q.name`
        found += _qfkit_imported(contract)
    names = list(dict.fromkeys(found))[:16]
    names += [name for name in dict.fromkeys(extra) if name not in names]
    if first:
        lead = list(dict.fromkeys(first))
        names = lead + [name for name in names if name not in lead]
    parts = []
    for name in names:
        try:
            obj = _qfkit_function(name)
            signature = str(inspect.signature(obj))
            doc = inspect.getdoc(obj) or 'No documentation; inspect implementation before use.'
            parts.append('qfkit.' + name + signature + '\n' + (_doc_cut(doc) if expose or name in first
                                                               else doc[:API_DOC_CHARS]))
        except (ImportError, AttributeError, TypeError, ValueError):
            parts.append('qfkit.' + name + ': unavailable; implement it or select a documented function.')
    return '\n\n'.join(parts)[:32000]


# OUTPUT FORMAT pieces (S0-16 B5). None of them holds a line that starts with a prompt marker (ACTION:, PENDING:,
# CURRENT VERSION:, ...): readers take the last ACTION line as the ask and rebuild the state from the markers.
FIT = ('The whole reply must stay well under the 4,000-token output limit (about 10,000 characters), so write '
       'compact code with small helper functions and no long literal data.')
SMALLER = ('keep each implemented calculation once, put top-level work in functions called from one __main__ guard, '
           'and drop repeated bindings, debug prints and long literal data.')
CUT_REPLACE = 'A cut REPLACE continues from the last complete line.'
# How ProgramVersion merges a BUNDLE (program_version._bundle, OB-A6), stated once per prompt that offers BUNDLE.
# Statements are keyed by their exact set of bound names (_bindings): a tuple assignment binds no key, and an import
# that adds a name to an existing import line overlaps it without matching it (ambiguous_binding). Every statement
# CURRENT PROGRAM already has keeps its place, and the BUNDLE's own order must be the merged order.
MERGE_RULE = ('BUNDLE merge rule: each def, class, import or assignment replaces the one CURRENT PROGRAM statement '
              'that binds exactly the same names, a __main__ guard replaces the existing guard, and all other code '
              'is kept. A statement CURRENT PROGRAM already has keeps its place there, changed or not, so list such '
              'statements in CURRENT PROGRAM order. A new import goes after the leading imports; a new def or class '
              'goes before the first top-level statement that makes a call or the guard, or earlier, just before '
              'the next statement of the BUNDLE that CURRENT PROGRAM already has; a new assignment goes before that '
              'first call or guard. The BUNDLE must list its statements in this merged order. A BUNDLE cannot merge '
              'a statement that shares only some of its names with a CURRENT PROGRAM statement (to add an import '
              'name, add an import line of its own for it), a name bound more than once, a second entry (a bare '
              'top-level call beside a __main__ guard), or a new top-level statement that binds no name, such as a '
              'bare call or a tuple assignment like A, B = 1, 2, while CURRENT PROGRAM already has one (otherwise it '
              'goes at the end): put such work in a function called from the existing entry. BUNDLE can change '
              'signatures and constants '
              'together with their callers and related helpers.')
LATER_BUNDLES = ('If the work is too long for one reply, deliver a runnable core now and add the remaining helper '
                 'functions in later BUNDLE replies.')
GUIDANCE = '''There is no function-count or statement-count limit. Do not create extra stub-only roundtrips. Supply an
entrypoint; use a complete __main__ guard for new top-level work. Preserve every required deliverable, and
avoid repeating simulations that can share draws or intermediate calculations. Never lower task requirements.
Make this step runnable with a small numerical check derived from an explicit task formula, identity,
or independent method. Print observed and expected values with a residual/tolerance; assert the relation.
Do not assert an answer against itself. Settle the current decision using execution evidence before
deriving later algorithms. Preserve useful functions and checks; no unimplemented stubs are needed for
later milestones. Intermediate checks may leave deliverables missing; do not create dummy outputs or
claim completion. Before final delivery, run the full required sample size, precision and checks.
'''
# complete_first (audit 2.7): the GUIDANCE sentences that ask for one step at a time, left out of every code prompt
# with the switch on (the second keeps its "no dummy outputs"). Each is found exactly once in GUIDANCE (a test checks).
STEP_GUIDANCE = (('Settle the current decision using execution evidence before\nderiving later algorithms. ', ''),
                 ('Intermediate checks may leave deliverables missing; do not create dummy outputs or\n'
                  'claim completion.', 'Do not create dummy outputs.'))
GUIDANCE_COMPLETE = GUIDANCE
for _old, _new in STEP_GUIDANCE:
    GUIDANCE_COMPLETE = GUIDANCE_COMPLETE.replace(_old, _new, 1)
# task_fidelity (switch, 10-28; from the R3/r3f failure analysis: 31 of 77 failed units used a different formula or
# method variant than the TASK states, many crash loops came from asserts on numbers the program invented, and
# milestone programs stopped early). The contract also lists every explicit TASK requirement; the code prompt keeps
# asserts to TASK- or data-given relations and states the fidelity rules. (old, new) pairs, each found once.
FIDELITY_PLAN = (('ACTION: Produce a short implementation contract, at most 250 words, no code.',
                  'ACTION: Produce a short implementation contract, at most 400 words, no code.'),
                 (' Do not repeat the task.',
                  ' Then list, one short line each, every explicit TASK requirement the outputs must meet: each output '
                  'file with its exact column or key names and order, each formula or method exactly as the TASK '
                  'states it (quote it when given), units, windows, date and boundary rules, rounding, sorting, random '
                  'seeds, and any validation or flag fields. Keep TASK wording; never substitute a textbook variant. '
                  'For every standard formula, estimator or test the TASK names (option and bond pricing, GARCH, HMM, '
                  'copula and tail fits, VaR/ES and their backtests, curve bootstraps, regressions), name the matching '
                  'documented qfkit function in APIS instead of re-deriving it, and note where its conventions must be '
                  'adapted to the TASK. Do not repeat the task otherwise.'))
FIDELITY_GUIDANCE = (('Print observed and expected values with a residual/tolerance; assert the relation.',
                      'Print observed and expected values with a residual/tolerance. Assert only relations the TASK or '
                      'the input data state; never assert a number you chose yourself, and write every deliverable '
                      'before any check that could stop the run.'),)
FIDELITY_ACTION = ('ACTION: Fix every failed check in ACTUAL FEEDBACK and finish every unfinished TASK step now, so '
                   'the next run writes all required deliverables with real computed values. Use REPLACE for the first '
                   'runnable file, then BUNDLE complete related definitions where possible.\n')
FIDELITY = ('TASK FIDELITY: implement every formula, convention, unit, window, date rule, rounding and output name '
            'exactly as the TASK states it, even where a textbook variant differs; write the exact file names and '
            'column/key names in the stated order; a date-only end bound includes that whole day; vectorise loops '
            'over observations and paths with numpy/pandas; keep comments short (no reasoning in comments).\n')


def _complete_names(program, pending):
    """Top-level names CURRENT PROGRAM already binds completely, in order: its functions, classes and module-level
    assignments to plain names, the statements a BUNDLE replaces by name (program_version._bindings; a tuple
    assignment is keyed by no name), without imports and without a name PENDING still owes (incomplete:<name>,
    stub:<name>)."""
    try:
        body = ast.parse(program).body
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return []
    owed = {item.split(':', 1)[1] for item in pending if isinstance(item, str) and ':' in item}
    names = []
    for node in body:
        if isinstance(node, (ast.Import, ast.ImportFrom)) or isinstance(node, ast.AnnAssign) and node.value is None:
            continue
        names.extend(_bindings(node))
    return [name for name in dict.fromkeys(names) if name not in owed]


def first_feedback(version):
    """The free loop's first ACTUAL FEEDBACK (S0-16 B1): FIRST_FEEDBACK and the first line its reply must start
    with, `version` being the empty program's. Offline readers that build the first code prompt (the static sweep)
    use it too, so they measure the prompt that is sent."""
    return FIRST_FEEDBACK + ' First line must be: REPLACE ' + version


# complete_first (split from S2-06's final_delivery, audit 2.7): the first code ask with the switch on (an empty
# program: REPLACE, see code_prompt).
COMPLETE_FIRST = ('ACTION: Write the complete first version now: one runnable file that carries out every '
                  'milestone of the IMPLEMENTATION CONTRACT and writes every required deliverable, with its checks, '
                  'not only the first milestone. Use REPLACE for this first file, then BUNDLE complete related '
                  'definitions for later fixes.\n')


def _ask(program, pending, feedback, *, recovering=False, replacement_required=False):
    """The ask code_prompt makes: 'required', 'module', 'first', 'recover' or 'default'."""
    # Asked once per version: after a reply that left program and pending unchanged, recovery asks on.
    module = 'incomplete:module' in pending and STALL.split('{', 1)[0] not in feedback
    return ('required' if replacement_required else 'module' if module else 'first' if not program
            else 'recover' if recovering else 'default')


def _final_note(why, missing):
    """The FINAL DELIVERY sentence (S2-06): why the end is near (a FINAL_WHY key) and the required deliverables
    still missing (the contract's file names, none when it names no file)."""
    names = [name for name in missing if isinstance(name, str) and name.isprintable()]
    shown = ', '.join(names[:FINAL_NAMES_SHOWN]) + (f' and {len(names) - FINAL_NAMES_SHOWN} more'
                                                    if len(names) > FINAL_NAMES_SHOWN else '')
    return ('FINAL DELIVERY: ' + FINAL_WHY[why] + ' and required deliverables are still missing'
            + (' (' + shown + ')' if names else '') + '. Make the next run write every required deliverable with '
            "the TASK's methods: finish the unfinished calculation and write each output file from the results the "
            'program computes, before any final assertion that could stop the run. Never write dummy or '
            'placeholder values. BUNDLE or REPLACE against CURRENT VERSION.')


def code_prompt(context, contract, program, version, feedback, pending, remaining, remaining_seconds=None,
                *, recovering=False, replacement_required=False, continue_room=False, complete_first=False,
                final=None, library=(), reviewing=None, checked=()):
    """The free-mode code request. One branch chooses the ACTION and its OUTPUT FORMAT together (S0-16), so the
    format names the exact first line(s), with the real version, that the ACTION allows: REPLACE only for a forced
    replacement, a cut module or an empty program (nothing retained to recover, whatever `recovering` says);
    BUNDLE, or REPLACE only for a complete file, for recovery while PENDING owes work, where the complete
    top-level names are listed as present (resend one only to change it); both for the default ask and for
    recovery with nothing pending. A REPLACE on a CURRENT PROGRAM over PROGRAM_BUDGET_CHARS gives its size and
    asks for a structurally smaller file instead of a reply well under the output limit; BUNDLE asks state the
    merge rule once. `continue_room`: a cut reply to this request can still be continued (solve:
    _continuation_room), the only case in which the prompt says that a cut REPLACE is continued. The ACTION line
    stays last. The complete_first switch: `complete_first` asks for a complete first version on an empty program
    (COMPLETE_FIRST), without the "later BUNDLE replies" offer, and every ask then gets GUIDANCE_COMPLETE (no
    one-step-at-a-time sentences). The final_delivery switch (S2-06): `final` = (FINAL_WHY key, missing
    deliverable names) makes the default ask FINAL DELIVERY and adds that sentence to the recovery ask (whose own
    sentence offline readers look for), with no "later BUNDLE replies" offer; the REPLACE-only asks are unchanged.
    `library` (switch lib_expose, _exposed): functions CONFIRMED REQUESTED APIs documents too while the program is empty
    (the unit's first code prompts), whether the contract names them or not. `checked` (switch lib_review): the
    functions a LIBRARY CROSS-CHECK or LIBRARY HINT in ACTUAL FEEDBACK names, which CONFIRMED REQUESTED APIs then
    documents first; without either in `feedback`, they are not added.
    `reviewing` (loop_fix: whether solve can still keep the reviewed run) offers KEEP; None reads the OUTPUT REVIEW
    mark in `feedback`."""
    seconds = (round(max(0, remaining_seconds), 1)
               if type(remaining_seconds) in (int, float) and math.isfinite(remaining_seconds) else 'unknown')
    # output_review: the ask that answers an OUTPUT REVIEW hold keeps the run with a reply without code
    reviewing = OUTPUT_REVIEW_MARK in feedback if reviewing is None else reviewing
    large = len(program) > PROGRAM_BUDGET_CHARS
    size = (f'CURRENT PROGRAM is {len(program):,} characters, too close to the reply limit of about 10,000 '
            'characters to resend whole. ' if large else '')
    smaller = SMALLER + (' ' + CUT_REPLACE if continue_room else '')
    ask = _ask(program, pending, feedback, recovering=recovering, replacement_required=replacement_required)
    deliver = final is not None and ask in FINAL_ASKS
    if ask == 'required':
        action = ('ACTION: REPLACE the complete file against CURRENT VERSION. The previous BUNDLE could not be '
                  'merged: it changes a name that is bound more than once, or top-level work cut from CURRENT '
                  'PROGRAM is still missing (see ACTUAL FEEDBACK). Bind each name once in one complete program, '
                  'preserving already implemented work and every required deliverable.\n')
    elif ask == 'module':
        action = ('ACTION: REPLACE the complete file against CURRENT VERSION. A reply was cut inside top-level '
                  'statements (PENDING incomplete:module, see ACTUAL FEEDBACK), so that work and everything '
                  'after it is missing. Keep the implemented functions and every required deliverable; put the '
                  'top-level work in functions called from one complete __main__ guard, and keep the file compact.\n')
    elif ask == 'recover':
        action = ('ACTION: BUNDLE complete related definitions for the missing or unfinished work against '
                  'CURRENT VERSION. An accepted reply reached the output limit. Keep the retained program; '
                  'do not spend this reply repeating unchanged imports, constants or functions. Complete named '
                  'PENDING bindings and their related helpers/callers. When PENDING is empty, use the actual '
                  'execution and missing-output evidence: syntactically complete code can still be unfinished. '
                  'Include a complete entrypoint to perform the remaining work. Send whole functions/groups, '
                  'never a literal tail. If the fix truly requires changing ambiguous top-level side effects '
                  'or definition ordering, '
                  + ('a REPLACE is still allowed only as a structurally smaller complete file (see OUTPUT FORMAT).\n'
                     if large else 'REPLACE with a compact complete file is still allowed.\n'))
    elif ask == 'first' and complete_first:
        action = COMPLETE_FIRST
    elif reviewing:
        action = REVIEW_ACTION
    elif _feature('task_fidelity'):
        action = FIDELITY_ACTION
    else:
        action = ('ACTION: Implement ONE executable milestone now. Resolve an actual failed check first; '
                  'otherwise extend the working calculation toward the missing deliverables. Use REPLACE '
                  'for the first runnable file, then BUNDLE complete related definitions where possible.\n')
    if deliver:
        note = _final_note(*final)
        action = action[:-1] + ' ' + note + '\n' if ask == 'recover' else 'ACTION: ' + note + '\n'
    # A REPLACE that is asked for or allowed on a program too long to resend: its size, a structurally smaller file.
    replace_large = 'a REPLACE must be a structurally smaller complete file: ' + smaller
    if ask in ('required', 'module', 'first'):
        output = ('First line exactly: REPLACE ' + version + '\nThen one ```python fence holding the complete file. '
                  'No new plan.\n' + (size + 'Make the new file structurally smaller: ' + smaller
                                      if large else FIT + ('' if program or ask == 'first' and complete_first
                                                           else ' ' + LATER_BUNDLES)))
    elif ask == 'recover' and pending:
        names = _complete_names(program, pending)
        output = ('First line exactly: BUNDLE ' + version + ' for the code still owed, or REPLACE ' + version
                  + ' only for a complete file; then one ```python fence. No new plan. A BUNDLE holds the PENDING '
                  'bindings, the helpers and callers they need added or changed, and a complete __main__ guard. '
                  + ('CURRENT PROGRAM already has its imports and these complete top-level names: '
                     + ', '.join(names) + '. Resend one of them only to change it.' if names else
                     'CURRENT PROGRAM already has its imports; resend one only to change it.')
                  + '\n' + MERGE_RULE + '\n'
                  + (size + 'Keep this BUNDLE well under the 4,000-token output limit; ' + replace_large
                     if large else FIT))
    else:
        output = (('If every TASK requirement is met: reply KEEP, with no code. Otherwise: ' if reviewing else '')
                  + 'First line exactly: REPLACE ' + version + ' for a complete file, or BUNDLE ' + version
                  + ' for complete related top-level definitions, imports or bindings; then one ```python fence. '
                  'No new plan.\n' + MERGE_RULE + '\n'
                  + (size + 'Send a BUNDLE of the changed definitions, well under the 4,000-token output limit; '
                     + replace_large if large else FIT + ('' if deliver or reviewing else ' ' + LATER_BUNDLES)))
    return (context + '\nIMPLEMENTATION CONTRACT:\n' + contract
            + '\nCONFIRMED REQUESTED APIs:\n' + api_context(
                contract + '\n' + program, () if program else library,
                checked if checked and (LIB_CROSS_CHECK_MARK in feedback or LIB_REPAIR_HINT_MARK in feedback) else ())
            + '\nCURRENT VERSION: ' + version + '\nCURRENT PROGRAM:\n```python\n' + program + '\n```\n'
            + 'PENDING: ' + ', '.join(pending) + '\nACTUAL FEEDBACK:\n' + feedback
            + f'\nREMAINING REQUESTS: {remaining}\n'
            + f'REMAINING SECONDS: {seconds}\n'
            + 'OUTPUT FORMAT: ' + output + '\n' + _guidance(complete_first) + action)


def _guidance(complete_first):
    """The code prompt's GUIDANCE (GUIDANCE_COMPLETE under complete_first); task_fidelity rewrites its assert sentence
    and adds FIDELITY; lib_expose adds LIBRARY_FIRST after it."""
    text = GUIDANCE_COMPLETE if complete_first else GUIDANCE
    if _feature('task_fidelity'):
        for old, new in FIDELITY_GUIDANCE:
            text = text.replace(old, new, 1)
        text += FIDELITY
    if _feature('lib_expose'):  # next to TASK FIDELITY when that is on
        text += LIBRARY_FIRST
    return text


def step_prompt(context, state, feedback, status, missing, remaining, seconds, *, replacement_required=False,
                action=None, base_version=None, repair_target=None, reference_code=""):
    """Select the action from observed execution; the model supplies the mathematics."""
    action = action or _direct_action(state, status, replacement_required=replacement_required)
    base_version = state.version if base_version is None else base_version
    if replacement_required:
        step = ('REPLACE the complete runnable file against CURRENT VERSION to resolve the rejected merge. '
                'Preserve existing calculations and required deliverables; repair the actual failed check '
                'or current calculation without adding unrelated later work.')
    elif state.program and state.pending:
        step = ('Complete the pending code for the current calculation, including its callers: '
                + ', '.join(state.pending) + '. Do not start a later calculation.')
    elif status.get('thin_executions', 0) and status.get('exit_code') != 0:
        step = 'Repair the actual runtime exception or failed numerical check below. Do not add later work.'
    elif status.get('thin_date_guard_blocking'):
        step = 'Repair the unresolved date parsing evidence below in the complete program. Do not add later work.'
    elif not status.get('thin_executions', 0):
        step = ('Implement ONE smallest prerequisite calculation from the task and an independent small '
                'numerical check. Read actual inputs if this calculation needs them. End this reply once '
                'that calculation can run. Do not implement the other algorithms or full delivery yet.')
    else:
        step = ('Extend the working program by ONE next runnable calculation, chosen from the TASK\'s '
                'mathematical and data dependencies using CURRENT PROGRAM and ACTUAL FEEDBACK. '
                'Check that addition using execution. Leave unrelated later calculations for later replies.')
    operation = ('Supply one complete runnable file, preserving needed existing calculations and deliverables.'
                 if action == 'replace' else
                 'Supply complete related definitions and callers for this step, with a complete __main__ '
                 'guard when changing top-level work. Unchanged code is retained; do not repeat the full file.')
    output = ('Return only raw Python code. One Python code fence enclosing the entire reply is also accepted. '
              'No action or version header, explanation, separate plan or additional code block.')
    if action == 'append_step':
        operation = ('Supply only the NEW executable code for the current calculation. The controller '
                     'keeps CURRENT PROGRAM byte-for-byte and appends this block after it. Do not repeat '
                     'the old script, its existing imports or completed calculations. New imports needed '
                     'by this calculation belong in the new block. Reuse names that the existing '
                     'program makes available at module scope; function locals are not shared. Include '
                     'the calls needed to run this addition and its independent check. Do not return '
                     'only declarations or stubs. Do not move or rewrite existing entrypoints. All '
                     'truncated additions are rejected in full; make this one calculation fit a complete '
                     'reply. The combined program runs from the beginning in a fresh worker, so earlier '
                     'outputs and calculations are regenerated. Use the controller output directory '
                     'specified above, not an output path assumed from the original task.')
    if repair_target is not None:
        action = 'repair_unit'
        step = ('Repair only the controller-selected code unit below. The reported failure location is an '
                'untrusted diagnostic hint, not proof of the root cause. Distinguish TASK requirements, '
                'facts derived from the input, and expectations invented by the program. Correct the actual '
                'calculation or unsupported expectation; do not remove a required check to suppress failure.')
        operation = (f'Replace only lines {repair_target.start}-{repair_target.end} in CURRENT VERSION. '
                     f'Return exactly one complete {repair_target.kind} unit'
                     + (f' named {repair_target.name}' if repair_target.name else '')
                     + '. All source outside this unit is retained byte-for-byte. Do not return the full '
                     'program, later work, action, location or version fields. Truncated patches are rejected '
                     'in full. The complete composed program will run again in a fresh worker.\n'
                     'SELECTED UNIT:\n```python\n' + repair_target.source + '```')
    retained = (
        'The controller already retains and reruns all CURRENT PROGRAM code. Your reply is only the new '
        'block after that code; do not reproduce the cumulative file. Preserve the specified random-number '
        'contract in the new calculation. '
        if action == 'append_step' else
        'Preserve earlier calculations, needed entrypoint calls and the specified random-number contract. '
        'Keep the accumulated file runnable. ')
    return (context + '\nCURRENT STEP (only action for this reply):\n' + step
            + '\nCONTROLLER OPERATION: ' + action.upper() + '\n' + operation
            + '\nCONFIRMED APIs ALREADY USED:\n' + api_context(state.program)
            + '\n' + api_returns.context(context, state.program, reference_code)
            + '\nCURRENT VERSION: ' + base_version + '\nCURRENT PROGRAM:\n```python\n'
            + state.program + '\n```\nACTUAL FEEDBACK:\n' + feedback
            + '\nUNFINISHED DELIVERABLES (final checklist, not execution order):\n'
            + json.dumps(sorted(missing), ensure_ascii=False)
            + '\nRUN SEMANTICS: Each execution starts a new process and clears the output directory. '
              'Only source code is retained; no previous Python variables or output files survive. '
            + retained + 'Intermediate steps can leave outputs missing; never write '
              'dummy outputs. Final acceptance still requires all original outputs, samples and methods. '
              'Print observed and independently expected values with a residual/tolerance and assert the '
              'relation for the current calculation. Check the documented meaning and shape of each API '
              'return used here, including whether it is a summary or per-observation data. A quantity '
              'compared with itself is not a check.\n'
            + f'REMAINING REQUESTS: {remaining}\nREMAINING SECONDS: {max(0, round(seconds, 1))}\n'
            + 'OUTPUT FORMAT: ' + output)


def _snapshot(out):
    """One bounded in-memory output version. Refuse links and oversized trees. An output that cannot be read (S3-03b
    OS-6: a file the run made unreadable, or one gone during capture) is refused the same way, as a local_limit
    HouseError, so a run that leaves one never ends solve with a bare OSError past the R6 restore."""
    files, total = {}, 0
    try:
        for path in sorted(out.rglob('*')):
            info = path.lstat()
            if stat.S_ISDIR(info.st_mode):
                continue
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise HouseError('Output contains links or special files', kind='local_limit',
                                 code='thin_output_special')
            size = info.st_size
            total += size
            if total > OUTPUT_CAP:
                raise HouseError('Output size limit exceeded', kind='local_limit', code='output_limit')
            data = path.read_bytes()
            if len(data) != size:
                raise HouseError('Output changed during capture', kind='local_limit', code='thin_output_changed')
            files[path.relative_to(out).as_posix()] = data
    except FileNotFoundError as exc:
        raise HouseError('Output changed during capture', kind='local_limit', code='thin_output_changed') from exc
    except OSError as exc:
        raise HouseError('Output could not be read', kind='local_limit', code='thin_output_special') from exc
    return files


# S3-03b OS-6: the running solve's own R6 restore (solve registers it once it can restore a run), called before an
# exception ends solve. One solve runs at a time in a process.
_RECOVER = []


def _restores_best(run):
    """solve, with any exception that ends it (in the loop body, the last-chance run or the final checks such as the
    last _snapshot) preceded by the R6 restore of the best run's outputs, measured status and source, so a later,
    worse run never stays on disk. The exception still ends solve as before; a failed restore never replaces it."""
    from functools import wraps

    @wraps(run)
    def solve(*args, **kwargs):
        _RECOVER.clear()
        try:
            return run(*args, **kwargs)
        except Exception:
            for recover in list(_RECOVER):
                try:
                    recover()
                except Exception:
                    pass
            raise
        finally:
            _RECOVER.clear()
    return solve


def _manifest(files):
    return {name: {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
            for name, data in files.items()}


def _execution_evidence(status, duration):
    """Only actual bounded resource numbers/enums, never inferred OOM from -9."""
    raw = status.get('execution_resource')
    raw = raw if isinstance(raw, dict) else {}
    event = raw.get('memory_event')
    resource = {'memory_event': event if event in resource_control.MEMORY_EVENTS else 'unknown'}
    resource.update({key: raw[key] for key in resource_control.MEMORY_FIELDS
                     if type(raw.get(key)) is int and 0 <= raw[key] < resource_control.MAX_NUMBER})
    exit_code = status.get('exit_code')
    return {'exit_code': exit_code if type(exit_code) is int and -(2**31) <= exit_code < 2**31 else None,
            'duration_sec': round(max(0, duration), 3)
            if type(duration) in (int, float) and math.isfinite(duration) else None,
            'execution_resource': resource}


# Every value solve writes to status['thin_stop']. The image has no scripts/public_eval.py, whose
# ENUMS['thin_stop'] exports exactly these; a test keeps the two equal. The last three: a run whose outputs could
# not be captured (_snapshot) ended the loop after an earlier run was restored, or ended solve.
# progress_time (switch, 10-28): the unit's time follows its own card instead of one share of the stage clock. The image
# budget (UNIT_BUDGET_SEC under the card, solve's deadline) becomes a SOFT deadline; the HARD one is the card's own
# budget (baseline.card_budget without the share, at most PROGRESS_HARD_CAP). Past the soft deadline the free loop starts
# another code request only while it is still improving: its best run got better within PROGRESS_WINDOW seconds. A unit
# that is stuck stops there ('soft_budget', the best run stands), so the shared 12-hour clock goes to units that move.
# The OUTPUT REVIEW and the second drafts keep to the soft deadline.
# A run keeps to the soft deadline's room, or PROGRESS_RUN_CAP seconds once that room is smaller: never the hard
# deadline's whole room, which one slow program would spend (and the stage clock with it). A unit that has not run a
# program by PROGRESS_WINDOW seconds past the soft deadline stops too.
# clock_fix (switch, 10-28): that window is read before a continuation request and before a run too, not only before a
# code request, so a stuck unit (one with a best run) spends no further iteration past it; REMAINING SECONDS shows the
# room a run gets (the cap above), not the hard deadline's; a parquet output is decoded only within the bound its
# metadata shows (stub_check.parquet_bound).
PROGRESS_HARD_CAP = 1500.0
PROGRESS_WINDOW = 240.0
PROGRESS_RUN_CAP = 240.0


def _progress_deadlines(task_dir, deadline):
    """(soft, hard) deadlines for progress_time: `deadline` stays the soft one; the hard one adds the card's budget beyond
    the image budget (none with an explicit harness budget, T1_TIMEOUT_SEC, or without a readable card)."""
    from . import baseline
    if os.environ.get('T1_TIMEOUT_SEC') or baseline.card_timeout(task_dir) is None:
        return deadline, deadline
    unit = baseline._seconds(os.environ.get('UNIT_BUDGET_SEC')) or baseline.STAGE_CAP
    soft_budget = baseline.card_budget(task_dir, unit, cap=unit)
    hard_budget = baseline.card_budget(task_dir, PROGRESS_HARD_CAP, cap=PROGRESS_HARD_CAP)
    return deadline, deadline + max(0.0, hard_budget - soft_budget)


THIN_STOPS = ('internal_checks_passed', 'no_progress', 'request_limit', 'execution_setup_error',
              'generation_budget', 'http_error', 'transport_failure', 'provider_error', 'response_limit',
              'response_invalid_json', 'response_invalid_shape', 'invalid_configuration',
              'invalid_prompt', 'prompt_limit', 'choice_unsupported', 'choice_dependencies_unsupported',
              'thin_output_special', 'output_limit', 'thin_output_changed', 'soft_budget')
# How the last-chance run of a held version ended (status['thin_preexec_fallback_outcome']); public_eval ENUMS.
FALLBACK_OUTCOMES = ('ran', 'passed', 'setup_error', 'output_rejected')
REQUEST_TIMEOUT_CAP = 180  # seconds one request may take (House's own limit)
# Pre-send time gate. A request is predicted to need GATE_BASE_SEC + expected tokens / measured rate, both
# from this unit's own replies only. There is no prior: until the unit has one measured reply (at least
# RATE_MIN_TOKENS completion tokens in more than RATE_MIN_SEC) no request is held back. The gate holds back
# only requests whose skip cannot cost the unit its only program: continuations (the cut prefix still goes
# to admission) and, once a program has executed, code requests; see _request.
GATE_BASE_SEC = 2
RATE_LATENCY_SEC = 1.5  # time to the first token, left out of a measured rate
RATE_MIN_TOKENS, RATE_MIN_SEC = 500, 3
CONTRACT_TOKENS = 2000  # expected contract reply before one was measured; code expects MAX_TOKENS
TIMEOUT_FAILURES = ('deadline', 'socket_timeout')  # transport failures ended by the timeout we gave


def _predicted_need(client, kind):
    """Seconds the next reply of this kind is predicted to take, or None before a measured reply.

    The rate is the median of completion_tokens / (duration - RATE_LATENCY_SEC) over this unit's
    returned replies of at least RATE_MIN_TOKENS tokens that took more than RATE_MIN_SEC. The expected
    size is the median of this unit's code replies (contract replies for a contract request), else
    MAX_TOKENS (CONTRACT_TOKENS). Continuations expect a code reply's size."""
    from statistics import median
    replies = [record for record in client.diagnostics.get('requests', [])
               if isinstance(record, dict) and record.get('reply_returned') is True
               and type(record.get('completion_tokens')) is int and record['completion_tokens'] >= 0]
    rates = [record['completion_tokens'] / (record['duration_sec'] - RATE_LATENCY_SEC) for record in replies
             if record['completion_tokens'] >= RATE_MIN_TOKENS and type(record.get('duration_sec')) in (int, float)
             and math.isfinite(record['duration_sec']) and record['duration_sec'] > RATE_MIN_SEC]
    if not rates:
        return None
    own = 'thin_contract' if kind == 'thin_contract' else 'thin_code'
    sizes = [record['completion_tokens'] for record in replies if record.get('kind') == own]
    expected = median(sizes) if sizes else CONTRACT_TOKENS if own == 'thin_contract' else MAX_TOKENS
    return GATE_BASE_SEC + expected / median(rates)


def _held_for_time(need, left, reserve):
    """The send-time gate itself: a reply predicted to take `need` seconds (None: not gated) does not fit in the
    `left` seconds before the deadline once `reserve` is kept (at most REQUEST_TIMEOUT_CAP is asked for).
    _request applies it before each send and retry; S0-11's block (_repair_fits) applies it to the next request."""
    return need is not None and left - reserve < min(need, REQUEST_TIMEOUT_CAP)


def _time_gate(exc, gate):
    """Mark a HouseError as a request the time gate held back (code, continue or retry)."""
    exc.time_gate = gate
    return exc


def _count_gate(status, exc):
    """Count a time-gate skip once, wherever the HouseError that carries it is handled."""
    gate = getattr(exc, 'time_gate', None)
    if gate is None or getattr(exc, 'time_gate_counted', False):
        return
    exc.time_gate_counted = True
    _count_skip(status, gate)


def _count_skip(status, gate):
    """One request held back for time: code, continue or retry."""
    if gate == 'code':
        status['thin_time_gate_skips_code'] = status.get('thin_time_gate_skips_code', 0) + 1
    elif gate == 'continue':
        status['thin_time_gate_skips_continue'] = status.get('thin_time_gate_skips_continue', 0) + 1
    else:
        status['thin_time_gate_skips_retry'] = status.get('thin_time_gate_skips_retry', 0) + 1


def record_stop(status, exc):
    """A HouseError that ends solve: count its time-gate skip and name the stop (the first one stands)."""
    _count_gate(status, exc)
    if status.get('thin_stop') is None and exc.code in THIN_STOPS:
        status['thin_stop'] = exc.code


def _named_stops(run):
    """solve, with every HouseError that ends it named in status before it propagates (record_stop),
    including stops before any execution (the contract and first code requests)."""
    from functools import wraps

    @wraps(run)
    def solve(client, task_dir, out, instruction, profile, hints, canaries, deadline, status, replay):
        global _FEATURES
        try:
            return run(client, task_dir, out, instruction, profile, hints, canaries, deadline, status, replay)
        except HouseError as exc:
            record_stop(status, exc)
            raise
        finally:
            _FEATURES = frozenset()  # one unit's switches never reach a later solve or a direct helper call
    return solve


def _cut_by_timeout(client, transports):
    """The last request's transport ended because the timeout it was given ran out."""
    details = client.diagnostics.get('transport_requests')
    return (isinstance(details, list) and len(details) > transports and isinstance(details[-1], dict)
            and details[-1].get('failure_reason') in TIMEOUT_FAILURES)


def _route_answering(client, offset, transports):
    """The route returned a reply to this unit, and no request since then (before the current one) ran out
    the full REQUEST_TIMEOUT_CAP. Only then is a timeout below the cap this unit's own deadline: a route
    that never answered, or that hung at the cap after it last did, is an outage, not the unit's budget.
    Request and transport records are read pairwise, as HouseClient appends one of each per request."""
    records = client.diagnostics.get('requests', [])[:offset]
    details = client.diagnostics.get('transport_requests', [])[:transports]
    for index in range(len(records) - 1, -1, -1):
        record = records[index]
        if not isinstance(record, dict):
            continue
        if record.get('reply_returned') is True:
            return True
        detail = details[index] if len(details) == len(records) else None
        if (type(record.get('timeout_sec')) in (int, float) and record['timeout_sec'] >= REQUEST_TIMEOUT_CAP
                and isinstance(detail, dict) and detail.get('failure_reason') in TIMEOUT_FAILURES):
            return False
    return False


def _transient(client, exc):
    """A failed transport, or 408/409/425/429 and server errors; never a local or client error."""
    if exc.code == 'transport_failure':
        return True
    code = client.diagnostics.get('http_status')
    return (exc.code == 'http_error' and type(code) is int
            and (code in (408, 409, 425, 429) or 500 <= code <= 599 and code not in (501, 505)))


def _request(client, prompt, deadline, replay, kind, thinking, reserve, *, messages=None, gate=None,
             temperature=None):
    """One reply. Transient failures are retried with bounded backoff (Retry-After honoured) while
    the request cap and the time before the reserve allow; the last actual error is kept.
    temperature: None sends the request exactly as before (House's greedy default); otherwise one of house's kind
    constants (temperature_for), passed to chat for every attempt (A2: resample).

    gate: whether the time gate applies; None means by kind, a continuation yes and a contract no.
    solve gates a code request only once a program has executed: holding back a unit's first program
    leaves it certainly empty, while a request that is sent and does not finish ends the same way. A
    version held back before its run (S0-11) runs before solve ends, so while one waits the gate applies.
    Once this unit has a measured reply, a gated request is sent, and retried, only while the time
    before the reserve holds its predicted reply (_predicted_need, at most REQUEST_TIMEOUT_CAP): a
    held-back request raises local_budget/generation_budget, a held-back retry (also one RETRY_FLOOR
    holds back) its actual error, each marked with time_gate. A transport that the timeout we gave
    ended below REQUEST_TIMEOUT_CAP, after the route answered and without a hang at the cap since
    (_route_answering), is the unit's own deadline: it raises local_budget/generation_budget (the
    replay keeps the transport failure). Each request record carries its timeout and the time ledger
    at sending (remaining, reserve, and the predicted need where the gate applied)."""
    from . import baseline
    messages = messages or [{'role': 'user', 'content': prompt}]
    error = None
    if gate is None:
        gate = kind == 'thin_continue'
    need = _predicted_need(client, kind) if gate else None
    for attempt in range(len(RETRY_WAITS) + 1):
        left = deadline - baseline.CLOCK.now()
        if client.attempts >= client.max_requests:
            raise error or HouseError('Request budget spent', kind='local_budget', code='request_limit')
        if left <= reserve + 1:
            raise error or HouseError('Execution reserve reached', kind='local_budget', code='generation_budget')
        if _held_for_time(need, left, reserve):
            if error is not None:
                raise _time_gate(error, 'retry')  # the wait left too little for the retried reply
            if kind in ('thin_code', 'thin_continue'):
                raise _time_gate(HouseError('The predicted reply does not fit before the execution reserve',
                                            kind='local_budget', code='generation_budget'),
                                 'code' if kind == 'thin_code' else 'continue')
        timeout = max(1, min(REQUEST_TIMEOUT_CAP, left-reserve))
        ledger = {'timeout_sec': round(timeout, 1), 'continuation_remaining_sec': round(left, 1),
                  'continuation_reserve_sec': round(reserve, 1)}
        if need is not None:
            ledger['continuation_need_sec'] = round(need, 1)
        offset = len(client.diagnostics.get('requests', []))
        transports = len(client.diagnostics.get('transport_requests', []))
        returned = False
        try:
            try:
                options = {} if temperature is None else {'temperature': temperature}
                reply = client.chat(messages, timeout=timeout, thinking=thinking, max_tokens=MAX_TOKENS, **options)
                returned = True
            finally:
                # Tag only records the actual client created, including failed transports.
                for record in client.diagnostics.get('requests', [])[offset:]:
                    record.update(kind=kind, attempt=attempt, reply_returned=returned, **ledger)
        except HouseError as exc:
            replay.request(client, kind, messages, error=exc.code)
            if (exc.code == 'transport_failure' and timeout < REQUEST_TIMEOUT_CAP
                    and _cut_by_timeout(client, transports) and _route_answering(client, offset, transports)):
                raise HouseError('The reply did not finish before the execution reserve',
                                 kind='local_budget', code='generation_budget') from exc
            if (not _transient(client, exc) or attempt == len(RETRY_WAITS)
                    or client.attempts >= client.max_requests):
                raise
            wait = RETRY_WAITS[attempt]
            hint = client.diagnostics.get('retry_after_sec')
            if type(hint) is int and hint > 0:
                wait = max(wait, hint)
            # Local evaluation only (T1_EVAL_THROTTLE): a throttled wait is credited back, as in legacy,
            # so only its uncredited part comes off the time left.
            credit = (baseline.CLOCK.creditable(wait) if getattr(client, 'throttle', False) is True
                      and exc.code == 'http_error'
                      and client.diagnostics.get('http_status') in baseline.EVAL_PAUSE_STATUSES else 0)
            after = deadline - baseline.CLOCK.now() - (wait - credit)
            if (after < reserve + RETRY_FLOOR  # no request would still fit after the wait
                    or _held_for_time(need, after, reserve)):
                raise exc if need is None else _time_gate(exc, 'retry')  # nor the predicted reply
            began = time.monotonic()
            time.sleep(wait)
            if credit:
                baseline.CLOCK.credit(min(credit, time.monotonic() - began))
            client.diagnostics['backoff_sec'] = client.diagnostics.get('backoff_sec', 0) + wait
            error = exc
            continue
        replay.request(client, kind, messages, reply=reply)
        return reply


def _cut_code(raw):
    """Complete lines of a cut reply that ends inside its one open python fence, else None.

    A reply that is one action line followed by raw code without any fence (the extractor reads such
    text as the code) is continued the same way: a ```python line is inserted after the action line,
    provided its complete code lines parse or end at a cut that length admission accepts, and none of
    them leaves a one-line string open. A second action line inside the text is a repeated reply, not
    one program, and is not continued."""
    if not isinstance(raw, str):
        return None
    text = raw.replace('\r\n', '\n').replace('\r', '\n')
    head = text[:text.rfind('\n') + 1]
    lines = head.split('\n')[:-1]
    # A fence on the cut line too (a reply cut just after its closing fence) is not open code.
    fences = [i for i, line in enumerate(text.split('\n')) if line.strip().startswith('```')]
    if not fences:
        header = re.match(HEADER, head)
        if (header is None or not head[header.end():].strip()
                or re.search(LATE_HEADER, text[header.end():] + '\n', flags=re.M)):
            return None
        code = head[header.end():]
        try:
            _parse(code.rstrip() + '\n')
        except SyntaxError as error:
            # Complete lines hold an open one-line string only when a trailing backslash carries it on;
            # otherwise it is prose with an apostrophe or a broken line, which no continuation repairs.
            if 'unterminated string literal' in error.msg and not code.rstrip().endswith('\\'):
                return None
        except (ValueError, RecursionError, MemoryError, OverflowError):
            return None
        fenced = head[:header.end()] + '```python\n' + code
        return fenced if _plausible(fenced, False) else None
    if len(fences) != 1 or fences[0] >= len(lines) or lines[fences[0]].strip().lower() not in CODE_FENCES:
        return None
    return head if any(line.strip() for line in lines[fences[0]+1:]) else None


def _retained(reply, text):
    """The reply that admission gets when continuation stops before the code is finished.

    An unfenced reply that nothing was stitched onto goes as received: its cut partial line is what
    lets length admission record the cut work as pending. Otherwise the retained complete lines go,
    still marked length."""
    raw = reply.get('content')
    rows = raw.replace('\r\n', '\n').replace('\r', '\n').split('\n') if isinstance(raw, str) else ()
    if text == _cut_code(raw) and not any(row.strip().startswith('```') for row in rows):
        return reply
    return dict(reply, content=text, finish_reason='length')


def _echo_size(previous, current):
    """The number of leading lines of current that repeat the last lines of previous (0 if none)."""
    return next((size for size in range(min(len(previous), len(current), 400), 0, -1)
                 if previous[-size:] == current[:size]), 0)


def _stitch(text, body, notes=None):
    """Append continuation lines to the retained text (complete lines inside one open fence).

    Leading lines that exactly repeat the retained tail are dropped; a continuation that restarts
    the program from its first code line replaces the retained code instead of duplicating it.
    Blank lines are not part of an echo: when the lines do not repeat as they stand, the retained
    text's trailing blank lines and the continuation's leading ones are left out of the comparison,
    and the stitched text keeps the retained blank lines once (an echoed anchor after a blank line
    is not run twice).
    A repeat made only of short lines (under 8 characters, such as `    },`) may also be new code:
    the stitch without it is preferred, and the one with it is kept only when it alone parses, or
    when neither parses and it alone ends at a cut that length admission accepts. A genuine new
    line equal to the retained short line is therefore dropped when both readings parse; `notes`
    (a dict, when given) gets short_echo_dropped=True whenever the result drops a short-only repeat."""
    lines = text.split('\n')[:-1]
    previous, current = [line.rstrip() for line in lines], [line.rstrip() for line in body]
    lead = next((i for i, line in enumerate(current) if line), len(current))  # leading blank lines
    echo = _echo_size(previous, current[lead:])
    if not echo:
        tail = previous[:next((i for i in range(len(previous), 0, -1) if previous[i-1]), 0)]
        echo = _echo_size(tail, current[lead:])
        if echo:
            # The retained text already ends with its blank lines: the continuation's own ones after the
            # echo add only what goes beyond those.
            rest = current[lead+echo:]
            echo += min(len(previous) - len(tail), next((i for i, line in enumerate(rest) if line), len(rest)))
    echo += lead if echo else 0
    dropped = text + ''.join(line + '\n' for line in body[echo:])
    if echo and any(len(line.strip()) >= 8 for line in current[:echo]):
        return dropped
    fence = next(i for i, line in enumerate(lines) if line.strip().startswith('```'))
    first = next((line for line in previous[fence+1:] if line.strip()), '')
    if lead < len(current) and current[lead] == first and len(first.strip()) >= 8:
        lines = lines[:fence+1]
    joined = ''.join(line + '\n' for line in lines + body)
    if echo:
        for done in (True, False):
            for candidate in (dropped, joined):
                if _plausible(candidate, done):
                    if candidate is dropped and candidate != joined and notes is not None:
                        notes['short_echo_dropped'] = True
                    return candidate
    return joined


def _continuation_lines(content, finish):
    """Candidate (new code lines, closed) readings of a continuation reply, preferred first.

    A tagged fence, or a fence with nothing before it, opens the code. Text before a bare fence is
    either code that the fence closes or prose that the fence follows; the caller keeps the first
    reading that parses. A JSON object is never code."""
    try:
        if isinstance(json.loads(content), dict):
            return []
    except (ValueError, RecursionError):
        pass
    lines = content.replace('\r\n', '\n').replace('\r', '\n').split('\n')
    if finish == 'length':
        lines = lines[:-1]  # the cut partial line
    fences = [i for i, line in enumerate(lines) if line.strip().startswith('```')]

    def trimmed(body, closed):
        while body and not body[0].strip():
            body = body[1:]
        return body, closed

    def inside(opening):
        closing = next((i for i in fences if i > opening), None)
        return trimmed(lines[opening+1:closing], closing is not None)
    if not fences:
        return [trimmed(lines, False)]
    first = fences[0]
    if not any(line.strip() for line in lines[:first]) or lines[first].strip().lower() in CODE_FENCES[1:]:
        return [inside(first)]
    return [trimmed(lines[:first], True), inside(first)]


def _plausible(text, done):
    """The code after the opening fence parses; while still cut it may instead end at a cut that
    ProgramVersion's length admission accepts, so the verified prefix stays recoverable."""
    lines = text.split('\n')
    fence = next(i for i, line in enumerate(lines) if line.strip().startswith('```'))
    source = '\n'.join(lines[fence+1:]).rstrip() + '\n'
    try:
        _parse(source)
    except SyntaxError as error:
        return not done and _cutoff(source, error)
    except (ValueError, RecursionError, MemoryError, OverflowError):
        return False
    return True


def _continue(client, prompt, reply, deadline, replay, thinking, reserve, canaries, status, *,
              limit=MAX_CONTINUATIONS, outcome=None, joint=False, copy=None, stop=None):
    """Continue a cut code reply from its last complete line; returns the stitched reply.

    At most `limit` continuations (0 sends none), each one counted request inside the cap and the
    time before the reserve. The result is admitted exactly like any other reply, still marked length
    unless finished. A finished stitch must parse; a continuation that would break the code (prose, an
    envelope, a wrong indentation at the joint) is dropped. Unfinished, admission gets the verified
    complete lines (the cut partial line dropped) for prefix recovery, or, for an unfenced reply that
    nothing was stitched onto, the reply as received (_retained). `outcome` (a dict), when given, gets
    exhausted=True when all `limit` continuations were sent and the stitch is still cut: only the limit ended it.
    loop_fix (solve passes both): `joint` re-indents a continuation whose joint does not parse (_joints) and never takes
    the empty reading after a fence that closes code as the finished program; `copy` (_copy_target) ends the chain as
    soon as the stitched text reproduces that executed program far enough (_copied), as the identical resend it is.
    `stop` (clock_fix), when given, is read before each continuation: true sends none (outcome gets stopped=True)."""
    from . import baseline
    if reply.get('continuable') is False:
        return reply  # already what admission gets with none sent: none can make it admissible (_skip_continuation)
    text = _cut_code(reply.get('content'))
    if text is None:
        return reply
    for _ in range(limit):
        if copy is not None and _copied(text, copy):
            return _copy_reply(reply, copy, status)
        if stop is not None and stop():
            if outcome is not None:
                outcome['stopped'] = True
            break
        if (client.attempts >= client.max_requests
                or deadline - baseline.CLOCK.now() < reserve + CONTINUE_FLOOR):
            if client.attempts < client.max_requests and _predicted_need(client, 'thin_continue') is not None:
                _count_skip(status, 'continue')  # held back for time below the floor: the gate's skip too
            break
        line = next((line for line in reversed(text.split('\n')) if line.strip()), '').rstrip()[:120]
        messages = [{'role': 'user', 'content': prompt},
                    {'role': 'assistant', 'content': baseline.withhold_canaries(text, canaries)},
                    {'role': 'user', 'content': baseline.withhold_canaries(CONTINUE.format(line=line), canaries)}]
        if sum(len(message['content'].encode()) for message in messages) > CONVERSATION_CAP:
            break
        try:
            more = _request(client, None, deadline, replay, 'thin_continue', thinking, reserve, messages=messages)
        except HouseError as exc:
            _count_gate(status, exc)
            break
        content, finish = more.get('content'), more.get('finish_reason')
        if not isinstance(content, str) or not content.strip():
            break
        before = text
        readings = _continuation_lines(content, finish)
        for index, (body, closed) in enumerate(readings):
            done = closed or finish == 'stop'
            if joint and index and not body and _code_like(readings[0][0]):
                continue  # loop_fix: code closed by a bare fence that did not join; the retained prefix is not the end
            for shifted, lines in ((False, body),) + tuple((True, moved) for moved in
                                                         (_joints(before, body) if joint else ())):
                notes = {}
                text = _stitch(before, lines, notes)
                if _plausible(text, done):
                    break
            else:
                continue
            break
        else:
            status['thin_continued_unparsed'] = status.get('thin_continued_unparsed', 0) + 1
            return _retained(reply, before)
        if shifted:
            status['thin_stitch_reindents'] = status.get('thin_stitch_reindents', 0) + 1
        if notes.get('short_echo_dropped'):
            # S0-10 B3's known cost: a genuine line equal to the retained short line would be dropped too.
            status['thin_stitch_short_echo_dropped'] = status.get('thin_stitch_short_echo_dropped', 0) + 1
        if done:
            status['thin_continued_complete'] = status.get('thin_continued_complete', 0) + 1
            return dict(reply, content=text + '```', finish_reason='stop')
        if finish != 'length' or text == before:
            break
    else:
        if limit and outcome is not None:
            outcome['exhausted'] = True
    if copy is not None and _copied(text, copy):
        return _copy_reply(reply, copy, status)
    return _retained(reply, text)


def _structure(program):
    """Top-level function and class names of a program, or its imported names when it has none."""
    try:
        body = ast.parse(program).body
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return None
    names = {node.name for node in body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
    return names or {alias.asname or (alias.name.split('.')[0] if isinstance(node, ast.Import) else alias.name)
                     for node in body if isinstance(node, (ast.Import, ast.ImportFrom))
                     and getattr(node, 'module', None) != '__future__'
                     for alias in node.names if alias.name != '*'}


def _headerless_check(state, raw, finish):
    """(program, note) for a reply without its action line, read by the same extractor as the header path.

    The selected code is admitted only as a full replacement that is ready to run, reads no name it
    never binds and keeps the current program's top-level functions and classes (its imports when it
    has none, and then also every module-level name it binds besides its imports: _rebinds). A fix snippet
    fails these tests; anything else keeps the explicit REPLACE/BUNDLE contract, and the note names the exact
    reason for the feedback ('' only for an empty or non-text reply). A cut reply on an empty program is
    _headerless_cut's (S0-16 B1); on any other program it is not taken without its action line."""
    if not isinstance(raw, str) or not raw.strip():
        return None, ''
    if finish != 'stop':
        return None, ('it was cut at the output limit, and a cut reply is taken without its action line only '
                      'while CURRENT PROGRAM is empty' if finish == 'length' else 'the reply did not finish normally')
    text = strip_reasoning(raw)
    try:
        extraction = extract_code(text, 'stop', expected_files=state.expected_files)
    except FormatError as error:
        return None, error.reason + ': ' + error.detail
    try:
        if isinstance(json.loads(text), dict):
            return None, 'the reply is a JSON object, not code in a ```python block'
    except (ValueError, RecursionError):
        pass
    trial = copy.deepcopy(state)
    result = trial.apply(extraction.source, 'stop', 'replace', state.version)
    if not result.accepted:
        if extraction.shape == 'raw' and result.reason == 'invalid_syntax':
            return None, 'no ```python block was found and the text is not Python'
        return None, trial.feedback.split('; ', 1)[0]
    if not trial.ready:
        return None, 'the code is not a complete program (pending: ' + ', '.join(trial.pending[:5]) + ')'
    if trial.advisories:
        return None, 'it calls undefined names: ' + ', '.join(
            advisory.split(':', 1)[-1] for advisory in trial.advisories[:5])
    names, keep = _module_names(trial.program), (_structure(state.program) if state.program else set())
    if names is None:
        return None, 'its module names cannot be checked (star import)'
    if names[1]:
        return None, 'it reads names it never defines: ' + ', '.join(sorted(names[1])[:5])
    if keep is None or not keep <= names[0]:
        dropped = sorted((keep or set()) - names[0])
        return None, 'it drops current top-level definitions: ' + ', '.join(dropped[:5])
    # On a script without functions or classes `keep` is its imports, which a re-importing snippet keeps too: the
    # reply must also bind the script's own module-level names again (S0-15's A3b guard). A script that binds
    # none besides its imports gives no such evidence, and keeps the import test alone.
    work = _work(state.program) if state.program and not _defines(state.program) else None
    if work and not _rebinds(state.program, trial.program):
        return None, ('it leaves module-level names of CURRENT PROGRAM unbound: '
                      + ', '.join(sorted(work - names[0])[:5]))
    return extraction.source, ''


def _defines(program):
    """Whether a program has a top-level function or class (False when it does not parse)."""
    try:
        return any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                   for node in ast.parse(program).body)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return False


def _headerless_cut(state, raw):
    """(payload, note) for a reply cut at the output limit without its action line while CURRENT PROGRAM is empty
    (S0-16 B1): the reply is the first program, so it is read as the header path reads a REPLACE of the empty
    version, with the prefix recovery of a cut reply; payload is the reply without reasoning. When ProgramVersion
    rejects it, that rejection (invalid_syntax, no_code_block ...) is the verdict, as with a header. A kept prefix
    must hold code and read no module name it never binds, apart from names that the cut leaves owed
    (PENDING incomplete:<name>, stub:<name>); otherwise (None, note), the missing-header rejection with its reason."""
    payload = strip_reasoning(raw)
    trial = copy.deepcopy(state)
    if not trial.apply(payload, 'length', 'replace', state.version).accepted:
        return payload, ''
    if not trial.program.strip():
        return None, 'no complete line of code came before the cut'
    names = _module_names(trial.program)
    if names is None:
        return None, 'its module names cannot be checked (star import)'
    owed = {item.split(':', 1)[1] for item in trial.pending if item.startswith(('incomplete:', 'stub:'))}
    if names[1] - owed:
        return None, 'it reads names it never defines: ' + ', '.join(sorted(names[1] - owed)[:5])
    return payload, ''


def _headerless(state, raw, finish):
    """The program of a headerless reply admitted as a full replacement, else None."""
    return _headerless_check(state, raw, finish)[0]


# S2-10 (always on): a run whose outputs hold an input canary GUID is flagged (the flag never names the GUID); when a
# contract deliverable holds it (or the contract names no files) the run ranks after every clean run (_rank,
# _canary_demotes), and the end restores the best run; any file that still holds one is deleted.
CANARY_FLAG = ('{name}: holds benchmark canary metadata copied from the inputs; an output never copies input '
               'metadata, banners or canary identifiers')


def _canary_needles(canaries):
    return [str(guid).lower().encode() for guid in canaries if str(guid)]


def _canary_outputs(files, canaries):
    """S2-10: the names, sorted, of the captured output files (`files`, as _snapshot returns them) whose bytes hold
    an input canary GUID, matched as baseline.output_canary_files matches it (bytes, case-insensitive)."""
    needles = _canary_needles(canaries)
    if not needles:
        return []
    found = []
    for name, data in files.items():
        lowered = data.lower()
        if any(needle in lowered for needle in needles):
            found.append(name)
    return sorted(found)


def _canary_demotes(expected, tainted):
    """S2-10 (review 10-22): whether a run whose outputs `tainted` (_canary_outputs) hold an input canary ranks after
    every clean run. Only when a contract deliverable holds it, or the contract names no files: a GUID only in side
    files (logs, debug dumps) leaves the run's rank, and the end deletes those files (_remove_canary_outputs)."""
    if not tainted:
        return False
    if not expected['files']:
        return True
    named = {Path(name).as_posix() for name in expected['files']}
    return any(name in named for name in tainted)


def _remove_canary_outputs(out, canaries):
    """S2-10: delete each regular output file whose bytes hold an input canary GUID, read as
    baseline.output_canary_files reads it (no links followed; a file it cannot read is left, and that function
    counts it as unscanned). Returns how many files were deleted."""
    needles, removed = _canary_needles(canaries), 0
    if not needles:
        return 0
    for path in sorted(out.rglob('*')):
        if path.is_symlink() or not path.is_file():
            continue
        try:
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            with os.fdopen(fd, 'rb') as stream:
                data = stream.read().lower()
            if any(needle in data for needle in needles):
                path.unlink()
                removed += 1
        except OSError:
            continue
    return removed


# S2-05 (T1_THIN_FEATURES false_green). An all-green run (every visible output check passed) is held back once per
# unit, while a repair still fits, when its outputs or its program show these advisory findings: a contract file that
# is empty (0 bytes or whitespace only, which counts as missing), baseline.advisory_flags' degenerate outputs,
# audit.nonfinite_flags, and the program's own comments or strings calling it unfinished (RPL-2: these phrases only;
# a bare "dummy" or "placeholder" is not one). A later all-green run then stands with fewer findings, or with as many
# when none of them is new (_fg_key: the same findings, numbers aside, so a repair that keeps a right value and fixes
# something else stands). The first reply after the hold that leaves the program unchanged (no code, or the same code)
# keeps the held run as final (thin_false_green_kept).
FALSE_GREEN_BLOCKS = 1
UNFINISHED_PHRASES = (('placeholder implementation', re.compile(r'\bplaceholder\s+implementation\b', re.I)),
                      ('dummy values', re.compile(r'\bdummy\s+values\b', re.I)),
                      ('dummy data', re.compile(r'\bdummy\s+data\b', re.I)),
                      ('TODO', re.compile(r'\bTODO\b')),  # case-sensitive
                      ('not implemented', re.compile(r'\bnot\s+implemented\b', re.I)))
FALSE_GREEN = ('\nNot accepted as final (asked once per unit): every visible output check passed, but these findings '
               'on the outputs or the program suggest an unfinished or degenerate result:\n{findings}\nRecheck each '
               'against the TASK. Fix what is wrong and resend against CURRENT VERSION; a value that is right as the '
               'TASK defines it stays as it is. If every finding is right as the TASK defines it, reply without code '
               '(or resend CURRENT VERSION unchanged): this run then stands as final.')
# output_review (switch, 10-28): the first run per unit that passes the internal checks is held once, while a repair fits,
# with the host's OUTPUT REPORT of its files (output_report.build) and the generic issues found there; the model checks
# every TASK requirement against it and keeps the run (no code, or CURRENT VERSION unchanged) or resends a fix.
REVIEW_ACTION = ('ACTION: Check this run against every TASK requirement using the OUTPUT REPORT in ACTUAL FEEDBACK. If '
                 'every requirement is met, reply KEEP with no code. Otherwise send the fix against CURRENT VERSION: a '
                 'BUNDLE of the changed definitions, or REPLACE with the complete corrected file.\n')
OUTPUT_REVIEW_MARK = 'Not yet accepted as final (asked once per unit): check this run against the TASK first.'
# outputs_fix (10-28): the visible flag of a clean run whose required outputs hold NaN/inf numbers or numbers past
# output_report.HUGE (output_report.blocking names the file, the column or path and the count)
OUTPUTS_BLOCKED = ('{finding}: not accepted as final; find the step of the computation that produces these values and '
                   'fix it (never overwrite them with placeholder numbers)')
OUTPUT_REVIEW_ROOM = 150  # seconds left at least: the review reply at House's 60 tok/s, a rerun and its repair
OUTPUT_REVIEW = ('\nNot yet accepted as final (asked once per unit): check this run against the TASK first.\n{report}\n'
                 '{violations}Go through the TASK requirement by requirement: every output file, exact column/key names '
                 'and order, row coverage, each formula exactly as written, units, windows, date rules, rounding, '
                 'sorting, flags and validation fields. Compare each with the OUTPUT REPORT. Fix every requirement the '
                 'outputs do not meet and resend against CURRENT VERSION. If every requirement is met, reply without '
                 'code (or resend CURRENT VERSION unchanged): this run then stands as final.')


def _unfinished_phrases(program):
    """S2-05 RPL-2: the UNFINISHED_PHRASES found in the program's comments and string literals (f-string text
    included), each named once in the table's order. Code and names are never read: NotImplementedError,
    dummy_values or a TODO() call do not count."""
    import io
    import tokenize
    kinds = {tokenize.COMMENT, tokenize.STRING} | {getattr(tokenize, name) for name in
                                                   ('FSTRING_MIDDLE', 'TSTRING_MIDDLE') if hasattr(tokenize, name)}
    texts = []
    try:
        for token in tokenize.generate_tokens(io.StringIO(program).readline):
            if token.type in kinds:
                texts.append(token.string)
    except (tokenize.TokenError, SyntaxError, ValueError):
        pass  # what was read before the error still counts
    return [label for label, pattern in UNFINISHED_PHRASES if any(pattern.search(text) for text in texts)]


def _false_green_findings(instruction, expected, out, files, program, flags, log=None, kinds=None):
    """S2-05: the advisory findings on an all-green run (see FALSE_GREEN_BLOCKS), as short lines. R2 rules 2, 3, 5
    and 6 (FALSE_GREEN_E1B) join them: JSON outputs mostly 0 and table columns empty in every row (_fg_shape_findings),
    the TASK's stated formats, tolerances and the program's own verdicts (literal, bound and self-validation flags),
    a field the TASK states is positive holding a negative value (_fg_stated_sign_findings), and the accepted
    run's log reporting non-convergence or an overflow (`log`, a _RunLogWarnings). With `kinds` (a dict), the
    count of each new rule's findings is added under FG_KINDS' names."""
    from . import baseline
    found = []
    for name in expected['files']:
        data = files.get(Path(name).as_posix())
        if data is None:
            try:
                path = out / name
                data = None if path.is_symlink() or not path.is_file() else path.read_bytes()
            except OSError:
                data = None
        if data is not None and not data.strip():
            found.append(f'missing file {name}: it is empty (0 bytes or whitespace only), which counts as missing')
    literal, degenerate = baseline.advisory_flags(instruction, out)
    found += degenerate
    try:
        found += audit.nonfinite_flags(instruction, out, flags)
    except Exception:  # an advisory never ends the unit
        pass
    phrases = _unfinished_phrases(program)
    if phrases:
        found.append("the program's own comments or strings call it unfinished: " + ', '.join(phrases))
    parts = {'zero_or_empty': [], 'task_stated': list(literal), 'stated_sign': [],
             'log_warning': log.findings() if log is not None else []}
    try:
        parts['zero_or_empty'] = _fg_shape_findings(files)
    except Exception:  # an advisory never ends the unit
        pass
    try:
        parts['task_stated'] += audit.bound_flags(instruction, out)
    except Exception:
        pass
    try:
        parts['task_stated'] += [text for _, text in audit.selfval_flags(instruction, out, stated_only=True)]
    except Exception:
        pass
    try:
        parts['stated_sign'] = _fg_stated_sign_findings(instruction, files)
    except Exception:
        pass
    for kind in FG_KINDS:
        found += parts[kind]
        if kinds is not None:
            kinds[kind] = kinds.get(kind, 0) + len(parts[kind])
    return found


# R2 rules (FALSE_GREEN_E1B, user decision 10-22). Each is one more S2-05 finding: asked once per unit at the
# all-green stop, never a veto. Rules 2, 3, 5, 6 are behind false_green; rule 4 (a JSON boolean false) has its own
# switch, fg_bool_false, and its own counters. The status counter of each kind is thin_fg_<kind>. Rule 3 also has a
# host step, the one change to outputs: _fg_normalize rewrites the TASK's stated literal formats after each clean run.
FG_KINDS = ('zero_or_empty', 'task_stated', 'stated_sign', 'log_warning')
FG_ZERO_MIN = 4  # decimal numbers (JSON floats) an output holds before "mostly 0" is judged
FG_ZERO_SHARE = 0.5  # share of those numbers that are exactly 0
FG_EMPTY_ROWS = 3  # data rows a table has before a column empty in every row is judged
FG_SCAN = 200_000  # JSON leaves and table rows read per output file
_FG_TABLES = {'.csv': ',', '.tsv': '\t'}
# Rule 5: a clause of the TASK (a sentence or a part between semicolons) that names exactly one backticked field and
# states that it is positive; in a table row, the one field of the row's first cell, stated positive in the row. Only
# the TASK's words outside code spans count, never a field's name (a span such as `mean(F) >= 0` states nothing about
# F), and `>= 0` only where no plain name stands before it (`k >= 0` bounds k). A clause that also speaks of negative or
# signed values, negates, is conditional or states a sign convention (`positive values are gains`) states nothing.
_STATED_POSITIVE = re.compile(r'\b(?:positive\s+(?:numbers?|values?|decimals?|loss(?:es)?|amounts?)|expressed\s+as\s+'
                              r'(?:a\s+)?positive|as\s+positive|non-?negative|(?:is|are|be)\s+(?:always\s+)?positive)\b'
                              r'|(?:^|[^\w\s)\]])\s*(?:>=|≥)\s*0(?![.\d])', re.I)
_STATED_EXCEPT = re.compile(r'(?<![-\w])negative\b|\bsigned\b|\bnot\b|\bnever\b|\bif\b|\belse\b|\bwhen\b|'
                            r'\botherwise\b|\bunless\b|\bindicat\w*|\bdenot\w*|\bmeans?\b|'
                            r'\bpositive\s+(?:values?|numbers?)\s+(?:are|is|represent\w*|correspond\w*|reflect\w*)\b',
                            re.I)
_CODE_SPAN = re.compile(r'`[^`]*`')
_CODE_MARK = ' \x00 '  # a code span in the prose: a non-word mark, so `x` >= 0 still reads as bounding x
_CLAUSE_END = re.compile(r'(?<=[.!?])\s+|;')
_BACKTICKED = re.compile(r'`([A-Za-z_][A-Za-z0-9_]*)`')
# Rule 6: what the accepted run's log may report. 'invalid value encountered' is not one (2 of 23 passed units).
RUN_LOG_WARNINGS = (('ConvergenceWarning', re.compile(r'\bConvergenceWarning\b'),
                     'a fit or solver that did not converge returns its starting or last values, so any output '
                     'computed from it is suspect'),
                    ('overflow encountered', re.compile(r'\boverflow encountered\b'),
                     'a value overflowed to inf during the run, so any output computed from it, or a fallback taken '
                     'after it, is suspect'))
FG_BOOL_FALSE = ('{name}: {key} is false. Recheck the computation it reports on against the TASK; if false is the '
                 'correct answer, keep it. Never set it to true without fixing that computation')


def _fg_outputs(files, suffixes):
    """(name, text) of the snapshot's output files with these suffixes, not the harness's own files."""
    for name in sorted(files):
        if Path(name).suffix.lower() in suffixes and Path(name).name not in audit._NOT_OURS:
            yield name, files[name].decode('utf-8', errors='replace')


def _fg_json(files):
    for name, text in _fg_outputs(files, ('.json',)):
        try:
            yield name, json.loads(text)
        except (ValueError, RecursionError):
            continue


def _fg_leaves(obj, path=()):
    """(path, value) of a JSON document's scalars, at most FG_SCAN, iteratively (no recursion limit)."""
    stack, seen = [(path, obj)], 0
    while stack and seen < FG_SCAN:
        keys, value = stack.pop()
        if isinstance(value, dict):
            stack.extend((keys + (str(k),), v) for k, v in reversed(list(value.items())))
        elif isinstance(value, list):
            stack.extend((keys + (i,), v) for i, v in reversed(list(enumerate(value))))
        else:
            seen += 1
            yield keys, value


def _fg_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _fg_tables(files):
    import csv
    import io
    for name, text in _fg_outputs(files, tuple(_FG_TABLES)):
        try:
            rows = []
            for row in csv.reader(io.StringIO(text), delimiter=_FG_TABLES[Path(name).suffix.lower()]):
                rows.append(row)
                if len(rows) > FG_SCAN:
                    break
        except csv.Error:
            continue
        if rows:
            yield name, rows[0], rows[1:]


def _fg_shape_findings(files):
    """Rule 2: a JSON output with at least FG_ZERO_MIN decimal numbers (floats: integer counts and labels are not
    read) of which at least FG_ZERO_SHARE are exactly 0, and a table column (at least FG_EMPTY_ROWS data rows) that is
    empty in every row. Shape and values only; the feedback says to keep what the TASK defines that way."""
    found = []
    for name, data in _fg_json(files):
        numbers = [value for _, value in _fg_leaves(data) if isinstance(value, float)]
        zeros = sum(1 for value in numbers if value == 0)
        if len(numbers) >= FG_ZERO_MIN and zeros >= FG_ZERO_SHARE * len(numbers):
            found.append(f'{name}: {zeros} of its {len(numbers)} decimal numbers are exactly 0; recheck that each is '
                         'computed as the TASK defines it, and keep the 0s the TASK defines as 0')
    for name, header, rows in _fg_tables(files):
        if len(rows) < FG_EMPTY_ROWS:
            continue
        for index, column in enumerate(header):
            if all(not (row[index] if index < len(row) else '').strip() for row in rows):
                found.append(f'{name}: column {column or index + 1} is empty in all {len(rows)} rows; recheck that '
                             'it is computed and written as the TASK defines it, and keep it empty where the TASK '
                             'allows that')
    return found


def _fg_states_positive(text):
    """Whether TASK text (code spans aside) states a sign of `positive` and nothing that undoes it (rule 5)."""
    prose = _CODE_SPAN.sub(_CODE_MARK, text)
    return bool(_STATED_POSITIVE.search(prose)) and not _STATED_EXCEPT.search(prose)


def _fg_clauses(line):
    """A TASK line's clauses: split at sentence ends and semicolons outside code spans."""
    spans = []

    def hide(match):
        spans.append(match.group(0))
        return f'\x01{len(spans) - 1}\x01'
    masked = _CODE_SPAN.sub(hide, line)
    for part in _CLAUSE_END.split(masked):
        yield re.sub('\x01(\\d+)\x01', lambda match: spans[int(match.group(1))], part)


def _fg_stated_positive(instruction):
    """{field: the TASK clause}: each field the TASK states is positive (rule 5): the one backticked field of a clause
    that states it, or of a table row's first cell where the row states it. A clause or row naming more fields says
    nothing about which one is meant, so it states nothing."""
    stated = {}
    for line in (instruction or '').splitlines():
        if line.lstrip().startswith('|'):
            cells = [cell for cell in _CODE_SPAN.sub(lambda m: m.group(0).replace('|', ' '), line).strip()
                     .strip('|').split('|')]
            first = next((cell for cell in cells if cell.strip()), '')
            names = set(_BACKTICKED.findall(first))
            if len(names) == 1 and _fg_states_positive(line):
                stated.setdefault(names.pop(), ' '.join(line.split())[:300])
            continue
        for clause in _fg_clauses(line):
            names = set(_BACKTICKED.findall(clause))
            if len(names) == 1 and _fg_states_positive(clause):
                stated.setdefault(names.pop(), ' '.join(clause.split())[:300])
    return stated


def _fg_stated_sign_findings(instruction, files):
    """Rule 5: a field the TASK states is positive (_fg_stated_positive) holding a negative number: a table column
    with that header, or a JSON key (the last name on the path, `value` aside) with that name."""
    stated = _fg_stated_positive(instruction)
    if not stated:
        return []
    found, seen = [], set()

    def add(name, field, value):
        if (name, field) not in seen:
            seen.add((name, field))
            found.append(f'{name}: {field} has a negative value ({value:.6g}) but the TASK says: "{stated[field]}"')
    for name, header, rows in _fg_tables(files):
        for index, column in enumerate(header):
            if column.strip() not in stated:
                continue
            for row in rows:
                try:
                    value = float(row[index]) if index < len(row) else math.nan
                except ValueError:
                    continue
                if math.isfinite(value) and value < 0:
                    add(name, column.strip(), value)
                    break
    for name, data in _fg_json(files):
        for keys, value in _fg_leaves(data):
            key = next((k for k in reversed(keys) if isinstance(k, str) and k != 'value'), None)
            if key in stated and _fg_number(value) and math.isfinite(value) and value < 0:
                add(name, key, value)
    return found


def _fg_normalize(instruction, out, status, files, integers=None):
    """R2 rule 3's host step (its own switch fg_normalize, split from false_green): the literal formats the TASK
    states (fixed decimals, booleans, numbers written as text at a template's numeric slots) rewritten in the outputs
    after each clean run of the free loop, as the baseline solver rewrites them (baseline.normalize_literals, which
    never raises). It costs no request; the literal findings (false_green) then name only what it could not rewrite.
    `files` is the run's own snapshot (_snapshot, taken first: it refuses links and special files, which the rewrite
    must never touch). Returns the outputs' snapshot after the rewrite. Rewritten outputs that no longer snapshot
    (wider numbers past OUTPUT_CAP) are put back as the program wrote them (`files`), counted in
    thin_fg_normalize_reverted: the rewrite never costs a run. The files of a rewrite that stands are counted in
    thin_fg_normalized_runs and thin_fg_normalized_files (FG_NORMALIZE_COUNTERS). With `integers` (a dict: outputs_fix)
    whole-number id, count and timestamp columns are written as plain integers, never N decimals; `integers` gets
    {output name: indexes of the columns so rewritten} (_raw_integer_view)."""
    from . import baseline
    try:
        counts = (baseline.normalize_literals(instruction, out) if integers is None
                  else baseline.normalize_literals(instruction, out, integers=integers))
        rewritten = int(counts.get('normalized_files', 0) or 0)
    except Exception:  # an output the rewrite cannot read stays as the program wrote it
        rewritten = 0
    try:
        after = _snapshot(out)
    except HouseError:
        baseline._restore(out, files)
        status['thin_fg_normalize_reverted'] = status.get('thin_fg_normalize_reverted', 0) + 1
        return files
    if rewritten > 0:
        status['thin_fg_normalized_runs'] = status.get('thin_fg_normalized_runs', 0) + 1
        status['thin_fg_normalized_files'] = status.get('thin_fg_normalized_files', 0) + rewritten
    return after


# The counters _fg_normalize writes; an asserts-off rerun (B) keeps its own rewrite out of them.
FG_NORMALIZE_COUNTERS = ('thin_fg_normalized_runs', 'thin_fg_normalized_files', 'thin_fg_normalize_reverted')


def _raw_integer_view(raw, files, integers):
    """outputs_fix: the outputs as the OUTPUT REPORT reads them: the host's rewrite (`files`), except that the cells
    of the columns _fg_normalize wrote as plain integers (`integers`: {name: column indexes}) are shown as the program
    wrote them (`raw`, the snapshot before the rewrite). A file whose rows no longer line up is shown as rewritten."""
    import csv
    import io
    view = dict(files)
    for name, columns in integers.items():
        if name not in raw or name not in files:
            continue
        try:
            text = files[name].decode('utf-8-sig')
            before = list(csv.reader(io.StringIO(raw[name].decode('utf-8-sig'), newline='')))
            after = list(csv.reader(io.StringIO(text, newline='')))
        except (UnicodeDecodeError, csv.Error):
            continue
        if len(before) != len(after):
            continue
        for old, new in zip(before[1:], after[1:]):
            for j in columns:
                if j < len(old) and j < len(new):
                    new[j] = old[j]
        stream = io.StringIO()
        csv.writer(stream, lineterminator='\r\n' if '\r\n' in text[:65536] else '\n').writerows(after)
        view[name] = stream.getvalue().encode('utf-8')
    return view


# outputs_fix: a required deliverable counts as present only when it holds something (_substance)
_TABLE_SUFFIXES = ('.csv', '.tsv')
_PARQUET_SUFFIXES = ('.parquet', '.pq')


def _held(name, data):
    """(whether an output file holds something, its data rows): a CSV/TSV a header and a data row (its non-blank
    lines after the first), a JSON a non-empty object or array, a parquet a row (any file a byte when its rows cannot
    be read), any other file a byte."""
    suffix = Path(name).suffix.lower()
    if suffix in _TABLE_SUFFIXES:
        rows = max(0, sum(1 for line in data.decode('utf-8-sig', errors='replace').splitlines() if line.strip()) - 1)
        return rows > 0, rows
    if suffix == '.json':
        try:
            value = json.loads(data.decode('utf-8-sig'))
        except (ValueError, RecursionError):
            return False, 0
        return isinstance(value, (dict, list)) and bool(value), 0
    if suffix in _PARQUET_SUFFIXES:
        try:
            import io
            import pyarrow.parquet
            rows = int(pyarrow.parquet.ParquetFile(io.BytesIO(data)).metadata.num_rows)
        except Exception:  # unreadable here: judged by its size, as any other file
            return bool(data), 0
        return rows > 0, rows
    return bool(data), 0


def _substance(expected, files):
    """outputs_fix: (deliverables present, their data rows) in a run's snapshot: the required files (every file when
    the contract names none) that hold something (_held). _rank orders runs by the first and breaks ties by the second,
    so a run that left 1-byte or header-only files never outranks one that left the data."""
    names = [Path(name).as_posix() for name in expected['files']] if expected['files'] else sorted(files)
    present = rows = 0
    for name in dict.fromkeys(names):
        data = files.get(name)
        if data is None:
            continue
        held, count = _held(name, data)
        present += held
        rows += count if held else 0
    return present, rows


def _fg_bool_false_findings(files):
    """Rule 4 (switch fg_bool_false): a JSON output's top-level key, or a key one level below it, whose value is the
    boolean false. Asked about, never vetoed: false may be the right answer, and the feedback says to keep it then."""
    found = []
    try:
        for name, data in _fg_json(files):
            if not isinstance(data, dict):
                continue
            for key, value in data.items():
                if value is False:
                    found.append(FG_BOOL_FALSE.format(name=name, key=key))
                elif isinstance(value, dict):
                    found += [FG_BOOL_FALSE.format(name=name, key=f'{key}.{inner}')
                              for inner, item in value.items() if item is False]
    except Exception:  # an advisory never ends the unit: what was found so far stands
        pass
    return found


class _RunLogWarnings:
    """Rule 6: what one run's log reports (RUN_LOG_WARNINGS), read through _run_program's `health` hook when the run
    exits 0; the log's first and last FIT_LOG_BYTES are read, as FitWarnings reads them. A new one per run, so a run
    that never reaches the hook (a timeout) reports nothing."""

    def __init__(self):
        self.counts = {}

    def read(self, path, canaries, clean):
        from . import baseline
        self.counts = {}
        if not clean:
            return
        limit = baseline.FIT_LOG_BYTES
        try:
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            with os.fdopen(fd, 'rb') as stream:
                info = os.fstat(stream.fileno())
                if not stat.S_ISREG(info.st_mode):
                    return
                data = stream.read(limit)
                if info.st_size > 2 * limit:
                    stream.seek(info.st_size - limit)
                    data += b'\n' + stream.read(limit)
                elif info.st_size > limit:
                    data += stream.read()
        except Exception:  # read inside _run_program: an advisory never ends the run or the unit
            return
        self.scan(data)

    def scan(self, data):
        """Count RUN_LOG_WARNINGS in the log bytes read (its head and tail)."""
        self.counts = {}
        text = data.decode('utf-8', errors='replace')
        for label, pattern, _ in RUN_LOG_WARNINGS:
            count = len(pattern.findall(text))
            if count:
                self.counts[label] = count

    def findings(self):
        return [f"the program's own run log reports {label}" + (f' ({self.counts[label]} times)'
                                                                 if self.counts[label] > 1 else '') + ': ' + why
                for label, _, why in RUN_LOG_WARNINGS if label in self.counts]


def _fg_key(line):
    """An S2-05 finding's identity across two runs: its text without numbers (baseline._flag_key) and without rule 6's
    repeat count, so the same finding with a value or a count that moved is the same finding."""
    from . import baseline
    return baseline._flag_key(_RUN_LOG_COUNT.sub('', line))


_RUN_LOG_COUNT = re.compile(r' \(\d+ times\)')


def _fg_sanity(expected, out, task_dir, profile, program):
    """Whether the outputs on disk break a generic sanity property S3-02 checks (sanity.wired_findings: none with every
    sanity feature off). S2-05 reads it on a run it may hold, which sanity is then never asked about."""
    from . import sanity
    return bool(sanity.wired_findings(expected, out, task_dir, profile, program))


def _code_reserve(seconds):
    """The execution reserve solve's loop keeps for a code request once a program has run, when the last run took
    `seconds`: 1.25 x the run plus 10 s, within 30-120 s. A test keeps solve's own expression equal to this."""
    return max(30, min(120, seconds * 1.25 + 10))


def _repair_room(client, left):
    """S2-04 (T1_THIN_FEATURES repair_time), diagnostic only: the seconds a run started `left` before the deadline
    would have to keep for one repair, else 0. A run stopped at its cap (left - room) leaves the next code request
    the reserve _code_reserve gives a run of that length; with need = the predicted reply (_predicted_need, the
    send-time gate's own prediction, at most REQUEST_TIMEOUT_CAP; PREEXEC_REPLY_ROOM before this unit has a
    measured reply) + PREEXEC_MARGIN, the repair fits only when the capped run is at least as long as the reserve
    its repair's run gets. That reserve r solves r = _code_reserve(left - need - r); below its 120 s ceiling no
    positive time fits (r = 1.25(L - r) + 10 and L - r >= r give 2.25 L >= 2.5 L + 20), so the room is either
    120 + need, when left - need - 120 >= 120 and a request is left, or 0. Time arithmetic only: no execution
    duration is read. solve keeps no room at all (see REPAIR_ROOM_RULE); this counts where one would fit."""
    if client.attempts >= client.max_requests:
        return 0
    need = _predicted_need(client, 'thin_code')
    need = min(PREEXEC_REPLY_ROOM if need is None else need, REQUEST_TIMEOUT_CAP) + PREEXEC_MARGIN
    return 120 + need if left - need - 120 >= 120 else 0


# S2-04 rule (review 10-22): with repair_time on, a run that is not the last one is never stopped before the deadline.
# The old fixed 45 s room never bought a repair: a run stopped at left - 45 (left > 90) leaves 45 s, below the
# reserve its own length gives the next code request (at least 1.25 x 45 + 10 = 66 s), so no request was sent. Any
# room large enough for a repair (120 s + the reply, _repair_room) stops runs the old cap let finish (with 397 s left,
# a 212-352 s program). So every run now gets at least its old cap, and a capped run's repair is given up.
REPAIR_ROOM_RULE = 'to_deadline'


def _rank(expected, files, flags, status, *, canary=False, findings=0, substance=None):
    """Outputs without an input canary first (S2-10, `canary` = _canary_demotes): such a run ranks after every
    clean run. Then passed internal checks, and among those fewer S2-05 findings (false_green). Then named
    deliverables present before a clean exit; when the contract names no files, a clean exit before the file
    count, so a crash's partial or debug files never outrank a clean run. Then fewer visible flags. No deliverable
    ranks lowest. With `substance` (outputs_fix: _substance's (present, data rows)) only deliverables that hold
    something count as present, and more data rows break a tie last."""
    if substance is None:
        present, tail = status['thin_deliverables_present'] if expected['files'] else len(files), ()
    else:
        present, tail = substance[0], (substance[1],)
    if not present:
        return (not canary, False, 0, False, False, 0) + tuple(0 for _ in tail)
    clean = status.get('status') == 'executed_ungraded' and status.get('exit_code') == 0
    order = (present, clean) if expected['files'] else (clean, present)
    return (not canary, bool(status.get('thin_internal_checks_passed')), -findings) + order + (-len(flags),) + tail


MERGE_REASONS = ('ambiguous_binding', 'ambiguous_top_level', 'bundle_order_requires_replace')
# After a merge rejection the next ask stays a BUNDLE (the rejection names the statements), except for a name
# bound more than once (ambiguous_binding) or dropped top-level work (incomplete:module): those need REPLACE.
REBUNDLE = ('; resend a BUNDLE against CURRENT VERSION that avoids this (a full REPLACE is not needed): keep '
            'statements in their CURRENT PROGRAM order and leave unchanged ones out, put new top-level work '
            'inside functions called from the existing entry, and add no second entry.')


def _work(source):
    """The module-level names `source` binds other than by an import, or None when that is unknowable."""
    import symtable
    names = _module_names(source)
    if names is None:
        return None
    return names[0] - {symbol.get_name() for symbol in symtable.symtable(source, '<current>', 'exec').get_symbols()
                       if symbol.is_imported() and not symbol.is_assigned() and not symbol.is_namespace()}


def _rebinds(current, program):
    """`program` binds again every module-level name `current` binds other than by an import, and there is
    one: a resent whole file does. A snippet on a script without definitions passes the other A3b thresholds
    by re-importing (_structure then reads imports) but leaves the script's own work unbound. An import binds
    no work: a resend may drop an unused one, and one still read is a free name (_headerless_check)."""
    work, new = _work(current), _module_names(program)
    if work is None or new is None:
        return False
    return bool(work) and work <= new[0]


def _admit(state, reply, status, *, recovering=False, replacement_required=False):
    """Admit one free-mode code reply into `state`: thin.solve's own admission, callable by re-judge tools.

    The action line comes first, else one on its own line after leading prose or reasoning and before
    any fence; without one, a complete program is taken as a full replacement (_headerless_check), and a reply
    cut while the program is empty as the REPLACE of the empty version the header path would read (_headerless_cut).
    A finished BUNDLE that cannot be merged is checked once more against the same thresholds and, when it
    passes them (ready, no advisory or free name, every current top-level function and class kept) and
    binds every current module-level name again (_rebinds), is admitted as the REPLACE it then is
    (thin_bundle_as_replace).
    Counters go to `status`. Returns a dict with the loop's next `changed`, `feedback`, `recovering` and
    `replacement_required`, and the verdict: `route` ('header', 'headerless' or 'missing_header'),
    `action` (as admitted), `accepted`, `reason` (ApplyResult.reason, or 'missing_header') and `payload`
    (the text after the action line, the headerless program, or None)."""
    raw = reply.get('content', '')
    header = re.match(HEADER, raw) if isinstance(raw, str) else None
    answer = raw
    if header is None and isinstance(raw, str):
        # A header line after leading prose (or reasoning), before any fence, is still the header.
        answer = strip_reasoning(raw)
        header = re.search(LATE_HEADER, answer, flags=re.M)
        if header is not None and '```' in answer[:header.start()]:
            header, answer = None, raw
        elif header is not None:
            status['thin_late_headers'] = status.get('thin_late_headers', 0) + 1
    # The header stays primary; one complete program without it is admitted as a full replacement, and so is a
    # cut reply on an empty program, read as the header path reads a REPLACE of it (S0-16 B1: _headerless_cut).
    finish = reply.get('finish_reason', 'stop')
    if header is not None:
        headerless, note = None, ''
    elif finish == 'length' and not state.program and isinstance(raw, str) and raw.strip():
        headerless, note = _headerless_cut(state, raw)
    else:
        headerless, note = _headerless_check(state, raw, finish)
    verdict = {'recovering': recovering, 'replacement_required': replacement_required}
    if header is None and headerless is None:
        if raw is None or (isinstance(raw, str) and not raw.strip()):
            reason = 'Code response empty; no program changes were accepted.'
        elif not isinstance(raw, str):
            reason = 'Code response is not text; no program changes were accepted.'
        elif note:
            # Short and host-built: the exact reason this reply was not taken as a full program.
            reason = 'Missing action/base version header; not taken as a full REPLACE: ' + note + '.'
            if note.split(':', 1)[0] in FORMAT_REASONS:
                status['thin_format_rejections'] = status.get('thin_format_rejections', 0) + 1
        else:
            reason = 'Missing action/base version header.'
        return dict(verdict, route='missing_header', action=None, accepted=False, reason='missing_header',
                    changed=False, payload=None, feedback=reason + ' Resend complete code with REPLACE or BUNDLE '
                                                                   'and the CURRENT VERSION shown above.')
    action = header[1].lower() if header else 'replace'
    base = header[2] if header else state.version
    payload = answer[header.end():] if header else headerless
    result = state.apply(payload, reply.get('finish_reason', 'stop'), action, base)
    if headerless is not None and result.accepted:
        status['thin_headerless_accepted'] = status.get('thin_headerless_accepted', 0) + 1
    if (not result.accepted and action == 'bundle' and result.reason in MERGE_REASONS
            and reply.get('finish_reason') == 'stop'):
        # The model often resends the whole file as a BUNDLE (PRETEST A3b): the same payload as a full file.
        program = _headerless_check(state, payload, 'stop')[0]
        replaced = (state.apply(program, 'stop', 'replace', state.version)
                    if program is not None and _rebinds(state.program, program) else None)
        if replaced is not None and replaced.accepted:
            result, action = replaced, 'replace'
            status['thin_bundle_as_replace'] = status.get('thin_bundle_as_replace', 0) + 1
    if result.accepted and state.extraction == 'tolerant':
        # Prose, stray fences or other blocks were dropped around the one selected code block.
        status['thin_tolerant_extractions'] = status.get('thin_tolerant_extractions', 0) + 1
    elif result.reason in FORMAT_REASONS:
        status['thin_format_rejections'] = status.get('thin_format_rejections', 0) + 1
    feedback = str(state.feedback or result.reason)
    if _feature('input_fidelity') and result.accepted and SYNTAX_NOTE in feedback:
        status['thin_syntax_notes'] = status.get('thin_syntax_notes', 0) + 1
    # Recovery follows accepted versions, not rejected replies or syntax alone. A cut that kept no complete line
    # leaves nothing to recover: the next ask is the first file again (code_prompt asks it on an empty program).
    if result.accepted:
        if reply.get('finish_reason') == 'length':
            verdict['recovering'] = bool(state.program)
        elif action == 'replace' and reply.get('finish_reason') == 'stop':
            verdict['recovering'] = False
        verdict['replacement_required'] = False
    elif result.reason == 'ambiguous_binding' or (result.reason in MERGE_REASONS
                                                  and 'incomplete:module' in state.pending):
        verdict['replacement_required'] = True
    elif result.reason in MERGE_REASONS:
        verdict['replacement_required'] = False
        feedback = feedback.rsplit('; ', 1)[0] + REBUNDLE  # the named statements, without "use REPLACE"
    return dict(verdict, route='header' if header else 'headerless', action=action, accepted=result.accepted,
                reason=result.reason, changed=result.changed, payload=payload, feedback=feedback)


# A cut reply rejected for this stays rejected however it is continued (PRETEST A3a): ProgramVersion.apply
# rejects a header naming another version before it reads the code or the mode, and A3b re-checks only merge
# rejections. A merge rejection (MERGE_REASONS) is not futile while A3b reads the finished BUNDLE.
FUTILE_CONTINUATION = ('stale_base',)


class _NoRequests:
    """A client with no request left: _continue sends nothing and returns what admission then gets."""
    attempts = max_requests = 0


def _fingerprint(program):
    """code_step.fingerprint (the code apart from comments and layout), or None for a program it cannot read."""
    try:
        return code_step.fingerprint(program)
    except (SyntaxError, ValueError, RecursionError, MemoryError, OverflowError):
        return None


_REPLY_HEADER = re.compile(r'^\s*(REPLACE|BUNDLE)\s+[0-9a-f]{64}\s*\n')


def _code_body(content):
    """A code reply's code body as offline triage reads it (README 7b): after the action header line, the lines of
    the first fence (to its closing fence or the end), else the whole text; trailing spaces and blank edges removed.
    '' for no text."""
    if not isinstance(content, str):
        return ''
    text = content.replace('\r\n', '\n').replace('\r', '\n')
    header = _REPLY_HEADER.match(text)
    if header:
        text = text[header.end():]
    lines = text.split('\n')
    fences = [i for i, line in enumerate(lines) if line.strip().startswith('```')]
    if fences:
        lines = lines[fences[0] + 1:fences[1] if len(fences) > 1 else len(lines)]
    lines = [line.rstrip() for line in lines]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return '\n'.join(lines)


def _skip_continuation(state, reply):
    """The cut reply to admit without continuing it, else None (PRETEST A3a).

    Held: a cut reply whose header, admitted on a copy with the cut prefix, is rejected for a
    FUTILE_CONTINUATION reason, REPLACE and BUNDLE alike. A cut BUNDLE whose prefix cannot be merged is
    continued: the lines still to come decide whether the finished reply passes A3b (every current definition
    and module-level name present, ready, no free name; _admit), and a rejection of the merged whole, such
    as a second entry, can go away with a later statement of the same BUNDLE.
    The returned reply, marked continuable=False, is what _continue hands to admission when it sends nothing."""
    head = _cut_code(reply.get('content'))
    if head is None:
        return None
    trial = _admit(copy.deepcopy(state), dict(reply, content=head, finish_reason='length'), {})
    if trial['accepted'] or trial['reason'] not in FUTILE_CONTINUATION:
        return None
    return dict(_continue(_NoRequests, None, reply, None, None, None, None, (), {}), continuable=False)


# T1_THIN_FEATURES (S2-01): the R2 mechanisms on for this unit, parsed once at solve start; empty outside solve and
# when the variable is absent (the image always sets it, to solver_profile.THIN_FEATURES_DEFAULT unless a run gives
# another value). Every R2a/R2b mechanism reads its own switch through _feature, never the environment.
_FEATURES = frozenset()


def _feature(name):
    """Whether the R2 mechanism `name` (one of solver_profile.THIN_FEATURES) is on for this unit."""
    if name not in THIN_FEATURES:
        raise ValueError('Unknown thin feature: ' + repr(name))
    return name in _FEATURES


def _continuable_now(client, reply, deadline, reserve):
    """Whether _continue could send a first continuation for this cut reply now: code to continue, a request left
    and time above the floor (classification only, for a reply the unit cap leaves without one)."""
    from . import baseline
    return (reply.get('continuable') is not False and _cut_code(reply.get('content')) is not None
            and client.attempts < client.max_requests
            and deadline - baseline.CLOCK.now() >= reserve + CONTINUE_FLOOR)


def _unit_continuations(client, limit, unit_cap):
    """Continuations the next cut code reply may get: `limit` (T1_THIN_MAX_CONTINUATIONS) and, under
    T1_THIN_MAX_CONTINUATIONS_UNIT (`unit_cap`, None when off), no more than the unit has left. A continuation
    counts once however often it was retried (its attempt-0 request record), as R2A_PARAMS.md counts them."""
    if unit_cap is None:
        return limit
    sent = sum(1 for record in client.diagnostics.get('requests', [])
               if record.get('kind') == 'thin_continue' and record.get('attempt') == 0)
    return max(0, min(limit, unit_cap - sent))


@_named_stops
@_restores_best
def solve(client, task_dir, out, instruction, profile, hints, canaries, deadline, status, replay):
    """Called after baseline's container/input guards; generated execution stays in its runner."""
    from . import baseline
    mode = os.environ.get('T1_THIN_THINKING', 'on')
    if mode not in ('on', 'off', 'low_effort'):
        raise HouseError('Invalid thin thinking mode', kind='configuration', code='invalid_configuration')
    interface = os.environ.get('T1_THIN_INTERFACE', 'free')
    if interface not in ('free', 'choice'):
        raise HouseError('Invalid thin interface', kind='configuration', code='invalid_configuration')
    direct_value = os.environ.get('T1_THIN_DIRECT_STEP', '0')
    if direct_value not in ('0', '1') or (direct_value == '1' and interface != 'free'):
        raise HouseError('Invalid direct step configuration', kind='configuration', code='invalid_configuration')
    direct = direct_value == '1'
    continuations = MAX_CONTINUATIONS
    if 'T1_THIN_MAX_CONTINUATIONS' in os.environ:
        try:
            continuations = parse_thin_max_continuations(os.environ['T1_THIN_MAX_CONTINUATIONS'])
        except ValueError:
            raise HouseError('Invalid thin continuation cap', kind='configuration',
                             code='invalid_configuration') from None
    global _FEATURES
    unit_value = os.environ.get('T1_THIN_MAX_CONTINUATIONS_UNIT', THIN_MAX_CONTINUATIONS_UNIT)
    try:
        _FEATURES = parse_thin_features(os.environ.get('T1_THIN_FEATURES', ''))
        unit_cap = parse_thin_max_continuations_unit(unit_value)
    except ValueError:
        raise HouseError('Invalid thin feature switch', kind='configuration', code='invalid_configuration') from None
    status.update(thin_features=sorted(_FEATURES), thin_max_continuations_unit=unit_value)
    budget = None
    if 'T1_THIN_REASONING_BUDGET' in os.environ:
        from .solver_profile import parse_thin_reasoning_budget
        try:
            budget = parse_thin_reasoning_budget(os.environ['T1_THIN_REASONING_BUDGET'])
        except ValueError:
            raise HouseError('Invalid thin reasoning budget', kind='configuration', code='invalid_configuration') from None
        if mode != 'on':
            raise HouseError('Reasoning budget requires thinking on', kind='configuration', code='invalid_configuration')
    # S2-06: the final_delivery trigger candidate, refused before any request when it is not one of them.
    trigger = os.environ.get('T1_THIN_FINAL_DELIVERY_TRIGGER', FINAL_TRIGGER)
    try:
        requests_cap, seconds_cap = parse_thin_final_delivery_trigger(trigger)
    except ValueError:
        raise HouseError('Invalid final delivery trigger', kind='configuration', code='invalid_configuration') from None
    status['thin_final_delivery_trigger'] = trigger  # what the unit ran with, switch on or off (lane checks)
    choices = None
    if interface == 'choice':
        from . import choice_interface as choices
    from .solver_profile import REQUEST_THINKING, parse_thin_request_thinking
    # Per request type: its own switch (the image sets both), else the older global switch where one
    # is set, else the default. thin_thinking names the global mode only where it was actually used.
    legacy = 'T1_THIN_THINKING' in os.environ or budget is not None
    fallback = budget if budget is not None else {'on': 'default', 'off': False, 'low_effort': True}[mode]
    thinking, global_used = {}, False
    for kind, key in (('contract', 'T1_THIN_CONTRACT_THINKING'), ('code', 'T1_THIN_CODE_THINKING')):
        try:
            thinking[kind] = (parse_thin_request_thinking(os.environ[key]) if key in os.environ else
                              fallback if legacy else parse_thin_request_thinking(REQUEST_THINKING[key]))
        except ValueError:
            raise HouseError('Invalid thin request thinking', kind='configuration',
                             code='invalid_configuration') from None
        global_used = global_used or (legacy and key not in os.environ)
    status.update(baseline='thin_v1', thin_thinking=mode if global_used else 'per_request', thin_interface=interface,
                  thin_internal_checks_passed=False,
                  thin_direct_step=direct, thin_max_continuations=continuations,
                  thin_code_versions=0, thin_executions=0, thin_stop=None)
    for kind, value in thinking.items():
        status['thin_'+kind+'_thinking'] = ('on' if value == 'default' else 'low_effort' if value is True
                                            else 'off' if value is False else 'budget')
        if type(value) is int:
            status['thin_'+kind+'_reasoning_budget'] = value
    if direct:
        status.update(thin_successful_advances=0, thin_no_progress=0)
    if budget is not None:
        status['thin_reasoning_budget'] = budget
    if choices and not baseline.library_allowed_by_task(instruction):
        # This small prototype is not an arbitrary dependency-policy compiler.
        # Do not offer its numpy/qfkit/matplotlib components under a narrower task list.
        status.update(status='generation_failed', failure_kind='local_input',
                      failure_code='choice_dependencies_unsupported',
                      thin_stop='choice_dependencies_unsupported')
        return {}
    try:
        input_bindings = input_binding.load(task_dir)
    except (OSError, ValueError, TypeError, RecursionError) as exc:
        raise HouseError('Invalid declared input relocation metadata', kind='local_input',
                         code='invalid_input_bindings') from exc
    status.update(thin_input_bound_files=len(input_bindings), thin_input_path_rewrites=0)
    # S3-03b (switch repair_feedback): the free loop's notes on a failed run (RepairNotes), read once per solve.
    repair_notes = RepairNotes(input_bindings) if not direct and not choices and _feature('repair_feedback') else None
    context = task_context(task_dir, out, instruction, profile, hints, canaries, status)
    replay.prompt(context)
    input_identity = hashlib.sha256((instruction + json.dumps(profile, sort_keys=True)).encode()).hexdigest()
    replay.put('input-identity.json', json.dumps({'visible_task_profile_sha256': input_identity,
                                               'scope': 'exact instruction and observed profile; raw input hashes held by evaluator'}))
    contract = ''
    if not direct:
        planning = choices.plan_prompt(context) if choices else plan_prompt(
            context, complete_first=_feature('complete_first'), fidelity=_feature('task_fidelity'))  # the free loop's (direct makes no contract)
        # Reasoning can spend the whole output limit before any answer: an empty contract is asked
        # once more without thinking (one more counted request); free code then proceeds without one.
        for value in (thinking['contract'],) + ((False,) if thinking['contract'] is not False else ()):
            reply = _request(client, planning, deadline, replay, 'thin_contract', value, 60)
            content = reply.get('content')
            if content is None:
                content = ''
            if not isinstance(content, str):
                raise HouseError('Contract response is not text', kind='model_output', code='thin_nontext_contract')
            # Inline reasoning is not the contract (the code path strips it too); reasoning alone is empty.
            contract = baseline.withhold_canaries(strip_reasoning(content), canaries).strip()
            if contract:
                break
            status['thin_empty_contracts'] = status.get('thin_empty_contracts', 0) + 1
            if client.max_requests - client.attempts < 2:
                break  # the last request belongs to code
        if not contract:
            if choices:
                raise HouseError('Contract response empty', kind='model_output', code='thin_empty_contract')
            status['thin_no_contract'] = True
            contract = NO_CONTRACT
        contract = contract[:12000]
    if choices:
        try:
            selection = choices.validate_plan(contract)
        except ValueError as exc:
            raise HouseError(str(exc), kind='model_output', code='choice_invalid_plan') from exc
        contract = json.dumps(selection, ensure_ascii=False)
        if 'unsupported' in selection:
            replay.put('plan.txt', contract)
            status.update(status='generation_failed', failure_kind='model_output',
                          failure_code='choice_unsupported', thin_stop='choice_unsupported')
            return {}
    if not direct:
        replay.put('plan.txt', contract)
    expected = audit.output_contract(instruction)
    state = ProgramVersion(expected_files=expected['files'])
    if _feature('input_fidelity'):  # a syntax error inside kept code is quoted in the feedback
        state.syntax_note = True
    recipe = {'steps': []}
    feedback = first_feedback(state.version) if not direct and not choices else FIRST_FEEDBACK
    execution_feedback = ''
    reference_code = ''
    prior = None
    fields = {}
    # http_status and finish_reason of the reply (or continuation) that last changed the program or its pending
    # work: the request fields of the version a run executes, read when it is admitted, not when it runs.
    version_fields = {}
    tried = set()
    tried_ast = set()
    preexec_reported = set()  # findings of _execution_check already sent back instead of a run
    appended_steps = set()
    no_progress = 0
    last_execution = 0
    recovering = False
    replacement_required = False
    replacement_requested = set()
    # (rank, version, program, output files, request fields, measured status, S2-05 reading) of the best run
    best = None
    fg_held = None  # S2-05: the version held back, until the reply after the hold is read
    free = not direct and not choices  # the free code loop: continuation, headerless admission, longer stall bound
    soft_deadline = None  # progress_time: the image budget's deadline, once the card's own budget goes beyond it
    if free and _feature('progress_time'):
        soft, hard = _progress_deadlines(task_dir, deadline)
        if hard > soft:
            soft_deadline, deadline = soft, hard
            now = baseline.CLOCK.now()
            status.update(thin_soft_budget_left=round(max(0, soft - now)), thin_hard_budget_left=round(max(0, hard - now)))
    room_deadline = soft_deadline if soft_deadline is not None else deadline  # the review and drafts keep to it
    last_progress = baseline.CLOCK.now()  # progress_time: when the best run last got better
    clock = free and _feature('clock_fix')  # clock_fix (10-28): see PROGRESS_HARD_CAP

    def window_closed():
        """clock_fix: past the soft deadline, the best run got no better within PROGRESS_WINDOW. A unit without a run
        keeps its one chance: the loop's own check before a code request stops it. So does a unit whose best run left
        no deliverable (_rank's lowest rank): no run stands for it, and the reply in flight may be the first to
        deliver."""
        if soft_deadline is None or best is None or not (best[0][3] or best[0][4]):
            return False
        now = baseline.CLOCK.now()
        return now >= soft_deadline and now - last_progress > PROGRESS_WINDOW

    def run_room(seconds):
        """REMAINING SECONDS for a code prompt, from the hard deadline's room. clock_fix: the room a run gets (execute's
        cap), the soft deadline's or PROGRESS_RUN_CAP once that is smaller, never more than the hard deadline's."""
        if not clock or soft_deadline is None:
            return seconds
        return min(seconds, max(soft_deadline - deadline + seconds, PROGRESS_RUN_CAP))
    cut_runs = 0  # OB-B7: thin_executions when an accepted cut reply last set recovering
    ended = None  # the HouseError that ended the code requests
    # (state copy, static findings, request fields) of the last version held back before its run, until a run
    withheld = None
    run_outcome = None  # how the last execute() ended: ran, passed, setup_error or output_rejected
    # R2a switches of the free loop, read once (T1_THIN_FEATURES): S2-06 final_delivery, S2-08 loop.
    deliver = free and _feature('final_delivery')
    loop = free and _feature('loop')
    first_complete = free and _feature('complete_first')  # the complete first version, split from final_delivery
    # S2-05 (false_green) and its R2 rules: rules 2, 3, 5, 6 behind false_green, rule 4 behind fg_bool_false, and rule
    # 3's host rewrite of the outputs behind fg_normalize
    fg_base, fg_bool = free and _feature('false_green'), free and _feature('fg_bool_false')
    fg_norm = free and _feature('fg_normalize')
    rerun_asserts = free and _feature('assert_rerun')  # B: off, no asserts-off rerun, counter or record
    noted = free and _feature('executed_note')  # A1: an executed version's resend is named with its result
    stub = free and _feature('stub_check')  # 10-28: unfinished-program evidence keeps a run from passing
    literal = free and _feature('literal_contract')  # 10-28: a column or key the TASK spells out, lacked or misspelt
    # 10-28 outputs_fix: whole-number id/count/timestamp columns stay integers under fg_normalize (the review sees them
    # as written), only deliverables that hold something rank a run, and non-finite or absurd numbers keep a run back
    outputs = free and _feature('outputs_fix')
    review = free and _feature('output_review')  # 10-28: one OUTPUT REPORT review before the first passing run stands
    # lib_expose (10-28): the confident library matches the unit's first code prompts document unasked (none when off)
    exposed = _exposed(instruction) if free and _feature('lib_expose') else ()
    # lib_review (10-28): the TASK's library matches (_library_matched; none when off or when the TASK restricts its
    # dependencies), the functions a LIBRARY HINT named (once per unit each), and the functions the next code prompt
    # documents first while its feedback carries the LIBRARY CROSS-CHECK or the LIBRARY HINT that named them
    lib_matches = _library_matched(instruction) if free and _feature('lib_review') else ()
    lib_hinted, lib_checked = set(), ()
    delivered = False  # S2-06: the last run left every required deliverable (any file when the contract names none)
    # S2-06 (audit 2.6): the required deliverables the last run left missing, as that run's execute() saw them; the
    # disk can hold another run's outputs after it (_restore_early). Before any run, every one is missing.
    last_missing = tuple(expected['files'])
    held_states = {(state.version, state.pending)}  # S2-08: every (version, pending) the loop has held
    executed_ast = {}  # S2-08: code_step.fingerprint of each executed program -> its version
    bodies = {}  # S2-08: code body sha256 of each code reply (README 7b) -> the indexes of the replies that had it
    code_replies = 0  # S2-08: answered code requests so far (their index in `bodies`)
    resample = free and _feature('resample')  # A2
    stuck = 0  # A2: consecutive code replies that left no new version to run (counted with or without the switch)
    best_rerun = False  # B: `best` is an asserts-off rerun (_assert_rerun)
    rerun_on_disk = False  # B: that rerun became best after the last run, whose status (not the rerun's) status holds
    # crash_redraft (switch): the failure signature of each version whose run crashed (_crash_signature), the last run's
    # (None when it did not crash), and the repeated one that makes the next code request a fresh program (None: none)
    crash_redraft = free and _feature('crash_redraft')
    crashes, last_crash, redraft_due = {}, None, None
    # redraft_fix (switch, with crash_redraft only): the required deliverables with data (_with_data) that each crashed
    # version's run left, and that the run behind redraft_due left (None: unknown); the best run when an accepted
    # fresh program replaced the loop's state, until that program's first run (None: none open)
    redraft_fix = crash_redraft and _feature('redraft_fix')
    crash_data, due_data, redraft_base = {}, None, None
    # loop_fix (switch): the failing line of each executed version's last run (None: no line), the executed versions a
    # complete reply already resent byte for byte (a verified copy), the versions held once for having no entry point,
    # and whether the last reply was a copy stall (an accepted ready resend of a version that ran, or its twin)
    loop_fix = free and _feature('loop_fix')
    failures, copies, entry_held, copy_due = {}, set(), set(), False
    # loop_fix: (version, program) of the run the OUTPUT REVIEW holds, from the hold until the next run: a change that
    # has not run (a fix held back before its run) leaves the review open, so KEEP is offered and honoured (_keeps)
    reviewed = None

    def restore_best():
        """R6: the best run's outputs, measured status, request fields and source replace a later, worse run's."""
        nonlocal fields, prior, rerun_on_disk
        baseline._restore(out, best[3])
        for key in baseline.MEASURED + RESULT_FIELDS:
            status.pop(key, None)
        status.update(best[5])
        fields, prior = best[4], best[2]
        status['thin_best_restored'] = True
        rerun_on_disk = False
        status.pop('thin_assert_rerun_delivered', None)
        if best_rerun:  # B: the outputs now delivered are an asserts-off rerun's
            status['thin_assert_rerun_delivered'] = True

    def recover():
        """S3-03b OS-6 (_restores_best): an exception ending solve after a run leaves the best run, as R6 does at the
        end of solve. Only a run that ended the loop as passed (run_outcome, decided by that run) stands: nothing runs
        after it. The status flag alone is not enough: a run that passed the checks but was held back (S2-05) sets it
        while the loop goes on, and it stays set while a later run fails part way."""
        if best is not None and run_outcome != 'passed':
            restore_best()
            status['thin_best_restored_on_error'] = True
    _RECOVER.append(recover)

    def execute(static, ast_version, request_fields, last=False):
        """Run the current version once: the loop's run branch, and the last-chance run of a version held back before
        its run (S0-11; `last`: no request follows, so no time is kept for a repair). `request_fields` are those of
        the reply that produced this version (version_fields when it was admitted), never of a later request.
        Returns whether the loop ends; run_outcome says how the run ended."""
        nonlocal fields, prior, last_execution, best, no_progress, execution_feedback, feedback, withheld, run_outcome
        nonlocal delivered, fg_held, best_rerun, rerun_on_disk, last_missing, last_progress
        nonlocal last_crash, redraft_due, due_data, redraft_base, recovering, replacement_required, reviewed, lib_checked
        withheld = None  # any run, even one that cannot start, comes after the held version
        reviewed = None  # loop_fix: a run ends the review of the run before it
        previous_status = {k: status[k] for k in baseline.MEASURED if k in status}
        previous_fields = fields
        previous_outputs = _snapshot(out)
        previous_code = prior
        fields = dict(request_fields)
        status['program_sha256'] = state.version
        replay.put('program.py', state.program)
        began = baseline.CLOCK.now()
        left = deadline - began
        if soft_deadline is not None:  # progress_time: the soft deadline's room, or PROGRESS_RUN_CAP past it
            left = min(left, max(soft_deadline - began, PROGRESS_RUN_CAP))
        repair_room = 45 if not last and client.attempts < client.max_requests and left > 90 else 0
        old_cap = None  # S2-04: the cap the rule before REPAIR_ROOM_RULE would have set, where it no longer applies
        if repair_room and _feature('repair_time'):  # S2-04 (REPAIR_ROOM_RULE): on to the deadline instead
            old_cap = max(1, left - repair_room)
            repair_room = 0
            status['thin_repair_room_waived'] = status.get('thin_repair_room_waived', 0) + 1
            if _repair_room(client, left):  # diagnostic: where a repair's room would have fitted
                status['thin_repair_room_fit'] = status.get('thin_repair_room_fit', 0) + 1
        date_report = {}
        run_log = _RunLogWarnings() if fg_base else None  # R2 rule 6: this run's own log, read when it exits 0
        try:
            tail, report = baseline._run_program(state.program, out, deadline, canaries, status,
                clear=True, replay=replay, input_root=task_dir, cap=max(1, left-repair_room),
                retain_success_output=True, date_guard=date_report, health=run_log)
        except Exception:
            run_outcome = 'setup_error'
            baseline._restore(out, previous_outputs)
            for key in baseline.MEASURED:
                status.pop(key, None)
            status.update(previous_status)
            fields = previous_fields
            prior = previous_code
            if prior is None:
                raise
            replay.put('program.py', prior)
            status['thin_stop'] = 'execution_setup_error'
            return True
        last_execution = baseline.CLOCK.now()-began
        if old_cap is not None and status.get('runs'):
            status['runs'][-1]['old_cap_sec'] = round(old_cap, 3)  # lets triage tell which run repair_time extended
        if old_cap is not None and last_execution > old_cap:
            # repair_time's trigger: this run went on past where the old rule would have stopped it
            status['thin_repair_room_overran'] = status.get('thin_repair_room_overran', 0) + 1
        rerun_on_disk = False  # this run's status and outputs replace those of any rerun before it
        status['thin_executions'] += 1
        tried.add(state.version)
        if loop_fix:  # the line this run's failure points at, read by a later copy of this version (_copy_target)
            diagnostic = status.get('execution_diagnostics')
            line = (diagnostic.get('generated_line') if status.get('status') == 'execution_failed'
                    and isinstance(diagnostic, dict) else None)
            failures[state.version] = line if type(line) is int and line > 0 else None
        if direct:
            tried_ast.add(ast_version)
        prior = state.program
        try:
            files = _snapshot(out)  # first: it refuses links and special files, which the rewrite must never touch
        except HouseError as exc:
            # Outputs that cannot be captured (a link or special file, over OUTPUT_CAP) never cost the unit a run it
            # already delivered: that best run is restored (R6) and the loop ends; without one the unit ends here.
            run_outcome = 'output_rejected'
            if exc.kind != 'local_limit' or best is None:
                raise
            restore_best()
            status['thin_stop'] = exc.code
            return True
        raw_files, integers = files, ({} if outputs else None)  # outputs_fix: the review's view of integer columns
        if fg_norm and status.get('status') == 'executed_ungraded':
            # R2 rule 3: the outputs as the host left them
            files = _fg_normalize(instruction, out, status, files, integers=integers)
        status['thin_output_version'] = state.version
        if status.get('runs'):
            status['runs'][-1]['program_sha256'] = state.version
        replay.put(f'version-{state.version}.json', json.dumps({
            'program_sha256': state.version, 'visible_task_profile_sha256': input_identity, 'outputs': _manifest(files),
            'status': status['status'], 'exit_code': status.get('exit_code')}, sort_keys=True))
        flags = audit.contract_flags(expected, out)
        date_flags = date_validation.flags(date_report, state.program)
        flags.extend(date_flags)
        if stub and status.get('status') == 'executed_ungraded':
            # stub_check (10-28): a clean run whose program or required files show it is unfinished does not pass
            try:
                unfinished = stub_check.findings(state.program, expected['files'], files, instruction=instruction,
                                                 bound_parquet=clock)
            except Exception:  # evidence only: a checker fault never costs the unit its run
                unfinished = []
            if unfinished:
                status['thin_stub_runs'] = status.get('thin_stub_runs', 0) + 1
                flags.extend(baseline.withhold_canaries(line, canaries) for line in unfinished)
        if literal and status.get('status') == 'executed_ungraded':
            # literal_contract (10-28): a clean run whose CSV header or JSON keys lack or misspell a name the TASK
            # spells out for that file ('Date' for `date`, an unnamed index column) does not pass; contract_flags only
            # tests that the files exist and parse. Evidence only: the outputs are never rewritten.
            try:
                slips = literal_contract.findings(instruction, expected, files)
            except Exception:  # a checker fault never costs the unit its run
                slips = []
            if slips:
                status['thin_literal_contract_runs'] = status.get('thin_literal_contract_runs', 0) + 1
                flags.extend(baseline.withhold_canaries(line, canaries) for line in slips)
        if outputs and status.get('status') == 'executed_ungraded':
            # outputs_fix (10-28): a clean run whose required outputs hold NaN/inf numbers (output_report's rule 1,
            # with the values the TASK allows aside) or numbers past 1e15 in absolute value does not pass
            try:
                broken = output_report.blocking(instruction, expected['files'], files)
            except Exception:  # a checker fault never costs the unit its run
                broken = []
            if broken:
                status['thin_outputs_blocked_runs'] = status.get('thin_outputs_blocked_runs', 0) + 1
                flags.extend(baseline.withhold_canaries(OUTPUTS_BLOCKED.format(finding=line), canaries)
                             for line in broken)
        tainted = _canary_outputs(files, canaries)  # S2-10: first, so the 30 shown flags always include it
        if tainted:
            status['thin_canary_runs'] = status.get('thin_canary_runs', 0) + 1
            if not _canary_demotes(expected, tainted):
                status['thin_canary_side_runs'] = status.get('thin_canary_side_runs', 0) + 1
            flags[:0] = [CANARY_FLAG.format(name=baseline.withhold_canaries(name, canaries))
                         for name in tainted[:_SHOWN]]
        if direct:
            status['thin_date_guard_blocking'] = bool(date_flags)
            if (status.get('exit_code') == 0 and status.get('status') == 'executed_ungraded'
                    and not date_flags):
                no_progress = 0
                status['thin_successful_advances'] += 1
                status['thin_no_progress'] = 0
        date_note = baseline.withhold_canaries(date_validation.feedback(date_report), canaries)
        status['thin_date_guard_status'] = date_report.get('status', 'unverified')
        status['thin_date_guard_calls'] = date_report.get('calls_observed', 0)
        status['thin_date_guard_unresolved'] = date_report.get('unresolved_inferred_date_loss', 0)
        replay.put('date-validation.json', baseline.withhold_canaries(
            json.dumps(date_report, ensure_ascii=False, sort_keys=True), canaries))
        last_missing = tuple(name for name in expected['files'] if not (out/name).is_file())
        status['thin_missing_files'] = len(last_missing)
        status['thin_deliverables_present'] = len(expected['files'])-status['thin_missing_files']
        status['thin_expected_deliverables'] = len(expected['files'])
        substance = _substance(expected, files) if outputs else None  # outputs_fix: only files that hold something
        if substance is not None and expected['files']:
            status['thin_deliverables_present'] = substance[0]
        status['thin_internal_checks_passed'] = bool(status['status'] == 'executed_ungraded'
                                                     and files and not flags)
        delivered = not status['thin_missing_files'] if expected['files'] else bool(files)
        if noted or loop:
            # A1 (switch executed_note) and S2-08 (switch loop), S2-08's text: what an accepted resend of this version
            # is told in place of READY, since it never runs again. With the loop switch also a resend of its code
            # apart from comments and layout (executed_ast), which the switch keeps from running again too.
            exception = baseline._exception_line(report, tail)
            state.executed[state.version] = (
                str(status.get('status')) + ' (exit code ' + str(status.get('exit_code')) + ')'
                + (': ' + exception if exception else '')
                + (f", {status['thin_missing_files']} of {len(expected['files'])} required files missing"
                   if status['thin_missing_files'] else ''))
        if loop:
            shape = _fingerprint(state.program)
            if shape is not None:
                executed_ast.setdefault(shape, state.version)
        green = bool(status['thin_internal_checks_passed'])
        fg_kinds = {}
        findings = (_false_green_findings(instruction, expected, out, files, state.program, flags, run_log, fg_kinds)
                    if green and fg_base else [])
        bool_findings = _fg_bool_false_findings(files) if green and fg_bool else []  # R2 rule 4, its own switch
        findings = findings + bool_findings
        rank = _rank(expected, files, flags, status, canary=_canary_demotes(expected, tainted),
                     findings=len(findings), substance=substance)
        keys = frozenset(_fg_key(line) for line in findings)
        # Ties keep the later version, except that an all-green run with S2-05 findings never replaces the one held
        # back for as many findings when one of its findings is new (_fg_key: numbers and counts aside). best[6] keeps
        # what S2-05 read on that run: its findings' keys, their count, rule 4's share, and (sanity on) whether it
        # breaks a sanity property, which a run S2-05 held was never asked about (sanity_hold below).
        if best is None or rank > best[0]:
            last_progress = baseline.CLOCK.now()  # progress_time: the best run got better
            if soft_deadline is not None and last_progress > soft_deadline:
                status['thin_extended_runs'] = status.get('thin_extended_runs', 0) + 1
        if best is None or rank > best[0] or (rank == best[0] and (not findings or keys <= best[6]['keys'])):
            best = (rank, state.version, state.program, files, fields,
                    copy.deepcopy({k: status[k] for k in baseline.MEASURED + RESULT_FIELDS if k in status}),
                    {'keys': keys, 'findings': len(findings), 'bool': len(bool_findings),
                     'sanity': bool(findings) and _fg_sanity(expected, out, task_dir, profile, state.program)})
            best_rerun = False
        # S2-05: held back once per unit, while the repair it asks for fits (the next code request's reserve after
        # this run); not on a last-chance run, which no request follows.
        hold = bool(findings and status.get('thin_false_green_blocks', 0) < FALSE_GREEN_BLOCKS and not last
                    and _repair_fits(client, deadline, _code_reserve(last_execution)))
        if hold:
            status['thin_false_green_blocks'] = status.get('thin_false_green_blocks', 0) + 1
            status['thin_false_green_findings'] = len(findings)
            # decision 10-22 20:15: what the hold rests on, rule 4 (fg_bool_false) apart from every other rule
            # (false_green): both > 0 is a joint intervention of the two switches
            status['thin_fg_hold_rule4'] = len(bool_findings)
            status['thin_fg_hold_other'] = len(findings) - len(bool_findings)
            # R2 metric (e): the requests and runs spent when it held, as thin_sanity_at and thin_sanity_runs
            status.update(thin_false_green_at=client.attempts, thin_false_green_runs=status['thin_executions'])
            fg_held = state.version  # the next reply may keep it (solve's loop, thin_false_green_kept)
            status['thin_false_green_unfinished'] = bool(fg_base and _unfinished_phrases(state.program))
            if fg_base:
                status['thin_fg_zero_or_empty'] = fg_kinds.get('zero_or_empty', 0)
                status['thin_fg_task_stated'] = fg_kinds.get('task_stated', 0)
                status['thin_fg_stated_sign'] = fg_kinds.get('stated_sign', 0)
                status['thin_fg_log_warning'] = fg_kinds.get('log_warning', 0)
            if fg_bool:
                status['thin_fg_bool_false_findings'] = len(bool_findings)
        review_text = None
        if (review and not hold and green and not last and not status.get('thin_review_held')
                and room_deadline - baseline.CLOCK.now() >= OUTPUT_REVIEW_ROOM
                and _repair_fits(client, deadline, _code_reserve(last_execution))):
            try:
                shown = _raw_integer_view(raw_files, files, integers) if integers else files  # outputs_fix
                built = output_report.build(instruction, expected['files'], shown, state.program,
                                           bound_parquet=clock)
            except Exception:  # the review needs a report; without one the run is reviewed on the TASK alone
                built = {'report': '(the host could not build the output report)', 'violations': []}
            issues = [baseline.withhold_canaries(str(line), canaries) for line in (built.get('violations') or [])[:25]]
            review_text = OUTPUT_REVIEW.format(
                report=baseline.withhold_canaries(built.get('report', ''), canaries),
                violations=('HOST-DETECTED ISSUES (generic checks; verify each against the TASK):\n'
                            + '\n'.join(issues) + '\n') if issues else '')
            shown = _library_unused(lib_matches, state.program)
            if shown:  # lib_review: the program computes by hand what the TASK's library matches implement
                review_text += LIB_CROSS_CHECK.format(functions='\n'.join(line for _, line in shown))
                lib_checked = tuple(function for function, _ in shown)
                status['thin_lib_review_cross_checks'] = status.get('thin_lib_review_cross_checks', 0) + 1
            hold = True
            fg_held = state.version  # the next reply may keep it (as after an S2-05 hold)
            if loop_fix:
                reviewed = (state.version, state.program)
            status.update(thin_review_held=1, thin_review_at=client.attempts, thin_review_issues=len(issues))
        # S3-02: asked only when S2-05 did not hold this run; `best` above keeps the run it holds as passed. Its room
        # is S2-05's: the free loop, not a last-chance run, and the repair fits.
        sanity_hold = bool(not hold and status['thin_internal_checks_passed'] and _sanity_hold(
            client, expected, out, task_dir, profile, state.program, status, flags,
            lambda: bool(free and not last) and _repair_fits(client, deadline, _code_reserve(last_execution))))
        run_outcome = 'passed' if status['thin_internal_checks_passed'] and not hold and not sanity_hold else 'ran'
        if status['thin_internal_checks_passed'] and not hold and not sanity_hold:
            if fg_base or fg_bool:
                # the run held back, which this one does not improve on (best), stands; unless it breaks a sanity
                # property this one does not (best[6]['sanity']: S2-05 held it before sanity could ask)
                stands = best[6] if findings and best[1] != state.version else None
                if stands is not None and stands['sanity'] and not _fg_sanity(expected, out, task_dir, profile,
                                                                              state.program):
                    status['thin_false_green_sanity_kept'] = True
                    stands = None
                if stands is not None:
                    restore_best()
                    status['thin_false_green_restored'] = True
                    if 'thin_sanity_outcome' in status:  # the delivered run's
                        status['thin_sanity_outcome'] = 'passed_flagged' if stands['sanity'] else 'passed_clean'
                # the counts of the run delivered
                status['thin_false_green_stop_findings'] = stands['findings'] if stands else len(findings)
                if fg_bool:
                    status['thin_fg_bool_false_stop_findings'] = stands['bool'] if stands else len(bool_findings)
            status['thin_stop'] = 'internal_checks_passed'
            return True
        evidence = _execution_evidence(status, last_execution)
        event = evidence['execution_resource']['memory_event']
        resource_note = ('cgroup OOM kill observed during this run; the event does not identify a Python line.'
                         if event == 'cgroup_oom_observed' else
                         'Signal 9 cause is unknown; SIGKILL alone does not establish OOM.'
                         if evidence['exit_code'] in (-9, 137) and event != 'controller_timeout' else '')
        # Admission advisories and the static findings stay visible after a run replaces the feedback.
        notes = static[0] + static[1] + (['admission advisory ' + ', '.join(state.advisories)]
                                         if free and state.advisories else [])
        execution_feedback = feedback = ('Executed version ' + state.version + '\nStatus: ' + status['status']
                    + '\nEXECUTION EVIDENCE:\n' + json.dumps(evidence, sort_keys=True)
                    + ('\n' + resource_note if resource_note else '')
                    + '\nRuntime output (untrusted diagnostic, not a verdict):\n' + (tail + '\n' + report)[-5000:]
                    + '\nVisible output checks:\n' + '\n'.join(flags[:30])
                    + (review_text if review_text else
                       FALSE_GREEN.format(findings='\n'.join(findings[:30])) if hold else '')
                    + ('\nDate conversion observations (main process only):\n' + date_note if date_note else '')
                    + ('\nNo deliverables were produced.' if not files else '')
                    + ('\nStatic notes on this version (found before the run, not a verdict): '
                       + '; '.join(notes) if notes else ''))
        if repair_notes is not None:  # S3-03b (switch repair_feedback): host notes on this run for the next ask
            execution_feedback = feedback = feedback + repair_notes.after_run(state.program, status, tail, report)
        if crash_redraft and not last:
            # crash_redraft: this run's failure signature; the same one as the run before makes the next code request
            # a fresh program (a run that did not crash, a held passing one included, starts the count again)
            crash = _crash_signature(state.program, status)
            stuck_output = False
            if redraft_fix:
                # redraft_fix: a run without a crash signature has its output signature instead (the required outputs
                # byte for byte, its status and the host-detected issues), so the same one twice is a stuck output
                with_data = _with_data(expected['files'], files)
                if crash is None:
                    crash = _output_signature(expected['files'], files, status, flags + findings)
                    stuck_output = crash is not None
                else:
                    crash_data[state.version] = with_data
            if crash is not None:
                if not stuck_output:
                    crashes[state.version] = crash
                if crash == last_crash:
                    redraft_due = crash
                    status['thin_crash_repeats'] = status.get('thin_crash_repeats', 0) + 1
                    if redraft_fix:
                        due_data = with_data
                        if stuck_output:
                            status['thin_crash_stuck_outputs'] = status.get('thin_crash_stuck_outputs', 0) + 1
            last_crash = crash
        if lib_matches and not status['thin_internal_checks_passed'] and not last and redraft_due is None:
            # lib_review: a failed run (a crash or failed checks) of a program that uses none of the TASK's library
            # matches: the next code request names those not hinted yet, each once per unit (not on a fresh-program
            # request, crash_redraft's, which does not carry this feedback)
            shown = _library_unused(lib_matches, state.program, lib_hinted)
            if shown:
                execution_feedback = feedback = feedback + LIB_REPAIR_HINT.format(
                    functions='; '.join(line for _, line in shown))
                lib_hinted.update(function for function, _ in shown)
                lib_checked = tuple(function for function, _ in shown)
                status['thin_lib_repair_hints'] = status.get('thin_lib_repair_hints', 0) + 1
        # B (switch assert_rerun): a run stopped at the program's own assert statement runs once more with asserts off;
        # None when it did not run (always, with the switch off), else whether it became the best run (its outputs
        # then stay on disk).
        rerun = (assert_rerun(last) if rerun_asserts and status.get('status') == 'execution_failed'
                 and _own_assert_line(state.program, status.get('execution_diagnostics')) is not None else None)
        if not direct and not status['thin_internal_checks_passed'] and (
                rerun is False or rerun is None and best[1] != state.version):
            # S3-03b OS-4: a run that did not become the best gives the disk back to the best run's outputs now, not
            # only when solve ends, so a unit stopped from outside meanwhile keeps them; its own evidence stays above.
            # A run that passed the internal checks but goes on (held back, S2-05) keeps its outputs: the end of solve
            # keeps such a run with its status, so the disk must keep its files too. A rerun that did not become the
            # best (B) gives the disk back too, also when the best run is the first run of this same version.
            _restore_early(out, best[3], status)
        if redraft_base is not None and not last:
            # redraft_fix: the fresh program's first run. When it did not pass and leaves fewer required deliverables
            # with data than the best run before it (as many: a lower rank), that run's program is CURRENT PROGRAM
            # again for the following requests; `best` and restore_best stay as they are.
            base, redraft_base = redraft_base, None
            fresh_data, base_data = _with_data(expected['files'], files), _with_data(expected['files'], base[3])
            fresh_rank = rank
            if rerun is True and best[1] == state.version:
                # B: the fresh program's asserts-off rerun became the best run; the fresh line is compared by the
                # better of its two runs, so a base program ranking below that rerun is never put back
                fresh_data, fresh_rank = max((fresh_data, rank), (_with_data(expected['files'], best[3]), best[0]))
            if not status['thin_internal_checks_passed'] and (fresh_data, fresh_rank) < (base_data, base[0]):
                back = ProgramVersion(expected_files=expected['files'])
                back.program, back.pending = base[2], ()
                fresh_version = state.version
                state.__dict__.update(back.__dict__, executed=state.executed)
                recovering = replacement_required = False
                failed = crashes.get(state.version)
                execution_feedback = feedback = feedback + '\n' + REDRAFT_RESTORED.format(
                    fresh=fresh_version, fresh_data=fresh_data, base=state.version, base_data=base_data,
                    failure=' (its run ended with ' + _crash_failure(failed) + ')' if failed is not None else '')
                status['thin_crash_redraft_restored'] = True
        return False

    def assert_rerun(last):
        """B: the asserts-off rerun (PYTHONOPTIMIZE=1) of the version that just stopped at its own assert statement.
        It starts only when, after it at its cap (max(ASSERT_RERUN_FLOOR, ASSERT_RERUN_FACTOR x that run)), the next
        code request still fits (_repair_fits with that run's reserve), or when no request can follow (`last`, or the
        request cap spent). Returns None when it did not start, else whether it became the best run: only when its
        rank (internal checks never passed) is above the best run's and it leaves more required deliverables (more
        files when the contract names none), so a clean exit alone never displaces a crash that wrote every file.
        The status keeps the first run's measured fields and run record either way (the loop, the model's feedback and
        the replay's run log are that run's); the rerun's own become `best`'s when it wins (rerun_on_disk until the
        next run). Never raises: a rerun that cannot start or be read counts as not the best."""
        nonlocal best, best_rerun, rerun_on_disk
        left = deadline - baseline.CLOCK.now()
        cap = max(ASSERT_RERUN_FLOOR, ASSERT_RERUN_FACTOR * last_execution)
        follows = not last and client.attempts < client.max_requests
        if left <= 1 or follows and not _repair_fits(client, deadline - cap, _code_reserve(last_execution)):
            status['thin_assert_rerun_skipped'] = status.get('thin_assert_rerun_skipped', 0) + 1
            return None
        saved = {k: copy.deepcopy(status[k]) for k in baseline.MEASURED + RESULT_FIELDS + ('run_capped',)
                 if k in status}
        had_runs = isinstance(status.get('runs'), list)
        count = len(status['runs']) if had_runs else 0
        status['thin_assert_reruns'] = status.get('thin_assert_reruns', 0) + 1
        won, date_report = False, {}
        try:
            baseline._run_program(state.program, out, deadline, canaries, status, clear=True,
                                  replay=baseline._NO_REPLAY, input_root=task_dir, cap=max(1, cap),
                                  retain_success_output=True, date_guard=date_report, optimize=True)
            files = _snapshot(out)
            if fg_norm and status.get('status') == 'executed_ungraded':
                # rule 3's rewrite applies to the rerun's outputs too, but it is counted apart: the fg_normalize
                # counters keep the loop's own runs only
                kept = {key: status.pop(key) for key in FG_NORMALIZE_COUNTERS if key in status}
                try:
                    files = _fg_normalize(instruction, out, status, files, integers={} if outputs else None)
                    if status.get('thin_fg_normalized_runs'):  # a rewrite that changed files and stood
                        status['thin_assert_rerun_fg_normalized'] = 1 + status.get('thin_assert_rerun_fg_normalized',
                                                                                   0)
                finally:
                    for key in FG_NORMALIZE_COUNTERS:
                        status.pop(key, None)
                    status.update(kept)
            status['thin_output_version'] = state.version
            status['thin_missing_files'] = sum(not (out/name).is_file() for name in expected['files'])
            status['thin_deliverables_present'] = len(expected['files']) - status['thin_missing_files']
            status['thin_expected_deliverables'] = len(expected['files'])
            substance = _substance(expected, files) if outputs else None  # outputs_fix
            if substance is not None and expected['files']:
                status['thin_deliverables_present'] = substance[0]
            status['thin_internal_checks_passed'] = False
            status['thin_date_guard_status'] = date_report.get('status', 'unverified')
            status['thin_date_guard_calls'] = date_report.get('calls_observed', 0)
            status['thin_date_guard_unresolved'] = date_report.get('unresolved_inferred_date_loss', 0)
            flags = audit.contract_flags(expected, out) + date_validation.flags(date_report, state.program)
            tainted = _canary_outputs(files, canaries)
            flags[:0] = [CANARY_FLAG.format(name=baseline.withhold_canaries(name, canaries)) for name in tainted[:_SHOWN]]
            rank = _rank(expected, files, flags, status, canary=_canary_demotes(expected, tainted), substance=substance)
            present = status['thin_deliverables_present'] if expected['files'] else len(files)
            ahead = best[5].get('thin_deliverables_present', 0) if expected['files'] else len(best[3])
            if substance is not None:  # outputs_fix: files that hold something, on both sides
                present, ahead = substance[0], _substance(expected, best[3])[0]
            if status.get('status') == 'executed_ungraded' and status.get('exit_code') == 0:
                status['thin_assert_rerun_exit0'] = status.get('thin_assert_rerun_exit0', 0) + 1
            replay.put('assert-rerun.json', json.dumps({
                'program_sha256': state.version, 'outputs': _manifest(files), 'status': status.get('status'),
                'exit_code': status.get('exit_code'), 'cap_sec': round(cap, 1)}, sort_keys=True))
            if rank > best[0] and present > ahead:
                won = True
                best = (rank, state.version, state.program, files, fields,
                        copy.deepcopy({k: status[k] for k in baseline.MEASURED + RESULT_FIELDS if k in status}),
                        {'keys': frozenset(), 'findings': 0, 'bool': 0, 'sanity': False})
                best_rerun = rerun_on_disk = True
                status['thin_assert_rerun_best'] = status.get('thin_assert_rerun_best', 0) + 1
        except Exception:
            won = False
        finally:
            for key in baseline.MEASURED + RESULT_FIELDS + ('run_capped',):
                status.pop(key, None)
            status.update(saved)
            if had_runs:
                del status['runs'][count:]
            else:
                status.pop('runs', None)
        return won

    while client.attempts < client.max_requests:
        if soft_deadline is not None and baseline.CLOCK.now() >= soft_deadline and (
                baseline.CLOCK.now() - (last_progress if best is not None else soft_deadline) > PROGRESS_WINDOW):
            # progress_time: past the image budget and no longer improving (or no run at all by the window's end)
            status['thin_stop'] = 'soft_budget'
            break
        if (status.get('thin_sanity_at') is not None and client.attempts > status['thin_sanity_at']
                and 'thin_sanity_outcome' not in status):
            break  # S3-02: the one sanity repair request is spent and no run passed after it
        if recovering and status['thin_executions'] > cut_runs and status.get('thin_output_version') == state.version:
            recovering = False  # OB-B7: this version ran after the cut reply; its evidence sets the next ask
        direct_action = _direct_action(state, status, replacement_required=replacement_required) if direct else None
        direct_base = state.version
        repair_target = (_local_repair_target(state, status, replacement_required=replacement_required)
                         if direct else None)
        reserve = max(30, min(120, last_execution * 1.25 + 10)) if prior else 60
        seconds_left = run_room(deadline - baseline.CLOCK.now()) if free else None
        final = replace_only = None
        if deliver and not delivered:
            # S2-06: the trigger reads REMAINING REQUESTS and REMAINING SECONDS as this prompt shows them.
            shown = round(max(0, seconds_left), 1) if math.isfinite(seconds_left) else None
            why = ('requests' if client.max_requests - client.attempts <= requests_cap
                   else 'seconds' if shown is not None and shown <= seconds_cap else None)
            if why is None:
                pass
            elif _ask(state.program, state.pending, feedback, recovering=recovering,
                      replacement_required=replacement_required) in FINAL_ASKS:
                final = (why, last_missing)  # audit 2.6: the last run's own list, as `delivered` is
            else:
                replace_only = why
        prompt = (choices.code_prompt(context, contract, recipe, state.version, feedback,
                                      client.max_requests-client.attempts, deadline-baseline.CLOCK.now())
                  if choices else step_prompt(context, state, feedback, status,
                      [name for name in expected['files'] if not (out/name).is_file()],
                      client.max_requests-client.attempts, deadline-baseline.CLOCK.now(),
                      replacement_required=replacement_required, action=direct_action, base_version=direct_base,
                      repair_target=repair_target, reference_code=reference_code)
                  if direct else code_prompt(context, contract, state.program, state.version, feedback,
                             state.pending, client.max_requests-client.attempts,
                             remaining_seconds=seconds_left,
                             recovering=recovering, replacement_required=replacement_required,
                             continue_room=_continuation_room(client, deadline, reserve,
                                                              _unit_continuations(client, continuations, unit_cap)),
                             complete_first=first_complete, final=final, library=exposed, checked=lib_checked,
                             **({'reviewing': reviewed is not None} if loop_fix else {})))
        prompt = baseline.withhold_canaries(prompt, canaries)
        if first_complete and 'thin_complete_first' not in status:  # the unit's first code prompt: complete first?
            status['thin_complete_first'] = int(prompt.endswith(COMPLETE_FIRST))
        temperature = None
        fresh = None  # crash_redraft: the empty program this request's reply is admitted into, in place of `state`
        if redraft_due is not None:
            # crash_redraft: the same failure repeated, so this request asks for a fresh program instead of a patch: the
            # unit's first code prompt again, sampled, with CRASH_REDRAFT naming the failure (once per unit, with
            # CRASH_REDRAFT_LEFT requests left and room for the reply); a due one that does not fit is dropped.
            # redraft_fix: a fresh program only while the failing run left fewer than half of the required
            # deliverables with data (_redraft_empty); otherwise, and once the fresh one is spent, this request is
            # the ordinary one on CURRENT PROGRAM and its evidence, sampled, with CRASH_REPAIR naming the failure.
            if (redraft_fix and client.max_requests - client.attempts >= CRASH_REDRAFT_LEFT
                    and (status.get('thin_crash_redrafts') or not _redraft_empty(expected['files'], due_data))
                    and _repair_fits(client, deadline, reserve)):
                prompt = baseline.withhold_canaries(prompt + CRASH_REPAIR.format(failure=_crash_failure(redraft_due)),
                                                    canaries)
                temperature = temperature_for(RESAMPLE_KIND)
                status['thin_crash_repairs'] = status.get('thin_crash_repairs', 0) + 1
                status.setdefault('thin_crash_repair_at', client.attempts)
            elif (not status.get('thin_crash_redrafts') and client.max_requests - client.attempts >= CRASH_REDRAFT_LEFT
                    and _repair_fits(client, deadline, reserve)):
                fresh = ProgramVersion(expected_files=expected['files'])
                if _feature('input_fidelity'):
                    fresh.syntax_note = True
                prompt = baseline.withhold_canaries(code_prompt(
                    context, contract, fresh.program, fresh.version, first_feedback(fresh.version), fresh.pending,
                    client.max_requests - client.attempts, remaining_seconds=seconds_left,
                    continue_room=_continuation_room(client, deadline, reserve,
                                                     _unit_continuations(client, continuations, unit_cap)),
                    complete_first=first_complete, library=exposed) + CRASH_REDRAFT.format(
                        failure=_crash_failure(redraft_due)),
                    canaries)
                final = replace_only = None  # not a FINAL DELIVERY ask
                temperature = temperature_for(RESAMPLE_KIND)
                status.update(thin_crash_redrafts=1, thin_crash_redraft_at=client.attempts)
            redraft_due = None
        if resample and stuck >= RESAMPLE_AFTER and fresh is None:
            # A2: never once the best run left every required deliverable (any file when the contract names none):
            # such a unit can still be delivered as it is, and a sampled reply that runs clean would replace it.
            complete = best is not None and (not best[5].get('thin_missing_files') if expected['files']
                                              else bool(best[3]))
            if complete:
                status['thin_resample_guarded'] = status.get('thin_resample_guarded', 0) + 1
            else:
                temperature = temperature_for(RESAMPLE_KIND)
                status.setdefault('thin_resample_at', client.attempts)
                status['thin_resample_requests'] = status.get('thin_resample_requests', 0) + 1
        if loop_fix and copy_due and fresh is None and temperature is None:
            # loop_fix: after a copy stall the next repair request (the normal prompt, CURRENT PROGRAM kept) is sampled,
            # once per stall: A2 with RESAMPLE_AFTER 1, and A2's guard for a best run that left every deliverable
            if not (best is not None and (not best[5].get('thin_missing_files') if expected['files'] else bool(best[3]))):
                temperature = temperature_for(RESAMPLE_KIND)
                status['thin_copy_resamples'] = status.get('thin_copy_resamples', 0) + 1
        try:
            # The unit's first program is never held back, unless a version held back before its run runs instead.
            reply = _request(client, prompt, deadline, replay, 'thin_code', thinking['code'], reserve,
                             gate=prior is not None or withheld is not None, temperature=temperature)
        except HouseError as exc:
            record_stop(status, exc)  # names a stop before any execution too (thin_stop was None)
            if prior is None and withheld is None:
                raise
            status['thin_stop'] = exc.code
            ended = exc
            break
        aside = None  # crash_redraft: the loop's own state and asks while the reply is admitted into `fresh`
        if fresh is not None:
            aside = (copy.deepcopy(state), recovering, replacement_required, cut_runs)
            state.__dict__.update(fresh.__dict__, executed=state.executed)
            recovering = replacement_required = False
        if final is not None:  # S2-06: a FINAL DELIVERY ask was answered
            status['thin_final_delivery_replies'] = status.get('thin_final_delivery_replies', 0) + 1
            if 'thin_final_delivery_first' not in status:
                status['thin_final_delivery_first'] = final[0]
        elif replace_only is not None:  # the trigger held, but a REPLACE-only ask kept its own ACTION
            status['thin_final_delivery_replace_only'] = status.get('thin_final_delivery_replace_only', 0) + 1
        if loop:
            # S2-08 (README 7b): this code reply's code body was in a code reply at least two before it.
            body = _code_body(reply.get('content'))
            if body:
                digest = hashlib.sha256(body.encode()).hexdigest()
                if any(seen < code_replies - 1 for seen in bodies.get(digest, ())):
                    status['thin_loop_duplicate_bodies'] = status.get('thin_loop_duplicate_bodies', 0) + 1
                bodies.setdefault(digest, []).append(code_replies)
            code_replies += 1
        # With no continuation allowed (T1_THIN_MAX_CONTINUATIONS=0, or T1_THIN_MAX_CONTINUATIONS_UNIT spent) nothing
        # is skipped: _continue sends none.
        allowed = _unit_continuations(client, continuations, unit_cap)
        cut = free and reply.get('finish_reason') == 'length'
        held = _skip_continuation(state, reply) if cut and allowed else None
        # thin_continue_unit_capped counts the cut replies whose chain the unit cap, not the per-reply one, ended: with
        # budget spent, a reply that would otherwise get a continuation (A3a would not hold it, one could be sent now);
        # with budget lowered, a chain that used all of it and is still cut (_continue: exhausted).
        unit_capped = (cut and allowed == 0 and continuations > 0 and _continuable_now(client, reply, deadline, reserve)
                       and _skip_continuation(state, reply) is None)
        if held is not None:  # PRETEST A3a: _continue sends nothing for it; admission gets what it hands over
            status['thin_continue_skipped'] = status.get('thin_continue_skipped', 0) + 1
            reply = held
        if direct and replacement_required:
            # Natural recovery requests also consume this version's one chance.
            replacement_requested.add(direct_base)
        if free and reply.get('finish_reason') == 'length':
            chain = {}
            fix = dict(joint=True, copy=_copy_target(state, tried, copies, failures, reviewed)) if loop_fix else {}
            reply = _continue(client, prompt, reply, deadline, replay, thinking['code'], reserve, canaries, status,
                              limit=allowed, outcome=chain, stop=window_closed if clock else None, **fix)
            if chain.get('stopped'):  # clock_fix: no continuation started past the window (nor a run, below)
                status['thin_clock_continue_stops'] = status.get('thin_clock_continue_stops', 0) + 1
            if unit_capped or (0 < allowed < continuations and chain.get('exhausted')):
                status['thin_continue_unit_capped'] = status.get('thin_continue_unit_capped', 0) + 1
        raw = reply.get('content', '')
        payload_error = None
        append_signature = None
        append_bound = 0
        prefix_recovered = False
        effective_suffix_sha256 = None
        admission_accepted = False
        admission_reason = None
        if direct:
            try:
                if repair_target is not None and reply.get('finish_reason') != 'stop':
                    raise ValueError('repair_incomplete_reply')
                if direct_action == 'append_step' and reply.get('finish_reason') != 'stop':
                    raise ValueError('append_incomplete_reply')
                payload = _code_payload(raw, reply.get('finish_reason'))
                reference_code = payload
                if repair_target is not None:
                    payload = local_repair.apply(state.program, repair_target, payload)
                elif direct_action == 'append_step':
                    payload, append_bound = input_binding.rewrite(payload, input_bindings)
                    supplied_append = payload
                    payload = code_step.compose(state.program, payload, base_version=direct_base,
                                                executed_version=status.get('thin_output_version'),
                                                previous_steps=appended_steps)
                    effective_suffix = payload[len(state.program):]
                    append_signature = code_step.fingerprint(effective_suffix)
                    effective_suffix_sha256 = hashlib.sha256(effective_suffix.encode()).hexdigest()
                    prefix_recovered = (payload != state.program +
                        ('' if state.program.endswith('\n') else '\n') + supplied_append)
            except ValueError as exc:
                payload_error = str(exc)
        previous_version, previous_pending = state.version, state.pending
        if free:
            admitted = _admit(state, reply, status, recovering=recovering, replacement_required=replacement_required)
            changed, feedback = admitted['changed'], admitted['feedback']
            recovering, replacement_required = admitted['recovering'], admitted['replacement_required']
            if admitted['accepted'] and reply.get('finish_reason') == 'length':
                cut_runs = status['thin_executions']  # only a later run of the current version ends recovery
            if aside is not None:
                # crash_redraft: a fresh program replaces the loop's state and the normal loop runs it (`best` and
                # restore_best stay as they are, so the best run stands if it does worse); a reply that gives no
                # program puts the loop's own state, asks and evidence back. Progress is read against the old state.
                previous_version, previous_pending = aside[0].version, aside[0].pending
                if admitted['accepted'] and state.program:
                    status['thin_crash_redraft_accepted'] = True
                    if redraft_fix:  # its first run is read against the best run so far (execute)
                        redraft_base = best
                else:
                    state.__dict__.update(aside[0].__dict__)
                    recovering, replacement_required, cut_runs = aside[1:]
                    changed, feedback = False, CRASH_REDRAFT_KEPT
                    admitted = dict(admitted, accepted=False)  # nothing of this reply was admitted into `state`
        elif choices:
            try:
                proposed, source, unsupported = choices.transition(raw, recipe, state.version,
                    input_root=str(task_dir), output_root=str(out))
                if unsupported is not None:
                    status['thin_stop'] = 'choice_unsupported'
                    replay.put('choice-unsupported.txt', baseline.withhold_canaries(unsupported, canaries))
                    break
                result = state.apply(source, reply.get('finish_reason', 'stop'), 'replace', state.version)
                changed = result.changed
                if result.accepted:
                    recipe = proposed
                    replay.put('recipe.json', json.dumps(recipe, ensure_ascii=False, sort_keys=True))
                feedback = str(state.feedback or result.reason)
            except (ValueError, RecursionError) as exc:
                changed = False
                feedback = 'Choice rejected without executing or changing the program: ' + str(exc)[:2000]
        elif payload_error is not None:
            changed = False
            guidance = ('Return only the new executable statements after CURRENT PROGRAM; do not resubmit '
                        'or edit old calculations. New imports needed only by this addition are allowed. '
                        if payload_error.startswith('append_') and payload_error != 'append_incomplete_reply' else
                        'Return only Python code for the controller operation; reduce the current step if truncated.')
            feedback = ('Code response rejected: ' + payload_error + '. Program and outputs are unchanged. '
                        + guidance)
        else:  # direct: the controller chose the operation and its base before the request
            proposed = copy.deepcopy(state)
            result = proposed.apply(payload, reply.get('finish_reason', 'stop'),
                                    'replace' if direct_action == 'append_step' else direct_action, direct_base)
            bound = 0
            if result.accepted:
                proposed.program, bound = input_binding.rewrite(proposed.program, input_bindings)
                if proposed.ready and code_step.fingerprint(proposed.program) in tried_ast:
                    result = ApplyResult(False, False, 'already_executed_ast', state.version)
                    state.feedback = 'already_executed_ast; current code and obligations are retained.'
                else:
                    state.__dict__.update(proposed.__dict__)
                    status['thin_input_path_rewrites'] += bound + append_bound
                    if append_signature is not None:
                        appended_steps.add(append_signature)
            else:
                state.feedback = proposed.feedback
            admission_accepted = result.accepted
            admission_reason = result.reason
            changed = result.changed
            if result.accepted:
                replacement_required = False
            elif result.reason in MERGE_REASONS:
                replacement_required = True
            feedback = (str(state.feedback or result.reason) if result.accepted else
                        'Code response rejected: ' + result.reason + '. Program and outputs are unchanged. '
                        'Return only Python code for the controller operation shown below.')
        if changed and not direct:
            # Literal relocation only: keep the normalized source as the actual
            # current version, so replay, execution and subsequent BUNDLE bases agree.
            state.program, bound = input_binding.rewrite(state.program, input_bindings)
            status['thin_input_path_rewrites'] += bound
        dropped = []
        if changed and free and state.ready:
            # C: a function-level import that repeats a module-level one, after an earlier read of its name in that
            # function (UnboundLocalError), becomes `pass`; like the relocation above, the result is the version.
            state.program, dropped = _drop_shadow_imports(state.program)
            if dropped:
                status['thin_shadow_imports_removed'] = status.get('thin_shadow_imports_removed', 0) + len(dropped)
        if direct:
            replay.put('code-admission.json', json.dumps({
                'schema': 'direct-code-admission-v1', 'action': direct_action,
                'base_version': direct_base, 'version_after': state.version,
                'accepted': admission_accepted, 'reason': payload_error or admission_reason,
                'finish_reason': reply.get('finish_reason'),
                'prefix_recovered': bool(admission_accepted and prefix_recovered),
                'effective_suffix_sha256': effective_suffix_sha256 if admission_accepted else None,
                'raw_reply_sha256': hashlib.sha256(raw.encode()).hexdigest() if isinstance(raw, str) else None,
            }, sort_keys=True))
        changed = state.version != previous_version or state.pending != previous_pending
        if changed:  # this reply (after its continuations) made the version a run would execute
            version_fields = {k: client.diagnostics.get(k) for k in ('http_status', 'finish_reason')}
        complete = reply.get('finish_reason') != 'length'
        # loop_fix: while the OUTPUT REVIEW is open, a reply whose code admission rejected is no keep: only one
        # without code (or KEEP first) or an accepted resend of the reviewed version keeps the held run
        kept = (fg_held is not None and not changed and complete and fg_held == status.get('thin_output_version')
                and best is not None and (not loop_fix or reviewed is None or _keeps(reply, admitted)
                                          or admitted['accepted'] and state.version == reviewed[0]))
        # loop_fix: while no run followed the OUTPUT REVIEW hold, a complete reply without code (KEEP), or one that
        # makes the reviewed program again, keeps the reviewed run, even after a fix that was held back before its run
        late = bool(loop_fix and not kept and reviewed is not None and complete and best is not None
                    and reviewed[0] == status.get('thin_output_version')
                    and (admitted['accepted'] and state.version == reviewed[0] or _keeps(reply, admitted)))
        if kept or late:
            # S2-05: the first complete reply after the hold leaves the program as it was (no code, or the same
            # code): the model holds every finding right, so the held run, the last one, stands as final.
            if best[1] != (reviewed[0] if late else fg_held):
                restore_best()
            if late:
                withheld = None  # the held fix never runs: the reviewed run stands
                status['thin_review_late_keeps'] = 1
            status['thin_false_green_restored'] = status['thin_false_green_kept'] = True
            if status.get('thin_review_held') and not status.get('thin_false_green_blocks'):
                status['thin_review_kept'] = True
            status['thin_false_green_stop_findings'] = best[6]['findings']
            if fg_bool:
                status['thin_fg_bool_false_stop_findings'] = best[6]['bool']
            run_outcome = 'passed'
            status['thin_stop'] = 'internal_checks_passed'
            break
        if changed or complete:
            fg_held = None  # only the first complete reply after the hold, or one that changes the program, is read
        # S2-08 (loop switch): a ready version that already ran, byte for byte (ProgramVersion.executed already told
        # the reply so) or apart from comments and layout (`twin`, told here), is not run again; neither it nor a
        # return to any (version, pending) the loop held before is progress.
        revisit = twin = None
        if (noted or loop) and admitted['accepted'] and state.ready:
            if state.version in state.executed:
                # A1 (switch executed_note): an executed version is named with its result, never READY, since it does
                # not run again; S2-08 counts it on its own under the loop switch.
                if noted:
                    status['thin_executed_resends'] = status.get('thin_executed_resends', 0) + 1
                if loop:
                    status['thin_loop_executed_resends'] = status.get('thin_loop_executed_resends', 0) + 1
                # ProgramVersion named it already unless the input binding rewrite (an /app input path) or the C
                # rewrite changed its version after admission: then its feedback still says READY.
                feedback = feedback.replace(READY, executed_note(state.version, state.executed[state.version]), 1)
            elif loop:
                twin = executed_ast.get(_fingerprint(state.program))
                if twin is not None:
                    status['thin_loop_ast_skips'] = status.get('thin_loop_ast_skips', 0) + 1
                    feedback = feedback.replace(READY, executed_note(twin, state.executed[twin], layout=True), 1)
        if loop and changed and (state.version, state.pending) in held_states:
            revisit = state.version
            status['thin_loop_revisits'] = status.get('thin_loop_revisits', 0) + 1
            feedback += REVISIT.format(version=revisit)
        if loop:
            held_states.add((state.version, state.pending))
        if free:
            # A2: unchanged program and pending work, or a version that already ran (or, under the loop switch, its
            # twin): no new version to run. Counted with the switch off too; only `resample` reads it.
            stuck = 0 if changed and state.version not in tried and twin is None else stuck + 1
        if loop_fix:
            copy_due = bool(admitted['accepted'] and state.ready and (state.version in tried or twin is not None))
            if copy_due and state.version in tried and not reply.get('copied'):
                copies.add(state.version)  # a complete reply resent this executed version byte for byte
        if crash_redraft and admitted['accepted'] and state.ready:
            # crash_redraft: an accepted resend of a version whose run crashed (byte for byte, or its twin under the
            # loop switch) repeats that crash without a run
            repeat = crashes.get(state.version if twin is None else twin)
            if repeat is not None:
                redraft_due = repeat
                status['thin_crash_repeats'] = status.get('thin_crash_repeats', 0) + 1
                if redraft_fix:
                    due_data = crash_data.get(state.version if twin is None else twin)
        if execution_feedback:
            feedback += '\n' + execution_feedback
        status['thin_code_versions'] += bool(changed)
        progress = changed and revisit is None and twin is None
        no_progress = no_progress + 1 if direct else (0 if progress else no_progress + 1)
        if free and no_progress:
            # Temperature 0: an unchanged prompt repeats the same reply, so a stall changes the ask.
            feedback += (STALL_LARGE if len(state.program) > PROGRAM_BUDGET_CHARS else STALL).format(
                n=no_progress, limit=NO_PROGRESS_LIMIT, size=len(state.program))
        status['thin_pending_count'] = len(state.pending)
        ast_version = code_step.fingerprint(state.program) if direct and state.ready else None
        runnable = (state.ready and state.version not in tried and (not direct or ast_version not in tried_ast)
                    and twin is None)
        static = _execution_check(state.program) if free and runnable else ([], [], frozenset())
        if dropped and runnable:  # C: what the host changed, beside the other static notes of this version
            static = (static[0], static[1] + ['the host replaced the function-level import of ' + ', '.join(
                f'{name} in {function} (line {line})' for function, name, line in dropped[:_SHOWN])
                + ' with pass: each made its name local to that function while an earlier line there read it, and '
                'the module-level import binds the same object'], static[2])
        # A run would stop at the first such finding: ask for the fix instead, but only about findings not sent
        # back before, at most PREEXEC_BLOCKS times, and only while the next code request is sent and the
        # send-time gate finds room for its reply (_repair_fits; nothing ran this round, so `reserve` is that
        # request's reserve too). Otherwise it runs now. A held version that no later run follows still runs once
        # before solve ends (`withheld`, below the loop): a repair reply that is cut, not ready or never comes
        # does not cost the unit that run.
        if (static[2] - preexec_reported and status.get('thin_preexec_blocks', 0) < PREEXEC_BLOCKS
                and _repair_fits(client, deadline, reserve)):
            preexec_reported.update(static[2])
            withheld = (copy.deepcopy(state), static, version_fields)
            # Like a run (OB-B7), the hold ends recovery: its findings set the next ask, which may change any
            # complete definition, so it is the default ask (and OUTPUT FORMAT), not "complete the PENDING work".
            recovering = False
            status['thin_preexec_blocks'] = status.get('thin_preexec_blocks', 0) + 1
            feedback = ('Not executed: version ' + state.version + ' ' + '; '.join(static[0])
                        + '. The run would stop at the first of them. '
                        + ('Define or import each name, or use one that exists, then resend against CURRENT '
                           'VERSION.' if any(not key.startswith('shadow ') for key in static[2])
                           else 'Resend against CURRENT VERSION.')
                        + (' Advisory: ' + ', '.join(state.advisories) if state.advisories else '')
                        + ('\n' + execution_feedback if execution_feedback else ''))
        # clock_fix: a program whose reply has come in still runs past the window (its request was already spent);
        # the window stops new requests and continuations only
        elif (runnable and loop_fix and state.version not in entry_held
              and status.get('thin_no_entry_holds', 0) < NO_ENTRY_HOLDS and _no_entry(state.program)
              and _repair_fits(client, deadline, reserve)):
            # loop_fix: functions that nothing calls and no __main__ guard; held once per version like a static
            # finding (a resend of it runs, and so does a held version that no later run follows)
            entry_held.add(state.version)
            withheld = (copy.deepcopy(state), static, version_fields)
            recovering = False
            status['thin_no_entry_holds'] = status.get('thin_no_entry_holds', 0) + 1
            feedback = NO_ENTRY.format(version=state.version) + ('\n' + execution_feedback if execution_feedback else '')
        elif runnable and execute(static, ast_version, version_fields):
            break
        if direct:
            status['thin_no_progress'] = no_progress
        if no_progress >= (NO_PROGRESS_LIMIT if free else 2) and not (
                direct and replacement_required and state.version not in replacement_requested):
            status['thin_stop'] = 'no_progress'
            break
    if withheld is not None and clock and window_closed():
        # clock_fix: the held version's last-chance run does not start past the window either; the best run stands
        status['thin_clock_run_stops'] = status.get('thin_clock_run_stops', 0) + 1
        withheld = None
    if withheld is not None:
        # S0-11: the version last held back before its run, with no run since (the repair reply was cut, not
        # ready, or never came): it runs now, as it would have without the check, with the request fields of the
        # reply that made it. thin_stop keeps why the code requests ended (the loop ended at the request cap when
        # it names nothing); the run's own end is thin_preexec_fallback_outcome, and thin_preexec_fallback_run says
        # whether it started.
        if status['thin_stop'] is None:
            status['thin_stop'] = 'request_limit'
        loop_stop, runs_before = status['thin_stop'], status['thin_executions']
        state.__dict__.update(withheld[0].__dict__)
        try:
            execute(withheld[1], None, withheld[2], last=True)
        finally:
            status['thin_preexec_fallback_run'] = status['thin_executions'] > runs_before
            status['thin_preexec_fallback_outcome'] = run_outcome
            status['thin_stop'] = loop_stop
    if ended is not None and prior is None:
        raise ended
    if status.get('thin_sanity_at') is not None and 'thin_sanity_outcome' not in status:
        # S3-02, never a veto: no run passed after the sanity repair request, so the run it questioned stands
        # (restore_best below brings it back when a later, worse run replaced its outputs).
        # The repair's own end stays visible in thin_sanity_repair_stop (R3 attribution of failed repairs): the
        # stop code it reached, else whether the repair reply ran and did not pass, else the request cap, else
        # only the one request was spent (its reply never ran).
        repair_stop = status.get('thin_stop')
        if repair_stop in (None, 'internal_checks_passed'):
            ran = status.get('thin_executions', 0) > status.get('thin_sanity_runs', 0)
            repair_stop = ('repair_run_not_passed' if ran
                           else 'request_limit' if client.attempts >= client.max_requests else 'request_spent')
        status.update(thin_stop='internal_checks_passed', thin_sanity_outcome='kept',
                      thin_sanity_repair_stop=repair_stop)
    if status['thin_stop'] is None:
        status['thin_stop'] = 'request_limit'
    if (best is not None and (best[1] != status.get('thin_output_version') or rerun_on_disk)
            and not status.get('thin_internal_checks_passed')):
        # A later, worse run never replaces the best delivered outputs (its source and status return too).
        # A final run that passed the internal checks always stands. B: when the best is the asserts-off rerun of the
        # last run's own version (rerun_on_disk), its outputs are on disk already and its status returns.
        restore_best()

    first_issues = [0]  # second_draft: the delivered run's non-finite/empty output issues

    def second_draft():
        """One more program from the empty program, run on its own (second_draft): (outputs, program, version, run
        status, request fields) when it ran clean and passed the internal checks, else None. The outputs on disk are
        then the draft's own; second_drafts always puts back the ones it keeps."""
        alt = ProgramVersion(expected_files=expected['files'])
        if _feature('input_fidelity'):
            alt.syntax_note = True
        reserve = max(30, min(120, last_execution * 1.25 + 10))
        allowed = _unit_continuations(client, continuations, unit_cap)
        prompt = code_prompt(context, contract, alt.program, alt.version, first_feedback(alt.version), alt.pending,
                             client.max_requests - client.attempts,
                             remaining_seconds=run_room(deadline - baseline.CLOCK.now()),
                             continue_room=_continuation_room(client, deadline, reserve, allowed),
                             complete_first=first_complete, library=exposed) + SECOND_DRAFT_NOTE
        prompt = baseline.withhold_canaries(prompt, canaries)
        reply = _request(client, prompt, deadline, replay, 'thin_code', thinking['code'], reserve, gate=True,
                         temperature=temperature_for(RESAMPLE_KIND))
        if reply.get('finish_reason') == 'length':
            reply = _continue(client, prompt, reply, deadline, replay, thinking['code'], reserve, canaries, status,
                              limit=allowed, outcome={}, joint=loop_fix)
        request_fields = {k: client.diagnostics.get(k) for k in ('http_status', 'finish_reason')}
        if not _admit(alt, reply, {})['accepted'] or not alt.ready:
            return None
        alt.program = input_binding.rewrite(alt.program, input_bindings)[0]
        alt.program = _drop_shadow_imports(alt.program)[0]
        # The draft runs in the real output directory (the program writes its own paths): it starts only with room
        # for twice the delivered run plus a margin, and is capped to that, so the restore always has time.
        left = deadline - baseline.CLOCK.now()
        if left < 2 * last_execution + SECOND_DRAFT_MARGIN:
            raise HouseError('No room left for a draft run', kind='local_budget', code='generation_budget')
        run, date_report = {}, {}
        baseline._run_program(alt.program, out, deadline, canaries, run, clear=True, replay=replay,
                              input_root=task_dir, cap=max(1, min(2 * last_execution + 10, left - SECOND_DRAFT_MARGIN)),
                              retain_success_output=True, date_guard=date_report)
        files = _snapshot(out)
        if fg_norm and run.get('status') == 'executed_ungraded':
            files = _fg_normalize(instruction, out, run, files, integers={} if outputs else None)
        if (run.get('status') != 'executed_ungraded' or not files
                or any(not (out/name).is_file() for name in expected['files'])
                or audit.contract_flags(expected, out) or date_validation.flags(date_report, alt.program)
                or _canary_outputs(files, canaries)):
            return None
        # A draft votes only when it is as finished as the run it would replace: no unfinished-program evidence and
        # no non-finite or empty-output issue the delivered run does not have (two placeholders agree with each other).
        try:
            if stub_check.findings(alt.program, expected['files'], files, instruction=instruction, bound_parquet=clock):
                return None
            issues = [line for line in output_report.build(instruction, expected['files'], files, alt.program,
                                                           bound_parquet=clock).get('violations', [])
                      if 'non-finite' in line or 'empty' in line or 'NaN' in line]
            if len(issues) > first_issues[0]:
                return None
        except Exception:
            return None
        return files, alt.program, alt.version, run, request_fields

    def second_drafts():
        """second_draft (SECOND_DRAFTS): drafts from the empty program vote on the outputs of the run that ended the
        loop. That run's outputs, program and status stay unless two drafts state the same answer against them; any
        error in a draft keeps them too (the outputs on disk are put back in every case)."""
        nonlocal prior, fields
        status['thin_second_drafts'] = 0
        if room_deadline - baseline.CLOCK.now() < SECOND_DRAFT_ROOM or client.attempts >= client.max_requests:
            status['thin_second_draft_outcome'] = 'no_room'
            return
        first = _snapshot(out)
        names = tuple(expected['files'])
        clean, chosen, outcome, backup = [], None, 'no_room', None
        # a run the OUTPUT REVIEW changed keeps that correction: drafts written from the first prompt may only agree
        reviewed_fix = bool(status.get('thin_review_held') and not status.get('thin_review_kept'))
        try:
            first_issues[0] = len([line for line in output_report.build(instruction, expected['files'], first,
                                                                         prior or '', bound_parquet=clock
                                                                         ).get('violations', [])
                                   if 'non-finite' in line or 'empty' in line or 'NaN' in line])
            backup = Path(tempfile.mkdtemp(prefix='second-draft-'))  # the delivered outputs, on disk too
            baseline._restore(backup, first)
            for _ in range(SECOND_DRAFTS):
                if room_deadline - baseline.CLOCK.now() < SECOND_DRAFT_ROOM or client.attempts >= client.max_requests:
                    break
                status['thin_second_drafts'] += 1
                draft = second_draft()
                outcome = 'disagreed' if clean or draft is not None else 'unfinished'
                if draft is None:
                    continue
                agree, why = draft_compare.outputs_agree(first, draft[0], names)
                if agree:
                    outcome = 'agreed'
                    break
                status.setdefault('thin_second_draft_differs', why.split(':', 1)[0])
                if (not reviewed_fix and set(first) <= set(draft[0])
                        and any(draft_compare.outputs_agree(seen[0], draft[0], names)[0] for seen in clean)):
                    chosen, outcome = draft, 'replaced'
                    break
                clean.append(draft)
        except Exception as exc:  # a draft never costs the unit the run it delivered
            status['thin_second_draft_error'] = 'house_error' if isinstance(exc, HouseError) else 'other'
            if outcome == 'no_room' and status['thin_second_drafts']:
                outcome = 'unfinished'
        finally:
            try:
                if chosen is not None:
                    files, program, version, run, request_fields = chosen
                    baseline._restore(out, files)
                    prior, fields = program, {**fields, **request_fields}
                    status.update({k: run[k] for k in baseline.MEASURED if k in run})
                    status.setdefault('runs', []).extend(run.get('runs', ()))
                    status['program_sha256'] = status['thin_output_version'] = version
                    status['thin_second_draft_replaced'] = True
                else:
                    baseline._restore(out, first)
            except Exception:  # the first run's outputs, from the memory snapshot or the disk backup
                outcome, chosen = 'unfinished', None
                status['thin_second_draft_error'] = 'other'
                try:
                    baseline._restore(out, first)
                except Exception:
                    if backup is not None:
                        baseline._restore(out, {path.relative_to(backup).as_posix(): path.read_bytes()
                                                for path in backup.rglob('*') if path.is_file()})
            finally:
                if backup is not None:
                    shutil.rmtree(backup, ignore_errors=True)
            status['thin_second_draft_outcome'] = outcome

    if (free and _feature('second_draft') and prior is not None and ended is None and run_outcome == 'passed'
            and status.get('thin_stop') == 'internal_checks_passed' and status.get('thin_internal_checks_passed')):
        second_drafts()
    # S2-10: the best run's deliverables are clean whenever any run's were (_rank); a file that still holds an input
    # canary (a side file, or a deliverable of a run no clean one outranked) is deleted.
    removed = _remove_canary_outputs(out, canaries)
    if removed:
        status['thin_canary_files_removed'] = removed
        if 'thin_missing_files' in status:
            status['thin_missing_files'] = sum(not (out/name).is_file() for name in expected['files'])
            status['thin_deliverables_present'] = len(expected['files'])-status['thin_missing_files']
    if prior is None:
        status.update(status='generation_failed', failure_kind='model_output', failure_code='incomplete_code')
    else:
        replay.put('program.py', prior)
        replay.put('outputs.txt', baseline.withhold_canaries(audit.digest(out), canaries))
    status.update(baseline.output_canary_files(out, canaries))
    _snapshot(out)
    return fields


# Before a ready version runs (free loop, OB-B4): names it reads but never binds and attributes of installed
# modules that do not exist, found without running it. Kept after solve() so the lines above stay where they are.
_ANONYMOUS_SCOPES = frozenset({'lambda', 'genexpr', 'listcomp', 'setcomp', 'dictcomp'})
_DYNAMIC_NAMESPACE = frozenset({'globals', 'exec', 'eval', '__builtins__'})  # may bind any module name
_DEFINED_TESTS = frozenset({'globals', 'locals', 'vars', 'dir'})  # 'name' in locals(): a defined-name test
_CATCH_ALL = frozenset({'Exception', 'BaseException'})
# Installed packages whose attributes may be looked up in this process. The program only picks names inside
# them: it never chooses what is imported beyond a public submodule of one of these roots.
_ATTRIBUTE_ROOTS = frozenset({
    'numpy', 'scipy', 'pandas', 'qfkit', 'math', 'cmath', 'statistics', 'random', 'datetime', 'json', 'csv', 'os',
    'itertools', 'functools', 'collections', 're', 'decimal', 'fractions', 'bisect', 'heapq', 'time', 'calendar',
    'string', 'operator', 'pathlib', 'io', 'copy', 'warnings'})
_SHOWN = 5
PREEXEC_BLOCKS = 2  # runs held back per unit: a wrong finding costs at most this many code requests
PREEXEC_REPLY_ROOM = 60  # seconds a repair reply needs at least (s14c code replies: p95 24 s, max 26.5 s)
PREEXEC_MARGIN = 5  # seconds between _repair_fits reading the clock and the repair request's own send-time gate
_COMPREHENSIONS = {ast.ListComp: 'listcomp', ast.SetComp: 'setcomp', ast.DictComp: 'dictcomp',
                   ast.GeneratorExp: 'genexpr'}
_COMPAT_PROVIDED = None  # {module: names} the runtime compatibility layer adds; read once from its source


def _sanity_hold(client, expected, out, task_dir, profile, program, status, flags, room):
    """S3-02 (T1_THIN_FEATURES sanity, sanity_constant): whether a run that passed the internal checks is held for
    one repair request because its outputs break a generic sanity property (sanity.py). Only the unit's first such
    run is checked, and only with room for the repair (`room()`: the free loop, a request and time left); the
    finding lines join `flags`, so the run's feedback names the property. The run's rank and outputs stay those of a
    passing run: a repair that does not pass leaves them standing (solve's `best`, restore_best). A later passing
    run records whether the property still fails (thin_sanity_outcome). With every sanity feature off nothing is
    read or computed here."""
    if not status.get('thin_internal_checks_passed'):
        return False
    from . import sanity
    if not sanity.enabled_checks():
        return False
    if status.get('thin_sanity_at') is not None:
        if 'thin_sanity_outcome' not in status:
            found = sanity.wired_findings(expected, out, task_dir, profile, program)
            status['thin_sanity_outcome'] = 'passed_flagged' if found else 'passed_clean'
        return False
    found = sanity.wired_findings(expected, out, task_dir, profile, program) if room() else []
    if not found:
        return False
    status.update(thin_sanity_at=client.attempts, thin_sanity_runs=status.get('thin_executions', 0),
                  thin_sanity_check=sanity.label(found))
    flags.extend(sanity.describe(finding) for finding in found)
    return True


def _reply_room(client):
    """Seconds a repair reply needs: the send-time gate's predicted code reply (_predicted_need, once this unit
    has a measured reply), and never less than PREEXEC_REPLY_ROOM or this unit's slowest code reply so far."""
    seconds = [record.get('duration_sec') for record in client.diagnostics.get('requests', [])
               if record.get('kind') == 'thin_code' and record.get('reply_returned')]
    need = _predicted_need(client, 'thin_code')
    return max([PREEXEC_REPLY_ROOM] + [value for value in seconds if type(value) in (int, float)]
               + ([need] if need is not None else []))


def _repair_fits(client, deadline, reserve):
    """Whether a version held back before its run gets its repair: _request's own checks on the next code
    request (a request left, more than 1 s before `reserve`) and its send-time gate (_held_for_time) on the room
    the repair reply needs (_reply_room), with PREEXEC_MARGIN seconds fewer than are left now (the request reads
    the clock again when it is sent). The held version still runs before solve ends when the repair does not
    replace it (solve's `withheld`); this check keeps that last-chance run the exception."""
    from . import baseline
    left = deadline - baseline.CLOCK.now() - PREEXEC_MARGIN
    return (client.attempts < client.max_requests and left > reserve + 1
            and not _held_for_time(_reply_room(client), left, reserve))


def _continuation_room(client, deadline, reserve, limit):
    """Whether a cut reply to the next code request could still be continued once (S0-16 B5: only then does
    code_prompt say that a cut REPLACE is continued): continuations are allowed (`limit`,
    T1_THIN_MAX_CONTINUATIONS > 0), a request is left after the code request, and after the code reply
    (_reply_room) the time before `reserve` still holds what _continue requires (CONTINUE_FLOOR) and the
    send-time gate's predicted continuation (_held_for_time, once this unit has a measured reply)."""
    from . import baseline
    if limit <= 0 or client.max_requests - client.attempts < 2:
        return False
    after = deadline - baseline.CLOCK.now() - _reply_room(client)
    return after >= reserve + CONTINUE_FLOOR and not _held_for_time(_predicted_need(client, 'thin_continue'),
                                                                    after, reserve)


def _compat_tables(source):
    """{module: names} that the compatibility source (compat.py) adds to a module after its import: the keys
    and entries of the module-level tables its _COMPAT_AFTER hook for that module uses, directly or through
    the functions it calls. Read from the source itself, so this check and the runtime cannot drift."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return {}
    values = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            values[node.targets[0].id] = node.value
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            values[node.name] = node
    hooks, provided = values.get('_COMPAT_AFTER'), {}
    if not isinstance(hooks, ast.Dict):
        return {}
    text = lambda items: {item.value for item in items
                          if isinstance(item, ast.Constant) and isinstance(item.value, str)}
    for key, hook in zip(hooks.keys, hooks.values):
        if not (isinstance(key, ast.Constant) and isinstance(key.value, str)):
            continue
        names, seen, todo = set(), set(), [hook]
        while todo:
            for sub in ast.walk(todo.pop()):
                if not isinstance(sub, ast.Name) or sub.id not in values or sub.id in seen:
                    continue
                seen.add(sub.id)
                value = values[sub.id]
                if isinstance(value, ast.Dict):
                    names |= text(value.keys)
                elif isinstance(value, (ast.Tuple, ast.List, ast.Set)):
                    names |= text(value.elts)
                elif isinstance(value, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                    todo.append(value)
        provided[key.value] = frozenset(names)
    return provided


def _compat_provided():
    """{module: names} every run has although the installed release may lack them: baseline.worker_site puts
    compat.py (baseline.COMPAT_SITE) into each generated program's sitecustomize."""
    global _COMPAT_PROVIDED
    if _COMPAT_PROVIDED is None:
        from . import baseline
        _COMPAT_PROVIDED = _compat_tables(baseline.COMPAT_SITE or '')
    return _COMPAT_PROVIDED


def _handled(node, parents, errors):
    """Whether node runs inside the body of a try, in its own scope, whose handlers catch one of errors or any
    exception (a bare except, Exception, BaseException)."""
    child = node
    while child in parents:
        parent = parents[child]
        if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)) and any(child is s for s in parent.body):
            return False
        if isinstance(parent, ast.Lambda) and child is parent.body:
            return False
        if isinstance(parent, (ast.Try, getattr(ast, 'TryStar', ast.Try))) and any(child is s for s in parent.body):
            for handler in parent.handlers:
                if handler.type is None:
                    return True
                kinds = handler.type.elts if isinstance(handler.type, ast.Tuple) else [handler.type]
                for kind in kinds:
                    name = (kind.id if isinstance(kind, ast.Name) else
                            kind.attr if isinstance(kind, ast.Attribute) else '')
                    if name in errors or name in _CATCH_ALL:
                        return True
        child = parent
    return False


def _reached_tables(top, decorated):
    """Symbol tables of code that runs: the module; class bodies, lambdas and generator expressions inside it;
    each function that code refers to by name (a module-level one only by a global read of that name, not by a
    local or parameter that shares it; a method once its class is referred to), and decorated functions."""
    reached, grew = {top.get_id(): top}, True
    while grew:
        grew = False
        referenced, global_reads = set(), set()
        for table in reached.values():
            for symbol in table.get_symbols():
                if symbol.is_referenced():
                    referenced.add(symbol.get_name())
                    if table.get_type() == 'module' or symbol.is_global():
                        global_reads.add(symbol.get_name())
        for table in list(reached.values()):
            for child in table.get_children():
                if child.get_id() in reached:
                    continue
                name = child.get_name()
                if (child.get_type() == 'function' and name not in _ANONYMOUS_SCOPES
                        and (name, child.get_lineno()) not in decorated
                        and (table.get_name() if table.get_type() == 'class' else name)
                        not in (global_reads if table.get_type() == 'module' else referenced)):
                    continue
                reached[child.get_id()] = child
                grew = True
    return list(reached.values())


def _names_in(node):
    """Identifiers a test mentions: its names and attribute names."""
    return {sub.id if isinstance(sub, ast.Name) else sub.attr
            for sub in ast.walk(node) if isinstance(sub, (ast.Name, ast.Attribute))}


def _branch_of(node, parents, test):
    """Whether node sits in a branch of an if or conditional expression, in its own scope, whose test the
    predicate accepts (test(names) on the identifiers of that if's test)."""
    child = node
    while child in parents:
        parent = parents[child]
        if isinstance(parent, (ast.If, ast.IfExp)) and child is not parent.test and test(_names_in(parent.test)):
            return True
        if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)) and any(child is s for s in parent.body):
            return False
        if isinstance(parent, ast.Lambda) and child is parent.body:
            return False
        child = parent
    return False


def _runs(node, parents, scopes):
    """Whether node is in code that runs: not in an except handler (it runs only after a failure), under
    `if TYPE_CHECKING:` or in a branch a constant test never takes (`if False:`, `if 0:`, the else of `if True:`,
    `while False:`), and its innermost def or lambda is one of scopes ((name, line) of the reached function tables),
    or it has none."""
    child = node
    while child in parents:
        parent = parents[child]
        if isinstance(parent, ast.ExceptHandler) and any(child is s for s in parent.body):
            return False
        if (isinstance(parent, (ast.If, ast.While)) and isinstance(parent.test, ast.Constant)
                and any(child is s for s in (parent.orelse if parent.test.value else parent.body))
                and not (isinstance(parent, ast.While) and parent.test.value)):
            return False
        if (isinstance(parent, ast.If) and any(child is s for s in parent.body)
                and (isinstance(parent.test, ast.Name) and parent.test.id == 'TYPE_CHECKING'
                     or isinstance(parent.test, ast.Attribute) and parent.test.attr == 'TYPE_CHECKING')):
            return False
        if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)) and any(child is s for s in parent.body):
            return (parent.name, parent.lineno) in scopes
        if isinstance(parent, ast.Lambda) and child is parent.body:
            return ('lambda', parent.lineno) in scopes
        child = parent
    return True


def _unevaluated(node, parents, future):
    """Whether node is inside an annotation Python never evaluates: a variable annotation in a function body,
    and, under `from __future__ import annotations` (future), any annotation."""
    child = node
    while child in parents:
        parent = parents[child]
        if future and (isinstance(parent, ast.arg) and child is parent.annotation
                       or isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)) and child is parent.returns):
            return True
        if isinstance(parent, ast.AnnAssign) and child is parent.annotation:
            if future:
                return True
            scope = parent
            while scope in parents and not isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda,
                                                                ast.ClassDef, ast.Module)):
                scope = parents[scope]
            return isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef))
        child = parent
    return False


def _future_annotations(tree):
    return any(isinstance(node, ast.ImportFrom) and node.module == '__future__'
               and any(item.name == 'annotations' for item in node.names) for node in tree.body)


def _defined_test(node, parents):
    """Whether a load of a name sits under a test that the name is defined ('name' in locals(), globals(), vars()
    or dir()) in an if, a conditional expression, a while or an earlier operand of and/or, in its own scope."""
    child = node
    while child in parents:
        parent = parents[child]
        tests = ([parent.test] if isinstance(parent, (ast.If, ast.IfExp, ast.While)) and child is not parent.test
                 else parent.values[:[v is child for v in parent.values].index(True)]
                 if isinstance(parent, ast.BoolOp) else [])
        for test in tests:
            for compare in ast.walk(test):
                if (isinstance(compare, ast.Compare) and isinstance(compare.left, ast.Constant)
                        and compare.left.value == node.id
                        and any(isinstance(op, (ast.In, ast.NotIn)) for op in compare.ops)
                        and any(isinstance(right, ast.Call) and isinstance(right.func, ast.Name)
                                and right.func.id in _DEFINED_TESTS for right in compare.comparators)):
                    return True
        if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)) and any(child is s for s in parent.body):
            return False
        if isinstance(parent, ast.Lambda) and child is parent.body:
            return False
        child = parent
    return False


def _dynamic_namespace(tree, parents):
    """Whether the program may bind module names out of sight: globals(), exec, eval, __builtins__, the builtins
    module, sys.modules, or locals()/vars() without arguments at module level (there they are globals())."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if node.id in _DYNAMIC_NAMESPACE:
                return True
            call = parents.get(node)
            if (node.id in ('locals', 'vars') and isinstance(call, ast.Call) and call.func is node
                    and not call.args and not call.keywords and _runs(node, parents, ())):
                return True
        elif isinstance(node, (ast.Import, ast.ImportFrom)) and any(
                (getattr(node, 'module', None) or item.name).split('.')[0] == 'builtins' for item in node.names):
            return True
        elif (isinstance(node, ast.Attribute) and node.attr == 'modules'
              and isinstance(node.value, ast.Name) and node.value.id == 'sys'):
            return True
    return False


def _scope_tables(top):
    """{(kind, name, line): tables} for every def, lambda, class and non-inlined comprehension table below top."""
    index, todo = {}, list(top.get_children())
    while todo:
        table = todo.pop()
        todo.extend(table.get_children())
        for kind in ('function', 'class'):  # plain strings: 3.13 returns an enum that hashes differently
            if table.get_type() == kind:
                index.setdefault((kind, table.get_name(), table.get_lineno()), []).append(table)
    return index


def _global_read(node, parents, index):
    """Whether this load of a name looks it up among the module's names: at module level, or in a def, lambda,
    class or generator expression whose own table has the name as a global. A load of a local, a parameter, a
    closure variable or a comprehension's target is not (the name may still be free elsewhere in the program);
    a scope whose table cannot be found is not judged."""
    name, below, child = node.id, None, node
    while child in parents:
        parent = parents[child]
        key = None
        if type(parent) in _COMPREHENSIONS:
            first = parent.generators[0]
            if child is first and below is first.iter:  # the first iterable is evaluated outside
                below, child = child, parent
                continue
            key = ('function', _COMPREHENSIONS[type(parent)], parent.lineno)
            if key not in index:  # inlined (3.12+ list, set, dict): the enclosing table holds its names
                below, child = child, parent
                continue
        elif (isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
              and any(child is statement for statement in parent.body)):
            key = ('class' if isinstance(parent, ast.ClassDef) else 'function', parent.name, parent.lineno)
        elif isinstance(parent, ast.Lambda) and child is parent.body:
            key = ('function', 'lambda', parent.lineno)
        if key is not None:
            try:
                return bool(index.get(key)) and all(table.lookup(name).is_global() for table in index[key])
            except KeyError:
                return False
        below, child = child, parent
    return True


def _comprehension_walrus(tree, parents):
    """Names an assignment expression binds inside a comprehension that no def, lambda or class encloses: PEP 572
    binds them in the module, while Python 3.13's symtable (an inlined comprehension, PEP 709) lists them as free."""
    bound = set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.NamedExpr) and isinstance(node.target, ast.Name)):
            continue
        child, comprehension = node, False
        while child in parents and comprehension is not None:
            parent = parents[child]
            if type(parent) in _COMPREHENSIONS:
                comprehension = True
            elif isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
                comprehension = None
            child = parent
        if comprehension:
            bound.add(node.target.id)
    return bound


def _unbound_reads(program, tree, parents, scopes, index):
    """(blocking, other): sorted names the program reads but never binds. blocking: a load that looks the name
    up among the module's names (_global_read) at a place that runs, outside a try that handles NameError and
    outside a defined-name test. other: the rest, and every such name when the program may bind names
    dynamically (_dynamic_namespace). A walrus target of a module-level comprehension is bound."""
    names = _module_names(program)
    if names is None or not names[1] - _comprehension_walrus(tree, parents):
        return [], []
    free = names[1] - _comprehension_walrus(tree, parents)
    if _dynamic_namespace(tree, parents):
        return [], sorted(free)
    future = _future_annotations(tree)
    blocking = sorted({node.id for node in ast.walk(tree)
                       if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id in free
                       and _global_read(node, parents, index)
                       and _runs(node, parents, scopes) and not _handled(node, parents, {'NameError'})
                       and not _defined_test(node, parents) and not _unevaluated(node, parents, future)})
    return blocking, sorted(free - set(blocking))


def _attribute(owner, name):
    """(True, value) when owner has attribute name, (False, None) when it certainly has not, (None, None) when
    that is unknown. A module is asked directly (its own __getattr__ may load a submodule); a public submodule
    of it counts, and so does a name the runtime compatibility layer adds to it (_compat_provided: it exists
    in every run, its value is unknown here). Any other object is read statically, so no property or
    __getattr__ code runs; an object whose type answers attributes dynamically is unknown."""
    import sys
    import types
    import warnings
    if isinstance(owner, types.ModuleType):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                return True, getattr(owner, name)
        except AttributeError:
            pass
        except Exception:
            return None, None
        if name in _compat_provided().get(getattr(owner, '__name__', None), ()):
            return None, None
        full = owner.__name__ + '.' + name
        try:
            if full in sys.modules:
                return True, sys.modules[full]
            if not hasattr(owner, '__path__') or importlib.util.find_spec(full) is None:
                return False, None
            if name.startswith('_'):
                return None, None
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                return True, importlib.import_module(full)
        except Exception:
            return None, None
    try:
        return True, inspect.getattr_static(owner, name)
    except AttributeError:
        pass
    except Exception:
        return None, None
    kind = type(owner)
    if (any('__getattr__' in vars(base) for base in kind.__mro__)
            or kind.__getattribute__ not in (object.__getattribute__, type.__getattribute__)):
        return None, None
    return False, None


def _trusted(dotted, cache):
    """The installed object a dotted import target names under a trusted root, else None (also when unknown)."""
    if dotted not in cache:
        parts, value = dotted.split('.'), None
        if parts[0] in _ATTRIBUTE_ROOTS:
            try:
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    value = importlib.import_module(parts[0])
                for part in parts[1:]:
                    found, value = _attribute(value, part)
                    if not found:
                        value = None
                        break
            except Exception:
                value = None
        cache[dotted] = value
    return cache[dotted]


def _version_branch(node, parents):
    """Whether node is in a branch chosen by a version test (an if whose test names a version), where an
    attribute missing from the installed release is the other release's spelling."""
    return _branch_of(node, parents, lambda names: any('version' in name.lower() for name in names))


def _missing_attributes(tree, parents, scopes):
    """(blocking, other): sorted 'module.name' a program imports from, or reads as an attribute of, a trusted
    installed module (or a class or object bound from one) that does not exist there. Only an alias that every
    binding in the program gives the same import, and that nothing rebinds or assigns into, is followed (none
    next to a star import); a module is followed down the chain, any other object one attribute deep. Reads
    that may never run or be evaluated, sit in a branch chosen by a version test or in a try that handles the
    error, or follow a hasattr/getattr test are in other."""
    import types
    targets, rebound, tested, star = {}, set(), set(), False  # tested: (alias, attr, ...) a hasattr call names
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for item in node.names:
                name = item.asname or item.name.split('.')[0]
                targets.setdefault(name, set()).add(item.name if item.asname else name)
        elif isinstance(node, ast.ImportFrom):
            for item in node.names:
                if item.name == '*':
                    star = True
                elif node.level or not node.module:
                    rebound.add(item.asname or item.name)
                else:
                    targets.setdefault(item.asname or item.name, set()).add(node.module + '.' + item.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            rebound.add(node.id)
        elif isinstance(node, ast.arg):
            rebound.add(node.arg)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            rebound.add(node.name)
        elif isinstance(node, (ast.ExceptHandler, ast.MatchAs, ast.MatchStar)) and node.name:
            rebound.add(node.name)
        elif isinstance(node, ast.MatchMapping) and node.rest:
            rebound.add(node.rest)
        elif isinstance(node, ast.Attribute) and isinstance(node.ctx, (ast.Store, ast.Del)):
            root = node.value
            while isinstance(root, ast.Attribute):
                root = root.value
            if isinstance(root, ast.Name):
                rebound.add(root.id)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.args:
            if node.func.id in ('setattr', 'delattr') and isinstance(node.args[0], ast.Name):
                rebound.add(node.args[0].id)
            elif (node.func.id in ('hasattr', 'getattr') and len(node.args) > 1
                  and isinstance(node.args[1], ast.Constant) and isinstance(node.args[1].value, str)):
                owner = node.args[0]
                path = []
                while isinstance(owner, ast.Attribute):
                    path.append(owner.attr)
                    owner = owner.value
                if isinstance(owner, ast.Name):
                    tested.add((owner.id,) + tuple(reversed(path)) + (node.args[1].value,))
    aliases = {} if star else {
        name: next(iter(found)) for name, found in targets.items()
        if len(found) == 1 and name not in rebound and next(iter(found)).split('.')[0] in _ATTRIBUTE_ROOTS}
    cache, blocking, other, future = {}, set(), set(), _future_annotations(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and not node.level and node.module and (
                node.module.split('.')[0] in _ATTRIBUTE_ROOTS):
            owner = _trusted(node.module, cache)
            for item in node.names:
                if owner is None or item.name == '*' or _attribute(owner, item.name)[0] is not False:
                    continue
                held = (not _runs(node, parents, scopes) or _version_branch(node, parents)
                        or _handled(node, parents, {'ImportError', 'ModuleNotFoundError'}))
                (other if held else blocking).add(node.module + '.' + item.name)
        elif (isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load)
              and not (isinstance(parents.get(node), ast.Attribute) and parents[node].value is node)):
            chain, root = [], node
            while isinstance(root, ast.Attribute):
                chain.append(root.attr)
                root = root.value
            if not isinstance(root, ast.Name) or root.id not in aliases:
                continue
            owner, dotted, path = _trusted(aliases[root.id], cache), aliases[root.id], (root.id,)
            for attr in reversed(chain):
                path += (attr,)
                if owner is None or path in tested:
                    break
                found, value = _attribute(owner, attr)
                dotted += '.' + attr
                if found is False:
                    held = (not _runs(node, parents, scopes) or _unevaluated(node, parents, future)
                            or _version_branch(node, parents) or _handled(node, parents, {'AttributeError'}))
                    (other if held else blocking).add(dotted)
                if not found or not isinstance(owner, types.ModuleType):
                    break
                owner = value
    return sorted(blocking), sorted(other - blocking)


_SCOPE_NODES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef, ast.ListComp, ast.SetComp,
                ast.DictComp, ast.GeneratorExp)


def _own_scope(body):
    """The nodes of a scope's statements (a def's body, or a module's) that belong to that scope, in source order: a
    nested def, lambda, class or comprehension is listed (a def or class binds its name there) but not entered, so
    its body, decorators and defaults are skipped (skipping only misses findings)."""
    todo = list(body)[::-1]
    while todo:
        node = todo.pop()
        yield node
        if not isinstance(node, _SCOPE_NODES):
            todo.extend(reversed(list(ast.iter_child_nodes(node))))


def _import_specs(node):
    """[(bound name, spec)] of an import statement; spec is the whole import (dotted name, or module, name, level), so
    two bindings with equal specs bind the same object."""
    if isinstance(node, ast.Import):
        return [(item.asname or item.name.split('.')[0], ('import', item.name, bool(item.asname)))
                for item in node.names]
    return [(item.asname or item.name, ('from', node.module, item.name, node.level)) for item in node.names
            if item.name != '*']


def _binding_position(node, parents):
    """Where a store of a name takes effect in its scope's source order: after the value an assignment, walrus,
    for-loop iterable or with-item evaluates first; at the statement's end for an import; else where it stands."""
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        return (node.end_lineno or node.lineno, node.end_col_offset or 0)
    child = node
    while child in parents:
        parent = parents[child]
        if isinstance(parent, ast.NamedExpr) and child is parent.target:
            return (parent.value.end_lineno, parent.value.end_col_offset)
        if isinstance(parent, (ast.For, ast.AsyncFor)) and child is parent.target:
            return (parent.iter.end_lineno, parent.iter.end_col_offset)
        if isinstance(parent, ast.withitem) and child is parent.optional_vars:
            return (parent.context_expr.end_lineno, parent.context_expr.end_col_offset)
        if isinstance(parent, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            return (parent.end_lineno, parent.end_col_offset)
        if isinstance(parent, ast.stmt):
            break
        child = parent
    return (node.lineno, node.col_offset)


def _in_loop_with(node, others, parents, function):
    """Whether node and one of `others` sit in the repeated part of the same for or while loop of `function` (its
    target, test or body, not a for's iterable or an else clause, which run once), where a later binding may run on
    an earlier pass, before a read above it."""
    def loops(start):
        found, child = set(), start
        while child in parents and child is not function:
            parent = parents[child]
            if (isinstance(parent, (ast.For, ast.AsyncFor, ast.While)) and child is not getattr(parent, 'iter', None)
                    and not any(child is item for item in parent.orelse)):
                found.add(parent)
            child = parent
        return found
    mine = loops(node)
    return bool(mine) and any(loops(other) & mine for other in others)


def _straight_line(node, parents, function):
    """C: whether node sits in straight-line code of `function`'s own body, which runs whenever the function runs up to
    it: not inside the body or else of an if, while or for (a for's iterable runs once and counts), an except handler,
    a try's else, a match case, a later operand of `and`/`or`, a branch of a conditional expression or an assert's
    message, each of which may never run. with-bodies, try-bodies and finally-bodies count as straight-line."""
    child = node
    while child in parents and child is not function:
        parent = parents[child]
        if isinstance(parent, (ast.If, ast.While, ast.IfExp)) and child is not parent.test:
            return False
        if isinstance(parent, (ast.For, ast.AsyncFor)) and child is not parent.iter:
            return False
        if isinstance(parent, (ast.ExceptHandler, getattr(ast, 'match_case', ast.ExceptHandler))):
            return False
        if (isinstance(parent, (ast.Try, getattr(ast, 'TryStar', ast.Try)))
                and any(child is item for item in parent.orelse)):
            return False
        if isinstance(parent, ast.BoolOp) and child is not parent.values[0]:
            return False
        if isinstance(parent, ast.Assert) and child is parent.msg:
            return False
        child = parent
    return child is function


def _shadowed_imports(tree, parents, scopes):
    """C: names a module binds by an import that a function also binds (any store, del, import or except-as; no
    global or nonlocal), with a read in that function's own scope before every such binding in source order (stores
    take effect as _binding_position says). Python makes the name local in the whole function, so that read raises
    UnboundLocalError. [(function, name, read line, first binding line, blocking, redundant)]: blocking when such a
    read runs (_runs; never judged without `scopes`), sits in the function's straight-line code (_straight_line: a read
    under an if, a loop body, a handler or a short-circuit may never run, so it is only a note), is not under a handler
    of NameError or a catch-all and shares no loop with a binding (a later pass may read what an earlier one bound);
    reads in unevaluated annotations do not count. redundant: every binding of the name in the function is the same import as every module-level binding of
    it, and nothing else binds it at module level (removing those imports changes no value: _drop_shadow_imports)."""
    module_specs, module_other = {}, set()
    for node in _own_scope(tree.body):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for name, spec in _import_specs(node):
                module_specs.setdefault(name, set()).add(spec)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            module_other.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            module_other.add(node.name)
    imported = {name for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))
                for name, _ in _import_specs(node)}
    future, found = _future_annotations(tree), []
    for function in ast.walk(tree):
        if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        declared = {name for node in ast.walk(function) if isinstance(node, (ast.Global, ast.Nonlocal))
                    for name in node.names}
        params = {arg.arg for arg in ast.walk(function.args) if isinstance(arg, ast.arg)}
        binds, reads = {}, {}
        for node in _own_scope(function.body):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for name, spec in _import_specs(node):
                    binds.setdefault(name, []).append((_binding_position(node, parents), node, spec))
            elif isinstance(node, ast.Name):
                if isinstance(node.ctx, ast.Load):
                    reads.setdefault(node.id, []).append(node)
                else:
                    binds.setdefault(node.id, []).append((_binding_position(node, parents), node, None))
                    if isinstance(parents.get(node), ast.AugAssign):  # `x += 1` reads x first
                        reads.setdefault(node.id, []).append(node)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node is not function:
                binds.setdefault(node.name, []).append(((node.lineno, node.col_offset), node, None))
            elif isinstance(node, ast.ExceptHandler) and node.name:
                binds.setdefault(node.name, []).append(((node.lineno, node.col_offset), node, None))
        for name in sorted(imported & set(binds) & set(reads) - declared - params):
            first = min(position for position, _, _ in binds[name])
            early = [node for node in reads[name] if (node.lineno, node.col_offset) < first
                     and not _unevaluated(node, parents, future)]
            if not early:
                continue
            read = min(early, key=lambda node: (node.lineno, node.col_offset))
            blocking = scopes is not None and any(
                _runs(node, parents, scopes) and _straight_line(node, parents, function)
                and not _handled(node, parents, {'NameError', 'UnboundLocalError'})
                and not _in_loop_with(node, [b for _, b, _ in binds[name]], parents, function) for node in early)
            specs = {spec for _, _, spec in binds[name]}
            redundant = (None not in specs and name not in module_other and len(module_specs.get(name, ())) == 1
                         and specs == module_specs[name])
            found.append((function.name, name, read.lineno, first[0], blocking, redundant))
    return found


def _drop_shadow_imports(program):
    """C: `program` with each function-level import statement that _shadowed_imports finds redundant replaced by
    `pass` in place (its other lines left blank, so every line number stays), when every name the statement binds is
    such a redundant shadow in that function and it stands alone on its lines; and [(function, name, line)] of what
    was removed. The result must parse and leave none of those findings, else `program` is returned unchanged with
    []. Never raises."""
    try:
        tree = ast.parse(program)
        parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
        redundant = {(function, name) for function, name, _, _, _, ok in _shadowed_imports(tree, parents, None) if ok}
        if not redundant:
            return program, []
        lines = program.split('\n')
        cuts, removed = [], []
        for function in ast.walk(tree):
            if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for node in _own_scope(function.body):
                if not isinstance(node, (ast.Import, ast.ImportFrom)):
                    continue
                names = [name for name, _ in _import_specs(node)]
                if not names or any((function.name, name) not in redundant for name in names):
                    continue
                first, last = node.lineno - 1, (node.end_lineno or node.lineno) - 1
                before = lines[first][:node.col_offset]
                after = lines[last][node.end_col_offset:]
                if before.strip() or after.strip() and not after.strip().startswith('#'):
                    continue  # shares its line with other code (`;`): left to the blocking finding
                cuts.append((first, last, node.col_offset, node.end_col_offset))
                removed.extend((function.name, name, node.lineno) for name in names)
        for first, last, start, end in sorted(cuts, reverse=True):
            tail = lines[last][end:]
            lines[first] = lines[first][:start] + 'pass' + (tail if first == last else '')
            for index in range(first + 1, last + 1):
                lines[index] = tail if index == last else ''
        result = '\n'.join(lines)
        check = ast.parse(result)
        check_parents = {child: node for node in ast.walk(check) for child in ast.iter_child_nodes(node)}
        left = {(f, n) for f, n, _, _, _, _ in _shadowed_imports(check, check_parents, None)}
        if not removed or left & {(f, n) for f, n, _ in removed}:
            return program, []
        compile(result, '<shadow-imports>', 'exec', dont_inherit=True)
        return result, removed
    except Exception:  # a rewrite only; a program it cannot read stays as it is
        return program, []


def _execution_check(program):
    """(blocking, notes, found) for a ready program, found without running it: short reasons it would stop at
    the first use of a name it never binds or of an installed attribute that does not exist, compatibility names
    included, or at a read of an import name that a function makes local before binding it (C, _shadowed_imports)
    (blocking); the same findings where the program may never reach them or handles the error (notes);
    one key per blocking finding (found), so the loop asks about each at most once. ([], [], frozenset()) when
    the program cannot be analysed; this check never raises."""
    import symtable
    try:
        tree = ast.parse(program)
        parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
        decorated = {(node.name, node.lineno) for node in ast.walk(tree)
                     if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                     and node.decorator_list}
        top = symtable.symtable(program, '<program>', 'exec')
        tables = _reached_tables(top, decorated)
        scopes = {(table.get_name(), table.get_lineno()) for table in tables if table.get_type() == 'function'}
        names, other_names = _unbound_reads(program, tree, parents, scopes, _scope_tables(top))
        attributes, other_attributes = _missing_attributes(tree, parents, scopes)
        shadows = _shadowed_imports(tree, parents, scopes)
    except Exception:  # analysis only; a program it cannot read runs as before
        return [], [], frozenset()
    shown = lambda values: ', '.join(values[:_SHOWN]) + (', ...' if len(values) > _SHOWN else '')
    where = lambda rows: shown([f'{name} in {function} (read at line {read}, bound at line {bound})'
                                for function, name, read, bound, _, _ in rows])
    blocking, notes = [], []
    if names:
        blocking.append('reads names it never defines: ' + shown(names))
    if attributes:
        blocking.append('reads attributes or imports names that do not exist: ' + shown(attributes))
    stops = [row for row in shadows if row[4]]
    if stops:
        blocking.append('makes a module-level import name local to a function (it is assigned, imported or deleted '
                        'there), so a read of it earlier in that function fails with UnboundLocalError: '
                        + where(stops) + '; give that local variable another name or drop its function-level import')
    if other_names:
        notes.append('names it never defines, read only where the run may not reach or evaluate them, where it '
                     'handles their absence or under a dynamic namespace: ' + shown(other_names))
    if other_attributes:
        notes.append('attributes or imported names that do not exist, read only where the run may not reach or '
                     'where it handles their absence: ' + shown(other_attributes))
    if len(stops) < len(shadows):
        notes.append('module-level import names a function makes local and reads before binding them, only where '
                     'the run may not reach that read (a branch, a loop body, a handler or a short-circuit), where it '
                     'handles the error or inside a loop that binds them: '
                     + where([row for row in shadows if not row[4]]))
    return blocking, notes, frozenset(['name ' + name for name in names] + ['attribute ' + a for a in attributes]
                                      + [f'shadow {function} {name}' for function, name, _, _, _, _ in stops])


def _restore_early(out, kept, status):
    """S3-03b OS-4: the best run's outputs back on disk right after a run that did not become the best. Never raises:
    a failed restore is counted and left to the restore at the end of solve, which writes them again."""
    from . import baseline
    try:
        baseline._restore(out, kept)
    except Exception:
        status['thin_early_restore_failures'] = status.get('thin_early_restore_failures', 0) + 1
        return
    status['thin_best_restored_early'] = status.get('thin_best_restored_early', 0) + 1


# crash_redraft (switch, 10-28): at temperature 0 the model often resends the program that crashed, or a patch that
# leaves the same crash, until the free loop stops on no_progress, usually with no deliverable. A run's failure
# signature is its exception type with the function and source of its failing line (_crash_signature). When two
# consecutive runs end with the same one, or an accepted reply resends a version whose run crashed, the next code
# request asks for a fresh program instead of a patch, built as second_draft builds a draft (code_prompt on an empty
# ProgramVersion, first_feedback, sampled at temperature_for(RESAMPLE_KIND)) with CRASH_REDRAFT naming the failure: its
# type and failing line, no other program text. A reply that gives a program replaces the loop's state and the normal
# loop runs it (`best` and restore_best stay as they are); otherwise the loop's own state comes back
# (CRASH_REDRAFT_KEPT). Once per unit (status thin_crash_redrafts), with at least CRASH_REDRAFT_LEFT requests left and
# room for the reply (_repair_fits).
CRASH_REDRAFT_LEFT = 3
CRASH_LINE_CHARS = 160  # characters of the failing line CRASH_REDRAFT names
CRASH_REDRAFT = ('\n\nFRESH PROGRAM: the previous program kept ending in the same failure ({failure}) and patching it '
                 'did not change that, so it is set aside and CURRENT PROGRAM is empty. Write a complete new program '
                 'from scratch that carries out the TASK and avoids this failure; do not build the failing step the '
                 'same way again.')
CRASH_REDRAFT_KEPT = 'The reply to the fresh-program request gave no program; CURRENT PROGRAM is unchanged.'
# redraft_fix (switch, 10-28; it changes crash_redraft only, and only with both on): a fresh program had replaced one
# whose run wrote every deliverable but one and needed a one-line fix (13f: the fresh one wrote empty books and was
# delivered). (1) A repeated failure gets a fresh program only while the failing run left fewer than half of the
# required deliverables with data (_redraft_empty; _with_data: a CSV with a data row, JSON that is not empty);
# otherwise, and once the fresh one is spent, the next request is the ordinary one on CURRENT PROGRAM with its run's
# evidence, sampled at temperature_for(RESAMPLE_KIND), with CRASH_REPAIR naming the failure (status
# thin_crash_repairs, thin_crash_repair_at). (2) When the fresh program's first run does not pass and leaves fewer
# required deliverables with data than the best run before it (as many: a lower _rank), that run's program and version
# are CURRENT PROGRAM again for the following requests, with REDRAFT_RESTORED in the feedback; `best` and restore_best
# stay as they are (thin_crash_redraft_restored). (3) A run without a crash signature has an output signature
# (_output_signature): the same one on consecutive runs, byte-identical required outputs with an unchanged status and
# unchanged host-detected issues, is a repeat too (a stuck output, thin_crash_stuck_outputs).
STUCK_OUTPUT = '<same outputs>'  # the first item of an output signature, never an exception type
STUCK_FAILURE = 'the same required outputs, byte for byte, with the same host-detected issues'
CRASH_REPAIR = ('\n\nREPAIR AGAIN: the previous program kept ending in the same failure ({failure}) and the last change '
                'did not alter that. Keep what works in CURRENT PROGRAM and fix that step in a different way; code '
                'that already ran is not run again.')
REDRAFT_RESTORED = ('Host note: the run above was of a fresh program (version {fresh}); it left {fresh_data} required '
                    'deliverables with data against {base_data} for version {base}, so CURRENT PROGRAM is version '
                    '{base} again{failure}: repair that program instead of starting over.')


DATA_PREFIX = 1 << 20  # redraft_fix: the bytes of a CSV _has_data reads, and the largest JSON it parses


def _has_data(name, data):
    """redraft_fix: whether one output file holds data: a CSV or TSV with a non-blank row after its first line (in its
    first DATA_PREFIX bytes), JSON that parses to anything but null or an empty string, list or object (a larger one
    always does), any other file a non-whitespace byte. Never raises."""
    if not isinstance(data, bytes) or re.search(rb'\S', data) is None:
        return False
    lower = name.lower()
    if lower.endswith(('.csv', '.tsv')):
        lines = data[:DATA_PREFIX].decode('utf-8-sig', errors='replace').splitlines()
        return sum(1 for line in lines if line.strip(' \t,;|')) > 1
    if lower.endswith('.json') and len(data) <= DATA_PREFIX:
        try:
            value = json.loads(data.decode('utf-8-sig', errors='replace'))
        except (ValueError, RecursionError):
            return False
        return value is not None and not (isinstance(value, (str, list, dict)) and not value)
    return True


def _with_data(expected_files, files):
    """redraft_fix: how many of the required deliverables (every file the run left when the contract names none) hold
    data in `files` (a run's outputs, {relative path: bytes})."""
    names = tuple(expected_files) if expected_files else tuple(files)
    return sum(1 for name in names if _has_data(name, files.get(name)))


def _redraft_empty(expected_files, count):
    """redraft_fix: whether a repeated failure gets a fresh program: its run left fewer than half of the required
    deliverables with data (none, when the contract names no file). An unknown count (None) is none."""
    count = count or 0
    return 2 * count < len(expected_files) if expected_files else not count


def _output_signature(expected_files, files, status, issues):
    """redraft_fix: the output signature of a run without a crash signature: (STUCK_OUTPUT, the sha256 of each
    required deliverable or None when it is missing (every file the run left when the contract names none), its
    status and exit code with the host-detected issues); None when the run left no required deliverable."""
    names = sorted(expected_files) if expected_files else sorted(files)
    outputs = tuple((name, hashlib.sha256(files[name]).hexdigest() if name in files else None) for name in names)
    if not any(digest for _, digest in outputs):
        return None
    return STUCK_OUTPUT, outputs, (status.get('status'), status.get('exit_code')) + tuple(str(line) for line in issues)


def _crash_signature(program, status):
    """crash_redraft: (exception type or None, enclosing function, stripped source) of the failing line of a run that
    failed with a traceback through `program` (execution_diagnostics, baseline._failing_statement), else None: a clean
    exit, a timeout, or a failure without a line of the program. Never raises."""
    from . import baseline
    try:
        diagnostic = status.get('execution_diagnostics')
        if status.get('status') != 'execution_failed' or not isinstance(diagnostic, dict):
            return None
        function, source = baseline._failing_statement(program, diagnostic.get('generated_line'))
        kind = diagnostic.get('exception_type')
        return (kind if isinstance(kind, str) and kind else None, function, source) if source else None
    except Exception:
        return None


def _crash_failure(signature):
    """The failure CRASH_REDRAFT names for a _crash_signature: its type and failing line (at most CRASH_LINE_CHARS),
    with the function that holds it."""
    kind, function, source = signature
    if kind == STUCK_OUTPUT:  # redraft_fix: an output signature
        return STUCK_FAILURE
    line = source if len(source) <= CRASH_LINE_CHARS else source[:CRASH_LINE_CHARS] + '...'
    return ((kind or 'an exception') + ' at `' + line + '`'
            + (' in ' + function if function and function != '<module>' else ''))


# S3-03b repair feedback (switch repair_feedback; the free loop only): host-built sentences added after a failed run's
# evidence. Each is one line of its own that starts with 'Host note:', never with a prompt marker, and carries no
# text of the run's output beyond a line number and the host's own input paths.
OWN_CHECK = ('Host note: the run stopped at an assert or raise statement of CURRENT PROGRAM itself (line {line}), not '
             'inside a library. Distinguish TASK requirements, facts derived from the input, and expectations invented '
             'by the program. Correct the actual calculation or unsupported expectation; do not remove a required '
             'check to suppress failure. Write the required deliverables before the final assertions, so that a '
             'failed final check does not leave them missing.')
SAME_EXCEPTION = ('Host note: this run ended with the same exception line as the previous run, so take a different '
                  'approach to the failing step instead of adjusting the same code again.')
MOUNTED_AT = 'Host note: the TASK path {original} is not mounted in this run; that input file is at {runtime}.'
MOUNTED_SHOWN = 2  # input files named per note at most


class RepairNotes:
    """The repair notes of one free-loop solve (switch repair_feedback), added after a failed run's evidence:
    RPL-8 OWN_CHECK when the run stopped in an assert or raise statement of the program itself (_own_check_line);
    RQ-9/RPL-7 SAME_EXCEPTION when its exception line (baseline._exception_line) is the one the previous run ended
    with (a line without a message, such as a bare AssertionError, only at the same failing source line); RPL-5
    MOUNTED_AT when a FileNotFoundError names exactly the original path of a declared input file (an input_bindings
    key), with that file's runtime path. status thin_repair_notes_* count the runs each note followed."""

    def __init__(self, bindings):
        self.files = dict(bindings)
        self.last = ('', '')  # the previous run's exception line ('' when it did not fail with one), failing source

    def after_run(self, program, status, tail, report):
        """The notes for this run ('' for none), one '\\n'-led line each."""
        from . import baseline
        failed = status.get('status') == 'execution_failed'
        line = baseline._exception_line(report, tail) if failed else ''
        diagnostic = status.get('execution_diagnostics') if failed else None
        number = diagnostic.get('generated_line') if isinstance(diagnostic, dict) else None
        rows = program.split('\n') if isinstance(program, str) else []
        source = rows[number - 1].strip() if type(number) is int and 0 < number <= len(rows) else ''
        notes = []
        own = _own_check_line(program, diagnostic)
        if own is not None:
            notes.append(OWN_CHECK.format(line=own))
            status['thin_repair_notes_own_check'] = status.get('thin_repair_notes_own_check', 0) + 1
        if line and line == self.last[0] and (line.partition(':')[2].strip() or source == self.last[1]):
            notes.append(SAME_EXCEPTION)
            status['thin_repair_notes_same_exception'] = status.get('thin_repair_notes_same_exception', 0) + 1
        mounted = _mounted_paths(line, self.files)
        if mounted:
            notes += [MOUNTED_AT.format(original=name, runtime=self.files[name]) for name in mounted[:MOUNTED_SHOWN]]
            status['thin_repair_notes_mounted'] = status.get('thin_repair_notes_mounted', 0) + 1
        self.last = (line, source)
        return ''.join('\n' + note for note in notes)


def _innermost_statement(program, line):
    """(statement, parents): the innermost statement of `program` whose lines hold `line` (None when there is none),
    and its tree's parent map; (None, None) when the program does not parse."""
    try:
        tree = ast.parse(program)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return None, None
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}

    def depth(node):
        count = 0
        while node in parents:
            node, count = parents[node], count + 1
        return count
    held = [node for node in ast.walk(tree)
            if isinstance(node, ast.stmt) and node.lineno <= line <= (node.end_lineno or node.lineno)]
    return max(held, key=depth, default=None), parents


def _own_assert_line(program, diagnostic):
    """B: the line of a failed run's innermost generated frame (execution_diagnostics generated_line, from a readable
    log) when the run raised AssertionError and the innermost statement there is an assert statement of the program,
    the statement PYTHONOPTIMIZE skips; else None (a `raise AssertionError(...)` or a library's assert is not one)."""
    if not isinstance(diagnostic, dict) or diagnostic.get('log_readable') is not True:
        return None
    line = diagnostic.get('generated_line')
    if type(line) is not int or diagnostic.get('exception_type') != 'AssertionError':
        return None
    statement, _ = _innermost_statement(program, line)
    return line if isinstance(statement, ast.Assert) else None


def _own_check_line(program, diagnostic):
    """The line of a failed run's innermost generated frame (execution_diagnostics generated_line) when the innermost
    statement there is the program's own check, else None: an assert statement that raised AssertionError, or a raise
    statement with an exception, outside any except handler of its function, that raised the class it names (or, when
    the log names no known class, a class that is not a builtin or known library one: the program's own)."""
    from . import baseline
    if not isinstance(diagnostic, dict) or diagnostic.get('log_readable') is not True:
        return None
    line, kind = diagnostic.get('generated_line'), diagnostic.get('exception_type')
    if type(line) is not int:
        return None
    statement, parents = _innermost_statement(program, line)
    if statement is None:
        return None
    if isinstance(statement, ast.Assert):
        return line if kind == 'AssertionError' else None
    if not isinstance(statement, ast.Raise) or statement.exc is None:
        return None
    child = statement
    while child in parents and not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
        if isinstance(parents[child], ast.ExceptHandler):
            return None  # a raise in a handler re-raises or wraps an error that happened elsewhere
        child = parents[child]
    target = statement.exc.func if isinstance(statement.exc, ast.Call) else statement.exc
    name = target.id if isinstance(target, ast.Name) else target.attr if isinstance(target, ast.Attribute) else None
    if name is None:
        return None
    if kind is not None:
        return line if kind == name else None
    known = baseline._BUILTIN_ERRORS | {error.rsplit('.', 1)[-1].lstrip('_') for error in baseline.THIRD_PARTY_ERRORS}
    return line if name not in known else None


def _mounted_paths(line, files):
    """The original paths of declared input files (`files`, input_bindings as a dict) that a FileNotFoundError
    exception line names exactly, as a whole path (not a longer one), in input_bindings order; a file mounted at
    its original path is never one (its note would name the same path twice)."""
    match = re.match(r'([A-Za-z_][\w.]*)\s*:', line or '')
    if match is None or match[1].rsplit('.', 1)[-1] != 'FileNotFoundError':
        return []
    message = line[match.end():]
    return [name for name in files if isinstance(name, str) and name and files[name] != name
            and re.search(r'(?<![\w./-])' + re.escape(name) + r'(?![\w/-]|\.\w)', message)]


# loop_fix (switch, 10-28; real runs: 13f): what the solver itself did wrong around the model's replies.
# 1. A continuation whose code before a bare closing fence did not join the retained text (its first line indented
#    deeper than the block it continues) was dropped, and the empty reading after the fence then ended the program
#    inside main() with no call. With the switch the joint is re-indented (_joints) and accepted only when the stitch
#    parses; otherwise the cut stays pending, as for any unparsed continuation.
# 2. A ready program that defines functions but never calls one from the top level and has no __main__ guard is not
#    run (NO_ENTRY), once per version and at most NO_ENTRY_HOLDS times per unit, while its repair fits.
# 3. A copy stall (an accepted ready resend of a version that ran, or its twin) makes the next request sampled; a cut
#    reply that reproduces an executed program verbatim, once a complete reply already resent it, stops its
#    continuations past the failing line and 80% of the program (_copy_target), as that identical resend.
# 4. The OUTPUT REVIEW hold stays open until a run follows it (solve's `reviewed`), not only until the program changes:
#    a fix held back before its run left KEEP offered (the prompt read the review text still in the feedback), a KEEP
#    then ran the held fix, and a resend of the reviewed program was neither kept nor run. While it is open a complete
#    reply without code (_keeps) or one that makes the reviewed program again keeps the reviewed run (the held fix is
#    dropped), and KEEP is offered exactly then. A cut REPLACE whose text so far is the whole reviewed program is that
#    program resent: KEEP at once, without its continuations.
NO_ENTRY_HOLDS = 2
NO_ENTRY = ('Not executed: version {version} defines functions, but no top-level statement calls one of them and there '
            'is no `if __name__ == "__main__":` guard, so a run would only import and define names and write nothing '
            '(the reply may have been cut before its entry point). Complete the cut function if any and end the '
            'program with a complete `if __name__ == "__main__":` guard that calls the main function; resend against '
            'CURRENT VERSION.')
COPY_SHARE = (4, 5)  # copy stop: more than this share of the executed program's lines reproduced (and the failing line)
_DEFINITIONS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def _indent(line):
    return len(line) - len(line.lstrip(' \t'))


def _trimmed(lines):
    """`lines` without trailing blank lines."""
    end = len(lines)
    while end and not lines[end - 1].strip():
        end -= 1
    return lines[:end]


def _code_like(lines):
    """loop_fix: whether the text before a bare fence of a continuation is code: an indented line, or lines that parse."""
    rows = [line for line in lines if line.strip()]
    if any(line[:1] in (' ', '\t') for line in rows):
        return True
    try:
        return bool(rows) and bool(_parse('\n'.join(rows) + '\n'))
    except (SyntaxError, ValueError, RecursionError, MemoryError, OverflowError):
        return False


def _joints(before, body):
    """loop_fix: `body` (continuation lines) with its first code line, and the deeper lines right after it, moved to
    each indentation the retained code allows after its last code line: that line's own level, then each enclosing
    level out to 0, then one level deeper (a block opener's body). Python decides: the caller keeps a variant only
    when the stitch parses. Variants equal to `body` are left out."""
    lines = before.split('\n')[:-1]
    fence = next((i for i, line in enumerate(lines) if line.strip().startswith('```')), None)
    if fence is None:
        return []
    code = [line for line in lines[fence + 1:] if line.strip() and not line.lstrip().startswith('#')]
    rows = [i for i, line in enumerate(body) if line.strip()]
    if not code or not rows or '\t' in body[rows[0]][:_indent(body[rows[0]])]:
        return []
    level = _indent(code[-1])
    levels = [level]
    for line in reversed(code[:-1]):
        if _indent(line) < level:
            level = _indent(line)
            levels.append(level)
            if not level:
                break
    levels += [0, _indent(code[-1]) + 4]
    first = rows[0]
    width = _indent(body[first])
    end = next((i for i in rows[1:] if _indent(body[i]) <= width), len(body))
    variants = []
    for target in dict.fromkeys(levels):
        if target == width:
            continue
        moved = list(body)
        for i in range(first, end):
            if body[i].strip():
                moved[i] = ' ' * (_indent(body[i]) + target - width) + body[i].lstrip(' \t')
        variants.append(moved)
    return variants


def _copy_target(state, tried, copies, failures, reviewed):
    """loop_fix: what a cut reply to this request is compared with (_copied), else None, as (program, the version its
    REPLACE names, lines to exceed, whether it answers the review). While the OUTPUT REVIEW is open (`reviewed`:
    solve's (version, program)) a REPLACE of CURRENT VERSION must hold the whole reviewed program (item 4); else
    CURRENT PROGRAM, when it is ready, already ran and a complete reply already resent it byte for byte, must be
    reproduced in full too (item 3), so no change in its tail is taken for a resend."""
    if reviewed is not None:
        return reviewed[1], state.version, len(_trimmed(reviewed[1].split('\n'))) - 1, True
    if not state.ready or state.version not in tried:
        return None
    rows = len(_trimmed(state.program.split('\n')))
    if state.version in copies:
        # the whole program, as for the review: a change in its tail (past any share of it) is never assumed away
        return state.program, state.version, rows - 1, False
    return None


def _copied(text, copy):
    """loop_fix: whether a cut reply's retained text is a REPLACE of the target version whose code lines so far are the
    first lines of its program, verbatim, and more than the target's count of them."""
    program, version, beyond, _ = copy
    header = re.match(HEADER, text)
    if header is None or header[1] != 'REPLACE' or header[2] != version:
        return False
    lines = text.split('\n')[:-1]
    fence = next((i for i, line in enumerate(lines) if line.strip().startswith('```')), None)
    if fence is None:
        return False
    code, rows = _trimmed(lines[fence + 1:]), _trimmed(program.split('\n'))
    return len(code) > beyond and code == rows[:len(code)]


def _copy_reply(reply, copy, status):
    """loop_fix: the reply as the identical resend it is: a complete REPLACE of the target program (marked copied)."""
    program, version, _, review = copy
    if review:
        status['thin_review_copies'] = status.get('thin_review_copies', 0) + 1
    else:
        status['thin_copy_stops'] = status.get('thin_copy_stops', 0) + 1
    return dict(reply, content='REPLACE ' + version + '\n```python\n' + program
                + ('' if program.endswith('\n') else '\n') + '```', finish_reason='stop', copied=True)


def _keeps(reply, admitted):
    """loop_fix: whether a complete reply to the OUTPUT REVIEW keeps the run: no action header and no code fence (or
    KEEP first), so admission took no program from it."""
    raw = reply.get('content')
    return admitted['route'] == 'missing_header' and (
        not isinstance(raw, str) or '```' not in raw or re.match(r'\s*\**KEEP\b', raw) is not None)


def _no_entry(program):
    """loop_fix: whether a program defines a top-level function but never enters its own code: no top-level
    `if __name__ ...` guard, no decorated definition, and no top-level statement outside a def or class that reads a
    name the program defines at the top level (False when it does not parse)."""
    try:
        body = ast.parse(program).body
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return False
    defined = {node.name for node in body if isinstance(node, _DEFINITIONS)}
    if not any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) for node in body):
        return False
    for node in body:
        if isinstance(node, _DEFINITIONS):
            if node.decorator_list:
                return False
            continue
        if isinstance(node, ast.If) and any(isinstance(item, ast.Name) and item.id == '__name__'
                                            for item in ast.walk(node.test)):
            return False
        stack = [node]
        while stack:
            item = stack.pop()
            if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Load) and item.id in defined:
                return False
            stack.extend(child for child in ast.iter_child_nodes(item)
                         if not isinstance(child, _DEFINITIONS + (ast.Lambda,)))
    return True
