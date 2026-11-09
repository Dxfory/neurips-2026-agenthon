#!/usr/bin/env python3
"""Stack 10 phase-1 probe (spec §12, §14 phase 1), local only and never in the image: the T1_S10 notes stage alone on
dev units, with no program generated or run and no container.

For each unit and repeat it stages the unit's participant view from the local source cache (as public_eval does;
checks are never read), builds first_prompt as the container would see it (inputs under /input, outputs under
/app/output), runs s10_triage, s10_decide, ADOPT (s10_adopt, and the cards under T1_CARDS) and s10_refine against the
House named by MODEL_ENDPOINT, MODEL_NAME and MODEL_TOKEN (T1_EVAL_BACKEND and T1_EVAL_THROTTLE apply as in the
solver), and writes to <out>/<unit>/r<k>/: the replay files (prompt.txt, request-*.json, s10-*.txt), notes.txt (the
notes block a code request would carry), plan.txt (stack9's fallback plan, when DECIDE left none) and status.json;
<out>/summary.jsonl gets one line per unit-repeat. --prompt-only builds the prompt and triage alone, without any
request. answer_words counts each answer's words as its word cap means them: the whitespace-separated tokens less
those that are only separators or punctuation (|, ->) and the bare IDs (D3, S1:); answer_split_words is the plain
whitespace split, for comparison.

Held-out units (holdout.json) and ids outside its dev list are refused before any file of the unit is opened. The
switches are arm B's (spec §10: the image's, T1_CONTRACT2, T1_S10 and T1_S10_TAGS); a switch already set in the
environment keeps its value, except T1_S10, which is always on here."""
import argparse
import contextlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from t1_phase1 import baseline  # noqa: E402
from t1_phase1.house import HouseClient, HouseConfig, HouseError  # noqa: E402
from t1_phase1.staging import stage_unit  # noqa: E402
import contract2_check  # noqa: E402

TASK_DIR, OUT_DIR = '/input', '/app/output'  # where the container sees the participant view and its outputs
LIBRARY_ROOT = str(ROOT / 'qfkit')  # the library the image installs
ARM_B = {name: '1' for name in ('T1_PLAN', 'T1_REVIEW', 'T1_FRESH', 'T1_SELFCHECK', 'T1_STAGED', 'T1_PRECHECK',
                                'T1_LIBHINT', 'T1_CONTRACT', 'T1_ACCEPT', 'T1_ADOPT', 'T1_RAISEGUARD', 'T1_CONTRACT2',
                                'T1_S10_TAGS')}
RECORD_KEYS = ('kind', 'attempt', 'http_status', 'completion_tokens', 'prompt_tokens', 'reasoning_budget',
               'reasoning_chars', 'content_chars', 'finish_reason', 'duration_sec', 'timeout_sec', 'provider')


class ProbeReplay(baseline.Replay):
    """The solver's replay into one unit-repeat's directory, keeping the S10 answers for the word counts."""

    def __init__(self, root):
        super().__init__(env={'T1_REPLAY_DIR': str(root)})
        self.texts = {}

    def put(self, name, data):
        if name.startswith(('s10-', 's11-')):
            self.texts[name] = data if isinstance(data, str) else bytes(data).decode('utf-8', 'replace')
        super().put(name, data)


@contextlib.contextmanager
def library_path():
    """The library findable while the probe runs, as in the image: the solver reads its catalog and sources, never
    imports it."""
    sys.path.insert(0, LIBRARY_ROOT)
    try:
        yield
    finally:
        sys.path.remove(LIBRARY_ROOT)


def check_unit(unit, dev, held):
    """unit when it is a dev unit; a held-out or unknown id stops the probe before any of its files is opened."""
    if unit in held:
        raise SystemExit(f'refused: {unit} is held out')
    if unit not in dev:
        raise SystemExit(f'refused: {unit} is not a dev unit of holdout.json')
    return unit


def arm_b(env=None):
    """Arm B's switches (spec §10) where the environment does not set them, and T1_S10 on."""
    env = os.environ if env is None else env
    for name, value in ARM_B.items():
        env.setdefault(name, value)
    env['T1_S10'] = '1'
    return env


def stage(unit, root, source=None):
    """The unit's participant view, staged under root (a temporary directory) as public_eval stages it."""
    units = Path(source or contract2_check.SOURCE / contract2_check.COMMIT / 'units')
    target = Path(root) / unit
    stage_unit(units / unit, target)
    return target.resolve()


def task_prompt(task_dir):
    """(prompt, instruction, profile, canaries, status) as solve builds them before its first request, with the
    paths the container sees: the participant view under TASK_DIR, the outputs under OUT_DIR. Under T1_S10 the
    in-program self-check is off, so the prompt has none."""
    task_dir, status = Path(task_dir).resolve(), {}
    profile, canaries, hints = baseline._profile(task_dir, status)
    text = (task_dir / 'instruction.md').read_bytes().decode('utf-8', errors='replace')
    instruction = ''.join(line for line in text.splitlines(keepends=True)
                          if not any(guid in line.lower() for guid in canaries))
    prompt = baseline.first_prompt(task_dir, Path(OUT_DIR), instruction, profile, hints, canaries, status,
                                   selfcheck=False)
    return prompt.replace(str(task_dir), TASK_DIR), instruction, profile, canaries, status


def notes_stage(client, prompt, instruction, profile, canaries, status, replay, budget=baseline.STAGE_CAP):
    """The T1_S10 notes stage as solve will run it (spec §12): triage, DECIDE (and DECIDE B, merged, under
    T1_S11_UNION), ADOPT and the cards, CHECK and STATE, on a fresh unit clock (budget seconds, less solve's 20 s
    final reserve). Returns (s10, notes)."""
    baseline.CLOCK.reset()
    deadline = baseline.CLOCK.now() + budget - 20
    pace = baseline.Pace()
    client.replay = replay
    triage = baseline.s10_triage(instruction, profile)
    s10 = baseline.s10_decide(client, prompt, triage, deadline, status, pace, canaries=canaries, replay=replay)
    if baseline.s11_union_on():
        s10 = baseline.s11_decide_b(client, prompt, triage, s10, deadline, status, pace, instruction=instruction,
                                    canaries=canaries, replay=replay)
    required_prompt, required = baseline.s10_adopt(prompt, s10, instruction, status)
    if baseline.cards_on():
        required_prompt = baseline.with_cards(required_prompt, required, instruction, status)
    s10 = baseline.s10_refine(client, required_prompt, instruction, s10, deadline, status, pace, canaries=canaries,
                              replay=replay)
    return s10, baseline.s10_notes(s10, tags=baseline.s10_tags_on())


def records(client):
    """Each request's numbers and enums, with its estimated reasoning tokens (the breaker's estimate)."""
    rows = []
    for record in client.diagnostics.get('requests', []):
        row = {key: record[key] for key in RECORD_KEYS if key in record}
        row['reasoning_tokens_est'] = round(baseline._s10_reasoning(record))
        rows.append(row)
    return rows


_BARE_ID = re.compile(r'[\W_]*[DS]\d{1,2}[\W_]*')


def answer_words(text):
    """The words of a note answer as its word cap means them: tokens with a letter or digit, less the bare IDs."""
    return sum(1 for token in text.split() if re.search(r'[^\W_]', token) and not _BARE_ID.fullmatch(token))


def word_caps(tier):
    """Each S10 answer file's word cap; a spill-over DISTILL (s10-<step>-spill.txt) has its step's."""
    caps = {'s10-decide.txt': baseline.s10_decide_words(tier), 's10-check.txt': baseline.S10_CHECK_WORDS,
            's10-state.txt': baseline.S10_STATE_WORDS}
    return dict(caps, **{name.replace('.txt', '-spill.txt'): cap for name, cap in caps.items()},
                **{'s10-distill.txt': baseline.s10_distill_words(tier),
                   's11-decide-b.txt': baseline.s10_decide_words(tier)})


def probe(task_dir, target, client, unit='', repeat=1, budget=baseline.STAGE_CAP):
    """One unit-repeat into target (created; it must not hold files yet). Returns the summary row."""
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    if any(target.iterdir()):
        raise SystemExit(f'refused: {target} is not empty')
    replay = ProbeReplay(target)
    prompt, instruction, profile, canaries, status = task_prompt(task_dir)
    replay.prompt(prompt)
    try:
        s10, notes = notes_stage(client, prompt, instruction, profile, canaries, status, replay, budget)
    except Exception as exc:  # the records so far still go out; the row says what stopped it
        s10, notes = None, ''
        status['probe_error'] = type(exc).__name__
    (target / 'notes.txt').write_text(notes)
    if s10 is not None and s10.get('plan'):
        (target / 'plan.txt').write_text(s10['plan'])
    tier = status.get('s10_tier', 'D')
    row = {'unit': unit, 'repeat': repeat, **status, 'requests': records(client),
           'answer_words': {name: answer_words(text) for name, text in sorted(replay.texts.items())},
           'answer_split_words': {name: len(text.split()) for name, text in sorted(replay.texts.items())},
           'word_caps': word_caps(tier if tier in baseline.S10_TIERS else 'D')}
    (target / 'status.json').write_text(json.dumps(row, indent=1, sort_keys=True) + '\n')
    return row


def client_from_env():
    """The House client solve would make: MODEL_ENDPOINT, MODEL_NAME and MODEL_TOKEN, and the solver's request cap."""
    return HouseClient(HouseConfig.from_env(), max_requests=baseline.MAX_REQUESTS)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('units', nargs='+', help='dev unit ids (t1-...)')
    parser.add_argument('--out', type=Path, required=True, help='output directory (unit-repeat folders, summary)')
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--budget', type=float, default=baseline.STAGE_CAP, help='unit seconds (default 420)')
    parser.add_argument('--source', type=Path, help='units directory of the source cache')
    parser.add_argument('--prompt-only', action='store_true', help='build the prompt and triage; no request')
    args = parser.parse_args(argv)
    dev, held = contract2_check.split()
    units = [check_unit(unit, set(dev), held) for unit in args.units]  # every id, before any unit is opened
    if not 1 <= args.repeats <= 10:
        raise SystemExit('--repeats must be 1-10')
    arm_b()
    if not args.prompt_only:
        try:  # an incomplete House configuration stops the probe before any unit is staged
            HouseConfig.from_env()
        except HouseError as exc:
            raise SystemExit(str(exc)) from None
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    with library_path():
        run(units, out, args)


def run(units, out, args):
    """Each unit-repeat on its own staged view, removed after it; one summary line each."""
    for unit in units:
        for repeat in range(1, (1 if args.prompt_only else args.repeats) + 1):
            with tempfile.TemporaryDirectory(prefix='s10-probe-') as root:
                task_dir = stage(unit, root, args.source)
                if args.prompt_only:
                    prompt, instruction, profile, _, _ = task_prompt(task_dir)
                    folder = out / unit
                    folder.mkdir(parents=True, exist_ok=True)
                    (folder / 'prompt.txt').write_text(prompt)
                    triage = baseline.s10_triage(instruction, profile)
                    (folder / 'triage.json').write_text(json.dumps(triage, indent=1, sort_keys=True) + '\n')
                    print(json.dumps({'unit': unit, **triage, 'prompt_chars': len(prompt)}))
                    continue
                row = probe(task_dir, out / unit / f'r{repeat}', client_from_env(), unit, repeat, args.budget)
            with (out / 'summary.jsonl').open('a') as stream:
                stream.write(json.dumps(row, sort_keys=True) + '\n')
            print(json.dumps({key: row.get(key) for key in (
                'unit', 'repeat', 's10_tier', 's10_decide', 's10_check', 's10_check_trigger', 's10_state',
                's10_d_lines', 's10_breaker', 's10_distill_spill', 's10_notes_sec', 's10_rate_after_decide',
                's11_decide_b', 's11_lines_added', 's11_conflicts', 'probe_error')}))


if __name__ == '__main__':
    main()
