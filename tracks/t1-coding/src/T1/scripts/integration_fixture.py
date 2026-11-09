#!/usr/bin/env python3
"""Exercise the real container solver/runner/checker with synthetic inputs only.

No generated program runs on the host. A loopback-only server inside the agent
container substitutes for House; this never measures official task performance.
"""
from __future__ import annotations

import argparse
import ast
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from t1_phase1.baseline import summarize_checker
from t1_phase1.docker_runner import build_agent_command, build_checker_command, resource_profile, run_container
from t1_phase1.solver_profile import (CANDIDATE, THIN_FEATURES_WIRED, THIN_FINAL_DELIVERY_TRIGGER,
                                      THIN_MAX_CONTINUATIONS, THIN_MAX_CONTINUATIONS_UNIT, THIN_RUN_PARAMS,
                                      parse_thin_features, parse_thin_final_delivery_trigger,
                                      parse_thin_max_continuations, parse_thin_max_continuations_unit, profile_env)
from t1_phase1.staging import stage_unit
from t1_phase1.evaluator import copy_for_grading, read_status


SCENARIOS = {
    'normal': {'expected_status': 'executed_ungraded', 'expected_calls': 1,
               'worker_started': True, 'budget_sec': 90},
    'missing_config': {'expected_status': 'not_run_missing_house', 'expected_calls': 0,
                       'worker_started': False, 'budget_sec': 45},
    'http_401': {'expected_status': 'infrastructure_house_error', 'expected_calls': 1,
                 'worker_started': False, 'budget_sec': 45},
    'http_403': {'expected_status': 'infrastructure_house_error', 'expected_calls': 1,
                 'worker_started': False, 'budget_sec': 45},
    # 503 backs off 2, 4, 8 and 15 s, and only while a full reply (44 s at the default rate) plus the
    # 20 s reserve still fits after the wait: in 120 s the fifth failure's 25 s wait no longer does.
    'http_503': {'expected_status': 'infrastructure_house_error', 'expected_calls': 5,
                 'worker_started': False, 'budget_sec': 120},
    # The same cut reply every time: R1, a statement continuation, a line continuation, then
    # no progress. 120 s keeps each continuation above the gate (full reply + 45 s).
    'truncated': {'expected_status': 'generation_failed', 'expected_calls': 3,
                  'worker_started': False, 'budget_sec': 120},
    'invalid_python': {'expected_status': 'generation_failed', 'expected_calls': 1,
                       'worker_started': False, 'budget_sec': 45},
    # The patch request gets the same program back, and so does the one retry: a no-op stops repair. The
    # fresh draw that follows (T1_FRESH) returns that program again, so it is recognised and not run.
    'execution_exception': {'expected_status': 'execution_failed', 'expected_calls': 4,
                            'worker_started': True, 'budget_sec': 120},
    # baseline reserves 20 seconds after generation; this leaves about 3 seconds
    # for the sleeping worker and still permits normal localhost HTTP overhead.
    'execution_timeout': {'expected_status': 'execution_timeout', 'expected_calls': 1,
                          'worker_started': True, 'budget_sec': 23},
}

# Stack10's conservative prior-rate gates can fit fewer requests into these
# short fault fixtures. This policy comes from exp/t1-stack10-on at f58a85e,
# T1/scripts/integration_fixture.py; exact classic counts remain unchanged.
# It is selected explicitly and checked against the inspected image, never a
# Dockerfile in a possibly different checkout.
FIXTURE_PROFILES = ('classic', 'stack10', 'minimal', 'thin', 'thin-features')
# The thin loop with the image defaults (T1_THIN_FEATURES = solver_profile.THIN_FEATURES_DEFAULT through profile_env,
# today every R2 mechanism off), and with one switched arm's own T1_THIN_FEATURES set, any set but the image default
# (S2-02): one features gate, and one receipt, per switched arm of the run card.
THIN_PROFILES = ('thin', 'thin-features')
S10_FEWER_CALLS = frozenset(('http_503', 'truncated', 'execution_exception'))
# --fixture-env: the only variables a gate run may set on top of the image (S2-02): the solver's registered R2 run
# parameters (solver_profile.THIN_RUN_PARAMS, every one a T1_THIN_* name), each checked by the solver's own parser
# before Docker starts. Never a model, House or credential variable.
FIXTURE_ENV = THIN_RUN_PARAMS
assert all(name.startswith('T1_THIN_') for name in FIXTURE_ENV)
# Exact calls per scenario on the thin loop: 'thin' with no count-changing feature (the image defaults while they
# hold none of COUNT_CHANGING), 'thin-features' with every
# wired T1_THIN_FEATURES name on (THIN_FEATURES_WIRED: k3 is registered but refused). Each table is derived on the
# real loop with trusted fake HTTP and StandIn workers, sleeps on the clock (test_thin_fixture), never copied from the
# other: a mechanism that changes a count changes the row of its own table, with the reason beside it. The truncated
# row is the number of cut code replies (thin_truncated_calls).
# http_503: the contract is retried after 2, 4, 8 and 15 s; a 25 s wait would leave less than the 60 s reserve plus a
# 30 s request, so the fifth failure is final. 4xx is never retried. invalid_python / execution_exception: four
# unchanged replies after the last change.
THIN_COUNTS = {'normal': 2, 'missing_config': 0, 'http_401': 1, 'http_403': 1, 'http_503': 5, 'truncated': 5,
               'invalid_python': 5, 'execution_exception': 6, 'execution_timeout': 2}
# 'thin-features' tables, one per set of count-changing mechanisms an arm turns on (COUNT_CHANGING & its features):
# the row an arm's gate checks is FEATURE_COUNTS[frozenset(its features & COUNT_CHANGING)]. A mechanism that changes a
# count joins COUNT_CHANGING and adds the tables for the arms that use it (e.g. frozenset({'k3'}) once S2-12 lands);
# an arm whose key has no table is refused, never checked against another arm's counts. Every wired mechanism has
# landed and none changes a count (test_thin_switches runs each name alone, all of them and R2's X'/Y' arms on the
# real loop; R2's arm gates passed at 40c1665), so COUNT_CHANGING is empty and every arm's table equals the
# features-off one (derived at 4003472). The 'thin' gate uses the table of the image defaults' count-changing names:
# THIN_COUNTS while they hold none.
COUNT_CHANGING = frozenset()
FEATURE_COUNTS = {frozenset(): {'normal': 2, 'missing_config': 0, 'http_401': 1, 'http_403': 1, 'http_503': 5,
                                'truncated': 5, 'invalid_python': 5, 'execution_exception': 6, 'execution_timeout': 2}}


def feature_table(features):
    """The exact-count table of a 'thin-features' arm with T1_THIN_FEATURES `features` (a set of names)."""
    key = frozenset(features) & COUNT_CHANGING
    if key not in FEATURE_COUNTS:
        raise ValueError('no exact-count table for the count-changing features ' + ','.join(sorted(key)))
    return FEATURE_COUNTS[key]


def default_features():
    """The image's default T1_THIN_FEATURES (profile_env, i.e. solver_profile.THIN_FEATURES_DEFAULT), sorted: what the
    'thin' profile runs with and what its receipt records."""
    return sorted(parse_thin_features(profile_env(CANDIDATE)['T1_THIN_FEATURES']))


def thin_table(profile, features):
    """The exact-count table of a thin profile run with T1_THIN_FEATURES `features`: THIN_COUNTS for 'thin' while the
    image defaults turn on no count-changing name, else that set's FEATURE_COUNTS table (feature_table)."""
    if profile == 'thin' and not frozenset(features) & COUNT_CHANGING:
        return THIN_COUNTS
    return feature_table(features)


def thin_truncated_calls(cut_replies, continuations, unit=None):
    """truncated: the contract, then `cut_replies` code replies that are each cut again (one change, then four
    unchanged ones: no progress), each continued up to `continuations` times (the per-reply cap N) while the unit cap
    U (None: off) lasts: 1 + R + min(R x N, U); with the defaults (R 5, N 3, off) 1 + 5 x (1 + N) = 21."""
    sent = cut_replies * continuations
    return 1 + cut_replies + (sent if unit is None else min(sent, unit))


def fixture_switches(profile, fixture_env):
    """The R2 switches a gate run uses: the image defaults (profile_env), overridden by --fixture-env.

    Refused (ValueError, never repeating a value): a name outside FIXTURE_ENV, a value the solver would refuse,
    --fixture-env on a profile that is not thin, a feature set that does not fit the profile ('thin': the image
    default, default_features; 'thin-features': any other set, the arm's own) and an arm with no exact-count table
    (feature_table)."""
    if not isinstance(fixture_env, dict) or any(name not in FIXTURE_ENV for name in fixture_env):
        raise ValueError('--fixture-env accepts only ' + ', '.join(FIXTURE_ENV))
    if fixture_env and profile not in THIN_PROFILES:
        raise ValueError('--fixture-env applies only to the thin profiles')
    for name, value in fixture_env.items():
        if not isinstance(value, str):
            raise ValueError(name + ' must be a string')
        FIXTURE_ENV[name](value)
    defaults = profile_env(CANDIDATE)
    switches = {name: fixture_env.get(name, defaults.get(name)) for name in FIXTURE_ENV}
    features = parse_thin_features(switches['T1_THIN_FEATURES'])
    default = frozenset(parse_thin_features(defaults['T1_THIN_FEATURES']))
    if profile == 'thin' and features != default:
        raise ValueError("profile 'thin' runs with the image's default T1_THIN_FEATURES; a switched arm uses "
                         "'thin-features'")
    if profile == 'thin-features':
        if features == default:
            raise ValueError("profile 'thin-features' needs the arm's own T1_THIN_FEATURES from --fixture-env (a set "
                             "other than the image default)")
        feature_table(features)
    # The image declares the default per-reply cap; --fixture-env may lower it for an arm (0-3).
    return {'thin_features': sorted(features),
            'thin_max_continuations': parse_thin_max_continuations(switches['T1_THIN_MAX_CONTINUATIONS']),
            'thin_max_continuations_unit': switches['T1_THIN_MAX_CONTINUATIONS_UNIT'],
            'thin_final_delivery_trigger': switches['T1_THIN_FINAL_DELIVERY_TRIGGER'],
            # Every registered run parameter the agent container sees: the image default, else the gate's value.
            'thin_run_params': {name: value for name, value in sorted(switches.items()) if value is not None}}


def scenario_plan(profile='classic', *, continuations=THIN_MAX_CONTINUATIONS,
                  unit_continuations=THIN_MAX_CONTINUATIONS_UNIT, features=None, run_params=None,
                  final_delivery_trigger=THIN_FINAL_DELIVERY_TRIGGER):
    """Exact expected calls per scenario; thin counts follow the continuation caps (per reply and per unit) and the
    feature set: on 'thin', the image default (default_features); on 'thin-features', the arm's (default: every wired
    name), any set but the image default."""
    if profile not in FIXTURE_PROFILES:
        raise ValueError('Unknown fixture profile')
    if type(continuations) is not int or not 0 <= continuations <= 3:
        raise ValueError('The continuation cap must be an integer from 0 to 3')
    unit = parse_thin_max_continuations_unit(unit_continuations)
    parse_thin_final_delivery_trigger(final_delivery_trigger)
    if profile not in THIN_PROFILES and final_delivery_trigger != THIN_FINAL_DELIVERY_TRIGGER:
        raise ValueError('A final delivery trigger applies only to the thin profiles')
    if profile not in THIN_PROFILES and unit_continuations != THIN_MAX_CONTINUATIONS_UNIT:
        raise ValueError('A unit continuation cap applies only to the thin profiles')
    if profile == 'thin-features':
        features = sorted(parse_thin_features(','.join(THIN_FEATURES_WIRED) if features is None
                                              else ','.join(features)))
        if features == default_features():
            raise ValueError("profile 'thin-features' needs a feature set other than the image default")
    elif features:
        raise ValueError("A feature set applies only to profile 'thin-features'")
    else:
        features = default_features() if profile == 'thin' else []
    plan = {name: dict(values) for name, values in SCENARIOS.items()}
    if profile == 'stack10':
        for name in S10_FEWER_CALLS:
            plan[name]['minimum_calls'] = 1
    if profile == 'minimal':
        # Frozen against the real controller with trusted fake House/workers.
        # 150 seconds admits the 2,200-token plan plus execution/patch reserve.
        # The 503 route retries after 2, 4 and 8 seconds; another 15 cannot fit.
        counts = {'http_503': 4, 'invalid_python': 3, 'execution_exception': 3}
        for name, row in plan.items():
            row['expected_calls'] = counts.get(name, row['expected_calls'])
            row['budget_sec'] = 45 if name == 'missing_config' else 150
            row['expected_token_caps'] = [2200] * row['expected_calls']
        for name in ('truncated', 'invalid_python'):
            plan[name]['expected_token_caps'] = [2200, 3200, 3200]
        plan['execution_exception']['expected_token_caps'] = [2200, 4000, 4000]
        # First response fits; remaining execution < 120-second sleeping payload.
        plan['execution_timeout']['budget_sec'] = 130
    if profile in THIN_PROFILES:
        # One short contract, then explicit version-bound source replies; the counts are THIN_COUNTS or
        # FEATURE_COUNTS (thin_table). truncated: each code reply is cut and continued (each continuation stays cut)
        # up to the caps, and the same cut program four more times is no progress (thin_truncated_calls; 21 at the
        # defaults).
        table = thin_table(profile, features)
        counts = dict(table, truncated=thin_truncated_calls(table['truncated'], continuations, unit))
        for name, row in plan.items():
            row['expected_calls'] = counts[name]
            row['budget_sec'] = 45 if name == 'missing_config' else 150
            row['expected_token_caps'] = [4000] * counts[name]
            # What the solver must report it ran with: its startup identity always, the loop's own record once
            # the loop has sent a request.
            row['expected_switches'] = {'thin_features': list(features),
                                        'thin_max_continuations_unit': unit_continuations,
                                        'thin_final_delivery_trigger': final_delivery_trigger}
            if run_params is not None:  # the identity's record of every registered run parameter it saw
                row['expected_run_params'] = dict(run_params)
        plan['execution_timeout']['budget_sec'] = 130
    return plan


# This program is a synthetic transport fixture, never an official task solution.
PROGRAM = '''import json, os, subprocess
from pathlib import Path

Path("/app/output/_fixture_worker_started").write_text("fixture_only\\n")

def environ_denied(path):
    try:
        with open(path, "rb") as stream:
            stream.read(1)
    except PermissionError:
        return True
    return False

def mounts():
    result = {}
    for line in Path("/proc/mounts").read_text().splitlines():
        fields = line.split()
        result[fields[1]] = set(fields[3].split(","))
    return result

def tmp_64_mib(options):
    size = next((value[5:] for value in options if value.startswith("size=")), "")
    factor = {"k": 1024, "m": 1024 ** 2, "g": 1024 ** 3}
    if size[-1:].lower() in factor:
        return int(size[:-1]) * factor[size[-1].lower()] == 64 * 1024 ** 2
    return size.isdigit() and int(size) == 64 * 1024 ** 2

def tmp_execution_denied():
    executable = Path("/tmp/t1-fixture-exec-probe")
    marker = Path("/tmp/t1-fixture-exec-probe.marker")
    marker.unlink(missing_ok=True)
    executable.write_text("#!/bin/sh\\nprintf executed > /tmp/t1-fixture-exec-probe.marker\\nexit 0\\n")
    executable.chmod(0o755)
    try:
        # Reading a script through its interpreter is allowed on a noexec mount.
        # This control proves the script and its marker write work before direct exec.
        control = subprocess.run(["/bin/sh", str(executable)], check=False,
                                 capture_output=True, timeout=2)
        if control.returncode != 0 or not marker.is_file():
            return False
        marker.unlink()
        try:
            result = subprocess.run([str(executable)], check=False, capture_output=True, timeout=2)
        except PermissionError:
            return not marker.exists()
        # QEMU/posix_spawn can report EACCES as child exit 127 instead of raising.
        # A script that ran, even if it then failed, is never evidence of rejection.
        return result.returncode in (126, 127) and not marker.exists()
    finally:
        executable.unlink(missing_ok=True)
        marker.unlink(missing_ok=True)

data = json.loads(Path("/input/data/inputs.json").read_text())
mount_options = mounts()
isolation = {
    "non_root": os.getuid() == 65534,
    "checks_hidden": not Path("/input/checks").exists(),
    "reference_hidden": not Path("/input/reference").exists(),
    "nested_reference_hidden": not Path("/input/data/reference").exists(),
    "model_token_absent": "MODEL_TOKEN" not in os.environ,
    "parent_environ_denied": environ_denied(f"/proc/{os.getppid()}/environ"),
    "pid1_environ_denied": environ_denied("/proc/1/environ"),
    "root_readonly": "ro" in mount_options.get("/", set()),
    "input_readonly": "ro" in mount_options.get("/input", set()),
    "tmp_noexec_flag": "noexec" in mount_options.get("/tmp", set()),
    "tmp_64_mib": tmp_64_mib(mount_options.get("/tmp", set())),
    "tmp_execution_denied": tmp_execution_denied(),
    "offline_loopback_only": {p.name for p in Path("/sys/class/net").iterdir()} == {"lo"},
}
Path("/app/output/answer.json").write_text(json.dumps({
    "answer": data["left"] + data["right"], "isolation": isolation,
    "fixture_only": True,
}) + "\\n")
assert all(isolation.values()), "Fixture isolation invariant failed"
'''


MARKER = ('from pathlib import Path\n'
          'Path("/app/output/_fixture_worker_started").write_text("fixture_only\\n")\n')
# Every request of a scenario gets the same reply. The truncated one carries at least 400
# characters of code (a shorter cut reply is rewritten, not continued) in at most 20 lines.
REPLIES = {
    'normal': PROGRAM,
    'truncated': MARKER + ''.join(f'value_{i:02d} = {i} * 1_000_003 + 7\n' for i in range(16)),
    'invalid_python': MARKER + 'if :\n',
    'execution_exception': MARKER + 'raise RuntimeError("synthetic_execution_failure")\n',
    'execution_timeout': MARKER + 'import time\ntime.sleep(120)\n',
}


RUNTIME = '''"""Synthetic loopback endpoint; no checker/reference material is here."""
import importlib, importlib.metadata, json, os, re, runpy, secrets, sys, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

programs = __REPLIES_LITERAL__
expected_caps = __CAPS_LITERAL__
fixture_profile = __PROFILE_LITERAL__
scenario = os.environ["T1_FIXTURE_SCENARIO"]
if scenario not in __SCENARIO_NAMES__:
    raise RuntimeError("Unknown integration fixture scenario")
state = {"fixture_only": True, "official_house_calls": 0,
         "requests": 0, "valid_requests": 0, "payloads_logged": False, "scenario": scenario,
         "request_token_caps": []}
for module in ("numpy", "pandas", "scipy", "pyarrow", "matplotlib", "statsmodels", "plotly", "pytest"):
    importlib.import_module(module)
state["installed_versions"] = {name: importlib.metadata.version(name) for name in (
    "numpy", "pandas", "scipy", "pyarrow", "matplotlib", "statsmodels", "plotly",
    "pytest", "pytest-json-report", "pytest-timeout")}

def token_cap_valid(name, index, value):
    caps = expected_caps.get(name)
    expected = 4000 if caps is None else caps[index - 1] if 1 <= index <= len(caps) else None
    return type(value) is int and value == expected

def conversation_valid(messages):
    roles = [m.get("role") for m in messages] if isinstance(messages, list) and all(
        isinstance(m, dict) and isinstance(m.get("content"), str) for m in messages) else []
    return (bool(roles) and roles[0] == roles[-1] == "user"
            and all(a != b and {a, b} == {"user", "assistant"} for a, b in zip(roles, roles[1:])))

def fixture_response(name, index, messages):
    if fixture_profile != "thin":
        return programs[name], "length" if name == "truncated" else "stop"
    if index == 1:
        return "Read inputs.json; add left/right; write answer.json. APIS: none.", "stop"
    if len(messages) == 3:
        # A continuation of a cut code reply: only the truncated scenario is ever cut. The reply adds
        # one distinct line and is cut again, so the program never completes.
        turn = messages[1].get("content", "")
        if (name != "truncated" or messages[1].get("role") != "assistant" or "```python" not in turn
                or len(re.findall(r"(?m)^CURRENT VERSION: ([0-9a-f]{64})$", messages[0].get("content", ""))) != 1):
            raise ValueError("unexpected continuation")
        return "```python\\n    " + str(len(turn)) + ",\\n", "length"
    last = messages[-1].get("content", "") if messages else ""
    matches = re.findall(r"(?m)^CURRENT VERSION: ([0-9a-f]{64})$", last)
    if len(matches) != 1:
        raise ValueError("one explicit program version required")
    source = programs[name]
    if name == "truncated":
        source += "unfinished = (\\n"
    return ("REPLACE " + matches[0] + "\\n```python\\n" + source
            + ("" if name == "truncated" else "\\n```")), "length" if name == "truncated" else "stop"

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        state["requests"] += 1
        length = int(self.headers.get("Content-Length", "0"))
        if not 0 < length <= 200000:
            self.send_error(413)
            return
        body = json.loads(self.rfile.read(length))
        token_cap = body.get("max_tokens")
        state["request_token_caps"].append(token_cap if type(token_cap) is int and 0 < token_cap <= 4000 else None)
        valid = (self.path == "/v1/chat/completions"
                 and self.headers.get("Authorization") == "Bearer " + os.environ["MODEL_TOKEN"]
                 and body.get("model") == "fixture-only-local"
                 and token_cap_valid(scenario, state["requests"], token_cap)
                 and conversation_valid(body.get("messages")))
        if not valid:
            self.send_error(400)
            return
        state["valid_requests"] += 1
        if scenario.startswith("http_"):
            response = b"synthetic service failure; no model program returned"
            self.send_response(int(scenario.split("_")[1]))
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)
            return
        try:
            content, finish = fixture_response(scenario, state["requests"], body["messages"])
        except ValueError:
            self.send_error(400)
            return
        response = json.dumps({"choices": [{"message": {
            "role": "assistant", "content": content}, "finish_reason": finish}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)

server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
for variable in ("MODEL_ENDPOINT", "MODEL_NAME", "MODEL_TOKEN"):
    os.environ.pop(variable, None)
os.environ.update(NO_PROXY="127.0.0.1,localhost", no_proxy="127.0.0.1,localhost")
if scenario != "missing_config":
    os.environ.update(MODEL_ENDPOINT=f"http://127.0.0.1:{server.server_port}",
                      MODEL_NAME="fixture-only-local", MODEL_TOKEN=secrets.token_urlsafe(32))
state["configured_model_variables"] = {name: bool(os.environ.get(name)) for name in
                                       ("MODEL_ENDPOINT", "MODEL_NAME", "MODEL_TOKEN")}
thread = threading.Thread(target=server.serve_forever, daemon=True)
thread.start()
exit_code = 1
try:
    # Execute the real solve launcher in this PID 1, without a shell or another
    # token-bearing parent. Only baseline.solve establishes nondumpability; the
    # fixture must not provide the protection it claims to test.
    sys.argv = ["solve", "--task-dir", "/input", "--out", "/app/output"]
    try:
        runpy.run_path("/usr/local/bin/solve", run_name="__main__")
        exit_code = 0
    except SystemExit as exc:
        exit_code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
finally:
    server.shutdown()
    server.server_close()
    thread.join(timeout=2)
    os.environ.pop("MODEL_TOKEN", None)
    Path("/app/output/_fixture_transport.json").write_text(json.dumps(state, indent=2) + "\\n")
sys.exit(exit_code)
'''


CHECKER = '''import json
from pathlib import Path
import pytest

def result():
    return json.loads(Path("/app/output/answer.json").read_text())

def test_synthetic_answer():
    assert result()["answer"] == 42

@pytest.mark.parametrize("name", [
    "non_root", "checks_hidden", "reference_hidden", "nested_reference_hidden",
    "model_token_absent", "parent_environ_denied", "pid1_environ_denied",
    "root_readonly", "input_readonly", "tmp_noexec_flag", "tmp_64_mib",
    "tmp_execution_denied", "offline_loopback_only",
])
def test_worker_isolation(name):
    assert result()["isolation"][name] is True

def test_exact_fixture_call_count():
    state = json.loads(Path("/app/output/_fixture_transport.json").read_text())
    assert state["requests"] == state["valid_requests"] == __NORMAL_CALLS__
    assert state["fixture_only"] is True
    assert state["official_house_calls"] == 0
'''


CHECKER_SH = '''#!/bin/bash
# Deliberately return zero regardless of pytest: the host must inspect artifacts.
set +e
python -m pytest /input/checks/test_output.py -q -p no:cacheprovider --json-report --json-report-file=/app/output/pytest_report.json
pytest_code=$?
PYTEST_EXIT_CODE="$pytest_code" python - <<'PY'
import json, os
from pathlib import Path
code = int(os.environ["PYTEST_EXIT_CODE"])
reward = 1.0 if code == 0 else 0.0
Path("/app/output/reward.json").write_text(json.dumps({"reward": reward, "pytest_exit_code": code}) + "\\n")
Path("/logs/verifier/reward.txt").write_text(str(int(reward)) + "\\n")
PY
exit 0
'''


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    path.chmod(0o644)


def prepare(run_dir: Path, plan=None, *, profile='classic') -> tuple[Path, Path, Path]:
    source, participant = run_dir / 'trusted_source', run_dir / 'participant'
    write(source / 'instruction.md', 'Synthetic integration fixture only. Read data/inputs.json, '
          'add left and right, and write answer.json. This is not an official task.\n')
    write(source / 'card.toml', 'id = "fixture-only-addition"\n[agent]\ntimeout_sec = 90\n')
    write(source / 'data/inputs.json', '{"left": 20, "right": 22}\n')
    write(source / 'reference/hidden.txt', 'Synthetic reference material; participant must not see this.\n')
    write(source / 'data/reference/hidden.csv', 'synthetic_reference\nnot_participant_input\n')
    selected_plan = SCENARIOS if plan is None else plan
    write(source / 'checks/test_output.py', CHECKER.replace(
        '__NORMAL_CALLS__', str(selected_plan['normal']['expected_calls'])))
    write(source / 'checks/test.sh', CHECKER_SH)
    stage_unit(source, participant)
    runtime = run_dir / 'fixture_runtime.py'
    selected_plan = SCENARIOS if plan is None else plan
    caps = {name: row['expected_token_caps'] for name, row in selected_plan.items()
            if 'expected_token_caps' in row}
    protocol = 'thin' if profile in THIN_PROFILES else profile  # the R2 switches change no fixture reply
    runtime_code = RUNTIME.replace('__REPLIES_LITERAL__', repr(REPLIES)).replace(
        '__SCENARIO_NAMES__', repr(tuple(SCENARIOS))).replace('__CAPS_LITERAL__', repr(caps)).replace('__PROFILE_LITERAL__', repr(protocol))
    # Syntax validation is not execution of generated code on the host.
    for name, code in REPLIES.items():
        if name != 'invalid_python':
            ast.parse(code)
    ast.parse(runtime_code)
    write(runtime, runtime_code)
    write(run_dir / 'scenario_plan.json', json.dumps(SCENARIOS if plan is None else plan, indent=2) + '\n')
    for path in (source, participant):
        for directory in [path, *(p for p in path.rglob('*') if p.is_dir())]:
            directory.chmod(0o755)
    forbidden = ('checks', 'reference', 'data/reference')
    if any((participant / name).exists() for name in forbidden):
        raise RuntimeError('Synthetic participant staging leaked grader material')
    return source, participant, runtime


def writable(path: Path) -> Path:
    path.mkdir(parents=True)
    path.chmod(0o777)
    return path


def inspect_image(context: str, image: str) -> dict:
    completed = subprocess.run(['docker', '--context', context, 'image', 'inspect', image],
        capture_output=True, text=True, timeout=30, check=False)
    if completed.returncode:
        raise RuntimeError('Image inspection failed; build the local image first')
    item = json.loads(completed.stdout)[0]
    metadata = {key: item.get(key) for key in ('Id', 'Architecture', 'Os', 'RepoDigests')}
    metadata['Entrypoint'] = item.get('Config', {}).get('Entrypoint')
    if metadata['Entrypoint']:
        raise RuntimeError('The shared agent/checker image must have no ENTRYPOINT')
    # Keep only controller switches; never save Config.Env model credentials.
    metadata['FixtureEnv'] = {}
    for name in ('T1_S10', 'T1_MINIMAL', 'T1_THIN'):
        flags = [value.split('=', 1)[1] for value in (item.get('Config', {}).get('Env') or [])
                 if isinstance(value, str) and value.startswith(name + '=')]
        if len(flags) > 1 or any(value not in ('0', '1') for value in flags):
            raise RuntimeError('Image has ambiguous fixture profile settings')
        metadata['FixtureEnv'][name] = flags[0] if flags else None
    caps = [value.split('=', 1)[1] for value in (item.get('Config', {}).get('Env') or [])
            if isinstance(value, str) and value.startswith('T1_THIN_MAX_CONTINUATIONS=')]
    try:
        if len(caps) > 1:
            raise ValueError('repeated')
        metadata['ThinMaxContinuations'] = parse_thin_max_continuations(caps[0]) if caps else None
    except ValueError:
        raise RuntimeError('Image has an ambiguous or invalid continuation cap') from None
    # The R2 switches the image declares (S2-01): each at most once and valid, else the image is refused.
    for name, key in (('T1_THIN_FEATURES', 'ThinFeatures'),
                      ('T1_THIN_MAX_CONTINUATIONS_UNIT', 'ThinMaxContinuationsUnit'),
                      ('T1_THIN_FINAL_DELIVERY_TRIGGER', 'ThinFinalDeliveryTrigger')):
        values = [value.split('=', 1)[1] for value in (item.get('Config', {}).get('Env') or [])
                  if isinstance(value, str) and value.startswith(name + '=')]
        try:
            if len(values) > 1:
                raise ValueError('repeated')
            parsed = FIXTURE_ENV[name](values[0]) if values else None
        except ValueError:
            raise RuntimeError('Image has an ambiguous or invalid ' + name) from None
        metadata[key] = (None if not values else sorted(parsed) if name == 'T1_THIN_FEATURES' else values[0])
    return metadata


def scenario_checks(name: str, record: dict, *, plan=None) -> dict:
    """Judge observed container outcomes, never fill missing evidence with defaults."""
    expected = (SCENARIOS if plan is None else plan)[name]
    status = record.get('agent_status') or {}
    process = record.get('process') or {}
    transport = record.get('transport') or {}
    config = transport.get('configured_model_variables', {})
    expected_configured = name != 'missing_config'

    def calls_ok(calls):
        if 'minimum_calls' in expected:
            return type(calls) is int and expected['minimum_calls'] <= calls <= expected['expected_calls']
        return type(calls) is int and calls == expected['expected_calls']

    switches = {}
    if 'expected_switches' in expected:
        identity = status.get('solver_identity') if isinstance(status.get('solver_identity'), dict) else {}
        recorded = [identity] + ([status] if expected['expected_calls'] > 0 else [])
        switches['thin_switches_recorded'] = (
            all(record.get(key) == value for record in recorded
                for key, value in expected['expected_switches'].items())
            and ('expected_run_params' not in expected
                 or identity.get('thin_run_params', {}) == expected['expected_run_params']))
    return {**switches,
        'controller_status': status.get('status') == expected['expected_status'],
        'model_call_count': (calls_ok(status.get('model_calls'))
                             and status.get('model_calls') == transport.get('requests')),
        'http_request_count': calls_ok(transport.get('requests')),
        'request_contract_valid': (calls_ok(transport.get('valid_requests'))
                                  and transport.get('valid_requests') == transport.get('requests')
                                  and ('expected_token_caps' not in expected or
                                       transport.get('request_token_caps') == expected['expected_token_caps'])),
        'solver_exit_code': process.get('returncode') == 0,  # every written status exits 0 (B3 P0-4)
        'not_outer_container_timeout': process.get('timed_out') is False,
        'worker_execution_marker': record.get('worker_started') is expected['worker_started'],
        'answer_presence': record.get('answer_exists') is (name == 'normal'),
        'real_environment_config': (set(config) == {'MODEL_ENDPOINT', 'MODEL_NAME', 'MODEL_TOKEN'}
                                    and all(value is expected_configured for value in config.values())),
        'fixture_only': transport.get('fixture_only') is True,
        'no_official_house_call': transport.get('official_house_calls') == 0,
        'correct_scenario': transport.get('scenario') == name,
    }


def run_scenario(args, name: str, participant: Path, runtime: Path, output: Path) -> dict:
    plan = getattr(args, 'scenario_plan', None)  # execute's plan; the profile's default plan only without one
    if plan is None:
        plan = scenario_plan(getattr(args, 'fixture_profile', 'classic'))
    budget = plan[name]['budget_sec']
    command = build_agent_command(args.image, participant, output,
        context=args.context, platform=args.platform, timeout_sec=budget, model_env=False,
        cpus=args.cpus, memory=args.memory)
    image_index = command.index(args.image)
    command[image_index:image_index] = ['--env', f'T1_FIXTURE_SCENARIO={name}', '--mount',
        f'type=bind,src={runtime.resolve()},dst=/opt/fixture_runtime.py,readonly']
    for key, value in sorted((getattr(args, 'fixture_env', None) or {}).items()):  # validated: fixture_switches
        command[image_index:image_index] = ['--env', f'{key}={value}']
    image_index = command.index(args.image)
    command[image_index + 1:] = ['python', '/opt/fixture_runtime.py']
    process = run_container(command, timeout_sec=budget + 30)
    record = {'process': process, 'agent_status': None, 'transport': None,
              'worker_started': (output / '_fixture_worker_started').is_file(),
              'answer_exists': (output / 'answer.json').is_file()}
    try:
        status_path = output / '_agent_status.json'
        if status_path.exists():
            record['agent_status'] = read_status(status_path)
        transport_path = output / '_fixture_transport.json'
        if transport_path.exists():
            record['transport'] = json.loads(transport_path.read_text())
    except (OSError, ValueError) as exc:
        record['evidence_error'] = type(exc).__name__
    record['checks'] = scenario_checks(name, record, plan=plan)
    record['passed'] = all(record['checks'].values())
    return record


def execute(args, run_dir: Path, report: dict) -> None:
    selected_profile = getattr(args, 'fixture_profile', 'classic')
    fixture_env = dict(getattr(args, 'fixture_env', None) or {})
    report['fixture_env'] = dict(sorted(fixture_env.items()))
    switches = fixture_switches(selected_profile, fixture_env)
    thin_profile = selected_profile in THIN_PROFILES
    args.scenario_plan = scenario_plan(
        selected_profile, continuations=switches['thin_max_continuations'],
        unit_continuations=switches['thin_max_continuations_unit'],
        features=switches['thin_features'] if selected_profile == 'thin-features' else None,
        run_params=switches['thin_run_params'] if thin_profile else None,
        final_delivery_trigger=switches['thin_final_delivery_trigger'])
    report['fixture_profile'] = selected_profile
    if thin_profile:
        report['fixture_max_continuations'] = switches['thin_max_continuations']
        report['fixture_max_continuations_unit'] = switches['thin_max_continuations_unit']
        report['fixture_final_delivery_trigger'] = switches['thin_final_delivery_trigger']
        report['fixture_thin_features'] = switches['thin_features']
    source, participant, runtime = prepare(run_dir, args.scenario_plan, profile=selected_profile)
    report['participant_files'] = sorted(p.relative_to(participant).as_posix()
                                         for p in participant.rglob('*') if p.is_file())
    if args.prepare_only:
        report['status'] = 'prepared_not_run'
        return
    if args.build:
        with (run_dir / 'build.log').open('w') as log:
            result = subprocess.run(['docker', '--context', args.context, 'build',
                '--platform', args.platform, '-t', args.image, '-f', str(ROOT / 'Dockerfile'),
                str(ROOT)], stdout=log, stderr=subprocess.STDOUT, timeout=1800, check=False)
        report['build_returncode'] = result.returncode
        if result.returncode:
            raise RuntimeError('Image build failed; see build.log')
    report['image'] = inspect_image(args.context, args.image)
    expected_arch = args.platform.removeprefix('linux/')
    if report['image']['Architecture'] != expected_arch:
        raise RuntimeError('Built image architecture differs from the selected local platform')
    setting = report['image'].get('FixtureEnv', {})
    if selected_profile in THIN_PROFILES:
        # The exact thin counts assume the default continuation cap and the default R2 switches (the image's default
        # features, no unit cap), which --fixture-env alone overrides, so the image must declare them.
        defaults = profile_env(CANDIDATE)
        matched = (setting.get('T1_THIN') == '1' and setting.get('T1_MINIMAL') == '0'
                   and setting.get('T1_S10') == '0'
                   and report['image'].get('ThinMaxContinuations') == THIN_MAX_CONTINUATIONS
                   and report['image'].get('ThinFeatures')
                   == sorted(parse_thin_features(defaults['T1_THIN_FEATURES']))
                   and report['image'].get('ThinMaxContinuationsUnit') == defaults['T1_THIN_MAX_CONTINUATIONS_UNIT']
                   and report['image'].get('ThinFinalDeliveryTrigger') == defaults['T1_THIN_FINAL_DELIVERY_TRIGGER'])
    else:
        matched = (setting.get('T1_THIN') != '1' and 'T1_S10' in setting and
                   (setting.get('T1_MINIMAL') == '1' and setting['T1_S10'] == '0' if selected_profile == 'minimal'
                    else setting.get('T1_MINIMAL') != '1' and
                    ((setting['T1_S10'] == '1') == (selected_profile == 'stack10'))))
    report['acceptance_checks'] = {'fixture_profile_matches_image': matched}
    if not matched:
        raise RuntimeError('Declared fixture profile does not match inspected T1_THIN/T1_MINIMAL/T1_S10 '
                           '(thin: T1_THIN_MAX_CONTINUATIONS, T1_THIN_MAX_CONTINUATIONS_UNIT, T1_THIN_FEATURES, '
                           'T1_THIN_FINAL_DELIVERY_TRIGGER) '
                           'settings')
    positive = writable(run_dir / 'positive/output')
    positive_reports = writable(run_dir / 'positive/verifier')
    report['scenarios'] = {}
    for name in SCENARIOS:
        output = positive if name == 'normal' else writable(run_dir / 'scenarios' / name / 'output')
        report['scenarios'][name] = run_scenario(args, name, participant, runtime, output)
    report['agent_process'] = report['scenarios']['normal']['process']
    report['agent_status'] = report['scenarios']['normal']['agent_status']
    checks = {'fixture_profile_matches_image': matched,
              **{f'scenario_{name}_expected_behavior': record['passed']
                 for name, record in report['scenarios'].items()}}
    report['acceptance_checks'] = checks
    if not report['scenarios']['normal']['passed']:
        report['status'] = 'failed'
        report['reason'] = 'Normal solver fixture failed; all fault scenarios were still collected'
        return
    positive_grade = run_dir / 'positive/grading-output'
    copy_for_grading(positive, positive_grade)
    positive_process = run_container(build_checker_command(args.image, source, positive_grade,
        positive_reports, context=args.context, platform=args.platform,
        cpus=args.cpus, memory=args.memory), timeout_sec=90)
    positive_verdict = summarize_checker(positive_grade, positive_process)
    report['positive'] = {'checker_process': positive_process, 'verdict': positive_verdict,
                          'reward': json.loads((positive_grade / 'reward.json').read_text())}
    negative = writable(run_dir / 'negative/output')
    negative_reports = writable(run_dir / 'negative/verifier')
    # Corrupt only the synthetic answer; do not call the model or solver again.
    for name in ('answer.json', '_agent_status.json', '_fixture_transport.json'):
        shutil.copyfile(positive / name, negative / name)
        (negative / name).chmod(0o666)
    answer = json.loads((negative / 'answer.json').read_text())
    answer['answer'] = 41
    (negative / 'answer.json').write_text(json.dumps(answer) + '\n')
    negative_process = run_container(build_checker_command(args.image, source, negative,
        negative_reports, context=args.context, platform=args.platform,
        cpus=args.cpus, memory=args.memory), timeout_sec=90)
    negative_verdict = summarize_checker(negative, negative_process)
    report['negative'] = {'checker_process': negative_process, 'verdict': negative_verdict,
                          'reward': json.loads((negative / 'reward.json').read_text())}
    status = report['agent_status']
    one_generation = status.get('model_calls') == 1
    if selected_profile in THIN_PROFILES:
        # One contract response plus one code response; planning is not code generation.
        kinds = status.get('request_kinds')
        one_generation = (type(status.get('model_calls')) is int and status['model_calls'] == 2
            and type(status.get('generation_replies')) is int and status['generation_replies'] == 1
            and kinds == {'thin_contract': 1, 'thin_code': 1}
            and all(type(value) is int for value in kinds.values()))
    checks.update({
        'one_real_solver_generation': one_generation,
        'positive_wrapper_zero': positive_process['returncode'] == 0,
        'negative_wrapper_zero': negative_process['returncode'] == 0,
        'positive_reward_one': report['positive']['reward'].get('reward') == 1.0,
        'negative_reward_zero': report['negative']['reward'].get('reward') == 0.0,
        'positive_verdict_passed': positive_verdict.get('status') == 'passed',
        'negative_verdict_failed': negative_verdict.get('status') == 'failed',
    })
    positive_answer = json.loads((positive / 'answer.json').read_text())
    checks['worker_no_model_token'] = positive_answer['isolation'].get('model_token_absent') is True
    checks['controller_proc_environ_protected'] = all(positive_answer['isolation'].get(name) is True
        for name in ('parent_environ_denied', 'pid1_environ_denied'))
    checks['checks_and_reference_hidden'] = all(positive_answer['isolation'].get(name) is True
        for name in ('checks_hidden', 'reference_hidden', 'nested_reference_hidden'))
    checks['readonly_mounts_verified'] = all(positive_answer['isolation'].get(name) is True
        for name in ('root_readonly', 'input_readonly'))
    checks['tmp_noexec_64m_verified'] = all(positive_answer['isolation'].get(name) is True
        for name in ('tmp_noexec_flag', 'tmp_64_mib', 'tmp_execution_denied'))
    checks['offline_loopback_only_verified'] = positive_answer['isolation'].get('offline_loopback_only') is True
    report['acceptance_checks'] = checks
    report['status'] = 'passed' if all(checks.values()) else 'failed'


def parse_fixture_env(pairs):
    """{NAME: VALUE} from repeated NAME=VALUE options: FIXTURE_ENV names only, each once (values are checked by
    fixture_switches)."""
    result = {}
    for pair in pairs:
        name, sep, value = pair.partition('=')
        if not sep or name not in FIXTURE_ENV:
            raise ValueError('--fixture-env takes NAME=VALUE with NAME one of ' + ', '.join(FIXTURE_ENV))
        if name in result:
            raise ValueError('--fixture-env sets ' + name + ' more than once')
        result[name] = value
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--context', default='colima')
    parser.add_argument('--platform', choices=['linux/arm64', 'linux/amd64'], default='linux/arm64')
    parser.add_argument('--image', default='t1-phase1:b0')
    parser.add_argument('--profile', choices=FIXTURE_PROFILES, default='classic', dest='fixture_profile',
                        help='explicit controller fixture profile; must match inspected T1_THIN/T1_MINIMAL/T1_S10')
    parser.add_argument('--fixture-env', action='append', default=[], metavar='NAME=VALUE',
                        help='thin profiles only: set one of ' + ', '.join(FIXTURE_ENV) + ' for the agent '
                             "container ('thin-features' requires the arm's own T1_THIN_FEATURES, a set other than "
                             "the image default); repeatable")
    parser.add_argument('--cpus', type=float, default=4)
    parser.add_argument('--memory', default='8g')
    parser.add_argument('--build', action='store_true', help='Explicitly build before running; never changes context')
    parser.add_argument('--prepare-only', action='store_true', help='Write fixtures without building or running Docker')
    parser.add_argument('--out-root', type=Path, default=ROOT / 'evidence/integration')
    parser.add_argument('--out-dir', type=Path, help='Use this fresh run directory instead of a generated name')
    parser.add_argument('--gate-receipt', type=Path, default=ROOT / 'evidence/integration/result.json',
                        help='Update this gate receipt after an attempted run; prepare-only leaves it unchanged')
    args = parser.parse_args()
    try:
        profile = resource_profile(args.cpus, args.memory)
    except ValueError as exc:
        parser.error(str(exc))
    if args.build and args.prepare_only:
        parser.error('--build and --prepare-only are mutually exclusive')
    try:
        args.fixture_env = parse_fixture_env(args.fixture_env)
        fixture_switches(args.fixture_profile, args.fixture_env)
    except ValueError as exc:
        parser.error(str(exc))
    run_dir = (args.out_dir.resolve() if args.out_dir else args.out_root.resolve() /
        (dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8]))
    if run_dir.exists():
        parser.error('--out-dir must not exist; previous evidence is never overwritten')
    run_dir.mkdir(parents=True)
    report = {'fixture_only': True, 'official_score': None, 'official_house_calls': 0,
              'official_tasks_used': False, 'context': args.context, 'platform': args.platform,
              'profile': profile, 'fixture_profile': args.fixture_profile, 'platform_equivalent': False,
              'fixture_env': dict(sorted(args.fixture_env.items())),
              'status': 'not_started', 'run_directory': str(run_dir)}
    try:
        execute(args, run_dir, report)
    except Exception as exc:
        report.update(status='integration_error', reason=type(exc).__name__ + ': ' + str(exc))
    finally:
        (run_dir / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        acceptance = {'fixture_only': True, 'status': report['status'],
                      'image_id': report.get('image', {}).get('Id'), 'platform': args.platform,
                      'fixture_profile': args.fixture_profile,
                      'fixture_max_continuations': report.get('fixture_max_continuations'),
                      # The gate's own variables for the agent container (S2-02), and the R2 switches they give.
                      'fixture_env': dict(sorted(args.fixture_env.items())),
                      'fixture_max_continuations_unit': report.get('fixture_max_continuations_unit'),
                      'fixture_final_delivery_trigger': report.get('fixture_final_delivery_trigger'),
                      'fixture_thin_features': report.get('fixture_thin_features'),
                      'checks': report.get('acceptance_checks', {'integration_completed': False}),
                      'report': str(run_dir / 'report.json'), 'official_score': None}
        (run_dir / 'result.json').write_text(json.dumps(acceptance, indent=2) + '\n')
        # Only an actual attempted integration updates the selected gate receipt.
        # A failed run invalidates the previous receipt; prepare-only never certifies it.
        if not args.prepare_only:
            args.gate_receipt.parent.mkdir(parents=True, exist_ok=True)
            args.gate_receipt.write_text(json.dumps(acceptance, indent=2) + '\n')
    print(json.dumps({'status': report['status'], 'fixture_only': True,
                      'official_score': None, 'report': str(run_dir / 'report.json')}))
    return 0 if report['status'] in ('passed', 'prepared_not_run') else 1


if __name__ == '__main__':
    raise SystemExit(main())
