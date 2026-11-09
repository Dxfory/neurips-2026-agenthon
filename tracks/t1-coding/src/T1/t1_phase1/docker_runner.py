"""Bounded local Docker execution. This profile is not platform-equivalent."""

import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import threading
import time
import uuid

from .staging import PROFILE, _destination, _relative, _safe_tree, data_mounts, participant_file


LOG_LIMIT = 65536
_MODEL_VARS = ('MODEL_ENDPOINT', 'MODEL_TOKEN', 'MODEL_NAME')


def _separate_directories(*directories: Path) -> None:
    resolved = [Path(path).resolve(strict=True) for path in directories]
    if any(not path.is_dir() for path in resolved):
        raise ValueError('Input, output and report mounts must be directories')
    for index, left in enumerate(resolved):
        for right in resolved[index + 1:]:
            if left.is_relative_to(right) or right.is_relative_to(left):
                raise ValueError('Input, output and report mounts must be disjoint')


def _mount(source: Path, destination: str, readonly: bool = False) -> list[str]:
    source = Path(source).resolve(strict=True)
    if ',' in str(source) or any(ord(c) < 32 for c in str(source)):
        raise ValueError('Unsupported Docker bind source path')
    return ['--mount', f'type=bind,src={source},dst={destination}' +
            (',readonly' if readonly else '')]


def resource_profile(cpus: float = 4, memory: str = '8g') -> str:
    """Validate explicit local limits; do not infer capacity from the host."""
    try:
        valid_cpu = type(cpus) in (int, float) and math.isfinite(float(cpus)) and cpus > 0
    except OverflowError:
        valid_cpu = False
    if not valid_cpu:
        raise ValueError('CPUs must be a finite positive number')
    if not isinstance(memory, str) or not re.fullmatch(r'[1-9][0-9]*[kmg]', memory):
        raise ValueError('Memory must be a positive integer followed by k, m or g')
    return f'local-{cpus:g}cpu-{memory}'


def _forwarded_model_env() -> list[str]:
    """The model variables, proxies and (local evaluation only) the evaluation backend and throttle switch that
    are set."""
    names = (*_MODEL_VARS, 'HTTP_PROXY', 'HTTPS_PROXY', 'NO_PROXY', 'http_proxy', 'https_proxy', 'no_proxy',
             'T1_EVAL_BACKEND', 'T1_EVAL_THROTTLE')
    forwarded = [name for name in names if os.environ.get(name)]
    for name in ('T1_REQUEST_SUPERVISOR', 'T1_LIBRARY86', 'T1_THIN_DIRECT_STEP'):
        if name in os.environ:
            if os.environ[name] not in ('0', '1'):
                raise ValueError(name + ' must be 0 or 1')
            forwarded.append(name)
    if 'T1_REQUEST_CAP' in os.environ:
        from .request_budget import parse_request_cap
        parse_request_cap(os.environ['T1_REQUEST_CAP'])
        forwarded.append('T1_REQUEST_CAP')
    if 'T1_THIN_THINKING' in os.environ:
        if os.environ['T1_THIN_THINKING'] not in ('on', 'off', 'low_effort'):
            raise ValueError('T1_THIN_THINKING must be on, off or low_effort')
        forwarded.append('T1_THIN_THINKING')
    if 'T1_THIN_REASONING_BUDGET' in os.environ:
        from .solver_profile import parse_thin_reasoning_budget
        parse_thin_reasoning_budget(os.environ['T1_THIN_REASONING_BUDGET'])
        forwarded.append('T1_THIN_REASONING_BUDGET')
    # The image sets both per-request switches, and each overrides the older T1_THIN_THINKING and
    # T1_THIN_REASONING_BUDGET for its request type; a forwarded per-request value replaces the image's.
    from .solver_profile import REQUEST_THINKING, parse_thin_request_thinking
    for name in REQUEST_THINKING:
        if name in os.environ:
            parse_thin_request_thinking(os.environ[name])
            forwarded.append(name)
    if 'T1_THIN_INTERFACE' in os.environ:
        if os.environ['T1_THIN_INTERFACE'] not in ('free', 'choice'):
            raise ValueError('T1_THIN_INTERFACE must be free or choice')
        forwarded.append('T1_THIN_INTERFACE')
    # The R2 run parameters (S2-01: the continuation caps, T1_THIN_FEATURES and whatever a mechanism registers in
    # solver_profile.THIN_RUN_PARAMS) replace the image's defaults for this run only; a value the loop would refuse is
    # refused here, before any container is launched.
    from .solver_profile import THIN_RUN_PARAMS
    for name, parse in THIN_RUN_PARAMS.items():
        if name in os.environ:
            parse(os.environ[name])
            forwarded.append(name)
    return forwarded


def _base(context: str, platform: str, role: str, *,
          cpus: float = 4, memory: str = '8g') -> list[str]:
    profile = resource_profile(cpus, memory)
    return ['docker', '--context', context, 'run', '--rm', '--name',
            f't1-phase1-{role}-{uuid.uuid4().hex[:16]}', f'--platform={platform}',
            '--read-only', '--user=65534:65534', '--cap-drop=ALL',
            '--security-opt=no-new-privileges', '--pids-limit=256', f'--cpus={cpus:g}',
            f'--memory={memory}', '--tmpfs=/tmp:rw,noexec,nosuid,size=64m,mode=1777',
            '--env', 'PYTHONDONTWRITEBYTECODE=1', '--env', 'HOME=/tmp',
            '--label', f't1.profile={profile}']


def _staged_manifest(participant: Path) -> dict:
    path = participant / 'staging_manifest.json'
    if path.is_symlink() or not path.is_file():
        raise ValueError('Agent inputs require a participant staging manifest')
    manifest = json.loads(path.read_text())
    if manifest.get('profile') != PROFILE or not isinstance(manifest.get('files'), dict):
        raise ValueError('Unrecognized participant staging manifest')
    actual = set()
    for entry in participant.iterdir():
        actual.update(p.relative_to(participant).as_posix() for p in _safe_tree(entry, participant))
    if actual != set(manifest['files']) | {'staging_manifest.json'}:
        raise ValueError('Participant staging contains unmanifested files')
    for name, digest in manifest['files'].items():
        relative = _relative(name)
        if not participant_file(relative):
            raise ValueError('Manifest contains a non-participant file')
        path = participant / Path(relative)
        _safe_tree(path, participant)
        with path.open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != digest:
                raise ValueError('Participant input hash does not match staging manifest')
    return manifest


def _data_binds(root: Path, mappings: list[dict]) -> list[str]:
    result = []
    for mapping in mappings:
        relative = _relative(mapping['source_relative'])
        if relative.parts[:1] != ('data',) and relative.parts[:2] != ('environment', 'data'):
            raise ValueError('Data mapping is outside input data')
        source = root / Path(relative)
        _safe_tree(source, root)
        destination = _destination(mapping['destination'], source)
        result.extend(_mount(source, destination, True))
    return result


def build_agent_command(image: str, participant_dir: Path, output_dir: Path, *,
                        context: str = 'colima', platform: str = 'linux/amd64',
                        timeout_sec: int = 420, model_env: bool = False,
                        cpus: float = 4, memory: str = '8g', replay_dir: Path | None = None,
                        budget_env: bool = True) -> list[str]:
    """Build a local command; model_env explicitly opts into local bridge access. replay_dir (local public
    evaluation only) is a writable bind at /replay that solve records raw replies and programs into.
    budget_env=False leaves the solve budget to the image (UNIT_BUDGET_SEC), as on the platform."""
    if timeout_sec <= 0:
        raise ValueError('Timeout must be positive')
    _separate_directories(participant_dir, output_dir, *([replay_dir] if replay_dir is not None else []))
    participant = Path(participant_dir).resolve(strict=True)
    _staged_manifest(participant)
    command = _base(context, platform, 'agent', cpus=cpus, memory=memory)
    command.append('--network=bridge' if model_env else '--network=none')
    if budget_env:
        command.extend(['--env', f'T1_TIMEOUT_SEC={timeout_sec}'])
    if model_env:
        if any(not os.environ.get(name) for name in _MODEL_VARS):
            raise ValueError('House model environment is incomplete; no container launched')
        command.extend(['--label', 't1.network=local-network-profile'])
        for name in _forwarded_model_env():
            command.extend(['--env', name])
    command.extend(_mount(participant, '/input', True))
    command.extend(_mount(output_dir, '/app/output'))
    command.extend(_mount(output_dir, '/output'))
    if replay_dir is not None:
        command.extend(['--env', 'T1_REPLAY_DIR=/replay', *_mount(replay_dir, '/replay')])
    # Like the platform (SUBMISSION_CLI invariant 3), agents get no /app data binds; solve()
    # maps the task's /app paths into /input. Only the checker receives those binds.
    command.extend([image, 'solve', '--task-dir', '/input', '--out', '/app/output'])
    return command


def build_checker_command(image: str, source_dir: Path, output_dir: Path, report_dir: Path, *,
                          context: str = 'colima', platform: str = 'linux/amd64',
                          cpus: float = 4, memory: str = '8g') -> list[str]:
    """Build an offline checker command that mounts only source, output and reports."""
    source = Path(source_dir).resolve(strict=True)
    _separate_directories(source, output_dir, report_dir)
    _safe_tree(source / 'checks/test.sh', source)
    command = _base(context, platform, 'checker', cpus=cpus, memory=memory)
    command.extend(['--network=none', '--env', 'OUTPUT_DIR=/app/output'])
    command.extend(_mount(source, '/input', True))
    command.extend(_mount(output_dir, '/app/output'))
    command.extend(_mount(output_dir, '/output'))
    command.extend(_mount(report_dir, '/logs/verifier'))
    command.extend(_data_binds(source, data_mounts(source)))
    # scoring.py _input_presentations shows <unit>/checks/reference_data at /tests/reference_data.
    # A verifier that finds nothing there can pass any well-formed output, so refuse instead.
    reference = source / 'checks/reference_data'
    if reference.exists() or reference.is_symlink():
        _safe_tree(reference, source)
        command.extend(_mount(reference, '/tests/reference_data', True))
    elif any('/tests/reference_data' in path.read_text(errors='replace')
             for path in sorted((source / 'checks').rglob('*.py'))):
        raise ValueError('Checker requires reference data that the unit does not ship')
    command.extend([image, 'bash', '/input/checks/test.sh'])
    return command


def _redact(value: str) -> str:
    for name, secret in os.environ.items():
        if re.search(r'token|secret|password|api.?key|credential', name, re.I) and len(secret) >= 4:
            value = value.replace(secret, '[REDACTED]')
    value = re.sub(r'(?i)(bearer\s+)\S+', r'\1[REDACTED]', value)
    return re.sub(r'(https?://)[^/@\s]+:[^/@\s]+@', r'\1[REDACTED]@', value)


def run_container(command: list[str], timeout_sec: float) -> dict:
    """Run a named Docker container, draining bounded logs and cleaning on timeout.

    The caller interprets reward artifacts separately: a completed checker process
    does not establish reward 1.0. No command or environment values are returned.
    """
    if (not command or Path(command[0]).name != 'docker' or 'run' not in command
            or '--name' not in command or timeout_sec <= 0):
        raise ValueError('Expected a named docker run command and a positive timeout')
    index = command.index('run')
    name_index = command.index('--name')
    if name_index <= index or name_index + 1 >= len(command):
        raise ValueError('Docker run requires an explicit container name')
    name = command[name_index + 1]
    if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_.-]*', name):
        raise ValueError('Invalid Docker container name')
    start = time.monotonic()
    result = {'status': 'launch_error', 'returncode': None, 'timed_out': False,
              'stdout': '', 'stderr': '', 'stdout_truncated': False,
              'stderr_truncated': False, 'cleanup': {'attempted': False}}
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   start_new_session=True)
    except OSError as exc:
        result['stderr'] = _redact(str(exc))[:LOG_LIMIT]
        result['duration_sec'] = round(time.monotonic() - start, 3)
        return result
    buffers = {'stdout': bytearray(), 'stderr': bytearray()}

    def drain(stream, key):
        with stream:
            while chunk := stream.read(8192):
                remaining = LOG_LIMIT - len(buffers[key])
                buffers[key].extend(chunk[:remaining])
                if len(chunk) > remaining:
                    result[key + '_truncated'] = True

    threads = [threading.Thread(target=drain, args=(stream, key), daemon=True)
               for key, stream in [('stdout', process.stdout), ('stderr', process.stderr)]]
    for thread in threads:
        thread.start()
    try:
        process.wait(timeout=timeout_sec)
        result['status'] = 'completed' if process.returncode == 0 else 'failed'
    except subprocess.TimeoutExpired:
        result.update(status='timeout', timed_out=True)
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)
        result['cleanup'] = {'attempted': True, 'ok': False}
        try:
            cleanup = subprocess.run(command[:index] + ['rm', '-f', name],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                     timeout=10, check=False)
            result['cleanup'].update(ok=cleanup.returncode == 0, returncode=cleanup.returncode)
        except (OSError, subprocess.TimeoutExpired):
            pass
    finally:
        for thread in threads:
            thread.join(timeout=2)
    result['returncode'] = process.returncode
    for key in buffers:
        result[key] = _redact(buffers[key].decode(errors='replace'))[:LOG_LIMIT]
    result['duration_sec'] = round(time.monotonic() - start, 3)
    return result
