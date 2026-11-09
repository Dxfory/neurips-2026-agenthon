"""POSIX supervisor for one trusted, interruptible request operation.

The caller selects a constant controller module/function, never a model-selected
entry point. The worker has signature ``operation(payload, emit) -> dict``;
``emit`` accepts only the progress fields below. Payload and result are sensitive
application data, not diagnostics. No retry, HTTP policy, or usage inference is
performed here. Missing usage remains the adapter's unknown value.

Startup, input transfer and result validation share the caller's absolute
``time.monotonic()`` deadline. Cleanup has its own bounded allowance, which the
controller must reserve. This is not a real-time kernel guarantee: process
creation/syscalls and bounded parent-side validation may incur scheduling delay.
It cannot promise upstream request cancellation or billing cancellation.

Only trusted code runs in the child; this is not a sandbox for generated code.
Workers must not escape their session or arrange for another process to reap
them. The parent must retain exclusive wait ownership of its child. Parent
process death/SIGKILL is outside the cleanup guarantee.
"""
from __future__ import annotations

from dataclasses import dataclass
import importlib
import json
import math
import os
from pathlib import Path
import re
import selectors
import signal
import socket
import struct
import subprocess
import sys
import time


MAX_MESSAGE_BYTES = 8 * 1024 * 1024
MAX_PROGRESS_BYTES = 4096
MAX_PROGRESS_MESSAGES = 128
MAX_TOTAL_BYTES = MAX_MESSAGE_BYTES + MAX_PROGRESS_BYTES * MAX_PROGRESS_MESSAGES
_ROOT = Path(__file__).resolve().parent.parent
_NAME = re.compile(r'[A-Za-z][A-Za-z0-9_]*\Z')
_PHASES = {'open', 'body_read', 'json_parse', 'shape', 'done'}
_COUNTERS = {'body_bytes': 2_147_483_647, 'body_chunks': 1_000_000,
             'http_status': 599}
_TIMINGS = {'headers_elapsed_sec', 'first_byte_sec', 'first_non_whitespace_sec'}
_PROTOCOL_REASONS = {'message_oversize', 'message_limit', 'invalid_progress',
                     'invalid_result', 'invalid_message', 'truncated_message'}


@dataclass(frozen=True, repr=False)
class Outcome:
    status: str
    reason: str | None
    result: dict | None
    progress: dict
    progress_count: int
    request_elapsed_sec: float
    elapsed_sec: float
    cleanup: dict


class _ProtocolError(Exception):
    pass


def _progress(value):
    if type(value) is not dict or not value:
        raise _ProtocolError('invalid_progress')
    for key, item in value.items():
        if key == 'phase':
            valid = type(item) is str and item in _PHASES
        elif key in _COUNTERS:
            low = 100 if key == 'http_status' else 0
            valid = type(item) is int and low <= item <= _COUNTERS[key]
        elif key in _TIMINGS:
            valid = type(item) in (int, float) and math.isfinite(item) and 0 <= item <= 86400
        else:
            valid = False
        if not valid:
            raise _ProtocolError('invalid_progress')
    return dict(value)


def _encode(value, *, reason='invalid_message', limit=MAX_MESSAGE_BYTES):
    try:
        data = json.dumps(value, ensure_ascii=False, allow_nan=False,
                          separators=(',', ':')).encode('utf-8')
    except (ValueError, TypeError, RecursionError, UnicodeError):
        raise _ProtocolError(reason) from None
    if len(data) > limit:
        raise _ProtocolError('message_oversize')
    return struct.pack('!I', len(data)) + data


def _plain_input(value, deadline):
    """Bound parent work and avoid callbacks from application container types."""
    pending = [(value, 0)]
    nodes = chars = 0
    while pending:
        if time.monotonic() >= deadline:
            return False
        item, depth = pending.pop()
        nodes += 1
        if nodes > 200_000 or depth > 32:
            raise _ProtocolError('message_oversize')
        kind = type(item)
        if kind is dict:
            if len(item) > 200_000 - nodes:
                raise _ProtocolError('message_oversize')
            for key, child in item.items():
                if type(key) is not str:
                    raise _ProtocolError('invalid_message')
                chars += len(key)
                pending.append((child, depth + 1))
        elif kind is list:
            if len(item) > 200_000 - nodes:
                raise _ProtocolError('message_oversize')
            pending.extend((child, depth + 1) for child in item)
        elif kind is str:
            chars += len(item)
        elif kind is float:
            if not math.isfinite(item):
                raise _ProtocolError('invalid_message')
        elif kind not in (int, bool, type(None)):
            raise _ProtocolError('invalid_message')
        if chars > MAX_MESSAGE_BYTES or len(pending) > 200_000:
            raise _ProtocolError('message_oversize')
    return True


def _decode(data):
    def pairs(items):
        obj = {}
        for key, value in items:
            if key in obj:
                raise ValueError()
            obj[key] = value
        return obj
    def nonfinite(value):
        raise ValueError()
    def finite_float(value):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError()
        return number
    try:
        return json.loads(data, object_pairs_hook=pairs, parse_constant=nonfinite,
                          parse_float=finite_float)
    except (ValueError, TypeError, RecursionError, UnicodeError):
        raise _ProtocolError('invalid_message') from None


def _target(module, function):
    if type(module) is not str or type(function) is not str:
        return False
    parts = module.split('.')
    if not parts or not all(_NAME.fullmatch(part) for part in parts):
        return False
    if not _NAME.fullmatch(function) or len(module) > 200 or len(function) > 100:
        return False
    path = _ROOT.joinpath(*parts).with_suffix('.py')
    # Only code shipped under this trusted controller root; never task paths,
    # stdlib functions, symlink targets, or a payload-selected external module.
    return path.is_file() and path.resolve() == path


def _cleanup(process, allowance):
    start = time.monotonic()
    confirmed = False
    failed = False
    try:
        # Do not poll()/reap before the signal: the unreaped direct child's PID
        # cannot be recycled. Never use a remembered/reaped PID to kill a group.
        confirmed = (process.returncode is None and
                     os.getpgid(process.pid) == process.pid and
                     os.getsid(process.pid) == process.pid)
        if confirmed:
            os.killpg(process.pid, signal.SIGKILL)
        else:
            failed = True
            process.kill()
    except ProcessLookupError:
        pass
    except OSError:
        failed = True
        try:
            process.kill()  # still our unreaped direct child, never another group
        except OSError:
            pass
    try:
        process.wait(timeout=max(0, allowance - (time.monotonic() - start)))
    except (OSError, subprocess.TimeoutExpired):
        failed = True
    return {'status': 'failed' if failed else 'ok', 'group_confirmed': confirmed,
            'elapsed_sec': time.monotonic() - start}


def run_request(worker_module, worker_function, payload, *, deadline,
                cleanup_timeout=0.5):
    """Run a trusted operation once; return safe status and optional result.

    ``deadline`` is absolute, includes spawn/transfer/parse, and is not reset by
    progress. No parent callback runs while supervising. Terminal success is
    accepted only after a complete valid frame and EOF, both before deadline.
    ``cleanup.status == 'failed'`` must be surfaced by the House adapter even
    if a valid result was received. Outcome repr never includes application data.
    """
    start = time.monotonic()
    status, reason, result = 'invalid_input', 'invalid_configuration', None
    progress, count = {}, 0
    cleanup = {'status': 'not_started', 'group_confirmed': False, 'elapsed_sec': 0.0}
    request_elapsed = 0.0
    process = parent = child = selector = None
    try:
        if (type(deadline) not in (int, float) or not math.isfinite(deadline) or
                type(cleanup_timeout) not in (int, float) or
                not math.isfinite(cleanup_timeout) or not 0 < cleanup_timeout <= 5 or
                type(payload) is not dict or not _target(worker_module, worker_function) or
                signal.getsignal(signal.SIGCHLD) not in (signal.SIG_DFL, None)):
            return Outcome(status, reason, None, {}, 0, 0.0,
                           time.monotonic() - start, cleanup)
        if time.monotonic() >= deadline or not _plain_input(payload, deadline):
            status, reason = 'deadline', 'deadline'
        else:
            wire = _encode({'module': worker_module, 'function': worker_function,
                            'payload': payload})
            if time.monotonic() >= deadline:
                status, reason = 'deadline', 'deadline'
            else:
                parent, child = socket.socketpair()
                parent.setblocking(False)
                # Credentials are passed only in the framed payload. Preserve
                # certificate/proxy routing settings, not provider environment.
                env = {k: v for k, v in os.environ.items()
                       if k in {'LANG', 'LC_ALL', 'TZ', 'SSL_CERT_FILE', 'SSL_CERT_DIR'}
                       or k.lower() in {'http_proxy', 'https_proxy', 'all_proxy', 'no_proxy'}}
                process = subprocess.Popen(
                    [sys.executable, '-I', '-B', '-S', str(Path(__file__).resolve()),
                     '--worker', str(child.fileno())], pass_fds=(child.fileno(),),
                    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL, close_fds=True, start_new_session=True,
                    env=env)
                child.close()
                child = None
                selector = selectors.DefaultSelector()
                selector.register(parent, selectors.EVENT_WRITE | selectors.EVENT_READ)
                sent, received, buffer, terminal = 0, 0, bytearray(), None
                status, reason = 'worker_error', 'worker_exit'
                while True:
                    left = deadline - time.monotonic()
                    if left <= 0:
                        status, reason, result = 'deadline', 'deadline', None
                        break
                    events = selector.select(left)
                    if time.monotonic() >= deadline:
                        status, reason, result = 'deadline', 'deadline', None
                        break
                    eof = False
                    for _, mask in events:
                        if mask & selectors.EVENT_WRITE:
                            try:
                                sent += parent.send(memoryview(wire)[sent:sent + 65536])
                            except BlockingIOError:
                                pass
                            if sent == len(wire):
                                selector.modify(parent, selectors.EVENT_READ)
                        if mask & selectors.EVENT_READ:
                            try:
                                block = parent.recv(65536)
                            except BlockingIOError:
                                continue
                            if not block:
                                eof = True
                                break
                            received += len(block)
                            if received > MAX_TOTAL_BYTES + 4 * (MAX_PROGRESS_MESSAGES + 1):
                                raise _ProtocolError('message_limit')
                            buffer.extend(block)
                            while len(buffer) >= 4:
                                size = struct.unpack('!I', buffer[:4])[0]
                                if size > MAX_MESSAGE_BYTES:
                                    raise _ProtocolError('message_oversize')
                                if len(buffer) < size + 4:
                                    break
                                if terminal is not None:
                                    raise _ProtocolError('invalid_message')
                                message = _decode(bytes(buffer[4:4 + size]))
                                del buffer[:4 + size]
                                if time.monotonic() >= deadline:
                                    break
                                if type(message) is not dict:
                                    raise _ProtocolError('invalid_message')
                                kind = message.get('type')
                                if kind == 'progress' and set(message) == {'type', 'value'}:
                                    if size > MAX_PROGRESS_BYTES:
                                        raise _ProtocolError('message_oversize')
                                    if count >= MAX_PROGRESS_MESSAGES:
                                        raise _ProtocolError('message_limit')
                                    progress.update(_progress(message['value']))
                                    count += 1
                                elif kind == 'result' and set(message) == {'type', 'value'}:
                                    if type(message['value']) is not dict:
                                        raise _ProtocolError('invalid_result')
                                    terminal = ('result', None, message['value'])
                                elif (kind == 'error' and set(message) == {'type', 'reason'} and
                                      type(message['reason']) is str and
                                      message['reason'] in _PROTOCOL_REASONS | {'worker_exception'}):
                                    terminal = ('worker_error' if message['reason'] == 'worker_exception'
                                                else 'protocol_error', message['reason'], None)
                                else:
                                    raise _ProtocolError('invalid_message')
                    if eof:
                        if time.monotonic() >= deadline:
                            status, reason = 'deadline', 'deadline'
                        elif buffer:
                            raise _ProtocolError('truncated_message')
                        elif terminal is not None:
                            status, reason, result = terminal
                        break
    except _ProtocolError as exc:
        status, reason, result = ('protocol_error' if process else 'invalid_input'), str(exc), None
    except OSError:
        status, reason, result = ('worker_error' if process else 'start_error'), 'worker_io', None
    finally:
        request_elapsed = time.monotonic() - start
        if selector is not None:
            selector.close()
        if parent is not None:
            parent.close()
        if child is not None:
            child.close()
        if process is not None:
            cleanup = _cleanup(process, cleanup_timeout)
    return Outcome(status, reason, result, progress, count, request_elapsed,
                   time.monotonic() - start, cleanup)


def _worker(fd):
    channel = socket.socket(fileno=fd)
    channel.set_inheritable(False)
    count = 0
    def send(message, *, reason='invalid_message', limit=MAX_MESSAGE_BYTES):
        channel.sendall(_encode(message, reason=reason, limit=limit))
    def emit(value):
        nonlocal count
        if count >= MAX_PROGRESS_MESSAGES:
            raise _ProtocolError('message_limit')
        send({'type': 'progress', 'value': _progress(value)}, limit=MAX_PROGRESS_BYTES)
        count += 1
    def exact(size):
        data = bytearray()
        while len(data) < size:
            part = channel.recv(min(65536, size - len(data)))
            if not part:
                raise _ProtocolError('truncated_message')
            data.extend(part)
        return data
    try:
        size = struct.unpack('!I', exact(4))[0]
        if size > MAX_MESSAGE_BYTES:
            raise _ProtocolError('message_oversize')
        request = _decode(exact(size))
        if (type(request) is not dict or set(request) != {'module', 'function', 'payload'} or
                type(request['payload']) is not dict or
                not _target(request['module'], request['function'])):
            raise _ProtocolError('invalid_message')
        sys.path.insert(0, str(_ROOT))  # fixed trusted controller root, never a task directory
        operation = getattr(importlib.import_module(request['module']), request['function'])
        result = operation(request['payload'], emit)
        if type(result) is not dict:
            raise _ProtocolError('invalid_result')
        send({'type': 'result', 'value': result}, reason='invalid_result')
    except BaseException as exc:
        reason = str(exc) if isinstance(exc, _ProtocolError) else 'worker_exception'
        if reason not in _PROTOCOL_REASONS:
            reason = 'worker_exception'
        try:
            send({'type': 'error', 'reason': reason})
        except BaseException:
            pass
    finally:
        channel.close()


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--worker' and sys.argv[2].isdigit():
        _worker(int(sys.argv[2]))
