"""One-shot House transport; no hidden retries and no credential-bearing logs."""
from __future__ import annotations

import ast
import http.client
import json
import math
import os
import re
import textwrap
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass

from .request_deadline import run_request as supervise_request
from .house_supervisor import publish_progress, request_worker, supervised_chat


class HouseError(RuntimeError):
    def __init__(self, message, *, kind='infrastructure', code='house_error', line=None):
        super().__init__(message)
        self.kind = kind
        self.code = code
        # Never retain SyntaxError.text, filenames, or arbitrary response text.
        self.line = line if type(line) is int and 0 < line <= 1_000_000 else None
        # The failed request's own HTTP status and Retry-After seconds (None when it had none, or when nothing was
        # sent). Set by HouseClient for every request it sent, so a caller of concurrent requests reads each
        # request's own values, never the client's last-request diagnostics.
        self.http_status = None
        self.retry_after_sec = None


@dataclass(frozen=True, repr=False)
class HouseConfig:
    endpoint: str
    model: str
    token: str

    def __post_init__(self):
        try:
            parsed = urllib.parse.urlsplit(self.endpoint)
            valid = (isinstance(self.endpoint, str) and parsed.scheme in ('http', 'https')
                     and parsed.hostname and not parsed.username and not parsed.password
                     and not parsed.query and not parsed.fragment and parsed.path in ('', '/'))
        except (ValueError, TypeError, AttributeError):
            valid = False
        if not valid:
            raise HouseError('MODEL_ENDPOINT must be an HTTP(S) origin without credentials or a path',
                             kind='configuration', code='invalid_configuration')
        if not all(isinstance(value, str) and value for value in (self.model, self.token)):
            raise HouseError('MODEL_NAME and MODEL_TOKEN must be nonempty',
                             kind='configuration', code='invalid_configuration')

    @classmethod
    def from_env(cls, env=None):
        env = os.environ if env is None else env
        keys = ('MODEL_ENDPOINT', 'MODEL_NAME', 'MODEL_TOKEN')
        missing = [key for key in keys if not env.get(key)]
        if missing:
            raise HouseError('Missing House configuration: ' + ', '.join(missing),
                             kind='configuration', code='missing_configuration')
        return cls(*(env[key] for key in keys))


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


RESPONSE_CAP = 2_000_001  # bytes read from one reply; more is a local limit


class _ReplyDeadline(TimeoutError):
    """The monotonic deadline was explicitly observed, not inferred from a socket error."""


class _ReadTimeoutConnection:
    def getresponse(self):
        # Capture the public socket before urllib detaches the connection. The
        # response's buffered reader retains the underlying descriptor until EOF.
        sock = self.sock
        response = super().getresponse()
        if sock is not None:
            response._t1_set_read_timeout = sock.settimeout
        return response


class _HTTPConnection(_ReadTimeoutConnection, http.client.HTTPConnection):
    pass


class _HTTPSConnection(_ReadTimeoutConnection, http.client.HTTPSConnection):
    pass


class _HTTPHandler(urllib.request.HTTPHandler):
    def http_open(self, req):
        return self.do_open(_HTTPConnection, req)


class _HTTPSHandler(urllib.request.HTTPSHandler):
    def https_open(self, req):
        return self.do_open(_HTTPSConnection, req, context=self._context)


_TRANSPORT_ENUMS = {
    'phase': ('open', 'body_read', 'json_parse', 'shape', 'done'),
    'failure_reason': ('deadline', 'socket_timeout', 'disconnect', 'os_error', 'other',
                       'http_error', 'invalid_json', 'invalid_shape', 'response_limit', 'provider_error', 'house_error'),
    'exception_class': ('deadline', 'timeout', 'remote_disconnected', 'incomplete_read',
                        'connection_error', 'os_error', 'other', 'http_error', 'house_error'),
    'read_timeout_mode': ('remaining', 'unavailable'),
    'supervisor_status': ('result', 'deadline', 'worker_error', 'protocol_error', 'start_error', 'invalid_input'),
    'supervisor_cleanup': ('ok', 'failed', 'not_started'),
    'supervisor_reason': ('deadline', 'worker_exit', 'worker_exception', 'worker_io', 'invalid_configuration',
                          'message_oversize', 'message_limit', 'invalid_progress', 'invalid_result',
                          'invalid_message', 'truncated_message', 'cleanup_failed'),
}


def safe_transport_detail(value):
    """Only fixed enums and bounded numbers cross status/replay/export boundaries.

    Unknown or missing measurements remain absent; no headers, IDs or arbitrary
    error text are retained. This metadata never changes the legacy error code.
    """
    if not isinstance(value, dict):
        return {}
    result = {key: value[key] for key, allowed in _TRANSPORT_ENUMS.items()
              if isinstance(value.get(key), str) and value[key] in allowed}
    for key in ('headers_sec', 'first_byte_sec', 'first_non_whitespace_sec', 'duration_sec',
                'supervisor_request_sec', 'supervisor_cleanup_sec'):
        number = value.get(key)
        if type(number) in (int, float) and 0 <= number <= 100000:
            result[key] = number
    for key in ('body_bytes', 'body_chunks'):
        number = value.get(key)
        if type(number) is int and 0 <= number <= RESPONSE_CAP:
            result[key] = number
    if type(value.get('provider_error_present')) is bool:
        result['provider_error_present'] = value['provider_error_present']
    return result


def _read_bounded(response, deadline: float) -> bytes:
    """Give each body read only its remaining wall time and check again afterwards.

    Open/DNS/headers retain urllib socket-timeout semantics. Fake responses without
    the callback remain supported, but a single blocking read cannot be interrupted.
    """
    read = getattr(response, 'read1', None) or response.read
    setter = getattr(response, '_t1_set_read_timeout', None)
    receipt = getattr(response, '_t1_transport_receipt', None)
    started = getattr(response, '_t1_started', deadline)
    if receipt is not None:
        receipt['read_timeout_mode'] = 'remaining' if callable(setter) else 'unavailable'
    chunks, size = [], 0
    while size < RESPONSE_CAP:
        remaining = deadline - time.monotonic()
        if remaining < 0:
            raise _ReplyDeadline('The reply exceeded its time limit')
        if callable(setter) and not getattr(response, 'isclosed', lambda: False)():
            setter(max(remaining, 1e-9))
        chunk = read(min(65536, RESPONSE_CAP - size))
        received = time.monotonic()
        if chunk:
            chunks.append(chunk)
            size += len(chunk)
            if receipt is not None:
                receipt['body_bytes'] = size
                receipt['body_chunks'] += 1
                elapsed = round(max(0, received - started), 3)
                receipt.setdefault('first_byte_sec', elapsed)
                if chunk.strip(b' \t\r\n'):
                    receipt.setdefault('first_non_whitespace_sec', elapsed)
                callback = getattr(response, '_t1_progress', None)
                if callback is not None:
                    callback()
        if received > deadline:
            raise _ReplyDeadline('The reply exceeded its time limit')
        if not chunk:
            break
    return b''.join(chunks)[:RESPONSE_CAP]


def _generation_id(headers):
    try:
        value = headers.get('X-Generation-Id') if headers is not None else None
    except Exception:
        return None
    return value if isinstance(value, str) and re.fullmatch(r'gen-[A-Za-z0-9_-]{1,180}', value) else None


HTTP_ERROR_CAP = 16_384


def _http_error_metadata(data):
    """Classify a small error envelope; never retain its message/raw fields."""
    if len(data) > HTTP_ERROR_CAP:
        return {}
    try:
        value = json.loads(data)
    except (ValueError, UnicodeError, RecursionError):
        return {}
    error = value.get('error') if isinstance(value, dict) else None
    if not isinstance(error, dict):
        return {}
    result = {}
    code = error.get('code')
    if type(code) is int and 0 <= code <= 999_999:
        result['error_code'] = code
    metadata = error.get('metadata')
    metadata = metadata if isinstance(metadata, dict) else {}
    provider = metadata.get('provider_name')
    if isinstance(provider, str) and _PROVIDER.fullmatch(provider):
        result['provider_name'] = provider
    source = metadata.get('limit_source')
    source = source if source in ('provider', 'router', 'account') else 'unknown'
    message = ' '.join(text.lower() for text in (error.get('message'), metadata.get('raw'))
                       if isinstance(text, str))
    limited = bool(re.search(r'rate[-_ ]limit|too many requests|requests per minute|tokens per minute', message))
    category = 'other'
    if limited:
        category = 'rate_limit'
        if 'upstream' in message:
            source, category = 'provider', 'upstream_rate_limit'
        elif 'account' in message or 'api key' in message:
            source, category = 'account', 'account_rate_limit'
    elif ('quota' in message or 'credit' in message) and any(
            word in message for word in ('exceed', 'exhaust', 'insufficient', 'limit')):
        category = 'quota_exhausted'
    result.update(limit_source=source, rate_limit_category=category)
    return result


def _inspect_http_error(exc, deadline):
    """At most 16 KiB and 0.2 seconds; an unavailable safe timeout means no read."""
    response = exc.fp
    setter = getattr(response, '_t1_set_read_timeout', None)
    if not isinstance(response, http.client.HTTPResponse) or not callable(setter):
        return {'error_body_state': 'unavailable'}
    deadline = min(deadline, time.monotonic() + 0.2)
    data = bytearray()
    try:
        while len(data) <= HTTP_ERROR_CAP:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return {'error_body_state': 'timeout'}
            if not response.isclosed():
                setter(remaining)
            chunk = response.read1(min(4096, HTTP_ERROR_CAP + 1 - len(data)))
            data.extend(chunk)
            if len(data) > HTTP_ERROR_CAP:
                return {'error_body_state': 'oversized'}
            if not chunk:
                result = _http_error_metadata(data)
                return {'error_body_state': 'parsed', 'error_metadata': result} if result else {
                    'error_body_state': 'unparsed'}
    except TimeoutError:
        return {'error_body_state': 'timeout'}
    except Exception:
        return {'error_body_state': 'read_error'}


def _transport_failure(exc):
    """Only fixed enums escape; arbitrary exception strings/classes never do."""
    if isinstance(exc, urllib.error.URLError) and isinstance(exc.reason, BaseException):
        exc = exc.reason
    if isinstance(exc, _ReplyDeadline):
        return 'deadline', 'deadline'
    if isinstance(exc, TimeoutError):
        return 'socket_timeout', 'timeout'
    if isinstance(exc, http.client.RemoteDisconnected):
        return 'disconnect', 'remote_disconnected'
    if isinstance(exc, http.client.IncompleteRead):
        return 'disconnect', 'incomplete_read'
    if isinstance(exc, ConnectionError):
        return 'disconnect', 'connection_error'
    if isinstance(exc, OSError):
        return 'os_error', 'os_error'
    return 'other', 'other'


def _safe_usage(value) -> dict:
    if not isinstance(value, dict):
        return {}
    return {key: count for key, count in value.items()
            if key in ('prompt_tokens', 'completion_tokens', 'total_tokens')
            and type(count) is int and 0 <= count <= 2**63 - 1}


# Local evaluation only (the platform never sets T1_EVAL_BACKEND): OpenRouter serves the House model's FP8
# weights under its own path and reasoning switch; requests are pinned to that one endpoint. openrouter-bf16
# is DeepInfra's BF16 build of the same model: faster for iterating, but not the House weights, so a change
# it shows is confirmed on openrouter before it counts.
EVAL_BACKENDS = {'openrouter': {'path': '/api/v1/chat/completions',
                                'provider': {'only': ['dekallm'], 'quantizations': ['fp8'],
                                             'allow_fallbacks': False}},
                 'openrouter-bf16': {'path': '/api/v1/chat/completions',
                                     'provider': {'only': ['deepinfra'], 'quantizations': ['bf16'],
                                                  'allow_fallbacks': False}},
                 'openrouter-free': {'path': '/api/v1/chat/completions',
                                     'model': 'nvidia/nemotron-3-super-120b-a12b:free',
                                     'provider': {'only': ['nvidia'], 'allow_fallbacks': False}}}
_PROVIDER = re.compile(r'[A-Za-z0-9 ._-]{1,40}')
REASONING_CAP = 64_000  # characters of a reply's reasoning kept in memory; never logged
# T1_THINK_FIELD: where a reasoning budget goes on the House route, which passes both places through untested
# (HOUSE-MODEL.md): kw inside chat_template_kwargs (the default), top a top-level reasoning_budget, both.
THINK_FIELDS = ('kw', 'top', 'both')


def eval_throttle(env=None) -> bool:
    """T1_EVAL_THROTTLE=1: local evaluation against a public, rate-limited endpoint. The official image never sets
    it, and without it neither the transport nor the solver changes in any way."""
    return (os.environ if env is None else env).get('T1_EVAL_THROTTLE') == '1'


# S2-11: the sampling temperature is an explicit constant per request kind, passed to chat() and send_wave() (and
# through the request supervisor) instead of being written into the payload. Every kind not listed is greedy: the
# temperature 0 every request has always been sent with. 'thin_draft' is R2b's concurrent first-round draft (run
# card: k=3, T=1.0); the solver that sends a wave passes temperature_for(its kind).
GREEDY_TEMPERATURE = 0
DRAFT_TEMPERATURE = 1.0
# thin's resample switch (A2): the free-loop code request sent after two consecutive code replies that left no new
# version to run. The request is recorded as a thin_code request (its record carries this temperature); the kind
# below only names its temperature here. Moderate rather than DRAFT_TEMPERATURE: one 4,000-token code reply, not one
# of several drafts, and no top_p is sent.
RESAMPLE_TEMPERATURE = 0.6
REQUEST_TEMPERATURES = {'thin_draft': DRAFT_TEMPERATURE, 'thin_code_resample': RESAMPLE_TEMPERATURE}
MAX_TEMPERATURE = 2
# The only temperatures a request may carry: the kind constants above. Any other value is refused before anything is
# sent, so the sampling of every request is fixed by this table, not by the caller.
_ALLOWED_TEMPERATURES = frozenset({GREEDY_TEMPERATURE, *REQUEST_TEMPERATURES.values()})


def temperature_for(kind) -> int | float:
    """The sampling temperature of one request kind (GREEDY_TEMPERATURE unless REQUEST_TEMPERATURES names it)."""
    return REQUEST_TEMPERATURES.get(kind, GREEDY_TEMPERATURE)


def _kind_temperature(kind, temperature):
    """A wave's temperature must be its kind's constant (temperature_for(kind)), else invalid_configuration."""
    if temperature != temperature_for(kind):
        raise HouseError('A wave temperature must be its request kind constant', kind='configuration',
                         code='invalid_configuration')


def _temperature(value):
    """A sampling temperature: a real number in 0..MAX_TEMPERATURE. Zero is sent as the integer 0, so a greedy
    request's wire bytes are exactly what they always were."""
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= MAX_TEMPERATURE:
        raise HouseError('temperature must be a number in 0..2', kind='configuration', code='invalid_configuration')
    if value not in _ALLOWED_TEMPERATURES:
        raise HouseError('temperature must be a request kind constant', kind='configuration',
                         code='invalid_configuration')
    return 0 if value == 0 else value


# A wave (send_wave): k independent concurrent requests, never one request with n=k. Its transient failures are
# retried from ONE budget the whole wave shares: by default the allowance one sequential request has (thin.RETRY_WAITS,
# six retries), now for all k requests together, each retry round waiting the next of these seconds (or the longest
# Retry-After of the requests it resends), and only while WAVE_RETRY_FLOOR seconds remain after the wait.
WAVE_RETRY_WAITS = (2, 4, 8, 15, 25, 40)
WAVE_RETRIES = len(WAVE_RETRY_WAITS)
WAVE_RETRY_FLOOR = 30  # request seconds that must remain before the wave's deadline after a retry wait (thin.RETRY_FLOOR)
WAVE_JOIN_GRACE_SEC = 2.0  # beyond a request's own timeout (the supervisor's 0.5 s cleanup and its receipt checks)
_KIND = re.compile(r'[a-z][a-z0-9_]{0,39}')


def retryable(error) -> bool:
    """A failed request worth resending: a failed transport, or HTTP 408/409/425/429 or a server error other than
    501/505, judged by the request's OWN status (thin._transient's rule, without the shared diagnostics)."""
    if not isinstance(error, HouseError):
        return False
    if error.code == 'transport_failure':
        return True
    status = error.http_status
    return (error.code == 'http_error' and type(status) is int
            and (status in (408, 409, 425, 429) or 500 <= status <= 599 and status not in (501, 505)))


def _valid_timeout(timeout):
    try:
        valid = type(timeout) in (int, float) and math.isfinite(timeout) and timeout > 0
    except OverflowError:
        valid = False
    if not valid:
        raise HouseError('No valid generation time remaining',
                         kind='local_budget', code='generation_budget')


def _valid_conversation(messages):
    roles = [m.get('role') if isinstance(m, dict) else None for m in messages] if isinstance(messages, list) else []
    if (not roles or roles[0] != 'user' or roles[-1] != 'user'
            or any(role not in ('user', 'assistant') for role in roles)
            or any(a == b for a, b in zip(roles, roles[1:]))
            or any(not isinstance(m.get('content'), str) for m in messages)):
        raise HouseError('Conversation must alternate user/assistant text, starting and ending with user',
                         kind='local_input', code='invalid_prompt')
    try:
        prompt_bytes = sum(len(m['content'].encode()) for m in messages)
    except UnicodeError:
        raise HouseError('Prompt must be UTF-8 encodable',
                         kind='local_input', code='invalid_prompt') from None
    if prompt_bytes > 180_000:
        raise HouseError('Prompt exceeds the local B0 input safety limit',
                         kind='local_input', code='prompt_limit')


class _Sent:
    """One request's private ledger while it is in flight: its record, transport receipt and the diagnostics it
    would have written (sink). Nothing here is shared with another request; HouseClient merges it under its lock."""
    __slots__ = ('record', 'transport', 'sink', 'reply', 'error', 'refused')

    def __init__(self, record):
        self.record, self.transport, self.reply, self.error, self.refused = record, None, None, None, False
        self.sink = {'usage': {}, 'requests': [], 'transport_requests': []}

    def refuse(self):
        self.refused = True

    def fail(self, error):
        record = self.record
        status, hint = record.get('http_status'), record.get('retry_after_sec')
        error.http_status = status if type(status) is int and 100 <= status <= 599 else None
        error.retry_after_sec = hint if type(hint) is int and 0 <= hint <= 3600 else None
        self.error = error


@dataclass(frozen=True, eq=False)
class WaveRequest:
    """One request a wave sent: its record and transport receipt (the very objects appended, pairwise, to the
    client's diagnostics['requests'] and ['transport_requests']) and its reply or HouseError. Its `diagnostics` is
    a one-request view, so baseline.Replay.request(wave_request, kind, messages, ...) writes this request's row."""
    record: dict
    transport: dict
    reply: dict | None
    error: HouseError | None

    @property
    def diagnostics(self):
        return {'requests': [self.record], 'transport_requests': [self.transport]}

    def __repr__(self):
        return (f'WaveRequest(ok={self.error is None}, http_status={self.record.get("http_status")}, '
                f'error={self.error.code if self.error else None})')


@dataclass(frozen=True, eq=False)
class WaveReply:
    """One of a wave's k independent requests, after its retries: every request sent for it in order (the last is
    the outcome) and that outcome's own status. Never read from the client's shared diagnostics."""
    slot: int
    requests: tuple

    @property
    def last(self) -> WaveRequest:
        return self.requests[-1]

    @property
    def reply(self):
        return self.last.reply

    @property
    def error(self):
        return self.last.error

    @property
    def ok(self) -> bool:
        return self.last.error is None

    @property
    def http_status(self):
        status = self.last.record.get('http_status')
        return status if type(status) is int and 100 <= status <= 599 else None

    @property
    def retry_after_sec(self):
        hint = self.last.record.get('retry_after_sec')
        return hint if type(hint) is int and 0 <= hint <= 3600 else None

    @property
    def sent(self) -> int:
        return len(self.requests)

    def __repr__(self):
        return (f'WaveReply(slot={self.slot}, sent={self.sent}, ok={self.ok}, http_status={self.http_status}, '
                f'error={self.error.code if self.error else None})')


class HouseClient:
    """Each chat() is one request; the organizer admits at most 25 per unit (DEVELOPMENT-RUNTIME). send_wave() sends
    k independent requests at once. Request slots are reserved under one lock, and each request's diagnostics are
    merged under it whole, so concurrent requests never exceed max_requests or interleave their records."""

    def __init__(self, config: HouseConfig, max_requests: int = 1):
        if type(max_requests) is not int or not 1 <= max_requests <= 25:
            raise HouseError('Request budget must be 1-25', kind='configuration', code='invalid_configuration')
        self.config = config
        self.max_requests = max_requests
        self.attempts = 0
        self.waves = 0  # send_wave() calls that sent requests; each wave's records carry its number
        self._lock = threading.Lock()
        # Local evaluation only: a public endpoint's 429 refuses a request rather than admitting it, so under
        # T1_EVAL_THROTTLE it is not counted in attempts; uncounted records how many there were.
        self.throttle = eval_throttle()
        self.uncounted = 0
        self.diagnostics = {'request_attempted': False, 'admission': 'not_attempted',
                            'response_received': False, 'usage': {}, 'requests': [], 'transport_requests': []}
        backend = os.environ.get('T1_EVAL_BACKEND') or None
        if backend is not None and backend not in EVAL_BACKENDS:
            raise HouseError('Unknown evaluation backend', kind='configuration', code='invalid_configuration')
        self.backend = EVAL_BACKENDS.get(backend)
        self.backend_name = backend
        self.supervisor = os.environ.get('T1_REQUEST_SUPERVISOR') == '1'
        if self.backend and self.backend.get('model') and config.model != self.backend['model']:
            raise HouseError('The free evaluation route requires its pinned free model',
                             kind='configuration', code='invalid_configuration')
        self.think_field = os.environ.get('T1_THINK_FIELD') or 'kw'
        if self.think_field not in THINK_FIELDS:
            raise HouseError('Unknown reasoning budget field', kind='configuration', code='invalid_configuration')

    def generate(self, prompt: str, *, timeout: float = 120) -> dict:
        if not isinstance(prompt, str):
            raise HouseError('Prompt must be text', kind='local_input', code='invalid_prompt')
        reply = self.chat([{'role': 'user', 'content': prompt}], timeout=timeout)
        try:
            code = extract_python(reply['content'], reply['finish_reason_raw'])
        except HouseError as exc:
            if exc.line is not None:
                self.diagnostics['code_line'] = exc.line
            raise
        return {'code': code, 'usage': reply['usage'], 'finish_reason': reply['finish_reason']}

    def chat(self, messages: list, *, timeout: float = 120, thinking: bool | int | str = False,
             max_tokens: int = 4000, temperature: int | float = GREEDY_TEMPERATURE) -> dict:
        """Send one conversation; return the reply content and bounded metadata. thinking=True turns House
        reasoning on, at low effort so that it and an answer can share one 4,000-token reply; a whole number of
        tokens (1-3999) turns it on with that reasoning budget instead (think_field places it).
        thinking='default' enables native thinking without specifying effort or a reasoning budget.
        max_tokens is an integer output cap in 1..4000; its default leaves the request unchanged.
        temperature is the request kind's sampling temperature (temperature_for): one of the kind constants
        (any other value is refused before anything is sent); its default, 0, leaves the request unchanged. A HouseError for a request that was sent carries that request's own http_status and
        retry_after_sec; the shared diagnostics are written exactly as before, once the request has finished."""
        temperature = self._options(thinking, max_tokens, temperature)
        self._room(1, 'The per-unit request budget is spent')
        _valid_timeout(timeout)
        _valid_conversation(messages)
        request, record, budget = self._build(messages, thinking, max_tokens, temperature)
        # Attempted transport is not proof of organizer admission.
        self._reserve(1, 'The per-unit request budget is spent')
        sent = _Sent(record)
        try:
            sent.reply = self._transport(request, messages, sent, timeout=timeout, thinking=thinking,
                                         max_tokens=max_tokens, temperature=temperature, budget=budget)
            return sent.reply
        except HouseError as exc:
            sent.fail(exc)
            raise
        finally:
            with self._lock:
                self._merge(sent)

    def send_wave(self, prompts, kind: str, temperature, k: int, *, timeout: float = 120, deadline=None,
                  clock=None, thinking: bool | int | str = False, max_tokens: int = 4000,
                  retries: int = WAVE_RETRIES, pause=None) -> list:
        """k independent requests sent at once (never one request with n=k). Returns k WaveReply in slot order,
        each with its own reply or HouseError and its own http_status / retry_after_sec.

        prompts: a list of one prompt (text or conversation), sent k times, or of k, one per slot. Every record is
        tagged kind, temperature (must equal temperature_for(kind), else invalid_configuration), wave (this client's wave
        number), wave_slot, attempt (the retry round), reply_returned and timeout_sec. Every check that can refuse
        the wave runs before anything is sent, and the k request slots are then reserved at once under the client's
        lock: a wave that does not fit max_requests raises request_limit and sends nothing.

        Time: each request may take min(timeout, what remains before deadline) seconds; deadline is a clock()
        reading (clock: default time.monotonic; a solver passes its unit clock), by default clock() + timeout.
        Transient failures (retryable, by each request's own status) are resent in rounds from ONE retry budget the
        whole wave shares (retries; lowest slot first). A round waits the next WAVE_RETRY_WAITS seconds, or the
        longest Retry-After among the requests it resends (pause(seconds, statuses); default time.sleep), only while
        WAVE_RETRY_FLOOR seconds remain after the wait, and reserves its slots under the lock. Each round's requests
        are merged into the diagnostics in slot order, as if sent one after another; backoff_sec adds the waits. A
        request still running WAVE_JOIN_GRACE_SEC after its timeout (a transport that ignored it) is recorded as a
        deadline transport failure and can no longer write to the client."""
        if type(k) is not int or not 1 <= k <= 25:
            raise HouseError('A wave sends 1-25 requests', kind='configuration', code='invalid_configuration')
        if not isinstance(kind, str) or not _KIND.fullmatch(kind):
            raise HouseError('A wave needs a request kind', kind='configuration', code='invalid_configuration')
        if type(retries) is not int or not 0 <= retries <= 24:
            raise HouseError('A wave retry budget is 0-24 requests', kind='configuration',
                             code='invalid_configuration')
        clock = time.monotonic if clock is None else clock
        pause = (lambda seconds, statuses: time.sleep(seconds)) if pause is None else pause
        if not callable(clock) or not callable(pause):
            raise HouseError('A wave clock and pause must be callable', kind='configuration',
                             code='invalid_configuration')
        temperature = self._options(thinking, max_tokens, temperature)
        _kind_temperature(kind, temperature)
        if not isinstance(prompts, (list, tuple)) or len(prompts) not in (1, k):
            raise HouseError('A wave takes one prompt or k prompts', kind='local_input', code='invalid_prompt')
        conversations = [[{'role': 'user', 'content': prompt}] if isinstance(prompt, str) else prompt
                         for prompt in prompts]
        conversations = conversations * k if len(conversations) == 1 else conversations
        self._room(k, 'The wave does not fit the per-unit request budget')
        _valid_timeout(timeout)
        now = clock()
        end = now + timeout if deadline is None else deadline
        _valid_timeout(end - now if type(end) in (int, float) else None)
        for messages in conversations:
            _valid_conversation(messages)
        request_timeout = min(timeout, end - now)
        self._reserve(k, 'The wave does not fit the per-unit request budget')
        with self._lock:
            self.waves += 1
            wave = self.waves
        history = [[] for _ in range(k)]
        pending, left, round_ = list(range(k)), retries, 0
        while True:
            sents = self._send_round(conversations, pending, request_timeout, thinking, max_tokens, temperature)
            with self._lock:
                for slot, sent in zip(pending, sents):
                    sent.record.update(kind=kind, attempt=round_, reply_returned=sent.error is None,
                                       temperature=temperature, wave=wave, wave_slot=slot,
                                       timeout_sec=round(request_timeout, 1))
                    self._merge(sent)
            for slot, sent in zip(pending, sents):
                history[slot].append(WaveRequest(sent.record, sent.transport, sent.reply, sent.error))
            failed = [slot for slot in pending if retryable(history[slot][-1].error)]
            with self._lock:
                room = min(len(failed), left, self.max_requests - self.attempts)
            if room <= 0 or round_ >= len(WAVE_RETRY_WAITS):
                break
            hints = [history[slot][-1].error.retry_after_sec for slot in failed[:room]]
            wait = max([WAVE_RETRY_WAITS[round_]] + [hint for hint in hints if type(hint) is int and hint > 0])
            if end - clock() - wait < WAVE_RETRY_FLOOR:  # no retried reply would still fit after the wait
                break
            pause(wait, tuple(history[slot][-1].error.http_status for slot in failed[:room]))
            request_timeout = min(timeout, end - clock())
            if request_timeout <= 0:
                break
            with self._lock:
                self.diagnostics['backoff_sec'] = self.diagnostics.get('backoff_sec', 0) + wait
                room = min(room, self.max_requests - self.attempts)
                if room <= 0:
                    break
                self.attempts += room
            left -= room
            pending = failed[:room]
            round_ += 1
        return [WaveReply(slot, tuple(history[slot])) for slot in range(k)]

    def _send_round(self, conversations, slots, timeout, thinking, max_tokens, temperature):
        """One request per slot (slots already reserved), all at once, each on its own thread and written only to
        its own _Sent; returns them in slot order once each has finished or been abandoned."""
        sents, threads, templates = [], [], []
        for slot in slots:
            request, record, budget = self._build(conversations[slot], thinking, max_tokens, temperature)
            templates.append(dict(record))
            sent = _Sent(record)
            options = dict(timeout=timeout, thinking=thinking, max_tokens=max_tokens, temperature=temperature,
                           budget=budget)
            sents.append(sent)
            threads.append(threading.Thread(target=self._run, args=(request, conversations[slot], sent, options),
                                            name=f't1-house-wave-{slot}', daemon=True))
        started = time.monotonic()
        live = []
        for index, thread in enumerate(threads):
            try:
                thread.start()
                live.append(index)
            except RuntimeError:  # no thread could start: the reserved request was never sent
                sents[index] = _unfinished(templates[index], sents[index].transport, started, 'other')
        limit = started + timeout + WAVE_JOIN_GRACE_SEC
        for index in live:
            threads[index].join(max(0.0, limit - time.monotonic()))
            if threads[index].is_alive():
                sents[index] = _unfinished(templates[index], sents[index].transport, started, 'deadline')
        return sents

    def _run(self, request, messages, sent, options):
        """A wave thread: one request, written only to its own `sent`."""
        try:
            sent.reply = self._transport(request, messages, sent, **options)
        except HouseError as exc:
            sent.fail(exc)
        except BaseException:  # never lose a reserved request: a failure the transport did not classify
            sent.fail(HouseError('House transport failure; admission unknown', code='transport_failure'))

    def _options(self, thinking, max_tokens, temperature):
        if isinstance(thinking, str) and thinking != 'default':
            raise HouseError('thinking string must be default',
                             kind='configuration', code='invalid_configuration')
        if type(max_tokens) is not int or not 1 <= max_tokens <= 4000:
            raise HouseError('max_tokens must be an integer in 1..4000',
                             kind='configuration', code='invalid_configuration')
        return _temperature(temperature)

    def _room(self, slots, message):
        with self._lock:
            if self.attempts + slots > self.max_requests:
                raise HouseError(message, kind='local_budget', code='request_limit')

    def _reserve(self, slots, message):
        """Take `slots` request slots at once, or none (request_limit): the check and the count are one step."""
        with self._lock:
            if self.attempts + slots > self.max_requests:
                raise HouseError(message, kind='local_budget', code='request_limit')
            self.attempts += slots

    def _build(self, messages, thinking, max_tokens, temperature):
        """The HTTP request, its record and its reasoning budget (None: none). Sends nothing."""
        default_thinking = isinstance(thinking, str) and thinking == 'default'
        # House thinks by default (HOUSE-MODEL.md); each 4,000-token reply must carry code, not reasoning.
        # A planning request thinks at low effort; a whole-number budget (the T1_*THINK switches) thinks within
        # that budget, placed by think_field. Both are passed through; their effect is not promised.
        payload = {'model': self.config.model, 'max_tokens': max_tokens, 'temperature': temperature, 'stream': False,
                   'messages': [{'role': m['role'], 'content': m['content']} for m in messages],
                   'chat_template_kwargs': ({'enable_thinking': True} if default_thinking else
                                            {'enable_thinking': True, 'low_effort': True} if thinking is True
                                            else {'enable_thinking': False})}
        budget = thinking if type(thinking) is int and 0 < thinking < 4000 else None
        if budget is not None:  # a bounded budget instead of low effort (passed through; its effect is not promised)
            payload['chat_template_kwargs'] = dict(enable_thinking=True, **(
                {'reasoning_budget': budget} if self.think_field in ('kw', 'both') else {}))
            if self.think_field in ('top', 'both'):
                payload['reasoning_budget'] = budget
        path = '/v1/chat/completions'
        if self.backend is not None:  # the same switches in OpenRouter's terms, pinned to one endpoint
            del payload['chat_template_kwargs']
            payload.pop('reasoning_budget', None)
            payload.update(reasoning=({'enabled': True} if default_thinking else
                                      {'max_tokens': budget} if budget is not None else
                                      {'effort': 'low'} if thinking is True else {'enabled': False}),
                           provider=self.backend['provider'])
            path = self.backend['path']
        request = urllib.request.Request(self.config.endpoint.rstrip('/')+path,
            data=json.dumps(payload).encode(), method='POST',
            headers={'Content-Type': 'application/json', 'Authorization': 'Bearer '+self.config.token})
        record = {} if budget is None else {'reasoning_budget': budget}
        if max_tokens != 4000:
            record['max_tokens'] = max_tokens
        if temperature != GREEDY_TEMPERATURE:
            record['temperature'] = temperature
        return request, record, budget

    def _transport(self, request, messages, sent, *, timeout, thinking, max_tokens, temperature, budget):
        """Send one prepared request whose slot is reserved. It writes only to `sent` (its record, its transport
        receipt and its own diagnostics sink, in the order the sequential client wrote the shared diagnostics),
        never to anything another request shares. Returns the reply or raises HouseError."""
        record, diag = sent.record, sent.sink
        started = time.monotonic()
        transport = sent.transport = {'phase': 'open', 'body_bytes': 0, 'body_chunks': 0}
        if self.supervisor:
            try:
                return supervised_chat(self, messages, timeout=timeout, thinking=thinking,
                                       max_tokens=max_tokens, started=started, record=record,
                                       transport=transport, supervise=supervise_request,
                                       temperature=temperature, diag=diag, refused=sent.refuse)
            finally:
                self._record_request(record, transport, started, diag)
        try:
            publish_progress(self, transport, record, started)
            opener = urllib.request.build_opener(_NoRedirect(), _HTTPHandler, _HTTPSHandler)
            with opener.open(request, timeout=timeout) as response:
                diag['response_received'] = True
                transport['headers_sec'] = round(max(0, time.monotonic() - started), 3)
                transport['phase'] = 'body_read'
                transport['generation_id'] = _generation_id(getattr(response, 'headers', None))
                response._t1_transport_receipt = transport
                response._t1_started = started
                response._t1_progress = lambda: publish_progress(self, transport, record, started)
                status = getattr(response, 'status', None)
                if type(status) is int and 100 <= status <= 599:
                    diag['http_status'] = record['http_status'] = status
                publish_progress(self, transport, record, started)
                data = _read_bounded(response, started + timeout)
            diag['response_bytes'] = len(data)
            if len(data) > 2_000_000:
                raise HouseError('House response exceeds local limit',
                                 kind='local_limit', code='response_limit')
            transport['phase'] = 'json_parse'
            publish_progress(self, transport, record, started)
            try:
                result = json.loads(data)
            except (ValueError, UnicodeError, RecursionError):
                raise HouseError('House response is not valid JSON', code='response_invalid_json') from None
            transport['phase'] = 'shape'
            publish_progress(self, transport, record, started)
            if not isinstance(result, dict):
                raise HouseError('House response has an invalid shape', code='response_invalid_shape')
            if result.get('error') is not None:
                transport['provider_error_present'] = True
            # Persist bounded metadata BEFORE extraction, including for rejected model code.
            usage = _safe_usage(result.get('usage'))
            total = diag['usage']
            for key, count in usage.items():
                total[key] = total.get(key, 0) + count
            for key in ('completion_tokens', 'prompt_tokens'):
                if key in usage:
                    record[key] = usage[key]
            provider = result.get('provider')
            if isinstance(provider, str) and _PROVIDER.fullmatch(provider):
                record['provider'] = provider
            if result.get('error') is not None:
                raise HouseError('House provider reported an error', code='provider_error')
            choices = result.get('choices')
            if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
                raise HouseError('House response has an invalid choice', code='response_invalid_shape')
            choice = choices[0]
            finish = choice.get('finish_reason')
            if choice.get('error') is not None or finish == 'error':
                transport['provider_error_present'] = True
                raise HouseError('House provider reported an error', code='provider_error')
            safe_finish = finish if isinstance(finish, str) and finish in (
                'stop', 'length', 'content_filter', 'tool_calls', 'function_call') else 'unknown'
            diag['finish_reason'] = record['finish_reason'] = safe_finish
            message = choice.get('message')
            if not isinstance(message, dict):
                raise HouseError('House response has an invalid message', code='response_invalid_shape')
            content = message.get('content')
            record['content_chars'] = len(content) if isinstance(content, str) else 0
            # Lengths only: shows whether thinking stayed off. vLLM may mirror one text under both names.
            reasoning = [len(text) for text in (message.get('reasoning_content'), message.get('reasoning'))
                         if isinstance(text, str)]
            record['reasoning_chars'] = max(reasoning, default=0)
            if reasoning:
                diag['reasoning_chars'] = diag.get('reasoning_chars', 0) + max(reasoning)
            texts = [text for text in (message.get('reasoning_content'), message.get('reasoning'))
                     if isinstance(text, str)]
            transport['phase'] = 'done'
            publish_progress(self, transport, record, started)
            return {'content': content, 'finish_reason': safe_finish,
                    'finish_reason_raw': finish, 'usage': usage,
                    'reasoning': max(texts, key=len)[:REASONING_CAP] if texts else ''}
        except urllib.error.HTTPError as exc:
            transport.update(headers_sec=round(max(0, time.monotonic() - started), 3),
                             failure_reason='http_error', exception_class='http_error',
                             generation_id=_generation_id(exc.headers))
            publish_progress(self, transport, {'http_status': exc.code}, started)
            transport.update(_inspect_http_error(exc, started + timeout))
            status = exc.code
            hint = _retry_after(exc.headers)
            if hint is not None:
                diag['retry_after_sec'] = record['retry_after_sec'] = hint
            exc.close()
            if type(status) is int and 100 <= status <= 599:
                diag.update(http_status=status, response_received=True)
                record['http_status'] = status
            if self.throttle and status == 429:  # local evaluation only: refused, not admitted
                sent.refuse()
            # A status may originate at a local proxy. Do not infer organizer admission.
            raise HouseError('House HTTP error; admission unknown', code='http_error') from None
        except HouseError as exc:
            transport['failure_reason'] = {'response_invalid_json': 'invalid_json',
                                          'response_invalid_shape': 'invalid_shape',
                                          'response_limit': 'response_limit',
                                          'provider_error': 'provider_error'}.get(exc.code, 'house_error')
            transport['exception_class'] = 'house_error'
            raise
        except Exception as exc:
            transport['failure_reason'], transport['exception_class'] = _transport_failure(exc)
            raise HouseError('House transport failure; admission unknown',
                             code='transport_failure') from None
        finally:
            self._record_request(record, transport, started, diag)

    @staticmethod
    def _record_request(record, transport, started, diag):
        record['duration_sec'] = diag['duration_sec'] = round(time.monotonic() - started, 3)
        diag['requests'].append(record)
        transport['duration_sec'] = record['duration_sec']
        # Keep bounded local diagnosis; status, replay and public exports sanitize it.
        diag['transport_requests'].append(transport)

    def _merge(self, sent):
        """Fold one finished request into the shared diagnostics exactly as the sequential client wrote them while
        it ran: this request's status and Retry-After replace the last ones; response_received, response_bytes,
        finish_reason and duration_sec are the last written; usage and reasoning_chars add up; its record and
        transport receipt are appended as one pair. Keys follow the order the request wrote them. The caller
        holds self._lock, so concurrent requests never interleave or lose counts."""
        shared = self.diagnostics
        shared.update(request_attempted=True, admission='unknown')
        shared.pop('http_status', None)
        shared.pop('retry_after_sec', None)
        for key, value in sent.sink.items():
            if key == 'usage':
                total = shared['usage']
                for name, count in value.items():
                    total[name] = total.get(name, 0) + count
            elif key in ('requests', 'transport_requests'):
                shared[key].extend(value)
            elif key == 'reasoning_chars':
                shared['reasoning_chars'] = shared.get('reasoning_chars', 0) + value
            elif key == 'response_received':
                if value is True:
                    shared['response_received'] = True
            else:  # http_status, retry_after_sec, response_bytes, finish_reason, duration_sec
                shared[key] = value
        if sent.refused:  # local evaluation only (T1_EVAL_THROTTLE): a refused 429 is not an admitted request
            self.attempts -= 1
            self.uncounted += 1


def _unfinished(template, transport, started, reason):
    """A reserved request whose thread never started ('other') or never finished in time ('deadline'): a transport
    failure in a fresh ledger (its record template and what its receipt last showed), as a running thread may
    still write to its own."""
    try:
        live = dict(transport) if isinstance(transport, dict) else {}
    except RuntimeError:  # changed while copied
        live = {}
    fresh = _Sent(dict(template))
    duration = round(time.monotonic() - started, 3)
    fresh.transport = {key: live[key] for key in ('phase', 'body_bytes', 'body_chunks', 'headers_sec',
                                                   'first_byte_sec', 'first_non_whitespace_sec') if key in live}
    fresh.transport.setdefault('phase', 'open')
    fresh.transport.setdefault('body_bytes', 0)
    fresh.transport.setdefault('body_chunks', 0)
    fresh.transport.update(failure_reason=reason, exception_class='deadline' if reason == 'deadline' else 'other',
                           duration_sec=duration)
    fresh.record['duration_sec'] = fresh.sink['duration_sec'] = duration
    fresh.sink['requests'].append(fresh.record)
    fresh.sink['transport_requests'].append(fresh.transport)
    fresh.fail(HouseError('House transport failure; admission unknown', code='transport_failure'))
    return fresh


def house_request_worker(payload, emit):
    """Constant trusted worker entry; never selected by model output."""
    return request_worker(payload, emit)


def _retry_after(headers):
    """Retry-After in whole seconds (the delta form only; an HTTP date is ignored), capped at an hour."""
    try:
        value = headers.get('Retry-After') if headers is not None else None
    except Exception:
        return None
    if not isinstance(value, str) or not re.fullmatch(r'\s*\d{1,6}\s*', value):
        return None
    return min(int(value), 3600)


_OPEN_FENCE = re.compile(r'^\s*(`{3,}|~{3,})\s*([^\s`~]*)(.*)$')
_TRAILING_FENCE = re.compile(r'(`{3,}|~{3,})\s*$')


def _python_tag(tag):
    tag = tag.lower()
    return (tag in ('py', 'py3') or tag.endswith('.py')
            or re.fullmatch(r'python3?(\.\d+)*', tag) is not None)


def _fenced_blocks(content):
    """Line-based fences, tolerant of closers glued to code or followed by prose.

    A non-python block (e.g. ````markdown) is searched for nested fences as well.
    """
    blocks, lines, index = [], content.split('\n'), 0
    while index < len(lines):
        match = _OPEN_FENCE.match(lines[index])
        index += 1
        if not match:
            continue
        fence, tag, rest = match.group(1), match.group(2).lower(), match.group(3).strip()
        if rest.endswith(fence):  # ```python print(1)```
            blocks.append((tag, rest[:-len(fence)]))
            continue
        body = []
        while index < len(lines):
            line = lines[index]
            index += 1
            if line.strip().startswith(fence):
                break
            glued = _TRAILING_FENCE.search(line)
            if glued and glued.group(1).startswith(fence):
                body.append(line[:glued.start()])
                break
            body.append(line)
        text = textwrap.dedent('\n'.join(body))
        blocks.append((tag, text))
        if not _python_tag(tag):
            blocks.extend(_fenced_blocks(text))
    return blocks


def _syntax_error_line(code):
    """None when the code parses; otherwise the error line (0 when unknown)."""
    try:
        ast.parse(code)
    except SyntaxError as exc:
        return exc.lineno or 0
    except (ValueError, RecursionError, MemoryError):  # MemoryError: parser stack overflow
        return 0
    return None


def _is_program(code):
    """Untagged fences also hold output examples: require a statement, not just a value."""
    if not code.strip() or _syntax_error_line(code) is not None:
        return False
    return any(not isinstance(node, (ast.Expr, ast.AnnAssign)) or isinstance(node.value, ast.Call)
               for node in ast.parse(code).body)


def strip_reasoning(content: str) -> str:
    """The answer part of a reply. Reasoning precedes the answer: a leading <think> block, or text before a
    bare </think>. Never strip inside the answer, where '<think>' may be ordinary program text."""
    content = content.replace('\r\n', '\n')
    content = re.sub(r'^\s*<think>.*?(?:</think>|\Z)', '', content, flags=re.S|re.I)
    close = re.search(r'^\s*</think>\s*$', content, flags=re.M|re.I)
    if close and not re.search(r'<think>', content[:close.start()], flags=re.I):
        content = content[close.end():]
    return content


def extract_python(content, finish_reason):
    if finish_reason != 'stop' or not isinstance(content, str) or not content.strip():
        raise HouseError('Missing or incomplete model code; nothing executed',
                         kind='model_output', code='incomplete_code')
    content = strip_reasoning(content)
    if not content.strip():  # reasoning only: an empty program would "succeed" with no output
        raise HouseError('Model output contains no Python program',
                         kind='model_output', code='empty_program')
    blocks = _fenced_blocks(content)
    tagged = [code for tag, code in blocks if _python_tag(tag)]
    candidates =([code for code in tagged if code.strip() and _syntax_error_line(code) is None]
                  or [code for tag, code in blocks if not tag and _is_program(code)])
    if not blocks and _syntax_error_line(content) is None:
        candidates = [content]
    if not candidates:  # The earlier regex parser, so no reply regresses.
        candidates = [code for code in re.findall(r'```(?:python|py)?\s*\n(.*?)```', content, flags=re.S)
                      if code.strip() and _syntax_error_line(code) is None]
    if (not candidates and tagged and not any(code.strip() for code in tagged)
            and not any(code.strip() for tag, code in blocks if not tag)):
        raise HouseError('Model output contains no Python program',
                         kind='model_output', code='empty_program')
    if not candidates:
        pool = [code for code in tagged if code.strip()] or [code for _, code in blocks if code.strip()]
        line = _syntax_error_line(max(pool, key=len) if pool else content)
        raise HouseError('Model output is not a complete Python program',
                         kind='model_output', code='invalid_python', line=line)
    return max(candidates, key=len).strip()+'\n'
