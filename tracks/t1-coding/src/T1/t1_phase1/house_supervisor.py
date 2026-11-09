"""Opt-in House adapter for the trusted one-request deadline supervisor.

No retry, route fallback or partial-content extraction occurs here. The caller
keeps the original request counter, and only a complete worker receipt supplies
usage. A cleanup failure rejects success but retains any already verified usage;
the existing controller's transport_failure retry policy is unchanged.

S2-11: the request's sampling temperature is carried to the worker explicitly
(never a worker default), and the adapter writes only the one request's own
diagnostics sink (diag) and reports a refused throttled 429 through `refused`,
so concurrent supervised requests share nothing but the client's locked merge.
"""
from __future__ import annotations

import math
import os
import re
import time


_ERROR_CODES = {'invalid_configuration', 'generation_budget', 'invalid_prompt',
                'prompt_limit', 'request_limit', 'response_limit', 'response_invalid_json',
                'response_invalid_shape', 'provider_error', 'http_error', 'transport_failure'}
_ERROR_KINDS = {'infrastructure', 'configuration', 'local_budget', 'local_input', 'local_limit'}


def publish_progress(client, transport, record, started):
    callback = getattr(client, '_progress_callback', None)
    if callback is not None:
        callback(transport, record, started)


def _local_detail(value):
    from .house import safe_transport_detail, _PROVIDER
    result = safe_transport_detail(value)
    generation = value.get('generation_id')
    if isinstance(generation, str) and re.fullmatch(r'gen-[A-Za-z0-9_-]{1,180}', generation):
        result['generation_id'] = generation
    if value.get('error_body_state') in ('unavailable', 'timeout', 'oversized', 'parsed',
                                        'unparsed', 'read_error'):
        result['error_body_state'] = value['error_body_state']
    metadata = value.get('error_metadata')
    if isinstance(metadata, dict):
        safe = {}
        if type(metadata.get('error_code')) is int and 0 <= metadata['error_code'] <= 999999:
            safe['error_code'] = metadata['error_code']
        if isinstance(metadata.get('provider_name'), str) and _PROVIDER.fullmatch(metadata['provider_name']):
            safe['provider_name'] = metadata['provider_name']
        if metadata.get('limit_source') in ('provider', 'router', 'account', 'unknown'):
            safe['limit_source'] = metadata['limit_source']
        if metadata.get('rate_limit_category') in ('other', 'rate_limit', 'upstream_rate_limit',
                                                  'account_rate_limit', 'quota_exhausted'):
            safe['rate_limit_category'] = metadata['rate_limit_category']
        if safe:
            result['error_metadata'] = safe
    return result


def request_worker(payload, emit):
    """Constant trusted entry; execute the original direct House chat exactly once."""
    from .house import HouseClient, HouseConfig, HouseError, EVAL_BACKENDS, THINK_FIELDS, MAX_TEMPERATURE
    from .house import _ALLOWED_TEMPERATURES
    backend = payload['backend']
    if backend is not None and backend not in EVAL_BACKENDS:
        raise ValueError('invalid worker backend')
    if payload['think_field'] not in THINK_FIELDS or type(payload['throttle']) is not bool:
        raise ValueError('invalid worker options')
    temperature = payload['temperature']
    if (type(temperature) not in (int, float) or not math.isfinite(temperature)
            or not 0 <= temperature <= MAX_TEMPERATURE or temperature not in _ALLOWED_TEMPERATURES):
        raise ValueError('invalid worker temperature')
    # Explicit non-secret selectors, never inherited provider configuration. In
    # particular the child must not silently revert OpenRouter to /v1.
    if backend is None:
        os.environ.pop('T1_EVAL_BACKEND', None)
    else:
        os.environ['T1_EVAL_BACKEND'] = backend
    os.environ['T1_THINK_FIELD'] = payload['think_field']
    os.environ['T1_EVAL_THROTTLE'] = '1' if payload['throttle'] else '0'
    os.environ['T1_REQUEST_SUPERVISOR'] = '0'  # no recursive worker launch
    client = HouseClient(HouseConfig(**payload['config']), max_requests=1)
    started_offset = 0.0
    sent = 0
    last_emit = -math.inf
    previous = {}
    interval = max(0.005, payload['timeout'] / 100)

    def progress(transport, record, started):
        nonlocal last_emit, previous, started_offset, sent
        started_offset = max(0.0, started - payload['started'])
        value = {key: transport[key] for key in ('phase', 'body_bytes', 'body_chunks') if key in transport}
        if value.get('body_chunks', 0) > 1_000_000:
            value.pop('body_chunks', None)
        for key, target in (('headers_sec', 'headers_elapsed_sec'),
                            ('first_byte_sec', 'first_byte_sec'),
                            ('first_non_whitespace_sec', 'first_non_whitespace_sec')):
            if key in transport:
                value[target] = round(transport[key] + started_offset, 3)
        if type(record.get('http_status')) is int and 100 <= record['http_status'] <= 599:
            value['http_status'] = record['http_status']
        now = time.monotonic()
        important = (value.get('phase') != previous.get('phase') or
                     any(key in value and key not in previous for key in
                         ('http_status', 'first_byte_sec', 'first_non_whitespace_sec')))
        # Bound message count independently of byte/chunk fragmentation. The
        # final receipt carries complete counters even when intermediate samples
        # are coalesced. Last observed progress is not a claim of final content.
        if sent < 120 and (important or now - last_emit >= interval):
            emit(value)
            sent += 1
            last_emit = now
            previous = value
    client._progress_callback = progress
    reply = error = None
    try:
        reply = client.chat(payload['messages'], timeout=payload['deadline'] - time.monotonic(),
                            thinking=payload['thinking'], max_tokens=payload['max_tokens'],
                            temperature=temperature)
    except HouseError as exc:
        error = {'code': exc.code if exc.code in _ERROR_CODES else 'transport_failure',
                 'kind': exc.kind if exc.kind in _ERROR_KINDS else 'infrastructure'}
    return {'reply': reply, 'error': error, 'diagnostics': client.diagnostics,
            'started_offset_sec': started_offset}


def supervised_chat(client, messages, *, timeout, thinking, max_tokens, started,
                    record, transport, supervise, temperature=0, diag=None, refused=None):
    """One supervised request. It writes `record`, `transport` and `diag` (this request's own diagnostics sink;
    None: the client's diagnostics) and reports a refused throttled 429 to `refused` (None: uncounted here)."""
    from .house import HouseError, _safe_usage, _PROVIDER
    diag = client.diagnostics if diag is None else diag
    budget = thinking if type(thinking) is int and 0 < thinking < 4000 else None
    payload = {'config': {'endpoint': client.config.endpoint, 'model': client.config.model,
                           'token': client.config.token},
               'backend': client.backend_name, 'think_field': client.think_field,
               'throttle': client.throttle, 'messages': messages,
               'thinking': ('default' if thinking == 'default' else
                            True if thinking is True else budget if budget is not None else False),
               'max_tokens': max_tokens, 'temperature': temperature, 'timeout': timeout, 'started': started,
               'deadline': started + timeout}
    try:
        outcome = supervise('t1_phase1.house', 'house_request_worker', payload,
                            deadline=started + timeout, cleanup_timeout=0.5)
        progress = outcome.progress
        transport.update({key: value for key, value in progress.items()
                          if key in ('phase', 'body_bytes', 'body_chunks', 'first_byte_sec',
                                     'first_non_whitespace_sec')})
        if 'headers_elapsed_sec' in progress:
            transport['headers_sec'] = progress['headers_elapsed_sec']
        if 'http_status' in progress:
            record['http_status'] = diag['http_status'] = progress['http_status']
            diag['response_received'] = True
        transport.update(supervisor_status=outcome.status,
                         supervisor_cleanup=outcome.cleanup['status'],
                         supervisor_request_sec=outcome.request_elapsed_sec,
                         supervisor_cleanup_sec=outcome.cleanup['elapsed_sec'])
        if outcome.reason is not None:
            transport['supervisor_reason'] = outcome.reason
        if outcome.status != 'result':
            is_deadline = outcome.status == 'deadline'
            transport.update(failure_reason='deadline' if is_deadline else 'other',
                             exception_class='deadline' if is_deadline else 'other')
            raise HouseError('House transport failure; admission unknown', code='transport_failure')
        envelope = outcome.result
        if not isinstance(envelope, dict) or set(envelope) != {'reply', 'error', 'diagnostics', 'started_offset_sec'}:
            raise ValueError('invalid worker receipt')
        child = envelope['diagnostics']
        if (not isinstance(child, dict) or not isinstance(child.get('requests'), list) or
                not isinstance(child.get('transport_requests'), list) or
                len(child['requests']) != 1 or len(child['transport_requests']) != 1):
            raise ValueError('invalid worker request count')
        child_record = child['requests'][0]
        if not isinstance(child_record, dict):
            raise ValueError('invalid worker request record')
        offset = envelope['started_offset_sec']
        if type(offset) not in (int, float) or not math.isfinite(offset) or not 0 <= offset <= timeout:
            raise ValueError('invalid worker clock offset')
        detail = _local_detail(child['transport_requests'][0])
        for key in ('headers_sec', 'first_byte_sec', 'first_non_whitespace_sec'):
            if key in detail:
                detail[key] = round(detail[key] + offset, 3)
        transport.update(detail)
        for key in ('http_status', 'retry_after_sec', 'reasoning_budget', 'max_tokens',
                    'prompt_tokens', 'completion_tokens', 'content_chars', 'reasoning_chars'):
            value = child_record.get(key)
            if type(value) is int and 0 <= value <= 2**63 - 1:
                record[key] = value
        if isinstance(child_record.get('provider'), str) and _PROVIDER.fullmatch(child_record['provider']):
            record['provider'] = child_record['provider']
        finish = child_record.get('finish_reason')
        if finish in ('stop', 'length', 'content_filter', 'tool_calls', 'function_call', 'unknown'):
            diag['finish_reason'] = record['finish_reason'] = finish
        if child.get('response_received') is True:
            diag['response_received'] = True
        for key in ('http_status', 'retry_after_sec', 'response_bytes'):
            if type(child.get(key)) is int and 0 <= child[key] <= 2**63 - 1:
                diag[key] = child[key]
        if type(child.get('reasoning_chars')) is int and child['reasoning_chars'] >= 0:
            diag['reasoning_chars'] = diag.get('reasoning_chars', 0) + child['reasoning_chars']
        for key, count in _safe_usage(child.get('usage')).items():
            diag['usage'][key] = diag['usage'].get(key, 0) + count
        if outcome.cleanup['status'] != 'ok':
            transport.update(failure_reason='other', exception_class='other',
                             supervisor_reason='cleanup_failed')
            raise HouseError('House transport cleanup failed', code='transport_failure')
        error = envelope['error']
        if error is not None:
            if (not isinstance(error, dict) or error.get('code') not in _ERROR_CODES or
                    error.get('kind') not in _ERROR_KINDS):
                raise ValueError('invalid worker error')
            raise HouseError('House request failed in worker', code=error['code'], kind=error['kind'])
        reply = envelope['reply']
        if not isinstance(reply, dict) or set(reply) != {'content', 'finish_reason', 'finish_reason_raw', 'usage', 'reasoning'}:
            raise ValueError('invalid worker reply')
        return reply
    except HouseError:
        raise
    except Exception:
        transport.update(failure_reason='other', exception_class='other')
        raise HouseError('House transport failure; admission unknown', code='transport_failure') from None
    finally:
        if client.throttle and record.get('http_status') == 429:
            if refused is not None:
                refused()
            else:
                client.attempts -= 1
                client.uncounted += 1
