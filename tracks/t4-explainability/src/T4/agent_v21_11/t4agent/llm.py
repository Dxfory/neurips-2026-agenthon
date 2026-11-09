"""House-route client: OpenAI-compatible chat completions under a hard request allowance.

Contract (Agenthon2026-public docs/HOUSE-MODEL.md): POST $MODEL_ENDPOINT/v1/chat/completions with
`Authorization: Bearer $MODEL_TOKEN` and `model=$MODEL_NAME`; proxies come from the environment.
25 admitted requests per unit, <= 4,000 output tokens per request including thinking. Every
attempt is charged except 401/403 (refused before admission), so retries are budgeted here and
nowhere else (no SDK retries).
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass

import requests

from . import config

log = logging.getLogger("t4agent.llm")


class BudgetExhausted(RuntimeError):
    pass


class LLMUnavailable(RuntimeError):
    pass


@dataclass
class Reply:
    content: str
    reasoning: str
    finish: str
    completion_tokens: int
    seconds: float


def margin(client, seconds: float) -> float:
    """v21.5: an absolute time margin before the LLM deadline, shrunk with the unit's clock (config.derive_clock): the
    margins were set for a 400 s deadline and are unchanged there."""
    return seconds * getattr(client, "clock_scale", 1.0)


def _text(x) -> str:
    """v21.5: a reply field as text -- a string, or the text parts of a list of content parts; '' otherwise."""
    if isinstance(x, str):
        return x
    if isinstance(x, list):
        return "".join(p if isinstance(p, str) else str(p.get("text") or "") if isinstance(p, dict) else "" for p in x)
    return ""


def _completion_tokens(j) -> int:
    """v21.5: the reply's completion tokens, 0 when the usage block is missing or oddly shaped (it only feeds the
    speed estimate: a usable reply is never discarded over it)."""
    try:
        return max(0, int((j.get("usage") or {}).get("completion_tokens") or 0))
    except (AttributeError, TypeError, ValueError):
        return 0


class HouseClient:
    def __init__(self, deadline: float, budget: int = config.REQUEST_BUDGET, clock_scale: float = 1.0):
        self.endpoint = (os.environ.get("MODEL_ENDPOINT") or "").strip()
        self.model = (os.environ.get("MODEL_NAME") or "").strip()
        self.token = (os.environ.get("MODEL_TOKEN") or "").strip()
        self.enabled = bool(self.endpoint and self.model and self.token)
        self.deadline = deadline
        self.clock_scale = clock_scale  # v21.5: scales the model path's absolute margins (see margin)
        self.budget = budget
        self.used = 0
        self.ok = 0
        self.json_mode_supported = True
        self._lock = threading.Lock()
        self._closed = False  # v21.13: set by close() when the main thread stops waiting for the model stage
        self._stamps: list[float] = []
        self._session = requests.Session()  # trust_env: honours HTTP(S)_PROXY / NO_PROXY
        self._auth_failed = False
        self._cool_until = 0.0  # shared pause after a rate limit or overload: every thread waits it out
        self._streak = 0
        self._speeds: list[float] = []  # output tokens per second of successful calls
        self._seconds: list[float] = []  # v21.5: wall seconds of successful calls
        try:
            extra = json.loads(os.environ.get("T4_EXTRA_BODY") or "{}")
            self.extra_body = extra if isinstance(extra, dict) else {}
        except json.JSONDecodeError:
            self.extra_body = {}
        # Local testing only, never set in the image: a route that caps thinking with OpenRouter's
        # reasoning.max_tokens instead of the NIM-style reasoning_budget (OpenRouter's free Nemotron endpoint).
        self.reasoning_field = os.environ.get("T4_REASONING_FIELD", "").strip().lower()

    # ------------------------------------------------------------------ accounting
    def close(self) -> None:
        """v21.13: no new request after this (the model stage was abandoned at its guard)."""
        self._closed = True

    @property
    def remaining(self) -> int:
        return max(0, self.budget - self.used)

    def _reserve(self) -> None:
        with self._lock:
            if self.used >= self.budget:
                raise BudgetExhausted(f"request budget {self.budget} spent")
            self.used += 1

    def _refund(self) -> None:
        with self._lock:
            self.used = max(0, self.used - 1)

    def _throttle(self) -> None:
        rpm = config.MAX_RPM
        if rpm <= 0:
            return
        while True:
            with self._lock:
                now = time.time()
                self._stamps = [t for t in self._stamps if now - t < 60]
                if len(self._stamps) < rpm:
                    self._stamps.append(now)
                    return
                wait = 60 - (now - self._stamps[0]) + 0.05
            time.sleep(max(0.05, wait))

    def _cooldown(self) -> None:
        while True:
            with self._lock:
                wait = min(self._cool_until, self.deadline - 25) - time.time()
            if wait <= 0:
                return
            time.sleep(min(wait, 5.0))

    def _throttled(self) -> None:
        """A 429/5xx: pause every thread, longer with each consecutive one, never past the deadline margin."""
        with self._lock:
            self._streak += 1
            pause = min(60.0, 5.0 * 2 ** (self._streak - 1), max(0.0, self.deadline - time.time() - 25))
            self._cool_until = max(self._cool_until, time.time() + pause)

    @property
    def cool_until(self) -> float:
        return self._cool_until

    @property
    def speed(self) -> float | None:
        """Median output tokens per second over the recent successful calls (latency included), or None."""
        with self._lock:
            recent = sorted(self._speeds[-5:])
        return recent[len(recent) // 2] if recent else None

    @property
    def typical_seconds(self) -> float | None:
        """v21.5: median wall seconds of the recent successful calls (the route's round trip), or None before any."""
        with self._lock:
            recent = sorted(self._seconds[-5:])
        return recent[len(recent) // 2] if recent else None

    # ------------------------------------------------------------------ call
    def chat(self, messages: list[dict], *, reasoning_budget: int | None, json_mode: bool = True,
             temperature: float = config.TEMPERATURE, seed: int | None = None, thinking: bool = True,
             max_attempts: int = 2, tag: str = "") -> Reply:
        if not self.enabled:
            raise LLMUnavailable("MODEL_ENDPOINT / MODEL_NAME / MODEL_TOKEN not set")
        if self._auth_failed:
            raise LLMUnavailable("route refused authentication earlier")
        if self._closed or time.time() >= self.deadline:
            raise LLMUnavailable("LLM deadline reached")
        url = self.endpoint.rstrip("/") + "/v1/chat/completions"
        last = ""
        for attempt in range(max_attempts):
            remaining_time = self.deadline - time.time()
            if remaining_time < 20:
                raise LLMUnavailable("no time left for another request")
            body: dict = {
                "model": self.model,
                "messages": messages,
                "max_tokens": config.MAX_TOKENS,
                "temperature": temperature,
                "top_p": 0.95,
                "chat_template_kwargs": {"enable_thinking": bool(thinking)},
            }
            if seed is not None:
                body["seed"] = int(seed)
            if thinking and reasoning_budget:
                body["reasoning_budget"] = int(reasoning_budget)
            use_json = json_mode and self.json_mode_supported
            if use_json:
                body["response_format"] = {"type": "json_object"}
            if self.extra_body:  # local testing only (e.g. provider pinning); never set in the image
                body.update(self.extra_body)
                if not thinking and "reasoning" in body:
                    body["reasoning"] = {"enabled": False}  # a local route that ignores chat_template_kwargs
            if self.reasoning_field == "openrouter":  # local testing only (see __init__)
                body.pop("reasoning_budget", None)
                if not thinking:
                    body["reasoning"] = {"enabled": False}
                elif reasoning_budget:
                    body["reasoning"] = {"max_tokens": int(reasoning_budget)}
            self._cooldown()
            self._throttle()
            # the waits above can be long: the request's own timeout is measured from now, so that no call outlives
            # the deadline (it was measured from before the waits, and one reply came back 94 s after it)
            remaining_time = self.deadline - time.time()
            if remaining_time < 20:
                raise LLMUnavailable("no time left for another request")
            self._reserve()
            t0 = time.time()
            try:
                r = self._post(
                    url,
                    headers={"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"},
                    data=json.dumps(body),
                    timeout=(10, max(15.0, min(config.REQUEST_TIMEOUT_SEC, remaining_time - 5))),
                )
            except LLMUnavailable:
                raise  # the call outlived the deadline: no attempt is left either
            except Exception as exc:  # noqa: BLE001 - v21.5: any failure to get a response is a transport error
                last = f"{type(exc).__name__}"
                log.warning("llm[%s] attempt %d transport error %s", tag, attempt + 1, last)
                continue
            dt = time.time() - t0
            if r.status_code in (401, 403):
                self._refund()  # refused before admission: not charged
                self._auth_failed = True
                raise LLMUnavailable(f"route refused the request ({r.status_code})")
            if r.status_code in (400, 422) and use_json:
                # The route may not accept response_format; retry once without it.
                self.json_mode_supported = False
                last = f"{r.status_code} (json mode off)"
                log.warning("llm[%s] %s", tag, last)
                continue
            if r.status_code != 200:
                last = f"http {r.status_code}"
                log.warning("llm[%s] attempt %d -> %s", tag, attempt + 1, last)
                if r.status_code in (429, 500, 502, 503, 504):
                    self._throttled()  # rate limits and overload come in bursts: every thread waits one out
                continue
            try:
                j = r.json()
                choice = j["choices"][0]
                msg = choice.get("message") or {}
                content = _text(msg.get("content"))
                reasoning = _text(msg.get("reasoning_content")) or _text(msg.get("reasoning"))
                finish = str(choice.get("finish_reason") or "")
            except (ValueError, KeyError, IndexError, TypeError, AttributeError):  # v21.5: and odd shapes
                last = "malformed response"
                continue
            tokens = _completion_tokens(j)  # v21.5: read apart, an odd usage block never spoils a usable reply
            with self._lock:
                self.ok += 1
                self._streak = 0
                if tokens >= 200 and dt > 0:
                    self._speeds.append(tokens / dt)
                self._seconds.append(dt)
            log.info("llm[%s] ok %.1fs tokens=%s finish=%s used=%d/%d", tag, dt, tokens, finish, self.used,
                     self.budget)
            return Reply(content, reasoning, finish, tokens, dt)
        raise LLMUnavailable(f"call failed: {last}")

    def _post(self, url: str, **kw) -> requests.Response:
        """v21.5: the POST, bounded in total time. The read timeout applies per socket read, so a reply that trickles
        in can outlive it; the request runs in a daemon thread and the call gives up at the LLM deadline (v21.13; was + 15 s), so every
        call returns by then whatever the socket does. Earlier, once its own connect + read timeouts and 5 s have
        passed, the request is given up as a timeout (a transport error: chat() may try again while time is left). A
        reply arrives at once after its first byte, so only a trickling socket runs past that; a hung plan held the
        main thread until the deadline + 15 s (before v21.13) and cost the analysts. An abandoned request stays counted as spent (House
        may charge it); its thread ends with its socket or with the process. A test double or recorder that tells
        calls apart must key on the request, not on the calling thread (threading.local is not carried over)."""
        box: dict = {}
        limit = self.deadline  # v21.13: every request in flight is given up at the LLM deadline (was + 15 s)
        t = kw.get("timeout")
        if isinstance(t, tuple) and len(t) == 2 and all(isinstance(x, (int, float)) for x in t):
            limit = min(limit, time.time() + t[0] + t[1] + 5)

        def work() -> None:
            try:
                box["r"] = self._session.post(url, **kw)
            except BaseException as exc:  # noqa: BLE001 - handed to the caller below
                box["e"] = exc

        th = threading.Thread(target=work, name="house-post", daemon=True)
        th.start()
        th.join(max(0.0, limit - time.time()))
        if th.is_alive():
            if limit < self.deadline:
                log.warning("request still open after its connect + read timeouts + 5 s: abandoned")
                raise requests.Timeout("request outlived its own timeouts")
            log.warning("request still open at the LLM deadline: abandoned")
            raise LLMUnavailable("request outlived the deadline")
        if "e" in box:
            raise box["e"]
        return box["r"]


# ---------------------------------------------------------------------------------------------- #
# JSON extraction                                                                                 #
# ---------------------------------------------------------------------------------------------- #
def _balanced(text: str, start: int) -> str | None:
    opener = text[start]
    closer = "}" if opener == "{" else "]"
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c == opener:
            depth += 1
        elif c == closer:
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


_TRAILING_COMMA = re.compile(r",\s*([}\]])")
_NONFINITE = re.compile(r"(?<![\w\"])(?:NaN|-?Infinity)(?![\w\"])")
_PY_LITERAL = re.compile(r"([:\[,]\s*)(True|False|None)\b")


def _lenient(text: str) -> str:
    """Common slips of a model writing JSON, fixed: trailing commas, NaN / Infinity, Python's True / False / None."""
    text = _TRAILING_COMMA.sub(r"\1", text)
    text = _NONFINITE.sub("null", text)
    return _PY_LITERAL.sub(lambda m: m.group(1) + {"True": "true", "False": "false", "None": "null"}[m.group(2)], text)


def _last_json(text: str, key: str):
    """The last parseable JSON object in `text` that contains `key` (a reply's final draft), or None."""
    for k in [m.start() for m in re.finditer(re.escape(f'"{key}"'), text)][::-1][:5]:
        for s in [m.start() for m in re.finditer(r"\{", text[:k])][::-1][:20]:
            blob = _balanced(text, s)
            if not blob:
                continue
            for cand in (blob, _lenient(blob)):
                try:
                    obj = json.loads(cand)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict) and key in obj:
                    return obj
    return None


def salvage_array(text: str, key: str) -> dict | None:
    """v21.13: the complete objects of a JSON array `key` cut off by the output cap ({"key": [{..}, {..}, {.. -> the
    first two), or None when not one is complete. The CPI reasons reply (2026-10-20) ended at 4,000 tokens mid-array
    and the unit went out on code reasons."""
    if not text:
        return None
    starts = [m.end() for m in re.finditer(r'"%s"\s*:\s*\[' % re.escape(key), text)]
    for i in reversed(starts):  # the last array with a complete object: thinking may draft one before the answer
        items: list = []
        while i < len(text):
            while i < len(text) and text[i] in " \t\r\n,":
                i += 1
            if i >= len(text) or text[i] != "{":
                break
            blob = _balanced(text, i)
            if blob is None:
                break
            try:
                items.append(json.loads(_lenient(blob)))
            except json.JSONDecodeError:
                break
            i += len(blob)
        if items:
            return {key: items}
    return None


def parse_reply(rep: "Reply", key: str):
    """v21: a reply's JSON without spending a request when it can be recovered -- the content as given, the content with
    common slips fixed, then the last object carrying `key` in the reasoning. v20's sweep spent 16 of its 194 requests
    on JSON repair calls. Raises ValueError when nothing is recoverable (the caller's repair path takes over)."""
    try:
        return extract_json(rep.content, key=key)
    except ValueError:
        pass
    obj = _last_json(_lenient(rep.content), key) if rep.content else None
    if obj is None and rep.reasoning:
        obj = _last_json(rep.reasoning, key)
    if obj is None:
        raise ValueError("no JSON object found")
    log.info("reply JSON for %r recovered without a repair call", key)
    return obj


def extract_json(text: str, key: str | None = None):
    """The first parseable JSON object in `text` (preferring one that contains `key`)."""
    if not text:
        raise ValueError("empty reply")
    t = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", t, re.S)
    candidates = [fence.group(1)] if fence else []
    candidates.append(t)
    for cand in candidates:
        starts = [m.start() for m in re.finditer(r"[{\[]", cand)]
        if key:
            k = cand.find(f'"{key}"')
            if k >= 0:
                starts = [s for s in starts if s < k][-1:] + starts
        for s in starts[:50]:
            blob = _balanced(cand, s)
            if not blob:
                continue
            try:
                obj = json.loads(blob)
            except json.JSONDecodeError:
                continue
            if key is None or (isinstance(obj, dict) and key in obj):
                return obj
    raise ValueError("no JSON object found")
