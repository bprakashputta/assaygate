"""Model access: one client wrapper that every judge (and, optionally, a system under test) uses.

What it adds on top of the Anthropic SDK:

* **Structured outputs** via ``output_config.format`` (JSON schema) plus code-side pydantic
  validation, with one re-ask if the model's JSON fails validation.
* **A disk cache** (sqlite) keyed by the canonical request, so re-running a suite or a
  calibration costs nothing for identical calls.
* **A hard spend cap** enforced by a ledger with *reservations*: before a call, the worst-case
  cost (prompt estimate + ``max_tokens`` of output) is reserved; afterwards it is settled to
  the real cost. Concurrent calls therefore cannot overshoot the cap.
* **Rate limiting** (requests per minute, token bucket) and retries/timeouts (the SDK retries
  429/5xx/connection errors with backoff; we set the budget for that).
* **Per-call telemetry**: tokens, cache tokens, $ cost, latency, cache hit, logged as JSON.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

log = logging.getLogger("assaygate.llm")

# ----------------------------------------------------------------------------- pricing

# USD per million tokens (Claude API first-party rates, from the claude-api skill, 2026-10).
# cache_write is the 5-minute write rate (1.25x input); cache_read is 0.1x input.
PRICES: dict[str, dict[str, float]] = {
    "claude-haiku-5-5": {"input": 0.10, "output": 0.50},
    "claude-sonnet-5-5": {"input": 2.00, "output": 10.00},
    "claude-opus-5-5": {"input": 4.00, "output": 20.00},
}
# Haiku 5.5 has a second rate card for prompts over 100K tokens.
LONG_PROMPT_PRICES: dict[str, tuple[int, dict[str, float]]] = {
    "claude-haiku-5-5": (100_000, {"input": 0.50, "output": 2.50}),
}


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

    @property
    def prompt_tokens(self) -> int:
        return self.input_tokens + self.cache_creation_input_tokens + self.cache_read_input_tokens


def cost_usd(model: str, usage: Usage) -> float:
    if model not in PRICES:
        raise KeyError(f"no price for model {model!r}; add it to assaygate.llm.PRICES")
    rate = PRICES[model]
    if model in LONG_PROMPT_PRICES:
        threshold, long_rate = LONG_PROMPT_PRICES[model]
        if usage.prompt_tokens > threshold:
            rate = long_rate
    per = 1e-6
    return (
        usage.input_tokens * rate["input"] * per
        + usage.cache_creation_input_tokens * rate["input"] * 1.25 * per
        + usage.cache_read_input_tokens * rate["input"] * 0.10 * per
        + usage.output_tokens * rate["output"] * per
    )


# Models that reject {"type": "disabled"}; "between_tools" is their thinking-off mode.
THINKING_OFF_AS_BETWEEN_TOOLS = {"claude-sonnet-5-5"}


def estimate_tokens(text: str) -> int:
    """Conservative prompt-size estimate used only for spend reservations (~3 chars/token)."""
    return len(text) // 3 + 16


# ----------------------------------------------------------------------------- spend ledger


class SpendCapExceeded(RuntimeError):
    """Raised before a call that could take total spend past the configured cap."""


class SpendLedger:
    """Append-only JSONL ledger of real spend, shared by every run that points at the file.

    ``reserve`` holds the worst-case cost of an in-flight call; ``settle`` replaces it with the
    actual cost and appends a line. The cap check counts settled spend + all reservations.
    """

    def __init__(self, path: Path | str | None, cap_usd: float):
        self.path = Path(path) if path else None
        self.cap_usd = cap_usd
        self._settled = 0.0
        self._reserved = 0.0
        self._lock = asyncio.Lock()
        if self.path and self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    self._settled += float(json.loads(line)["cost_usd"])

    @property
    def spent_usd(self) -> float:
        return self._settled

    @property
    def remaining_usd(self) -> float:
        return self.cap_usd - self._settled - self._reserved

    async def reserve(self, amount: float) -> None:
        async with self._lock:
            if self._settled + self._reserved + amount > self.cap_usd:
                raise SpendCapExceeded(
                    f"call needs up to ${amount:.4f}; spent ${self._settled:.4f} + reserved "
                    f"${self._reserved:.4f} of ${self.cap_usd:.2f} cap"
                )
            self._reserved += amount

    async def settle(self, reserved: float, actual: float, record: dict[str, Any]) -> None:
        async with self._lock:
            self._reserved = max(0.0, self._reserved - reserved)
            self._settled += actual
            if self.path:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a") as fh:
                    fh.write(json.dumps({**record, "cost_usd": actual}) + "\n")

    async def release(self, reserved: float) -> None:
        async with self._lock:
            self._reserved = max(0.0, self._reserved - reserved)


# ----------------------------------------------------------------------------- cache


class ResponseCache:
    """sqlite cache of model responses keyed by a hash of the canonical request."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS responses (key TEXT PRIMARY KEY, value TEXT NOT NULL, "
            "created REAL NOT NULL)"
        )
        self._db.commit()

    @staticmethod
    def key(request: dict[str, Any]) -> str:
        blob = json.dumps(request, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(blob.encode()).hexdigest()

    def get(self, key: str) -> dict[str, Any] | None:
        row = self._db.execute("SELECT value FROM responses WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key: str, value: dict[str, Any]) -> None:
        self._db.execute(
            "INSERT OR REPLACE INTO responses (key, value, created) VALUES (?, ?, ?)",
            (key, json.dumps(value), time.time()),
        )
        self._db.commit()

    def close(self) -> None:
        self._db.close()


# ----------------------------------------------------------------------------- rate limit


class RateLimiter:
    """Token bucket: at most ``rpm`` request starts per minute, smoothly spaced."""

    def __init__(self, rpm: float | None):
        self.interval = 60.0 / rpm if rpm else 0.0
        self._next = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        if not self.interval:
            return
        async with self._lock:
            now = time.monotonic()
            wait = self._next - now
            self._next = max(now, self._next) + self.interval
        if wait > 0:
            await asyncio.sleep(wait)


# ----------------------------------------------------------------------------- client


@dataclass
class LLMResult:
    text: str
    parsed: Any
    usage: Usage
    cost_usd: float
    latency_ms: float
    cached: bool
    model: str
    stop_reason: str | None = None
    attempts: int = 1
    # What the call cost/took when it actually ran (equal to cost_usd/latency_ms unless cached).
    nominal_cost_usd: float = 0.0
    nominal_latency_ms: float = 0.0


@dataclass
class LLMSettings:
    model: str = "claude-haiku-5-5"
    max_tokens: int = 1024
    effort: str | None = None  # low | medium | high (None = model default)
    thinking: str | None = "disabled"  # "disabled" | "adaptive" | None (model default)
    timeout_s: float = 60.0
    max_retries: int = 4
    extra: dict[str, Any] = field(default_factory=dict)


class LLMClient:
    """Async structured-output client with cache, spend cap, rate limit and telemetry."""

    def __init__(
        self,
        *,
        ledger: SpendLedger,
        cache: ResponseCache | None = None,
        limiter: RateLimiter | None = None,
        client: Any = None,
        default: LLMSettings | None = None,
        max_in_flight: int = 4,
        backoff_retries: int = 6,
    ):
        self.ledger = ledger
        self._in_flight = asyncio.Semaphore(max_in_flight)
        self.backoff_retries = backoff_retries
        self.cache = cache
        self.limiter = limiter or RateLimiter(None)
        self.default = default or LLMSettings()
        self._client = client
        self.calls = 0
        self.cache_hits = 0

    def _sdk(self, settings: LLMSettings) -> Any:
        if self._client is None:
            import anthropic

            self._client = anthropic.AsyncAnthropic(
                timeout=settings.timeout_s, max_retries=settings.max_retries
            )
        return self._client

    def _request(
        self,
        settings: LLMSettings,
        system: str | list[dict[str, Any]] | None,
        messages: list[dict[str, Any]],
        schema: dict[str, Any] | None,
    ) -> dict[str, Any]:
        req: dict[str, Any] = {
            "model": settings.model,
            "max_tokens": settings.max_tokens,
            "messages": messages,
        }
        if system:
            req["system"] = system
        output_config: dict[str, Any] = {}
        if schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": schema}
        if settings.effort:
            output_config["effort"] = settings.effort
        if output_config:
            req["output_config"] = output_config
        if settings.thinking:
            thinking = settings.thinking
            if thinking == "disabled" and settings.model in THINKING_OFF_AS_BETWEEN_TOOLS:
                thinking = "between_tools"
            req["thinking"] = {"type": thinking}
        req.update(settings.extra)
        return req

    async def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        system: str | list[dict[str, Any]] | None = None,
        output_model: type[BaseModel] | None = None,
        settings: LLMSettings | None = None,
        tag: str = "",
    ) -> LLMResult:
        s = settings or self.default
        schema = _strict_schema(output_model) if output_model else None
        request = self._request(s, system, messages, schema)
        key = ResponseCache.key(request)
        self.calls += 1

        if self.cache is not None and (hit := self.cache.get(key)) is not None:
            self.cache_hits += 1
            parsed = output_model.model_validate_json(hit["text"]) if output_model else None
            log.info(
                json.dumps({"event": "llm_call", "tag": tag, "model": s.model, "cached": True})
            )
            usage = Usage(**hit["usage"])
            return LLMResult(
                text=hit["text"],
                parsed=parsed,
                usage=usage,
                cost_usd=0.0,
                latency_ms=0.0,
                cached=True,
                model=s.model,
                stop_reason=hit.get("stop_reason"),
                nominal_cost_usd=cost_usd(s.model, usage),
                nominal_latency_ms=float(hit.get("latency_ms", 0.0)),
            )

        prompt_chars = len(json.dumps(request.get("system", ""))) + len(json.dumps(messages))
        worst = cost_usd(
            s.model, Usage(input_tokens=estimate_tokens("x" * prompt_chars), output_tokens=0)
        ) + cost_usd(s.model, Usage(output_tokens=s.max_tokens))

        attempts = 0
        last_error: Exception | None = None
        total_cost = 0.0
        total_latency = 0.0
        usage_sum = Usage()
        while attempts < 2:  # one re-ask on invalid structured output
            attempts += 1
            await self.ledger.reserve(worst)
            try:
                resp, latency = await self._create_with_backoff(s, request, tag)
            except BaseException:
                await self.ledger.release(worst)
                raise
            usage = _usage(resp)
            actual = cost_usd(s.model, usage)
            total_cost += actual
            total_latency += latency
            usage_sum = _add(usage_sum, usage)
            await self.ledger.settle(
                worst,
                actual,
                {"ts": time.time(), "model": s.model, "tag": tag, **usage.__dict__},
            )
            text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
            stop = getattr(resp, "stop_reason", None)
            log.info(
                json.dumps(
                    {
                        "event": "llm_call",
                        "tag": tag,
                        "model": s.model,
                        "cached": False,
                        "input_tokens": usage.input_tokens,
                        "output_tokens": usage.output_tokens,
                        "cache_read_tokens": usage.cache_read_input_tokens,
                        "cost_usd": round(actual, 6),
                        "latency_ms": round(latency, 1),
                        "stop_reason": stop,
                        "attempt": attempts,
                    }
                )
            )
            if stop == "refusal":
                raise ModelRefusal(f"model refused ({tag})")
            parsed = None
            if output_model is not None:
                try:
                    parsed = output_model.model_validate_json(text)
                except ValidationError as exc:
                    last_error = exc
                    if stop == "max_tokens":
                        # Truncated JSON: re-ask with room to finish instead of repeating.
                        request = {**request, "max_tokens": request["max_tokens"] * 2}
                        worst += cost_usd(s.model, Usage(output_tokens=request["max_tokens"]))
                    continue
            if self.cache is not None:
                self.cache.put(
                    key,
                    {
                        "text": text,
                        "usage": usage.__dict__,
                        "stop_reason": stop,
                        "latency_ms": latency,
                    },
                )
            return LLMResult(
                text=text,
                parsed=parsed,
                usage=usage_sum,
                cost_usd=total_cost,
                latency_ms=total_latency,
                cached=False,
                model=s.model,
                stop_reason=stop,
                attempts=attempts,
                nominal_cost_usd=total_cost,
                nominal_latency_ms=total_latency,
            )
        raise StructuredOutputError(f"invalid structured output after retry ({tag})") from (
            last_error
        )

    async def _create_with_backoff(
        self, s: LLMSettings, request: dict[str, Any], tag: str
    ) -> tuple[Any, float]:
        """SDK call under the in-flight cap; extra jittered backoff on 429/529/5xx.

        The SDK already retries a few times quickly. Organisation-wide *concurrency* limits
        (shared with other workloads on the same key) need longer waits than that, so we add
        up to ``backoff_retries`` waits of 2, 4, 8 ... 60 s (+ jitter, or Retry-After).
        """
        import random

        import anthropic

        retryable = (
            anthropic.RateLimitError,
            anthropic.InternalServerError,
            anthropic.APIConnectionError,
        )
        for attempt in range(self.backoff_retries + 1):
            await self.limiter.acquire()
            async with self._in_flight:
                t0 = time.perf_counter()
                try:
                    resp = await self._sdk(s).messages.create(**request)
                    return resp, (time.perf_counter() - t0) * 1000
                except retryable as exc:
                    if attempt == self.backoff_retries:
                        raise
                    err_name = type(exc).__name__
                    retry_after = None
                    headers = getattr(getattr(exc, "response", None), "headers", None) or {}
                    try:
                        retry_after = float(headers.get("retry-after", ""))
                    except ValueError:
                        retry_after = None
                    wait = retry_after or min(60.0, 2.0 * 2**attempt)
                    wait += random.uniform(0, wait / 2)
            log.warning(
                json.dumps(
                    {
                        "event": "llm_backoff",
                        "tag": tag,
                        "attempt": attempt + 1,
                        "wait_s": round(wait, 1),
                        "error": err_name,
                    }
                )
            )
            await asyncio.sleep(wait)
        raise AssertionError("unreachable")  # pragma: no cover


class ModelRefusal(RuntimeError):
    pass


class StructuredOutputError(RuntimeError):
    pass


def _usage(resp: Any) -> Usage:
    u = resp.usage
    return Usage(
        input_tokens=getattr(u, "input_tokens", 0) or 0,
        output_tokens=getattr(u, "output_tokens", 0) or 0,
        cache_creation_input_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0,
        cache_read_input_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
    )


def _add(a: Usage, b: Usage) -> Usage:
    return Usage(
        a.input_tokens + b.input_tokens,
        a.output_tokens + b.output_tokens,
        a.cache_creation_input_tokens + b.cache_creation_input_tokens,
        a.cache_read_input_tokens + b.cache_read_input_tokens,
    )


def _strict_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Pydantic JSON schema tightened for structured outputs (no extra keys, all required)."""
    schema = model.model_json_schema()

    def tighten(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("type") == "object" and "properties" in node:
                node["additionalProperties"] = False
                node["required"] = list(node["properties"])
            for meta in ("title", "default", "description"):
                if not isinstance(node.get(meta), dict):
                    node.pop(meta, None)
            for v in node.values():
                tighten(v)
        elif isinstance(node, list):
            for v in node:
                tighten(v)

    tighten(schema)
    return schema
