from __future__ import annotations

import asyncio
import json
from typing import Literal

import anthropic
import httpx
import pytest
from pydantic import BaseModel

from assaygate.llm import (
    LLMSettings,
    ModelRefusal,
    ResponseCache,
    SpendCapExceeded,
    SpendLedger,
    StructuredOutputError,
    Usage,
    _strict_schema,
    cost_usd,
)


class Out(BaseModel):
    winner: Literal["A", "B", "tie"]


def test_cost_uses_skill_prices():
    u = Usage(input_tokens=50_000, output_tokens=50_000)
    assert cost_usd("claude-haiku-5-5", u) == pytest.approx(0.005 + 0.025)
    assert cost_usd("claude-sonnet-5-5", u) == pytest.approx(0.1 + 0.5)
    cached = Usage(cache_read_input_tokens=50_000, cache_creation_input_tokens=50_000)
    assert cost_usd("claude-haiku-5-5", cached) == pytest.approx(0.0005 + 0.00625)
    # long-prompt rate card for Haiku 5.5 (> 100K prompt tokens)
    assert cost_usd("claude-haiku-5-5", Usage(input_tokens=200_000)) == pytest.approx(0.10)
    with pytest.raises(KeyError):
        cost_usd("gpt-4", u)


def test_strict_schema_closes_objects_and_keeps_a_property_named_title():
    class M(BaseModel):
        title: str
        n: int = 3

    s = _strict_schema(M)
    assert s["additionalProperties"] is False
    assert set(s["required"]) == {"title", "n"}
    assert "title" in s["properties"]


async def test_structured_output_parsed_and_cached(make_llm, tmp_path):
    cache = ResponseCache(tmp_path / "c.sqlite")
    llm, sdk = make_llm(lambda r: {"winner": "B"}, cache=cache)
    msgs = [{"role": "user", "content": "hi"}]
    r1 = await llm.complete(msgs, output_model=Out)
    r2 = await llm.complete(msgs, output_model=Out)
    assert r1.parsed.winner == "B" and not r1.cached
    assert r2.cached and r2.cost_usd == 0 and r2.nominal_cost_usd == pytest.approx(r1.cost_usd)
    assert len(sdk.messages.requests) == 1
    req = sdk.messages.requests[0]
    assert req["output_config"]["format"]["type"] == "json_schema"
    assert req["thinking"] == {"type": "disabled"}
    assert "temperature" not in req  # rejected by current models


async def test_invalid_json_is_reasked_once_then_raises(make_llm):
    answers = iter(["not json", json.dumps({"winner": "A"})])
    llm, sdk = make_llm(lambda r: next(answers))
    res = await llm.complete([{"role": "user", "content": "x"}], output_model=Out)
    assert res.parsed.winner == "A" and res.attempts == 2

    llm2, _ = make_llm(lambda r: "still not json")
    with pytest.raises(StructuredOutputError):
        await llm2.complete([{"role": "user", "content": "x"}], output_model=Out)


async def test_refusal_raises(make_llm):
    from types import SimpleNamespace

    resp = SimpleNamespace(
        content=[],
        usage=SimpleNamespace(input_tokens=10, output_tokens=0),
        stop_reason="refusal",
    )
    llm, _ = make_llm(lambda r: resp)
    with pytest.raises(ModelRefusal):
        await llm.complete([{"role": "user", "content": "x"}])


async def test_spend_cap_refuses_before_calling(make_llm):
    llm, sdk = make_llm(lambda r: {"winner": "A"}, cap=0.0001)
    with pytest.raises(SpendCapExceeded):
        await llm.complete([{"role": "user", "content": "x"}], output_model=Out)
    assert sdk.messages.requests == []


async def test_reservations_prevent_concurrent_overshoot(make_llm):
    # Each call reserves worst case ~ $0.0005 (max_tokens 1000 * $0.5/M) and really costs
    # $0.00015. Cap $0.0016 -> concurrency must never push settled spend over the cap.
    async def slow(_):
        return {"winner": "A"}

    llm, sdk = make_llm(lambda r: {"winner": "A"}, cap=0.0016)
    settings = LLMSettings(max_tokens=1000)
    results = await asyncio.gather(
        *(
            llm.complete(
                [{"role": "user", "content": f"q{i}"}], output_model=Out, settings=settings
            )
            for i in range(20)
        ),
        return_exceptions=True,
    )
    ok = [r for r in results if not isinstance(r, BaseException)]
    refused = [r for r in results if isinstance(r, SpendCapExceeded)]
    assert ok and refused
    assert llm.ledger.spent_usd <= 0.0016
    assert len(ok) + len(refused) == 20


async def test_ledger_persists_across_instances(tmp_path, make_llm):
    llm, _ = make_llm(lambda r: {"winner": "A"})
    await llm.complete([{"role": "user", "content": "x"}], output_model=Out)
    again = SpendLedger(tmp_path / "spend.jsonl", 5.0)
    assert again.spent_usd == pytest.approx(llm.ledger.spent_usd) and again.spent_usd > 0


async def test_rate_limit_backoff_then_success(make_llm, monkeypatch):
    calls = {"n": 0}

    def responder(_):
        calls["n"] += 1
        if calls["n"] <= 2:
            resp = httpx.Response(
                429, headers={"retry-after": "0"}, request=httpx.Request("POST", "http://x")
            )
            return anthropic.RateLimitError("slow down", response=resp, body=None)
        return {"winner": "tie"}

    async def no_sleep(_):
        return None

    monkeypatch.setattr("assaygate.llm.asyncio.sleep", no_sleep)
    llm, _ = make_llm(responder)
    res = await llm.complete([{"role": "user", "content": "x"}], output_model=Out)
    assert res.parsed.winner == "tie" and calls["n"] == 3


async def test_thinking_off_maps_per_model(make_llm):
    llm, sdk = make_llm(lambda r: {"winner": "A"})
    await llm.complete(
        [{"role": "user", "content": "x"}],
        output_model=Out,
        settings=LLMSettings(model="claude-sonnet-5-5"),
    )
    assert sdk.messages.requests[-1]["thinking"] == {"type": "between_tools"}
