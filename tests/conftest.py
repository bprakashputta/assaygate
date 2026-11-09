"""Shared fixtures: a fake Anthropic SDK so every test runs offline and free."""

from __future__ import annotations

import json
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest

from assaygate.llm import LLMClient, SpendLedger


class FakeMessages:
    def __init__(self, responder: Callable[[dict[str, Any]], Any]):
        self.responder = responder
        self.requests: list[dict[str, Any]] = []

    async def create(self, **request: Any) -> Any:
        self.requests.append(request)
        out = self.responder(request)
        if isinstance(out, BaseException):
            raise out
        if isinstance(out, SimpleNamespace):
            return out
        text = out if isinstance(out, str) else json.dumps(out)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=text)],
            usage=SimpleNamespace(
                input_tokens=1000,
                output_tokens=100,
                cache_creation_input_tokens=0,
                cache_read_input_tokens=0,
            ),
            stop_reason="end_turn",
        )


class FakeSDK:
    def __init__(self, responder: Callable[[dict[str, Any]], Any]):
        self.messages = FakeMessages(responder)


@pytest.fixture
def make_llm(tmp_path):
    def factory(responder, cap: float = 5.0, cache=None, **kw) -> tuple[LLMClient, FakeSDK]:
        sdk = FakeSDK(responder)
        client = LLMClient(
            ledger=SpendLedger(tmp_path / "spend.jsonl", cap),
            cache=cache,
            client=sdk,
            **kw,
        )
        return client, sdk

    return factory


def user_text(request: dict[str, Any]) -> str:
    return request["messages"][-1]["content"]
