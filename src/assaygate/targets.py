"""Systems under test: any Python callable, or an HTTP endpoint.

A target receives an :class:`~assaygate.data.Item` input and returns a :class:`TargetResult`.
A callable may return a plain value (the output) or a dict with ``output`` and optional
``cost_usd`` / ``metadata`` keys, so an app that knows its own model spend can report it.
"""

from __future__ import annotations

import asyncio
import importlib
import inspect
import re
from collections.abc import Callable
from typing import Any

import httpx
from pydantic import BaseModel, Field

from .config import CallableTarget, HttpTarget
from .data import Item


class TargetResult(BaseModel):
    output: Any = None
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


def import_ref(ref: str) -> Callable[..., Any]:
    if ":" not in ref:
        raise ValueError(f"expected 'module:attr', got {ref!r}")
    mod_name, attr = ref.split(":", 1)
    obj: Any = importlib.import_module(mod_name)
    for part in attr.split("."):
        obj = getattr(obj, part)
    if not callable(obj):
        raise TypeError(f"{ref} is not callable")
    return obj


def _normalise(value: Any) -> TargetResult:
    if isinstance(value, TargetResult):
        return value
    if isinstance(value, dict) and "output" in value:
        return TargetResult(
            output=value["output"],
            cost_usd=float(value.get("cost_usd") or 0.0),
            metadata=dict(value.get("metadata") or {}),
        )
    return TargetResult(output=value)


class Target:
    async def __call__(self, item: Item) -> TargetResult:  # pragma: no cover - interface
        raise NotImplementedError

    async def aclose(self) -> None:
        return None


class FunctionTarget(Target):
    def __init__(self, fn: Callable[..., Any]):
        self.fn = fn
        self.is_async = inspect.iscoroutinefunction(fn)

    async def __call__(self, item: Item) -> TargetResult:
        if self.is_async:
            value = await self.fn(item.input)
        else:
            value = await asyncio.to_thread(self.fn, item.input)
        return _normalise(value)


_PLACEHOLDER = re.compile(r"\{\{\s*([\w.]+)\s*\}\}")


def render_template(value: Any, item: Item) -> Any:
    """Substitute ``{{input}}`` / ``{{field}}`` in strings, recursively through dicts/lists.

    A string that is *exactly* one placeholder keeps the field's type (so a dict input stays a
    dict in the JSON body); a placeholder inside a longer string is stringified.
    """
    if isinstance(value, dict):
        return {k: render_template(v, item) for k, v in value.items()}
    if isinstance(value, list):
        return [render_template(v, item) for v in value]
    if not isinstance(value, str):
        return value

    def lookup(name: str) -> Any:
        if name == "input":
            return item.input
        if name == "id":
            return item.id
        return dig(item.fields, name)

    whole = _PLACEHOLDER.fullmatch(value)
    if whole:
        return lookup(whole.group(1))
    return _PLACEHOLDER.sub(lambda m: str(lookup(m.group(1))), value)


def dig(obj: Any, path: str) -> Any:
    for part in path.split("."):
        if isinstance(obj, list):
            obj = obj[int(part)]
        elif isinstance(obj, dict):
            obj = obj[part]
        else:
            raise KeyError(path)
    return obj


class HTTPTarget(Target):
    def __init__(self, cfg: HttpTarget, client: httpx.AsyncClient | None = None):
        self.cfg = cfg
        self._client = client or httpx.AsyncClient(timeout=cfg.timeout_s)

    async def __call__(self, item: Item) -> TargetResult:
        body = render_template(self.cfg.body, item)
        if self.cfg.method == "GET":
            resp = await self._client.get(self.cfg.url, params=body, headers=self.cfg.headers)
        else:
            resp = await self._client.post(self.cfg.url, json=body, headers=self.cfg.headers)
        resp.raise_for_status()
        data = resp.json()
        cost = float(dig(data, self.cfg.cost_path)) if self.cfg.cost_path else 0.0
        return TargetResult(output=dig(data, self.cfg.output_path), cost_usd=cost)

    async def aclose(self) -> None:
        await self._client.aclose()


def build_target(cfg: CallableTarget | HttpTarget | Callable[..., Any]) -> Target:
    if isinstance(cfg, HttpTarget):
        return HTTPTarget(cfg)
    if isinstance(cfg, CallableTarget):
        return FunctionTarget(import_ref(cfg.ref))
    if callable(cfg):
        return FunctionTarget(cfg)
    raise TypeError(f"cannot build a target from {cfg!r}")
