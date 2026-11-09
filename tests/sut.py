"""Tiny systems under test used by the runner/CLI tests."""

from __future__ import annotations

import asyncio

ANSWERS = {
    "capital of France": "Paris",
    "2+2": "4",
    "color of sky": "blue",
    "opposite of hot": "cold",
}


def good(question: str) -> dict:
    return {"output": ANSWERS.get(question, "?"), "cost_usd": 0.0001}


def bad(question: str) -> dict:
    return {"output": "Paris" if question == "capital of France" else "no idea", "cost_usd": 0.0003}


async def slow_async(question: str) -> str:
    await asyncio.sleep(0)
    return ANSWERS.get(question, "?")


def broken(question: str) -> str:
    raise RuntimeError("upstream down")
