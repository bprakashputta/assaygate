"""Demo system under test: a Banking77 intent classifier behind FastAPI.

It is deliberately an ordinary LLM app: one structured-output call to Claude Haiku per
request, the label constrained to the 77 intents by a JSON-schema enum and re-checked in code.
``PROMPT_VERSION`` selects the prompt:

* ``v1`` (baseline): label list with two examples each + disambiguation notes; thinking off.
* ``v2`` (the "simplified" change under review): examples and notes removed to cut tokens,
  adaptive thinking turned on "to make up for it". assaygate is what decides if that is OK.

Run:  PROMPT_VERSION=v1 uv run uvicorn examples.banking77.app:app --port 8200
"""

from __future__ import annotations

import json
import logging
import os
import time
from functools import cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, create_model

from assaygate.llm import LLMClient, LLMSettings, RateLimiter, SpendLedger

HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get("BANKING77_DATA", HERE.parents[1] / "data" / "banking77"))
log = logging.getLogger("banking77")
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"), format="%(message)s")


@cache
def labels() -> list[str]:
    return json.loads((DATA / "labels.json").read_text())


@cache
def system_prompt(version: str) -> str:
    text = (HERE / "prompts" / f"{version}.md").read_text()
    text = "\n".join(ln for ln in text.splitlines() if not ln.startswith("<!--")).strip()
    if "{{examples}}" in text:
        ex = json.loads((DATA / "examples.json").read_text())
        block = "\n".join(f"- {lab}: " + " | ".join(ex[lab]) for lab in labels())
        text = text.replace("{{examples}}", block)
    return text.replace("{{labels}}", "\n".join(f"- {lab}" for lab in labels()))


@cache
def output_model(version: str) -> type[BaseModel]:
    from typing import Literal

    label_type = Literal[tuple(labels())]  # type: ignore[valid-type]
    return create_model(f"Intent_{version}", label=(label_type, ...))


SETTINGS = {
    "v1": LLMSettings(model="claude-haiku-5-5", max_tokens=64, thinking="disabled"),
    "v2": LLMSettings(model="claude-haiku-5-5", max_tokens=2048, thinking="adaptive", effort="low"),
}


class ClassifyRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)


class ClassifyResponse(BaseModel):
    label: str
    prompt_version: str
    cost_usd: float
    latency_ms: float
    usage: dict[str, int]


def create_app() -> FastAPI:
    version = os.environ.get("PROMPT_VERSION", "v1")
    if version not in SETTINGS:
        raise RuntimeError(f"unknown PROMPT_VERSION {version!r}")
    llm = LLMClient(
        ledger=SpendLedger(
            os.environ.get("ASSAYGATE_LEDGER", "data/spend.jsonl"),
            float(os.environ.get("ASSAYGATE_SPEND_CAP_USD", "5.0")),
        ),
        cache=None,  # a classifier in production does not replay answers; latency stays real
        limiter=RateLimiter(float(os.environ.get("APP_RPM", "240"))),
        max_in_flight=int(os.environ.get("APP_MAX_IN_FLIGHT", "3")),
    )
    app = FastAPI(title="banking77-classifier", version="1.0")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "prompt_version": version, "labels": len(labels())}

    @app.post("/classify", response_model=ClassifyResponse)
    async def classify(req: ClassifyRequest) -> ClassifyResponse:
        t0 = time.perf_counter()
        system = [
            {
                "type": "text",
                "text": system_prompt(version),
                "cache_control": {"type": "ephemeral"},
            }
        ]
        try:
            res = await llm.complete(
                [{"role": "user", "content": f"Customer message:\n{req.text}"}],
                system=system,
                output_model=output_model(version),
                settings=SETTINGS[version],
                tag=f"banking77-{version}",
            )
        except Exception as exc:  # noqa: BLE001
            log.warning(json.dumps({"event": "classify_error", "error": str(exc)[:200]}))
            raise HTTPException(502, "model call failed") from exc
        label = res.parsed.label
        if label not in labels():  # schema enum should guarantee this; verify anyway
            raise HTTPException(502, "model returned an unknown label")
        latency = (time.perf_counter() - t0) * 1000
        log.info(
            json.dumps(
                {
                    "event": "classify",
                    "version": version,
                    "label": label,
                    "cost_usd": round(res.cost_usd, 7),
                    "latency_ms": round(latency, 1),
                }
            )
        )
        return ClassifyResponse(
            label=label,
            prompt_version=version,
            cost_usd=res.cost_usd,
            latency_ms=latency,
            usage={k: int(v) for k, v in res.usage.__dict__.items()},
        )

    return app


app = create_app() if os.environ.get("BANKING77_NO_APP") != "1" else None
