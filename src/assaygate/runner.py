"""Run a suite: call the target on every item (bounded concurrency), score, write artifacts.

Artifacts for run ``<id>`` live in ``<runs_dir>/<id>/``:

* ``records.jsonl`` — one line per item: output, per-scorer scores, latency, cost, error
* ``manifest.json`` — suite name, config hash, dataset hash, git sha, timings, totals
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import subprocess
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from .config import SuiteConfig
from .data import Item, file_sha256, load_items
from .llm import LLMClient, RateLimiter, ResponseCache, SpendLedger
from .scorers import Score, ScorerContext, build_scorer
from .targets import Target, TargetResult, build_target

log = logging.getLogger("assaygate.run")


class Record(BaseModel):
    id: str
    input: Any
    expected: Any = None
    output: Any = None
    error: str | None = None
    latency_ms: float
    cost_usd: float = Field(0.0, description="system-under-test spend for this item")
    eval_cost_usd: float = Field(0.0, description="judge spend for this item")
    scores: dict[str, Score] = Field(default_factory=dict)


class Manifest(BaseModel):
    run_id: str
    suite: str
    label: str | None = None
    created_at: str
    duration_s: float
    n_items: int
    n_errors: int
    metrics: list[str]
    config_sha256: str
    dataset_sha256: str
    git_sha: str | None = None
    totals: dict[str, float] = Field(default_factory=dict)
    gate: dict[str, Any] = Field(default_factory=dict, description="the suite's gate config")


class Run(BaseModel):
    manifest: Manifest
    records: list[Record]

    def metric_values(self, metric: str) -> dict[str, float]:
        return {r.id: r.scores[metric].value for r in self.records if metric in r.scores}


def _git_sha() -> str | None:
    if sha := os.environ.get("GITHUB_SHA"):
        return sha
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5, check=False
        )
        sha = out.stdout.strip()
        return sha if out.returncode == 0 and re.fullmatch(r"[0-9a-f]{40}", sha) else None
    except (OSError, subprocess.SubprocessError):
        return None


def make_llm(suite: SuiteConfig) -> LLMClient:
    ex = suite.execution
    return LLMClient(
        ledger=SpendLedger(suite.resolve(ex.ledger_path), ex.spend_cap_usd),
        cache=ResponseCache(suite.resolve(ex.cache_path)) if ex.cache_path else None,
        limiter=RateLimiter(ex.rpm),
    )


async def _run_item(
    item: Item,
    target: Target,
    scorers: dict[str, Any],
    sem: asyncio.Semaphore,
    limiter: RateLimiter,
    timeout_s: float,
) -> Record:
    async with sem:
        await limiter.acquire()
        t0 = time.perf_counter()
        try:
            result = await asyncio.wait_for(target(item), timeout_s)
        except Exception as exc:  # noqa: BLE001 - a failing item must not kill the run
            result = TargetResult(error=f"{type(exc).__name__}: {exc}"[:500])
        elapsed = (time.perf_counter() - t0) * 1000
        latency = result.latency_ms or elapsed
        record = Record(
            id=item.id,
            input=item.input,
            expected=item.expected,
            output=result.output,
            error=result.error,
            latency_ms=latency,
            cost_usd=result.cost_usd,
        )
        if result.error is None:
            outs = await asyncio.gather(
                *(fn(item, result.output) for fn in scorers.values()), return_exceptions=True
            )
            for name, out in zip(scorers, outs, strict=True):
                if isinstance(out, BaseException):
                    if type(out).__name__ == "SpendCapExceeded":
                        raise out
                    record.scores[name] = Score(
                        value=0.0, passed=False, detail=f"scorer error: {out}"[:300]
                    )
                else:
                    record.scores[name] = out
                    record.eval_cost_usd += out.cost_usd
        else:
            # An errored item scores 0 on every metric: errors are regressions, not omissions.
            for name in scorers:
                record.scores[name] = Score(value=0.0, passed=False, detail="target error")
        return record


async def run_suite(
    suite: SuiteConfig,
    *,
    runs_dir: Path | str = "runs",
    label: str | None = None,
    target: Target | None = None,
    llm: LLMClient | None = None,
    run_id: str | None = None,
) -> Run:
    items = load_items(
        suite.resolve(suite.dataset.path),
        id_field=suite.dataset.id_field,
        input_field=suite.dataset.input_field,
        expected_field=suite.dataset.expected_field,
        limit=suite.dataset.limit,
    )
    needs_llm = any(s.type.startswith("llm_") for s in suite.scorers)
    if llm is None and needs_llm:
        llm = make_llm(suite)
    ctx = ScorerContext(llm=llm)
    scorers = {s.metric_name: build_scorer(s.type, s.params, ctx) for s in suite.scorers}
    if len(scorers) != len(suite.scorers):
        raise ValueError("scorer names must be unique (set `name:` on duplicates)")
    tgt = target or build_target(suite.target)
    sem = asyncio.Semaphore(suite.execution.concurrency)
    limiter = RateLimiter(suite.execution.rpm)

    started = time.time()
    try:
        records = await asyncio.gather(
            *(
                _run_item(it, tgt, scorers, sem, limiter, suite.execution.target_timeout_s)
                for it in items
            )
        )
    finally:
        await tgt.aclose()
    duration = time.time() - started

    rid = run_id or f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:6]}"
    cfg_blob = suite.model_dump_json(exclude={"base_dir"}).encode()
    manifest = Manifest(
        run_id=rid,
        suite=suite.name,
        label=label,
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        duration_s=round(duration, 3),
        n_items=len(records),
        n_errors=sum(r.error is not None for r in records),
        metrics=list(scorers),
        config_sha256=hashlib.sha256(cfg_blob).hexdigest(),
        dataset_sha256=file_sha256(suite.resolve(suite.dataset.path)),
        git_sha=_git_sha(),
        totals={
            "system_cost_usd": round(sum(r.cost_usd for r in records), 6),
            "eval_cost_usd": round(sum(r.eval_cost_usd for r in records), 6),
            **{
                f"mean_{m}": round(sum(r.scores[m].value for r in records) / len(records), 6)
                for m in scorers
            },
        },
        gate=suite.gate.model_dump(),
    )
    run = Run(manifest=manifest, records=list(records))
    save_run(run, Path(runs_dir) / rid)
    log.info(json.dumps({"event": "run_done", "run_id": rid, **manifest.totals}))
    return run


def save_run(run: Run, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "records.jsonl").open("w") as fh:
        for r in run.records:
            fh.write(r.model_dump_json() + "\n")
    (directory / "manifest.json").write_text(run.manifest.model_dump_json(indent=2) + "\n")
    return directory


def load_run(directory: Path | str) -> Run:
    directory = Path(directory)
    manifest = Manifest.model_validate_json((directory / "manifest.json").read_text())
    records = [
        Record.model_validate_json(line)
        for line in (directory / "records.jsonl").read_text().splitlines()
        if line.strip()
    ]
    return Run(manifest=manifest, records=records)
