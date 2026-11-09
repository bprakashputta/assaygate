"""Suite configuration: what a YAML file (or Python code) declares.

A suite is: a dataset, a target (the system under test), a list of scorers, optional gate
thresholds, and execution settings (concurrency, rate limit, spend cap, cache).
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ----------------------------------------------------------------------------- dataset


class DatasetConfig(_Strict):
    path: str
    id_field: str = "id"
    input_field: str = "input"
    expected_field: str | None = "expected"
    limit: int | None = None


# ----------------------------------------------------------------------------- targets


class CallableTarget(_Strict):
    type: Literal["callable"] = "callable"
    ref: str = Field(description="'package.module:function'. Sync or async; gets the input.")


class HttpTarget(_Strict):
    type: Literal["http"] = "http"
    url: str
    method: Literal["POST", "GET"] = "POST"
    body: dict[str, Any] = Field(
        default_factory=lambda: {"input": "{{input}}"},
        description="JSON body; string values may contain {{input}} or {{field}} placeholders.",
    )
    headers: dict[str, str] = Field(default_factory=dict)
    output_path: str = Field("output", description="dot path to the output in the JSON reply")
    cost_path: str | None = Field(None, description="dot path to a $ cost in the reply, if any")
    timeout_s: float = 60.0


TargetConfig = Annotated[CallableTarget | HttpTarget, Field(discriminator="type")]


# ----------------------------------------------------------------------------- scorers


class ScorerConfig(BaseModel):
    """One scorer. ``type`` picks the implementation; other keys are its parameters."""

    model_config = ConfigDict(extra="allow")
    type: str
    name: str | None = None

    @property
    def params(self) -> dict[str, Any]:
        return dict(self.model_extra or {})

    @property
    def metric_name(self) -> str:
        return self.name or self.type


# ----------------------------------------------------------------------------- gate


class MetricGate(_Strict):
    metric: str
    max_drop: float = Field(
        0.0, ge=0, description="regression margin on the 0-1 metric scale (0.02 = 2 points)"
    )
    mode: Literal["regression", "non_inferiority"] = Field(
        "regression",
        description=(
            "regression: fail when the CI says the candidate is worse by more than max_drop. "
            "non_inferiority: fail unless the CI rules out a drop larger than max_drop."
        ),
    )
    min_value: float | None = Field(None, description="absolute floor on the candidate mean")


class CostGate(_Strict):
    max_increase: float | None = Field(0.10, description="fail if cost/item ratio CI > 1+this")
    max_per_item_usd: float | None = None


class LatencyGate(_Strict):
    p95_max_increase: float | None = Field(0.20, description="fail if p95 ratio CI > 1+this")
    p95_max_ms: float | None = None


class GateConfig(_Strict):
    level: float = 0.95
    n_resamples: int = 10_000
    seed: int = 0
    quality: list[MetricGate] = Field(default_factory=list)
    cost: CostGate | None = Field(default_factory=CostGate)
    latency: LatencyGate | None = Field(default_factory=LatencyGate)


# ----------------------------------------------------------------------------- execution


class ExecutionConfig(_Strict):
    concurrency: int = Field(8, ge=1)
    rpm: float | None = Field(None, description="max model/target requests per minute")
    spend_cap_usd: float = Field(
        default_factory=lambda: float(os.environ.get("ASSAYGATE_SPEND_CAP_USD", "5.0"))
    )
    ledger_path: str = Field(
        default_factory=lambda: os.environ.get("ASSAYGATE_LEDGER", ".assaygate/spend.jsonl")
    )
    cache_path: str | None = Field(
        default_factory=lambda: os.environ.get("ASSAYGATE_CACHE", ".assaygate/cache.sqlite")
    )
    target_timeout_s: float = 120.0


class SuiteConfig(_Strict):
    name: str
    description: str = ""
    dataset: DatasetConfig
    target: TargetConfig
    scorers: list[ScorerConfig]
    gate: GateConfig = Field(default_factory=GateConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    base_dir: str = Field(".", exclude=True, description="paths resolve relative to this")

    def resolve(self, path: str) -> Path:
        p = Path(path)
        return p if p.is_absolute() else Path(self.base_dir) / p


_ENV = re.compile(r"\$\{(\w+)(?::-([^}]*))?\}")


def expand_env(text: str) -> str:
    """Expand ``${VAR}`` and ``${VAR:-default}`` (unset VAR without default -> error)."""

    def sub(m: re.Match[str]) -> str:
        name, default = m.group(1), m.group(2)
        if name in os.environ:
            return os.environ[name]
        if default is None:
            raise KeyError(f"environment variable {name} is not set")
        return default

    return _ENV.sub(sub, text)


def load_suite(path: str | Path) -> SuiteConfig:
    path = Path(path)
    raw = yaml.safe_load(expand_env(path.read_text()))
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a mapping at the top level")
    raw.setdefault("base_dir", str(path.parent))
    return SuiteConfig.model_validate(raw)
