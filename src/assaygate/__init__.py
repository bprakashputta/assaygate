"""assaygate: LLM evaluation and regression gating for CI."""

from .calibrate import CalibrationConfig, VariantSpec, calibrate
from .config import GateConfig, MetricGate, SuiteConfig, load_suite
from .gate import GateResult, compare, to_markdown
from .judge import PairwiseCase, PairwiseJudge, PairwiseJudgeConfig, RubricJudge
from .llm import LLMClient, LLMSettings, ResponseCache, SpendCapExceeded, SpendLedger
from .runner import Run, load_run, run_suite
from .scorers import Score, register

__all__ = [
    "CalibrationConfig",
    "GateConfig",
    "GateResult",
    "LLMClient",
    "LLMSettings",
    "MetricGate",
    "PairwiseCase",
    "PairwiseJudge",
    "PairwiseJudgeConfig",
    "ResponseCache",
    "RubricJudge",
    "Run",
    "Score",
    "SpendCapExceeded",
    "SpendLedger",
    "SuiteConfig",
    "VariantSpec",
    "calibrate",
    "compare",
    "load_run",
    "load_suite",
    "register",
    "run_suite",
    "to_markdown",
]
__version__ = "0.1.0"
