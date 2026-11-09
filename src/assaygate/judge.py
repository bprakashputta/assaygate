"""LLM-as-judge: pairwise preference and single-answer rubric grading.

The pairwise judge is built from switchable techniques so their effect can be measured one at a
time (``assaygate calibrate`` does exactly that):

``rubric``     explicit criteria, bias guards and a tie policy instead of "which is better?"
``reference``  a reference answer is shown when the item has one (math, reasoning, code)
``cot``        the structured output has an ``analysis`` field *before* ``winner``, so the
               verdict is generated after the reasoning (chain-of-thought-then-verdict)
``swap``       the pair is judged in both orders and the two verdicts are combined, which
               cancels position bias (swap-and-average)
``few_shot``   human-labelled examples from a *dev* split are shown for calibration

Prompts are versioned files in ``assaygate/prompts``; the judge records which ones it used.
"""

from __future__ import annotations

import asyncio
from functools import cache
from importlib import resources
from typing import Literal

from pydantic import BaseModel, Field

from .llm import LLMClient, LLMSettings, StructuredOutputError

Winner = Literal["A", "B", "tie"]


@cache
def load_prompt(name: str) -> str:
    text = resources.files("assaygate.prompts").joinpath(f"{name}.md").read_text()
    lines = [ln for ln in text.splitlines() if not ln.startswith("<!--")]
    return "\n".join(lines).strip()


# ----------------------------------------------------------------------------- pairwise


class PairwiseCase(BaseModel):
    """One comparison. ``answer_a``/``answer_b`` may be full transcripts for multi-turn."""

    id: str
    question: str
    answer_a: str
    answer_b: str
    reference: str | None = None
    focus: str | None = Field(None, description="e.g. 'Judge only the final assistant turn.'")


class FewShotExample(BaseModel):
    case: PairwiseCase
    verdict: Winner
    note: str | None = None


class VerdictOnly(BaseModel):
    winner: Winner


class VerdictWithAnalysis(BaseModel):
    analysis: str = Field(description="brief comparison, written before the verdict")
    winner: Winner


class PairwiseJudgeConfig(BaseModel):
    name: str = "judge"
    rubric: bool = True
    reference: bool = True
    cot: bool = True
    swap: bool = True
    swap_policy: Literal["average", "conservative"] = "average"
    few_shot: list[FewShotExample] = Field(default_factory=list)
    settings: LLMSettings = Field(default_factory=LLMSettings)

    model_config = {"arbitrary_types_allowed": True}


class PairwiseVerdict(BaseModel):
    winner: Winner
    forward: Winner
    backward: Winner | None = None  # verdict from the swapped order, mapped back to A/B
    consistent: bool | None = None
    analysis: list[str] = Field(default_factory=list)
    cost_usd: float = 0.0
    latency_ms: float = 0.0

    @property
    def score_a(self) -> float:
        """1 = A wins, 0.5 = tie, 0 = B wins."""
        return {"A": 1.0, "tie": 0.5, "B": 0.0}[self.winner]


_FLIP: dict[str, Winner] = {"A": "B", "B": "A", "tie": "tie"}


def combine(forward: Winner, backward: Winner, policy: str = "average") -> Winner:
    """Combine the verdicts from both orders (``backward`` already mapped back to A/B)."""
    if forward == backward:
        return forward
    if policy == "conservative":
        return "tie"
    score = (
        {"A": 1.0, "tie": 0.5, "B": 0.0}[forward] + {"A": 1.0, "tie": 0.5, "B": 0.0}[backward]
    ) / 2
    return "A" if score > 0.5 else "B" if score < 0.5 else "tie"


def _render_case(case: PairwiseCase, *, a: str, b: str, show_reference: bool) -> str:
    parts = [f"<user_request>\n{case.question}\n</user_request>"]
    if show_reference and case.reference:
        parts.append(f"<reference_answer>\n{case.reference}\n</reference_answer>")
    parts.append(f"<response_A>\n{a}\n</response_A>")
    parts.append(f"<response_B>\n{b}\n</response_B>")
    if case.focus:
        parts.append(case.focus)
    return "\n\n".join(parts)


class PairwiseJudge:
    def __init__(self, config: PairwiseJudgeConfig, client: LLMClient):
        self.config = config
        self.client = client
        self._system = self._build_system()

    @property
    def prompt_files(self) -> list[str]:
        c = self.config
        files = ["pairwise_rubric" if c.rubric else "pairwise_naive"]
        if c.reference:
            files.append("pairwise_reference")
        if c.cot:
            files.append("pairwise_cot")
        if c.few_shot:
            files.append("pairwise_fewshot")
        return files

    def _build_system(self) -> list[dict]:
        c = self.config
        text = "\n\n".join(load_prompt(f) for f in self.prompt_files if f != "pairwise_fewshot")
        blocks = [{"type": "text", "text": text}]
        if c.few_shot:
            shots = [load_prompt("pairwise_fewshot")]
            for i, ex in enumerate(c.few_shot, 1):
                body = _render_case(
                    ex.case, a=ex.case.answer_a, b=ex.case.answer_b, show_reference=c.reference
                )
                note = f"\nAnnotator note: {ex.note}" if ex.note else ""
                shots.append(
                    f"<example_{i}>\n{body}\n\nExpert verdict: {ex.verdict}{note}\n</example_{i}>"
                )
            # Few-shot block is stable across items: mark it cacheable (prompt caching).
            blocks.append(
                {"type": "text", "text": "\n\n".join(shots), "cache_control": {"type": "ephemeral"}}
            )
        return blocks

    async def _one(self, case: PairwiseCase, swapped: bool) -> tuple[Winner, str, float, float]:
        a, b = (case.answer_b, case.answer_a) if swapped else (case.answer_a, case.answer_b)
        content = _render_case(case, a=a, b=b, show_reference=self.config.reference)
        model = VerdictWithAnalysis if self.config.cot else VerdictOnly
        try:
            res = await self.client.complete(
                [{"role": "user", "content": content}],
                system=self._system,
                output_model=model,
                settings=self.config.settings,
                tag=f"{self.config.name}:{case.id}:{'ba' if swapped else 'ab'}",
            )
        except StructuredOutputError:
            # No usable verdict after a re-ask: abstain (tie) and say so, rather than crash a
            # whole run. Calibration counts these; a gate run shows them in the item detail.
            return "tie", "[judge output invalid; abstained]", 0.0, 0.0
        winner: Winner = res.parsed.winner
        if swapped:
            winner = _FLIP[winner]
        analysis = getattr(res.parsed, "analysis", "")
        return winner, analysis, res.nominal_cost_usd, res.nominal_latency_ms

    async def judge(
        self, case: PairwiseCase, *, both_orders: bool | None = None
    ) -> PairwiseVerdict:
        """Judge one pair. ``both_orders`` forces the swapped call (for bias measurement)."""
        run_swap = self.config.swap if both_orders is None else (both_orders or self.config.swap)
        if not run_swap:
            w, an, cost, lat = await self._one(case, swapped=False)
            return PairwiseVerdict(
                winner=w, forward=w, analysis=[an], cost_usd=cost, latency_ms=lat
            )
        (fw, fa, fc, fl), (bw, ba, bc, bl) = await asyncio.gather(
            self._one(case, swapped=False), self._one(case, swapped=True)
        )
        winner = combine(fw, bw, self.config.swap_policy) if self.config.swap else fw
        return PairwiseVerdict(
            winner=winner,
            forward=fw,
            backward=bw,
            consistent=fw == bw,
            analysis=[fa, ba],
            cost_usd=fc + bc,
            latency_ms=max(fl, bl),
        )


# ----------------------------------------------------------------------------- rubric


class RubricGrade(BaseModel):
    score: int


class RubricGradeWithAnalysis(BaseModel):
    analysis: str
    score: int


class RubricJudge:
    """Single-answer grading on a 1..scale rubric; returns a 0-1 normalised score."""

    def __init__(
        self,
        client: LLMClient,
        rubric: str,
        *,
        scale: int = 5,
        cot: bool = True,
        settings: LLMSettings | None = None,
        name: str = "rubric",
    ):
        self.client = client
        self.scale = scale
        self.cot = cot
        self.settings = settings or LLMSettings()
        self.name = name
        self.system = (
            load_prompt("rubric_single")
            .replace("{{rubric}}", rubric)
            .replace("{{scale}}", str(scale))
        )
        if cot:
            self.system += "\n\n" + "Write a brief analysis (at most 80 words) before the score."

    async def grade(
        self, question: str, answer: str, reference: str | None = None, item_id: str = ""
    ) -> tuple[float, str, float]:
        parts = [f"<user_request>\n{question}\n</user_request>"]
        if reference:
            parts.append(f"<reference_answer>\n{reference}\n</reference_answer>")
        parts.append(f"<response>\n{answer}\n</response>")
        res = await self.client.complete(
            [{"role": "user", "content": "\n\n".join(parts)}],
            system=self.system,
            output_model=RubricGradeWithAnalysis if self.cot else RubricGrade,
            settings=self.settings,
            tag=f"{self.name}:{item_id}",
        )
        raw = min(max(int(res.parsed.score), 1), self.scale)  # validate model's number in code
        normalised = (raw - 1) / (self.scale - 1) if self.scale > 1 else float(raw)
        return normalised, getattr(res.parsed, "analysis", ""), res.nominal_cost_usd
