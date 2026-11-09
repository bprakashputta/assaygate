"""Judge calibration: measure an LLM judge against human labels before trusting it in a gate.

For each judge variant this reports

* **agreement** with humans (3-way A/B/tie) with a Wilson CI, and **Cohen's kappa** with a
  bootstrap CI (agreement corrected for chance, which matters when one label dominates);
* **non-tie agreement**: on items where the human picked a winner and the judge did too
  (the MT-Bench paper's "w/o tie" setup), plus how often the judge abstained to a tie;
* the **confusion matrix** (rows human, columns judge);
* **position bias**: every item is judged in both orders, whatever the variant's own policy.
  ``consistency`` = share of items whose verdict survives the swap; ``first_position_rate`` =
  among inconsistent pairs, how often the response shown first won;
* **verbosity bias**: on pairs whose lengths differ by >= 1.5x and where a winner was picked,
  how often the judge chose the longer response versus how often the humans did;
* a **paired** comparison of every variant against the first one (bootstrap CI on the
  difference in per-item agreement, McNemar exact p-value).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field

from . import stats
from .data import load_rows
from .judge import (
    FewShotExample,
    PairwiseCase,
    PairwiseJudge,
    PairwiseJudgeConfig,
    PairwiseVerdict,
    Winner,
)
from .llm import LLMClient, LLMSettings

LABELS: list[Winner] = ["A", "B", "tie"]


class LabeledCase(BaseModel):
    case: PairwiseCase
    human: Winner
    category: str | None = None


class VariantSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    rubric: bool = False
    reference: bool = False
    cot: bool = False
    swap: bool = False
    swap_policy: str = "average"
    few_shot_ids: list[str] = Field(default_factory=list)
    llm: dict[str, Any] = Field(default_factory=dict)


class CalibrationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = "judge-calibration"
    dataset: str
    few_shot_pool: str | None = Field(None, description="JSONL of labelled dev cases")
    limit: int | None = None
    concurrency: int = 8
    rpm: float | None = None
    spend_cap_usd: float = 5.0
    ledger_path: str = ".assaygate/spend.jsonl"
    cache_path: str | None = ".assaygate/cache.sqlite"
    variants: list[VariantSpec]
    base_dir: str = "."


class ItemResult(BaseModel):
    id: str
    human: Winner
    judge: Winner
    forward: Winner
    backward: Winner
    category: str | None = None
    len_a: int
    len_b: int
    invalid: bool = False


class VariantMetrics(BaseModel):
    name: str
    prompt_files: list[str]
    n: int
    agreement: dict[str, float]
    kappa: dict[str, float]
    non_tie_agreement: float | None
    non_tie_n: int
    judge_tie_rate: float
    human_tie_rate: float
    confusion: list[list[int]]
    consistency: float
    first_position_rate: float | None
    verbosity_judge_longer: float | None
    verbosity_human_longer: float | None
    verbosity_n: int
    n_invalid: int = 0
    vs_first: dict[str, float] | None = None
    by_category: dict[str, float] = Field(default_factory=dict)
    cost_usd: float
    cost_per_item_usd: float
    latency_ms_p50: float
    latency_ms_p95: float


class CalibrationResult(BaseModel):
    name: str
    labels: list[str] = LABELS
    n_items: int
    variants: list[VariantMetrics]
    items: dict[str, list[ItemResult]]


def row_to_case(row: dict[str, Any]) -> LabeledCase:
    return LabeledCase(
        case=PairwiseCase(
            id=str(row["id"]),
            question=row["question"],
            answer_a=row["answer_a"],
            answer_b=row["answer_b"],
            reference=row.get("reference"),
            focus=row.get("focus"),
        ),
        human=row["human"],
        category=row.get("category"),
    )


def load_cases(path: Path | str, limit: int | None = None) -> list[LabeledCase]:
    rows = load_rows(path)
    return [row_to_case(r) for r in (rows[:limit] if limit else rows)]


def _verbosity(items: list[ItemResult], who: str) -> tuple[float | None, int]:
    picks = []
    for it in items:
        la, lb = it.len_a, it.len_b
        if max(la, lb) < 1.5 * max(1, min(la, lb)):
            continue
        verdict = it.human if who == "human" else it.judge
        if verdict == "tie" or it.human == "tie":
            continue
        longer = "A" if la > lb else "B"
        picks.append(verdict == longer)
    return (float(np.mean(picks)) if picks else None), len(picks)


def summarise(
    name: str,
    prompt_files: list[str],
    items: list[ItemResult],
    costs: list[float],
    lats: list[float],
) -> VariantMetrics:
    human = [i.human for i in items]
    judge = [i.judge for i in items]
    n = len(items)
    agree = sum(h == j for h, j in zip(human, judge, strict=True))
    w = stats.wilson_interval(agree, n)
    k = stats.kappa_ci(human, judge, LABELS)
    non_tie = [(i.human, i.judge) for i in items if i.human != "tie" and i.judge != "tie"]
    inconsistent = [i for i in items if i.forward != i.backward]
    # forward = order A,B (A shown first); backward mapped back: "A" means A won while shown second
    first_wins = []
    for i in inconsistent:
        if i.forward in ("A", "B") and i.backward in ("A", "B"):
            first_wins.append(i.forward == "A" and i.backward == "B")  # first slot won both times
        elif i.forward == "tie" or i.backward == "tie":
            # one order tie, other picked: count the pick as first-position if it was shown first
            pick_fwd = i.forward != "tie"
            winner = i.forward if pick_fwd else i.backward
            shown_first = (winner == "A") if pick_fwd else (winner == "B")
            first_wins.append(shown_first)
    vj, vn = _verbosity(items, "judge")
    vh, _ = _verbosity(items, "human")
    cats: dict[str, list[bool]] = {}
    for i in items:
        if i.category:
            cats.setdefault(i.category, []).append(i.human == i.judge)
    return VariantMetrics(
        name=name,
        prompt_files=prompt_files,
        n=n,
        agreement=w.as_dict(),
        kappa=k.as_dict(),
        non_tie_agreement=(float(np.mean([h == j for h, j in non_tie])) if non_tie else None),
        non_tie_n=len(non_tie),
        judge_tie_rate=judge.count("tie") / n,
        human_tie_rate=human.count("tie") / n,
        confusion=stats.confusion_matrix(human, judge, LABELS).tolist(),
        consistency=1 - len(inconsistent) / n,
        first_position_rate=float(np.mean(first_wins)) if first_wins else None,
        verbosity_judge_longer=vj,
        verbosity_human_longer=vh,
        verbosity_n=vn,
        n_invalid=sum(i.invalid for i in items),
        by_category={c: float(np.mean(v)) for c, v in sorted(cats.items())},
        cost_usd=float(sum(costs)),
        cost_per_item_usd=float(sum(costs) / n),
        latency_ms_p50=float(np.percentile(lats, 50)) if lats else 0.0,
        latency_ms_p95=float(np.percentile(lats, 95)) if lats else 0.0,
    )


def build_judge(
    spec: VariantSpec, llm: LLMClient, pool: dict[str, LabeledCase] | None = None
) -> PairwiseJudge:
    shots = []
    for sid in spec.few_shot_ids:
        if not pool or sid not in pool:
            raise KeyError(f"few-shot id {sid!r} not in few_shot_pool")
        lc = pool[sid]
        shots.append(FewShotExample(case=lc.case, verdict=lc.human))
    cfg = PairwiseJudgeConfig(
        name=spec.name,
        rubric=spec.rubric,
        reference=spec.reference,
        cot=spec.cot,
        swap=spec.swap,
        swap_policy=spec.swap_policy,  # type: ignore[arg-type]
        few_shot=shots,
        settings=LLMSettings(**spec.llm),
    )
    return PairwiseJudge(cfg, llm)


async def run_variant(
    judge: PairwiseJudge, cases: list[LabeledCase], concurrency: int
) -> tuple[list[ItemResult], list[float], list[float]]:
    sem = asyncio.Semaphore(concurrency)

    async def one(lc: LabeledCase) -> PairwiseVerdict:
        async with sem:
            return await judge.judge(lc.case, both_orders=True)

    verdicts = await asyncio.gather(*(one(lc) for lc in cases))
    items = [
        ItemResult(
            id=lc.case.id,
            human=lc.human,
            judge=v.winner,
            forward=v.forward,
            backward=v.backward or v.forward,
            category=lc.category,
            len_a=len(lc.case.answer_a),
            len_b=len(lc.case.answer_b),
            invalid=any(a.startswith("[judge output invalid") for a in v.analysis),
        )
        for lc, v in zip(cases, verdicts, strict=True)
    ]
    return items, [v.cost_usd for v in verdicts], [v.latency_ms for v in verdicts if v.latency_ms]


def paired_vs(first: list[ItemResult], other: list[ItemResult]) -> dict[str, float]:
    a = np.array([i.human == i.judge for i in first], dtype=float)
    b = np.array([i.human == i.judge for i in other], dtype=float)
    ci = stats.paired_bootstrap_diff(a, b)
    lost = int(np.sum((a == 1) & (b == 0)))
    won = int(np.sum((a == 0) & (b == 1)))
    return {
        "delta": ci.estimate,
        "low": ci.low,
        "high": ci.high,
        "mcnemar_p": stats.mcnemar_exact(lost, won),
        "lost": lost,
        "gained": won,
    }


async def calibrate(
    config: CalibrationConfig, llm: LLMClient, *, cases: list[LabeledCase] | None = None
) -> CalibrationResult:
    base = Path(config.base_dir)
    cases = cases or load_cases(base / config.dataset, config.limit)
    pool = None
    if config.few_shot_pool:
        pool = {lc.case.id: lc for lc in load_cases(base / config.few_shot_pool)}
        overlap = set(pool) & {c.case.id for c in cases}
        if overlap:
            raise ValueError(f"few-shot pool overlaps the evaluation set: {sorted(overlap)[:5]}")
    metrics: list[VariantMetrics] = []
    all_items: dict[str, list[ItemResult]] = {}
    for spec in config.variants:
        judge = build_judge(spec, llm, pool)
        items, costs, lats = await run_variant(judge, cases, config.concurrency)
        m = summarise(spec.name, judge.prompt_files, items, costs, lats)
        if all_items:
            m.vs_first = paired_vs(next(iter(all_items.values())), items)
        metrics.append(m)
        all_items[spec.name] = items
    return CalibrationResult(
        name=config.name, n_items=len(cases), variants=metrics, items=all_items
    )


def load_calibration_config(path: Path | str) -> CalibrationConfig:
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    raw.setdefault("base_dir", str(path.parent))
    return CalibrationConfig.model_validate(raw)
