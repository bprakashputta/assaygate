from __future__ import annotations

import json

import pytest

from assaygate.judge import (
    FewShotExample,
    PairwiseCase,
    PairwiseJudge,
    PairwiseJudgeConfig,
    RubricJudge,
    combine,
    load_prompt,
)

from .conftest import user_text

CASE = PairwiseCase(id="c1", question="2+2?", answer_a="4", answer_b="5", reference="4")


@pytest.mark.parametrize(
    "fwd,bwd,policy,expected",
    [
        ("A", "A", "average", "A"),
        ("A", "B", "average", "tie"),
        ("A", "tie", "average", "A"),
        ("tie", "B", "average", "B"),
        ("A", "tie", "conservative", "tie"),
        ("B", "B", "conservative", "B"),
    ],
)
def test_combine(fwd, bwd, policy, expected):
    assert combine(fwd, bwd, policy) == expected


async def test_position_biased_judge_is_neutralised_by_swap(make_llm):
    # A judge that always says "A" (pure first-position bias).
    llm, sdk = make_llm(lambda r: {"winner": "A"})
    judge = PairwiseJudge(PairwiseJudgeConfig(cot=False, swap=True), llm)
    v = await judge.judge(CASE)
    assert v.forward == "A" and v.backward == "B" and v.consistent is False
    assert v.winner == "tie"
    # the swapped call really showed answer_b first
    texts = [user_text(r) for r in sdk.messages.requests]
    assert any("<response_A>\n5\n</response_A>" in t for t in texts)
    assert any("<response_A>\n4\n</response_A>" in t for t in texts)


async def test_single_order_judge_and_both_orders_flag(make_llm):
    llm, sdk = make_llm(lambda r: {"analysis": "A is right", "winner": "A"})
    judge = PairwiseJudge(PairwiseJudgeConfig(swap=False, cot=True), llm)
    v = await judge.judge(CASE)
    assert v.winner == "A" and v.backward is None and len(sdk.messages.requests) == 1
    v2 = await judge.judge(CASE, both_orders=True)  # measure bias, keep single-order verdict
    assert v2.winner == v2.forward == "A" and v2.backward == "B"


async def test_prompt_composition_by_technique(make_llm):
    llm, sdk = make_llm(lambda r: {"analysis": "x", "winner": "B"})
    naive = PairwiseJudge(
        PairwiseJudgeConfig(rubric=False, reference=False, cot=False, swap=False), llm
    )
    await naive.judge(CASE)
    req = sdk.messages.requests[-1]
    assert "reference_answer" not in user_text(req)
    assert req["system"][0]["text"] == load_prompt("pairwise_naive")
    schema = req["output_config"]["format"]["schema"]
    assert list(schema["properties"]) == ["winner"]

    full = PairwiseJudge(
        PairwiseJudgeConfig(
            few_shot=[FewShotExample(case=CASE, verdict="A")], swap=False, cot=True
        ),
        llm,
    )
    await full.judge(CASE)
    req = sdk.messages.requests[-1]
    assert "<reference_answer>" in user_text(req)
    assert list(req["output_config"]["format"]["schema"]["properties"]) == ["analysis", "winner"]
    assert req["system"][-1]["cache_control"] == {"type": "ephemeral"}
    assert "Expert verdict: A" in req["system"][-1]["text"]
    assert full.prompt_files == [
        "pairwise_rubric",
        "pairwise_reference",
        "pairwise_cot",
        "pairwise_fewshot",
    ]


async def test_rubric_judge_normalises_and_clamps(make_llm):
    llm, _ = make_llm(lambda r: {"analysis": "ok", "score": 9})
    j = RubricJudge(llm, "Is it correct?", scale=5)
    v, analysis, _ = await j.grade("q", "a")
    assert v == 1.0 and analysis == "ok"
    llm2, _ = make_llm(lambda r: json.dumps({"analysis": "bad", "score": 2}))
    v2, *_ = await RubricJudge(llm2, "r").grade("q", "a")
    assert v2 == pytest.approx(0.25)
