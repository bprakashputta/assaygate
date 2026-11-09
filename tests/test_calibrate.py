from __future__ import annotations

import json

import pytest

from assaygate.calibrate import (
    CalibrationConfig,
    ItemResult,
    VariantSpec,
    calibrate,
    load_cases,
    summarise,
)

from .conftest import user_text


def item(i, human, judge, fwd, bwd, la=100, lb=100, cat="math"):
    return ItemResult(
        id=str(i),
        human=human,
        judge=judge,
        forward=fwd,
        backward=bwd,
        len_a=la,
        len_b=lb,
        category=cat,
    )


def test_summarise_metrics():
    items = [
        item(1, "A", "A", "A", "A"),
        item(2, "B", "B", "B", "B"),
        item(3, "A", "B", "B", "B"),
        item(4, "tie", "A", "A", "B"),  # inconsistent, first position won both times
        item(5, "B", "tie", "tie", "B", la=400, lb=100),
    ]
    m = summarise("v", ["p"], items, [0.001] * 5, [100.0] * 5)
    assert m.agreement["estimate"] == pytest.approx(2 / 5)
    assert m.consistency == pytest.approx(3 / 5)
    # item 4: first slot won both orders; item 5: tie then "B" when B shown first -> first
    assert m.first_position_rate == pytest.approx(1.0)
    assert m.non_tie_n == 3 and m.non_tie_agreement == pytest.approx(2 / 3)
    assert m.confusion[0] == [1, 1, 0]  # human A row
    assert m.cost_per_item_usd == pytest.approx(0.001)


def test_verbosity_bias():
    items = [item(i, "A", "B", "B", "B", la=100, lb=500) for i in range(4)] + [
        item(9, "B", "B", "B", "B", la=100, lb=500)
    ]
    m = summarise("v", [], items, [0] * 5, [1] * 5)
    assert m.verbosity_judge_longer == 1.0 and m.verbosity_human_longer == pytest.approx(0.2)


async def test_calibrate_end_to_end_with_fake_judge(tmp_path, make_llm):
    rows = [
        {
            "id": f"c{i}",
            "question": "q",
            "answer_a": "good" if i % 2 else "bad",
            "answer_b": "bad" if i % 2 else "good",
            "human": "A" if i % 2 else "B",
        }
        for i in range(10)
    ]
    (tmp_path / "cases.jsonl").write_text("\n".join(json.dumps(r) for r in rows))

    def responder(req):  # picks whichever slot holds "good": an unbiased, accurate judge
        t = user_text(req)
        return {"winner": "A" if "<response_A>\ngood" in t else "B"}

    llm, _ = make_llm(responder)
    cfg = CalibrationConfig(
        dataset="cases.jsonl",
        base_dir=str(tmp_path),
        variants=[VariantSpec(name="naive"), VariantSpec(name="swap", swap=True)],
    )
    res = await calibrate(cfg, llm)
    assert res.n_items == 10
    for v in res.variants:
        assert v.agreement["estimate"] == 1.0 and v.consistency == 1.0
    assert res.variants[1].vs_first["delta"] == 0.0
    assert len(load_cases(tmp_path / "cases.jsonl")) == 10


async def test_few_shot_pool_must_not_overlap(tmp_path, make_llm):
    row = {"id": "x", "question": "q", "answer_a": "a", "answer_b": "b", "human": "A"}
    (tmp_path / "c.jsonl").write_text(json.dumps(row))
    llm, _ = make_llm(lambda r: {"winner": "A"})
    cfg = CalibrationConfig(
        dataset="c.jsonl",
        few_shot_pool="c.jsonl",
        base_dir=str(tmp_path),
        variants=[VariantSpec(name="v")],
    )
    with pytest.raises(ValueError, match="overlaps"):
        await calibrate(cfg, llm)


async def test_calibration_report_features_declared_final_variant(tmp_path, make_llm):
    from assaygate.report import render_calibration_html

    rows = [
        {
            "id": f"c{i}",
            "question": "q",
            "answer_a": "x" * (10 + i),
            "answer_b": "y" * 40,
            "human": "AB"[i % 2],
        }
        for i in range(8)
    ]
    (tmp_path / "c.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    llm, _ = make_llm(lambda r: {"winner": "A"})  # maximally position-biased judge
    cfg = CalibrationConfig(
        dataset="c.jsonl",
        base_dir=str(tmp_path),
        variants=[
            VariantSpec(name="naive"),
            VariantSpec(name="final", swap=True, swap_policy="conservative"),
        ],
    )
    res = await calibrate(cfg, llm)
    naive, final = res.variants
    assert naive.consistency == 0.0 and naive.first_position_rate == 1.0
    assert final.judge_tie_rate == 1.0  # every pair flips, conservative swap -> tie
    html = render_calibration_html(res, {"final_variant": "final", "judge_model": "fake"})
    assert "Position bias" in html and "final" in html and "<html" in html
