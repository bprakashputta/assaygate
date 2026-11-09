from __future__ import annotations

import pytest

from assaygate.data import Item
from assaygate.scorers import (
    ScorerContext,
    available_scorers,
    build_scorer,
    rouge_l,
    token_f1,
)


def item(expected, **fields):
    return Item(id="1", input="q", expected=expected, fields={"expected": expected, **fields})


async def score(type_, params, expected, output, ctx=None):
    fn = build_scorer(type_, params, ctx or ScorerContext())
    return await fn(item(expected), output)


def test_registry_lists_builtins():
    assert {
        "exact",
        "regex",
        "json_schema",
        "numeric",
        "token_f1",
        "rouge_l",
        "embedding_similarity",
        "llm_rubric",
        "llm_pairwise",
    } <= set(available_scorers())


async def test_exact_normalises_case_and_whitespace():
    assert (await score("exact", {}, "card_arrival", "  Card_Arrival ")).passed
    assert not (await score("exact", {}, "a", "b")).passed
    assert (await score("exact", {"output_path": "label"}, "x", {"label": "x"})).passed


async def test_regex():
    assert (await score("regex", {"pattern": r"^\d{3}-\d{4}$"}, None, "555-1234")).passed
    assert (
        await score(
            "regex", {"pattern": "sorry", "must_match": False, "ignore_case": True}, None, "Sure"
        )
    ).passed


async def test_json_schema():
    schema = {"type": "object", "required": ["n"], "properties": {"n": {"type": "integer"}}}
    assert (await score("json_schema", {"schema": schema}, None, '{"n": 3}')).passed
    bad = await score("json_schema", {"schema": schema}, None, '{"n": "3"}')
    assert not bad.passed and "integer" in bad.detail
    assert not (await score("json_schema", {"schema": schema}, None, "{oops")).passed


async def test_numeric_tolerance():
    assert (await score("numeric", {"abs_tol": 0.01}, "3.14", "The answer is 3.141")).passed
    assert (await score("numeric", {}, 1200, "about 1,200 units")).passed
    assert not (await score("numeric", {"rel_tol": 0.01}, 100, "103")).passed
    assert not (await score("numeric", {}, 1, "no digits")).passed


def test_token_f1_known_values():
    assert token_f1("the cat sat", "the cat sat") == 1.0
    # pred 4 tokens, ref 3, overlap 2: P=0.5 R=2/3 F1=4/7
    assert token_f1("a cat on mat", "the cat mat") == pytest.approx(4 / 7)
    assert token_f1("", "x") == 0.0


def test_rouge_l_known_value():
    # LCS("police killed the gunman", "police kill the gunman") = 3 -> P=R=3/4 -> F=0.75
    assert rouge_l("police kill the gunman", "police killed the gunman") == pytest.approx(0.75)


async def test_embedding_similarity_with_custom_embedder():
    s = await score(
        "embedding_similarity",
        {"embed": "tests.test_scorers:fake_embed", "threshold": 0.9},
        "hello world",
        "hello world",
    )
    assert s.value == pytest.approx(1.0) and s.passed


def fake_embed(texts):
    return [[len(t), t.count("o"), 1.0] for t in texts]


def test_unknown_scorer():
    with pytest.raises(KeyError):
        build_scorer("nope", {}, ScorerContext())


async def test_llm_scorers_need_client():
    with pytest.raises(RuntimeError):
        build_scorer("llm_rubric", {"rubric": "x"}, ScorerContext())


async def test_llm_pairwise_scorer(make_llm):
    llm, _ = make_llm(lambda r: {"analysis": "a", "winner": "A"})
    fn = build_scorer("llm_pairwise", {"swap": False}, ScorerContext(llm=llm))
    s = await fn(item("ref answer"), "candidate answer")
    assert s.value == 1.0 and s.passed and s.cost_usd > 0
