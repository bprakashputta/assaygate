"""Scorers turn (item, output) into a 0-1 value. Deterministic ones are free; judges cost money.

Every scorer returns a :class:`Score`; ``value`` is what the gate compares (mean over items),
``passed`` is the per-item pass/fail used for the failure diff in the report.

Register your own with :func:`register`::

    @register("starts_with_yes")
    def build(params, ctx):
        async def score(item, output):
            ok = str(output).lower().startswith("yes")
            return Score(value=float(ok), passed=ok)
        return score
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import jsonschema
from pydantic import BaseModel

from .data import Item
from .judge import PairwiseCase, PairwiseJudge, PairwiseJudgeConfig, RubricJudge
from .llm import LLMClient, LLMSettings
from .targets import dig, import_ref


class Score(BaseModel):
    value: float
    passed: bool
    detail: str | None = None
    cost_usd: float = 0.0


ScoreFn = Callable[[Item, Any], Awaitable[Score]]


@dataclass
class ScorerContext:
    llm: LLMClient | None = None


Builder = Callable[[dict[str, Any], ScorerContext], ScoreFn]
_REGISTRY: dict[str, Builder] = {}


def register(name: str) -> Callable[[Builder], Builder]:
    def deco(fn: Builder) -> Builder:
        _REGISTRY[name] = fn
        return fn

    return deco


def build_scorer(type_: str, params: dict[str, Any], ctx: ScorerContext) -> ScoreFn:
    if type_ not in _REGISTRY:
        raise KeyError(f"unknown scorer {type_!r}; known: {sorted(_REGISTRY)}")
    return _REGISTRY[type_](params, ctx)


def available_scorers() -> list[str]:
    return sorted(_REGISTRY)


# ----------------------------------------------------------------------------- helpers


def _text(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    return json.dumps(x, sort_keys=True, ensure_ascii=False)


def _expected(item: Item, params: dict[str, Any]) -> Any:
    field = params.get("expected_field")
    return item.fields.get(field) if field else item.expected


def _select(output: Any, params: dict[str, Any]) -> Any:
    """Optionally pick a sub-field of a structured output (``output_path: label``)."""
    path = params.get("output_path")
    if not path:
        return output
    if isinstance(output, str):
        try:
            output = json.loads(output)
        except json.JSONDecodeError:
            return None
    try:
        return dig(output, path)
    except (KeyError, IndexError, ValueError):
        return None


_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]")


def normalise(text: str, *, case: bool = False, punct: bool = False) -> str:
    t = text if case else text.lower()
    if not punct:
        t = _PUNCT.sub(" ", t)
    return _WS.sub(" ", t).strip()


def tokens(text: str) -> list[str]:
    return normalise(text).split()


def token_f1(pred: str, ref: str) -> float:
    """SQuAD-style token F1 (bag of words overlap)."""
    p, r = tokens(pred), tokens(ref)
    if not p and not r:
        return 1.0
    if not p or not r:
        return 0.0
    common = Counter(p) & Counter(r)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(p), overlap / len(r)
    return 2 * precision * recall / (precision + recall)


def rouge_l(pred: str, ref: str) -> float:
    """ROUGE-L F-measure from the longest common subsequence of tokens."""
    p, r = tokens(pred), tokens(ref)
    if not p or not r:
        return float(p == r)
    prev = [0] * (len(r) + 1)
    for tp in p:
        cur = [0]
        for j, tr in enumerate(r, 1):
            cur.append(prev[j - 1] + 1 if tp == tr else max(prev[j], cur[j - 1]))
        prev = cur
    lcs = prev[-1]
    if lcs == 0:
        return 0.0
    precision, recall = lcs / len(p), lcs / len(r)
    return 2 * precision * recall / (precision + recall)


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


# ----------------------------------------------------------------------------- deterministic


@register("exact")
def _exact(params: dict[str, Any], ctx: ScorerContext) -> ScoreFn:
    case = bool(params.get("case_sensitive", False))
    strip_punct = not params.get("keep_punctuation", True)

    async def score(item: Item, output: Any) -> Score:
        got = _text(_select(output, params))
        exp = _text(_expected(item, params))
        g = normalise(got, case=case, punct=not strip_punct)
        e = normalise(exp, case=case, punct=not strip_punct)
        ok = g == e
        return Score(value=float(ok), passed=ok, detail=None if ok else f"expected {exp!r}")

    return score


@register("regex")
def _regex(params: dict[str, Any], ctx: ScorerContext) -> ScoreFn:
    pattern = re.compile(params["pattern"], re.IGNORECASE if params.get("ignore_case") else 0)
    must_match = params.get("must_match", True)

    async def score(item: Item, output: Any) -> Score:
        found = pattern.search(_text(_select(output, params))) is not None
        ok = found if must_match else not found
        return Score(value=float(ok), passed=ok)

    return score


@register("json_schema")
def _json_schema(params: dict[str, Any], ctx: ScorerContext) -> ScoreFn:
    schema = params["schema"]
    validator = jsonschema.Draft202012Validator(schema)

    async def score(item: Item, output: Any) -> Score:
        value = _select(output, params)
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError as exc:
                return Score(value=0.0, passed=False, detail=f"not JSON: {exc.msg}")
        errors = sorted(validator.iter_errors(value), key=lambda e: list(e.path))
        if errors:
            return Score(value=0.0, passed=False, detail=errors[0].message[:200])
        return Score(value=1.0, passed=True)

    return score


@register("numeric")
def _numeric(params: dict[str, Any], ctx: ScorerContext) -> ScoreFn:
    abs_tol = float(params.get("abs_tol", 0.0))
    rel_tol = float(params.get("rel_tol", 1e-9))
    number = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?(?:[eE]-?\d+)?")

    def parse(x: Any) -> float | None:
        if isinstance(x, int | float):
            return float(x)
        found = number.findall(_text(x))
        return float(found[-1].replace(",", "")) if found else None

    async def score(item: Item, output: Any) -> Score:
        got, exp = parse(_select(output, params)), parse(_expected(item, params))
        if got is None or exp is None:
            return Score(value=0.0, passed=False, detail="no number found")
        ok = math.isclose(got, exp, rel_tol=rel_tol, abs_tol=abs_tol)
        return Score(value=float(ok), passed=ok, detail=None if ok else f"{got} vs {exp}")

    return score


@register("token_f1")
def _token_f1(params: dict[str, Any], ctx: ScorerContext) -> ScoreFn:
    threshold = float(params.get("threshold", 0.5))

    async def score(item: Item, output: Any) -> Score:
        v = token_f1(_text(_select(output, params)), _text(_expected(item, params)))
        return Score(value=v, passed=v >= threshold)

    return score


@register("rouge_l")
def _rouge_l(params: dict[str, Any], ctx: ScorerContext) -> ScoreFn:
    threshold = float(params.get("threshold", 0.4))

    async def score(item: Item, output: Any) -> Score:
        v = rouge_l(_text(_select(output, params)), _text(_expected(item, params)))
        return Score(value=v, passed=v >= threshold)

    return score


@register("embedding_similarity")
def _embedding(params: dict[str, Any], ctx: ScorerContext) -> ScoreFn:
    """Cosine similarity from a user-supplied embedder ``embed: 'pkg.mod:fn'``.

    ``fn(list[str]) -> list[list[float]]`` (sync or async). Kept provider-agnostic on purpose.
    """
    import inspect

    embed = import_ref(params["embed"])
    threshold = float(params.get("threshold", 0.8))

    async def score(item: Item, output: Any) -> Score:
        texts = [_text(_select(output, params)), _text(_expected(item, params))]
        vecs = embed(texts)
        if inspect.isawaitable(vecs):
            vecs = await vecs
        v = cosine(vecs[0], vecs[1])
        return Score(value=v, passed=v >= threshold)

    return score


# ----------------------------------------------------------------------------- LLM judges


def _settings(params: dict[str, Any]) -> LLMSettings:
    return LLMSettings(**params.get("llm", {}))


def _need_llm(ctx: ScorerContext) -> LLMClient:
    if ctx.llm is None:
        raise RuntimeError("this scorer needs an LLM client (ScorerContext.llm)")
    return ctx.llm


@register("llm_rubric")
def _llm_rubric(params: dict[str, Any], ctx: ScorerContext) -> ScoreFn:
    judge = RubricJudge(
        _need_llm(ctx),
        params["rubric"],
        scale=int(params.get("scale", 5)),
        cot=bool(params.get("cot", True)),
        settings=_settings(params),
        name=params.get("name", "llm_rubric"),
    )
    threshold = float(params.get("threshold", 0.75))
    question_field = params.get("question_field")
    reference_field = params.get("reference_field")

    async def score(item: Item, output: Any) -> Score:
        q = _text(item.fields.get(question_field) if question_field else item.input)
        ref = _text(item.fields.get(reference_field)) if reference_field else None
        v, analysis, cost = await judge.grade(q, _text(_select(output, params)), ref, item.id)
        return Score(value=v, passed=v >= threshold, detail=analysis or None, cost_usd=cost)

    return score


@register("llm_pairwise")
def _llm_pairwise(params: dict[str, Any], ctx: ScorerContext) -> ScoreFn:
    """Candidate output (as A) vs the item's expected/reference output (as B).

    Value: 1 win, 0.5 tie, 0 loss — the mean is a win rate against the reference outputs.
    """
    cfg = PairwiseJudgeConfig(
        name=params.get("name", "llm_pairwise"),
        rubric=params.get("rubric", True),
        reference=False,
        cot=params.get("cot", True),
        swap=params.get("swap", True),
        swap_policy=params.get("swap_policy", "average"),
        settings=_settings(params),
    )
    judge = PairwiseJudge(cfg, _need_llm(ctx))
    question_field = params.get("question_field")

    async def score(item: Item, output: Any) -> Score:
        case = PairwiseCase(
            id=item.id,
            question=_text(item.fields.get(question_field) if question_field else item.input),
            answer_a=_text(_select(output, params)),
            answer_b=_text(_expected(item, params)),
        )
        v = await judge.judge(case)
        return Score(
            value=v.score_a,
            passed=v.winner != "B",
            detail=f"{v.winner} (fwd {v.forward}, bwd {v.backward})",
            cost_usd=v.cost_usd,
        )

    return score
