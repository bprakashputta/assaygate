"""Judge calibration on MT-Bench human judgments: a technique ladder, measured.

Each variant adds one technique to the previous one, so every step's effect is visible:

  1 naive         "which is better? A/B/tie", one order
  2 +rubric       criteria, bias guards, explicit tie policy
  3 +cot          analysis field generated before the verdict
  4 +reference    GPT-4 reference answers for math/reasoning/coding items
  5 +swap         judge both orders, average the two verdicts
  6 +few-shot     6 human-labelled examples from the dev split (2 A, 2 B, 2 tie)

All variants are judged in both orders so position bias is measured for each of them; only
variants 5-6 *use* the second order in their verdict. Judge: claude-haiku-5-5, thinking
disabled. Identical calls are cached, so variant 5 reuses variant 4's calls for free.

Usage:
  uv run python evals/mtbench/run.py --split dev     # iterate here
  uv run python evals/mtbench/run.py --split test    # report this
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path

from assaygate.calibrate import CalibrationConfig, VariantSpec, calibrate, load_cases
from assaygate.llm import LLMClient, RateLimiter, ResponseCache, SpendLedger
from assaygate.report import render_calibration_html

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "mtbench"
RESULTS = ROOT / "evals" / "results"


def pick_few_shot(pool_path: Path, per_label: int = 2) -> list[str]:
    """Deterministic: shortest unanimous multi-vote examples, per label, distinct categories."""
    rows = [json.loads(line) for line in pool_path.open()]
    rows = [r for r in rows if r["unanimous"] and r["n_votes"] >= 2]
    rows.sort(key=lambda r: (len(r["question"]) + len(r["answer_a"]) + len(r["answer_b"]), r["id"]))
    chosen: list[str] = []
    for label in ("A", "B", "tie"):
        cats: set[str] = set()
        for r in rows:
            if r["human"] == label and r["category"] not in cats:
                chosen.append(r["id"])
                cats.add(r["category"])
                if len(cats) == per_label:
                    break
    return chosen


def variants(few_shot_ids: list[str], model: str) -> list[VariantSpec]:
    llm = {"model": model, "max_tokens": 600, "thinking": "disabled"}
    full = {"rubric": True, "cot": True, "reference": True}
    return [
        VariantSpec(name="1 naive", llm=llm),
        VariantSpec(name="2 +rubric", rubric=True, llm=llm),
        VariantSpec(name="3 +cot", rubric=True, cot=True, llm=llm),
        VariantSpec(name="4 +reference", **full, llm=llm),
        # swap policy chosen on dev (conservative beat average by 4.2 pp there)
        VariantSpec(name="5 +swap", **full, swap=True, swap_policy="conservative", llm=llm),
        VariantSpec(
            name="6 +few-shot",
            **full,
            swap=True,
            swap_policy="conservative",
            few_shot_ids=few_shot_ids,
            llm=llm,
        ),
        # Side branches off the final judge (one change each), decided on dev, reported on test:
        VariantSpec(
            name="6a average swap",
            **full,
            swap=True,
            swap_policy="average",
            few_shot_ids=few_shot_ids,
            llm=llm,
        ),
        VariantSpec(name="6b no swap", **full, few_shot_ids=few_shot_ids, llm=llm),
        VariantSpec(
            name="6c thinking, no CoT field",
            rubric=True,
            reference=True,
            swap=True,
            swap_policy="conservative",
            few_shot_ids=few_shot_ids,
            llm={**llm, "thinking": "adaptive", "effort": "medium", "max_tokens": 4000},
        ),
    ]


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--model", default="claude-haiku-5-5")
    ap.add_argument("--only", nargs="*", help="variant name prefixes to run (e.g. 1 6)")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "WARNING"))

    shots = pick_few_shot(DATA / "fewshot_pool.jsonl")
    vs = variants(shots, args.model)
    if args.only:
        vs = [v for v in vs if any(v.name.startswith(p) for p in args.only)]
    cfg = CalibrationConfig(
        name=f"MT-Bench human judgments · {args.split}",
        dataset=str(DATA / f"{args.split}.jsonl"),
        few_shot_pool=str(DATA / "fewshot_pool.jsonl"),
        limit=args.limit,
        concurrency=6,
        variants=vs,
    )
    llm = LLMClient(
        ledger=SpendLedger(ROOT / "data" / "spend.jsonl", cap_usd=5.0),
        cache=ResponseCache(ROOT / "data" / "cache.sqlite"),
        limiter=RateLimiter(rpm=300),
        max_in_flight=6,
    )
    cases = load_cases(cfg.dataset, cfg.limit)
    before = llm.ledger.spent_usd
    result = await calibrate(cfg, llm, cases=cases)
    spent = llm.ledger.spent_usd - before

    summary = json.loads((DATA / "summary.json").read_text())
    stem = f"mtbench_{args.split}{('_' + args.tag) if args.tag else ''}"
    RESULTS.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": "lmsys/mt_bench_human_judgments (human split), CC BY 4.0",
        "split": args.split,
        "n": result.n_items,
        "judge_model": args.model,
        "few_shot_ids": shots,
        "final_variant": "6 +few-shot",
        "human_human": {
            "with_tie": summary["human_human_agreement_with_tie"],
            "without_tie": summary["human_human_agreement_without_tie"],
            "vote_pairs": summary["human_human_vote_pairs"],
        },
        "spend_this_run_usd": spent,
        "llm_calls": llm.calls,
        "cache_hits": llm.cache_hits,
        "variants": [v.model_dump() for v in result.variants],
    }
    (RESULTS / f"{stem}.json").write_text(json.dumps(payload, indent=2) + "\n")
    (RESULTS / f"{stem}.md").write_text(markdown(payload))
    (RESULTS / f"{stem}.html").write_text(render_calibration_html(result, payload))
    print(markdown(payload))
    print(f"spend this run ${spent:.4f} · ledger total ${llm.ledger.spent_usd:.4f}")


def pct(x: float | None) -> str:
    return "–" if x is None else f"{100 * x:.1f}%"


def markdown(p: dict) -> str:
    hh = p["human_human"]
    lines = [
        f"# Judge calibration: MT-Bench human judgments ({p['split']}, N={p['n']})",
        "",
        f"Judge `{p['judge_model']}`. Human-human agreement on the same data: "
        f"{pct(hh['with_tie'])} with ties, {pct(hh['without_tie'])} without "
        f"({hh['vote_pairs']} vote pairs).",
        "",
        "| variant | agreement (95% CI) | kappa (95% CI) | non-tie agreement | judge tie rate"
        " | swap consistency | picks longer (judge / human) | Δ vs naive (95% CI) | $/item |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for v in p["variants"]:
        a, k = v["agreement"], v["kappa"]
        vs = v.get("vs_first")
        delta = (
            f"{100 * vs['delta']:+.1f} [{100 * vs['low']:+.1f}, {100 * vs['high']:+.1f}]"
            if vs
            else "–"
        )
        lines.append(
            f"| {v['name']} | {pct(a['estimate'])} [{pct(a['low'])}, {pct(a['high'])}] "
            f"| {k['estimate']:.3f} [{k['low']:.3f}, {k['high']:.3f}] "
            f"| {pct(v['non_tie_agreement'])} (n={v['non_tie_n']}) | {pct(v['judge_tie_rate'])} "
            f"| {pct(v['consistency'])} "
            f"| {pct(v['verbosity_judge_longer'])} / {pct(v['verbosity_human_longer'])} "
            f"| {delta} | ${v['cost_per_item_usd']:.5f} |"
        )
    lines += [
        "",
        f"Human tie rate: {pct(p['variants'][0]['human_tie_rate'])}. "
        f"Spend for this run: ${p['spend_this_run_usd']:.4f} "
        f"({p['llm_calls']} judge calls, {p['cache_hits']} served from cache).",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    asyncio.run(main())
