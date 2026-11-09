"""Turn MT-Bench human judgments into assaygate calibration cases.

Source: lmsys/mt_bench_human_judgments (CC BY 4.0), "human" split: 3,355 expert votes on
pairwise comparisons of six models' answers to the 80 MT-Bench questions, turns 1 and 2.

Steps
1. Group votes by (question, unordered model pair, turn). Votes on the same pair with the
   models in the other order are flipped so every vote refers to the same A/B.
2. Label = majority vote; pairs with no strict majority are dropped (documented count).
3. Split by **question id**, stratified by category: 3 questions per category -> dev (24),
   7 -> test (56). Prompt iteration and few-shot examples only ever use dev.
4. Sample a fixed number of test and dev pairs with a fixed seed.
5. Human-human agreement on multi-vote pairs (the ceiling a judge is measured against).

Usage: uv run python evals/mtbench/prepare.py [--test-n 400] [--dev-n 120]
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "mtbench"
REF_CATEGORIES = {"math", "reasoning", "coding"}
SEED = 20261108
FLIP = {"model_a": "model_b", "model_b": "model_a", "tie": "tie"}
TO_AB = {"model_a": "A", "model_b": "B", "tie": "tie"}


def render(conv: list[dict], turn: int) -> tuple[str, str]:
    """Return (question, answer) text for the judged turn."""
    if turn == 1:
        return conv[0]["content"], conv[1]["content"]
    question = f"[Turn 1] {conv[0]['content']}\n\n[Turn 2] {conv[2]['content']}"
    answer = (
        f"First reply:\n{conv[1]['content']}\n\n"
        f"User follow-up:\n{conv[2]['content']}\n\n"
        f"Second reply (judge this one):\n{conv[3]['content']}"
    )
    return question, answer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-n", type=int, default=400)
    ap.add_argument("--dev-n", type=int, default=120)
    args = ap.parse_args()

    votes = pq.read_table(RAW / "human.parquet").to_pylist()
    questions = {q["question_id"]: q for q in map(json.loads, (RAW / "question.jsonl").open())}
    refs = {r["question_id"]: r for r in map(json.loads, (RAW / "reference_gpt4.jsonl").open())}

    groups: dict[tuple, list[str]] = defaultdict(list)
    first_seen: dict[tuple, dict] = {}
    for v in votes:
        ma, mb = v["model_a"], v["model_b"]
        key = (v["question_id"], *sorted((ma, mb)), v["turn"])
        winner = v["winner"] if ma < mb else FLIP[v["winner"]]
        groups[key].append(winner)
        if key not in first_seen or ma < mb:
            first_seen[key] = (
                v
                if ma < mb
                else {
                    **v,
                    "model_a": mb,
                    "model_b": ma,
                    "conversation_a": v["conversation_b"],
                    "conversation_b": v["conversation_a"],
                }
            )

    cases = []
    dropped = 0
    multi_vote_pairs = []
    for key, ws in groups.items():
        if len(ws) >= 2:  # before the majority filter, so the ceiling isn't inflated
            multi_vote_pairs.append((key[0], ws))
        top, count = Counter(ws).most_common(1)[0]
        if count * 2 <= len(ws):
            dropped += 1
            continue
        v = first_seen[key]
        qid, turn = key[0], key[3]
        cat = questions[qid]["category"]
        q_text, ans_a = render(v["conversation_a"], turn)
        _, ans_b = render(v["conversation_b"], turn)
        ref = None
        if cat in REF_CATEGORIES and qid in refs:
            ref = refs[qid]["choices"][0]["turns"][turn - 1]
        cid = f"q{qid}-t{turn}-{key[1]}-vs-{key[2]}"
        cases.append(
            {
                "id": cid,
                "question_id": qid,
                "turn": turn,
                "category": cat,
                "model_a": key[1],
                "model_b": key[2],
                "question": q_text,
                "answer_a": ans_a,
                "answer_b": ans_b,
                "reference": ref,
                "focus": "Judge only the second reply; the first exchange is context."
                if turn == 2
                else None,
                "human": TO_AB[top],
                "n_votes": len(ws),
                "unanimous": len(set(ws)) == 1,
            }
        )

    rng = random.Random(SEED)
    by_cat: dict[str, list[int]] = defaultdict(list)
    for qid, q in sorted(questions.items()):
        by_cat[q["category"]].append(qid)
    dev_q: set[int] = set()
    for _cat, qids in sorted(by_cat.items()):
        dev_q.update(rng.sample(qids, 3))
    test_cases = sorted((c for c in cases if c["question_id"] not in dev_q), key=lambda c: c["id"])
    dev_cases = sorted((c for c in cases if c["question_id"] in dev_q), key=lambda c: c["id"])
    test = rng.sample(test_cases, min(args.test_n, len(test_cases)))
    dev = rng.sample(dev_cases, min(args.dev_n, len(dev_cases)))
    dev_ids = {c["id"] for c in dev}
    pool = [c for c in dev_cases if c["id"] not in dev_ids]  # few-shot candidates

    # Human-human agreement: every pair of votes on the same item (test questions only).
    agree = agree_nt = total = total_nt = 0
    for qid, ws in multi_vote_pairs:
        if qid in dev_q:
            continue
        for x, y in combinations(ws, 2):
            total += 1
            agree += x == y
            if x != "tie" and y != "tie":
                total_nt += 1
                agree_nt += x == y

    OUT.mkdir(parents=True, exist_ok=True)
    for name, rows in (("test", test), ("dev", dev), ("fewshot_pool", pool)):
        with (OUT / f"{name}.jsonl").open("w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
    summary = {
        "source_votes": len(votes),
        "unique_pairs": len(groups),
        "dropped_no_majority": dropped,
        "labelled_pairs": len(cases),
        "dev_questions": sorted(dev_q),
        "test_pool": len(test_cases),
        "dev_pool": len(dev_cases),
        "test_n": len(test),
        "dev_n": len(dev),
        "fewshot_pool_n": len(pool),
        "test_label_counts": Counter(c["human"] for c in test),
        "test_turn_counts": Counter(c["turn"] for c in test),
        "human_human_agreement_with_tie": agree / total if total else None,
        "human_human_vote_pairs": total,
        "human_human_agreement_without_tie": agree_nt / total_nt if total_nt else None,
        "human_human_vote_pairs_without_tie": total_nt,
        "seed": SEED,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=dict) + "\n")
    print(json.dumps(summary, indent=2, default=dict))


if __name__ == "__main__":
    main()
