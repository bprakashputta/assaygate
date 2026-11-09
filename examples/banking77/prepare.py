"""Build the Banking77 demo splits (Banking77: PolyAI, CC BY 4.0).

data/banking77/labels.json     the 77 intent names
data/banking77/examples.json   2 training utterances per intent (the app's few-shot examples)
data/banking77/dev.jsonl       154 other training utterances (2 per intent) for iterating
data/banking77/test.jsonl      300 test utterances, seeded random sample: the gated eval set
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
RAW, OUT = ROOT / "data" / "raw", ROOT / "data" / "banking77"
SEED = 77


def main() -> None:
    meta = json.loads(pq.read_metadata(RAW / "banking77_test.parquet").metadata[b"huggingface"])
    names = meta["info"]["features"]["label"]["names"]
    train = pq.read_table(RAW / "banking77_train.parquet").to_pylist()
    test = pq.read_table(RAW / "banking77_test.parquet").to_pylist()
    rng = random.Random(SEED)
    by_label: dict[int, list[str]] = defaultdict(list)
    for row in train:
        by_label[row["label"]].append(row["text"])
    examples, dev = {}, []
    for lab in range(len(names)):
        picks = rng.sample(by_label[lab], 4)
        examples[names[lab]] = picks[:2]
        dev += [{"text": t, "label": names[lab]} for t in picks[2:]]
    sample = rng.sample(test, 300)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "labels.json").write_text(json.dumps(names, indent=1))
    (OUT / "examples.json").write_text(json.dumps(examples, indent=1))
    for name, rows in (
        ("dev", dev),
        ("test", [{"text": r["text"], "label": names[r["label"]]} for r in sample]),
    ):
        with (OUT / f"{name}.jsonl").open("w") as fh:
            for i, r in enumerate(rows):
                fh.write(
                    json.dumps(
                        {"id": f"{name}-{i:04d}", "input": r["text"], "expected": r["label"]}
                    )
                    + "\n"
                )
    print(
        f"labels={len(names)} examples={sum(map(len, examples.values()))} "
        f"dev={len(dev)} test={len(sample)}"
    )


if __name__ == "__main__":
    main()
