#!/usr/bin/env bash
# Downloads the public datasets used by the evals into data/raw (gitignored).
#   MT-Bench human judgments  lmsys/mt_bench_human_judgments  CC BY 4.0
#   MT-Bench questions + GPT-4 reference answers  lm-sys/FastChat  Apache-2.0
#   Banking77  PolyAI/banking77  CC BY 4.0
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/raw
fetch() { [ -s "$2" ] || curl -fsSL --retry 3 -o "$2" "$1"; echo "ok  $2"; }
HF=https://huggingface.co/datasets
FC=https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/data/mt_bench
fetch "$HF/lmsys/mt_bench_human_judgments/resolve/main/data/human-00000-of-00001-25f4910818759289.parquet" data/raw/human.parquet
fetch "$FC/question.jsonl" data/raw/question.jsonl
fetch "$FC/reference_answer/gpt-4.jsonl" data/raw/reference_gpt4.jsonl
fetch "$HF/PolyAI/banking77/resolve/main/data/test-00000-of-00001.parquet" data/raw/banking77_test.parquet
fetch "$HF/PolyAI/banking77/resolve/main/data/train-00000-of-00001.parquet" data/raw/banking77_train.parquet
