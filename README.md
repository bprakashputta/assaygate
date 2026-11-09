# assaygate

[![CI](https://github.com/bprakashputta/assaygate/actions/workflows/ci.yml/badge.svg)](https://github.com/bprakashputta/assaygate/actions/workflows/ci.yml)
[![Gate demo](https://github.com/bprakashputta/assaygate/actions/workflows/gate-demo.yml/badge.svg)](https://github.com/bprakashputta/assaygate/actions/workflows/gate-demo.yml)

**LLM evaluation and regression gating for CI: block a pull request when quality, $ cost or p95 latency gets worse, with confidence intervals instead of guesses, and with an LLM judge you have measured against humans first.**

Once an LLM app is past the demo, every prompt edit, model swap or refactor is a bet. assaygate turns it into a test: run a suite on the change, compare it item by item with the main-branch run, and fail the build only when the statistics say it really got worse.

- **Suites in YAML or Python:** a JSONL/CSV dataset, a system under test (any Python callable or an HTTP endpoint), and scorers: exact, regex, JSON schema, numeric tolerance, token-F1, ROUGE-L, embedding similarity (bring your embedder), and **LLM-as-judge** (rubric and pairwise) with structured outputs
- **A gate that doesn't cry wolf:** paired bootstrap CIs over items, permutation and McNemar tests, thresholds for quality, $ per item and p95 latency; exit code 1 fails CI. An unchanged prompt re-run passes; a prompt "simplification" that drops 6 points fails ([below](#end-to-end-the-gate-catching-a-real-regression))
- **Judge calibration:** agreement, Cohen's kappa, confusion matrix, position bias (every pair judged in both orders) and verbosity bias against human labels, with each judge technique switchable so its effect is measured, not assumed
- **Cheap and safe to run often:** sqlite response cache, bounded concurrency + rate limit + backoff, a hard spend cap enforced by a shared ledger with reservations, per-call token/$/latency logs
- **Artifacts you can read:** JSONL records + manifest per run, a self-contained HTML report (deltas with CIs, failure diff, cost and latency distributions), a Markdown PR comment, a FastAPI service to browse runs
- **A GitHub Action** that runs the gate and comments the summary on the PR

```
 dataset.jsonl ─┐         ┌────────────────────────── assaygate ──────────────────────────┐
 suite.yaml ────┼────────▶│ runner     bounded concurrency · rate limit · per-item timing │
                │         │   │  target: python callable │ HTTP endpoint (your app)        │
                │         │   ▼                                                            │
                │         │ scorers    exact · regex · json_schema · numeric · token_f1    │
                │         │            rouge_l · embedding · llm_rubric · llm_pairwise ─┐  │
                │         │                                                             ▼  │
                │         │ llm.py     structured outputs · cache · spend ledger · backoff │──▶ Claude API
                │         │   ▼                                                            │
                │         │ runs/<id>/records.jsonl + manifest.json                        │
                │         │   ▼                                                            │
 baseline run ──┴────────▶│ gate.py    paired bootstrap CIs · permutation · McNemar        │
                          │   ▼        quality / cost-ratio / p95-ratio thresholds         │
                          │ report.py  HTML report · PR markdown · JSON · exit 0 / 1       │
                          │ server.py  FastAPI: /health /api/runs /api/compare /runs/{id}  │
                          └────────────────────────────────────────────────────────────────┘
 human-labelled pairs ──▶ calibrate.py  judge variants × both orders ─▶ agreement · kappa · bias
```

## Quick start

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/). Installs from GitHub (not published to PyPI):

```bash
uv tool install git+https://github.com/bprakashputta/assaygate
# or, to hack on it
git clone https://github.com/bprakashputta/assaygate && cd assaygate && uv sync
```

Re-gate the committed demo runs (no API key needed):

```bash
uv run assaygate compare examples/banking77/runs/baseline-v1 examples/banking77/runs/candidate-v2 --html gate.html
echo $?   # 1: regression caught
```

The whole stack in Docker: the report/gate service over the demo runs, and the demo classifier app.

```bash
cp .env.example .env            # add ANTHROPIC_API_KEY for the classifier
make data                       # downloads Banking77 + MT-Bench judgments into data/ (gitignored)
docker compose up --build -d
open http://localhost:8100      # runs, reports, /compare?baseline=baseline-v1&candidate=candidate-v2
BANKING77_URL=http://localhost:8200/classify uv run assaygate run examples/banking77/suite.yaml
```

### A suite

```yaml
name: support-intents
dataset: {path: data/test.jsonl}            # {"id": ..., "input": ..., "expected": ...} per line
target:
  type: http                                # or: {type: callable, ref: "myapp.pipeline:answer"}
  url: ${APP_URL:-http://localhost:8200/classify}
  body: {text: "{{input}}"}
  output_path: label
  cost_path: cost_usd                       # optional: the app reports its own model spend
scorers:
  - {type: exact, name: accuracy}
  - type: llm_rubric
    name: helpfulness
    rubric: "Answers the customer's actual question, correctly and concisely."
    llm: {model: claude-haiku-5-5}
gate:
  quality: [{metric: accuracy, max_drop: 0.0}, {metric: helpfulness, max_drop: 0.02}]
  cost: {max_increase: 0.25}                # fail if cost/item is confidently >25% higher
  latency: {p95_max_increase: 0.30}         # fail if p95 is confidently >30% higher
execution: {concurrency: 8, spend_cap_usd: 5.0}
```

```bash
assaygate run suite.yaml --run-id main            # runs/main/{records.jsonl,manifest.json,report.html}
assaygate run suite.yaml --run-id pr-42
assaygate compare runs/main runs/pr-42 --markdown comment.md --html gate.html   # exit 1 on regression
assaygate calibrate judge-calibration.yaml        # judge vs human labels
assaygate report runs/pr-42 --baseline runs/main  # HTML only
assaygate serve --runs-dir runs                   # FastAPI on :8100
```

From Python: `await assaygate.run_suite(load_suite("suite.yaml"))`, `assaygate.compare(base, cand)`; custom scorers register with `@assaygate.register("name")`.

### In CI

```yaml
- uses: bprakashputta/assaygate@main
  with: {baseline: baseline, suite: evals/suite.yaml}   # baseline = main's run, as an artifact
  env: {ANTHROPIC_API_KEY: "${{ secrets.ANTHROPIC_API_KEY }}"}
```

The action installs itself, runs the suite if no candidate run is given, gates, writes the job summary, and posts or updates one PR comment. [`examples/workflows/llm-gate.yml`](examples/workflows/llm-gate.yml) is a complete workflow (main uploads the baseline, PRs gate against it). This repository's own [`gate-demo.yml`](.github/workflows/gate-demo.yml) runs the action on the committed Banking77 runs and asserts that the re-run passes and the regression fails.

## Evaluation

Two experiments, both reproducible with one command, both run for real (every number below comes from the committed result files).

### Judge calibration on MT-Bench human judgments

**Data.** [`lmsys/mt_bench_human_judgments`](https://huggingface.co/datasets/lmsys/mt_bench_human_judgments) (CC BY 4.0), human split: 3,355 expert votes comparing six models' answers to the 80 MT-Bench questions, turns 1 and 2. Votes are grouped per pair (1,814 pairs); the label is the majority vote, and 214 pairs with no majority are dropped. The data is split **by question**, stratified by category: 24 questions for dev, 56 for test. Fixed seeded samples: **N = 400 test pairs** (168 A / 153 B / 79 tie; 205 first-turn, 195 second-turn) and 120 dev pairs. GPT-4 reference answers for math, reasoning and coding come from [FastChat](https://github.com/lm-sys/FastChat) (Apache-2.0). Everything prompt-related (wording, the swap policy, the 6 few-shot examples) was chosen on dev; test was run once with the final configuration.

**Judge.** `claude-haiku-5-5`, thinking disabled except where noted, structured outputs. Every variant judges every pair in **both** orders, so position bias is measured even for single-order judges.

`uv run python evals/mtbench/run.py --split test` · results: [`evals/results/mtbench_test.md`](evals/results/mtbench_test.md), [JSON](evals/results/mtbench_test.json), [HTML report](evals/results/mtbench_test.html)

<!-- CALIBRATION_TABLE -->

<!-- CALIBRATION_NOTES -->

### End to end: the gate catching a real regression

**System under test.** [`examples/banking77/app.py`](examples/banking77/app.py), a FastAPI intent classifier for [Banking77](https://huggingface.co/datasets/PolyAI/banking77) (CC BY 4.0, 77 intents): one Claude Haiku 5.5 call per message, the label constrained by a JSON-schema enum and re-checked in code. **Baseline (v1):** label list with two training examples per intent plus disambiguation notes. **Candidate (v2), the change under review:** examples and notes removed "to cut tokens", adaptive thinking switched on "to make up for it". Both versions were run against the live API on the same **300 test utterances** (seeded sample of the 3,080-utterance test split); the baseline was also re-run unchanged to check for false alarms. The runs are committed in [`examples/banking77/runs/`](examples/banking77/runs).

| comparison | accuracy | Δ accuracy (95% CI) | items lost / gained | cost per item | p95 latency | gate |
|---|---|---|---|---|---|---|
| v1 re-run vs v1 | 0.857 → 0.850 | −0.7 pt [−2.3, +1.0] | 4 / 2 (McNemar p = 0.69) | $0.000061 → $0.000061 | 1,148 → 1,051 ms | **pass** |
| v2 vs v1 | 0.857 → 0.797 | −6.0 pt [−10.0, −2.0] | 28 / 10 (McNemar p = 0.005) | $0.000061 → $0.000032 (x0.52) | 1,148 → 1,071 ms | **fail** |

The "simplified" prompt halves the cost per item, which is exactly why a reviewer might approve it, and quietly loses six points of accuracy. The gate fails it on quality alone; cost and latency pass. The same prompt change on the separate dev split (154 utterances drawn from the training set) showed the same direction: −9.1 pt [−14.3, −3.9]. The unchanged re-run moves accuracy by 0.7 points and passes: pairing items is what makes the interval narrow enough (an unpaired bootstrap of the same two runs gives [−6.3, +5.0], which would have hidden the regression entirely).

The three test runs cost $0.046 in total.

### Total spend

All development and evaluation runs for this project went through one ledger with a hard $5.00 cap: <!-- TOTAL_SPEND -->.

## Design decisions

- **Paired statistics, not point estimates** ([ADR 0002](docs/decisions/0002-paired-bootstrap-gate.md)). Items are paired by id; the gate fails only when the CI of the paired difference clears the allowed margin. Cost and latency are gated as ratios (mean $ per item, p95), so one threshold works for any scale. Errored items score 0: an outage is a regression.
- **Reservations make the spend cap hard** ([ADR 0003](docs/decisions/0003-spend-ledger-reservations.md)). Each call reserves its worst-case cost before it starts and settles to the real `usage` after, so concurrent calls cannot jointly overshoot.
- **Judge techniques are flags, measured one at a time** ([ADR 0004](docs/decisions/0004-judge-technique-ladder.md)). Rubric, reference answers, analysis-before-verdict, swap-and-combine and few-shot examples are switchable; prompts are versioned files under [`src/assaygate/prompts/`](src/assaygate/prompts) and each verdict records the files it used.
- **Direct Anthropic SDK** ([ADR 0001](docs/decisions/0001-anthropic-sdk-not-langchain.md)). The cache key and the cost ledger depend on the exact request and `usage` block; there is no agent graph to justify a framework. The system under test can be built with anything.
- **Reports are one HTML file** ([ADR 0005](docs/decisions/0005-report-html-css-charts.md)) with HTML/CSS charts that stay legible on a phone, light and dark themes, tooltips, and a table view under every chart.
- **Org-wide concurrency limits are real.** The first full calibration run died on a 429 "concurrent requests across all models" error from other workloads sharing the key. The client now caps in-flight calls and adds jittered backoff (honouring `Retry-After`) on top of the SDK's retries; truncated structured outputs are re-asked with a larger `max_tokens` rather than repeated, and a judge that still can't produce valid JSON abstains (and is counted) instead of crashing the run.

## Limitations

- The calibration covers one public dataset (MT-Bench: chat answers from 2023-era models, judged by a small pool of experts). Judge agreement on your domain needs your own labels; `assaygate calibrate` exists for that.
- MT-Bench pairs are labelled by majority vote, which is less noisy than any single expert. That is why a judge can match the majority more often than two individual humans match each other; it does not mean the judge is "better than humans".
- Small suites give wide intervals and therefore only catch large regressions. The report shows the interval so this is visible; use `non_inferiority` mode and more items when you need to prove "no worse than X".
- Latency in the demo is measured through the app's own rate limiter and a shared API key, so it includes queueing noise; the p95 ratio CI is wide for that reason.
- One provider (Anthropic) is wired into the judge client; the system under test can be anything.
- Not published to PyPI; install from GitHub.

## Development

```bash
make install lint test     # ruff + pytest; tests mock every model call (offline, free)
make data                  # public datasets -> data/ (gitignored)
make eval-judge-dev        # iterate on dev
make eval-judge            # the reported test numbers
make demo-gate             # re-gate the committed Banking77 runs
```

<!-- TEST_COUNT -->

## Credits

- MT-Bench human judgments: Zheng et al., *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena* (2023), [`lmsys/mt_bench_human_judgments`](https://huggingface.co/datasets/lmsys/mt_bench_human_judgments), CC BY 4.0. Questions and GPT-4 reference answers from [lm-sys/FastChat](https://github.com/lm-sys/FastChat), Apache-2.0.
- Banking77: Casanueva et al., *Efficient Intent Detection with Dual Sentence Encoders* (2020), [`PolyAI/banking77`](https://huggingface.co/datasets/PolyAI/banking77), CC BY 4.0. A 300-utterance sample appears in the committed demo run artifacts.

MIT licensed.
