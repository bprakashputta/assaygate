# 0004. Build the judge as switchable techniques and measure each one

Status: accepted · 2026-11-08

## Context
"Use an LLM as judge" is not one design. Rubrics, reference answers, chain-of-thought, order swapping and few-shot calibration each cost tokens, and their benefit depends on the model and the data. Claims like "judge agreement went from X to Y" are only meaningful with the steps measured separately on human labels the prompts were not tuned on.

## Decision
* `PairwiseJudgeConfig` exposes each technique as a flag; prompts are versioned files in `src/assaygate/prompts/`, and the judge records which files a verdict used.
* `assaygate calibrate` evaluates a list of variants against human labels and reports agreement (Wilson CI), Cohen's kappa (bootstrap CI), non-tie agreement, the confusion matrix, single-call position consistency, first-position win rate, verbosity bias against the humans' own rate, and a paired comparison with the first variant.
* Every variant is judged in **both** orders regardless of its policy, so position bias is measured even for single-order judges.
* MT-Bench is split **by question** (24 dev / 56 test questions, stratified by category). Prompt wording, the swap policy and the few-shot examples were chosen on dev only; test was run once with the final configuration.
* Swap policy: on dev, mapping order-inconsistent verdicts to **tie** ("conservative") beat averaging the two verdicts by 4.2 points, and brought the judge's tie rate (22.5%) to the humans' (23.3%). Adopted.

## Consequences
The calibration costs about 2× a single-order run because of the second order; the cache makes re-runs free. The report is honest about which steps did not help (on this data a rubric alone did not move agreement).

## Outcome on test (N = 400, run once)
The final judge improved agreement by +2.5 points over the naive judge (CI [−1.2, +6.2]); kappa 0.551 → 0.606. Dev had suggested +11.7. The gain is concentrated where answers are checkable: +12.1 points on math/reasoning/coding (N = 140, CI [+5.0, +19.3]), −2.7 on open-ended categories (N = 260, CI [−6.9, +1.5]). The same split shows the same direction on dev. We keep the full judge as the default for verifiable tasks and recommend a lighter configuration for open-ended writing.
