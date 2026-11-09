# 0002. Gate on paired bootstrap CIs, not on point estimates

Status: accepted · 2026-11-08

## Context
A naive gate ("fail if accuracy dropped") fires on noise: LLM apps are non-deterministic, and on 300 items a re-run of the *same* prompt moves accuracy by a point or two. A gate that cries wolf gets disabled. A gate with a fixed tolerance ("allow −2 points") misses real regressions on small suites and over-reacts on large ones.

## Decision
* Pair items by id and analyse the per-item difference `candidate − baseline`. Pairing removes item-difficulty variance; on the Banking77 demo the paired 95% CI for an unchanged re-run is [−2.3, +1.0] points, where an unpaired bootstrap of the same two runs gives [−6.3, +5.0]: a 3.4x narrower interval from the same 300 items.
* Quality: bootstrap CI over items (10,000 resamples, seeded), sign-flip permutation p-value, McNemar exact test for 0/1 metrics. **Fail when the CI's upper bound is below −max_drop** ("we are confident it got worse by more than the margin"). An opt-in `non_inferiority` mode instead requires the lower bound to clear −max_drop.
* Cost and latency: paired bootstrap CI on the **ratio** (mean cost/item, p95 latency), failing when the CI's lower bound exceeds `1 + max_increase`. Ratios make one threshold work for 40 ms and 4 s services. Absolute ceilings are available as point checks.
* Errored items score 0 on every metric (an outage is a regression, not missing data).

## Evidence
* Unit test: 40 simulated re-runs of an unchanged non-deterministic system → at most 3 false alarms (expected ≈ 1 at a one-sided 2.5% rate).
* Demo: the unchanged-prompt re-run passes (−0.7 pt, CI [−2.3, +1.0]); the simplified prompt fails (−6.0 pt, CI [−10.0, −2.0], McNemar p = 0.005).

## Consequences
Small suites produce wide CIs and therefore only catch large regressions; the report shows the CI so that is visible rather than hidden. Teams that need to prove "no worse than X" use `non_inferiority` and more items.
