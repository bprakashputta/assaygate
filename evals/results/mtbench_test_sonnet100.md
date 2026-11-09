# Judge calibration: MT-Bench human judgments (test, N=100)

Judge `claude-sonnet-5-5`. Human-human agreement on the same data: 64.2% with ties, 80.6% without (1621 vote pairs).

| variant | agreement (95% CI) | kappa (95% CI) | non-tie agreement | judge tie rate | swap consistency | picks longer (judge / human) | Δ vs naive (95% CI) | $/item |
|---|---|---|---|---|---|---|---|---|
| 1 naive | 71.0% [61.5%, 79.0%] | 0.515 [0.377, 0.646] | 85.0% (n=80) | 4.0% | 90.0% | 62.7% / 70.6% | – | $0.00549 |
| 6 +few-shot | 73.0% [63.6%, 80.7%] | 0.559 [0.427, 0.686] | 89.5% (n=76) | 10.0% | 97.0% | 66.0% / 70.6% | +2.0 [-2.0, +6.0] | $0.01083 |

By category group (paired Δ agreement vs naive, 95% bootstrap CI):

| subset | variant | N | agreement | kappa | Δ vs naive |
|---|---|---|---|---|---|
| math, reasoning, coding | 1 naive | 32 | 65.6% | 0.479 | – |
| math, reasoning, coding | 6 +few-shot | 32 | 68.8% | 0.527 | +3.1 [+0.0, +9.4] |
| writing, roleplay, extraction, stem, humanities | 1 naive | 68 | 73.5% | 0.529 | – |
| writing, roleplay, extraction, stem, humanities | 6 +few-shot | 68 | 75.0% | 0.571 | +1.5 [-2.9, +5.9] |

Human tie rate: 19.0%. Spend for this run: $1.6316 (400 judge calls, 0 served from cache).
