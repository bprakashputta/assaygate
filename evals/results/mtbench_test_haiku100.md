# Judge calibration: MT-Bench human judgments (test, N=100)

Judge `claude-haiku-5-5`. Human-human agreement on the same data: 64.2% with ties, 80.6% without (1621 vote pairs).

| variant | agreement (95% CI) | kappa (95% CI) | non-tie agreement | judge tie rate | swap consistency | picks longer (judge / human) | Δ vs naive (95% CI) | $/item |
|---|---|---|---|---|---|---|---|---|
| 1 naive | 73.0% [63.6%, 80.7%] | 0.549 [0.409, 0.688] | 85.2% (n=81) | 4.0% | 89.0% | 64.7% / 70.6% | – | $0.00027 |
| 6 +few-shot | 69.0% [59.4%, 77.2%] | 0.505 [0.370, 0.639] | 88.7% (n=71) | 16.0% | 88.0% | 66.7% / 70.6% | -4.0 [-10.0, +2.0] | $0.00053 |

By category group (paired Δ agreement vs naive, 95% bootstrap CI):

| subset | variant | N | agreement | kappa | Δ vs naive |
|---|---|---|---|---|---|
| math, reasoning, coding | 1 naive | 32 | 68.8% | 0.526 | – |
| math, reasoning, coding | 6 +few-shot | 32 | 75.0% | 0.622 | +6.2 [+0.0, +15.6] |
| writing, roleplay, extraction, stem, humanities | 1 naive | 68 | 75.0% | 0.556 | – |
| writing, roleplay, extraction, stem, humanities | 6 +few-shot | 68 | 66.2% | 0.444 | -8.8 [-17.6, -1.5] |

Human tie rate: 19.0%. Spend for this run: $0.0000 (400 judge calls, 400 served from cache).
