# Judge calibration: MT-Bench human judgments (test, N=400)

Judge `claude-haiku-5-5`. Human-human agreement on the same data: 64.2% with ties, 80.6% without (1621 vote pairs).

| variant | agreement (95% CI) | kappa (95% CI) | non-tie agreement | judge tie rate | swap consistency | picks longer (judge / human) | Δ vs naive (95% CI) | $/item |
|---|---|---|---|---|---|---|---|---|
| 1 naive | 72.8% [68.2%, 76.9%] | 0.551 [0.485, 0.618] | 86.5% (n=318) | 4.8% | 89.0% | 68.9% / 74.0% | – | $0.00028 |
| 2 +rubric | 72.2% [67.7%, 76.4%] | 0.542 [0.475, 0.608] | 86.2% (n=318) | 4.5% | 89.0% | 69.9% / 74.0% | -0.5 [-2.8, +1.8] | $0.00035 |
| 3 +cot | 73.8% [69.2%, 77.8%] | 0.567 [0.502, 0.635] | 88.3% (n=316) | 5.2% | 89.8% | 70.7% / 74.0% | +1.0 [-1.8, +3.8] | $0.00061 |
| 4 +reference | 74.5% [70.0%, 78.5%] | 0.581 [0.515, 0.644] | 89.5% (n=313) | 6.5% | 87.2% | 72.1% / 74.0% | +1.8 [-1.3, +4.8] | $0.00065 |
| 5 +swap | 75.2% [70.8%, 79.2%] | 0.607 [0.543, 0.668] | 91.4% (n=291) | 16.2% | 87.2% | 74.1% / 74.0% | +2.5 [-1.2, +6.2] | $0.00065 |
| 6 +few-shot | 75.2% [70.8%, 79.2%] | 0.606 [0.543, 0.667] | 90.1% (n=294) | 15.8% | 88.5% | 73.1% / 74.0% | +2.5 [-1.2, +6.2] | $0.00053 |
| 6a average swap | 73.5% [69.0%, 77.6%] | 0.570 [0.506, 0.633] | 89.8% (n=304) | 9.5% | 88.5% | 73.1% / 74.0% | +0.8 [-2.5, +4.2] | $0.00053 |
| 6b no swap | 76.0% [71.6%, 79.9%] | 0.608 [0.544, 0.670] | 88.6% (n=315) | 7.8% | 88.5% | 70.6% / 74.0% | +3.2 [+0.2, +6.5] | $0.00053 |
| 6c thinking, no CoT field | 74.0% [69.5%, 78.1%] | 0.586 [0.522, 0.651] | 91.0% (n=290) | 15.8% | 89.0% | 73.8% / 74.0% | +1.2 [-2.0, +4.8] | $0.00043 |

By category group (paired Δ agreement vs naive, 95% bootstrap CI):

| subset | variant | N | agreement | kappa | Δ vs naive |
|---|---|---|---|---|---|
| math, reasoning, coding | 1 naive | 140 | 64.3% | 0.450 | – |
| math, reasoning, coding | 2 +rubric | 140 | 66.4% | 0.481 | +2.1 [-2.1, +6.4] |
| math, reasoning, coding | 3 +cot | 140 | 71.4% | 0.558 | +7.1 [+1.4, +12.9] |
| math, reasoning, coding | 4 +reference | 140 | 73.6% | 0.592 | +9.3 [+2.9, +15.7] |
| math, reasoning, coding | 5 +swap | 140 | 79.3% | 0.684 | +15.0 [+7.1, +22.9] |
| math, reasoning, coding | 6 +few-shot | 140 | 76.4% | 0.640 | +12.1 [+5.0, +19.3] |
| math, reasoning, coding | 6a average swap | 140 | 73.6% | 0.592 | +9.3 [+3.6, +15.7] |
| math, reasoning, coding | 6b no swap | 140 | 77.1% | 0.648 | +12.9 [+7.1, +19.3] |
| math, reasoning, coding | 6c thinking, no CoT field | 140 | 72.9% | 0.586 | +8.6 [+2.1, +15.0] |
| writing, roleplay, extraction, stem, humanities | 1 naive | 260 | 77.3% | 0.608 | – |
| writing, roleplay, extraction, stem, humanities | 2 +rubric | 260 | 75.4% | 0.574 | -1.9 [-4.6, +0.4] |
| writing, roleplay, extraction, stem, humanities | 3 +cot | 260 | 75.0% | 0.567 | -2.3 [-5.0, +0.4] |
| writing, roleplay, extraction, stem, humanities | 4 +reference | 260 | 75.0% | 0.569 | -2.3 [-5.4, +0.8] |
| writing, roleplay, extraction, stem, humanities | 5 +swap | 260 | 73.1% | 0.554 | -4.2 [-8.1, -0.4] |
| writing, roleplay, extraction, stem, humanities | 6 +few-shot | 260 | 74.6% | 0.583 | -2.7 [-6.9, +1.5] |
| writing, roleplay, extraction, stem, humanities | 6a average swap | 260 | 73.5% | 0.557 | -3.8 [-7.7, +0.0] |
| writing, roleplay, extraction, stem, humanities | 6b no swap | 260 | 75.4% | 0.579 | -1.9 [-5.4, +1.2] |
| writing, roleplay, extraction, stem, humanities | 6c thinking, no CoT field | 260 | 74.6% | 0.581 | -2.7 [-6.5, +1.2] |

Human tie rate: 19.8%. Spend for this run: $0.0000 (7200 judge calls, 7200 served from cache).
