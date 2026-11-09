# Judge calibration: MT-Bench human judgments (dev, N=120)

Judge `claude-haiku-5-5`. Human-human agreement on the same data: 64.2% with ties, 80.6% without (1621 vote pairs).

| variant | agreement (95% CI) | kappa (95% CI) | non-tie agreement | judge tie rate | swap consistency | picks longer (judge / human) | Δ vs naive (95% CI) | $/item |
|---|---|---|---|---|---|---|---|---|
| 1 naive | 67.5% [58.7%, 75.2%] | 0.480 [0.357, 0.598] | 85.4% (n=89) | 6.7% | 88.3% | 75.0% / 80.0% | – | $0.00026 |
| 2 +rubric | 65.8% [57.0%, 73.7%] | 0.452 [0.330, 0.567] | 82.2% (n=90) | 5.8% | 85.0% | 71.9% / 80.0% | -1.7 [-5.8, +2.5] | $0.00032 |
| 3 +cot | 74.2% [65.7%, 81.2%] | 0.592 [0.471, 0.705] | 86.5% (n=89) | 12.5% | 81.7% | 73.0% / 80.0% | +6.7 [-0.8, +14.2] | $0.00058 |
| 4 +reference | 74.2% [65.7%, 81.2%] | 0.590 [0.470, 0.702] | 85.7% (n=91) | 10.0% | 86.7% | 72.3% / 80.0% | +6.7 [+0.0, +14.2] | $0.00062 |
| 5 +swap | 77.5% [69.2%, 84.1%] | 0.651 [0.535, 0.758] | 87.2% (n=86) | 20.0% | 86.7% | 73.3% / 80.0% | +10.0 [+2.5, +18.3] | $0.00062 |
| 6 +few-shot | 79.2% [71.1%, 85.5%] | 0.679 [0.562, 0.782] | 92.8% (n=83) | 22.5% | 85.8% | 76.3% / 80.0% | +11.7 [+4.2, +19.2] | $0.00053 |
| 6a average swap | 75.0% [66.6%, 81.9%] | 0.609 [0.490, 0.717] | 90.7% (n=86) | 15.0% | 85.8% | 74.2% / 80.0% | +7.5 [+0.8, +14.2] | $0.00053 |
| 6b no swap | 77.5% [69.2%, 84.1%] | 0.645 [0.520, 0.752] | 88.9% (n=90) | 12.5% | 85.8% | 75.0% / 80.0% | +10.0 [+3.3, +17.5] | $0.00053 |
| 6c thinking, no CoT field | 76.7% [68.3%, 83.3%] | 0.641 [0.521, 0.753] | 91.5% (n=82) | 22.5% | 83.3% | 76.3% / 80.0% | +9.2 [+0.8, +17.5] | $0.00042 |

By category group (paired Δ agreement vs naive, 95% bootstrap CI):

| subset | variant | N | agreement | kappa | Δ vs naive |
|---|---|---|---|---|---|
| math, reasoning, coding | 1 naive | 52 | 55.8% | 0.331 | – |
| math, reasoning, coding | 2 +rubric | 52 | 53.8% | 0.305 | -1.9 [-9.6, +5.8] |
| math, reasoning, coding | 3 +cot | 52 | 75.0% | 0.622 | +19.2 [+7.7, +30.8] |
| math, reasoning, coding | 4 +reference | 52 | 76.9% | 0.652 | +21.2 [+9.6, +34.6] |
| math, reasoning, coding | 5 +swap | 52 | 82.7% | 0.738 | +26.9 [+15.4, +40.4] |
| math, reasoning, coding | 6 +few-shot | 52 | 82.7% | 0.738 | +26.9 [+15.4, +38.5] |
| math, reasoning, coding | 6a average swap | 52 | 75.0% | 0.629 | +19.2 [+9.6, +30.8] |
| math, reasoning, coding | 6b no swap | 52 | 78.8% | 0.683 | +23.1 [+11.5, +34.6] |
| math, reasoning, coding | 6c thinking, no CoT field | 52 | 76.9% | 0.650 | +21.2 [+7.7, +34.6] |
| writing, roleplay, extraction, stem, humanities | 1 naive | 68 | 76.5% | 0.590 | – |
| writing, roleplay, extraction, stem, humanities | 2 +rubric | 68 | 75.0% | 0.563 | -1.5 [-5.9, +2.9] |
| writing, roleplay, extraction, stem, humanities | 3 +cot | 68 | 73.5% | 0.539 | -2.9 [-11.8, +4.4] |
| writing, roleplay, extraction, stem, humanities | 4 +reference | 68 | 72.1% | 0.513 | -4.4 [-11.8, +2.9] |
| writing, roleplay, extraction, stem, humanities | 5 +swap | 68 | 73.5% | 0.561 | -2.9 [-13.2, +7.4] |
| writing, roleplay, extraction, stem, humanities | 6 +few-shot | 68 | 76.5% | 0.613 | +0.0 [-8.8, +8.8] |
| writing, roleplay, extraction, stem, humanities | 6a average swap | 68 | 75.0% | 0.585 | -1.5 [-10.3, +7.4] |
| writing, roleplay, extraction, stem, humanities | 6b no swap | 68 | 76.5% | 0.593 | +0.0 [-8.8, +7.4] |
| writing, roleplay, extraction, stem, humanities | 6c thinking, no CoT field | 68 | 76.5% | 0.608 | +0.0 [-8.8, +8.8] |

Human tie rate: 23.3%. Spend for this run: $0.0000 (2160 judge calls, 2160 served from cache).
