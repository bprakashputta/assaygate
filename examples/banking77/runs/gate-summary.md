### assaygate: FAILED

`candidate-v2` vs baseline `baseline-v1` · 300 paired items · 95% bootstrap CIs

| check | baseline | candidate | change | CI | result |
|---|---:|---:|---:|---|---|
| accuracy | 0.857 | 0.797 | -0.060 | [-0.100, -0.020] | **FAIL**: worse by 0.060 (CI excludes 0); 28 items lost, 10 gained (McNemar p=0.0051) |
| cost_per_item_usd | $0.00006 | $0.00003 | x0.52 | [x0.51, x0.53] | pass: within budget |
| latency_p95_ms | 1,148 ms | 1,071 ms | x0.93 | [x0.86, x1.09] | pass: within budget |

<details><summary>28 item-level regressions (first 10)</summary>

| item | metric | baseline output | candidate output | expected |
|---|---|---|---|---|
| `test-0006` | accuracy | why_verify_identity | verify_my_identity | why_verify_identity |
| `test-0026` | accuracy | virtual_card_not_working | disposable_card_limits | virtual_card_not_working |
| `test-0030` | accuracy | exchange_via_app | fiat_currency_support | exchange_via_app |
| `test-0037` | accuracy | declined_cash_withdrawal | cash_withdrawal_not_recognised | declined_cash_withdrawal |
| `test-0043` | accuracy | order_physical_card | get_physical_card | order_physical_card |
| `test-0048` | accuracy | get_physical_card | change_pin | get_physical_card |
| `test-0054` | accuracy | transfer_into_account | transfer_timing | transfer_into_account |
| `test-0076` | accuracy | compromised_card | card_payment_not_recognised | compromised_card |
| `test-0086` | accuracy | getting_spare_card | order_physical_card | getting_spare_card |
| `test-0095` | accuracy | getting_spare_card | card_linking | getting_spare_card |

</details>
