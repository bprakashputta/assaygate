# 0003. Enforce the spend cap with reservations in a shared ledger

Status: accepted · 2026-11-08

## Context
Evals fan out: a calibration run makes thousands of judge calls with 6+ in flight. Checking "spent < cap" before each call is racy: N concurrent calls all pass the check and together overshoot. The cap must also hold across processes and re-runs (development spend counts too).

## Decision
`SpendLedger` is an append-only JSONL file shared by every run that points at it. Before a call, the client **reserves** the worst case (prompt-size estimate at the input rate + `max_tokens` at the output rate) under an asyncio lock and refuses with `SpendCapExceeded` if settled + reserved + this would pass the cap. After the call the reservation is replaced by the real cost from `usage` (including cache read/write tokens) and a line is appended. Failed calls release their reservation.

## Evidence
`tests/test_llm.py::test_reservations_prevent_concurrent_overshoot` fires 20 concurrent calls at a cap that fits only a few; settled spend never exceeds the cap. The project's own development and evaluation ran under a $5.00 cap on this ledger.

## Consequences
Worst-case reservations are pessimistic, so near the cap some calls are refused that would have fit. That is the right direction for a hard budget.
