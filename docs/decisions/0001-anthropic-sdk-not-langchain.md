# 0001. Call the Anthropic SDK directly, not through LangChain

Status: accepted · 2026-11-08

## Context
The house default is to reach models through LangChain where an agent graph is involved. assaygate has no agent graph: every model call is one structured-output request (a judge verdict, a rubric grade, a classification) whose exact request body matters, because

* the **response cache** keys on the canonical request; any hidden field a wrapper adds or reorders changes the key and silently disables caching;
* the **spend ledger** must price each call from the provider's `usage` block, including cache-read and cache-write tokens, which wrappers often drop or rename;
* structured outputs use `output_config.format` (JSON schema), and current models reject `temperature` and forced `tool_choice`; we need to control precisely what is sent.

## Decision
`assaygate.llm.LLMClient` wraps `anthropic.AsyncAnthropic().messages.create` directly. It owns caching, the spend cap, rate limiting, backoff, schema validation and telemetry. Scorers and judges depend only on `LLMClient.complete(...)`, so another provider can be added behind the same method without touching them.

## Consequences
* One dependency fewer and a request body we can test byte for byte (see `tests/test_llm.py`).
* Users who already standardise on LangChain can still evaluate LangChain apps: the system under test is any callable or HTTP endpoint; only the judges use `LLMClient`.
