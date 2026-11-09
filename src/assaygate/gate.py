"""Compare a candidate run with a baseline run and decide pass/fail.

Quality metrics use a **paired** analysis over the items both runs share: the per-item
difference ``candidate - baseline`` gets a bootstrap CI (resampling items) and a sign-flip
permutation p-value; for 0/1 metrics McNemar's exact test is reported too. Pairing matters:
the same 200 items scored twice have far less noise in the *difference* than two independent
means would suggest, so real regressions are caught with fewer items.

Decision rules (``max_drop`` = m, on the 0-1 metric scale):

* ``regression`` mode (default): FAIL when ``ci.high < -m`` — we are confident the candidate is
  worse by more than the allowed margin. Noise alone fails a gate ~2.5% of the time at 95%.
* ``non_inferiority`` mode: FAIL unless ``ci.low > -m`` — the candidate must *prove* it is not
  worse by more than m. Stricter; needs more items.

Cost and latency use paired bootstrap CIs on ratios (mean cost per item; p95 latency) and fail
only when the CI lower bound exceeds ``1 + max_increase``. Absolute ceilings are point checks.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

from . import stats
from .config import GateConfig
from .runner import Run

Status = Literal["pass", "fail", "warn"]


class Check(BaseModel):
    name: str
    kind: Literal["quality", "cost", "latency"]
    status: Status
    baseline: float
    candidate: float
    delta: float = Field(description="candidate - baseline (quality) or ratio (cost/latency)")
    ci_low: float
    ci_high: float
    p_value: float | None = None
    rule: str
    reason: str


class ItemChange(BaseModel):
    id: str
    metric: str
    baseline: float
    candidate: float
    baseline_output: str
    candidate_output: str
    expected: str | None = None


class GateResult(BaseModel):
    passed: bool
    baseline_run: str
    candidate_run: str
    n_paired: int
    n_baseline_only: int
    n_candidate_only: int
    level: float
    checks: list[Check]
    regressions: list[ItemChange] = Field(default_factory=list)
    improvements: list[ItemChange] = Field(default_factory=list)


def _neg(m: float) -> str:
    return f"-{m:g}" if m else "0"


def _short(x: object, n: int = 280) -> str:
    s = x if isinstance(x, str) else ("" if x is None else str(x))
    return s if len(s) <= n else s[: n - 1] + "…"


def compare(baseline: Run, candidate: Run, gate: GateConfig | None = None) -> GateResult:
    gate = gate or GateConfig.model_validate(candidate.manifest.gate or {})
    b_by = {r.id: r for r in baseline.records}
    c_by = {r.id: r for r in candidate.records}
    ids = sorted(set(b_by) & set(c_by))
    if len(ids) < 2:
        raise ValueError("fewer than 2 items in common between the runs; nothing to compare")
    checks: list[Check] = []
    kw = {"n_resamples": gate.n_resamples, "level": gate.level, "seed": gate.seed}

    # Gate every shared metric; metrics with an explicit MetricGate use its thresholds.
    shared = [m for m in candidate.manifest.metrics if m in baseline.manifest.metrics]
    configured = {g.metric: g for g in gate.quality}
    regressions: list[ItemChange] = []
    improvements: list[ItemChange] = []
    for metric in shared:
        b = np.array([b_by[i].scores[metric].value for i in ids])
        c = np.array([c_by[i].scores[metric].value for i in ids])
        ci = stats.paired_bootstrap_diff(b, c, **kw)
        p = stats.paired_permutation_test(b, c, seed=gate.seed)
        binary = set(np.unique(np.concatenate([b, c]))) <= {0.0, 1.0}
        g = configured.get(metric)
        m = g.max_drop if g else 0.0
        mode = g.mode if g else "regression"
        status: Status = "pass"
        if mode == "regression":
            rule = f"fail if CI high < {_neg(m)}"
            reason = "no significant drop"
            if ci.high < -m:
                status, reason = "fail", f"worse by {-ci.estimate:.3f} (CI excludes {_neg(m)})"
        else:
            rule = f"fail unless CI low > {_neg(m)}"
            reason = "non-inferior"
            if not ci.low > -m:
                status, reason = "fail", f"cannot rule out a drop of more than {m:g}"
        if g and g.min_value is not None and c.mean() < g.min_value:
            status, reason = "fail", f"mean {c.mean():.3f} below floor {g.min_value:g}"
        if binary:
            lost = int(np.sum((b == 1) & (c == 0)))
            won = int(np.sum((b == 0) & (c == 1)))
            p_mc = stats.mcnemar_exact(lost, won)
            reason += f"; {lost} items lost, {won} gained (McNemar p={p_mc:.3g})"
        checks.append(
            Check(
                name=metric,
                kind="quality",
                status=status,
                baseline=float(b.mean()),
                candidate=float(c.mean()),
                delta=ci.estimate,
                ci_low=ci.low,
                ci_high=ci.high,
                p_value=p,
                rule=rule,
                reason=reason,
            )
        )
        for i, bv, cv in zip(ids, b, c, strict=True):
            change = ItemChange(
                id=i,
                metric=metric,
                baseline=float(bv),
                candidate=float(cv),
                baseline_output=_short(b_by[i].output if b_by[i].error is None else b_by[i].error),
                candidate_output=_short(c_by[i].output if c_by[i].error is None else c_by[i].error),
                expected=_short(c_by[i].expected) if c_by[i].expected is not None else None,
            )
            if cv < bv:
                regressions.append(change)
            elif cv > bv:
                improvements.append(change)

    # Cost: mean $ per item of the system under test.
    if gate.cost is not None:
        b = np.array([b_by[i].cost_usd for i in ids])
        c = np.array([c_by[i].cost_usd for i in ids])
        if b.mean() > 0:
            ci = stats.paired_bootstrap_ratio(b, c, np.mean, **kw)
            status, reason = "pass", "within budget"
            rule = "report only"
            if gate.cost.max_increase is not None:
                lim = 1 + gate.cost.max_increase
                rule = f"fail if ratio CI low > {lim:.2f}"
                if ci.low > lim:
                    status, reason = "fail", f"cost/item x{ci.estimate:.2f} (CI low {ci.low:.2f})"
            if gate.cost.max_per_item_usd is not None and c.mean() > gate.cost.max_per_item_usd:
                status, reason = (
                    "fail",
                    f"${c.mean():.5f}/item over ${gate.cost.max_per_item_usd:g}",
                )
            checks.append(
                Check(
                    name="cost_per_item_usd",
                    kind="cost",
                    status=status,
                    baseline=float(b.mean()),
                    candidate=float(c.mean()),
                    delta=ci.estimate,
                    ci_low=ci.low,
                    ci_high=ci.high,
                    rule=rule,
                    reason=reason,
                )
            )

    # Latency: p95 of per-item latency.
    if gate.latency is not None:
        b = np.array([b_by[i].latency_ms for i in ids])
        c = np.array([c_by[i].latency_ms for i in ids])
        if np.percentile(b, 95) > 0:
            ci = stats.paired_bootstrap_ratio(b, c, stats.p95, **kw)
            status, reason, rule = "pass", "within budget", "report only"
            if gate.latency.p95_max_increase is not None:
                lim = 1 + gate.latency.p95_max_increase
                rule = f"fail if p95 ratio CI low > {lim:.2f}"
                if ci.low > lim:
                    status, reason = "fail", f"p95 x{ci.estimate:.2f} (CI low {ci.low:.2f})"
            cp95 = float(np.percentile(c, 95))
            if gate.latency.p95_max_ms is not None and cp95 > gate.latency.p95_max_ms:
                status, reason = "fail", f"p95 {cp95:.0f} ms over {gate.latency.p95_max_ms:g} ms"
            checks.append(
                Check(
                    name="latency_p95_ms",
                    kind="latency",
                    status=status,
                    baseline=float(np.percentile(b, 95)),
                    candidate=cp95,
                    delta=ci.estimate,
                    ci_low=ci.low,
                    ci_high=ci.high,
                    rule=rule,
                    reason=reason,
                )
            )

    regressions.sort(key=lambda x: (x.metric, x.candidate - x.baseline, x.id))
    improvements.sort(key=lambda x: (x.metric, x.baseline - x.candidate, x.id))
    return GateResult(
        passed=all(ch.status != "fail" for ch in checks),
        baseline_run=baseline.manifest.run_id,
        candidate_run=candidate.manifest.run_id,
        n_paired=len(ids),
        n_baseline_only=len(set(b_by) - set(c_by)),
        n_candidate_only=len(set(c_by) - set(b_by)),
        level=gate.level,
        checks=checks,
        regressions=regressions,
        improvements=improvements,
    )


def _fmt(check: Check, value: float) -> str:
    if check.kind == "cost":
        return f"${value:.5f}"
    if check.kind == "latency":
        return f"{value:,.0f} ms"
    return f"{value:.3f}"


def to_markdown(result: GateResult, *, max_rows: int = 10) -> str:
    """PR-comment summary."""
    verdict = "PASSED" if result.passed else "FAILED"
    lines = [
        f"### assaygate: {verdict}",
        "",
        f"`{result.candidate_run}` vs baseline `{result.baseline_run}` · "
        f"{result.n_paired} paired items · {int(result.level * 100)}% bootstrap CIs",
        "",
        "| check | baseline | candidate | change | CI | result |",
        "|---|---:|---:|---:|---|---|",
    ]
    for ch in result.checks:
        if ch.kind == "quality":
            change = f"{ch.delta:+.3f}"
            ci = f"[{ch.ci_low:+.3f}, {ch.ci_high:+.3f}]"
        else:
            change = f"x{ch.delta:.2f}"
            ci = f"[x{ch.ci_low:.2f}, x{ch.ci_high:.2f}]"
        mark = "**FAIL**" if ch.status == "fail" else "pass"
        lines.append(
            f"| {ch.name} | {_fmt(ch, ch.baseline)} | {_fmt(ch, ch.candidate)} | {change} | "
            f"{ci} | {mark}: {ch.reason} |"
        )
    if result.regressions:
        lines += ["", f"<details><summary>{len(result.regressions)} item-level regressions "
                  f"(first {min(max_rows, len(result.regressions))})</summary>", "",
                  "| item | metric | baseline output | candidate output | expected |",
                  "|---|---|---|---|---|"]  # fmt: skip
        for r in result.regressions[:max_rows]:
            cells = [r.baseline_output, r.candidate_output, r.expected or ""]
            cells = [c.replace("|", "\\|").replace("\n", " ")[:90] for c in cells]
            lines.append(f"| `{r.id}` | {r.metric} | {cells[0]} | {cells[1]} | {cells[2]} |")
        lines += ["", "</details>"]
    return "\n".join(lines) + "\n"
