from __future__ import annotations

import numpy as np

from assaygate.config import CostGate, GateConfig, LatencyGate, MetricGate
from assaygate.gate import compare, to_markdown

from .helpers import make_run, pair


def checks(result):
    return {c.name: c for c in result.checks}


def test_identical_runs_pass():
    b, c = pair()
    r = compare(b, c)
    assert r.passed and r.n_paired == 300
    assert checks(r)["accuracy"].delta == 0
    assert not r.regressions


def test_accuracy_drop_fails_with_ci_below_zero():
    b, c = pair(drop=0.06)
    r = compare(b, c)
    acc = checks(r)["accuracy"]
    assert not r.passed and acc.status == "fail"
    assert acc.ci_high < 0 and acc.delta == -0.06
    assert len(r.regressions) == 18
    assert "McNemar" in acc.reason


def test_margin_tolerates_small_drop():
    gate = GateConfig(quality=[MetricGate(metric="accuracy", max_drop=0.05)])
    b, c = pair(drop=0.02, gate=gate)
    assert checks(compare(b, c, gate))["accuracy"].status == "pass"


def test_non_inferiority_needs_proof():
    gate = GateConfig(
        quality=[MetricGate(metric="accuracy", max_drop=0.001, mode="non_inferiority")]
    )
    b, c = pair(drop=0.01, gate=gate)
    assert checks(compare(b, c, gate))["accuracy"].status == "fail"


def test_noise_alone_rarely_fails():
    # 40 re-runs of an unchanged, non-deterministic system: each item has its own chance of
    # being right and both runs draw from it independently. A 95% two-sided CI should fire
    # the one-sided regression rule about 2.5% of the time.
    rng = np.random.default_rng(123)
    fails = 0
    for _ in range(40):
        p_item = rng.beta(4, 1, 200)
        acc_b = (rng.random(200) < p_item).astype(float)
        acc_c = (rng.random(200) < p_item).astype(float)
        lat = rng.lognormal(6, 0.3, 200)
        cost = np.full(200, 5e-5)
        r = compare(make_run("b", acc_b, lat, cost), make_run("c", acc_c, lat, cost))
        fails += checks(r)["accuracy"].status == "fail"
    assert fails <= 3


def test_cost_and_latency_gates():
    b, c = pair(cost_mult=2.0, lat_mult=1.8)
    r = checks(compare(b, c))
    assert r["cost_per_item_usd"].status == "fail" and r["cost_per_item_usd"].delta == 2.0
    assert r["latency_p95_ms"].status == "fail" and r["latency_p95_ms"].ci_low > 1.2

    gate = GateConfig(cost=CostGate(max_increase=1.5), latency=LatencyGate(p95_max_increase=1.0))
    r2 = checks(compare(b, c, gate))
    assert r2["cost_per_item_usd"].status == "pass" and r2["latency_p95_ms"].status == "pass"

    gate_abs = GateConfig(cost=CostGate(max_increase=None, max_per_item_usd=1e-5))
    assert checks(compare(b, c, gate_abs))["cost_per_item_usd"].status == "fail"


def test_unpaired_items_are_reported():
    b, c = pair()
    c.records = c.records[:-5]
    r = compare(b, c)
    assert r.n_paired == 295 and r.n_baseline_only == 5


def test_markdown_summary():
    b, c = pair(drop=0.06)
    md = to_markdown(compare(b, c))
    assert md.startswith("### assaygate: FAILED")
    assert "| accuracy |" in md and "**FAIL**" in md and "item-level regressions" in md
