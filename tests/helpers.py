"""Build synthetic runs for gate/report/server tests."""

from __future__ import annotations

import numpy as np

from assaygate.config import GateConfig
from assaygate.runner import Manifest, Record, Run
from assaygate.scorers import Score


def make_run(
    run_id: str,
    acc: np.ndarray,
    latency: np.ndarray,
    cost: np.ndarray,
    gate: GateConfig | None = None,
) -> Run:
    records = [
        Record(
            id=f"i{i:03d}",
            input=f"q{i}",
            expected="x",
            output="x" if a else "y",
            latency_ms=float(lat),
            cost_usd=float(c),
            scores={"accuracy": Score(value=float(a), passed=bool(a))},
        )
        for i, (a, lat, c) in enumerate(zip(acc, latency, cost, strict=True))
    ]
    return Run(
        manifest=Manifest(
            run_id=run_id,
            suite="t",
            created_at="2026-11-08T00:00:00+00:00",
            duration_s=1.0,
            n_items=len(records),
            n_errors=0,
            metrics=["accuracy"],
            config_sha256="0",
            dataset_sha256="0",
            gate=(gate or GateConfig()).model_dump(),
        ),
        records=records,
    )


def pair(n=300, base_acc=0.85, drop=0.0, lat_mult=1.0, cost_mult=1.0, seed=0, gate=None):
    rng = np.random.default_rng(seed)
    acc_b = (rng.random(n) < base_acc).astype(float)
    acc_c = acc_b.copy()
    if drop:
        idx = np.flatnonzero(acc_c == 1)
        flip = rng.choice(idx, int(drop * n), replace=False)
        acc_c[flip] = 0
    lat_b = rng.lognormal(6.5, 0.3, n)
    lat_c = lat_b * lat_mult * rng.lognormal(0, 0.05, n)
    cost_b = rng.uniform(4e-5, 6e-5, n)
    cost_c = cost_b * cost_mult
    return make_run("base", acc_b, lat_b, cost_b, gate), make_run(
        "cand", acc_c, lat_c, cost_c, gate
    )
