"""Statistics checked against known values: textbook examples, scipy and scikit-learn."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats as sps
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix as sk_confusion

from assaygate import stats


def test_norm_ppf_matches_scipy():
    for p in (1e-6, 0.001, 0.025, 0.2, 0.5, 0.8, 0.975, 0.999):
        assert stats._norm_ppf(p) == pytest.approx(sps.norm.ppf(p), abs=1e-8)


def test_cohen_kappa_textbook_example():
    # Wikipedia "Cohen's kappa" worked example: 50 items, a=20 b=5 c=10 d=15 -> kappa 0.4
    truth = ["yes"] * 25 + ["no"] * 25
    pred = ["yes"] * 20 + ["no"] * 5 + ["yes"] * 10 + ["no"] * 15
    assert stats.cohen_kappa(truth, pred) == pytest.approx(0.4)


def test_cohen_kappa_matches_sklearn_three_class():
    rng = np.random.default_rng(1)
    labels = ["A", "B", "tie"]
    truth = rng.choice(labels, 500, p=[0.4, 0.4, 0.2])
    pred = np.where(rng.random(500) < 0.6, truth, rng.choice(labels, 500))
    assert stats.cohen_kappa(truth, pred, labels) == pytest.approx(
        cohen_kappa_score(truth, pred, labels=labels)
    )
    np.testing.assert_array_equal(
        stats.confusion_matrix(truth, pred, labels), sk_confusion(truth, pred, labels=labels)
    )


def test_kappa_perfect_and_chance():
    assert stats.cohen_kappa(list("ABAB"), list("ABAB")) == 1.0
    assert stats.cohen_kappa(list("AAAA"), list("AAAA")) == 1.0  # degenerate p_e = 1


def test_kappa_ci_contains_estimate():
    truth = ["A"] * 40 + ["B"] * 40 + ["tie"] * 20
    pred = ["A"] * 30 + ["B"] * 10 + ["B"] * 35 + ["tie"] * 5 + ["tie"] * 12 + ["A"] * 8
    ci = stats.kappa_ci(truth, pred, ["A", "B", "tie"], seed=0)
    assert ci.low < ci.estimate < ci.high
    assert ci.estimate == pytest.approx(cohen_kappa_score(truth, pred))


def test_bootstrap_percentile_matches_scipy():
    rng = np.random.default_rng(42)
    x = rng.normal(10, 2, 200)
    ours = stats.bootstrap_ci(x, np.mean, n_resamples=20_000, seed=0)
    ref = sps.bootstrap((x,), np.mean, n_resamples=20_000, method="percentile", random_state=0)
    assert ours.low == pytest.approx(ref.confidence_interval.low, abs=0.03)
    assert ours.high == pytest.approx(ref.confidence_interval.high, abs=0.03)
    # and with the analytic normal interval for a mean (CLT): mean +/- 1.96 * s / sqrt(n)
    half = 1.96 * x.std(ddof=1) / math.sqrt(x.size)
    assert ours.low == pytest.approx(x.mean() - half, abs=0.04)
    assert ours.high == pytest.approx(x.mean() + half, abs=0.04)


def test_bootstrap_bca_matches_scipy_on_skewed_data():
    rng = np.random.default_rng(7)
    x = rng.exponential(1.0, 120)
    ours = stats.bootstrap_ci(x, np.mean, n_resamples=20_000, method="bca", seed=0)
    ref = sps.bootstrap((x,), np.mean, n_resamples=20_000, method="BCa", random_state=0)
    assert ours.low == pytest.approx(ref.confidence_interval.low, abs=0.02)
    assert ours.high == pytest.approx(ref.confidence_interval.high, abs=0.02)


def test_bootstrap_constant_sample_is_degenerate():
    ci = stats.bootstrap_ci([1.0, 1.0, 1.0])
    assert (ci.low, ci.estimate, ci.high) == (1.0, 1.0, 1.0)


def test_paired_diff_uses_pairing():
    # Same large item-to-item spread in both runs, tiny consistent improvement:
    # the paired CI must be narrow and exclude zero.
    rng = np.random.default_rng(3)
    base = rng.uniform(0, 1, 300)
    cand = base + 0.01
    ci = stats.paired_bootstrap_diff(base, cand)
    assert ci.estimate == pytest.approx(0.01)
    assert ci.low > 0
    # unpaired intuition would be +/- ~0.05 here
    assert ci.high - ci.low < 1e-9


def test_paired_ratio_ci_covers_truth():
    rng = np.random.default_rng(5)
    base = rng.lognormal(0, 0.5, 400)
    cand = base * 1.5 * rng.lognormal(0, 0.05, 400)
    ci = stats.paired_bootstrap_ratio(base, cand, np.mean)
    assert ci.low < 1.5 < ci.high + 0.02
    p95 = stats.paired_bootstrap_ratio(base, cand, stats.p95)
    assert 1.3 < p95.estimate < 1.7


def test_permutation_test_exact_small_example():
    # differences +1,+1,+1,+1 (n=4): exact two-sided p = 2/16 = 0.125
    assert stats.paired_permutation_test([0, 0, 0, 0], [1, 1, 1, 1]) == pytest.approx(0.125)
    # identical runs: p = 1
    assert stats.paired_permutation_test([1, 0, 1], [1, 0, 1]) == 1.0


def test_permutation_test_agrees_with_scipy():
    rng = np.random.default_rng(11)
    base = rng.normal(0, 1, 60)
    cand = base + rng.normal(0.3, 1, 60)
    ours = stats.paired_permutation_test(base, cand, n_resamples=40_000, seed=0)
    ref = sps.permutation_test(
        (cand - base,),
        np.mean,
        permutation_type="samples",
        n_resamples=40_000,
        alternative="two-sided",
        random_state=0,
    )
    assert ours == pytest.approx(ref.pvalue, abs=0.01)


def test_mcnemar_matches_binomtest():
    for b, c in ((10, 2), (25, 25), (0, 7), (3, 14)):
        expected = sps.binomtest(min(b, c), b + c, 0.5).pvalue if b + c else 1.0
        assert stats.mcnemar_exact(b, c) == pytest.approx(expected)


def test_wilson_interval_known_value():
    # 8 successes of 10 at 95%: (0.4902, 0.9433), e.g. statsmodels proportion_confint(wilson)
    ci = stats.wilson_interval(8, 10)
    assert ci.low == pytest.approx(0.4902, abs=1e-4)
    assert ci.high == pytest.approx(0.9433, abs=1e-4)


def test_percentile_matches_numpy():
    x = list(range(1, 101))
    assert stats.percentile(x, 95) == pytest.approx(np.percentile(x, 95))
