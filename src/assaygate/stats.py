"""Statistics used by the gate and the judge calibration.

Everything here is plain numpy so results are reproducible from a seed and the module has
no heavy dependencies. Each function is checked in ``tests/test_stats.py`` against known
values (textbook examples, scipy, scikit-learn).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import product

import numpy as np

Statistic = Callable[[np.ndarray], float]


@dataclass(frozen=True)
class Interval:
    """A point estimate with a two-sided confidence interval."""

    estimate: float
    low: float
    high: float
    level: float = 0.95

    def contains(self, value: float) -> bool:
        return self.low <= value <= self.high

    def as_dict(self) -> dict[str, float]:
        return {"estimate": self.estimate, "low": self.low, "high": self.high, "level": self.level}


def _rng(seed: int | None) -> np.random.Generator:
    return np.random.default_rng(seed)


def _as_array(values: Sequence[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError("expected a 1-D sequence")
    if arr.size == 0:
        raise ValueError("need at least one value")
    return arr


def _norm_ppf(p: float) -> float:
    """Inverse standard normal CDF (Acklam's rational approximation, |err| < 1.2e-9)."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    a = (-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00)  # fmt: skip
    b = (-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01)  # fmt: skip
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00)  # fmt: skip
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00)  # fmt: skip
    lo, hi = 0.02425, 1 - 0.02425
    if p < lo:
        q = math.sqrt(-2 * math.log(p))
        num = ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
        return num / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > hi:
        q = math.sqrt(-2 * math.log(1 - p))
        num = ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
        return -num / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    num = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
    return num / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# --------------------------------------------------------------------------- bootstrap


def bootstrap_ci(
    values: Sequence[float] | np.ndarray,
    statistic: Statistic = np.mean,
    *,
    n_resamples: int = 10_000,
    level: float = 0.95,
    method: str = "percentile",
    seed: int | None = 0,
) -> Interval:
    """Nonparametric bootstrap CI for ``statistic`` of one sample.

    ``method`` is ``"percentile"`` or ``"bca"`` (bias-corrected and accelerated, Efron 1987).
    Resampling is vectorised: one ``(n_resamples, n)`` index matrix, one pass of the statistic
    along axis 1 where the statistic supports ``axis=`` (numpy reductions do).
    """
    x = _as_array(values)
    n = x.size
    est = float(statistic(x))
    if n == 1 or np.all(x == x[0]):
        return Interval(est, est, est, level)
    idx = _rng(seed).integers(0, n, size=(n_resamples, n))
    boot = _apply_rows(statistic, x[idx])
    alpha = 1 - level
    if method == "percentile":
        lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
        return Interval(est, float(lo), float(hi), level)
    if method != "bca":
        raise ValueError(f"unknown bootstrap method {method!r}")
    # bias correction
    prop_less = (np.sum(boot < est) + 0.5 * np.sum(boot == est)) / n_resamples
    prop_less = min(max(prop_less, 1 / n_resamples), 1 - 1 / n_resamples)
    z0 = _norm_ppf(prop_less)
    # acceleration from the jackknife
    jack = np.array([statistic(np.delete(x, i)) for i in range(n)], dtype=float)
    diff = jack.mean() - jack
    denom = 6.0 * (np.sum(diff**2) ** 1.5)
    accel = float(np.sum(diff**3) / denom) if denom > 0 else 0.0

    def adjusted(q: float) -> float:
        z = _norm_ppf(q)
        return _norm_cdf(z0 + (z0 + z) / (1 - accel * (z0 + z)))

    lo, hi = np.quantile(boot, [adjusted(alpha / 2), adjusted(1 - alpha / 2)])
    return Interval(est, float(lo), float(hi), level)


def _apply_rows(statistic: Statistic, samples: np.ndarray) -> np.ndarray:
    try:
        out = np.asarray(statistic(samples, axis=1), dtype=float)  # type: ignore[call-arg]
        if out.shape == (samples.shape[0],):
            return out
    except TypeError:
        pass
    return np.array([statistic(row) for row in samples], dtype=float)


def paired_bootstrap_diff(
    baseline: Sequence[float] | np.ndarray,
    candidate: Sequence[float] | np.ndarray,
    *,
    n_resamples: int = 10_000,
    level: float = 0.95,
    seed: int | None = 0,
) -> Interval:
    """CI for ``mean(candidate) - mean(baseline)`` resampling *items* (keeps the pairing)."""
    b, c = _paired(baseline, candidate)
    return bootstrap_ci(c - b, np.mean, n_resamples=n_resamples, level=level, seed=seed)


def paired_bootstrap_ratio(
    baseline: Sequence[float] | np.ndarray,
    candidate: Sequence[float] | np.ndarray,
    statistic: Statistic = np.mean,
    *,
    n_resamples: int = 10_000,
    level: float = 0.95,
    seed: int | None = 0,
) -> Interval:
    """CI for ``statistic(candidate) / statistic(baseline)`` with paired item resampling.

    Used for cost (ratio of means) and latency (ratio of p95s): a ratio is scale-free, so a
    threshold like "no more than 20% slower" reads the same for a 40 ms and a 4 s service.
    """
    b, c = _paired(baseline, candidate)
    base_stat = float(statistic(b))
    if base_stat == 0:
        raise ValueError("baseline statistic is zero; ratio undefined")
    est = float(statistic(c)) / base_stat
    n = b.size
    idx = _rng(seed).integers(0, n, size=(n_resamples, n))
    num = _apply_rows(statistic, c[idx])
    den = _apply_rows(statistic, b[idx])
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = np.where(den > 0, num / den, np.nan)
    ratios = ratios[np.isfinite(ratios)]
    alpha = 1 - level
    lo, hi = np.quantile(ratios, [alpha / 2, 1 - alpha / 2])
    return Interval(est, float(lo), float(hi), level)


def _paired(a: Sequence[float] | np.ndarray, b: Sequence[float] | np.ndarray):
    x, y = _as_array(a), _as_array(b)
    if x.shape != y.shape:
        raise ValueError(f"paired samples differ in length: {x.size} vs {y.size}")
    return x, y


# --------------------------------------------------------------------------- tests


def paired_permutation_test(
    baseline: Sequence[float] | np.ndarray,
    candidate: Sequence[float] | np.ndarray,
    *,
    n_resamples: int = 20_000,
    seed: int | None = 0,
    exact_max_n: int = 16,
) -> float:
    """Two-sided sign-flip permutation test on paired differences (H0: mean diff = 0).

    Exact (all 2^m sign patterns over the m non-zero differences) when m <= ``exact_max_n``,
    Monte Carlo otherwise with the +1 correction so the p-value is never 0.
    """
    b, c = _paired(baseline, candidate)
    d = c - b
    d = d[d != 0]
    m = d.size
    if m == 0:
        return 1.0
    observed = abs(d.mean())
    tol = 1e-12
    if m <= exact_max_n:
        signs = np.array(list(product((-1.0, 1.0), repeat=m)))
        stats = np.abs((signs * d).mean(axis=1))
        return float(np.mean(stats >= observed - tol))
    signs = _rng(seed).choice((-1.0, 1.0), size=(n_resamples, m))
    stats = np.abs((signs * d).mean(axis=1))
    return float((np.sum(stats >= observed - tol) + 1) / (n_resamples + 1))


def binom_two_sided(k: int, n: int, p: float = 0.5) -> float:
    """Exact two-sided binomial test p-value (sum of outcomes no more likely than k)."""
    if n == 0:
        return 1.0
    pmf = np.array([math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(n + 1)])
    return float(min(1.0, pmf[pmf <= pmf[k] * (1 + 1e-7)].sum()))


def mcnemar_exact(b: int, c: int) -> float:
    """Exact McNemar test for paired binary outcomes.

    ``b`` = items that passed in baseline and failed in candidate, ``c`` = the reverse.
    """
    return binom_two_sided(min(b, c), b + c, 0.5)


def wilson_interval(successes: int, n: int, level: float = 0.95) -> Interval:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return Interval(float("nan"), 0.0, 1.0, level)
    z = _norm_ppf(1 - (1 - level) / 2)
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return Interval(p, max(0.0, centre - half), min(1.0, centre + half), level)


# --------------------------------------------------------------------------- agreement


def confusion_matrix(
    truth: Sequence[str], pred: Sequence[str], labels: Sequence[str]
) -> np.ndarray:
    """Rows = truth, columns = prediction, in ``labels`` order."""
    if len(truth) != len(pred):
        raise ValueError("truth and pred differ in length")
    pos = {lab: i for i, lab in enumerate(labels)}
    mat = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(truth, pred, strict=True):
        mat[pos[t], pos[p]] += 1
    return mat


def cohen_kappa(
    truth: Sequence[str], pred: Sequence[str], labels: Sequence[str] | None = None
) -> float:
    """Cohen's kappa: (p_o - p_e) / (1 - p_e) from the confusion matrix."""
    labs = list(labels) if labels is not None else sorted(set(truth) | set(pred))
    mat = confusion_matrix(truth, pred, labs).astype(float)
    n = mat.sum()
    if n == 0:
        return float("nan")
    p_o = np.trace(mat) / n
    p_e = float((mat.sum(axis=1) * mat.sum(axis=0)).sum() / n**2)
    if p_e == 1.0:
        return 1.0 if p_o == 1.0 else 0.0
    return float((p_o - p_e) / (1 - p_e))


def kappa_ci(
    truth: Sequence[str],
    pred: Sequence[str],
    labels: Sequence[str],
    *,
    n_resamples: int = 2_000,
    level: float = 0.95,
    seed: int | None = 0,
) -> Interval:
    """Bootstrap percentile CI for Cohen's kappa (resampling items)."""
    t = np.asarray(truth)
    p = np.asarray(pred)
    est = cohen_kappa(t, p, labels)
    n = t.size
    idx = _rng(seed).integers(0, n, size=(n_resamples, n))
    boots = np.array([cohen_kappa(t[i], p[i], labels) for i in idx])
    boots = boots[np.isfinite(boots)]
    alpha = 1 - level
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return Interval(est, float(lo), float(hi), level)


def percentile(values: Sequence[float] | np.ndarray, q: float) -> float:
    """Linear-interpolated percentile, ``q`` in [0, 100] (numpy's default method)."""
    return float(np.percentile(_as_array(values), q))


def p95(values: np.ndarray, axis: int | None = None) -> np.ndarray | float:
    """95th percentile usable as a bootstrap statistic (supports ``axis=``)."""
    return np.percentile(values, 95, axis=axis)
