"""Comparing stochastic runs: ranks, Spearman correlation, rank tests and a bootstrap interval.

Self-implemented (R1) over ``scipy.special`` for the normal tail; scipy.stats is a TEST oracle only.

* :func:`rank_sum_test` -- Wilcoxon-Mann-Whitney, two-sided, normal approximation with the tie and
  continuity corrections: are two independent samples (final indicator values of two solvers over
  seeds) drawn from one distribution?
* :func:`signed_rank_test` -- Wilcoxon signed-rank on paired samples (two solvers on the same seeds),
  zero differences dropped, the same corrections.
* :func:`bootstrap_interval` -- the percentile interval of any statistic of one sample.
* :func:`cluster_bootstrap` -- a statistic over resampled CLUSTERS (whole seeds, whole cases), so paired
  quantities sharing a cluster carry its noise once; the ``(point, low, high)`` 95 % percentile triple.
* :func:`kendall_tau_b` -- Kendall's rank correlation, tau-b (ties corrected as scipy does).
* :func:`wilson` -- a binomial share and its Wilson score interval.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.special import ndtr

from optimi_lab.core import refuse

__all__ = [
    'RankTest',
    'bootstrap_interval',
    'cluster_bootstrap',
    'kendall_tau_b',
    'rank',
    'rank_sum_test',
    'signed_rank_test',
    'spearman',
    'wilson',
]


@dataclass(frozen=True, slots=True)
class RankTest:
    """A test's statistic and its two-sided p-value."""

    statistic: float
    p_value: float


def rank(values: np.ndarray, axis: int = 0) -> np.ndarray:
    """Ranks from 1 along *axis*, ties sharing their average rank."""
    values = np.moveaxis(np.asarray(values, dtype=float), axis, 0)
    out = np.empty_like(values)
    for index in np.ndindex(values.shape[1:]):
        column = values[(slice(None), *index)]
        order = np.argsort(column, kind='mergesort')
        ordered = column[order]
        starts = np.flatnonzero(np.r_[True, ordered[1:] != ordered[:-1]])
        ends = np.r_[starts[1:], len(ordered)]
        ranks = np.empty(len(ordered))
        for start, end in zip(starts, ends, strict=True):
            ranks[start:end] = 0.5 * (start + end + 1)
        out[(order, *index)] = ranks
    return np.moveaxis(out, 0, axis)


def spearman(values: np.ndarray) -> np.ndarray | float:
    """Spearman's rho between the columns of *values* (a float for two columns, else the matrix)."""
    ranks = rank(np.asarray(values, dtype=float), axis=0)
    with np.errstate(invalid='ignore', divide='ignore'):
        rho = np.corrcoef(ranks, rowvar=False)
    return float(rho[0, 1]) if rho.shape == (2, 2) else rho


def _tie_term(ranks_of: np.ndarray) -> float:
    _, counts = np.unique(ranks_of, return_counts=True)
    return float(np.sum(counts**3 - counts))


def _two_sided(z: float) -> float:
    return float(min(1.0, 2.0 * ndtr(-abs(z))))


def rank_sum_test(first: np.ndarray, second: np.ndarray) -> RankTest:
    """Mann-Whitney U of *first* and its two-sided asymptotic p-value (tie and continuity corrected)."""
    first, second = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    n1, n2 = len(first), len(second)
    if n1 < 1 or n2 < 1:
        refuse(f'a rank-sum test needs two non-empty samples, got {n1} and {n2}')
    pooled = np.concatenate([first, second])
    ranks = rank(pooled)
    u1 = ranks[:n1].sum() - n1 * (n1 + 1) / 2.0
    n = n1 + n2
    sigma = np.sqrt(n1 * n2 / 12.0 * ((n + 1) - _tie_term(pooled) / (n * (n - 1))))
    u = max(u1, n1 * n2 - u1)
    return RankTest(float(u1), _two_sided((u - n1 * n2 / 2.0 - 0.5) / sigma) if sigma > 0 else 1.0)


def signed_rank_test(first: np.ndarray, second: np.ndarray) -> RankTest:
    """Wilcoxon's T = min(R+, R-) of the paired differences, two-sided normal p-value (corrected)."""
    difference = np.asarray(second, dtype=float) - np.asarray(first, dtype=float)
    difference = difference[difference != 0.0]
    n = len(difference)
    if n < 1:
        refuse('a signed-rank test needs at least one non-zero paired difference')
    ranks = rank(np.abs(difference))
    plus = ranks[difference > 0].sum()
    minus = ranks[difference < 0].sum()
    mean = n * (n + 1) / 4.0
    sigma = np.sqrt(n * (n + 1) * (2 * n + 1) / 24.0 - _tie_term(np.abs(difference)) / 48.0)
    statistic = min(plus, minus)
    z = (statistic - mean + 0.5 * np.sign(mean - statistic)) / sigma if sigma > 0 else 0.0
    return RankTest(float(statistic), _two_sided(z))


def bootstrap_interval(
    sample: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    *,
    confidence: float,
    n_resamples: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """The percentile bootstrap interval of ``statistic(sample)`` at *confidence*."""
    if not 0.0 < confidence < 1.0:
        refuse(f'a confidence level lies strictly between 0 and 1, got {confidence}')
    sample = np.asarray(sample)
    draws = np.array([statistic(sample[rng.integers(0, len(sample), len(sample))]) for _ in range(n_resamples)])
    tails = 100.0 * np.array([(1.0 - confidence) / 2.0, (1.0 + confidence) / 2.0])
    low, high = np.percentile(draws, tails)
    return float(low), float(high)


def kendall_tau_b(first: object, second: object) -> float:
    """Kendall's tau-b of two paired samples: ``(C - D) / sqrt(untied_1 * untied_2)`` over all pairs.

    NaN when either sample is constant, shorter than two, or holds a NaN (scipy's ``propagate``).
    """
    a, b = np.asarray(first, dtype=float).ravel(), np.asarray(second, dtype=float).ravel()
    if a.size != b.size:
        refuse(f'tau-b pairs two samples of one length, got {a.size} and {b.size}')
    if a.size < 2 or np.isnan(a).any() or np.isnan(b).any():
        return float('nan')
    i, j = np.triu_indices(a.size, 1)
    da, db = np.sign(a[i] - a[j]), np.sign(b[i] - b[j])
    untied = float(np.count_nonzero(da)) * float(np.count_nonzero(db))
    return float(np.sum(da * db) / np.sqrt(untied)) if untied > 0 else float('nan')


def wilson(hits: int, n: int, z: float = 1.959964) -> tuple[float, float, float]:
    """``(p, low, high)``: the share *hits* / *n* and its Wilson score interval; ``(nan, 0, 1)`` for ``n = 0``."""
    if n < 0 or not 0 <= hits <= max(n, 0):
        refuse(f'a binomial share needs 0 <= hits <= n, got {hits} of {n}')
    if n == 0:
        return float('nan'), 0.0, 1.0
    p = hits / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return p, max(0.0, centre - half), min(1.0, centre + half)


def cluster_bootstrap(
    statistic: Callable[[np.ndarray], float], n: int, *, draws: int = 2000, seed: int
) -> tuple[float, float, float]:
    """``statistic`` of all *n* clusters, and its 95 % percentile interval over clusters resampled with replacement.

    ``statistic`` takes the INDICES of the chosen clusters (``arange(n)`` for the point estimate). Each
    draw is ``rng.integers(0, n, n)`` from ``default_rng(seed)``; non-finite draws are dropped, and when
    none remain the interval is NaN.
    """
    rng = np.random.default_rng(seed)
    point = statistic(np.arange(n))
    values = [statistic(rng.integers(0, n, n)) for _ in range(draws)]
    values = [v for v in values if np.isfinite(v)]
    if not values:
        return point, float('nan'), float('nan')
    return point, float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))
