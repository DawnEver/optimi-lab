"""Robust choice from a fixed candidate pool: thin the pool, bound the scenarios, choose, validate.

* :func:`thin_candidates` -- the nondominated rows of a minimized table, thinned by farthest-point
  sampling in their min-max scaled frame, starting at the least first objective among feasible rows.
* :func:`box_draw` / :func:`box_corners` -- uniform scenarios on an axis-aligned box from a seed, and
  chosen vertices of it. One draw shared by every candidate is COMMON RANDOM NUMBERS: two candidates
  then differ by design, not by draw.
* :func:`choose` -- three formulations over a ``(design, scenario)`` table of scores (minimized) and
  violations (0 when every bound holds): ``DET`` (feasible at nominal, least nominal score), ``WC``
  (feasible in every selection scenario, least worst score), ``CC`` (at most ``floor((1 - level) n)``
  train misses, least mean train score). A formulation no design satisfies returns its least
  violating design marked infeasible -- never a silently relaxed rule.
* :func:`robust_validation` -- one design on an independent draw: score mean and 95th percentile and
  a floor quantity's 5th percentile with bootstrap intervals, P(violation) with a Wilson interval.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from optimi_lab.core.errors import refuse
from optimi_lab.core.pareto import non_dominated_sorting
from optimi_lab.metric.statistics import cluster_bootstrap, wilson

__all__ = ['FORMULATIONS', 'Choice', 'box_corners', 'box_draw', 'choose', 'robust_validation', 'thin_candidates']

#: The robust formulations :func:`choose` judges, in report order.
FORMULATIONS: tuple[str, ...] = ('DET', 'WC', 'CC')


@dataclass(frozen=True, slots=True)
class Choice:
    """A formulation's design, whether it satisfies its own rule, and its rule score."""

    design: str
    feasible: bool
    score: float


def thin_candidates(f: np.ndarray, k: int, *, feasible: np.ndarray) -> list[int]:
    """*k* row indices of *f* ``(n, m)``, minimized: nondominated, then spread (see the module docstring)."""
    f, feasible = np.asarray(f, dtype=float), np.asarray(feasible, dtype=bool)
    front = np.sort(np.asarray(non_dominated_sorting(f)).ravel())
    if len(front) < k:
        refuse(f'the front has {len(front)} row(s); thinning asks for {k}')
    span = np.ptp(f[front], axis=0)
    scaled = (f[front] - f[front].min(axis=0)) / np.where(span > 0, span, 1.0)
    allowed = [(i, f[j, 0]) for i, j in enumerate(front) if feasible[j]]
    start = min(allowed, key=lambda item: item[1])[0] if allowed else int(np.argmin(f[front, 0]))
    chosen = [start]
    distance = np.linalg.norm(scaled - scaled[start], axis=1)
    while len(chosen) < k:
        nxt = int(np.argmax(distance))
        chosen.append(nxt)
        distance = np.minimum(distance, np.linalg.norm(scaled - scaled[nxt], axis=1))
    return [int(front[i]) for i in chosen]


def box_draw(n: int, lower: np.ndarray, upper: np.ndarray, *, seed: int) -> np.ndarray:
    """*n* uniform rows on ``[lower, upper]``: ``lower + u (upper - lower)``, ``u = default_rng(seed).random``."""
    lower, upper = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    return lower + np.random.default_rng(seed).random((n, lower.size)) * (upper - lower)


def box_corners(lower: np.ndarray, upper: np.ndarray, picks: Sequence[Sequence[bool]]) -> np.ndarray:
    """One vertex per entry of *picks*: per axis the upper bound where True, the lower where False."""
    return np.where(np.asarray(picks, dtype=bool), np.asarray(upper, dtype=float), np.asarray(lower, dtype=float))


def choose(
    designs: Sequence[str],
    score: np.ndarray,
    violation: np.ndarray,
    *,
    nominal: int,
    corners: Sequence[int],
    train: Sequence[int],
    chance_level: float,
) -> dict[str, Choice]:
    """Each of :data:`FORMULATIONS` -> its :class:`Choice`; ``score``/``violation`` are ``(design, scenario)``.

    Ties break on the lower score, then on the design name.
    """
    score, violation = np.asarray(score, dtype=float), np.asarray(violation, dtype=float)
    if not len(designs) or score.shape != violation.shape or score.shape[0] != len(designs) or not train:
        refuse('choose needs one score and violation row per design and at least one train scenario')
    needed = [nominal, *corners, *train]
    allowed = int(np.floor((1.0 - chance_level) * len(train) + 1e-9))

    def det(d: int) -> tuple[bool, float, float]:
        return violation[d, nominal] == 0.0, float(score[d, nominal]), float(violation[d, nominal])

    def worst(d: int) -> tuple[bool, float, float]:
        v = float(violation[d, needed].max())
        return v == 0.0, float(score[d, needed].max()), v

    def chance(d: int) -> tuple[bool, float, float]:
        misses = int(np.sum(violation[d, train] > 0.0))
        return misses <= allowed, float(np.mean(score[d, train])), misses / len(train)

    rules: dict[str, Callable[[int], tuple[bool, float, float]]] = {'DET': det, 'WC': worst, 'CC': chance}
    out = {}
    for name in FORMULATIONS:
        judged = {d: rules[name](d) for d in range(len(designs))}
        feasible = [d for d, j in judged.items() if j[0]]
        if feasible:
            best = min(feasible, key=lambda d: (judged[d][1], designs[d]))
        else:
            best = min(judged, key=lambda d: (judged[d][2], judged[d][1], designs[d]))
        out[name] = Choice(designs[best], bool(feasible), judged[best][1])
    return out


def _interval(values: np.ndarray, statistic: Callable[[np.ndarray], float], draws: int, seed: int):
    return cluster_bootstrap(lambda idx: float(statistic(values[idx])), len(values), draws=draws, seed=seed)


def robust_validation(
    score: np.ndarray, floor: np.ndarray, violated: np.ndarray, *, draws: int = 2000, seed: int
) -> dict[str, object]:
    """``n``, ``score_mean``, ``score_p95``, ``floor_p5`` (each ``(point, low, high)``), ``floor_min``, ``violation``."""
    score, floor = np.asarray(score, dtype=float), np.asarray(floor, dtype=float)
    violated = np.asarray(violated, dtype=bool)
    return {
        'n': len(score),
        'score_mean': _interval(score, np.mean, draws, seed),
        'score_p95': _interval(score, lambda v: np.quantile(v, 0.95), draws, seed),
        'floor_p5': _interval(floor, lambda v: np.quantile(v, 0.05), draws, seed),
        'floor_min': float(floor.min()),
        'violation': wilson(int(violated.sum()), len(violated)),
    }
