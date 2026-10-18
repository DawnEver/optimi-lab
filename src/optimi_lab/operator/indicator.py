"""Selection by an indicator over the population: SPEA2 strength and hypervolume contribution.

* :func:`spea2_fitness` / :func:`spea2_select` -- Zitzler, Laumanns and Thiele (2001): raw fitness is
  the summed strength of a row's dominators, density ``1 / (sigma_k + 2)`` from the distance to its
  k-th nearest neighbour (``k = floor(sqrt(N))``); the non-dominated rows survive, truncated by
  iteratively removing the row nearest to another (ties broken by the next-nearest), or filled by
  the best dominated rows by fitness.
* :func:`hypervolume_select` -- SMS-EMOA's reduction (Beume et al., 2007): whole fronts, then the
  overflowing front loses its least exclusive contributor, one at a time.

Constraints enter through Deb's rule: a row of smaller violation dominates one of larger, and two
feasible rows compare by Pareto dominance.
"""

from functools import partial

import numpy as np

from optimi_lab.core import constrained_dominates
from optimi_lab.metric import hypervolume_contributions
from optimi_lab.operator.selection import fronts_then_fill, require_rankable, require_selectable, violation_of

__all__ = ['hypervolume_select', 'spea2_fitness', 'spea2_select']


def _distances(values: np.ndarray) -> np.ndarray:
    gaps = np.linalg.norm(values[:, None, :] - values[None, :, :], axis=-1)
    np.fill_diagonal(gaps, np.inf)
    return gaps


def spea2_fitness(values: np.ndarray, *, violation: np.ndarray | None = None) -> np.ndarray:
    """SPEA2 fitness: raw (summed dominator strength) plus density; below 1 exactly for the non-dominated."""
    values = require_rankable(values)
    violation = violation_of(violation, len(values))
    beats = constrained_dominates(values[:, None, :], values[None, :, :], violation[:, None], violation[None, :])
    strength = beats.sum(axis=1)
    raw = (beats * strength[:, None]).sum(axis=0).astype(float)
    if len(values) == 1:
        return raw + 0.5
    k = max(1, min(int(np.sqrt(len(values))), len(values) - 1))
    sigma = np.sort(_distances(values), axis=1)[:, k - 1]
    return raw + 1.0 / (sigma + 2.0)


def spea2_select(values: np.ndarray, n_selected: int, *, violation: np.ndarray | None = None) -> np.ndarray:
    """SPEA2 environmental selection of *n_selected* rows."""
    values = require_rankable(values)
    require_selectable(n_selected, len(values))
    fitness = spea2_fitness(values, violation=violation)
    chosen = np.flatnonzero(fitness < 1.0)
    if len(chosen) <= n_selected:
        rest = np.setdiff1d(np.argsort(fitness, kind='stable'), chosen, assume_unique=True)
        rest = rest[np.argsort(fitness[rest], kind='stable')]
        return np.concatenate([chosen, rest[: n_selected - len(chosen)]]).astype(np.intp)
    kept = list(chosen)
    while len(kept) > n_selected:
        gaps = np.sort(_distances(values[kept]), axis=1)
        kept.pop(min(range(len(kept)), key=lambda i: tuple(gaps[i])))
    return np.array(kept, dtype=np.intp)


def _hypervolume_fill(front: np.ndarray, n_fill: int, *, reference: np.ndarray | None) -> np.ndarray:
    point = front.max(axis=0) + 1.0 if reference is None else reference
    kept = list(range(len(front)))
    while len(kept) > n_fill:
        kept.pop(int(np.argmin(hypervolume_contributions(front[kept], point))))
    return np.array(kept, dtype=np.intp)


def hypervolume_select(
    values: np.ndarray,
    n_selected: int,
    *,
    reference: np.ndarray | None = None,
    violation: np.ndarray | None = None,
) -> np.ndarray:
    """Whole fronts, the overflowing one reduced by least exclusive hypervolume contribution.

    ``reference`` is in the minimised normalised frame; None is that front's nadir plus one unit,
    so the extreme points keep a positive contribution.
    """
    values = require_rankable(values)
    return fronts_then_fill(values, n_selected, partial(_hypervolume_fill, reference=reference), violation)
