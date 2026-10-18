"""Learning which objectives conflict, and which are redundant (objective reduction).

* :func:`conflict_matrix` -- for each objective pair, the share of point pairs on which they DISAGREE
  (one improves while the other worsens; ties on either skipped). 1 everywhere along a two-objective
  front, 0 between an objective and a monotone copy.
* :func:`essential_objectives` -- greedy objective reduction in the spirit of Brockhoff and Zitzler
  (2009): drop, one at a time, the objective whose removal changes the dominance relation on the
  given points least, while the share of changed ordered pairs stays within ``tolerance``. What is
  kept preserves who dominates whom; a solver may then run on fewer objectives.

Minimisation frame; the points are any evaluated set, a front or a whole archive.
"""

import numpy as np

from optimi_lab.core import dominates, refuse, require_finite

__all__ = ['conflict_matrix', 'essential_objectives']


def conflict_matrix(values: np.ndarray) -> np.ndarray:
    """``(m, m)`` share of discordant point pairs per objective pair; the diagonal is 0."""
    values = require_finite(np.asarray(values, dtype=float), 'an objective matrix')
    sign = np.sign(values[:, None, :] - values[None, :, :])
    upper = np.triu_indices(len(values), 1)
    sign = sign[upper]
    m = values.shape[1]
    out = np.zeros((m, m))
    for i in range(m):
        for j in range(i + 1, m):
            both = (sign[:, i] != 0) & (sign[:, j] != 0)
            out[i, j] = out[j, i] = float(np.mean(sign[both, i] != sign[both, j])) if both.any() else 0.0
    return out


def _relation(values: np.ndarray) -> np.ndarray:
    return dominates(values[:, None, :], values[None, :, :])


def essential_objectives(values: np.ndarray, *, tolerance: float = 0.0) -> tuple[int, ...]:
    """The objectives kept by greedy reduction; ``tolerance`` is the share of ordered pairs allowed to change."""
    if tolerance < 0.0:
        refuse(f'tolerance is a share of ordered point pairs, at least 0, got {tolerance!r}')
    values = require_finite(np.asarray(values, dtype=float), 'an objective matrix')
    full = _relation(values)
    pairs = max(1, len(values) * (len(values) - 1))
    kept = list(range(values.shape[1]))
    while len(kept) > 1:
        errors = [
            (np.count_nonzero(_relation(values[:, [k for k in kept if k != drop]]) != full) / pairs, drop)
            for drop in kept
        ]
        error, drop = min(errors)
        if error > tolerance:
            break
        kept.remove(drop)
    return tuple(kept)
