"""Stratified sampling of a mixed space: every discrete level equally often, every Real a Latin column.

:func:`optimi_lab.core.sampling.sample` draws over a bare unit cube and knows nothing of levels, so
a Latin column decoded onto a 6-level Integer is balanced only up to rounding at the level edges.
:func:`stratified` reads the :class:`~optimi_lab.problem.Space`: each Integer and embedded
Categorical gets each of its ``L`` levels ``n // L`` or ``n // L + 1`` times (the remainder drawn
without replacement), shuffled independently per variable; each Real or Field column is a Latin
hypercube column.
"""

import numpy as np

from optimi_lab.core import require_positive
from optimi_lab.core.qmc import latin_hypercube
from optimi_lab.operator.discrete import lattice
from optimi_lab.problem import Space

__all__ = ['stratified']


def stratified(space: Space, n_points: int, rng: np.random.Generator) -> np.ndarray:
    """*n_points* unit-cube rows over *space*, balanced per discrete level and stratified per Real column."""
    require_positive(n_points, 'n_points')
    points = latin_hypercube(n_points, space.width, rng)
    for block in lattice(space).blocks:
        count = len(block.levels)
        assignment = np.concatenate([
            np.repeat(np.arange(count), n_points // count),
            rng.choice(count, n_points % count, replace=False),
        ])
        points[:, block.columns] = block.levels[rng.permutation(assignment)]
    return points
