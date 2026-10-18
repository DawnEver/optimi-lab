"""Morris elementary effects (Morris, 1991; mu* by Campolongo et al., 2007): a cheap screening of inputs.

``trajectories`` random one-at-a-time paths over a ``levels``-level grid of the unit cube, each of
``k + 1`` points, step ``delta = levels / (2 (levels - 1))``. The effect of a step in ``x_i`` is the
output change per unit of the variable's RANGE (output units), so a linear term's effect is its
coefficient times the range width. ``mu*`` (mean absolute effect) ranks importance; ``sigma`` (the
effects' standard deviation) flags non-linearity or interaction.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from optimi_lab.core import refuse, require_finite
from optimi_lab.problem import Real, Space

__all__ = ['MorrisEffects', 'morris']


@dataclass(frozen=True, slots=True)
class MorrisEffects:
    """Per variable: ``mu`` (signed mean), ``mu_star`` (mean absolute) and ``sigma`` of the effects."""

    names: tuple[str, ...]
    mu: np.ndarray
    mu_star: np.ndarray
    sigma: np.ndarray
    n_evaluations: int


def morris(
    space: Space, function: Callable[[np.ndarray], np.ndarray], *, trajectories: int, levels: int = 4, seed: int
) -> MorrisEffects:
    """Elementary effects of *function* (``(n, k)`` design matrix -> ``n`` values) over a Real *space*."""
    strangers = [v.name for v in space if not isinstance(v, Real) or v.active_if is not None]
    if strangers:
        refuse(f'Morris screening here needs unconditional Real variables; {strangers} are not')
    if levels < 2 or levels % 2 or trajectories < 2:
        refuse(f'Morris needs an even levels >= 2 and trajectories >= 2, got {levels} and {trajectories}')
    k = space.width
    rng = np.random.default_rng(seed)
    delta = levels / (2.0 * (levels - 1))
    grid = np.arange(levels // 2) / (levels - 1)
    paths, signs = [], []
    for _ in range(trajectories):
        start = rng.choice(grid, k)
        order = rng.permutation(k)
        path = [start.copy()]
        for i in order:
            point = path[-1].copy()
            point[i] += delta
            path.append(point)
        paths.append(path)
        signs.append(order)
    unit = np.array(paths).reshape(-1, k)
    lows = np.array([v.lo for v in space])
    widths = np.array([v.hi - v.lo for v in space])
    outputs = require_finite(np.asarray(function(lows + unit * widths), dtype=float), 'the function values')
    if outputs.shape != (len(unit),):
        refuse(f'the function must return one value per point, shape ({len(unit)},); got {outputs.shape}')
    outputs = outputs.reshape(trajectories, k + 1)
    effects = np.zeros((trajectories, k))
    for t, order in enumerate(signs):
        effects[t, order] = np.diff(outputs[t]) / delta
    return MorrisEffects(
        names=space.names,
        mu=effects.mean(axis=0),
        mu_star=np.abs(effects).mean(axis=0),
        sigma=effects.std(axis=0, ddof=1),
        n_evaluations=len(unit),
    )
