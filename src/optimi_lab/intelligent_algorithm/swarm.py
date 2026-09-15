"""MOPSO's loop: particles with personal bests, led by a bounded archive.

Every array this loop holds is one row per particle, allocated from the row count of the first
told evaluation, so a swarm cannot be shaped by anything but its own population -- the measured
failure was an 8-particle swarm broadcast against a 121-row sample grid. A particle whose
evaluation produced no value is dropped from all four arrays together, which the mask keeps
aligned.
"""

from collections.abc import Callable

import numpy as np

from optimi_lab.core import Evaluation, VariableSet, dominates, refuse, require_positive
from optimi_lab.intelligent_algorithm.proposers import PopulationProposer
from optimi_lab.intelligent_algorithm.selection import crowding_select
from optimi_lab.intelligent_algorithm.variation import pso_position, pso_velocity

__all__ = ['SwarmProposer']

_ARCHIVE_PER_POPULATION = 2
"""A MOPSO archive is bounded by this many times the population before it is thinned."""

_EXPLORATION_RATE = 0.1
"""Probability that a particle keeps a new position that neither dominates nor is dominated."""


class SwarmProposer(PopulationProposer):
    """MOPSO: particles with personal bests, led by a bounded archive of the non-dominated points.

    ``max_velocity_fraction`` is the fraction of each variable's own range a particle may cross in
    one step, ``archive_size`` how many points the archive keeps (None is twice the population),
    ``cognitive``/``social``/``inertia`` the three weights of the velocity update, and ``selection``
    the rule the archive is thinned by.
    """

    def __init__(
        self,
        space: VariableSet,
        pop_size: int,
        *,
        inertia: float = 0.4,
        cognitive: float = 2.0,
        social: float = 2.0,
        max_velocity_fraction: float = 0.1,
        archive_size: int | None = None,
        selection: Callable[[np.ndarray, int], np.ndarray] = crowding_select,
        seed: int = 0,
    ) -> None:
        super().__init__(space, pop_size, seed=seed)
        if not 0.0 <= inertia <= 1.0:
            refuse(f'inertia must be in [0, 1], got {inertia!r}')
        if cognitive < 0 or social < 0:
            refuse(f'the learning factors must be non-negative, got cognitive={cognitive!r} and social={social!r}')
        if not 0.0 < max_velocity_fraction <= 1.0:
            refuse(
                f'max_velocity_fraction is the fraction of a variable range a particle may cross in one step, so it '
                f'must be in (0, 1], got {max_velocity_fraction!r}'
            )
        self._inertia = inertia
        self._cognitive = cognitive
        self._social = social
        self._max_velocity_fraction = max_velocity_fraction
        self._selection = selection
        self._archive_size = (
            int(require_positive(archive_size, 'archive_size'))
            if archive_size is not None
            else _ARCHIVE_PER_POPULATION * self._pop_size
        )
        self._positions: np.ndarray | None = None
        self._values: np.ndarray | None = None
        self._velocities: np.ndarray | None = None
        self._personal_best: np.ndarray | None = None
        self._personal_best_values: np.ndarray | None = None
        self._archive = np.zeros((0, self._space.n_var))
        self._archive_values = np.zeros((0, 1))

    def _admit(self, evaluation: Evaluation) -> None:
        inputs, values, mask = self._complete(evaluation)
        if self._positions is None:
            self._start(inputs, values)
            return
        best = self._personal_best[mask]
        best_values = self._personal_best_values[mask]
        improves = dominates(values, best_values)
        neither = ~(improves | dominates(best_values, values))
        improves |= neither & (self._rng.random(len(values)) < _EXPLORATION_RATE)
        best[improves], best_values[improves] = inputs[improves], values[improves]
        self._personal_best, self._personal_best_values = best, best_values
        self._positions, self._values = inputs, values
        self._velocities = self._velocities[mask]
        self._grow_archive(inputs, values)

    def _start(self, inputs: np.ndarray, values: np.ndarray) -> None:
        """Take the first told batch as the particles, at rest and each its own personal best."""
        self._positions = inputs
        self._values = values
        self._velocities = np.zeros_like(inputs)
        self._personal_best = inputs.copy()
        self._personal_best_values = values.copy()
        self._archive_values = np.zeros((0, values.shape[1]))
        self._grow_archive(inputs, values)

    def _grow_archive(self, inputs: np.ndarray, values: np.ndarray) -> None:
        """Add the new points, then thin the archive back to ``archive_size``.

        The count asked of the selection is the archive bound WHERE THE POPULATION REACHES IT: the
        first generation holds fewer points than the bound, and asking a selector for more rows
        than exist is what the old MOPSO did on every run shorter than two generations.
        """
        combined_inputs = np.vstack([self._archive, inputs])
        combined_values = np.vstack([self._archive_values, values])
        keep = self._selection(combined_values, min(self._archive_size, len(combined_values)))
        self._archive, self._archive_values = combined_inputs[keep], combined_values[keep]

    def _propose(self) -> np.ndarray:
        self._velocities = pso_velocity(
            self._velocities,
            self._positions,
            self._personal_best,
            self._archive,
            self._space,
            inertia=self._inertia,
            cognitive=self._cognitive,
            social=self._social,
            max_velocity_fraction=self._max_velocity_fraction,
            rng=self._rng,
        )
        return pso_position(self._positions, self._velocities, self._space)
