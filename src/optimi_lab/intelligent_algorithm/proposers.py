"""The base proposer, and the evolutionary loop three of the five algorithms share.

:func:`optimi_lab.core.optimize.optimize` draws the first batch, tells a proposer what it
produced, then asks for every batch after it. A proposer here runs ONE GENERATION per
:meth:`PopulationProposer.tell`, so the initial sample of a run belongs to the driver.

THE POPULATION SIZE AND THE OBJECTIVE COUNT COME FROM THE DATA. ``build_proposer`` is handed the
row count :func:`optimi_lab.core.sampling.sample` produced, and the objective count is read off
the first told evaluation's width. The old base class took both as constructor arguments AND took
its first batch from ``variable_space.var_space_matrix``: measured, four iterations evaluated
``[121, 8, 8, 8]`` points for a declared ``pop_size`` of 8, and MOPSO broadcast an 8-particle swarm
against a 121-row grid.

The population stays ``pop_size`` wide while that many points produce values; when evaluations
fail it is as wide as the complete points that remain, because a quota filled by duplicating a
point would misstate how much of the search space a run covered.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable

import numpy as np

from optimi_lab.core import Evaluation, VariableSet, refuse, require_axis, require_positive

__all__ = ['EvolutionaryProposer', 'PopulationProposer']


class PopulationProposer(ABC):
    """A proposer that keeps a population and builds the next batch inside :meth:`tell`.

    ``space`` is what every batch must lie inside, ``pop_size`` is the row count the first
    :meth:`tell` will carry, and ``seed`` seeds every draw -- two runs with one seed are one run.
    """

    def __init__(self, space: VariableSet, pop_size: int, *, seed: int = 0) -> None:
        self._space = space
        self._pop_size = int(require_positive(pop_size, 'pop_size'))
        self._rng = np.random.default_rng(seed)
        self._offspring: np.ndarray | None = None

    def ask(self) -> np.ndarray:
        """Return the batch the last :meth:`tell` built, shape ``(n_points, n_var)``.

        Refuses before the first :meth:`tell`: the first batch of a run is
        :func:`optimi_lab.core.sampling.sample`'s, and a proposer inventing one would be a second
        source for a count the driver has already fixed.
        """
        if self._offspring is None:
            refuse(
                'ask() was called before tell(); the first batch of a run is drawn by optimi_lab.core.sample and '
                'told to the proposer, so call tell() with the initial evaluation before asking for a batch'
            )
        return self._offspring

    def tell(self, evaluation: Evaluation) -> None:
        """Accept what the last batch produced, and build the next batch from it.

        ``evaluation`` holds one row per point asked for, values in the MINIMIZATION frame, and one
        :class:`~optimi_lab.core.outcomes.Outcome` per cell.
        """
        self._admit(evaluation)
        self._offspring = self._propose()

    def _complete(self, evaluation: Evaluation) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(inputs, values, mask)`` for the rows that produced a value for every objective.

        A row with any other outcome is DROPPED, never scored as bad: an unevaluated point has no
        value, and every ranking here refuses a non-finite column rather than sorting an arithmetic
        accident onto a front. A batch in which no row produced a value leaves nothing to rank, so
        it is refused naming the outcomes it did produce.
        """
        inputs = require_axis(evaluation.inputs, self._space.n_var, 1, 'the inputs of a told evaluation')
        mask = evaluation.complete_mask()
        if not mask.any():
            outcomes = sorted({outcome.value for outcome in evaluation.point_outcomes()})
            refuse(
                f'none of the {len(evaluation)} told point(s) produced a value for every objective, with outcome(s) '
                f'{outcomes}; there is no population to build the next batch from'
            )
        return inputs[mask], evaluation.values[mask], mask

    @abstractmethod
    def _admit(self, evaluation: Evaluation) -> None:
        """Fold one told evaluation into this proposer's state, seeding it on the first call."""

    @abstractmethod
    def _propose(self) -> np.ndarray:
        """Return the next batch, shape ``(n_points, n_var)``, with at least one point."""


class EvolutionaryProposer(PopulationProposer):
    """A generation of an evolutionary algorithm: vary a population, then survive from the union.

    One loop, three operator sets. ``variation`` turns a population into offspring, ``selection``
    keeps the survivors from parents and offspring together, and ``parent_selection`` picks the
    mating pool from the population's own values -- None mates every individual with the one
    selected before it, which is what a differential evolution wants. The count asked of a
    selector is the population size where that many rows exist and every row available otherwise,
    so a run whose evaluations fail shrinks rather than padding itself with a duplicate.
    """

    def __init__(
        self,
        space: VariableSet,
        pop_size: int,
        *,
        variation: Callable[..., np.ndarray],
        selection: Callable[[np.ndarray, int], np.ndarray],
        parent_selection: Callable[[np.ndarray, int, np.random.Generator], np.ndarray] | None = None,
        seed: int = 0,
    ) -> None:
        super().__init__(space, pop_size, seed=seed)
        self._variation = variation
        self._selection = selection
        self._parent_selection = parent_selection
        self._population: np.ndarray | None = None
        self._values: np.ndarray | None = None

    def _admit(self, evaluation: Evaluation) -> None:
        inputs, values, _ = self._complete(evaluation)
        if self._population is None:
            self._population, self._values = inputs, values
            return
        combined_inputs = np.vstack([self._population, inputs])
        combined_values = np.vstack([self._values, values])
        keep = self._selection(combined_values, min(self._pop_size, len(combined_values)))
        self._population, self._values = combined_inputs[keep], combined_values[keep]

    def _propose(self) -> np.ndarray:
        population = self._population
        if self._parent_selection is not None:
            pool = self._parent_selection(self._values, min(self._pop_size, len(population)), self._rng)
            population = population[pool]
        return self._variation(population, self._space, self._rng)
