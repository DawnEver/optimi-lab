"""MOEA/D's loop: one subproblem per weight vector, ranked by Tchebycheff distance.

The weight vectors and their neighbourhoods are built from the objective count of the FIRST told
evaluation rather than from a constructor argument, so the grid cannot describe a different
problem from the matrix it ranks. The partition count comes from
:func:`~optimi_lab.core.pareto.partition_count` -- the ceiling, the refusal of a one-objective
grid and its coverage guarantee are that function's, not a second copy's.
"""

import numpy as np

from optimi_lab.core import Evaluation, VariableSet, refuse
from optimi_lab.intelligent_algorithm.proposers import PopulationProposer
from optimi_lab.intelligent_algorithm.selection import decomposition_weights, nearest_neighbors, tchebycheff
from optimi_lab.intelligent_algorithm.variation import polynomial_mutation, simulated_binary_crossover

__all__ = ['DecompositionProposer']


class DecompositionProposer(PopulationProposer):
    """MOEA/D: one subproblem per weight vector, each ranked by Tchebycheff distance to the ideal.

    The population is one solution per subproblem and never a shorter list of them: a child that
    produced no value leaves its own subproblem's parent in place, because dropping the
    subproblem would discard a solution nothing was wrong with. ``neighborhood_size`` is a count
    of neighbours, and None is the whole population.
    """

    def __init__(
        self,
        space: VariableSet,
        pop_size: int,
        *,
        neighborhood_size: int | None = None,
        neighbor_rate: float = 0.9,
        max_replace: int = 2,
        crossover_rate: float = 0.9,
        eta_crossover: float = 20.0,
        mutation_rate: float = 0.1,
        eta_mutation: float = 20.0,
        seed: int = 0,
    ) -> None:
        super().__init__(space, pop_size, seed=seed)
        if neighborhood_size is None:
            neighborhood_size = self._pop_size
        if neighborhood_size < 2:
            refuse(
                f'neighborhood_size must be at least 2 -- both parents of a child come from it -- '
                f'got {neighborhood_size!r}'
            )
        if neighborhood_size > self._pop_size:
            refuse(
                f'neighborhood_size {neighborhood_size} exceeds the {self._pop_size} subproblem(s) a weight vector '
                f'belongs to; the valid neighborhood sizes for this population are 2 to {self._pop_size}'
            )
        if max_replace < 1:
            refuse(f'max_replace must be at least 1, got {max_replace!r}')
        if max_replace > neighborhood_size:
            refuse(f'max_replace must not exceed neighborhood_size={neighborhood_size}, got {max_replace!r}')
        if not 0.0 <= neighbor_rate <= 1.0:
            refuse(f'neighbor_rate must be in [0, 1], got {neighbor_rate!r}')
        self._neighborhood_size = int(neighborhood_size)
        self._neighbor_rate = neighbor_rate
        self._max_replace = int(max_replace)
        self._crossover_rate = crossover_rate
        self._eta_crossover = eta_crossover
        self._mutation_rate = mutation_rate
        self._eta_mutation = eta_mutation
        self._population: np.ndarray | None = None
        self._values: np.ndarray | None = None
        self._weights: np.ndarray | None = None
        self._neighbors: np.ndarray | None = None
        self._ideal: np.ndarray | None = None

    def _child(self, population: np.ndarray, space: VariableSet, rng: np.random.Generator) -> np.ndarray:
        """One child per subproblem, from two parents its own neighbourhood supplies."""
        count = len(population)
        pairs = np.empty((count, 2), dtype=np.intp)
        for row in range(count):
            pool = self._neighbors[row] if rng.random() < self._neighbor_rate else np.arange(count)
            pairs[row] = rng.choice(pool, 2, replace=False)
        children = simulated_binary_crossover(
            population[pairs[:, 0]],
            population[pairs[:, 1]],
            space,
            crossover_rate=self._crossover_rate,
            eta_crossover=self._eta_crossover,
            rng=rng,
        )
        return polynomial_mutation(
            children, space, mutation_rate=self._mutation_rate, eta_mutation=self._eta_mutation, rng=rng
        )

    def _admit(self, evaluation: Evaluation) -> None:
        inputs, values, mask = self._complete(evaluation)
        if self._population is None:
            self._population, self._values = inputs, values
            self._weights = decomposition_weights(values.shape[1], len(inputs), rng=self._rng)
            # A neighbourhood is bounded by the subproblems that exist: the first batch can lose a
            # point, and the count it was built with is the constructor's, not this batch's.
            self._neighbors = nearest_neighbors(self._weights, min(self._neighborhood_size, len(inputs)))
            self._ideal = values.min(axis=0)
            return
        self._ideal = np.minimum(self._ideal, values.min(axis=0))
        self._replace(inputs, values, np.flatnonzero(mask))

    def _replace(self, inputs: np.ndarray, values: np.ndarray, subproblems: np.ndarray) -> None:
        """Let each child displace up to ``max_replace`` of the neighbours it beats on Tchebycheff.

        A child belongs to the subproblem whose row it was proposed for, so the survivors keep
        their own subproblem index, and a child that produced nothing leaves every parent alone.
        """
        for position, subproblem in enumerate(subproblems):
            neighbors = self._neighbors[subproblem]
            scores = tchebycheff(values[position : position + 1], self._weights[neighbors], self._ideal)[0]
            current = tchebycheff(self._values[neighbors], self._weights[neighbors], self._ideal)
            better = neighbors[scores < current][: self._max_replace]
            self._population[better] = inputs[position]
            self._values[better] = values[position]

    def _propose(self) -> np.ndarray:
        return self._child(self._population, self._space, self._rng)
