"""MOEA/D's loop: one subproblem per weight vector, ranked by Tchebycheff distance.

The weight vectors and their neighbourhoods are built from the objective count of the FIRST told
evaluation rather than from a constructor argument, so the grid cannot describe a different
problem from the matrix it ranks. The partition count comes from
:func:`~optimi_lab.core.pareto.partition_count` -- the ceiling, the refusal of a one-objective
grid and its coverage guarantee are that function's, not a second copy's.

Under constraints a child displaces a neighbour by Deb's rule: a smaller total violation wins
outright, and only an equal violation -- zero, for two feasible points -- falls through to the
Tchebycheff comparison. With no constraint every violation is zero, which is the plain loop.
"""

import numpy as np

from optimi_lab.core import refuse
from optimi_lab.operator import (
    decomposition_weights,
    nearest_neighbors,
    polynomial_mutation,
    simulated_binary_crossover,
    tchebycheff,
)
from optimi_lab.oracle import Capability
from optimi_lab.problem import Problem
from optimi_lab.solver.population import PopulationSolver, build

__all__ = ['DecompositionSolver', 'moead']

_WEIGHT_FLOOR = 1e-6
"""A zero Tchebycheff weight ignores its objective, so its subproblem's optimum is only WEAKLY Pareto;
every weight is floored here, as Zhang and Li's implementation does."""


class DecompositionSolver(PopulationSolver):
    """MOEA/D: one subproblem per weight vector, each ranked by Tchebycheff distance to the ideal.

    The population is one solution per subproblem and never a shorter list of them: a child that
    produced no value leaves its own subproblem's parent in place, because dropping the
    subproblem would discard a solution nothing was wrong with. ``neighborhood_size`` is a count
    of neighbours, and None is the whole population.
    """

    def __init__(
        self,
        problem: Problem,
        capability: Capability,
        *,
        neighborhood_size: int | None = None,
        neighbor_rate: float = 0.9,
        max_replace: int = 2,
        crossover_rate: float = 0.9,
        eta_crossover: float = 20.0,
        mutation_rate: float = 0.1,
        eta_mutation: float = 20.0,
        **base: object,
    ) -> None:
        super().__init__(problem, capability, **base)
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
        self._violation: np.ndarray | None = None
        self._weights: np.ndarray | None = None
        self._neighbors: np.ndarray | None = None
        self._ideal: np.ndarray | None = None

    def _child(self, population: np.ndarray, width: int, rng: np.random.Generator) -> np.ndarray:
        """One child per subproblem, from two parents its own neighbourhood supplies."""
        count = len(population)
        pairs = np.empty((count, 2), dtype=np.intp)
        for row in range(count):
            pool = self._neighbors[row] if rng.random() < self._neighbor_rate else np.arange(count)
            pairs[row] = rng.choice(pool, 2, replace=False)
        children = simulated_binary_crossover(
            population[pairs[:, 0]],
            population[pairs[:, 1]],
            width,
            crossover_rate=self._crossover_rate,
            eta_crossover=self._eta_crossover,
            rng=rng,
        )
        return polynomial_mutation(
            children, width, mutation_rate=self._mutation_rate, eta_mutation=self._eta_mutation, rng=rng
        )

    def _admit(self, inputs: np.ndarray, values: np.ndarray, violation: np.ndarray, mask: np.ndarray) -> None:
        if self._population is None:
            self._population, self._values, self._violation = inputs, values, violation
            self._weights = np.maximum(
                decomposition_weights(values.shape[1], len(inputs), rng=self._rng), _WEIGHT_FLOOR
            )
            # A neighbourhood is bounded by the subproblems that exist: the first batch can lose a
            # point, and the count it was built with is the constructor's, not this batch's.
            self._neighbors = nearest_neighbors(self._weights, min(self._neighborhood_size, len(inputs)))
            self._ideal = values.min(axis=0)
            return
        self._ideal = np.minimum(self._ideal, values.min(axis=0))
        self._replace(inputs, values, violation, np.flatnonzero(mask))

    def _replace(self, inputs: np.ndarray, values: np.ndarray, violation: np.ndarray, subproblems: np.ndarray) -> None:
        """Let each child displace up to ``max_replace`` of the neighbours it beats on Tchebycheff.

        A child belongs to the subproblem whose row it was proposed for, so the survivors keep
        their own subproblem index, and a child that produced nothing leaves every parent alone.
        """
        for position, subproblem in enumerate(subproblems):
            neighbors = self._neighbors[subproblem]
            scores = tchebycheff(values[position : position + 1], self._weights[neighbors], self._ideal)
            current = tchebycheff(self._values[neighbors], self._weights[neighbors], self._ideal)
            held = self._violation[neighbors]
            wins = (violation[position] < held) | ((violation[position] == held) & (scores < current))
            better = neighbors[wins][: self._max_replace]
            self._population[better] = inputs[position]
            self._values[better] = values[position]
            self._violation[better] = violation[position]

    def _propose(self) -> np.ndarray:
        return self._child(self._population, self._width, self._rng)


def moead(problem: Problem, capability: Capability, **settings: object) -> DecompositionSolver:
    """MOEA/D (Zhang and Li, 2007): one Tchebycheff subproblem per weight vector, two objectives or more.

    ``**settings`` are :class:`DecompositionSolver`'s neighbourhood and SBX/mutation rates and the base
    solver's ``spec``, ``initial`` and ``seed``.
    """
    return build(DecompositionSolver, problem, capability, 'moead', **settings)
