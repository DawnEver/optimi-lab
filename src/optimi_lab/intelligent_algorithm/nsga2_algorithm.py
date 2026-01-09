from collections.abc import Callable

import numpy as np

from optimi_lab.intelligent_algorithm.operators.crossover import SimulatedBinaryCrossover
from optimi_lab.intelligent_algorithm.operators.mutation import PolynomialMutation
from optimi_lab.intelligent_algorithm.operators.selection import (
    ParetoCrowdingSelection,
    RandomSelection,
    TournamentSelection,
)
from optimi_lab.utils.variable_space import VariableSpace

from .intelligent_algorithm_base import IntelligentAlgorithmBase


class NSGA2_Algorithm(IntelligentAlgorithmBase):
    """Non-Dominated Sorting Genetic Algorithm II (NSGA-II).

    NSGA-II is an evolutionary algorithm for multi-objective optimization.
    It uses non-dominated sorting and crowding distance to preserve diversity
    and approximate the Pareto front.

    Reference:
        K. Deb, A. Pratap, S. Agarwal, and T. Meyarivan. A fast and elitist
        multiobjective genetic algorithm: NSGA-II. Trans. Evol. Comp, 6(2):182-197, 2002.

    Attributes:
        _parent_selection_operator (TournamentSelection|RandomSelection): Parent selection operator.
        _mutation_operator (PolynomialMutation): Mutation operator.
        _crossover_operator (SimulatedBinaryCrossover): Crossover operator.
        _selection_operator (ParetoCrowdingSelection): Environmental selection operator.

    """

    _parent_selection_operator: TournamentSelection | RandomSelection
    _mutation_operator: PolynomialMutation
    _crossover_operator: SimulatedBinaryCrossover
    _selection_operator: ParetoCrowdingSelection

    def __init__(
        self,
        variable_space: VariableSpace,
        n_obj: int,
        pop_size: int,
        max_iter: int,
        object_function: Callable | None = None,
        parent_selection_operator: TournamentSelection | RandomSelection = None,
        mutation_operator: PolynomialMutation = None,
        crossover_operator: SimulatedBinaryCrossover = None,
        selection_operator: ParetoCrowdingSelection = None,
    ) -> None:
        super().__init__(
            object_function=object_function,
            variable_space=variable_space,
            n_obj=n_obj,
            pop_size=pop_size,
            max_iter=max_iter,
        )

        # if not provided, use default operators
        self._parent_selection_operator = parent_selection_operator or TournamentSelection(n_selected=pop_size)
        self._mutation_operator = mutation_operator or PolynomialMutation()
        self._crossover_operator = crossover_operator or SimulatedBinaryCrossover()
        self._selection_operator = selection_operator or ParetoCrowdingSelection(n_selected=pop_size)

    def _perform_crossover(self, parents_inputs: np.ndarray) -> np.ndarray:
        """Perform crossover to generate offspring.

        Args:
            parents_inputs (np.ndarray): Parent pairs with shape (pop_size, 2, n_var).

        Returns:
            np.ndarray: Offspring individuals, shape (pop_size, n_var).

        """
        offspring_list = []
        for i in range(self._pop_size):
            parent1 = parents_inputs[i, 0:1]
            parent2 = parents_inputs[i, 1:2]

            child1, child2 = self._crossover_operator.do(
                parent1=parent1, parent2=parent2, variable_space=self._variable_space
            )
            offspring_list.append(child1[0])
            if len(offspring_list) < self._pop_size:
                offspring_list.append(child2[0])

        # Ensure the number of offspring is correct
        return np.array(offspring_list[: self._pop_size])

    def _step(self):
        """Perform a single NSGA-II iteration."""
        # 1. Parent selection - tournament or configured selector
        parent_indices = self._parent_selection_operator.do(fitness=self._current_outputs)

        # Reorganize parents into pairs with shape (pop_size, 2)
        parent_pairs = np.zeros((self._pop_size, 2), dtype=int)
        for i in range(self._pop_size):
            if i % 2 == 0:
                # Even indices: take current and next
                parent_pairs[i, 0] = parent_indices[i]
                parent_pairs[i, 1] = parent_indices[i + 1] if i + 1 < self._pop_size else parent_indices[0]
            else:
                # Odd indices: take current and previous
                parent_pairs[i, 0] = parent_indices[i]
                parent_pairs[i, 1] = parent_indices[i - 1]

        parents_inputs = self._current_inputs[parent_pairs]

        # 2. Crossover - pairwise crossover to create offspring
        offspring_inputs = self._perform_crossover(parents_inputs)

        # 3. Mutation
        offspring_inputs = self._mutation_operator.do(
            current_inputs=offspring_inputs, best_inputs=self._pareto_inputs, variable_space=self._variable_space
        )

        # 4. Evaluate offspring
        offspring_outputs = self._object_function(offspring_inputs)

        # 5. Environmental selection
        # Selection operation
        combined_inputs = np.vstack([self._current_inputs, offspring_inputs])
        combined_outputs = np.vstack([self._current_outputs, offspring_outputs])
        selected_indices = self._selection_operator.do(fitness=combined_outputs)

        # Update population
        self._current_inputs = combined_inputs[selected_indices]
        self._current_outputs = combined_outputs[selected_indices]
