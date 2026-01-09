from collections.abc import Callable

import numpy as np

from optimi_lab.intelligent_algorithm.operators.crossover import BinomialCrossover
from optimi_lab.intelligent_algorithm.operators.mutation import DifferentialMutation
from optimi_lab.intelligent_algorithm.operators.selection import ParetoCrowdingSelection
from optimi_lab.utils.variable_space import VariableSpace

from .intelligent_algorithm_base import IntelligentAlgorithmBase


class MODE_Algorithm(IntelligentAlgorithmBase):
    """Multi-Objective Differential Evolution (MODE).

    MODE is an extension of Differential Evolution for multi-objective optimization problems.
    It uses Pareto dominance and crowding distance to maintain diversity and approximate the Pareto front.
    Refer to *F. Xue, A. C. Sanderson and R. J. Graves, "Multi-objective differential evolution - algorithm, convergence analysis, and applications," 2005 IEEE Congress on Evolutionary Computation, Edinburgh, UK, 2005, pp. 743-750 Vol.1, doi: 10.1109/CEC.2005.1554757*

    Attributes:
        _mutation_operator(DifferentialMutation): Mutation operator.
        _crossover_operator(BinomialCrossover): Crossover operator.
        _selection_operator(ParetoCrowdingSelection): Selection operator.

    """

    _mutation_operator: DifferentialMutation
    _crossover_operator: BinomialCrossover
    _selection_operator: ParetoCrowdingSelection

    def __init__(
        self,
        variable_space: VariableSpace,
        n_obj: int,
        pop_size: int,
        max_iter: int,
        object_function: Callable | None = None,
        mutation_operator: DifferentialMutation = None,
        crossover_operator: BinomialCrossover = None,
        select_operator: ParetoCrowdingSelection = None,
    ) -> None:
        super().__init__(
            object_function=object_function,
            variable_space=variable_space,
            n_obj=n_obj,
            pop_size=pop_size,
            max_iter=max_iter,
        )
        self._mutation_operator = mutation_operator or DifferentialMutation()
        self._crossover_operator = crossover_operator or BinomialCrossover()
        self._selection_operator = select_operator or ParetoCrowdingSelection(n_selected=pop_size)

    def _step(self):
        """Perform one iteration (implemented in parallel for performance)."""
        # Differential mutation
        trial_vectors = self._mutation_operator.do(
            current_inputs=self._current_inputs, best_inputs=self._pareto_inputs, variable_space=self._variable_space
        )

        # Crossover operation
        offspring_inputs = self._crossover_operator.do(
            parent1=self._current_inputs, parent2=trial_vectors, variable_space=self._variable_space
        )

        # Evaluate new individuals
        offspring_outputs = self._object_function(offspring_inputs)

        # Selection operation
        combined_inputs = np.vstack([self._current_inputs, offspring_inputs])
        combined_outputs = np.vstack([self._current_outputs, offspring_outputs])
        selected_indices = self._selection_operator.do(fitness=combined_outputs)

        # Update population
        self._current_inputs = combined_inputs[selected_indices]
        self._current_outputs = combined_outputs[selected_indices]
