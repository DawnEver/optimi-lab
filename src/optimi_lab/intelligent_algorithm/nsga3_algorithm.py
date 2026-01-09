from collections.abc import Callable

from optimi_lab.intelligent_algorithm.operators.crossover import SimulatedBinaryCrossover
from optimi_lab.intelligent_algorithm.operators.mutation import PolynomialMutation
from optimi_lab.intelligent_algorithm.operators.selection import (
    ParetoRefSelection,
    RandomSelection,
    TournamentSelection,
)
from optimi_lab.utils.variable_space import VariableSpace

from .nsga2_algorithm import NSGA2_Algorithm


class NSGA3_Algorithm(NSGA2_Algorithm):
    """Non-Dominated Sorting Genetic Algorithm III (NSGA-III).

    NSGA-III is an evolutionary algorithm for many-objective optimization problems.
    It uses non-dominated sorting and reference points to maintain diversity and
    find the Pareto-optimal front. Compared to NSGA-II, the primary difference
    lies in the implementation of the selection operator.

    Reference:
        Kalyanmoy Deb and Himanshu Jain. An evolutionary many-objective optimization
        algorithm using reference-point-based nondominated sorting approach, part I:
        solving problems with box constraints. IEEE Transactions on Evolutionary
        Computation, 18(4):577-601, 2014. doi:10.1109/TEVC.2013.2281535.

    Attributes:
        _selection_operator (ParetoRefSelection): NSGA-III selection operator

    """

    _selection_operator: ParetoRefSelection

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
        selection_operator: ParetoRefSelection = None,
    ) -> None:
        super(NSGA2_Algorithm, self).__init__(
            object_function=object_function,
            variable_space=variable_space,
            n_obj=n_obj,
            pop_size=pop_size,
            max_iter=max_iter,
        )

        # Use default operators when not provided
        self._parent_selection_operator = parent_selection_operator or TournamentSelection(n_selected=pop_size)
        self._mutation_operator = mutation_operator or PolynomialMutation()
        self._crossover_operator = crossover_operator or SimulatedBinaryCrossover()
        self._selection_operator = selection_operator or ParetoRefSelection(n_selected=pop_size, n_objectives=n_obj)
