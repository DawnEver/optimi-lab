"""Tests for MODE_Algorithm implementation"""

from collections.abc import Callable

import numpy as np
import pytest

from optimi_lab.intelligent_algorithm.mode_algorithm import MODE_Algorithm
from optimi_lab.intelligent_algorithm.moead_algorithm import MOEAD_Algorithm, generate_weight_vectors
from optimi_lab.intelligent_algorithm.mopso_algorithm import MOPSO_Algorithm
from optimi_lab.intelligent_algorithm.nsga2_algorithm import NSGA2_Algorithm
from optimi_lab.intelligent_algorithm.nsga3_algorithm import NSGA3_Algorithm
from optimi_lab.intelligent_algorithm.operators.crossover import BinomialCrossover, SimulatedBinaryCrossover
from optimi_lab.intelligent_algorithm.operators.mutation import PolynomialMutation
from optimi_lab.intelligent_algorithm.operators.pso_operators import (
    PSOPositionUpdate,
    PSOVelocityUpdate,
)
from optimi_lab.intelligent_algorithm.operators.selection import (
    ParetoCrowdingSelection,
    ParetoRefSelection,
    ParetoSelection,
    RandomSelection,
    TournamentSelection,
)
from optimi_lab.utils.logger import log
from optimi_lab.utils.variable_space import VariableSpace

pop_size = 50
max_iter = 5

n_var = 10

# Create variable space
var_names = [f'x{i}' for i in range(n_var)]
lower_bounds = [0.0] * n_var
upper_bounds = [1.0] * n_var

variable_space = VariableSpace(
    var_name_list=var_names,
    lower_bounds=lower_bounds,
    upper_bounds=upper_bounds,
    sample_type='latin_hypercube',
    n_count=pop_size,
)


def single_objective_problem(x: np.ndarray) -> np.ndarray:
    """Single-objective test problem

    Objective: f(x) = sum(x^2)

    Args:
        x: input matrix, shape (pop_size, n_var)

    Returns:
        objective values matrix, shape (pop_size, 1)

    """
    if x.ndim == 1:
        x = x.reshape(1, -1)
    return np.sum(x**2, axis=1, keepdims=True)


def zdt1_problem(x: np.ndarray) -> np.ndarray:
    """ZDT1 test problem

    Bi-objective optimization problem:
    f1(x) = x1
    f2(x) = g(x) * (1 - sqrt(x1/g(x)))
    g(x) = 1 + 9 * sum(x2:xn) / (n-1)

    Args:
        x: input matrix, shape (pop_size, n_var)

    Returns:
        objective values matrix, shape (pop_size, 2)

    """
    if x.ndim == 1:
        x = x.reshape(1, -1)
    f1 = x[:, 0]
    if x.shape[1] > 1:
        g = 1 + 9 * np.sum(x[:, 1:], axis=1) / (x.shape[1] - 1)
    else:
        g = np.ones(x.shape[0])
    f2 = g * (1 - np.sqrt(f1 / g))
    return np.column_stack([f1, f2])


def display_results(algorithm: MODE_Algorithm):
    msg = '\nOptimization completed!\n'
    msg += f'Number of Pareto individuals: {len(algorithm._pareto_inputs)}\n'
    msg += 'Pareto objective value ranges:\n'
    for i_obj in range(algorithm._n_obj):
        msg += f'  - f{i_obj + 1}: [{algorithm._pareto_outputs[:, i_obj].min():.4f}, {algorithm._pareto_outputs[:, i_obj].max():.4f}]\n'
    msg += '\nTop 5 Pareto solutions:\n'
    for i_output in range(min(5, len(algorithm._pareto_outputs))):
        msg += f'  Solution {i_output + 1}: '
        for i_obj in range(algorithm._n_obj):
            msg += f'f{i_obj}={algorithm._pareto_outputs[i_output, i_obj]:.4f}'
        msg += '\n'
    log(msg=msg, level='info')


def test_no_obj_func():
    """Main test"""
    nsga2 = NSGA2_Algorithm(
        variable_space=variable_space,
        n_obj=3,
        pop_size=pop_size,
        max_iter=max_iter,
        parent_selection_operator=RandomSelection(n_selected=pop_size),
        mutation_operator=PolynomialMutation(),
        crossover_operator=SimulatedBinaryCrossover(),
        selection_operator=ParetoCrowdingSelection(n_selected=pop_size),
    )
    with pytest.raises(
        AttributeError,
        match='The objective function is not defined, please define it first',
    ):
        nsga2.minimize()


@pytest.mark.parametrize(('object_function', 'n_obj'), [(single_objective_problem, 1), (zdt1_problem, 2)])
def test_nsga2(object_function, n_obj):
    """Main test"""
    nsga2 = NSGA2_Algorithm(
        object_function=object_function,
        variable_space=variable_space,
        n_obj=n_obj,
        pop_size=pop_size,
        max_iter=max_iter,
        parent_selection_operator=RandomSelection(n_selected=pop_size),
        mutation_operator=PolynomialMutation(),
        crossover_operator=SimulatedBinaryCrossover(),
        selection_operator=ParetoCrowdingSelection(n_selected=pop_size),
    )

    nsga2.minimize()
    display_results(nsga2)


@pytest.mark.parametrize(
    ('object_function', 'n_obj'),
    [
        # (single_objective_problem, 1),
        (zdt1_problem, 2)
    ],
)
def test_nsga3(object_function, n_obj):
    """Main test"""
    nsga3 = NSGA3_Algorithm(
        object_function=object_function,
        variable_space=variable_space,
        n_obj=n_obj,
        pop_size=pop_size,
        max_iter=max_iter,
        parent_selection_operator=TournamentSelection(n_selected=pop_size),
        mutation_operator=PolynomialMutation(),
        crossover_operator=SimulatedBinaryCrossover(),
        selection_operator=ParetoRefSelection(n_selected=pop_size, n_objectives=n_obj),
    )

    nsga3.minimize()
    display_results(nsga3)


@pytest.mark.parametrize(('object_function', 'n_obj'), [(single_objective_problem, 1), (zdt1_problem, 2)])
def test_mode(object_function, n_obj):
    """Main test"""
    # Create MODE algorithm instance
    mode = MODE_Algorithm(
        object_function=object_function,
        variable_space=variable_space,
        n_obj=n_obj,
        pop_size=pop_size,
        max_iter=max_iter,
        mutation_operator=PolynomialMutation(),
        crossover_operator=BinomialCrossover(),
        select_operator=ParetoSelection(n_selected=pop_size),
    )
    # Run algorithm
    mode.minimize()
    display_results(mode)


@pytest.mark.parametrize(('object_function', 'n_obj'), [(single_objective_problem, 1), (zdt1_problem, 2)])
def test_mopso(object_function, n_obj):
    mopso = MOPSO_Algorithm(
        object_function=object_function,
        variable_space=variable_space,
        n_obj=n_obj,
        pop_size=pop_size,
        max_iter=max_iter,
        velocity_update_operator=PSOVelocityUpdate(),
        position_update_operator=PSOPositionUpdate(),
        selection_operator=ParetoCrowdingSelection(n_selected=2 * pop_size),
    )
    # Run algorithm
    mopso.minimize()
    display_results(mopso)


# MOEAD algorithm tests
@pytest.mark.parametrize(
    ('n_obj', 'pop_size', 'expected_weights'),
    [
        (2, 3, np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])),
        (3, 3, None),
        (5, 2, None),
    ],
)
def test_generate_weight_vectors(n_obj: int, pop_size: int, expected_weights: np.ndarray | None):
    weights = generate_weight_vectors(n_obj=n_obj, pop_size=pop_size)
    assert weights.shape == (pop_size, n_obj)
    assert np.allclose(np.sum(weights, axis=1), 1.0)
    assert np.all(weights >= 0)
    if expected_weights is not None:
        assert np.allclose(weights, expected_weights)


@pytest.mark.parametrize(
    ('object_function', 'n_obj', 'max_replace'), [(single_objective_problem, 1, 3), (zdt1_problem, 2, 100)]
)
def test_moead(object_function: Callable, n_obj: int, max_replace: int):
    """Test MOEA/D algorithm"""
    moead = MOEAD_Algorithm(
        object_function=object_function,
        variable_space=variable_space,
        n_obj=n_obj,
        pop_size=pop_size,
        max_iter=max_iter,
        neighborhood_size=20,
        neighbor_rate=0.9,
        max_replace=max_replace,
        crossover_operator=SimulatedBinaryCrossover(),
        mutation_operator=PolynomialMutation(),
    )
    # Run algorithm
    moead.minimize()
    display_results(moead)


if __name__ == '__main__':
    pytest.main([__file__])
