"""Tests for MODE_Algorithm implementation"""

from collections.abc import Callable
from math import comb
import inspect

import numpy as np
import pytest

from optimi_lab.intelligent_algorithm.mode_algorithm import MODE_Algorithm
from optimi_lab.intelligent_algorithm.moead_algorithm import (
    MOEAD_Algorithm,
    das_dennis_partition_count,
    generate_weight_vectors,
)
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


def three_objective_problem(x: np.ndarray) -> np.ndarray:
    """Three-objective test problem: three quadratic bowls on the same box.

    Exists so the 3-objective code paths (MOEA/D's Das-Dennis weight vectors, NSGA-III's
    reference directions) are exercised end to end rather than only at the helper level.

    Args:
        x: input matrix, shape (pop_size, n_var)

    Returns:
        objective values matrix, shape (pop_size, 3)

    """
    if x.ndim == 1:
        x = x.reshape(1, -1)
    return np.column_stack(
        [
            np.sum(x**2, axis=1),
            np.sum((x - 0.5) ** 2, axis=1),
            np.sum((x - 1.0) ** 2, axis=1),
        ]
    )


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


@pytest.mark.parametrize(
    'algorithm_class', [MODE_Algorithm, MOPSO_Algorithm, NSGA2_Algorithm, NSGA3_Algorithm]
)
def test_selection_operator_is_spelled_the_same_by_every_algorithm(algorithm_class):
    """Every algorithm takes its environmental selector as `selection_operator`.

    One spelling for one concept: MODE called it `select_operator` while its four siblings
    said `selection_operator`, so a caller could not move a selector between algorithms.
    """
    parameters = inspect.signature(algorithm_class.__init__).parameters

    assert 'selection_operator' in parameters, f'{algorithm_class.__name__} must accept `selection_operator`'
    assert 'select_operator' not in parameters, (
        f'{algorithm_class.__name__} accepts `select_operator` -- the four sibling algorithms all say '
        '`selection_operator`, and only one spelling may survive'
    )


def test_nsga3_supplies_its_own_operators_and_skips_nsga2s_defaults():
    """NSGA-III's `super(NSGA2_Algorithm, self).__init__(...)` is intentional, not a slip.

    It replaces all four operators NSGA-II's `__init__` would install, and the one NSGA-II
    would install -- `ParetoCrowdingSelection` -- is precisely the operator NSGA-III exists
    to replace, so calling that `__init__` would build a selector only to throw it away.
    Two things are asserted instead of the skip being trusted:

    - the four operators really are NSGA-III's own (`ParetoRefSelection`, not
      `ParetoCrowdingSelection`);
    - the BASE initialisation did run, i.e. the shared state NSGA-III relies on is present.
    """
    nsga3 = NSGA3_Algorithm(
        variable_space=variable_space,
        n_obj=3,
        pop_size=pop_size,
        max_iter=max_iter,
    )

    assert isinstance(nsga3._selection_operator, ParetoRefSelection)
    assert not isinstance(nsga3._selection_operator, ParetoCrowdingSelection)
    assert isinstance(nsga3._parent_selection_operator, TournamentSelection)
    assert isinstance(nsga3._mutation_operator, PolynomialMutation)
    assert isinstance(nsga3._crossover_operator, SimulatedBinaryCrossover)
    assert nsga3._n_obj == 3
    assert nsga3._pop_size == pop_size
    assert nsga3._n_var == n_var
    assert nsga3._all_outputs.shape == (0, 3)
    assert nsga3._pareto_outputs.shape == (0, 3)


def test_callback_fires_once_per_iteration():
    """`callback` is invoked exactly once per iteration.

    It is the hook a caller overrides to observe progress, so double-invoking it doubles
    every side effect the override has (two log lines, two progress-bar steps per
    iteration).
    """
    algorithm = MODE_Algorithm(
        object_function=zdt1_problem,
        variable_space=variable_space,
        n_obj=2,
        pop_size=pop_size,
        max_iter=max_iter,
    )
    calls = []
    algorithm.callback = lambda: calls.append(algorithm._id_iter)

    algorithm.minimize()

    assert calls == list(range(max_iter)), 'callback must fire once per iteration, with the iteration index'


@pytest.mark.parametrize('population', [4, 5, 9])
def test_nsga2_parent_pairs_wrap_cyclically(population: int):
    """The mating pairs are cyclic neighbours in the selection order -- including the wrap.

    Planted with a NON-trivial selection order (a reversal, so no value equals its slot) and
    asserted as a property: each individual is the first parent of exactly one pair, and its
    partner is a cyclic neighbour. The old code's wrap was a bare `parent_indices[0]` for
    the last even slot; on an odd population that IS the cyclic neighbour, but nothing in
    the code said so, so a reader could not tell it from a fallback to a fixed parent.
    """
    algorithm = NSGA2_Algorithm(
        variable_space=variable_space,
        n_obj=2,
        pop_size=population,
        max_iter=1,
    )
    order = np.arange(population)[::-1]

    pairs = algorithm._pair_parents(order)

    # The rule, written out independently of the implementation.
    expected = np.array(
        [
            (order[slot], order[(slot + 1) % population] if slot % 2 == 0 else order[(slot - 1) % population])
            for slot in range(population)
        ]
    )
    np.testing.assert_array_equal(pairs, expected)

    positions = {individual: slot for slot, individual in enumerate(order)}
    assert sorted(pairs[:, 0]) == sorted(order), 'every individual must be the first parent of exactly one pair'
    for first, second in pairs:
        gap = abs(positions[first] - positions[second])
        assert min(gap, population - gap) == 1, f'{first} and {second} are not cyclic neighbours'

    if population % 2 == 1:
        last_pair = pairs[-1]
        assert last_pair[1] == order[0], 'an odd population wraps the last slot to the first individual'
        assert last_pair[0] == order[-1]


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
        selection_operator=ParetoSelection(n_selected=pop_size),
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


@pytest.mark.parametrize('pop_size', [1, 2, 3, 6, 7, 10, 45, 100, 201, 1000, 4999])
def test_three_objective_weight_vectors_are_total(pop_size: int):
    """Every population size yields exactly that many 3-objective weight vectors.

    The 3-objective branch subsamples a Das-Dennis grid with
    `np.random.choice(len(weights), pop_size, replace=False)`, which RAISES as soon as
    `pop_size > len(weights)`. The grid is therefore built to cover the population before
    any subsampling, and this sweeps the sizes that a fixed grid cannot serve at all.
    """
    weights = generate_weight_vectors(n_obj=3, pop_size=pop_size)

    assert weights.shape == (pop_size, 3)
    assert np.allclose(np.sum(weights, axis=1), 1.0)
    assert np.all(weights >= 0)
    assert len(np.unique(weights, axis=0)) == pop_size, 'the subsample must not repeat a direction'


@pytest.mark.parametrize('pop_size', [1, 2, 3, 6, 7, 10, 45, 100, 201, 1000, 4999])
def test_das_dennis_partition_count_covers_and_is_minimal(pop_size: int):
    """The partition count covers `pop_size` points and is the smallest that does.

    This is the invariant the subsample above depends on: a grid of `h` partitions over 3
    objectives holds C(h+2, 2) points, and the old ``int(sqrt(2 * pop_size)) + 1`` estimate
    was large enough only because of arithmetic, not by construction.
    """
    h = das_dennis_partition_count(n_obj=3, pop_size=pop_size)

    assert comb(h + 2, 2) >= pop_size, f'h={h} does not cover pop_size={pop_size}'
    if h > 1:
        assert comb(h + 1, 2) < pop_size, f'h={h} is not minimal for pop_size={pop_size}'


def test_three_objective_subsampling_never_asks_for_more_than_the_grid(monkeypatch):
    """Plant the precondition of the subsample: size <= the grid it samples from.

    ``np.random.choice(len(weights), pop_size, replace=False)`` raises from inside numpy the
    moment ``pop_size`` exceeds the grid, so the grid size is asserted where it is consumed
    instead of trusted. The wrapped ``choice`` records every call, and the sweep proves the
    subsampling path really runs rather than being asserted about vacuously.
    """
    calls = []
    real_choice = np.random.choice

    def guarded_choice(a, size=None, replace=True, p=None):
        population = a if isinstance(a, int) else len(a)
        calls.append((population, size))
        assert size is None or size <= population, (
            f'subsampling {size} without replacement from a grid of {population} -- a Das-Dennis grid must '
            'cover the population it is subsampled to'
        )
        return real_choice(a, size, replace, p)

    monkeypatch.setattr(np.random, 'choice', guarded_choice)

    for pop_size in (3, 7, 45, 1000):
        assert generate_weight_vectors(n_obj=3, pop_size=pop_size).shape == (pop_size, 3)

    assert calls, 'the subsample path was never exercised, so this guard proved nothing'
    assert any(size < population for population, size in calls), 'no grid larger than the population was sampled'


@pytest.mark.parametrize(
    ('object_function', 'n_obj', 'max_replace'),
    [(single_objective_problem, 1, 3), (zdt1_problem, 2, 100), (three_objective_problem, 3, 3)],
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
