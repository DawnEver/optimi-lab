"""D79, D100, D101 -- incremental sorting and the relaxed dominance relations, each against a definition.

The oracle for :func:`incremental_sort` is the fast non-dominated sort already in ``core`` and the
brute-force definition of a front; the relaxed relations are checked against their definitions on
hand-built pairs, objective units being whatever the minimised frame carries (here dimensionless).
"""

import numpy as np
import pytest

from optimi_lab.core import (
    Refusal,
    epsilon_dominates,
    incremental_sort,
    non_dominated_sorting,
    tolerance_dominates,
)


def _brute_fronts(values: np.ndarray) -> list[set[int]]:
    left, fronts = set(range(len(values))), []
    while left:
        front = {
            i for i in left if not any(np.all(values[j] <= values[i]) and np.any(values[j] < values[i]) for j in left)
        }
        fronts.append(front)
        left -= front
    return fronts


@pytest.mark.parametrize('seed', range(6))
@pytest.mark.parametrize('n_obj', [2, 3, 5])
def test_incremental_sort_is_the_brute_force_front_decomposition(seed, n_obj):
    values = np.random.default_rng(seed).integers(0, 6, (60, n_obj)).astype(float)
    ours = [set(front.tolist()) for front in incremental_sort(values)]
    assert ours == _brute_fronts(values)
    assert ours == [set(f.tolist()) for f in non_dominated_sorting(values, only_first_front=False)]


def test_incremental_sort_refuses_a_nan_like_every_sort():
    with pytest.raises(Refusal, match='must be finite'):
        incremental_sort(np.array([[0.0, np.nan], [1.0, 1.0]]))


def test_epsilon_dominance_is_dominance_after_an_additive_shift():
    a, b = np.array([1.05, 2.0]), np.array([1.0, 2.5])
    assert not epsilon_dominates(a, b, 0.0)
    assert epsilon_dominates(a, b, 0.1)
    assert not epsilon_dominates(a, b, np.array([0.01, 0.1])), 'per-objective epsilon applies per column'


def test_tolerance_dominance_counts_a_gap_inside_the_tolerance_as_a_tie():
    a, b = np.array([1.0, 2.0]), np.array([1.001, 2.5])
    assert tolerance_dominates(a, b, 0.0)
    assert not tolerance_dominates(np.array([1.0, 2.0]), np.array([1.001, 2.0005]), 0.01), 'every gap is a tie'
    assert tolerance_dominates(np.array([1.0, 2.0]), np.array([1.001, 2.5]), 0.01)
    assert not tolerance_dominates(np.array([1.002, 2.0]), np.array([1.0, 2.5]), 0.001), 'worse beyond tolerance'


def test_a_negative_relaxation_is_refused():
    with pytest.raises(Refusal, match='non-negative'):
        epsilon_dominates(np.zeros(2), np.ones(2), -0.1)
    with pytest.raises(Refusal, match='non-negative'):
        tolerance_dominates(np.zeros(2), np.ones(2), -0.1)
