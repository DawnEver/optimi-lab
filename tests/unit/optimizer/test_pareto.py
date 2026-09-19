"""Dominance, the fronts it sorts into, and the reference-grid size -- on hand-built cases.

THE SECOND DISPLACEMENT, and it is the same one `test_benchmarks.py` names. `dominates`,
`non_dominated_sorting` and `partition_count` are on the package's TOP-LEVEL `__all__` and
`non_dominated_sorting` is what `Record.pareto()` is built on, so it decides the front of every
run this library performs. Measured 2026-09-19 on trunk: `grep -rw` over `tests/` returned ZERO
hits for each of the three, and zero for `crowding_distance` besides. The hand-built cases that
did exist lived in the consumer (`motronics/tests/unit/pareto/test_nsga2_benchmark.py`), so the
dominance core of this library was checked only by a repository that imports it. They are ported
here, where the code is.

THE NaN CASE IS THE ONE THIS FILE EXISTS FOR. `core/pareto.py`'s module docstring spends a
paragraph on it -- "`NaN <= x` is False for every comparison, so a point holding a NaN is
dominated by nothing and dominates nothing: it lands on the first front by arithmetic accident" --
and nothing called the guard that prevents it. A scope claim needs a test that PLANTS the thing in
the region and calls the REAL guard, so :func:`test_a_planted_nan_is_refused_rather_than_sorted`
does exactly that, and asserts the arithmetic accident the refusal forecloses.
"""

from math import comb

import numpy as np
import pytest

from optimi_lab import Refusal, dominates, non_dominated_sorting, partition_count
from optimi_lab.intelligent_algorithm import crowding_distance

#: Four points chosen so that every front has a different size: `(0, 0)` dominates everything,
#: `(1, 1)` dominates the two trade-off points, and those two dominate nothing and each other.
LADDER = np.array([[1.0, 2.0], [2.0, 1.0], [1.0, 1.0], [0.0, 0.0]])


# --- Dominance ---------------------------------------------------------------------------------


def test_dominance_is_no_worse_everywhere_and_better_somewhere():
    assert bool(dominates(np.array([1.0, 2.0]), np.array([2.0, 3.0]))) is True
    # A trade-off: better on one objective, worse on the other. Neither dominates.
    assert bool(dominates(np.array([1.0, 2.0]), np.array([2.0, 1.0]))) is False
    # Equality is not STRICT domination, so a point does not dominate itself.
    assert bool(dominates(np.array([1.0, 2.0]), np.array([1.0, 2.0]))) is False
    # Equal on one objective and better on the other IS domination.
    assert bool(dominates(np.array([1.0, 1.0]), np.array([1.0, 2.0]))) is True


def test_dominance_broadcasts_to_every_pair_at_once():
    """The vectorised form is what the sorter uses; a per-pair loop would be a second definition."""
    pairwise = dominates(LADDER[:, None, :], LADDER[None, :, :])
    assert pairwise.shape == (4, 4)
    expected = np.array([[bool(dominates(a, b)) for b in LADDER] for a in LADDER])
    np.testing.assert_array_equal(pairwise, expected)


# --- Sorting -----------------------------------------------------------------------------------


def test_the_first_front_is_what_nothing_dominates():
    np.testing.assert_array_equal(non_dominated_sorting(LADDER), [3])


def test_every_front_comes_back_in_dominance_order():
    fronts = non_dominated_sorting(LADDER, only_first_front=False)
    assert [list(front) for front in fronts] == [[3], [2], [0, 1]]


def test_an_empty_population_sorts_to_an_empty_front():
    """Zero points is a legal answer, and it must not be confused with a refusal."""
    assert non_dominated_sorting(np.zeros((0, 2))).size == 0


def test_a_planted_nan_is_refused_rather_than_sorted():
    """PLANT the arithmetic accident the module docstring describes and call the REAL guard.

    The second assertion is what makes the first one mean something: `dominates` really does
    report that a NaN row is dominated by nothing, so WITHOUT the refusal that row reaches the
    first front of every run it appears in -- a point that did not compute, returned as the best
    result. The guard is the only thing standing between those two facts.
    """
    planted = np.array([[0.0, 0.0], [1.0, float('nan')]])
    assert not bool(dominates(planted[0], planted[1])), 'the accident this guard forecloses has changed shape'
    with pytest.raises(Refusal, match='must be finite'):
        non_dominated_sorting(planted)


@pytest.mark.parametrize('shape', [(2,), (3, 0)])
def test_a_matrix_that_is_not_points_by_objectives_is_refused(shape):
    with pytest.raises(Refusal, match='2-D with one column per objective'):
        non_dominated_sorting(np.zeros(shape))


# --- The Das-Dennis grid -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ('n_obj', 'n_points', 'partitions'),
    [(2, 5, 4), (2, 50, 49), (3, 50, 9), (3, 91, 12), (5, 100, 5)],
)
def test_the_partition_count_is_the_smallest_grid_that_covers(n_obj, n_points, partitions):
    """MEASURED 2026-09-19, and each row is checked against the closed form it claims to invert."""
    assert partition_count(n_obj, n_points) == partitions
    assert comb(partitions + n_obj - 1, n_obj - 1) >= n_points, 'the grid returned does not cover'
    assert comb(partitions + n_obj - 2, n_obj - 1) < n_points, 'a smaller grid would have covered, so it is not least'


def test_one_objective_is_refused_rather_than_searched_forever():
    """`comb(h, 0)` is 1 for every `h`, so the old loop incremented without end. It is named now."""
    with pytest.raises(Refusal, match='at least 2 objectives'):
        partition_count(1, 2)


def test_a_grid_for_no_points_is_refused():
    with pytest.raises(Refusal, match='must be greater than 0'):
        partition_count(2, 0)


# --- Crowding ----------------------------------------------------------------------------------


def test_crowding_puts_the_boundaries_at_infinity_and_measures_the_interior():
    """The tie-break NSGA-II selects on: boundaries are never crowded out, interiors are ranked."""
    np.testing.assert_allclose(crowding_distance(LADDER), [np.inf, np.inf, 1.0, np.inf])


def test_a_population_below_three_is_all_boundary():
    np.testing.assert_array_equal(crowding_distance(np.array([[0.0, 1.0], [1.0, 0.0]])), [np.inf, np.inf])
