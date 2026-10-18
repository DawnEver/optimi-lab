"""D73, D76, D77 -- stratified sampling of discrete variables, discrete and permutation variation, repair.

Every check is against a definition: a stratified column holds each level ``n // L`` or ``n // L + 1``
times; a snapped point decodes to the level it sits on; an ordered mutation moves one step on the
level ladder; a permutation operator returns a permutation; a repaired point is inside the cube.
"""

import numpy as np
import pytest

from optimi_lab.core import Refusal
from optimi_lab.operator import (
    Repair,
    discrete_mutation,
    inversion_mutation,
    lattice,
    order_crossover,
    repair,
    snap,
    stratified,
    swap_mutation,
)
from optimi_lab.problem import Categorical, Integer, Real, Space

SPACE = Space((
    Real('r', 0.0, 1.0),
    Integer('poles', 2, 12, step=2),
    Categorical('coolant', ('air', 'water', 'oil'), embedding=((0.0,), (0.5,), (1.0,))),
))
RNG = np.random.default_rng


def test_a_stratified_sample_balances_every_discrete_level_and_strata_the_reals():
    """D73: 30 rows over 6 pole counts is 5 each; over 3 coolants, 10 each; the Real is a Latin column."""
    points = stratified(SPACE, 30, RNG(0))
    designs = [SPACE.decode(row) for row in points]
    poles = [d['poles'] for d in designs]
    assert {p: poles.count(p) for p in set(poles)} == dict.fromkeys(range(2, 13, 2), 5)
    coolants = [d['coolant'] for d in designs]
    assert sorted(coolants.count(c) for c in ('air', 'water', 'oil')) == [10, 10, 10]
    assert sorted(np.floor(points[:, 0] * 30).astype(int)) == list(range(30))


def test_a_stratified_count_that_does_not_divide_differs_by_at_most_one():
    poles = [SPACE.decode(row)['poles'] for row in stratified(SPACE, 20, RNG(1))]
    counts = [poles.count(p) for p in range(2, 13, 2)]
    assert max(counts) - min(counts) <= 1 and sum(counts) == 20


def test_snap_puts_every_discrete_column_on_a_level_and_leaves_reals_alone():
    raw = RNG(2).random((25, SPACE.width))
    snapped = snap(raw, lattice(SPACE))
    assert np.array_equal(snapped[:, 0], raw[:, 0])
    for row in snapped:
        assert np.allclose(row, SPACE.encode(SPACE.decode(row))), 'a snapped point is its own decode'


def test_an_ordered_discrete_mutation_moves_one_step_on_the_level_ladder():
    start = np.tile(SPACE.encode({'r': 0.5, 'poles': 6, 'coolant': 'water'}), (200, 1))
    moved = discrete_mutation(start, lattice(SPACE), mutation_rate=1.0, rng=RNG(3))
    poles = {SPACE.decode(row)['poles'] for row in moved}
    assert poles == {4, 8}, 'an integer moves to a neighbouring level, never further'
    coolants = {SPACE.decode(row)['coolant'] for row in moved}
    assert coolants == {'air', 'oil'}, 'a category resets to ANOTHER level, any of them'
    assert np.array_equal(moved[:, 0], start[:, 0]), 'a Real column is not a discrete mutation subject'


def test_a_zero_rate_discrete_mutation_is_the_identity():
    start = snap(RNG(4).random((10, SPACE.width)), lattice(SPACE))
    assert np.array_equal(discrete_mutation(start, lattice(SPACE), mutation_rate=0.0, rng=RNG(0)), start)


def test_order_crossover_keeps_a_parent_segment_and_returns_permutations():
    first = np.array([np.arange(8)] * 20)
    second = np.array([RNG(5).permutation(8) for _ in range(20)])
    children = order_crossover(first, second, rng=RNG(6))
    for child, other in zip(children, second, strict=True):
        assert sorted(child) == list(range(8))
        kept = child == np.arange(8)
        assert kept.any(), 'a segment of the first parent survives in place'
        assert [g for g in child if g not in np.arange(8)[kept]] == [g for g in other if g not in np.arange(8)[kept]]


@pytest.mark.parametrize('mutation', [swap_mutation, inversion_mutation])
def test_a_permutation_mutation_returns_permutations(mutation):
    perms = np.array([RNG(s).permutation(9) for s in range(30)])
    mutated = mutation(perms, mutation_rate=1.0, rng=RNG(7))
    assert all(sorted(row) == list(range(9)) for row in mutated)
    assert not np.array_equal(mutated, perms)


def test_a_permutation_operator_refuses_a_row_that_is_not_a_permutation():
    with pytest.raises(Refusal, match='permutation'):
        swap_mutation(np.array([[0, 0, 1]]), mutation_rate=1.0, rng=RNG(0))


@pytest.mark.parametrize('how', list(Repair))
def test_every_repair_returns_a_point_in_the_unit_cube(how):
    outside = np.array([[-0.3, 1.4, 0.5], [2.6, -1.2, 0.2]])
    repaired = repair(outside, how, rng=RNG(8))
    assert np.all((repaired >= 0.0) & (repaired <= 1.0))
    assert np.array_equal(repaired[:, 2], outside[:, 2]), 'a coordinate already inside is left alone'


def test_reflection_mirrors_at_the_violated_bound():
    assert np.allclose(repair(np.array([[-0.3, 1.4]]), Repair.REFLECT, rng=RNG(0)), [[0.3, 0.6]])
    assert repair(np.array([[-0.3, 1.4]]), Repair.CLIP, rng=RNG(0)).tolist() == [[0.0, 1.0]]
