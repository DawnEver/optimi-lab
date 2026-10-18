"""The variation and selection operators (D74, D75, D78), at the level their defects were measured on."""

import numpy as np
import pytest

from optimi_lab.core import Refusal
from optimi_lab.operator import (
    DEStrategy,
    binomial_crossover,
    crowding_select,
    differential_mutation,
    exponential_crossover,
    polynomial_mutation,
    pso_position,
    pso_velocity,
    random_select,
    reference_select,
    simulated_binary_crossover,
    tournament_select,
)
from optimi_lab.oracle import Capability
from optimi_lab.problem import Objective, Problem, Real, Space
from optimi_lab.solver import nsga2

SPACE = 2  # every operator works in the unit cube of this width
VALUES = np.array([[1.0, 2.0], [2.0, 1.0], [1.0, 1.0], [0.5, 0.5], [0.0, 3.0]])


@pytest.mark.parametrize('selector', [random_select, tournament_select])
def test_the_two_parent_selectors_agree_on_more_than_the_population(selector):
    """One behaviour for one request: a refusal naming both counts, never numpy's own error.

    ``RandomSelection`` raised "Cannot take a larger sample than population" from inside
    ``np.random.choice`` and ``TournamentSelection`` silently duplicated indices instead.
    """
    with pytest.raises(Refusal, match=r'exceeds the 5 row\(s\) available'):
        selector(VALUES, 6, np.random.default_rng(0))


@pytest.mark.parametrize('selector', [crowding_select, reference_select])
def test_the_two_environmental_selectors_agree_on_more_than_the_population(selector):
    with pytest.raises(Refusal, match=r'exceeds the 5 row\(s\) available'):
        selector(VALUES, 6)


def test_a_random_selection_of_every_row_returns_every_row():
    chosen = random_select(VALUES, 5, np.random.default_rng(0))

    assert sorted(int(index) for index in chosen) == list(range(5))


@pytest.mark.parametrize('size', [2, 4])
def test_a_tournament_answers_with_indices_of_the_rows_it_was_given(size):
    """A tournament may pick one parent twice, since it selects per slot.

    A foreign index would mean the answer had stopped describing the population it was given.
    """
    chosen = tournament_select(VALUES, 4, np.random.default_rng(0), tournament_size=size)

    assert len(chosen) == 4
    assert {int(index) for index in chosen} <= set(range(5))


def test_pso_velocity_limit_is_a_fraction_of_each_variable_own_range():
    """``max_velocity_fraction`` is relative: in the unit cube, a fraction of EVERY variable's range.

    The old ``v_max`` was documented "relative to variable range" and clipped as an absolute 0.1,
    so the widest variable a particle could cross in one step was 0.1 on every problem. Encoding
    each variable onto ``[0, 1]`` (:meth:`optimi_lab.problem.Space.encode`) makes the limit a range
    fraction for a 5000-wide variable and a 1-wide one alike.
    """
    velocities = pso_velocity(
        np.zeros((1, 2)),
        np.array([[0.0, 0.0]]),
        np.array([[1.0, 1.0]]),
        None,
        SPACE,
        max_velocity_fraction=0.1,
        rng=np.random.default_rng(0),
    )
    np.testing.assert_allclose(np.abs(velocities), [[0.1, 0.1]])
    wide = Space((Real('wide', 0.0, 5000.0), Real('narrow', 0.0, 1.0)))
    step = wide.decode(np.abs(velocities[0]))
    assert step == {'wide': pytest.approx(500.0), 'narrow': pytest.approx(0.1)}


def test_pso_velocity_without_an_archive_has_no_leader():
    """An empty archive leaves the social term at zero; a leader at a distance is what moves one."""
    positions = np.array([[0.5, 0.5]])
    rng = np.random.default_rng(0)
    at_rest = pso_velocity(np.zeros_like(positions), positions, positions, np.zeros((0, 2)), SPACE, rng=rng)
    led = pso_velocity(np.zeros_like(positions), positions, positions, positions + 100.0, SPACE, rng=rng)

    np.testing.assert_allclose(at_rest, [[0.0, 0.0]])
    assert np.all(led > 0.0), 'a distant leader must pull the particle, or the assertion above is vacuous'


def test_pso_position_stays_in_the_unit_cube():
    """The clip is the only place a particle is held in the interval."""
    positions = pso_position(np.array([[0.0, 0.0]]), np.array([[1e9, -1e9]]), SPACE)

    np.testing.assert_allclose(positions, [[1.0, 0.0]])


def test_the_crossover_and_mutation_stay_inside_the_unit_cube():
    parents = np.array([[0.0, 0.0], [1.0, 1.0], [1e-4, 1e-4], [1.0, 1.0 - 1e-4]])
    children = simulated_binary_crossover(parents[:2], parents[2:], SPACE, rng=np.random.default_rng(0))
    mutated = polynomial_mutation(children, SPACE, mutation_rate=1.0, rng=np.random.default_rng(0))
    for points in (children, mutated):
        assert points.shape == (2, 2)
        assert np.all(points >= 0.0) and np.all(points <= 1.0)


def test_differential_mutation_refuses_a_population_below_its_own_floor():
    """DE/rand/1 draws three other rows; the old body copied the individual through instead."""
    with pytest.raises(Refusal, match=r'needs at least 4 rows, got 3'):
        differential_mutation(np.zeros((3, 2)), SPACE, rng=np.random.default_rng(0))


@pytest.mark.parametrize('selector', [crowding_select, reference_select])
def test_a_selection_of_every_row_returns_every_row(selector):
    np.testing.assert_array_equal(sorted(int(index) for index in selector(VALUES, 5)), list(range(5)))


def test_the_reference_selection_needs_two_objectives_and_says_so():
    with pytest.raises(Refusal, match='at least 2 objectives'):
        reference_select(VALUES[:, :1], 3)


def test_a_misspelled_setting_names_the_valid_ones():
    """The accepted names come from the operator's own signature, so a refusal cannot drift from it."""
    with pytest.raises(Refusal) as refusal:
        nsga2(
            Problem(Space((Real('x', 0.0, 1.0),)), (Objective('f'), Objective('g'))),
            Capability(outputs=('f', 'g')),
            eta_crossoverr=1.0,
        )

    assert 'eta_crossoverr' in str(refusal.value)
    assert 'eta_crossover' in str(refusal.value)


POPULATION = np.array([[0.1, 0.2], [0.4, 0.9], [0.7, 0.3], [0.2, 0.6], [0.9, 0.8], [0.5, 0.5], [0.3, 0.1]])

#: Each strategy's definition: which rows it reads, as a function of the row, best and donors.
DEFINITIONS = {
    DEStrategy.RAND_1: (3, lambda x, _i, _b, r, f: x[r[0]] + f * (x[r[1]] - x[r[2]])),
    DEStrategy.RAND_2: (5, lambda x, _i, _b, r, f: x[r[0]] + f * (x[r[1]] - x[r[2]]) + f * (x[r[3]] - x[r[4]])),
    DEStrategy.BEST_1: (2, lambda x, _i, b, r, f: x[b] + f * (x[r[0]] - x[r[1]])),
    DEStrategy.BEST_2: (4, lambda x, _i, b, r, f: x[b] + f * (x[r[0]] - x[r[1]]) + f * (x[r[2]] - x[r[3]])),
    DEStrategy.CURRENT_TO_BEST_1: (2, lambda x, i, b, r, f: x[i] + f * (x[b] - x[i]) + f * (x[r[0]] - x[r[1]])),
    DEStrategy.CURRENT_TO_RAND_1: (3, lambda x, i, _b, r, f: x[i] + f * (x[r[0]] - x[i]) + f * (x[r[1]] - x[r[2]])),
}


@pytest.mark.parametrize('strategy', list(DEStrategy))
def test_every_de_strategy_builds_its_trial_by_its_published_formula(strategy):
    """D75: a trial is reproduced by the definition for SOME distinct donors other than the row itself."""
    from itertools import permutations  # noqa: PLC0415 -- the brute-force search is this test's alone

    count, formula = DEFINITIONS[strategy]
    best = np.full(len(POPULATION), 5)
    trial = differential_mutation(
        POPULATION, 2, strategy=strategy, sizing_factor=0.3, best=best, rng=np.random.default_rng(1), clip=False
    )
    for i, row in enumerate(trial):
        others = [k for k in range(len(POPULATION)) if k != i]
        assert any(
            np.allclose(row, formula(POPULATION, i, 5, donors, 0.3)) for donors in permutations(others, count)
        ), f'{strategy} row {i} matches no donor choice'


def test_a_planted_wrong_formula_matches_no_trial():
    """The control: the brute-force search above can fail -- rand/1 trials are not best/1 trials."""
    from itertools import permutations  # noqa: PLC0415

    trial = differential_mutation(POPULATION, 2, sizing_factor=0.3, rng=np.random.default_rng(1), clip=False)
    _, best_formula = DEFINITIONS[DEStrategy.BEST_1]
    misses = [
        not any(np.allclose(row, best_formula(POPULATION, i, 5, d, 0.3)) for d in permutations(range(7), 2))
        for i, row in enumerate(trial)
    ]
    assert any(misses)


def test_a_best_strategy_without_a_best_row_is_refused():
    with pytest.raises(Refusal, match='best'):
        differential_mutation(POPULATION, 2, strategy=DEStrategy.BEST_1, rng=np.random.default_rng(0))


def test_exponential_crossover_takes_one_contiguous_run_from_the_donor():
    parent, donor = np.zeros((50, 6)), np.ones((50, 6))
    child = exponential_crossover(parent, donor, 6, crossover_rate=0.7, rng=np.random.default_rng(2))
    for row in child.astype(int):
        doubled = ''.join(map(str, row)) * 2
        assert row.sum() >= 1 and '1' * row.sum() in doubled, 'the donor genes form one cyclic run'


def test_binomial_crossover_takes_at_least_one_donor_gene():
    child = binomial_crossover(np.zeros((40, 3)), np.ones((40, 3)), 3, crossover_rate=0.0, rng=np.random.default_rng(0))
    assert np.all(child.sum(axis=1) == 1)


def test_sbx_is_symmetric_between_its_parents():
    """SBX's two children mirror each other about the parents' midpoint, so their mean IS that midpoint.

    The P8 operator returned only the child beside the LOWER parent: its mean sat below the midpoint
    and every generation drifted toward smaller coordinates.
    """
    low, high = np.full((20000, 1), 0.3), np.full((20000, 1), 0.7)
    children = simulated_binary_crossover(
        low, high, 1, crossover_rate=1.0, eta_crossover=2.0, rng=np.random.default_rng(0)
    )
    assert children.mean() == pytest.approx(0.5, abs=0.005)
