"""The operators, at the level the operator defects were measured on."""

import numpy as np
import pytest

from optimi_lab.core import Refusal
from optimi_lab.intelligent_algorithm import (
    crowding_select,
    differential_mutation,
    nsga2,
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
