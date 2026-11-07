"""How each solver takes its first batch, and the driver guards built on it."""

from __future__ import annotations

import numpy as np
import pytest

from optimi_lab.core import Refusal, SampleKind, SampleSpec
from optimi_lab.problem import Categorical, Objective, Problem, Real, Space, When
from optimi_lab.solver import (
    METHODS,
    SEEDING,
    SEQUENTIAL,
    SOLVERS,
    Descriptor,
    Seeding,
    batch_bound,
    descriptors_of,
    require_seedable,
    search_embedding,
)

PROBLEM = Problem(Space((Real('x', 0.0, 2.0), Real('y', 0.0, 1.0))), (Objective('f'),))
LATIN = SampleSpec(n_samples=8, kind=SampleKind.LATIN_HYPERCUBE)


def test_every_seeded_solver_is_registered_and_only_algebraic_ones_lack_a_row() -> None:
    assert set(SEEDING) <= set(SOLVERS)
    assert set(SOLVERS) - set(SEEDING) == {'lp', 'milp', 'qp', 'cp', 'dynamic_programming', 'benders', 'admm'}
    assert set(SEEDING) >= SEQUENTIAL
    assert {SEEDING[name] for name in METHODS} == {Seeding.SLOPES}
    assert not Seeding.POPULATION.draws_own and Seeding.MODEL.draws_own


def test_a_sequential_solver_is_bounded_by_evaluations() -> None:
    assert batch_bound('nsga2', 10, 4) == 5
    assert batch_bound('nelder_mead', 10, 4) == 50
    with pytest.raises(Refusal, match='takes no first batch'):
        batch_bound('lp', 10, 4)


def test_a_declared_first_batch_a_solver_would_replace_is_refused() -> None:
    rows = np.full((3, 2), 0.5)
    require_seedable('nsga2', rows)
    require_seedable('bo', LATIN)
    require_seedable('bo', None)
    require_seedable('nelder_mead', rows[:1])
    require_seedable('lbfgsb', rows[:1])
    for name, initial in (
        ('bo', rows),
        ('cmaes', SampleSpec(n_samples=8, kind=SampleKind.SOBOL)),
        ('nelder_mead', rows),
        ('bo', SampleSpec(n_samples=8, kind=SampleKind.LATIN_HYPERCUBE, anchor=(0.5, 0.5))),
    ):
        with pytest.raises(Refusal, match='draws its own first batch'):
            require_seedable(name, initial)


def test_only_a_conditional_space_is_searched_through_an_embedding() -> None:
    assert search_embedding(PROBLEM) is None
    conditional = Problem(
        Space((Categorical('kind', ('a', 'b')), Real('r', 0.0, 1.0, active_if=When('kind', ('a',))))), (Objective('f'),)
    )
    assert search_embedding(conditional) is not None


def test_map_elites_descriptors_are_declared_outputs_or_variables() -> None:
    assert descriptors_of(PROBLEM, {'f': [0.0, 1.0, 4], 'x': [5]}) == (
        Descriptor('f', 0.0, 1.0, 4),
        Descriptor('x', 0.0, 2.0, 5),
    )
    for declared in ({}, {'z': [0.0, 1.0, 3]}, {'f': [3]}):
        with pytest.raises(Refusal):
            descriptors_of(PROBLEM, declared)
