"""D12: a problem and each design under it have an ``outcomes.Identity``."""

from __future__ import annotations

from lab_commons.outcomes import Identity

from optimi_lab.problem import Categorical, Explicit, Fidelity, Level, Objective, Problem, Real, Relation, Space, When


def _ratio(design: dict) -> float:
    return design['x']


def _problem(*, bound: float = 1.0) -> Problem:
    space = Space((
        Categorical('magnet', ('none', 'bar')),
        Real('x', 0.0, 1.0, active_if=When('magnet', ('bar',))),
    ))
    return Problem(
        space,
        (Objective('loss'),),
        constraints=(Explicit('ratio', _ratio, Relation.AT_MOST, bound),),
        fidelity=Fidelity((Level('mec', 1.0), Level('fea', 10.0))),
    )


def test_d12_the_problem_identity_is_an_outcomes_identity_of_its_declaration() -> None:
    assert isinstance(_problem().identity, Identity)
    assert _problem().identity == _problem().identity
    assert _problem().identity != _problem(bound=2.0).identity


def test_d12_planted_control_an_inactive_value_does_not_split_a_design_and_floats_are_exact() -> None:
    problem = _problem()
    off = problem.identify({'magnet': 'none'})
    assert problem.identify({'magnet': 'none', 'x': 0.3}) == off
    on = problem.identify({'magnet': 'bar', 'x': 0.1 + 0.2})
    assert on != problem.identify({'magnet': 'bar', 'x': 0.3})
    assert problem.identify({'magnet': 'bar', 'x': 0.3}, 'mec') != problem.identify({'magnet': 'bar', 'x': 0.3}, 'fea')
