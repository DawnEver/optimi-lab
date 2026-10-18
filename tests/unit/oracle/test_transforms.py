"""D17-D18: finite differences, and the penalise and embed transforms."""

from __future__ import annotations

import asyncio
import itertools

import numpy as np
import pytest
from lab_commons.outcomes import Cell, Status

from optimi_lab.core.errors import Refusal
from optimi_lab.oracle import (
    Action,
    Capability,
    Information,
    Penalty,
    Result,
    embed,
    finite_difference,
    from_function,
    with_finite_differences,
)
from optimi_lab.problem import (
    BlackBox,
    Categorical,
    Chance,
    Integer,
    Normal,
    Normalization,
    Objective,
    Problem,
    Real,
    Sense,
    Space,
    Uncertainty,
)


def _f(x: np.ndarray) -> np.ndarray:
    return np.array([np.sin(x[0]) * x[1] ** 2, np.exp(x[0] - x[1])])


def _exact(x: np.ndarray) -> np.ndarray:
    return np.array([
        [np.cos(x[0]) * x[1] ** 2, 2 * np.sin(x[0]) * x[1]],
        [np.exp(x[0] - x[1]), -np.exp(x[0] - x[1])],
    ])


@pytest.mark.parametrize(('scheme', 'tolerance'), [('forward', 1e-6), ('central', 1e-9), ('complex', 1e-14)])
def test_d17_each_scheme_reaches_its_order(scheme: str, tolerance: float) -> None:
    x = np.array([0.7, 1.3])
    assert np.max(np.abs(finite_difference(_f, x, scheme=scheme) - _exact(x))) < tolerance


def test_d17_planted_control_the_forward_scheme_is_not_central_accurate() -> None:
    """The order claim is measured: forward at the central tolerance must FAIL, else the test is blind."""
    x = np.array([0.7, 1.3])
    assert np.max(np.abs(finite_difference(_f, x, scheme='forward') - _exact(x))) > 1e-9
    with pytest.raises(Refusal, match='scheme'):
        finite_difference(_f, x, scheme='backward')


PROBLEM = Problem(Space((Real('x', 0.0, 2.0), Real('y', 0.0, 4.0))), (Objective('f'),))
CAPABILITY = Capability(outputs=('f',), version='quad')


def _quadratic(action: Action) -> dict:
    return {'f': action.design['x'] ** 2 + 3 * action.design['y']}


def test_d17_the_oracle_transform_returns_a_unit_cube_gradient() -> None:
    oracle = with_finite_differences(from_function(_quadratic, CAPABILITY), PROBLEM, scheme='central')
    assert Information.VJP in oracle.capability.provides
    action = Action.of(
        PROBLEM, oracle.capability, {'x': 1.0, 'y': 1.0}, want=frozenset({Information.VALUES, Information.VJP})
    )
    result = asyncio.run(oracle(action))
    # d f / d u = d f / d x * span: (2 x * 2, 3 * 4)
    assert result.gradient['f'].tolist() == pytest.approx([4.0, 12.0], rel=1e-6, abs=0.0)
    assert result.cell.values['f'] == pytest.approx(4.0)


def test_d17_planted_control_a_discrete_variable_and_a_complex_oracle_step_are_refused() -> None:
    discrete = Problem(Space((Integer('n', 0, 4),)), (Objective('f'),))
    with pytest.raises(Refusal, match="'n'"):
        with_finite_differences(from_function(_quadratic, CAPABILITY), discrete)
    with pytest.raises(Refusal, match='complex'):
        with_finite_differences(from_function(_quadratic, CAPABILITY), PROBLEM, scheme='complex')


CONSTRAINED = Problem(
    Space((Real('x', 0.0, 1.0),)),
    (Objective('cost', normalization=Normalization(0.0, 10.0)), Objective('gain', Sense.MAXIMIZE)),
    constraints=(BlackBox('temp', '<=', 100.0, scale=10.0),),
)


def test_d18_penalise_worsens_each_objective_by_the_scaled_violation() -> None:
    penalty = Penalty(CONSTRAINED, weight=2.0)
    assert penalty.problem.constraints == ()
    action = Action.of(CONSTRAINED, Capability(outputs=('cost', 'gain', 'temp')), {'x': 0.5})
    hot = Result(action, Cell(Status.OK, values={'cost': 1.0, 'gain': 5.0, 'temp': 115.0}))
    penalised = penalty.apply(hot)
    # violation 1.5, weight 2 -> 3 normalised units: cost scale 10 -> +30, gain scale 1 -> -3
    assert penalised.cell.values['cost'] == pytest.approx(31.0)
    assert penalised.cell.values['gain'] == pytest.approx(2.0)
    cool = Result(action, Cell(Status.OK, values={'cost': 1.0, 'gain': 5.0, 'temp': 90.0}))
    assert dict(penalty.apply(cool).cell.values) == dict(cool.cell.values)


def test_d18_planted_control_a_chance_constraint_cannot_be_penalised_from_one_sample() -> None:
    chance = Problem(
        Space((Real('x', 0.0, 1.0),)),
        (Objective('f'),),
        constraints=(Chance('demag', 'field', '>=', 1.0, probability=0.95),),
        uncertainty=Uncertainty((Normal('br', 1.2, 0.02),)),
    )
    with pytest.raises(Refusal, match='Chance'):
        Penalty(chance, weight=1.0)


def test_d18_embed_gives_an_order_free_metric_b0_11() -> None:
    def distances(levels: tuple[str, ...]) -> dict:
        space = embed(Space((Categorical('coolant', levels),)))
        return {
            (a, b): float(np.linalg.norm(space.encode({'coolant': a}) - space.encode({'coolant': b})))
            for a, b in itertools.combinations(sorted(levels), 2)
        }

    assert distances(('air', 'water', 'oil')) == distances(('oil', 'air', 'water'))
    assert len(set(distances(('air', 'water', 'oil')).values())) == 1


def test_d18_planted_control_an_unembedded_categorical_has_no_distance_b0_11() -> None:
    with pytest.raises(Refusal, match='embedding='):
        Space((Categorical('coolant', ('air', 'water', 'oil')),)).encode({'coolant': 'air'})
