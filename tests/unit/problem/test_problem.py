"""D7-D12: objectives, constraint kinds, structure, uncertainty, fidelity and the problem identity."""

from __future__ import annotations

import numpy as np
import pytest

from optimi_lab.core import SampleSpec, sample
from optimi_lab.core.errors import Refusal
from optimi_lab.problem import (
    BlackBox,
    Categorical,
    Chance,
    Explicit,
    Fidelity,
    Form,
    Level,
    Linear,
    Normal,
    Normalization,
    Objective,
    Path,
    Problem,
    Real,
    Relation,
    Sense,
    Space,
    Structure,
    Uncertainty,
    Uniform,
    When,
)

SPACE = Space((Real('x', 0.0, 1.0, unit='m'), Real('y', 0.0, 2.0, unit='m')))


def test_d7_objective_normalisation_is_fixed_and_maximize_is_negated() -> None:
    torque = Objective('torque', Sense.MAXIMIZE, unit='N m', normalization=Normalization(100.0, 50.0))
    assert torque.minimized(150.0) == -1.0
    assert Objective('loss', 'minimize').minimized(3.0) == 3.0


def test_d7_planted_control_a_zero_or_nan_scale_is_refused() -> None:
    for scale in (0.0, -1.0, float('nan')):
        with pytest.raises(Refusal, match='scale'):
            Normalization(0.0, scale)
    with pytest.raises(Refusal, match='sense'):
        Objective('loss', 'smaller')


def test_d8_constraint_kinds_measure_a_violation_in_their_scale() -> None:
    cap = BlackBox('temperature', Relation.AT_MOST, 120.0, unit='degC', scale=10.0)
    assert cap.violation(125.0) == 0.5
    assert cap.violation(100.0) == 0.0
    floor = BlackBox('torque', '>=', 200.0)
    assert floor.violation(150.0) == 0.25  # scale defaults to |bound|
    linear = Linear('budget', {'x': 1.0, 'y': 2.0}, hi=2.0)
    assert linear.violation({'x': 1.0, 'y': 1.0}) == 1.0
    assert linear.violation({'x': 0.5, 'y': 0.5}) == 0.0
    explicit = Explicit('ratio', lambda d: d['x'] / d['y'], Relation.AT_MOST, 1.0)
    assert explicit.violation({'x': 1.0, 'y': 0.5}) == pytest.approx(1.0)
    path = Path('current', 'phase_current', Relation.AT_MOST, 10.0)
    assert path.violation([1.0, 12.0, 11.0]) == pytest.approx(0.2)
    assert np.isnan(cap.violation(float('nan')))


def test_d8_planted_control_a_zero_bound_without_a_scale_and_a_bad_probability_are_refused() -> None:
    with pytest.raises(Refusal, match='scale'):
        BlackBox('margin', Relation.AT_LEAST, 0.0)
    assert BlackBox('margin', Relation.AT_LEAST, 0.0, scale=1.0).violation(-2.0) == 2.0
    for probability in (0.0, 1.0, 1.5):
        with pytest.raises(Refusal, match='probability'):
            Chance('demag', 'field', Relation.AT_LEAST, 1.0, probability=probability)
    with pytest.raises(Refusal, match='lo <= hi'):
        Linear('budget', {'x': 1.0}, lo=3.0, hi=2.0)


def test_d9_structure_names_forms_of_declared_outputs() -> None:
    structure = Structure({'loss': frozenset({Form.QUADRATIC, Form.CONVEX})})
    assert structure.has('loss', Form.CONVEX)
    assert not structure.has('loss', Form.LINEAR)
    with pytest.raises(Refusal, match="'mass'"):
        Problem(SPACE, (Objective('loss'),), structure=Structure({'mass': frozenset({Form.LINEAR})}))


def test_d10_uncertainty_samples_through_an_explicit_generator() -> None:
    uncertainty = Uncertainty((Normal('br', 1.2, 0.02, unit='T'), Uniform('temp', 20.0, 80.0, unit='degC')))
    first = uncertainty.sample(1000, np.random.default_rng(3))
    again = uncertainty.sample(1000, np.random.default_rng(3))
    assert np.array_equal(first['br'], again['br'])
    assert abs(first['br'].mean() - 1.2) < 0.005
    assert first['temp'].min() >= 20.0 and first['temp'].max() <= 80.0


def test_d10_planted_control_a_global_generator_and_a_name_clash_are_refused() -> None:
    uncertainty = Uncertainty((Normal('br', 1.2, 0.02),))
    with pytest.raises(Refusal, match='Generator'):
        uncertainty.sample(10, None)  # type: ignore[arg-type]
    with pytest.raises(Refusal, match='std'):
        Normal('br', 1.2, 0.0)
    with pytest.raises(Refusal, match="'x'"):
        Problem(SPACE, (Objective('loss'),), uncertainty=Uncertainty((Normal('x', 0.0, 1.0),)))


def test_d11_fidelity_levels_rise_in_cost_and_the_last_is_the_truth() -> None:
    fidelity = Fidelity((Level('mec', 1.0), Level('fea', 30.0), Level('reference', 600.0)))
    assert fidelity.truth.name == 'reference'
    assert fidelity.index('fea') == 1
    with pytest.raises(Refusal, match='increasing'):
        Fidelity((Level('fea', 30.0), Level('mec', 1.0)))
    with pytest.raises(Refusal, match="'tet'"):
        fidelity.index('tet')


def test_d12_problem_checks_its_declarations_against_each_other() -> None:
    problem = Problem(
        SPACE,
        (Objective('loss'), Objective('torque', Sense.MAXIMIZE)),
        constraints=(BlackBox('temp', '<=', 120.0), Linear('budget', {'x': 1.0}, hi=0.8)),
    )
    assert problem.outputs == ('loss', 'torque', 'temp')
    with pytest.raises(Refusal, match="'z'"):
        Problem(SPACE, (Objective('loss'),), constraints=(Linear('budget', {'z': 1.0}, hi=1.0),))
    with pytest.raises(Refusal, match='Uncertainty'):
        Problem(SPACE, (Objective('loss'),), constraints=(Chance('d', 'field', '>=', 1.0, probability=0.99),))
    with pytest.raises(Refusal, match='unique'):
        Problem(SPACE, (Objective('loss'),), constraints=(BlackBox('loss', '<=', 1.0),))
    with pytest.raises(Refusal, match='at least one'):
        Problem(SPACE, ())


def test_d12_linear_constraint_refuses_a_categorical_variable() -> None:
    space = Space((Categorical('c', ('a', 'b')), Real('x', 0.0, 1.0, active_if=When('c', ('a',)))))
    with pytest.raises(Refusal, match='numeric'):
        Problem(space, (Objective('loss'),), constraints=(Linear('l', {'c': 1.0}, hi=1.0),))


def test_a_natural_sample_decodes_the_unit_sample_and_keeps_a_natural_anchor_as_row_zero() -> None:
    space = Space((Real('a', 0.0, 1.0), Real('b', 2.0, 4.0)))
    drawn = space.natural_sample(SampleSpec(n_samples=4, seed=7))
    unit = sample(2, SampleSpec(n_samples=4, seed=7))
    np.testing.assert_allclose(drawn, np.column_stack([unit[:, 0], 2.0 + 2.0 * unit[:, 1]]))
    anchored = space.natural_sample(SampleSpec(n_samples=3, seed=7, anchor=(0.25, 3.5)))
    np.testing.assert_allclose(anchored[0], [0.25, 3.5])
