"""D27: a solver's ``Requires`` matched against a problem and an oracle -- eligible, transformed or refused."""

from __future__ import annotations

import pytest

from optimi_lab.core.errors import Refusal
from optimi_lab.oracle import Capability, Information
from optimi_lab.problem import (
    BlackBox,
    Categorical,
    Chance,
    Form,
    Integer,
    Normal,
    Objective,
    Problem,
    Real,
    Space,
    Structure,
    Uncertainty,
)
from optimi_lab.solver import SOLVERS, Eligible, Ineligible, MultiObjective, NeedsTransform, Requires, match

REAL = Space((Real('x', 0.0, 1.0), Real('y', 0.0, 1.0)))
TWO = (Objective('f'), Objective('g'))
VALUES = Capability(outputs=('f', 'g', 't'))
GRADIENT = Requires(information=frozenset({Information.VALUES, Information.VJP}), objectives=MultiObjective.SINGLE)


def test_d27_a_values_only_population_solver_is_eligible_on_a_plain_problem() -> None:
    requires = Requires(objectives=MultiObjective.NATIVE_PARETO, constraints=frozenset({BlackBox}))
    assert match(requires, Problem(REAL, TWO), VALUES) == Eligible()


def test_d27_missing_gradients_need_finite_differences_with_their_cost() -> None:
    verdict = match(GRADIENT, Problem(REAL, (Objective('f'),)), VALUES)
    assert verdict == NeedsTransform(('finite_differences',), evaluations_per_action=1 + 2 * REAL.width)


def test_d27_planted_control_finite_differences_are_refused_over_noise_or_a_discrete_variable() -> None:
    noisy = Capability(outputs=('f',), provides=frozenset({Information.VALUES, Information.STOCHASTIC}))
    assert isinstance(match(GRADIENT, Problem(REAL, (Objective('f'),)), noisy), Ineligible)
    discrete = Problem(Space((Integer('n', 0, 4),)), (Objective('f'),))
    verdict = match(GRADIENT, discrete, VALUES)
    assert isinstance(verdict, Ineligible) and "'n'" in verdict.reason


def test_d27_an_unembedded_categorical_needs_embed_and_a_blackbox_constraint_needs_penalise() -> None:
    space = Space((Categorical('coolant', ('air', 'oil')), Real('x', 0.0, 1.0)))
    problem = Problem(space, TWO, constraints=(BlackBox('t', '<=', 1.0),))
    requires = Requires(objectives=MultiObjective.NATIVE_PARETO, metric=True)
    assert match(requires, problem, VALUES) == NeedsTransform(('embed', 'penalise'), evaluations_per_action=1.0)


def test_d27_b0_20_how_a_solver_handles_many_objectives_is_declared_and_single_is_refused() -> None:
    assert {m.value for m in MultiObjective} == {
        'single',
        'native_pareto',
        'weighted',
        'lexicographic',
        'external_scan',
    }
    verdict = match(Requires(objectives=MultiObjective.SINGLE), Problem(REAL, TWO), VALUES)
    assert isinstance(verdict, Ineligible) and 'scalarisation' in verdict.reason
    assert all(isinstance(r.objectives, MultiObjective) for r in SOLVERS.values())


def test_d27_chance_structure_and_uncertainty_are_refused_with_the_reason() -> None:
    uncertain = Problem(
        REAL,
        (Objective('f'),),
        constraints=(Chance('c', 't', '<=', 1.0, probability=0.9),),
        uncertainty=Uncertainty((Normal('br', 1.0, 0.1),)),
    )
    verdict = match(Requires(objectives=MultiObjective.SINGLE, constraints=frozenset({BlackBox})), uncertain, VALUES)
    assert isinstance(verdict, Ineligible) and 'Chance' in verdict.reason
    quadratic = Requires(objectives=MultiObjective.SINGLE, forms=frozenset({Form.QUADRATIC}))
    plain = Problem(REAL, (Objective('f'),))
    assert 'QUADRATIC' in match(quadratic, plain, VALUES).reason.upper()
    declared = Problem(REAL, (Objective('f'),), structure=Structure({'f': frozenset({Form.QUADRATIC})}))
    assert match(quadratic, declared, VALUES) == Eligible()


def test_d27_planted_control_an_unregistered_solver_name_is_refused_naming_the_registry() -> None:
    with pytest.raises(Refusal, match='registered'):
        Requires.of('nsga9')
