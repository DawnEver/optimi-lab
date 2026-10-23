"""A unit-cube map must preserve declared endpoints and produce valid native Actions."""

from __future__ import annotations

import numpy as np
import pytest

from optimi_lab import Action, Capability, Information, Objective, Problem, Real, Space


@pytest.mark.parametrize(('lo', 'hi'), [(0.001, 0.01), (-0.3, 0.1), (1e-200, 1e-199)])
@pytest.mark.parametrize('unit', [0.0, 1.0])
def test_linear_endpoints_are_the_declared_values_exactly(lo: float, hi: float, unit: float) -> None:
    variable = Real('x', lo, hi)
    decoded = variable.decode(np.array([unit]))
    assert decoded == (lo if unit == 0.0 else hi)
    assert variable.contains(decoded)


def test_a_decoded_upper_endpoint_is_a_valid_action_not_an_out_of_bounds_design() -> None:
    space = Space((Real('width', 0.001, 0.01, unit='m'),))
    problem = Problem(space, (Objective('response'),))
    capability = Capability(('response',), frozenset({Information.VALUES}), version='closed-interval-test')
    action = Action.of(problem, capability, space.decode(np.array([1.0])))
    assert action.design['width'] == 0.01


@pytest.mark.parametrize(('unit', 'expected'), [(0.25, -5e307), (0.5, 0.0), (0.75, 5e307)])
def test_a_finite_interval_does_not_overflow_its_intermediate_width(unit: float, expected: float) -> None:
    variable = Real('x', -1e308, 1e308)
    decoded = variable.decode(np.array([unit]))
    assert decoded == pytest.approx(expected)
    assert variable.contains(decoded)


@pytest.mark.parametrize(
    ('value', 'expected'),
    [(-1e308, 0.0), (-5e307, 0.25), (0.0, 0.5), (5e307, 0.75), (1e308, 1.0)],
)
def test_the_inverse_map_uses_the_same_finite_interval(value: float, expected: float) -> None:
    variable = Real('x', -1e308, 1e308)
    encoded = variable.encode(value)
    assert encoded.tolist() == pytest.approx([expected])
    assert variable.decode(encoded) == pytest.approx(value)


@pytest.mark.parametrize('unit', [0.0, 1.0])
def test_logarithmic_endpoints_are_exact_without_a_log_exp_roundtrip(unit: float) -> None:
    variable = Real('frequency', 0.01, 0.3, scale='log')
    assert variable.decode(np.array([unit])) == (variable.lo if unit == 0.0 else variable.hi)


def test_existing_interior_and_unit_projection_semantics_are_unchanged() -> None:
    variable = Real('x', 2.0, 4.0)
    assert variable.decode(np.array([0.25])) == 2.5
    assert variable.decode(np.array([-0.5])) == variable.lo
    assert variable.decode(np.array([1.5])) == variable.hi
