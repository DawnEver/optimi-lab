"""P9: an oracle's declared gradients are checked against finite differences of its own values."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest
from lab_commons.outcomes import Cell, Status

from optimi_lab.core.errors import Refusal
from optimi_lab.oracle import Action, Capability, Information, Result, check_gradient, from_function
from optimi_lab.problem import Objective, Problem, Real, Scale, Space

SPACE = Space((Real('a', 1.0, 3.0), Real('b', 0.1, 10.0, scale=Scale.LOG)))
PROBLEM = Problem(SPACE, (Objective('f'),))
DECLARING = Capability(outputs=('f',), provides=frozenset({Information.VALUES, Information.VJP}))


def _oracle(planted: float = 0.0):
    """``f = a^2 b``; the gradient in UNIT coordinates (the chain rule through each variable's decode)."""

    def evaluate(action: Action) -> Result:
        a, b = action.design['a'], action.design['b']
        values = {'f': a * a * b}
        if Information.VJP not in action.want:
            return values
        da_du, db_du = 2.0, b * np.log(100.0)
        gradient = np.array([2 * a * b * da_du + planted, a * a * db_du])
        return Result(action, Cell(Status.OK, values=values), gradient={'f': gradient})

    return from_function(evaluate, DECLARING)


def test_p9_a_correct_declared_gradient_agrees_with_finite_differences() -> None:
    errors = asyncio.run(check_gradient(_oracle(), PROBLEM, {'a': 2.0, 'b': 1.5}))
    assert set(errors) == {'f'} and errors['f'] <= 1e-6


def test_p9_planted_control_a_wrong_declared_gradient_is_named() -> None:
    errors = asyncio.run(check_gradient(_oracle(planted=1e-3), PROBLEM, {'a': 2.0, 'b': 1.5}))
    assert errors['f'] > 1e-6


def test_p9_an_oracle_that_declares_no_gradient_is_refused() -> None:
    values_only = from_function(lambda _: {'f': 1.0}, Capability(outputs=('f',)))
    with pytest.raises(Refusal, match='declares no gradient'):
        asyncio.run(check_gradient(values_only, PROBLEM, {'a': 2.0, 'b': 1.5}))


@pytest.mark.parametrize('scheme', ['central', 'forward'])
@pytest.mark.parametrize('a', [1.0, 3.0])
def test_p9_differences_at_a_bound_stay_inside_the_space(scheme: str, a: float) -> None:
    """A step past a bound would be clipped by the decode map and halve the slope: it turns one-sided instead."""
    errors = asyncio.run(check_gradient(_oracle(), PROBLEM, {'a': a, 'b': 1.5}, scheme=scheme))
    assert errors['f'] <= 1e-5
