"""P9 ruling (b): a Hessian-vector product is asked WITH its direction, and the direction is identity."""

from __future__ import annotations

import numpy as np
import pytest
from lab_commons.outcomes import Cell, Status

from optimi_lab.core.errors import Refusal
from optimi_lab.oracle import Action, Capability, Information, Result
from optimi_lab.problem import Objective, Problem, Real, Space

PROBLEM = Problem(Space((Real('a', 0.0, 1.0), Real('b', 0.0, 1.0))), (Objective('f'),))
HVP = Capability(outputs=('f',), provides=frozenset({Information.VALUES, Information.VJP, Information.HVP}))
WANT = frozenset({Information.VALUES, Information.HVP})


def test_p9b_an_hvp_is_refused_without_a_direction() -> None:
    with pytest.raises(Refusal, match='direction'):
        Action.of(PROBLEM, HVP, {'a': 0.5, 'b': 0.5}, want=WANT)


def test_p9b_a_direction_has_the_space_width_and_needs_an_hvp() -> None:
    with pytest.raises(Refusal, match='width'):
        Action.of(PROBLEM, HVP, {'a': 0.5, 'b': 0.5}, want=WANT, direction=(1.0,))
    with pytest.raises(Refusal, match='direction'):
        Action.of(PROBLEM, HVP, {'a': 0.5, 'b': 0.5}, direction=(1.0, 0.0))


def test_p9b_the_direction_is_part_of_the_identity_planted_control() -> None:
    one = Action.of(PROBLEM, HVP, {'a': 0.5, 'b': 0.5}, want=WANT, direction=(1.0, 0.0))
    same = Action.of(PROBLEM, HVP, {'a': 0.5, 'b': 0.5}, want=WANT, direction=(1.0, 0.0))
    other = Action.of(PROBLEM, HVP, {'a': 0.5, 'b': 0.5}, want=WANT, direction=(0.0, 1.0))
    assert one.identity.key == same.identity.key
    assert one.identity.key != other.identity.key


def test_p9b_a_result_carries_an_hvp_only_for_an_action_with_a_direction() -> None:
    plain = Action.of(PROBLEM, HVP, {'a': 0.5, 'b': 0.5})
    with pytest.raises(Refusal, match='direction'):
        Result(plain, Cell(Status.OK, values={'f': 1.0}), hvp={'f': np.ones(2)})
    asked = Action.of(PROBLEM, HVP, {'a': 0.5, 'b': 0.5}, want=WANT, direction=(1.0, 0.0))
    assert Result(asked, Cell(Status.OK, values={'f': 1.0}), hvp={'f': np.ones(2)}).hvp['f'].shape == (2,)
