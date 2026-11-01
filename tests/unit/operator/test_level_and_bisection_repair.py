"""Rounding a relaxed level index, and bisecting a point toward an admissible target."""

import numpy as np
import pytest

from optimi_lab.core import Refusal
from optimi_lab.operator import bisect_to_admissible, round_to_level


def test_round_to_level_rounds_clips_and_says_whether_it_moved():
    assert round_to_level(1.0, 3) == (1, False)
    assert round_to_level(1.4, 3) == (1, True)
    assert round_to_level(-0.49, 3) == (0, True)
    assert round_to_level(7.0, 3) == (2, True)
    assert round_to_level(2.5, 4) == (2, True)  # half to even, as rint
    with pytest.raises(Refusal):
        round_to_level(0.0, 0)


def test_bisection_finds_the_boundary_of_a_half_space():
    """Admissible is x0 >= 0.3; from 0 toward 1 the boundary sits at fraction 0.3."""
    point, target = np.array([0.0, 0.5]), np.array([1.0, 0.5])
    repaired, fraction = bisect_to_admissible(point, target, lambda u: u[0] >= 0.3, steps=30)
    assert fraction == pytest.approx(0.3, abs=2**-30)
    assert fraction >= 0.3 and repaired[0] == pytest.approx(fraction) and repaired[1] == 0.5


def test_bisection_of_an_admissible_point_moves_by_at_most_one_halving_per_step():
    repaired, fraction = bisect_to_admissible(np.zeros(2), np.ones(2), lambda _u: True, steps=10)
    assert fraction == 2**-10
    np.testing.assert_allclose(repaired, np.full(2, 2**-10))
