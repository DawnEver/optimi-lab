"""D108, D114 -- Morris elementary effects and objective-conflict learning, against closed forms.

Morris on a linear function: every elementary effect of ``x_i`` IS its coefficient (in output units
per unit of the variable's range), so ``mu*`` is ``|a_i|`` exactly and ``sigma`` is 0. Conflict: on a
two-objective front the objectives disagree on every pair; a monotone copy of an objective agrees
with it on every pair and is redundant.
"""

import numpy as np
import pytest

from optimi_lab.analysis import conflict_matrix, essential_objectives, morris
from optimi_lab.core import Refusal
from optimi_lab.problem import Real, Space

SPACE = Space((Real('a', 0.0, 2.0), Real('b', -1.0, 1.0), Real('c', 0.0, 10.0)))


def test_morris_mu_star_of_a_linear_function_is_each_coefficient_times_its_range():
    """Y = 3a - 2b + 0c: an effect per unit of the RANGE is the coefficient times the range width."""
    result = morris(SPACE, lambda x: 3 * x[:, 0] - 2 * x[:, 1], trajectories=20, levels=4, seed=1)
    assert result.mu_star == pytest.approx([3 * 2.0, 2 * 2.0, 0.0])
    assert result.sigma == pytest.approx([0.0, 0.0, 0.0], abs=1e-12)
    assert result.n_evaluations == 20 * 4


def test_morris_separates_an_interaction_by_its_spread():
    result = morris(SPACE, lambda x: x[:, 0] * x[:, 2], trajectories=30, levels=4, seed=2)
    assert result.sigma[0] > 0.1 and result.sigma[2] > 0.1 and result.mu_star[1] == 0.0


def test_conflict_is_one_along_a_front_and_zero_for_a_monotone_copy():
    f1 = np.linspace(0.0, 1.0, 30)
    values = np.column_stack([f1, 1.0 - f1, 2.0 * f1 + 1.0])
    matrix = conflict_matrix(values)
    assert matrix[0, 1] == pytest.approx(1.0) and matrix[0, 2] == pytest.approx(0.0)
    assert np.allclose(matrix, matrix.T)


def test_a_redundant_objective_is_learned_and_dropped():
    rng = np.random.default_rng(0)
    f1, f2 = rng.random(40), rng.random(40)
    values = np.column_stack([f1, f2, 2.0 * f1 + 1.0])
    kept = essential_objectives(values)
    assert len(kept) == 2 and 1 in kept and len({0, 2} & set(kept)) == 1
    assert essential_objectives(np.column_stack([f1, f2])) == (0, 1), 'two conflicting objectives both stay'


def test_essential_objectives_refuse_a_negative_tolerance():
    with pytest.raises(Refusal, match='tolerance'):
        essential_objectives(np.ones((3, 2)), tolerance=-0.1)
