"""P13 -- an Uncertainty maps the unit cube and the standard normal space onto its parameters.

The reliability and QMC operators work in one of those two reference spaces; the map to physical
values is the parameter's own (inverse CDF), so it lives on the declaration.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from optimi_lab.core import Refusal
from optimi_lab.problem import Normal, Uncertainty, Uniform

#: Inverse-CDF agreement, in the parameter's own unit (O(1) values here).
TOLERANCE = 1e-12

UNCERTAINTY = Uncertainty((Normal('r', 5.0, 0.5), Uniform('t', 20.0, 80.0)))


def test_the_unit_cube_maps_through_each_inverse_cdf() -> None:
    u = np.array([[0.025, 0.25], [0.5, 0.5], [0.975, 0.9]])
    found = UNCERTAINTY.at_unit(u)
    assert np.max(np.abs(found['r'] - norm.ppf(u[:, 0], 5.0, 0.5))) <= TOLERANCE
    assert np.max(np.abs(found['t'] - (20.0 + 60.0 * u[:, 1]))) <= TOLERANCE


def test_the_standard_normal_space_maps_through_each_parameter() -> None:
    z = np.array([[-1.0, 0.0], [2.0, 1.959963984540054]])
    found = UNCERTAINTY.at_standard(z)
    assert np.max(np.abs(found['r'] - (5.0 + 0.5 * z[:, 0]))) <= TOLERANCE
    assert np.max(np.abs(found['t'] - (20.0 + 60.0 * norm.cdf(z[:, 1])))) <= TOLERANCE


def test_planted_control_a_point_of_the_wrong_width_is_refused() -> None:
    with pytest.raises(Refusal, match='2 columns'):
        UNCERTAINTY.at_standard(np.zeros((4, 3)))
    with pytest.raises(Refusal, match=r'\[0, 1\]'):
        UNCERTAINTY.at_unit(np.array([[0.5, 1.5]]))
