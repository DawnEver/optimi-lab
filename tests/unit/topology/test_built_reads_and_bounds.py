"""The BUILT design and the bounds a provider declares on it.

* :func:`built` is the nominal threshold of the physical density, :data:`NOMINAL_ETA` the middle robust one;
* a connectivity measure read on the smoothed threshold (:class:`Threshold`) refuses a design joined only
  through cells that threshold to void -- the measure read on the density is fooled by the same design;
* :func:`volume_fraction` weights cells by their measure; a ``lower`` bound is held ``value >= lower``.
"""

from __future__ import annotations

import numpy as np
import pytest

from optimi_lab.core import Refusal
from optimi_lab.topology import (
    BUILT_READ,
    NOMINAL_ETA,
    ROBUST_ETAS,
    DensityFilter,
    DesignChain,
    DesignDomain,
    ProviderResponse,
    Role,
    Threshold,
    TopologyProblem,
    TopologyResult,
    VirtualTemperature,
    built,
    components,
    fd_check,
    volume_fraction,
)

GAMMA = 30.0


def test_the_built_design_is_the_nominal_threshold() -> None:
    assert NOMINAL_ETA == ROBUST_ETAS[len(ROBUST_ETAS) // 2] == 0.5
    assert built(np.array([0.2, 0.5, 0.51, 1.0])).tolist() == [False, False, True, True]
    assert (BUILT_READ.beta, BUILT_READ.eta) == (16.0, NOMINAL_ETA)


def _bridged(bridge: float) -> tuple[DesignDomain, np.ndarray, np.ndarray]:
    """A 12 x 16 mm grid: a held block (sink: the left column) and an island, joined by a 6 x 6 *bridge* density."""
    domain = DesignDomain.grid((12, 16), spacing=1.0, length_unit='mm')
    row, col = np.repeat(np.arange(12), 16), np.tile(np.arange(16), 12)
    x = ((col < 4) | (col >= 10)).astype(float)
    x[(row >= 3) & (row < 9) & (col >= 4) & (col < 10)] = bridge
    return domain, col == 0, x


@pytest.mark.parametrize('penalty', [1.0, 3.0])
def test_a_design_joined_only_through_gray_violates_the_built_read(penalty: float) -> None:
    bound = np.log(GAMMA)
    domain, sink, gray = _bridged(0.45)  # thresholds to VOID: two bodies are built
    assert components(domain, built(gray)) == 2
    # Planted control: the measure read on the density itself is fooled, at either penalty.
    assert VirtualTemperature(domain, sink, penalty=penalty).log_ratio(gray)[0] <= bound
    measure = VirtualTemperature(domain, sink, read_on=BUILT_READ)
    assert measure.log_ratio(gray)[0] > bound
    domain, sink, solid = _bridged(0.55)  # thresholds to SOLID: one body is built
    assert components(domain, built(solid)) == 1
    assert measure.log_ratio(solid)[0] <= bound


def test_the_built_read_gradient_is_the_central_difference_limit() -> None:
    domain, sink, _ = _bridged(1.0)
    x = np.random.default_rng(3).uniform(0.35, 0.65, domain.n_cells)
    measure = VirtualTemperature(domain, sink, read_on=Threshold(beta=16.0))
    rows = fd_check(measure.log_ratio, x, [0, 40, 77, 100, 150, 191], step=1e-5)
    analytic, fd = np.array([a for _, a, _ in rows]), np.array([f for _, _, f in rows])
    # h = 1e-5 through a beta-16 projection: truncation O(h^2 beta^2) sits orders below the bar.
    assert np.max(np.abs(analytic - fd)) / np.max(np.abs(fd)) < 1e-5
    unread = VirtualTemperature(domain, sink).log_ratio(x)[1][[0, 40, 77, 100, 150, 191]]
    assert np.max(np.abs(unread - fd)) / np.max(np.abs(fd)) > 1e-2, 'planted control: the unchained slope fails'


def test_the_volume_fraction_weights_cells_by_their_measure() -> None:
    vertices = np.array([[0.0, 0.0], [3.0, 0.0], [3.0, 1.0], [0.0, 1.0], [1.0, 0.0]])
    domain = DesignDomain.from_simplices(vertices, [[0, 4, 3], [4, 1, 2], [4, 2, 3]], 'mm')  # areas 0.5, 1, 1.5
    x = np.array([1.0, 0.0, 0.0])
    value, slope = volume_fraction(domain, x)
    assert value == pytest.approx(0.5 / 3.0)
    np.testing.assert_allclose(slope, [0.5 / 3.0, 1.0 / 3.0, 1.5 / 3.0])
    assert value != pytest.approx(x.mean()), 'planted control: the uniform mean is wrong on this mesh'


class _Fill:
    """Objective: minus the volume fraction; constraints on the same fraction from above and below."""

    def __init__(self, responses) -> None:
        self.domain = DesignDomain.grid((3, 2), spacing=1.0, length_unit='mm')
        self.responses = responses

    def evaluate(self, physical):
        value, slope = volume_fraction(self.domain, physical)
        return {'volume': (-value, -slope), 'fill': (value, slope)}


def test_a_lower_bound_is_held_from_below_and_both_sides_are_two_rows() -> None:
    provider = _Fill((
        ProviderResponse('volume', Role.OBJECTIVE),
        ProviderResponse('fill', Role.CONSTRAINT, bound=0.8, lower=0.4, robust=False),
    ))
    problem = TopologyProblem(provider, DesignChain(DensityFilter(provider.domain, 1.5, unit='mm'), beta=0.0))
    point = problem.evaluate(np.full(6, 0.3))
    assert point.constraints == pytest.approx(((0.3 - 0.8) / 0.8, (0.4 - 0.3) / 0.4))
    assert np.all(point.linearization.dc[0] > 0.0)
    np.testing.assert_allclose(point.linearization.dc[1], -point.linearization.dc[0] * 0.8 / 0.4)


def test_a_constraint_with_no_bound_on_either_side_is_refused() -> None:
    provider = _Fill((ProviderResponse('volume', Role.OBJECTIVE), ProviderResponse('fill', Role.CONSTRAINT)))
    with pytest.raises(Refusal, match='needs a bound'):
        TopologyProblem(provider, DesignChain(DensityFilter(provider.domain, 1.5, unit='mm')))


def test_the_nominal_realization_is_the_middle_one() -> None:
    physical = (np.zeros(2), np.full(2, 0.5), np.ones(2))
    assert TopologyResult(np.zeros(2), physical, ()).nominal is physical[1]
    assert TopologyResult(np.zeros(2), physical[:1], ()).nominal is physical[0]
