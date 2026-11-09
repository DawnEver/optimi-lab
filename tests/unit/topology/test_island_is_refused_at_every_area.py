"""A built island is refused by the virtual-temperature measure at EVERY area, by construction.

The planted fixture is a 40 x 40 mm grid: a sink column, a solid body joined to it, and a k x k island cut off
from it by void -- a design that BUILDS to two bodies. Measured on the measure as it stood (``s_min`` 1e-6, the
beta-16 read): an island's temperature is ``~ its area / (s_min * its perimeter conductance)``, so it SHRINKS with
the island (a one-cell island read ``log_ratio`` 1.3 against ``log 20 = 3.0``) -- ``s_min`` alone never made the
gap ``~1/s_min`` the old docstring claimed -- and a void cell just under the build threshold conducted
``H(eta)^p = 1/8`` of solid through the read. Declared ``gamma`` now DERIVES the void conductance and the
conductance read so that :attr:`VirtualTemperature.island_floor`, a proven lower bound on the ``log_ratio`` of any
design with a built island, exceeds ``log(gamma)``.
"""

from __future__ import annotations

import numpy as np
import pytest

from optimi_lab.core import Refusal
from optimi_lab.topology import BUILT_READ, NOMINAL_ETA, DesignDomain, VirtualTemperature, built, components, fd_check

N = 40


def _planted(island: int, *, gap_value: float = 0.0, anchored_island: bool = False):
    """``(domain, sink, x)``: a body on the sink column (cols 0..19) and a *island* x *island* block at col 30.

    The gap (cols 20..29) holds *gap_value* on the island's rows: void, or a just-void gray bridge.
    """
    domain = DesignDomain.grid((N, N), spacing=1.0, length_unit='mm')
    row, col = np.repeat(np.arange(N), N), np.tile(np.arange(N), N)
    x = (col < 20).astype(float)
    rows = (row >= 10) & (row < 10 + island)
    x[rows & (col >= 30) & (col < 30 + island)] = 1.0
    x[rows & (col >= 20) & (col < 30) & (row == 10)] = gap_value
    sink = col == 0
    if anchored_island:
        sink = sink | (rows & (col == 30))
    return domain, sink, x


@pytest.mark.parametrize('gamma', [5.0, 10.0, 20.0])
@pytest.mark.parametrize('island', [8, 4, 2, 1])
def test_an_island_of_any_area_is_refused(gamma: float, island: int) -> None:
    domain, sink, x = _planted(island)
    assert components(domain, built(x)) == 2
    measure = VirtualTemperature(domain, sink, read_on=BUILT_READ, gamma=gamma)
    assert measure.island_floor > np.log(gamma)
    assert measure.log_ratio(x)[0] >= measure.island_floor


def test_planted_control_a_fixed_void_conductance_reads_a_smaller_island_cooler() -> None:
    ratios = []
    for island in (8, 4, 2, 1):
        domain, sink, x = _planted(island)
        ratios.append(VirtualTemperature(domain, sink, read_on=BUILT_READ).log_ratio(x)[0])
    # ~ log(area / perimeter): no floor independent of the island, so some island passes any fixed gamma.
    assert ratios == sorted(ratios, reverse=True)
    assert ratios[0] - ratios[-1] > np.log(8.0) - 0.5


def test_a_bridge_just_under_the_build_threshold_is_refused() -> None:
    domain, sink, x = _planted(4, gap_value=NOMINAL_ETA - 1e-3)
    assert components(domain, built(x)) == 2
    assert VirtualTemperature(domain, sink, read_on=BUILT_READ).log_ratio(x)[0] < np.log(20.0), 'planted control'
    assert VirtualTemperature(domain, sink, read_on=BUILT_READ, gamma=20.0).log_ratio(x)[0] > np.log(20.0)


@pytest.mark.parametrize('gamma', [5.0, 20.0])
def test_a_connected_design_passes(gamma: float) -> None:
    domain, sink, x = _planted(4, gap_value=1.0)
    assert components(domain, built(x)) == 1
    assert VirtualTemperature(domain, sink, read_on=BUILT_READ, gamma=gamma).log_ratio(x)[0] <= np.log(gamma)


def test_separately_anchored_bodies_pass() -> None:
    domain, sink, x = _planted(4, anchored_island=True)
    assert components(domain, built(x)) == 2
    assert VirtualTemperature(domain, sink, read_on=BUILT_READ, gamma=5.0).log_ratio(x)[0] <= np.log(5.0)


def test_the_enforcing_gradient_is_the_central_difference_limit() -> None:
    domain, sink, x = _planted(4, gap_value=1.0)
    x = np.clip(x + np.random.default_rng(5).uniform(-0.3, 0.3, domain.n_cells), 0.05, 0.95)
    measure = VirtualTemperature(domain, sink, read_on=BUILT_READ, gamma=20.0)
    cells = [0, 415, 433, 820, 1000, 1599]
    rows = fd_check(measure.log_ratio, x, cells, step=1e-6)
    analytic, fd = np.array([a for _, a, _ in rows]), np.array([f for _, _, f in rows])
    assert np.max(np.abs(analytic - fd)) / np.max(np.abs(fd)) < 1e-4


def test_gamma_needs_a_build_read_and_an_explicit_void_that_admits_an_island_is_refused() -> None:
    domain, sink, _ = _planted(1)
    with pytest.raises(Refusal, match='build'):
        VirtualTemperature(domain, sink, gamma=20.0)
    with pytest.raises(Refusal, match='island'):
        VirtualTemperature(domain, sink, read_on=BUILT_READ, gamma=20.0, s_min=1e-2)
