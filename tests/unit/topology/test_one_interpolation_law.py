"""A cell's heat source and its conductance follow ONE interpolation law, so a one-body design reads one body.

Measured in motronics (2026-11-10): with ``gamma`` enforced, the conductance was read on a higher threshold
(~0.72) than the source (0.5, beta 16) at ``s_min ~6.9e-10``. A gray interface cell between the two thresholds
heated but barely conducted and read as a one-cell island: the case's own one-body design read ``log_ratio``
4.7-8.6 on its filtered field against ``log(gamma)`` 2.30 (0.98 on its exact 0/1 form), infeasible from
iteration 0. Planted here: two concentric rings joined by a bridge, the sink inside the inner ring.
"""

from __future__ import annotations

import numpy as np
import pytest

from optimi_lab.topology import BUILT_READ, DesignDomain, VirtualTemperature, built, components, fd_check
from optimi_lab.topology.density import DensityFilter, Threshold

N = 48
GAMMA = 10.0


def _rings(*, bridge: bool):
    """``(domain, sink, x)``: rings at radius 6..10 and 15..19 (cells), four 4-cell-wide spokes joining them if *bridge*."""
    domain = DesignDomain.grid((N, N), spacing=1.0, length_unit='mm')
    centre = (N - 1) / 2.0
    dx, dy = domain.centroids[:, 0] - centre, domain.centroids[:, 1] - centre
    r = np.hypot(dx, dy)
    x = (((r >= 6) & (r < 10)) | ((r >= 15) & (r < 19))).astype(float)
    if bridge:
        spokes = (np.abs(dy) < 2) | (np.abs(dx) < 2)
        x[spokes & (r >= 6) & (r < 19)] = 1.0
    sink = (np.abs(dy) < 2) & (dx < -6) & (dx > -10)  # an anchor inside the inner ring's left side
    return domain, sink, x


def _filtered(x: np.ndarray, domain: DesignDomain) -> np.ndarray:
    return DensityFilter(domain, 2.5, unit='mm').forward(x)


def test_a_one_body_design_reads_below_the_bound_on_its_exact_form() -> None:
    domain, sink, x = _rings(bridge=True)
    assert components(domain, built(x)) == 1
    measure = VirtualTemperature(domain, sink, read_on=BUILT_READ, gamma=GAMMA)
    assert measure.island_floor > np.log(GAMMA * 10.0)
    assert measure.log_ratio(x)[0] <= np.log(GAMMA)


@pytest.mark.parametrize('beta', [1.0, 4.0, 16.0, 64.0])
def test_a_one_body_design_reads_below_the_bound_filtered_and_projected(beta: float) -> None:
    domain, sink, x = _rings(bridge=True)
    gray = _filtered(x, domain)
    assert components(domain, built(gray)) == 1
    measure = VirtualTemperature(domain, sink, read_on=Threshold(beta=beta), gamma=GAMMA)
    assert measure.log_ratio(gray)[0] <= np.log(GAMMA)


@pytest.mark.parametrize('beta', [4.0, 16.0, 64.0])
def test_a_disconnected_ring_reads_above_the_bound(beta: float) -> None:
    domain, sink, x = _rings(bridge=False)
    assert components(domain, built(x)) == 2
    measure = VirtualTemperature(domain, sink, read_on=Threshold(beta=beta), gamma=GAMMA)
    assert measure.log_ratio(x)[0] >= measure.island_floor > np.log(GAMMA)


@pytest.mark.parametrize('beta', [16.0, 64.0])
def test_a_filtered_disconnected_ring_reads_above_its_stated_gray_floor(beta: float) -> None:
    domain, sink, x = _rings(bridge=False)
    gray = _filtered(x, domain)
    assert components(domain, built(gray)) == 2
    measure = VirtualTemperature(domain, sink, read_on=Threshold(beta=beta), gamma=GAMMA)
    island = built(gray) & (np.hypot(*(domain.centroids - (N - 1) / 2.0).T) > 12.5)
    a, b = domain.adjacency.T
    rim = np.unique(np.concatenate([b[island[a] & ~island[b]], a[island[b] & ~island[a]]]))
    void = float(gray[rim].max())  # the densest neighbour the island leaks through
    assert measure.log_ratio(gray)[0] >= measure.floor(void)
    assert measure.log_ratio(gray)[0] > np.log(GAMMA)


#: An island under the enforced ``s_min`` (~1e-9 here) leaves ``K`` conditioned ~1/s_min, so a central difference
#: of its log_ratio is roundoff-bound (measured: relative error GROWS from 5e-4 to 0.8 as the step falls from
#: 1e-4 to 1e-8). The island case checks the same code path at a void the difference can resolve.
@pytest.mark.parametrize(('bridge', 'enforce'), [(True, {'gamma': GAMMA}), (False, {'s_min': 1e-4})])
def test_the_gradient_on_a_gray_field_is_the_central_difference_limit(bridge: bool, enforce: dict) -> None:
    domain, sink, x = _rings(bridge=bridge)
    gray = np.clip(_filtered(x, domain) + np.random.default_rng(3).uniform(-0.1, 0.1, domain.n_cells), 0.02, 0.98)
    measure = VirtualTemperature(domain, sink, read_on=BUILT_READ, **enforce)
    grad = measure.log_ratio(gray)[1]
    cells = [*np.argsort(np.abs(grad))[-6:], 0, N * N // 2 + 30]
    rows = fd_check(measure.log_ratio, gray, cells, step=1e-6)
    analytic, fd = np.array([a for _, a, _ in rows]), np.array([f for _, _, f in rows])
    assert np.max(np.abs(analytic - fd)) / np.max(np.abs(fd)) < 1e-4
