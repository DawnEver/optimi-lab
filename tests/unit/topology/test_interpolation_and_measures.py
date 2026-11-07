"""Property interpolation, aggregation, and the connectivity / member-width measures of a density design.

Closed forms where they exist (endpoints, bounds of a smooth maximum, the width of a straight band, the
body count of a planted design); central differences for every gradient, with planted controls.
"""

from __future__ import annotations

import numpy as np
import pytest

from optimi_lab.core import Refusal
from optimi_lab.topology import (
    DesignDomain,
    VirtualTemperature,
    components,
    fd_check,
    feature_sizes,
    interpolate,
    ks,
    member_widths,
    p_norm,
)


def _agrees(rows, rtol: float = 1e-5) -> bool:
    return all(abs(a - fd) <= rtol * max(1.0, abs(fd)) for _, a, fd in rows)


@pytest.mark.parametrize('scheme', ['simp', 'ramp'])
def test_two_material_interpolation_meets_both_endpoints(scheme: str) -> None:
    value, _ = interpolate(np.array([0.0, 1.0]), (2.0, 7.0), scheme=scheme, penalty=3.0)
    np.testing.assert_allclose(value, [2.0, 7.0])


def test_simp_penalises_gray_below_the_linear_mix() -> None:
    value, slope = interpolate(np.array([0.5]), (0.0, 1.0), scheme='simp', penalty=3.0)
    np.testing.assert_allclose(value, 0.125)
    np.testing.assert_allclose(slope, 0.75)
    ramp, _ = interpolate(np.array([0.5]), (0.0, 1.0), scheme='ramp', penalty=8.0)
    np.testing.assert_allclose(ramp, 0.5 / 5.0)


@pytest.mark.parametrize('scheme', ['simp', 'ramp'])
@pytest.mark.parametrize('values', [(1.0, 1000.0), (1000.0, 1.0), (1.0, 5.0, 20.0)], ids=['up', 'down', 'three'])
def test_interpolation_slopes_are_exact(scheme: str, values) -> None:
    rho = np.linspace(0.03, 0.97, 8)  # avoids the multi-material kinks at 1/2

    def total(r):
        v, d = interpolate(r, values, scheme=scheme, penalty=3.0)
        return float(v.sum()), d

    assert _agrees(fd_check(total, rho, range(rho.size), step=1e-7), rtol=1e-6)


def test_an_unknown_scheme_is_refused_naming_the_set() -> None:
    with pytest.raises(Refusal, match="'simp', 'ramp'"):
        interpolate(np.array([0.5]), (0.0, 1.0), scheme='linear')


@pytest.mark.parametrize('aggregate', [p_norm, ks], ids=['p-norm', 'ks'])
def test_an_aggregate_bounds_the_maximum_and_its_gradient_is_exact(aggregate) -> None:
    values = np.array([0.2, 0.9, 0.85, 0.1])
    top, grad = aggregate(values, 20.0)
    assert 0.9 <= top <= 0.9 * 4 ** (1 / 20) + np.log(4) / 20
    assert _agrees(fd_check(lambda v: aggregate(v, 20.0), values, range(4)))
    wrong = [(k, a + (1e-3 if k == 1 else 0.0), fd) for k, a, fd in fd_check(lambda v: aggregate(v, 20.0), values, [1])]
    assert not _agrees(wrong, rtol=1e-6)
    assert grad[1] > grad[0]


def _grid(nx: int = 10, ny: int = 10) -> DesignDomain:
    return DesignDomain.grid((nx, ny), spacing=1.0, length_unit='mm')


def _mask(nx: int, ny: int, rule) -> np.ndarray:
    i, j = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')
    return rule(i, j).ravel()


def test_components_count_face_connected_bodies() -> None:
    domain = _grid()
    assert components(domain, np.zeros(100, dtype=bool)) == 0
    assert components(domain, _mask(10, 10, lambda i, _: (i < 2) | (i > 7))) == 2
    corner_touch = _mask(10, 10, lambda i, j: ((i == 3) & (j == 3)) | ((i == 4) & (j == 4)))
    assert components(domain, corner_touch) == 2, 'cells touching at a corner only are two bodies'


def test_a_straight_band_reads_its_width_in_the_domain_unit() -> None:
    domain = DesignDomain.grid((20, 20), spacing=0.5, length_unit='mm')
    band = _mask(20, 20, lambda i, _: (i >= 6) & (i < 11))
    widths = member_widths(domain, band)
    np.testing.assert_allclose(widths.min(), 2.5)
    solid, gap = feature_sizes(domain, band)
    np.testing.assert_allclose(solid, 2.5)
    assert gap == np.inf, 'void open to the exterior is not a gap'


def _support(domain: DesignDomain) -> np.ndarray:
    return np.isin(np.arange(domain.n_cells), np.flatnonzero(domain.centroids[:, 0] < 1.0))


def test_virtual_temperature_gradient_is_exact() -> None:
    domain = _grid(6, 5)
    measure = VirtualTemperature(domain, _support(domain), penalty=3.0)
    rho = np.random.default_rng(3).uniform(0.2, 0.9, domain.n_cells)
    rows = fd_check(measure.response, rho, range(domain.n_cells), step=1e-7)
    assert _agrees(rows, rtol=1e-5)
    assert _agrees(fd_check(lambda r: measure.constraint(r, 30.0), rho, range(0, 30, 4), step=1e-7), rtol=1e-5)


def test_virtual_temperature_flags_an_island_by_orders_of_magnitude() -> None:
    domain = _grid()
    measure = VirtualTemperature(domain, _support(domain), penalty=3.0)
    bar = _mask(10, 10, lambda _, j: (j >= 4) & (j < 6))
    island = _mask(10, 10, lambda i, j: (j >= 4) & (j < 6) & (i != 4))
    assert measure.constraint(bar.astype(float), 30.0)[0] < 0.0
    assert measure.constraint(island.astype(float), 30.0)[0] > np.log(1e3)


def test_virtual_temperature_refuses_a_support_with_no_boundary_face() -> None:
    domain = _grid()
    inner = np.zeros(100, dtype=bool)
    inner[55] = True
    with pytest.raises(Refusal, match='boundary face'):
        VirtualTemperature(domain, inner)
