"""The design -> physical density chain: domain, density and Helmholtz filters, projection, robust triplet.

Every map is checked against a closed form where one exists (a filter preserves a constant field, the
projection's limits) and every sensitivity against a CENTRAL DIFFERENCE through the whole chain, with a
planted control: the same check on a gradient with one entry perturbed must fail.
"""

from __future__ import annotations

import numpy as np
import pytest

from optimi_lab.core import Refusal
from optimi_lab.topology import (
    ROBUST_ETAS,
    DensityFilter,
    DesignChain,
    DesignDomain,
    HelmholtzFilter,
    fd_check,
    grayness,
    project,
)


def _grid(nx: int = 8, ny: int = 5, **kw) -> DesignDomain:
    return DesignDomain.grid((nx, ny), spacing=0.5, length_unit='mm', **kw)


def _agrees(rows, rtol: float = 1e-5) -> bool:
    return all(abs(a - fd) <= rtol * max(1.0, abs(fd)) for _, a, fd in rows)


def test_a_grid_domain_declares_its_cells_faces_and_boundary() -> None:
    domain = _grid(4, 3)
    assert domain.n_cells == 12
    assert domain.n_design == 12
    np.testing.assert_allclose(domain.measures, 0.25)
    assert len(domain.adjacency) == 3 * 3 + 4 * 2
    np.testing.assert_allclose(domain.faces, 0.5)
    assert len(domain.boundary_cells) == 2 * 4 + 2 * 3
    np.testing.assert_allclose(domain.boundary_distances, 0.25)
    np.testing.assert_allclose(domain.centroids[0], [0.25, 0.25])


def test_a_domain_refuses_a_cell_both_solid_and_void() -> None:
    mask = np.zeros(12, dtype=bool)
    mask[3] = True
    with pytest.raises(Refusal, match='both solid and void'):
        _grid(4, 3, solid=mask, void=mask)


def test_a_mirror_ties_cells_to_one_design_variable() -> None:
    domain = _grid(4, 3).mirrored(axis=0)
    assert domain.n_design == 6
    z = np.arange(6.0)
    full = domain.expand(z)
    np.testing.assert_array_equal(full.reshape(4, 3)[0], full.reshape(4, 3)[3])
    g = np.ones(12)
    np.testing.assert_allclose(domain.reduce(g), 2.0)


@pytest.mark.parametrize('make', [DensityFilter, HelmholtzFilter], ids=['density', 'helmholtz'])
def test_a_filter_keeps_a_constant_field_and_smooths_a_spike(make) -> None:
    domain = _grid()
    flt = make(domain, radius=1.2, unit='mm')
    np.testing.assert_allclose(flt.forward(np.full(domain.n_cells, 0.3)), 0.3, atol=1e-12)
    spike = np.zeros(domain.n_cells)
    spike[17] = 1.0
    out = flt.forward(spike)
    assert 0.0 < out[17] < 1.0
    assert np.count_nonzero(out > 1e-9) > 1


@pytest.mark.parametrize('make', [DensityFilter, HelmholtzFilter], ids=['density', 'helmholtz'])
def test_a_filter_radius_carries_the_domains_length_unit(make) -> None:
    with pytest.raises(Refusal, match="'m'.*'mm'"):
        make(_grid(), radius=1.2, unit='m')


def test_the_projection_is_the_identity_at_zero_and_a_step_in_the_limit() -> None:
    x = np.linspace(0.0, 1.0, 11)
    value, slope = project(x, 0.0)
    np.testing.assert_array_equal(value, x)
    np.testing.assert_array_equal(slope, 1.0)
    sharp, _ = project(x, 500.0, 0.5)
    np.testing.assert_allclose(sharp[x < 0.45], 0.0, atol=1e-9)
    np.testing.assert_allclose(sharp[x > 0.55], 1.0, atol=1e-9)
    np.testing.assert_allclose(project(np.array([0.0, 1.0]), 8.0, 0.3)[0], [0.0, 1.0], atol=1e-12)


def test_grayness_is_zero_for_a_binary_design_and_one_for_all_half() -> None:
    assert grayness(np.array([0.0, 1.0, 1.0])) == 0.0
    assert grayness(np.full(4, 0.5)) == 1.0


def _objective(weights: list[np.ndarray]):
    """A nonlinear scalar of every realization: ``sum_k w_k . x_k^2``."""

    def f(z, chain):
        reals = chain.forward(z)
        value = sum(float(w @ r**2) for w, r in zip(weights, reals, strict=True))
        return value, chain.pull_back([2.0 * w * r for w, r in zip(weights, reals, strict=True)])

    return f


@pytest.mark.parametrize('make', [DensityFilter, HelmholtzFilter], ids=['density', 'helmholtz'])
@pytest.mark.parametrize('robust', [False, True], ids=['nominal', 'robust'])
def test_the_chain_pulls_back_the_exact_gradient(make, robust: bool) -> None:
    solid, void = np.zeros(40, dtype=bool), np.zeros(40, dtype=bool)
    solid[0], void[39] = True, True
    domain = _grid(solid=solid, void=void).mirrored(axis=0)
    chain = DesignChain(make(domain, radius=1.2, unit='mm'), beta=4.0, etas=ROBUST_ETAS if robust else (0.5,))
    rng = np.random.default_rng(7)
    weights = [rng.uniform(0.5, 1.5, domain.n_cells) for _ in chain.etas]
    f = _objective(weights)
    z = rng.uniform(0.2, 0.8, domain.n_design)
    rows = fd_check(lambda v: f(v, chain), z, range(domain.n_design))
    assert _agrees(rows)
    wrong = [(k, a * (1.05 if k == 3 else 1.0), fd) for k, a, fd in rows]
    assert not _agrees(wrong), 'planted control: a 5 % error in one entry must fail the same check'


def test_passive_cells_are_held_and_carry_no_sensitivity() -> None:
    solid, void = np.zeros(40, dtype=bool), np.zeros(40, dtype=bool)
    solid[:3], void[-2:] = True, True
    domain = _grid(solid=solid, void=void)
    chain = DesignChain(DensityFilter(domain, radius=1.2, unit='mm'), beta=2.0)
    (phys,) = chain.forward(np.full(40, 0.5))
    np.testing.assert_array_equal(phys[:3], 1.0)
    np.testing.assert_array_equal(phys[-2:], 0.0)
    g = chain.pull_back([np.where(solid | void, 1.0, 0.0)])
    np.testing.assert_allclose(g, 0.0)
