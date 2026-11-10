"""The provider seam and the driver, on the compliance benchmark: the MBB half-beam under SIMP.

The provider is a tiny plane-stress Q4 model (unit square cells, the closed-form element of Andreassen
et al. 2011, "top88") written HERE, so the library never meets the physics. Checked:

* the chain the optimizer sees -- KS over robust realizations, constraints per realization, passive
  cells, symmetry ties, both filters -- against central differences, with a planted control;
* the benchmark qualitatively reproduces the known MBB result: the volume bound is met, grayness falls
  with the projection continuation, compliance falls monotonically after warm-up, to well under half the
  uniform design's (top88 reaches ~0.3 of it at this volume);
* every refusal the seam declares.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
import scipy.sparse as sp
from numerics_lab.linear import solve_linear_system

from optimi_lab.core import Refusal
from optimi_lab.topology import (
    ROBUST_ETAS,
    Continuation,
    DensityFilter,
    DesignChain,
    DesignDomain,
    HelmholtzFilter,
    ProviderResponse,
    Role,
    TopologyFailure,
    TopologyProblem,
    fd_check,
    interpolate,
    topology_optimize,
)


def _q4(nu: float = 0.3) -> np.ndarray:
    k = np.array([
        1 / 2 - nu / 6, 1 / 8 + nu / 8, -1 / 4 - nu / 12, -1 / 8 + 3 * nu / 8,
        -1 / 4 + nu / 12, -1 / 8 - nu / 8, nu / 6, 1 / 8 - 3 * nu / 8,
    ])  # fmt: skip
    idx = [
        [0, 1, 2, 3, 4, 5, 6, 7], [1, 0, 7, 6, 5, 4, 3, 2], [2, 7, 0, 5, 6, 3, 4, 1], [3, 6, 5, 0, 7, 2, 1, 4],
        [4, 5, 6, 7, 0, 1, 2, 3], [5, 4, 3, 2, 1, 0, 7, 6], [6, 3, 4, 1, 2, 7, 0, 5], [7, 2, 1, 4, 3, 6, 5, 0],
    ]  # fmt: skip
    return k[np.array(idx)] / (1.0 - nu**2)


class MbbBeam:
    """The MBB half-beam: ``u_x = 0`` on the left edge, a roller bottom-right, unit load down top-left."""

    def __init__(self, nelx: int, nely: int, volfrac: float, *, solid=None, void=None, robust_volume=False) -> None:
        self.domain = DesignDomain.grid((nelx, nely), spacing=1.0, length_unit='mm', solid=solid, void=void)
        self.responses = (
            ProviderResponse('compliance', Role.OBJECTIVE),
            ProviderResponse('volume', Role.CONSTRAINT, bound=volfrac, robust=robust_volume),
        )
        ex, ey = np.meshgrid(np.arange(nelx), np.arange(nely), indexing='ij')
        n1 = (nely + 1) * ex.ravel() + ey.ravel()
        n2 = (nely + 1) * (ex.ravel() + 1) + ey.ravel()
        self.edof = np.column_stack([
            2 * n1 + 2,
            2 * n1 + 3,
            2 * n2 + 2,
            2 * n2 + 3,
            2 * n2,
            2 * n2 + 1,
            2 * n1,
            2 * n1 + 1,
        ])
        self.ndof = 2 * (nelx + 1) * (nely + 1)
        self.free = np.setdiff1d(np.arange(self.ndof), np.union1d(np.arange(0, 2 * (nely + 1), 2), [self.ndof - 1]))
        self.force = np.zeros(self.ndof)
        self.force[1] = -1.0
        self.k0 = _q4()
        self.rows, self.cols = np.repeat(self.edof, 8, axis=1).ravel(), np.tile(self.edof, (1, 8)).ravel()

    def evaluate(self, physical: np.ndarray) -> dict[str, tuple[float, np.ndarray]]:
        e, de = interpolate(physical, (1e-9, 1.0), scheme='simp', penalty=3.0)
        k = sp.coo_matrix(((self.k0.ravel()[None, :] * e[:, None]).ravel(), (self.rows, self.cols)), (self.ndof,) * 2)
        k = k.tocsc()[self.free][:, self.free]
        u = np.zeros(self.ndof)
        u[self.free] = np.asarray(solve_linear_system(k, self.force[self.free]), dtype=float).reshape(-1)
        ue = u[self.edof]
        m = self.domain.measures
        return {
            'compliance': (float(self.force @ u), -de * np.einsum('ij,jk,ik->i', ue, self.k0, ue)),
            'volume': (float(physical @ m / m.sum()), m / m.sum()),
        }


def _agrees(rows, rtol: float = 1e-5) -> bool:
    return all(abs(a - fd) <= rtol * max(1.0, abs(fd)) for _, a, fd in rows)


@pytest.mark.parametrize('make', [DensityFilter, HelmholtzFilter], ids=['density', 'helmholtz'])
@pytest.mark.parametrize('robust', [False, True], ids=['nominal', 'robust'])
def test_the_problem_the_optimizer_sees_has_exact_gradients(make, robust: bool) -> None:
    solid = np.zeros(48, dtype=bool)
    solid[:2] = True
    provider = MbbBeam(8, 6, 0.4, solid=solid, robust_volume=robust)
    chain = DesignChain(make(provider.domain, 1.5, unit='mm'), beta=3.0, etas=ROBUST_ETAS if robust else (0.5,))
    problem = TopologyProblem(provider, chain, ks_rho=10.0)
    z = np.random.default_rng(11).uniform(0.2, 0.8, provider.domain.n_design)
    assert problem.evaluate(z).linearization.c.size == (3 if robust else 1)
    for row in range(1 + (3 if robust else 1)):

        def part(v, row=row):
            point = problem.evaluate(v).linearization
            return (point.f[0], point.df[0]) if row == 0 else (point.c[row - 1], point.dc[row - 1])

        # The eroded solve is ill-conditioned (E_min = 1e-9): a 1e-6 step reads its rounding, 1e-4 does not.
        rows = fd_check(part, z, range(0, 48, 5), step=1e-4)
        assert _agrees(rows, rtol=1e-5), rows
    planted = [(k, a * 1.05 if k == 10 else a, fd) for k, a, fd in rows]
    assert not _agrees(planted), 'planted control: a 5 % error in one entry must fail the same check'


@pytest.mark.parametrize('optimizer', ['mma', 'oc'])
def test_the_mbb_benchmark_meets_its_volume_and_turns_black_and_white(optimizer: str) -> None:
    provider = MbbBeam(30, 10, 0.5)
    result = topology_optimize(
        provider,
        DensityFilter(provider.domain, 1.5, unit='mm'),
        optimizer=optimizer,
        continuation=Continuation(betas=(1.0, 4.0, 16.0), iterations=60, tol=1e-3),
    )
    history = result.history
    first = [it for it in history if it.stage == 0]
    uniform = first[0].objective
    assert history[-1].constraints[0] <= 0.01, 'the volume bound is met'
    assert history[-1].objective < 0.5 * uniform
    assert history[-1].grayness < 0.5 * first[-1].grayness
    assert history[-1].grayness < 0.15
    warm = [it.objective for it in first[10:]]
    assert all(b <= a * (1 + 1e-2) for a, b in itertools.pairwise(warm)), warm
    assert result.physical[0].shape == (300,)


def test_a_robust_run_reads_every_realization_and_holds_the_nominal_volume() -> None:
    provider = MbbBeam(24, 8, 0.5)
    result = topology_optimize(
        provider,
        HelmholtzFilter(provider.domain, 2.0, unit='mm'),
        robust=True,
        continuation=Continuation(betas=(2.0, 8.0), iterations=40),
    )
    assert len(result.physical) == 3
    eroded, nominal, dilated = (p.mean() for p in result.physical)
    assert eroded <= nominal <= dilated
    assert result.history[-1].constraints[0] <= 0.01


class _Declared(MbbBeam):
    def __init__(self, responses) -> None:
        super().__init__(4, 3, 0.5)
        self.responses = responses


@pytest.mark.parametrize(
    ('responses', 'message'),
    [
        ((ProviderResponse('volume', Role.CONSTRAINT, bound=0.5),), 'exactly one objective'),
        (
            (ProviderResponse('compliance', Role.OBJECTIVE), ProviderResponse('volume', Role.OBJECTIVE)),
            'exactly one objective',
        ),
        (
            (ProviderResponse('compliance', Role.OBJECTIVE), ProviderResponse('volume', Role.CONSTRAINT)),
            'needs a bound',
        ),
    ],
    ids=['none', 'two', 'unbounded'],
)
def test_the_seam_refuses_a_malformed_declaration(responses, message: str) -> None:
    provider = _Declared(responses)
    with pytest.raises(Refusal, match=message):
        TopologyProblem(provider, DesignChain(DensityFilter(provider.domain, 1.5, unit='mm')))


def test_the_driver_refuses_an_unknown_optimizer_and_a_foreign_filter() -> None:
    provider = MbbBeam(4, 3, 0.5)
    with pytest.raises(Refusal, match="'mma', 'gcmma', 'oc'"):
        topology_optimize(provider, DensityFilter(provider.domain, 1.5, unit='mm'), optimizer='sgd')
    other = DesignDomain.grid((4, 3), spacing=1.0, length_unit='mm')
    with pytest.raises(Refusal, match='domain'):
        topology_optimize(provider, DensityFilter(other, 1.5, unit='mm'))


def test_a_response_the_provider_does_not_return_is_refused() -> None:
    provider = _Declared((
        ProviderResponse('compliance', Role.OBJECTIVE),
        ProviderResponse('stress', Role.CONSTRAINT, bound=1.0),
    ))
    problem = TopologyProblem(provider, DesignChain(DensityFilter(provider.domain, 1.5, unit='mm')))
    with pytest.raises(Refusal, match='stress'):
        problem.evaluate(np.full(12, 0.5))


class _SingularAt(MbbBeam):
    """The MBB beam whose *at*-th evaluation fails as a bare numpy ``Singular matrix`` (HPC 7967593's message)."""

    def __init__(self, at: int) -> None:
        super().__init__(6, 4, 0.5)
        self.at, self.calls = at, 0

    def evaluate(self, physical: np.ndarray) -> dict[str, tuple[float, np.ndarray]]:
        self.calls += 1
        if self.calls == self.at:
            msg = 'Singular matrix'
            raise np.linalg.LinAlgError(msg)
        return super().evaluate(physical)


def test_a_numerical_failure_names_where_its_evaluation_stage_and_beta() -> None:
    """A numerical failure in the loop is a :class:`TopologyFailure`: where, which evaluation, stage and beta."""
    provider = _SingularAt(5)
    continuation = Continuation(betas=(1.0, 2.0), iterations=3)
    with pytest.raises(TopologyFailure) as caught:
        topology_optimize(provider, DensityFilter(provider.domain, 1.5, unit='mm'), continuation=continuation)
    failure = caught.value
    assert (failure.where, failure.evaluation, failure.stage, failure.beta) == ('the provider evaluation', 5, 1, 2.0)
    assert isinstance(failure.__cause__, np.linalg.LinAlgError)
    for part in ('the provider evaluation', 'evaluation 5', 'stage 1', 'beta = 2', 'LinAlgError: Singular matrix'):
        assert part in str(failure)
