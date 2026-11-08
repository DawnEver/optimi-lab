"""Differentiable connectivity measures: is every solid cell joined to a support through solid?

Both measures load the solid with its own weight (a heat source, a body force), let void carry almost nothing
(``s_min``) and read the smooth maximum (:func:`~optimi_lab.topology.interpolation.p_norm`) of the response PER
UNIT LOAD against the solid design's: :meth:`log_ratio` ``= log(r / r_solid)``, a design is connected at
``<= log(gamma)``. An island sheds its load through void only, so its ratio rises by ``~1 / s_min``.

* :class:`VirtualTemperature` (Liu et al. 2015, Front. Mech. Eng. 10:126; Li et al. 2016, Struct. Multidisc.
  Optim. 54:971): the *support* CELLS are the sink, held at zero temperature wherever they sit (a boundary
  layer or an interior anchor). Cell-centred two-point fluxes across shared FACES with a harmonic-mean
  conductance, so it and :func:`~optimi_lab.topology.measures.components` mean the same thing by connected.
* :class:`SelfWeightSupport`: plane-stress self-weight over a triangle mesh, clamped at support NODES (after
  Bruyneel and Duysinx 2005, Struct. Multidisc. Optim. 29:245) -- a proxy for a mounting load, not a stress
  analysis. P1 elasticity carries load through a shared vertex, so it does not refuse a vertex hinge.

``read_on`` (:class:`~optimi_lab.topology.density.Threshold`; :data:`~optimi_lab.topology.density.BUILT_READ`)
makes either read the BUILT design rather than the density, so gray cells that build to void carry nothing.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from numerics_lab.linear import solve_linear_system

from optimi_lab.core.errors import refuse
from optimi_lab.topology.density import Threshold
from optimi_lab.topology.domain import DesignDomain
from optimi_lab.topology.interpolation import p_norm

__all__ = ['PENALTY', 'P_NORM', 'S_MIN', 'SelfWeightSupport', 'VirtualTemperature']

#: Defaults of the connectivity measures: the SIMP exponent of the conductance (stiffness), the void's
#: conductance (stiffness) relative to solid, and the p-norm exponent of the smooth maximum.
PENALTY = 3.0
S_MIN = 1e-6
P_NORM = 8.0


class VirtualTemperature:
    """The virtual-temperature connectivity measure, sunk at the *support* cells (zero temperature there)."""

    def __init__(
        self,
        domain: DesignDomain,
        support: np.ndarray,
        *,
        penalty: float = PENALTY,
        s_min: float = S_MIN,
        p: float = P_NORM,
        read_on: Threshold | None = None,
    ) -> None:
        sink = np.asarray(support, dtype=bool)
        if sink.shape != (domain.n_cells,) or not sink.any():
            refuse(f'the support names no sink cell among the {domain.n_cells} (a mask of one bool per cell)')
        if sink.all():
            refuse('every cell is a sink: there is no temperature to measure')
        a, b = domain.adjacency.T
        self.domain, self.penalty, self.s_min, self.p = domain, float(penalty), float(s_min), float(p)
        self.read_on = read_on
        self.w_pair = domain.faces / np.linalg.norm(domain.centroids[a] - domain.centroids[b], axis=1)
        self.free = np.flatnonzero(~sink)
        self.reference = self.specific(np.ones(domain.n_cells))[0]

    def _solve(self, k: sp.csc_matrix, rhs: np.ndarray) -> np.ndarray:
        """``K_ff x_f = rhs_f`` with the sink cells held at zero; the full vector."""
        out = np.zeros(self.domain.n_cells)
        f = self.free
        out[f] = np.asarray(solve_linear_system(k[f][:, f], rhs[f]), dtype=float).reshape(-1)
        return out

    def response(self, phys: np.ndarray) -> tuple[float, np.ndarray]:
        """``(J, dJ / d phys)``, the smooth maximum cell temperature, read through ``read_on`` when declared."""
        x, slope = _read(self.read_on, phys)
        j, dj = self._response(x)
        return j, dj * slope

    def _response(self, phys: np.ndarray) -> tuple[float, np.ndarray]:
        """``J`` at the density the measure reads; adjoint ``K lam = dJ/dT`` (``K`` symmetric)."""
        n = self.domain.n_cells
        s = self.s_min + phys**self.penalty * (1.0 - self.s_min)
        ds = self.penalty * phys ** (self.penalty - 1.0) * (1.0 - self.s_min)
        a, b = self.domain.adjacency.T
        g = self.w_pair * 2.0 * s[a] * s[b] / (s[a] + s[b])
        k = sp.coo_matrix(
            (np.concatenate([g, g, -g, -g]), (np.concatenate([a, b, a, b]), np.concatenate([a, b, b, a]))), (n, n)
        ).tocsc()
        area = self.domain.measures
        t = self._solve(k, phys * area)
        j, dj = p_norm(t, self.p)
        if j == 0.0:
            return 0.0, np.zeros(n)
        lam = self._solve(k, dj)
        dt, dl = t[a] - t[b], lam[a] - lam[b]
        dg_da = self.w_pair * 2.0 * s[b] ** 2 / (s[a] + s[b]) ** 2
        dg_db = self.w_pair * 2.0 * s[a] ** 2 / (s[a] + s[b]) ** 2
        grad = lam * area
        np.add.at(grad, a, -dg_da * ds[a] * dl * dt)
        np.add.at(grad, b, -dg_db * ds[b] * dl * dt)  # a pair onto a sink cell: t = lam = 0 there
        return j, grad

    def specific(self, phys: np.ndarray) -> tuple[float, np.ndarray]:
        """``(J / Q, d / d phys)``, ``Q = sum phys V`` the total source: the response PER UNIT LOAD.

        ``J`` alone is linear in the source, so removing material always lowers it; per unit load a
        connected design is not cheaper for being smaller. Read through ``read_on`` when declared.
        """
        x, slope = _read(self.read_on, phys)
        return _per_unit_load(*self._response(x), x, self.domain.measures, slope)

    def log_ratio(self, phys: np.ndarray) -> tuple[float, np.ndarray]:
        """``(log(r / r_solid), its gradient)``, ``r`` :meth:`specific`; a design is connected at ``<= log(gamma)``."""
        return _log_ratio(*self.specific(phys), self.reference)


class SelfWeightSupport:
    """The self-weight support connectivity measure over a TRIANGLE mesh, clamped at the *support* nodes.

    Plane-stress linear triangles (Poisson ratio *poisson*, unit modulus scaled by ``s_min + x^penalty (1 -
    s_min)``) under a body force ``-y`` per unit solid area; ``J`` the smooth maximum of the nodal deflections.
    *vertices* ``(m, 2)``, *cells* ``(n, 3)``, *support* one bool per vertex; ``read_on`` as the thermal one.
    """

    def __init__(
        self,
        vertices: np.ndarray,
        cells: np.ndarray,
        support: np.ndarray,
        *,
        penalty: float = PENALTY,
        s_min: float = S_MIN,
        p: float = P_NORM,
        poisson: float = 0.3,
        read_on: Threshold | None = None,
    ) -> None:
        cells = np.asarray(cells, dtype=np.intp)
        if cells.ndim != 2 or cells.shape[1] != 3:
            refuse(f'the self-weight support is built on triangles, got cells of shape {cells.shape}')
        used, local = np.unique(cells, return_inverse=True)
        self.cells = local.reshape(cells.shape)
        self.vertices = np.asarray(vertices, dtype=float)[used, :2]
        clamped = np.flatnonzero(np.asarray(support, dtype=bool)[used])
        if len(clamped) == 0:
            refuse('the self-weight support clamps no support node of the mesh')
        self.penalty, self.s_min, self.p, self.read_on = float(penalty), float(s_min), float(p), read_on
        self._elements(float(poisson))
        fixed = (2 * clamped[:, None] + np.arange(2)[None, :]).ravel()
        self.free = np.setdiff1d(np.arange(self.ndof), fixed)
        self.rows = np.repeat(self.dofs, 6, axis=1).ravel()
        self.cols = np.tile(self.dofs, (1, 6)).ravel()
        self.reference = self.specific(np.ones(len(self.cells)))[0]

    def _elements(self, nu: float) -> None:
        p = self.vertices[self.cells]
        b = np.stack([p[:, 1, 1] - p[:, 2, 1], p[:, 2, 1] - p[:, 0, 1], p[:, 0, 1] - p[:, 1, 1]], axis=1)
        c = np.stack([p[:, 2, 0] - p[:, 1, 0], p[:, 0, 0] - p[:, 2, 0], p[:, 1, 0] - p[:, 0, 0]], axis=1)
        area = 0.5 * (b[:, 0] * c[:, 1] - b[:, 1] * c[:, 0])
        n = len(area)
        bm = np.zeros((n, 3, 6))
        bm[:, 0, 0::2] = b
        bm[:, 1, 1::2] = c
        bm[:, 2, 0::2] = c
        bm[:, 2, 1::2] = b
        bm /= (2.0 * area)[:, None, None]
        d = np.array([[1.0, nu, 0.0], [nu, 1.0, 0.0], [0.0, 0.0, 0.5 * (1.0 - nu)]]) / (1.0 - nu**2)
        self.area = np.abs(area)
        self.ke = np.einsum('eki,kl,elj->eij', bm, d, bm) * area[:, None, None]
        self.fe = np.zeros((n, 6))
        self.fe[:, 1::2] = -self.area[:, None] / 3.0
        self.dofs = np.stack([2 * self.cells, 2 * self.cells + 1], axis=2).reshape(n, 6)
        self.ndof = 2 * len(self.vertices)

    def _solve(self, k: sp.csc_matrix, rhs: np.ndarray) -> np.ndarray:
        """``K_ff u_f = rhs_f`` with the clamped dofs held at zero; the full vector."""
        out = np.zeros(self.ndof)
        f = self.free
        out[f] = np.asarray(solve_linear_system(k[f][:, f], rhs[f]), dtype=float).reshape(-1)
        return out

    def response(self, phys: np.ndarray) -> tuple[float, np.ndarray]:
        """``(J, dJ / d phys)``, read through ``read_on`` when declared."""
        x, slope = _read(self.read_on, phys)
        j, dj = self._response(x)
        return j, dj * slope

    def _response(self, phys: np.ndarray) -> tuple[float, np.ndarray]:
        """``J`` at the density the measure reads; adjoint ``K lam = dJ/du`` (``K`` symmetric)."""
        s = self.s_min + phys**self.penalty * (1.0 - self.s_min)
        ds = self.penalty * phys ** (self.penalty - 1.0) * (1.0 - self.s_min)
        k = sp.coo_matrix(((self.ke * s[:, None, None]).ravel(), (self.rows, self.cols)), (self.ndof, self.ndof))
        k = k.tocsc()
        f = np.zeros(self.ndof)
        np.add.at(f, self.dofs, self.fe * phys[:, None])
        u = self._solve(k, f)
        magnitude = np.linalg.norm(u.reshape(-1, 2), axis=1)
        j, dj = p_norm(magnitude, self.p)
        if j == 0.0:
            return 0.0, np.zeros_like(phys)
        factor = np.divide(dj, magnitude, out=np.zeros_like(magnitude), where=magnitude > 0.0)
        lam = self._solve(k, (factor[:, None] * u.reshape(-1, 2)).ravel())
        ue, le = u[self.dofs], lam[self.dofs]
        return j, np.sum(le * self.fe, axis=1) - ds * np.einsum('ei,eij,ej->e', le, self.ke, ue)

    def specific(self, phys: np.ndarray) -> tuple[float, np.ndarray]:
        """``(J / Q, d / d phys)``, ``Q = sum phys A``: the response PER UNIT LOAD."""
        x, slope = _read(self.read_on, phys)
        return _per_unit_load(*self._response(x), x, self.area, slope)

    def log_ratio(self, phys: np.ndarray) -> tuple[float, np.ndarray]:
        """``(log(r / r_solid), its gradient)``, ``r`` :meth:`specific`; a design is connected at ``<= log(gamma)``."""
        return _log_ratio(*self.specific(phys), self.reference)


def _read(read_on: Threshold | None, phys: np.ndarray) -> tuple[np.ndarray, np.ndarray | float]:
    """``(the density a measure reads, its slope)``: *phys* itself, or its declared threshold."""
    phys = np.asarray(phys, dtype=float)
    return (phys, 1.0) if read_on is None else read_on(phys)


def _per_unit_load(
    j: float, dj: np.ndarray, x: np.ndarray, measures: np.ndarray, slope: np.ndarray | float
) -> tuple[float, np.ndarray]:
    """``(J / Q, d / d phys)`` with ``Q = sum x m`` the load at the read density *x* (``d x / d phys`` *slope*)."""
    q = float(x @ measures)
    if q == 0.0:
        return 0.0, np.zeros_like(x)
    return j / q, (dj / q - j * measures / q**2) * slope


def _log_ratio(r: float, dr: np.ndarray, reference: float) -> tuple[float, np.ndarray]:
    if r == 0.0:
        return -np.inf, np.zeros_like(dr)
    return float(np.log(r / reference)), dr / r
