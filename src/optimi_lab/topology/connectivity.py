"""Differentiable connectivity measures: is every solid cell joined to a support through solid?

Both measures load the solid with its own weight (a heat source, a body force), let void carry almost nothing
(``s_min``) and read the smooth maximum (:func:`~optimi_lab.topology.interpolation.p_norm`) of the response PER
UNIT LOAD against the solid design's: :meth:`log_ratio` ``= log(r / r_solid)``, a design is connected at
``<= log(gamma)``. An island sheds its load through void only, so its hottest cell is at least its load over its
boundary's void conductance -- ``~ (cell / domain)^2 / s_min`` for a one-cell island, NOT ``~1 / s_min``: a fixed
``s_min`` admits a small enough island, and a cell just under the build threshold, read at beta 16, conducts
``H(eta)^penalty = 1/8`` of solid whatever ``s_min`` is.

So a :class:`VirtualTemperature` declared with ``gamma`` ENFORCES one body per sink region: its source and
conductance follow ONE interpolation law, and it derives the void conductance so that
:attr:`VirtualTemperature.island_floor` -- a proven lower bound on ``log_ratio`` over every design whose BUILT
form holds a solid cell cut off from every sink by void -- exceeds ``log(gamma)`` by ``log ISLAND_MARGIN``.

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
from optimi_lab.topology.density import NOMINAL_ETA, Threshold
from optimi_lab.topology.domain import DesignDomain
from optimi_lab.topology.interpolation import p_norm

__all__ = ['ISLAND_MARGIN', 'PENALTY', 'P_NORM', 'S_MIN', 'SelfWeightSupport', 'VirtualTemperature']

#: Defaults of the connectivity measures: the SIMP exponent of the conductance (stiffness), the void's
#: conductance (stiffness) relative to solid, and the p-norm exponent of the smooth maximum.
PENALTY = 3.0
S_MIN = 1e-6
P_NORM = 8.0
#: The factor on the ratio by which a declared ``gamma`` sits below the island floor it derives.
ISLAND_MARGIN = 10.0


class VirtualTemperature:
    """The virtual-temperature connectivity measure, sunk at the *support* cells (zero temperature there).

    ONE interpolation law: a cell's heat source and its conductance are the same quantity
    ``s(c) = s_min + (1 - s_min) c^penalty``, ``c`` the density read on ``read_on`` (the density when ``None``).
    A gray cell heats exactly as much as it conducts, so it cannot read as a hot one-cell island.

    The island floor. Let the BUILT form (``x > eta``, ``eta`` the build threshold of ``read_on``) hold an island
    ``I`` reaching no sink, every face neighbour of ``I`` outside it at density ``<= v``. Each island cell sources
    at least ``s_eta = s(H(eta))`` per unit measure; a face onto a neighbour conducts the harmonic mean
    ``2 s_i s_n / (s_i + s_n) <= 2 s(H(v))`` per ``face / distance``; temperatures are ``>= 0``, so the island's
    load leaves through its boundary at most at ``max T`` times its boundary conductance, and
    ``max T >= s_eta C / (2 s(H(v)))``, ``C`` the smallest one-cell ``measure / sum(face / distance)``. With
    ``J >= max T`` and the total load ``Q <= |domain|``::

        log_ratio >= floor(v) = log(s_eta C / (2 s(H(v)) |domain| r_solid))

    :attr:`island_floor` is ``floor(0)``: ``H(0) = 0`` so ``s(H(0)) = s_min`` -- an island cut off by EXACT void.
    Declared *gamma* (which needs ``read_on``) derives ``s_min = target / 2``, ``target = H(eta)^penalty C /
    (2 |domain| r_solid gamma ISLAND_MARGIN)``, so ``island_floor > log(gamma ISLAND_MARGIN)``; a given ``s_min >=
    target`` is refused. Gray is stated, not hidden: a neighbour read near ``eta`` conducts nearly as well as the
    island heats, so ``floor(v)`` falls toward ``log(C / (2 |domain| r_solid))`` as ``v -> eta`` -- a bridge
    just under the build threshold reads as joined, the price of a gray cell never reading as an island.
    Separately anchored pieces (one sink region each) are not islands: N anchors admit N bodies.
    """

    def __init__(
        self,
        domain: DesignDomain,
        support: np.ndarray,
        *,
        penalty: float = PENALTY,
        s_min: float | None = None,
        p: float = P_NORM,
        read_on: Threshold | None = None,
        gamma: float | None = None,
    ) -> None:
        sink = np.asarray(support, dtype=bool)
        if sink.shape != (domain.n_cells,) or not sink.any():
            refuse(f'the support names no sink cell among the {domain.n_cells} (a mask of one bool per cell)')
        if sink.all():
            refuse('every cell is a sink: there is no temperature to measure')
        a, b = domain.adjacency.T
        self.domain, self.penalty, self.p = domain, float(penalty), float(p)
        self.s_min = S_MIN if s_min is None else float(s_min)
        self.read_on = read_on
        self.w_pair = domain.faces / np.linalg.norm(domain.centroids[a] - domain.centroids[b], axis=1)
        self.free = np.flatnonzero(~sink)
        self.reference = self.specific(np.ones(domain.n_cells))[0]
        conductance = np.zeros(domain.n_cells)
        np.add.at(conductance, a, self.w_pair)
        np.add.at(conductance, b, self.w_pair)
        joined = self.free[conductance[self.free] > 0.0]
        # min(measure / sum(face / distance)) over the non-sink cells: an island's load over its boundary's
        # conductance is a ratio of sums over its cells, so at least the smallest one-cell ratio.
        self._capacity = float(np.min(domain.measures[joined] / conductance[joined]))
        if gamma is not None:
            self._enforce(float(gamma), given_s_min=s_min is not None)

    @property
    def _eta(self) -> float:
        """The build threshold: a cell above it is built solid."""
        return NOMINAL_ETA if self.read_on is None else self.read_on.eta

    def _law(self, c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``(s(c), ds / dc)``: the one law a cell's source and its conductance both follow."""
        return (
            self.s_min + (1.0 - self.s_min) * c**self.penalty,
            self.penalty * (1.0 - self.s_min) * c ** (self.penalty - 1.0),
        )

    def floor(self, void: float = 0.0) -> float:
        """The lower bound on :meth:`log_ratio` over designs with a built island whose neighbours read ``<= void``."""
        s_eta, s_void = self._law(_read(self.read_on, np.array([self._eta, void]))[0])[0]
        total = float(self.domain.measures.sum())
        return float(np.log(s_eta * self._capacity / (2.0 * s_void * total * self.reference)))

    @property
    def island_floor(self) -> float:
        """:meth:`floor` at exact void: the bound for any design whose built island is cut off by void."""
        return self.floor(0.0)

    def _enforce(self, gamma: float, *, given_s_min: bool) -> None:
        """Derive ``s_min`` (unless given) so the island floor clears ``log(gamma ISLAND_MARGIN)``."""
        if self.read_on is None:
            refuse('an enforced gamma needs read_on: the build threshold an island is defined by')
        if not gamma > 1.0:
            refuse(f'gamma bounds the ratio to the solid design from above and must exceed 1, got {gamma}')
        lit = float(_read(self.read_on, np.array([self._eta]))[0][0]) ** self.penalty  # s_eta at s_min = 0
        total = float(self.domain.measures.sum())
        target = lit * self._capacity / (2.0 * total * self.reference * gamma * ISLAND_MARGIN)
        if given_s_min and self.s_min >= target:
            refuse(
                f'a void conductance s_min = {self.s_min:.3g} admits an island under gamma = {gamma:g}: the island '
                f'floor needs a void cell to conduct below {target:.3g}'
            )
        if not given_s_min:
            self.s_min = 0.5 * target

    def _solve(self, k: sp.csc_matrix, rhs: np.ndarray) -> np.ndarray:
        """``K_ff x_f = rhs_f`` with the sink cells held at zero; the full vector."""
        out = np.zeros(self.domain.n_cells)
        f = self.free
        out[f] = np.asarray(solve_linear_system(k[f][:, f], rhs[f]), dtype=float).reshape(-1)
        return out

    def response(self, phys: np.ndarray) -> tuple[float, np.ndarray]:
        """``(J, dJ / d phys)``, the smooth maximum cell temperature under the one law on the read density."""
        c, slope = _read(self.read_on, phys)
        s, ds = self._law(c)
        j, dj = self._response(s)
        return j, dj * ds * slope

    def _response(self, s: np.ndarray) -> tuple[float, np.ndarray]:
        """``(J, dJ / ds)``, *s* each cell's source and conductance; adjoint ``K lam = dJ/dT`` (``K`` symmetric)."""
        n = self.domain.n_cells
        a, b = self.domain.adjacency.T
        g = self.w_pair * 2.0 * s[a] * s[b] / (s[a] + s[b])
        k = sp.coo_matrix(
            (np.concatenate([g, g, -g, -g]), (np.concatenate([a, b, a, b]), np.concatenate([a, b, b, a]))), (n, n)
        ).tocsc()
        area = self.domain.measures
        t = self._solve(k, s * area)
        j, dj = p_norm(t, self.p)
        if j == 0.0:
            return 0.0, np.zeros(n)
        lam = self._solve(k, dj)
        dt, dl = t[a] - t[b], lam[a] - lam[b]
        dg_da = self.w_pair * 2.0 * s[b] ** 2 / (s[a] + s[b]) ** 2
        dg_db = self.w_pair * 2.0 * s[a] ** 2 / (s[a] + s[b]) ** 2
        d_s = lam * area
        np.add.at(d_s, a, -dg_da * dl * dt)
        np.add.at(d_s, b, -dg_db * dl * dt)  # a pair onto a sink cell: t = lam = 0 there
        return j, d_s

    def specific(self, phys: np.ndarray) -> tuple[float, np.ndarray]:
        """``(J / Q, d / d phys)``, ``Q = sum s V`` the total source: the response PER UNIT LOAD.

        ``J`` alone is linear in the source, so removing material always lowers it; per unit load a
        connected design is not cheaper for being smaller.
        """
        c, slope = _read(self.read_on, phys)
        s, ds = self._law(c)
        return _per_unit_load(*self._response(s), s, self.domain.measures, ds * slope)

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
