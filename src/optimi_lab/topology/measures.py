"""Manufacturability measures of a density design on its cell graph: bodies, member widths, connectivity.

* :func:`component_labels` / :func:`components` -- the bodies of a 0/1 design, connected through the
  domain's graph (on a face graph, cells touching at a corner or an edge only are separate bodies).
* :func:`member_widths` / :func:`feature_sizes` -- the narrowest solid member and void gap, in the
  domain's length unit. ``d`` is each cell's distance to the other phase (centroid to centroid, or to the
  outer boundary through ``boundary_distances`` when the outside IS the other phase); a RIDGE cell holds a
  local maximum of ``d`` over its face neighbours -- a member's centre line -- and reads ``2 d - h``
  (``h`` the cell size). A tapering tip or a corner is not a ridge, so it never reads as a zero-width
  member; a ridge on a boundary open to the same phase is not a member centre and is not read.
* :class:`VirtualTemperature` -- a differentiable connectivity measure (Liu et al. 2015, Front. Mech.
  Eng. 10:126; Li et al. 2016, Struct. Multidisc. Optim. 54:971): solid conducts and carries a uniform
  source, void insulates (``s_min``), the *support* CELLS are the sink (held at zero temperature, wherever
  they sit: a boundary layer or an interior anchor). An island sheds
  its heat through void only, so its temperature rises by ``~1 / s_min``; the smooth maximum
  (:func:`~optimi_lab.topology.interpolation.p_norm`) of the cell temperatures per unit source, against
  the solid design's, is :meth:`~VirtualTemperature.log_ratio`, held ``<= log(gamma)``. Cell-centred two-point
  fluxes across shared FACES with a harmonic-mean conductance, so it and :func:`components` mean the
  same thing by connected. ``read_on`` (:class:`~optimi_lab.topology.density.Threshold`) makes it read the
  BUILT design rather than the density, so gray cells that build to void do not conduct.
* :func:`volume_fraction` -- the measure-weighted solid share, value and gradient.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from numerics_lab.linear import solve_linear_system

from optimi_lab.core.errors import refuse
from optimi_lab.topology.density import Threshold
from optimi_lab.topology.domain import DesignDomain
from optimi_lab.topology.interpolation import p_norm

__all__ = [
    'PENALTY',
    'P_NORM',
    'S_MIN',
    'VirtualTemperature',
    'component_labels',
    'components',
    'feature_sizes',
    'member_widths',
    'volume_fraction',
]

_CHUNK = 2048
#: Defaults of the connectivity measures: the SIMP exponent of the conductance (stiffness), the void's
#: conductance (stiffness) relative to solid, and the p-norm exponent of the smooth maximum.
PENALTY = 3.0
S_MIN = 1e-6
P_NORM = 8.0


def component_labels(domain: DesignDomain, solid: np.ndarray) -> np.ndarray:
    """Each cell's body: ``0..k-1`` over the *solid* cells connected through the domain graph, ``-1`` on void."""
    solid = np.asarray(solid, dtype=bool)
    a, b = domain.adjacency.T
    keep = solid[a] & solid[b]
    a, b = a[keep], b[keep]
    labels = np.arange(domain.n_cells)
    while True:
        low = np.minimum(labels[a], labels[b])
        updated = labels.copy()
        np.minimum.at(updated, a, low)
        np.minimum.at(updated, b, low)
        updated = updated[updated]
        if np.array_equal(updated, labels):
            out = np.full(domain.n_cells, -1)
            out[solid] = np.unique(labels[solid], return_inverse=True)[1]
            return out
        labels = updated


def components(domain: DesignDomain, solid: np.ndarray) -> int:
    """The number of bodies among the *solid* cells (:func:`component_labels`); 0 for an empty design."""
    return int(component_labels(domain, solid).max()) + 1


def _nearest(points: np.ndarray, targets: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """``min_j |p_i - t_j| + offset_j`` for each point (``inf`` when there is no target)."""
    out = np.full(len(points), np.inf)
    if len(targets) == 0:
        return out
    for start in range(0, len(points), _CHUNK):
        block = points[start : start + _CHUNK]
        dist = np.sqrt(((block[:, None, :] - targets[None, :, :]) ** 2).sum(axis=2)) + offsets[None, :]
        out[start : start + _CHUNK] = dist.min(axis=1)
    return out


def member_widths(domain: DesignDomain, phase: np.ndarray, *, exterior: bool = False) -> np.ndarray:
    """Widths of the members of *phase* (True = the phase), in the domain's length unit.

    *exterior*: whether the outside of the domain is this phase (else it is the other one).
    """
    phase = np.asarray(phase, dtype=bool)
    c = domain.centroids
    other = np.flatnonzero(~phase)
    d = _nearest(c, c[other], np.zeros(other.size))
    if not exterior:
        d = np.minimum(d, _nearest(c, c[domain.boundary_cells], domain.boundary_distances))
    d[~phase] = 0.0
    a, b = domain.adjacency.T
    neighbour = np.zeros(domain.n_cells)
    np.maximum.at(neighbour, a, d[b])
    np.maximum.at(neighbour, b, d[a])
    ridge = phase & (d >= neighbour) & np.isfinite(d) & (d > 0.0)
    if exterior:
        ridge[domain.boundary_cells] = False
    return 2.0 * d[ridge] - domain.sizes[ridge]


def feature_sizes(domain: DesignDomain, solid: np.ndarray) -> tuple[float, float]:
    """``(narrowest solid member, narrowest void gap)``; the outside is void, so void open to it is no gap."""
    solid = np.asarray(solid, dtype=bool)
    out = []
    for phase, exterior in ((solid, False), (~solid, True)):
        w = member_widths(domain, phase, exterior=exterior)
        out.append(float(w.min()) if w.size else float('inf'))
    return out[0], out[1]


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

    def _conductance(self, phys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        s = self.s_min + phys**self.penalty * (1.0 - self.s_min)
        return s, self.penalty * phys ** (self.penalty - 1.0) * (1.0 - self.s_min)

    def response(self, phys: np.ndarray) -> tuple[float, np.ndarray]:
        """``(J, dJ / d phys)``, the smooth maximum cell temperature, read through ``read_on`` when declared."""
        x, slope = _read(self.read_on, phys)
        j, dj = self._response(x)
        return j, dj * slope

    def _response(self, phys: np.ndarray) -> tuple[float, np.ndarray]:
        """``J`` at the density the measure reads; adjoint ``K lam = dJ/dT`` (``K`` symmetric)."""
        n = self.domain.n_cells
        s, ds = self._conductance(phys)
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


def volume_fraction(domain: DesignDomain, physical: np.ndarray) -> tuple[float, np.ndarray]:
    """``(sum x m / sum m, its gradient)``: the solid share of the domain, each cell weighted by its measure."""
    share = domain.measures / float(domain.measures.sum())
    return float(np.asarray(physical, dtype=float) @ share), share.copy()
