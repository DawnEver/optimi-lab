"""Manufacturability measures of a density design on its cell graph: bodies, member widths, connectivity.

* :func:`components` -- face-connected bodies of a 0/1 design (cells touching at a corner or an edge
  only are separate bodies).
* :func:`member_widths` / :func:`feature_sizes` -- the narrowest solid member and void gap, in the
  domain's length unit. ``d`` is each cell's distance to the other phase (centroid to centroid, or to the
  outer boundary through ``boundary_distances`` when the outside IS the other phase); a RIDGE cell holds a
  local maximum of ``d`` over its face neighbours -- a member's centre line -- and reads ``2 d - h``
  (``h`` the cell size). A tapering tip or a corner is not a ridge, so it never reads as a zero-width
  member; a ridge on a boundary open to the same phase is not a member centre and is not read.
* :class:`VirtualTemperature` -- a differentiable connectivity measure (Liu et al. 2015, Front. Mech.
  Eng. 10:126; Li et al. 2016, Struct. Multidisc. Optim. 54:971): solid conducts and carries a uniform
  source, void insulates (``s_min``), the *support* cells' boundary faces are the sink. An island sheds
  its heat through void only, so its temperature rises by ``~1 / s_min``; the smooth maximum
  (:func:`~optimi_lab.topology.interpolation.p_norm`) of the cell temperatures per unit source, against
  the solid design's, is the constraint ``log(r / r_solid) <= log(gamma)``. Cell-centred two-point
  fluxes across shared FACES with a harmonic-mean conductance, so it and :func:`components` mean the
  same thing by connected.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from numerics_lab.linear import solve_linear_system

from optimi_lab.core.errors import refuse
from optimi_lab.topology.domain import DesignDomain
from optimi_lab.topology.interpolation import p_norm

__all__ = ['VirtualTemperature', 'components', 'feature_sizes', 'member_widths']

_CHUNK = 2048


def components(domain: DesignDomain, solid: np.ndarray) -> int:
    """Face-connected bodies among the *solid* cells; 0 for an empty design."""
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
            return int(np.unique(labels[solid]).size)
        labels = updated


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
    """The virtual-temperature connectivity measure, sunk through the *support* cells' boundary faces."""

    def __init__(
        self, domain: DesignDomain, support: np.ndarray, *, penalty: float = 3.0, s_min: float = 1e-6, p: float = 8.0
    ) -> None:
        on = np.asarray(support, dtype=bool)[domain.boundary_cells]
        if not on.any():
            refuse('the support cells carry no boundary face to sink through')
        a, b = domain.adjacency.T
        self.domain, self.penalty, self.s_min, self.p = domain, float(penalty), float(s_min), float(p)
        self.w_pair = domain.faces / np.linalg.norm(domain.centroids[a] - domain.centroids[b], axis=1)
        self.sink = domain.boundary_cells[on]
        self.w_sink = domain.boundary_faces[on] / domain.boundary_distances[on]
        self.reference = self.specific(np.ones(domain.n_cells))[0]

    def _conductance(self, phys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        s = self.s_min + phys**self.penalty * (1.0 - self.s_min)
        return s, self.penalty * phys ** (self.penalty - 1.0) * (1.0 - self.s_min)

    def response(self, phys: np.ndarray) -> tuple[float, np.ndarray]:
        """``(J, dJ / d phys)``, the smooth maximum cell temperature; adjoint ``K lam = dJ/dT`` (``K`` symmetric)."""
        phys = np.asarray(phys, dtype=float)
        n = self.domain.n_cells
        s, ds = self._conductance(phys)
        a, b = self.domain.adjacency.T
        g = self.w_pair * 2.0 * s[a] * s[b] / (s[a] + s[b])
        gs = self.w_sink * s[self.sink]
        k = sp.coo_matrix(
            (
                np.concatenate([g, g, -g, -g, gs]),
                (np.concatenate([a, b, a, b, self.sink]), np.concatenate([a, b, b, a, self.sink])),
            ),
            (n, n),
        ).tocsc()
        area = self.domain.measures
        t = np.asarray(solve_linear_system(k, phys * area), dtype=float).reshape(-1)
        j, dj = p_norm(t, self.p)
        if j == 0.0:
            return 0.0, np.zeros(n)
        lam = np.asarray(solve_linear_system(k, dj), dtype=float).reshape(-1)
        dt, dl = t[a] - t[b], lam[a] - lam[b]
        dg_da = self.w_pair * 2.0 * s[b] ** 2 / (s[a] + s[b]) ** 2
        dg_db = self.w_pair * 2.0 * s[a] ** 2 / (s[a] + s[b]) ** 2
        grad = lam * area
        np.add.at(grad, a, -dg_da * ds[a] * dl * dt)
        np.add.at(grad, b, -dg_db * ds[b] * dl * dt)
        np.add.at(grad, self.sink, -self.w_sink * ds[self.sink] * lam[self.sink] * t[self.sink])
        return j, grad

    def specific(self, phys: np.ndarray) -> tuple[float, np.ndarray]:
        """``(J / Q, d / d phys)``, ``Q = sum phys V`` the total source: the response PER UNIT LOAD.

        ``J`` alone is linear in the source, so removing material always lowers it; per unit load a
        connected design is not cheaper for being smaller.
        """
        phys = np.asarray(phys, dtype=float)
        j, dj = self.response(phys)
        q = float(phys @ self.domain.measures)
        if q == 0.0:
            return 0.0, np.zeros_like(phys)
        return j / q, dj / q - j * self.domain.measures / q**2

    def constraint(self, phys: np.ndarray, gamma: float) -> tuple[float, np.ndarray]:
        """``(log(r / r_solid) - log(gamma), its gradient)``, ``r`` :meth:`specific`: ``<= 0`` is connected."""
        r, dr = self.specific(phys)
        if r == 0.0:
            return -np.inf, np.zeros_like(dr)
        return float(np.log(r / self.reference) - np.log(gamma)), dr / r
