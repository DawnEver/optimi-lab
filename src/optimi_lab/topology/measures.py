"""Geometric measures of a density design on its cell graph: bodies, member widths, the solid share.

* :func:`component_labels` / :func:`components` -- the bodies of a 0/1 design, connected through the
  domain's graph (on a face graph, cells touching at a corner or an edge only are separate bodies).
* :func:`member_widths` / :func:`feature_sizes` -- the narrowest solid member and void gap, in the
  domain's length unit. ``d`` is each cell's distance to the other phase (centroid to centroid, or to the
  outer boundary through ``boundary_distances`` when the outside IS the other phase); a RIDGE cell holds a
  local maximum of ``d`` over its graph neighbours -- a member's centre line -- and reads ``2 d - h``
  (``h`` the cell size). A tapering tip or a corner is not a ridge, so it never reads as a zero-width
  member; a ridge on a boundary open to the same phase is not a member centre and is not read.
* :func:`volume_fraction` -- the measure-weighted solid share, value and gradient.

The differentiable connectivity measures live in :mod:`~optimi_lab.topology.connectivity`.
"""

from __future__ import annotations

import numpy as np

from optimi_lab.topology.domain import DesignDomain

__all__ = ['component_labels', 'components', 'feature_sizes', 'member_widths', 'volume_fraction']

_CHUNK = 2048


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


def volume_fraction(domain: DesignDomain, physical: np.ndarray) -> tuple[float, np.ndarray]:
    """``(sum x m / sum m, its gradient)``: the solid share of the domain, each cell weighted by its measure."""
    share = domain.measures / float(domain.measures.sum())
    return float(np.asarray(physical, dtype=float) @ share), share.copy()
