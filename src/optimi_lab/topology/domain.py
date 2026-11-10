"""The design domain: the cells a density lives on, as geometry and a graph -- nothing about the physics.

A :class:`DesignDomain` is what every map of the density method needs and no more:

* ``centroids (n, d)`` and ``measures (n,)`` (area in 2-D, volume in 3-D), in ONE declared
  ``length_unit`` -- a filter radius or a member width is a length, and a length carries its unit;
* the cell GRAPH: ``adjacency (k, 2)`` pairs of cells sharing a face and ``faces (k,)`` that face's
  measure; the outer boundary as ``boundary_cells``, ``boundary_faces`` and ``boundary_distances``
  (centroid to face) -- what a two-point flux on the cells needs, and nothing a mesh format implies;
* PASSIVE masks: ``solid`` cells held at density 1, ``void`` cells at 0 -- held BEFORE any filter
  (:meth:`expand` reads their state), so a neighbour's filter sees the declared state, never a gray value;
* ``design_index (n,)``: the design variable each cell reads, ``-1`` on a passive cell (no variable). Symmetry
  and periodicity are a many-to-one ``design_index`` (:meth:`mirrored` builds the mirror one); a variable
  only passive cells read is dropped. :meth:`expand` and :meth:`reduce` are the map and its adjoint.

Two builders: :meth:`DesignDomain.grid` (square / cubic cells; its :class:`Neighbourhood` is the face graph or
every touching cell) and :meth:`DesignDomain.from_simplices` (a triangle or tetrahedron mesh: cells sharing a
whole FACE are adjacent, a vertex- or edge-only contact is not).
"""

from __future__ import annotations

import itertools
import math
from enum import StrEnum

import numpy as np

from optimi_lab.core.errors import coerce_enum, refuse

__all__ = ['DesignDomain', 'Neighbourhood']


class Neighbourhood(StrEnum):
    """Which grid cells a :meth:`DesignDomain.grid` graph joins.

    ``FACE``: cells sharing a face; the outside sits half a cell from a boundary centroid (its face).
    ``FULL``: every touching cell (a corner or an edge too -- the 8-neighbourhood in 2-D), the graph a pixel
    raster's features are read on: a ridge is a maximum over ALL touching cells, so a square corner is no
    member, and the outside is the ring of cells one spacing beyond the border. A diagonal pair's ``faces``
    entry is the axis face's measure: that graph is for reading features, not for two-point fluxes.
    """

    FACE = 'face'
    FULL = 'full'


def _simplex_measure(points: np.ndarray) -> np.ndarray:
    """The k-volume of each simplex ``points (n, k + 1, d)``: ``sqrt(det(E E^T)) / k!`` (Gram determinant)."""
    edges = points[:, 1:] - points[:, :1]
    gram = np.einsum('nid,njd->nij', edges, edges)
    return np.sqrt(np.maximum(np.linalg.det(gram), 0.0)) / math.factorial(edges.shape[1])


def _frozen(value, dtype, shape_head: tuple[int, ...] | None = None) -> np.ndarray:
    array = np.array(value, dtype=dtype)
    if shape_head is not None and array.shape[: len(shape_head)] != shape_head:
        refuse(f'expected an array of leading shape {shape_head}, got {array.shape}')
    array.setflags(write=False)
    return array


class DesignDomain:
    """The cells of a density design; see the module docstring. Every array is stored read-only."""

    def __init__(
        self,
        centroids: np.ndarray,
        measures: np.ndarray,
        adjacency: np.ndarray,
        faces: np.ndarray,
        length_unit: str,
        *,
        boundary_cells: np.ndarray = (),
        boundary_faces: np.ndarray = (),
        boundary_distances: np.ndarray = (),
        solid: np.ndarray | None = None,
        void: np.ndarray | None = None,
        design_index: np.ndarray | None = None,
    ) -> None:
        self.centroids = _frozen(centroids, float)
        n = len(self.centroids)
        if self.centroids.ndim != 2 or n == 0:
            refuse(f'centroids must be a non-empty (n, d) array, got shape {self.centroids.shape}')
        if not length_unit:
            refuse('a design domain declares its length unit')
        self.length_unit = length_unit
        self.measures = _frozen(measures, float, (n,))
        self.adjacency = _frozen(np.reshape(adjacency, (-1, 2)), int)
        self.faces = _frozen(faces, float, (len(self.adjacency),))
        b = len(np.atleast_1d(boundary_cells))
        self.boundary_cells = _frozen(boundary_cells, int, (b,))
        self.boundary_faces = _frozen(boundary_faces, float, (b,))
        self.boundary_distances = _frozen(boundary_distances, float, (b,))
        self.solid = _frozen(np.zeros(n, bool) if solid is None else solid, bool, (n,))
        self.void = _frozen(np.zeros(n, bool) if void is None else void, bool, (n,))
        self.design_index = _frozen(np.arange(n) if design_index is None else design_index, int, (n,))
        if np.any(self.measures <= 0.0):
            refuse('every cell measure must be positive')
        if self.adjacency.size and (self.adjacency.min() < 0 or self.adjacency.max() >= n):
            refuse(f'adjacency names a cell outside 0..{n - 1}')
        if np.any(self.solid & self.void):
            refuse(f'cells {np.flatnonzero(self.solid & self.void).tolist()} are both solid and void')
        index = self.design_index
        if index.min() != 0 or np.unique(index).size != index.max() + 1:
            refuse('design_index must number the design variables 0..m-1 with none unused')
        # A PASSIVE cell is no design variable: it reads -1, and a variable only passive cells read is dropped.
        passive = self.solid | self.void
        if passive.all():
            refuse('every cell is passive: there is nothing to design')
        renumbered = np.full(n, -1, dtype=int)
        renumbered[~passive] = np.unique(index[~passive], return_inverse=True)[1]
        self.design_index = _frozen(renumbered, int, (n,))
        self.state = _frozen(self.solid.astype(float), float, (n,))

    @property
    def n_cells(self) -> int:
        return len(self.centroids)

    @property
    def n_design(self) -> int:
        return int(self.design_index.max()) + 1

    @property
    def dimension(self) -> int:
        return self.centroids.shape[1]

    @property
    def sizes(self) -> np.ndarray:
        """Each cell's characteristic length, ``measure ** (1 / d)``."""
        return self.measures ** (1.0 / self.dimension)

    def require_unit(self, unit: str, what: str) -> None:
        """Refuse a length stated in another unit than this domain's."""
        if unit != self.length_unit:
            refuse(f'{what} is in {unit!r} but the domain is in {self.length_unit!r}; convert it first')

    @property
    def free(self) -> np.ndarray:
        """Mask of the cells that read a design variable (every cell neither solid nor void)."""
        return self.design_index >= 0

    def expand(self, design: np.ndarray) -> np.ndarray:
        """Design variables -> one value per cell; a passive cell reads its declared state (solid 1, void 0)."""
        out = self.state.copy()
        out[self.free] = np.asarray(design, dtype=float)[self.design_index[self.free]]
        return out

    def reduce(self, per_cell: np.ndarray) -> np.ndarray:
        """The adjoint of :meth:`expand`: per-cell sensitivities summed onto their design variable."""
        free = self.free
        return np.bincount(self.design_index[free], weights=np.asarray(per_cell)[free], minlength=self.n_design)

    def mirrored(self, axis: int) -> DesignDomain:
        """A copy of the domain with each cell tied to its mirror image across the mid-plane of *axis*."""
        c = self.centroids
        lo, hi = c[:, axis].min(), c[:, axis].max()
        image = c.copy()
        image[:, axis] = lo + hi - c[:, axis]
        scale = float(np.max(np.ptp(c, axis=0))) or 1.0
        order = np.lexsort(np.round(c / scale, 9).T[::-1])
        partner = np.empty(self.n_cells, dtype=int)
        partner[np.lexsort(np.round(image / scale, 9).T[::-1])] = order
        if not np.allclose(c[partner], image, atol=1e-9 * scale):
            refuse(f'the domain is not mirror-symmetric across axis {axis}')
        first = np.minimum(np.arange(self.n_cells), partner)
        return DesignDomain(
            self.centroids,
            self.measures,
            self.adjacency,
            self.faces,
            self.length_unit,
            boundary_cells=self.boundary_cells,
            boundary_faces=self.boundary_faces,
            boundary_distances=self.boundary_distances,
            solid=self.solid,
            void=self.void,
            design_index=np.unique(first, return_inverse=True)[1],
        )

    @classmethod
    def grid(
        cls,
        shape: tuple[int, ...],
        *,
        spacing: float,
        length_unit: str,
        neighbourhood: Neighbourhood | str = Neighbourhood.FACE,
        solid: np.ndarray | None = None,
        void: np.ndarray | None = None,
    ) -> DesignDomain:
        """A structured grid of ``prod(shape)`` square (cubic) cells, numbered C-order (last axis fastest)."""
        neighbourhood = coerce_enum(Neighbourhood, neighbourhood, 'the grid neighbourhood')
        shape = tuple(int(s) for s in shape)
        index = np.arange(int(np.prod(shape))).reshape(shape)
        axes = np.meshgrid(*[(np.arange(s) + 0.5) * spacing for s in shape], indexing='ij')
        centroids = np.column_stack([a.ravel() for a in axes])
        face = spacing ** (len(shape) - 1)
        pairs, boundary = [], []
        for axis in range(len(shape)):
            lead = np.delete(index, -1, axis=axis).ravel()
            trail = np.delete(index, 0, axis=axis).ravel()
            pairs.append(np.column_stack([lead, trail]))
            boundary += [np.take(index, 0, axis=axis).ravel(), np.take(index, -1, axis=axis).ravel()]
        if neighbourhood is Neighbourhood.FULL:
            # Every offset moving two or more axes, counted once (its first non-zero component positive).
            for offset in itertools.product((-1, 0, 1), repeat=len(shape)):
                moved = [o for o in offset if o]
                if len(moved) < 2 or moved[0] < 0:
                    continue
                lead = index[tuple(slice(max(0, -o), s - max(0, o)) for o, s in zip(offset, shape, strict=True))]
                trail = index[tuple(slice(max(0, o), s - max(0, -o)) for o, s in zip(offset, shape, strict=True))]
                pairs.append(np.column_stack([lead.ravel(), trail.ravel()]))
        adjacency = np.vstack(pairs)
        cells = np.concatenate(boundary)
        outside = spacing if neighbourhood is Neighbourhood.FULL else 0.5 * spacing
        return cls(
            centroids,
            np.full(index.size, spacing ** len(shape)),
            adjacency,
            np.full(len(adjacency), face),
            length_unit,
            boundary_cells=cells,
            boundary_faces=np.full(cells.size, face),
            boundary_distances=np.full(cells.size, outside),
            solid=solid,
            void=void,
        )

    @classmethod
    def from_simplices(
        cls,
        vertices: np.ndarray,
        cells: np.ndarray,
        length_unit: str,
        *,
        solid: np.ndarray | None = None,
        void: np.ndarray | None = None,
        design_index: np.ndarray | None = None,
    ) -> DesignDomain:
        """A simplex mesh -- *cells* ``(n, d + 1)`` indices into *vertices* ``(m, d)`` -- as a design domain.

        Two cells are adjacent when they share a whole face (``d`` vertices); the face measure is its
        (d-1)-volume. A face owned by one cell is an outer face, its distance the cell centroid to the face
        centroid. Shared faces are listed in the order of their sorted vertex tuples.
        """
        vertices = np.asarray(vertices, dtype=float)
        cells = np.asarray(cells, dtype=np.intp)
        d = vertices.shape[1] if vertices.ndim == 2 else 0
        if cells.ndim != 2 or cells.shape[1] != d + 1:
            refuse(f'{d}-D vertices take simplices of {d + 1} vertices, got cells of shape {cells.shape}')
        points = vertices[cells]
        centroids = points.mean(axis=1)
        # Face k starts at vertex k (a triangle's: (0,1), (1,2), (2,0)), each sorted so a shared face matches.
        rolled = [cells[:, [(k + j) % (d + 1) for j in range(d)]] for k in range(d + 1)]
        faces = np.sort(np.concatenate(rolled), axis=1)
        owner = np.tile(np.arange(len(cells)), d + 1)
        order = np.lexsort(faces.T[::-1])
        f, o = faces[order], owner[order]
        same = np.all(f[1:] == f[:-1], axis=1)
        paired = np.zeros(len(f), dtype=bool)
        paired[:-1] |= same
        paired[1:] |= same
        outer_cells, outer = o[~paired], f[~paired]
        return cls(
            centroids,
            _simplex_measure(points),
            np.column_stack([o[:-1][same], o[1:][same]]),
            _simplex_measure(vertices[f[:-1][same]]),
            length_unit,
            boundary_cells=outer_cells,
            boundary_faces=_simplex_measure(vertices[outer]),
            boundary_distances=np.linalg.norm(centroids[outer_cells] - vertices[outer].mean(axis=1), axis=1),
            solid=solid,
            void=void,
            design_index=design_index,
        )
