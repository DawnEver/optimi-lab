"""The design domain: the cells a density lives on, as geometry and a graph -- nothing about the physics.

A :class:`DesignDomain` is what every map of the density method needs and no more:

* ``centroids (n, d)`` and ``measures (n,)`` (area in 2-D, volume in 3-D), in ONE declared
  ``length_unit`` -- a filter radius or a member width is a length, and a length carries its unit;
* the cell GRAPH: ``adjacency (k, 2)`` pairs of cells sharing a face and ``faces (k,)`` that face's
  measure; the outer boundary as ``boundary_cells``, ``boundary_faces`` and ``boundary_distances``
  (centroid to face) -- what a two-point flux on the cells needs, and nothing a mesh format implies;
* PASSIVE masks: ``solid`` cells held at density 1, ``void`` cells at 0;
* ``design_index (n,)``: the design variable each cell reads. Symmetry and periodicity are a many-to-one
  ``design_index`` (:meth:`mirrored` builds the mirror one); :meth:`expand` and :meth:`reduce` are the
  map and its adjoint.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from optimi_lab.core.errors import refuse

__all__ = ['DesignDomain']


def _frozen(value, dtype, shape_head: tuple[int, ...] | None = None) -> np.ndarray:
    array = np.array(value, dtype=dtype)
    if shape_head is not None and array.shape[: len(shape_head)] != shape_head:
        refuse(f'expected an array of leading shape {shape_head}, got {array.shape}')
    array.setflags(write=False)
    return array


@dataclass(frozen=True, eq=False)
class DesignDomain:
    """The cells of a density design; see the module docstring."""

    centroids: np.ndarray
    measures: np.ndarray
    adjacency: np.ndarray
    faces: np.ndarray
    length_unit: str
    boundary_cells: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    boundary_faces: np.ndarray = field(default_factory=lambda: np.zeros(0))
    boundary_distances: np.ndarray = field(default_factory=lambda: np.zeros(0))
    solid: np.ndarray | None = None
    void: np.ndarray | None = None
    design_index: np.ndarray | None = None

    def __post_init__(self) -> None:
        centroids = _frozen(self.centroids, float)
        n = len(centroids)
        if centroids.ndim != 2 or n == 0:
            refuse(f'centroids must be a non-empty (n, d) array, got shape {centroids.shape}')
        if not self.length_unit:
            refuse('a design domain declares its length unit')
        adjacency = _frozen(np.reshape(self.adjacency, (-1, 2)), int)
        b = len(np.atleast_1d(self.boundary_cells))
        parts = {
            'centroids': centroids,
            'measures': _frozen(self.measures, float, (n,)),
            'adjacency': adjacency,
            'faces': _frozen(self.faces, float, (len(adjacency),)),
            'boundary_cells': _frozen(self.boundary_cells, int, (b,)),
            'boundary_faces': _frozen(self.boundary_faces, float, (b,)),
            'boundary_distances': _frozen(self.boundary_distances, float, (b,)),
            'solid': _frozen(np.zeros(n, bool) if self.solid is None else self.solid, bool, (n,)),
            'void': _frozen(np.zeros(n, bool) if self.void is None else self.void, bool, (n,)),
            'design_index': _frozen(np.arange(n) if self.design_index is None else self.design_index, int, (n,)),
        }
        if np.any(parts['measures'] <= 0.0):
            refuse('every cell measure must be positive')
        if adjacency.size and (adjacency.min() < 0 or adjacency.max() >= n):
            refuse(f'adjacency names a cell outside 0..{n - 1}')
        if np.any(parts['solid'] & parts['void']):
            refuse(f'cells {np.flatnonzero(parts["solid"] & parts["void"]).tolist()} are both solid and void')
        index = parts['design_index']
        if index.min() != 0 or np.unique(index).size != index.max() + 1:
            refuse('design_index must number the design variables 0..m-1 with none unused')
        for name, value in parts.items():
            object.__setattr__(self, name, value)

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

    def expand(self, design: np.ndarray) -> np.ndarray:
        """Design variables -> one value per cell."""
        return np.asarray(design, dtype=float)[self.design_index]

    def reduce(self, per_cell: np.ndarray) -> np.ndarray:
        """The adjoint of :meth:`expand`: per-cell sensitivities summed onto their design variable."""
        return np.bincount(self.design_index, weights=per_cell, minlength=self.n_design)

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
        return replace(self, design_index=np.unique(first, return_inverse=True)[1])

    @classmethod
    def grid(
        cls,
        shape: tuple[int, ...],
        *,
        spacing: float,
        length_unit: str,
        solid: np.ndarray | None = None,
        void: np.ndarray | None = None,
    ) -> DesignDomain:
        """A structured grid of ``prod(shape)`` square (cubic) cells, numbered C-order (last axis fastest)."""
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
        adjacency = np.vstack(pairs)
        cells = np.concatenate(boundary)
        return cls(
            centroids=centroids,
            measures=np.full(index.size, spacing ** len(shape)),
            adjacency=adjacency,
            faces=np.full(len(adjacency), face),
            length_unit=length_unit,
            boundary_cells=cells,
            boundary_faces=np.full(cells.size, face),
            boundary_distances=np.full(cells.size, 0.5 * spacing),
            solid=solid,
            void=void,
        )
