"""The design-domain builders: a simplex mesh as a cell graph, and a grid's neighbourhood.

The simplex golden is derived by hand (a square fanned into four triangles about its centre) and its array
ORDER is the one the first downstream mesh adapter produced, so a caller moving onto
:meth:`DesignDomain.from_simplices` reads identical arrays.
"""

from __future__ import annotations

import numpy as np
import pytest

from optimi_lab.core import Refusal
from optimi_lab.topology import DesignDomain, component_labels, components, feature_sizes

#: A 2 x 2 square, vertices 0..3 counter-clockwise from the origin, 4 at its centre; four fan triangles.
FAN_VERTICES = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0], [1.0, 1.0]])
FAN_CELLS = np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]])


def test_a_triangle_fan_is_its_hand_derived_cell_graph() -> None:
    domain = DesignDomain.from_simplices(FAN_VERTICES, FAN_CELLS, 'mm')
    third = 1.0 / 3.0
    np.testing.assert_allclose(domain.centroids, [[1.0, third], [2.0 - third, 1.0], [1.0, 2.0 - third], [third, 1.0]])
    np.testing.assert_allclose(domain.measures, [1.0, 1.0, 1.0, 1.0])
    # Shared edges sorted by their node pair: (0,4) joins cells 3 and 0, (1,4) 0-1, (2,4) 1-2, (3,4) 2-3.
    np.testing.assert_array_equal(domain.adjacency, [[3, 0], [0, 1], [1, 2], [2, 3]])
    np.testing.assert_allclose(domain.faces, np.full(4, np.sqrt(2.0)))
    # Outer edges (0,1) (0,3) (1,2) (2,3): each 2 long, each a third of the height from its cell's centroid.
    np.testing.assert_array_equal(domain.boundary_cells, [0, 3, 1, 2])
    np.testing.assert_allclose(domain.boundary_faces, np.full(4, 2.0))
    np.testing.assert_allclose(domain.boundary_distances, np.full(4, third))
    assert domain.length_unit == 'mm'


def test_a_vertex_contact_is_no_face_and_passive_masks_pass_through() -> None:
    """Planted control: two triangles touching at ONE vertex are two cells with no adjacency."""
    vertices = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [2.0, 0.0], [1.0, -1.0]])
    domain = DesignDomain.from_simplices(vertices, [[0, 1, 2], [1, 3, 4]], 'm', solid=[True, False])
    assert domain.adjacency.shape == (0, 2)
    assert len(domain.boundary_cells) == 6
    assert components(domain, np.ones(2, dtype=bool)) == 2
    assert domain.solid.tolist() == [True, False] and domain.design_index.tolist() == [-1, 0]


def test_two_tetrahedra_share_one_triangular_face() -> None:
    vertices = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, -1]])
    domain = DesignDomain.from_simplices(vertices, [[0, 1, 2, 3], [0, 1, 2, 4]], 'm')
    np.testing.assert_allclose(domain.measures, [1.0 / 6.0, 1.0 / 6.0])
    np.testing.assert_array_equal(domain.adjacency, [[0, 1]])
    np.testing.assert_allclose(domain.faces, [0.5])
    assert len(domain.boundary_cells) == 6


def test_a_simplex_of_the_wrong_order_is_refused() -> None:
    with pytest.raises(Refusal, match='simplices'):
        DesignDomain.from_simplices(FAN_VERTICES, FAN_CELLS[:, :2], 'mm')


def test_the_full_neighbourhood_joins_corner_contacts_the_face_one_does_not() -> None:
    diagonal = np.eye(3, dtype=bool).ravel()
    face = DesignDomain.grid((3, 3), spacing=1.0, length_unit='mm')
    full = DesignDomain.grid((3, 3), spacing=1.0, length_unit='mm', neighbourhood='full')
    assert components(face, diagonal) == 3
    assert components(full, diagonal) == 1
    assert len(full.adjacency) == len(face.adjacency) + 2 * 2 * 2  # two diagonals per 2 x 2 block, four blocks


def test_the_full_neighbourhood_reads_a_border_strip_and_no_corner_member() -> None:
    """A 1-cell strip along the border is 1 cell wide; a square's corner is no member on the full graph."""
    strip = np.zeros((8, 8), dtype=bool)
    strip[0, :] = True
    full = DesignDomain.grid((8, 8), spacing=0.5, length_unit='mm', neighbourhood='full')
    assert feature_sizes(full, strip.ravel())[0] == pytest.approx(0.5)
    square = np.zeros((12, 12), dtype=bool)
    square[3:8, 3:8] = True  # five cells, 2.5 mm
    full = DesignDomain.grid((12, 12), spacing=0.5, length_unit='mm', neighbourhood='full')
    assert feature_sizes(full, square.ravel())[0] == pytest.approx(2.5)
    face = DesignDomain.grid((12, 12), spacing=0.5, length_unit='mm')
    assert feature_sizes(face, square.ravel())[0] < 2.5, 'planted control: the face graph reads the corner'


def test_an_unknown_neighbourhood_is_refused_naming_the_set() -> None:
    with pytest.raises(Refusal, match="'face', 'full'"):
        DesignDomain.grid((2, 2), spacing=1.0, length_unit='mm', neighbourhood='queen')


def test_component_labels_number_the_bodies_and_mark_void() -> None:
    domain = DesignDomain.grid((4, 4), spacing=1.0, length_unit='mm')
    solid = np.zeros((4, 4), dtype=bool)
    solid[0, :2] = solid[3, 3] = solid[2, 3] = True
    labels = component_labels(domain, solid.ravel()).reshape(4, 4)
    assert labels[0, 0] == labels[0, 1] != labels[3, 3] == labels[2, 3]
    assert sorted(set(labels[solid].tolist())) == [0, 1]
    assert np.all(labels[~solid] == -1)
    assert components(domain, solid.ravel()) == 2
    assert components(domain, np.zeros(16, dtype=bool)) == 0
