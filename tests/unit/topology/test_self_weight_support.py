"""Both connectivity measures on planted 0/1 designs over a triangle mesh, and the self-weight adjoint.

A 12 x 12 mm square of right triangles, held at a 2 x 2 mm boss on the middle of its left edge: the
self-weight support clamps the boss's edge NODES, the virtual temperature sinks at the cells on that edge.
Each planted design is one body with the boss or not; a measure must say the same. The one legitimate
miss: P1 elasticity carries load through a shared vertex, so only the cell-centred thermal measure refuses
a block hung on two vertex hinges (a strict xfail, so a measure that starts refusing it is noticed). One
pin alone is a mechanism -- the block swings -- and the structural measure does refuse that.
"""

from __future__ import annotations

import numpy as np
import pytest

from optimi_lab.core import Refusal
from optimi_lab.topology import (
    BUILT_READ,
    DesignDomain,
    SelfWeightSupport,
    VirtualTemperature,
    components,
    fd_check,
)

N = 12
#: The connectivity bound per measure (an island exceeds it by orders of magnitude).
GAMMA = {'thermal': 30.0, 'structural': 40.0}


def _mesh() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(vertices, cells, square x, square y, upper-left?)``: each unit square split by its rising diagonal."""
    i, j = np.meshgrid(np.arange(N + 1), np.arange(N + 1), indexing='ij')
    vertices = np.column_stack([i.ravel(), j.ravel()]).astype(float)
    v = np.arange((N + 1) ** 2).reshape(N + 1, N + 1)
    si, sj = (a.ravel() for a in np.meshgrid(np.arange(N), np.arange(N), indexing='ij'))
    lower = np.column_stack([v[si, sj], v[si + 1, sj], v[si + 1, sj + 1]])
    upper = np.column_stack([v[si, sj], v[si + 1, sj + 1], v[si, sj + 1]])
    cells = np.vstack([lower, upper])
    return vertices, cells, np.tile(si, 2), np.tile(sj, 2), np.repeat([False, True], N * N)


VERTICES, CELLS, SX, SY, UPPER = _mesh()
DOMAIN = DesignDomain.from_simplices(VERTICES, CELLS, 'mm')
BOSS = (SX < 2) & (SY >= 5) & (SY < 7)
MOUNT_NODES = (VERTICES[:, 0] == 0.0) & (VERTICES[:, 1] >= 5.0) & (VERTICES[:, 1] <= 7.0)
MOUNT_CELLS = MOUNT_NODES[CELLS].sum(axis=1) >= 2


def _planted() -> dict[str, tuple[np.ndarray, bool]]:
    guides = (SY < 3) | (SY >= 9)
    spine = SX < 2
    # A block under the boss whose top row keeps only the lower-right triangles: each touches the boss at ONE
    # vertex, (1, 5) and (2, 5) -- two pins, so the block is held without a moment, through no face.
    hung = ((SX < 2) & (SY >= 1) & (SY < 4)) | (~UPPER & (SX < 2) & (SY == 4))
    return {
        'C core': (guides | spine | BOSS, True),
        'two guides, no spine': (guides | BOSS, False),
        'C core + island': (guides | spine | BOSS | ((SX >= 6) & (SX < 9) & (SY >= 5) & (SY < 7)), False),
        'block on two vertex hinges': (hung | BOSS, False),
    }


PLANTED = _planted()


def _measure(kind: str):
    if kind == 'thermal':
        return VirtualTemperature(DOMAIN, MOUNT_CELLS)
    return SelfWeightSupport(VERTICES, CELLS, MOUNT_NODES)


_HINGE = pytest.mark.xfail(
    strict=True, reason='P1 elasticity carries load through a shared vertex: only the thermal measure refuses a hinge'
)


@pytest.mark.parametrize(
    ('kind', 'name'),
    [
        pytest.param(kind, name, marks=_HINGE if (kind, name) == ('structural', 'block on two vertex hinges') else ())
        for kind in GAMMA
        for name in PLANTED
    ],
)
def test_a_planted_design_meets_the_bound_iff_it_is_one_body(kind: str, name: str) -> None:
    cells, one_body = PLANTED[name]
    assert (components(DOMAIN, cells) == 1) == one_body
    value, _ = _measure(kind).log_ratio(cells.astype(float))
    assert (value <= np.log(GAMMA[kind])) == one_body, value


def _agrees(rows, rtol: float) -> bool:
    scale = max(abs(fd) for _, _, fd in rows)
    return all(abs(a - fd) <= rtol * scale for _, a, fd in rows)


@pytest.mark.parametrize('penalty', [1.0, 3.0])
def test_the_self_weight_adjoint_is_the_central_difference_limit(penalty: float) -> None:
    measure = SelfWeightSupport(VERTICES, CELLS, MOUNT_NODES, penalty=penalty)
    phys = np.random.default_rng(0).uniform(0.2, 0.9, DOMAIN.n_cells)
    rows = fd_check(measure.log_ratio, phys, [3, 100, 200, 287], step=1e-5)
    # Measured worst 1.1e-6 of the largest entry (FD truncation at h = 1e-5); a flipped sign errs by 2.
    assert _agrees(rows, rtol=1e-5), rows
    flipped = [(k, -a, fd) for k, a, fd in rows]
    assert not _agrees(flipped, rtol=1e-5), 'planted control: a flipped gradient fails the same check'


def test_the_self_weight_read_on_the_built_design_chains_its_slope() -> None:
    measure = SelfWeightSupport(VERTICES, CELLS, MOUNT_NODES, read_on=BUILT_READ)
    phys = np.random.default_rng(1).uniform(0.4, 0.6, DOMAIN.n_cells)
    assert _agrees(fd_check(measure.log_ratio, phys, [3, 100, 200, 287], step=1e-6), rtol=1e-5)


def test_a_self_weight_support_without_a_clamped_node_is_refused() -> None:
    with pytest.raises(Refusal, match='support node'):
        SelfWeightSupport(VERTICES, CELLS, np.zeros(len(VERTICES), dtype=bool))
