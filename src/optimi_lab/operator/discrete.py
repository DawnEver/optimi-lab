"""Variation for discrete variables in the unit cube, and for permutation genomes.

A :class:`Lattice` records, for every Integer and embedded Categorical of a
:class:`~optimi_lab.problem.Space`, the unit-cube block it occupies and the coordinates of its
levels; Real and Field columns are free. On it:

* :func:`snap` puts each discrete block on its nearest level (the repair a continuous operator
  needs before its child is a design);
* :func:`discrete_mutation` moves an ORDERED variable (Integer) one level up or down, and resets an
  UNORDERED one (Categorical) to a different level drawn uniformly -- a category has no neighbour.

Permutation genomes are integer rows holding ``0 .. k-1`` once each: :func:`order_crossover` (OX1,
Davis 1985) keeps a segment of the first parent in place and fills the rest in the second parent's
order; :func:`swap_mutation` exchanges two genes; :func:`inversion_mutation` reverses a segment.
"""

from dataclasses import dataclass

import numpy as np

from optimi_lab.core import refuse
from optimi_lab.problem import Categorical, Integer, Space

__all__ = [
    'Block',
    'Lattice',
    'discrete_mutation',
    'inversion_mutation',
    'lattice',
    'order_crossover',
    'snap',
    'swap_mutation',
]


@dataclass(frozen=True, slots=True)
class Block:
    """One discrete variable: its unit-cube columns, its level coordinates ``(L, width)``, and whether ordered."""

    start: int
    levels: np.ndarray
    ordered: bool

    @property
    def columns(self) -> slice:
        return slice(self.start, self.start + self.levels.shape[1])


@dataclass(frozen=True, slots=True)
class Lattice:
    """The discrete blocks of a space, in column order."""

    blocks: tuple[Block, ...]


def lattice(space: Space) -> Lattice:
    """The discrete blocks of *space*; a Categorical must carry an embedding to have coordinates."""
    blocks, start = [], 0
    for variable in space:
        if isinstance(variable, Integer):
            coordinates = np.array([variable.encode(level) for level in variable.levels])
            blocks.append(Block(start, coordinates, ordered=True))
        elif isinstance(variable, Categorical):
            coordinates = np.array([variable.encode(level) for level in variable.levels])
            blocks.append(Block(start, coordinates, ordered=False))
        start += variable.width
    return Lattice(tuple(blocks))


def _nearest(points: np.ndarray, block: Block) -> np.ndarray:
    return np.argmin(((points[:, None, block.columns] - block.levels[None]) ** 2).sum(axis=-1), axis=1)


def snap(points: np.ndarray, grid: Lattice) -> np.ndarray:
    """A copy of *points* with every discrete block on its nearest level; free columns untouched."""
    out = np.array(points, dtype=float)
    for block in grid.blocks:
        out[:, block.columns] = block.levels[_nearest(out, block)]
    return out


def discrete_mutation(
    points: np.ndarray, grid: Lattice, *, mutation_rate: float, rng: np.random.Generator
) -> np.ndarray:
    """Per discrete block with probability *mutation_rate*: one step on an ordered ladder, a reset otherwise."""
    if not 0.0 <= mutation_rate <= 1.0:
        refuse(f'mutation_rate must be in [0, 1], got {mutation_rate!r}')
    out = snap(points, grid)
    for block in grid.blocks:
        count = len(block.levels)
        current = _nearest(out, block)
        hit = rng.random(len(out)) < mutation_rate
        if block.ordered:
            step = np.where(rng.random(len(out)) < 0.5, -1, 1)
            step = np.where((current + step < 0) | (current + step >= count), -step, step)
            moved = current + step
        else:
            moved = (current + rng.integers(1, count, len(out))) % count
        out[hit, block.columns] = block.levels[moved[hit]]
    return out


def _require_permutations(rows: np.ndarray, what: str) -> np.ndarray:
    rows = np.asarray(rows)
    if rows.ndim != 2 or np.any(np.sort(rows, axis=1) != np.arange(rows.shape[1])):
        refuse(f'{what} must be rows each holding the permutation 0..k-1 once, got shape {rows.shape}')
    return rows


def order_crossover(first: np.ndarray, second: np.ndarray, *, rng: np.random.Generator) -> np.ndarray:
    """OX1: a random segment of *first* in place, the remaining genes in *second*'s order."""
    first = _require_permutations(first, 'first')
    second = _require_permutations(second, 'second')
    children = np.empty_like(first)
    k = first.shape[1]
    for row, (a, b) in enumerate(zip(first, second, strict=True)):
        lo, hi = sorted(rng.choice(k + 1, 2, replace=False))
        kept = set(a[lo:hi].tolist())
        rest = [gene for gene in b if gene not in kept]
        children[row] = np.concatenate([rest[:lo], a[lo:hi], rest[lo:]])
    return children


def swap_mutation(rows: np.ndarray, *, mutation_rate: float, rng: np.random.Generator) -> np.ndarray:
    """With probability *mutation_rate* per row, exchange two distinct genes."""
    out = _require_permutations(rows, 'a permutation population').copy()
    for row in np.flatnonzero(rng.random(len(out)) < mutation_rate):
        i, j = rng.choice(out.shape[1], 2, replace=False)
        out[row, [i, j]] = out[row, [j, i]]
    return out


def inversion_mutation(rows: np.ndarray, *, mutation_rate: float, rng: np.random.Generator) -> np.ndarray:
    """With probability *mutation_rate* per row, reverse a segment of at least two genes."""
    out = _require_permutations(rows, 'a permutation population').copy()
    for row in np.flatnonzero(rng.random(len(out)) < mutation_rate):
        lo, hi = sorted(rng.choice(out.shape[1] + 1, 2, replace=False))
        if hi - lo < 2:
            lo, hi = 0, out.shape[1]
        out[row, lo:hi] = out[row, lo:hi][::-1]
    return out
