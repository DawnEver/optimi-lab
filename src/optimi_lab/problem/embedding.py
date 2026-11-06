"""The embedding a model or a variation operator reads a mixed, conditional space through.

:meth:`Space.encode` is honest: an inactive variable's dimensions are NaN (it has no value) and a
categorical without an ``embedding=`` has no coordinates at all. A Gaussian process or an SBX
crossover needs a finite row of fixed width for every design, so an :class:`Embedding` DECLARES the
two choices that turn one into the other:

* ``categorical`` -- how a categorical that carries no embedding of its own is placed:
  :attr:`Encoding.ONE_HOT` (one column per level, no order implied, B0-11) or
  :attr:`Encoding.ORDINAL` (one column, the levels in the order they were written -- a claim that
  the order means something, so it is the caller's to make). A declared ``embedding=`` is kept.
* ``inactive`` -- what an inactive variable's columns hold: :attr:`Inactive.NEUTRAL` puts the
  declared ``neutral`` unit coordinate there (default the middle of the cube, per variable by
  name); :attr:`Inactive.INDICATOR` does the same and appends one column per conditional variable,
  1 when it is active and 0 when not -- the hierarchical embedding, which lets a model tell a
  switched-off variable from one that sits at the neutral value.

DECODING IS ALWAYS LEGAL. :meth:`Embedding.decode` reads the space's own columns through
:meth:`Space.decode`, which decides activity from the decoded parents and snaps every categorical
onto a level; neutral values and indicator columns are never read back. So the oracle never sees an
inactive variable set or a category the space does not declare, whatever a variation did to a row.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from enum import StrEnum

import numpy as np

from optimi_lab.core.errors import coerce_enum, refuse
from optimi_lab.problem.space import Space
from optimi_lab.problem.variables import Categorical

__all__ = ['Embedding', 'Encoding', 'Inactive', 'embed_categoricals', 'needs_embedding']


class Encoding(StrEnum):
    """How a categorical without a declared embedding gets coordinates."""

    ONE_HOT = 'one_hot'
    ORDINAL = 'ordinal'


class Inactive(StrEnum):
    """What the columns of an inactive variable hold."""

    NEUTRAL = 'neutral'
    INDICATOR = 'indicator'


def _rows(levels: int, encoding: Encoding) -> tuple[tuple[float, ...], ...]:
    if encoding is Encoding.ORDINAL:
        return tuple((float(i),) for i in range(levels))
    return tuple(tuple(float(i == j) for j in range(levels)) for i in range(levels))


def embed_categoricals(space: Space, encoding: Encoding = Encoding.ONE_HOT) -> Space:
    """*space* with an *encoding* embedding on every categorical that declares none."""
    encoding = coerce_enum(Encoding, encoding, 'the categorical encoding')
    return Space(
        tuple(
            replace(v, embedding=_rows(len(v.levels), encoding))
            if isinstance(v, Categorical) and v.embedding is None
            else v
            for v in space
        )
    )


def needs_embedding(space: Space) -> list[str]:
    """The variables a metric unit cube cannot hold as declared: conditional ones and unembedded categoricals."""
    return [v.name for v in space if v.active_if is not None or (isinstance(v, Categorical) and v.embedding is None)]


class Embedding:
    """*space* as finite rows of :attr:`width`: categoricals placed by *categorical*, inactive columns by *inactive*."""

    def __init__(
        self,
        space: Space,
        *,
        categorical: Encoding = Encoding.ONE_HOT,
        inactive: Inactive = Inactive.NEUTRAL,
        neutral: float | Mapping[str, float] = 0.5,
    ) -> None:
        self.categorical = coerce_enum(Encoding, categorical, 'the categorical encoding')
        self.inactive = coerce_enum(Inactive, inactive, 'the inactive embedding')
        self.source = space
        self.space = embed_categoricals(space, self.categorical)
        self.conditional = tuple(v.name for v in space if v.active_if is not None)
        given = dict(neutral) if isinstance(neutral, Mapping) else dict.fromkeys(self.conditional, neutral)
        strangers = sorted(set(given) - set(self.conditional))
        if strangers:
            refuse(f'neutral values name {strangers}, which are not conditional; the conditional ones are '
                   f'{list(self.conditional)}')  # fmt: skip
        values = {name: float(given.get(name, 0.5)) for name in self.conditional}
        bad = {name: value for name, value in values.items() if not 0.0 <= value <= 1.0}
        if bad:
            refuse(f'a neutral value is a unit coordinate in [0, 1], got {bad}')
        self.neutral = values
        self._fill = np.concatenate([np.full(v.width, values.get(v.name, 0.0)) for v in self.space] or [np.zeros(0)])
        self._indicators = len(self.conditional) if self.inactive is Inactive.INDICATOR else 0

    @classmethod
    def declared(cls, space: Space, embedding: Embedding | None) -> Embedding:
        """*embedding* checked against *space*; with none, the identity, refused when *space* needs one."""
        if embedding is None:
            needed = needs_embedding(space)
            if needed:
                refuse(
                    f'{needed} are conditional (active_if) or unembedded categoricals; declare how a metric '
                    f'unit cube holds them: embedding=optimi_lab.problem.Embedding(space, categorical=..., '
                    f'inactive=...)'
                )
            return cls(space)
        if not isinstance(embedding, Embedding):
            refuse(f'embedding= is an optimi_lab.problem.Embedding, got {type(embedding).__name__}')
        if embedding.source != space:
            refuse("the embedding was declared over another space; build it from the problem's own space")
        return embedding

    @property
    def width(self) -> int:
        """Columns of an embedded row: the embedded space's, plus one indicator per conditional variable."""
        return self.space.width + self._indicators

    def encode(self, design: Mapping[str, object]) -> np.ndarray:
        """The finite row of *design*: inactive columns at their neutral value, then the indicators."""
        unit = self.space.encode(design)
        inactive = np.isnan(unit)
        unit[inactive] = self._fill[inactive]
        if not self._indicators:
            return unit
        active = set(self.space.active(design))
        return np.concatenate([unit, [float(name in active) for name in self.conditional]])

    def decode(self, row: np.ndarray) -> dict[str, object]:
        """The legal design at *row*: activity from the decoded parents; neutral and indicator columns unread."""
        row = np.asarray(row, dtype=float)
        if row.shape != (self.width,):
            refuse(f'an embedded row of this space has shape ({self.width},), got {row.shape}')
        return self.space.decode(row[: self.space.width])
