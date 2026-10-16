"""Where the first batch of points comes from — one declared kind list, one count.

A sweep, a design of experiments and an optimizer's first population are ONE thing: a point rule
over the declared variables. This module is that rule's one owner -- the full factorial
(``grid``), the space-filling draws, an imported matrix, and the nominal ``anchor`` a study
prepends -- so a caller never writes its own ``linspace`` or cartesian product.

Two measured defects are foreclosed here:

* ``VariableSpace(sample_type='poisson_disk')`` returned ONE row at every ``n_count``: its radius
  was derived as ``(1 / n_count) ** (-d)``, the exponent inverted, and scipy answers a radius
  that large with a single point. ``sample_type='import'`` was worse — it left
  ``var_space_matrix = None``, and the consumer that subscripted it raised
  ``TypeError: 'NoneType' object is not subscriptable``.
* The initial sample size and the population size were TWO numbers. The first iteration
  evaluated the whole grid (measured ``[121, 8, 8, 8]``) while ``_pop_size`` was documented
  "per iteration". Here :func:`sample` returns exactly ``n_samples`` rows for every kind, and the
  driver derives the population size FROM that matrix — see
  :func:`optimi_lab.core.optimize.optimize`.

Every kind therefore either returns ``n_samples`` rows or refuses naming what it got.
"""

from dataclasses import dataclass
from enum import StrEnum
from itertools import product
from math import prod

import numpy as np
from scipy.stats.qmc import LatinHypercube, PoissonDisk

from optimi_lab.core.errors import coerce_enum, refuse, require_finite, require_positive
from optimi_lab.core.space import VariableSet

__all__ = [
    'DEFAULT_SAMPLES',
    'MAX_GRID_POINTS',
    'SAMPLE_KINDS',
    'SampleKind',
    'SampleSpec',
    'grid_counts',
    'sample',
]


class SampleKind(StrEnum):
    """How to draw the first batch of points."""

    GRID = 'grid'
    LATIN_HYPERCUBE = 'latin_hypercube'
    POISSON_DISK = 'poisson_disk'
    IMPORT = 'import'


SAMPLE_KINDS = tuple(kind.value for kind in SampleKind)
"""The declared kinds, in one place — a refusal names this tuple rather than a constant in
another module."""

DEFAULT_SAMPLES = 32
"""The batch size of a kind that draws its own points when ``n_samples`` is not given."""

MAX_GRID_POINTS = 100_000
"""Ceiling on a full-factorial grid. A product of level counts grows multiplicatively, so a grid
over it is REFUSED before a point is built rather than truncated: a truncated factorial is no
longer a factorial, and nothing downstream could tell."""

_MAX_RADIUS_HALVINGS = 32
"""Ceiling on the Poisson-disk radius search, so a radius that cannot be met terminates."""


@dataclass(frozen=True, slots=True)
class _SampleSpecFields:
    kind: SampleKind = SampleKind.LATIN_HYPERCUBE
    n_samples: int | None = None
    seed: int = 0
    levels: tuple[int, ...] | None = None
    radius: float | None = None
    matrix: np.ndarray | None = None
    anchor: tuple[float, ...] | None = None


@dataclass(frozen=True, slots=True, init=False)
class SampleSpec(_SampleSpecFields):
    """A request for the first batch.

    ``kind`` selects the sampler; ``n_samples`` is how many rows come back, the anchor included,
    and every kind returns exactly that many. ``None`` is the kind's own count: the product of
    ``levels`` (plus the anchor) for ``grid``, the matrix's rows for ``import``, and
    :data:`DEFAULT_SAMPLES` otherwise. ``seed`` seeds the stochastic kinds. Three fields each
    belong to ONE kind and are refused on any other: ``levels`` (the per-variable level counts of
    the full factorial, derived from ``n_samples`` when ``None``) to ``grid``, ``radius`` (derived
    from ``n_samples`` when ``None``) to ``poisson_disk``, and ``matrix`` to ``import``.

    ``anchor`` is a nominal point returned as ROW 0, ahead of the drawn points; it is checked
    against the space's bounds when the batch is drawn. An imported matrix is already the whole
    batch, so it takes no anchor.
    """

    def __init__(  # noqa: PLR0917 -- the public dataclass's positional parameters.
        self,
        kind: SampleKind = SampleKind.LATIN_HYPERCUBE,
        n_samples: int | None = None,
        seed: int = 0,
        levels: tuple[int, ...] | None = None,
        radius: float | None = None,
        matrix: np.ndarray | None = None,
        anchor: tuple[float, ...] | None = None,
    ) -> None:
        kind = coerce_enum(SampleKind, kind, 'sample kind')
        if n_samples is not None:
            require_positive(n_samples, 'n_samples')
        _require_kind_field('levels', levels, SampleKind.GRID, kind)
        _require_kind_field('radius', radius, SampleKind.POISSON_DISK, kind)
        _require_kind_field('matrix', matrix, SampleKind.IMPORT, kind)
        if levels is not None:
            levels = tuple(int(level) for level in levels)
            if min(levels, default=0) < 2:
                refuse(f'every grid level count must be at least 2 -- one level is not an axis -- got {list(levels)}')
        if radius is not None:
            require_positive(radius, 'radius')
        if kind is SampleKind.IMPORT and matrix is None:
            refuse(
                f'sample kind {SampleKind.IMPORT.value!r} needs the matrix to import, of shape (n_samples, n_var); '
                f'the kinds that draw their own points are {[k for k in SAMPLE_KINDS if k != SampleKind.IMPORT.value]}'
            )
        if anchor is not None:
            anchor = _require_anchor_spec(kind, n_samples, anchor)
        _SampleSpecFields.__init__(self, kind, n_samples, seed, levels, radius, matrix, anchor)


def _require_kind_field(field_name: str, value: object, field_kind: SampleKind, kind: SampleKind) -> None:
    """Refuse a field that is set but belongs to another kind, naming the kind that owns it."""
    if value is not None and kind is not field_kind:
        refuse(
            f'{field_name} belongs to sample kind {field_kind.value!r}, not {kind.value!r}; '
            f'the declared kinds are {list(SAMPLE_KINDS)}'
        )


def _require_anchor_spec(kind: SampleKind, n_samples: int | None, anchor: tuple[float, ...]) -> tuple[float, ...]:
    """Return the anchor as finite floats, refusing it on ``import`` or in a batch with no room."""
    if kind is SampleKind.IMPORT:
        refuse(
            f'an imported matrix is the whole batch, so sample kind {SampleKind.IMPORT.value!r} takes no anchor; '
            f'put the nominal point in the matrix as its first row'
        )
    if n_samples is not None and n_samples < 2:
        refuse(f'an anchor is row 0 of the batch, so n_samples must be at least 2 to draw a point, got {n_samples}')
    anchor = tuple(float(coordinate) for coordinate in anchor)
    require_finite(np.array(anchor), 'the anchor')
    return anchor


def grid_counts(n_var: int, n_samples: int) -> tuple[int, ...]:
    """Split ``n_samples`` into ``n_var`` per-variable grid counts, largest first.

    A full grid over ``d`` variables holds ``prod(counts)`` points, so a target that is not a
    ``d``-fold product of integers at least 2 cannot be a grid — such an ``n_samples`` is refused
    with a reachable size rather than the sampler silently returning a different number of rows.
    Each prime factor is assigned to the currently smallest count, which keeps the grid as close
    to cubic as an integer factorization allows.
    """
    require_positive(n_var, 'n_var')
    require_positive(n_samples, 'n_samples')
    factors = sorted(_prime_factors(n_samples), reverse=True)
    if len(factors) < n_var:
        refuse(
            f'a grid over {n_var} variable(s) needs {n_var} counts of at least 2, and {n_samples} factorizes into '
            f'{list(reversed(factors))}, only {len(factors)} factor(s); the smallest n_samples a {n_var}-variable '
            f'grid can fill is {2**n_var}'
        )
    counts = [1] * n_var
    for factor in factors:
        smallest = counts.index(min(counts))
        counts[smallest] *= factor
    return tuple(sorted(counts, reverse=True))


def _prime_factors(number: int) -> list[int]:
    """Return the prime factors of ``number`` with multiplicity, by trial division."""
    factors = []
    divisor = 2
    while divisor * divisor <= number:
        while number % divisor == 0:
            factors.append(divisor)
            number //= divisor
        divisor += 1
    if number > 1:
        factors.append(number)
    return factors


def sample(space: VariableSet, spec: SampleSpec | None = None) -> np.ndarray:
    """Draw the first batch: exactly ``n_samples`` points, inside ``space``'s bounds.

    Returns a fresh array of shape ``(n_samples, space.n_var)`` -- the anchor, when given, as row
    0 -- never a view of anything the caller or this module holds, and refuses if the requested
    kind cannot fill exactly that many rows.
    """
    spec = SampleSpec() if spec is None else spec
    samplers = {
        SampleKind.GRID: _grid,
        SampleKind.LATIN_HYPERCUBE: _latin_hypercube,
        SampleKind.POISSON_DISK: _poisson_disk,
        SampleKind.IMPORT: _import,
    }
    anchor = _require_anchor(space, spec.anchor)
    n_anchor = 0 if anchor is None else 1
    n_drawn = _drawn_count(space, spec) if spec.n_samples is None else spec.n_samples - n_anchor
    points = samplers[spec.kind](space, spec, n_drawn)
    if points.shape != (n_drawn, space.n_var):
        refuse(
            f'sampler {spec.kind.value!r} returned shape {points.shape} for {n_drawn} drawn point(s) and '
            f'n_var={space.n_var}; every kind must fill the requested batch or refuse'
        )
    return points if anchor is None else np.vstack([anchor, points])


def _drawn_count(space: VariableSet, spec: SampleSpec) -> int:
    """The kind's own count when ``n_samples`` is not given: levels, matrix rows, or the default."""
    if spec.kind is SampleKind.GRID and spec.levels is not None:
        return _grid_size(space, spec.levels)
    if spec.kind is SampleKind.IMPORT:
        return int(np.asarray(spec.matrix).shape[0])
    return DEFAULT_SAMPLES - (0 if spec.anchor is None else 1)


def _require_anchor(space: VariableSet, anchor: tuple[float, ...] | None) -> np.ndarray | None:
    """Return the anchor as a ``(1, n_var)`` row, or refuse one outside the declared bounds."""
    if anchor is None:
        return None
    if len(anchor) != space.n_var:
        refuse(f'the anchor must name one coordinate per variable {list(space.names)}, got {list(anchor)}')
    for variable, coordinate in zip(space, anchor, strict=True):
        if not variable.lower <= coordinate <= variable.upper:
            refuse(
                f'the anchor puts variable {variable.name!r} at {coordinate!r}, outside its bounds '
                f'[{variable.lower!r}, {variable.upper!r}]'
            )
    return np.array([anchor], dtype=float)


def _grid_size(space: VariableSet, levels: tuple[int, ...]) -> int:
    """The point count of a full factorial, refused over :data:`MAX_GRID_POINTS` before it is built."""
    if len(levels) != space.n_var:
        refuse(f'levels must name one count per variable, expected {space.n_var}, got {len(levels)}: {list(levels)}')
    size = prod(levels)
    if size > MAX_GRID_POINTS:
        refuse(
            f'levels {list(levels)} describe a grid of {size} points, over the ceiling of {MAX_GRID_POINTS}; '
            f'a truncated factorial is not a factorial, so take fewer levels or draw a latin_hypercube instead'
        )
    return size


def _grid(space: VariableSet, spec: SampleSpec, n_drawn: int) -> np.ndarray:
    """The full factorial: every combination of evenly spaced levels, the last variable fastest."""
    levels = spec.levels if spec.levels is not None else grid_counts(space.n_var, n_drawn)
    size = _grid_size(space, levels)
    if size != n_drawn:
        refuse(
            f'levels {list(levels)} describe a grid of {size} point(s), but {n_drawn} were requested '
            f'(n_samples={spec.n_samples}, the anchor counted in it); a grid returns one row per grid point, so '
            f'leave n_samples unset or pick levels whose counts multiply to {n_drawn}'
        )
    columns = [
        np.linspace(variable.lower, variable.upper, count) for variable, count in zip(space, levels, strict=True)
    ]
    return np.array(list(product(*columns)), dtype=float)


def _latin_hypercube(space: VariableSet, spec: SampleSpec, n_drawn: int) -> np.ndarray:
    """A Latin hypercube: one point per stratum in every dimension."""
    unit_points = LatinHypercube(d=space.n_var, seed=spec.seed).random(n=n_drawn)
    return space.scale(unit_points)


def _poisson_disk(space: VariableSet, spec: SampleSpec, n_drawn: int) -> np.ndarray:
    """A Poisson-disk (blue-noise) set, subsampled to exactly ``n_drawn`` points.

    scipy's ``random(n)`` returns FEWER than ``n`` points when the radius leaves no room, so
    the batch is filled by ``fill_space()`` — every point the radius admits — and then thinned
    to ``n_drawn``. The radius search is bounded, and a radius that stays too large is
    refused naming the count it could reach rather than returning a short batch.
    """
    radius = spec.radius if spec.radius is not None else 0.5 * n_drawn ** (-1.0 / space.n_var)
    admitted = np.empty((0, space.n_var))
    for _ in range(_MAX_RADIUS_HALVINGS):
        admitted = PoissonDisk(d=space.n_var, radius=radius, seed=spec.seed).fill_space()
        if admitted.shape[0] >= n_drawn:
            break
        radius /= 2
    else:
        refuse(
            f'poisson_disk admitted {admitted.shape[0]} point(s) at the smallest radius tried in '
            f'{_MAX_RADIUS_HALVINGS} halvings, short of {n_drawn}; pass radius= a smaller positive '
            f'value, or a smaller n_samples'
        )
    if admitted.shape[0] > n_drawn:
        chosen = np.random.default_rng(spec.seed).choice(admitted.shape[0], size=n_drawn, replace=False)
        admitted = admitted[np.sort(chosen)]
    return space.scale(admitted)


def _import(space: VariableSet, spec: SampleSpec, n_drawn: int) -> np.ndarray:
    """A matrix the caller already has, checked against the space it claims to sample."""
    matrix = np.asarray(spec.matrix, dtype=float)
    if matrix.shape != (n_drawn, space.n_var):
        refuse(
            f'the imported matrix must have shape ({n_drawn}, {space.n_var}) — one row per sample and one '
            f'column per variable — got {matrix.shape}'
        )
    if not np.all(np.isfinite(matrix)):
        refuse('every entry of the imported matrix must be finite; an unevaluated point has no coordinates')
    return matrix.copy()
