"""Where the first batch of points comes from — one declared kind list, one count.

A sweep, a design of experiments and an optimizer's first population are ONE thing: a point rule
over the unit cube a :class:`~optimi_lab.problem.Space` encodes into. This module is that rule's one
owner -- the full factorial (``grid``), the space-filling draws, and the nominal ``anchor`` a study
prepends -- so a caller never writes its own ``linspace`` or cartesian product. A caller that already
has its points asks for them directly; there is no import kind, because a matrix handed in is not a
sample drawn.

Two measured defects are foreclosed here:

* ``VariableSpace(sample_type='poisson_disk')`` returned ONE row at every ``n_count``: its radius
  was derived as ``(1 / n_count) ** (-d)``, the exponent inverted, and a radius that large admits
  a single point. ``sample_type='import'`` was worse — it left nothing at all.
* The initial sample size and the population size were TWO numbers. The first iteration
  evaluated the whole grid (measured ``[121, 8, 8, 8]``) while ``_pop_size`` was documented
  "per iteration". Here :func:`sample` returns exactly ``n_samples`` rows for every kind, and the
  population solver derives its size FROM that matrix (``optimi_lab.solver.population``).

Every kind therefore either returns ``n_samples`` rows or refuses naming what it got. Every kind is
self-implemented (R1): the point sets are :mod:`optimi_lab.core.qmc`'s, the Poisson disk is below.
"""

from dataclasses import dataclass
from enum import StrEnum
from itertools import product
from math import prod

import numpy as np

from optimi_lab.core.errors import coerce_enum, refuse, require_finite, require_positive
from optimi_lab.core.qmc import halton, latin_hypercube, maximin_latin_hypercube, sobol

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
    MAXIMIN_LATIN_HYPERCUBE = 'maximin_latin_hypercube'
    POISSON_DISK = 'poisson_disk'
    SOBOL = 'sobol'
    HALTON = 'halton'


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

_DART_CANDIDATES = 30
"""Bridson's k: candidates tried around an active point before it is retired."""


@dataclass(frozen=True, slots=True)
class _SampleSpecFields:
    kind: SampleKind = SampleKind.LATIN_HYPERCUBE
    n_samples: int | None = None
    seed: int = 0
    levels: tuple[int, ...] | None = None
    radius: float | None = None
    anchor: tuple[float, ...] | None = None


@dataclass(frozen=True, slots=True, init=False)
class SampleSpec(_SampleSpecFields):
    """A request for the first batch.

    ``kind`` selects the sampler; ``n_samples`` is how many rows come back, the anchor included,
    and every kind returns exactly that many. ``None`` is the kind's own count: the product of
    ``levels`` (plus the anchor) for ``grid``, and
    :data:`DEFAULT_SAMPLES` otherwise. ``seed`` seeds the stochastic kinds. Two fields each
    belong to ONE kind and are refused on any other: ``levels`` (the per-variable level counts of
    the full factorial, derived from ``n_samples`` when ``None``) to ``grid``, ``radius`` (derived
    from ``n_samples`` when ``None``) to ``poisson_disk``.

    ``anchor`` is a nominal point returned as ROW 0, ahead of the drawn points; it is checked
    against the unit cube when the batch is drawn.
    """

    def __init__(  # noqa: PLR0917 -- the public dataclass's positional parameters.
        self,
        kind: SampleKind = SampleKind.LATIN_HYPERCUBE,
        n_samples: int | None = None,
        seed: int = 0,
        levels: tuple[int, ...] | None = None,
        radius: float | None = None,
        anchor: tuple[float, ...] | None = None,
    ) -> None:
        kind = coerce_enum(SampleKind, kind, 'sample kind')
        if n_samples is not None:
            require_positive(n_samples, 'n_samples')
        _require_kind_field('levels', levels, SampleKind.GRID, kind)
        _require_kind_field('radius', radius, SampleKind.POISSON_DISK, kind)
        if levels is not None:
            levels = tuple(int(level) for level in levels)
            if min(levels, default=0) < 2:
                refuse(f'every grid level count must be at least 2 -- one level is not an axis -- got {list(levels)}')
        if radius is not None:
            require_positive(radius, 'radius')
        if anchor is not None:
            anchor = _require_anchor_spec(n_samples, anchor)
        _SampleSpecFields.__init__(self, kind, n_samples, seed, levels, radius, anchor)


def _require_kind_field(field_name: str, value: object, field_kind: SampleKind, kind: SampleKind) -> None:
    """Refuse a field that is set but belongs to another kind, naming the kind that owns it."""
    if value is not None and kind is not field_kind:
        refuse(
            f'{field_name} belongs to sample kind {field_kind.value!r}, not {kind.value!r}; '
            f'the declared kinds are {list(SAMPLE_KINDS)}'
        )


def _require_anchor_spec(n_samples: int | None, anchor: tuple[float, ...]) -> tuple[float, ...]:
    """Return the anchor as finite floats, refusing it in a batch with no room for a drawn point."""
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


def sample(width: int, spec: SampleSpec | None = None) -> np.ndarray:
    """Draw the first batch: exactly ``n_samples`` points, inside the unit cube of ``width`` dimensions.

    Returns a fresh array of shape ``(n_samples, width)`` -- the anchor, when given, as row
    0 -- never a view of anything the caller or this module holds, and refuses if the requested
    kind cannot fill exactly that many rows.
    """
    spec = SampleSpec() if spec is None else spec
    samplers = {
        SampleKind.GRID: _grid,
        SampleKind.LATIN_HYPERCUBE: lambda w, s, n: latin_hypercube(n, w, np.random.default_rng(s.seed)),
        SampleKind.MAXIMIN_LATIN_HYPERCUBE: lambda w, s, n: maximin_latin_hypercube(
            n, w, np.random.default_rng(s.seed)
        ),
        SampleKind.POISSON_DISK: _poisson_disk,
        SampleKind.SOBOL: lambda w, s, n: sobol(n, w, rng=np.random.default_rng(s.seed)),
        SampleKind.HALTON: lambda w, s, n: halton(n, w, rng=np.random.default_rng(s.seed)),
    }
    anchor = _require_anchor(width, spec.anchor)
    n_anchor = 0 if anchor is None else 1
    n_drawn = _drawn_count(width, spec) if spec.n_samples is None else spec.n_samples - n_anchor
    # The empty product: every kind draws the same zero-width rows, and a width-0 sampler would refuse.
    points = np.zeros((n_drawn, 0)) if width == 0 else samplers[spec.kind](width, spec, n_drawn)
    if points.shape != (n_drawn, width):
        refuse(
            f'sampler {spec.kind.value!r} returned shape {points.shape} for {n_drawn} drawn point(s) and '
            f'n_var={width}; every kind must fill the requested batch or refuse'
        )
    return points if anchor is None else np.vstack([anchor, points])


def _drawn_count(width: int, spec: SampleSpec) -> int:
    """The kind's own count when ``n_samples`` is not given: levels, or the default."""
    if spec.kind is SampleKind.GRID and spec.levels is not None:
        return _grid_size(width, spec.levels)
    return DEFAULT_SAMPLES - (0 if spec.anchor is None else 1)


def _require_anchor(width: int, anchor: tuple[float, ...] | None) -> np.ndarray | None:
    """Return the anchor as a ``(1, n_var)`` row, or refuse one outside the declared bounds."""
    if anchor is None:
        return None
    if len(anchor) != width:
        refuse(f'the anchor must name one coordinate per unit-cube dimension ({width}), got {list(anchor)}')
    outside = [index for index, coordinate in enumerate(anchor) if not 0.0 <= coordinate <= 1.0]
    if outside:
        refuse(f'the anchor is a unit-cube point; coordinates {outside} of {list(anchor)} lie outside [0, 1]')
    return np.array([anchor], dtype=float)


def _grid_size(width: int, levels: tuple[int, ...]) -> int:
    """The point count of a full factorial, refused over :data:`MAX_GRID_POINTS` before it is built."""
    if len(levels) != width:
        refuse(f'levels must name one count per variable, expected {width}, got {len(levels)}: {list(levels)}')
    size = prod(levels)
    if size > MAX_GRID_POINTS:
        refuse(
            f'levels {list(levels)} describe a grid of {size} points, over the ceiling of {MAX_GRID_POINTS}; '
            f'a truncated factorial is not a factorial, so take fewer levels or draw a latin_hypercube instead'
        )
    return size


def _grid(width: int, spec: SampleSpec, n_drawn: int) -> np.ndarray:
    """The full factorial: every combination of evenly spaced levels, the last variable fastest."""
    levels = spec.levels if spec.levels is not None else grid_counts(width, n_drawn)
    size = _grid_size(width, levels)
    if size != n_drawn:
        refuse(
            f'levels {list(levels)} describe a grid of {size} point(s), but {n_drawn} were requested '
            f'(n_samples={spec.n_samples}, the anchor counted in it); a grid returns one row per grid point, so '
            f'leave n_samples unset or pick levels whose counts multiply to {n_drawn}'
        )
    columns = [np.linspace(0.0, 1.0, count) for count in levels]
    return np.array(list(product(*columns)), dtype=float)


def _bridson(width: int, radius: float, rng: np.random.Generator) -> np.ndarray:
    """Every point a radius admits (Bridson 2007): darts in the shell ``[r, 2r)`` around active points.

    Distances are checked against every accepted point at once, so the cost is quadratic in the
    count and independent of the width -- a background grid would cost ``5 ** width`` cells a probe.
    """
    accepted = [rng.random(width)]
    active = [0]
    while active:
        slot = rng.integers(len(active))
        centre = accepted[active[slot]]
        direction = rng.normal(size=(_DART_CANDIDATES, width))
        direction /= np.linalg.norm(direction, axis=1, keepdims=True)
        candidates = centre + direction * radius * (1.0 + rng.random((_DART_CANDIDATES, 1)))
        placed = False
        for candidate in candidates[np.all((candidates >= 0.0) & (candidates < 1.0), axis=1)]:
            if np.min(np.linalg.norm(np.asarray(accepted) - candidate, axis=1)) >= radius:
                accepted.append(candidate)
                active.append(len(accepted) - 1)
                placed = True
                break
        if not placed:
            active.pop(slot)
    return np.asarray(accepted)


def _poisson_disk(width: int, spec: SampleSpec, n_drawn: int) -> np.ndarray:
    """A Poisson-disk (blue-noise) set: every pair at least ``radius`` apart, thinned to ``n_drawn``.

    A radius the caller GAVE is a guarantee and is never shrunk: a set it cannot fill is refused,
    naming the count it reached. A radius derived from ``n_drawn`` is halved until the set fills,
    within a bounded search.
    """
    rng = np.random.default_rng(spec.seed)
    radius = spec.radius if spec.radius is not None else 0.5 * n_drawn ** (-1.0 / width)
    halvings = 1 if spec.radius is not None else _MAX_RADIUS_HALVINGS
    admitted = np.empty((0, width))
    for _ in range(halvings):
        admitted = _bridson(width, radius, rng)
        if admitted.shape[0] >= n_drawn:
            break
        radius /= 2
    else:
        refuse(
            f'poisson_disk admitted {admitted.shape[0]} point(s) at radius {radius * 2:g}, short of {n_drawn}; '
            f'pass a smaller radius= or a smaller n_samples'
        )
    chosen = rng.choice(admitted.shape[0], size=n_drawn, replace=False)
    return admitted[np.sort(chosen)]
