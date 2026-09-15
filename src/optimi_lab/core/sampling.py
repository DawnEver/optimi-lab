"""Where the first batch of points comes from — one declared kind list, one count.

Two measured defects are foreclosed here:

* ``VariableSpace(sample_type='poisson_disk')`` returned ONE row at every ``n_count``: its radius
  was derived as ``(1 / n_count) ** (-d)``, the exponent inverted, and scipy answers a radius
  that large with a single point. ``sample_type='import'`` was worse — it left
  ``var_space_matrix = None``, and the consumer that subscripted it raised
  ``TypeError: 'NoneType' object is not subscriptable``.
* The initial sample size and the population size were TWO numbers. The first iteration
  evaluated the whole grid (measured ``[121, 8, 8, 8]``) while ``_pop_size`` was documented
  "per iteration". Here :func:`sample` returns exactly ``spec.n_samples`` rows for every kind,
  and the driver derives the population size FROM that matrix — see
  :func:`optimi_lab.core.optimize.optimize`.

Every kind therefore either returns ``n_samples`` rows or refuses naming what it got.
"""

from dataclasses import dataclass
from enum import StrEnum
from itertools import product

import numpy as np
from scipy.stats.qmc import LatinHypercube, PoissonDisk

from optimi_lab.core.errors import coerce_enum, refuse, require_positive
from optimi_lab.core.space import VariableSet

__all__ = ['SAMPLE_KINDS', 'SampleKind', 'SampleSpec', 'grid_counts', 'sample']


class SampleKind(StrEnum):
    """How to draw the first batch of points."""

    UNIFORM = 'uniform'
    LATIN_HYPERCUBE = 'latin_hypercube'
    POISSON_DISK = 'poisson_disk'
    IMPORT = 'import'


SAMPLE_KINDS = tuple(kind.value for kind in SampleKind)
"""The declared kinds, in one place — a refusal names this tuple rather than a constant in
another module."""

_MAX_RADIUS_HALVINGS = 32
"""Ceiling on the Poisson-disk radius search, so a radius that cannot be met terminates."""


@dataclass(frozen=True, slots=True)
class SampleSpec:
    """A request for the first batch.

    ``kind`` selects the sampler; ``n_samples`` is how many rows come back, and every kind
    returns exactly that many. ``seed`` seeds the stochastic kinds. The three remaining fields
    each belong to ONE kind and are refused on any other: ``steps`` (the per-variable grid
    counts, derived from ``n_samples`` when ``None``) to ``uniform``, ``radius`` (derived from
    ``n_samples`` when ``None``) to ``poisson_disk``, and ``matrix`` to ``import``.
    """

    kind: SampleKind = SampleKind.LATIN_HYPERCUBE
    n_samples: int = 32
    seed: int = 0
    steps: tuple[int, ...] | None = None
    radius: float | None = None
    matrix: np.ndarray | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, SampleKind):
            object.__setattr__(self, 'kind', coerce_enum(SampleKind, self.kind, 'sample kind'))
        require_positive(self.n_samples, 'n_samples')
        _require_kind_field('steps', self.steps, SampleKind.UNIFORM, self.kind)
        _require_kind_field('radius', self.radius, SampleKind.POISSON_DISK, self.kind)
        _require_kind_field('matrix', self.matrix, SampleKind.IMPORT, self.kind)
        if self.steps is not None:
            object.__setattr__(self, 'steps', tuple(int(step) for step in self.steps))
        if self.radius is not None:
            require_positive(self.radius, 'radius')
        if self.kind is SampleKind.IMPORT and self.matrix is None:
            refuse(
                f'sample kind {SampleKind.IMPORT.value!r} needs the matrix to import, of shape (n_samples, n_var); '
                f'the kinds that draw their own points are {[k for k in SAMPLE_KINDS if k != SampleKind.IMPORT.value]}'
            )


def _require_kind_field(field_name: str, value: object, field_kind: SampleKind, kind: SampleKind) -> None:
    """Refuse a field that is set but belongs to another kind, naming the kind that owns it."""
    if value is not None and kind is not field_kind:
        refuse(
            f'{field_name} belongs to sample kind {field_kind.value!r}, not {kind.value!r}; '
            f'the declared kinds are {list(SAMPLE_KINDS)}'
        )


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
    """Draw the first batch: exactly ``spec.n_samples`` points, inside ``space``'s bounds.

    Returns a fresh array of shape ``(spec.n_samples, space.n_var)``, never a view of anything
    the caller or this module holds, and refuses if the requested kind cannot fill exactly that
    many rows.
    """
    spec = SampleSpec() if spec is None else spec
    samplers = {
        SampleKind.UNIFORM: _uniform,
        SampleKind.LATIN_HYPERCUBE: _latin_hypercube,
        SampleKind.POISSON_DISK: _poisson_disk,
        SampleKind.IMPORT: _import,
    }
    points = samplers[spec.kind](space, spec)
    if points.shape != (spec.n_samples, space.n_var):
        refuse(
            f'sampler {spec.kind.value!r} returned shape {points.shape} for n_samples={spec.n_samples} and '
            f'n_var={space.n_var}; every kind must fill the requested batch or refuse'
        )
    return points


def _uniform(space: VariableSet, spec: SampleSpec) -> np.ndarray:
    """The full grid: one column of ``linspace`` values per variable."""
    counts = spec.steps if spec.steps is not None else grid_counts(space.n_var, spec.n_samples)
    if len(counts) != space.n_var:
        refuse(f'steps must name one count per variable, expected {space.n_var}, got {len(counts)}: {counts}')
    product_of_steps = int(np.prod(counts))
    if product_of_steps != spec.n_samples:
        refuse(
            f'steps {list(counts)} describe a grid of {product_of_steps} point(s), but n_samples is '
            f'{spec.n_samples}; a grid returns one row per grid point, so set n_samples={product_of_steps} '
            f'or pick steps whose counts multiply to {spec.n_samples}'
        )
    columns = [
        np.linspace(variable.lower, variable.upper, count) for variable, count in zip(space, counts, strict=True)
    ]
    return np.array(list(product(*columns)), dtype=float)


def _latin_hypercube(space: VariableSet, spec: SampleSpec) -> np.ndarray:
    """A Latin hypercube: one point per stratum in every dimension."""
    unit_points = LatinHypercube(d=space.n_var, seed=spec.seed).random(n=spec.n_samples)
    return space.scale(unit_points)


def _poisson_disk(space: VariableSet, spec: SampleSpec) -> np.ndarray:
    """A Poisson-disk (blue-noise) set, subsampled to exactly ``n_samples`` points.

    scipy's ``random(n)`` returns FEWER than ``n`` points when the radius leaves no room, so
    the batch is filled by ``fill_space()`` — every point the radius admits — and then thinned
    to ``n_samples``. The radius search is bounded, and a radius that stays too large is
    refused naming the count it could reach rather than returning a short batch.
    """
    radius = spec.radius if spec.radius is not None else 0.5 * spec.n_samples ** (-1.0 / space.n_var)
    admitted = np.empty((0, space.n_var))
    for _ in range(_MAX_RADIUS_HALVINGS):
        admitted = PoissonDisk(d=space.n_var, radius=radius, seed=spec.seed).fill_space()
        if admitted.shape[0] >= spec.n_samples:
            break
        radius /= 2
    else:
        refuse(
            f'poisson_disk admitted {admitted.shape[0]} point(s) at the smallest radius tried in '
            f'{_MAX_RADIUS_HALVINGS} halvings, short of n_samples={spec.n_samples}; pass radius= a smaller positive '
            f'value, or a smaller n_samples'
        )
    if admitted.shape[0] > spec.n_samples:
        chosen = np.random.default_rng(spec.seed).choice(admitted.shape[0], size=spec.n_samples, replace=False)
        admitted = admitted[np.sort(chosen)]
    return space.scale(admitted)


def _import(space: VariableSet, spec: SampleSpec) -> np.ndarray:
    """A matrix the caller already has, checked against the space it claims to sample."""
    matrix = np.asarray(spec.matrix, dtype=float)
    if matrix.shape != (spec.n_samples, space.n_var):
        refuse(
            f'the imported matrix must have shape ({spec.n_samples}, {space.n_var}) — one row per sample and one '
            f'column per variable — got {matrix.shape}'
        )
    if not np.all(np.isfinite(matrix)):
        refuse('every entry of the imported matrix must be finite; an unevaluated point has no coordinates')
    return matrix.copy()
