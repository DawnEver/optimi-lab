"""Space-filling point sets over the unit cube: Latin hypercubes and the Sobol and Halton sequences.

Self-implemented (R1); scipy.stats.qmc is a TEST oracle only (``tests/unit/core/test_qmc.py``).

* :func:`sobol` -- the base-2 digital sequence in Gray-code order (Antonov-Saleev), 32 bits, from Joe
  and Kuo's direction numbers (:mod:`optimi_lab.core._joe_kuo`, data). ``rng`` scrambles it by a
  random linear matrix and a digital shift (Matousek's LMS + shift), which keeps every elementary
  interval holding its share of points; ``None`` is the raw sequence.
* :func:`halton` -- radical inverses in the first ``width`` primes; ``rng`` permutes every digit by
  its own random permutation (Braaten-Weller style scrambling), ``None`` is the raw sequence.
* :func:`latin_hypercube` -- one point per stratum per column; :func:`maximin_latin_hypercube` then
  swaps entries within columns while the Morris-Mitchell criterion improves, which pushes the
  closest pair apart and keeps the Latin property, because a swap within a column permutes strata.

Every function returns a fresh ``(n_points, width)`` array in ``[0, 1)``.
"""

from math import ceil, log2

import numpy as np

from optimi_lab.core._joe_kuo import DIRECTIONS
from optimi_lab.core.errors import refuse, require_positive

__all__ = ['MAX_SOBOL_DIMENSION', 'halton', 'latin_hypercube', 'maximin_latin_hypercube', 'sobol']

_BITS = 32
"""Digits of a Sobol coordinate; the sequence has 2**32 points before it repeats."""

MAX_SOBOL_DIMENSION = len(DIRECTIONS) + 1
"""The widest Sobol set the embedded table supports: the van der Corput column plus one per row."""

_MAXIMIN_POWER = 50.0
"""Morris-Mitchell exponent: large enough that the criterion is ruled by the closest pairs."""

_MAXIMIN_SWAPS = 2000
"""Swap attempts of the maximin search; a fixed count, so its cost does not grow with the batch."""


def _directions(width: int) -> np.ndarray:
    """``(width, _BITS)`` direction integers ``v_i = m_i << (_BITS - 1 - i)``, by the Sobol recurrence."""
    table = np.zeros((width, _BITS), dtype=np.uint64)
    table[0] = [1 << (_BITS - 1 - i) for i in range(_BITS)]
    for dim in range(1, width):
        polynomial, initial = DIRECTIONS[dim - 1]
        degree = polynomial.bit_length() - 1
        m = list(initial)
        for i in range(degree, _BITS):
            value = m[i - degree] ^ (m[i - degree] << degree)
            for k in range(1, degree):
                if (polynomial >> (degree - k)) & 1:
                    value ^= m[i - k] << k
            m.append(value)
        table[dim] = [m[i] << (_BITS - 1 - i) for i in range(_BITS)]
    return table


def _scramble(directions: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Left-multiply each dimension's bit matrix by a random unit lower-triangular matrix, and draw a shift."""
    width = directions.shape[0]
    rows = np.arange(_BITS)
    lower = np.tril(rng.integers(0, 2, (width, _BITS, _BITS), dtype=np.uint64), -1)
    lower[:, rows, rows] = 1
    bits = (directions[:, None, :] >> (np.uint64(_BITS - 1) - rows[None, :, None].astype(np.uint64))) & np.uint64(1)
    mixed = np.einsum('drk,dki->dri', lower, bits) & np.uint64(1)
    weights = (np.uint64(1) << (np.uint64(_BITS - 1) - rows.astype(np.uint64)))[None, :, None]
    scrambled = (mixed * weights).sum(axis=1).astype(np.uint64)
    shift = rng.integers(0, 1 << _BITS, width, dtype=np.uint64)
    return scrambled, shift


def sobol(n_points: int, width: int, *, rng: np.random.Generator | None) -> np.ndarray:
    """The first *n_points* Sobol points in *width* dimensions; scrambled when *rng* is given."""
    require_positive(n_points, 'n_points')
    require_positive(width, 'width')
    if width > MAX_SOBOL_DIMENSION:
        refuse(f'the embedded direction table reaches {MAX_SOBOL_DIMENSION} dimensions, and {width} were asked')
    if n_points > 1 << _BITS:
        refuse(f'a {_BITS}-bit Sobol sequence holds {1 << _BITS} points, and {n_points} were asked')
    directions = _directions(width)
    shift = np.zeros(width, dtype=np.uint64)
    if rng is not None:
        directions, shift = _scramble(directions, rng)
    index = np.arange(n_points, dtype=np.uint64)
    gray = index ^ (index >> np.uint64(1))
    points = np.tile(shift, (n_points, 1))
    for bit in range(_BITS):
        on = ((gray >> np.uint64(bit)) & np.uint64(1)).astype(bool)
        if not on.any():
            break
        points[on] ^= directions[:, bit]
    return points.astype(float) / float(1 << _BITS)


def _primes(count: int) -> list[int]:
    primes: list[int] = []
    candidate = 2
    while len(primes) < count:
        if all(candidate % p for p in primes if p * p <= candidate):
            primes.append(candidate)
        candidate += 1
    return primes


def halton(n_points: int, width: int, *, rng: np.random.Generator | None) -> np.ndarray:
    """The first *n_points* Halton points in *width* dimensions; digit-scrambled when *rng* is given."""
    require_positive(n_points, 'n_points')
    require_positive(width, 'width')
    points = np.zeros((n_points, width))
    for column, base in enumerate(_primes(width)):
        digits = ceil(54 / log2(base)) if rng is not None else max(1, ceil(np.log(n_points + 1) / np.log(base)) + 1)
        index = np.arange(n_points)
        scale = 1.0
        for _ in range(digits):
            scale /= base
            digit = index % base
            if rng is not None:
                digit = rng.permutation(base)[digit]
            points[:, column] += digit * scale
            index //= base
    return points


def latin_hypercube(n_points: int, width: int, rng: np.random.Generator) -> np.ndarray:
    """One point per stratum ``[k/n, (k+1)/n)`` in every column, the strata paired at random."""
    require_positive(n_points, 'n_points')
    require_positive(width, 'width')
    strata = np.argsort(rng.random((width, n_points)), axis=1).T
    return (strata + rng.random((n_points, width))) / n_points


def _morris_mitchell(points: np.ndarray) -> float:
    """``(sum d_ij ** -p) ** (1/p)``, factored through the closest gap so no power overflows."""
    gaps = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)[np.triu_indices(len(points), 1)]
    closest = max(float(gaps.min()), np.finfo(float).tiny)
    return float(np.sum((gaps / closest) ** -_MAXIMIN_POWER) ** (1.0 / _MAXIMIN_POWER) / closest)


def maximin_latin_hypercube(n_points: int, width: int, rng: np.random.Generator) -> np.ndarray:
    """A Latin hypercube whose closest pairs are pushed apart by accepted within-column swaps."""
    points = latin_hypercube(n_points, width, rng)
    if n_points < 3:
        return points
    best = _morris_mitchell(points)
    for _ in range(_MAXIMIN_SWAPS):
        column = rng.integers(width)
        a, b = rng.choice(n_points, 2, replace=False)
        points[[a, b], column] = points[[b, a], column]
        score = _morris_mitchell(points)
        if score < best:
            best = score
        else:
            points[[a, b], column] = points[[b, a], column]
    return points
