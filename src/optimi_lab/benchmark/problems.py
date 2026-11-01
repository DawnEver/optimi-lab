"""ZDT1-4 and DTLZ2: evaluation over rows ``(n, n_var)`` and the true front sampled at *n* points.

* ZDT1 / ZDT2 -- the connected convex / concave fronts.
* ZDT3 -- a DISCONNECTED front, so a search converging to one segment is told apart.
* ZDT4 -- the ZDT1 front behind a Rastrigin ``g`` with many local fronts: the multimodal case.
* DTLZ2 -- ``n_obj`` objectives, the optimal set on the positive octant of the unit sphere.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np

from optimi_lab.core.errors import refuse
from optimi_lab.core.pareto import non_dominated_sorting

__all__ = ['PROBLEMS', 'Benchmark', 'dtlz2', 'zdt1', 'zdt2', 'zdt3', 'zdt4']

#: The ZDT3 front is the non-dominated part of ``f2 = 1 - sqrt(f1) - f1 sin(10 pi f1)`` on [0, 1];
#: a sample this many times denser than asked is filtered down to it.
_ZDT3_OVERSAMPLE = 40


@dataclass(frozen=True)
class Benchmark:
    """A named analytic problem: box bounds, a row-wise evaluation and its true front."""

    name: str
    n_var: int
    n_obj: int
    xl: np.ndarray
    xu: np.ndarray
    evaluate: Callable[[np.ndarray], np.ndarray]
    pareto_front: Callable[[int], np.ndarray]

    def __post_init__(self) -> None:
        if self.n_var < 2 or not 2 <= self.n_obj <= self.n_var:
            refuse(f'a benchmark needs n_var >= 2 and 2 <= n_obj <= n_var, got {self.n_var} and {self.n_obj}')


def _zdt_g(x: np.ndarray) -> np.ndarray:
    return 1.0 + 9.0 * np.mean(x[:, 1:], axis=1)


def _unit(n_var: int) -> tuple[np.ndarray, np.ndarray]:
    return np.zeros(n_var), np.ones(n_var)


def _require(n_var: int) -> None:
    if n_var < 2:
        refuse(f'a ZDT benchmark needs at least two variables, got {n_var}')


def zdt1(n_var: int = 10) -> Benchmark:
    """ZDT1: the connected convex front ``f2 = 1 - sqrt(f1)``."""
    _require(n_var)

    def evaluate(x: np.ndarray) -> np.ndarray:
        f1, g = x[:, 0], _zdt_g(x)
        return np.column_stack((f1, g * (1.0 - np.sqrt(f1 / g))))

    def pareto_front(n: int) -> np.ndarray:
        f1 = np.linspace(0.0, 1.0, n)
        return np.column_stack((f1, 1.0 - np.sqrt(f1)))

    return Benchmark('ZDT1', n_var, 2, *_unit(n_var), evaluate, pareto_front)


def zdt2(n_var: int = 10) -> Benchmark:
    """ZDT2: the connected concave front ``f2 = 1 - f1^2``."""
    _require(n_var)

    def evaluate(x: np.ndarray) -> np.ndarray:
        f1, g = x[:, 0], _zdt_g(x)
        return np.column_stack((f1, g * (1.0 - (f1 / g) ** 2)))

    def pareto_front(n: int) -> np.ndarray:
        f1 = np.linspace(0.0, 1.0, n)
        return np.column_stack((f1, 1.0 - f1**2))

    return Benchmark('ZDT2', n_var, 2, *_unit(n_var), evaluate, pareto_front)


def zdt3(n_var: int = 10) -> Benchmark:
    """ZDT3: two objectives, a front of five disconnected segments."""
    _require(n_var)

    def evaluate(x: np.ndarray) -> np.ndarray:
        f1 = x[:, 0]
        g = 1.0 + 9.0 * np.sum(x[:, 1:], axis=1) / (n_var - 1)
        r = f1 / g
        return np.column_stack([f1, g * (1.0 - np.sqrt(r) - r * np.sin(10.0 * np.pi * f1))])

    def pareto_front(n: int) -> np.ndarray:
        f1 = np.linspace(0.0, 1.0, n * _ZDT3_OVERSAMPLE)
        f = np.column_stack([f1, 1.0 - np.sqrt(f1) - f1 * np.sin(10.0 * np.pi * f1)])
        return f[np.sort(np.asarray(non_dominated_sorting(f)).ravel())]

    return Benchmark('ZDT3', n_var, 2, *_unit(n_var), evaluate, pareto_front)


def zdt4(n_var: int = 10) -> Benchmark:
    """ZDT4: the convex ZDT1 front behind a Rastrigin ``g`` over ``[-5, 5]``."""
    _require(n_var)

    def evaluate(x: np.ndarray) -> np.ndarray:
        f1, rest = x[:, 0], x[:, 1:]
        g = 1.0 + 10.0 * (n_var - 1) + np.sum(rest**2 - 10.0 * np.cos(4.0 * np.pi * rest), axis=1)
        return np.column_stack([f1, g * (1.0 - np.sqrt(f1 / g))])

    xl = np.concatenate([[0.0], np.full(n_var - 1, -5.0)])
    xu = np.concatenate([[1.0], np.full(n_var - 1, 5.0)])
    return Benchmark('ZDT4', n_var, 2, xl, xu, evaluate, zdt1(n_var).pareto_front)


def _simplex_lattice(n_obj: int, divisions: int) -> list[list[float]]:
    """Every point of the unit simplex whose coordinates are multiples of ``1 / divisions``."""
    if n_obj == 1:
        return [[1.0]]
    return [
        [k / divisions, *((divisions - k) / divisions * c for c in rest)]
        for k in range(divisions + 1)
        for rest in (_simplex_lattice(n_obj - 1, divisions - k) if k < divisions else [[0.0] * (n_obj - 1)])
    ]


def dtlz2(n_var: int = 10, n_obj: int = 3) -> Benchmark:
    """DTLZ2: ``n_obj`` objectives whose optimal set is the positive octant of the unit sphere."""

    def evaluate(x: np.ndarray) -> np.ndarray:
        g = np.sum((x[:, n_obj - 1 :] - 0.5) ** 2, axis=1)
        angles = x[:, : n_obj - 1] * np.pi / 2.0
        f = np.tile((1.0 + g)[:, None], (1, n_obj))
        for m in range(n_obj):
            f[:, m] *= np.prod(np.cos(angles[:, : n_obj - 1 - m]), axis=1)
            if m > 0:
                f[:, m] *= np.sin(angles[:, n_obj - 1 - m])
        return f

    def pareto_front(n: int) -> np.ndarray:
        divisions = 1
        while len(_simplex_lattice(n_obj, divisions)) < n:
            divisions += 1
        lattice = np.array(_simplex_lattice(n_obj, divisions))
        return lattice / np.linalg.norm(lattice, axis=1, keepdims=True)

    return Benchmark('DTLZ2', n_var, n_obj, *_unit(n_var), evaluate, pareto_front)


#: The benchmark problems by name, each built from its variable count.
PROBLEMS: Mapping[str, Callable[[int], Benchmark]] = {
    'ZDT1': zdt1,
    'ZDT2': zdt2,
    'ZDT3': zdt3,
    'ZDT4': zdt4,
    'DTLZ2': dtlz2,
}
