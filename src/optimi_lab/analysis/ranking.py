"""Sensitivity results ranked: each variable's measures, with intervals, most influential first.

A :class:`SensitivityRanking` holds one or more :class:`SensitivityMeasure` per variable and names the one it
ranks by; :meth:`SensitivityRanking.rows` returns :class:`RankedRow` records in descending order of
the ranking measure's MAGNITUDE (a correlation of -0.9 outranks one of 0.5).

* :func:`rank_sobol` -- ranked by the total index ``S_T``, first order beside it, both with intervals.
* :func:`rank_morris` -- ranked by ``mu*``; ``mu``, ``sigma`` and the standard error ``sigma / sqrt(r)``
  of ``mu`` beside it (``r`` the trajectory count, which :class:`MorrisEffects` does not carry).
* :func:`rank_by_correlation` -- Pearson, Spearman and Kendall tau-b of every input column against one
  output, each with a percentile bootstrap interval; ranked by ``|by|``. A correlation is an
  ASSOCIATION, so such a ranking is flagged :attr:`SensitivityRanking.correlational`.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np

from optimi_lab.analysis.morris import MorrisEffects
from optimi_lab.analysis.sensitivity import SobolIndices
from optimi_lab.core.errors import refuse
from optimi_lab.metric.statistics import bootstrap_interval, kendall_tau_b, pearson, spearman

__all__ = [
    'CORRELATIONS',
    'RankedRow',
    'SensitivityMeasure',
    'SensitivityRanking',
    'rank_by_correlation',
    'rank_morris',
    'rank_sobol',
]

_COEFFICIENTS: Final[dict[str, Callable[[np.ndarray, np.ndarray], float]]] = {
    'pearson': lambda x, y: float(pearson(np.column_stack([x, y]))),
    'spearman': lambda x, y: float(spearman(np.column_stack([x, y]))),
    'kendall': lambda x, y: float(kendall_tau_b(x, y)),
}

#: The declared correlation coefficients.
CORRELATIONS: Final = tuple(_COEFFICIENTS)


@dataclass(frozen=True, slots=True)
class SensitivityMeasure:
    """One measure per variable: its ``(k,)`` estimate and ``(k, 2)`` interval, or ``None`` when it has none."""

    name: str
    estimate: np.ndarray
    interval: np.ndarray | None = None


@dataclass(frozen=True, slots=True)
class RankedRow:
    """One variable's place: ``rank`` from 1, the ranking ``value`` and ``interval``, and every measure's estimate."""

    rank: int
    name: str
    value: float
    interval: tuple[float, float] | None
    measures: dict[str, float]


@dataclass(frozen=True, slots=True)
class SensitivityRanking:
    """One method's measures over *names*, ranked by the magnitude of the measure *rank_by*."""

    method: str
    names: tuple[str, ...]
    measures: tuple[SensitivityMeasure, ...]
    rank_by: str
    n_evaluations: int
    confidence: float | None
    correlational: bool

    def __post_init__(self) -> None:
        known = [m.name for m in self.measures]
        if self.rank_by not in known:
            refuse(f'a ranking ranks by one of its measures {known}, got {self.rank_by!r}')

    def measure(self, name: str) -> SensitivityMeasure:
        """The measure called *name*; a stranger is refused naming the set."""
        for measure in self.measures:
            if measure.name == name:
                return measure
        return refuse(f'a measure is one of {[m.name for m in self.measures]}, got {name!r}')

    def order(self) -> tuple[int, ...]:
        """Variable indices, most influential first (stable on ties)."""
        return tuple(int(i) for i in np.argsort(-np.abs(self.measure(self.rank_by).estimate), kind='stable'))

    def rows(self) -> tuple[RankedRow, ...]:
        """The variables as :class:`RankedRow`, most influential first."""
        key = self.measure(self.rank_by)
        return tuple(
            RankedRow(
                rank=place + 1,
                name=self.names[i],
                value=float(key.estimate[i]),
                interval=None if key.interval is None else (float(key.interval[i, 0]), float(key.interval[i, 1])),
                measures={m.name: float(m.estimate[i]) for m in self.measures},
            )
            for place, i in enumerate(self.order())
        )


def rank_sobol(indices: SobolIndices, *, confidence: float) -> SensitivityRanking:
    """*indices* ranked by the total index; *confidence* is the level their intervals were drawn at."""
    return SensitivityRanking(
        'sobol',
        tuple(indices.names),
        (
            SensitivityMeasure('total', indices.total, indices.total_interval),
            SensitivityMeasure('first_order', indices.first_order, indices.first_order_interval),
        ),
        'total',
        indices.n_evaluations,
        confidence,
        correlational=False,
    )


def rank_morris(effects: MorrisEffects, *, trajectories: int) -> SensitivityRanking:
    """*effects* ranked by ``mu*``; *trajectories* gives the standard error ``sigma / sqrt(r)`` of ``mu``."""
    if trajectories < 2:
        refuse(f'Morris effects come from at least two trajectories, got {trajectories}')
    return SensitivityRanking(
        'morris',
        tuple(effects.names),
        (
            SensitivityMeasure('mu_star', effects.mu_star),
            SensitivityMeasure('mu', effects.mu),
            SensitivityMeasure('sigma', effects.sigma),
            SensitivityMeasure('mu_standard_error', effects.sigma / np.sqrt(trajectories)),
        ),
        'mu_star',
        effects.n_evaluations,
        None,
        correlational=False,
    )


def rank_by_correlation(
    inputs: np.ndarray,
    output: np.ndarray,
    names: Sequence[str],
    *,
    by: str = 'spearman',
    confidence: float = 0.95,
    n_bootstrap: int = 500,
    seed: int,
) -> SensitivityRanking:
    """Every :data:`CORRELATIONS` coefficient of each column of *inputs* against *output*, ranked by ``|by|``.

    Column ``j``'s interval resamples the ``(x_j, y)`` pairs with ``default_rng(seed + j)``.
    """
    inputs, output = np.asarray(inputs, dtype=float), np.asarray(output, dtype=float)
    if inputs.ndim != 2 or output.shape != (inputs.shape[0],) or len(names) != inputs.shape[1]:
        refuse(
            f'a correlation ranking pairs (n, k) inputs with n outputs and k names; '
            f'got {inputs.shape}, {output.shape} and {len(names)}'
        )
    if by not in _COEFFICIENTS:
        refuse(f'a correlation is one of {list(CORRELATIONS)}, got {by!r}')
    measures = []
    for name, coefficient in _COEFFICIENTS.items():
        estimates, intervals = [], []
        for column in range(inputs.shape[1]):
            pair = np.column_stack([inputs[:, column], output])
            estimates.append(coefficient(pair[:, 0], pair[:, 1]))
            intervals.append(
                bootstrap_interval(
                    pair,
                    lambda drawn, coefficient=coefficient: coefficient(drawn[:, 0], drawn[:, 1]),
                    confidence=confidence,
                    n_resamples=n_bootstrap,
                    rng=np.random.default_rng(seed + column),
                )
            )
        measures.append(SensitivityMeasure(name, np.array(estimates), np.array(intervals)))
    return SensitivityRanking(
        'correlation', tuple(names), tuple(measures), by, inputs.shape[0], confidence, correlational=True
    )
