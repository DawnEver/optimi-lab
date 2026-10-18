"""D59-D60 -- acquisition functions: closed forms over a Gaussian prediction, sample forms over joint draws.

Every acquisition reads the MINIMISED frame (:meth:`~optimi_lab.problem.Objective.minimized`) and
returns a score where LARGER IS BETTER. A prediction is ``mean`` and ``std`` (the latent standard
deviation, ``>= 0``) per point; ``std = 0`` is the limit of each closed form, never a division.

Single objective, incumbent ``best`` (the lowest observed feasible value), ``z = (best - mean) / std``:

* :func:`expected_improvement` ``std h(z)`` with ``h(z) = z Phi(z) + phi(z)``; :func:`log_expected_improvement`
  is its logarithm computed WITHOUT forming it (Ament et al., 2023): ``log h`` directly for ``z > -1``,
  through ``erfcx`` for ``-1e3 < z <= -1`` and by the asymptotic series below, so the score stays
  finite and ordered where ``EI`` underflows to 0;
* :func:`probability_of_improvement` ``Phi(z)`` and :func:`log_probability_of_improvement` ``log_ndtr(z)``;
* :func:`upper_confidence_bound` ``-mean + sqrt(beta) std`` -- the lower confidence bound, negated;
* :func:`knowledge_gradient` ``E[max_i a_i + b_i Z] - max_i a_i`` over a discrete set (Frazier,
  Powell and Dayanik, 2009), the upper envelope of the lines read off exactly.

Constraints: :func:`log_probability_of_feasibility` -- every modelled margin ``<= 0`` (independent
Gaussians); the constrained EI (Schonlau et al., 1998) is ``log EI + log PF``.

Several objectives, reference point ``reference``:

* :func:`expected_hypervolume_improvement` -- EXACT for independent Gaussian objectives in any number
  of objectives: the region the front does not dominate is cut into the cells of the front's
  coordinate grid, and ``EHVI = sum_cells prod_j [Psi_j(u_j) - Psi_j(l_j)]`` with
  ``Psi_j(t) = std_j h((t - mean_j) / std_j)`` (``Psi(-inf) = 0``). It costs ``(k + 1) ** m`` cells
  for ``k`` front points.
* :func:`hypervolume_improvement` -- the exact improvement of single points, over the same cells,
  vectorised: what a sample-average (MC) acquisition averages.
* :func:`noisy_expected_hypervolume_improvement` -- NEHVI (Daulton et al., 2021): over joint draws
  of the baseline (observed and pending points) and the candidate, the mean improvement of the
  candidate over each draw's own front, with its standard error.
* :func:`parego_scores` -- ParEGO's scalarisation (Knowles, 2006): the augmented achievement
  function with the weights as MULTIPLIERS, ``max_j w_j f_j + rho sum_j w_j f_j`` (Tchebycheff plus
  ``rho`` times the weighted sum), on columns normalised to ``[0, 1]`` by their range. A weight
  row may hold zeros; a divided ASF would make a zero weight the dominant term.
"""

from typing import Final

import numpy as np
from scipy.special import erfcx, log_ndtr, ndtr

from optimi_lab.core import non_dominated_sorting, refuse
from optimi_lab.operator.estimators import Estimate
from optimi_lab.operator.scalarisation import tchebycheff, weighted_sum

__all__ = [
    'expected_hypervolume_improvement',
    'expected_improvement',
    'hypervolume_improvement',
    'knowledge_gradient',
    'log_expected_improvement',
    'log_h',
    'log_probability_of_feasibility',
    'log_probability_of_improvement',
    'noisy_expected_hypervolume_improvement',
    'parego_scores',
    'probability_of_improvement',
    'upper_confidence_bound',
]

_LOG_SQRT_2PI: Final = 0.5 * float(np.log(2.0 * np.pi))
_LOG_SQRT_HALF_PI: Final = 0.5 * float(np.log(np.pi / 2.0))
_SERIES_FROM: Final = -1e3
"""Below this ``z`` the ``erfcx`` form loses digits to cancellation and the asymptotic series is exact to eps."""
_SAME_SLOPE: Final = 1e-12
"""Relative to the largest slope: lines closer than this are one line (their crossing is noise)."""
_FAR: Final = 40.0
"""A breakpoint beyond ``|z| = 40`` adds ``h(-40) ~ 1e-350``: zero in double precision."""


def _gaussian(mean: object, std: object) -> tuple[np.ndarray, np.ndarray]:
    mean, std = np.asarray(mean, dtype=float), np.asarray(std, dtype=float)
    if mean.shape != std.shape:
        refuse(f'a prediction pairs one std with each mean, got shapes {mean.shape} and {std.shape}')
    if np.any(~(std >= 0.0)):
        refuse('a standard deviation is >= 0 and not NaN')
    return mean, std


def _h(z: np.ndarray) -> np.ndarray:
    return z * ndtr(z) + np.exp(-0.5 * z**2) / np.sqrt(2.0 * np.pi)


def _log1mexp(a: np.ndarray) -> np.ndarray:
    """``log(1 - exp(a))`` for ``a < 0`` without cancellation."""
    return np.where(a > -np.log(2.0), np.log(-np.expm1(a)), np.log1p(-np.exp(a)))


def log_h(z: object) -> np.ndarray:
    """``log(z Phi(z) + phi(z))``, finite for every finite ``z``."""
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    upper, tail = z > -1.0, z <= _SERIES_FROM
    middle = ~upper & ~tail
    out[upper] = np.log(_h(z[upper]))
    zm = z[middle]
    out[middle] = -0.5 * zm**2 - _LOG_SQRT_2PI + _log1mexp(np.log(erfcx(-zm / np.sqrt(2.0)) * -zm) + _LOG_SQRT_HALF_PI)
    zt = z[tail]
    out[tail] = -0.5 * zt**2 - _LOG_SQRT_2PI - 2.0 * np.log(-zt) + np.log1p(-3.0 / zt**2 + 15.0 / zt**4)
    return out


def _standardised(mean: np.ndarray, std: np.ndarray, best: float) -> tuple[np.ndarray, np.ndarray]:
    positive = std > 0.0
    z = np.where(positive, (best - mean) / np.where(positive, std, 1.0), 0.0)
    return z, positive


def expected_improvement(mean: object, std: object, best: float) -> np.ndarray:
    """``E[max(best - Y, 0)]`` for ``Y ~ N(mean, std^2)``."""
    mean, std = _gaussian(mean, std)
    z, positive = _standardised(mean, std, best)
    return np.where(positive, std * _h(z), np.maximum(best - mean, 0.0))


def log_expected_improvement(mean: object, std: object, best: float) -> np.ndarray:
    """``log EI``; ``-inf`` only where ``std = 0`` and no improvement is possible."""
    mean, std = _gaussian(mean, std)
    z, positive = _standardised(mean, std, best)
    with np.errstate(divide='ignore'):
        exact = np.log(np.maximum(best - mean, 0.0))
        return np.where(positive, np.log(np.where(positive, std, 1.0)) + log_h(z), exact)


def probability_of_improvement(mean: object, std: object, best: float) -> np.ndarray:
    """``P(Y < best)``."""
    mean, std = _gaussian(mean, std)
    z, positive = _standardised(mean, std, best)
    return np.where(positive, ndtr(z), (mean < best).astype(float))


def log_probability_of_improvement(mean: object, std: object, best: float) -> np.ndarray:
    """``log P(Y < best)``, finite far into the tail."""
    mean, std = _gaussian(mean, std)
    z, positive = _standardised(mean, std, best)
    with np.errstate(divide='ignore'):
        return np.where(positive, log_ndtr(z), np.log((mean < best).astype(float)))


def upper_confidence_bound(mean: object, std: object, beta: float) -> np.ndarray:
    """``-mean + sqrt(beta) std``: the optimistic bound of a minimisation, signed so larger is better."""
    if not beta >= 0.0:
        refuse(f'beta is a squared width multiplier >= 0, got {beta!r}')
    mean, std = _gaussian(mean, std)
    return -mean + np.sqrt(beta) * std


def knowledge_gradient(a: object, b: object) -> float:
    """``E[max_i (a_i + b_i Z)] - max_i a_i``, ``Z ~ N(0, 1)``: the value of one more observation.

    ``a`` are the current values of the set (larger is better), ``b`` how each moves per unit of the
    observation's standardised outcome. Lines sorted by slope (slopes within ``1e-12`` relative are
    one line, the best of them) are reduced to their upper envelope; with breakpoints ``c_j`` it is
    ``sum_j (b_{j+1} - b_j) h(-|c_j|)``.
    """
    a, b = np.asarray(a, dtype=float).ravel(), np.asarray(b, dtype=float).ravel()
    if a.shape != b.shape or a.size == 0:
        refuse(f'a and b pair one value with one slope per point, got {a.shape} and {b.shape}')
    order = np.lexsort((a, b))
    a, b = a[order], b[order]
    keep = np.append(np.diff(b) > _SAME_SLOPE * max(float(np.abs(b).max()), np.finfo(float).tiny), True)
    a, b = a[keep], b[keep]
    lines: list[int] = []
    cuts: list[float] = []
    for i in range(len(a)):
        while lines:
            j = lines[-1]
            cut = (a[j] - a[i]) / (b[i] - b[j])
            if cuts and cut <= cuts[-1]:
                lines.pop()
                cuts.pop()
                continue
            cuts.append(cut)
            break
        lines.append(i)
    if len(lines) == 1:
        return 0.0
    slopes = np.diff(b[lines])
    return float(np.sum(slopes * _h(-np.minimum(np.abs(np.array(cuts)), _FAR))))


def log_probability_of_feasibility(margin_mean: object, margin_std: object) -> np.ndarray:
    """``sum_c log P(G_c <= 0)`` per row of ``(n, c)`` margins; a row with no margin is certain."""
    mean, std = _gaussian(margin_mean, margin_std)
    mean, std = np.atleast_2d(mean), np.atleast_2d(std)
    z, positive = _standardised(mean, std, 0.0)
    with np.errstate(divide='ignore'):
        return np.where(positive, log_ndtr(z), np.log((mean <= 0.0).astype(float))).sum(axis=1)


def _front_and_reference(front: object, reference: object) -> tuple[np.ndarray, np.ndarray]:
    reference = np.asarray(reference, dtype=float).ravel()
    front = np.asarray(front, dtype=float).reshape(-1, reference.size)
    if reference.size < 2 or not np.all(np.isfinite(reference)) or not np.all(np.isfinite(front)):
        refuse(f'a hypervolume acquisition needs m >= 2 finite objectives and a finite reference, got {reference}')
    front = front[np.all(front < reference, axis=1)]
    return (front[non_dominated_sorting(front)] if len(front) else front), reference


def _cells(front: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(lower, upper)`` corners ``(C, m)`` of the grid cells *front* does not dominate below *reference*."""
    axes = [np.concatenate([[-np.inf], np.unique(front[:, j]), [reference[j]]]) for j in range(reference.size)]
    lower = np.stack(np.meshgrid(*(axis[:-1] for axis in axes), indexing='ij'), axis=-1).reshape(-1, reference.size)
    upper = np.stack(np.meshgrid(*(axis[1:] for axis in axes), indexing='ij'), axis=-1).reshape(-1, reference.size)
    open_ = upper > lower
    keep = np.all(open_, axis=1)
    if len(front):
        keep &= ~np.any(np.all(front[None, :, :] <= lower[:, None, :], axis=2), axis=1)
    return lower[keep], upper[keep]


def expected_hypervolume_improvement(mean: object, std: object, front: object, reference: object) -> np.ndarray:
    """Exact EHVI of each row of independent Gaussian objectives ``(n, m)`` over *front*, below *reference*."""
    mean, std = _gaussian(mean, std)
    front, reference = _front_and_reference(front, reference)
    mean, std = mean.reshape(-1, reference.size), std.reshape(-1, reference.size)
    lower, upper = _cells(front, reference)

    def psi(t: np.ndarray) -> np.ndarray:
        positive = std[:, None, :] > 0.0
        scale = np.where(positive, std[:, None, :], 1.0)
        with np.errstate(invalid='ignore'):
            smooth = scale * _h((t[None] - mean[:, None, :]) / scale)
        exact = np.maximum(t[None] - mean[:, None, :], 0.0)
        return np.where(np.isneginf(t)[None], 0.0, np.where(positive, smooth, exact))

    return np.prod(psi(upper) - psi(lower), axis=2).sum(axis=1)


def hypervolume_improvement(points: object, front: object, reference: object) -> np.ndarray:
    """The exact hypervolume each row of *points* ``(n, m)`` adds to *front* alone, below *reference*."""
    front, reference = _front_and_reference(front, reference)
    points = np.asarray(points, dtype=float).reshape(-1, reference.size)
    lower, upper = _cells(front, reference)
    sides = upper[None] - np.maximum(lower[None], points[:, None, :])
    return np.prod(np.maximum(sides, 0.0), axis=2).sum(axis=1)


def noisy_expected_hypervolume_improvement(baseline: object, candidate: object, reference: object) -> Estimate:
    """NEHVI: the mean over draws ``s`` of the candidate's improvement over draw ``s``'s baseline front.

    *baseline* ``(s, n, m)`` and *candidate* ``(s, m)`` are JOINT draws, one per row ``s``.
    """
    reference = np.asarray(reference, dtype=float).ravel()
    baseline = np.asarray(baseline, dtype=float)
    candidate = np.asarray(candidate, dtype=float).reshape(-1, reference.size)
    if baseline.ndim != 3 or baseline.shape[0] != len(candidate) or baseline.shape[2] != reference.size:
        refuse(f'baseline draws are (s, n, {reference.size}) beside (s, {reference.size}) candidate draws')
    gains = np.array([hypervolume_improvement(c, b, reference)[0] for b, c in zip(baseline, candidate, strict=True)])
    error = float(gains.std(ddof=1) / np.sqrt(len(gains))) if len(gains) > 1 else float('nan')
    return Estimate(float(gains.mean()), error, len(gains))


def parego_scores(values: object, weights: object, *, rho: float = 0.05) -> np.ndarray:
    """ParEGO's score of each minimised row: the augmented Tchebycheff of the range-normalised columns."""
    values = np.atleast_2d(np.asarray(values, dtype=float))
    low, high = values.min(axis=0), values.max(axis=0)
    scaled = (values - low) / np.where(high > low, high - low, 1.0)
    origin = np.zeros(values.shape[1])
    return tchebycheff(scaled, weights, origin) + rho * weighted_sum(scaled, weights, origin)
