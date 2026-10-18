"""Variance-based global sensitivity: Sobol' indices by Saltelli sampling, and a binned main effect.

:func:`sobol_indices` draws two scrambled Sobol' matrices ``A`` and ``B`` of ``n`` rows over the
unit cube (one ``2k``-dimensional sequence split in half), builds the ``k`` cross matrices ``AB_i``
(``A`` with column ``i`` taken from ``B``), maps all of them onto the :class:`~optimi_lab.problem.Space` (Real variables only, so the
measure is a product of intervals), and
calls the vectorised function ONCE on the stacked ``n (k + 2)`` points. The estimators are the
standard pair (Saltelli et al. 2010, Table 2): first order ``mean(f_B (f_AB_i - f_A)) / V`` and
Jansen's total ``mean((f_A - f_AB_i)^2) / (2 V)``. A percentile bootstrap over the ``n`` base rows
gives each index its interval; the rows are resampled jointly, so ``A``, ``B`` and ``AB_i`` stay
paired.

:func:`main_effect` is the cheap reading from a sample that ALREADY exists -- a DoE or an
optimizer's history: the variance of the output's conditional mean over equal-count bins of each
input, divided by the output variance. It approximates the first-order index and needs no new
evaluation, at the price of a positive bias of about ``n_bins / n_points`` from bin-mean noise.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from optimi_lab.core.errors import refuse, require_finite, require_positive
from optimi_lab.core.qmc import sobol
from optimi_lab.problem import Real, Space

__all__ = ['SobolIndices', 'main_effect', 'sobol_indices']


@dataclass(frozen=True, slots=True)
class SobolIndices:
    """First-order and total indices per variable, each with a ``(k, 2)`` bootstrap interval."""

    names: tuple[str, ...]
    first_order: np.ndarray
    total: np.ndarray
    first_order_interval: np.ndarray
    total_interval: np.ndarray
    n_evaluations: int


def _estimate(f_a: np.ndarray, f_b: np.ndarray, f_ab: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Saltelli first order and Jansen total over the trailing row axis; ``f_ab`` is ``(..., k, n)``."""
    variance = np.var(np.concatenate([f_a, f_b], axis=-1), axis=-1)[..., None]
    first = np.mean(f_b[..., None, :] * (f_ab - f_a[..., None, :]), axis=-1) / variance
    total = 0.5 * np.mean((f_a[..., None, :] - f_ab) ** 2, axis=-1) / variance
    return first, total


def sobol_indices(
    space: Space,
    function: Callable[[np.ndarray], np.ndarray],
    n_base: int,
    *,
    seed: int,
    n_bootstrap: int = 500,
    confidence: float = 0.95,
) -> SobolIndices:
    """Sobol' first-order and total indices of ``function`` over ``space``.

    ``function`` maps a ``(n_points, n_var)`` matrix to ``n_points`` values and is called once on
    ``n_base * (n_var + 2)`` points. ``n_base`` must be a power of two, the size at which a Sobol'
    sequence keeps its balance; a function whose output does not vary has no variance to apportion
    and is refused rather than divided by zero.
    """
    require_positive(n_base, 'n_base')
    if n_base & (n_base - 1):
        refuse(f'n_base must be a power of two for a balanced Sobol sequence, got {n_base}')
    if not 0.0 < confidence < 1.0:
        refuse(f'a confidence level lies strictly between 0 and 1, got {confidence}')
    strangers = [v.name for v in space if not isinstance(v, Real) or v.active_if is not None]
    if strangers:
        refuse(f'Sobol indices here need unconditional Real variables; {strangers} are not')
    n_var = space.width
    unit = sobol(n_base, 2 * n_var, rng=np.random.default_rng(seed))
    unit_a, unit_b = unit[:, :n_var], unit[:, n_var:]
    unit_ab = np.repeat(unit_a[None], n_var, axis=0)
    unit_ab[np.arange(n_var), :, np.arange(n_var)] = unit_b.T
    unit_points = np.vstack([unit_a, unit_b, *unit_ab])
    points = np.array([[v.decode(row[i : i + 1]) for i, v in enumerate(space)] for row in unit_points])
    values = np.asarray(function(points), dtype=float)
    if values.shape != (points.shape[0],):
        refuse(f'the function must return one value per point, shape ({points.shape[0]},); got {values.shape}')
    require_finite(values, 'the function values')
    f_a, f_b, f_ab = values[:n_base], values[n_base : 2 * n_base], values[2 * n_base :].reshape(n_var, n_base)
    if np.var(np.concatenate([f_a, f_b])) == 0.0:
        refuse('the function output has zero variance over the sample; there is nothing to apportion')
    first, total = _estimate(f_a, f_b, f_ab)
    rows = np.random.default_rng(seed).integers(0, n_base, (n_bootstrap, n_base))
    boot_first, boot_total = _estimate(f_a[rows], f_b[rows], np.moveaxis(f_ab[:, rows], 0, 1))
    tails = 100.0 * np.array([(1.0 - confidence) / 2.0, (1.0 + confidence) / 2.0])
    return SobolIndices(
        names=space.names,
        first_order=first,
        total=total,
        first_order_interval=np.percentile(boot_first, tails, axis=0).T,
        total_interval=np.percentile(boot_total, tails, axis=0).T,
        n_evaluations=points.shape[0],
    )


def main_effect(inputs: np.ndarray, outputs: np.ndarray, *, n_bins: int = 10) -> np.ndarray:
    """Per input column, ``Var(E[y | x_i]) / Var(y)`` from equal-count bins of an existing sample."""
    inputs = np.asarray(inputs, dtype=float)
    outputs = np.asarray(outputs, dtype=float)
    if inputs.ndim != 2 or outputs.shape != (inputs.shape[0],):
        refuse(f'a main effect needs one output per input row; got inputs {inputs.shape} and outputs {outputs.shape}')
    require_positive(n_bins, 'n_bins')
    require_finite(inputs, 'the sample inputs')
    require_finite(outputs, 'the sample outputs')
    variance = np.var(outputs)
    if variance == 0.0:
        refuse('the sample output has zero variance; there is nothing to apportion')
    effects = np.empty(inputs.shape[1])
    for column in range(inputs.shape[1]):
        bins = np.array_split(np.argsort(inputs[:, column], kind='stable'), n_bins)
        means = np.array([outputs[members].mean() for members in bins if members.size])
        counts = np.array([members.size for members in bins if members.size])
        effects[column] = np.sum(counts * (means - outputs.mean()) ** 2) / outputs.size / variance
    return effects
