"""D125 optimal computing budget allocation (Chen, Lin, Yucesan and Chick, 2000).

Given the sample *means* and standard deviations *stds* of ``k`` designs (MINIMISED) and a *budget*
of replicates, OCBA asymptotically maximises the probability of correct selection by

    N_i / N_j = (s_i / d_i)^2 / (s_j / d_j)^2           for i, j != b,   d_i = mean_i - mean_b
    N_b = s_b sqrt(sum_{i != b} N_i^2 / s_i^2)

where ``b`` is the observed best. The real-valued shares are rounded by largest remainder, so the
integers sum to the budget exactly. A sequential procedure calls this once per stage on the updated
statistics (the L8 replicate-allocation policy owns that loop).
"""

from __future__ import annotations

import numpy as np

from optimi_lab.core.errors import refuse

__all__ = ['ocba']


def ocba(means, stds, budget: int) -> np.ndarray:
    """Integer replicates per design summing to *budget*; see the module docstring."""
    means, stds = np.asarray(means, dtype=float), np.asarray(stds, dtype=float)
    if means.ndim != 1 or means.shape != stds.shape or means.size < 2:
        refuse(f'OCBA needs one mean and one std per design, at least two designs; got {means.shape}, {stds.shape}')
    if not np.all(stds > 0):
        refuse('OCBA needs a positive standard deviation for every design')
    best = int(np.argmin(means))
    gaps = means - means[best]
    others = np.arange(means.size) != best
    if np.any(gaps[others] == 0.0):
        refuse(f'OCBA needs a unique best design; there is a tie at mean {means[best]!r}')
    shares = np.zeros(means.size)
    shares[others] = (stds[others] / gaps[others]) ** 2
    shares[best] = stds[best] * np.sqrt(np.sum(shares[others] ** 2 / stds[others] ** 2))
    exact = budget * shares / shares.sum()
    allocation = np.floor(exact).astype(int)
    remainder = int(budget - allocation.sum())
    allocation[np.argsort(-(exact - allocation), kind='stable')[:remainder]] += 1
    return allocation
