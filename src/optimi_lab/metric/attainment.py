"""The empirical attainment function over repeated runs (Fonseca et al., 2001).

A point ``z`` is attained by a run when some point of that run's front weakly dominates it; the
attainment function at ``z`` is the fraction of runs that attain it. Its level sets -- the 50 %
attainment surface is the median front -- are how repeated stochastic runs are compared without
collapsing each to one indicator. Minimisation frame.
"""

from collections.abc import Sequence

import numpy as np

from optimi_lab.core import refuse

__all__ = ['attained']


def attained(runs: Sequence[np.ndarray], probes: np.ndarray) -> np.ndarray:
    """For each probe point, the fraction of *runs* whose front weakly dominates it."""
    if not runs:
        refuse('an attainment function needs at least one run')
    probes = np.atleast_2d(np.asarray(probes, dtype=float))
    hits = np.zeros(len(probes))
    for front in runs:
        front = np.asarray(front, dtype=float).reshape(-1, probes.shape[1])
        hits += np.any(np.all(front[None, :, :] <= probes[:, None, :], axis=-1), axis=1)
    return hits / len(runs)
