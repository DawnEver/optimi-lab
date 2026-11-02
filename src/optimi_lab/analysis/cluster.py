"""k-means (Lloyd's iterations) over a finished sample: a region label per point, for stratifying a pick."""

import numpy as np

from optimi_lab.core.errors import refuse

__all__ = ['kmeans']


def kmeans(x: np.ndarray, k: int, rng: np.random.Generator, iterations: int = 50) -> np.ndarray:
    """The cluster label of each row of *x*: *k* centres seeded at distinct rows drawn by *rng*, then Lloyd.

    An emptied cluster keeps its previous centre. Exactly *iterations* assignment-update rounds run.
    """
    x = np.asarray(x, dtype=float)
    if not 1 <= k <= len(x):
        refuse(f'k-means needs 1 <= k <= the number of points ({len(x)}), got {k}')
    centres = x[rng.choice(len(x), size=k, replace=False)]
    labels = np.zeros(len(x), dtype=int)
    for _ in range(iterations):
        labels = np.argmin(((x[:, None, :] - centres[None]) ** 2).sum(-1), axis=1)
        centres = np.array([x[labels == j].mean(0) if np.any(labels == j) else centres[j] for j in range(k)])
    return labels
