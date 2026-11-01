"""A stratified pick: round-robin over sorted strata, each stratum shuffled by an explicit generator."""

from collections import defaultdict
from collections.abc import Hashable, Sequence

import numpy as np

from optimi_lab.core.errors import refuse

__all__ = ['stratified_pick']


def stratified_pick(
    strata: Sequence[Hashable], k: int, rng: np.random.Generator, *, chosen: Sequence[int] = ()
) -> list[int]:
    """*k* point indices: *chosen* first, then one per stratum per round, strata in sorted key order.

    ``strata[i]`` is point *i*'s (sortable) stratum key. Each stratum is filled in the order of one
    ``rng.permutation`` of every point and drawn from its END, so a stratum that runs out is skipped.
    """
    if not len(chosen) <= k <= len(strata):
        refuse(f'a stratified pick of {k} needs len(chosen) <= k <= {len(strata)} points')
    picked = list(chosen)
    groups: defaultdict[Hashable, list[int]] = defaultdict(list)
    for index in rng.permutation(len(strata)):
        if index not in picked:
            groups[strata[index]].append(int(index))
    keys = sorted(groups)
    while len(picked) < k:
        for key in keys:
            if groups[key] and len(picked) < k:
                picked.append(groups[key].pop())
    return picked
