"""L4 metric: indicators over fronts and statistics over runs; dominance itself stays in ``core``."""

from optimi_lab.metric.attainment import attained
from optimi_lab.metric.indicators import (
    hypervolume,
    hypervolume_2d,
    hypervolume_contributions,
    hypervolume_mc,
    igd,
    igd_plus,
    spread,
)
from optimi_lab.metric.statistics import (
    RankTest,
    bootstrap_interval,
    rank,
    rank_sum_test,
    signed_rank_test,
    spearman,
)

__all__ = [
    'RankTest',
    'attained',
    'bootstrap_interval',
    'hypervolume',
    'hypervolume_2d',
    'hypervolume_contributions',
    'hypervolume_mc',
    'igd',
    'igd_plus',
    'rank',
    'rank_sum_test',
    'signed_rank_test',
    'spearman',
    'spread',
]
