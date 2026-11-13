"""L6 surrogate use: judging which learn-lab model a search should be guided by (:mod:`~optimi_lab.surrogate.bench`).

The models themselves are learn-lab's (:mod:`learn_lab.models.response`); this layer scores them by what a search
needs -- rank, front recovery, feasibility -- and sits under the solvers that search with them.
"""

__all__ = []
