"""Routing -- which evaluator, if any, spends its budget on each point.

The optimizer decides where to SEARCH; a :data:`Router` decides where to SPEND. It maps a batch of
points to one integer per point: ``k`` sends the point to ``evaluators[k]`` (by convention cheapest
first), :data:`SCREENED` sends it nowhere. What the router consults to decide -- a surrogate, a
rule, a model -- is the router's own business; this module knows only the integers.

:func:`routed` turns a router and its evaluators into ONE :class:`~optimi_lab.core.protocols.
Evaluator`, so :func:`~optimi_lab.core.optimize.optimize` and every proposer run unchanged. A
screened point carries :attr:`~optimi_lab.core.outcomes.Outcome.SCREENED` and no number: it is not
a bad design, it is a design nobody paid to evaluate, and ``complete()`` leaves it out of every
ranking exactly as it leaves out an error.
"""

from collections.abc import Callable, Sequence

import numpy as np

from optimi_lab.core.errors import refuse
from optimi_lab.core.outcomes import Evaluation, Outcome
from optimi_lab.core.protocols import Evaluator
from optimi_lab.core.space import ObjectiveSet

__all__ = ['SCREENED', 'Router', 'routed']

#: The route that evaluates nothing.
SCREENED = -1

#: Points ``(n_points, n_var)`` in, one integer route per point out.
Router = Callable[[np.ndarray], np.ndarray]


def routed(objectives: ObjectiveSet, evaluators: Sequence[Evaluator], router: Router) -> Evaluator:
    """Return an evaluator that routes each batch, point by point, through ``router``.

    Raises:
        Refusal: If there is no evaluator; at call time, if the router does not return one integer
            route per point inside ``[SCREENED, len(evaluators) - 1]``, or if an evaluator returns
            anything other than an evaluation of exactly the points it was handed.

    """
    evaluators = tuple(evaluators)
    if not evaluators:
        refuse('routed() needs at least one evaluator; a router with nowhere to send a point screens everything')

    def evaluate(inputs: np.ndarray) -> Evaluation:
        inputs = np.asarray(inputs, dtype=float)
        route = _require_route(router(inputs), len(inputs), len(evaluators))
        values = np.full((len(inputs), objectives.n_obj), np.nan)
        outcomes = np.full(values.shape, Outcome.SCREENED, dtype=object)
        for index, evaluator in enumerate(evaluators):
            mask = route == index
            if not mask.any():
                continue
            part = evaluator(inputs[mask])
            if not isinstance(part, Evaluation) or not np.array_equal(part.inputs, inputs[mask]):
                refuse(
                    f'evaluator {index} returned points other than the {int(mask.sum())} it was routed; '
                    f'its rows are written back by position, so a reordered batch misnames every value'
                )
            values[mask] = part.values
            outcomes[mask] = part.outcomes
        return Evaluation(inputs=inputs, values=values, outcomes=outcomes)

    return evaluate


def _require_route(route: object, n_points: int, n_evaluators: int) -> np.ndarray:
    route = np.asarray(route)
    if route.shape != (n_points,):
        refuse(f'the router must return one route per point, shape ({n_points},), got {route.shape}')
    if not np.issubdtype(route.dtype, np.integer):
        refuse(f'a route is an integer evaluator index or SCREENED ({SCREENED}), got dtype {route.dtype}')
    if route.size and (route.min() < SCREENED or route.max() >= n_evaluators):
        refuse(
            f'a route must lie between {SCREENED} and {n_evaluators - 1} '
            f'(SCREENED or an index into {n_evaluators} evaluator(s)), got {sorted(set(route.tolist()))}'
        )
    return route
