"""D118 robust evaluation: the expectation of a response under a problem's :class:`Uncertainty`.

A *response* maps a sample (parameter name -> ``(n,)`` physical values) to ``(n,)`` outputs, so one
vectorised call evaluates a whole batch. Points come from one of three :class:`Rule` s:

* ``mc`` -- independent draws; the standard error is the CLT one;
* ``sobol`` / ``halton`` -- RANDOMISED QMC (:mod:`optimi_lab.core.qmc`, scrambled): *replicates*
  independent scramblings of ``n / replicates`` points each, pooled by
  :func:`~optimi_lab.operator.estimators.replicated`. A single QMC set has no error bar, so fewer
  than two replicates is refused.

Unit-cube points are kept strictly inside ``(0, 1)`` (a scrambled point may sit on 0, where a normal
inverse CDF is infinite) and mapped by :meth:`Uncertainty.at_unit`.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum

import numpy as np

from optimi_lab.core.errors import coerce_enum, refuse, require_positive
from optimi_lab.core.qmc import halton, sobol
from optimi_lab.operator.estimators import Estimate, mean_estimate, replicated
from optimi_lab.problem import Uncertainty

__all__ = ['Response', 'Rule', 'draw', 'robust_mean']

Response = Callable[[dict[str, np.ndarray]], np.ndarray]

#: Unit-cube points are clipped to ``[_EDGE, 1 - _EDGE]``: the 53-bit neighbours of 0 and 1.
_EDGE = 2.0**-53


class Rule(StrEnum):
    """Where the points of a robust evaluation come from."""

    MC = 'mc'
    SOBOL = 'sobol'
    HALTON = 'halton'


def _generator(rng: object) -> np.random.Generator:
    if not isinstance(rng, np.random.Generator):
        refuse(f'robust evaluation needs an explicit numpy Generator, got {type(rng).__name__}')
    return rng


def draw(uncertainty: Uncertainty, n: int, rng: np.random.Generator, *, rule: Rule = Rule.MC) -> dict[str, np.ndarray]:
    """*n* points of *uncertainty* by *rule*, in physical values (one scrambling for a QMC rule)."""
    rng, rule = _generator(rng), coerce_enum(Rule, rule, 'rule')
    require_positive(n, 'n')
    width = len(uncertainty.parameters)
    if rule is Rule.MC:
        unit = rng.random((n, width))
    else:
        unit = (sobol if rule is Rule.SOBOL else halton)(n, width, rng=rng)
    return uncertainty.at_unit(np.clip(unit, _EDGE, 1.0 - _EDGE))


def robust_mean(
    response: Response,
    uncertainty: Uncertainty,
    n: int,
    rng: np.random.Generator,
    *,
    rule: Rule = Rule.MC,
    replicates: int = 16,
) -> Estimate:
    """``E[response]`` over *n* evaluations, with its standard error; see the module docstring."""
    rule = coerce_enum(Rule, rule, 'rule')
    if rule is Rule.MC:
        return mean_estimate(response(draw(uncertainty, n, rng, rule=rule)))
    if replicates < 2 or n % replicates:
        refuse(f'randomised QMC needs at least two replicates dividing n={n}, got replicates={replicates}')
    per = n // replicates
    return replicated(
        Estimate(float(np.mean(response(draw(uncertainty, per, rng, rule=rule)))), 0.0, per) for _ in range(replicates)
    )
