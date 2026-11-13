"""Gradient verification: analytic sensitivities against central differences.

* :func:`fd_check` -- ``(index, analytic, central difference)`` rows of any ``f(x) -> (value, gradient)``.
* :func:`verify_gradients` -- every response of a :class:`~.provider.PhysicsProvider`, seen through a
  :class:`~.density.DesignChain` (tie, filter, projection, passive cells) at the nominal realization, checked
  on chosen design variables. One provider evaluation serves every response at a point. A response passes
  when ``max |analytic - fd|`` over the checked variables, per the larger of ``max |fd|`` and
  ``max |analytic|``, is within *rtol* -- a relative bar, so it means the same in any unit of the response.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from optimi_lab.core.errors import refuse
from optimi_lab.topology.density import DesignChain
from optimi_lab.topology.provider import PhysicsProvider

__all__ = ['GradientCheck', 'ResponseCheck', 'fd_check', 'verify_gradients']


def fd_check(
    f: Callable[[np.ndarray], tuple[float, np.ndarray]], x: np.ndarray, indices: Iterable[int], step: float = 1e-6
) -> list[tuple[int, float, float]]:
    """``(index, analytic, central difference)`` of ``f(x) -> (value, gradient)`` at each of *indices*."""
    x = np.asarray(x, dtype=float)
    analytic = np.asarray(f(x)[1], dtype=float).copy()
    rows = []
    for k in indices:
        plus, minus = x.copy(), x.copy()
        plus[k] += step
        minus[k] -= step
        rows.append((int(k), float(analytic[k]), (f(plus)[0] - f(minus)[0]) / (2.0 * step)))
    return rows


@dataclass(frozen=True, slots=True)
class ResponseCheck:
    """One response's rows ``(variable, analytic, fd)``, its worst relative error and whether it is within the bar."""

    rows: tuple[tuple[int, float, float], ...]
    max_rel_error: float
    passed: bool


@dataclass(frozen=True, slots=True)
class GradientCheck:
    """The check of every response at one design: the variables, the step and the bar it was run with."""

    variables: tuple[int, ...]
    step: float
    rtol: float
    responses: Mapping[str, ResponseCheck]

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.responses.values())


def verify_gradients(
    provider: PhysicsProvider,
    chain: DesignChain,
    design: np.ndarray,
    variables: Sequence[int],
    *,
    step: float = 1e-6,
    rtol: float = 1e-3,
) -> GradientCheck:
    """Check every declared response of *provider* through *chain* at *design*; see the module docstring."""
    if len(variables) == 0:
        refuse('a gradient check perturbs at least one design variable')
    if chain.domain is not provider.domain:
        refuse("the chain's filter is built on another domain than the provider's")
    nominal = chain.nominal
    memo: dict[bytes, dict[str, tuple[float, np.ndarray]]] = {}

    def responses(z: np.ndarray) -> dict[str, tuple[float, np.ndarray]]:
        key = np.asarray(z, dtype=float).tobytes()
        if key not in memo:
            physical = chain.forward(z)
            out = provider.evaluate(physical[nominal])
            missing = sorted({r.name for r in provider.responses} - set(out))
            if missing:
                refuse(f'the provider declared {missing} but did not return them; it returned {sorted(out)}')
            pulled = {}
            for name, (value, slope) in out.items():
                per = [None] * len(physical)
                per[nominal] = slope
                pulled[name] = (float(value), chain.pull_back(per))
            memo[key] = pulled
        return memo[key]

    checks = {}
    for response in provider.responses:
        rows = fd_check(lambda z, name=response.name: responses(z)[name], design, variables, step=step)
        analytic = np.array([a for _, a, _ in rows])
        fd = np.array([f for _, _, f in rows])
        scale = float(max(np.max(np.abs(fd)), np.max(np.abs(analytic))))
        error = float(np.max(np.abs(analytic - fd)) / scale) if scale > 0.0 else 0.0
        checks[response.name] = ResponseCheck(tuple(rows), error, error <= rtol)
    return GradientCheck(tuple(int(k) for k in variables), float(step), float(rtol), checks)
