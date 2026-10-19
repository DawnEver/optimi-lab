"""An analytic oracle whose outputs are constant-coefficient polynomials of the natural design vector.

    value_o(x) = 1/2 x.Q_o x + a_o.x + k_o,   x = (design[name] for name in space.names)

It answers VALUES always and, when an action wants them, the coefficients its capability declares
(:data:`~optimi_lab.oracle.COEFFICIENTS_FOR`): ``LINEAR_COEFFICIENTS`` when no output has a ``Q``,
``QUADRATIC_COEFFICIENTS`` (each ``a`` and ``Q``, zero where undeclared) otherwise -- so it never
declares a linear output it does not have.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from lab_commons.outcomes import Cell, Status

from optimi_lab.core.errors import refuse
from optimi_lab.oracle.action import Action
from optimi_lab.oracle.capability import Capability, Information
from optimi_lab.oracle.oracle import Oracle, from_function
from optimi_lab.oracle.result import Result
from optimi_lab.problem import Space

__all__ = ['analytic']


def analytic(
    space: Space,
    *,
    linear: Mapping[str, Sequence[float]],
    quadratic: Mapping[str, object] | None = None,
    constant: Mapping[str, float] | None = None,
    version: str = '',
) -> Oracle:
    """The oracle of the polynomials above; *linear* names every output, *quadratic*/*constant* default to 0."""
    names, n = space.names, len(space.names)
    a = {o: np.asarray(v, dtype=float).reshape(-1) for o, v in linear.items()}
    q = {o: np.asarray(m, dtype=float) for o, m in (quadratic or {}).items()}
    k = {o: float(v) for o, v in (constant or {}).items()}
    strangers = sorted((set(q) | set(k)) - set(a))
    if strangers:
        refuse(f'outputs {strangers} have no linear coefficients; name every output in linear=')
    bad = sorted(o for o, v in a.items() if v.shape != (n,)) + sorted(o for o, m in q.items() if m.shape != (n, n))
    if bad:
        refuse(f'coefficients of {bad} do not fit the {n} variable(s) {list(names)}')
    q = {o: q.get(o, np.zeros((n, n))) for o in a}
    supplies = Information.QUADRATIC_COEFFICIENTS if quadratic else Information.LINEAR_COEFFICIENTS
    capability = Capability(outputs=tuple(a), provides=frozenset({Information.VALUES, supplies}), version=version)

    def evaluate(action: Action) -> Result:
        x = np.array([float(action.design[name]) for name in names])  # type: ignore[arg-type]
        values = {o: 0.5 * float(x @ q[o] @ x) + float(a[o] @ x) + k.get(o, 0.0) for o in a}
        extras: dict[str, object] = {}
        if supplies in action.want:
            extras['linear_coefficients'] = a
            if supplies is Information.QUADRATIC_COEFFICIENTS:
                extras['quadratic_coefficients'] = q
        return Result(action, Cell(Status.OK, values=values), **extras)  # type: ignore[arg-type]

    return from_function(evaluate, capability)
