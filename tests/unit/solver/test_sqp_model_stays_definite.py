"""SQP never ends a search with an exception: its BFGS model stays well conditioned, an unsolvable QP is a halt.

The incident (a downstream, 2026-11-12): a memetic GA's SQP refine on 82 axes with noisy finite-difference
gradients raised ``the QP Hessian is not positive definite on the null space of the active constraints``
and the whole search ended with no report. Powell damping bounds the model's curvature from BELOW only;
noisy gradient differences over short steps drove its largest eigenvalue to ~1e15 against a smallest of
~1e-3, so the KKT solve judged it singular (and roundoff turned the smallest eigenvalue negative).
"""

from __future__ import annotations

import numpy as np
import pytest

from optimi_lab.core.errors import refuse
from optimi_lab.operator import Linearization
from optimi_lab.solver import Halt, constrained, sqp


def _noisy_staircase(seed: int):
    """A nonconvex objective with noisy gradients, under ``x[i+1] - x[i] >= 0.05`` on the unit box."""
    rng = np.random.default_rng(seed)
    n = int(rng.integers(3, 12))
    q = rng.normal(size=(n, n))
    q = q + q.T
    b = rng.normal(size=n)
    rows = np.zeros((n - 1, n))
    for i in range(n - 1):
        rows[i, i], rows[i, i + 1] = 1.0, -1.0

    def evaluate(x: np.ndarray) -> Linearization:
        f = 0.5 * x @ q @ x + b @ x + np.sin(5 * x).sum()
        g = q @ x + b + 5 * np.cos(5 * x) + 1e-3 * rng.normal(size=n)
        return Linearization(x=x, f=[f], df=[g], c=rows @ x + 0.05, dc=rows)

    return evaluate, rng.uniform(0, 1, n), n


#: Seeds whose damped BFGS model, before the fix, reached condition ~1e15 and refused inside the QP.
_BLOWN_SEEDS = (73, 93, 98, 164, 165)


@pytest.mark.parametrize('seed', _BLOWN_SEEDS)
def test_sqp_on_noisy_gradients_returns_a_solution(seed: int) -> None:
    evaluate, x0, n = _noisy_staircase(seed)
    found = sqp(evaluate, x0, np.zeros(n), np.ones(n))
    assert found.halt in set(Halt)
    assert np.all(np.diff(found.x) >= 0.05 - 1e-8)


def test_a_reset_model_is_named_in_the_evidence() -> None:
    evaluate, x0, n = _noisy_staircase(164)
    found = sqp(evaluate, x0, np.zeros(n), np.ones(n))
    assert any(e.startswith('hessian_resets:') for e in found.evidence)


def test_an_unsolvable_qp_is_a_halt_with_its_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    def unsolvable(*_args, **_kwargs):
        refuse('the QP Hessian is not positive definite on the null space of the active constraints')

    monkeypatch.setattr(constrained, 'active_set_qp', unsolvable)
    evaluate, x0, n = _noisy_staircase(0)
    found = sqp(evaluate, x0, np.zeros(n), np.ones(n))
    assert found.halt is Halt.STALLED
    assert found.reason.startswith('the QP subproblem: ')
    assert np.array_equal(found.x, np.clip(x0, 0.0, 1.0))
