"""The active-set QP settles at a vertex of near-parallel rows, alone and as SQP's subproblem.

The incident (a downstream synchronous-reluctance refine, 2026-11-14, HPC job 8011663): an SQP refine from a GA best
halted ``stalled after 0 iterations -- the active-set QP did not settle ... (cycling on a degenerate vertex)``. Its
start sat 1.7e-4 of the unit box off the lower bound of the bridge-width axis, three manufacturing floors driven
chiefly by that axis held slacks 2.67e-6, 1.78e-4 and 1.68e-4, and a fourth floor on another axis was exactly on.

Planted: that structure, linear, in four unit-cube variables, and a linear objective pushing the bridge axis down.
The first QP fills its working set with four rows -- a vertex -- of condition ~1e4. A full KKT solve returned a step
of ``eps * cond * |g|`` ~ 1.4e-10 there instead of zero, just over the stationarity test, so the loop took that junk
step without ever reading the multipliers (one of which is negative) and never settled. The reference optimum is
scipy's linprog (a REFERENCE ORACLE, tests only).
"""

from __future__ import annotations

from typing import Final

import numpy as np
import pytest
from scipy.optimize import linprog

from optimi_lab.core.errors import Refusal
from optimi_lab.operator import Linearization, active_set_qp
from optimi_lab.solver import Halt, sqp

_N = 4
_LO, _HI = np.zeros(_N), np.ones(_N)
_START = np.array([0.5631, 1.7e-4, 0.7513, 0.4605])
_GRADIENT = np.array([-66.60248409447004, 72.78281656254732, 7.243169542241652, 49.40775982095845])
_FLOORS = np.array([
    [0.02606151048296584, -0.41967207626969827, -0.007148230934144518, -0.01934387087744844],
    [0.0527640286523885, -1.573776287844323, 0.02185855582151621, -0.05146740370568923],
    [0.14106911024099344, -1.5498448197436985, -0.02497155900956887, -0.03155610850108988],
    [-1.7241997099183117, 0.0, 0.16476943998405172, -0.7931914101717658],
])
_SLACK = np.array([2.67e-6, 1.78e-4, 1.68e-4, -4.44e-16])


def _evaluate(x: np.ndarray) -> Linearization:
    step = x - _START
    return Linearization(x=x, f=[float(_GRADIENT @ step)], df=[_GRADIENT], c=_FLOORS @ step - _SLACK, dc=_FLOORS)


def _optimum() -> np.ndarray:
    reference = linprog(
        _GRADIENT, A_ub=_FLOORS, b_ub=_SLACK + _FLOORS @ _START, bounds=list(zip(_LO, _HI, strict=False))
    )
    assert reference.status == 0
    return reference.x


def test_the_qp_settles_on_the_vertex_of_near_parallel_rows() -> None:
    a_in = np.vstack([_FLOORS, np.eye(_N), -np.eye(_N)])
    b_in = np.concatenate([_SLACK, _HI - _START, _START - _LO])
    found = active_set_qp(np.eye(_N), _GRADIENT, a_in=a_in, b_in=b_in)
    assert found.feasible
    assert float(np.max(a_in @ found.x - b_in)) <= 1e-12 * float(np.max(np.abs(b_in)))
    assert float(np.min(found.multipliers_in)) >= 0.0
    residual = found.x + _GRADIENT + a_in.T @ found.multipliers_in
    assert float(np.max(np.abs(residual))) <= 1e-9 * float(np.max(np.abs(_GRADIENT)))


def test_sqp_from_the_degenerate_start_reaches_the_reference_optimum() -> None:
    found = sqp(_evaluate, _START, _LO, _HI)
    assert found.halt is Halt.CONVERGED, found.reason
    assert found.x == pytest.approx(_optimum(), abs=1e-6)


def test_planted_control_without_the_floors_the_box_vertex_settles_unchanged() -> None:
    a_in = np.vstack([np.eye(_N), -np.eye(_N)])
    found = active_set_qp(np.eye(_N), _GRADIENT, a_in=a_in, b_in=np.concatenate([_HI - _START, _START - _LO]))
    assert found.x == pytest.approx(np.clip(-_GRADIENT, _LO - _START, _HI - _START), abs=1e-12)


#: The evaluations a clean solve from this vertex makes (`test_planted_control_a_clean_solve_reaches_every_injection`):
#: a bad answer injected later would never be asked, and the case could not say no.
_INJECTED_AT: Final = (1, 2)


# The solver never emits a non-finite design (2026-11-15, HPC job 8012559: a refine from this vertex handed the
# oracle `barrier_thickness_ratio = nan`). A non-finite oracle answer -- a value or a slope -- halts STALLED by
# name; a non-finite start is refused before anything is evaluated.
@pytest.mark.parametrize('bad', [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize('part', ['f', 'df', 'c', 'dc'])
@pytest.mark.parametrize('first', _INJECTED_AT)
def test_a_non_finite_oracle_answer_halts_by_name_and_proposes_no_non_finite_design(
    part: str, bad: float, first: int
) -> None:
    asked: list[np.ndarray] = []

    def evaluate(x: np.ndarray) -> Linearization:
        asked.append(np.array(x))
        step = x - _START
        parts = {'f': [float(_GRADIENT @ step)], 'df': [_GRADIENT.copy()], 'c': _FLOORS @ step - _SLACK}
        parts['dc'] = _FLOORS.copy()
        parts = {k: np.array(v, dtype=float) for k, v in parts.items()}
        if len(asked) >= first:
            parts[part].flat[0] = bad
        return Linearization(x=x, f=parts['f'], df=parts['df'], c=parts['c'], dc=parts['dc'])

    found = sqp(evaluate, _START, _LO, _HI)
    assert all(np.all(np.isfinite(x)) for x in asked)
    assert found.halt is Halt.STALLED
    if first == 1 and part in ('df', 'dc'):
        assert f'non-finite {part}' in found.reason


def test_planted_control_a_clean_solve_reaches_every_injection() -> None:
    """Every evaluation the injection cases poison is one a clean solve makes, so each of them can fail."""
    asked: list[np.ndarray] = []

    def evaluate(x: np.ndarray) -> Linearization:
        asked.append(np.array(x))
        step = x - _START
        c = _FLOORS @ step - _SLACK
        return Linearization(x=x, f=np.array([float(_GRADIENT @ step)]), df=_GRADIENT.copy(), c=c, dc=_FLOORS.copy())

    sqp(evaluate, _START, _LO, _HI)
    assert len(asked) >= max(_INJECTED_AT)


def test_a_non_finite_start_is_refused_before_any_evaluation() -> None:
    def evaluate(x: np.ndarray) -> Linearization:
        msg = f'evaluated {x}'
        raise AssertionError(msg)

    with pytest.raises(Refusal, match=r'non-finite coordinates at \[0\]'):
        sqp(evaluate, np.array([np.nan, 0.5, 0.5, 0.5]), _LO, _HI)
