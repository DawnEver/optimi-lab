"""D13-D16: capability, action, result, and the async oracle with its sync adapter."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest
from lab_commons.outcomes import Cell, Evidence, Reason, Status

from optimi_lab.core.errors import Refusal
from optimi_lab.oracle import (
    Action,
    Capability,
    Infeasible,
    Information,
    Invalid,
    NotConverged,
    Oracle,
    Result,
    Unavailable,
    from_function,
)
from optimi_lab.problem import BlackBox, Fidelity, Level, Objective, Problem, Real, Space

PROBLEM = Problem(
    Space((Real('x', 0.0, 1.0), Real('y', 0.0, 1.0))),
    (Objective('f'),),
    constraints=(BlackBox('g', '<=', 1.0),),
    fidelity=Fidelity((Level('low', 1.0), Level('high', 5.0))),
)
FULL = Capability(
    outputs=('f', 'g'),
    provides=frozenset({Information.VALUES, Information.VJP}),
    fidelities=('low', 'high'),
    version='toy 1.0',
)


def test_d13_capability_names_what_is_missing_for_a_problem() -> None:
    FULL.require(PROBLEM)
    with pytest.raises(Refusal, match=r"\['g'\]"):
        Capability(outputs=('f',), fidelities=('low', 'high')).require(PROBLEM)
    with pytest.raises(Refusal, match=r"\['high'\]"):
        Capability(outputs=('f', 'g'), fidelities=('low',)).require(PROBLEM)
    assert (
        Information.VJP
        in Capability(outputs=('f',), provides=frozenset({Information.VALUES, Information.VJP})).provides
    )


def test_d13_planted_control_a_capability_without_values_or_nodes_is_refused() -> None:
    with pytest.raises(Refusal, match='VALUES'):
        Capability(outputs=('f',), provides=frozenset({Information.JVP}))
    with pytest.raises(Refusal, match='nodes'):
        Capability(outputs=('f',), nodes=0)


def test_d14_action_identity_covers_design_fidelity_derivatives_and_oracle_version() -> None:
    design = {'x': 0.25, 'y': 0.5}
    plain = Action.of(PROBLEM, FULL, design, fidelity='low')
    assert plain.identity == Action.of(PROBLEM, FULL, {'y': 0.5, 'x': 0.25}, fidelity='low').identity
    assert plain.identity != Action.of(PROBLEM, FULL, design, fidelity='high').identity
    gradient = Action.of(PROBLEM, FULL, design, fidelity='low', want=frozenset({Information.VALUES, Information.VJP}))
    assert gradient.identity != plain.identity
    other_version = Capability(
        outputs=('f', 'g'), provides=FULL.provides, fidelities=('low', 'high'), version='toy 2.0'
    )
    assert Action.of(PROBLEM, other_version, design, fidelity='low').identity != plain.identity


def test_d14_planted_control_an_action_wanting_what_the_oracle_cannot_give_is_refused() -> None:
    with pytest.raises(Refusal, match='HVP'):
        Action.of(PROBLEM, FULL, {'x': 0.1, 'y': 0.1}, want=frozenset({Information.VALUES, Information.HVP}))


def test_d15_result_is_a_cell_plus_derivatives_and_checks_its_shape() -> None:
    action = Action.of(PROBLEM, FULL, {'x': 0.1, 'y': 0.2})
    result = Result(action, Cell(Status.OK, values={'f': 1.0, 'g': 0.5}), gradient={'f': np.array([1.0, 2.0])})
    result.conform(PROBLEM)
    assert result.gradient['f'].tolist() == [1.0, 2.0]
    with pytest.raises(Refusal, match='width'):
        Result(action, Cell(Status.OK, values={'f': 1.0, 'g': 0.5}), gradient={'f': np.array([1.0])}).conform(PROBLEM)
    with pytest.raises(Refusal, match=r"\['g'\]"):
        Result(action, Cell(Status.OK, values={'f': 1.0})).conform(PROBLEM)


def test_d15_planted_control_a_failed_cell_carries_no_derivative() -> None:
    action = Action.of(PROBLEM, FULL, {'x': 0.1, 'y': 0.2})
    with pytest.raises(Refusal, match='derivative'):
        Result(action, Cell(Status.ERROR, reason=Reason('error')), gradient={'f': np.array([1.0, 2.0])})


def _toy(action: Action) -> dict:
    design = action.design
    if design['x'] > 0.9:
        msg = 'x past the magnet pocket'
        raise Invalid(msg)
    if design['x'] > 0.8:
        msg = 'g cannot hold'
        raise Infeasible(msg)
    if design['x'] > 0.7:
        msg = 'Newton stalled at 1e-3'
        raise NotConverged(msg)
    if design['x'] > 0.6:
        msg = 'no JMAG licence'
        raise Unavailable(msg)
    if design['x'] > 0.5:
        msg = 'boom'
        raise RuntimeError(msg)
    return {'f': design['x'] + design['y'], 'g': design['x']}


@pytest.mark.parametrize(
    ('x', 'status', 'kind'),
    [
        (0.1, Status.OK, None),
        (0.95, Status.REJECTED, 'invalid'),
        (0.85, Status.REJECTED, 'infeasible'),
        (0.75, Status.NONCONVERGED, 'nonconverged'),
        (0.65, Status.UNAVAILABLE, 'unavailable'),
        (0.55, Status.ERROR, 'error'),
    ],
)
def test_d16_the_sync_adapter_maps_each_exception_onto_ii3(x: float, status: Status, kind: str | None) -> None:
    oracle = from_function(_toy, FULL)
    assert isinstance(oracle, Oracle)
    result = asyncio.run(oracle(Action.of(PROBLEM, FULL, {'x': x, 'y': 0.0})))
    assert result.cell.status is status
    assert (result.cell.reason.kind if result.cell.reason else None) == kind
    assert result.cell.evidence is Evidence.OBSERVED
    assert result.cell.cost is not None and result.cell.cost.wall_s >= 0.0


def test_d16_planted_control_an_infrastructure_failure_is_never_a_rejection() -> None:
    """B0-06 in miniature: the adapter must not file a licence miss as an infeasible design."""
    oracle = from_function(_toy, FULL)
    result = asyncio.run(oracle(Action.of(PROBLEM, FULL, {'x': 0.65, 'y': 0.0})))
    assert not result.cell.status.decided
