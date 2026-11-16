"""``minimise``: reaches a known optimum over a mixed space, refuses what a method cannot read, never overspends.

Each guard has its planted negative beside it: an unsupported kind raises for EVERY (method, kind) cell
the table leaves out and runs for every cell it lists, so the table cannot lie in either direction; the
budget is counted by the objective itself, not read back from the run.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import pytest
from learn_lab.space import Choice, Integer, Real, Space

from optimi_lab.core import Refusal
from optimi_lab.tuning import TUNING_METHODS, Kind, minimise

_PENALTY = {'a': 1.0, 'b': 0.0, 'c': 2.0}
MIXED = {'rate': Real(1e-4, 1.0, log=True), 'depth': Integer(0, 10), 'kind': Choice(('a', 'b', 'c'))}
ONE_OF_EACH = {Kind.REAL: Real(-2.0, 2.0), Kind.INTEGER: Integer(0, 6), Kind.CHOICE: Choice(('x', 'y'))}
LEVELS = {'cheap': 1.0, 'full': 4.0}


def mixed_quadratic(settings: Mapping[str, object]) -> float:
    """0 exactly at rate=1e-2, depth=3, kind='b'."""
    rate, depth = float(settings['rate']), int(settings['depth'])  # type: ignore[arg-type]
    return (math.log10(rate) + 2.0) ** 2 + (depth - 3) ** 2 + _PENALTY[str(settings['kind'])]


def any_quadratic(settings: Mapping[str, object]) -> float:
    """A bowl over whichever of ONE_OF_EACH's names are present: 0 at r=0.5, i=2, c='y'."""
    total = 0.0
    if 'real' in settings:
        total += (float(settings['real']) - 0.5) ** 2  # type: ignore[arg-type]
    if 'integer' in settings:
        total += (int(settings['integer']) - 2) ** 2  # type: ignore[call-overload]
    if 'choice' in settings:
        total += 0.0 if settings['choice'] == 'y' else 1.0
    return total


class Counted:
    """An objective that counts its own calls -- the budget is checked against THIS, not the run's ledger."""

    def __init__(self, function) -> None:  # type: ignore[no-untyped-def]
        self.function, self.calls = function, 0

    def __call__(self, settings: Mapping[str, object], *fidelity: str) -> float:
        self.calls += 1
        return self.function(settings)


def _levels(name: str) -> dict[str, float] | None:
    return LEVELS if TUNING_METHODS[name].fidelity else None


@pytest.mark.parametrize(('method', 'budget'), [('bayesian', 60), ('evolutionary', 300), ('cmaes', 300)])
def test_a_mixed_quadratic_reaches_its_optimum_within_the_budget(method: str, budget: int) -> None:
    tuning = minimise(mixed_quadratic, MIXED, method=method, budget=budget, seed=1)
    assert tuning.best is not None
    assert tuning.best.settings['depth'] == 3
    assert tuning.best.settings['kind'] == 'b'
    assert tuning.best.score == pytest.approx(0.0, abs=0.05)  # score is in log10-decades squared


@pytest.mark.parametrize('method', ['nelder_mead', 'bobyqa', 'model_trust_region'])
def test_a_real_quadratic_reaches_its_optimum_for_the_real_only_methods(method: str) -> None:
    tuning = minimise(any_quadratic, {'real': Real(-2.0, 2.0)}, method=method, budget=40, seed=1)
    assert tuning.best is not None
    assert tuning.best.score == pytest.approx(0.0, abs=1e-3)


def test_the_planted_far_optimum_is_not_reported_as_reached() -> None:
    """Control for the optimum checks: a budget too small to search fails the same assertion."""
    tuning = minimise(mixed_quadratic, MIXED, method='bayesian', budget=3, seed=1)
    assert tuning.best is not None
    assert tuning.best.score > 0.05


CELLS = [(name, kind) for name in TUNING_METHODS for kind in Kind]


@pytest.mark.parametrize(('method', 'kind'), CELLS, ids=[f'{m}-{k}' for m, k in CELLS])
def test_a_method_runs_exactly_the_kinds_it_declares(method: str, kind: Kind) -> None:
    space = {kind.value: ONE_OF_EACH[kind]}
    objective = Counted(any_quadratic)
    if kind in TUNING_METHODS[method].kinds:
        tuning = minimise(objective, space, method=method, budget=12, seed=0, fidelities=_levels(method))
        assert tuning.trials, f'{method} ran no trial on a {kind} dimension'
        assert all(kind.value in trial.settings for trial in tuning.trials)
    else:
        with pytest.raises(Refusal, match=f'{method} cannot search'):
            minimise(objective, space, method=method, budget=12, seed=0, fidelities=_levels(method))
        assert objective.calls == 0


def test_an_unsupported_dimension_is_refused_inside_a_mixed_space_too() -> None:
    with pytest.raises(Refusal, match=r"nelder_mead cannot search \{'depth': 'integer', 'kind': 'choice'\}"):
        minimise(mixed_quadratic, MIXED, method='nelder_mead', budget=10)


@pytest.mark.parametrize('method', sorted(TUNING_METHODS))
@pytest.mark.parametrize('budget', [3, 17])
def test_the_budget_is_never_exceeded_and_every_call_is_a_trial(method: str, budget: int) -> None:
    space = {k.value: d for k, d in ONE_OF_EACH.items() if k in TUNING_METHODS[method].kinds}
    objective = Counted(any_quadratic)
    tuning = minimise(objective, space, method=method, budget=budget, seed=2, fidelities=_levels(method))
    assert 0 < objective.calls <= budget
    assert len(tuning.trials) == objective.calls


def test_the_budget_counter_sees_a_planted_overspend() -> None:
    """Control: the counter that guards the budget does count every call."""
    objective = Counted(any_quadratic)
    for _ in range(5):
        objective({'real': 0.0})
    assert objective.calls == 5


def test_a_failed_fit_is_a_trial_scored_inf_and_never_the_best() -> None:
    def fails_on_a(settings: Mapping[str, object]) -> float:
        return math.inf if settings['kind'] == 'a' else mixed_quadratic(settings)

    tuning = minimise(fails_on_a, MIXED, method='evolutionary', budget=30, seed=3)
    failed = [t for t in tuning.trials if t.settings['kind'] == 'a']
    assert failed and all(t.score == math.inf for t in failed)
    assert tuning.best is not None and tuning.best.settings['kind'] != 'a'


def test_fidelity_bo_reports_its_best_at_the_truth_level() -> None:
    seen: set[str] = set()

    def objective(settings: Mapping[str, object], fidelity: str) -> float:
        seen.add(fidelity)
        return mixed_quadratic(settings) + (0.3 if fidelity == 'cheap' else 0.0)

    tuning = minimise(objective, MIXED, method='fidelity_bo', budget=80, seed=1, fidelities=LEVELS)
    assert seen == {'cheap', 'full'}
    assert tuning.best is not None and tuning.best.fidelity == 'full'
    assert tuning.best.score < 1.0


def test_fidelity_levels_are_refused_where_they_would_be_ignored_and_required_where_read() -> None:
    with pytest.raises(Refusal, match='bayesian evaluates one fidelity'):
        minimise(mixed_quadratic, MIXED, budget=5, fidelities=LEVELS)
    with pytest.raises(Refusal, match='fidelity_bo promotes through fidelity levels'):
        minimise(mixed_quadratic, MIXED, method='fidelity_bo', budget=5)


def test_a_stranger_method_and_a_stranger_dimension_are_refused() -> None:
    with pytest.raises(Refusal, match='no tuning method'):
        minimise(mixed_quadratic, MIXED, method='grid', budget=5)
    with pytest.raises(Refusal, match='a tuning dimension is one of'):
        minimise(mixed_quadratic, {'x': (0.0, 1.0)}, budget=5)  # type: ignore[dict-item]


def test_a_seed_drawn_for_the_caller_is_reported_and_replays() -> None:
    first = minimise(mixed_quadratic, MIXED, method='evolutionary', budget=8)
    again = minimise(mixed_quadratic, MIXED, method='evolutionary', budget=8, seed=first.seed)
    assert [t.settings for t in again.trials] == [t.settings for t in first.trials]


def test_a_search_that_exhausts_a_finite_space_stops_instead_of_re_asking_for_free() -> None:
    """A re-asked decided design costs no budget, so without the stop this run never ends."""
    objective = Counted(any_quadratic)
    tuning = minimise(objective, {'choice': Choice(('x', 'y'))}, method='bayesian', budget=50, seed=0)
    assert objective.calls == 2
    assert tuning.best is not None and tuning.best.settings['choice'] == 'y'


def test_a_declared_learn_lab_space_is_searched_as_declared() -> None:
    """The space a model family declares (learn_lab.space.Space) is the search's input, with no translation."""
    declared = Space({'rate': Real(1e-4, 1.0, log=True), 'depth': Integer(0, 10), 'kind': Choice(('a', 'b', 'c'))})
    tuning = minimise(mixed_quadratic, declared, method='bayesian', budget=60, seed=1)
    assert tuning.best is not None
    assert declared.holds(tuning.best.settings)
