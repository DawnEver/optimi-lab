"""A study round fits its surrogate through ONE public seam, ``optimi_lab.study.rounds.fit_round``.

A consumer (motronics' GP-steering tests) observed the fit by calling and patching the private
``rounds._fit``; the seam is public so it is named, not reached into.
"""

from __future__ import annotations

from types import SimpleNamespace

from optimi_lab.policy import GP_MAX_ROWS, STEERING_MODELS
from optimi_lab.study import rounds
from optimi_lab.study.strategy import SurrogateStrategy


def _observed(monkeypatch) -> dict:
    seen: dict = {}
    monkeypatch.setattr(rounds, 'fit_surrogate', lambda _problem, _results, **kw: seen.update(kw))
    return seen


def _bound_and_searched():
    return SimpleNamespace(problem=None), SimpleNamespace(ordered_results=tuple, evidence=None)


def test_the_round_fit_is_public():
    assert 'fit_round' in rounds.__all__
    assert not hasattr(rounds, '_fit')


def test_a_gp_round_fits_its_pool_with_the_constraints_and_a_row_cap(monkeypatch):
    seen = _observed(monkeypatch)
    rounds.fit_round(SurrogateStrategy(rounds=1, candidates=4, model='gp'), *_bound_and_searched())
    assert seen == {'evidence': None, 'pool': STEERING_MODELS['gp'], 'feasibility': True, 'max_rows': GP_MAX_ROWS}


def test_a_pool_round_and_no_strategy_fit_the_default_pool(monkeypatch):
    seen = _observed(monkeypatch)
    rounds.fit_round(SurrogateStrategy(rounds=1, candidates=4), *_bound_and_searched())
    assert seen == {'evidence': None}
    seen.clear()
    rounds.fit_round(None, *_bound_and_searched())
    assert seen == {'evidence': None}
