"""Population and model-based solvers search a conditional, categorical space through a declared embedding.

THE BENCHMARK. A switch ``kind in (flat, bowl, ramp)`` (unembedded, unordered), an always-active
``x in [0, 1]`` and a radius ``r in [0, 1]`` that exists only for ``bowl``:

* ``flat``: ``f = 0.5 + (x - 0.2)^2`` (minimum 0.5),
* ``bowl``: ``f = (x - 0.7)^2 + (r - 0.3)^2`` (minimum 0 at ``x = 0.7, r = 0.3``),
* ``ramp``: ``f = 1 + x`` (minimum 1).

The global minimum 0 is closed form and sits behind the switch, so a solver that cannot move the
category or that reads the switched-off radius as a value lands on a wrong branch. The oracle checks
every design it is asked for: an inactive ``r``, a missing active ``r`` or an undeclared category is
recorded, and the record must stay empty -- the oracle never sees an illegal point.

``f`` is dimensionless; :data:`GAP` is in its unit.
"""

import sys
from pathlib import Path

import pytest

from optimi_lab.core import Refusal, SampleSpec
from optimi_lab.oracle import Action, Capability, from_function
from optimi_lab.problem import (
    Categorical,
    Embedding,
    Encoding,
    Fidelity,
    Inactive,
    Level,
    Objective,
    Problem,
    Real,
    Space,
    When,
)
from optimi_lab.solver import (
    Requires,
    bo,
    cmaes,
    ehvi,
    ga,
    match,
    mfbo,
    mode,
    nelder_mead,
    nsga2,
    parego,
    prescreened,
    spea2,
    turbo,
)
from optimi_lab.solver.requires import Ineligible, NeedsTransform

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'benchmarks'))
from catalogue import drive

SPACE = Space((
    Categorical('kind', ('flat', 'bowl', 'ramp')),
    Real('x', 0.0, 1.0),
    Real('r', 0.0, 1.0, active_if=When('kind', ('bowl',))),
))
SINGLE = Problem(SPACE, (Objective('f'),))
PAIR = Problem(SPACE, (Objective('f'), Objective('x_out')))
#: The largest best-found ``f`` a solver may leave (dimensionless); the branch minima are 0, 0.5 and 1.
GAP = 0.05
EMBEDDINGS = [(Encoding.ONE_HOT, Inactive.NEUTRAL), (Encoding.ORDINAL, Inactive.INDICATOR)]
#: Every solver the registry declares able to read a conditional space through an embedding.
CONDITIONAL = (
    'nsga2',
    'nsga3',
    'unsga3',
    'rnsga3',
    'spea2',
    'sms_emoa',
    'rvea',
    'mode',
    'mopso',
    'moead',
    'ga',
    'cmaes',
    'ipop_cmaes',
    'map_elites',
    'bo',
    'parego',
    'ehvi',
    'nehvi',
    'mfbo',
    'turbo',
    'prescreened',
)


def _f(design: dict[str, object]) -> float:
    x = float(design['x'])
    if design['kind'] == 'bowl':
        return (x - 0.7) ** 2 + (float(design['r']) - 0.3) ** 2
    return 0.5 + (x - 0.2) ** 2 if design['kind'] == 'flat' else 1.0 + x


def _oracle(problem: Problem, fidelities: tuple[str, ...] = ()):
    illegal: list[dict[str, object]] = []

    def evaluate(action: Action) -> dict[str, float]:
        design = dict(action.design)
        legal = design['kind'] in ('flat', 'bowl', 'ramp') and ('r' in design) == (design['kind'] == 'bowl')
        if not legal or SPACE.canonical(design) != design:
            illegal.append(design)
        return {k: v for k, v in {'f': _f(design), 'x_out': float(design['x'])}.items() if k in problem.outputs}

    return from_function(evaluate, Capability(outputs=problem.outputs, fidelities=fidelities)), illegal


def _best(results) -> tuple[float, str]:
    told = [r for r in results if r.cell.values]
    best = min(told, key=lambda r: r.cell.values['f'])
    return best.cell.values['f'], best.action.design['kind']


@pytest.mark.parametrize(('categorical', 'inactive'), EMBEDDINGS)
@pytest.mark.parametrize(
    ('factory', 'settings', 'evaluations'),
    [
        (ga, {'spec': SampleSpec(n_samples=20, seed=1)}, 400),
        (cmaes, {}, 400),
        (prescreened, {'spec': SampleSpec(n_samples=12, seed=1)}, 120),
        (bo, {}, 40),
        (turbo, {}, 60),
    ],
)
def test_the_solver_finds_the_optimum_behind_the_switch(factory, settings, evaluations, categorical, inactive):
    oracle, illegal = _oracle(SINGLE)
    embedding = Embedding(SPACE, categorical=categorical, inactive=inactive)
    solver = factory(SINGLE, oracle.capability, seed=1, embedding=embedding, **settings)
    value, kind = _best(drive(solver, oracle, evaluations))
    assert illegal == []
    assert kind == 'bowl'
    assert value <= GAP


@pytest.mark.parametrize(
    ('factory', 'settings'),
    [
        (nsga2, {'spec': SampleSpec(n_samples=12, seed=1)}),
        (spea2, {'spec': SampleSpec(n_samples=12, seed=1)}),
        (mode, {'spec': SampleSpec(n_samples=12, seed=1)}),
        (prescreened, {'spec': SampleSpec(n_samples=12, seed=1)}),
        (parego, {}),
        (ehvi, {}),
    ],
)
@pytest.mark.parametrize(('categorical', 'inactive'), EMBEDDINGS)
def test_every_design_a_pareto_solver_asks_is_legal(factory, settings, categorical, inactive):
    oracle, illegal = _oracle(PAIR)
    embedding = Embedding(SPACE, categorical=categorical, inactive=inactive)
    solver = factory(PAIR, oracle.capability, seed=1, embedding=embedding, **settings)
    results = drive(solver, oracle, 40)
    assert len(results) > 12  # past the first generation / initial design: proposals, not just samples
    assert illegal == []


def test_multi_fidelity_bo_asks_legal_designs_through_the_embedding():
    fidelity = Fidelity((Level('coarse', 1.0), Level('fine', 10.0)))
    problem = Problem(SPACE, (Objective('f'),), fidelity=fidelity)
    oracle, illegal = _oracle(problem, ('coarse', 'fine'))
    solver = mfbo(problem, oracle.capability, seed=1, embedding=Embedding(SPACE))
    assert len(drive(solver, oracle, 60)) > 0
    assert illegal == []


@pytest.mark.parametrize('factory', [ga, cmaes, bo, turbo, nsga2])
def test_a_conditional_space_without_a_declared_embedding_is_refused_naming_it(factory):
    with pytest.raises(Refusal, match='embedding='):
        factory(SINGLE, _oracle(SINGLE)[0].capability)


def test_an_embedding_declared_over_another_space_is_refused():
    other = Embedding(Space((Real('x', 0.0, 1.0),)))
    with pytest.raises(Refusal, match='another space'):
        ga(SINGLE, _oracle(SINGLE)[0].capability, embedding=other)


@pytest.mark.parametrize('name', CONDITIONAL)
def test_the_registry_declares_the_embedding(name):
    problem = PAIR if Requires.of(name).objectives.value != 'single' else SINGLE
    if Requires.of(name).fidelity:
        problem = Problem(SPACE, (Objective('f'),), fidelity=Fidelity((Level('coarse', 1.0), Level('fine', 10.0))))
    assert match(Requires.of(name), problem, _oracle(problem)[0].capability) == NeedsTransform(('embed',), 1.0)


@pytest.mark.parametrize('name', ['nelder_mead', 'mads', 'bobyqa', 'model_trust_region'])
def test_a_solver_without_an_embedding_is_ineligible_on_a_conditional_space(name):
    verdict = match(Requires.of(name), SINGLE, _oracle(SINGLE)[0].capability)
    assert isinstance(verdict, Ineligible)
    assert 'conditional' in verdict.reason


def test_a_local_solver_still_refuses_a_conditional_space():
    with pytest.raises(Refusal):
        nelder_mead(SINGLE, _oracle(SINGLE)[0].capability)
