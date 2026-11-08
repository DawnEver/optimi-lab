"""``minimise``: a hyper-parameter search over a tuning space, driven by the solvers this package ships.

The caller supplies only an OBJECTIVE -- ``objective(settings) -> float`` to minimise, ``inf`` when the
fit failed -- and, for a multi-fidelity method, the fidelity levels it can be evaluated at
(``objective(settings, fidelity)``). Everything else is the existing machinery: the space becomes a
:class:`~optimi_lab.problem.Problem` with one objective ``score``, the objective an oracle
(:func:`~optimi_lab.oracle.from_function`), the method a registered solver
(:func:`~optimi_lab.solver.build_solver`), and the search one :func:`~optimi_lab.run.run` under an
evaluation :class:`~optimi_lab.run.Budget`. A non-finite score is filed ``NONCONVERGED``, so every
solver drops it exactly as it drops any failed point.

:data:`TUNING_METHODS` is the one table of what a method reads. A method handed a dimension kind it does not
declare is refused before anything is evaluated (UNSUPPORTED-RAISES), naming the kind -- never run with
the dimension dropped. The local continuous searches (Nelder-Mead, BOBYQA, MADS, the model trust region)
read real dimensions only (``solver.sequential`` refuses a discrete variable), and the successive-halving
schedules (hyperband, asha) take no embedding, so no OneOf.

THE BUDGET COUNTS OBJECTIVE CALLS. The run charges every dispatched evaluation and asks the solver for no
more than remain; a design asked twice is evaluated once, and a search whose whole ask repeats what it
already asked has exhausted its space and stops. :attr:`Tuning.trials` is every call, in order.
"""

from __future__ import annotations

import math
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final

import numpy as np
from lab_commons.outcomes import Record

from optimi_lab.core.errors import refuse
from optimi_lab.oracle import Action, Capability, NotConverged, Result, from_function
from optimi_lab.problem import Fidelity, Level, Objective, Problem
from optimi_lab.run import Budget, run
from optimi_lab.solver import Solver, build_solver
from optimi_lab.tuning.space import Dimension, Kind, kind_of, problem_space

__all__ = ['TUNING_METHODS', 'Method', 'Trial', 'Tuning', 'minimise']

_SCORE: Final = 'score'


@dataclass(frozen=True, slots=True)
class Method:
    """A tuning method: the registered *solver* it builds, the dimension *kinds* it reads, whether it needs *fidelity*."""

    solver: str
    kinds: frozenset[Kind]
    fidelity: bool = False


_ALL: Final = frozenset(Kind)
_REAL: Final = frozenset({Kind.REAL})
_NUMERIC: Final = frozenset({Kind.REAL, Kind.INTEGER})

#: Every tuning method, by name -- the one table of what each reads.
TUNING_METHODS: Final[Mapping[str, Method]] = MappingProxyType({
    'bayesian': Method('bo', _ALL),
    'turbo': Method('turbo', _ALL),
    'cmaes': Method('cmaes', _ALL),
    'ipop_cmaes': Method('ipop_cmaes', _ALL),
    'evolutionary': Method('ga', _ALL),
    'mads': Method('mads', _REAL),
    'nelder_mead': Method('nelder_mead', _REAL),
    'bobyqa': Method('bobyqa', _REAL),
    'model_trust_region': Method('model_trust_region', _REAL),
    'fidelity_bo': Method('mfbo', _ALL, fidelity=True),
    'hyperband': Method('hyperband', _NUMERIC, fidelity=True),
    'asha': Method('asha', _NUMERIC, fidelity=True),
})


@dataclass(frozen=True, slots=True)
class Trial:
    """One objective call: its *settings*, its *score* (``inf`` if it failed), the *fidelity* it ran at."""

    settings: Mapping[str, object]
    score: float
    fidelity: str | None = None


@dataclass(frozen=True, slots=True)
class Tuning:
    """Every :class:`Trial` in call order, the *best*, and the *method* and *seed* that produced them.

    *best* is the lowest finite score at the truth fidelity (the only one without levels); None when
    no such trial succeeded.
    """

    trials: tuple[Trial, ...]
    best: Trial | None
    method: str
    seed: int


def minimise(
    objective: Callable[..., float],
    space: Mapping[str, Dimension],
    *,
    method: str = 'bayesian',
    budget: int,
    seed: int | None = None,
    initial: int | None = None,
    fidelities: Mapping[str, float] | None = None,
) -> Tuning:
    """Minimise *objective* over *space* with *method* in at most *budget* objective calls.

    *initial* is the size of the first batch (a population, a model's initial design, CMA-ES's
    population); None takes ``max(2, min(budget, 2 (d + 1)))``. *fidelities* maps each level name to its cost,
    cheapest first and the last the truth; a fidelity method needs it and calls ``objective(settings,
    level)``, every other method refuses it. *seed* None draws one, reported on the result.
    """
    chosen = _method(method, space, fidelities)
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 1:
        refuse(f'a tuning budget is a count of objective calls >= 1, got {budget!r}')
    seed = int(np.random.SeedSequence().entropy % 2**31) if seed is None else int(seed)
    levels = None if fidelities is None else Fidelity(tuple(Level(n, float(c)) for n, c in fidelities.items()))
    problem = Problem(problem_space(space), (Objective(_SCORE),), fidelity=levels)
    capability = Capability((_SCORE,), fidelities=tuple(fidelities or ()))
    trials: list[Trial] = []

    def evaluate(action: Action) -> Mapping[str, float]:
        settings = MappingProxyType(dict(action.design))
        score = float(objective(settings) if levels is None else objective(settings, action.fidelity))
        trials.append(Trial(settings, score if math.isfinite(score) else math.inf, action.fidelity))
        if not math.isfinite(score):
            message = f'the objective scored {score!r}'
            raise NotConverged(message)
        return {_SCORE: score}

    population = max(2, min(budget, 2 * (len(space) + 1))) if initial is None else initial
    solver = build_solver(
        chosen.solver, problem, capability, seed=seed, population=population, initial=None, generations=budget
    )
    with tempfile.TemporaryDirectory() as root:
        record = Record(Path(root) / 'tuning.jsonl')
        run(problem, from_function(evaluate, capability), _Novel(solver), record, Budget(evaluations=budget))
    truth = None if levels is None else levels.truth.name
    finished = [t for t in trials if math.isfinite(t.score) and t.fidelity == truth]
    return Tuning(tuple(trials), min(finished, key=lambda t: t.score, default=None), method, seed)


def _method(name: str, space: Mapping[str, Dimension], fidelities: Mapping[str, float] | None) -> Method:
    """The :data:`TUNING_METHODS` row *name*, once it reads every dimension of *space* and its fidelity need is met."""
    if name not in TUNING_METHODS:
        refuse(f'no tuning method {name!r}; the methods are {sorted(TUNING_METHODS)}')
    method = TUNING_METHODS[name]
    unread = {n: kind_of(d).value for n, d in space.items() if kind_of(d) not in method.kinds}
    if unread:
        refuse(
            f'{name} cannot search {unread}: it reads {sorted(k.value for k in method.kinds)} dimensions only; '
            f'pick a method that reads them, e.g. bayesian'
        )
    fidelity_methods = sorted(k for k, m in TUNING_METHODS.items() if m.fidelity)
    if method.fidelity and not fidelities:
        refuse(f'{name} promotes through fidelity levels; pass fidelities= (level -> cost, cheapest first)')
    if not method.fidelity and fidelities:
        refuse(f'{name} evaluates one fidelity; fidelities= is read only by {fidelity_methods}')
    return method


class _Novel:
    """*solver*, concluded once a whole ask repeats designs already asked.

    The run answers a decided design from its archive without charging the budget, so a solver that
    has exhausted a finite space (a few integers, a few choices) would re-ask forever at no cost.
    """

    def __init__(self, solver: Solver) -> None:
        self._solver, self._asked, self._stale = solver, set[str](), False

    def ask(self, capacity: int) -> Sequence[Action]:
        actions = self._solver.ask(capacity)
        keys = {action.identity.key for action in actions}
        self._stale = bool(keys) and keys <= self._asked
        self._asked |= keys
        return actions

    def tell(self, results: Sequence[Result]) -> None:
        self._solver.tell(results)

    @property
    def done(self) -> bool:
        return self._stale or self._solver.done
