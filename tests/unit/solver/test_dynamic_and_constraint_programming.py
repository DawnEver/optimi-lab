"""D129 dynamic programming and D130 small constraint programming, against enumeration and known counts.

* DP: a staged battery schedule (state of charge 0..4, charge -1/0/+1 per step at a time-varying
  price) is checked against brute force over all 3^T action sequences; a one-step gamble against its
  closed-form expectation.
* CP: eight queens has 92 solutions (Gauss/Nauck); a 5x5 assignment's minimum cost is checked against
  all 120 permutations, with and without the branch-and-bound lower bound.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from optimi_lab.core.errors import Refusal
from optimi_lab.solver import Halt, Rule, all_different, cp, dynamic_programming

PRICE = (3.0, 1.0, 4.0, 1.0, 5.0, 9.0)
#: Agreement with enumeration, in currency units of PRICE.
COST_TOLERANCE = 1e-12


def battery(terminal_value: float = 2.0):
    def actions(_t, s):
        return [a for a in (-1, 0, 1) if 0 <= s + a <= 4]

    def cost(t, _s, a):
        return PRICE[t] * a + 0.1 * abs(a)

    return dynamic_programming(
        len(PRICE), range(5), actions, lambda _t, s, a: s + a, cost, terminal=lambda s: -terminal_value * s
    )


def brute_force(start: int, terminal_value: float = 2.0) -> float:
    best = np.inf
    for sequence in itertools.product((-1, 0, 1), repeat=len(PRICE)):
        s, total = start, 0.0
        for t, a in enumerate(sequence):
            if not 0 <= s + a <= 4:
                break
            total, s = total + PRICE[t] * a + 0.1 * abs(a), s + a
        else:
            best = min(best, total - terminal_value * s)
    return best


@pytest.mark.parametrize('start', range(5))
def test_d129_backward_induction_matches_enumeration_of_every_schedule(start: int) -> None:
    states, actions, report = battery().plan(start)
    assert report.halt is Halt.CONVERGED
    assert abs(report.value - brute_force(start)) <= COST_TOLERANCE * 100
    assert len(actions) == len(PRICE) and states[0] == start


def test_d129_a_stochastic_transition_takes_the_expectation() -> None:
    gamble = dynamic_programming(
        1,
        ('home', 'win', 'lose'),
        lambda _t, s: ('bet', 'stay') if s == 'home' else ('stay',),
        lambda _t, s, a: ((0.3, 'win'), (0.7, 'lose')) if a == 'bet' else ((1.0, s),),
        lambda *_: 0.0,
        terminal={'home': -1.0, 'win': -10.0, 'lose': 2.0}.get,
        stochastic=True,
    )
    assert gamble.values[0]['home'] == pytest.approx(0.3 * -10 + 0.7 * 2)
    assert gamble.policy[0]['home'] == 'bet'


def test_d129_a_transition_off_the_declared_grid_is_refused_naming_it() -> None:
    with pytest.raises(Refusal, match='outside the 2 declared states'):
        dynamic_programming(1, (0, 1), lambda *_: (1,), lambda _t, s, a: s + a, lambda *_: 0.0)


def test_d129_planted_control_a_valuable_terminal_charge_changes_the_plan() -> None:
    assert battery(20.0).plan(2)[1] != battery().plan(2)[1]


def queens(n: int) -> tuple[dict, list[Rule]]:
    names = [f'q{i}' for i in range(n)]
    rules = list(all_different(*names))
    for i, j in itertools.combinations(range(n), 2):
        rules.append(Rule((names[i], names[j]), lambda a, b, d=j - i: abs(a - b) != d))
    return dict.fromkeys(names, range(n)), rules


def test_d130_eight_queens_has_ninety_two_solutions() -> None:
    domains, rules = queens(8)
    found = cp(domains, rules, count=True)
    assert found.halt is Halt.CONVERGED
    assert found.solutions == 92


COST = np.random.default_rng(0).integers(1, 20, size=(5, 5)).astype(float)


def assignment_cost(assignment: dict) -> float:
    return sum(COST[int(w[1:]), job] for w, job in assignment.items())


def optimistic(partial: dict) -> float:
    rest = [i for i in range(5) if f'w{i}' not in partial]
    return assignment_cost(partial) + sum(COST[i].min() for i in rest)


@pytest.mark.parametrize('bound', [None, optimistic])
def test_d130_minimum_cost_assignment_matches_all_permutations(bound) -> None:
    names = [f'w{i}' for i in range(5)]
    found = cp(dict.fromkeys(names, range(5)), all_different(*names), objective=assignment_cost, bound=bound)
    exact = min(sum(COST[i, p[i]] for i in range(5)) for p in itertools.permutations(range(5)))
    assert found.halt is Halt.CONVERGED
    assert found.value == exact


def test_d130_the_lower_bound_cuts_the_tree_and_an_unsatisfiable_model_is_named() -> None:
    names = [f'w{i}' for i in range(5)]
    plain = cp(dict.fromkeys(names, range(5)), all_different(*names), objective=assignment_cost)
    bounded = cp(dict.fromkeys(names, range(5)), all_different(*names), objective=assignment_cost, bound=optimistic)
    assert bounded.nodes < plain.nodes
    assert cp(dict.fromkeys(('a', 'b', 'c'), (0, 1)), all_different('a', 'b', 'c')).halt is Halt.INFEASIBLE


def test_d130_planted_control_dropping_the_diagonal_rules_counts_permutations() -> None:
    domains, rules = queens(6)
    assert cp(domains, rules[:15], count=True).solutions == 720
