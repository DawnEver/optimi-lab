"""The benchmark problems and the two metrics, against their closed forms.

WHY THIS FILE EXISTS, AND WHY IT IS NEW. `optimi_lab.benchmarks` shipped 182 lines and six names
on the package's top-level `__all__` with NOT ONE test reaching any of them -- measured 2026-09-19:
`grep -rw zdt1 tests/` returned nothing, and so did `zdt2`, `igd`, `hypervolume_2d`, `Problem` and
`solve`. The evidence for those six names existed, but it lived in the CONSUMER
(`motronics/tests/unit/pareto/test_nsga2_benchmark.py`), one repository away from the code it
judges. An `__all__` entry with no test behind it is a declaration that lies: it asserts a public
surface the package does not check. Owning a fact means holding its evidence, so the evidence
came home and the hand-built cases below are ported from that consumer.

THE BANDS ARE PINNED AS A RATIO, NOT AS A VALUE AT ONE OPERATING POINT. A threshold read off
`seed=0` pins the luckiest draw: measured over six seeds, the ZDT2 hypervolume ratio spans
0.312 to 0.685 and `seed=0` is the BEST of the six. So the guard that carries the weight here is
:func:`test_a_longer_budget_converges_closer`, which asserts the property no seed can flatter --
sixty generations beat five, on both metrics, every time -- and the absolute readings are recorded
alongside it as characterization, each quoting the worst seed measured rather than the best.
"""

import numpy as np
import pytest

from optimi_lab import Problem, Record, hypervolume_2d, igd, solve, zdt1, zdt2
from optimi_lab.intelligent_algorithm import mode, moead, mopso, nsga2, nsga3

#: The reference point both hypervolume readings are taken against, just outside the unit box so
#: that a front touching ``(1, 0)`` or ``(0, 1)`` still contributes.
REF_POINT = np.array([1.1, 1.1])

#: The budget every convergence reading below uses: 50 points per batch, 60 generations after the
#: initial sample. It is the budget the consumer's guardrail used, kept so the two readings compare.
POP_SIZE = 50
LONG_BUDGET = 60

#: The SHORT half of the ratio guard. Five generations is enough to be clearly unconverged and
#: cheap enough that the paired run costs about a tenth of the long one.
SHORT_BUDGET = 5

#: The seeds every ratio assertion is driven over. Four, because the property is claimed for every
#: seed and one seed cannot say that -- and because the ZDT2 spread measured over six seeds is wide
#: enough that a single draw is not a reading about the algorithm.
SEEDS = (0, 1, 2, 3)

PROBLEMS = ((zdt1, 'ZDT1'), (zdt2, 'ZDT2'))

#: Every algorithm this package ships. The benchmark exists to JUDGE them, so the set it is driven
#: over is the whole set and not the one the helper happens to default to.
ALGORITHMS = (nsga2, nsga3, mode, moead, mopso)


def front_of(problem: Problem, *, max_iter: int, seed: int, algorithm=nsga2) -> np.ndarray:
    """The objective values on the first front of one run."""
    return solve(problem, algorithm=algorithm, pop_size=POP_SIZE, max_iter=max_iter, seed=seed).pareto().values


def hv_ratio(front: np.ndarray, reference: np.ndarray) -> float:
    """The run's dominated hypervolume as a fraction of the analytic front's."""
    return hypervolume_2d(front, REF_POINT) / hypervolume_2d(reference, REF_POINT)


# --- The problems ------------------------------------------------------------------------------


def test_zdt1_evaluates_to_its_closed_form():
    problem = zdt1(n_var=3)
    assert (problem.name, problem.n_var, problem.n_obj) == ('ZDT1', 3, 2)
    np.testing.assert_array_equal(problem.xl, np.zeros(3))
    np.testing.assert_array_equal(problem.xu, np.ones(3))
    # The tail is zero, so g = 1 and f2 = 1 - sqrt(f1).
    np.testing.assert_allclose(problem.evaluate(np.array([[0.0, 0.0, 0.0]])), [[0.0, 1.0]])
    np.testing.assert_allclose(problem.evaluate(np.array([[0.25, 0.0, 0.0]])), [[0.25, 0.5]])


def test_zdt2_evaluates_to_its_closed_form():
    problem = zdt2(n_var=3)
    assert (problem.name, problem.n_var, problem.n_obj) == ('ZDT2', 3, 2)
    # Same g = 1 tail; the concave shape is f2 = 1 - f1**2, so 0.5 maps to 0.75 where ZDT1 maps
    # to 1 - sqrt(0.5) = 0.293. THAT DIFFERENCE IS THE WHOLE POINT OF SHIPPING BOTH: an algorithm
    # tuned on a convex front and broken on a concave one agrees with ZDT1 and only ZDT1.
    np.testing.assert_allclose(problem.evaluate(np.array([[0.0, 0.0, 0.0]])), [[0.0, 1.0]])
    np.testing.assert_allclose(problem.evaluate(np.array([[0.5, 0.0, 0.0]])), [[0.5, 0.75]])


def test_a_non_zero_tail_lifts_the_point_off_the_front():
    """``g > 1`` is what makes ZDT a search problem rather than a curve to read off."""
    problem = zdt1(n_var=3)
    on_front = problem.evaluate(np.array([[0.25, 0.0, 0.0]]))[0]
    lifted = problem.evaluate(np.array([[0.25, 0.5, 0.5]]))[0]
    assert lifted[0] == on_front[0], 'f1 is x0 and must not move when only the tail does'
    assert lifted[1] > on_front[1], 'a non-zero tail must cost f2, else the tail is not searched'


@pytest.mark.parametrize(('maker', 'name'), PROBLEMS)
def test_the_analytic_front_runs_corner_to_corner(maker, name):
    front = maker().pareto_front(50)
    assert front.shape == (50, 2), name
    np.testing.assert_allclose(front[0], [0.0, 1.0])
    np.testing.assert_allclose(front[-1], [1.0, 0.0])


def test_the_two_fronts_differ_in_curvature_and_not_only_in_name():
    """ZDT1 is convex and ZDT2 concave, so the midpoint separates them; the endpoints do not."""
    convex = zdt1().pareto_front(3)[1]
    concave = zdt2().pareto_front(3)[1]
    np.testing.assert_allclose(convex, [0.5, 1.0 - np.sqrt(0.5)])
    np.testing.assert_allclose(concave, [0.5, 0.75])
    assert concave[1] > convex[1]


# --- The metrics -------------------------------------------------------------------------------


def test_igd_is_zero_when_the_front_covers_the_reference():
    reference = zdt1().pareto_front(20)
    assert igd(reference, reference) == pytest.approx(0.0)


def test_an_empty_front_scores_inf_rather_than_zero():
    """A run that found nothing is the worst result, and by arithmetic it would read as the best."""
    reference = zdt1().pareto_front(20)
    assert igd(np.zeros((0, 2)), reference) == float('inf')


def test_igd_grows_with_the_distance_to_the_reference():
    reference = np.array([[0.0, 1.0], [1.0, 0.0]])
    near = np.array([[0.1, 1.0], [1.0, 0.1]])
    far = np.array([[0.5, 1.0], [1.0, 0.5]])
    assert 0.0 < igd(near, reference) < igd(far, reference)


def test_hypervolume_of_hand_built_staircases():
    unit = np.array([1.0, 1.0])
    # One point at the origin dominates the whole unit square.
    assert hypervolume_2d(np.array([[0.0, 0.0]]), unit) == pytest.approx(1.0)
    # Two steps: (0, 0.5) covers 1.0 x 0.5, then (0.5, 0) adds 0.5 x 0.5.
    assert hypervolume_2d(np.array([[0.0, 0.5], [0.5, 0.0]]), unit) == pytest.approx(0.75)


def test_a_point_that_does_not_dominate_the_reference_contributes_nothing():
    unit = np.array([1.0, 1.0])
    assert hypervolume_2d(np.array([[1.5, 1.5]]), unit) == 0.0


def test_hypervolume_refuses_a_third_objective_rather_than_reading_the_first_two():
    """A 3-objective hypervolume is a different algorithm, and a plausible wrong number is worse."""
    with pytest.raises(ValueError, match='2 objectives'):
        hypervolume_2d(np.zeros((3, 3)), np.array([1.0, 1.0, 1.0]))


# --- The run -----------------------------------------------------------------------------------


@pytest.mark.parametrize('algorithm', ALGORITHMS, ids=lambda factory: factory.__name__)
@pytest.mark.parametrize(('maker', 'name'), PROBLEMS)
def test_every_algorithm_this_package_ships_can_be_benchmarked(maker, name, algorithm):
    """THE BENCHMARK JUDGES THE WHOLE SET, which is the point of shipping a benchmark at all.

    Until 2026-09-19 :func:`solve` named NSGA-II and took no say in it, so four of the five
    algorithms had no route through the helper the package offers for scoring one. That default
    was not a design decision taken here -- it was the CONSUMER's, whose own wrapper described
    itself as "thin wiring over the optimi_lab NSGA-II", and it survived the rewrite underneath it.

    What it hid is in the numbers, measured 2026-09-19 at pop_size=50, max_iter=60, seed=0:
    ``mode`` reaches IGD 0.0006 on ZDT1 and 0.0005 on ZDT2, while the hardcoded ``nsga2`` reaches
    0.0665 and 0.1204 -- two orders of magnitude, on the very problems this module ships in order
    to reveal exactly that. Nobody could see it, because nothing could run the other four.
    """
    problem = maker(n_var=3)
    reference = problem.pareto_front(200)
    short = front_of(problem, max_iter=SHORT_BUDGET, seed=0, algorithm=algorithm)
    long = front_of(problem, max_iter=LONG_BUDGET, seed=0, algorithm=algorithm)
    assert igd(long, reference) < igd(short, reference), f'{algorithm.__name__} on {name} is no closer for the budget'
    assert hv_ratio(long, reference) > hv_ratio(short, reference), f'{algorithm.__name__} on {name} gained no volume'


def test_the_algorithm_that_solve_defaults_to_is_not_the_best_one_it_ships():
    """A TRUE claim no test could have failed while the choice was hardcoded, so it is pinned here.

    Measured 2026-09-19 on ZDT2 at seed 0: ``mode`` scores IGD 0.0005 against ``nsga2``'s 0.1204,
    a factor of 200. The assertion is the RATIO and not either value -- it says the default is
    beaten by an order of magnitude, which stays true if both implementations improve and reds if
    the gap closes or inverts, either of which is news worth hearing.
    """
    problem = zdt2(n_var=3)
    reference = problem.pareto_front(200)
    default = igd(front_of(problem, max_iter=LONG_BUDGET, seed=0, algorithm=nsga2), reference)
    best = igd(front_of(problem, max_iter=LONG_BUDGET, seed=0, algorithm=mode), reference)
    assert best * 10 < default, f'mode={best:.4f} no longer beats nsga2={default:.4f} by an order of magnitude'


def test_solve_hands_back_the_self_describing_record():
    """The columns of a run are named by the record, not by the caller's memory of the argument order."""
    problem = zdt1(n_var=3)
    record = solve(problem, pop_size=8, max_iter=2, seed=0)
    assert isinstance(record, Record)
    assert list(record.space.names) == ['x0', 'x1', 'x2']
    assert list(record.objectives.names) == ['f0', 'f1']
    assert len(record) == 8 * 3, 'max_iter counts the generations AFTER the initial sample'
    assert record.pareto().values.shape[1] == problem.n_obj


@pytest.mark.parametrize(('maker', 'name'), PROBLEMS)
def test_one_seed_is_one_run(maker, name):
    """The seed reaches the sampler AND the proposer's generator, or a benchmark measures noise."""
    problem = maker(n_var=3)
    first = front_of(problem, max_iter=SHORT_BUDGET, seed=7)
    again = front_of(problem, max_iter=SHORT_BUDGET, seed=7)
    np.testing.assert_array_equal(first, again, err_msg=name)


@pytest.mark.parametrize('seed', SEEDS)
@pytest.mark.parametrize(('maker', 'name'), PROBLEMS)
def test_a_longer_budget_converges_closer(maker, name, seed):
    """THE RATIO GUARD, and the one assertion here that no seed can flatter.

    Sixty generations must beat five on BOTH metrics. Measured 2026-09-19 over the eight
    (problem, seed) pairs below, the margin is never close: ZDT1 IGD went 0.130 -> 0.067 at its
    tightest pair and 0.438 -> 0.161 at its widest, ZDT2 0.289 -> 0.120 and 0.677 -> 0.400. An
    absolute threshold could not have said this -- ZDT2 `seed=1` ends at IGD 0.400, worse than
    ZDT1's UNCONVERGED reading, so any bar that passed one would have to pass the other.
    """
    problem = maker(n_var=3)
    reference = problem.pareto_front(200)
    short = front_of(problem, max_iter=SHORT_BUDGET, seed=seed)
    long = front_of(problem, max_iter=LONG_BUDGET, seed=seed)
    assert igd(long, reference) < igd(short, reference), f'{name} seed={seed}: the longer run is no closer'
    assert hv_ratio(long, reference) > hv_ratio(short, reference), f'{name} seed={seed}: no volume gained'


def test_nsga2_reaches_the_convex_front():
    """CHARACTERIZATION, quoted at the WORST of six seeds and not at the best.

    Measured 2026-09-19 over seeds 0-5: IGD spans 0.067 (seed 0) to 0.173 (seed 4) and the
    hypervolume ratio spans 0.935 down to 0.857. The bars below sit just outside the worst
    reading, so a regression in any of the six draws reds -- a bar set at seed 0 would have had
    115% of headroom on the seed it was not measured on.
    """
    problem = zdt1(n_var=3)
    reference = problem.pareto_front(200)
    for seed in range(6):
        front = front_of(problem, max_iter=LONG_BUDGET, seed=seed)
        assert igd(front, reference) < 0.20, f'seed={seed}'
        assert hv_ratio(front, reference) > 0.82, f'seed={seed}'


def test_nsga2_reaches_the_concave_front_by_igd():
    """The same reading on ZDT2, where the worst of six seeds is IGD 0.400 against ZDT1's 0.173.

    THAT GAP IS THE FINDING, not the bar: the concave front costs this implementation a factor of
    two on the same budget, and the hypervolume half of it is carried as a failure on record by
    the xfail below rather than by widening anything here.
    """
    problem = zdt2(n_var=3)
    reference = problem.pareto_front(200)
    for seed in range(6):
        assert igd(front_of(problem, max_iter=LONG_BUDGET, seed=seed), reference) < 0.45, f'seed={seed}'


@pytest.mark.xfail(
    strict=True,
    reason=(
        'A FAILURE ON RECORD WITH ITS RESIDUAL, and BOTH numbers are quoted because a rewrite that '
        'agrees worse is a loosened band arriving through the evidence file. The retired NSGA-II this '
        'package shipped before the 2026-09-14 rewrite measured hypervolume ratio 0.800 on ZDT2 at '
        'pop_size=50, max_iter=60, seed=0 -- the reading its consumer pinned a 0.700 bar against. The '
        'same three numbers through the rewritten loop measure 0.685 (re-measured 2026-09-19), and over '
        'seeds 0-5 the ratio falls as far as 0.312 while ZDT1 never drops below 0.857. So the concave '
        'front is UNDER-FILLED by this implementation and the bar is left where the better '
        'implementation put it. Remove this xfail when the front is filled to it again; do not move it.'
    ),
)
def test_nsga2_fills_the_concave_front_to_the_bar_the_retired_loop_reached():
    problem = zdt2(n_var=3)
    reference = problem.pareto_front(200)
    assert hv_ratio(front_of(problem, max_iter=LONG_BUDGET, seed=0), reference) >= 0.700
