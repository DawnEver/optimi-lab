---
created: 2026-11-04
---
# Population solvers estimate the declared risk over shared scenarios

Gap (motronics handoff 10-25): robust search was declarable but no registered solver had a risk
estimator, so `match` refused every problem with `Uncertainty`.

- `operator.Risk(measure, alpha=, k=)` declares one of mean / mean_plus_std / var / cvar / worst_case
  over the existing `operator/risk.py` functions.
- `oracle.Action.scenario`: one draw of the problem uncertainty, in the identity only when set
  (existing keys unchanged); archive reads old records without it (upgrade on read).
- `solver/scenarios.py::Scenarios`: `scenarios` draws (default 16) taken once from the seed, shared by
  every design (common random numbers). `PopulationSolver` asks one action per (design, scenario),
  scores each design as the declared risk of each column and of the violation; a `Chance` constraint
  is the share of scenarios satisfying it, violation = shortfall / probability.
- Registry: nsga2/nsga3/unsga3/rnsga3/spea2/sms_emoa/rvea/mode/mopso, moead, ga/cmaes/ipop_cmaes/
  map_elites declare `uncertainty=True` and take `Chance`. Not declared: prescreened (its model
  scores predictions, not scenarios), nelder_mead/mads/bobyqa/multistart, every BO/gradient solver.
- Benchmark `tests/unit/solver/test_robust_population.py`: f=(x-0.3)^2+xi(1-x), xi~U(-.5,.5) has
  three optima (mean 0.3, worst 0.55, CVaR90 0.525); ga and cmaes land on each within 0.04. Chance
  `P(x+xi<=1)>=0.9` lands on the sample-average optimum of its 20 draws within 0.02.
- Open: `analysis/robust.py` (post-hoc DET/WC/CC choice) is still separate from the solver path;
  no QMC rule option for the scenario draw (MC only); motronics side not wired to pass `risk=`.
