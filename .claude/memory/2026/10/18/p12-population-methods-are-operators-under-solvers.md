---
name: p12-population-methods-are-operators-under-solvers
description: Track D phase P12 of PLAN-engines-and-one-outcome-vocabulary (downstream plan, 2026-10-17) -- D69-D115 delivered on rebuild/p12-population over P8; the 0.3 algorithm package split by layer into optimi_lab.operator (L5) and optimi_lab.solver (L7); R1 met outside surrogates; two P8 operator defects found by the gate; what was not verified is listed.
metadata:
  type: project
created: 2026-10-18
accessed: 2026-10-18
---

# P12: population methods are operators under solvers

Plan: the downstream plan `PLAN-engines-and-one-outcome-vocabulary`, Track D, P12.
Base: P8 (`rebuild/p8-base`), then merged forward to optimi-lab main 240ed83.

## Layout

- `core`: `qmc` (Sobol in Gray-code order from embedded Joe-Kuo direction numbers `_joe_kuo`, 256
  dimensions; scrambled Halton; LHS; maximin LHS), `sampling` kinds grid/LHS/maximin/Poisson
  (Bridson, a given radius is a guarantee)/Sobol/Halton, `pareto` + epsilon/tolerance dominance and
  ENS incremental sort.
- `operator` (L5): variation (SBX, PM, DE strategies rand/1 rand/2 best/1 best/2 current-to-best/1
  current-to-rand/1, binomial/exponential crossover, PSO), discrete (lattice, snap, ladder/reset
  mutation, OX, swap, inversion), repair, selection, reference (NSGA-III Deb-Jain normalisation,
  niching tournament, R-NSGA-III aspiration directions, RVEA APD), indicator (SPEA2, HV selection),
  scalarisation (six, each with a declared `Reach`), stratified sampling.
- `metric` (L4): HV exact/MC/contributions, IGD, IGD+, spread, attainment, rank tests, bootstrap.
- `analysis`: Morris, conflict learning (objective reduction); scipy.stats removed.
- `solver` (L7): ga, nsga2, nsga3, unsga3, rnsga3, spea2, sms_emoa, mode, rvea, moead, mopso, cmaes,
  map_elites, nelder_mead, mads (progressive barrier), bobyqa (BOBYQA-like), Chain, scans,
  multistart, restarts, ipop_cmaes, warm_start, SuccessiveHalving, hyperband, ASHA. Sequential
  local methods are generators driven by `SequentialSolver`.
- The 0.3 algorithm package, its factory module and the private direction grid are retired; the
  spellings and their replacements are in `tests/architecture/test_a_retired_spelling_stays_retired.py`.
- R1 guard: `test_the_runtime_takes_only_the_r1_scipy_submodules` (planted control).

## Oracles and what they showed

- pymoo could NOT be installed: the repo's dependency door (`scripts/dep.py`) needs
  `lab_commons.dev.boxlock`, absent from the kit at a3e0e2b, and the hook refuses direct installs.
  The population gate (`tests/benchmarks/test_population_gate.py`) therefore uses the closed-form
  fronts of ZDT1/DTLZ2, median IGD of three seeds at a fixed budget. A pymoo equal-seed comparison
  is still owed.
- scipy oracles (tests only): Sobol/Halton unscrambled EXACT; Nelder-Mead visits exactly scipy's
  points; CMA-ES and BOBYQA reach scipy's Rosenbrock optimum within 1e-8 / 1e-6; rank-sum and
  signed-rank p-values match to 1e-12.
- DEFECTS FOUND IN THE P8 OPERATORS by the gate: SBX returned only the child beside the lower
  parent (NSGA-II ZDT1 IGD 0.22 -> 0.004 once symmetric); MOPSO's archive held dominated leaders and
  its velocity cap was 0.1 of the range (IGD 0.9 -> 0.002 with a non-dominated archive and OMOPSO's
  half-range cap). NSGA-III counted niches over the overflow front only (fixed, planted test).
- The run layer costs ~12 ms per evaluation (two fsyncs per result); gates drive solvers in memory.

## Not verified / open

- MOEA/D (Tchebycheff) on DTLZ2: IGD 0.10-0.19, gated at 0.2; its population is still at radius 1.05
  after 3600 evaluations. Undiagnosed without a pymoo reference.
- MADS on 4-D Rosenbrock reaches only 0.027 in 6000 evaluations; constrained optimum within 4.3e-4.
- Kit-drift reds not touched: bounded/hooks/dep door/install doors/verdict box/famconfig, registry
  adoption, private markers in the upstream gitea URL (pyproject/uv.lock), `optimi_lab.__version__`.
