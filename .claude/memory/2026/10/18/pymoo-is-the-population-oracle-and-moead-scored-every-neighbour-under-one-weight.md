---
name: pymoo-is-the-population-oracle-and-moead-scored-every-neighbour-under-one-weight
description: Closes P12's owed item -- pymoo 0.6.2 as a test-only reference oracle for every population method it also implements, median IGD of three seeds, gap tolerance in objective units, random-search control; MOEA/D's DTLZ2 gap was a replacement bug that scored a child under the proposer's weight only.
metadata:
  type: project
created: 2026-10-18
accessed: 2026-10-18
---

# pymoo is the population oracle; MOEA/D scored every neighbour under one weight

Follows `p12-population-methods-are-operators-under-solvers.md`. Ruling R1: runtime never imports pymoo.

## Door

- pymoo is in the `dev` extra (the only test extra here). Installed by `-m lab_commons.dev.dep --bootstrap <worktree>`
  (current gitea lab-commons, which has `boxlock`), then `-m lab_commons.dev.dep --root . --sync --upgrade-package pymoo`.
  The kit has no `--lock-upgrade` flag; `--sync` re-locks and syncs. The re-lock also moved lab-commons in `uv.lock`.

## Oracle

- `tests/benchmarks/pymoo_reference.py`: pymoo evaluates the catalogue's own oracle and records every
  feasible row; IGD is read off the whole evaluated set against the catalogue's closed-form front, at
  pymoo's defaults, same population/budget/seeds.
- Pairs: nsga2-NSGA2, nsga3-NSGA3, unsga3-UNSGA3, spea2-SPEA2, sms_emoa-SMSEMOA, mode-NSDE, moead-MOEAD,
  rvea-RVEA, mopso-MOPSO_CD. Unpaired: rnsga3 (aspiration method; whole-front IGD does not measure it).
- `tests/benchmarks/test_pymoo_oracle.py`: gap `median(ours) - median(pymoo) <= tol`; tol 0.01 on
  ZDT1/DTLZ2 (dimensionless), 0.2 on BNH (objectives 0-140). Control: random search at the same budget
  fails the gap against every reference.
- Caveats: pymoo NSGA3/UNSGA3/MOEAD/RVEA use the full 66-direction Das-Dennis grid on DTLZ2 (optimi-lab
  thins to 60); pymoo MOEAD uses PBI on 3 objectives, optimi-lab Tchebycheff.

## Measured 2026-10-18 (median IGD, seeds 1-3; after the MOEA/D fix)

| bench | method | optimi-lab | pymoo |
|---|---|---|---|
| zdt1 | nsga2 / nsga3 / unsga3 / spea2 | 0.0042 / 0.0038 / 0.0031 / 0.0036 | 0.0044 / 0.0071 / 0.0082 / 0.0050 |
| zdt1 | sms_emoa / mode / moead / rvea / mopso | 0.0013 / 0.0009 / 0.0044 / 0.0154 / 0.0015 | 0.0035 / 0.0025 / 0.0879 / 0.0193 / 0.0020 |
| dtlz2 | nsga2 / nsga3 / unsga3 / spea2 | 0.0265 / 0.0252 / 0.0245 / 0.0255 | 0.0285 / 0.0276 / 0.0259 / 0.0286 |
| dtlz2 | mode / moead / rvea / mopso | 0.0257 / 0.0313 / 0.0259 / 0.0686 | 0.0361 / 0.0266 / 0.0254 / 0.0766 |
| bnh | nsga2 / spea2 | 0.2151 / 0.1454 | 0.1392 / 0.1303 |
| control | random search zdt1 / dtlz2 / bnh | 0.761 / 0.114 / 0.777 | |

## MOEA/D cause and fix

- `DecompositionSolver._replace` computed the child's Tchebycheff score against every neighbour's weight
  and then took `[0]` -- the score under the PROPOSER's weight only -- and compared that scalar with each
  neighbour's incumbent under the neighbour's own weight. A child was judged on the wrong subproblem
  everywhere but its own. Before the fix DTLZ2 sat at 0.153, WORSE than random search (0.114).
- Fix: drop `[0]`. Red-first test `test_a_moead_child_is_scored_under_each_neighbours_own_weight`.
  DTLZ2 0.153 -> 0.031 (pymoo 0.027); gate tightened 0.2 -> 0.035 (DTLZ2) and 0.05 -> 0.01 (ZDT1).

## Kit-drift reds, not touched

- tests/architecture: bounded/hook_install/resources/installdoor/famtests import errors, family config
  render, shared-registry adoption, agent-guard rendered rules, roster census, project-file single source
  (docs-src/dev/index.md pointer table), private marker in pyproject.toml:24 (gitea URL).
