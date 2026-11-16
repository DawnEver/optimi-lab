---
created: 2026-11-13
---
# The tuning space is learn-lab's; optimi-lab searches it

- Gap: `optimi_lab.tuning.space` (RealRange / IntegerRange / OneOf) duplicated `learn_lab.space` (Real / Integer /
  Choice). One vocabulary now: `learn_lab.space`; the three optimi-lab types are deleted, no alias.
- Why learn-lab owns it: learn-lab is the numpy-only leaf and refuses importing optimi_lab
  (learn-lab `tests/architecture/test_the_layers_point_one_way.py`); optimi-lab already depends on heavier
  siblings. The only cycle-free owner both can import is learn-lab. optimi-lab keeps what is the SEARCH's: `Kind`,
  `kind_of` and `problem_space` (the mapping onto `optimi_lab.problem` variables).
- Cost: learn-lab is a sixth runtime distribution (`test_the_runtime_stays_pure.py` RUNTIME). The relock that adds it
  floated dev-lab too; the guard, CI workflows, door rules, pointer table, rule adoption (FORMATS-MIGRATE-ONCE and four
  new rules mapped, ceiling unchanged) and stored readings were re-rendered in the same commit.
