---
created: 2026-11-12
---
# SQP resets an ill-conditioned BFGS model, and a refused QP is a halt, never an exception

Incident (motronics HPC job 7990796, spline11 rib4): a memetic GA + SQP refine on 82 axes exited 2 with
`the QP Hessian is not positive definite on the null space of the active constraints`
(`operator/qp.py`), ending the search with no report.

- CAUSE, measured on a noisy-gradient staircase probe (`tests/unit/solver/test_sqp_model_stays_definite.py`,
  5 of 200 seeds red): the model was NOT lacking damping -- Powell damping (`damped_secant`) was already
  on. Damping bounds curvature from BELOW only; noisy gradient differences over short steps drove the
  model's LARGEST eigenvalue to 1e12-1e15 against a smallest of 1e-3..1e-5 (one went -4.7e-4 by
  roundoff). The reduced Hessian was positive (~1e10) in most failures: `solve_kkt`'s pivot bar judged a
  condition-1e15 system singular. An eigenvalue SHIFT would not have helped; the big end is the defect.
- FIX (`solver/constrained.py::_secant`): the damped update is symmetrised; if it leaves
  `_CONDITION_CEILING = 1e8` or is not PD, it is reset to the identity (the start model). `sqp` counts
  resets into `Solution.evidence` as `hessian_resets:<k>`. `interior_point` gets the same reset (its
  count is not recorded; it already turned a singular KKT into STALLED).
- A `Refusal` from `active_set_qp` (singular KKT, cycling) now halts `sqp` STALLED with reason
  `the QP subproblem: <refusal>` at the last accepted point. Under `memetic` the refine concludes and the
  chain ends; the exploration's results stay in the archive.
- OPEN for motronics: nothing reads a refine's `Solution` (halt/reason/evidence) onto the run report.
