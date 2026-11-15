---
name: sqp-never-proposes-a-non-finite-design
description: sqp refuses a non-finite start and halts STALLED by name on a non-finite linearisation or QP step, so no NaN design reaches the oracle; the NaN source of HPC job 8012559 was not reproduced inside sqp.
created: 2026-11-15
---
# SQP never proposes a non-finite design

Incident: HPC paramfloor8 (job 8012559, pinned to b8671ec) passed the degenerate vertex, then died with
`exposed variable 'barrier_thickness_ratio' = nan is outside its declared bound` -- a NaN design reached the oracle.

Reading `solver/constrained.py::sqp` at b8671ec: every proposal is `clip(x + t d)` with `t` a power of 0.5, `x` a
previously evaluated point and `d` the QP step. A NaN in `d` makes `l1_slope` NaN and `not slope < 0` halts before
any proposal; an inf in `d` clips to a bound. So from a FINITE start and finite oracle answers sqp cannot propose NaN;
the open routes were (a) a non-finite START (`np.clip` keeps NaN; the start is the encoded refine design, e.g. a
polish design from the exploration record), and (b) a non-finite oracle answer, which made the QP loop on NaN to
its iteration cap or raise `LinAlgError` from `lstsq` (an exception, not a halt). Not reproduced end to end: the
downstream oracle's NaN source is outside this repo.

Invariant now: a non-finite start is refused before any evaluation (names the coordinates); `_subproblem` refuses a
linearisation holding a non-finite f/df/c/dc/h/dh (named) or a non-finite QP step, and sqp halts STALLED with it.
Tests: `tests/unit/solver/test_sqp_survives_a_degenerate_vertex.py` (non-finite answers at the 1st/2nd/3rd
evaluation, every part, NaN/+inf/-inf; non-finite start).
