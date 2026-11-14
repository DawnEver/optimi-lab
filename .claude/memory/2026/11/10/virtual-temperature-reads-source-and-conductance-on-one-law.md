---
created: 2026-11-10
---
# VirtualTemperature reads source and conductance on ONE interpolation law

Coordinator ruling 2026-11-10 (from a downstream consumer's topology run): the enforced-gamma measure read conductance
on a derived threshold (~0.72) above the source's (0.5, beta 16), `s_min ~6.9e-10`. A gray interface cell
between them heated but barely conducted and read as a one-cell island: a one-body design read
`log_ratio` 4.7-8.6 filtered (0.98 exact) against `log(gamma)` 2.30, infeasible from iteration 0.

- Now `s(c) = s_min + (1 - s_min) c^penalty`, `c` read on `read_on`, is BOTH the source and the
  conductance; `conduct_on` and the threshold bisection are gone.
- Bound: built island, its outside neighbours at density `<= v` -> `log_ratio >= floor(v) =
  log(s(H(eta)) C / (2 s(H(v)) |domain| r_solid))`, `C = min measure / sum(face/distance)`.
  `island_floor = floor(0)` (H(0) = 0, so the void conducts `s_min`). `gamma` derives
  `s_min = target / 2`, `target = H(eta)^p C / (2 |domain| r_solid gamma ISLAND_MARGIN)`.
- Stated price: a bridge just under eta reads as joined (`floor(v)` collapses as `v -> eta`); the old
  test that refused it was replaced by one pinning both sides (0.3 refused, eta - 1e-3 joined).
- Red on e8adda0: the two-ring+spokes one-body fixture read 6.21 (beta 1) and 4.10 against 2.30.
- FD on an island at the enforced `s_min` is roundoff-bound (K conditioned ~1/s_min); the island FD
  case uses `s_min = 1e-4`, the same code path.

> Amended 2026-11-13: the downstream project's name replaced with "a downstream consumer" in a public repo, nothing else changed -- approved: "fix all"
