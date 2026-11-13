---
created: 2026-11-08
---
# The L6 model package moved to learn-lab (A1); only the bench stayed

- Every surrogate model, the protocol, registry, validation and generations now live in the sibling library
  learn-lab (`learn_lab.models.response`, `learn_lab.protocol|registry|validation|generations`); the old L6 model package
  is a retired spelling (`tests/architecture/test_a_retired_spelling_stays_retired.py`). No alias.
- `bench` stayed, as `optimi_lab.surrogate.bench` at L6 (the layer row was renamed `model` -> `surrogate`): it scores
  models by Pareto front, IGD+ and a feasibility acquisition -- search domain, so learn-lab must not own it.
- learn-lab joined the runtime (`test_the_runtime_stays_pure.py`, sixth entry); scikit-learn left the dev group with
  the model tests (learn-lab keeps it as its oracle).
- The lock was NOT relocked: learn-lab is unpublished (its first push waits on dev-lab #317). Verified locally with
  learn-lab's worktree `src` on PYTHONPATH. Relock (`dev_lab.dev.netverb -- uv lock`) once learn-lab is on GitHub.
- `pr/sensitivity-bagging-hpo` adds `model/bagging.py` and `model/tuning.py`: when it integrates, they move to
  learn-lab (model domain: bagging is a model, tuning is validation) and its `model/__init__` hunk drops.
