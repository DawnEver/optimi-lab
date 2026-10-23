---
name: real-unit-map-endpoints
description: Preserve exact declared endpoints and finite wide-interval forward/inverse maps; qualified with an isolated HPC environment before local commit.
created: 2026-10-23
metadata:
  type: project
---

# Real unit maps preserve the declared interval

Baseline: `dca72b71758ee4c70fb991fbcb44c56a9e52498d`.

The old linear decoder computed `lo + (hi - lo)` even for unit coordinate 1.
For `[0.001, 0.01]` this produced `0.010000000000000002`, which the same variable's
strict `contains` and `Action.of` correctly refused. Finite endpoints could also have an
overflowing intermediate width, making the decoder/inverse invalid on `[-1e308, 1e308]`.

The owner fix returns declared endpoints exactly in both directions, uses a finite scaled
inverse/weighted interpolation when the intermediate width overflows, and preserves the existing
unit-vector projection semantics. No product tolerance, subclass or runtime monkeypatch is added.
Ordinary finite interior interpolation remains unchanged; no physical acceptance bar is changed.

## Actual HPC evidence

- RED job 7805105, exact engine frozen development environment `f091c945f1cf6249`:
  18 selected new cases, 13 failed / 5 passed. The official verifier additionally reported an
  uninstalled-hook infrastructure failure, separately from these behavioral failures.
- GREEN job 7805113: declared `uv sync --frozen --extra dev`, official hooks installed in the
  independent clone, then `python -m lab_commons.dev.verify -- tests/unit/problem tests/unit/oracle`.
  80 tests passed in 5.28 s; Ruff/style over 247 files and hook checks passed.
- GREEN environment `49642387416af10a`, measured tree `sha256:0ff88960da384a9f`.
- Dirty source overlay SHA256:
  `a638c38c171e8c59a0b0439765111046a203287334c7c26dc267a30bd48ea14d`.
- `variables.py` SHA256:
  `725caddce67337c7993df579b05f6196a282e7dc87e0ee53d75cc17474d3bf12`.

These are actual independent compute-node tests, not local heavy execution or a remote forge
merge verdict. Engine tests used their own frozen dependency graph rather than borrowing the
product's incompatible graph. Earlier setup refusals ran no tests and are not behavioral reds.

The patch is prepared in a repository cloned inside the Motronics workspace. Remote branch
publication and the consuming product's dependency update require the user's explicit approval;
neither a push nor a remotely fetchable new dependency is claimed here.
