# Workflow — how work gets done in optimi-lab

- **The shared rules live in `lab_commons.dev.rules`; this repo supplies the MECHANISM.**
  `tests/architecture/test_optimi_lab_adopts_the_shared_registry.py` names, as a NAMED SET, every
  rule this repo enforces and every rule it does not. A rule added upstream reds that test until
  someone here decides about it — it never arrives adopted by default, and it never arrives
  silently unadopted.
- **A gap is DECLARED with a ceiling, never left silent.** The 16 rules optimi-lab does not
  enforce are listed by name with a ceiling that may only go DOWN. Closing one deletes its name
  and lowers the number in the same edit.
- **This page and `invariant.md` are HARD CONSTRAINTS ONLY.** Reasoning, incident write-ups and
  measurements go to `.claude/memory/`; how a thing works goes in the module docstring next to it.
  A rule that no reader could violate does not belong here — it is a description.
- **A new public name is exported from its module's `__all__` in the same commit that defines it**,
  and one name has one definition across the package. The package `__init__` re-exports; it does
  not redefine.
- **Tests before commit.** The suite is small and runs in seconds; there is no reason to push a
  red. A known failure is an `xfail` carrying its reason, never a `skip` — a skip is the one
  disposition that records nothing.
- **No `_legacy` / `_compat` shim and no deprecation alias.** This package was already rewritten
  once to remove that shape (`refactor(core)!: the library becomes pure numerics`); callers and
  tests move WITH the code, in the same commit.
