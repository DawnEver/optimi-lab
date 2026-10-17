---
name: PLAN-one-native-architecture-for-every-algorithm
description: POINTER -- the optimi-lab rebuild plan moved into motronics memory (user ruling 2026-10-17, product architecture plans live in motronics) as PLAN-engines-and-one-outcome-vocabulary, Track D
created: 2026-10-17
accessed: 2026-10-17
tags: [plan, pointer, optimi-lab]
metadata:
  type: project
---

# Pointer: the optimi-lab rebuild plan lives in motronics

User ruling, 2026-10-17: architecture plans for the product live in motronics memory. The single
source for this repository's rebuild is motronics-studio
`.claude/memory/2026/10/17/PLAN-engines-and-one-outcome-vocabulary.md`:

- Part II.7: the optimi-lab architecture.
- Track D: the checklist.

Summary for a reader here:
- optimi-lab becomes a self-implemented decision library, layered L1-L9.
- At runtime it depends on numpy, the low-level scipy modules, numerics-lab and lab-commons.
- It records results with `lab_commons.outcomes`.
- No backward compatibility.
