---
created: 2026-11-08
---
# RULING: two published entries drop the downstream consumer's name (user, 2026-11-08)

- `test_no_tracked_file_carries_a_private_marker` refused the consumer's name in
  `2026/11/04/conditional-spaces-embed-and-models-fit-the-risk.md` and
  `2026/11/04/population-solvers-estimate-the-declared-risk.md`. A `corrects:` entry cannot clear it (the old text
  stays tracked), and `dev_lab.dev.memoryadds` has no declared exception.
- The user authorised rewriting exactly these two entries (the name only, meaning kept) and committed it
  themselves past the memory-adds hook. No other entry is covered.
- The third hit, the rendered `.claude/hooks/deny-rules.json`, is fixed at its cause in dev-lab #331.
