"""optimi-lab holds itself to ``CODE-IN-CODE-ROOTS`` and ``SCRATCH-ARCHIVED-OR-PROMOTED``.

ARRIVED 2026-10-01 with the two registry rules: `lab_commons.dev.codeplace` publishes the walk and
the scratch lifecycle, and what is left here is this tree's answer -- which directories may hold code,
which generated trees are not walked, and how long a one-off may wait in ``scratch/``.

The walk is of the FILESYSTEM, not the index, because the defect it refuses is an untracked probe in
``output/`` or ``usr/`` that never reaches the package or its tests.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from lab_commons.dev.codeplace import misplaced_code, overdue_scratch

_ROOT: Final = Path(__file__).resolve().parents[2]

#: Where code may live: the package, its tests, the repo's tooling, the runnable examples, the
#: installed agent-guard engine, and the one-off and archive homes.
CODE_ROOTS: Final = ('src/', 'tests/', 'scripts/', 'examples/', '.claude/hooks/', 'scratch/', '.claude/memory/')

#: Generated trees, ignored by git and written by a tool rather than a person: the built API docs
#: (`scripts/pdoc.py` emits `search.js`) and the coverage report.
#:
#: `.claude/worktrees` IS THE THIRD AND IT IS NOT GENERATED -- it holds SECOND CHECKOUTS of this repo,
#: one per open lane, each with its own `src/` and `scripts/`. THE KIT'S OWN DOCSTRING IS THE
#: ARGUMENT: another checkout's worktrees "are judged by the run that happens inside it". Walking
#: into them made this guard report eleven files of optimi-lab's own source as code outside the code
#: roots -- a reading about which branches happen to be checked out on this box, and about nothing
#: else. Added 2026-10-03, the same day and for the same reason as the identical line in the kit's
#: and wdg-lab's copies of this guard.
PRUNED: Final = ('.claude/worktrees', 'docs', 'htmlcov')

#: How long a one-off may wait in ``scratch/`` before it is archived or promoted.
SCRATCH_AGE_DAYS_CEILING: Final = 3.0


def test_no_code_file_lives_outside_a_code_root() -> None:
    found = misplaced_code(_ROOT, code_roots=CODE_ROOTS, pruned_paths=PRUNED)
    assert not found, f'code outside {list(CODE_ROOTS)}: {found[:10]}'


def test_no_scratch_file_outlives_its_time() -> None:
    late = overdue_scratch(_ROOT / 'scratch', root=_ROOT, max_age_days=SCRATCH_AGE_DAYS_CEILING)
    assert not late, f'archive or promote these one-offs: {late[:10]}'
