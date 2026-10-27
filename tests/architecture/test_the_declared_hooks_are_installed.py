"""HOOKS-ARE-WIRED, for THIS checkout -- and the gap was VIOLATED, not merely unenforced.

MEASURED 2026-09-16: the hook declaration here named 19 hooks at the `pre-commit` stage and **ZERO
were installed** in the directory git actually consults. Every commit through this checkout had run
none of them while the tree read as guarded.

THE DECLARATION IS `[tool.dev_lab.hooks]` IN `pyproject.toml` since 2026-10-18: the family
retired the pre-commit framework for one TOML table and one runner (`dev_lab.dev.githooks`), and
this repo takes the base as is (`dev_lab/dev/githooks/base.toml`). The body is the family's
(`dev_lab.dev.famtests.hookinstall`); what belongs to optimi-lab is the tree being asked about and
the named set of stages the merged table declares.

WHAT THIS PROVES, EXACTLY: the current lab_commons shim in every declared slot of the directory git
consults. Strictly weaker than "these hooks pass". The stages carry the push-side rules this repo
cites in its registry adoption: `commit-msg` runs the merge audit (MERGE-DEVIATIONS-NAMED), and
`pre-push` runs `dev_lab.dev.sessionbranch` (PUBLISH-IS-THE-LOCAL-REF, SESSION-HISTORY-ONLY-GROWS,
INTEGRATION-IS-MERGES-ONLY) and `branchset_push`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pytest
from dev_lab.dev.famtests import hookinstall

#: THE TREE UNDER TEST, not the working directory: a scan rooted at the CWD answers, plausibly,
#: about somebody else's checkout.
_ROOT: Final = Path(__file__).resolve().parents[2]

#: The two-sided pin, a SET rather than a count: a stage added and not pinned reds, a stage removed
#: reds. `pre-push` ARRIVED 2026-10-18 with the family base adopted as is -- the push-side session-
#: branch, branch-set, behind and lock rows -- where the retired configuration declared none.
DECLARED_STAGES: Final = frozenset({'commit-msg', 'post-commit', 'pre-commit', 'pre-push'})


def test_this_repository_declares_hooks_at_all() -> None:
    """CONTROL FOR THE INSTRUMENT and the two-sided pin, in one arm because they are one question."""
    hookinstall.assert_stages_are_declared(root=_ROOT, declared_stages=DECLARED_STAGES)


@pytest.mark.workstation
def test_every_hook_this_repository_declares_is_installed() -> None:
    """THE PROPERTY. Green means the commits and pushes through this checkout are guarded for real."""
    hookinstall.assert_declared_hooks_are_installed(root=_ROOT)


@pytest.mark.workstation
def test_the_hooks_directory_is_the_one_git_will_consult() -> None:
    """A guard that string-built `.git/hooks` answers about a directory git may not be using."""
    hookinstall.assert_hooks_dir_is_the_one_git_consults(root=_ROOT)


def test_this_guard_can_go_both_ways(tmp_path: Path) -> None:
    """PLANTED CONTROL: absent, then installed, in a real repository carrying THIS `pyproject.toml`."""
    hookinstall.assert_the_guard_can_go_both_ways(root=_ROOT, scratch=tmp_path)


def test_a_foreign_hook_is_not_an_installed_one(tmp_path: Path) -> None:
    """Somebody else's hook in every declared slot is its own finding, never a stale shim of ours."""
    hookinstall.assert_a_foreign_hook_is_not_installed(root=_ROOT, scratch=tmp_path)
