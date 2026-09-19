"""SHARED-CHECKOUT -- and closing it found real debris on the first run.

THE OLD REASON, verbatim from the sibling that wrote it first: *"the push obligation is a fact about
origin, not about any file in this checkout"*. The premise is true; the conclusion is not. A guard
does not have to read a FILE. `git` answers "what is here that origin does not have" exactly,
cheaply, and without a network call -- and on this repo the first honest run of it named
`fix/p0-integration-blockers`: a local branch with no origin counterpart, no upstream and no
worktree holding it. Work that exists only on one box exists for nobody, and nothing here had ever
said so.

THE BODY IS THE FAMILY'S AS OF 2026-09-17, AND THE SEAM WAS NARROWER THAN THE ROSTER ROW READ.
`lab_commons.dev.checkout` already published the MEASUREMENT and this file already imported it; what
was still forked three ways was the assertions around it and, above all, the CONTROLS -- the same
two planted git fixtures written twice, factored out here and inlined per function in the sibling
lab. A control is by definition the part nobody checks, so a control written twice can be wrong in
one copy. `lab_commons.dev.famtests.visibility` owns both now, `plant_checkout` included.

WHAT OPTIMI-LAB SUPPLIES, and every one of them is a keyword with NO DEFAULT upstream:

* `TRUNK` -- the branch this checkout must be on, and the `LAB_CZ_BASE_REF` shape in its purest
  form: a repo whose trunk is named otherwise, judged against a guessed `main`, resolves to nothing,
  finds no commits and REPORTS ITSELF CLEAN. The family body refuses to guess; this repo answers.
* `LOCAL_ONLY_BRANCHES` -- PINNED AS A NAMED SET RATHER THAN ASSERTED EMPTY. A count could not say
  WHICH branch moved. Two-sided: a NEW local-only branch reds, and so does the pin outliving the
  branch it names -- AND THE SECOND SIDE IS THE ONE THAT FIRED, on 2026-09-19. The entry said the
  named set was "the only honest shape while that branch is still being decided about"; the decision
  was taken, `fix/p0-integration-blockers` exists neither locally nor on origin, and the pin was the
  half nobody deleted with it. The set is empty now and that is a MEASUREMENT of the same kind
  `ORIGIN_BRANCHES` has carried since 2026-09-16, held from meaning "nobody looked" by the same
  readability floor. The first side is undiminished: the day a local-only branch appears, it reds.
* `ORIGIN_BRANCHES` -- the published branches besides the trunk, for the same reason.

MEASURED AGAINST THE REMOTE, NEVER A LOCAL POINTER. A local trunk ref goes stale the moment another
party pushes, and against a stale one an already-landed change reads as unmerged -- which is how an
audit written to stop debt accumulating starts manufacturing it. That trap is PLANTED below on real
repositories with a real remote.

WHAT THIS CANNOT SEE, stated so a reader does not supply "everything": it answers about REFS, not
about the working tree. An uncommitted edit is invisible here by design -- that is a fact about a
moment, not about what origin has.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from lab_commons.dev.famtests import visibility

_ROOT: Final = Path(__file__).resolve().parents[2]

#: THE TRUNK, named rather than inferred: "is this worktree the trunk?" is an identity question, and
#: a repo that renames its trunk should have to say so here.
TRUNK: Final = 'main'

#: Local branches that exist ONLY on this box, pinned BY NAME. Each entry is a decision somebody
#: owes: push it, or delete it -- and edit this set in the SAME change, because a named set is what
#: says which one moved.
#:
#: RE-MEASURED 2026-09-19: EMPTY. `fix/p0-integration-blockers` was pinned here on 2026-09-16 and is
#: now gone from this box and from origin alike, so the pin was naming nothing and the guard's SECOND
#: side convicted it -- which is the arm working, since a pin outliving its branch reads as a
#: decision still open when it is closed. The deletion of the branch is NOT this change's and no
#: record of who took it survives in the reflog; what is repaired here is only the declaration.
LOCAL_ONLY_BRANCHES: Final[frozenset[str]] = frozenset()

#: Branches on origin besides the trunk, pinned BY NAME for the same reason. MEASURED 2026-09-16:
#: empty, and an empty pin is a real measurement -- the readability floor above is what keeps it
#: from meaning "nobody looked".
ORIGIN_BRANCHES: Final[frozenset[str]] = frozenset()


def test_this_checkout_is_readable_at_all() -> None:
    """THE FLOOR. An unreadable repository must never render as a clean one."""
    visibility.assert_readable(root=_ROOT, trunk=TRUNK)


def test_one_session_maintains_one_pushable_branch() -> None:
    """A second branch origin already carries is a second thing somebody must push and may forget."""
    visibility.assert_one_pushable_branch(root=_ROOT, trunk=TRUNK)


def test_nothing_of_value_exists_only_on_this_box() -> None:
    """Pinned BY NAME and two-sided: an arrival reds, and a pin outliving its branch reds too."""
    visibility.assert_local_only_branches(root=_ROOT, declared=LOCAL_ONLY_BRANCHES)


def test_the_origin_branch_set_is_the_declared_one() -> None:
    """BOTH SIDES: a branch appearing on origin reds, and a pin nobody deleted reds too."""
    visibility.assert_origin_branch_set(root=_ROOT, declared=ORIGIN_BRANCHES)


def test_integration_is_measured_against_the_remote(tmp_path: Path) -> None:
    """PLANTED CONTROL for the trap that makes this audit lie: a STALE LOCAL TRUNK.

    The branch does not change; the local pointer does. Against the stale ref an already-landed
    change reads as unmerged, and a branch reported unintegrated is a branch nobody deletes. The
    fixture takes `TRUNK` too -- one hard-coding `main` could not build the repository that proves
    why the assertions above need a trunk at all.
    """
    visibility.assert_the_remote_is_the_authority(tmp_path, trunk=TRUNK)


def test_this_guard_names_a_planted_second_pushable_branch_and_a_planted_orphan(tmp_path: Path) -> None:
    """Both violations, PLANTED and driven through the REAL functions rather than described.

    It reads the CLEAN state first: a control that never saw the green side cannot tell a working
    guard from one that refuses everything.
    """
    visibility.assert_the_planted_debris_is_named(tmp_path, trunk=TRUNK)
