"""SHARED-CHECKOUT -- and closing it found real debris on the first run.

THE OLD REASON, verbatim from the sibling that wrote it first: *"the push obligation is a fact about
origin, not about any file in this checkout"*. The premise is true; the conclusion is not. A guard
does not have to read a FILE. `git` answers "what is here that origin does not have" exactly,
cheaply, and without a network call -- and on this repo the first honest run of it named
`fix/p0-integration-blockers`: a local branch with no origin counterpart, no upstream and no
worktree holding it. Work that exists only on one box exists for nobody, and nothing here had ever
said so.

IT IS PINNED AS A NAMED SET RATHER THAN ASSERTED EMPTY, which is the only honest shape while that
branch is still being decided about. A count could not say WHICH branch moved, and an empty
assertion would have to be either a red suite or a deletion nobody agreed to. The set is two-sided:
a NEW local-only branch reds, and so does the pin outliving the branch it names.

MEASURED AGAINST THE REMOTE, NEVER A LOCAL POINTER. A local trunk ref goes stale the moment another
party pushes, and against a stale one an already-landed change reads as unmerged -- which is how an
audit written to stop debt accumulating starts manufacturing it. That trap is PLANTED below on real
repositories with a real remote.

WHAT THIS CANNOT SEE, stated so a reader does not supply "everything": it answers about REFS, not
about the working tree. An uncommitted edit is invisible here by design -- that is a fact about a
moment, not about what origin has.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Final

from lab_commons.dev.checkout import (
    authority_for,
    classify,
    debris,
    origin_branches,
    stale_branches,
    unmerged_changes,
)

_ROOT: Final = Path(__file__).resolve().parents[2]
_GIT: Final = shutil.which('git') or 'git'

#: THE TRUNK, named rather than inferred: "is this worktree the trunk?" is an identity question, and
#: a repo that renames its trunk should have to say so here.
TRUNK: Final = 'main'

#: Local branches that exist ONLY on this box, pinned BY NAME. MEASURED 2026-09-16. Each entry is a
#: decision somebody owes: push it, or delete it -- and edit this set in the SAME change, because a
#: named set is what says which one moved.
LOCAL_ONLY_BRANCHES: Final = frozenset({'fix/p0-integration-blockers'})

#: Branches on origin besides the trunk, pinned BY NAME for the same reason. MEASURED 2026-09-16:
#: empty, and an empty pin is a real measurement -- the readability floor above is what keeps it
#: from meaning "nobody looked".
ORIGIN_BRANCHES: Final[frozenset[str]] = frozenset()


def test_this_checkout_is_readable_at_all() -> None:
    """THE FLOOR. An unreadable repository must never render as a clean one."""
    rows = classify(_ROOT, trunk=TRUNK)
    assert rows is not None, f'{_ROOT} could not be read as a git repository; this guard proved nothing'
    assert len(rows) >= 1, 'a checkout with no worktrees is an unread answer, not a tidy one'
    assert [row.branch for row in rows if row.kind == 'trunk'] == [TRUNK], (
        f'the trunk row is {[row.branch for row in rows if row.kind == "trunk"]}, not [{TRUNK!r}]. '
        f'Either this checkout is not on its trunk, or TRUNK above is stale.'
    )
    assert debris(_ROOT).readable, 'the debris census could not be read; a silent census is not a clean one'


def test_one_session_maintains_one_pushable_branch() -> None:
    """A second branch origin already carries is a second thing somebody must push and may forget."""
    rows = classify(_ROOT, trunk=TRUNK)
    assert rows is not None
    pushable = sorted(row.branch for row in rows if row.kind == 'primary')
    assert pushable == [], (
        f'{len(pushable)} branch(es) besides the trunk are on origin and not integrated: {pushable}. '
        f'Reconcile them into {TRUNK} and delete them, or finish them and let one become the trunk.'
    )


def test_nothing_of_value_exists_only_on_this_box() -> None:
    """Pinned BY NAME and two-sided: an arrival reds, and a pin outliving its branch reds too."""
    local = stale_branches(_ROOT)
    assert local is not None, 'the branch census could not be read'
    assert set(local) == LOCAL_ONLY_BRANCHES, (
        f'branches this box knows and origin does not: {sorted(set(local) - LOCAL_ONLY_BRANCHES)}; '
        f'pinned names that are gone: {sorted(LOCAL_ONLY_BRANCHES - set(local))}. Push it or delete '
        f'it, and edit the pin in the same change.'
    )


def test_the_origin_branch_set_is_the_declared_one() -> None:
    """BOTH SIDES: a branch appearing on origin reds, and a pin nobody deleted reds too."""
    names = origin_branches(_ROOT)
    assert names is not None, 'origin could not be read'
    assert set(names) == ORIGIN_BRANCHES, (
        f'new on origin: {sorted(set(names) - ORIGIN_BRANCHES)}; pinned and gone: '
        f'{sorted(ORIGIN_BRANCHES - set(names))}'
    )


def _run(cwd: Path, *args: str) -> None:
    subprocess.run([_GIT, *args], cwd=cwd, check=True, capture_output=True, timeout=60)


def _commit(repo: Path, name: str) -> None:
    (repo / name).write_text(name, encoding='utf-8')
    _run(repo, 'add', name)
    _run(repo, 'commit', '-q', '-m', f'add {name}')


def _cloned(tmp_path: Path, name: str, bare: Path) -> Path:
    work = tmp_path / name
    _run(tmp_path, 'clone', '-q', str(bare), str(work))
    _run(work, 'config', 'user.email', 'a@b.invalid')
    _run(work, 'config', 'user.name', 'Test')
    return work


def test_integration_is_measured_against_the_remote(tmp_path: Path) -> None:
    """PLANTED CONTROL for the trap that makes this audit lie: a STALE LOCAL TRUNK.

    The branch does not change; the local pointer does. Against the stale ref an already-landed
    change reads as unmerged, and a branch reported unintegrated is a branch nobody deletes.
    """
    bare = tmp_path / 'origin.git'
    _run(tmp_path, 'init', '-q', '--bare', '-b', 'main', str(bare))
    work = _cloned(tmp_path, 'work', bare)
    _commit(work, 'base.txt')
    _run(work, 'push', '-q', 'origin', 'main')

    other = _cloned(tmp_path, 'other', bare)
    _commit(other, 'landed.txt')
    _run(other, 'push', '-q', 'origin', 'main')
    _run(work, 'fetch', '-q', 'origin')

    assert authority_for(work, 'main') == 'origin/main', 'the remote is the authority whenever it carries the trunk'
    assert unmerged_changes(work, 'main', 'origin/main') == 1, 'against the STALE local ref the change looks new'
    assert unmerged_changes(work, 'origin/main', 'origin/main') == 0, 'against the remote it is landed'


def test_this_guard_names_a_planted_second_pushable_branch_and_a_planted_orphan(tmp_path: Path) -> None:
    """Both violations, PLANTED and driven through the REAL functions rather than described."""
    bare = tmp_path / 'origin.git'
    _run(tmp_path, 'init', '-q', '--bare', '-b', 'main', str(bare))
    work = _cloned(tmp_path, 'work', bare)
    _commit(work, 'base.txt')
    _run(work, 'push', '-q', 'origin', 'main')

    assert [row.kind for row in classify(work, trunk='main') or ()] == ['trunk'], 'the clean state, first'
    assert stale_branches(work) == ()

    lane = tmp_path / 'lane'
    _run(work, 'worktree', 'add', '-q', '-b', 'feat/second', str(lane))
    _commit(lane, 'x.txt')
    _run(lane, 'push', '-q', 'origin', 'feat/second')
    _run(work, 'fetch', '-q', 'origin')

    rows = classify(work, trunk='main')
    assert rows is not None
    assert sorted(row.branch for row in rows if row.kind == 'primary') == ['feat/second']
    assert origin_branches(work) == ('feat/second',)

    _run(work, 'branch', 'abandoned')
    assert stale_branches(work) == ('abandoned',), 'no origin, no upstream, no worktree -- exactly the planted shape'
    assert 'feat/second' not in (stale_branches(work) or ()), 'a checked-out, pushed branch is live'
