r"""FAMILY-CONFIG-IS-RENDERED, for THIS checkout -- a hand edit to a shared artifact REDS.

THE DEFECT THIS CLOSES. `.gitignore` and `.gitattributes` are hand-maintained
once per repo across the family, so a fix in one is a fix in one. `dev_lab.dev.famconfig` owns
each BASE as data plus a renderer; `_famconfig` here declares only what is a fact about optimi-lab;
the assertions are `dev_lab.dev.famtests.configrender`, which re-renders from the LIVE base and
delta and compares against disk. Nothing is cached and no digest is stored, so a base edited
upstream moves this verdict on the next run rather than on the next time somebody remembers to
re-sync.

THIS WAS THE CHEAPEST OF THE FIVE ROWS TO ADOPT BECAUSE THE SEAM WAS ALREADY DRAWN. Both labs wrote
this file within hours of `famconfig` landing, both already imported `DELTAS` and `REPO` from a local `_famconfig`, and both were 59% identical in CODE. What was missing was the
other side of a seam this tree had already cut, so what moved is the verdict half and what stayed is
the declaration half plus the numbers underneath it -- the hook floor, the directory floor, the
re-render remedy and the planted edit.

WHAT GREEN MEANS, STATED EXACTLY: the bytes on disk are what base plus delta produce. It says nothing
about whether any rule is CORRECT.

THREE ARMS THIS FILE USED TO HOLD ARE GONE BECAUSE THE KIT ALREADY DRIVES THEM AGAINST THE REAL
BASE. They planted an ambiguous anchor, an anchor on a line the base lacks and an anchor carrying no
lines; `lab-commons/tests/test_dev_famconfig.py` plants all three, on this same base, alongside the
delta-restates-a-base-line refusal the sibling lab planted. Their retired local spellings are
deliberately not repeated here: a name that resolves to nothing in THIS checkout sends a reader
looking for a guard they will not find, which is the defect `test_cited_tests_resolve.py` beside
this file exists to refuse. A duplicate control that cannot diverge from its original is a
maintenance cost with no evidence value, and the consumer half of all four is `assert_delta_is_not_a_fork`, which
this file still runs on every declared artifact.

THE GIT-HOOK ARTIFACT LEFT THE FAMILY ON 2026-10-11 (lab-commons 900c5bf): hooks are declared in
`[tool.dev_lab.hooks]` and run by `dev_lab.dev.githooks`, so the kit publishes no YAML base and
the arms that read the pre-commit engine's stage resolution went with it. What the hook table holds is
`test_the_declared_hooks_are_installed.py`'s subject now.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pytest
from _famconfig import DELTAS, REPO
from dev_lab.dev.famconfig import RENDERED, artifact_base
from dev_lab.dev.famtests import configrender

#: THE TREE UNDER TEST, not the working directory: a scan rooted at the CWD answers, plausibly,
#: about somebody else's checkout.
_ROOT: Final = Path(__file__).resolve().parents[2]

#: The floor under the reopening scan, and it is a fact about the BASE rather than about this repo:
#: MEASURED 2026-09-17, the `.gitignore` base carries five directory rules (`**/__pycache__/`,
#: `*.egg-info/`, `.mypy_cache/`, `.pytest_cache/`, `.ruff_cache/`). A scan over zero of them would
#: report every delta line harmless. It replaces this file's old floor, which asserted that THIS
#: repo has a negation at all -- a fact about one delta, where the hazard is a fact about the base.
_DIRECTORY_FLOOR: Final = 5

#: The command that re-renders this repo's artifacts. It is optimi-lab's own -- one lab runs an
#: inline interpreter line and the other a Makefile target -- so a guessed remedy would be a refusal
#: nobody can act on, which is how a refusal gets routed around.
_RERENDER_HINT: Final = 're-render it from tests/architecture/_famconfig.py through famconfig.render'


def test_every_family_base_is_either_declared_or_refused() -> None:
    """COMPLETENESS, and it is what stops a new base arriving unnoticed.

    A test that only ever checks the artifacts this repo happens to declare cannot tell a base this
    repo has not adopted from one the kit has not published.
    """
    configrender.assert_every_base_is_accounted_for(deltas=DELTAS, repo=REPO)


@pytest.mark.parametrize('artifact', sorted(DELTAS))
def test_the_artifact_on_disk_is_what_the_live_base_and_our_delta_render(artifact: str) -> None:
    """THE PROPERTY. Green means this file is the family's answer plus lines this repo declared."""
    configrender.assert_artifact_is_rendered(
        root=_ROOT, artifact=artifact, deltas=DELTAS, repo=REPO, rerender_hint=_RERENDER_HINT
    )


@pytest.mark.parametrize('artifact', sorted(DELTAS))
def test_our_delta_is_a_delta_rather_than_a_fork(artifact: str) -> None:
    """The anti-fork arm, driven on the DECLARATION so it is a separate answer from "the file drifted".

    It is also the consumer half of every anchor control the kit already drives against this base,
    which is why this file no longer plants its own.
    """
    configrender.assert_delta_is_not_a_fork(artifact=artifact, deltas=DELTAS, repo=REPO)


def test_no_delta_line_reopens_a_rule_the_base_closed() -> None:
    """TWO SHAPES OF ONE HAZARD, and the second is written to bite HERE even though it does not TODAY.

    `.gitignore` is last-match-wins per path and the delta renders AFTER the base. A BARE
    re-statement of a slashed base rule is a different RULE wearing a duplicate's clothes -- a
    trailing slash matches a directory only -- and it wins by coming later; this repo wrote two of
    the base's five that way and the adoption deliberately did not carry them. A NEGATION re-includes
    whatever the base excluded above it: this repo has exactly one (`!example.log`, naming a fixture
    no base rule touches) while the sibling lab has twenty and must close them again. Neither is
    visible to a byte comparison, because both render exactly as declared.
    """
    configrender.assert_no_rule_is_reopened(
        base=artifact_base('.gitignore'),
        delta=DELTAS['.gitignore'],
        repo=REPO,
        directory_floor=_DIRECTORY_FLOOR,
    )


def test_this_guard_can_go_both_ways(tmp_path: Path) -> None:
    """PLANTED CONTROL. A guard green on the real tree could be asserting a constant.

    Both states are driven through the same function the property uses: the rendered artifact is
    written to a scratch path and must install, then one base line is edited by hand and it must
    not -- and the refusal has to NAME the line, or the guard reported a failure it never located.
    """
    configrender.assert_render_round_trips(
        artifact='.gitignore',
        deltas=DELTAS,
        repo=REPO,
        scratch=tmp_path,
        edit=('**/__pycache__/', '**/python-cache/'),
    )


def test_every_artifact_is_judged_rendered() -> None:
    """The mode is NAMED so a base arriving in another mode changes a byte here, not only a meaning.

    The kit retired the weaker REQUIRED mode with its `Makefile` and `.rgignore` bases; both
    artifacts this repo declares are judged byte for byte.
    """
    configrender.assert_modes_are_as_agreed(modes=dict.fromkeys(DELTAS, RENDERED))


@pytest.mark.parametrize('artifact', sorted(DELTAS))
def test_the_survey_still_sizes_this_repo_the_way_the_adoption_recorded(artifact: str) -> None:
    """The instrument that sized this change, re-run against every adopted file.

    After adoption the survey must find NOTHING to drop: every base line is on disk. Pinning that
    reading makes a silent removal of a base line visible by a route that does not go through the
    rendered bytes.
    """
    configrender.assert_no_base_line_left_undeclared(root=_ROOT, artifact=artifact, repo=REPO)
