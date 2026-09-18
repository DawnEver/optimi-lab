r"""FAMILY-CONFIG-IS-RENDERED, for THIS checkout -- a hand edit to a shared artefact REDS.

THE DEFECT THIS CLOSES. `.gitignore`, `.pre-commit-config.yaml` and `Makefile` are hand-maintained
once per repo across the family, so a fix in one is a fix in one. `lab_commons.dev.famconfig` owns
each BASE as data plus a renderer; `_famconfig` here declares only what is a fact about optimi-lab;
the assertions are `lab_commons.dev.famtests.configrender`, which re-renders from the LIVE base and
delta and compares against disk. Nothing is cached and no digest is stored, so a base edited
upstream moves this verdict on the next run rather than on the next time somebody remembers to
re-sync.

THIS WAS THE CHEAPEST OF THE FIVE ROWS TO ADOPT BECAUSE THE SEAM WAS ALREADY DRAWN. Both labs wrote
this file within hours of `famconfig` landing, both already imported `DELTAS`, `EXTRA_HOOK_IDS` and
`REPO` from a local `_famconfig`, and both were 59% identical in CODE. What was missing was the
other side of a seam this tree had already cut, so what moved is the verdict half and what stayed is
the declaration half plus the numbers underneath it -- the hook floor, the directory floor, the
re-render remedy and the planted edit.

WHAT GREEN MEANS, STATED EXACTLY, because overclaiming it would be the defect the family calls
dominant. For a RENDERED artefact green means the bytes on disk are what base plus delta produce.
For a REQUIRED one it means every base line is PRESENT -- the Makefile's three shared targets share
a NAME and no RECIPE, so demanding byte equality there would assert a portability that does not
exist. It says nothing about whether any rule or target is CORRECT.

THREE ARMS THIS FILE USED TO HOLD ARE GONE BECAUSE THE KIT ALREADY DRIVES THEM AGAINST THE REAL
BASE. `test_an_anchor_that_names_no_position_is_still_refused` planted an ambiguous anchor, an
anchor on a line the base lacks and an anchor carrying no lines; `tests/test_dev_famconfig.py`
upstream plants all three, on this same base, alongside the delta-restates-a-base-line refusal the
sibling lab planted. A duplicate control that cannot diverge from its original is a maintenance cost
with no evidence value, and the consumer half of all four is `assert_delta_is_not_a_fork`, which
this file still runs on every declared artefact.

WHAT THE ADOPTION MOVED IN THIS TREE, read back off the REAL hook engine rather than off this file's
YAML: the `pre-commit-hooks` pin from v5.0.0 to v6.0.0 and `commitizen` from v4.6.0 to v4.13.9, and
`default_stages: [pre-commit]`, which narrows thirteen stock hooks from eleven stages to one. THE
NARROWING COSTS THIS CHECKOUT NOTHING, and that is a fact about the checkout rather than an
argument: the only git hook installed here is `pre-commit`, so the ten stages it removes had nothing
to run at. The sibling lab takes the identical base and records the identical change as a real LOSS
because it HAS a pre-push hook. Same config, opposite premise -- which is why both the narrowed set
and the premise under it are now pinned here instead of only the half this repo happened to notice.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pytest
from _famconfig import DELTAS, EXTRA_HOOK_IDS, PRECOMMIT_STAGE_MOVE, REPO
from lab_commons.dev import floors
from lab_commons.dev.famconfig import RENDERED, REQUIRED, artefact_base
from lab_commons.dev.famtests import configrender
from lab_commons.dev.hook_install import hooks_dir
from pre_commit.clientlib import load_config
from pre_commit.repository import all_hooks
from pre_commit.store import Store

#: THE TREE UNDER TEST, not the working directory: a scan rooted at the CWD answers, plausibly,
#: about somebody else's checkout.
_ROOT: Final = Path(__file__).resolve().parents[2]

#: The artefact the stage tests read. Named once so a rename cannot leave one of them measuring a
#: path that no longer exists and reporting nothing found as nothing wrong.
_PRECOMMIT: Final = '.pre-commit-config.yaml'

#: The floor under a stage reading. A config the engine resolved to nothing is consistent with every
#: claim about which hooks narrowed, so finding nothing there is vacuous rather than green. Set below
#: the 19 this repo declares: a floor refuses an unread file, it is not a second pin on the count.
_HOOK_FLOOR: Final = 15

#: THE OTHER SIDE OF ``_HOOK_FLOOR``, re-measured 2026-09-18 at 19 resolved hooks: today's
#: reading is 19 - 15 = 4.
_HOOK_HEADROOM: Final = 10

#: The floor under the reopening scan, and it is a fact about the BASE rather than about this repo:
#: MEASURED 2026-09-17, the `.gitignore` base carries five directory rules (`**/__pycache__/`,
#: `*.egg-info/`, `.mypy_cache/`, `.pytest_cache/`, `.ruff_cache/`). A scan over zero of them would
#: report every delta line harmless. It replaces this file's old floor, which asserted that THIS
#: repo has a negation at all -- a fact about one delta, where the hazard is a fact about the base.
_DIRECTORY_FLOOR: Final = 5

#: The hooks that still reach pre-push after the adoption, MEASURED 2026-09-17 through
#: `pre_commit.repository.all_hooks`. All five are stock and all five are here for the same reason:
#: their UPSTREAM manifest declares `[pre-commit, pre-push, manual]`, so the base's
#: `default_stages: [pre-commit]` never governed them. This repo declares no hook of its own, so
#: there is nothing else on the list.
_KEEPS_PRE_PUSH: Final = frozenset({
    'check-added-large-files',
    'check-shebang-scripts-are-executable',
    'destroyed-symlinks',
    'end-of-file-fixer',
    'trailing-whitespace',
})

#: NO PRE-PUSH HOOK IS INSTALLED IN THIS CHECKOUT, MEASURED 2026-09-17, and that single bit is what
#: makes the stage narrowing free here and a real loss in the sibling lab. Two-sided: installing one
#: reds, which is the correct moment to re-decide the narrowing rather than to discover it later as
#: a miss.
_PRE_PUSH_INSTALLED: Final = False

#: The command that re-renders this repo's artefacts. It is optimi-lab's own -- one lab runs an
#: inline interpreter line and the other a Makefile target -- so a guessed remedy would be a refusal
#: nobody can act on, which is how a refusal gets routed around.
_RERENDER_HINT: Final = 're-render it from tests/architecture/_famconfig.py through famconfig.render'


def _resolve(config: Path) -> list:
    """Resolve *config* with pre-commit's OWN engine -- the code the git hook itself runs.

    Not by parsing the YAML: a hook with no `stages:` key inherits them from the UPSTREAM manifest
    of the repo it comes from, so a reading taken off this file alone would be a guess about another
    repository dressed as a measurement. It is supplied as an argument because `pre_commit` is not a
    dependency of `lab_commons.dev` and must not become one.
    """
    return all_hooks(load_config(str(config)), Store())


def test_every_family_base_is_either_declared_or_refused() -> None:
    """COMPLETENESS, and it is what stops a new base arriving unnoticed.

    A test that only ever checks the artefacts this repo happens to declare cannot tell a base this
    repo has not adopted from one the kit has not published.
    """
    configrender.assert_every_base_is_accounted_for(deltas=DELTAS, repo=REPO)


@pytest.mark.parametrize('artefact', sorted(DELTAS))
def test_the_artefact_on_disk_is_what_the_live_base_and_our_delta_render(artefact: str) -> None:
    """THE PROPERTY. Green means this file is the family's answer plus lines this repo declared."""
    configrender.assert_artefact_is_rendered(
        root=_ROOT, artefact=artefact, deltas=DELTAS, repo=REPO, rerender_hint=_RERENDER_HINT
    )


@pytest.mark.parametrize('artefact', sorted(DELTAS))
def test_our_delta_is_a_delta_rather_than_a_fork(artefact: str) -> None:
    """The anti-fork arm, driven on the DECLARATION so it is a separate answer from "the file drifted".

    It is also the consumer half of every anchor control the kit already drives against this base,
    which is why this file no longer plants its own.
    """
    configrender.assert_delta_is_not_a_fork(artefact=artefact, deltas=DELTAS, repo=REPO)


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
        base=artefact_base('.gitignore'),
        delta=DELTAS['.gitignore'],
        repo=REPO,
        directory_floor=_DIRECTORY_FLOOR,
    )


def test_the_stage_narrowing_the_adoption_made_is_what_the_hook_engine_reads() -> None:
    """THE BEHAVIOUR CHANGE, MEASURED against the real engine rather than asserted in a docstring.

    Before adoption this repo declared no `default_stages`, so any hook whose UPSTREAM manifest
    declares none either inherited all eleven stages. The base declares `default_stages:
    [pre-commit]`, which narrows them to one. `_famconfig.PRECOMMIT_STAGE_MOVE` names exactly which
    thirteen ids that is -- a pin this file did not have before 2026-09-17 and could not see a hook
    silently JOINING the narrowing without.

    BOTH SETS ARE EQUALITIES. A declaration listing only what narrowed cannot see a hook joining the
    move, and one listing only what survived cannot see a hook leaving.
    """
    configrender.assert_stage_declaration(
        config=_ROOT / _PRECOMMIT,
        resolve=_resolve,
        hook_floor=_HOOK_FLOOR,
        narrowed=frozenset(PRECOMMIT_STAGE_MOVE),
        keeps_pre_push=_KEEPS_PRE_PUSH,
    )


def test_the_narrowing_is_free_here_only_while_no_pre_push_hook_is_installed() -> None:
    """WHY THE NARROWING IS ADOPTED WITHOUT A DROP REASON, and the condition it rests on, PINNED.

    Thirteen stock hooks lose ten stages each, and that costs this checkout nothing for one reason
    only: the sole git hook installed here is `pre-commit`, so those stages had nothing to run at.
    The sibling lab takes the identical base and records the identical change as a REAL loss,
    because it has a pre-push hook. The difference is the installation, not the config.

    THE FAMILY BODY MAKES THIS TWO-SIDED, which this repo's own version was not: a pre-push hook
    arriving reds, and so does the declaration outliving an installation the sibling lab has. Either
    way the premise is re-decided rather than discovered later as a miss.
    """
    configrender.assert_pre_push_premise(hooks_dir=hooks_dir(_ROOT), pre_push_installed=_PRE_PUSH_INSTALLED)


def test_the_pins_the_adoption_moved_are_the_bases_own() -> None:
    """THE OTHER THING THE ADOPTION MOVED, and it is a pin bump that must not pass unremarked.

    This repo held `pre-commit-hooks` at v5.0.0 and `commitizen` at v4.6.0; the base takes the newest
    measured pin of the three consumers, v6.0.0 and v4.13.9. Nothing in the rendered-bytes property
    would distinguish a base whose pin moved from one whose hook LIST moved, so the thing worth
    checking is that every id this repo declared before still RESOLVES at the new pin -- a stock hook
    renamed or removed upstream would otherwise arrive as a config that validates and a guard that
    quietly stopped running. It is a different question from the arm below, which reads the ids a
    HUMAN can see in the file.
    """
    hooks = configrender.resolved_stages(_ROOT / _PRECOMMIT, resolve=_resolve)
    floors.assert_floor(len(hooks), floor=_HOOK_FLOOR, what='FAMILY-CONFIG (resolved hooks)')
    floors.assert_floor_still_binds(
        len(hooks), floor=_HOOK_FLOOR, headroom=_HOOK_HEADROOM, what='FAMILY-CONFIG (resolved hooks)'
    )
    lost = sorted(set(EXTRA_HOOK_IDS) - set(hooks))
    assert not lost, (
        f'{lost} no longer resolve at the pins the base declares. A stock hook that vanished in an '
        f'upstream release is a guard this repo lost by upgrading, not a line to delete.'
    )
    assert 'commitizen' in hooks, 'commitizen did not resolve at the pin the base declares'


def test_the_extra_hook_ids_are_really_declared_in_this_repo() -> None:
    """THE ANCHOR'S SUBJECT, read off the live file rather than restated.

    `EXTRA_HOOK_IDS` is the evidence that the append-only limitation actually cost this repo
    something: every one of these belongs INSIDE the `pre-commit-hooks` entry the base renders. A
    named set nobody checks against the artefact is a claim about another tree, and a set that only
    ever agreed with itself would stay green through losing every hook it names.
    """
    configrender.assert_declared_ids_survive(
        config=_ROOT / _PRECOMMIT, extra_hook_ids=EXTRA_HOOK_IDS, hook_floor=_HOOK_FLOOR
    )


def test_this_guard_can_go_both_ways(tmp_path: Path) -> None:
    """PLANTED CONTROL. A guard green on the real tree could be asserting a constant.

    Both states are driven through the same function the property uses: the rendered artefact is
    written to a scratch path and must install, then one base line is edited by hand and it must
    not -- and the refusal has to NAME the line, or the guard reported a failure it never located.
    """
    configrender.assert_render_round_trips(
        artefact='.gitignore',
        deltas=DELTAS,
        repo=REPO,
        scratch=tmp_path,
        edit=('uv.lock', 'uv.lock.bak'),
    )


def test_the_makefile_check_is_required_rather_than_rendered() -> None:
    """The weaker mode is NAMED so that applying it stayed a decision somebody typed.

    If `Makefile` ever arrives as RENDERED, every recipe in this tree -- the `purity` and `adoption`
    targets, the `rm -rf` clean -- becomes a diff against a family file that cannot hold them. That
    is a change to argue with the kit, not to absorb here.
    """
    configrender.assert_modes_are_as_agreed(
        modes={'.gitignore': RENDERED, '.pre-commit-config.yaml': RENDERED, 'Makefile': REQUIRED}
    )


def test_the_survey_still_sizes_this_repo_the_way_the_adoption_recorded() -> None:
    """The instrument that sized this change, re-run against the adopted file.

    After adoption the survey must find NOTHING to drop: every base line is on disk. Before it, it
    found two -- the slashed `**/__pycache__/` and `*.egg-info/` this repo wrote bare. Pinning the
    post-adoption reading is what makes a silent removal of a base line visible here as well as in
    the main property, by a route that does not go through the rendered bytes.
    """
    configrender.assert_no_base_line_left_undeclared(root=_ROOT, artefact='.gitignore', repo=REPO)
