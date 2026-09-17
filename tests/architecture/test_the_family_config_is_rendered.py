r"""FAMILY-CONFIG-IS-RENDERED, for THIS checkout -- a hand edit to a shared artefact REDS.

THE DEFECT THIS CLOSES. `.gitignore`, `.pre-commit-config.yaml` and `Makefile` are hand-maintained
once per repo across the family, so a fix in one is a fix in one. `lab_commons.dev.famconfig` owns
each BASE as data plus a renderer; `_famconfig` here declares only what is a fact about optimi-lab;
this file re-renders from the LIVE base and delta and compares against disk. Nothing is cached and no
digest is stored, so a base edited upstream moves this verdict on the next run rather than on the
next time somebody remembers to re-sync.

WHAT GREEN MEANS, STATED EXACTLY, because overclaiming it would be the defect the family calls
dominant. For a RENDERED artefact green means the bytes on disk are what base plus delta produce.
For a REQUIRED one it means every base line is PRESENT -- the Makefile's three shared targets share
a NAME and no RECIPE, so demanding byte equality there would assert a portability that does not
exist. It says nothing about whether any rule or target is CORRECT.

THE RATCHET HAS TWO SIDES AND BOTH ARE HERE. A base line cannot silently vanish, because a missing
one with no declared drop reds; and a delta cannot silently grow into a fork, because every delta
carries a ceiling -- anchored lines included -- and a delta re-stating a base CONTENT line is refused
at render time.

`.pre-commit-config.yaml` WAS THE THIRD SIDE UNTIL 2026-09-17: an artefact quietly left unadopted,
held visible by a NAMED REFUSAL whose reason a test here re-drove. `Delta.anchored` killed both
causes that refusal stated, so the artefact is adopted and the refusal and its test are gone WITH
their subject, in the same commit -- a reason that outlives its cause is the declaration that lies.

WHAT REPLACED THEM IS NOT PROSE. The adoption moves this repo's two upstream pins and narrows every
hook the base's `default_stages` governs, and both are read back below off the REAL hook engine
rather than off this file's YAML: a hook with no `stages:` key inherits them from the manifest of
the repo it comes from, so a table computed here would be a guess about another repository wearing a
measurement's clothes. The narrowing costs this checkout nothing because only `pre-commit` is
installed here -- and THAT is asserted too, because the day a pre-push hook is installed the
narrowing stops being free and this repo should find out from a red rather than from a miss.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pytest
from _famconfig import DELTAS, EXTRA_HOOK_IDS, REPO
from lab_commons.dev.famconfig import (
    BASES,
    INSTALLED,
    RENDERED,
    REQUIRED,
    Delta,
    artefact_base,
    delta_problems,
    inspect_file,
    measured_delta,
    render,
)
from lab_commons.dev.hook_install import DEFAULT_CONFIG_NAME, hook_installation, hooks_dir
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

#: The hooks that still reach pre-push after the adoption, MEASURED 2026-09-17 through
#: `pre_commit.repository.all_hooks`. All five are stock and all five are here for the same reason:
#: their UPSTREAM manifest declares `[pre-commit, pre-push, manual]`, so the base's
#: `default_stages: [pre-commit]` never governed them. This repo declares no hook of its own, so
#: there is nothing else on the list.
_KEEPS_PRE_PUSH: Final = (
    'check-added-large-files',
    'check-shebang-scripts-are-executable',
    'destroyed-symlinks',
    'end-of-file-fixer',
    'trailing-whitespace',
)


def test_every_family_base_is_either_declared_or_refused() -> None:
    """COMPLETENESS, and it is what stops a new base arriving unnoticed.

    A test that only ever checks the artefacts this repo happens to declare cannot tell a base this
    repo has not adopted from one the kit has not published. So the two named sets must together
    cover the kit's, exactly -- and neither may name an artefact the kit does not own.
    """
    assert set(DELTAS) == set(BASES), (
        f"{REPO} accounts for {sorted(DELTAS)} against the kit's {sorted(BASES)}. A base the kit "
        f'publishes and this repo neither adopts nor refuses is an artefact drifting with nobody '
        f'saying so: add a Delta to _famconfig.DELTAS, or -- if it genuinely cannot be adopted -- a '
        f'named refusal carrying the reason, which is the shape this repo held '
        f'.pre-commit-config.yaml in until the kit learned to anchor an addition on 2026-09-17.'
    )


@pytest.mark.parametrize('artefact', sorted(DELTAS))
def test_the_artefact_on_disk_is_what_the_live_base_and_our_delta_render(artefact: str) -> None:
    """THE PROPERTY. Green means this file is the family's answer plus lines this repo declared."""
    base = artefact_base(artefact)
    report = inspect_file(_ROOT / artefact, base, DELTAS[artefact])
    offending = '\n  '.join(report.offending) or '(none listed)'
    remedy = (
        're-render it from tests/architecture/_famconfig.py through famconfig.render'
        if base.mode == RENDERED
        else f'add the missing line(s) to {artefact}, or declare each as a Delta.dropped entry saying why'
    )
    assert report.status == INSTALLED, (
        f'{artefact} is {report.status} in {REPO} ({base.mode} mode): {report.detail}\n  {offending}\n\n'
        f'A shared artefact edited by hand stops being shared the moment the edit lands, and nothing '
        f'else in this tree would have said so. If the edit was wanted it is a delta line or a '
        f'declared drop, and both of those live in tests/architecture/_famconfig.py.\n  {remedy}\n'
    )


@pytest.mark.parametrize('artefact', sorted(DELTAS))
def test_our_delta_is_a_delta_rather_than_a_fork(artefact: str) -> None:
    """The anti-fork arm, driven directly so it reds on the DECLARATION and not only on the file.

    `inspect_file` raises rather than reports when a delta is malformed, which is the right shape for
    a renderer and the wrong one for a reader trying to tell "the file drifted" from "the
    declaration is broken". Asking `delta_problems` separates them.
    """
    base = artefact_base(artefact)
    problems = delta_problems(base, DELTAS[artefact])
    assert not problems, f"{REPO}'s {artefact} delta is not a delta of the base:\n  " + '\n  '.join(problems)


def test_the_gitignore_delta_does_not_reinstate_the_bare_spellings() -> None:
    """THE BEHAVIOUR CHANGE THIS ADOPTION MADE, pinned so a later edit cannot quietly undo it.

    The base spells five directory patterns with a trailing slash and this repo wrote two of them
    BARE. A trailing slash matches a DIRECTORY ONLY, so the two spellings are different RULES rather
    than different formattings -- and `.gitignore` is last-match-wins per path, with the delta
    rendering AFTER the base. A bare spelling reappearing in the delta would therefore override the
    base's slashed rule while every byte comparison stayed green, which is the one regression this
    file's main property cannot see.
    """
    bare = {'**/__pycache__', '*.egg-info', '.mypy_cache', '.pytest_cache', '.ruff_cache'}
    reinstated = bare & set(DELTAS['.gitignore'].added)
    assert not reinstated, (
        f'the .gitignore delta re-adds {sorted(reinstated)}. Those render AFTER the base, and '
        f'gitignore is last-match-wins, so each one silently reverts the slashed base rule it looks '
        f'like a duplicate of. A trailing slash matches a directory only; drop the bare line.'
    )


def test_a_negation_in_the_delta_cannot_quietly_reopen_a_base_rule() -> None:
    """THE ORDERING HAZARD, written to bite HERE even though it does not bite here TODAY.

    `.gitignore` is last-match-wins per path, so a `!` line in the delta re-includes whatever the
    base excluded above it. This repo has exactly one negation (`!example.log`, which re-includes a
    fixture no base rule touches), while the sibling lab has twenty and must keep `__pycache__/` as
    its FINAL line to re-exclude the caches its `.claude/` re-inclusions would otherwise let back in
    -- the incident motronics' own `.gitignore` records, where a `.pyc` ended up TRACKED.

    So this asserts the PROPERTY rather than the current count: every negation must be narrower than
    a directory the base excludes, or the delta has to close it again after. Green today because the
    one negation names a single file.
    """
    base_dirs = {line.rstrip('/').lstrip('*/') for line in artefact_base('.gitignore').content_lines}
    negations = [line for line in DELTAS['.gitignore'].added if line.startswith('!')]
    assert negations, 'no negation at all makes this guard vacuous rather than green; it must find its subject'
    reopened = [line for line in negations if line.lstrip('!').rstrip('/*').lstrip('*/') in base_dirs]
    assert not reopened, (
        f'{reopened} re-include(s) a path the base excludes, and the delta renders AFTER the base. '
        f'Either narrow the negation, or close the rule again as the LAST line of the delta the way '
        f'wdg-lab does with `__pycache__/`. Last-match-wins means order is the rule.'
    )


def _resolved_stages() -> dict[str, frozenset[str]]:
    """Which stages each hook in the live artefact runs at, resolved by pre-commit's OWN engine.

    Not by parsing the YAML here: a hook with no `stages:` key inherits them from the UPSTREAM
    manifest of the repo it comes from, so a reading taken off this file alone would be a guess about
    another repository dressed as a measurement. This is the code the git hook itself runs.
    """
    return {hook.id: frozenset(hook.stages) for hook in all_hooks(load_config(str(_ROOT / _PRECOMMIT)), Store())}


def test_the_stage_narrowing_the_adoption_made_is_what_the_hook_engine_reads() -> None:
    """THE BEHAVIOUR CHANGE, MEASURED against the real engine rather than asserted in a docstring.

    Before adoption this repo declared no `default_stages`, so any hook whose UPSTREAM manifest
    declares none either inherited all eleven stages. The base declares `default_stages:
    [pre-commit]`, which narrows them to one. Five stock ids keep `[pre-commit, pre-push, manual]`
    because their own manifest says so and an explicit declaration beats a default, and `commitizen`
    keeps `commit-msg` the same way.

    A FLOOR, because a table read off a config the engine resolved to nothing agrees with every claim
    made about it, and both sides, because a set that only named what narrowed could not see a hook
    silently JOINING the narrowing.
    """
    hooks = _resolved_stages()
    assert len(hooks) >= _HOOK_FLOOR, (
        f'the engine resolved {len(hooks)} hooks from {_PRECOMMIT}, below the {_HOOK_FLOOR} floor. '
        f'A stage table read off an empty config agrees with every claim made about it.'
    )
    measured = frozenset(hook_id for hook_id, stages in hooks.items() if 'pre-push' in stages)
    assert measured == frozenset(_KEEPS_PRE_PUSH), (
        f'the hooks still reaching pre-push are {sorted(measured)}, and this repo declares '
        f'{sorted(_KEEPS_PRE_PUSH)}. Extra in the engine is a hook that kept a stage nobody recorded; '
        f"extra in the declaration is a reason that outlived its cause. Either way the adoption's "
        f'measurement has stopped describing the file.'
    )


def test_the_narrowing_is_free_here_only_while_no_pre_push_hook_is_installed() -> None:
    """WHY THE NARROWING IS ADOPTED WITHOUT A DROP REASON, and the condition it rests on, PINNED.

    Thirteen stock hooks lose ten stages each, and that costs this checkout nothing for one reason
    only: the sole git hook installed here is `pre-commit`, so those stages had nothing to run at.
    The sibling lab takes the identical base and records the identical change as a REAL loss,
    because it has a pre-push hook. The difference is the installation, not the config.

    So the premise is asserted rather than described. Install a pre-push hook and this reds, which is
    the correct moment to decide whether those thirteen should carry `stages:` of their own -- far
    better than the narrowing silently becoming a loss nobody re-derived.
    """
    report = hook_installation(_ROOT, config_name=DEFAULT_CONFIG_NAME)
    assert report.config is not None, f'no {DEFAULT_CONFIG_NAME} at {_ROOT}; this guard is watching nothing'
    live = hooks_dir(_ROOT)
    assert live.is_dir(), f'no hooks directory resolved at {live}; this guard looked nowhere'
    assert not (live / 'pre-push').exists(), (
        f'a pre-push hook is installed at {report.hooks_dir}, so the ten stages the base takes off '
        f'thirteen stock hooks are no longer free here. Re-measure the adoption: either declare '
        f'`stages:` on the hooks this repo wants at push, or record the loss as a DROP with a reason '
        f'the way the sibling lab does in _famconfig.PRECOMMIT_STAGE_MOVE.'
    )


def test_the_pins_the_adoption_moved_are_the_bases_own() -> None:
    """THE OTHER THING THE ADOPTION MOVED, and it is a pin bump that must not pass unremarked.

    This repo held `pre-commit-hooks` at v5.0.0 and `commitizen` at v4.6.0; the base takes the newest
    measured pin of the three consumers, v6.0.0 and v4.13.9. Nothing in the rendered-bytes property
    would distinguish a base whose pin moved from one whose hook LIST moved, so the thing worth
    checking is that every id this repo declared before still RESOLVES at the new pin -- a stock hook
    renamed or removed upstream would otherwise arrive as a config that validates and a guard that
    quietly stopped running.
    """
    hooks = _resolved_stages()
    lost = sorted(set(EXTRA_HOOK_IDS) - set(hooks))
    assert not lost, (
        f'{lost} no longer resolve at the pins the base declares. A stock hook that vanished in an '
        f'upstream release is a guard this repo lost by upgrading, not a line to delete.'
    )
    assert 'commitizen' in hooks, 'commitizen did not resolve at the pin the base declares'


def test_an_anchor_that_names_no_position_is_still_refused() -> None:
    """PLANTED CONTROL FOR THE CAPABILITY THIS ADOPTION RESTS ON, driven through the REAL renderer.

    `Delta.anchored` is what made `.pre-commit-config.yaml` adoptable, and a positioning mechanism
    that accepted anything would be a fork with better manners. So its refusals are planted here
    rather than trusted upstream: an anchor on a line the base REPEATS names no position at all --
    `    hooks:` occurs twice in this base, so picking one would be a coin flip the reader cannot
    see -- and an anchor carrying no lines is a waiver nothing uses.

    This repo's real anchor is the control's other side: it must be accepted, or the test would be
    asserting that anchoring never works.
    """
    base = artefact_base(_PRECOMMIT)
    repeated = '    hooks:'
    assert base.occurrences(repeated) > 1, (
        f'{repeated!r} occurs {base.occurrences(repeated)} time(s) in the base, so the ambiguity this '
        f'plants no longer exists there. Pick another repeated line or drop the control.'
    )
    assert delta_problems(base, Delta(repo=REPO, added=(), dropped={}, ceiling=1, anchored={repeated: ('x',)})), (
        'an anchor on a line the base repeats was accepted -- it names no position, so the rendered '
        'result is a coin flip nothing in the artefact records'
    )
    absent = '      - id: a-hook-no-base-declares'
    assert delta_problems(base, Delta(repo=REPO, added=(), dropped={}, ceiling=1, anchored={absent: ('x',)})), (
        'an anchor on a line the base does not have was accepted -- a declaration outliving its subject'
    )
    empty = '      - id: check-added-large-files'
    assert delta_problems(base, Delta(repo=REPO, added=(), dropped={}, ceiling=1, anchored={empty: ()})), (
        'an anchor adding nothing was accepted -- a waiver nothing uses is as wrong as a capability that disappears'
    )
    assert not delta_problems(base, DELTAS[_PRECOMMIT]), "and this repo's real anchor must be accepted"


def test_the_extra_hook_ids_are_really_declared_in_this_repo() -> None:
    """THE WAIVER'S SUBJECT, read off the live file rather than restated.

    `EXTRA_HOOK_IDS` is the evidence that the append-only limitation actually costs this repo
    something. A named set nobody checks against the artefact is a claim about another tree, and the
    day these ids are dropped from `.pre-commit-config.yaml` the waiver stops having a subject.
    """
    text = (_ROOT / '.pre-commit-config.yaml').read_text(encoding='utf-8')
    declared = {line.strip().removeprefix('- id: ') for line in text.splitlines() if line.strip().startswith('- id: ')}
    assert declared, 'no hook id parsed out of .pre-commit-config.yaml; this guard read nothing'
    missing = sorted(set(EXTRA_HOOK_IDS) - declared)
    assert not missing, (
        f'_famconfig.EXTRA_HOOK_IDS names {missing}, which this repo no longer declares. Either the '
        f'waiver shrank, in which case correct the named set, or the hooks were lost.'
    )


def test_this_guard_can_go_both_ways(tmp_path: Path) -> None:
    """PLANTED CONTROL. A guard green on the real tree could be asserting a constant.

    Both states are driven through the same function the property uses: the rendered artefact is
    written to a scratch path and passes, then one line is edited by hand and it fails.
    """
    base = artefact_base('.gitignore')
    delta = DELTAS['.gitignore']
    planted = tmp_path / '.gitignore'
    planted.write_text(render(base, delta), encoding='utf-8')
    assert inspect_file(planted, base, delta).status == INSTALLED

    planted.write_text(planted.read_text(encoding='utf-8').replace('uv.lock', 'uv.lock.bak'), encoding='utf-8')
    edited = inspect_file(planted, base, delta)
    assert edited.status != INSTALLED, 'a hand edit to a base line did not red -- the guard is theatre'
    assert any('uv.lock' in line for line in edited.offending), edited.offending


def test_the_makefile_check_is_required_rather_than_rendered() -> None:
    """The weaker mode is NAMED so that applying it stayed a decision somebody typed.

    If `Makefile` ever arrives as RENDERED, every recipe in this tree -- the `purity` and `adoption`
    targets, the `rm -rf` clean -- becomes a diff against a family file that cannot hold them. That
    is a change to argue with the kit, not to absorb here.
    """
    assert artefact_base('Makefile').mode == REQUIRED
    assert artefact_base('.gitignore').mode == RENDERED


def test_the_survey_still_sizes_this_repo_the_way_the_adoption_recorded() -> None:
    """The instrument that sized this change, re-run against the adopted file.

    After adoption the survey must find NOTHING to drop: every base line is on disk. Before it, it
    found two -- the slashed `**/__pycache__/` and `*.egg-info/` this repo wrote bare. Pinning the
    post-adoption reading is what makes a silent removal of a base line visible here as well as in
    the main property, by a route that does not go through the rendered bytes.
    """
    survey = measured_delta(_ROOT / '.gitignore', artefact_base('.gitignore'), REPO)
    assert not survey.dropped, (
        f'the survey reports {sorted(survey.dropped)} absent from .gitignore on disk. The adoption '
        f'commit measured zero; a base line has left this file without a declared drop.'
    )
