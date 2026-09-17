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
carries a ceiling and a delta re-stating a base line is refused at render time. The third side is
the one a single repo can still get wrong -- an artefact quietly left unadopted -- so
`_famconfig.PRECOMMIT_BLOCKED` is a DECLARATION rather than an absence, and the test below re-drives
the refusal it names. The day the renderer can express a nested addition, that test reds and the
artefact gets adopted; a waiver nothing uses is as wrong as a capability that disappears.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pytest
from _famconfig import DELTAS, EXTRA_HOOK_IDS, PRECOMMIT_BLOCKED, REPO
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

#: THE TREE UNDER TEST, not the working directory: a scan rooted at the CWD answers, plausibly,
#: about somebody else's checkout.
_ROOT: Final = Path(__file__).resolve().parents[2]


def test_every_family_base_is_either_declared_or_refused() -> None:
    """COMPLETENESS, and it is what stops a new base arriving unnoticed.

    A test that only ever checks the artefacts this repo happens to declare cannot tell a base this
    repo has not adopted from one the kit has not published. So the two named sets must together
    cover the kit's, exactly -- and neither may name an artefact the kit does not own.
    """
    accounted = set(DELTAS) | set(PRECOMMIT_BLOCKED)
    assert accounted == set(BASES), (
        f"{REPO} accounts for {sorted(accounted)} against the kit's {sorted(BASES)}. A base the kit "
        f'publishes and this repo neither adopts nor refuses is an artefact drifting with nobody '
        f'saying so: add a Delta to _famconfig.DELTAS, or a row to PRECOMMIT_BLOCKED with the reason.'
    )
    assert not (set(DELTAS) & set(PRECOMMIT_BLOCKED)), 'an artefact cannot be both adopted and refused'
    assert all(reason.strip() for reason in PRECOMMIT_BLOCKED.values()), (
        'a refusal with an empty reason is an absence wearing a declaration. A removal and a drift '
        'are the same bytes on disk; the reason is the only thing that tells them apart.'
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


def test_the_refused_artefact_is_still_actually_refused() -> None:
    """THE WAIVER'S OTHER SIDE. A reason that outlives its cause is a declaration that lies.

    This drives the REAL renderer with the minimal delta `.pre-commit-config.yaml` would need -- one
    structural YAML line that any second repo entry repeats -- and asserts it is still refused. When
    `famconfig` learns to express a nested addition this reds, and the remedy is to adopt the
    artefact rather than to delete this test.
    """
    base = artefact_base('.pre-commit-config.yaml')
    assert base.mode == RENDERED, 'the refusal below is about RENDERED byte equality'
    assert EXTRA_HOOK_IDS, 'the refusal claims this repo runs extra stock hook ids; it must name them'
    structural = '    hooks:'
    assert structural in base.lines, (
        f'{structural!r} is no longer a base line, so the collision this waiver names has moved. '
        f'Re-measure the refusal before trusting it.'
    )
    probe = Delta(repo=REPO, added=(structural,), dropped={structural: 'redeclared with the extra ids'}, ceiling=1)
    assert delta_problems(base, probe), (
        'a second repo entry is now expressible, so _famconfig.PRECOMMIT_BLOCKED is stale. Adopt '
        '.pre-commit-config.yaml: declare the eight EXTRA_HOOK_IDS as a Delta, render it, and delete '
        'this test with the waiver it guards.'
    )


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
