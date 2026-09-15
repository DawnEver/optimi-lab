"""optimi-lab ADOPTS the shared rules registry -- 12 of 28, with the other 16 named as gaps.

WHY THIS FILE EXISTS. `lab_commons.dev.rules` holds one canonical copy of the development rules
that are not about any repo's subject. Before this file, optimi-lab had no `.claude/` at all: its
rules were whatever the last reader remembered, so there was no statement to violate and nothing
that could red. The registry's `assert_adopted` is what turns "we follow the house rules" into a
check -- it resolves each rule this repo claims to enforce against a mechanism that must be a
TRACKED file in THIS tree.

WHAT IT PROVES, stated exactly, because overclaiming it would be the defect it guards against: every
rule is ACCOUNTED FOR -- enforced with a live mechanism, or declared absent -- and no ID is a typo.
It does NOT run the mechanisms and cannot say they pass. An existing red mechanism is still an
existing mechanism; whether it is green is the suite's business.

BOTH SETS ARE PINNED BY NAME AND TYPED OUT. Deriving `_ENFORCED` from `RULES` would silently absorb
a rule added upstream instead of making this repo decide about it -- and an integer pin could not
say WHICH rule moved, which is the failure mode the family already paid for once.

WHY ONLY 12. optimi-lab is a LIBRARY, not a workflow: it has no gate runner, no agent hooks, no
memory tree, no production CLI, no vendor arbitration and no physical units. Adopting those rules
by writing a mechanism that checks nothing would be worse than the gap -- so they are DECLARED
ABSENT, by name, under a ceiling that may only go down. Each entry in `_ABSENT` has its reason in
the comment above it, and the honest ones (`MEMORY-SHAPE`, `DOCS-SPLIT`, `FIX-THE-CAUSE`) are gaps
this repo could close rather than facts about its subject.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from lab_commons.dev.profile import RepoProfile
from lab_commons.dev.rules import RULES, Adoption, TestPath, assert_adopted, unadopted, waived

_ROOT: Final = Path(__file__).resolve().parents[2]

_PROFILE: Final = RepoProfile(
    app_name='optimi_lab',
    package='optimi_lab',
    root=_ROOT,
    lint_config=_ROOT / 'pyproject.toml',
)

_PURITY: Final = 'tests/architecture/test_the_runtime_stays_pure.py'
_SURFACE: Final = 'tests/architecture/test_the_public_surface_is_declared.py'
_STRANGER: Final = 'tests/architecture/test_a_declared_set_refuses_a_stranger.py'
_ADOPTION: Final = 'tests/architecture/test_optimi_lab_adopts_the_shared_registry.py'

#: THIS REPO'S MECHANISMS -- a rule ID to the tracked file(s) in THIS tree that refuse a violation
#: of it. Not the registry's own rows: those name motronics paths, which resolve in exactly one
#: checkout on earth and would grade this repo against a tree it does not have.
_MECHANISMS: Final[dict[str, tuple[str, ...]]] = {
    # The package docstring claims a three-distribution runtime; the guard reads the dependency
    # table and the import graph instead of believing it.
    'DECLARATION-LIES': (_PURITY,),
    # The purity guard is called on a PLANTED import of each dependency this package dropped.
    'PLANTED-CONTROL': (_PURITY,),
    # Both scanners refuse to report green over a file list shorter than their measured floor.
    'FLOOR-ON-EVERY-SCAN': (_PURITY, _SURFACE),
    # No upper bound on a runtime requirement, and lab-commons is a bare git+https URL, never a sha.
    'LATEST-DEPENDENCIES': (_PURITY,),
    # `__all__` in every shipped module, and one public name with exactly one home.
    'PUBLIC-SURFACE-DECLARED': (_SURFACE,),
    # No import inside a function body; the exemption map is empty and is checked both ways.
    'NO-LAZY-IMPORT': (_SURFACE,),
    # A size band with the oversize set pinned by name rather than by a count.
    'MODULE-SIZE-ALARM': (_SURFACE,),
    # A skip records nothing, so the suite may not contain one.
    'XFAIL-NOT-SKIP': (_SURFACE,),
    # An unregistered surrogate key or sample kind RAISES `Refusal` and the message quotes the set.
    'UNSUPPORTED-RAISES': (_STRANGER,),
    # Every pin in this repo is a named set: the adoption sets here, the oversize and lazy-import
    # pins in the surface guard, the runtime distribution set in the purity guard.
    'NAMED-SETS-NOT-COUNTS': (_ADOPTION, _SURFACE, _PURITY),
    # The absent set carries `_ABSENT_CEILING`, which may only go down.
    'ESCAPE-HATCH-CEILING': (_ADOPTION,),
    # Two-sided pins: an arrival reds, and so does a waiver nothing uses -- in the surface guard's
    # oversize and lazy-import sets, and in both adoption sets below.
    'RATCHET-TWO-SIDES': (_ADOPTION, _SURFACE),
}

_ENFORCED: Final = frozenset(_MECHANISMS)

#: NOT ENFORCED HERE, each with its reason. Three kinds, and the kind matters:
#:
#: * NO SUBJECT IN THIS REPO -- a mechanism would check an empty set, which is the vacuous green
#:   this family refuses: BAR-IS-A-CONSTANT and IMPLEMENT-EVERYTHING (no reference ladder and no
#:   accuracy matrix: a metric here is exact arithmetic, not a measurement against a vendor),
#:   PRODUCTION-ENTRY-POINT (the API IS the entry point; there is no CLI), TOLERANCE-CARRIES-A-UNIT
#:   and UNITS-GO-THROUGH-PINT (`Variable.unit` is a free-text LABEL this package never computes
#:   with -- it is SI-agnostic pure numerics and pint is not a dependency it may take),
#:   REGISTRY-OWNS-THE-DECISION (there is one registry and the surrogate keys already live in it).
#: * WORKFLOW RULES WITH NO MACHINERY HERE -- optimi-lab has no gate runner, no hooks directory and
#:   no agent deny-list: SHARED-CHECKOUT, VERDICT-BAR-IS-THE-INCREMENT, REFUSAL-NAMES-THE-REMEDY,
#:   NETWORK-RETRY-THEN-REPORT, HOOKS-ARE-WIRED, AGENT-GUARD.
#: * REAL GAPS, closable without importing anything -- these are work items, not facts:
#:   FIX-THE-CAUSE and RETIRED-NAMES-REGISTERED (no `retired.py` registry exists yet, so the two
#:   renames in `refactor(core)!` are recorded only in a commit message), MEMORY-SHAPE (no
#:   `.claude/memory/` tree yet), DOCS-SPLIT (the two rules pages exist but no line ratchet keeps
#:   mechanism out of them).
_ABSENT: Final = frozenset({
    'BAR-IS-A-CONSTANT',
    'IMPLEMENT-EVERYTHING',
    'PRODUCTION-ENTRY-POINT',
    'TOLERANCE-CARRIES-A-UNIT',
    'UNITS-GO-THROUGH-PINT',
    'REGISTRY-OWNS-THE-DECISION',
    'SHARED-CHECKOUT',
    'VERDICT-BAR-IS-THE-INCREMENT',
    'REFUSAL-NAMES-THE-REMEDY',
    'NETWORK-RETRY-THEN-REPORT',
    'HOOKS-ARE-WIRED',
    'AGENT-GUARD',
    'FIX-THE-CAUSE',
    'RETIRED-NAMES-REGISTERED',
    'MEMORY-SHAPE',
    'DOCS-SPLIT',
})

#: The ceiling on the gap, MEASURED the day this file was written. It may only go DOWN: the hand
#: that builds a mechanism deletes the name above and lowers this number in the same edit.
_ABSENT_CEILING: Final = 16


def _adoption() -> Adoption:
    """This repo's adoption: its own mechanisms, plus the gaps it has on record."""
    mechanisms = {rule: tuple(TestPath(path) for path in paths) for rule, paths in _MECHANISMS.items()}
    return Adoption(app_name='optimi_lab', mechanisms=mechanisms, declared_absent=_ABSENT)


def test_optimi_lab_adopts_the_shared_registry() -> None:
    """THE CHECK. Every rule is enforced here by a tracked mechanism, or named as a gap."""
    assert_adopted(_PROFILE, _adoption())


def test_the_two_sets_account_for_the_registry_exactly() -> None:
    """A rule added upstream must be typed into one set or the other before anything else passes."""
    accounted = _ENFORCED | _ABSENT
    live = {rule.id for rule in RULES}
    assert accounted == live, (
        f'rules this repo says nothing about: {sorted(live - accounted)}; names that are not rules: '
        f'{sorted(accounted - live)}. A new universal rule is a decision optimi-lab makes for itself, '
        f'not a line that appears in a list nobody read.'
    )


def test_the_absent_set_may_only_shrink() -> None:
    """The ceiling on the escape hatch: closing a gap lowers the number in the same edit."""
    assert len(_ABSENT) <= _ABSENT_CEILING, (
        f'{len(_ABSENT)} rules are declared absent, above the {_ABSENT_CEILING} ceiling. A waiver nobody '
        f'must justify is how a check reaches zero without anything being fixed.'
    )


def test_the_gap_is_visible_rather_than_silent() -> None:
    """A declared gap and an unexamined rule are different facts, and only one is acceptable."""
    adoption = _adoption()
    assert unadopted(RULES, adoption) == (), 'a rule is cited by neither set; the check above says which.'
    assert set(waived(RULES, adoption)) == _ABSENT


def test_every_mechanism_this_repo_names_is_a_file_in_it() -> None:
    """The control for the check above, read off the filesystem rather than off git.

    `assert_adopted` resolves a mechanism against `git ls-files`, which answers for what is
    COMMITTED. This answers for what is on disk. The two disagree exactly when a mechanism was
    written and never added -- a file nobody else has -- and that is a state worth naming directly.
    """
    missing = sorted({path for paths in _MECHANISMS.values() for path in paths if not (_ROOT / path).is_file()})
    assert not missing, (
        f'{missing} are named as mechanisms and do not exist. A rule whose mechanism is a path that '
        f"resolves to nothing is prose wearing a guarantee's clothes."
    )
