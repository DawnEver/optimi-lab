"""optimi-lab ADOPTS the shared rules registry -- 21 of 32, with the other 11 named as gaps.

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

WHY ONLY 16. optimi-lab is a LIBRARY, not a workflow: it has no gate runner, no agent hooks, no
production CLI, no vendor arbitration and no physical units. Adopting those rules by writing a
mechanism that checks nothing would be worse than the gap -- so they are DECLARED ABSENT, by name,
under a ceiling that may only go down. Each entry in `_ABSENT` has its reason in the comment above
it.

TWO RULES ARRIVED UPSTREAM AND WERE ADOPTED ON THE SAME DAY, 2026-09-16:
`NO-CJK-IN-TRACKED-SOURCE` and `INJECTED-DOC-WIDTH-CEILING`, each driving the SHARED scanner in
`lab_commons.dev.cjk` / `lab_commons.dev.docwidth` over this tree's own corpus. Both measured CLEAN
here (53 files scanned, zero CJK; two injected docs, zero lines past 120 columns), so both declared
sets are EMPTY and each carries a FLOOR pinned at its measurement -- an empty declaration over an
unread corpus is the one way those guards could be vacuous. Two more arrived the same day with no
subject in this repo and went to `_ABSENT`, which is why the ceiling below moved UP; that move has
its own note where the number lives.

THE AUDIT OF 2026-09-15, because a count that moved without saying why is the thing this file
exists to prevent. The first version of this file claimed 12 of 28. Three specific overstatements
were alleged and each was checked against the code rather than taken on trust:

* **Three rules rest on a module that cannot be collected** -- REFUTED. The module is this one and
  the three are `NAMED-SETS-NOT-COUNTS`, `ESCAPE-HATCH-CEILING`, `RATCHET-TWO-SIDES`. It imports
  `lab_commons.dev`, which is genuinely missing from one shared venv's STALE installed
  `lab_commons` -- but that is a fact about an install, not about the file, the dev extra declares
  the dependency, and a missing import here is a COLLECTION ERROR, which is the loudest failure
  pytest has. A mechanism that reds when its dependency is absent is behaving.
* **`_SURFACE` plants nothing** -- CONFIRMED, and it was cited for SEVEN rules. Fixed rather than
  demoted: its five scans now route through `assert_floor`, `second_homes`, `oversize`, `skipping`
  and `pin_gap`, and a control drives each of those REAL functions on planted input.
* **`OVERSIZE_PINS` and `LAZY_IMPORT_PINS` are empty, so the ratchet never executes** -- CONFIRMED.
  Both are empty as correct MEASUREMENTS, so the live comparison ran an empty set against an empty
  set. Seeding a fake pin would be the waiver-nothing-uses side of the same defect; the two-sided
  control on `pin_gap` is what makes the empty answer mean clean rather than unread.

A FOURTH one was found that the review had not named, and it is the one that cost a rule.
`NAMED-SETS-NOT-COUNTS` cited three modules that merely OBEY it: nothing refused replacing a named
pin with an integer. A rule exemplified by the code is not a rule enforced by it, so it now cites
`test_a_pin_is_a_named_set.py`, which refuses a numeric module-level constant unless its name
declares it a threshold.

FOUR GAPS CLOSED in the same pass, each with a floor and a planted control: `FIX-THE-CAUSE` (the
one-name-one-home scan was already here and unclaimed), `RETIRED-NAMES-REGISTERED` (seven names
the two breaking refactors deleted, and the three docstrings still writing them), `MEMORY-SHAPE`
and `DOCS-SPLIT`. So the ceiling falls 16 -> 12 -- and had the audit gone the other way it would
have RISEN, which is the correct direction for an honesty pass and is not a thing to hide.
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
_PIN_SHAPE: Final = 'tests/architecture/test_a_pin_is_a_named_set.py'
_RETIRED: Final = 'tests/architecture/test_a_retired_spelling_stays_retired.py'
_MEMORY: Final = 'tests/architecture/test_memory_lives_under_a_date.py'
_RULES_RATCHET: Final = 'tests/architecture/test_the_rules_pages_are_a_ratchet.py'
_NO_CJK: Final = 'tests/architecture/test_no_cjk_in_tracked_source.py'
_DOC_WIDTH: Final = 'tests/architecture/test_injected_doc_width_ceiling.py'
_HOOKS: Final = 'tests/architecture/test_the_declared_hooks_are_installed.py'
_BOUNDED: Final = 'tests/architecture/test_a_bounded_wait_names_its_remedy.py'
_ORIGIN: Final = 'tests/architecture/test_this_checkout_is_visible_on_origin.py'

#: THIS REPO'S MECHANISMS -- a rule ID to the tracked file(s) in THIS tree that refuse a violation
#: of it. Not the registry's own rows: those name motronics paths, which resolve in exactly one
#: checkout on earth and would grade this repo against a tree it does not have.
_MECHANISMS: Final[dict[str, tuple[str, ...]]] = {
    # The package docstring claims a three-distribution runtime; the guard reads the dependency
    # table and the import graph instead of believing it.
    'DECLARATION-LIES': (_PURITY,),
    # Every guard is called on PLANTED input: a dropped dependency's import, a silent module, a
    # second home for one name, a deferred import, an oversize module, a skip, a count pin, a
    # forked memory path, a stale frontmatter date, a resurrected spelling, each ratchet direction.
    'PLANTED-CONTROL': (_PURITY, _SURFACE, _STRANGER, _PIN_SHAPE, _RETIRED, _MEMORY, _RULES_RATCHET),
    # No scanner reports green over a file list shorter than its measured floor, and `assert_floor`
    # is one function with one control rather than a line repeated per scan.
    'FLOOR-ON-EVERY-SCAN': (_PURITY, _SURFACE, _PIN_SHAPE, _RETIRED, _MEMORY, _RULES_RATCHET),
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
    # A numeric module-level constant in any architecture module is REFUSED unless its name
    # declares it a threshold. The three modules previously cited here merely obeyed the rule,
    # which is exemplification and not enforcement -- see the audit note in the docstring.
    'NAMED-SETS-NOT-COUNTS': (_PIN_SHAPE,),
    # The absent set carries `_ABSENT_CEILING`, which may only go down.
    'ESCAPE-HATCH-CEILING': (_ADOPTION,),
    # Two-sided pins: an arrival reds, and so does a waiver nothing uses. `pin_gap` and
    # `ratchet_breaks` each answer in BOTH directions and each has a control that drives them so,
    # which is what the live empty pin sets cannot do for themselves.
    'RATCHET-TWO-SIDES': (_ADOPTION, _SURFACE, _RULES_RATCHET),
    # One public name has one home: a second spelling is how a patched symptom outlives its cause,
    # and the scan reads the merged tree, which is the only place the clash is visible.
    'FIX-THE-CAUSE': (_SURFACE,),
    # Seven names the two breaking refactors deleted, each with its replacement, and a scan that
    # reaches PROSE -- where a retired spelling survives longest -- as well as code.
    'RETIRED-NAMES-REGISTERED': (_RETIRED,),
    # `.claude/memory/YYYY/MM/DD/`, plus the frontmatter date agreeing with the path it sits on.
    'MEMORY-SHAPE': (_MEMORY,),
    # The always-loaded rule pages are pinned per file, ratcheting down only.
    'DOCS-SPLIT': (_RULES_RATCHET,),
    # The shared scanner over this tree's whole tracked corpus, with the declared set EMPTY --
    # measured 2026-09-16 at 53 files scanned and zero CJK characters. Adopted at zero, so the
    # declaration's job is to refuse the FIRST arrival rather than to record a debt.
    'NO-CJK-IN-TRACKED-SOURCE': (_NO_CJK,),
    # The same shape over the injected-doc corpus (here: the two `.claude/rules/` pages), measured
    # 2026-09-16 at zero lines past 120 columns. It is the WIDTH dimension the line-count ratchet
    # in `_RULES_RATCHET` cannot see, over the same pages.
    'INJECTED-DOC-WIDTH-CEILING': (_DOC_WIDTH, _RULES_RATCHET),
    # CLOSED 2026-09-16, and it was VIOLATED here rather than merely unenforced: the reason on
    # record said this repo has no hooks directory, while `.pre-commit-config.yaml` declared 19
    # hooks and ZERO were installed in the directory git consults. The mechanism is the family's
    # (`lab_commons.dev.hook_install`, which ASKS GIT for that directory instead of string-building
    # it); what is this repo's is the tree asked about, the one-stage named pin, and controls that
    # plant absent / installed / foreign in a real repository.
    'HOOKS-ARE-WIRED': (_HOOKS,),
    # CLOSED 2026-09-16. The old reason -- "no gate runner" -- answered about a RUNNER while the
    # rule is about a WAIT, and this package starts subprocesses. The wall is driven against a
    # planted GRANDCHILD that really inherits stdout (the shape `subprocess.run`'s timeout cannot
    # bound), the reaper is asked to kill this process's own tree and must refuse while still
    # reaping a planted stranger, and the refusal must name a DIFFERENT remedy in each of its three
    # states.
    'REFUSAL-NAMES-THE-REMEDY': (_BOUNDED,),
    # CLOSED 2026-09-16, and the first honest run of it found real debris: `fix/p0-integration-
    # blockers`, a branch with no origin counterpart, no upstream and no worktree. The old reason
    # was a wrong conclusion from a true premise -- the push obligation IS a fact about origin, and
    # git answers it without a network call. Named sets on both sides, and the stale-local-trunk
    # trap planted on real repositories with a real remote.
    'SHARED-CHECKOUT': (_ORIGIN,),
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
#: * WORKFLOW RULES WITH NO MACHINERY HERE -- optimi-lab has no gate runner, no retry wrapper and
#:   no agent deny-list: VERDICT-BAR-IS-THE-INCREMENT,
#:   NETWORK-RETRY-THEN-REPORT, AGENT-GUARD, and -- added 2026-09-16 -- the two
#:   rules ENV-MUTATION-THROUGH-THE-DOOR and ONE-BOX-ONE-LOCK: both are about a SHARED box and a
#:   SHARED environment, and this repo has neither. Its `.venv` is its own, nothing here issues a
#:   verdict another party cites, and there is no runner to queue on a box lock. A mechanism for
#:   either would be a check over an empty set.
#:
#: The fourth kind is gone: the REAL GAPS this set used to hold (FIX-THE-CAUSE,
#: RETIRED-NAMES-REGISTERED, MEMORY-SHAPE, DOCS-SPLIT) were closed on 2026-09-15 and their names
#: were deleted here in the same edit that lowered the ceiling, which is the only way this number
#: is allowed to move.
_ABSENT: Final = frozenset({
    'BAR-IS-A-CONSTANT',
    'IMPLEMENT-EVERYTHING',
    'PRODUCTION-ENTRY-POINT',
    'TOLERANCE-CARRIES-A-UNIT',
    'UNITS-GO-THROUGH-PINT',
    'REGISTRY-OWNS-THE-DECISION',
    'VERDICT-BAR-IS-THE-INCREMENT',
    'NETWORK-RETRY-THEN-REPORT',
    'AGENT-GUARD',
    'ENV-MUTATION-THROUGH-THE-DOOR',
    'ONE-BOX-ONE-LOCK',
})

#: The ceiling on the gap, MEASURED the day this file was written. It may only go DOWN FOR A RULE
#: THIS REPO ALREADY CARRIED: the hand that builds a mechanism deletes the name above and lowers
#: this number in the same edit.
#:
#: IT MOVED UP ONCE, 12 -> 14 on 2026-09-16, and the reason is stated here rather than left to be
#: read off the digit. The registry grew by FOUR rules that day. Two of them --
#: NO-CJK-IN-TRACKED-SOURCE and INJECTED-DOC-WIDTH-CEILING -- were ADOPTED in the same commit, with
#: mechanisms above, so they never touched this number. The other two are about a shared box and a
#: shared environment, neither of which exists here, and a ceiling that cannot admit an upstream
#: rule with no subject here would be cleared by the only move left: deleting it from `_ABSENT`,
#: which is the silent gap this file exists to prevent. So the ceiling absorbs the two it must and
#: nothing else -- every gap this repo could close is still on record, and none was reopened.
#:
#: AND DOWN AGAIN, 14 -> 11 later the same day, when three of the six workflow rules stopped being
#: machinery this repo does not have. Their MECHANISMS moved into `lab_commons.dev`
#: (`hook_install`, `bounded`, `checkout`) and this repo wired each one to its own tree.
#: HOOKS-ARE-WIRED was the sharpest: the reason on record claimed there was no subject here, and
#: there were 19 declared hooks with none installed.
_ABSENT_CEILING: Final = 11


def _adoption() -> Adoption:
    """The repo's adoption: its own mechanisms, plus the gaps it has on record."""
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
