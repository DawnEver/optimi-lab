"""A PIN in this repo is a NAMED SET; a number may only be a THRESHOLD, and it must say so.

WHAT IS LEFT HERE AFTER THE SPLIT, and it is the whole file: THE VOCABULARY. The walk, the grader,
the floor order, the waiver arm and the planted control ship as `lab_commons.dev.famtests.countpins`
-- carved from this file, because `NAMED-SETS-NOT-COUNTS` is a family rule and the sibling wdg-lab
cites it at six modules while holding no guard at all. What no library may decide is WHICH NAMES
EXCUSE A NUMBER: the suffix set IS the exemption mechanism, so a default shipped by a package
imported for something else would be a waiver this repo never wrote. The kit therefore takes every
word below as a keyword argument with NO DEFAULT, and this file supplies optimi-lab's answers.

WHY THE FILE EXISTS AT ALL, and it is a correction. `NAMED-SETS-NOT-COUNTS` was claimed as enforced
by this repo on 2026-09-15 while citing three modules that merely HAPPEN to use named sets. That is
the declaration-that-lies shape one level up: nothing refused the edit that replaces
`OVERSIZE_PINS = frozenset()` with `OVERSIZE_COUNT = 0`, so the rule was exemplified rather than
enforced, and exemplifying a rule and enforcing it are different facts of which only one is a
mechanism.

THE PROPERTY, stated so a reader can violate it on purpose. In every optimi-lab architecture module
a module-level constant bound to a NUMBER must be named as a threshold; everything else that is
pinned is a SET, a frozenset, a dict or a tuple of names. An integer pin cannot say WHICH row moved,
so a reader cannot tell a delivered capability from a pending one -- and the honest-looking repair
when it disagrees is to edit the digit.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from lab_commons.dev.famtests import countpins

#: The trees this guard answers for, repo-relative. NO DEFAULT UPSTREAM, on purpose: a guessed
#: directory that resolves to nothing REPORTS THE TREE CLEAN. optimi-lab keeps its architecture
#: guards in one place and `src/optimi_lab` has its own; this is that one place.
ROOTS: Final = ('tests/architecture',)

#: Suffixes that make a numeric constant legitimate: its content IS the magnitude, and there is no
#: set of rows a number could have failed to name. Everything else numeric is a count pin.
#:
#: `_HEADROOM` JOINED 2026-09-18 WITH THE FAMILY FLOOR. A headroom is the largest slack a floor may
#: carry before it must be re-measured -- `lab_commons.dev.floors.assert_floor_still_binds`' second
#: argument -- so its content IS a magnitude and there is no set of rows it could have named instead.
#: It is the most magnitude-shaped constant in this tree: it exists precisely to stop a number from
#: silently ceasing to mean anything.
#:
#: THE LIST IS NOT WIDENED TO MAKE ANYTHING GREEN, and the temptation was measured 2026-09-19: driven
#: over lab-commons' own tests the same grader named 21 constants in 14 files -- `SLOW_S`, `_POLL_S`,
#: `NOISE_LINES`, `SEPARATION_FACTOR` -- all genuine magnitudes spelled outside THIS vocabulary. That
#: repo adopted the walk and declined the conviction rather than author fifteen suffixes to fit its
#: own answer. Eight is what optimi-lab earned; a ninth must be argued from a constant here.
THRESHOLD_SUFFIXES: Final = ('_FLOOR', '_CEILING', '_BAND', '_DEPTH', '_MAX', '_MIN', '_LIMIT', '_HEADROOM')

#: THE SAME WORDS STANDING ALONE. A module whose single threshold is just `CEILING` is making
#: exactly the claim a suffix makes, and refusing it taught nothing except to add a prefix. Measured
#: 2026-09-18: `test_the_rules_pages_are_a_ratchet.CEILING` is the one such constant in this tree,
#: and it was a live red the day the suffix rule was written.
THRESHOLD_NAMES: Final = frozenset(suffix.lstrip('_') for suffix in THRESHOLD_SUFFIXES)

#: Modules whose pins are permitted. EMPTY, AND THAT IS A CLAIM rather than an omission: measured
#: 2026-09-19 across 25 optimi-lab architecture modules and 165 constants, this tree convicts
#: nothing, so any exemption would be a waiver with no offender under it. The kit reds on an entry
#: naming a file the walk never reached, which is the arm that keeps an empty set honest rather than
#: merely unfilled.
EXEMPT: Final = frozenset()

#: The floor on the scan, measured 2026-09-15: the architecture modules carry well over twenty
#: module-level constants between them. Below this the walk did not reach the tree it reports on.
#: The readings this floor has been taken against, in order: 15 (the original guess), then 117, then
#: 134, then 147, 163, and 165 today.
#:
#: WHAT EACH RE-TAKE TAUGHT, kept because the lesson is the same one every time. The old number was
#: 15 against 117 -- a slack of 102, which refused only a total collapse -- and adopting
#: `floors.assert_floor_still_binds` is what found it. The kit's remedy is to RE-MEASURE THE FLOOR,
#: never to widen the headroom: a guard that declares five or thirteen bounded numbers is exactly the
#: growth this population is supposed to have, and a headroom widened to absorb it is the arm kept
#: while the guard it stands for is given up.
#:
#: RE-TAKEN 2026-09-19 AT 136, from 120, when the echo-scan adoption brought the reading to 163: that
#: was 43 clear of 120 and past the headroom, and 136 restored the SAME margin of 27 the previous two
#: pairs carried, measured rather than chosen. NOT RE-TAKEN AGAIN for the 165 the stored-reading
#: repair left behind: 165 is 29 above 136 and inside the headroom, and re-taking a floor that still
#: binds is how a floor stops being evidence and becomes a running total.
CONSTANT_FLOOR: Final = 136

#: THE OTHER SIDE OF ``CONSTANT_FLOOR``. 136 + 40 = 176 against today's 165. The headroom is
#: UNCHANGED at 40 across all three re-takes on purpose -- it is the side that refuses, and every
#: breach of it has been answered by moving the floor.
CONSTANT_HEADROOM: Final = 40


def test_no_architecture_constant_is_a_count_pin() -> None:
    """THE CHECK, over every optimi-lab architecture module, with FLOORS FIRST and offenders last."""
    scan = countpins.take_scan(
        Path(__file__).resolve().parents[2],
        roots=ROOTS,
        threshold_suffixes=THRESHOLD_SUFFIXES,
        threshold_names=THRESHOLD_NAMES,
        exempt=EXEMPT,
    )
    countpins.assert_no_constant_is_a_count_pin(scan, floor=CONSTANT_FLOOR, headroom=CONSTANT_HEADROOM)


def test_the_grader_still_convicts_under_this_vocabulary() -> None:
    """THE CONTROL, driven with THIS repo's eight suffixes rather than a fixture set.

    The kit plants the module source and asserts EQUALITY on the real grader's answer, including the
    two assignment spellings. Passing optimi-lab's own vocabulary in is what makes it a control OF
    THIS FILE: a suffix declared here that the grader cannot act on reds here, rather than quietly
    reporting the tree clean.
    """
    countpins.assert_the_grader_still_convicts(
        threshold_suffixes=THRESHOLD_SUFFIXES,
        threshold_names=THRESHOLD_NAMES,
    )
