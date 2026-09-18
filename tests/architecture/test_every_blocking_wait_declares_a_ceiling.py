"""NO CHILD IS WAITED ON WITHOUT A CEILING -- the scan, over a tree that currently has nothing to find.

WHY A LINT AND NOT A REVIEW ITEM. A hung child parks whatever ran it at 0% CPU until a deadlock
detector fires, and the verdict written ~28 minutes later is INCONCLUSIVE -- evidence about nothing.
That is not a failure a reader can diagnose from the log: a blocked wait, a spin and a crash all end
the same run, and only the CPU trace tells them apart. The originating repo in this family found
ELEVEN such sites at once, four of them added the same day the three INCONCLUSIVE verdicts appeared,
which is what makes this a scan rather than a code-review habit -- an exception firing on the first
offender would have reported one of them eleven times.

THE WHOLE READING IS THE FAMILY'S. `lab_commons.dev.famtests.untimedwaits` holds the walk, the AST
reader, the two named sets of calls it convicts, the arm and the planted control, and
`lab_commons.dev.floors` holds BOTH sides of the file floor. What is left here is the half the kit
refuses to guess -- every repo-shaped fact arrives as a keyword argument with NO DEFAULT -- and this
file is where optimi-lab's three answers are written down: which trees are walked, what a file in
each is called, and the two numbers that bound the population.

THE ROOTS ARE THE ARGUMENT A GUESS DESTROYS SILENTLY, and that is why they are declared here rather
than defaulted upstream. A `(directory, glob)` pair naming a tree that is not there does not raise:
it walks nothing, finds nothing and reports CLEAN. Measured 2026-09-18 across this family the
candidate populations differ by 8x -- 443 modules in the sibling lab against 56 here -- so one
repo's number handed to another is a floor nothing measured.

THIS TREE HAS ZERO OFFENDERS TODAY, AND THAT IS PRECISELY WHY THE FLOOR IS NOT OPTIONAL. Measured
2026-09-18: 57 modules read, 2 blocking call sites, both already carrying a `timeout=` -- a
`git ls-files` at 60 s and a `Popen.wait` at 20 s. A guard whose
offender set is empty reports exactly what a broken walk reports, so the only thing separating "this
repo bounds its waits" from "this scan stopped reading" is the number below -- and the arm binds it
BEFORE it looks at an offender, on both sides.

WHAT THIS DOES NOT PROVE, stated so the row cannot overclaim. A `timeout=` keyword does not prove the
ceiling is SIZED: a call bounded at 86400 clears every arm here. Sizing is
`lab_commons.dev.bounded.wall_reason`'s subject, driven by
`tests/architecture/test_a_bounded_wait_names_its_remedy.py` beside this file. `shell=True` is also
out of scope -- ruff's `S602` owns it, and one defect with two mechanisms has two messages and no
owner.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from lab_commons.dev.famtests import untimedwaits

#: This checkout: `tests/architecture/<this file>` sits two directories below it.
ROOT: Final = Path(__file__).resolve().parents[2]

#: THE TREES WALKED AND WHAT A FILE IN EACH IS CALLED. `src/` is in because a library function that
#: blocks forever hangs every caller including the suite; `scripts/` takes EVERY module rather than a
#: `test_*` glob, because a helper imported by a hook hangs exactly as hard as the hook; and `tests/`
#: takes every module for the same reason -- a fixture is not named `test_*` and waits just as long.
#: `examples/` is deliberately absent and the reason is measured rather than assumed: it holds no
#: `.py` file at all in this checkout, so declaring it would add a root that walks nothing.
ROOTS: Final = (('src', '*.py'), ('scripts', '*.py'), ('tests', '*.py'))

#: FILES WHOSE OWN SOURCE NAMES THESE CALLS AS DATA. Empty, and honestly so: the planted control
#: below writes its offenders into a `tmp_path` through the kit rather than into this tree, so no
#: file here carries an unbounded wait as a fixture. A name added to this tuple must be asserted to
#: exist -- an exemption naming a deleted file covers nothing while still reading as a decision.
EXEMPT: Final[tuple[str, ...]] = ()

#: MEASURED 2026-09-18: the walk reads 57 modules across the three trees. The floor sits below that
#: with room for ordinary deletion, and far above the zero a mistyped root returns.
FILE_FLOOR: Final = 48

#: THE OTHER SIDE, and the side no copy in this family ever wrote. This is how far past its floor the
#: population may grow before the floor is RE-MEASURED -- never how much slack the floor may be
#: given. At 57 read against 48 the margin is 9, so 15 leaves room for a tranche of new modules and
#: still refuses a floor that has stopped separating a clean scan from a broken walk.
FILE_HEADROOM: Final = 15


def scan() -> untimedwaits.UntimedScan:
    """One walk over the declared trees, with this repo's three answers supplied."""
    return untimedwaits.take_scan(ROOT, roots=ROOTS, exempt=EXEMPT)


def test_every_exemption_names_a_file_that_is_here() -> None:
    """A waiver nothing uses is as wrong as a capability that disappeared -- so assert it, empty or not."""
    missing = sorted(name for name in EXEMPT if not (ROOT / name).exists())
    assert not missing, (
        f'{missing} are exempt from a scan they are no longer part of. An exemption naming a deleted '
        f'file is a silent widening: delete the row in the same edit that deleted the file.'
    )


def test_no_blocking_wait_in_this_tree_is_unbounded() -> None:
    """THE CHECK, with BOTH sides of the file floor bound FIRST so an empty walk cannot read as clean."""
    untimedwaits.assert_every_wait_is_bounded(scan(), floor=FILE_FLOOR, headroom=FILE_HEADROOM)


def test_the_scanner_still_convicts_a_planted_unbounded_wait(tmp_path: Path) -> None:
    """THE CONTROL, both ways, through the SHIPPED scanner and over THIS repo's declared roots.

    A lint that has never been shown to fire proves nothing when it is green, and one that refuses
    every subprocess it sees is deleted rather than obeyed. The kit plants both at once: an unbounded
    `run` beside a bounded one, a bare `Popen().wait()` beside the `Popen` itself, and a POSITIONAL
    `wait(5)`, which is bounded and which a rule copied from the blocking calls would wrongly catch.
    """
    untimedwaits.assert_the_scanner_still_convicts(tmp_path, roots=ROOTS)
