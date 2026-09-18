"""A PIN in this repo is a NAMED SET; a number may only be a THRESHOLD, and it must say so.

WHY THIS FILE EXISTS, and it is a correction. `NAMED-SETS-NOT-COUNTS` was claimed as enforced by
this repo on 2026-09-15 while citing three modules that merely HAPPEN to use named sets. That is
the declaration-that-lies shape one level up: nothing refused the edit that replaces
`OVERSIZE_PINS = frozenset()` with `OVERSIZE_COUNT = 0`, so the rule was exemplified rather than
enforced. Exemplifying a rule and enforcing it are different facts and only one of them is a
mechanism.

THE PROPERTY, stated so a reader can violate it on purpose. In every architecture module:

* a module-level constant bound to a NUMBER must be named as a threshold -- `_FLOOR`, `_CEILING`,
  `_BAND`, `_DEPTH`, `_MAX`, `_MIN`. Those are quantities whose whole content IS the magnitude.
* everything else that is pinned is a SET, a frozenset, a dict or a tuple of names. An integer
  pin cannot say WHICH row moved, so a reader cannot tell a delivered capability from a pending
  one -- and the honest-looking repair when it disagrees is to edit the digit.

THE FLOOR IS NOT A FORMALITY HERE. This scan's natural answer is the empty set, exactly like a
scan that walked the wrong directory. So it asserts a minimum number of constants ACTUALLY READ
before it is allowed to report clean, and a control plants the count pin and calls the real guard.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Final

from lab_commons.dev import floors

_ROOT: Final = Path(__file__).resolve().parents[2]
_ARCHITECTURE: Final = Path(__file__).resolve().parent

#: Suffixes that make a numeric constant legitimate: its content IS the magnitude, and there is no
#: set of rows a number could have failed to name. Everything else numeric is a count pin.
#:
#: `_HEADROOM` JOINED 2026-09-18 WITH THE FAMILY FLOOR. A headroom is the largest slack a floor may
#: carry before it must be re-measured -- `lab_commons.dev.floors.assert_floor_still_binds`' second
#: argument -- so its content IS a magnitude and there is no set of rows it could have named instead.
#: It is the most magnitude-shaped constant in this tree: it exists precisely to stop a number from
#: silently ceasing to mean anything.
THRESHOLD_SUFFIXES: Final = ('_FLOOR', '_CEILING', '_BAND', '_DEPTH', '_MAX', '_MIN', '_LIMIT', '_HEADROOM')

#: THE SAME WORDS STANDING ALONE. A module whose single threshold is just `CEILING` is making
#: exactly the claim a suffix makes, and refusing it taught nothing except to add a prefix. Measured
#: 2026-09-18: `test_the_rules_pages_are_a_ratchet.CEILING` is the one such constant in this tree,
#: and it was a live red the day the suffix rule was written.
THRESHOLD_NAMES: Final = frozenset(suffix.lstrip('_') for suffix in THRESHOLD_SUFFIXES)

#: The floor on the scan, measured 2026-09-15: the architecture modules carry well over twenty
#: module-level constants between them. Below this the walk did not reach the tree it reports on.
#: RE-MEASURED 2026-09-18: 117 module-level constants across the architecture modules. THE OLD
#: NUMBER WAS 15, and adopting `floors.assert_floor_still_binds` is what found it: a slack of 102
#: refused only a total collapse. The kit`s remedy is to re-measure the floor, never to widen the
#: headroom.
#:
#: RE-MEASURED AGAIN 2026-09-18 AT THE TWO-GUARD ADOPTION, and the other side is what forced it: the
#: unbounded-wait guard brings the reading to 134 and the cited-test guard beside it to 147, which
#: is 57 clear of 90 and past the headroom. THE REMEDY TAKEN IS THE ONE THE KIT NAMES -- the FLOOR
#: moves, the headroom does not. A guard that declares five or thirteen bounded numbers is exactly
#: the growth this population is supposed to have; a headroom widened to absorb it would be the arm
#: kept while the guard it stands for is given up.
CONSTANT_FLOOR: Final = 120

#: THE OTHER SIDE OF ``CONSTANT_FLOOR``. Today's reading is 147 - 120 = 27, the same margin the
#: previous pair carried, so the band is re-measured rather than relaxed.
CONSTANT_HEADROOM: Final = 40


def _architecture_modules() -> list[Path]:
    return sorted(_ARCHITECTURE.rglob('test_*.py'))


def module_constants(module: ast.Module) -> dict[str, ast.expr]:
    """Every module-level UPPER_SNAKE constant in *module*, to the expression it is bound to.

    Annotated (`X: Final = ...`) and plain assignments both count: `Final` is the spelling this
    repo uses and excluding it would exempt every constant in the tree.
    """
    found: dict[str, ast.expr] = {}
    for node in module.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
            found[node.target.id] = node.value
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and node.value is not None:
                    found[target.id] = node.value
    return {name: value for name, value in found.items() if name.lstrip('_').isupper() and name.strip('_')}


def count_pins(module: ast.Module) -> frozenset[str]:
    """THE GUARD. Constants bound to a bare number whose name does not declare them a threshold.

    Each one is a pin an integer stands in for. A reader cannot tell from `2` whether the row that
    left was a delivered capability or a pending one, and when the number disagrees with the tree
    the repair that looks honest is to edit the digit -- which this family has already paid for.
    """
    return frozenset(
        name
        for name, value in module_constants(module).items()
        if isinstance(value, ast.Constant)
        and isinstance(value.value, (int, float))
        and not isinstance(value.value, bool)
        and not name.upper().endswith(THRESHOLD_SUFFIXES)
        and name.upper().strip('_') not in THRESHOLD_NAMES
    )


def test_no_architecture_constant_is_a_count_pin() -> None:
    """THE CHECK, over every architecture module, with the number of constants read asserted first."""
    modules = _architecture_modules()
    trees = {path: ast.parse(path.read_text(encoding='utf-8')) for path in modules}
    read = sum(len(module_constants(tree)) for tree in trees.values())
    floors.assert_floor(read, floor=CONSTANT_FLOOR, what='COUNT-PIN (architecture constants)')
    floors.assert_floor_still_binds(
        read, floor=CONSTANT_FLOOR, headroom=CONSTANT_HEADROOM, what='COUNT-PIN (architecture constants)'
    )
    offenders = {
        path.relative_to(_ROOT).as_posix(): sorted(found) for path, tree in trees.items() if (found := count_pins(tree))
    }
    assert not offenders, (
        f'{offenders} pin something with a NUMBER. An integer cannot say which row moved: pin the named '
        f'set instead, or -- if the number IS the quantity -- rename it to end in one of '
        f'{THRESHOLD_SUFFIXES}, which is a claim about magnitude rather than about membership.'
    )


def test_the_guard_names_a_planted_count_pin() -> None:
    """THE CONTROL. Plant the count pin and call the REAL guard, not a copy of the comparison.

    The three shapes that must be told apart: a count pin (named), a threshold (exempt by its own
    suffix, which is the declaration the exemption rests on), and a named set (never numeric).
    """
    planted = ast.parse(
        'from typing import Final\n'
        'OVERSIZE_PINS: Final = 2\n'
        'RETIRED_COUNT = 11\n'
        'SOURCE_FILE_FLOOR: Final = 15\n'
        'MODULE_SIZE_BAND = 320\n'
        "NAMED_PINS: Final = frozenset({'a'})\n"
        'LAZY_IMPORT_PINS: Final[dict[str, str]] = {}\n'
        'lowercase_is_not_a_constant = 3\n'
    )
    assert count_pins(planted) == frozenset({'OVERSIZE_PINS', 'RETIRED_COUNT'}), (
        f'the guard answered {sorted(count_pins(planted))} on planted input. It must name both count '
        f'pins, exempt both thresholds by suffix, and never name a set-valued pin or a lowercase name.'
    )


def test_the_guard_reads_both_assignment_spellings() -> None:
    """A `Final` annotation is this repo's usual spelling; missing it would exempt the whole tree."""
    annotated = ast.parse('from typing import Final\nPINNED: Final = 4\n')
    plain = ast.parse('PINNED = 4\n')
    assert count_pins(annotated) == count_pins(plain) == frozenset({'PINNED'}), (
        'the guard reads one assignment spelling and not the other, so the shape it misses is the one '
        'a violation would arrive in.'
    )
