"""NO CLAUSE IN THIS TREE CHECKS ITS OWN ECHO -- the shape that makes an assertion structurally unfailable.

WHY optimi-lab NEEDS THIS AND WHY IT WAS THE LAST CAPABILITY LEFT UNCLAIMED. The whole architecture
tree here is a consumer of SHARED assertion bodies: nearly every module under this directory hands
`lab_commons.dev` a repo-shaped word with no default -- a BAR, a NOUN, a tier, a remedy -- and then
reads a verdict built from that word. That is the exact precondition `lab_commons.dev.famtests.
echoedtoken` was written for: the moment the producer INTERPOLATES the token the asserter searches
for, the clause holds for any string whatever, and it still reads as a check. The kit convicted the
shape inside its OWN `boundedremedy` arm, driven green on `'ZZZ_NO_SUCH_TIER_ANYWHERE'`.

THE CAPABILITY WAS PUBLISHED AND THIS REPO CONSUMED NOTHING. Measured 2026-09-19 against
`0.2.2.dev140+g73d3ec99b`: `echoedtoken` had no importer anywhere in optimi-lab and no roster row.
A published kit body with no consumer is the stale-roster defect pointing the other way -- a ratchet
has two sides, and a capability nobody claims disappears as quietly as one that is deleted.

THE ANSWER TODAY IS ZERO OFFENDERS, AND THAT IS WHY THE FLOOR IS THE POINT. An offender scan's
natural reading is the empty set, which is also what a walk of the wrong directory returns. So the
arm below counts FUNCTIONS READ first and refuses a reading that proves nothing, and the control
plants the shape in a temporary tree and calls the REAL scanner. Green here means this tree is clean
BY AN INTRAPROCEDURAL READING -- the kit says so itself: an echo through a file, an attribute or a
second function is outside what any such scan can see, so the population is a LOWER BOUND.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from lab_commons.dev.famtests import echoedtoken

_ROOT: Final = Path(__file__).resolve().parents[2]

#: THE TREES optimi-lab KEEPS ITS ASSERTION BODIES IN, with what a file in each is called. No default
#: upstream, and rightly: a guessed pair walks a directory that is not there and reports clean. These
#: are the same two trees `_placement.SCANNED` answers for, for the same reason -- `tests/` because
#: every architecture module here drives a shared body, `scripts/` because the dev entry points do too.
ROOTS: Final = (('tests', '*.py'), ('scripts', '*.py'))

#: Files whose own source carries the shape as DATA. EMPTY, and measured rather than assumed: the
#: kit plants its control into a `tmp_path` tree of its own, so optimi-lab keeps no offender fixture
#: on disk and has nothing to exempt. An arrival here must be a real file or it covers nothing.
EXEMPT: Final = ()

#: THE FLOOR ON THE READING, measured 2026-09-19: the two trees above hold 240 function bodies. The
#: floor sits a fifth below that so a walk that lost a whole tree cannot report clean, and the
#: headroom is the side that fires when growth makes the floor stop separating the two.
FUNCTION_FLOOR: Final = 190

#: THE OTHER SIDE. 190 + 65 = 255 against today's 240; re-measure the FLOOR when it fires.
FUNCTION_HEADROOM: Final = 65


def test_no_clause_in_this_tree_searches_for_a_token_its_own_producer_was_handed() -> None:
    """THE CHECK, with the number of function bodies read asserted before the empty answer is believed."""
    scan = echoedtoken.take_scan(_ROOT, roots=ROOTS, exempt=EXEMPT)
    echoedtoken.assert_no_clause_checks_its_own_echo(scan, floor=FUNCTION_FLOOR, headroom=FUNCTION_HEADROOM)


def test_every_exempt_path_is_a_real_file() -> None:
    """An exemption naming a deleted file covers nothing while still reading as a decision."""
    missing = [rel for rel in EXEMPT if not (_ROOT / rel).is_file()]
    assert not missing, f'{missing} are exempted from the echo scan and do not exist.'


def test_the_scanner_still_convicts_a_planted_echo(tmp_path: Path) -> None:
    """THE CONTROL: plant the shape and call the REAL scanner, so a silent walk cannot pass."""
    echoedtoken.assert_the_scanner_still_convicts(tmp_path, roots=ROOTS)
