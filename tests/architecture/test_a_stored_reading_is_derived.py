"""A ROSTER ROW MAY NOT STORE A NUMBER IT COULD DERIVE -- the stale-reading guard over `_placement`.

THE DEFECT, MEASURED IN THIS TREE BEFORE THE GUARD EXISTED. Every row of `_placement.PLACEMENT` is
`path -> Placement(side, why)`, and the `why` is where the author records the density reading that
decided the side -- `RE-MEASURED 2026-09-19: own=27 hits=0 -> 0.00%`. Those numbers are DERIVABLE:
`_placement.measure` re-computes every one of them from the file the row names. They are also
STORED, so they go stale the moment that file changes and nothing anywhere noticed. Fourteen of
optimi-lab's went stale that way and were re-derived by hand on 2026-09-19; three MORE went stale
the same afternoon, when the `famtests.countpins` adoption shrank the files three rows describe.
Three in one afternoon is the argument for a guard rather than another sweep.

THE DIRECTION IS THE FINDING. Most deltas are large and NEGATIVE -- a file SHRANK when its kit
adoption landed and the sentence describing it was never re-read -- so a roster SYSTEMATICALLY
OVER-STATES the work it has left. A progress meter that over-reports remaining work is the one
failure a green suite cannot show you, because nothing compares a stored number against its
derivation.

WHAT IS THE KIT'S AND WHAT IS OPTIMI-LAB'S. The reader, the live/historical separation, the four
ways a claim is classified (`live`, `marker`, `superseded`, `arrow`) and both assertions are
`lab_commons.dev.famtests.storedreadings`. What cannot be guessed and is written here: WHICH
SPELLINGS this repo's rows quote a reading in, HOW a reading is DERIVED for a path in this checkout,
and the three floors. `derive` in particular is this repo's whole density convention -- `measure`
with optimi-lab's noun signal and delegation home, and the percentage rounded exactly as the prose
rounds it.

THE FLOOR IS NOT A FORMALITY. Finding NO disagreement is the answer a fixed roster gives and also
the answer a reader whose spellings cannot fire gives, so the population of LIVE CLAIMS READ is
floored before any of them is judged, and the kit raises rather than returning empty when handed a
spelling set that matches nothing.

SCOPE, STATED SO A READER DOES NOT SUPPLY "EVERYTHING". This reads `Placement.why` and nothing else.
It does NOT read the module-level `#:` CALIBRATION comments that bound `OWN_MECHANISM_CEILING` -- a
dated derivation kept as evidence, which `assert_ceiling_is_bounded` already makes an argument -- and
it does NOT read the `BELOW_THE_BAR` dict, whose entries quote readings the same way a row does and
are invisible from here. That second blind spot is a real gap and is named rather than implied.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import _placement
import pytest
from _placement import PLACEMENT, measure, repo_root
from lab_commons.dev.famtests import storedreadings

#: How a density reading is spelled in this repo's rows: `own=27`, `hits=0`, `-> 0.00%`. `project=`
#: appears too and is deliberately NOT here -- it is `_placement.py`'s own row quoting a count of
#: PROJECT FILES, which is not derivable from the file the row names, and a guard that guessed at it
#: would convict prose it cannot re-measure.
SPELLINGS: Final = frozenset({'own', 'hits', storedreadings.PERCENT})

#: THE LIVE POPULATION, measured 2026-09-19 at 64 claims across 31 rows. 64 - 13 = 51, the slack this
#: checkout applies at every floor re-take. Below this the reader lost most of its corpus, and a
#: reader that read nothing reports exactly what a fixed roster reports.
LIVE_CLAIM_FLOOR: Final = 51

#: THE OTHER SIDE. 51 + 20 = 71 against today's 64. The remedy for a breach is to RE-MEASURE THE
#: FLOOR: the claim population grows every time a row records a re-measurement, which is the growth
#: this table is supposed to have, and a headroom widened to absorb it is the arm kept while the
#: guard it stands for is given up.
LIVE_CLAIM_HEADROOM: Final = 20

#: THE HISTORICAL POPULATION, measured 2026-09-19 at 24 claims -- every `down from own=88` and
#: `up from ... -> 7.46%` this roster carries. Floored at 16 for the same reason: the dating arm's
#: natural answer over zero historical claims is green, which is also what a reader blind to the
#: markers returns.
HISTORICAL_CLAIM_FLOOR: Final = 16


def _derive(rel: str) -> dict[str, float]:
    """THE REPO'S OWN DENSITY CONVENTION, as the function the kit refuses to guess.

    The percentage is rounded to two places because that is how the prose writes it; deriving it at
    full precision would convict every correctly-rounded row in the table. A zero-line file reads
    0.00% rather than raising -- `measure` can return `own=0` for an empty module, and a refusal
    there would be about the file rather than about its row.
    """
    reading = measure(Path(repo_root() / rel).read_text(encoding='utf-8'))
    percent = round(100 * reading.hits / reading.own, 2) if reading.own else 0.0
    return {'own': float(reading.own), 'hits': float(reading.hits), storedreadings.PERCENT: percent}


def _rows() -> dict[str, str]:
    """Every row's `why`, which is the only text this guard reads."""
    return {path: row.why for path, row in PLACEMENT.items()}


def test_no_row_quotes_a_reading_its_own_file_no_longer_gives() -> None:
    """THE CHECK: every LIVE reading in the roster agrees with re-deriving it, floors first."""
    storedreadings.assert_stored_readings_are_live(
        _rows(),
        spellings=SPELLINGS,
        derive=_derive,
        floor=LIVE_CLAIM_FLOOR,
        headroom=LIVE_CLAIM_HEADROOM,
        what='STORED-READING (optimi-lab placement rows)',
    )


def test_every_historical_reading_says_when_it_was_true() -> None:
    """THE OTHER HALF, and it is what makes the first one possible.

    A row is ALLOWED to quote a former reading -- `own=27 hits=0, down from own=88` is the most
    useful sentence in this table, because it records the direction a kit adoption moved a file. It
    is allowed only while it says WHEN it was true: an undated former reading is indistinguishable
    from a live one that has gone stale, and a guard that cannot tell them apart must either convict
    every history or exempt every stale claim.
    """
    storedreadings.assert_history_is_dated(
        _rows(),
        spellings=SPELLINGS,
        floor=HISTORICAL_CLAIM_FLOOR,
        what='STORED-READING history (optimi-lab placement rows)',
    )


def test_the_reader_still_convicts_a_planted_stale_reading() -> None:
    """THE CONTROL: the kit plants the prose and drives the REAL reader, for THIS repo's spelling.

    Driven on `own`, which is the spelling every row in this table carries. A control aimed at a
    spelling the roster does not use would pass while the live corpus went unread.
    """
    storedreadings.assert_the_reader_still_convicts(spelling='own')


def test_the_guard_reads_the_roster_it_names_and_not_a_copy() -> None:
    """A SCOPE CLAIM NEEDS ITS OWN TEST: plant a stale row IN the roster and call the REAL check.

    The docstring above claims this guard reads `_placement.PLACEMENT`. Nothing so far would have
    failed if `_rows` had been wired to a fixture, a stale import or an empty dict -- the floors
    would red, but only after someone noticed the corpus, and the floors are exactly what a green
    run stops anyone from reading.
    """
    row = next(iter(PLACEMENT))
    poisoned = dict(_rows())
    poisoned[row] = 'MEASURED 2026-09-19: own=999999 hits=0 -> 0.00%'
    with pytest.raises(storedreadings.StaleReading) as convicted:
        storedreadings.assert_stored_readings_are_live(
            poisoned,
            spellings=SPELLINGS,
            derive=_derive,
            floor=LIVE_CLAIM_FLOOR,
            headroom=LIVE_CLAIM_HEADROOM,
            what='STORED-READING (planted)',
        )
    assert row in str(convicted.value), (
        f'the reader convicted a planted stale row but its refusal does not name {row}, so an author '
        f'is told a number is wrong without being told which line to open'
    )
    assert _rows() == {path: r.why for path, r in _placement.PLACEMENT.items()}, (
        'the rows this guard reads are not the live manifest, so every assertion above is about a copy'
    )
