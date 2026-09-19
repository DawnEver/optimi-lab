"""`RELATIVE-TOLERANCE` -- optimi-lab's corpus, its bar and its two floors, driven by the kit's arms.

The scan, the two bars, the positional-tolerance reading and the planted control live in
`lab_commons.dev.famtests.approxfloors`; this module supplies only what a shared body cannot hold:
THIS tree's corpus, the BAR it is judged under, and the two floors with their headrooms.

THE OLD REASON, RE-MEASURED AND STILL TRUE. This repo declared `UNITS-GO-THROUGH-PINT` absent among
the "no physical units" group: `Variable.unit` is a free-text LABEL this package never computes with,
pint is not a dependency it may take, and there is no golden generated in millimetres waiting to be
tightened by 1000x. None of that is the whole rule. The rule's last clause is about a tolerance's
SHAPE rather than about a unit -- *never write a `rel=` without the `abs=` floor it is combined
with* -- and that clause needs no unit at all, only a comparison whose two sides can both approach
zero.

WHY IT BITES IN A PACKAGE LIKE THIS ONE. `pytest.approx(x, rel=1e-9)` is a RATIO, so at `x == 0` it
admits only exactly zero, and this package's subject is full of quantities that pass through zero: a
crowding distance on a front of one, a weight in a decomposition, a velocity at a converged swarm, an
r2 on a constant column. A bare `rel=` there is not a loose test, it is a test whose tolerance
silently becomes zero for the inputs most likely to be interesting.

ADOPTED 2026-09-18. Until then this file held its own `tracked_python_files`, `_is_approx`,
`approx_calls` and `unfloored` -- near-twins of the kit's, and one of them measurably WEAKER: it read
`node.keywords` alone, so `approx(x, 1e-9)` (a ratio stated POSITIONALLY) read as a call that stated
nothing. The kit reads the positional pair, which is why the adoption is a strictening rather than a
move.

**THE BAR IS `'ratio'`, AND IT IS THIS REPO'S RULING RATHER THAN A DEFAULT.** The kit publishes two,
they convict different populations, and `bar` has NO DEFAULT for exactly that reason. Under
`'ratio'` a call that overrode the relative half and left the absolute half unstated is refused, and
a call with NEITHER keyword is left alone -- it has overridden nothing and is asking for `approx`'s
own combined default. Under `'unconditional'` that second call is an offender too. This file keeps
the question it has always asked; the other bar is defensible and is a different repo's.

MEASURED HERE 2026-09-18 AND THE PREDICTED RED DID NOT ARRIVE, which is worth recording rather than
passing over. The adoption was expected to red on the kit's positional reading, on the argument that
this repo's two call sites would be re-read. They are not: both are `pytest.approx(x)` with no
tolerance argument at all -- `test_operators.py:80` and `test_surrogate_models.py:99` -- so they
carry no ratio positionally or otherwise and are clean under `'ratio'`. They WOULD both convict under
`'unconditional'`, which is the same two-bar disagreement one level down and is the reason the bar is
stated here rather than inherited.

THE TWO FLOORS REFUSE DIFFERENT SILENCES, and adopting the kit is what brought the second one. The
CORPUS floor refuses a walk that stopped reaching the tree. The CALL-SITE floor refuses a corpus that
is fully read and holds no `approx` at all -- a suite whose tolerance sites were renamed, or a
matcher that stopped matching, reads exactly like a suite that states every floor. An empty offender
tuple is the natural answer to all three situations and only both floors together tell them apart.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from lab_commons.dev.famtests import approxfloors
from lab_commons.dev.rules import tracked_files

_ROOT: Final = Path(__file__).resolve().parents[2]

#: WHICH QUESTION THIS REPO ASKS -- a member of `approxfloors.BARS`. See the docstring above: the two
#: bars convict different sets and the kit refuses to pick one, so this constant is the ruling.
BAR: Final = 'ratio'

#: The floor on the corpus itself, so a scan that parsed nothing cannot report clean.
#: RE-MEASURED 2026-09-18: 57 tracked Python files (was 55 when this floor was last written).
PYTHON_FILE_FLOOR: Final = 40

#: THE OTHER SIDE OF `PYTHON_FILE_FLOOR`. Today's slack is 57 - 40 = 17, inside this band.
PYTHON_FILE_HEADROOM: Final = 25

#: The floor on the call-site population, which refuses the silence the corpus floor cannot see: a
#: fully-read tree whose every `approx` was renamed. RE-MEASURED 2026-09-18 at exactly 2 call sites.
APPROX_CALL_FLOOR: Final = 2

#: THE OTHER SIDE OF `APPROX_CALL_FLOOR`. The floor sits ON the population, so any growth at all is
#: worth re-reading; 2 + 5 = 7 is the point at which this constant must be re-measured.
APPROX_CALL_HEADROOM: Final = 5

_WHAT: Final = 'RELATIVE-TOLERANCE'


def _corpus() -> list[Path]:
    """Every tracked `.py` file -- what is COMMITTED, not what a walk happens to find.

    The one input a shared module cannot supply. A guessed walk does not raise: it reads a directory
    that is not there and reports clean, which is why the two floors above are mandatory.
    """
    return [_ROOT / name for name in sorted(tracked_files(_ROOT)) if name.endswith('.py')]


def test_no_relative_tolerance_is_written_without_its_floor() -> None:
    """THE CHECK: both floors bound on both sides FIRST, then the offenders under THIS repo's bar."""
    scan = approxfloors.take_scan(_corpus(), root=_ROOT, bar=BAR)
    approxfloors.assert_every_tolerance_states_its_floor(
        scan,
        bar=BAR,
        file_floor=PYTHON_FILE_FLOOR,
        file_headroom=PYTHON_FILE_HEADROOM,
        call_floor=APPROX_CALL_FLOOR,
        call_headroom=APPROX_CALL_HEADROOM,
        what=_WHAT,
    )


def test_the_scanner_tells_every_planted_shape_apart(tmp_path: Path) -> None:
    """CONTROL, BOTH DIRECTIONS, through the REAL scanner on a REAL file.

    An empty live answer means nothing unless the scanner is shown to answer at all -- and a lint
    convicting everything is deleted rather than obeyed, so the kit's control plants each offender
    beside an honest neighbour that must NOT be named, including the positional pair this file's own
    reader was blind to before the adoption.

    THE AXIS THIS CONTROL IS BLIND TO: it plants SOURCE TEXT and proves nothing about the CORPUS. A
    `_corpus()` that returned an empty list, pointed at a renamed directory, or silently dropped
    `tests/` passes every arm here. That axis belongs to the two floors above and to
    `test_the_corpus_is_this_repo_tracked_python`, which is a different argument entirely.
    """
    approxfloors.assert_the_scanner_still_convicts(tmp_path, bar=BAR)


def test_the_bar_is_one_the_kit_publishes() -> None:
    """A bar nobody declared cannot judge anything, and a misspelling reads as a plausible number."""
    assert BAR in approxfloors.BARS, f'{BAR!r} is not one of {sorted(approxfloors.BARS)}'


def test_the_corpus_is_this_repo_tracked_python() -> None:
    """WHAT IS BEING READ, named -- a floor over the wrong corpus is a floor over nothing.

    This is the axis the planted control is blind to. It pins the SHAPE of the narrowing rather than
    a count the floors already hold: every path is a real tracked `.py` file under this root, and the
    file holding this very assertion is one of them, so a corpus that stopped reaching `tests/`
    fails here by NAME rather than by arithmetic.
    """
    corpus = _corpus()
    assert corpus, 'the corpus is empty, so every arm above is vacuous'
    assert all(path.suffix == '.py' and path.exists() for path in corpus), 'the corpus is not tracked Python'
    assert Path(__file__).resolve() in corpus, 'this guard is outside the corpus it scans'
