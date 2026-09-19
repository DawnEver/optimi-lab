"""ENV-MUTATION-THROUGH-THE-DOOR in optimi-lab: both adapters supplied, and the gap rendered if either goes.

THE OLD REASON, AND WHY IT DID NOT SURVIVE BEING MEASURED. optimi-lab declared the rule absent
because it "has no shared box and no shared environment -- its `.venv` is its own, and nothing here
issues a verdict another party cites". The second clause is the one that decides it, and it is
false: `python -m lab_commons.dev.verify` is the ONLY verdict command optimi-lab has, it writes a
citable log carrying an `env=` key, and that key is computed when the run ENDS. So the invariant the
door exists for -- no verdict may cite an environment it did not run in -- has a subject here.

THE ASSERTIONS ARE THE FAMILY'S AS OF 2026-09-19, AND THE SEAM WAS ONE NOUN WIDE.
`lab_commons.dev.dep` published the MECHANISM the day it landed -- `Port`, `mutate`, the two gap
sentences, the refusal, the retirement -- and this file already imported it. What stayed forked was
the eight-arm test DRIVING it, measured by two independent audit lanes at 91.5% identical to
wdg-lab's twin, with a byte diff showing the 8.5% to be a docstring, a repo NAME four times, and a
`tempfile` prefix. `lab_commons.dev.famtests.depdoor` owns the arms now, planted control included.

WHAT OPTIMI-LAB SUPPLIES, each a keyword with NO DEFAULT upstream, so a repo that forgets one gets a
`TypeError` rather than a silent wrong answer:

* `REPO_NAME` -- what `scripts/dep.py`'s `Port` calls this checkout, and what the report PRINTS. A
  report naming the wrong repo is read by whoever is deciding whether to stop their own run.
* `port`, `live_holders`, `verdict_anchors` -- optimi-lab's own three adapters, so the floor arms
  judge THIS repo's readers rather than the kit's.
* `ANCHOR_SHAPE` -- `.verify/verify-*.log`, the repo-relative name shape `run_verify` writes. Only
  optimi-lab knows it, and every retirement arm is worthless if it retires a shape nothing produces.

WHAT IS LOCAL AND STAYS BELOW THE KIT CALLS: the JOIN between the two halves --
`scripts/dep.py` looks for anchors in the directory `lab_commons.dev.verify` actually writes to, and
answers for THIS checkout rather than the caller's working directory. The kit cannot ask that; it is
a fact about one repo's two files agreeing.

THE BROKER AND THE TREE ARE ISOLATED AND THAT IS THE DOOR WORKING, not a way around it: this suite
runs inside a `verify` run holding the real box seat, so the real port would REFUSE here --
correctly, and before it could report anything. The adapters are optimi-lab's either way; only the
records directory they read is the case's. NOTHING BELOW INSTALLS ANYTHING.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Final

from lab_commons.dev.famtests import depdoor
from lab_commons.dev.verify import LOG_DIRECTORY
from lab_commons.resources import Broker

_ROOT: Final = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.dep import ANCHOR_GLOB, live_holders, port, repo_root, verdict_anchors  # noqa: E402

#: What optimi-lab's port calls this checkout. The report prints it, and a reader deciding whether to
#: stop their own verdict run has only that text to decide on.
REPO_NAME: Final = 'optimi-lab'

#: The repo-relative shape a verdict log takes here, composed from the two names that own its halves
#: rather than spelled again: `lab_commons.dev.verify` owns the directory, `scripts/dep.py` the glob.
ANCHOR_SHAPE: Final = f'{LOG_DIRECTORY}/{ANCHOR_GLOB}'


def test_this_repo_supplies_both_halves_of_the_door(tmp_path: Path) -> None:
    """THE PROPERTY. Neither gap sentence may appear in a report optimi-lab's own port produced."""
    depdoor.assert_this_repo_supplies_both_halves_of_the_door(
        port=port,
        repo_name=REPO_NAME,
        broker=Broker(resource_dir=tmp_path),
    )


def test_a_port_supplying_neither_half_renders_both_gaps() -> None:
    """THE PLANTED CONTROL, and it is the state optimi-lab was in until 2026-09-17."""
    depdoor.assert_a_port_declaring_neither_renders_both_gaps()


def test_the_holders_adapter_names_a_real_planted_holder(tmp_path: Path) -> None:
    """PLANTED ON THE MACHINE. An adapter that always answers "nobody" guards nothing forever."""
    depdoor.assert_the_holders_adapter_names_a_planted_holder(
        live_holders=live_holders,
        broker=Broker(resource_dir=tmp_path),
    )


def test_the_door_refuses_while_a_verdict_is_in_flight() -> None:
    """H1, PREVENTED -- and the refusal names the holder, because "busy" cannot be acted on.

    THE KIT RECORDS THIS ARM'S OWN WEAKNESS AND SO DOES THIS LINE: the holder it plants is built out
    of `repo_name`, so the string it searches the refusal for is by construction the string it
    planted. Every value of the argument is green and the red is not plantable HERE; what the arm
    really convicts is a door that refuses without naming anybody at all.
    """
    depdoor.assert_the_door_refuses_while_a_verdict_is_in_flight(
        repo_name=REPO_NAME,
        verdict_anchors=verdict_anchors,
    )


def test_a_moved_key_retires_this_repos_verdict_anchors(tmp_path: Path) -> None:
    """H2, REMEDIED. The planted anchor is the shape optimi-lab really writes -- see the floor below."""
    anchor = tmp_path / ANCHOR_SHAPE.replace('*', '20260916T000000Z')
    anchor.parent.mkdir(parents=True)
    anchor.write_text('VERDICT result=PASS env=before\n', encoding='utf-8')
    depdoor.assert_a_moved_key_retires_a_planted_anchor(
        repo_name=REPO_NAME,
        verdict_anchors=lambda: verdict_anchors(tmp_path),
        anchor=anchor,
    )


def test_an_unmoved_key_retires_nothing(tmp_path: Path) -> None:
    """THE OTHER SIDE OF THE RATCHET. A retirement that always fires deletes evidence nobody voided."""
    anchor = tmp_path / ANCHOR_SHAPE.replace('*', '20260916T000001Z')
    anchor.parent.mkdir(parents=True)
    anchor.write_text('VERDICT result=PASS env=same\n', encoding='utf-8')
    depdoor.assert_an_unmoved_key_retires_nothing(
        repo_name=REPO_NAME,
        verdict_anchors=lambda: verdict_anchors(tmp_path),
        anchor=anchor,
    )


def test_the_anchor_adapter_reads_the_shape_this_repo_writes(tmp_path: Path) -> None:
    """FLOOR AND CEILING ON THE ANCHOR SCAN: it finds optimi-lab's real log, and it invents none."""
    depdoor.assert_the_anchor_reader_has_a_floor_and_a_ceiling(
        verdict_anchors=verdict_anchors,
        anchor_glob=ANCHOR_SHAPE,
        tree=tmp_path,
    )


def test_the_anchor_directory_is_the_one_verify_writes_to() -> None:
    """LOCAL, AND THE KIT CANNOT ASK IT: optimi-lab's two halves must agree on one directory.

    `scripts/dep.py` names the anchor directory by importing it from the verdict entry point rather
    than by copying the string, and answers for THIS checkout rather than the caller's cwd. Both are
    facts about two files in one repo agreeing, which no vendor-neutral body can state.
    """
    assert (repo_root() / LOG_DIRECTORY).name == LOG_DIRECTORY, (
        'the anchors are looked for somewhere other than the directory the verdict entry point writes to'
    )
    assert repo_root() == _ROOT, 'the script answers for a different checkout than the one this suite is in'
