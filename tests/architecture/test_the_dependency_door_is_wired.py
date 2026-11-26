"""ENV-MUTATION-THROUGH-THE-DOOR in optimi-lab: its verdict anchors supplied, and the gap rendered if they go.

THE OLD REASON, AND WHY IT DID NOT SURVIVE BEING MEASURED. optimi-lab declared the rule absent
because it "has no shared box and no shared environment -- its `.venv` is its own, and nothing here
issues a verdict another party cites". The second clause is the one that decides it, and it is
false: `python -m dev_lab.dev.verify` is the ONLY verdict command optimi-lab has, it writes a
citable log carrying an `env=` key, and that key is computed when the run ENDS. So the invariant the
door exists for -- no verdict may cite an environment it did not run in -- has a subject here.

WHAT WAS ACTUALLY MISSING was not a door but optimi-lab's answer to it. H1 -- a mutation DURING a
verdict run -- is the kit's alone since the CPU seat and the environment seat were split: `verify`
holds this environment's `EnvironmentSeat` and `dep.mutate` refuses while it is held. Without the
anchors `dev_lab.dev.dep` renders `NO_ANCHORS_DECLARED` (H2 detected and not remediable);
`scripts/dep.py` supplies the `verify-*.log` files under the dated `verify` output kind.

THE ASSERTIONS ARE NOW THE FAMILY'S, ADOPTED 2026-09-19 from `dev_lab.dev.famtests.depdoor`.
Two independent audit lanes measured this file against the sibling lab's twin at 91.5% identical, and the
8.5% was a docstring, the repo's name four times, a `tempfile` prefix and two invented timestamps --
not one assertion and not one message. What is LOCAL and arrives below as a keyword with NO DEFAULT
is exactly that: the name `optimi-lab`, `scripts/dep.py`'s anchor adapter and its port factory, and the
repo-relative shape `<output>/*/*/*/verify/verify-*.log` that optimi-lab's own `verify` writes. A repo that forgets
one gets a `TypeError` rather than a silent wrong answer.

ONE ARM STAYS LOCAL because the kit has no place for it: the JOIN between the two halves -- that the
directory the anchors are looked for in is the directory the verdict entry point writes to. That is
a fact about this checkout and about `dev_lab.dev.verifysteps.output_dir(root, LOG_KIND)` together, and each of
the kit's arms answers for one half only.

THE MUTATE-DRIVING ARMS NEED A PRIVATE SEAT RECORDS ROOT, and the kit's own suite is where this
file learned it. A verdict run takes `EnvironmentSeat(sys.prefix)` on the box ledger for
the WHOLE of its run, and `dev_lab.dev.dep._change` takes the same seat for a mutation --
so a dry run inside a verdict is refused before it can report, which is what MEASURED 2026-10-07 as
four reds in this file. Pointing
`lab_commons.box.BOX_ENV` -- the one box ledger file, relocated FOR TESTS, and the
one the kit's own dependency-door tests apply with this same autouse shape -- at a per-test
directory puts the arms' own seats in a private namespace while the enclosing verdict keeps the
real one. A REAL mutation is still excluded, by the verdict holding this environment's seat.

THE PRIVATE BOX KEEPS A FREE-SPACE FLOOR OF ITS OWN (optimi-lab #221), the shape of dev-lab's
`tests/conftest.py::_a_box_of_its_own`. The box refuses a hold on a volume under the machine's
`[storage] free_floor_gb` (default 50 GiB), and the private ledger sits on the system temp volume: a
hosted Windows runner with 29.2 GiB free refused three arms here at once. The test's own machine file
declares the test box's floor, so the host's free disk never decides these results; the machine
default is untouched, and the control below shows it still refuses.
"""

from __future__ import annotations

import os
import sys
from fnmatch import fnmatchcase
from pathlib import Path, PurePosixPath
from typing import Final

import pytest
from dev_lab.dev.datedlog import date_parts, declared_output
from dev_lab.dev.famtests import depdoor
from dev_lab.dev.verifysteps import LOG_KIND
from lab_commons.box import BOX_ENV, box_path
from lab_commons.box.disk import verdict
from lab_commons.compute import floor
from lab_commons.config import CONFIG_ENV
from lab_commons.storage import FREE_FLOOR_GB_DEFAULT, free_floor

_ROOT: Final = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.dep import ANCHOR_GLOB, anchor_shape, port, repo_root, verdict_anchors  # noqa: E402

#: What the report must call this checkout. It is what a reader deciding whether to stop their own
#: run has to go on, so naming the wrong repo is not cosmetic.
REPO_NAME: Final = 'optimi-lab'

#: The timestamp the planted verdict logs carry. A constant, because a reading off the clock would
#: make the arms below a fact about the day they ran.
_A_TIMESTAMP: Final = '20260919T000000Z'


#: The test box's machine file: one MiB free, never the host's 50 GiB default (dev-lab's `TEST_MACHINE`).
_TEST_MACHINE: Final = '[storage]\nfree_floor_gb = 0.001\n'

_GIB: Final = 1024**3
#: What the hosted Windows runner's system drive had free when these arms were refused.
_HOSTED_FREE: Final = 29 * _GIB


@pytest.fixture(autouse=True)
def _private_seat_records(
    tmp_path: Path, tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The door's own locks contend in a private root, on a box with its own floor -- see the docstring."""
    monkeypatch.setenv(BOX_ENV, str(tmp_path / 'box' / 'box.sqlite'))
    machine = tmp_path_factory.mktemp('lab-commons-machine') / 'config.toml'  # beside tmp_path, which starts empty
    machine.write_text(_TEST_MACHINE, encoding='utf-8')
    monkeypatch.setenv(CONFIG_ENV, str(machine))


def _refusal(monkeypatch: pytest.MonkeyPatch) -> str | None:
    """What the box says of this test's volumes (its ledger's and its cwd's) on the hosted runner's free disk."""
    monkeypatch.setattr(floor, 'free_bytes', lambda _volume: _HOSTED_FREE)
    return verdict((box_path(), Path.cwd()), free_floor())[0]


def test_the_private_box_is_not_refused_on_a_host_under_the_machine_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """optimi-lab #221: the hosted runner's free disk does not refuse this module's box."""
    assert _HOSTED_FREE < FREE_FLOOR_GB_DEFAULT * _GIB, 'the planted volume must sit under the default floor'
    assert _refusal(monkeypatch) is None


def test_planted_control_the_machine_default_floor_refuses_that_host(monkeypatch: pytest.MonkeyPatch) -> None:
    """PLANTED CONTROL: without the test machine's floor, the same reading is the refusal the runner gave."""
    declared = f'[storage]\nfree_floor_gb = {FREE_FLOOR_GB_DEFAULT}\n'
    Path(os.environ[CONFIG_ENV]).write_text(declared, encoding='utf-8')
    said = _refusal(monkeypatch)
    assert said is not None
    assert 'floor' in said


def _plant(tmp_path: Path) -> Path:
    """Write one verdict log into *tmp_path*, in the shape this repo really produces."""
    anchor = tmp_path / anchor_shape().replace('*', _A_TIMESTAMP)
    anchor.parent.mkdir(parents=True, exist_ok=True)
    anchor.write_text('VERDICT result=PASS env=planted\n', encoding='utf-8')
    return anchor


def test_this_repo_declares_its_verdict_anchors() -> None:
    """THE PROPERTY: no gap sentence may appear in a report optimi-lab's own port produced."""
    depdoor.assert_this_repo_declares_its_verdict_anchors(port=port, repo_name=REPO_NAME)


def test_a_port_declaring_no_anchors_renders_the_gap() -> None:
    """THE CONTROL, and it is the state optimi-lab was in until 2026-09-17."""
    depdoor.assert_a_port_declaring_no_anchors_renders_the_gap()


def test_a_verdict_in_another_environment_does_not_refuse(tmp_path: Path) -> None:
    """THE OTHER SIDE OF H1: a verdict holding an unrelated environment never refuses this one."""
    depdoor.assert_a_verdict_in_another_environment_does_not_refuse(
        repo_name=REPO_NAME,
        verdict_anchors=verdict_anchors,
        elsewhere=tmp_path / 'another-environment',
    )


def test_the_door_refuses_while_a_verdict_is_in_flight() -> None:
    """H1, PREVENTED -- and the refusal names the holder, because "busy" cannot be acted on.

    EVERY VALUE OF `repo_name` IS GREEN HERE and the kit records it: the planted holder is BUILT
    from `repo_name`, so the string searched for in the refusal is by construction the string
    planted. This arm measures that the door refuses at all and that it names its holder and its
    env_key; it cannot measure that the name is optimi-lab's. The arm above, which reads the RENDERED
    report of this repo's real port, is where a wrong name is plantable.
    """
    depdoor.assert_the_door_refuses_while_a_verdict_is_in_flight(
        repo_name=REPO_NAME,
        verdict_anchors=verdict_anchors,
    )


def test_a_moved_key_retires_this_repos_verdict_anchors(tmp_path: Path) -> None:
    """H2, REMEDIED. The planted anchor is the shape this repo really writes -- see the floor arm."""
    anchor = _plant(tmp_path)
    depdoor.assert_a_moved_key_retires_a_planted_anchor(
        repo_name=REPO_NAME,
        verdict_anchors=lambda: verdict_anchors(tmp_path),
        anchor=anchor,
    )


def test_an_unmoved_key_retires_nothing(tmp_path: Path) -> None:
    """THE OTHER SIDE. A retirement that always fires deletes evidence nobody invalidated."""
    anchor = _plant(tmp_path)
    depdoor.assert_an_unmoved_key_retires_nothing(
        repo_name=REPO_NAME,
        verdict_anchors=lambda: verdict_anchors(tmp_path),
        anchor=anchor,
    )


def test_the_anchor_adapter_reads_the_shape_this_repo_writes(tmp_path: Path) -> None:
    """FLOOR AND CEILING ON THE ANCHOR SCAN: it finds a real log, and it invents none."""
    depdoor.assert_the_anchor_reader_has_a_floor_and_a_ceiling(
        verdict_anchors=verdict_anchors,
        anchor_glob=anchor_shape(),
        tree=tmp_path,
    )


def test_the_anchor_directory_is_the_one_verify_writes_to() -> None:
    """THE JOIN, and the one arm the kit has no place for: each of its arms answers for one half.

    The verify log of TODAY -- the declared output root, the date parts and `LOG_KIND`, exactly as
    `dev_lab.dev.verifysteps` composes it -- must match the anchor shape, so a `verify` that moved its
    logs reds here instead of leaving every retirement arm above green over a directory nothing
    writes to.
    """
    written = PurePosixPath(declared_output(_ROOT), *date_parts(), LOG_KIND, ANCHOR_GLOB.replace('*', _A_TIMESTAMP))
    assert fnmatchcase(str(written), anchor_shape()), f'anchors are looked for in {anchor_shape()!r}, not at {written}'
    assert repo_root() == _ROOT, 'the script answers for a different checkout than the one this suite is in'
