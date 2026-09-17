"""ENV-MUTATION-THROUGH-THE-DOOR: both adapters supplied, and the gap rendered if either goes.

THE OLD REASON, AND WHY IT DID NOT SURVIVE BEING MEASURED. This repo declared the rule absent
because it "has no shared box and no shared environment -- its `.venv` is its own, and nothing here
issues a verdict another party cites". The second clause is the one that decides it, and it is
false: `python -m lab_commons.dev.verify` is the ONLY verdict command this repo has, it writes a
citable log carrying an `env=` key, and that key is computed when the run ENDS. So the invariant the
door exists for -- no verdict may cite an environment it did not run in -- has a subject here, and
the shared module's own docstring names this repo as the one supplying neither adapter.

WHAT WAS ACTUALLY MISSING was not a door but this repo's two answers to it. Without them
`lab_commons.dev.dep` renders `LOCK_UNDECLARED` (H1 -- a mutation DURING a verdict run -- unguarded
here, which is a different fact from absent) and `NO_ANCHORS_DECLARED` (H2 detected and not
remediable). `scripts/dep.py` supplies both: the box seat as the exclusion, `.verify/verify-*.log`
as the anchors.

WHAT IS PROVED, and every case drives the REAL door rather than a copy of its arithmetic:

1. Both adapters answer, so a report from THIS repo's port renders NEITHER gap -- against a control
   port that supplies neither and must render BOTH. One direction alone would pass on a port that
   answered nothing, or on a scanner that never looked.
2. The holders adapter names a PLANTED holder, taken on a real `BoxLock` in an isolated records
   directory. An adapter that returns an empty tuple whatever is happening is the vacuous green here
   -- it reads as "nobody is running a verdict" forever.
3. The door REFUSES while a verdict is in flight, and the refusal names the holder.
4. A moved key RETIRES a planted verdict log, and an unmoved key retires nothing. Two-sided, because
   a retirement that always fires deletes evidence and one that never fires leaves a verdict about
   a dead environment on record.
5. The anchor adapter finds a real log file and finds nothing in an empty tree -- so case 4 is
   retiring the shape this repo actually writes rather than a shape invented for the test.
"""

from __future__ import annotations

import re
import sys
import tempfile
from pathlib import Path
from typing import Final

import pytest
from lab_commons.dev.boxlock import BoxLock
from lab_commons.dev.dep import (
    LOCK_UNDECLARED,
    NO_ANCHORS_DECLARED,
    HeldEnvironmentError,
    Port,
    mutate,
)
from lab_commons.dev.verify import LOG_DIRECTORY
from lab_commons.resources import Broker

_ROOT: Final = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.dep import ANCHOR_GLOB, live_holders, port, repo_root, verdict_anchors  # noqa: E402

#: The requirement the cases name. Nothing is ever installed -- every call is a dry run or is given
#: an injected runner -- but the door refuses an empty requirement list, correctly, so it needs one.
_A_REQUIREMENT: Final = ('lab-commons',)


class _Ran:
    """A stand-in for `subprocess.run` that installs nothing and reports the code it was given."""

    def __init__(self, returncode: int = 0) -> None:
        self.returncode = returncode
        self.calls: list[list[str]] = []

    def __call__(self, argv: list[str], **_: object) -> _Ran:
        self.calls.append(argv)
        return self


def _keys(*values: str):
    """An env_key that reads a scripted sequence, so a MOVE can be planted without installing."""
    sequence = iter(values)
    last = values[-1]
    return lambda: next(sequence, last)


def test_this_repo_supplies_both_halves_of_the_door() -> None:
    """THE CHECK. Neither gap sentence may appear in a report this repo's port produced.

    THE BROKER IS ISOLATED AND THAT IS THE DOOR WORKING, not a way around it: this suite runs inside
    a `verify` run, which holds the box, so the real port would REFUSE here -- correctly, and before
    it could report anything. The adapters are the repo's own either way; only the records directory
    they read is the case's.
    """
    report = mutate(_A_REQUIREMENT, port=port(broker=Broker(resource_dir=Path(tempfile.mkdtemp()))), dry_run=True)
    assert report.gaps == (), (
        f'the door rendered {report.gaps}. A gap here is not cosmetic: LOCK_UNDECLARED means a dependency '
        f'change during a verdict run is unchecked rather than impossible.'
    )
    rendered = report.render()
    assert LOCK_UNDECLARED not in rendered and NO_ANCHORS_DECLARED not in rendered
    assert 'optimi-lab' in rendered and '-m pip install' in rendered, (
        f'the report does not say who changed what: {rendered!r}'
    )


def test_a_port_supplying_neither_half_renders_both_gaps() -> None:
    """THE CONTROL, and it is the state this repo was in until today.

    Without it the case above passes on a door that never renders anything, which is exactly the
    silent skip the shared module refuses.
    """
    report = mutate(_A_REQUIREMENT, port=Port(name='a-repo-that-declares-neither'), dry_run=True)
    assert set(report.gaps) == {LOCK_UNDECLARED, NO_ANCHORS_DECLARED}, (
        f'a port answering neither question rendered {report.gaps}. A gap that is not printed is assumed away.'
    )


def test_the_holders_adapter_names_a_real_planted_holder() -> None:
    """PLANTED ON THE MACHINE. An adapter that always answers "nobody" guards nothing forever."""
    broker = Broker(resource_dir=Path(tempfile.mkdtemp(prefix='optimi-lab-door-')))
    assert live_holders(broker=broker) == (), 'a fresh records directory already named a verdict run'
    with BoxLock('verify:the-planted-run', broker=broker).held():
        named = live_holders(broker=broker)
    assert any('verify:the-planted-run' in holder for holder in named), (
        f'the adapter answered {named} while a run really held the box. H1 cannot be refused by an adapter '
        f'that cannot see the run it is meant to refuse.'
    )
    assert live_holders(broker=broker) == (), 'the planted holder outlived the block that took it'


def test_the_door_refuses_while_a_verdict_is_in_flight() -> None:
    """H1, PREVENTED -- and the refusal names the holder, because "busy" cannot be acted on."""
    held = Port(
        name='optimi-lab',
        holders=lambda: ('verify:optimi-lab pid=4242',),
        anchor_paths=verdict_anchors,
    )
    with pytest.raises(HeldEnvironmentError) as refusal:
        mutate(_A_REQUIREMENT, port=held, dry_run=True)
    assert 'verify:optimi-lab pid=4242' in str(refusal.value)
    assert re.search(r'env_key', str(refusal.value)), 'the refusal does not say WHY a running verdict is at risk'


def test_a_moved_key_retires_this_repos_verdict_anchors(tmp_path: Path) -> None:
    """H2, REMEDIED. The planted anchor is the shape this repo really writes -- see the case below."""
    directory = tmp_path / LOG_DIRECTORY
    directory.mkdir()
    anchor = directory / 'verify-20260916T000000Z.log'
    anchor.write_text('VERDICT result=PASS env=before\n', encoding='utf-8')

    moved = Port(
        name='optimi-lab',
        holders=lambda: (),
        anchor_paths=lambda: verdict_anchors(tmp_path),
        key=_keys('before', 'after'),
    )
    report = mutate(_A_REQUIREMENT, port=moved, run=_Ran())
    assert report.moved and report.retired == (str(anchor),), (
        f'the key moved and {report.retired} was retired. A verdict citing an environment that no longer '
        f'exists is the thing this half of the door removes.'
    )
    assert not anchor.exists(), 'the anchor was reported retired and is still on disk'


def test_an_unmoved_key_retires_nothing(tmp_path: Path) -> None:
    """THE OTHER SIDE. A retirement that always fires deletes evidence nobody invalidated."""
    directory = tmp_path / LOG_DIRECTORY
    directory.mkdir()
    anchor = directory / 'verify-20260916T000001Z.log'
    anchor.write_text('VERDICT result=PASS env=same\n', encoding='utf-8')

    still = Port(
        name='optimi-lab',
        holders=lambda: (),
        anchor_paths=lambda: verdict_anchors(tmp_path),
        key=_keys('same', 'same'),
    )
    report = mutate(_A_REQUIREMENT, port=still, run=_Ran())
    assert not report.moved and report.retired == ()
    assert anchor.exists(), 'an unchanged environment took a live verdict down with it'


def test_the_anchor_adapter_reads_the_shape_this_repo_writes(tmp_path: Path) -> None:
    """FLOOR ON THE ANCHOR SCAN, both ways: it finds a real log, and it invents none."""
    assert verdict_anchors(tmp_path) == (), 'a tree with no verdict directory reported an anchor'
    directory = tmp_path / LOG_DIRECTORY
    directory.mkdir()
    (directory / 'not-a-verdict.txt').write_text('', encoding='utf-8')
    assert verdict_anchors(tmp_path) == (), 'a stray file in the verdict directory was read as a verdict'
    written = directory / ANCHOR_GLOB.replace('*', '20260916T120000Z')
    written.write_text('VERDICT result=PASS\n', encoding='utf-8')
    assert verdict_anchors(tmp_path) == (written,), (
        'the adapter cannot see the log name `lab_commons.dev.verify` writes, so every retirement above '
        'would be retiring a shape this repo never produces.'
    )


def test_the_anchor_directory_is_the_one_verify_writes_to() -> None:
    """The join between the two halves, read off the family module rather than copied into a string."""
    assert (repo_root() / LOG_DIRECTORY).name == LOG_DIRECTORY, (
        'the anchors are looked for somewhere other than the directory the verdict entry point writes to'
    )
    assert repo_root() == _ROOT, 'the script answers for a different checkout than the one this suite is in'
