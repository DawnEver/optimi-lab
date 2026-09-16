"""REFUSAL-NAMES-THE-REMEDY, and the reason this repo gave for the gap did not survive reading it.

THE OLD REASON: optimi-lab "has no gate runner", so there was nothing to bound. That is a statement
about a RUNNER and the rule is a statement about a WAIT. This package launches subprocesses --
`scripts/pdoc.py` drives a documentation build, the verify entry point drives ruff and pytest -- and
every one of those is a wait somebody can forget to bound. The rule's second clause is not about a
runner at all: *kill a process TREE by its ROOT pid, since stopping a wrapper leaves its children
running*, which is true of any subprocess this repo ever starts.

WHAT `subprocess.run(timeout=...)` DOES NOT DO, and it is the whole reason a mechanism exists rather
than a convention. It kills only the DIRECT child, and then -- on Windows -- reaps with an UNBOUNDED
`communicate()`. Any grandchild inherits the stdout write handle, so the pipe stays open while one
of them lives: a hang detector that hangs, where raising the wall makes it strictly worse.

THE THREE THINGS PROVED HERE, each with its violation PLANTED rather than described:

1. A wall terminates the TREE, measured on the CLOCK against a real grandchild that really inherits
   stdout and really outlives the wall.
2. A reaper never kills its own lineage -- this process asks for its own tree to be killed and must
   survive, which is the self-match a human is otherwise told to notice and ignore.
3. A refusal names a remedy, and a DIFFERENT one per state: an unmeasured width, a narrowed run and
   a healthy run that still did not fit are three situations, and two of the three remedies are
   actively wrong in the other's case.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

import pytest
from lab_commons.dev.bounded import (
    BLAS_THREAD_VARS,
    NARROWED_FLOOR,
    blas_threads,
    is_narrowed,
    logical_cores,
    reap_tree,
    run_bounded,
    wall_reason,
    worker_width,
)
from lab_commons.proc import process_tree

#: The wall these cases use. Small, because the claim is a RATIO -- the call comes back near its own
#: wall rather than near the child's lifetime -- and a ratio is what survives a busy box.
WALL_SECONDS_LIMIT = 3.0
REAP_SECONDS_CEILING = 4.0

#: The child sleeps far past the wall AND spawns a grandchild that does too. The grandchild holds
#: the inherited stdout pipe open; without it this case would pass against the implementation the
#: rule was written about.
_PARENT = (
    'import subprocess, sys, time; '
    "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)']); "
    'time.sleep(120)'
)

#: The name THIS repo would use for an unbounded run. A remedy naming a tier that does not exist
#: here would be a dead end wearing a remedy's clothes, so the word is the repo's, not the module's.
WIDER_TIER = 'a full run with an explicit go-ahead'


def test_the_wall_terminates_the_whole_tree_rather_than_the_wrapper() -> None:
    """THE PROPERTY, asserted on the CLOCK because the clock is the subject.

    A pid table on a busy box is a snapshot of a mutating machine, but "did this call come back" is
    a fact nothing can blur. The child sleeps 120 s, so a run waiting on the grandchild's inherited
    pipe cannot return in single-digit seconds.
    """
    started = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        run_bounded([sys.executable, '-c', _PARENT], timeout=WALL_SECONDS_LIMIT, text=True)
    elapsed = time.monotonic() - started
    assert elapsed < WALL_SECONDS_LIMIT + REAP_SECONDS_CEILING, (
        f'the bounded call took {elapsed:.1f}s against a {WALL_SECONDS_LIMIT}s wall -- it waited on the very tree '
        f'it was supposed to end.'
    )


def test_a_run_inside_the_wall_is_left_alone() -> None:
    """THE OTHER SIDE. A wall that also breaks the passing case is a fault, not a bound."""
    done = run_bounded([sys.executable, '-c', 'print("ok")'], timeout=60, text=True)
    assert done.returncode == 0
    assert done.stdout.strip() == 'ok'


def test_the_reaper_refuses_its_own_lineage() -> None:
    """PLANTED, AND THE PLANT IS THIS PROCESS. Ask for our own tree to be killed; be refused.

    A REFUSAL RATHER THAN A FILTER, and the difference is the point: filtering ourselves out still
    leaves an ANCESTOR in the same tree, and killing that takes the shell or the runner with it --
    after which there is nobody left to report what happened.
    """
    assert os.getpid() in process_tree(os.getpid()), 'a tree not containing its own root cannot protect it'
    assert reap_tree(os.getpid()) == frozenset(), 'the reaper was asked to end its own tree and must refuse'


def test_the_reaper_still_kills_a_process_that_is_not_ours() -> None:
    """THE OTHER SIDE OF THE SPARING. A reaper that spares everything is not a reaper.

    Confirmed by the planted process's own exit rather than by the return value alone, so the claim
    rests on an observation of the machine and not on the function agreeing with itself.
    """
    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])
    try:
        assert child.poll() is None, 'the planted process did not start, so nothing was proved'
        assert child.pid in reap_tree(child.pid), f'{child.pid} is outside this lineage and must be reaped'
        assert child.wait(timeout=20) is not None
    finally:
        if child.poll() is None:
            child.kill()


def test_the_refusal_names_a_different_remedy_in_each_state() -> None:
    """The rule itself: TAKE THE REMEDY A REFUSAL NAMES -- so three states may not share one."""
    capacity = max(logical_cores(), 4)
    unmeasured = wall_reason(workers=None, capacity=capacity, wider_tier=WIDER_TIER)
    narrowed = wall_reason(workers=1, capacity=capacity, wider_tier=WIDER_TIER)
    healthy = wall_reason(workers=capacity, capacity=capacity, wider_tier=WIDER_TIER)

    assert len({unmeasured, narrowed, healthy}) == 3, 'one sentence for three states is a diagnosis nobody computed'
    for sentence in (unmeasured, narrowed, healthy):
        assert WIDER_TIER in sentence, 'every refusal names its remedy, in a word this repo uses'
    assert 'NARROWED' in narrowed and 'NARROWED' not in healthy
    assert is_narrowed(NARROWED_FLOOR, capacity) and not is_narrowed(capacity, capacity)


def test_a_width_is_computed_rather_than_guessed() -> None:
    """FLOOR ON THE BOX READING. Memory first and cores second, and never below the stated floor."""
    cores = logical_cores()
    assert cores >= 1, 'a box reporting no cores makes every width answer arithmetic on nothing'
    width = worker_width(gb_per_worker=4.0, max_width=cores, floor=1, reserve_cores=1, fallback=max(1, cores - 1))
    assert 1 <= width <= max(1, cores), 'the width is bounded by the box at both ends'
    assert blas_threads(width) >= 1, 'a worker with zero threads computes nothing'
    assert len(BLAS_THREAD_VARS) >= 4, 'the pool-spawning runtimes, by name rather than by count'
