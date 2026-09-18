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

THE BODIES ARE THE FAMILY'S, and this file is the repo's ANSWERS. `lab_commons.dev.famtests.
boundedremedy` holds every assertion below. Measured before adoption, this file and wdg-lab's
near-twin carried an identical wall, an identical reap ceiling, a byte-identical planted-parent
snippet and five assertion bodies written twice; a shared body under two names is the fork that
package exists to remove, so the copy is gone rather than wrapped or re-exported. What stays here is
what only this checkout can say -- the word this repo would use for an unbounded run, the capacity it
prices against, the width it calls narrowed, and the two numbers its anti-flake arithmetic is priced
from -- and every one of them is a keyword argument with no default, because a default is one repo's
answer handed silently to another.

WHY NO NUMBER BELOW MAY BE RAISED TO MAKE AN ARM PASS. The wall arm asserts on ELAPSED TIME, and the
tempting repair -- a bigger wall -- makes a hang detector strictly worse. So the claim is a
SEPARATION rather than a duration, the slack is PRICED from this box in the same second, and the
pricing has a CEILING past which the arm reports `BoxTooLoaded`: INCONCLUSIVE, neither green nor red,
because an arm that can no longer fail must not report a pass. The two refusals are told apart by
re-pricing with this box's contribution removed, since "wait for a quiet box" is the wrong advice for
numbers that were never going to work.

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

from lab_commons.dev.bounded import (
    BLAS_THREAD_VARS,
    NARROWED_FLOOR,
    blas_threads,
    logical_cores,
    worker_width,
)
from lab_commons.dev.famtests.boundedremedy import (
    assert_a_run_inside_the_wall_is_left_alone,
    assert_each_state_names_its_own_remedy,
    assert_the_reaper_kills_what_is_not_ours,
    assert_the_reaper_refuses_its_own_lineage,
    assert_the_wall_terminates_the_tree,
    default_interpreter,
)

#: The wall these cases use. Small, because the claim is a RATIO -- the call comes back near its own
#: wall rather than near the child's lifetime -- and a ratio is what survives a busy box.
WALL_SECONDS_LIMIT = 3.0
REAP_SECONDS_CEILING = 4.0

#: How many spawn-overheads of slack this repo grants before it would rather call a reading
#: inconclusive. Priced from a measurement rather than chosen: an empty bounded child costs ~0.13 s
#: on this box (2026-09-18, five readings, 0.121-0.140), so eight of them is ~1.0 s of slack here.
#: LOWER than wdg-lab's twelve, and the difference is a real one: this repo compiles nothing and
#: declares no slow tier, so a run of it that is already many times slower than this reading is
#: reporting on the machine rather than on the code.
SPAWN_OVERHEAD_BAND = 8.0

#: How many times the permitted return the planted child's lifetime must be before a pass is
#: evidence at all. Six -- tighter than wdg-lab's four for the same reason the band is narrower:
#: with nothing heavy sharing the box, this repo asks for a wider margin before it calls a bounded
#: return distinguishable from a wait on the tree. The declared numbers give a 7 s permitted return
#: against a 120 s child, clearing six-fold with room to spare.
SEPARATION_MIN = 6.0

#: How long a reaped process gets to actually die. BOUNDED, because an unbounded wait inside the
#: guard against unbounded waits is the defect wearing the uniform.
EXIT_SECONDS_CEILING = 20.0

#: The name THIS repo would use for an unbounded run. A remedy naming a tier that does not exist
#: here would be a dead end wearing a remedy's clothes, so the word is the repo's, not the module's.
WIDER_TIER = 'a full run with an explicit go-ahead'

#: The width this repo prices its refusal sentences against. FLOORED at 4 rather than capped: this
#: repo's suite is short enough that a small box still runs it wide, and a capacity below the
#: narrowed reading would make the healthy state unreachable.
REMEDY_CAPACITY = max(logical_cores(), 4)

#: A width this repo calls narrowed -- the kit's own floor, because this repo has no separate
#: fallback width of its own to name.
NARROWED_WORKERS = NARROWED_FLOOR


def test_the_wall_terminates_the_whole_tree_rather_than_the_wrapper() -> None:
    """THE PROPERTY, asserted on the CLOCK because the clock is the subject.

    A pid table on a busy box is a snapshot of a mutating machine, but "did this call come back" is
    a fact nothing can blur. May raise `BoxTooLoaded` instead of failing, and that is the arm
    working rather than flaking: INCONCLUSIVE is the honest reading of a box too loaded to tell a
    bounded return from a wait, and the remedy is a quiet box rather than a wider wall.
    """
    assert_the_wall_terminates_the_tree(
        interpreter=default_interpreter(),
        wall_seconds=WALL_SECONDS_LIMIT,
        reap_ceiling_seconds=REAP_SECONDS_CEILING,
        overhead_multiple=SPAWN_OVERHEAD_BAND,
        separation_factor=SEPARATION_MIN,
    )


def test_a_run_inside_the_wall_is_left_alone() -> None:
    """THE OTHER SIDE. A wall that also breaks the passing case is a fault, not a bound."""
    assert_a_run_inside_the_wall_is_left_alone(interpreter=default_interpreter(), wall_seconds=60.0)


def test_the_reaper_refuses_its_own_lineage() -> None:
    """PLANTED, AND THE PLANT IS THIS PROCESS. Ask for our own tree to be killed; be refused.

    A REFUSAL RATHER THAN A FILTER, and the difference is the point: filtering ourselves out still
    leaves an ANCESTOR in the same tree, and killing that takes the shell or the runner with it --
    after which there is nobody left to report what happened.
    """
    assert_the_reaper_refuses_its_own_lineage()


def test_the_reaper_still_kills_a_process_that_is_not_ours() -> None:
    """THE OTHER SIDE OF THE SPARING. A reaper that spares everything is not a reaper.

    Confirmed by the planted process's own exit rather than by the return value alone, so the claim
    rests on an observation of the machine and not on the function agreeing with itself.
    """
    assert_the_reaper_kills_what_is_not_ours(
        interpreter=default_interpreter(), exit_ceiling_seconds=EXIT_SECONDS_CEILING
    )


def test_the_refusal_names_a_different_remedy_in_each_state() -> None:
    """The rule itself: TAKE THE REMEDY A REFUSAL NAMES -- so three states may not share one."""
    assert_each_state_names_its_own_remedy(
        capacity=REMEDY_CAPACITY, wider_tier=WIDER_TIER, narrowed_workers=NARROWED_WORKERS
    )


def test_a_width_is_computed_rather_than_guessed() -> None:
    """FLOOR ON THE BOX READING. Memory first and cores second, and never below the stated floor.

    STAYS LOCAL: the kit's shared body asserts the REFUSAL SENTENCES, and this asks a different
    question -- whether the width those sentences report is arithmetic on a real box reading. The
    two labs answer it against different bounds, so there is no family body here to take.
    """
    cores = logical_cores()
    assert cores >= 1, 'a box reporting no cores makes every width answer arithmetic on nothing'
    width = worker_width(gb_per_worker=4.0, max_width=cores, floor=1, reserve_cores=1, fallback=max(1, cores - 1))
    assert 1 <= width <= max(1, cores), 'the width is bounded by the box at both ends'
    assert blas_threads(width) >= 1, 'a worker with zero threads computes nothing'
    assert len(BLAS_THREAD_VARS) >= 4, 'the pool-spawning runtimes, by name rather than by count'
