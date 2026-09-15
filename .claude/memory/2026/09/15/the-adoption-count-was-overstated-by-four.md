---
name: the-adoption-count-was-overstated-by-four
description: optimi-lab's rules adoption claimed 12 of 28 enforced; two of the three overstatement claims held (the surface guard planted nothing for seven rules, and both of its pin sets were empty so the two-sided ratchet never executed), one was refuted (the collection error is a stale venv, not the file), and NAMED-SETS-NOT-COUNTS was found exemplified rather than enforced. Fixed by building the missing controls rather than demoting, and four real gaps closed: 16 of 28, ceiling 16 -> 12.
metadata:
  type: project
created: 2026-09-15
accessed: 2026-09-15
---

# The adoption count was overstated, and the interesting part is which claim was wrong

optimi-lab joined the shared rules registry claiming **12 of 28** rules enforced with 16 declared
absent. A review named three ways that number was inflated. Two held. One did not, and the one
that did not is the more useful finding.

## Confirmed: the surface guard planted nothing

`tests/architecture/test_the_public_surface_is_declared.py` was cited as the mechanism for SEVEN
rules -- including `PLANTED-CONTROL` itself, by way of `FLOOR-ON-EVERY-SCAN` -- and contained no
test that could demonstrate any of its five scans failing. Its sibling purity guard did plant
(`test_the_guard_names_a_planted_heavyweight`), which is presumably why nobody noticed the other
one did not: the two files read as a pair.

The repair was not to demote the seven rules. It was to extract the comparisons the live tests
perform (`assert_floor`, `second_homes`, `oversize`, `skipping`, `pin_gap`) and have controls call
**those same functions** on planted input. A control that re-implements the comparison agrees with
itself and proves nothing about the code that ships.

**The control's first run was itself the evidence.** Writing the planted skip markers as literals
made this module the live skip scan's first offender. The scan fired, on the file that exists to
prove it can. The planted tokens are assembled from fragments now, for the same reason the
registered-spelling rule says a dead name survives longest in the prose explaining it.

## Confirmed: both pin sets are empty, so neither side of the ratchet ran

`OVERSIZE_PINS` and `LAZY_IMPORT_PINS` are both empty -- correctly, as measurements: no module is
over the band and nothing imports lazily. But the live assertion was `found == set(PINS)`, an
empty set compared against an empty set, which is a tautology wearing a ratchet's clothes and the
same shape that let an emptied table pass elsewhere in this family.

Seeding a fake pin would have been the *other* side of the same defect -- a waiver nothing uses.
The answer was `pin_gap`, driven in each direction by a control, so the empty live answer now
means *clean* rather than *unread*.

## Refuted: the collection error is the environment, not the file

Three rules cite the adoption module, which imports `lab_commons.dev`. That import does fail --
but only against the shared venv's stale installed `lab_commons`. Under the run this repo
prescribes the module collects and its four tests pass, and when it cannot import it raises a
COLLECTION ERROR, which is loud. A mechanism that reds when its dependency is missing is behaving;
a stale install is a fact about one box.

## Found instead: a rule exemplified rather than enforced

The claim worth making was one nobody had made. `NAMED-SETS-NOT-COUNTS` cited three modules that
merely *happen* to use named sets. Nothing refused the edit replacing a named pin with an integer
one -- the rule was illustrated by the code rather than checked by it. That is the
declaration-that-lies defect at the level of the registry itself, and it is the failure mode a
review looking for *broken* mechanisms will walk straight past, because everything it points at is
green and correct.

`test_a_pin_is_a_named_set.py` now refuses a numeric module-level constant in any architecture
module unless its name declares it a threshold. The exemption is a *suffix*, which is a
declaration the code consults rather than a list a maintainer curates.

## The lesson, stated so it can be reused

Three kinds of overstatement turned up, and they are worth telling apart:

* a mechanism that cannot RUN -- loud, and usually not the problem;
* a mechanism that runs and cannot FAIL -- silent, and found by asking each guard "what input
  would red this?";
* a mechanism that does not EXIST, hidden behind a citation of code that obeys the rule.

The third is the hardest to see and was the only one this review had not already named.
