r"""optimi-lab's DELTA against each family config base -- data, computed by nothing.

The bases live in :mod:`lab_commons.dev._famconfig_rows` and the renderer that turns base plus delta
into a file is :mod:`lab_commons.dev.famconfig`. What belongs HERE is only the part of each artefact
that is a fact about optimi-lab: a `usr/local/` tree, a viztracer `result.json`, a coredumpy
`dumps/`. Everything else is the family's and arrives from the kit, so a fix upstream lands here by
upgrading the dependency rather than by remembering to copy it.

WHY THE DELTA IS HAND-WRITTEN AND NOT `measured_delta` OUTPUT. `measured_delta` is the SURVEY half --
it reads the file back and reports what it would have to declare -- and it strips comments, because
it compares meaningful lines. Handing its result straight back as the declaration would delete every
comment in the artefact on the first render. So the survey SIZED this change (18 added / 2 dropped)
and a human wrote the lines.

THE ONE BEHAVIOUR CHANGE, STATED RATHER THAN SLIPPED IN AS FORMATTING. The base spells five patterns
with a trailing slash -- `**/__pycache__/`, `*.egg-info/`, `.mypy_cache/`, `.pytest_cache/`,
`.ruff_cache/` -- and this repo wrote the first two BARE. A trailing slash matches a DIRECTORY ONLY,
so adopting the base narrows those two rules: a FILE named `__pycache__` or ending in `.egg-info`
stops being ignored. MEASURED over this tree on 2026-09-17 with `git status --ignored` before and
after: the only line that moved was `.gitignore` itself, so no file's tracked-or-ignored status
changed, because no such file exists here. The bare spellings are deliberately NOT carried in the
delta -- the delta renders AFTER the base and gitignore is last-match-wins, so keeping them would
silently reinstate exactly the rule the base replaced.

WHAT THIS REPO DOES NOT HAVE, and it is worth one sentence because the sibling lab does. There is no
`**/.claude/**` block here and no SUBTREE re-inclusion, so the last-match-wins hazard does not arise.
THE QUALIFIER IS LOAD-BEARING AND THIS SENTENCE USED TO READ "no re-inclusions", WHICH IS FALSE: the
delta carries `!example.log`, and it has since the adoption. What makes it harmless is not its
absence but its SHAPE -- `reopenings` reports a negation only when it ends in a wildcard, because a
negation naming ONE FILE re-includes that path and nothing under it and so owes no closure, where a
subtree negation brings back every cache directory a floating base rule excludes at any depth. So the
correct claim is about the shape of this repo's one negation rather than about there being none, and
the blanket version was a declaration asserting a property the file does not have.

THE SHAPE OF THE NEGATION IS A FACT ABOUT THIS TREE TODAY rather than a guarantee, so the ordering
property in the test module is written to bite if a SUBTREE negation is ever added -- and since
2026-09-17 it bites through the kit rather than through anything written here, because
`assert_no_rule_is_reopened` refuses a base directory rule re-stated ABOVE a
subtree negation that re-includes it. The condition is BELOW EVERY RE-INCLUSION IT CLOSES; naming a
particular line as the FINAL one was a sufficient approximation of it that no repo needs any more.

`.pre-commit-config.yaml` IS ADOPTED AS OF 2026-09-17, and the waiver that used to sit here is gone
with its subject. Both causes it named are dead: `Delta.anchored` places a line INSIDE a rendered
block without restating one, and `delta_problems` now compares against `content_lines`, so a blank
is no longer read as a re-statement. See :data:`PRECOMMIT_DELTA`.

THE ADOPTION MOVES THREE THINGS HERE AND EACH IS MEASURED IN THE TEST MODULE RATHER THAN ASSERTED:
the `pre-commit-hooks` pin from v5.0.0 to v6.0.0 and the `commitizen` pin from v4.6.0 to v4.13.9,
which is the base taking the newest measured pin of the three consumers; and `default_stages:
[pre-commit]`, which narrows every hook whose upstream manifest declares no stages of its own from
eleven stages to one. THE NARROWING COSTS THIS REPO NOTHING, and that is a fact about this checkout
rather than an argument: MEASURED 2026-09-17, the only git hook installed here is `pre-commit`, so
the ten stages the narrowing removes had no hook to run at. The sibling lab, which HAS a pre-push
hook, records the same change as a real one -- the base is identical and the consequence is not.

A FOURTH THING DISAPPEARS AND IT IS ONLY COMMENTS. This repo's `.pre-commit-config.yaml` carried a
commented-out `- repo: local` block -- `generate-changelog`, `lint-python`, `fmt-python`, `codespell`,
`mypy`, `pytest`, none of them live. A rendered artefact holds the base plus this repo's declared
delta, so a commented draft of hooks nobody runs is not carried across. Nothing executed it before
and nothing executes it now; what is lost is a menu, and `Makefile` already holds the live spelling
of every item on it.
"""

from __future__ import annotations

from typing import Final

from lab_commons.dev.famconfig import Delta

__all__ = [
    'DELTAS',
    'EXTRA_HOOK_IDS',
    'GITIGNORE_DELTA',
    'MAKEFILE_DELTA',
    'PRECOMMIT_DELTA',
    'PRECOMMIT_STAGE_MOVE',
    'REPO',
]

#: How this repo names itself in a rendered file's drop comments.
REPO: Final = 'optimi-lab'

#: 18 content lines were MEASURED by `famconfig.measured_delta`; 16 are declared, because the two
#: bare slash-less spellings are what the base REPLACES rather than what this repo adds.
GITIGNORE_DELTA: Final = Delta(
    repo=REPO,
    added=(
        '## Build and distribution',
        'build/',
        'dist/',
        '**/__version__.py',
        '',
        '## Development tools',
        '## Documentation (pdoc)',
        '**/docs/',
        '## viztracer',
        'result.json',
        '## coredumpy',
        'dumps/',
        '',
        '## Quality and testing',
        'report/',
        'htmlcov/',
        '# Logs written by the family verify entry point (`lab_commons.dev.verify`).',
        '.verify/',
        '',
        '## Runtime and environment',
        '**/log/*.png',
        '**/log/*.jpg',
        '!example.log',
        'usr/local/',
        'output/',
        '',
        '## System and others',
        '**/ignore/',
        '**/.DS_Store',
    ),
    dropped={},
    #: 29 declared lines against 16 measured content lines. The number is the point at which this
    #: stops being a delta: this tree is one Python package with no second language and no agent
    #: state to re-include, so a delta that doubled would be saying the base had stopped fitting.
    ceiling=32,
)

#: The `pre-commit-hooks` ids this repo runs BEYOND the family's eleven, MEASURED 2026-09-17 against
#: the live file. Held as data because it is the SUBJECT of the anchor in :data:`PRECOMMIT_DELTA`:
#: these eight are the lines that had nowhere to go while a delta could only append, and the test
#: module reads them back off the rendered artefact so the named set cannot drift from the file.
EXTRA_HOOK_IDS: Final[tuple[str, ...]] = (
    'check-builtin-literals',
    'check-illegal-windows-names',
    'check-shebang-scripts-are-executable',
    'check-symlinks',
    'check-vcs-permalinks',
    'destroyed-symlinks',
    'fix-byte-order-marker',
    'requirements-txt-fixer',
)

#: THE ONE BEHAVIOUR CHANGE THE ADOPTION MAKES, MEASURED through `pre_commit.repository.all_hooks`
#: on 2026-09-17 rather than assumed. Before: no `default_stages` here, so every stock hook whose
#: UPSTREAM manifest declares no stages of its own inherited all eleven. After: the base's
#: `default_stages: [pre-commit]` narrows these thirteen to pre-commit alone.
#:
#: IT IS NOT A DROP AND IT COSTS THIS CHECKOUT NOTHING, which is a fact about the checkout rather
#: than an argument: the only git hook installed here is `pre-commit`, so the ten stages it removes
#: had nothing to run at. The sibling lab records the IDENTICAL change against the IDENTICAL base as
#: a real loss, because it has a pre-push hook -- so the test module pins the premise as well as the
#: set, in both directions.
#:
#: DECLARED 2026-09-17 WITH THE FAMILY BODY, and its absence until then was a one-sided reading: the
#: file pinned which hooks KEPT pre-push and nothing named which ones MOVED, so a hook joining the
#: narrowing would have arrived in silence. A ratchet has two sides.
PRECOMMIT_STAGE_MOVE: Final[tuple[str, ...]] = (
    'check-ast',
    'check-builtin-literals',
    'check-case-conflict',
    'check-illegal-windows-names',
    'check-json',
    'check-merge-conflict',
    'check-symlinks',
    'check-toml',
    'check-vcs-permalinks',
    'debug-statements',
    'fix-byte-order-marker',
    'mixed-line-ending',
    'requirements-txt-fixer',
)

#: What this repo adds to the family's `.pre-commit-config.yaml`, and it adds nothing at the END.
#: Every one of :data:`EXTRA_HOOK_IDS` belongs INSIDE the `pre-commit-hooks` entry the base renders,
#: which is what `Delta.anchored` is for: the ids hang off `check-added-large-files`, a base line
#: that occurs exactly ONCE, which is what makes it a position an anchor can name at all. Nothing is
#: restated, so the anti-fork arm is undiminished.
#:
#: THERE IS NO `- repo: local` ENTRY HERE and that is why this delta is eight lines while the sibling
#: lab's is sixty-two. This repo runs no hook of its own: `lint`, `fmt` and `test` are Makefile
#: targets a human invokes, not git hooks, and the commented-out block the old file carried was a
#: draft nobody had ever enabled.
PRECOMMIT_DELTA: Final = Delta(
    repo=REPO,
    added=(),
    dropped={},
    anchored={'      - id: check-added-large-files': tuple(f'      - id: {hook_id}' for hook_id in EXTRA_HOOK_IDS)},
    #: Eight declared lines, one per extra id, and anchored lines count against a ceiling or
    #: anchoring would be a ceiling nobody chose. Four of headroom: this repo adds stock hook ids and
    #: nothing else, so a delta reaching thirteen would mean it had started running hooks of its own,
    #: which is a decision to take deliberately rather than to discover here.
    ceiling=12,
)

#: The Makefile base is REQUIRED rather than RENDERED -- target NAMES must be present and the recipes
#: are this repo's. So this repo adds no lines and drops none: MEASURED 2026-09-17, all nine base
#: lines are already satisfied here, including `verify:`'s recipe, which is the family's line
#: verbatim because this repo declares no `slow` marker to tier against.
MAKEFILE_DELTA: Final = Delta(repo=REPO, added=(), dropped={}, ceiling=0)

#: Every artefact this repo declares a delta against, by name. All three of the kit's bases are
#: here as of 2026-09-17, and the test pins that this set covers the kit's EXACTLY, so a fourth
#: base cannot arrive unnoticed.
DELTAS: Final[dict[str, Delta]] = {
    '.gitignore': GITIGNORE_DELTA,
    '.pre-commit-config.yaml': PRECOMMIT_DELTA,
    'Makefile': MAKEFILE_DELTA,
}
