"""What has been RETIRED from `optimi_lab`, and the scan that makes a resurrection impossible.

THIS REGISTRY HAS A SUBJECT, which is the only reason it exists. Two breaking refactors in
September 2026 deleted public names outright -- deliberately, with no compat shim and no
deprecation alias -- and recorded which ones only in their commit messages. A commit message is
read once. A deletion is an event; a retirement is a standing property, and the spelling comes
back through a copied example, a stale branch, a resurrected doc or an agent working from an old
snippet. An EMPTY registry would be the vacuous green this family refuses; this one is populated
from names that were actually removed, and the scan below is what is responsible for keeping them
gone rather than anyone's memory of having deleted them.

WHAT BELONGS HERE: a symbol or module basename that must no longer appear anywhere in `src/`,
`tests/`, `scripts/` or `.claude/`. Every entry names its REPLACEMENT, so the failure message
tells a reader what to write instead rather than only what not to. An entry is a hard NO; if it
needs an exception it is not retired yet.

THE COST OF REGISTERING, discovered here exactly as it was discovered upstream: **a hard NO
applies to the EXPLANATION as much as to a caller.** Registering these names reddened three
docstrings that were describing their own removal, because the scan does not know prose from code
and is right not to -- a copied example is precisely how a dead spelling comes back. So those
docstrings now name the SHAPE of what went and point here, which is why they read the way they do.
This module is the one place a retired spelling may be written, and it excludes itself by path.

THE MATCH IS ON IDENTIFIER BOUNDARIES, not substrings. Plain substring matching cannot hold any
prefix-or-suffix rename -- registering `foo` when the replacement is `all_foo` flags every use of
the replacement, so the entry could never be green and would simply never be written, leaving the
guarantee quietly absent rather than loudly broken.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Final

from lab_commons.dev import floors

_ROOT: Final = Path(__file__).resolve().parents[2]

#: A retired spelling to its REPLACEMENT -- what a reader who reaches for the old name writes now.
#: Taken from the two breaking commits that removed them, `refactor(core)!` and
#: `refactor(algorithms,surrogates)!`.
RETIRED: Final[dict[str, str]] = {
    # The single `inf` sentinel folded into the output matrix, defined byte-identically in two
    # modules. A non-OK cell now holds NaN and cannot be read as "very bad".
    'PENALTY_VALUE': 'optimi_lab.core.outcomes.Outcome, one per cell',
    # The flag that let predicted values be scored as if measured.
    'use_surrogate_model': 'Surrogate.predict returns a plain array no Outcome can be built from',
    # A list-valued parameter defaulting to `[]`, read at `[0]`, repaired by aliasing one dict.
    'base_params_dict_list': 'optimi_lab.core.space.Variable',
    # NOT REGISTERED, and the reason is the registry's own admission rule rather than an oversight:
    # the post-training validity flag is ALREADY refused by a live `not hasattr(model, ...)`
    # assertion in `tests/unit/optimizer/test_surrogate_models.py`, which names the attribute in
    # order to prove it is gone. A spelling that one file must legitimately write is a spelling
    # this scan would need an exception for, and an entry needing an exception is not retired yet.
    # A `hasattr` check on the real object is the stronger of the two mechanisms anyway: it answers
    # for the class, where this scan only answers for the text.
    #
    # The eleven one-regressor modules and the two base modules they shared.
    'surrogate_model_base': 'optimi_lab.surrogate_model.base',
    'mixture_surrogate_model': 'optimi_lab.surrogate_model.mixture',
    'pso_operators': 'optimi_lab.intelligent_algorithm.swarm',
    # The `key@oc<N>@max|min` grammar and the package that parsed it.
    'object_function': 'optimi_lab.core.space.Objective, whose fields are not a parse',
}

#: The trees a retired spelling may not appear in. `archived/` and `examples/` are deliberately
#: outside it: the first is a record of what the package WAS, and a record that is rewritten when a
#: name is retired is no longer a record.
SCANNED: Final = ('src', 'tests', 'scripts', '.claude')

#: Floors. Both directions matter: a registry with nothing in it proves nothing, and a scan that
#: reached three files reports exactly what a clean tree reports. Measured 2026-09-15.
#: RE-MEASURED 2026-09-18: 7 spellings registered.
REGISTRY_FLOOR: Final = 6

#: THE OTHER SIDE OF ``REGISTRY_FLOOR``. Today's reading is 7 - 6 = 1.
REGISTRY_HEADROOM: Final = 5

#: RE-MEASURED 2026-09-18: the scan reaches 63 files. THE OLD NUMBER WAS 25, a slack of 38 that
#: would have passed a walk reaching two fifths of the tree; re-measured rather than absorbed by a
#: wider headroom.
SCANNED_FILE_FLOOR: Final = 50

#: THE OTHER SIDE OF ``SCANNED_FILE_FLOOR``. Today's reading is 63 - 50 = 13.
SCANNED_FILE_HEADROOM: Final = 25

#: The one file allowed to write these spellings: this one. Every other mention is a resurrection,
#: including a comment explaining the retirement.
_SELF: Final = Path(__file__).resolve()

_SUFFIXES: Final = ('.py', '.md', '.toml', '.cfg', '.yaml', '.yml')


def matcher(spelling: str) -> re.Pattern[str]:
    """*spelling* as a whole identifier -- never as a substring of a longer one.

    This is what lets the registry hold a rename whose replacement CONTAINS the retired name.
    """
    return re.compile(rf'(?<![A-Za-z0-9_]){re.escape(spelling)}(?![A-Za-z0-9_])')


def scanned_files(root: Path) -> list[Path]:
    """Every text file under the scanned trees of *root*, this registry excluded."""
    return sorted(
        path
        for tree in SCANNED
        for path in (root / tree).rglob('*')
        if path.is_file() and path.suffix in _SUFFIXES and '__pycache__' not in path.parts and path.resolve() != _SELF
    )


def resurrections(files: list[Path], retired: dict[str, str]) -> dict[str, list[str]]:
    """THE GUARD. Each retired spelling to the files that still write it."""
    found: dict[str, list[str]] = {}
    for path in files:
        text = path.read_text(encoding='utf-8', errors='ignore')
        for spelling in retired:
            if matcher(spelling).search(text):
                found.setdefault(spelling, []).append(
                    path.relative_to(_ROOT).as_posix() if path.is_relative_to(_ROOT) else path.as_posix()
                )
    return found


def test_the_registry_has_a_subject() -> None:
    """A registry with nothing in it is not an enforcement, it is a file shaped like one."""
    floors.assert_floor(len(RETIRED), floor=REGISTRY_FLOOR, what='RETIRED-SPELLING registry')
    floors.assert_floor_still_binds(
        len(RETIRED), floor=REGISTRY_FLOOR, headroom=REGISTRY_HEADROOM, what='RETIRED-SPELLING registry'
    )
    unexplained = sorted(name for name, replacement in RETIRED.items() if not replacement.strip())
    assert not unexplained, (
        f'{unexplained} are registered with no replacement. A refusal that says only what not to write '
        f'sends the reader to the source for the answer it was supposed to hand them.'
    )


def test_no_retired_spelling_is_written_anywhere() -> None:
    """THE CHECK, over every scanned tree, with the size of the scan asserted before its verdict."""
    files = scanned_files(_ROOT)
    floors.assert_floor(len(files), floor=SCANNED_FILE_FLOOR, what='RETIRED-SPELLING (scanned trees)')
    floors.assert_floor_still_binds(
        len(files), floor=SCANNED_FILE_FLOOR, headroom=SCANNED_FILE_HEADROOM, what='RETIRED-SPELLING (scanned trees)'
    )
    found = resurrections(files, RETIRED)
    assert not found, '\n'.join(
        f'{spelling!r} is retired and still written in {where} -- write {RETIRED[spelling]} instead.'
        for spelling, where in sorted(found.items())
    )


def test_the_guard_names_a_planted_resurrection(tmp_path: Path) -> None:
    """THE CONTROL. Plant each retired spelling in a file and call the REAL guard on it."""
    planted = tmp_path / 'caller.py'
    planted.write_text('\n'.join(f'x = {name}' for name in RETIRED) + '\n', encoding='utf-8')
    found = resurrections([planted], RETIRED)
    assert set(found) == set(RETIRED), (
        f'the guard missed {sorted(set(RETIRED) - set(found))} when every one of them was planted in a '
        f'single file. A scan that cannot fail on a planted violation is not evidence about the tree.'
    )


def test_the_match_is_on_identifier_boundaries() -> None:
    """The limit that would otherwise make a prefix-or-suffix rename impossible to register."""
    pattern = matcher('object_function')
    assert pattern.search('from optimi_lab import object_function')
    assert pattern.search('module.object_function')
    assert not pattern.search('an_object_function()'), 'a longer identifier ending in the name is not it.'
    assert not pattern.search('object_functional'), 'a longer identifier starting with the name is not it.'


def test_the_scan_excludes_only_this_registry() -> None:
    """Only this registry may write the spellings -- not even a doc that explains them.

    Takes no `tmp_path`: it asks the REAL corpus which files are scanned, and a temporary tree
    would answer a question nobody asked.
    """
    files = scanned_files(_ROOT)
    assert _SELF not in {path.resolve() for path in files}, 'the registry must exclude itself.'
    assert any(path.suffix == '.md' for path in files), (
        'the scan reaches no markdown, so prose -- where a retired spelling survives longest -- is '
        'outside it, and the guarantee is narrower than it reads.'
    )
