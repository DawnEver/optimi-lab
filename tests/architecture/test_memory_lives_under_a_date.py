"""Every memory entry sits under `YYYY/MM/DD/`, and its frontmatter agrees with the date it sits on.

WHY A SHAPE GUARD AND NOT A DOCUMENTED CONVENTION. The convention is `.claude/memory/YYYY/MM/DD/`
across every repo in this fleet, and the fleet has already measured what happens without a check:
one writer composed its records under a literal `lanes/` directory instead of a date, grew a
seventeen-file parallel tree beside the real one, and the index builder listed every file without
a word. It survived for days because the layout was an instruction, and an instruction is not a
control.

TWO PROPERTIES, and the second is what keeps this from being a directory-name test:

* the first three path parts are the DATE. Nesting BELOW the date (`attachments/`, say) is fine
  and is not what this is about.
* the `created:` field in an entry's frontmatter EQUALS the directory it sits in. A file copied
  forward into a new day keeps the old date in its header and becomes two conflicting answers to
  when the work happened; this is the only one of those two answers a reader ever sees.

THE FLOOR IS ONE, AND THAT IS STATED RATHER THAN HIDDEN. Measured 2026-09-15: the tree holds a
single entry, because this is the day it was created. One is the weakest a floor can be -- it
separates a tree that exists from a directory that does not, and nothing more. It goes UP with the
tree, and a walk that reaches zero files is refused rather than reported clean.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Final

_ROOT: Final = Path(__file__).resolve().parents[2]
_MEMORY: Final = _ROOT / '.claude' / 'memory'

#: `YYYY / MM / DD` -- the three leading parts of every memory path.
DATE_DEPTH: Final = 3

#: Measured 2026-09-15, the day the tree was created. Raise it with the tree; never lower it.
MEMORY_FILE_FLOOR: Final = 1

#: DEVICE-LOCAL and not part of the shared record: the memory tooling writes a `_meta.json` beside
#: the entries in each directory it manages. It reaches no index and no other checkout, so it is
#: not the parallel tree this guard exists for. A real fork is NOT in this set and is still caught.
DEVICE_LOCAL_NAMES: Final = frozenset({'_meta.json'})
DEVICE_LOCAL_DIRS: Final = frozenset({'__pycache__'})

_CREATED: Final = re.compile(r'^created:\s*(\d{4})-(\d{2})-(\d{2})\s*$', re.MULTILINE)


def entries(root: Path) -> list[Path]:
    """Every shared memory file under *root*, device-local debris excluded."""
    return sorted(
        path
        for path in root.rglob('*')
        if path.is_file()
        and path.name not in DEVICE_LOCAL_NAMES
        and not DEVICE_LOCAL_DIRS.intersection(path.relative_to(root).parts)
    )


def undated(root: Path, files: list[Path]) -> list[str]:
    """THE SHAPE GUARD. Files whose first three path parts are not a date, whatever put them there."""
    out = []
    for path in files:
        parts = path.relative_to(root).parts
        if len(parts) <= DATE_DEPTH or not all(part.isdigit() for part in parts[:DATE_DEPTH]):
            out.append(path.relative_to(root).as_posix())
    return sorted(out)


def date_disagreements(root: Path, files: list[Path]) -> dict[str, str]:
    """THE FRONTMATTER GUARD. Entries whose `created:` field contradicts the directory they sit in.

    A markdown entry with no `created:` line at all is a disagreement too: the field is how the
    entry answers for itself, and an entry that declines to is a date only the path asserts.
    """
    out: dict[str, str] = {}
    for path in files:
        if path.suffix != '.md':
            continue
        parts = path.relative_to(root).parts
        if len(parts) <= DATE_DEPTH:
            continue
        found = _CREATED.search(path.read_text(encoding='utf-8'))
        if found is None:
            out[path.relative_to(root).as_posix()] = 'no `created:` field'
        elif found.groups() != parts[:DATE_DEPTH]:
            out[path.relative_to(root).as_posix()] = f'frontmatter says {"-".join(found.groups())}'
    return out


def test_every_memory_entry_sits_under_its_own_date() -> None:
    """THE CHECK, floor asserted first, over the tree whatever route a file arrived by."""
    assert _MEMORY.is_dir(), (
        f'{_MEMORY} does not exist, so this test scanned nothing. The memory tree is the shape this '
        f'guard is about; an absent one is the gap, not a pass.'
    )
    files = entries(_MEMORY)
    assert len(files) >= MEMORY_FILE_FLOOR, (
        f'{len(files)} memory file(s) under {_MEMORY}, below the {MEMORY_FILE_FLOOR} floor -- a green '
        f'here would be a statement about an empty directory.'
    )
    forked = undated(_MEMORY, files)
    assert not forked, (
        f'{forked} sit outside a YYYY/MM/DD directory -- a parallel tree an index will list and no '
        f'reader will find. The date is the path, not a field inside the file.'
    )
    disagreeing = date_disagreements(_MEMORY, files)
    assert not disagreeing, (
        f'{disagreeing} contradict the directory they sit in. An entry carried forward into a new day '
        f'keeps its old header, and the header is the only date a reader sees.'
    )


def test_the_shape_guard_names_a_planted_fork(tmp_path: Path) -> None:
    """THE CONTROL. Plant the fork in the region and call the REAL guard on it."""
    good = tmp_path / '2026' / '09' / '15'
    nested = good / 'attachments'
    nested.mkdir(parents=True)
    (good / 'entry.md').write_text('---\ncreated: 2026-09-15\n---\n', encoding='utf-8')
    (nested / 'figure.txt').write_text('below the date is fine\n', encoding='utf-8')
    (tmp_path / 'lanes').mkdir()
    (tmp_path / 'lanes' / 'a-lane.md').write_text('---\ncreated: 2026-09-15\n---\n', encoding='utf-8')
    (tmp_path / 'loose.md').write_text('---\ncreated: 2026-09-15\n---\n', encoding='utf-8')
    (good / '_meta.json').write_text('{}', encoding='utf-8')

    found = entries(tmp_path)
    assert '_meta.json' not in {p.name for p in found}, 'device-local debris must not enter the scan.'
    assert undated(tmp_path, found) == ['lanes/a-lane.md', 'loose.md'], (
        f'the guard named {undated(tmp_path, found)}. The literal-directory fork and the loose file at '
        f'the root must both come back, and nesting below the date must not.'
    )


def test_the_frontmatter_guard_names_a_planted_stale_date(tmp_path: Path) -> None:
    """THE CONTROL for the second property: a copied-forward header, and a missing one."""
    day = tmp_path / '2026' / '09' / '15'
    day.mkdir(parents=True)
    (day / 'agrees.md').write_text('---\nname: a\ncreated: 2026-09-15\n---\n# a\n', encoding='utf-8')
    (day / 'stale.md').write_text('---\nname: b\ncreated: 2026-09-04\n---\n# b\n', encoding='utf-8')
    (day / 'silent.md').write_text('# c\n', encoding='utf-8')
    found = date_disagreements(tmp_path, entries(tmp_path))
    assert found == {
        '2026/09/15/stale.md': 'frontmatter says 2026-09-04',
        '2026/09/15/silent.md': 'no `created:` field',
    }, f'the guard answered {found}; the agreeing entry must not be named and the silent one must be.'
