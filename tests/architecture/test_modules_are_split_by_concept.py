"""optimi_lab's empty size-split debt: modules name concepts, not overflow positions."""

from pathlib import Path

import pytest
from dev_lab.dev.famtests.sizesplits import SizeSplit, assert_no_new_size_splits
from dev_lab.dev.rules import tracked_files

_ROOT = Path(__file__).resolve().parents[2]


def test_module_names_do_not_number_their_overflow() -> None:
    modules = frozenset(path for path in tracked_files(_ROOT) if path.endswith('.py'))
    assert len(modules) >= 200
    assert_no_new_size_splits(modules, debt=frozenset())


def test_the_guard_refuses_a_planted_overflow_module() -> None:
    with pytest.raises(SizeSplit, match='rows_part2'):
        assert_no_new_size_splits(('src/pkg/rows.py', 'src/pkg/rows_part2.py'), debt=frozenset())
