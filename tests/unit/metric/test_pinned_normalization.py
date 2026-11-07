"""A normalization frame is pinned on first read, so a report never rescales a run already scored."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from optimi_lab.metric import fixed_normalization, pinned_normalization


def test_the_first_read_pins_and_a_later_read_keeps_the_frame(tmp_path: Path) -> None:
    path = tmp_path / 'frames' / 'normalization.json'
    first = np.array([[0.0, 2.0], [1.0, 1.0]])
    frames = pinned_normalization(path, {'a': first})
    ideal, nadir = fixed_normalization(first)
    np.testing.assert_array_equal(frames['a'][0], ideal)
    np.testing.assert_array_equal(frames['a'][1], nadir)
    wider = np.array([[-5.0, 9.0], [9.0, -5.0]])
    again = pinned_normalization(path, {'a': wider, 'b': wider})
    np.testing.assert_array_equal(again['a'][0], ideal)  # pinned, never re-derived
    np.testing.assert_array_equal(again['b'][0], fixed_normalization(wider)[0])  # a new key pins now
    assert sorted(json.loads(path.read_text(encoding='utf-8'))) == ['a', 'b']


def test_a_key_with_no_rows_is_not_pinned_and_nothing_is_written(tmp_path: Path) -> None:
    path = tmp_path / 'normalization.json'
    assert pinned_normalization(path, {'a': np.empty((0, 2))}) == {}
    assert not path.exists()
    assert list(tmp_path.iterdir()) == []
