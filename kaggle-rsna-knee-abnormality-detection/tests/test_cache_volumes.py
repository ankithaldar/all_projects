#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for the incremental volume-shard cacher (pure helpers only).

No network, no DICOM: covers done-set planning (annotations + stray
local npz), shard-index continuation, and fold annotations.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

from cache_volumes import (  # noqa: E402  # pylint: disable=wrong-import-position
  _VOL_SHARD_COL,
  annotate,
  load_folds,
  next_shard_index,
  plan_work,
)

_BASE = 'ah2022-rsna-knee-abnormality-detection'


def _folds(uids: list[str], shards: list[str] | None = None):
  """Folds frame with the given series and optional shard stamps.

  Args:
      uids: SeriesInstanceUID list.
      shards: Matching vol_shard values ('' when omitted).

  Returns:
      Frame with StudyInstanceUID/fold/series/vol_shard columns.
  """
  n = len(uids)
  return pd.DataFrame(
    {
      'StudyInstanceUID': [f'st{i}' for i in range(n)],
      'fold': [i % 5 for i in range(n)],
      'SeriesInstanceUID': uids,
      _VOL_SHARD_COL: shards if shards is not None else [''] * n,
    }
  )


class TestPlanWork:
  def test_annotations_mark_done(self, tmp_path: Path):
    table = pd.DataFrame(
      {
        'SeriesInstanceUID': ['a', 'b', 'c'],
        'StudyInstanceUID': ['s0', 's1', 's2'],
      }
    )
    folds = _folds(['a', 'b', 'c'], [f'{_BASE}-vol00', f'{_BASE}-vol00', ''])
    todo = plan_work(table, folds, tmp_path)
    assert list(todo['SeriesInstanceUID']) == ['c']

  def test_stray_local_npz_counts_done(self, tmp_path: Path):
    cache = tmp_path / 'volumes_cache' / f'{_BASE}-vol00'
    cache.mkdir(parents=True)
    (cache / 'b.npz').touch()
    table = pd.DataFrame(
      {
        'SeriesInstanceUID': ['a', 'b'],
        'StudyInstanceUID': ['s0', 's1'],
      }
    )
    todo = plan_work(table, _folds(['a', 'b']), tmp_path)
    assert list(todo['SeriesInstanceUID']) == ['a']

  def test_sorted_deterministic_order(self, tmp_path: Path):
    table = pd.DataFrame(
      {
        'SeriesInstanceUID': ['z', 'a', 'm'],
        'StudyInstanceUID': ['s'] * 3,
      }
    )
    todo = plan_work(table, _folds(['z', 'a', 'm']), tmp_path)
    assert list(todo['SeriesInstanceUID']) == ['a', 'm', 'z']


class TestNextShardIndex:
  def test_first_run_starts_at_zero(self):
    assert next_shard_index(_folds(['a']), _BASE) == 0

  def test_continues_after_published_shards(self):
    folds = _folds(
      ['a', 'b', 'c'],
      [
        f'{_BASE}-vol00',
        f'{_BASE}-vol01',
        f'{_BASE}-vol00',
      ],
    )
    assert next_shard_index(folds, _BASE) == 2

  def test_garbage_values_are_skipped(self):
    folds = _folds(['a'], ['not-a-shard'])
    assert next_shard_index(folds, _BASE) == 0


class TestAnnotate:
  def test_stamps_only_target_uids(self):
    folds = _folds(['a', 'b', 'c'])
    out = annotate(folds, ['b', 'c'], f'{_BASE}-vol03')
    assert list(out[_VOL_SHARD_COL]) == ['', f'{_BASE}-vol03', f'{_BASE}-vol03']
    # original untouched (copy semantics)
    assert list(folds[_VOL_SHARD_COL]) == ['', '', '']


class TestLoadFolds:
  def test_adds_missing_column(self, tmp_path: Path):
    path = tmp_path / 'train_folds.csv'
    pd.DataFrame({'StudyInstanceUID': ['s0'], 'fold': [0], 'a': [1]}).to_csv(
      path, index=False
    )
    folds = load_folds(path)
    assert _VOL_SHARD_COL in folds.columns
    assert list(folds['a']) == [1]
