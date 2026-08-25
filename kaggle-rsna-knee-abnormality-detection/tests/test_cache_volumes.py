#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for incremental volume-shard caching (pure helpers only).

No network, no DICOM: covers done-set planning from the series index +
stray local npz, shard-index continuation, index annotation, and
study-completion stamping into the study-level folds frame.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

from cache_volumes import (  # noqa: E402  # pylint: disable=wrong-import-position
  _VOL_SHARD_COL,
  annotate_index,
  load_index,
  next_shard_index,
  plan_work,
  stamp_completed_studies,
)

_BASE = 'ah2022-rsna-knee-abnormality-detection'


def _table(mapping: dict[str, list[str]]) -> pd.DataFrame:
  """Series table from study -> series list.

  Args:
      mapping: StudyInstanceUID to its series UIDs.

  Returns:
      Frame with Study/SeriesInstanceUID columns.
  """
  rows = [
    {'StudyInstanceUID': s, 'SeriesInstanceUID': v}
    for s, values in mapping.items()
    for v in values
  ]
  return pd.DataFrame(rows)


def _index(cached: dict[str, str]) -> pd.DataFrame:
  """Index frame from series -> shard name.

  Args:
      cached: SeriesInstanceUID to vol_shard value.

  Returns:
      Frame in load_index's shape.
  """
  return pd.DataFrame(
    {
      'SeriesInstanceUID': list(cached),
      'StudyInstanceUID': [f'st-{u}' for u in cached],
      _VOL_SHARD_COL: list(cached.values()),
    }
  )


class TestPlanWork:
  def test_indexed_series_are_done(self, tmp_path: Path):
    table = _table({'s0': ['a', 'b'], 's1': ['c']})
    todo = plan_work(table, _index({'a': 'sh0', 'b': 'sh0'}), tmp_path)
    assert list(todo['SeriesInstanceUID']) == ['c']

  def test_stray_local_npz_counts_done(self, tmp_path: Path):
    cache = tmp_path / 'volumes_cache' / f'{_BASE}-vol00'
    cache.mkdir(parents=True)
    (cache / 'a.npz').touch()
    table = _table({'s0': ['a', 'b']})
    todo = plan_work(table, _index({}), tmp_path)
    assert list(todo['SeriesInstanceUID']) == ['b']

  def test_sorted_deterministic_order(self, tmp_path: Path):
    table = _table({'s0': ['z', 'a', 'm']})
    todo = plan_work(table, _index({}), tmp_path)
    assert list(todo['SeriesInstanceUID']) == ['a', 'm', 'z']


class TestNextShardIndex:
  def test_first_run_starts_at_zero(self):
    assert next_shard_index([], _BASE) == 0

  def test_continues_after_published_shards(self):
    names = [f'{_BASE}-vol00', f'{_BASE}-vol01', f'{_BASE}-vol00']
    assert next_shard_index(names, _BASE) == 2

  def test_garbage_values_are_skipped(self):
    assert next_shard_index(['not-a-shard'], _BASE) == 0


class TestAnnotateIndex:
  def test_appends_rows(self):
    out = annotate_index(
      _index({'a': 'v0'}),
      [
        {
          'SeriesInstanceUID': 'c',
          'StudyInstanceUID': 'st-c',
          _VOL_SHARD_COL: 'v1',
        },
      ],
    )
    assert set(out['SeriesInstanceUID']) == {'a', 'c'}
    assert out.shape[1] == 3


class TestStampCompletedStudies:
  def test_stamps_only_fully_cached_studies(self):
    folds = pd.DataFrame(
      {
        'StudyInstanceUID': ['st-a', 'st-c'],
        'fold': [0, 1],
        _VOL_SHARD_COL: ['', ''],
      }
    )
    table = _table({'st-a': ['a', 'b'], 'st-c': ['c']})
    out = stamp_completed_studies(
      folds, table, cached_series={'a', 'b'}, shard_name='v0'
    )
    # st-a complete -> stamped; st-c missing its series -> untouched.
    assert out.loc[0, _VOL_SHARD_COL] == 'v0'
    assert out.loc[1, _VOL_SHARD_COL] == ''

  def test_does_not_overwrite_existing_stamp(self):
    folds = pd.DataFrame(
      {
        'StudyInstanceUID': ['st-a'],
        'fold': [0],
        _VOL_SHARD_COL: [f'{_BASE}-vol00'],
      }
    )
    table = _table({'st-a': ['a']})
    out = stamp_completed_studies(folds, table, {'a'}, 'v9')
    assert out.loc[0, _VOL_SHARD_COL] == f'{_BASE}-vol00'


class TestLoadIndex:
  def test_missing_file_returns_typed_empty(self, tmp_path: Path):
    empty = load_index(tmp_path / 'nope.parquet')
    assert empty.empty
    assert list(empty.columns) == [
      'SeriesInstanceUID',
      'StudyInstanceUID',
      _VOL_SHARD_COL,
    ]
