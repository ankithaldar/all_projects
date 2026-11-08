#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''P3-A purged CV folds tests (Contract 4).'''

import inspect
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest


def _daily_frame(n=100, start='2024-01-01'):
  '''Return daily frame with f1 and target.'''
  times = pd.date_range(start=start, periods=n, freq='D', tz='UTC')
  return pd.DataFrame({
    'time': times,
    'f1': np.arange(n, dtype=float),
    'target': [0, 1] * (n // 2),
  })


def test_folds_golden_73_20_7_via_fold_path():
  '''Two folds reach the 73/20/7 golden on the last fold.'''
  from forecast_ml_framework.pipelines.modeling.purged_cv import (
    purged_cv_folds,
  )
  from forecast_ml_framework.pipelines.modeling.data_split import (
    assert_no_leakage,
  )
  df = _daily_frame(100)
  params = {'time_col': 'time', 'n_splits': 2, 'test_size': 0.2,
            'embargo': timedelta(days=7), 'purge': None,
            'window': 'expanding'}
  folds = purged_cv_folds(df, params)
  assert len(folds) == 2
  assert folds[0]['fold_index'] == 0
  assert folds[1]['fold_index'] == 1
  last = folds[1]['fold_meta']
  assert last['n_train'] == 73
  assert last['n_test'] == 20
  assert last['n_embargoed'] == 7
  assert last['n_purged'] == 0
  assert last['embargo'] == 'P7D'
  assert last['purge'] == 'P7D'
  first = folds[0]['fold_meta']
  assert first['n_train'] == 53
  assert first['n_test'] == 20
  assert first['n_embargoed'] == 7
  assert first['n_purged'] == 0
  for fold, total in zip(folds, [80, 100]):
    meta = fold['fold_meta']
    assert meta['n_train'] + meta['n_embargoed'] \
        + meta['n_purged'] + meta['n_test'] == total
    assert meta['sampling_method'] == 'purged_cv'
    assert meta['time_col'] == 'time'
    assert meta['n_splits'] == 2
    assert meta['test_size'] == 0.2
    assert meta['leakage_checked'] is True
    assert meta['window'] == 'expanding'
    assert meta['max_time_train'] is not None
    assert meta['min_time_test'] is not None
    assert meta['fold_index'] == fold['fold_index']
    assert_no_leakage(meta)


def test_folds_pairwise_disjoint_and_increasing():
  '''Test index sets are disjoint and mins increase.'''
  from forecast_ml_framework.pipelines.modeling.purged_cv import (
    purged_cv_folds,
  )
  df = _daily_frame(100)
  folds = purged_cv_folds(df, {'time_col': 'time', 'n_splits': 2,
                               'test_size': 0.2,
                               'embargo': timedelta(days=7)})
  assert set(folds[0]['test_df'].index).isdisjoint(
    folds[1]['test_df'].index)
  mins = [pd.to_datetime(f['fold_meta']['min_time_test'], utc=True)
          for f in folds]
  assert mins[1] > mins[0]
  assert set(folds[0]['train_df'].index) <= set(df.index)
  assert set(folds[1]['train_df'].index) <= set(df.index)


def test_folds_purge_beyond_embargo_70_20_7_3():
  '''Purge 10d beyond embargo 7d yields 70/20/7/3.'''
  from forecast_ml_framework.pipelines.modeling.purged_cv import (
    purged_cv_folds,
  )
  df = _daily_frame(100)
  params = {'time_col': 'time', 'n_splits': 2, 'test_size': 0.2,
            'embargo': timedelta(days=7),
            'purge': timedelta(days=10)}
  folds = purged_cv_folds(df, params)
  last = folds[1]['fold_meta']
  assert last['n_train'] == 70
  assert last['n_test'] == 20
  assert last['n_embargoed'] == 7
  assert last['n_purged'] == 3
  assert last['embargo'] == 'P10D'
  assert last['purge'] == 'P10D'
  first = folds[0]['fold_meta']
  assert first['n_train'] == 50
  assert first['n_embargoed'] == 7
  assert first['n_purged'] == 3


def test_folds_embargo_above_purge_keeps_checked_max():
  '''Embargo above purge records max and purge apart.'''
  from forecast_ml_framework.pipelines.modeling.purged_cv import (
    purged_cv_folds,
  )
  df = _daily_frame(100)
  params = {'time_col': 'time', 'n_splits': 1, 'test_size': 0.2,
            'embargo': timedelta(days=10),
            'purge': timedelta(days=7)}
  meta = purged_cv_folds(df, params)[0]['fold_meta']
  assert meta['n_train'] == 70
  assert meta['n_test'] == 20
  assert meta['n_embargoed'] == 10
  assert meta['n_purged'] == 0
  assert meta['embargo'] == 'P10D'
  assert meta['purge'] == 'P7D'


def test_purge_enforced_by_checked_field():
  '''Gap 8d fails a P10D checked field but not P7D alone.'''
  from forecast_ml_framework.pipelines.modeling.data_split import (
    assert_no_leakage,
  )
  meta = {'max_time_train': '2024-01-01T00:00:00+00:00',
          'min_time_test': '2024-01-09T00:00:00+00:00',
          'embargo': 'P10D', 'purge': 'P10D'}
  with pytest.raises(ValueError, match='gap'):
    assert_no_leakage(meta)


def test_window_rolling_raises():
  '''Only the expanding window is valid.'''
  from forecast_ml_framework.pipelines.modeling.purged_cv import (
    purged_cv_folds,
  )
  df = _daily_frame(100)
  with pytest.raises(ValueError, match='window'):
    purged_cv_folds(df, {'time_col': 'time', 'window': 'rolling'})


def test_missing_time_col_raises():
  '''Missing time_col fails loud with ValueError.'''
  from forecast_ml_framework.pipelines.modeling.purged_cv import (
    purged_cv_folds,
  )
  df = _daily_frame(100)
  with pytest.raises(ValueError, match='time_col'):
    purged_cv_folds(df, {})
  with pytest.raises(ValueError, match='time_col'):
    purged_cv_folds(df, None)
  with pytest.raises(ValueError, match='time'):
    purged_cv_folds(df, {'time_col': 'missing'})


def test_mixed_tz_raises_valueerror():
  '''Mixed naive/aware raises ValueError, never TypeError.'''
  from forecast_ml_framework.pipelines.modeling.purged_cv import (
    purged_cv_folds,
  )
  naive = pd.Timestamp('2024-01-01')
  aware = pd.Timestamp('2024-01-02', tz='UTC')
  df = pd.DataFrame({
    'time': [naive, aware, pd.Timestamp('2024-01-03', tz='UTC')],
    'f1': [1.0, 2.0, 3.0],
    'target': [0, 1, 0],
  })
  with pytest.raises(ValueError, match='mixed naive/aware'):
    purged_cv_folds(df, {'time_col': 'time', 'n_splits': 1,
                         'test_size': 0.34})


def test_single_timestamp_raises():
  '''A frame with one timestamp cannot form folds.'''
  from forecast_ml_framework.pipelines.modeling.purged_cv import (
    purged_cv_folds,
  )
  base = pd.Timestamp('2024-01-01', tz='UTC')
  df = pd.DataFrame({'time': [base] * 10,
                     'f1': list(range(10)),
                     'target': [0, 1] * 5})
  with pytest.raises(ValueError):
    purged_cv_folds(df, {'time_col': 'time'})


def test_n_splits_invalid_raises():
  '''n_splits below one or of bad type raises.'''
  from forecast_ml_framework.pipelines.modeling.purged_cv import (
    purged_cv_folds,
  )
  df = _daily_frame(100)
  for bad in [0, -1, -5, True, 2.5, '2']:
    with pytest.raises(ValueError, match='n_splits'):
      purged_cv_folds(df, {'time_col': 'time', 'n_splits': bad})


def test_single_split_valid_1_fold_golden():
  '''n_splits == 1 reaches the 73/20/7 golden too.'''
  from forecast_ml_framework.pipelines.modeling.purged_cv import (
    purged_cv_folds,
  )
  df = _daily_frame(100)
  folds = purged_cv_folds(df, {'time_col': 'time', 'n_splits': 1,
                               'test_size': 0.2,
                               'embargo': timedelta(days=7)})
  assert len(folds) == 1
  meta = folds[0]['fold_meta']
  assert meta['n_train'] == 73
  assert meta['n_test'] == 20
  assert meta['n_embargoed'] == 7
  assert meta['fold_index'] == 0


def test_boundary_negative_start_raises():
  '''Geometry with a non-positive start raises ValueError.'''
  from forecast_ml_framework.pipelines.modeling.purged_cv import (
    purged_cv_folds,
  )
  df = _daily_frame(4)
  with pytest.raises(ValueError):
    purged_cv_folds(df, {'time_col': 'time', 'n_splits': 3,
                         'test_size': 0.5})


def test_min_train_rows_enforced():
  '''min_train_rows gates the earliest failing fold.'''
  from forecast_ml_framework.pipelines.modeling.purged_cv import (
    purged_cv_folds,
  )
  df = _daily_frame(100)
  with pytest.raises(ValueError, match='fold 0'):
    purged_cv_folds(df, {'time_col': 'time', 'n_splits': 2,
                         'test_size': 0.2,
                         'embargo': timedelta(days=7),
                         'min_train_rows': 74})
  with pytest.raises(ValueError, match='min_train_rows'):
    purged_cv_folds(df, {'time_col': 'time', 'min_train_rows': 0})
  with pytest.raises(ValueError, match='min_train_rows'):
    purged_cv_folds(df, {'time_col': 'time', 'min_train_rows': True})
  with pytest.raises(ValueError, match='min_train_rows'):
    purged_cv_folds(df, {'time_col': 'time', 'min_train_rows': 1.5})
  folds = purged_cv_folds(df, {'time_col': 'time', 'n_splits': 2,
                               'test_size': 0.2,
                               'embargo': timedelta(days=7),
                               'min_train_rows': 1})
  assert len(folds) == 2


def test_determinism_identical_indices():
  '''Two calls return identical index lists per fold.'''
  from forecast_ml_framework.pipelines.modeling.purged_cv import (
    purged_cv_folds,
  )
  df = _daily_frame(100)
  params = {'time_col': 'time', 'n_splits': 2, 'test_size': 0.2,
            'embargo': timedelta(days=7)}
  first = purged_cv_folds(df, params)
  second = purged_cv_folds(df, params)
  for a, b in zip(first, second):
    assert list(a['train_df'].index) == list(b['train_df'].index)
    assert list(a['test_df'].index) == list(b['test_df'].index)


def test_fold_leakage_check_pass_and_fail():
  '''The check returns None clean and names bad folds.'''
  from forecast_ml_framework.pipelines.modeling.purged_cv import (
    fold_leakage_check,
    purged_cv_folds,
  )
  df = _daily_frame(100)
  folds = purged_cv_folds(df, {'time_col': 'time', 'n_splits': 2,
                               'test_size': 0.2,
                               'embargo': timedelta(days=7)})
  assert fold_leakage_check(folds) is None
  bad = [{'fold_index': 7,
          'test_df': folds[0]['test_df'],
          'fold_meta': {
            'max_time_train': '2024-01-10T00:00:00+00:00',
            'min_time_test': '2024-01-10T00:00:00+00:00',
            'embargo': 'P0D'}}]
  with pytest.raises(ValueError, match='7'):
    fold_leakage_check(bad)
  with pytest.raises(ValueError):
    fold_leakage_check([])


def test_fold_leakage_check_malformed_raises():
  '''A fold without meta raises naming its index.'''
  from forecast_ml_framework.pipelines.modeling.purged_cv import (
    fold_leakage_check,
  )
  df = _daily_frame(100)
  bad = [{'fold_index': 3, 'test_df': df, 'fold_meta': None}]
  with pytest.raises(ValueError, match='3'):
    fold_leakage_check(bad)


def test_test_size_none_equal_blocks():
  '''None test_size yields equal trailing blocks.'''
  from forecast_ml_framework.pipelines.modeling.purged_cv import (
    purged_cv_folds,
  )
  df = _daily_frame(100)
  folds = purged_cv_folds(df, {'time_col': 'time', 'n_splits': 3,
                               'test_size': None, 'embargo': 0})
  assert len(folds) == 3
  for fold in folds:
    assert len(fold['test_df']) == 25
    meta = fold['fold_meta']
    assert meta['test_size'] is None
    total = meta['n_train'] + meta['n_embargoed'] \
        + meta['n_purged'] + meta['n_test']
    expect = (fold['fold_index'] + 2) * 25
    assert total == expect


def test_invalid_test_size_raises():
  '''A bad test_size raises ValueError.'''
  from forecast_ml_framework.pipelines.modeling.purged_cv import (
    purged_cv_folds,
  )
  df = _daily_frame(100)
  for bad in [True, '0.2', 0, 1, 1.5, -0.1]:
    with pytest.raises(ValueError, match='test_size'):
      purged_cv_folds(df, {'time_col': 'time', 'test_size': bad})


def test_empty_and_none_df_raises():
  '''Empty or missing frames raise ValueError.'''
  from forecast_ml_framework.pipelines.modeling.purged_cv import (
    purged_cv_folds,
  )
  with pytest.raises(ValueError, match='non-empty'):
    purged_cv_folds(pd.DataFrame({'time': [], 'f1': [],
                                   'target': []}),
                    {'time_col': 'time'})
  with pytest.raises(ValueError, match='non-empty'):
    purged_cv_folds(None, {'time_col': 'time'})


def test_original_indices_preserved():
  '''Fold frames keep the original row index.'''
  from forecast_ml_framework.pipelines.modeling.purged_cv import (
    purged_cv_folds,
  )
  df = _daily_frame(100)
  df.index = list(range(1000, 1100))
  folds = purged_cv_folds(df, {'time_col': 'time', 'n_splits': 2,
                               'test_size': 0.2,
                               'embargo': timedelta(days=7)})
  assert min(folds[0]['train_df'].index) >= 1000
  assert max(folds[1]['test_df'].index) <= 1099
  assert not folds[1]['test_df'].index.equals(
    pd.RangeIndex(20))


def test_non_dict_params_raises():
  '''Non-dict params raise ValueError via missing keys.'''
  from forecast_ml_framework.pipelines.modeling.purged_cv import (
    purged_cv_folds,
  )
  df = _daily_frame(100)
  with pytest.raises(ValueError, match='time_col'):
    purged_cv_folds(df, 'not-a-dict')


def test_grep_guard_single_assignment_owner():
  '''No second sort or embargo parsing lives here.'''
  import forecast_ml_framework.pipelines.modeling.purged_cv as mod
  source = inspect.getsource(mod)
  assert '_temporal_fold_assignment' in source
  assert 'assert_no_leakage' in source
  assert 'sort_values' not in source
  assert '_parse_embargo' not in source
  assert 'train_end' not in source
  assert 'Timedelta' not in source


def test_golden_reassert_temporal_split():
  '''Frozen 73/20/7 trace is re-run beside folds.'''
  from forecast_ml_framework.pipelines.modeling.data_split import (
    temporal_split,
  )
  df = _daily_frame(100)
  params = {'time_col': 'time', 'test_size': 0.2,
            'embargo': timedelta(days=7), 'method': 'temporal',
            'target_col': 'target', 'task_type': 'classification'}
  _, _, meta = temporal_split(df, params)
  assert meta['n_train'] == 73
  assert meta['n_test'] == 20
  assert meta['n_embargoed'] == 7
  assert meta['n_train'] + meta['n_embargoed'] \
      + meta['n_test'] == 100
