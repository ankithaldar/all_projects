#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''P1-B1 temporal split tests (Contract 4).'''

import datetime
import math

import numpy as np
import pandas as pd
import pytest


def _daily_frame(n=100, start='2020-01-01'):
  '''Return daily frame with f1 and target.'''
  times = pd.date_range(start=start, periods=n, freq='D', tz='UTC')
  return pd.DataFrame({
    'time': times,
    'f1': np.arange(n, dtype=float),
    'target': [0, 1] * (n // 2) if n % 2 == 0 else [0, 1] * (n // 2) + [0],
  })


def test_temporal_split_post_embargo_fraction():
  '''100-row daily: 73/20/7 post-embargo, control 80/20.'''
  from datetime import timedelta
  from forecast_ml_framework.pipelines.modeling.data_split import (
    temporal_split,
  )
  df = _daily_frame(100)
  params = {'time_col': 'time', 'test_size': 0.2,
            'embargo': timedelta(days=7), 'method': 'temporal',
            'target_col': 'target', 'task_type': 'classification'}
  train, test, meta = temporal_split(df, params)
  assert len(test) == 20
  assert len(train) == 73
  assert meta['n_embargoed'] == 7
  assert meta['n_train'] == 73
  assert meta['n_test'] == 20
  assert len(test) / len(df) == pytest.approx(0.2, abs=1 / 100)
  assert len(train) + meta['n_embargoed'] + len(test) == 100
  assert meta['sampling_method'] == 'temporal'
  assert meta['requested_method'] == 'temporal'
  assert meta['shuffle'] is False
  assert meta['random_state'] is None
  assert meta['leakage_checked'] is True
  assert meta['time_col'] == 'time'
  assert meta['test_size'] == 0.2
  # control embargo 0
  params0 = dict(params, embargo=0)
  train0, test0, meta0 = temporal_split(df, params0)
  assert len(test0) == 20
  assert len(train0) == 80
  assert meta0['n_embargoed'] == 0


def test_temporal_split_strict_invariant():
  '''max_train < min_test - embargo strict.'''
  from datetime import timedelta
  from forecast_ml_framework.pipelines.modeling.data_split import (
    temporal_split,
  )
  df = _daily_frame(100)
  train, test, meta = temporal_split(
    df, {'time_col': 'time', 'test_size': 0.2,
         'embargo': timedelta(days=7), 'method': 'temporal'})
  max_train = pd.to_datetime(meta['max_time_train'], utc=True)
  min_test = pd.to_datetime(meta['min_time_test'], utc=True)
  assert max_train < min_test - timedelta(days=7)
  assert meta['max_time_train'] is not None
  assert meta['min_time_test'] is not None


def test_assert_no_leakage_overlapping_raises_naming_gap():
  '''Overlapping meta raises ValueError naming gap.'''
  from forecast_ml_framework.pipelines.modeling.data_split import (
    assert_no_leakage,
  )
  from datetime import timedelta
  meta = {
    'max_time_train': '2020-01-10T00:00:00+00:00',
    'min_time_test': '2020-01-10T00:00:00+00:00',
    'embargo': timedelta(0),
  }
  with pytest.raises(ValueError, match='gap'):
    assert_no_leakage(meta)
  # equal boundary with embargo 0 is leakage (strict)
  meta2 = {
    'max_time_train': '2020-01-11T00:00:00+00:00',
    'min_time_test': '2020-01-10T00:00:00+00:00',
    'embargo': 'P0D',
  }
  with pytest.raises(ValueError, match='gap'):
    assert_no_leakage(meta2)


def test_assert_no_leakage_gap_equal_embargo_raises():
  '''Gap == embargo fails strict > embargo.'''
  from forecast_ml_framework.pipelines.modeling.data_split import (
    assert_no_leakage,
  )
  from datetime import timedelta
  # gap exactly 7 days, embargo 7 days -> max == min - embargo, not <
  meta = {
    'max_time_train': '2020-01-01T00:00:00+00:00',
    'min_time_test': '2020-01-08T00:00:00+00:00',
    'embargo': timedelta(days=7),
  }
  with pytest.raises(ValueError, match='gap'):
    assert_no_leakage(meta)
  # gap 8 days passes
  meta_ok = {
    'max_time_train': '2020-01-01T00:00:00+00:00',
    'min_time_test': '2020-01-09T00:00:00+00:00',
    'embargo': timedelta(days=7),
  }
  assert_no_leakage(meta_ok)


def test_mixed_tz_raises_valueerror():
  '''Mixed naive/aware raises ValueError naming tz.'''
  from forecast_ml_framework.pipelines.modeling.data_split import (
    _check_no_mixed_tz,
    temporal_split,
  )
  naive = pd.Timestamp('2020-01-01')
  aware = pd.Timestamp('2020-01-02', tz='UTC')
  series = pd.Series([naive, aware, naive])
  with pytest.raises(ValueError, match='mixed naive/aware'):
    _check_no_mixed_tz(series)
  df = pd.DataFrame({
    'time': [naive, aware, pd.Timestamp('2020-01-03', tz='UTC')],
    'f1': [1.0, 2.0, 3.0],
    'target': [0, 1, 0],
  })
  with pytest.raises(ValueError, match='mixed naive/aware'):
    temporal_split(
      df, {'time_col': 'time', 'test_size': 0.34, 'method': 'temporal'})


def test_check_no_mixed_tz_uniform_passes():
  '''Uniform naive and uniform aware pass.'''
  from forecast_ml_framework.pipelines.modeling.data_split import (
    _check_no_mixed_tz,
  )
  naive_series = pd.Series(
    [pd.Timestamp('2020-01-01'), pd.Timestamp('2020-01-02')])
  _check_no_mixed_tz(naive_series)
  aware_series = pd.Series(
    [pd.Timestamp('2020-01-01', tz='UTC'),
     pd.Timestamp('2020-01-02', tz='US/Eastern')])
  _check_no_mixed_tz(aware_series)
  # numeric uniform passes
  _check_no_mixed_tz(pd.Series([1, 2, 3]))


def test_temporal_split_missing_time_col_raises():
  '''Temporal without time_col raises ValueError.'''
  from forecast_ml_framework.pipelines.modeling.data_split import (
    temporal_split,
  )
  df = pd.DataFrame({'f1': [1.0, 2.0, 3.0, 4.0], 'target': [0, 1, 0, 1]})
  with pytest.raises(ValueError, match='time_col'):
    temporal_split(df, {'method': 'temporal', 'test_size': 0.5})
  # time_col named but missing column raises
  with pytest.raises(ValueError, match='time'):
    temporal_split(
      df, {'time_col': 'missing', 'method': 'temporal', 'test_size': 0.5})


def test_temporal_split_duplicate_timestamp_advances():
  '''Duplicate timestamps at horizon move train_end earlier.'''
  from forecast_ml_framework.pipelines.modeling.data_split import (
    temporal_split,
  )
  # Times: 0,1,2,2,2,3,4,5,6,7 daily-like as ints? Use datetime duplicates
  base = pd.Timestamp('2020-01-01', tz='UTC')
  times = [base + pd.Timedelta(days=d)
           for d in [0, 1, 2, 2, 2, 3, 4, 5, 6, 7]]
  df = pd.DataFrame({'time': times, 'f1': list(range(10)),
                     'target': [0, 1] * 5})
  train, test, meta = temporal_split(
    df, {'time_col': 'time', 'test_size': 0.2, 'embargo': 0,
         'method': 'temporal'})
  # test is last 2: day 6,7 ; train max must be < min test
  assert len(test) == 2
  assert meta['n_embargoed'] == meta['n_embargoed']  # sanity
  max_train = pd.to_datetime(meta['max_time_train'], utc=True)
  min_test = pd.to_datetime(meta['min_time_test'], utc=True)
  assert max_train < min_test
  # duplicate horizon case: embargo 0 with duplicates at cut
  times2 = [base + pd.Timedelta(days=d)
            for d in [0, 1, 2, 3, 3, 3, 3, 4, 5, 6]]
  df2 = pd.DataFrame({'time': times2, 'f1': list(range(10)),
                      'target': [0, 1] * 5})
  train2, test2, meta2 = temporal_split(
    df2, {'time_col': 'time', 'test_size': 0.5, 'embargo': 0,
          'method': 'temporal'})
  # test_start =5 (time at idx5 = day3), train must be last < day3
  # idx 0..3 are day0..3, but day3 at idx3 is not < day3, so train ends at idx2
  assert len(test2) == 5
  assert len(train2) == 5 - meta2['n_embargoed']
  assert len(train2) + meta2['n_embargoed'] + len(test2) == 10


def test_temporal_split_all_same_timestamp_raises():
  '''All same timestamp raises ValueError.'''
  from forecast_ml_framework.pipelines.modeling.data_split import (
    temporal_split,
  )
  base = pd.Timestamp('2020-01-01', tz='UTC')
  df = pd.DataFrame({
    'time': [base] * 10, 'f1': list(range(10)), 'target': [0, 1] * 5})
  with pytest.raises(ValueError):
    temporal_split(
      df, {'time_col': 'time', 'test_size': 0.2, 'method': 'temporal'})


def test_temporal_split_embargo_consumes_raises():
  '''Embargo larger than data raises ValueError.'''
  from forecast_ml_framework.pipelines.modeling.data_split import (
    temporal_split,
  )
  from datetime import timedelta
  df = _daily_frame(20)
  with pytest.raises(ValueError, match='embargo consumes'):
    temporal_split(
      df, {'time_col': 'time', 'test_size': 0.2,
           'embargo': timedelta(days=30), 'method': 'temporal'})


def test_temporal_split_numeric_time_col():
  '''Numeric time_col works with int embargo.'''
  from forecast_ml_framework.pipelines.modeling.data_split import (
    temporal_split,
  )
  df = pd.DataFrame({'time': list(range(100)), 'f1': list(range(100)),
                     'target': [0, 1] * 50})
  train, test, meta = temporal_split(
    df, {'time_col': 'time', 'test_size': 0.2, 'embargo': 7,
         'method': 'temporal'})
  assert len(test) == 20
  assert len(train) == 73
  assert meta['n_embargoed'] == 7
  # shuffled input still sorts by time
  shuffled = df.sample(frac=1, random_state=0).reset_index(drop=True)
  train_s, test_s, _ = temporal_split(
    shuffled, {'time_col': 'time', 'test_size': 0.2, 'embargo': 7,
               'method': 'temporal'})
  assert len(test_s) == 20
  assert train_s['time'].max() < test_s['time'].min() - 7


def test_temporal_split_int_embargo_days():
  '''Int embargo for datetime means days.'''
  from forecast_ml_framework.pipelines.modeling.data_split import (
    temporal_split,
  )
  df = _daily_frame(100)
  train, test, meta = temporal_split(
    df, {'time_col': 'time', 'test_size': 0.2, 'embargo': 7,
         'method': 'temporal'})
  assert len(train) == 73
  assert len(test) == 20
  assert meta['n_embargoed'] == 7


def test_presplit_temporal_split_leaky_raises():
  '''Presplit temporal_split raises on leaky caller cut.'''
  from forecast_ml_framework.pipelines.modeling.data_split import (
    temporal_split,
  )
  base = pd.Timestamp('2020-01-01', tz='UTC')
  train = pd.DataFrame({
    'time': [base + pd.Timedelta(days=d) for d in [0, 5, 10]],
    'f1': [1.0, 2.0, 3.0], 'target': [0, 1, 0]})
  test = pd.DataFrame({
    'time': [base + pd.Timedelta(days=d) for d in [3, 7, 12]],
    'f1': [4.0, 5.0, 6.0], 'target': [1, 0, 1]})
  # frozen signature: temporal_split(train_df, params, test_df)
  with pytest.raises(ValueError, match='gap'):
    temporal_split(
      train, {'method': 'presplit', 'time_col': 'time', 'test_size': 0.5,
              'embargo': 0}, test)
  # deleted params keys are not a frozen path: no test frame raises
  with pytest.raises(ValueError, match='presplit requires'):
    temporal_split(
      train, {'method': 'presplit', 'time_col': 'time', 'test_size': 0.5,
              'test_df': test, 'embargo': 0})
  with pytest.raises(ValueError, match='presplit requires'):
    temporal_split(
      train, {'method': 'presplit', 'time_col': 'time',
              'train_df': train, 'test_df': test})


def test_presplit_temporal_split_clean_annotates():
  '''Clean presplit annotates without moving rows.'''
  from forecast_ml_framework.pipelines.modeling.data_split import (
    temporal_split,
  )
  base = pd.Timestamp('2020-01-01', tz='UTC')
  train = pd.DataFrame({
    'time': [base + pd.Timedelta(days=d) for d in [0, 1, 2]],
    'f1': [1.0, 2.0, 3.0], 'target': [0, 1, 0]})
  test = pd.DataFrame({
    'time': [base + pd.Timedelta(days=d) for d in [5, 6, 7]],
    'f1': [4.0, 5.0, 6.0], 'target': [1, 0, 1]})
  train_out, test_out, meta = temporal_split(
    train, {'method': 'presplit', 'time_col': 'time'}, test)
  assert train_out.equals(train)
  assert test_out.equals(test)
  assert meta['sampling_method'] == 'presplit'
  assert meta['requested_method'] == 'presplit'
  assert meta['n_embargoed'] == 0
  assert meta['leakage_checked'] is True


def test_split_presplit_data_leaky_raises():
  '''split_presplit_data leaky fixture raises ValueError.'''
  from forecast_ml_framework.pipelines.modeling.pipeline import (
    split_presplit_data,
  )
  base = pd.Timestamp('2020-01-01', tz='UTC')
  train = pd.DataFrame({
    'time': [base + pd.Timedelta(days=d) for d in [0, 5, 10]],
    'f1': [1.0, 2.0, 3.0], 'target': [0, 1, 0]})
  test = pd.DataFrame({
    'time': [base + pd.Timedelta(days=d) for d in [3, 7, 12]],
    'f1': [4.0, 5.0, 6.0], 'target': [1, 0, 1]})
  with pytest.raises(ValueError, match='gap'):
    split_presplit_data(
      train, test, {'target_col': 'target', 'time_col': 'time'})
  # leaky via raw_meta
  raw_meta = {
    'sampling_method': 'temporal', 'requested_method': 'temporal',
    'shuffle': False, 'time_col': 'time',
    'max_time_train': '2020-01-10T00:00:00+00:00',
    'min_time_test': '2020-01-10T00:00:00+00:00',
    'embargo': 'P0D',
  }
  train2 = pd.DataFrame({'f1': [1.0], 'target': [0]})
  test2 = pd.DataFrame({'f1': [2.0], 'target': [1]})
  with pytest.raises(ValueError, match='gap'):
    split_presplit_data(
      train2, test2, {'target_col': 'target', 'time_col': 'time'},
      raw_meta)


def test_split_presplit_data_clean_annnotates():
  '''Clean split_presplit_data annotates leakage_checked.'''
  from forecast_ml_framework.pipelines.modeling.pipeline import (
    split_presplit_data,
  )
  base = pd.Timestamp('2020-01-01', tz='UTC')
  train = pd.DataFrame({
    'time': [base + pd.Timedelta(days=d) for d in [0, 1, 2]],
    'f1': [1.0, 2.0, 3.0], 'target': [0, 1, 0]})
  test = pd.DataFrame({
    'time': [base + pd.Timedelta(days=d) for d in [5, 6, 7]],
    'f1': [4.0, 5.0, 6.0], 'target': [1, 0, 1]})
  x_tr, x_te, y_tr, y_te, meta = split_presplit_data(
    train, test, {'target_col': 'target', 'time_col': 'time'})
  assert meta['sampling_method'] == 'presplit'
  assert meta['n_embargoed'] == 0
  assert meta['leakage_checked'] is True
  assert list(x_tr['f1']) == [1.0, 2.0, 3.0]
  # no time info -> no raise, leakage_checked reflects absence
  x2, x2te, y2, y2te, meta2 = split_presplit_data(
    pd.DataFrame({'f1': [1.0, 2.0], 'target': [0, 1]}),
    pd.DataFrame({'f1': [3.0], 'target': [1]}),
    {'target_col': 'target'})
  assert meta2['sampling_method'] == 'presplit'
  assert meta2['n_train'] == 2


def test_split_data_delegates_temporal():
  '''split_data with method temporal delegates to temporal_split.'''
  from forecast_ml_framework.pipelines.modeling.data_split import (
    split_data,
  )
  from datetime import timedelta
  df = _daily_frame(100)
  params = {
    'target_col': 'target', 'feature_cols': ['f1', 'time'],
    'time_col': 'time', 'test_size': 0.2, 'random_state': 42,
    'method': 'temporal', 'embargo': timedelta(days=7),
    'task_type': 'classification'}
  x_tr, x_te, y_tr, y_te, meta = split_data(df, params)
  assert len(x_te) == 20
  assert len(x_tr) == 73
  assert meta['n_embargoed'] == 7
  assert meta['n_test'] == 20
  assert meta['n_train'] == 73
  assert meta['leakage_checked'] is True
  assert meta['shuffle'] is False
  # via sampling_params method temporal
  params2 = {
    'target_col': 'target', 'feature_cols': ['f1'],
    'task_type': 'classification', 'random_state': 42,
    'sampling_params': {'method': 'temporal', 'time_col': 'time',
                        'test_size': 0.2, 'embargo': timedelta(days=7)}}
  x_tr2, x_te2, _, _, meta2 = split_data(df, params2)
  assert len(x_te2) == 20


def test_split_data_time_delegation_with_time_col():
  '''split_data method time with time_col delegates temporal.'''
  from forecast_ml_framework.pipelines.modeling.data_split import (
    split_data,
  )
  df = _daily_frame(100)
  params = {
    'target_col': 'target', 'feature_cols': ['f1'],
    'task_type': 'classification', 'random_state': 42,
    'sampling_params': {'method': 'time', 'time_col': 'time',
                        'test_size': 0.2}}
  x_tr, x_te, _, _, meta = split_data(df, params)
  # back-compat sampling_method stays time
  assert meta['sampling_method'] == 'time'
  assert meta['n_test'] == 20
  assert meta['leakage_checked'] is True
  assert meta['n_embargoed'] == 0


def test_split_data_time_no_time_col_legacy_ordered():
  '''Legacy time without time_col keeps ordered split.'''
  from forecast_ml_framework.pipelines.modeling.data_split import (
    split_data,
  )
  df = pd.DataFrame({
    'id': [0, 1, 2, 3, 4, 5],
    'f1': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    'target': [0, 1, 0, 1, 0, 1]})
  params = {
    'target_col': 'target', 'feature_cols': ['f1'],
    'task_type': 'classification', 'random_state': 42,
    'sampling_params': {'method': 'time', 'test_size': 0.5}}
  x_tr, x_te, _, _, meta = split_data(df, params)
  assert meta['sampling_method'] == 'time'
  assert len(x_tr) + len(x_te) == 6
  # ordered: train ids precede test (f1 order preserved)
  assert list(x_tr['f1']) == [1.0, 2.0, 3.0]
  assert list(x_te['f1']) == [4.0, 5.0, 6.0]
  assert meta['n_embargoed'] == 0
  assert meta['embargo'] is None


def test_split_data_stratified_still_works_with_new_keys():
  '''Stratified/random meta keeps new keys defaulted.'''
  from forecast_ml_framework.pipelines.modeling.data_split import (
    split_data,
  )
  rng = np.random.RandomState(0)
  df = pd.DataFrame({'f1': rng.randn(20), 'f2': rng.randn(20),
                     'target': rng.randint(0, 2, 20)})
  params = {'target_col': 'target', 'feature_cols': ['f1', 'f2'],
            'task_type': 'classification', 'test_size': 0.25,
            'random_state': 7,
            'sampling_params': {'method': 'random', 'test_size': 0.25}}
  _, _, _, _, meta = split_data(df, params)
  assert meta['n_embargoed'] == 0
  assert meta['embargo'] is None


class _StubModel:
  '''Stub model for unwrap tests.'''

  def predict(self, x):
    '''Return fixed labels.'''
    return np.asarray([1, 0])

  def predict_proba(self, x):
    '''Return fixed proba.'''
    return np.asarray([[0.2, 0.8], [0.9, 0.1]])


def test_predict_unwraps_envelope():
  '''_predict unwraps dict envelope once.'''
  from forecast_ml_framework.pipelines.modeling.pipeline import _predict
  bare = _StubModel()
  envelope = {'model': bare, 'label_set': ['0', '1'],
              'registry_ref': None, 'onnx_sidecar': None}
  assert list(_predict(bare, None)) == [1, 0]
  assert list(_predict(envelope, None)) == [1, 0]


def test_predict_proba_unwraps():
  '''_predict_proba unwraps and keeps shapes.'''
  from forecast_ml_framework.pipelines.modeling.pipeline import (
    _predict_proba,
  )
  bare = _StubModel()
  envelope = {'model': bare, 'label_set': ['0', '1'],
              'registry_ref': None, 'onnx_sidecar': None}
  assert _predict_proba(bare, None, 'classification').shape == (2, 2)
  assert _predict_proba(envelope, None, 'classification').shape == (2, 2)
  assert _predict_proba(envelope, None, 'regression') is None


def test_explain_model_unwraps(monkeypatch):
  '''explain_model unwraps before shap_explain.'''
  import forecast_ml_framework.pipelines.modeling.pipeline as pipe_mod
  seen = []

  def fake_explain(model, x, params):
    '''Capture model and return canned output.'''
    seen.append(model)
    return {'method': 'shap', 'global_importance': [
      {'feature': 'f1', 'importance': 1.0}], 'per_row': []}

  monkeypatch.setattr(pipe_mod, 'shap_explain', fake_explain)
  bare = _StubModel()
  envelope = {'model': bare, 'label_set': ['0', '1'],
              'registry_ref': None, 'onnx_sidecar': None}
  out = pipe_mod.explain_model(envelope, None, None, None)
  assert seen and seen[0] is bare
  assert out['shap_available'] is True
  seen.clear()
  out2 = pipe_mod.explain_model(bare, None, None, None)
  assert seen and seen[0] is bare


def test_make_charts_no_envelope_input():
  '''make_charts node never receives envelope.'''
  from forecast_ml_framework.pipelines.modeling.pipeline import (
    create_pipeline,
  )
  pipe = create_pipeline()
  by_name = {n.name: n for n in pipe.nodes}
  charts = by_name['make_charts_node']
  assert 'calibrated_model' not in charts.inputs
  assert 'best_model' not in charts.inputs
  assert charts.inputs == [
    'y_test', 'y_pred', 'y_pred_proba', 'x_test',
    'params:modeling_params']


def test_temporal_split_stable_sort_input_shuffled():
  '''Shuffled daily still splits by time order.'''
  from forecast_ml_framework.pipelines.modeling.data_split import (
    temporal_split,
  )
  df = _daily_frame(100)
  shuffled = df.sample(frac=1, random_state=1).reset_index(drop=True)
  train, test, meta = temporal_split(
    shuffled, {'time_col': 'time', 'test_size': 0.2, 'embargo': 0,
               'method': 'temporal'})
  assert len(test) == 20
  assert train['time'].max() < test['time'].min()


def test_assert_no_leakage_string_embargo_P7D():
  '''String embargo P7D parsed.'''
  from forecast_ml_framework.pipelines.modeling.data_split import (
    assert_no_leakage,
  )
  meta = {
    'max_time_train': '2020-01-01T00:00:00+00:00',
    'min_time_test': '2020-01-09T00:00:00+00:00',
    'embargo': 'P7D',
  }
  assert_no_leakage(meta)
  meta_bad = {
    'max_time_train': '2020-01-01T00:00:00+00:00',
    'min_time_test': '2020-01-07T00:00:00+00:00',
    'embargo': 'P7D',
  }
  with pytest.raises(ValueError, match='gap'):
    assert_no_leakage(meta_bad)


def test_assert_no_leakage_missing_times_noop():
  '''Missing times do not raise (non-temporal presplit).'''
  from forecast_ml_framework.pipelines.modeling.data_split import (
    assert_no_leakage,
  )
  assert_no_leakage({})
  assert_no_leakage({'max_time_train': None, 'min_time_test': None,
                     'embargo': 0})
  assert_no_leakage({'max_time_train': '2020-01-01T00:00:00+00:00',
                     'min_time_test': None, 'embargo': 0})


def test_temporal_split_small_n_ceil():
  '''Ceil handling small N.'''
  from forecast_ml_framework.pipelines.modeling.data_split import (
    temporal_split,
  )
  base = pd.Timestamp('2020-01-01', tz='UTC')
  times = [base + pd.Timedelta(days=d) for d in range(5)]
  df = pd.DataFrame({'time': times, 'f1': list(range(5)),
                     'target': [0, 1, 0, 1, 0]})
  train, test, meta = temporal_split(
    df, {'time_col': 'time', 'test_size': 0.5, 'embargo': 0,
         'method': 'temporal'})
  assert len(test) == 3  # ceil(0.5*5)=3 frozen
  assert len(train) == 2
  assert meta['n_test'] == 3


def test_split_data_temporal_via_time_col_no_method():
  '''split_data with time_col + shuffle False delegates temporal.'''
  from forecast_ml_framework.pipelines.modeling.data_split import (
    split_data,
  )
  df = _daily_frame(20)
  params = {'target_col': 'target', 'feature_cols': ['f1'],
            'task_type': 'classification', 'test_size': 0.25,
            'random_state': 42, 'time_col': 'time', 'shuffle': False}
  x_tr, x_te, _, _, meta = split_data(df, params)
  assert meta['sampling_method'] == 'time'
  assert meta['leakage_checked'] is True


def test_split_data_none_and_empty_raises():
  '''split_data raises for None/empty/missing target.'''
  from forecast_ml_framework.pipelines.modeling.data_split import (
    split_data,
  )
  with pytest.raises(ValueError, match='non-empty'):
    split_data(None, {'target_col': 'target'})
  with pytest.raises(ValueError, match='non-empty'):
    split_data(pd.DataFrame({'f1': [], 'target': []}),
               {'target_col': 'target'})
  df = pd.DataFrame({'f1': [1.0, 2.0], 'other': [0, 1]})
  with pytest.raises(ValueError, match='target'):
    split_data(df, {'target_col': 'target'})
  df2 = pd.DataFrame({'f1': [1.0, 2.0, 3.0], 'target': [0, 1, 0]})
  with pytest.raises(ValueError, match='test_size'):
    split_data(df2, {'target_col': 'target', 'test_size': True})
  with pytest.raises(ValueError, match='test_size'):
    split_data(df2, {'target_col': 'target', 'test_size': 1.5})
  with pytest.raises(ValueError, match='test_size'):
    split_data(df2, {'target_col': 'target', 'test_size': float('nan')})
  with pytest.raises(ValueError, match='test_size'):
    split_data(df2, {'target_col': 'target', 'test_size': '0.2'})
  with pytest.raises(ValueError, match='random_state'):
    split_data(df2, {'target_col': 'target', 'random_state': True})
  with pytest.raises(ValueError, match='random_state'):
    split_data(df2, {'target_col': 'target', 'random_state': -1})
  with pytest.raises(ValueError, match='random_state'):
    split_data(df2, {'target_col': 'target', 'random_state': 1.5})


def test_split_data_missing_time_col_via_split():
  '''split_data time with missing time_col raises.'''
  from forecast_ml_framework.pipelines.modeling.data_split import (
    split_data,
  )
  df = pd.DataFrame({'f1': [1.0, 2.0, 3.0, 4.0], 'target': [0, 1, 0, 1]})
  params = {'target_col': 'target', 'feature_cols': ['f1'],
            'sampling_params': {'method': 'time', 'time_col': 'missing',
                                'test_size': 0.5}}
  with pytest.raises(ValueError, match='time column'):
    split_data(df, params)
  # temporal via split_data without time_col raises
  params2 = {'target_col': 'target', 'feature_cols': ['f1'],
             'method': 'temporal', 'test_size': 0.5}
  with pytest.raises(ValueError, match='time_col'):
    split_data(df, params2)


def test_split_data_id_cols_variants():
  '''Default features exclude id via all id params.'''
  from forecast_ml_framework.pipelines.modeling.data_split import (
    split_data,
  )
  # default ['id']
  frame = pd.DataFrame({'id': range(10), 'x': list(range(10)),
                        'target': [0, 1] * 5})
  params = {'target_col': 'target', 'task_type': 'classification',
            'test_size': 0.2, 'random_state': 1,
            'sampling_params': {'method': 'random'}}
  x_tr, _, _, _, _ = split_data(frame, params)
  assert 'id' not in x_tr.columns
  # id_cols param
  frame2 = pd.DataFrame({'uid': range(10), 'x': list(range(10)),
                         'target': [0, 1] * 5})
  x_tr2, _, _, _, _ = split_data(
    frame2, dict(params, id_cols=['uid']))
  assert 'uid' not in x_tr2.columns
  # id_col string
  x_tr3, _, _, _, _ = split_data(
    frame2, dict(params, id_col='uid'))
  assert 'uid' not in x_tr3.columns
  # variables dict
  x_tr4, _, _, _, _ = split_data(
    frame2, dict(params, variables={'id_cols': ['uid']}))
  assert 'uid' not in x_tr4.columns
  x_tr5, _, _, _, _ = split_data(
    frame2, dict(params, variables={'id_col': 'uid'}))
  assert 'uid' not in x_tr5.columns
  x_tr6, _, _, _, _ = split_data(
    frame2, dict(params, variables={'id': ['uid']}))
  assert 'uid' not in x_tr6.columns
  # sampling id_cols
  x_tr7, _, _, _, _ = split_data(
    frame2, {'target_col': 'target', 'task_type': 'classification',
             'sampling_params': {'method': 'random', 'id_cols': ['uid'],
                                 'test_size': 0.2}})
  assert 'uid' not in x_tr7.columns
  # sampling id_col string
  x_tr8, _, _, _, _ = split_data(
    frame2, {'target_col': 'target', 'task_type': 'classification',
             'sampling_params': {'method': 'random', 'id_col': 'uid',
                                 'test_size': 0.2}})
  assert 'uid' not in x_tr8.columns
  # multiple ids unioned, string counts as single column
  frame3 = pd.DataFrame({'a': range(10), 'b': range(10),
                         'x': list(range(10)), 'target': [0, 1] * 5})
  x_tr9, _, _, _, _ = split_data(
    frame3, dict(params, id_cols=['a', 'b']))
  assert 'a' not in x_tr9.columns and 'b' not in x_tr9.columns
  # empty feature_cols yields zero-column frame
  rng = np.random.RandomState(0)
  df = pd.DataFrame({'f1': rng.randn(8), 'target': rng.randint(0, 2, 8)})
  x_tr10, x_te10, _, _, _ = split_data(
    df, {'target_col': 'target', 'feature_cols': [],
         'random_state': 1, 'test_size': 0.5,
         'sampling_params': {'method': 'random'}})
  assert x_tr10.shape[1] == 0


def test_split_data_random_no_shuffle():
  '''shuffle False with random preserves order.'''
  from forecast_ml_framework.pipelines.modeling.data_split import (
    split_data,
  )
  df = pd.DataFrame({'f1': [float(x) for x in range(10)], 'target': [0, 1] * 5})
  params = {'target_col': 'target', 'feature_cols': ['f1'],
            'task_type': 'classification', 'random_state': 42,
            'sampling_params': {'method': 'random', 'shuffle': False,
                                'test_size': 0.2}}
  x_tr, x_te, _, _, meta = split_data(df, params)
  assert meta['sampling_method'] == 'random'
  assert list(x_tr['f1']) == [float(x) for x in range(8)]
  assert list(x_te['f1']) == [8.0, 9.0]
  # shuffle False without method also ordered random path
  params2 = {'target_col': 'target', 'feature_cols': ['f1'],
             'sampling_params': {'shuffle': False, 'test_size': 0.2}}
  x_tr2, _, _, _, meta2 = split_data(
    pd.DataFrame({'f1': [float(x) for x in range(10)],
                  'target': [0, 1] * 5}), params2)
  assert meta2['sampling_method'] == 'random'


def test_split_data_ordered_capped():
  '''Ordered split caps n_test to n-1.'''
  from forecast_ml_framework.pipelines.modeling.data_split import (
    split_data,
  )
  df = pd.DataFrame({'f1': [1.0, 2.0], 'target': [0, 1]})
  params = {'target_col': 'target', 'feature_cols': ['f1'],
            'sampling_params': {'method': 'time', 'test_size': 0.9}}
  x_tr, x_te, _, _, meta = split_data(df, params)
  assert len(x_tr) == 1 and len(x_te) == 1
  assert meta['sampling_method'] == 'time'


def test_split_data_stratified_fallback_warns():
  '''Impossible stratify warns and meta is random.'''
  from forecast_ml_framework.pipelines.modeling.data_split import (
    split_data,
  )
  frame = pd.DataFrame({'f1': np.arange(10.0), 'target': [0] * 9 + [1]})
  params = {'target_col': 'target', 'task_type': 'classification',
            'sampling_params': {'method': 'stratified', 'test_size': 0.2}}
  with pytest.warns(UserWarning, match='stratified'):
    _, _, _, _, meta = split_data(frame, params)
  assert meta['sampling_method'] == 'random'


def test_split_data_stratified_success():
  '''Stratified succeeds with balanced classes.'''
  from forecast_ml_framework.pipelines.modeling.data_split import (
    split_data,
  )
  frame = pd.DataFrame({'f1': np.arange(20.0),
                        'target': [0, 1] * 10})
  params = {'target_col': 'target', 'task_type': 'classification',
            'random_state': 1,
            'sampling_params': {'method': 'stratified', 'test_size': 0.2}}
  _, _, _, _, meta = split_data(frame, params)
  assert meta['sampling_method'] == 'stratified'


def test_split_data_regression_defaults_random():
  '''Regression without stratify is random.'''
  from forecast_ml_framework.pipelines.modeling.data_split import (
    split_data,
  )
  rng = np.random.RandomState(0)
  df = pd.DataFrame({'f1': rng.randn(12), 'target': rng.randn(12)})
  params = {'target_col': 'target', 'feature_cols': ['f1'],
            'task_type': 'regression', 'random_state': 1, 'test_size': 0.25}
  _, _, _, _, meta = split_data(df, params)
  assert meta['sampling_method'] == 'random'


def test_split_data_regression_stratify_dropped():
  '''Explicit regression stratify on target is dropped.'''
  from forecast_ml_framework.pipelines.modeling.data_split import (
    split_data,
  )
  rng = np.random.RandomState(0)
  df = pd.DataFrame({'f1': rng.randn(12), 'target': rng.randn(12)})
  params = {'target_col': 'target', 'feature_cols': ['f1'],
            'task_type': 'regression', 'random_state': 1,
            'sampling_params': {'method': 'stratified',
                                'stratify_col': 'target',
                                'test_size': 0.25}}
  _, _, _, _, meta = split_data(df, params)
  assert meta['sampling_method'] == 'random'


def test_split_data_single_class_random():
  '''Single-class stratify nulled to random.'''
  from forecast_ml_framework.pipelines.modeling.data_split import (
    split_data,
  )
  frame = pd.DataFrame({'f1': np.arange(12.0), 'target': [0] * 12})
  params = {'target_col': 'target', 'feature_cols': ['f1'],
            'task_type': 'classification', 'random_state': 1,
            'test_size': 0.25}
  _, _, _, _, meta = split_data(frame, params)
  assert meta['sampling_method'] == 'random'


def test_split_data_missing_stratify_col_keyerror():
  '''Missing stratify column raises KeyError.'''
  from forecast_ml_framework.pipelines.modeling.data_split import (
    split_data,
  )
  rng = np.random.RandomState(0)
  df = pd.DataFrame({'f1': rng.randn(12), 'target': rng.randint(0, 2, 12)})
  params = {'target_col': 'target', 'feature_cols': ['f1'],
            'sampling_params': {'method': 'stratified',
                                'stratify_col': 'missing'}}
  with pytest.raises(KeyError, match='missing'):
    split_data(df, params)


def test_check_no_mixed_tz_none_and_nan():
  '''None/NaT/nan skipped in tz check.'''
  from forecast_ml_framework.pipelines.modeling.data_split import (
    _check_no_mixed_tz,
  )
  _check_no_mixed_tz(None)
  _check_no_mixed_tz(pd.Series([], dtype=object))
  series = pd.Series([pd.NaT, None, np.nan,
                      pd.Timestamp('2020-01-01')])
  _check_no_mixed_tz(series)
  # string dates uniform
  _check_no_mixed_tz(pd.Series(['2020-01-01', '2020-01-02']))
  _check_no_mixed_tz(
    pd.Series(['2020-01-01T00:00:00+00:00',
               '2020-01-02T00:00:00+00:00']))
  # object with tzinfo fallback (datetime with tz)
  import datetime as _dt
  series2 = pd.Series([_dt.datetime(2020, 1, 1, tzinfo=_dt.timezone.utc),
                       _dt.datetime(2020, 1, 2, tzinfo=_dt.timezone.utc)])
  _check_no_mixed_tz(series2)
  # np.datetime64 naive uniform
  _check_no_mixed_tz(
    pd.Series([np.datetime64('2020-01-01'), np.datetime64('2020-01-02')]))
  # object dtype numeric (non-numeric dtype branch)
  _check_no_mixed_tz(pd.Series([1, 2, 3], dtype=object))


def test_parse_embargo_variants():
  '''Embargo parsing covers all input shapes.'''
  from datetime import timedelta
  from forecast_ml_framework.pipelines.modeling.data_split import (
    _parse_embargo,
    _embargo_str,
    assert_no_leakage,
  )
  assert _parse_embargo(None, is_datetime=True) == pd.Timedelta(0)
  assert _parse_embargo(None, is_datetime=False) == 0.0
  assert _parse_embargo(pd.Timedelta(days=2), is_datetime=True) == pd.Timedelta(
    days=2)
  assert _parse_embargo(pd.Timedelta(days=2), is_datetime=False) == 172800.0
  assert _parse_embargo(timedelta(days=3), is_datetime=True) == pd.Timedelta(
    days=3)
  assert _parse_embargo(timedelta(days=1), is_datetime=False) == 86400.0
  assert _parse_embargo(7, is_datetime=True) == pd.Timedelta(days=7)
  assert _parse_embargo(7, is_datetime=False) == 7.0
  assert _parse_embargo(1.5, is_datetime=True) == pd.Timedelta(days=1.5)
  assert _embargo_str(pd.Timedelta(days=1.5), is_datetime=True) == 'P1.5D'
  assert _embargo_str(2.5, is_datetime=False) == '2.5'
  assert _parse_embargo('7', is_datetime=True) == pd.Timedelta(days=7)
  assert _parse_embargo('7 days', is_datetime=True) == pd.Timedelta(days=7)
  assert _parse_embargo('', is_datetime=True) == pd.Timedelta(0)
  assert _parse_embargo('P7D', is_datetime=True) == pd.Timedelta(days=7)
  assert _parse_embargo('P7D', is_datetime=False) == 604800.0
  assert _parse_embargo('P0D', is_datetime=True) == pd.Timedelta(0)
  assert _parse_embargo('  2 ', is_datetime=False) == 2.0
  with pytest.raises(ValueError, match='embargo'):
    _parse_embargo(True, is_datetime=True)
  with pytest.raises(ValueError, match='embargo'):
    _parse_embargo('not-a-duration!', is_datetime=True)
  with pytest.raises(ValueError, match='embargo'):
    _parse_embargo([], is_datetime=True)
  with pytest.raises(ValueError, match='embargo'):
    _parse_embargo('P?', is_datetime=True)
  # assert_no_leakage non-dict and numeric gap
  with pytest.raises(ValueError, match='temporal leakage'):
    assert_no_leakage('not-a-dict')
  with pytest.raises(ValueError, match='temporal leakage'):
    assert_no_leakage({'max_time_train': True, 'min_time_test': 5,
                       'embargo': 0})
  assert_no_leakage({'max_time_train': '10', 'min_time_test': '20',
                     'embargo': '5'})
  with pytest.raises(ValueError, match='gap'):
    assert_no_leakage({'max_time_train': '10', 'min_time_test': '12',
                       'embargo': '5'})
  # meta times as objects
  assert_no_leakage(
    {'max_time_train': pd.Timestamp('2020-01-01', tz='UTC'),
     'min_time_test': pd.Timestamp('2020-01-10', tz='UTC'),
     'embargo': 0})
  assert_no_leakage(
    {'max_time_train': 10, 'min_time_test': 20, 'embargo': 5})
  # fallback parse via to_datetime for exotic type
  import datetime as _dt
  assert_no_leakage(
    {'max_time_train': _dt.datetime(2020, 1, 1, tzinfo=_dt.timezone.utc),
     'min_time_test': _dt.datetime(2020, 1, 10, tzinfo=_dt.timezone.utc),
     'embargo': 0})


def test_temporal_split_tuple_and_raw_meta():
  '''Tuple/list and raw-meta inputs rejected; frozen presplit works.'''
  from forecast_ml_framework.pipelines.modeling.data_split import (
    temporal_split,
  )
  base = pd.Timestamp('2020-01-01', tz='UTC')
  train = pd.DataFrame({
    'time': [base + pd.Timedelta(days=d) for d in [0, 1, 2]],
    'f1': [1.0, 2.0, 3.0], 'target': [0, 1, 0]})
  test = pd.DataFrame({
    'time': [base + pd.Timedelta(days=d) for d in [5, 6, 7]],
    'f1': [4.0, 5.0, 6.0], 'target': [1, 0, 1]})
  # tuple input rejected (frozen: use positional test_df instead)
  with pytest.raises(ValueError, match='tuple/list'):
    temporal_split(
      (train, test), {'method': 'presplit', 'time_col': 'time'})
  # list form rejected
  with pytest.raises(ValueError, match='tuple/list'):
    temporal_split(
      [train, test], {'method': 'presplit', 'time_col': 'time'})
  # split_meta_raw fallback removed: no test frame raises
  df = _daily_frame(10)
  raw = {'max_time_train': '2020-01-01T00:00:00+00:00',
         'min_time_test': '2020-01-10T00:00:00+00:00', 'embargo': 'P0D'}
  with pytest.raises(ValueError, match='presplit requires'):
    temporal_split(
      df, {'method': 'presplit', 'time_col': 'time',
           'split_meta_raw': raw})
  # raw meta alone does not stand in for frames (leaky raw too)
  raw_bad = {'max_time_train': '2020-01-10T00:00:00+00:00',
             'min_time_test': '2020-01-10T00:00:00+00:00',
             'embargo': 'P0D'}
  with pytest.raises(ValueError, match='presplit requires'):
    temporal_split(df, {'method': 'presplit', 'split_meta_raw': raw_bad})
  # no cut info raises
  with pytest.raises(ValueError, match='presplit requires'):
    temporal_split(df, {'method': 'presplit', 'time_col': 'time'})
  # frozen presplit: test frame via third positional argument
  tr, te, _ = temporal_split(
    train, {'method': 'presplit', 'time_col': 'time'}, test_df=test)
  assert len(tr) == 3 and len(te) == 3
  tr2, te2, meta2 = temporal_split(
    train, {'method': 'presplit', 'time_col': 'time'}, test)
  assert tr2.equals(train) and te2.equals(test)
  assert meta2['leakage_checked'] is True
  # deleted params-key form with raw meta raises (no silent fallback)
  with pytest.raises(ValueError, match='presplit requires'):
    temporal_split(
      train, {'method': 'presplit', 'time_col': None,
              'test_df': test,
              'split_meta_raw': {'max_time_train':
                                 '2020-01-02T00:00:00+00:00',
                                 'min_time_test':
                                 '2020-01-05T00:00:00+00:00',
                                 'embargo': 'P0D'}})
  # presplit train/test None raises via direct helper
  from forecast_ml_framework.pipelines.modeling.data_split import (
    _temporal_split_presplit,
  )
  with pytest.raises(ValueError, match='presplit requires'):
    _temporal_split_presplit(None, test, {})
  # presplit with numeric fractional times (frozen helper order)
  train_f = pd.DataFrame({'time': [0.5, 1.5], 'f1': [1.0, 2.0],
                          'target': [0, 1]})
  test_f = pd.DataFrame({'time': [5.5, 6.5], 'f1': [3.0, 4.0],
                         'target': [1, 0]})
  _, _, meta_f = _temporal_split_presplit(
    train_f, test_f, {'time_col': 'time', 'embargo': 1.5})
  assert meta_f['max_time_train'] == '1.5'
  assert meta_f['min_time_test'] == '5.5'
  # malformed test_size raises (fail-loud, no silent 0.2 fallback)
  with pytest.raises(ValueError, match='test_size'):
    _temporal_split_presplit(
      train_f, test_f, {'time_col': 'time', 'test_size': 'bad'})
  with pytest.raises(ValueError, match='test_size'):
    temporal_split(
      train, {'method': 'presplit', 'time_col': 'time',
              'test_size': 'bad'}, test)


def test_temporal_split_empty_bad_and_single():
  '''Empty/bad test_size/single-row temporal errors.'''
  from forecast_ml_framework.pipelines.modeling.data_split import (
    temporal_split,
  )
  with pytest.raises(ValueError, match='non-empty'):
    temporal_split(pd.DataFrame({'time': [], 'f1': [], 'target': []}),
                   {'time_col': 'time', 'method': 'temporal'})
  df = _daily_frame(10)
  with pytest.raises(ValueError, match='test_size'):
    temporal_split(df, {'time_col': 'time', 'test_size': True})
  with pytest.raises(ValueError, match='test_size'):
    temporal_split(df, {'time_col': 'time', 'test_size': 2})
  # single row: test_start <=0
  base = pd.Timestamp('2020-01-01', tz='UTC')
  single = pd.DataFrame({'time': [base], 'f1': [1.0], 'target': [0]})
  with pytest.raises(ValueError, match='embargo consumes'):
    temporal_split(single, {'time_col': 'time', 'test_size': 0.5})
  # numeric fractional times
  df_f = pd.DataFrame({'time': [0.5, 1.5, 2.5, 3.5, 4.5],
                       'f1': list(range(5)), 'target': [0, 1, 0, 1, 0]})
  train, test, meta = temporal_split(
    df_f, {'time_col': 'time', 'test_size': 0.4, 'embargo': 0.5,
           'method': 'temporal'})
  assert meta['max_time_train'] == '2.5'
  assert meta['min_time_test'] == '3.5'
  # embargo via sampling_params
  train2, test2, _ = temporal_split(
    _daily_frame(20),
    {'sampling_params': {'method': 'temporal', 'time_col': 'time',
                         'embargo': 0, 'test_size': 0.25}})
  assert len(test2) == 5
  # sampling_params embargo overrides top-level
  train3, test3, meta3 = temporal_split(
    _daily_frame(100),
    {'time_col': 'time', 'test_size': 0.2, 'embargo': 0,
     'method': 'temporal',
     'sampling_params': {'embargo': __import__('datetime').timedelta(
       days=7)}})
  assert meta3['n_embargoed'] == 7
  # params not dict defaults to temporal missing time_col
  with pytest.raises(ValueError, match='time_col'):
    temporal_split(df, 'not-a-dict')
