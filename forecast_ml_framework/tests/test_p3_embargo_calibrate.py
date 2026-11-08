#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''P3-C guards and calibrate prohibition tests.'''

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from forecast_ml_framework.pipelines.modeling.calibration import (
  _is_temporal_split,
  calibrate_model,
)
from forecast_ml_framework.pipelines.modeling.data_split import (
  _temporal_fold_assignment,
  assert_no_overlap,
  embargo_gap,
  temporal_split,
)


def _daily_frame(n=100, start='2024-01-01'):
  '''Return a daily frame with f1 and target.'''
  times = pd.date_range(start=start, periods=n, freq='D', tz='UTC')
  return pd.DataFrame({
    'time': times,
    'f1': np.arange(n, dtype=float),
    'target': [0, 1] * (n // 2),
  })


def _fitted_base(n=20):
  '''Return a fitted adapter on a binary fixture.'''
  from forecast_ml_framework.pipelines.modeling.adapters import (
    SklearnAdapter,
  )
  rng = np.random.RandomState(0)
  x = pd.DataFrame({'f1': rng.randn(n), 'f2': rng.randn(n)})
  y = pd.Series([0, 1] * (n // 2))
  base = SklearnAdapter({}).fit(x, y)
  return base, x, y


def test_fold_assignment_golden_73_20_7():
  '''Helper reproduces the frozen 73/20/7 assignment.'''
  df = _daily_frame(100)
  out = _temporal_fold_assignment(
    df, time_col='time', test_start=80, n_test_block=20,
    embargo=timedelta(days=7), purge=None)
  assert out['n_train'] == 73
  assert out['n_test'] == 20
  assert out['n_embargoed'] == 7
  assert out['n_purged'] == 0
  assert out['embargo'] == 'P7D'
  assert out['purge'] == 'P7D'
  assert len(out['train_df']) == 73
  assert len(out['test_df']) == 20


def test_fold_assignment_purge_beyond_embargo():
  '''Purge beyond embargo moves train_end earlier by 3.'''
  df = _daily_frame(100)
  out = _temporal_fold_assignment(
    df, time_col='time', test_start=80, n_test_block=20,
    embargo=timedelta(days=7), purge=timedelta(days=10))
  assert out['n_train'] == 70
  assert out['n_embargoed'] == 7
  assert out['n_purged'] == 3
  assert out['embargo'] == 'P10D'
  assert out['purge'] == 'P10D'


def test_fold_assignment_boundary_violations():
  '''Invalid assignment geometry raises ValueError.'''
  df = _daily_frame(100)
  with pytest.raises(ValueError):
    _temporal_fold_assignment(
      df, time_col='time', test_start=0, n_test_block=20)
  with pytest.raises(ValueError):
    _temporal_fold_assignment(
      df, time_col='time', test_start=80, n_test_block=0)
  with pytest.raises(ValueError):
    _temporal_fold_assignment(
      df, time_col='time', test_start=81, n_test_block=20)
  with pytest.raises(ValueError):
    _temporal_fold_assignment(
      df, time_col='time', test_start=True, n_test_block=20)
  with pytest.raises(ValueError):
    _temporal_fold_assignment(
      df, time_col='time', test_start=80.0, n_test_block=20)
  with pytest.raises(ValueError):
    _temporal_fold_assignment(
      df, time_col='time', test_start=80, n_test_block=False)


def test_fold_assignment_bad_frame_and_tz():
  '''Empty, missing, and mixed-tz inputs raise ValueError.'''
  empty = pd.DataFrame({'time': [], 'f1': [], 'target': []})
  with pytest.raises(ValueError):
    _temporal_fold_assignment(
      empty, time_col='time', test_start=1, n_test_block=1)
  df = _daily_frame(10)
  with pytest.raises(ValueError):
    _temporal_fold_assignment(
      df, time_col='missing', test_start=5, n_test_block=2)
  naive = pd.Timestamp('2024-01-01')
  aware = pd.Timestamp('2024-01-02', tz='UTC')
  mixed = pd.DataFrame({
    'time': [naive, aware, aware, aware],
    'f1': [1.0, 2.0, 3.0, 4.0],
    'target': [0, 1, 0, 1],
  })
  with pytest.raises(ValueError, match='mixed naive/aware'):
    _temporal_fold_assignment(
      mixed, time_col='time', test_start=2, n_test_block=1)


def test_fold_assignment_numeric_times():
  '''Numeric time_col splits in time units.'''
  df = pd.DataFrame({
    'time': list(range(100)),
    'f1': np.arange(100, dtype=float),
    'target': [0, 1] * 50,
  })
  out = _temporal_fold_assignment(
    df, time_col='time', test_start=80, n_test_block=20,
    embargo=7, purge=None)
  assert out['n_train'] == 73
  assert out['n_test'] == 20
  assert out['embargo'] == '7'


def test_embargo_gap_datetime_numeric_fraction():
  '''Gap is float days for datetime and units for numeric.'''
  meta = {
    'max_time_train': '2024-01-01T00:00:00+00:00',
    'min_time_test': '2024-01-09T00:00:00+00:00',
  }
  gap = embargo_gap(meta)
  assert gap == 8.0
  assert type(gap) is float
  numeric = {'max_time_train': '72', 'min_time_test': '80'}
  assert embargo_gap(numeric) == 8.0
  assert type(embargo_gap(numeric)) is float
  frac = {
    'max_time_train': '2024-01-01T00:00:00+00:00',
    'min_time_test': '2024-01-02T12:00:00+00:00',
  }
  assert embargo_gap(frac) == 1.5


def test_embargo_gap_missing_and_mixed():
  '''Missing or mixed-type gap inputs raise ValueError.'''
  with pytest.raises(ValueError):
    embargo_gap({'min_time_test': '80'})
  with pytest.raises(ValueError):
    embargo_gap({'max_time_train': '72'})
  with pytest.raises(ValueError):
    embargo_gap({'max_time_train': None, 'min_time_test': '80'})
  mixed = {
    'max_time_train': '72',
    'min_time_test': '2024-01-09T00:00:00+00:00',
  }
  with pytest.raises(ValueError):
    embargo_gap(mixed)


def _fold(index, start_idx, max_time, min_time):
  '''Build a hand-crafted fold dict for overlap checks.'''
  test_df = pd.DataFrame({
    'time': [min_time], 'f1': [1.0], 'target': [1],
  }, index=[start_idx])
  meta = {
    'max_time_train': max_time,
    'min_time_test': min_time,
    'embargo': 'P0D',
  }
  return {
    'fold_index': index, 'test_df': test_df, 'fold_meta': meta,
  }


def _passing_folds():
  '''Return two disjoint increasing folds.'''
  return [
    _fold(0, 10, '2024-01-01T00:00:00+00:00',
          '2024-01-03T00:00:00+00:00'),
    _fold(1, 20, '2024-01-03T00:00:00+00:00',
          '2024-01-05T00:00:00+00:00'),
  ]


def test_assert_no_overlap_passing():
  '''Disjoint increasing folds pass.'''
  assert assert_no_overlap(_passing_folds()) is None


def test_assert_no_overlap_failures():
  '''Overlap, order, malformed, and empty folds raise.'''
  with pytest.raises(ValueError):
    assert_no_overlap([])
  folds = _passing_folds()
  folds[1]['test_df'] = folds[0]['test_df']
  with pytest.raises(ValueError, match='overlap|disjoint|index'):
    assert_no_overlap(folds)
  folds = _passing_folds()
  folds[0]['fold_meta']['min_time_test'] = '2024-01-06T00:00:00+00:00'
  folds[0]['fold_meta']['max_time_train'] = '2024-01-04T00:00:00+00:00'
  with pytest.raises(ValueError):
    assert_no_overlap(folds)
  folds = _passing_folds()
  del folds[0]['fold_meta']['max_time_train']
  with pytest.raises(ValueError):
    assert_no_overlap(folds)
  folds = _passing_folds()
  folds[0]['fold_meta']['min_time_test'] = None
  with pytest.raises(ValueError):
    assert_no_overlap(folds)
  folds = _passing_folds()
  del folds[0]['test_df']
  with pytest.raises(ValueError):
    assert_no_overlap(folds)
  folds = _passing_folds()
  del folds[0]['fold_meta']
  with pytest.raises(ValueError):
    assert_no_overlap(folds)


def test_assert_no_overlap_leaky_fold_meta():
  '''A leaky fold_meta raises even with disjoint indices.'''
  folds = _passing_folds()
  folds[0]['fold_meta']['max_time_train'] = (
    '2024-01-03T00:00:00+00:00')
  with pytest.raises(ValueError, match='gap'):
    assert_no_overlap(folds)


def test_assert_no_overlap_mixed_time_types():
  '''Folds with mixed time types raise ValueError.'''
  folds = _passing_folds()
  folds[1]['fold_meta']['max_time_train'] = '90'
  folds[1]['fold_meta']['min_time_test'] = '95'
  folds[1]['fold_meta']['embargo'] = '0'
  with pytest.raises(ValueError, match='mixed'):
    assert_no_overlap(folds)


_SIGNALS = [
  ({'method': 'temporal',
    'calibration': {'method': 'sigmoid', 'cv': 2}}, None),
  ({'sampling_params': {'method': 'temporal'},
    'calibration': {'method': 'sigmoid', 'cv': 2}}, None),
  ({'sampling_params': {'method': 'time'},
    'calibration': {'method': 'sigmoid', 'cv': 2}}, None),
  ({'sampling_params': {'method': 'ordered'},
    'calibration': {'method': 'sigmoid', 'cv': 2}}, None),
  ({'calibration': {'method': 'sigmoid', 'cv': 2}},
   {'sampling_method': 'temporal'}),
  ({'calibration': {'method': 'sigmoid', 'cv': 2}},
   {'sampling_method': 'time'}),
  ({'calibration': {'method': 'sigmoid', 'cv': 2}},
   {'sampling_method': 'presplit',
    'raw_sampling_method': 'time',
    'raw_requested_method': 'time'}),
  ({'calibration': {'method': 'sigmoid', 'cv': 2}},
   {'raw_requested_method': 'ordered'}),
]


@pytest.mark.parametrize('params, split_meta', _SIGNALS)
def test_calibrate_temporal_signals_raise(params, split_meta):
  '''Every temporal signal raises when calibrating.'''
  base, x, y = _fitted_base()
  with pytest.raises(
      ValueError, match='calibrate_model prohibited with temporal split'):
    calibrate_model(base, x, y, params, split_meta)


def test_calibrate_handle_override_raises():
  '''BetaCalibration handle overrides none to betacal.'''
  base, x, y = _fitted_base()
  params = {
    'sampling_params': {'method': 'temporal'},
    'calibration': {'method': 'none', 'cv': 2},
    'calibration_handle': 'betacal.BetaCalibration',
  }
  with pytest.raises(
      ValueError, match='calibrate_model prohibited with temporal split'):
    calibrate_model(base, x, y, params)


def test_is_temporal_split_presplit_alone_not_temporal():
  '''Presplit alone, and non-dict meta, are not temporal.'''
  assert _is_temporal_split({}, {'sampling_method': 'presplit'}) is False
  assert _is_temporal_split(None, None) is False
  assert _is_temporal_split({'method': ' Temporal '}, {}) is True
  assert _is_temporal_split(
    {'sampling_params': None}, {}) is False


def test_calibrate_allowed_temporal_proceeds_true():
  '''allow_temporal opt-in proceeds and records True.'''
  base, x, y = _fitted_base()
  params = {
    'sampling_params': {'method': 'temporal'},
    'calibration': {'method': 'sigmoid', 'cv': 2,
                    'allow_temporal': True},
  }
  envelope, meta = calibrate_model(base, x, y, params)
  assert meta['calibrated'] is True
  assert 'temporal_calibration_allowed' in meta
  assert meta['temporal_calibration_allowed'] is True
  assert type(meta['temporal_calibration_allowed']) is bool


def test_calibrate_temporal_none_passthrough_false():
  '''Temporal plus method none proceeds with False.'''
  base, x, y = _fitted_base()
  params = {
    'sampling_params': {'method': 'temporal'},
    'calibration': {'method': 'none', 'cv': 2},
  }
  envelope, meta = calibrate_model(base, x, y, params)
  assert 'temporal_calibration_allowed' in meta
  assert meta['temporal_calibration_allowed'] is False
  assert type(meta['temporal_calibration_allowed']) is bool


def test_calibrate_non_temporal_records_false():
  '''Non-temporal sigmoid proceeds with False meta.'''
  base, x, y = _fitted_base()
  params = {'calibration': {'method': 'sigmoid', 'cv': 2}}
  envelope, meta = calibrate_model(base, x, y, params, None)
  assert meta['calibrated'] is True
  assert meta['temporal_calibration_allowed'] is False
  assert type(meta['temporal_calibration_allowed']) is bool


def test_calibrate_presplit_alone_proceeds_false():
  '''Presplit alone without raw signals proceeds.'''
  base, x, y = _fitted_base()
  params = {'calibration': {'method': 'sigmoid', 'cv': 2}}
  envelope, meta = calibrate_model(
    base, x, y, params, {'sampling_method': 'presplit'})
  assert meta['calibrated'] is True
  assert meta['temporal_calibration_allowed'] is False


def test_calibrate_four_arg_call_temporal_raises():
  '''Four-arg callers keep working and still raise.'''
  base, x, y = _fitted_base()
  params = {
    'sampling_params': {'method': 'temporal'},
    'calibration': {'method': 'sigmoid', 'cv': 2},
  }
  with pytest.raises(
      ValueError, match='calibrate_model prohibited with temporal split'):
    calibrate_model(base, x, y, params)


def test_calibrate_regression_passthrough_meta_bool():
  '''Regression passthrough also carries the bool meta.'''
  from forecast_ml_framework.pipelines.modeling.adapters import (
    SklearnAdapter,
  )
  x = pd.DataFrame({'f1': [1.0, 2.0, 3.0, 4.0]})
  y = pd.Series([1.0, 2.0, 3.0, 4.0])
  base = SklearnAdapter({'model_type': 'linear_regression'}).fit(x, y)
  params = {'task_type': 'regression',
            'calibration': {'method': 'sigmoid', 'cv': 2}}
  envelope, meta = calibrate_model(base, x, y, params)
  assert meta['calibrated'] is False
  assert meta['temporal_calibration_allowed'] is False
  assert type(meta['temporal_calibration_allowed']) is bool


def test_golden_via_temporal_split_helper_gap_overlap():
  '''Frozen split, helper, gap, and 1-fold overlap agree.'''
  df = _daily_frame(100)
  params = {'time_col': 'time', 'test_size': 0.2,
            'embargo': timedelta(days=7), 'method': 'temporal',
            'target_col': 'target', 'task_type': 'classification'}
  train, test, meta = temporal_split(df, params)
  assert len(train) == 73
  assert len(test) == 20
  assert meta['n_embargoed'] == 7
  out = _temporal_fold_assignment(
    df, time_col='time', test_start=80, n_test_block=20,
    embargo=timedelta(days=7), purge=None)
  assert out['n_train'] == 73
  assert out['n_test'] == 20
  assert out['n_embargoed'] == 7
  assert embargo_gap(meta) == 8.0
  assert type(embargo_gap(meta)) is float
  fold = {
    'fold_index': 0, 'test_df': out['test_df'],
    'fold_meta': {'max_time_train': out['max_time_train'],
                  'min_time_test': out['min_time_test'],
                  'embargo': out['embargo']},
  }
  assert assert_no_overlap([fold]) is None


def test_calibrate_probability_edge_branches():
  '''Real-rate extremes, 2-col, frame, and list paths.'''
  from forecast_ml_framework.pipelines.modeling.calibration import (
    calibrate_probability,
  )
  two = np.array([[0.8, 0.2], [0.3, 0.7]])
  out_zero = calibrate_probability(two, 0.0, 0.5)
  assert np.allclose(out_zero[:, 0], 1.0)
  out_one = calibrate_probability(np.array([0.2, 0.8]), 1.0, 0.5)
  assert np.allclose(out_one, 1.0)
  out_two = calibrate_probability(two, 0.1, 0.5)
  assert out_two.shape == (2, 2)
  assert np.allclose(out_two.sum(axis=1), 1.0)
  frame = pd.DataFrame(two, columns=['a', 'b'])
  out_frame = calibrate_probability(frame, 0.1, 0.5)
  assert isinstance(out_frame, pd.DataFrame)
  assert list(out_frame.columns) == ['a', 'b']
  out_list = calibrate_probability([0.2, 0.8], 0.1, 0.5)
  assert isinstance(out_list, np.ndarray)
  with pytest.raises(ValueError, match='proba'):
    calibrate_probability('bad', 0.1, 0.5)


def test_wrap_take_and_cv_validation_helpers():
  '''Wrap, take, and cv validation helpers behave.'''
  from forecast_ml_framework.pipelines.modeling.calibration import (
    _take,
    _validate_calibrate_cv,
    _wrap,
  )
  base, x, y = _fitted_base()
  wrapped = _wrap(base.model)
  assert wrapped.model is base.model
  assert _take(np.array([1, 2, 3]), [0, 2]).tolist() == [1, 3]
  assert list(_take(x, [0, 1]).index) == [0, 1]
  assert _validate_calibrate_cv(2, y) == 2
  with pytest.raises(ValueError, match='non-empty'):
    _validate_calibrate_cv(2, pd.Series([], dtype=float))
  with pytest.raises(ValueError, match='2 classes'):
    _validate_calibrate_cv(2, pd.Series([0, 0, 0]))


def test_oof_proba_direct_and_betacal_nonbinary():
  '''OOF probas are binary; betacal rejects 3 classes.'''
  from forecast_ml_framework.pipelines.modeling.calibration import (
    _oof_proba,
  )
  base, x, y = _fitted_base()
  oof = _oof_proba(base.model, x, y, 2)
  assert oof.shape == (20,)
  assert np.all((oof >= 0) & (oof <= 1))
  base3, x3, _ = _fitted_base()
  y3 = pd.Series([0, 1, 2] * 6 + [0, 1])
  params = {'calibration': {'method': 'betacal', 'cv': 2}}
  with pytest.raises(ValueError, match='binary'):
    calibrate_model(base3, x3, y3, params)


def test_calibrate_nonstring_method_and_envelope_input():
  '''Non-string method raises; envelope input calibrates.'''
  from forecast_ml_framework.pipelines.modeling.envelope import (
    wrap_envelope,
  )
  base, x, y = _fitted_base()
  params = {'calibration': {'method': 123, 'cv': 2}}
  with pytest.raises(ValueError, match='method'):
    calibrate_model(base, x, y, params)
  envelope_in = wrap_envelope(base.model, ['0', '1'])
  params2 = {'calibration': {'method': 'sigmoid', 'cv': 2}}
  envelope, meta = calibrate_model(envelope_in, x, y, params2)
  assert meta['calibrated'] is True
  assert meta['temporal_calibration_allowed'] is False
  assert envelope['label_set'] == ['0', '1']

