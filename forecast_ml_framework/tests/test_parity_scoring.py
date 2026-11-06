#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Parity Task 6: scoring ntile/calib/high_score columns.'''

import numpy as np
import pandas as pd
import pytest

from forecast_ml_framework.pipelines.scoring.nodes import (
  _score_pandas_chunk,
)


class _FakeClassifier:
  '''Fake classifier whose max-proba spans (0, 1).'''

  def __init__(self, scores):
    self._scores = np.asarray(scores, dtype=float)

  def predict(self, x):
    return (self._scores[:len(x)] >= 0.5).astype(int)

  def predict_proba(self, x):
    scores = self._scores[:len(x)]
    return np.column_stack([1 - scores, scores])


class _FakeRegressor:
  '''Fake regressor without predict_proba.'''

  def __init__(self, values):
    self._values = np.asarray(values, dtype=float)

  def predict(self, x):
    return self._values[:len(x)]


def _frame(n=100):
  rng = np.random.default_rng(2)
  return pd.DataFrame({'id': np.arange(n), 'f1': rng.random(n)})


def test_cls_score_decile_centile_and_high_score():
  '''Classification chunk gains decile/centile/high_score columns.'''
  scores = np.linspace(0.05, 0.95, 100)
  params = {'task_type': 'classification', 'model_threshold': 0.5}
  out = _score_pandas_chunk(_frame(), _FakeClassifier(scores), params,
                              ['id'], 1000)
  for col in ('score_decile', 'score_centile', 'high_score_ind'):
    assert col in out.columns
  assert out['score_decile'].max() <= 10
  assert out['score_decile'].min() >= 1
  assert out['score_centile'].max() <= 100
  assert out['score_centile'].min() >= 1
  assert set(out['high_score_ind'].unique()) <= {'Y', 'N'}
  expect = np.where(
      (out['proba'] > 0.5) & (out['score_decile'] >= 5), 'Y', 'N')
  assert out['high_score_ind'].tolist() == expect.tolist()


def test_reg_score_decile_centile_no_high_score():
  '''Regression gains decile/centile but no high_score_ind.'''
  values = np.linspace(1.0, 100.0, 100)
  params = {'task_type': 'regression'}
  out = _score_pandas_chunk(_frame(), _FakeRegressor(values), params,
                              ['id'], 1000)
  assert 'score_decile' in out.columns
  assert 'score_centile' in out.columns
  assert 'high_score_ind' not in out.columns
  assert out['score_centile'].max() <= 100


def test_calib_columns_when_calib_proba_param_present():
  '''Calib decile/centile appear only when calib values are given.'''
  scores = np.linspace(0.05, 0.95, 100)
  calib = np.linspace(0.1, 0.9, 100)
  params = {
      'task_type': 'classification',
      'calib_proba': calib.tolist(),
  }
  out = _score_pandas_chunk(_frame(), _FakeClassifier(scores), params,
                              ['id'], 1000)
  assert 'calib_decile' in out.columns
  assert 'calib_centile' in out.columns
  assert out['calib_centile'].max() <= 100
  params_no_calib = {'task_type': 'classification'}
  out2 = _score_pandas_chunk(_frame(), _FakeClassifier(scores),
                              params_no_calib, ['id'], 1000)
  assert 'calib_decile' not in out2.columns
  assert 'calib_centile' not in out2.columns


def test_calib_columns_from_named_column():
  '''calib_proba naming a df column also yields calib columns.'''
  scores = np.linspace(0.05, 0.95, 100)
  df = _frame()
  df['calib'] = np.linspace(0.2, 0.8, 100)
  params = {'task_type': 'classification', 'calib_proba': 'calib'}
  out = _score_pandas_chunk(df, _FakeClassifier(scores), params,
                              ['id'], 1000)
  assert 'calib_decile' in out.columns
  assert 'calib_centile' in out.columns


class _FeatureClassifier:
  '''Classifier reading positive proba from the f1 feature.'''

  def predict(self, x):
    return (np.asarray(x['f1'], dtype=float) >= 0.5).astype(int)

  def predict_proba(self, x):
    pos = np.asarray(x['f1'], dtype=float)
    return np.column_stack([1 - pos, pos])


class _OneDimProbaClassifier:
  '''Classifier returning 1-D proba (positive-class scores).'''

  def predict(self, x):
    return np.zeros(len(x), dtype=int)

  def predict_proba(self, x):
    return np.linspace(0.05, 0.95, len(x))


class _MultiPartitionSpark:
  '''Fake Spark frame slicing partitions, concatenating results.'''

  def __init__(self, frame, partition_size=10):
    self._frame = frame.copy()
    self.columns = list(frame.columns)
    self._partition_size = partition_size

  def count(self):
    return len(self._frame)

  def mapPartitions(self, func):
    parts = []
    for start in range(0, len(self._frame), self._partition_size):
      part = self._frame.iloc[start:start + self._partition_size]
      parts.append(func(part))
    if not parts:
      return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def test_small_constant_chunk_fallback_no_crash():
  '''Constant proba on few rows falls back to 1s, never crashes.'''
  scores = np.full(5, 0.8)
  params = {'task_type': 'classification'}
  out = _score_pandas_chunk(_frame(5), _FakeClassifier(scores), params,
                              ['id'], 1000)
  assert 'score_decile' in out.columns
  assert 'score_centile' in out.columns
  assert (out['score_decile'] == 1).all()
  assert (out['score_centile'] == 1).all()


def test_asymmetric_proba_uses_positive_class():
  '''[0.95, 0.05] row scores by positive proba 0.05, not confidence.'''
  scores = np.concatenate([[0.05], np.linspace(0.5, 0.99, 99)])
  params = {'task_type': 'classification', 'model_threshold': 0.5}
  out = _score_pandas_chunk(_frame(), _FakeClassifier(scores), params,
                             ['id'], 1000)
  assert out['proba'].iloc[0] == 0.05
  assert out['score_decile'].iloc[0] == 1
  assert out['high_score_ind'].iloc[0] == 'N'


def test_one_dim_proba_warns_and_treated_as_positive_class():
  '''1-D predict_proba warns and is used as positive-class proba.'''
  import warnings
  params = {'task_type': 'classification'}
  with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always')
    out = _score_pandas_chunk(_frame(), _OneDimProbaClassifier(), params,
                               ['id'], 1000)
  assert any(issubclass(w.category, UserWarning) for w in caught)
  assert np.allclose(out['proba'], np.linspace(0.05, 0.95, 100))
  assert out['score_decile'].iloc[0] == 1
  assert out['score_decile'].iloc[-1] == 10


def test_spark_global_ntile_across_partitions():
  '''Multi-partition Spark ntile matches global (not per-partition).'''
  from forecast_ml_framework.pipelines.scoring.nodes import predict
  frame = pd.DataFrame({'id': np.arange(100),
                        'f1': np.linspace(0.01, 0.99, 100)})
  spark = _MultiPartitionSpark(frame, partition_size=10)
  params = {'task_type': 'classification',
            'spark_scoring': {'enabled': True},
            'model_threshold': 0.5}
  out = predict(spark, params, model_loader=lambda: _FeatureClassifier())
  assert (out['score_decile'].iloc[:10] == 1).all()
  assert (out['score_decile'].iloc[-10:] == 10).all()
  assert out['score_decile'].iloc[-1] > out['score_decile'].iloc[0]
  expected = _ntile_global(out['proba'])
  assert out['score_decile'].tolist() == expected.tolist()


def _ntile_global(values):
  return np.asarray(pd.qcut(np.asarray(values, dtype=float), q=10,
                            labels=False, duplicates='drop') + 1)


def test_ntile_edges_and_labels_roundtrip():
  '''Global edges reproduce qcut tiles for the same scores.'''
  from forecast_ml_framework.pipelines.scoring.nodes import (
    _labels_from_edges,
    _ntile_edges,
  )
  scores = np.linspace(0.01, 0.99, 100)
  edges = _ntile_edges(scores, 10)
  labels = _labels_from_edges(scores, edges, 10)
  assert labels[0] == 1 and labels[-1] == 10
  assert (labels[1:] >= labels[:-1]).all()
  assert set(labels.tolist()) <= set(range(1, 11))


def test_labels_from_edges_constant_falls_back_to_one():
  '''Degenerate edges map every score to tile 1.'''
  from forecast_ml_framework.pipelines.scoring.nodes import (
    _labels_from_edges,
    _ntile_edges,
  )
  edges = _ntile_edges(np.full(5, 0.8), 10)
  assert _labels_from_edges(np.full(5, 0.8), edges, 10).tolist() == [1] * 5


def test_empty_predictions_full_schema():
  '''Empty predictions carry full ntile/high/calib schema.'''
  from forecast_ml_framework.pipelines.scoring.nodes import (
    _empty_predictions,
  )
  out = _empty_predictions({})
  for col in ('score_decile', 'score_centile', 'high_score_ind',
              'calib_decile', 'calib_centile'):
    assert col in out.columns
  assert len(out) == 0
  assert pd.api.types.is_integer_dtype(out['score_decile'])
  assert pd.api.types.is_integer_dtype(out['score_centile'])


def test_model_threshold_unvalidated_raises():
  '''model_threshold must be finite in [0, 1] else informative error.'''
  import pytest
  scores = np.linspace(0.05, 0.95, 100)
  for bad in (float('inf'), float('nan'), -5, 2, 'abc'):
    params = {'task_type': 'classification', 'model_threshold': bad}
    with pytest.raises(ValueError, match='model_threshold'):
      _score_pandas_chunk(_frame(), _FakeClassifier(scores), params,
                          ['id'], 1000)


def test_task_alias_cls_scores_as_classification():
  '''Task alias cls in scoring is classification (proba+high_score).'''
  scores = np.linspace(0.05, 0.95, 100)
  params = {'task_type': 'cls'}
  out = _score_pandas_chunk(_frame(), _FakeClassifier(scores), params,
                            ['id'], 1000)
  assert 'high_score_ind' in out.columns
  assert not out['proba'].isna().all()


# --- high-fix: OHE collision + feature_names_in_ alignment (RED) ---

def test_ohe_dummy_name_collision_raises():
  '''a/b_c vs a_b/c both expand to a_b_c -> ValueError.'''
  import pytest
  df = pd.DataFrame({'id': [1, 2], 'a': ['b_c', 'b_c'],
                     'a_b': ['c', 'c']})
  with pytest.raises(ValueError, match='a_b_c|collision|duplicate'):
    _score_pandas_chunk(df, _FakeRegressor([0.0, 1.0]),
                        {'task_type': 'regression'}, ['id'], 1000)


def test_ohe_dummy_collides_with_existing_column_raises():
  '''Dummy a_b collides with existing numeric column a_b.'''
  import pytest
  df = pd.DataFrame({'id': [1, 2], 'a': ['b', 'b'],
                     'a_b': [1.0, 2.0]})
  with pytest.raises(ValueError, match='a_b|collision|duplicate'):
    _score_pandas_chunk(df, _FakeRegressor([0.0, 1.0]),
                        {'task_type': 'regression'}, ['id'], 1000)


class _RecordingRegressor:
  '''Regressor recording column order it was called with.'''

  def __init__(self, feature_names):
    self.feature_names_in_ = list(feature_names)
    self.seen_columns = []

  def predict(self, x):
    self.seen_columns.append(list(x.columns))
    return np.zeros(len(x), dtype=float)


def test_scoring_aligns_to_feature_names_in_order():
  '''Expanded/present columns reordered to model fitted order.'''
  df = pd.DataFrame({'id': [1, 2], 'f1': [10.0, 20.0],
                     'f2': [1.0, 2.0]})
  model = _RecordingRegressor(['f2', 'f1'])
  _score_pandas_chunk(df, model, {'task_type': 'regression'},
                      ['id'], 1000)
  assert model.seen_columns
  assert model.seen_columns[0] == ['f2', 'f1']


def test_scoring_fills_missing_feature_names_in_with_zero():
  '''Missing fitted feature is filled with 0, order preserved.'''
  captured = {}

  class _MissingRegressor:
    feature_names_in_ = ('f2', 'f_missing', 'f1')

    def predict(self, x):
      captured['columns'] = list(x.columns)
      captured['missing'] = list(x['f_missing'])
      return np.zeros(len(x), dtype=float)

  df = pd.DataFrame({'id': [1, 2], 'f1': [10.0, 20.0],
                     'f2': [1.0, 2.0]})
  _score_pandas_chunk(df, _MissingRegressor(),
                      {'task_type': 'regression'}, ['id'], 1000)
  assert captured['columns'] == ['f2', 'f_missing', 'f1']
  assert captured['missing'] == [0, 0]


def test_scoring_ohe_aligns_to_feature_names_in_order():
  '''OHE expansion order follows feature_names_in_, not appearance.'''
  captured = {}

  class _OheRegressor:
    feature_names_in_ = ('cat_b', 'f1', 'cat_a')

    def predict(self, x):
      captured['columns'] = list(x.columns)
      return np.zeros(len(x), dtype=float)

  df = pd.DataFrame({'id': [1, 2], 'cat': ['a', 'b'],
                     'f1': [3.0, 4.0]})
  _score_pandas_chunk(df, _OheRegressor(),
                      {'task_type': 'regression'}, ['id'], 1000)
  assert captured['columns'] == ['cat_b', 'f1', 'cat_a']


# --- loop8 non-finite predictions (RED) ---

class _BadPredictionRegressor:
  '''Regressor returning a fixed non-finite / bad prediction.'''

  def __init__(self, value):
    self._value = value

  def predict(self, x):
    n = len(x)
    value = self._value
    if isinstance(value, dict):
      return {'a': [value['a']] * n}
    if isinstance(value, np.ndarray):
      return np.full(n, value.flat[0], dtype=value.dtype)
    return [value] * n


def test_nonfinite_predictions_raise():
  '''nan/inf/Decimal/complex/np predictions raise ValueError.'''
  import pytest
  from decimal import Decimal
  bad = [
    float('nan'),
    float('inf'),
    float('-inf'),
    Decimal('Infinity'),
    Decimal('NaN'),
    complex(float('inf'), 0.0),
    complex(0.0, float('nan')),
    np.float32(float('nan')),
    np.float64(float('inf')),
    np.array([float('inf')]),
    np.array([np.float32(float('nan'))], dtype=object),
    {'a': float('inf')},
    {'a': np.float32(float('nan'))},
    [float('nan')],
  ]
  for value in bad:
    with pytest.raises(ValueError, match='finite'):
      _score_pandas_chunk(_frame(5), _BadPredictionRegressor(value),
                          {'task_type': 'regression'}, ['id'], 1000)


def test_nonfinite_proba_raises():
  '''Non-finite predict_proba output raises ValueError.'''
  import pytest

  class _BadProbaClassifier:
    def predict(self, x):
      return np.zeros(len(x), dtype=int)

    def predict_proba(self, x):
      return np.column_stack(
          [np.full(len(x), float('inf')), np.full(len(x), 0.0)])

  with pytest.raises(ValueError, match='finite'):
    _score_pandas_chunk(_frame(5), _BadProbaClassifier(),
                        {'task_type': 'classification'}, ['id'], 1000)


def test_string_predictions_pass():
  '''String class-label predictions are finite-checked as labels.'''
  class _StringClassifier:
    def predict(self, x):
      return ['yes', 'no'] * (len(x) // 2) + (['yes'] if len(x) % 2 else [])

    def predict_proba(self, x):
      pos = np.linspace(0.05, 0.95, len(x))
      return np.column_stack([1 - pos, pos])

  out = _score_pandas_chunk(_frame(4), _StringClassifier(),
                            {'task_type': 'classification'}, ['id'], 1000)
  assert out['prediction'].tolist() == ['yes', 'no', 'yes', 'no']


# --- loop9 scoring masked (RED) ---

def test_scoring_masked_predictions_raise():
  import pytest
  from forecast_ml_framework.pipelines.scoring.nodes import (
    _check_predictions_finite,
  )
  with pytest.raises(ValueError, match='finite'):
    _check_predictions_finite(np.ma.masked)
  with pytest.raises(ValueError, match='finite'):
    _check_predictions_finite(
      np.ma.array([1.0, 2.0], mask=[False, True]))


def test_scoring_masked_array_clean_passes_and_masked_model_raises():
  import pytest

  class _MaskedRegressor:
    def predict(self, x):
      return np.ma.array([1.0] * len(x))

  class _MaskedBadRegressor:
    def predict(self, x):
      return np.ma.masked

  out = _score_pandas_chunk(_frame(3), _MaskedRegressor(),
                            {'task_type': 'regression'}, ['id'], 1000)
  assert out['prediction'].tolist() == [1.0, 1.0, 1.0]
  with pytest.raises(ValueError, match='finite'):
    _score_pandas_chunk(_frame(3), _MaskedBadRegressor(),
                        {'task_type': 'regression'}, ['id'], 1000)


# --- loop10 generator scoring (RED) ---

def test_scoring_generator_predict_regression():
  '''Generator predict output is materialized once, not consumed.'''

  class _GenRegressor:
    def predict(self, x):
      return (float(v) for v in [1.0, 2.0, 3.0][:len(x)])

  out = _score_pandas_chunk(_frame(3), _GenRegressor(),
                            {'task_type': 'regression'}, ['id'], 1000)
  assert out['prediction'].tolist() == [1.0, 2.0, 3.0]


def test_scoring_generator_proba_and_predict_classification():
  '''Generator proba+predict outputs are materialized once.'''

  class _GenClassifier:
    def predict(self, x):
      return (int(v) for v in [0, 1, 0][:len(x)])

    def predict_proba(self, x):
      rows = [[0.9, 0.1], [0.2, 0.8], [0.7, 0.3]][:len(x)]
      return (row for row in rows)

  out = _score_pandas_chunk(_frame(3), _GenClassifier(),
                            {'task_type': 'classification'}, ['id'], 1000)
  assert out['prediction'].tolist() == [0, 1, 0]
  assert out['proba'].tolist() == [0.1, 0.8, 0.3]


def test_predictions_canonical_home_in_core():
  '''Serving and scoring share the single core predictions check.'''
  from forecast_ml_framework.core import predictions as core_pred
  from forecast_ml_framework.pipelines.scoring import nodes as scoring
  from forecast_ml_framework.serving import app as serving

  assert serving._check_predictions_finite is (
      core_pred.check_predictions_finite)
  assert scoring._check_predictions_finite is (
      core_pred.check_predictions_finite)
  assert serving._materialize_predictions is (
      core_pred.materialize_predictions)
  assert core_pred.materialize_predictions(
      (v for v in [1.0, 2.0])) == [1.0, 2.0]
  core_pred.check_predictions_finite([1.0, 2.0])
  with pytest.raises(ValueError, match='finite'):
    core_pred.check_predictions_finite([float('inf')])
