#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Boundary tests for split_data.

Covers None frames, bad test_size types, empty feature
sets, determinism, and stratify edges.
'''

import numpy as np
import pandas as pd
import pytest

from forecast_ml_framework.pipelines.modeling.data_split import (
  split_data,
)


def _frame(n=20, seed=0):
  '''Return a small classification frame.

  Args:
    n: Row count of the synthetic frame.
    seed: Random seed for the frame values.

  Returns:
    DataFrame with f1/f2 features and a binary target.
  '''
  rng = np.random.RandomState(seed)
  return pd.DataFrame({
    'f1': rng.randn(n),
    'f2': rng.randn(n),
    'target': rng.randint(0, 2, n),
  })


def _params(**overrides):
  '''Return base split params with overrides applied.

  Args:
    **overrides: Keys overriding the base params dict.

  Returns:
    Modeling params dict for split_data.
  '''
  params = {
    'target_col': 'target',
    'feature_cols': ['f1', 'f2'],
    'task_type': 'classification',
    'test_size': 0.25,
    'random_state': 7,
  }
  params.update(overrides)
  return params


def test_split_none_frame_raises():
  '''None frame raises ValueError.

  Fails if split_data accepts a None dataframe.
  '''
  with pytest.raises(ValueError, match='non-empty'):
    split_data(None, _params())


def test_split_bool_test_size_raises():
  '''Bool test_size raises ValueError.

  Fails if bool slips past the test_size guard.
  '''
  with pytest.raises(ValueError, match='test_size'):
    split_data(_frame(), _params(test_size=True))


def test_split_nan_test_size_raises():
  '''NaN test_size raises ValueError.

  Fails if NaN slips past the test_size guard.
  '''
  with pytest.raises(ValueError, match='test_size'):
    split_data(_frame(), _params(test_size=float('nan')))


def test_split_str_test_size_raises():
  '''String test_size raises ValueError.

  Fails if string test_size slips past the guard.
  '''
  with pytest.raises(ValueError, match='test_size'):
    split_data(_frame(), _params(test_size='0.2'))


def test_split_empty_feature_cols():
  '''Empty feature_cols split into zero-column frames.

  Fails if empty feature sets raise instead of splitting.
  '''
  x_train, x_test, _, _, _ = split_data(
    _frame(), _params(feature_cols=[]))
  assert x_train.shape[1] == 0
  assert x_test.shape[1] == 0
  assert len(x_train) + len(x_test) == 20


def test_split_deterministic_same_seed():
  '''Same seed gives identical splits.

  Fails if random_state is ignored during splitting.
  '''
  first = split_data(_frame(), _params())
  second = split_data(_frame(), _params())
  assert first[0].equals(second[0])
  assert first[1].equals(second[1])
  assert first[2].tolist() == second[2].tolist()
  assert first[3].tolist() == second[3].tolist()


def test_split_single_row_raises():
  '''Single-row frame raises ValueError.

  Fails if single-row input returns an empty train set.
  '''
  single = pd.DataFrame({'f1': [1.0], 'target': [0]})
  with pytest.raises(ValueError, match='train set'):
    split_data(single, _params(
      test_size=0.5, random_state=1, feature_cols=['f1']))


def test_split_default_excludes_id():
  '''Default features (no feature_cols) exclude id.

  Fails if the direct-use path leaks the id column
  into the feature frame (ML-24c).
  '''
  frame = pd.DataFrame({
    'id': range(20),
    'x': np.arange(20, dtype=float),
    'target': [0, 1] * 10,
  })
  params = _params()
  del params['feature_cols']
  x_train, x_test, _, _, _ = split_data(frame, params)
  assert 'id' not in x_train.columns
  assert 'id' not in x_test.columns
  assert list(x_train.columns) == ['x']


def test_split_id_cols_param_excludes_custom_id():
  '''id_cols param excludes the named id column by default.

  Fails if a custom id_cols list is ignored when
  feature_cols is omitted.
  '''
  frame = pd.DataFrame({
    'user_id': range(20),
    'x': np.arange(20, dtype=float),
    'target': [0, 1] * 10,
  })
  params = _params(id_cols=['user_id'])
  del params['feature_cols']
  x_train, _, _, _, _ = split_data(frame, params)
  assert 'user_id' not in x_train.columns
  assert list(x_train.columns) == ['x']


def test_split_variables_id_cols_excludes():
  '''variables id_cols passed via params are excluded.

  Fails if a variables-style id list in params is
  ignored when feature_cols is omitted.
  '''
  frame = pd.DataFrame({
    'user_id': range(20),
    'x': np.arange(20, dtype=float),
    'target': [0, 1] * 10,
  })
  params = _params(variables={'id_cols': ['user_id']})
  del params['feature_cols']
  x_train, _, _, _, _ = split_data(frame, params)
  assert 'user_id' not in x_train.columns
  assert list(x_train.columns) == ['x']


def test_split_stratify_missing_col_raises():
  '''Missing stratify column raises KeyError.

  Fails if a missing stratify column is ignored.
  '''
  with pytest.raises(KeyError, match='missing'):
    split_data(_frame(), _params(sampling_params={
      'method': 'stratified', 'stratify_col': 'missing'}))


def test_split_regression_default_meta_is_random():
  '''Regression default (no sampling_params) meta is random.

  Default method is stratified, but without stratify the
  split is random; meta must not claim stratified.
  '''
  params = _params(task_type='regression')
  _, _, _, _, meta = split_data(_frame(), params)
  assert meta['requested_method'] == 'stratified'
  assert meta['sampling_method'] == 'random'


def test_split_single_class_meta_is_random():
  '''Single-class stratify nulled: meta is random.

  Stratify is nulled after guards; meta must not claim
  stratified for a split that used no stratification.
  '''
  frame = pd.DataFrame({
    'f1': np.arange(20, dtype=float),
    'f2': np.arange(20, dtype=float),
    'target': [0] * 20,
  })
  _, _, _, _, meta = split_data(frame, _params())
  assert meta['sampling_method'] == 'random'
  assert meta['sampling_method'] != 'stratified'
