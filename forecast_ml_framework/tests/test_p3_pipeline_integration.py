#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''P3-D pipeline integration tests (Contract 4).'''

from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml


def _root():
  '''Return project root.'''
  return Path(__file__).resolve().parents[1]


def _daily_frame(n=100, start='2024-01-01'):
  '''Return daily frame with f1 and target.'''
  times = pd.date_range(start=start, periods=n, freq='D', tz='UTC')
  return pd.DataFrame({
    'time': times,
    'f1': np.arange(n, dtype=float),
    'target': [0, 1] * (n // 2),
  })


def _nested_params(**overrides):
  '''Return real nested temporal params.'''
  params = {
    'sampling_params': {
      'method': 'temporal',
      'time_col': 'time',
      'test_size': 0.2,
      'embargo': timedelta(days=7),
    },
    'calibration': {'method': 'sigmoid', 'cv': 2},
    'target_col': 'target',
    'task_type': 'classification',
  }
  params.update(overrides)
  return params


def test_end_to_end_temporal_73_20_7_real_shapes():
  '''Nested temporal split preserves 73/20/7 through presplit.'''
  from forecast_ml_framework.pipelines.data_prep.nodes import (
    split_raw_data_with_meta,
  )
  from forecast_ml_framework.pipelines.modeling.pipeline import (
    split_presplit_data,
  )
  df = _daily_frame(100)
  params = _nested_params()
  assert 'method' not in params
  train_raw, test_raw, meta_raw = split_raw_data_with_meta(
    df, params)
  assert meta_raw['n_train'] == 73
  assert meta_raw['n_test'] == 20
  assert meta_raw['n_embargoed'] == 7
  assert len(train_raw) == 73
  assert len(test_raw) == 20
  x_tr, x_te, y_tr, y_te, meta = split_presplit_data(
    train_raw, test_raw, params, meta_raw)
  assert meta['n_train'] == 73
  assert meta['n_test'] == 20
  assert meta['n_embargoed'] == 7
  assert meta['sampling_method'] == 'presplit'
  assert meta['raw_sampling_method'] == 'time'
  assert meta['raw_requested_method'] == 'time'
  assert list(x_tr['f1'][:3]) == [0.0, 1.0, 2.0]


def test_nested_temporal_without_time_col_raises():
  '''Nested temporal without time_col raises, no fallback.'''
  from forecast_ml_framework.pipelines.data_prep.nodes import (
    split_raw_data_with_meta,
  )
  df = _daily_frame(100)
  params = {
    'sampling_params': {'method': 'temporal', 'test_size': 0.2},
    'target_col': 'target',
    'task_type': 'classification',
  }
  with pytest.raises(ValueError, match='temporal_split'):
    split_raw_data_with_meta(df, params)


def test_nested_temporal_mixed_tz_raises_value_error():
  '''Mixed naive/aware nested temporal raises ValueError.'''
  from forecast_ml_framework.pipelines.data_prep.nodes import (
    split_raw_data_with_meta,
  )
  times = [
    pd.Timestamp('2024-01-01') + pd.Timedelta(days=i)
    for i in range(100)
  ]
  times[50] = pd.Timestamp('2024-02-20', tz='UTC')
  df = pd.DataFrame({
    'time': times,
    'f1': np.arange(100, dtype=float),
    'target': [0, 1] * 50,
  })
  params = _nested_params()
  with pytest.raises(ValueError, match='mixed naive/aware'):
    split_raw_data_with_meta(df, params)


def test_leaky_presplit_raises():
  '''Swapped train/test (future in train) still raises.'''
  from forecast_ml_framework.pipelines.data_prep.nodes import (
    split_raw_data_with_meta,
  )
  from forecast_ml_framework.pipelines.modeling.pipeline import (
    split_presplit_data,
  )
  df = _daily_frame(100)
  params = _nested_params()
  train_raw, test_raw, meta_raw = split_raw_data_with_meta(
    df, params)
  with pytest.raises(ValueError, match='gap'):
    split_presplit_data(test_raw, train_raw, params, {})


def test_calibrate_node_wires_split_meta():
  '''Calibrate node receives split_meta as fifth input.'''
  from forecast_ml_framework.pipelines.modeling.pipeline import (
    create_pipeline,
  )
  pipe = create_pipeline()
  matched = [node for node in pipe.nodes
             if node.name == 'calibrate_model_node']
  assert len(matched) == 1
  assert list(matched[0].inputs) == [
    'best_model', 'x_train', 'y_train',
    'params:modeling_params', 'split_meta']


def test_prohibition_end_to_end_presplit_shape():
  '''Temporal presplit plus sigmoid raises without opt-in.'''
  from forecast_ml_framework.pipelines.data_prep.nodes import (
    split_raw_data_with_meta,
  )
  from forecast_ml_framework.pipelines.modeling.adapters import (
    SklearnAdapter,
  )
  from forecast_ml_framework.pipelines.modeling.calibration import (
    calibrate_model,
  )
  from forecast_ml_framework.pipelines.modeling.pipeline import (
    split_presplit_data,
  )
  df = _daily_frame(100)
  params = _nested_params()
  train_raw, test_raw, meta_raw = split_raw_data_with_meta(
    df, params)
  cal_params = dict(params)
  cal_params['feature_cols'] = ['f1']
  x_tr, x_te, y_tr, y_te, split_meta = split_presplit_data(
    train_raw, test_raw, cal_params, meta_raw)
  assert split_meta['sampling_method'] == 'presplit'
  assert split_meta['raw_sampling_method'] == 'time'
  base = SklearnAdapter({}).fit(x_tr, y_tr)
  with pytest.raises(
      ValueError, match='calibrate_model prohibited with temporal split'):
    calibrate_model(base, x_tr, y_tr, params, split_meta)
  allowed = {
    'sampling_params': dict(params['sampling_params']),
    'calibration': {'method': 'sigmoid', 'cv': 2,
                    'allow_temporal': True},
    'target_col': 'target',
    'task_type': 'classification',
  }
  envelope, meta = calibrate_model(
    base, x_tr, y_tr, allowed, split_meta)
  assert meta['temporal_calibration_allowed'] is True
  assert type(meta['temporal_calibration_allowed']) is bool
  assert meta['calibrated'] is True


def test_modeling_yml_exact_paths():
  '''Modeling params hold the exact P3-D yaml shape.'''
  data = yaml.safe_load(
    (_root() / 'conf/base/parameters/modeling.yml').read_text())
  sampling = data['modeling_params']['sampling_params']
  assert sampling['method'] == 'stratified'
  assert sampling['time_col'] is None
  assert sampling['embargo'] == 0
  assert sampling['purge'] is None
  assert sampling['n_splits'] == 3
  assert sampling['window'] == 'expanding'
  calib = data['modeling_params']['calibration']
  assert calib['method'] == 'betacal'
  assert calib['cv'] == 5
  assert calibration_allow_temporal(data) is False
  assert data['modeling_params']['purged_cv'] == {
    'n_splits': 3,
    'test_size': 0.2,
    'embargo': 0,
    'purge': None,
    'window': 'expanding',
    'min_train_rows': 1,
  }
  assert data['modeling_params']['backtest'] == {
    'n_splits': 3,
    'test_size': 0.2,
    'embargo': 0,
    'purge': None,
    'window': 'expanding',
    'target_col': 'target',
    'task_type': 'classification',
    'feature_cols': None,
  }
  assert 'purged_cv' not in data
  assert 'backtest' not in data
  assert 'purged_cv' not in data['modeling_reg_params']
  assert 'backtest' not in data['modeling_reg_params']
  assert 'allow_temporal' not in data['modeling_reg_params'][
    'calibration']
  assert 'allow_temporal' not in data['calibration_params']


def calibration_allow_temporal(data):
  '''Return modeling calibration allow_temporal flag.'''
  return data['modeling_params']['calibration']['allow_temporal']


def test_no_second_sort_in_owned_files():
  '''Owned pipeline files hold no second time sort.'''
  nodes_path = (_root() / 'src/forecast_ml_framework/pipelines'
                '/data_prep/nodes.py')
  prep_path = (_root() / 'src/forecast_ml_framework/pipelines'
               '/data_prep/pipeline.py')
  model_path = (_root() / 'src/forecast_ml_framework/pipelines'
                '/modeling/pipeline.py')
  for path in (nodes_path, prep_path, model_path):
    assert 'sort_values' not in path.read_text()
