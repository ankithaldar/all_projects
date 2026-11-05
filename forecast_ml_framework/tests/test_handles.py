#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Shipped parameters.yml handles resolve to importable callables.'''

import importlib.util
from pathlib import Path

import pytest
import yaml

from forecast_ml_framework.pipelines.data_prep.nodes import resolve_handle

HANDLE_KEYS = {
  'handle',
  'model_handle',
  'calibration_handle',
  'tuning_handle',
  'label_encode_handle',
  'ohe_handle',
}


def _load_shipped_params():
  '''Load merged shipped parameters from parameters/*.yml.'''
  root = Path(__file__).resolve().parents[1]
  merged = {}
  for path in sorted((root / 'conf' / 'base' / 'parameters').glob('*.yml')):
    with path.open() as handle:
      merged.update(yaml.safe_load(handle) or {})
  assert merged, 'no shipped parameters found'
  return merged


def _collect_handles(node, found=None):
  '''Collect every *_handle/handle string in a nested config.'''
  if found is None:
    found = []
  if isinstance(node, dict):
    for key, value in node.items():
      if key in HANDLE_KEYS and isinstance(value, str):
        found.append(value)
      _collect_handles(value, found)
  elif isinstance(node, list):
    for item in node:
      _collect_handles(item, found)
  return found


def test_shipped_handles_resolve():
  '''Every shipped handle resolves (skip-unless-available).'''
  params = _load_shipped_params()
  handles = sorted(set(_collect_handles(params)))
  assert handles, 'no handles found in shipped parameters.yml'
  resolved, skipped = [], []
  for handle in handles:
    module = handle.rsplit('.', 1)[0]
    try:
      available = importlib.util.find_spec(module) is not None
    except (ImportError, AttributeError, ValueError):
      available = False
    if not available:
      skipped.append(handle)  # skip-unless-available, per handle
      continue
    obj = resolve_handle(handle)
    assert callable(obj), handle
    resolved.append(handle)
  assert resolved, 'no shipped handle resolved: %s' % skipped


def test_expected_handles_present():
  '''Key vz-style handles are present in shipped config.'''
  params = _load_shipped_params()
  handles = set(_collect_handles(params))
  assert 'xgboost.XGBClassifier' in handles
  assert 'xgboost.XGBRegressor' in handles
  assert 'sklearn.ensemble.RandomForestClassifier' in handles
  assert 'sklearn.calibration.CalibratedClassifierCV' in handles
  assert 'sklearn.model_selection.GridSearchCV' in handles
  assert 'pyspark.ml.feature.StringIndexer' in handles
  assert 'pyspark.ml.feature.OneHotEncoder' in handles
  assert 'pyspark.ml.feature.QuantileDiscretizer' in handles


def test_fe_steps_with_real_handles():
  '''FE steps mapping to real classes carry a handle.'''
  params = _load_shipped_params()
  steps = {}
  for item in params['data_prep_params']:
    for name, cfg in item.items():
      steps[name] = cfg.get('args', {}) or {}
  assert '.' in steps['Imputations'].get('handle', '')
  assert '.' in steps['QuantileDiscretizer'].get('handle', '')
  assert '.' in steps['Normalizations'].get('handle', '')
  assert '.' in steps['CategoricalEncoding'].get(
    'label_encode_handle', '')
  assert '.' in steps['CategoricalEncoding'].get('ohe_handle', '')
  assert '.' in steps['TargetEncoding'].get('handle', '')
