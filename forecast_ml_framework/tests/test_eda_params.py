#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests for EdaParams schema (eda_params typed section).'''

import pytest

pytest.importorskip('pydantic')
pytest.importorskip('omegaconf')

from forecast_ml_framework.core.config_schema import (  # noqa: E402
  EdaParams,
  ParametersConfig,
  load_config,
  validate_parameters,
)


def test_eda_params_defaults_match_eda_node_defaults():
  '''EdaParams defaults align with eda.yml + pipelines/eda/nodes.py.'''
  params = EdaParams()
  assert params.target_col == 'target'
  assert params.missing_threshold == 0.2
  assert params.cardinality_threshold == 50
  assert params.leakage_threshold == 0.95
  assert params.leakage_corr_threshold == 0.95
  assert params.duplicate_corr_threshold == 0.99
  assert params.leakage_auc_threshold == 0.99
  assert params.near_constant_threshold == 0.95
  assert params.leakage_name_keywords == [
    'target', 'churn', 'label', 'outcome', 'result', 'after', 'post']
  assert params.imbalance_ratio_threshold == 5.0
  assert params.zscore_threshold == 3.0
  assert params.top_correlations == 10
  assert params.top_values == 3
  # Invented fields from the old schema must be gone.
  for invented in ('max_categories', 'correlation_threshold',
                   'outlier_z', 'leakage_corr', 'top_k'):
    assert not hasattr(params, invented), invented
  cfg = ParametersConfig()
  assert isinstance(cfg.eda_params, EdaParams)
  assert cfg.eda_params.model_dump() == params.model_dump()


def test_eda_params_accepts_eda_yml_shape():
  '''ParametersConfig validates an eda_params mapping (eda.yml).

  Typed field, not an extra: known section lands in EdaParams
  and values round-trip through model_dump.
  '''
  cfg = validate_parameters({'eda_params': {
    'target_col': 'y',
    'missing_threshold': 0.1,
    'cardinality_threshold': 20,
    'leakage_threshold': 0.9,
    'leakage_corr_threshold': 0.9,
    'duplicate_corr_threshold': 0.98,
    'leakage_auc_threshold': 0.98,
    'near_constant_threshold': 0.9,
    'leakage_name_keywords': ['target'],
    'imbalance_ratio_threshold': 4.0,
    'zscore_threshold': 2.5,
    'top_correlations': 5,
    'top_values': 2,
  }})
  assert isinstance(cfg.eda_params, EdaParams)
  assert 'eda_params' not in (cfg.model_extra or {})
  assert cfg.eda_params.target_col == 'y'
  assert cfg.eda_params.missing_threshold == 0.1
  assert cfg.eda_params.cardinality_threshold == 20
  assert cfg.eda_params.leakage_threshold == 0.9
  assert cfg.eda_params.leakage_corr_threshold == 0.9
  assert cfg.eda_params.duplicate_corr_threshold == 0.98
  assert cfg.eda_params.leakage_auc_threshold == 0.98
  assert cfg.eda_params.near_constant_threshold == 0.9
  assert cfg.eda_params.leakage_name_keywords == ['target']
  assert cfg.eda_params.imbalance_ratio_threshold == 4.0
  assert cfg.eda_params.zscore_threshold == 2.5
  assert cfg.eda_params.top_correlations == 5
  assert cfg.eda_params.top_values == 2
  dumped = cfg.model_dump()
  assert dumped['eda_params']['zscore_threshold'] == 2.5
  assert dumped['eda_params']['top_correlations'] == 5


def test_load_config_base_eda_params_typed_not_extras():
  '''load_config('base') eda_params lands in typed EdaParams.'''
  plain = load_config('base')
  cfg = validate_parameters(plain['parameters'])
  assert isinstance(cfg.eda_params, EdaParams)
  assert 'eda_params' not in (cfg.model_extra or {})
  # Every shipped eda.yml key is a typed field, not an extra.
  shipped = plain['parameters']['eda_params']
  typed = set(EdaParams.model_fields)
  for key in shipped:
    assert key in typed, key
  dumped = cfg.model_dump()['eda_params']
  for key, value in shipped.items():
    assert dumped[key] == value, key
