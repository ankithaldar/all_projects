#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Parity Task 9: catalog anchors + eval_metric config.'''

from pathlib import Path

import yaml

from forecast_ml_framework.core.config_schema import (
  ModelingParams,
  validate_parameters,
)

_ROOT = Path(__file__).resolve().parents[1]


def _catalog():
  '''Load and return the parsed catalog.yml.'''
  with open(_ROOT / 'conf/base/catalog.yml') as handle:
    return yaml.safe_load(handle)


def test_classification_decile_anchor_table():
  '''Classification anchor points at classification decile metrics table.'''
  catalog = _catalog()
  assert catalog['_classification_model_decile_metrics'][
      'table_name'] == 'classification_model_decile_metrics'


def test_classification_lift_params_train_valid_test():
  '''Train/valid/test lift params share the classification anchor.'''
  catalog = _catalog()
  for name in (
      'train_model_lift_params',
      'valid_model_lift_params',
      'test_model_lift_params',
  ):
    assert catalog[name]['table_name'] == (
      'classification_model_decile_metrics')
    assert catalog[name]['type'] == (
      'forecast_ml_framework.core.datasets.SQLQueryDataSet')


def test_regression_anchors_and_registry_model_unchanged():
  '''Regression anchors and registry_model are untouched.'''
  catalog = _catalog()
  assert catalog['_regression_model_decile_metrics'][
      'table_name'] == 'regression_model_decile_metrics'
  assert catalog['train_model_lift_reg_params']['table_name'] == (
    'regression_model_decile_metrics')
  assert catalog['test_model_lift_reg_params']['table_name'] == (
    'regression_model_decile_metrics')
  assert catalog['registry_model']['filepath'] == (
    'artifacts/models/registry_model.pkl')


def test_modeling_params_accepts_eval_metric():
  '''ModelingParams accepts an optional top-level eval_metric.'''
  params = ModelingParams.model_validate(
    {'eval_metric': 'LIFT_DECILE_10'})
  assert params.eval_metric == 'LIFT_DECILE_10'
  default = ModelingParams.model_validate({})
  assert default.eval_metric is None
  validated = validate_parameters(
    {'modeling_params': {'eval_metric': 'LIFT_CENTILE_100'}})
  assert validated.modeling_params.eval_metric == 'LIFT_CENTILE_100'


def test_modeling_params_old_score_explicit():
  '''ModelingParams declares old_score explicitly, default None.'''
  assert 'old_score' in ModelingParams.model_fields
  default = ModelingParams.model_validate({})
  assert default.old_score is None
  params = ModelingParams.model_validate({'old_score': 0.97})
  assert params.old_score == 0.97
  validated = validate_parameters(
    {'modeling_params': {'old_score': 1.25}})
  assert validated.modeling_params.old_score == 1.25


def test_modeling_pipeline_outputs_in_catalog():
  '''Every modeling pipeline output has a catalog entry.'''
  from forecast_ml_framework.pipelines.modeling.pipeline import (
    create_pipeline,
  )
  catalog = _catalog()
  outputs = set()
  for node in create_pipeline().nodes:
    outputs.update(node.outputs)
  assert outputs, 'pipeline must declare outputs'
  missing = sorted(name for name in outputs if name not in catalog)
  assert not missing, f'missing catalog entries: {missing}'


def test_modeling_pipeline_output_types_are_writable():
  '''Modeling outputs reuse writable dataset types only.'''
  from forecast_ml_framework.pipelines.modeling.pipeline import (
    create_pipeline,
  )
  catalog = _catalog()
  outputs = set()
  for node in create_pipeline().nodes:
    outputs.update(node.outputs)
  allowed_prefixes = (
    'kedro_datasets.pandas.ParquetDataset',
    'kedro_datasets.pandas.CSVDataset',
    'kedro_datasets.json.JSONDataset',
    'kedro_datasets.spark.SparkDataset',
    'forecast_ml_framework.core.datasets.RestrictedPickleDataset',
    'kedro.io.MemoryDataset',
  )
  for name in sorted(outputs):
    assert name in catalog, name
    dataset_type = catalog[name].get('type', '')
    assert dataset_type.startswith(allowed_prefixes), (name, dataset_type)


def test_modeling_yml_commented_eval_metric_examples():
  '''modeling.yml documents eval_metric separately from xgboost.'''
  text = (_ROOT / 'conf/base/parameters/modeling.yml').read_text()
  assert 'LIFT_DECILE_10' in text
  assert 'LIFT_CENTILE_100' in text
  assert 'RMSE_CENTILE_100' in text
  assert 'eval_metric: \'aucpr\'' in text
  lines = [line for line in text.splitlines() if 'LIFT_DECILE_10' in line]
  assert any(line.strip().startswith('#') for line in lines)
