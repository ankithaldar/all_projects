#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Base catalog.yml aligns to vz anchors/*_params patterns.'''

from pathlib import Path

import yaml


def _root():
  '''Return project root.'''
  return Path(__file__).resolve().parents[1]


def _load_catalog(env='base'):
  '''Load a catalog.yml for an env.'''
  with (_root() / 'conf' / env / 'catalog.yml').open() as handle:
    return yaml.safe_load(handle)


def _raw(env='base'):
  '''Return raw catalog text for an env.'''
  return (_root() / 'conf' / env / 'catalog.yml').read_text()


def test_anchors_present():
  '''YAML anchors from the vz pattern are present.'''
  raw = _raw()
  for anchor in ('&spark_bq_data', '&save_parquet', '&save_json',
                 '&regression_model_overall_metrics',
                 '&regression_model_decile_metrics'):
    assert anchor in raw, anchor


def test_spark_bq_input_params():
  '''train/test/score inputs merge the Spark BQ anchor + SQL.'''
  catalog = _load_catalog()
  for name in ('train_input_params', 'test_input_params',
               'score_input_params'):
    entry = catalog[name]
    assert entry['type'] == (
        'forecast_ml_framework.core.datasets.SparkSQLDataSet')
    assert 'sql' in entry and '`' in entry['sql']
    assert '${params:' in entry['sql']
    assert entry['write_mode'] == 'overwrite'


def test_parquet_and_pickle_params():
  '''Parquet/pickle *_params entries use the save anchors.'''
  catalog = _load_catalog()
  for name in ('sampled_df_params', 'score_df_params',
               'train_pp_data_df_params'):
    entry = catalog[name]
    assert entry['type'] == 'kedro_datasets.spark.SparkDataset'
    assert entry['filepath'].startswith('${params:output_gcs_location}')
  for name in ('fitted_model_params', 'calibrated_model_params'):
    entry = catalog[name]
    assert entry['type'] == 'kedro_datasets.pickle.PickleDataset'
  for name in ('train_prediction_df_params', 'prediction_df_params'):
    assert catalog[name]['filepath'].startswith(
        '${params:output_gcs_location}')


def test_bq_metrics_and_decile_anchor():
  '''BQ metrics entries carry partition + shared decile anchor.'''
  catalog = _load_catalog()
  for name in ('train_model_evaluation_params',
               'test_model_evaluation_params'):
    entry = catalog[name]
    assert entry['project'] and entry['dataset']
    assert entry['partition_field'] == 'run_date'
    assert entry['partition_type'] == 'DAY'
    assert 'write_disposition' in entry['save_args']
  decile = catalog['train_model_lift_reg_params']
  assert decile['table_name'] == 'faas_regression_model_decile_metrics'
  assert catalog['valid_model_lift_reg_params'] == decile
  assert catalog['test_model_lift_reg_params'] == decile
  overall = catalog['train_model_evaluation_reg_params']
  assert catalog['test_model_evaluation_reg_params'] == overall


def test_existing_generic_datasets_kept():
  '''Pre-existing generic datasets are not dropped.'''
  catalog = _load_catalog()
  for name in ('raw_data', 'processed_data', 'train_data', 'test_data',
               'selected_features', 'model', 'shap_values', 'charts',
               'metrics', 'predictions'):
    assert name in catalog, name


def test_params_namespace_only():
  '''Interpolations use the params: namespace, never runtime_params:.'''
  raw = _raw()
  assert '${params:' in raw
  assert '${runtime_params:' not in raw


def test_env_overlays_parse():
  '''Env overlay catalogs still parse as YAML.'''
  for env in ('local', 'gcp', 'aws'):
    assert _load_catalog(env)


def test_every_type_importable():
  '''Every catalog type string imports (skip-unless-available).'''
  import importlib
  import importlib.util

  catalog = _load_catalog()
  checked = 0
  for name, entry in catalog.items():
    if not isinstance(entry, dict) or 'type' not in entry:
      continue
    path = entry['type']
    module, _, attr = path.rpartition('.')
    try:
      available = importlib.util.find_spec(module) is not None
    except (ImportError, AttributeError, ValueError):
      available = False
    if not available:
      continue  # e.g. kedro_datasets/spark absent in some envs
    try:
      obj = getattr(importlib.import_module(module), attr)
    except ImportError:
      continue  # optional dep of the dataset (e.g. hdfs for spark)
    assert callable(obj), (name, path)
    checked += 1
  assert checked > 0
