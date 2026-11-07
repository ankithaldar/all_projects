#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Catalog gaps + sample data (clean-ux Task 5).'''

from pathlib import Path

import pandas as pd
import yaml


def _root():
  '''Return project root.'''
  return Path(__file__).resolve().parents[1]


def test_sample_csv_matches_synthetic_data():
  '''Sample CSV has 100 rows matching synthetic_data() contract.'''
  from forecast_ml_framework.pipelines.data_prep.nodes import (
    synthetic_data,
  )
  df = pd.read_csv(_root() / 'data/raw/raw_data.csv')
  ref = synthetic_data(n=100, task='classification', seed=42)
  assert list(df.columns) == list(ref.columns)
  assert list(df.columns) == [
    'id', 'f1', 'f2', 'f3', 'f4', 'cat', 'seq', 'target']
  assert len(df) == 100 and set(df['target']) <= {0, 1}
  assert set(df['cat']) <= {'a', 'b', 'c'}


def test_catalog_covers_pipeline_datasets():
  '''Every pipeline input/output has a typed catalog entry.'''
  catalog = yaml.safe_load(
    (_root() / 'conf/base/catalog.yml').read_text())
  for name in ['raw_data', 'validated_data', 'validation_report',
               'train_data_selected', 'modeling_input', 'x_train',
               'x_test', 'y_train', 'y_test', 'eda_summary',
               'eda_report', 'preprocessed_data', 'split_meta']:
    assert name in catalog, f'missing catalog entry: {name}'
    assert 'type' in catalog[name]


def _pipeline_datasets():
  '''Collect dataset names wired by every shipped pipeline.'''
  import importlib
  modules = [
    'forecast_ml_framework.pipelines.cost_tracking.pipeline',
    'forecast_ml_framework.pipelines.data_prep.pipeline',
    'forecast_ml_framework.pipelines.eda.pipeline',
    'forecast_ml_framework.pipelines.interpretability.pipeline',
    'forecast_ml_framework.pipelines.modeling.pipeline',
    'forecast_ml_framework.pipelines.monitoring.pipeline',
    'forecast_ml_framework.pipelines.retraining.pipeline',
    'forecast_ml_framework.pipelines.scoring.pipeline',
  ]
  names = set()
  for module_name in modules:
    module = importlib.import_module(module_name)
    pipeline = module.create_pipeline()
    for node in pipeline.nodes:
      for attr in ('inputs', 'outputs'):
        value = getattr(node, attr, None)
        if value is None:
          continue
        if isinstance(value, str):
          value = [value]
        for item in value:
          if isinstance(item, str):
            names.add(item)
  return names


def test_pipeline_datasets_subset_of_catalog():
  '''Every pipeline dataset is catalogued (V3).

  set(pipeline_datasets) ⊆ set(catalog) ∪ {params:, variables}.
  '''
  catalog = yaml.safe_load(
    (_root() / 'conf/base/catalog.yml').read_text())
  catalog_keys = {k for k in catalog if not k.startswith('_')}
  datasets = _pipeline_datasets()
  assert datasets, 'no pipeline datasets collected'
  non_param = {
    name for name in datasets
    if not name.startswith('params:') and name != 'variables'}
  missing = non_param - catalog_keys
  assert not missing, f'missing catalog entries: {sorted(missing)}'
  for name in sorted(non_param):
    assert 'type' in catalog[name], name


def _load_catalog(env):
  '''Load one env catalog.yml (base/local/gcp/aws).'''
  return yaml.safe_load(
    (_root() / 'conf' / env / 'catalog.yml').read_text())


def test_custom_dataset_types_resolve_and_anchors_merge():
  '''R8: custom *_dataset types resolve; anchors merge a type.

  Every catalog entry (base + overlays) whose ``type`` is a
  custom ``forecast_ml_framework.*DataSet`` class must name an
  importable class, and anchor-merged entries (``<<: *anchor``)
  must carry a ``type`` after YAML merge (the ``_`` anchor keys
  themselves are merge fragments, not datasets).
  '''
  import importlib
  checked = 0
  for env in ('base', 'local', 'gcp', 'aws'):
    catalog = _load_catalog(env)
    for name, entry in catalog.items():
      if name.startswith('_') or not isinstance(entry, dict):
        continue
      assert 'type' in entry, f'{env}:{name} has no type'
      dtype = entry['type']
      if isinstance(dtype, str) and dtype.startswith(
          'forecast_ml_framework.'):
        module_path, _, attr = dtype.rpartition('.')
        module = importlib.import_module(module_path)
        assert hasattr(module, attr), dtype
        checked += 1
  assert checked > 0, 'no custom dataset types found to probe'


def test_overlay_custom_dataset_overrides_keep_base_type():
  '''R8: env overlays override filepath only, never the type.

  Overlay entries that override a base entry must keep the
  base ``type`` (custom ``*_dataset`` overrides included), so
  an env cannot silently swap dataset semantics.
  '''
  base = _load_catalog('base')
  for env in ('local', 'gcp', 'aws'):
    overlay = _load_catalog(env)
    for name, entry in overlay.items():
      if name.startswith('_') or not isinstance(entry, dict):
        continue
      if name in base and isinstance(base[name], dict):
        assert entry.get('type') == base[name].get('type'), (
            f'{env}:{name} type drift: {entry.get('type')} vs '
            f'base {base[name].get('type')}')
