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
