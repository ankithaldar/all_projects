#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Two-stage scoring DAG (vz-high Task 9).'''

import pandas as pd


def test_scoring_pipeline_has_score_data_prep():
  '''Scoring pipeline has score_data_prep node.'''
  from forecast_ml_framework.pipelines.scoring.pipeline import (
    create_pipeline,
  )
  pipe = create_pipeline()
  names = [n.name for n in pipe.nodes]
  assert names == ['score_data_prep', 'predict', 'write_predictions']


def test_catalog_has_fitted_registry():
  '''Catalog has fitted_registry entry.'''
  import yaml

  with open('conf/base/catalog.yml', encoding='utf-8') as fh:
    catalog = yaml.safe_load(fh)
  assert 'fitted_registry' in catalog
  assert catalog['fitted_registry']['type'] == (
      'kedro_datasets.json.JSONDataset')
  assert 'score_prep_params' in catalog
  assert catalog['score_prep_params']['type'] == 'kedro.io.MemoryDataset'


def test_score_data_prep_empty_registry_is_identity():
  '''Empty registry score_data_prep is identity.'''
  from forecast_ml_framework.pipelines.scoring.pipeline import (
    score_data_prep,
  )
  df = pd.DataFrame({'f1': [1.0, 2.0]})
  pd.testing.assert_frame_equal(score_data_prep(df, {}), df)
  pd.testing.assert_frame_equal(score_data_prep(df, None), df)
