#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests for data-prep routine transformers and registry order.'''

def test_knn_imputation_fills_missing_values():
  '''KNN imputation fills missing values.'''
  import pandas as pd, numpy as np
  from forecast_ml_framework.pipelines.data_prep.nodes import preprocess
  df = pd.DataFrame({
    'f1': [1.0, 2.0, np.nan, 4.0],
    'f2': [1.0, 2.0, 3.0, 4.0],
    'target': [0, 1, 0, 1]})
  params = [{'KnnImputation': {'args': {
    'n_neighbors': 2, 'cols': {'include_cols': ['f1', 'f2']}}}}]
  out = preprocess(
    df, params, {'num_cols': ['f1', 'f2'], 'cat_cols': [], 'seq_cols': []})
  assert out['f1'].isna().sum() == 0
  assert abs(out['f1'].iloc[2] - 3.0) < 1.5


def test_sequence_change_ratio():
  '''Sequence change ratio computed over window.'''
  import pandas as pd
  from forecast_ml_framework.pipelines.data_prep.nodes import preprocess
  df = pd.DataFrame({
    'seq': [[10.0, 10.0, 20.0], [5.0, 5.0, 5.0]],
    'target': [0, 1]})
  params = [{'SequenceChangeRatio': {'args': {
    'last_n_val': 2, 'cols': {'include_cols': ['seq']}}}}]
  out = preprocess(
    df, params, {'num_cols': [], 'cat_cols': [], 'seq_cols': ['seq']})
  assert abs(out['seq_change_ratio'].iloc[0] - 2.0) < 1e-9
  assert abs(out['seq_change_ratio'].iloc[1] - 1.0) < 1e-9


def test_sequence_cov_zero_and_constant():
  '''Sequence cov is zero for constant sequences.'''
  import pandas as pd
  from forecast_ml_framework.pipelines.data_prep.nodes import preprocess
  df = pd.DataFrame({
    'seq': [[5.0, 5.0, 5.0], [0.0, 0.0, 0.0]],
    'target': [0, 1]})
  params = [{'SequenceCov': {'args': {
    'last_n_val': 3, 'cols': {'include_cols': ['seq']}}}}]
  out = preprocess(
    df, params, {'num_cols': [], 'cat_cols': [], 'seq_cols': ['seq']})
  assert out['seq_cov'].iloc[0] == 0.0
  assert out['seq_cov'].iloc[1] == 0.0


def test_existing_registry_unchanged_order():
  '''Existing registry order unchanged.'''
  from forecast_ml_framework.pipelines.data_prep.nodes import _REGISTRY
  assert list(_REGISTRY)[:16] == [
    'Imputations', 'CapOutliers', 'QuantileDiscretizer',
    'Normalizations', 'HighLabelBinning', 'CategoricalEncoding',
    'TargetEncoding', 'FixedLength', 'Slope', 'Similarity',
    'SequenceAverage', 'SequenceMin', 'SequenceMax', 'SequenceLast',
    'SequenceDelta', 'SequenceNormalize']
