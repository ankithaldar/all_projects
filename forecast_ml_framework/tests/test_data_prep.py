#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests for data_prep nodes.'''

import numpy as np
import pandas as pd
import pytest

from forecast_ml_framework.pipelines.data_prep.nodes import (
  preprocess,
  resolve_handle,
  select_features,
  synthetic_data,
  validate_data,
)
from forecast_ml_framework.pipelines.data_prep.pipeline import (
  create_pipeline,
)


def _vars():
  '''Return variables dict.'''
  return {
    'target_col': 'target',
    'id_col': 'id',
    'feature_cols': ['x1', 'x2', 'cat', 'seq'],
    'cat_cols': ['cat'],
    'num_cols': ['x1', 'x2'],
    'seq_cols': ['seq'],
  }


def _df():
  '''Return small dataframe.'''
  return pd.DataFrame({
    'id': [1, 2, 3, 4],
    'x1': [0.0, 10.0, 20.0, None],
    'x2': [1.0, 1.0, 1.0, 1.0],
    'cat': ['a', 'b', 'a', 'b'],
    'seq': [[1.0, 2.0, 3.0]] * 4,
    'target': [0, 1, 0, 1],
  })


def test_validate_passes_and_reports():
  '''Validate returns report with null rates.'''
  df, rep = validate_data(_df(), {}, _vars())
  assert rep['n_rows'] == 4
  assert rep['null_rates']['x1'] == pytest.approx(0.25)
  assert rep['target_distribution']


def test_validate_missing_col_fail_fast():
  '''Validate raises on missing col when fail_fast.'''
  df = _df().drop(columns=['x1'])
  with pytest.raises(ValueError):
    validate_data(df, {'fail_fast': True}, _vars())


def test_validate_missing_col_no_fail_fast():
  '''Validate logs missing col when not fail_fast.'''
  df = _df().drop(columns=['x1'])
  _, rep = validate_data(df, {'fail_fast': False}, _vars())
  assert 'x1' in rep['missing_cols']


def test_preprocess_skip_respected():
  '''Preprocess respects skip true.'''
  out = preprocess(
    _df(),
    [{'Imputations': {'skip': True, 'args': {}}}],
    _vars(),
  )
  assert out['x1'].isna().sum() == 1


def test_preprocess_imputations_mean():
  '''Imputations fills numeric mean.'''
  steps = [{'Imputations': {
    'skip': False,
    'args': {'strategy': 'mean', 'cols': {
      'include_cols': ['x1'], 'exclude_cols': []}},
  }}]
  out = preprocess(_df(), steps, _vars())
  assert out['x1'].isna().sum() == 0
  assert out['x1'].iloc[3] == pytest.approx(10.0)


def test_preprocess_cap_outliers_and_normalize():
  '''Cap and normalize apply.'''
  steps = [
    {'CapOutliers': {'skip': False, 'args': {
      'method': 'bound', 'bound': {'lower': 0.25, 'upper': 0.75},
      'cols': {'include_cols': ['x1'], 'exclude_cols': []}}}},
    {'Normalizations': {'skip': False, 'args': {
      'handle': 'MinMaxScaler',
      'cols': {'include_cols': ['x1'], 'exclude_cols': []}}}},
  ]
  out = preprocess(_df(), steps, _vars())
  assert out['x1'].max() <= 1.0 + 1e-9
  assert out['x1'].min() >= -1e-9


def test_preprocess_encoding_and_binning():
  '''Encoding, binning, discretizer apply.'''
  steps = [
    {'HighLabelBinning': {'skip': False, 'args': {
      'val_count': 2,
      'cols': {'include_cols': ['cat'], 'exclude_cols': []}}}},
    {'CategoricalEncoding': {'skip': False, 'args': {
      'ohe_flag': True,
      'cols': {'include_cols': ['cat'], 'exclude_cols': []}}}},
    {'QuantileDiscretizer': {'skip': False, 'args': {
      'params': {'numBuckets': 2},
      'cols': {'include_cols': ['x1'], 'exclude_cols': []}}}},
  ]
  out = preprocess(_df(), steps, _vars())
  assert 'cat_a' in out.columns or 'cat' in out.columns
  assert out['x1'].nunique() <= 2


def test_preprocess_target_encoding():
  '''Target encoding maps category to mean.'''
  steps = [{'TargetEncoding': {'skip': False, 'args': {
    'cols': {'include_cols': ['cat'], 'exclude_cols': []}}}}]
  out = preprocess(_df(), steps, _vars())
  assert out.loc[0, 'cat'] == pytest.approx(0.0)
  assert out.loc[1, 'cat'] == pytest.approx(1.0)


def test_preprocess_sequence_steps():
  '''Sequence steps create columns.'''
  steps = [
    {'SequenceAverage': {'skip': False, 'args': {
      'last_n_val': 2,
      'cols': {'include_cols': ['seq'], 'exclude_cols': []}}}},
    {'SequenceMin': {'skip': False, 'args': {
      'last_n_val': 2,
      'cols': {'include_cols': ['seq'], 'exclude_cols': []}}}},
    {'SequenceMax': {'skip': False, 'args': {
      'last_n_val': 2,
      'cols': {'include_cols': ['seq'], 'exclude_cols': []}}}},
    {'SequenceLast': {'skip': False, 'args': {
      'cols': {'include_cols': ['seq'], 'exclude_cols': []}}}},
    {'SequenceDelta': {'skip': False, 'args': {
      'last_n_val': 2,
      'cols': {'include_cols': ['seq'], 'exclude_cols': []}}}},
    {'Slope': {'skip': False, 'args': {
      'last_n_val': 2, 'suffix': 'slope',
      'cols': {'include_cols': ['seq'], 'exclude_cols': []}}}},
    {'Similarity': {'skip': False, 'args': {
      'last_n_val': 2,
      'cols': {'include_cols': ['seq'], 'exclude_cols': []}}}},
    {'FixedLength': {'skip': False, 'args': {
      'fixed_length': 2,
      'cols': {'include_cols': ['seq'], 'exclude_cols': []}}}},
    {'SequenceNormalize': {'skip': False, 'args': {
      'round_off': 3,
      'cols': {'include_cols': ['seq'], 'exclude_cols': []}}}},
  ]
  out = preprocess(_df(), steps, _vars())
  assert out['seq_avg'].iloc[0] == pytest.approx(2.5)
  assert out['seq_last'].iloc[0] == pytest.approx(3.0)
  assert out['seq_delta'].iloc[0] == pytest.approx(1.0)
  assert 'seq_slope' in out.columns
  assert 'seq_similarity' in out.columns


def test_select_override_business_exclude():
  '''Select respects core override/business/exclude rule.'''
  df = synthetic_data(120, 'classification')
  params = {
    'override_features': ['f1', 'f2'],
    'business_relevant_features': ['f3'],
    'exclude_features': ['f2'],
  }
  out, sel, rep = select_features(df, params, {
    'target_col': 'target', 'feature_cols': [
      'f1', 'f2', 'f3', 'f4']})
  # Core rule: override wins alone, exclude not applied.
  assert sel == ['f1', 'f2']
  assert 'f1' in out.columns
  params2 = {
    'business_relevant_features': ['f3'],
    'exclude_features': ['f2'],
  }
  _, sel2, _ = select_features(df, params2, {
    'target_col': 'target', 'feature_cols': [
      'f1', 'f2', 'f3', 'f4']})
  assert 'f3' in sel2 and 'f2' not in sel2


def test_select_variance_and_correlation():
  '''Variance and correlation pruning apply.'''
  df = synthetic_data(120, 'regression')
  df['f_dup'] = df['f1']
  df['f_const'] = 1.0
  params = {
    'variance_threshold': 0.0,
    'correlation': {'enabled': True, 'threshold': 0.99},
  }
  _, sel, _ = select_features(df, params, {
    'target_col': 'target', 'feature_cols': [
      'f1', 'f_dup', 'f_const', 'f2']})
  assert 'f_const' not in sel
  assert not ('f1' in sel and 'f_dup' in sel)


def test_select_kbest_l1_tree():
  '''KBest, L1, tree selection run.'''
  df = synthetic_data(120, 'classification')
  params = {
    'kbest': {'enabled': True, 'k': 2,
              'score_func': 'f_classif'},
    'l1': {'enabled': False},
    'tree': {'enabled': False},
  }
  _, sel, rep = select_features(df, params, {
    'target_col': 'target', 'feature_cols': ['f1', 'f2', 'f3']})
  assert len(sel) == 2
  assert 'kbest' in rep


def test_synthetic_data_shapes():
  '''Synthetic data shapes and tasks.'''
  clf = synthetic_data(50, 'classification')
  reg = synthetic_data(50, 'regression')
  assert clf.shape[0] == 50 and reg.shape[0] == 50
  assert set(clf['target'].unique()) <= {0, 1}
  assert np.issubdtype(reg['target'].dtype, np.number)


def test_create_pipeline_nodes():
  '''Pipeline has at least three nodes.'''
  pipe = create_pipeline()
  assert len(pipe.nodes) >= 3
  names = [n.name for n in pipe.nodes]
  assert any('valid' in n or 'ingest' in n for n in names)
  assert any('preprocess' in n for n in names)
  assert any('select' in n for n in names)


def _load_shipped_params():
  '''Load merged shipped parameters from parameters/*.yml.'''
  from pathlib import Path

  import yaml
  root = Path(__file__).resolve().parents[1]
  merged = {}
  for path in sorted((root / 'conf' / 'base' / 'parameters').glob('*.yml')):
    with path.open() as handle:
      merged.update(yaml.safe_load(handle) or {})
  assert merged, 'no shipped parameters found'
  return merged


def test_shipped_config_drives_selection():
  '''Shipped feature_selection_params drives select_features.'''
  params = _load_shipped_params()
  fparams = params['feature_selection_params']
  assert 'variance_threshold' in fparams
  assert 'missing_threshold' in fparams
  assert 'correlation' in fparams
  assert 'kbest' in fparams
  assert 'l1' in fparams
  assert 'tree' in fparams
  df = synthetic_data(120, 'classification')
  df['f_const'] = 1.0
  _, sel, _ = select_features(df, fparams, {
    'target_col': 'target', 'task': 'classification',
    'feature_cols': ['f1', 'f2', 'f_const']})
  assert 'f_const' not in sel


def test_shipped_config_drives_validation():
  '''Shipped validation_params drives validate_data.'''
  params = _load_shipped_params()
  vparams = params['validation_params']
  assert 'fail_fast' in vparams
  assert 'required_cols' in vparams
  assert 'max_null_ratio' in vparams
  df = _df()
  df['x1'] = None
  _, rep = validate_data(df, vparams, _vars())
  assert 'high_null_cols' in rep
  assert 'x1' in rep['high_null_cols']


def test_feature_rule_matches_config_loader():
  '''select_features rule matches ConfigLoader merged list.'''
  from forecast_ml_framework.core.config_loader import ConfigLoader
  df = synthetic_data(120, 'classification')
  variables = {'target_col': 'target',
               'feature_cols': ['f1', 'f2', 'f3', 'f4']}
  # Core alias 'feature_override' must be honoured via ConfigLoader.
  params = {'feature_override': ['f2', 'f2', 'f1']}
  _, sel, _ = select_features(df, params, variables)
  expected = ConfigLoader(
    variables=variables, parameters=params).merged_feature_list(
      selected=['f1', 'f2', 'f3', 'f4'])
  assert sel == expected
  # Business + exclude path also matches core.
  params2 = {'business_features': ['f3'],
             'exclude_features': ['f2']}
  _, sel2, _ = select_features(df, params2, variables)
  expected2 = ConfigLoader(
    variables=variables, parameters=params2).merged_feature_list(
      selected=['f1', 'f2', 'f3', 'f4'])
  assert sel2 == expected2


def test_unknown_step_in_report():
  '''Unknown preprocessing step appears in report.'''
  steps = [{'NopeStep': {'skip': False, 'args': {}}}]
  out, rep = preprocess(_df(), steps, _vars(), return_report=True)
  assert out['x1'].isna().sum() == 1
  assert 'NopeStep' in rep.get('unknown_steps', [])
  assert any('NopeStep' in w for w in rep.get('warnings', []))


def test_validate_warns_single_class_and_all_null():
  '''Validate warns on single-class target and all-null col.'''
  df = _df()
  df['target'] = 0
  df['x2'] = None
  _, rep = validate_data(df, {}, _vars())
  warnings = rep.get('warnings', [])
  assert any('single' in w.lower() for w in warnings)
  assert any('x2' in w for w in warnings)


def test_select_warns_nonexistent_override():
  '''Select warns on override cols missing from frame.'''
  df = synthetic_data(120, 'classification')
  params = {'override_features': ['ghost_col']}
  _, sel, rep = select_features(df, params, {
    'target_col': 'target', 'feature_cols': ['f1', 'f2']})
  assert sel == []
  assert any('ghost_col' in w for w in rep.get('warnings', []))


class _FakeSparkCountOnly:
  '''Fake Spark frame reporting rows via count() only.'''

  def __init__(self, rows):
    self._rows = rows
    self.columns = ['id', 'x1', 'x2', 'cat', 'seq', 'target']
    self.to_pandas_calls = 0

  def count(self):
    '''Return row count without materialising data.'''
    return self._rows

  def toPandas(self):
    '''Record call; the guard must prevent this.'''
    self.to_pandas_calls += 1
    raise AssertionError('toPandas must not be called')


def test_preprocess_fake_spark_over_5m_raises():
  '''Preprocess enforces default pandas guard on fake Spark >5M rows.'''
  spark_df = _FakeSparkCountOnly(5_000_001)
  with pytest.raises(ValueError, match='max_rows_for_pandas'):
    preprocess(spark_df, [], _vars())
  assert spark_df.to_pandas_calls == 0


def test_preprocess_fake_spark_over_configured_limit_raises():
  '''Preprocess honours pandas_conversion_params limit.'''
  spark_df = _FakeSparkCountOnly(100)
  with pytest.raises(ValueError, match='max_rows_for_pandas'):
    preprocess(spark_df, [], _vars(),
               pandas_conversion_params={'max_rows_for_pandas': 10})
  assert spark_df.to_pandas_calls == 0


@pytest.mark.parametrize('handle', [
  'os.system', 'subprocess.Popen', 'builtins.eval', 'builtins.exec',
])
def test_resolve_handle_rejects_non_allowlisted(handle):
  '''resolve_handle rejects handles outside the allowlist.'''
  with pytest.raises(ValueError):
    resolve_handle(handle)


def test_resolve_handle_rejects_non_estimator_allowlisted():
  '''Allowlisted non-estimator (function) raises ValueError.'''
  with pytest.raises(ValueError):
    resolve_handle('sklearn.datasets.load_iris')


def test_resolve_handle_accepts_allowlisted_estimator():
  '''Allowlisted estimator class resolves.'''
  obj = resolve_handle('sklearn.ensemble.RandomForestClassifier')
  assert obj.__name__ == 'RandomForestClassifier'
  assert hasattr(obj, 'fit') and hasattr(obj, 'predict')

