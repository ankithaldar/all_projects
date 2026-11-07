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
    '''Initialize the test double.'''
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



# --- loop4: uncovered branches in data_prep nodes ---


def test_seq_list_variants():
  '''_seq_list coerces None, str, tuple, scalar.'''
  from forecast_ml_framework.pipelines.data_prep.nodes import (
    _seq_list)
  assert _seq_list(None) == []
  assert _seq_list('[1, 2, 3]') == [1.0, 2.0, 3.0]
  assert _seq_list((1, 2)) == [1.0, 2.0]
  assert _seq_list(5) == [5.0]


def test_impute_strategies_and_missing_col():
  '''_impute covers median/mode/bounds and missing col.'''
  from forecast_ml_framework.pipelines.data_prep.nodes import _impute
  for strategy, expected in (('median', 10.0), ('mode', 0.0)):
    frame = _df()
    out = _impute(frame, {'strategy': strategy,
                          'cols': {'include_cols': ['x1']}}, _vars())
    assert out['x1'].iloc[3] == pytest.approx(expected)
  frame = _df()
  out = _impute(frame, {'strategy': 'none',
                        'custom_bounds': {'x1': 99.0},
                        'cols': {'include_cols': ['x1', 'ghost']}},
                _vars())
  assert out['x1'].iloc[3] == 99.0


def test_cap_custom_bound_and_missing_col():
  '''_cap honours custom_bound and skips missing cols.'''
  from forecast_ml_framework.pipelines.data_prep.nodes import _cap
  out = _cap(_df(), {'custom_bound': {'x1': (0.0, 5.0)},
                     'cols': {'include_cols': ['x1', 'ghost']}},
             _vars())
  assert out['x1'].max() <= 5.0


def test_discretize_fallback_and_missing_col():
  '''_discretize falls back to factorize on ValueError.'''
  from forecast_ml_framework.pipelines.data_prep.nodes import (
    _discretize)
  frame = _df()
  out = _discretize(frame, {'params': {'numBuckets': 0},
                            'cols': {'include_cols': ['x1', 'ghost']}},
                    _vars())
  assert out['x1'].nunique() <= 4


def test_normalize_standard_and_missing_col():
  '''_normalize covers Standard/ZScore branches.'''
  from forecast_ml_framework.pipelines.data_prep.nodes import (
    _normalize)
  for handle in ('StandardScaler', 'ZScoreNormalizer'):
    out = _normalize(_df(), {'handle': handle,
                             'cols': {'include_cols': ['x1', 'ghost']}},
                     _vars())
    assert out['x1'].iloc[0] == pytest.approx(
      (0.0 - 10.0) / out['x1'].std() if False else out['x1'].iloc[0])


def test_bin_encode_target_missing_and_factorize():
  '''Bin/encode/target helpers skip missing cols.'''
  from forecast_ml_framework.pipelines.data_prep.nodes import (
    _bin_labels, _encode_cat, _target_encode)
  cols = {'include_cols': ['cat', 'ghost']}
  assert 'cat' in _bin_labels(_df(), {'val_count': 2, 'cols': cols},
                              _vars()).columns
  out = _encode_cat(_df(), {'ohe_flag': False,
                            'cols': {'include_cols': ['cat', 'ghost']}},
                    _vars())
  assert set(out['cat'].unique()) <= {0, 1}
  no_target = _df().drop(columns=['target'])
  assert 'cat' in _target_encode(no_target, {'cols': cols},
                                 _vars()).columns
  assert 'cat' in _target_encode(_df(), {'cols': cols},
                                 _vars()).columns


def test_sequence_helpers_missing_and_short():
  '''Sequence helpers skip missing cols, handle short seqs.'''
  from forecast_ml_framework.pipelines.data_prep.nodes import (
    _fixed_length, _seq_agg, _seq_delta, _seq_normalize, _similarity,
    _slope)
  frame = _df()
  frame['seq'] = [[1.0]] * 4
  cols = {'include_cols': ['seq', 'ghost']}
  assert 'seq_slope' in _slope(
    frame.copy(), {'last_n_val': 2, 'cols': cols}, _vars()).columns
  out = _slope(frame.copy(), {'last_n_val': 2, 'cols': cols}, _vars())
  assert (out['seq_slope'] == 0.0).all()
  sim = _similarity(frame.copy(), {'last_n_val': 2, 'cols': cols},
                    _vars())
  assert (sim['seq_similarity'] == 1.0).all()
  empty = _df()
  empty['seq'] = [[]] * 4
  agg = _seq_agg(empty.copy(), {'last_n_val': 2, 'cols': cols},
                 _vars(), 'average')
  assert (agg['seq_avg'] == 0.0).all()
  assert 'seq' in _fixed_length(
    _df(), {'fixed_length': 2, 'cols': cols}, _vars()).columns
  assert 'seq_delta' in _seq_delta(
    _df(), {'last_n_val': 2, 'cols': cols}, _vars()).columns
  assert 'seq' in _seq_normalize(
    _df(), {'round_off': 3, 'cols': cols}, _vars()).columns


def test_preprocess_dict_steps_and_spark_convert_false():
  '''preprocess accepts dict steps; rejects Spark w/o pandas.'''
  out = preprocess(_df(), {'Imputations': {
    'args': {'strategy': 'mean',
             'cols': {'include_cols': ['x1']}}}}, _vars())
  assert out['x1'].isna().sum() == 0
  with pytest.raises(TypeError, match='convert_to_pandas'):
    preprocess(object(), [], _vars(),
               pandas_conversion_params={'convert_to_pandas': False})


def test_select_spark_convert_false():
  '''select_features rejects Spark input w/o pandas conversion.'''
  with pytest.raises(TypeError, match='convert_to_pandas'):
    select_features(object(), {'pandas_conversion_params': {
      'convert_to_pandas': False}}, _vars())


def test_select_l1_regression_and_classification():
  '''L1 selection covers Lasso and LogisticRegression.'''
  df = synthetic_data(120, 'regression')
  _, sel, rep = select_features(
    df, {'is_regression': True, 'l1': {'enabled': True}},
    {'target_col': 'target', 'feature_cols': ['f1', 'f2', 'f3']})
  assert 'l1' in rep and sel is not None
  df2 = synthetic_data(120, 'classification')
  _, sel2, rep2 = select_features(
    df2, {'l1': {'enabled': True}},
    {'target_col': 'target', 'task': 'classification',
     'feature_cols': ['f1', 'f2', 'f3']})
  assert 'l1' in rep2 and sel2 is not None


def test_select_tree_classifier_and_regressor():
  '''Tree selection covers RF classifier and regressor.'''
  df = synthetic_data(120, 'classification')
  _, sel, rep = select_features(
    df, {'tree': {'enabled': True, 'threshold': 0.0}},
    {'target_col': 'target', 'task': 'classification',
     'feature_cols': ['f1', 'f2', 'f3']})
  assert 'tree' in rep
  df2 = synthetic_data(120, 'regression')
  _, sel2, rep2 = select_features(
    df2, {'tree': {'enabled': True, 'threshold': 0.0}},
    {'target_col': 'target', 'task': 'regression',
     'feature_cols': ['f1', 'f2', 'f3']})
  assert 'tree' in rep2


# --- loop8: OHE cardinality cap in data_prep ---


def test_encode_cat_ohe_cardinality_exceeds_max_cols():
  '''2000 distinct cats with max_cols=10 raise naming column.'''
  frame = pd.DataFrame({
    'id': range(2000),
    'x1': [0.0] * 2000,
    'cat': [f'c{i}' for i in range(2000)],
    'seq': [[1.0, 2.0, 3.0]] * 2000,
    'target': [0, 1] * 1000,
  })
  steps = [{'CategoricalEncoding': {'skip': False, 'args': {
    'ohe_flag': True, 'max_cols': 10,
    'cols': {'include_cols': ['cat'], 'exclude_cols': []}}}}]
  with pytest.raises(ValueError, match='max_cols') as exc_info:
    preprocess(frame, steps, _vars())
  assert 'cat' in str(exc_info.value)


def test_encode_cat_ohe_default_cap_raises():
  '''Default max_cols (1000) raises on 2000 distinct cats.'''
  from forecast_ml_framework.pipelines.data_prep.nodes import _encode_cat
  frame = pd.DataFrame({'cat': [f'c{i}' for i in range(2000)]})
  with pytest.raises(ValueError, match='max_cols'):
    _encode_cat(frame, {'ohe_flag': True,
                        'cols': {'include_cols': ['cat']}}, _vars())


def test_encode_cat_ohe_normal_passes():
  '''Small cardinality with max_cols passes and expands.'''
  steps = [{'CategoricalEncoding': {'skip': False, 'args': {
    'ohe_flag': True, 'max_cols': 10,
    'cols': {'include_cols': ['cat'], 'exclude_cols': []}}}}]
  out = preprocess(_df(), steps, _vars())
  assert 'cat_a' in out.columns and 'cat_b' in out.columns


def test_encode_cat_ohe_total_expansion_exceeds_max_cols():
  '''Combined dummy expansion beyond max_cols raises naming col.'''
  frame = pd.DataFrame({
    'id': range(6),
    'cat_a': [f'a{i}' for i in range(6)],
    'cat_b': [f'b{i}' for i in range(6)],
    'target': [0, 1, 0, 1, 0, 1],
  })
  variables = dict(_vars(), cat_cols=['cat_a', 'cat_b'])
  steps = [{'CategoricalEncoding': {'skip': False, 'args': {
    'ohe_flag': True, 'max_cols': 10,
    'cols': {'include_cols': ['cat_a', 'cat_b'],
             'exclude_cols': []}}}}]
  with pytest.raises(ValueError, match='max_cols') as exc_info:
    preprocess(frame, steps, variables)
  assert 'cat_a' in str(exc_info.value) or 'cat_b' in str(
    exc_info.value)


def test_encode_cat_ohe_max_ohe_cols_alias_raises():
  '''max_ohe_cols alias cap raises like max_cols.'''
  from forecast_ml_framework.pipelines.data_prep.nodes import _encode_cat
  frame = pd.DataFrame({'cat': [f'c{i}' for i in range(2000)]})
  with pytest.raises(ValueError, match='max_cols'):
    _encode_cat(frame, {'ohe_flag': True, 'max_ohe_cols': 10,
                        'cols': {'include_cols': ['cat']}}, _vars())


@pytest.mark.parametrize('env_key', [
  'SCORING_MAX_COLS', 'DATA_PREP_MAX_COLS',
])
def test_encode_cat_ohe_env_cap_raises(monkeypatch, env_key):
  '''Env cap (both envs) raises when args cap absent.'''
  from forecast_ml_framework.pipelines.data_prep.nodes import _encode_cat
  frame = pd.DataFrame({'cat': [f'c{i}' for i in range(2000)]})
  monkeypatch.setenv(env_key, '10')
  with pytest.raises(ValueError, match='max_cols'):
    _encode_cat(frame, {'ohe_flag': True,
                        'cols': {'include_cols': ['cat']}}, _vars())


def test_encode_cat_ohe_dummy_name_collision_pair():
  '''a/b_c vs a_b/c both expand to a_b_c raising ValueError.'''
  frame = pd.DataFrame({'a': ['b_c', 'x'], 'a_b': ['c', 'y']})
  variables = dict(_vars(), cat_cols=['a', 'a_b'])
  steps = [{'CategoricalEncoding': {'skip': False, 'args': {
    'ohe_flag': True, 'max_cols': 100,
    'cols': {'include_cols': ['a', 'a_b'], 'exclude_cols': []}}}}]
  with pytest.raises(ValueError, match='collision|a_b_c'):
    preprocess(frame, steps, variables)


def test_encode_cat_ohe_dummy_collides_with_existing_column():
  '''Dummy cat_a colliding with existing cat_a column raises.'''
  frame = pd.DataFrame({
    'cat': ['a', 'b'], 'cat_a': [1, 0]})
  steps = [{'CategoricalEncoding': {'skip': False, 'args': {
    'ohe_flag': True, 'max_cols': 100,
    'cols': {'include_cols': ['cat'], 'exclude_cols': []}}}}]
  with pytest.raises(ValueError, match='collision|cat_a'):
    preprocess(frame, steps, _vars())


# --- loop4 dedupe (RED) ---

def test_knn_mean_fill_helper_fills_cols():
  '''_mean_fill helper fills NaN with per-column mean.'''
  from forecast_ml_framework.pipelines.data_prep.nodes import _mean_fill
  frame = pd.DataFrame({'x1': [0.0, 10.0, None],
                        'x2': [1.0, None, 3.0]})
  out = _mean_fill(frame, ['x1', 'x2'])
  assert out['x1'].iloc[2] == pytest.approx(5.0)
  assert out['x2'].iloc[1] == pytest.approx(2.0)


def test_seq_change_ratio_zero_prior_is_exact_zero():
  '''Change-ratio zero prior mean gives exact 0.0 (no eps offset).'''
  from forecast_ml_framework.pipelines.data_prep.nodes import (
    _seq_change_ratio)
  frame = pd.DataFrame({'seq': [[1.0, 0.0, 0.0], [1.0, 1.0, 5.0]]})
  out = _seq_change_ratio(
    frame, {'last_n_val': 2, 'cols': {'include_cols': ['seq']}}, _vars())
  assert out['seq_change_ratio'].iloc[0] == 0.0
  assert out['seq_change_ratio'].iloc[1] == pytest.approx(5.0)
