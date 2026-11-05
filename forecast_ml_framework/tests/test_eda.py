#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests for the EDA pipeline (Phase 3).'''

import numpy as np
import pandas as pd
import pytest

from forecast_ml_framework.pipelines.eda.nodes import (
  detect_leakage,
  eda_quality_checks,
  eda_report,
  eda_summary,
)
from forecast_ml_framework.pipelines.eda.pipeline import (
  create_pipeline,
)


def _variables():
  '''Return variables dict for tests.'''
  return {
    'target_col': 'target',
    'feature_cols': ['x', 'leak', 'cat', 'const', 'mostly_missing'],
  }


def _params(**overrides):
  '''Return eda params for tests.'''
  base = {
    'missing_threshold': 0.3,
    'cardinality_threshold': 10,
    'leakage_threshold': 0.95,
    'imbalance_ratio_threshold': 5.0,
    'zscore_threshold': 3.0,
    'top_correlations': 5,
    'top_values': 3,
  }
  base.update(overrides)
  return base


def _synthetic(n=100, seed=42):
  '''Return synthetic frame with leak + imbalance.'''
  rng = np.random.RandomState(seed)
  x = rng.normal(size=n)
  # 90/10 imbalance, leak ~ target (corr > 0.95).
  target = np.array([0] * 90 + [1] * 10)
  leak = target.astype(float) + rng.normal(scale=0.01, size=n)
  mostly_missing = rng.normal(size=n).astype(float)
  mostly_missing[:60] = np.nan
  const = np.full(n, 7.0)
  cat = np.array(['a'] * 50 + ['b'] * 30 + ['c'] * 20)
  df = pd.DataFrame({
    'x': x,
    'leak': leak,
    'cat': cat,
    'const': const,
    'mostly_missing': mostly_missing,
    'target': target,
  })
  # Duplicate a few rows for duplicate-row check (rows with
  # no missing values, so missing/target counts stay simple).
  return pd.concat([df, df.iloc[70:73]], ignore_index=True)


def test_eda_summary_shape_and_dtypes():
  '''Summary reports shape, dtypes, missing per column.'''
  df = _synthetic()
  summary = eda_summary(df, _params(), _variables())
  assert summary['shape'] == [len(df), len(df.columns)]
  assert summary['n_rows'] == len(df)
  assert summary['n_cols'] == len(df.columns)
  assert set(summary['dtypes']) == set(df.columns)
  assert summary['missing']['mostly_missing'] == 60
  assert summary['missing_rate']['mostly_missing'] == pytest.approx(
    60 / len(df))


def test_eda_summary_numeric_and_categorical():
  '''Numeric describe and categorical cardinality/top values.'''
  df = _synthetic()
  summary = eda_summary(df, _params(), _variables())
  assert 'x' in summary['numeric_describe']
  desc = summary['numeric_describe']['x']
  for key in ('count', 'mean', 'std', 'min', 'max'):
    assert key in desc
  assert summary['categorical']['cat']['cardinality'] == 3
  assert summary['categorical']['cat']['top_values'][0] == ['a', 50]
  assert summary['categorical']['cat']['top_values'][2] == ['c', 20]


def test_eda_summary_target_and_correlations():
  '''Target distribution and top correlation pairs present.'''
  df = _synthetic()
  summary = eda_summary(df, _params(), _variables())
  assert summary['target_distribution'] == {0: 93, 1: 10}
  assert summary['target_col'] == 'target'
  pairs = summary['correlations']
  assert isinstance(pairs, list) and pairs
  top = pairs[0]
  assert {'pair', 'correlation'} <= set(top)
  assert 'leak' in top['pair'] and 'target' in top['pair']
  assert top['correlation'] == pytest.approx(
    1.0, abs=0.05)


def test_eda_summary_outliers_duplicates_constant():
  '''Outlier, duplicate, and constant-column stats.'''
  df = _synthetic()
  summary = eda_summary(df, _params(), _variables())
  assert 'x' in summary['outliers']
  assert 'iqr' in summary['outliers']['x']
  assert 'zscore' in summary['outliers']['x']
  assert summary['duplicate_rows'] == 3
  assert summary['constant_cols'] == ['const']


def test_quality_checks_detect_leakage_and_imbalance():
  '''Leakage (corr>0.95) and class imbalance are flagged.'''
  summary = eda_summary(_synthetic(), _params(), _variables())
  findings = eda_quality_checks(summary, _params())
  checks = {f['check'] for f in findings}
  assert 'leakage_suspect' in checks
  assert 'class_imbalance' in checks
  for finding in findings:
    assert set(finding) >= {'check', 'severity', 'detail'}
  leak = next(f for f in findings if f['check'] == 'leakage_suspect')
  assert leak['severity'] == 'high'
  assert 'leak' in leak['detail']


def test_quality_checks_missing_cardinality_constant_duplicates():
  '''High missing, cardinality explosion, constant, duplicates.'''
  df = _synthetic()
  df['high_card'] = [f'v{i % 25}' for i in range(len(df))]
  summary = eda_summary(df, _params(), _variables())
  findings = eda_quality_checks(summary, _params())
  checks = {f['check'] for f in findings}
  assert 'high_missing' in checks
  assert 'cardinality_explosion' in checks
  assert 'constant_cols' in checks
  # Duplicates are covered by the summary duplicate test;
  # the per-row high_card column breaks row identity here.


def test_clean_data_no_high_findings():
  '''Balanced clean data yields no leakage/imbalance findings.'''
  rng = np.random.RandomState(0)
  n = 100
  df = pd.DataFrame({
    'x': rng.normal(size=n),
    'y': rng.normal(size=n),
    'target': np.array([0, 1] * (n // 2)),
  })
  variables = {'target_col': 'target', 'feature_cols': ['x', 'y']}
  summary = eda_summary(df, _params(), variables)
  findings = eda_quality_checks(summary, _params())
  checks = {f['check'] for f in findings}
  assert 'leakage_suspect' not in checks
  assert 'class_imbalance' not in checks


def test_empty_df_empty_report_with_warning(caplog):
  '''Empty df returns empty report with warning, no crash.'''
  df = pd.DataFrame({'x': pd.Series([], dtype=float)})
  with caplog.at_level('WARNING'):
    summary = eda_summary(df, _params(), _variables())
    report = eda_report(summary, _params())
  assert summary['shape'] == [0, 1]
  assert summary['warnings']
  assert report['summary'] == summary
  assert report['findings'] == []
  assert report['recommendations'] == []
  assert report['warnings']
  assert caplog.records


def test_empty_summary_quality_checks_no_crash():
  '''Quality checks on empty summary return no findings.'''
  df = pd.DataFrame()
  summary = eda_summary(df, _params(), _variables())
  assert eda_quality_checks(summary, _params()) == []


def test_eda_report_recommendations_are_strings():
  '''eda_report bundles summary, findings, recommendations.'''
  summary = eda_summary(_synthetic(), _params(), _variables())
  report = eda_report(summary, _params())
  assert report['summary'] == summary
  assert report['findings']
  assert all(isinstance(r, str) for r in report['recommendations'])
  assert report['recommendations']


def test_create_pipeline_two_nodes():
  '''Pipeline has exactly two nodes, eda_summary then report.'''
  pipeline = create_pipeline()
  nodes = pipeline.nodes
  nodes = nodes() if callable(nodes) else nodes
  names = [n.name for n in nodes]
  assert len(names) == 2
  assert names == ['eda_summary', 'eda_report']


def test_pipeline_run_on_synthetic_data():
  '''Kedro pipeline run produces eda_report with findings.'''
  from kedro.io import DataCatalog, MemoryDataset
  from kedro.runner import SequentialRunner

  catalog = DataCatalog(datasets={
    'raw_data': MemoryDataset(data=_synthetic()),
    'params:eda_params': MemoryDataset(data=_params()),
    'variables': MemoryDataset(data=_variables()),
  })
  SequentialRunner().run(create_pipeline(), catalog)
  report = catalog.load('eda_report')
  assert set(report) >= {'summary', 'findings', 'recommendations'}
  checks = {f['check'] for f in report['findings']}
  assert 'leakage_suspect' in checks
  assert 'class_imbalance' in checks
  assert 'leakage' in checks


def _leakage_params(**overrides):
  '''Return leakage params for tests.'''
  base = _params()
  base.update(overrides)
  return base


def _assert_leakage_finding(finding):
  '''Assert the fixed leakage finding shape.'''
  assert finding['check'] == 'leakage'
  assert finding['severity'] in ('warning', 'high')
  assert finding['feature']
  assert finding['detail']
  assert finding['recommendation'] == 'drop or investigate'


def test_detect_leakage_high_correlation():
  '''Feature-target correlation above threshold is flagged.'''
  findings = detect_leakage(
    _synthetic(), 'target', _leakage_params())
  leak = [f for f in findings if f['feature'] == 'leak']
  assert leak
  assert any('correlation' in f['detail'] for f in leak)
  for finding in findings:
    _assert_leakage_finding(finding)


def test_detect_leakage_duplicate_columns():
  '''Near-duplicate feature columns (corr > 0.99) are flagged.'''
  rng = np.random.RandomState(1)
  n = 100
  base = rng.normal(size=n)
  df = pd.DataFrame({
    'a': base,
    'a_copy': base + rng.normal(scale=0.001, size=n),
    'target': np.array([0, 1] * (n // 2)),
  })
  findings = detect_leakage(df, 'target', _leakage_params())
  dup = [f for f in findings
         if 'duplicate' in f['detail'].lower()]
  assert dup
  assert dup[0]['feature'] in ('a', 'a_copy')


def test_detect_leakage_perfect_separation_numeric():
  '''Numeric feature perfectly separating target is flagged.'''
  target = np.array([0] * 50 + [1] * 50)
  sep = np.concatenate([np.zeros(50), np.ones(50) * 10])
  df = pd.DataFrame({'sep': sep, 'target': target})
  findings = detect_leakage(df, 'target', _leakage_params())
  sep_findings = [f for f in findings if f['feature'] == 'sep']
  assert sep_findings
  assert any('separat' in f['detail'].lower()
             or 'auc' in f['detail'].lower()
             or 'correlation' in f['detail'].lower()
             for f in sep_findings)


def test_detect_leakage_categorical_pure_categories():
  '''Categorical feature with pure categories is flagged.'''
  df = pd.DataFrame({
    'cat_leak': ['a'] * 50 + ['b'] * 50,
    'target': np.array([0] * 50 + [1] * 50),
  })
  findings = detect_leakage(df, 'target', _leakage_params())
  cat_findings = [f for f in findings if f['feature'] == 'cat_leak']
  assert cat_findings
  assert any('pure' in f['detail'].lower()
             or 'mutual' in f['detail'].lower()
             or 'identical' in f['detail'].lower()
             or 'mapped' in f['detail'].lower()
             for f in cat_findings)


def test_detect_leakage_identical_to_target():
  '''Feature identical (or mapped 1-1) to target is flagged.'''
  df = pd.DataFrame({
    'same': [0, 1] * 50,
    'renamed': ['no', 'yes'] * 50,
    'target': [0, 1] * 50,
  })
  findings = detect_leakage(df, 'target', _leakage_params())
  features = {f['feature'] for f in findings}
  assert 'same' in features
  assert 'renamed' in features


def test_detect_leakage_name_keyword_warning():
  '''Post-outcome name keywords are flagged as warning.'''
  rng = np.random.RandomState(2)
  df = pd.DataFrame({
    'post_outcome_score': rng.normal(size=100),
    'churn_flag': rng.choice([0, 1], size=100),
    'target': np.array([0, 1] * 50),
  })
  findings = detect_leakage(df, 'target', _leakage_params())
  by_feature = {f['feature']: f for f in findings}
  assert by_feature['post_outcome_score']['severity'] == 'warning'
  assert 'post_outcome_score' in by_feature['post_outcome_score'][
    'detail']
  assert 'churn_flag' in by_feature
  assert by_feature['churn_flag']['severity'] == 'warning'


def test_detect_leakage_constant_feature():
  '''Constant / near-constant features are flagged.'''
  df = pd.DataFrame({
    'const': np.full(100, 3.0),
    'near_const': [1.0] * 99 + [2.0],
    'target': np.array([0, 1] * 50),
  })
  findings = detect_leakage(df, 'target', _leakage_params())
  features = {f['feature'] for f in findings}
  assert 'const' in features
  assert 'near_const' in features


def test_detect_leakage_clean_data_no_findings():
  '''Clean balanced data yields no leakage findings.'''
  rng = np.random.RandomState(0)
  n = 100
  df = pd.DataFrame({
    'x': rng.normal(size=n),
    'y': rng.normal(size=n),
    'target': np.array([0, 1] * (n // 2)),
  })
  assert detect_leakage(df, 'target', _leakage_params()) == []


def test_detect_leakage_empty_no_crash():
  '''Empty frame yields no leakage findings, no crash.'''
  assert detect_leakage(pd.DataFrame(), 'target', {}) == []
  assert detect_leakage(None, 'target', {}) == []


def test_detect_leakage_custom_threshold_and_keywords():
  '''Custom corr threshold and keyword list are honoured.'''
  df = pd.DataFrame({
    'custom_name': [0.0] * 50 + [1.0] * 50,
    'weak': np.random.RandomState(3).normal(size=100),
    'target': np.array([0] * 50 + [1] * 50),
  })
  params = _leakage_params(leakage_name_keywords=['custom'])
  findings = detect_leakage(df, 'target', params)
  assert any(f['feature'] == 'custom_name'
             and f['severity'] == 'warning'
             and 'keyword' in f['detail'].lower()
             for f in findings)


def test_eda_report_wires_leakage_findings():
  '''eda_report findings include detect_leakage findings.'''
  summary = eda_summary(_synthetic(), _params(), _variables())
  report = eda_report(summary, _params())
  leakage = [f for f in report['findings']
             if f['check'] == 'leakage']
  assert leakage
  assert any(f['feature'] == 'leak' for f in leakage)
  for finding in leakage:
    _assert_leakage_finding(finding)
  assert 'drop or investigate' in report['recommendations']
