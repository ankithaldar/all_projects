#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Degenerate-input tests for EDA leakage detection.

Covers constant, all-NaN, empty, and single-row frames so
degenerate columns cannot crash or slip past detection.
'''

import numpy as np
import pandas as pd
import pytest

from forecast_ml_framework.pipelines.eda.nodes import (
  _entropy,
  _normalized_mi,
  detect_leakage,
)


def test_entropy_empty_and_zero_counts():
  '''Zero/empty counts give zero entropy.

  Fails if _entropy divides by zero on empty counts.
  '''
  assert _entropy([]) == 0.0
  assert _entropy([0, 0]) == 0.0


def test_entropy_single_and_uniform_counts():
  '''Single and uniform counts match known values.

  Fails if _entropy mis-normalizes the probabilities.
  '''
  assert _entropy([1]) == pytest.approx(0.0)
  assert _entropy([1, 1]) == pytest.approx(0.6931471805599453)
  assert _entropy([5, 5]) == pytest.approx(np.log(2))


def test_normalized_mi_empty():
  '''Empty series give zero mutual information.

  Fails if _normalized_mi divides by zero on empty input.
  '''
  empty = pd.Series([], dtype=float)
  assert _normalized_mi(empty, empty) == 0.0


def test_normalized_mi_constant_feature():
  '''Constant feature gives zero mutual information.

  Fails if _normalized_mi returns nonzero for constants.
  '''
  feature = pd.Series(['a'] * 5)
  target = pd.Series(['x', 'y', 'x', 'y', 'x'])
  assert _normalized_mi(feature, target) == 0.0


def test_detect_leakage_none_and_empty():
  '''None and empty frames give no findings.

  Fails if detect_leakage crashes on None/empty input.
  '''
  assert detect_leakage(None, 't') == []
  assert detect_leakage(
    pd.DataFrame({'f': [], 't': []}), 't') == []


def test_detect_leakage_constant_feature_warns():
  '''Constant feature earns a warning finding.

  Fails if constant features go unreported.
  '''
  frame = pd.DataFrame({'f': [1.0] * 10, 't': [0, 1] * 5})
  findings = detect_leakage(frame, 't')
  assert any(
    item['feature'] == 'f' and item['severity'] == 'warning'
    for item in findings)


def test_detect_leakage_all_nan_column_warns():
  '''All-NaN column earns a warning, not a crash.

  Fails if all-NaN columns crash the leakage check.
  '''
  frame = pd.DataFrame({'f': [np.nan] * 10, 't': [0, 1] * 5})
  findings = detect_leakage(frame, 't')
  assert any(item['feature'] == 'f' for item in findings)


def test_detect_leakage_target_with_nan(caplog):
  '''Target with NaN returns findings, not a crash.

  Rows with a NaN target are dropped for the leakage
  computation and a warning is logged. Fails if
  detect_leakage raises AttributeError on the
  factorized ndarray target.
  '''
  frame = pd.DataFrame(
    {'f': [1.0, 2.0, 3.0, 4.0], 't': [0, 1, np.nan, 1]})
  with caplog.at_level('WARNING'):
    findings = detect_leakage(frame, 't')
  assert isinstance(findings, list)
  assert all(set(item) >= {'check', 'feature', 'detail'}
             for item in findings)
  assert any('nan' in record.getMessage().lower()
             for record in caplog.records)


def test_detect_leakage_target_nan_still_detects_leakage():
  '''Leakage on clean rows is still detected with NaN target.

  Fails if NaN target rows suppress or corrupt the
  correlation / separation findings for the remaining
  rows.
  '''
  n = 50
  target = np.array([0] * n + [1] * n, dtype=float)
  frame = pd.DataFrame({'leak': target.copy(), 't': target})
  frame.loc[0, 't'] = np.nan
  frame.loc[1, 't'] = np.nan
  findings = detect_leakage(frame, 't')
  leak_findings = [item for item in findings
                   if item['feature'] == 'leak']
  assert leak_findings
  assert any('correlation' in item['detail'].lower()
             or 'separat' in item['detail'].lower()
             or 'identical' in item['detail'].lower()
             for item in leak_findings)


def test_detect_leakage_all_nan_target_no_crash(caplog):
  '''All-NaN target returns no findings, not a crash.

  Fails if an all-NaN target raises AttributeError or
  returns spurious findings.
  '''
  frame = pd.DataFrame({'f': [1.0, 2.0, 3.0],
                        't': [np.nan] * 3})
  with caplog.at_level('WARNING'):
    findings = detect_leakage(frame, 't')
  assert findings == []
  assert caplog.records


def test_detect_leakage_categorical_target_with_nan():
  '''Categorical target with missing values does not crash.

  Fails if missing categorical labels corrupt the
  purity / mutual-information checks.
  '''
  frame = pd.DataFrame({
    'f': ['a', 'b', 'a', 'b', 'a', 'b'],
    't': ['x', 'y', None, 'y', 'x', np.nan]})
  findings = detect_leakage(frame, 't')
  assert isinstance(findings, list)
  assert all('feature' in item for item in findings)


def test_detect_leakage_single_row():
  '''Single-row frame returns findings without crashing.

  Fails if single-row input crashes the checks.
  '''
  frame = pd.DataFrame({'f': [1.0], 't': [0]})
  findings = detect_leakage(frame, 't')
  assert isinstance(findings, list)
  assert len(findings) >= 1
  assert all('feature' in item for item in findings)
