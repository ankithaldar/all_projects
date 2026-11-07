#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Edge tests for metric selection and confusion rates.

Pins defaults, case-sensitivity, the unknown-metric error
contract, and confusion-table degenerates.
'''

import math

import pytest

from forecast_ml_framework.pipelines.modeling.metrics import (
  confusion_rates_table,
  resolve_selected_metrics,
)


def test_resolve_defaults_by_task():
  '''None returns task-filtered defaults.

  Fails if defaults are unfiltered across tasks.
  '''
  classification = resolve_selected_metrics(None, 'classification')
  regression = resolve_selected_metrics(None, 'regression')
  assert 'accuracy' in classification
  assert 'mae' not in classification
  assert 'mae' in regression
  assert 'accuracy' not in regression


def test_resolve_all_by_task():
  ''''all' returns every valid metric for the task.

  Fails if 'all' returns the other task's metrics.
  '''
  classification = resolve_selected_metrics('all', 'classification')
  regression = resolve_selected_metrics('all', 'regression')
  assert 'accuracy' in classification
  assert 'mae' in regression
  assert 'mae' not in classification


def test_resolve_single_selection():
  '''Single known name resolves to itself.

  Fails if single-name selection is dropped.
  '''
  assert resolve_selected_metrics(
    ['accuracy'], 'classification') == ['accuracy']
  assert resolve_selected_metrics(
    'accuracy', 'classification') == ['accuracy']


def test_resolve_unknown_raises_with_valid_list():
  '''Unknown metric raises listing valid names.

  Fails if unknown metrics pass silently.
  '''
  with pytest.raises(ValueError, match='Unknown metric'):
    resolve_selected_metrics(['unknown_xyz'], 'classification')
  with pytest.raises(ValueError, match='Valid metrics'):
    resolve_selected_metrics(['unknown_xyz'], 'classification')


def test_resolve_empty_selection():
  '''Empty selection resolves to empty.

  Fails if empty selection falls back to defaults.
  '''
  assert resolve_selected_metrics([], 'classification') == []


def test_resolve_is_case_sensitive_no_alias():
  '''Names are case-sensitive; fuzzy aliases do not exist.

  Fails if selection is case-insensitive or aliased.
  '''
  with pytest.raises(ValueError, match='Unknown metric'):
    resolve_selected_metrics(['Accuracy'], 'classification')
  with pytest.raises(ValueError, match='Unknown metric'):
    resolve_selected_metrics(['f1'], 'classification')


def test_resolve_cross_task_raises():
  '''Cross-task metric raises for the wrong task.

  Fails if validation ignores the task type.
  '''
  with pytest.raises(ValueError, match='Unknown metric'):
    resolve_selected_metrics(['mae'], 'classification')
  with pytest.raises(ValueError, match='Unknown metric'):
    resolve_selected_metrics(['accuracy'], 'regression')


def test_confusion_rates_perfect():
  '''Perfect binary predictions give unit tpr/tnr.

  Fails if confusion rates mis-assign the cells.
  '''
  rows = confusion_rates_table([0, 1, 0, 1], [0, 1, 0, 1])
  assert [row['label'] for row in rows] == [
    'Actual Positive', 'Actual Negative']
  assert rows[0]['tpr'] == pytest.approx(1.0)
  assert rows[0]['tnr'] == pytest.approx(1.0)
  assert rows[0]['fpr'] == pytest.approx(0.0)
  assert rows[0]['fnr'] == pytest.approx(0.0)


def test_confusion_rates_empty_nan():
  '''Empty inputs give NaN rates, two rows.

  Fails if empty input raises instead of returning NaN rows.
  '''
  rows = confusion_rates_table([], [])
  assert len(rows) == 2
  assert math.isnan(rows[0]['tpr'])
  assert math.isnan(rows[0]['fpr'])
