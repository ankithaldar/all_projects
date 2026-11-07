#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Parity Task 8: pipeline lift_table + promotion wiring.'''

import math

import numpy as np
import pandas as pd

from forecast_ml_framework.pipelines.modeling.pipeline import (
  build_lift_table,
  build_promotion,
  create_pipeline,
)


def _cls_data(n=1000):
  '''Return synthetic classification labels and probabilities.'''
  rng = np.random.default_rng(0)
  proba_pos = rng.random(n)
  y_true = (proba_pos > 0.5).astype(int)
  y_pred = (proba_pos >= 0.5).astype(int)
  y_proba = np.column_stack([1 - proba_pos, proba_pos])
  return y_true, y_pred, y_proba


def test_create_pipeline_has_lift_and_promotion_nodes():
  '''create_pipeline registers lift_table and promotion nodes.'''
  pipe = create_pipeline()
  names = {node.name for node in pipe.nodes}
  assert 'build_lift_table_node' in names
  assert 'build_promotion_node' in names
  outputs = set()
  for node in pipe.nodes:
    outputs.update(node.outputs)
  assert 'lift_table' in outputs
  assert 'promotion' in outputs


def test_build_lift_table_returns_ten_rows():
  '''build_lift_table on synthetic data returns a 10-row table.'''
  y_true, y_pred, y_proba = _cls_data()
  out = build_lift_table(y_true, y_pred, y_proba)
  assert isinstance(out, pd.DataFrame)
  assert len(out) == 10
  assert out.iloc[0]['lift'] > 1


def test_build_lift_table_regression_none_proba_is_empty():
  '''Without proba (regression) lift table is empty, not a crash.'''
  y_true, y_pred, _ = _cls_data(n=100)
  out = build_lift_table(y_true, y_pred, None)
  assert isinstance(out, pd.DataFrame)
  assert len(out) == 0


def test_build_promotion_uses_selected_ntile_and_old_score():
  '''Promotion compares selected_ntile score against old_score.'''
  evaluation = {
    'selected_ntile': {
      'eval_metric': 'LIFT_DECILE_10',
      'metric': 'lift',
      'score': 0.97,
      'ntile': 10,
    },
  }
  out = build_promotion(
    evaluation, {'task_type': 'classification', 'old_score': 1.0})
  assert out['promote'] is True
  assert out['deviation_pct'] == -3.0
  assert out['new_score'] == 0.97
  assert out['old_score'] == 1.0


def test_build_promotion_without_old_score_never_promotes():
  '''Missing old_score (or selected_ntile) never promotes.'''
  evaluation = {
    'selected_ntile': {'metric': 'lift', 'score': 0.97, 'ntile': 10},
  }
  out = build_promotion(evaluation, {'task_type': 'classification'})
  assert out['promote'] is False
  assert math.isnan(out['deviation_pct'])
  out = build_promotion(
    {'selected_ntile': None},
    {'task_type': 'classification', 'old_score': 1.0})
  assert out['promote'] is False
  assert math.isnan(out['deviation_pct'])


def test_build_promotion_bad_old_score_never_promotes():
  '''Negative/non-finite old_score never promotes via pipeline.'''
  evaluation = {
    'selected_ntile': {'metric': 'rmse', 'score': 90.0, 'ntile': 10},
  }
  for old in (-100.0, float('inf'), float('nan')):
    out = build_promotion(
      evaluation, {'task_type': 'regression', 'old_score': old})
    assert out['promote'] is False
    assert math.isnan(out['deviation_pct'])


def test_build_lift_table_with_lift_threshold_filters_cumsum():
  '''Pipeline lift table honours lift_threshold filtered cumsum.'''
  y_true, y_pred, y_proba = _cls_data()
  proba = y_proba[:, 1]
  out = build_lift_table(
    y_true, y_pred, y_proba, {'lift_threshold': 0.6})
  expected_customers = int((proba >= 0.6).sum())
  assert out.iloc[-1]['customers_at_or_above_threshold'] == (
    expected_customers)
  assert out.iloc[-1]['true_positive_at_or_above_threshold'] == int(
    y_true[proba >= 0.6].sum())
  assert out.iloc[-1]['customers_at_or_above_threshold'] < len(y_true)


def test_build_lift_table_with_model_threshold_alias():
  '''Pipeline lift table honours model_threshold alias.'''
  y_true, y_pred, y_proba = _cls_data()
  proba = y_proba[:, 1]
  out = build_lift_table(
    y_true, y_pred, y_proba, {'model_threshold': 0.7})
  assert out.iloc[-1]['customers_at_or_above_threshold'] == int(
    (proba >= 0.7).sum())


def test_build_lift_table_threshold_kwarg_filters_cumsum():
  '''Pipeline lift table honours threshold= kwarg.'''
  y_true, y_pred, y_proba = _cls_data()
  proba = y_proba[:, 1]
  out = build_lift_table(
    y_true, y_pred, y_proba, threshold=0.6)
  assert out.iloc[-1]['customers_at_or_above_threshold'] == int(
    (proba >= 0.6).sum())


def test_pipeline_has_selected_threshold_and_tuning_report_nodes():
  '''Pipeline registers threshold + tuning_report persistence nodes.'''
  pipe = create_pipeline()
  names = {node.name for node in pipe.nodes}
  assert 'select_model_threshold_node' in names
  assert 'build_tuning_report_node' in names
  outputs = set()
  for node in pipe.nodes:
    outputs.update(node.outputs)
  assert 'selected_threshold' in outputs
  assert 'tuning_report' in outputs


def test_select_model_threshold_uses_select_threshold():
  '''select_model_threshold wraps select_threshold on y_test/proba.'''
  from forecast_ml_framework.pipelines.modeling.pipeline import (
    select_model_threshold,
  )
  from forecast_ml_framework.pipelines.modeling.threshold import (
    select_threshold,
  )
  y_true, _y_pred, y_proba = _cls_data()
  proba = y_proba[:, 1]
  expected = select_threshold(y_true, proba, 'roc_curve')
  got = select_model_threshold(
    y_true, y_proba, {'threshold_method': 'roc_curve'})
  assert got == expected
  # 1-D proba and default method also work.
  assert 0.0 <= select_model_threshold(y_true, proba, {}) <= 1.0
  assert select_model_threshold(y_true, None, {}) == 0.5


def test_build_tuning_report_extracts_per_model_tuning():
  '''build_tuning_report pulls tuning dicts out of per_model_metrics.'''
  from forecast_ml_framework.pipelines.modeling.pipeline import (
    build_tuning_report,
  )
  per_model = {
    'sklearn': {'f1': 0.9, 'tuning': {'method': 'grid',
                                      'best_params': {'C': 1.0}}},
    'xgboost': {'f1': 0.8},
  }
  out = build_tuning_report(per_model)
  assert out == {'sklearn': {'method': 'grid',
                             'best_params': {'C': 1.0}}}
  assert build_tuning_report({}) == {}
  assert build_tuning_report(None) == {}


def test_lift_node_wired_with_modeling_params():
  '''Lift node is wired with modeling_params for threshold.'''
  pipe = create_pipeline()
  lift_nodes = [
    node for node in pipe.nodes
    if node.name == 'build_lift_table_node']
  assert len(lift_nodes) == 1
  inputs = list(lift_nodes[0].inputs)
  assert 'params:modeling_params' in inputs
