#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Parity Task 10 rollout: both grammars, lift + gate.'''

import numpy as np
import pandas as pd

from forecast_ml_framework.pipelines.modeling.evaluate import evaluate_model
from forecast_ml_framework.pipelines.modeling.pipeline import (
  build_lift_table,
  build_promotion,
)


def _cls_data(n=1000):
  '''Return synthetic classification data.'''
  rng = np.random.default_rng(0)
  proba_pos = rng.random(n)
  y_true = (proba_pos > 0.5).astype(int)
  y_pred = (proba_pos >= 0.5).astype(int)
  y_proba = np.column_stack([1 - proba_pos, proba_pos])
  x = pd.DataFrame({'f1': rng.random(n)})
  return y_true, y_pred, y_proba, x


def _rollout(eval_metric):
  '''Run evaluate, lift table, and promotion for a metric.'''
  y_true, y_pred, y_proba, x = _cls_data()
  params = {
    'task_type': 'classification',
    'metrics': ['lift_top_decile', 'roc_auc'],
    'eval_metric': eval_metric,
    'old_score': 1.0,
  }
  evaluation = evaluate_model(y_true, y_pred, y_proba, x, params)
  lift = build_lift_table(y_true, y_pred, y_proba)
  promotion = build_promotion(evaluation, params)
  return evaluation, lift, promotion


def test_rollout_decile_grammar_lift_and_gate():
  '''DECILE grammar: selected score, 10 lift rows, gate result.'''
  evaluation, lift, promotion = _rollout('LIFT_DECILE_10')
  assert evaluation['selected_ntile']['ntile'] == 10
  assert isinstance(evaluation['selected_ntile']['score'], float)
  assert len(lift) == 10
  assert lift.iloc[0]['lift'] > 1
  assert promotion['new_score'] == evaluation['selected_ntile']['score']
  assert promotion['old_score'] == 1.0
  assert isinstance(promotion['promote'], bool)
  assert isinstance(promotion['deviation_pct'], float)


def test_rollout_centile_grammar_lift_and_gate():
  '''CENTILE grammar: by_centile present, lift + gate still hold.'''
  evaluation, lift, promotion = _rollout('LIFT_CENTILE_100')
  assert 'by_centile' in evaluation
  assert evaluation['selected_ntile']['ntile'] == 100
  assert len(lift) == 10
  assert promotion['new_score'] == evaluation['selected_ntile']['score']
  assert isinstance(promotion['promote'], bool)
