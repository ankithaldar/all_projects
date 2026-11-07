#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Smoke test: end-to-end demo pipeline on tiny synthetic data.

Runs the real pipeline nodes (EDA, data_prep, modeling,
interpretability, monitoring, scoring) on a small synthetic
classification frame. Does not depend on pre-generated
demo_outputs artifacts.
'''

import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from forecast_ml_framework.pipelines.data_prep.nodes import (
  preprocess,
  select_features,
)
from forecast_ml_framework.pipelines.eda.nodes import (
  detect_leakage,
  eda_report,
  eda_summary,
)
from forecast_ml_framework.pipelines.interpretability.nodes import (
  permutation_importance,
  shap_explain,
)
from forecast_ml_framework.pipelines.modeling.adapters import (
  train_models,
)
from forecast_ml_framework.pipelines.modeling.calibration import (
  calibrate_model,
)
from forecast_ml_framework.pipelines.modeling.charts import make_charts
from forecast_ml_framework.pipelines.modeling.data_split import (
  split_data,
)
from forecast_ml_framework.pipelines.modeling.evaluate import (
  evaluate_model,
)
from forecast_ml_framework.pipelines.modeling.pipeline import (
  explain_model,
)
from forecast_ml_framework.pipelines.monitoring.nodes import (
  compute_drift,
)
from forecast_ml_framework.pipelines.scoring.nodes import predict


def _frame(n=200, seed=42):
  '''Return tiny synthetic classification frame.'''
  rng = np.random.RandomState(seed)
  f1 = rng.randn(n)
  f2 = rng.randn(n)
  f3 = rng.randn(n)
  cat = rng.choice(['a', 'b', 'c'], n)
  seg = rng.choice(['s1', 's2'], n)
  logit = 1.5 * f1 - f2 + np.where(cat == 'a', 0.8, 0.0)
  prob = 1.0 / (1.0 + np.exp(-logit))
  target = (rng.rand(n) < prob).astype(int)
  return pd.DataFrame({
    'id': np.arange(n),
    'date': pd.date_range('2024-01-01', periods=n, freq='D'),
    'f1': f1,
    'f2': f2,
    'f3': f3,
    'cat': cat,
    'segment': seg,
    'target': target,
  })


def _variables():
  '''Return variables dict for the tiny frame.'''
  return {
    'target_col': 'target',
    'id_col': 'id',
    'feature_cols': ['f1', 'f2', 'f3', 'cat', 'segment'],
    'cat_cols': ['cat', 'segment'],
    'num_cols': ['f1', 'f2', 'f3'],
  }


def _model_params():
  '''Return modeling params for the tiny frame.'''
  return {
    'task_type': 'classification',
    'target_col': 'target',
    'models_to_use': ['sklearn'],
    'selection_metric': 'f1_weighted',
    'random_state': 42,
    'test_size': 0.25,
    'sampling_params': {'method': 'stratified'},
    'calibration': {'method': 'sigmoid', 'cv': 2},
    'bucket_segment_col': 'segment',
    'model_params': {'sklearn': {}},
    'shap_params': {'method': 'auto', 'n_repeats': 2},
  }


def _prep_steps():
  '''Return preprocessing steps for the tiny frame.'''
  return [
    {'Imputations': {'skip': False, 'args': {
      'strategy': 'mean',
      'cols': {'include_cols': ['f1', 'f2', 'f3'],
               'exclude_cols': []}}}},
    {'CategoricalEncoding': {'skip': False, 'args': {
      'ohe_flag': True,
      'cols': {'include_cols': ['cat', 'segment'],
               'exclude_cols': []}}}},
  ]


@pytest.mark.serial
def test_demo_pipeline_end_to_end_tiny():
  '''All demo stages run and return expected keys/shapes.'''
  df = _frame()
  variables = _variables()
  params = _model_params()

  summary = eda_summary(df, {}, variables)
  assert summary['shape'] == [200, 8]
  assert summary['n_rows'] == 200
  report = eda_report(summary, {}, df, variables)
  assert 'findings' in report and 'recommendations' in report
  leakage = detect_leakage(df, 'target', {})
  assert isinstance(leakage, list)

  prepped = preprocess(df, _prep_steps(), variables)
  assert len(prepped) == 200
  assert 'cat_a' in prepped.columns or 'cat' not in prepped.columns

  enc_vars = dict(variables, feature_cols=[
      c for c in prepped.columns
      if c not in ('id', 'date', 'target')])
  selected_df, selected, sel_report = select_features(
      prepped, {'feature_selection_params': {
          'variance_threshold': 0.0, 'missing_threshold': 0.5}},
      enc_vars)
  assert selected and 'target' in selected_df.columns
  assert sel_report['selected'] == selected

  features = [c for c in selected if c != 'target']
  seg_col = next((c for c in features if c.startswith('segment')),
                 None)
  params = dict(params, feature_cols=features,
                bucket_segment_col=seg_col)
  x_tr, x_te, y_tr, y_te, _meta = split_data(selected_df, params)
  assert len(x_tr) + len(x_te) == 200

  model, per_model, best = train_models(x_tr, y_tr, x_te, y_te, params)
  assert best == 'sklearn' and 'sklearn' in per_model
  calibrated, _cal_meta = calibrate_model(model, x_tr, y_tr, params)
  y_pred = calibrated.predict(x_te)
  proba = calibrated.predict_proba(x_te)
  assert len(y_pred) == len(y_te)

  metrics = evaluate_model(y_te, y_pred, proba, x_te, params)
  assert 'f1_weighted' in metrics['global']
  assert 'accuracy' in metrics['global']
  assert metrics['by_bucket'] and metrics['by_decile']

  figs = make_charts(y_te, y_pred, proba, x_te, params)
  assert len(figs) >= 5
  for fig in figs.values():
    fig.savefig(io.BytesIO(), format='png')
    plt.close(fig)

  artifact = explain_model(calibrated, x_te, y_te, params)
  assert artifact['global_importance']
  assert artifact['method'] in ('shap', 'permutation')

  # Interpretability nodes on the raw fitted estimator.
  raw = getattr(calibrated, 'model', calibrated)
  interp = shap_explain(raw, x_te, {'max_rows': 3})
  assert interp['global_importance']
  perm = permutation_importance(
      raw, x_te, y_te, {'n_repeats': 2, 'random_state': 42})
  assert isinstance(perm, list) and perm

  drift = compute_drift(prepped[features], prepped[features], {})
  assert features and all('psi' in drift[f] for f in features)
  assert max(abs(drift[f]['psi']) for f in features) < 1.0

  scoring = prepped.drop(columns=['target'], errors='ignore')
  preds = predict(
      scoring, {'id_cols': ['id'], 'feature_cols': features,
                'task_type': 'classification',
                'score_model_id': 'demo-test'},
      model_loader=calibrated)
  assert len(preds) == 200
  assert 'prediction' in preds.columns and 'proba' in preds.columns
