#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''SHAP node with permutation-importance fallback.'''

import numpy as np


def _columns(x):
  '''Return feature names for X.

  Args:
    x: Features with columns, or array-like.

  Returns:
    List of feature names, generated if absent.
  '''
  if hasattr(x, 'columns'):
    return list(x.columns)
  return [f'f{i}' for i in range(np.asarray(x).shape[1])]


def _permutation(model, x, y, params):
  '''Compute permutation importance as fallback.

  Args:
    model: Fitted adapter or estimator.
    x: Features.
    y: Target values.
    params: Dict with shap_params settings.

  Returns:
    Dict artifact with permutation importance.
  '''
  from sklearn.inspection import permutation_importance
  cfg = params.get('shap_params', {}) or {}
  raw = getattr(model, 'model', model)
  result = permutation_importance(
      raw, x, y, n_repeats=cfg.get('n_repeats', 5),
      random_state=cfg.get('random_state', 42))
  cols = _columns(x)
  importance = {cols[i]: float(v) for i, v in
                enumerate(result.importances_mean)}
  per_row = {}
  try:
    arr = np.asarray(x)
    for idx in range(min(len(arr), cfg.get('sample_rows', 100))):
      per_row[str(idx)] = {cols[i]: 0.0 for i in range(len(cols))}
  except Exception:  # noqa: BLE001
    per_row = {}
  return {'method': 'permutation', 'global_importance': importance,
          'per_row': per_row, 'shap_available': False}


def explain_model(model, x, y, params):
  '''Explain a fitted model globally and per row.

  Args:
    model: Fitted adapter or estimator.
    x: Features to explain.
    y: Target values.
    params: modeling_params dict with shap_params
      {method: shap|permutation|auto, sample_rows}.

  Returns:
    Dict artifact with method, global_importance, per_row.
  '''
  params = params or {}
  cfg = params.get('shap_params', {}) or {}
  method = cfg.get('method', 'auto')
  if method in ('shap', 'auto'):
    try:
      import shap
      raw = getattr(model, 'model', model)
      explainer = shap.Explainer(raw, np.asarray(x))
      values = explainer(np.asarray(x)).values
      if values.ndim == 3:
        values = values[:, :, 1]
      cols = _columns(x)
      importance = {cols[i]: float(np.abs(values[:, i]).mean())
                    for i in range(values.shape[1])}
      per_row = {str(i): {cols[j]: float(values[i, j])
                          for j in range(values.shape[1])}
                 for i in range(min(len(values),
                                    cfg.get('sample_rows', 100)))}
      return {'method': 'shap', 'global_importance': importance,
              'per_row': per_row, 'shap_available': True}
    except Exception:  # noqa: BLE001 - fallback to permutation
      if method == 'shap':
        raise
  return _permutation(model, x, y, params)
