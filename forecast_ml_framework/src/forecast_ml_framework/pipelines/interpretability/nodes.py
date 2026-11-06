#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Interpretability and diagnostics nodes (Phase 3).

Structured artifacts (plain dicts/lists) that chart helpers
can render; no plotting happens here.

Params keys (``params:interpretability_params`` dict, all
optional):

- ``max_rows`` (int, default 20): cap on per-row
  explanations. Aliases: ``max_per_row_rows``.
- ``top_k`` (int, default 10): features kept per-row /
  in global lists.
- ``n_repeats`` (int, default 5): permutation repeats.
- ``random_state`` / ``seed`` (int, default 42).
- ``n_grid`` (int, default 20): PDP grid size (percentiles
  of the feature column). Alias: ``grid_size``.
- ``pdp_features`` (list, default all features): features
  for :func:`interpret_features` PDP.
- ``n_calibration_bins`` (int, default 10): bins for
  calibration in :func:`diagnostics`. Aliases:
  ``calibration_bins``, ``bins``.
- ``segment_col`` (str or None): column of X used for
  ``error_by_segment``. When absent, segments are
  quantile bins of the predictions.
- ``scorer`` (str or None): sklearn scorer for
  permutation importance; default scorer when None.

``shap`` and ``lime`` are imported lazily inside the
nodes that use them. When unavailable, ``shap_explain``
falls back to an occlusion/permutation surrogate and
``lime_explain`` falls back to a permutation-based
surrogate (its ``method`` field records which ran).
'''

import logging

import numpy as np
import pandas as pd

from forecast_ml_framework.core.params import (
  flatten_params,
)

LOG = logging.getLogger(__name__)



def _int(value, default):
  '''Return value as int, or default when missing/invalid.'''
  try:
    return int(value)
  except (TypeError, ValueError):
    return int(default)


def _seed(params):
  '''Resolve the random seed from params.'''
  params = flatten_params(params, 'interpretability_params')
  return _int(params.get('random_state', params.get('seed', 42)), 42)


def _is_valid_X(X) -> bool:
  '''Return True when X is a 2-D (or 1-D) numeric frame/array.

  Scalars, strings/bytes, dicts, 0-d arrays, and non-numeric
  arrays are invalid; DataFrames are accepted as-is (their
  columns are converted by downstream nodes where needed).
  '''
  if X is None or isinstance(X, (str, bytes, dict)):
    return False
  if isinstance(X, pd.DataFrame):
    return X.ndim == 2
  try:
    arr = np.asarray(X)
  except Exception:  # noqa: BLE001 - validation probe
    return False
  if arr.ndim == 0:
    return False
  try:
    arr.astype(float)
  except (TypeError, ValueError):
    return False
  return True


def _row_count(X) -> int:
  '''Return number of rows of X, or 0 when it has none/invalid.'''
  if not _is_valid_X(X):
    return 0
  try:
    return len(X)
  except TypeError:
    return 0


def _seq_len(value) -> int:
  '''Return len of a sequence, or 0 for scalars/None.'''
  if value is None:
    return 0
  try:
    return len(np.asarray(value, dtype=object))
  except TypeError:
    return 0


def _frame(X, features=None):
  '''Return X as a DataFrame with named columns.'''
  if isinstance(X, pd.DataFrame):
    return X
  arr = np.asarray(X)
  if arr.ndim == 1:
    arr = arr.reshape(-1, 1)
  names = features or ['f%d' % i for i in range(arr.shape[1])]
  return pd.DataFrame(arr, columns=list(names)[:arr.shape[1]])


def _features(X):
  '''Return feature names for X (DataFrame or array).'''
  if isinstance(X, pd.DataFrame):
    return list(X.columns)
  arr = np.asarray(X)
  if arr.ndim == 1:
    arr = arr.reshape(-1, 1)
  return ['f%d' % i for i in range(arr.shape[1])]


def _predict(model, X):
  '''Return 1-D float predictions from model.predict.'''
  preds = np.asarray(model.predict(X), dtype=float)
  if preds.ndim > 1:
    preds = preds[:, -1]
  return preds.flatten()


def _sorted_importance(scores, top_k=None):
  '''Return sorted [{'feature', 'importance'}] list.'''
  entries = [{'feature': str(f), 'importance': float(v)}
             for f, v in scores]
  entries.sort(key=lambda e: e['importance'], reverse=True)
  if top_k is not None:
    entries = entries[:max(int(top_k), 0)]
  return entries


def _pred_change_importance(model, frame, params):
  '''Permutation importance via prediction change (no y).'''
  rng = np.random.RandomState(_seed(params))
  repeats = max(_int(flatten_params(
      params, 'interpretability_params').get(
          'n_repeats', 5), 5), 1)
  base = _predict(model, frame)
  scores = []
  for col in frame.columns:
    diffs = []
    for _ in range(repeats):
      shuffled = frame.copy()
      shuffled[col] = rng.permutation(frame[col].to_numpy())
      diffs.append(np.mean(np.abs(_predict(model, shuffled) - base)))
    scores.append((col, float(np.mean(diffs))))
  return scores


def _occlusion_rows(model, frame, params, max_rows):
  '''Per-row occlusion surrogate SHAP values.'''
  rows = []
  medians = {c: float(np.nanmedian(frame[c].astype(float)))
             for c in frame.columns}
  for idx in range(min(len(frame), max_rows)):
    row = frame.iloc[[idx]]
    base = float(_predict(model, row)[0])
    values = {}
    for col in frame.columns:
      occluded = row.copy()
      occluded[col] = medians[col]
      values[str(col)] = float(base - _predict(model, occluded)[0])
    rows.append({'row': int(idx), 'shap_values': values,
                 'prediction': base})
  return rows


def shap_explain(model, X, params=None):
  '''Explain a model with SHAP, permutation fallback.

  Args:
    model: Fitted sklearn-agnostic model with ``predict``.
    X: Feature frame/array (empty frames do not raise).
    params: Optional dict; see module docstring.

  Returns:
    ``{'global_importance', 'per_row', 'baseline',
    'method', 'warnings'}``. ``global_importance`` is a
    sorted ``[{'feature', 'importance'}]`` list,
    ``per_row`` is capped at ``max_rows``, and
    ``baseline`` is the mean prediction. Falls back to
    occlusion/permutation when ``shap`` is unavailable
    or explainer construction fails.
  '''
  params = flatten_params(params, 'interpretability_params')
  warnings = []
  if not _is_valid_X(X):
    warnings.append('invalid X: expected a 2-D numeric '
                    'frame/array')
    LOG.warning(warnings[-1])
    return {'global_importance': [], 'per_row': [],
            'baseline': 0.0, 'method': 'none',
            'warnings': warnings}
  if _row_count(X) == 0:
    warnings.append('empty data: no rows to explain')
    LOG.warning(warnings[-1])
    return {'global_importance': [], 'per_row': [],
            'baseline': 0.0, 'method': 'none',
            'warnings': warnings}
  frame = _frame(X)
  max_rows = _int(params.get('max_rows',
                             params.get('max_per_row_rows', 20)), 20)
  baseline = 0.0
  try:
    baseline = float(np.mean(_predict(model, frame)))
    import shap  # lazy: optional dependency
    explainer = shap.Explainer(model, frame)
    result = explainer(frame)
    values = np.asarray(result.values, dtype=float)
    if values.ndim == 3:  # (rows, features, classes)
      values = values[:, :, -1]
    mean_abs = np.mean(np.abs(values), axis=0)
    per_row = [
      {'row': int(i),
       'shap_values': {str(c): float(v)
                       for c, v in zip(frame.columns, values[i])},
       'prediction': float(_predict(model, frame.iloc[[i]])[0])}
      for i in range(min(len(frame), max_rows))]
    return {
      'global_importance': _sorted_importance(
        list(zip(frame.columns, mean_abs))),
      'per_row': per_row, 'baseline': baseline,
      'method': 'shap', 'warnings': warnings}
  except Exception as exc:  # noqa: BLE001 - fallback path
    warnings.append('shap unavailable, using permutation/'
                    'occlusion fallback: %s' % exc)
    LOG.warning(warnings[-1])
  try:
    scores = _pred_change_importance(model, frame, params)
    return {
      'global_importance': _sorted_importance(scores),
      'per_row': _occlusion_rows(model, frame, params, max_rows),
      'baseline': baseline, 'method': 'permutation_fallback',
      'warnings': warnings}
  except Exception as exc:  # noqa: BLE001 - hostile model
    warnings.append('explanation failed for model: %s' % exc)
    LOG.warning(warnings[-1])
    return {'global_importance': [], 'per_row': [],
            'baseline': 0.0, 'method': 'none',
            'warnings': warnings}


def permutation_importance(model, X, y=None, params=None):
  '''Permutation importance as a sorted list.

  Uses sklearn's scorer-based permutation importance
  when ``y`` is given, otherwise prediction-change
  permutation. Empty inputs return ``[]``.
  '''
  params = flatten_params(params, 'interpretability_params')
  if not _is_valid_X(X) or _row_count(X) == 0:
    LOG.warning('empty/invalid data: no rows for permutation '
                'importance')
    return []
  frame = _frame(X)
  if y is not None and _seq_len(y) == len(frame) \
      and len(frame) > 0:
    try:
      from sklearn.inspection import permutation_importance as _pi
      result = _pi(model, frame, np.asarray(y),
                   n_repeats=max(
                     _int(params.get('n_repeats', 5), 5), 1),
                   random_state=_seed(params),
                   scorer=params.get('scorer'))
      scores = list(zip(frame.columns, result.importances_mean))
      return _sorted_importance(scores)
    except Exception as exc:  # noqa: BLE001 - fallback path
      LOG.warning('sklearn permutation failed, using '
                  'prediction-change fallback: %s' % exc)
  try:
    return _sorted_importance(
      _pred_change_importance(model, frame, params))
  except Exception as exc:  # noqa: BLE001 - hostile model
    LOG.warning('permutation importance failed for model: %s' % exc)
    return []


def pdp_1d(model, X, feature, params=None):
  '''Manual 1-D partial dependence for one feature.

  Grid = percentiles of the feature column; values =
  mean prediction with the feature set to each grid
  value across all rows (sklearn-agnostic, ``predict``
  only). Unknown/empty inputs warn, never raise.
  '''
  params = flatten_params(params, 'interpretability_params')
  warnings = []
  if not _is_valid_X(X) or _row_count(X) == 0:
    warnings.append('empty/invalid data: no rows for PDP')
    LOG.warning(warnings[-1])
    return {'feature': str(feature), 'grid': [], 'values': [],
            'warnings': warnings}
  frame = _frame(X)
  if feature not in frame.columns:
    warnings.append('feature %r not in data' % (feature,))
    LOG.warning(warnings[-1])
    return {'feature': str(feature), 'grid': [], 'values': [],
            'warnings': warnings}
  n_grid = _int(params.get('n_grid', params.get('grid_size', 20)), 20)
  column = frame[feature].astype(float).to_numpy()
  grid = np.unique(np.percentile(
    column, np.linspace(0, 100, max(n_grid, 1))))
  values = []
  try:
    for value in grid:
      modified = frame.copy()
      modified[feature] = float(value)
      values.append(float(np.mean(_predict(model, modified))))
  except Exception as exc:  # noqa: BLE001 - hostile model
    warnings.append('PDP failed for model: %s' % exc)
    LOG.warning(warnings[-1])
    return {'feature': str(feature), 'grid': [], 'values': [],
            'warnings': warnings}
  return {'feature': str(feature),
          'grid': [float(g) for g in grid], 'values': values,
          'warnings': warnings}


def _calibration_bins(preds, y, n_bins):
  '''Equal-width calibration bins of predictions.'''
  n_bins = max(_int(n_bins, 10), 1)
  low, high = float(np.min(preds)), float(np.max(preds))
  if low == high:
    edges = np.array([low, high + 1.0])
  else:
    edges = np.linspace(low, high, n_bins + 1)
  bins = []
  for i in range(len(edges) - 1):
    if i == len(edges) - 2:
      mask = (preds >= edges[i]) & (preds <= edges[i + 1])
    else:
      mask = (preds >= edges[i]) & (preds < edges[i + 1])
    count = int(mask.sum())
    if count == 0:
      continue
    mean_pred = float(np.mean(preds[mask]))
    mean_actual = float(np.mean(y[mask]))
    bins.append({'bin': i, 'count': count,
                 'mean_predicted': mean_pred,
                 'mean_actual': mean_actual,
                 'gap': float(mean_pred - mean_actual)})
  return bins


def _segments(frame, preds, params):
  '''Return segment labels for each row.'''
  segment_col = flatten_params(
      params, 'interpretability_params').get('segment_col')
  if segment_col and segment_col in frame.columns:
    return frame[segment_col].astype(str).to_numpy()
  labels = np.full(len(frame), 'segment_0', dtype=object)
  if len(frame) > 1 and np.unique(preds).size > 1:
    edges = np.unique(np.quantile(preds, [0, .5, 1.0]))
    if len(edges) > 2:
      idx = np.digitize(preds, edges[1:-1])
      labels = np.array(['segment_%d' % i for i in idx], dtype=object)
  return labels


def diagnostics(model, X, y, params=None):
  '''Structured calibration/residual/segment diagnostics.

  Returns ``{'calibration_bins', 'residual_stats',
  'error_by_segment', 'prediction_summary', 'warnings'}``.
  Empty inputs return empty bins with a warning.
  '''
  params = flatten_params(params, 'interpretability_params')
  warnings = []
  if not _is_valid_X(X) or _row_count(X) == 0 \
      or _seq_len(y) == 0:
    warnings.append('empty/invalid data: no rows for diagnostics')
    LOG.warning(warnings[-1])
    return {'calibration_bins': [], 'residual_stats': {},
            'error_by_segment': [], 'prediction_summary': {},
            'warnings': warnings}
  frame = _frame(X)
  y_arr = np.asarray(y, dtype=float).flatten()
  segment_col = params.get('segment_col')
  predict_frame = frame.drop(columns=[segment_col]) \
    if segment_col in frame.columns else frame
  try:
    preds = _predict(model, predict_frame)
  except Exception as exc:  # noqa: BLE001 - hostile model
    warnings.append('diagnostics failed for model: %s' % exc)
    LOG.warning(warnings[-1])
    return {'calibration_bins': [], 'residual_stats': {},
            'error_by_segment': [], 'prediction_summary': {},
            'warnings': warnings}
  n = min(len(frame), len(y_arr))
  frame, y_arr, preds = frame.iloc[:n], y_arr[:n], preds[:n]
  residuals = y_arr - preds
  segments = _segments(frame, preds, params)
  error_by_segment = []
  for segment in sorted(set(segments)):
    mask = segments == segment
    seg_res = residuals[mask]
    error_by_segment.append({
      'segment': str(segment), 'count': int(mask.sum()),
      'mae': float(np.mean(np.abs(seg_res))),
      'rmse': float(np.sqrt(np.mean(seg_res ** 2))),
      'bias': float(np.mean(seg_res))})
  n_bins = _int(params.get('n_calibration_bins',
                           params.get('calibration_bins',
                                      params.get('bins', 10))), 10)
  return {
    'calibration_bins': _calibration_bins(preds, y_arr, n_bins),
    'residual_stats': {
      'mean': float(np.mean(residuals)),
      'std': float(np.std(residuals)),
      'mae': float(np.mean(np.abs(residuals))),
      'rmse': float(np.sqrt(np.mean(residuals ** 2))),
      'bias': float(np.mean(residuals)),
      'min': float(np.min(residuals)), 'max': float(np.max(residuals))},
    'error_by_segment': error_by_segment,
    'prediction_summary': {
      'count': int(n), 'mean': float(np.mean(preds)),
      'std': float(np.std(preds)), 'min': float(np.min(preds)),
      'max': float(np.max(preds)),
      'mean_actual': float(np.mean(y_arr))},
    'warnings': warnings}


def lime_explain(model, X, params=None):
  '''Per-row top features via LIME, permutation surrogate.

  Returns ``{'method', 'rows', 'warnings'}`` where
  ``rows`` is ``[{'row', 'top_features': [{'feature',
  'weight'}]}]`` capped at ``max_rows``. When ``lime``
  is unavailable the surrogate ranks features by
  global (prediction-change) permutation importance,
  documented by ``method='permutation_surrogate'``.
  '''
  params = flatten_params(params, 'interpretability_params')
  warnings = []
  if not _is_valid_X(X) or _row_count(X) == 0:
    warnings.append('empty/invalid data: no rows for LIME')
    LOG.warning(warnings[-1])
    return {'method': 'none', 'rows': [], 'warnings': warnings}
  frame = _frame(X)
  max_rows = _int(params.get('max_rows',
                             params.get('max_per_row_rows', 20)), 20)
  top_k = _int(params.get('top_k', 10), 10)
  try:
    from lime.lime_tabular import LimeTabularExplainer  # lazy
    explainer = LimeTabularExplainer(
      frame.to_numpy(dtype=float), feature_names=list(frame.columns),
      class_names=['target'], mode='regression',
      random_state=_seed(params))
    rows = []
    for idx in range(min(len(frame), max_rows)):
      exp = explainer.explain_instance(
        frame.iloc[idx].to_numpy(dtype=float),
        lambda data: _predict(model, _frame(data, frame.columns)),
        num_features=max(top_k, 1))
      weight = dict(exp.as_list())
      ranked = sorted(weight.items(), key=lambda kv: -abs(kv[1]))
      rows.append({'row': int(idx),
                   'top_features': [
                     {'feature': str(f), 'weight': float(w)}
                     for f, w in ranked[:top_k]]})
    return {'method': 'lime', 'rows': rows, 'warnings': warnings}
  except Exception as exc:  # noqa: BLE001 - surrogate path
    warnings.append('lime unavailable, using permutation '
                    'surrogate: %s' % exc)
    LOG.warning(warnings[-1])
  try:
    ranked = permutation_importance(model, frame, None, params)
    if not ranked:
      raise RuntimeError('permutation surrogate returned no '
                         'importance')
    top = [{'feature': e['feature'], 'weight': e['importance']}
           for e in ranked[:max(top_k, 0)]]
    rows = [{'row': int(idx), 'top_features': list(top)}
            for idx in range(min(len(frame), max_rows))]
  except Exception as exc:  # noqa: BLE001 - hostile model
    warnings.append('explanation failed for model: %s' % exc)
    LOG.warning(warnings[-1])
    return {'method': 'none', 'rows': [], 'warnings': warnings}
  return {'method': 'permutation_surrogate', 'rows': rows,
          'warnings': warnings}


def explain_model(model, X, y, params=None):
  '''Pipeline node: SHAP + permutation explanations.'''
  params = flatten_params(params, 'interpretability_params')
  return {'shap': shap_explain(model, X, params),
          'permutation': permutation_importance(model, X, y, params)}


def interpret_features(model, X, params=None):
  '''Pipeline node: PDP per feature + LIME rows.'''
  params = flatten_params(params, 'interpretability_params')
  frame = _frame(X) if _row_count(X) else None
  features = params.get('pdp_features') or (
    list(frame.columns) if frame is not None else [])
  pdp = {str(f): pdp_1d(model, X, f, params) for f in features}
  return {'pdp': pdp, 'lime': lime_explain(model, X, params)}


def run_diagnostics(model, X, y, params=None):
  '''Pipeline node: diagnostics report.'''
  return diagnostics(model, X, y, params)


def _add_chart(figs, name, func, *args, **kwargs):
  '''Add a chart figure when the function returns one.

  Figures leaked by a failing chart call are closed,
  mirroring ``modeling.charts`` defensive behaviour.
  '''
  import matplotlib.pyplot as plt
  before = set(plt.get_fignums())
  try:
    fig = func(*args, **kwargs)
    if fig is not None:
      figs[name] = fig
  except Exception as exc:  # noqa: BLE001 - skip bad chart
    LOG.warning('chart %r skipped: %s' % (name, exc))
    for num in set(plt.get_fignums()) - before:
      plt.close(num)


def _chart_section(name, assemble):
  '''Run one chart assembly, skipping it on malformed input.'''
  try:
    assemble()
  except Exception as exc:  # noqa: BLE001 - skip bad chart data
    LOG.warning('chart %r skipped: %s' % (name, exc))


def _dict_or_empty(value):
  '''Return value when it is a dict, else {} (with a warning).'''
  if isinstance(value, dict):
    return value
  if value:
    LOG.warning('ignoring malformed chart input: %r' % (value,))
  return {}


def _list_or_empty(value):
  '''Return value when it is a list, else [] (with a warning).'''
  if isinstance(value, list):
    return value
  if value:
    LOG.warning('ignoring malformed chart input: %r' % (value,))
  return []


def make_interpretability_charts(explain_outputs, params=None):
  '''Pipeline node: interpretability chart figures.

  Args:
    explain_outputs: Output of :func:`explain_model`,
      optionally merged with ``interpret_features``
      output (``pdp`` / ``lime`` keys honoured when
      present). Standalone tree/decision-boundary
      charts need model/X and are not built here.
    params: Optional dict; see module docstring.

  Returns:
    Dict of chart name to (closed) matplotlib Figure,
    only for charts with sufficient data.
  '''
  from forecast_ml_framework.pipelines.interpretability import (
    charts,
  )
  outputs = _dict_or_empty(explain_outputs)
  shap_out = _dict_or_empty(outputs.get('shap'))
  per_row = _list_or_empty(shap_out.get('per_row'))
  global_imp = _list_or_empty(shap_out.get('global_importance'))
  figs = {}

  def _names_and_matrix():
    '''Build feature names + SHAP matrix from valid entries.'''
    names, clean_imp = [], []
    for entry in global_imp:
      if isinstance(entry, dict) and 'feature' in entry:
        names.append(str(entry['feature']))
        clean_imp.append(entry)
    rows = []
    for row in per_row:
      if not isinstance(row, dict):
        continue
      values = row.get('shap_values')
      if not isinstance(values, dict):
        continue
      try:
        rows.append([float(values.get(n, 0.0)) for n in names])
      except (TypeError, ValueError):
        continue
    matrix = np.array(rows) if rows and names else None
    return names, clean_imp, matrix

  def _shap_charts():
    names, clean_imp, matrix = _names_and_matrix()
    if matrix is not None and len(matrix) >= 2:
      _add_chart(figs, 'shap_beeswarm', charts.shap_beeswarm,
                 matrix, names)
      _add_chart(figs, 'shap_heatmap', charts.shap_heatmap,
                 per_row, names)
      _add_chart(figs, 'shap_dependence', charts.shap_dependence,
                 matrix, None, 0)
      _add_chart(figs, 'shap_waterfall', charts.shap_waterfall,
                 matrix[0], names)
      _add_chart(figs, 'shap_force', charts.shap_force,
                 matrix[0], names)
    _add_chart(figs, 'shap_bar', charts.shap_bar, clean_imp)

  _chart_section('shap', _shap_charts)

  def _perm_chart():
    _add_chart(figs, 'permutation_importance_bar',
               charts.permutation_importance_bar,
               _list_or_empty(outputs.get('permutation')))

  _chart_section('permutation_importance_bar', _perm_chart)

  def _pdp_chart():
    pdp = _dict_or_empty(outputs.get('pdp'))
    if pdp:
      first = next(iter(pdp.values()))
      _add_chart(figs, 'pdp_1d', charts.pdp_1d_chart, first)

  _chart_section('pdp_1d', _pdp_chart)

  def _lime_chart():
    lime = _dict_or_empty(outputs.get('lime'))
    if lime:
      _add_chart(figs, 'lime_bar', charts.lime_bar,
                 _list_or_empty(lime.get('rows')))

  _chart_section('lime_bar', _lime_chart)
  return figs


def make_diagnostics_charts(diagnostics, params=None):
  '''Pipeline node: diagnostics chart figures.

  Args:
    diagnostics: Output of :func:`diagnostics`,
      optionally carrying ``y_true`` / ``y_pred`` /
      ``proba`` / score histories for pair-based
      charts (also accepted via ``params``).
    params: Optional dict; see module docstring.

  Returns:
    Dict of chart name to (closed) matplotlib Figure,
    only for charts with sufficient data.
  '''
  from forecast_ml_framework.pipelines.interpretability import (
    charts,
  )
  params = flatten_params(params, 'interpretability_params')
  report = _dict_or_empty(diagnostics)

  def _get(key):
    '''Look up a value from the report, falling back to params.

    Args:
      key: Lookup key to resolve.

    Returns:
      The report value when present, otherwise the
        params value (possibly None).
    '''
    return report.get(key) if report.get(key) is not None \
      else params.get(key)

  figs = {}

  def _chart(name, func, *args):
    '''Add one diagnostics chart; malformed input is skipped.'''
    _chart_section(name, lambda: _add_chart(figs, name, func, *args))

  _chart('calibration_curve', charts.calibration_curve, report)
  _chart('error_by_segment_bar', charts.error_by_segment_bar,
         _list_or_empty(report.get('error_by_segment')))
  y_true, y_pred = _get('y_true'), _get('y_pred')
  if y_true is not None and y_pred is not None:
    _chart('residual_vs_fitted', charts.residual_vs_fitted,
           y_true, y_pred)
    _chart('residual_hist_qq', charts.residual_hist_qq,
           y_true, y_pred)
    _chart('scale_location', charts.scale_location, y_true, y_pred)
    _chart('prediction_distribution_by_class',
           charts.prediction_distribution_by_class, y_true, y_pred)
    _chart('residuals_over_time', charts.residuals_over_time_chart,
           y_true, y_pred)
    _chart('cooks_distance', charts.cooks_distance,
           y_true, y_pred, _get('X'))
    _chart('rec_curve', charts.rec_curve, y_true, y_pred)
  _chart('confidence_histogram', charts.confidence_histogram,
         _get('proba'))
  _chart('learning_curve', charts.learning_curve,
         _get('train_scores'), _get('val_scores'))
  _chart('validation_curve', charts.validation_curve,
         _get('param_values'), _get('train_scores'),
         _get('val_scores'))
  _chart('loss_vs_epoch', charts.loss_vs_epoch,
         _get('train_loss'), _get('val_loss'))
  _chart('cv_box', charts.cv_box, _get('cv_scores'))
  _chart('overfitting_gap', charts.overfitting_gap,
         _get('train_scores'), _get('val_scores'))
  return figs
