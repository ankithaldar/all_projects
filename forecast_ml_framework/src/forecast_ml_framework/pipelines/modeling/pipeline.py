#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Kedro pipeline for the modeling phase.

Params (params:modeling_params, a dict):
  task_type: 'classification' (default) or 'regression'.
  target_col: target column name (default 'target').
  feature_cols: list of features or None for all others.
  models_to_use: adapter names, e.g. ['sklearn', 'xgboost'].
    Supported: sklearn, xgboost, lightgbm, catboost, h2o_automl.
  selection_metric: metric used to pick the best model,
    e.g. 'f1_weighted' (default) or 'rmse'.
  selection_metric_higher_better: bool or None (inferred).
  random_state: int seed (default 42).
  test_size: float test fraction (default 0.2).
  sampling_params: {method: 'stratified'|'random',
    test_size: float, stratify_col: str}.
  calibration: {method: 'sigmoid'|'isotonic'|'betacal'|'none',
    cv: int} (classification only).
  loss_function: plain string, top-level default for every
    model in models_to_use. Per-model value inside
    model_params[adapter_name]['loss_function'] wins over
    this top-level default. Mapped per adapter via
    LOSS_REGISTRY / get_loss() in adapters.py:
      sklearn: kwarg 'loss' (log_loss, squared_error,
        absolute_error, huber, quantile,
        epsilon_insensitive, poisson, gamma, tweedie).
      xgboost: kwarg 'objective' (binary:logistic,
        reg:squarederror, reg:gamma, reg:tweedie,
        reg:absoluteerror, count:poisson).
      lightgbm: kwarg 'objective' (binary, regression,
        regression_l1, huber, fair, poisson, gamma,
        tweedie, quantile).
      catboost: kwarg 'loss_function' (Logloss, RMSE,
        MAE, Poisson, Gamma, Tweedie, Quantile).
      h2o_automl: kwarg 'loss_function' (logloss,
        squared_error, poisson, gamma, tweedie, huber).
    Aliases accepted case-insensitively (logloss,
    squared_error/mse/rmse, mae, poisson, ...); native
    values accepted as-is. Custom loss: dotted importable
    string resolving to a callable, e.g.
    'forecast_ml_framework.losses.huber_loss' or
    'sklearn.metrics.mean_absolute_error', resolved via
    resolve_handle() and passed through as loss kwarg.
    Unknown loss -> ValueError listing valid losses.
    Explicit loss_function overrides a per-model
    'objective' (xgboost/lightgbm) or 'loss' (sklearn);
    a recognised 'objective' is normalised, an
    unrecognised native objective passes through.
    No loss_function -> estimator defaults unchanged.
  objective: per-model native objective (xgboost/
    lightgbm only), e.g. 'binary:logistic'. Normalised
    via the registry when recognised, else passed
    through unchanged for back-compat.
  model_params: {adapter_name: {hyperparameters}}.
    May include loss_function/objective as above, plus
    model_handle. Flat {hyperparameters} (no adapter
    nesting) is also accepted for a single model.
  model_handle: dotted importable string (preferred form,
    e.g. 'sklearn.linear_model.LogisticRegression',
    'xgboost.XGBClassifier') or short adapter name in
    models_to_use. Also accepted per-model in model_params
    or as a models_to_use entry itself.
  calibration_handle: dotted importable string, e.g.
    'sklearn.calibration.CalibratedClassifierCV' or
    'betacal.BetaCalibration'.
  tuning_handle: dotted importable string, resolved and
    validated via resolve_handle() when present.
  bucket_segment_col: column for per-bucket metrics/charts.
  shap_params: {method: 'shap'|'permutation'|'auto',
    n_repeats: int, sample_rows: int, random_state: int}.

Datasets (split-first, A9, breaking at 0.1.0):
  train_data_selected + test_data_selected (in, already
  split in data_prep) + split_meta_raw (in, data_prep
  honest split meta, merged as raw_* keys) ->
  x_train/x_test/y_train/y_test/split_meta
  (sampling_method='presplit', no shuffle)
  -> best_model/per_model_metrics/best_model_name.
  ``modeling_input`` is deprecated and no longer
  consumed by this pipeline (kept in the catalog for
  back-compat only).
  -> calibrated_model/calibration_meta
  -> evaluation_metrics, plus y_pred/y_pred_proba
  -> lift_table, promotion (parity: lift table from
  y_test/y_pred/y_pred_proba with modeling_params
  lift_threshold/model_threshold; promotion gate from
  selected_ntile score vs params old_score)
  -> charts (dict of figures), shap_artifact (dict).

  ``old_score`` is explicit in ``ModelingParams``
  (``Optional[float] = None``); a missing/None,
  non-positive, or non-finite incumbent never promotes
  (see ``validation.should_promote``).
'''

import numpy as np
from kedro.pipeline import Pipeline, node

from forecast_ml_framework.pipelines.interpretability.nodes import (
  shap_explain,
)
from forecast_ml_framework.pipelines.modeling.adapters import (
  train_models,
)
from forecast_ml_framework.pipelines.modeling.calibration import (
  calibrate_model,
)
from forecast_ml_framework.pipelines.modeling.charts import (
  make_charts,
)
from forecast_ml_framework.pipelines.modeling.data_split import (
  split_data,
)
from forecast_ml_framework.pipelines.modeling.evaluate import (
  evaluate_model,
)
from forecast_ml_framework.pipelines.modeling.lift_table import (
  compute_lift_table,
)
from forecast_ml_framework.pipelines.modeling.ntile import (
  normalize_task_kind,
)
from forecast_ml_framework.pipelines.modeling.threshold import (
  select_threshold,
)
from forecast_ml_framework.pipelines.modeling.validation import (
  should_promote,
)


def split_presplit_data(train_df, test_df, params=None,
                        split_meta_raw=None):
  '''Split pre-split train/test frames into X/y (presplit).

  Split-first step (A9): data_prep already split raw
  data and fitted preprocessing/selection on train
  only, so this node does not split or shuffle at
  all -- it slices the two selected frames into
  features/target and reports
  ``sampling_method='presplit'``. When
  ``feature_cols`` is None, id columns are excluded
  by default (ML-24c): the ``id_cols`` param when
  given, otherwise ``['id']``.

  R6: when ``split_meta_raw`` (the data_prep
  ``split_meta_raw`` catalog dataset: actual
  ``sampling_method``, ``requested_method``,
  ``shuffle``, ``time_col``) is given, it is merged
  into the returned meta as ``raw_*`` keys, so the
  modeling meta records how data_prep actually split
  the raw frame. ``sampling_method`` itself stays
  ``'presplit'`` (this node did not split).

  For direct-use compat ``split_data`` stays
  available: called as ``(df, params)`` (second
  argument a params dict, no third argument) this
  function delegates to ``split_data``. The Kedro
  pipeline never uses the legacy path -- it wires
  only the presplit inputs.

  Args:
    train_df: Pre-split, preprocessed, selected
      train frame (legacy path: the full frame).
    test_df: Pre-split test frame, or the params
      dict in the legacy direct-use call.
    params: Modeling-params dict (presplit path).
    split_meta_raw: Optional data_prep split meta
      dict to merge as ``raw_*`` keys.

  Returns:
    Tuple (x_train, x_test, y_train, y_test, meta).

  Raises:
    ValueError: If either frame is empty/missing or
      the target column is absent.
  '''
  if params is None and isinstance(test_df, dict):
    return split_data(train_df, test_df)
  if train_df is None or test_df is None:
    raise ValueError(
      'split_presplit_data requires non-empty train '
      'and test frames')
  if len(train_df) == 0 or len(test_df) == 0:
    raise ValueError(
      'split_presplit_data requires non-empty train '
      'and test frames')
  params = params or {}
  sampling = params.get('sampling_params', {}) or {}
  target = params.get('target_col', 'target')
  if target not in train_df.columns:
    raise ValueError('target column not found: %s' % target)
  if target not in test_df.columns:
    raise ValueError('target column not found: %s' % target)
  features = params.get('feature_cols')
  if features is None:
    if params.get('id_cols') is not None:
      raw_ids = params.get('id_cols')
    elif params.get('id_col') is not None:
      raw_ids = params.get('id_col')
    else:
      raw_ids = ['id']
    id_set = ({raw_ids} if isinstance(raw_ids, str)
              else set(raw_ids or []))
    features = [c for c in train_df.columns
                if c != target and c not in id_set]
  else:
    features = [c for c in features if c != target]
  x_train = train_df[features]
  x_test = test_df[features]
  y_train = np.asarray(train_df[target])
  y_test = np.asarray(test_df[target])
  test_size = sampling.get('test_size', params.get('test_size'))
  if test_size is None:
    total = len(y_train) + len(y_test)
    test_size = float(len(y_test) / total) if total else 0.0
  raw = split_meta_raw or {}
  raw = raw if isinstance(raw, dict) else {}
  meta = {
    'task_type': params.get('task_type', 'classification'),
    'test_size': test_size,
    'random_state': params.get(
      'random_state', sampling.get('random_state', 42)),
    'sampling_method': 'presplit',
    'requested_method': 'presplit',
    'shuffle': False,
    'time_col': params.get('time_col', sampling.get('time_col')),
    'n_train': int(len(y_train)),
    'n_test': int(len(y_test)),
    'raw_sampling_method': raw.get('sampling_method'),
    'raw_requested_method': raw.get('requested_method'),
    'raw_shuffle': raw.get('shuffle'),
    'raw_time_col': raw.get('time_col'),
  }
  return x_train, x_test, y_train, y_test, meta


def _predict(model, x_test):
  '''Predict labels with the calibrated model.'''
  return model.predict(x_test)


def _predict_proba(model, x_test, task_type):
  '''Predict proba, or None for regression.'''
  if normalize_task_kind(task_type) != 'classification':
    return None
  return model.predict_proba(x_test)


def build_lift_table(y_test, y_pred, y_proba, params=None,
                     threshold=None):
  '''Build the per-decile lift table from test predictions.

  Args:
    y_test: True labels or values.
    y_pred: Predicted labels or values.
    y_proba: Probabilities (2 columns, 1 column, 1-D) or
      None for regression.
    params: modeling_params dict; ``lift_threshold`` (or its
      alias ``model_threshold``) is passed through as the
      lift threshold. A numeric/None value is taken as the
      threshold directly.
    threshold: Explicit threshold in [0, 1]; wins over the
      value in ``params`` when given.

  Returns:
    Lift DataFrame from ``compute_lift_table`` (empty when
    without proba).
  '''
  if threshold is None:
    if isinstance(params, dict):
      threshold = params.get('lift_threshold')
      if threshold is None:
        threshold = params.get('model_threshold')
    elif params is not None:
      threshold = params
  if y_proba is None:
    return compute_lift_table([], [], [])
  proba = np.asarray(y_proba, dtype=float)
  if proba.ndim == 2:
    proba = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
  return compute_lift_table(y_test, y_pred, proba,
                            threshold=threshold)


def build_promotion(evaluation_metrics, params):
  '''Decide promotion from selected_ntile vs old_score.

  Args:
    evaluation_metrics: ``evaluate_model`` result dict.
    params: modeling_params; ``old_score`` (incumbent
      score) and ``task_type`` drive ``should_promote``.

  Returns:
    Dict with ``promote``, ``deviation_pct``, ``new_score``,
    ``old_score``, ``eval_metric``, ``metric`` and
    ``task_type``. Missing old_score or selected_ntile
    never promotes.
  '''
  params = params or {}
  selected = (evaluation_metrics or {}).get('selected_ntile') or {}
  new_score = selected.get('score')
  old_score = params.get('old_score')
  metric = selected.get('metric', '')
  task = params.get('task_type', 'classification')
  promote, deviation = should_promote(new_score, old_score, task, metric)
  return {
    'promote': bool(promote),
    'deviation_pct': float(deviation),
    'new_score': new_score,
    'old_score': old_score,
    'eval_metric': selected.get('eval_metric'),
    'metric': metric,
    'task_type': task,
  }


def explain_model(model, x, y, params):
  '''Adapt interpretability SHAP output to the legacy artifact.

  Args:
    model: Fitted calibrated model.
    x: Features to explain.
    y: Target values (unused; kept for node signature).
    params: modeling_params; ``shap_params`` keys are
      mapped onto interpretability params.

  Returns:
    Dict with ``method``, ``global_importance`` (name to
    value), ``per_row`` (row index to SHAP values), and
    ``shap_available``.
  '''
  _ = y
  cfg = (params or {}).get('shap_params', {}) or {}
  interp_params = {
    'max_rows': cfg.get('sample_rows', 100),
    'n_repeats': cfg.get('n_repeats', 5),
    'random_state': cfg.get('random_state', 42),
  }
  out = shap_explain(model, x, interp_params)
  method = out.get('method', 'none')
  global_importance = {
    str(entry['feature']): float(entry['importance'])
    for entry in out.get('global_importance') or []
    if isinstance(entry, dict) and 'feature' in entry}
  per_row = {}
  for row in out.get('per_row') or []:
    if not isinstance(row, dict):
      continue
    values = row.get('shap_values') or {}
    per_row[str(row.get('row', len(per_row)))] = {
      str(name): float(value) for name, value in values.items()}
  return {'method': method if method != 'permutation_fallback'
          else 'permutation',
          'global_importance': global_importance,
          'per_row': per_row,
          'shap_available': method == 'shap'}


def _threshold_method(params):
  '''Extract the threshold method from modeling_params.'''
  if isinstance(params, str):
    return params or 'pr_curve'
  if not isinstance(params, dict):
    return 'pr_curve'
  for key in ('threshold_method', 'threshold_select_method',
              'select_threshold_method'):
    if params.get(key):
      return params[key]
  nested = params.get('threshold_params')
  if isinstance(nested, dict) and nested.get('method'):
    return nested['method']
  threshold = params.get('threshold')
  if isinstance(threshold, dict) and threshold.get('method'):
    return threshold['method']
  return 'pr_curve'


def select_model_threshold(y_test, y_proba, params=None):
  '''Select a classification threshold from test proba.

  Args:
    y_test: True labels.
    y_proba: Probabilities (2 columns, 1 column, 1-D) or
      None for regression (returns the 0.5 default).
    params: modeling_params dict; the method comes from
      ``threshold_method`` (aliases/nested forms accepted,
      default 'pr_curve'). A plain string is taken as method.

  Returns:
    Selected threshold as float via ``select_threshold``.
  '''
  if y_proba is None:
    return 0.5
  proba = np.asarray(y_proba, dtype=float)
  if proba.ndim == 2:
    proba = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
  return float(select_threshold(y_test, proba,
                                _threshold_method(params)))


# Alias matching the node name.
build_selected_threshold = select_model_threshold


def build_tuning_report(per_model_metrics):
  '''Extract per-model tuning reports from per_model_metrics.

  Args:
    per_model_metrics: Dict from ``train_models``; models
      with a ``tuning`` entry contribute it to the report.

  Returns:
    Dict mapping model name to its tuning report dict
    (empty when no model was tuned).
  '''
  report = {}
  for name, metrics in (per_model_metrics or {}).items():
    if isinstance(metrics, dict) and metrics.get('tuning') is not None:
      report[str(name)] = metrics['tuning']
  return report


def create_pipeline(**kwargs):
  '''Create the modeling Kedro pipeline.

  Args:
    **kwargs: Forwarded to Pipeline (e.g. tags, namespace).

  Returns:
    Kedro Pipeline: split -> train -> calibrate -> evaluate
    -> charts + shap.
  '''
  pipe = Pipeline([
    node(func=split_presplit_data,
         inputs=['train_data_selected', 'test_data_selected',
                 'params:modeling_params', 'split_meta_raw'],
         outputs=['x_train', 'x_test', 'y_train', 'y_test',
                  'split_meta'],
         name='split_data_node'),
    node(func=train_models,
         inputs=['x_train', 'y_train', 'x_test', 'y_test',
                 'params:modeling_params'],
         outputs=['best_model', 'per_model_metrics',
                  'best_model_name'],
         name='train_models_node'),
    node(func=calibrate_model,
         inputs=['best_model', 'x_train', 'y_train',
                 'params:modeling_params'],
         outputs=['calibrated_model', 'calibration_meta'],
         name='calibrate_model_node'),
    node(func=_predict,
         inputs=['calibrated_model', 'x_test'],
         outputs='y_pred',
         name='predict_node'),
    node(func=_predict_proba,
         inputs=['calibrated_model', 'x_test',
                 'params:modeling_params.task_type'],
         outputs='y_pred_proba',
         name='predict_proba_node'),
    node(func=evaluate_model,
         inputs=['y_test', 'y_pred', 'y_pred_proba', 'x_test',
                 'params:modeling_params'],
         outputs='evaluation_metrics',
         name='evaluate_model_node'),
    node(func=build_lift_table,
         inputs=['y_test', 'y_pred', 'y_pred_proba',
                 'params:modeling_params'],
         outputs='lift_table',
         name='build_lift_table_node'),
    node(func=build_promotion,
         inputs=['evaluation_metrics', 'params:modeling_params'],
         outputs='promotion',
         name='build_promotion_node'),
    node(func=select_model_threshold,
         inputs=['y_test', 'y_pred_proba',
                 'params:modeling_params'],
         outputs='selected_threshold',
         name='select_model_threshold_node'),
    node(func=build_tuning_report,
         inputs='per_model_metrics',
         outputs='tuning_report',
         name='build_tuning_report_node'),
    node(func=make_charts,
         inputs=['y_test', 'y_pred', 'y_pred_proba', 'x_test',
                 'params:modeling_params'],
         outputs='charts',
         name='make_charts_node'),
    node(func=explain_model,
         inputs=['calibrated_model', 'x_test', 'y_test',
                 'params:modeling_params'],
         outputs='shap_artifact',
         name='explain_model_node'),
  ], **kwargs)
  return pipe
