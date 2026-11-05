#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Metric computation for classification and regression.'''
# pylint: disable=import-outside-toplevel,broad-exception-caught

import warnings

import numpy as np


def _safe(fn, *args, **kwargs):
  '''Return fn result as float, or nan on failure.

  Args:
    fn: Callable to invoke.
    *args: Positional arguments for fn.
    **kwargs: Keyword arguments for fn.

  Returns:
    Float result of fn, or nan if fn raises.
  '''
  try:
    return float(fn(*args, **kwargs))
  except Exception as exc:  # noqa: BLE001 - never crash
    warnings.warn(f'metric failed: {exc}', stacklevel=3)
    return float('nan')


def _safe_list(fn, *args, **kwargs):
  '''Return fn result as list of floats, or [] on failure.

  Args:
    fn: Callable to invoke.
    *args: Positional arguments for fn.
    **kwargs: Keyword arguments for fn.

  Returns:
    List of floats, or empty list if fn raises.
  '''
  try:
    return [float(v) for v in fn(*args, **kwargs)]
  except Exception as exc:  # noqa: BLE001
    warnings.warn(f'metric failed: {exc}', stacklevel=3)
    return []


def _safe_dict(fn, *args, **kwargs):
  '''Return fn result as dict, or {} on failure.

  Args:
    fn: Callable to invoke.
    *args: Positional arguments for fn.
    **kwargs: Keyword arguments for fn.

  Returns:
    Dict result of fn, or empty dict if fn raises.
  '''
  try:
    result = fn(*args, **kwargs)
    return dict(result)
  except Exception as exc:  # noqa: BLE001
    warnings.warn(f'metric failed: {exc}', stacklevel=3)
    return {}


def _labels(y_true, y_pred):
  '''Return sorted unique labels from true and predicted.

  Args:
    y_true: True labels.
    y_pred: Predicted labels.

  Returns:
    Sorted list of unique labels.
  '''
  vals = np.union1d(np.asarray(y_true), np.asarray(y_pred))
  return sorted(vals.tolist())


def _pos_proba(proba, y_true):
  '''Return positive-class probability column.

  Args:
    proba: Probability array.
    y_true: True labels (unused, for symmetry).

  Returns:
    One-dimensional positive probability array, or nan array.
  '''
  _ = y_true  # kept for symmetry with callers
  proba = np.asarray(proba, dtype=float)
  if proba.ndim == 1:
    return proba
  if proba.shape[1] >= 2:
    return proba[:, 1]
  return proba[:, 0]


def _calibration(y_true, proba, n_bins=10):
  '''Compute ECE, MCE and brier decomposition.

  Args:
    y_true: True binary labels.
    proba: Probability array.
    n_bins: Number of equal-width bins.

  Returns:
    Tuple of (ece, mce, decomposition dict).
  '''
  try:
    pos = np.asarray(_pos_proba(proba, y_true), dtype=float)
    y = np.asarray(y_true, dtype=float)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    mce = 0.0
    reliability = 0.0
    resolution = 0.0
    base = y.mean()
    for lo, hi in zip(edges[:-1], edges[1:]):
      if hi < 1:
        mask = (pos >= lo) & (pos < hi)
      else:
        mask = (pos >= lo) & (pos <= hi)
      if mask.sum():
        weight = mask.mean()
        gap = abs(pos[mask].mean() - y[mask].mean())
        ece += weight * gap
        mce = max(mce, gap)
        reliability += weight * (pos[mask].mean() - y[mask].mean()) ** 2
        resolution += weight * (y[mask].mean() - base) ** 2
    uncertainty = base * (1 - base)
    decomp = {'reliability': float(reliability),
              'resolution': float(resolution),
              'uncertainty': float(uncertainty)}
    return float(ece), float(mce), decomp
  except Exception as exc:  # noqa: BLE001
    warnings.warn(f'calibration failed: {exc}', stacklevel=3)
    nan_d = {'reliability': float('nan'),
             'resolution': float('nan'),
             'uncertainty': float('nan')}
    return float('nan'), float('nan'), nan_d


def _ece(y_true, proba, n_bins=10):
  '''Compute expected calibration error.

  Args:
    y_true: True binary labels.
    proba: Two-column probability array.
    n_bins: Number of equal-width bins.

  Returns:
    Expected calibration error as float, or nan.
  '''
  return _calibration(y_true, proba, n_bins)[0]


def _top_k(y_true, proba, k):
  '''Compute top-k accuracy from probabilities.

  Args:
    y_true: True labels.
    proba: Probability array.
    k: Cutoff k.

  Returns:
    Top-k accuracy as float.
  '''
  proba = np.asarray(proba, dtype=float)
  if proba.ndim == 1:
    proba = np.column_stack([1 - proba, proba])
  labels = _labels(y_true, y_true)
  # Map labels to columns assuming sorted-label column order.
  # For binary 0/1 this matches the proba layout used here.
  order = np.argsort(-proba, axis=1)
  top = order[:, :max(1, min(k, proba.shape[1]))]
  hits = 0
  for true, row in zip(np.asarray(y_true), top):
    if true in labels:
      idx = labels.index(true)
      hits += int(idx in row)
    else:
      hits += 0
  return float(hits / max(len(y_true), 1))


def _log_loss(y_true, proba):
  '''Compute log loss, handling single-class labels.

  Args:
    y_true: True labels.
    proba: Probability array.

  Returns:
    Log loss as float.
  '''
  from sklearn.metrics import log_loss
  proba = np.asarray(proba, dtype=float)
  if proba.ndim == 1:
    proba = np.column_stack([1 - proba, proba])
  try:
    return float(log_loss(y_true, proba))
  except Exception:  # noqa: BLE001
    labels = _labels(y_true, y_true)
    if len(labels) == 1 and proba.shape[1] == 2:
      # Single class present: keep only its column, renormalised.
      col = labels[0]
      proba = proba[:, col:col + 1] if col in (0, 1) else proba
      proba = np.clip(proba, 1e-15, 1)
    return float(log_loss(y_true, proba, labels=labels))


def _hinge(y_true, proba):
  '''Compute hinge loss from probabilities.

  Args:
    y_true: True labels.
    proba: Probability array.

  Returns:
    Hinge loss as float.
  '''
  from sklearn.metrics import hinge_loss
  proba = np.asarray(proba, dtype=float)
  y = np.asarray(y_true)
  if proba.ndim == 2 and proba.shape[1] > 2:
    return float(hinge_loss(y, proba))
  pos = _pos_proba(proba, y)
  labels = _labels(y, y)
  pos_label = labels[-1] if labels else 1
  signed = np.where(y == pos_label, 1, -1)
  return float(hinge_loss(signed, pos * 2 - 1))


def _classification(y_true, y_pred, proba):
  '''Compute all approved classification metrics.

  Args:
    y_true: True labels.
    y_pred: Predicted labels.
    proba: Probability array.

  Returns:
    Dict of classification metric name to value.
  '''
  from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, classification_report, cohen_kappa_score,
    confusion_matrix, f1_score, jaccard_score,
    matthews_corrcoef, precision_score, recall_score, roc_auc_score,
    zero_one_loss,
  )
  out = {}
  y_true = np.asarray(y_true)
  y_pred = np.asarray(y_pred)
  proba = np.asarray(proba, dtype=float)
  if proba.ndim == 1:
    proba = np.column_stack([1 - proba, proba])
  is_binary = proba.shape[1] == 2
  pos = proba[:, 1] if is_binary else proba
  out['accuracy'] = _safe(accuracy_score, y_true, y_pred)
  out['balanced_accuracy'] = _safe(
      balanced_accuracy_score, y_true, y_pred)
  for avg in ('macro', 'micro', 'weighted'):
    out[f'precision_{avg}'] = _safe(
        precision_score, y_true, y_pred, average=avg,
        zero_division=0)
    out[f'recall_{avg}'] = _safe(
        recall_score, y_true, y_pred, average=avg, zero_division=0)
    out[f'f1_{avg}'] = _safe(
        f1_score, y_true, y_pred, average=avg, zero_division=0)
    out[f'jaccard_{avg}'] = _safe(
        jaccard_score, y_true, y_pred, average=avg, zero_division=0)
  out['per_class_precision'] = _safe_list(
      precision_score, y_true, y_pred, average=None, zero_division=0)
  out['per_class_recall'] = _safe_list(
      recall_score, y_true, y_pred, average=None, zero_division=0)
  out['per_class_f1'] = _safe_list(
      f1_score, y_true, y_pred, average=None, zero_division=0)
  # ROC AUC (ovr / ovo where multiclass).
  if is_binary:
    out['roc_auc'] = _safe(roc_auc_score, y_true, pos)
    out['roc_auc_ovr'] = out['roc_auc']
    out['roc_auc_ovo'] = out['roc_auc']
    out['pr_auc'] = _safe(average_precision_score, y_true, pos)
    out['average_precision'] = out['pr_auc']
    out['pr_auc_ovr'] = out['pr_auc']
  else:
    out['roc_auc_ovr'] = _safe(
        roc_auc_score, y_true, proba, multi_class='ovr',
        average='macro')
    out['roc_auc_ovo'] = _safe(
        roc_auc_score, y_true, proba, multi_class='ovo',
        average='macro')
    out['roc_auc'] = out['roc_auc_ovr']
    out['pr_auc'] = _safe(
        average_precision_score, y_true, proba, average='macro')
    out['average_precision'] = out['pr_auc']
    out['pr_auc_ovr'] = out['pr_auc']
  out['log_loss'] = _safe(_log_loss, y_true, proba)
  if is_binary:
    out['brier'] = _safe(brier_score_loss, y_true, pos)
  else:
    out['brier'] = float('nan')
    warnings.warn('brier undefined for multiclass', stacklevel=2)
  out['brier_score'] = out['brier']
  out['kappa'] = _safe(cohen_kappa_score, y_true, y_pred)
  out['kappa_linear'] = _safe(
      cohen_kappa_score, y_true, y_pred, weights='linear')
  out['kappa_quadratic'] = _safe(
      cohen_kappa_score, y_true, y_pred, weights='quadratic')
  out['cohen_kappa_linear'] = out['kappa_linear']
  out['cohen_kappa_quadratic'] = out['kappa_quadratic']
  out['cohen_kappa'] = out['kappa']
  out['mcc'] = _safe(matthews_corrcoef, y_true, y_pred)
  out['matthews_corrcoef'] = out['mcc']
  out['zero_one_loss'] = _safe(zero_one_loss, y_true, y_pred)
  out['hinge_loss'] = _safe(_hinge, y_true, proba)
  out['top_k_accuracy_2'] = _safe(_top_k, y_true, proba, 2)
  out['top_k_accuracy_3'] = _safe(_top_k, y_true, proba, 3)
  out['jaccard'] = out.get('jaccard_macro', float('nan'))
  out['jaccard_binary'] = _safe(
      jaccard_score, y_true, y_pred, average='binary',
      zero_division=0)
  # Binary confusion-matrix derived rates.
  try:
    cm = np.asarray(confusion_matrix(y_true, y_pred), dtype=float)
  except Exception as exc:  # noqa: BLE001
    warnings.warn(f'confusion_matrix failed: {exc}', stacklevel=2)
    cm = np.zeros((0, 0))
  if cm.shape == (2, 2):
    tn, fp, fn, tp = cm.ravel()
    out['sensitivity'] = float(tp / (tp + fn)) if tp + fn else 0.0
    out['specificity'] = float(tn / (tn + fp)) if tn + fp else 0.0
    out['npv'] = float(tn / (tn + fn)) if tn + fn else 0.0
    out['ppv'] = float(tp / (tp + fp)) if tp + fp else 0.0
    out['fpr'] = float(fp / (fp + tn)) if fp + tn else 0.0
    out['fnr'] = float(fn / (fn + tp)) if fn + tp else 0.0
    out['fdr'] = float(fp / (fp + tp)) if fp + tp else 0.0
  else:
    for key in ('sensitivity', 'specificity', 'npv', 'ppv',
                'fpr', 'fnr', 'fdr'):
      out[key] = float('nan')
  out['ks'] = _safe(_ks, y_true, proba)
  out['ks_statistic'] = out['ks']
  try:
    out['confusion_matrix'] = confusion_matrix(
        y_true, y_pred).tolist()
  except Exception as exc:  # noqa: BLE001
    warnings.warn(f'confusion_matrix failed: {exc}', stacklevel=2)
    out['confusion_matrix'] = []
  out['classification_report'] = _safe_dict(
      classification_report, y_true, y_pred, output_dict=True,
      zero_division=0)
  ece, mce, decomp = _calibration(y_true, proba)
  out['ece'] = ece
  out['mce'] = mce
  out['calibration_ece'] = ece
  out['calibration_mce'] = mce
  out['brier_decomposition'] = decomp
  return out


def _ks(y_true, proba):
  '''Compute the Kolmogorov-Smirnov statistic.

  Args:
    y_true: True binary labels.
    proba: Two-column probability array.

  Returns:
    KS statistic as float.
  '''
  from scipy.stats import ks_2samp
  y = np.asarray(y_true)
  pos = _pos_proba(proba, y)[y == 1]
  neg = _pos_proba(proba, y)[y == 0]
  return ks_2samp(pos, neg).statistic


def _pinball(y_true, y_pred, quantile):
  '''Compute pinball (quantile) loss.

  Args:
    y_true: True values.
    y_pred: Predicted values.
    quantile: Quantile in (0, 1).

  Returns:
    Pinball loss as float.
  '''
  err = np.asarray(y_true, dtype=float) - np.asarray(
      y_pred, dtype=float)
  return float(np.mean(np.maximum(
      quantile * err, (quantile - 1) * err)))


def _residual_moment(values, kind):
  '''Compute residual skew or kurtosis without scipy fallback.

  Args:
    values: Residual array.
    kind: 'skew' or 'kurtosis'.

  Returns:
    Moment as float.
  '''
  values = np.asarray(values, dtype=float)
  std = values.std()
  if std == 0:
    return 0.0
  z = (values - values.mean()) / std
  if kind == 'skew':
    return float(np.mean(z ** 3))
  return float(np.mean(z ** 4) - 3)


def _regression(y_true, y_pred):
  '''Compute all approved regression metrics.

  Args:
    y_true: True values.
    y_pred: Predicted values.

  Returns:
    Dict of regression metric name to value.
  '''
  from sklearn.metrics import (
    d2_absolute_error_score, explained_variance_score, max_error,
    mean_absolute_error, mean_absolute_percentage_error,
    mean_gamma_deviance, mean_poisson_deviance,
    mean_squared_error, mean_squared_log_error,
    mean_tweedie_deviance, median_absolute_error, r2_score,
  )
  y_true = np.asarray(y_true, dtype=float)
  y_pred = np.asarray(y_pred, dtype=float)
  n = len(y_true)
  resid = y_pred - y_true
  out = {}
  out['mae'] = _safe(mean_absolute_error, y_true, y_pred)
  out['mean_absolute_error'] = out['mae']
  out['medae'] = _safe(median_absolute_error, y_true, y_pred)
  out['mse'] = _safe(mean_squared_error, y_true, y_pred)
  out['mean_squared_error'] = out['mse']
  out['rmse'] = _safe(lambda: np.sqrt(out['mse']))
  out['root_mean_squared_error'] = out['rmse']
  clipped = (np.clip(y_true, 0, None), np.clip(y_pred, 0, None))
  out['msle'] = _safe(mean_squared_log_error, *clipped)
  out['rmsle'] = _safe(lambda: np.sqrt(out['msle']))
  out['r2'] = _safe(r2_score, y_true, y_pred)
  out['adj_r2'] = _safe(
      lambda: 1 - (1 - out['r2']) * (n - 1) / max(n - 2, 1))
  out['mape'] = _safe(
      mean_absolute_percentage_error, y_true, y_pred)
  out['mean_absolute_percentage_error'] = out['mape']
  out['medape'] = _safe(_medape, y_true, y_pred)
  out['median_absolute_percentage_error'] = out['medape']
  out['smape'] = _safe(_smape, y_true, y_pred)
  out['explained_variance'] = _safe(
      explained_variance_score, y_true, y_pred)
  out['max_error'] = _safe(max_error, y_true, y_pred)
  out['bias'] = _safe(lambda: float(np.mean(resid)))
  out['mean_residual'] = out['bias']
  out['residual_std'] = _safe(lambda: float(np.std(resid)))
  out['residual_skew'] = _safe(
      _residual_moment, resid, 'skew')
  out['residual_kurtosis'] = _safe(
      _residual_moment, resid, 'kurtosis')
  for q, tag in ((0.1, '0_1'), (0.5, '0_5'), (0.9, '0_9')):
    val = _safe(_pinball, y_true, y_pred, q)
    out[f'quantile_loss_{tag}'] = val
    out[f'pinball_{tag}'] = val
  pos_true = np.clip(y_true, 1e-9, None)
  pos_pred = np.clip(y_pred, 1e-9, None)
  out['tweedie_deviance'] = _safe(
      mean_tweedie_deviance, pos_true, pos_pred)
  out['poisson_deviance'] = _safe(
      mean_poisson_deviance, pos_true, pos_pred)
  out['gamma_deviance'] = _safe(
      mean_gamma_deviance, pos_true, pos_pred)
  out['d2_absolute_error_score'] = _safe(
      d2_absolute_error_score, y_true, y_pred)
  out['d2_abs'] = out['d2_absolute_error_score']
  return out


def _medape(y_true, y_pred):
  '''Compute median absolute percentage error.

  Args:
    y_true: True values.
    y_pred: Predicted values.

  Returns:
    Median absolute percentage error as float.
  '''
  mask = np.asarray(y_true) != 0
  return float(np.median(
      np.abs((np.asarray(y_true)[mask] - np.asarray(y_pred)[mask])
             / np.asarray(y_true)[mask])))


def _smape(y_true, y_pred):
  '''Compute symmetric mean absolute percentage error.

  Args:
    y_true: True values.
    y_pred: Predicted values.

  Returns:
    Symmetric mean absolute percentage error as float.
  '''
  denom = (np.abs(y_true) + np.abs(y_pred)) / 2
  denom = np.where(denom == 0, 1e-9, denom)
  return float(np.mean(np.abs(y_pred - y_true) / denom))


def compute_metrics(y_true, y_pred, y_proba, task_type):
  '''Compute metrics dict for the given task type.

  Args:
    y_true: True labels or values.
    y_pred: Predicted labels or values.
    y_proba: Predicted probabilities (classification) or None.
    task_type: 'classification' or 'regression'.

  Returns:
    Dict of metric name to float (lists for per-class/cm).
  '''
  if task_type == 'classification':
    if y_proba is None:
      raise ValueError('y_proba required for classification')
    return _classification(y_true, y_pred, np.asarray(y_proba))
  return _regression(y_true, y_pred)


LOWER_BETTER = {'rmse', 'mae', 'mse', 'log_loss', 'brier', 'ece',
                'medae', 'rmsle', 'mape', 'smape', 'max_error',
                'mce', 'hinge_loss', 'zero_one_loss', 'msle',
                'tweedie_deviance', 'poisson_deviance',
                'gamma_deviance', 'fpr', 'fnr', 'fdr',
                'quantile_loss_0_1', 'quantile_loss_0_5',
                'quantile_loss_0_9', 'pinball_0_1',
                'pinball_0_5', 'pinball_0_9', 'residual_std'}
