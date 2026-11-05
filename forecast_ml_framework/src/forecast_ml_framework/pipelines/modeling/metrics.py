#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Metric computation for classification and regression.

Metric selection (for parameters.yml ``metrics`` /
``metrics_to_calculate``):

- ``METRIC_REGISTRY`` maps every metric name to a callable
  ``fn(y_true, y_pred, y_proba) -> {name: value}``.
- ``DEFAULT_METRICS`` is a curated ~15-name subset shown by
  default; ``evaluate_model`` computes only selected metrics,
  sharing intermediates (confusion matrix, MAE, ROC-AUC, ...)
  once and skipping unselected families (e.g. no ROC-AUC when
  only ``mae`` is selected).
- ``metrics='all'`` computes every registry metric; a list
  computes only those names (unknown -> ``ValueError``).
'''
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


def _want(selected, *names):
  '''Return True if any of names is selected (None = all).

  Args:
    selected: Set of selected names or None for all.
    *names: Candidate metric names.

  Returns:
    True if computation for any name is wanted.
  '''
  return selected is None or any(name in selected for name in names)


def _classification(y_true, y_pred, proba, selected=None):
  '''Compute selected classification metrics (None = all).

  Args:
    y_true: True labels.
    y_pred: Predicted labels.
    proba: Probability array.
    selected: Set of metric names to compute, or None for all.
      Shared intermediates (confusion matrix, ROC-AUC, ...)
      are computed once and only if a dependent metric is
      selected; unselected families are skipped entirely.

  Returns:
    Dict of classification metric name to value.
  '''
  from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, classification_report, cohen_kappa_score,
    confusion_matrix, f1_score, fbeta_score, jaccard_score,
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
  accuracy_val = float('nan')
  if _want(selected, 'accuracy', 'error_rate'):
    accuracy_val = _safe(accuracy_score, y_true, y_pred)
    if _want(selected, 'accuracy'):
      out['accuracy'] = accuracy_val
    if _want(selected, 'error_rate'):
      out['error_rate'] = _safe(lambda: 1.0 - accuracy_val)
  if _want(selected, 'balanced_accuracy'):
    out['balanced_accuracy'] = _safe(
        balanced_accuracy_score, y_true, y_pred)
  for avg in ('macro', 'micro', 'weighted'):
    if _want(selected, f'precision_{avg}'):
      out[f'precision_{avg}'] = _safe(
          precision_score, y_true, y_pred, average=avg,
          zero_division=0)
    if _want(selected, f'recall_{avg}'):
      out[f'recall_{avg}'] = _safe(
          recall_score, y_true, y_pred, average=avg,
          zero_division=0)
    if _want(selected, f'f1_{avg}'):
      out[f'f1_{avg}'] = _safe(
          f1_score, y_true, y_pred, average=avg, zero_division=0)
    if _want(selected, f'jaccard_{avg}'):
      out[f'jaccard_{avg}'] = _safe(
          jaccard_score, y_true, y_pred, average=avg,
          zero_division=0)
  if _want(selected, 'per_class_precision'):
    out['per_class_precision'] = _safe_list(
        precision_score, y_true, y_pred, average=None,
        zero_division=0)
  if _want(selected, 'per_class_recall'):
    out['per_class_recall'] = _safe_list(
        recall_score, y_true, y_pred, average=None, zero_division=0)
  if _want(selected, 'per_class_f1'):
    out['per_class_f1'] = _safe_list(
        f1_score, y_true, y_pred, average=None, zero_division=0)
  # ROC AUC (ovr / ovo where multiclass) - shared intermediate.
  roc_val = float('nan')
  pr_val = float('nan')
  if _want(selected, 'roc_auc', 'roc_auc_ovr', 'roc_auc_ovo',
           'gini_coefficient'):
    if is_binary:
      roc_val = _safe(roc_auc_score, y_true, pos)
    else:
      roc_val = _safe(
          roc_auc_score, y_true, proba, multi_class='ovr',
          average='macro')
    if _want(selected, 'roc_auc'):
      out['roc_auc'] = roc_val
    if _want(selected, 'roc_auc_ovr'):
      out['roc_auc_ovr'] = roc_val
    if _want(selected, 'roc_auc_ovo'):
      if is_binary:
        out['roc_auc_ovo'] = roc_val
      else:
        out['roc_auc_ovo'] = _safe(
            roc_auc_score, y_true, proba, multi_class='ovo',
            average='macro')
  if _want(selected, 'pr_auc', 'pr_auc_ovr', 'average_precision',
           'mean_average_precision', 'map'):
    if is_binary:
      pr_val = _safe(average_precision_score, y_true, pos)
    else:
      pr_val = _safe(
          average_precision_score, y_true, proba, average='macro')
    for name in ('pr_auc', 'pr_auc_ovr', 'average_precision',
                 'mean_average_precision', 'map'):
      if _want(selected, name):
        out[name] = pr_val
  if _want(selected, 'log_loss'):
    out['log_loss'] = _safe(_log_loss, y_true, proba)
  if _want(selected, 'brier', 'brier_score'):
    if is_binary:
      brier_val = _safe(brier_score_loss, y_true, pos)
    else:
      brier_val = float('nan')
      warnings.warn('brier undefined for multiclass', stacklevel=2)
    if _want(selected, 'brier'):
      out['brier'] = brier_val
    if _want(selected, 'brier_score'):
      out['brier_score'] = brier_val
  if _want(selected, 'kappa', 'cohen_kappa'):
    out['kappa'] = _safe(cohen_kappa_score, y_true, y_pred)
    if _want(selected, 'cohen_kappa'):
      out['cohen_kappa'] = out['kappa']
  if _want(selected, 'kappa_linear', 'cohen_kappa_linear'):
    val = _safe(cohen_kappa_score, y_true, y_pred,
                weights='linear')
    if _want(selected, 'kappa_linear'):
      out['kappa_linear'] = val
    if _want(selected, 'cohen_kappa_linear'):
      out['cohen_kappa_linear'] = val
  if _want(selected, 'kappa_quadratic', 'cohen_kappa_quadratic'):
    val = _safe(cohen_kappa_score, y_true, y_pred,
                weights='quadratic')
    if _want(selected, 'kappa_quadratic'):
      out['kappa_quadratic'] = val
    if _want(selected, 'cohen_kappa_quadratic'):
      out['cohen_kappa_quadratic'] = val
  if _want(selected, 'mcc', 'matthews_corrcoef'):
    val = _safe(matthews_corrcoef, y_true, y_pred)
    if _want(selected, 'mcc'):
      out['mcc'] = val
    if _want(selected, 'matthews_corrcoef'):
      out['matthews_corrcoef'] = val
  if _want(selected, 'zero_one_loss'):
    out['zero_one_loss'] = _safe(zero_one_loss, y_true, y_pred)
  if _want(selected, 'hinge_loss'):
    out['hinge_loss'] = _safe(_hinge, y_true, proba)
  if _want(selected, 'top_k_accuracy_2'):
    out['top_k_accuracy_2'] = _safe(_top_k, y_true, proba, 2)
  if _want(selected, 'top_k_accuracy_3'):
    out['top_k_accuracy_3'] = _safe(_top_k, y_true, proba, 3)
  if _want(selected, 'jaccard'):
    if _want(selected, 'jaccard_macro'):
      out['jaccard'] = out['jaccard_macro']
    else:
      out['jaccard'] = _safe(
          jaccard_score, y_true, y_pred, average='macro',
          zero_division=0)
  if _want(selected, 'jaccard_binary'):
    out['jaccard_binary'] = _safe(
        jaccard_score, y_true, y_pred, average='binary',
        zero_division=0)
  # Confusion matrix: shared intermediate, computed once here
  # (reused for rates, DOR, confusion_matrix output below).
  cm_wanted = _want(
      selected, 'sensitivity', 'specificity', 'npv', 'ppv', 'fpr',
      'fnr', 'fdr', 'diagnostic_odds_ratio', 'fowlkes_mallows',
      'g_mean', 'youden_j', 'markedness', 'prevalence_threshold',
      'confusion_matrix')
  cm = np.zeros((0, 0))
  if cm_wanted:
    try:
      cm = np.asarray(confusion_matrix(y_true, y_pred), dtype=float)
    except Exception as exc:  # noqa: BLE001
      warnings.warn(f'confusion_matrix failed: {exc}',
                    stacklevel=2)
      cm = np.zeros((0, 0))
  cm_rates = {}
  if cm_wanted:
    if cm.shape == (2, 2):
      tn, fp, fn, tp = cm.ravel()
      cm_rates = {
          'sensitivity': float(tp / (tp + fn)) if tp + fn else 0.0,
          'specificity': float(tn / (tn + fp)) if tn + fp else 0.0,
          'npv': float(tn / (tn + fn)) if tn + fn else 0.0,
          'ppv': float(tp / (tp + fp)) if tp + fp else 0.0,
          'fpr': float(fp / (fp + tn)) if fp + tn else 0.0,
          'fnr': float(fn / (fn + tp)) if fn + tp else 0.0,
          'fdr': float(fp / (fp + tp)) if fp + tp else 0.0}
    else:
      cm_rates = {key: float('nan') for key in (
          'sensitivity', 'specificity', 'npv', 'ppv', 'fpr',
          'fnr', 'fdr')}
    for key, val in cm_rates.items():
      if _want(selected, key):
        out[key] = val
  if _want(selected, 'ks', 'ks_statistic'):
    ks_val = _safe(_ks, y_true, proba)
    if _want(selected, 'ks'):
      out['ks'] = ks_val
    if _want(selected, 'ks_statistic'):
      out['ks_statistic'] = ks_val
  # Phase-3 classification metrics.
  if _want(selected, 'f_beta_0_5', 'f0_5', 'f_beta'):
    f05 = _safe(
        fbeta_score, y_true, y_pred, beta=0.5, average='weighted',
        zero_division=0)
    if _want(selected, 'f_beta_0_5'):
      out['f_beta_0_5'] = f05
    if _want(selected, 'f0_5'):
      out['f0_5'] = f05
    if _want(selected, 'f_beta'):
      out['f_beta'] = f05
  if _want(selected, 'f_beta_2', 'f2'):
    f2v = _safe(
        fbeta_score, y_true, y_pred, beta=2, average='weighted',
        zero_division=0)
    if _want(selected, 'f_beta_2'):
      out['f_beta_2'] = f2v
    if _want(selected, 'f2'):
      out['f2'] = f2v
  rates = (_safe_dict(_per_class_rates, y_true, y_pred)
           if _want(selected, 'g_mean', 'youden_j', 'markedness',
                    'prevalence_threshold', 'fowlkes_mallows')
           else {})
  if rates:
    sens_m = float(np.mean(rates['sens']))
    spec_m = float(np.mean(rates['spec']))
    ppv_m = float(np.mean(rates['ppv']))
    npv_m = float(np.mean(rates['npv']))
  else:
    sens_m = spec_m = ppv_m = npv_m = float('nan')
  if _want(selected, 'g_mean'):
    out['g_mean'] = _safe(lambda: float(np.sqrt(sens_m * spec_m)))
  if _want(selected, 'youden_j'):
    out['youden_j'] = _safe(lambda: float(sens_m + spec_m - 1))
  if _want(selected, 'markedness', 'prevalence_threshold'):
    marked = _safe(lambda: float(ppv_m + npv_m - 1))
    if _want(selected, 'markedness'):
      out['markedness'] = marked
    if _want(selected, 'prevalence_threshold'):
      out['prevalence_threshold'] = marked
  if _want(selected, 'diagnostic_odds_ratio'):
    if cm.shape == (2, 2):
      tn_, fp_, fn_, tp_ = cm.ravel()
      denom = fp_ * fn_
      out['diagnostic_odds_ratio'] = _safe(
          lambda: float(tp_ * tn_ / denom)
          if denom else float('nan'))
    else:
      out['diagnostic_odds_ratio'] = float('nan')
  if _want(selected, 'fowlkes_mallows'):
    if cm.shape == (2, 2):
      out['fowlkes_mallows'] = _safe(
          lambda: float(np.sqrt(
              cm_rates['ppv'] * cm_rates['sensitivity'])))
    else:
      out['fowlkes_mallows'] = _safe(
          lambda: float(np.sqrt(ppv_m * sens_m)))
  if _want(selected, 'kl_divergence'):
    out['kl_divergence'] = _safe(_kl_divergence, y_true, y_pred)
  if _want(selected, 'gini_coefficient'):
    if not _want(selected, 'roc_auc', 'roc_auc_ovr',
                 'roc_auc_ovo'):
      roc_val = (_safe(roc_auc_score, y_true, pos) if is_binary
                 else _safe(roc_auc_score, y_true, proba,
                            multi_class='ovr', average='macro'))
    out['gini_coefficient'] = _safe(
        lambda: float(2 * roc_val - 1))
  if _want(selected, 'mrr', 'ndcg', 'ndcg_at_10'):
    ranks = np.asarray(_true_ranks(y_true, proba), dtype=float)
    if _want(selected, 'mrr'):
      out['mrr'] = _safe(lambda: float(np.mean(1.0 / ranks)))
    ndcg_val = _safe(
        lambda: float(np.mean(1.0 / np.log2(ranks + 1))))
    if _want(selected, 'ndcg'):
      out['ndcg'] = ndcg_val
    if _want(selected, 'ndcg_at_10'):
      out['ndcg_at_10'] = ndcg_val
  if _want(selected, 'lift_top_decile',
           'cumulative_gain_top_decile', 'cum_gain_top_decile'):
    lift, gain = (float('nan'), float('nan'))
    try:
      lift, gain = _top_decile(y_true, y_pred, proba)
    except Exception as exc:  # noqa: BLE001
      warnings.warn(f'top_decile failed: {exc}', stacklevel=2)
    if _want(selected, 'lift_top_decile'):
      out['lift_top_decile'] = float(lift)
    if _want(selected, 'cumulative_gain_top_decile'):
      out['cumulative_gain_top_decile'] = float(gain)
    if _want(selected, 'cum_gain_top_decile'):
      out['cum_gain_top_decile'] = float(gain)
  for k in (5, 10):
    at_names = (f'precision_at_{k}', f'recall_at_{k}',
                f'f1_at_{k}', f'precision_at_k_{k}',
                f'recall_at_k_{k}', f'f1_at_k_{k}')
    if not _want(selected, *at_names):
      continue
    p_k, r_k, f_k = (float('nan'),) * 3
    try:
      p_k, r_k, f_k = _at_k(y_true, y_pred, proba, k)
    except Exception as exc:  # noqa: BLE001
      warnings.warn(f'at_k failed: {exc}', stacklevel=2)
    if _want(selected, f'precision_at_{k}'):
      out[f'precision_at_{k}'] = float(p_k)
    if _want(selected, f'recall_at_{k}'):
      out[f'recall_at_{k}'] = float(r_k)
    if _want(selected, f'f1_at_{k}'):
      out[f'f1_at_{k}'] = float(f_k)
    if _want(selected, f'precision_at_k_{k}'):
      out[f'precision_at_k_{k}'] = float(p_k)
    if _want(selected, f'recall_at_k_{k}'):
      out[f'recall_at_k_{k}'] = float(r_k)
    if _want(selected, f'f1_at_k_{k}'):
      out[f'f1_at_k_{k}'] = float(f_k)
  if _want(selected, 'confusion_matrix'):
    try:
      out['confusion_matrix'] = cm.astype(int).tolist() if (
          cm.size) else confusion_matrix(
          y_true, y_pred).tolist()
    except Exception as exc:  # noqa: BLE001
      warnings.warn(f'confusion_matrix failed: {exc}',
                    stacklevel=2)
      out['confusion_matrix'] = []
  if _want(selected, 'classification_report'):
    out['classification_report'] = _safe_dict(
        classification_report, y_true, y_pred, output_dict=True,
        zero_division=0)
  if _want(selected, 'ece', 'mce', 'calibration_ece',
           'calibration_mce', 'brier_decomposition'):
    ece, mce, decomp = _calibration(y_true, proba)
    if _want(selected, 'ece'):
      out['ece'] = ece
    if _want(selected, 'mce'):
      out['mce'] = mce
    if _want(selected, 'calibration_ece'):
      out['calibration_ece'] = ece
    if _want(selected, 'calibration_mce'):
      out['calibration_mce'] = mce
    if _want(selected, 'brier_decomposition'):
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


def _true_ranks(y_true, proba):
  '''Return 1-based rank of true label in proba ordering per row.

  Args:
    y_true: True labels.
    proba: Probability array (n, n_classes).

  Returns:
    Array of ranks (1 = highest probability).
  '''
  proba = np.asarray(proba, dtype=float)
  labels = _labels(y_true, y_true)
  order = np.argsort(-proba, axis=1)
  ranks = np.empty(len(y_true), dtype=float)
  for i, true in enumerate(np.asarray(y_true)):
    idx = labels.index(true) if true in labels else 0
    ranks[i] = int(np.where(order[i] == idx)[0][0]) + 1
  return ranks


def _at_k(y_true, y_pred, proba, k):
  '''Compute precision/recall/F1 at k by proba ranking.

  Args:
    y_true: True labels.
    y_pred: Predicted labels (multiclass correctness).
    proba: Probability array.
    k: Cutoff k.

  Returns:
    Tuple of (precision, recall, f1) at k.
  '''
  y_true = np.asarray(y_true)
  y_pred = np.asarray(y_pred)
  proba = np.asarray(proba, dtype=float)
  n = len(y_true)
  kk = max(1, min(int(k), n))
  if proba.shape[1] == 2:
    score = proba[:, 1]
    order = np.argsort(-score)
    top = order[:kk]
    tp = int(np.sum(y_true[top] == 1))
    positives = int(np.sum(y_true == 1))
    precision = tp / kk
    recall = tp / positives if positives else 0.0
  else:
    score = proba.max(axis=1)
    order = np.argsort(-score)
    top = order[:kk]
    correct = int(np.sum(y_true[top] == y_pred[top]))
    precision = correct / kk
    recall = correct / n if n else 0.0
  f1 = (2 * precision * recall / (precision + recall)
        if precision + recall else 0.0)
  return float(precision), float(recall), float(f1)


def _top_decile(y_true, y_pred, proba):
  '''Compute lift and cumulative gain at the top decile.

  Args:
    y_true: True labels.
    y_pred: Predicted labels.
    proba: Probability array.

  Returns:
    Tuple of (lift, cumulative_gain) at top decile.
  '''
  y_true = np.asarray(y_true)
  y_pred = np.asarray(y_pred)
  proba = np.asarray(proba, dtype=float)
  n = len(y_true)
  kk = max(1, int(np.ceil(0.1 * n)))
  if proba.shape[1] == 2:
    score = proba[:, 1]
    order = np.argsort(-score)
    top = order[:kk]
    tp = int(np.sum(y_true[top] == 1))
    positives = int(np.sum(y_true == 1))
    prevalence = positives / n if n else 0.0
    precision_top = tp / kk
    lift = precision_top / prevalence if prevalence else float('nan')
    gain = tp / positives if positives else 0.0
  else:
    score = proba.max(axis=1)
    order = np.argsort(-score)
    top = order[:kk]
    correct_top = int(np.sum(y_true[top] == y_pred[top]))
    acc = float(np.mean(y_true == y_pred)) if n else 0.0
    acc_top = correct_top / kk
    lift = acc_top / acc if acc else float('nan')
    gain = correct_top / n if n else 0.0
  return float(lift), float(gain)


def _kl_divergence(y_true, y_pred):
  '''Compute KL divergence of true vs predicted distributions.

  Args:
    y_true: True labels.
    y_pred: Predicted labels.

  Returns:
    KL(true || predicted) as float.
  '''
  labels = _labels(y_true, y_pred)
  p = np.array([np.mean(np.asarray(y_true) == lab)
                for lab in labels], dtype=float)
  q = np.array([np.mean(np.asarray(y_pred) == lab)
                for lab in labels], dtype=float)
  p = np.clip(p, 1e-15, None)
  q = np.clip(q, 1e-15, None)
  return float(np.sum(p * np.log(p / q)))


def _per_class_rates(y_true, y_pred):
  '''Compute per-class sensitivity/specificity/ppv/npv via cm.

  Args:
    y_true: True labels.
    y_pred: Predicted labels.

  Returns:
    Dict of per-class rate lists.
  '''
  from sklearn.metrics import confusion_matrix
  labels = _labels(y_true, y_pred)
  cm = np.asarray(confusion_matrix(y_true, y_pred, labels=labels),
                  dtype=float)
  rates = {'sens': [], 'spec': [], 'ppv': [], 'npv': []}
  for i in range(len(labels)):
    tp = cm[i, i]
    fn = cm[i].sum() - tp
    fp = cm[:, i].sum() - tp
    tn = cm.sum() - tp - fn - fp
    rates['sens'].append(tp / (tp + fn) if tp + fn else 0.0)
    rates['spec'].append(tn / (tn + fp) if tn + fp else 0.0)
    rates['ppv'].append(tp / (tp + fp) if tp + fp else 0.0)
    rates['npv'].append(tn / (tn + fn) if tn + fn else 0.0)
  return rates


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


def _regression(y_true, y_pred, selected=None):
  '''Compute selected regression metrics (None = all).

  Args:
    y_true: True values.
    y_pred: Predicted values.
    selected: Set of metric names to compute, or None for all.
      Shared intermediates (MAE/MSE/residuals) are computed
      once and only if a dependent metric is selected.

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
  mae_val = float('nan')
  if _want(selected, 'mae', 'mean_absolute_error', 'mase',
           'mean_absolute_scaled_error', 'wmae', 'weighted_mae'):
    mae_val = _safe(mean_absolute_error, y_true, y_pred)
  if _want(selected, 'mae'):
    out['mae'] = mae_val
  if _want(selected, 'mean_absolute_error'):
    out['mean_absolute_error'] = mae_val
  if _want(selected, 'medae'):
    out['medae'] = _safe(median_absolute_error, y_true, y_pred)
  mse_val = float('nan')
  if _want(selected, 'mse', 'mean_squared_error', 'rmse',
           'root_mean_squared_error'):
    mse_val = _safe(mean_squared_error, y_true, y_pred)
  if _want(selected, 'mse'):
    out['mse'] = mse_val
  if _want(selected, 'mean_squared_error'):
    out['mean_squared_error'] = mse_val
  rmse_val = _safe(lambda: np.sqrt(mse_val))
  if _want(selected, 'rmse'):
    out['rmse'] = rmse_val
  if _want(selected, 'root_mean_squared_error'):
    out['root_mean_squared_error'] = rmse_val
  msle_val = float('nan')
  if _want(selected, 'msle', 'rmsle'):
    clipped = (np.clip(y_true, 0, None), np.clip(y_pred, 0, None))
    msle_val = _safe(mean_squared_log_error, *clipped)
  if _want(selected, 'msle'):
    out['msle'] = msle_val
  if _want(selected, 'rmsle'):
    out['rmsle'] = _safe(lambda: np.sqrt(msle_val))
  r2_val = float('nan')
  if _want(selected, 'r2', 'adj_r2'):
    r2_val = _safe(r2_score, y_true, y_pred)
  if _want(selected, 'r2'):
    out['r2'] = r2_val
  if _want(selected, 'adj_r2'):
    out['adj_r2'] = _safe(
        lambda: 1 - (1 - r2_val) * (n - 1) / max(n - 2, 1))
  if _want(selected, 'mape', 'mean_absolute_percentage_error'):
    mape_val = _safe(
        mean_absolute_percentage_error, y_true, y_pred)
    if _want(selected, 'mape'):
      out['mape'] = mape_val
    if _want(selected, 'mean_absolute_percentage_error'):
      out['mean_absolute_percentage_error'] = mape_val
  if _want(selected, 'medape', 'median_absolute_percentage_error'):
    med_val = _safe(_medape, y_true, y_pred)
    if _want(selected, 'medape'):
      out['medape'] = med_val
    if _want(selected, 'median_absolute_percentage_error'):
      out['median_absolute_percentage_error'] = med_val
  if _want(selected, 'smape'):
    out['smape'] = _safe(_smape, y_true, y_pred)
  if _want(selected, 'explained_variance'):
    out['explained_variance'] = _safe(
        explained_variance_score, y_true, y_pred)
  if _want(selected, 'max_error'):
    out['max_error'] = _safe(max_error, y_true, y_pred)
  bias_val = float('nan')
  if _want(selected, 'bias', 'mean_residual',
           'mean_bias_deviation'):
    bias_val = _safe(lambda: float(np.mean(resid)))
  if _want(selected, 'bias'):
    out['bias'] = bias_val
  if _want(selected, 'mean_residual'):
    out['mean_residual'] = bias_val
  if _want(selected, 'mean_bias_deviation'):
    out['mean_bias_deviation'] = bias_val
  if _want(selected, 'residual_std'):
    out['residual_std'] = _safe(lambda: float(np.std(resid)))
  if _want(selected, 'residual_skew'):
    out['residual_skew'] = _safe(
        _residual_moment, resid, 'skew')
  if _want(selected, 'residual_kurtosis'):
    out['residual_kurtosis'] = _safe(
        _residual_moment, resid, 'kurtosis')
  for q, tag in ((0.1, '0_1'), (0.5, '0_5'), (0.9, '0_9')):
    if not _want(selected, f'quantile_loss_{tag}',
                 f'pinball_{tag}'):
      continue
    val = _safe(_pinball, y_true, y_pred, q)
    if _want(selected, f'quantile_loss_{tag}'):
      out[f'quantile_loss_{tag}'] = val
    if _want(selected, f'pinball_{tag}'):
      out[f'pinball_{tag}'] = val
  pos_true = np.clip(y_true, 1e-9, None)
  pos_pred = np.clip(y_pred, 1e-9, None)
  if _want(selected, 'tweedie_deviance'):
    out['tweedie_deviance'] = _safe(
        mean_tweedie_deviance, pos_true, pos_pred)
  if _want(selected, 'poisson_deviance'):
    out['poisson_deviance'] = _safe(
        mean_poisson_deviance, pos_true, pos_pred)
  if _want(selected, 'gamma_deviance'):
    out['gamma_deviance'] = _safe(
        mean_gamma_deviance, pos_true, pos_pred)
  if _want(selected, 'd2_absolute_error_score', 'd2_abs'):
    d2_val = _safe(d2_absolute_error_score, y_true, y_pred)
    if _want(selected, 'd2_absolute_error_score'):
      out['d2_absolute_error_score'] = d2_val
    if _want(selected, 'd2_abs'):
      out['d2_abs'] = d2_val
  # Phase-3 regression metrics.
  if _want(selected, 'mase', 'mean_absolute_scaled_error'):
    naive = float(np.mean(np.abs(np.diff(y_true)))) if n > 1 else 0.0
    mase_val = _safe(
        lambda: float(mae_val / naive) if naive else float('nan'))
    if _want(selected, 'mase'):
      out['mase'] = mase_val
    if _want(selected, 'mean_absolute_scaled_error'):
      out['mean_absolute_scaled_error'] = mase_val
  if _want(selected, 'wmae', 'weighted_mae'):
    weights = np.abs(y_true)
    wmae_val = _safe(
        lambda: float(np.sum(weights * np.abs(resid))
                      / np.sum(weights))
        if np.sum(weights) else float(mae_val))
    if _want(selected, 'wmae'):
      out['wmae'] = wmae_val
    if _want(selected, 'weighted_mae'):
      out['weighted_mae'] = wmae_val
  mask = y_true != 0
  if np.any(mask):
    pct = (y_true[mask] - y_pred[mask]) / y_true[mask]
    pct_signed = (y_pred[mask] - y_true[mask]) / y_true[mask]
  else:
    pct = np.array([float('nan')])
    pct_signed = np.array([float('nan')])
  mspe_val = float('nan')
  if _want(selected, 'mspe', 'rmspe',
           'root_mean_squared_percentage_error'):
    mspe_val = _safe(lambda: float(np.mean(pct ** 2)))
  if _want(selected, 'mspe'):
    out['mspe'] = mspe_val
  rmspe_val = _safe(lambda: float(np.sqrt(mspe_val)))
  if _want(selected, 'rmspe'):
    out['rmspe'] = rmspe_val
  if _want(selected, 'root_mean_squared_percentage_error'):
    out['root_mean_squared_percentage_error'] = rmspe_val
  if _want(selected, 'mpe', 'mean_percentage_error'):
    mpe_val = _safe(lambda: float(np.mean(pct_signed)))
    if _want(selected, 'mpe'):
      out['mpe'] = mpe_val
    if _want(selected, 'mean_percentage_error'):
      out['mean_percentage_error'] = mpe_val
  if _want(selected, 'maape',
           'mean_absolute_arctangent_percentage_error'):
    maape_val = _safe(
        lambda: float(np.mean(np.arctan(np.abs(pct)))))
    if _want(selected, 'maape'):
      out['maape'] = maape_val
    if _want(selected,
             'mean_absolute_arctangent_percentage_error'):
      out['mean_absolute_arctangent_percentage_error'] = maape_val
  if _want(selected, 'pearson_r', 'pearson'):
    pear_val = _safe(_pearson, y_true, y_pred)
    if _want(selected, 'pearson_r'):
      out['pearson_r'] = pear_val
    if _want(selected, 'pearson'):
      out['pearson'] = pear_val
  if _want(selected, 'spearman_rho', 'spearman'):
    spear_val = _safe(_spearman, y_true, y_pred)
    if _want(selected, 'spearman_rho'):
      out['spearman_rho'] = spear_val
    if _want(selected, 'spearman'):
      out['spearman'] = spear_val
  if _want(selected, 'huber_loss'):
    out['huber_loss'] = _safe(_huber, y_true, y_pred, 1.35)
  if _want(selected, 'log_cosh_loss'):
    out['log_cosh_loss'] = _safe(_log_cosh, y_true, y_pred)
  if _want(selected, 'r2_log', 'r2_log1p'):
    r2log_val = _safe(
        r2_score, np.log1p(np.clip(y_true, 0, None)),
        np.log1p(np.clip(y_pred, 0, None)))
    if _want(selected, 'r2_log'):
      out['r2_log'] = r2log_val
    if _want(selected, 'r2_log1p'):
      out['r2_log1p'] = r2log_val
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


def _rankdata(values):
  '''Return ranks (average ties) for Spearman fallback.

  Args:
    values: One-dimensional array.

  Returns:
    Array of ranks.
  '''
  values = np.asarray(values, dtype=float)
  order = np.argsort(values, kind='mergesort')
  ranks = np.empty(len(values), dtype=float)
  sorted_vals = values[order]
  i = 0
  while i < len(values):
    j = i
    while j + 1 < len(values) and sorted_vals[j + 1] == sorted_vals[i]:
      j += 1
    ranks[order[i:j + 1]] = (i + j) / 2.0 + 1
    i = j + 1
  return ranks


def _spearman(y_true, y_pred):
  '''Compute Spearman rho, scipy preferred, manual fallback.

  Args:
    y_true: True values.
    y_pred: Predicted values.

  Returns:
    Spearman rho as float.
  '''
  try:
    from scipy.stats import spearmanr
    return float(spearmanr(y_true, y_pred).statistic)
  except Exception:  # noqa: BLE001
    return float(np.corrcoef(
        _rankdata(y_true), _rankdata(y_pred))[0, 1])


def _pearson(y_true, y_pred):
  '''Compute Pearson r.

  Args:
    y_true: True values.
    y_pred: Predicted values.

  Returns:
    Pearson correlation as float.
  '''
  return float(np.corrcoef(
      np.asarray(y_true, dtype=float),
      np.asarray(y_pred, dtype=float))[0, 1])


def _huber(y_true, y_pred, delta=1.35):
  '''Compute Huber loss with given delta.

  Args:
    y_true: True values.
    y_pred: Predicted values.
    delta: Huber threshold.

  Returns:
    Huber loss as float.
  '''
  err = np.abs(np.asarray(y_pred, dtype=float)
               - np.asarray(y_true, dtype=float))
  quad = err <= delta
  return float(np.mean(np.where(quad, 0.5 * err ** 2,
                                delta * (err - 0.5 * delta))))


def _log_cosh(y_true, y_pred):
  '''Compute log-cosh loss (numerically stable).

  Args:
    y_true: True values.
    y_pred: Predicted values.

  Returns:
    Log-cosh loss as float.
  '''
  err = np.asarray(y_pred, dtype=float) - np.asarray(
      y_true, dtype=float)
  abs_err = np.abs(err)
  return float(np.mean(abs_err + np.log1p(np.exp(-2 * abs_err))
                       - np.log(2)))


def _validate_lengths(y_true, y_pred, y_proba):
  '''Require y_true, y_pred and y_proba to share one length.'''
  try:
    n_true = len(y_true)
  except TypeError:
    raise ValueError('y_true must be a sized sequence') from None
  try:
    n_pred = len(y_pred)
  except TypeError:
    raise ValueError('y_pred must be a sized sequence') from None
  if n_pred != n_true:
    raise ValueError(
        f'length mismatch: y_true has {n_true} samples but '
        f'y_pred has {n_pred}')
  if y_proba is not None:
    try:
      n_proba = len(y_proba)
    except TypeError:
      raise ValueError(
          'y_proba must be a sized sequence') from None
    if n_proba != n_true:
      raise ValueError(
          f'length mismatch: y_true has {n_true} samples but '
          f'y_proba has {n_proba}')


def compute_metrics(y_true, y_pred, y_proba, task_type,
                    selected=None):
  '''Compute metrics dict for the given task type.

  Args:
    y_true: True labels or values.
    y_pred: Predicted labels or values.
    y_proba: Predicted probabilities (classification) or None.
    task_type: 'classification' or 'regression'.
    selected: Set/list of metric names to compute, or None
      for all metrics. Only selected families are computed;
      shared intermediates are computed once.

  Returns:
    Dict of metric name to float (lists for per-class/cm).
  Raises:
    ValueError: If a selected name is unknown for the task.
  '''
  _validate_lengths(y_true, y_pred, y_proba)
  sel = None
  if selected is not None:
    sel = set(resolve_selected_metrics(selected, task_type))
  if task_type == 'classification':
    if y_proba is None:
      raise ValueError('y_proba required for classification')
    return _classification(y_true, y_pred, np.asarray(y_proba),
                           selected=sel)
  return _regression(y_true, y_pred, selected=sel)


CLASSIFICATION_METRICS = [
    'accuracy', 'average_precision', 'balanced_accuracy',
    'brier', 'brier_decomposition', 'brier_score',
    'calibration_ece', 'calibration_mce', 'classification_report',
    'cohen_kappa', 'cohen_kappa_linear', 'cohen_kappa_quadratic',
    'confusion_matrix', 'cum_gain_top_decile',
    'cumulative_gain_top_decile', 'diagnostic_odds_ratio',
    'ece', 'error_rate', 'f0_5', 'f1_at_10', 'f1_at_5',
    'f1_at_k_10', 'f1_at_k_5', 'f1_macro', 'f1_micro',
    'f1_weighted', 'f2', 'f_beta', 'f_beta_0_5', 'f_beta_2',
    'fdr', 'fnr', 'fowlkes_mallows', 'fpr', 'g_mean',
    'gini_coefficient', 'hinge_loss', 'jaccard',
    'jaccard_binary', 'jaccard_macro', 'jaccard_micro',
    'jaccard_weighted', 'kappa', 'kappa_linear',
    'kappa_quadratic', 'kl_divergence', 'ks', 'ks_statistic',
    'lift_top_decile', 'log_loss', 'map', 'markedness',
    'matthews_corrcoef', 'mcc', 'mce', 'mean_average_precision',
    'mrr', 'ndcg', 'ndcg_at_10', 'npv', 'per_class_f1',
    'per_class_precision', 'per_class_recall', 'ppv', 'pr_auc',
    'pr_auc_ovr', 'precision_at_10', 'precision_at_5',
    'precision_at_k_10', 'precision_at_k_5', 'precision_macro',
    'precision_micro', 'precision_weighted',
    'prevalence_threshold', 'recall_at_10', 'recall_at_5',
    'recall_at_k_10', 'recall_at_k_5', 'recall_macro',
    'recall_micro', 'recall_weighted', 'roc_auc', 'roc_auc_ovo',
    'roc_auc_ovr', 'sensitivity', 'specificity',
    'top_k_accuracy_2', 'top_k_accuracy_3', 'youden_j',
    'zero_one_loss',
]

REGRESSION_METRICS = [
    'adj_r2', 'bias', 'd2_abs', 'd2_absolute_error_score',
    'explained_variance', 'gamma_deviance', 'huber_loss',
    'log_cosh_loss', 'maape', 'mae', 'mape', 'mase',
    'max_error', 'mean_absolute_arctangent_percentage_error',
    'mean_absolute_error', 'mean_absolute_percentage_error',
    'mean_absolute_scaled_error', 'mean_bias_deviation',
    'mean_percentage_error', 'mean_residual',
    'mean_squared_error', 'medae', 'medape',
    'median_absolute_percentage_error', 'mpe', 'mse', 'msle',
    'mspe', 'pearson', 'pearson_r', 'pinball_0_1',
    'pinball_0_5', 'pinball_0_9', 'poisson_deviance',
    'quantile_loss_0_1', 'quantile_loss_0_5',
    'quantile_loss_0_9', 'r2', 'r2_log', 'r2_log1p',
    'residual_kurtosis', 'residual_skew', 'residual_std',
    'rmse', 'rmsle', 'rmspe', 'root_mean_squared_error',
    'root_mean_squared_percentage_error', 'smape', 'spearman',
    'spearman_rho', 'tweedie_deviance', 'weighted_mae',
    'wmae',
]

# Curated default subset (~15 per task combined): only these
# are calculated/shown unless parameters.yml selects more.
DEFAULT_CLASSIFICATION_METRICS = [
    'accuracy', 'balanced_accuracy', 'precision_weighted',
    'recall_weighted', 'f1_weighted', 'roc_auc', 'pr_auc',
    'log_loss', 'brier', 'kappa', 'mcc',
]

DEFAULT_REGRESSION_METRICS = [
    'mae', 'rmse', 'r2', 'mape', 'smape',
]

DEFAULT_METRICS = (
    DEFAULT_CLASSIFICATION_METRICS + DEFAULT_REGRESSION_METRICS)


def _metric_fn(name, task_type):
  '''Return callable computing a single registry metric.

  Args:
    name: Metric name.
    task_type: 'classification' or 'regression'.

  Returns:
    Callable ``fn(y_true, y_pred, y_proba) -> {name: value}``.
  '''
  def fn(y_true, y_pred, y_proba):
    '''Compute this single metric efficiently.

    Args:
      y_true: True labels or values.
      y_pred: Predicted labels or values.
      y_proba: Probabilities or None.

    Returns:
      Dict with only this metric.
    '''
    result = compute_metrics(y_true, y_pred, y_proba, task_type,
                             selected={name})
    return {name: result[name]} if name in result else {}
  return fn


METRIC_REGISTRY = {}
for _name in CLASSIFICATION_METRICS:
  METRIC_REGISTRY[_name] = _metric_fn(_name, 'classification')
for _name in REGRESSION_METRICS:
  METRIC_REGISTRY[_name] = _metric_fn(_name, 'regression')
del _name


def resolve_selected_metrics(requested, task_type=None):
  '''Resolve a metric selection request to a list of names.

  Args:
    requested: None (defaults), 'all', a single name, or an
      iterable of names.
    task_type: 'classification' or 'regression' or None. When
      given, defaults are filtered to that task and validation
      is against that task's metric list.

  Returns:
    List of metric names to calculate (task-filtered).

  Raises:
    ValueError: If a name is unknown; message lists valid
      metric names for the task.
  '''
  valid_all = (CLASSIFICATION_METRICS if task_type ==
               'classification' else REGRESSION_METRICS
               if task_type == 'regression' else
               list(METRIC_REGISTRY))
  if requested is None:
    default = (DEFAULT_CLASSIFICATION_METRICS
               if task_type == 'classification' else
               DEFAULT_REGRESSION_METRICS
               if task_type == 'regression' else
               list(DEFAULT_METRICS))
    return [name for name in default if name in valid_all]
  if isinstance(requested, str):
    if requested == 'all':
      return list(valid_all)
    requested = [requested]
  names = list(requested)
  unknown = [name for name in names if name not in valid_all]
  if unknown:
    raise ValueError(
        f'Unknown metric(s): {unknown}. '
        f'Valid metrics: {sorted(valid_all)}')
  # Filter defaults (which span both tasks) to this task.
  return [name for name in names if name in valid_all]


LOWER_BETTER = {'rmse', 'mae', 'mse', 'log_loss', 'brier', 'ece',
                'medae', 'rmsle', 'mape', 'smape', 'max_error',
                'mce', 'hinge_loss', 'zero_one_loss', 'msle',
                'tweedie_deviance', 'poisson_deviance',
                'gamma_deviance', 'fpr', 'fnr', 'fdr',
                'quantile_loss_0_1', 'quantile_loss_0_5',
                'quantile_loss_0_9', 'pinball_0_1',
                'pinball_0_5', 'pinball_0_9', 'residual_std',
                'error_rate', 'kl_divergence', 'mase', 'wmae',
                'mspe', 'rmspe', 'maape', 'huber_loss',
                'log_cosh_loss'}
