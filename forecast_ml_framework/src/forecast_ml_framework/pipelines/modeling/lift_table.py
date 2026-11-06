#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Pandas lift table (lift_cal parity, Task 4).'''

import numpy as np
import pandas as pd

from forecast_ml_framework.pipelines.modeling.ntile import (
  assign_ntile,
)

COLUMNS = [
  'decile', 'min_prob', 'max_prob', 'avg_prob', 'decile_count',
  'non_target_count', 'target_count', 'target_percentage',
  'non_target_percentage', 'cum_target_percentage',
  'cum_non_target_percentage', 'ks', 'lift',
  'decile_disconnect_rate', 'predicted_positive', 'predicted_negative',
  'tp', 'fp', 'tn', 'fn', 'precision', 'recall',
  'customers_at_or_above_threshold',
  'true_positive_at_or_above_threshold',
  'precision_at_or_above_threshold',
  'recall_at_or_above_threshold', 'tp_above_threshold',
  'fp_above_threshold', 'tn_above_threshold', 'fn_above_threshold',
  'precision_above_threshold', 'recall_above_threshold',
]


def _empty():
  '''Return empty lift DataFrame with parity columns.'''
  return pd.DataFrame(columns=COLUMNS)


def _pct(part, whole):
  '''Return part/whole*100, or 0 when whole is 0.'''
  return part / whole * 100 if whole else 0.0


def compute_lift_table(y_true, y_pred, proba, n=10, threshold=None):
  '''Compute per-decile lift table sorted by decile descending.

  Args:
    y_true: Binary target (0/1).
    y_pred: Binary prediction (0/1).
    proba: Predicted probability of the positive class.
    n: Number of ntile buckets (10 = decile, 100 = centile);
      validated to 1..100 and capped at the input length.
    threshold: When given (in [0, 1]), the
      ``*_at_or_above_threshold`` / ``*_above_threshold``
      columns are cumulative-from-top sums over the subset
      ``proba >= threshold``. When None (default), they are
      cumulative from the top decile down over all rows
      (top-decile semantics). ``pipeline.build_lift_table``
      passes the threshold from ``modeling_params``
      (``lift_threshold``/``model_threshold``) through.

  Returns:
    DataFrame with parity columns (corrected spellings:
    precision_at_or_above_threshold, precision,
    precision_above_threshold), decile 1..n low to high proba,
    rows sorted decile descending. Empty/single-class input
    returns an empty DataFrame.

  Raises:
    ValueError: If ``n`` is outside 1..100, input lengths
      differ, labels are not binary (0/1), ``threshold``
      is outside [0, 1], or ``proba`` is non-finite or
      outside [0, 1].
  '''
  y_true = np.asarray(y_true)
  y_pred = np.asarray(y_pred)
  proba = np.asarray(proba, dtype=float)
  if not (len(y_true) == len(y_pred) == len(proba)):
    raise ValueError(
      'length mismatch: y_true, y_pred and proba must have '
      'equal lengths')
  try:
    n_value = int(n)
  except (TypeError, ValueError):
    raise ValueError(f'n must be in 1..100, got {n!r}') from None
  if isinstance(n, bool) or n_value != n or not 1 <= n_value <= 100:
    raise ValueError(f'n must be in 1..100, got {n!r}')
  n = n_value
  if threshold is not None:
    if isinstance(threshold, bool):
      raise ValueError(
        f'threshold must be in [0, 1], got {threshold!r}')
    try:
      threshold = float(threshold)
    except (TypeError, ValueError):
      raise ValueError(
        f'threshold must be in [0, 1], got {threshold!r}') from None
    if not 0 <= threshold <= 1:
      raise ValueError(
        f'threshold must be in [0, 1], got {threshold!r}')
  if len(y_true) and (not set(np.unique(y_true)) <= {0, 1}
                      or not set(np.unique(y_pred)) <= {0, 1}):
    raise ValueError(
      'binary labels required: y_true and y_pred must be 0/1')
  if len(y_true) and (not np.isfinite(proba).all()
                      or not ((proba >= 0) & (proba <= 1)).all()):
    raise ValueError('proba must be finite and in [0, 1]')
  if len(y_true) == 0 or len(np.unique(y_true)) < 2:
    return _empty()
  bucket = assign_ntile(proba, 'decile', n)
  frame = pd.DataFrame({
    'decile': bucket.astype(int),
    'proba': proba,
    'target': y_true.astype(int),
    'pred': y_pred.astype(int),
  })
  frame['tp'] = ((frame['pred'] == 1) & (frame['target'] == 1))
  frame['fp'] = ((frame['pred'] == 1) & (frame['target'] == 0))
  frame['tn'] = ((frame['pred'] == 0) & (frame['target'] == 0))
  frame['fn'] = ((frame['pred'] == 0) & (frame['target'] == 1))
  frame['predicted_positive'] = (frame['pred'] == 1)
  frame['predicted_negative'] = (frame['pred'] == 0)
  grouped = frame.groupby('decile').agg(
    min_prob=('proba', 'min'),
    max_prob=('proba', 'max'),
    avg_prob=('proba', 'mean'),
    decile_count=('target', 'size'),
    target_count=('target', 'sum'),
    predicted_positive=('predicted_positive', 'sum'),
    predicted_negative=('predicted_negative', 'sum'),
    tp=('tp', 'sum'), fp=('fp', 'sum'),
    tn=('tn', 'sum'), fn=('fn', 'sum'),
  ).reset_index()
  grouped = grouped.sort_values('decile', ascending=False)
  grouped = grouped.reset_index(drop=True)
  total_target = int(grouped['target_count'].sum())
  grouped['non_target_count'] = (
    grouped['decile_count'] - grouped['target_count'])
  total_non = int(grouped['non_target_count'].sum())
  grouped['target_percentage'] = grouped['target_count'].apply(
    lambda v: _pct(v, total_target))
  grouped['non_target_percentage'] = grouped['non_target_count'].apply(
    lambda v: _pct(v, total_non))
  grouped['cum_target_percentage'] = (
    grouped['target_percentage'].cumsum())
  grouped['cum_non_target_percentage'] = (
    grouped['non_target_percentage'].cumsum())
  grouped['ks'] = (
    grouped['cum_target_percentage']
    - grouped['cum_non_target_percentage']).round(4)
  target_mean = grouped['target_count'].mean()
  grouped['lift'] = grouped['target_count'].apply(
    lambda v: v / target_mean if target_mean else 0.0)
  grouped['decile_disconnect_rate'] = grouped.apply(
    lambda r: _pct(r['target_count'], r['decile_count']), axis=1)
  grouped['precision'] = grouped.apply(
    lambda r: _pct(r['tp'], r['tp'] + r['fp']), axis=1)
  grouped['recall'] = grouped.apply(
    lambda r: _pct(r['tp'], r['tp'] + r['fn']), axis=1)
  if threshold is None:
    subset_mask = np.ones(len(frame), dtype=bool)
  else:
    subset_mask = frame['proba'].to_numpy() >= threshold
  subset = frame[subset_mask]

  def _per_decile(column):
    '''Map a per-decile subset sum onto grouped deciles.'''
    counts = subset.groupby('decile')[column].sum()
    return grouped['decile'].map(counts).fillna(0).astype(int)

  sub_count = subset.groupby('decile').size()
  sub_count = grouped['decile'].map(sub_count).fillna(0).astype(int)
  grouped['customers_at_or_above_threshold'] = sub_count.cumsum()
  grouped['true_positive_at_or_above_threshold'] = (
    _per_decile('target').cumsum())
  grouped['precision_at_or_above_threshold'] = grouped.apply(
    lambda r: _pct(r['true_positive_at_or_above_threshold'],
                   r['customers_at_or_above_threshold']), axis=1)
  grouped['recall_at_or_above_threshold'] = grouped.apply(
    lambda r: _pct(r['true_positive_at_or_above_threshold'],
                   total_target), axis=1)
  for col in ('tp', 'fp', 'tn', 'fn'):
    grouped[col + '_above_threshold'] = _per_decile(col).cumsum()
  grouped['precision_above_threshold'] = grouped.apply(
    lambda r: _pct(r['tp_above_threshold'],
                   r['tp_above_threshold']
                   + r['fp_above_threshold']), axis=1)
  grouped['recall_above_threshold'] = grouped.apply(
    lambda r: _pct(r['tp_above_threshold'],
                   r['tp_above_threshold']
                   + r['fn_above_threshold']), axis=1)
  return grouped[COLUMNS]
