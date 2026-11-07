#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Monitoring / drift nodes (Phase 3).

Detects data, prediction, and performance drift between a
reference (baseline) window and a current window, and turns
drift into severity-graded alerts. All statistics are computed
in pure pandas/numpy (no extra services); empty or very small
inputs return NaN statistics plus a warning instead of raising.

Params keys (``params:monitoring_params`` dict, all optional):

- ``bins`` (int, default 10): number of quantile bins for
  numeric PSI. Aliases: ``n_bins``, ``num_bins``.
- ``epsilon`` (float, default 1e-4): proportion floor used
  when smoothing PSI bins/categories so log ratios stay finite.
- ``min_rows`` (int, default 10): inputs with fewer rows are
  treated as too small -> NaN stats + warning.
- ``feature_cols`` (list): restrict drift to these columns.
  Default is every column present in either frame.
- ``categorical_cols`` (list): force these columns to be
  treated as categorical; other non-numeric columns are
  categorical, numeric columns are numeric.
- ``baseline_metrics`` (dict): baseline metric values for
  :func:`performance_decay`, e.g. ``{'accuracy': 0.9}``.
- ``thresholds`` (dict): alert thresholds by metric, each a
  ``{'warning': float, 'critical': float}`` mapping (or a
  single float, meaning the warning threshold). Defaults:
  ``psi`` 0.1/0.25, ``ks`` 0.05/0.25,
  ``missing_rate_delta`` 0.05/0.1 (aligned to ``conf``
  ``psi_threshold`` / ``ks_threshold``). ``psi_threshold`` /
  ``ks_threshold`` scalars and ``alert_thresholds`` /
  ``severity_thresholds`` aliases are also accepted.
  Flat critical/missing keys (proposed conf/schema keys,
  see handoff to W10): ``psi_critical_threshold`` (0.25),
  ``ks_critical_threshold`` (0.25),
  ``missing_rate_delta_threshold`` (0.05),
  ``missing_rate_delta_critical_threshold`` (0.1).
  Flat keys override the ``thresholds`` dict.
- PSI proportions are clipped to ``epsilon`` and then
  renormalized so both reference and current vectors sum
  to 1 before the log-ratio sum (see :func:`_psi_from_props`).
  ``performance_decay`` reports ``decay`` with a per-metric
  direction: positive always means worse (see
  :func:`_is_lower_better`).
- ``prediction_col`` (str, default 'prediction'): column
  used by :func:`prediction_drift` for DataFrame inputs.
- ``prediction_type`` (str, default 'auto'): 'numeric',
  'categorical', or 'auto' (numeric when a numeric series has
  more than 10 distinct values, else categorical).
'''

import logging

import numpy as np
import pandas as pd

from forecast_ml_framework.core.params import (
  flatten_params,
)

LOG = logging.getLogger(__name__)


DEFAULT_BINS = 10
DEFAULT_EPSILON = 1e-4
MIN_EPSILON = 1e-9
DEFAULT_MIN_ROWS = 10
DEFAULT_THRESHOLDS = {
  'psi': {'warning': 0.1, 'critical': 0.25},
  'ks': {'warning': 0.05, 'critical': 0.25},
  'missing_rate_delta': {'warning': 0.05, 'critical': 0.1},
}
_PERFORMANCE_THRESHOLDS = {'warning': 0.05, 'critical': 0.1}


def _bins(params):
  '''Resolve the number of PSI bins from params.'''
  params = flatten_params(params, 'monitoring_params')
  for key in ('bins', 'n_bins', 'num_bins'):
    if params.get(key) is not None:
      raw = params[key]
      if isinstance(raw, bool):
        raise ValueError('bins must be a positive int, got %r' % (raw,))
      if isinstance(raw, float) and (
          not np.isfinite(raw) or not raw.is_integer()):
        raise ValueError('bins must be a positive int, got %r' % (raw,))
      try:
        bins = int(raw)
      except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
          'bins must be a positive int, got %r' % (raw,)) from exc
      if bins < 1:
        raise ValueError('bins must be a positive int, got %r'
                         % (raw,))
      return bins
  return DEFAULT_BINS


def _epsilon(params):
  '''Resolve the PSI smoothing epsilon from params.'''
  raw = flatten_params(
      params, 'monitoring_params').get('epsilon', DEFAULT_EPSILON)
  if isinstance(raw, bool):
    raise ValueError('epsilon must be a positive number, got %r'
                     % (raw,))
  try:
    epsilon = float(raw)
  except (TypeError, ValueError, OverflowError) as exc:
    raise ValueError('epsilon must be a positive number, got %r'
                     % (raw,)) from exc
  if not np.isfinite(epsilon) or epsilon <= 0:
    LOG.warning('monitoring: epsilon %r is not finite/positive;'
                ' clamping to %g', raw, MIN_EPSILON)
    return MIN_EPSILON
  return max(epsilon, MIN_EPSILON)


def _min_rows(params):
  '''Resolve the small-input row threshold from params.'''
  raw = flatten_params(
      params, 'monitoring_params').get('min_rows', DEFAULT_MIN_ROWS)
  try:
    return int(raw)
  except (TypeError, ValueError, OverflowError) as exc:
    raise ValueError('min_rows must be a positive int, got %r'
                     % (raw,)) from exc


def _clipped_props(props, epsilon):
  '''Clip proportions to ``epsilon`` then renormalize to sum 1.

  Args:
    props: Aligned proportion array (reference or current).
    epsilon: Proportion floor applied before renormalizing.

  Returns:
    Float array with the same shape whose entries sum to 1
    (empty input yields an empty array). Clipping first
    keeps log ratios finite; renormalizing removes the
    epsilon bias that otherwise makes clipped sums exceed
    1 (e.g. 100 zero bins at epsilon 0.01 sum to 2.0 before
    renormalization and to 1.0 after).

  Raises:
    None: Non-finite or non-positive ``epsilon`` falls back
      to ``MIN_EPSILON``; a non-finite total is returned
      unrenormalized so callers surface NaN PSI.
  '''
  try:
    epsilon = float(epsilon)
  except (TypeError, ValueError, OverflowError):
    epsilon = MIN_EPSILON
  if not np.isfinite(epsilon) or epsilon <= 0:
    epsilon = MIN_EPSILON
  epsilon = max(epsilon, MIN_EPSILON)
  clipped = np.clip(np.asarray(props, dtype=float), epsilon, None)
  if clipped.size == 0:
    return clipped
  total = float(np.sum(clipped))
  if np.isfinite(total) and total > 0:
    return clipped / total
  LOG.warning('monitoring: clipped proportions sum to %r;'
              ' returning unrenormalized values', total)
  return clipped


def _psi_from_props(ref_props, cur_props, epsilon):
  '''Compute PSI from two aligned proportion arrays.

  Both vectors are clipped to ``epsilon`` and renormalized
  to sum 1 (see :func:`_clipped_props`), so PSI is computed
  on genuine distributions. Larger ``epsilon`` smooths
  more and yields a smaller (more conservative) PSI for
  the same shift; the effect is monotone in practice for
  fixed inputs.
  '''
  ref = _clipped_props(ref_props, epsilon)
  cur = _clipped_props(cur_props, epsilon)
  with np.errstate(invalid='ignore', divide='ignore'):
    value = float(np.sum((cur - ref) * np.log(cur / ref)))
  if not np.isfinite(value):
    LOG.warning('monitoring: PSI is non-finite (%r); returning NaN',
                value)
    return float('nan')
  return value


def _numeric_psi(ref, cur, bins, epsilon):
  '''Compute PSI with reference-quantile bins.'''
  if not (np.all(np.isfinite(ref)) and np.all(np.isfinite(cur))):
    LOG.warning('monitoring: numeric PSI inputs contain'
                ' non-finite values; returning NaN PSI')
    return float('nan')
  edges = np.unique(np.quantile(
    ref, np.linspace(0.0, 1.0, bins + 1)))
  if len(edges) < 3:
    return _categorical_psi(ref, cur, epsilon)
  edges = edges.astype(float)
  edges[0], edges[-1] = -np.inf, np.inf
  ref_counts, _ = np.histogram(ref, bins=edges)
  cur_counts, _ = np.histogram(cur, bins=edges)
  return _psi_from_props(ref_counts / len(ref),
                         cur_counts / len(cur), epsilon)


def _categorical_psi(ref, cur, epsilon):
  '''Compute PSI over the union of categories.'''
  ref = pd.Series(ref)
  cur = pd.Series(cur)
  categories = sorted(set(ref) | set(cur), key=str)
  ref_props = ref.value_counts(normalize=True)
  cur_props = cur.value_counts(normalize=True)
  return _psi_from_props(
    [ref_props.get(cat, 0.0) for cat in categories],
    [cur_props.get(cat, 0.0) for cat in categories], epsilon)


def _ks_statistic(ref, cur):
  '''Compute the two-sample KS statistic manually.'''
  ref_sorted = np.sort(np.asarray(ref, dtype=float))
  cur_sorted = np.sort(np.asarray(cur, dtype=float))
  pooled = np.unique(np.concatenate([ref_sorted, cur_sorted]))
  ref_cdf = np.searchsorted(ref_sorted, pooled, side='right')
  cur_cdf = np.searchsorted(cur_sorted, pooled, side='right')
  diff = np.abs(ref_cdf / len(ref_sorted)
                - cur_cdf / len(cur_sorted))
  return float(np.max(diff)) if len(diff) else float('nan')


def _categorical_ks(ref, cur):
  '''Total-variation distance as a categorical KS analogue.'''
  ref = pd.Series(ref)
  cur = pd.Series(cur)
  categories = set(ref) | set(cur)
  ref_props = ref.value_counts(normalize=True)
  cur_props = cur.value_counts(normalize=True)
  total = sum(abs(ref_props.get(cat, 0.0) - cur_props.get(cat, 0.0))
              for cat in categories)
  return float(0.5 * total)


def _empty_stats(kind, warning):
  '''Build a NaN stats dict for empty/small inputs.'''
  LOG.warning(warning)
  nan = float('nan')
  return {
    'type': kind,
    'psi': nan,
    'ks': nan,
    'ks_statistic': nan,
    'mean_shift': nan,
    'mean_shift_std': nan,
    'std_shift': nan,
    'std_shift_std': nan,
    'missing_rate_reference': nan,
    'missing_rate_current': nan,
    'missing_rate_delta': nan,
    'missing_delta': nan,
  }


def _feature_stats(ref_col, cur_col, params, forced_cat):
  '''Compute drift stats for a single feature column.'''
  ref = pd.Series(ref_col)
  cur = pd.Series(cur_col)
  numeric = (pd.api.types.is_numeric_dtype(ref)
             and not pd.api.types.is_bool_dtype(ref)
             and not forced_cat)
  kind = 'numeric' if numeric else 'categorical'
  missing_ref = float(ref.isna().mean()) if len(ref) else float('nan')
  missing_cur = float(cur.isna().mean()) if len(cur) else float('nan')
  ref_clean = ref.dropna()
  cur_clean = cur.dropna()
  if len(ref_clean) < _min_rows(params) or len(cur_clean) < _min_rows(
          params):
    stats = _empty_stats(
      kind,
      'compute_drift: small input for column %r (ref=%d, cur=%d rows);'
      ' returning NaN stats' % (ref.name, len(ref_clean),
                                len(cur_clean)))
    stats['missing_rate_reference'] = missing_ref
    stats['missing_rate_current'] = missing_cur
    stats['missing_rate_delta'] = missing_cur - missing_ref
    stats['missing_delta'] = stats['missing_rate_delta']
    return stats
  bins = _bins(params)
  epsilon = _epsilon(params)
  if numeric:
    ref_num = ref_clean.astype(float).to_numpy()
    cur_num = cur_clean.astype(float).to_numpy()
    psi = _numeric_psi(ref_num, cur_num, bins, epsilon)
    ks = _ks_statistic(ref_num, cur_num)
    mean_ref, mean_cur = float(np.mean(ref_num)), float(np.mean(cur_num))
    std_ref, std_cur = float(np.std(ref_num)), float(np.std(cur_num))
    denom = std_ref if std_ref > 0 else 1.0
    return {
      'type': kind,
      'psi': psi,
      'ks': ks,
      'ks_statistic': ks,
      'mean_shift': mean_cur - mean_ref,
      'mean_shift_std': (mean_cur - mean_ref) / denom,
      'std_shift': std_cur - std_ref,
      'std_shift_std': (std_cur - std_ref) / denom,
      'mean_reference': mean_ref,
      'mean_current': mean_cur,
      'std_reference': std_ref,
      'std_current': std_cur,
      'missing_rate_reference': missing_ref,
      'missing_rate_current': missing_cur,
      'missing_rate_delta': missing_cur - missing_ref,
      'missing_delta': missing_cur - missing_ref,
    }
  ref_cat = ref_clean.astype(str)
  cur_cat = cur_clean.astype(str)
  psi = _categorical_psi(ref_cat, cur_cat, epsilon)
  ks = _categorical_ks(ref_cat, cur_cat)
  nan = float('nan')
  return {
    'type': kind,
    'psi': psi,
    'ks': ks,
    'ks_statistic': ks,
    'mean_shift': nan,
    'mean_shift_std': nan,
    'std_shift': nan,
    'std_shift_std': nan,
    'missing_rate_reference': missing_ref,
    'missing_rate_current': missing_cur,
    'missing_rate_delta': missing_cur - missing_ref,
    'missing_delta': missing_cur - missing_ref,
  }


def _as_dataframe(value, label):
  '''Coerce a dataframe-like value to a DataFrame.'''
  if isinstance(value, pd.DataFrame):
    return value
  try:
    return pd.DataFrame(value)
  except (TypeError, ValueError) as exc:
    raise ValueError(
      'compute_drift: %s must be a DataFrame or a dict of'
      ' equal-length columns, got %r (%s)' % (label, value, exc)
    ) from exc


def compute_drift(reference_df, current_df, params=None):
  '''Compute per-feature drift between two dataframes.

  Args:
    reference_df: Baseline dataframe.
    current_df: Current dataframe to compare against.
    params: Optional dict; see module docstring for keys
      (``bins``, ``epsilon``, ``min_rows``, ``feature_cols``,
      ``categorical_cols``).

  Returns:
    Dict mapping each feature name to a stats dict with keys
    ``type``, ``psi``, ``ks`` / ``ks_statistic``,
    ``mean_shift`` (raw and ``mean_shift_std`` standardised),
    ``std_shift``, ``missing_rate_reference`` /
    ``missing_rate_current`` / ``missing_rate_delta``
    (alias ``missing_delta``). Numeric features use quantile
    bins for PSI; categorical features use category
    proportions for PSI and total variation for KS. Empty
    or small inputs (fewer than ``min_rows`` non-null rows)
    return NaN stats plus a warning, never an exception.
  '''
  params = flatten_params(params, 'monitoring_params')
  _bins(params)
  _epsilon(params)
  _min_rows(params)
  reference = _as_dataframe(reference_df, 'reference_df')
  current = _as_dataframe(current_df, 'current_df')
  if reference.empty or current.empty:
    cols = list(dict.fromkeys(
      list(reference.columns) + list(current.columns)))
    if not cols:
      LOG.warning('compute_drift: empty input with no columns;'
                  ' returning empty drift report')
      return {}
    return {col: _empty_stats(
      'numeric',
      'compute_drift: empty input; column %r returns NaN stats'
      % (col,)) for col in cols}
  cols = params.get('feature_cols') or list(
    dict.fromkeys(list(reference.columns) + list(current.columns)))
  forced = set(params.get('categorical_cols') or [])
  report = {}
  for col in cols:
    ref_col = (reference[col] if col in reference.columns
               else pd.Series([np.nan] * len(reference), name=col))
    cur_col = (current[col] if col in current.columns
               else pd.Series([np.nan] * len(current), name=col))
    report[col] = _feature_stats(ref_col, cur_col, params, col in forced)
  return report


def _as_series(values, params):
  '''Coerce predictions input to a pandas Series.'''
  params = flatten_params(params, 'monitoring_params')
  col = params.get('prediction_col', 'prediction')
  if isinstance(values, pd.DataFrame):
    if col in values.columns:
      return pd.Series(values[col].to_numpy())
    if 'proba' in values.columns:
      return pd.Series(values['proba'].to_numpy())
    if len(values.columns):
      return pd.Series(values.iloc[:, 0].to_numpy())
    return pd.Series([], dtype=float)
  if isinstance(values, pd.Series):
    return pd.Series(values.to_numpy())
  if isinstance(values, dict) and col in values:
    return pd.Series(list(values[col]))
  if values is None:
    return pd.Series([], dtype=float)
  return pd.Series(list(values))


def _prediction_stats(ref, cur, params):
  '''Compute PSI/KS for one pair of prediction series.'''
  ref_clean = ref.dropna()
  cur_clean = cur.dropna()
  if len(ref_clean) < _min_rows(params) or len(cur_clean) < _min_rows(
          params):
    stats = _empty_stats(
      'numeric',
      'prediction_drift: empty/small predictions (ref=%d, cur=%d);'
      ' returning NaN stats' % (len(ref_clean), len(cur_clean)))
    return stats
  ptype = flatten_params(
      params, 'monitoring_params').get('prediction_type', 'auto')
  numeric = pd.api.types.is_numeric_dtype(ref_clean)
  if ptype == 'auto':
    numeric = numeric and ref_clean.nunique() > 10
  epsilon = _epsilon(params)
  if numeric and ptype != 'categorical':
    ref_num = ref_clean.astype(float).to_numpy()
    cur_num = cur_clean.astype(float).to_numpy()
    psi = _numeric_psi(ref_num, cur_num, _bins(params), epsilon)
    ks = _ks_statistic(ref_num, cur_num)
    mean_ref = float(np.mean(ref_num))
    mean_cur = float(np.mean(cur_num))
    std_ref = float(np.std(ref_num))
    denom = std_ref if std_ref > 0 else 1.0
    return {
      'type': 'numeric',
      'psi': psi,
      'ks': ks,
      'ks_statistic': ks,
      'mean_shift': mean_cur - mean_ref,
      'mean_shift_std': (mean_cur - mean_ref) / denom,
      'std_shift': float(np.std(cur_num)) - std_ref,
      'mean_reference': mean_ref,
      'mean_current': mean_cur,
      'missing_rate_reference': float(ref.isna().mean()),
      'missing_rate_current': float(cur.isna().mean()),
      'missing_rate_delta': float(cur.isna().mean()
                                   - ref.isna().mean()),
      'missing_delta': float(cur.isna().mean() - ref.isna().mean()),
    }
  ref_cat = ref_clean.astype(str)
  cur_cat = cur_clean.astype(str)
  psi = _categorical_psi(ref_cat, cur_cat, epsilon)
  ks = _categorical_ks(ref_cat, cur_cat)
  return {
    'type': 'categorical',
    'psi': psi,
    'ks': ks,
    'ks_statistic': ks,
    'mean_shift': float('nan'),
    'mean_shift_std': float('nan'),
    'std_shift': float('nan'),
    'std_shift_std': float('nan'),
    'missing_rate_reference': float(ref.isna().mean()),
    'missing_rate_current': float(cur.isna().mean()),
    'missing_rate_delta': float(cur.isna().mean()
                                 - ref.isna().mean()),
    'missing_delta': float(cur.isna().mean() - ref.isna().mean()),
  }


def prediction_drift(reference=None, current=None, params=None,
                     **kwargs):
  '''Compute drift of the prediction distribution.

  Args:
    reference: Reference predictions: array-like, Series, or
      DataFrame with a ``prediction`` column.
    current: Current predictions, same shapes as ``reference``.
    params: Optional dict; see module docstring (``bins``,
      ``epsilon``, ``min_rows``, ``prediction_col``,
      ``prediction_type``).
    **kwargs: Aliases ``reference_predictions`` /
      ``current_predictions`` / ``ref`` / ``cur`` when the
      positional arguments are omitted.

  Returns:
    Flat stats dict (same keys as :func:`compute_drift`
    feature stats) plus a ``prediction`` entry holding the
    same stats, so callers may use either shape. Empty/small
    inputs return NaN stats plus a warning.
  '''
  if reference is None:
    reference = kwargs.get('reference_predictions', kwargs.get('ref'))
  if current is None:
    current = kwargs.get('current_predictions', kwargs.get('cur'))
  params = flatten_params(params, 'monitoring_params')
  _bins(params)
  _epsilon(params)
  _min_rows(params)
  ref = _as_series(reference, params)
  cur = _as_series(current, params)
  stats = _prediction_stats(ref, cur, params)
  stats['prediction'] = dict(stats)
  return stats


def _is_lower_better(metric):
  '''Return True when lower values of ``metric`` are better.

  Args:
    metric: Metric name (case-insensitive; ``-``/space are
      normalized to ``_``).

  Returns:
    True for metrics in ``metrics.LOWER_BETTER`` (RMSE, MAE,
    log-loss, ...), False for higher-is-better metrics
    (accuracy, F1, AUC, ...). The r2 family (``r2``,
    ``adj_r2``, ``r2_log``, ...) is always higher-is-better
    even if a future ``LOWER_BETTER`` entry lists it.
  '''
  name = str(metric).strip().lower().replace('-', '_')
  name = name.replace(' ', '_')
  if 'r2' in name:
    return False
  try:
    # pylint: disable=import-outside-toplevel
    from forecast_ml_framework.pipelines.modeling.metrics import (
      LOWER_BETTER)
    # pylint: enable=import-outside-toplevel
  except ImportError:
    LOG.warning('monitoring: metrics.LOWER_BETTER unavailable;'
                ' treating %r as higher-is-better', metric)
    return False
  return name in LOWER_BETTER


def _decay_for(metric, delta):
  '''Return direction-adjusted decay for ``metric`` and ``delta``.

  Args:
    metric: Metric name (see :func:`_is_lower_better`).
    delta: ``current - baseline`` for the metric.

  Returns:
    Decay where positive always means worse: ``delta`` for
    lower-is-better metrics (an increase is worse) and
    ``-delta`` for higher-is-better metrics (a drop is
    worse).
  '''
  return float(delta) if _is_lower_better(metric) else float(-delta)


def performance_decay(current_metrics=None, baseline_metrics=None,
                      params=None, **kwargs):
  '''Compare current metrics against baseline metrics.

  Args:
    current_metrics: Dict of current metric values, e.g.
      ``{'accuracy': 0.8}``. May also be the params dict when
      it carries ``current_metrics``.
    baseline_metrics: Dict of baseline metric values, or the
      params dict (when it carries ``baseline_metrics``).
    params: Optional dict; ``baseline_metrics`` inside it
      wins when no explicit baseline dict is given.
    **kwargs: Aliases ``baseline`` / ``current``.

  Returns:
    Dict mapping each metric to ``current``, ``baseline``,
    ``delta`` (current - baseline), ``decay`` (positive
    always means worse, direction-adjusted per metric via
    :func:`_is_lower_better`), ``lower_better`` (bool), and
    ``relative_delta`` (raw ``delta / baseline``). Empty
    input returns ``{}`` plus a warning, never an
    exception.
  '''
  params = flatten_params(params, 'monitoring_params')
  if callable(current_metrics):
    current_metrics = {}
  if isinstance(current_metrics, dict) and (
          'current_metrics' in current_metrics
          and not any(isinstance(v, (int, float))
                      for v in current_metrics.values())):
    params = {**params, **current_metrics}
    current_metrics = params.get('current_metrics')
  if isinstance(baseline_metrics, dict) and (
          'baseline_metrics' in baseline_metrics
          or 'thresholds' in baseline_metrics
          or 'bins' in baseline_metrics):
    params = {**params, **baseline_metrics}
    baseline_metrics = params.get('baseline_metrics')
  if baseline_metrics is None:
    baseline_metrics = params.get('baseline_metrics')
  if current_metrics is None:
    current_metrics = kwargs.get('current') or {}
  if isinstance(baseline_metrics, dict) and (
          baseline_metrics is params):
    baseline_metrics = params.get('baseline_metrics')
  if baseline_metrics is None:
    baseline_metrics = kwargs.get('baseline') or {}
  if not isinstance(current_metrics, dict):
    raise ValueError('current_metrics must be a dict, got %r'
                     % (current_metrics,))
  if not isinstance(baseline_metrics, dict):
    raise ValueError('baseline_metrics must be a dict, got %r'
                     % (baseline_metrics,))
  current_metrics = dict(current_metrics or {})
  baseline_metrics = dict(baseline_metrics or {})
  if not current_metrics and not baseline_metrics:
    LOG.warning('performance_decay: empty metrics; returning empty'
                ' report')
    return {}
  report = {}

  def _nan_entry(cur_entry, base_entry):
    '''Build a NaN decay entry preserving raw values.'''
    return {'current': cur_entry, 'baseline': base_entry,
            'delta': float('nan'), 'decay': float('nan'),
            'lower_better': _is_lower_better(metric),
            'relative_delta': float('nan')}

  for metric in dict.fromkeys(
          list(baseline_metrics) + list(current_metrics)):
    cur = current_metrics.get(metric)
    base = baseline_metrics.get(metric)
    if cur is None or base is None:
      LOG.warning('performance_decay: metric %r missing on one side;'
                  ' returning NaN delta' % (metric,))
      report[metric] = _nan_entry(cur, base)
      continue
    try:
      cur_value = float(cur)
      base_value = float(base)
    except (TypeError, ValueError, OverflowError):
      LOG.warning('performance_decay: metric %r has non-numeric'
                  ' values (current=%r, baseline=%r); returning NaN'
                  ' delta' % (metric, cur, base))
      report[metric] = _nan_entry(cur, base)
      continue
    if not (np.isfinite(cur_value) and np.isfinite(base_value)):
      LOG.warning('performance_decay: metric %r has non-finite'
                  ' values (current=%r, baseline=%r); returning NaN'
                  ' delta' % (metric, cur, base))
      report[metric] = _nan_entry(cur_value, base_value)
      continue
    delta = cur_value - base_value
    if not np.isfinite(delta):
      LOG.warning('performance_decay: metric %r delta is non-finite;'
                  ' returning NaN delta' % (metric,))
      report[metric] = _nan_entry(cur_value, base_value)
      continue
    relative = (delta / base_value if base_value != 0
                else float('nan'))
    if not np.isfinite(relative):
      relative = float('nan')
    report[metric] = {
      'current': cur_value,
      'baseline': base_value,
      'delta': delta,
      'decay': _decay_for(metric, delta),
      'lower_better': _is_lower_better(metric),
      'relative_delta': relative,
    }
  return report


def _threshold_pair(value):
  '''Normalise a threshold spec to (warning, critical).'''
  if isinstance(value, dict):
    warning = value.get('warning', value.get('low', value.get('medium')))
    critical = value.get('critical', value.get('high'))
    return (float(warning) if warning is not None else None,
            float(critical) if critical is not None else None)
  if value is not None:
    return float(value), None
  return None, None


def _resolve_thresholds(params):
  '''Merge supplied thresholds over the documented defaults.'''
  params = flatten_params(params, 'monitoring_params')
  raw = {}
  for key in ('thresholds', 'alert_thresholds', 'severity_thresholds',
              'thresholds_by_severity'):
    if isinstance(params.get(key), dict):
      raw.update(params[key])
  merged = {k: dict(v) for k, v in DEFAULT_THRESHOLDS.items()}
  if 'warning' in raw or 'critical' in raw:
    # A bare severity map applies to psi/ks.
    for metric in ('psi', 'ks'):
      warning, critical = _threshold_pair(raw)
      if warning is not None:
        merged[metric]['warning'] = warning
      if critical is not None:
        merged[metric]['critical'] = critical
  for metric, spec in raw.items():
    if metric in ('warning', 'critical'):
      continue
    warning, critical = _threshold_pair(spec)
    entry = merged.setdefault(metric, {})
    if warning is not None:
      entry['warning'] = warning
    if critical is not None:
      entry['critical'] = critical
  for metric in ('psi', 'ks', 'missing_rate_delta',
                 'missing_delta'):
    scalar = params.get(metric + '_threshold')
    if scalar is not None:
      merged[metric]['warning'] = float(scalar)
  # Flat critical keys (proposed conf/schema keys for W10):
  # ``psi_critical_threshold``, ``ks_critical_threshold``,
  # ``missing_rate_delta_critical_threshold``. Flat keys win
  # over the ``thresholds`` dict, matching the scalar rule.
  for metric in ('psi', 'ks', 'missing_rate_delta',
                 'missing_delta'):
    critical = params.get(metric + '_critical_threshold')
    if critical is not None:
      merged[metric]['critical'] = float(critical)
  merged['missing_delta'] = merged['missing_rate_delta']
  return merged


def _alert(feature, metric, value, thresholds):
  '''Build one alert dict, or None when below warning.'''
  if value is None or (isinstance(value, float)
                       and np.isnan(value)):
    return None
  magnitude = abs(float(value))
  warning = thresholds.get('warning')
  critical = thresholds.get('critical')
  if critical is not None and magnitude >= critical:
    return {'feature': feature, 'metric': metric,
            'value': float(value), 'threshold': float(critical),
            'severity': 'critical'}
  if warning is not None and magnitude >= warning:
    return {'feature': feature, 'metric': metric,
            'value': float(value), 'threshold': float(warning),
            'severity': 'warning'}
  return None


def _alerts_for_stats(feature, stats, thresholds):
  '''Build alerts for one feature stats dict.'''
  alerts = []
  if not isinstance(stats, dict):
    return alerts
  for metric, spec in thresholds.items():
    if metric == 'missing_delta':
      continue  # alias of missing_rate_delta, checked once
    if metric in stats:
      alert = _alert(feature, metric, stats[metric], spec)
      if alert is not None:
        alerts.append(alert)
  return alerts


def evaluate_alerts(drift_report=None, params=None, **kwargs):
  '''Turn a drift report into severity-graded alerts.

  Args:
    drift_report: Output of :func:`compute_drift` (per-feature
      stats), or a combined report from
      :func:`build_monitoring_report` with ``features``,
      ``prediction``, and ``performance`` sections.
    params: Optional dict; see module docstring for the
      ``thresholds`` keys. Defaults: psi warning 0.1,
      critical 0.25; ks warning 0.05, critical 0.25;
      missing-rate delta 0.05/0.1; performance
      delta 0.05/0.1.
    **kwargs: Alias ``report`` for ``drift_report``.

  Returns:
    List of alert dicts ``{'feature', 'metric', 'value',
    'threshold', 'severity'}`` where severity is 'warning'
    or 'critical'. NaN values never alert.
  '''
  if drift_report is None:
    drift_report = kwargs.get('report') or {}
  params = flatten_params(params, 'monitoring_params')
  thresholds = _resolve_thresholds(params)
  report = drift_report or {}
  alerts = []
  if not isinstance(report, dict):
    LOG.warning('evaluate_alerts: empty/invalid drift report;'
                ' returning no alerts')
    return alerts
  features = report.get('features')
  if isinstance(features, dict):
    for feature, stats in features.items():
      alerts.extend(_alerts_for_stats(feature, stats, thresholds))
  else:
    for feature, stats in report.items():
      if feature in ('prediction', 'performance', 'features',
                     'drift'):
        continue
      alerts.extend(_alerts_for_stats(feature, stats, thresholds))
  prediction = report.get('prediction')
  if isinstance(prediction, dict) and 'psi' in prediction:
    alerts.extend(_alerts_for_stats('prediction', prediction,
                                    thresholds))
  performance = report.get('performance')
  if isinstance(performance, dict):
    for metric, stats in performance.items():
      if not isinstance(stats, dict) or 'delta' not in stats:
        continue
      spec = thresholds.get('delta') or _PERFORMANCE_THRESHOLDS
      alert = _alert(metric, 'delta', stats['delta'], spec)
      if alert is not None:
        alerts.append(alert)
  return alerts


def build_monitoring_report(drift_report=None,
                            reference_predictions=None,
                            current_predictions=None,
                            current_metrics=None, params=None,
                            **kwargs):
  '''Combine drift, prediction, and performance sections.

  Args:
    drift_report: Per-feature stats from :func:`compute_drift`.
    reference_predictions: Reference predictions (see
      :func:`prediction_drift`).
    current_predictions: Current predictions.
    current_metrics: Current metric dict for
      :func:`performance_decay` (baseline from params).
    params: Monitoring params dict (see module docstring).
    **kwargs: Aliases ``predictions`` as a
      ``(reference, current)`` pair and ``metrics`` for
      ``current_metrics``.

  Returns:
    Dict with ``features`` (the drift report), ``prediction``
    (prediction drift stats), and ``performance`` (decay
    report) sections -- the input shape of
    :func:`evaluate_alerts`.
  '''
  params = flatten_params(params, 'monitoring_params')
  if 'predictions' in kwargs:
    pair = kwargs['predictions']
    reference_predictions, current_predictions = pair[0], pair[1]
  if current_metrics is None:
    current_metrics = kwargs.get('metrics')
  return {
    'features': drift_report or {},
    'prediction': prediction_drift(
      reference_predictions, current_predictions, params),
    'performance': performance_decay(current_metrics, params),
  }
