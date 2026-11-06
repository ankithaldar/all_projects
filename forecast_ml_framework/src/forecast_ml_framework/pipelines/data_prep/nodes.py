#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Data preparation nodes: validate, preprocess, select.

Public handle helper: :func:`resolve_handle` resolves dotted
importable strings used as ``handle`` / ``*_handle`` values in
the ``conf/base/parameters/*.yml`` buckets (e.g.
``model_handle``, ``calibration_handle``,
``tuning_handle``) under the same
allowlist as scoring (sklearn / xgboost / lightgbm / catboost /
h2o / forecast_ml_framework roots, estimator-like classes
with ``fit`` and ``predict`` only).

One-hot cap: ``CategoricalEncoding`` with ``ohe_flag: true``
expands via ``get_dummies``. The cap key is ``max_cols`` in
the step args (alias ``max_ohe_cols``), resolved with scoring
semantics: args win, then ``DATA_PREP_MAX_COLS`` /
``SCORING_MAX_COLS`` env, then default 1000; ``null`` means
unbounded. Pre-expansion per-column and total-expansion
estimates are checked against the cap (``ValueError`` naming
the column) and the result is re-checked after expansion.
'''

import logging
import os

import numpy as np
import pandas as pd

from forecast_ml_framework.core import spark_factory
from forecast_ml_framework.core.config_loader import ConfigLoader
from forecast_ml_framework.pipelines.scoring.nodes import (
  _resolve_handle_strict as _resolve_handle_strict,
)

LOG = logging.getLogger(__name__)

DEFAULT_MAX_OHE_COLS = 1000


def _resolve_ohe_max_cols(args):
  '''Resolve OHE column cap with scoring semantics.

  Args win (``max_cols``, alias ``max_ohe_cols``), then
  ``DATA_PREP_MAX_COLS`` / ``SCORING_MAX_COLS`` env, then
  :data:`DEFAULT_MAX_OHE_COLS`. ``None`` / ``null`` means
  unbounded.

  Args:
    args: CategoricalEncoding step args dict.

  Returns:
    Int cap, or None when unbounded.
  '''
  args = args or {}
  for key in ('max_cols', 'max_ohe_cols'):
    if key in args:
      value = args[key]
      return None if value is None else int(value)
  for env_key in ('DATA_PREP_MAX_COLS', 'SCORING_MAX_COLS'):
    env = os.environ.get(env_key)
    if env is not None:
      text = str(env).strip().lower()
      if text in ('', 'null', 'none'):
        return None
      return int(text)
  return DEFAULT_MAX_OHE_COLS


def resolve_handle(handle):
  '''Resolve a dotted handle under the allowlist (reused).

  Delegates to scoring's ``_resolve_handle_strict`` so both
  pipelines share one allowlist (sklearn / xgboost / lightgbm /
  catboost / h2o / forecast_ml_framework) and estimator check
  (a class with ``fit`` and ``predict``); handles such as
  ``os.system`` never resolve.

  Args:
    handle: Dotted importable string, e.g.
      'xgboost.XGBClassifier',
      'sklearn.ensemble.RandomForestClassifier' or
      'sklearn.calibration.CalibratedClassifierCV'.

  Returns:
    The estimator-like class named by handle.

  Raises:
    ValueError: If handle is not allowlisted, cannot be
      resolved, or is not an estimator-like class.
  '''
  return _resolve_handle_strict(handle)


def _resolve_cols(args, default):
  '''Resolve include minus exclude columns.'''
  cols = args.get('cols', {}) if isinstance(args, dict) else {}
  include = cols.get('include_cols') or list(default)
  exclude = set(cols.get('exclude_cols') or [])
  return [c for c in include if c not in exclude]


def _seq_list(val):
  '''Coerce a cell to list of floats.'''
  if val is None:
    return []
  if isinstance(val, (list, tuple, np.ndarray)):
    return [float(v) for v in val]
  if isinstance(val, str):
    return [float(v) for v in val.replace(
      '[', '').replace(']', '').split(',') if v.strip()]
  return [float(val)]


def _last_n(seq, last_n):
  '''Return last n elements of seq.'''
  return seq[-last_n:] if last_n else list(seq)


def synthetic_data(n=100, task='classification', seed=42):
  '''Generate synthetic demo dataframe.

  Args:
    n: Number of rows.
    task: 'classification' or 'regression'.
    seed: Random seed for reproducibility.

  Returns:
    DataFrame with f1..f4, cat, seq, target, id.
  '''
  rng = np.random.RandomState(seed)
  f1 = rng.normal(size=n)
  f2 = rng.normal(size=n)
  f3 = rng.normal(size=n)
  f4 = rng.normal(size=n)
  if task == 'regression':
    target = 2 * f1 + 0.5 * f2 + rng.normal(scale=0.1, size=n)
  else:
    logit = f1 + 0.5 * f2
    target = (logit > 0).astype(int)
  return pd.DataFrame({
    'id': np.arange(n),
    'f1': f1,
    'f2': f2,
    'f3': f3,
    'f4': f4,
    'cat': rng.choice(['a', 'b', 'c'], size=n),
    'seq': [[float(v) for v in rng.normal(size=3)]
            for _ in range(n)],
    'target': target,
  })


def validate_data(df, params, variables):
  '''Validate dataframe schema and distributions.

  Args:
    df: Input dataframe.
    params: Dict possibly holding ``validation_params`` with
      ``fail_fast``, ``required_cols``, ``max_null_ratio``.
    variables: Dict with target and feature columns.

  Returns:
    Tuple of (df, report dict).

  Raises:
    ValueError: If fail_fast and required columns missing.
  '''
  params = params or {}
  vparams = params.get('validation_params', params)
  fail_fast = bool(vparams.get('fail_fast', False))
  features = list(variables.get('feature_cols') or [])
  required = list(vparams.get('required_cols') or [])
  if not required:
    required = list(features)
  target = variables.get('target_col', 'target')
  if target and target not in required:
    required.append(target)
  cols = list(df.columns)
  missing = [c for c in required if c not in cols]
  report = {
    'n_rows': int(len(df)),
    'n_cols': int(len(cols)),
    'columns': cols,
    'missing_cols': missing,
    'null_rates': {
      c: float(df[c].isna().mean()) if c in cols else 1.0
      for c in cols},
    'target_distribution': (
      df[target].value_counts(dropna=False).to_dict()
      if target in cols else {}),
    'feature_cols_present': [c for c in features if c in cols],
  }
  max_null = vparams.get('max_null_ratio')
  if max_null is not None:
    report['high_null_cols'] = [
      c for c, r in report['null_rates'].items() if r > max_null]
  warnings = []
  all_null = [c for c, r in report['null_rates'].items() if r >= 1.0]
  for col in all_null:
    warnings.append('all-null column: %s' % col)
  dist = report.get('target_distribution') or {}
  if dist and len(dist) == 1:
    warnings.append('single-class target: %s' % list(dist)[0])
  report['warnings'] = warnings
  for warn in warnings:
    LOG.warning(warn)
  if missing:
    msg = 'Missing required columns: %s' % missing
    if fail_fast:
      raise ValueError(msg)
    LOG.warning(msg)
  return df, report


def _impute(df, args, variables):
  '''Apply imputation step.'''
  cols = _resolve_cols(args, variables.get('num_cols', []))
  strategy = args.get('strategy', 'none')
  bounds = args.get('custom_bounds') or {}
  for col in cols:
    if col not in df.columns:
      continue
    if strategy == 'mean':
      df[col] = df[col].fillna(df[col].mean())
    elif strategy == 'median':
      df[col] = df[col].fillna(df[col].median())
    elif strategy == 'mode':
      mode = df[col].mode(dropna=True)
      if len(mode):
        df[col] = df[col].fillna(mode.iloc[0])
    elif col in bounds:
      df[col] = df[col].fillna(bounds[col])
  return df


def _cap(df, args, variables):
  '''Apply cap outliers step.'''
  cols = _resolve_cols(args, variables.get('num_cols', []))
  bound = args.get('bound') or {}
  custom = args.get('custom_bound') or {}
  for col in cols:
    if col not in df.columns:
      continue
    if col in custom:
      lo, hi = custom[col]
    else:
      lo = df[col].quantile(bound.get('lower', 0.01))
      hi = df[col].quantile(bound.get('upper', 0.99))
    df[col] = df[col].clip(lower=lo, upper=hi)
  return df


def _discretize(df, args, variables):
  '''Apply quantile discretizer step.'''
  cols = _resolve_cols(args, variables.get('num_cols', []))
  buckets = (args.get('params') or {}).get('numBuckets', 10)
  for col in cols:
    if col not in df.columns:
      continue
    try:
      df[col] = pd.qcut(
        df[col], q=buckets, labels=False, duplicates='drop')
    except ValueError:
      df[col] = pd.factorize(df[col])[0]
  return df


def _normalize(df, args, variables):
  '''Apply normalization step.'''
  cols = _resolve_cols(args, variables.get('num_cols', []))
  handle = str(args.get('handle', 'MinMaxScaler'))
  for col in cols:
    if col not in df.columns:
      continue
    if 'Standard' in handle or 'ZScore' in handle:
      mu, sd = df[col].mean(), df[col].std() or 1.0
      df[col] = (df[col] - mu) / sd
    else:
      lo, hi = df[col].min(), df[col].max()
      rng = (hi - lo) or 1.0
      df[col] = (df[col] - lo) / rng
  return df


def _bin_labels(df, args, variables):
  '''Apply high label binning step.'''
  cols = _resolve_cols(args, variables.get('cat_cols', []))
  val_count = int(args.get('val_count', 300))
  for col in cols:
    if col not in df.columns:
      continue
    counts = df[col].value_counts()
    keep = set(counts[counts >= val_count].index)
    df[col] = df[col].where(df[col].isin(keep), 'Other')
  return df


def _encode_cat(df, args, variables):
  '''Apply categorical encoding step.

  With ``ohe_flag``, per-column cardinality and total
  expansion estimates are checked against the cap from
  :func:`_resolve_ohe_max_cols` before ``get_dummies`` and
  the expanded frame is re-checked after, mirroring scoring.

  Raises:
    ValueError: If expansion would exceed the cap, naming
      the offending categorical column(s).
  '''
  cols = _resolve_cols(args, variables.get('cat_cols', []))
  if not args.get('ohe_flag'):
    for col in cols:
      if col in df.columns:
        df[col] = pd.factorize(df[col])[0]
    return df
  present = [c for c in cols if c in df.columns]
  if not present:
    return df
  max_cols = _resolve_ohe_max_cols(args)
  if max_cols is not None:
    expanded = len(df.columns) - len(present)
    for col in present:
      distinct = int(df[col].nunique(dropna=False))
      if distinct > max_cols:
        raise ValueError(
          'data_prep input exceeds max_cols=%d: categorical '
          'column %r has %d distinct values'
          % (max_cols, col, distinct))
      expanded += distinct
    if expanded > max_cols:
      raise ValueError(
        'data_prep input exceeds max_cols=%d: get_dummies '
        'expansion would produce %d columns for categorical '
        'columns %r' % (max_cols, expanded, present))
  for col in present:
    dummies = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
  if max_cols is not None and len(df.columns) > max_cols:
    raise ValueError(
      'data_prep input exceeds max_cols=%d: got %d columns '
      'after get_dummies expansion for categorical columns %r'
      % (max_cols, len(df.columns), present))
  return df


def _target_encode(df, args, variables):
  '''Apply target encoding step (train only).'''
  cols = _resolve_cols(args, variables.get('cat_cols', []))
  target = variables.get('target_col', 'target')
  if target not in df.columns:
    return df
  for col in cols:
    if col not in df.columns:
      continue
    mapping = df.groupby(col)[target].mean()
    df[col] = df[col].map(mapping).astype(float)
  return df


def _seq_series(df, col):
  '''Return series of lists for a column.'''
  return df[col].apply(_seq_list)


def _fixed_length(df, args, variables):
  '''Apply fixed length step to sequences.'''
  cols = _resolve_cols(args, variables.get('seq_cols', []))
  length = int(args.get('fixed_length', 6))
  for col in cols:
    if col not in df.columns:
      continue
    df[col] = _seq_series(df, col).apply(
      lambda s: (s + [0.0] * length)[:length])
  return df


def _slope(df, args, variables):
  '''Apply slope step to sequences.'''
  cols = _resolve_cols(args, variables.get('seq_cols', []))
  last_n = int(args.get('last_n_val', 2))
  suffix = args.get('suffix', 'slope')
  for col in cols:
    if col not in df.columns:
      continue

    def _slope_val(seq):
      '''Compute slope of last segment of seq.'''
      seg = _last_n(seq, last_n)
      if len(seg) < 2:
        return 0.0
      x = np.arange(len(seg))
      return float(np.polyfit(x, seg, 1)[0])

    df['%s_%s' % (col, suffix)] = _seq_series(
      df, col).apply(_slope_val)
  return df


def _similarity(df, args, variables):
  '''Apply similarity step to sequences.'''
  cols = _resolve_cols(args, variables.get('seq_cols', []))
  last_n = int(args.get('last_n_val', 2))
  rmap = args.get('rational_vector_map') or {}
  for col in cols:
    if col not in df.columns:
      continue

    def _sim(seq):
      '''Compute weighted similarity of last segment.'''
      seg = _last_n(seq, last_n)
      if len(seg) < 2:
        return 1.0
      weights = [float(rmap.get(i, 1.0)) for i in range(len(seg))]
      dist = float(np.sqrt(np.sum(
        [(a - b) ** 2 * w for a, b, w in zip(
          seg[:-1], seg[1:], weights[:-1])])))
      return 1.0 / (1.0 + dist)

    df['%s_similarity' % col] = _seq_series(df, col).apply(_sim)
  return df


def _seq_agg(df, args, variables, kind):
  '''Apply sequence aggregation step.'''
  cols = _resolve_cols(args, variables.get('seq_cols', []))
  last_n = int(args.get('last_n_val', 2))
  suffix = {'average': 'avg', 'min': 'min', 'max': 'max'}[kind]
  for col in cols:
    if col not in df.columns:
      continue

    def _val(seq):
      '''Aggregate last segment of seq by kind.'''
      seg = _last_n(seq, last_n)
      if not seg:
        return 0.0
      fn = {'average': np.mean, 'min': np.min, 'max': np.max}[kind]
      return float(fn(seg))

    df['%s_%s' % (col, suffix)] = _seq_series(df, col).apply(_val)
  return df


def _seq_last(df, args, variables):
  '''Apply sequence last step.'''
  cols = _resolve_cols(args, variables.get('seq_cols', []))
  for col in cols:
    if col in df.columns:
      df['%s_last' % col] = _seq_series(df, col).apply(
        lambda s: float(s[-1]) if s else 0.0)
  return df


def _seq_delta(df, args, variables):
  '''Apply sequence delta step.'''
  cols = _resolve_cols(args, variables.get('seq_cols', []))
  last_n = int(args.get('last_n_val', 2))
  for col in cols:
    if col not in df.columns:
      continue

    def _d(seq):
      '''Compute delta of last segment of seq.'''
      seg = _last_n(seq, last_n)
      return float(seg[-1] - seg[-2]) if len(seg) >= 2 else 0.0

    df['%s_delta' % col] = _seq_series(df, col).apply(_d)
  return df


def _seq_normalize(df, args, variables):
  '''Apply sequence normalize step.'''
  cols = _resolve_cols(args, variables.get('seq_cols', []))
  places = int(args.get('round_off', 3))
  for col in cols:
    if col not in df.columns:
      continue

    def _n(seq):
      '''Normalize seq by peak absolute value.'''
      peak = max((abs(v) for v in seq), default=0.0) or 1.0
      return [round(v / peak, places) for v in seq]

    df[col] = _seq_series(df, col).apply(_n)
  return df


_REGISTRY = {
  'Imputations': _impute,
  'CapOutliers': _cap,
  'QuantileDiscretizer': _discretize,
  'Normalizations': _normalize,
  'HighLabelBinning': _bin_labels,
  'CategoricalEncoding': _encode_cat,
  'TargetEncoding': _target_encode,
  'FixedLength': _fixed_length,
  'Slope': _slope,
  'Similarity': _similarity,
  'SequenceAverage': lambda d, a, v: _seq_agg(d, a, v, 'average'),
  'SequenceMin': lambda d, a, v: _seq_agg(d, a, v, 'min'),
  'SequenceMax': lambda d, a, v: _seq_agg(d, a, v, 'max'),
  'SequenceLast': _seq_last,
  'SequenceDelta': _seq_delta,
  'SequenceNormalize': _seq_normalize,
}


def _iter_steps(prep_params):
  '''Yield (name, skip, args) from list or dict config.'''
  if isinstance(prep_params, dict):
    items = [{k: v} for k, v in prep_params.items()]
  else:
    items = list(prep_params or [])
  for item in items:
    for name, cfg in item.items():
      cfg = cfg or {}
      yield name, bool(cfg.get('skip', False)), cfg.get('args', {}) or {}


def preprocess(df, prep_params, variables,
               pandas_conversion_params=None,
               return_report=False):
  '''Apply preprocessing registry steps in order.

  Args:
    df: Pandas or Spark dataframe.
    prep_params: List of step dicts as in the
      ``conf/base/parameters/*.yml`` buckets.
    variables: Dict with column groups and target.
    pandas_conversion_params: Optional Spark conversion dict.
    return_report: If True, return ``(df, report)`` where
      report holds ``warnings`` and ``unknown_steps``.

  Returns:
    Preprocessed pandas dataframe, or tuple with report.
  '''
  warnings, unknown = [], []
  pandas_conversion_params = pandas_conversion_params or {}
  if (not isinstance(df, pd.DataFrame)
      and pandas_conversion_params.get('convert_to_pandas', True)
      is False):
    raise TypeError('Spark input requires convert_to_pandas=True')
  df = spark_factory.to_pandas(df, pandas_conversion_params).copy()
  for name, skip, args in _iter_steps(prep_params):
    if skip:
      continue
    fn = _REGISTRY.get(name)
    if fn is None:
      msg = 'Unknown preprocessing step: %s' % name
      LOG.warning(msg)
      warnings.append(msg)
      unknown.append(name)
      continue
    df = fn(df, args, variables)
  if return_report:
    return df, {'warnings': warnings, 'unknown_steps': unknown}
  return df


def select_features(train_df, params, variables):
  '''Select features via enabled methods.

  Args:
    train_df: Training dataframe.
    params: Dict possibly holding ``feature_selection_params``.
    variables: Dict with target and feature columns.

  Returns:
    Tuple of (selected df, selected list, report dict).
  '''
  params = params or {}
  fparams = params.get('feature_selection_params', params)
  target = variables.get('target_col', 'target')
  pandas_conversion_params = (
      fparams.get('pandas_conversion_params')
      or params.get('pandas_conversion_params') or {})
  if (not isinstance(train_df, pd.DataFrame)
      and isinstance(pandas_conversion_params, dict)
      and pandas_conversion_params.get('convert_to_pandas', True)
      is False):
    raise TypeError('Spark input requires convert_to_pandas=True')
  df = spark_factory.to_pandas(
      train_df, pandas_conversion_params).copy()
  base = list(variables.get('feature_cols') or [])
  candidates = [c for c in base if c in df.columns]
  report = {'initial': list(candidates), 'warnings': []}
  numeric = [c for c in candidates
             if pd.api.types.is_numeric_dtype(df[c])]
  # Variance / missingness filter.
  vt = fparams.get('variance_threshold')
  mt = fparams.get('missing_threshold')
  if vt is not None:
    keep = [c for c in numeric if df[c].var() > float(vt)]
    report['variance'] = keep
    candidates = [c for c in candidates
                  if c not in numeric or c in keep]
    numeric = keep
  if mt is not None:
    keep = [c for c in numeric if df[c].isna().mean() <= float(mt)]
    report['missingness'] = keep
    candidates = [c for c in candidates
                  if c not in numeric or c in keep]
    numeric = keep
  # Correlation pruning.
  corr_cfg = fparams.get('correlation') or {}
  if corr_cfg.get('enabled'):
    thr = float(corr_cfg.get('threshold', 0.95))
    drop = set()
    corr = df[numeric].corr().abs()
    for i, c1 in enumerate(numeric):
      for c2 in numeric[i + 1:]:
        if corr.loc[c1, c2] > thr:
          drop.add(c2)
    report['correlation_drop'] = sorted(drop)
    candidates = [c for c in candidates if c not in drop]
    numeric = [c for c in numeric if c not in drop]
  # SelectKBest.
  kb = fparams.get('kbest') or {}
  if kb.get('enabled') and numeric and target in df.columns:
    from sklearn.feature_selection import (
      SelectKBest,
      f_classif,
      f_regression,
      mutual_info_classif,
      mutual_info_regression,
    )
    funcs = {'f_classif': f_classif, 'f_regression': f_regression,
             'mutual_info_classif': mutual_info_classif,
             'mutual_info_regression': mutual_info_regression}
    func = funcs.get(kb.get('score_func', 'f_classif'), f_classif)
    k = min(int(kb.get('k', len(numeric))), len(numeric))
    selector = SelectKBest(score_func=func, k=k)
    x = df[numeric].fillna(0).values
    selector.fit(x, df[target].values)
    kept = [numeric[i] for i in selector.get_support(indices=True)]
    report['kbest'] = kept
    candidates = [c for c in candidates
                  if c not in numeric or c in kept]
    numeric = kept
  # L1 selection.
  l1 = fparams.get('l1') or {}
  if l1.get('enabled') and numeric and target in df.columns:
    from sklearn.linear_model import Lasso, LogisticRegression
    if fparams.get('is_regression') or str(
        variables.get('task', '')).startswith('reg'):
      model = Lasso(alpha=float(l1.get('alpha', 0.01)))
    else:
      model = LogisticRegression(
        penalty='l1', solver='liblinear',
        C=float(l1.get('C', 1.0)), max_iter=1000)
    model.fit(df[numeric].fillna(0).values, df[target].values)
    coef = np.ravel(model.coef_)
    kept = [c for c, v in zip(numeric, coef) if abs(v) > 1e-8]
    report['l1'] = kept
    candidates = [c for c in candidates
                  if c not in numeric or c in kept]
    numeric = kept
  # Tree importance.
  tree = fparams.get('tree') or {}
  if tree.get('enabled') and numeric and target in df.columns:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import RandomForestRegressor
    is_reg = str(variables.get('task', '')).startswith('reg')
    cls = RandomForestRegressor if is_reg else RandomForestClassifier
    model = cls(n_estimators=int(tree.get('n_estimators', 50)),
                random_state=42)
    model.fit(df[numeric].fillna(0).values, df[target].values)
    imp = model.feature_importances_
    thr = float(tree.get('threshold', 0.0))
    kept = [c for c, v in zip(numeric, imp) if v > thr]
    report['tree'] = kept
    candidates = [c for c in candidates
                  if c not in numeric or c in kept]
  # Core rule via ConfigLoader: override wins alone, else
  # (selected + business) deduped minus exclude. Translate
  # legacy business_relevant_features alias for core, which
  # only knows business_features, without duplicating rule.
  loader_params = dict(fparams)
  legacy = (variables.get('business_relevant_features')
            or fparams.get('business_relevant_features'))
  if legacy and not loader_params.get('business_features'):
    loader_params['business_features'] = list(legacy)
  loader_vars = dict(variables)
  if legacy and not loader_vars.get('business_features'):
    loader_vars['business_features'] = list(legacy)
  merged = ConfigLoader(
    variables=loader_vars,
    parameters=loader_params).merged_feature_list(
      selected=list(candidates))
  missing = [c for c in merged if c not in df.columns]
  for col in missing:
    report['warnings'].append(
      'selected feature not in data: %s' % col)
  selected = [c for c in merged if c in df.columns]
  if not selected:
    report['warnings'].append('no features selected')
  report['selected'] = selected
  # Preserve target and non-features, drop unselected features.
  drop = [c for c in variables.get('feature_cols', [])
          if c in df.columns and c not in selected]
  out = df.drop(columns=drop)
  return out, selected, report
