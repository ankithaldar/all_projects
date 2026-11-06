#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Exploratory data analysis nodes (Phase 3).

Standalone summary statistics computed in pure pandas/numpy.
When a dataframe is given to :func:`eda_report`,
``make_data_charts`` from ``pipelines.modeling.charts`` is
called directly and its charts are merged into the report
as ``data_charts``, so this pipeline uses the charts
worker's module without a defensive probe.

Params keys (``params:eda_params`` dict, all optional):

- ``target_col`` (str, default 'target'): target column name.
  ``variables['target_col']`` wins when present.
- ``missing_threshold`` (float, default 0.2): missing rate above
  which :func:`eda_quality_checks` flags ``high_missing``.
- ``cardinality_threshold`` (int, default 50): unique-value count
  above which a categorical column flags
  ``cardinality_explosion``. Alias: ``max_cardinality``.
- ``leakage_threshold`` (float, default 0.95): absolute
  feature-target correlation above which a feature flags
  ``leakage_suspect``.
- ``leakage_corr_threshold`` (float, default 0.95): alias for
  ``leakage_threshold`` used by :func:`detect_leakage`.
- ``duplicate_corr_threshold`` (float, default 0.99):
  feature-feature correlation above which a column flags
  as a near-duplicate in :func:`detect_leakage`.
- ``leakage_auc_threshold`` (float, default 0.99): AUC above
  which a numeric feature flags as perfectly separating
  the target in :func:`detect_leakage`.
- ``near_constant_threshold`` (float, default 0.95):
  dominant-value proportion above which a feature flags
  as constant / near-constant in :func:`detect_leakage`.
- ``leakage_name_keywords`` (list, default ['target',
  'attrition', 'label', 'outcome', 'result', 'after', 'post']):
  name substrings flagging a feature as a post-outcome
  suspect in :func:`detect_leakage`.
- ``imbalance_ratio_threshold`` (float, default 5.0): majority /
  minority target-count ratio above which the target flags
  ``class_imbalance``.
- ``zscore_threshold`` (float, default 3.0): |z-score| above
  which a numeric value counts as a z-score outlier.
- ``top_correlations`` (int, default 10): number of correlation
  pairs to keep in the summary. Alias: ``n_top_correlations``.
- ``top_values`` (int, default 3): number of top values kept
  per categorical column.
'''

import logging

import numpy as np
import pandas as pd

from forecast_ml_framework.core.params import (
  flatten_params,
)

LOG = logging.getLogger(__name__)


_RECOMMENDATIONS = {
  'high_missing': 'Investigate or impute columns with high missingness.',
  'cardinality_explosion': 'Encode or bin high-cardinality columns.',
  'leakage_suspect': 'Drop or verify leakage-suspect features.',
  'leakage': 'drop or investigate',
  'class_imbalance': 'Consider resampling or class weights.',
  'constant_cols': 'Drop constant columns with no signal.',
  'duplicates': 'Deduplicate rows before training.',
}

_DEFAULT_LEAKAGE_KEYWORDS = [
  'target', 'attrition', 'label', 'outcome', 'result', 'after', 'post',
]


def _float(value, default):
  '''Return value as float, or default when missing/invalid.'''
  try:
    return float(value)
  except (TypeError, ValueError):
    return float(default)


def _int(value, default):
  '''Return value as int, or default when missing/invalid.'''
  try:
    return int(value)
  except (TypeError, ValueError):
    return int(default)


def _scalar(value):
  '''Return a plain Python scalar for a numpy/pandas value.'''
  if isinstance(value, np.generic):
    return value.item()
  if value is None or isinstance(value, (bool, int, float, str)):
    return value
  return str(value)


def _target_col(params, variables, df):
  '''Resolve the target column name, or None when absent.'''
  variables = variables or {}
  params = flatten_params(params, 'eda_params')
  col = variables.get('target_col') or params.get('target_col')
  if col and col in df.columns:
    return col
  if 'target' in df.columns:
    return 'target'
  return col if col in df.columns else None


def _is_numeric(df, col):
  '''True for numeric (non-bool) columns.'''
  return (pd.api.types.is_numeric_dtype(df[col])
          and not pd.api.types.is_bool_dtype(df[col]))


def _empty_summary(df, columns):
  '''Build an empty summary skeleton for empty input.'''
  warning = 'empty dataframe: no rows to summarise'
  LOG.warning(warning)
  return {
    'shape': [0, len(columns)],
    'n_rows': 0,
    'n_cols': len(columns),
    'columns': list(columns),
    'dtypes': {c: str(df[c].dtype) if c in df.columns
               else 'unknown' for c in columns},
    'missing': {c: 0 for c in columns},
    'missing_rate': {c: 0.0 for c in columns},
    'numeric_describe': {},
    'categorical': {},
    'target_col': None,
    'target_distribution': {},
    'correlations': [],
    'outliers': {},
    'duplicate_rows': 0,
    'duplicate_rate': 0.0,
    'constant_cols': [],
    'leakage_findings': [],
    'warnings': [warning],
  }


def eda_summary(df, params=None, variables=None):
  '''Summarise a dataframe for exploratory data analysis.

  Args:
    df: Input dataframe (empty frames return a skeleton
      summary with a warning, never an exception).
    params: Optional dict; see module docstring for keys.
    variables: Optional dict with ``target_col`` and
      ``feature_cols``.

  Returns:
    Summary dict with ``shape`` ([rows, cols]), ``dtypes``,
    ``missing`` / ``missing_rate`` per column,
    ``numeric_describe``, ``categorical`` (cardinality +
    top values), ``target_distribution``, ``correlations``
    (top pairs), ``outliers`` (IQR + z-score counts),
    ``duplicate_rows``, ``constant_cols``, and ``warnings``.
  '''
  params = flatten_params(params, 'eda_params')
  variables = dict(variables) if variables else {}
  if df is None:
    df = pd.DataFrame()
  if not isinstance(df, pd.DataFrame):
    df = pd.DataFrame(df)
  columns = list(df.columns)
  if len(df) == 0:
    summary = _empty_summary(df, columns)
    target = _target_col(params, variables, df)
    summary['target_col'] = target
    return summary
  top_values_n = _int(params.get('top_values', 3), 3)
  top_corr_n = _int(params.get('top_correlations',
                               params.get('n_top_correlations', 10)), 10)
  z_threshold = _float(params.get('zscore_threshold', 3.0), 3.0)
  numeric_cols = [c for c in columns if _is_numeric(df, c)]
  cat_cols = [c for c in columns if c not in numeric_cols]
  missing = {c: int(df[c].isna().sum()) for c in columns}
  missing_rate = {c: float(df[c].isna().mean()) for c in columns}
  numeric_describe = {}
  outliers = {}
  for col in numeric_cols:
    series = df[col].dropna().astype(float)
    desc = series.describe()
    numeric_describe[col] = {
      key: (None if pd.isna(desc[key]) else float(desc[key]))
      for key in ('count', 'mean', 'std', 'min', '25%', '50%',
                  '75%', 'max') if key in desc}
    if len(series):
      q1, q3 = series.quantile(0.25), series.quantile(0.75)
      iqr = q3 - q1
      iqr_out = int(((series < q1 - 1.5 * iqr)
                     | (series > q3 + 1.5 * iqr)).sum())
      std = float(series.std())
      if std and not np.isnan(std):
        z_out = int((abs(series - series.mean()) / std
                     > z_threshold).sum())
      else:
        z_out = 0
    else:
      iqr_out, z_out = 0, 0
    outliers[col] = {'iqr': iqr_out, 'zscore': z_out}
  categorical = {}
  for col in cat_cols:
    counts = df[col].value_counts(dropna=False)
    categorical[col] = {
      'cardinality': int(df[col].nunique(dropna=True)),
      'top_values': [[_scalar(k), int(v)]
                     for k, v in counts.head(top_values_n).items()],
    }
  target = _target_col(params, variables, df)
  target_distribution = {}
  if target is not None:
    counts = df[target].value_counts(dropna=False)
    target_distribution = {_scalar(k): int(v)
                           for k, v in counts.items()}
  correlations = []
  if len(numeric_cols) >= 2:
    corr = df[numeric_cols].corr(numeric_only=True)
    for i, c1 in enumerate(numeric_cols):
      for c2 in numeric_cols[i + 1:]:
        value = corr.loc[c1, c2]
        if pd.isna(value):
          continue
        correlations.append({'pair': [c1, c2],
                             'correlation': float(value)})
    correlations.sort(key=lambda r: abs(r['correlation']),
                      reverse=True)
    correlations = correlations[:max(top_corr_n, 0)]
  duplicate_rows = int(df.duplicated().sum())
  constant_cols = [c for c in columns
                   if int(df[c].nunique(dropna=True)) <= 1]
  leakage_findings = (detect_leakage(df, target, params)
                      if target is not None else [])
  return {
    'shape': [int(len(df)), len(columns)],
    'n_rows': int(len(df)),
    'n_cols': len(columns),
    'columns': columns,
    'dtypes': {c: str(df[c].dtype) for c in columns},
    'missing': missing,
    'missing_rate': missing_rate,
    'numeric_describe': numeric_describe,
    'categorical': categorical,
    'target_col': target,
    'target_distribution': target_distribution,
    'correlations': correlations,
    'outliers': outliers,
    'duplicate_rows': duplicate_rows,
    'duplicate_rate': float(duplicate_rows / len(df)),
    'constant_cols': constant_cols,
    'leakage_findings': leakage_findings,
    'warnings': [],
  }


def _finding(check, severity, detail):
  '''Build one quality-check finding dict.'''
  return {'check': check, 'severity': severity, 'detail': detail}


def eda_quality_checks(summary, params=None):
  '''Turn an EDA summary into quality-check findings.

  Args:
    summary: Dict from :func:`eda_summary`.
    params: Optional dict; see module docstring for keys.

  Returns:
    List of ``{'check', 'severity', 'detail'}`` findings for
    high missingness, cardinality explosion, leakage-suspect
    feature-target correlation (> 0.95 by default), class
    imbalance, constant columns, and duplicate rows. Empty
    summaries yield no findings.
  '''
  params = flatten_params(params, 'eda_params')
  summary = summary or {}
  if not summary or summary.get('n_rows', 0) == 0:
    return []
  missing_threshold = _float(params.get('missing_threshold', 0.2), 0.2)
  cardinality_threshold = _int(params.get(
    'cardinality_threshold', params.get('max_cardinality', 50)), 50)
  leakage_threshold = _float(params.get('leakage_threshold', 0.95), 0.95)
  imbalance_threshold = _float(
    params.get('imbalance_ratio_threshold', 5.0), 5.0)
  findings = []
  for col, rate in (summary.get('missing_rate') or {}).items():
    if float(rate) > missing_threshold:
      findings.append(_finding(
        'high_missing', 'warning',
        'column %r missing rate %.2f > %.2f'
        % (col, float(rate), missing_threshold)))
  for col, info in (summary.get('categorical') or {}).items():
    cardinality = int(info.get('cardinality', 0))
    if cardinality > cardinality_threshold:
      findings.append(_finding(
        'cardinality_explosion', 'warning',
        'column %r cardinality %d > %d'
        % (col, cardinality, cardinality_threshold)))
  target = summary.get('target_col')
  if target:
    for entry in summary.get('correlations') or []:
      pair = entry.get('pair') or []
      corr = entry.get('correlation')
      if corr is None or target not in pair:
        continue
      if abs(float(corr)) > leakage_threshold:
        other = next(c for c in pair if c != target)
        findings.append(_finding(
          'leakage_suspect', 'high',
          'feature %r correlation %.3f with target %r > %.2f'
          % (other, float(corr), target, leakage_threshold)))
    distribution = summary.get('target_distribution') or {}
    counts = [v for v in distribution.values() if int(v) > 0]
    if len(counts) >= 2:
      ratio = max(counts) / min(counts)
      if ratio > imbalance_threshold:
        findings.append(_finding(
          'class_imbalance', 'warning',
          'target %r imbalance ratio %.1f > %.1f (%s)'
          % (target, ratio, imbalance_threshold, distribution)))
  constant_cols = summary.get('constant_cols') or []
  if constant_cols:
    findings.append(_finding(
      'constant_cols', 'warning',
      'constant columns: %s' % ', '.join(map(str, constant_cols))))
  duplicate_rows = int(summary.get('duplicate_rows') or 0)
  if duplicate_rows:
    findings.append(_finding(
      'duplicates', 'warning',
      '%d duplicate rows' % duplicate_rows))
  leakage = summary.get('leakage_findings') or []
  for entry in leakage:
    if isinstance(entry, dict) and entry.get('check') == 'leakage':
      findings.append(entry)
  return findings


def _leakage_finding(feature, severity, detail):
  '''Build one leakage finding dict.'''
  return {
    'check': 'leakage',
    'severity': severity,
    'feature': str(feature),
    'detail': detail,
    'recommendation': 'drop or investigate',
  }


def _resolve_target(df, target):
  '''Resolve target to (series, name) or (None, name).'''
  if isinstance(target, str):
    if target in df.columns:
      return df[target], target
    return None, target
  if isinstance(target, dict):
    col = target.get('target_col') or target.get('target')
    return _resolve_target(df, col)
  if isinstance(target, (pd.Series, np.ndarray, list)):
    return pd.Series(list(target)), 'target'
  return None, 'target'


def _auc_score(values, positive):
  '''Rank-based AUC for numeric values vs binary labels.'''
  frame = pd.DataFrame({'x': np.asarray(values, dtype=float),
                        'y': np.asarray(positive, dtype=int)})
  frame = frame.dropna()
  n_pos = int((frame['y'] == 1).sum())
  n_neg = int((frame['y'] == 0).sum())
  if n_pos < 2 or n_neg < 2:
    return None
  ranks = frame['x'].rank(method='average')
  rank_sum = float(ranks[frame['y'] == 1].sum())
  return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _entropy(counts):
  '''Shannon entropy of a count sequence.'''
  probs = np.asarray([c for c in counts if c > 0], dtype=float)
  if len(probs) == 0:
    return 0.0
  probs = probs / probs.sum()
  return float(-np.sum(probs * np.log(probs)))


def _normalized_mi(feature, target):
  '''Normalised mutual information between two series.'''
  frame = pd.DataFrame({'f': feature.astype(str).to_numpy(),
                        't': target.astype(str).to_numpy()})
  joint = frame.value_counts()
  f_counts = frame['f'].value_counts().to_numpy()
  t_counts = frame['t'].value_counts().to_numpy()
  total = len(frame)
  if total == 0:
    return 0.0
  h_f, h_t = _entropy(f_counts), _entropy(t_counts)
  denom = min(h_f, h_t)
  if denom <= 0:
    return 0.0
  mi = 0.0
  f_probs = frame['f'].value_counts(normalize=True)
  t_probs = frame['t'].value_counts(normalize=True)
  for (f_val, t_val), count in joint.items():
    p_joint = count / total
    mi += p_joint * np.log(
      p_joint / (f_probs[f_val] * t_probs[t_val]))
  return float(mi / denom)


def _series_equal(left, right):
  '''True when two series agree on every (non-null) row.'''
  if len(left) != len(right) or len(left) == 0:
    return False
  both = left.notna() & right.notna()
  if int(both.sum()) != len(left):
    return False
  try:
    return bool((left[both] == right[both]).all())
  except (TypeError, ValueError):
    return bool((left[both].astype(str)
                 == right[both].astype(str)).all())


def _purity_groups(feature, target):
  '''Return (n_groups, min_group_size, all_pure) for a feature.'''
  frame = pd.DataFrame({'f': feature.to_numpy(),
                        't': target.to_numpy()})
  groups = frame.groupby('f')['t']
  sizes = groups.size()
  if len(sizes) == 0:
    return 0, 0, False
  all_pure = bool((groups.nunique() == 1).all())
  return len(sizes), int(sizes.min()), all_pure


def detect_leakage(df, target, params=None):
  '''Detect leakage-suspect features in a dataframe.

  Args:
    df: Input dataframe (empty/None yields no findings).
    target: Target column name, Series/array of labels, or
      dict with ``target_col``.
    params: Optional dict; see module docstring for keys
      (``leakage_corr_threshold``, ``duplicate_corr_threshold``,
      ``leakage_auc_threshold``, ``near_constant_threshold``,
      ``leakage_name_keywords``; ``leakage_threshold`` is
      accepted as an alias for the correlation threshold).

  Returns:
    List of ``{'check': 'leakage', 'severity', 'feature',
    'detail', 'recommendation': 'drop or investigate'}``
    findings covering high feature-target correlation
    (Pearson / point-biserial for numeric, normalised mutual
    information for categorical), duplicate / near-duplicate
    columns (corr > 0.99), single-feature perfect
    separation (numeric AUC, categorical pure categories),
    post-outcome name keywords (warning), constant /
    near-constant features, and features identical or
    mapped 1-1 to the target.
  '''
  params = flatten_params(params, 'eda_params')
  if df is None or not isinstance(df, pd.DataFrame) or df.empty:
    if df is not None and len(df) == 0:
      LOG.warning('detect_leakage: empty dataframe; no findings')
    return []
  target_series, target_name = _resolve_target(df, target)
  if target_series is None or len(target_series) == 0:
    LOG.warning('detect_leakage: target %r not found; no findings'
                % (target_name,))
    return []
  target_series = target_series.reset_index(drop=True)
  corr_threshold = _float(params.get(
    'leakage_corr_threshold',
    params.get('leakage_threshold',
               params.get('corr_threshold', 0.95))), 0.95)
  duplicate_threshold = _float(
    params.get('duplicate_corr_threshold', 0.99), 0.99)
  auc_threshold = _float(params.get(
    'leakage_auc_threshold',
    params.get('separability_auc_threshold',
               params.get('separability_threshold', 0.99))), 0.99)
  near_constant_threshold = _float(
    params.get('near_constant_threshold', 0.95), 0.95)
  keywords = params.get('leakage_name_keywords',
                        params.get('name_keywords',
                                   params.get(
                                       'post_outcome_keywords',
                                       _DEFAULT_LEAKAGE_KEYWORDS)))
  keywords = [str(k).lower() for k in (keywords or [])]
  features = [c for c in df.columns if c != target_name]
  target_numeric = pd.to_numeric(target_series, errors='coerce')
  if target_numeric.notna().all():
    target_values = target_numeric.astype(float)
  else:
    target_values = pd.factorize(target_series)[0].astype(float)
  target_binary = target_series.nunique(dropna=True) == 2
  if target_binary:
    unique = sorted(target_series.dropna().unique(), key=str)
    positive = (target_series == unique[-1]).astype(int)
  else:
    positive = None
  findings = []
  numeric_features = []
  for col in features:
    series = df[col].reset_index(drop=True)
    clean = series.dropna()
    # Constant / near-constant.
    if len(clean) == 0:
      dominant = 1.0
    else:
      dominant = float(clean.value_counts(normalize=True).iloc[0])
    if int(series.nunique(dropna=True)) <= 1 or (
            dominant >= near_constant_threshold):
      findings.append(_leakage_finding(
        col, 'warning',
        'feature %r is constant/near-constant (dominant proportion'
        ' %.2f)' % (col, dominant)))
    # Post-outcome name keyword (warning).
    lowered = str(col).lower()
    for keyword in keywords:
      if keyword and keyword in lowered:
        findings.append(_leakage_finding(
          col, 'warning',
          'feature %r name contains leakage keyword %r'
          ' (post-outcome suspect)' % (col, keyword)))
        break
    # Identical / mapped 1-1 to target.
    if _series_equal(series, target_series):
      findings.append(_leakage_finding(
        col, 'high',
        'feature %r is identical to target %r' % (col, target_name)))
    elif int(series.nunique(dropna=True)) == int(
            target_series.nunique(dropna=True)) and (
            1 < int(series.nunique(dropna=True)) <= 20):
      n_groups, min_size, f_pure = _purity_groups(
        series, target_series)
      _, _, t_pure = _purity_groups(target_series, series)
      if f_pure and t_pure and min_size >= 2:
        findings.append(_leakage_finding(
          col, 'high',
          'feature %r is mapped 1-1 to target %r'
          % (col, target_name)))
    if _is_numeric(df, col):
      numeric_features.append(col)
      values = pd.to_numeric(series, errors='coerce')
      valid = values.notna() & target_values.notna()
      if int(valid.sum()) >= 3 and float(
              values[valid].std() or 0.0) > 0:
        corr = float(values[valid].corr(target_values[valid]))
        if abs(corr) > corr_threshold:
          findings.append(_leakage_finding(
            col, 'high',
            'feature %r correlation %.3f with target %r > %.2f'
            ' (point-biserial/Pearson)' % (col, corr, target_name,
                                            corr_threshold)))
      if positive is not None:
        auc = _auc_score(values[valid], positive[valid])
        if auc is not None and (auc >= auc_threshold
                                or auc <= 1.0 - auc_threshold):
          findings.append(_leakage_finding(
            col, 'high',
            'feature %r perfectly separates target %r (AUC=%.3f)'
            % (col, target_name, auc)))
    else:
      # Categorical: pure categories + normalised MI.
      n_groups, min_size, all_pure = _purity_groups(
        series, target_series)
      if all_pure and n_groups >= 2 and min_size >= 2:
        findings.append(_leakage_finding(
          col, 'high',
          'feature %r categories are pure for target %r'
          ' (perfect separation)' % (col, target_name)))
      cardinality = int(series.nunique(dropna=True))
      if (2 <= cardinality <= 50 and cardinality <= len(series) / 2
              and min_size >= 2):
        nmi = _normalized_mi(series, target_series)
        if nmi > corr_threshold:
          findings.append(_leakage_finding(
            col, 'high',
            'feature %r normalised mutual information %.3f with'
            ' target %r > %.2f' % (col, nmi, target_name,
                                    corr_threshold)))
  # Duplicate / near-duplicate feature columns.
  for i, c1 in enumerate(features):
    for c2 in features[i + 1:]:
      s1 = df[c1].reset_index(drop=True)
      s2 = df[c2].reset_index(drop=True)
      if _series_equal(s1, s2):
        findings.append(_leakage_finding(
          c2, 'high',
          'feature %r duplicates feature %r (identical values)'
          % (c2, c1)))
        continue
      if c1 in numeric_features and c2 in numeric_features:
        pair = pd.DataFrame({
          'a': pd.to_numeric(s1, errors='coerce'),
          'b': pd.to_numeric(s2, errors='coerce')}).dropna()
        if len(pair) >= 3 and float(pair['a'].std() or 0) > 0 and (
                float(pair['b'].std() or 0) > 0):
          corr = float(pair['a'].corr(pair['b']))
          if abs(corr) > duplicate_threshold:
            findings.append(_leakage_finding(
              c2, 'high',
              'feature %r is a near-duplicate of feature %r'
              ' (correlation %.3f > %.2f)'
              % (c2, c1, corr, duplicate_threshold)))
  return findings


def eda_report(summary, params=None, df=None, variables=None):
  '''Build the EDA report from a summary.

  Args:
    summary: Dict from :func:`eda_summary`.
    params: Optional dict; see module docstring for keys.
    df: Optional raw dataframe; when given, leakage
      findings are recomputed via :func:`detect_leakage`
      and merged into the report findings, and
      ``make_data_charts(df, target, params)`` charts are
      merged into the report as ``data_charts``.
    variables: Optional dict with ``target_col`` used
      when ``df`` is given.

  Returns:
    Dict with ``summary``, ``findings`` (from
    :func:`eda_quality_checks` plus leakage findings),
    ``recommendations`` (list of strings), ``warnings``,
    and ``data_charts`` (chart name to Figure from
    ``make_data_charts`` when ``df`` is given, else {}).
    Empty summaries return empty findings/recommendations
    plus the summary warning.
  '''
  params = flatten_params(params, 'eda_params')
  summary = summary or {}
  findings = eda_quality_checks(summary, params)
  data_charts = {}
  if df is not None:
    frame = df if isinstance(df, pd.DataFrame) else pd.DataFrame()
    target = _target_col(params, variables or {}, frame)
    if target is not None and not frame.empty:
      seen = {(f.get('feature'), f.get('detail')) for f in findings}
      for entry in detect_leakage(df, target, params):
        key = (entry.get('feature'), entry.get('detail'))
        if key not in seen:
          findings.append(entry)
          seen.add(key)
      from forecast_ml_framework.pipelines.modeling.charts import (
        make_data_charts)
      data_charts = make_data_charts(df, target, params) or {}
  recommendations = []
  for finding in findings:
    if finding['check'] == 'leakage':
      recommendations.append(
        finding.get('recommendation') or 'drop or investigate')
    elif finding['check'] in _RECOMMENDATIONS:
      recommendations.append(_RECOMMENDATIONS[finding['check']])
  warnings = list(summary.get('warnings') or [])
  if warnings:
    for warning in warnings:
      LOG.warning(warning)
  return {
    'summary': summary,
    'findings': findings,
    'recommendations': recommendations,
    'warnings': warnings,
    'data_charts': data_charts,
  }
