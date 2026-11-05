#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Data splitting and sampling nodes.'''

import numpy as np


def split_data(df, params):
  '''Split a frame into train/test features and target.

  Args:
    df: Input DataFrame with a target column.
    params: modeling_params dict. Keys used: target_col,
      feature_cols, task_type, test_size, random_state and
      sampling_params {method: stratified|random,
      test_size, stratify_col}.

  Returns:
    Tuple (x_train, x_test, y_train, y_test, meta).
  '''
  from sklearn.model_selection import train_test_split
  params = params or {}
  sampling = params.get('sampling_params', {}) or {}
  target = params.get('target_col', 'target')
  features = params.get('feature_cols')
  if features is None:
    features = [c for c in df.columns if c != target]
  test_size = sampling.get('test_size',
                           params.get('test_size', 0.2))
  seed = params.get('random_state', 42)
  method = sampling.get('method', 'stratified')
  stratify = None
  if method == 'stratified':
    stratify = df[sampling.get('stratify_col', target)] if (
        params.get('task_type') == 'classification'
        or 'stratify_col' in sampling) else None
    if stratify is not None and stratify.nunique() < 2:
      stratify = None
  x = df[features]
  y = df[target]
  try:
    x_tr, x_te, y_tr, y_te = train_test_split(
        x, y, test_size=test_size, random_state=seed,
        stratify=stratify)
  except ValueError:
    x_tr, x_te, y_tr, y_te = train_test_split(
        x, y, test_size=test_size, random_state=seed)
  meta = {'task_type': params.get('task_type', 'classification'),
          'test_size': test_size, 'random_state': seed,
          'sampling_method': method,
          'n_train': int(len(y_tr)), 'n_test': int(len(y_te))}
  return x_tr, x_te, np.asarray(y_tr), np.asarray(y_te), meta
