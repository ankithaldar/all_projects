#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Split-first (A9/ML-2), OOF (ML-3), ddof (ML-9), id (ML-24c) tests.

RED first: each test pins the post-fix behaviour and must fail
on the pre-fix wiring for the stated reason (leak, ddof=1,
in-sample target encoding, id-as-feature, modeling re-split).
'''

import numpy as np
import pandas as pd
import pytest


def _norm_params():
  '''Return Standard scaler step for f1.'''
  return [{
    'Normalizations': {
      'args': {
        'handle': 'StandardScaler',
        'cols': {'include_cols': ['f1']}}},
  }]


def test_fitted_train_only_mu_sd_not_full():
  '''Fit on train [1,2,3] only: mu 2.0, ddof0 sd 0.816..., test 100 -> 120.'''
  from forecast_ml_framework.pipelines.data_prep.fitted import (
    apply_preprocess,
    fit_preprocess,
  )
  train = pd.DataFrame({'f1': [1.0, 2.0, 3.0], 'target': [0, 1, 0]})
  _, registry = fit_preprocess(
    train, _norm_params(),
    {'num_cols': ['f1'], 'cat_cols': []})
  state = registry['Normalizations']['f1']
  mu, sd = state['mu'], state['sd']
  assert mu == pytest.approx(2.0)
  # Population (ddof=0) sd of [1,2,3] is sqrt(2/3) ~ 0.8165.
  assert sd == pytest.approx(0.816496580927726, rel=1e-12)
  out = apply_preprocess(
    pd.DataFrame({'f1': [100.0], 'target': [1]}), registry)
  assert out['f1'].iloc[0] == pytest.approx(
    (100.0 - 2.0) / 0.816496580927726, rel=1e-12)
  # Leaky full-frame value would be ~1.50 (mu 26.5, sd 49.0).
  assert abs(out['f1'].iloc[0] - 1.50) > 10.0


def test_stateless_normalize_ddof0_matches_sklearn():
  '''Stateless _normalize equals StandardScaler to 1e-12 on [1,2,3,4].'''
  from sklearn.preprocessing import StandardScaler
  from forecast_ml_framework.pipelines.data_prep.nodes import (
    _normalize)
  frame = pd.DataFrame({'f1': [1.0, 2.0, 3.0, 4.0]})
  out = _normalize(
    frame.copy(),
    {'handle': 'StandardScaler',
     'cols': {'include_cols': ['f1']}},
    {'num_cols': ['f1']})
  expected = StandardScaler().fit_transform(
    np.asarray([1.0, 2.0, 3.0, 4.0]).reshape(-1, 1)).ravel()
  assert np.allclose(out['f1'].to_numpy(), expected, atol=1e-12)


def test_fitted_normalize_ddof0_matches_sklearn():
  '''Fitted normalize equals StandardScaler to 1e-12 on [1,2,3,4].'''
  from sklearn.preprocessing import StandardScaler
  from forecast_ml_framework.pipelines.data_prep.fitted import (
    apply_preprocess,
    fit_preprocess,
  )
  train = pd.DataFrame({'f1': [1.0, 2.0, 3.0, 4.0],
                        'target': [0, 1, 0, 1]})
  _, registry = fit_preprocess(
    train, _norm_params(), {'num_cols': ['f1']})
  out = apply_preprocess(train, registry)
  expected = StandardScaler().fit_transform(
    np.asarray([1.0, 2.0, 3.0, 4.0]).reshape(-1, 1)).ravel()
  assert np.allclose(out['f1'].to_numpy(), expected, atol=1e-12)


def test_target_encode_oof_single_row_not_own_label():
  '''Fitted fit output is OOF: single-row 'b' != its own label 0.'''
  from forecast_ml_framework.pipelines.data_prep.fitted import (
    fit_preprocess,
  )
  train = pd.DataFrame({
    'cat': ['a', 'a', 'b'], 'target': [1.0, 0.0, 0.0]})
  fitted, _registry = fit_preprocess(
    train,
    [{'TargetEncoding': {'args': {
      'cols': {'include_cols': ['cat']}}}}],
    {'cat_cols': ['cat'], 'target_col': 'target'})
  # 'b' occurs once with label 0.0; in-sample mean would be 0.0.
  # OOF must not contain its own label (fold mapping has no 'b',
  # so it falls back to the other rows' global mean 0.5).
  assert fitted['cat'].iloc[2] != pytest.approx(0.0)
  assert fitted['cat'].iloc[2] == pytest.approx(0.5)


def test_target_encode_stateless_oof_unseen_global():
  '''Stateless _target_encode: OOF, unseen -> global mean not NaN.'''
  from forecast_ml_framework.pipelines.data_prep.nodes import (
    _target_encode)
  train = pd.DataFrame({
    'cat': ['a', 'a', 'b'], 'target': [1.0, 0.0, 0.0]})
  out = _target_encode(
    train.copy(),
    {'cols': {'include_cols': ['cat']}},
    {'cat_cols': ['cat'], 'target_col': 'target'})
  assert out['cat'].iloc[2] != pytest.approx(0.0)
  # Smoothed full mapping check via fitted apply': unseen -> global.
  from forecast_ml_framework.pipelines.data_prep.fitted import (
    apply_preprocess,
    fit_preprocess,
  )
  train2 = pd.DataFrame({
    'cat': ['a', 'a', 'b', 'b'], 'target': [0.0, 2.0, 4.0, 6.0]})
  _, registry = fit_preprocess(
    train2,
    [{'TargetEncoding': {'args': {
      'cols': {'include_cols': ['cat']},
      'smoothing': 0.0}}}],
    {'cat_cols': ['cat'], 'target_col': 'target'})
  unseen = apply_preprocess(
    pd.DataFrame({'cat': ['zzz'], 'target': [0.0]}), registry)
  assert unseen['cat'].iloc[0] == pytest.approx(3.0)
  assert not np.isnan(unseen['cat'].iloc[0])


def test_target_encode_smoothing_formula():
  '''Smoothing (n*mean+m*global)/(n+m): n=2,mean=1,m=2,g=3 -> 2.0.'''
  from forecast_ml_framework.pipelines.data_prep.fitted import (
    apply_preprocess,
    fit_preprocess,
  )
  train = pd.DataFrame({
    'cat': ['x', 'x', 'y', 'y'],
    'target': [0.0, 2.0, 4.0, 6.0]})
  _, registry = fit_preprocess(
    train,
    [{'TargetEncoding': {'args': {
      'cols': {'include_cols': ['cat']}, 'smoothing': 2.0}}}],
    {'cat_cols': ['cat'], 'target_col': 'target'})
  out = apply_preprocess(
    pd.DataFrame({'cat': ['x', 'y'], 'target': [0.0, 0.0]}), registry)
  # x: (2*1 + 2*3)/4 = 2.0; y: (2*5 + 2*3)/4 = 4.0
  assert out['cat'].iloc[0] == pytest.approx(2.0)
  assert out['cat'].iloc[1] == pytest.approx(4.0)


def test_split_raw_excludes_id_cols():
  '''split_raw_data train/test frames; id retained but split meta
  feature handling excludes id (via params feature_cols).'''
  from forecast_ml_framework.pipelines.data_prep.nodes import (
    split_raw_data)
  df = pd.DataFrame({
    'id': [0, 1, 2, 3, 4, 5],
    'f1': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    'target': [0, 1, 0, 1, 0, 1]})
  variables = {
    'target_col': 'target', 'feature_cols': ['f1'],
    'id_cols': ['id'], 'num_cols': ['f1'], 'cat_cols': []}
  params = {
    'target_col': 'target', 'task_type': 'classification',
    'test_size': 0.5, 'random_state': 42,
    'sampling_params': {'method': 'random', 'test_size': 0.5}}
  train_raw, test_raw = split_raw_data(df, params, variables)
  assert len(train_raw) + len(test_raw) == 6
  assert set(train_raw['id']).isdisjoint(set(test_raw['id']))
  # Time split: train indices all precede test indices.
  train2, test2 = split_raw_data(
    df,
    dict(params, sampling_params={
      'method': 'time', 'test_size': 0.5}),
    variables)
  assert train2['id'].max() < test2['id'].min()


def test_apply_selection_subsets_test():
  '''apply_feature_selection keeps only selected features + target.'''
  from forecast_ml_framework.pipelines.data_prep.nodes import (
    apply_feature_selection)
  test = pd.DataFrame({
    'f1': [1.0], 'f2': [2.0], 'f3': [3.0],
    'id': [0], 'target': [1]})
  out = apply_feature_selection(
    test, ['f1', 'f3'],
    {'target_col': 'target', 'feature_cols': ['f1', 'f2', 'f3']})
  assert set(out.columns) >= {'f1', 'f3', 'target'}
  assert 'f2' not in out.columns


def test_data_prep_pipeline_split_first_wiring():
  '''Pipeline: validate->split->fit->apply->select->apply-selection.'''
  from forecast_ml_framework.pipelines.data_prep.pipeline import (
    create_pipeline)
  pipe = create_pipeline()
  by_name = {n.name: n for n in pipe.nodes}
  assert 'split_raw' in by_name
  split = by_name['split_raw']
  assert 'validated_data' in split.inputs
  assert 'train_raw' in split.outputs
  assert 'test_raw' in split.outputs
  # Modeling params carry the split config (single source).
  assert 'params:modeling_params' in split.inputs
  fit = by_name['fit_preprocess']
  assert 'train_raw' in fit.inputs
  # Apply + selection wiring for test side.
  assert 'apply_preprocess' in by_name
  assert 'apply_selection' in by_name
  apply_node = by_name['apply_preprocess']
  assert 'test_raw' in apply_node.inputs
  assert 'fitted_registry' in apply_node.inputs
  apply_sel = by_name['apply_selection']
  assert 'test_data_selected' in apply_sel.outputs
  assert 'selected_features' in apply_sel.inputs


def test_modeling_pipeline_presplit_wiring():
  '''Modeling split node consumes pre-split frames, meta=presplit.'''
  from forecast_ml_framework.pipelines.modeling.pipeline import (
    create_pipeline)
  pipe = create_pipeline()
  by_name = {n.name: n for n in pipe.nodes}
  split = by_name['split_data_node']
  assert 'train_data_selected' in split.inputs
  assert 'test_data_selected' in split.inputs
  assert 'modeling_input' not in split.inputs
  x_tr, x_te, y_tr, y_te, meta = split.func(
    pd.DataFrame({'f1': [1.0, 2.0], 'target': [0, 1]}),
    pd.DataFrame({'f1': [3.0], 'target': [1]}),
    {'target_col': 'target', 'task_type': 'classification'})
  assert meta['sampling_method'] == 'presplit'
  assert list(x_tr['f1']) == [1.0, 2.0]
  assert list(x_te['f1']) == [3.0]
  # id column is excluded from features by default.
  x2, _xt, _y1, _y2, _m = split.func(
    pd.DataFrame({'id': [0, 1], 'f1': [1.0, 2.0], 'target': [0, 1]}),
    pd.DataFrame({'id': [2], 'f1': [3.0], 'target': [1]}),
    {'target_col': 'target', 'task_type': 'classification'})
  assert 'id' not in x2.columns
