#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests for fitted preprocessing fit/apply and registry validation.'''

def test_fit_apply_replays_train_bounds_on_shifted_data():
  '''Fit/apply replays train bounds on shifted scoring data.'''
  import pandas as pd
  from forecast_ml_framework.pipelines.data_prep.fitted import (
    fit_preprocess,
    apply_preprocess,
  )
  train = pd.DataFrame({'f1': [0.0, 10.0], 'target': [0, 1]})
  params = [{'Normalizations': {'args': {
    'handle': 'MinMaxScaler', 'cols': {'include_cols': ['f1']}}}}]
  fitted, registry = fit_preprocess(
    train, params, {'num_cols': ['f1'], 'cat_cols': [], 'seq_cols': []})
  assert list(fitted['f1']) == [0.0, 1.0]
  score = pd.DataFrame({'f1': [10.0, 20.0]})
  out = apply_preprocess(score, registry)
  assert abs(out['f1'].iloc[0] - 1.0) < 1e-9
  assert abs(out['f1'].iloc[1] - 2.0) < 1e-9


def test_imputer_replays_train_median_not_score_median():
  '''Imputer replays train median on scoring data, not score median.'''
  import pandas as pd, numpy as np
  from forecast_ml_framework.pipelines.data_prep.fitted import (
    fit_preprocess,
    apply_preprocess,
  )
  train = pd.DataFrame({'f1': [1.0, 3.0, 5.0], 'target': [0, 1, 0]})
  params = [{'Imputations': {'args': {
    'strategy': 'median', 'cols': {'include_cols': ['f1']}}}}]
  _, registry = fit_preprocess(train, params, {'num_cols': ['f1']})
  out = apply_preprocess(pd.DataFrame({'f1': [np.nan, 100.0]}), registry)
  assert out['f1'].iloc[0] == 3.0


def test_registry_json_roundtrip():
  '''Registry JSON roundtrip preserves apply behaviour.'''
  import pandas as pd
  from forecast_ml_framework.pipelines.data_prep.fitted import (
    fit_preprocess,
    apply_preprocess,
    registry_to_json,
    registry_from_json,
  )
  train = pd.DataFrame({'f1': [0.0, 10.0], 'target': [0, 1]})
  params = [{'Normalizations': {'args': {
    'handle': 'MinMaxScaler', 'cols': {'include_cols': ['f1']}}}}]
  _, registry = fit_preprocess(train, params, {'num_cols': ['f1']})
  rt = registry_from_json(registry_to_json(registry))
  out = apply_preprocess(pd.DataFrame({'f1': [5.0]}), rt)
  assert abs(out['f1'].iloc[0] - 0.5) < 1e-9


def test_fit_cap_skewed_bounds_raise():
  '''Fit cap raises on skewed custom bounds.'''
  import pandas as pd
  import pytest
  from forecast_ml_framework.pipelines.data_prep.fitted import (
    fit_preprocess,
  )
  train = pd.DataFrame({'f1': [0.0, 10.0], 'target': [0, 1]})
  params = [{'CapOutliers': {'args': {
    'custom_bound': {'f1': [5.0, 1.0]},
    'cols': {'include_cols': ['f1']}}}}]
  with pytest.raises(ValueError, match='f1'):
    fit_preprocess(train, params, {'num_cols': ['f1']})


def test_fit_cap_quantile_out_of_range_raise():
  '''Fit cap raises on out-of-range quantile bounds.'''
  import pandas as pd
  import pytest
  from forecast_ml_framework.pipelines.data_prep.fitted import (
    fit_preprocess,
  )
  train = pd.DataFrame({'f1': [0.0, 10.0], 'target': [0, 1]})
  params = [{'CapOutliers': {'args': {
    'bound': {'lower': 0.9, 'upper': 0.1},
    'cols': {'include_cols': ['f1']}}}}]
  with pytest.raises(ValueError):
    fit_preprocess(train, params, {'num_cols': ['f1']})


def test_fit_cap_all_nan_raise():
  '''Fit cap raises on all-NaN column.'''
  import numpy as np
  import pandas as pd
  import pytest
  from forecast_ml_framework.pipelines.data_prep.fitted import (
    fit_preprocess,
  )
  train = pd.DataFrame({'f1': [np.nan, np.nan], 'target': [0, 1]})
  params = [{'CapOutliers': {'args': {
    'cols': {'include_cols': ['f1']}}}}]
  with pytest.raises(ValueError, match='f1'):
    fit_preprocess(train, params, {'num_cols': ['f1']})


def test_fit_cap_bad_pair_type_raise():
  '''Fit cap raises on bad pair type.'''
  import pandas as pd
  import pytest
  from forecast_ml_framework.pipelines.data_prep.fitted import (
    fit_preprocess,
  )
  train = pd.DataFrame({'f1': [0.0, 10.0], 'target': [0, 1]})
  params = [{'CapOutliers': {'args': {
    'custom_bound': {'f1': 'bad'},
    'cols': {'include_cols': ['f1']}}}}]
  with pytest.raises(ValueError, match='f1'):
    fit_preprocess(train, params, {'num_cols': ['f1']})


def test_fit_discretize_num_buckets_unbounded_raise():
  '''Fit discretize raises on unbounded numBuckets.'''
  import pandas as pd
  import pytest
  from forecast_ml_framework.pipelines.data_prep.fitted import (
    fit_preprocess,
  )
  train = pd.DataFrame({'f1': [0.0, 10.0], 'target': [0, 1]})
  params = [{'QuantileDiscretizer': {'args': {
    'params': {'numBuckets': 100000},
    'cols': {'include_cols': ['f1']}}}}]
  with pytest.raises(ValueError, match='numBuckets'):
    fit_preprocess(train, params, {'num_cols': ['f1']})


def test_fit_duplicate_step_names_raise():
  '''Fit raises on duplicate step names.'''
  import pandas as pd
  import pytest
  from forecast_ml_framework.pipelines.data_prep.fitted import (
    fit_preprocess,
  )
  train = pd.DataFrame({'f1': [0.0, 10.0], 'target': [0, 1]})
  step = {'Normalizations': {'args': {
    'handle': 'MinMaxScaler', 'cols': {'include_cols': ['f1']}}}}
  with pytest.raises(ValueError, match='duplicate|Normalizations'):
    fit_preprocess(train, [step, dict(step)], {'num_cols': ['f1']})


def test_fit_non_fitted_step_raises():
  '''Fit raises on non-fitted step.'''
  import pandas as pd
  import pytest
  from forecast_ml_framework.pipelines.data_prep.fitted import (
    fit_preprocess,
  )
  train = pd.DataFrame({'f1': [0.0, 10.0], 'target': [0, 1]})
  params = [{'KnnImputation': {'args': {
    'cols': {'include_cols': ['f1']}}}}]
  with pytest.raises(ValueError, match='KnnImputation'):
    fit_preprocess(train, params, {'num_cols': ['f1']})


def test_registry_has_version_and_schema():
  '''Registry carries version and schema (v2 nested steps).'''
  import pandas as pd
  from forecast_ml_framework.pipelines.data_prep.fitted import (
    fit_preprocess,
    read_registry,
  )
  train = pd.DataFrame({'f1': [0.0, 10.0], 'target': [0, 1]})
  params = [{'Normalizations': {'args': {
    'handle': 'MinMaxScaler', 'cols': {'include_cols': ['f1']}}}}]
  _, registry = fit_preprocess(train, params, {'num_cols': ['f1']})
  assert registry.get('version') == 2
  assert registry.get('schema') == 'fitted_registry.v2'
  # v2 writer nests step state under 'steps' (no flat step keys).
  assert 'steps' in registry
  assert 'Normalizations' not in registry
  steps = read_registry(registry)
  assert set(steps) == {'Normalizations'}
  assert steps['Normalizations']['f1']['lo'] == 0.0
  assert steps['Normalizations']['f1']['hi'] == 10.0


def test_apply_ohe_filters_unseen_categories():
  '''Apply OHE filters unseen categories.'''
  import pandas as pd
  from forecast_ml_framework.pipelines.data_prep.fitted import (
    apply_preprocess,
    fit_preprocess,
  )
  train = pd.DataFrame({'cat': ['a', 'b', 'a', 'b'],
                        'target': [0, 1, 0, 1]})
  params = [{'CategoricalEncoding': {'args': {
    'ohe_flag': True, 'cols': {'include_cols': ['cat']}}}}]
  _, registry = fit_preprocess(
    train, params, {'cat_cols': ['cat']})
  score = pd.DataFrame({'cat': ['a', 'unseen1', 'unseen2', 'b']})
  out = apply_preprocess(score, registry)
  assert 'cat_unseen1' not in out.columns
  assert 'cat_unseen2' not in out.columns
  assert set(out.columns) == {'cat_a', 'cat_b'}


def test_apply_ohe_respects_max_cols_cap():
  '''Apply OHE respects max_cols cap.'''
  import pandas as pd
  import pytest
  from forecast_ml_framework.pipelines.data_prep.fitted import (
    apply_preprocess,
  )
  score = pd.DataFrame({'cat': ['a', 'b'] * 600})
  registry = {'version': 2, 'schema': 'fitted_registry.v2',
              'steps': {'CategoricalEncoding': {'cat': {
                'categories': ['c%d' % i for i in range(2000)],
                'ohe': True, 'max_cols': 1000}}}}
  with pytest.raises(ValueError, match='max_cols'):
    apply_preprocess(score, registry)


def test_fit_ohe_exceeds_max_cols_raise():
  '''Fit OHE raises when exceeding max_cols.'''
  import pandas as pd
  import pytest
  from forecast_ml_framework.pipelines.data_prep.fitted import (
    fit_preprocess,
  )
  train = pd.DataFrame({'cat': ['c%d' % i for i in range(1500)],
                        'target': [0, 1] * 750})
  params = [{'CategoricalEncoding': {'args': {
    'ohe_flag': True, 'max_cols': 1000,
    'cols': {'include_cols': ['cat']}}}}]
  with pytest.raises(ValueError, match='max_cols'):
    fit_preprocess(train, params, {'cat_cols': ['cat']})


def test_data_prep_pipeline_fit_node_outputs_registry():
  '''Data-prep pipeline fit node outputs registry.'''
  from forecast_ml_framework.pipelines.data_prep.pipeline import (
    create_pipeline,
  )
  pipe = create_pipeline()
  nodes = {n.name: n for n in pipe.nodes}
  fit_nodes = [n for n in pipe.nodes
               if 'fitted_registry' in n.outputs]
  assert fit_nodes
  assert 'preprocessed_data' in fit_nodes[0].outputs
  func = fit_nodes[0].func
  assert getattr(func, '__name__', '') == 'fit_preprocess'


def test_nodes_fit_preprocess_returns_registry_and_frame():
  '''Nodes fit_preprocess returns registry and frame.'''
  import pandas as pd
  from forecast_ml_framework.pipelines.data_prep.nodes import (
    fit_preprocess,
  )
  train = pd.DataFrame({'f1': [0.0, 10.0], 'target': [0, 1]})
  params = [{'Normalizations': {'args': {
    'handle': 'MinMaxScaler', 'cols': {'include_cols': ['f1']}}}}]
  out, registry = fit_preprocess(
    train, params, {'num_cols': ['f1']})
  assert list(out['f1']) == [0.0, 1.0]
  assert registry['version'] == 2
  assert registry['schema'] == 'fitted_registry.v2'
  assert 'steps' in registry
  assert 'Normalizations' not in registry
  from forecast_ml_framework.pipelines.data_prep.fitted import (
    read_registry,
  )
  assert set(read_registry(registry)) == {'Normalizations'}


def test_qcut_small_data_does_not_crash():
  '''Qcut on small data does not crash.'''
  import pandas as pd
  from forecast_ml_framework.pipelines.data_prep.fitted import (
    fit_preprocess,
    apply_preprocess,
  )
  train = pd.DataFrame({'f1': [1.0, 1.0], 'target': [0, 1]})
  params = [{'QuantileDiscretizer': {'args': {
    'params': {'numBuckets': 10},
    'cols': {'include_cols': ['f1']}}}}]
  fitted, registry = fit_preprocess(train, params, {'num_cols': ['f1']})
  out = apply_preprocess(train, registry)
  assert len(out) == 2


# --- high-fix: fitted finite-state / inf / all-NaN / version (RED) ---

def test_apply_normalize_nonfinite_mu_raises():
  '''Apply normalize raises on nonfinite mu.'''
  import numpy as np
  import pandas as pd
  import pytest
  from forecast_ml_framework.pipelines.data_prep.fitted import (
    apply_preprocess,
  )
  for mu in (float('nan'), float('inf'), 'bad', None):
    reg = {'version': 2, 'schema': 'fitted_registry.v2',
           'steps': {'Normalizations': {
             'f1': {'kind': 'standard', 'mu': mu, 'sd': 1.0}}}}
    with pytest.raises(ValueError, match='f1'):
      apply_preprocess(pd.DataFrame({'f1': [1.0]}), reg)


def test_apply_normalize_nonfinite_sd_and_lohi_raises():
  '''Apply normalize raises on nonfinite sd/lo/hi.'''
  import pandas as pd
  import pytest
  from forecast_ml_framework.pipelines.data_prep.fitted import (
    apply_preprocess,
  )
  reg = {'version': 2, 'schema': 'fitted_registry.v2',
         'steps': {'Normalizations': {
           'f1': {'kind': 'standard', 'mu': 0.0, 'sd': float('inf')}}}}
  with pytest.raises(ValueError, match='f1'):
    apply_preprocess(pd.DataFrame({'f1': [1.0]}), reg)
  reg = {'version': 2, 'schema': 'fitted_registry.v2',
         'steps': {'Normalizations': {
           'f1': {'kind': 'minmax', 'lo': float('nan'), 'hi': 1.0}}}}
  with pytest.raises(ValueError, match='f1'):
    apply_preprocess(pd.DataFrame({'f1': [1.0]}), reg)


def test_apply_discretize_unsorted_or_nan_edges_raises():
  '''Apply discretize raises on unsorted or NaN edges.'''
  import pandas as pd
  import pytest
  from forecast_ml_framework.pipelines.data_prep.fitted import (
    apply_preprocess,
  )
  for edges in ([0.0, 2.0, 1.0], [0.0, float('nan'), 1.0],
                [0.0, float('inf')], [], [1.0], ['a', 'b']):
    reg = {'version': 2, 'schema': 'fitted_registry.v2',
           'steps': {'QuantileDiscretizer': {
             'f1': {'edges': edges}}}}
    with pytest.raises(ValueError, match='f1'):
      apply_preprocess(pd.DataFrame({'f1': [0.5]}), reg)


def test_apply_impute_nonfinite_fill_raises():
  '''Apply impute raises on nonfinite fill.'''
  import pandas as pd
  import pytest
  from forecast_ml_framework.pipelines.data_prep.fitted import (
    apply_preprocess,
  )
  for fill in (float('nan'), float('inf'), None):
    reg = {'version': 2, 'schema': 'fitted_registry.v2',
           'steps': {'Imputations': {
             'f1': {'fill': fill, 'strategy': 'mean'}}}}
    with pytest.raises(ValueError, match='f1'):
      apply_preprocess(pd.DataFrame({'f1': [float('nan')]}), reg)


def test_apply_target_encode_nonfinite_global_raises():
  '''Apply target encode raises on nonfinite global.'''
  import pandas as pd
  import pytest
  from forecast_ml_framework.pipelines.data_prep.fitted import (
    apply_preprocess,
  )
  for glob in (float('nan'), float('inf')):
    reg = {'version': 2, 'schema': 'fitted_registry.v2',
           'steps': {'TargetEncoding': {'cat': {
             'mapping': [['a', 0.5]], 'global': glob}}}}
    with pytest.raises(ValueError, match='cat'):
      apply_preprocess(pd.DataFrame({'cat': ['a']}), reg)


def test_apply_registry_version_mismatch_raises():
  '''Apply raises on registry version mismatch.'''
  import pandas as pd
  import pytest
  from forecast_ml_framework.pipelines.data_prep.fitted import (
    apply_preprocess,
  )
  reg = {'version': 999, 'schema': 'fitted_registry.v1',
         'steps': {'Normalizations': {
           'f1': {'kind': 'minmax', 'lo': 0.0, 'hi': 1.0}}}}
  with pytest.raises(ValueError, match='version'):
    apply_preprocess(pd.DataFrame({'f1': [0.5]}), reg)


def test_fit_normalize_all_nan_and_inf_raises():
  '''Fit normalize raises on all-NaN and inf data.'''
  import numpy as np
  import pandas as pd
  import pytest
  from forecast_ml_framework.pipelines.data_prep.fitted import (
    fit_preprocess,
  )
  params = [{'Normalizations': {'args': {
    'handle': 'MinMaxScaler', 'cols': {'include_cols': ['f1']}}}}]
  with pytest.raises(ValueError, match='f1'):
    fit_preprocess(pd.DataFrame({'f1': [np.nan, np.nan]}), params,
                   {'num_cols': ['f1']})
  with pytest.raises(ValueError, match='f1'):
    fit_preprocess(pd.DataFrame({'f1': [0.0, np.inf]}), params,
                   {'num_cols': ['f1']})


def test_fit_discretize_all_nan_and_inf_raises():
  '''Fit discretize raises on all-NaN and inf data.'''
  import numpy as np
  import pandas as pd
  import pytest
  from forecast_ml_framework.pipelines.data_prep.fitted import (
    fit_preprocess,
  )
  params = [{'QuantileDiscretizer': {'args': {
    'params': {'numBuckets': 2},
    'cols': {'include_cols': ['f1']}}}}]
  with pytest.raises(ValueError, match='f1'):
    fit_preprocess(pd.DataFrame({'f1': [np.nan, np.nan]}), params,
                   {'num_cols': ['f1']})
  with pytest.raises(ValueError, match='f1'):
    fit_preprocess(pd.DataFrame({'f1': [0.0, np.inf, 1.0]}), params,
                   {'num_cols': ['f1']})


def test_fit_impute_all_nan_and_inf_raises():
  '''Fit impute raises on all-NaN and inf data.'''
  import numpy as np
  import pandas as pd
  import pytest
  from forecast_ml_framework.pipelines.data_prep.fitted import (
    fit_preprocess,
  )
  for strategy in ('mean', 'median', 'mode'):
    params = [{'Imputations': {'args': {
      'strategy': strategy, 'cols': {'include_cols': ['f1']}}}}]
    with pytest.raises(ValueError, match='f1'):
      fit_preprocess(pd.DataFrame({'f1': [np.nan, np.nan]}), params,
                     {'num_cols': ['f1']})
  params = [{'Imputations': {'args': {
    'strategy': 'mean', 'cols': {'include_cols': ['f1']}}}}]
  with pytest.raises(ValueError, match='f1'):
    fit_preprocess(pd.DataFrame({'f1': [0.0, np.inf]}), params,
                   {'num_cols': ['f1']})


def test_fit_target_encode_nonfinite_global_raises():
  '''Fit target encode raises on nonfinite global.'''
  import numpy as np
  import pandas as pd
  import pytest
  from forecast_ml_framework.pipelines.data_prep.fitted import (
    fit_preprocess,
  )
  params = [{'TargetEncoding': {'args': {
    'cols': {'include_cols': ['cat']}}}}]
  with pytest.raises(ValueError):
    fit_preprocess(
      pd.DataFrame({'cat': ['a', 'b'], 'target': [np.inf, 1.0]}),
      params, {'cat_cols': ['cat'], 'target_col': 'target'})
  with pytest.raises(ValueError):
    fit_preprocess(
      pd.DataFrame({'cat': ['a', 'b'], 'target': [np.nan, np.nan]}),
      params, {'cat_cols': ['cat'], 'target_col': 'target'})


def test_fit_cap_inf_raises():
  '''Fit cap raises on inf data.'''
  import numpy as np
  import pandas as pd
  import pytest
  from forecast_ml_framework.pipelines.data_prep.fitted import (
    fit_preprocess,
  )
  params = [{'CapOutliers': {'args': {
    'cols': {'include_cols': ['f1']}}}}]
  with pytest.raises(ValueError, match='f1'):
    fit_preprocess(pd.DataFrame({'f1': [0.0, np.inf, 1.0]}), params,
                   {'num_cols': ['f1']})


def test_registry_to_json_rejects_nonfinite():
  '''Registry to JSON rejects nonfinite values.'''
  import pytest
  from forecast_ml_framework.pipelines.data_prep.fitted import (
    registry_to_json,
  )
  with pytest.raises(ValueError):
    registry_to_json({'version': 1, 'x': float('nan')})
  with pytest.raises(ValueError):
    registry_to_json({'version': 1, 'x': float('inf')})


def test_fit_ohe_dummy_name_collision_pair():
  '''Fit OHE raises on dummy name collision pair.'''
  import pandas as pd
  import pytest
  from forecast_ml_framework.pipelines.data_prep.fitted import (
    fit_preprocess,
  )
  train = pd.DataFrame({'a': ['b_c', 'x'], 'a_b': ['c', 'y'],
                        'target': [0, 1]})
  params = [{'CategoricalEncoding': {'args': {
    'ohe_flag': True, 'max_cols': 100,
    'cols': {'include_cols': ['a', 'a_b']}}}}]
  with pytest.raises(ValueError, match='collision|a_b_c'):
    fit_preprocess(train, params, {'cat_cols': ['a', 'a_b']})


def test_fit_ohe_dummy_collides_with_existing_column():
  '''Fit OHE raises when dummy collides with existing column.'''
  import pandas as pd
  import pytest
  from forecast_ml_framework.pipelines.data_prep.fitted import (
    fit_preprocess,
  )
  train = pd.DataFrame({'cat': ['a', 'b'], 'cat_a': [1, 0],
                        'target': [0, 1]})
  params = [{'CategoricalEncoding': {'args': {
    'ohe_flag': True, 'max_cols': 100,
    'cols': {'include_cols': ['cat']}}}}]
  with pytest.raises(ValueError, match='collision|cat_a'):
    fit_preprocess(train, params, {'cat_cols': ['cat']})


def test_apply_ohe_dummy_name_collision_pair():
  '''Apply OHE raises on dummy name collision pair.'''
  import pandas as pd
  import pytest
  from forecast_ml_framework.pipelines.data_prep.fitted import (
    apply_preprocess,
  )
  score = pd.DataFrame({'a': ['b_c'], 'a_b': ['c']})
  registry = {'version': 2, 'schema': 'fitted_registry.v2',
              'steps': {'CategoricalEncoding': {
                'a': {'categories': ['b_c'], 'ohe': True, 'max_cols': 100},
                'a_b': {'categories': ['c'], 'ohe': True, 'max_cols': 100}}}}
  with pytest.raises(ValueError, match='collision|a_b_c'):
    apply_preprocess(score, registry)


def test_apply_ohe_dummy_collides_with_existing_column():
  '''Apply OHE raises when dummy collides with existing column.'''
  import pandas as pd
  import pytest
  from forecast_ml_framework.pipelines.data_prep.fitted import (
    apply_preprocess,
  )
  score = pd.DataFrame({'cat': ['a'], 'cat_a': [1]})
  registry = {'version': 2, 'schema': 'fitted_registry.v2',
              'steps': {'CategoricalEncoding': {
                'cat': {'categories': ['a'], 'ohe': True, 'max_cols': 100}}}}
  with pytest.raises(ValueError, match='collision|cat_a'):
    apply_preprocess(score, registry)


def test_apply_registry_schema_mismatch_raises():
  '''Apply raises on registry schema mismatch.'''
  import pandas as pd
  import pytest
  from forecast_ml_framework.pipelines.data_prep.fitted import (
    apply_preprocess,
  )
  reg = {'version': 1, 'schema': 'other.v9',
         'steps': {'Normalizations': {
           'f1': {'kind': 'minmax', 'lo': 0.0, 'hi': 1.0}}}}
  with pytest.raises(ValueError, match='schema'):
    apply_preprocess(pd.DataFrame({'f1': [0.5]}), reg)
  reg2 = {'version': 1, 'schema': 'fitted_registry.v1',
          '__meta__': {'version': 999, 'schema': 'fitted_registry.v1'},
          'steps': {'Normalizations': {
            'f1': {'kind': 'minmax', 'lo': 0.0, 'hi': 1.0}}}}
  with pytest.raises(ValueError, match='version'):
    apply_preprocess(pd.DataFrame({'f1': [0.5]}), reg2)



# --- loop4 dedupe (RED) ---

def test_fitted_reuses_nodes_resolve_cols_and_iter_steps():
  '''fitted must import _resolve_cols/_iter_steps from nodes.'''
  from forecast_ml_framework.pipelines.data_prep import fitted, nodes
  assert fitted._resolve_cols is nodes._resolve_cols
  assert fitted._iter_steps is nodes._iter_steps


def test_jsonable_delegates_to_native():
  '''_jsonable output equals _native on nested numpy/tuple/dict.'''
  import numpy as np
  from forecast_ml_framework.pipelines.data_prep.fitted import (
    _jsonable, _native)
  obj = {'a': (np.int64(2), np.array([1.5, 2.5])), 'b': [np.bool_(True)]}
  assert _jsonable(obj) == _native(obj) == {
    'a': [2, [1.5, 2.5]], 'b': [True]}


# --- loop5 typed categories (RED) ---

def test_fit_apply_tuple_category_ohe_correct_ones():
  '''Tuple categories OHE gives correct ones on fit/apply.'''
  import pandas as pd
  from forecast_ml_framework.pipelines.data_prep.fitted import (
    fit_preprocess,
    apply_preprocess,
  )
  train = pd.DataFrame({'cat': [(1, 2), (3, 4), (1, 2), (3, 4)],
                        'target': [0, 1, 0, 1]})
  params = [{'CategoricalEncoding': {'args': {
    'ohe_flag': True, 'cols': {'include_cols': ['cat']}}}}]
  fitted, registry = fit_preprocess(
    train, params, {'cat_cols': ['cat']})
  assert fitted[[c for c in fitted.columns if c != 'target']].sum(
    axis=1).tolist() == [1, 1, 1, 1]
  out = apply_preprocess(train, registry)
  assert out[[c for c in out.columns if c != 'target']].sum(
    axis=1).tolist() == [1, 1, 1, 1]


def test_fit_apply_tuple_category_label_encoding_not_minus_one():
  '''Tuple categories label encoding is not minus one.'''
  import pandas as pd
  from forecast_ml_framework.pipelines.data_prep.fitted import (
    fit_preprocess,
    apply_preprocess,
  )
  train = pd.DataFrame({'cat': [(1, 2), (3, 4), (1, 2), (3, 4)],
                        'target': [0, 1, 0, 1]})
  params = [{'CategoricalEncoding': {'args': {
    'ohe_flag': False, 'cols': {'include_cols': ['cat']}}}}]
  _, registry = fit_preprocess(train, params, {'cat_cols': ['cat']})
  out = apply_preprocess(train, registry)
  assert out['cat'].tolist() == [0, 1, 0, 1]


def test_fit_apply_timestamp_category_ohe_correct_ones():
  '''Timestamp categories OHE gives correct ones on fit/apply.'''
  import pandas as pd
  from forecast_ml_framework.pipelines.data_prep.fitted import (
    fit_preprocess,
    apply_preprocess,
  )
  ts = [pd.Timestamp('2020-01-01'), pd.Timestamp('2020-01-02')]
  train = pd.DataFrame({'cat': ts * 2, 'target': [0, 1, 0, 1]})
  params = [{'CategoricalEncoding': {'args': {
    'ohe_flag': True, 'cols': {'include_cols': ['cat']}}}}]
  fitted, registry = fit_preprocess(
    train, params, {'cat_cols': ['cat']})
  dummy_cols = [c for c in fitted.columns if c != 'target']
  assert fitted[dummy_cols].sum(axis=1).tolist() == [1, 1, 1, 1]
  out = apply_preprocess(train, registry)
  assert out[dummy_cols].sum(axis=1).tolist() == [1, 1, 1, 1]


def test_fit_apply_tuple_category_target_encoding_not_global():
  '''Tuple categories target encoding is not global.'''
  import pandas as pd
  from forecast_ml_framework.pipelines.data_prep.fitted import (
    fit_preprocess,
    apply_preprocess,
  )
  train = pd.DataFrame({'cat': [(1, 2), (3, 4), (1, 2), (3, 4)],
                        'target': [0.0, 10.0, 0.0, 10.0]})
  params = [{'TargetEncoding': {'args': {
    'cols': {'include_cols': ['cat']}}}}]
  _, registry = fit_preprocess(
    train, params, {'cat_cols': ['cat'], 'target_col': 'target'})
  out = apply_preprocess(train, registry)
  assert out['cat'].tolist() == [0.0, 10.0, 0.0, 10.0]


def test_fit_apply_tuple_category_bin_labels_preserved():
  '''Tuple categories bin labels preserved on apply.'''
  import pandas as pd
  from forecast_ml_framework.pipelines.data_prep.fitted import (
    fit_preprocess,
    apply_preprocess,
  )
  train = pd.DataFrame({'cat': [(1, 2)] * 3 + [(3, 4)],
                        'target': [0, 1, 0, 1]})
  params = [{'HighLabelBinning': {'args': {
    'val_count': 2, 'cols': {'include_cols': ['cat']}}}}]
  _, registry = fit_preprocess(
    train, params, {'cat_cols': ['cat']})
  out = apply_preprocess(train, registry)
  assert out['cat'].tolist() == [(1, 2), (1, 2), (1, 2), 'Other']


def test_require_dict_shared_from_core_params():
  '''cost_tracking/retraining _require_dict is core.params helper.'''
  from forecast_ml_framework.core import params as core_params
  from forecast_ml_framework.pipelines.cost_tracking import nodes as cost
  from forecast_ml_framework.pipelines.retraining import nodes as retr
  assert cost._require_dict is core_params.require_dict
  assert retr._require_dict is core_params.require_dict
  assert core_params.require_dict(None, 'x') == {}
  import pytest
  with pytest.raises(ValueError, match='x must be a dict'):
    core_params.require_dict([1], 'x')


# --- loop6 typed categories (RED) ---

def test_typed_encode_roundtrip_date_decimal_uuid_bytes_period_timedelta():
  '''Typed encode roundtrips date/decimal/uuid/bytes/period/timedelta.'''
  import datetime
  import uuid
  from decimal import Decimal
  import pandas as pd
  from forecast_ml_framework.pipelines.data_prep.fitted import (
    _decode_typed,
    _encode_typed,
  )
  values = [
    datetime.date(2020, 1, 2),
    datetime.datetime(2020, 1, 2, 3, 4, 5),
    Decimal('12.50'),
    uuid.UUID('12345678-1234-5678-1234-567812345678'),
    b'\x00\xffabc',
    pd.Period('2020-01', freq='M'),
    pd.Timedelta(days=1, seconds=3),
    datetime.timedelta(days=2),
  ]
  import json
  for value in values:
    encoded = json.loads(json.dumps(_encode_typed(value)))
    decoded = _decode_typed(encoded)
    assert decoded == value, (value, decoded)
    assert type(decoded) is type(value), (value, decoded)


def test_fit_apply_typed_categories_json_roundtrip_label_encoding():
  '''Typed categories survive JSON roundtrip label encoding.'''
  import datetime
  import uuid
  from decimal import Decimal
  import pandas as pd
  from forecast_ml_framework.pipelines.data_prep.fitted import (
    apply_preprocess,
    fit_preprocess,
    registry_from_json,
    registry_to_json,
  )
  cats = [
    datetime.date(2020, 1, 1),
    Decimal('1.5'),
    uuid.UUID('12345678-1234-5678-1234-567812345678'),
    b'ab',
    pd.Period('2020-01', freq='M'),
    pd.Timedelta(days=1),
  ]
  train = pd.DataFrame({'cat': cats * 2, 'target': [0, 1] * 6})
  params = [{'CategoricalEncoding': {'args': {
    'ohe_flag': False, 'cols': {'include_cols': ['cat']}}}}]
  _, registry = fit_preprocess(
    train, params, {'cat_cols': ['cat']})
  rt = registry_from_json(registry_to_json(registry))
  out = apply_preprocess(train, rt)
  assert out['cat'].tolist() == [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5]


def test_legacy_plain_list_registry_tuple_data_raises_refit():
  '''Legacy plain-list registry with tuple data raises refit.'''
  import pandas as pd
  import pytest
  from forecast_ml_framework.pipelines.data_prep.fitted import (
    apply_preprocess,
  )
  reg = {'version': 1, 'schema': 'fitted_registry.v1',
         'CategoricalEncoding': {'cat': {
           'categories': [[1, 2], [3, 4]], 'ohe': False,
           'max_cols': None}}}
  with pytest.raises(ValueError, match='legacy|refit'):
    apply_preprocess(pd.DataFrame({'cat': [(1, 2), (3, 4)]}), reg)
