#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''P1-A1: fitted_registry v2 dual-read (Contract 2).

RED first: read_registry / require_v2 / v2 writer / node wiring /
YAML flags are missing, so these tests fail before GREEN.
'''

import copy
import datetime
import uuid
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


STEP_NAMES = [
  'Imputations', 'Normalizations', 'QuantileDiscretizer',
  'HighLabelBinning', 'CategoricalEncoding', 'TargetEncoding',
  'CapOutliers',
]


def _train():
  '''Return train frame exercising all seven fitted steps.'''
  n = 20
  num_imp = np.arange(n, dtype=float)
  num_imp[::5] = np.nan
  return pd.DataFrame({
    'num_imp': num_imp,
    'num_norm': np.arange(n, dtype=float) * 10.0,
    'num_disc': np.arange(n, dtype=float),
    'cat_bin': ['x'] * 16 + ['y'] * 4,
    'cat_enc': ['a', 'b', 'c', 'a'] * 5,
    'cat_tgt': ['p', 'q'] * 10,
    'num_cap': list(np.arange(n, dtype=float)) + [1000.0] * 0,
    'target': [0, 1] * 10,
  })


def _score():
  '''Return scoring frame (shifted values, same columns).'''
  return pd.DataFrame({
    'num_imp': [np.nan, 5.0, 100.0],
    'num_norm': [0.0, 95.0, 190.0],
    'num_disc': [0.0, 9.0, 19.0],
    'cat_bin': ['x', 'y', 'x'],
    'cat_enc': ['a', 'b', 'zzz'],
    'cat_tgt': ['p', 'q', 'p'],
    'num_cap': [0.0, 5.0, 999.0],
    'target': [0, 1, 0],
  })


def _variables():
  '''Return variables dict for the seven-step fixture.'''
  return {
    'num_cols': ['num_imp', 'num_norm', 'num_disc', 'num_cap'],
    'cat_cols': ['cat_bin', 'cat_enc', 'cat_tgt'],
    'target_col': 'target',
  }


def _params():
  '''Return prep params for all seven fitted steps.'''
  return [
    {'Imputations': {'args': {
      'strategy': 'mean',
      'cols': {'include_cols': ['num_imp']}}}},
    {'Normalizations': {'args': {
      'handle': 'MinMaxScaler',
      'cols': {'include_cols': ['num_norm']}}}},
    {'QuantileDiscretizer': {'args': {
      'params': {'numBuckets': 4},
      'cols': {'include_cols': ['num_disc']}}}},
    {'HighLabelBinning': {'args': {
      'val_count': 5,
      'cols': {'include_cols': ['cat_bin']}}}},
    {'CategoricalEncoding': {'args': {
      'ohe_flag': False,
      'cols': {'include_cols': ['cat_enc']}}}},
    {'TargetEncoding': {'args': {
      'smoothing': 10,
      'cols': {'include_cols': ['cat_tgt']}}}},
    {'CapOutliers': {'args': {
      'bound': {'lower': 0.05, 'upper': 0.95},
      'cols': {'include_cols': ['num_cap']}}}},
  ]


def _fit_v2():
  '''Fit on train and return registry (expect v2 after GREEN).'''
  from forecast_ml_framework.pipelines.data_prep.fitted import (
    fit_preprocess,
  )
  _, registry = fit_preprocess(_train(), _params(), _variables())
  return registry


def _steps_of(registry):
  '''Return steps mapping from either envelope shape.'''
  if isinstance(registry.get('steps'), dict):
    return dict(registry['steps'])
  meta = {'version', '__version__', 'schema', '__schema__', '__meta__',
          'steps'}
  return {k: v for k, v in registry.items() if k not in meta}


def _to_v1(registry):
  '''Convert a fitted registry to the v1 flat envelope.'''
  steps = _steps_of(registry)
  out = {
    'version': 1, '__version__': 1,
    'schema': 'fitted_registry.v1', '__schema__': 'fitted_registry.v1',
    '__meta__': {'version': 1, 'schema': 'fitted_registry.v1',
                 'typed_categories': True},
  }
  out.update(copy.deepcopy(steps))
  return out


def _valid_erasure():
  '''Return a valid __meta__.erasure marker.'''
  return {
    'subject_hash': 'a' * 64,
    'requested_at': '2026-11-08T12:00:00+00:00',
    'action': 'retrain_required',
  }


# --- writer / reader -------------------------------------------------------

def test_fit_writer_emits_v2_nested_only():
  '''fit_preprocess writes v2 envelope, nested steps only.'''
  registry = _fit_v2()
  assert registry['version'] == 2
  assert registry['__version__'] == 2
  assert registry['schema'] == 'fitted_registry.v2'
  assert registry['__schema__'] == 'fitted_registry.v2'
  assert registry['__meta__']['version'] == 2
  assert registry['__meta__']['schema'] == 'fitted_registry.v2'
  assert registry['__meta__']['typed_categories'] is True
  assert isinstance(registry['steps'], dict)
  assert set(registry['steps']) == set(STEP_NAMES)
  # No flat step keys at top level (nested-only writer).
  assert set(registry) - {
    'version', '__version__', 'schema', '__schema__', '__meta__',
    'steps'} == set()


def test_read_registry_v1_flat_returns_steps_only():
  '''read_registry(v1) returns 7-step dict, no meta keys.'''
  from forecast_ml_framework.pipelines.data_prep.fitted import (
    read_registry,
  )
  v1 = _to_v1(_fit_v2())
  steps = read_registry(v1)
  assert set(steps) == set(STEP_NAMES)
  assert 'version' not in steps
  assert 'schema' not in steps
  assert '__meta__' not in steps
  assert '__version__' not in steps
  assert '__schema__' not in steps
  assert 'steps' not in steps


def test_read_registry_v2_nested_returns_steps_only():
  '''read_registry(v2) returns 7-step dict, no meta keys.'''
  from forecast_ml_framework.pipelines.data_prep.fitted import (
    read_registry,
  )
  v2 = _fit_v2()
  steps = read_registry(v2)
  assert set(steps) == set(STEP_NAMES)
  assert steps == _steps_of(v2)
  for key in ('version', '__version__', 'schema', '__schema__',
              '__meta__', 'steps'):
    assert key not in steps


def test_read_registry_v1_v2_steps_equal():
  '''read_registry v1 and v2 steps_dict are equal.'''
  from forecast_ml_framework.pipelines.data_prep.fitted import (
    read_registry,
  )
  v2 = _fit_v2()
  assert read_registry(_to_v1(v2)) == read_registry(v2)


def test_parity_apply_v1_v2_frames_equal():
  '''apply_preprocess(df, v1) == apply_preprocess(df, v2).'''
  from forecast_ml_framework.pipelines.data_prep.fitted import (
    apply_preprocess,
  )
  v2 = _fit_v2()
  v1 = _to_v1(v2)
  out_v1 = apply_preprocess(_score(), v1)
  out_v2 = apply_preprocess(_score(), v2)
  pd.testing.assert_frame_equal(out_v1, out_v2)


def test_read_registry_require_v2_rejects_v1():
  '''read_registry(require_v2=True) rejects v1.'''
  from forecast_ml_framework.pipelines.data_prep.fitted import (
    read_registry,
  )
  with pytest.raises(ValueError, match='requires fitted_registry v2'):
    read_registry(_to_v1(_fit_v2()), require_v2=True)


def test_read_registry_require_v2_accepts_v2():
  '''read_registry(require_v2=True) accepts v2.'''
  from forecast_ml_framework.pipelines.data_prep.fitted import (
    read_registry,
  )
  steps = read_registry(_fit_v2(), require_v2=True)
  assert set(steps) == set(STEP_NAMES)


def test_apply_preprocess_require_v2_rejects_v1():
  '''fitted.apply_preprocess(require_v2=True) rejects v1.'''
  from forecast_ml_framework.pipelines.data_prep.fitted import (
    apply_preprocess,
  )
  with pytest.raises(ValueError, match='requires fitted_registry v2'):
    apply_preprocess(_score(), _to_v1(_fit_v2()), require_v2=True)


def test_apply_preprocess_require_v2_accepts_v2():
  '''fitted.apply_preprocess(require_v2=True) accepts v2.'''
  from forecast_ml_framework.pipelines.data_prep.fitted import (
    apply_preprocess,
  )
  out = apply_preprocess(_score(), _fit_v2(), require_v2=True)
  assert len(out) == 3


def test_check_registry_version_mixed_version_schema_raises():
  '''version=2 + schema v1 (and reverse) raises mismatch.'''
  from forecast_ml_framework.pipelines.data_prep.fitted import (
    _check_registry_version, read_registry,
  )
  for reg in (
    {'version': 2, 'schema': 'fitted_registry.v1'},
    {'version': 1, 'schema': 'fitted_registry.v2'},
    {'version': 2, '__schema__': 'fitted_registry.v1'},
    {'version': 2, 'schema': 'fitted_registry.v2',
     '__meta__': {'version': 2, 'schema': 'fitted_registry.v1'}},
  ):
    with pytest.raises(ValueError, match='schema mismatch'):
      _check_registry_version(reg)
    with pytest.raises(ValueError, match='schema mismatch'):
      read_registry(reg)


def test_check_registry_version_two_versions_raises():
  '''Two different version values present raises mismatch.'''
  from forecast_ml_framework.pipelines.data_prep.fitted import (
    _check_registry_version,
  )
  reg = {'version': 1, '__version__': 2}
  with pytest.raises(ValueError, match='version'):
    _check_registry_version(reg)
  reg2 = {'version': 1, '__meta__': {'version': 2}}
  with pytest.raises(ValueError, match='version'):
    _check_registry_version(reg2)


def test_check_registry_version_unknown_v3_raises():
  '''Unknown version 3 raises (never partial apply).'''
  from forecast_ml_framework.pipelines.data_prep.fitted import (
    read_registry,
  )
  reg = {'version': 3, 'schema': 'fitted_registry.v3',
         'Normalizations': {'f1': {'kind': 'minmax', 'lo': 0.0,
                                   'hi': 1.0}}}
  with pytest.raises(ValueError, match='version|schema'):
    read_registry(reg)


def test_check_registry_version_unknown_schema_raises():
  '''Unknown schema string raises mismatch.'''
  from forecast_ml_framework.pipelines.data_prep.fitted import (
    read_registry,
  )
  reg = {'version': 1, 'schema': 'other.v9',
         'Normalizations': {'f1': {'kind': 'minmax', 'lo': 0.0,
                                   'hi': 1.0}}}
  with pytest.raises(ValueError, match='schema'):
    read_registry(reg)
  reg2 = {'schema': 'fitted_registry.v3'}
  with pytest.raises(ValueError, match='schema'):
    read_registry(reg2)


def test_check_registry_version_schema_only_and_version_only():
  '''Schema-only v2 and version-only v2 are consistent.'''
  from forecast_ml_framework.pipelines.data_prep.fitted import (
    _check_registry_version,
  )
  _check_registry_version({'schema': 'fitted_registry.v2'})
  _check_registry_version({'version': 2})
  _check_registry_version({'version': 1})
  _check_registry_version({})
  # Schema-only v2 satisfies require_v2; version-only v1 does not.
  _check_registry_version({'schema': 'fitted_registry.v2'},
                          require_v2=True)
  with pytest.raises(ValueError, match='requires fitted_registry v2'):
    _check_registry_version({'version': 1}, require_v2=True)
  with pytest.raises(ValueError, match='requires fitted_registry v2'):
    _check_registry_version({}, require_v2=True)


def test_read_registry_non_dict_raises():
  '''read_registry rejects non-dict input.'''
  from forecast_ml_framework.pipelines.data_prep.fitted import (
    _registry_steps, read_registry,
  )
  with pytest.raises(ValueError):
    read_registry(['not', 'a', 'dict'])
  with pytest.raises(ValueError):
    _registry_steps('nope')


def test_read_registry_duplicate_flat_nested_raises():
  '''Flat + nested same step name raises duplicate.'''
  from forecast_ml_framework.pipelines.data_prep.fitted import (
    read_registry,
  )
  reg = {'version': 2, 'schema': 'fitted_registry.v2',
         'steps': {'Normalizations': {'f1': {'kind': 'minmax',
                                             'lo': 0.0, 'hi': 1.0}}},
         'Normalizations': {'f1': {'kind': 'minmax',
                                   'lo': 0.0, 'hi': 1.0}}}
  with pytest.raises(ValueError, match='Duplicate'):
    read_registry(reg)


def test_registry_steps_delegates_to_read_registry():
  '''_registry_steps returns steps_dict and honours require_v2.'''
  from forecast_ml_framework.pipelines.data_prep.fitted import (
    _registry_steps, read_registry,
  )
  v2 = _fit_v2()
  assert _registry_steps(v2) == read_registry(v2)
  with pytest.raises(ValueError, match='requires fitted_registry v2'):
    _registry_steps(_to_v1(v2), require_v2=True)


def test_erasure_marker_valid_accepted_and_excluded():
  '''Valid __meta__.erasure passes and is excluded from steps.'''
  from forecast_ml_framework.pipelines.data_prep.fitted import (
    read_registry,
  )
  v2 = _fit_v2()
  v2['__meta__']['erasure'] = _valid_erasure()
  steps = read_registry(v2)
  assert set(steps) == set(STEP_NAMES)
  assert '__meta__' not in steps
  assert 'erasure' not in steps


def test_erasure_marker_invalid_raises():
  '''Malformed __meta__.erasure raises ValueError.'''
  from forecast_ml_framework.pipelines.data_prep.fitted import (
    read_registry,
  )
  base = _fit_v2()
  bad_markers = [
    'not-a-dict',
    {'subject_hash': 'a' * 64, 'requested_at':
     '2026-11-08T12:00:00+00:00'},  # missing action
    dict(_valid_erasure(), extra='x'),
    dict(_valid_erasure(), subject_hash='short'),
    dict(_valid_erasure(), subject_hash='z' * 64),
    dict(_valid_erasure(), requested_at='not-a-date'),
    dict(_valid_erasure(), action='delete'),
    dict(_valid_erasure(), subject_hash=123),
  ]
  for marker in bad_markers:
    reg = copy.deepcopy(base)
    reg['__meta__']['erasure'] = marker
    with pytest.raises(ValueError, match='erasure'):
      read_registry(reg)


def test_apply_preprocess_replays_verbatim_no_refit():
  '''apply replays train stats, never recomputes on score.'''
  from forecast_ml_framework.pipelines.data_prep.fitted import (
    apply_preprocess, fit_preprocess,
  )
  train = pd.DataFrame({'f1': [1.0, 3.0, 5.0], 'target': [0, 1, 0]})
  params = [{'Imputations': {'args': {
    'strategy': 'median', 'cols': {'include_cols': ['f1']}}}}]
  _, registry = fit_preprocess(train, params, {'num_cols': ['f1']})
  out = apply_preprocess(pd.DataFrame({'f1': [np.nan, 100.0]}), registry)
  assert out['f1'].iloc[0] == 3.0
  # Normalizer train bounds replayed on shifted data.
  train2 = pd.DataFrame({'f1': [0.0, 10.0], 'target': [0, 1]})
  params2 = [{'Normalizations': {'args': {
    'handle': 'MinMaxScaler', 'cols': {'include_cols': ['f1']}}}}]
  _, reg2 = fit_preprocess(train2, params2, {'num_cols': ['f1']})
  out2 = apply_preprocess(pd.DataFrame({'f1': [10.0, 20.0]}), reg2)
  assert abs(out2['f1'].iloc[0] - 1.0) < 1e-9
  assert abs(out2['f1'].iloc[1] - 2.0) < 1e-9


def test_apply_preprocess_json_string_and_empty():
  '''apply accepts JSON string registry; empty string is empty.'''
  from forecast_ml_framework.pipelines.data_prep.fitted import (
    apply_preprocess, registry_from_json, registry_to_json,
  )
  v2 = _fit_v2()
  rt = registry_from_json(registry_to_json(v2))
  pd.testing.assert_frame_equal(
    apply_preprocess(_score(), v2), apply_preprocess(_score(), rt))
  pd.testing.assert_frame_equal(
    apply_preprocess(_score(), registry_to_json(v2)), apply_preprocess(
      _score(), v2))
  frame = _score()
  pd.testing.assert_frame_equal(apply_preprocess(frame, ''), frame)


def test_apply_preprocess_unsupported_step_raises():
  '''apply rejects unknown step in registry.'''
  from forecast_ml_framework.pipelines.data_prep.fitted import (
    apply_preprocess,
  )
  reg = {'version': 1, 'UnknownStep': {'f1': {}}}
  with pytest.raises(ValueError, match='UnknownStep'):
    apply_preprocess(pd.DataFrame({'f1': [1.0]}), reg)


def test_fit_preprocess_duplicate_unsupported_skip():
  '''fit rejects duplicate/unsupported; skip=True is ignored.'''
  from forecast_ml_framework.pipelines.data_prep.fitted import (
    fit_preprocess,
  )
  train = pd.DataFrame({'f1': [0.0, 10.0], 'target': [0, 1]})
  step = {'Normalizations': {'args': {
    'handle': 'MinMaxScaler', 'cols': {'include_cols': ['f1']}}}}
  with pytest.raises(ValueError, match='Duplicate'):
    fit_preprocess(train, [step, dict(step)], {'num_cols': ['f1']})
  bad = [{'KnnImputation': {'args': {
    'cols': {'include_cols': ['f1']}}}}]
  with pytest.raises(ValueError, match='KnnImputation'):
    fit_preprocess(train, bad, {'num_cols': ['f1']})
  skipped = [{'Normalizations': {'skip': True, 'args': {
    'handle': 'MinMaxScaler', 'cols': {'include_cols': ['f1']}}}}]
  fitted, registry = fit_preprocess(
    train, skipped, {'num_cols': ['f1']})
  assert list(fitted['f1']) == [0.0, 10.0]
  assert _steps_of(registry) == {}


# --- node / pipeline wiring ------------------------------------------------

def test_nodes_apply_preprocess_forwards_require_v2():
  '''nodes.apply_preprocess forwards modeling_params flag.'''
  from forecast_ml_framework.pipelines.data_prep.nodes import (
    apply_preprocess as node_apply,
  )
  v2 = _fit_v2()
  v1 = _to_v1(v2)
  out = node_apply(_score(), v1, {'fitted_registry_v2': False})
  pd.testing.assert_frame_equal(out, apply_out := node_apply(
    _score(), v2, {'fitted_registry_v2': True}))
  out_none = node_apply(_score(), v1, None)
  pd.testing.assert_frame_equal(out_none, out)
  out_default = node_apply(_score(), v1)
  pd.testing.assert_frame_equal(out_default, out)
  with pytest.raises(ValueError, match='requires fitted_registry v2'):
    node_apply(_score(), v1, {'fitted_registry_v2': True})


def test_data_prep_pipeline_apply_node_inputs_exact():
  '''data_prep apply_node inputs are exactly the H1 list.'''
  from forecast_ml_framework.pipelines.data_prep.pipeline import (
    create_pipeline,
  )
  nodes = {n.name: n for n in create_pipeline().nodes}
  assert nodes['apply_preprocess'].inputs == [
    'test_raw', 'fitted_registry', 'params:modeling_params']


def test_data_prep_pipeline_apply_node_flag_behaviour():
  '''data_prep node: True+v1 raises, False scores.'''
  from forecast_ml_framework.pipelines.data_prep.pipeline import (
    create_pipeline,
  )
  nodes = {n.name: n for n in create_pipeline().nodes}
  func = nodes['apply_preprocess'].func
  v2 = _fit_v2()
  v1 = _to_v1(v2)
  with pytest.raises(ValueError, match='requires fitted_registry v2'):
    func(_score(), v1, {'fitted_registry_v2': True})
  out = func(_score(), v1, {'fitted_registry_v2': False})
  assert len(out) == 3


def test_scoring_pipeline_score_node_reads_scoring_params():
  '''scoring score_data_prep node reads params:scoring_params.'''
  from forecast_ml_framework.pipelines.scoring.pipeline import (
    create_pipeline,
  )
  nodes = {n.name: n for n in create_pipeline().nodes}
  assert 'params:scoring_params' in nodes['score_data_prep'].inputs
  assert 'fitted_registry' in nodes['score_data_prep'].inputs
  func = nodes['score_data_prep'].func
  v2 = _fit_v2()
  v1 = _to_v1(v2)
  with pytest.raises(ValueError, match='requires fitted_registry v2'):
    func(_score(), v1, {'fitted_registry_v2': True})
  out = func(_score(), v1, {'fitted_registry_v2': False})
  assert len(out) == 3


def test_scoring_score_data_prep_forwards_require_v2():
  '''score_data_prep forwards scoring_params flag to reader.'''
  from forecast_ml_framework.pipelines.scoring.pipeline import (
    score_data_prep,
  )
  v2 = _fit_v2()
  v1 = _to_v1(v2)
  with pytest.raises(ValueError, match='requires fitted_registry v2'):
    score_data_prep(_score(), v1, {'fitted_registry_v2': True})
  out = score_data_prep(_score(), v1, {'fitted_registry_v2': False})
  assert len(out) == 3
  out_v2 = score_data_prep(_score(), v2, {'fitted_registry_v2': True})
  pd.testing.assert_frame_equal(out, out_v2)


def test_scoring_empty_registry_guard_unchanged():
  '''Empty registry guard: fitted steps raise, else identity.'''
  from forecast_ml_framework.pipelines.scoring.pipeline import (
    score_data_prep,
  )
  frame = _score()
  params = {'data_prep_params': [{'Normalizations': {'args': {}}}]}
  with pytest.raises(ValueError, match='fitted_registry'):
    score_data_prep(frame, {}, params)
  with pytest.raises(ValueError, match='fitted_registry'):
    score_data_prep(frame, None, params)
  pd.testing.assert_frame_equal(score_data_prep(frame, {}, {}), frame)


def test_yaml_both_flags_false_no_top_level():
  '''Both YAML flags exist as false, no top-level key.'''
  import yaml
  base = Path(__file__).resolve().parents[1]
  modeling = yaml.safe_load(
    (base / 'conf' / 'base' / 'parameters' / 'modeling.yml').read_text())
  platform = yaml.safe_load(
    (base / 'conf' / 'base' / 'parameters' / 'platform.yml').read_text())
  assert modeling['modeling_params']['fitted_registry_v2'] is False
  assert platform['scoring_params']['fitted_registry_v2'] is False
  assert 'fitted_registry_v2' not in modeling
  assert 'fitted_registry_v2' not in platform


def test_no_per_caller_version_branches_outside_reader():
  '''Callers do not duplicate version== checks (grep guard).'''
  import inspect
  from forecast_ml_framework.pipelines.data_prep import nodes
  from forecast_ml_framework.pipelines.scoring import pipeline
  for func in (nodes.apply_preprocess, pipeline.score_data_prep):
    src = inspect.getsource(func)
    assert 'version ==' not in src
    assert 'version==' not in src


# --- coverage helpers: native / typed / json ------------------------------

def test_native_and_typed_roundtrip_covers_helpers():
  '''_native/_encode/_decode cover typed scalar kinds.'''
  import json
  from forecast_ml_framework.pipelines.data_prep.fitted import (
    _decode_typed, _encode_typed, _jsonable, _native,
  )
  values = [
    np.int64(2), np.float64(1.5), np.bool_(True),
    np.array([1.5, 2.5]), (1, 2), {'k': np.int64(3)},
    pd.Timestamp('2020-01-01'), datetime.datetime(2020, 1, 2, 3, 4, 5),
    datetime.date(2020, 1, 2), Decimal('12.50'),
    uuid.UUID('12345678-1234-5678-1234-567812345678'), b'\x00abc',
    pd.Period('2020-01', freq='M'), pd.Timedelta(days=1),
    datetime.timedelta(days=2), 'plain', 7, 2.5, None,
  ]
  for value in values:
    encoded = json.loads(json.dumps(_jsonable(_encode_typed(value))))
    decoded = _decode_typed(encoded)
    if isinstance(value, (np.ndarray, tuple, dict)):
      continue
    if isinstance(value, (np.int64, np.bool_)):
      assert decoded == value
    else:
      assert decoded == value, (value, decoded)
  assert _native(np.array([np.int64(1)])) == [1]
  assert _jsonable({'a': (np.int64(2),)}) == {'a': [2]}


def test_registry_json_nonfinite_rejected():
  '''registry_to_json rejects NaN/inf.'''
  from forecast_ml_framework.pipelines.data_prep.fitted import (
    registry_to_json,
  )
  with pytest.raises(ValueError):
    registry_to_json({'x': float('nan')})
  with pytest.raises(ValueError):
    registry_to_json({'x': float('inf')})


def test_fit_impute_strategies_and_all_nan():
  '''Imputations fit: mean/median/mode/custom/none + all-NaN.'''
  from forecast_ml_framework.pipelines.data_prep.fitted import (
    fit_preprocess,
  )
  train = pd.DataFrame({'f1': [1.0, np.nan, 5.0], 'target': [0, 1, 0]})
  for strategy in ('mean', 'median', 'mode'):
    params = [{'Imputations': {'args': {
      'strategy': strategy, 'cols': {'include_cols': ['f1']}}}}]
    _, reg = fit_preprocess(train, params, {'num_cols': ['f1']})
    assert 'Imputations' in _steps_of(reg)
  params = [{'Imputations': {'args': {
    'strategy': 'none', 'custom_bounds': {'f1': 9.0},
    'cols': {'include_cols': ['f1']}}}}]
  _, reg = fit_preprocess(train, params, {'num_cols': ['f1']})
  assert _steps_of(reg)['Imputations']['f1']['fill'] == 9.0
  params_none = [{'Imputations': {'args': {
    'strategy': 'none', 'cols': {'include_cols': ['f1']}}}}]
  _, reg_none = fit_preprocess(train, params_none, {
    'num_cols': ['f1']})
  assert _steps_of(reg_none)['Imputations'] == {}
  for strategy in ('mean', 'median', 'mode'):
    bad = [{'Imputations': {'args': {
      'strategy': strategy, 'cols': {'include_cols': ['f1']}}}}]
    with pytest.raises(ValueError, match='f1'):
      fit_preprocess(pd.DataFrame({'f1': [np.nan, np.nan]}), bad,
                     {'num_cols': ['f1']})
  with pytest.raises(ValueError, match='f1'):
    fit_preprocess(pd.DataFrame({'f1': [0.0, np.inf]}), [
      {'Imputations': {'args': {
        'strategy': 'mean', 'cols': {'include_cols': ['f1']}}}}],
      {'num_cols': ['f1']})


def test_fit_normalize_standard_and_errors():
  '''Normalizations fit: standard (ddof=0) + all-NaN/inf.'''
  from forecast_ml_framework.pipelines.data_prep.fitted import (
    apply_preprocess, fit_preprocess,
  )
  train = pd.DataFrame({'f1': [0.0, 10.0], 'target': [0, 1]})
  params = [{'Normalizations': {'args': {
    'handle': 'StandardScaler', 'cols': {'include_cols': ['f1']}}}}]
  fitted, reg = fit_preprocess(train, params, {'num_cols': ['f1']})
  assert list(fitted['f1']) == [-1.0, 1.0]
  out = apply_preprocess(pd.DataFrame({'f1': [5.0]}), reg)
  assert abs(out['f1'].iloc[0]) < 1e-9
  # Constant column: sd 0 -> 1.0 fallback.
  const = pd.DataFrame({'f1': [3.0, 3.0], 'target': [0, 1]})
  fitted_c, _ = fit_preprocess(const, params, {'num_cols': ['f1']})
  assert list(fitted_c['f1']) == [0.0, 0.0]
  for bad_frame in (pd.DataFrame({'f1': [np.nan, np.nan]}),
                    pd.DataFrame({'f1': [0.0, np.inf]})):
    with pytest.raises(ValueError, match='f1'):
      fit_preprocess(bad_frame, params, {'num_cols': ['f1']})


def test_fit_discretize_errors_and_constant():
  '''QuantileDiscretizer fit: bucket guards, all-NaN, constant.'''
  from forecast_ml_framework.pipelines.data_prep.fitted import (
    fit_preprocess,
  )
  train = pd.DataFrame({'f1': [0.0, 10.0], 'target': [0, 1]})
  for buckets in ('bad', 0, 100000, True):
    params = [{'QuantileDiscretizer': {'args': {
      'params': {'numBuckets': buckets},
      'cols': {'include_cols': ['f1']}}}}]
    with pytest.raises(ValueError, match='numBuckets'):
      fit_preprocess(train, params, {'num_cols': ['f1']})
  params = [{'QuantileDiscretizer': {'args': {
    'params': {'numBuckets': 2}, 'cols': {'include_cols': ['f1']}}}}]
  with pytest.raises(ValueError, match='f1'):
    fit_preprocess(pd.DataFrame({'f1': [np.nan, np.nan]}), params,
                   {'num_cols': ['f1']})
  with pytest.raises(ValueError, match='f1'):
    fit_preprocess(pd.DataFrame({'f1': [0.0, np.inf, 1.0]}), params,
                   {'num_cols': ['f1']})
  fitted, _ = fit_preprocess(
    pd.DataFrame({'f1': [1.0, 1.0], 'target': [0, 1]}), params,
    {'num_cols': ['f1']})
  assert len(fitted) == 2


def test_apply_step_state_errors():
  '''Apply helpers reject malformed state (all steps).'''
  from forecast_ml_framework.pipelines.data_prep.fitted import (
    apply_preprocess,
  )
  df = pd.DataFrame({'f1': [0.5], 'cat': ['a']})
  cases = [
    {'version': 1, 'Normalizations': {'f1': 'bad'}},
    {'version': 1, 'Normalizations': {'f1': {'kind': 'weird'}}},
    {'version': 1, 'Normalizations': {'f1': {
      'kind': 'standard', 'mu': float('nan'), 'sd': 1.0}}},
    {'version': 1, 'Normalizations': {'f1': {
      'kind': 'standard', 'mu': 0.0, 'sd': 0.0}}},
    {'version': 1, 'Normalizations': {'f1': {
      'kind': 'standard', 'mu': 0.0, 'sd': float('inf')}}},
    {'version': 1, 'Normalizations': {'f1': {
      'kind': 'minmax', 'lo': float('nan'), 'hi': 1.0}}},
    {'version': 1, 'QuantileDiscretizer': {'f1': 'bad'}},
    {'version': 1, 'QuantileDiscretizer': {'f1': {'edges': [1.0]}}},
    {'version': 1, 'QuantileDiscretizer': {
      'f1': {'edges': [0.0, 2.0, 1.0]}}},
    {'version': 1, 'QuantileDiscretizer': {
      'f1': {'edges': [0.0, float('inf')]}}},
    {'version': 1, 'Imputations': {'f1': 'bad'}},
    {'version': 1, 'Imputations': {'f1': {'fill': [1.0]}}},
    {'version': 1, 'Imputations': {'f1': {
      'fill': float('inf'), 'strategy': 'mean'}}},
    {'version': 1, 'TargetEncoding': {'cat': {
      'mapping': [['a', float('nan')]], 'global': 0.0}}},
    {'version': 1, 'TargetEncoding': {'cat': {
      'mapping': {'a': float('nan')}, 'global': 0.0}}},
    {'version': 1, 'TargetEncoding': {'cat': {
      'mapping': [['a']], 'global': 0.0}}},
    {'version': 1, 'TargetEncoding': {'cat': {
      'mapping': [['a', 0.5]], 'global': float('inf')}}},
    {'version': 1, 'CapOutliers': {'f1': {'lo': 'x', 'hi': 1.0}}},
    {'version': 1, 'CapOutliers': {'f1': {'lo': 5.0, 'hi': 1.0}}},
  ]
  for reg in cases:
    with pytest.raises(ValueError):
      apply_preprocess(df.copy(), reg)
  # TargetEncoding dict mapping + missing global -> 0.0 fallback.
  reg = {'version': 1, 'TargetEncoding': {'cat': {
    'mapping': {'a': 0.7}}}}
  out = apply_preprocess(pd.DataFrame({'cat': ['a', 'zz']}), reg)
  assert out['cat'].tolist() == [0.7, 0.0]
  # Cap with None bounds is a no-op clip.
  reg_cap = {'version': 1, 'CapOutliers': {'f1': {}}}
  out_cap = apply_preprocess(pd.DataFrame({'f1': [5.0]}), reg_cap)
  assert out_cap['f1'].tolist() == [5.0]


def test_fit_cap_errors_and_custom():
  '''CapOutliers fit: custom, quantile guards, bound types.'''
  from forecast_ml_framework.pipelines.data_prep.fitted import (
    fit_preprocess,
  )
  train = pd.DataFrame({'f1': [0.0, 10.0], 'target': [0, 1]})
  fitted, reg = fit_preprocess(train, [{'CapOutliers': {'args': {
    'custom_bound': {'f1': [2.0, 8.0]},
    'cols': {'include_cols': ['f1']}}}}], {'num_cols': ['f1']})
  assert list(fitted['f1']) == [2.0, 8.0]
  bad_cases = [
    {'custom_bound': {'f1': [5.0, 1.0]}},
    {'custom_bound': {'f1': 'bad'}},
    {'bound': {'lower': 0.9, 'upper': 0.1}},
    {'bound': {'lower': -0.1, 'upper': 0.5}},
    {'bound': 'bad'},
    {'custom_bound': 'bad'},
  ]
  for args in bad_cases:
    params = [{'CapOutliers': {'args': dict(
      args, cols={'include_cols': ['f1']})}}]
    with pytest.raises(ValueError):
      fit_preprocess(train, params, {'num_cols': ['f1']})
  with pytest.raises(ValueError, match='f1'):
    fit_preprocess(pd.DataFrame({'f1': [np.nan, np.nan]}), [
      {'CapOutliers': {'args': {'cols': {'include_cols': ['f1']}}}}],
      {'num_cols': ['f1']})
  with pytest.raises(ValueError, match='f1'):
    fit_preprocess(pd.DataFrame({'f1': [0.0, np.inf, 1.0]}), [
      {'CapOutliers': {'args': {'cols': {'include_cols': ['f1']}}}}],
      {'num_cols': ['f1']})


def test_fit_target_encode_and_oof():
  '''TargetEncoding fit: smoothing, no-target, nonfinite.'''
  from forecast_ml_framework.pipelines.data_prep.fitted import (
    fit_preprocess,
  )
  train = pd.DataFrame({'cat': ['a', 'b', 'a', 'b'],
                        'target': [0.0, 10.0, 0.0, 10.0]})
  # No target column -> empty state, no error.
  _, reg = fit_preprocess(
    pd.DataFrame({'cat': ['a', 'b']}),
    [{'TargetEncoding': {'args': {'cols': {'include_cols': ['cat']}}}}],
    {'cat_cols': ['cat'], 'target_col': 'target'})
  assert _steps_of(reg)['TargetEncoding'] == {}
  # Smoothing 0 (default) and >0 both fit.
  for smoothing in (None, 10):
    args = {'cols': {'include_cols': ['cat']}}
    if smoothing is not None:
      args['smoothing'] = smoothing
    fitted, reg2 = fit_preprocess(
      train, [{'TargetEncoding': {'args': args}}],
      {'cat_cols': ['cat'], 'target_col': 'target'})
    assert len(fitted) == 4
    assert 'TargetEncoding' in _steps_of(reg2)
  for bad_target in ([np.inf, 1.0], [np.nan, np.nan]):
    with pytest.raises(ValueError):
      fit_preprocess(
        pd.DataFrame({'cat': ['a', 'b'], 'target': bad_target}),
        [{'TargetEncoding': {'args': {
          'cols': {'include_cols': ['cat']}}}}],
        {'cat_cols': ['cat'], 'target_col': 'target'})


def test_fit_encode_cat_ohe_and_collisions():
  '''CategoricalEncoding fit: OHE, max_cols, dummy collisions.'''
  from forecast_ml_framework.pipelines.data_prep.fitted import (
    apply_preprocess, fit_preprocess,
  )
  train = pd.DataFrame({'cat': ['a', 'b', 'a', 'b'],
                        'target': [0, 1, 0, 1]})
  params = [{'CategoricalEncoding': {'args': {
    'ohe_flag': True, 'cols': {'include_cols': ['cat']}}}}]
  fitted, reg = fit_preprocess(train, params, {'cat_cols': ['cat']})
  assert set(fitted.columns) == {'cat_a', 'cat_b', 'target'}
  out = apply_preprocess(
    pd.DataFrame({'cat': ['a', 'unseen', 'b']}), reg)
  assert 'cat_unseen' not in out.columns
  # Label encoding unseen -> -1.
  params_lab = [{'CategoricalEncoding': {'args': {
    'ohe_flag': False, 'cols': {'include_cols': ['cat']}}}}]
  _, reg_lab = fit_preprocess(train, params_lab, {'cat_cols': ['cat']})
  out_lab = apply_preprocess(pd.DataFrame({'cat': ['zz']}), reg_lab)
  assert out_lab['cat'].tolist() == [-1]
  # Per-column max_cols exceeded at fit.
  big = pd.DataFrame({'cat': ['c%d' % i for i in range(1500)],
                      'target': [0, 1] * 750})
  with pytest.raises(ValueError, match='max_cols'):
    fit_preprocess(big, [{'CategoricalEncoding': {'args': {
      'ohe_flag': True, 'max_cols': 1000,
      'cols': {'include_cols': ['cat']}}}}], {'cat_cols': ['cat']})
  # Dummy collision pair and existing-column collision at fit.
  pair = pd.DataFrame({'a': ['b_c', 'x'], 'a_b': ['c', 'y'],
                       'target': [0, 1]})
  with pytest.raises(ValueError, match='collision|a_b_c'):
    fit_preprocess(pair, [{'CategoricalEncoding': {'args': {
      'ohe_flag': True, 'max_cols': 100,
      'cols': {'include_cols': ['a', 'a_b']}}}}],
      {'cat_cols': ['a', 'a_b']})
  exist = pd.DataFrame({'cat': ['a', 'b'], 'cat_a': [1, 0],
                        'target': [0, 1]})
  with pytest.raises(ValueError, match='collision|cat_a'):
    fit_preprocess(exist, [{'CategoricalEncoding': {'args': {
      'ohe_flag': True, 'max_cols': 100,
      'cols': {'include_cols': ['cat']}}}}], {'cat_cols': ['cat']})


def test_apply_encode_cat_collisions_and_caps():
  '''CategoricalEncoding apply: collisions, caps, duplicate cols.'''
  from forecast_ml_framework.pipelines.data_prep.fitted import (
    apply_preprocess,
  )
  # Collision pair at apply.
  reg = {'CategoricalEncoding': {
    'a': {'categories': ['b_c'], 'ohe': True, 'max_cols': 100},
    'a_b': {'categories': ['c'], 'ohe': True, 'max_cols': 100}}}
  with pytest.raises(ValueError, match='collision|a_b_c'):
    apply_preprocess(pd.DataFrame({'a': ['b_c'], 'a_b': ['c']}), reg)
  # Existing-column collision at apply.
  reg2 = {'CategoricalEncoding': {
    'cat': {'categories': ['a'], 'ohe': True, 'max_cols': 100}}}
  with pytest.raises(ValueError, match='collision|cat_a'):
    apply_preprocess(pd.DataFrame({'cat': ['a'], 'cat_a': [1]}), reg2)
  # Per-column cap at apply (max_cols in state).
  reg3 = {'CategoricalEncoding': {'cat': {
    'categories': ['c%d' % i for i in range(2000)],
    'ohe': True, 'max_cols': 1000}}}
  with pytest.raises(ValueError, match='max_cols'):
    apply_preprocess(pd.DataFrame({'cat': ['a', 'b'] * 600}), reg3)
  # Total-expansion cap at apply: many cols each under cap.
  frame = pd.DataFrame({('c%d' % i): ['a', 'b']
                        for i in range(600)})
  reg4 = {'CategoricalEncoding': {
    ('c%d' % i): {'categories': ['a', 'b'], 'ohe': True,
                  'max_cols': 1000} for i in range(600)}}
  with pytest.raises(ValueError, match='max_cols'):
    apply_preprocess(frame, reg4)
  # OHE without max_cols in state (env/default resolution path).
  reg5 = {'CategoricalEncoding': {
    'cat': {'categories': ['a', 'b'], 'ohe': True}}}
  out = apply_preprocess(pd.DataFrame({'cat': ['a', 'b']}), reg5)
  assert set(out.columns) == {'cat_a', 'cat_b'}


def test_fit_bin_labels_and_apply():
  '''HighLabelBinning fit/apply keeps frequent labels only.'''
  from forecast_ml_framework.pipelines.data_prep.fitted import (
    apply_preprocess, fit_preprocess,
  )
  train = pd.DataFrame({'cat': ['x'] * 5 + ['y'] * 2,
                        'target': [0, 1] * 3 + [0]})
  params = [{'HighLabelBinning': {'args': {
    'val_count': 3, 'cols': {'include_cols': ['cat']}}}}]
  _, reg = fit_preprocess(train, params, {'cat_cols': ['cat']})
  out = apply_preprocess(pd.DataFrame({'cat': ['x', 'y']}), reg)
  assert out['cat'].tolist() == ['x', 'Other']


def test_legacy_typed_data_raises_and_bypass():
  '''Legacy registry + tuple data raises refit; typed bypasses.'''
  from forecast_ml_framework.pipelines.data_prep.fitted import (
    apply_preprocess,
  )
  reg = {'version': 1, 'schema': 'fitted_registry.v1',
         'CategoricalEncoding': {'cat': {
           'categories': [[1, 2], [3, 4]], 'ohe': False,
           'max_cols': None}}}
  with pytest.raises(ValueError, match='legacy|refit'):
    apply_preprocess(pd.DataFrame({'cat': [(1, 2), (3, 4)]}), reg)
  # Top-level typed_categories is recognised by the helper.
  from forecast_ml_framework.pipelines.data_prep.fitted import (
    _registry_has_typed_categories,
  )
  assert _registry_has_typed_categories({'typed_categories': True})
  assert not _registry_has_typed_categories({'typed_categories': False})
  assert not _registry_has_typed_categories('not-a-dict')
  reg_plain = {'version': 1, 'schema': 'fitted_registry.v1',
               'CategoricalEncoding': {'cat': {
                 'categories': ['a', 'b'], 'ohe': False,
                 'max_cols': None}}}
  # __meta__ typed_categories bypasses as well (v2 writer shape).
  reg_meta = copy.deepcopy(reg_plain)
  reg_meta['__meta__'] = {'typed_categories': True}
  out2 = apply_preprocess(pd.DataFrame({'cat': ['a']}), reg_meta)
  assert len(out2) == 1
  # Legacy registry with plain data passes the guard.
  out3 = apply_preprocess(pd.DataFrame({'cat': ['a']}), reg_plain)
  assert len(out3) == 1


def test_typed_category_fit_apply_tuple_and_timestamp():
  '''Tuple/timestamp categories survive fit/apply + JSON.'''
  from forecast_ml_framework.pipelines.data_prep.fitted import (
    apply_preprocess, fit_preprocess, registry_from_json,
    registry_to_json,
  )
  train = pd.DataFrame({'cat': [(1, 2), (3, 4), (1, 2), (3, 4)],
                        'target': [0, 1, 0, 1]})
  params = [{'CategoricalEncoding': {'args': {
    'ohe_flag': False, 'cols': {'include_cols': ['cat']}}}}]
  _, reg = fit_preprocess(train, params, {'cat_cols': ['cat']})
  rt = registry_from_json(registry_to_json(reg))
  out = apply_preprocess(train, rt)
  assert out['cat'].tolist() == [0, 1, 0, 1]
  ts = [pd.Timestamp('2020-01-01'), pd.Timestamp('2020-01-02')]
  train_ts = pd.DataFrame({'cat': ts * 2, 'target': [0, 1, 0, 1]})
  params_ohe = [{'CategoricalEncoding': {'args': {
    'ohe_flag': True, 'cols': {'include_cols': ['cat']}}}}]
  fitted_ts, reg_ts = fit_preprocess(
    train_ts, params_ohe, {'cat_cols': ['cat']})
  dummies = [c for c in fitted_ts.columns if c != 'target']
  assert fitted_ts[dummies].sum(axis=1).tolist() == [1, 1, 1, 1]
  out_ts = apply_preprocess(train_ts, reg_ts)
  assert out_ts[dummies].sum(axis=1).tolist() == [1, 1, 1, 1]


def test_read_registry_empty_and_legacy_no_version():
  '''Empty / version-less legacy registries read as empty steps.'''
  from forecast_ml_framework.pipelines.data_prep.fitted import (
    read_registry,
  )
  assert read_registry({}) == {}
  steps = read_registry({'Normalizations': {
    'f1': {'kind': 'minmax', 'lo': 0.0, 'hi': 1.0}}})
  assert set(steps) == {'Normalizations'}
  # __meta__ non-dict is ignored for version purposes.
  assert read_registry({'version': 1, '__meta__': 'junk'}) == {}


def test_fit_preprocess_dict_config_and_missing_columns():
  '''fit accepts dict config; missing columns are skipped.'''
  from forecast_ml_framework.pipelines.data_prep.fitted import (
    apply_preprocess, fit_preprocess,
  )
  train = pd.DataFrame({'f1': [0.0, 10.0], 'target': [0, 1]})
  # Dict-style prep_params (one step config as mapping).
  cfg = {'Normalizations': {'args': {
    'handle': 'MinMaxScaler', 'cols': {'include_cols': ['nope']}}}}
  fitted, reg = fit_preprocess(train, cfg, {'num_cols': ['f1']})
  assert list(fitted['f1']) == [0.0, 10.0]
  assert _steps_of(reg)['Normalizations'] == {}
  # Apply to frame missing the fitted column is a no-op for it.
  out = apply_preprocess(pd.DataFrame({'other': [1.0]}), _fit_v2())
  assert list(out.columns) == ['other']
