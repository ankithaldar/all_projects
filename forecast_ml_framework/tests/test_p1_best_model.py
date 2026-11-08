#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''P1-A2 best_model v2 envelope + loader + registry tests.'''
# pylint: disable=import-outside-toplevel, too-many-lines, invalid-name

import json
import pickle
import pickletools
import types
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest


def _tiny_classification():
  from sklearn.linear_model import LogisticRegression
  x = np.array([[0.0], [1.0], [2.0], [3.0], [0.5], [2.5], [1.5], [3.5]])
  y = np.array([0, 0, 1, 1, 0, 1, 0, 1])
  est = LogisticRegression(max_iter=1000).fit(x, y)
  return est, x, y


def _tiny_regression():
  from sklearn.linear_model import LinearRegression
  x = np.array([[0.0], [1.0], [2.0], [3.0]])
  y = np.array([0.0, 1.0, 2.0, 3.0])
  est = LinearRegression().fit(x, y)
  return est, x, y


# ---------- envelope.py basics ----------

def test_wrap_envelope_exact_four_keys():
  from forecast_ml_framework.pipelines.modeling.envelope import wrap_envelope
  est, _, _ = _tiny_classification()
  env = wrap_envelope(est, ['0', '1'])
  assert set(env.keys()) == {'model', 'label_set', 'registry_ref', 'onnx_sidecar'}
  assert env['model'] is est
  assert env['label_set'] == ['0', '1']
  assert env['registry_ref'] is None
  assert env['onnx_sidecar'] is None


def test_unwrap_envelope_dict_and_bare():
  from forecast_ml_framework.pipelines.modeling.envelope import unwrap_envelope, wrap_envelope
  est, _, _ = _tiny_classification()
  env = wrap_envelope(est, ['0', '1'])
  assert unwrap_envelope(env) is est
  assert unwrap_envelope(est) is est
  assert unwrap_envelope(None) is None


def test_model_classes_on_estimator_adapter_and_calibrated():
  from forecast_ml_framework.pipelines.modeling.adapters import SklearnAdapter
  from forecast_ml_framework.pipelines.modeling.envelope import model_classes, wrap_envelope
  from forecast_ml_framework.pipelines.modeling.calibration import _wrap, _BetaCalWrapper
  est, x, y = _tiny_classification()
  assert list(model_classes(est)) == [0, 1]
  adapter = SklearnAdapter({})
  adapter.model = est
  assert list(model_classes(adapter)) == [0, 1]
  env = wrap_envelope(est, ['0', '1'])
  assert list(model_classes(env)) == [0, 1]
  # CalibratedClassifierCV
  from forecast_ml_framework.pipelines.modeling.calibration import calibrate_model
  cal_env, meta = calibrate_model(env, x, y, {'task_type': 'classification', 'calibration': {'method': 'sigmoid', 'cv': 2}})
  assert list(model_classes(cal_env)) == [0, 1]
  # _BetaCalWrapper
  wrapper = _BetaCalWrapper(est, SimpleNamespace(predict_proba=lambda p: p))
  assert list(model_classes(wrapper)) == [0, 1]
  wrapped_adapter = _wrap(wrapper)
  assert list(model_classes(wrapped_adapter)) == [0, 1]


def test_model_classes_regression_none():
  from forecast_ml_framework.pipelines.modeling.envelope import model_classes
  est, _, _ = _tiny_regression()
  assert model_classes(est) is None


def test_validate_envelope_ok_classification_and_regression():
  from forecast_ml_framework.pipelines.modeling.envelope import validate_envelope, wrap_envelope
  est, _, _ = _tiny_classification()
  validate_envelope(wrap_envelope(est, ['0', '1']), 'classification')
  rest, _, _ = _tiny_regression()
  validate_envelope(wrap_envelope(rest, None), 'regression')


def test_validate_envelope_rejects_extra_keys_and_missing_model():
  from forecast_ml_framework.pipelines.modeling.envelope import validate_envelope, wrap_envelope
  est, _, _ = _tiny_classification()
  env = wrap_envelope(est, ['0', '1'])
  env['extra'] = 1
  with pytest.raises(ValueError):
    validate_envelope(env, 'classification')
  with pytest.raises(ValueError):
    validate_envelope({'label_set': ['0', '1']}, 'classification')
  with pytest.raises(ValueError):
    validate_envelope(wrap_envelope(est, ['0', '1']) | {'model': None, 'label_set': ['0', '1']}, 'classification')


def test_validate_envelope_rejects_label_set_mismatch():
  from forecast_ml_framework.pipelines.modeling.envelope import validate_envelope, wrap_envelope
  est, _, _ = _tiny_classification()
  with pytest.raises(ValueError):
    validate_envelope(wrap_envelope(est, ['1', '0']), 'classification')
  with pytest.raises(ValueError):
    validate_envelope(wrap_envelope(est, ['0']), 'classification')


def test_validate_envelope_regression_none_bijection():
  from forecast_ml_framework.pipelines.modeling.envelope import validate_envelope, wrap_envelope
  est, _, _ = _tiny_classification()
  with pytest.raises(ValueError):
    validate_envelope(wrap_envelope(est, None), 'classification')
  rest, _, _ = _tiny_regression()
  with pytest.raises(ValueError):
    validate_envelope(wrap_envelope(rest, ['0', '1']), 'regression')


def test_validate_envelope_onnx_sidecar_shape():
  from forecast_ml_framework.pipelines.modeling.envelope import validate_envelope, wrap_envelope
  est, _, _ = _tiny_classification()
  ok = wrap_envelope(est, ['0', '1'], onnx_sidecar={'path': 'm.onnx', 'opset': 14, 'sha256': None})
  validate_envelope(ok, 'classification')
  ok2 = wrap_envelope(est, ['0', '1'], onnx_sidecar={'path': 'm.onnx', 'opset': 14, 'sha256': 'a' * 64})
  validate_envelope(ok2, 'classification')
  with pytest.raises(ValueError):
    validate_envelope(wrap_envelope(est, ['0', '1'], onnx_sidecar={'path': 'm.txt', 'opset': 14, 'sha256': None}), 'classification')
  with pytest.raises(ValueError):
    validate_envelope(wrap_envelope(est, ['0', '1'], onnx_sidecar={'path': 'm.onnx', 'opset': '14', 'sha256': None}), 'classification')
  with pytest.raises(ValueError):
    validate_envelope(wrap_envelope(est, ['0', '1'], onnx_sidecar={'opset': 14}), 'classification')
  with pytest.raises(ValueError):
    validate_envelope(wrap_envelope(est, ['0', '1'], onnx_sidecar='m.onnx'), 'classification')
  with pytest.raises(ValueError):
    validate_envelope(wrap_envelope(est, ['0', '1'], onnx_sidecar={'path': 'm.onnx', 'opset': True, 'sha256': None}), 'classification')


def test_validate_envelope_erasure_optional():
  from forecast_ml_framework.pipelines.modeling.envelope import validate_envelope, wrap_envelope
  est, _, _ = _tiny_classification()
  env = wrap_envelope(est, ['0', '1'])
  env['erasure'] = {'subject_hash': 'a' * 64, 'requested_at': '2026-11-08T00:00:00+00:00', 'action': 'retrain_required'}
  validate_envelope(env, 'classification')
  bad = wrap_envelope(est, ['0', '1'])
  bad['erasure'] = {'subject_hash': 'not-hex', 'requested_at': 'x', 'action': 'delete'}
  with pytest.raises(ValueError):
    validate_envelope(bad, 'classification')
  bad2 = wrap_envelope(est, ['0', '1'])
  bad2['erasure'] = {'subject_hash': 'a' * 64}
  with pytest.raises(ValueError):
    validate_envelope(bad2, 'classification')
  bad3 = wrap_envelope(est, ['0', '1'])
  bad3['erasure'] = {'subject_hash': 'a' * 64, 'requested_at': '2026-11-08T00:00:00+00:00', 'action': 'retrain_required', 'extra': 1}
  with pytest.raises(ValueError):
    validate_envelope(bad3, 'classification')
  # bad requested_at
  bad4 = wrap_envelope(est, ['0', '1'])
  bad4['erasure'] = {'subject_hash': 'a' * 64, 'requested_at': 'not-a-date', 'action': 'retrain_required'}
  with pytest.raises(ValueError):
    validate_envelope(bad4, 'classification')
  # bad hash length / non-hex
  bad5 = wrap_envelope(est, ['0', '1'])
  bad5['erasure'] = {'subject_hash': 'a' * 63, 'requested_at': '2026-11-08T00:00:00+00:00', 'action': 'retrain_required'}
  with pytest.raises(ValueError):
    validate_envelope(bad5, 'classification')


def test_validate_envelope_registry_ref_shape():
  from forecast_ml_framework.pipelines.modeling.envelope import validate_envelope, wrap_envelope
  est, _, _ = _tiny_classification()
  validate_envelope(wrap_envelope(est, ['0', '1'], registry_ref={'name': 'my-model', 'alias': 'production', 'version': '7'}), 'classification')
  validate_envelope(wrap_envelope(est, ['0', '1'], registry_ref={'name': 'my-model', 'alias': 'production', 'version': None}), 'classification')
  with pytest.raises(ValueError):
    validate_envelope(wrap_envelope(est, ['0', '1'], registry_ref={'name': 'my-model'}), 'classification')
  with pytest.raises(ValueError):
    validate_envelope(wrap_envelope(est, ['0', '1'], registry_ref='my-model'), 'classification')
  with pytest.raises(ValueError):
    validate_envelope(wrap_envelope(est, ['0', '1'], registry_ref={'name': 1, 'alias': 'production'}), 'classification')
  with pytest.raises(ValueError):
    validate_envelope(wrap_envelope(est, ['0', '1'], registry_ref={'name': 'my-model', 'alias': 'production', 'version': 7}), 'classification')
  with pytest.raises(ValueError):
    validate_envelope(wrap_envelope(est, ['0', '1'], onnx_sidecar={'path': 'm.onnx', 'opset': 14, 'sha256': 123}), 'classification')
  with pytest.raises(ValueError):
    validate_envelope('not-a-dict', 'classification')
  with pytest.raises(ValueError):
    validate_envelope(wrap_envelope(est, []), 'classification')
  with pytest.raises(ValueError):
    validate_envelope(wrap_envelope(est, ['0', '0']), 'classification')
  with pytest.raises(ValueError):
    validate_envelope(wrap_envelope(est, [0, 1]), 'classification')
  rest, _, _ = _tiny_regression()
  with pytest.raises(ValueError):
    validate_envelope(wrap_envelope(rest, ['0', '1']), 'classification')
  bad_action = wrap_envelope(est, ['0', '1'])
  bad_action['erasure'] = {'subject_hash': 'a' * 64, 'requested_at': '2026-11-08T00:00:00+00:00', 'action': 'delete'}
  with pytest.raises(ValueError):
    validate_envelope(bad_action, 'classification')
  # Z suffix valid
  zenv = wrap_envelope(est, ['0', '1'])
  zenv['erasure'] = {'subject_hash': 'a' * 64, 'requested_at': '2026-11-08T00:00:00Z', 'action': 'retrain_required'}
  validate_envelope(zenv, 'classification')
  bad_iso = wrap_envelope(est, ['0', '1'])
  bad_iso['erasure'] = {'subject_hash': 'a' * 64, 'requested_at': 123, 'action': 'retrain_required'}
  with pytest.raises(ValueError):
    validate_envelope(bad_iso, 'classification')
  # unknown task_type raises
  with pytest.raises(ValueError):
    validate_envelope(wrap_envelope(est, ['0', '1']), 'other')
  with pytest.raises(ValueError):
    validate_envelope(wrap_envelope(est, ['1', '0']), 'other')
  with pytest.raises(ValueError):
    validate_envelope(wrap_envelope(est, 'not-list'), 'other')
  with pytest.raises(ValueError):
    validate_envelope(wrap_envelope(rest, None), 'other')
  # extra keys raise
  with pytest.raises(ValueError):
    validate_envelope(wrap_envelope(est, ['0', '1'], registry_ref={'name': 'my-model', 'alias': 'production', 'version': '7', 'extra': 1}), 'classification')
  # valid exact passes
  validate_envelope(wrap_envelope(est, ['0', '1'], registry_ref={'name': 'my-model', 'alias': 'production', 'version': '7'}), 'classification')


def test_wrap_envelope_is_raw_estimator_not_adapter():
  from forecast_ml_framework.pipelines.modeling.adapters import SklearnAdapter
  from forecast_ml_framework.pipelines.modeling.envelope import unwrap_envelope, wrap_envelope
  est, _, _ = _tiny_classification()
  adapter = SklearnAdapter({})
  adapter.model = est
  env = wrap_envelope(adapter.model, ['0', '1'])
  assert env['model'] is est
  assert unwrap_envelope(env) is est


# ---------- calibrate_model ----------

def test_calibrate_model_returns_envelope_unchanged_label_set():
  from forecast_ml_framework.pipelines.modeling.envelope import unwrap_envelope, wrap_envelope
  from forecast_ml_framework.pipelines.modeling.calibration import calibrate_model
  est, x, y = _tiny_classification()
  env = wrap_envelope(est, ['0', '1'])
  cal_env, meta = calibrate_model(env, x, y, {'task_type': 'classification', 'calibration': {'method': 'sigmoid', 'cv': 2}})
  assert isinstance(cal_env, dict)
  assert set(cal_env.keys()) == {'model', 'label_set', 'registry_ref', 'onnx_sidecar'}
  assert cal_env['label_set'] == ['0', '1']
  assert meta['calibrated'] is True
  proba = unwrap_envelope(cal_env).predict_proba(x)
  assert proba.shape == (len(x), 2)
  assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_calibrate_model_passthrough_returns_envelope_regression_and_none():
  from forecast_ml_framework.pipelines.modeling.envelope import unwrap_envelope, wrap_envelope
  from forecast_ml_framework.pipelines.modeling.calibration import calibrate_model
  rest, x, y = _tiny_regression()
  out, meta = calibrate_model(rest, x, y, {'task_type': 'regression', 'calibration': {'method': 'sigmoid', 'cv': 2}})
  assert isinstance(out, dict)
  assert set(out.keys()) == {'model', 'label_set', 'registry_ref', 'onnx_sidecar'}
  assert out['label_set'] is None
  assert meta['calibrated'] is False
  # method none on classification returns envelope with label_set
  est, xc, yc = _tiny_classification()
  env = wrap_envelope(est, ['0', '1'])
  out2, meta2 = calibrate_model(env, xc, yc, {'task_type': 'classification', 'calibration': {'method': 'none', 'cv': 2}})
  assert isinstance(out2, dict)
  assert out2['label_set'] == ['0', '1']
  assert meta2['calibrated'] is False


def test_calibrate_model_bare_adapter_input_returns_envelope():
  from forecast_ml_framework.pipelines.modeling.adapters import SklearnAdapter
  from forecast_ml_framework.pipelines.modeling.calibration import calibrate_model
  est, x, y = _tiny_classification()
  adapter = SklearnAdapter({})
  adapter.model = est
  out, _ = calibrate_model(adapter, x, y, {'task_type': 'classification', 'calibration': {'method': 'none', 'cv': 2}})
  assert isinstance(out, dict)
  assert out['label_set'] == ['0', '1']
  # model in envelope is raw estimator
  assert out['model'] is est


def test_calibrated_envelope_restricted_roundtrip():
  from forecast_ml_framework.core.artifact_store import restricted_loads, _SAFE_GLOBALS
  from forecast_ml_framework.pipelines.modeling.envelope import unwrap_envelope, wrap_envelope
  from forecast_ml_framework.pipelines.modeling.calibration import calibrate_model
  est, x, y = _tiny_classification()
  env = wrap_envelope(est, ['0', '1'])
  for method in ('sigmoid', 'isotonic'):
    cal_env, _ = calibrate_model(env, x, y, {'task_type': 'classification', 'calibration': {'method': method, 'cv': 2}})
    raw = pickle.dumps(cal_env)
    restored = restricted_loads(raw)
    assert isinstance(restored, dict)
    assert restored['label_set'] == ['0', '1']
    before = unwrap_envelope(cal_env).predict_proba(x)
    after = unwrap_envelope(restored).predict_proba(x)
    assert np.allclose(before, after)
  # pickletools residual subset of the 7 tuples
  def _collect_globals(data):
    out = set()
    stack = []
    memo = {}
    for op, arg, _pos in pickletools.genops(data):
      if op.name in ('SHORT_BINUNICODE', 'BINUNICODE', 'UNICODE', 'SHORT_BINBYTES', 'BINBYTES', 'BINSTRING'):
        stack.append(arg)
      elif op.name == 'GLOBAL' and isinstance(arg, str) and ' ' in arg:
        mod, name = arg.split(None, 1)
        out.add((mod, name))
        stack.append(('global', mod, name))
      elif op.name == 'MEMOIZE':
        if stack:
          memo[len(memo)] = stack[-1]
      elif op.name in ('BINGET', 'GET', 'LONG_BINGET'):
        stack.append(memo.get(arg))
      elif op.name in ('PUT', 'BINPUT', 'LONG_BINPUT'):
        if stack:
          memo[arg] = stack[-1]
      elif op.name == 'STACK_GLOBAL':
        name = stack.pop() if stack else None
        mod = stack.pop() if stack else None
        if isinstance(mod, str) and isinstance(name, str):
          out.add((mod, name))
        stack.append(('global', mod, name))
      elif op.name in ('TUPLE1', 'TUPLE2', 'TUPLE3'):
        for _ in range(int(op.name[-1])):
          if stack:
            stack.pop()
        stack.append(None)
      elif op.name == 'EMPTY_TUPLE':
        stack.append(None)
      elif op.name in ('NEWOBJ', 'REDUCE'):
        # pop args and cls
        for _ in range(2):
          if stack:
            stack.pop()
        stack.append(None)
      elif op.name == 'BUILD':
        if stack:
          stack.pop()
        # keep object
      elif op.name == 'SETITEM':
        for _ in range(2):
          if stack:
            stack.pop()
        # keep dict
      elif op.name == 'MARK':
        stack.append('MARK')
      elif op.name in ('TUPLE', 'DICT', 'LIST', 'FROZENSET'):
        # pop until mark
        while stack and stack[-1] != 'MARK':
          stack.pop()
        if stack:
          stack.pop()
        stack.append(None)
      elif op.name == 'STOP':
        pass
      else:
        # for other ops leave stack as None placeholder where sensible
        # push None for value ops to keep stack roughly balanced
        if op.name in ('EMPTY_DICT', 'EMPTY_LIST', 'EMPTY_SET'):
          stack.append(None)
        elif op.name in ('APPEND', 'SETITEMS', 'APPENDS', 'ADDITEMS', 'POP', 'POP_MARK', 'DUP'):
          pass
    return out
  globals_stack = set()
  for target_env in (
    calibrate_model(env, x, y, {'task_type': 'classification', 'calibration': {'method': 'sigmoid', 'cv': 2}})[0],
    calibrate_model(env, x, y, {'task_type': 'classification', 'calibration': {'method': 'isotonic', 'cv': 2}})[0],
  ):
    globals_stack |= _collect_globals(pickle.dumps(target_env))
  expected_7 = {
    ('sklearn.calibration', 'CalibratedClassifierCV'),
    ('sklearn.calibration', '_CalibratedClassifier'),
    ('sklearn.calibration', '_SigmoidCalibration'),
    ('sklearn.isotonic', 'IsotonicRegression'),
    ('forecast_ml_framework.pipelines.modeling.calibration', '_BetaCalWrapper'),
    ('betacal', 'BetaCalibration'),
    ('betacal.beta_calibration', '_BetaCal'),
  }
  residual = globals_stack - set(_SAFE_GLOBALS)
  assert residual.issubset(expected_7)
  # Post-D1, all 7 are allowlisted, so residual is empty but globals were seen
  assert ('sklearn.calibration', 'CalibratedClassifierCV') in globals_stack
  assert ('sklearn.calibration', '_CalibratedClassifier') in globals_stack


def test_betacal_envelope_restricted_roundtrip():
  pytest.importorskip('betacal')
  from forecast_ml_framework.core.artifact_store import restricted_loads
  from forecast_ml_framework.pipelines.modeling.envelope import unwrap_envelope, wrap_envelope
  from forecast_ml_framework.pipelines.modeling.calibration import calibrate_model
  est, x, y = _tiny_classification()
  env = wrap_envelope(est, ['0', '1'])
  cal_env, meta = calibrate_model(env, x, y, {'task_type': 'classification', 'calibration': {'method': 'betacal', 'cv': 2}})
  assert meta['method'] in ('betacal', 'sigmoid')
  raw = pickle.dumps(cal_env)
  restored = restricted_loads(raw)
  assert np.allclose(unwrap_envelope(cal_env).predict_proba(x), unwrap_envelope(restored).predict_proba(x))
  if cal_env['model'].__class__.__name__ == '_BetaCalWrapper':
    def _collect_globals2(data):
      out = set()
      stack = []
      memo = {}
      for op, arg, _pos in pickletools.genops(data):
        if op.name in ('SHORT_BINUNICODE', 'BINUNICODE', 'UNICODE'):
          stack.append(arg)
        elif op.name == 'GLOBAL' and isinstance(arg, str) and ' ' in arg:
          mod, name = arg.split(None, 1)
          out.add((mod, name))
          stack.append(('global', mod, name))
        elif op.name == 'MEMOIZE':
          if stack:
            memo[len(memo)] = stack[-1]
        elif op.name in ('BINGET', 'GET', 'LONG_BINGET'):
          stack.append(memo.get(arg))
        elif op.name in ('PUT', 'BINPUT', 'LONG_BINPUT'):
          if stack:
            memo[arg] = stack[-1]
        elif op.name == 'STACK_GLOBAL':
          name = stack.pop() if stack else None
          mod = stack.pop() if stack else None
          if isinstance(mod, str) and isinstance(name, str):
            out.add((mod, name))
          stack.append(('global', mod, name))
      return out
    bc_globals = _collect_globals2(pickle.dumps(cal_env))
    assert ('forecast_ml_framework.pipelines.modeling.calibration', '_BetaCalWrapper') in bc_globals


def test_uncalibrated_envelope_restricted_roundtrip():
  from forecast_ml_framework.core.artifact_store import restricted_loads
  from forecast_ml_framework.pipelines.modeling.envelope import unwrap_envelope, wrap_envelope
  est, x, _ = _tiny_classification()
  env = wrap_envelope(est, ['0', '1'])
  restored = restricted_loads(pickle.dumps(env))
  assert restored['label_set'] == ['0', '1']
  assert np.allclose(unwrap_envelope(restored).predict_proba(x), unwrap_envelope(env).predict_proba(x))


# ---------- pipeline _predict_proba shapes (P1-B1 call sites) ----------

def test_pipeline_predict_proba_shapes_binary_multiclass_regression():
  from sklearn.linear_model import LogisticRegression, LinearRegression
  from forecast_ml_framework.pipelines.modeling.pipeline import _predict, _predict_proba
  from forecast_ml_framework.pipelines.modeling.envelope import wrap_envelope
  # binary
  est, x, _ = _tiny_classification()
  env = wrap_envelope(est, ['0', '1'])
  proba = _predict_proba(env, x, 'classification')
  assert proba.shape == (len(x), 2)
  assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)
  pred = _predict(env, x)
  assert list(pred) == list(est.predict(x))
  # multiclass 3-class
  xm = np.array([[0.0], [1.0], [2.0], [3.0], [0.5], [2.5], [1.5], [3.5]])
  ym = np.array([0, 1, 2, 0, 1, 2, 0, 1])
  estm = LogisticRegression(max_iter=1000).fit(xm, ym)
  envm = wrap_envelope(estm, ['0', '1', '2'])
  probam = _predict_proba(envm, xm, 'classification')
  assert probam.shape == (len(xm), 3)
  assert np.allclose(probam.sum(axis=1), 1.0, atol=1e-6)
  # regression -> None
  rest, xr, _ = _tiny_regression()
  envr = wrap_envelope(rest, None)
  assert _predict_proba(envr, xr, 'regression') is None


# ---------- loader ----------

def test_loader_dict_unwrap_and_onnx_rejection(tmp_path):
  from forecast_ml_framework.serving.model_loader import ModelCorruptError, FileModelLoader, load_model
  from forecast_ml_framework.pipelines.modeling.envelope import wrap_envelope
  est, _, _ = _tiny_classification()
  env = wrap_envelope(est, ['0', '1'])
  assert load_model(env) is est
  # .onnx always raises, even with allow_pyfunc True, and pyfunc never called
  p = tmp_path / 'm.onnx'
  p.write_bytes(b'not-onnx')
  with pytest.raises(ModelCorruptError):
    load_model(str(p))
  with pytest.raises(ModelCorruptError):
    load_model(str(p), allow_pyfunc=True)
  with pytest.raises(ModelCorruptError):
    FileModelLoader(str(p)).load()
  with pytest.raises(ModelCorruptError):
    load_model(tmp_path / 'm.onnx')


def test_loader_pickle_via_restricted(tmp_path):
  from forecast_ml_framework.serving.model_loader import load_model, FileModelLoader
  est, x, _ = _tiny_classification()
  p = tmp_path / 'm.pkl'
  p.write_bytes(pickle.dumps(est))
  loaded = load_model(str(p))
  assert np.allclose(loaded.predict_proba(x), est.predict_proba(x))
  assert np.allclose(FileModelLoader(str(p)).load().predict_proba(x), est.predict_proba(x))


def test_loader_pyfunc_never_called_when_allow_false(tmp_path):
  import forecast_ml_framework.serving.model_loader as ml_module
  est, _, _ = _tiny_classification()
  p = tmp_path / 'bad.pkl'
  p.write_bytes(b'not a pickle')
  with mock.patch('sys.modules', {**__import__('sys').modules, 'mlflow.pyfunc': mock.MagicMock()}) as _:
    # Instead patch mlflow.pyfunc via module attribute path
    pass
  # Patch mlflow.pyfunc.load_model via import machinery
  fake_pyfunc = mock.MagicMock()
  fake_mlflow = types.ModuleType('mlflow')
  fake_mlflow.pyfunc = fake_pyfunc
  import sys
  old = sys.modules.get('mlflow')
  sys.modules['mlflow'] = fake_mlflow
  try:
    # Should raise ModelCorruptError, not call pyfunc
    from forecast_ml_framework.serving.model_loader import ModelCorruptError, load_model
    with pytest.raises(ModelCorruptError):
      load_model(str(p), allow_pyfunc=False)
    fake_pyfunc.load_model.assert_not_called()
  finally:
    if old is not None:
      sys.modules['mlflow'] = old
    else:
      sys.modules.pop('mlflow', None)


def _fake_mlflow_loader(monkeypatch, version_obj):
  fake = types.ModuleType('mlflow')
  fake.__path__ = []
  tracking = types.ModuleType('mlflow.tracking')
  pyfunc = types.ModuleType('mlflow.pyfunc')
  pyfunc.load_model = mock.MagicMock(side_effect=AssertionError('pyfunc called'))
  fake.pyfunc = pyfunc
  _src = getattr(version_obj, 'source', None) or getattr(version_obj, 'artifact_uri', None)

  class FakeClient:
    def __init__(self, tracking_uri=None, registry_uri=None):
      self.tracking_uri = tracking_uri
      self.registry_uri = registry_uri
    def get_model_version_by_alias(self, name, alias):
      return version_obj
    def download_artifacts(self, run_id, path, dst_path=None, **kw):
      return _src
  tracking.MlflowClient = FakeClient
  fake.tracking = tracking
  artifacts = types.ModuleType('mlflow.artifacts')
  artifacts.download_artifacts_from_uri = mock.MagicMock()
  fake.artifacts = artifacts
  monkeypatch.setitem(__import__('sys').modules, 'mlflow', fake)
  monkeypatch.setitem(__import__('sys').modules, 'mlflow.tracking', tracking)
  monkeypatch.setitem(__import__('sys').modules, 'mlflow.pyfunc', pyfunc)
  monkeypatch.setitem(__import__('sys').modules, 'mlflow.artifacts', artifacts)
  return fake


def test_registry_loader_resolved_version_and_tags(monkeypatch, tmp_path):
  from forecast_ml_framework.serving.model_loader import RegistryModelLoader
  from forecast_ml_framework.pipelines.modeling.envelope import model_classes
  est, x, _ = _tiny_classification()
  est_path = tmp_path / 'est.pkl'
  est_path.write_bytes(pickle.dumps(est))
  version_obj = SimpleNamespace(version='7', run_id='run-1', source=str(est_path), tags={'label_set': json.dumps(['0', '1']), 'task_type': 'classification'})
  _fake_mlflow_loader(monkeypatch, version_obj)
  loader = RegistryModelLoader(registry_uri='mlruns', model_name='my-model', alias='production', allow_pyfunc=False)
  loaded = loader.load()
  assert loader.resolved_version == '7'
  assert np.allclose(loaded.predict_proba(x), est.predict_proba(x))
  # caller reconstruction via version tags
  label_set = json.loads(version_obj.tags['label_set'])
  assert label_set == ['0', '1']
  task_type = version_obj.tags['task_type']
  assert task_type == 'classification'
  registry_ref = {'name': loader.model_name, 'alias': loader.alias, 'version': loader.resolved_version}
  assert registry_ref['version'] == '7'
  # fallback when tags empty
  version_obj2 = SimpleNamespace(version='7', run_id='run-1', source=str(est_path), tags={})
  _fake_mlflow_loader(monkeypatch, version_obj2)
  loader2 = RegistryModelLoader(registry_uri='mlruns', model_name='my-model', alias='production', allow_pyfunc=False)
  loaded2 = loader2.load()
  fallback_label = [str(c) for c in model_classes(loaded2)]
  assert fallback_label == ['0', '1']


def test_registry_loader_resolved_version_via_load_model_path(tmp_path):
  # dict->unwrap via load_model does not touch registry, but should not set resolved_version attr incorrectly
  from forecast_ml_framework.serving.model_loader import load_model
  est, _, _ = _tiny_classification()
  from forecast_ml_framework.pipelines.modeling.envelope import wrap_envelope
  assert load_model(wrap_envelope(est, ['0', '1'])) is est


# ---------- registry writer ----------

def _fake_mlflow_registry(monkeypatch):
  fake = types.ModuleType('mlflow')
  fake.__path__ = []
  sklearn = types.ModuleType('mlflow.sklearn')
  tracking = types.ModuleType('mlflow.tracking')
  calls = {'log_model': [], 'set_tags': [], 'log_params': [], 'log_metrics': []}

  class FakeRun:
    def __init__(self):
      self.info = SimpleNamespace(run_id='run-123')
    def __enter__(self):
      return self
    def __exit__(self, *a):
      return False
  fake.start_run = lambda **kw: FakeRun()
  fake.log_params = lambda p: calls['log_params'].append(p)
  fake.log_metrics = lambda m: calls['log_metrics'].append(m)
  fake.set_tags = lambda t: calls['set_tags'].append(t)
  fake.set_tracking_uri = lambda u: None
  fake.set_registry_uri = lambda u: None
  fake.set_experiment = lambda n: None

  def _log_model(sk_model=None, name=None, registered_model_name=None, tags=None, **kw):
    calls['log_model'].append({'sk_model': sk_model, 'name': name, 'registered_model_name': registered_model_name, 'tags': tags})
    return SimpleNamespace(model_uri='models:/%s/1' % registered_model_name, run_id='run-123', version='1')
  sklearn.log_model = _log_model
  fake.sklearn = sklearn

  class FakeClient:
    def __init__(self, tracking_uri=None, registry_uri=None):
      pass
    def set_registered_model_alias(self, name, alias, version):
      pass
    def search_model_versions(self, filter_string=None, **kw):
      return [SimpleNamespace(name='my-model', version='1', run_id='run-123')]
  tracking.MlflowClient = FakeClient
  monkeypatch.setitem(__import__('sys').modules, 'mlflow', fake)
  monkeypatch.setitem(__import__('sys').modules, 'mlflow.sklearn', sklearn)
  monkeypatch.setitem(__import__('sys').modules, 'mlflow.tracking', tracking)
  return fake, sklearn, calls


def test_registry_log_model_unwrapped_with_tags(monkeypatch):
  _, _, calls = _fake_mlflow_registry(monkeypatch)
  from forecast_ml_framework.registry.model_registry import ModelRegistry
  from forecast_ml_framework.pipelines.modeling.envelope import wrap_envelope
  est, _, _ = _tiny_classification()
  env = wrap_envelope(est, ['0', '1'])
  registry = ModelRegistry({'tracking_uri': 'mlruns'})
  registry.register_model(env, 'my-model', tags={'env': 'test'})
  logged = calls['log_model'][0]
  assert logged['sk_model'] is est  # unwrapped, not dict, not adapter
  tags = logged['tags']
  assert tags is not None
  assert json.loads(tags['label_set']) == ['0', '1']
  assert tags['task_type'] == 'classification'
  assert tags['env'] == 'test'
  # version tags must NOT go via run-level set_tags
  for t in calls['set_tags']:
    assert 'label_set' not in t
    assert 'task_type' not in t


def test_registry_log_model_bare_estimator_tags_derived(monkeypatch):
  _, _, calls = _fake_mlflow_registry(monkeypatch)
  from forecast_ml_framework.registry.model_registry import ModelRegistry
  est, _, _ = _tiny_classification()
  registry = ModelRegistry({'tracking_uri': 'mlruns'})
  registry.register_model(est, 'my-model', params={'task_type': 'classification'})
  logged = calls['log_model'][0]
  assert logged['sk_model'] is est
  assert json.loads(logged['tags']['label_set']) == ['0', '1']
  assert logged['tags']['task_type'] == 'classification'
