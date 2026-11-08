#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Envelope, authenticate and lineage validation gaps.'''
# pylint: disable=import-outside-toplevel, invalid-name

import numpy as np
import pytest


def _tiny_classification():
  '''Return a fitted classifier.'''
  from sklearn.linear_model import LogisticRegression
  x = np.array([[0.0], [1.0], [2.0], [3.0], [0.5], [2.5],
                [1.5], [3.5]])
  y = np.array([0, 0, 1, 1, 0, 1, 0, 1])
  return LogisticRegression(max_iter=1000).fit(x, y)


def _tiny_regression():
  '''Return a fitted regressor without classes_.'''
  from sklearn.linear_model import LinearRegression
  x = np.array([[0.0], [1.0], [2.0], [3.0]])
  y = np.array([0.0, 1.0, 2.0, 3.0])
  return LinearRegression().fit(x, y)


def _envelope(model, label_set):
  '''Return a raw envelope dict.'''
  return {
    'model': model,
    'label_set': label_set,
    'registry_ref': None,
    'onnx_sidecar': None,
  }


@pytest.mark.parametrize('task_type', ['other', 'bogus', 'cluster'])
def test_validate_envelope_unknown_task_type_raises(task_type):
  '''validate_envelope rejects an unknown task_type.'''
  from forecast_ml_framework.pipelines.modeling.envelope import (
    validate_envelope,
  )
  env = _envelope(_tiny_regression(), None)
  with pytest.raises(ValueError):
    validate_envelope(env, task_type)


def test_wrap_envelope_unknown_task_type_dummy_labels_raises():
  '''wrap rejects unknown task_type with dummy labels.'''
  from forecast_ml_framework.pipelines.modeling.envelope import (
    wrap_envelope,
  )
  with pytest.raises(ValueError):
    wrap_envelope(
      _tiny_regression(), ['dummy'], task_type='unknown')


@pytest.mark.parametrize(
  'sha256', ['not-hex', 'a' * 63, 'a' * 65, 'g' * 64, 123])
def test_envelope_onnx_sha256_must_be_64_hex(sha256):
  '''onnx_sidecar sha256 must be 64-hex when not None.'''
  from forecast_ml_framework.pipelines.modeling.envelope import (
    wrap_envelope,
  )
  sidecar = {'path': 'm.onnx', 'opset': 14, 'sha256': sha256}
  with pytest.raises(ValueError):
    wrap_envelope(
      _tiny_classification(), ['0', '1'], onnx_sidecar=sidecar)


def test_envelope_onnx_sha256_valid_64_hex_passes():
  '''onnx_sidecar sha256 accepts a 64-hex digest.'''
  from forecast_ml_framework.pipelines.modeling.envelope import (
    wrap_envelope,
  )
  sidecar = {'path': 'm.onnx', 'opset': 14, 'sha256': 'a' * 64}
  env = wrap_envelope(
    _tiny_classification(), ['0', '1'], onnx_sidecar=sidecar)
  assert env['onnx_sidecar']['sha256'] == 'a' * 64


@pytest.mark.parametrize('ref', [
  {'name': 'm', 'alias': 'production', 'version': '1', 'x': 2},
  {'name': 'm', 'alias': 'production'},
  {'name': 'm', 'version': '1'},
  {'alias': 'production', 'version': '1'},
])
def test_envelope_registry_ref_exact_keys(ref):
  '''registry_ref must have exactly name/alias/version.'''
  from forecast_ml_framework.pipelines.modeling.envelope import (
    wrap_envelope,
  )
  with pytest.raises(ValueError):
    wrap_envelope(
      _tiny_classification(), ['0', '1'], registry_ref=ref)


def test_envelope_registry_ref_exact_keys_pass():
  '''registry_ref with exactly name/alias/version passes.'''
  from forecast_ml_framework.pipelines.modeling.envelope import (
    wrap_envelope,
  )
  ref = {'name': 'm', 'alias': 'production', 'version': '7'}
  env = wrap_envelope(
    _tiny_classification(), ['0', '1'], registry_ref=ref)
  assert env['registry_ref'] == ref
  ref_none = {'name': 'm', 'alias': 'production', 'version': None}
  env = wrap_envelope(
    _tiny_classification(), ['0', '1'], registry_ref=ref_none)
  assert env['registry_ref'] == ref_none


class _FakeVerifier:
  '''Verifier stub returning fixed claims.'''

  def __init__(self, claims):
    '''Initialise with claims to return.'''
    self._claims = claims

  def verify(self, token):
    '''Return the fixed claims.'''
    return dict(self._claims)


@pytest.mark.parametrize(
  'claims',
  [
    {'scope': 'predict:read'},
    {'scope': 'predict:read', 'sub': ''},
    {'scope': 'predict:read', 'sub': '   '},
    {'scope': 'predict:read', 'sub': None},
    {'scope': 'predict:read', 'sub': 123},
  ],
)
def test_authenticate_missing_sub_raises_independently(claims):
  '''authenticate rejects claims without a usable sub.'''
  from forecast_ml_framework.serving.auth import authenticate
  with pytest.raises(ValueError, match='missing sub'):
    authenticate(
      {'authorization': 'Bearer token'},
      verifier=_FakeVerifier(claims),
      revocation=None,
      shared_key=None,
    )


def test_authenticate_with_sub_passes():
  '''authenticate accepts claims with a usable sub.'''
  from forecast_ml_framework.serving.auth import authenticate
  claims = {'scope': 'predict:read', 'sub': 'user-1'}
  result = authenticate(
    {'authorization': 'Bearer token'},
    verifier=_FakeVerifier(claims),
    revocation=None,
    shared_key=None,
  )
  assert result['auth_mode'] == 'oidc'
  assert result['principal_hash']


@pytest.mark.parametrize('run_info', [
  {'inputs': 'raw'},
  {'inputs': ('raw',)},
  {'inputs': {'raw': 1}},
  {'inputs': 123},
  {'outputs': 'pred'},
  {'outputs': ('pred',)},
  {'features': 'f1'},
  {'features': ('f1',)},
  {'features': {'f1': 1}},
])
def test_build_lineage_non_list_fields_raise(run_info):
  '''build_lineage rejects non-list inputs/outputs/features.'''
  from forecast_ml_framework.governance.lineage import build_lineage
  with pytest.raises(ValueError):
    build_lineage(run_info)


def test_build_lineage_list_fields_pass():
  '''build_lineage accepts list inputs/outputs/features.'''
  from forecast_ml_framework.governance.lineage import build_lineage
  out = build_lineage({
    'inputs': ['raw'],
    'outputs': ['pred'],
    'features': ['f1'],
  })
  assert out['inputs'] == ['raw']
  assert out['outputs'] == ['pred']
  assert out['features'] == ['f1']
  assert build_lineage({})['inputs'] == []
  assert build_lineage({})['outputs'] == []
  assert build_lineage({})['features'] == []
