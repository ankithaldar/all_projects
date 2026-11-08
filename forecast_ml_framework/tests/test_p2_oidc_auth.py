#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''OIDC JWT and RBAC tests for P2-AU1.'''

import base64
import hashlib
import json
import logging
import time

import pytest

try:
  from forecast_ml_framework.serving.auth import (  # noqa: F401
    ALLOWED_SCOPES as _PRELOAD_SCOPES,
  )
except ImportError:
  from forecast_ml_framework.serving.auth import (  # noqa: F401
    ALLOWED_SCOPES as _PRELOAD_SCOPES,
  )

_ISSUER = 'https://login.microsoftonline.com/test-tenant/v2.0'
_AUDIENCE = 'api://forecast-ml-api'
_WORKLOAD_AUD = 'api://AzureADTokenExchange'
_JWKS_URL = 'https://example.invalid/jwks'
_KID = 'test-kid-1'
_SUB = 'user-test-subject-001'
_JTI = 'jti-test-0001'
_N_TEXT = (
  '13154795294850887407863477426699701512057757536210801852'
  '1234763974647973076954887419083128706355839209597447931'
  '3698082220601528365235128732620739256397331094028586563'
  '8431376820280815490943413956087509233102207817246921268'
  '6064353834945442051736721758142668781280907030275218816'
  '243305983696491711926674900577501'
)
_D_TEXT = (
  '76314954470029256640824177451382066845970358961614'
  '76515826702055809872188818262690806018819011625504'
  '29359746456304089079867404801047341111345354632165'
  '44119238076927878101046809744936633021034504985123'
  '82351974648728236322642055991589080719547541118321'
  '59709021183798265493275809991257965664093313329797'
  '5608473'
)
_TEST_N = int(_N_TEXT)
_TEST_D = int(_D_TEXT)
_TEST_E = 65537
_DIGEST_PREFIX = bytes.fromhex(
  '3031300d060960864801650304020105000420'
)


class DummyModel:
  '''Dummy model returning deterministic predictions.'''

  def predict(self, rows):
    '''Predict sum of each row.'''
    return [sum(row) for row in rows]


def dummy_loader():
  '''Return dummy model.'''
  return DummyModel()


class FakeClock:
  '''Mutable clock returning seconds.'''

  def __init__(self, now=1700000000.0):
    '''Initialise clock at a fixed epoch.'''
    self.now = now

  def __call__(self):
    '''Return current fake time.'''
    return self.now

  def advance(self, seconds):
    '''Advance fake time by seconds.'''
    self.now += seconds


class FakeFetcher:
  '''Fake JWKS fetcher recording call arguments.'''

  def __init__(self, jwks):
    '''Initialise with a JWKS dict to return.'''
    self.jwks = jwks
    self.calls = []

  def __call__(self, url, *, timeout, max_bytes):
    '''Record call and return JWKS.'''
    self.calls.append(
      {'url': url, 'timeout': timeout, 'max_bytes': max_bytes}
    )
    return self.jwks


def _b64u(data):
  '''Return base64url without padding.'''
  encoded = base64.urlsafe_b64encode(data)
  return encoded.rstrip(b'=').decode('ascii')


def _int_bytes(value):
  '''Return big-endian bytes for an integer.'''
  size = (value.bit_length() + 7) // 8
  return value.to_bytes(size, 'big')


def _make_jwks():
  '''Build a JWKS dict for the embedded test key.'''
  return {
    'keys': [
      {
        'kty': 'RSA',
        'kid': _KID,
        'n': _b64u(_int_bytes(_TEST_N)),
        'e': _b64u(_int_bytes(_TEST_E)),
        'alg': 'RS256',
        'use': 'sig',
      }
    ]
  }


def _sign(signing_input):
  '''Return RS256 signature bytes for signing input.'''
  digest = hashlib.sha256(signing_input).digest()
  prefix = _DIGEST_PREFIX
  tail = prefix + digest
  size = (_TEST_N.bit_length() + 7) // 8
  ps_len = size - len(tail) - 3
  memory = b'\x00\x01' + b'\xff' * ps_len + b'\x00' + tail
  number = int.from_bytes(memory, 'big')
  signed = pow(number, _TEST_D, _TEST_N)
  return signed.to_bytes(size, 'big')


def _make_token(claims, header=None, alg='RS256', kid=_KID):
  '''Build a JWT string for given claims.'''
  base_header = {'alg': alg, 'typ': 'JWT', 'kid': kid}
  if header is not None:
    base_header.update(header)
  head = _b64u(
    json.dumps(base_header, separators=(',', ':'))
    .encode('utf-8')
  )
  payload = _b64u(
    json.dumps(claims, separators=(',', ':'))
    .encode('utf-8')
  )
  signing_input = (head + '.' + payload).encode('ascii')
  if alg == 'none':
    sig = ''
  else:
    sig = _b64u(_sign(signing_input))
  return head + '.' + payload + '.' + sig


def _claims(**overrides):
  '''Return baseline valid claims.'''
  found = {
    'iss': _ISSUER,
    'aud': _AUDIENCE,
    'exp': 1700000000.0 + 600.0,
    'sub': _SUB,
    'jti': _JTI,
    'scope': 'predict:read',
  }
  found.update(overrides)
  return found


def _verifier(clock=None, fetcher=None, ttl=300):
  '''Build a verifier with fake clock and fetcher.'''
  from forecast_ml_framework.serving.auth import OIDCVerifier
  fake_clock = clock if clock is not None else FakeClock()
  fake_fetcher = fetcher if fetcher is not None else FakeFetcher(
    _make_jwks()
  )
  found = OIDCVerifier(
    issuer=_ISSUER,
    audience=_AUDIENCE,
    jwks_url=_JWKS_URL,
    clock=fake_clock,
    fetcher=fake_fetcher,
    logger=None,
    cache_ttl_s=ttl,
  )
  return found, fake_clock, fake_fetcher


def _clean_oidc_env(monkeypatch):
  '''Delete ambient OIDC and key env vars.'''
  for name in (
    'FORECAST_OIDC_ISSUER',
    'FORECAST_OIDC_AUDIENCE',
    'FORECAST_OIDC_JWKS_URL',
    'FORECAST_OIDC_REVOKED_HASHES',
    'FORECAST_OIDC_REVOKED_FILE',
    'FORECAST_API_KEY',
    'FORECAST_REQUIRE_AUTH',
    'FORECAST_ENV',
    'ENVIRONMENT',
    'APP_ENV',
  ):
    monkeypatch.delenv(name, raising=False)


def _app_client(verifier, revocation=None, **create_kwargs):
  '''Return a TestClient with a dummy model.'''
  from fastapi.testclient import TestClient
  from forecast_ml_framework.serving.app import create_app
  kwargs = dict(create_kwargs)
  kwargs['oidc_verifier'] = verifier
  if revocation is not None:
    kwargs['oidc_revocation'] = revocation
  app = create_app(dummy_loader, **kwargs)
  return TestClient(app), app


def _post_predict(client, token):
  '''POST one predict request with Bearer token.'''
  return client.post(
    '/predict',
    json={'features': {'a': 1.0}},
    headers={'Authorization': 'Bearer ' + token},
  )


def test_allowed_scopes_minimal_rbac_vocabulary():
  '''Allowed scopes are exactly predict read and admin.'''
  from forecast_ml_framework.serving.auth import ALLOWED_SCOPES
  assert ALLOWED_SCOPES == ('predict:read', 'predict:admin')


def test_verify_valid_rs256_roundtrip():
  '''Valid RS256 token verifies and returns claims.'''
  verifier, _, _ = _verifier()
  token = _make_token(_claims())
  claims = verifier.verify(token)
  assert claims['sub'] == _SUB
  assert claims['jti'] == _JTI
  assert claims['iss'] == _ISSUER
  assert claims['aud'] == _AUDIENCE


def test_verify_expired_raises():
  '''Expired token raises ValueError.'''
  verifier, _, _ = _verifier()
  token = _make_token(_claims(exp=1700000000.0 - 10.0))
  with pytest.raises(ValueError):
    verifier.verify(token)


def test_verify_wrong_audience_raises():
  '''Wrong audience token raises ValueError.'''
  verifier, _, _ = _verifier()
  token = _make_token(_claims(aud='api://other-api'))
  with pytest.raises(ValueError):
    verifier.verify(token)


def test_verify_aud_list_allowed():
  '''Audience list containing ours verifies.'''
  verifier, _, _ = _verifier()
  claims = _claims(aud=['api://other', _AUDIENCE])
  token = _make_token(claims)
  assert verifier.verify(token)['aud'] == claims['aud']


def test_verify_wrong_issuer_raises():
  '''Wrong issuer token raises ValueError.'''
  verifier, _, _ = _verifier()
  token = _make_token(_claims(iss='https://evil.example/v2.0'))
  with pytest.raises(ValueError):
    verifier.verify(token)


def test_verify_alg_none_raises():
  '''Alg none token raises ValueError.'''
  verifier, _, _ = _verifier()
  token = _make_token(_claims(), alg='none')
  with pytest.raises(ValueError):
    verifier.verify(token)


def test_verify_alg_hs256_raises():
  '''Alg HS256 confusion token raises ValueError.'''
  verifier, _, _ = _verifier()
  token = _make_token(_claims(), alg='HS256')
  with pytest.raises(ValueError):
    verifier.verify(token)


def test_verify_unknown_kid_raises():
  '''Unknown kid token raises ValueError.'''
  verifier, _, _ = _verifier()
  token = _make_token(_claims(), kid='unknown-kid-999')
  with pytest.raises(ValueError):
    verifier.verify(token)


def test_verify_malformed_token_raises():
  '''Malformed token raises ValueError.'''
  verifier, _, _ = _verifier()
  with pytest.raises(ValueError):
    verifier.verify('not-a-jwt')
  with pytest.raises(ValueError):
    verifier.verify('a.b')
  with pytest.raises(ValueError):
    verifier.verify('')


def test_verify_bad_signature_raises():
  '''Tampered token raises ValueError.'''
  verifier, _, _ = _verifier()
  token = _make_token(_claims())
  heads, body, _ = token.split('.')
  fake = _b64u(b'\x00' * 32)
  tampered = heads + '.' + body + '.' + fake
  with pytest.raises(ValueError):
    verifier.verify(tampered)


def test_verify_missing_exp_raises():
  '''Token without exp raises ValueError.'''
  verifier, _, _ = _verifier()
  claims = _claims()
  del claims['exp']
  token = _make_token(claims)
  with pytest.raises(ValueError):
    verifier.verify(token)


def test_fetcher_called_with_timeout_and_max_bytes():
  '''Fetcher receives timeout 5 and max_bytes 262144.'''
  verifier, _, fetcher = _verifier()
  verifier.verify(_make_token(_claims()))
  assert fetcher.calls
  call = fetcher.calls[0]
  assert call['url'] == _JWKS_URL
  assert call['timeout'] == 5
  assert call['max_bytes'] == 262144


def test_jwks_cache_ttl_no_refetch():
  '''Second verify within TTL does not refetch JWKS.'''
  verifier, clock, fetcher = _verifier(ttl=300)
  token = _make_token(_claims())
  verifier.verify(token)
  verifier.verify(token)
  assert len(fetcher.calls) == 1
  clock.advance(301.0)
  verifier.verify(token)
  assert len(fetcher.calls) == 2


def test_verify_oversized_jwks_raises():
  '''Oversized JWKS response raises ValueError.'''
  big = {'keys': _make_jwks()['keys']}
  big['pad'] = 'x' * 300000
  verifier, _, fetcher = _verifier()
  fetcher.jwks = big
  with pytest.raises(ValueError):
    verifier.verify(_make_token(_claims()))


def test_require_scope_string_and_list():
  '''Require scope accepts string and list forms.'''
  from forecast_ml_framework.serving.auth import require_scope
  require_scope({'scope': 'predict:read'}, 'predict:read')
  require_scope({'scope': ['predict:read']}, 'predict:read')
  require_scope({'roles': 'predict:admin'}, 'predict:read')
  require_scope({'scope': 'predict:read predict:admin'},
                'predict:admin')
  require_scope({'roles': ['predict:read']}, 'predict:read')


def test_require_scope_admin_superset():
  '''Predict admin satisfies predict read.'''
  from forecast_ml_framework.serving.auth import require_scope
  require_scope({'scope': 'predict:admin'}, 'predict:read')
  require_scope({'roles': ['predict:admin']}, 'predict:read')
  require_scope({'scope': 'predict:admin'}, 'predict:admin')


def test_require_scope_missing_raises():
  '''Missing scope raises ValueError.'''
  from forecast_ml_framework.serving.auth import require_scope
  with pytest.raises(ValueError):
    require_scope({'scope': 'other:scope'}, 'predict:read')
  with pytest.raises(ValueError):
    require_scope({}, 'predict:read')
  with pytest.raises(ValueError):
    require_scope({'scope': ''}, 'predict:read')


def test_revocation_list_hashes_jti_and_sub():
  '''Revocation matches by hashed jti or sub.'''
  from forecast_ml_framework.governance.scrub import subject_hash
  from forecast_ml_framework.serving.auth import RevocationList
  found = RevocationList()
  assert found.is_revoked(jti=_JTI, sub=_SUB) is False
  found.revoke(jti=_JTI, sub=None)
  assert found.is_revoked(jti=_JTI, sub=_SUB) is True
  assert found.is_revoked(jti='other-jti', sub='other-sub') is False
  second = RevocationList(
    denied_hashes={subject_hash(_SUB)}
  )
  assert second.is_revoked(jti=None, sub=_SUB) is True
  assert second.is_revoked(jti=None, sub=None) is False


def test_load_revocation_from_env_and_file_merges(tmp_path):
  '''Loader merges env hashes and file hashes.'''
  from forecast_ml_framework.governance.scrub import subject_hash
  from forecast_ml_framework.serving.auth import (
    RevocationList,
    load_revocation_list,
  )
  target = tmp_path / 'revoked.txt'
  target.write_text(subject_hash('file-sub') + '\n\n', 'utf-8')
  note = load_revocation_list(
    revoked_hashes=subject_hash('env-sub') + ', ',
    revoked_file=str(target),
  )
  assert isinstance(note, RevocationList)
  assert note.is_revoked(jti=None, sub='file-sub') is True
  assert note.is_revoked(jti=None, sub='env-sub') is True


def test_revocation_file_reload_denies(tmp_path):
  '''New instance from file still denies revoked hash.'''
  from forecast_ml_framework.governance.scrub import subject_hash
  from forecast_ml_framework.serving.auth import load_revocation_list
  target = tmp_path / 'revoked.txt'
  target.write_text(subject_hash(_SUB) + '\n', 'utf-8')
  first = load_revocation_list(revoked_file=str(target))
  assert first.is_revoked(jti=None, sub=_SUB) is True
  second = load_revocation_list(revoked_file=str(target))
  assert second.is_revoked(jti=None, sub=_SUB) is True


def test_load_revocation_missing_file_empty(tmp_path):
  '''Missing revocation file yields empty set.'''
  from forecast_ml_framework.serving.auth import load_revocation_list
  gone = tmp_path / 'missing.txt'
  note = load_revocation_list(revoked_file=str(gone))
  assert note.is_revoked(jti=_JTI, sub=_SUB) is False


def test_authenticate_oidc_success_hashed_return():
  '''Authenticate OIDC returns hashed principals only.'''
  from forecast_ml_framework.governance.scrub import subject_hash
  from forecast_ml_framework.serving.auth import (
    RevocationList,
    authenticate,
  )
  verifier, _, _ = _verifier()
  token = _make_token(_claims())
  result = authenticate(
    {'authorization': 'Bearer ' + token},
    verifier=verifier,
    revocation=RevocationList(),
    shared_key=None,
  )
  assert result['auth_mode'] == 'oidc'
  assert result['principal_hash'] == subject_hash(_SUB)
  assert result['sub_hash'] == subject_hash(_SUB)
  assert result['jti_hash'] == subject_hash(_JTI)
  blob = json.dumps(result)
  assert _SUB not in blob
  assert _JTI not in blob
  assert token not in blob


def test_authenticate_oidc_revoked_raises():
  '''Revoked OIDC token raises ValueError.'''
  from forecast_ml_framework.serving.auth import (
    RevocationList,
    authenticate,
  )
  verifier, _, _ = _verifier()
  revocation = RevocationList()
  revocation.revoke(jti=_JTI, sub=None)
  token = _make_token(_claims())
  with pytest.raises(ValueError):
    authenticate(
      {'authorization': 'Bearer ' + token},
      verifier=verifier,
      revocation=revocation,
      shared_key=None,
    )


def test_authenticate_oidc_missing_scope_raises():
  '''OIDC without predict scope raises ValueError.'''
  from forecast_ml_framework.serving.auth import (
    RevocationList,
    authenticate,
  )
  verifier, _, _ = _verifier()
  token = _make_token(_claims(scope='other:scope'))
  with pytest.raises(ValueError):
    authenticate(
      {'authorization': 'Bearer ' + token},
      verifier=verifier,
      revocation=RevocationList(),
      shared_key=None,
    )


def test_authenticate_oidc_ignores_shared_key():
  '''Shared key is not consulted when verifier set.'''
  from forecast_ml_framework.serving.auth import (
    RevocationList,
    authenticate,
  )
  verifier, _, _ = _verifier()
  token = _make_token(_claims())
  result = authenticate(
    {'authorization': 'Bearer ' + token},
    verifier=verifier,
    revocation=RevocationList(),
    shared_key='completely-wrong-shared-key',
  )
  assert result['auth_mode'] == 'oidc'
  good = authenticate(
    {'authorization': 'Bearer ' + token,
     'x-api-key': 'completely-wrong-shared-key'},
    verifier=verifier,
    revocation=RevocationList(),
    shared_key='completely-wrong-shared-key',
  )
  assert good['auth_mode'] == 'oidc'


def test_authenticate_oidc_missing_bearer_raises():
  '''Missing Bearer header raises ValueError.'''
  from forecast_ml_framework.serving.auth import (
    RevocationList,
    authenticate,
  )
  verifier, _, _ = _verifier()
  with pytest.raises(ValueError):
    authenticate(
      {'x-api-key': 'whatever-key-value-here-123456'},
      verifier=verifier,
      revocation=RevocationList(),
      shared_key='whatever-key-value-here-123456',
    )


def test_authenticate_shared_key_paths():
  '''Shared key fallback works only without verifier.'''
  from forecast_ml_framework.serving.auth import authenticate
  good = authenticate(
    {'x-api-key': 'shared-secret-1234567890'},
    verifier=None,
    revocation=None,
    shared_key='shared-secret-1234567890',
  )
  assert good['auth_mode'] == 'shared_key'
  bearer = authenticate(
    {'authorization': 'Bearer shared-secret-1234567890'},
    verifier=None,
    revocation=None,
    shared_key='shared-secret-1234567890',
  )
  assert bearer['auth_mode'] == 'shared_key'
  with pytest.raises(ValueError):
    authenticate(
      {'x-api-key': 'wrong'},
      verifier=None,
      revocation=None,
      shared_key='shared-secret-1234567890',
    )
  with pytest.raises(ValueError):
    authenticate(
      {},
      verifier=None,
      revocation=None,
      shared_key='shared-secret-1234567890',
    )


def test_authenticate_shared_non_ascii_not_500():
  '''Non-ASCII shared key fails as ValueError only.'''
  from forecast_ml_framework.serving.auth import authenticate
  with pytest.raises(ValueError):
    authenticate(
      {'x-api-key': 'café-non-ascii-key-value-123456'},
      verifier=None,
      revocation=None,
      shared_key='shared-secret-1234567890',
    )


def test_authenticate_dev_mode():
  '''Both None returns dev mode scopes.'''
  from forecast_ml_framework.serving.auth import authenticate
  result = authenticate(
    {}, verifier=None, revocation=None, shared_key=None
  )
  assert result == {
    'auth_mode': 'dev',
    'scopes': ['predict:read'],
  }


def test_default_fetcher_uses_urlopen(monkeypatch):
  '''Default fetcher reads JWKS via urlopen timeout.'''
  import io
  import urllib.request as _urlreq
  from forecast_ml_framework.serving.auth import (
    default_fetcher,
  )
  payload = json.dumps(_make_jwks()).encode('utf-8')
  seen = {}

  class FakeResponse:
    '''Fake urlopen response.'''

    def __init__(self, body):
      '''Store body bytes.'''
      self.body = body

    def read(self, size=-1):
      '''Return body up to size.'''
      if size is None or size < 0:
        return self.body
      return self.body[:size]

    def __enter__(self):
      '''Enter context.'''
      return self

    def __exit__(self, *args):
      '''Exit context.'''
      return False

  def fake_urlopen(url, *, timeout):
    '''Record call and return response.'''
    seen['url'] = url
    seen['timeout'] = timeout
    return FakeResponse(payload)

  monkeypatch.setattr(
    _urlreq, 'urlopen', fake_urlopen
  )
  got = default_fetcher(
    _JWKS_URL, timeout=5, max_bytes=262144
  )
  assert got['keys'][0]['kid'] == _KID
  assert seen['timeout'] == 5
  assert seen['url'] == _JWKS_URL
  with pytest.raises(ValueError):
    default_fetcher(
      _JWKS_URL, timeout=5, max_bytes=10
    )
  big_stream = io.BytesIO(b'x' * 20)
  del big_stream


def test_verifier_logs_hash_only_no_clear(caplog):
  '''Verify never logs clear token jti sub.'''
  verifier, _, _ = _verifier()
  token = _make_token(_claims())
  with caplog.at_level(logging.INFO):
    verifier.verify(token)
  from forecast_ml_framework.governance.scrub import subject_hash
  assert token not in caplog.text
  assert _JTI not in caplog.text
  assert _SUB not in caplog.text
  assert subject_hash(_SUB)[:16] in caplog.text


def test_api_audience_differs_from_workload():
  '''API audience differs from token exchange audience.'''
  assert _AUDIENCE != _WORKLOAD_AUD
  assert _AUDIENCE == 'api://forecast-ml-api'
  assert _WORKLOAD_AUD == 'api://AzureADTokenExchange'


def test_oidc_roundtrip_predict_200(monkeypatch):
  '''Valid OIDC token returns 200 on predict.'''
  _clean_oidc_env(monkeypatch)
  verifier, _, _ = _verifier()
  client, _ = _app_client(verifier)
  assert client.get('/health').status_code == 200
  token = _make_token(_claims())
  result = _post_predict(client, token)
  assert result.status_code == 200
  assert result.json() == {'predictions': [1.0]}


def test_oidc_expired_predict_401(monkeypatch):
  '''Expired OIDC token returns generic 401.'''
  _clean_oidc_env(monkeypatch)
  verifier, _, _ = _verifier()
  client, _ = _app_client(verifier)
  token = _make_token(_claims(exp=1700000000.0 - 5.0))
  result = _post_predict(client, token)
  assert result.status_code == 401
  assert result.json() == {'detail': 'unauthorized'}


def test_oidc_wrong_aud_predict_401(monkeypatch):
  '''Wrong audience OIDC token returns 401.'''
  _clean_oidc_env(monkeypatch)
  verifier, _, _ = _verifier()
  client, _ = _app_client(verifier)
  token = _make_token(_claims(aud='api://wrong-api'))
  result = _post_predict(client, token)
  assert result.status_code == 401
  assert result.json() == {'detail': 'unauthorized'}


def test_oidc_wrong_iss_predict_401(monkeypatch):
  '''Wrong issuer OIDC token returns 401.'''
  _clean_oidc_env(monkeypatch)
  verifier, _, _ = _verifier()
  client, _ = _app_client(verifier)
  token = _make_token(_claims(iss='https://bad.example/v2.0'))
  result = _post_predict(client, token)
  assert result.status_code == 401
  assert result.json() == {'detail': 'unauthorized'}


def test_oidc_alg_none_predict_401(monkeypatch):
  '''Alg none token returns 401.'''
  _clean_oidc_env(monkeypatch)
  verifier, _, _ = _verifier()
  client, _ = _app_client(verifier)
  token = _make_token(_claims(), alg='none')
  result = _post_predict(client, token)
  assert result.status_code == 401
  assert result.json() == {'detail': 'unauthorized'}


def test_oidc_hs256_predict_401(monkeypatch):
  '''Alg HS256 token returns 401.'''
  _clean_oidc_env(monkeypatch)
  verifier, _, _ = _verifier()
  client, _ = _app_client(verifier)
  token = _make_token(_claims(), alg='HS256')
  result = _post_predict(client, token)
  assert result.status_code == 401


def test_oidc_revoked_predict_401(monkeypatch):
  '''Revoked jti token returns 401.'''
  from forecast_ml_framework.serving.auth import RevocationList
  _clean_oidc_env(monkeypatch)
  verifier, _, _ = _verifier()
  revocation = RevocationList()
  revocation.revoke(jti=_JTI, sub=None)
  client, _ = _app_client(verifier, revocation)
  token = _make_token(_claims())
  result = _post_predict(client, token)
  assert result.status_code == 401
  assert result.json() == {'detail': 'unauthorized'}


def test_oidc_missing_scope_predict_401(monkeypatch):
  '''OIDC without predict scope returns 401.'''
  _clean_oidc_env(monkeypatch)
  verifier, _, _ = _verifier()
  client, _ = _app_client(verifier)
  token = _make_token(_claims(scope='other:scope'))
  result = _post_predict(client, token)
  assert result.status_code == 401


def test_oidc_admin_scope_predict_200(monkeypatch):
  '''Admin scope satisfies predict via superset.'''
  _clean_oidc_env(monkeypatch)
  verifier, _, _ = _verifier()
  client, _ = _app_client(verifier)
  token = _make_token(_claims(scope='predict:admin'))
  result = _post_predict(client, token)
  assert result.status_code == 200


def test_partial_oidc_issuer_only_runtimeerror(monkeypatch):
  '''Issuer without audience fails closed at startup.'''
  from forecast_ml_framework.serving.app import create_app
  _clean_oidc_env(monkeypatch)
  monkeypatch.setenv('FORECAST_OIDC_ISSUER', _ISSUER)
  with pytest.raises(RuntimeError):
    create_app(dummy_loader)


def test_partial_oidc_jwks_only_runtimeerror(monkeypatch):
  '''JWKS URL alone fails closed at startup.'''
  from forecast_ml_framework.serving.app import create_app
  _clean_oidc_env(monkeypatch)
  monkeypatch.setenv('FORECAST_OIDC_JWKS_URL', _JWKS_URL)
  with pytest.raises(RuntimeError):
    create_app(dummy_loader)


def test_partial_oidc_audience_only_runtimeerror(monkeypatch):
  '''Audience without issuer fails closed at startup.'''
  from forecast_ml_framework.serving.app import create_app
  _clean_oidc_env(monkeypatch)
  monkeypatch.setenv('FORECAST_OIDC_AUDIENCE', _AUDIENCE)
  with pytest.raises(RuntimeError):
    create_app(dummy_loader)


def test_oidc_exempt_from_min_api_key_length(monkeypatch):
  '''Valid OIDC bypasses short shared key floor.'''
  _clean_oidc_env(monkeypatch)
  monkeypatch.setenv('FORECAST_API_KEY', 'short')
  verifier, _, _ = _verifier()
  client, _ = _app_client(verifier)
  token = _make_token(_claims())
  result = _post_predict(client, token)
  assert result.status_code == 200


def test_oidc_exempt_from_production_shared_key(monkeypatch):
  '''Prod profile plus OIDC needs no shared key.'''
  _clean_oidc_env(monkeypatch)
  monkeypatch.setenv('FORECAST_ENV', 'prod')
  verifier, _, _ = _verifier()
  client, _ = _app_client(verifier)
  token = _make_token(_claims())
  result = _post_predict(client, token)
  assert result.status_code == 200


def test_shared_key_still_enforces_floor_without_oidc(
  monkeypatch
):
  '''Short shared key still refused without verifier.'''
  from forecast_ml_framework.serving.app import create_app
  _clean_oidc_env(monkeypatch)
  monkeypatch.setenv('FORECAST_API_KEY', 'short')
  with pytest.raises(RuntimeError):
    create_app(dummy_loader)


def test_shared_key_fallback_only_when_verifier_none(monkeypatch):
  '''Shared key path works when verifier is None.'''
  _clean_oidc_env(monkeypatch)
  monkeypatch.setenv(
    'FORECAST_API_KEY', 'shared-secret-1234567890'
  )
  from fastapi.testclient import TestClient
  from forecast_ml_framework.serving.app import create_app
  app = create_app(
    dummy_loader, oidc_verifier=None, oidc_revocation=None
  )
  client = TestClient(app)
  payload = {'features': {'a': 1.0}}
  wrong = client.post(
    '/predict', json=payload,
    headers={'X-API-Key': 'wrong-key-value-1234567890'},
  )
  assert wrong.status_code == 401
  good = client.post(
    '/predict', json=payload,
    headers={'X-API-Key': 'shared-secret-1234567890'},
  )
  assert good.status_code == 200
  non_ascii = client.post(
    '/predict', json=payload,
    headers=[(b'x-api-key', b'\xe9')],
  )
  assert non_ascii.status_code == 401


def test_dev_mode_predict_200_with_warning(monkeypatch, caplog):
  '''Dev mode with no auth returns 200 plus warning.'''
  _clean_oidc_env(monkeypatch)
  from fastapi.testclient import TestClient
  from forecast_ml_framework.serving.app import create_app
  with caplog.at_level(logging.WARNING):
    app = create_app(dummy_loader)
  assert any(
    record.levelno == logging.WARNING
    and 'FORECAST_API_KEY' in record.getMessage()
    for record in caplog.records
  )
  client = TestClient(app)
  result = client.post(
    '/predict', json={'features': {'a': 1.0}}
  )
  assert result.status_code == 200


def test_oidc_no_token_in_logs_or_audit(monkeypatch, caplog):
  '''Clear token jti sub absent; hash present.'''
  from forecast_ml_framework.governance.scrub import subject_hash
  _clean_oidc_env(monkeypatch)
  verifier, _, _ = _verifier()
  client, app = _app_client(verifier)
  token = _make_token(_claims())
  with caplog.at_level(logging.INFO):
    result = _post_predict(client, token)
  assert result.status_code == 200
  audit_text = json.dumps(app.state.audit_log)
  assert token not in caplog.text
  assert token not in audit_text
  assert _JTI not in caplog.text
  assert _JTI not in audit_text
  assert _SUB not in caplog.text
  assert _SUB not in audit_text
  assert subject_hash(_SUB)[:16] in caplog.text
  assert subject_hash(_SUB)[:16] in audit_text
  assert 'oidc-' in caplog.text
  assert 'oidc-' in audit_text


def test_revocation_file_denies_via_app(monkeypatch, tmp_path):
  '''Revocation file hash denies via app wiring.'''
  from forecast_ml_framework.governance.scrub import subject_hash
  _clean_oidc_env(monkeypatch)
  target = tmp_path / 'revoked.txt'
  target.write_text(subject_hash(_SUB) + '\n', 'utf-8')
  monkeypatch.setenv('FORECAST_OIDC_REVOKED_FILE', str(target))
  verifier, _, _ = _verifier()
  client, _ = _app_client(verifier)
  token = _make_token(_claims())
  result = _post_predict(client, token)
  assert result.status_code == 401


def test_revoked_hashes_env_denies_via_app(monkeypatch):
  '''Revoked hashes env denies via app wiring.'''
  from forecast_ml_framework.governance.scrub import subject_hash
  _clean_oidc_env(monkeypatch)
  monkeypatch.setenv(
    'FORECAST_OIDC_REVOKED_HASHES', subject_hash(_JTI)
  )
  verifier, _, _ = _verifier()
  client, _ = _app_client(verifier)
  token = _make_token(_claims())
  result = _post_predict(client, token)
  assert result.status_code == 401


def test_oidc_built_from_env_uses_fake_fetcher(monkeypatch):
  '''Env built verifier signs through fake fetcher.'''
  _clean_oidc_env(monkeypatch)
  monkeypatch.setenv('FORECAST_OIDC_ISSUER', _ISSUER)
  monkeypatch.setenv('FORECAST_OIDC_AUDIENCE', _AUDIENCE)
  monkeypatch.setenv('FORECAST_OIDC_JWKS_URL', _JWKS_URL)
  fetcher = FakeFetcher(_make_jwks())
  clock = FakeClock()
  from fastapi.testclient import TestClient
  from forecast_ml_framework.serving.app import create_app
  app = create_app(
    dummy_loader, oidc_fetcher=fetcher, oidc_clock=clock
  )
  client = TestClient(app)
  token = _make_token(_claims())
  result = _post_predict(client, token)
  assert result.status_code == 200
  assert fetcher.calls
  assert fetcher.calls[0]['timeout'] == 5


def test_oidc_401_has_no_stack_trace(monkeypatch):
  '''401 body is generic detail only.'''
  _clean_oidc_env(monkeypatch)
  verifier, _, _ = _verifier()
  client, _ = _app_client(verifier)
  result = _post_predict(client, 'garbage-token-value')
  assert result.status_code == 401
  assert result.json() == {'detail': 'unauthorized'}
  assert 'Traceback' not in result.text
  assert 'token' not in result.text.lower() or 'unauthorized' in (
    result.text
  )


def test_verify_unknown_kid_via_verify_raises():
  '''Unknown kid verify raises ValueError directly.'''
  verifier, _, _ = _verifier()
  token = _make_token(_claims(), kid='nope-kid')
  with pytest.raises(ValueError):
    verifier.verify(token)


def test_clock_object_with_now_method():
  '''Verifier accepts clock object with now.'''
  from forecast_ml_framework.serving.auth import OIDCVerifier

  class NowClock:
    '''Clock object exposing now.'''

    def now(self):
      '''Return fixed time.'''
      return 1700000000.0

  verifier = OIDCVerifier(
    issuer=_ISSUER,
    audience=_AUDIENCE,
    jwks_url=_JWKS_URL,
    clock=NowClock(),
    fetcher=FakeFetcher(_make_jwks()),
  )
  assert verifier.verify(_make_token(_claims()))['sub'] == _SUB


def test_missing_jwks_url_raises():
  '''Missing JWKS URL raises ValueError on verify.'''
  from forecast_ml_framework.serving.auth import OIDCVerifier
  verifier = OIDCVerifier(
    issuer=_ISSUER,
    audience=_AUDIENCE,
    clock=FakeClock(),
    fetcher=FakeFetcher(_make_jwks()),
  )
  with pytest.raises(ValueError):
    verifier.verify(_make_token(_claims()))


def test_headers_case_insensitive_authenticate():
  '''Authenticate handles header case variants.'''
  from forecast_ml_framework.serving.auth import (
    RevocationList,
    authenticate,
  )
  verifier, _, _ = _verifier()
  token = _make_token(_claims())
  result = authenticate(
    {'Authorization': 'Bearer ' + token},
    verifier=verifier,
    revocation=RevocationList(),
    shared_key=None,
  )
  assert result['auth_mode'] == 'oidc'


def test_require_scope_roles_string_form():
  '''Roles string grants scope too.'''
  from forecast_ml_framework.serving.auth import require_scope
  require_scope({'roles': 'predict:read'}, 'predict:read')
  with pytest.raises(ValueError):
    require_scope({'roles': 'predict:read'}, 'predict:admin')


def test_verify_uses_current_time_for_expiry():
  '''Expiry compares against injected clock.'''
  verifier, clock, _ = _verifier()
  token = _make_token(_claims(exp=1700000000.0 + 60.0))
  verifier.verify(token)
  clock.advance(61.0)
  with pytest.raises(ValueError):
    verifier.verify(token)


def test_time_now_fallback_works(monkeypatch):
  '''Verifier works with default time clock.'''
  verifier, _, _ = _verifier(clock=lambda: 1700000000.0)
  assert verifier.verify(_make_token(_claims()))['sub'] == _SUB


def test_b64url_decode_rejects_non_ascii():
  '''Non-ASCII base64url input raises ValueError.'''
  from forecast_ml_framework.serving.auth import _b64url_decode
  with pytest.raises(ValueError):
    _b64url_decode('café-token')


def test_b64url_uint_rejects_empty():
  '''Empty integer text raises ValueError.'''
  from forecast_ml_framework.serving.auth import _b64url_uint
  with pytest.raises(ValueError):
    _b64url_uint('')


def test_now_clock_with_time_method():
  '''Clock object with time method is used.'''
  from forecast_ml_framework.serving.auth import _now_from_clock

  class TimeClock:
    '''Clock object exposing time.'''

    def time(self):
      '''Return fixed time.'''
      return 1700000000.0

  assert _now_from_clock(TimeClock()) == 1700000000.0


def test_now_clock_fallback_uses_time():
  '''Unknown clock object falls back to wall time.'''
  from forecast_ml_framework.serving.auth import _now_from_clock
  value = _now_from_clock(object())
  assert value > 1000000000.0


def test_extract_scopes_numeric_value():
  '''Numeric scope value is collected as text.'''
  from forecast_ml_framework.serving.auth import require_scope
  require_scope({'scope': 123}, '123')
  require_scope({'roles': [123]}, '123')


def test_parse_jwks_list_and_single_forms():
  '''JWKS list and single JWK forms parse.'''
  from forecast_ml_framework.serving.auth import _parse_jwks
  entry = _make_jwks()['keys'][0]
  assert _parse_jwks([entry])[_KID][0] == _TEST_N
  assert _parse_jwks(entry)[_KID][0] == _TEST_N
  bad = pytest.raises(ValueError)
  with bad:
    _parse_jwks('not-a-jwks')
  with bad:
    _parse_jwks({'keys': []})
  with bad:
    _parse_jwks({'keys': [{'kid': 'x'}]})


def test_default_fetcher_fetch_failures(monkeypatch):
  '''Default fetcher maps failures to ValueError.'''
  import urllib.request as _urlreq
  from forecast_ml_framework.serving.auth import (
    default_fetcher,
  )
  def boom(*args, **kwargs):
    '''Raise fetch error.'''
    raise OSError('network down')
  monkeypatch.setattr(
    _urlreq, 'urlopen', boom
  )
  with pytest.raises(ValueError):
    default_fetcher(
      _JWKS_URL, timeout=5, max_bytes=262144
    )


def test_default_fetcher_read_failure(monkeypatch):
  '''Default fetcher read failure is ValueError.'''
  import urllib.request as _urlreq
  from forecast_ml_framework.serving.auth import (
    default_fetcher,
  )

  class BadResp:
    '''Response raising on read.'''

    def read(self, size=-1):
      '''Raise read error.'''
      raise OSError('read down')

  def fake_urlopen(url, *, timeout):
    '''Return bad response.'''
    return BadResp()
  monkeypatch.setattr(
    _urlreq, 'urlopen', fake_urlopen
  )
  with pytest.raises(ValueError):
    default_fetcher(
      _JWKS_URL, timeout=5, max_bytes=262144
    )


def test_default_fetcher_bad_json(monkeypatch):
  '''Default fetcher bad JSON is ValueError.'''
  import urllib.request as _urlreq
  from forecast_ml_framework.serving.auth import (
    default_fetcher,
  )

  class TextResp:
    '''Response with bad JSON body.'''

    def read(self, size=-1):
      '''Return bad JSON bytes.'''
      return b'not-json'

  def fake_urlopen(url, *, timeout):
    '''Return bad JSON response.'''
    return TextResp()
  monkeypatch.setattr(
    _urlreq, 'urlopen', fake_urlopen
  )
  with pytest.raises(ValueError):
    default_fetcher(
      _JWKS_URL, timeout=5, max_bytes=262144
    )


def test_default_fetcher_non_dict_json(monkeypatch):
  '''Default fetcher non-dict JSON is ValueError.'''
  import urllib.request as _urlreq
  from forecast_ml_framework.serving.auth import (
    default_fetcher,
  )

  class ListResp:
    '''Response with list JSON body.'''

    def read(self, size=-1):
      '''Return list JSON bytes.'''
      return b'[]'

  def fake_urlopen(url, *, timeout):
    '''Return list JSON response.'''
    return ListResp()
  monkeypatch.setattr(
    _urlreq, 'urlopen', fake_urlopen
  )
  with pytest.raises(ValueError):
    default_fetcher(
      _JWKS_URL, timeout=5, max_bytes=262144
    )


def test_load_revocation_oserror_empty(tmp_path):
  '''Directory as revoked file yields empty set.'''
  from forecast_ml_framework.serving.auth import (
    load_revocation_list,
  )
  note = load_revocation_list(revoked_file=str(tmp_path))
  assert note.is_revoked(jti=_JTI, sub=_SUB) is False


def test_verify_bad_token_json_raises():
  '''Non-JSON header raises ValueError.'''
  verifier, _, _ = _verifier()
  head = _b64u(b'not-json')
  body = _b64u(b'not-json')
  sig = _b64u(b'xxx')
  with pytest.raises(ValueError):
    verifier.verify(head + '.' + body + '.' + sig)


def test_verify_non_dict_header_raises():
  '''Non-dict JSON header raises ValueError.'''
  verifier, _, _ = _verifier()
  head = _b64u(json.dumps([1]).encode())
  body = _b64u(
    json.dumps(_claims(), separators=(',', ':')).encode()
  )
  sig = _b64u(b'xxx')
  with pytest.raises(ValueError):
    verifier.verify(head + '.' + body + '.' + sig)


def test_verify_missing_kid_raises():
  '''Missing kid header raises ValueError.'''
  verifier, _, _ = _verifier()
  token = _make_token(_claims(), header={'kid': ''})
  with pytest.raises(ValueError):
    verifier.verify(token)


def test_verify_wrong_aud_list_raises():
  '''Audience list without ours raises ValueError.'''
  verifier, _, _ = _verifier()
  token = _make_token(_claims(aud=['api://a', 'api://b']))
  with pytest.raises(ValueError):
    verifier.verify(token)


def test_verify_nbf_future_raises():
  '''Future nbf token raises ValueError.'''
  verifier, _, _ = _verifier()
  claims = _claims(nbf=1700000000.0 + 500.0)
  token = _make_token(claims)
  with pytest.raises(ValueError):
    verifier.verify(token)


def test_verify_missing_signature_raises():
  '''RS256 token without signature raises ValueError.'''
  verifier, _, _ = _verifier()
  token = _make_token(_claims())
  heads, body, _sig = token.split('.')
  with pytest.raises(ValueError):
    verifier.verify(heads + '.' + body + '.')


def test_verify_oversize_signature_raises():
  '''Oversize signature int raises ValueError.'''
  verifier, _, _ = _verifier()
  token = _make_token(_claims())
  heads, body, _sig = token.split('.')
  big = _b64u(b'\xff' * 200)
  with pytest.raises(ValueError):
    verifier.verify(heads + '.' + body + '.' + big)


def test_verify_fetch_generic_error_raises():
  '''Generic fetcher error is wrapped as ValueError.'''
  from forecast_ml_framework.serving.auth import OIDCVerifier

  def bad_fetcher(url, *, timeout, max_bytes):
    '''Raise generic error.'''
    raise RuntimeError('jwks down')

  verifier = OIDCVerifier(
    issuer=_ISSUER,
    audience=_AUDIENCE,
    jwks_url=_JWKS_URL,
    clock=FakeClock(),
    fetcher=bad_fetcher,
  )
  with pytest.raises(ValueError):
    verifier.verify(_make_token(_claims()))


def test_authenticate_empty_bearer_raises():
  '''Empty Bearer token raises ValueError.'''
  from forecast_ml_framework.serving.auth import (
    RevocationList,
    authenticate,
  )
  verifier, _, _ = _verifier()
  with pytest.raises(ValueError):
    authenticate(
      {'authorization': 'Bearer '},
      verifier=verifier,
      revocation=RevocationList(),
      shared_key=None,
    )


def test_parse_jwks_skips_bad_entries():
  '''JWKS bad entries are skipped before missing.'''
  from forecast_ml_framework.serving.auth import _parse_jwks
  jwks = {
    'keys': [123, {'kid': '', 'n': 'AQ', 'e': 'AQ'},
             _make_jwks()['keys'][0]]
  }
  assert _parse_jwks(jwks)[_KID][0] == _TEST_N


def test_verify_tiny_key_rejected():
  '''Tiny modulus JWK raises ValueError.'''
  from forecast_ml_framework.serving.auth import OIDCVerifier
  tiny = {
    'keys': [{
      'kty': 'RSA', 'kid': _KID,
      'n': _b64u(_int_bytes(1234567)), 'e': _b64u(
        _int_bytes(3)),
    }]
  }
  verifier = OIDCVerifier(
    issuer=_ISSUER, audience=_AUDIENCE, jwks_url=_JWKS_URL,
    clock=FakeClock(), fetcher=FakeFetcher(tiny),
  )
  with pytest.raises(ValueError):
    verifier.verify(_make_token(_claims()))


def test_verify_fetcher_valueerror_propagates():
  '''Fetcher ValueError propagates as ValueError.'''
  from forecast_ml_framework.serving.auth import OIDCVerifier

  def bad_fetcher(url, *, timeout, max_bytes):
    '''Raise ValueError.'''
    raise ValueError('bad jwks payload')

  verifier = OIDCVerifier(
    issuer=_ISSUER, audience=_AUDIENCE, jwks_url=_JWKS_URL,
    clock=FakeClock(), fetcher=bad_fetcher,
  )
  with pytest.raises(ValueError):
    verifier.verify(_make_token(_claims()))


def test_verify_default_clock_is_wall_time():
  '''Verifier with no clock uses wall time.'''
  from forecast_ml_framework.serving.auth import OIDCVerifier
  verifier = OIDCVerifier(
    issuer=_ISSUER, audience=_AUDIENCE, jwks_url=_JWKS_URL,
    fetcher=FakeFetcher(_make_jwks()),
  )
  token = _make_token(_claims(exp=time.time() + 600.0))
  assert verifier.verify(token)['sub'] == _SUB
