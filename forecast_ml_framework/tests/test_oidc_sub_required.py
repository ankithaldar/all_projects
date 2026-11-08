#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''OIDC sub claim is required tests (FIX-OIDC-SUB).'''

import base64
import hashlib
import json

import pytest

from forecast_ml_framework.serving.auth import (
  OIDCVerifier,
  RevocationList,
  authenticate,
)

_ISSUER = 'https://login.microsoftonline.com/test-tenant/v2.0'
_AUDIENCE = 'api://forecast-ml-api'
_JWKS_URL = 'https://example.invalid/jwks'
_KID = 'test-kid-1'
_SUB = 'user-test-subject-001'
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
_PREFIX = bytes.fromhex(
  '3031300d060960864801650304020105000420'
)


class FakeClock:
  '''Mutable clock returning fixed seconds.'''

  def __call__(self):
    '''Return current fake time.'''
    return 1700000000.0


class FakeFetcher:
  '''Fake JWKS fetcher returning a fixed document.'''

  def __init__(self, jwks):
    '''Initialise with a JWKS dict to return.'''
    self.jwks = jwks

  def __call__(self, url, *, timeout, max_bytes):
    '''Return the fixed JWKS document.'''
    return self.jwks


def _b64u(data):
  '''Return base64url text without padding.'''
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
  tail = _PREFIX + digest
  size = (_TEST_N.bit_length() + 7) // 8
  ps_len = size - len(tail) - 3
  memory = b'\x00\x01' + b'\xff' * ps_len + b'\x00' + tail
  number = int.from_bytes(memory, 'big')
  signed = pow(number, _TEST_D, _TEST_N)
  return signed.to_bytes(size, 'big')


def _make_token(claims):
  '''Build a signed JWT string for given claims.'''
  header = {'alg': 'RS256', 'typ': 'JWT', 'kid': _KID}
  head = _b64u(
    json.dumps(header, separators=(',', ':'))
    .encode('utf-8')
  )
  payload = _b64u(
    json.dumps(claims, separators=(',', ':'))
    .encode('utf-8')
  )
  signing_input = (head + '.' + payload).encode('ascii')
  sig = _b64u(_sign(signing_input))
  return head + '.' + payload + '.' + sig


def _claims(**overrides):
  '''Return baseline valid claims.'''
  found = {
    'iss': _ISSUER,
    'aud': _AUDIENCE,
    'exp': 1700000000.0 + 600.0,
    'sub': _SUB,
    'scope': 'predict:read',
  }
  found.update(overrides)
  return found


def _verifier():
  '''Build a verifier with fake clock and fetcher.'''
  return OIDCVerifier(
    issuer=_ISSUER,
    audience=_AUDIENCE,
    jwks_url=_JWKS_URL,
    clock=FakeClock(),
    fetcher=FakeFetcher(_make_jwks()),
  )


def test_verify_valid_sub_still_verifies():
  '''Token with a valid sub still verifies.'''
  verifier = _verifier()
  claims = verifier.verify(_make_token(_claims()))
  assert claims['sub'] == _SUB


def test_verify_missing_sub_raises():
  '''Token without sub raises ValueError.'''
  verifier = _verifier()
  claims = _claims()
  del claims['sub']
  with pytest.raises(ValueError, match='missing sub'):
    verifier.verify(_make_token(claims))


@pytest.mark.parametrize('bad_sub', ['', '   ', None])
def test_verify_empty_sub_raises(bad_sub):
  '''Token with empty or null sub raises ValueError.'''
  verifier = _verifier()
  token = _make_token(_claims(sub=bad_sub))
  with pytest.raises(ValueError, match='missing sub'):
    verifier.verify(token)


@pytest.mark.parametrize(
  'bad_sub', [123, 0, True, ['user'], {'id': 'user'}]
)
def test_verify_non_string_sub_raises(bad_sub):
  '''Token with non-string sub raises ValueError.'''
  verifier = _verifier()
  token = _make_token(_claims(sub=bad_sub))
  with pytest.raises(ValueError, match='missing sub'):
    verifier.verify(token)


def test_authenticate_no_sub_raises_before_scope():
  '''Authenticate rejects no-sub before scope check.'''
  verifier = _verifier()
  claims = _claims(scope='other:scope')
  del claims['sub']
  token = _make_token(claims)
  with pytest.raises(ValueError, match='missing sub'):
    authenticate(
      {'authorization': 'Bearer ' + token},
      verifier=verifier,
      revocation=RevocationList(),
      shared_key=None,
    )
