#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''JWKS SSRF guard tests for FIX-SEC-M3.'''

import json
import re
import socket
import urllib.request as urlreq
from pathlib import Path

import pytest

_JWKS = {'keys': [{'kid': 'kid-1', 'kty': 'RSA'}]}
_JWKS_BODY = json.dumps(_JWKS).encode('utf-8')
_PUBLIC_JWKS_URL = 'https://login.example.com/jwks'
_PUBLIC_DNS = [
  (socket.AF_INET, socket.SOCK_STREAM, 6, '', ('8.8.8.8', 0))
]


class FakeFetcher:
  '''Fake JWKS fetcher recording its calls.'''

  def __init__(self, jwks):
    '''Initialise with a JWKS dict to return.'''
    self.jwks = jwks
    self.calls = []

  def __call__(self, url, *, timeout, max_bytes):
    '''Record the call and return the JWKS dict.'''
    self.calls.append(
      {'url': url, 'timeout': timeout, 'max_bytes': max_bytes}
    )
    return self.jwks


class FakeClock:
  '''Clock returning a fixed epoch unless advanced.'''

  def __init__(self, now=1700000000.0):
    '''Initialise the clock at a fixed epoch.'''
    self.now = now

  def __call__(self):
    '''Return the current fake time.'''
    return self.now


def _fake_response(monkeypatch, payload=None):
  '''Install a fake urlopen and return the call list.'''
  seen = []

  class FakeResponse:
    '''Fake urlopen response holding a body.'''

    def __init__(self, body):
      '''Store the body bytes.'''
      self.body = body

    def read(self, size=-1):
      '''Return the body up to size.'''
      if size is None or size < 0:
        return self.body
      return self.body[:size]

  def fake_urlopen(url, *, timeout):
    '''Record the call and return the payload.'''
    seen.append({'url': url, 'timeout': timeout})
    return FakeResponse(_JWKS_BODY if payload is None else payload)

  monkeypatch.setattr(urlreq, 'urlopen', fake_urlopen)
  return seen


def _public_dns(monkeypatch):
  '''Make every DNS name resolve to one public IP.'''

  def _resolve(host, port, *args, **kwargs):
    '''Return one public address.'''
    return list(_PUBLIC_DNS)

  monkeypatch.setattr(socket, 'getaddrinfo', _resolve)


def _dns_result(monkeypatch, ips):
  '''Make DNS resolve to the given IP list.'''

  def _resolve(host, port, *args, **kwargs):
    '''Return the given addresses.'''
    return [
      (socket.AF_INET, socket.SOCK_STREAM, 6, '', (ip, 0))
      for ip in ips
    ]

  monkeypatch.setattr(socket, 'getaddrinfo', _resolve)


def _verifier(jwks_url):
  '''Build a verifier with a fake fetcher for one URL.'''
  from forecast_ml_framework.serving.auth import OIDCVerifier
  fetcher = FakeFetcher(_JWKS)
  verifier = OIDCVerifier(
    issuer='https://issuer.example/v2.0',
    audience='api://test',
    jwks_url=jwks_url,
    clock=FakeClock(),
    fetcher=fetcher,
  )
  return verifier, fetcher


@pytest.mark.parametrize('url', [
  'file:///etc/passwd',
  'file://evil.example.com/etc/passwd',
  'file:///tmp/jwks.json',
])
def test_jwks_default_fetcher_rejects_file_urls(
  monkeypatch, url
):
  '''File scheme URLs raise before any fetch attempt.'''
  from forecast_ml_framework.serving.auth import default_fetcher
  _public_dns(monkeypatch)
  seen = _fake_response(monkeypatch)
  with pytest.raises(ValueError):
    default_fetcher(url, timeout=5, max_bytes=262144)
  assert seen == []


@pytest.mark.parametrize('url', [
  'http://8.8.8.8/jwks',
  'http://169.254.169.254/latest/meta-data/',
  'http://public.example.com/jwks',
])
def test_jwks_default_fetcher_rejects_http_urls(
  monkeypatch, url
):
  '''Plain HTTP URLs raise because only HTTPS is allowed.'''
  from forecast_ml_framework.serving.auth import default_fetcher
  _public_dns(monkeypatch)
  seen = _fake_response(monkeypatch)
  with pytest.raises(ValueError):
    default_fetcher(url, timeout=5, max_bytes=262144)
  assert seen == []


@pytest.mark.parametrize('url', [
  'https://169.254.169.254/latest/meta-data/',
  'https://100.100.100.200/metadata',
  'https://10.0.0.1/jwks',
  'https://192.168.0.1/jwks',
  'https://172.16.0.1/jwks',
  'https://127.0.0.1/jwks',
  'https://[::1]/jwks',
])
def test_jwks_default_fetcher_rejects_ip_literals(
  monkeypatch, url
):
  '''Non-public IP literals raise before any fetch attempt.'''
  from forecast_ml_framework.serving.auth import default_fetcher
  _public_dns(monkeypatch)
  seen = _fake_response(monkeypatch)
  with pytest.raises(ValueError):
    default_fetcher(url, timeout=5, max_bytes=262144)
  assert seen == []


@pytest.mark.parametrize('url', [
  'https://metadata.google.internal/computeMetadata/v1/',
  'https://metadata/jwks',
  'https://metadata.azure/jwks',
  'https://localhost/jwks',
  'https://foo.localhost/jwks',
  'https://foo.localhost.localdomain/jwks',
])
def test_jwks_default_fetcher_rejects_hostnames(
  monkeypatch, url
):
  '''Blocked hostnames raise before any fetch attempt.'''
  from forecast_ml_framework.serving.auth import default_fetcher
  _public_dns(monkeypatch)
  seen = _fake_response(monkeypatch)
  with pytest.raises(ValueError):
    default_fetcher(url, timeout=5, max_bytes=262144)
  assert seen == []


@pytest.mark.parametrize('url', [
  'https://0xA9FEA9FE/jwks',
  'https://2852039166/jwks',
])
def test_jwks_default_fetcher_rejects_encoded_hosts(
  monkeypatch, url
):
  '''Encoded IP hosts raise before any fetch attempt.'''
  from forecast_ml_framework.serving.auth import default_fetcher
  _public_dns(monkeypatch)
  seen = _fake_response(monkeypatch)
  with pytest.raises(ValueError):
    default_fetcher(url, timeout=5, max_bytes=262144)
  assert seen == []


def test_jwks_default_fetcher_rejects_userinfo(monkeypatch):
  '''URLs with userinfo raise before any fetch.'''
  from forecast_ml_framework.serving.auth import default_fetcher
  _public_dns(monkeypatch)
  seen = _fake_response(monkeypatch)
  with pytest.raises(ValueError):
    default_fetcher(
      'https://user:secret@login.example.com/jwks',
      timeout=5,
      max_bytes=262144,
    )
  assert seen == []


def test_jwks_default_fetcher_rejects_dns_private(monkeypatch):
  '''Hostnames resolving to a private IP raise.'''
  from forecast_ml_framework.serving.auth import default_fetcher
  _dns_result(monkeypatch, ['10.0.0.8'])
  seen = _fake_response(monkeypatch)
  with pytest.raises(ValueError):
    default_fetcher(
      'https://evil.example.com/jwks',
      timeout=5,
      max_bytes=262144,
    )
  assert seen == []


def test_jwks_default_fetcher_rejects_dns_metadata(monkeypatch):
  '''Hostnames resolving to metadata IP raise.'''
  from forecast_ml_framework.serving.auth import default_fetcher
  _dns_result(monkeypatch, ['169.254.169.254'])
  seen = _fake_response(monkeypatch)
  with pytest.raises(ValueError):
    default_fetcher(
      'https://evil.example.com/jwks',
      timeout=5,
      max_bytes=262144,
    )
  assert seen == []


def test_jwks_default_fetcher_rejects_mixed_dns(monkeypatch):
  '''One non-public address among public ones raises.'''
  from forecast_ml_framework.serving.auth import default_fetcher
  _dns_result(monkeypatch, ['8.8.8.8', '10.0.0.8'])
  seen = _fake_response(monkeypatch)
  with pytest.raises(ValueError):
    default_fetcher(
      'https://mixed.example.com/jwks',
      timeout=5,
      max_bytes=262144,
    )
  assert seen == []


def test_jwks_default_fetcher_dns_failure_proceeds(
  monkeypatch
):
  '''Unresolvable hosts still reach the fetch.'''

  def _fail(host, port, *args, **kwargs):
    '''Raise a resolution failure.'''
    raise socket.gaierror('no such host')

  from forecast_ml_framework.serving.auth import default_fetcher
  monkeypatch.setattr(socket, 'getaddrinfo', _fail)
  seen = _fake_response(monkeypatch)
  got = default_fetcher(
    'https://unresolvable.example.com/jwks',
    timeout=5,
    max_bytes=262144,
  )
  assert got == _JWKS
  assert seen


def test_jwks_default_fetcher_accepts_https_public(monkeypatch):
  '''HTTPS public URLs fetch successfully.'''
  from forecast_ml_framework.serving.auth import default_fetcher
  _public_dns(monkeypatch)
  seen = _fake_response(monkeypatch)
  got = default_fetcher(
    _PUBLIC_JWKS_URL, timeout=5, max_bytes=262144
  )
  assert got == _JWKS
  assert seen
  assert seen[0]['url'] == _PUBLIC_JWKS_URL


def test_jwks_verifier_rejects_file_without_fetch():
  '''Verifier rejects file URLs before the fetcher runs.'''
  verifier, fetcher = _verifier('file:///etc/passwd')
  with pytest.raises(ValueError):
    verifier._get_jwks(1700000000.0)
  assert fetcher.calls == []


def test_jwks_verifier_accepts_https_public_fake_fetcher():
  '''Verifier accepts an HTTPS public URL with fake fetcher.'''
  verifier, fetcher = _verifier(_PUBLIC_JWKS_URL)
  got = verifier._get_jwks(1700000000.0)
  assert got == _JWKS
  assert fetcher.calls
  assert fetcher.calls[0]['url'] == _PUBLIC_JWKS_URL


def test_dockerfile_pins_base_and_uv():
  '''Dockerfile pins the base digest and the uv version.'''
  root = Path(__file__).resolve().parents[1]
  text = (root / 'Dockerfile').read_text(encoding='utf-8')
  pinned = [
    line for line in text.splitlines() if line.startswith('FROM ')
  ]
  assert len(pinned) == 2
  for line in pinned:
    assert 'python:3.14-slim@sha256:' in line
  assert re.search(r'uv==\d+\.\d+\.\d+', text)
  assert 'pip install --no-cache-dir uv\n' not in text
