#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''JWKS redirect revalidation tests for FIX-JWKS-REDIRECT.'''

import io
import json
import socket
import urllib.error
import urllib.request as urlreq
from email.message import Message

import pytest

_JWKS = {'keys': [{'kid': 'kid-1', 'kty': 'RSA'}]}
_JWKS_BODY = json.dumps(_JWKS).encode('utf-8')
_START_URL = 'https://login.example.com/jwks'
_PUBLIC_URL = 'https://cdn.example.com/jwks'
_PUBLIC_DNS = [
  (socket.AF_INET, socket.SOCK_STREAM, 6, '', ('8.8.8.8', 0))
]


class FakeResponse:
  '''Fake urlopen response holding a body and final URL.'''

  def __init__(self, body, final_url=None):
    '''Store the body bytes and final URL.'''
    self.body = body
    self._final_url = final_url

  def read(self, size=-1):
    '''Return the body up to size.'''
    if size is None or size < 0:
      return self.body
    return self.body[:size]

  def geturl(self):
    '''Return the final URL after any redirects.'''
    return self._final_url


def _public_dns(monkeypatch):
  '''Make every DNS name resolve to one public IP.'''

  def _resolve(unused_host, unused_port, *unused_args, **unused_kw):
    '''Return one public address.'''
    return list(_PUBLIC_DNS)

  monkeypatch.setattr(socket, 'getaddrinfo', _resolve)


def _dns_by_host(monkeypatch, mapping):
  '''Resolve hosts per mapping of host to IP list.'''

  def _resolve(host, unused_port, *unused_args, **unused_kw):
    '''Return mapped addresses for host.'''
    ips = mapping.get(str(host).lower(), ['8.8.8.8'])
    return [
      (socket.AF_INET, socket.SOCK_STREAM, 6, '', (ip, 0))
      for ip in ips
    ]

  monkeypatch.setattr(socket, 'getaddrinfo', _resolve)


def _redirect_error(url, location):
  '''Build an HTTP 302 error carrying a Location header.'''
  headers = Message()
  headers['Location'] = location
  return urllib.error.HTTPError(
    url, 302, 'Found', headers, io.BytesIO(b'')
  )


def _fake_redirects(monkeypatch, routes):
  '''Install fake urlopen driven by a routes mapping.

  Args:
    monkeypatch: Pytest monkeypatch fixture.
    routes: Mapping of URL to a redirect location string
      or to a body bytes payload for a final response.

  Returns:
    List recording each urlopen call URL.
  '''
  seen = []

  def fake_urlopen(url, *, timeout):
    '''Raise 302 for redirect routes, else return body.'''
    target = str(url)
    seen.append(target)
    outcome = routes[target]
    if isinstance(outcome, str):
      raise _redirect_error(target, outcome)
    return FakeResponse(outcome, final_url=target)

  monkeypatch.setattr(urlreq, 'urlopen', fake_urlopen)
  return seen


def _fake_autofollow(monkeypatch, body, final_url):
  '''Install fake urlopen simulating urllib auto-follow.

  The returned response reports final_url via geturl,
  as real urllib does after following a redirect.
  '''
  seen = []

  def fake_urlopen(url, *, timeout):
    '''Record call and return body with final URL.'''
    seen.append(str(url))
    return FakeResponse(body, final_url=final_url)

  monkeypatch.setattr(urlreq, 'urlopen', fake_urlopen)
  return seen


@pytest.mark.parametrize('location', [
  'https://169.254.169.254/latest/meta-data/',
  'https://10.0.0.1/jwks',
  'https://192.168.0.1/jwks',
  'https://127.0.0.1/jwks',
])
def test_redirect_to_private_ip_rejected(monkeypatch, location):
  '''302 to a private IP is rejected before following.'''
  from forecast_ml_framework.serving.auth import default_fetcher
  _public_dns(monkeypatch)
  seen = _fake_redirects(
    monkeypatch, {_START_URL: location}
  )
  with pytest.raises(ValueError):
    default_fetcher(_START_URL, timeout=5, max_bytes=262144)
  assert seen == [_START_URL]


@pytest.mark.parametrize('location', [
  'https://metadata.google.internal/computeMetadata/v1/',
  'https://metadata/jwks',
  'https://metadata.azure/jwks',
  'https://localhost/jwks',
])
def test_redirect_to_metadata_host_rejected(monkeypatch, location):
  '''302 to a metadata hostname is rejected.'''
  from forecast_ml_framework.serving.auth import default_fetcher
  _public_dns(monkeypatch)
  seen = _fake_redirects(
    monkeypatch, {_START_URL: location}
  )
  with pytest.raises(ValueError):
    default_fetcher(_START_URL, timeout=5, max_bytes=262144)
  assert seen == [_START_URL]


@pytest.mark.parametrize('location', [
  'file:///etc/passwd',
  'file://evil.example.com/etc/passwd',
  'http://cdn.example.com/jwks',
  'ftp://cdn.example.com/jwks',
])
def test_redirect_to_bad_scheme_rejected(monkeypatch, location):
  '''302 to a non-HTTPS scheme is rejected.'''
  from forecast_ml_framework.serving.auth import default_fetcher
  _public_dns(monkeypatch)
  seen = _fake_redirects(
    monkeypatch, {_START_URL: location}
  )
  with pytest.raises(ValueError):
    default_fetcher(_START_URL, timeout=5, max_bytes=262144)
  assert seen == [_START_URL]


def test_redirect_to_dns_private_rejected(monkeypatch):
  '''302 to a host resolving privately is rejected.'''
  from forecast_ml_framework.serving.auth import default_fetcher
  _dns_by_host(monkeypatch, {
    'login.example.com': ['8.8.8.8'],
    'evil.example.com': ['10.0.0.8'],
  })
  seen = _fake_redirects(
    monkeypatch,
    {_START_URL: 'https://evil.example.com/jwks'},
  )
  with pytest.raises(ValueError):
    default_fetcher(_START_URL, timeout=5, max_bytes=262144)
  assert seen == [_START_URL]


def test_redirect_to_public_https_passes(monkeypatch):
  '''302 to a public HTTPS URL is followed and passes.'''
  from forecast_ml_framework.serving.auth import default_fetcher
  _public_dns(monkeypatch)
  seen = _fake_redirects(monkeypatch, {
    _START_URL: _PUBLIC_URL,
    _PUBLIC_URL: _JWKS_BODY,
  })
  got = default_fetcher(_START_URL, timeout=5, max_bytes=262144)
  assert got == _JWKS
  assert seen == [_START_URL, _PUBLIC_URL]


def test_relative_redirect_to_public_passes(monkeypatch):
  '''Relative 302 to a public path is followed.'''
  from forecast_ml_framework.serving.auth import default_fetcher
  _public_dns(monkeypatch)
  target = 'https://login.example.com/real/jwks'
  seen = _fake_redirects(monkeypatch, {
    _START_URL: '/real/jwks',
    target: _JWKS_BODY,
  })
  got = default_fetcher(_START_URL, timeout=5, max_bytes=262144)
  assert got == _JWKS
  assert seen == [_START_URL, target]


def test_relative_redirect_to_private_rejected(monkeypatch):
  '''Relative 302 cannot escape to a private host.'''
  from forecast_ml_framework.serving.auth import default_fetcher
  _public_dns(monkeypatch)
  seen = _fake_redirects(monkeypatch, {
    _START_URL: '//169.254.169.254/latest/meta-data/',
  })
  with pytest.raises(ValueError):
    default_fetcher(_START_URL, timeout=5, max_bytes=262144)
  assert seen == [_START_URL]


def test_autofollowed_private_final_url_rejected(monkeypatch):
  '''Response with private final URL is rejected.'''
  from forecast_ml_framework.serving.auth import default_fetcher
  _public_dns(monkeypatch)
  seen = _fake_autofollow(
    monkeypatch,
    _JWKS_BODY,
    'https://169.254.169.254/latest/meta-data/',
  )
  with pytest.raises(ValueError):
    default_fetcher(_START_URL, timeout=5, max_bytes=262144)
  assert seen == [_START_URL]


def test_autofollowed_public_final_url_passes(monkeypatch):
  '''Response with public final URL is accepted.'''
  from forecast_ml_framework.serving.auth import default_fetcher
  _public_dns(monkeypatch)
  seen = _fake_autofollow(
    monkeypatch, _JWKS_BODY, _PUBLIC_URL
  )
  got = default_fetcher(_START_URL, timeout=5, max_bytes=262144)
  assert got == _JWKS
  assert seen == [_START_URL]


def test_too_many_redirects_rejected(monkeypatch):
  '''A chain of four redirects exceeds the limit.'''
  from forecast_ml_framework.serving.auth import default_fetcher
  _public_dns(monkeypatch)
  first = 'https://a.example.com/jwks'
  second = 'https://b.example.com/jwks'
  third = 'https://c.example.com/jwks'
  fourth = 'https://d.example.com/jwks'
  seen = _fake_redirects(monkeypatch, {
    _START_URL: first,
    first: second,
    second: third,
    third: fourth,
    fourth: _JWKS_BODY,
  })
  with pytest.raises(ValueError):
    default_fetcher(_START_URL, timeout=5, max_bytes=262144)
  assert fourth not in seen


def test_three_redirects_pass(monkeypatch):
  '''A chain of three redirects is within the limit.'''
  from forecast_ml_framework.serving.auth import default_fetcher
  _public_dns(monkeypatch)
  first = 'https://a.example.com/jwks'
  second = 'https://b.example.com/jwks'
  third = 'https://c.example.com/jwks'
  seen = _fake_redirects(monkeypatch, {
    _START_URL: first,
    first: second,
    second: third,
    third: _JWKS_BODY,
  })
  got = default_fetcher(_START_URL, timeout=5, max_bytes=262144)
  assert got == _JWKS
  assert seen == [_START_URL, first, second, third]


def test_redirect_without_location_rejected(monkeypatch):
  '''302 lacking a Location header is rejected.'''
  from forecast_ml_framework.serving.auth import default_fetcher
  _public_dns(monkeypatch)
  seen = []

  def fake_urlopen(url, *, timeout):
    '''Raise a bare 302 with no Location.'''
    seen.append(str(url))
    raise urllib.error.HTTPError(
      str(url), 302, 'Found', Message(), io.BytesIO(b'')
    )

  monkeypatch.setattr(urlreq, 'urlopen', fake_urlopen)
  with pytest.raises(ValueError):
    default_fetcher(_START_URL, timeout=5, max_bytes=262144)
  assert seen == [_START_URL]
