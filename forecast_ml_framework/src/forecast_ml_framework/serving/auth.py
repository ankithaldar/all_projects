#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''RS256 OIDC verifier, RBAC and revocation.

Only ``RS256`` is accepted, verified with stdlib ``pow``
over JWK ``n``/``e`` with no new dependency. No token,
``jti`` or ``sub`` is ever logged in clear; only
``subject_hash`` digests may reach logs.
'''

import base64
import hashlib
import hmac
import ipaddress
import json
import logging
import re
import secrets
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

from forecast_ml_framework.governance.scrub import subject_hash

ALLOWED_SCOPES: tuple[str, ...] = (
  'predict:read', 'predict:admin'
)

_JWKS_TIMEOUT = 5
_JWKS_MAX_BYTES = 262144
_JWKS_MAX_REDIRECTS = 3
_JWKS_REDIRECT_CODES = frozenset({301, 302, 303, 307, 308})
_JWKS_OPEN_LOCK = threading.Lock()
_SHA256_PREFIX = bytes.fromhex(
  '3031300d060960864801650304020105000420'
)

#: Hostnames serving cloud metadata; never valid for JWKS.
_BLOCKED_JWKS_HOSTS = frozenset({
  'metadata.google.internal', 'metadata', 'metadata.azure',
  'metadata.azure.internal', '100.100.100.200',
})

#: Numeric label: decimal/octal digits or 0x hex.
_NUMERIC_PART_RE = re.compile(
  r'(?:0[xX][0-9a-fA-F]+|[0-9]+)'
)


def _b64url_decode(data):
  '''Decode base64url text to bytes.

  Args:
    data: Base64url text without padding.

  Returns:
    Decoded bytes.

  Raises:
    ValueError: When decoding fails.
  '''
  text = str(data).strip()
  try:
    padded = text + '=' * (-len(text) % 4)
    return base64.urlsafe_b64decode(padded.encode('ascii'))
  except (ValueError, UnicodeEncodeError):
    raise ValueError('invalid base64url') from None


def _b64url_uint(data):
  '''Decode base64url unsigned integer text.

  Args:
    data: Base64url text encoding a big-endian integer.

  Returns:
    The integer value.

  Raises:
    ValueError: When decoding fails.
  '''
  raw = _b64url_decode(data)
  if not raw:
    raise ValueError('empty integer')
  return int.from_bytes(raw, 'big')


def _now_from_clock(clock):
  '''Return current seconds from an injectable clock.

  Args:
    clock: Callable, object with ``now``/``time``, or None.

  Returns:
    Float seconds since epoch.
  '''
  if clock is None:
    return time.time()
  if callable(clock):
    return float(clock())
  if hasattr(clock, 'now'):
    return float(clock.now())
  if hasattr(clock, 'time'):
    return float(clock.time())
  return time.time()


def _extract_scopes(claims):
  '''Collect granted scopes from scope/roles claims.

  Args:
    claims: Verified claims mapping.

  Returns:
    Set of granted scope strings.
  '''
  found = set()
  for key in ('scope', 'roles'):
    value = claims.get(key)
    if value is None:
      continue
    if isinstance(value, str):
      for piece in value.replace(',', ' ').split():
        piece = piece.strip()
        if piece:
          found.add(piece)
    elif isinstance(value, (list, tuple, set)):
      for piece in value:
        text = str(piece).strip()
        if text:
          found.add(text)
    else:
      text = str(value).strip()
      if text:
        found.add(text)
  return found


def _parse_jwks(jwks):
  '''Map JWKS keys by kid to (n, e) integers.

  Args:
    jwks: JWKS dict with a ``keys`` list, a list, or one JWK.

  Returns:
    Dict mapping kid to a (n, e) pair.

  Raises:
    ValueError: When JWKS is malformed or unusable.
  '''
  if isinstance(jwks, dict) and 'keys' in jwks:
    entries = jwks['keys']
  elif isinstance(jwks, list):
    entries = jwks
  elif isinstance(jwks, dict):
    entries = [jwks]
  else:
    raise ValueError('invalid jwks')
  found = {}
  for entry in entries:
    if not isinstance(entry, dict):
      continue
    kid = entry.get('kid')
    if not kid:
      continue
    if 'n' not in entry or 'e' not in entry:
      continue
    found[str(kid)] = (
      _b64url_uint(entry['n']), _b64url_uint(entry['e'])
    )
  if not found:
    raise ValueError('invalid jwks')
  return found


def _is_numeric_host(host):
  '''Return True when every label of host is numeric.

  Covers decimal, hex and octal label forms, e.g.
  ``0xa9.0xfe.0xa9.0xfe``, which ``getaddrinfo``
  resolves to an IP that ``ipaddress`` alone would
  not parse from the literal text.

  Args:
    host: Lowercased hostname string.

  Returns:
    True for all-numeric label forms.
  '''
  parts = host.split('.')
  if not parts or any(part == '' for part in parts):
    return False
  return all(
    _NUMERIC_PART_RE.fullmatch(part) for part in parts
  )


def _jwks_ip_blocked(ip):
  '''Return True when ip is not a public address.

  Unwraps IPv4-mapped IPv6 and rejects 6to4/Teredo
  transition addresses. Any non-global address is
  blocked, including CGNAT 100.64.0.0/10 where
  ``is_private`` is False but ``is_global`` is also
  False, plus private, loopback, link-local,
  reserved, multicast and unspecified addresses.

  Args:
    ip: Parsed IP address object.

  Returns:
    True for every non-public address.
  '''
  if isinstance(ip, ipaddress.IPv6Address):
    mapped = ip.ipv4_mapped
    if mapped is not None:
      return _jwks_ip_blocked(mapped)
    if ip.sixtofour is not None or ip.teredo is not None:
      return True
  return bool(
    not ip.is_global or ip.is_private or ip.is_loopback
    or ip.is_link_local or ip.is_reserved
    or ip.is_multicast or ip.is_unspecified
    or str(ip) == '100.100.100.200'
  )


def _validate_jwks_host(host):
  '''Reject blocked or non-public JWKS host literals.

  Args:
    host: Lowercased hostname, trailing dot removed.

  Raises:
    ValueError: When the host is blocked, encoded,
      empty, or a non-public IP literal. Plain DNS
      names pass here; DNS resolution is checked
      separately by ``_validate_jwks_dns``.
  '''
  if host and not host.isascii():
    raise ValueError('invalid jwks url')
  if '%' in host:
    raise ValueError('invalid jwks url')
  if host in _BLOCKED_JWKS_HOSTS:
    raise ValueError('invalid jwks url')
  if host == 'localhost.localdomain' or host.endswith(
    ('.localhost', '.localhost.localdomain')):
    raise ValueError('invalid jwks url')
  if not host or host == 'localhost':
    raise ValueError('invalid jwks url')
  try:
    ip = ipaddress.ip_address(host)
  except ValueError:
    ip = None
  if _is_numeric_host(host) and (
    ip is None or str(ip) != host):
    raise ValueError('invalid jwks url')
  if ip is not None and _jwks_ip_blocked(ip):
    raise ValueError('invalid jwks url')


def validate_jwks_url(url):
  '''Validate a JWKS URL without resolving DNS.

  Only ``https`` URLs pass. Any other scheme
  including ``file`` and ``http``, userinfo,
  malformed text, or a blocked/non-public host
  literal raises. DNS hostnames are validated for
  blocked literals here and resolved only by the
  default fetcher before connecting.

  Args:
    url: JWKS document URL string.

  Returns:
    The URL unchanged.

  Raises:
    ValueError: When the scheme, form or host is
      not an allowed public HTTPS JWKS URL.
  '''
  if not isinstance(url, str) or not url.strip() \
    or url != url.strip() \
    or any(ord(ch) < 32 or ord(ch) == 127 for ch in url) \
    or ' ' in url:
    raise ValueError('invalid jwks url')
  if not url.lower().startswith('https://'):
    raise ValueError('invalid jwks url')
  try:
    parsed = urllib.parse.urlparse(url)
  except ValueError:
    raise ValueError('invalid jwks url') from None
  if parsed.scheme.lower() != 'https':
    raise ValueError('invalid jwks url')
  if parsed.username is not None or parsed.password is not None:
    raise ValueError('invalid jwks url')
  host = (parsed.hostname or '').lower().rstrip('.')
  if not host:
    raise ValueError('invalid jwks url')
  _validate_jwks_host(host)
  return url


def _validate_jwks_dns(host):
  '''Reject DNS names resolving to non-public IPs.

  Only successfully resolved non-public addresses
  are blocked. A resolution failure or empty answer
  is ignored here because the subsequent fetch then
  fails without contacting any host; no SSRF target
  exists at an unresolvable name. This differs from
  config-time registry validation which fails
  closed. A production IdP hostname must resolve
  for the fetch to succeed, where the public-IP
  check still applies. Residual DNS TOCTOU between
  this check and the connect-time lookup remains;
  run behind a network egress allowlist.

  Args:
    host: Lowercased hostname to resolve.

  Raises:
    ValueError: When any resolved address is
      non-public.
  '''
  try:
    infos = socket.getaddrinfo(
      host, None, type=socket.SOCK_STREAM)
  except Exception:  # pylint: disable=broad-except
    return
  if not infos:
    return
  for info in infos:
    try:
      raw = info[4][0]
      resolved = ipaddress.ip_address(raw)
    except (IndexError, TypeError, ValueError):
      raise ValueError('invalid jwks url') from None
    if _jwks_ip_blocked(resolved):
      raise ValueError('invalid jwks url')


def _validate_jwks_dns_for_url(url):
  '''Check DNS for a hostname JWKS URL host.

  Args:
    url: Validated JWKS URL string.

  Raises:
    ValueError: When the hostname does not resolve
      only to public addresses.
  '''
  parsed = urllib.parse.urlparse(url)
  host = (parsed.hostname or '').lower().rstrip('.')
  if not host:
    raise ValueError('invalid jwks url')
  try:
    ipaddress.ip_address(host)
  except ValueError:
    _validate_jwks_dns(host)


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
  '''Redirect handler that never follows redirects.'''

  def redirect_request(self, *unused_args):
    '''Return None so redirects surface as errors.

    Args:
      *unused_args: Arguments from the base handler call.

    Returns:
      None, always.
    '''
    return None


def _redirect_location(error):
  '''Extract the Location target from a redirect error.

  Args:
    error: ``HTTPError`` carrying response headers.

  Returns:
    Location string, or None when absent or empty.
  '''
  headers = getattr(error, 'headers', None)
  if headers is None:
    headers = getattr(error, 'hdrs', None)
  if headers is None:
    return None
  getter = getattr(headers, 'get', None)
  if getter is None:
    return None
  for name in ('Location', 'location', 'URI', 'uri'):
    try:
      value = getter(name)
    except Exception:  # pylint: disable=broad-except
      value = None
    if value:
      text = str(value).strip()
      if text:
        return text
  return None


def _response_final_url(response):
  '''Return the final URL reported by a response.

  Args:
    response: Fetched response object.

  Returns:
    Final URL string, or None when not reported.
  '''
  getter = getattr(response, 'geturl', None)
  if callable(getter):
    try:
      final = getter()
    except Exception:  # pylint: disable=broad-except
      final = None
    if final:
      return str(final)
  final = getattr(response, 'url', None)
  if final:
    return str(final)
  return None


def _open_jwks_url(url, timeout):
  '''Open a URL with automatic redirects disabled.

  ``urlopen`` follows redirects through a global opener,
  so a no-redirect opener is installed for the call
  under a lock and the previous opener is restored.

  Args:
    url: Validated JWKS URL string.
    timeout: Timeout in seconds.

  Returns:
    Open response object for the final URL.

  Raises:
    urllib.error.HTTPError: On a redirect response.
    Exception: On any other fetch failure.
  '''
  opener = urllib.request.build_opener(_NoRedirectHandler())
  previous = getattr(urllib.request, '_opener', None)
  with _JWKS_OPEN_LOCK:
    urllib.request.install_opener(opener)
    try:
      return urllib.request.urlopen(url, timeout=timeout)
    finally:
      urllib.request.install_opener(previous)


def default_fetcher(url, *, timeout, max_bytes):
  '''Fetch JWKS over HTTPS with stdlib urllib.

  The URL is validated first: only HTTPS URLs with
  a public host pass, and DNS names must resolve
  only to public addresses. Redirects are never
  followed automatically; each redirect target is
  re-validated for scheme, host and DNS before
  it is requested, up to a maximum of three.
  Timeout and max-bytes limits bound the fetch.

  Args:
    url: JWKS document URL.
    timeout: Timeout in seconds.
    max_bytes: Maximum accepted body bytes.

  Returns:
    Parsed JWKS dict.

  Raises:
    ValueError: On blocked URL, oversized body,
      bad JSON, or fetch error.
  '''
  current = url
  validate_jwks_url(current)
  _validate_jwks_dns_for_url(current)
  redirects = 0
  body = b''
  while True:
    try:
      response = _open_jwks_url(current, timeout)
    except urllib.error.HTTPError as error:
      if error.code not in _JWKS_REDIRECT_CODES:
        raise ValueError('jwks fetch failed') from None
      if redirects >= _JWKS_MAX_REDIRECTS:
        raise ValueError('jwks fetch failed') from None
      location = _redirect_location(error)
      if not location:
        raise ValueError('jwks fetch failed') from None
      target = urllib.parse.urljoin(current, location)
      validate_jwks_url(target)
      _validate_jwks_dns_for_url(target)
      redirects += 1
      current = target
      continue
    except ValueError:
      raise
    except Exception:  # pylint: disable=broad-except
      raise ValueError('jwks fetch failed') from None
    final_url = _response_final_url(response)
    if final_url and final_url != current:
      validate_jwks_url(final_url)
      _validate_jwks_dns_for_url(final_url)
    try:
      body = response.read(max_bytes + 1)
    except Exception:  # pylint: disable=broad-except
      raise ValueError('jwks fetch failed') from None
    break
  if len(body) > max_bytes:
    raise ValueError('jwks too large')
  try:
    parsed = json.loads(body.decode('utf-8'))
  except (ValueError, UnicodeDecodeError):
    raise ValueError('invalid jwks json') from None
  if not isinstance(parsed, dict):
    raise ValueError('invalid jwks json')
  return parsed


def load_revocation_list(*, revoked_hashes=None, revoked_file=None):
  '''Build a RevocationList from env and file hashes.

  Args:
    revoked_hashes: Comma-separated hashes, hashes only.
    revoked_file: Path with one hash per line; a missing
      file yields an empty set.

  Returns:
    RevocationList merging both sources.
  '''
  denied = set()
  if revoked_hashes:
    for piece in str(revoked_hashes).split(','):
      text = piece.strip().lower()
      if text:
        denied.add(text)
  if revoked_file:
    try:
      with open(str(revoked_file), encoding='utf-8') as handle:
        for line in handle:
          text = line.strip().lower()
          if text:
            denied.add(text)
    except FileNotFoundError:
      pass
    except OSError:
      pass
  return RevocationList(denied_hashes=denied)


class RevocationList:
  '''Hashed per-principal revocation list.

  Args:
    denied_hashes: Set of ``subject_hash`` digests that
      are denied; revoked on controller checks.
  '''

  def __init__(self, *, denied_hashes=None):
    '''Initialise with denied hashes.

    Args:
      denied_hashes: Set of hashes, hashes only.
    '''
    self._denied = set()
    if denied_hashes is not None:
      for value in denied_hashes:
        text = str(value).strip().lower()
        if text:
          self._denied.add(text)

  def is_revoked(self, *, jti, sub):
    '''Check if jti or sub is revoked by hash.

    Args:
      jti: Clear token identifier, or None.
      sub: Clear subject identifier, or None.

    Returns:
      True when a non-None value hashes to a denied entry.
    '''
    for value in (jti, sub):
      if value is None:
        continue
      if subject_hash(value) in self._denied:
        return True
    return False

  def revoke(self, *, jti, sub):
    '''Revoke one principal by clear jti/sub.

    Args:
      jti: Clear token identifier, or None.
      sub: Clear subject identifier, or None.
    '''
    for value in (jti, sub):
      if value is None:
        continue
      self._denied.add(subject_hash(value))


class OIDCVerifier:
  '''RS256 OIDC token verifier with JWKS caching.

  Args:
    issuer: Expected exact ``iss`` value.
    audience: Expected ``aud`` value.
    jwks_url: JWKS document URL, or None.
    clock: Injectable clock; callable or now() object.
    fetcher: Injectable JWKS fetcher with signature
      ``fetcher(url, *, timeout, max_bytes) -> dict``.
    logger: Injectable logger, else stdlib logging.
    cache_ttl_s: JWKS cache TTL in seconds.
  '''

  def __init__(
    self,
    *,
    issuer,
    audience,
    jwks_url=None,
    clock=None,
    fetcher=None,
    logger=None,
    cache_ttl_s=300,
  ):
    '''Initialise verifier and cache.'''
    self._issuer = str(issuer)
    self._audience = str(audience)
    self._jwks_url = jwks_url
    self._clock = clock
    self._fetcher = fetcher if fetcher is not None else (
      default_fetcher
    )
    self._logger = logger if logger is not None else (
      logging.getLogger(__name__)
    )
    self._cache_ttl_s = int(cache_ttl_s)
    self._cached = None
    self._fetched_at = 0.0
    self._lock = threading.Lock()

  def verify(self, token):
    '''Verify one JWT and return its claims.

    Args:
      token: Compact JWT string.

    Returns:
      Claims dict for a valid RS256 token.

    Raises:
      ValueError: On malformed, expired, wrong iss/aud,
        unknown kid, bad alg, bad signature, or missing sub.
    '''
    text = str(token or '').strip()
    parts = text.split('.')
    if len(parts) != 3 or not text:
      raise ValueError('invalid token')
    head_b64, body_b64, sig_b64 = parts
    header_raw = _b64url_decode(head_b64)
    body_raw = _b64url_decode(body_b64)
    try:
      header = json.loads(header_raw.decode('utf-8'))
      claims = json.loads(body_raw.decode('utf-8'))
    except (ValueError, UnicodeDecodeError):
      raise ValueError('invalid token json') from None
    if not isinstance(header, dict) or not isinstance(
      claims, dict
    ):
      raise ValueError('invalid token json')
    if header.get('alg') != 'RS256':
      raise ValueError('unsupported alg')
    kid = header.get('kid')
    if not kid:
      raise ValueError('missing kid')
    if claims.get('iss') != self._issuer:
      raise ValueError('wrong issuer')
    audience = claims.get('aud')
    if isinstance(audience, list):
      if self._audience not in audience:
        raise ValueError('wrong audience')
    elif audience != self._audience:
      raise ValueError('wrong audience')
    now = _now_from_clock(self._clock)
    exp = claims.get('exp')
    if not isinstance(exp, (int, float)):
      raise ValueError('missing exp')
    if float(exp) <= now:
      raise ValueError('token expired')
    if 'nbf' in claims and isinstance(
      claims.get('nbf'), (int, float)
    ):
      if float(claims['nbf']) > now:
        raise ValueError('token not yet valid')
    jwks = self._get_jwks(now)
    mapping = _parse_jwks(jwks)
    if str(kid) not in mapping:
      raise ValueError('unknown kid')
    modulus, exponent = mapping[str(kid)]
    signature = _b64url_decode(sig_b64)
    signing_input = (head_b64 + '.' + body_b64).encode('ascii')
    digest = hashlib.sha256(signing_input).digest()
    tail = _SHA256_PREFIX + digest
    size = (modulus.bit_length() + 7) // 8
    if size - len(tail) - 3 < 8:
      raise ValueError('key too small')
    if not signature:
      raise ValueError('missing signature')
    sig_int = int.from_bytes(signature, 'big')
    if sig_int >= modulus:
      raise ValueError('bad signature')
    plain = pow(sig_int, exponent, modulus).to_bytes(size, 'big')
    ps_len = size - len(tail) - 3
    expected = (
      b'\x00\x01' + b'\xff' * ps_len + b'\x00' + tail
    )
    if not hmac.compare_digest(plain, expected):
      raise ValueError('bad signature')
    sub = claims.get('sub')
    if not isinstance(sub, str) or not sub.strip():
      raise ValueError('missing sub')
    marker = subject_hash(sub)[:16]
    self._logger.info(
      'oidc_verify_ok sub_hash=%s kid_hash=%s',
      marker,
      subject_hash(str(kid))[:16],
    )
    return claims

  def _get_jwks(self, now):
    '''Return cached or freshly fetched JWKS.

    Args:
      now: Current seconds for TTL comparison.

    Returns:
      JWKS dict.

    Raises:
      ValueError: On missing URL, fetch error, or size.
    '''
    if self._cached is not None:
      if now - self._fetched_at < self._cache_ttl_s:
        return self._cached
    if not self._jwks_url:
      raise ValueError('missing jwks_url')
    validate_jwks_url(str(self._jwks_url))
    with self._lock:
      try:
        fetched = self._fetcher(
          str(self._jwks_url),
          timeout=_JWKS_TIMEOUT,
          max_bytes=_JWKS_MAX_BYTES,
        )
      except ValueError:
        raise
      except Exception:  # pylint: disable=broad-except
        raise ValueError('jwks fetch failed') from None
      blob = json.dumps(
        fetched, separators=(',', ':'), sort_keys=True
      ).encode('utf-8')
      if len(blob) > _JWKS_MAX_BYTES:
        raise ValueError('jwks too large')
      self._cached = fetched
      self._fetched_at = _now_from_clock(self._clock)
      return fetched


def require_scope(claims, scope):
  '''Require one scope in claims or admin superset.

  Args:
    claims: Verified claims mapping.
    scope: Required scope string.

  Raises:
    ValueError: When scope is not granted.
  '''
  granted = _extract_scopes(claims)
  if scope in granted:
    return
  if scope == 'predict:read' and 'predict:admin' in granted:
    return
  raise ValueError('missing scope')


def authenticate(
  request_headers, *, verifier, revocation, shared_key=None
):
  '''Authenticate one /predict request.

  The ``sub`` claim is re-checked here, independently
  of the verifier, so a lax verifier cannot admit a
  token without a usable subject.

  Args:
    request_headers: Header mapping, any case.
    verifier: OIDCVerifier or None when unconfigured.
    revocation: RevocationList or None.
    shared_key: Server shared key, or None.

  Returns:
    Dict with hashed principal only, never clear values.

  Raises:
    ValueError: On any authentication failure.
  '''
  headers = {
    str(name).lower(): str(value)
    for name, value in dict(request_headers).items()
  }
  if verifier is not None:
    auth_header = headers.get('authorization', '')
    if not auth_header.lower().startswith('bearer '):
      raise ValueError('unauthorized')
    token = auth_header[7:].strip()
    if not token:
      raise ValueError('unauthorized')
    claims = verifier.verify(token)
    sub = claims.get('sub')
    if not isinstance(sub, str) or not sub.strip():
      raise ValueError('missing sub')
    jti = claims.get('jti')
    if revocation is not None and revocation.is_revoked(
      jti=jti, sub=sub
    ):
      raise ValueError('unauthorized')
    require_scope(claims, 'predict:read')
    sub_hash = subject_hash(sub) if sub else None
    jti_hash = subject_hash(jti) if jti else None
    granted = sorted(_extract_scopes(claims))
    return {
      'auth_mode': 'oidc',
      'principal_hash': sub_hash,
      'sub_hash': sub_hash,
      'jti_hash': jti_hash,
      'scopes': granted,
    }
  if shared_key:
    provided = headers.get('x-api-key', '')
    if not provided:
      auth_header = headers.get('authorization', '')
      if auth_header.lower().startswith('bearer '):
        provided = auth_header[7:].strip()
    if not provided:
      raise ValueError('unauthorized')
    matched = secrets.compare_digest(
      str(provided).encode('utf-8'),
      str(shared_key).encode('utf-8'),
    )
    if not matched:
      raise ValueError('unauthorized')
    return {
      'auth_mode': 'shared_key',
      'scopes': ['predict:read'],
    }
  return {'auth_mode': 'dev', 'scopes': ['predict:read']}
