#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Minimal FastAPI serving app.

Endpoints: GET /health, POST /predict, GET /metrics.
The /metrics producer (``serving.metrics``) renders
Prometheus text: an ``http_request_duration_seconds``
histogram, ``predict_requests_total`` /
``predict_errors_total`` counters, and ``drift_psi`` /
``drift_ks`` gauges refreshed by the monitoring
pipeline hook. Like /health it is public (aggregate
counts only, no keys or payloads) so Prometheus can
scrape it without credentials.

Models come from an injected loader (preferred) or from the
MLflow registry via explicit config or MODEL_* env vars.

Security: request bodies are capped at 10MB by a streaming
byte counter (chunked bodies without Content-Length are
rejected mid-stream, not only header-checked). When the
FORECAST_API_KEY env var is set, /predict requires the
matching key via X-API-Key or Authorization: Bearer ...
(401 otherwise); /health stays public for probes. Auth
is fail-closed: FORECAST_REQUIRE_AUTH=1 or a production
profile (FORECAST_ENV/ENVIRONMENT/APP_ENV=prod) without
a key raises RuntimeError at startup, a key shorter
than MIN_API_KEY_LENGTH is refused, and a dev startup
without a key logs a WARNING. Binding a non-loopback
host without a valid key is refused (see
:func:`validate_bind_host`), including a non-loopback
FORECAST_BIND_HOST read by the module-level app, so
importing this module cannot bypass the check. /predict
is rate-limited in-app with a token bucket (429 when
exceeded) and every /predict attempt writes a
structured access log with an HMAC key-id, outcome,
and latency (never the raw key or payload), plus a
persisted audit entry (``governance.audit_log``, actor
``serving``) on the bounded ``app.state.audit_log`` list. The access/audit
boundary covers the endpoint (success, 422, 500, 503
and the ``_check_caps`` 413) and the middleware
(401, 429, the streaming body-cap 413, and schema
422s). Validation failures return a generic 422 detail
(``validation failed``); offending field names are
logged server-side only, scrubbed. Security headers
(nosniff, DENY, no-referrer, no-store, restrictive CSP)
are applied to every response, plus Strict-Transport-Security
when the request arrives over TLS. Docker image shipped
(Dockerfile, fail-closed); production promotion gated P11.

Audit limits (accepted, documented): a single shared
FORECAST_API_KEY, so revocation is all-or-nothing;
HMAC key-id uses that key as the HMAC secret (or
``anonymous`` when no server key is configured);
the in-process rate bucket and audit list are
single-process (use a gateway for distributed limits).
'''
# pylint: disable=import-outside-toplevel, broad-exception-raised

import hashlib
import hmac
import ipaddress
import logging
import math
import os
import secrets
import threading
import time

from forecast_ml_framework.core.predictions import (
  check_predictions_finite, materialize_predictions)
from forecast_ml_framework.serving.auth import (
  authenticate, load_revocation_list, OIDCVerifier)
from forecast_ml_framework.serving.metrics import (
  CONTENT_TYPE as _METRICS_CONTENT_TYPE, MetricsRegistry)

try:
  from fastapi import FastAPI, HTTPException, Request, Response
  from pydantic import BaseModel, model_validator
  _IMPORT_ERROR = None
except ImportError as _exc:  # pragma: no cover - missing deps
  FastAPI = None
  BaseModel = object
  HTTPException = Exception
  Request = object  # type: ignore[no-redef]
  Response = object  # type: ignore[no-redef]
  _IMPORT_ERROR = _exc

_LOG = logging.getLogger(__name__)

MAX_RECORDS = 10000
MAX_FEATURE_KEYS = 1000
MAX_BODY_BYTES = 10 * 1024 * 1024

#: Minimum accepted FORECAST_API_KEY length.
MIN_API_KEY_LENGTH = 16

#: Default /predict token-bucket rate (requests/minute).
DEFAULT_RATE_LIMIT_PER_MINUTE = 600

#: Default health-cache TTL in seconds.
DEFAULT_HEALTH_CACHE_TTL = 30.0

#: Maximum entries kept in the bounded per-app audit list.
MAX_AUDIT_ENTRIES = 1000

_OIDC_UNSET = object()


def _truthy(value):
  '''Return True for truthy env spellings (1/true/yes/on).'''
  return str(value or '').strip().lower() in (
      '1', 'true', 'yes', 'on')


def _require_auth_enabled():
  '''Return True when FORECAST_REQUIRE_AUTH is enabled.'''
  return _truthy(os.environ.get('FORECAST_REQUIRE_AUTH'))


def _is_production_profile():
  '''Return True for prod/production env profiles.'''
  for name in ('FORECAST_ENV', 'ENVIRONMENT', 'APP_ENV'):
    if os.environ.get(name, '').strip().lower() in (
        'prod', 'production'):
      return True
  return False


def _is_loopback_host(host):
  '''Return True when ``host`` is an explicit loopback bind.'''
  if host is None:
    return True
  text = str(host).strip()
  if text in ('127.0.0.1', 'localhost', '::1'):
    return True
  try:
    return ipaddress.ip_address(text).is_loopback
  except ValueError:
    return False


def _bind_host_from_env():
  '''Return the bind host from env, or None when unset.

  Returns:
    FORECAST_BIND_HOST value, or None when unset
    or blank. Lets the module-level app apply the
    same non-loopback check as an explicit host.
  '''
  value = os.environ.get('FORECAST_BIND_HOST')
  if value is None or not str(value).strip():
    return None
  return str(value).strip()


def _oidc_env_values():
  '''Return OIDC environment values mapping.'''
  return {
    'issuer': os.environ.get('FORECAST_OIDC_ISSUER'),
    'audience': os.environ.get('FORECAST_OIDC_AUDIENCE'),
    'jwks_url': os.environ.get('FORECAST_OIDC_JWKS_URL'),
  }


def _oidc_partial():
  '''Return True when OIDC env is partially set.'''
  values = _oidc_env_values()
  present = any(
    values[name] and str(values[name]).strip()
    for name in ('issuer', 'audience', 'jwks_url')
  )
  if not present:
    return False
  issuer = values['issuer']
  audience = values['audience']
  complete = bool(
    issuer and str(issuer).strip()
    and audience and str(audience).strip()
  )
  return not complete


def _oidc_configured_from_env():
  '''Return True when issuer and audience are set.'''
  values = _oidc_env_values()
  issuer = values['issuer']
  audience = values['audience']
  return bool(
    issuer and str(issuer).strip()
    and audience and str(audience).strip()
  )


def _check_partial_oidc():
  '''Raise RuntimeError when OIDC env is partial.

  Raises:
    RuntimeError: When any of issuer/audience/jwks_url
      is set without both issuer and audience.
  '''
  if _oidc_partial():
    raise RuntimeError(
      'FORECAST_OIDC_* is incomplete; set '
      'FORECAST_OIDC_ISSUER and FORECAST_OIDC_AUDIENCE '
      'together or clear OIDC env')


def _build_oidc_verifier(fetcher=None, clock=None):
  '''Build an OIDCVerifier from env, or None.

  Args:
    fetcher: Injectable JWKS fetcher for tests.
    clock: Injectable clock for tests.

  Returns:
    OIDCVerifier when issuer/audience are set,
    otherwise None.

  Raises:
    RuntimeError: When OIDC env is partial.
  '''
  _check_partial_oidc()
  values = _oidc_env_values()
  if not _oidc_configured_from_env():
    return None
  issuer = str(values['issuer']).strip()
  audience = str(values['audience']).strip()
  jwks_url = values['jwks_url']
  if jwks_url is not None:
    jwks_url = str(jwks_url).strip() or None
  return OIDCVerifier(
    issuer=issuer,
    audience=audience,
    jwks_url=jwks_url,
    fetcher=fetcher,
    clock=clock,
  )


def _build_revocation_list():
  '''Build a RevocationList from revocation env.

  Returns:
    RevocationList merging hashes env and file.
  '''
  return load_revocation_list(
    revoked_hashes=os.environ.get(
      'FORECAST_OIDC_REVOKED_HASHES'
    ),
    revoked_file=os.environ.get(
      'FORECAST_OIDC_REVOKED_FILE'
    ),
  )


def validate_bind_host(host):
  '''Refuse non-loopback binds without valid auth.

  Args:
    host: Bind host (for example ``--host 0.0.0.0``).

  Raises:
    RuntimeError: When a non-loopback host is bound
      without a configured key of at least
      MIN_API_KEY_LENGTH characters, or when OIDC
      env is partial. Complete OIDC config exempts
      the shared-key requirement.
  '''
  if _is_loopback_host(host):
    return
  _check_partial_oidc()
  if _oidc_configured_from_env():
    return
  key = os.environ.get('FORECAST_API_KEY')
  if not key:
    raise RuntimeError(
        'refusing to bind non-loopback host without '
        'FORECAST_API_KEY; set FORECAST_API_KEY or bind '
        '127.0.0.1')
  if len(key) < MIN_API_KEY_LENGTH:
    raise RuntimeError(
        'FORECAST_API_KEY is too short; minimum length '
        f'is {MIN_API_KEY_LENGTH}')


def _validate_auth_config(verifier=_OIDC_UNSET):
  '''Enforce fail-closed auth config at startup.

  Args:
    verifier: Injected OIDCVerifier for tests, None to
      force no OIDC, or _OIDC_UNSET to read env.

  Raises:
    RuntimeError: When OIDC env is partial, when auth
      is required (or the profile is production)
      without any auth, or when a shared key is too
      short. OIDC principals are exempt from the
      shared-key length and production checks.
  '''
  _check_partial_oidc()
  if verifier is _OIDC_UNSET:
    if _oidc_configured_from_env():
      return
  elif verifier is not None:
    return
  key = os.environ.get('FORECAST_API_KEY')
  required = _require_auth_enabled() or _is_production_profile()
  if required and not key:
    raise RuntimeError(
        'FORECAST_API_KEY is required when '
        'FORECAST_REQUIRE_AUTH=1 or in production; '
        'refusing to start without authentication')
  if key and len(key) < MIN_API_KEY_LENGTH:
    raise RuntimeError(
        'FORECAST_API_KEY is too short; minimum length '
        f'is {MIN_API_KEY_LENGTH}')
  if not key and not required:
    _LOG.warning(
        'FORECAST_API_KEY is not set; /predict is '
        'unauthenticated (dev mode). Set FORECAST_API_KEY '
        'in production.')


def _log_exception(event, exc, level=logging.ERROR):
  '''Log a failure by event code and type only.

  The exception message is never logged and no
  traceback is attached, so secrets, paths, and URIs
  carried by the exception cannot reach the log. The
  scrubbed message is deliberately not logged either:
  scrubbing can leave path prefixes intact.
  '''
  _LOG.log(
      level, '%s code=%s type=%s', event, event,
      type(exc).__name__)


def _provided_key(headers):
  '''Extract the provided API key from a header mapping.'''
  provided = headers.get('x-api-key', '')
  if provided:
    return provided
  auth = headers.get('authorization', '')
  if auth.lower().startswith('bearer '):
    return auth[7:].strip()
  return ''


def _keys_match(provided, expected):
  '''Return True when a provided key matches the expected key.

  Args:
    provided: Client-supplied key (may contain non-ASCII
      characters from latin-1 decoded header bytes).
    expected: Server-configured key.

  Returns:
    True only on an exact match. Comparison is done on
    UTF-8 bytes: str ``compare_digest`` raises TypeError
    on non-ASCII input, which previously crashed a
    non-ASCII header byte to a 500 before any audit
    entry. Bytes never raise and non-ASCII simply
    fails to match an ASCII expected key.
  '''
  if not provided or not expected:
    return False
  return secrets.compare_digest(
      str(provided).encode('utf-8'),
      str(expected).encode('utf-8'))


def _hmac_key_id(provided):
  '''Return a truncated HMAC key-id for logging.

  The HMAC secret is the server FORECAST_API_KEY itself
  (``anonymous`` when no server key is configured), so the
  key-id differs per client key and cannot be reversed to
  the raw key. Empty provided keys yield ``anonymous``.
  Precomputed ``oidc-`` key-ids pass through unchanged so
  OIDC principals are audited by hashed subject, never
  by raw token.
  '''
  if not provided:
    return 'anonymous'
  text = str(provided)
  if text.startswith('oidc-'):
    return text
  secret = os.environ.get('FORECAST_API_KEY') or 'anonymous'
  return hmac.new(
      secret.encode('utf-8'), text.encode('utf-8'),
      hashlib.sha256).hexdigest()[:16]


def _audit_predict(provided, outcome, latency, records=0,
                   audit_list=None):
  '''Persist one ``predict`` audit entry (actor ``serving``).

  Details carry only the HMAC key-id, outcome, latency and
  record count -- never the raw key or payload. The list
  is bounded (oldest entries dropped). Audit failures are
  logged by code+type only and never break a response.
  '''
  if audit_list is None:
    return
  try:
    # pylint: disable=import-outside-toplevel
    from forecast_ml_framework.governance import audit_log
    details = {
        'key_id_hmac': _hmac_key_id(provided),
        'outcome': outcome,
        'latency_ms': round(float(latency) * 1000.0, 3),
        'records': int(records),
    }
    audit_log(
        event='predict', actor='serving', details=details,
        log=audit_list)
    while len(audit_list) > MAX_AUDIT_ENTRIES:
      del audit_list[0]
  except Exception as exc:  # pylint: disable=broad-except
    _log_exception('audit_failed', exc, level=logging.WARNING)


def _log_access(provided, outcome, latency, records=0,
                audit_list=None):
  '''Write the structured /predict access log entry.

  Only an HMAC key-id, outcome, latency, and record
  count are logged; the raw key and payload never are.
  Boundary: every /predict attempt in the endpoint and
  the middleware, including the ``_check_caps`` 413 and
  the streaming body-cap 413, plus 401/429. When
  ``audit_list`` is given a matching audit entry is
  persisted via :func:`_audit_predict`.
  '''
  _LOG.info(
      'predict_access key_id=%s outcome=%s latency_ms=%s '
      'records=%s', _hmac_key_id(provided), outcome,
      round(float(latency) * 1000.0, 3), int(records))
  _audit_predict(provided, outcome, latency, records,
                 audit_list=audit_list)


def _log_validation_error(exc):
  '''Log a validation failure server-side only (scrubbed).

  Field names are kept out of the 422 response body but
  remain available to operators in the server log, after
  scrubbing. Control characters are stripped so a field
  name containing a newline cannot forge a second log
  line (``scrub_text`` preserves newlines). Never raises.
  '''
  try:
    # pylint: disable=import-outside-toplevel
    from forecast_ml_framework.governance.scrub import scrub_text
    detail = scrub_text(str(exc))
  except Exception:  # pylint: disable=broad-except
    detail = 'validation failed'
  detail = ''.join(
      char for char in detail
      if ord(char) >= 32 and ord(char) != 127)
  _LOG.warning(
      'validation_error code=%s type=%s detail=%s',
      'validation_failed', type(exc).__name__, detail)


def _rate_limit_per_minute(explicit=None):
  '''Return the effective /predict rate limit.

  Args:
    explicit: Caller-supplied limit, winning over env.

  Returns:
    Requests per minute.

  Raises:
    ValueError: When the configured limit is invalid.
  '''
  raw = explicit
  if raw is None:
    raw = os.environ.get('FORECAST_RATE_LIMIT_PER_MINUTE')
  if raw is None or str(raw).strip() == '':
    return DEFAULT_RATE_LIMIT_PER_MINUTE
  try:
    limit = int(str(raw), 10)
  except (TypeError, ValueError):
    raise ValueError('invalid rate limit') from None
  if limit <= 0:
    raise ValueError('invalid rate limit')
  return limit


def _health_cache_ttl(explicit=None):
  '''Return the effective health-cache TTL in seconds.

  Args:
    explicit: Caller-supplied TTL, winning over env.

  Returns:
    TTL in seconds.
  '''
  raw = explicit
  if raw is None:
    raw = os.environ.get('FORECAST_HEALTH_CACHE_TTL')
  if raw is None or str(raw).strip() == '':
    return DEFAULT_HEALTH_CACHE_TTL
  return float(str(raw))


class _TokenBucket:
  '''Thread-safe token bucket for /predict rate limiting.

  Args:
    per_minute: Sustained requests per minute. The
      burst capacity equals the per-minute rate.
  '''

  def __init__(self, per_minute):
    '''Initialise bucket from a per-minute rate.

    Args:
      per_minute: Sustained requests per minute. The
        burst capacity equals the per-minute rate.
    '''
    self._per_minute = float(per_minute)
    self._capacity = float(per_minute)
    self._tokens = float(per_minute)
    self._updated = time.monotonic()
    self._lock = threading.Lock()

  def allow(self):
    '''Consume one token, returning False when empty.'''
    with self._lock:
      now = time.monotonic()
      elapsed = now - self._updated
      self._updated = now
      refill = elapsed * self._per_minute / 60.0
      self._tokens = min(
          self._capacity, self._tokens + refill)
      if self._tokens < 1.0:
        return False
      self._tokens -= 1.0
      return True


if _IMPORT_ERROR is None:

  class HealthResponse(BaseModel):
    '''Health response model.'''

    status: str
    model_loaded: bool = False

  class PredictRequest(BaseModel):
    '''Predict request model.'''

    features: dict | None = None
    records: list[dict] | None = None

    @model_validator(mode='after')
    def _require_input(self):
      '''Require features or records.'''
      if not self.features and not self.records:
        raise ValueError('features or records is required')
      return self

  class PredictResponse(BaseModel):
    '''Predict response model.'''

    predictions: list

else:

  class HealthResponse:
    '''Placeholder when fastapi is missing.'''

  class PredictRequest:
    '''Placeholder when fastapi is missing.'''

  class PredictResponse:
    '''Placeholder when fastapi is missing.'''


def _to_list(value):
  '''Convert model output to a plain list.'''
  if isinstance(value, dict):
    return list(value.values())
  if hasattr(value, 'tolist'):
    return list(value.tolist())
  return list(value)


_materialize_predictions = materialize_predictions
_check_predictions_finite = check_predictions_finite


def _records_from(request):
  '''Normalize request to a list of record dicts.'''
  if request.records:
    return request.records
  return [request.features or {}]


def _check_caps(records):
  '''Reject oversized payloads with HTTP 413.

  Args:
    records: List of record dicts.

  Raises:
    HTTPException: 413 when caps are exceeded.
  '''
  if len(records) > MAX_RECORDS:
    raise HTTPException(status_code=413,
                        detail='too many records')
  for record in records:
    if len(record) > MAX_FEATURE_KEYS:
      raise HTTPException(status_code=413,
                          detail='too many feature keys')


def _validate_records(model, records):
  '''Validate dict[str, float]-ish records strictly.

  Args:
    model: Model (may carry ``feature_names_in_``).
    records: List of record dicts.

  Returns:
    Rows as lists of floats.

  Raises:
    ValueError: 422-ready message naming the offending field
      for missing, nested, wrongly typed, or non-finite values.
  '''
  if not records:
    raise ValueError('features or records is required')
  keys = list(records[0].keys())
  if not keys:
    raise ValueError('features or records is required')
  _names = getattr(model, 'feature_names_in_', None)
  if _names is None or len(_names) == 0:
    required = keys
  else:
    # feature_names_in_ may be a numpy array / list.
    required = list(_names.tolist() if hasattr(
        _names, 'tolist') else _names)
    required = [str(name) for name in required]
  # Validate EVERY record against required, not only the
  # first: a later record carrying an unknown key must be
  # rejected even when the first record is clean.
  for record in records:
    if not record:
      raise ValueError('features or records is required')
    unknown = [name for name in record if name not in required]
    if unknown:
      raise ValueError(f'unknown feature: {unknown}')
  rows = []
  for record in records:
    for name in required:
      if name not in record:
        raise ValueError(f'missing feature: {name}')
    row = []
    for name in required:
      value = record[name]
      if isinstance(value, bool) or not isinstance(
          value, (int, float)):
        raise ValueError(
            f'feature {name} must be a finite number')
      try:
        number = float(value)
      except OverflowError:
        raise ValueError(
            f'feature {name} must be a finite number') from None
      if not math.isfinite(number):
        raise ValueError(
            f'feature {name} must be a finite number')
      row.append(number)
    rows.append(row)
  return rows


def _predict(model, records):
  '''Run model prediction on records.

  Validation (``_validate_records``) is caller-separated:
  its ValueError is a 422. Errors from ``model.predict``
  (including ValueError) propagate for the caller to map
  to a generic 500.
  '''
  rows = _validate_records(model, records)
  if hasattr(model, 'predict'):
    raw = model.predict(rows)
  elif callable(model):
    raw = model(records)
  else:
    raise TypeError('model must have predict() or be callable')
  raw = _materialize_predictions(raw)
  _check_predictions_finite(raw)
  predictions = _to_list(raw)
  _check_predictions_finite(predictions)
  return predictions


def _resolve_loader(model_loader, registry_uri, model_name, alias,
                    artifact_path=None):
  '''Resolve the effective loader, preferring injection.

  Falls back to an explicit or MODEL_ARTIFACT_PATH file
  loader, then explicit registry config or MODEL_NAME,
  then a degraded loader that reports the model as not
  configured. Never raises for a missing loader.
  '''
  if model_loader is not None:
    return model_loader
  artifact_path = artifact_path or os.environ.get(
      'MODEL_ARTIFACT_PATH')
  if artifact_path:
    from forecast_ml_framework.serving.model_loader import (  # noqa: E501
      FileModelLoader,
    )
    return FileModelLoader(artifact_path)
  registry_uri = registry_uri or os.environ.get('MODEL_REGISTRY_URI')
  model_name = model_name or os.environ.get('MODEL_NAME')
  alias = alias or os.environ.get('MODEL_ALIAS') or 'production'
  if not model_name:
    from forecast_ml_framework.serving.model_loader import (  # noqa: E501
      DegradedLoader,
    )
    return DegradedLoader()
  from forecast_ml_framework.serving.model_loader import (  # noqa: E501
    RegistryModelLoader,
  )
  return RegistryModelLoader(registry_uri, model_name, alias=alias)


# Re-exported for callers/tests that look for the degraded
# loader next to the app factory.
try:
  from forecast_ml_framework.serving.model_loader import (  # noqa: E501
    DegradedLoader,
    ModelCorruptError,
    ModelNotConfiguredError,
    ModelNotFoundError,
  )
except Exception:  # pragma: no cover - import guard
  class DegradedLoader:  # type: ignore[no-redef]
    '''Fallback degraded loader.'''

    def __call__(self):
      '''Raise ModelNotConfiguredError.'''
      raise ModelNotConfiguredError('model not configured')

  class ModelNotConfiguredError(RuntimeError):  # type: ignore[no-redef]
    '''Fallback not-configured error.'''

  class ModelNotFoundError(RuntimeError):  # type: ignore[no-redef]
    '''Fallback not-found error.'''

  class ModelCorruptError(RuntimeError):  # type: ignore[no-redef]
    '''Fallback corrupt error.'''


def _load_model(model_loader):
  '''Return the model from a loader or model object.'''
  return model_loader() if callable(model_loader) else model_loader


SECURITY_HEADERS = {
    'x-content-type-options': 'nosniff',
    'x-frame-options': 'DENY',
    'referrer-policy': 'no-referrer',
    'cache-control': 'no-store',
    'content-security-policy': 'default-src \'none\'; frame-ancestors \'none\'',
    'x-permitted-cross-domain-policies': 'none',
}


def _header_map(scope):
  '''Return lower-cased header dict from an ASGI scope.'''
  return {
      key.decode('latin-1').lower(): value.decode('latin-1')
      for key, value in scope.get('headers', [])
  }


def _has_duplicate_auth_headers(scope):
  '''Return True when an auth header appears more than once.

  Args:
    scope: ASGI connection scope.

  Returns:
    True when ``x-api-key`` or ``authorization`` occurs
    more than once (case-insensitive) in the raw header
    list, so a single canonical key cannot be picked
    unambiguously from duplicate credentials.
  '''
  counts = {}
  for pair in scope.get('headers', []):
    name = pair[0].decode('latin-1').lower()
    if name in ('x-api-key', 'authorization'):
      counts[name] = counts.get(name, 0) + 1
      if counts[name] > 1:
        return True
  return False


def _security_headers(scope):
  '''Return security headers, adding HSTS over TLS.

  HSTS is sent when the ASGI scheme is https, when
  X-Forwarded-Proto is https (TLS-terminating proxy),
  or when FORECAST_TLS_ENABLED=1. Plain HTTP responses
  omit it, as documented here and in the module
  docstring; a TLS-terminating proxy should set
  FORECAST_TLS_ENABLED=1 so HSTS is always sent.
  '''
  headers = dict(SECURITY_HEADERS)
  raw_headers = _header_map(scope)
  scheme = str(scope.get('scheme', 'http')).lower()
  forwarded = raw_headers.get('x-forwarded-proto', '').lower()
  if scheme == 'https' or forwarded == 'https' or _truthy(
          os.environ.get('FORECAST_TLS_ENABLED')):
    headers['strict-transport-security'] = (
        'max-age=31536000; includeSubDomains')
  return headers


async def _send_json(send, status_code, detail, headers=None):
  '''Send a JSON error response (with security headers).'''
  import json as _json
  body = _json.dumps({'detail': detail}).encode('utf-8')
  raw_headers = [(b'content-type', b'application/json'),
                 (b'content-length', str(len(body)).encode('latin-1'))]
  raw_headers.extend(
      (k.encode('latin-1'), v.encode('latin-1'))
      for k, v in (headers or SECURITY_HEADERS).items())
  await send({
      'type': 'http.response.start',
      'status': status_code,
      'headers': raw_headers,
  })
  await send({'type': 'http.response.body', 'body': body})


class _CapAuthHeadersMiddleware:
  '''Pure ASGI: body cap, auth, rate limit, headers.

  The body cap counts bytes as ``http.request`` messages
  stream in (works for chunked bodies with no
  Content-Length) and aborts with 413 as soon as the
  running total passes MAX_BODY_BYTES, instead of
  buffering the whole body. The 413 branch writes an
  access-log and audit entry (outcome
  ``payload_too_large``) before responding. Bodies
  within the cap are replayed to the app unchanged.
  Schema 422s (FastAPI/pydantic) never reach the
  endpoint, so the middleware logs them as
  ``validation_error`` (access + audit) on the
  response start, deduped via the per-request scope
  marker (C4-R1) so an endpoint 422 is logged
  exactly once.
  When an OIDC verifier is configured, every path
  except /health requires a valid Bearer OIDC token
  (401 otherwise); the shared key is not consulted.
  Otherwise when FORECAST_API_KEY is set, every path
  except /health requires it (401, constant-time).
  /predict is token-bucket rate-limited (429 when the
  bucket is empty); use a gateway for distributed
  limits in production.

  Args:
    app: Downstream ASGI application.
    rate_limit_per_minute: /predict rate limit,
      overriding FORECAST_RATE_LIMIT_PER_MINUTE.
    audit_log: Bounded list receiving per-predict
      audit entries (shared with ``app.state``).
    oidc_verifier: OIDCVerifier or None.
    oidc_revocation: RevocationList or None.
    metrics_registry: MetricsRegistry recording
      HTTP timings and /predict counters; a new
      registry is created when omitted.
  '''

  def __init__(
    self, app, rate_limit_per_minute=None, audit_log=None,
    oidc_verifier=None, oidc_revocation=None,
    metrics_registry=None
  ):
    '''Initialise app, rate limit, and audit sink.

    Args:
      app: Downstream ASGI application.
      rate_limit_per_minute: /predict rate limit,
        overriding FORECAST_RATE_LIMIT_PER_MINUTE.
      audit_log: Bounded list receiving per-predict
        audit entries (shared with ``app.state``).
      oidc_verifier: OIDCVerifier or None.
      oidc_revocation: RevocationList or None.
      metrics_registry: MetricsRegistry recording
        HTTP timings and /predict counters; a new
        registry is created when omitted.
    '''
    self.app = app
    self._bucket = _TokenBucket(
        _rate_limit_per_minute(rate_limit_per_minute))
    self._audit_log = audit_log
    self._oidc_verifier = oidc_verifier
    self._oidc_revocation = oidc_revocation
    self._metrics = (
      metrics_registry if metrics_registry is not None
      else MetricsRegistry())

  def _record_metrics(self, path, code, started):
    '''Record timing and /predict counters for one request.

    Args:
      path: ASGI request path.
      code: HTTP status code of the response.
      started: ``time.monotonic()`` start timestamp.

    Returns:
      None. /metrics itself is never recorded (it
      would pollute latency series with scrapes),
      and unknown paths are bucketed as ``unknown``
      so arbitrary URLs cannot explode label
      cardinality.
    '''
    if path == '/metrics':
      return
    route = path if path in ('/health', '/predict') else 'unknown'
    self._metrics.observe_http(
      time.monotonic() - started, route, code)
    if path == '/predict':
      self._metrics.record_predict(code)

  async def __call__(self, scope, receive, send):
    '''Handle one ASGI request through auth and limits.

    Args:
      scope: ASGI connection scope.
      receive: ASGI receive callable.
      send: ASGI send callable.

    Returns:
      None.
    '''
    if scope.get('type') != 'http':
      await self.app(scope, receive, send)
      return
    started = time.monotonic()
    sec_headers = _security_headers(scope)
    expected = os.environ.get('FORECAST_API_KEY')
    headers = _header_map(scope)
    provided = _provided_key(headers)
    auth_result = None
    needs_auth = (
      self._oidc_verifier is not None or bool(expected)
    )
    if scope.get('path') in ('/health', '/metrics'):
      needs_auth = False
    if needs_auth:
      if _has_duplicate_auth_headers(scope):
        log_key = (
          'oidc-anonymous'
          if self._oidc_verifier is not None
          else provided
        )
        if scope.get('path') == '/predict':
          _log_access(
              log_key, 'unauthorized',
              time.monotonic() - started,
              audit_list=self._audit_log)
        self._record_metrics(
            scope.get('path'), 401, started)
        await _send_json(
            send, 401, 'unauthorized', headers=sec_headers)
        return
      try:
        auth_result = authenticate(
          headers,
          verifier=self._oidc_verifier,
          revocation=self._oidc_revocation,
          shared_key=expected,
        )
      except ValueError:
        log_key = (
          'oidc-anonymous'
          if self._oidc_verifier is not None
          else provided
        )
        if scope.get('path') == '/predict':
          _log_access(
              log_key, 'unauthorized',
              time.monotonic() - started,
              audit_list=self._audit_log)
        self._record_metrics(
            scope.get('path'), 401, started)
        await _send_json(
            send, 401, 'unauthorized', headers=sec_headers)
        return
      if self._oidc_verifier is not None:
        principal = auth_result.get('principal_hash')
        if principal:
          provided = 'oidc-' + str(principal)[:16]
        else:
          provided = 'oidc-anonymous'
    # C13-03: publish the canonical (last-wins) key the
    # middleware authenticated so the endpoint audits
    # exactly this key, not a first-wins dict(headers).
    scope['forecast_provided_key'] = provided
    scope['forecast_auth_mode'] = (
      auth_result.get('auth_mode') if auth_result else None
    )
    if scope.get('path') == '/predict':
      if not self._bucket.allow():
        _log_access(
            provided, 'rate_limited',
            time.monotonic() - started,
            audit_list=self._audit_log)
        self._record_metrics(
            scope.get('path'), 429, started)
        await _send_json(
            send, 429, 'rate limit exceeded',
            headers=sec_headers)
        return
    total = 0
    chunks = []
    while True:
      message = await receive()
      if message.get('type') == 'http.disconnect':
        return
      if message.get('type') != 'http.request':
        break
      chunk = message.get('body') or b''
      total += len(chunk)
      if total > MAX_BODY_BYTES:
        if scope.get('path') == '/predict':
          _log_access(
              provided, 'payload_too_large',
              time.monotonic() - started,
              audit_list=self._audit_log)
        self._record_metrics(
            scope.get('path'), 413, started)
        await _send_json(
            send, 413, 'request body too large',
            headers=sec_headers)
        return
      chunks.append(chunk)
      if not message.get('more_body'):
        break
    body = b''.join(chunks)
    replayed = False
    response_status = {'code': None}

    async def _replay_receive():
      '''Replay the buffered body to the downstream app.

      Returns:
        The single replayed ``http.request`` message,
        then ``http.disconnect`` on later calls.
      '''
      nonlocal replayed
      if not replayed:
        replayed = True
        return {
            'type': 'http.request', 'body': body,
            'more_body': False,
        }
      return {'type': 'http.disconnect'}

    async def _send_with_headers(message):
      '''Send a message with security headers attached.

      Args:
        message: ASGI response message.

      Returns:
        None.
      '''
      if message.get('type') == 'http.response.start':
        status = message.get('status')
        response_status['code'] = status
        if status == 422 and scope.get('path') == '/predict':
          # C4-R1: pydantic schema 422 never reached the
          # endpoint, so record repudiation here unless
          # the endpoint already logged this request.
          if not scope.get('forecast_audit_logged', False):
            _log_access(
                provided, 'validation_error',
                time.monotonic() - started,
                audit_list=self._audit_log)
            scope['forecast_audit_logged'] = True
        existing = [
            (k, v) for k, v in message.get('headers', [])
            if k.decode('latin-1').lower() not in sec_headers
        ]
        existing.extend(
            (k.encode('latin-1'), v.encode('latin-1'))
            for k, v in sec_headers.items())
        message = dict(message)
        message['headers'] = existing
      await send(message)

    await self.app(scope, _replay_receive, _send_with_headers)
    if response_status['code'] is not None:
      self._record_metrics(
          scope.get('path'), response_status['code'], started)


def create_app(model_loader=None, registry_uri=None, model_name=None,
               alias=None, artifact_path=None, host=None,
               rate_limit_per_minute=None, health_cache_ttl=None,
               oidc_verifier=_OIDC_UNSET,
               oidc_revocation=_OIDC_UNSET, oidc_fetcher=None,
               oidc_clock=None, metrics_registry=None):
  '''Create FastAPI app with injected or registry model loader.

  Never raises for a missing loader: without an injected
  loader, MODEL_ARTIFACT_PATH, or registry config, the
  app runs in degraded mode (health reports degraded and
  /predict returns 503). Auth is fail-closed (see
  :func:`_validate_auth_config`). Complete OIDC config
  (explicit verifier or FORECAST_OIDC_ISSUER/AUDIENCE
  env) makes OIDC authoritative and exempts the
  shared-key checks; the shared key is not consulted.

  Args:
    model_loader: Callable returning a model, or a model.
      Preferred when given (back-compat); registry config
      and MODEL_* env vars are ignored in that case.
    registry_uri: MLflow registry URI (or MODEL_REGISTRY_URI).
    model_name: Registered model name (or MODEL_NAME).
    alias: Registry alias (or MODEL_ALIAS, default
      ``production``).
    artifact_path: File-loader artifact path (or
      MODEL_ARTIFACT_PATH); takes precedence over
      registry config.
    host: Bind host; non-loopback binds require valid
      auth (see :func:`validate_bind_host`). When
      None, FORECAST_BIND_HOST is consulted so the
      module-level app cannot bypass the check.
    rate_limit_per_minute: /predict rate limit,
      overriding FORECAST_RATE_LIMIT_PER_MINUTE.
    health_cache_ttl: Health-cache TTL in seconds,
      overriding FORECAST_HEALTH_CACHE_TTL.
    oidc_verifier: Injected OIDCVerifier for tests,
      None to force no OIDC, or default to read env.
    oidc_revocation: Injected RevocationList for tests,
      None to disable, or default to read env.
    oidc_fetcher: JWKS fetcher for env-built verifier.
    oidc_clock: Clock for env-built verifier.
    metrics_registry: MetricsRegistry for /metrics;
      a new one is created when omitted. The drift
      hook is ``metrics_registry.update_from_...``
      (see ``serving.metrics``); reach it via
      ``app.state.metrics_registry``.

  Returns:
    Configured FastAPI application.

  Raises:
    ImportError: If fastapi or pydantic is not installed.
    RuntimeError: When fail-closed auth or bind-host
      validation refuses startup.
  '''
  _validate_auth_config(oidc_verifier)
  if oidc_verifier is _OIDC_UNSET:
    oidc_verifier = _build_oidc_verifier(
      fetcher=oidc_fetcher, clock=oidc_clock
    )
  if oidc_revocation is _OIDC_UNSET:
    oidc_revocation = _build_revocation_list()
  effective_host = host
  if effective_host is None:
    effective_host = _bind_host_from_env()
  if effective_host is not None and oidc_verifier is None:
    validate_bind_host(effective_host)
  model_loader = _resolve_loader(
      model_loader, registry_uri, model_name, alias,
      artifact_path=artifact_path)
  if _IMPORT_ERROR is not None:
    raise ImportError(
      'fastapi and pydantic are required to serve: install them '
      'with pip install fastapi pydantic'
    ) from _IMPORT_ERROR
  ttl = _health_cache_ttl(health_cache_ttl)
  app = FastAPI(title='forecast-ml serving')
  if metrics_registry is None:
    metrics_registry = MetricsRegistry()
  # Bounded audit list shared by the middleware (413/401/429
  # branches) and the endpoint; exposed as app.state.audit_log.
  _audit_entries = []
  app.add_middleware(
      _CapAuthHeadersMiddleware,
      rate_limit_per_minute=rate_limit_per_minute,
      audit_log=_audit_entries,
      oidc_verifier=oidc_verifier,
      oidc_revocation=oidc_revocation,
      metrics_registry=metrics_registry)
  app.state.audit_log = _audit_entries
  app._audit_log = _audit_entries
  app.state.metrics_registry = metrics_registry
  # Health probe cache on the app instance: dict with
  # 'result' -> HealthResponse or None when not yet
  # probed, and 'probed_at' -> monotonic timestamp of
  # the last probe. The cache refreshes after its TTL
  # and on every successful /predict.
  _health_cache = {'result': None, 'probed_at': 0.0}
  app.state.health_cache = _health_cache
  app._health_cache = _health_cache

  def _refresh_health_cache(result):
    '''Store a health result with the current timestamp.'''
    _health_cache['result'] = result
    _health_cache['probed_at'] = time.monotonic()

  @app.get('/health', response_model=HealthResponse)
  def health() -> HealthResponse:
    '''Return service health and whether the model loads.'''
    fresh = (
        _health_cache['result'] is not None
        and time.monotonic() - _health_cache['probed_at'] < ttl)
    if fresh:
      return _health_cache['result']
    try:
      _load_model(model_loader)
    except Exception:  # pylint: disable=broad-exception-caught
      _refresh_health_cache(HealthResponse(
        status='degraded', model_loaded=False))
    else:
      _refresh_health_cache(HealthResponse(
        status='ok', model_loaded=True))
    return _health_cache['result']

  @app.post('/predict', response_model=PredictResponse)
  def predict(
      request: PredictRequest,
      http_request: Request) -> PredictResponse:
    '''Return predictions for supplied features or records.'''
    started = time.monotonic()
    # C13-03: reuse the canonical key the middleware
    # authenticated (stored in scope), falling back to
    # the same last-wins _header_map extraction. Never
    # use dict(headers), which is first-wins and would
    # audit a different key than the one authenticated.
    provided = http_request.scope.get('forecast_provided_key')
    if provided is None:
      provided = _provided_key(_header_map(http_request.scope))
    records = _records_from(request)
    try:
      _check_caps(records)
    except HTTPException:
      # C2-F5/R-04: the _check_caps 413 is inside the
      # access/audit boundary.
      _log_access(provided, 'payload_too_large',
                  time.monotonic() - started, len(records),
                  audit_list=_audit_entries)
      http_request.scope['forecast_audit_logged'] = True
      raise
    try:
      model = _load_model(model_loader)
    except ModelNotConfiguredError as exc:
      _log_exception(
          'model_not_configured', exc, level=logging.WARNING)
      _log_access(provided, 'model_not_configured',
                  time.monotonic() - started, len(records),
                  audit_list=_audit_entries)
      http_request.scope['forecast_audit_logged'] = True
      raise HTTPException(
          status_code=503, detail='model not configured') from None
    except ModelNotFoundError as exc:
      _log_exception(
          'model_not_found', exc, level=logging.WARNING)
      _log_access(provided, 'model_not_found',
                  time.monotonic() - started, len(records),
                  audit_list=_audit_entries)
      http_request.scope['forecast_audit_logged'] = True
      raise HTTPException(
          status_code=503,
          detail='model artifact not found') from None
    except ModelCorruptError as exc:
      _log_exception('model_corrupt', exc)
      _log_access(provided, 'model_corrupt',
                  time.monotonic() - started, len(records),
                  audit_list=_audit_entries)
      http_request.scope['forecast_audit_logged'] = True
      raise HTTPException(
          status_code=500, detail='model failed to load') from None
    except Exception as exc:  # pylint: disable=broad-exception-caught
      _log_exception('model_load_failed', exc)
      _log_access(provided, 'model_load_failed',
                  time.monotonic() - started, len(records),
                  audit_list=_audit_entries)
      http_request.scope['forecast_audit_logged'] = True
      raise HTTPException(
          status_code=500, detail='model failed to load') from None
    # Validation separate from model execution: validation
    # ValueError -> generic 422; model ValueError/other -> 500.
    try:
      rows = _validate_records(model, records)
    except ValueError as exc:
      # R-09: generic client detail; field names are logged
      # scrubbed server-side only (see _log_validation_error).
      _log_validation_error(exc)
      _log_access(provided, 'validation_error',
                  time.monotonic() - started, len(records),
                  audit_list=_audit_entries)
      http_request.scope['forecast_audit_logged'] = True
      raise HTTPException(status_code=422,
                          detail='validation failed') from None
    try:
      if hasattr(model, 'predict'):
        raw = model.predict(rows)
      elif callable(model):
        raw = model(records)
      else:
        raise TypeError('model must have predict() or be callable')
      raw = _materialize_predictions(raw)
      _check_predictions_finite(raw)
      predictions = _to_list(raw)
      _check_predictions_finite(predictions)
    except Exception as exc:  # pylint: disable=broad-exception-caught
      # Log code+type only; never leak detail to client/log.
      _log_exception('prediction_failed', exc)
      _log_access(provided, 'prediction_failed',
                  time.monotonic() - started, len(records),
                  audit_list=_audit_entries)
      http_request.scope['forecast_audit_logged'] = True
      raise HTTPException(
          status_code=500, detail='prediction failed') from None
    _refresh_health_cache(HealthResponse(
        status='ok', model_loaded=True))
    _log_access(provided, 'success',
                time.monotonic() - started, len(records),
                audit_list=_audit_entries)
    http_request.scope['forecast_audit_logged'] = True
    return PredictResponse(predictions=predictions)

  @app.get('/metrics')
  def metrics():
    '''Render Prometheus metrics in text format.'''
    return Response(
      content=metrics_registry.render(),
      media_type=_METRICS_CONTENT_TYPE)

  return app


def create_app_from_registry(registry_uri, model_name,
                             alias='production', tracking_uri=None):
  '''Create app loading its model from the MLflow registry.

  Args:
    registry_uri: MLflow registry URI, or None for ambient.
    model_name: Registered model name.
    alias: Registry alias, default ``production``.
    tracking_uri: MLflow tracking URI, kept separate
      from ``registry_uri``.

  Returns:
    Configured FastAPI application.
  '''
  from forecast_ml_framework.serving.model_loader import (  # noqa: E501
    RegistryModelLoader,
  )
  loader = RegistryModelLoader(
      registry_uri, model_name, alias=alias,
      tracking_uri=tracking_uri)
  return create_app(loader)


# Module-level app for uvicorn (``app:app``). Creation
# is fail-closed: a non-loopback FORECAST_BIND_HOST
# without a key raises RuntimeError here, so importing
# this module (and therefore serving) is refused.
try:
  app = create_app()
except ImportError:  # pragma: no cover - missing fastapi
  app = None
