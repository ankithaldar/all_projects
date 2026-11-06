#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Minimal FastAPI serving app.

Endpoints: GET /health, POST /predict.

Models come from an injected loader (preferred) or from the
MLflow registry via explicit config or MODEL_* env vars.

Security: request bodies are capped at 10MB by a streaming
byte counter (chunked bodies without Content-Length are
rejected mid-stream, not only header-checked). When the
FORECAST_API_KEY env var is set, /predict requires the
matching key via X-API-Key or Authorization: Bearer ...
(401 otherwise); /health stays public for probes. API-key
and security headers (nosniff, DENY, no-referrer,
no-store, restrictive CSP) are applied to every response.
No Docker is provided.
'''
# pylint: disable=import-outside-toplevel, broad-exception-raised

import logging
import math
import os

try:
  from fastapi import FastAPI, HTTPException
  from pydantic import BaseModel, model_validator
  _IMPORT_ERROR = None
except ImportError as _exc:  # pragma: no cover - missing deps
  FastAPI = None
  BaseModel = object
  HTTPException = Exception
  _IMPORT_ERROR = _exc

_LOG = logging.getLogger(__name__)

MAX_RECORDS = 10000
MAX_FEATURE_KEYS = 1000
MAX_BODY_BYTES = 10 * 1024 * 1024


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
  if hasattr(value, 'tolist'):
    return list(value.tolist())
  return list(value)


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


def _check_predictions_finite(predictions):
  '''Raise ValueError when any numeric prediction is non-finite.'''
  stack = list(predictions)
  while stack:
    value = stack.pop()
    if isinstance(value, (list, tuple)):
      stack.extend(value)
    elif isinstance(value, bool):
      continue
    elif isinstance(value, (int, float)):
      try:
        number = float(value)
      except OverflowError:
        raise ValueError(
            'predictions must be finite numbers') from None
      if not math.isfinite(number):
        raise ValueError('predictions must be finite numbers')


def _predict(model, records):
  '''Run model prediction on records.

  Validation (``_validate_records``) is caller-separated:
  its ValueError is a 422. Errors from ``model.predict``
  (including ValueError) propagate for the caller to map
  to a generic 500.
  '''
  rows = _validate_records(model, records)
  if hasattr(model, 'predict'):
    predictions = _to_list(model.predict(rows))
  elif callable(model):
    predictions = _to_list(model(records))
  else:
    raise TypeError('model must have predict() or be callable')
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
    'content-security-policy': "default-src 'none'; frame-ancestors 'none'",
    'x-permitted-cross-domain-policies': 'none',
}


def _header_map(scope):
  '''Return lower-cased header dict from an ASGI scope.'''
  return {
      key.decode('latin-1').lower(): value.decode('latin-1')
      for key, value in scope.get('headers', [])
  }


async def _send_json(send, status_code, detail):
  '''Send a JSON error response (with security headers).'''
  import json as _json
  body = _json.dumps({'detail': detail}).encode('utf-8')
  headers = [(b'content-type', b'application/json'),
             (b'content-length', str(len(body)).encode('latin-1'))]
  headers.extend(
      (k.encode('latin-1'), v.encode('latin-1'))
      for k, v in SECURITY_HEADERS.items())
  await send({
      'type': 'http.response.start',
      'status': status_code,
      'headers': headers,
  })
  await send({'type': 'http.response.body', 'body': body})


class _CapAuthHeadersMiddleware:
  '''Pure ASGI: streaming body cap, optional API key, headers.

  The body cap counts bytes as ``http.request`` messages
  stream in (works for chunked bodies with no
  Content-Length) and aborts with 413 as soon as the
  running total passes MAX_BODY_BYTES, instead of
  buffering the whole body. Bodies within the cap are
  replayed to the app unchanged. When FORECAST_API_KEY
  is set, every path except /health requires it (401).
  '''

  def __init__(self, app):
    self.app = app

  async def __call__(self, scope, receive, send):
    if scope.get('type') != 'http':
      await self.app(scope, receive, send)
      return
    expected = os.environ.get('FORECAST_API_KEY')
    if expected and scope.get('path') != '/health':
      headers = _header_map(scope)
      provided = headers.get('x-api-key', '')
      auth = headers.get('authorization', '')
      if not provided and auth.lower().startswith('bearer '):
        provided = auth[7:].strip()
      import secrets
      if not provided or not secrets.compare_digest(
              provided, expected):
        await _send_json(send, 401, 'unauthorized')
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
        await _send_json(send, 413, 'request body too large')
        return
      chunks.append(chunk)
      if not message.get('more_body'):
        break
    body = b''.join(chunks)
    replayed = False

    async def _replay_receive():
      nonlocal replayed
      if not replayed:
        replayed = True
        return {
            'type': 'http.request', 'body': body,
            'more_body': False,
        }
      return {'type': 'http.disconnect'}

    async def _send_with_headers(message):
      if message.get('type') == 'http.response.start':
        existing = [
            (k, v) for k, v in message.get('headers', [])
            if k.decode('latin-1').lower() not in SECURITY_HEADERS
        ]
        existing.extend(
            (k.encode('latin-1'), v.encode('latin-1'))
            for k, v in SECURITY_HEADERS.items())
        message = dict(message)
        message['headers'] = existing
      await send(message)

    await self.app(scope, _replay_receive, _send_with_headers)


def create_app(model_loader=None, registry_uri=None, model_name=None,
               alias=None, artifact_path=None):
  '''Create FastAPI app with injected or registry model loader.

  Never raises for a missing loader: without an injected
  loader, MODEL_ARTIFACT_PATH, or registry config, the
  app runs in degraded mode (health reports degraded and
  /predict returns 503).

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

  Returns:
    Configured FastAPI application.

  Raises:
    ImportError: If fastapi or pydantic is not installed.
  '''
  model_loader = _resolve_loader(
      model_loader, registry_uri, model_name, alias,
      artifact_path=artifact_path)
  if _IMPORT_ERROR is not None:
    raise ImportError(
      'fastapi and pydantic are required to serve: install them '
      'with pip install fastapi pydantic'
    ) from _IMPORT_ERROR
  app = FastAPI(title='forecast-ml serving')
  app.add_middleware(_CapAuthHeadersMiddleware)
  # Health probe cache on the app instance: dict with
  # 'result' -> HealthResponse or None when not yet probed.
  _health_cache = {'result': None}
  app.state.health_cache = _health_cache
  app._health_cache = _health_cache

  @app.get('/health', response_model=HealthResponse)
  def health() -> HealthResponse:
    '''Return service health and whether the model loads.'''
    if _health_cache['result'] is not None:
      return _health_cache['result']
    try:
      _load_model(model_loader)
    except Exception:  # pylint: disable=broad-exception-caught
      _health_cache['result'] = HealthResponse(
        status='degraded', model_loaded=False)
    else:
      _health_cache['result'] = HealthResponse(
        status='ok', model_loaded=True)
    return _health_cache['result']

  @app.post('/predict', response_model=PredictResponse)
  def predict(request: PredictRequest) -> PredictResponse:
    '''Return predictions for supplied features or records.'''
    records = _records_from(request)
    _check_caps(records)
    try:
      model = _load_model(model_loader)
    except ModelNotConfiguredError as exc:
      _LOG.warning('model not configured')
      raise HTTPException(
          status_code=503, detail='model not configured') from exc
    except ModelNotFoundError as exc:
      _LOG.warning('model not found')
      raise HTTPException(
          status_code=503, detail='model artifact not found') from exc
    except ModelCorruptError as exc:
      _LOG.exception('model corrupt')
      raise HTTPException(
          status_code=500, detail='model failed to load') from exc
    except Exception as exc:  # pylint: disable=broad-exception-caught
      _LOG.exception('model load failed')
      raise HTTPException(
          status_code=500, detail='model failed to load') from exc
    # Validation separate from model execution: validation
    # ValueError -> 422; model ValueError/other -> generic 500.
    try:
      rows = _validate_records(model, records)
    except ValueError as exc:
      raise HTTPException(status_code=422,
                          detail=str(exc)) from exc
    try:
      if hasattr(model, 'predict'):
        predictions = _to_list(model.predict(rows))
      elif callable(model):
        predictions = _to_list(model(records))
      else:
        raise TypeError('model must have predict() or be callable')
      _check_predictions_finite(predictions)
    except Exception as exc:  # pylint: disable=broad-exception-caught
      # Log only; never leak model/internal detail to client.
      _LOG.exception('prediction failed')
      raise HTTPException(status_code=500,
                          detail='prediction failed') from exc
    return PredictResponse(predictions=predictions)

  return app


def create_app_from_registry(registry_uri, model_name,
                             alias='production'):
  '''Create app loading its model from the MLflow registry.

  Args:
    registry_uri: MLflow registry URI, or None for ambient.
    model_name: Registered model name.
    alias: Registry alias, default ``production``.

  Returns:
    Configured FastAPI application.
  '''
  from forecast_ml_framework.serving.model_loader import (  # noqa: E501
    RegistryModelLoader,
  )
  loader = RegistryModelLoader(registry_uri, model_name, alias=alias)
  return create_app(loader)


try:
  app = create_app()
except ImportError:  # pragma: no cover - missing fastapi
  app = None
