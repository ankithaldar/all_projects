#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Minimal FastAPI serving app.

Endpoints: GET /health, POST /predict.

Models come from an injected loader (preferred) or from the
MLflow registry via explicit config or MODEL_* env vars.

No auth or Docker is provided (out of scope for Phase 1).
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


if _IMPORT_ERROR is None:

  class HealthResponse(BaseModel):
    '''Health response model.'''

    status: str

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
  required = list(getattr(model, 'feature_names_in_', keys) or keys)
  rows = []
  for record in records:
    for name in required:
      if name not in record:
        raise ValueError(f'missing feature: {name}')
    row = []
    for name in keys:
      value = record[name]
      if isinstance(value, bool) or not isinstance(
          value, (int, float)):
        raise ValueError(
            f'feature {name} must be a finite number')
      number = float(value)
      if not math.isfinite(number):
        raise ValueError(
            f'feature {name} must be a finite number')
      row.append(number)
    rows.append(row)
  return rows


def _predict(model, records):
  '''Run model prediction on records.'''
  rows = _validate_records(model, records)
  if hasattr(model, 'predict'):
    return _to_list(model.predict(rows))
  if callable(model):
    return _to_list(model(records))
  raise TypeError('model must have predict() or be callable')


def _resolve_loader(model_loader, registry_uri, model_name, alias):
  '''Resolve the effective loader, preferring injection.

  Falls back to explicit registry config, then MODEL_*
  environment variables. Errors are generic (no leak).
  '''
  if model_loader is not None:
    return model_loader
  registry_uri = registry_uri or os.environ.get('MODEL_REGISTRY_URI')
  model_name = model_name or os.environ.get('MODEL_NAME')
  alias = alias or os.environ.get('MODEL_ALIAS') or 'production'
  if not model_name:
    raise ValueError('model loader or registry config is required')
  from forecast_ml_framework.serving.model_loader import (  # noqa: E501
    RegistryModelLoader,
  )
  return RegistryModelLoader(registry_uri, model_name, alias=alias)


def create_app(model_loader=None, registry_uri=None, model_name=None,
               alias=None):
  '''Create FastAPI app with injected or registry model loader.

  Args:
    model_loader: Callable returning a model, or a model.
      Preferred when given (back-compat); registry config
      and MODEL_* env vars are ignored in that case.
    registry_uri: MLflow registry URI (or MODEL_REGISTRY_URI).
    model_name: Registered model name (or MODEL_NAME).
    alias: Registry alias (or MODEL_ALIAS, default
      ``production``).

  Returns:
    Configured FastAPI application.

  Raises:
    ImportError: If fastapi or pydantic is not installed.
    ValueError: If no loader or registry config is given.
  '''
  model_loader = _resolve_loader(
      model_loader, registry_uri, model_name, alias)
  if _IMPORT_ERROR is not None:
    raise ImportError(
      'fastapi and pydantic are required to serve: install them '
      'with pip install fastapi pydantic'
    ) from _IMPORT_ERROR
  app = FastAPI(title='forecast-ml serving')

  @app.get('/health', response_model=HealthResponse)
  def health() -> HealthResponse:
    '''Return service health.'''
    return HealthResponse(status='ok')

  @app.post('/predict', response_model=PredictResponse)
  def predict(request: PredictRequest) -> PredictResponse:
    '''Return predictions for supplied features or records.'''
    records = _records_from(request)
    _check_caps(records)
    try:
      model = model_loader() if callable(model_loader) else model_loader
    except Exception as exc:  # pylint: disable=broad-exception-caught
      _LOG.exception('model load failed')
      raise HTTPException(
          status_code=500, detail='model unavailable') from exc
    try:
      predictions = _predict(model, records)
    except ValueError as exc:
      raise HTTPException(status_code=422,
                          detail=str(exc)) from exc
    except HTTPException:  # pylint: disable=duplicate-except
      raise
    except Exception as exc:  # pylint: disable=broad-exception-caught
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
