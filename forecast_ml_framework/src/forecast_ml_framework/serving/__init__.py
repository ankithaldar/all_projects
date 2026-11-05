#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Serving package: FastAPI app and model loaders.'''

from forecast_ml_framework.serving.app import (
  HealthResponse,
  PredictRequest,
  PredictResponse,
  create_app,
  create_app_from_registry,
)
from forecast_ml_framework.serving.model_loader import (
  FileModelLoader,
  RegistryModelLoader,
)

__all__ = [
  'create_app',
  'create_app_from_registry',
  'FileModelLoader',
  'HealthResponse',
  'PredictRequest',
  'PredictResponse',
  'RegistryModelLoader',
]
