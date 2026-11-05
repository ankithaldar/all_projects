#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''MLflow model registry with alias promotion.'''

from forecast_ml_framework.registry.model_registry import (
    KNOWN_ALIASES, ModelRegistry, validate_alias,
    validate_model_name)

__all__ = [
    'KNOWN_ALIASES',
    'ModelRegistry',
    'validate_alias',
    'validate_model_name',
]
