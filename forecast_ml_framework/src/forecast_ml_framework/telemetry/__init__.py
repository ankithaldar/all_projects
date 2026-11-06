#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Telemetry: typed errors (full API lands per task).'''

from forecast_ml_framework.telemetry.exceptions import (
    CODE_REGISTRY,
    CircuitOpenError,
    DataIngestionError,
    DataValidationError,
    EvaluationError,
    FeatureEngineeringError,
    FrameworkError,
    InterpretationError,
    ResilienceError,
    TrainingFailedError,
    wrap,
)

__all__ = [
    'CODE_REGISTRY',
    'CircuitOpenError',
    'DataIngestionError',
    'DataValidationError',
    'EvaluationError',
    'FeatureEngineeringError',
    'FrameworkError',
    'InterpretationError',
    'ResilienceError',
    'TrainingFailedError',
    'wrap',
]
