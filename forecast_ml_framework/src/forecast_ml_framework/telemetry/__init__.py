#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Telemetry: typed errors and ``wrap`` helper.

Exports the ``FrameworkError`` hierarchy, its
subclasses (and their factories, which are the public
API per A2 -- never delete on YAGNI grounds),
``CODE_REGISTRY``, and ``wrap`` for
copy-on-wrap conversion of arbitrary exceptions
into typed errors.
'''

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
