#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Modelling layer: trainers, scoring, calibration, metrics and promotion.

The dependency discipline is strictly one-way: nothing here imports a node, a
pipeline or a configuration module.
"""

from forecasting_ml_framework.modeling.calibration import calibrate_probability, select_threshold
from forecasting_ml_framework.modeling.metrics import (
  METRIC_REGISTRY,
  MetricSpec,
  acceptance_threshold,
  band_metrics,
  evaluate_acceptance,
  parse_metric,
  regression_band_metrics,
  select_band,
)
from forecasting_ml_framework.modeling.models import ModelTraining, SparkModelTraining, TransformerTraining
from forecasting_ml_framework.modeling.promotion import PromotionDecision, apply_promotion, decide_promotion
from forecasting_ml_framework.modeling.scoring import project_score_frame, score_frame

__all__ = [
  'METRIC_REGISTRY',
  'MetricSpec',
  'ModelTraining',
  'PromotionDecision',
  'SparkModelTraining',
  'TransformerTraining',
  'acceptance_threshold',
  'apply_promotion',
  'band_metrics',
  'calibrate_probability',
  'decide_promotion',
  'evaluate_acceptance',
  'parse_metric',
  'project_score_frame',
  'regression_band_metrics',
  'score_frame',
  'select_band',
  'select_threshold',
]
