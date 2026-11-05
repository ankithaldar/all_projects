#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests for forecast_ml_framework.pipeline_registry.'''

# pylint: disable=import-outside-toplevel


def test_registry_includes_five_new_pipelines_twelve_keys():
  '''Registry maps 5 new display names to real pipelines.'''
  from forecast_ml_framework.pipeline_registry import register_pipelines
  pipelines = register_pipelines()
  assert set(pipelines) == {
    'data', 'preprocessing', 'feature_selection',
    'modeling', 'evaluation', 'predict', 'scoring',
    'eda', 'monitoring', 'retraining',
    'cost_tracking', 'interpretability'}
  assert len(pipelines) == 12


def test_new_pipelines_are_real_kedro_pipelines():
  '''Each new entry is a pipeline object with nodes.'''
  from forecast_ml_framework.pipeline_registry import register_pipelines
  pipelines = register_pipelines()
  for name in ('eda', 'monitoring', 'retraining',
               'cost_tracking', 'interpretability'):
    pipeline = pipelines[name]
    assert pipeline is not None
    assert len(pipeline.nodes) > 0


def test_new_modules_expose_no_arg_factory():
  '''Each new module's create_pipeline works with no args.'''
  from forecast_ml_framework.pipelines.cost_tracking import (
    create_pipeline as cost_factory)
  from forecast_ml_framework.pipelines.eda import (
    create_pipeline as eda_factory)
  from forecast_ml_framework.pipelines.interpretability import (
    create_pipeline as interpretability_factory)
  from forecast_ml_framework.pipelines.monitoring import (
    create_pipeline as monitoring_factory)
  from forecast_ml_framework.pipelines.retraining import (
    create_pipeline as retraining_factory)
  for factory in (eda_factory, monitoring_factory,
                  retraining_factory, cost_factory,
                  interpretability_factory):
    assert factory() is not None
    assert len(factory().nodes) > 0
