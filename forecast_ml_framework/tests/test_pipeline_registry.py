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


def test_missing_pipeline_module_returns_none_via_registry(monkeypatch):
  '''Missing pipeline module is skipped, not an error.'''
  import builtins
  import forecast_ml_framework.pipeline_registry as registry
  real_import = builtins.__import__

  def fake_import(name, *args, **kwargs):
    if name == 'forecast_ml_framework.pipelines.modeling':
      raise ModuleNotFoundError(
        "No module named 'forecast_ml_framework.pipelines.modeling'",
        name=name)
    return real_import(name, *args, **kwargs)

  monkeypatch.setattr(builtins, '__import__', fake_import)
  pipelines = registry.register_pipelines()
  assert 'modeling' not in pipelines
  assert 'data' in pipelines


def test_internal_import_error_surfaces(monkeypatch):
  '''Internal ImportError inside a pipeline module is re-raised.'''
  import builtins
  import forecast_ml_framework.pipeline_registry as registry
  real_import = builtins.__import__

  def fake_import(name, *args, **kwargs):
    if name == 'forecast_ml_framework.pipelines.modeling':
      raise ModuleNotFoundError(
        "No module named 'some_internal_dep'", name='some_internal_dep')
    return real_import(name, *args, **kwargs)

  monkeypatch.setattr(builtins, '__import__', fake_import)
  try:
    registry.register_pipelines()
  except ModuleNotFoundError as exc:
    assert exc.name == 'some_internal_dep'
  else:
    raise AssertionError('internal ImportError was swallowed')


def test_internal_import_error_other_subclass_surfaces(monkeypatch):
  '''Plain ImportError inside a pipeline module is re-raised.'''
  import builtins
  import forecast_ml_framework.pipeline_registry as registry
  real_import = builtins.__import__

  def fake_import(name, *args, **kwargs):
    if name == 'forecast_ml_framework.pipelines.modeling':
      raise ImportError('boom inside pipeline')
    return real_import(name, *args, **kwargs)

  monkeypatch.setattr(builtins, '__import__', fake_import)
  try:
    registry.register_pipelines()
  except ImportError as exc:
    assert 'boom inside pipeline' in str(exc)
  else:
    raise AssertionError('internal ImportError was swallowed')


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
