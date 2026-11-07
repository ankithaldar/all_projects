#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Core abstractions for the forecast ML framework.

Lazy package init (PEP 562): importing this package -- or
the ``config_loader`` leaf -- does not import ``datasets``
(kedro) / ``artifact_store`` / ``spark_factory``. That
keeps plain-dict ``ConfigLoader`` use free of kedro and
other heavy dependencies (V11), mirroring the governance
package pattern. Public names are resolved on first
access via ``__getattr__`` and cached.
'''
# pylint: disable=undefined-all-variable,invalid-name
# (PEP 562 lazy names are intentionally not static.)

import importlib

__all__ = [
    'ArtifactStore',
    'ConfigLoader',
    'GCSArtifactStore',
    'LocalArtifactStore',
    'S3ArtifactStore',
    'SparkSessionFactory',
    'SparkSQLDataSet',
    'SQLQueryDataSet',
    'create_artifact_store',
    'to_pandas',
]

_EXPORTS = {
    'ArtifactStore': ('artifact_store', 'ArtifactStore'),
    'GCSArtifactStore': ('artifact_store', 'GCSArtifactStore'),
    'LocalArtifactStore': ('artifact_store', 'LocalArtifactStore'),
    'S3ArtifactStore': ('artifact_store', 'S3ArtifactStore'),
    'create_artifact_store': ('artifact_store', 'create_artifact_store'),
    'ConfigLoader': ('config_loader', 'ConfigLoader'),
    'SparkSQLDataSet': ('datasets', 'SparkSQLDataSet'),
    'SQLQueryDataSet': ('datasets', 'SQLQueryDataSet'),
    'SparkSessionFactory': ('spark_factory', 'SparkSessionFactory'),
    'to_pandas': ('spark_factory', 'to_pandas'),
}

_SUBMODULES = frozenset({
    'artifact_store',
    'config_loader',
    'config_schema',
    'datasets',
    'params',
    'predictions',
    'spark_factory',
})


def __getattr__(name):
  '''Resolve public names/submodules lazily (PEP 562).'''
  if name in _EXPORTS:
    module_name, attr = _EXPORTS[name]
    module = importlib.import_module(
      f'forecast_ml_framework.core.{module_name}')
    value = getattr(module, attr)
    globals()[name] = value
    return value
  if name in _SUBMODULES:
    module = importlib.import_module(
      f'forecast_ml_framework.core.{name}')
    globals()[name] = module
    return module
  raise AttributeError(
    f'module {__name__!r} has no attribute {name!r}')


def __dir__():
  '''Return public names plus module globals.'''
  return sorted(set(globals()) | set(__all__) | _SUBMODULES)
