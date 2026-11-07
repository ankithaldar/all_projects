#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Pipeline registry for the forecast ML framework.

Twelve display names map onto the real pipeline modules:
``data``, ``preprocessing`` and ``feature_selection`` are
served by ``pipelines.data_prep``; ``modeling``,
``evaluation`` and
``predict`` are served by ``pipelines.modeling``; ``scoring``
is served by ``pipelines.scoring``; and ``eda``,
``monitoring``, ``retraining``, ``cost_tracking`` and
``interpretability`` are served by their eponymous modules
under ``pipelines``.

Imports are lazy and defensive so that this module imports
cleanly even when a pipeline module is unavailable. A
missing pipeline module (or a missing Kedro install
surfacing from inside it) is skipped with a ``UserWarning``
naming the omitted display name (V9), so a Kedro-less
self-test degrades to a partial registry instead of
crashing; any other internal ``ImportError`` still
surfaces, never swallowed.

Name divergence (``predict``, intentional and pinned):
the registry display name ``predict`` maps to the
``modeling`` pipeline (``pipelines.modeling``), so
``kedro run --pipeline predict`` runs ``modeling``. That
differs from ``forecast-ml predict``, which runs the
``scoring`` pipeline (see ``cli``). Both mappings are
pinned by tests; aligning them would change pinned
behaviour, so the divergence is documented here rather
than removed.
'''

import warnings

_PIPELINE_MODULES = {
  'data': 'data_prep',
  'preprocessing': 'data_prep',
  'feature_selection': 'data_prep',
  'modeling': 'modeling',
  'evaluation': 'modeling',
  'predict': 'modeling',
  'scoring': 'scoring',
  'eda': 'eda',
  'monitoring': 'monitoring',
  'retraining': 'retraining',
  'cost_tracking': 'cost_tracking',
  'interpretability': 'interpretability',
}


def _load_pipeline(module_name):
  '''Load a pipeline factory by module name.

  Args:
    module_name: Real module suffix, e.g. 'data_prep'.

  Returns:
    The ``create_pipeline`` factory, or None when the named
    pipeline module is unavailable.
  '''
  full_name = 'forecast_ml_framework.pipelines.' + module_name
  try:
    module = __import__(
      full_name,
      fromlist=['create_pipeline'],
    )
  except ModuleNotFoundError as exc:
    # Only a missing pipeline module itself is unavailable.
    # Any other missing module is an internal dependency of an
    # existing pipeline and must surface, not be swallowed.
    # Exception (V9): a missing Kedro install surfacing from
    # inside an otherwise-present pipeline is also treated
    # as unavailable, so a Kedro-less self-test still works.
    missing = exc.name
    if missing == full_name:
      warnings.warn(
        f'pipeline module {full_name} omitted: '
        'module unavailable', stacklevel=3)
      return None
    if missing == 'kedro' or (missing or '').startswith('kedro.'):
      warnings.warn(
        f'pipeline module {full_name} omitted: '
        'kedro unavailable', stacklevel=3)
      return None
    raise
  return getattr(module, 'create_pipeline', None)


def register_pipelines():
  '''Register all available pipelines.

  Returns:
    Mapping of display name to pipeline object. Aliases for
    the same module share one pipeline object (the factory
    is called once per module).
  '''
  pipelines = {}
  cache = {}
  for name, module_name in _PIPELINE_MODULES.items():
    if module_name not in cache:
      factory = _load_pipeline(module_name)
      cache[module_name] = factory() if factory else None
    pipeline = cache[module_name]
    if pipeline is not None:
      pipelines[name] = pipeline
    else:
      warnings.warn(
        f'pipeline {name!r} omitted: '
        f'module {module_name!r} unavailable', stacklevel=2)
  return pipelines


__all__ = ['register_pipelines']
