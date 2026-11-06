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
cleanly even when a pipeline module is unavailable.
'''

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
    if exc.name == full_name:
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
  return pipelines


__all__ = ['register_pipelines']
