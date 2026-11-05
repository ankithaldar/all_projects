#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Pipeline registry for the forecast ML framework.

Display names map onto the real pipeline modules: ``data``,
``preprocessing`` and ``feature_selection`` are served by
``pipelines.data_prep``; ``modeling``, ``evaluation`` and
``predict`` are served by ``pipelines.modeling``.

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
}


def _load_pipeline(module_name):
  '''Load a pipeline factory by module name.

  Args:
    module_name: Real module suffix, e.g. 'data_prep'.

  Returns:
    The ``create_pipeline`` factory, or None when its module
    is not yet implemented (scaffold phase).
  '''
  try:
    module = __import__(
      'forecast_ml_framework.pipelines.' + module_name,
      fromlist=['create_pipeline'],
    )
  except ImportError:
    return None
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
