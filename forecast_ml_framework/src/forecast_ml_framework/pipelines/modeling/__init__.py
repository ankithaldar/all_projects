#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Modeling pipeline package (lazy, PEP 562).

Importing this package -- or any leaf such as
``pipelines.modeling.validation`` -- does not import
``.pipeline`` (and therefore does not pull numpy/sklearn
through the pipeline wiring). ``create_pipeline`` is
resolved on first access via ``__getattr__`` and cached,
mirroring ``governance/__init__``.
'''
# pylint: disable=undefined-all-variable,invalid-name
# (PEP 562 lazy names are intentionally not static.)

import importlib

__all__ = ['create_pipeline']

_EXPORTS = {
  'create_pipeline': (
    'forecast_ml_framework.pipelines.modeling.pipeline',
    'create_pipeline',
  ),
}


def __getattr__(name):
  '''Resolve public names lazily (PEP 562).'''
  if name in _EXPORTS:
    module_name, attr = _EXPORTS[name]
    module = importlib.import_module(module_name)
    value = getattr(module, attr)
    globals()[name] = value
    return value
  raise AttributeError(
    f'module {__name__!r} has no attribute {name!r}')


def __dir__():
  '''Return public names plus module globals.'''
  return sorted(set(globals()) | set(__all__))
