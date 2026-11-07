#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Governance: lineage, audit log, and retention policy.

Lazy package init (PEP 562): importing this package -- or
the ``scrub`` leaf via ``governance.scrub`` -- does not
import ``audit``/``lineage``/``retention``. That keeps
``telemetry.exceptions`` (which imports only the scrub
leaf) free of transitive coupling to governance code
that may itself raise ``FrameworkError``. Public names
are resolved on first access via ``__getattr__`` and
cached in the module namespace.
'''
# pylint: disable=undefined-all-variable,invalid-name
# (PEP 562 lazy names are intentionally not static.)

import importlib

__all__ = [
  'REDACTED',
  'audit_log',
  'build_lineage',
  'check_retention',
  'compute_config_hash',
  'scrub_details',
]

_EXPORTS = {
  'REDACTED': ('scrub', 'REDACTED'),
  'audit_log': ('audit', 'audit_log'),
  'build_lineage': ('lineage', 'build_lineage'),
  'check_retention': ('retention', 'check_retention'),
  'compute_config_hash': ('lineage', 'compute_config_hash'),
  'scrub_details': ('scrub', 'scrub_details'),
}

_SUBMODULES = frozenset({'audit', 'lineage', 'retention', 'scrub'})


def __getattr__(name):
  '''Resolve public names/submodules lazily (PEP 562).'''
  if name in _EXPORTS:
    module_name, attr = _EXPORTS[name]
    module = importlib.import_module(
      f'forecast_ml_framework.governance.{module_name}')
    value = getattr(module, attr)
    globals()[name] = value
    return value
  if name in _SUBMODULES:
    module = importlib.import_module(
      f'forecast_ml_framework.governance.{name}')
    globals()[name] = module
    return module
  raise AttributeError(
    f'module {__name__!r} has no attribute {name!r}')


def __dir__():
  '''Return public names plus module globals.'''
  return sorted(set(globals()) | set(__all__) | _SUBMODULES)
