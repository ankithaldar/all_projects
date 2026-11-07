#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Canonical dotted-handle resolver and tile re-exports (core layer).

This module is the single owner of the model-handle allowlist
policy (R1 layering): ``data_prep`` and ``scoring`` import
the resolver from here instead of from each other, so no
pipeline imports another pipeline to resolve a ``handle`` /
``*_handle`` config string. The resolver only resolves
estimator-like classes (with ``fit`` and ``predict``) from
allowlisted top-level modules; handles such as ``os.system``
never resolve.

Tile helpers (``assign_ntile``, ``normalize_task_kind``) still
live in ``pipelines.modeling.ntile`` (unmoved, so modeling
imports do not break) and are re-exported lazily from here via
PEP 562 ``__getattr__``: ``scoring`` can import tile helpers
from ``core.handle`` without a top-level pipeline import,
and importing this module never loads the modeling stack.
'''
# pylint: disable=undefined-all-variable,invalid-name
# (tile names in __all__ are PEP 562 lazy re-exports.)

import importlib
import inspect

ALLOWED_MODEL_MODULES = (
  'sklearn',
  'xgboost',
  'lightgbm',
  'catboost',
  'h2o',
  'forecast_ml_framework',
)

_TILE_NAMES = frozenset({
  'assign_ntile',
  'normalize_task_kind',
})

__all__ = [
  'ALLOWED_MODEL_MODULES',
  'assign_ntile',
  'normalize_task_kind',
  'resolve_handle',
  'resolve_handle_strict',
]


def resolve_handle_strict(handle):
  '''Resolve a dotted handle under the model-loader allowlist.

  Args:
    handle: Dotted importable string, e.g.
      ``sklearn.ensemble.RandomForestClassifier``.

  Returns:
    The estimator-like class named by handle.

  Raises:
    ValueError: If handle is not a dotted string, its top-level
      module is not allowlisted, it cannot be imported, or it
      does not name a class with ``fit`` and ``predict``.
  '''
  if not isinstance(handle, str) or '.' not in handle:
    raise ValueError(
        f'expected a dotted importable string, got {handle!r}')
  top_level = handle.split('.', 1)[0]
  if top_level not in ALLOWED_MODEL_MODULES:
    allowed = ', '.join(ALLOWED_MODEL_MODULES)
    raise ValueError(
        f'model_handle {handle!r} is not allowlisted: '
        f'top-level module {top_level!r} is not one of {allowed}')
  module_path, _, attr = handle.rpartition('.')
  try:
    module = importlib.import_module(module_path)
  except ImportError as exc:
    raise ValueError(
        f'cannot resolve handle {handle!r}: module '
        f'{module_path!r} failed to import') from exc
  try:
    obj = getattr(module, attr)
  except AttributeError as exc:
    raise ValueError(
        f'cannot resolve handle {handle!r}: module '
        f'{module_path!r} has no {attr!r}') from exc
  if not inspect.isclass(obj) or not (
          hasattr(obj, 'fit') and hasattr(obj, 'predict')):
    raise ValueError(
        f'model_handle {handle!r} must resolve to an '
        f'estimator-like class with fit and predict, got {obj!r}')
  return obj


def resolve_handle(handle):
  '''Resolve a dotted handle under the model-loader allowlist.

  Public alias for :func:`resolve_handle_strict` (same policy,
  same errors).

  Args:
    handle: Dotted importable string, e.g.
      ``sklearn.ensemble.RandomForestClassifier``.

  Returns:
    The estimator-like class named by handle.

  Raises:
    ValueError: If handle is not allowlisted, cannot be
      resolved, or is not an estimator-like class.
  '''
  return resolve_handle_strict(handle)


def __getattr__(name):
  '''Resolve tile helpers lazily from modeling.ntile (PEP 562).'''
  if name in _TILE_NAMES:
    # pylint: disable=import-outside-toplevel
    from forecast_ml_framework.pipelines.modeling import (
      ntile as _ntile,
    )
    # pylint: enable=import-outside-toplevel
    return getattr(_ntile, name)
  raise AttributeError(
      f'module {__name__!r} has no attribute {name!r}')


def __dir__():
  '''Return public names plus lazily re-exported tile helpers.'''
  return sorted(set(globals()) | _TILE_NAMES)
