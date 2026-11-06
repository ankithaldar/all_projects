#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Canonical params flattening shared by pipeline nodes.

Replaces the former per-module ``_params_dict`` copies in
the EDA, monitoring, and interpretability nodes.
'''


def require_dict(value, label):
  '''Return value as a dict, raising an informative ValueError.'''
  if value is None:
    return {}
  if not isinstance(value, dict):
    raise ValueError('%s must be a dict, got %r' % (label, value))
  return value


def flatten_params(params, key):
  '''Return params as a flat dict (None -> {}).

  Args:
    params: Params mapping, or a mapping wrapping it under
      ``key`` (e.g. ``{'eda_params': {...}}``). Nested
      values are the base; top-level keys win.
    key: Nested params key to unwrap, or None for no
      unwrapping.

  Returns:
    Flat dict copy of params.

  Raises:
    ValueError: If ``params`` is not dict-convertible.
  '''
  if params is None:
    return {}
  if isinstance(params, dict):
    flat = dict(params)
  else:
    try:
      flat = dict(params)
    except (TypeError, ValueError) as exc:
      raise ValueError(
        'params must be a dict, got %r' % (params,)) from exc
  if key is not None:
    nested = flat.get(key)
    if isinstance(nested, dict):
      merged = dict(nested)
      merged.update({k: v for k, v in flat.items() if k != key})
      return merged
  return flat
