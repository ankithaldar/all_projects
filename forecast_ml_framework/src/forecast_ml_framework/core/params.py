#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Canonical params flattening shared by pipeline nodes.

Replaces the former per-module ``_params_dict`` copies in
the EDA, monitoring, and interpretability nodes.

Also owns the canonical scalar helpers
(:func:`parse_int`, :func:`parse_float`, :func:`get_first`,
:func:`parse_datetime`) so pipeline modules do not each
redefine ``_int``/``_float``/``_get``/``_parse_datetime``
(F8); modules that still carry their own copies should
import these instead (see W10 handoff).
'''

from datetime import date, datetime, timezone


def parse_int(value, default):
  '''Return value as int, or default when missing/invalid.

  Args:
    value: Value to coerce.
    default: Fallback when coercion fails.

  Returns:
    The coerced int, or ``default``.
  '''
  try:
    return int(value)
  except (TypeError, ValueError):
    return int(default)


def parse_float(value, default):
  '''Return value as float, or default when missing/invalid.

  Args:
    value: Value to coerce.
    default: Fallback when coercion fails.

  Returns:
    The coerced float, or ``default``.
  '''
  try:
    return float(value)
  except (TypeError, ValueError):
    return float(default)


def get_first(mapping, *keys):
  '''Return the first present, non-None value among keys.

  Args:
    mapping: Mapping to look up, or None.
    *keys: Candidate key names, in priority order.

  Returns:
    The first non-None value, or None when absent.
  '''
  if not mapping:
    return None
  for key in keys:
    if key in mapping and mapping[key] is not None:
      return mapping[key]
  return None


def parse_datetime(value):
  '''Parse a datetime/date/ISO string into an aware datetime.

  Args:
    value: Datetime, date, or ISO-format string. Naive
      values are treated as UTC.

  Returns:
    Aware datetime.

  Raises:
    ValueError: If the value cannot be parsed.
  '''
  if isinstance(value, datetime):
    parsed = value
  elif isinstance(value, date):
    parsed = datetime(value.year, value.month, value.day)
  elif isinstance(value, str):
    text = value.strip().replace('Z', '+00:00')
    try:
      parsed = datetime.fromisoformat(text)
    except ValueError as exc:
      raise ValueError(
        'cannot parse datetime value %r' % (value,)) from exc
  else:
    raise ValueError('cannot parse datetime value %r' % (value,))
  if parsed.tzinfo is None:
    parsed = parsed.replace(tzinfo=timezone.utc)
  return parsed


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
