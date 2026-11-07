#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Artifact retention: split artifacts into keep/delete by age.

Pure functions only: no persistence and no deletion happens
here. The caller deletes whatever ``check_retention`` puts in
the ``delete`` list. An artifact is deleted only when its age
is strictly greater than ``retention_days``; artifacts with no
usable age are conservatively kept.
'''

from datetime import datetime, timezone

from forecast_ml_framework.core.params import (
  parse_datetime as _canonical_parse_datetime,
)

_DATE_KEYS = ('created_at', 'timestamp', 'created', 'date',
              'mtime', 'modified_at', 'updated_at')


def _resolve_retention_days(params, retention_days):
  '''Resolve retention_days from int, params mapping, or kwarg.'''
  if retention_days is not None:
    days = retention_days
  elif isinstance(params, (int, float)):
    days = params
  elif isinstance(params, dict):
    days = None
    for key in ('retention_days', 'retention', 'ttl_days',
                'max_age_days'):
      if params.get(key) is not None:
        days = params[key]
        break
    if days is None:
      raise ValueError(
        'check_retention requires retention_days in params')
  elif params is None:
    raise ValueError(
      'check_retention requires retention_days in params')
  else:
    raise ValueError(
      'check_retention requires retention_days in params')
  days = float(days)
  if days < 0:
    raise ValueError('retention_days must be non-negative')
  return days


def _resolve_now(now, clock):
  '''Resolve the reference time via injected now/clock.'''
  stamp = now
  if stamp is None:
    stamp = clock() if callable(clock) else clock
  if stamp is None:
    stamp = datetime.now(timezone.utc)
  if isinstance(stamp, (int, float)):
    return datetime.fromtimestamp(stamp, tz=timezone.utc)
  if isinstance(stamp, str):
    return _parse_datetime(stamp)
  return stamp


def _parse_datetime(value):
  '''Parse an ISO-8601 string (or epoch number) to datetime.

  Epoch numbers use ``fromtimestamp`` (UTC); every other value
  delegates to :func:`core.params.parse_datetime`, which
  normalises naive datetimes to UTC. Retention keeps the
  epoch branch because canonical parsing rejects numbers.
  '''
  if isinstance(value, (int, float)):
    return datetime.fromtimestamp(value, tz=timezone.utc)
  return _canonical_parse_datetime(value)


def _align(reference, stamp):
  '''Drop tzinfo from one side when only one side has it.'''
  ref_aware = reference.tzinfo is not None
  stamp_aware = stamp.tzinfo is not None
  if ref_aware != stamp_aware:
    if ref_aware:
      return stamp.replace(tzinfo=timezone.utc)
    return reference.replace(tzinfo=timezone.utc)
  return stamp


def _age_days(artifact, reference):
  '''Return artifact age in days, or None when unknown.'''
  if not isinstance(artifact, dict):
    return None
  for key in ('age_days', 'age'):
    if artifact.get(key) is not None:
      return float(artifact[key])
  for key in _DATE_KEYS:
    value = artifact.get(key)
    if value is None:
      continue
    if isinstance(value, datetime):
      stamp = value
    else:
      stamp = _parse_datetime(value)
    stamp = _align(reference, stamp)
    return (reference - stamp).total_seconds() / 86400.0
  return None


def check_retention(artifacts, params=None, retention_days=None,
                    now=None, clock=None):
  '''Split ``artifacts`` into keep/delete lists by age.

  Args:
    artifacts: Iterable of artifact mappings (each carrying a
      date field such as ``created_at`` or an ``age_days``
      field), or a mapping of name to creation datetime/ISO
      string. Mapping input returns names in the two lists;
      list input returns the original artifact entries.
    params: Retention config: a mapping with ``retention_days``
      (or ``retention``/``ttl_days``/``max_age_days``), or the
      retention in days as a number.
    retention_days: Optional keyword override for the age limit.
    now: Optional reference datetime; defaults to UTC now.
    clock: Optional injectable clock returning "now".

  Returns:
    Dict ``{'keep': [...], 'delete': [...]}`` preserving input
    order. Ages strictly greater than retention_days delete.

  Raises:
    ValueError: When retention_days is missing or negative.
  '''
  days = _resolve_retention_days(params, retention_days)
  reference = _resolve_now(now, clock)
  if isinstance(artifacts, dict):
    items = [{'name': name, 'created_at': stamp}
             for name, stamp in artifacts.items()]
    keep = [item['name'] for item in items
            if _is_kept(item, reference, days)]
    delete = [item['name'] for item in items
              if not _is_kept(item, reference, days)]
    return {'keep': keep, 'delete': delete}
  keep, delete = [], []
  for artifact in artifacts or []:
    target = keep if _is_kept(artifact, reference, days) else delete
    target.append(artifact)
  return {'keep': keep, 'delete': delete}


def _is_kept(artifact, reference, days):
  '''Return True unless the artifact is older than the limit.'''
  age = _age_days(artifact, reference)
  return age is None or age <= days
