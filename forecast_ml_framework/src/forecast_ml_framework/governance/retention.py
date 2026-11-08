#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Artifact retention: split artifacts into keep/delete by age.

``check_retention`` is pure: no persistence and no deletion
happens there. The caller deletes whatever it puts in the
``delete`` list. An artifact is deleted only when its age is
strictly greater than ``retention_days``; artifacts with no
usable age are conservatively kept.

Enforcement lives in :func:`enforce_retention`, the single
caller that enumerates one store, plans with
``check_retention``, deletes through :func:`delete_record`,
and audits with hashed actors only. No other delete call
site for retention lives in this module's callers.
'''
# pylint: disable=broad-exception-caught, import-outside-toplevel

from datetime import datetime, timezone

from forecast_ml_framework.core.params import (
  parse_datetime as _canonical_parse_datetime,
)
from forecast_ml_framework.governance.audit import audit_log
from forecast_ml_framework.governance.scrub import subject_hash

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
  '''Drop tzinfo from one side when only one side has it.

  A naive side is read as UTC. When the reference is naive
  and the stamp is aware, the stamp is converted to naive
  UTC (never the reference object), so the subtraction in
  :func:`_age_days` always compares like with like.
  '''
  ref_aware = reference.tzinfo is not None
  stamp_aware = stamp.tzinfo is not None
  if ref_aware != stamp_aware:
    if ref_aware:
      return stamp.replace(tzinfo=timezone.utc)
    return stamp.astimezone(timezone.utc).replace(tzinfo=None)
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


def _iso_now(at=None, now=None, clock=None):
  '''Resolve a stamp to ISO text (sole ISO helper).

  This is the single ISO-text definition for the
  governance modules: ``erasure`` imports it instead of
  keeping its own copy (sole-leaf rule). ``at`` wins
  first, then ``now``, then ``clock`` (callable or
  stamp); the current UTC time is used when all are
  None. Datetimes go through ``isoformat``; any other
  stamp goes through ``str``.

  Args:
    at: Explicit stamp (datetime or text), wins first.
    now: Fallback stamp (datetime or text).
    clock: Fallback clock (callable returning a stamp,
      or a stamp itself).

  Returns:
    ISO-8601 text; the current UTC time when all are
    None.
  '''
  stamp = at if at is not None else now
  if stamp is None:
    stamp = clock() if callable(clock) else clock
  if stamp is None:
    stamp = datetime.now(timezone.utc)
  iso = getattr(stamp, 'isoformat', None)
  if callable(iso):
    return iso()
  return str(stamp)


def _iso_text(stamp):
  '''Return ``stamp`` as ISO text, via :func:`_iso_now`.'''
  return _iso_now(at=stamp)


def _redis_entities(client):
  '''Return entity ids from Redis feature keys, or [].

  Args:
    client: Redis-like client with a ``keys`` method, or None.

  Returns:
    Entity ids decoded from ``feature:<entity>`` keys, in
    first-seen order without duplicates. Clients without a
    usable ``keys`` method (memory-only fakes) yield [] --
    that is a not-found, not an error.

  Raises:
    Exception: Any ``keys`` enumeration error propagates.
    A Redis outage must never be swallowed to ``[]``:
    that would silently report a sweep over an empty
    store instead of failing the sweep.
  '''
  keys_fn = getattr(client, 'keys', None)
  if not callable(keys_fn):
    return []
  keys = keys_fn('feature:*')
  entities = []
  for key in keys or []:
    if isinstance(key, (bytes, bytearray)):
      key = bytes(key).decode('utf-8', errors='replace')
    if not isinstance(key, str) or not key.startswith('feature:'):
      continue
    entity = key[len('feature:'):]
    if entity and entity not in entities:
      entities.append(entity)
  return entities


def _is_online_store(store):
  '''Return True for online (memory/Redis) feature stores.'''
  if callable(getattr(store, '_records_for', None)):
    return True
  memory = getattr(store, '_store', None)
  return isinstance(memory, dict) and callable(
      getattr(store, 'delete', None))


def _is_offline_store(store):
  '''Return True for offline or bare artifact stores.'''
  if callable(getattr(store, '_all_keys', None)):
    return True
  if getattr(store, 'artifact_store', None) is not None:
    return True
  return callable(getattr(store, 'list', None)) and callable(
      getattr(store, 'delete', None))


def list_snapshots(store):
  '''Enumerate offline snapshots as key plus timestamp pairs.

  Args:
    store: Offline feature store (``_all_keys``) or a bare
      artifact store (``list``). Keys are read via
      ``_all_keys`` first, then via ``list('')``.

  Returns:
    List of ``{'key': key, 'created_at': ISO str or None}``.
    A key with no parseable timestamp yields None, which
    ``check_retention`` conservatively keeps.

  Raises:
    ValueError: When the store exposes neither enumeration.
  '''
  from forecast_ml_framework.feature_store.store import (
    _snapshot_timestamp)
  all_keys_fn = getattr(store, '_all_keys', None)
  if callable(all_keys_fn):
    keys = list(all_keys_fn() or [])
  elif callable(getattr(store, 'list', None)):
    keys = list(store.list('') or [])
  else:
    raise ValueError('unknown store/record type')
  records = []
  for key in keys:
    stamp = _snapshot_timestamp(key)
    created = _iso_text(stamp) if stamp is not None else None
    records.append({'key': key, 'created_at': created})
  return records


def list_online_entities(store):
  '''Enumerate online entities with their max as_of.

  Args:
    store: Online feature store. Entities are the union of
      memory ``_store`` keys and Redis ``feature:`` keys;
      each entity age basis is the max ``as_of`` over the
      merged memory plus Redis records from ``_records_for``.

  Returns:
    List of ``{'entity_id': entity, 'as_of': ISO str,
    'created_at': ISO str}``. Entities with no records are
    skipped. ``created_at`` mirrors ``as_of`` so the pure
    ``check_retention`` planner (which reads date fields)
    sees the same age basis.

  Raises:
    ValueError: When the store has no ``_records_for``.
    Exception: Store / Redis enumeration errors
      propagate (never swallowed to an empty sweep).
  '''
  records_for = getattr(store, '_records_for', None)
  if not callable(records_for):
    raise ValueError('unknown store/record type')
  entities = []
  memory = getattr(store, '_store', None)
  if isinstance(memory, dict):
    for entity in memory:
      if entity not in entities:
        entities.append(entity)
  client = None
  client_fn = getattr(store, '_client', None)
  if callable(client_fn):
    client = client_fn()
  if client is not None:
    for entity in _redis_entities(client):
      if entity not in entities:
        entities.append(entity)
  records = []
  for entity in entities:
    stored = records_for(entity) or []
    stamps = [item.get('as_of') for item in stored
              if isinstance(item, dict) and item.get('as_of') is not None]
    if not stamps:
      continue
    stamp = max(stamps)
    records.append({'entity_id': entity, 'as_of': _iso_text(stamp),
                    'created_at': _iso_text(stamp)})
  return records


def delete_record(store, record):
  '''Delete one retention record via typed dispatch.

  This is the single delete entry point for retention: a
  record carrying ``entity_id`` deletes the online entity,
  a record carrying ``key`` deletes the offline snapshot
  through ``artifact_store.delete`` (or a bare artifact
  store's own ``delete``). Store errors propagate.

  Args:
    store: Feature or artifact store to delete from.
    record: Record dict from ``list_snapshots`` or
      ``list_online_entities``.

  Raises:
    ValueError: When the store/record pairing is unknown.
  '''
  if not isinstance(record, dict):
    raise ValueError('unknown store/record type')
  if 'entity_id' in record:
    delete_fn = getattr(store, 'delete', None)
    if callable(delete_fn):
      delete_fn(record['entity_id'])
      return
    raise ValueError('unknown store/record type')
  if 'key' in record:
    artifact = getattr(store, 'artifact_store', None)
    artifact_delete = getattr(artifact, 'delete', None)
    if callable(artifact_delete):
      artifact_delete(record['key'])
      return
    delete_fn = getattr(store, 'delete', None)
    if callable(delete_fn):
      delete_fn(record['key'])
      return
    raise ValueError('unknown store/record type')
  raise ValueError('unknown store/record type')


def enforce_retention(store, params, actor, *, now=None, clock=None,
                      log=None):
  '''Enumerate, plan, delete expired records, and audit.

  Args:
    store: Offline or online feature store (a bare artifact
      store counts as offline). The enumeration follows the
      store type; the plan is the pure ``check_retention``.
    params: Retention config passed to ``check_retention``
      (for example ``{'retention_days': 30}``).
    actor: Operator identity. Only its scrub hash is ever
      audited or logged, never the clear value.
    now: Optional reference time for the planner.
    clock: Optional injectable clock for the planner.
    log: Optional list to append the audit entry to. The
      entry is returned either way; the caller owns
      persistence of the list.

  Returns:
    Dict ``{'keep': [...], 'delete': [...],
    'audit_entry': {...}}`` from the planner plus the sweep
    audit entry (``swept`` / ``kept`` counts only).

  Raises:
    ValueError: When the actor is blank or the store type
      is unknown.
    Exception: Any store error aborts the sweep: a partial
      audit entry (``partial: True``, swept counts only
      completed deletes) is appended first, then the error
      propagates. Enumeration errors (including Redis
      enumeration outages) audit a partial sweep with
      ``swept: 0`` before propagating. A partial sweep
      is never claimed swept.
  '''
  if actor is None or (isinstance(actor, str)
                       and not actor.strip()):
    raise ValueError('enforce_retention requires a non-empty actor')
  actor_hash = subject_hash(actor)
  try:
    if _is_online_store(store):
      records = list_online_entities(store)
    elif _is_offline_store(store):
      records = list_snapshots(store)
    else:
      raise ValueError('unknown store/record type')
  except Exception:
    # Enumeration failed (for example a Redis outage):
    # audit a partial sweep (swept counts only) before
    # re-raising, so the failure is never reported as a
    # successful empty sweep.
    audit_log('retention_sweep', actor_hash,
              {'swept': 0, 'kept': 0, 'partial': True},
              log=log)
    raise
  plan = check_retention(records, params, now=now, clock=clock)
  deleted = []
  try:
    for record in plan['delete']:
      delete_record(store, record)
      deleted.append(record)
  except Exception:
    audit_log('retention_sweep', actor_hash,
              {'swept': len(deleted), 'kept': len(plan['keep']),
               'partial': True}, log=log)
    raise
  entry = audit_log('retention_sweep', actor_hash,
                    {'swept': len(deleted),
                     'kept': len(plan['keep'])}, log=log)
  return {'keep': plan['keep'], 'delete': plan['delete'],
          'audit_entry': entry}
