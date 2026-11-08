#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Online feature store: in-memory dict plus optional Redis.'''
# pylint: disable=import-outside-toplevel, broad-exception-caught

import json
import logging

from forecast_ml_framework.feature_store.store import (
    ENTITY_COL, FeatureStore, _check_not_empty, _naive_utc,
    _naive_utc_frame, _native_features, _normalize_frame,
    _resolve_time)
from forecast_ml_framework.governance.scrub import subject_hash

LOG = logging.getLogger(__name__)


def _resolve_naive(as_of=None, request_time=None):
  '''Resolve cutoff to a naive UTC Timestamp (or None).'''
  raw = _resolve_time(as_of=as_of, request_time=request_time)
  return _naive_utc(raw) if raw is not None else None


class OnlineFeatureStore(FeatureStore):
  '''Serve point-in-time features from memory / optional Redis.

  Redis is fully lazy: nothing is imported at construction, a
  missing ``redis`` package or unreachable server falls back to
  the in-memory dict, and a supplied client only needs
  ``get``/``set``.

  Args:
    redis_client: Optional client with ``get``/``set``.
    redis_url: Optional URL; ``redis`` is imported lazily on
      first use and failures fall back to memory.
    client: Alias for ``redis_client``.
    entity_col: Default entity column name.
    timestamp_col: Default timestamp column name.
    strict_pit: When True, ``get_features`` raises without
      a cutoff (training mode); a per-call ``strict_pit``
      argument overrides this default.
  '''

  def __init__(self, redis_client=None, redis_url=None, client=None,
               entity_col=ENTITY_COL, timestamp_col=None,
               strict_pit=False):
    '''Initialize with no imports of optional dependencies.'''
    self.redis_client = redis_client or client
    self.redis_url = redis_url
    self._redis = self.redis_client
    self.entity_col = entity_col
    self.timestamp_col = timestamp_col
    self.strict_pit = bool(strict_pit)
    # entity_id -> list of {'as_of': Timestamp, 'features': dict}
    self._store = {}
    self.store = self._store

  # -- Redis (lazy, best-effort) ---------------------------------
  def _client(self):
    '''Return a Redis client, creating it lazily if needed.

    Returns:
      The client, or None when Redis is unavailable.
    '''
    if self._redis is not None:
      return self._redis
    if not self.redis_url:
      return None
    try:
      import redis
    except ImportError as exc:
      LOG.debug('online feature store: redis package unavailable: %s',
                exc)
      return None
    try:
      self._redis = redis.Redis.from_url(self.redis_url)
    except (RuntimeError, OSError, TypeError, ValueError,
            AttributeError) as exc:
      LOG.warning('online feature store: redis from_url failed: %s;'
                  ' falling back to memory', exc)
      return None
    return self._redis

  def _redis_put(self, entity, records):
    '''Write ``records`` for ``entity`` to Redis, logging failures.

    Redis write failures are surfaced as a warning (never a
    silent pass) while the in-memory record remains served.
    Feature values are converted to plain Python types first
    so numpy values round-trip through JSON.
    '''
    client = self._client()
    if client is None:
      return
    try:
      payload = json.dumps([
          {'as_of': str(r['as_of']),
           'features': _native_features(r['features'])}
          for r in records])
      client.set(f'feature:{entity}', payload)
    except (TypeError, ValueError, RuntimeError, OSError,
            AttributeError) as exc:
      LOG.warning('online feature store: redis write failed for'
                  ' entity %s: %s; in-memory record retained',
                  subject_hash(entity), exc)

  def _redis_get(self, entity):
    '''Best-effort read; returns a record list or None.'''
    client = self._client()
    if client is None:
      return None
    try:
      raw = client.get(f'feature:{entity}')
    except (RuntimeError, OSError, TypeError, ValueError,
            AttributeError) as exc:
      LOG.warning('online feature store: redis read failed for'
                  ' entity %s: %s; using memory',
                  subject_hash(entity), exc)
      return None
    if raw is None:
      return None
    try:
      if isinstance(raw, bytes):
        raw = raw.decode('utf-8')
      data = json.loads(raw)
      records = [{'as_of': _naive_utc(r['as_of']),
                  'features': dict(r['features'])} for r in data]
      records.sort(key=lambda r: r['as_of'])
      return records
    except (TypeError, ValueError, UnicodeDecodeError,
            AttributeError, KeyError) as exc:
      LOG.warning('online feature store: bad redis payload for'
                  ' entity %s: %s; using memory',
                  subject_hash(entity), exc)
      return None

  def _records_for(self, entity):
    '''Return merged memory+Redis records for ``entity``.

    Both sources are merged by timestamp (latest wins per
    entity) instead of preferring one: stale Redis records
    never hide newer memory records, and newer Redis records
    from another writer are not lost. Duplicate timestamps
    keep the Redis copy (writers mirror memory to Redis).
    '''
    merged = []
    for record in self._store.get(entity, []):
      merged.append({
          'as_of': _naive_utc(record['as_of']),
          'features': dict(record['features']),
      })
    redis_records = self._redis_get(entity)
    if redis_records:
      merged.extend(redis_records)
    merged.sort(key=lambda r: r['as_of'])
    deduped = []
    for record in merged:
      if deduped and deduped[-1]['as_of'] == record['as_of']:
        deduped[-1] = record
      else:
        deduped.append(record)
    return deduped

  # -- Writes ------------------------------------------------------
  def set_features(self, entity_id, features, as_of=None,
                   request_time=None, pii_classes=None,
                   consent_ledger=None):
    '''Store ``features`` for one entity at ``as_of``.

    Args:
      entity_id: Entity id.
      features: Mapping of feature name to value. Numpy
        scalars/arrays are converted to plain Python values
        so they round-trip through Redis JSON.
      as_of: Timestamp for this record.
      request_time: Alias timestamp (wins over ``as_of``).
      pii_classes: Declared PII classes for the fail-closed
        write guard (explicit ``['none']`` asserts no PII).
      consent_ledger: Optional consent ledger; when supplied,
        a granted record per entity is required.

    Raises:
      ValueError: When ``features`` carries the entity
        column (``entity_id``) -- the guard frame always
        checks the explicit ``entity_id`` argument, so a
        smuggled entity value in ``features`` can never
        bypass the consent check -- or when validation /
        consent fails.
    '''
    from forecast_ml_framework.governance.pii import guard_write
    import pandas as pd
    feature_dict = dict(features)
    if ENTITY_COL in feature_dict:
      raise ValueError(
          'features must not contain the entity column'
          ' (entity_id): consent checks use the explicit'
          ' entity_id argument')
    guard_frame = pd.DataFrame(
        [{**feature_dict, ENTITY_COL: entity_id}])
    guard_write(guard_frame, pii_classes=pii_classes,
                consent_ledger=consent_ledger,
                entity_col=ENTITY_COL)
    when = _resolve_naive(as_of=as_of, request_time=request_time)
    if when is None:
      raise ValueError('set_features requires as_of')
    records = self._records_for(entity_id)
    records.append(
        {'as_of': when, 'features': _native_features(features)})
    records.sort(key=lambda r: r['as_of'])
    self._store[entity_id] = records
    self._redis_put(entity_id, records)

  # Aliases for the single-entity write.
  put_features = set_features
  set_feature = set_features

  def materialize(self, dataframe, as_of=None, feature_names=None,
                  entity_col=None, timestamp_col=None,
                  pii_classes=None, consent_ledger=None):
    '''Load features from ``dataframe`` into the online store.

    Args:
      dataframe: Source DataFrame.
      as_of: Fallback timestamp when the frame has none.
      feature_names: Optional feature columns to keep.
      entity_col: Entity column override.
      timestamp_col: Timestamp column override.
      pii_classes: Declared PII classes for the fail-closed
        write guard (explicit ``['none']`` asserts no PII).
      consent_ledger: Optional consent ledger; when supplied,
        a granted record per distinct entity is required.

    Returns:
      Number of entity records stored.
    '''
    _check_not_empty(dataframe)
    resolved_entity_col = entity_col or self.entity_col
    frame = _naive_utc_frame(_normalize_frame(
        dataframe, as_of=as_of,
        entity_col=resolved_entity_col,
        timestamp_col=timestamp_col or self.timestamp_col))
    if frame.empty:
      raise ValueError(
          'cannot materialize empty dataframe: '
          'online feature store is empty')
    # Normalized frame carries ENTITY_COL; pass it
    # explicitly to the guard (no silent default).
    from forecast_ml_framework.governance.pii import guard_write
    guard_write(frame, pii_classes=pii_classes,
                consent_ledger=consent_ledger,
                entity_col=ENTITY_COL)
    if feature_names is not None:
      keep = [ENTITY_COL, 'as_of'] + [
          n for n in feature_names if n in frame.columns]
      frame = frame[keep]
    count = 0
    for _, row in frame.iterrows():
      entity = row[ENTITY_COL]
      features = {name: row[name] for name in frame.columns
                  if name not in (ENTITY_COL, 'as_of')}
      records = self._records_for(entity)
      records.append(
          {'as_of': row['as_of'],
           'features': _native_features(features)})
      records.sort(key=lambda r: r['as_of'])
      self._store[entity] = records
      self._redis_put(entity, records)
      count += 1
    return count

  load_from_dataframe = materialize

  def delete(self, entity_id):
    '''Delete all records for ``entity_id`` (idempotent).

    Memory is always purged. Redis is purged only when a
    client exists (``_client() is not None``); a Redis error
    propagates to the caller after the memory purge, so the
    caller retries.

    Args:
      entity_id: Entity id to delete.
    '''
    self._store.pop(entity_id, None)
    client = self._client()
    if client is not None:
      client.delete(f'feature:{entity_id}')

  # -- Reads -------------------------------------------------------
  def get_features(self, entity_ids, feature_names=None, as_of=None,
                   request_time=None, strict_pit=None):
    '''Serve latest features per entity with ``as_of <= request``.

    Args:
      entity_ids: Entity ids to look up.
      feature_names: Feature names to return (all when None).
      as_of: Point-in-time cutoff.
      request_time: Point-in-time cutoff (wins over ``as_of``).
      strict_pit: When True (or when the store was built
        with ``strict_pit=True`` and this is None), raise
        ``ValueError`` without a cutoff instead of warning.

    Without a cutoff this returns the latest available
    features and warns; callers must pass a cutoff for
    point-in-time-correct (training) use to avoid future
    leakage.

    Returns:
      Dict mapping each found entity id to its feature dict.
      Future-only entities are omitted.

    Raises:
      ValueError: When strict mode is on and no cutoff is
        given, or a requested feature is unknown.
    '''
    import warnings
    when = _resolve_naive(as_of=as_of, request_time=request_time)
    strict = self.strict_pit if strict_pit is None else bool(
        strict_pit)
    if when is None and strict:
      raise ValueError(
          'strict_pit=True requires an as_of/request_time'
          ' cutoff; refusing to return latest features'
          ' without a point-in-time cutoff')
    if when is None:
      warnings.warn(
          'get_features without a cutoff returns the latest '
          'available features; pass as_of/request_time for '
          'point-in-time correctness (future leakage risk)',
          UserWarning, stacklevel=2)
    if feature_names is not None:
      # Union of feature names across all stored records.
      known = set()
      for stored in self._store.values():
        for record in stored:
          known.update(record.get('features', {}))
      # Also consider Redis-backed entities requested here.
      for entity in entity_ids:
        for record in self._records_for(entity):
          known.update(record.get('features', {}))
      unknown = [n for n in feature_names if n not in known]
      if unknown:
        raise ValueError(f'Unknown features: {unknown}')
    result = {}
    for entity in entity_ids:
      chosen = None
      records = sorted(
          self._records_for(entity), key=lambda r: r['as_of'])
      for record in records:
        if when is None or record['as_of'] <= when:
          chosen = record
      if chosen is None:
        continue
      if feature_names is not None:
        result[entity] = {name: chosen['features'][name]
                          for name in feature_names
                          if name in chosen['features']}
      else:
        result[entity] = dict(chosen['features'])
    return result

  def get_features_df(self, entity_ids, feature_names=None,
                      as_of=None, request_time=None,
                      strict_pit=None):
    '''DataFrame view of :meth:`get_features` (for callers that
    expect a frame, e.g. consistency checks).'''
    import pandas as pd
    found = self.get_features(
        entity_ids, feature_names, as_of=as_of,
        request_time=request_time, strict_pit=strict_pit)
    names = list(feature_names) if feature_names is not None else sorted(
        {k for v in found.values() for k in v})
    rows = []
    for entity in entity_ids:
      if entity not in found:
        continue
      row = {ENTITY_COL: entity}
      row.update(found[entity])
      rows.append(row)
    return pd.DataFrame(rows, columns=[ENTITY_COL] + names)
