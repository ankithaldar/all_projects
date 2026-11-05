#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Online feature store: in-memory dict plus optional Redis.'''
# pylint: disable=import-outside-toplevel, broad-exception-caught

import json

from forecast_ml_framework.feature_store.store import (
    ENTITY_COL, FeatureStore, _normalize_frame, _resolve_time)


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
  '''

  def __init__(self, redis_client=None, redis_url=None, client=None,
               entity_col=ENTITY_COL, timestamp_col=None):
    '''Initialize with no imports of optional dependencies.'''
    self.redis_client = redis_client or client
    self.redis_url = redis_url
    self._redis = self.redis_client
    self.entity_col = entity_col
    self.timestamp_col = timestamp_col
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
    except ImportError:
      return None
    try:
      self._redis = redis.Redis.from_url(self.redis_url)
    except Exception:
      return None
    return self._redis

  def _redis_put(self, entity, records):
    '''Best-effort write of ``records`` for ``entity`` to Redis.'''
    client = self._client()
    if client is None:
      return
    try:
      payload = json.dumps([
          {'as_of': str(r['as_of']), 'features': r['features']}
          for r in records])
      client.set(f'feature:{entity}', payload)
    except Exception:
      pass

  def _redis_get(self, entity):
    '''Best-effort read; returns a record list or None.'''
    client = self._client()
    if client is None:
      return None
    try:
      raw = client.get(f'feature:{entity}')
    except Exception:
      return None
    if raw is None:
      return None
    try:
      if isinstance(raw, bytes):
        raw = raw.decode('utf-8')
      data = json.loads(raw)
      from forecast_ml_framework.feature_store.store import (
          _to_timestamp)
      records = [{'as_of': _to_timestamp(r['as_of']),
                  'features': dict(r['features'])} for r in data]
      records.sort(key=lambda r: r['as_of'])
      return records
    except Exception:
      return None

  def _records_for(self, entity):
    '''Return stored records for ``entity`` (Redis wins if set).'''
    redis_records = self._redis_get(entity)
    if redis_records:
      return redis_records
    return list(self._store.get(entity, []))

  # -- Writes ------------------------------------------------------
  def set_features(self, entity_id, features, as_of=None,
                   request_time=None):
    '''Store ``features`` for one entity at ``as_of``.

    Args:
      entity_id: Entity id.
      features: Mapping of feature name to value.
      as_of: Timestamp for this record.
      request_time: Alias timestamp (wins over ``as_of``).
    '''
    when = _resolve_time(as_of=as_of, request_time=request_time)
    if when is None:
      raise ValueError('set_features requires as_of')
    records = self._records_for(entity_id)
    records.append(
        {'as_of': when, 'features': dict(features)})
    records.sort(key=lambda r: r['as_of'])
    self._store[entity_id] = records
    self._redis_put(entity_id, records)

  # Aliases for the single-entity write.
  put_features = set_features
  set_feature = set_features

  def materialize(self, dataframe, as_of=None, feature_names=None,
                  entity_col=None, timestamp_col=None):
    '''Load features from ``dataframe`` into the online store.

    Args:
      dataframe: Source DataFrame.
      as_of: Fallback timestamp when the frame has none.
      feature_names: Optional feature columns to keep.
      entity_col: Entity column override.
      timestamp_col: Timestamp column override.

    Returns:
      Number of entity records stored.
    '''
    frame = _normalize_frame(
        dataframe, as_of=as_of,
        entity_col=entity_col or self.entity_col,
        timestamp_col=timestamp_col or self.timestamp_col)
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
          {'as_of': row['as_of'], 'features': features})
      records.sort(key=lambda r: r['as_of'])
      self._store[entity] = records
      self._redis_put(entity, records)
      count += 1
    return count

  load_from_dataframe = materialize

  # -- Reads -------------------------------------------------------
  def get_features(self, entity_ids, feature_names=None, as_of=None,
                   request_time=None):
    '''Serve latest features per entity with ``as_of <= request``.

    Args:
      entity_ids: Entity ids to look up.
      feature_names: Feature names to return (all when None).
      as_of: Point-in-time cutoff.
      request_time: Point-in-time cutoff (wins over ``as_of``).

    Without a cutoff this returns the latest available
    features and warns; callers must pass a cutoff for
    point-in-time-correct (training) use to avoid future
    leakage.

    Returns:
      Dict mapping each found entity id to its feature dict.
      Future-only entities are omitted.
    '''
    import warnings
    when = _resolve_time(as_of=as_of, request_time=request_time)
    if when is None:
      warnings.warn(
          'get_features without a cutoff returns the latest '
          'available features; pass as_of/request_time for '
          'point-in-time correctness (future leakage risk)',
          UserWarning, stacklevel=2)
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
                      as_of=None, request_time=None):
    '''DataFrame view of :meth:`get_features` (for callers that
    expect a frame, e.g. consistency checks).'''
    import pandas as pd
    found = self.get_features(
        entity_ids, feature_names, as_of=as_of,
        request_time=request_time)
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
