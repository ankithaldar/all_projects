#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Feature store abstractions and offline/online consistency.'''
# pylint: disable=import-outside-toplevel, broad-exception-caught

import abc

ENTITY_COL = 'entity_id'
TIMESTAMP_COLS = ('as_of', 'event_timestamp', 'timestamp',
                  'event_time', 'date')


def _to_timestamp(value):
  '''Parse ``value`` to a pandas Timestamp (lazy pandas import).

  Args:
    value: String, date, datetime, or Timestamp.

  Returns:
    The parsed Timestamp, or None for None input.
  '''
  if value is None:
    return None
  import pandas as pd
  return pd.Timestamp(value)


def _resolve_time(as_of=None, request_time=None):
  '''Return request time, preferring ``request_time`` over ``as_of``.'''
  return _to_timestamp(
      request_time if request_time is not None else as_of)


def _find_timestamp_col(df, timestamp_col=None):
  '''Return the timestamp column present in ``df``, or None.'''
  if timestamp_col and timestamp_col in df.columns:
    return timestamp_col
  for col in TIMESTAMP_COLS:
    if col in df.columns:
      return col
  return None


def _find_entity_col(df, entity_col=None):
  '''Return the entity column present in ``df``, or None.'''
  if entity_col and entity_col in df.columns:
    return entity_col
  for col in (ENTITY_COL, 'entity', 'id'):
    if col in df.columns:
      return col
  return None


def _normalize_frame(df, as_of=None, entity_col=None,
                     timestamp_col=None):
  '''Copy ``df`` with canonical entity/as_of columns.

  Args:
    df: Source DataFrame (pandas, imported lazily).
    as_of: Fallback timestamp when ``df`` has no timestamp column.
    entity_col: Entity column name override.
    timestamp_col: Timestamp column name override.

  Returns:
    A copy with ``entity_id`` and ``as_of`` columns present.

  Raises:
    ValueError: When no entity column or no timestamp is available.
  '''
  import pandas as pd
  frame = df.copy()
  found_entity = _find_entity_col(frame, entity_col)
  if found_entity is None:
    raise ValueError('Dataframe requires an entity_id column')
  if found_entity != ENTITY_COL:
    frame = frame.rename(columns={found_entity: ENTITY_COL})
  found_ts = _find_timestamp_col(frame, timestamp_col)
  if found_ts is None:
    if as_of is None:
      raise ValueError('Dataframe requires an as_of timestamp')
    frame['as_of'] = _to_timestamp(as_of)
  elif found_ts != 'as_of':
    frame = frame.rename(columns={found_ts: 'as_of'})
  frame['as_of'] = pd.to_datetime(frame['as_of'])
  return frame


def _version_key(prefix, as_of):
  '''Return the versioned artifact key for ``as_of``.

  Args:
    prefix: Key prefix (no trailing slash handling needed beyond
      stripping).
    as_of: Date/datetime/string version.

  Returns:
    Key like ``features/2024-01-03T09-00-00.pkl``. The full
    timestamp (not just the date) is part of the key so that
    multiple snapshots on the same day coexist instead of
    overwriting each other.
  '''
  stamp = _to_timestamp(as_of)
  date = stamp.strftime('%Y-%m-%dT%H-%M-%S') if stamp is not None \
      else 'latest'
  base = str(prefix or 'features').strip('/')
  return f'{base}/{date}.pkl'


def _snapshot_timestamp(key):
  '''Parse the snapshot timestamp encoded in an artifact key.

  Args:
    key: Artifact key like ``features/2024-01-03T09-00-00.pkl``
      or legacy ``features/2024-01-03.pkl``.

  Returns:
    The parsed Timestamp, or None when the key carries no
    parseable date/timestamp stem.
  '''
  from datetime import datetime
  stem = str(key).rsplit('/', 1)[-1]
  if stem.endswith('.pkl'):
    stem = stem[:-4]
  for fmt in ('%Y-%m-%dT%H-%M-%S', '%Y-%m-%d'):
    try:
      return _strip_tz(_to_timestamp(datetime.strptime(stem, fmt)))
    except ValueError:
      continue
  try:
    return _strip_tz(_to_timestamp(stem))
  except Exception:
    return None


def _strip_tz(stamp):
  '''Return ``stamp`` without timezone info (or None).'''
  if stamp is None:
    return None
  if getattr(stamp, 'tzinfo', None) is not None:
    return _to_timestamp(stamp.to_pydatetime().replace(tzinfo=None))
  return stamp


def _naive_utc(value):
  '''Parse ``value`` to a naive UTC Timestamp (or None).'''
  if value is None:
    return None
  import pandas as pd
  stamp = pd.Timestamp(value)
  if getattr(stamp, 'tzinfo', None) is not None:
    stamp = stamp.tz_convert('UTC').tz_localize(None)
  return stamp


def _naive_utc_frame(frame):
  '''Return ``frame`` with tz-aware as_of converted to naive UTC.'''
  import pandas as pd
  frame = frame.copy()
  if 'as_of' in frame.columns and len(frame):
    converted = pd.to_datetime(frame['as_of'])
    if getattr(converted.dtype, 'tz', None) is not None:
      converted = converted.dt.tz_convert('UTC').dt.tz_localize(None)
    frame['as_of'] = converted
  return frame


def _check_not_empty(dataframe):
  '''Raise an informative ValueError for an empty dataframe.'''
  try:
    empty = dataframe is None or len(dataframe) == 0
  except TypeError:
    empty = False
  if empty:
    raise ValueError(
        'cannot materialize empty dataframe: '
        'feature store is empty')


def _offline_as_dict(offline, entity_ids, feature_names, when):
  '''Read offline features as ``{entity: {feature: value}}``.'''
  out = offline.get_features(
      entity_ids, feature_names, as_of=when)
  result = {}
  try:
    import pandas as pd
    if isinstance(out, pd.DataFrame):
      for _, row in out.iterrows():
        result[str(row[ENTITY_COL])] = {
            name: row[name] for name in feature_names}
      return result
  except Exception:
    pass
  if isinstance(out, dict):
    return {str(k): dict(v) for k, v in out.items()}
  return result


def _online_as_dict(online, entity_ids, feature_names, when):
  '''Read online features as ``{entity: {feature: value}}``.'''
  out = online.get_features(
      entity_ids, feature_names, as_of=when)
  if isinstance(out, dict):
    return {str(k): dict(v) for k, v in out.items()}
  result = {}
  try:
    import pandas as pd
    if isinstance(out, pd.DataFrame):
      for _, row in out.iterrows():
        result[str(row[ENTITY_COL])] = {
            name: row[name] for name in feature_names}
  except Exception:
    pass
  return result


def check_consistency(offline, online, entity_ids, feature_names,
                      as_of=None, request_time=None):
  '''Compare offline vs online values for sample entities.

  Args:
    offline: Store with ``get_features`` returning a DataFrame.
    online: Store with ``get_features`` returning a dict or frame.
    entity_ids: Sample entity ids to compare.
    feature_names: Feature names to compare.
    as_of: Point-in-time cutoff.
    request_time: Point-in-time cutoff (wins over ``as_of``).

  Returns:
    Dict with ``max_abs_diff`` (float), ``n_compared`` (int), and
    ``diffs`` mapping ``entity:feature`` to absolute differences.
  '''
  when = _resolve_time(as_of=as_of, request_time=request_time)
  off = _offline_as_dict(offline, entity_ids, feature_names, when)
  on = _online_as_dict(online, entity_ids, feature_names, when)
  diffs = {}
  for entity in entity_ids:
    entity = str(entity)
    if entity not in off or entity not in on:
      continue
    for name in feature_names:
      if name not in off[entity] or name not in on[entity]:
        continue
      try:
        diff = abs(float(off[entity][name]) - float(on[entity][name]))
      except (TypeError, ValueError):
        continue
      diffs[f'{entity}:{name}'] = diff
  return {
      'max_abs_diff': max(diffs.values()) if diffs else 0.0,
      'n_compared': len(diffs),
      'diffs': diffs,
  }


# Backwards / forwards-compatible aliases for the same check.
consistency_check = check_consistency
compare_offline_online = check_consistency


class FeatureStore(abc.ABC):
  '''Abstract feature store interface.

  Implementations materialize features from a DataFrame and serve
  point-in-time-correct features: only rows with ``as_of`` less
  than or equal to the request time are returned.
  '''

  @abc.abstractmethod
  def materialize(self, dataframe, as_of=None, feature_names=None,
                  entity_col=None, timestamp_col=None):
    '''Persist features from ``dataframe``.

    Args:
      dataframe: Source DataFrame.
      as_of: Version / fallback timestamp.
      feature_names: Optional feature columns to keep.
      entity_col: Entity column override.
      timestamp_col: Timestamp column override.

    Returns:
      Identifier for the materialized snapshot (e.g. path/key).
    '''

  @abc.abstractmethod
  def get_features(self, entity_ids, feature_names=None, as_of=None,
                   request_time=None):
    '''Serve features for ``entity_ids`` at a point in time.

    Without a cutoff (both ``as_of``/``request_time`` None)
    implementations return the latest available features and
    warn; callers must pass a cutoff for point-in-time-correct
    (training) use to avoid future leakage.

    Args:
      entity_ids: Entity ids to look up.
      feature_names: Feature names to return (all when None).
      as_of: Point-in-time cutoff.
      request_time: Point-in-time cutoff (wins over ``as_of``).

    Returns:
      Served features (DataFrame offline, dict online).
    '''
