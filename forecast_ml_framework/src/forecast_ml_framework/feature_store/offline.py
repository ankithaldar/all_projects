#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Offline feature store backed by an ArtifactStore.'''
# pylint: disable=import-outside-toplevel, broad-exception-caught

from forecast_ml_framework.feature_store.store import (
    ENTITY_COL, FeatureStore, _check_not_empty, _naive_utc,
    _naive_utc_frame, _normalize_frame, _resolve_time,
    _snapshot_timestamp, _version_key)


class OfflineFeatureStore(FeatureStore):
  '''Materialize DataFrames to a versioned artifact path.

  Args:
    artifact_store: ArtifactStore-like object with
      ``save``/``load``/``exists``/``list``. A filesystem path
      (str/Path) is wrapped in ``LocalArtifactStore``; None
      defaults to ``LocalArtifactStore('./artifacts')``.
    base_path: Root when ``artifact_store`` is a path or None.
    prefix: Artifact key prefix for feature snapshots.
    entity_col: Default entity column name.
    timestamp_col: Default timestamp column name.
  '''

  def __init__(self, artifact_store=None, base_path=None,
               prefix='features', entity_col=ENTITY_COL,
               timestamp_col=None):
    '''Initialize with an artifact store (lazy imports only).'''
    self.prefix = prefix
    self.entity_col = entity_col
    self.timestamp_col = timestamp_col
    if artifact_store is not None and not hasattr(
            artifact_store, 'save'):
      from forecast_ml_framework.core.artifact_store import (
          LocalArtifactStore)
      artifact_store = LocalArtifactStore(str(artifact_store))
    if artifact_store is None:
      from forecast_ml_framework.core.artifact_store import (
          LocalArtifactStore)
      artifact_store = LocalArtifactStore(base_path or './artifacts')
    self.artifact_store = artifact_store
    self.store = artifact_store

  def materialize(self, dataframe, as_of=None, feature_names=None,
                  entity_col=None, timestamp_col=None):
    '''Save ``dataframe`` versioned by ``as_of`` date.

    Args:
      dataframe: Source DataFrame.
      as_of: Version date; defaults to the max timestamp in the
        frame when omitted.
      feature_names: Optional columns to keep besides entity/as_of.
      entity_col: Entity column override.
      timestamp_col: Timestamp column override.

    Returns:
      The artifact key the snapshot was saved to.

    Raises:
      ValueError: When the frame lacks an entity column.
    '''
    import pandas as pd
    _check_not_empty(dataframe)
    frame = _naive_utc_frame(_normalize_frame(
        dataframe, as_of=as_of,
        entity_col=entity_col or self.entity_col,
        timestamp_col=timestamp_col or self.timestamp_col))
    if frame.empty:
      raise ValueError(
          'cannot materialize empty dataframe: '
          'offline feature store is empty')
    if feature_names is not None:
      keep = [ENTITY_COL, 'as_of'] + [
          name for name in feature_names if name in frame.columns]
      frame = frame[keep].copy()
    version = _naive_utc(as_of) if as_of is not None \
        else frame['as_of'].max()
    key = _version_key(self.prefix, version)
    self.artifact_store.save(key, frame)
    return key

  # Alias matching ArtifactStore-style naming.
  save = materialize

  def _all_keys(self):
    '''Return all snapshot keys under the prefix, sorted.'''
    try:
      keys = list(self.artifact_store.list(self.prefix))
    except Exception:
      keys = []
    if keys:
      return sorted(keys)
    # Fall back to listing everything (some stores ignore prefix).
    try:
      all_keys = list(self.artifact_store.list(''))
    except Exception:
      all_keys = []
    return sorted(k for k in all_keys if self.prefix in k)

  def _load_frame(self, key):
    '''Load one snapshot frame, returning None on failure.'''
    try:
      loaded = self.artifact_store.load(key)
    except Exception:
      return None
    import pandas as pd
    if isinstance(loaded, pd.DataFrame):
      return _naive_utc_frame(_normalize_frame(loaded))
    return None

  def load(self, as_of=None, key=None):
    '''Load a snapshot by key, exact key, or ``as_of`` date.

    Args:
      as_of: Version date/timestamp. A full timestamp selects
        the latest snapshot at or before it (point-in-time);
        a date-only value selects the latest snapshot on that
        day. Legacy date-only keys are also matched.
      key: Exact artifact key.

    Returns:
      The snapshot DataFrame.

    Raises:
      FileNotFoundError: When no matching snapshot exists.
    '''
    if key is not None and self.artifact_store.exists(key):
      frame = self._load_frame(key)
      if frame is not None:
        return frame
    if as_of is not None:
      import re
      text = str(as_of)
      candidates = []
      for candidate in self._all_keys():
        stamp = _snapshot_timestamp(candidate)
        if stamp is not None:
          candidates.append((_naive_utc(stamp), candidate))
      try:
        parsed = _naive_utc(as_of)
      except Exception:  # pylint: disable=broad-exception-caught
        parsed = None
      parsed_ok = parsed is not None
      if candidates and parsed_ok:
        if re.fullmatch(r'\d{4}-\d{2}-\d{2}', text.strip()):
          day = parsed
          same_day = [(s, k) for s, k in candidates
                      if s.normalize() == day.normalize()]
          if same_day:
            frame = self._load_frame(sorted(same_day)[-1][1])
            if frame is not None:
              return frame
        else:
          when = parsed
          eligible = [(s, k) for s, k in candidates if s <= when]
          if eligible:
            frame = self._load_frame(sorted(eligible)[-1][1])
            if frame is not None:
              return frame
          # No snapshot at or before the request: do not fall
          # back to a future snapshot.
          raise FileNotFoundError(
              f'No offline feature snapshot for as_of={as_of!r}')
      if not candidates or not parsed_ok:
        for candidate in self._all_keys():
          if text in candidate:
            frame = self._load_frame(candidate)
            if frame is not None:
              return frame
      if parsed_ok:
        try:
          direct = _version_key(self.prefix, parsed)
        except Exception:  # pylint: disable=broad-exception-caught
          direct = None
        if direct is not None and self.artifact_store.exists(direct):
          return self._load_frame(direct)
    raise FileNotFoundError(
        f'No offline feature snapshot for as_of={as_of!r}')

  def _combined_frame(self, when=None, config=None):
    '''Concatenate snapshots PIT-filtered to ``when``, or empty.

    Each loaded frame is row-filtered to ``as_of <= when``
    before concatenation, so future rows never enter the
    combined frame. Each loaded frame and the combined
    frame are guarded by ``check_pandas_limits`` against
    a row cap instead of growing unbounded.

    Args:
      when: Point-in-time cutoff; None loads every row.
      config: Optional ``check_pandas_limits`` config.

    Returns:
      Concatenated frame, or an empty frame when no snapshot
      is eligible.
    '''
    import pandas as pd
    from forecast_ml_framework.core.spark_factory import (
      check_pandas_limits)
    when_ts = _naive_utc(when) if when is not None else None
    frames = []
    seen_columns = [ENTITY_COL, 'as_of']
    for key in self._all_keys():
      frame = self._load_frame(key)
      if frame is not None and not frame.empty:
        for col in frame.columns:
          if col not in seen_columns:
            seen_columns.append(col)
        if when_ts is not None and 'as_of' in frame.columns:
          frame = frame[frame['as_of'] <= when_ts]
        if frame.empty:
          continue
        check_pandas_limits(frame, config)
        frames.append(frame)
    if not frames:
      return pd.DataFrame(columns=seen_columns)
    combined = pd.concat(frames, ignore_index=True)
    check_pandas_limits(combined, config)
    return combined

  def get_features(self, entity_ids, feature_names=None, as_of=None,
                   request_time=None):
    '''Serve latest features per entity with ``as_of <= request``.

    Args:
      entity_ids: Entity ids to look up (order preserved).
      feature_names: Feature columns to return (all non-key
        columns when None).
      as_of: Point-in-time cutoff.
      request_time: Point-in-time cutoff (wins over ``as_of``).

    Without a cutoff (both ``as_of``/``request_time`` None) this
    returns the latest available snapshot and warns: training
    callers must pass a cutoff to avoid future leakage.

    Returns:
      DataFrame with ``entity_id`` plus requested features,
      one row per found entity, in ``entity_ids`` order.

    Raises:
      ValueError: When a requested feature is unknown.
    '''
    import warnings
    import pandas as pd
    raw_when = _resolve_time(as_of=as_of, request_time=request_time)
    when = _naive_utc(raw_when) if raw_when is not None else None
    if when is None:
      warnings.warn(
          'get_features without a cutoff returns the latest '
          'available features; pass as_of/request_time for '
          'point-in-time correctness (future leakage risk)',
          UserWarning, stacklevel=2)
    frame = self._combined_frame(when=when)
    available = [c for c in frame.columns
                 if c not in (ENTITY_COL, 'as_of')]
    names = list(feature_names) if feature_names is not None \
        else available
    unknown = [n for n in names if n not in available]
    if unknown:
      raise ValueError(f'Unknown features: {unknown}')
    if frame.empty:
      cols = [ENTITY_COL] + list(feature_names or [])
      return pd.DataFrame(columns=cols)
    if when is not None:
      frame = frame[frame['as_of'] <= when]
    if frame.empty:
      return pd.DataFrame(columns=[ENTITY_COL] + names)
    frame = frame.sort_values('as_of')
    latest = frame.drop_duplicates(subset=[ENTITY_COL], keep='last')
    latest = latest.set_index(ENTITY_COL)
    rows = []
    for entity in entity_ids:
      if entity not in latest.index:
        continue
      row = {ENTITY_COL: entity}
      for name in names:
        row[name] = latest.loc[entity, name]
      rows.append(row)
    return pd.DataFrame(rows, columns=[ENTITY_COL] + names)
