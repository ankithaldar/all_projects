#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests for forecast_ml_framework.feature_store.'''

import pytest


def _df():
  import pandas as pd
  return pd.DataFrame({
      'entity_id': ['e1', 'e1', 'e2'],
      'as_of': ['2024-01-01', '2024-01-03', '2024-01-02'],
      'f1': [1.0, 3.0, 2.0],
      'f2': [10.0, 30.0, 20.0],
  })


def _offline(tmp_path):
  from forecast_ml_framework.core.artifact_store import (
      LocalArtifactStore)
  from forecast_ml_framework.feature_store.offline import (
      OfflineFeatureStore)
  return OfflineFeatureStore(LocalArtifactStore(str(tmp_path)))


def test_feature_store_abc_not_instantiable():
  from forecast_ml_framework.feature_store.store import FeatureStore
  with pytest.raises(TypeError):
    FeatureStore()


def test_offline_materialize_serve_roundtrip(tmp_path):
  offline = _offline(tmp_path)
  key = offline.materialize(_df(), as_of='2024-01-03')
  assert key
  assert offline.artifact_store.exists(key)
  out = offline.get_features(
      ['e1', 'e2'], ['f1', 'f2'], as_of='2024-01-03')
  assert list(out['entity_id']) == ['e1', 'e2']
  assert list(out['f1']) == [3.0, 2.0]
  # Fresh instance over same artifact root sees the same data.
  fresh = _offline(tmp_path)
  again = fresh.get_features(['e1'], ['f1'], as_of='2024-01-03')
  assert list(again['f1']) == [3.0]


def test_offline_versioned_by_as_of(tmp_path):
  offline = _offline(tmp_path)
  k1 = offline.materialize(_df(), as_of='2024-01-01')
  k2 = offline.materialize(_df(), as_of='2024-01-03')
  assert k1 != k2
  assert '2024-01-01' in k1
  assert '2024-01-03' in k2
  loaded = offline.load('2024-01-01')
  assert len(loaded) == 3


def test_offline_point_in_time_excludes_future(tmp_path):
  offline = _offline(tmp_path)
  offline.materialize(_df(), as_of='2024-01-03')
  out = offline.get_features(
      ['e1'], ['f1'], as_of='2024-01-02')
  # e1's 2024-01-03 row is in the future; latest <= request is 01-01.
  assert list(out['f1']) == [1.0]
  out = offline.get_features(
      ['e1'], ['f1'], request_time='2024-01-03')
  assert list(out['f1']) == [3.0]


def test_offline_materialize_requires_entity_col(tmp_path):
  import pandas as pd
  offline = _offline(tmp_path)
  with pytest.raises(ValueError):
    offline.materialize(
        pd.DataFrame({'f1': [1.0]}), as_of='2024-01-01')


def test_online_roundtrip_and_point_in_time():
  from forecast_ml_framework.feature_store.online import (
      OnlineFeatureStore)
  online = OnlineFeatureStore()
  online.materialize(_df(), as_of='2024-01-03')
  out = online.get_features(['e1', 'e2'], ['f1'], as_of='2024-01-02')
  assert out['e1']['f1'] == 1.0
  assert out['e2']['f1'] == 2.0
  out = online.get_features(['e1'], ['f1', 'f2'], as_of='2024-01-03')
  assert out['e1'] == {'f1': 3.0, 'f2': 30.0}
  # Future-only entity is absent before its as_of.
  assert online.get_features(['e2'], ['f1'], as_of='2024-01-01') == {}


def test_online_set_features_point_in_time():
  from forecast_ml_framework.feature_store.online import (
      OnlineFeatureStore)
  online = OnlineFeatureStore()
  online.set_features('e1', {'f1': 5.0}, as_of='2024-01-05')
  assert online.get_features(['e1'], ['f1'], as_of='2024-01-04') == {}
  assert online.get_features(
      ['e1'], ['f1'], as_of='2024-01-05') == {'e1': {'f1': 5.0}}


def test_online_lazy_redis_optional_and_fake_client():
  from forecast_ml_framework.feature_store.online import (
      OnlineFeatureStore)
  # No redis installed / no url: plain in-memory works.
  plain = OnlineFeatureStore(redis_url='redis://localhost:6379/0')
  plain.set_features('e1', {'f1': 1.0}, as_of='2024-01-01')
  assert plain.get_features(['e1'], ['f1'], as_of='2024-01-01') == {
      'e1': {'f1': 1.0}}

  class FakeRedis:
    def __init__(self):
      self.data = {}

    def set(self, key, value):
      self.data[key] = value

    def get(self, key):
      return self.data.get(key)

  fake = FakeRedis()
  online = OnlineFeatureStore(redis_client=fake)
  online.set_features('e1', {'f1': 7.0}, as_of='2024-01-01')
  assert fake.data
  assert online.get_features(['e1'], ['f1'], as_of='2024-01-01') == {
      'e1': {'f1': 7.0}}


def test_consistency_zero_for_identical(tmp_path):
  from forecast_ml_framework.feature_store.online import (
      OnlineFeatureStore)
  from forecast_ml_framework.feature_store.store import (
      check_consistency)
  offline = _offline(tmp_path)
  offline.materialize(_df(), as_of='2024-01-03')
  online = OnlineFeatureStore()
  online.materialize(_df(), as_of='2024-01-03')
  report = check_consistency(
      offline, online, ['e1', 'e2'], ['f1', 'f2'],
      as_of='2024-01-03')
  assert report['max_abs_diff'] == 0.0
  assert report['n_compared'] == 4


def test_consistency_reports_diff_for_diverged(tmp_path):
  from forecast_ml_framework.feature_store.online import (
      OnlineFeatureStore)
  from forecast_ml_framework.feature_store.store import (
      check_consistency)
  offline = _offline(tmp_path)
  offline.materialize(_df(), as_of='2024-01-03')
  online = OnlineFeatureStore()
  online.set_features('e1', {'f1': 4.5}, as_of='2024-01-03')
  report = check_consistency(
      offline, online, ['e1'], ['f1'], as_of='2024-01-03')
  assert report['max_abs_diff'] == pytest.approx(1.5)


def test_package_exports():
  import forecast_ml_framework.feature_store as fs
  assert fs.FeatureStore is not None
  assert fs.OfflineFeatureStore is not None
  assert fs.OnlineFeatureStore is not None
  assert fs.check_consistency is not None


def test_offline_same_day_pit_preserves_morning_snapshot(tmp_path):
  import pandas as pd
  offline = _offline(tmp_path)
  morning = pd.DataFrame({
      'entity_id': ['e1'],
      'as_of': ['2024-01-03 09:00:00'],
      'f1': [1.0],
  })
  evening = pd.DataFrame({
      'entity_id': ['e1'],
      'as_of': ['2024-01-03 18:00:00'],
      'f1': [3.0],
  })
  k_morning = offline.materialize(
      morning, as_of='2024-01-03T09:00:00')
  k_evening = offline.materialize(
      evening, as_of='2024-01-03T18:00:00')
  assert k_morning != k_evening
  assert '2024-01-03T09-00-00' in k_morning
  assert '2024-01-03T18-00-00' in k_evening
  midday = offline.get_features(
      ['e1'], ['f1'], request_time='2024-01-03T12:00:00')
  assert list(midday['f1']) == [1.0]
  late = offline.get_features(
      ['e1'], ['f1'], request_time='2024-01-03T19:00:00')
  assert list(late['f1']) == [3.0]
  loaded = offline.load('2024-01-03T12:00:00')
  assert list(loaded['f1']) == [1.0]
  loaded_dashed = offline.load('2024-01-03T12-00-00')
  assert list(loaded_dashed['f1']) == [1.0]


def test_online_redis_unsorted_payload_still_point_in_time():
  import json
  from forecast_ml_framework.feature_store.online import (
      OnlineFeatureStore)

  class FakeRedis:
    def __init__(self):
      # Reversed: future record first.
      self.data = {'feature:e1': json.dumps([
          {'as_of': '2024-01-03', 'features': {'f1': 3.0}},
          {'as_of': '2024-01-01', 'features': {'f1': 1.0}},
      ])}

    def set(self, key, value):
      self.data[key] = value

    def get(self, key):
      return self.data.get(key)

  online = OnlineFeatureStore(redis_client=FakeRedis())
  out = online.get_features(['e1'], ['f1'], as_of='2024-01-02')
  assert out == {'e1': {'f1': 1.0}}


def test_get_features_without_cutoff_warns_and_returns_latest(tmp_path):
  import warnings
  from forecast_ml_framework.feature_store.online import (
      OnlineFeatureStore)
  offline = _offline(tmp_path)
  offline.materialize(_df(), as_of='2024-01-03')
  with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always')
    out = offline.get_features(['e1'], ['f1'])
  assert any(issubclass(w.category, UserWarning) for w in caught)
  assert list(out['f1']) == [3.0]

  online = OnlineFeatureStore()
  online.materialize(_df(), as_of='2024-01-03')
  with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always')
    result = online.get_features(['e1'], ['f1'])
  assert any(issubclass(w.category, UserWarning) for w in caught)
  assert result == {'e1': {'f1': 3.0}}


def test_offline_materialize_empty_raises(tmp_path):
  import pandas as pd
  offline = _offline(tmp_path)
  with pytest.raises(ValueError, match='(?i)empty'):
    offline.materialize(
        pd.DataFrame({'entity_id': [], 'as_of': [], 'f1': []}),
        as_of='2024-01-01')


def test_offline_get_features_unknown_on_empty_raises(tmp_path):
  offline = _offline(tmp_path)
  with pytest.raises(ValueError):
    offline.get_features(
        ['e1'], ['nope'], as_of='2024-01-03')


# --- Coverage: store helpers / offline / online branches ---


def test_store_helpers_entity_timestamp_and_version():
  import pandas as pd
  from forecast_ml_framework.feature_store.store import (
    _find_entity_col, _find_timestamp_col, _normalize_frame, _version_key)
  frame = pd.DataFrame({'id': ['e1'], 'timestamp': ['2024-01-01'],
                        'f1': [1.0]})
  assert _find_entity_col(frame) == 'id'
  assert _find_entity_col(frame, 'id') == 'id'
  assert _find_entity_col(frame, 'missing') == 'id'
  assert _find_timestamp_col(frame) == 'timestamp'
  assert _find_timestamp_col(frame, 'timestamp') == 'timestamp'
  assert _find_timestamp_col(frame, 'missing') == 'timestamp'
  assert _find_timestamp_col(pd.DataFrame({'x': [1]})) is None
  assert _find_entity_col(pd.DataFrame({'x': [1]})) is None
  assert _version_key('features', None) == 'features/latest.pkl'
  assert _version_key('/features/', '2024-01-03') == (
      'features/2024-01-03T00-00-00.pkl')
  normalized = _normalize_frame(frame)
  assert list(normalized.columns) == ['entity_id', 'as_of', 'f1']
  custom = _normalize_frame(frame, entity_col='id',
                            timestamp_col='timestamp')
  assert custom['entity_id'].iloc[0] == 'e1'


def test_normalize_frame_missing_entity_and_timestamp():
  import pandas as pd
  from forecast_ml_framework.feature_store.store import _normalize_frame
  with pytest.raises(ValueError, match='entity_id'):
    _normalize_frame(pd.DataFrame({'f1': [1.0]}))
  with pytest.raises(ValueError, match='as_of'):
    _normalize_frame(pd.DataFrame({'entity_id': ['e1'], 'f1': [1.0]}))
  filled = _normalize_frame(
      pd.DataFrame({'entity_id': ['e1'], 'f1': [1.0]}),
      as_of='2024-01-01')
  assert filled['as_of'].iloc[0] is not None


def test_snapshot_timestamp_parsing():
  from forecast_ml_framework.feature_store.store import (
    _snapshot_timestamp, _strip_tz)
  assert _snapshot_timestamp('features/2024-01-03T09-00-00.pkl') is not None
  assert _snapshot_timestamp('features/2024-01-03.pkl') is not None
  assert _snapshot_timestamp('features/latest.pkl') is None
  assert _snapshot_timestamp('features/not-a-date.pkl') is None
  aware = _snapshot_timestamp('features/2024-01-03T09-00-00.pkl')
  assert _strip_tz(aware).tzinfo is None
  assert _strip_tz(None) is None


def test_consistency_dict_and_frame_paths_and_skips():
  from forecast_ml_framework.feature_store.store import check_consistency

  class DictStore:
    def __init__(self, data):
      self._data = data

    def get_features(self, entity_ids, feature_names, as_of=None):
      return {e: self._data[e] for e in entity_ids if e in self._data}

  class FrameStore:
    def __init__(self, frame):
      self._frame = frame

    def get_features(self, entity_ids, feature_names, as_of=None):
      return self._frame

  import pandas as pd
  frame = pd.DataFrame({'entity_id': ['e1'], 'f1': [2.0]})
  off = FrameStore(frame)
  on = DictStore({'e1': {'f1': 1.5}, 'e2': {'f1': 'bad'}})
  out = check_consistency(off, on, ['e1', 'e2'], ['f1'],
                          as_of='2024-01-01')
  assert out['n_compared'] == 1
  assert out['max_abs_diff'] == 0.5
  assert out['diffs'] == {'e1:f1': 0.5}
  # Non-numeric values are skipped, not crashed on.
  out2 = check_consistency(
      DictStore({'e1': {'f1': 'x'}}), DictStore({'e1': {'f1': 1.0}}),
      ['e1'], ['f1'])
  assert out2 == {'max_abs_diff': 0.0, 'n_compared': 0, 'diffs': {}}
  # Aliases point at the same function.
  from forecast_ml_framework.feature_store.store import (
    compare_offline_online, consistency_check)
  assert consistency_check is check_consistency
  assert compare_offline_online is check_consistency


def test_offline_init_from_path_and_none(tmp_path):
  from forecast_ml_framework.feature_store.offline import (
      OfflineFeatureStore)
  from forecast_ml_framework.core.artifact_store import LocalArtifactStore
  by_path = OfflineFeatureStore(str(tmp_path / 'p'))
  assert isinstance(by_path.artifact_store, LocalArtifactStore)
  by_none = OfflineFeatureStore(None, base_path=str(tmp_path / 'q'))
  assert isinstance(by_none.artifact_store, LocalArtifactStore)
  assert by_none.artifact_store.base_path == str(tmp_path / 'q')


def test_offline_all_keys_fallback_and_bad_frames(tmp_path):
  offline = _offline(tmp_path)
  key = offline.materialize(_df(), as_of='2024-01-03')
  assert offline._all_keys() == [key]
  # Store whose list(prefix) raises falls back to list('').
  class FlakyStore:
    def __init__(self, inner):
      self._inner = inner

    def save(self, path, data):
      return self._inner.save(path, data)

    def load(self, path):
      return self._inner.load(path)

    def exists(self, path):
      return self._inner.exists(path)

    def list(self, prefix=''):
      if prefix:
        raise RuntimeError('prefix unsupported')
      return self._inner.list(prefix)

  from forecast_ml_framework.feature_store.offline import (
      OfflineFeatureStore)
  flaky = OfflineFeatureStore(FlakyStore(offline.artifact_store))
  assert flaky._all_keys() == [key]
  assert flaky._load_frame('missing.pkl') is None
  offline.artifact_store.save('features/bad.pkl', 'not-a-frame')
  assert offline._load_frame('features/bad.pkl') is None
  assert offline._combined_frame() is not None


def test_offline_load_future_only_raises(tmp_path):
  offline = _offline(tmp_path)
  offline.materialize(_df(), as_of='2024-01-03')
  with pytest.raises(FileNotFoundError, match='No offline feature snapshot'):
    offline.load(as_of='2024-01-01T00:00:00')
  with pytest.raises(FileNotFoundError):
    offline.load(as_of='2099-01-01', key='missing.pkl')
  # Date-only as_of picks the snapshot on that day.
  frame = offline.load(as_of='2024-01-03')
  assert not frame.empty
  # Key load roundtrip.
  key = offline.materialize(_df(), as_of='2024-01-02')
  assert not offline.load(key=key).empty


def test_offline_get_features_empty_and_filtered(tmp_path):
  import pandas as pd
  offline = _offline(tmp_path)
  empty_out = offline.get_features(['e1'], as_of='2024-01-01')
  assert empty_out.empty
  with pytest.raises(ValueError, match='Unknown features'):
    offline.get_features(['e1'], ['f1'], as_of='2024-01-01')
  offline.materialize(_df(), as_of='2024-01-03')
  before = offline.get_features(['e1'], ['f1'], as_of='2023-12-31')
  assert before.empty
  missing_entity = offline.get_features(
      ['ghost', 'e1'], ['f1'], as_of='2024-01-03')
  assert list(missing_entity['entity_id']) == ['e1']
  filtered = offline.materialize(
      _df(), as_of='2024-01-04', feature_names=['f1', 'absent'])
  assert filtered
  out = offline.get_features(['e1'], as_of='2024-01-04')
  assert list(out['f1']) == [3.0]  # latest snapshot wins per entity


def test_online_redis_lazy_fallbacks(monkeypatch):
  import sys
  from forecast_ml_framework.feature_store.online import (
      OnlineFeatureStore)
  monkeypatch.setitem(sys.modules, 'redis', None)
  store = OnlineFeatureStore(redis_url='redis://localhost:6379/0')
  assert store._client() is None
  store.set_features('e1', {'f1': 1.0}, as_of='2024-01-01')
  assert store.get_features(['e1'], as_of='2024-01-02') == {
      'e1': {'f1': 1.0}}
  with pytest.raises(ValueError, match='requires as_of'):
    store.set_features('e1', {'f1': 1.0})


def test_online_redis_from_url_exception_falls_back(monkeypatch):
  import sys
  import types
  redis_mod = types.ModuleType('redis')

  class BrokenRedis:
    @classmethod
    def from_url(cls, url):
      raise RuntimeError('unreachable')

  redis_mod.Redis = BrokenRedis
  monkeypatch.setitem(sys.modules, 'redis', redis_mod)
  from forecast_ml_framework.feature_store.online import (
      OnlineFeatureStore)
  store = OnlineFeatureStore(redis_url='redis://bad:6379/0')
  assert store._client() is None


def test_online_redis_bad_payloads_swallowed():
  from forecast_ml_framework.feature_store.online import (
      OnlineFeatureStore)

  class BadRedis:
    def set(self, key, value):
      raise RuntimeError('set failed')

    def get(self, key):
      return b'not-json{{{'

  store = OnlineFeatureStore(redis_client=BadRedis())
  store.set_features('e1', {'f1': 2.0}, as_of='2024-01-01')
  # Bad Redis payload is ignored; in-memory record wins.
  assert store.get_features(['e1'], as_of='2024-01-02') == {
      'e1': {'f1': 2.0}}

  class GetRaisesRedis:
    def set(self, key, value):
      pass

    def get(self, key):
      raise RuntimeError('get failed')

  store2 = OnlineFeatureStore(redis_client=GetRaisesRedis())
  assert store2._redis_get('e1') is None


def test_online_materialize_with_feature_filter_and_df_view():
  from forecast_ml_framework.feature_store.online import (
      OnlineFeatureStore)
  online = OnlineFeatureStore()
  count = online.materialize(_df(), as_of='2024-01-03',
                             feature_names=['f1'])
  assert count == 3
  out = online.get_features(['e1'], as_of='2024-01-03')
  assert out == {'e1': {'f1': 3.0}}
  frame = online.get_features_df(['e1', 'e2'], ['f1'],
                                 as_of='2024-01-03')
  assert list(frame['entity_id']) == ['e1', 'e2']
  assert list(frame['f1']) == [3.0, 2.0]
  online.put_features('e3', {'f1': 9.0}, as_of='2024-01-03')
  online.set_feature('e4', {'f1': 8.0}, as_of='2024-01-03')
  assert online.get_features(['e3', 'e4'], as_of='2024-01-03') == {
      'e3': {'f1': 9.0}, 'e4': {'f1': 8.0}}


def test_feature_store_parquet_roundtrip(tmp_path):
  import pandas as pd
  parquet_path = tmp_path / 'features.parquet'
  frame = _df()
  frame.to_parquet(parquet_path, index=False)
  loaded = pd.read_parquet(parquet_path)
  assert list(loaded['entity_id']) == ['e1', 'e1', 'e2']
  assert loaded['f1'].tolist() == [1.0, 3.0, 2.0]
  offline = _offline(tmp_path)
  key = offline.materialize(loaded, as_of='2024-01-03')
  out = offline.get_features(['e1'], ['f1', 'f2'], as_of='2024-01-03')
  assert list(out['f1']) == [3.0]
  assert list(out['f2']) == [30.0]
  # Raw parquet bytes also roundtrip through the local artifact store.
  from forecast_ml_framework.core.artifact_store import LocalArtifactStore
  blob_store = LocalArtifactStore(str(tmp_path))
  blob_store.save('features.parquet', parquet_path.read_bytes())
  assert pd.read_parquet(
      __import__('io').BytesIO(
          blob_store.load('features.parquet'))).shape[0] == 3


def test_consistency_online_frame_and_offline_dict_paths():
  from forecast_ml_framework.feature_store.store import check_consistency
  import pandas as pd

  class DictStore:
    def get_features(self, entity_ids, feature_names, as_of=None):
      return {'e1': {'f1': 1.0}, 'e2': {'f1': 5.0}}

  class FrameStore:
    def get_features(self, entity_ids, feature_names, as_of=None):
      return pd.DataFrame({'entity_id': ['e1', 'e2'], 'f1': [1.0, 7.0]})

  out = check_consistency(
      DictStore(), FrameStore(), ['e1', 'e2'], ['f1'])
  assert out['n_compared'] == 2
  assert out['max_abs_diff'] == 2.0


def test_offline_tz_aware_pit(tmp_path):
  import pandas as pd
  offline = _offline(tmp_path)
  frame = pd.DataFrame({
      'entity_id': ['e1'],
      'as_of': pd.to_datetime(['2024-01-03']).tz_localize('UTC'),
      'f1': [1.0],
  })
  offline.materialize(frame, as_of='2024-01-03')
  out = offline.get_features(
      ['e1'], ['f1'], as_of='2024-01-03')
  assert list(out['f1']) == [1.0]
  out = offline.get_features(
      ['e1'], ['f1'],
      as_of=pd.Timestamp('2024-01-03', tz='UTC'))
  assert list(out['f1']) == [1.0]
  loaded = offline.load(
      as_of=pd.Timestamp('2024-01-03', tz='UTC'))
  assert list(loaded['f1']) == [1.0]


def test_online_tz_aware_pit():
  import pandas as pd
  from forecast_ml_framework.feature_store.online import (
      OnlineFeatureStore)
  online = OnlineFeatureStore()
  online.set_features(
      'e1', {'f1': 1.0},
      as_of=pd.Timestamp('2024-01-03', tz='UTC'))
  assert online.get_features(
      ['e1'], ['f1'], as_of='2024-01-03') == {'e1': {'f1': 1.0}}
  assert online.get_features(
      ['e1'], ['f1'],
      as_of=pd.Timestamp('2024-01-03', tz='UTC')) == {
          'e1': {'f1': 1.0}}


def test_online_materialize_empty_raises():
  import pandas as pd
  from forecast_ml_framework.feature_store.online import (
      OnlineFeatureStore)
  online = OnlineFeatureStore()
  with pytest.raises(ValueError, match='(?i)empty'):
    online.materialize(
        pd.DataFrame({'entity_id': [], 'as_of': [], 'f1': []}),
        as_of='2024-01-01')


def test_online_get_features_unknown_on_empty_raises():
  from forecast_ml_framework.feature_store.online import (
      OnlineFeatureStore)
  online = OnlineFeatureStore()
  with pytest.raises(ValueError, match='(?i)unknown'):
    online.get_features(['e1'], ['nope'], as_of='2024-01-03')
  with pytest.raises(ValueError, match='(?i)unknown'):
    online.get_features_df(['e1'], ['nope'], as_of='2024-01-03')


def test_offline_load_bad_date_is_file_not_found(tmp_path):
  offline = _offline(tmp_path)
  offline.materialize(_df(), as_of='2024-01-03')
  with pytest.raises(FileNotFoundError):
    offline.load(as_of='not-a-date')
  with pytest.raises(FileNotFoundError):
    offline.load(as_of='2024-13-45')


def test_offline_load_unparseable_key_via_text_match(tmp_path):
  offline = _offline(tmp_path)
  offline.artifact_store.save('features/custom.pkl', _df())
  frame = offline.load(as_of='custom')
  assert not frame.empty


def test_feature_helpers_canonical_in_store():
  '''Offline/online helpers are the store.py canonical copies.'''
  from forecast_ml_framework.feature_store import offline, online, store
  for name in ('_naive_utc', '_naive_utc_frame', '_check_not_empty'):
    assert getattr(offline, name) is getattr(store, name)
    assert getattr(online, name) is getattr(store, name)


def test_offline_combined_frame_pit_filter_and_row_cap(tmp_path):
  offline = _offline(tmp_path)
  offline.materialize(_df(), as_of='2024-01-01')
  offline.materialize(_df(), as_of='2024-01-03')
  # PIT: only rows at/before when enter the combined frame.
  pit = offline._combined_frame(when='2024-01-02')
  assert len(pit) == 4
  import pandas as pd
  assert (pit['as_of'] <= pd.Timestamp('2024-01-02')).all()
  assert len(offline._combined_frame()) == 6
  # Row cap via check_pandas_limits reuse.
  with pytest.raises(ValueError, match='max_rows_for_pandas'):
    offline._combined_frame(
        config={'max_rows_for_pandas': 2})
