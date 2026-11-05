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
