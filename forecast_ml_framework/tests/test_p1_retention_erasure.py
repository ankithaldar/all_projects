#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''P1-D2 tests: retention enforcement, GDPR erasure, consent.

TDD RED first: ``enforce_retention`` / ``list_snapshots`` /
``list_online_entities`` / ``delete_record`` / ``erase_subject`` /
``record_consent`` / ``check_consent`` / ``revoke_consent`` do
not exist yet, so the lazy imports inside each test fail for
the right reason (feature missing) until GREEN lands.
'''

import json
import logging
from datetime import datetime, timezone

import pytest

NOW = datetime(2026, 11, 8, tzinfo=timezone.utc)
CLEAR = 'SECRET-SUBJECT-9f3a-ENTITY'
ACTOR = 'SECRET-ACTOR-77bc'
OTHER = 'other-entity-1'
RETAINED = ('predictions/monitoring delete surface deferred '
            'per D-PRED-DELETE-1')


# --- fakes ---------------------------------------------------------------

class _FakeArtifactStore:
  '''Fake ArtifactStore recording deletes (offline key deletes).'''

  def __init__(self, keys=None):
    '''Store keys and record delete calls.'''
    self._data = {key: b'x' for key in (keys or [])}
    self.deleted = []
    self.fail_on = set()

  def save(self, path, data):
    '''Record a save.'''
    self._data[path] = data

  def load(self, path):
    '''Return stored bytes.'''
    return self._data[path]

  def exists(self, path):
    '''Return True when the key exists.'''
    return path in self._data

  def list(self, prefix=''):
    '''Return sorted keys under prefix.'''
    return sorted(k for k in self._data if k.startswith(prefix))

  def delete(self, path):
    '''Delete a key, raising for keys in fail_on.'''
    self.deleted.append(path)
    if path in self.fail_on:
      raise RuntimeError('artifact delete boom')
    self._data.pop(path, None)


class _FakeOfflineStore:
  '''Fake offline store: _all_keys + _load_frame + delete_entity.'''

  def __init__(self, frames):
    '''frames maps snapshot key to entity id list.'''
    self._frames = {key: list(ids) for key, ids in frames.items()}
    self.artifact_store = _FakeArtifactStore(list(frames))
    self.delete_calls = []

  def _all_keys(self):
    '''Return snapshot keys.'''
    return sorted(self._frames)

  def _load_frame(self, key):
    '''Return a frame for key or None.'''
    import pandas as pd
    if key not in self._frames:
      return None
    return pd.DataFrame({'entity_id': list(self._frames[key])})

  def delete_entity(self, entity_id):
    '''Remove entity rows, deleting emptied keys.'''
    self.delete_calls.append(entity_id)
    for key in list(self._frames):
      ids = self._frames[key]
      if entity_id not in ids:
        continue
      kept = [item for item in ids if item != entity_id]
      if kept:
        self._frames[key] = kept
      else:
        del self._frames[key]
        self.artifact_store.delete(key)

  def get_features(self, entity_ids, feature_names=None, as_of=None,
                   **kwargs):
    '''Return a frame of present entities (PIT ignored).'''
    import pandas as pd
    present = []
    for key in sorted(self._frames):
      for entity in self._frames[key]:
        if entity in entity_ids and entity not in present:
          present.append(entity)
    ordered = [entity for entity in entity_ids if entity in present]
    return pd.DataFrame({'entity_id': ordered})


class _FakeOnlineStore:
  '''Fake online store: _store + _records_for + delete.'''

  def __init__(self, records, redis_client=None, noop_delete=False):
    '''records maps entity to record dicts.'''
    self._store = {key: list(val) for key, val in records.items()}
    self._redis = redis_client
    self.noop_delete = noop_delete
    self.delete_calls = []
    self.fail_delete = set()

  def _client(self):
    '''Return the fake Redis client or None.'''
    return self._redis

  def _records_for(self, entity):
    '''Return sorted records for entity.'''
    records = sorted(self._store.get(entity, []),
                     key=lambda item: item['as_of'])
    return [dict(item) for item in records]

  def delete(self, entity_id):
    '''Pop entity, raising for entities in fail_delete.'''
    self.delete_calls.append(entity_id)
    if not self.noop_delete:
      self._store.pop(entity_id, None)
    if entity_id in self.fail_delete:
      raise RuntimeError('redis boom')

  def get_features(self, entity_ids, feature_names=None, as_of=None,
                   **kwargs):
    '''Return latest features per present entity.'''
    result = {}
    for entity in entity_ids:
      records = self._records_for(entity)
      if records:
        result[entity] = dict(records[-1].get('features', {}))
    return result


class _FakeRedis:
  '''Fake Redis with get/set/delete/keys.'''

  def __init__(self):
    '''Start empty.'''
    self.data = {}
    self.delete_calls = []

  def get(self, key):
    '''Return stored value or None.'''
    return self.data.get(key)

  def set(self, key, value):
    '''Store a value.'''
    self.data[key] = value

  def delete(self, key):
    '''Record and remove a key.'''
    self.delete_calls.append(key)
    self.data.pop(key, None)
    return 1

  def keys(self, pattern=''):
    '''Return stored keys.'''
    return list(self.data)


class _Record:
  '''Simple record factory for fake online stores.'''

  @staticmethod
  def make(as_of, value=1.0):
    '''Build one online record dict.'''
    return {'as_of': as_of, 'features': {'f1': value}}


def _registry_fixture():
  '''Fit a one-step registry and return it.'''
  import pandas as pd
  from forecast_ml_framework.pipelines.data_prep.fitted import (
    fit_preprocess)
  train = pd.DataFrame({'num_imp': [1.0, None, 3.0, 4.0],
                        'target': [0, 1, 0, 1]})
  params = [{'Imputations': {'args': {
    'strategy': 'mean',
    'cols': {'include_cols': ['num_imp']}}}}]
  variables = {'num_cols': ['num_imp'], 'cat_cols': [],
               'target_col': 'target'}
  _, registry = fit_preprocess(train, params, variables)
  return registry


def _envelope_fixture():
  '''Fit a tiny classifier and return its envelope.'''
  import numpy as np
  from sklearn.linear_model import LogisticRegression
  from forecast_ml_framework.pipelines.modeling.envelope import (
    wrap_envelope)
  x = np.array([[0.0], [1.0], [2.0], [3.0],
                [0.5], [2.5], [1.5], [3.5]])
  y = np.array([0, 0, 1, 1, 0, 1, 0, 1])
  est = LogisticRegression(max_iter=1000).fit(x, y)
  return wrap_envelope(est, ['0', '1'])


# --- retention: enumeration ---------------------------------------------

def test_list_snapshots_offline_fake():
  '''list_snapshots enumerates keys with ISO or None timestamps.'''
  from forecast_ml_framework.governance.retention import list_snapshots
  store = _FakeOfflineStore({
      'features/2020-01-01T00-00-00.pkl': ['a'],
      'features/2026-11-01T00-00-00.pkl': ['b'],
      'features/latest.pkl': ['c'],
  })
  records = list_snapshots(store)
  by_key = {item['key']: item for item in records}
  assert by_key['features/2020-01-01T00-00-00.pkl']['created_at'] \
      == '2020-01-01T00:00:00'
  assert by_key['features/2026-11-01T00-00-00.pkl']['created_at'] \
      == '2026-11-01T00:00:00'
  assert by_key['features/latest.pkl']['created_at'] is None


def test_list_snapshots_artifact_store_direct():
  '''list_snapshots also enumerates a bare ArtifactStore.'''
  from forecast_ml_framework.governance.retention import list_snapshots
  store = _FakeArtifactStore([
      'features/2020-01-01T00-00-00.pkl', 'features/latest.pkl'])
  records = list_snapshots(store)
  assert [item['key'] for item in records] == [
      'features/2020-01-01T00-00-00.pkl', 'features/latest.pkl']
  assert records[0]['created_at'] == '2020-01-01T00:00:00'
  assert records[1]['created_at'] is None


def test_list_online_entities_uses_max_as_of():
  '''list_online_entities keeps entities by max as_of, skips empty.'''
  from forecast_ml_framework.governance.retention import (
    list_online_entities)
  store = _FakeOnlineStore({
      'old': [_Record.make(datetime(2020, 1, 1))],
      'mixed': [_Record.make(datetime(2020, 1, 1)),
                _Record.make(datetime(2026, 11, 1))],
      'empty': [],
  })
  records = list_online_entities(store)
  by_entity = {item['entity_id']: item for item in records}
  assert 'empty' not in by_entity
  assert by_entity['old']['as_of'] == '2020-01-01T00:00:00'
  assert by_entity['mixed']['as_of'] == '2026-11-01T00:00:00'


def test_list_online_entities_redis_union():
  '''list_online_entities unions Redis-only entities.'''
  from forecast_ml_framework.governance.retention import (
    list_online_entities)
  redis = _FakeRedis()
  redis.data['feature:redis-only'] = json.dumps(
      [{'as_of': '2020-01-01 00:00:00', 'features': {'f1': 1.0}}])
  store = _FakeOnlineStore({}, redis_client=redis)

  class _UnionStore(_FakeOnlineStore):
    '''Fake store serving Redis records via _records_for.'''

    def _records_for(self, entity):
      '''Return Redis records for redis-only entity.'''
      if entity == 'redis-only':
        return [_Record.make(datetime(2020, 1, 1))]
      return super()._records_for(entity)

  records = list_online_entities(_UnionStore({}, redis_client=redis))
  assert [item['entity_id'] for item in records] == ['redis-only']
  assert records[0]['as_of'] == '2020-01-01T00:00:00'


# --- retention: typed delete dispatch ------------------------------------

def test_delete_record_typed_dispatch():
  '''delete_record routes keys and entities, rejects unknown.'''
  from forecast_ml_framework.governance.retention import delete_record
  offline = _FakeOfflineStore({
      'features/2020-01-01T00-00-00.pkl': ['a']})
  delete_record(offline, {'key': 'features/2020-01-01T00-00-00.pkl'})
  assert offline.artifact_store.deleted == [
      'features/2020-01-01T00-00-00.pkl']
  online = _FakeOnlineStore({'e1': [_Record.make(NOW)]})
  delete_record(online, {'entity_id': 'e1'})
  assert online.delete_calls == ['e1']
  bare = _FakeArtifactStore(['k.pkl'])
  delete_record(bare, {'key': 'k.pkl'})
  assert bare.deleted == ['k.pkl']
  with pytest.raises(ValueError):
    delete_record(online, {'bogus': 1})
  with pytest.raises(ValueError):
    delete_record(offline, {'entity_id': 'e1'})
  with pytest.raises(ValueError):
    delete_record(object(), {'key': 'k.pkl'})
  with pytest.raises(ValueError):
    delete_record(object(), {'entity_id': 'e1'})
  with pytest.raises(ValueError):
    delete_record(online, 'not-a-dict')


def test_enforce_retention_fake_offline():
  '''enforce deletes expired, keeps fresh and unknown-age.'''
  from forecast_ml_framework.governance.retention import enforce_retention
  from forecast_ml_framework.governance.scrub import subject_hash
  store = _FakeOfflineStore({
      'features/2020-01-01T00-00-00.pkl': ['a'],
      'features/2026-11-01T00-00-00.pkl': ['b'],
      'features/latest.pkl': ['c'],
  })
  log = []
  result = enforce_retention(
      store, {'retention_days': 30}, ACTOR, now=NOW, log=log)
  assert store.artifact_store.deleted == [
      'features/2020-01-01T00-00-00.pkl']
  assert [item['key'] for item in result['keep']] == [
      'features/2026-11-01T00-00-00.pkl', 'features/latest.pkl']
  assert [item['key'] for item in result['delete']] == [
      'features/2020-01-01T00-00-00.pkl']
  assert result['audit_entry']['details']['swept'] == 1
  assert result['audit_entry']['details']['kept'] == 2
  assert result['audit_entry']['actor'] == subject_hash(ACTOR)
  assert log == [result['audit_entry']]
  assert ACTOR not in str(result)
  assert ACTOR not in str(log)


def test_enforce_retention_fake_online():
  '''enforce deletes expired online entities, keeps fresh.'''
  from forecast_ml_framework.governance.retention import enforce_retention
  store = _FakeOnlineStore({
      'old': [_Record.make(datetime(2020, 1, 1))],
      'fresh': [_Record.make(datetime(2026, 11, 1))],
  })
  result = enforce_retention(
      store, {'retention_days': 30}, ACTOR, now=NOW, log=[])
  assert store.delete_calls == ['old']
  assert [item['entity_id'] for item in result['keep']] == ['fresh']
  assert [item['entity_id'] for item in result['delete']] == ['old']


def test_enforce_retention_abort_on_error_audit_partial():
  '''Store error aborts the sweep and audits a partial sweep.'''
  from forecast_ml_framework.governance.retention import enforce_retention
  store = _FakeOnlineStore({
      'old-1': [_Record.make(datetime(2020, 1, 1))],
      'old-2': [_Record.make(datetime(2020, 1, 1))],
  })
  store.fail_delete = {'old-2'}
  log = []
  with pytest.raises(RuntimeError):
    enforce_retention(
        store, {'retention_days': 30}, ACTOR, now=NOW, log=log)
  assert store.delete_calls == ['old-1', 'old-2']
  assert len(log) == 1
  assert log[0]['details']['partial'] is True
  assert log[0]['details']['swept'] == 1
  assert ACTOR not in str(log)


def test_enforce_retention_unknown_store_and_blank_actor():
  '''Unknown stores and blank actors fail loud.'''
  from forecast_ml_framework.governance.retention import enforce_retention
  with pytest.raises(ValueError):
    enforce_retention(object(), {'retention_days': 30}, ACTOR, now=NOW)
  with pytest.raises(ValueError):
    enforce_retention(_FakeOnlineStore({}), {'retention_days': 30}, '')
  with pytest.raises(ValueError):
    enforce_retention(_FakeOnlineStore({}), {'retention_days': 30}, None)


def test_enforce_retention_real_stores_integration(tmp_path):
  '''Real Local+Offline+Online: expired gone, fresh kept.'''
  import pandas as pd
  from forecast_ml_framework.core.artifact_store import (
    LocalArtifactStore)
  from forecast_ml_framework.feature_store.offline import (
    OfflineFeatureStore)
  from forecast_ml_framework.feature_store.online import (
    OnlineFeatureStore)
  from forecast_ml_framework.governance.retention import enforce_retention
  offline = OfflineFeatureStore(LocalArtifactStore(str(tmp_path)))
  key_old = offline.materialize(
      pd.DataFrame({'entity_id': ['old-e'], 'f1': [1.0]}),
      as_of='2020-01-01', pii_classes=['none'])
  key_fresh = offline.materialize(
      pd.DataFrame({'entity_id': ['fresh-e'], 'f1': [2.0]}),
      as_of='2026-11-01', pii_classes=['none'])
  result = enforce_retention(
      offline, {'retention_days': 30}, ACTOR, now=NOW, log=[])
  assert not offline.artifact_store.exists(key_old)
  assert offline.artifact_store.exists(key_fresh)
  assert [item['key'] for item in result['delete']] == [key_old]

  online = OnlineFeatureStore()
  online.set_features('old-e', {'f1': 1.0}, as_of='2020-01-01',
                      pii_classes=['none'])
  online.set_features('fresh-e', {'f1': 2.0}, as_of='2026-11-01',
                      pii_classes=['none'])
  enforce_retention(
      online, {'retention_days': 30}, ACTOR, now=NOW, log=[])
  assert online.get_features(
      ['old-e'], as_of='2026-11-08', strict_pit=True) == {}
  assert online.get_features(
      ['fresh-e'], as_of='2026-11-08', strict_pit=True) == {
      'fresh-e': {'f1': 2.0}}


def test_enforce_retention_redis_error_propagates_reread():
  '''Redis delete error propagates; memory already purged.'''
  from forecast_ml_framework.feature_store.online import (
    OnlineFeatureStore)
  from forecast_ml_framework.governance.retention import enforce_retention

  class _RaisingRedis(_FakeRedis):
    '''Fake Redis whose delete raises.'''

    def delete(self, key):
      '''Record then raise.'''
      self.delete_calls.append(key)
      raise RuntimeError('redis boom')

  online = OnlineFeatureStore()
  online.set_features('old-e', {'f1': 1.0}, as_of='2020-01-01',
                      pii_classes=['none'])
  raising = _RaisingRedis()
  online._redis = raising
  log = []
  with pytest.raises(RuntimeError):
    enforce_retention(
        online, {'retention_days': 30}, ACTOR, now=NOW, log=log)
  assert raising.delete_calls == ['feature:old-e']
  assert online.get_features(
      ['old-e'], as_of='2026-11-08', strict_pit=True) == {}
  assert log[0]['details']['partial'] is True
  assert log[0]['details']['swept'] == 0


# --- retention: pure check_retention coverage ------------------------------

def test_check_retention_pure_param_variants():
  '''check_retention resolves params and clock variants.'''
  from forecast_ml_framework.governance.retention import check_retention
  old = {'created_at': '2020-01-01T00:00:00'}
  fresh = {'created_at': '2026-11-07T00:00:00'}
  assert check_retention([old, fresh], params=30, now=NOW)['delete'] \
      == [old]
  assert check_retention([old], params=30.0, now=NOW)['delete'] == [old]
  for key in ('retention_days', 'retention', 'ttl_days', 'max_age_days'):
    assert check_retention([old], params={key: 30}, now=NOW)['delete'] \
        == [old]
  assert check_retention(
      [old], params={'retention_days': 9999},
      retention_days=30, now=NOW)['delete'] == [old]
  with pytest.raises(ValueError):
    check_retention([old], params=None, now=NOW)
  with pytest.raises(ValueError):
    check_retention([old], params=[30], now=NOW)
  with pytest.raises(ValueError):
    check_retention([old], params={'other': 30}, now=NOW)
  with pytest.raises(ValueError):
    check_retention([old], params=-1, now=NOW)
  assert check_retention([old], params=30, clock=lambda: NOW)[
      'delete'] == [old]
  assert check_retention([old], params=30, clock=NOW)['delete'] == [old]
  assert check_retention(
      [old], params=30, now='2026-11-08T00:00:00+00:00')['delete'] \
      == [old]
  epoch = datetime(2026, 11, 8, tzinfo=timezone.utc).timestamp()
  assert check_retention([old], params=30, now=epoch)['delete'] == [old]
  assert check_retention([old], params=30)['delete'] == [old]


def test_check_retention_pure_age_variants():
  '''check_retention handles age fields, tz, and unknown ages.'''
  from forecast_ml_framework.governance.retention import check_retention
  assert check_retention(
      [{'age_days': 99}], params=30, now=NOW)['delete'] == [
      {'age_days': 99}]
  assert check_retention(
      [{'age': 1}], params=30, now=NOW)['keep'] == [{'age': 1}]
  assert check_retention(
      [{'created_at': 0}], params=30, now=NOW)['delete'] == [
      {'created_at': 0}]
  # Future-dated and unknown-age are conservatively kept.
  assert check_retention(
      [{'created_at': '2099-01-01T00:00:00'}], params=30, now=NOW)[
      'keep'] == [{'created_at': '2099-01-01T00:00:00'}]
  assert check_retention(
      [{'name': 'x'}], params=30, now=NOW)['keep'] == [{'name': 'x'}]
  assert check_retention(
      ['not-a-dict'], params=30, now=NOW)['keep'] == ['not-a-dict']
  assert check_retention(None, params=30, now=NOW) == {
      'keep': [], 'delete': []}
  # Naive reference with aware stamp, and aware with naive stamp.
  aware = {'created_at': '2020-01-01T00:00:00+00:00'}
  naive_now = datetime(2026, 11, 8)
  assert check_retention([aware], params=30, now=naive_now)['delete'] \
      == [aware]
  naive = {'created_at': '2020-01-01T00:00:00'}
  assert check_retention([naive], params=30, now=NOW)['delete'] == [naive]
  # Dict input returns names.
  out = check_retention(
      {'old': '2020-01-01T00:00:00', 'new': '2026-11-07T00:00:00'},
      params=30, now=NOW)
  assert out == {'keep': ['new'], 'delete': ['old']}


# --- erasure --------------------------------------------------------------

def test_erase_subject_fake_stores_and_markers(caplog):
  '''erase_subject deletes rows and flags both artifacts (hashed).'''
  import pandas as pd
  from forecast_ml_framework.governance.erasure import erase_subject
  from forecast_ml_framework.governance.scrub import subject_hash
  from forecast_ml_framework.pipelines.data_prep.fitted import (
    apply_preprocess, read_registry)
  from forecast_ml_framework.pipelines.modeling.envelope import (
    validate_envelope)
  offline = _FakeOfflineStore({
      'features/2024-01-01T00-00-00.pkl': [CLEAR, OTHER],
      'features/2024-01-02T00-00-00.pkl': [OTHER],
  })
  online = _FakeOnlineStore({
      CLEAR: [_Record.make(datetime(2026, 11, 1))],
      OTHER: [_Record.make(datetime(2026, 11, 1))],
  })
  registry = _registry_fixture()
  envelope = _envelope_fixture()
  steps_before = read_registry(registry)
  log = []
  with caplog.at_level(logging.WARNING):
    report = erase_subject(
        CLEAR, stores=[offline, online],
        artifacts={'fitted_registry': registry, 'best_model': envelope},
        actor=ACTOR, log=log, at='2026-11-08T00:00:00+00:00')
  digest = subject_hash(CLEAR)
  assert report['subject_hash'] == digest
  assert report['removed'] == [f'offline:{digest}',
                               f'online:{digest}']
  assert report['retained_legal_hold'] == [RETAINED]
  assert report['audit_id'] == report['audit_entry']['timestamp']
  assert offline.delete_calls == [CLEAR]
  assert online.delete_calls == [CLEAR]
  assert offline.get_features([CLEAR]).empty
  assert online.get_features([CLEAR]) == {}
  assert not offline.get_features([OTHER]).empty
  assert online.get_features([OTHER]) != {}
  marker = registry['__meta__']['erasure']
  assert marker == {'subject_hash': digest,
                    'requested_at': '2026-11-08T00:00:00+00:00',
                    'action': 'retrain_required'}
  assert envelope['erasure'] == marker
  assert read_registry(registry) == steps_before
  validate_envelope(envelope, 'classification')
  smoked = apply_preprocess(
      pd.DataFrame({'num_imp': [None, 5.0]}), registry)
  assert smoked['num_imp'].notna().all()
  assert CLEAR not in str(report)
  assert ACTOR not in str(report)
  assert CLEAR not in str(log)
  assert ACTOR not in str(log)
  assert CLEAR not in caplog.text
  assert log and log[0]['actor'] == subject_hash(ACTOR)


def test_erase_subject_not_found_returns_empty_report():
  '''Not-found subject returns empty removed plus an audit entry.'''
  from forecast_ml_framework.governance.erasure import erase_subject
  online = _FakeOnlineStore({OTHER: [_Record.make(NOW)]})
  log = []
  report = erase_subject(
      CLEAR, stores=[online], artifacts={}, actor=ACTOR, log=log)
  assert report['removed'] == []
  assert report['retained_legal_hold'] == [RETAINED]
  assert log and log[0]['event'] == 'erasure'
  assert CLEAR not in str(report)


def test_erase_subject_two_subject_overwrite_audit_keeps_both():
  '''Second erasure overwrites marker; audit preserves both hashes.'''
  from forecast_ml_framework.governance.erasure import erase_subject
  from forecast_ml_framework.governance.scrub import subject_hash
  registry = _registry_fixture()
  envelope = _envelope_fixture()
  artifacts = {'fitted_registry': registry, 'best_model': envelope}
  log = []
  erase_subject('subject-AAAA', stores=[], artifacts=artifacts,
                actor=ACTOR, log=log, at='2026-11-08T00:00:00+00:00')
  erase_subject('subject-BBBB', stores=[], artifacts=artifacts,
                actor=ACTOR, log=log, at='2026-11-08T01:00:00+00:00')
  assert registry['__meta__']['erasure']['subject_hash'] == \
      subject_hash('subject-BBBB')
  assert envelope['erasure']['subject_hash'] == \
      subject_hash('subject-BBBB')
  blob = str(log)
  assert subject_hash('subject-AAAA') in blob
  assert subject_hash('subject-BBBB') in blob
  assert 'subject-AAAA' not in blob
  assert 'subject-BBBB' not in blob


def test_erase_subject_persistence_roundtrip(tmp_path):
  '''Flagged artifacts round-trip via JSON/Pickle datasets.'''
  from forecast_ml_framework.core.datasets import (
    RestrictedPickleDataset)
  from forecast_ml_framework.governance.erasure import erase_subject
  from forecast_ml_framework.pipelines.data_prep.fitted import (
    read_registry)
  from forecast_ml_framework.pipelines.modeling.envelope import (
    validate_envelope)
  from kedro_datasets.json import JSONDataset
  registry = _registry_fixture()
  envelope = _envelope_fixture()
  erase_subject('subject-RT', stores=[],
                artifacts={'fitted_registry': registry,
                           'best_model': envelope},
                actor=ACTOR, at='2026-11-08T02:00:00+00:00')
  reg_path = tmp_path / 'fitted_registry.json'
  reg_ds = JSONDataset(filepath=str(reg_path))
  reg_ds.save(registry)
  loaded_reg = reg_ds.load()
  assert loaded_reg['__meta__']['erasure']['action'] == \
      'retrain_required'
  assert 'Imputations' in read_registry(loaded_reg)
  env_path = tmp_path / 'best_model.pkl'
  env_ds = RestrictedPickleDataset(filepath=str(env_path))
  env_ds.save(envelope)
  loaded_env = env_ds.load()
  validate_envelope(loaded_env, 'classification')
  assert loaded_env['erasure']['requested_at'] == \
      '2026-11-08T02:00:00+00:00'


def test_erase_subject_real_stores(tmp_path):
  '''erase_subject consumes real Offline and Online deletes.'''
  import pandas as pd
  from forecast_ml_framework.core.artifact_store import (
    LocalArtifactStore)
  from forecast_ml_framework.feature_store.offline import (
    OfflineFeatureStore)
  from forecast_ml_framework.feature_store.online import (
    OnlineFeatureStore)
  from forecast_ml_framework.governance.erasure import erase_subject
  offline = OfflineFeatureStore(LocalArtifactStore(str(tmp_path)))
  offline.materialize(
      pd.DataFrame({'entity_id': [CLEAR, OTHER], 'f1': [1.0, 2.0]}),
      as_of='2026-11-01', pii_classes=['none'])
  online = OnlineFeatureStore()
  online.set_features(CLEAR, {'f1': 1.0}, as_of='2026-11-01',
                      pii_classes=['none'])
  online.set_features(OTHER, {'f1': 2.0}, as_of='2026-11-01',
                      pii_classes=['none'])
  report = erase_subject(
      CLEAR, stores=[offline, online], artifacts={}, actor=ACTOR)
  assert len(report['removed']) == 2
  assert offline.get_features(
      [CLEAR], as_of='2026-11-08', strict_pit=True).empty
  assert online.get_features(
      [CLEAR], as_of='2026-11-08', strict_pit=True) == {}
  assert not offline.get_features(
      [OTHER], as_of='2026-11-08', strict_pit=True).empty
  assert online.get_features(
      [OTHER], as_of='2026-11-08', strict_pit=True) != {}
  assert CLEAR not in str(report)


def test_erase_subject_redis_failure_propagates_reread():
  '''Redis delete error propagates from erase_subject.'''
  from forecast_ml_framework.feature_store.online import (
    OnlineFeatureStore)
  from forecast_ml_framework.governance.erasure import erase_subject

  class _RaisingRedis(_FakeRedis):
    '''Fake Redis whose delete raises.'''

    def delete(self, key):
      '''Record then raise.'''
      self.delete_calls.append(key)
      raise RuntimeError('redis boom')

  online = OnlineFeatureStore()
  online.set_features(CLEAR, {'f1': 1.0}, as_of='2026-11-01',
                      pii_classes=['none'])
  raising = _RaisingRedis()
  online._redis = raising
  with pytest.raises(RuntimeError):
    erase_subject(CLEAR, stores=[online], artifacts={}, actor=ACTOR)
  assert raising.delete_calls == [f'feature:{CLEAR}']
  assert online.get_features(
      [CLEAR], as_of='2026-11-08', strict_pit=True) == {}


def test_erase_subject_verification_failure_raises():
  '''A store that keeps the subject fails erasure verification.'''
  from forecast_ml_framework.governance.erasure import erase_subject
  online = _FakeOnlineStore(
      {CLEAR: [_Record.make(NOW)]}, noop_delete=True)
  with pytest.raises(RuntimeError) as excinfo:
    erase_subject(CLEAR, stores=[online], artifacts={}, actor=ACTOR)
  assert CLEAR not in str(excinfo.value)
  assert ACTOR not in str(excinfo.value)


def test_erase_subject_artifact_edge_cases():
  '''Artifact edge cases: no meta, bad meta, bad envelope, None.'''
  from forecast_ml_framework.governance.erasure import erase_subject
  registry = {'version': 2, 'steps': {}}
  report = erase_subject('subject-X', stores=[], artifacts={
      'fitted_registry': registry, 'best_model': None},
      actor=ACTOR, at='2026-11-08T00:00:00+00:00')
  assert registry['__meta__']['erasure']['action'] == \
      'retrain_required'
  assert report['removed'] == []
  registry2 = {'__meta__': 'not-a-dict'}
  with pytest.raises(ValueError):
    erase_subject('subject-X', stores=[], artifacts={
        'fitted_registry': registry2}, actor=ACTOR)
  with pytest.raises(ValueError):
    erase_subject('subject-X', stores=[], artifacts={
        'best_model': 'not-a-dict'}, actor=ACTOR)
  with pytest.raises(ValueError):
    erase_subject('subject-X', stores=[], artifacts={
        'fitted_registry': 'not-a-dict'}, actor=ACTOR)
  with pytest.raises(ValueError):
    erase_subject('subject-X', stores=[], artifacts='not-a-dict',
                  actor=ACTOR)
  with pytest.raises(ValueError):
    erase_subject(CLEAR, stores=[object()], artifacts={}, actor=ACTOR)
  with pytest.raises(ValueError):
    erase_subject('', stores=[], artifacts={}, actor=ACTOR)
  with pytest.raises(ValueError):
    erase_subject(CLEAR, stores=[], artifacts={}, actor='')
  # artifacts/stores may be None (treated as empty).
  report = erase_subject('subject-X', stores=None, artifacts=None,
                         actor=ACTOR)
  assert report['removed'] == []
  # Datetime and clock requested_at forms.
  report = erase_subject('subject-X', stores=[], artifacts={
      'fitted_registry': registry}, actor=ACTOR, now=NOW)
  assert registry['__meta__']['erasure']['requested_at'] == \
      NOW.isoformat()
  assert '2026' in report['audit_id']


def test_erase_subject_no_predictions_delete_surface():
  '''No predictions/monitoring delete surface exists in P1.'''
  from pathlib import Path
  root = Path(__file__).resolve().parents[1]
  text = (root / 'src' / 'forecast_ml_framework' / 'governance'
          / 'erasure.py').read_text(encoding='utf-8')
  assert 'def delete_predictions' not in text
  assert 'def delete_monitoring' not in text
  assert 'D-PRED-DELETE-1' in text


# --- consent ---------------------------------------------------------------

def test_record_and_check_consent_hash_only():
  '''record_consent stores hash-only entries; check honors grant.'''
  from forecast_ml_framework.governance.consent import (
    check_consent, record_consent)
  from forecast_ml_framework.governance.scrub import subject_hash
  ledger = {}
  log = []
  entry = record_consent(
      ledger, CLEAR, granted=True, actor=ACTOR,
      at='2026-11-08T00:00:00+00:00', log=log)
  digest = subject_hash(CLEAR)
  assert entry == {'subject_hash': digest, 'granted': True,
                   'actor_hash': subject_hash(ACTOR),
                   'at': '2026-11-08T00:00:00+00:00'}
  assert list(ledger) == [digest]
  blob = json.dumps(ledger)
  assert CLEAR not in blob
  assert ACTOR not in blob
  assert check_consent(ledger, CLEAR) is True
  assert check_consent(ledger, OTHER) is False
  assert log and log[0]['actor'] == subject_hash(ACTOR)
  assert CLEAR not in str(log)
  # Granted False records are not consent.
  record_consent(ledger, OTHER, granted=False, actor=ACTOR)
  assert check_consent(ledger, OTHER) is False


def test_revoke_consent_tombstone_and_erase_followup():
  '''revoke_consent tombstones and requires erase_subject next.'''
  from forecast_ml_framework.governance.consent import (
    check_consent, record_consent, revoke_consent)
  from forecast_ml_framework.governance.scrub import subject_hash
  ledger = {}
  record_consent(ledger, CLEAR, granted=True, actor=ACTOR)
  assert check_consent(ledger, CLEAR) is True
  result = revoke_consent(
      ledger, CLEAR, actor=ACTOR, at='2026-11-08T01:00:00+00:00')
  assert result == {'subject_hash': subject_hash(CLEAR)}
  assert check_consent(ledger, CLEAR) is False
  # Tombstone: the hash record remains, never deleted.
  assert subject_hash(CLEAR) in ledger
  assert ledger[subject_hash(CLEAR)]['granted'] is False
  assert CLEAR not in json.dumps(ledger)
  # Revoking an unknown subject still creates a tombstone.
  result = revoke_consent(ledger, OTHER, actor=ACTOR)
  assert result == {'subject_hash': subject_hash(OTHER)}
  assert check_consent(ledger, OTHER) is False
  assert 'erase_subject' in revoke_consent.__doc__


def test_consent_invalid_inputs_and_defaults():
  '''Consent helpers fail closed on invalid ledgers/subjects.'''
  from forecast_ml_framework.governance.consent import (
    check_consent, record_consent, revoke_consent)
  from forecast_ml_framework.governance.scrub import subject_hash
  assert check_consent(None, CLEAR) is False
  assert check_consent({}, CLEAR) is False
  assert check_consent({subject_hash(CLEAR): 'bogus'}, CLEAR) is False
  assert check_consent({}, None) is False
  with pytest.raises(ValueError):
    record_consent(None, CLEAR, granted=True, actor=ACTOR)
  with pytest.raises(ValueError):
    record_consent({}, '', granted=True, actor=ACTOR)
  with pytest.raises(ValueError):
    record_consent({}, CLEAR, granted=True, actor='')
  with pytest.raises(ValueError):
    revoke_consent([], CLEAR, actor=ACTOR)
  with pytest.raises(ValueError):
    revoke_consent({}, None, actor=ACTOR)
  ledger = {}
  record_consent(ledger, CLEAR, granted=True, actor=ACTOR)
  assert ledger[subject_hash(CLEAR)]['at']  # default timestamp present


def test_consent_write_guard_integration():
  '''D1 write guard refuses writes without granted consent.'''
  from forecast_ml_framework.feature_store.online import (
    OnlineFeatureStore)
  from forecast_ml_framework.governance.consent import (
    record_consent, revoke_consent)
  ledger = {}
  online = OnlineFeatureStore()
  with pytest.raises(ValueError):
    online.set_features(CLEAR, {'f1': 1.0}, as_of='2026-11-01',
                        pii_classes=['none'], consent_ledger=ledger)
  record_consent(ledger, CLEAR, granted=True, actor=ACTOR)
  online.set_features(CLEAR, {'f1': 1.0}, as_of='2026-11-01',
                      pii_classes=['none'], consent_ledger=ledger)
  assert online.get_features(
      [CLEAR], as_of='2026-11-01', strict_pit=True) == {
      CLEAR: {'f1': 1.0}}
  revoke_consent(ledger, CLEAR, actor=ACTOR)
  with pytest.raises(ValueError):
    online.set_features(CLEAR, {'f1': 2.0}, as_of='2026-11-02',
                        pii_classes=['none'], consent_ledger=ledger)


def test_check_retention_naive_datetime_stamp_aligns_utc():
  '''Aware reference plus naive datetime stamp aligns to UTC.'''
  from forecast_ml_framework.governance.retention import check_retention
  naive = {'created_at': datetime(2020, 1, 1)}
  assert check_retention([naive], params=30, now=NOW)['delete'] == [
      naive]
  fresh = {'created_at': datetime(2026, 11, 7)}
  assert check_retention([fresh], params=30, now=NOW)['keep'] == [
      fresh]


def test_list_snapshots_unknown_store_raises():
  '''list_snapshots rejects stores without enumeration.'''
  from forecast_ml_framework.governance.retention import list_snapshots
  with pytest.raises(ValueError):
    list_snapshots(object())


def test_list_online_entities_unknown_store_raises():
  '''list_online_entities rejects stores without _records_for.'''
  from forecast_ml_framework.governance.retention import (
    list_online_entities)
  with pytest.raises(ValueError):
    list_online_entities(object())


def test_list_online_entities_string_as_of_and_no_redis_keys():
  '''String as_of stamps and key-less Redis clients enumerate.'''
  from forecast_ml_framework.governance.retention import (
    list_online_entities)
  store = _FakeOnlineStore(
      {'e1': [{'as_of': '2020-01-01', 'features': {'f1': 1.0}}]},
      redis_client=object())
  records = list_online_entities(store)
  assert records[0]['entity_id'] == 'e1'
  assert records[0]['as_of'] == '2020-01-01'


def test_list_online_entities_redis_keys_decoding_and_errors():
  '''Redis key decoding skips bad keys; client errors raise.'''
  from forecast_ml_framework.governance.retention import (
    list_online_entities)

  class _OddRedis:
    '''Fake Redis returning bytes/bad keys.'''

    def keys(self, pattern=''):
      '''Return mixed key types.'''
      return [b'feature:bytes-e', 'feature:str-e', 'unrelated',
              42, 'feature:']

  class _RaisingKeysRedis:
    '''Fake Redis whose keys call raises.'''

    def keys(self, pattern=''):
      '''Raise like a dead Redis.'''
      raise RuntimeError('redis down')

  class _UnionStore(_FakeOnlineStore):
    '''Fake store serving Redis entities via _records_for.'''

    def _records_for(self, entity):
      '''Return a record for every enumerated entity.'''
      if entity in ('bytes-e', 'str-e'):
        return [_Record.make(datetime(2026, 11, 1))]
      return super()._records_for(entity)

  records = list_online_entities(
      _UnionStore({}, redis_client=_OddRedis()))
  assert sorted(item['entity_id'] for item in records) == [
      'bytes-e', 'str-e']
  # Enumeration errors propagate (fail-closed): a dead
  # Redis must never be reported as an empty sweep.
  with pytest.raises(RuntimeError):
    list_online_entities(
        _UnionStore({'mem': [_Record.make(datetime(2026, 11, 1))]},
                    redis_client=_RaisingKeysRedis()))

  class _RaisingClientStore(_FakeOnlineStore):
    '''Fake store whose _client raises.'''

    def _client(self):
      '''Raise like a broken client factory.'''
      raise RuntimeError('no client')

  with pytest.raises(RuntimeError):
    list_online_entities(
        _RaisingClientStore({'mem': [_Record.make(NOW)]}))


def test_enforce_retention_artifact_store_wrapper():
  '''A store exposing only artifact_store counts as offline.'''
  from forecast_ml_framework.governance.retention import enforce_retention

  class _Wrapper:
    '''Store wrapper delegating list/delete to artifacts.'''

    def __init__(self):
      '''Wrap a fake artifact store.'''
      self.artifact_store = _FakeArtifactStore(
          ['features/2020-01-01T00-00-00.pkl',
           'features/2026-11-01T00-00-00.pkl'])

    def list(self, prefix=''):
      '''Delegate list to the artifact store.'''
      return self.artifact_store.list(prefix)

    def delete(self, path):
      '''Delegate delete to the artifact store.'''
      self.artifact_store.delete(path)

  wrapper = _Wrapper()
  result = enforce_retention(
      wrapper, {'retention_days': 30}, ACTOR, now=NOW, log=[])
  assert wrapper.artifact_store.deleted == [
      'features/2020-01-01T00-00-00.pkl']
  assert [item['key'] for item in result['keep']] == [
      'features/2026-11-01T00-00-00.pkl']


def test_erase_subject_store_without_read_surface():
  '''Stores without get_features erase with empty removed.'''
  from forecast_ml_framework.governance.erasure import erase_subject

  class _NoReadStore:
    '''Store with delete only (no get_features).'''

    def __init__(self):
      '''Record deletes.'''
      self.delete_calls = []

    def delete(self, entity_id):
      '''Record the delete.'''
      self.delete_calls.append(entity_id)

  store = _NoReadStore()
  report = erase_subject(CLEAR, stores=[store], artifacts={},
                         actor=ACTOR)
  assert store.delete_calls == [CLEAR]
  assert report['removed'] == []


def test_erase_subject_list_and_bare_result_reads():
  '''Unusual get_features results drive presence checks.'''
  from forecast_ml_framework.governance.erasure import erase_subject

  class _BareResult:
    '''Result with truthiness but no len/empty.'''

    def __init__(self, present):
      '''Store the present flag.'''
      self.present = present

    def __bool__(self):
      '''Return the present flag.'''
      return self.present

  class _ListStore:
    '''Store returning list results from get_features.'''

    def __init__(self):
      '''Start present.'''
      self.present = True

    def get_features(self, entity_ids, as_of=None, **kwargs):
      '''Return a list of present entities.'''
      return list(entity_ids) if self.present else []

    def delete(self, entity_id):
      '''Become absent.'''
      self.present = False

  report = erase_subject(CLEAR, stores=[_ListStore()], artifacts={},
                         actor=ACTOR)
  assert len(report['removed']) == 1

  class _BareStore:
    '''Store returning bare results from get_features.'''

    def __init__(self):
      '''Start present.'''
      self.present = True

    def get_features(self, entity_ids, as_of=None, **kwargs):
      '''Return a bare truthy/falsy result.'''
      return _BareResult(self.present)

    def delete(self, entity_id):
      '''Become absent.'''
      self.present = False

  report = erase_subject(CLEAR, stores=[_BareStore()], artifacts={},
                         actor=ACTOR)
  assert len(report['removed']) == 1


def test_erase_subject_offline_verification_failure_raises():
  '''Offline store keeping the subject fails verification.'''
  from forecast_ml_framework.governance.erasure import erase_subject

  class _StickyOffline(_FakeOfflineStore):
    '''Offline store whose delete_entity is a no-op.'''

    def delete_entity(self, entity_id):
      '''Record the call but keep every row.'''
      self.delete_calls.append(entity_id)

  store = _StickyOffline({
      'features/2024-01-01T00-00-00.pkl': [CLEAR]})
  with pytest.raises(RuntimeError) as excinfo:
    erase_subject(CLEAR, stores=[store], artifacts={}, actor=ACTOR)
  assert CLEAR not in str(excinfo.value)


def test_new_modules_sole_leaf_guards():
  '''New modules reuse the scrub leaf and invent no second hasher.'''
  from pathlib import Path
  root = Path(__file__).resolve().parents[1]
  base = root / 'src' / 'forecast_ml_framework'
  for rel in ('governance/retention.py', 'governance/erasure.py',
              'governance/consent.py'):
    text = (base / rel).read_text(encoding='utf-8')
    for banned in ('re.sub', 're.compile', 'hashlib.sha256',
                   'import hashlib', 'def subject_hash', 'def scrub_',
                   'pickle.load(', 'pickle.loads('):
      assert banned not in text, f'{banned} in {rel}'
    assert 'subject_hash' in text
    assert text.startswith('#!/usr/bin/env python')
