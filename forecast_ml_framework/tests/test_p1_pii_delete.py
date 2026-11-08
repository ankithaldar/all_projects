#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''P1-D1 tests: scrub subject_hash leaf, artifact delete surface,
calibrated allowlist, online/offline delete, PII detect/mask/guard.

TDD RED->GREEN: this file is written before pii.py / delete
methods / subject_hash exist, so collection of the new imports
fails with ImportError until GREEN lands. Framework imports are
lazy (inside test functions) so each test fails for the right
reason (feature missing) rather than at collection.
'''

import hashlib
import logging
import pickle

import pytest


# --- scrub subject_hash sole leaf -------------------------------------

def test_subject_hash_matches_sha256():
  '''subject_hash equals sha256 hexdigest.'''
  from forecast_ml_framework.governance.scrub import subject_hash
  assert subject_hash('x') == hashlib.sha256(b'x').hexdigest()
  assert subject_hash('alice') == hashlib.sha256(
      'alice'.encode('utf-8')).hexdigest()


def test_subject_hash_is_64_hex_and_deterministic():
  '''subject_hash is lowercase 64-hex and deterministic.'''
  from forecast_ml_framework.governance.scrub import subject_hash
  digest = subject_hash('subject-1')
  assert len(digest) == 64
  assert digest == digest.lower()
  int(digest, 16)  # parses as hex (raises if not)
  assert subject_hash('subject-1') == digest
  assert subject_hash('subject-2') != digest
  assert subject_hash('') == hashlib.sha256(b'').hexdigest()


def test_subject_hash_handles_non_str_via_str():
  '''subject_hash stringifies its input.'''
  from forecast_ml_framework.governance.scrub import subject_hash
  assert subject_hash('123') == subject_hash(123)


# --- PII vocabulary + validation ---------------------------------------

def test_pii_classes_exact():
  '''PII_CLASSES is the frozen 7-tuple.'''
  from forecast_ml_framework.governance.pii import PII_CLASSES
  assert PII_CLASSES == (
      'email', 'user_id', 'device_id', 'location', 'financial',
      'health', 'none')
  assert isinstance(PII_CLASSES, tuple)


def test_require_pii_declared_none_raises():
  '''require_pii_declared(None) fails closed.'''
  from forecast_ml_framework.governance.pii import require_pii_declared
  with pytest.raises(ValueError):
    require_pii_declared(pii_classes=None)


def test_require_pii_declared_unknown_raises():
  '''require_pii_declared rejects unknown classes.'''
  from forecast_ml_framework.governance.pii import require_pii_declared
  with pytest.raises(ValueError):
    require_pii_declared(pii_classes=['email', 'bogus'])
  with pytest.raises(ValueError):
    require_pii_declared(pii_classes=['EMAIL'])


def test_require_pii_declared_empty_and_mixed_none_raise():
  '''Empty or none-mixed declarations fail closed.'''
  from forecast_ml_framework.governance.pii import require_pii_declared
  with pytest.raises(ValueError):
    require_pii_declared(pii_classes=[])
  with pytest.raises(ValueError):
    require_pii_declared(pii_classes=['none', 'email'])
  # Valid declarations pass.
  require_pii_declared(pii_classes=['none'])
  require_pii_declared(pii_classes=['email', 'user_id'])


def test_detect_pii_finds_email_user_device():
  '''detect_pii flags email/user/device values.'''
  import pandas as pd
  from forecast_ml_framework.governance.pii import detect_pii
  df = pd.DataFrame({
      'email': ['alice@example.com'],
      'user_id': ['user-123'],
      'device_id': ['device-abc'],
      'score': [1.0],
  })
  findings = detect_pii(df)
  by_class = {f.pii_class for f in findings}
  assert 'email' in by_class
  assert 'user_id' in by_class
  assert 'device_id' in by_class
  # Detail never carries a raw PII value.
  joined = ' '.join(f.detail for f in findings)
  assert 'alice@example.com' not in joined
  assert 'user-123' not in joined
  assert 'device-abc' not in joined


def test_detect_pii_column_names_flag_location_financial_health():
  '''detect_pii flags column-name PII classes.'''
  import pandas as pd
  from forecast_ml_framework.governance.pii import detect_pii
  df = pd.DataFrame({
      'location': ['Paris'],
      'financial': ['123-45-6789'],
      'health': ['ok'],
      'score': [1.0],
  })
  by_class = {f.pii_class for f in detect_pii(df)}
  assert 'location' in by_class
  assert 'financial' in by_class
  assert 'health' in by_class


def test_detect_pii_clean_frame_empty():
  '''detect_pii on a clean frame returns no findings.'''
  import pandas as pd
  from forecast_ml_framework.governance.pii import detect_pii
  df = pd.DataFrame({'entity_id': ['e1'], 'f1': [1.0], 'f2': [2.0]})
  assert detect_pii(df) == []


def test_mask_pii_deterministic():
  '''mask_pii deterministically redacts/hashes declared PII.'''
  import pandas as pd
  from forecast_ml_framework.governance.pii import mask_pii
  from forecast_ml_framework.governance.scrub import REDACTED
  df = pd.DataFrame({
      'email': ['alice@example.com', 'bob@example.com'],
      'user_id': ['user-123', 'user-456'],
      'device_id': ['device-abc', 'device-abc'],
      'score': [1.0, 2.0],
  })
  masked = mask_pii(
      df, pii_classes=['email', 'user_id', 'device_id'])
  blob = masked.astype(str).to_string()
  assert 'alice@example.com' not in blob
  assert 'bob@example.com' not in blob
  assert 'user-123' not in blob
  assert 'device-abc' not in blob
  assert list(masked['email']) == [REDACTED, REDACTED]
  assert REDACTED == '***REDACTED***'
  for cell in masked['user_id']:
    assert len(cell) == 64
    int(cell, 16)
  for cell in masked['device_id']:
    assert len(cell) == 64
    int(cell, 16)
  # Deterministic: same input -> same hashes; equality preserved.
  assert masked['device_id'][0] == masked['device_id'][1]
  assert masked['user_id'][0] != masked['user_id'][1]
  again = mask_pii(
      df, pii_classes=['email', 'user_id', 'device_id'])
  assert list(again['user_id']) == list(masked['user_id'])
  assert list(again['device_id']) == list(masked['device_id'])
  # Non-PII columns untouched.
  assert list(masked['score']) == [1.0, 2.0]


def test_mask_pii_location_financial_health_redacted():
  '''mask_pii redacts location/financial/health cells.'''
  import pandas as pd
  from forecast_ml_framework.governance.pii import mask_pii
  from forecast_ml_framework.governance.scrub import REDACTED
  df = pd.DataFrame({
      'location': ['Paris'],
      'financial': ['SECRET-FIN'],
      'health': ['SECRET-HEALTH'],
      'score': [1.0],
  })
  masked = mask_pii(
      df, pii_classes=['location', 'financial', 'health'])
  assert list(masked['location']) == [REDACTED]
  assert list(masked['financial']) == [REDACTED]
  assert list(masked['health']) == [REDACTED]
  blob = masked.astype(str).to_string()
  assert 'SECRET-FIN' not in blob
  assert 'SECRET-HEALTH' not in blob


def test_mask_pii_undeclared_raises_no_passthrough():
  '''mask_pii fails closed on undeclared PII.'''
  import pandas as pd
  from forecast_ml_framework.governance.pii import mask_pii
  df = pd.DataFrame({'email': ['alice@example.com']})
  with pytest.raises(ValueError):
    mask_pii(df, pii_classes=['none'])
  with pytest.raises(ValueError):
    mask_pii(df, pii_classes=['user_id'])
  with pytest.raises(ValueError):
    mask_pii(df, pii_classes=['bogus'])
  with pytest.raises(ValueError):
    mask_pii(df, pii_classes=None)


def test_guard_write_fail_closed():
  '''guard_write raises on undeclared PII, passes when declared.'''
  import pandas as pd
  from forecast_ml_framework.governance.consent import record_consent
  from forecast_ml_framework.governance.pii import guard_write
  clean = pd.DataFrame({'entity_id': ['e1'], 'f1': [1.0]})
  guard_write(clean, pii_classes=['none'])
  # Declared PII without a consent ledger fails closed.
  with pytest.raises(ValueError):
    guard_write(clean, pii_classes=['email'])
  ledger = {}
  record_consent(ledger, 'e1', granted=True, actor='op-1')
  guard_write(clean, pii_classes=['email'], consent_ledger=ledger)
  dirty = pd.DataFrame(
      {'entity_id': ['e1'], 'email': ['alice@example.com']})
  with pytest.raises(ValueError):
    guard_write(dirty, pii_classes=['none'])
  with pytest.raises(ValueError):
    guard_write(dirty, pii_classes=['user_id'])
  with pytest.raises(ValueError):
    guard_write(dirty, pii_classes=['bogus'])
  # Declared PII without a ledger fails closed.
  with pytest.raises(ValueError):
    guard_write(dirty, pii_classes=['email'])
  # Declared PII passes with a granted ledger.
  guard_write(dirty, pii_classes=['email'], consent_ledger=ledger)


def test_guard_write_consent_refuses_without_grant():
  '''guard_write refuses writes when consent is not granted.'''
  import pandas as pd
  from forecast_ml_framework.governance.pii import guard_write
  df = pd.DataFrame({'entity_id': ['e1'], 'f1': [1.0]})
  with pytest.raises(ValueError):
    guard_write(df, pii_classes=['none'], consent_ledger={})


# --- artifact_store delete surface --------------------------------------

def test_artifact_store_delete_is_abstract():
  '''ArtifactStore.delete is abstract.'''
  from forecast_ml_framework.core.artifact_store import ArtifactStore
  with pytest.raises(TypeError):
    ArtifactStore()


def test_local_delete_idempotent_and_jail(tmp_path):
  '''Local delete removes once, missing is a no-op, jail holds.'''
  from forecast_ml_framework.core.artifact_store import (
      LocalArtifactStore)
  store = LocalArtifactStore(str(tmp_path))
  store.save('a/b.pkl', b'data')
  assert store.exists('a/b.pkl')
  store.delete('a/b.pkl')
  assert not store.exists('a/b.pkl')
  # Missing key: no-op, never an error (retryable).
  store.delete('a/b.pkl')
  store.delete('never/existed.pkl')
  # Jail: absolute / escaping keys still raise.
  with pytest.raises(ValueError):
    store.delete('/etc/passwd')
  with pytest.raises(ValueError):
    store.delete('../escape.pkl')


def test_local_store_save_load_list_roundtrip(tmp_path):
  '''Local store save/load/list still work (delete regression).'''
  from forecast_ml_framework.core.artifact_store import (
      LocalArtifactStore)
  store = LocalArtifactStore(str(tmp_path))
  store.save('x.pkl', b'hello')
  assert store.load('x.pkl') == 'hello'
  store.save('y.txt', 'world')
  assert store.load('y.txt') == 'world'
  assert sorted(store.list('')) == ['x.pkl', 'y.txt']


def test_create_artifact_store_dispatches(tmp_path):
  '''create_artifact_store returns the right backend.'''
  from forecast_ml_framework.core.artifact_store import (
      GCSArtifactStore, LocalArtifactStore, S3ArtifactStore,
      create_artifact_store)
  assert isinstance(
      create_artifact_store({'type': 'local', 'base_path': str(tmp_path)}),
      LocalArtifactStore)
  assert isinstance(
      create_artifact_store({'type': 'gcs', 'bucket': 'b'}),
      GCSArtifactStore)
  assert isinstance(
      create_artifact_store({'type': 's3', 'bucket': 'b'}),
      S3ArtifactStore)
  with pytest.raises(ValueError):
    create_artifact_store({'type': 'bogus'})


class _FakeBlob:
  '''Fake GCS blob recording deletes.'''

  def __init__(self, exc=None):
    self.exc = exc
    self.deleted = []

  def delete(self):
    self.deleted.append('delete')
    if self.exc is not None:
      raise self.exc


class _FakeBucket:
  '''Fake GCS bucket returning a fixed blob.'''

  def __init__(self, blob):
    self._blob = blob
    self.blob_keys = []

  def blob(self, key):
    self.blob_keys.append(key)
    return self._blob


class _FakeGCSClient:
  '''Fake GCS client.'''

  def __init__(self, blob):
    self._bucket = _FakeBucket(blob)

  def bucket(self, name):
    return self._bucket


def test_gcs_delete_mocked_and_missing_noop():
  '''GCS delete calls blob.delete; NotFound is a no-op.'''
  from forecast_ml_framework.core.artifact_store import (
      GCSArtifactStore)

  class NotFound(Exception):
    '''Stand-in for google.cloud.exceptions.NotFound.'''

  blob = _FakeBlob()
  store = GCSArtifactStore('bkt')
  store._client = lambda: _FakeGCSClient(blob)
  store.delete('models/m.pkl')
  assert blob.deleted == ['delete']

  # Missing key -> no-op, never an error.
  blob2 = _FakeBlob(exc=NotFound('gone'))
  store2 = GCSArtifactStore('bkt')
  store2._client = lambda: _FakeGCSClient(blob2)
  store2.delete('models/missing.pkl')  # must not raise

  # Non-missing errors propagate.
  blob3 = _FakeBlob(exc=RuntimeError('boom'))
  store3 = GCSArtifactStore('bkt')
  store3._client = lambda: _FakeGCSClient(blob3)
  with pytest.raises(RuntimeError):
    store3.delete('models/m.pkl')


def test_gcs_delete_jail_validates_key():
  '''GCS delete validates the cloud key before any client call.'''
  from forecast_ml_framework.core.artifact_store import (
      GCSArtifactStore)

  def _boom():
    raise AssertionError('client must not be called for bad keys')

  store = GCSArtifactStore('bkt')
  store._client = _boom
  with pytest.raises(ValueError):
    store.delete('/absolute.pkl')
  with pytest.raises(ValueError):
    store.delete('../escape.pkl')


class _FakeS3Client:
  '''Fake S3 client recording delete_object calls.'''

  def __init__(self, exc=None):
    self.exc = exc
    self.calls = []

  def delete_object(self, **kwargs):
    self.calls.append(kwargs)
    if self.exc is not None:
      raise self.exc


def test_s3_delete_mocked_and_missing_noop():
  '''S3 delete calls delete_object; NoSuchKey is a no-op.'''
  from forecast_ml_framework.core.artifact_store import (
      S3ArtifactStore)

  class NoSuchKey(Exception):
    '''Stand-in for a NoSuchKey client error.'''

  fake = _FakeS3Client()
  store = S3ArtifactStore('bkt')
  store._client = lambda: fake
  store.delete('models/m.pkl')
  assert fake.calls == [{'Bucket': 'bkt', 'Key': 'models/m.pkl'}]

  # Missing key -> no-op, never an error.
  fake2 = _FakeS3Client(exc=NoSuchKey('gone'))
  store2 = S3ArtifactStore('bkt')
  store2._client = lambda: fake2
  store2.delete('models/missing.pkl')  # must not raise

  # Non-missing errors propagate.
  fake3 = _FakeS3Client(exc=RuntimeError('boom'))
  store3 = S3ArtifactStore('bkt')
  store3._client = lambda: fake3
  with pytest.raises(RuntimeError):
    store3.delete('models/m.pkl')


def test_s3_delete_jail_validates_key():
  '''S3 delete validates the cloud key before any client call.'''
  from forecast_ml_framework.core.artifact_store import (
      S3ArtifactStore)

  def _boom():
    raise AssertionError('client must not be called for bad keys')

  store = S3ArtifactStore('bkt')
  store._client = _boom
  with pytest.raises(ValueError):
    store.delete('/absolute.pkl')
  with pytest.raises(ValueError):
    store.delete('../escape.pkl')


# --- online delete: conditional Redis ------------------------------------

class _FakeRedis:
  '''Fake Redis with get/set/delete.'''

  def __init__(self):
    self.data = {}
    self.delete_calls = []

  def get(self, key):
    return self.data.get(key)

  def set(self, key, value):
    self.data[key] = value

  def delete(self, key):
    self.delete_calls.append(key)
    self.data.pop(key, None)
    return 1


class _FakeRedisRaisingDelete:
  '''Fake Redis whose delete always raises.'''

  def __init__(self):
    self.delete_calls = []

  def get(self, key):
    return None

  def set(self, key, value):
    pass

  def delete(self, key):
    self.delete_calls.append(key)
    raise RuntimeError('redis boom')


class _FakeRedisRaisingSet:
  '''Fake Redis whose set always raises (write failure).'''

  def get(self, key):
    return None

  def set(self, key, value):
    raise RuntimeError('redis write boom')

  def delete(self, key):
    return 1


def test_online_delete_conditional_redis():
  '''Online delete: Redis + memory-only + error propagation.'''
  from forecast_ml_framework.feature_store.online import (
      OnlineFeatureStore)

  # (a) Fake client: memory purged AND fake.delete called once.
  fake = _FakeRedis()
  store = OnlineFeatureStore(redis_client=fake)
  store.set_features('e1', {'f1': 1.0}, as_of='2024-01-01',
                     pii_classes=['none'])
  assert store.get_features(['e1'], as_of='2024-01-01',
                            strict_pit=True) == {'e1': {'f1': 1.0}}
  store.delete('e1')
  assert fake.delete_calls == ['feature:e1']
  assert store.get_features(['e1'], as_of='2024-01-01',
                            strict_pit=True) == {}

  # (b) Memory-only (_client() is None): purges with no error.
  store2 = OnlineFeatureStore()
  assert store2._client() is None
  store2.set_features('e1', {'f1': 1.0}, as_of='2024-01-01',
                      pii_classes=['none'])
  store2.delete('e1')  # must not raise (None.delete would crash)
  assert store2.get_features(['e1'], as_of='2024-01-01',
                             strict_pit=True) == {}
  store2.delete('e1')  # idempotent

  # (c) Redis error propagates; memory already purged; re-read empty.
  store3 = OnlineFeatureStore()
  store3.set_features('e1', {'f1': 1.0}, as_of='2024-01-01',
                      pii_classes=['none'])
  raising = _FakeRedisRaisingDelete()
  store3._redis = raising
  with pytest.raises(RuntimeError):
    store3.delete('e1')
  assert raising.delete_calls == ['feature:e1']
  assert store3.get_features(['e1'], as_of='2024-01-01',
                             strict_pit=True) == {}


def test_undeclared_pii_write_raises():
  '''Writes without a PII declaration fail closed, no write.'''
  import pandas as pd
  from forecast_ml_framework.core.artifact_store import (
      LocalArtifactStore)
  from forecast_ml_framework.feature_store.offline import (
      OfflineFeatureStore)
  from forecast_ml_framework.feature_store.online import (
      OnlineFeatureStore)
  from forecast_ml_framework.governance.consent import record_consent
  ledger = {}
  record_consent(ledger, 'e1', granted=True, actor='op-1')

  # Online set_features with undeclared email -> ValueError, no write.
  store = OnlineFeatureStore()
  with pytest.raises(ValueError):
    store.set_features('e1', {'email': 'alice@example.com'},
                       as_of='2024-01-01')
  assert store.get_features(['e1'], as_of='2024-01-01',
                            strict_pit=True) == {}
  # Declared write without a ledger fails closed, no write.
  with pytest.raises(ValueError):
    store.set_features('e1', {'email': 'alice@example.com'},
                       as_of='2024-01-01', pii_classes=['email'])
  assert store.get_features(['e1'], as_of='2024-01-01',
                            strict_pit=True) == {}
  # Declared write succeeds with a granted ledger.
  store.set_features('e1', {'email': 'alice@example.com'},
                     as_of='2024-01-01', pii_classes=['email'],
                     consent_ledger=ledger)
  assert store.get_features(['e1'], as_of='2024-01-01',
                            strict_pit=True) != {}
  # Unknown declaration fails closed.
  with pytest.raises(ValueError):
    store.set_features('e1', {'f1': 1.0}, as_of='2024-01-01',
                       pii_classes=['bogus'])

  # Offline materialize with undeclared email -> ValueError, no save.
  import tempfile
  with tempfile.TemporaryDirectory() as tmp:
    offline = OfflineFeatureStore(LocalArtifactStore(tmp))
    df = pd.DataFrame({'entity_id': ['e1'], 'as_of': ['2024-01-01'],
                       'email': ['alice@example.com']})
    with pytest.raises(ValueError):
      offline.materialize(df, pii_classes=['none'])
    assert offline._all_keys() == []
    # Declared write without a ledger fails closed, no save.
    with pytest.raises(ValueError):
      offline.materialize(df, pii_classes=['email'])
    assert offline._all_keys() == []
    offline.materialize(df, pii_classes=['email'],
                        consent_ledger=ledger)
    assert offline._all_keys() != []
    # Online materialize guard.
    df2 = pd.DataFrame({'entity_id': ['e1'], 'as_of': ['2024-01-01'],
                        'f1': [1.0]})
    online = OnlineFeatureStore()
    online.materialize(df2, pii_classes=['none'])
    with pytest.raises(ValueError):
      online.materialize(df, pii_classes=['none'])
    with pytest.raises(ValueError):
      online.materialize(df2, pii_classes=['bogus'])


# --- offline delete_entity ------------------------------------------------

def _offline_tmp(tmp_path):
  from forecast_ml_framework.core.artifact_store import (
      LocalArtifactStore)
  from forecast_ml_framework.feature_store.offline import (
      OfflineFeatureStore)
  return OfflineFeatureStore(LocalArtifactStore(str(tmp_path)))


def _frame(rows):
  import pandas as pd
  return pd.DataFrame(rows)


def test_delete_entity_1_of_3(tmp_path):
  '''1-of-3: subject in exactly one snapshot -> emptied key deleted.'''
  offline = _offline_tmp(tmp_path)
  key_subject_only = offline.materialize(
      _frame([{'entity_id': 'subject-A', 'f1': 1.0}]),
      as_of='2024-01-01', pii_classes=['none'])
  key_others = offline.materialize(
      _frame([{'entity_id': 'alice', 'f1': 2.0},
              {'entity_id': 'bob', 'f1': 3.0}]),
      as_of='2024-01-02', pii_classes=['none'])
  key_alice = offline.materialize(
      _frame([{'entity_id': 'alice', 'f1': 4.0}]),
      as_of='2024-01-03', pii_classes=['none'])
  assert len(offline._all_keys()) == 3

  offline.delete_entity('subject-A')

  # Subject snapshot emptied -> key deleted via ArtifactStore.delete.
  assert not offline.artifact_store.exists(key_subject_only)
  # Other snapshots intact.
  assert offline.artifact_store.exists(key_others)
  assert offline.artifact_store.exists(key_alice)
  # Re-read: subject empty, others intact.
  assert offline.get_features(
      ['subject-A'], as_of='2024-01-03', strict_pit=True).empty
  out = offline.get_features(['alice', 'bob'],
                             as_of='2024-01-03', strict_pit=True)
  assert list(out['entity_id']) == ['alice', 'bob']
  # Idempotent: second delete is a no-op.
  offline.delete_entity('subject-A')
  assert offline.get_features(
      ['subject-A'], as_of='2024-01-03', strict_pit=True).empty


def test_delete_entity_rewrite_keeps_others(tmp_path):
  '''Mixed snapshot is row-filtered and rewritten, not deleted.'''
  offline = _offline_tmp(tmp_path)
  key = offline.materialize(
      _frame([{'entity_id': 'subject-A', 'f1': 1.0},
              {'entity_id': 'alice', 'f1': 2.0}]),
      as_of='2024-01-04', pii_classes=['none'])
  offline.delete_entity('subject-A')
  assert offline.artifact_store.exists(key)
  frame = offline.load(key=key)
  assert list(frame['entity_id']) == ['alice']
  assert offline.get_features(
      ['subject-A'], as_of='2024-01-04', strict_pit=True).empty


# --- no clear-ID logging ---------------------------------------------------

def test_no_clear_id_in_logs_or_audit(caplog):
  '''Clear subject IDs never reach logs or audit entries.'''
  from forecast_ml_framework.feature_store.online import (
      OnlineFeatureStore)
  from forecast_ml_framework.governance.audit import audit_log
  from forecast_ml_framework.governance.pii import detect_pii
  from forecast_ml_framework.governance.scrub import subject_hash

  clear = 'SECRET-SUBJECT-9f3a-ENTITY'
  digest = subject_hash(clear)
  assert clear not in digest
  assert len(digest) == 64

  # Audit: only the hash is recorded, never the clear ID.
  entries = []
  audit_log('erasure', actor=digest,
            details={'subject_hash': digest}, log=entries)
  assert entries
  assert clear not in str(entries)
  assert digest in str(entries)

  # Findings: detail carries no raw PII.
  import pandas as pd
  df = pd.DataFrame({'email': [f'{clear}@example.com']})
  findings = detect_pii(df)
  assert findings
  assert clear not in ' '.join(f.detail for f in findings)

  # Logs: Redis write failure logs the hash, not the clear ID.
  store = OnlineFeatureStore(redis_client=_FakeRedisRaisingSet())
  with caplog.at_level(logging.WARNING):
    store.set_features(clear, {'f1': 1.0}, as_of='2024-01-01',
                       pii_classes=['none'])
  assert clear not in caplog.text
  assert digest in caplog.text


# --- sole-leaf grep guards --------------------------------------------------

def _read_src(rel):
  from pathlib import Path
  root = Path(__file__).resolve().parents[1]
  return (root / 'src' / 'forecast_ml_framework' / rel).read_text(
      encoding='utf-8')


def test_sole_leaf_grep_guards():
  '''One subject_hash, one scrub surface, no second hasher/scrubber.'''
  scrub = _read_src('governance/scrub.py')
  assert scrub.count('def subject_hash') == 1
  assert 'import hashlib' in scrub
  assert 'def scrub_text' in scrub or 'scrub_text =' in scrub

  for rel in ('governance/pii.py', 'governance/retention.py',
             'governance/erasure.py', 'governance/consent.py',
             'pipelines/data_prep/streaming.py',
             'pipelines/data_prep/quality.py'):
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    path = root / 'src' / 'forecast_ml_framework' / rel
    if not path.exists():
      continue  # P1-D2 / ADR-gated modules not landed yet.
    text = path.read_text(encoding='utf-8')
    for banned in ('re.sub', 're.compile', 'hashlib.sha256',
                   'import hashlib'):
      assert banned not in text, f'{banned} found in {rel}'

  for rel in ('governance/pii.py', 'core/artifact_store.py',
             'feature_store/online.py', 'feature_store/offline.py'):
    text = _read_src(rel)
    assert 'def subject_hash' not in text
    assert 'def scrub_' not in text
    assert 'pickle.load(' not in text
    assert 'pickle.loads(' not in text

  # pii.py consumes the leaf, it does not reimplement it.
  pii_src = _read_src('governance/pii.py')
  assert 'subject_hash' in pii_src


# --- calibrated allowlist: restricted round-trip ------------------------------

_CALIBRATED_7 = {
    ('sklearn.calibration', 'CalibratedClassifierCV'),
    ('sklearn.calibration', '_CalibratedClassifier'),
    ('sklearn.calibration', '_SigmoidCalibration'),
    ('sklearn.isotonic', 'IsotonicRegression'),
    ('forecast_ml_framework.pipelines.modeling.calibration',
     '_BetaCalWrapper'),
    ('betacal', 'BetaCalibration'),
    ('betacal.beta_calibration', '_BetaCal'),
}


def test_calibrated_envelope_restricted_roundtrip():
  '''Calibrated models round-trip through restricted_loads.'''
  import numpy as np
  from sklearn.calibration import CalibratedClassifierCV
  from sklearn.linear_model import LogisticRegression

  from forecast_ml_framework.core.artifact_store import (
      _SAFE_GLOBALS, restricted_loads)

  # The 7 allowlist tuples are present (consumed by P1-A2).
  for entry in _CALIBRATED_7:
    assert entry in _SAFE_GLOBALS, entry

  rng_data = np.array(
      [[0], [1], [2], [3], [4], [5], [6], [7]], dtype=float)
  labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
  for method in ('sigmoid', 'isotonic'):
    calibrated = CalibratedClassifierCV(
        LogisticRegression(max_iter=1000), method=method, cv=2)
    calibrated.fit(rng_data, labels)
    envelope = {'model': calibrated,
                'label_set': ['0', '1'],
                'registry_ref': None,
                'onnx_sidecar': None}
    raw = pickle.dumps(envelope)
    restored = restricted_loads(raw)
    assert isinstance(restored, dict)
    expected = calibrated.predict_proba(rng_data)
    got = restored['model'].predict_proba(rng_data)
    assert np.allclose(got, expected)

  # BetaCal path (betacal is optional).
  betacal = pytest.importorskip('betacal')
  from forecast_ml_framework.pipelines.modeling.calibration import (
      _BetaCalWrapper)
  base = LogisticRegression(max_iter=1000)
  base.fit(rng_data, labels)
  calibrator = betacal.BetaCalibration()
  calibrator.fit(base.predict_proba(rng_data)[:, 1], labels)
  wrapped = _BetaCalWrapper(base, calibrator)
  restored = restricted_loads(pickle.dumps({'model': wrapped}))
  assert np.allclose(restored['model'].predict_proba(rng_data),
                     wrapped.predict_proba(rng_data))


def test_restricted_loads_rejects_non_allowlisted_global():
  '''restricted_loads rejects globals outside the allowlist.'''
  import os
  from forecast_ml_framework.core.artifact_store import (
      restricted_loads)
  with pytest.raises(ValueError):
    restricted_loads(pickle.dumps(os.system))


def test_detect_pii_skips_missing_and_hostile_values():
  '''detect_pii skips None and unstringifiable cells.'''
  import pandas as pd
  from forecast_ml_framework.governance.pii import detect_pii

  class _Hostile:
    '''Value whose str() always raises.'''

    def __str__(self):
      raise RuntimeError('nope')

    def __repr__(self):
      raise RuntimeError('nope')

  df = pd.DataFrame({'f1': [None, _Hostile(), 1.0]})
  assert detect_pii(df) == []


def test_mask_pii_preserves_missing_cells():
  '''mask_pii leaves None/NaN/NA cells missing, hashes the rest.'''
  import pandas as pd
  from forecast_ml_framework.governance.pii import mask_pii
  df = pd.DataFrame({'user_id': ['user-a', None, float('nan')]})
  masked = mask_pii(df, pii_classes=['user_id'])
  assert len(masked['user_id'][0]) == 64
  assert pd.isna(masked['user_id'][1])
  assert pd.isna(masked['user_id'][2])
  df2 = pd.DataFrame({'user_id': ['user-b', pd.NA]})
  masked2 = mask_pii(df2, pii_classes=['user_id'])
  assert len(masked2['user_id'][0]) == 64
  assert pd.isna(masked2['user_id'][1])


def test_guard_write_consent_granted_and_missing_column():
  '''guard_write honors granted consent; missing column fails.'''
  import sys
  import types
  import pandas as pd
  from forecast_ml_framework.governance.pii import guard_write

  real = sys.modules.get(
      'forecast_ml_framework.governance.consent')
  fake = types.ModuleType('forecast_ml_framework.governance.consent')
  fake.check_consent = lambda ledger, entity: True
  sys.modules['forecast_ml_framework.governance.consent'] = fake
  try:
    df = pd.DataFrame({'entity_id': ['e1', 'e1'], 'f1': [1.0, 2.0]})
    guard_write(df, pii_classes=['none'], consent_ledger={})
    # Missing entity column fails closed even with consent.
    with pytest.raises(ValueError):
      guard_write(pd.DataFrame({'f1': [1.0]}),
                  pii_classes=['none'], consent_ledger={})
    fake.check_consent = lambda ledger, entity: False
    with pytest.raises(ValueError):
      guard_write(df, pii_classes=['none'], consent_ledger={})
  finally:
    if real is None:
      sys.modules.pop(
          'forecast_ml_framework.governance.consent', None)
    else:
      sys.modules['forecast_ml_framework.governance.consent'] = real


def test_set_features_entity_collision_cannot_bypass_consent():
  '''Smuggled entity_id in features cannot bypass consent.'''
  from forecast_ml_framework.feature_store.online import (
    OnlineFeatureStore)
  from forecast_ml_framework.governance.consent import (
    record_consent)
  ledger = {}
  record_consent(ledger, 'consented-e', granted=True, actor='op-1')
  store = OnlineFeatureStore()
  # Victim 'victim-e' has no consent; the features dict
  # smuggles the consented entity id under 'entity_id'.
  with pytest.raises(ValueError):
    store.set_features(
      'victim-e', {'entity_id': 'consented-e', 'f1': 1.0},
      as_of='2024-01-01', pii_classes=['none'],
      consent_ledger=ledger)
  assert store.get_features(
    ['victim-e'], as_of='2024-01-01', strict_pit=True) == {}
