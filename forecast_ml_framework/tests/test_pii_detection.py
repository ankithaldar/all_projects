#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''PII detection for phone, SSN, IBAN, and full name.

TDD RED: phone/full_name/SSN/IBAN values in a neutral
column (``notes``) must be detected so ``guard_write``
and ``mask_pii`` fail closed on ``['none']``/None.
'''

import pytest


def _frame(column, value):
  '''Build a one-row frame with a neutral column name.'''
  import pandas as pd
  return pd.DataFrame({column: [value], 'entity_id': ['e1']})


def test_detect_phone_in_neutral_column():
  '''detect_pii flags a dashed phone in ``notes``.'''
  from forecast_ml_framework.governance.pii import detect_pii
  findings = detect_pii(_frame('notes', 'call 555-123-4567'))
  by_class = {f.pii_class for f in findings}
  assert 'user_id' in by_class
  joined = ' '.join(f.detail for f in findings)
  assert '555-123-4567' not in joined


def test_detect_phone_plus_one_in_neutral_column():
  '''detect_pii flags a +1 phone in ``notes``.'''
  from forecast_ml_framework.governance.pii import detect_pii
  findings = detect_pii(_frame('notes', '+1-555-123-4567'))
  assert 'user_id' in {f.pii_class for f in findings}


def test_detect_ssn_in_neutral_column():
  '''detect_pii flags an SSN in ``notes`` as financial.'''
  from forecast_ml_framework.governance.pii import detect_pii
  findings = detect_pii(_frame('notes', '123-45-6789'))
  assert 'financial' in {f.pii_class for f in findings}
  joined = ' '.join(f.detail for f in findings)
  assert '123-45-6789' not in joined


def test_detect_iban_in_neutral_column():
  '''detect_pii flags an IBAN in ``notes`` as financial.'''
  from forecast_ml_framework.governance.pii import detect_pii
  iban = 'DE89370400440532013000'
  findings = detect_pii(_frame('notes', iban))
  assert 'financial' in {f.pii_class for f in findings}
  joined = ' '.join(f.detail for f in findings)
  assert iban not in joined


def test_detect_full_name_in_neutral_column():
  '''detect_pii flags a full name in ``notes``.'''
  from forecast_ml_framework.governance.pii import detect_pii
  findings = detect_pii(_frame('notes', 'John Doe'))
  assert 'user_id' in {f.pii_class for f in findings}
  joined = ' '.join(f.detail for f in findings)
  assert 'John Doe' not in joined


def test_detect_phone_and_name_column_keywords():
  '''detect_pii flags phone/full_name column names.'''
  import pandas as pd
  from forecast_ml_framework.governance.pii import detect_pii
  df = pd.DataFrame({
      'phone': ['call me'], 'full_name': ['Madonna'],
      'notes': ['hello world']})
  by_class = {f.pii_class for f in detect_pii(df)}
  assert 'user_id' in by_class


def test_guard_write_phone_in_notes_blocked():
  '''guard_write blocks a phone in ``notes`` on none/None.'''
  from forecast_ml_framework.governance.pii import guard_write
  frame = _frame('notes', '555-123-4567')
  with pytest.raises(ValueError):
    guard_write(frame, pii_classes=['none'])
  with pytest.raises(ValueError):
    guard_write(frame, pii_classes=None)


def test_guard_write_ssn_in_notes_blocked():
  '''guard_write blocks an SSN in ``notes`` on none.'''
  from forecast_ml_framework.governance.pii import guard_write
  with pytest.raises(ValueError):
    guard_write(_frame('notes', '123-45-6789'),
                pii_classes=['none'])


def test_guard_write_iban_in_notes_blocked():
  '''guard_write blocks an IBAN in ``notes`` on none.'''
  from forecast_ml_framework.governance.pii import guard_write
  with pytest.raises(ValueError):
    guard_write(_frame('notes', 'DE89370400440532013000'),
                pii_classes=['none'])


def test_guard_write_full_name_in_notes_blocked():
  '''guard_write blocks a full name in ``notes`` on none.'''
  from forecast_ml_framework.governance.pii import guard_write
  with pytest.raises(ValueError):
    guard_write(_frame('notes', 'John Doe'), pii_classes=['none'])


def test_mask_pii_phone_in_notes_fails_closed():
  '''mask_pii fails closed on a phone in ``notes``.'''
  from forecast_ml_framework.governance.pii import mask_pii
  with pytest.raises(ValueError):
    mask_pii(_frame('notes', '555-123-4567'),
             pii_classes=['none'])


def test_mask_pii_ssn_iban_redacted_when_declared():
  '''mask_pii redacts SSN/IBAN cells when financial declared.'''
  import pandas as pd
  from forecast_ml_framework.governance.pii import mask_pii
  from forecast_ml_framework.governance.scrub import REDACTED
  df = pd.DataFrame({'notes': ['123-45-6789',
                               'DE89370400440532013000']})
  masked = mask_pii(df, pii_classes=['financial'])
  assert list(masked['notes']) == [REDACTED, REDACTED]
  blob = masked.astype(str).to_string()
  assert '123-45-6789' not in blob
  assert 'DE89370400440532013000' not in blob


def test_mask_pii_phone_name_hashed_when_declared():
  '''mask_pii hashes phone/name cells when user_id declared.'''
  import pandas as pd
  from forecast_ml_framework.governance.pii import mask_pii
  df = pd.DataFrame({'notes': ['555-123-4567', 'John Doe']})
  masked = mask_pii(df, pii_classes=['user_id'])
  blob = masked.astype(str).to_string()
  assert '555-123-4567' not in blob
  assert 'John Doe' not in blob
  for cell in masked['notes']:
    assert len(cell) == 64
    int(cell, 16)


def test_guard_write_declared_pii_missing_entity_raises():
  '''guard_write raises when declared PII lacks an entity col.'''
  import pandas as pd
  from forecast_ml_framework.governance.consent import record_consent
  from forecast_ml_framework.governance.pii import guard_write
  ledger = {}
  record_consent(ledger, 'e1', granted=True, actor='op-1')
  frame = pd.DataFrame(
      {'customer_id': ['e1'], 'email': ['alice@example.com']})
  with pytest.raises(ValueError):
    guard_write(frame, pii_classes=['email'],
                consent_ledger=ledger)


def test_guard_write_explicit_entity_col_passes():
  '''guard_write honors an explicit non-canonical entity col.'''
  import pandas as pd
  from forecast_ml_framework.governance.consent import record_consent
  from forecast_ml_framework.governance.pii import guard_write
  ledger = {}
  record_consent(ledger, 'e1', granted=True, actor='op-1')
  frame = pd.DataFrame(
      {'customer_id': ['e1'], 'email': ['alice@example.com']})
  guard_write(frame, pii_classes=['email'], consent_ledger=ledger,
              entity_col='customer_id')


def test_detect_clean_values_no_finding():
  '''detect_pii leaves clean values unflagged.'''
  import pandas as pd
  from forecast_ml_framework.governance.pii import detect_pii
  df = pd.DataFrame({'notes': ['hello world', 'Paris', '2024-01-01',
                               '555-123', 'DE89', '123-45'],
                     'entity_id': ['e1', 'e2', 'e3', 'e4', 'e5',
                                   'e6']})
  assert detect_pii(df) == []


def test_online_set_features_phone_in_notes_blocked():
  '''Online write with a phone in a feature fails closed.'''
  from forecast_ml_framework.feature_store.online import (
      OnlineFeatureStore)
  store = OnlineFeatureStore()
  with pytest.raises(ValueError):
    store.set_features('e1', {'notes': '555-123-4567'},
                       as_of='2024-01-01', pii_classes=['none'])
  assert store.get_features(['e1'], as_of='2024-01-01',
                            strict_pit=True) == {}


def test_offline_materialize_phone_in_notes_blocked(tmp_path):
  '''Offline write with a phone in a column fails closed.'''
  import pandas as pd
  from forecast_ml_framework.core.artifact_store import (
      LocalArtifactStore)
  from forecast_ml_framework.feature_store.offline import (
      OfflineFeatureStore)
  offline = OfflineFeatureStore(
      LocalArtifactStore(str(tmp_path)))
  frame = pd.DataFrame({'entity_id': ['e1'],
                        'as_of': ['2024-01-01'],
                        'notes': ['555-123-4567']})
  with pytest.raises(ValueError):
    offline.materialize(frame, pii_classes=['none'])
  assert offline._all_keys() == []
