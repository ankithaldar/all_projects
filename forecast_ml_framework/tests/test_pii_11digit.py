#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Bare 11-digit phone (country code without plus).

TDD RED: continuous '15551234567' and the integer 15551234567
in a neutral column (``notes``) must be detected so
``guard_write``, ``mask_pii``, and ``materialize`` fail
closed on ``['none']``/None. Clean controls must stay
unflagged and 11 digits not starting with 1 must not be
flagged as a phone.
'''

import pytest


def _frame(column, value):
  '''Build a one-row frame with a neutral column.'''
  import pandas as pd
  return pd.DataFrame({column: [value], 'entity_id': ['e1']})


def test_detect_bare_11digit_phone_in_notes():
  '''detect_pii flags bare 11-digit phone.'''
  from forecast_ml_framework.governance.pii import detect_pii
  findings = detect_pii(_frame('notes', '15551234567'))
  assert 'user_id' in {f.pii_class for f in findings}
  joined = ' '.join(f.detail for f in findings)
  assert '15551234567' not in joined


def test_detect_int_bare_11digit_phone_in_notes():
  '''detect_pii flags integer bare 11-digit phone.'''
  from forecast_ml_framework.governance.pii import detect_pii
  findings = detect_pii(_frame('notes', 15551234567))
  assert 'user_id' in {f.pii_class for f in findings}


def test_detect_bare_11digit_phone_variants():
  '''detect_pii flags spaced/dashed bare 11-digit phone.'''
  from forecast_ml_framework.governance.pii import detect_pii
  for value in ['1 555 123 4567', '1-555-123-4567',
                '1.555.123.4567']:
    findings = detect_pii(_frame('notes', value))
    assert 'user_id' in {f.pii_class for f in findings}


def test_detect_11digit_non_one_not_phone():
  '''detect_pii leaves 11 digits not starting 1 unflagged.'''
  from forecast_ml_framework.governance.pii import detect_pii
  findings = detect_pii(_frame('notes', '25551234567'))
  assert 'user_id' not in {f.pii_class for f in findings}
  assert findings == []


def test_detect_clean_values_no_finding():
  '''detect_pii leaves clean values unflagged.'''
  import pandas as pd
  from forecast_ml_framework.governance.pii import detect_pii
  df = pd.DataFrame({'notes': ['hello world', 'Paris',
                                '2024-01-01', '555-123',
                                'DE89', '123-45'],
                      'entity_id': ['e1', 'e2', 'e3', 'e4',
                                    'e5', 'e6']})
  assert detect_pii(df) == []


def test_guard_write_bare_11digit_blocked():
  '''guard_write blocks bare 11-digit payloads.'''
  from forecast_ml_framework.governance.pii import guard_write
  for value in ['15551234567', 15551234567,
                '1 555 123 4567', '1-555-123-4567']:
    frame = _frame('notes', value)
    with pytest.raises(ValueError):
      guard_write(frame, pii_classes=['none'])
    with pytest.raises(ValueError):
      guard_write(frame, pii_classes=None)


def test_mask_pii_bare_11digit_fails_closed():
  '''mask_pii fails closed on bare 11-digit phone.'''
  from forecast_ml_framework.governance.pii import mask_pii
  with pytest.raises(ValueError):
    mask_pii(_frame('notes', '15551234567'),
             pii_classes=['none'])


def test_mask_pii_bare_11digit_hashed_when_declared():
  '''mask_pii hashes bare 11-digit phone when declared.'''
  import pandas as pd
  from forecast_ml_framework.governance.pii import mask_pii
  df = pd.DataFrame({'notes': ['15551234567']})
  masked = mask_pii(df, pii_classes=['user_id'])
  blob = masked.astype(str).to_string()
  assert '15551234567' not in blob
  for cell in masked['notes']:
    assert len(cell) == 64
    int(cell, 16)


def test_offline_materialize_bare_11digit_blocked(
    tmp_path):
  '''Offline write blocks bare 11-digit phone by default.'''
  import pandas as pd
  from forecast_ml_framework.core.artifact_store import (
      LocalArtifactStore)
  from forecast_ml_framework.feature_store.offline import (
      OfflineFeatureStore)
  offline = OfflineFeatureStore(
      LocalArtifactStore(str(tmp_path)))
  frame = pd.DataFrame({'entity_id': ['e1'],
                        'as_of': ['2024-01-01'],
                        'notes': ['15551234567']})
  with pytest.raises(ValueError):
    offline.materialize(frame, pii_classes=['none'])
  with pytest.raises(ValueError):
    offline.materialize(frame)
  assert offline._all_keys() == []
