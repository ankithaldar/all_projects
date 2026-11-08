#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Separator-optional PII detection for phone/SSN/IBAN.

TDD RED: continuous, spaced, and lower case values in a
neutral column (``notes``) must be detected so
``guard_write`` and ``materialize`` fail closed on
``['none']``/None. Clean controls must stay unflagged.
'''

import pytest


def _frame(column, value):
  '''Build a one-row frame with a neutral column name.'''
  import pandas as pd
  return pd.DataFrame({column: [value], 'entity_id': ['e1']})


def test_detect_continuous_phone_in_notes():
  '''detect_pii flags a continuous 10-digit phone.'''
  from forecast_ml_framework.governance.pii import detect_pii
  findings = detect_pii(_frame('notes', '5551234567'))
  assert 'user_id' in {f.pii_class for f in findings}
  joined = ' '.join(f.detail for f in findings)
  assert '5551234567' not in joined


def test_detect_int_phone_in_notes():
  '''detect_pii flags an integer phone cell.'''
  from forecast_ml_framework.governance.pii import detect_pii
  findings = detect_pii(_frame('notes', 5551234567))
  assert 'user_id' in {f.pii_class for f in findings}


def test_detect_plus_one_continuous_phone():
  '''detect_pii flags +1 continuous phone.'''
  from forecast_ml_framework.governance.pii import detect_pii
  findings = detect_pii(_frame('notes', '+15551234567'))
  assert 'user_id' in {f.pii_class for f in findings}


def test_detect_parenthesized_phone_no_separator():
  '''detect_pii flags paren phone with no separator.'''
  from forecast_ml_framework.governance.pii import detect_pii
  findings = detect_pii(_frame('notes', '(555)1234567'))
  assert 'user_id' in {f.pii_class for f in findings}


def test_detect_spaced_ssn_in_notes():
  '''detect_pii flags a spaced SSN as financial.'''
  from forecast_ml_framework.governance.pii import detect_pii
  findings = detect_pii(_frame('notes', '123 45 6789'))
  assert 'financial' in {f.pii_class for f in findings}
  joined = ' '.join(f.detail for f in findings)
  assert '123 45 6789' not in joined


def test_detect_continuous_ssn_in_notes():
  '''detect_pii flags a continuous 9-digit SSN.'''
  from forecast_ml_framework.governance.pii import detect_pii
  findings = detect_pii(_frame('notes', '123456789'))
  assert 'financial' in {f.pii_class for f in findings}


def test_detect_lowercase_iban_in_notes():
  '''detect_pii flags a lowercase IBAN as financial.'''
  from forecast_ml_framework.governance.pii import detect_pii
  iban = 'de89370400440532013000'
  findings = detect_pii(_frame('notes', iban))
  assert 'financial' in {f.pii_class for f in findings}
  joined = ' '.join(f.detail for f in findings)
  assert iban not in joined


def test_detect_lowercase_spaced_iban_in_notes():
  '''detect_pii flags a lowercase spaced IBAN.'''
  from forecast_ml_framework.governance.pii import detect_pii
  iban = 'de89 3704 0044 0532 0130 00'
  findings = detect_pii(_frame('notes', iban))
  assert 'financial' in {f.pii_class for f in findings}


def test_guard_write_separator_payloads_blocked():
  '''guard_write blocks C-2 payloads on none/None.'''
  from forecast_ml_framework.governance.pii import guard_write
  cases = ['5551234567', 5551234567, '+15551234567',
           '(555)1234567', '123 45 6789', '123456789',
           'de89370400440532013000']
  for value in cases:
    frame = _frame('notes', value)
    with pytest.raises(ValueError):
      guard_write(frame, pii_classes=['none'])
    with pytest.raises(ValueError):
      guard_write(frame, pii_classes=None)


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


def test_offline_materialize_separator_payloads_blocked(
    tmp_path):
  '''Offline write blocks separator payloads by default.'''
  import pandas as pd
  from forecast_ml_framework.core.artifact_store import (
      LocalArtifactStore)
  from forecast_ml_framework.feature_store.offline import (
      OfflineFeatureStore)
  offline = OfflineFeatureStore(
      LocalArtifactStore(str(tmp_path)))
  frame = pd.DataFrame({'entity_id': ['e1'],
                        'as_of': ['2024-01-01'],
                        'notes': ['5551234567']})
  with pytest.raises(ValueError):
    offline.materialize(frame, pii_classes=['none'])
  with pytest.raises(ValueError):
    offline.materialize(frame)
  assert offline._all_keys() == []
