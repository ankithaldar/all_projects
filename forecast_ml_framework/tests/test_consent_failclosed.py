#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Fail-closed consent ledger guard (FIX-ADV-3).

``guard_write`` must not bypass consent when no ledger is
supplied: with ``consent_ledger=None`` a write is allowed
only for an explicit ``['none']`` declaration (no PII) or
an undeclared frame with no detected PII. Declared PII
requires a ledger, and a granted ledger record lets a
PII frame pass.
'''

import pytest


def test_pii_frame_without_ledger_raises():
  '''PII frame with no ledger fails closed.'''
  import pandas as pd
  from forecast_ml_framework.governance.pii import guard_write
  frame = pd.DataFrame(
      {'entity_id': ['e1'], 'email': ['alice@example.com']})
  with pytest.raises(ValueError):
    guard_write(frame, pii_classes=['email'])


def test_declared_pii_without_ledger_raises():
  '''Declared PII with no ledger fails closed.'''
  import pandas as pd
  from forecast_ml_framework.governance.pii import guard_write
  frame = pd.DataFrame({'entity_id': ['e1'], 'f1': [1.0]})
  with pytest.raises(ValueError):
    guard_write(frame, pii_classes=['email'])


def test_undeclared_pii_without_ledger_raises():
  '''PII frame with no declaration and no ledger fails closed.'''
  import pandas as pd
  from forecast_ml_framework.governance.pii import guard_write
  frame = pd.DataFrame(
      {'entity_id': ['e1'], 'email': ['alice@example.com']})
  with pytest.raises(ValueError):
    guard_write(frame, pii_classes=None)


def test_explicit_none_without_ledger_passes():
  '''Explicit ['none'] needs no ledger.'''
  import pandas as pd
  from forecast_ml_framework.governance.pii import guard_write
  frame = pd.DataFrame({'entity_id': ['e1'], 'f1': [1.0]})
  guard_write(frame, pii_classes=['none'])


def test_granted_ledger_passes_for_pii_frame():
  '''PII frame with granted consent passes.'''
  import pandas as pd
  from forecast_ml_framework.governance.consent import record_consent
  from forecast_ml_framework.governance.pii import guard_write
  ledger = {}
  record_consent(ledger, 'e1', granted=True, actor='op-1')
  frame = pd.DataFrame(
      {'entity_id': ['e1'], 'email': ['alice@example.com']})
  guard_write(frame, pii_classes=['email'], consent_ledger=ledger)
