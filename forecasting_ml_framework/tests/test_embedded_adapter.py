#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Phase 2 retention and routing verification.

The embedded adapter introduces no retention contract because the second
persistence path (managed relational backend) does not exist yet. Once
§7.2 delivers the embedded adapter, retention can be added by comparing the
local file's artefacts against the current registry entry. This test pins
that gap so a future upgrade does not silently claim retention.
'''

from __future__ import annotations

from pathlib import Path

import pytest

from forecasting_ml_framework.persistence.sqlite_adapter import EmbeddedSqlDataset, WRITE_MODES
from forecasting_ml_framework import constants


def test_embedded_adapter_contract_and_no_retention() -> None:
  '''The adapter provides load/save/exists/describe; retention is absent.'''
  adapter = EmbeddedSqlDataset(
    table_name='audit_table',
    database='test',
    path=str(Path('/tmp/test_embedded')),
  )
  desc = adapter._describe()
  assert desc['adapter_type'] == 'embedded_sqlite'
  assert 'table_name' in desc
  assert 'path' in desc
  # The embedded adapter does not declare a retention mechanism,
  # because the second persistence path does not exist yet (§10.4).
  assert 'retention_contract' not in desc
  assert adapter.write_mode in WRITE_MODES
