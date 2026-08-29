#!/usr/bin/env python
# -- coding: utf-8 --

'''Shared fixtures for the Chapter 1 test suite.'''


from __future__ import annotations

import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / 'src'
SRC = SRC / 'learning_agentic_ai_with_ai'
if str(SRC) not in sys.path:
  sys.path.insert(0, str(SRC))


# pylint: disable=redefined-outer-name


@pytest.fixture(scope='session', autouse=True)
def _seed_ops_db() -> None:
  '''Point the mock ops DB at a temp file and seed it once for the suite.'''
  from agentic_common import paths
  from chapter01_mcp.servers.ops_db import seed_if_empty

  paths.ensure_data_dirs()
  seed_if_empty()
  yield


@pytest.fixture()
def settings():
  '''Default settings instance (production-shaped defaults).'''
  from agentic_common.settings import default_settings

  return default_settings()
