#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests for the __main__ entry point version handling.'''

import importlib
import importlib.metadata
import os
import subprocess
import sys
from pathlib import Path


def test_version_from_metadata_with_fallback(monkeypatch):
  '''_VERSION uses importlib.metadata, falling back to 0.1.0.'''
  import forecast_ml_framework.__main__ as main_mod

  monkeypatch.setattr(
      importlib.metadata, 'version', lambda name: '9.9.9')
  importlib.reload(main_mod)
  try:
    assert '9.9.9' in main_mod._VERSION
    assert 'forecast-ml' in main_mod._VERSION

    def _missing(name):
      raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, 'version', _missing)
    importlib.reload(main_mod)
    assert '0.1.0' in main_mod._VERSION
  finally:
    importlib.reload(main_mod)


def test_python_m_version():
  '''python -m forecast_ml_framework --version exits zero.'''
  root = Path(__file__).resolve().parents[1]
  src = str(root / 'src')
  env = dict(os.environ)
  env['PYTHONPATH'] = src + os.pathsep + env.get('PYTHONPATH', '')
  proc = subprocess.run(
      [sys.executable, '-m', 'forecast_ml_framework', '--version'],
      capture_output=True,
      text=True,
      env=env,
      timeout=60,
      check=False,
  )
  assert proc.returncode == 0
  assert 'forecast-ml' in proc.stdout
  assert '0.1.0' in proc.stdout or '9.9.9' not in proc.stdout
