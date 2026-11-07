#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Lazy modeling package init tests (Tester-I12, TDD RED first).'''

import subprocess
import sys
from pathlib import Path


def test_modeling_import_is_lazy_subprocess():
  '''Importing modeling package must not import .pipeline.'''
  root = Path(__file__).resolve().parents[1]
  import os
  code = (
      'import sys\n'
      'import forecast_ml_framework.pipelines.modeling as m\n'
      'assert "forecast_ml_framework.pipelines.modeling.pipeline"'
      ' not in sys.modules, list(sys.modules)\n'
      'assert hasattr(m, "create_pipeline")\n'
      'print("lazy ok")\n')
  proc = subprocess.run(
      [sys.executable, '-c', code],
      capture_output=True, text=True, timeout=60,
      env={**dict(os.environ),
           'PYTHONPATH': str(root / 'src') + os.pathsep
           + os.environ.get('PYTHONPATH', '')},
      check=False)
  assert proc.returncode == 0, proc.stderr
  assert 'lazy ok' in proc.stdout


def test_modeling_leaf_import_does_not_load_pipeline():
  '''Leaf import must not pull in .pipeline.'''
  root = Path(__file__).resolve().parents[1]
  import os
  code = (
      'import sys\n'
      'import forecast_ml_framework.pipelines.modeling.validation\n'
      'assert "forecast_ml_framework.pipelines.modeling.pipeline"'
      ' not in sys.modules, list(sys.modules)\n'
      'print("leaf ok")\n')
  proc = subprocess.run(
      [sys.executable, '-c', code],
      capture_output=True, text=True, timeout=60,
      env={**dict(os.environ),
           'PYTHONPATH': str(root / 'src') + os.pathsep
           + os.environ.get('PYTHONPATH', '')},
      check=False)
  assert proc.returncode == 0, proc.stderr
  assert 'leaf ok' in proc.stdout
