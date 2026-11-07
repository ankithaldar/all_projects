#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Telemetry test fixtures (stub per plan T1, loop14).

The full T1 context fixtures (run_id reset, spool
lifecycle) land with the T1 correlation-context work.
Until then this stub provides only a spool tmp dir and
a no-op run_id reset hook, so telemetry tests can
depend on a stable fixture location without importing
not-yet-landed T1 modules.
'''

import pytest


@pytest.fixture(name='telemetry_spool_tmp')
def fixture_telemetry_spool_tmp(tmp_path):
  '''Provide a spool tmp directory for telemetry tests.'''
  spool = tmp_path / 'spool'
  spool.mkdir(parents=True, exist_ok=True)
  return spool


@pytest.fixture(autouse=True)
def fixture_reset_telemetry_run_id():
  '''Reset telemetry run_id state when T1 context exists.

  No-op while the T1 context module is absent: the
  reset hook is imported defensively so this stub
  never breaks tests for the typed-error hierarchy.
  '''
  yield
  try:
    from forecast_ml_framework.telemetry import context as ctx
  except Exception:
    return
  reset = getattr(ctx, 'reset_run_id', None)
  if callable(reset):
    try:
      reset()
    except Exception:
      pass
