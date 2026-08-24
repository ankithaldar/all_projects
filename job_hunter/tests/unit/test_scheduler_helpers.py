#!/usr/bin/env python
# -- coding: utf-8 --

'''Tests for cron conversion and worker helpers.'''


from __future__ import annotations

import pytest
from job_hunter.workers.scheduler import cron_to_apscheduler


def test_cron_passthrough() -> None:
  '''Daily crons convert unchanged.'''
  assert cron_to_apscheduler('15 6 * * *') == ('15', '6', '*', '*', '*')


def test_cron_dow_shift() -> None:
  '''Cron 0=Sunday maps to APScheduler 6; ranges shift too.'''
  assert cron_to_apscheduler('30 3 * * 0') == ('30', '3', '*', '*', '6')
  assert cron_to_apscheduler('0 5 * * 1-5') == ('0', '5', '*', '*', '0-4')
  assert cron_to_apscheduler('* * * * sat,sun')[4] == 'sat,sun'


def test_cron_rejects_malformed() -> None:
  '''Wrong field count raises ValueError.'''
  with pytest.raises(ValueError):
    cron_to_apscheduler('15 6 *')
