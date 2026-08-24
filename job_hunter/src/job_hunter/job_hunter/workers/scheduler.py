#!/usr/bin/env python
# -- coding: utf-8 --

'''APScheduler wiring for discovery, polls, sweeps, and refreshes.'''


from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Tuple

from apscheduler.schedulers.blocking import BlockingScheduler
from job_hunter.core.logging import setup_logging

logger = logging.getLogger(__name__)


def cron_to_apscheduler(expression: str) -> Tuple[int, int, int, int, str]:
  '''Convert a 5-field cron string to APScheduler trigger kwargs.

  Args:
    expression: Standard cron fields minute hour dom month dow.

  Returns:
    (minute, hour, day, month, day_of_week) tuple for CronTrigger.
  '''
  fields = expression.split()
  if len(fields) != 5:
    raise ValueError(f'cron must have 5 fields: {expression!r}')
  minute, hour, day, month, dow = [field.strip() for field in fields]
  converted_dow = '*'
  if dow != '*':
    tokens = []
    for part in dow.split(','):
      token = _shift_dow_token(part)
      tokens.append(token)
    converted_dow = ','.join(tokens)
  return minute, hour, day, month, converted_dow


def _shift_dow_token(token: str) -> str:
  '''Shift one cron day-of-week token to APScheduler numbering.

  Args:
    token: Token like '0', '1-5', or 'mon'.

  Returns:
    Converted token.

  Raises:
    ValueError: For malformed ranges.
  '''
  names = {'sun': 'sun', 'mon': 'mon', 'tue': 'tue', 'wed': 'wed', 'thu': 'thu', 'fri': 'fri', 'sat': 'sat'}
  lowered = token.lower()
  if lowered in names:
    return lowered
  if '-' in lowered:
    start, end = lowered.split('-', 1)
    return f'{(int(start) - 1) % 7}-{(int(end) - 1) % 7}'
  if lowered.isdigit():
    return str((int(lowered) - 1) % 7)
  raise ValueError(f'unsupported day-of-week token: {token}')


def start_scheduler(config_path: str | Path) -> None:
  '''Run the blocking scheduler loop until interrupted.

  Args:
    config_path: app.yaml path.
  '''
  from job_hunter.core.bootstrap import bootstrap
  settings = bootstrap(config_path)
  setup_logging(settings.log_level, settings.data_dir / 'logs')
  schedule = settings.schedule
  scheduler = BlockingScheduler()

  from job_hunter.workers.jobs import (
    enqueue_run,
    execute_pending_run,
    recover_orphans,
    stale_sweep,
  )
  from job_hunter.services.company_discovery import verify_pending

  def run_discovery_now() -> None:
    '''Enqueue then execute a full discovery run.'''
    try:
      run_id = enqueue_run(config_path, triggered_by='scheduler:discovery')
      result = execute_pending_run(config_path, run_id)
      logger.info('discovery run finished: %s', result['status'])
    except RuntimeError as exc:
      logger.info('skipping discovery: %s', exc)

  def run_quick_poll() -> None:
    '''Enqueue then execute a pinned-company quick poll.'''
    try:
      run_id = enqueue_run(config_path, kind='refresh', triggered_by='scheduler:quick_poll')
      asyncio.run(execute_pending_run(config_path, run_id))
    except RuntimeError as exc:
      logger.info('skipping quick poll: %s', exc)

  def run_company_refresh() -> None:
    '''Verify careers pages for companies due.'''
    result = asyncio.run(verify_pending(
      settings,
      chunk=int(settings.discovery.get('company_refresh_chunk', 50)),
    ))
    logger.info('company refresh: %s', result)

  def run_stale_sweep() -> None:
    '''Expire stale postings and recommendations.'''
    count = stale_sweep(config_path)
    logger.info('stale sweep marked %s jobs', count)

  def run_inbox_scan() -> None:
    '''Scan the manual inbox when that feature is present.'''
    try:
      from job_hunter.adapters.manual_inbox import scan_inbox
      count = asyncio.run(scan_inbox(settings))
      if count:
        logger.info('inbox scan imported %s postings', count)
    except ImportError:
      pass

  recover_orphans(config_path)
  minute, hour, day, month, dow = cron_to_apscheduler(schedule.get('discovery_cron', '15 6 * * *'))
  scheduler.add_job(run_discovery_now, 'cron', minute=minute, hour=hour, day=day, month=month, day_of_week=dow, id='discovery')
  scheduler.add_job(run_quick_poll, 'interval', minutes=int(schedule.get('quick_poll_minutes', 120)), id='quick_poll')
  minute, hour, day, month, dow = cron_to_apscheduler(schedule.get('company_refresh_cron', '30 3 * * 0'))
  scheduler.add_job(run_company_refresh, 'cron', minute=minute, hour=hour, day=day, month=month, day_of_week=dow, id='company_refresh')
  minute, hour, day, month, dow = cron_to_apscheduler(schedule.get('stale_sweep_cron', '0 4 * * *'))
  scheduler.add_job(run_stale_sweep, 'cron', minute=minute, hour=hour, day=day, month=month, day_of_week=dow, id='stale_sweep')
  scheduler.add_job(run_inbox_scan, 'interval', minutes=int(schedule.get('inbox_scan_minutes', 30)), id='inbox_scan')

  logger.info('worker scheduler started (config=%s)', config_path)
  try:
    scheduler.start()
  except (KeyboardInterrupt, SystemExit):
    logger.info('worker stopped')


if __name__ == '__main__':
  start_scheduler(Path(__file__).resolve().parents[4] / 'config' / 'app.yaml')
