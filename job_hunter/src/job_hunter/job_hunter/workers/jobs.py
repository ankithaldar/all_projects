#!/usr/bin/env python
# -- coding: utf-8 --

'''Worker jobs: enqueue and execute discovery runs on demand or schedule.'''


from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict, Optional

from job_hunter.core.bootstrap import bootstrap
from job_hunter.core.config import AppSettings


def _settings(config_path: str | Path) -> AppSettings:
  '''Load settings with data dirs prepared.

  Args:
    config_path: app.yaml path.

  Returns:
    Bootstrapped settings.
  '''
  return bootstrap(config_path)


def enqueue_run(
  config_path: str | Path,
  kind: str = 'discovery',
  triggered_by: str = 'scheduler',
) -> int:
  '''Create a pending run row.

  Args:
    config_path: app.yaml path.
    kind: Run kind.
    triggered_by: Origin label.

  Returns:
    New run id.

  Raises:
    RuntimeError: When a run is already active.
  '''
  from job_hunter.services.run_manager import RunManager
  return RunManager(_settings(config_path)).enqueue(kind, triggered_by)


def execute_pending_run(
  config_path: str | Path,
  run_id: Optional[int] = None,
) -> Dict[str, Any]:
  '''Claim (specific or oldest pending) run and drive it to completion.

  Args:
    config_path: app.yaml path.
    run_id: Explicit run id when given.

  Returns:
    Result mapping {run_id, status, stats}.
  '''
  from job_hunter.services.run_manager import RunManager
  manager = RunManager(_settings(config_path))
  if run_id is None and not manager.runs.has_pending():
    return {'run_id': None, 'status': 'nothing_pending', 'stats': {}}
  return asyncio.run(manager.execute(run_id))


def recover_orphans(config_path: str | Path) -> list:
  '''Fail stale running rows from crashed workers.

  Args:
    config_path: app.yaml path.

  Returns:
    Recovered run ids.
  '''
  from job_hunter.services.run_manager import RunManager
  return RunManager(_settings(config_path)).recover_orphans()


def stale_sweep(config_path: str | Path) -> int:
  '''Mark aged postings stale and expire their recommendations.

  Args:
    config_path: app.yaml path.

  Returns:
    Jobs marked stale.
  '''
  settings = _settings(config_path)
  from job_hunter.db.repositories.jobs import JobsRepository
  days = int(settings.discovery.get('stale_after_days', 45))
  return JobsRepository(settings.db_path).stale_sweep(days)
