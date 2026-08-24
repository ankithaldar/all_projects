#!/usr/bin/env python
# -- coding: utf-8 --

'''Run lifecycle orchestration bridging the runs table and LangGraph.'''


from __future__ import annotations

from contextvars import ContextVar
from typing import Any, Dict, Optional

from job_hunter.core.config import AppSettings
from job_hunter.core.logging import node_var, run_id_var
from job_hunter.db.repositories.runs import RunsRepository

current_run_id: ContextVar[Optional[int]] = ContextVar('current_run_id', default=None)


class RunManager:
  '''Create, claim, observe, and finalize discovery runs.'''

  def __init__(self, settings: AppSettings) -> None:
    '''Initialize with repositories.

    Args:
      settings: Application settings.
    '''
    self._settings = settings
    self._runs = RunsRepository(settings.db_path)

  @property
  def runs(self) -> RunsRepository:
    '''Expose the underlying repository.

    Returns:
      Runs repository.
    '''
    return self._runs

  def enqueue(self, kind: str = 'discovery', triggered_by: str = 'scheduler') -> int:
    '''Insert a pending run unless another is active.

    Args:
      kind: Run kind.
      triggered_by: Origin label.

    Returns:
      New run id.

    Raises:
      RuntimeError: When a run is already active.
    '''
    if self._runs.has_active():
      raise RuntimeError('another run is pending or running')
    run_id = self._runs.create(kind, triggered_by)
    token = current_run_id.set(run_id)
    _ = token
    return run_id

  async def execute(self, run_id: Optional[int] = None) -> Dict[str, Any]:
    '''Claim (specific or oldest pending) and execute to completion.

    Args:
      run_id: Specific pending run id, or None for oldest.

    Returns:
      Final stats mapping.
    '''
    from job_hunter.graph.discovery_graph import run_discovery
    claimed = self._runs.claim_pending(run_id)
    if claimed is None:
      return {'run_id': run_id, 'status': 'skipped', 'stats': {}}
    token = current_run_id.set(claimed)
    run_token = run_id_var.set(str(claimed))
    try:
      final_state = await run_discovery(self._settings, claimed)
      stats = dict(final_state.get('stats') or {})
      errors = list(final_state.get('errors') or [])
      for error in errors:
        self.log_event(claimed, 'error', error.get('node', ''), str(error.get('message')))
      status = 'failed' if any(not e.get('recoverable', True) for e in errors) else (
        'success' if not errors else 'partial'
      )
      self._runs.finish(claimed, status, stats)
      return {'run_id': claimed, 'status': status, 'stats': stats}
    except Exception as exc:
      self._runs.finish(claimed, 'failed', {}, error_text=str(exc)[:500])
      self.log_event(claimed, 'error', 'run_manager', f'run crashed: {exc}')
      return {'run_id': claimed, 'status': 'failed', 'stats': {}}
    finally:
      current_run_id.reset(token)
      run_id_var.reset(run_token)

  def log_event(
    self,
    run_id: int,
    level: str,
    node: str,
    message: str,
    data: Optional[dict] = None,
  ) -> int:
    '''Record a progress event (also drives SSE).

    Args:
      run_id: Run id.
      level: debug|info|warn|error.
      node: Node name.
      message: Text.
      data: Optional payload.

    Returns:
      Event id.
    '''
    node_var.set(node or None)
    return self._runs.log_event(run_id, level, node, message, data)

  def recover_orphans(self, ttl_minutes: int = 120) -> list:
    '''Fail stale non-terminal rows after crashes.

    Args:
      ttl_minutes: Age threshold for orphaned runs.

    Returns:
      Recovered run ids.
    '''
    return self._runs.mark_orphans_failed(ttl_minutes=ttl_minutes)
