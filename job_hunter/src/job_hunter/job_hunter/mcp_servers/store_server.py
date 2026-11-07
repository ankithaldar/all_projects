#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''MCP server exposing safe, allow-listed database tools.

Run: python -m job_hunter.mcp_servers.store_server
'''


from __future__ import annotations

try:
  from mcp.server.fastmcp import FastMCP
except ModuleNotFoundError:  # mcp 2.x renamed FastMCP to MCPServer
  from mcp.server.mcpserver import MCPServer as FastMCP
from job_hunter.mcp_servers.common import dumps, ensure_sys_path
from job_hunter.mcp_servers.common import (
  get_settings_obj as _settings_obj,
)

ensure_sys_path()

from job_hunter.adapters.base import valid_domain  # noqa: E402
# bootstrap kept as legacy monkeypatch target (common.get_settings_obj
# honours a patched store_server.bootstrap).
from job_hunter.core.bootstrap import bootstrap  # noqa: E402,F401
from job_hunter.core.config import AppSettings  # noqa: E402
from job_hunter.db.repositories.companies import CompaniesRepository  # noqa: E402
from job_hunter.db.repositories.jobs import JobsRepository  # noqa: E402
from job_hunter.db.repositories.profile import ProfileRepository  # noqa: E402
from job_hunter.db.repositories.recommendations import (  # noqa: E402
  RecommendationsRepository,
)
from job_hunter.db.repositories.runs import RunsRepository  # noqa: E402

mcp = FastMCP('job_hunter-store')
# Legacy hook kept for existing tests that monkeypatch store_server._settings
# directly (common.get_settings_obj honours it). The shared cache lives in
# common.py; this variable is not the cache.
_settings: AppSettings | None = None


@mcp.tool()
def get_profile(candidate_id: int = 1) -> str:
  '''Fetch the current candidate profile as JSON.

  Args:
    candidate_id: Candidate id.

  Returns:
    JSON profile or error object.
  '''
  data = ProfileRepository(_settings_obj().db_path).get_profile(candidate_id)
  return dumps(data or {'error': 'no profile yet'})


@mcp.tool()
def list_companies(status: str = 'active', limit: int = 100) -> str:
  '''List companies by status.

  Args:
    status: Lifecycle status filter.
    limit: Max rows.

  Returns:
    JSON array of companies.
  '''
  return dumps(CompaniesRepository(_settings_obj().db_path).list_companies(
    status=status or None, limit=min(limit, 500),
  ))


@mcp.tool()
def upsert_company(name: str, domain: str = '', vertical: str = '') -> str:
  '''Insert or update a target company.

  Args:
    name: Company display name.
    domain: Registrable domain when known.
    vertical: Vertical label when known.

  Returns:
    JSON with company_id, or an error object for an invalid domain.
  '''
  try:
    domain = valid_domain(domain)
  except ValueError as exc:
    return dumps({'error': str(exc)})
  company_id = CompaniesRepository(_settings_obj().db_path).upsert(
    name=name,
    domain=domain,
    vertical=vertical or None,
    discovered_via='mcp',
  )
  return dumps({'company_id': company_id})


@mcp.tool()
def jobs_recent(limit: int = 25, city: str = '', work_mode: str = '') -> str:
  '''List the most recent active jobs.

  Args:
    limit: Page size cap.
    city: Optional exact-city filter.
    work_mode: Optional work-mode filter.

  Returns:
    JSON array of jobs.
  '''
  return dumps(JobsRepository(_settings_obj().db_path).list_jobs(
    city=city, work_mode=work_mode, per_page=min(limit, 200),
  ))


@mcp.tool()
def recommendations_top(limit: int = 10) -> str:
  '''Return top-scoring gate-passing recommendations.

  Args:
    limit: Max rows.

  Returns:
    JSON array of recommendation rows.
  '''
  return dumps(
    RecommendationsRepository(_settings_obj().db_path)
    .list_for_candidate(per_page=min(limit, 100))
  )


@mcp.tool()
def record_run_event(
  run_id: int,
  level: str,
  node: str,
  message: str,
) -> str:
  '''Append an event to a run's audit stream.

  Args:
    run_id: Run id.
    level: debug|info|warn|error.
    node: Component name.
    message: Event text.

  Returns:
    JSON with event id.
  '''
  event_id = RunsRepository(_settings_obj().db_path).log_event(
    run_id, level, node, message,
  )
  return dumps({'event_id': event_id})


def main() -> None:
  '''Serve the store server over stdio.'''
  _settings_obj()
  mcp.run()


if __name__ == '__main__':
  main()
