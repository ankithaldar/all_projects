#!/usr/bin/env python
# -- coding: utf-8 --

'''Telecom network operations MCP server: site status + dispatch tools.

Tools (telecom_ops_*):
- telecom_site_status      : read  - latest metrics + status for one site
- telecom_degraded_sites   : read  - degraded sites (optional region)
- telecom_dispatch_technician : write - create a dispatch order (policy-checked)
'''


from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List

from chapter01_mcp.logging_util import get_mcp_logger
from chapter01_mcp.ops_models import (
  TelecomDegradedInput,
  TelecomDispatchInput,
  TelecomDispatchOutput,
  TelecomSiteStatusInput,
)
from chapter01_mcp.server_core import MCPServerCore
from chapter01_mcp.servers.ops_db import connect

logger = get_mcp_logger(__name__)


def _handle_site_status(data: TelecomSiteStatusInput) -> Dict[str, Any]:
  '''Return status plus latest-hour metrics for one cell site.

  Args:
    data: Validated input.

    Returns:
      Dict describing the site.

    Raises:
      ValueError: If the site is unknown.
  '''
  conn = connect()
  try:
    site = conn.execute(
      'SELECT site_id, name, region, status, battery_backup_hours '
      'FROM cell_sites WHERE site_id = ?',
      (data.site_id,),
    ).fetchone()
    if site is None:
      raise ValueError(f'unknown site: {data.site_id}')

    metric = conn.execute(
      'SELECT hour, active_users, latency_ms, packet_loss_pct '
      'FROM site_metrics WHERE site_id = ? ORDER BY hour DESC LIMIT 1',
      (data.site_id,),
    ).fetchone()

    last_hour: Dict[str, Any] = {}
    if metric is not None:
      last_hour = {
        'hour': metric['hour'],
        'active_users': metric['active_users'],
        'latency_ms': metric['latency_ms'],
        'packet_loss_pct': metric['packet_loss_pct'],
      }

    return {
      'site_id': site['site_id'],
      'name': site['name'],
      'region': site['region'],
      'status': site['status'],
      'battery_backup_hours': site['battery_backup_hours'],
      'last_hour': last_hour,
    }
  finally:
    conn.close()


def _handle_degraded_sites(data: TelecomDegradedInput) -> Dict[str, Any]:
  '''List sites in degraded state with their latest metrics.

  Args:
    data: Validated input (optional region filter).

    Returns:
      Dict with sites list.
  '''
  conn = connect()
  try:
    query = 'SELECT site_id FROM cell_sites WHERE status = ?'
    params: List[Any] = ['degraded']
    if data.region:
      query += ' AND region = ?'
      params.append(data.region)

    sites: List[Dict[str, Any]] = []
    for row in conn.execute(query, params).fetchall():
      metric = conn.execute(
        'SELECT hour, active_users, latency_ms, packet_loss_pct '
        'FROM site_metrics WHERE site_id = ? ORDER BY hour DESC LIMIT 1',
        (row['site_id'],),
      ).fetchone()
      sites_entry: Dict[str, Any] = {
        'site_id': row['site_id'],
        'latency_ms': metric['latency_ms'] if metric else None,
        'packet_loss_pct': metric['packet_loss_pct'] if metric else None,
        'active_users': metric['active_users'] if metric else 0,
        'hour': metric['hour'] if metric else None,
      }
      sites.append(sites_entry)

    return {'sites': sites}
  finally:
    conn.close()


def _handle_dispatch(data: TelecomDispatchInput) -> Dict[str, Any]:
  '''Create a dispatch order after validating site and technician duty.

  Args:
    data: Validated input.

    Returns:
      Dispatch payload dict.

    Raises:
      ValueError: On unknown site/technician or off-duty technician.
  '''
  conn = connect()
  try:
    site = conn.execute(
      'SELECT site_id FROM cell_sites WHERE site_id = ?', (data.site_id,)
    ).fetchone()
    if site is None:
      raise ValueError(f'unknown site: {data.site_id}')

    tech = conn.execute(
      'SELECT tech_id, on_duty FROM technicians WHERE tech_id = ?',
      (data.tech_id,),
    ).fetchone()
    if tech is None:
      raise ValueError(f'unknown technician: {data.tech_id}')
    if not tech['on_duty']:
      raise ValueError(f'technician {data.tech_id} is off duty')

    now_iso = datetime.now(timezone.utc).isoformat()
    cursor = conn.execute(
      'INSERT INTO dispatch_orders '
      '(site_id, tech_id, priority, note, created_at, status) '
      'VALUES (?,?,?,?,?,?)',
      (
        data.site_id, data.tech_id, data.priority, data.note,
        now_iso, 'dispatched',
      ),
    )
    conn.commit()

    return TelecomDispatchOutput(
      dispatch_id=int(cursor.lastrowid or 0),
      site_id=data.site_id,
      tech_id=data.tech_id,
      priority=data.priority,
      status='dispatched',
      created_at=now_iso,
    ).model_dump()
  finally:
    conn.close()


def build_telecom_server():
  '''Build the telecom operations MCP server core with tools registered.

  Returns:
    Configured MCPServerCore.
  '''
  server = MCPServerCore(name='telecom-ops', version='1.0.0')

  server.register_tool(
    name='telecom_site_status',
    description=(
      'Return current status and the latest hourly metrics (active users, '
      'latency, packet loss) for one cell site.'
    ),
    input_model=TelecomSiteStatusInput,
    handler=_handle_site_status,
  )

  server.register_tool(
    name='telecom_degraded_sites',
    description=(
      'List cell sites currently degraded (high latency or packet loss), '
      'optionally filtered by region.'
    ),
    input_model=TelecomDegradedInput,
    handler=_handle_degraded_sites,
  )

  server.register_tool(
    name='telecom_dispatch_technician',
    description=(
      'Dispatch a field technician to a cell site. WRITE tool: requires an '
      'on-duty technician id and a priority (low/medium/high).'
    ),
    input_model=TelecomDispatchInput,
    handler=_handle_dispatch,
  )

  return server
