#!/usr/bin/env python
# -- coding: utf-8 --

'''Session orchestrator: discovery -> sessions -> toolbox wiring.

`MCPSessionManager` opens MCP sessions for servers picked by discovery.
`ops_policy_engine` builds the write-tool policy for the ops use case.
'''


from __future__ import annotations

from typing import Any, Dict, List

from agentic_common.logging import log_event
from agentic_common.settings import Settings
from chapter01_mcp.client.mcp_client import MCPClient
from chapter01_mcp.client.policy import ToolPolicyEngine
from chapter01_mcp.logging_util import get_mcp_logger
from chapter01_mcp.schemas import ServerDescriptor

logger = get_mcp_logger(__name__)


def ops_policy_engine(settings: Settings) -> Any:
  '''Build the policy engine for the retail/telecom ops use case.

  Write tools (state-mutating) are listed explicitly. The approval callback
  allows writes only within safe operational bounds taken from settings, so
  operators can tighten policy via environment variables.

  Args:
    settings: Runtime settings.

  Returns:
    A configured ToolPolicyEngine.
  '''
  write_tools = {
    'retail-ops.retail_restock_order',
    'telecom-ops.telecom_dispatch_technician',
  }

  def approver(args: Dict[str, Any], qualified: str) -> bool:
    '''Approve write calls inside safe operational bounds.

    Args:
      args: Tool arguments proposed by the LLM.
      qualified: `server.tool` name.

    Returns:
      True when the call is within policy.
    '''
    if qualified == 'retail-ops.retail_restock_order':
      try:
        quantity = int(args.get('quantity', 0))
      except (TypeError, ValueError):
        return False
      return 0 < quantity <= settings.max_restock_quantity

    if qualified == 'telecom-ops.telecom_dispatch_technician':
      priority = str(args.get('priority', ''))
      return priority in settings.allowed_dispatch_priorities

    return False

  approval = approver if settings.require_write_approval else None
  return ToolPolicyEngine(
    write_tools=write_tools,
    max_result_chars=settings.tool_max_result_chars,
    approval_callback=approval,
  )


class MCPSessionManager:
  '''Opens MCP sessions for servers picked by discovery.

  A server that fails to start is skipped (with a warning) rather than
  aborting the whole run - partial capability beats total failure.
  '''

  def __init__(
    self,
    catalog: Any,
    selected: List[str],
    timeout_seconds: float = 15.0,
  ) -> None:
    '''Prepare the session manager.

    Args:
      catalog: ServerCatalog instance.
      selected: Server names picked by discovery.
      timeout_seconds: Per-request timeout.
    '''
    self._catalog = catalog
    self._selected = selected
    self._timeout = timeout_seconds
    self.clients: Dict[str, Any] = {}
    self.tools_by_server: Dict[str, List[Any]] = {}

  def open_all(self) -> None:
    '''Connect to every selected server and discover its tools.'''
    for name in self._selected:
      descriptor: ServerDescriptor | None = self._catalog.get(name)
      if descriptor is None:
        continue

      client = MCPClient(descriptor, timeout_seconds=self._timeout)
      try:
        client.start()
        tools = client.list_tools()
      # Narrowing is impractical: transports raise TransportError,
      # MCPClientError, httpx and subprocess errors.
      except Exception as exc:  # pylint: disable=broad-exception-caught
        log_event(
          logger, 30, 'server_unavailable',
          server=name, error=str(exc),
        )
        client.close()
        continue

      self.clients[name] = client
      self.tools_by_server[name] = tools
      log_event(
        logger, 20, 'tools_discovered',
        server=name, tools=[tool.name for tool in tools],
      )

  def close(self) -> None:
    '''Close all sessions, ignoring individual teardown errors.'''
    for client in self.clients.values():
      try:
        client.close()
      except Exception:  # pylint: disable=broad-exception-caught  # teardown must never raise
        pass
    self.clients.clear()

