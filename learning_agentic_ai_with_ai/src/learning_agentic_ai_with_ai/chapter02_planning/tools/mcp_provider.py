#!/usr/bin/env python
# -- coding: utf-8 --

'''MCP tool provider over the Chapter 1 client and servers.

This is the production-shaped backend: the agent talks to real MCP servers
over their declared transport (stdio subprocess by default), reusing
Chapter 1's `ServerCatalog` for discovery and `MCPSessionManager` for
session lifecycle. Tool names are qualified `server.tool`, matching the
in-process provider exactly, so swapping backends is a one-argument change.

Failure handling mirrors Chapter 1's philosophy: a server that fails to
start is skipped with a warning (partial capability beats total failure),
and tool-level errors come back as `ToolResult(ok=False)` instead of
exceptions.
'''


from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from agentic_common.logging import get_logger, log_event
from chapter01_mcp.agent.orchestrator import MCPSessionManager
from chapter01_mcp.client.registry import ServerCatalog, pick_servers
from chapter02_planning.schemas import ToolResult, ToolSpec
from chapter02_planning.tools.in_process import DEFAULT_WRITE_TOOLS
from chapter02_planning.tools.provider import ToolProvider

logger = get_logger(__name__)


class MCPToolProvider(ToolProvider):
  '''Serves tools discovered from live MCP servers.'''

  name = 'mcp'

  def __init__(
    self,
    task: str = '',
    selected_servers: Optional[List[str]] = None,
    catalog: Optional[ServerCatalog] = None,
    timeout_seconds: float = 15.0,
    write_tools: Optional[List[str]] = None,
  ) -> None:
    '''Configure the provider (sessions connect lazily).

    Args:
      task: Task text used for hint-based server selection.
      selected_servers: Explicit server names; overrides `task` selection.
      catalog: Optional catalog override (tests inject fake servers).
      timeout_seconds: Per-request timeout for MCP calls.
      write_tools: Qualified names considered state-changing.
    '''
    self._task = task
    self._selected = selected_servers
    self._catalog = catalog or ServerCatalog()
    self._timeout = timeout_seconds
    self._write_tools = set(write_tools or DEFAULT_WRITE_TOOLS)
    self._manager: Optional[MCPSessionManager] = None
    self._clients: Dict[str, Any] = {}
    self._specs: List[ToolSpec] = []
    self._connected = False

  def list_tools(self) -> List[ToolSpec]:
    '''Connect (once) and list discovered tools.

    Returns:
      Qualified tool specs.
    '''
    self._ensure_connected()
    return list(self._specs)

  def execute(self, tool_name: str, arguments: Dict[str, Any]) -> ToolResult:
    '''Call one MCP tool.

    Args:
      tool_name: Qualified `server.tool` name.
      arguments: Tool arguments (untrusted).

    Returns:
      Normalized ToolResult.
    '''
    self._ensure_connected()
    started = time.perf_counter()
    safe_args = arguments if isinstance(arguments, dict) else {}
    server_name, _, raw_tool = tool_name.partition('.')
    client = self._clients.get(server_name)

    if client is None:
      return ToolResult(
        tool=tool_name, args=safe_args, ok=False,
        text=f'tool error: no MCP session for server {server_name!r}',
        error=f'no MCP session for server {server_name!r}',
        latency_ms=(time.perf_counter() - started) * 1000.0,
      )

    try:
      result = client.call_tool(raw_tool, safe_args)
    except Exception as exc:  # pylint: disable=broad-exception-caught
      log_event(
        logger, 30, 'mcp_tool_failed',
        tool=tool_name, error=str(exc),
      )
      return ToolResult(
        tool=tool_name, args=safe_args, ok=False,
        text=f'tool error: {exc}', error=str(exc),
        latency_ms=(time.perf_counter() - started) * 1000.0,
      )

    return ToolResult(
      tool=tool_name,
      args=safe_args,
      ok=not result.isError,
      text=result.text,
      error=result.text if result.isError else None,
      latency_ms=(time.perf_counter() - started) * 1000.0,
    )

  def close(self) -> None:
    '''Close every MCP session.'''
    if self._manager is not None:
      self._manager.close()
      self._manager = None
      self._clients = {}
      self._connected = False

  # ------------------------------------------------------------------
  # Internals
  # ------------------------------------------------------------------

  def _ensure_connected(self) -> None:
    '''Open sessions and discover tools on first use.'''
    if self._connected:
      return
    self._connected = True

    selected = self._selected
    if selected is None:
      if self._task:
        selected = [
          descriptor.name
          for descriptor in pick_servers(
            self._task, self._catalog, max_servers=2,
          )
        ]
      else:
        selected = [
          descriptor.name
          for descriptor in self._catalog.iter_servers()
          if descriptor.enabled
        ]

    manager = MCPSessionManager(
      catalog=self._catalog,
      selected=selected,
      timeout_seconds=self._timeout,
    )
    manager.open_all()
    self._manager = manager
    self._clients = manager.clients

    specs: List[ToolSpec] = []
    for server_name, tools in manager.tools_by_server.items():
      for descriptor in tools:
        qualified = f'{server_name}.{descriptor.name}'
        specs.append(
          ToolSpec(
            name=qualified,
            description=descriptor.description,
            parameters=descriptor.inputSchema,
            read_only=qualified not in self._write_tools,
            server=server_name,
          )
        )
    self._specs = specs

    if not specs:
      log_event(logger, 30, 'mcp_no_tools', selected=selected)
