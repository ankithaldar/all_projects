#!/usr/bin/env python
# -- coding: utf-8 --

'''In-process tool provider over the Chapter 1 server cores.

Purpose: run the exact same tool implementations Chapter 1 uses (same
pydantic argument models, same JSON Schema, same error semantics) without
spawning MCP subprocesses. This is the adapter that makes "chapter01_mcp
code works along with llm_gateway" literal: `build_retail_server()` and
`build_telecom_server()` are imported unchanged and dispatched through
`MCPServerCore.handle_message`, exactly as the stdio transport does.

Trade-offs (used intentionally in tests and offline demos):
  - fastest path: no process boundary, no protocol serialization on a pipe,
  - same validation behavior as MCP because the same core handles the call,
  - does not exercise transport code (use MCPToolProvider for that).
'''


from __future__ import annotations

import itertools
import time
from typing import Any, Dict, List, Optional

from chapter01_mcp.mcp_protocol import METHOD_TOOLS_CALL
from chapter01_mcp.schemas import JsonRpcRequest, ToolCallResult
from chapter01_mcp.server_core import MCPServerCore
from chapter01_mcp.servers.retail_server import build_retail_server
from chapter01_mcp.servers.telecom_server import build_telecom_server
from chapter02_planning.schemas import ToolResult, ToolSpec
from chapter02_planning.tools.provider import ToolProvider

DEFAULT_WRITE_TOOLS = frozenset({
  'retail-ops.retail_restock_order',
  'telecom-ops.telecom_dispatch_technician',
})


class InProcessToolProvider(ToolProvider):
  '''Serves Chapter 1 retail/telecom tools in the current process.'''

  name = 'inproc'

  def __init__(
    self,
    servers: Optional[Dict[str, MCPServerCore]] = None,
    write_tools: Optional[List[str]] = None,
  ) -> None:
    '''Initialize the provider and discover tools.

    Args:
      servers: Optional mapping of server name to server core (tests can
        inject fakes). Defaults to the Chapter 1 retail and telecom cores.
      write_tools: Qualified names considered state-changing. Defaults to the
        two Chapter 1 write tools.
    '''
    self._servers: Dict[str, MCPServerCore] = servers or {
      'retail-ops': build_retail_server(),
      'telecom-ops': build_telecom_server(),
    }
    self._write_tools = set(write_tools or DEFAULT_WRITE_TOOLS)
    self._ids = itertools.count(1)
    self._tool_names: Dict[str, set[str]] = {}
    self._specs = self._discover()

  def _discover(self) -> List[ToolSpec]:
    '''Build qualified tool specs from the server cores.

    Returns:
      List of ToolSpec.
    '''
    specs: List[ToolSpec] = []
    for server_name, core in self._servers.items():
      names: set[str] = set()
      for descriptor in core.tool_descriptors():
        names.add(descriptor.name)
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
      self._tool_names[server_name] = names
    return specs

  def list_tools(self) -> List[ToolSpec]:  # noqa: D102
    return list(self._specs)

  def execute(self, tool_name: str, arguments: Dict[str, Any]) -> ToolResult:
    '''Dispatch one call through the matching server core.

    Args:
      tool_name: Qualified `server.tool` name.
      arguments: Tool arguments (untrusted).

    Returns:
      Normalized ToolResult.
    '''
    started = time.perf_counter()
    safe_args = arguments if isinstance(arguments, dict) else {}
    server_name, _, raw_tool = tool_name.partition('.')
    core = self._servers.get(server_name)

    if core is None or raw_tool not in self._tool_names.get(server_name, set()):
      return ToolResult(
        tool=tool_name,
        args=safe_args,
        ok=False,
        text=f'tool error: unknown tool {tool_name}',
        error=f'unknown tool {tool_name}',
        latency_ms=(time.perf_counter() - started) * 1000.0,
      )

    request = JsonRpcRequest(
      id=next(self._ids),
      method=METHOD_TOOLS_CALL,
      params={'name': raw_tool, 'arguments': safe_args},
    )
    response = core.handle_message(request)
    latency_ms = (time.perf_counter() - started) * 1000.0

    if response is None:
      return ToolResult(
        tool=tool_name, args=safe_args, ok=False,
        text='tool error: no response from server core',
        error='no response from server core',
        latency_ms=latency_ms,
      )

    error = response.get('error')
    if error is not None:
      message = str(error.get('message', 'unknown protocol error'))
      return ToolResult(
        tool=tool_name, args=safe_args, ok=False,
        text=f'tool error: {message}', error=message,
        latency_ms=latency_ms,
      )

    result = ToolCallResult.model_validate(response.get('result') or {})
    return ToolResult(
      tool=tool_name,
      args=safe_args,
      ok=not result.isError,
      text=result.text,
      error=result.text if result.isError else None,
      latency_ms=latency_ms,
    )

  def close(self) -> None:
    '''No resources to release (handlers own their connections).'''
