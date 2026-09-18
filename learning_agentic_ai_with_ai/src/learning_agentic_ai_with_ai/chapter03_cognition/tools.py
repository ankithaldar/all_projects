#!/usr/bin/env python
# -- coding: utf-8 --

'''Tool access for the Action layer, built directly on Chapter 1.

Chapter 3 keeps its decision layer standalone, but tools are shared
infrastructure: this module imports Chapter 1's server cores (in-process)
and client/registry (MCP stdio) so both chapters expose the same retail and
telecom tools with the same validation. Two backends, one interface:

    ToolBox(backend='inproc')  -> MCPServerCore.handle_message, no subprocess
    ToolBox(backend='mcp')     -> ServerCatalog + MCPSessionManager + MCPClient

Every call passes through one gate that enforces least privilege, optional
approval, observation sanitization (untrusted-data wrapping), and audit
emission - the same controls as Chapters 1 and 2, owned by this chapter.

`FaultInjector` wraps any toolbox to simulate transient failures. It exists
so the adaptive loop (retry / tool switching) can be demonstrated and tested
deterministically without breaking real servers.
'''


from __future__ import annotations

import itertools
import time
from typing import Any, Callable, Dict, List, Optional, Set

from agentic_common.logging import get_logger, log_event
from agentic_common.security import sanitize_untrusted
from agentic_common.tracing import Tracer
from chapter01_mcp.agent.orchestrator import MCPSessionManager
from chapter01_mcp.client.registry import ServerCatalog, pick_servers
from chapter01_mcp.mcp_protocol import METHOD_TOOLS_CALL
from chapter01_mcp.schemas import JsonRpcRequest, ToolCallResult
from chapter01_mcp.server_core import MCPServerCore
from chapter01_mcp.servers.retail_server import build_retail_server
from chapter01_mcp.servers.telecom_server import build_telecom_server
from chapter03_cognition.schemas import (
  ToolExecutionAudit,
  ToolResult,
  ToolSpec,
)

logger = get_logger(__name__)

DEFAULT_WRITE_TOOLS = frozenset({
  'retail-ops.retail_restock_order',
  'telecom-ops.telecom_dispatch_technician',
})

UNTRUSTED_OPEN = '[BEGIN UNTRUSTED TOOL DATA]'
UNTRUSTED_CLOSE = '[END UNTRUSTED TOOL DATA]'

Approver = Callable[[str, Dict[str, Any]], bool]
AuditSink = Callable[[ToolExecutionAudit], None]


def wrap_untrusted(text: str) -> str:
  '''Wrap tool output in explicit untrusted-data markers.

  Args:
    text: Sanitized tool output.

  Returns:
    Wrapped text with delimiters.
  '''
  return f'{UNTRUSTED_OPEN}\n{text}\n{UNTRUSTED_CLOSE}'


class ToolBox:
  '''Chapter 1 retail/telecom tools with a single safety gate.'''

  def __init__(
    self,
    backend: str = 'inproc',
    task: str = '',
    selected_servers: Optional[List[str]] = None,
    allow_writes: bool = False,
    require_approval: bool = True,
    approver: Optional[Approver] = None,
    max_result_chars: int = 4000,
    sanitize: bool = True,
    audit_sink: Optional[AuditSink] = None,
    tracer: Optional[Tracer] = None,
    trace_id: str = '',
    timeout_seconds: float = 15.0,
  ) -> None:
    '''Configure the toolbox (MCP sessions connect lazily).

    Args:
      backend: 'inproc' or 'mcp'.
      task: Task text for MCP hint-based server selection.
      selected_servers: Explicit MCP server names.
      allow_writes: Whether state-changing tools may execute.
      require_approval: Whether writes additionally need `approver`.
      approver: Optional callback `(tool, args) -> bool`.
      max_result_chars: Observation length cap.
      sanitize: Whether to sanitize and wrap observations.
      audit_sink: Optional callback receiving each audit record.
      tracer: Optional tracer for `tool.call` spans.
      trace_id: Trace id for span correlation.
      timeout_seconds: MCP per-request timeout.
    '''
    self._backend = backend
    self._task = task
    self._selected = selected_servers
    self._allow_writes = allow_writes
    self._require_approval = require_approval
    self._approver = approver
    self._max_result_chars = max(200, int(max_result_chars))
    self._sanitize = sanitize
    self._audit_sink = audit_sink
    self._tracer = tracer
    self._trace_id = trace_id
    self._timeout = timeout_seconds

    self._servers: Dict[str, MCPServerCore] = {}
    self._tool_names: Dict[str, Set[str]] = {}
    self._clients: Dict[str, Any] = {}
    self._manager: Optional[MCPSessionManager] = None
    self._ids = itertools.count(1)
    self._specs: List[ToolSpec] = []
    self._connected = False
    self._connect()

  # ------------------------------------------------------------------
  # Public API
  # ------------------------------------------------------------------

  def list_tools(self) -> List[ToolSpec]:
    '''List available tools.

    Returns:
      Qualified tool specs.
    '''
    return list(self._specs)

  def allowed_names(self) -> Set[str]:
    '''Names of tools callable right now.

    Returns:
      Tool name set (write tools excluded when writes are disabled).
    '''
    return {
      spec.name for spec in self._specs
      if spec.read_only or self._allow_writes
    }

  def write_tools(self) -> Set[str]:
    '''Names of state-changing tools.

    Returns:
      Write tool name set.
    '''
    return {
      spec.name for spec in self._specs if not spec.read_only
    }

  def call(self, tool_name: str, arguments: Dict[str, Any]) -> ToolResult:
    '''Gate, execute, sanitize, and audit one tool call.

    Args:
      tool_name: Qualified `server.tool` name.
      arguments: Tool arguments (untrusted).

    Returns:
      Normalized ToolResult; transport failures never raise.
    '''
    started = time.perf_counter()
    safe_args = arguments if isinstance(arguments, dict) else {}
    spec = self._spec(tool_name)

    if spec is None:
      result = ToolResult(
        tool=tool_name, args=safe_args, ok=False,
        text=f'BLOCKED by safety gate: unknown tool {tool_name!r}',
        error=f'unknown tool {tool_name!r}',
      )
      self._record(result, spec=None)
      return result

    if not spec.read_only and not self._allow_writes:
      result = ToolResult(
        tool=tool_name, args=safe_args, ok=False,
        text=(
          'BLOCKED by safety gate: state-changing tools are disabled '
          'for this run (set allow_writes=True to enable)'
        ),
        error='writes disabled for this run',
      )
      self._record(result, spec)
      return result

    if (
      not spec.read_only
      and self._require_approval
      and self._approver is not None
      and not self._approver(tool_name, safe_args)
    ):
      result = ToolResult(
        tool=tool_name, args=safe_args, ok=False,
        text='BLOCKED by safety gate: write call was not approved',
        error='write call was not approved',
      )
      self._record(result, spec)
      return result

    result = self._execute(tool_name, safe_args)
    result.latency_ms = result.latency_ms or (
      (time.perf_counter() - started) * 1000.0
    )
    if self._sanitize and result.text:
      sanitized = sanitize_untrusted(result.text, self._max_result_chars)
      result.text = wrap_untrusted(sanitized)
    elif len(result.text) > self._max_result_chars:
      result.text = result.text[: self._max_result_chars - 3] + '...'

    self._record(result, spec)
    return result

  def close(self) -> None:
    '''Release MCP sessions (in-process backends need nothing).'''
    if self._manager is not None:
      self._manager.close()
      self._manager = None
      self._clients = {}
      self._connected = False

  def set_trace(self, tracer: Optional[Tracer], trace_id: str) -> None:
    '''Attach (or clear) tracing for tool.call spans.

    Args:
      tracer: Tracer instance or None.
      trace_id: Trace id for span correlation.
    '''
    self._tracer = tracer
    self._trace_id = trace_id

  # ------------------------------------------------------------------
  # Backend connection and execution
  # ------------------------------------------------------------------

  def _connect(self) -> None:
    '''Discover tools from the selected backend.

    Raises:
      ValueError: On an unknown backend name.
    '''
    if self._connected:
      return
    self._connected = True
    if self._backend == 'inproc':
      self._connect_inproc()
    elif self._backend == 'mcp':
      self._connect_mcp()
    else:
      raise ValueError(
        f'unknown tool backend {self._backend!r}; use inproc or mcp'
      )
    log_event(
      logger, 20, 'toolbox_ready',
      backend=self._backend, tools=len(self._specs),
    )

  def _connect_inproc(self) -> None:
    '''Discover tools from Chapter 1 server cores.'''
    self._servers = {
      'retail-ops': build_retail_server(),
      'telecom-ops': build_telecom_server(),
    }
    specs: List[ToolSpec] = []
    for server_name, core in self._servers.items():
      names: Set[str] = set()
      for descriptor in core.tool_descriptors():
        names.add(descriptor.name)
        qualified = f'{server_name}.{descriptor.name}'
        specs.append(
          ToolSpec(
            name=qualified,
            description=descriptor.description,
            parameters=descriptor.inputSchema,
            read_only=qualified not in DEFAULT_WRITE_TOOLS,
            server=server_name,
          )
        )
      self._tool_names[server_name] = names
    self._specs = specs

  def _connect_mcp(self) -> None:
    '''Discover tools from live MCP servers (lazy session open).'''
    catalog = ServerCatalog()
    selected = self._selected
    if selected is None:
      if self._task:
        selected = [
          descriptor.name for descriptor in pick_servers(
            self._task, catalog, max_servers=2,
          )
        ]
      else:
        selected = [
          descriptor.name for descriptor in catalog.iter_servers()
          if descriptor.enabled
        ]

    manager = MCPSessionManager(
      catalog=catalog, selected=selected, timeout_seconds=self._timeout,
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
            read_only=qualified not in DEFAULT_WRITE_TOOLS,
            server=server_name,
          )
        )
    self._specs = specs

  def _execute(
    self,
    tool_name: str,
    arguments: Dict[str, Any],
  ) -> ToolResult:
    '''Dispatch a call to the active backend.

    Args:
      tool_name: Qualified tool name.
      arguments: Tool arguments.

    Returns:
      Normalized ToolResult.
    '''
    if self._backend == 'mcp':
      return self._execute_mcp(tool_name, arguments)
    return self._execute_inproc(tool_name, arguments)

  def _execute_inproc(
    self,
    tool_name: str,
    arguments: Dict[str, Any],
  ) -> ToolResult:
    '''Dispatch through a Chapter 1 server core.

    Args:
      tool_name: Qualified tool name.
      arguments: Tool arguments.

    Returns:
      Normalized ToolResult.
    '''
    started = time.perf_counter()
    server_name, _, raw_tool = tool_name.partition('.')
    core = self._servers.get(server_name)
    if core is None or raw_tool not in self._tool_names.get(
      server_name, set(),
    ):
      return ToolResult(
        tool=tool_name, args=arguments, ok=False,
        text=f'tool error: unknown tool {tool_name}',
        error=f'unknown tool {tool_name}',
        latency_ms=(time.perf_counter() - started) * 1000.0,
      )

    response = core.handle_message(
      JsonRpcRequest(
        id=next(self._ids),
        method=METHOD_TOOLS_CALL,
        params={'name': raw_tool, 'arguments': arguments},
      )
    )
    latency_ms = (time.perf_counter() - started) * 1000.0
    if response is None:
      return ToolResult(
        tool=tool_name, args=arguments, ok=False,
        text='tool error: no response from server core',
        error='no response from server core', latency_ms=latency_ms,
      )

    error = response.get('error')
    if error is not None:
      message = str(error.get('message', 'unknown protocol error'))
      return ToolResult(
        tool=tool_name, args=arguments, ok=False,
        text=f'tool error: {message}', error=message,
        latency_ms=latency_ms,
      )

    result = ToolCallResult.model_validate(response.get('result') or {})
    return ToolResult(
      tool=tool_name, args=arguments, ok=not result.isError,
      text=result.text,
      error=result.text if result.isError else None,
      latency_ms=latency_ms,
    )

  def _execute_mcp(
    self,
    tool_name: str,
    arguments: Dict[str, Any],
  ) -> ToolResult:
    '''Dispatch through an MCP client session.

    Args:
      tool_name: Qualified tool name.
      arguments: Tool arguments.

    Returns:
      Normalized ToolResult.
    '''
    started = time.perf_counter()
    server_name, _, raw_tool = tool_name.partition('.')
    client = self._clients.get(server_name)
    if client is None:
      return ToolResult(
        tool=tool_name, args=arguments, ok=False,
        text=f'tool error: no MCP session for {server_name!r}',
        error=f'no MCP session for {server_name!r}',
        latency_ms=(time.perf_counter() - started) * 1000.0,
      )
    try:
      result = client.call_tool(raw_tool, arguments)
    except Exception as exc:  # pylint: disable=broad-exception-caught
      log_event(logger, 30, 'mcp_tool_failed',
                tool=tool_name, error=str(exc))
      return ToolResult(
        tool=tool_name, args=arguments, ok=False,
        text=f'tool error: {exc}', error=str(exc),
        latency_ms=(time.perf_counter() - started) * 1000.0,
      )
    return ToolResult(
      tool=tool_name, args=arguments, ok=not result.isError,
      text=result.text,
      error=result.text if result.isError else None,
      latency_ms=(time.perf_counter() - started) * 1000.0,
    )

  # ------------------------------------------------------------------
  # Internals
  # ------------------------------------------------------------------

  def _spec(self, tool_name: str) -> Optional[ToolSpec]:
    '''Look up a tool spec.

    Args:
      tool_name: Qualified tool name.

    Returns:
      Tool spec or None.
    '''
    for spec in self._specs:
      if spec.name == tool_name:
        return spec
    return None

  def _record(
    self,
    result: ToolResult,
    spec: Optional[ToolSpec],
  ) -> None:
    '''Publish the audit record for one call.

    Args:
      result: Tool result (possibly blocked).
      spec: Tool spec when known.
    '''
    blocked = (
      not result.ok and result.text.startswith('BLOCKED by safety gate')
    )
    audit = ToolExecutionAudit(
      tool=result.tool,
      args=result.args,
      ok=result.ok,
      approved=not blocked,
      blocked=blocked,
      error=result.error,
      latency_ms=result.latency_ms,
      result_preview=sanitize_untrusted(result.text, 200),
    )
    log_event(
      logger, 20, 'tool_call',
      tool=result.tool,
      read_only=spec.read_only if spec else None,
      ok=result.ok,
      blocked=blocked,
      latency_ms=round(result.latency_ms, 2),
      error=result.error,
    )
    if self._audit_sink is not None:
      try:
        self._audit_sink(audit)
      except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.warning(
          'audit sink failed',
          extra={'extra_fields': {'error': str(exc)}},
        )
    if self._tracer is not None and self._trace_id:
      with self._tracer.span(
        self._trace_id, 'tool.call', tool=result.tool, blocked=blocked,
      ) as span:
        span.set_attr('ok', result.ok)
        span.set_attr('latency_ms', round(result.latency_ms, 1))


class FaultInjector:
  '''Toolbox decorator that fails selected tools a fixed number of times.

  Purpose: exercise the adaptive loop (retry, tool switching) against real
  servers without breaking them. Injected failures are marked
  `error='injected transient failure'` and count down per tool.
  '''

  def __init__(
    self,
    inner: ToolBox,
    failures: Dict[str, int],
    error: str = 'injected transient failure',
  ) -> None:
    '''Initialize the injector.

    Args:
      inner: Wrapped toolbox.
      failures: Map of tool name to remaining injected failures.
      error: Error message for injected failures.
    '''
    self._inner = inner
    self._failures = dict(failures)
    self._error = error

  def list_tools(self) -> List[ToolSpec]:  # noqa: D102
    return self._inner.list_tools()

  def allowed_names(self) -> Set[str]:  # noqa: D102
    return self._inner.allowed_names()

  def write_tools(self) -> Set[str]:  # noqa: D102
    return self._inner.write_tools()

  def call(self, tool_name: str, arguments: Dict[str, Any]) -> ToolResult:
    '''Inject a failure while the counter for the tool is positive.

    Args:
      tool_name: Qualified tool name.
      arguments: Tool arguments.

    Returns:
      Injected failure result or the real result.
    '''
    remaining = self._failures.get(tool_name, 0)
    if remaining > 0:
      self._failures[tool_name] = remaining - 1
      log_event(
        logger, 30, 'fault_injected',
        tool=tool_name, remaining=remaining - 1,
      )
      return ToolResult(
        tool=tool_name,
        args=arguments,
        ok=False,
        text=f'tool error: {self._error}',
        error=self._error,
        latency_ms=0.0,
      )
    return self._inner.call(tool_name, arguments)

  def close(self) -> None:  # noqa: D102
    self._inner.close()
