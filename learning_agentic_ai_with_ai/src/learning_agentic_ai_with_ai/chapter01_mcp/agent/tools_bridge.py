#!/usr/bin/env python
# -- coding: utf-8 --

'''Bridge between the LLM gateway's tool-calling format and MCP.

The gateway speaks OpenAI-style tools:
    {"type":"function","function":{"name","description","parameters"}}
MCP speaks ToolDescriptor: {"name","description","inputSchema"}.

This module converts between them and routes gateway tool_calls back into
MCP servers, with every hop recorded (trace span + audit + store).
'''


from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Tuple

from agentic_common.logging import log_event
from agentic_common.tracing import TokenUsage
from chapter01_mcp.client.mcp_client import MCPClient
from chapter01_mcp.logging_util import get_mcp_logger
from chapter01_mcp.schemas import ToolCallAudit, ToolCallResult, ToolDescriptor

logger = get_mcp_logger(__name__)


class MCPToolbox:
  '''Owns MCP sessions, exposes tools to the LLM, and executes calls.

  Responsibilities:
  - expose one merged tool list (namespace-qualified names) to the gateway,
  - route a gateway tool call back to the right MCP server,
  - enforce policy (schema/limits/approval) BEFORE the server is hit,
  - record audit rows and tracing spans for every call.
  '''

  def __init__(
    self,
    clients: Dict[str, MCPClient],
    tools_by_server: Dict[str, List[ToolDescriptor]],
    policy_engine: Any,
    store: Any = None,
    tracer: Any = None,
    trace_id: str = '',
  ) -> None:
    '''Build the bridge from ready-to-use clients.

    Args:
      clients: Mapping of server name to connected MCPClient.
      tools_by_server: Server name -> discovered ToolDescriptor list.
      policy_engine: ToolPolicyEngine gate.
      store: Optional AgentStore for audits.
      tracer: Optional Tracer for spans.
      trace_id: Trace id for span correlation.
    '''
    self._clients = clients
    self._policy = policy_engine
    self._store = store
    self._tracer = tracer
    self._trace_id = trace_id
    # qualified name -> (server, descriptor)
    self._descriptors: Dict[str, Tuple[str, ToolDescriptor]] = {}
    for server_name, tools in tools_by_server.items():
      for descriptor in tools:
        qualified = f'{server_name}.{descriptor.name}'
        self._descriptors[qualified] = (server_name, descriptor)

  # ------------------------------------------------------------------
  # LLM-facing view
  # ------------------------------------------------------------------

  def to_gateway_tools(self) -> List[Dict[str, Any]]:
    '''Convert discovered descriptors into OpenAI-style tool definitions.

    Names are qualified `server.tool` because different servers may expose
    identical tool names; the LLM must disambiguate.

    Returns:
      Gateway tool definitions list.
    '''
    gateway_tools: List[Dict[str, Any]] = []

    for qualified, entry in self._descriptors.items():
      descriptor = entry[1]
      gateway_tools.append(
        {
          'type': 'function',
          'function': {
            'name': qualified,
            'description': descriptor.description or f'Tool {descriptor.name}',
            'parameters': descriptor.inputSchema
            or {'type': 'object', 'properties': {}},
          },
        }
      )

    return gateway_tools

  def close(self) -> None:
    '''Close every client session.'''
    for client in self._clients.values():
      try:
        client.close()
      except Exception:  # pylint: disable=broad-exception-caught
        pass
    self._clients.clear()

  # ------------------------------------------------------------------
  # Execution
  # ------------------------------------------------------------------

  def execute(
    self,
    qualified_name: str,
    arguments: Dict[str, Any],
    session_id: str,
  ) -> Tuple[bool, str, ToolCallAudit]:
    '''Policy-check and execute one tool call via MCP.

    Args:
      qualified_name: `server.tool` name emitted by the LLM.
      arguments: Raw (untrusted) arguments.
      session_id: Session id for auditing.

    Returns:
      (ok, text_for_llm, audit) triple.
    '''
    started = time.perf_counter()

    parts = qualified_name.split('.', 1)
    if len(parts) != 2 or parts[0] not in self._clients:
      audit = self._rejected_audit(
        qualified_name, arguments, started,
        f'unknown tool or server: {qualified_name}',
      )
      return False, f'unknown tool: {qualified_name}', audit

    server_name, tool_name = parts
    entry = self._descriptors.get(qualified_name)
    if entry is None:
      audit = self._rejected_audit(
        qualified_name, arguments, started, 'unknown tool',
      )
      return False, f'unknown tool: {qualified_name}', audit

    descriptor = entry[1]

    # ---- Policy gate (schema, limits, allowed values, approval) ----
    allowed, reason, approved = self._policy.check(
      server_name, tool_name, arguments, descriptor,
    )
    if not allowed:
      audit = ToolCallAudit(
        server=server_name, tool=tool_name, args=arguments, ok=False,
        approved=approved, error=reason,
        latency_ms=(time.perf_counter() - started) * 1000.0,
      )
      self._record_audit(session_id, audit)
      return False, f'BLOCKED by policy: {reason}', audit

    # ---- MCP call with tracing ----
    result = self._invoke(server_name, tool_name, qualified_name, arguments)
    if isinstance(result, ToolCallAudit):
      self._record_audit(session_id, result)
      return False, f'tool error: {result.error}', result

    text = self._policy.sanitize_result(result)
    audit = ToolCallAudit(
      server=server_name,
      tool=tool_name,
      args=arguments,
      ok=not result.isError,
      approved=approved,
      latency_ms=(time.perf_counter() - started) * 1000.0,
      result_preview=text[:200],
    )
    self._record_audit(session_id, audit)

    return (not result.isError), text, audit

  def _invoke(
    self,
    server_name: str,
    tool_name: str,
    qualified_name: str,
    arguments: Dict[str, Any],
  ) -> ToolCallResult | ToolCallAudit:
    '''Run the MCP call, optionally wrapped in a tracer span.

    Args:
      server_name: Server owning the tool.
      tool_name: Tool name on the server.
      qualified_name: Namespaced name for tracing.
      arguments: Tool arguments.

    Returns:
      ToolCallResult on success, ToolCallAudit carrying the error otherwise.
    '''
    started = time.perf_counter()
    client = self._clients[server_name]

    if self._tracer is None:
      try:
        return client.call_tool(tool_name, arguments)
      except Exception as exc:  # pylint: disable=broad-exception-caught
        return ToolCallAudit(
          server=server_name, tool=tool_name, args=arguments, ok=False,
          approved=True, error=str(exc),
          latency_ms=(time.perf_counter() - started) * 1000.0,
        )

    with self._tracer.span(
      self._trace_id,
      'tool.call',
      tool=qualified_name,
      args=json.dumps(arguments, default=str)[:500],
    ) as span:
      try:
        result = client.call_tool(tool_name, arguments)
      except Exception as exc:  # pylint: disable=broad-exception-caught
        span.set_attr('error', str(exc))
        return ToolCallAudit(
          server=server_name, tool=tool_name, args=arguments, ok=False,
          approved=True, error=str(exc),
          latency_ms=(time.perf_counter() - started) * 1000.0,
        )

      span.add_usage(_usage_of_result(result))
      span.set_attr('is_error', result.isError)
      return result

  def _rejected_audit(
    self,
    qualified_name: str,
    arguments: Dict[str, Any],
    started: float,
    error: str,
  ) -> ToolCallAudit:
    '''Build an audit row for a call rejected before execution.

    Args:
      qualified_name: Proposed tool name.
      arguments: Proposed arguments.
      started: Start timestamp.
      error: Rejection reason.

    Returns:
      Audit row with ok=False.
    '''
    server = qualified_name.split('.', 1)[0]
    return ToolCallAudit(
      server=server, tool=qualified_name, args=arguments, ok=False,
      approved=False, error=error,
      latency_ms=(time.perf_counter() - started) * 1000.0,
    )

  def _record_audit(self, session_id: str, audit: ToolCallAudit) -> None:
    '''Persist and log one audit row.

    Args:
      session_id: Owning session.
      audit: Audit payload.
    '''
    if self._store is not None:
      try:
        self._store.log_tool_call(
          session_id=session_id,
          server=audit.server,
          tool=audit.tool,
          args=audit.args,
          result={'preview': audit.result_preview},
          ok=audit.ok,
          error=audit.error,
          latency_ms=audit.latency_ms,
          approved=audit.approved,
        )
      except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.warning(
          'audit persist failed',
          extra={'extra_fields': {'error': str(exc)}},
        )

    log_event(
      logger,
      20,
      'tool_call',
      server=audit.server,
      tool=audit.tool,
      ok=audit.ok,
      approved=audit.approved,
      error=audit.error,
      latency_ms=round(audit.latency_ms, 2),
    )


def _usage_of_result(result: ToolCallResult) -> TokenUsage:
  '''Estimate token usage of a tool result (chars/4 heuristic).

  Args:
    result: Tool result.

  Returns:
    TokenUsage.
  '''
  text = result.text or ''
  tokens = max(1, len(text) // 4)
  return TokenUsage(input_tokens=tokens, output_tokens=0, total_tokens=tokens)
