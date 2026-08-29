#!/usr/bin/env python
# -- coding: utf-8 --

'''The MCP client: lifecycle, discovery, and invocation over any transport.

Lifecycle (per MCP spec):
  1. send `initialize` (protocol version + capabilities + client info)
  2. receive server's `initialize` result
  3. send `notifications/initialized` (notification, no reply)
  4. now tools/list and tools/call are allowed

The client is transport-agnostic: it holds any ClientTransport (stdio
subprocess or SSE HTTP) and speaks plain JSON-RPC through it.
'''


from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from chapter01_mcp import jsonrpc, mcp_protocol
from chapter01_mcp.client.transports import (
  ClientTransport,
  TransportError,
  transport_from_descriptor,
)
from chapter01_mcp.logging_util import get_mcp_logger
from chapter01_mcp.schemas import (
  InitializeResult,
  ServerDescriptor,
  ToolCallResult,
  ToolDescriptor,
)

logger = get_mcp_logger(__name__)


class MCPClientError(RuntimeError):
  '''Raised when a server request fails at the protocol level.'''


class MCPClient:
  '''Session with one MCP server.'''

  def __init__(
    self,
    descriptor: ServerDescriptor,
    timeout_seconds: float = 15.0,
    max_retries: int = 2,
    retry_backoff_seconds: float = 0.3,
  ) -> None:
    '''Create a client session (does not connect yet).

    Args:
      descriptor: How to reach the server.
      timeout_seconds: Per-request timeout.
      max_retries: Attempts for transport-level failures.
      retry_backoff_seconds: Base delay between attempts.
    '''
    self.descriptor = descriptor
    self._timeout = timeout_seconds
    self._max_retries = max_retries
    self._backoff = retry_backoff_seconds
    self._transport: Optional[ClientTransport] = None
    self._request_id = 0
    self._initialized = False
    self.server_info: Optional[InitializeResult] = None

  # ------------------------------------------------------------------
  # Lifecycle
  # ------------------------------------------------------------------

  def start(self) -> InitializeResult:
    '''Connect and run the MCP handshake.

    Returns:
      The server's initialize result.

    Raises:
      MCPClientError: If the handshake fails.
    '''
    if self._initialized:
      assert self.server_info is not None
      return self.server_info

    self._transport = transport_from_descriptor(
      self.descriptor,
      timeout_seconds=self._timeout,
    )

    self._request_id += 1
    response = self._request(
      mcp_protocol.METHOD_INITIALIZE,
      {
        'protocolVersion': mcp_protocol.PROTOCOL_VERSION,
        'capabilities': {},
        'clientInfo': {'name': 'agentic-course-client', 'version': '1.0.0'},
      },
    )

    self.server_info = InitializeResult.model_validate(response)
    self._transport.send_notification(
      jsonrpc.encode_notification(mcp_protocol.METHOD_INITIALIZED)
    )
    self._initialized = True
    logger.info(
      'session started',
      extra={'extra_fields': {
        'server': self.server_info.serverInfo.get('name', self.descriptor.name),
        'protocol': self.server_info.protocolVersion,
      }},
    )
    return self.server_info

  def close(self) -> None:
    '''Shut the transport down.'''
    if self._transport is not None:
      self._transport.close()
      self._transport = None
    self._initialized = False

  # ------------------------------------------------------------------
  # MCP operations
  # ------------------------------------------------------------------

  def list_tools(self) -> List[ToolDescriptor]:
    '''Discover the server's tools.

    Returns:
      List of ToolDescriptor.

    Raises:
      MCPClientError: If the call fails or the session is not initialized.
    '''
    self._ensure_started()
    response = self._request(mcp_protocol.METHOD_TOOLS_LIST, {})
    tools_raw = response.get('tools') or []
    tools: List[ToolDescriptor] = []
    for item in tools_raw:
      try:
        tools.append(ToolDescriptor.model_validate(item))
      except Exception as exc:  # pylint: disable=broad-exception-caught
        # Malformed descriptor from a third-party server: skip, never crash.
        logger.warning(
          'skipping malformed tool descriptor',
          extra={'extra_fields': {'error': str(exc)}},
        )
    return tools

  def call_tool(
    self,
    name: str,
    arguments: Optional[Dict[str, Any]] = None,
  ) -> ToolCallResult:
    '''Invoke a tool and return its parsed result.

    Note: tool-level failures arrive as ToolCallResult(isError=True); only
    protocol failures raise MCPClientError.

    Args:
      name: Tool name.
      arguments: Arguments dict (validated later by the server anyway).

    Returns:
      ToolCallResult.

    Raises:
      MCPClientError: On protocol failure (JSON-RPC error / timeout).
    '''
    self._ensure_started()
    self._request_id += 1
    response = self._request(
      mcp_protocol.METHOD_TOOLS_CALL,
      {'name': name, 'arguments': arguments or {}},
      request_id=self._request_id,
    )

    try:
      return ToolCallResult.model_validate(response)
    except Exception as exc:
      raise MCPClientError(f'malformed tool result: {exc}') from exc

  def ping(self) -> bool:
    '''Check server liveness.

    Returns:
      True when the server answers ping.
    '''
    try:
      self._ensure_started()
      self._request(mcp_protocol.METHOD_PING, {})
      return True
    except MCPClientError:
      return False

  # ------------------------------------------------------------------
  # Internals
  # ------------------------------------------------------------------

  def _ensure_started(self) -> None:
    '''Ensure the session has completed the initialize handshake.

    Raises:
      MCPClientError: If not initialized.
    '''
    if not self._initialized:
      self.start()

  def _request(
    self,
    method: str,
    params: Dict[str, Any],
    request_id: Optional[int] = None,
  ) -> Dict[str, Any]:
    '''Send one request with retries, validating the JSON-RPC response.

    Args:
      method: MCP method name.
      params: Method parameters.
      request_id: Optional explicit id.

    Returns:
      The `result` payload.

    Raises:
      MCPClientError: On error responses or exhausted retries.
    '''
    assert self._transport is not None

    if request_id is None:
      self._request_id += 1
      request_id = self._request_id

    line = jsonrpc.encode_request(request_id, method, params)
    last_error: Optional[str] = None

    for attempt in range(self._max_retries + 1):
      try:
        response_line = self._transport.send(line)
        decoded = jsonrpc.decode_response(response_line)
      except (jsonrpc.JsonRpcCodecError, TransportError) as exc:
        # Transport/codec failures are retryable.
        last_error = str(exc)
        if attempt < self._max_retries:
          time.sleep(self._backoff * (2 ** attempt))
          continue
        break

      if isinstance(decoded, jsonrpc.JsonRpcFailure):
        error = decoded.error
        raise MCPClientError(
          f'server error {error.code} for {method}: {error.message}'
          + (f' ({error.data})' if error.data else '')
        )

      return decoded.result

    raise MCPClientError(
      f'transport failed for {method} after {self._max_retries + 1} '
      f'attempts: {last_error}'
    )
