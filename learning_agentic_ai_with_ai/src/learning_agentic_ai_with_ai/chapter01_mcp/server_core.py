#!/usr/bin/env python
# -- coding: utf-8 --

'''Hand-rolled MCP server core: tool registry + JSON-RPC dispatch.

Design:
- `MCPServerCore` is transport-agnostic. It knows MCP methods (initialize,
  tools/list, tools/call, ping) and a registry of tools.
- Tools are registered with (name, description, pydantic input model,
  handler). The pydantic model generates the JSON Schema advertised to LLMs
  AND validates arguments at the boundary.
- Tool *execution* failures are returned as normal results with
  `isError: true` (the calling LLM can read them and adapt). Only protocol
  violations produce JSON-RPC errors.
'''


from __future__ import annotations

import json
import time
import traceback
from typing import Any, Callable, Dict, List, Optional, Type

from pydantic import BaseModel, ValidationError

from chapter01_mcp import jsonrpc, mcp_protocol
from chapter01_mcp.logging_util import get_mcp_logger
from chapter01_mcp.schemas import (
  InitializeParams,
  InitializeResult,
  JsonRpcNotification,
  JsonRpcRequest,
  ToolDescriptor,
)

logger = get_mcp_logger(__name__)

ToolHandler = Callable[[BaseModel], Any]


class ToolRegistration(BaseModel):
  '''One registered tool (internal model).'''

  name: str
  description: str
  input_model: Type[BaseModel]
  handler: ToolHandler


class MCPServerCore:
  '''Transport-agnostic MCP server: registry + JSON-RPC dispatch.'''

  def __init__(self, name: str, version: str = '1.0.0') -> None:
    '''Initialize the server core.

    Args:
      name: Server name reported in initialize.
      version: Server version string.
    '''
    self.name = name
    self.version = version
    self._tools: Dict[str, ToolRegistration] = {}

  # ------------------------------------------------------------------
  # Registration
  # ------------------------------------------------------------------

  def register_tool(
    self,
    name: str,
    description: str,
    input_model: Type[BaseModel],
    handler: ToolHandler,
  ) -> None:
    '''Register a tool.

    Args:
      name: Unique tool name.
      description: Description the LLM sees.
      input_model: Pydantic model for arguments (also becomes JSON Schema).
      handler: Function from validated input model to JSON-serializable data.

    Raises:
      ValueError: If a tool with the same name is already registered.
    '''
    if name in self._tools:
      raise ValueError(f'tool already registered: {name}')
    self._tools[name] = ToolRegistration(
      name=name,
      description=description,
      input_model=input_model,
      handler=handler,
    )

  def tool_descriptors(self) -> List[ToolDescriptor]:
    '''List tool descriptors advertised over tools/list.

    Returns:
      List of ToolDescriptor.
    '''
    return [
      mcp_protocol.tool_descriptor(t.name, t.description, t.input_model)
      for t in self._tools.values()
    ]

  # ------------------------------------------------------------------
  # Dispatch
  # ------------------------------------------------------------------

  def handle_message(
    self,
    message: object,
  ) -> Optional[Dict[str, Any]]:
    '''Process a decoded JSON-RPC message and produce a response dict.

    Args:
      message: Decoded JsonRpcRequest or JsonRpcNotification.

    Returns:
      Response dict, or None for notifications.
    '''
    if isinstance(message, JsonRpcNotification):
      # Notifications never get a response.
      return None

    if isinstance(message, JsonRpcRequest):
      try:
        result = self._dispatch(message.method, message.params)
        return json.loads(jsonrpc.encode_success(message.id, result))
      except jsonrpc.JsonRpcCodecError as exc:
        return json.loads(
          jsonrpc.encode_error(message.id, jsonrpc.INVALID_PARAMS, str(exc))
        )
      except MethodNotFound as exc:
        return json.loads(
          jsonrpc.encode_error(message.id, jsonrpc.METHOD_NOT_FOUND, str(exc))
        )
      except ValidationError as exc:
        return json.loads(
          jsonrpc.encode_error(
            message.id,
            jsonrpc.INVALID_PARAMS,
            'invalid params',
            {'detail': exc.errors(include_url=False)},
          )
        )
      except Exception as exc:  # pylint: disable=broad-exception-caught
        # Tool handler crash: last-resort guard keeps the server alive.
        logger.error(
          'internal error handling %s',
          message.method,
          extra={'extra_fields': {'trace': traceback.format_exc()}},
        )
        return json.loads(
          jsonrpc.encode_error(
            message.id,
            jsonrpc.INTERNAL_ERROR,
            f'internal error: {exc}',
          )
        )

    return json.loads(
      jsonrpc.encode_error(
        None, jsonrpc.INVALID_REQUEST, 'unsupported message type'
      )
    )

  def handle_line(self, line: str) -> Optional[str]:
    '''Decode one raw line, dispatch it, and encode the response line.

    Malformed lines produce an error response with id=None (per JSON-RPC).

    Args:
      line: Raw JSON text line.

    Returns:
      Response line, or None for notifications/malformed-notify cases.
    '''
    try:
      message = jsonrpc.decode_line(line)
    except jsonrpc.JsonRpcCodecError as exc:
      return jsonrpc.encode_error(None, jsonrpc.PARSE_ERROR, str(exc))

    response = self.handle_message(message)
    if response is None:
      return None
    return json.dumps(response, separators=(',', ':'), default=str)

  def _dispatch(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
    '''Execute one MCP method.

    Args:
      method: Method name.
      params: Method parameters.

    Returns:
      Result dict.

    Raises:
      MethodNotFound: For unknown methods.
    '''
    if method == mcp_protocol.METHOD_INITIALIZE:
      return self._initialize(params)

    if method == mcp_protocol.METHOD_PING:
      return {}

    if method == mcp_protocol.METHOD_TOOLS_LIST:
      return {'tools': [d.model_dump() for d in self.tool_descriptors()]}

    if method == mcp_protocol.METHOD_TOOLS_CALL:
      return self._call_tool(params)

    raise MethodNotFound(f'method not found: {method}')

  def _initialize(self, params: Dict[str, Any]) -> Dict[str, Any]:
    '''Handle initialize handshake.

    Args:
      params: Initialize parameters.

    Returns:
      InitializeResult payload.
    '''
    client_params = InitializeParams.model_validate(params)
    result = InitializeResult(
      protocolVersion=mcp_protocol.PROTOCOL_VERSION,
      capabilities=mcp_protocol.server_capabilities(self.tool_descriptors()),
      serverInfo=mcp_protocol.default_server_info(self.name, self.version),
    )
    logger.info(
      'initialized by client',
      extra={'extra_fields': {
        'client': client_params.clientInfo.get('name', 'unknown'),
        'client_version': client_params.protocolVersion,
      }},
    )
    return result.model_dump()

  def _call_tool(self, params: Dict[str, Any]) -> Dict[str, Any]:
    '''Handle tools/call.

    Tool-level failures (bad usage, empty data, business errors) are returned
    as isError results; protocol-level problems raise ValidationError (mapped
    to INVALID_PARAMS) or generic exceptions (INTERNAL_ERROR).

    Args:
      params: CallToolParams.

    Returns:
      ToolCallResult payload.
    '''
    name = str(params.get('name', ''))
    arguments = params.get('arguments') or {}

    registration = self._tools.get(name)
    if registration is None:
      return mcp_protocol.text_result(f'unknown tool: {name}', is_error=True)

    started = time.perf_counter()
    try:
      validated = registration.input_model.model_validate(arguments)
    except ValidationError as exc:
      # Argument mismatch is *tool usage* failure -> isError result.
      return mcp_protocol.text_result(
        'invalid arguments for ' + name + ': ' + exc.json(include_url=False),
        is_error=True,
      )

    try:
      output = registration.handler(validated)
      payload = self._serialize_output(output)
    except Exception as exc:  # pylint: disable=broad-exception-caught
      # Tool-level failure: report as isError result, never crash dispatch.
      logger.warning(
        'tool raised',
        extra={'extra_fields': {'tool': name, 'error': str(exc)}},
      )
      return mcp_protocol.text_result(
        f'tool {name} failed: {exc}', is_error=True
      )

    latency_ms = (time.perf_counter() - started) * 1000.0
    logger.debug(
      'tool executed',
      extra={
        'extra_fields': {'tool': name, 'latency_ms': round(latency_ms, 2)}
      },
    )
    return mcp_protocol.text_result(payload)

  @staticmethod
  def _serialize_output(output: Any) -> str:
    '''Convert handler output into result text.

    Args:
      output: str, or JSON-serializable object.

    Returns:
      Text payload for the content block.
    '''
    if isinstance(output, str):
      return output
    return json.dumps(output, default=str)


class MethodNotFound(Exception):
  '''Raised when a JSON-RPC method has no handler.'''
