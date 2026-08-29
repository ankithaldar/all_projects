#!/usr/bin/env python
# -- coding: utf-8 --

'''MCP protocol constants and helpers (the layer above raw JSON-RPC).

MCP = Model Context Protocol. It defines *which* JSON-RPC methods mean what:
  initialize            -> handshake: exchange versions + capabilities
  notifications/initialized -> client confirms handshake (notification)
  tools/list            -> discover tools: [{name, description, inputSchema}]
  tools/call            -> invoke a tool: {name, arguments}
  ping                  -> liveness check

It also fixes result shapes (e.g. tool results are {content: [...], isError}).
'''


from __future__ import annotations

from typing import Any, Dict, List, Type

from pydantic import BaseModel

from chapter01_mcp.schemas import ToolDescriptor

PROTOCOL_VERSION = '2024-11-05'

METHOD_INITIALIZE = 'initialize'
METHOD_INITIALIZED = 'notifications/initialized'
METHOD_TOOLS_LIST = 'tools/list'
METHOD_TOOLS_CALL = 'tools/call'
METHOD_PING = 'ping'


def pydantic_to_json_schema(model: Type[BaseModel]) -> Dict[str, Any]:
  '''Generate a JSON Schema from a pydantic model.

  This is the bridge between "typed Python inputs" and "the schema string an
  LLM sees". The LLM fills arguments as JSON; we validate them back into the
  pydantic model at the server boundary.

  Args:
    model: Pydantic model class describing tool arguments.

  Returns:
    JSON Schema dict.
  '''
  schema: Dict[str, Any] = model.model_json_schema()

  # Drop internals that confuse some LLM providers.
  schema.pop('title', None)
  definitions = schema.pop('$defs', None)
  if definitions:
    schema['definitions'] = definitions

  return schema


def tool_descriptor(
  name: str,
  description: str,
  input_model: Type[BaseModel],
) -> ToolDescriptor:
  '''Build a ToolDescriptor from a pydantic input model.

  Args:
    name: Tool name.
    description: What the tool does (the LLM reads this to decide usage).
    input_model: Pydantic model for arguments.

  Returns:
    Complete ToolDescriptor.
  '''
  return ToolDescriptor(
    name=name,
    description=description,
    inputSchema=pydantic_to_json_schema(input_model),
  )


def text_result(text: str, is_error: bool = False) -> Dict[str, Any]:
  '''Build a tools/call result payload with one text block.

  Args:
    text: Result text (often JSON-serialized data).
    is_error: Whether this result represents a tool-level failure.

  Returns:
    MCP result dict.
  '''
  return {'content': [{'type': 'text', 'text': text}], 'isError': is_error}


def server_capabilities(tools: List[ToolDescriptor]) -> Dict[str, Any]:
  '''Build server capabilities for the initialize result.

  Args:
    tools: Registered tools.

  Returns:
    Capabilities dict.
  '''
  return {'tools': {'listChanged': False}, 'toolCount': len(tools)}


def default_server_info(name: str, version: str) -> Dict[str, str]:
  '''Build serverInfo for the initialize result.

  Args:
    name: Server name.
    version: Server version.

  Returns:
    serverInfo dict.
  '''
  return {'name': name, 'version': version}
