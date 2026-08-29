#!/usr/bin/env python
# -- coding: utf-8 --

'''Unit tests: server core registry, dispatch, and error semantics.'''


from __future__ import annotations

import json

import pytest
from pydantic import BaseModel, Field

from chapter01_mcp.jsonrpc import METHOD_NOT_FOUND
from chapter01_mcp.server_core import MCPServerCore

# pytest fixtures are injected as method args (W0621 is by design here).
# pylint: disable=redefined-outer-name


class EchoInput(BaseModel):
  text: str = Field(min_length=1, max_length=100)


class BoundedInput(BaseModel):
  qty: int = Field(ge=1, le=10)


@pytest.fixture()
def server() -> MCPServerCore:
  '''A server with two test tools.'''
  core = MCPServerCore(name='test-server', version='0.1')

  def echo_handler(data: EchoInput):
    return {'echo': data.text}

  def fail_handler(data: EchoInput):
    raise RuntimeError('boom')

  def qty_handler(data: BoundedInput):
    return {'doubled': data.qty * 2}

  core.register_tool('echo', 'Echo the text back', EchoInput, echo_handler)
  core.register_tool('fail_tool', 'Always fails', EchoInput, fail_handler)
  core.register_tool('qty_tool', 'Doubles qty', BoundedInput, qty_handler)
  return core


def _request(request_id: int, method: str, params: dict) -> str:
  '''Serialize one JSON-RPC request line.'''
  return json.dumps(
    {'jsonrpc': '2.0', 'id': request_id, 'method': method, 'params': params}
  )


class TestRegistry:
  '''Tool registration and descriptor generation.'''

  def test_tool_descriptors_include_schema(self, server: MCPServerCore) -> None:
    tools = server.tool_descriptors()
    names = {t.name for t in tools}
    assert {'echo', 'fail_tool', 'qty_tool'} == names
    for descriptor in tools:
      assert descriptor.inputSchema.get('type') == 'object'

  def test_duplicate_registration_rejected(
    self, server: MCPServerCore,
  ) -> None:
    '''Registering the same tool name twice must fail.'''
    with pytest.raises(ValueError):
      server.register_tool('echo', 'dup', EchoInput, lambda d: d)


class TestDispatch:
  '''JSON-RPC dispatch and MCP method semantics.'''

  def test_initialize_handshake(self, server: MCPServerCore) -> None:
    response = json.loads(server.handle_line(
      _request(
        1,
        'initialize',
        {
          'protocolVersion': '2024-11-05',
          'capabilities': {},
          'clientInfo': {'name': 't'},
        },
      )
    ))
    assert response['result']['serverInfo']['name'] == 'test-server'
    assert response['result']['protocolVersion']

  def test_tools_list(self, server: MCPServerCore) -> None:
    response = json.loads(server.handle_line(_request(2, 'tools/list', {})))
    tool_names = [t['name'] for t in response['result']['tools']]
    assert 'echo' in tool_names

  def test_tools_call_success(self, server: MCPServerCore) -> None:
    response = json.loads(server.handle_line(
      _request(3, 'tools/call', {'name': 'echo', 'arguments': {'text': 'hi'}})
    ))
    assert response['result']['isError'] is False
    echoed = json.loads(response['result']['content'][0]['text'])
    assert echoed == {'echo': 'hi'}

  def test_tools_call_invalid_args_is_tool_error_not_rpc_error(
    self, server: MCPServerCore,
  ) -> None:
    '''Argument mismatch = tool-level failure (isError result).'''
    response = json.loads(server.handle_line(
      _request(4, 'tools/call', {'name': 'echo', 'arguments': {'wrong': True}})
    ))
    assert response['result']['isError'] is True
    assert 'invalid arguments' in response['result']['content'][0]['text']

  def test_tools_call_unknown_tool(self, server: MCPServerCore) -> None:
    response = json.loads(server.handle_line(
      _request(5, 'tools/call', {'name': 'ghost', 'arguments': {}})
    ))
    assert response['result']['isError'] is True
    text = response['result']['content'][0]['text']
    assert 'unknown tool' in text

  def test_tools_call_handler_crash_is_tool_error(
    self, server: MCPServerCore,
  ) -> None:
    '''Handler crash = tool-level failure (isError), server stays alive.'''
    response = json.loads(server.handle_line(
      _request(
        6, 'tools/call',
        {'name': 'fail_tool', 'arguments': {'text': 'x'}},
      )
    ))
    assert response['result']['isError'] is True
    assert 'boom' in response['result']['content'][0]['text']

  def test_unknown_method_is_rpc_error(self, server: MCPServerCore) -> None:
    response = json.loads(server.handle_line(_request(7, 'no/such/method', {})))
    assert response['error']['code'] == METHOD_NOT_FOUND

  def test_malformed_line_gets_parse_error_with_null_id(
    self, server: MCPServerCore,
  ) -> None:
    response = json.loads(server.handle_line('garbage {'))
    assert response['error']['code'] == -32700
    assert response['id'] is None

  def test_notification_returns_none(self, server: MCPServerCore) -> None:
    line = json.dumps(
      {'jsonrpc': '2.0', 'method': 'notifications/initialized'},
    )
    assert server.handle_line(line) is None

  def test_ping(self, server: MCPServerCore) -> None:
    response = json.loads(server.handle_line(_request(9, 'ping', {})))
    assert response['result'] == {}
