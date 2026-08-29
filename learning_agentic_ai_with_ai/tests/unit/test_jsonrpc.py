#!/usr/bin/env python
# -- coding: utf-8 --

'''Unit tests: JSON-RPC codec and MCP protocol helpers.'''


from __future__ import annotations

import json

import pytest
from pydantic import BaseModel

from chapter01_mcp import jsonrpc, mcp_protocol
from chapter01_mcp.jsonrpc import JsonRpcCodecError
from chapter01_mcp.schemas import (
  JsonRpcFailure,
  JsonRpcRequest,
  JsonRpcSuccess,
)


class TestEncode:
  '''Encoding helpers produce valid single-line JSON-RPC.'''

  def test_encode_request_shape(self) -> None:
    line = jsonrpc.encode_request(1, 'tools/list', {'a': 1})
    payload = json.loads(line)
    assert payload == {
      'jsonrpc': '2.0',
      'id': 1,
      'method': 'tools/list',
      'params': {'a': 1},
    }

  def test_encode_request_without_params(self) -> None:
    line = jsonrpc.encode_request('x1', 'ping')
    payload = json.loads(line)
    assert 'params' not in payload
    assert payload['id'] == 'x1'

  def test_encode_notification_has_no_id(self) -> None:
    line = jsonrpc.encode_notification('notifications/initialized')
    payload = json.loads(line)
    assert 'id' not in payload
    assert payload['method'] == 'notifications/initialized'

  def test_encode_error_includes_code_and_message(self) -> None:
    line = jsonrpc.encode_error(7, jsonrpc.METHOD_NOT_FOUND, 'no such method')
    payload = json.loads(line)
    assert payload['error']['code'] == -32601
    assert payload['id'] == 7


class TestDecodeRequest:
  '''Request/notification decoding accepts valid lines only.'''

  def test_roundtrip(self) -> None:
    line = jsonrpc.encode_request(
      3, 'tools/call', {'name': 't', 'arguments': {'x': 1}}
    )
    message = jsonrpc.decode_line(line)
    assert isinstance(message, JsonRpcRequest)
    assert message.id == 3
    assert message.method == 'tools/call'

  def test_notification_has_no_id(self) -> None:
    line = jsonrpc.encode_notification('notifications/initialized')
    message = jsonrpc.decode_line(line)
    assert message.method == 'notifications/initialized'

  def test_invalid_json_raises(self) -> None:
    with pytest.raises(JsonRpcCodecError):
      jsonrpc.decode_line('not json at all')

  def test_wrong_jsonrpc_version(self) -> None:
    try:
      jsonrpc.decode_line('{"jsonrpc":"1.0","id":1,"method":"ping"}')
      raise AssertionError('expected error')
    except jsonrpc.JsonRpcCodecError:
      pass

  def test_empty_line_rejected(self) -> None:
    try:
      jsonrpc.decode_line('   ')
      raise AssertionError('expected error')
    except jsonrpc.JsonRpcCodecError:
      pass


class TestDecodeResponse:
  '''Response decoding handles success and failure payloads.'''

  def test_success_roundtrip(self) -> None:
    line = jsonrpc.encode_success(11, {'ok': True})
    decoded = jsonrpc.decode_response(line)
    assert isinstance(decoded, JsonRpcSuccess)
    assert decoded.result == {'ok': True}

  def test_failure_decoded(self) -> None:
    line = jsonrpc.encode_error(5, -32601, 'missing')
    decoded = jsonrpc.decode_response(line)
    assert isinstance(decoded, JsonRpcFailure)
    assert decoded.error.code == -32601


class TestMcpProtocol:
  '''Protocol helpers convert pydantic models to schemas.'''

  def test_pydantic_to_json_schema(self) -> None:
    class ExampleInput(BaseModel):
      site_id: str
      quantity: int = 1

    schema = mcp_protocol.pydantic_to_json_schema(ExampleInput)
    assert schema['type'] == 'object'
    assert 'site_id' in schema['properties']

  def test_text_result_shape(self) -> None:
    result = mcp_protocol.text_result('hello', is_error=True)
    assert result['isError'] is True
    assert result['content'][0]['text'] == 'hello'
