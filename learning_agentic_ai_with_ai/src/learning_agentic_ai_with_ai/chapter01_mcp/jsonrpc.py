#!/usr/bin/env python
# -- coding: utf-8 --

'''Hand-rolled JSON-RPC 2.0 codec.

JSON-RPC 2.0 is the *wire format* MCP rides on. Before MCP existed, this
protocol was already used by LSP (Language Server Protocol). A message is
just one line of JSON (newline-delimited over stdio):

  Request      -> {"jsonrpc":"2.0","id":1,"method":"tools/list",
                  "params":{}}
  Response     -> {"jsonrpc":"2.0","id":1,"result":{"tools":[...]}}
  Error resp   -> {"jsonrpc":"2.0","id":1,"error":{"code":-32601,
                  "message":"..."}}
  Notification -> {"jsonrpc":"2.0","method":"notifications/initialized"}
                  (no id!)
'''


from __future__ import annotations

import json
from typing import Any, Dict, Optional, Union

from chapter01_mcp.schemas import (
  JsonRpcFailure,
  JsonRpcNotification,
  JsonRpcRequest,
  JsonRpcSuccess,
)

PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603


class JsonRpcCodecError(ValueError):
  '''Raised when a line cannot be decoded into a valid JSON-RPC message.'''


JsonRpcIncoming = Union[JsonRpcRequest, JsonRpcNotification]


def encode_request(
  request_id: Union[int, str],
  method: str,
  params: Optional[Dict[str, Any]] = None,
) -> str:
  '''Encode a request line.

  Args:
    request_id: Correlation id (int or str).
    method: Method name.
    params: Optional parameters object.

  Returns:
    Single-line JSON string (no trailing newline).
  '''
  message: Dict[str, Any] = {
    'jsonrpc': '2.0',
    'id': request_id,
    'method': method,
  }
  if params:
    message['params'] = params
  return json.dumps(message, separators=(',', ':'), default=str)


def encode_notification(
  method: str,
  params: Optional[Dict[str, Any]] = None,
) -> str:
  '''Encode a notification line (no id, no reply expected).

  Args:
    method: Method name.
    params: Optional parameters object.

  Returns:
    Single-line JSON string.
  '''
  message: Dict[str, Any] = {'jsonrpc': '2.0', 'method': method}
  if params:
    message['params'] = params
  return json.dumps(message, separators=(',', ':'), default=str)


def encode_success(request_id: Union[int, str], result: Dict[str, Any]) -> str:
  '''Encode a successful response line.

  Args:
    request_id: Id copied from the request.
    result: Result payload.

  Returns:
    Single-line JSON string.
  '''
  return json.dumps(
    {'jsonrpc': '2.0', 'id': request_id, 'result': result},
    separators=(',', ':'),
    default=str,
  )


def encode_error(
  request_id: Optional[Union[int, str]],
  code: int,
  message: str,
  data: Optional[Dict[str, Any]] = None,
) -> str:
  '''Encode an error response line.

  Args:
    request_id: Id from the request, or None when unknown.
    code: JSON-RPC error code.
    message: Human-readable error message.
    data: Optional structured error data.

  Returns:
    Single-line JSON string.
  '''
  error: Dict[str, Any] = {'code': code, 'message': message}
  if data:
    error['data'] = data
  return json.dumps(
    {'jsonrpc': '2.0', 'id': request_id, 'error': error},
    separators=(',', ':'),
    default=str,
  )


def decode_line(line: str) -> JsonRpcIncoming:
  '''Decode one incoming line into a request or notification model.

  Args:
    line: Raw JSON text (newline already stripped).

  Returns:
    JsonRpcRequest or JsonRpcNotification.

  Raises:
    JsonRpcCodecError: On malformed JSON or protocol violations.
  '''
  stripped = (line or '').strip()
  if not stripped:
    raise JsonRpcCodecError('empty line')

  try:
    payload = json.loads(stripped)
  except json.JSONDecodeError as exc:
    raise JsonRpcCodecError(f'invalid JSON: {exc}') from exc

  if not isinstance(payload, dict):
    raise JsonRpcCodecError('message must be a JSON object')

  if payload.get('jsonrpc') != '2.0':
    raise JsonRpcCodecError(
      "missing or invalid 'jsonrpc' field (expected '2.0')"
    )

  method = payload.get('method')
  if not isinstance(method, str) or not method:
    raise JsonRpcCodecError("missing or invalid 'method' field")

  params = payload.get('params') or {}
  if not isinstance(params, dict):
    raise JsonRpcCodecError("'params' must be an object")

  if 'id' in payload:
    try:
      return JsonRpcRequest(
        id=payload['id'],
        method=method,
        params=params,
      )
    except Exception as exc:
      raise JsonRpcCodecError(f'invalid request: {exc}') from exc

  try:
    return JsonRpcNotification(method=method, params=params)
  except Exception as exc:
    raise JsonRpcCodecError(f'invalid notification: {exc}') from exc


def decode_response(line: str) -> Union[JsonRpcSuccess, JsonRpcFailure]:
  '''Decode one incoming response line on the client side.

  Args:
    line: Raw JSON text.

  Returns:
    JsonRpcSuccess or JsonRpcFailure.

  Raises:
    JsonRpcCodecError: On malformed responses.
  '''
  stripped = (line or '').strip()
  if not stripped:
    raise JsonRpcCodecError('empty response line')

  try:
    payload = json.loads(stripped)
  except json.JSONDecodeError as exc:
    raise JsonRpcCodecError(f'invalid JSON in response: {exc}') from exc

  if not isinstance(payload, dict) or payload.get('jsonrpc') != '2.0':
    raise JsonRpcCodecError('not a JSON-RPC 2.0 response')

  if 'error' in payload and payload['error'] is not None:
    try:
      return JsonRpcFailure.model_validate(payload)
    except Exception as exc:
      raise JsonRpcCodecError(f'invalid error response: {exc}') from exc

  if 'result' not in payload:
    raise JsonRpcCodecError('response must contain result or error')

  try:
    return JsonRpcSuccess.model_validate(payload)
  except Exception as exc:
    raise JsonRpcCodecError(f'invalid success response: {exc}') from exc
