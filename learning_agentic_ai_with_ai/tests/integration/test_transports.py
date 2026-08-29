#!/usr/bin/env python
# -- coding: utf-8 --

'''Integration tests: MCP over real transports (stdio subprocess, SSE).'''


from __future__ import annotations

import sys
import time

import pytest

# pytest fixture injection is the standard pattern here.
# pylint: disable=redefined-outer-name,unused-import

@pytest.fixture()
def retail_stdio_client():
  '''A connected MCPClient against the retail stdio server subprocess.'''
  from chapter01_mcp.client.mcp_client import MCPClient
  from chapter01_mcp.schemas import ServerDescriptor

  descriptor = ServerDescriptor(
    name='retail-ops',
    transport='stdio',
    command=sys.executable,
    args=['-m', 'chapter01_mcp.servers.retail_main'],
  )
  client = MCPClient(descriptor, timeout_seconds=20)
  try:
    client.start()
    yield client
  finally:
    client.close()


@pytest.fixture()
def telecom_sse_client():
  '''A connected MCPClient against an in-thread SSE server (auto port).'''
  from chapter01_mcp.client.mcp_client import MCPClient
  from chapter01_mcp.schemas import ServerDescriptor
  from chapter01_mcp.server.sse import SseServerRunner
  from chapter01_mcp.servers.telecom_server import build_telecom_server

  runner = SseServerRunner(build_telecom_server(), port=0)
  runner.serve_in_background()
  time.sleep(0.5)

  descriptor = ServerDescriptor(
    name='telecom-ops',
    transport='sse',
    url=f'http://127.0.0.1:{runner.port}',
  )
  client = MCPClient(descriptor, timeout_seconds=10)
  try:
    client.start()
    yield client
  finally:
    client.close()


class TestStdioTransport:
  '''Full client lifecycle against the retail stdio subprocess.'''

  def test_full_lifecycle(self, retail_stdio_client) -> None:
    tools = retail_stdio_client.list_tools()
    expected = {
      'retail_low_stock_report',
      'retail_sales_trend',
      'retail_restock_order',
    }
    assert expected == {tool.name for tool in tools}

  def test_tool_roundtrip_data(self, retail_stdio_client) -> None:
    result = retail_stdio_client.call_tool('retail_low_stock_report', {})
    assert result.isError is False
    assert 'items' in result.text

  def test_tool_error_is_result_not_exception(
    self, retail_stdio_client,
  ) -> None:
    '''Unknown store yields a normal result, not an exception.'''
    result = retail_stdio_client.call_tool(
      'retail_sales_trend', {'store_id': 'NOPE', 'sku': 'X'},
    )
    assert isinstance(result.isError, bool)

  def test_ping(self, retail_stdio_client) -> None:
    assert retail_stdio_client.ping() is True


class TestSseTransport:
  '''Full client session against the in-process SSE server.'''

  def test_full_session(self, telecom_sse_client) -> None:
    tools = telecom_sse_client.list_tools()
    assert 'telecom_degraded_sites' in [t.name for t in tools]

    result = telecom_sse_client.call_tool('telecom_degraded_sites', {})
    assert result.isError is False
    assert 'CS-77' in result.text or 'sites' in result.text

  def test_write_tool_rejects_off_duty(self, telecom_sse_client) -> None:
    result = telecom_sse_client.call_tool(
      'telecom_dispatch_technician',
      {'site_id': 'CS-77', 'tech_id': 'T-04', 'priority': 'high'},
    )
    assert result.isError is True
    assert 'off duty' in result.text
