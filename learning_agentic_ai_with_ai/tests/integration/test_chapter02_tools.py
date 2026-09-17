#!/usr/bin/env python
# -- coding: utf-8 --

'''Integration tests: Chapter 2 tool providers over real MCP stdio.'''


from __future__ import annotations

import pytest

from chapter02_planning.tools.in_process import InProcessToolProvider
from chapter02_planning.tools.mcp_provider import MCPToolProvider
from chapter02_planning.tools.provider import GatedToolProvider


@pytest.fixture()
def retail_mcp_provider():
  '''A connected MCPToolProvider for the retail server.

  Skips when MCP servers cannot start in this environment (for example when
  the subprocess cannot import the project package).
  '''
  provider = MCPToolProvider(
    selected_servers=['retail-ops'], timeout_seconds=20.0,
  )
  tools = provider.list_tools()
  if not tools:
    provider.close()
    pytest.skip('retail-ops MCP server unavailable in this environment')
  try:
    yield provider
  finally:
    provider.close()


class TestMCPProvider:
  '''Discovery and execution through MCP.'''

  def test_lists_qualified_retail_tools(self, retail_mcp_provider) -> None:
    names = {tool.name for tool in retail_mcp_provider.list_tools()}
    assert 'retail-ops.retail_low_stock_report' in names
    assert 'retail-ops.retail_restock_order' in names

  def test_write_flags_match_inproc(self, retail_mcp_provider) -> None:
    mcp_specs = {tool.name: tool for tool in retail_mcp_provider.list_tools()}
    inproc_specs = {
      tool.name: tool for tool in InProcessToolProvider().list_tools()
    }
    for name, spec in mcp_specs.items():
      assert spec.read_only == inproc_specs[name].read_only

  def test_execute_read_tool(self, retail_mcp_provider) -> None:
    result = retail_mcp_provider.execute(
      'retail-ops.retail_low_stock_report', {},
    )
    assert result.ok is True
    assert 'items' in result.text

  def test_gate_blocks_write_before_server(
    self, retail_mcp_provider,
  ) -> None:
    gate = GatedToolProvider(
      retail_mcp_provider, allow_writes=False,
    )
    result = gate.execute(
      'retail-ops.retail_restock_order',
      {'store_id': 'S01', 'sku': 'R-101', 'quantity': 1},
    )
    assert result.ok is False
    assert result.text.startswith('BLOCKED by safety gate')
    assert result.latency_ms < 500.0


class TestBackendEquivalence:
  '''Both backends expose the same tool namespace.'''

  def test_retail_tool_names_match(self, retail_mcp_provider) -> None:
    mcp_names = {tool.name for tool in retail_mcp_provider.list_tools()}
    inproc_names = {
      tool.name
      for tool in InProcessToolProvider().list_tools()
      if tool.server == 'retail-ops'
    }
    assert mcp_names == inproc_names
