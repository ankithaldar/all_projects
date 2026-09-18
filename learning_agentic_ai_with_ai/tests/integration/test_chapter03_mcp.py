#!/usr/bin/env python
# -- coding: utf-8 --

'''Integration tests: Chapter 3 toolbox over real MCP stdio.'''


from __future__ import annotations

import pytest

from chapter03_cognition.tools import ToolBox

LOW_STOCK = 'retail-ops.retail_low_stock_report'
RESTOCK = 'retail-ops.retail_restock_order'


@pytest.fixture()
def retail_mcp_toolbox():
  '''A connected MCP ToolBox for the retail server.

  Skips when MCP servers cannot start in this environment.
  '''
  toolbox = ToolBox(
    backend='mcp', selected_servers=['retail-ops'],
    timeout_seconds=20.0,
  )
  if not toolbox.list_tools():
    toolbox.close()
    pytest.skip('retail-ops MCP server unavailable in this environment')
  try:
    yield toolbox
  finally:
    toolbox.close()


class TestMCPToolBox:
  '''Discovery and execution through MCP.'''

  def test_lists_qualified_tools(self, retail_mcp_toolbox) -> None:
    names = {spec.name for spec in retail_mcp_toolbox.list_tools()}
    assert LOW_STOCK in names
    assert RESTOCK in names

  def test_execute_read_tool(self, retail_mcp_toolbox) -> None:
    result = retail_mcp_toolbox.call(LOW_STOCK, {})
    assert result.ok is True
    assert 'items' in result.text

  def test_gate_blocks_write_before_server(
    self, retail_mcp_toolbox,
  ) -> None:
    result = retail_mcp_toolbox.call(
      RESTOCK, {'store_id': 'S01', 'sku': 'R-101', 'quantity': 1},
    )
    assert result.ok is False
    assert result.text.startswith('BLOCKED by safety gate')
    assert result.latency_ms < 500.0
