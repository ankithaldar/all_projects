#!/usr/bin/env python
# -- coding: utf-8 --

'''Official-SDK MCP server for the retail tools (Python, FastMCP).

This shows the *production* way to build the same retail server: the `mcp`
package's FastMCP decorator style. Pydantic models still define the inputs,
so the schema guarantees match our hand-rolled server exactly.

Run standalone:
  python -m chapter01_mcp.servers.retail_sdk_server
'''


from __future__ import annotations

from typing import Any, Dict

from mcp.server.fastmcp import FastMCP

from chapter01_mcp.logging_util import get_mcp_logger
from chapter01_mcp.ops_models import (
  RetailLowStockInput,
  RetailRestockInput,
  RetailSalesTrendInput,
)
from chapter01_mcp.servers.ops_db import seed_if_empty
from chapter01_mcp.servers.retail_server import (
  _handle_low_stock,
  _handle_restock,
  _handle_sales_trend,
)

logger = get_mcp_logger(__name__)

mcp = FastMCP('retail-ops-sdk')


@mcp.tool()
def retail_low_stock_report(data: RetailLowStockInput) -> Dict[str, Any]:
  '''List retail items below their reorder point, optionally by store.'''
  return _handle_low_stock(data)


@mcp.tool()
def retail_sales_trend(data: RetailSalesTrendInput) -> Dict[str, Any]:
  '''Return daily units sold for a store+SKU over the last N days and avg.'''
  return _handle_sales_trend(data)


@mcp.tool()
def retail_restock_order(data: RetailRestockInput) -> Dict[str, Any]:
  '''Create a restock order for a store and SKU (write tool, policy-capped).'''
  return _handle_restock(data)


def main() -> None:
  '''Seed the mock DB and serve over stdio using the official SDK.'''
  seed_if_empty()
  mcp.run()  # defaults to stdio transport


if __name__ == '__main__':
  main()
