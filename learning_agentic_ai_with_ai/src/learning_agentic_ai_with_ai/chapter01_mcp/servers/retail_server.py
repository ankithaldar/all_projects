#!/usr/bin/env python
# -- coding: utf-8 --

'''Pydantic input models + handlers for the Retail Operations MCP server.

Each tool:
1. declares a pydantic input model (typed args + JSON Schema for the LLM),
2. a handler function doing the business work (SQLite reads/writes),
3. is registered on an MCPServerCore.

Tools (retail_ops_*):
- retail_low_stock_report : read   - which items are below reorder point?
- retail_sales_trend      : read   - avg units/day for a store+SKU
- retail_restock_order    : write  - create a restock order (policy-capped)
'''


from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

from chapter01_mcp.logging_util import get_mcp_logger
from chapter01_mcp.ops_models import (
  RetailLowStockInput,
  RetailRestockInput,
  RetailSalesTrendInput,
  RetailSalesTrendOutput,
  RetailSalesTrendPoint,
)
from chapter01_mcp.server_core import MCPServerCore
from chapter01_mcp.servers.ops_db import connect

logger = get_mcp_logger(__name__)


def _handle_low_stock(data: RetailLowStockInput) -> Dict[str, Any]:
  '''Return inventory items below their product reorder point.

  Args:
    data: Validated input (optional store filter).

  Returns:
    Dict with items list.
  '''
  conn = connect()
  try:
    query = '''
      SELECT i.store_id, i.sku, i.on_hand, p.name, p.reorder_point, p.lead_time_days
      FROM inventory i
      JOIN products p ON p.sku = i.sku
      WHERE i.on_hand < p.reorder_point
    '''
    params: List[Any] = []
    if data.store_id:
      query += ' AND i.store_id = ?'
      params.append(data.store_id)
    query += ' ORDER BY (p.reorder_point - i.on_hand) DESC LIMIT 50'

    rows = conn.execute(query, params).fetchall()
    return {
      'items': [
        {
          'store_id': r['store_id'],
          'sku': r['sku'],
          'product': r['name'],
          'on_hand': r['on_hand'],
          'reorder_point': r['reorder_point'],
          'lead_time_days': r['lead_time_days'],
        }
        for r in rows
      ]
    }
  finally:
    conn.close()


def _handle_sales_trend(data: RetailSalesTrendInput) -> Dict[str, Any]:
  '''Return daily unit sales for store+SKU over the last N days.

  Args:
    data: Validated input.

    Returns:
      Dict with series and summary.
  '''
  conn = connect()
  try:
    rows = conn.execute(
      '''
      SELECT day, SUM(units) AS units
      FROM sales
      WHERE store_id = ? AND sku = ? AND day >= ?
      GROUP BY day ORDER BY day ASC
      ''',
      (
        data.store_id,
        data.sku,
        (
          datetime.now(timezone.utc) - timedelta(days=data.days - 1)
        ).date().isoformat(),
      ),
    ).fetchall()

    points = [
      RetailSalesTrendPoint(day=row['day'], units=int(row['units']))
      for row in rows
    ]
    units_list = [p.units for p in points]
    avg = round(sum(units_list) / len(units_list), 2) if units_list else 0.0
    return RetailSalesTrendOutput(
      store_id=data.store_id,
      sku=data.sku,
      points=points,
      avg_daily_units=avg,
    ).model_dump()
  finally:
    conn.close()


def _handle_restock(data: RetailRestockInput) -> Dict[str, Any]:
  '''Create a restock order after policy checks (write tool).

  Args:
    data: Validated input.

    Returns:
      Dict with order id and echo.

    Raises:
      ValueError: On unknown store/SKU or quantity policy violation.
  '''
  conn = connect()
  try:
    product = conn.execute(
      'SELECT sku, name FROM products WHERE sku = ?', (data.sku,)
    ).fetchone()
    if product is None:
      raise ValueError(f'unknown sku: {data.sku}')

    store = conn.execute(
      'SELECT store_id FROM stores WHERE store_id = ?', (data.store_id,)
    ).fetchone()
    if store is None:
      raise ValueError(f'unknown store: {data.store_id}')

    now = datetime.now(timezone.utc).isoformat()
    cursor = conn.execute(
      'INSERT INTO restock_orders '
      '(store_id, sku, quantity, created_at, status) '
      'VALUES (?,?,?,?,?)',
      (data.store_id, data.sku, data.quantity, now, 'created'),
    )
    conn.commit()
    return {
      'order_id': int(cursor.lastrowid or 0),
      'store_id': data.store_id,
      'sku': data.sku,
      'quantity': data.quantity,
      'status': 'created',
      'created_at': now,
    }
  finally:
    conn.close()


def build_retail_server():
  '''Build the retail operations MCP server core with tools registered.

  Returns:
    Configured MCPServerCore.
  '''
  server = MCPServerCore(name='retail-ops', version='1.0.0')

  server.register_tool(
    name='retail_low_stock_report',
    description=(
      'List retail inventory items currently below their reorder point, '
      'optionally filtered by store_id. Includes lead time for planning.'
    ),
    input_model=RetailLowStockInput,
    handler=_handle_low_stock,
  )

  server.register_tool(
    name='retail_sales_trend',
    description=(
      'Return daily units sold for a specific store and SKU over the last N '
      'days plus the average daily units. Use before deciding restock size.'
    ),
    input_model=RetailSalesTrendInput,
    handler=_handle_sales_trend,
  )

  server.register_tool(
    name='retail_restock_order',
    description=(
      'Create a restock order for a store and SKU. WRITE tool: quantity is '
      'capped by policy. Confirm with the user if approval is required.'
    ),
    input_model=RetailRestockInput,
    handler=_handle_restock,
  )

  return server
