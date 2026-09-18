#!/usr/bin/env python
# -- coding: utf-8 --

'''Unit tests: the Chapter 1-backed toolbox and fault injector.'''


from __future__ import annotations

from typing import List

import pytest

from chapter03_cognition.schemas import ToolExecutionAudit
from chapter03_cognition.tools import (
  UNTRUSTED_CLOSE,
  UNTRUSTED_OPEN,
  FaultInjector,
  ToolBox,
)

LOW_STOCK = 'retail-ops.retail_low_stock_report'
RESTOCK = 'retail-ops.retail_restock_order'
TREND = 'retail-ops.retail_sales_trend'


class TestToolBoxInProc:
  '''Discovery and execution through Chapter 1 server cores.'''

  def test_lists_qualified_tools(self) -> None:
    toolbox = ToolBox(backend='inproc')
    names = {spec.name for spec in toolbox.list_tools()}
    assert LOW_STOCK in names
    assert RESTOCK in names

  def test_write_flags(self) -> None:
    toolbox = ToolBox(backend='inproc')
    specs = {spec.name: spec for spec in toolbox.list_tools()}
    assert specs[RESTOCK].read_only is False
    assert specs[LOW_STOCK].read_only is True
    assert RESTOCK in toolbox.write_tools()

  def test_read_call_succeeds_and_wraps(self) -> None:
    toolbox = ToolBox(backend='inproc')
    result = toolbox.call(LOW_STOCK, {})
    assert result.ok is True
    assert result.text.startswith(UNTRUSTED_OPEN)
    assert result.text.rstrip().endswith(UNTRUSTED_CLOSE)

  def test_write_blocked_by_default(self) -> None:
    toolbox = ToolBox(backend='inproc', allow_writes=False)
    result = toolbox.call(
      RESTOCK, {'store_id': 'S01', 'sku': 'R-101', 'quantity': 5},
    )
    assert result.ok is False
    assert result.text.startswith('BLOCKED by safety gate')

  def test_write_allowed_with_approver(self) -> None:
    toolbox = ToolBox(
      backend='inproc',
      allow_writes=True,
      require_approval=True,
      approver=lambda _tool, args: int(args.get('quantity', 0)) <= 500,
    )
    result = toolbox.call(
      RESTOCK, {'store_id': 'S02', 'sku': 'R-401', 'quantity': 5},
    )
    assert result.ok is True

  def test_write_denied_by_approver(self) -> None:
    toolbox = ToolBox(
      backend='inproc',
      allow_writes=True,
      require_approval=True,
      approver=lambda _tool, _args: False,
    )
    result = toolbox.call(
      RESTOCK, {'store_id': 'S01', 'sku': 'R-101', 'quantity': 5},
    )
    assert result.ok is False
    assert 'not approved' in (result.error or '')

  def test_unknown_tool_blocked(self) -> None:
    toolbox = ToolBox(backend='inproc')
    result = toolbox.call('bogus.tool', {})
    assert result.ok is False
    assert 'unknown tool' in result.text

  def test_invalid_arguments_are_tool_errors(self) -> None:
    toolbox = ToolBox(backend='inproc')
    result = toolbox.call(TREND, {'days': 'many'})
    assert result.ok is False
    assert result.error

  def test_audit_sink_receives_records(self) -> None:
    audits: List[ToolExecutionAudit] = []
    toolbox = ToolBox(backend='inproc', audit_sink=audits.append)
    toolbox.call(LOW_STOCK, {})
    toolbox.call(RESTOCK, {'quantity': 1})
    assert len(audits) == 2
    assert audits[0].ok is True
    assert audits[1].blocked is True

  def test_allowed_names_excludes_writes(self) -> None:
    toolbox = ToolBox(backend='inproc', allow_writes=False)
    assert RESTOCK not in toolbox.allowed_names()
    assert LOW_STOCK in toolbox.allowed_names()

  def test_unknown_backend_raises(self) -> None:
    with pytest.raises(ValueError):
      ToolBox(backend='quantum')


class TestFaultInjector:
  '''Deterministic failure injection for adaptive tests.'''

  def test_fails_then_passes_through(self) -> None:
    toolbox = FaultInjector(
      ToolBox(backend='inproc'), failures={TREND: 2},
    )
    first = toolbox.call(TREND, {'store_id': 'S01', 'sku': 'R-101'})
    second = toolbox.call(TREND, {'store_id': 'S01', 'sku': 'R-101'})
    third = toolbox.call(TREND, {'store_id': 'S01', 'sku': 'R-101'})
    assert first.ok is False
    assert second.ok is False
    assert third.ok is True
    assert first.error == 'injected transient failure'

  def test_delegates_discovery(self) -> None:
    toolbox = FaultInjector(ToolBox(backend='inproc'), failures={})
    assert toolbox.list_tools()
    assert toolbox.allowed_names()
    assert toolbox.write_tools()
