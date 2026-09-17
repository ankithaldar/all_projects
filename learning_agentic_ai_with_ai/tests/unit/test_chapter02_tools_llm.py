#!/usr/bin/env python
# -- coding: utf-8 --

'''Unit tests: tool providers/gates and the reasoning LLM wrapper.'''


from __future__ import annotations

import json

from agentic_common.gateway_client import MockGateway
from chapter02_planning.config import PlanningConfig
from chapter02_planning.llm import ReasoningLLM
from chapter02_planning.schemas import Critique
from chapter02_planning.tools.factory import BACKENDS, build_tool_provider
from chapter02_planning.tools.in_process import InProcessToolProvider
from chapter02_planning.tools.provider import (
  UNTRUSTED_CLOSE,
  UNTRUSTED_OPEN,
  GatedToolProvider,
)
from llm_gateway.schemas import GatewayRequest, GatewayResponse


class TestInProcessProvider:
  '''Chapter 1 server cores behind the ToolProvider interface.'''

  def test_lists_qualified_tools(self) -> None:
    provider = InProcessToolProvider()
    names = {tool.name for tool in provider.list_tools()}
    assert 'retail-ops.retail_low_stock_report' in names
    assert 'telecom-ops.telecom_dispatch_technician' in names

  def test_write_flags(self) -> None:
    provider = InProcessToolProvider()
    specs = {tool.name: tool for tool in provider.list_tools()}
    assert specs['retail-ops.retail_restock_order'].read_only is False
    assert specs['retail-ops.retail_low_stock_report'].read_only is True
    assert specs['telecom-ops.telecom_dispatch_technician'].read_only is False

  def test_execute_read_tool(self) -> None:
    provider = InProcessToolProvider()
    result = provider.execute(
      'retail-ops.retail_low_stock_report', {},
    )
    assert result.ok is True
    assert result.text
    assert isinstance(json.loads(result.text), dict)

  def test_execute_unknown_tool(self) -> None:
    provider = InProcessToolProvider()
    result = provider.execute('retail-ops.nope', {})
    assert result.ok is False
    assert 'unknown tool' in (result.error or '')

  def test_invalid_arguments_are_tool_errors(self) -> None:
    provider = InProcessToolProvider()
    result = provider.execute(
      'retail-ops.retail_sales_trend', {'days': 'many'},
    )
    assert result.ok is False
    assert result.error


class TestGatedProvider:
  '''Least-privilege and hygiene decorator.'''

  def test_blocks_writes_by_default(self) -> None:
    audits = []
    gate = GatedToolProvider(
      InProcessToolProvider(), allow_writes=False, audit_sink=audits.append,
    )
    result = gate.execute(
      'retail-ops.retail_restock_order',
      {'store_id': 'S01', 'sku': 'R-101', 'quantity': 5},
    )
    assert result.ok is False
    assert result.text.startswith('BLOCKED by safety gate')
    assert audits and audits[0].blocked is True

  def test_allows_writes_when_enabled_and_approved(self) -> None:
    gate = GatedToolProvider(
      InProcessToolProvider(),
      allow_writes=True,
      require_approval=True,
      approver=lambda _tool, args: args.get('quantity', 0) <= 500,
    )
    result = gate.execute(
      'retail-ops.retail_restock_order',
      {'store_id': 'S02', 'sku': 'R-401', 'quantity': 5},
    )
    assert result.ok is True

  def test_approver_denies_oversized_write(self) -> None:
    gate = GatedToolProvider(
      InProcessToolProvider(),
      allow_writes=True,
      require_approval=True,
      approver=lambda _tool, _args: False,
    )
    result = gate.execute(
      'retail-ops.retail_restock_order',
      {'store_id': 'S01', 'sku': 'R-101', 'quantity': 5},
    )
    assert result.ok is False
    assert 'not approved' in result.text

  def test_blocks_unknown_tool_without_backend_hit(self) -> None:
    gate = GatedToolProvider(InProcessToolProvider())
    result = gate.execute('bogus.tool', {})
    assert result.ok is False
    assert 'unknown tool' in result.text

  def test_observations_are_sanitized_and_wrapped(self) -> None:
    gate = GatedToolProvider(InProcessToolProvider())
    result = gate.execute('retail-ops.retail_low_stock_report', {})
    assert result.text.startswith(UNTRUSTED_OPEN)
    assert result.text.rstrip().endswith(UNTRUSTED_CLOSE)

  def test_gateway_tools_shape(self) -> None:
    gate = GatedToolProvider(InProcessToolProvider())
    definitions = gate.gateway_tools()
    assert definitions
    first = definitions[0]
    assert first['type'] == 'function'
    assert first['function']['name']


class TestFactory:
  '''Backend factory behavior.'''

  def test_builds_inproc(self) -> None:
    provider = build_tool_provider('inproc')
    assert provider.name == 'inproc'
    assert provider.list_tools()

  def test_unknown_backend_raises(self) -> None:
    import pytest

    with pytest.raises(ValueError):
      build_tool_provider('quantum')

  def test_backends_constant(self) -> None:
    assert set(BACKENDS) == {'inproc', 'mcp'}


class TestReasoningLLM:
  '''Retries, budgets, and structured completion.'''

  def test_retries_transient_failures(self) -> None:
    class FlakyGateway:
      def __init__(self) -> None:
        self.attempts = 0

      def complete(self, request: GatewayRequest) -> GatewayResponse:
        self.attempts += 1
        if self.attempts <= 2:
          raise RuntimeError('transient')
        return GatewayResponse(
          provider='flaky', model='m', content='hello',
        )

    flaky = FlakyGateway()
    llm = ReasoningLLM(flaky, PlanningConfig(retry_base_seconds=0.0))
    result = llm.complete('prompt')
    assert result is not None
    assert result.content == 'hello'
    assert flaky.attempts == 3
    assert llm.calls == 1

  def test_returns_none_after_exhausting_retries(self) -> None:
    class DeadGateway:
      def complete(self, request: GatewayRequest) -> GatewayResponse:
        raise RuntimeError('down')

    llm = ReasoningLLM(
      DeadGateway(),
      PlanningConfig(llm_max_attempts=2, retry_base_seconds=0.0),
    )
    assert llm.complete('prompt') is None
    assert llm.failures == 2
    assert llm.last_error == 'down'

  def test_budget_tracking(self) -> None:
    gateway = MockGateway(
      lambda _messages: GatewayResponse(
        provider='mock', model='m', content='answer text',
      ),
    )
    llm = ReasoningLLM(gateway, PlanningConfig(token_budget=1))
    assert llm.complete('prompt') is not None
    assert llm.budget.exceeded is True
    assert llm.budget.remaining == 0
    assert llm.usage_snapshot()['calls'] == 1

  def test_complete_json_repairs(self) -> None:
    calls = {'n': 0}

    def planner(_messages):
      calls['n'] += 1
      if calls['n'] == 1:
        return GatewayResponse(
          provider='mock', model='m', content='not json at all',
        )
      return GatewayResponse(
        provider='mock', model='m',
        content='{"verdict": "pass", "issues": []}',
      )

    llm = ReasoningLLM(MockGateway(planner), PlanningConfig())
    critique = llm.complete_json('prompt', Critique)
    assert critique is not None
    assert critique.verdict == 'pass'
    assert llm.parse_repairs == 1

  def test_complete_json_gives_up(self) -> None:
    gateway = MockGateway(
      lambda _messages: GatewayResponse(
        provider='mock', model='m', content='still not json',
      ),
    )
    llm = ReasoningLLM(
      gateway, PlanningConfig(llm_max_attempts=1),
    )
    assert llm.complete_json('prompt', Critique) is None

  def test_complete_json_retries_empty_completion(self) -> None:
    calls = {'n': 0}

    def planner(_messages):
      calls['n'] += 1
      if calls['n'] == 1:
        return GatewayResponse(provider='mock', model='m', content='   ')
      return GatewayResponse(
        provider='mock', model='m',
        content='{"verdict": "pass", "issues": []}',
      )

    llm = ReasoningLLM(MockGateway(planner), PlanningConfig())
    critique = llm.complete_json('prompt', Critique)
    assert critique is not None
    assert calls['n'] == 2
    assert llm.parse_repairs == 0
