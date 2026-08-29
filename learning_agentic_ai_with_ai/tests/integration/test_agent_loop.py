#!/usr/bin/env python
# -- coding: utf-8 --

'''Integration tests: the full agent loop (mock LLM to MCP).'''


from __future__ import annotations

from typing import Any

import pytest

from agentic_common import paths
from agentic_common.eval.harness import EvalCase, RunOutcome, evaluate_case
from agentic_common.gateway_client import MockGateway
from agentic_common.persistence import AgentStore
from agentic_common.settings import Settings
from agentic_common.tracing import NullTracer


def build_agent(settings: Settings, store=None, tracer: Any = None):
  '''Build the OpsAgent with a scripted mock planner.'''
  from chapter01_mcp.agent.ops_agent import OpsAgent
  from chapter01_mcp.demo import make_mock_planner

  return OpsAgent(
    llm=MockGateway(make_mock_planner(settings)),
    settings=settings,
    store=store,
    tracer=tracer or NullTracer(),
  )


class TestAgentLoopRestock:
  '''The retail restock task flows end-to-end.'''

  def test_full_flow(self, settings: Settings) -> None:
    from chapter01_mcp.schemas import AgentTaskInput

    agent = build_agent(settings)
    output = agent.run_task(
      AgentTaskInput(
        task=(
          'Find the most critical low-stock item in store S01, check its sales '
          'trend, and place a restock order sized to about one week of demand.'
        ),
        session_id='it-restock',
      )
    )

    assert output.status == 'completed'
    tools_used = [c.tool for c in output.tool_calls]
    assert 'retail_low_stock_report' in tools_used
    assert 'retail_sales_trend' in tools_used
    assert 'retail_restock_order' in tools_used
    assert all(c.ok for c in output.tool_calls)
    assert output.usage.get('total_tokens', 0) > 0
    assert 'restock order' in output.answer.lower()

  def test_execution_history_persisted(
    self, settings: Settings, tmp_path: Any,
  ) -> None:
    '''Task start/finish events and tool audits are persisted.'''
    from chapter01_mcp.schemas import AgentTaskInput

    store = AgentStore(tmp_path / 'state.db')
    agent = build_agent(settings, store=store)
    output = agent.run_task(
      AgentTaskInput(
        task=(
          'Find the most critical low-stock item in store S01, check its '
          'sales trend, and place a restock order.'
        ),
        session_id='it-persist',
      ),
    )
    try:
      assert output.status == 'completed'
      events = store.history('it-persist')
      assert events[0].event_type == 'task_start'
      assert events[-1].event_type == 'task_finish'
      assert len(store.tool_calls('it-persist')) >= 3
    finally:
      store.close()


class TestAgentLoopTelecom:
  '''The telecom dispatch task flows end-to-end.'''

  def test_full_flow(self, settings: Settings) -> None:
    from chapter01_mcp.schemas import AgentTaskInput

    agent = build_agent(settings)
    output = agent.run_task(
      AgentTaskInput(
        task=(
          'Check which cell sites are degraded right now, inspect the worst '
          'one, and dispatch technician T-06 there with high priority.'
        ),
        session_id='it-telecom',
      )
    )

    assert output.status == 'completed'
    tools_used = [c.tool for c in output.tool_calls]
    assert 'telecom_degraded_sites' in tools_used
    assert 'telecom_site_status' in tools_used
    assert 'telecom_dispatch_technician' in tools_used


class TestSafety:
  '''Policy engine blocks out-of-bounds writes.'''

  def test_unsafe_restock_blocked(self, settings: Settings) -> None:
    from chapter01_mcp.schemas import AgentTaskInput

    agent = build_agent(settings)
    output = agent.run_task(
      AgentTaskInput(
        task='Restock store S01 with 9000 units of R-101 immediately.',
        session_id='it-unsafe',
      )
    )

    blocked_calls = [
      call for call in output.tool_calls
      if call.tool == 'retail_restock_order' and not call.ok
    ]
    assert blocked_calls, 'policy must block the oversized write'
    assert blocked_calls[0].approved is False
    answer_words = ('blocked', 'policy')
    assert any(word in output.answer.lower() for word in answer_words)

  def test_policy_blocks_before_server(self) -> None:
    '''The policy engine rejects the call BEFORE any MCP traffic.'''
    from chapter01_mcp.client.policy import ToolPolicyEngine
    from chapter01_mcp.schemas import ToolDescriptor

    def _deny(*_args):
      return False

    engine = ToolPolicyEngine(
      write_tools={'srv.w'}, approval_callback=_deny,
    )
    descriptor = ToolDescriptor(
      name='w',
      inputSchema={
        'type': 'object',
        'properties': {'qty': {'type': 'integer'}},
      },
    )
    allowed, reason, _ = engine.check(
      'srv', 'w', {'qty': 1}, descriptor,
    )
    assert not allowed
    assert 'not approved' in reason


class TestEvalHarnessUnit:
  '''Eval cases score tool usage, answers, and budgets.'''

  def test_case_scoring(self) -> None:
    from agentic_common.eval.harness import TokenUsage

    case = EvalCase(
      id='t',
      task='x',
      expect_tool_called=['a.t1'],
      expect_tool_not_called=['a.t2'],
      expect_answer_contains=['done'],
      max_iterations=3,
    )
    good = RunOutcome(
      answer='done t1',
      tool_calls=[{'tool': 'a.t1'}],
      iterations=2,
      usage=TokenUsage(total_tokens=10),
    )
    result = evaluate_case(case, good)
    assert result.passed

    bad = RunOutcome(
      answer='done',
      tool_calls=[{'tool': 'a.t2'}],
      iterations=5,
      usage=TokenUsage(total_tokens=10),
    )
    result_bad = evaluate_case(case, bad)
    assert not result_bad.passed


class TestTypeScriptInterop:
  '''Python client drives the TypeScript SDK server over stdio.'''

  def test_python_client_drives_ts_server(self) -> None:
    '''Cross-language interop: Python MCPClient -> TypeScript server.'''
    import shutil

    node = shutil.which('node')
    if node is None:
      pytest.skip('node not available')

    ts_dir = paths.CHAPTER1_TYPESCRIPT_DIR
    if not (ts_dir / 'node_modules').exists():
      pytest.skip(
        'typescript deps not installed (npm install in typescript/)'
      )

    from chapter01_mcp.client.mcp_client import MCPClient
    from chapter01_mcp.schemas import ServerDescriptor

    descriptor = ServerDescriptor(
      name='telecom-ops-ts',
      transport='stdio',
      command=node,
      args=['--experimental-strip-types', 'src/telecom_server.ts'],
      cwd=str(ts_dir),
    )
    client = MCPClient(descriptor, timeout_seconds=20)
    try:
      client.start()
      tools = client.list_tools()
      assert 'telecom_site_status' in [t.name for t in tools]

      result = client.call_tool('telecom_site_status', {'site_id': 'CS-77'})
      assert result.isError is False
      assert 'CS-77' in result.text
    finally:
      client.close()
