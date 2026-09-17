#!/usr/bin/env python
# -- coding: utf-8 --

'''Integration tests: the full Chapter 2 agent stack (mock LLM to tools).'''


from __future__ import annotations

from typing import Any

from agentic_common.eval.harness import RunOutcome
from agentic_common.gateway_client import MockGateway
from agentic_common.persistence import AgentStore
from agentic_common.settings import Settings
from agentic_common.tracing import NullTracer, Tracer
from chapter02_planning.agent import PlanningAgent
from chapter02_planning.config import PlanningConfig, load_planning_config
from chapter02_planning.demo import (
  SCENARIOS,
  make_mock_planner,
  make_write_approver,
)
from chapter02_planning.evals.cases import (
  PlanningEvalCase,
  evaluate_planning_case,
)
from chapter02_planning.schemas import PlanningTaskInput
from chapter02_planning.tools.factory import build_tool_provider


def build_agent(
  settings: Settings,
  config: PlanningConfig,
  store: Any = None,
  tracer: Any = None,
  allow_writes_approver: bool = True,
) -> PlanningAgent:
  '''Build a PlanningAgent with the scripted mock planner.

  Args:
    settings: Runtime settings.
    config: Chapter 2 configuration.
    store: Optional AgentStore.
    tracer: Optional tracer.
    allow_writes_approver: Whether to wire the bounded approver.

  Returns:
    PlanningAgent.
  '''
  return PlanningAgent(
    llm=MockGateway(make_mock_planner(config)),
    tools=build_tool_provider(
      'inproc', task='retail telecom operations',
    ),
    config=config,
    settings=settings,
    store=store,
    tracer=tracer or NullTracer(),
    write_approver=make_write_approver(settings) if allow_writes_approver else None,
  )


class TestRoutes:
  '''Every route runs end-to-end through the agent.'''

  def test_direct_route(self, settings: Settings) -> None:
    config = load_planning_config(settings)
    agent = build_agent(settings, config)
    try:
      output = agent.run_task(
        PlanningTaskInput(
          task=SCENARIOS['direct'], session_id='it-direct',
        )
      )
    finally:
      agent.close()

    assert output.route == 'direct'
    assert output.status == 'completed'
    assert '168' in output.answer

  def test_cot_route(self, settings: Settings) -> None:
    config = load_planning_config(settings)
    agent = build_agent(settings, config)
    try:
      output = agent.run_task(
        PlanningTaskInput(
          task=SCENARIOS['cot'], session_id='it-cot',
          cot_samples=1,
        )
      )
    finally:
      agent.close()

    assert output.route == 'cot'
    assert output.status == 'completed'
    assert '110' in output.answer
    assert output.reasoning_steps
    assert output.validation is not None
    assert output.validation.passed is True

  def test_react_route(self, settings: Settings) -> None:
    config = load_planning_config(settings)
    agent = build_agent(settings, config)
    try:
      output = agent.run_task(
        PlanningTaskInput(
          task=SCENARIOS['react'], session_id='it-react',
        )
      )
    finally:
      agent.close()

    assert output.route == 'react'
    assert output.status == 'completed'
    assert output.react is not None
    assert output.react.stop_reason == 'final_answer'
    tools_used = [audit.tool for audit in output.tool_calls]
    assert 'retail-ops.retail_low_stock_report' in tools_used
    assert 'retail-ops.retail_sales_trend' in tools_used
    assert all(audit.ok for audit in output.tool_calls)

  def test_plan_route(self, settings: Settings) -> None:
    config = load_planning_config(settings)
    agent = build_agent(settings, config)
    task = PlanningTaskInput(
      task=SCENARIOS['plan'], session_id='it-plan', allow_writes=True,
    )
    try:
      output = agent.run_task(task)
    finally:
      agent.close()

    assert output.route == 'plan'
    assert output.plan is not None
    assert len(output.plan.nodes) == 3
    assert output.execution is not None
    assert output.execution.waves == [
      ['inspect_stock'], ['review_trend'], ['place_order'],
    ]
    assert all(
      result.status == 'completed'
      for result in output.execution.results
    )
    assert all(
      result.tool_calls for result in output.execution.results
    )
    assert output.status == 'completed'
    assert output.validation is not None
    assert output.validation.passed is True


class TestSafety:
  '''The gate and approver bound write behavior.'''

  def test_unsafe_write_blocked_and_acknowledged(
    self, settings: Settings,
  ) -> None:
    config = load_planning_config(settings)
    agent = build_agent(settings, config)
    try:
      output = agent.run_task(
        PlanningTaskInput(
          task=SCENARIOS['unsafe'], session_id='it-unsafe',
          allow_writes=True,
        )
      )
    finally:
      agent.close()

    blocked = [audit for audit in output.tool_calls if audit.blocked]
    assert blocked, 'oversized write must be blocked'
    assert blocked[0].approved is False
    assert not any(
      audit.ok and 'restock_order' in audit.tool
      for audit in output.tool_calls
    )
    assert 'blocked' in output.answer.lower()
    assert output.validation is not None
    assert output.validation.passed is True

  def test_read_only_run_blocks_writes(self, settings: Settings) -> None:
    config = load_planning_config(settings)
    agent = build_agent(settings, config)
    try:
      output = agent.run_task(
        PlanningTaskInput(
          task=SCENARIOS['unsafe'], session_id='it-readonly',
          allow_writes=False,
        )
      )
    finally:
      agent.close()

    blocked = [audit for audit in output.tool_calls if audit.blocked]
    assert blocked
    assert 'disabled' in (blocked[0].error or '')


class TestPersistenceAndObservability:
  '''Events, memory, tool audits, and traces are written.'''

  def test_store_records_full_history(
    self, settings: Settings, tmp_path: Any,
  ) -> None:
    config = load_planning_config(settings)
    store = AgentStore(tmp_path / 'state.db')
    agent = build_agent(settings, config, store=store)
    try:
      output = agent.run_task(
        PlanningTaskInput(
          task=SCENARIOS['react'], session_id='it-persist',
        )
      )
      assert output.status == 'completed'

      events = store.history('it-persist')
      event_types = [event.event_type for event in events]
      assert event_types[0] == 'task_start'
      assert 'route_decision' in event_types
      assert 'pattern_finished' in event_types
      assert event_types[-1] == 'task_finish'

      memory = store.all_memory('it-persist')
      assert memory['last_answer']['route'] == 'react'

      audits = store.tool_calls('it-persist')
      assert len(audits) >= 2
      assert all(audit.ok for audit in audits)
    finally:
      agent.close()
      store.close()

  def test_traces_written_as_jsonl(
    self, settings: Settings, tmp_path: Any,
  ) -> None:
    config = load_planning_config(settings)
    tracer = Tracer(trace_dir=tmp_path)
    agent = build_agent(settings, config, tracer=tracer)
    try:
      output = agent.run_task(
        PlanningTaskInput(
          task=SCENARIOS['cot'], session_id='it-trace', cot_samples=1,
        )
      )
    finally:
      agent.close()

    assert output.status == 'completed'
    trace_files = list(tmp_path.glob('*.jsonl'))
    assert len(trace_files) == 1
    content = trace_files[0].read_text(encoding='utf-8')
    assert 'llm.call' in content

  def test_token_budget_exhaustion_fails_safely(
    self, settings: Settings,
  ) -> None:
    config = load_planning_config(settings).model_copy(
      update={'token_budget': 1},
    )
    agent = build_agent(settings, config)
    try:
      output = agent.run_task(
        PlanningTaskInput(
          task=SCENARIOS['direct'], session_id='it-budget',
        )
      )
    finally:
      agent.close()

    assert output.status == 'error'
    assert any(
      'token budget' in error for error in output.errors
    )


class TestRoutingOverride:
  '''Caller hints always win.'''

  def test_prefer_route_override(self, settings: Settings) -> None:
    config = load_planning_config(settings)
    agent = build_agent(settings, config)
    try:
      output = agent.run_task(
        PlanningTaskInput(
          task=SCENARIOS['direct'], session_id='it-forced',
          prefer_route='cot', cot_samples=1,
        )
      )
    finally:
      agent.close()

    assert output.route == 'cot'
    assert output.policy is not None
    assert output.policy.forced is True


class TestEvalScoring:
  '''Chapter 2 eval checks score route/validation/plan shape.'''

  def test_eval_case_scores(self) -> None:
    case = PlanningEvalCase(
      id='case',
      task='x',
      expect_route='react',
      expect_validation_passed=True,
      plan_nodes_completed_min=2,
      expect_blocked_tool='srv.write',
      min_tool_calls=1,
      max_iterations=5,
    )
    good = RunOutcome(
      answer='done',
      tool_calls=[
        {'tool': 'srv.read', 'blocked': False},
        {'tool': 'srv.write', 'blocked': True},
      ],
      iterations=3,
      extra={
        'route': 'react',
        'validation_passed': True,
        'plan_nodes_completed': 2,
        'blocked_tools': ['srv.write'],
      },
    )
    result = evaluate_planning_case(case, good)
    assert result.passed

    bad = RunOutcome(
      answer='done',
      tool_calls=[],
      iterations=9,
      extra={
        'route': 'direct',
        'validation_passed': False,
        'plan_nodes_completed': 0,
        'blocked_tools': [],
      },
    )
    result_bad = evaluate_planning_case(case, bad)
    assert not result_bad.passed
    failed_checks = {
      check.check for check in result_bad.checks if not check.passed
    }
    assert 'route:react' in failed_checks
    assert 'validation:expected' in failed_checks
    assert 'blocked:srv.write' in failed_checks
