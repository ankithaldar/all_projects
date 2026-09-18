#!/usr/bin/env python
# -- coding: utf-8 --

'''Integration tests: the full 4-layer cognitive pipeline.'''


from __future__ import annotations

from typing import Any, List, Optional

from agentic_common.gateway_client import MockGateway
from agentic_common.persistence import AgentStore
from agentic_common.settings import Settings
from agentic_common.tracing import Tracer
from chapter03_cognition.agent import CognitiveAgent
from chapter03_cognition.config import CognitionConfig, load_cognition_config
from chapter03_cognition.demo import (
  SCENARIOS,
  make_mock_planner,
  make_write_approver,
)
from chapter03_cognition.memory import MemoryStore
from chapter03_cognition.schemas import (
  CognitiveTaskInput,
  ToolResult,
)
from chapter03_cognition.tools import FaultInjector, ToolBox

TREND = 'retail-ops.retail_sales_trend'


class FailingToolBox:
  '''Toolbox whose every call fails transiently (escalation tests).'''

  def __init__(self, inner: ToolBox) -> None:
    '''Wrap a real toolbox for discovery.

    Args:
      inner: Real toolbox (specs only).
    '''
    self._inner = inner

  def list_tools(self) -> List[Any]:  # noqa: D102
    return self._inner.list_tools()

  def allowed_names(self) -> set:  # noqa: D102
    return self._inner.allowed_names()

  def write_tools(self) -> set:  # noqa: D102
    return self._inner.write_tools()

  def call(self, tool: str, args: dict) -> ToolResult:
    '''Return a transient failure for every tool.

    Args:
      tool: Tool name.
      args: Tool arguments.

    Returns:
      Failed ToolResult.
    '''
    return ToolResult(
      tool=tool, args=args, ok=False,
      text='tool error: injected transient failure',
      error='injected transient failure',
    )

  def close(self) -> None:  # noqa: D102
    self._inner.close()


def build_agent(
  settings: Settings,
  config: CognitionConfig,
  tmp_path: Any,
  store: Optional[AgentStore] = None,
  memory: Optional[MemoryStore] = None,
  tracer: Any = None,
  with_tools: bool = True,
  fault_injection: bool = False,
  failing_tools: bool = False,
  allow_writes: bool = False,
) -> CognitiveAgent:
  '''Build a CognitiveAgent with the scripted mock planner.

  Args:
    settings: Runtime settings.
    config: Cognition config.
    tmp_path: Temp directory for memory (when not provided).
    store: Optional execution-history store.
    memory: Optional memory store.
    tracer: Optional tracer.
    with_tools: Whether to attach a toolbox.
    fault_injection: Whether to inject trend failures.
    failing_tools: Whether every tool call fails.
    allow_writes: Whether writes are permitted.

  Returns:
    CognitiveAgent.
  '''
  agent = CognitiveAgent(
    llm=MockGateway(make_mock_planner(config)),
    toolbox=None,
    config=config,
    settings=settings,
    store=store,
    memory=memory or MemoryStore(tmp_path / 'memory.db'),
    tracer=tracer,
  )
  if with_tools:
    toolbox: Any = ToolBox(
      backend='inproc',
      allow_writes=allow_writes,
      require_approval=config.require_write_approval,
      approver=make_write_approver(settings),
      audit_sink=agent.audit_sink,
    )
    if failing_tools:
      toolbox = FailingToolBox(toolbox)
    elif fault_injection:
      toolbox = FaultInjector(
        toolbox, failures={TREND: 3},
        error='injected transient failure',
      )
    agent.attach_toolbox(toolbox)
  return agent


class TestPipelineRoutes:
  '''Each plan shape runs end-to-end through all four layers.'''

  def test_dag_pipeline(self, settings: Settings, tmp_path: Any) -> None:
    config = load_cognition_config(settings)
    agent = build_agent(settings, config, tmp_path)
    try:
      output = agent.run_task(
        CognitiveTaskInput(
          task=SCENARIOS['dag_retail'], session_id='it3-dag',
        )
      )
    finally:
      agent.close()

    assert output.status == 'completed'
    assert output.percept is not None
    assert output.percept.signals.domain == 'retail'
    assert output.decision is not None
    assert output.decision.plan.source == 'dag'
    assert output.decision.insight.step_count == 2
    assert output.action is not None
    assert all(
      step.status == 'completed' for step in output.action.steps
    )
    assert output.answer

  def test_code_pipeline(self, settings: Settings, tmp_path: Any) -> None:
    config = load_cognition_config(settings)
    agent = build_agent(settings, config, tmp_path)
    try:
      output = agent.run_task(
        CognitiveTaskInput(
          task=SCENARIOS['code_restock'], session_id='it3-code',
          plan_mode='code',
        )
      )
    finally:
      agent.close()

    assert output.status == 'completed'
    assert output.decision is not None
    assert output.decision.plan.source == 'code'
    assert output.action is not None
    assert output.action.code_runner == 'restricted'
    assert '110' in output.answer

  def test_subprocess_pipeline(
    self, settings: Settings, tmp_path: Any,
  ) -> None:
    config = load_cognition_config(settings)
    agent = build_agent(settings, config, tmp_path)
    try:
      output = agent.run_task(
        CognitiveTaskInput(
          task=SCENARIOS['code_restock'], session_id='it3-sub',
          plan_mode='code', code_runner='subprocess',
        )
      )
    finally:
      agent.close()

    assert output.status == 'completed'
    assert output.action is not None
    assert output.action.code_runner == 'subprocess'
    assert '110' in output.answer

  def test_fallback_pipeline(
    self, settings: Settings, tmp_path: Any,
  ) -> None:
    config = load_cognition_config(settings)
    agent = build_agent(settings, config, tmp_path, with_tools=False)
    try:
      output = agent.run_task(
        CognitiveTaskInput(
          task=SCENARIOS['fallback'], session_id='it3-fallback',
          strategy_hint='fallback',
        )
      )
    finally:
      agent.close()

    assert output.status == 'completed'
    assert output.decision is not None
    assert output.decision.strategy == 'fallback'
    lowered = output.answer.lower()
    assert 'unverified' in lowered or 'degraded' in lowered


class TestAdaptation:
  '''Retry, tool switching, and escalation.'''

  def test_retry_and_tool_switch(
    self, settings: Settings, tmp_path: Any,
  ) -> None:
    config = load_cognition_config(settings)
    agent = build_agent(
      settings, config, tmp_path, fault_injection=True,
    )
    try:
      output = agent.run_task(
        CognitiveTaskInput(
          task=SCENARIOS['adaptive'], session_id='it3-adaptive',
          strategy_hint='exploratory',
        )
      )
    finally:
      agent.close()

    assert output.status == 'completed'
    assert output.action is not None
    kinds = {record.kind for record in output.action.adaptations}
    assert {'retry', 'tool_switch'} <= kinds

  def test_strategy_escalation(
    self, settings: Settings, tmp_path: Any,
  ) -> None:
    config = load_cognition_config(settings)
    agent = build_agent(
      settings, config, tmp_path, failing_tools=True,
    )
    try:
      output = agent.run_task(
        CognitiveTaskInput(
          task=SCENARIOS['adaptive'], session_id='it3-escalate',
          strategy_hint='exploratory',
        )
      )
    finally:
      agent.close()

    assert output.status in ('error', 'blocked')
    assert output.decision is not None
    assert output.decision.strategy == 'fallback'
    assert output.action is not None
    kinds = {record.kind for record in output.action.adaptations}
    assert 'strategy_escalation' in kinds


class TestSafety:
  '''Write approval bounds.'''

  def test_unsafe_write_blocked(
    self, settings: Settings, tmp_path: Any,
  ) -> None:
    config = load_cognition_config(settings)
    agent = build_agent(
      settings, config, tmp_path, allow_writes=True,
    )
    try:
      output = agent.run_task(
        CognitiveTaskInput(
          task=SCENARIOS['unsafe'], session_id='it3-unsafe',
          allow_writes=True,
        )
      )
    finally:
      agent.close()

    assert output.status == 'blocked'
    assert output.action is not None
    blocked = [
      audit for audit in output.action.tool_audits if audit.blocked
    ]
    assert blocked
    assert any(
      'not approved' in error for error in output.errors
    )

  def test_read_only_blocks_writes(
    self, settings: Settings, tmp_path: Any,
  ) -> None:
    config = load_cognition_config(settings)
    agent = build_agent(settings, config, tmp_path, allow_writes=False)
    try:
      output = agent.run_task(
        CognitiveTaskInput(
          task=SCENARIOS['unsafe'], session_id='it3-readonly',
          allow_writes=False,
        )
      )
    finally:
      agent.close()

    assert output.status == 'blocked'
    assert any('disabled' in error for error in output.errors)


class TestPersistenceAndMemory:
  '''Execution history, episodes, stats, and traces.'''

  def test_store_records_history(
    self, settings: Settings, tmp_path: Any,
  ) -> None:
    config = load_cognition_config(settings)
    store = AgentStore(tmp_path / 'state.db')
    agent = build_agent(
      settings, config, tmp_path, store=store,
    )
    try:
      output = agent.run_task(
        CognitiveTaskInput(
          task=SCENARIOS['dag_retail'], session_id='it3-persist',
        )
      )
      assert output.status == 'completed'

      events = store.history('it3-persist')
      event_types = [event.event_type for event in events]
      assert event_types[0] == 'task_start'
      assert 'percept_and_strategy' in event_types
      assert event_types[-1] == 'task_finish'

      audits = store.tool_calls('it3-persist')
      assert len(audits) >= 2
      assert all(audit.ok for audit in audits)
    finally:
      agent.close()
      store.close()

  def test_memory_learns_across_runs(
    self, settings: Settings, tmp_path: Any,
  ) -> None:
    config = load_cognition_config(settings)
    memory = MemoryStore(tmp_path / 'memory.db')
    agent = build_agent(settings, config, tmp_path, memory=memory)
    try:
      first = agent.run_task(
        CognitiveTaskInput(
          task=SCENARIOS['dag_retail'], session_id='it3-learn-1',
        )
      )
      second = agent.run_task(
        CognitiveTaskInput(
          task=SCENARIOS['dag_retail'], session_id='it3-learn-2',
        )
      )
      assert first.status == 'completed'
      assert second.status == 'completed'
      assert second.memory is not None
      assert len(second.memory.relevant_episodes) >= 1

      stats = {stat.strategy: stat for stat in memory.strategy_stats()}
      assert stats['conservative'].runs >= 2
      assert memory.get_fact('last_answer.it3-learn-2') is not None
    finally:
      agent.close()
      memory.close()

  def test_traces_written(
    self, settings: Settings, tmp_path: Any,
  ) -> None:
    config = load_cognition_config(settings)
    tracer = Tracer(trace_dir=tmp_path / 'traces')
    agent = build_agent(settings, config, tmp_path, tracer=tracer)
    try:
      output = agent.run_task(
        CognitiveTaskInput(
          task=SCENARIOS['dag_retail'], session_id='it3-trace',
        )
      )
    finally:
      agent.close()

    assert output.status == 'completed'
    files = list((tmp_path / 'traces').glob('*.jsonl'))
    assert len(files) == 1
    content = files[0].read_text(encoding='utf-8')
    assert 'llm.call' in content
    assert 'tool.call' in content


class TestBudgets:
  '''Token budget is enforced.'''

  def test_budget_exhaustion(
    self, settings: Settings, tmp_path: Any,
  ) -> None:
    config = load_cognition_config(settings).model_copy(
      update={'token_budget': 1},
    )
    agent = build_agent(settings, config, tmp_path)
    try:
      output = agent.run_task(
        CognitiveTaskInput(
          task=SCENARIOS['dag_retail'], session_id='it3-budget',
        )
      )
    finally:
      agent.close()

    assert output.status == 'error'
    assert any(
      'token budget' in error for error in output.errors
    )
