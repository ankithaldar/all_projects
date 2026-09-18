#!/usr/bin/env python
# -- coding: utf-8 --

'''Unit tests: planners and plan introspection.'''


from __future__ import annotations

import json
from typing import Any, Callable, Dict, List

import pytest

from agentic_common.gateway_client import MockGateway
from chapter03_cognition.config import CognitionConfig
from chapter03_cognition.llm import ReasoningLLM
from chapter03_cognition.perception import PerceptionLayer
from chapter03_cognition.planner import (
  CodePlanner,
  DagPlanner,
  DecisionLayer,
  PlannerIntrospector,
  fallback_plan,
  topological_waves,
  validate_plan,
)
from chapter03_cognition.schemas import (
  CognitivePlan,
  PlanStep,
  ToolSpec,
)
from chapter03_cognition.strategies import profile_for
from llm_gateway.schemas import GatewayResponse

TOOLS = [
  ToolSpec(name='retail-ops.retail_low_stock_report', read_only=True),
  ToolSpec(name='retail-ops.retail_sales_trend', read_only=True),
  ToolSpec(name='retail-ops.retail_restock_order', read_only=False),
]


def make_llm(
  planner: Callable[[List[Dict[str, Any]]], GatewayResponse],
) -> ReasoningLLM:
  '''Wrap a scripted planner in a ReasoningLLM.

  Args:
    planner: Scripted planner.

  Returns:
    ReasoningLLM.
  '''
  return ReasoningLLM(MockGateway(planner), CognitionConfig())


def text_response(content: str) -> GatewayResponse:
  '''Build a plain text response.

  Args:
    content: Response text.

  Returns:
    GatewayResponse.
  '''
  return GatewayResponse(provider='mock', model='m', content=content)


def percept_for(task: str):
  '''Build a percept for a task.

  Args:
    task: Task text.

  Returns:
    Percept.
  '''
  return PerceptionLayer().perceive(task, 'it-planner')


def diamond_plan() -> CognitivePlan:
  '''Build a diamond-shaped plan: a -> (b, c) -> d.

  Returns:
    CognitivePlan.
  '''
  return CognitivePlan(
    goal='diamond',
    steps=[
      PlanStep(id='a', objective='start'),
      PlanStep(id='b', objective='left', depends_on=['a']),
      PlanStep(id='c', objective='right', depends_on=['a']),
      PlanStep(id='d', objective='join', depends_on=['b', 'c']),
    ],
  )


class TestGraphLayer:
  '''networkx plan validation and ordering.'''

  def test_topological_waves(self) -> None:
    assert topological_waves(diamond_plan()) == [
      ['a'], ['b', 'c'], ['d'],
    ]

  def test_cycle_detected(self) -> None:
    cyclic = CognitivePlan(
      goal='cycle',
      steps=[
        PlanStep(id='a', objective='x', depends_on=['b']),
        PlanStep(id='b', objective='y', depends_on=['a']),
      ],
    )
    with pytest.raises(ValueError):
      topological_waves(cyclic)
    assert any('cycle' in error for error in validate_plan(cyclic))

  def test_step_cap(self) -> None:
    assert validate_plan(diamond_plan(), max_steps=2)


class TestDagPlanner:
  '''JSON DAG planning with fallback.'''

  def test_valid_plan(self) -> None:
    payload = {
      'goal': 'g',
      'steps': [
        {
          'id': 'one', 'objective': 'first', 'kind': 'tool',
          'tool': 'retail-ops.retail_low_stock_report',
        },
        {
          'id': 'two', 'objective': 'second', 'kind': 'tool',
          'tool': 'retail-ops.retail_sales_trend', 'depends_on': ['one'],
        },
      ],
    }

    def planner(_messages):
      return text_response(json.dumps(payload))

    plan, notes = DagPlanner(
      make_llm(planner), CognitionConfig(),
    ).plan('g', percept_for('Check low stock.'), None,
           profile_for('conservative'), TOOLS)
    assert notes == []
    assert plan.source == 'dag'
    assert [step.id for step in plan.steps] == ['one', 'two']

  def test_unknown_tool_becomes_reason_step(self) -> None:
    payload = {
      'goal': 'g',
      'steps': [
        {
          'id': 'one', 'objective': 'call a fantasy system',
          'kind': 'tool', 'tool': 'fantasy.system',
        }
      ],
    }

    def planner(_messages):
      return text_response(json.dumps(payload))

    plan, notes = DagPlanner(
      make_llm(planner), CognitionConfig(),
    ).plan('g', percept_for('Check low stock.'), None,
           profile_for('conservative'), TOOLS)
    assert plan.steps[0].kind == 'reason'
    assert plan.steps[0].tool == ''
    assert any('unknown tool' in note for note in notes)

  def test_invalid_plan_falls_back(self) -> None:
    def planner(_messages):
      return text_response('I cannot plan today.')

    plan, notes = DagPlanner(
      make_llm(planner), CognitionConfig(),
    ).plan('g', percept_for('Check low stock.'), None,
           profile_for('conservative'), TOOLS)
    assert plan.source == 'fallback'
    assert any('fallback' in note for note in notes)

  def test_cyclic_plan_falls_back(self) -> None:
    payload = {
      'goal': 'g',
      'steps': [
        {'id': 'a', 'objective': 'x', 'depends_on': ['b']},
        {'id': 'b', 'objective': 'y', 'depends_on': ['a']},
      ],
    }

    def planner(_messages):
      return text_response(json.dumps(payload))

    plan, notes = DagPlanner(
      make_llm(planner), CognitionConfig(),
    ).plan('g', percept_for('Check low stock.'), None,
           profile_for('conservative'), TOOLS)
    assert plan.source == 'fallback'
    assert any('cycle' in note for note in notes)


class TestCodePlanner:
  '''Agent-written Python planning with validation retry.'''

  SAFE = (
    'def solve(context):\n'
    '    return {"answer": "ok", "used_tools": []}\n'
  )

  def test_valid_code_plan(self) -> None:
    def planner(_messages):
      return text_response(f'```python\n{self.SAFE}\n```')

    plan, notes = CodePlanner(
      make_llm(planner), CognitionConfig(),
    ).plan('g', percept_for('How many units?'), None,
           profile_for('conservative'), TOOLS)
    assert plan.source == 'code'
    assert 'def solve' in plan.code
    assert notes == []

  def test_unsafe_code_falls_back_after_retry(self) -> None:
    calls = {'n': 0}

    def planner(_messages):
      calls['n'] += 1
      return text_response('```python\nimport os\n' + self.SAFE + '\n```')

    plan, notes = CodePlanner(
      make_llm(planner), CognitionConfig(),
    ).plan('g', percept_for('How many units?'), None,
           profile_for('conservative'), TOOLS)
    assert calls['n'] == 2
    assert plan.source == 'fallback'
    assert any('rejected' in note for note in notes)


class TestDecisionLayer:
  '''Planner selection rules.'''

  def test_code_for_computation(self) -> None:
    percept = percept_for(
      'A product sells 30 units per day. Lead time is 4 days. We have 20 '
      'units on hand and want a buffer of 10 units. How many to order?'
    )
    assert DecisionLayer._use_code(
      percept, profile_for('conservative'), 'auto',
    ) is True

  def test_dag_for_write_intent(self) -> None:
    percept = percept_for(
      'Restock store S01 with 30 units of R-101 now.'
    )
    assert DecisionLayer._use_code(
      percept, profile_for('conservative'), 'auto',
    ) is False

  def test_plan_mode_override(self) -> None:
    percept = percept_for('Check low stock.')
    assert DecisionLayer._use_code(
      percept, profile_for('exploratory'), 'code',
    ) is True
    assert DecisionLayer._use_code(
      percept, profile_for('conservative'), 'dag',
    ) is False


class TestIntrospector:
  '''Static plan review.'''

  def test_write_before_read_warning(self) -> None:
    plan = CognitivePlan(
      goal='g',
      steps=[
        PlanStep(
          id='w', objective='write', kind='write',
          tool='retail-ops.retail_restock_order',
        ),
        PlanStep(
          id='r', objective='read', kind='tool',
          tool='retail-ops.retail_low_stock_report',
        ),
      ],
    )
    insight = PlannerIntrospector().introspect(
      plan, percept_for('Restock store S01 with 5 units of R-101.'),
      TOOLS, profile_for('conservative'),
    )
    assert insight.has_writes is True
    assert any('writes before' in warning for warning in insight.warnings)
    assert insight.risk_level == 'high'

  def test_missing_capability_warning(self) -> None:
    plan = CognitivePlan(
      goal='g',
      steps=[PlanStep(id='r', objective='read stock')],
    )
    percept = percept_for('Check sales trend for store S01.')
    insight = PlannerIntrospector().introspect(
      plan, percept, TOOLS, profile_for('conservative'),
    )
    assert 'sales_read' in insight.missing_capabilities

  def test_fallback_plan_low_confidence(self) -> None:
    plan = fallback_plan('g', profile_for('fallback'), 'test')
    insight = PlannerIntrospector().introspect(
      plan, percept_for('Check low stock.'), TOOLS,
      profile_for('fallback'),
    )
    assert plan.source == 'fallback'
    assert insight.confidence <= 0.5

  def test_placeholder_argument_warning(self) -> None:
    plan = CognitivePlan(
      goal='g',
      steps=[
        PlanStep(
          id='trend', objective='read trend', kind='tool',
          tool='retail-ops.retail_sales_trend',
          arguments={'sku': '<sku of the worst item>'},
        )
      ],
    )
    insight = PlannerIntrospector().introspect(
      plan, percept_for('Check the sales trend.'), TOOLS,
      profile_for('conservative'),
    )
    assert any(
      'placeholder' in warning for warning in insight.warnings
    )
