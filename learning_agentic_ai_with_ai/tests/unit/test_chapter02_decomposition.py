#!/usr/bin/env python
# -- coding: utf-8 --

'''Unit tests: dependency-aware planning and plan execution.'''


from __future__ import annotations

import json

from agentic_common.gateway_client import MockGateway
from chapter02_planning.config import PlanningConfig
from chapter02_planning.llm import ReasoningLLM
from chapter02_planning.patterns.decomposition import (
  PlanExecutor,
  TaskPlanner,
  plan_to_graph,
  topological_waves,
  validate_plan,
)
from chapter02_planning.prompting import PLAN_SYSTEM_TEMPLATE
from chapter02_planning.schemas import (
  NodeResult,
  Plan,
  PlanNode,
  ToolSpec,
)


def make_llm(planner) -> ReasoningLLM:
  '''Wrap a scripted planner in a ReasoningLLM.

  Args:
    planner: Callable(messages) -> GatewayResponse.

  Returns:
    ReasoningLLM.
  '''
  return ReasoningLLM(MockGateway(planner), PlanningConfig())


def diamond_plan() -> Plan:
  '''Build a diamond-shaped plan: a -> (b, c) -> d.

  Returns:
    Plan instance.
  '''
  return Plan(
    goal='diamond',
    nodes=[
      PlanNode(id='a', objective='start'),
      PlanNode(id='b', objective='left', depends_on=['a']),
      PlanNode(id='c', objective='right', depends_on=['a']),
      PlanNode(id='d', objective='join', depends_on=['b', 'c']),
    ],
  )


class TestGraphLayer:
  '''networkx graph construction and ordering.'''

  def test_plan_to_graph_edges(self) -> None:
    graph = plan_to_graph(diamond_plan())
    assert set(graph.edges()) == {
      ('a', 'b'), ('a', 'c'), ('b', 'd'), ('c', 'd'),
    }

  def test_topological_waves(self) -> None:
    assert topological_waves(diamond_plan()) == [['a'], ['b', 'c'], ['d']]

  def test_cycle_detection(self) -> None:
    cyclic = Plan(
      goal='cycle',
      nodes=[
        PlanNode(id='a', objective='x', depends_on=['b']),
        PlanNode(id='b', objective='y', depends_on=['a']),
      ],
    )
    errors = validate_plan(cyclic)
    assert any('cycle' in error for error in errors)

  def test_validate_plan_caps(self) -> None:
    plan = diamond_plan()
    assert validate_plan(plan, max_nodes=3) != []
    assert validate_plan(plan, max_nodes=8, max_waves=2) != []
    assert validate_plan(plan, max_nodes=8, max_waves=6) == []


class TestTaskPlanner:
  '''Structured plan generation with safe fallback.'''

  def test_valid_plan_from_model(self) -> None:
    payload = {
      'goal': 'g',
      'nodes': [
        {'id': 'one', 'objective': 'first', 'depends_on': []},
        {'id': 'two', 'objective': 'second', 'depends_on': ['one']},
      ],
    }

    def planner(_messages):
      from llm_gateway.schemas import GatewayResponse
      return GatewayResponse(
        provider='mock', model='m', content=json.dumps(payload),
      )

    plan, errors = TaskPlanner(
      make_llm(planner), PlanningConfig(),
    ).plan('g', tools=[ToolSpec(name='srv.t', read_only=True)])
    assert errors == []
    assert len(plan.nodes) == 2

  def test_invalid_plan_falls_back(self) -> None:
    def planner(_messages):
      from llm_gateway.schemas import GatewayResponse
      return GatewayResponse(
        provider='mock', model='m', content='I cannot plan today.',
      )

    plan, errors = TaskPlanner(
      make_llm(planner), PlanningConfig(),
    ).plan('g', tools=[ToolSpec(name='srv.t')])
    assert any('fallback' in error for error in errors)
    assert len(plan.nodes) == 1
    assert plan.nodes[0].route == 'react'

  def test_overflowing_plan_falls_back(self) -> None:
    nodes = [
      {'id': f'n{index}', 'objective': 'step', 'depends_on': []}
      for index in range(20)
    ]

    def planner(_messages):
      from llm_gateway.schemas import GatewayResponse
      return GatewayResponse(
        provider='mock', model='m',
        content=json.dumps({'nodes': nodes}),
      )

    plan, errors = TaskPlanner(
      make_llm(planner), PlanningConfig(),
    ).plan('g')
    assert errors
    assert len(plan.nodes) == 1

  def test_plan_system_template_renders(self) -> None:
    rendered = PLAN_SYSTEM_TEMPLATE.render(max_nodes=5)
    assert '5' in rendered

  def test_suggested_tools_filtered_to_catalog(self) -> None:
    payload = {
      'goal': 'g',
      'nodes': [
        {
          'id': 'one',
          'objective': 'pull the trend',
          'suggested_tools': [
            'analytics platform', 'sales_trend',
            'retail-ops.retail_sales_trend',
          ],
        }
      ],
    }

    def planner(_messages):
      from llm_gateway.schemas import GatewayResponse
      return GatewayResponse(
        provider='mock', model='m', content=json.dumps(payload),
      )

    plan, errors = TaskPlanner(
      make_llm(planner), PlanningConfig(),
    ).plan(
      'g',
      tools=[
        ToolSpec(name='retail-ops.retail_sales_trend', read_only=True),
        ToolSpec(name='retail-ops.retail_low_stock_report', read_only=True),
      ],
    )
    assert errors == []
    assert plan.nodes[0].suggested_tools == [
      'retail-ops.retail_sales_trend',
    ]


class TestPlanExecutor:
  '''Execution order, partial completion, and skip propagation.'''

  def test_all_nodes_complete(self) -> None:
    calls = []

    def runner(node, prior):
      calls.append(node.id)
      return NodeResult(node_id=node.id, status='completed', output=node.id)

    report = PlanExecutor(runner, PlanningConfig()).execute(diamond_plan())
    assert report.status == 'completed'
    assert calls == ['a', 'b', 'c', 'd']
    assert report.waves == [['a'], ['b', 'c'], ['d']]

  def test_failed_node_skips_dependents(self) -> None:
    def runner(node, prior):
      if node.id == 'b':
        return NodeResult(
          node_id='b', status='failed', errors=['boom'],
        )
      return NodeResult(node_id=node.id, status='completed', output='ok')

    report = PlanExecutor(runner, PlanningConfig()).execute(diamond_plan())
    results = {result.node_id: result for result in report.results}
    assert results['a'].status == 'completed'
    assert results['b'].status == 'failed'
    assert results['c'].status == 'completed'
    assert results['d'].status == 'skipped'
    assert 'dependencies not completed' in results['d'].skipped_reason
    assert report.status == 'needs_review'
    assert 'boom' in report.errors

  def test_crashing_runner_fails_node_only(self) -> None:
    def runner(node, prior):
      if node.id == 'a':
        raise RuntimeError('exploded')
      return NodeResult(node_id=node.id, status='completed')

    report = PlanExecutor(runner, PlanningConfig()).execute(diamond_plan())
    results = {result.node_id: result for result in report.results}
    assert results['a'].status == 'failed'
    assert any('exploded' in error for error in report.errors)

  def test_usage_aggregation(self) -> None:
    def runner(node, prior):
      return NodeResult(
        node_id=node.id, status='completed',
        usage={
          'input_tokens': 10, 'output_tokens': 2, 'total_tokens': 12,
        },
      )

    report = PlanExecutor(runner, PlanningConfig()).execute(diamond_plan())
    assert report.usage['total_tokens'] == 48
