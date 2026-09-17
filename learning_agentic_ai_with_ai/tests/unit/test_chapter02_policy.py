#!/usr/bin/env python
# -- coding: utf-8 --

'''Unit tests: the reasoning router (reason vs enforce structure).'''


from __future__ import annotations

from chapter02_planning.config import PlanningConfig
from chapter02_planning.policy import (
  ReasoningPolicy,
  choose_node_route,
  route_explanations,
)
from chapter02_planning.schemas import PlanNode

DIRECT_TASK = 'Convert 7 days into hours.'
COT_TASK = (
  'A product sells 30 units per day. Lead time is 4 days. We have 20 '
  'units in the warehouse and want a buffer of 10 units. How many units '
  'should we buy?'
)
REACT_TASK = (
  'Check which items are below their reorder point and inspect the '
  'sales trend of the worst one.'
)
PLAN_TASK = (
  'Audit all stores for low-stock items, check the sales trend of the '
  'most critical item, and then place a restock order sized to one week '
  'of demand.'
)


def make_policy() -> ReasoningPolicy:
  '''Build a policy with default config.

  Returns:
    ReasoningPolicy instance.
  '''
  return ReasoningPolicy(PlanningConfig())


class TestRouteSelection:
  '''Tasks route to the expected reasoning shape.'''

  def test_direct_for_simple_conversion(self) -> None:
    decision = make_policy().decide(DIRECT_TASK)
    assert decision.route == 'direct'
    assert decision.forced is False

  def test_cot_for_arithmetic(self) -> None:
    decision = make_policy().decide(COT_TASK)
    assert decision.route == 'cot'
    assert decision.scores['numeric'] > 0

  def test_react_for_single_data_lookup(self) -> None:
    decision = make_policy().decide(REACT_TASK)
    assert decision.route == 'react'
    assert decision.scores['tool_need'] > 0

  def test_plan_for_multi_objective_campaign(self) -> None:
    decision = make_policy().decide(PLAN_TASK)
    assert decision.route == 'plan'
    assert decision.scores['multi_step'] >= 2

  def test_hint_overrides_policy(self) -> None:
    decision = make_policy().decide(DIRECT_TASK, route_hint='cot')
    assert decision.route == 'cot'
    assert decision.forced is True
    assert decision.confidence == 1.0

  def test_no_tools_zeroes_tool_need(self) -> None:
    decision = make_policy().decide(PLAN_TASK, has_tools=False)
    assert decision.route != 'plan'
    assert decision.scores['tool_need'] == 0
    assert decision.scores['write_risk'] == 0

  def test_rationale_is_populated(self) -> None:
    decision = make_policy().decide(REACT_TASK)
    assert decision.rationale


class TestNodeRouting:
  '''Per-node route selection inside a plan.'''

  def test_explicit_route_wins(self) -> None:
    node = PlanNode(
      id='a', objective='anything', route='direct',
      suggested_tools=['srv.tool'],
    )
    assert choose_node_route(node, has_tools=True) == 'direct'

  def test_tools_suggested_use_react(self) -> None:
    node = PlanNode(
      id='a', objective='pull data', suggested_tools=['srv.tool'],
    )
    assert choose_node_route(node, has_tools=True) == 'react'

  def test_numeric_objective_uses_cot(self) -> None:
    node = PlanNode(
      id='a',
      objective=(
        'compute the total weekly demand from the average sales '
        'numbers provided above in the report and then summarise'
      ),
    )
    assert choose_node_route(node, has_tools=False) == 'cot'

  def test_defaults_to_direct(self) -> None:
    node = PlanNode(id='a', objective='say hello')
    assert choose_node_route(node, has_tools=False) == 'direct'

  def test_route_explanations_order(self) -> None:
    rules = route_explanations()
    assert [rule['order'] for rule in rules] == list(
      range(1, len(rules) + 1)
    )
    assert rules[-1]['route'] == 'direct'
