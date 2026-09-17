#!/usr/bin/env python
# -- coding: utf-8 --

'''Reasoning router: when to let the model reason vs. enforce structure.

The chapter's central engineering question is not "can the model reason?"
but "which execution shape does this task deserve?". Letting a model
free-form reason about a simple lookup wastes tokens and adds variance;
forcing rigid JSON on a genuinely multi-step problem produces confident
nonsense. This module encodes the choice as an explainable, testable
rule-based policy instead of an intuition:

  direct  - no visible reasoning: lookups, formatting, single facts.
  cot     - visible step-labeled reasoning: calculations and comparisons
            that need no external data.
  react   - interleaved reason/act/observe: the answer depends on live data
            or on actions with side effects.
  plan    - dependency-aware decomposition: several interleaved objectives,
            risky writes, or parallel information needs.

A caller hint (`prefer_route`) always wins, so product teams can pin the
route for latency or compliance reasons. Every decision carries feature
scores and a rationale - decisions are auditable, not vibes.
'''


from __future__ import annotations

from typing import Any, Dict, List, Optional

from agentic_common.logging import get_logger, log_event
from chapter02_planning.config import PlanningConfig
from chapter02_planning.schemas import (
  PlanNode,
  PolicyDecision,
  ReasoningRoute,
)

logger = get_logger(__name__)

_TOOL_HINTS = (
  'check', 'find', 'look up', 'lookup', 'report', 'status', 'inventory',
  'stock', 'sales', 'trend', 'site', 'network', 'degraded',
  'dispatch', 'restock', 'replenish', 'order', 'sku', 'store',
  'technician', 'outage', 'latency', 'battery', 'customer',
)

_WRITE_HINTS = (
  'dispatch', 'restock', 'replenish', 'place an order', 'create order',
  'cancel', 'update', 'delete', 'refund', 'approve', 'escalate',
)

_MULTI_STEP_HINTS = (
  ' and then ', ' then ', ' after that', ' finally', ' first ',
  ' next ', ' followed by', 'plus ', ' as well as', ' audit',
  ' campaign', ' across all', ' every store', ' all stores',
  ' all sites', ' plan', ' roadmap', ' sequence', ' multiple',
)

_NUMERIC_HINTS = (
  'per day', 'per week', 'per month', 'lead time', 'average', 'total',
  ' within ', ' at least', ' at most', ' percent', '%', 'ratio',
  ' how many', ' how much', ' cost', ' budget', ' rate',
)

_AMBIGUITY_HINTS = (
  'figure out', 'investigate', 'diagnose', 'why ', 'root cause',
  'unclear', 'seems', 'might', 'possibly', 'recommend', 'decide',
  'best ', 'optimize',
)

_CALCULATION_HINTS = (
  'calculate', 'compute', 'total', 'average', 'how many', 'estimate',
  'size the', 'sized to', 'project', 'forecast', 'convert',
)


def _score(text: str, phrases: tuple[str, ...], weight: int = 1) -> int:
  '''Count phrase occurrences in a lowercased task.

  Args:
    text: Lowercased task text.
    phrases: Phrases to look for.
    weight: Points per matched phrase.

  Returns:
    Integer score.
  '''
  return sum(weight for phrase in phrases if phrase in text)


def choose_node_route(
  node: PlanNode,
  has_tools: bool,
) -> ReasoningRoute:
  '''Choose a reasoning route for one plan node.

  Node-level routing keeps plans cheap: read-only gathering steps run as
  ReAct when tools are suggested, calculation steps as CoT, and everything
  else directly. An explicit `node.route` from the planner always wins.

  Args:
    node: Plan node about to execute.
    has_tools: Whether any tools are available.

  Returns:
    The route for this node.
  '''
  if node.route is not None:
    return node.route
  if has_tools and node.suggested_tools:
    return 'react'
  objective = node.objective.lower()
  if any(hint in objective for hint in _CALCULATION_HINTS):
    return 'cot'
  if any(char.isdigit() for char in objective) and len(objective) > 60:
    return 'cot'
  return 'direct'


class ReasoningPolicy:
  '''Explainable rule-based router from task text to reasoning route.'''

  def __init__(self, config: PlanningConfig) -> None:
    '''Initialize the policy.

    Args:
      config: Chapter 2 configuration.
    '''
    self._config = config

  def decide(
    self,
    task: str,
    has_tools: bool = True,
    route_hint: Optional[ReasoningRoute] = None,
  ) -> PolicyDecision:
    '''Decide which reasoning route a task deserves.

    Args:
      task: Natural-language task.
      has_tools: Whether tools are available to the agent.
      route_hint: Optional caller override.

    Returns:
      A transparent PolicyDecision with feature scores and rationale.
    '''
    if route_hint is not None:
      decision = PolicyDecision(
        route=route_hint,
        rationale='caller hint overrides rule-based routing',
        scores={},
        confidence=1.0,
        forced=True,
      )
      self._log(task, decision)
      return decision

    raw_task = task or ''
    text = f' {raw_task.lower()} '
    scores = self._scores(text, has_tools)
    route, rationale = self._route_from_scores(scores, has_tools)
    confidence = self._confidence(route, scores)

    decision = PolicyDecision(
      route=route,
      rationale=rationale,
      scores=scores,
      confidence=confidence,
      forced=False,
    )
    self._log(task, decision)
    return decision

  # ------------------------------------------------------------------
  # Internals
  # ------------------------------------------------------------------

  def _scores(self, text: str, has_tools: bool) -> Dict[str, float]:
    '''Compute interpretable feature scores for a task.

    Args:
      text: Lowercased task text padded with spaces.
      has_tools: Whether tools are available.

    Returns:
      Mapping of feature name to score.
    '''
    tool_need = _score(text, _TOOL_HINTS, 2)
    write_risk = _score(text, _WRITE_HINTS, 3)
    multi_step = _score(text, _MULTI_STEP_HINTS, 2)
    numeric = _score(text, _NUMERIC_HINTS, 2)
    ambiguity = _score(text, _AMBIGUITY_HINTS)

    if ' and ' in text:
      multi_step += 1
    if text.count('?') > 1:
      multi_step += 1
    if not has_tools:
      tool_need = 0
      write_risk = 0

    return {
      'tool_need': float(tool_need),
      'write_risk': float(write_risk),
      'multi_step': float(multi_step),
      'numeric': float(numeric),
      'ambiguity': float(ambiguity),
    }

  def _route_from_scores(
    self,
    scores: Dict[str, float],
    has_tools: bool,
  ) -> tuple[ReasoningRoute, str]:
    '''Apply the routing rules to feature scores.

    Args:
      scores: Feature scores.
      has_tools: Whether tools are available.

    Returns:
      (route, rationale) pair.
    '''
    tool_need = scores['tool_need']
    write_risk = scores['write_risk']
    multi_step = scores['multi_step']
    numeric = scores['numeric']
    ambiguity = scores['ambiguity']

    if tool_need > 0 and multi_step >= 2:
      return 'plan', (
        'multiple objectives need live data: decompose into a '
        'dependency-aware plan'
      )

    if write_risk > 0 and multi_step >= 1:
      return 'plan', (
        'task includes a state-changing action plus other steps: plan '
        'first, then execute with approval gates'
      )

    if write_risk > 0 and has_tools:
      return 'react', (
        'task requires a write action: interleave reasoning with '
        'guarded tool calls'
      )

    if tool_need > 0:
      return 'react', (
        'answer depends on live tool data: use the ReAct loop'
      )

    if numeric >= 2 or (numeric >= 1 and ambiguity >= 1):
      return 'cot', (
        'calculation or comparison present: step-labeled reasoning '
        'reduces arithmetic errors'
      )

    if multi_step >= 3 or ambiguity >= 2:
      return 'cot', (
        'task needs explicit assumptions: visible reasoning for auditability'
      )

    return 'direct', (
      'simple, self-contained request: extra reasoning adds cost without '
      'accuracy'
    )

  def _confidence(
    self,
    route: ReasoningRoute,
    scores: Dict[str, float],
  ) -> float:
    '''Estimate confidence from the strength of supporting features.

    Args:
      route: Chosen route.
      scores: Feature scores.

    Returns:
      Confidence in the 0..1 range.
    '''
    if route == 'plan':
      support = scores['multi_step'] + scores['write_risk']
    elif route == 'react':
      support = scores['tool_need'] + scores['write_risk']
    elif route == 'cot':
      support = scores['numeric'] + scores['ambiguity']
    else:
      support = max(1.0, 4.0 - sum(scores.values()) / 4.0)

    return max(0.3, min(1.0, 0.4 + 0.1 * support))

  def _log(self, task: str, decision: PolicyDecision) -> None:
    '''Emit a structured routing event.

    Args:
      task: Original task text.
      decision: The routing decision.
    '''
    log_event(
      logger,
      20,
      'route_decision',
      route=decision.route,
      confidence=round(decision.confidence, 2),
      forced=decision.forced,
      scores=decision.scores,
      task_preview=task[:120],
    )


def route_explanations() -> List[Dict[str, Any]]:
  '''Describe the routing rules for docs and debugging.

  Returns:
    List of rule descriptions in evaluation order.
  '''
  return [
    {
      'order': 1,
      'when': 'tool_need > 0 and multi_step >= 2',
      'route': 'plan',
    },
    {
      'order': 2,
      'when': 'write_risk > 0 and multi_step >= 1',
      'route': 'plan',
    },
    {
      'order': 3,
      'when': 'write_risk > 0 and tools available',
      'route': 'react',
    },
    {
      'order': 4,
      'when': 'tool_need > 0',
      'route': 'react',
    },
    {
      'order': 5,
      'when': 'numeric >= 2 or (numeric >= 1 and ambiguity >= 1)',
      'route': 'cot',
    },
    {
      'order': 6,
      'when': 'multi_step >= 3 or ambiguity >= 2',
      'route': 'cot',
    },
    {'order': 7, 'when': 'otherwise', 'route': 'direct'},
  ]
