#!/usr/bin/env python
# -- coding: utf-8 --

'''Strategy profiles: how boldly the agent plans, acts, and adapts.

A strategy is a *named operating policy*, not a prompt. Each profile binds
together the knobs that must move together in production:

  - how many attempts a step gets and whether tools may be switched,
  - whether failures may trigger replanning or strategy escalation,
  - whether writes and agent-written code are allowed at all,
  - planner temperature and step budget.

Profiles:

  conservative  verify everything, no exploratory calls, minimal retries,
                no writes. Default for risk-bearing tasks.
  exploratory   multiple attempts, tool switching, bounded replans.
                Default when the task is ambiguous or complex.
  fallback      degraded mode: reason/memory only, no writes, no code,
                small step budget, explicit caveats in the answer.
                Used when primary strategies have a poor track record.

The selector is explainable: hint > risk > memory recommendation >
ambiguity/complexity > default. Every decision is logged with a reason.
'''


from __future__ import annotations

from typing import Dict, Optional, Tuple

from agentic_common.logging import get_logger, log_event
from chapter03_cognition.schemas import (
  MemoryContext,
  Percept,
  RetryPolicy,
  StrategyName,
  StrategyProfile,
)

logger = get_logger(__name__)

STRATEGY_PROFILES: Dict[StrategyName, StrategyProfile] = {
  'conservative': StrategyProfile(
    name='conservative',
    description=(
      'single approved write, verify each step, no retries or switching'
    ),
    retry=RetryPolicy(
      max_attempts_per_step=1,
      max_replans=0,
      tool_switch=False,
      escalate_on_failure=False,
      verify_each_step=True,
      allow_writes=True,
      allow_code_execution=True,
    ),
    prefer_code_plans=True,
    temperature=0.0,
    max_steps=6,
  ),
  'exploratory': StrategyProfile(
    name='exploratory',
    description=(
      'multiple attempts, tool switching, bounded replans, approved writes'
    ),
    retry=RetryPolicy(
      max_attempts_per_step=3,
      max_replans=1,
      tool_switch=True,
      escalate_on_failure=True,
      verify_each_step=True,
      allow_writes=True,
      allow_code_execution=True,
    ),
    prefer_code_plans=False,
    temperature=0.2,
    max_steps=10,
  ),
  'fallback': StrategyProfile(
    name='fallback',
    description=(
      'degraded mode: memory and reasoning only, no writes, explicit caveats'
    ),
    retry=RetryPolicy(
      max_attempts_per_step=1,
      max_replans=0,
      tool_switch=False,
      escalate_on_failure=False,
      verify_each_step=False,
      allow_writes=False,
      allow_code_execution=False,
    ),
    prefer_code_plans=False,
    temperature=0.0,
    max_steps=3,
  ),
}


def profile_for(name: StrategyName) -> StrategyProfile:
  '''Return a strategy profile by name.

  Args:
    name: Strategy name.

  Returns:
    The shared profile instance.
  '''
  return STRATEGY_PROFILES[name]


class StrategySelector:
  '''Explainable selection of a strategy profile for one task.'''

  def select(
    self,
    percept: Percept,
    memory: Optional[MemoryContext] = None,
    hint: Optional[StrategyName] = None,
  ) -> Tuple[StrategyProfile, str]:
    '''Choose a strategy profile.

    Args:
      percept: Perception-layer output.
      memory: Optional memory context (carries a recommendation).
      hint: Optional caller override.

    Returns:
      (profile, reason) pair.
    '''
    profile, reason = self._select(percept, memory, hint)
    log_event(
      logger, 20, 'strategy_selected',
      session_id=percept.session_id,
      strategy=profile.name,
      reason=reason,
    )
    return profile, reason

  def _select(
    self,
    percept: Percept,
    memory: Optional[MemoryContext],
    hint: Optional[StrategyName],
  ) -> Tuple[StrategyProfile, str]:
    '''Apply the selection rules in priority order.

    Args:
      percept: Perception-layer output.
      memory: Optional memory context.
      hint: Optional caller override.

    Returns:
      (profile, reason) pair.
    '''
    if hint is not None:
      return profile_for(hint), 'caller strategy hint'

    if percept.signals.write_intent or percept.signals.risk >= 0.5:
      return (
        profile_for('conservative'),
        'write intent or elevated risk detected',
      )

    if memory is not None and memory.recommended_strategy is not None:
      recommended = memory.recommended_strategy
      return (
        profile_for(recommended),
        f'memory recommendation ({recommended})',
      )

    if (
      percept.signals.ambiguity >= 0.5
      or percept.signals.complexity >= 0.6
    ):
      return (
        profile_for('exploratory'),
        'ambiguous or complex task benefits from exploration',
      )

    return profile_for('conservative'), 'default profile for simple tasks'
