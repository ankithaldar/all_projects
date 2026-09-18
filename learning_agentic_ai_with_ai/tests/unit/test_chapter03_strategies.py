#!/usr/bin/env python
# -- coding: utf-8 --

'''Unit tests: strategy profiles and selection.'''


from __future__ import annotations

from typing import Any

from chapter03_cognition.memory import MemoryLayer, MemoryStore
from chapter03_cognition.perception import PerceptionLayer
from chapter03_cognition.schemas import MemoryContext
from chapter03_cognition.strategies import (
  STRATEGY_PROFILES,
  StrategySelector,
  profile_for,
)


def percept_for(task: str):
  '''Build a percept for a task.

  Args:
    task: Task text.

  Returns:
    Percept.
  '''
  return PerceptionLayer().perceive(task, 'it-strategies')


class TestProfiles:
  '''Profile definitions are coherent.'''

  def test_all_profiles_exist(self) -> None:
    assert set(STRATEGY_PROFILES) == {
      'conservative', 'exploratory', 'fallback',
    }

  def test_fallback_is_degraded(self) -> None:
    fallback = profile_for('fallback')
    assert fallback.retry.allow_code_execution is False
    assert fallback.retry.allow_writes is False
    assert fallback.retry.max_attempts_per_step == 1

  def test_exploratory_is_most_adaptive(self) -> None:
    exploratory = profile_for('exploratory')
    conservative = profile_for('conservative')
    assert (
      exploratory.retry.max_attempts_per_step
      > conservative.retry.max_attempts_per_step
    )
    assert exploratory.retry.tool_switch is True
    assert exploratory.retry.escalate_on_failure is True


class TestSelection:
  '''Selection rules are explainable and ordered.'''

  def test_hint_wins(self) -> None:
    profile, reason = StrategySelector().select(
      percept_for('Check low stock.'), None, 'fallback',
    )
    assert profile.name == 'fallback'
    assert 'hint' in reason

  def test_write_intent_selects_conservative(self) -> None:
    profile, reason = StrategySelector().select(
      percept_for('Restock store S01 with 10 units of R-101.'), None,
    )
    assert profile.name == 'conservative'
    assert 'risk' in reason or 'write' in reason

  def test_memory_recommendation_is_honored(self) -> None:
    memory = MemoryContext(recommended_strategy='exploratory')
    profile, reason = StrategySelector().select(
      percept_for('Check low stock.'), memory,
    )
    assert profile.name == 'exploratory'
    assert 'memory' in reason

  def test_ambiguity_selects_exploratory(self) -> None:
    profile, _ = StrategySelector().select(
      percept_for(
        'Investigate why the site is degraded and find the root cause',
      ),
      None,
    )
    assert profile.name == 'exploratory'

  def test_default_is_conservative(self) -> None:
    profile, reason = StrategySelector().select(
      percept_for('Check which items are below their reorder point.'),
      None,
    )
    assert profile.name == 'conservative'
    assert 'default' in reason

  def test_selector_uses_real_memory(
    self, tmp_path: Any,
  ) -> None:
    store = MemoryStore(tmp_path / 'memory.db')
    try:
      percept = percept_for('Check low stock for store S02.')
      context = MemoryLayer(store).build_context(percept)
      profile, _ = StrategySelector().select(percept, context)
      assert profile.name == 'conservative'
    finally:
      store.close()
