#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Composable reward components and shaper.'''

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import numpy as np

from src.core.items import NUM_CRAFTABLE, NUM_ITEMS


class RewardComponent(ABC):
  '''Abstract single reward component.'''
  @abstractmethod
  def compute(
    self, state: Dict, action_info: Dict, next_state: Dict
  ) -> float:
    '''Compute component value.'''

  @property
  @abstractmethod
  def name(self) -> str:
    '''Return component name.'''


class SlotUtilizationReward(RewardComponent):
  '''Reward component: slot_utilization.'''
  @property
  def name(self) -> str:
    '''Return component name.'''
    return 'slot_utilization'

  def compute(self, state: Dict, action_info: Dict, next_state: Dict) -> float:
    '''Compute component value.'''
    active = next_state.get('active_slot_count', 0)
    return active / NUM_CRAFTABLE


class WasteMinimizationReward(RewardComponent):
  '''Reward component: waste_minimization.'''
  @property
  def name(self) -> str:
    '''Return component name.'''
    return 'waste_minimization'

  def compute(self, state: Dict, action_info: Dict, next_state: Dict) -> float:
    '''Compute component value.'''
    targets_remaining = next_state.get('targets_remaining', np.zeros(NUM_ITEMS))
    excess = 0.0
    for i in range(NUM_ITEMS):
      if targets_remaining[i] < 0:
        excess += abs(targets_remaining[i])
    total_targets = np.sum(np.abs(next_state.get('targets_total', np.ones(1))))
    if total_targets == 0:
      return 0.0
    return -excess / max(total_targets, 1.0)


class TargetCompletionReward(RewardComponent):
  '''Reward component: target_completion.'''
  @property
  def name(self) -> str:
    '''Return component name.'''
    return 'target_completion'

  def compute(self, state: Dict, action_info: Dict, next_state: Dict) -> float:
    '''Compute component value.'''
    prev_frac = state.get('fraction_complete', 0.0)
    curr_frac = next_state.get('fraction_complete', 0.0)
    progress = (curr_frac - prev_frac) * 10.0

    done_bonus = 100.0 if next_state.get('targets_complete', False) else 0.0
    time_pressure = -0.001

    return progress + done_bonus + time_pressure


class ExcessInventoryPenalty(RewardComponent):
  '''Reward component: excess_inventory.'''
  @property
  def name(self) -> str:
    '''Return component name.'''
    return 'excess_inventory'

  def compute(self, state: Dict, action_info: Dict, next_state: Dict) -> float:
    '''Compute component value.'''
    stash = next_state.get('stash', np.zeros(NUM_ITEMS))
    return -0.01 * np.sum(stash) / 1000.0


class CoinEfficiencyReward(RewardComponent):
  '''Reward component: coin_efficiency.'''
  @property
  def name(self) -> str:
    '''Return component name.'''
    return 'coin_efficiency'

  def compute(self, state: Dict, action_info: Dict, next_state: Dict) -> float:
    '''Compute component value.'''
    coins_spent = action_info.get('total_coins_spent', 0)
    applied = action_info.get('applied', {})
    if not applied or coins_spent <= 0:
      return 0.0
    total_items = sum(applied.values())
    return -coins_spent / max(total_items, 1) / 1000.0


class TimeEfficiencyReward(RewardComponent):
  '''Reward component: time_efficiency.'''
  @property
  def name(self) -> str:
    '''Return component name.'''
    return 'time_efficiency'

  def compute(self, state: Dict, action_info: Dict, next_state: Dict) -> float:
    '''Compute component value.'''
    if next_state.get('targets_complete', False):
      max_ticks = next_state.get('max_ticks', 2016)
      tick = next_state.get('tick', max_ticks)
      return 10.0 * (1.0 - tick / max(max_ticks, 1))
    return 0.0


class BatchOptimizationReward(RewardComponent):
  '''Reward component: batch_optimization.'''
  @property
  def name(self) -> str:
    '''Return component name.'''
    return 'batch_optimization'

  def compute(self, state: Dict, action_info: Dict, next_state: Dict) -> float:
    '''Compute component value.'''
    applied = action_info.get('applied', {})
    if not applied:
      return 0.0
    batch_sizes = list(applied.values())
    avg_batch = sum(batch_sizes) / len(batch_sizes)
    return avg_batch / 20.0


class RewardShaper:
  '''Weighted sum of reward components.'''
  def __init__(self, weights: Dict[str, float] | None = None):
    '''Initialize with optional weight overrides.'''
    self._components: List[Tuple[float, RewardComponent]] = []

    default_weights = {
      'slot_utilization': 0.3,
      'waste_minimization': 0.1,
      'target_completion': 1.0,
      'excess_inventory': 0.05,
      'coin_efficiency': 0.4,
      'time_efficiency': 0.5,
      'batch_optimization': 0.2,
    }
    w = weights or default_weights

    self._components.append(
      (w.get('slot_utilization', 0.3), SlotUtilizationReward())
    )
    self._components.append(
      (w.get('waste_minimization', 0.1), WasteMinimizationReward())
    )
    self._components.append(
      (w.get('target_completion', 1.0), TargetCompletionReward())
    )
    self._components.append(
      (w.get('excess_inventory', 0.05), ExcessInventoryPenalty())
    )
    self._components.append(
      (w.get('coin_efficiency', 0.4), CoinEfficiencyReward())
    )
    self._components.append(
      (w.get('time_efficiency', 0.5), TimeEfficiencyReward())
    )
    self._components.append(
      (w.get('batch_optimization', 0.2), BatchOptimizationReward())
    )

  def compute(
    self,
    state: Dict,
    action_info: Dict,
    next_state: Dict,
  ) -> Tuple[float, Dict[str, float]]:
    '''Compute weighted total and breakdown.'''
    total = 0.0
    breakdown = {}
    for weight, component in self._components:
      value = component.compute(state, action_info, next_state)
      weighted = weight * value
      breakdown[component.name] = value
      breakdown[f'{component.name}_weighted'] = weighted
      total += weighted
    breakdown['total'] = total
    return total, breakdown
