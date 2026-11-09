#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Reward shaping strategies for the multi-agent environment.'''

from __future__ import annotations

import statistics
from typing import Protocol

from world_of_supply.rl.agents import (
    consumer_agent_id,
    is_retailer,
    producer_agent_id,
)


class RewardShaper(Protocol):
  '''Strategy turning step outcomes into per-agent rewards.

  Implementations receive per-facility balance sheets for one tick and
  produce one reward per RL agent.

  Attributes:
    None: Protocol; implementations hold their own state.
  '''

  def shape(
      self,
      step_balance_sheets: dict[str, object],
      iteration: int,
      n_iterations: int,
  ) -> dict[str, float]:
    '''Compute rewards for both agents of every facility.

    Args:
      step_balance_sheets: Per-facility balance sheets of the tick.
      iteration: Current training iteration (curriculum input).
      n_iterations: Total planned training iterations.

    Returns:
      dict[str, float]: Reward keyed by agent id.
    '''
    ...


class RetailerProfitRewardShaper:
  '''Blends retailer revenue and global profit with a curriculum ramp.

  Early training rewards only mean retailer revenue; as iterations progress,
  mean total profit across all facilities is blended in up to weight 0.8.
  Each agent then mixes this global signal with its own facility result:

  ``reward(agent) = w_role * global + (1 - w_role) * own_total``

  Attributes:
    producer_global_weight: Weight of the global term for producers.
    consumer_global_weight: Weight of the global term for consumers.
  '''

  def __init__(
      self,
      producer_global_weight: float = 0.9,
      consumer_global_weight: float = 0.9,
  ) -> None:
    '''Configure role-specific trust in the global signal.

    Args:
      producer_global_weight: Weight of the global term for producers.
      consumer_global_weight: Weight of the global term for consumers.

    Returns:
      None: The shaper is mutated in place.
    '''
    self.producer_global_weight = producer_global_weight
    self.consumer_global_weight = consumer_global_weight

  def _curriculum_weight(self, iteration: int, n_iterations: int) -> float:
    '''Compute the ramping blend between retail revenue and total profit.

    Args:
      iteration: Current iteration.
      n_iterations: Total iterations.

    Returns:
      float: Blend weight growing linearly to 0.8.
    '''
    progress = iteration / n_iterations if n_iterations > 0 else 0.0
    return 0.8 * progress

  def shape(
      self,
      step_balance_sheets: dict[str, 'BalanceSheetLike'],
      iteration: int,
      n_iterations: int,
  ) -> dict[str, float]:
    '''Compute blended rewards for every facility's two agents.

    Args:
      step_balance_sheets: Per-facility balance sheets; keys must contain
        the facility class name and end with producer/consumer suffixes.
      iteration: Current training iteration.
      n_iterations: Total planned iterations.

    Returns:
      dict[str, float]: Rewards keyed by agent id.
    '''
    retailer_totals = [
        sheet.profit
        for facility_id, sheet in step_balance_sheets.items()
        if is_retailer(facility_id)
    ]
    own_totals = {
        facility_id: sheet.total()
        for facility_id, sheet in step_balance_sheets.items()
    }
    if retailer_totals:
      mean_retail_profit = statistics.mean(retailer_totals)
    else:
      mean_retail_profit = 0.0
    if own_totals:
      mean_total_profit = statistics.mean(own_totals.values())
    else:
      mean_total_profit = 0.0

    w = self._curriculum_weight(iteration, n_iterations)
    global_signal = (1 - w) * mean_retail_profit + w * mean_total_profit

    rewards: dict[str, float] = {}
    for facility_id, own_total in own_totals.items():
      producer_reward = self.producer_global_weight * global_signal
      producer_reward += (1 - self.producer_global_weight) * own_total
      consumer_reward = self.consumer_global_weight * global_signal
      consumer_reward += (1 - self.consumer_global_weight) * own_total
      rewards[producer_agent_id(facility_id)] = producer_reward
      rewards[consumer_agent_id(facility_id)] = consumer_reward
    return rewards


class BalanceSheetLike(Protocol):
  '''Structural type mirroring :class:`world_of_supply.economy.BalanceSheet`.

  Attributes:
    profit: Non-negative profit of one step sheet.
  '''

  profit: int

  def total(self) -> int:
    '''Return the net monetary effect.

    Returns:
      int: ``profit + loss`` net effect of the sheet.
    '''
    ...
