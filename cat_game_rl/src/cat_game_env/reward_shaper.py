from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import numpy as np

from src.core.items import NUM_CRAFTABLE, NUM_ITEMS


class RewardComponent(ABC):
  @abstractmethod
  def compute(
    self, state: Dict, action_info: Dict, next_state: Dict
  ) -> float:
    ...

  @property
  @abstractmethod
  def name(self) -> str:
    ...


class SlotUtilizationReward(RewardComponent):
  @property
  def name(self) -> str:
    return "slot_utilization"

  def compute(self, state: Dict, action_info: Dict, next_state: Dict) -> float:
    active = next_state.get("active_slot_count", 0)
    return active / NUM_CRAFTABLE


class WasteMinimizationReward(RewardComponent):
  @property
  def name(self) -> str:
    return "waste_minimization"

  def compute(self, state: Dict, action_info: Dict, next_state: Dict) -> float:
    targets_remaining = next_state.get("targets_remaining", np.zeros(NUM_ITEMS))
    excess = 0.0
    for i in range(NUM_ITEMS):
      if targets_remaining[i] < 0:
        excess += abs(targets_remaining[i])
    total_targets = np.sum(np.abs(next_state.get("targets_total", np.ones(1))))
    if total_targets == 0:
      return 0.0
    return -excess / max(total_targets, 1.0)


class TargetCompletionReward(RewardComponent):
  @property
  def name(self) -> str:
    return "target_completion"

  def compute(self, state: Dict, action_info: Dict, next_state: Dict) -> float:
    prev_frac = state.get("fraction_complete", 0.0)
    curr_frac = next_state.get("fraction_complete", 0.0)
    progress = (curr_frac - prev_frac) * 10.0

    done_bonus = 100.0 if next_state.get("targets_complete", False) else 0.0
    time_pressure = -0.001

    return progress + done_bonus + time_pressure


class ExcessInventoryPenalty(RewardComponent):
  @property
  def name(self) -> str:
    return "excess_inventory"

  def compute(self, state: Dict, action_info: Dict, next_state: Dict) -> float:
    stash = next_state.get("stash", np.zeros(NUM_ITEMS))
    return -0.01 * np.sum(stash) / 1000.0


class CoinEfficiencyReward(RewardComponent):
  @property
  def name(self) -> str:
    return "coin_efficiency"

  def compute(self, state: Dict, action_info: Dict, next_state: Dict) -> float:
    coins_spent = action_info.get("total_coins_spent", 0)
    applied = action_info.get("applied", {})
    if not applied or coins_spent <= 0:
      return 0.0
    total_items = sum(applied.values())
    return -coins_spent / max(total_items, 1) / 1000.0


class TimeEfficiencyReward(RewardComponent):
  @property
  def name(self) -> str:
    return "time_efficiency"

  def compute(self, state: Dict, action_info: Dict, next_state: Dict) -> float:
    if next_state.get("targets_complete", False):
      max_ticks = next_state.get("max_ticks", 2016)
      tick = next_state.get("tick", max_ticks)
      return 10.0 * (1.0 - tick / max(max_ticks, 1))
    return 0.0


class BatchOptimizationReward(RewardComponent):
  @property
  def name(self) -> str:
    return "batch_optimization"

  def compute(self, state: Dict, action_info: Dict, next_state: Dict) -> float:
    applied = action_info.get("applied", {})
    if not applied:
      return 0.0
    batch_sizes = list(applied.values())
    avg_batch = sum(batch_sizes) / len(batch_sizes)
    return avg_batch / 20.0


class RewardShaper:
  def __init__(self, weights: Dict[str, float] | None = None):
    self._components: List[Tuple[float, RewardComponent]] = []

    default_weights = {
      "slot_utilization": 0.3,
      "waste_minimization": 0.1,
      "target_completion": 1.0,
      "excess_inventory": 0.05,
      "coin_efficiency": 0.4,
      "time_efficiency": 0.5,
      "batch_optimization": 0.2,
    }
    w = weights or default_weights

    self._components.append(
      (w.get("slot_utilization", 0.3), SlotUtilizationReward())
    )
    self._components.append(
      (w.get("waste_minimization", 0.1), WasteMinimizationReward())
    )
    self._components.append(
      (w.get("target_completion", 1.0), TargetCompletionReward())
    )
    self._components.append(
      (w.get("excess_inventory", 0.05), ExcessInventoryPenalty())
    )
    self._components.append(
      (w.get("coin_efficiency", 0.4), CoinEfficiencyReward())
    )
    self._components.append(
      (w.get("time_efficiency", 0.5), TimeEfficiencyReward())
    )
    self._components.append(
      (w.get("batch_optimization", 0.2), BatchOptimizationReward())
    )

  def compute(
    self,
    state: Dict,
    action_info: Dict,
    next_state: Dict,
  ) -> Tuple[float, Dict[str, float]]:
    total = 0.0
    breakdown = {}
    for weight, component in self._components:
      value = component.compute(state, action_info, next_state)
      weighted = weight * value
      breakdown[component.name] = value
      breakdown[f"{component.name}_weighted"] = weighted
      total += weighted
    breakdown["total"] = total
    return total, breakdown
