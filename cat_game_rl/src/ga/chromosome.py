from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np

from src.core.items import (
  CraftingTree, ItemId, CRAFTABLE_ITEM_IDS, NUM_CRAFTABLE, BASE_ITEM_IDS,
)
from src.core.inventory import Stash
from src.core.coin_generator import CoinGenerator
from src.core.slot_scheduler import SlotScheduler
from src.core.cost_calculator import CostCalculator
from src.core.target_provider import TargetProvider

MAX_TICKS = 2016
PENALTY_TICK = MAX_TICKS + 1


class Chromosome:
  @staticmethod
  def create_random(density: float = 0.05, max_batch: int = 10) -> np.ndarray:
    genes = np.zeros((MAX_TICKS, NUM_CRAFTABLE), dtype=np.uint8)
    mask = np.random.random((MAX_TICKS, NUM_CRAFTABLE)) < density
    genes[mask] = np.random.randint(1, max_batch + 1, size=mask.sum()).astype(
      np.uint8
    )
    return genes

  @staticmethod
  def decode(genes: np.ndarray) -> List[Dict[int, int]]:
    schedule = []
    for t in range(genes.shape[0]):
      tick_actions = {}
      for i in range(genes.shape[1]):
        if genes[t, i] > 0:
          tick_actions[CRAFTABLE_ITEM_IDS[i]] = int(genes[t, i])
      schedule.append(tick_actions)
    return schedule

  @staticmethod
  def evaluate(
    genes: np.ndarray,
    crafting_tree: CraftingTree,
    targets: Dict[ItemId, int],
  ) -> Tuple[float, float, float, float]:
    stash = Stash()
    coins = CoinGenerator(0)
    slots = SlotScheduler(crafting_tree)
    delivered = {k: 0 for k in targets}
    total_cost = 0.0
    completion_tick = PENALTY_TICK
    total_items_produced = 0
    total_batches = 0

    for t in range(MAX_TICKS):
      coins.tick()

      for base_id in BASE_ITEM_IDS:
        deficit = 9999 - stash.get(base_id)
        if deficit > 0:
          stash.add(base_id, deficit)

      for i in range(NUM_CRAFTABLE):
        batch_size = int(genes[t, i])
        if batch_size <= 0:
          continue

        item_id = ItemId(CRAFTABLE_ITEM_IDS[i])
        if slots.is_busy(item_id):
          continue

        recipe = crafting_tree.get_recipe(item_id)
        max_mat = stash.max_affordable_batch_materials(recipe.ingredients)
        max_coin = CostCalculator.max_affordable_batch(
          recipe.coin_cost, coins.balance
        )
        feasible = min(batch_size, max_mat, max_coin, 20)

        if feasible <= 0:
          continue

        cost = CostCalculator.total_cost(recipe.coin_cost, feasible)
        cost_int = math.ceil(cost)

        removed_ings = []
        ok = True
        for ing in recipe.ingredients:
          if not stash.remove(ing.item_id, ing.quantity * feasible):
            for prev_ing in removed_ings:
              stash.add(prev_ing.item_id, prev_ing.quantity * feasible)
            ok = False
            break
          removed_ings.append(ing)

        if not ok:
          continue

        if not coins.spend(cost_int):
          for ing in recipe.ingredients:
            stash.add(ing.item_id, ing.quantity * feasible)
          continue

        slots.start(item_id, feasible)
        total_cost += cost_int
        total_items_produced += feasible
        total_batches += 1

      completed = slots.tick()
      for item_id, qty in completed:
        stash.add(item_id, qty)
        if item_id in delivered:
          delivered[item_id] += qty

      all_done = all(delivered.get(k, 0) >= v for k, v in targets.items())
      if all_done and completion_tick == PENALTY_TICK:
        completion_tick = t

    waste = 0.0
    for item_id, target in targets.items():
      excess = max(0, delivered.get(item_id, 0) - target)
      waste += excess

    if total_items_produced > 0:
      cost_per_item = total_cost / total_items_produced
    else:
      cost_per_item = float(PENALTY_TICK)

    return (total_cost, float(completion_tick), waste, cost_per_item)
