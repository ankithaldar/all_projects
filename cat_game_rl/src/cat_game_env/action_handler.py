from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np

from src.core.items import CraftingTree, ItemId, CRAFTABLE_ITEM_IDS, NUM_CRAFTABLE
from src.core.inventory import Stash
from src.core.coin_generator import CoinGenerator
from src.core.slot_scheduler import SlotScheduler
from src.core.cost_calculator import CostCalculator


class ActionHandler:
  def __init__(self, crafting_tree: CraftingTree, max_batch: int = 20):
    self._tree = crafting_tree
    self._max_batch = max_batch

  @property
  def max_batch(self) -> int:
    return self._max_batch

  def decode(self, action: np.ndarray) -> Dict[ItemId, int]:
    result = {}
    for i, item_id_int in enumerate(CRAFTABLE_ITEM_IDS):
      batch_size = int(action[i])
      if batch_size > 0:
        result[ItemId(item_id_int)] = batch_size
    return result

  def validate_and_apply(
    self,
    decoded: Dict[ItemId, int],
    stash: Stash,
    coins: CoinGenerator,
    slots: SlotScheduler,
  ) -> Dict[str, object]:
    applied = {}
    rejected = {}
    total_coins_spent = 0

    for item_id in self._tree.topo_order:
      if item_id not in decoded:
        continue

      batch_size = decoded[item_id]

      if slots.is_busy(item_id):
        rejected[item_id] = "slot_busy"
        continue

      recipe = self._tree.get_recipe(item_id)

      max_mat = stash.max_affordable_batch_materials(recipe.ingredients)
      max_coin = CostCalculator.max_affordable_batch(
        recipe.coin_cost, coins.balance
      )
      feasible_batch = min(batch_size, max_mat, max_coin, self._max_batch)

      if feasible_batch <= 0:
        rejected[item_id] = "insufficient_resources"
        continue

      cost = CostCalculator.total_cost(recipe.coin_cost, feasible_batch)
      cost_int = math.ceil(cost)

      removed_ings = []
      mat_ok = True
      for ing in recipe.ingredients:
        success = stash.remove(ing.item_id, ing.quantity * feasible_batch)
        if not success:
          for prev_ing in removed_ings:
            stash.add(prev_ing.item_id, prev_ing.quantity * feasible_batch)
          rejected[item_id] = "material_deduction_failed"
          mat_ok = False
          break
        removed_ings.append(ing)

      if not mat_ok:
        continue

      if not coins.spend(cost_int):
        for ing in recipe.ingredients:
          stash.add(ing.item_id, ing.quantity * feasible_batch)
        rejected[item_id] = "coin_deduction_failed"
        continue

      slots.start(item_id, feasible_batch)
      applied[item_id] = feasible_batch
      total_coins_spent += cost_int

    return {
      "applied": applied,
      "rejected": rejected,
      "total_coins_spent": total_coins_spent,
    }

  def compute_action_mask(
    self,
    stash: Stash,
    coins: CoinGenerator,
    slots: SlotScheduler,
  ) -> np.ndarray:
    mask_size = NUM_CRAFTABLE * (self._max_batch + 1)
    mask = np.zeros(mask_size, dtype=np.bool_)

    for i, item_id_int in enumerate(CRAFTABLE_ITEM_IDS):
      item_id = ItemId(item_id_int)
      offset = i * (self._max_batch + 1)
      mask[offset] = True  # batch_size=0 always valid

      if slots.is_busy(item_id):
        continue

      recipe = self._tree.get_recipe(item_id)
      max_mat = stash.max_affordable_batch_materials(recipe.ingredients)
      max_coin = CostCalculator.max_affordable_batch(
        recipe.coin_cost, coins.balance
      )
      max_feasible = min(max_mat, max_coin, self._max_batch)

      for b in range(1, max_feasible + 1):
        mask[offset + b] = True

    return mask
