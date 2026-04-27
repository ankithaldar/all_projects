from __future__ import annotations

import csv
import io
import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.core.items import (
  CraftingTree, ItemId, ITEM_NAME_TO_ID, CRAFTABLE_ITEM_IDS,
  BASE_ITEM_IDS, NUM_ITEMS,
)
from src.core.inventory import Stash
from src.core.coin_generator import CoinGenerator
from src.core.slot_scheduler import SlotScheduler
from src.core.cost_calculator import CostCalculator


def parse_batch_schedule(path: str) -> pd.DataFrame:
  rows = []
  with open(path, "r") as f:
    lines = f.readlines()
  for line in lines[2:]:
    parts = line.split()
    if len(parts) >= 4:
      rows.append({
        "tick_index": int(parts[0]),
        "elapsed_minutes": int(parts[1]),
        "item_name": parts[2],
        "batch_size_decided": int(parts[3]),
      })
  return pd.DataFrame(rows)


def parse_ga_log(path: str) -> List[Dict[str, Any]]:
  entries = []
  with open(path, "r") as f:
    for line in f:
      line = line.strip()
      if line:
        entries.append(json.loads(line))
  return entries


def simulate_schedule(
  schedule_df: pd.DataFrame,
  crafting_tree: CraftingTree,
  max_ticks: int = 2016,
  initial_coins: int = 0,
) -> Dict[str, Any]:
  stash = Stash()
  coins = CoinGenerator(initial_coins)
  slots = SlotScheduler(crafting_tree)

  stash_history = []
  coin_history = []
  slot_history = []
  active_slot_history = []

  schedule_by_tick = {}
  for _, row in schedule_df.iterrows():
    tick = int(row["tick_index"])
    if tick not in schedule_by_tick:
      schedule_by_tick[tick] = []
    schedule_by_tick[tick].append({
      "item_name": row["item_name"],
      "batch_size": int(row["batch_size_decided"]),
    })

  for t in range(max_ticks):
    coins.tick()

    for base_id in BASE_ITEM_IDS:
      deficit = max(0, 9999 - stash.get(base_id))
      if deficit > 0:
        stash.add(base_id, deficit)

    if t in schedule_by_tick:
      for action in schedule_by_tick[t]:
        item_name = action["item_name"]
        batch_size = action["batch_size"]
        item_id = ITEM_NAME_TO_ID.get(item_name)
        if item_id is None or slots.is_busy(item_id):
          continue

        recipe = crafting_tree.get_recipe(item_id)
        max_mat = stash.max_affordable_batch_materials(recipe.ingredients)
        max_coin = CostCalculator.max_affordable_batch(
          recipe.coin_cost, coins.balance
        )
        feasible = min(batch_size, max_mat, max_coin)
        if feasible <= 0:
          continue

        cost = math.ceil(
          CostCalculator.total_cost(recipe.coin_cost, feasible)
        )
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
        if not coins.spend(cost):
          for ing in recipe.ingredients:
            stash.add(ing.item_id, ing.quantity * feasible)
          continue
        slots.start(item_id, feasible)

    completed = slots.tick()
    for item_id, qty in completed:
      stash.add(item_id, qty)

    stash_history.append(stash.as_array().copy())
    coin_history.append(coins.balance)
    active_slot_history.append(slots.active_count())
    slot_history.append(slots.get_slot_array().copy())

  return {
    "stash_history": np.array(stash_history),
    "coin_history": np.array(coin_history),
    "slot_history": np.array(slot_history),
    "active_slot_history": np.array(active_slot_history),
  }


def export_replay_csv(sim_data: Dict[str, Any], max_ticks: int = 2016) -> str:
  output = io.StringIO()
  writer = csv.writer(output)

  header = ["tick", "elapsed_minutes", "coin_balance", "active_slots"]
  for item_id in ItemId:
    header.append(f"stash_{item_id.name.lower()}")
  writer.writerow(header)

  stash_hist = sim_data["stash_history"]
  coin_hist = sim_data["coin_history"]
  active_hist = sim_data["active_slot_history"]

  for t in range(min(max_ticks, len(stash_hist))):
    row = [t, t * 5, coin_hist[t], active_hist[t]]
    for i in range(NUM_ITEMS):
      row.append(int(stash_hist[t, i]))
    writer.writerow(row)

  return output.getvalue()
