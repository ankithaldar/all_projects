#!/usr/bin/env python3
from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.core.items import (
  CraftingTree, ItemId, ITEM_NAME_TO_ID,
  CRAFTABLE_ITEM_IDS, BASE_ITEM_IDS, NUM_CRAFTABLE,
)
from src.core.inventory import Stash
from src.core.coin_generator import CoinGenerator
from src.core.slot_scheduler import SlotScheduler
from src.core.cost_calculator import CostCalculator

MAX_TICKS = 8064
PENALTY_TICK = MAX_TICKS + 1


def critical_path_ticks(
  tree: CraftingTree, item_id: ItemId, memo: Dict[ItemId, int] | None = None
) -> int:
  if memo is None:
    memo = {}
  if item_id in memo:
    return memo[item_id]
  if tree.is_base(item_id) or item_id not in tree.recipes:
    memo[item_id] = 0
    return 0
  recipe = tree.get_recipe(item_id)
  max_dep = 0
  for ing in recipe.ingredients:
    max_dep = max(max_dep, critical_path_ticks(tree, ing.item_id, memo))
  result = max_dep + tree.craft_time_ticks(item_id)
  memo[item_id] = result
  return result


def compute_total_requirements(
  tree: CraftingTree,
  targets: Dict[ItemId, int],
  initial_stash: Dict[str, int],
) -> Dict[ItemId, int]:
  reqs: Dict[ItemId, int] = {}
  for item_id, qty in targets.items():
    reqs[item_id] = reqs.get(item_id, 0) + qty

  for item_id in reversed(tree.topo_order):
    needed = reqs.get(item_id, 0)
    if needed <= 0:
      continue
    if item_id not in tree.recipes:
      continue
    recipe = tree.get_recipe(item_id)
    for ing in recipe.ingredients:
      reqs[ing.item_id] = reqs.get(ing.item_id, 0) + ing.quantity * needed

  for name, qty in initial_stash.items():
    iid = ITEM_NAME_TO_ID.get(name.lower())
    if iid is not None and iid in reqs:
      reqs[iid] = max(0, reqs[iid] - qty)

  return reqs


@dataclass
class SimResult:
  name: str
  total_cost: float
  completion_tick: int
  waste: float
  cost_per_item: float
  slot_utilization: float
  coins_per_target: float
  total_items_produced: int


StrategyFn = Callable[
  [int, CraftingTree, Stash, SlotScheduler, CoinGenerator,
   Dict[ItemId, int], Dict[ItemId, int]],
  Dict[ItemId, int],
]


def run_simulation(
  name: str,
  strategy_fn: StrategyFn,
  crafting_tree: CraftingTree,
  targets: Dict[ItemId, int],
  initial_coins: int,
  initial_stash: Dict[str, int],
) -> SimResult:
  stash = Stash()
  for sname, qty in initial_stash.items():
    iid = ITEM_NAME_TO_ID.get(sname.lower())
    if iid is not None and qty > 0:
      stash.add(iid, qty)
  coins = CoinGenerator(initial_coins)
  slots = SlotScheduler(crafting_tree)
  delivered: Dict[ItemId, int] = {k: 0 for k in targets}
  total_cost = 0.0
  completion_tick = PENALTY_TICK
  total_items_produced = 0
  total_active_ticks = 0

  for t in range(MAX_TICKS):
    coins.tick()

    for base_id in BASE_ITEM_IDS:
      deficit = max(0, 9999 - stash.get(base_id))
      if deficit > 0:
        stash.add(base_id, deficit)

    actions = strategy_fn(
      t, crafting_tree, stash, slots, coins, targets, delivered
    )

    for item_id in crafting_tree.topo_order:
      if item_id not in actions:
        continue
      batch_size = actions[item_id]
      if batch_size <= 0 or slots.is_busy(item_id):
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

      removed = []
      ok = True
      for ing in recipe.ingredients:
        if not stash.remove(ing.item_id, ing.quantity * feasible):
          for prev in removed:
            stash.add(prev.item_id, prev.quantity * feasible)
          ok = False
          break
        removed.append(ing)
      if not ok:
        continue

      if not coins.spend(cost_int):
        for ing in recipe.ingredients:
          stash.add(ing.item_id, ing.quantity * feasible)
        continue

      slots.start(item_id, feasible)
      total_cost += cost_int
      total_items_produced += feasible

    completed = slots.tick()
    for item_id, qty in completed:
      stash.add(item_id, qty)
      if item_id in delivered:
        delivered[item_id] += qty

    total_active_ticks += slots.active_count()

    all_done = all(delivered.get(k, 0) >= v for k, v in targets.items())
    if all_done and completion_tick == PENALTY_TICK:
      completion_tick = t

  waste = 0.0
  for item_id, target in targets.items():
    excess = max(0, delivered.get(item_id, 0) - target)
    waste += excess

  target_sum = sum(targets.values())
  return SimResult(
    name=name,
    total_cost=total_cost,
    completion_tick=completion_tick,
    waste=waste,
    cost_per_item=(
      total_cost / total_items_produced
      if total_items_produced > 0 else float(PENALTY_TICK)
    ),
    slot_utilization=total_active_ticks / (MAX_TICKS * NUM_CRAFTABLE),
    coins_per_target=total_cost / max(target_sum, 1),
    total_items_produced=total_items_produced,
  )


def make_greedy_strategy(
  tree: CraftingTree, reqs: Dict[ItemId, int]
) -> StrategyFn:
  tier_sorted = sorted(
    tree.topo_order, key=lambda x: (tree.tier.get(x, 0), int(x))
  )

  def strategy(tick, tree, stash, slots, coins, targets, delivered):
    actions = {}
    for item_id in tier_sorted:
      needed = reqs.get(item_id, 0) + targets.get(item_id, 0)
      produced = delivered.get(item_id, 0) + stash.get(item_id)
      if produced >= needed and delivered.get(item_id, 0) >= targets.get(item_id, 0):
        continue
      if not slots.is_busy(item_id):
        actions[item_id] = 1
    return actions
  return strategy


def make_critical_path_strategy(
  tree: CraftingTree, reqs: Dict[ItemId, int], crit: Dict[ItemId, int]
) -> StrategyFn:
  crit_sorted = sorted(
    tree.topo_order, key=lambda x: (-crit.get(x, 0), -tree.tier.get(x, 0))
  )

  def strategy(tick, tree, stash, slots, coins, targets, delivered):
    actions = {}
    for item_id in crit_sorted:
      needed = reqs.get(item_id, 0) + targets.get(item_id, 0)
      produced = delivered.get(item_id, 0) + stash.get(item_id)
      if produced >= needed and delivered.get(item_id, 0) >= targets.get(item_id, 0):
        continue
      if not slots.is_busy(item_id):
        actions[item_id] = 1
    return actions
  return strategy


def make_coin_min_strategy(
  tree: CraftingTree, reqs: Dict[ItemId, int]
) -> StrategyFn:
  def strategy(tick, tree, stash, slots, coins, targets, delivered):
    actions = {}
    for item_id in tree.topo_order:
      needed = reqs.get(item_id, 0) + targets.get(item_id, 0)
      produced = delivered.get(item_id, 0) + stash.get(item_id)
      if produced >= needed and delivered.get(item_id, 0) >= targets.get(item_id, 0):
        continue
      if not slots.is_busy(item_id):
        actions[item_id] = 1
    return actions
  return strategy


def make_time_min_strategy(
  tree: CraftingTree, reqs: Dict[ItemId, int]
) -> StrategyFn:
  def strategy(tick, tree, stash, slots, coins, targets, delivered):
    actions = {}
    for item_id in tree.topo_order:
      needed = reqs.get(item_id, 0) + targets.get(item_id, 0)
      produced = delivered.get(item_id, 0) + stash.get(item_id)
      if produced >= needed and delivered.get(item_id, 0) >= targets.get(item_id, 0):
        continue
      if not slots.is_busy(item_id):
        actions[item_id] = 20
    return actions
  return strategy


def print_dag_analysis(
  tree: CraftingTree, crit: Dict[ItemId, int], targets: Dict[ItemId, int]
) -> None:
  print("=" * 65)
  print("STATIC DAG ANALYSIS")
  print("=" * 65)
  print(
    f"{'Item':<16}{'Tier':<6}{'Craft(ticks)':<14}"
    f"{'CritPath(ticks)':<17}{'CoinCost':<10}"
  )
  print("-" * 65)
  for item_id in tree.topo_order:
    recipe = tree.get_recipe(item_id)
    marker = " *" if item_id in targets else ""
    print(
      f"{item_id.name.lower():<16}{tree.tier[item_id]:<6}"
      f"{tree.craft_time_ticks(item_id):<14}{crit[item_id]:<17}"
      f"{recipe.coin_cost:<10}{marker}"
    )
  print("\n* = target item\n")

  print("TARGET ITEMS:")
  for item_id, qty in sorted(targets.items(), key=lambda x: -crit[x[0]]):
    print(
      f"  {item_id.name.lower():<16} qty={qty:<4} "
      f"crit_path={crit[item_id]} ticks "
      f"({crit[item_id] * 5 / 60:.1f} hours)"
    )
  print()


def print_results(results: List[SimResult]) -> None:
  print("=" * 95)
  print("BASELINE COMPARISON")
  print("=" * 95)
  print(
    f"{'Strategy':<16}{'TotalCost':>12}{'Completion':>12}"
    f"{'Waste':>8}{'Cost/Item':>11}{'SlotUtil':>10}"
    f"{'Coin/Tgt':>10}{'Items':>8}"
  )
  print("-" * 95)
  for r in results:
    tick_str = (
      str(r.completion_tick) if r.completion_tick < PENALTY_TICK
      else "FAILED"
    )
    days = (
      f"({r.completion_tick * 5 / 1440:.1f}d)"
      if r.completion_tick < PENALTY_TICK else ""
    )
    print(
      f"{r.name:<16}{r.total_cost:>12,.0f}{tick_str:>8} {days:<5}"
      f"{r.waste:>7.0f}{r.cost_per_item:>11.1f}"
      f"{r.slot_utilization:>9.1%}{r.coins_per_target:>10.1f}"
      f"{r.total_items_produced:>8}"
    )
  print()


def main() -> None:
  tree = CraftingTree.from_yaml("config/crafting_tree.yaml")

  with open("config/training.yaml", "r") as f:
    train_cfg = yaml.safe_load(f)
  env_cfg = train_cfg.get("environment", {})
  initial_coins = env_cfg.get("initial_coins", 0)
  initial_stash = env_cfg.get("initial_stash", {})

  with open("config/targets.yaml", "r") as f:
    targets_data = yaml.safe_load(f)["targets"]
  targets = {ITEM_NAME_TO_ID[k.lower()]: v for k, v in targets_data.items()}

  memo: Dict[ItemId, int] = {}
  crit = {
    item_id: critical_path_ticks(tree, item_id, memo)
    for item_id in tree.topo_order
  }
  reqs = compute_total_requirements(tree, targets, initial_stash)

  print_dag_analysis(tree, crit, targets)

  print("REQUIREMENTS (after subtracting initial stash):")
  for item_id in tree.topo_order:
    if reqs.get(item_id, 0) > 0:
      print(f"  {item_id.name.lower():<16} {reqs[item_id]:>6}")
  print()

  strategies = [
    ("greedy", make_greedy_strategy(tree, reqs)),
    ("critical_path", make_critical_path_strategy(tree, reqs, crit)),
    ("coin_min", make_coin_min_strategy(tree, reqs)),
    ("time_min", make_time_min_strategy(tree, reqs)),
  ]

  results = []
  for sname, fn in strategies:
    print(f"Running {sname}...", end=" ", flush=True)
    r = run_simulation(
      sname, fn, tree, targets, initial_coins, initial_stash
    )
    status = (
      f"done (tick {r.completion_tick})"
      if r.completion_tick < PENALTY_TICK else "FAILED"
    )
    print(status)
    results.append(r)

  print()
  print_results(results)


if __name__ == "__main__":
  main()
