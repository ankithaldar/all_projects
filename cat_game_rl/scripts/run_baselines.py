#!/usr/bin/env python3
"""Order-based manufacturing baseline for the crafting RL system.

Top tiers place purchase orders on lower tiers. Lower tiers fulfill
orders and push finished goods up. Two baselines:
  - Baseline Time (batch=1): minimum coins per unit, maximum time
  - Baseline Coins (batch=all): minimum production runs, shows true cost
"""
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


# ── Static DAG Analysis ──────────────────────────────────────────

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


# ── Order Book: top-down demand explosion ─────────────────────────

def build_order_book(
  tree: CraftingTree,
  targets: Dict[ItemId, int],
  initial_stash: Dict[str, int],
) -> Dict[ItemId, int]:
  gross: Dict[ItemId, int] = {}
  for item_id, qty in targets.items():
    gross[item_id] = gross.get(item_id, 0) + qty

  for item_id in reversed(tree.topo_order):
    needed = gross.get(item_id, 0)
    if needed <= 0 or item_id not in tree.recipes:
      continue
    recipe = tree.get_recipe(item_id)
    for ing in recipe.ingredients:
      gross[ing.item_id] = gross.get(ing.item_id, 0) + ing.quantity * needed

  net: Dict[ItemId, int] = {}
  for item_id, qty in gross.items():
    if tree.is_base(item_id):
      continue
    stash_name = item_id.name.lower()
    on_hand = 0
    for name, sq in initial_stash.items():
      if name.lower() == stash_name:
        on_hand = sq
        break
    net[item_id] = max(0, qty - on_hand)

  for item_id, qty in targets.items():
    net[item_id] = max(net.get(item_id, 0), qty)

  return net


def print_order_cascade(
  tree: CraftingTree,
  order_book: Dict[ItemId, int],
  targets: Dict[ItemId, int],
) -> None:
  print("=" * 70)
  print("ORDER CASCADE (top-down demand → bottom-up fulfillment)")
  print("=" * 70)

  max_tier = max(tree.tier[i] for i in tree.topo_order)
  for tier in range(max_tier, 0, -1):
    tier_items = [
      i for i in tree.topo_order
      if tree.tier.get(i, 0) == tier and order_book.get(i, 0) > 0
    ]
    if not tier_items:
      continue

    print(f"\n  Tier {tier}:")
    for item_id in tier_items:
      qty = order_book[item_id]
      tgt = targets.get(item_id, 0)
      tgt_str = f" (target: {tgt})" if tgt > 0 else ""
      recipe = tree.get_recipe(item_id)
      ing_str = " + ".join(
        f"{ing.quantity * qty} {ing.item_id.name.lower()}"
        for ing in recipe.ingredients
      )
      print(f"    {item_id.name.lower():<16} order={qty:<5}{tgt_str}")
      print(f"      → places order: {ing_str}")

  print()


# ── Simulation runner ─────────────────────────────────────────────

@dataclass
class ScheduleRow:
  tick_index: int
  elapsed_minutes: int
  item_name: str
  batch_size_decided: int


@dataclass
class SimResult:
  name: str
  total_cost: float
  completion_tick: int
  waste: float
  cost_per_item: float
  slot_utilization: float
  slot_idle_ratio: float
  coins_per_target: float
  total_items_produced: int
  batches_run: int
  queue_blocks: int
  coin_starvation_ticks: int
  inventory_imbalance: float
  schedule: List[ScheduleRow]


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
  batches_run = 0
  queue_blocks = 0
  coin_starvation_ticks = 0
  schedule: List[ScheduleRow] = []

  for t in range(MAX_TICKS):
    coins.tick()

    for base_id in BASE_ITEM_IDS:
      deficit = max(0, 9999 - stash.get(base_id))
      if deficit > 0:
        stash.add(base_id, deficit)

    actions = strategy_fn(
      t, crafting_tree, stash, slots, coins, targets, delivered
    )

    tick_blocked = False
    for item_id in crafting_tree.topo_order:
      if item_id not in actions:
        continue
      batch_size = actions[item_id]
      if batch_size <= 0:
        continue
      if slots.is_busy(item_id):
        queue_blocks += 1
        continue

      recipe = crafting_tree.get_recipe(item_id)
      max_mat = stash.max_affordable_batch_materials(recipe.ingredients)
      max_coin = CostCalculator.max_affordable_batch(
        recipe.coin_cost, coins.balance
      )
      feasible = min(batch_size, max_mat, max_coin, 20)
      if feasible <= 0:
        if max_coin == 0 and max_mat > 0:
          tick_blocked = True
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
      batches_run += 1
      schedule.append(ScheduleRow(
        tick_index=t,
        elapsed_minutes=t * 5,
        item_name=item_id.name.lower(),
        batch_size_decided=feasible,
      ))

    if tick_blocked:
      coin_starvation_ticks += 1

    completed = slots.tick()
    for item_id, qty in completed:
      stash.add(item_id, qty)
      if item_id in delivered:
        delivered[item_id] += qty

    active = slots.active_count()
    total_active_ticks += active

    all_done = all(delivered.get(k, 0) >= v for k, v in targets.items())
    if all_done and completion_tick == PENALTY_TICK:
      completion_tick = t

  waste = 0.0
  for item_id, target in targets.items():
    excess = max(0, delivered.get(item_id, 0) - target)
    waste += excess

  final_stash = stash.as_array()
  craftable_stash = [int(final_stash[i]) for i in CRAFTABLE_ITEM_IDS]
  stash_mean = np.mean(craftable_stash) if craftable_stash else 0
  stash_std = np.std(craftable_stash) if craftable_stash else 0
  imbalance = stash_std / max(stash_mean, 1)

  target_sum = sum(targets.values())
  util = total_active_ticks / (MAX_TICKS * NUM_CRAFTABLE)
  return SimResult(
    name=name,
    total_cost=total_cost,
    completion_tick=completion_tick,
    waste=waste,
    cost_per_item=(
      total_cost / total_items_produced
      if total_items_produced > 0 else float(PENALTY_TICK)
    ),
    slot_utilization=util,
    slot_idle_ratio=1.0 - util,
    coins_per_target=total_cost / max(target_sum, 1),
    total_items_produced=total_items_produced,
    batches_run=batches_run,
    queue_blocks=queue_blocks,
    coin_starvation_ticks=coin_starvation_ticks,
    inventory_imbalance=imbalance,
    schedule=schedule,
  )


def export_schedule(result: SimResult, path: str) -> None:
  os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
  with open(path, "w") as f:
    f.write(
      f"{'tick_index':<12}{'elapsed_minutes':<18}"
      f"{'item_name':<20}{'batch_size_decided'}\n"
    )
    f.write("-" * 68 + "\n")
    for row in result.schedule:
      f.write(
        f"{row.tick_index:<12}{row.elapsed_minutes:<18}"
        f"{row.item_name:<20}{row.batch_size_decided}\n"
      )


# ── Order-based strategies ────────────────────────────────────────

def make_order_strategy(
  tree: CraftingTree,
  order_book: Dict[ItemId, int],
  batch_mode: str,
) -> StrategyFn:
  def strategy(tick, tree, stash, slots, coins, targets, delivered):
    actions: Dict[ItemId, int] = {}

    for item_id in tree.topo_order:
      order_qty = order_book.get(item_id, 0)
      if order_qty <= 0:
        continue
      if slots.is_busy(item_id):
        continue

      if item_id in targets:
        remaining = targets[item_id] - delivered.get(item_id, 0)
      else:
        produced = delivered.get(item_id, 0) + stash.get(item_id)
        remaining = order_qty - produced

      if remaining <= 0:
        continue

      if batch_mode == "one":
        actions[item_id] = 1
      else:
        actions[item_id] = min(remaining, 20)

    return actions
  return strategy


# ── Output formatting ─────────────────────────────────────────────

def print_dag_analysis(
  tree: CraftingTree, crit: Dict[ItemId, int], targets: Dict[ItemId, int]
) -> None:
  print("=" * 70)
  print("STATIC DAG ANALYSIS")
  print("=" * 70)
  print(
    f"{'Item':<16}{'Tier':<6}{'Craft(ticks)':<14}"
    f"{'CritPath(ticks)':<17}{'CoinCost':<10}"
  )
  print("-" * 70)
  for item_id in tree.topo_order:
    recipe = tree.get_recipe(item_id)
    marker = " *" if item_id in targets else ""
    print(
      f"{item_id.name.lower():<16}{tree.tier[item_id]:<6}"
      f"{tree.craft_time_ticks(item_id):<14}{crit[item_id]:<17}"
      f"{recipe.coin_cost:<10}{marker}"
    )
  print("\n* = target item\n")


def print_order_book(order_book: Dict[ItemId, int], tree: CraftingTree) -> None:
  print("=" * 50)
  print("ORDER BOOK (net quantities to produce)")
  print("=" * 50)
  total_orders = 0
  for item_id in tree.topo_order:
    qty = order_book.get(item_id, 0)
    if qty > 0:
      batches = math.ceil(qty / 20)
      cost_b1 = math.ceil(CostCalculator.total_cost(
        tree.get_recipe(item_id).coin_cost, 1
      )) * qty
      cost_bmax = sum(
        math.ceil(CostCalculator.total_cost(
          tree.get_recipe(item_id).coin_cost, min(qty - i * 20, 20)
        ))
        for i in range(batches)
      )
      print(
        f"  {item_id.name.lower():<16} qty={qty:<5} "
        f"batches(b=1)={qty:<4} batches(b=max)={batches:<3} "
        f"coins(b=1)={cost_b1:>8,}  coins(b=max)={cost_bmax:>8,}"
      )
      total_orders += qty
  print(f"\n  Total items to produce: {total_orders}\n")


def print_results(results: List[SimResult]) -> None:
  print("=" * 105)
  print("BASELINE COMPARISON — Production Metrics")
  print("=" * 105)
  print(
    f"{'Strategy':<20}{'TotalCost':>12}{'Tick':>8}{'Days':>7}"
    f"{'Waste':>7}{'Cost/Item':>11}{'SlotUtil':>10}"
    f"{'Coin/Tgt':>10}{'Batches':>9}{'Items':>7}"
  )
  print("-" * 105)
  for r in results:
    if r.completion_tick < PENALTY_TICK:
      tick_str = str(r.completion_tick)
      days_str = f"{r.completion_tick * 5 / 1440:.1f}"
    else:
      tick_str = "FAIL"
      days_str = "-"
    print(
      f"{r.name:<20}{r.total_cost:>12,.0f}{tick_str:>8}{days_str:>7}"
      f"{r.waste:>7.0f}{r.cost_per_item:>11.1f}"
      f"{r.slot_utilization:>9.1%}{r.coins_per_target:>10.1f}"
      f"{r.batches_run:>9}{r.total_items_produced:>7}"
    )
  print()

  print("=" * 75)
  print("INEFFICIENCY METRICS (Section 3.2)")
  print("=" * 75)
  print(
    f"{'Strategy':<20}{'SlotIdle':>10}{'QueueBlocks':>13}"
    f"{'CoinStarve':>12}{'InvImbalance':>14}"
  )
  print("-" * 75)
  for r in results:
    print(
      f"{r.name:<20}{r.slot_idle_ratio:>9.1%}"
      f"{r.queue_blocks:>13}{r.coin_starvation_ticks:>12}"
      f"{r.inventory_imbalance:>14.2f}"
    )
  print()

  if len(results) == 2:
    t, c = results[0], results[1]
    print("INTERPRETATION:")
    print(
      f"  Time baseline (batch=1):   {t.name} — "
      f"cheapest per unit ({t.cost_per_item:.1f} coins/item), "
      f"slowest ({t.completion_tick} ticks)"
    )
    print(
      f"  Coins baseline (batch=max): {c.name} — "
      f"fewest batches ({c.batches_run}), "
      f"fastest ({c.completion_tick} ticks), "
      f"total cost {c.total_cost:,.0f}"
    )
    if t.completion_tick < PENALTY_TICK and c.completion_tick < PENALTY_TICK:
      time_saved = t.completion_tick - c.completion_tick
      cost_diff = c.total_cost - t.total_cost
      print(
        f"  Trade-off: batch=max saves {time_saved} ticks "
        f"({time_saved * 5 / 1440:.1f} days) "
        f"but costs {cost_diff:,.0f} more coins"
      )
    print()
    print("INEFFICIENCY ANALYSIS:")
    for r in results:
      print(f"  {r.name}:")
      print(f"    Slot idle ratio:     {r.slot_idle_ratio:.1%} of capacity unused")
      print(f"    Queue blocks:        {r.queue_blocks} (wanted to craft but slot busy)")
      print(f"    Coin starvation:     {r.coin_starvation_ticks} ticks with materials but no coins")
      print(f"    Inventory imbalance: {r.inventory_imbalance:.2f} (std/mean of final craftable stash)")
    print()


# ── Main ──────────────────────────────────────────────────────────

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

  order_book = build_order_book(tree, targets, initial_stash)

  print()
  print_dag_analysis(tree, crit, targets)
  print_order_cascade(tree, order_book, targets)
  print_order_book(order_book, tree)

  print(
    f"Initial state: {initial_coins:,} coins, "
    f"{sum(initial_stash.values())} items in stash"
  )
  print(f"Horizon: {MAX_TICKS} ticks ({MAX_TICKS * 5 / 1440:.0f} days)")
  print(
    f"Coin income: 210/tick = {MAX_TICKS * 210:,} over horizon"
  )
  print(
    f"Total coin budget: {initial_coins + MAX_TICKS * 210:,}\n"
  )

  strat_time = make_order_strategy(tree, order_book, "one")
  strat_coins = make_order_strategy(tree, order_book, "all")

  strategies = [
    ("baseline_time_b1", strat_time),
    ("baseline_coins_bmax", strat_coins),
  ]

  results = []
  for sname, fn in strategies:
    print(f"Running {sname}...", end=" ", flush=True)
    r = run_simulation(
      sname, fn, tree, targets, initial_coins, initial_stash
    )
    status = (
      f"done (tick {r.completion_tick}, "
      f"{r.completion_tick * 5 / 1440:.1f} days)"
      if r.completion_tick < PENALTY_TICK else "FAILED"
    )
    print(status)
    results.append(r)

  print()
  print_results(results)

  schedule_paths = {
    "baseline_time_b1": [
      "output/rl_batch_schedule_train.txt",
      "output/rl_batch_schedule_eval.txt",
    ],
    "baseline_coins_bmax": [
      "output/ga_results/ga_batch_schedule_train.txt",
      "output/ga_results/ga_batch_schedule_eval.txt",
    ],
  }

  print("EXPORTING SCHEDULES:")
  for r in results:
    paths = schedule_paths.get(r.name, [])
    for p in paths:
      export_schedule(r, p)
      print(f"  {p} ({len(r.schedule)} rows)")

  export_schedule(results[0], "output/batch_schedule.txt")
  print(f"  output/batch_schedule.txt (RL default)")
  print()


if __name__ == "__main__":
  main()
