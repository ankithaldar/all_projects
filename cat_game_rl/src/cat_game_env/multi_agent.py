#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Multi-agent tier-based crafting system.

Each tier is an independent Gym environment connected via an OrderBoard.
Lower tiers fulfill orders from higher tiers. An orchestrator coordinates
the tick loop across all tier agents.
'''
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.core.items import (
  CraftingTree, ItemId, ITEM_NAME_TO_ID,
  CRAFTABLE_ITEM_IDS, MAX_AFFORDABLE_BATCH,
)
from src.core.inventory import Stash
from src.core.coin_generator import CoinGenerator
from src.core.slot_scheduler import SlotScheduler
from src.core.cost_calculator import CostCalculator
from src.core.target_provider import TargetProvider
from src.cat_game_env.replenish import replenish_base_materials


class OrderBoard:
  '''Board tracking orders posted by higher tiers.

  Orders are posted for items a tier needs and fulfilled as lower
  tiers complete them. Pending quantities drive order propagation
  in the orchestrator.
  '''
  def __init__(self):
    '''Initialize an empty order board.

    Args:
      None.

    Returns:
      None. Both the order and fulfilled maps start empty.
    '''
    self._orders: Dict[ItemId, int] = {}
    self._fulfilled: Dict[ItemId, int] = {}

  def post_order(self, item_id: ItemId, quantity: int) -> None:
    '''Post an order for an item.

    Args:
      item_id: Item being ordered.
      quantity: Quantity ordered; the recorded order only grows.

    Returns:
      None.
    '''
    self._orders[item_id] = max(
      self._orders.get(item_id, 0), quantity
    )

  def get_pending(self, item_id: ItemId) -> int:
    '''Return pending (unfulfilled) quantity for an item.

    Args:
      item_id: Item to look up.

    Returns:
      Ordered quantity minus fulfilled quantity, floored at zero.
    '''
    ordered = self._orders.get(item_id, 0)
    filled = self._fulfilled.get(item_id, 0)
    return max(0, ordered - filled)

  def fulfill(self, item_id: ItemId, quantity: int) -> None:
    '''Mark a quantity of an item as fulfilled.

    Args:
      item_id: Item that was delivered.
      quantity: Quantity delivered.

    Returns:
      None.
    '''
    self._fulfilled[item_id] = (
      self._fulfilled.get(item_id, 0) + quantity
    )

  def get_all_pending_for_tier(
    self, item_ids: List[ItemId]
  ) -> Dict[ItemId, int]:
    '''Return pending orders for a set of items.

    Args:
      item_ids: Items to check.

    Returns:
      Mapping of item to pending quantity, only for items with a
      positive pending quantity.
    '''
    return {
      iid: self.get_pending(iid) for iid in item_ids
      if self.get_pending(iid) > 0
    }

  def reset(self) -> None:
    '''Reset the board, clearing all orders and fulfillments.

    Args:
      None.

    Returns:
      None.
    '''
    self._orders.clear()
    self._fulfilled.clear()


def group_items_by_tier(
  tree: CraftingTree,
) -> Dict[int, List[ItemId]]:
  '''Group craftable items by crafting tier.

  Args:
    tree: Crafting tree providing each item's tier.

  Returns:
    Mapping of tier number to the craftable items in that tier.
  '''
  groups: Dict[int, List[ItemId]] = {}
  for item_id_int in CRAFTABLE_ITEM_IDS:
    item_id = ItemId(item_id_int)
    tier = tree.tier.get(item_id, 0)
    if tier not in groups:
      groups[tier] = []
    groups[tier].append(item_id)
  return groups


def get_ingredient_ids(
  tree: CraftingTree, tier_items: List[ItemId]
) -> List[ItemId]:
  '''Return ingredient ids for tier items.

  Args:
    tree: Crafting tree providing recipes.
    tier_items: Items whose recipe ingredients are collected.

  Returns:
    Sorted ingredient ids used by any of the tier items.
  '''
  ing_set: set = set()
  for item_id in tier_items:
    recipe = tree.get_recipe(item_id)
    for ing in recipe.ingredients:
      ing_set.add(ing.item_id)
  return sorted(ing_set, key=int)


class TierEnv(gym.Env):
  '''Gym environment for a single crafting tier.

  Shares stash, coins, slots, order board and targets with the
  other tiers via the orchestrator. Observations and actions are
  local to this tier's items.
  '''
  metadata = {'render_modes': ['ansi']}

  def __init__(
    self,
    tier: int,
    tier_items: List[ItemId],
    ingredient_ids: List[ItemId],
    crafting_tree: CraftingTree,
    stash: Stash,
    coins: CoinGenerator,
    slots: SlotScheduler,
    order_board: OrderBoard,
    targets: TargetProvider,
    max_batch: int = 20,
    max_ticks: int = 8064,
  ):
    '''Initialize a tier environment.

    Args:
      tier: Tier number this environment controls.
      tier_items: Craftable items owned by this tier.
      ingredient_ids: Items this tier consumes as ingredients.
      crafting_tree: Crafting tree used for recipes and slots.
      stash: Shared stash across tiers.
      coins: Shared coin generator.
      slots: Shared slot scheduler.
      order_board: Shared order board.
      targets: Shared target provider.
      max_batch: Per-item batch cap for actions.
      max_ticks: Episode tick limit used for time fraction.

    Returns:
      None. Builds the local observation and action spaces.
    '''
    super().__init__()
    self.tier = tier
    self.tier_items = tier_items
    self.ingredient_ids = ingredient_ids
    self._tree = crafting_tree
    self._stash = stash
    self._coins = coins
    self._slots = slots
    self._order_board = order_board
    self._targets = targets
    self._max_batch = max_batch
    self._max_ticks = max_ticks

    self._n_items = len(tier_items)
    self._n_ingredients = len(ingredient_ids)

    self._item_to_local = {
      iid: i for i, iid in enumerate(tier_items)
    }

    self.observation_space = self._build_obs_space()
    self.action_space = spaces.MultiDiscrete(
      [max_batch + 1] * self._n_items
    )

    self._prev_fraction = 0.0
    self._tick = 0
    self._orchestrator = None

  def set_orchestrator(self, orch: MultiAgentOrchestrator) -> None:
    '''Set the orchestrator that coordinates this tier.

    Args:
      orch: Orchestrator owning all tier environments.

    Returns:
      None.
    '''
    self._orchestrator = orch

  def reset(
    self, seed: Optional[int] = None, options: Optional[Dict] = None
  ) -> Tuple[Dict[str, np.ndarray], Dict]:
    '''Reset the tier via the orchestrator.

    Args:
      seed: Random seed forwarded to the orchestrator.
      options: Optional reset options (unused).

    Returns:
      Tuple of this tier's observation and an empty info dict.
    '''
    super().reset(seed=seed)
    if self._orchestrator is not None:
      obs_all = self._orchestrator.reset(seed=seed)
      return obs_all[self.tier], {}
    self.reset_state()
    return self.build_obs(0), {}

  def step(
    self, action: np.ndarray
  ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict]:
    '''Advance one step through the orchestrator.

    Args:
      action: Batch sizes for this tier's items.

    Returns:
      Tuple of observation, reward, terminated, truncated and
      info for this tier. Other tiers receive zero actions.

    Raises:
      RuntimeError: If no orchestrator has been set.
    '''
    if self._orchestrator is None:
      raise RuntimeError('TierEnv.step() requires an orchestrator')
    actions: Dict[int, np.ndarray] = {}
    for t_num, t_env in self._orchestrator.tier_envs.items():
      if t_num == self.tier:
        actions[t_num] = action
      else:
        n = len(t_env.tier_items)
        actions[t_num] = np.zeros(n, dtype=np.int64)
    obs_all, rewards, terminated, truncated, info = (
      self._orchestrator.step(actions)
    )
    return obs_all[self.tier], rewards[self.tier], terminated, truncated, info

  def _build_obs_space(self) -> spaces.Dict:
    '''Build the local observation space for this tier.

    Args:
      None.

    Returns:
      Dict space with local/input stash, slots, coins, time
      fraction, pending orders and remaining targets.
    '''
    return spaces.Dict({
      'local_stash': spaces.Box(
        low=0, high=MAX_AFFORDABLE_BATCH,
        shape=(self._n_items,), dtype=np.int32
      ),
      'input_stash': spaces.Box(
        low=0, high=MAX_AFFORDABLE_BATCH,
        shape=(max(self._n_ingredients, 1),), dtype=np.int32
      ),
      'local_slots': spaces.Box(
        low=0, high=1000,
        shape=(self._n_items, 3), dtype=np.int32
      ),
      'coins': spaces.Box(
        low=0, high=99_999_999, shape=(1,), dtype=np.int32
      ),
      'time_fraction': spaces.Box(
        low=0.0, high=1.0, shape=(1,), dtype=np.float32
      ),
      'orders_pending': spaces.Box(
        low=0, high=MAX_AFFORDABLE_BATCH,
        shape=(self._n_items,), dtype=np.int32
      ),
      'targets_remaining': spaces.Box(
        low=-MAX_AFFORDABLE_BATCH, high=MAX_AFFORDABLE_BATCH,
        shape=(self._n_items,), dtype=np.int32
      ),
    })

  def build_obs(self, tick: int) -> Dict[str, np.ndarray]:
    '''Build the observation dict for a tick.

    Args:
      tick: Current tick, used for the time fraction.

    Returns:
      Observation dict clipped to MAX_AFFORDABLE_BATCH bounds.
    '''
    self._tick = tick
    local = np.array(
      [
        min(MAX_AFFORDABLE_BATCH, self._stash.get(iid))
        for iid in self.tier_items
      ],
      dtype=np.int32,
    )
    if self._n_ingredients > 0:
      inputs = np.array(
        [
          min(MAX_AFFORDABLE_BATCH, self._stash.get(iid))
          for iid in self.ingredient_ids
        ],
        dtype=np.int32,
      )
    else:
      inputs = np.zeros(1, dtype=np.int32)

    slot_arr = np.zeros((self._n_items, 3), dtype=np.int32)
    for i, iid in enumerate(self.tier_items):
      slot = self._slots.get_slot(iid)
      slot_arr[i, 0] = 1 if slot.is_active else 0
      slot_arr[i, 1] = slot.remaining_ticks
      slot_arr[i, 2] = slot.batch_size

    orders = np.array(
      [self._order_board.get_pending(iid) for iid in self.tier_items],
      dtype=np.int32,
    )

    remaining = self._targets.targets_remaining_array()
    local_remaining = np.array(
      [remaining[int(iid)] for iid in self.tier_items],
      dtype=np.int32,
    )

    return {
      'local_stash': np.clip(local, 0, MAX_AFFORDABLE_BATCH),
      'input_stash': np.clip(inputs, 0, MAX_AFFORDABLE_BATCH),
      'local_slots': slot_arr,
      'coins': np.array([self._coins.balance], dtype=np.int32),
      'time_fraction': np.array(
        [tick / max(self._max_ticks, 1)], dtype=np.float32
      ),
      'orders_pending': np.clip(orders, 0, MAX_AFFORDABLE_BATCH),
      'targets_remaining': np.clip(
        local_remaining, -MAX_AFFORDABLE_BATCH, MAX_AFFORDABLE_BATCH
      ),
    }

  def action_masks(self) -> np.ndarray:
    '''Return the valid-action mask for this tier.

    Args:
      None.

    Returns:
      Boolean mask with one ``max_batch + 1`` block per tier item;
      batch size zero is always valid.
    '''
    mask_size = self._n_items * (self._max_batch + 1)
    mask = np.zeros(mask_size, dtype=np.bool_)

    for i, iid in enumerate(self.tier_items):
      offset = i * (self._max_batch + 1)
      mask[offset] = True

      if self._slots.is_busy(iid):
        continue

      recipe = self._tree.get_recipe(iid)
      max_mat = self._stash.max_affordable_batch_materials(
        recipe.ingredients
      )
      max_coin = CostCalculator.max_affordable_batch(
        recipe.coin_cost, self._coins.balance
      )
      max_feasible = min(max_mat, max_coin, self._max_batch)

      for b in range(1, max_feasible + 1):
        mask[offset + b] = True

    return mask

  def decode_and_apply(
    self, action: np.ndarray
  ) -> Dict[str, Any]:
    '''Decode and apply a tier action vector.

    Args:
      action: Batch sizes, one per tier item.

    Returns:
      Mapping with applied batches, rejected batches and total
      coins spent. Infeasible batches are clipped or rejected.
    '''
    applied: Dict[ItemId, int] = {}
    rejected: Dict[ItemId, int] = {}
    total_coins_spent = 0

    for i, iid in enumerate(self.tier_items):
      batch_size = int(action[i])
      if batch_size <= 0:
        continue
      if self._slots.is_busy(iid):
        rejected[iid] = batch_size
        continue

      recipe = self._tree.get_recipe(iid)
      max_mat = self._stash.max_affordable_batch_materials(
        recipe.ingredients
      )
      max_coin = CostCalculator.max_affordable_batch(
        recipe.coin_cost, self._coins.balance
      )
      feasible = min(batch_size, max_mat, max_coin, self._max_batch)
      if feasible <= 0:
        rejected[iid] = batch_size
        continue

      cost = CostCalculator.total_cost(recipe.coin_cost, feasible)
      cost_int = math.ceil(cost)

      removed = []
      ok = True
      for ing in recipe.ingredients:
        if not self._stash.remove(ing.item_id, ing.quantity * feasible):
          for prev in removed:
            self._stash.add(prev.item_id, prev.quantity * feasible)
          ok = False
          break
        removed.append(ing)
      if not ok:
        rejected[iid] = batch_size
        continue

      if not self._coins.spend(cost_int):
        for ing in recipe.ingredients:
          self._stash.add(ing.item_id, ing.quantity * feasible)
        rejected[iid] = batch_size
        continue

      self._slots.start(iid, feasible)
      applied[iid] = feasible
      total_coins_spent += cost_int

    return {
      'applied': applied,
      'rejected': rejected,
      'total_coins_spent': total_coins_spent,
    }

  def compute_reward(
    self, action_info: Dict[str, Any]
  ) -> Tuple[float, Dict[str, float]]:
    '''Compute reward and its breakdown for this tier.

    Args:
      action_info: Applied/rejected batches and coins spent
        from decode_and_apply.

    Returns:
      Tuple of total reward and a breakdown dict including slot
      utilization, target progress, coin efficiency and order
      fulfillment.
    '''
    breakdown: Dict[str, float] = {}

    local_active = sum(
      1 for iid in self.tier_items if self._slots.is_busy(iid)
    )
    slot_util = local_active / max(self._n_items, 1)
    breakdown['slot_utilization'] = slot_util

    curr_frac = self._targets.fraction_complete()
    progress = (curr_frac - self._prev_fraction) * 10.0
    done_bonus = 50.0 if self._targets.is_complete() else 0.0
    breakdown['target_progress'] = progress + done_bonus
    self._prev_fraction = curr_frac

    applied = action_info.get('applied', {})
    coins_spent = action_info.get('total_coins_spent', 0)
    n_produced = sum(applied.values()) if applied else 0
    coin_eff = (
      -coins_spent / max(n_produced, 1) / 1000.0
      if coins_spent > 0 else 0.0
    )
    breakdown['coin_efficiency'] = coin_eff

    pending = self._order_board.get_all_pending_for_tier(
      self.tier_items
    )
    total_pending = sum(pending.values()) if pending else 0
    fulfilled_this_step = sum(applied.values()) if applied else 0
    order_rate = (
      fulfilled_this_step / max(total_pending, 1)
      if total_pending > 0 else 1.0
    )
    breakdown['order_fulfillment'] = order_rate

    reward = (
      0.4 * (progress + done_bonus)
      + 0.2 * slot_util
      + 0.2 * order_rate
      + 0.1 * coin_eff
      - 0.001
    )
    breakdown['total'] = reward
    return reward, breakdown

  def reset_state(self) -> None:
    '''Reset tier-local state (previous progress fraction, tick).

    Args:
      None.

    Returns:
      None. Shared stash/coins/slots are reset by the orchestrator.
    '''
    self._prev_fraction = 0.0
    self._tick = 0


class MultiAgentOrchestrator:
  '''Coordinates all tier environments in a shared world.

  Owns the shared stash, coins, slots, targets and order board,
  propagates orders down tiers, and steps every tier per tick.
  '''
  def __init__(self, config: Dict[str, Any]):
    '''Initialize the orchestrator and its tier environments.

    Args:
      config: Mapping with crafting_tree_path, targets_path,
        max_batch_size, max_ticks, initial_coins and
        initial_stash.

    Returns:
      None. Creates one TierEnv per tier found in the tree.
    '''
    tree_path = config.get(
      'crafting_tree_path', 'config/crafting_tree.yaml'
    )
    targets_path = config.get('targets_path', 'config/targets.yaml')
    self._max_batch = config.get('max_batch_size', 20)
    self._max_ticks = config.get('max_ticks', 8064)
    self._initial_coins = config.get('initial_coins', 0)
    self._initial_stash = config.get('initial_stash', {})

    self.tree = CraftingTree.from_yaml(tree_path)
    self.stash = Stash()
    self.coins = CoinGenerator(self._initial_coins)
    self.slots = SlotScheduler(self.tree)
    self.targets = TargetProvider(targets_path)
    self.order_board = OrderBoard()
    self.current_tick = 0

    self.tier_groups = group_items_by_tier(self.tree)
    self.tier_numbers = sorted(self.tier_groups.keys())

    self.tier_envs: Dict[int, TierEnv] = {}
    for tier_num in self.tier_numbers:
      tier_items = self.tier_groups[tier_num]
      ing_ids = get_ingredient_ids(self.tree, tier_items)
      self.tier_envs[tier_num] = TierEnv(
        tier=tier_num,
        tier_items=tier_items,
        ingredient_ids=ing_ids,
        crafting_tree=self.tree,
        stash=self.stash,
        coins=self.coins,
        slots=self.slots,
        order_board=self.order_board,
        targets=self.targets,
        max_batch=self._max_batch,
        max_ticks=self._max_ticks,
      )

    for env in self.tier_envs.values():
      env.set_orchestrator(self)

  def reset(self, seed: int | None = None) -> Dict[int, Dict]:
    '''Reset the shared world and every tier.

    Args:
      seed: Random seed (currently unused beyond signature).

    Returns:
      Mapping of tier number to that tier's initial observation.
    '''
    _ = seed
    self.stash.reset()
    for name, qty in self._initial_stash.items():
      iid = ITEM_NAME_TO_ID.get(name.lower())
      if iid is not None and qty > 0:
        self.stash.add(iid, qty)
    self.coins.reset(self._initial_coins)
    self.slots.reset()
    self.targets.reset()
    self.order_board.reset()
    self.current_tick = 0

    self._propagate_orders()

    for env in self.tier_envs.values():
      env.reset_state()

    obs = {}
    for tier_num, env in self.tier_envs.items():
      obs[tier_num] = env.build_obs(self.current_tick)
    return obs

  def _propagate_orders(self) -> None:
    '''Propagate remaining target orders down the tiers.

    Args:
      None.

    Returns:
      None. Posts orders on the shared order board for items
      still needed by higher tiers, excluding base items.
    '''
    targets_dict = self.targets.targets
    # pylint: disable=protected-access
    for item_id, qty in targets_dict.items():
      remaining = qty - self.targets._delivered.get(item_id, 0)
      if remaining > 0:
        self.order_board.post_order(item_id, remaining)

    for tier_num in reversed(self.tier_numbers):
      for item_id in self.tier_groups[tier_num]:
        pending = self.order_board.get_pending(item_id)
        if pending <= 0 or item_id not in self.tree.recipes:
          continue
        recipe = self.tree.get_recipe(item_id)
        for ing in recipe.ingredients:
          if not self.tree.is_base(ing.item_id):
            needed = ing.quantity * pending
            available = self.stash.get(ing.item_id)
            deficit = max(0, needed - available)
            if deficit > 0:
              self.order_board.post_order(ing.item_id, deficit)

  def step(
    self, actions: Dict[int, np.ndarray]
  ) -> Tuple[Dict[int, Dict], Dict[int, float], bool, bool, Dict]:
    '''Advance the shared world one tick for all tiers.

    Args:
      actions: Mapping of tier number to that tier's action vector.

    Returns:
      Tuple of per-tier observations, per-tier rewards,
      terminated, truncated and an info dict with tick, action
      info and completed items.
    '''
    self.coins.tick()

    replenish_base_materials(self.stash)

    all_action_info: Dict[int, Dict] = {}
    for tier_num in self.tier_numbers:
      if tier_num in actions:
        env = self.tier_envs[tier_num]
        info = env.decode_and_apply(actions[tier_num])
        all_action_info[tier_num] = info

    completed = self.slots.tick()
    for item_id, qty in completed:
      self.stash.add(item_id, qty)
      self.targets.deliver(item_id, qty)
      self.order_board.fulfill(item_id, qty)

    self.current_tick += 1
    self._propagate_orders()

    obs = {}
    rewards = {}
    for tier_num, env in self.tier_envs.items():
      obs[tier_num] = env.build_obs(self.current_tick)
      action_info = all_action_info.get(tier_num, {
        'applied': {}, 'rejected': {}, 'total_coins_spent': 0,
      })
      r, _ = env.compute_reward(action_info)
      rewards[tier_num] = r

    terminated = self.targets.is_complete()
    truncated = self.current_tick >= self._max_ticks

    info = {
      'tick': self.current_tick,
      'action_info': all_action_info,
      'completed': completed,
    }

    return obs, rewards, terminated, truncated, info

  def get_action_masks(self) -> Dict[int, np.ndarray]:
    '''Return valid-action masks for all tiers.

    Args:
      None.

    Returns:
      Mapping of tier number to that tier's action mask.
    '''
    return {
      tier_num: env.action_masks()
      for tier_num, env in self.tier_envs.items()
    }
