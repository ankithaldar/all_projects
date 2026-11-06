#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Single-agent crafting Gym environment.'''

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.core.items import (
  CraftingTree, ItemId, NUM_CRAFTABLE,
  ITEM_NAME_TO_ID,
)
from src.core.inventory import Stash
from src.core.coin_generator import CoinGenerator
from src.core.slot_scheduler import SlotScheduler
from src.core.target_provider import TargetProvider
from src.cat_game_env.replenish import replenish_base_materials
from src.cat_game_env.observation_builder import ObservationBuilder
from src.cat_game_env.action_handler import ActionHandler
from src.cat_game_env.reward_shaper import RewardShaper


class CraftingEnv(gym.Env):
  '''Single-agent crafting environment.'''
  metadata = {'render_modes': ['human', 'ansi']}

  def __init__(self, config: Dict[str, Any]):
    '''Initialize env from config dict.'''
    super().__init__()

    tree_path = config.get('crafting_tree_path', 'config/crafting_tree.yaml')
    targets_path = config.get('targets_path', 'config/targets.yaml')
    self._max_batch = config.get('max_batch_size', 20)
    self._max_ticks = config.get('max_ticks', 8064)
    self._initial_coins = config.get('initial_coins', 0)
    self._initial_stash = config.get('initial_stash', {})
    reward_weights = config.get('reward_weights', None)

    self.crafting_tree = CraftingTree.from_yaml(tree_path)
    self.stash = Stash()
    self.coins = CoinGenerator(self._initial_coins)
    self.slots = SlotScheduler(self.crafting_tree)
    self.targets = TargetProvider(targets_path)
    self.obs_builder = ObservationBuilder(self._max_ticks)
    self.action_handler = ActionHandler(self.crafting_tree, self._max_batch)
    self.reward_shaper = RewardShaper(reward_weights)

    self.current_tick = 0

    self.observation_space = self.obs_builder.build_space()
    self.action_space = spaces.MultiDiscrete(
      [self._max_batch + 1] * NUM_CRAFTABLE
    )

  def reset(
    self,
    seed: Optional[int] = None,
    options: Optional[Dict] = None,
  ) -> Tuple[Dict[str, np.ndarray], Dict]:
    '''Reset env state and return initial obs.'''
    super().reset(seed=seed)
    self.stash.reset()
    for name, qty in self._initial_stash.items():
      item_id = ITEM_NAME_TO_ID.get(name.lower())
      if item_id is not None and qty > 0:
        self.stash.add(item_id, qty)
    self.coins.reset(self._initial_coins)
    self.slots.reset()
    self.targets.reset()
    self.current_tick = 0

    obs = self.obs_builder.build_obs(
      self.stash, self.coins, self.slots, self.targets, self.current_tick
    )
    return obs, {}

  def step(
    self, action: np.ndarray
  ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict]:
    '''Apply action, step world, return result.'''
    state_before = self._build_reward_state()

    self.coins.tick()

    self._replenish_base_materials()

    decoded = self.action_handler.decode(action)
    action_info = self.action_handler.validate_and_apply(
      decoded, self.stash, self.coins, self.slots
    )

    completed = self.slots.tick()
    for item_id, qty in completed:
      self.stash.add(item_id, qty)
      self.targets.deliver(item_id, qty)

    self.current_tick += 1

    state_after = self._build_reward_state()
    reward, reward_breakdown = self.reward_shaper.compute(
      state_before, action_info, state_after
    )

    terminated = self.targets.is_complete()
    truncated = self.current_tick >= self._max_ticks

    obs = self.obs_builder.build_obs(
      self.stash, self.coins, self.slots, self.targets, self.current_tick
    )

    info = {
      'reward_components': reward_breakdown,
      'action_info': action_info,
      'completed_items': completed,
      'tick': self.current_tick,
    }

    return obs, reward, terminated, truncated, info

  def action_masks(self) -> np.ndarray:
    '''Return mask of valid actions.'''
    return self.action_handler.compute_action_mask(
      self.stash, self.coins, self.slots
    )

  def _replenish_base_materials(self) -> None:
    '''Refill base materials to the cap.'''
    replenish_base_materials(self.stash)

  def _build_reward_state(self) -> Dict:
    '''Build state snapshot for reward shaping.'''
    return {
      'stash': self.stash.as_array(),
      'active_slot_count': self.slots.active_count(),
      'fraction_complete': self.targets.fraction_complete(),
      'targets_complete': self.targets.is_complete(),
      'targets_remaining': self.targets.targets_remaining_array(),
      'targets_total': self.targets.targets_total_array(),
      'tick': self.current_tick,
      'max_ticks': self._max_ticks,
    }

  def render(self) -> Optional[str]:
    '''Render a text summary of state.'''
    lines = [
      f'=== Tick {self.current_tick}/{self._max_ticks} ===',
      f'Coins: {self.coins.balance}',
      f'Active slots: {self.slots.active_count()}/{NUM_CRAFTABLE}',
      f'Target progress: {self.targets.fraction_complete():.1%}',
      'Stash (non-zero):',
    ]
    stash_arr = self.stash.as_array()
    for item_id in ItemId:
      count = stash_arr[int(item_id)]
      if count > 0:
        lines.append(f'  {item_id.name}: {count}')
    return '\n'.join(lines)
