from __future__ import annotations

import numpy as np
import pytest

from src.core.items import ItemId, NUM_CRAFTABLE
from src.cat_game_env.crafting_env import CraftingEnv


class TestCraftingEnv:
  def test_reset_returns_obs_and_info(self, env: CraftingEnv):
    obs, info = env.reset()
    assert isinstance(obs, dict)
    assert isinstance(info, dict)

  def test_obs_in_space(self, env: CraftingEnv):
    obs, _ = env.reset()
    assert env.observation_space.contains(obs)

  def test_action_space_shape(self, env: CraftingEnv):
    assert env.action_space.shape == (NUM_CRAFTABLE,)

  def test_step_returns_tuple(self, env: CraftingEnv):
    env.reset()
    action = env.action_space.sample()
    result = env.step(action)
    assert len(result) == 5
    obs, reward, terminated, truncated, info = result

  def test_step_obs_in_space(self, env: CraftingEnv):
    env.reset()
    action = np.zeros(NUM_CRAFTABLE, dtype=np.int64)
    obs, _, _, _, _ = env.step(action)
    assert env.observation_space.contains(obs)

  def test_coins_increase_per_tick(self, env: CraftingEnv):
    obs, _ = env.reset()
    initial_coins = obs["coins"][0]
    action = np.zeros(NUM_CRAFTABLE, dtype=np.int64)
    obs, _, _, _, _ = env.step(action)
    assert obs["coins"][0] > initial_coins

  def test_base_materials_replenished(self, env: CraftingEnv):
    obs, _ = env.reset()
    action = np.zeros(NUM_CRAFTABLE, dtype=np.int64)
    obs, _, _, _, _ = env.step(action)
    assert obs["stash"][int(ItemId.COTTON)] > 0
    assert obs["stash"][int(ItemId.TREE)] > 0
    assert obs["stash"][int(ItemId.ROCK)] > 0
    assert obs["stash"][int(ItemId.QUARTZ)] > 0

  def test_crafting_string(self, env: CraftingEnv):
    env.reset()
    action = np.zeros(NUM_CRAFTABLE, dtype=np.int64)

    # First step: start crafting string (batch=1)
    action[0] = 1  # STRING
    obs, reward, _, _, info = env.step(action)

    applied = info["action_info"]["applied"]
    assert ItemId.STRING in applied

  def test_truncated_at_max_ticks(self, env_config: dict):
    env_config["max_ticks"] = 5
    env = CraftingEnv(env_config)
    env.reset()

    action = np.zeros(NUM_CRAFTABLE, dtype=np.int64)
    for i in range(5):
      obs, reward, terminated, truncated, info = env.step(action)

    assert truncated is True

  def test_action_masks_shape(self, env: CraftingEnv):
    env.reset()
    mask = env.action_masks()
    assert mask.shape == (NUM_CRAFTABLE * 21,)
    assert mask.dtype == np.bool_

  def test_action_masks_zero_always_valid(self, env: CraftingEnv):
    env.reset()
    mask = env.action_masks()
    for i in range(NUM_CRAFTABLE):
      assert mask[i * 21]

  def test_reward_is_finite(self, env: CraftingEnv):
    env.reset()
    action = env.action_space.sample()
    _, reward, _, _, _ = env.step(action)
    assert np.isfinite(reward)

  def test_stash_never_negative(self, env: CraftingEnv):
    env.reset()
    for _ in range(50):
      action = env.action_space.sample()
      obs, _, terminated, truncated, _ = env.step(action)
      assert np.all(obs["stash"] >= 0)
      if terminated or truncated:
        break

  def test_render(self, env: CraftingEnv):
    env.reset()
    output = env.render()
    assert isinstance(output, str)
    assert "Tick" in output

  def test_craft_chain_cotton_to_string(self, env: CraftingEnv):
    env.reset()
    action = np.zeros(NUM_CRAFTABLE, dtype=np.int64)
    action[0] = 1  # STRING batch=1
    obs, _, _, _, _ = env.step(action)

    # STRING takes 1 tick (5 min / 5 min = 1 tick), so it should complete
    # after the tick() call in this same step
    has_string = obs["stash"][int(ItemId.STRING)] >= 1
    assert has_string or env.slots.is_busy(ItemId.STRING)

  def test_info_has_reward_components(self, env: CraftingEnv):
    env.reset()
    action = np.zeros(NUM_CRAFTABLE, dtype=np.int64)
    _, _, _, _, info = env.step(action)
    assert "reward_components" in info
    assert "total" in info["reward_components"]

  def test_reset_deterministic(self, env: CraftingEnv):
    obs1, _ = env.reset(seed=42)
    obs2, _ = env.reset(seed=42)
    for key in obs1:
      assert np.array_equal(obs1[key], obs2[key])

  def test_reset_clears_state(self, env: CraftingEnv):
    env.reset()
    action = np.zeros(NUM_CRAFTABLE, dtype=np.int64)
    action[0] = 1
    for _ in range(10):
      env.step(action)
    obs, _ = env.reset()
    assert env.current_tick == 0
    assert obs["coins"][0] == 0
    assert obs["current_tick"][0] == 0

  def test_obs_within_space_bounds(self, env: CraftingEnv):
    env.reset()
    for _ in range(20):
      action = env.action_space.sample()
      obs, _, terminated, truncated, _ = env.step(action)
      assert env.observation_space.contains(obs)
      if terminated or truncated:
        break

  def test_coins_increment_exact(self, env: CraftingEnv):
    env.reset()
    action = np.zeros(NUM_CRAFTABLE, dtype=np.int64)
    obs, _, _, _, _ = env.step(action)
    assert obs["coins"][0] == 210
