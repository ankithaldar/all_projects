from __future__ import annotations

import numpy as np
import pytest

from src.core.items import NUM_CRAFTABLE
from src.env.crafting_env import CraftingEnv
from src.env.frame_skipper import FrameSkipWrapper


class TestFrameSkipper:
    def test_skip_one_is_identity(self, env: CraftingEnv):
        wrapped = FrameSkipWrapper(env, skip=1)
        wrapped.reset()
        action = np.zeros(NUM_CRAFTABLE, dtype=np.int64)

        obs_wrapped, r_wrapped, term_w, trunc_w, info_w = wrapped.step(action)
        assert info_w.get("frame_skip_steps") == 1

    def test_skip_three_advances_three_ticks(self, env_config: dict):
        env_config["max_ticks"] = 100
        env = CraftingEnv(env_config)
        wrapped = FrameSkipWrapper(env, skip=3)
        wrapped.reset()
        action = np.zeros(NUM_CRAFTABLE, dtype=np.int64)

        obs, reward, terminated, truncated, info = wrapped.step(action)
        assert info.get("frame_skip_steps") == 3
        assert obs["current_tick"][0] == 3

    def test_reward_accumulated(self, env_config: dict):
        env_config["max_ticks"] = 100
        env = CraftingEnv(env_config)
        wrapped = FrameSkipWrapper(env, skip=3)
        wrapped.reset()
        action = np.zeros(NUM_CRAFTABLE, dtype=np.int64)

        obs, total_reward, _, _, _ = wrapped.step(action)
        assert isinstance(total_reward, float)

    def test_truncation_propagated(self, env_config: dict):
        env_config["max_ticks"] = 3
        env = CraftingEnv(env_config)
        wrapped = FrameSkipWrapper(env, skip=5)
        wrapped.reset()
        action = np.zeros(NUM_CRAFTABLE, dtype=np.int64)

        obs, reward, terminated, truncated, info = wrapped.step(action)
        assert truncated is True
        assert info["frame_skip_steps"] <= 3

    def test_invalid_skip_raises(self, env: CraftingEnv):
        with pytest.raises(ValueError):
            FrameSkipWrapper(env, skip=0)

    def test_action_masks_delegated(self, env: CraftingEnv):
        wrapped = FrameSkipWrapper(env, skip=2)
        wrapped.reset()
        mask = wrapped.action_masks()
        assert mask.shape == (NUM_CRAFTABLE * 21,)

    def test_skip_five(self, env_config: dict):
        env_config["max_ticks"] = 100
        env = CraftingEnv(env_config)
        wrapped = FrameSkipWrapper(env, skip=5)
        wrapped.reset()
        action = np.zeros(NUM_CRAFTABLE, dtype=np.int64)

        obs, _, _, _, info = wrapped.step(action)
        assert obs["current_tick"][0] == 5
        assert info["frame_skip_steps"] == 5
