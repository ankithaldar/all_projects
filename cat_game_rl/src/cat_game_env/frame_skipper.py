from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np


class FrameSkipWrapper(gym.Wrapper):
  def __init__(self, env: gym.Env, skip: int = 1):
    super().__init__(env)
    if skip < 1:
      raise ValueError(f"Frame skip must be >= 1, got {skip}")
    self._skip = skip

  @property
  def skip(self) -> int:
    return self._skip

  def step(
    self, action: np.ndarray
  ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict]:
    total_reward = 0.0
    all_info: Dict[str, Any] = {}

    for i in range(self._skip):
      obs, reward, terminated, truncated, info = self.env.step(action)
      total_reward += reward
      all_info = info

      if terminated or truncated:
        break

    all_info["frame_skip_steps"] = i + 1
    return obs, total_reward, terminated, truncated, all_info

  def action_masks(self) -> np.ndarray:
    return self.env.action_masks()
