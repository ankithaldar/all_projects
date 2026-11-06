#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Frame-skipping Gym wrapper.'''

from __future__ import annotations

from typing import Any, Dict, Tuple

import gymnasium as gym
import numpy as np


class FrameSkipWrapper(gym.Wrapper):
  '''Repeat each action for a fixed skip.'''
  def __init__(self, env: gym.Env, skip: int = 1):
    '''Initialize with env and skip count.'''
    super().__init__(env)
    if skip < 1:
      raise ValueError(f'Frame skip must be >= 1, got {skip}')
    self._skip = skip

  @property
  def skip(self) -> int:
    '''Return the skip count.'''
    return self._skip

  def step(
    self, action: np.ndarray
  ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict]:
    '''Step wrapped env skip times, sum reward.'''
    total_reward = 0.0
    all_info: Dict[str, Any] = {}

    for i in range(self._skip):
      obs, reward, terminated, truncated, info = self.env.step(action)
      total_reward += reward
      all_info = info

      if terminated or truncated:
        break

    all_info['frame_skip_steps'] = i + 1
    return obs, total_reward, terminated, truncated, all_info

  def action_masks(self) -> np.ndarray:
    '''Delegate action mask to wrapped env.'''
    return self.env.action_masks()
