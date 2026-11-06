#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Curriculum manager promoting RL training stages.'''

from __future__ import annotations

import math
from collections import deque

import numpy as np

from truck_carton.config import (
  CurriculumConfig,
  CurriculumStage,
)


class CurriculumManager:
  '''Manages progression through curriculum
  stages based on sustained reward thresholds.'''

  def __init__(self, config: CurriculumConfig) -> None:
    '''Initialize the instance.'''
    if not config.stages:
      raise ValueError('curriculum stages must be non-empty')
    for stage in config.stages:
      if stage.promotion_window < 1:
        raise ValueError(
          'promotion_window must be >= 1 '
          f'(got {stage.promotion_window})'
        )
    self._config = config
    self.current_stage: int = 0
    self._episode_rewards: deque[float] = deque(
      maxlen=config.stages[0].promotion_window
    )
    self._episodes_since_promotion: int = 0
    self.total_episodes: int = 0

  @property
  def stage(self) -> CurriculumStage:
    '''Stage.'''
    return self._config.stages[self.current_stage]

  @property
  def is_final_stage(self) -> bool:
    '''Is final stage.'''
    return self.current_stage >= (len(self._config.stages) - 1)

  @property
  def num_stages(self) -> int:
    '''Num stages.'''
    return len(self._config.stages)

  @property
  def mean_reward(self) -> float:
    '''Mean reward.'''
    if not self._episode_rewards:
      return 0.0
    return float(np.mean(self._episode_rewards))

  def record_episode(self, reward: float) -> bool:
    '''Record episode.'''
    if not math.isfinite(reward):
      raise ValueError(
        f'reward must be finite (got {reward!r})'
      )
    self._episode_rewards.append(reward)
    self.total_episodes += 1
    self._episodes_since_promotion += 1

    if self.is_final_stage:
      return False

    if (
      self._episodes_since_promotion
      >= (self.stage.promotion_window)
      and self.mean_reward >= self.stage.promotion_threshold
    ):
      return self._promote()

    return False

  def _promote(self) -> bool:
    '''Promote.'''
    if self.is_final_stage:
      return False

    self.current_stage += 1
    self._episodes_since_promotion = 0
    return True
