#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Stable-Baselines3 training callbacks.'''

from __future__ import annotations

import os
from stable_baselines3.common.callbacks import BaseCallback


class TensorBoardCallback(BaseCallback):
  '''Logs reward and action aggregates to TensorBoard.'''

  def _on_step(self) -> bool:
    '''Logs metrics for the current rollout step.

    Returns:
      True always, so training continues.
    '''
    infos = self.locals.get('infos', [])
    if not infos:
      return True

    reward_sums = {}
    reward_counts = {}
    applied_total = 0
    rejected_total = 0
    last_tick = 0

    for info in infos:
      reward_components = info.get('reward_components', {})
      for key, value in reward_components.items():
        if isinstance(value, (int, float)):
          reward_sums[key] = reward_sums.get(key, 0.0) + value
          reward_counts[key] = reward_counts.get(key, 0) + 1

      action_info = info.get('action_info', {})
      applied_total += len(action_info.get('applied', {}))
      rejected_total += len(action_info.get('rejected', {}))
      last_tick = max(last_tick, info.get('tick', 0))

    n = len(infos)
    for key, cnt in reward_counts.items():
      self.logger.record(
        f'reward/{key}', reward_sums[key] / cnt if cnt else 0.0
      )

    self.logger.record('actions/applied_count', applied_total / n)
    self.logger.record('actions/rejected_count', rejected_total / n)
    self.logger.record('env/tick', last_tick)
    return True


class CheckpointCallback(BaseCallback):
  '''Saves model checkpoints at a fixed step frequency.'''

  def __init__(
    self,
    save_freq: int = 50000,
    save_path: str = 'output/models',
    verbose: int = 0,
  ):
    '''Initialize checkpoint frequency and path.'''
    super().__init__(verbose)
    self._save_freq = save_freq
    self._save_path = save_path
    os.makedirs(save_path, exist_ok=True)

  def _on_step(self) -> bool:
    '''Save checkpoint when due.'''
    if self.n_calls % self._save_freq == 0:
      path = os.path.join(
        self._save_path, f'checkpoint_{self.n_calls}'
      )
      self.model.save(path)
      if self.verbose:
        print(f'Saved checkpoint to {path}')
    return True
