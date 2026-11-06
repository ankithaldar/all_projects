#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Stable-Baselines3 training callbacks.'''

from __future__ import annotations

import logging
import os
from typing import Any

from stable_baselines3.common.callbacks import BaseCallback

_LOGGER = logging.getLogger(__name__)


class TensorBoardCallback(BaseCallback):
  '''Logs reward and action aggregates to TensorBoard.'''

  @property
  def logger(self) -> Any:
    '''Return the SB3-configured or injected logger.

    Returns:
      The logger to record metrics with, or ``None`` when no
      logger is available.
    '''
    logger = self.__dict__.get('_logger')
    if logger is not None:
      return logger
    try:
      return super().logger
    except AttributeError:
      return logger

  @logger.setter
  def logger(self, value: Any) -> None:
    '''Store logger, allowing injection in tests.

    SB3 exposes ``logger`` as a read-only property on some
    versions, so this setter stores to ``_logger`` to keep
    SB3 behavior while allowing tests to inject a fake
    logger.

    Args:
      value: Logger to record metrics with.
    '''
    self._logger = value

  def _on_step(self) -> bool:
    '''Logs metrics for the current rollout step.

    Aggregates reward components and action counts across
    all infos in the current rollout and records their means.

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
  '''Saves model checkpoints at a fixed step frequency.

  Args:
    save_freq: Save a checkpoint every this many steps.
    save_path: Directory checkpoints are written to.
    verbose: Verbosity level; when truthy, each save is
      logged at INFO level.
  '''

  def __init__(
    self,
    save_freq: int = 50000,
    save_path: str = 'output/models',
    verbose: int = 0,
  ):
    '''Initialize checkpoint frequency and path.

    Args:
      save_freq: Save a checkpoint every this many steps.
      save_path: Directory checkpoints are written to.
      verbose: Verbosity level; when truthy, each save is
        logged at INFO level.
    '''
    super().__init__(verbose)
    self._save_freq = save_freq
    self._save_path = save_path
    os.makedirs(save_path, exist_ok=True)

  def _on_step(self) -> bool:
    '''Save a checkpoint when the step count is due.

    Returns:
      True always, so training continues.
    '''
    if self.n_calls % self._save_freq == 0:
      path = os.path.join(
        self._save_path, f'checkpoint_{self.n_calls}'
      )
      self.model.save(path)
      if self.verbose:
        _LOGGER.info('Saved checkpoint to %s', path)
    return True
