#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Episode metrics tracker producing balance and reward plots.'''

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


class SimulationTracker:
  '''Accumulates per-tick balances across episodes for later plotting.

  Attributes:
    episode_length: Number of simulation ticks tracked per episode.
    facility_names: Ordered facility ids (must be stable across samples).
    global_balances: Matrix of shape ``(n_episodes, episode_length)``.
    step_balances: Tensor of shape ``(n_episodes, episode_length, n_facilities)``.
  '''

  def __init__(self, episode_length: int, n_episodes: int, facility_names: list[str]) -> None:
    '''Preallocate tracking buffers.

    Args:
      episode_length: Ticks per episode.
      n_episodes: Episodes that will be recorded.
      facility_names: Stable ordered facility ids.

    Raises:
      ValueError: If ``facility_names`` is empty.
    '''
    if not facility_names:
      raise ValueError('facility_names must not be empty')
    self.episode_length = episode_length
    self.facility_names = list(facility_names)
    self.global_balances = np.zeros((n_episodes, episode_length))
    self.step_balances = np.zeros((n_episodes, episode_length, len(self.facility_names)))

  def add_sample(
      self,
      episode: int,
      tick: int,
      global_balance: float,
      rewards_by_facility: dict[str, float],
  ) -> None:
    '''Record one tick of balance data.

    Args:
      episode: Episode index.
      tick: Tick index inside the episode.
      global_balance: System-wide net balance at this tick.
      rewards_by_facility: Per-facility step totals keyed by facility id.

    Raises:
      AssertionError: If facility order differs from initialization.
    '''
    assert self.facility_names == list(rewards_by_facility.keys()), 'Facility order must be preserved'
    self.global_balances[episode, tick] = global_balance
    self.step_balances[episode, tick, :] = np.array(list(rewards_by_facility.values()))

  def render(self) -> None:
    '''Plot global balance, cumulative reward, and per-agent breakdown.'''
    _, axes = plt.subplots(3, 1, figsize=(16, 12))
    ticks = np.linspace(0, self.episode_length, self.episode_length)

    axes[0].set_title('Global balance')
    axes[0].plot(ticks, self.global_balances.T)

    axes[1].set_title('Cumulative sum of rewards')
    cumulative = np.cumsum(np.sum(self.step_balances, axis=2), axis=1)
    axes[1].plot(ticks, cumulative.T)

    axes[2].set_title('Reward breakdown by agent (first episode)')
    axes[2].plot(ticks, np.cumsum(self.step_balances[0], axis=0))
    axes[2].legend(self.facility_names, loc='upper left')

    plt.show()
