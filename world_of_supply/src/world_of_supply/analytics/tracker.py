#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Episode metrics tracker producing balance and reward plots.'''

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def _tick_axis(episode_length: int) -> np.ndarray:
  '''Build the x-axis tick positions for one episode.

  Args:
    episode_length: Number of ticks tracked per episode.

  Returns:
    np.ndarray: Tick positions from 0 to ``episode_length - 1``.
  '''
  return np.linspace(0, episode_length, episode_length)


def _cumulative_rewards(step_balances: np.ndarray) -> np.ndarray:
  '''Compute cumulative per-episode reward totals.

  Args:
    step_balances: Tensor of shape ``(n_episodes, episode_length,
      n_facilities)``.

  Returns:
    np.ndarray: Cumulative totals of shape ``(n_episodes,
      episode_length)``.
  '''
  return np.cumsum(np.sum(step_balances, axis=2), axis=1)


def _first_episode_cumulative(step_balances: np.ndarray) -> np.ndarray:
  '''Compute the cumulative reward breakdown for episode zero.

  Args:
    step_balances: Tensor of shape ``(n_episodes, episode_length,
      n_facilities)``.

  Returns:
    np.ndarray: Cumulative per-facility rewards for the first
      episode.
  '''
  return np.cumsum(step_balances[0], axis=0)


def _plot_tracker(tracker: 'SimulationTracker') -> None:
  '''Plot global balance, cumulative reward, and agent breakdown.

  Args:
    tracker: Tracker holding the accumulated balance buffers.

  Returns:
    None: The plot is shown; no value is returned.
  '''
  _, axes = plt.subplots(3, 1, figsize=(16, 12))
  ticks = _tick_axis(tracker.episode_length)

  axes[0].set_title('Global balance')
  axes[0].plot(ticks, tracker.global_balances.T)

  axes[1].set_title('Cumulative sum of rewards')
  cumulative = _cumulative_rewards(tracker.step_balances)
  axes[1].plot(ticks, cumulative.T)

  axes[2].set_title('Reward breakdown by agent (first episode)')
  first = _first_episode_cumulative(tracker.step_balances)
  axes[2].plot(ticks, first)
  axes[2].legend(tracker.facility_names, loc='upper left')

  plt.show()


class SimulationTracker:
  '''Accumulates per-tick balances across episodes for later plotting.

  Attributes:
    episode_length: Number of simulation ticks tracked per episode.
    facility_names: Ordered facility ids (must be stable across
      samples).
    global_balances: Matrix of shape ``(n_episodes,
      episode_length)``.
    step_balances: Tensor of shape ``(n_episodes, episode_length,
      n_facilities)``.
  '''

  def __init__(
      self,
      episode_length: int,
      n_episodes: int,
      facility_names: list[str],
  ) -> None:
    '''Preallocate tracking buffers.

    Args:
      episode_length: Ticks per episode.
      n_episodes: Episodes that will be recorded.
      facility_names: Stable ordered facility ids.

    Returns:
      None: The tracker is mutated in place.

    Raises:
      ValueError: If ``facility_names`` is empty.
    '''
    if not facility_names:
      raise ValueError('facility_names must not be empty')
    self.episode_length = episode_length
    self.facility_names = list(facility_names)
    self.global_balances = np.zeros((n_episodes, episode_length))
    balances_shape = (
        n_episodes,
        episode_length,
        len(self.facility_names),
    )
    self.step_balances = np.zeros(balances_shape)

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
      rewards_by_facility: Per-facility step totals keyed by
        facility id.

    Returns:
      None: The tracker is mutated in place.

    Raises:
      AssertionError: If facility order differs from initialization.
    '''
    assert self.facility_names == list(rewards_by_facility.keys()), (
        'Facility order must be preserved'
    )
    self.global_balances[episode, tick] = global_balance
    reward_values = np.array(list(rewards_by_facility.values()))
    self.step_balances[episode, tick, :] = reward_values

  def render(self) -> None:
    '''Plot global balance, cumulative reward, and per-agent breakdown.

    Delegates to :func:`_plot_tracker` so accumulation stays separate
    from display.

    Returns:
      None: The plot is shown; no value is returned.
    '''
    _plot_tracker(self)
