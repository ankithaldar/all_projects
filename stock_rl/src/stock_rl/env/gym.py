#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Gymnasium-compatible protocol, in the standard library.

RL libraries expect a specific interface. Rather than take a dependency
to get one, this module declares the two Protocols that matter. Any
environment implementing ``Env`` and any algorithm expecting one can be
used together, and a future migration to Gymnasium or Stable-Baselines3
means deleting this file rather than rewriting every environment.

Only the surface actually used by :mod:`stock_rl.env.nse` is declared.
Declaring the full Gymnasium API would be speculative.
'''

from __future__ import annotations

from typing import Protocol, runtime_checkable

__all__ = ['Env', 'Info', 'Obs']


#: Observation returned by ``reset`` and ``step``.
Obs = dict[str, object]

#: Auxiliary values returned alongside the observation and reward. The
#: key name matches Gymnasium so adapters stay trivial.
Info = dict[str, object]


@runtime_checkable
class Env(Protocol):
  '''The subset of the Gymnasium environment interface we rely on.

  Attributes:
    action_symbols: Symbols the agent may act on, in a fixed order, so
      that an action vector is always interpretable.
    n_steps: Length of one episode in steps.
  '''

  action_symbols: tuple[str, ...]
  n_steps: int

  def reset(self, seed: int | None = None) -> Obs:
    '''Reset to the start of an episode and return the first observation.

    Args:
      seed: Optional seed. Implementations that have stochastic
        behaviour must make the episode reproducible from it.

    Returns:
      The initial observation.
    '''

  def step(self, action: dict[str, int]) -> tuple[Obs, float, bool, Info]:
    '''Advance one bar.

    Args:
      action: Target direction per symbol, where -1 is reduce, 0 is
        hold and 1 is add.

    Returns:
      Tuple of (observation, reward, terminated, info).
    '''

  def render(self) -> str:
    '''Return a short human-readable summary of the episode so far.

    Returns:
      A single-line summary. Used by the CLI, not by training.
    '''
