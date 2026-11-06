#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Observation space and state construction.'''

from __future__ import annotations

from typing import Dict

import numpy as np
from gymnasium import spaces

from src.core.items import NUM_ITEMS, NUM_CRAFTABLE, MAX_AFFORDABLE_BATCH
from src.core.inventory import Stash
from src.core.coin_generator import CoinGenerator
from src.core.slot_scheduler import SlotScheduler
from src.core.target_provider import TargetProvider


class ObservationBuilder:
  '''Builds observation spaces and states.'''
  def __init__(self, max_ticks: int = 8064):
    '''Initialize with episode tick limit.'''
    self._max_ticks = max_ticks

  def build_space(self) -> spaces.Dict:
    '''Return the observation space spec.'''
    return spaces.Dict({
      'stash': spaces.Box(
        low=0, high=MAX_AFFORDABLE_BATCH, shape=(NUM_ITEMS,), dtype=np.int32
      ),
      'coins': spaces.Box(
        low=0, high=99_999_999, shape=(1,), dtype=np.int32
      ),
      'slots': spaces.Box(
        low=0, high=1000, shape=(NUM_CRAFTABLE, 3), dtype=np.int32
      ),
      'time_fraction': spaces.Box(
        low=0.0, high=1.0, shape=(1,), dtype=np.float32
      ),
      'current_tick': spaces.Box(
        low=0, high=self._max_ticks, shape=(1,), dtype=np.int32
      ),
      'targets_remaining': spaces.Box(
        low=-MAX_AFFORDABLE_BATCH, high=MAX_AFFORDABLE_BATCH,
        shape=(NUM_ITEMS,), dtype=np.int32
      ),
      'targets_total': spaces.Box(
        low=0, high=MAX_AFFORDABLE_BATCH, shape=(NUM_ITEMS,), dtype=np.int32
      ),
    })

  def build_obs(
    self,
    stash: Stash,
    coins: CoinGenerator,
    slots: SlotScheduler,
    targets: TargetProvider,
    tick: int,
  ) -> Dict[str, np.ndarray]:
    '''Build observation dict for current state.'''
    time_frac = tick / max(self._max_ticks, 1)
    return {
      'stash': np.clip(stash.as_array(), 0, MAX_AFFORDABLE_BATCH),
      'coins': np.array([coins.balance], dtype=np.int32),
      'slots': slots.get_slot_array(),
      'time_fraction': np.array([time_frac], dtype=np.float32),
      'current_tick': np.array([tick], dtype=np.int32),
      'targets_remaining': np.clip(
        targets.targets_remaining_array(),
        -MAX_AFFORDABLE_BATCH, MAX_AFFORDABLE_BATCH
      ),
      'targets_total': targets.targets_total_array(),
    }
