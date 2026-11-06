#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Travel distance reward for truck routing.'''

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from truck_carton.reward.calculator import (
    EnvironmentState,
  )


class TravelDistanceReward:
  '''Normalized fleet travel distance [0, 1].'''

  def compute(self, state: EnvironmentState) -> float:
    '''Compute.'''
    if (
      not math.isfinite(state.max_possible_distance)
      or state.max_possible_distance <= 0
    ):
      return 0.0
    if (
      not math.isfinite(state.total_travel_distance)
      or state.total_travel_distance <= 0
    ):
      return 0.0
    return min(
      state.total_travel_distance / state.max_possible_distance,
      1.0,
    )
