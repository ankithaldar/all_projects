#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Completion reward for delivered cartons.'''

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from truck_carton.reward.calculator import (
    EnvironmentState,
  )


class CompletionReward:
  '''Placement progress ratio, 1.0 when all placed.'''

  def compute(self, state: EnvironmentState) -> float:
    '''Compute.'''
    if state.total_cartons <= 0:
      return 0.0

    known_ids = {c.carton_id for c in state.all_cartons}
    placed_known = sum(
      1 for cid in state.placed_cartons if cid in known_ids
    )
    fraction = min(
      placed_known / state.total_cartons,
      1.0,
    )
    all_placed = placed_known == state.total_cartons
    # Scale so 100% placement yields 1.0, with
    # a steeper curve rewarding near-completion.
    if all_placed:
      return 1.0
    return fraction * 0.5
