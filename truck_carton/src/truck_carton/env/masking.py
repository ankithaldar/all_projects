#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Action mask provider for masked policy training.'''

from __future__ import annotations

import numpy as np

from truck_carton.env.action import ActionManager


class ActionMaskProvider:
  '''Provides action masks from an action manager.

  Thin adapter kept intentionally: packing_env.py
  constructs this provider as its mask seam and
  tests/test_masking.py exercises it directly, so
  removing it would be a breaking change for a
  non-owned caller. It centralizes the delegation
  plus the has_valid_actions convenience.

  Attributes:
    action_manager: Action manager supplying candidates.
  '''

  def __init__(self, action_manager: ActionManager) -> None:
    '''Initialize with an action manager.

    Args:
      action_manager: Action manager supplying candidates.
    '''
    self.action_manager = action_manager

  def get_mask(self) -> np.ndarray:
    '''Return the current action mask.

    Returns:
      Boolean mask over the unified action space.
    '''
    return self.action_manager.get_action_mask()

  @property
  def has_valid_actions(self) -> bool:
    '''Return whether any action is currently valid.

    Returns:
      True when at least one action is unmasked.
    '''
    return bool(self.get_mask().any())
