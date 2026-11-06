#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Action-mask function for the crafting env.'''

from __future__ import annotations

from typing import TYPE_CHECKING, Union

import numpy as np

from src.cat_game_env.crafting_env import CraftingEnv

if TYPE_CHECKING:
  from src.cat_game_env.multi_agent import TierEnv


def compute_action_mask(
  env: Union[CraftingEnv, 'TierEnv'],
) -> np.ndarray:
  '''Return the action mask for an environment.

  This is the single source of truth for action masking. All
  agents wrap their environment with this function, so the mask
  logic lives in exactly one place.

  Args:
    env: Environment exposing an ``action_masks()`` method,
      such as a crafting or tier environment.

  Returns:
    The action-mask array returned by ``env.action_masks()``.
  '''
  return env.action_masks()
