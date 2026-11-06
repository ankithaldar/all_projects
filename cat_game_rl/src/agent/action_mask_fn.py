#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Action-mask function for the crafting env.'''

from __future__ import annotations

import numpy as np

from src.cat_game_env.crafting_env import CraftingEnv


def compute_action_mask(env: CraftingEnv) -> np.ndarray:
  '''Return action mask for env.'''
  return env.action_masks()
