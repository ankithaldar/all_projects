from __future__ import annotations

import numpy as np

from src.cat_game_env.crafting_env import CraftingEnv


def compute_action_mask(env: CraftingEnv) -> np.ndarray:
  return env.action_masks()
