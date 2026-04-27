from __future__ import annotations

import random
from typing import Any

import numpy as np

from src.core.items import CraftingTree, NUM_CRAFTABLE
from src.ga.chromosome import MAX_TICKS


class GaOperators:
  def __init__(self, crafting_tree: CraftingTree, config: dict):
    self._tree = crafting_tree
    self._config = config
    self._block_size = config.get("time_block_size", 48)
    self._mut_batch_indpb = config.get("mut_batch_indpb", 0.02)
    self._mut_time_shift_prob = config.get("mut_time_shift_prob", 0.15)
    self._max_batch = config.get("max_batch_size", 20)

  def cx_time_block(
    self, ind1: np.ndarray, ind2: np.ndarray
  ) -> tuple[np.ndarray, np.ndarray]:
    n_blocks = MAX_TICKS // self._block_size
    for b in range(n_blocks):
      if random.random() < 0.5:
        start = b * self._block_size
        end = min(start + self._block_size, MAX_TICKS)
        ind1[start:end], ind2[start:end] = (
          ind2[start:end].copy(),
          ind1[start:end].copy(),
        )
    return ind1, ind2

  def mut_batch_size(self, individual: np.ndarray) -> tuple[np.ndarray]:
    mask = np.random.random(individual.shape) < self._mut_batch_indpb
    new_vals = np.random.randint(
      0, self._max_batch + 1, size=mask.sum()
    ).astype(np.uint8)
    individual[mask] = new_vals
    return (individual,)

  def mut_time_shift(self, individual: np.ndarray) -> tuple[np.ndarray]:
    if random.random() > self._mut_time_shift_prob:
      return (individual,)

    col = random.randint(0, NUM_CRAFTABLE - 1)
    shift = random.randint(-6, 6)
    if shift == 0 or abs(shift) >= MAX_TICKS:
      return (individual,)

    column = individual[:, col].copy()
    individual[:, col] = 0

    if shift > 0:
      individual[shift:MAX_TICKS, col] = column[: MAX_TICKS - shift]
    else:
      abs_shift = abs(shift)
      individual[: MAX_TICKS - abs_shift, col] = column[abs_shift:MAX_TICKS]

    return (individual,)

  def mutate(self, individual: np.ndarray) -> tuple[np.ndarray]:
    individual = self.mut_batch_size(individual)[0]
    individual = self.mut_time_shift(individual)[0]
    return (individual,)
