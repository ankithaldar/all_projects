#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Monotonic binning via Pool Adjacent Violators Algorithm.

Fits a monotone trend to the Weight of Evidence curve rather than searching for
bins that happen to come out monotonic. The distinction matters: a search that
only accepts already-monotonic candidates discards the rest and can end up with
no valid answer at all, whereas PAVA always returns a solution because it
constructs one.
"""


# imports
import pandas as pd
import numpy as np
from typing import List, Tuple
#    script imports
from .base import (BinningStrategy, counts_by_position, edges_for_boundaries,
                   full_edges, prebin, single_bin, woe_from_counts)
# imports


# constants
# constants


# classes
class MonotonicPavaBinning(BinningStrategy):
  """Monotonic binning via Pool Adjacent Violators Algorithm."""

  def __init__(self, min_bin_size: float = 0.05, max_bins: int = 10,
               enforce_monotonicity: bool = True, **kwargs):
    super().__init__(min_bin_size, max_bins, enforce_monotonicity)

  def bin(self, x: pd.Series, y: pd.Series) -> Tuple[pd.Series, list]:
    """Bin the feature so that the WoE trend is monotone.

    Starts from quantile bins, pools adjacent WoE violations until the trend is
    non-decreasing, then collapses bins that share a fitted value.

    Args:
      X: Numeric feature.
      y: Binary target.

    Returns:
      tuple: Binned series and the resulting bin edges. Edges are empty when
      the feature had too few distinct values to bin.
    """
    clean = x.dropna()
    if len(clean) == 0 or clean.nunique() <= 1:
      return x.copy(), []

    positions, prebin_edges = prebin(x, self.max_bins)
    if positions is None:
      return x.copy(), []

    goods, bads = counts_by_position(positions, y)
    populated = np.flatnonzero(goods + bads > 0)
    if len(populated) < 2:
      return x.copy(), []

    total_goods, total_bads = goods.sum(), bads.sum()
    weights = (goods + bads)[populated]
    woe = woe_from_counts(
      goods[populated], bads[populated], total_goods, total_bads)

    # PAVA fits a non-decreasing trend, so on a feature whose WoE genuinely
    # falls with the feature value it would pool every bin into one and throw
    # away the whole signal. Fitting the negated curve as well covers the
    # descending case, and the direction that keeps more bins is the one used.
    rising = self._pool_adjacent_violators(woe, weights)
    falling = self._pool_adjacent_violators(-woe, weights)
    blocks = rising if len(rising) >= len(falling) else falling
    if self.enforce_monotonicity:
      blocks = self._merge_flat_blocks(blocks)

    interior = edges_for_boundaries(
      prebin_edges, [b[0] for b in blocks[1:]])
    if not interior:
      # A monotone fit that pools everything carries no ordering signal.
      return single_bin(x)

    edges = full_edges(interior)
    return pd.cut(x, bins=edges), edges

  @staticmethod
  def _pool_adjacent_violators(
    values: np.ndarray, weights: np.ndarray
  ) -> List[tuple]:
    """Run PAVA to obtain a non-decreasing fit.

    Args:
      values: WoE per provisional bin, ordered by ascending feature value.
      weights: Population of each provisional bin.

    Returns:
      list: ``(start, stop, fitted_value, weight)`` blocks in ascending order,
      where the fitted value is non-decreasing across blocks.
    """
    blocks: List[list] = []
    for i, (value, weight) in enumerate(zip(values, weights)):
      blocks.append([i, i + 1, float(value), float(weight)])
      # Pool backwards for as long as the monotone fit is violated.
      while len(blocks) > 1 and blocks[-2][2] > blocks[-1][2]:
        right = blocks.pop()
        left = blocks.pop()
        total_weight = left[3] + right[3]
        if total_weight <= 0:
          fitted = 0.0
        else:
          fitted = (left[2] * left[3] + right[2] * right[3]) / total_weight
        blocks.append([left[0], right[1], fitted, total_weight])
    return [(b[0], b[1], b[2], b[3]) for b in blocks]

  @staticmethod
  def _merge_flat_blocks(blocks: list) -> List[tuple]:
    """Collapse neighbouring blocks that ended up on the same fitted value.

    Two adjacent bins whose fitted WoE is identical carry no information, so
    they become one bin.

    Args:
      blocks: Blocks from :meth:`_pool_adjacent_violators`.

    Returns:
      list: Blocks with consecutive equal fitted values merged.
    """
    merged: List[list] = []
    for start, stop, fitted, weight in blocks:
      if merged and abs(merged[-1][2] - fitted) < 1e-12:
        previous = merged.pop()
        merged.append([previous[0], stop, previous[2], previous[3] + weight])
      else:
        merged.append([start, stop, fitted, weight])
    return [(b[0], b[1], b[2], b[3]) for b in merged]
# classes
