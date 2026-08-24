#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Greedy forward-selection binning.

Grows the binning one cut at a time, each time adding the split point that
improves Information Value most, and stops as soon as another cut stops paying
for itself. Unlike the dynamic-programming strategies this is a genuine greedy
search, so it can find cuts that look good locally but lead nowhere, which makes
it a useful independent check on the optimal solvers.
"""


# imports
import pandas as pd
import numpy as np
from typing import List, Tuple
#    script imports
from .base import (BinningStrategy, counts_by_position, edges_for_boundaries,
                   full_edges, iv_from_counts, prebin, single_bin)
# imports


# constants
# constants


# classes
class GreedyIVBinning(BinningStrategy):
  """Greedy forward-selection binning."""

  def __init__(self, min_bin_size: float = 0.05, max_bins: int = 10,
               enforce_monotonicity: bool = False, min_gain: float = 1e-6,
               **kwargs):
    super().__init__(min_bin_size, max_bins, enforce_monotonicity)
    self.min_gain = min_gain

  def bin(self, x: pd.Series, y: pd.Series) -> Tuple[pd.Series, list]:
    """Bin the feature by greedily adding the most IV-improving cut.

    Args:
      X: Numeric feature.
      y: Binary target.

    Returns:
      tuple: Binned series and the resulting bin edges.
    """
    clean = x.dropna()
    if len(clean) == 0 or clean.nunique() <= 1:
      return x.copy(), []

    positions, prebin_edges = prebin(
      x, max(self.max_bins * 4, self.max_bins + 4))
    if positions is None:
      return x.copy(), []

    goods, bads = counts_by_position(positions, y)
    keep = goods + bads > 0
    if keep.sum() < 2:
      return x.copy(), []
    goods, bads = goods[keep], bads[keep]

    total_goods, total_bads = goods.sum(), bads.sum()
    goods_prefix = np.concatenate([[0.0], np.cumsum(goods)])
    bads_prefix = np.concatenate([[0.0], np.cumsum(bads)])

    def iv_of(start: int, stop: int) -> float:
      """IV of the group spanning provisional items [start, stop)."""
      g = goods_prefix[stop] - goods_prefix[start]
      b = bads_prefix[stop] - bads_prefix[start]
      return iv_from_counts(np.array([g]), np.array([b]),
                            total_goods, total_bads)[0]

    n_items = len(goods)
    min_items = self._min_items(n_items)
    cuts: List[int] = []
    current = iv_of(0, n_items)

    while len(cuts) + 1 < self.max_bins:
      best_gain, best_pos = self.min_gain, None
      for pos in range(min_items, n_items - min_items + 1):
        if self._too_close(pos, cuts, min_items):
          continue
        gain = (iv_of(0, pos) + iv_of(pos, n_items)) - current
        if gain > best_gain:
          best_gain, best_pos = gain, pos
      if best_pos is None:
        break
      cuts.append(best_pos)
      current += best_gain

    if not cuts:
      return x.copy(), []

    interior = edges_for_boundaries(prebin_edges, cuts)
    if not interior:
      # The search concluded the feature is not worth splitting.
      return single_bin(x)

    edges = full_edges(interior)
    return pd.cut(x, bins=edges), edges

  @staticmethod
  def _too_close(pos: int, cuts: list, min_items: int) -> bool:
    """Whether a candidate cut would create a bin below the population floor.

    Args:
      pos: Candidate provisional position.
      cuts: Cuts already chosen, ascending.
      min_items: Minimum provisional items per bin.

    Returns:
      bool: True when the candidate must be rejected.
    """
    if not cuts:
      return pos < min_items
    return any(abs(pos - existing) < min_items for existing in cuts)

  def _min_items(self, n_items: int) -> int:
    """Smallest number of provisional items a bin may contain.

    The floor is expressed in provisional items rather than rows, because that
    is the unit the partition search works in. The provisional grid is cut at
    quantiles, so each item holds roughly 1 / n_items of the data and a floor of
    ceil(min_bin_size * n_items) items is the population floor restated in the
    grid's own units. Converting from rows instead would be off by the grid
    size and would leave no feasible partition at all.

    Args:
      n_items: Number of provisional items in the grid.

    Returns:
      int: Minimum items per group, at least one.
    """
    return max(1, int(np.ceil(self.min_bin_size * n_items)))
# classes
