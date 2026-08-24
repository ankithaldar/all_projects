#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Optimal binning by dynamic programming.

Maximises total Information Value over contiguous groupings of a provisional
grid. Because the items are already ordered by feature value, every candidate
binning is a contiguous partition, so the maximum can be found exactly with a
shortest-path search instead of enumerating binning after binning. The result is
optimal with respect to the provisional grid, not over all possible binning.
"""


# imports
import pandas as pd
import numpy as np
from typing import Tuple
#    script imports
from .base import (BinningStrategy, counts_by_position, edges_for_boundaries,
                   full_edges, iv_from_counts, optimal_partition, prebin,
                   single_bin)
# imports


# constants
# constants


# classes
class OptimalDPBinning(BinningStrategy):
  """Optimal binning by dynamic programming."""

  def __init__(self, min_bin_size: float = 0.05, max_bins: int = 10,
               enforce_monotonicity: bool = False, **kwargs):
    super().__init__(min_bin_size, max_bins, enforce_monotonicity)

  def bin(self, x: pd.Series, y: pd.Series) -> Tuple[pd.Series, list]:
    """Bin the feature to maximise total Information Value.

    Args:
      X: Numeric feature.
      y: Binary target.

    Returns:
      tuple: Binned series and the resulting bin edges.
    """
    clean = x.dropna()
    if len(clean) == 0 or clean.nunique() <= 1:
      return x.copy(), []

    # The dynamic program is O(n^2 * k) in the grid size, so a fine grid stays
    # affordable here where an exhaustive search would not.
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

    def cost(start: int, stop: int) -> float:
      """Negated IV of the group spanning provisional items [start, stop)."""
      g = goods_prefix[stop] - goods_prefix[start]
      b = bads_prefix[stop] - bads_prefix[start]
      return -iv_from_counts(
        np.array([g]), np.array([b]), total_goods, total_bads)[0]

    groups = optimal_partition(
      cost, len(goods), self._min_items(len(goods)), self.max_bins)
    if not groups:
      return x.copy(), []

    interior = edges_for_boundaries(
      prebin_edges, [start for start, _ in groups[1:]])
    if not interior:
      # No split improves on a single bin, so say so rather than falling back
      # to the raw values, which would read as one bin per distinct value.
      return single_bin(x)

    edges = full_edges(interior)
    return pd.cut(x, bins=edges), edges

  def _min_items(self, n_items: int) -> int:
    """Smallest number of provisional items a bin may contain.

    The floor is expressed in provisional items rather than rows, because that
    is the unit the partition search works in. The provisional grid is cut at
    quantiles, so each item holds roughly 1 / n_items of the data and a floor of
    ceil(min_bin_size * n_items) items is the population floor restated in the
    grid's own units. Converting from rows instead would overshoot the grid size
    and leave no feasible partition at all.

    Args:
      n_items: Number of provisional items in the grid.

    Returns:
      int: Minimum items per group, at least one.
    """
    return max(1, int(np.ceil(self.min_bin_size * n_items)))
# classes
