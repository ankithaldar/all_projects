#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Minimum Description Length binning.

Adopts the MDL principle of Fearnhead and Liu, *Minimum description length
discretisation for ordinal and nominal data*: a binning is good when the
distribution it implies is short to describe. The score balances two competing
pressures, the cost of describing which observations fall together and the cost
of describing the target inside each group, so it naturally balances bin count
against bin purity rather than chasing IV alone.
"""


# imports
import pandas as pd
import numpy as np
from typing import Tuple
#    script imports
from .base import (BinningStrategy, counts_by_position, edges_for_boundaries,
                   full_edges, optimal_partition, prebin, single_bin)
# imports


# constants
# constants


# classes
class MDLBinning(BinningStrategy):
  """Minimum Description Length binning."""

  def __init__(self, min_bin_size: float = 0.05, max_bins: int = 10,
               enforce_monotonicity: bool = False, **kwargs):
    super().__init__(min_bin_size, max_bins, enforce_monotonicity)

  def bin(self, x: pd.Series, y: pd.Series) -> Tuple[pd.Series, list]:
    """Bin the feature to minimise total description length.

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

    counts = goods + bads
    total = counts.sum()

    count_prefix = np.concatenate([[0.0], np.cumsum(counts)])
    goods_prefix = np.concatenate([[0.0], np.cumsum(goods)])

    def entropy(goods_count: float, total_count: float) -> float:
      """Binary entropy in bits of a group, safe at the boundaries."""
      if total_count <= 0:
        return 0.0
      p = min(max(goods_count / total_count, 0.0), 1.0)
      if p <= 0.0 or p >= 1.0:
        return 0.0
      return -p * np.log2(p) - (1.0 - p) * np.log2(1.0 - p)

    # Cost of specifying the grouping, paid once per group.
    group_spec = np.log2(total) if total > 0 else 0.0

    def cost(start: int, stop: int) -> float:
      """Description length of items [start, stop)."""
      n_group = count_prefix[stop] - count_prefix[start]
      g_group = goods_prefix[stop] - goods_prefix[start]
      if n_group <= 0:
        return 0.0
      return (n_group * np.log2(n_group)
              + n_group * entropy(g_group, n_group)
              + group_spec)

    groups = optimal_partition(
      cost, len(counts), self._min_items(len(counts)), self.max_bins)
    if not groups:
      return x.copy(), []

    interior = edges_for_boundaries(
      prebin_edges, [start for start, _ in groups[1:]])
    if not interior:
      # The search concluded the feature is not worth splitting.
      return single_bin(x)

    edges = full_edges(interior)
    return pd.cut(x, bins=edges), edges

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
