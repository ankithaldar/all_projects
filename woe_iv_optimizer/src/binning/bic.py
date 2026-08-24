#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Bayesian Information Criterion binning.

Chooses the grouping that minimises a BIC-style score, ``-2 * log-likelihood +
k * log(n)``, where each bin is charged for the parameters it spends. Unlike an
IV-maximising search, this penalises extra bins, which makes it the useful
counterweight when a feature looks strong only because it has been cut finely
enough to memorise noise.
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
class BICBinning(BinningStrategy):
  """Bayesian Information Criterion binning."""

  def __init__(self, min_bin_size: float = 0.05, max_bins: int = 10,
               enforce_monotonicity: bool = False, **kwargs):
    super().__init__(min_bin_size, max_bins, enforce_monotonicity)

  def bin(self, x: pd.Series, y: pd.Series) -> Tuple[pd.Series, list]:
    """Bin the feature to minimise the BIC score.

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
    # Two free parameters per bin: the bin probability and the interior cut.
    penalty = 2.0 * np.log2(total) if total > 0 else 0.0

    def cost(start: int, stop: int) -> float:
      """BIC contribution of items [start, stop)."""
      n_group = count_prefix[stop] - count_prefix[start]
      g_group = goods_prefix[stop] - goods_prefix[start]
      if n_group <= 0:
        return 0.0
      p = min(max(g_group / n_group, 0.0), 1.0)
      if p <= 0.0 or p >= 1.0:
        neg2_loglik = 0.0
      else:
        neg2_loglik = 2.0 * n_group * (
          -p * np.log2(p) - (1.0 - p) * np.log2(1.0 - p))
      return neg2_loglik + penalty

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
