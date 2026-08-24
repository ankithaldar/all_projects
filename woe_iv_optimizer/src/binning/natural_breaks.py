#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Fisher-Jenks natural breaks binning.

The classic unsupervised discretisation from Jenks, *The Data Model Concept in
Statistical Mapping* (1977): cut the feature so the spread within bins is as
small as possible, ignoring the target entirely. It is useful as a control in a
strategy comparison, since any IV a feature shows is then clearly coming from
the target rather than from the clustering of the feature's own values.
"""


# imports
import pandas as pd
import numpy as np
from typing import Tuple
#    script imports
from .base import (BinningStrategy, edges_for_boundaries, full_edges,
                   optimal_partition, prebin, single_bin)
# imports


# constants
# constants


# classes
class NaturalBreaksBinning(BinningStrategy):
  """Fisher-Jenks natural breaks binning."""

  def __init__(self, min_bin_size: float = 0.05, max_bins: int = 10,
               enforce_monotonicity: bool = False, **kwargs):
    super().__init__(min_bin_size, max_bins, enforce_monotonicity)

  def bin(self, x: pd.Series, y: pd.Series) -> Tuple[pd.Series, list]:
    """Bin the feature by minimising within-bin variance.

    Args:
      X: Numeric feature.
      y: Binary target, ignored by this strategy.

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

    # Summarise each provisional bin by count, sum and sum of squares, which is
    # all that is needed to evaluate the variance of any contiguous grouping.
    clean = clean.sort_values()
    n_bins = int(positions.max()) + 1
    count = np.zeros(n_bins, dtype=float)
    total = np.zeros(n_bins, dtype=float)
    total_sq = np.zeros(n_bins, dtype=float)

    order = clean.index.to_numpy()
    codes = positions.loc[order].to_numpy()
    values = clean.to_numpy(dtype=float)
    np.add.at(count, codes, 1.0)
    np.add.at(total, codes, values)
    np.add.at(total_sq, codes, values * values)

    keep = count > 0
    if keep.sum() < 2:
      return x.copy(), []
    count, total, total_sq = count[keep], total[keep], total_sq[keep]

    count_prefix = np.concatenate([[0.0], np.cumsum(count)])
    total_prefix = np.concatenate([[0.0], np.cumsum(total)])
    sq_prefix = np.concatenate([[0.0], np.cumsum(total_sq)])

    def cost(start: int, stop: int) -> float:
      """Within-group sum of squares for provisional items [start, stop)."""
      n_group = count_prefix[stop] - count_prefix[start]
      if n_group <= 0:
        return 0.0
      sum_group = total_prefix[stop] - total_prefix[start]
      sq_group = sq_prefix[stop] - sq_prefix[start]
      return float(sq_group - (sum_group * sum_group) / n_group)

    groups = optimal_partition(
      cost, len(count), self._min_items(len(count)), self.max_bins)
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
