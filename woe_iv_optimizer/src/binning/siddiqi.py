#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Scorecard-standard binning by greedy IV-loss merging.

Implements the rule described by Siddiqi, *Credit Risk Scorecards* (2nd ed.):
start from fine quantile bins, then repeatedly merge the adjacent pair whose
merge costs the least Information Value until the bin budget is met. The merge
order is what distinguishes this from a plain quantile split, and the search is
greedy rather than exhaustive, so it stays fast on large features.
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
class SiddiqiScorecardBinning(BinningStrategy):
  """Scorecard-standard binning by greedy IV-loss merging."""

  def __init__(self, min_bin_size: float = 0.05, max_bins: int = 10,
               enforce_monotonicity: bool = True, **kwargs):
    super().__init__(min_bin_size, max_bins, enforce_monotonicity)

  def bin(self, x: pd.Series, y: pd.Series) -> Tuple[pd.Series, list]:
    """Bin the feature by greedy minimum-IV-loss merging.

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

    # Start finer than the target so that the merges have something to work
    # with; a merge that starts from exactly max_bins bins has no freedom.
    n_start = max(self.max_bins * 3, self.max_bins + 2)
    positions, prebin_edges = prebin(x, n_start)
    if positions is None:
      return x.copy(), []

    goods, bads = counts_by_position(positions, y)
    populated = np.flatnonzero(goods + bads > 0)
    if len(populated) < 2:
      return x.copy(), []


    total_goods, total_bads = goods.sum(), bads.sum()

    # Each group is a half-open provisional range, kept ascending and covering
    # the populated positions exactly once.
    groups: List[Tuple[int, int]] = [
      (int(pos), int(pos) + 1) for pos in populated]
    goods = goods[populated]
    bads = bads[populated]

    iv_parts = iv_from_counts(goods, bads, total_goods, total_bads)

    while len(groups) > self.max_bins:
      best = None
      for i in range(len(groups) - 1):
        merged_goods = goods[i] + goods[i + 1]
        merged_bads = bads[i] + bads[i + 1]
        merged_iv = iv_from_counts(
          np.array([merged_goods]), np.array([merged_bads]),
          total_goods, total_bads)[0]
        loss = (iv_parts[i] + iv_parts[i + 1]) - merged_iv
        if best is None or loss < best[0]:
          best = (loss, i)

      if best is None:
        break

      _, i = best
      merged_goods = goods[i] + goods[i + 1]
      merged_bads = bads[i] + bads[i + 1]
      merged_iv = iv_from_counts(
        np.array([merged_goods]), np.array([merged_bads]),
        total_goods, total_bads)[0]
      groups[i] = (groups[i][0], groups[i + 1][1])
      del groups[i + 1]
      goods = np.append(np.delete(goods, i + 1), merged_goods)
      bads = np.append(np.delete(bads, i + 1), merged_bads)
      iv_parts = np.append(np.delete(iv_parts, i + 1), merged_iv)

    interior = edges_for_boundaries(prebin_edges, [g[0] for g in groups[1:]])
    if not interior:
      return single_bin(x)

    edges = full_edges(interior)
    return pd.cut(x, bins=edges), edges
# classes
