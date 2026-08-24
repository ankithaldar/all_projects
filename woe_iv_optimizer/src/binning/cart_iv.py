#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""CART binning pruned by Information Value.

Grows a decision tree, then collapses its leaves by repeatedly merging the
adjacent pair that costs the least Information Value, which is what a scorecard
developer does when tidying up a tree-driven binning. The existing tree_based
strategy stops at the tree's own leaf budget and therefore optimises impurity
rather than IV; this one prunes explicitly on IV, so it targets the quantity the
rest of the pipeline actually reports.
"""


# imports
import pandas as pd
import numpy as np
from typing import List, Tuple
from sklearn.tree import DecisionTreeClassifier
#    script imports
from .base import BinningStrategy, iv_from_counts
# imports


# constants
# constants


# classes
class CartIVBinning(BinningStrategy):
  """CART binning pruned by Information Value."""

  def __init__(self, min_bin_size: float = 0.05, max_bins: int = 10,
               enforce_monotonicity: bool = False, **kwargs):
    super().__init__(min_bin_size, max_bins, enforce_monotonicity)

  def bin(self, x: pd.Series, y: pd.Series) -> Tuple[pd.Series, list]:
    """Bin the feature with a decision tree, then prune leaves by IV.

    Args:
      X: Numeric feature.
      y: Binary target.

    Returns:
      tuple: Binned series and the resulting bin edges.
    """
    frame = pd.DataFrame({'X': x, 'y': y}).dropna()
    if len(frame) == 0 or frame['X'].nunique() <= 1:
      return x.copy(), []

    tree = DecisionTreeClassifier(
      max_leaf_nodes=max(self.max_bins * 2, self.max_bins + 2),
      min_samples_leaf=max(1, int(len(frame) * self.min_bin_size)),
      random_state=42
    )
    tree.fit(frame[['X']], frame['y'])

    thresholds = sorted(set(
      tree.tree_.threshold[tree.tree_.threshold > -2].tolist()))
    if not thresholds:
      return x.copy(), []

    edges = [-np.inf] + thresholds + [np.inf]
    binned = pd.cut(x, bins=edges)
    if binned.isna().all():
      return x.copy(), []

    edges = self._prune(binned, y, edges)
    interior = [e for e in edges if np.isfinite(e)]
    if not interior:
      return x.copy(), []

    return pd.cut(x, bins=edges), edges

  def _prune(self, binned: pd.Series, y: pd.Series, edges: list) -> list:
    """Drop interior cuts, cheapest IV loss first, until max_bins remain.

    Args:
      binned: Binned feature at the tree's resolution.
      y: Binary target.
      edges: Starting edges with infinite outer bounds.

    Returns:
      list: Edges after pruning, or the input unchanged when it cannot be
      improved.
    """
    goods, bads = self._bin_stats(binned, y)
    if goods is None or len(goods) < 2:
      return edges

    total_goods, total_bads = goods.sum(), bads.sum()
    iv_parts = iv_from_counts(goods, bads, total_goods, total_bads)

    # Each entry is the starting bin index of one surviving group.
    groups: List[int] = list(range(len(goods)))
    while len(groups) > self.max_bins:
      best_loss, best_slot = None, None
      for slot in range(len(groups) - 1):
        left, right = groups[slot], groups[slot + 1]
        before = iv_parts[left] + iv_parts[right]
        after = iv_from_counts(
          np.array([goods[left] + goods[right]]),
          np.array([bads[left] + bads[right]]),
          total_goods, total_bads)[0]
        loss = after - before
        if best_loss is None or loss < best_loss:
          best_loss, best_slot = loss, slot
      if best_slot is None:
        break
      del groups[best_slot + 1]

    if len(groups) < 2:
      return edges

    # edges[start + 1] is the cut that opens the group at ``start``. The last
    # group has no interior cut, so clamp to the final real edge to keep the
    # infinite outer bounds from being duplicated.
    last_interior = len(edges) - 2
    interior = [edges[min(start + 1, last_interior)] for start in groups[1:]]
    interior = sorted(set(interior))
    return [-np.inf] + list(interior) + [np.inf]

  @staticmethod
  def _bin_stats(binned: pd.Series, y: pd.Series) -> tuple:
    """Per-bin goods and bads in ascending bin order.

    Args:
      binned: Binned feature.
      y: Binary target.

    Returns:
      tuple: ``(goods, bads)`` arrays, or ``(None, None)`` when the target is
      single-class and IV is undefined.
    """
    frame = pd.DataFrame({'bin': binned, 'y': y})
    agg = frame.groupby(
      'bin', observed=True, dropna=False)['y'].agg(['count', 'sum'])
    if agg.empty:
      return None, None
    total = agg['count'].to_numpy(dtype=float)
    bads = agg['sum'].to_numpy(dtype=float)
    goods = total - bads
    if goods.sum() <= 0 or bads.sum() <= 0:
      return None, None
    return goods, bads
# classes
