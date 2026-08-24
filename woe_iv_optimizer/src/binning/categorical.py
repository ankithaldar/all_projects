#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Frequency-and-IV grouping for categorical features.

Scorecard work needs a binning for non-numeric features too, and the numeric
strategies cannot help there because they all cut on ordered values. This one
orders categories by frequency, folds the tail into a single Rare bucket, then
merges adjacent categories by IV loss. Numeric input is passed through to the
equal-frequency strategy, since grouping a numeric column by frequency would
destroy the ordering the other strategies rely on.
"""


# imports
import pandas as pd
import numpy as np
from typing import List, Tuple
#    script imports
from .base import (BinningStrategy, iv_from_counts)
from .equal_freq import EqualFreqBinning
# imports


# constants
DEFAULT_RARE_THRESHOLD = 0.01
# constants


# classes
class CategoricalBinning(BinningStrategy):
  """Frequency-and-IV grouping for categorical features."""

  def __init__(self, min_bin_size: float = 0.05, max_bins: int = 10,
               enforce_monotonicity: bool = False,
               rare_threshold: float = DEFAULT_RARE_THRESHOLD, **kwargs):
    super().__init__(min_bin_size, max_bins, enforce_monotonicity)
    self.rare_threshold = rare_threshold

  def bin(self, x: pd.Series, y: pd.Series) -> Tuple[pd.Series, list]:
    """Group categories of a categorical feature.

    Args:
      X: Feature, categorical or numeric.
      y: Binary target.

    Returns:
      tuple: Binned series holding group labels, and an empty edge list,
      because categories have no ordered cut points to report.
    """
    if pd.api.types.is_numeric_dtype(x):
      return EqualFreqBinning(
        self.min_bin_size, self.max_bins,
        self.enforce_monotonicity).bin(x, y)

    clean = x.dropna()
    if len(clean) == 0 or clean.nunique() <= 1:
      return x.copy(), []

    counts = clean.value_counts()
    total = counts.sum()

    # Anything rarer than the threshold collapses into one bucket, otherwise
    # long-tailed categories eat the whole bin budget.
    rare_mask = (counts / total) < self.rare_threshold
    grouped = x.astype(object).where(~x.isin(counts[rare_mask].index), 'Rare')

    labels = self._merge_by_iv(grouped, y)
    binned = grouped.map(labels).astype('object')
    binned = binned.where(~x.isna(), 'Missing')
    return binned, []

  def _merge_by_iv(self, grouped: pd.Series, y: pd.Series) -> dict:
    """Map each category to a group label by greedy IV-loss merging.

    Args:
      grouped: Feature with rare categories already folded into ``Rare``.
      y: Binary target.

    Returns:
      dict: Category value to group label.
    """
    counts = grouped.value_counts()
    # Order by frequency so that the most common categories keep their own
    # bins and the sparse ones are the first to be merged together.
    ordered = counts.index.tolist()
    n = len(ordered)

    stats = self._stats(grouped, y, ordered)
    if stats is None:
      return {value: value for value in ordered}

    goods, bads = stats
    total_goods, total_bads = goods.sum(), bads.sum()
    iv_parts = iv_from_counts(goods, bads, total_goods, total_bads)

    groups: List[int] = list(range(n))
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

    labels = {}
    for position, start in enumerate(groups):
      label = f'Group{position}'
      stop = groups[position + 1] if position + 1 < len(groups) else n
      for index in range(start, stop):
        labels[ordered[index]] = label
    return labels

  @staticmethod
  def _stats(grouped: pd.Series, y: pd.Series, ordered: list) -> tuple:
    """Goods and bads per category, aligned to ``ordered``.

    Args:
      grouped: Feature with rare categories folded in.
      y: Binary target.
      ordered: Category values in the order to be tabulated.

    Returns:
      tuple: ``(goods, bads)`` arrays, or ``(None, None)`` when the target is
      single-class.
    """
    frame = pd.DataFrame({'cat': grouped.to_numpy(), 'y': y.to_numpy()})
    agg = frame.groupby('cat', observed=True)['y'].agg(['count', 'sum'])
    if agg.empty:
      return None, None

    totals = agg['count'].reindex(ordered).fillna(0).to_numpy(dtype=float)
    bads = agg['sum'].reindex(ordered).fillna(0).to_numpy(dtype=float)
    goods = totals - bads
    if goods.sum() <= 0 or bads.sum() <= 0:
      return None, None
    return goods, bads
# classes
