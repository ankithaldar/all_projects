#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""ChiMerge Binning Strategy"""


# imports
import pandas as pd
import numpy as np
from scipy.stats import chi2, chi2_contingency
from typing import Tuple
#    script imports
from .base import BinningStrategy
# imports


# constants
# constants


# classes
class ChiMergeBinning(BinningStrategy):
  """ChiMerge Binning Strategy"""

  def __init__(self, min_bin_size: float = 0.05, max_bins: int = 10,
               chi_threshold: float = 0.1, **kwargs):
    super().__init__(
      min_bin_size, max_bins,
      kwargs.get('enforce_monotonicity', False))
    self.chi_threshold = chi_threshold
    self._crit = chi2.ppf(1 - chi_threshold, df=1)

  def bin(self, x: pd.Series, y: pd.Series) -> Tuple[pd.Series, list]:
    if x.nunique() <= self.max_bins:
      return x.copy(), []
    # A missing category is added at the end, so reserve a bin for it here to
    # keep the total within max_bins rather than one over.
    has_missing = bool(x.isna().any())
    budget = self.max_bins - 1 if has_missing else self.max_bins
    if budget < 1:
      return x.copy(), []

    df = pd.DataFrame({'X': x, 'y': y}).dropna()
    df = df.sort_values('X')

    # Provisional fine-grained start. ChiMerge collapses these down toward the
    # budget, so the start is finer than the target but capped by
    # 1 / min_bin_size to keep provisional bins usefully populated.
    max_by_floor = max(2, int(1 / self.min_bin_size)) if (
      self.min_bin_size) else budget
    n_init = min(budget * 4, max_by_floor, df['X'].nunique())
    if n_init <= 1:
      return x.copy(), []
    df['bin'] = pd.qcut(df['X'], q=n_init, duplicates='drop').astype(str)

    n_bins = df['bin'].nunique()
    while n_bins > budget:
      idx = self._best_merge(df)
      if idx is None:
        # Nothing is statistically distinguishable any more, but the bin cap is
        # a hard constraint, so keep merging the most populated pair until it
        # is met. Stopping here would leave the result above the budget.
        if n_bins <= 1:
          break
        idx = self._largest_merge(df)
      bins = sorted(df['bin'].unique())
      merged = f'{bins[idx]}|{bins[idx + 1]}'
      df.loc[df['bin'].isin(bins[idx:idx + 2]), 'bin'] = merged
      n_bins = df['bin'].nunique()

    result = pd.Series(index=x.index, dtype='object')
    result.loc[df.index] = df['bin']
    result.loc[x.isna()] = 'Missing'
    return result, []

  @staticmethod
  def _largest_merge(df: pd.DataFrame) -> int:
    """Index of the adjacent pair holding the most rows.

    Used only to bring the bin count down to the configured cap once no pair is
    statistically significant any more.

    Args:
      df: Working frame with a ``bin`` column.

    Returns:
      int: Index into the ascending bin list.
    """
    bins = sorted(df['bin'].unique())
    counts = df['bin'].value_counts()
    best_index, best_total = 0, -1.0
    for i in range(len(bins) - 1):
      total = counts.get(bins[i], 0) + counts.get(bins[i + 1], 0)
      if total > best_total:
        best_index, best_total = i, total
    return best_index

  def _best_merge(self, df: pd.DataFrame) -> int | None:
    """Index of the adjacent bin pair to merge, or None to stop.

    Picks the adjacent pair with the lowest chi-square statistic, but only when
    that statistic is significant at ``chi_threshold``. Returns None once no
    candidate is significant, which is the standard ChiMerge stopping rule.
    """
    bins = sorted(df['bin'].unique())
    best_chi, best_idx = np.inf, None
    for i in range(len(bins) - 1):
      pair = df[df['bin'].isin(bins[i:i + 2])]
      contingency = pd.crosstab(pair['bin'], pair['y'])
      if contingency.shape[0] < 2 or contingency.shape[1] < 2:
        continue
      chi, _, _, _ = chi2_contingency(contingency)
      if chi < best_chi:
        best_chi, best_idx = chi, i
    if best_idx is None or best_chi > self._crit:
      return None
    return best_idx
# classes
