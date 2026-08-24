#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Binning Strategy Optimized By Information Value"""


# imports
import pandas as pd
import numpy as np
from itertools import combinations
#    script imports
from .base import BinningStrategy
from src.monotonicity_checker import MonotonicityChecker
# imports


# constants
# constants


# classes
class IVOptimizedBinning(BinningStrategy):
  """Binning Strategy Optimized By Information Value"""

  def __init__(self, min_bin_size: float = 0.05, max_bins: int = 10, enforce_monotonicity: bool = True):
    super().__init__(min_bin_size, max_bins)
    self.enforce_monotonicity = enforce_monotonicity

  def bin(self, X: pd.Series, y: pd.Series) -> Tuple[pd.Series, list]:
    df = pd.DataFrame({'X': X, 'y': y}).dropna().sort_values('X')
    if len(df) == 0 or df['X'].nunique() <= 1:
      return X.copy(), []

    unique_vals = df['X'].unique()
    if len(unique_vals) <= self.max_bins:
      bins = pd.cut(X, bins=len(unique_vals), duplicates='drop')
      return bins, []

    best_iv = -np.inf
    best_bins = None

    # Try all combinations up to max_bins
    for k in range(2, min(self.max_bins + 1, len(unique_vals) + 1)):
      for split_points in combinations(range(1, len(unique_vals)), k - 1):
        edges = [-np.inf] + [unique_vals[i] for i in split_points] + [np.inf]
        binned = pd.cut(df['X'], bins=edges)
        woe_df = self._compute_woe(binned, df['y'])
        iv = woe_df['IV'].sum()

        if self.enforce_monotonicity:
          woe_vals = woe_df['WoE'].values
          if not MonotonicityChecker.is_monotonic(woe_vals):
            continue

        if iv > best_iv:
          best_iv = iv
          best_bins = edges

    if best_bins is None:
      # Fallback to equal-freq
      return EqualFreqBinning(self.min_bin_size, self.max_bins).bin(X, y)

    result = pd.cut(X, bins=best_bins)
    return result, best_bins

  def _compute_woe(self, bins: pd.Series, y: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({'bin': bins, 'y': y})
    agg = df.groupby('bin')['y'].agg(['count', 'sum']).reset_index()
    agg.columns = ['bin', 'total', 'bads']
    agg['goods'] = agg['total'] - agg['bads']
    total_bads = agg['bads'].sum()
    total_goods = agg['goods'].sum()
    agg['dist_bads'] = agg['bads'] / total_bads
    agg['dist_goods'] = agg['goods'] / total_goods
    agg['WoE'] = np.log(agg['dist_goods'] / agg['dist_bads']).replace([np.inf, -np.inf], 0)
    agg['IV'] = (agg['dist_goods'] - agg['dist_bads']) * agg['WoE']
    return agg
# classes
