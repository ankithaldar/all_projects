#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Binning Strategy Optimized By Information Value"""


# imports
import pandas as pd
import numpy as np
from itertools import combinations
from typing import Tuple
#    script imports
from .base import BinningStrategy
from .equal_freq import EqualFreqBinning
from monotonicity_checker import MonotonicityChecker
# imports


# constants
# constants


# classes
class IVOptimizedBinning(BinningStrategy):
  """Binning Strategy Optimized By Information Value"""

  def __init__(self, min_bin_size: float = 0.05, max_bins: int = 10,
               enforce_monotonicity: bool = True, **kwargs):
    super().__init__(min_bin_size, max_bins, enforce_monotonicity)

  def bin(self, x: pd.Series, y: pd.Series) -> Tuple[pd.Series, list]:
    df = pd.DataFrame({'X': x, 'y': y}).dropna().sort_values('X')
    if len(df) == 0 or df['X'].nunique() <= 1:
      return x.copy(), []

    unique_vals = df['X'].unique()
    if len(unique_vals) <= self.max_bins:
      bins = pd.cut(x, bins=len(unique_vals), duplicates='drop')
      return bins, []

    candidates = self._candidate_edges(df)
    best_iv = -np.inf
    best_bins = None

    for k in range(2, min(self.max_bins, len(candidates) - 1) + 1):
      for split_points in combinations(candidates, k - 1):
        edges = [-np.inf] + list(split_points) + [np.inf]
        binned = pd.cut(df['X'], bins=edges)
        if binned.isna().sum() > len(df) * (1 - self.min_bin_size):
          continue
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
      return EqualFreqBinning(
        self.min_bin_size, self.max_bins, self.enforce_monotonicity).bin(x, y)

    result = pd.cut(x, bins=best_bins)
    return result, best_bins

  def _candidate_edges(self, df: pd.DataFrame) -> list:
    """Return the interior split points the search is allowed to consider.

    Exhaustive search over every combination of the feature's distinct values
    is combinatorially infeasible: with 80 distinct values and max_bins=10 it
    would evaluate C(79, 9), roughly 2e11 candidate binnings. Instead the
    search is restricted to a grid of quantile-derived candidate edges. The
    grid size grows linearly with max_bins, because the number of subsets grows
    as the power set: 12 candidates yield about 4e3 evaluations, while 28
    candidates yield about 1.2e7 and take days at pandas grouping speeds.

    Returns:
      list: Ascending interior edges strictly between the observed minimum
      and maximum of the feature.
    """
    n_candidates = max(2, self.max_bins + 2)
    quantiles = np.linspace(0, 1, n_candidates + 1)[1:-1]
    raw = df['X'].quantile(quantiles).to_numpy()
    edges = np.unique(raw)
    lo, hi = df['X'].min(), df['X'].max()
    edges = edges[(edges > lo) & (edges < hi)]
    return list(edges)

  def _compute_woe(self, bins: pd.Series, y: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({'bin': bins, 'y': y})
    agg = df.groupby('bin')['y'].agg(['count', 'sum']).reset_index()
    agg.columns = ['bin', 'total', 'bads']
    agg['goods'] = agg['total'] - agg['bads']
    total_bads = agg['bads'].sum()
    total_goods = agg['goods'].sum()
    agg['dist_bads'] = agg['bads'] / total_bads
    agg['dist_goods'] = agg['goods'] / total_goods
    ratio = agg['dist_goods'] / agg['dist_bads']
    agg['WoE'] = np.log(ratio).replace([np.inf, -np.inf], 0)
    agg['IV'] = (agg['dist_goods'] - agg['dist_bads']) * agg['WoE']
    return agg
# classes
