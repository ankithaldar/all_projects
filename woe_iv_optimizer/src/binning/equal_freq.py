#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Equal Frequency Binning"""


# imports
import pandas as pd
#    script imports
from .base import BinningStrategy
# imports


# constants
# constants


# classes
class EqualFreqBinning(BinningStrategy):
  """Equal Frequency Binning"""

  def bin(self, X: pd.Series, y: pd.Series) -> Tuple[pd.Series, list]:
    n_bins = min(self.max_bins, len(X.dropna().unique()))
    if n_bins < 2:
      return X.copy(), []
    try:
      bins, edges = pd.qcut(X, q=n_bins, retbins=True, duplicates='drop')
      return bins, edges.tolist()
    except ValueError:
      # Fallback to unique quantiles
      unique_vals = X.dropna().sort_values().unique()
      if len(unique_vals) <= n_bins:
        bins = pd.cut(X, bins=len(unique_vals), duplicates='drop')
        return bins, []
      raise
# classes
