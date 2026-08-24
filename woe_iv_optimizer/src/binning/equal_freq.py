#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Equal Frequency Binning"""


# imports
import pandas as pd
from typing import Tuple
#    script imports
from .base import BinningStrategy
# imports


# constants
# constants


# classes
class EqualFreqBinning(BinningStrategy):
  """Equal Frequency Binning"""

  def bin(self, x: pd.Series, y: pd.Series) -> Tuple[pd.Series, list]:
    n_bins = min(self.max_bins, len(x.dropna().unique()))
    if n_bins < 2:
      return x.copy(), []
    try:
      bins, edges = pd.qcut(x, q=n_bins, retbins=True, duplicates='drop')
      return bins, edges.tolist()
    except ValueError:
      # Fallback to unique quantiles
      unique_vals = x.dropna().sort_values().unique()
      if len(unique_vals) <= n_bins:
        bins = pd.cut(x, bins=len(unique_vals), duplicates='drop')
        return bins, []
      raise
# classes
