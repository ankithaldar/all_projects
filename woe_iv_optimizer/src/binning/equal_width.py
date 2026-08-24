#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Equal Width Binning Strategy"""


# imports
import pandas as pd
import numpy as np
#    script imports
from .base import BinningStrategy
# imports


# constants
# constants


# classes
class EqualWidthBinning(BinningStrategy):
  """Equal Width Binning Strategy"""

  def bin(self, X: pd.Series, y: pd.Series) -> Tuple[pd.Series, list]:
    n_bins = min(self.max_bins, len(X.dropna().unique()))
    if n_bins < 2:
      return X.copy(), []
    bins, edges = pd.cut(X, bins=n_bins, retbins=True, duplicates='drop')
    return bins, edges.tolist()
# classes
