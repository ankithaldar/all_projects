#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Equal Width Binning Strategy"""


# imports
import pandas as pd
from typing import Tuple
#    script imports
from .base import BinningStrategy
# imports


# constants
# constants


# classes
class EqualWidthBinning(BinningStrategy):
  """Equal Width Binning Strategy"""

  def bin(self, x: pd.Series, y: pd.Series) -> Tuple[pd.Series, list]:
    n_bins = min(self.max_bins, len(x.dropna().unique()))
    if n_bins < 2:
      return x.copy(), []
    bins, edges = pd.cut(x, bins=n_bins, retbins=True, duplicates='drop')
    return bins, edges.tolist()
# classes
