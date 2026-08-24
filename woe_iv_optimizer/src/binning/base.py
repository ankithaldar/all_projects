#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Abstract Class for Binning Strategies"""


# imports
from abc import ABC, abstractmethod
import pandas as pd
from typing import Tuple, Optional
#    script imports
# imports


# constants
# constants


# classes
class BinningStrategy(ABC):
  """Abstract Class for Binning Strategies"""

  def __init__(self, min_bin_size: float = 0.05, max_bins: int = 10):
    self.min_bin_size = min_bin_size
    self.max_bins = max_bins

  @abstractmethod
  def bin(self, X: pd.Series, y: pd.Series) -> Tuple[pd.Series, list]:
    """Return binned series and bin edges (or categories)."""
    pass

# classes
