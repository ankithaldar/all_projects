#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Winsorised equal-frequency binning.

Caps the feature at configurable outer quantiles before splitting it into
equal-frequency bins. Outliers are the main reason the plain quantile strategies
produce unusable bins on skewed features: on a variable with a median near 40
and a maximum of 440, the top quantile holds only a handful of rows, so
equal-width binning ends up with single-row bins. Capping first keeps the bins
populated.
"""


# imports
import pandas as pd
import numpy as np
from typing import Tuple
#    script imports
from .base import BinningStrategy
# imports


# constants
DEFAULT_LOWER_QUANTILE = 0.01
DEFAULT_UPPER_QUANTILE = 0.99
# constants


# classes
class WinsorizedBinning(BinningStrategy):
  """Winsorised equal-frequency binning."""

  def __init__(self, min_bin_size: float = 0.05, max_bins: int = 10,
               enforce_monotonicity: bool = False,
               lower_quantile: float = DEFAULT_LOWER_QUANTILE,
               upper_quantile: float = DEFAULT_UPPER_QUANTILE, **kwargs):
    super().__init__(min_bin_size, max_bins, enforce_monotonicity)
    self.lower_quantile = lower_quantile
    self.upper_quantile = upper_quantile

  def bin(self, x: pd.Series, y: pd.Series) -> Tuple[pd.Series, list]:
    """Cap the feature's tails, then split it into equal-frequency bins.

    Args:
      X: Numeric feature.
      y: Binary target.

    Returns:
      tuple: Binned series and the resulting bin edges. The returned edges are
      the winsorised cut points, so scoring with them reproduces the training
      bin assignment.
    """
    clean = x.dropna()
    if len(clean) == 0 or clean.nunique() <= 1:
      return x.copy(), []

    lower = float(clean.quantile(self.lower_quantile))
    upper = float(clean.quantile(self.upper_quantile))
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
      return x.copy(), []

    capped = clean.clip(lower=lower, upper=upper)
    n_bins = min(self.max_bins, capped.nunique())
    if n_bins < 2:
      return x.copy(), []

    binned, edges = pd.qcut(capped, q=n_bins, retbins=True, duplicates='drop')
    if len(edges) < 2:
      return x.copy(), []

    # Open the outer bounds so uncapped values land in the outermost bin
    # instead of becoming null, and drop missing values the same way the other
    # strategies do.
    edges = ([float(edges[0])] + [float(e) for e in edges[1:-1]]
             + [float(edges[-1])])
    binned = binned.reindex(x.index)
    return binned, edges
# classes
