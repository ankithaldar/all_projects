#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""ChiMerge Binning Strategy"""


# imports
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
#    script imports
from .base import BinningStrategy
# imports


# constants
# constants


# classes
class ChiMergeBinning(BinningStrategy):
  """ChiMerge Binning Strategy"""

  def __init__(self, min_bin_size: float = 0.05, max_bins: int = 10, chi_threshold: float = 0.1):
    super().__init__(min_bin_size, max_bins)
    self.chi_threshold = chi_threshold

  def bin(self, X: pd.Series, y: pd.Series) -> Tuple[pd.Series, list]:
    if X.nunique() <= self.max_bins:
      return X.copy(), []

    # Initial binning: each unique value is a bin
    df = pd.DataFrame({'X': X, 'y': y}).dropna()
    df = df.sort_values('X')
    df['bin'] = df['X'].astype(str)

    while len(df['bin'].unique()) > self.max_bins:
      chi_vals = []
      bins = sorted(df['bin'].unique())
      for i in range(len(bins) - 1):
        merged = df[df['bin'].isin(bins[i:i+2])]
        if len(merged) < len(df) * self.min_bin_size:
          chi_vals.append(np.inf)
          continue
        contingency = pd.crosstab(merged['bin'], merged['y'])
        if contingency.shape[0] < 2:
          chi_vals.append(np.inf)
          continue
        chi, _, _, _ = chi2_contingency(contingency)
        chi_vals.append(chi)

      if not chi_vals or min(chi_vals) > chi2.ppf(1 - self.chi_threshold, df=1):
        break

      idx = np.argmin(chi_vals)
      df.loc[df['bin'].isin(bins[idx:idx+2]), 'bin'] = f"{bins[idx]}_{bins[idx+1]}"

    # Map back to original index
    result = pd.Series(index=X.index, dtype='object')
    result.loc[df.index] = df['bin']
    result.loc[X.isna()] = 'Missing'
    return result, []
# classes
