#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Binning Strategy Based on Decision Tree Classifier"""


# imports
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from typing import Tuple
#    script imports
from .base import BinningStrategy
# imports


# constants
# constants


# classes
class TreeBasedBinning(BinningStrategy):
  """Binning Strategy Based on Decision Tree Classifier"""

  def bin(self, x: pd.Series, y: pd.Series) -> Tuple[pd.Series, list]:
    df = pd.DataFrame({'X': x, 'y': y}).dropna()
    if len(df) == 0 or df['X'].nunique() <= 1:
      return x.copy(), []

    tree = DecisionTreeClassifier(
      max_leaf_nodes=self.max_bins,
      min_samples_leaf=int(len(df) * self.min_bin_size),
      random_state=42
    )
    tree.fit(df[['X']], df['y'])

    thresholds = sorted(tree.tree_.threshold[tree.tree_.threshold > -2])
    if not thresholds:
      return x.copy(), []

    bins = pd.cut(x, bins=[-np.inf] + thresholds + [np.inf])
    return bins, thresholds
# classes
