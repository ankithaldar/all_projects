#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""One-dimensional k-means binning.

Lloyd's algorithm on a single dimension, which places cuts midway between
cluster centres. Unlike the quantile strategies it is not tied to the
population, so a feature with a heavy tail collapses those outliers into one
cluster and leaves the remaining bins wide. Initialising from quantiles rather
than randomly keeps the result deterministic.
"""


# imports
import pandas as pd
import numpy as np
from typing import Tuple
#    script imports
from .base import BinningStrategy
# imports


# constants
MAX_ITERATIONS = 100
# constants


# classes
class KMeansBinning(BinningStrategy):
  """One-dimensional k-means binning."""

  def __init__(self, min_bin_size: float = 0.05, max_bins: int = 10,
               enforce_monotonicity: bool = False,
               max_iter: int = MAX_ITERATIONS, **kwargs):
    super().__init__(min_bin_size, max_bins, enforce_monotonicity)
    self.max_iter = max_iter

  def bin(self, x: pd.Series, y: pd.Series) -> Tuple[pd.Series, list]:
    """Bin the feature by one-dimensional k-means.

    Args:
      X: Numeric feature.
      y: Binary target, ignored by this strategy.

    Returns:
      tuple: Binned series and the resulting bin edges.
    """
    clean = x.dropna()
    values = clean.to_numpy(dtype=float)
    if len(values) == 0 or clean.nunique() <= 1:
      return x.copy(), []

    k = min(self.max_bins, clean.nunique())
    if k < 2:
      return x.copy(), []

    # Deterministic seeding: interior quantiles spread the centres across the
    # observed range, which is what random seeding converges to anyway.
    probs = (np.arange(k) + 0.5) / k
    centres = np.quantile(values, probs)

    labels = np.zeros(len(values), dtype=int)
    for _ in range(self.max_iter):
      distances = np.abs(values[:, None] - centres[None, :])
      new_labels = np.argmin(distances, axis=1)
      if np.array_equal(new_labels, labels):
        break
      labels = new_labels
      for cluster in range(k):
        members = values[labels == cluster]
        if len(members) > 0:
          centres[cluster] = members.mean()

    # Recompute the final assignment for the converged centres.
    labels = np.argmin(np.abs(values[:, None] - centres[None, :]), axis=1)

    # Drop empty clusters before deriving cuts, otherwise two neighbouring
    # centres can coincide and produce a zero-width bin.
    order = np.argsort(centres)
    centres = centres[order]
    occupied = [c for c in order if np.any(labels == c)]
    if len(occupied) < 2:
      return x.copy(), []

    interior = (centres[occupied[:-1]] + centres[occupied[1:]]) / 2.0
    interior = np.unique(interior)
    if len(interior) == 0:
      return x.copy(), []

    edges = [-np.inf] + interior.tolist() + [np.inf]
    return pd.cut(x, bins=edges), edges
# classes
