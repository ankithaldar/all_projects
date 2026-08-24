#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Test Binning Strategies"""


# imports
import unittest
import pandas as pd
import numpy as np
#    script imports
from src.binning.equal_freq import EqualFreqBinning
from src.binning.iv_optimized import IVOptimizedBinning
# imports


# constants
# constants


# classes
class TestBinningStrategies(unittest.TestCase):
  """Test Binning Strategies"""

  def setUp(self):
    np.random.seed(42)
    self.X = pd.Series(np.random.randn(1000))
    self.y = pd.Series(np.random.binomial(1, 0.3, 1000))

  def test_equal_freq_binning(self):
    binning = EqualFreqBinning(max_bins=5)
    binned, edges = binning.bin(self.X, self.y)
    self.assertEqual(binned.nunique(), 5)

  def test_iv_optimized_binning(self):
    binning = IVOptimizedBinning(max_bins=4, enforce_monotonicity=True)
    binned, edges = binning.bin(self.X, self.y)
    self.assertLessEqual(binned.nunique(), 4)
# classes
