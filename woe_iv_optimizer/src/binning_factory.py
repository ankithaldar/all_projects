#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Factory class for creating binning strategies."""


# imports
from typing import Dict, Type
#    script imports
from binning.base import BinningStrategy
from binning.equal_width import EqualWidthBinning
from binning.equal_freq import EqualFreqBinning
from binning.chimerge import ChiMergeBinning
from binning.tree_based import TreeBasedBinning
from binning.iv_optimized import IVOptimizedBinning
from binning.monotonic_pava import MonotonicPavaBinning
from binning.siddiqi import SiddiqiScorecardBinning
from binning.optimal_dp import OptimalDPBinning
from binning.mdl import MDLBinning
from binning.bic import BICBinning
from binning.greedy_iv import GreedyIVBinning
from binning.natural_breaks import NaturalBreaksBinning
from binning.kmeans import KMeansBinning
from binning.winsorized import WinsorizedBinning
from binning.cart_iv import CartIVBinning
from binning.categorical import CategoricalBinning
# imports


# constants
# constants


# classes
class BinningStrategyFactory:
  """Factory class for creating binning strategies."""

  _strategies: Dict[str, Type[BinningStrategy]] = {
    # Ordered by how the search space is explored.
    'equal_width': EqualWidthBinning,
    'equal_freq': EqualFreqBinning,
    'winsorized': WinsorizedBinning,
    'natural_breaks': NaturalBreaksBinning,
    'kmeans': KMeansBinning,
    'tree_based': TreeBasedBinning,
    'cart_iv': CartIVBinning,
    'chimerge': ChiMergeBinning,
    'siddiqi': SiddiqiScorecardBinning,
    'greedy_iv': GreedyIVBinning,
    'iv_optimized': IVOptimizedBinning,
    'optimal_dp': OptimalDPBinning,
    'monotonic_pava': MonotonicPavaBinning,
    'mdl': MDLBinning,
    'bic': BICBinning,
    'categorical': CategoricalBinning,
  }

  @classmethod
  def create(cls, name: str, **kwargs) -> BinningStrategy:
    if name not in cls._strategies:
      raise ValueError(f"Unknown binning strategy: {name}")
    return cls._strategies[name](**kwargs)

  @classmethod
  def available(cls) -> list:
    """Return every registered strategy name.

    Returns:
      list: Registered names in registration order.
    """
    return list(cls._strategies.keys())
# classes
