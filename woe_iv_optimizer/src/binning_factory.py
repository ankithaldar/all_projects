#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Factory class for creating binning strategies."""


# imports
from typing import Dict, Type
#    script imports
from .binning.base import BinningStrategy
from .binning.equal_width import EqualWidthBinning
from .binning.equal_freq import EqualFreqBinning
from .binning.chimerge import ChiMergeBinning
from .binning.tree_based import TreeBasedBinning
from .binning.iv_optimized import IVOptimizedBinning
# imports


# constants
# constants


# classes
class BinningStrategyFactory:
  """Factory class for creating binning strategies."""

  _strategies: Dict[str, Type[BinningStrategy]] = {
    "equal_width": EqualWidthBinning,
    "equal_freq": EqualFreqBinning,
    "chimerge": ChiMergeBinning,
    "tree_based": TreeBasedBinning,
    "iv_optimized": IVOptimizedBinning,
  }

  @classmethod
  def create(cls, name: str, **kwargs) -> BinningStrategy:
    if name not in cls._strategies:
      raise ValueError(f"Unknown binning strategy: {name}")
    return cls._strategies[name](**kwargs)
# classes
