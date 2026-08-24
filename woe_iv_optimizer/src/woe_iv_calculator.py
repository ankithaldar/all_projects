#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Weight of Evidence and Information Value calculator."""


# imports
import pandas as pd
import numpy as np
from typing import Tuple
#    script imports
# imports


# constants
# constants


# classes
class WoeIvCalculator:
  """Weight of Evidence and Information Value calculator."""

  @staticmethod
  def compute_woe_iv(binned: pd.Series, y: pd.Series) -> Tuple[pd.DataFrame, float]:
    df = pd.DataFrame({'bin': binned, 'y': y})
    agg = df.groupby('bin', dropna=False)['y'].agg(['count', 'sum']).reset_index()
    agg.columns = ['bin', 'total', 'bads']
    agg['goods'] = agg['total'] - agg['bads']

    total_bads = agg['bads'].sum()
    total_goods = agg['goods'].sum()

    if total_bads == 0 or total_goods == 0:
      agg['WoE'] = 0.0
      agg['IV'] = 0.0
    else:
      agg['dist_bads'] = agg['bads'] / total_bads
      agg['dist_goods'] = agg['goods'] / total_goods
      agg['WoE'] = np.log(agg['dist_goods'] / agg['dist_bads']).replace([np.inf, -np.inf], 0)
      agg['IV'] = (agg['dist_goods'] - agg['dist_bads']) * agg['WoE']

    iv_total = agg['IV'].sum()
    return agg, iv_total
# classes
