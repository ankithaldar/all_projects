#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Handle missing values and save binning rules."""


# imports
import pandas as pd
import json
from typing import Dict, Any
#    script imports
# imports


# functions
def handle_missing_values(series: pd.Series, strategy: str = "separate_bin") -> pd.Series:
  if strategy == "separate_bin":
    return series.fillna("Missing")
  elif strategy == "impute_median":
    if pd.api.types.is_numeric_dtype(series):
      return series.fillna(series.median())
    else:
      return series.fillna(series.mode()[0] if not series.mode().empty else "Missing")
  elif strategy == "drop":
    return series.dropna()
  else:
    raise ValueError(f"Unknown missing handling strategy: {strategy}")

def save_binning_rules(rules: Dict[str, Any], path: str):
  with open(path, 'w') as f:
    json.dump(rules, f, indent=2, default=str)
# functions
