#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Visualize the information value of features"""


# imports
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, List
#    script imports
# imports


# constants
# constants


# classes
class Visualizer:
  """Visualize the information value of features"""

  def __init__(self, output_dir: str, fmt: str = "png"):
    self.output_dir = output_dir
    self.fmt = fmt
    os.makedirs(output_dir, exist_ok=True)

  def plot_woe_trend(self, feature: str, woe_df: pd.DataFrame, strategy: str):
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(woe_df)), woe_df['WoE'], marker='o')
    plt.title(f"WoE Trend - {feature} ({strategy})")
    plt.xlabel("Bin Index")
    plt.ylabel("Weight of Evidence")
    plt.grid(True)
    plt.savefig(os.path.join(self.output_dir, f"woe_{feature}_{strategy}.{self.fmt}"))
    plt.close()

  def plot_iv_comparison(self, iv_results: Dict[str, Dict[str, float]]):
    df = pd.DataFrame(iv_results).T
    df.plot(kind='bar', figsize=(12, 6))
    plt.title("IV Comparison Across Features and Strategies")
    plt.ylabel("Information Value")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(self.output_dir, f"iv_comparison.{self.fmt}"))
    plt.close()
# classes
