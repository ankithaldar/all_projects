#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Entry Point of the Module"""


# imports
import os
import logging
import sys
from typing import Dict, List, Tuple
import pandas as pd
#    script imports
from config_loader import ConfigLoader
from data_loader.base import DataLoader
from data_loader.bigquery_loader import BigQueryDataLoader
from data_loader.csv_loader import CSVDataLoader
from data_loader.parquet_loader import ParquetDataLoader
from binning_factory import BinningStrategyFactory
from woe_iv_calculator import WoeIvCalculator
from visualizer import Visualizer
from utils import handle_missing_values, save_binning_rules
from monotonicity_checker import MonotonicityChecker
# imports



# functions
def setup_logging(config: Dict[str, Any]):
  log_level = getattr(logging, config['logging']['level'].upper())
  log_format = config['logging']['format']
  log_file = config['output']['log_file']
  os.makedirs(os.path.dirname(log_file), exist_ok=True)

  logging.basicConfig(
    level=log_level,
    format=log_format,
    handlers=[
      logging.FileHandler(log_file),
      logging.StreamHandler(sys.stdout)
    ]
  )


def get_data_loader(config: Dict[str, Any]) -> DataLoader:
  """Factory function to instantiate the correct data loader."""
  source = config.get('data_source', 'bigquery').lower()

  loader_map = {
    'bigquery': BigQueryDataLoader,
    'csv': CSVDataLoader,
    'parquet': ParquetDataLoader,
  }

  if source not in loader_map:
    raise ValueError(f"Unsupported data source: {source}. Choose from {list(loader_map.keys())}")

  return loader_map[source](config)


def main():
  config_loader = ConfigLoader("config/config.yaml")
  config = config_loader.load()
  setup_logging(config)

  logger = logging.getLogger("Main")
  logger.info("Starting WoE/IV Binning Optimization")

  # Load data
  # data_loader = BigQueryDataLoader(config)

  data_loader = get_data_loader(config)
  df = data_loader.load_data()
  features = data_loader.get_features(df)
  target = config['target_column']
  y = df[target]

  # Setup
  viz = Visualizer(config['output']['viz_dir'], config['output']['viz_format'])
  constraints = config['constraints']
  missing_strategy = config['missing_handling']['strategy']
  strategies_to_run = config['binning_strategies']

  results = {}
  binning_rules = {}

  for feature in features:
    logger.info(f"Processing feature: {feature}")
    X = df[feature].copy()
    X = handle_missing_values(X, missing_strategy)

    feature_results = {}
    for strat_name in strategies_to_run:
      logger.debug(f"  Applying strategy: {strat_name}")
      try:
        strategy = BinningStrategyFactory.create(
          strat_name,
          min_bin_size=constraints['min_bin_size'],
          max_bins=constraints['max_bins'],
          enforce_monotonicity=constraints.get('enforce_monotonicity', False),
          chi_threshold=0.1  # for ChiMerge
        )
        binned, edges = strategy.bin(X, y)
        woe_df, iv = WoeIvCalculator.compute_woe_iv(binned, y)

        # Check monotonicity for numerical features
        is_numerical = pd.api.types.is_numeric_dtype(df[feature])
        is_monotonic = True
        if is_numerical and constraints.get('enforce_monotonicity', False):
          woe_vals = woe_df['WoE'].values
          is_monotonic = MonotonicityChecker.is_monotonic(woe_vals)
          if not is_monotonic:
            logger.warning(f"Non-monotonic WoE for {feature} with {strat_name}")

        feature_results[strat_name] = {
          'iv': iv,
          'woe_df': woe_df,
          'edges': edges,
          'monotonic': is_monotonic
        }

        # Visualize
        if iv >= constraints['min_iv_threshold']:
          viz.plot_woe_trend(feature, woe_df, strat_name)

        # Save best rule (by IV) for this feature
        if strat_name not in binning_rules.get(feature, {}) or iv > binning_rules[feature].get('iv', -1):
          binning_rules[feature] = {
            'strategy': strat_name,
            'iv': iv,
            'edges': edges,
            'woe_table': woe_df.to_dict()
          }

      except Exception as e:
        logger.error(f"Error processing {feature} with {strat_name}: {e}")
        continue

    results[feature] = feature_results

  # Aggregate IV results for comparison plot
  iv_comparison = {}
  for feat, res in results.items():
    iv_comparison[feat] = {s: r['iv'] for s, r in res.items()}
  viz.plot_iv_comparison(iv_comparison)

  # Save binning rules
  save_binning_rules(binning_rules, config['output']['rules_output'])
  logger.info("Binning optimization completed successfully.")
# functions


if __name__ == "__main__":
  main()
