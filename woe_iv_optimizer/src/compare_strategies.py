#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Compare every registered binning strategy and write a ranked report.

Runs each strategy in the factory against every numeric feature, then writes
both a human-readable log and a machine-readable CSV so the winning strategy
per feature is easy to read off. Failures are recorded rather than raised, since
a comparison is only useful if it shows which strategies actually work.
"""


# imports
import argparse
import sys
import time
import logging
from typing import Any, Dict, List
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
#    script imports
from binning_factory import BinningStrategyFactory
from binning.base import strategies_for_feature
from woe_iv_calculator import WoeIvCalculator
from monotonicity_checker import MonotonicityChecker
from data_loader.sqlite_loader import SQLiteDataLoader
# imports


# constants
DEFAULT_CONFIG = 'config/config.yaml'


def output_paths(config_path: str) -> tuple:
  """Derive the log and CSV paths from the config file name.

  Deriving the paths from the config file name keeps a second comparison from
  overwriting the first, and the default config keeps the bare names so its
  outputs land at the documented paths.

  Args:
    config_path: Path to the YAML config driving the run.

  Returns:
    tuple: ``(log_path, csv_path)`` for this run.
  """
  stem = Path(config_path).stem
  if stem.startswith('config_'):
    stem = stem[len('config_'):]
  elif stem == Path(DEFAULT_CONFIG).stem:
    stem = ''
  suffix = f'_{stem}' if stem else ''
  return (f'logs/strategy_comparison{suffix}.log',
          f'output/strategy_comparison{suffix}.csv')
# constants


# functions
def load_frame(config_path: str) -> tuple:
  """Load the feature matrix described by a config file.

  Args:
    config_path: Path to the YAML config.

  Returns:
    tuple: ``(features, target)`` extracted from the loaded data.
  """
  with open(config_path, 'r', encoding='utf-8') as handle:
    config = yaml.safe_load(handle)

  loader = SQLiteDataLoader(config)
  frame = loader.load_data()
  target = config['target_column']
  score = config.get('score_column')
  drop = {target} | ({score} if score else set())
  features = [c for c in frame.columns if c not in drop]
  return frame[features], frame[target]


def evaluate(strategy_name: str, feature: str, x: pd.Series, y: pd.Series,
             constraints: Dict[str, Any]) -> Dict[str, Any]:
  """Run one strategy on one feature and score the result.

  Args:
    strategy_name: Registered strategy name.
    feature: Feature column name, used for reporting.
    X: Feature values.
    y: Binary target.
    constraints: Constraint block from the config.

  Returns:
    dict: Metrics for the run, including ``ok`` and ``error`` keys.
  """
  row: Dict[str, Any] = {
    'feature': feature,
    'strategy': strategy_name,
    'ok': False,
    'error': '',
    'iv': np.nan,
    'bins': 0,
    'min_bin_share': np.nan,
    'max_bin_share': np.nan,
    'monotonic': None,
    'seconds': np.nan,
  }

  start = time.time()
  try:
    strategy = BinningStrategyFactory.create(
      strategy_name,
      min_bin_size=constraints['min_bin_size'],
      max_bins=constraints['max_bins'],
      enforce_monotonicity=constraints.get('enforce_monotonicity', False),
      chi_threshold=0.1,
    )
    binned, _ = strategy.bin(x, y)
    woe_df, iv = WoeIvCalculator.compute_woe_iv(binned, y)

    if woe_df.empty:
      raise ValueError('no bins produced')

    shares = woe_df['total'] / woe_df['total'].sum()
    row.update(
      ok=True,
      iv=round(float(iv), 6),
      bins=int(woe_df['bin'].nunique()),
      min_bin_share=round(float(shares.min()), 6),
      max_bin_share=round(float(shares.max()), 6),
      monotonic=bool(MonotonicityChecker.is_monotonic(woe_df['WoE'].values)),
    )
  # A failure is a reportable result here: the comparison is only useful if it
  # shows which strategies worked, so the error is recorded and the run goes on.
  except Exception as exc:  # pylint: disable=broad-exception-caught
    row['error'] = f'{type(exc).__name__}: {exc}'
  finally:
    row['seconds'] = round(time.time() - start, 3)

  return row
# functions


# classes
class StrategyComparison:
  """Compare every registered binning strategy and write a ranked report."""

  def __init__(self, config_path: str = DEFAULT_CONFIG):
    self.config_path = config_path
    self.log_path, self.csv_path = output_paths(config_path)
    with open(config_path, 'r', encoding='utf-8') as handle:
      self.config = yaml.safe_load(handle)
    self.constraints = self.config['constraints']
    self.strategies: List[str] = self.config['binning_strategies']
    self.logger = logging.getLogger('StrategyComparison')

  def run(self) -> pd.DataFrame:
    """Execute the full comparison and write the report files.

    Returns:
      pd.DataFrame: One row per feature and strategy.
    """
    features, target = load_frame(self.config_path)
    self.logger.info('Comparing %d strategies on %d features',
                     len(self.strategies), features.shape[1])

    rows = []
    for feature in features.columns:
      x = features[feature]
      names = strategies_for_feature(x, self.strategies)
      if names != self.strategies:
        self.logger.info('%-22s is not numeric, using %s', feature, names)
      for name in names:
        row = evaluate(name, feature, x, target, self.constraints)
        rows.append(row)
        if row['ok']:
          self.logger.info('%-12s %-22s IV=%-9s bins=%-3s %.2fs',
                           row['strategy'], feature, row['iv'],
                           row['bins'], row['seconds'])
        else:
          self.logger.warning('%-12s %-22s FAILED: %s',
                              row['strategy'], feature, row['error'])

    report = pd.DataFrame(rows)
    report.attrs['rows'] = len(target)
    self._write(report)
    return report

  def _write(self, report: pd.DataFrame) -> None:
    """Write the CSV and the ranked text log.

    Args:
      report: Per feature and strategy metrics.
    """
    Path(self.csv_path).parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(self.csv_path, index=False)

    winners = report[report['ok']].loc[
      report[report['ok']].groupby('feature')['iv'].idxmax()]

    lines: List[str] = []
    lines.append('Binning strategy comparison')
    lines.append(f'dataset rows: {report.attrs.get('rows', '?')}')
    lines.append(f'max bins    : {self.constraints['max_bins']}')
    lines.append(f'strategies : {len(self.strategies)}')
    lines.append(f'features   : {report['feature'].nunique()}')
    lines.append('')

    lines.append('Best strategy per feature (highest IV)')
    lines.append('-' * 78)
    # pylint: disable=consider-using-f-string
    # Nested same-quote braces are not valid below Python 3.12.
    header = '{:<24}{:<16}{:>9}{:>6}{:>7}'.format(
      'feature', 'strategy', 'IV', 'bins', 'mono')
    lines.append(header)
    for _, row in winners.iterrows():
      feature, strategy = row['feature'], row['strategy']
      iv_value, bin_count = row['iv'], row['bins']
      monotonic = str(row['monotonic'])
      lines.append(f'{feature:<24}{strategy:<16}'
                   f'{iv_value:>9.4f}{bin_count:>6}{monotonic:>7}')
    lines.append('')

    lines.append('IV by feature and strategy')
    lines.append('-' * 78)
    pivot = report.pivot_table(index='feature', columns='strategy',
                               values='iv', aggfunc='first')
    pivot = pivot[[s for s in self.strategies if s in pivot.columns]]
    lines.append(pivot.round(4).to_string())

    lines.append('')
    lines.append('Aggregate across features')
    lines.append('-' * 78)
    summary = report[report['ok']].groupby('strategy').agg(
      mean_iv=('iv', 'mean'),
      median_iv=('iv', 'median'),
      total_seconds=('seconds', 'sum'),
      failures=('ok', lambda s: int((~s).sum())),
    ).sort_values('mean_iv', ascending=False)
    lines.append(summary.round(4).to_string())

    failed = report[~report['ok']]
    if not failed.empty:
      lines.append('')
      lines.append('Failures')
      lines.append('-' * 78)
      for _, row in failed.iterrows():
        lines.append(f'{row['strategy']:<16}{row['feature']:<24}{row['error']}')

    Path(self.log_path).parent.mkdir(parents=True, exist_ok=True)
    Path(self.log_path).write_text('\n'.join(lines) + '\n', encoding='utf-8')
# classes


# main
def main() -> int:
  """Entry point for the comparison run.

  Returns:
    int: Process exit code.
  """
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--config', default=DEFAULT_CONFIG)
  args = parser.parse_args()

  log_path, csv_path = output_paths(args.config)
  Path(log_path).parent.mkdir(parents=True, exist_ok=True)
  logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],
  )

  report = StrategyComparison(args.config).run()
  print()
  print(report[report['ok']].groupby('strategy')['iv'].mean()
        .sort_values(ascending=False).round(4).to_string())
  print()
  print(f'wrote {csv_path} and {log_path}')
  return 0
# main


if __name__ == '__main__':
  raise SystemExit(main())
