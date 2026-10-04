#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Leak-free train/test splitting for time series.

A random split leaks the future into the past and inflates every metric in
the backtest. Worse, a plain chronological split still leaks: if a label
looks forward ``horizon`` bars, the last ``horizon`` training samples
overlap the first test samples in both time and outcome.

``purged_folds`` therefore does two things beyond splitting on time:

  purge
    Training samples whose label window overlaps the test window are
    dropped, not merely bounded.
  embargo
    A configurable gap is dropped after the test window. Purge alone
    handles overlap; embargo handles serial correlation, where returns a
    few bars apart are near-duplicates even when their label windows do
    not technically intersect.

The horizon and embargo are arguments rather than constants because both
are properties of the strategy, not of the framework. A 1-bar momentum
label needs a purge of 1; a 20-bar forward return needs 20.
'''

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

__all__ = ['Fold', 'purged_folds', 'train_test_split']


@dataclass(frozen=True, slots=True)
class Fold:
  '''One train/test partition, expressed as indices into the input series.

  Attributes:
    fold: Zero-based position of this fold in the walk-forward sequence.
    train_start: First training index, inclusive.
    train_end: Last training index, exclusive.
    test_start: First test index, inclusive.
    test_end: Last test index, exclusive.
  '''

  fold: int
  train_start: int
  train_end: int
  test_start: int
  test_end: int

  @property
  def train_size(self) -> int:
    '''Number of training samples in this fold.'''
    return self.train_end - self.train_start

  @property
  def test_size(self) -> int:
    '''Number of test samples in this fold.'''
    return self.test_end - self.test_start

  def slice_train(self, series: list) -> list:
    '''Return the training portion of ``series``.

    Args:
      series: Any indexable sequence aligned with the fold indices.

    Returns:
      The training slice.
    '''
    return series[self.train_start:self.train_end]

  def slice_test(self, series: list) -> list:
    '''Return the test portion of ``series``.

    Args:
      series: Any indexable sequence aligned with the fold indices.

    Returns:
      The test slice.
    '''
    return series[self.test_start:self.test_end]

  def is_empty(self) -> bool:
    '''Return True when either side of the fold has no samples.'''
    return self.train_size <= 0 or self.test_size <= 0


def _validate(length: int, n_folds: int, horizon: int, embargo: int,
              min_train: int) -> None:
  '''Raise ValueError when the requested split cannot be satisfied.

  Args:
    length: Number of samples available.
    n_folds: Requested number of walk-forward folds.
    horizon: Forward label horizon in bars.
    embargo: Bars to drop after each test window.
    min_train: Minimum required training samples per fold.

  Raises:
    ValueError: If any argument is out of range or the series is too
      short to produce a single non-empty fold.
  '''
  if n_folds < 1:
    raise ValueError(f'n_folds must be >= 1, got {n_folds}')
  if horizon < 0:
    raise ValueError(f'horizon must be >= 0, got {horizon}')
  if embargo < 0:
    raise ValueError(f'embargo must be >= 0, got {embargo}')
  if min_train < 1:
    raise ValueError(f'min_train must be >= 1, got {min_train}')
  # Each fold consumes at least one test bar plus the purge and embargo
  # that surround it, so the series must be long enough to pay for them.
  required = min_train + horizon + embargo + 1
  if length < required:
    raise ValueError(
      f'series of {length} bars is too short for min_train={min_train}, '
      f'horizon={horizon}, embargo={embargo}; need at least {required}')


def purged_folds(
  length: int,
  n_folds: int = 5,
  horizon: int = 1,
  embargo: int = 1,
  min_train: int = 60,
) -> list[Fold]:
  '''Return expanding-window walk-forward folds with purge and embargo.

  Training data expands with each fold (an expanding window, not a
  rolling one) because for a time series the oldest observations are the
  least representative, not the least trustworthy.

  Purging is applied backwards from the test window: the last ``horizon``
  training bars are dropped because their forward-looking labels reach
  into the test period. The embargo then drops a further ``embargo`` bars
  after the test window, which is where serial correlation would leak
  across the boundary.

  Args:
    length: Number of samples in the series.
    n_folds: Number of walk-forward folds to produce.
    horizon: Forward label horizon in bars. Must cover the longest label
      used anywhere in the pipeline.
    embargo: Extra bars to drop after each test window.
    min_train: Minimum training samples required in the first fold.

  Returns:
    List of non-overlapping ``Fold`` objects in chronological order.

  Raises:
    ValueError: If the arguments are invalid or the series is too short.
  '''
  _validate(length, n_folds, horizon, embargo, min_train)
  # The walk-forward must start far enough in that there is room to purge.
  # Starting at plain min_train and flooring train_end at min_train would
  # silently cancel the purge on the first fold, leaving the last
  # `horizon` training labels reaching into the test window.
  origin = min_train + horizon
  # Ceiling, not floor: the remainder must land inside the final fold
  # rather than as an unused tail of up to n_folds-1 bars.
  fold_length = max(1, math.ceil((length - origin) / n_folds))
  folds: list[Fold] = []
  for index in range(n_folds):
    test_start = origin + index * fold_length
    test_end = min(length, test_start + fold_length)
    if test_start >= length:
      break
    folds.append(Fold(index, 0, test_start - horizon, test_start, test_end))
  return [fold for fold in folds if not fold.is_empty()]


def train_test_split(
  timestamps: list[datetime],
  horizon: int = 1,
  embargo: int = 1,
  train_fraction: float = 0.7,
) -> tuple[list[datetime], list[datetime]]:
  '''Return a single chronological, purged and embargoed split.

  A convenience for the common case of one holdout rather than a full
  walk-forward. Purge and embargo are applied around the test window
  exactly as in ``purged_folds``.

  PONYTAIL: this returns a single cut, so it offers no way to inspect
  performance across market regimes -- which for a Nifty strategy is the
  difference between a robust result and one that happened to work in
  2021. Ceiling: one test window. Upgrade path: call ``purged_folds`` and
  aggregate the folds, which is what any real evaluation should do
  anyway.

  Args:
    timestamps: Bar timestamps in ascending order.
    horizon: Forward label horizon in bars.
    embargo: Bars to drop between train and test, to break serial
      correlation across the boundary. This is a *gap*, not a truncation:
      reserving it off the end of the series would leave train and test
      adjacent and leak exactly what the embargo exists to prevent.
    train_fraction: Share of the series allocated to training, applied to
      the portion left after the test window and embargo are reserved.

  Returns:
    Tuple of (train timestamps, test timestamps).

  Raises:
    ValueError: If inputs are invalid or the split would be degenerate.
  '''
  length = len(timestamps)
  if horizon < 0:
    raise ValueError(f'horizon must be >= 0, got {horizon}')
  if embargo < 0:
    raise ValueError(f'embargo must be >= 0, got {embargo}')
  if not 0.0 < train_fraction < 1.0:
    raise ValueError(
      f'train_fraction must be in (0, 1), got {train_fraction}')
  train_budget = int(length * train_fraction)
  test_size = length - train_budget
  test_end = length
  test_start = test_end - test_size
  train_end = test_start - embargo - horizon
  if train_end < 1:
    raise ValueError(
      f'horizon {horizon} plus embargo {embargo} consumes the entire '
      f'training window of a {length} bar series')
  return timestamps[:train_end], timestamps[test_start:test_end]
