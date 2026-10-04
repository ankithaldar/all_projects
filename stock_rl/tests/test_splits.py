#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Tests for leak-free train/test splitting.'''

from datetime import datetime, timedelta

import pytest

from stock_rl.splits import Fold, purged_folds, train_test_split


def stamps(count: int) -> list[datetime]:
  '''Return ``count`` consecutive hourly timestamps from a fixed origin.

  Args:
    count: Number of timestamps.

  Returns:
    List of datetimes one hour apart.
  '''
  start = datetime(2020, 1, 1)
  return [start + timedelta(hours=index) for index in range(count)]


class TestPurgedFolds:
  '''Structure and leak-freedom of walk-forward folds.'''

  def test_requested_fold_count(self):
    folds = purged_folds(1000, n_folds=5)
    assert len(folds) == 5

  def test_test_windows_do_not_overlap(self):
    folds = purged_folds(1000, n_folds=6)
    for earlier, later in zip(folds, folds[1:]):
      assert earlier.test_end <= later.test_start

  def test_test_windows_advance_in_time(self):
    folds = purged_folds(1000, n_folds=5)
    starts = [fold.test_start for fold in folds]
    assert starts == sorted(starts)

  def test_train_window_expands(self):
    # Expanding, not rolling: each fold keeps the whole earlier history.
    folds = purged_folds(1000, n_folds=5)
    ends = [fold.train_end for fold in folds]
    assert ends == sorted(ends)
    assert all(fold.train_start == 0 for fold in folds)

  def test_purge_and_embargo_both_shorten_training(self):
    # The gap between train and test must be the horizon PLUS the
    # embargo. The embargo used to be validated and then never applied,
    # so every value of it produced identical folds.
    horizon = 10
    embargo = 7
    folds = purged_folds(1000, n_folds=5, horizon=horizon, embargo=embargo)
    for fold in folds:
      assert fold.train_end == fold.test_start - horizon - embargo

  def test_embargo_actually_changes_the_folds(self):
    # The vacuous version of this test passed on the broken code.
    few = purged_folds(2000, n_folds=6, horizon=5, embargo=1)
    many = purged_folds(2000, n_folds=6, horizon=5, embargo=40)
    assert [fold.train_end for fold in many] \
      != [fold.train_end for fold in few]

  def test_no_train_index_reaches_into_test_window(self):
    # This is the property that matters. Asserting the gap explicitly is
    # what makes a regression here visible rather than silent.
    horizon = 7
    for fold in purged_folds(2000, n_folds=8, horizon=horizon):
      train = set(range(fold.train_start, fold.train_end))
      test = set(range(fold.test_start, fold.test_end))
      assert train.isdisjoint(test)
      assert max(train) < fold.test_start

  def test_larger_horizon_never_leaks_more(self):
    # Every fold from a longer horizon must be contained within the
    # corresponding fold from a shorter one.
    short = purged_folds(2000, n_folds=6, horizon=1, min_train=200)
    long = purged_folds(2000, n_folds=6, horizon=20, min_train=200)
    for near, far in zip(short, long):
      assert far.train_end <= near.train_end

  def test_embargo_reserves_space_after_test(self):
    folds = purged_folds(1000, n_folds=5, horizon=1, embargo=20)
    # The next fold's training data must stop before this fold's test
    # ends, leaving the embargo gap unused.
    for earlier, later in zip(folds, folds[1:]):
      assert later.train_end <= earlier.test_end + 20

  def test_folds_are_non_empty(self):
    for fold in purged_folds(2000, n_folds=10):
      assert not fold.is_empty()
      assert fold.train_size > 0
      assert fold.test_size > 0

  def test_last_fold_reaches_end_of_series(self):
    folds = purged_folds(1000, n_folds=5)
    assert folds[-1].test_end == 1000

  def test_series_is_used_efficiently(self):
    # Guard against a fold_length calculation that silently discards
    # most of the series.
    folds = purged_folds(1000, n_folds=5)
    covered = sum(fold.test_size for fold in folds)
    assert covered > 800

  def test_asks_for_more_folds_than_fit(self):
    # Ceiling division can overshoot the remainder, pushing the final
    # requested fold past the end of the series. That fold must be
    # dropped rather than emitted with inverted bounds.
    folds = purged_folds(13, n_folds=7, horizon=1, min_train=1)
    assert len(folds) < 7
    for fold in folds:
      assert fold.test_start < 13
      assert fold.train_end < fold.test_start


class TestFoldSlicing:
  '''Index arithmetic on the Fold record.'''

  def test_slices_match_bounds(self):
    fold = Fold(0, 0, 10, 20, 30)
    series = list(range(50))
    assert fold.slice_train(series) == list(range(0, 10))
    assert fold.slice_test(series) == list(range(20, 30))

  def test_sizes_match_slices(self):
    fold = Fold(1, 0, 10, 20, 30)
    series = list(range(50))
    assert fold.train_size == len(fold.slice_train(series))
    assert fold.test_size == len(fold.slice_test(series))

  def test_empty_detection(self):
    # Empty means zero samples on one side, not adjacency. Adjacent
    # train and test is legitimate, just tight.
    assert not Fold(0, 0, 10, 10, 20).is_empty()
    assert Fold(0, 0, 10, 10, 10).is_empty()
    assert Fold(0, 0, 0, 10, 20).is_empty()


class TestSplitValidation:
  '''Bad configuration must fail loudly, not silently mis-slice.'''

  @pytest.mark.parametrize('kwargs', [
    {'n_folds': 0},
    {'horizon': -1},
    {'embargo': -1},
    {'min_train': 0},
  ])
  def test_rejects_nonsense_arguments(self, kwargs):
    with pytest.raises(ValueError):
      purged_folds(1000, **kwargs)

  def test_rejects_series_too_short(self):
    with pytest.raises(ValueError):
      purged_folds(20, horizon=50, min_train=100)

  def test_error_message_names_the_shortfall(self):
    with pytest.raises(ValueError, match='too short'):
      purged_folds(20, horizon=50, min_train=100)


class TestTrainTestSplit:
  '''The single-holdout convenience wrapper.'''

  def test_split_is_chronological(self):
    train, test = train_test_split(stamps(1000))
    assert max(train) < min(test)

  def test_split_sizes_are_roughly_respected(self):
    train, test = train_test_split(stamps(1000), train_fraction=0.7)
    assert len(train) == pytest.approx(700, abs=5)
    assert len(test) == pytest.approx(300, abs=5)

  def test_purge_applied_before_test(self):
    horizon = 15
    train, test = train_test_split(stamps(1000), horizon=horizon)
    # The gap between the last train bar and the first test bar must be
    # at least the horizon.
    gap = test[0] - train[-1]
    assert gap >= timedelta(hours=horizon)

  def test_embargo_creates_a_gap_between_train_and_test(self):
    train, test = train_test_split(stamps(1000), embargo=50, horizon=1)
    # The embargo is a hole between the two sides. If it were instead
    # reserved off the end of the series, train and test would sit
    # adjacent and serial correlation would leak straight across.
    gap = test[0] - train[-1]
    assert gap >= timedelta(hours=51)

  def test_embargo_shrinks_training_not_testing(self):
    tight_train, tight_test = train_test_split(stamps(1000), embargo=1)
    loose_train, loose_test = train_test_split(stamps(1000), embargo=50)
    assert len(loose_test) == len(tight_test)
    assert len(loose_train) < len(tight_train)

  def test_test_window_is_anchored_at_the_end(self):
    series = stamps(1000)
    _, test = train_test_split(series, train_fraction=0.7)
    assert test[-1] == series[-1]

  @pytest.mark.parametrize('kwargs', [
    {'horizon': -1},
    {'embargo': -1},
    {'train_fraction': 0.0},
    {'train_fraction': 1.0},
    {'train_fraction': 1.5},
  ])
  def test_rejects_nonsense_arguments(self, kwargs):
    with pytest.raises(ValueError):
      train_test_split(stamps(1000), **kwargs)

  def test_rejects_degenerate_series(self):
    with pytest.raises(ValueError, match='consumes'):
      train_test_split(stamps(2), embargo=10)

  def test_horizon_consuming_training_window_raises(self):
    with pytest.raises(ValueError, match='consumes'):
      train_test_split(stamps(20), horizon=500, train_fraction=0.5)
