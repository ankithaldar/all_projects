#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for the pure sequence helpers.

The numeric half of the sequence stack is well defended by construction: every
dividing helper coerces its input and guards its denominator. The three places
that needed explicit tests are the ungarded empty case, the two date helpers
that must accept both ``str`` and ``datetime`` inputs, and the expression that
has to normalise array length before two windows are compared. All three are
covered here.

These are pure functions, so the tests are exhaustive rather than representative
-- there is no fixture to arrange and no cluster to start, which makes this the
cheapest place in the suite to be thorough.
"""

from __future__ import annotations

from datetime import date, datetime

import pytest

from forecasting_ml_framework.preprocessing import sequence_ops as ops


class TestEmptyInputs:
  """An empty behavioural sequence is realistic, not an error.

  A customer with no events in the window is precisely the low-activity signal
  the model exists to detect, so every aggregation must return a neutral value
  rather than raise.
  """

  @pytest.mark.parametrize(
    ('helper', 'args'),
    [
      (ops.seq_average, (3,)),
      (ops.seq_sum, (3,)),
      (ops.seq_min, (3,)),
      (ops.seq_max, (3,)),
      (ops.seq_last, ()),
      (ops.seq_delta, (2,)),
      (ops.get_slope_func, (1, 3)),
      (ops.change_ratio, (2,)),
      (ops.coefficient_of_variation, (3,)),
      (ops.seq_date_delta, (3,)),
    ],
  )
  def test_empty_array_returns_neutral(self, helper, args) -> None:
    """No helper may raise on an empty array.

    Args:
      helper: The helper under test.
      args: Its trailing arguments.
    """
    result = helper([], *args)
    assert result == pytest.approx(0.0)

  def test_min_max_last_are_the_three_that_used_to_raise(self) -> None:
    """The three specific regressions must be named in the suite."""
    assert ops.seq_min([], 3) == 0.0
    assert ops.seq_max([], 3) == 0.0
    assert ops.seq_last([]) == 0.0


class TestFixedLength:
  """the helper padded but never truncated."""

  def test_pads_short_array_on_the_left(self) -> None:
    """Short arrays are left-padded, which preserves recency ordering."""
    assert ops.get_fixed_length([3.0, 4.0], 6) == [0.0, 0.0, 0.0, 0.0, 3.0, 4.0]

  def test_truncates_long_array_to_the_most_recent(self) -> None:
    """Long arrays keep only the most recent values, which makes the result rectangular."""
    result = ops.get_fixed_length([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], 6)
    assert len(result) == 6
    assert result == [3.0, 4.0, 5.0, 6.0, 7.0, 8.0]

  @pytest.mark.parametrize('length', [1, 3, 6, 12])
  def test_output_is_always_rectangular(self, length: int) -> None:
    """Whatever the input length, the output is exactly ``length``.

    Args:
      length: The configured target length.
    """
    for size in range(0, 20):
      result = ops.get_fixed_length(list(range(size)), length)
      assert len(result) == length, f'ragged output for input length {size}'


class TestDateHelpers:
  """the type test was inverted, so parsing ran unconditionally."""

  def test_accepts_iso_strings(self) -> None:
    """ISO strings parse correctly."""
    result = ops.seq_date_delta(['2026-01-01', '2026-01-04', '2026-01-07'], 3)
    assert result == 3

  def test_accepts_datetime_objects(self) -> None:
    """Already-parsed values must not be re-parsed.

    A applied ``strptime`` unconditionally, so an array that already
    held ``date`` or ``datetime`` objects raised ``TypeError``.
    """
    values = [date(2026, 1, 1), date(2026, 1, 4)]
    assert ops.seq_date_delta(values, 3) == 3
    assert ops.seq_date_delta([datetime(2026, 1, 1), datetime(2026, 1, 4)], 3) == 3

  def test_recency_uses_the_sentinel_for_an_empty_array(self) -> None:
    """An empty array returns the sentinel rather than raising.

    The sentinel exists for exactly this case, and a indexed
    ``arr[-1]`` unconditionally, so the one case it was designed to represent
    raised ``IndexError``.
    """
    assert ops.seq_recency([], '2026-09-29', 9_999_999) == 9_999_999

  def test_recency_ignores_a_falsy_sentinel(self) -> None:
    """A falsy sentinel is replaced by the default rather than returned as-is.

    A guard was ``if not fill_na_val: return fill_na_val``, which
    returns the falsy value it was given and therefore cannot do what its name
    says.
    """
    assert ops.seq_recency([], '2026-09-29', 0) == ops.DEFAULT_RECENCY_SENTINEL
    assert ops.seq_recency([], '2026-09-29', None) == ops.DEFAULT_RECENCY_SENTINEL

  def test_recency_measures_age_in_days(self) -> None:
    """A known gap produces the known age."""
    assert ops.seq_recency(['2026-09-20'], '2026-09-29', 9_999_999) == 9

  def test_recency_without_a_reference_returns_the_sentinel(self) -> None:
    """No run date means no comparison is possible."""
    assert ops.seq_recency(['2026-09-20'], None) == ops.DEFAULT_RECENCY_SENTINEL


class TestNumericalGuards:
  """The numeric half was already correct; the tests pin that down."""

  def test_coercion_survives_non_float_elements(self) -> None:
    """A string or decimal element must not break the arithmetic."""
    assert ops.seq_average(['1', '2', '3'], 3) == pytest.approx(2.0)
    assert ops.seq_sum([1, 2, 3], 3) == pytest.approx(6.0)

  @pytest.mark.parametrize('helper', [ops.change_ratio, ops.coefficient_of_variation, ops.seq_normalize])
  def test_zero_baseline_does_not_divide_by_zero(self, helper) -> None:
    """An all-zero array must not produce a division by zero.

    Args:
      helper: The helper under test.
    """
    result = helper([0.0, 0.0, 0.0], 3) if helper is not ops.seq_normalize else helper([0.0, 0.0], 4)
    assert result is not None


class TestCosineSimilarity:
  """mismatched vector lengths used to raise inside the UDF."""

  def test_identical_vectors_give_one(self) -> None:
    """A vector against itself has similarity one."""
    assert ops.cos_sim([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)

  def test_orthogonal_vectors_give_zero(self) -> None:
    """Orthogonal vectors have similarity zero."""
    assert ops.cos_sim([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

  def test_mismatched_lengths_give_zero_rather_than_raising(self) -> None:
    """A length mismatch is a well-defined outcome, not a job failure.

    This is the case that failed the whole job for every customer with fewer
    events than twice the configured window -- the low-activity customers a
    retention model most needs to score.
    """
    assert ops.cos_sim([1.0, 2.0, 3.0], [1.0, 2.0]) == 0.0

  def test_zero_vector_gives_zero(self) -> None:
    """A zero vector has undefined similarity and returns a neutral value."""
    assert ops.cos_sim([0.0, 0.0], [1.0, 2.0]) == 0.0


class TestSlope:
  """The slope's order parameter is its most useful feature."""

  def test_linear_trend(self) -> None:
    """A rising sequence has a positive slope."""
    assert ops.get_slope_func([1.0, 2.0, 3.0, 4.0], 1, 4) > 0

  def test_flat_series(self) -> None:
    """A constant series has no slope, and a zero sum is guarded."""
    assert ops.get_slope_func([2.0, 2.0, 2.0], 1, 3) == pytest.approx(0.0)

  def test_zero_sum_is_guarded(self) -> None:
    """An all-zero series is guarded rather than dividing by zero."""
    assert ops.get_slope_func([0.0, 0.0, 0.0], 1, 3) == 0.0
