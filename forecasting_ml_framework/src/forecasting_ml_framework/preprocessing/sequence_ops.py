#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Pure sequence helpers.

Seventeen stateless functions, all shaped ``(array) -> scalar`` or
``(array) -> array``, all designed to be called from a Spark UDF. They are the
reason the sequence routine wrappers are so short, and they are the only part of
the routine library that is straightforward to unit-test in isolation.

Four properties are worth stating up front, because each of them is a decision
rather than an implementation detail:

* **Rectangular output.** ``get_fixed_length`` pads *and* slices, so the result
  is always exactly the configured length. It left-pads, which preserves recency
  ordering with the most recent event last, and truncates from the front, which
  is what makes the array rectangular -- a dense tensor cannot be built from a
  ragged batch, and the customers with the most history are the ones that would
  break it.
* **Empty input is a normal input.** Every aggregation returns a neutral value
  rather than raising. A customer with no events in the window is not an error
  case; it is precisely the low-activity signal a retention model exists to
  detect, and ``SequenceFillNa`` manufactures such arrays deliberately.
* **Sentinels over nulls.** ``seq_recency`` represents "no event" with a value,
  not a null, because a null would propagate into imputation and destroy the
  very signal the sentinel encodes.
* **Scalar summaries, not arrays.** Every helper returns a scalar or a
  same-shaped array, matching the return type its routine declares, so a summary
  is always something an estimator can consume.
"""

from __future__ import annotations

import sys
from datetime import date, datetime
from typing import Any

import numpy as np

#: Guard applied to every denominator. Every numeric helper uses it, and every
#: numeric helper also coerces its input first, so a ``decimal``, ``long`` or
#: ``string`` Spark array element cannot break the arithmetic.
EPSILON = sys.float_info.epsilon

#: Date format accepted by the ETL for sequence date elements.
DATE_FORMAT = '%Y-%m-%d'

#: Sentinel returned by :func:`seq_recency` when no date is supplied.
DEFAULT_RECENCY_SENTINEL = 9_999_999


# --------------------------------------------------------------------------- #
# Coercion
# --------------------------------------------------------------------------- #
def _to_floats(arr: Any) -> list[float]:
  """Coerce an array of any numeric representation to Python floats.

  Args:
    arr: The input array.

  Returns:
    A list of floats.
  """
  return [float(item) for item in (arr or [])]


def _tail(arr: list[float], window: int | None) -> list[float]:
  """Return the most recent ``window`` elements of an array.

  Args:
    arr: The input array.
    window: The window size. ``None`` or a non-positive value returns everything.

  Returns:
    The tail of the array.
  """
  if not window or window <= 0:
    return arr
  return arr[-window:]


def _to_date(value: Any) -> date:
  """Coerce a value to a ``date``.

  a guarded its parsing with ``type(i) != 'str'``. ``type``
  returns a *class object*, which never equals a string, so the condition was
  always true and ``strptime`` was applied to every element unconditionally --
  including elements that were already ``date`` or ``datetime`` objects, which
  made it raise ``TypeError``. The intended check is ``isinstance``, and the
  three cases are enumerated explicitly rather than relying on a fallback
  that happens to work.

  Args:
    value: A string, ``date``, or ``datetime``.

  Returns:
    The coerced date.

  Raises:
    ValueError: If the value is neither a recognised type nor a parseable string.
  """
  if isinstance(value, datetime):
    return value.date()
  if isinstance(value, date):
    return value
  if isinstance(value, str):
    return datetime.strptime(value, DATE_FORMAT).date()
  raise ValueError(
    f'Cannot interpret {value!r} as a date. The sequence date helpers accept an ISO date string, '
    f'a datetime.date, or a datetime.datetime; got {type(value).__name__}.'
  )


# --------------------------------------------------------------------------- #
# Array-shape transforms
# --------------------------------------------------------------------------- #
def get_fixed_length(arr: Any, fixed_length: int = 6) -> list[float]:
  """Pad or truncate an array to exactly ``fixed_length`` elements.

  Short arrays are **left**-padded with zeros, which preserves recency ordering:
  the most recent event stays last. Long arrays are truncated to the most recent
  ``fixed_length`` elements, which is what makes the result rectangular.

  Args:
    arr: The input array.
    fixed_length: The target length.

  Returns:
    A list of exactly ``fixed_length`` floats.
  """
  tail = _to_floats(arr)[-fixed_length:] if fixed_length > 0 else []
  padding = [0.0] * max(0, fixed_length - len(tail))
  return padding + tail


def slice_array(arr: Any, start: int = 0, end: int = 3) -> list[float]:
  """Return a Python slice of an array.

  Args:
    arr: The input array.
    start: The inclusive start index.
    end: The exclusive end index.

  Returns:
    The sliced array.
  """
  return _to_floats(arr)[start:end]


def normalise_window(arr: Any, start: int, end: int, fixed_length: int) -> list[float]:
  """Return a Python slice of an array padded and truncated to a fixed length.

  :func:`slice_array` on its own returns whatever the slice happens to hold,
  which is shorter than the window whenever the history is. Two vectors of
  different lengths cannot be compared, so a helper that compares them -- cosine
  similarity, say -- silently returns its degenerate answer for every short
  history. Routing both sides of such a comparison through this function, and
  through :func:`get_fixed_length` for the primary window, is what makes the
  lengths agree by construction rather than by luck of the data.

  Args:
    arr: The input array.
    start: The inclusive start index of the slice.
    end: The exclusive end index of the slice.
    fixed_length: The length the result is padded or truncated to.

  Returns:
    The sliced array, always exactly ``fixed_length`` elements long.
  """
  return get_fixed_length(_to_floats(arr)[start:end], fixed_length=fixed_length)


def domain_normalization(arr: Any, normalize_by_val: Any, round_off: int = 4) -> list[float]:
  """Divide each element by a reference value.

  Args:
    arr: The input array.
    normalize_by_val: The reference value.
    round_off: The number of decimal places.

  Returns:
    The elementwise ratio, rounded.
  """
  reference = float(normalize_by_val) if normalize_by_val else EPSILON
  reference = reference if reference != 0 else EPSILON
  return [round(value / reference, round_off) for value in _to_floats(arr)]


def seq_normalize(arr: Any, round_off: int = 4) -> list[float]:
  """Express each element as a share of the array total.

  Args:
    arr: The input array.
    round_off: The number of decimal places.

  Returns:
    The share-of-total array, rounded.
  """
  values = _to_floats(arr)
  total = sum(values) if values else 0.0
  total = total if total != 0 else EPSILON
  return [round(value / total, round_off) for value in values]


# --------------------------------------------------------------------------- #
# Shape descriptors
# --------------------------------------------------------------------------- #
def get_slope_func(arr: Any, order: int = 1, last_n_val: int = 2) -> float:
  """Return the least-squares slope of the most recent values.

  Args:
    arr: The input array.
    order: The polynomial degree, so ``1`` is a linear trend and ``2`` captures
      curvature. A degree below 1 has no trend coefficient to report and yields
      ``0.0``.
    last_n_val: The window size.

  Returns:
    The slope, or ``0.0`` when the window is empty, sums to zero, or the degree
    admits no trend coefficient.
  """
  values = _tail(_to_floats(arr), last_n_val)
  if len(values) < 2:
    return 0.0
  denominator = sum(values)
  if denominator == 0:
    return 0.0
  # `order: 0` is a legal configuration value -- the key exists precisely so an
  # `order: 1` and an `order: 2` slope of one column can coexist -- and
  # `np.polyfit(..., 0)` returns a single coefficient, so reading `[-2]` runs off
  # the end. The check belongs here rather than in the `except` tuple because a
  # degree check is a statement about the request, not about a failure to fit.
  if int(order) < 1:
    return 0.0
  normalised = [value / denominator for value in values]
  x = list(range(len(normalised)))
  try:
    coefficients = np.polyfit(x, normalised, order)
  except (ValueError, np.linalg.LinAlgError, TypeError):
    return 0.0
  # `[-2]` is the linear term, the coefficient on x. `polyfit` returns exactly
  # `order + 1` coefficients, so the length check is what makes that read safe.
  if len(coefficients) < 2:
    return 0.0
  return float(coefficients[-2])


def cos_sim(x: Any, y: Any) -> float:
  """Return the cosine similarity of two vectors.

  Args:
    x: The first vector.
    y: The second vector.

  Returns:
    The cosine similarity, or ``0.0`` when either vector is all zeros or the two
    vectors have different lengths. The length guard is what turns what would be
    a hard ``ValueError`` inside a UDF — raised for every customer with fewer
    events than the configured window — into a well-defined value.
  """
  left = _to_floats(x)
  right = _to_floats(y)
  if len(left) != len(right):
    return 0.0
  if not left or sum(left) == 0 or sum(right) == 0:
    return 0.0
  return float(np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right)))


def change_ratio(arr: Any, last_n_val: int = 2) -> float:
  """Return the most recent value relative to the mean of the preceding window.

  Args:
    arr: The input array.
    last_n_val: The window size.

  Returns:
    The change ratio, or ``0.0`` when the baseline is zero.
  """
  values = _to_floats(arr)
  if len(values) < 2:
    return 0.0
  baseline = values[-1 - last_n_val : -1]
  if not baseline:
    return 0.0
  average = sum(baseline) / len(baseline)
  average = average if average != 0 else EPSILON
  return float(values[-1] / average)


def coefficient_of_variation(arr: Any, last_n_val: int = 3) -> float:
  """Return the coefficient of variation, sigma over mu, over a window.

  Args:
    arr: The input array.
    last_n_val: The window size.

  Returns:
    The coefficient of variation, or ``0.0`` when the mean is zero.
  """
  values = _tail(_to_floats(arr), last_n_val)
  if len(values) < 2:
    return 0.0
  mean = sum(values) / len(values)
  mean = mean if mean != 0 else EPSILON
  variance = sum((value - sum(values) / len(values)) ** 2 for value in values) / len(values)
  return float((variance**0.5) / mean)


# --------------------------------------------------------------------------- #
# Aggregations
# --------------------------------------------------------------------------- #
def seq_average(arr: Any, last_n_val: int = 3) -> float:
  """Return the mean of the most recent values.

  Args:
    arr: The input array.
    last_n_val: The window size.

  Returns:
    The mean, or ``0.0`` for an empty window.
  """
  values = _tail(_to_floats(arr), last_n_val)
  if not values:
    return 0.0
  return sum(values) / len(values)


def seq_sum(arr: Any, last_n_val: int = 3) -> float:
  """Return the sum of the most recent values.

  Args:
    arr: The input array.
    last_n_val: The window size.

  Returns:
    The sum, or ``0.0`` for an empty window.
  """
  return float(sum(_tail(_to_floats(arr), last_n_val)))


def seq_min(arr: Any, last_n_val: int = 3) -> float:
  """Return the minimum of the most recent values.

  Args:
    arr: The input array.
    last_n_val: The window size.

  Returns:
    The minimum, or ``0.0`` for an empty window. a
      raised ``ValueError: min() arg is an empty sequence``.
  """
  values = _tail(_to_floats(arr), last_n_val)
  return float(min(values)) if values else 0.0


def seq_max(arr: Any, last_n_val: int = 3) -> float:
  """Return the maximum of the most recent values.

  Args:
    arr: The input array.
    last_n_val: The window size.

  Returns:
    The maximum, or ``0.0`` for an empty window.
  """
  values = _tail(_to_floats(arr), last_n_val)
  return float(max(values)) if values else 0.0


def seq_last(arr: Any) -> float:
  """Return the final element of an array.

  Args:
    arr: The input array.

  Returns:
    The final element, or ``0.0`` for an empty array. a
      raised ``IndexError: list index out of range``.
  """
  values = _to_floats(arr)
  return float(values[-1]) if values else 0.0


def seq_delta(arr: Any, last_n_val: int = 2) -> float:
  """Return the total change across the most recent window.

  returning a *list* of consecutive differences, a
  ragged array of length ``k-1`` that no estimator consumes. It is a scalar
  that matches the ``FloatType`` its routine declares.

  Args:
    arr: The input array.
    last_n_val: The window size.

  Returns:
    The difference between the last and the first element of the window, or
    ``0.0`` for a window with fewer than two elements.
  """
  values = _tail(_to_floats(arr), last_n_val)
  if len(values) < 2:
    return 0.0
  return float(values[-1] - values[0])


# --------------------------------------------------------------------------- #
# Date helpers
# --------------------------------------------------------------------------- #
def seq_date_delta(arr: Any, last_n_val: int = 3) -> int:
  """Return the mean gap in days between consecutive dates.

  returning a *list* of day-gaps, which the routine then
  wrote over its source column — destroying the sequence of event dates and
  producing a feature no estimator accepts. It is a scalar summary, so the
  routine can register it like its eleven siblings.

  Args:
    arr: An array of dates or ISO date strings.
    last_n_val: The window size.

  Returns:
    The mean gap in days, or ``0`` for a window with fewer than two dates.
  """
  dates = sorted(_to_date(item) for item in _tail(list(arr or []), last_n_val))
  if len(dates) < 2:
    return 0
  gaps = [(later - earlier).days for earlier, later in zip(dates, dates[1:], strict=False)]
  # Rounded, not floored. A mean gap of 1.5 days is 2, not 1: `//` truncates
  # toward zero, which biases the feature downward on every window whose mean is
  # fractional -- a systematic error that grows as the window lengthens, and one
  # that is invisible because the values look like plausible day counts.
  return int(round(sum(gaps) / len(gaps))) if gaps else 0


def seq_recency(
  arr: Any,
  run_dt_val: Any = None,
  fill_na_val: int = DEFAULT_RECENCY_SENTINEL,
) -> int:
  """Return the days between the most recent event and the run date.

  For behavioural sequences the *absence* of a recent event is itself a signal,
  so an empty sequence must be represented by a value rather than a null — a
  null would propagate into imputation and destroy that signal. The most recent
  event is therefore located behind a length check, so the one case the sentinel
  exists to represent is the one case that returns it.

  Args:
    arr: An array of dates or ISO date strings.
    run_dt_val: The run date, used as the reference point.
    fill_na_val: The sentinel returned when no event is available.

  Returns:
    The age in days of the most recent event, or the sentinel.
  """
  values = list(arr or [])
  reference = _to_date(run_dt_val) if run_dt_val else None
  # A falsy sentinel is treated as unset and replaced by the default. Honouring a
  # falsy value literally would be the opposite of what the name suggests: a zero
  # sentinel reads as "the event happened today", which is an assertion about the
  # customer rather than a statement about missing data.
  sentinel = (
    int(fill_na_val)
    if fill_na_val
    else DEFAULT_RECENCY_SENTINEL
  )
  if not values or reference is None:
    return sentinel
  try:
    latest = max(_to_date(item) for item in values)
  except (TypeError, ValueError):
    return sentinel
  return int((reference - latest).days)
