#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Data-quality assertions.

the framework had comprehensive *instrumentation* — per-column
summaries, class distributions, per-routine row counts — and **zero
enforcement**. Nothing anywhere asserted a null rate, a range, a cardinality or a
row-count delta. A routine that silently dropped 40 % of a frame would produce a
complete, well-formed, entirely plausible model and score table with no alert.

This module converts observation into enforcement, and does so with
``SparkOptimizer``-free, dependency-light helpers so that every stage of every
pipeline can afford to call them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from forecasting_ml_framework.exceptions import (
  ColumnRoleConflictError,
  RowCountAssertionError,
  TargetCardinalityError,
)
from forecasting_ml_framework.observability.logging import get_logger

LOGGER = get_logger(__name__)


class QualityPolicy:
  """Declarative policy for the assertions a pipeline stage will enforce.

  The policy is a value object rather than a set of function flags so that a
  model program can express its own tolerance in configuration without the
  validation code knowing about configuration.

  Attributes:
    expected_row_count: The row count the stage received, used as the denominator
      for the tolerance check. ``None`` disables the row-count assertion.
    row_count_tolerance: The maximum permitted *relative* drift in row count.
    max_null_rate: The maximum permitted null rate on any validated column.
    min_rows_for_metrics: Below this row count, a metric stage is a no-op.
  """

  __slots__ = ('expected_row_count', 'row_count_tolerance', 'max_null_rate', 'min_rows_for_metrics')

  def __init__(
    self,
    expected_row_count: int | None = None,
    row_count_tolerance: float = 0.0,
    max_null_rate: float = 1.0,
    min_rows_for_metrics: int = 1,
  ) -> None:
    """Build the policy.

    Args:
      expected_row_count: Row count observed at the previous stage.
      row_count_tolerance: Permitted relative drift, e.g. ``0.01`` for 1 %.
      max_null_rate: Permitted null rate on validated columns, in ``[0, 1]``.
      min_rows_for_metrics: Row count below which metric computation is skipped.
    """
    self.expected_row_count = expected_row_count
    self.row_count_tolerance = row_count_tolerance
    self.max_null_rate = max_null_rate
    self.min_rows_for_metrics = min_rows_for_metrics

  def with_expected(self, row_count: int) -> QualityPolicy:
    """Return a copy of this policy anchored on a given row count.

    Args:
      row_count: The observed row count.

    Returns:
      A new policy.
    """
    return QualityPolicy(
      expected_row_count=row_count,
      row_count_tolerance=self.row_count_tolerance,
      max_null_rate=self.max_null_rate,
      min_rows_for_metrics=self.min_rows_for_metrics,
    )


def assert_row_count(observed: int, policy: QualityPolicy, stage: str, **context: object) -> None:
  """Assert that a stage preserved the row count it was given.

  Args:
    observed: The row count at the current stage.
    policy: The active quality policy.
    stage: A human-readable stage name used in the error message.
    **context: Additional diagnostic context.

  Raises:
    RowCountAssertionError: If the relative drift exceeds the tolerance.
  """
  if policy.expected_row_count is None or policy.expected_row_count == 0:
    return
  drift = abs(observed - policy.expected_row_count) / policy.expected_row_count
  if drift > policy.row_count_tolerance:
    raise RowCountAssertionError(
      f'Stage {stage!r} changed the row count by {drift:.4%}, exceeding the configured tolerance of '
      f'{policy.row_count_tolerance:.4%}. A transform is most likely dropping or duplicating rows, '
      'which would produce a plausible but wrong model.',
      stage=stage,
      observed=observed,
      expected=policy.expected_row_count,
      **context,
    )
  LOGGER.debug('row count assertion passed', extra={'stage': stage, 'observed': observed})


def assert_target_cardinality(distinct_target_values: int, target_col: str, **context: object) -> None:
  """Assert that the target has at least two distinct values.

  Args:
    distinct_target_values: The observed cardinality.
    target_col: The target column name.
    **context: Additional diagnostic context.

  Raises:
    TargetCardinalityError: If the target is single-class.
  """
  if distinct_target_values < 2:
    raise TargetCardinalityError(
      f'The target column {target_col!r} contains {distinct_target_values} distinct value(s). '
      'AUC, lift, calibration and threshold selection are all undefined for a single-class frame, '
      'so the run cannot proceed. Check the label-derivation SQL and the maturity-window filter.',
      target_col=target_col,
      distinct_values=distinct_target_values,
      **context,
    )


def assert_column_roles_disjoint(roles: Mapping[str, Sequence[str]], **context: object) -> None:
  """Assert that no column is declared in two mutually exclusive role lists.

  array (sequence) columns must not also appear in the numerical,
  categorical or indicator lists. ``feature_cols`` is seeded from the
  concatenation of those three, so an array column would be handed to an estimator
  that cannot fit it. A code had no validation for this invariant, so
  the failure surfaced only after the entire feature pipeline had run.

  Args:
    roles: A mapping of role name to its declared columns.
    **context: Additional diagnostic context.

  Raises:
    ColumnRoleConflictError: If any column appears in more than one role.
  """
  from forecasting_ml_framework import constants  # noqa: PLC0415 - avoids a cycle

  exclusive = [constants.ROLE_NUMERICAL, constants.ROLE_CATEGORICAL, constants.ROLE_INDICATOR, constants.ROLE_SEQUENCE]
  seen: dict[str, str] = {}
  conflicts: dict[str, list[str]] = {}
  for role in exclusive:
    for column in roles.get(role, []) or []:
      if column in seen:
        conflicts.setdefault(column, [seen[column]]).append(role)
      else:
        seen[column] = role
  if conflicts:
    rendered = '; '.join(f'{column} declared as {sorted(roles)}' for column, roles in sorted(conflicts.items()))
    raise ColumnRoleConflictError(
      f'Column role declarations overlap: {rendered}. A sequence (array) column must not also be '
      'declared numerical, categorical or indicator, and no column may occupy two scalar roles.',
      conflicts=sorted(conflicts),
      **context,
    )


def assert_columns_present(available: Sequence[str], required: Sequence[str], stage: str, **context: object) -> None:
  """Assert that every required column is present in a frame.

  Args:
    available: The columns the frame actually has.
    required: The columns the stage needs.
    stage: A human-readable stage name.
    **context: Additional diagnostic context.

  Raises:
    ColumnRoleConflictError: If a required column is absent.
  """
  present = set(available)
  missing = [column for column in required if column not in present]
  if missing:
    raise ColumnRoleConflictError(
      f'Stage {stage!r} is missing {len(missing)} required column(s): {sorted(missing)[:20]}. '
      f'Available columns: {sorted(present)[:40]}',
      stage=stage,
      missing=sorted(missing),
      **context,
    )


def null_rate_summary(frame: Any, columns: Sequence[str]) -> dict[str, float]:
  """Compute the null rate of each named column in a pandas frame.

  Args:
    frame: A pandas ``DataFrame``.
    columns: The columns to profile.

  Returns:
    A mapping of column name to null rate in ``[0, 1]``.
  """
  if frame is None or not hasattr(frame, 'empty') or frame.empty:
    return dict.fromkeys(columns, 0.0)
  return {column: float(frame[column].isna().mean()) for column in columns if column in frame.columns}


def assert_null_rate(
  frame: Any,
  columns: Sequence[str],
  policy: QualityPolicy,
  stage: str,
  **context: object,
) -> None:
  """Assert that no validated column exceeds the permitted null rate.

  Args:
    frame: A pandas ``DataFrame``.
    columns: The columns to profile.
    policy: The active quality policy.
    stage: A human-readable stage name.
    **context: Additional diagnostic context.

  Raises:
    RowCountAssertionError: If any column exceeds the permitted null rate.
  """
  if policy.max_null_rate >= 1.0 or frame is None or getattr(frame, 'empty', True):
    return
  offenders = {
    column: rate for column, rate in null_rate_summary(frame, columns).items() if rate > policy.max_null_rate
  }
  if offenders:
    rendered = ', '.join(f'{column}={rate:.2%}' for column, rate in sorted(offenders.items()))
    raise RowCountAssertionError(
      f'Stage {stage!r} produced columns above the permitted null rate of {policy.max_null_rate:.2%}: '
      f'{rendered}. An imputation step is most likely configured with a scope that does not cover '
      'these columns.',
      stage=stage,
      offenders=offenders,
      **context,
    )
