#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""The dataframe schema contract carried between nodes.

``Metadata`` is the single source of truth for the model feature vector. The
``feature_resolution`` method is the most consequential piece of the domain
layer: every model resolves its features exactly once, during ``train``, and the
result is baked into the pickled model handle. That is what makes the handle a
self-describing, self-sufficient artefact.
"""

from __future__ import annotations

import copy as copy_module
from collections.abc import Iterable, Sequence
from typing import Any

from forecasting_ml_framework.observability.logging import get_logger
from forecasting_ml_framework.utils.text import match_any

LOGGER = get_logger(__name__)


class Metadata:
  """A mutable, serialisable description of the frame flowing between nodes.

  The object carries the declared column roles, the derived imbalance statistics
  produced by the sampling stage, and the operating-point configuration mirrored
  by the trainer. It is passed by reference between routines, so the mutation
  discipline is deliberate and documented per member: routines that *write* new
  features take a defensive copy, routines that merely read do not.
  """

  #: Column-role attributes that :meth:`feature_resolution` can address by name.
  ROLE_ATTRIBUTES = (
    'feature_cols',
    'numerical_cols',
    'categorical_cols',
    'seq_cols',
    'id_cols',
  )

  def __init__(
    self,
    target_col: str | None = None,
    feature_cols: Sequence[str] | None = None,
    numerical_cols: Sequence[str] | None = None,
    categorical_cols: Sequence[str] | None = None,
    seq_cols: Sequence[str] | None = None,
    id_cols: Sequence[str] | None = None,
    intermediate_file_path: str | None = None,
  ) -> None:
    """Build the metadata object.

    Args:
      target_col: The label column name.
      feature_cols: Every column eligible to be a model feature, **in model
        order**. Order is significant: it becomes the estimator's feature-vector
        order for the life of the persisted artefact.
      numerical_cols: Columns carrying numeric features.
      categorical_cols: Columns carrying categorical features.
      seq_cols: Columns carrying ``array<float>`` behavioural sequences.
      id_cols: Identifier columns used for projection and join keys.
      intermediate_file_path: Root under which routines that cannot pickle their
        fitted state in place write external artefacts.
    """
    self.target_col = target_col
    self.feature_cols = list(feature_cols) if feature_cols else []
    self.numerical_cols = list(numerical_cols) if numerical_cols else []
    self.categorical_cols = list(categorical_cols) if categorical_cols else []
    self.seq_cols = list(seq_cols) if seq_cols else []
    self.id_cols = list(id_cols) if id_cols else []
    self.intermediate_file_path = intermediate_file_path

    # --- Derived imbalance statistics (set by the sampling stage) ------------
    #: P(y = 1) in the *population*, measured before any reweighting. Together
    #: with ``sample_rate`` this is the sole input to prior-shift calibration.
    self.real_rate: float | None = None
    #: Negative/positive count ratio (binary only).
    self.scale_pos_weight: float | None = None
    #: ``[neg_w, pos_w]`` for binary or ``{label: w}`` for multiclass.
    self.class_weights: Any = None

    # --- Model identity (lifted out of the data by the registry nodes) -------
    self.model_key: str | None = None

    # --- Operating-point mirrors (set by the trainer) -----------------------
    self.threshold: float | None = None
    self.calib_threshold: float | None = None

  # ---------------------------------------------------------------------- #
  # Introspection
  # ---------------------------------------------------------------------- #
  def copy(self) -> Metadata:
    """Return a deep copy of this object.

    Routines that mutate the contract take a copy first, so a mutation cannot
    escape into the caller's frame unintentionally.

    Returns:
      A deep copy.
    """
    return copy_module.deepcopy(self)

  def clean_duplicates(self) -> None:
    """Deduplicate every column list while preserving first-seen order."""
    for attribute in self.ROLE_ATTRIBUTES:
      seen: set[str] = set()
      ordered: list[str] = []
      for column in getattr(self, attribute):
        if column not in seen:
          seen.add(column)
          ordered.append(column)
      setattr(self, attribute, ordered)

  def __str__(self) -> str:
    """Render the contract for the run log.

    Returns:
      A comma-joined ``key: value`` string.
    """
    return ', '.join(
      f'{key}: {value}' for key, value in self.__dict__.items() if value not in (None, [], {})
    )

  def __repr__(self) -> str:
    """Render a short, unambiguous representation.

    Returns:
      The representation string.
    """
    return (
      f'Metadata(target_col={self.target_col!r}, features={len(self.feature_cols)}, '
      f'numerical={len(self.numerical_cols)}, categorical={len(self.categorical_cols)}, '
      f'seq={len(self.seq_cols)}, ids={len(self.id_cols)})'
    )

  # ---------------------------------------------------------------------- #
  # The feature-resolution funnel
  # ---------------------------------------------------------------------- #
  def feature_resolution(
    self,
    args: Sequence[str | Sequence[str]] = ('feature_cols',),
    feature_exclusions: Sequence[str] | None = None,
    feature_inclusions: Sequence[str] | None = None,
  ) -> list[str]:
    """Resolve the ordered feature list for a routine or a trainer.

    The resolution is *double-gated*: a column must appear both in the requested
    role and in ``feature_cols``. The two failure modes this prevents are the
    ones that make a mis-registered transform silently invisible:

    * a routine that appends to a role list without appending to
      ``feature_cols`` produces an invisible column;
    * a routine that appends to ``feature_cols`` for a column it did not emit
      produces a ``KeyError`` deep inside the trainer.

    Order is preserved from the source list, which is
    ``numerical + categorical + indicator`` as constructed by the load node. That
    ordering is therefore the model's feature order for all time, because the
    resolved list is baked into the persisted model handle.

    Args:
      args: Role attribute names, and/or literal column lists to union in.
      feature_exclusions: Regular expressions matched at the start of a column
        name. ``'^tenure_.*'`` excludes all tenure-derived columns; a bare
        ``'tenure'`` also matches ``'tenure_01'``.
      feature_inclusions: A literal allow-list. Applied before exclusions, so
        exclusions win.

    Returns:
      The resolved, ordered feature list.
    """
    candidates: list[str] = []
    for arg in args or ():
      if isinstance(arg, str):
        candidates.extend(getattr(self, arg, []) or [])
      else:
        candidates.extend(arg or [])

    declared = set(self.feature_cols)
    resolved = [column for column in _dedupe(candidates) if column in declared]

    if feature_inclusions:
      allowed = set(feature_inclusions)
      resolved = [column for column in resolved if column in allowed]

    return [column for column in resolved if not match_any(feature_exclusions, column)]

  # ---------------------------------------------------------------------- #
  # Mutation helpers used by the routine family
  # ---------------------------------------------------------------------- #
  def register_derived_feature(self, name: str, *, numerical: bool = True) -> None:
    """Register a column emitted by a transform as a model feature.

    The double registration in ``feature_cols`` *and* ``numerical_cols`` is the
    established protocol of the routine family: it makes the new column visible
    both to the trainer and to every later numeric-only routine.

    Args:
      name: The emitted column name.
      numerical: Whether the column should also enter ``numerical_cols``.
    """
    self.feature_cols.append(name)
    if numerical:
      self.numerical_cols.append(name)
    self.clean_duplicates()

  def replace_feature(self, old_name: str, new_name: str, *, numerical: bool = True) -> None:
    """Replace a source column with an encoded derivative of it.

    The retired name is removed from *every* role list it appears in, not only from
    ``feature_cols``. Leaving it behind produced a contract describing columns the
    frame no longer carried, and a routine scoping off ``numerical_cols`` later
    resolved one of them and raised on a perfectly valid configuration -- an error
    attributed to the data when the cause was the contract.

    Args:
      old_name: The column to retire.
      new_name: The column that replaces it.
      numerical: Whether the new column should enter ``numerical_cols``.
    """
    self.feature_cols = [column for column in self.feature_cols if column != old_name]
    self.numerical_cols = [column for column in self.numerical_cols if column != old_name]
    self.categorical_cols = [column for column in self.categorical_cols if column != old_name]
    self.feature_cols.append(new_name)
    if numerical:
      self.numerical_cols.append(new_name)
    self.clean_duplicates()

  def reclassify_as_numerical(self, names: Iterable[str]) -> None:
    """Move columns from the categorical namespace to the numerical namespace.

    Used by target-supervised binning, whose output is a numeric bin index even
    when the input was a string category.

    Args:
      names: The columns to reclassify.
    """
    wanted = set(names)
    self.categorical_cols = [column for column in self.categorical_cols if column not in wanted]
    self.numerical_cols.extend(sorted(wanted))
    self.clean_duplicates()

  def detach_sequence_features(self) -> list[str]:
    """Remove array columns from the feature list before they reach an estimator.

    The sequence family registers its *summaries*, which are scalar and
    estimator-consumable, but the array column it read from is itself a member of
    the feature list. Enforcing the disjointness here —
      once, at construction — is strictly better than relying on operators to
      remember the configuration precondition.

    Returns:
      The array columns that were removed.
    """
    if not self.seq_cols:
      return []
    sequences = set(self.seq_cols)
    removed = [column for column in self.feature_cols if column in sequences]
    self.feature_cols = [column for column in self.feature_cols if column not in sequences]
    if removed:
      LOGGER.info(
        'Detached %d sequence column(s) from the model feature list; summaries remain as features',
        len(removed),
        extra={'columns': removed},
      )
    return removed


def _dedupe(values: Iterable[str]) -> list[str]:
  """Deduplicate an iterable while preserving first-seen order.

  Args:
    values: The values to deduplicate.

  Returns:
    The ordered unique values.
  """
  seen: set[str] = set()
  ordered: list[str] = []
  for value in values:
    if value not in seen:
      seen.add(value)
      ordered.append(value)
  return ordered
