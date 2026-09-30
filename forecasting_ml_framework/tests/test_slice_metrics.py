#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for the slice metric row the evaluation node publishes.

The four confusion counts are the part of this row a human actually reads. The
scalar metrics beside them are computed by scikit-learn and are therefore
trustworthy; the counts are unpacked by hand, and a hand-unpacked two-by-two
matrix is exactly the kind of thing that is transposed and still looks right.

The test that matters is the cross-check, not the arithmetic. A transposed
confusion matrix still sums to the population, so "do the four counts reconcile"
passes on a wrong matrix. What does not survive a transposition is agreement
with precision and recall, because those are derived from the same matrix by a
different route -- precision from the predicted-positive column, recall from the
actual-positive row. If the counts were read in the wrong order, the counts
would contradict the scalars printed next to them, and that is detectable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import confusion_matrix

from forecasting_ml_framework import constants
from forecasting_ml_framework.core.metadata import Metadata
from forecasting_ml_framework.nodes.model_nodes import model_evaluation


def _labelled(pairs: list[tuple[float, float]], threshold: float = 0.5) -> pd.DataFrame:
  """Build a scored frame from (score, label) pairs.

  The hard prediction is produced here rather than inside the evaluation helper,
  so a test can read it back to compute what it expects. Deriving it in a copy
  the test cannot see would make every cross-check vacuous.

  Args:
    pairs: One ``(probability, label)`` pair per row.
    threshold: The decision threshold applied to the probability column.

  Returns:
    A frame carrying the columns the evaluation node reads.
  """
  probabilities = [pair[0] for pair in pairs]
  return pd.DataFrame(
    {
      'entity_id': [f'E{index:04d}' for index in range(len(pairs))],
      constants.DEFAULT_PREDICTION_PROBABILITY_COLUMN: probabilities,
      constants.DEFAULT_PREDICTION_COLUMN: [int(value >= threshold) for value in probabilities],
      'label': [pair[1] for pair in pairs],
    }
  )


def _parameters() -> dict[str, object]:
  """Return the minimal parameters document the evaluation node reads.

  Returns:
    A parameters document carrying the model key and the run date.
  """
  return {
    'model_key': 'metric_fixture',
    'run_date': '2024-01-01',
    'modeling_params': {'prediction_col': constants.DEFAULT_PREDICTION_COLUMN},
  }


def _metadata() -> Metadata:
  """Return a contract naming the columns the fixture carries.

  Returns:
    A contract whose feature list is empty, so nothing is inferred from it.
  """
  metadata = Metadata()
  metadata.target_col = 'label'
  metadata.id_cols = ['entity_id']
  metadata.feature_cols = []
  metadata.numerical_cols = []
  return metadata


def _evaluate(frame: pd.DataFrame) -> dict[str, float]:
  """Run the evaluation node and return the metric row as a mapping.

  Args:
    frame: A scored frame whose hard prediction column is already present.

  Returns:
    The metric row.
  """
  row, _ = model_evaluation(
    frame,
    _parameters(),
    _metadata(),
    constants.DEFAULT_PREDICTION_COLUMN,
    constants.DEFAULT_PREDICTION_PROBABILITY_COLUMN,
    'train',
    # The binary case. `multi_class` is required so that a graph omitting it fails
    # at construction rather than silently defaulting -- which is how the flag came
    # to be dead in the first place. The node also cross-checks this against the
    # observed label cardinality, so the two cannot disagree.
    multi_class=False,
    sequence_flag=False,
  )
  return row.iloc[0].to_dict()


class TestConfusionCounts:
  """The four counts must mean what their names say."""

  @pytest.mark.parametrize(
    'pairs',
    [
      # A heavily imbalanced case, which is the production shape and the one
      # where a transposition is easiest to miss: the majority count is large
      # enough to look plausible in either slot.
      [(0.9, 1)] * 40 + [(0.2, 0)] * 60 + [(0.7, 0)] * 10 + [(0.4, 1)] * 10,
      # A balanced case, where a transposition is obvious.
      [(0.8, 1)] * 5 + [(0.3, 0)] * 5,
      # A single error of each kind.
      [(0.9, 1)] * 3 + [(0.1, 0)] * 3 + [(0.6, 0)] + [(0.2, 1)],
      # A perfect prediction, where every off-diagonal count is zero.
      [(0.95, 1)] * 7 + [(0.05, 0)] * 13,
    ],
  )
  def test_counts_agree_with_scikit_learn(self, pairs: list[tuple[float, float]]) -> None:
    """The counts must equal the matrix scikit-learn computes.

    Args:
      pairs: The ``(probability, label)`` fixture.
    """
    frame = _labelled(pairs)
    row = _evaluate(frame)
    y_true = frame['label'].to_numpy()
    y_pred = frame[constants.DEFAULT_PREDICTION_COLUMN].to_numpy()
    expected = confusion_matrix(y_true, y_pred, labels=[0, 1])
    assert row['tn'] == int(expected[0][0])
    assert row['fp'] == int(expected[0][1])
    assert row['fn'] == int(expected[1][0])
    assert row['tp'] == int(expected[1][1])

  @pytest.mark.parametrize(
    'pairs',
    [
      [(0.9, 1)] * 40 + [(0.2, 0)] * 60 + [(0.7, 0)] * 10 + [(0.4, 1)] * 10,
      [(0.8, 1)] * 5 + [(0.3, 0)] * 5,
      [(0.95, 1)] * 7 + [(0.05, 0)] * 13,
    ],
  )
  def test_precision_is_derived_from_the_reported_counts(self, pairs: list[tuple[float, float]]) -> None:
    """Precision must equal ``tp / (tp + fp)`` from the counts in the same row.

    This is the check that a transposed matrix cannot survive. A matrix read in
    the wrong order still sums to the population, so a reconciliation check
    passes; but it computes a different precision from the one scikit-learn
    reports, and the two then disagree on the same row.

    Args:
      pairs: The ``(probability, label)`` fixture.
    """
    row = _evaluate(_labelled(pairs))
    predicted_positive = row['tp'] + row['fp']
    if predicted_positive:
      assert row['precision'] == pytest.approx(row['tp'] / predicted_positive, abs=0.01)

  @pytest.mark.parametrize(
    'pairs',
    [
      [(0.9, 1)] * 40 + [(0.2, 0)] * 60 + [(0.7, 0)] * 10 + [(0.4, 1)] * 10,
      [(0.8, 1)] * 5 + [(0.3, 0)] * 5,
      [(0.95, 1)] * 7 + [(0.05, 0)] * 13,
    ],
  )
  def test_recall_is_derived_from_the_reported_counts(self, pairs: list[tuple[float, float]]) -> None:
    """Recall must equal ``tp / (tp + fn)`` from the counts in the same row.

    Args:
      pairs: The ``(probability, label)`` fixture.
    """
    row = _evaluate(_labelled(pairs))
    actual_positive = row['tp'] + row['fn']
    if actual_positive:
      assert row['recall'] == pytest.approx(row['tp'] / actual_positive, abs=0.01)

  @pytest.mark.parametrize(
    'pairs',
    [
      [(0.9, 1)] * 40 + [(0.2, 0)] * 60 + [(0.7, 0)] * 10 + [(0.4, 1)] * 10,
      [(0.8, 1)] * 5 + [(0.3, 0)] * 5,
    ],
  )
  def test_counts_sum_to_the_population(self, pairs: list[tuple[float, float]]) -> None:
    """The four counts must reconcile with the row count.

    Necessary but not sufficient -- a transposed matrix satisfies this too. It is
    asserted because it is cheap and because its *insufficiency* is the point
    the other two tests exist to cover.

    Args:
      pairs: The ``(probability, label)`` fixture.
    """
    frame = _labelled(pairs)
    row = _evaluate(frame)
    total = row['tn'] + row['fp'] + row['fn'] + row['tp']
    assert total == len(frame)

  def test_actual_positives_are_recoverable(self) -> None:
    """``tp + fn`` must equal the number of positive labels actually present.

    This is the assertion that names the bug directly. In an imbalanced frame the
    majority class dominates every other count, so a transposition between the
    two diagonal entries moves a small number into a large slot; summing the
    actual-positive row recovers the true class balance regardless of the
    threshold, so any disagreement is a mislabelled count.
    """
    frame = _labelled([(0.9, 1)] * 40 + [(0.2, 0)] * 60 + [(0.7, 0)] * 10 + [(0.4, 1)] * 10)
    row = _evaluate(frame)
    actual_positives = int(frame['label'].sum())
    assert row['tp'] + row['fn'] == actual_positives
    assert row['fp'] + row['tn'] == len(frame) - actual_positives

  def test_perfect_prediction_has_no_off_diagonal(self) -> None:
    """A perfect prediction must leave ``fp`` and ``fn`` at zero.

    Under a transposition the two off-diagonal cells stay correct -- they are the
    off-diagonal of a transpose too -- so this case is clean in either reading
    and is included precisely because it *cannot* detect the error. It is here to
    pin the diagonal semantics independently.
    """
    frame = _labelled([(0.95, 1)] * 7 + [(0.05, 0)] * 13)
    row = _evaluate(frame)
    assert row['fp'] == 0
    assert row['fn'] == 0
    assert row['tp'] == 7
    assert row['tn'] == 13

  def test_inverted_prediction_moves_the_diagonal(self) -> None:
    """Predicting the opposite class must move both diagonal counts to zero.

    Every prediction is wrong under this threshold, so the matrix has zeros on
    the diagonal. A transposed reading would report the same zeros, but the
    *off-diagonal* assignment is what the next assertion pins -- and a matrix
    whose every count is non-zero off the diagonal has ``tp + fp`` equal to the
    prediction count and ``tp + fn`` equal to the actual count.
    """
    frame = _labelled([(0.1, 1)] * 4 + [(0.9, 0)] * 6)
    row = _evaluate(frame)
    assert row['tp'] == 0
    assert row['tn'] == 0
    # Four positives fall below the threshold, so they are false negatives; six
    # negatives rise above it, so they are false positives. The two cells hold
    # the same *kind* of value under a transposition, so only their magnitudes
    # distinguish them -- which is why the magnitudes are what is asserted.
    assert row['fn'] == 4
    assert row['fp'] == 6
    assert row['tp'] + row['fp'] == 6
    assert row['tp'] + row['fn'] == 4


class TestScalarMetrics:
  """The scalar metrics must be the scikit-learn values."""

  def test_auc_is_reported_under_its_own_name(self) -> None:
    """The row publishes ``roc_auc``; a reader looking for ``auc`` finds nothing.

    The name is the warehouse contract, so this asserts the contract rather than
    the metric.
    """
    row = _evaluate(_labelled([(0.9, 1)] * 8 + [(0.1, 0)] * 8))
    assert 'roc_auc' in row
    assert row['roc_auc'] == pytest.approx(1.0)

  def test_perfect_separation_scores_one(self) -> None:
    """Perfectly separated classes must score an AUC of one."""
    row = _evaluate(_labelled([(0.9, 1)] * 10 + [(0.1, 0)] * 10))
    assert row['roc_auc'] == pytest.approx(1.0)
    assert row['accuracy'] == pytest.approx(1.0)

  def test_accuracy_matches_the_counts(self) -> None:
    """Accuracy must equal ``(tp + tn) / n`` from the reported counts."""
    frame = _labelled([(0.9, 1)] * 8 + [(0.2, 0)] * 8 + [(0.6, 0)])
    row = _evaluate(frame)
    assert row['accuracy'] == pytest.approx((row['tp'] + row['tn']) / len(frame), abs=0.01)

  def test_every_slice_uses_the_same_label(self) -> None:
    """The data-type label is carried into the row, so slices stay distinguishable."""
    frame = _labelled([(0.9, 1)] * 5 + [(0.2, 0)] * 5)
    row, _ = model_evaluation(
      frame,
      _parameters(),
      _metadata(),
      constants.DEFAULT_PREDICTION_COLUMN,
      constants.DEFAULT_PREDICTION_PROBABILITY_COLUMN,
      'test',
      False,
      False,
    )
    assert row.iloc[0]['data_type'] == 'test'

  def test_the_frame_is_not_mutated(self) -> None:
    """Evaluation must not write into the caller's frame.

    The node is called once per slice over the *same* scored frame, so a
    mutation would make the second slice's numbers differ from the first for
    reasons that have nothing to do with the data.
    """
    frame = _labelled([(0.9, 1)] * 5 + [(0.2, 0)] * 5)
    before = frame.copy()
    _evaluate(frame)
    pd.testing.assert_frame_equal(frame, before)

  def test_scores_outside_zero_one_are_still_ranked(self) -> None:
    """AUC depends on ranking, not on the probability range.

    A calibrated score and an uncalibrated one are both legitimate inputs, and
    neither is required to lie in ``[0, 1]``. Asserting the metric is computed
    from the ranking is what makes the two interchangeable.
    """
    frame = _labelled([(9.0, 1)] * 6 + [(2.0, 0)] * 6)
    row = _evaluate(frame)
    assert row['roc_auc'] == pytest.approx(1.0)
    assert np.isfinite(row['roc_auc'])
