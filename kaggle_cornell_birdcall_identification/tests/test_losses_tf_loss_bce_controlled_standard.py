#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''The tests of the L-soft BCE criterion.

The criterion of the winning configs: the target is softened by the
prediction, and the prediction is **not** sigmoided again.
'''

import numpy as np
import pytest

from birdcall.losses.tf_loss_bce_controlled_standard import (
    BCEControlledStandardLoss)

CLASSES = 264
BATCH = 3
SNIPPETS = 2


def batch(seed: int = 8) -> tuple[np.ndarray, np.ndarray]:
  '''Builds a deterministic batch of labels and predictions.

  Args:
    seed: The seed of the generator.

  Returns:
    A tuple ``(labels, predictions)``.
  '''
  rng = np.random.default_rng(seed)
  labels = (rng.random((BATCH, SNIPPETS, CLASSES)) > 0.9).astype('float32')
  predictions = rng.random((BATCH, SNIPPETS, CLASSES)).astype('float32')
  return labels, predictions


def reference(labels: np.ndarray, predictions: np.ndarray,
              beta: float = 0.5) -> float:
  '''Computes the L-soft BCE of a batch.

  Args:
    labels: The labels.
    predictions: The predictions.
    beta: The softness of the label.

  Returns:
    The scalar loss.
  '''
  clipped = np.clip(predictions, 1e-5, 1.0 - 1e-5)
  soft = (beta * labels + (1.0 - beta) * clipped) * labels
  bce = -(soft * np.log(clipped) + (1.0 - soft) * np.log(1.0 - clipped))
  return float(bce.mean(axis=2).mean())


def test_the_loss_keys_are_empty() -> None:
  '''The criterion contributes no component metric, on purpose.'''
  assert BCEControlledStandardLoss().loss_keys == []


def test_the_components_are_empty() -> None:
  '''The components dict is empty too.'''
  labels, predictions = batch()
  criterion = BCEControlledStandardLoss(beta=0.5)
  criterion(labels, predictions)
  assert criterion.components == {}


def test_the_loss_matches_the_reference() -> None:
  '''The criterion is the L-soft BCE of the original.'''
  labels, predictions = batch()
  value = float(BCEControlledStandardLoss(beta=0.5)(labels, predictions))
  assert value == pytest.approx(reference(labels, predictions, 0.5), abs=1e-6)


def test_the_beta_knob_exists() -> None:
  '''A different softness gives a different loss.'''
  labels, predictions = batch()
  half = float(BCEControlledStandardLoss(beta=0.5)(labels, predictions))
  full = float(BCEControlledStandardLoss(beta=1.0)(labels, predictions))
  assert half != pytest.approx(full, abs=1e-6)
  # With beta=1 the target is the label itself, and the criterion is
  # the plain BCE.
  plain = -(labels * np.log(predictions) +
            (1.0 - labels) * np.log(1.0 - predictions)).mean()
  assert full == pytest.approx(plain, abs=1e-5)


def test_the_prediction_is_not_sigmoided_again() -> None:
  '''The prediction arrives sigmoided, and is not sigmoided again.'''
  labels, predictions = batch()
  value = float(BCEControlledStandardLoss(beta=0.5)(labels, predictions))
  twice = float(BCEControlledStandardLoss(beta=0.5)(
      labels, 1.0 / (1.0 + np.exp(-predictions))))
  assert value != pytest.approx(twice, abs=1e-4)


def test_a_swapped_call_is_visibly_different() -> None:
  '''The argument order is inverted, and a swap does not raise.'''
  labels, predictions = batch()
  criterion = BCEControlledStandardLoss(beta=0.5)
  straight = float(criterion(y_true=labels, y_pred=predictions))
  swapped = float(criterion(y_true=predictions, y_pred=labels))
  assert straight != pytest.approx(swapped, abs=1e-4)
  assert np.isfinite(swapped)


def test_the_compute_loss_override_takes_the_model_output() -> None:
  '''The durable extension point takes the output dict.'''
  labels, predictions = batch()
  criterion = BCEControlledStandardLoss(beta=0.5)
  value = float(criterion.compute_loss(
      x=None, y={'all_labels': labels},
      y_pred={'clipwise_output': predictions}))
  assert value == pytest.approx(reference(labels, predictions, 0.5), abs=1e-6)


def test_the_config_round_trips() -> None:
  '''The configuration of the criterion can be read back.'''
  config = BCEControlledStandardLoss(beta=0.25).get_config()
  assert config['beta'] == 0.25
