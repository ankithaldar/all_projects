#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''The tests of the plain focal loss.

The two things that are easy to get wrong are asserted here: the
inertness of the focal term at the shipped parameters, and the
inversion of the argument order.
'''


import numpy as np
import pytest
import tensorflow as tf

from birdcall.losses.tf_loss_focal_standard import FocalLossStandard


CLASSES = 264
BATCH = 3
SNIPPETS = 2

def batch(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
  '''Builds a deterministic batch of labels and predictions.

  Args:
    seed: The seed of the generator.

  Returns:
    A tuple ``(labels, predictions)``, each of shape
    ``(batch, snippets, classes)``.
  '''
  rng = np.random.default_rng(seed)
  labels = (rng.random((BATCH, SNIPPETS, CLASSES)) > 0.9).astype('float32')
  predictions = rng.random((BATCH, SNIPPETS, CLASSES)).astype('float32')
  return labels, predictions

def mean_bce(labels: np.ndarray, predictions: np.ndarray) -> float:
  '''Computes the mean binary cross entropy of a batch.

  Args:
    labels: The labels.
    predictions: The predictions.

  Returns:
    The mean binary cross entropy over every element.
  '''
  criterion = tf.keras.losses.BinaryCrossentropy(from_logits=False,
                                        reduction='sum_over_batch_size')
  flat_labels = labels.reshape(-1)
  flat_predictions = predictions.reshape(-1)
  return float(criterion(flat_labels, flat_predictions))

def test_the_loss_is_a_scalar() -> None:
  '''The criterion returns one number.'''
  labels, predictions = batch()
  value = FocalLossStandard()(labels, predictions)
  assert tuple(value.shape) == ()

def test_the_inert_focal_loss_is_the_mean_bce() -> None:
  '''At gamma=0 the focal term is inert and the loss is the BCE.'''
  labels, predictions = batch()
  value = float(FocalLossStandard(gamma=0.0, alpha=1.0)(labels,
                                                         predictions))
  assert value == pytest.approx(mean_bce(labels, predictions), abs=1e-6)

def test_the_gamma_knob_still_exists() -> None:
  '''A non-zero gamma changes the loss.'''
  labels, predictions = batch()
  inert = float(FocalLossStandard(gamma=0.0)(labels, predictions))
  focused = float(FocalLossStandard(gamma=2.0)(labels, predictions))
  assert focused != pytest.approx(inert, abs=1e-4)
  # The modulating factor is at most one, so the focal loss is never
  # bigger than the plain BCE.
  assert focused < inert

def test_the_reduction_divides_by_the_number_of_elements() -> None:
  '''The reduction is not the sum over the batch size.'''
  labels, predictions = batch()
  value = float(FocalLossStandard(gamma=0.0)(labels, predictions))
  assert value == pytest.approx(mean_bce(labels, predictions), abs=1e-6)

def test_a_swapped_call_is_visibly_different() -> None:
  '''The argument order is inverted, and a swap does not raise.'''
  labels, predictions = batch()
  criterion = FocalLossStandard(gamma=0.0)
  straight = float(criterion(y_true=labels, y_pred=predictions))
  swapped = float(criterion(y_true=predictions, y_pred=labels))
  assert straight != pytest.approx(swapped, abs=1e-4)
  assert np.isfinite(swapped)

def test_the_label_dict_is_accepted() -> None:
  '''The criterion takes the label dict of the engine.'''
  labels, predictions = batch()
  criterion = FocalLossStandard(gamma=0.0)
  with_dict = float(criterion(
      y_true={'all_labels': labels, 'secondary_labels': labels * 0.0},
      y_pred=predictions))
  assert with_dict == pytest.approx(float(criterion(labels, predictions)),
                                    abs=1e-6)

def test_the_compute_loss_override_takes_the_model_output() -> None:
  '''The durable extension point takes the output dict.'''
  labels, predictions = batch()
  criterion = FocalLossStandard(gamma=0.0)
  value = float(criterion.compute_loss(
      x=None, y={'all_labels': labels},
      y_pred={'clipwise_output': predictions}))
  assert value == pytest.approx(mean_bce(labels, predictions), abs=1e-6)

def test_the_components_are_published() -> None:
  '''The loss components are the instrumentation of the criterion.'''
  labels, predictions = batch()
  criterion = FocalLossStandard(gamma=0.0)
  criterion(labels, predictions)
  assert sorted(criterion.components) == sorted(criterion.loss_keys)
  assert float(criterion.components['bce_loss']) == pytest.approx(
      mean_bce(labels, predictions), abs=1e-6)
  assert float(criterion.components['F_loss']) == pytest.approx(
      float(criterion.components['bce_loss']), abs=1e-6)

def test_the_prediction_is_not_sigmoided_again() -> None:
  '''The prediction arrives sigmoided, and is not sigmoided again.'''
  labels, predictions = batch()
  value = float(FocalLossStandard(gamma=0.0)(labels, predictions))
  twice = float(FocalLossStandard(gamma=0.0)(labels, 1.0 / (1.0 + np.exp(
      -predictions))))
  assert value != pytest.approx(twice, abs=1e-4)

def test_the_config_round_trips() -> None:
  '''The configuration of the criterion can be read back.'''
  criterion = FocalLossStandard(gamma=2.0, alpha=0.5)
  config = criterion.get_config()
  assert config['gamma'] == 2.0
  assert config['alpha'] == 0.5

def test_the_criterion_works_inside_a_graph() -> None:
  '''The criterion is callable from a compiled step.'''

  @tf.function
  def step(labels, predictions):
    return FocalLossStandard(gamma=0.0)(labels, predictions)

  labels, predictions = batch()
  value = float(step(tf.constant(labels), tf.constant(predictions)))
  assert value == pytest.approx(mean_bce(labels, predictions), abs=1e-6)
