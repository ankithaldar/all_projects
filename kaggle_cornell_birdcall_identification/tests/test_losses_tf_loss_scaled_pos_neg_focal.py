#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''The tests of the scaled positive/negative focal loss.

The criterion of nineteen of the twenty-one shipped configs, and the
one whose **inertness** is the documented invariant: at the shipped
parameters it is bit for bit the mean BCE.
'''


import tensorflow as tf
import numpy as np
import pytest

from birdcall.losses.tf_loss_scaled_pos_neg_focal import (
    SedScaledPosNegFocalLoss)


CLASSES = 264
BATCH = 3
SNIPPETS = 2

def batch(seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  '''Builds a deterministic batch of labels, secondary labels and scores.

  Args:
    seed: The seed of the generator.

  Returns:
    A tuple ``(labels, secondary labels, predictions)``.
  '''
  rng = np.random.default_rng(seed)
  labels = (rng.random((BATCH, SNIPPETS, CLASSES)) > 0.9).astype('float32')
  secondary = np.zeros_like(labels)
  # A few positions are both positive and secondary.
  secondary[:, :, :5] = labels[:, :, :5]
  predictions = rng.random((BATCH, SNIPPETS, CLASSES)).astype('float32')
  return labels, secondary, predictions

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
  return float(criterion(labels.reshape(-1), predictions.reshape(-1)))

def test_the_loss_keys_are_published() -> None:
  '''The five component names are the documented ones.'''
  criterion = SedScaledPosNegFocalLoss()
  assert criterion.loss_keys == ['bce_loss', 'F_loss', 'FScaled_loss',
                                 'F_loss_0', 'F_loss_1']

def test_the_inert_loss_is_the_mean_bce() -> None:
  '''The shipped parameters collapse the criterion onto the BCE.'''
  labels, secondary, predictions = batch()
  criterion = SedScaledPosNegFocalLoss(gamma=0.0, alpha_0=1.0, alpha_1=1.0,
                                       secondary_factor=1.0)
  value = float(criterion(
      y_true={'all_labels': labels, 'secondary_labels': secondary},
      y_pred=predictions))
  assert value == pytest.approx(mean_bce(labels, predictions), abs=1e-6)

def test_the_secondary_factor_down_weights_the_secondary_positions() -> None:
  '''The secondary species are scaled down, not removed.'''
  labels, secondary, predictions = batch()
  scaled = float(SedScaledPosNegFocalLoss(gamma=0.0, alpha_0=1.0,
                                          alpha_1=1.0,
                                          secondary_factor=0.1)(
      y_true={'all_labels': labels, 'secondary_labels': secondary},
      y_pred=predictions))
  inert = float(SedScaledPosNegFocalLoss(gamma=0.0, alpha_0=1.0, alpha_1=1.0,
                                         secondary_factor=1.0)(
      y_true={'all_labels': labels, 'secondary_labels': secondary},
      y_pred=predictions))
  assert scaled < inert

def test_the_gamma_knob_still_exists() -> None:
  '''A non-zero gamma changes the loss.'''
  labels, secondary, predictions = batch()
  target = {'all_labels': labels, 'secondary_labels': secondary}
  inert = float(SedScaledPosNegFocalLoss(gamma=0.0)(target, predictions))
  focused = float(SedScaledPosNegFocalLoss(gamma=2.0)(target, predictions))
  assert focused != pytest.approx(inert, abs=1e-4)

def test_a_swapped_call_is_visibly_different() -> None:
  '''The argument order is inverted, and a swap does not raise.'''
  labels, secondary, predictions = batch()
  criterion = SedScaledPosNegFocalLoss(gamma=0.0)
  straight = float(criterion(y_true=labels, y_pred=predictions))
  swapped = float(criterion(y_true=predictions, y_pred=labels))
  assert straight != pytest.approx(swapped, abs=1e-4)
  assert np.isfinite(swapped)

def test_the_components_are_published() -> None:
  '''The five components are the instrumentation of the criterion.'''
  labels, secondary, predictions = batch()
  criterion = SedScaledPosNegFocalLoss(gamma=0.0, alpha_0=1.0, alpha_1=1.0,
                                       secondary_factor=1.0)
  criterion(y_true={'all_labels': labels, 'secondary_labels': secondary},
           y_pred=predictions)
  assert sorted(criterion.components) == sorted(criterion.loss_keys)
  assert float(criterion.components['bce_loss']) == pytest.approx(
      mean_bce(labels, predictions), abs=1e-6)
  assert float(criterion.components['FScaled_loss']) == pytest.approx(
      mean_bce(labels, predictions), abs=1e-6)
  # The positive and the negative parts add up to the focal term.
  assert float(criterion.components['F_loss_0']) + float(
      criterion.components['F_loss_1']) == pytest.approx(
          float(criterion.components['F_loss']), abs=1e-5)

def test_the_compute_loss_override_takes_the_model_output() -> None:
  '''The durable extension point takes the output dict.'''
  labels, secondary, predictions = batch()
  criterion = SedScaledPosNegFocalLoss(gamma=0.0, alpha_0=1.0, alpha_1=1.0,
                                       secondary_factor=1.0)
  value = float(criterion.compute_loss(
      x=None,
      y={'all_labels': labels, 'secondary_labels': secondary},
      y_pred={'clipwise_output': predictions}))
  assert value == pytest.approx(mean_bce(labels, predictions), abs=1e-6)

def test_the_config_round_trips() -> None:
  '''The configuration of the criterion can be read back.'''
  criterion = SedScaledPosNegFocalLoss(gamma=2.0, alpha_0=0.5, alpha_1=0.25,
                                       secondary_factor=0.1)
  config = criterion.get_config()
  assert config['gamma'] == 2.0
  assert config['alpha_0'] == 0.5
  assert config['alpha_1'] == 0.25
  assert config['secondary_factor'] == 0.1
