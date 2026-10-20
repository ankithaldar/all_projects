#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''The tests of the scaled focal loss with a primary label dropout.

The dropout drops 30 percent of the primary positions, it does not
renormalise the loss, and the reported ``FScaled_loss`` is the
**pre-drop** value.
'''

import numpy as np
import pytest

from birdcall.losses.tf_loss_scaled_pos_neg_focal_augd import (
    SedScaledPosNegFocalLossAugd)

CLASSES = 264
BATCH = 4
SNIPPETS = 2


def batch(seed: int = 6, positives: int = 40) -> tuple[np.ndarray,
                                                      np.ndarray]:
  '''Builds a deterministic batch of labels and predictions.

  Args:
    seed: The seed of the generator.
    positives: The number of positive positions per row.

  Returns:
    A tuple ``(labels, predictions)``.
  '''
  rng = np.random.default_rng(seed)
  labels = np.zeros((BATCH, SNIPPETS, CLASSES), dtype='float32')
  labels[:, :, :positives] = 1.0
  predictions = rng.random((BATCH, SNIPPETS, CLASSES)).astype('float32')
  return labels, predictions


def test_the_loss_keys_are_the_same_five() -> None:
  '''The augd criterion publishes the same five components.'''
  assert SedScaledPosNegFocalLossAugd().loss_keys == [
      'bce_loss', 'F_loss', 'FScaled_loss', 'F_loss_0', 'F_loss_1']


def test_the_dropout_drops_thirty_percent_of_the_primary_positions() -> (
    None):
  '''Thirty percent of the primary positions are dropped.

  The original's mask is 1 on a primary position with probability
  ``primary_drop_out_factor`` and the loss is multiplied by ``(1 - mask)``,
  so 30 percent of the primary positions contribute nothing and 70 percent
  survive.
  '''
  classes = 8192
  positives = classes // 2
  labels = np.zeros((1, 1, classes), dtype='float32')
  labels[0, 0, :positives] = 1.0
  secondary = np.zeros((1, 1, classes), dtype='float32')
  # The primary positions carry a large loss and the others a small one,
  # so the fraction that is dropped is directly measurable.
  predictions = np.full((1, 1, classes), 0.001, dtype='float32')
  predictions[0, 0, :positives] = 0.01
  criterion = SedScaledPosNegFocalLossAugd(gamma=0.0, alpha_0=1.0,
                                           alpha_1=1.0, secondary_factor=1.0,
                                           primary_drop_out_factor=0.3,
                                           seed=1234)
  value = float(criterion(y_true=labels, y_pred=predictions))
  positive_bce = float(-np.log(0.01))
  negative_bce = float(-np.log1p(-0.001))
  pre_drop = 0.5 * positive_bce + 0.5 * negative_bce
  dropped = (pre_drop - value) / (0.5 * positive_bce)
  assert dropped == pytest.approx(0.3, abs=0.05)


def test_the_dropped_positions_are_counted_in_the_denominator() -> None:
  '''The loss is not renormalised over the surviving positions.'''
  labels, predictions = batch()
  criterion = SedScaledPosNegFocalLossAugd(gamma=0.0, alpha_0=1.0,
                                           alpha_1=1.0, secondary_factor=1.0,
                                           primary_drop_out_factor=0.3,
                                           seed=1234)
  value = float(criterion(y_true=labels, y_pred=predictions))
  bce = -(labels * np.log(predictions) +
          (1.0 - labels) * np.log(1.0 - predictions))
  # The positive positions are dropped, the negative ones are not, and
  # the mean is over every element: the loss is not renormalised over
  # the surviving positions.
  all_dropped = float((bce * (1.0 - labels)).mean())
  none_dropped = float(bce.mean())
  assert all_dropped < value < none_dropped
  expected = float((bce * (1.0 - 0.3 * labels)).mean())
  assert value == pytest.approx(expected, rel=0.25)


def test_the_reported_scaled_loss_is_the_pre_drop_value() -> None:
  '''The component is the value before the dropout.'''
  labels, predictions = batch()
  criterion = SedScaledPosNegFocalLossAugd(gamma=0.0, alpha_0=1.0,
                                           alpha_1=1.0, secondary_factor=1.0,
                                           primary_drop_out_factor=0.3,
                                           seed=1234)
  value = float(criterion(y_true=labels, y_pred=predictions))
  scaled = float(criterion.components['FScaled_loss'])
  assert scaled != pytest.approx(value, abs=1e-6)


def test_the_mask_is_seeded() -> None:
  '''Two criteria with the same seed draw the same mask.'''
  labels, predictions = batch()
  first = SedScaledPosNegFocalLossAugd(gamma=0.0, seed=1234)
  second = SedScaledPosNegFocalLossAugd(gamma=0.0, seed=1234)
  assert float(first(y_true=labels, y_pred=predictions)) == pytest.approx(
      float(second(y_true=labels, y_pred=predictions)), abs=1e-6)


def test_a_swapped_call_is_visibly_different() -> None:
  '''The argument order is inverted, and a swap does not raise.'''
  labels, predictions = batch()
  criterion = SedScaledPosNegFocalLossAugd(gamma=0.0)
  straight = float(criterion(y_true=labels, y_pred=predictions))
  swapped = float(criterion(y_true=predictions, y_pred=labels))
  assert straight != pytest.approx(swapped, abs=1e-4)
  assert np.isfinite(swapped)


def test_the_config_round_trips() -> None:
  '''The configuration of the criterion can be read back.'''
  criterion = SedScaledPosNegFocalLossAugd(primary_drop_out_factor=0.5,
                                           seed=99)
  config = criterion.get_config()
  assert config['primary_drop_out_factor'] == 0.5
  assert config['seed'] == 99
