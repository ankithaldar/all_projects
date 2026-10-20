#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests for the scaled focal loss with a primary label dropout.

The criterion keeps every primary position with probability
``primary_drop_out_factor`` (0.3), so thirty percent of the primary
positions contribute nothing.  Two details of the original are
reproduced on purpose: the ``FScaled_loss`` entry of the dict is the
**pre-drop** value, and the dropped positions still count in the
denominator.
'''

import pytest
import torch

from birdcall.losses.pl_loss_scaled_pos_neg_focal import (
    SedScaledPosNegFocalLoss,
)
from birdcall.losses.pl_loss_scaled_pos_neg_focal_augd import (
    SedScaledPosNegFocalLossAugd,
)


def test_augd_drops_about_thirty_percent_of_the_primary_positions():
  '''The dropped positions do not contribute to the loss.'''
  classes = 8192
  positives = classes // 2
  all_labels = torch.zeros(1, 1, classes)
  all_labels[0, 0, :positives] = 1.0
  secondary = torch.zeros(1, 1, classes)
  # The primary positions carry a large loss, the others a small one, so
  # the drop is easy to see.
  y_pred = torch.full((1, 1, classes), 0.001)
  y_pred[0, 0, :positives] = 0.01
  criterion = SedScaledPosNegFocalLossAugd(gamma=0.0, alpha_0=1.0,
                                           alpha_1=1.0, secondary_factor=1.0,
                                           primary_drop_out_factor=0.3,
                                           seed=1234)
  loss, parts = criterion(y_pred, {'all_labels': all_labels,
                                   'secondary_labels': secondary})
  positive_bce = float(-torch.log(torch.tensor(0.01)))
  negative_bce = float(-torch.log1p(-torch.tensor(0.001)))
  pre_drop = 0.5 * positive_bce + 0.5 * negative_bce
  assert float(parts['FScaled_loss']) == pytest.approx(pre_drop, rel=1e-6)
  # The pre-drop value is the one in the dict, the returned scalar is
  # the post-drop one.
  assert float(loss) < float(parts['FScaled_loss'])
  dropped = (pre_drop - float(loss)) / (0.5 * positive_bce)
  # The original's mask is 1 on a primary position with probability
  # `primary_drop_out_factor` and the loss is multiplied by `(1 - mask)`,
  # so 30 percent of the primary positions are dropped.
  assert dropped == pytest.approx(0.3, abs=0.05)


def test_augd_reports_the_pre_drop_value_in_the_dict(loss_prediction,
                                                    loss_labels):
  '''The dict holds the value the criterion would have had.'''
  criterion = SedScaledPosNegFocalLossAugd(seed=1234)
  y_pred = loss_prediction
  target = loss_labels
  loss, parts = criterion(y_pred, target)
  plain = SedScaledPosNegFocalLoss()
  _, plain_parts = plain(y_pred, target)
  assert float(parts['FScaled_loss']) == pytest.approx(
      float(plain_parts['FScaled_loss']), rel=1e-6)
  assert float(loss) <= float(parts['FScaled_loss'])


def test_augd_is_reproducible_for_a_fixed_seed(loss_prediction, loss_labels):
  '''The same seed gives the same dropout mask.'''
  y_pred = loss_prediction
  target = loss_labels
  first = SedScaledPosNegFocalLossAugd(seed=7)(y_pred, target)[0]
  second = SedScaledPosNegFocalLossAugd(seed=7)(y_pred, target)[0]
  assert float(first) == pytest.approx(float(second), rel=1e-6)


def test_loss_keys_are_published_as_a_class_attribute():
  '''The published keys are the ones the engine logs.'''
  assert SedScaledPosNegFocalLossAugd.loss_keys == [
      'bce_loss', 'F_loss', 'FScaled_loss', 'F_loss_0', 'F_loss_1']
  assert isinstance(SedScaledPosNegFocalLossAugd.loss_keys, list)


def test_criterion_returns_a_scalar_and_the_published_keys(loss_prediction,
                                                           loss_labels):
  '''The criterion returns `(scalar, dict)` and the dict has the keys.'''
  criterion = SedScaledPosNegFocalLossAugd()
  loss, parts = criterion(loss_prediction, loss_labels)
  assert isinstance(loss, torch.Tensor)
  assert loss.ndim == 0
  assert list(parts.keys()) == ['bce_loss', 'F_loss', 'FScaled_loss',
                                'F_loss_0', 'F_loss_1']
  for value in parts.values():
    assert isinstance(value, torch.Tensor)
    assert value.ndim == 0
