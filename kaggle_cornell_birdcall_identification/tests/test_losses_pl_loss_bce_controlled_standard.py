#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests for the L-soft binary cross entropy.

The target is softened with the prediction itself and the criterion
publishes **no** component metrics: ``loss_keys`` is empty and the
returned dict is empty.  That is the criterion's real behaviour and it
is not a defect.
'''

import torch
import torch.nn.functional as F

from birdcall.losses.pl_loss_bce_controlled_standard import (
    BCEControlledStandardLoss,
)


def test_bce_controlled_standard_loss_is_the_l_soft_bce(loss_prediction,
                                                         loss_labels):
  '''The soft target is `(beta * y_true + (1 - beta) * y_pred) * y_true`.'''
  criterion = BCEControlledStandardLoss(beta=0.5)
  y_pred = loss_prediction
  target = loss_labels
  loss, parts = criterion(y_pred, target)
  classes = y_pred.shape[-1]
  clamped = torch.clamp(y_pred, 1e-5, 1.0 - 1e-5)
  y_true = target['all_labels'].reshape(-1, classes)
  soft = (0.5 * y_true + 0.5 * clamped.reshape(-1, classes)) * y_true
  expected = F.binary_cross_entropy(clamped.reshape(-1, classes), soft,
                                    reduction='none').mean(1).mean()
  assert torch.allclose(loss, expected, atol=1e-6)
  assert parts == {}


def test_bce_controlled_standard_loss_publishes_no_keys():
  '''The criterion contributes zero extra component metrics.'''
  assert BCEControlledStandardLoss.loss_keys == []
  criterion = BCEControlledStandardLoss()
  assert criterion.loss_keys == []


def test_criterion_returns_a_scalar_and_an_empty_dict(loss_prediction,
                                                       loss_labels):
  '''The criterion returns `(scalar, dict)` and the dict is empty.'''
  criterion = BCEControlledStandardLoss()
  loss, parts = criterion(loss_prediction, loss_labels)
  assert isinstance(loss, torch.Tensor)
  assert loss.ndim == 0
  assert parts == {}
