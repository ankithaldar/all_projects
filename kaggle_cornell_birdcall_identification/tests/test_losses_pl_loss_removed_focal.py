#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests for the focal loss with the secondary species removed.

The secondary species are the weak, noisier part of the weak label, so
they receive **exactly zero** gradient here.  This is the hard version
of the down-weighting the scaled criterion applies continuously.
'''

import torch
import torch.nn.functional as F

from birdcall.losses.pl_loss_removed_focal import SedRemovedFocalLoss


def test_removed_focal_loss_zeroes_the_secondary_species(loss_prediction,
                                                         loss_labels):
  '''The secondary positions do not contribute to the loss.'''
  criterion = SedRemovedFocalLoss(gamma=0.0, alpha=1.0)
  y_pred = loss_prediction
  target = loss_labels
  loss, parts = criterion(y_pred, target)
  classes = y_pred.shape[-1]
  clamped = torch.clamp(y_pred, 1e-5, 1.0 - 1e-5)
  bce = F.binary_cross_entropy(clamped.reshape(-1, classes),
                               target['all_labels'].reshape(-1, classes),
                               reduction='none')
  secondary = target['secondary_labels'].reshape(-1, classes)
  expected = ((secondary <= 0.0).float() * bce).mean()
  assert torch.allclose(parts['FRemoved_loss'], expected, atol=1e-6)
  assert torch.allclose(loss, expected, atol=1e-6)


def test_loss_keys_are_published_as_a_class_attribute():
  '''The published keys are the ones the engine logs.'''
  assert SedRemovedFocalLoss.loss_keys == ['bce_loss', 'F_loss',
                                           'FRemoved_loss']
  assert isinstance(SedRemovedFocalLoss.loss_keys, list)


def test_criterion_returns_a_scalar_and_the_published_keys(loss_prediction,
                                                           loss_labels):
  '''The criterion returns `(scalar, dict)` and the dict has the keys.'''
  criterion = SedRemovedFocalLoss()
  loss, parts = criterion(loss_prediction, loss_labels)
  assert isinstance(loss, torch.Tensor)
  assert loss.ndim == 0
  assert list(parts.keys()) == ['bce_loss', 'F_loss', 'FRemoved_loss']
  for value in parts.values():
    assert isinstance(value, torch.Tensor)
    assert value.ndim == 0
