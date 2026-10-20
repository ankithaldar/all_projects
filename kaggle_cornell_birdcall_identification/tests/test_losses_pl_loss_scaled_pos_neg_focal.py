#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests for the scaled positive/negative focal loss.

This is the criterion nineteen of the twenty-one shipped configs select.
The most important assertion in this file is the **inertness** test:
``SedScaledPosNegFocalLoss(gamma=0, alpha_0=1, alpha_1=1,
secondary_factor=1)`` must equal ``nn.BCELoss()``.  The focal machinery
is present, instrumented and inert; "fixing" it would be a different
experiment with a different result.
'''

import torch
import torch.nn.functional as F
from torch import nn

from birdcall.losses.pl_loss_scaled_pos_neg_focal import (
    SedScaledPosNegFocalLoss,
)


def test_scaled_pos_neg_focal_loss_is_inert_and_equals_bce(loss_prediction,
                                                            loss_labels):
  '''At the shipped parameters the criterion is the mean BCE.'''
  criterion = SedScaledPosNegFocalLoss(gamma=0.0, alpha_0=1.0, alpha_1=1.0,
                                       secondary_factor=1.0)
  y_pred = loss_prediction
  target = loss_labels
  loss, parts = criterion(y_pred, target)
  classes = y_pred.shape[-1]
  reference = nn.BCELoss()(y_pred.reshape(-1, classes),
                           target['all_labels'].reshape(-1, classes))
  assert torch.allclose(loss, reference, atol=1e-6)
  assert torch.allclose(parts['bce_loss'], reference, atol=1e-6)
  assert torch.allclose(parts['F_loss'], reference, atol=1e-6)
  assert torch.allclose(parts['FScaled_loss'], reference, atol=1e-6)
  # The positive and negative halves reconstruct the BCE exactly.
  assert torch.allclose(parts['F_loss_0'] + parts['F_loss_1'],
                         parts['bce_loss'], atol=1e-6)


def test_secondary_factor_down_weights_the_secondary_species(
    loss_prediction, loss_labels):
  '''The secondary positions are scaled by the secondary factor.'''
  criterion = SedScaledPosNegFocalLoss(gamma=0.0, alpha_0=1.0, alpha_1=1.0,
                                       secondary_factor=0.1)
  y_pred = loss_prediction
  target = loss_labels
  loss, parts = criterion(y_pred, target)
  classes = y_pred.shape[-1]
  clamped = torch.clamp(y_pred, 1e-5, 1.0 - 1e-5)
  bce = F.binary_cross_entropy(clamped.reshape(-1, classes),
                               target['all_labels'].reshape(-1, classes),
                               reduction='none')
  secondary = target['secondary_labels'].reshape(-1, classes)
  mask = torch.where(secondary > 0.0, 0.1, 1.0)
  assert torch.allclose(parts['FScaled_loss'], (mask * bce).mean(), atol=1e-6)
  assert float(loss) < float(parts['bce_loss'])


def test_mask_construction_carries_no_gradient(loss_prediction, loss_labels):
  '''The secondary labels only ever enter through the mask.'''
  criterion = SedScaledPosNegFocalLoss(gamma=2.0, alpha_0=0.5, alpha_1=1.5,
                                       secondary_factor=0.1)
  y_pred = loss_prediction.requires_grad_(True)
  target = loss_labels
  target['all_labels'] = target['all_labels'].clone().requires_grad_(True)
  target['secondary_labels'] = (target['secondary_labels'].clone()
                                 .requires_grad_(True))
  loss, _ = criterion(y_pred, target)
  loss.backward()
  # The secondary labels only ever enter through the mask, so the mask
  # must not be a differentiable path.
  assert target['secondary_labels'].grad is None
  # The labels themselves still are, through the BCE.
  assert target['all_labels'].grad is not None
  assert torch.isfinite(target['all_labels'].grad).all()
  assert torch.isfinite(y_pred.grad).all()


def test_masks_are_built_under_no_grad(monkeypatch, loss_prediction,
                                        loss_labels):
  '''No mask is ever built with the gradient enabled.'''
  seen = []
  original = torch.where

  def spy(*args, **kwargs):
    seen.append(torch.is_grad_enabled())
    return original(*args, **kwargs)

  monkeypatch.setattr(torch, 'where', spy)
  criterion = SedScaledPosNegFocalLoss(gamma=2.0, alpha_0=0.5, alpha_1=1.5,
                                       secondary_factor=0.1)
  y_pred = loss_prediction.requires_grad_(True)
  criterion(y_pred, loss_labels)
  assert seen, 'the criterion built no mask with torch.where'
  assert not any(seen), 'a mask was built with the gradient enabled'


def test_loss_keys_are_published_as_a_class_attribute():
  '''The published keys are the ones the engine logs.'''
  assert SedScaledPosNegFocalLoss.loss_keys == [
      'bce_loss', 'F_loss', 'FScaled_loss', 'F_loss_0', 'F_loss_1']
  assert isinstance(SedScaledPosNegFocalLoss.loss_keys, list)


def test_criterion_returns_a_scalar_and_the_published_keys(loss_prediction,
                                                           loss_labels):
  '''The criterion returns `(scalar, dict)` and the dict has the keys.'''
  criterion = SedScaledPosNegFocalLoss()
  loss, parts = criterion(loss_prediction, loss_labels)
  assert isinstance(loss, torch.Tensor)
  assert loss.ndim == 0
  assert list(parts.keys()) == ['bce_loss', 'F_loss', 'FScaled_loss',
                                'F_loss_0', 'F_loss_1']
  for value in parts.values():
    assert isinstance(value, torch.Tensor)
    assert value.ndim == 0
