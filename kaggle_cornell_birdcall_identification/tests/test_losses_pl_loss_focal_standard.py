#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests for the plain focal loss.

The prediction arrives **already sigmoided** by the attention block, so
the criterion must not apply a second sigmoid.  The modulating base is
the BCE value itself, so the criterion is inert at the shipped
parameters (`gamma = 0`): it is exactly `alpha * bce`.
'''

import torch
from torch import nn

from birdcall.losses.pl_loss_focal_standard import FocalLossStandard


def test_focal_loss_standard_is_bce_at_gamma_zero(loss_prediction,
                                                   loss_labels):
  '''The focal term is inert at `gamma = 0`.'''
  criterion = FocalLossStandard(gamma=0.0, alpha=1.0)
  loss, parts = criterion(loss_prediction, loss_labels)
  classes = loss_prediction.shape[-1]
  reference = nn.BCELoss()(loss_prediction.reshape(-1, classes),
                           loss_labels['all_labels'].reshape(-1, classes))
  assert torch.allclose(loss, reference, atol=1e-6)
  assert torch.allclose(parts['F_loss'], reference, atol=1e-6)


def test_loss_keys_are_published_as_a_class_attribute():
  '''The published keys are the ones the engine logs.'''
  assert FocalLossStandard.loss_keys == ['bce_loss', 'F_loss']
  assert isinstance(FocalLossStandard.loss_keys, list)


def test_criterion_returns_a_scalar_and_the_published_keys(loss_prediction,
                                                           loss_labels):
  '''The criterion returns `(scalar, dict)` and the dict has the keys.'''
  criterion = FocalLossStandard()
  loss, parts = criterion(loss_prediction, loss_labels)
  assert isinstance(loss, torch.Tensor)
  assert loss.ndim == 0
  assert list(parts.keys()) == ['bce_loss', 'F_loss']
  for value in parts.values():
    assert isinstance(value, torch.Tensor)
    assert value.ndim == 0
