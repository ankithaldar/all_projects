#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''The live SED criterion: the scaled positive/negative focal loss.

Transcribed from ``sed_scaled_pos_neg_focal_loss.py``, the criterion that
19 of the 21 shipped configs select.  It has three independent knobs on
top of the plain BCE:

    bce      = BCE(clamp(y_pred), y_true)
    pt       = exp(-bce)
    F_loss_0 = alpha_0 * (1 - m) * (1 - pt) ** gamma * bce   (negatives)
    F_loss_1 = alpha_1 *      m  * (1 - pt) ** gamma * bce   (positives)
    F_loss   = F_loss_0 + F_loss_1
    FScaled  = s_secondary * F_loss

``m`` is 1 on the positive positions, ``s_secondary`` is
``secondary_factor`` on the secondary species and 1 everywhere else.

**The focal term is inert at the shipped parameters.**  With
``gamma=0``, ``alpha_0 = alpha_1 = 1`` and ``secondary_factor = 1`` the
expression collapses to the mean BCE, bit for bit.  That collapse is the
objective the 1st place solution was won with, so it is reproduced
deliberately and must not be "fixed".
'''

import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn

EPSILON_FP16 = 1e-5


class SedScaledPosNegFocalLoss(nn.Module):
  '''The scaled positive/negative focal loss over the clip head.

  Attributes:
    loss_keys: The names of the scalar components the criterion returns.
    gamma: The focal exponent.  Zero makes the focal term inert.
    alpha_1: The scale of the positive positions.
    alpha_0: The scale of the negative positions.
    secondary_factor: The scale of the secondary species.  One disables
      the secondary masking.
  '''

  loss_keys: list[str] = ['bce_loss', 'F_loss', 'FScaled_loss',
                           'F_loss_0', 'F_loss_1']

  def __init__(self, gamma: float = 0.0, alpha_1: float = 1.0,
               alpha_0: float = 1.0, secondary_factor: float = 0.1) -> None:
    '''Builds the criterion.

    Args:
      gamma: The focal exponent.  Zero makes the focal term inert, which
        is the value every shipped config uses.
      alpha_1: The scale of the positive positions.
      alpha_0: The scale of the negative positions.
      secondary_factor: The scale of the secondary species.  One disables
        the secondary masking.
    '''
    super().__init__()
    self.gamma = gamma
    self.alpha_1 = alpha_1
    self.alpha_0 = alpha_0
    self.secondary_factor = secondary_factor

  def _terms(self, y_pred: Tensor,
             y_target: dict) -> dict[str, Tensor]:
    '''Computes the elementwise terms of the criterion.

    The masks are built under ``torch.no_grad()``: they are constants of
    the loss, never a differentiable path.

    Args:
      y_pred: The already sigmoided clip head, shape
        ``(batch, snippets, classes)``.
      y_target: The label dict with ``all_labels`` and
        ``secondary_labels``.

    Returns:
      The elementwise ``bce``, ``F``, ``F_0``, ``F_1``, ``scaled`` and
      ``primary`` tensors, each with shape ``(batch * snippets, classes)``.
    '''
    y_true = y_target['all_labels']
    y_secondary = y_target['secondary_labels']
    batch, snippets, classes = y_true.shape
    y_pred = torch.clamp(y_pred, EPSILON_FP16, 1.0 - EPSILON_FP16)
    y_pred = y_pred.reshape(batch * snippets, classes)
    y_true = y_true.reshape(batch * snippets, classes)
    y_secondary = y_secondary.reshape(batch * snippets, classes)
    with torch.no_grad():
      ones = torch.ones_like(y_true)
      zeros = torch.zeros_like(y_true)
      all_mask = torch.where(y_true > 0.0, ones, zeros)
      secondary_mask = torch.where(y_secondary > 0.0,
                                   ones * self.secondary_factor, ones)
      primary_mask = torch.where(y_true - y_secondary > 0.0, ones, zeros)
    bce = F.binary_cross_entropy(y_pred, y_true, reduction='none')
    pt = torch.exp(-bce)
    modulating = (1.0 - pt) ** self.gamma
    f_0 = self.alpha_0 * (1.0 - all_mask) * modulating * bce
    f_1 = self.alpha_1 * all_mask * modulating * bce
    focal = f_0 + f_1
    return {'bce': bce,
            'F': focal,
            'F_0': f_0,
            'F_1': f_1,
            'scaled': secondary_mask * focal,
            'primary': primary_mask}

  def forward(self, y_pred: Tensor,
              y_target: dict) -> tuple[Tensor, dict[str, Tensor]]:
    '''Scores one batch.

    Args:
      y_pred: The already sigmoided clip head, shape
        ``(batch, snippets, classes)``.
      y_target: The label dict with ``all_labels`` and
        ``secondary_labels``.

    Returns:
      The scalar loss and the dict of scalar components, whose keys are
      exactly ``loss_keys``.
    '''
    terms = self._terms(y_pred, y_target)
    loss = terms['scaled'].mean()
    return loss, {'bce_loss': terms['bce'].mean().detach(),
                  'F_loss': terms['F'].mean().detach(),
                  'FScaled_loss': loss.detach(),
                  'F_loss_0': terms['F_0'].mean().detach(),
                  'F_loss_1': terms['F_1'].mean().detach()}
