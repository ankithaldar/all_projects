#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''The plain focal loss, transcribed from ``focal_loss_standard.py``.

The prediction arrives **already sigmoided** by the ``AttBlock``; a loss
must not apply a second sigmoid.  The modulating base is the BCE value
itself, not ``log p``:

    F = alpha * (1 - exp(-bce)) ** gamma * bce

so the criterion is **inert** at the shipped parameters (``gamma=0``):
it is exactly ``alpha * bce``.

Every criterion in this package returns ``(scalar, dict)`` and publishes
``loss_keys``.  The dict is instrumentation, not decoration: the engine
turns each key into a logged metric.
'''

import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn

EPSILON_FP16 = 1e-5


class FocalLossStandard(nn.Module):
  '''The focal loss over the clip head, with no mask.

  Attributes:
    loss_keys: The names of the scalar components the criterion returns.
    gamma: The focal exponent.  Zero makes the focal term inert.
    alpha: The scale of the focal term.
  '''

  loss_keys: list[str] = ['bce_loss', 'F_loss']

  def __init__(self, gamma: float = 0.0, alpha: float = 1.0) -> None:
    '''Builds the criterion.

    Args:
      gamma: The focal exponent.  Zero makes the focal term inert, which
        is the value every shipped config uses.
      alpha: The scale of the focal term.
    '''
    super().__init__()
    self.gamma = gamma
    self.alpha = alpha

  def _prepare(self, y_pred: Tensor,
               y_target: dict) -> tuple[Tensor, Tensor, Tensor]:
    '''Clamps the prediction and flattens the snippet axis.

    Args:
      y_pred: The already sigmoided clip head, shape
        ``(batch, snippets, classes)``.
      y_target: The label dict with ``all_labels`` and
        ``secondary_labels``.

    Returns:
      The clamped prediction, the labels and the secondary labels, each
      with shape ``(batch * snippets, classes)``.
    '''
    y_true = y_target['all_labels']
    y_secondary = y_target['secondary_labels']
    batch, snippets, classes = y_true.shape
    y_pred = torch.clamp(y_pred, EPSILON_FP16, 1.0 - EPSILON_FP16)
    return (y_pred.reshape(batch * snippets, classes),
            y_true.reshape(batch * snippets, classes),
            y_secondary.reshape(batch * snippets, classes))

  def _focal_terms(self, y_pred: Tensor,
                   y_true: Tensor) -> tuple[Tensor, Tensor]:
    '''Computes the elementwise BCE and focal terms.

    Args:
      y_pred: The clamped prediction, shape ``(rows, classes)``.
      y_true: The labels, shape ``(rows, classes)``.

    Returns:
      The elementwise BCE and the elementwise focal term.
    '''
    bce = F.binary_cross_entropy(y_pred, y_true, reduction='none')
    pt = torch.exp(-bce)
    return bce, self.alpha * (1.0 - pt) ** self.gamma * bce

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
    y_pred, y_true, _ = self._prepare(y_pred, y_target)
    bce, focal = self._focal_terms(y_pred, y_true)
    loss = focal.mean()
    return loss, {'bce_loss': bce.mean().detach(),
                  'F_loss': loss.detach()}
