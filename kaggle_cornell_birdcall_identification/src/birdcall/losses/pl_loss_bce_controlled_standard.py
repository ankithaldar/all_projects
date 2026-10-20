#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''The L-soft BCE criterion, from ``bce_controlled_standard_loss.py``.

    y_true' = (beta * y_true + (1 - beta) * y_pred) * y_true
    loss    = BCE(y_pred, y_true', reduction='none').mean(1).mean()

The prediction is **already sigmoided**; it is not re-sigmoided here.
The soft target is built under ``torch.no_grad()``: it is a target, not a
differentiable path.

``loss_keys`` is **empty** and the returned dict is empty.  That is the
criterion's real behaviour and it is not a defect: it contributes zero
extra component metrics, unlike the four focal criteria.
'''

import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn

EPSILON_FP16 = 1e-5


class BCEControlledStandardLoss(nn.Module):
  '''The L-soft binary cross entropy over the clip head.

  Attributes:
    loss_keys: The names of the scalar components the criterion returns.
      It is empty on purpose.
    beta: The softness of the label.
  '''

  loss_keys: list[str] = []

  def __init__(self, beta: float = 0.5) -> None:
    '''Builds the criterion.

    Args:
      beta: The softness of the label.  The target becomes
        ``(beta * y_true + (1 - beta) * y_pred) * y_true``.
    '''
    super().__init__()
    self.beta = beta

  def forward(self, y_pred: Tensor,
              target: dict) -> tuple[Tensor, dict[str, Tensor]]:
    '''Scores one batch.

    Args:
      y_pred: The already sigmoided clip head, shape
        ``(batch, snippets, classes)``.
      target: The label dict.  Only ``all_labels`` is read.

    Returns:
      The scalar loss and an **empty** dict of scalar components, which
      matches the empty ``loss_keys``.
    '''
    y_true = target['all_labels']
    batch, snippets, classes = y_true.shape
    y_pred = torch.clamp(y_pred, EPSILON_FP16, 1.0 - EPSILON_FP16)
    y_pred = y_pred.reshape(batch * snippets, classes)
    y_true = y_true.reshape(batch * snippets, classes)
    with torch.no_grad():
      soft = (self.beta * y_true + (1.0 - self.beta) * y_pred) * y_true
    loss = F.binary_cross_entropy(y_pred, soft, reduction='none').mean(1).mean()
    return loss, {}
