#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''The focal loss with the secondary species removed.

Transcribed from ``sed_removed_focal_loss.py``.  It is the plain focal
loss of ``pl_loss_focal_standard.py`` followed by a hard removal of the
secondary species:

    FRemoved = (1 - 1[secondary]) * F

The secondary species are the weak, noisier part of the weak label, so
they receive **exactly zero** gradient here.  This is the hard version
of the down-weighting that ``pl_loss_scaled_pos_neg_focal.py`` applies
continuously.

No shipped config selects this criterion; it is dispatched and correct.
'''

import torch
from torch import Tensor

from birdcall.losses.pl_loss_focal_standard import FocalLossStandard


class SedRemovedFocalLoss(FocalLossStandard):
  '''The focal loss with the secondary species zeroed out.

  Attributes:
    loss_keys: The names of the scalar components the criterion returns.
  '''

  loss_keys: list[str] = ['bce_loss', 'F_loss', 'FRemoved_loss']

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
    y_pred, y_true, y_secondary = self._prepare(y_pred, y_target)
    bce, focal = self._focal_terms(y_pred, y_true)
    with torch.no_grad():
      ones = torch.ones_like(y_true)
      secondary_mask = torch.where(y_secondary > 0.0, ones,
                                   torch.zeros_like(y_true))
    removed = (1.0 - secondary_mask) * focal
    loss = removed.mean()
    return loss, {'bce_loss': bce.mean().detach(),
                  'F_loss': focal.mean().detach(),
                  'FRemoved_loss': loss.detach()}
