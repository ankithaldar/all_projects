#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''The scaled positive/negative focal loss with a primary label dropout.

Transcribed from ``sed_scaled_pos_neg_focal_loss_augd.py``, the criterion
of the two ``nomix_fold*_augd`` configs.  It is the criterion of
``pl_loss_scaled_pos_neg_focal.py`` followed by a Bernoulli dropout on
the **primary** species positions: every primary position is **dropped**
with probability ``primary_drop_out_factor`` (0.3), so 30 percent of the
primary positions contribute nothing to the loss.

Two details of the original are reproduced on purpose:

* the returned ``FScaled_loss`` in the dict is the **pre-drop** value,
  while the scalar the criterion returns is the post-drop one;
* the dropped positions still count in the denominator, so the loss is
  not renormalised.

The original drew the mask from the **global, unseeded** RNG, which made
the validation loss non-deterministic from epoch to epoch.  Here the draw
comes from a generator seeded with the constructor's ``seed``, so a run
is reproducible.

The polarity is the original's, and it is worth spelling out because the
documentation had it the other way round.  The original builds a mask that
is **1** on a primary position with probability ``primary_drop_out_factor``
and **0** otherwise, and then multiplies by ``(1 - mask)``.  A mask of 1
therefore zeroes the position.  So 30 percent of the primary positions are
dropped and 70 percent are kept -- not the other way round.
'''

import torch
from torch import Tensor

from birdcall.helpers.seed import make_generator
from birdcall.losses.pl_loss_scaled_pos_neg_focal import (
    SedScaledPosNegFocalLoss,
)


class SedScaledPosNegFocalLossAugd(SedScaledPosNegFocalLoss):
  '''The scaled focal loss with a Bernoulli dropout on the primary labels.

  Attributes:
    loss_keys: The names of the scalar components the criterion returns.
    primary_drop_out_factor: The probability that a primary position is
      dropped.  The default of 0.3 means that 30 percent of the primary
      positions contribute nothing.
    seed: The seed of the generator the dropout mask is drawn from.
  '''

  loss_keys: list[str] = ['bce_loss', 'F_loss', 'FScaled_loss',
                           'F_loss_0', 'F_loss_1']

  def __init__(self, gamma: float = 0.0, alpha_1: float = 1.0,
               alpha_0: float = 1.0, secondary_factor: float = 0.1,
               primary_drop_out_factor: float = 0.3,
               seed: int = 1234) -> None:
    '''Builds the criterion.

    Args:
      gamma: The focal exponent.  Zero makes the focal term inert, which
        is the value every shipped config uses.
      alpha_1: The scale of the positive positions.
      alpha_0: The scale of the negative positions.
      secondary_factor: The scale of the secondary species.  One disables
        the secondary masking.
      primary_drop_out_factor: The probability that a primary position
        is dropped.  The default of 0.3 means that 30 percent of the
        primary positions contribute nothing.
      seed: The seed of the generator the dropout mask is drawn from.
    '''
    super().__init__(gamma, alpha_1, alpha_0, secondary_factor)
    self.primary_drop_out_factor = primary_drop_out_factor
    self.seed = seed
    self._generator = make_generator(seed)

  def forward(self, y_pred: Tensor,
              y_target: dict) -> tuple[Tensor, dict[str, Tensor]]:
    '''Scores one batch.

    Args:
      y_pred: The already sigmoided clip head, shape
        ``(batch, snippets, classes)``.
      y_target: The label dict with ``all_labels`` and
        ``secondary_labels``.

    Returns:
      The scalar loss, with the primary positions dropped, and the dict of
      scalar components, whose keys are exactly ``loss_keys``.  The
      ``FScaled_loss`` entry is the **pre-drop** value.
    '''
    terms = self._terms(y_pred, y_target)
    scaled = terms['scaled']
    with torch.no_grad():
      primary = terms['primary'] > 0.5
      draw = torch.as_tensor(
          self._generator.random(tuple(primary.shape)),
          device=primary.device)
      keep = primary & (draw > self.primary_drop_out_factor)
      dropped = primary & ~keep
      final = torch.where(dropped, torch.zeros_like(scaled), scaled)
    loss = final.mean()
    return loss, {'bce_loss': terms['bce'].mean().detach(),
                  'F_loss': terms['F'].mean().detach(),
                  'FScaled_loss': scaled.mean().detach(),
                  'F_loss_0': terms['F_0'].mean().detach(),
                  'F_loss_1': terms['F_1'].mean().detach()}
