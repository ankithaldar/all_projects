#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''The focal loss with the secondary species removed.

Transcribed from ``sed_removed_focal_loss.py``.  It is the plain focal
loss of ``tf_loss_focal_standard.py`` followed by a hard removal of the
secondary species:

    FRemoved = (1 - 1[secondary]) * F

The secondary species are the weak, noisier part of the weak label, so
they receive **exactly zero** gradient here.  This is the hard version
of the down-weighting that ``tf_loss_scaled_pos_neg_focal.py`` applies
continuously.

No shipped config selects this criterion; it is dispatched and correct.
'''


import tensorflow as tf

from birdcall.losses.tf_loss_focal_standard import FocalLossStandard


@tf.keras.utils.register_keras_serializable(package='birdcall',
                                     name='SedRemovedFocalLoss')
class SedRemovedFocalLoss(FocalLossStandard):
  '''The focal loss with the secondary species zeroed out.

  Attributes:
    loss_keys: The names of the scalar components the criterion returns.
  '''

  loss_keys: list[str] = ['bce_loss', 'F_loss', 'FRemoved_loss']

  def call(self, y_true, y_pred):
    '''Scores one batch.

    Args:
      y_true: The label dict, or a plain label tensor.
      y_pred: The already sigmoided clip head.

    Returns:
      The scalar loss, with the secondary positions zeroed.
    '''
    labels, secondary, prediction = self._prepare(y_true, y_pred)
    bce, focal = self._focal_terms(prediction, labels)
    secondary_mask = tf.cast(secondary > 0.0, focal.dtype)
    removed = (1.0 - secondary_mask) * focal
    loss = tf.reduce_mean(removed)
    self.components = {
        'bce_loss': tf.reduce_mean(bce),
        'F_loss': tf.reduce_mean(focal),
        'FRemoved_loss': loss,
    }
    return loss
