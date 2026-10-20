#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''The scaled positive/negative focal loss with a primary label dropout.

Transcribed from ``sed_scaled_pos_neg_focal_loss_augd.py``, the criterion
of the two ``nomix_fold*_augd`` configs.  It is the criterion of
``tf_loss_scaled_pos_neg_focal.py`` followed by a Bernoulli dropout on
the **primary** species positions: every primary position is **dropped**
with probability ``primary_drop_out_factor`` (0.3), so 30 percent of the
primary positions contribute nothing to the loss.

Two details of the original are reproduced on purpose:

* the returned ``FScaled_loss`` in the dict is the **pre-drop** value,
  while the scalar the criterion returns is the post-drop one;
* the dropped positions still count in the denominator, so the loss is
  not renormalised.

The original drew the mask from the **global, unseeded** RNG, which made
the validation loss non-deterministic from epoch to epoch.  Here the
draw comes from a generator seeded with the constructor's ``seed``, so a
run is reproducible.
'''


import tensorflow as tf

from birdcall.losses.tf_loss_scaled_pos_neg_focal import (
    SedScaledPosNegFocalLoss)


@tf.keras.utils.register_keras_serializable(package='birdcall',
                                     name='SedScaledPosNegFocalLossAugd')
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
               primary_drop_out_factor: float = 0.3, seed: int = 1234,
               name: str | None = None,
               reduction: str = 'sum_over_batch_size',
               dtype: str | None = None) -> None:
    '''Builds the criterion.

    Args:
      gamma: The focal exponent.  Zero makes the focal term inert, which
        is the value every shipped config uses.
      alpha_1: The scale of the positive positions.
      alpha_0: The scale of the negative positions.
      secondary_factor: The scale of the secondary species.  One disables
        the secondary masking.
      primary_drop_out_factor: The probability that a primary position is
        dropped.  The default of 0.3 means that 30 percent of the primary
        positions contribute nothing.
      seed: The seed of the generator the dropout mask is drawn from.
      name: The loss name.
      reduction: The Keras reduction.  The mean is taken inside the
        call, so the reduction of a scalar is a no-op.
      dtype: The dtype of the loss.
    '''
    super().__init__(gamma=gamma, alpha_1=alpha_1, alpha_0=alpha_0,
                     secondary_factor=secondary_factor, name=name,
                     reduction=reduction, dtype=dtype)
    self.primary_drop_out_factor = primary_drop_out_factor
    self.seed = seed
    # A stateful generator, so the mask is drawn inside the graph: the
    # loss is called from a compiled `train_step`.
    self._generator = tf.random.Generator.from_seed(seed)

  def _drop_mask(self, primary):
    '''Draws the dropout mask of the primary positions.

    Args:
      primary: The primary mask, of shape ``(rows, classes)``.

    Returns:
      A boolean tensor of the same shape, ``True`` where the position is
      **dropped**.
    '''
    draw = self._generator.uniform(tf.shape(primary), dtype=tf.float32)
    keep = primary & (draw > self.primary_drop_out_factor)
    return primary & ~keep

  def call(self, y_true, y_pred):
    '''Scores one batch.

    Args:
      y_true: The label dict, or a plain label tensor.
      y_pred: The already sigmoided clip head.

    Returns:
      The scalar loss, with the primary positions dropped, and the
      components in ``self.components``.  The ``FScaled_loss`` entry is
      the **pre-drop** value.
    '''
    terms = self._terms(y_true, y_pred)
    scaled = terms['scaled']
    primary = terms['primary'] > 0.5
    dropped = self._drop_mask(primary)
    final = tf.where(dropped, tf.zeros_like(scaled), scaled)
    loss = tf.reduce_mean(final)
    self.components = {
        'bce_loss': tf.reduce_mean(terms['bce']),
        'F_loss': tf.reduce_mean(terms['F']),
        'FScaled_loss': tf.reduce_mean(scaled),
        'F_loss_0': tf.reduce_mean(terms['F_0']),
        'F_loss_1': tf.reduce_mean(terms['F_1']),
    }
    return loss

  def get_config(self) -> dict:
    '''Returns the serialisable configuration of the criterion.

    Returns:
      The configuration dictionary.
    '''
    config = super().get_config()
    config.update({
        'primary_drop_out_factor': self.primary_drop_out_factor,
        'seed': self.seed,
    })
    return config
