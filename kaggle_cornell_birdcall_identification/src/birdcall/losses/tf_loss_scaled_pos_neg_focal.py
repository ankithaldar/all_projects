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

The masks are built under ``tf.stop_gradient``: they are constants of
the loss, never a differentiable path.
'''


import tensorflow as tf

from birdcall.losses.tf_loss_focal_standard import EPSILON_FP16


@tf.keras.utils.register_keras_serializable(package='birdcall',
                                     name='SedScaledPosNegFocalLoss')
class SedScaledPosNegFocalLoss(tf.keras.losses.Loss):
  '''The scaled positive/negative focal loss over the clip head.

  Attributes:
    loss_keys: The names of the scalar components the criterion returns.
    gamma: The focal exponent.  Zero makes the focal term inert.
    alpha_1: The scale of the positive positions.
    alpha_0: The scale of the negative positions.
    secondary_factor: The scale of the secondary species.  One disables
      the secondary masking.
    components: The scalar components of the last call, keyed by the
      names of ``loss_keys``.
  '''

  loss_keys: list[str] = ['bce_loss', 'F_loss', 'FScaled_loss',
                           'F_loss_0', 'F_loss_1']

  def __init__(self, gamma: float = 0.0, alpha_1: float = 1.0,
               alpha_0: float = 1.0, secondary_factor: float = 0.1,
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
      name: The loss name.
      reduction: The Keras reduction.  The mean is taken inside the
        call, so the reduction of a scalar is a no-op.
      dtype: The dtype of the loss.
    '''
    super().__init__(name=name, reduction=reduction, dtype=dtype)
    self.gamma = gamma
    self.alpha_1 = alpha_1
    self.alpha_0 = alpha_0
    self.secondary_factor = secondary_factor
    self.components: dict[str, tf.Tensor] = {}

  def _prepare(self, y_true, y_pred):
    '''Flattens the labels and the prediction.

    Args:
      y_true: The label dict, with the keys ``all_labels`` and
        ``secondary_labels``, or a plain label tensor.
      y_pred: The already sigmoided clip head, shape
        ``(batch, snippets, classes)``.

    Returns:
      A tuple ``(labels, secondary labels, clamped prediction)``, each
      with shape ``(rows, classes)``.
    '''
    if isinstance(y_true, dict):
      labels = tf.convert_to_tensor(y_true['all_labels'])
      secondary = y_true.get('secondary_labels')
      if secondary is None:
        secondary = tf.zeros_like(labels)
      else:
        secondary = tf.convert_to_tensor(secondary)
    else:
      labels = tf.convert_to_tensor(y_true)
      secondary = tf.zeros_like(labels)
    classes = labels.shape[-1]
    prediction = tf.clip_by_value(tf.convert_to_tensor(y_pred), EPSILON_FP16,
                         1.0 - EPSILON_FP16)
    return (tf.reshape(labels, (-1, classes)),
            tf.reshape(secondary, (-1, classes)),
            tf.reshape(prediction, (-1, classes)))

  def _terms(self, y_true, y_pred):
    '''Computes the elementwise terms of the criterion.

    Args:
      y_true: The label dict, or a plain label tensor.
      y_pred: The already sigmoided clip head.

    Returns:
      A dict with the elementwise ``bce``, ``F``, ``F_0``, ``F_1``,
      ``scaled`` and ``primary`` tensors, each with shape
      ``(rows, classes)``.
    '''
    labels, secondary, prediction = self._prepare(y_true, y_pred)
    # The masks are constants of the loss, never a differentiable path.
    all_mask = tf.stop_gradient(tf.cast(labels > 0.0, labels.dtype))
    primary_mask = tf.stop_gradient(
        tf.cast(labels - secondary > 0.0, labels.dtype))
    secondary_mask = tf.stop_gradient(tf.where(
        secondary > 0.0,
        tf.fill(tf.shape(secondary),
                tf.cast(self.secondary_factor, secondary.dtype)),
        tf.ones_like(secondary)))
    bce = -(labels * tf.math.log(prediction) +
            (1.0 - labels) * tf.math.log(1.0 - prediction))
    pt = tf.exp(-bce)
    modulating = tf.pow(1.0 - pt, self.gamma)
    f_0 = self.alpha_0 * (1.0 - all_mask) * modulating * bce
    f_1 = self.alpha_1 * all_mask * modulating * bce
    focal = f_0 + f_1
    return {
        'bce': bce,
        'F': focal,
        'F_0': f_0,
        'F_1': f_1,
        'scaled': secondary_mask * focal,
        'primary': primary_mask,
    }

  def call(self, y_true, y_pred):
    '''Scores one batch.

    Args:
      y_true: The label dict, or a plain label tensor.
      y_pred: The already sigmoided clip head.

    Returns:
      The scalar loss.
    '''
    terms = self._terms(y_true, y_pred)
    loss = tf.reduce_mean(terms['scaled'])
    self.components = {
        'bce_loss': tf.reduce_mean(terms['bce']),
        'F_loss': tf.reduce_mean(terms['F']),
        'FScaled_loss': loss,
        'F_loss_0': tf.reduce_mean(terms['F_0']),
        'F_loss_1': tf.reduce_mean(terms['F_1']),
    }
    return loss

  def compute_loss(self, x=None, y=None, y_pred=None, sample_weight=None,
                   training: bool = True):
    '''Scores one batch from the label dict and the model output dict.

    Args:
      x: Unused.
      y: The label dict, with the keys ``all_labels`` and
        ``secondary_labels``.
      y_pred: The model output dict, or the clip head itself.
      sample_weight: Unused.
      training: Unused.

    Returns:
      The scalar loss.
    '''
    del x, sample_weight, training
    if isinstance(y_pred, dict):
      y_pred = y_pred['clipwise_output']
    return self.call(y_true=y, y_pred=y_pred)

  def get_config(self) -> dict:
    '''Returns the serialisable configuration of the criterion.

    Returns:
      The configuration dictionary.
    '''
    config = super().get_config()
    config.update({
        'gamma': self.gamma,
        'alpha_1': self.alpha_1,
        'alpha_0': self.alpha_0,
        'secondary_factor': self.secondary_factor,
    })
    return config
