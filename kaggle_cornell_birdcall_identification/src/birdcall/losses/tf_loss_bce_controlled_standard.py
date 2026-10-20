#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''The L-soft BCE criterion, as a Keras loss.

Transcribed from ``bce_controlled_standard_loss.py``:

    y_true' = (beta * y_true + (1 - beta) * y_pred) * y_true
    loss    = BCE(y_pred, y_true', reduction='none').mean(1).mean()

The prediction is **already sigmoided**; it is not re-sigmoided here.
The soft target is built under ``tf.stop_gradient``: it is a target, not
a differentiable path.

``loss_keys`` is **empty** and the components dict is empty.  That is the
criterion's real behaviour and it is not a defect: it contributes zero
extra component metrics, unlike the four focal criteria.
'''


import tensorflow as tf

from birdcall.losses.tf_loss_focal_standard import EPSILON_FP16


@tf.keras.utils.register_keras_serializable(package='birdcall',
                                     name='BCEControlledStandardLoss')
class BCEControlledStandardLoss(tf.keras.losses.Loss):
  '''The L-soft binary cross entropy over the clip head.

  Attributes:
    loss_keys: The names of the scalar components the criterion returns.
      It is empty on purpose.
    beta: The softness of the label.
    components: The scalar components of the last call.  It is empty on
      purpose.
  '''

  loss_keys: list[str] = []

  def __init__(self, beta: float = 0.5, name: str | None = None,
               reduction: str = 'sum_over_batch_size',
               dtype: str | None = None) -> None:
    '''Builds the criterion.

    Args:
      beta: The softness of the label.  The target becomes
        ``(beta * y_true + (1 - beta) * y_pred) * y_true``.
      name: The loss name.
      reduction: The Keras reduction.  The mean is taken inside the
        call, so the reduction of a scalar is a no-op.
      dtype: The dtype of the loss.
    '''
    super().__init__(name=name, reduction=reduction, dtype=dtype)
    self.beta = beta
    self.components: dict[str, tf.Tensor] = {}

  def call(self, y_true, y_pred):
    '''Scores one batch.

    Args:
      y_true: The label dict, or a plain label tensor.
      y_pred: The already sigmoided clip head.

    Returns:
      The scalar loss.
    '''
    if isinstance(y_true, dict):
      labels = tf.convert_to_tensor(y_true['all_labels'])
    else:
      labels = tf.convert_to_tensor(y_true)
    classes = labels.shape[-1]
    prediction = tf.clip_by_value(tf.convert_to_tensor(y_pred), EPSILON_FP16,
                         1.0 - EPSILON_FP16)
    flat_pred = tf.reshape(prediction, (-1, classes))
    flat_true = tf.reshape(labels, (-1, classes))
    soft = tf.stop_gradient((self.beta * flat_true +
                             (1.0 - self.beta) * flat_pred) * flat_true)
    bce = -(soft * tf.math.log(flat_pred) +
            (1.0 - soft) * tf.math.log(1.0 - flat_pred))
    loss = tf.reduce_mean(bce)
    self.components = {}
    return loss

  def compute_loss(self, x=None, y=None, y_pred=None, sample_weight=None,
                   training: bool = True):
    '''Scores one batch from the label dict and the model output dict.

    Args:
      x: Unused.
      y: The label dict, with the key ``all_labels``.
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
    config.update({'beta': self.beta})
    return config
