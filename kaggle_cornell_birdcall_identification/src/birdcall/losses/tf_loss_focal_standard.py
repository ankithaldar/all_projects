#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''The plain focal loss, as a Keras loss.

The prediction arrives **already sigmoided** by the ``AttBlock``; a loss
must not apply a second sigmoid.  The modulating base is the BCE value
itself, not ``log p``:

    F = alpha * (1 - exp(-bce)) ** gamma * bce

so the criterion is **inert** at the shipped parameters (``gamma=0``):
it is exactly ``alpha * bce``.

The argument order is **inverted** relative to the torch port: Keras
calls ``call(y_true, y_pred)``.  The two have identical shapes and
identical dtypes, so a swapped call does not raise -- it just computes a
different number.  Everything here is therefore called with keyword
arguments, and the criterion also exposes ``compute_loss``, the durable
extension point, which takes the label dict and the model output dict.

The **reduction** is the other trap.  Keras' default,
``sum_over_batch_size``, divides by the number of *elements* of the
per-element loss, which for the shipped shapes is a factor of about
792264.  The mean is therefore taken **inside** ``call`` and the scalar
that comes out of it is passed through the reduction unchanged.
'''


import tensorflow as tf


EPSILON_FP16 = 1e-5
'''The clamp of the prediction, as in the original.'''

@tf.keras.utils.register_keras_serializable(package='birdcall',
                                     name='FocalLossStandard')
class FocalLossStandard(tf.keras.losses.Loss):
  '''The focal loss over the clip head, with no mask.

  Attributes:
    loss_keys: The names of the scalar components the criterion returns.
    gamma: The focal exponent.  Zero makes the focal term inert.
    alpha: The scale of the focal term.
    components: The scalar components of the last call, keyed by the
      names of ``loss_keys``.
  '''

  loss_keys: list[str] = ['bce_loss', 'F_loss']

  def __init__(self, gamma: float = 0.0, alpha: float = 1.0,
               name: str | None = None,
               reduction: str = 'sum_over_batch_size',
               dtype: str | None = None) -> None:
    '''Builds the criterion.

    Args:
      gamma: The focal exponent.  Zero makes the focal term inert, which
        is the value every shipped config uses.
      alpha: The scale of the focal term.
      name: The loss name.
      reduction: The Keras reduction.  The mean is taken inside the
        call, so the reduction of a scalar is a no-op.
      dtype: The dtype of the loss.
    '''
    super().__init__(name=name, reduction=reduction, dtype=dtype)
    self.gamma = gamma
    self.alpha = alpha
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

  def _focal_terms(self, prediction, labels):
    '''Computes the elementwise BCE and focal terms.

    Args:
      prediction: The clamped prediction, shape ``(rows, classes)``.
      labels: The labels, shape ``(rows, classes)``.

    Returns:
      A tuple ``(bce, focal)``, both of shape ``(rows, classes)``.
    '''
    bce = -(labels * tf.math.log(prediction) +
            (1.0 - labels) * tf.math.log(1.0 - prediction))
    pt = tf.exp(-bce)
    return bce, self.alpha * tf.pow(1.0 - pt, self.gamma) * bce

  def call(self, y_true, y_pred):
    '''Scores one batch.

    Args:
      y_true: The label dict, or a plain label tensor.
      y_pred: The already sigmoided clip head.

    Returns:
      The scalar loss.
    '''
    labels, _, prediction = self._prepare(y_true, y_pred)
    bce, focal = self._focal_terms(prediction, labels)
    loss = tf.reduce_mean(focal)
    self.components = {
        'bce_loss': tf.reduce_mean(bce),
        'F_loss': loss,
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
    config.update({'gamma': self.gamma, 'alpha': self.alpha})
    return config
