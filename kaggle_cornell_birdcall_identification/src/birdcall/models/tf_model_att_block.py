#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''The attention head, as a Keras layer.

``AttBlock`` produces the two readings the competition needs from one
tensor: a clip level score, pooled over time with a learned attention,
and a frame level score, which is the same classifier read per frame.

**The softmax is over time.**  In the torch port the tensor is
``(batch, classes, frames)``, so the time axis is the last one.  Keras is
channels last, so the tensor is ``(batch, frames, classes)`` and the time
axis is ``-2``.  ``Softmax(axis=-1)`` is a silent wrong answer: it trains,
and it scores badly.  This is the single highest probability silent
failure of the whole port, so the axis is written down here and the
attention is exposed for inspection.

``bn_att`` and ``temperature`` are **dead** in the original: ``bn_att`` is
created and initialised but never called, and the ``softmax(x /
temperature)`` was never written.  They are accepted and kept, and they
are not used.
'''


import tensorflow as tf


ATT_AXIS = -2
'''The time axis of a channels last ``(batch, frames, classes)`` tensor.'''

@tf.keras.utils.register_keras_serializable(package='birdcall', name='AttBlock')
class AttBlock(tf.keras.layers.Layer):
  '''The clip level and frame level heads, sharing one classifier.

  Attributes:
    in_features: The number of input channels.
    out_features: The number of classes.
    activation: The activation of the classifier output, ``'sigmoid'`` or
      ``'linear'``.
    temperature: Dead in the original, and dead here.  The upstream
      ``softmax(x / temperature)`` was never written, so the value is
      accepted and ignored.
    att: The attention convolution, a 1x1 convolution over time.
    cla: The classifier convolution, a 1x1 convolution over time.
    bn_att: Dead in the original too: it is created and initialised but
      never used.  It is kept because the original checkpoint carries
      it, and the parameter count is part of the documented contract.
    last_attention: The attention distribution of the last call, for
      inspection.  It sums to one over the time axis.
  '''

  def __init__(self, in_features: int, out_features: int,
               activation: str = 'linear', temperature: float = 1.0,
               name: str | None = None) -> None:
    '''Builds the attention block.

    Args:
      in_features: The number of input channels, 1024 for the
        DenseNet-121 variant.
      out_features: The number of classes.
      activation: The activation applied to the classifier output,
        ``'sigmoid'`` or ``'linear'``.
      temperature: Dead in the original, and dead here.  The value is
        accepted and ignored.
      name: The layer name.

    Raises:
      ValueError: If the activation is not supported.
    '''
    super().__init__(name=name)
    if activation not in ('linear', 'sigmoid'):
      raise ValueError(f'Unknown activation: {activation!r}')
    self.in_features = in_features
    self.out_features = out_features
    self.activation = activation
    self.temperature = temperature
    self.att = tf.keras.layers.Conv1D(out_features, kernel_size=1,
                                      use_bias=True,
                             name='att')
    self.cla = tf.keras.layers.Conv1D(out_features, kernel_size=1,
                                      use_bias=True,
                             name='cla')
    # The softmax is over TIME.  The tensor is channels last, so the time
    # axis is -2 and the class axis is -1.
    self.softmax = tf.keras.layers.Softmax(axis=ATT_AXIS, name='norm_att')
    self.activation_layer = (tf.keras.layers.Activation('sigmoid',
                                                name='cla_act')
                             if activation == 'sigmoid' else None)
    # Dead in the original: never called.  Kept for the checkpoint.
    self.bn_att = tf.keras.layers.BatchNormalization(axis=-1, name='bn_att')
    self.last_attention = None

  def build(self, input_shape):
    '''Builds the sublayers of the block.

    Args:
      input_shape: The shape of the input, ``(batch, frames, channels)``.
    '''
    self.att.build(input_shape)
    self.cla.build(input_shape)
    self.softmax.build(input_shape)
    if self.activation_layer is not None:
      self.activation_layer.build(input_shape)
    self.bn_att.build(input_shape)
    super().build(input_shape)

  def attention_weights(self, x):
    '''Returns the attention distribution of a feature tensor.

    Args:
      x: A ``(batch, frames, in_features)`` feature tensor.

    Returns:
      A ``(batch, frames, out_features)`` tensor that sums to one over
      the time axis.
    '''
    return self.softmax(tf.tanh(self.att(x)))

  def call(self, x, training: bool = False):
    '''Reads a feature tensor at both granularities.

    Args:
      x: A ``(batch, frames, in_features)`` feature tensor.
      training: Unused.  The block has no stochastic part.

    Returns:
      A dict with the keys ``clipwise_output``, of shape
      ``(batch, out_features)``, ``norm_att``, of shape
      ``(batch, frames, out_features)``, and ``cla``, of the same shape
      as ``norm_att``.  ``norm_att`` sums to one over the time axis.
    '''
    del training
    norm_att = self.attention_weights(x)
    cla = self.cla(x)
    if self.activation_layer is not None:
      cla = self.activation_layer(cla)
    # The clip level score is the attention weighted sum over time.
    clipwise = tf.reduce_sum(norm_att * cla, axis=1)
    self.last_attention = norm_att
    return {
        'clipwise_output': clipwise,
        'norm_att': norm_att,
        'cla': cla,
    }

  def get_config(self) -> dict:
    '''Returns the serialisable configuration of the layer.

    Returns:
      The configuration dictionary.
    '''
    config = super().get_config()
    config.update({
        'in_features': self.in_features,
        'out_features': self.out_features,
        'activation': self.activation,
        'temperature': self.temperature,
    })
    return config
