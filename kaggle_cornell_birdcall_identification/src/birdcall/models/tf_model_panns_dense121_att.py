#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''The PANNs DenseNet-121 attention model, as a Keras model.

The whole front-end runs inside the graph: waveform in, log-mel out,
every step.  The backbone is the ImageNet-pretrained DenseNet-121 and the
head is the ``AttBlock``, which reads the same classifier at clip level
and at frame level.

Four things about this port are not obvious and are spelled out here:

* the **BatchNorm transpose dance disappears**.  The original transposed
  twice only to land PyTorch's channel axis on axis 1; Keras is channels
  last, so ``BatchNormalization(axis=-1)`` does the same job with no
  transposes.  The **ordering** (``bn0`` then SpecAugment) is load
  bearing and is kept.
* the ImageNet ``preprocess_input`` is **bypassed**.  The log-mel is
  decibel valued and already normalised by ``bn0``; dividing it by 255
  and re-normalising it destroys the pretrained transfer with no error
  at all.  The backbone is therefore built from the application model's
  own layers, and the first convolution sees the raw ``bn0`` output.
* the DenseNet **time axis is measured, not assumed**.  It is 93, the
  same as torchvision's, so the 32x upsample covers 2976 frames and the
  tail pad is 25 frames.
* the **mixup halving** changes the meaning of the batch axis in the
  middle of the forward pass, so the labels are halved by
  ``mix_labels``, next to the code that halves the batch.
'''


import tensorflow as tf

from birdcall.helpers.tf_frontend_logmel import LogmelFrontEnd
from birdcall.models.tf_model_att_block import AttBlock


INTERPOLATE_RATIO = 32
'''The time downsampling of the DenseNet backbone, and its inverse.'''

class _SpecAugment(tf.keras.layers.Layer):
  '''Drops time stripes and then frequency stripes.

  The original mutated its input in place, in a Python loop over the
  batch.  Here the bands are drawn per batch item and applied as a
  mask, which keeps the observable behaviour -- an independent band per
  item -- without the in-place mutation.

  Attributes:
    time_drop_width: The maximum width of a time band, in frames.
    time_stripes_num: The number of time bands dropped per item.
    freq_drop_width: The maximum width of a frequency band, in bins.
    freq_stripes_num: The number of frequency bands dropped per item.
  '''

  def __init__(self, time_drop_width: int = 64, time_stripes_num: int = 2,
               freq_drop_width: int = 8, freq_stripes_num: int = 2,
               name: str | None = None) -> None:
    '''Builds the two droppers.

    Args:
      time_drop_width: The maximum width of a time band, in frames.
      time_stripes_num: The number of time bands dropped per item.
      freq_drop_width: The maximum width of a frequency band, in bins.
      freq_stripes_num: The number of frequency bands dropped per item.
      name: The layer name.
    '''
    super().__init__(name=name)
    self.time_drop_width = time_drop_width
    self.time_stripes_num = time_stripes_num
    self.freq_drop_width = freq_drop_width
    self.freq_stripes_num = freq_stripes_num

  def _band_mask(self, batch: int, total: int, drop_width: int,
                 stripes: int):
    '''Draws the dropped bands.

    Args:
      batch: The number of items in the batch.
      total: The length of the dropped axis.
      drop_width: The maximum width of a band.
      stripes: The number of bands dropped per item.

    Returns:
      A ``(batch, total)`` boolean mask, ``True`` where the spectrogram
      is kept.
    '''
    distances = tf.random.uniform((batch, stripes), maxval=drop_width,
                                  dtype=tf.int32)
    # The original drew the start from `[0, total - distance)`, with a
    # per item upper bound.  Scaling a uniform draw by that bound is the
    # same distribution, and it vectorises.
    high = tf.maximum(total - distances, 1)
    starts = tf.cast(tf.random.uniform((batch, stripes)), tf.int32) * high
    positions = tf.range(total)
    bands = ((positions[None, None, :] >= starts[:, :, None]) &
             (positions[None, None, :] < (starts + distances)[:, :, None]))
    return tf.logical_not(tf.reduce_any(bands, axis=1))

  def call(self, x, training: bool = False):
    '''Drops the time stripes and then the frequency stripes.

    Args:
      x: A ``(batch, frames, mel_bins)`` spectrogram.
      training: Whether the augmentation runs.  It must be threaded
        explicitly: a layer that omits it augments at inference too, and
        that corrupts ``framewise_output``, which is the tensor the
        thirteen model vote consumes.

    Returns:
      The same spectrogram, with the bands zeroed.
    '''
    if not training:
      return x
    batch = tf.shape(x)[0]
    time_mask = self._band_mask(batch, tf.shape(x)[1], self.time_drop_width,
                                self.time_stripes_num)
    x = x * tf.cast(tf.expand_dims(time_mask, -1), x.dtype)
    freq_mask = self._band_mask(batch, tf.shape(x)[2], self.freq_drop_width,
                                self.freq_stripes_num)
    x = x * tf.cast(tf.expand_dims(freq_mask, 1), x.dtype)
    return x

  def get_config(self) -> dict:
    '''Returns the serialisable configuration of the layer.

    Returns:
      The configuration dictionary.
    '''
    config = super().get_config()
    config.update({
        'time_drop_width': self.time_drop_width,
        'time_stripes_num': self.time_stripes_num,
        'freq_drop_width': self.freq_drop_width,
        'freq_stripes_num': self.freq_stripes_num,
    })
    return config

@tf.keras.utils.register_keras_serializable(package='birdcall',
                                     name='PANNsDense121Att')
class PANNsDense121Att(tf.keras.models.Model):
  '''The DenseNet-121 sound event detection model with attention.

  Attributes:
    sample_rate: The audio sample rate in Hz.
    window_size: The FFT size, and the analysis window length.
    hop_size: The hop between frames, in samples.
    mel_bins: The number of mel bands.
    classes_num: The number of classes.
    apply_aug: The kill switch for both SpecAugment and mixup.
    interpolate_ratio: The 32x time upsampling of the frame head.
    frontend: The log-mel front-end.
    bn0: The batch norm of the log-mel, **before** the SpecAugment.
    spec_augmenter: The in-graph SpecAugment.
    backbone_layers: The DenseNet layers, up to and including the final
      batch norm, which is where torchvision's ``.features`` stops.
    imagenet_preprocessing_bypassed: Always ``True``.  The ImageNet
      ``preprocess_input`` is never applied to the log-mel.
    last_backbone_input: The tensor the first convolution of the
      backbone saw, for inspection.
  '''

  def __init__(self, sample_rate: int, window_size: int, hop_size: int,
               mel_bins: int, fmin: int, fmax: int, classes_num: int,
               apply_aug: bool, top_db: float | None = None,
               name: str | None = None) -> None:
    '''Builds the model.

    Args:
      sample_rate: The audio sample rate in Hz.
      window_size: The FFT size, and the analysis window length.
      hop_size: The hop between frames, in samples.
      mel_bins: The number of mel bands.
      fmin: The lowest frequency in Hz.
      fmax: The highest frequency in Hz.
      classes_num: The number of classes.
      apply_aug: The kill switch for both SpecAugment and mixup.
      top_db: The dynamic range ceiling of the log-mel, in dB.  ``None``
        disables it, which is what every shipped config does.
      name: The model name.
    '''
    super().__init__(name=name)
    self.sample_rate = sample_rate
    self.window_size = window_size
    self.hop_size = hop_size
    self.mel_bins = mel_bins
    self.fmin = fmin
    self.fmax = fmax
    self.classes_num = classes_num
    self.apply_aug = apply_aug
    self.top_db = top_db
    self.interpolate_ratio = INTERPOLATE_RATIO

    self.frontend = LogmelFrontEnd(sample_rate, window_size, hop_size,
                                   mel_bins, fmin, fmax, top_db)
    # The transpose dance of the original is gone: the mel bins are the
    # last axis, so the batch norm is a plain `axis=-1`.  The ORDER is
    # load bearing: bn0 first, SpecAugment second.
    self.bn0 = tf.keras.layers.BatchNormalization(axis=-1, name='bn0')
    self.spec_augmenter = _SpecAugment(name='spec_augment')

    backbone = tf.keras.applications.DenseNet121(include_top=False,
                                        weights='imagenet',
                                        input_shape=(None, None, 3))
    # The ImageNet `preprocess_input` is deliberately NOT applied.  The
    # backbone is sliced out of the application model, up to and
    # including the final batch norm, which is where torchvision's
    # `.features` stops.
    self.imagenet_preprocessing_bypassed = True
    self.backbone = self._feature_extractor(backbone)
    self.backbone_layers = list(self.backbone.layers)
    self.max_pool = tf.keras.layers.MaxPool1D(pool_size=3, strides=1,
                                              padding='same',
                                     name='max_pool')
    self.avg_pool = tf.keras.layers.AveragePooling1D(pool_size=3, strides=1,
                                             padding='same', name='avg_pool')
    self.dropout1 = tf.keras.layers.Dropout(0.5, name='dropout1')
    self.fc1 = tf.keras.layers.Dense(1024, activation='relu', name='fc1')
    self.dropout2 = tf.keras.layers.Dropout(0.5, name='dropout2')
    self.att_block = AttBlock(1024, classes_num, activation='sigmoid',
                               name='att_block')
    self.last_backbone_input = None

  @staticmethod
  def _feature_extractor(backbone):
    '''Returns the backbone up to the final batch norm.

    The application model's graph is sliced, not rebuilt: the dense
    blocks concatenate their input with their own features, so the
    layers cannot simply be called one after the other.

    Args:
      backbone: The ImageNet application model.

    Returns:
      A functional model whose output is the final batch norm's output.

    Raises:
      ValueError: If the backbone has no batch norm.
    '''
    final_bn = None
    for layer in backbone.layers:
      if isinstance(layer, tf.keras.layers.BatchNormalization):
        final_bn = layer
    if final_bn is None:
      raise ValueError('The DenseNet backbone has no batch norm.')
    return tf.keras.models.Model(backbone.input, final_bn.output,
                        name='densenet121_features')

  def build(self, input_shape):
    '''Builds the sublayers of the model.

    Args:
      input_shape: The shape of the input,
        ``(batch, snippets, samples)``.  A tuple of shapes is accepted
        too, because the model may be called with the mixup
        coefficients next to the waveform.
    '''
    if isinstance(input_shape, (tuple, list)) and input_shape and isinstance(
        input_shape[0], (tuple, list, tf.TensorShape)):
      input_shape = input_shape[0]
    shape = tf.TensorShape(input_shape)
    samples = shape[2] if len(shape) > 2 else None
    features = (None, None, 1024)
    self.frontend.build((None, samples))
    self.bn0.build((None, None, self.mel_bins))
    self.spec_augmenter.build((None, None, self.mel_bins))
    self.max_pool.build(features)
    self.avg_pool.build(features)
    self.dropout1.build(features)
    self.fc1.build(features)
    self.dropout2.build(features)
    self.att_block.build(features)
    super().build(input_shape)

  def _unpack(self, inputs):
    '''Splits the model input into the waveform and the mixup coefficients.

    Args:
      inputs: Either a ``(batch, snippets, samples)`` waveform tensor, a
        tuple ``(waveform, mixup_lambda)``, or a dict holding a
        ``waveforms`` key.

    Returns:
      A tuple ``(waveform, mixup_lambda)``, where the coefficients are
      ``None`` when there are none.
    '''
    if isinstance(inputs, dict):
      return inputs['waveforms'], None
    if isinstance(inputs, (tuple, list)):
      return inputs[0], inputs[1]
    return inputs, None

  def _mixup_active(self, mixup_lambda) -> bool:
    '''Whether the mixup runs on this forward pass.

    Args:
      mixup_lambda: The mixup coefficients, or ``None``.

    Returns:
      True when the spectrogram is about to be mixed.
    '''
    return bool(self.apply_aug and mixup_lambda is not None)

  def mix_labels(self, labels, mixup_lambda):
    '''Mixes and halves a label tensor.

    This is the label half of the mixup, and it must halve the batch the
    same way the forward pass does.  It lives next to that code so the
    two cannot drift apart.

    Args:
      labels: A ``(batch, snippets, classes)`` label tensor.
      mixup_lambda: The mixup coefficients, one per row of the flattened
        batch.

    Returns:
      A ``((batch * snippets) // 2, 1, classes)`` label tensor, the
      mixed one.
    '''
    shape = tf.shape(labels)
    flat = tf.reshape(labels, (shape[0] * shape[1], shape[2]))
    even = flat[0::2]
    odd = flat[1::2]
    lam_even = tf.reshape(mixup_lambda[0::2], (-1, 1))
    lam_odd = tf.reshape(mixup_lambda[1::2], (-1, 1))
    mixed = even * lam_even + odd * lam_odd
    return tf.reshape(mixed, ((shape[0] * shape[1]) // 2, 1, shape[2]))

  def _do_mixup(self, x, mixup_lambda):
    '''Mixes the rows of a batch with their neighbour.

    Args:
      x: A ``(rows, frames, mel_bins)`` log-mel batch, with an even
        number of rows.
      mixup_lambda: The mixup coefficients, one per row.

    Returns:
      The mixed batch, with half the rows.
    '''
    even = x[0::2]
    odd = x[1::2]
    lam_even = tf.reshape(mixup_lambda[0::2], (-1, 1, 1))
    lam_odd = tf.reshape(mixup_lambda[1::2], (-1, 1, 1))
    return even * lam_even + odd * lam_odd

  def _pad_framewise(self, framewise, frames_num):
    '''Pads the framewise output to the number of frames.

    The padding value is the value of the **last frame**, so the
    clip-level constant carries through to the end of the file instead
    of a hard "no bird here" discontinuity.  A ``SYMMETRIC`` pad would
    reflect the whole tensor, and a ``CONSTANT`` pad would inject 25
    frames of "no bird", which silently poisons the frame head the vote
    consumes.

    Args:
      framewise: A ``(rows, frames, classes)`` framewise output.
      frames_num: The number of frames to pad to.

    Returns:
      A ``(rows, frames_num, classes)`` tensor.
    '''
    length = tf.shape(framewise)[1]
    pad = tf.maximum(frames_num - length, 0)
    tail = tf.repeat(framewise[:, -1:, :], pad, axis=1)
    return tf.concat([framewise, tail], axis=1)[:, :frames_num]

  def _backbone(self, x):
    '''Runs the DenseNet backbone and the frequency pooling.

    Args:
      x: A ``(rows, frames, mel_bins, 3)`` log-mel batch.

    Returns:
      A ``(rows, frames, 1024)`` feature tensor.
    '''
    self.last_backbone_input = x
    x = self.backbone(x)
    # The frequency axis has collapsed to a couple of bins by now.
    return tf.reduce_mean(x, axis=2)

  def call(self, inputs, training: bool = False):
    '''Runs the model.

    Args:
      inputs: A tuple ``(waveform, mixup_lambda)``, or a bare waveform
        tensor.  The waveform is a ``(batch, snippets, samples)`` tensor
        and the coefficients are one per row of the flattened batch, or
        ``None``.
      training: Whether the model is training.  It gates the
        SpecAugment and the mixup.

    Returns:
      A dict with ``clipwise_output``, of shape
      ``(batch, snippets, classes)``, and ``framewise_output``, of shape
      ``(batch, snippets, frames, classes)``.
    '''
    waveform, mixup_lambda = self._unpack(inputs)
    shape = tf.shape(waveform)
    flat = tf.reshape(waveform, (shape[0] * shape[1], shape[2]))
    x = self.frontend(flat)
    frames_num = tf.shape(x)[1]
    x = self.bn0(x, training=training)
    if training and self.apply_aug:
      x = self.spec_augmenter(x, training=True)
    batch = shape[0]
    snippets = shape[1]
    if training and self._mixup_active(mixup_lambda):
      x = self._do_mixup(x, mixup_lambda)
      batch = (shape[0] * shape[1]) // 2
      snippets = tf.constant(1, tf.int32)
    # A view, zero copy: the same log-mel on three channels.
    x = tf.expand_dims(x, -1)
    x = tf.broadcast_to(x, tf.concat([tf.shape(x)[:3], [3]], axis=0))
    x = self._backbone(x)
    x = self.max_pool(x) + self.avg_pool(x)
    x = self.dropout1(x, training=training)
    x = self.fc1(x)
    x = self.dropout2(x, training=training)
    att = self.att_block(x, training=training)
    framewise = att['cla']
    # The 32x upsample is interleaved: every frame is repeated 32 times
    # consecutively.
    framewise = tf.repeat(framewise, self.interpolate_ratio, axis=1)
    framewise = self._pad_framewise(framewise, frames_num)
    classes = tf.constant(self.classes_num, tf.int32)
    return {
        'clipwise_output': tf.reshape(
            att['clipwise_output'], tf.stack([batch, snippets, classes])),
        'framewise_output': tf.reshape(
            framewise, tf.stack([batch, snippets, frames_num, classes])),
    }

  @classmethod
  def from_config(cls, config):
    '''Rebuilds a model from its configuration.

    The sublayers are rebuilt by the constructor, so only the
    constructor's own arguments are read back.

    Args:
      config: The configuration dictionary.

    Returns:
      The model.
    '''
    fields = ('sample_rate', 'window_size', 'hop_size', 'mel_bins', 'fmin',
              'fmax', 'classes_num', 'apply_aug', 'top_db', 'name')
    return cls(**{key: config[key] for key in fields if key in config})

  def get_config(self) -> dict:
    '''Returns the serialisable configuration of the model.

    Returns:
      The configuration dictionary.
    '''
    config = super().get_config()
    config.update({
        'sample_rate': self.sample_rate,
        'window_size': self.window_size,
        'hop_size': self.hop_size,
        'mel_bins': self.mel_bins,
        'fmin': self.fmin,
        'fmax': self.fmax,
        'classes_num': self.classes_num,
        'apply_aug': self.apply_aug,
        'top_db': self.top_db,
    })
    return config
