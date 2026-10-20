#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''The waveform to log-mel front-end, as one Keras layer.

The original vendored a 250 line STFT implementation.  The Keras port
replaces it with five lines of ``tf.signal``, and the four details that
matter are the ones this module spells out:

* the **reflect pad is explicit** -- ``tf.signal.frame`` has none, and
  without it a 30 second waveform yields 2977 frames instead of 3001;
* the **power** is fed to the decibel formula, not the magnitude, which
  is a silent factor of two;
* the mel matrix is **librosa's** Slaney, area normalised matrix, not
  ``tf.signal``'s HTK one, which is a different matrix for the same
  ``(sr, n_fft, n_mels, fmin, fmax)``;
* there is **no** ``tf.signal.magnitude_to_db``, so the conversion is
  the one line below.

The analysis window is the **periodic Hann** window the original's
``librosa.filters.get_window('hann', win_length, fftbins=True)``
produces.  It is applied here rather than dropped, because the parity
check against the torch front-end is on the log-mel values themselves
and an unwindowed frame is a different number by several decibels.
'''


import numpy as np
import tensorflow as tf

from birdcall.helpers.mel_matrix import load_mel_matrix


AMIN = 1e-10
'''The floor of the mel power, before the logarithm.'''

REF = 1.0
'''The reference power of the decibel formula.  One gives absolute dB.'''

LOG10_SCALE = 4.342944819032518
'''Ten times one over the natural logarithm of ten, in float32.

TensorFlow has no ``tf.math.log10``; this is the constant that turns a
natural logarithm into a decibel value.
'''

def hann_window(length: int) -> np.ndarray:
  '''Builds the periodic Hann window.

  Args:
    length: The number of samples of the window.

  Returns:
    A float64 window of ``length`` samples.
  '''
  positions = np.arange(length)
  return 0.5 * (1.0 - np.cos(2.0 * np.pi * positions / length))

class LogmelFrontEnd(tf.keras.layers.Layer):
  '''The waveform to log-mel front-end.

  Attributes:
    sample_rate: The audio sample rate in Hz.
    n_fft: The FFT size, and the analysis window length.
    hop_size: The hop between frames, in samples.
    mel_bins: The number of mel bands.
    fmin: The lowest frequency of the mel bank, in Hz.
    fmax: The highest frequency of the mel bank, in Hz.
    top_db: The dynamic range ceiling of the log-mel, in dB.  ``None``
      disables it, which is what every shipped config does.
    mel_weight: The frozen mel matrix, shape
      ``(n_fft // 2 + 1, mel_bins)``.
    window: The frozen analysis window, shape ``(n_fft,)``.
  '''

  def __init__(self, sample_rate: int, n_fft: int, hop_size: int,
               mel_bins: int, fmin: int, fmax: int,
               top_db: float | None = None, name: str | None = None) -> None:
    '''Builds the front-end.

    Args:
      sample_rate: The audio sample rate in Hz.
      n_fft: The FFT size, and the analysis window length.
      hop_size: The hop between frames, in samples.
      mel_bins: The number of mel bands.
      fmin: The lowest frequency of the mel bank, in Hz.
      fmax: The highest frequency of the mel bank, in Hz.
      top_db: The dynamic range ceiling of the log-mel, in dB.  ``None``
        disables it, which is what every shipped config does.
      name: The layer name.

    Raises:
      ValueError: If ``top_db`` is negative, or the mel matrix cannot be
        loaded.
    '''
    super().__init__(name=name)
    if top_db is not None and top_db < 0:
      raise ValueError('top_db must be non-negative')
    self.sample_rate = sample_rate
    self.n_fft = n_fft
    self.hop_size = hop_size
    self.mel_bins = mel_bins
    self.fmin = fmin
    self.fmax = fmax
    self.top_db = top_db
    # The reflect pad is the original's ``center=True`` behaviour: the
    # frame count is ``1 + samples // hop`` and not ``samples // hop``.
    self.pad = n_fft // 2
    mel = load_mel_matrix(sample_rate, n_fft, mel_bins, fmin, fmax)
    # The mel matrix and the window are constants, not weights: they are
    # frozen in the original too, and a frozen buffer must not reach the
    # optimiser.
    self.mel_weight = self.add_weight(
        name='mel_weight',
        shape=mel.shape,
        initializer=tf.keras.initializers.Constant(mel),
        trainable=False)
    self.window = self.add_weight(
        name='window',
        shape=(n_fft,),
        initializer=tf.keras.initializers.Constant(hann_window(n_fft)),
        trainable=False)

  def frames_num(self, samples: int) -> int:
    '''Returns the number of frames of a waveform of the given length.

    Args:
      samples: The number of samples of the waveform.

    Returns:
      The number of frames, ``1 + samples // hop_size``.
    '''
    return 1 + samples // self.hop_size

  def call(self, inputs, training: bool = False):
    '''Turns a waveform batch into a log-mel batch.

    Args:
      inputs: A ``(batch, samples)`` waveform batch.
      training: Unused.  The front-end has no stochastic part; the
        SpecAugment of the model is the layer that needs the flag.

    Returns:
      A ``(batch, frames, mel_bins)`` log-mel batch, in decibels.

    Raises:
      ValueError: If the input is not two dimensional.
    '''
    del training
    if len(inputs.shape) != 2:
      raise ValueError(
          f'Expected a (batch, samples) waveform, got shape {inputs.shape}')
    x = tf.cast(inputs, tf.float32)
    # The reflect pad is explicit: `tf.signal.frame` has no padding of
    # its own, and without this the frame count is 2977, not 3001.
    x = tf.pad(x, [[0, 0], [self.pad, self.pad]], mode='REFLECT')
    frames = tf.signal.frame(x, frame_length=self.n_fft,
                             frame_step=self.hop_size)
    frames = frames * tf.cast(self.window, frames.dtype)
    # Power, not magnitude: the decibel formula below is 10*log10(power).
    spec = tf.abs(tf.signal.rfft(frames, fft_length=[self.n_fft])) ** 2
    mel = tf.matmul(spec, tf.cast(self.mel_weight, spec.dtype))
    logmel = tf.math.log(tf.maximum(mel, AMIN)) * tf.cast(
        LOG10_SCALE, mel.dtype)
    logmel = logmel - 10.0 * float(np.log10(max(AMIN, REF)))
    if self.top_db is not None:
      logmel = tf.maximum(logmel, tf.reduce_max(logmel) - self.top_db)
    return logmel

  def get_config(self) -> dict:
    '''Returns the serialisable configuration of the layer.

    Returns:
      The configuration dictionary.
    '''
    config = super().get_config()
    config.update({
        'sample_rate': self.sample_rate,
        'n_fft': self.n_fft,
        'hop_size': self.hop_size,
        'mel_bins': self.mel_bins,
        'fmin': self.fmin,
        'fmax': self.fmax,
        'top_db': self.top_db,
    })
    return config
