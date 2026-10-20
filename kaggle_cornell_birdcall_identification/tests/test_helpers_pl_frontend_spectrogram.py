#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests for the PyTorch power spectrogram.

The power, not the magnitude, is load bearing: the dB formula downstream
is `10 * log10(power)`, so feeding the magnitude is a silent factor of
two error.  The output is `(B, 1, n_frames, n_bins)`.
'''

import torch

from birdcall.helpers.pl_frontend_spectrogram import Spectrogram
from birdcall.helpers.pl_frontend_stft import STFT

SAMPLE_RATE = 32000
WINDOW_SIZE = 1024
HOP_SIZE = 320
N_SAMPLES = 2 * SAMPLE_RATE
# The centred frame count: 1 + n_samples // hop_length.
FRAMES = 1 + N_SAMPLES // HOP_SIZE
N_BINS = WINDOW_SIZE // 2 + 1


def _spectrogram() -> Spectrogram:
  '''Builds the spectrogram extractor of the shipped configuration.'''
  return Spectrogram(n_fft=WINDOW_SIZE, hop_length=HOP_SIZE,
                     win_length=WINDOW_SIZE, window='hann', center=True,
                     pad_mode='reflect', freeze_parameters=True)


def test_spectrogram_shape():
  '''The spectrogram is (B, 1, n_frames, n_bins).'''
  waveform = torch.randn(2, N_SAMPLES)
  spectrogram = _spectrogram()(waveform)
  assert spectrogram.shape == (2, 1, FRAMES, N_BINS)


def test_spectrogram_is_power_not_magnitude():
  '''The spectrogram is `real ** 2 + imag ** 2`.'''
  waveform = torch.randn(1, N_SAMPLES)
  spectrogram = _spectrogram()(waveform)
  stft = STFT(n_fft=WINDOW_SIZE, hop_length=HOP_SIZE, win_length=WINDOW_SIZE,
              window='hann', center=True, pad_mode='reflect',
              freeze_parameters=True)
  real, imag = stft(waveform)
  power = (real ** 2 + imag ** 2).transpose(1, 2)
  assert torch.allclose(spectrogram[0, 0], power, atol=1e-4)


def test_the_spectrogram_is_the_power_of_the_magnitude():
  '''The magnitude squared is the same thing, computed the other way.'''
  waveform = torch.randn(1, N_SAMPLES)
  spectrogram = _spectrogram()(waveform)
  stft = STFT(n_fft=WINDOW_SIZE, hop_length=HOP_SIZE, win_length=WINDOW_SIZE,
              window='hann', center=True, pad_mode='reflect',
              freeze_parameters=True)
  real, imag = stft(waveform)
  magnitude = torch.sqrt(real ** 2 + imag ** 2).transpose(1, 2)
  assert torch.allclose(spectrogram[0, 0], magnitude ** 2, atol=1e-4)


def test_a_non_unit_power_takes_the_root():
  '''A power of one gives the magnitude, not the magnitude squared.'''
  waveform = torch.randn(1, N_SAMPLES)
  extractor = Spectrogram(n_fft=WINDOW_SIZE, hop_length=HOP_SIZE,
                          win_length=WINDOW_SIZE, window='hann', center=True,
                          pad_mode='reflect', power=1.0,
                          freeze_parameters=True)
  squared = _spectrogram()(waveform)
  root = extractor(waveform)
  assert torch.allclose(root ** 2, squared, atol=1e-4)


def test_the_spectrogram_owns_a_frozen_stft():
  '''The embedded transform is frozen, as in the original checkpoint.'''
  extractor = _spectrogram()
  assert isinstance(extractor.stft, STFT)
  assert all(not parameter.requires_grad
             for parameter in extractor.parameters())
