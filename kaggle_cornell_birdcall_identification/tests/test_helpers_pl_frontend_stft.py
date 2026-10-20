#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests for the PyTorch short-time Fourier transform.

The transform is two frozen convolutions over a DFT window, so what
matters is the frame count, the number of linear frequency bins and the
fact that the real and the imaginary parts come out separately.
'''

import pytest
import torch

from birdcall.helpers.pl_frontend_stft import STFT

SAMPLE_RATE = 32000
WINDOW_SIZE = 1024
HOP_SIZE = 320
N_SAMPLES = 2 * SAMPLE_RATE
# The centred frame count: 1 + n_samples // hop_length.
FRAMES = 1 + N_SAMPLES // HOP_SIZE
N_BINS = WINDOW_SIZE // 2 + 1


def _stft() -> STFT:
  '''Builds the transform of the shipped configuration.'''
  return STFT(n_fft=WINDOW_SIZE, hop_length=HOP_SIZE, win_length=WINDOW_SIZE,
              window='hann', center=True, pad_mode='reflect',
              freeze_parameters=True)


def test_stft_frame_count_is_one_plus_samples_over_hop():
  '''The centred STFT emits `1 + n_samples // hop_length` frames.'''
  stft = STFT(n_fft=WINDOW_SIZE, hop_length=HOP_SIZE, win_length=WINDOW_SIZE,
              window='hann', center=True, pad_mode='reflect',
              freeze_parameters=True)
  waveform = torch.randn(2, N_SAMPLES)
  real, imag = stft(waveform)
  assert real.shape == (2, N_BINS, FRAMES)
  assert imag.shape == (2, N_BINS, FRAMES)


def test_the_two_parts_are_the_real_and_the_imaginary_ones():
  '''The imaginary part of a real signal is not the same as the real one.'''
  stft = _stft()
  waveform = torch.randn(2, N_SAMPLES)
  real, imag = stft(waveform)
  assert not torch.equal(real, imag)
  assert real.shape == imag.shape


def test_the_transform_is_frozen_by_default():
  '''The convolution weights do not require a gradient.'''
  stft = _stft()
  assert all(not parameter.requires_grad
             for parameter in stft.parameters())


def test_an_unsupported_pad_mode_is_refused():
  '''An unknown pad mode is a clear error.'''
  with pytest.raises(ValueError, match='edge'):
    STFT(n_fft=WINDOW_SIZE, hop_length=HOP_SIZE, win_length=WINDOW_SIZE,
         window='hann', center=True, pad_mode='edge',
         freeze_parameters=True)


def test_an_unsupported_window_is_refused():
  '''An unknown window is a clear error.'''
  with pytest.raises(ValueError, match='blackman'):
    STFT(n_fft=WINDOW_SIZE, hop_length=HOP_SIZE, win_length=WINDOW_SIZE,
         window='blackman', center=True, pad_mode='reflect',
         freeze_parameters=True)
