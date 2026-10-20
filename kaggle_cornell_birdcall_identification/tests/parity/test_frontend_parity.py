#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''The cross-backend parity of the front-end.

The single check that covers the mel scale, the decibel formula, the
reflect pad and the power-versus-magnitude question at once.
'''

import numpy as np
import pytest

from birdcall.helpers.tf_frontend_logmel import LogmelFrontEnd

SR = 32000
N_FFT = 1024
HOP = 320
MELS = 64
FMIN = 50
FMAX = 14000


def waveform(samples: int = 32000, seed: int = 17) -> np.ndarray:
  '''Builds a deterministic waveform.

  Args:
    samples: The number of samples.
    seed: The seed of the generator.

  Returns:
    A float32 waveform.
  '''
  rng = np.random.default_rng(seed)
  return (0.3 * rng.standard_normal(samples)).astype(np.float32)


def torch_logmel(audio: np.ndarray) -> np.ndarray:
  '''Computes the log-mel with the torch front-end.

  Args:
    audio: The waveform.

  Returns:
    A ``(frames, mels)`` array.
  '''
  torch = pytest.importorskip('torch')
  from birdcall.helpers.pl_frontend_logmel import LogmelFilterBank
  from birdcall.helpers.pl_frontend_spectrogram import Spectrogram

  spectrogram = Spectrogram(n_fft=N_FFT, hop_length=HOP, win_length=N_FFT,
                            window='hann', center=True, pad_mode='reflect',
                            freeze_parameters=True)
  bank = LogmelFilterBank(sr=SR, n_fft=N_FFT, n_mels=MELS, fmin=FMIN,
                           fmax=FMAX, ref=1.0, amin=1e-10, top_db=None)
  with torch.no_grad():
    out = bank(spectrogram(torch.from_numpy(audio)[None, :]))
  return out[0, 0].numpy()


@pytest.mark.parity
def test_the_two_front_ends_agree_to_1e_4() -> None:
  '''The log-mel of the same waveform is the same number.'''
  audio = waveform()
  expected = torch_logmel(audio)
  got = LogmelFrontEnd(SR, N_FFT, HOP, MELS, FMIN, FMAX)(audio[None, :])
  got = got.numpy()[0]
  assert got.shape == expected.shape
  assert np.max(np.abs(got - expected)) < 1e-4


@pytest.mark.parity
def test_the_two_front_ends_agree_on_a_long_waveform() -> None:
  '''A 30 second waveform agrees as well.'''
  audio = waveform(SR * 30)
  expected = torch_logmel(audio)
  got = LogmelFrontEnd(SR, N_FFT, HOP, MELS, FMIN, FMAX)(audio[None, :])
  got = got.numpy()[0]
  assert got.shape == expected.shape == (3001, MELS)
  assert np.max(np.abs(got - expected)) < 1e-4


@pytest.mark.parity
def test_the_two_front_ends_agree_on_the_frame_count() -> None:
  '''Both front-ends produce 3001 frames of a 30 second waveform.'''
  audio = waveform(SR * 30)
  assert torch_logmel(audio).shape[0] == 3001
  front = LogmelFrontEnd(SR, N_FFT, HOP, MELS, FMIN, FMAX)
  assert front.frames_num(SR * 30) == 3001
