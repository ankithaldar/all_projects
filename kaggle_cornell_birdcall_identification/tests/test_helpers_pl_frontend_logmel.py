#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests for the PyTorch log-mel filterbank.

The mel matrix is the committed Slaney matrix, not a runtime librosa
call, so the reference is librosa itself.  The dB conversion is forced to
fp32: the `amin = 1e-10` floor underflows to zero in fp16 and
`log10(0)` is `-inf`, which is why the original hard-disabled AMP.
'''

import numpy as np
import pytest
import torch

from birdcall.helpers.pl_frontend_logmel import LogmelFilterBank
from birdcall.helpers.pl_frontend_spectrogram import Spectrogram

SAMPLE_RATE = 32000
WINDOW_SIZE = 1024
HOP_SIZE = 320
MEL_BINS = 64
N_SAMPLES = 2 * SAMPLE_RATE
# The centred frame count: 1 + n_samples // hop_length.
FRAMES = 1 + N_SAMPLES // HOP_SIZE
N_BINS = WINDOW_SIZE // 2 + 1


def _spectrogram() -> Spectrogram:
  '''Builds the spectrogram extractor of the shipped configuration.'''
  return Spectrogram(n_fft=WINDOW_SIZE, hop_length=HOP_SIZE,
                     win_length=WINDOW_SIZE, window='hann', center=True,
                     pad_mode='reflect', freeze_parameters=True)


def _logmel(top_db=None) -> LogmelFilterBank:
  '''Builds the log-mel filterbank of the shipped configuration.'''
  return LogmelFilterBank(sr=SAMPLE_RATE, n_fft=WINDOW_SIZE,
                          n_mels=MEL_BINS, fmin=50, fmax=14000, ref=1.0,
                          amin=1e-10, top_db=top_db, freeze_parameters=True)


def test_logmel_shape():
  '''The log-mel spectrogram is (B, 1, n_frames, n_mels).'''
  waveform = torch.randn(2, N_SAMPLES)
  logmel = _logmel()( _spectrogram()(waveform))
  assert logmel.shape == (2, 1, FRAMES, MEL_BINS)


def test_logmel_is_finite_and_fp32_under_autocast():
  '''The dB conversion is forced to fp32, so it is never `-inf`.'''
  waveform = torch.randn(2, N_SAMPLES)
  spectrogram = _spectrogram()
  logmel = _logmel()
  with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
    out = logmel(spectrogram(waveform))
  assert out.dtype == torch.float32
  assert torch.isfinite(out).all()
  assert out.shape == (2, 1, FRAMES, MEL_BINS)


def test_silence_maps_to_the_amin_floor():
  '''Digital silence is -100 dB, the `amin = 1e-10` floor.'''
  waveform = torch.zeros(1, N_SAMPLES)
  out = _logmel()(_spectrogram()(waveform))
  assert torch.allclose(out, torch.full_like(out, -100.0), atol=1e-3)


def test_logmel_matches_librosa():
  '''The ported front-end reproduces the librosa log-mel to 1e-4 dB.'''
  librosa = pytest.importorskip('librosa')
  rng = np.random.default_rng(0)
  waveform = (rng.standard_normal(N_SAMPLES) * 0.1).astype(np.float32)
  out = _logmel()(_spectrogram()(torch.from_numpy(waveform)[None]))
  reference = librosa.feature.melspectrogram(
      y=waveform, sr=SAMPLE_RATE, n_fft=WINDOW_SIZE, hop_length=HOP_SIZE,
      n_mels=MEL_BINS, fmin=50, fmax=14000, power=2.0, pad_mode='reflect')
  reference = librosa.power_to_db(reference, ref=1.0, amin=1e-10, top_db=None)
  reference = torch.from_numpy(reference.astype(np.float32)).T
  assert torch.allclose(out[0, 0], reference, atol=1e-4, rtol=0.0)


def test_a_three_dimensional_input_keeps_its_shape():
  '''A (B, n_bins, n_frames) input comes back as (B, n_frames, n_mels).'''
  waveform = torch.randn(2, N_SAMPLES)
  spectrogram = _spectrogram()(waveform)
  transposed = spectrogram[:, 0].transpose(1, 2)
  out = _logmel()(transposed)
  assert out.shape == (2, FRAMES, MEL_BINS)


def test_the_mel_matrix_is_a_frozen_parameter():
  '''The mel matrix is a frozen parameter, as in the original.

  The original held it as an ``nn.Parameter`` with
  ``requires_grad=False``, not as a buffer, and that is what the original
  checkpoint's ``logmel_extractor.melW`` key is.
  '''
  logmel = _logmel()
  parameters = dict(logmel.named_parameters())
  assert 'melW' in parameters
  assert not parameters['melW'].requires_grad
  assert logmel.melW.shape == (N_BINS, MEL_BINS)


def test_a_negative_top_db_is_refused():
  '''A negative dynamic range ceiling is a clear error.'''
  with pytest.raises(ValueError, match='top_db'):
    _logmel(top_db=-1.0)


def test_the_top_db_ceiling_clamps_the_dynamic_range():
  '''The ceiling is relative to the maximum of the batch.'''
  waveform = torch.randn(1, N_SAMPLES)
  spectrogram = _spectrogram()(waveform)
  unclamped = _logmel(top_db=None)(spectrogram)
  clamped = _logmel(top_db=20.0)(spectrogram)
  assert float(clamped.max() - clamped.min()) <= 20.0 + 1e-4
  assert float(unclamped.max() - unclamped.min()) > 20.0
