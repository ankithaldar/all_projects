#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''The tests of the Keras log-mel front-end.

The front-end is the one part of the port where a near miss changes
every number downstream, so it is tested against an independent numpy
implementation of the same pipeline.
'''

import numpy as np
import pytest
import tensorflow as tf

from birdcall.helpers.mel_matrix import load_mel_matrix
from birdcall.helpers.tf_frontend_logmel import LogmelFrontEnd

SR = 32000
N_FFT = 1024
HOP = 320
MELS = 64
FMIN = 50
FMAX = 14000


def waveform(samples: int = 32000, seed: int = 11) -> np.ndarray:
  '''Builds a deterministic test waveform.

  Args:
    samples: The number of samples.
    seed: The seed of the generator.

  Returns:
    A float32 waveform.
  '''
  rng = np.random.default_rng(seed)
  return (0.3 * rng.standard_normal(samples)).astype(np.float32)


def reference_logmel(audio: np.ndarray) -> np.ndarray:
  '''Computes the log-mel of a waveform, independently of the project.

  The pipeline is written out with numpy: the reflect pad, the periodic
  Hann window, the power spectrogram, the Slaney mel projection and the
  decibel conversion.  It is deliberately not the project's code.

  Args:
    audio: The waveform.

  Returns:
    A ``(frames, mels)`` log-mel array.
  '''
  librosa = pytest.importorskip('librosa')
  padded = np.pad(audio, (N_FFT // 2, N_FFT // 2), mode='reflect')
  window = np.hanning(N_FFT + 1)[:-1]
  frames = np.lib.stride_tricks.sliding_window_view(padded, N_FFT)[::HOP]
  spec = np.abs(np.fft.rfft(frames * window, n=N_FFT)) ** 2
  mel = librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=MELS, fmin=FMIN,
                             fmax=FMAX)
  power = spec @ mel.T
  return 10.0 * np.log10(np.maximum(power, 1e-10))


def front_end(**kwargs) -> LogmelFrontEnd:
  '''Builds the front-end of the shipped configuration.

  Args:
    **kwargs: The overrides of the constructor.

  Returns:
    The front-end.
  '''
  params = {'sample_rate': SR, 'n_fft': N_FFT, 'hop_size': HOP,
            'mel_bins': MELS, 'fmin': FMIN, 'fmax': FMAX}
  params.update(kwargs)
  return LogmelFrontEnd(**params)


def test_output_shape_is_frames_by_mels() -> None:
  '''The output is a (batch, frames, mel bins) tensor.'''
  layer = front_end()
  out = layer(waveform(32000)[None, :])
  assert out.shape == (1, 101, MELS)


def test_frames_num_of_a_thirty_second_waveform_is_3001() -> None:
  '''A 30 second waveform is 3001 frames, not 2977.'''
  layer = front_end()
  assert layer.frames_num(32000 * 30) == 3001
  out = layer(waveform(32000 * 30)[None, :])
  assert out.shape[1] == 3001


def test_frames_num_without_the_reflect_pad_would_be_wrong() -> None:
  '''The reflect pad is what makes the frame count 3001.'''
  padded = 32000 * 30 + 2 * (N_FFT // 2)
  with_pad = (padded - N_FFT) // HOP + 1
  assert with_pad == 3001
  # The unpadded framing would give 2997 frames, which is the bug the
  # explicit pad avoids.  (The porting guide says 2977; the arithmetic
  # gives 2997.  Either way it is not 3001.)
  assert (32000 * 30 - N_FFT) // HOP + 1 == 2997
  assert (32000 * 30 - N_FFT) // HOP + 1 != 3001


def test_logmel_matches_an_independent_implementation() -> None:
  '''The log-mel matches the numpy reference, to 1e-4.'''
  layer = front_end()
  audio = waveform(32000)
  got = layer(audio[None, :]).numpy()[0]
  expected = reference_logmel(audio)
  assert got.shape == expected.shape
  assert np.max(np.abs(got - expected)) < 1e-4


def test_the_power_is_fed_to_the_decibel_formula() -> None:
  '''Feeding the magnitude would change every decibel value.'''
  layer = front_end()
  audio = waveform(32000)
  got = layer(audio[None, :]).numpy()[0]
  librosa = pytest.importorskip('librosa')
  padded = np.pad(audio, (N_FFT // 2, N_FFT // 2), mode='reflect')
  window = np.hanning(N_FFT + 1)[:-1]
  frames = np.lib.stride_tricks.sliding_window_view(padded, N_FFT)[::HOP]
  magnitude = np.abs(np.fft.rfft(frames * window, n=N_FFT))
  mel = librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=MELS, fmin=FMIN,
                             fmax=FMAX)
  magnitude_db = 10.0 * np.log10(np.maximum(magnitude @ mel.T, 1e-10))
  # The mel filters mix several linear bins, so the magnitude is not
  # simply the square root of the power: the two versions differ by
  # several decibels, and the question matters.
  live = magnitude_db > -99.0
  assert live.sum() > 0
  assert np.max(np.abs(got[live] - magnitude_db[live])) > 1.0


def test_the_mel_matrix_is_the_slaney_one_and_not_the_htk_one() -> None:
  '''The mel matrix is librosa's, and not tf.signal's HTK one.'''
  committed = load_mel_matrix(SR, N_FFT, MELS, FMIN, FMAX)
  htk = tf.signal.linear_to_mel_weight_matrix(
      num_mel_bins=MELS, num_spectrogram_bins=N_FFT // 2 + 1,
      sample_rate=SR, lower_edge_hertz=FMIN, upper_edge_hertz=FMAX).numpy()
  assert committed.shape == (N_FFT // 2 + 1, MELS)
  assert not np.allclose(committed, htk, atol=1e-6)
  librosa = pytest.importorskip('librosa')
  expected = librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=MELS, fmin=FMIN,
                                 fmax=FMAX).T
  assert np.max(np.abs(committed - expected)) < 1e-6


def test_the_mel_matrix_and_the_window_are_frozen() -> None:
  '''The frozen buffers must not reach the optimiser.'''
  layer = front_end()
  assert not layer.mel_weight.trainable
  assert not layer.window.trainable
  assert [v.name for v in layer.trainable_variables] == []


def test_the_window_is_the_periodic_hann_window() -> None:
  '''The analysis window is the periodic Hann window of librosa.'''
  layer = front_end()
  expected = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(N_FFT) / N_FFT))
  assert np.max(np.abs(layer.window.numpy() - expected)) < 1e-6


def test_the_decibel_floor_is_applied() -> None:
  '''A silent waveform reads the floor, not minus infinity.'''
  layer = front_end()
  out = layer(np.zeros((1, 32000), dtype=np.float32)).numpy()
  assert np.all(np.isfinite(out))
  assert np.allclose(out, -100.0, atol=1e-4)


def test_top_db_clamps_the_dynamic_range() -> None:
  '''The top_db clamp is relative to the maximum of the batch.'''
  layer = front_end(top_db=20.0)
  out = layer(waveform(32000)[None, :]).numpy()
  assert out.max() - out.min() <= 20.0 + 1e-4


def test_top_db_must_not_be_negative() -> None:
  '''A negative top_db is refused.'''
  with pytest.raises(ValueError, match='top_db'):
    front_end(top_db=-1.0)


def test_a_three_dimensional_input_is_refused() -> None:
  '''The front-end takes a (batch, samples) waveform.'''
  layer = front_end()
  with pytest.raises(ValueError, match='batch, samples'):
    layer(np.zeros((2, 1, 32000), dtype=np.float32))


def test_the_config_round_trips() -> None:
  '''The configuration of the layer can be read back.'''
  layer = front_end(top_db=40.0)
  config = layer.get_config()
  assert config['n_fft'] == N_FFT
  assert config['hop_size'] == HOP
  assert config['mel_bins'] == MELS
  assert config['top_db'] == 40.0


@pytest.mark.parity
def test_the_torch_front_end_agrees_to_1e_4() -> None:
  '''The torch and the Keras front-ends produce the same log-mel.'''
  torch = pytest.importorskip('torch')
  from birdcall.helpers.pl_frontend_logmel import LogmelFilterBank
  from birdcall.helpers.pl_frontend_spectrogram import Spectrogram

  audio = waveform(32000)
  spectrogram = Spectrogram(n_fft=N_FFT, hop_length=HOP, win_length=N_FFT,
                            window='hann', center=True, pad_mode='reflect',
                            freeze_parameters=True)
  bank = LogmelFilterBank(sr=SR, n_fft=N_FFT, n_mels=MELS, fmin=FMIN,
                           fmax=FMAX, ref=1.0, amin=1e-10, top_db=None)
  with torch.no_grad():
    expected = bank(spectrogram(torch.from_numpy(audio)[None, :]))
  expected = expected[0, 0].numpy()
  got = front_end()(audio[None, :]).numpy()[0]
  assert got.shape == expected.shape
  assert np.max(np.abs(got - expected)) < 1e-4
