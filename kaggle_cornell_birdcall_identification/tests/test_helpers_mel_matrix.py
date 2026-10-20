#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests of the committed mel matrix.

The original built the mel matrix at model construction time with
``librosa.filters.mel(sr, n_fft, n_mels, fmin, fmax).T``.  That call uses
librosa's defaults, which are the **Slaney** mel scale with **Slaney
(area) normalisation** -- not the HTK scale that
``tf.signal.linear_to_mel_weight_matrix`` produces.  The two matrices are
different for the same parameters, so the Keras port reuses this one.
'''

import numpy as np
import pytest

from birdcall.helpers.mel_matrix import ASSETS_DIR
from birdcall.helpers.mel_matrix import LIBROSA_VERSION
from birdcall.helpers.mel_matrix import build_mel_matrix
from birdcall.helpers.mel_matrix import load_mel_matrix
from birdcall.helpers.mel_matrix import mel_matrix_filename
from birdcall.helpers.mel_matrix import mel_matrix_path
from birdcall.helpers.mel_matrix import save_mel_matrix

SAMPLE_RATE = 32000
N_FFT = 1024
N_MELS = 64
FMIN = 50
FMAX = 14000


def test_the_filename_carries_every_parameter():
  '''The matrix is identified by the four integers and the version.'''
  name = mel_matrix_filename(SAMPLE_RATE, N_FFT, N_MELS, FMIN, FMAX)
  assert 'sr32000' in name
  assert 'nfft1024' in name
  assert 'nmel64' in name
  assert 'fmin50' in name
  assert 'fmax14000' in name
  assert LIBROSA_VERSION in name
  assert name.endswith('.npy')


def test_the_path_is_under_the_assets_directory():
  '''The committed matrix lives in the assets directory.'''
  path = mel_matrix_path(SAMPLE_RATE, N_FFT, N_MELS, FMIN, FMAX)
  assert path.parent == ASSETS_DIR


def test_the_committed_matrix_loads():
  '''The matrix of the shipped configuration is committed.'''
  matrix = load_mel_matrix(SAMPLE_RATE, N_FFT, N_MELS, FMIN, FMAX)
  assert matrix.shape == (N_FFT // 2 + 1, N_MELS)
  assert matrix.dtype == np.float32


def test_the_committed_matrix_matches_librosa():
  '''The committed matrix is the Slaney matrix, to 1e-4.'''
  librosa = pytest.importorskip('librosa')
  committed = load_mel_matrix(SAMPLE_RATE, N_FFT, N_MELS, FMIN, FMAX)
  reference = librosa.filters.mel(sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS,
                                  fmin=FMIN, fmax=FMAX).T
  assert np.allclose(committed, reference, atol=1e-4)


def test_the_committed_matrix_is_not_the_htk_one():
  '''The TensorFlow matrix is a different matrix.'''
  import tensorflow as tf
  committed = load_mel_matrix(SAMPLE_RATE, N_FFT, N_MELS, FMIN, FMAX)
  htk = tf.signal.linear_to_mel_weight_matrix(
      num_mel_bins=N_MELS, num_spectrogram_bins=N_FFT // 2 + 1,
      sample_rate=SAMPLE_RATE, lower_edge_hertz=FMIN,
      upper_edge_hertz=FMAX).numpy()
  assert htk.shape == committed.shape
  assert not np.allclose(htk, committed, atol=1e-4)


def test_a_missing_matrix_is_a_clear_error():
  '''A matrix that was never built says how to build it.'''
  with pytest.raises(FileNotFoundError, match='build_mel_matrix'):
    load_mel_matrix(16000, 512, 32, 100, 7000)


def test_the_matrix_can_be_built_and_saved(tmp_path, monkeypatch):
  '''The build and the save are the inverse of the load.'''
  librosa = pytest.importorskip('librosa')
  matrix = build_mel_matrix(SAMPLE_RATE, N_FFT, N_MELS, FMIN, FMAX)
  assert matrix.shape == (N_FFT // 2 + 1, N_MELS)
  assert np.issubdtype(matrix.dtype, np.floating)
  monkeypatch.setattr('birdcall.helpers.mel_matrix.ASSETS_DIR', tmp_path)
  path = save_mel_matrix(matrix, SAMPLE_RATE, N_FFT, N_MELS, FMIN, FMAX)
  assert path.exists()
  loaded = load_mel_matrix(SAMPLE_RATE, N_FFT, N_MELS, FMIN, FMAX)
  assert loaded.dtype == np.float32
  assert np.allclose(loaded, matrix, atol=1e-6)
