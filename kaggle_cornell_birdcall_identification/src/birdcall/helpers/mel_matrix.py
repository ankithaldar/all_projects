#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''The Slaney mel filterbank matrix, precomputed and committed.

The original built the mel matrix at model construction time with
``librosa.filters.mel(sr, n_fft, n_mels, fmin, fmax).T``.  That call uses
librosa's defaults, which are the **Slaney** mel scale with **Slaney
(area) normalisation** -- not the HTK scale that
``tf.signal.linear_to_mel_weight_matrix`` produces.  The two matrices are
different for the same ``(sr, n_fft, n_mels, fmin, fmax)``, so the Keras
port must reuse this one rather than the TensorFlow one.

Building the matrix needs librosa, which is an awkward runtime dependency
(it drags in numba, which pins numpy).  The matrix depends only on four
integers, so it is built once, committed as a ``.npy`` next to this package,
and loaded from there at runtime.  ``scripts/build_mel_matrix.py`` rebuilds
it.
'''

from pathlib import Path

import numpy as np

ASSETS_DIR = Path(__file__).resolve().parent.parent / 'assets'
LIBROSA_VERSION = '0.10.2'


def mel_matrix_filename(sample_rate: int, n_fft: int, n_mels: int,
                        fmin: int, fmax: int) -> str:
  '''Builds the committed filename for a mel matrix.

  Args:
    sample_rate: The audio sample rate in Hz.
    n_fft: The FFT size.
    n_mels: The number of mel bands.
    fmin: The lowest frequency in Hz.
    fmax: The highest frequency in Hz.

  Returns:
    The filename, including the librosa version that produced it.
  '''
  name = (f'mel_slaney_sr{sample_rate}_nfft{n_fft}_nmel{n_mels}'
          f'_fmin{fmin}_fmax{fmax}')
  return f'{name}_librosa{LIBROSA_VERSION}.npy'


def mel_matrix_path(sample_rate: int, n_fft: int, n_mels: int, fmin: int,
                    fmax: int) -> Path:
  '''Returns the path of the committed mel matrix.

  Args:
    sample_rate: The audio sample rate in Hz.
    n_fft: The FFT size.
    n_mels: The number of mel bands.
    fmin: The lowest frequency in Hz.
    fmax: The highest frequency in Hz.

  Returns:
    The path the matrix is expected at.
  '''
  return ASSETS_DIR / mel_matrix_filename(sample_rate, n_fft, n_mels, fmin,
                                          fmax)


def build_mel_matrix(sample_rate: int, n_fft: int, n_mels: int, fmin: int,
                     fmax: int) -> np.ndarray:
  '''Builds the mel filterbank matrix with librosa.

  Args:
    sample_rate: The audio sample rate in Hz.
    n_fft: The FFT size.
    n_mels: The number of mel bands.
    fmin: The lowest frequency in Hz.
    fmax: The highest frequency in Hz.

  Returns:
    A ``(n_fft // 2 + 1, n_mels)`` float64 matrix.
  '''
  import librosa  # pylint: disable=import-outside-toplevel

  matrix = librosa.filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=n_mels,
                               fmin=fmin, fmax=fmax)
  return np.ascontiguousarray(matrix.T)


def save_mel_matrix(matrix: np.ndarray, sample_rate: int, n_fft: int,
                    n_mels: int, fmin: int, fmax: int) -> Path:
  '''Writes a mel matrix to the committed assets directory.

  Args:
    matrix: The matrix to write.
    sample_rate: The audio sample rate in Hz.
    n_fft: The FFT size.
    n_mels: The number of mel bands.
    fmin: The lowest frequency in Hz.
    fmax: The highest frequency in Hz.

  Returns:
    The path the matrix was written to.
  '''
  path = mel_matrix_path(sample_rate, n_fft, n_mels, fmin, fmax)
  path.parent.mkdir(parents=True, exist_ok=True)
  np.save(path, np.asarray(matrix, dtype=np.float64))
  return path


def load_mel_matrix(sample_rate: int, n_fft: int, n_mels: int, fmin: int,
                    fmax: int) -> np.ndarray:
  '''Loads the committed mel matrix.

  Args:
    sample_rate: The audio sample rate in Hz.
    n_fft: The FFT size.
    n_mels: The number of mel bands.
    fmin: The lowest frequency in Hz.
    fmax: The highest frequency in Hz.

  Returns:
    A ``(n_fft // 2 + 1, n_mels)`` float32 matrix.

  Raises:
    FileNotFoundError: If the matrix has not been built yet.
  '''
  path = mel_matrix_path(sample_rate, n_fft, n_mels, fmin, fmax)
  if not path.exists():
    raise FileNotFoundError(
        f'Mel matrix not found at {path}. Build it with '
        '`python -m birdcall.scripts.build_mel_matrix`.')
  return np.load(path).astype(np.float32)
