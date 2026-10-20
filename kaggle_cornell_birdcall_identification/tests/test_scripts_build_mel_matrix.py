#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests of the mel matrix build script.

librosa is a **build-time only** dependency, so the script that uses it
has to fail with a message that says so rather than with a traceback
from deep inside an import.  The matrix it writes is the committed one,
so the tests write it into a temporary directory instead of overwriting
the package asset.
'''

import sys

import numpy as np
import pytest

from birdcall.helpers import mel_matrix as mel_helper
import birdcall.scripts.build_mel_matrix as mel_script


def test_the_defaults_are_the_shipped_values():
  '''The shipped matrix was built with these four numbers.'''
  assert mel_script.DEFAULT_SAMPLE_RATE == 32000
  assert mel_script.DEFAULT_N_FFT == 1024
  assert mel_script.DEFAULT_N_MELS == 64
  assert mel_script.DEFAULT_FMIN == 50
  assert mel_script.DEFAULT_FMAX == 14000


def test_the_script_writes_a_matrix_of_the_right_shape(tmp_path, monkeypatch,
                                                        capsys):
  '''A 1024 point FFT gives 513 frequency bins and 64 mel bands.'''
  monkeypatch.setattr(mel_helper, 'ASSETS_DIR', tmp_path)
  assert mel_script.main([]) == 0
  written = list(tmp_path.glob('*.npy'))
  assert len(written) == 1
  matrix = np.load(written[0])
  assert matrix.shape == (513, 64)
  assert matrix.dtype == np.float64


def test_the_script_prints_the_path_it_wrote(tmp_path, monkeypatch, capsys):
  '''The operator is told what was written.'''
  monkeypatch.setattr(mel_helper, 'ASSETS_DIR', tmp_path)
  mel_script.main([])
  captured = capsys.readouterr()
  assert 'Wrote' in captured.out
  assert str(tmp_path) in captured.out


def test_the_matrix_is_the_shipped_one(tmp_path, monkeypatch):
  '''The default build reproduces the committed matrix.'''
  monkeypatch.setattr(mel_helper, 'ASSETS_DIR', tmp_path)
  mel_script.main([])
  built = np.load(next(tmp_path.glob('*.npy')))
  shipped = mel_helper.load_mel_matrix(32000, 1024, 64, 50, 14000)
  assert built.shape == shipped.shape
  assert np.allclose(built, shipped, atol=1e-6)


def test_the_parameters_reach_the_matrix(tmp_path, monkeypatch):
  '''The four numbers are the matrix's only inputs.'''
  monkeypatch.setattr(mel_helper, 'ASSETS_DIR', tmp_path)
  assert mel_script.main(['--sample-rate', '16000', '--n-fft', '512',
                           '--n-mels', '32', '--fmin', '40',
                           '--fmax', '8000']) == 0
  written = list(tmp_path.glob('*.npy'))
  assert len(written) == 1
  matrix = np.load(written[0])
  assert matrix.shape == (257, 32)


def test_a_missing_librosa_fails_with_a_clear_message(tmp_path, monkeypatch):
  '''librosa is not a runtime dependency, so its absence is expected here.'''
  monkeypatch.setattr(mel_helper, 'ASSETS_DIR', tmp_path)
  monkeypatch.setitem(sys.modules, 'librosa', None)
  with pytest.raises(SystemExit) as error:
    mel_script.main([])
  message = str(error.value)
  assert 'librosa' in message
  assert 'build-time only' in message


def test_a_missing_librosa_writes_nothing(tmp_path, monkeypatch):
  '''A failed build leaves the committed matrix alone.'''
  monkeypatch.setattr(mel_helper, 'ASSETS_DIR', tmp_path)
  monkeypatch.setitem(sys.modules, 'librosa', None)
  with pytest.raises(SystemExit):
    mel_script.main([])
  assert not list(tmp_path.glob('*.npy'))
