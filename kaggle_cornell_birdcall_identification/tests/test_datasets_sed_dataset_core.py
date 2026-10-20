#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests of the shared half of the SED dataset.

The core is framework agnostic: it reads the fold CSV, parses the label
cells, resolves the audio paths and slices the windows.  The behaviour
that is inherited verbatim from the original is the one that matters:
the secondary labels are a **Python literal** parsed with
``ast.literal_eval``, never with ``eval``.
'''

import pathlib

import numpy as np
import pandas as pd
import pytest

from birdcall.config_params.configs import get_bird_code
from birdcall.datasets import sed_dataset_core
from birdcall.datasets.sed_dataset_core import background_rows
from birdcall.datasets.sed_dataset_core import build_label_vector
from birdcall.datasets.sed_dataset_core import read_fold_rows
from birdcall.datasets.sed_dataset_core import resolve_audio_path
from birdcall.datasets.sed_dataset_core import slice_for_training
from birdcall.datasets.sed_dataset_core import slice_for_validation


def test_secondary_labels_are_parsed_with_literal_eval(tmp_path):
  '''The secondary labels cell is a Python literal.'''
  csv_path = tmp_path / 'fold_0.csv'
  pd.DataFrame([{
      'filename': 'REC0_0.mp3',
      'ebird_code': 'AMEBR',
      'secondary_labels': '["Black-billed Cuckoo"]',
  }]).to_csv(csv_path, index=False)
  rows = sed_dataset_core.read_fold_rows(csv_path)
  assert rows[0]['secondary_labels'] == ['Black-billed Cuckoo']
  assert rows[0]['ebird_code'] == 'AMEBR'
  assert rows[0]['filename'] == 'REC0_0.mp3'


def test_a_nan_secondary_labels_cell_does_not_raise(tmp_path):
  '''A missing cell means "no secondary label".'''
  csv_path = tmp_path / 'fold_0.csv'
  pd.DataFrame([{
      'filename': 'REC0_0.mp3',
      'ebird_code': 'AMEBR',
      'secondary_labels': None,
  }]).to_csv(csv_path, index=False)
  rows = sed_dataset_core.read_fold_rows(csv_path)
  assert rows[0]['secondary_labels'] == []


def test_the_datasets_package_never_calls_eval():
  '''The original ran `eval` on a data file; this port does not.'''
  package = pathlib.Path(sed_dataset_core.__file__).parent
  for path in sorted(package.glob('*.py')):
    source = path.read_text()
    assert 'eval(' not in source.replace('literal_eval(', '')
  assert 'ast.literal_eval' in pathlib.Path(
      sed_dataset_core.__file__).read_text()


def test_a_non_literal_secondary_labels_cell_is_refused(tmp_path):
  '''A cell that is not a literal is a clear error.'''
  csv_path = tmp_path / 'fold_0.csv'
  pd.DataFrame([{
      'filename': 'REC0_0.mp3',
      'ebird_code': 'AMEBR',
      'secondary_labels': 'os.system("ls")',
  }]).to_csv(csv_path, index=False)
  with pytest.raises(ValueError, match='not a Python literal'):
    read_fold_rows(csv_path)


def test_a_missing_secondary_labels_column_is_tolerated(tmp_path):
  '''A fold CSV without the column is read as "no secondary label".'''
  csv_path = tmp_path / 'fold_0.csv'
  pd.DataFrame([{'filename': 'REC0_0.mp3', 'ebird_code': 'AMEBR'}
                ]).to_csv(csv_path, index=False)
  rows = read_fold_rows(csv_path)
  assert rows[0]['secondary_labels'] == []
  assert rows[0]['ebird_code'] == 'AMEBR'


def test_the_label_vector_is_a_one_hot_of_264():
  '''The label is a 264-vector, not a per-frame matrix.'''
  bird_code = get_bird_code()
  codes = sorted(bird_code)[:2]
  vector = build_label_vector(codes, bird_code)
  assert vector.shape == (264,)
  assert vector.dtype == np.float32
  assert vector.sum() == 2.0
  assert vector[bird_code[codes[0]]] == 1.0


def test_an_unknown_code_is_skipped():
  '''A code that is not in the label space is skipped.'''
  bird_code = get_bird_code()
  vector = build_label_vector(['NOT_A_SPECIES'], bird_code)
  assert vector.shape == (264,)
  assert vector.sum() == 0.0


def test_the_training_window_is_cropped_at_a_random_offset():
  '''A long recording is cropped, a short one is padded.'''
  rng = np.random.default_rng(0)
  audio = np.arange(100, dtype=np.float32)
  window = slice_for_training(audio, 50, rng)
  assert window.shape == (50,)
  assert window.dtype == np.float32
  assert set(window.tolist()) <= set(audio.tolist())
  short = slice_for_training(np.arange(1, 11, dtype=np.float32), 50, rng)
  assert short.shape == (50,)
  assert np.count_nonzero(short) == 10


def test_the_validation_windows_are_contiguous():
  '''The validation windows are contiguous and non-overlapping.'''
  audio = np.arange(100, dtype=np.float32)
  windows = slice_for_validation(audio, 50, 2)
  assert len(windows) == 2
  assert windows[0].shape == (50,)
  assert np.array_equal(windows[0], audio[:50])
  assert np.array_equal(windows[1], audio[50:100])


def test_a_short_file_is_zero_padded_at_the_tail():
  '''A short file is padded, not repeated.'''
  audio = np.arange(1, 11, dtype=np.float32)
  windows = slice_for_validation(audio, 50, 2)
  assert len(windows) == 2
  assert np.count_nonzero(windows[0]) == 10
  assert np.count_nonzero(windows[1]) == 0


def test_the_audio_path_rewrites_the_extension():
  '''The original rewrote `.mp3` to `.wav` under the species folder.'''
  path = resolve_audio_path('/root', {'filename': 'REC0_0.mp3',
                                      'ebird_code': 'AMEBR'})
  assert path == pathlib.Path('/root/AMEBR/REC0_0.wav')


def test_a_background_row_resolves_to_its_own_path():
  '''A background row carries no species, its filename is the path.'''
  path = resolve_audio_path('/root', {'filename': '/root/SSW.wav'})
  assert path == pathlib.Path('/root/SSW.wav')


def test_the_background_rows_are_label_less(tmp_path):
  '''A background directory yields rows with a single key.'''
  assert background_rows(None) == []
  (tmp_path / 'SSW_002.wav').write_bytes(b'')
  rows = background_rows(tmp_path)
  assert rows == [{'filename': str(tmp_path / 'SSW_002.wav')}]
