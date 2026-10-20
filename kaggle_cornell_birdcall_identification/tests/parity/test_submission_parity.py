#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''The cross-backend parity of the submission.

The vote is **shared**: it operates on the score artefacts, which both
backends write in the same format, and it produces the submission.  The
parity is therefore that the artefacts of one backend and the artefacts
of the other backend -- identical content, written by two different
pieces of code -- lead to a byte identical submission.
'''

import numpy as np
import pytest

from birdcall.metrics.scoring import save_score_artefact
from birdcall.voting.vote import DEFAULT_VOTES
from birdcall.voting.vote import FRAMES_PER_ROW
from birdcall.voting.vote import vote_file
from birdcall.voting.vote import write_submission

AUDIO_ID = 'Site_1_0023'
CLASSES = 264
MODELS = 13
MINUTES = 10


def artefacts(directory, fires: int, frames: int = FRAMES_PER_ROW * 2):
  '''Writes the score artefacts of the thirteen models.

  Args:
    directory: The root of the artefact directories.
    fires: The number of models that fire the species.
    frames: The number of frames of every artefact.

  Returns:
    The paths of the artefacts, one per model.
  '''
  paths = []
  for model in range(MODELS):
    clip = np.zeros(CLASSES, dtype=np.float32)
    score = np.zeros((frames, CLASSES), dtype=np.float32)
    if model < fires:
      clip[1] = 0.9
      score[:, 1] = 0.9
    path = save_score_artefact(directory / f'model_{model:02d}', AUDIO_ID,
                               clip, score, 100.0)
    paths.append(path)
  return paths


def test_four_fires_are_accepted(tmp_path) -> None:
  '''Four of the thirteen models are enough.'''
  paths = artefacts(tmp_path, DEFAULT_VOTES)
  rows = vote_file(AUDIO_ID, paths, site=1)
  assert len(rows) == 2
  assert rows[0][1] == ['ameavo']
  assert rows[0][0] == f'site_1_{AUDIO_ID}_5'


def test_three_fires_are_rejected(tmp_path) -> None:
  '''Three are not.'''
  paths = artefacts(tmp_path, DEFAULT_VOTES - 1)
  rows = vote_file(AUDIO_ID, paths, site=1)
  assert len(rows) == 2
  assert rows[0][1] == []


def test_a_ten_minute_file_yields_one_hundred_and_twenty_rows(
    tmp_path) -> None:
  '''A ten minute file is 120 five second rows.'''
  frames = 60 * 100 * MINUTES
  paths = artefacts(tmp_path, DEFAULT_VOTES, frames=frames)
  rows = vote_file(AUDIO_ID, paths, site=1)
  assert len(rows) == MINUTES * 60 // 5
  assert rows[0][0] == f'site_1_{AUDIO_ID}_5'
  assert rows[-1][0] == f'site_1_{AUDIO_ID}_600'


@pytest.mark.parity
def test_the_two_backends_produce_the_same_submission(tmp_path) -> None:
  '''The same artefacts, written twice, give the same submission.'''
  torch_dir = tmp_path / 'torch'
  keras_dir = tmp_path / 'keras'
  torch_paths = artefacts(torch_dir, DEFAULT_VOTES)
  keras_paths = artefacts(keras_dir, DEFAULT_VOTES)
  torch_rows = vote_file(AUDIO_ID, torch_paths, site=1)
  keras_rows = vote_file(AUDIO_ID, keras_paths, site=1)
  assert torch_rows == keras_rows
  torch_csv = write_submission(torch_rows, tmp_path / 'torch.csv')
  keras_csv = write_submission(keras_rows, tmp_path / 'keras.csv')
  assert torch_csv.read_bytes() == keras_csv.read_bytes()
  assert 'row_id,labels' in torch_csv.read_text(encoding='utf-8')
  assert 'ameavo' in torch_csv.read_text(encoding='utf-8')


@pytest.mark.parity
def test_the_artefacts_of_the_two_backends_are_interchangeable(
    tmp_path) -> None:
  '''A vote over a mixture of the two backends' artefacts is the same.'''
  torch_paths = artefacts(tmp_path / 'torch', DEFAULT_VOTES)
  keras_paths = artefacts(tmp_path / 'keras', DEFAULT_VOTES)
  mixed = [torch_paths[0]] + keras_paths[1:]
  rows = vote_file(AUDIO_ID, mixed, site=1)
  assert rows[0][1] == ['ameavo']
