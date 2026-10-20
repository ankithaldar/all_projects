#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests of the shared vote and of the submission writer.

Everything here is synthetic: a thirteen model ensemble is written to
``tmp_path`` with :func:`birdcall.metrics.scoring.save_score_artefact`
and voted on.  The class count is kept small, because the vote only
cares about the shape of the score vectors, but the indices are real
ones so that the species codes the vote emits are the real codes.
'''

import csv

import numpy as np
import pytest

from birdcall.config_params.configs import get_inv_bird_code
from birdcall.metrics.scoring import load_score_artefact
from birdcall.metrics.scoring import save_score_artefact

import birdcall.voting.vote as vote

AUDIO_ID = '0a997dff022e3ad9744d4e7bbf923288'
NUM_CLASSES = 2
INV_BIRD_CODE = get_inv_bird_code()
CODE_OF = {index: code for index, code in INV_BIRD_CODE.items()}


def _artefact(directory, audio_id, clip, frames):
  '''Writes one model's score artefact.

  Args:
    directory: The directory the artefact is written to.
    audio_id: The identifier of the audio file.
    clip: The clip level scores.
    frames: The frame level scores.

  Returns:
    The path of the artefact.
  '''
  return save_score_artefact(directory, audio_id, clip, frames,
                             vote.FRAME_RATE)


def _scores(num_frames, num_classes=NUM_CLASSES, clip_value=0.9,
            frame_value=0.9):
  '''The scores of a model that fires for the first species.

  Args:
    num_frames: The number of frames.
    num_classes: The number of species.
    clip_value: The clip score of the species that fires.
    frame_value: The frame score of the species that fires.

  Returns:
    The clip scores and the frame scores.
  '''
  clip = np.zeros(num_classes, dtype=np.float32)
  clip[0] = clip_value
  frames = np.zeros((num_frames, num_classes), dtype=np.float32)
  frames[:, 0] = frame_value
  return clip, frames


def _ensemble(tmp_path, num_firing, num_quiet, num_frames,
              num_classes=NUM_CLASSES):
  '''Writes one artefact per model of a synthetic ensemble.

  Args:
    tmp_path: The pytest temporary directory.
    num_firing: The number of models that fire for the first species.
    num_quiet: The number of models that fire for nothing.
    num_frames: The number of frames of every model.
    num_classes: The number of species.

  Returns:
    The paths of the artefacts.
  '''
  fire = _scores(num_frames, num_classes, 0.9, 0.9)
  quiet = _scores(num_frames, num_classes, 0.1, 0.1)
  paths = []
  for index in range(num_firing + num_quiet):
    clip, frames = fire if index < num_firing else quiet
    directory = tmp_path / f'model_{index}'
    paths.append(_artefact(directory, AUDIO_ID, clip, frames))
  return paths


def test_model_fires_needs_both_gates() -> None:
  '''A species fires only when both gates are cleared.'''
  clip = np.array([0.9, 0.1], dtype=np.float32)
  frames = np.array([[0.9, 0.9],
                     [0.2, 0.9],
                     [0.9, 0.1]], dtype=np.float32)
  fires = vote.model_fires(clip, frames)
  assert fires.dtype == np.bool_
  assert fires.shape == (3, 2)
  assert fires.tolist() == [[True, False],
                             [False, False],
                             [True, False]]


def test_model_fires_gates_are_independent() -> None:
  '''The clip gate and the frame gate do not substitute for each other.'''
  high = np.array([[0.9]], dtype=np.float32)
  low = np.array([[0.1]], dtype=np.float32)
  assert not vote.model_fires(np.array([0.9], dtype=np.float32),
                               low)[0, 0]
  assert not vote.model_fires(np.array([0.1], dtype=np.float32),
                               high)[0, 0]
  assert vote.model_fires(np.array([0.9], dtype=np.float32),
                           high)[0, 0]


def test_model_fires_thresholds_are_inclusive() -> None:
  '''A score exactly at the threshold fires.'''
  clip = np.array([0.3], dtype=np.float32)
  frames = np.array([[0.3]], dtype=np.float32)
  assert vote.model_fires(clip, frames, 0.3, 0.3)[0, 0]
  assert not vote.model_fires(clip, frames, 0.31, 0.3)[0, 0]
  assert not vote.model_fires(clip, frames, 0.3, 0.31)[0, 0]


def test_model_fires_rejects_bad_shapes() -> None:
  '''The shapes of the two score vectors have to agree.'''
  with pytest.raises(ValueError, match='one dimensional'):
    vote.model_fires(np.zeros((2, 2), dtype=np.float32),
                      np.zeros((4, 2), dtype=np.float32))
  with pytest.raises(ValueError, match='two dimensional'):
    vote.model_fires(np.zeros(2, dtype=np.float32),
                      np.zeros(4, dtype=np.float32))
  with pytest.raises(ValueError, match='classes'):
    vote.model_fires(np.zeros(3, dtype=np.float32),
                      np.zeros((4, 2), dtype=np.float32))


def test_slice_rows_splits_a_ten_minute_file() -> None:
  '''A ten minute file is 120 rows of 500 frames.'''
  frames = np.arange(60000 * 2, dtype=np.float32).reshape(60000, 2)
  rows = vote.slice_rows(frames)
  assert rows.shape == (120, 500, 2)
  assert np.array_equal(rows[0], frames[:500])
  assert np.array_equal(rows[119], frames[59500:60000])


def test_slice_rows_drops_the_partial_row() -> None:
  '''A row shorter than five seconds is not a row.'''
  frames = np.arange(60001 * 2, dtype=np.float32).reshape(60001, 2)
  rows = vote.slice_rows(frames)
  assert rows.shape == (120, 500, 2)
  assert np.array_equal(rows[119], frames[59500:60000])
  assert rows[119][-1, 0] == 2 * 59999
  short = np.zeros((499, 2), dtype=np.float32)
  assert vote.slice_rows(short).shape == (0, 500, 2)


def test_count_votes_four_of_thirteen_is_accepted(tmp_path) -> None:
  '''Four of the thirteen models are enough.'''
  paths = _ensemble(tmp_path, 4, 9, 1000)
  votes = vote.count_votes(paths)
  assert votes.shape == (2, NUM_CLASSES)
  assert votes[0, 0] == 4
  accepted = vote.accepted_species(votes, vote.DEFAULT_VOTES)
  assert accepted[0, 0]
  assert not accepted[0, 1]


def test_count_votes_three_of_thirteen_is_rejected(tmp_path) -> None:
  '''Three of the thirteen models are not enough.'''
  paths = _ensemble(tmp_path, 3, 10, 1000)
  votes = vote.count_votes(paths)
  assert votes[0, 0] == 3
  accepted = vote.accepted_species(votes, vote.DEFAULT_VOTES)
  assert not accepted[0, 0]


def test_count_votes_ignores_a_model_that_misses_a_gate(tmp_path) -> None:
  '''A model that clears one gate and misses the other does not vote.'''
  clip_ok = np.array([0.9, 0.0], dtype=np.float32)
  frame_ok = np.full((500, 2), 0.9, dtype=np.float32)
  clip_bad = np.array([0.1, 0.0], dtype=np.float32)
  frame_bad = np.full((500, 2), 0.9, dtype=np.float32)
  low = np.zeros((500, 2), dtype=np.float32)
  paths = []
  for index in range(13):
    if index < 4:
      clip, frames = clip_ok, low
    else:
      clip, frames = clip_bad, frame_ok
    directory = tmp_path / f'model_{index}'
    paths.append(_artefact(directory, AUDIO_ID, clip, frames))
  votes = vote.count_votes(paths)
  assert votes[0, 0] == 0
  assert not vote.accepted_species(votes, vote.DEFAULT_VOTES)[0, 0]


def test_count_votes_is_an_integer_sum_over_the_model_axis(tmp_path) -> None:
  '''The vote count is the sum of the per model masks.'''
  paths = _ensemble(tmp_path, 4, 9, 1000)
  votes = vote.count_votes(paths)
  assert np.issubdtype(votes.dtype, np.integer)
  masks = []
  for path in paths:
    data = load_score_artefact(path)
    masks.append(vote.model_fires(data['clip'], data['frames']))
  expected = sum(vote.slice_rows(mask, vote.FRAMES_PER_ROW).any(axis=1)
                 for mask in masks)
  assert np.array_equal(votes, expected)
  assert votes[0, 0] == 4
  assert votes.max() <= 13


def test_count_votes_of_a_ten_minute_file(tmp_path) -> None:
  '''A ten minute file is 120 rows, and the trailing partial row is gone.'''
  paths = _ensemble(tmp_path, 4, 9, 60001)
  votes = vote.count_votes(paths)
  assert votes.shape == (120, NUM_CLASSES)
  assert votes[0, 0] == 4
  assert votes[119, 0] == 4


def test_count_votes_rejects_models_that_disagree(tmp_path) -> None:
  '''The models of one file have to agree on its shape.'''
  fire = _scores(1000)
  other = _scores(1000, 3)
  paths = []
  for index, (clip, frames) in enumerate([fire, other]):
    directory = tmp_path / f'model_{index}'
    paths.append(_artefact(directory, AUDIO_ID, clip, frames))
  with pytest.raises(ValueError, match='classes'):
    vote.count_votes(paths)
  longer = _scores(1500)
  paths = []
  for index, (clip, frames) in enumerate([fire, longer]):
    directory = tmp_path / f'model_{index}'
    paths.append(_artefact(directory, AUDIO_ID, clip, frames))
  with pytest.raises(ValueError, match='rows'):
    vote.count_votes(paths)


def test_count_votes_needs_at_least_one_model() -> None:
  '''An empty ensemble cannot vote.'''
  with pytest.raises(ValueError, match='at least one score artefact'):
    vote.count_votes([])


def test_accepted_species_needs_a_positive_quorum() -> None:
  '''A quorum of zero votes would accept everything.'''
  with pytest.raises(ValueError, match='at least 1'):
    vote.accepted_species(np.zeros((1, 2), dtype=np.int64), 0)


def test_build_row_id_gives_one_id_per_window() -> None:
  '''Every five second window of a site 1 file has its own id.'''
  first = vote.build_row_id(1, AUDIO_ID, 0)
  second = vote.build_row_id(1, AUDIO_ID, 1)
  third = vote.build_row_id(1, AUDIO_ID, 2)
  assert first != second != third
  assert first == f'site_1_{AUDIO_ID}_5'
  assert second == f'site_1_{AUDIO_ID}_10'
  assert third == f'site_1_{AUDIO_ID}_15'
  assert vote.build_row_id(2, AUDIO_ID, 0) == f'site_2_{AUDIO_ID}_5'


def test_build_row_id_gives_one_id_for_site_three() -> None:
  '''Site 3 is labelled per file, so it has a single id.'''
  first = vote.build_row_id(3, AUDIO_ID, 0)
  assert first == f'site_3_{AUDIO_ID}'
  assert vote.build_row_id(3, AUDIO_ID, 7) == first


def test_build_row_id_rejects_bad_arguments() -> None:
  '''The site and the row index have to make sense.'''
  with pytest.raises(ValueError, match='Unknown site'):
    vote.build_row_id(4, AUDIO_ID, 0)
  with pytest.raises(ValueError, match='must not be negative'):
    vote.build_row_id(1, AUDIO_ID, -1)
  with pytest.raises(ValueError, match='frames_per_row must be positive'):
    vote.build_row_id(1, AUDIO_ID, 0, 0)


def test_vote_file_emits_one_row_per_window(tmp_path) -> None:
  '''A site 1 file gets one row per five second window.'''
  paths = _ensemble(tmp_path, 4, 9, 1500)
  rows = vote.vote_file(AUDIO_ID, paths, site=1)
  assert len(rows) == 3
  assert [row_id for row_id, _ in rows] == [
      f'site_1_{AUDIO_ID}_5', f'site_1_{AUDIO_ID}_10',
      f'site_1_{AUDIO_ID}_15']
  assert rows[0][1] == [CODE_OF[0]]
  assert rows[2][1] == [CODE_OF[0]]


def test_vote_file_emits_several_species(tmp_path) -> None:
  '''Two species can reach the quorum in the same row.'''
  paths = []
  for index in range(13):
    clip = np.zeros(NUM_CLASSES, dtype=np.float32)
    frames = np.zeros((1000, NUM_CLASSES), dtype=np.float32)
    if index < 4:
      clip[0] = 0.9
      frames[:, 0] = 0.9
    elif index < 8:
      clip[1] = 0.9
      frames[:, 1] = 0.9
    directory = tmp_path / f'model_{index}'
    paths.append(_artefact(directory, AUDIO_ID, clip, frames))
  rows = vote.vote_file(AUDIO_ID, paths, site=1)
  assert rows[0][1] == [CODE_OF[0], CODE_OF[1]]
  target = vote.write_submission(rows, tmp_path / 'submission.csv')
  with target.open(encoding='utf-8', newline='') as handle:
    reader = list(csv.reader(handle))
  assert reader[1] == [rows[0][0], f'{CODE_OF[0]} {CODE_OF[1]}']


def test_vote_file_emits_one_row_for_site_three(tmp_path) -> None:
  '''A site 3 file gets a single row for the whole file.'''
  paths = _ensemble(tmp_path, 4, 9, 1500)
  rows = vote.vote_file(AUDIO_ID, paths, site=3)
  assert len(rows) == 1
  assert rows[0][0] == f'site_3_{AUDIO_ID}'
  assert rows[0][1] == [CODE_OF[0]]


def test_vote_file_drops_the_partial_row(tmp_path) -> None:
  '''The trailing partial row of a file is not a row.'''
  paths = _ensemble(tmp_path, 4, 9, 1200)
  rows = vote.vote_file(AUDIO_ID, paths, site=1)
  assert len(rows) == 2


def test_vote_file_rejects_an_unknown_site(tmp_path) -> None:
  '''Only the three sites of the test set exist.'''
  paths = _ensemble(tmp_path, 2, 2, 500)
  with pytest.raises(ValueError, match='Unknown site'):
    vote.vote_file(AUDIO_ID, paths, site=9)


def test_write_submission_writes_nocall_for_an_empty_row(tmp_path) -> None:
  '''A row with no accepted species carries the string nocall.'''
  target = vote.write_submission([(f'site_1_{AUDIO_ID}_5', [])],
                                 tmp_path / 'submission.csv')
  with target.open(encoding='utf-8', newline='') as handle:
    reader = list(csv.reader(handle))
  assert reader[0] == ['row_id', 'labels']
  assert reader[1] == [f'site_1_{AUDIO_ID}_5', 'nocall']


def test_write_submission_separates_species_with_spaces(tmp_path) -> None:
  '''Several species in one row are space separated.'''
  codes = [CODE_OF[0], CODE_OF[1]]
  target = vote.write_submission([(f'site_1_{AUDIO_ID}_5', codes)],
                                 tmp_path / 'submission.csv')
  with target.open(encoding='utf-8', newline='') as handle:
    reader = list(csv.reader(handle))
  assert reader[1] == [f'site_1_{AUDIO_ID}_5', ' '.join(codes)]
  assert ' ' in reader[1][1]


def test_write_submission_has_one_row_per_emitted_row(tmp_path) -> None:
  '''The CSV has the two columns and one row per emitted row.'''
  rows = [(f'site_1_{AUDIO_ID}_{5 * (index + 1)}', [CODE_OF[0]])
          for index in range(4)]
  target = vote.write_submission(rows, tmp_path / 'submission.csv')
  with target.open(encoding='utf-8', newline='') as handle:
    reader = list(csv.reader(handle))
  assert reader[0] == ['row_id', 'labels']
  assert len(reader) == len(rows) + 1
  assert [row[0] for row in reader[1:]] == [row_id for row_id, _ in rows]


def test_write_submission_round_trips(tmp_path) -> None:
  '''What is written can be read back unchanged.'''
  rows = [(f'site_1_{AUDIO_ID}_5', [CODE_OF[0], CODE_OF[1]]),
          (f'site_1_{AUDIO_ID}_10', []),
          (f'site_3_{AUDIO_ID}', [CODE_OF[3]])]
  target = vote.write_submission(rows, tmp_path / 'submission.csv')
  with target.open(encoding='utf-8', newline='') as handle:
    reader = list(csv.reader(handle))
  assert reader[0] == ['row_id', 'labels']
  assert [row[0] for row in reader[1:]] == [row_id for row_id, _ in rows]
  assert reader[1][1] == f'{CODE_OF[0]} {CODE_OF[1]}'
  assert reader[2][1] == 'nocall'
  assert reader[3][1] == CODE_OF[3]


def test_write_submission_maps_class_indices(tmp_path) -> None:
  '''A class index is turned into its code by the bird code map.'''
  target = vote.write_submission([(f'site_1_{AUDIO_ID}_5', [0, 1])],
                                 tmp_path / 'submission.csv',
                                 bird_codes={0: 'amecro', 1: 'amerob'})
  with target.open(encoding='utf-8', newline='') as handle:
    reader = list(csv.reader(handle))
  assert reader[1] == [f'site_1_{AUDIO_ID}_5', 'amecro amerob']


def test_write_submission_rejects_an_unknown_class_index(tmp_path) -> None:
  '''A class index with no code is an error, not a silent empty label.'''
  with pytest.raises(ValueError, match='no species code'):
    vote.write_submission([(f'site_1_{AUDIO_ID}_5', [7])],
                           tmp_path / 'submission.csv',
                           bird_codes={0: 'amecro'})
