#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests of the vote entry point and of the ensemble manifest.

The manifest that ships with the package is checked against the
original's thirteen models, and a synthetic manifest is voted over in
``tmp_path`` to check the entry point end to end.
'''

import csv
from pathlib import Path

import numpy as np
import pytest
import yaml

from birdcall.config_params.loader import load_parameters
from birdcall.metrics.scoring import save_score_artefact

import birdcall.run_vote as run_vote
import birdcall.voting.vote as vote

AUDIO_ID = '0a997dff022e3ad9744d4e7bbf923288'
OTHER_AUDIO_ID = '1b998dff022e3ad9744d4e7bbf923288'
NUM_CLASSES = 2
NUM_FRAMES = 1000

EXPECTED_MEMBERS = [
    'final_sed_dense121_nomix_fold0',
    'final_sed_dense121_nomix_fold1',
    'final_sed_dense121_nomix_fold2_augd',
    'final_sed_dense121_nomix_fold3_augd',
    'final_5fold_sed_dense121_nomix_fold0',
    'final_5fold_sed_dense121_nomix_fold1',
    'final_5fold_sed_dense121_nomix_fold2',
    'final_5fold_sed_dense121_nomix_fold3',
    'final_5fold_sed_dense121_nomix_fold4',
    'final_sed_dense121_mix_fold0',
    'final_sed_dense121_mix_fold1',
    'final_sed_dense121_mix_fold2',
    'final_sed_dense121_mix_fold3',
]


def _scores(num_frames=NUM_FRAMES, num_classes=NUM_CLASSES, clip_value=0.9,
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


def _manifest(tmp_path, num_members=5, num_firing=4, audio_ids=(AUDIO_ID,),
              **overrides):
  '''Writes a synthetic ensemble manifest and its score artefacts.

  Args:
    tmp_path: The pytest temporary directory.
    num_members: The number of members of the ensemble.
    num_firing: The number of members that fire for the first species.
    audio_ids: The audio files every member scored.
    overrides: The manifest keys to override.

  Returns:
    The path of the manifest.
  '''
  scores = tmp_path / 'scores'
  members = []
  for index in range(num_members):
    name = f'model_{index}'
    clip, frames = _scores(clip_value=0.9 if index < num_firing else 0.1,
                           frame_value=0.9 if index < num_firing else 0.1)
    for audio_id in audio_ids:
      save_score_artefact(scores / name, audio_id, clip, frames,
                          vote.FRAME_RATE)
    members.append({
        'name': name,
        'experiment': ('config_params/experiments/example_config/'
                       'example_config.yaml'),
        'scores': name,
    })
  document = {
      'name': 'test_ensemble',
      'votes': 4,
      'clip_threshold': 0.3,
      'frame_threshold': 0.3,
      'frame_rate': 100.0,
      'default_site': 1,
      'members': members,
  }
  document.update(overrides)
  path = tmp_path / 'ensemble.yaml'
  path.write_text(yaml.safe_dump(document), encoding='utf-8')
  return path


def _read_csv(path):
  '''Reads a submission CSV back.

  Args:
    path: The path of the CSV file.

  Returns:
    The rows of the CSV file.
  '''
  with path.open(encoding='utf-8', newline='') as handle:
    return list(csv.reader(handle))


def test_shipped_ensemble_lists_thirteen_members() -> None:
  '''The packaged manifest is the original's winning ensemble.'''
  manifest = run_vote.read_ensemble(run_vote.DEFAULT_ENSEMBLE)
  members = manifest['members']
  assert len(members) == 13
  assert [member['name'] for member in members] == EXPECTED_MEMBERS
  assert manifest['votes'] == 4
  assert manifest['clip_threshold'] == 0.3
  assert manifest['frame_threshold'] == 0.3
  assert manifest['frame_rate'] == 100.0
  families = [member['family'] for member in members]
  assert families.count('nomix_4fold') == 4
  assert families.count('nomix_5fold') == 5
  assert families.count('mix_4fold') == 4


def test_shipped_ensemble_names_existing_experiments() -> None:
  '''Every member names an experiment configuration that loads.'''
  manifest = run_vote.read_ensemble(run_vote.DEFAULT_ENSEMBLE)
  for member in manifest['members']:
    path = run_vote.resolve_experiment_path(member['experiment'])
    assert path.is_file(), f'{path} does not exist'
    parameters = load_parameters(path)
    assert parameters.name == member['name']
    assert parameters.model_name == 'sed_dense121att'


def test_shipped_ensemble_names_a_score_directory() -> None:
  '''Every member names the directory its artefacts live in.'''
  manifest = run_vote.read_ensemble(run_vote.DEFAULT_ENSEMBLE)
  for member in manifest['members']:
    assert member['scores'] == member['name']
    assert run_vote.member_score_directory(member, Path('scores')) == (
        Path('scores') / member['name'])


def test_vote_ensemble_writes_the_submission(tmp_path) -> None:
  '''The vote of a synthetic ensemble produces the expected rows.'''
  manifest = _manifest(tmp_path)
  output = tmp_path / 'submission.csv'
  target = run_vote.vote_ensemble(manifest, tmp_path / 'scores', output)
  assert target == output
  rows = _read_csv(target)
  assert rows[0] == ['row_id', 'labels']
  assert len(rows) == 3
  assert rows[1][0] == f'site_1_{AUDIO_ID}_5'
  assert rows[1][1] == vote.get_inv_bird_code()[0]
  assert rows[2][0] == f'site_1_{AUDIO_ID}_10'


def test_vote_ensemble_discovers_the_audio_files(tmp_path) -> None:
  '''Without an audio list the files come from the score artefacts.'''
  manifest = _manifest(tmp_path, audio_ids=(AUDIO_ID, OTHER_AUDIO_ID))
  output = tmp_path / 'submission.csv'
  target = run_vote.vote_ensemble(manifest, tmp_path / 'scores', output)
  rows = _read_csv(target)
  assert len(rows) == 5
  assert rows[1][0] == f'site_1_{AUDIO_ID}_5'
  assert rows[3][0] == f'site_1_{OTHER_AUDIO_ID}_5'


def test_vote_ensemble_pins_the_site_of_a_file(tmp_path) -> None:
  '''A listed audio file is voted at the site the manifest gives it.'''
  manifest = _manifest(tmp_path, audio=[{'audio_id': AUDIO_ID, 'site': 3}])
  output = tmp_path / 'submission.csv'
  target = run_vote.vote_ensemble(manifest, tmp_path / 'scores', output)
  rows = _read_csv(target)
  assert len(rows) == 2
  assert rows[1][0] == f'site_3_{AUDIO_ID}'


def test_vote_ensemble_honours_the_quorum_override(tmp_path) -> None:
  '''A quorum of five rejects the four models that fired.'''
  manifest = _manifest(tmp_path)
  output = tmp_path / 'submission.csv'
  target = run_vote.vote_ensemble(manifest, tmp_path / 'scores', output,
                                  votes=5)
  rows = _read_csv(target)
  assert rows[1][1] == 'nocall'


def test_vote_ensemble_honours_the_threshold_override(tmp_path) -> None:
  '''A clip threshold of 0.95 stops the models from firing.'''
  manifest = _manifest(tmp_path)
  output = tmp_path / 'submission.csv'
  target = run_vote.vote_ensemble(manifest, tmp_path / 'scores', output,
                                  clip_threshold=0.95)
  rows = _read_csv(target)
  assert rows[1][1] == 'nocall'


def test_vote_ensemble_rejects_a_missing_artefact(tmp_path) -> None:
  '''A model that did not score a file is an error, not a silent skip.'''
  manifest = _manifest(tmp_path, audio_ids=(AUDIO_ID, OTHER_AUDIO_ID),
                       audio=[{'audio_id': OTHER_AUDIO_ID, 'site': 1}])
  (tmp_path / 'scores' / 'model_0' / f'{OTHER_AUDIO_ID}.npz').unlink()
  with pytest.raises(FileNotFoundError, match='no score artefact'):
    run_vote.vote_ensemble(manifest, tmp_path / 'scores',
                           tmp_path / 'submission.csv')


def test_vote_ensemble_rejects_a_manifest_without_members(tmp_path) -> None:
  '''An ensemble without members is not an ensemble.'''
  path = tmp_path / 'ensemble.yaml'
  path.write_text(yaml.safe_dump({'name': 'empty'}), encoding='utf-8')
  with pytest.raises(ValueError, match='lists no members'):
    run_vote.read_ensemble(path)


def test_vote_ensemble_rejects_a_foreign_frame_rate(tmp_path) -> None:
  '''The row id format is tied to the frame rate.'''
  manifest = _manifest(tmp_path, frame_rate=50.0)
  with pytest.raises(ValueError, match='frame rate'):
    run_vote.vote_ensemble(manifest, tmp_path / 'scores',
                           tmp_path / 'submission.csv')


def test_main_writes_the_submission(tmp_path) -> None:
  '''The command line entry point writes the submission.'''
  manifest = _manifest(tmp_path)
  output = tmp_path / 'submission.csv'
  code = run_vote.main(['--ensemble', str(manifest),
                         '--scores', str(tmp_path / 'scores'),
                         '--output', str(output)])
  assert code == 0
  rows = _read_csv(output)
  assert rows[0] == ['row_id', 'labels']
  assert len(rows) == 3


def test_main_overrides_the_manifest(tmp_path) -> None:
  '''The command line overrides the manifest.'''
  manifest = _manifest(tmp_path)
  output = tmp_path / 'submission.csv'
  code = run_vote.main(['--ensemble', str(manifest),
                         '--scores', str(tmp_path / 'scores'),
                         '--output', str(output),
                         '--votes', '5',
                         '--clip-threshold', '0.3',
                         '--frame-threshold', '0.3'])
  assert code == 0
  rows = _read_csv(output)
  assert rows[1][1] == 'nocall'


def test_main_needs_the_scores_directory(tmp_path) -> None:
  '''The scores directory is not optional.'''
  with pytest.raises(SystemExit):
    run_vote.main(['--ensemble', str(_manifest(tmp_path))])
