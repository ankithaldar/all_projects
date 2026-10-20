#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests of the score artefact.

The artefact is the handoff between a model and the vote: a ``.npz`` per
audio file per model, holding the clip scores and the **frame scores with
their time axis intact**.  The original's JSON dump max-pooled the time
axis away, so the temporal information was structurally unrecoverable.
'''

import json

import numpy as np
import pytest

from birdcall.metrics.scoring import ARTEFACT_VERSION
from birdcall.metrics.scoring import DEFAULT_FRAME_RATE
from birdcall.metrics.scoring import JSON_KEYS
from birdcall.metrics.scoring import NUM_CLASSES
from birdcall.metrics.scoring import load_score_artefact
from birdcall.metrics.scoring import merge_json_dumps
from birdcall.metrics.scoring import save_score_artefact
from birdcall.metrics.scoring import write_json_dump

AUDIO_ID = 'XYZ123'
FRAMES = 120
CLASSES = NUM_CLASSES


def _clip() -> np.ndarray:
  '''Builds a clip score vector.

  Returns:
    A ``(num_classes,)`` float32 vector.
  '''
  return np.linspace(0.0, 1.0, CLASSES, dtype=np.float32)


def _frames() -> np.ndarray:
  '''Builds a frame score matrix with its time axis intact.

  Returns:
    A ``(num_frames, num_classes)`` float32 matrix.
  '''
  return np.linspace(0.0, 1.0, FRAMES * CLASSES,
                     dtype=np.float32).reshape(FRAMES, CLASSES)


def test_the_artefact_round_trips(tmp_path):
  '''The clip and the frame scores come back as they went in.'''
  path = save_score_artefact(tmp_path, AUDIO_ID, _clip(), _frames(),
                             DEFAULT_FRAME_RATE)
  assert path.name == f'{AUDIO_ID}.npz'
  artefact = load_score_artefact(path)
  assert artefact['audio_id'] == AUDIO_ID
  assert artefact['frame_rate'] == DEFAULT_FRAME_RATE
  assert artefact['clip'].shape == (CLASSES,)
  assert artefact['frames'].shape == (FRAMES, CLASSES)
  assert np.allclose(artefact['frames'], _frames())


def test_the_time_axis_is_preserved(tmp_path):
  '''A ten minute file keeps its hundred and twenty rows.'''
  path = save_score_artefact(tmp_path, AUDIO_ID, _clip(), _frames(),
                             DEFAULT_FRAME_RATE)
  artefact = load_score_artefact(path)
  assert artefact['frames'].shape[0] == FRAMES
  # The rows are not identical: the time axis is really there.
  assert not np.array_equal(artefact['frames'][0], artefact['frames'][-1])


def test_the_artefact_directory_is_created(tmp_path):
  '''A missing directory is created.'''
  target = tmp_path / 'scores' / 'model_a'
  path = save_score_artefact(target, AUDIO_ID, _clip(), _frames(),
                             DEFAULT_FRAME_RATE)
  assert path.parent == target
  assert path.exists()


def test_the_artefact_carries_its_version(tmp_path):
  '''The version is written into the artefact.'''
  path = save_score_artefact(tmp_path, AUDIO_ID, _clip(), _frames(),
                             DEFAULT_FRAME_RATE)
  with np.load(path, allow_pickle=False) as data:
    assert int(data['version']) == ARTEFACT_VERSION
    assert int(data['num_classes']) == NUM_CLASSES


def test_the_json_dump_carries_the_original_key_set(tmp_path):
  '''The key set is byte identical to the original's.'''
  path = write_json_dump(tmp_path, 'REC0_0.mp3', _clip(), _clip(), _clip(),
                          0.5, 0.25)
  document = json.loads(path.read_text(encoding='utf-8'))
  assert tuple(document.keys()) == JSON_KEYS
  assert document['filename'] == 'REC0_0.mp3'
  assert document['lraps_score_clip'] == 0.5
  assert document['lraps_score_frames'] == 0.25
  assert len(document['predicted_frames']) == CLASSES


def test_the_json_dumps_are_merged_into_one_array(tmp_path):
  '''The merge produces a single JSON array.'''
  write_json_dump(tmp_path, 'a', _clip(), _clip(), _clip(), 0.5, 0.25)
  write_json_dump(tmp_path, 'b', _clip(), _clip(), _clip(), 0.5, 0.25)
  merged = merge_json_dumps(tmp_path)
  assert merged.name == 'merged_results.json'
  document = json.loads(merged.read_text(encoding='utf-8'))
  assert len(document) == 2
  assert [entry['filename'] for entry in document] == ['a', 'b']


def test_a_truncated_json_dump_is_refused(tmp_path):
  '''A truncated file is detected rather than corrupting the array.'''
  (tmp_path / 'broken.json').write_text('{"filename": "a"', encoding='utf-8')
  with pytest.raises(ValueError, match='Truncated'):
    merge_json_dumps(tmp_path)
