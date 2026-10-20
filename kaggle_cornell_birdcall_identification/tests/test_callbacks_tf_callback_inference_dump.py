#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''The tests of the inference dump callback.

The JSON dump keeps the original's key set, and the score artefact
keeps the **time axis**, which is what the vote consumes and what the
original threw away.
'''

import json

import numpy as np
import tensorflow as tf

from birdcall.callbacks.tf_callback_inference_dump import InferenceDumpCallback
from birdcall.metrics.scoring import JSON_KEYS
from birdcall.metrics.scoring import load_score_artefact

CLASSES = 8
FRAMES = 5


class TinyModel(tf.keras.Model):
  '''A model that returns fixed heads.'''

  def call(self, inputs, training: bool = False):
    '''Runs the fake model.

    Args:
      inputs: The waveform.
      training: Whether the model is training.

    Returns:
      A dict of the two heads.
    '''
    del training
    shape = tf.shape(inputs)
    clip = tf.reshape(tf.range(CLASSES, dtype=tf.float32) / CLASSES,
                      (1, 1, CLASSES)) * tf.ones((shape[0], shape[1], 1))
    frames = (tf.reshape(tf.range(FRAMES, dtype=tf.float32) / FRAMES,
                         (1, 1, FRAMES, 1)) *
              tf.ones((shape[0], shape[1], 1, 1)))
    return {
        'clipwise_output': tf.broadcast_to(clip, (shape[0], shape[1], CLASSES)),
        'framewise_output': tf.broadcast_to(frames, (shape[0], shape[1],
                                                      FRAMES, CLASSES)),
    }


def items():
  '''Builds a dataset of two items.

  Returns:
    The dataset.
  '''
  waveforms = np.zeros((2, 1, 16), dtype='float32')
  labels = np.zeros((2, 1, CLASSES), dtype='float32')
  labels[0, 0, 1] = 1.0
  return tf.data.Dataset.from_tensor_slices({
      'filename': ['one', 'two'],
      'waveforms': waveforms,
      'all_labels': labels,
  }).batch(2)


def test_the_json_dump_is_written(tmp_path) -> None:
  '''One JSON per audio file, with the original's key set.'''
  callback = InferenceDumpCallback(save_dir=tmp_path, dataset=items(),
                                   model=TinyModel())
  callback.on_test_end({})
  dumps = sorted(tmp_path.glob('*.json'))
  assert [path.name for path in dumps] == ['one.json', 'two.json']
  payload = json.loads(dumps[0].read_text(encoding='utf-8'))
  assert tuple(payload) == JSON_KEYS
  assert payload['filename'] == 'one'
  assert len(payload['predicted_frames']) == CLASSES
  assert len(payload['predicted_clip']) == CLASSES
  assert len(payload['targets']) == CLASSES


def test_the_score_artefact_keeps_the_time_axis(tmp_path) -> None:
  '''The artefact holds the frames, and the vote needs them.'''
  callback = InferenceDumpCallback(save_dir=tmp_path, dataset=items(),
                                   model=TinyModel())
  callback.on_test_end({})
  artefact = load_score_artefact(tmp_path / 'one.npz')
  assert artefact['frames'].shape == (FRAMES, CLASSES)
  assert artefact['clip'].shape == (CLASSES,)
  assert artefact['audio_id'] == 'one'


def test_the_targets_are_the_labels(tmp_path) -> None:
  '''The dump holds the label vector.'''
  callback = InferenceDumpCallback(save_dir=tmp_path, dataset=items(),
                                   model=TinyModel())
  callback.on_test_end({})
  payload = json.loads((tmp_path / 'one.json').read_text(encoding='utf-8'))
  assert payload['targets'][1] == 1.0
  assert payload['targets'][0] == 0.0


def test_the_dumps_are_merged_on_request(tmp_path) -> None:
  '''The merged file holds every dump.'''
  callback = InferenceDumpCallback(save_dir=tmp_path, dataset=items(),
                                   model=TinyModel(), combine_files=True)
  callback.on_test_end({})
  merged = tmp_path / 'final' / 'merged_results.json'
  assert merged.exists()
  payload = json.loads(merged.read_text(encoding='utf-8'))
  assert len(payload) == 2
  assert payload[0]['filename'] == 'one'


def test_nothing_happens_without_a_dataset(tmp_path) -> None:
  '''A callback without a dataset does nothing, and does not raise.'''
  callback = InferenceDumpCallback(save_dir=tmp_path, model=TinyModel())
  callback.on_test_end({})
  assert list(tmp_path.glob('*.json')) == []
