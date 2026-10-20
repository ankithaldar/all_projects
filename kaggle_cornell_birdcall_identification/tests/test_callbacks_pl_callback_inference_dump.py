#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests of the inference dump callback.

The artefact is the point of the port: the original's JSON dump
max-pooled the frames over the whole time axis, so the temporal
information a five second row needs was gone.  The artefact keeps it.
'''

import json

import numpy as np
import pytest
import torch

from birdcall.callbacks.pl_callback_inference_dump import InferenceDumpCallback
from birdcall.metrics.scoring import load_score_artefact

CLASSES = 264
SNIPPETS = 2
FRAMES = 7


def _outputs(batch_size: int = 2) -> dict:
  '''Builds what a test step returns.

  Args:
    batch_size: The number of files in the batch.

  Returns:
    The output of the test step.
  '''
  generator = torch.Generator().manual_seed(11)
  clipwise = torch.rand(batch_size, SNIPPETS, CLASSES, generator=generator)
  framewise = torch.rand(batch_size, SNIPPETS, FRAMES, CLASSES,
                          generator=generator)
  labels = torch.zeros(batch_size, SNIPPETS, CLASSES)
  labels[:, 0, 3] = 1.0
  return {
      'y_pred': {
          'clipwise_output': clipwise,
          'framewise_output': framewise,
      },
      'x': (torch.zeros(batch_size, SNIPPETS, 10), None),
      'y': {
          'all_labels': labels,
          'filename': [f'REC{index}.wav' for index in range(batch_size)],
      },
  }


def _batch(batch_size: int = 2) -> dict:
  '''Builds the batch of the step.

  Args:
    batch_size: The number of files in the batch.

  Returns:
    The batch.
  '''
  return {'filename': [f'REC{index}.wav' for index in range(batch_size)]}


def test_the_artefact_keeps_the_frame_axis(tmp_path):
  '''The frame scores are written with their time axis.'''
  callback = InferenceDumpCallback(tmp_path)
  outputs = _outputs()
  callback.on_test_batch_end(None, None, outputs, _batch(), 0)
  assert len(callback.artefacts) == 2
  artefact = load_score_artefact(callback.artefacts[0])
  assert artefact['frames'].shape == (SNIPPETS * FRAMES, CLASSES)
  assert artefact['clip'].shape == (CLASSES,)
  assert artefact['audio_id'] == 'REC0.wav'
  assert artefact['frame_rate'] == pytest.approx(100.0)


def test_the_artefact_holds_the_frame_scores(tmp_path):
  '''The written frames are the ones the model produced.'''
  callback = InferenceDumpCallback(tmp_path)
  outputs = _outputs(batch_size=1)
  callback.on_test_batch_end(None, None, outputs, _batch(batch_size=1), 0)
  artefact = load_score_artefact(callback.artefacts[0])
  expected = outputs['y_pred']['framewise_output'][0].reshape(-1, CLASSES)
  assert np.allclose(artefact['frames'], expected.numpy(), atol=1e-6)


def test_the_clip_score_is_the_max_over_the_snippets(tmp_path):
  '''The clip head is pooled over the snippets, as the metrics do.'''
  callback = InferenceDumpCallback(tmp_path)
  outputs = _outputs(batch_size=1)
  callback.on_test_batch_end(None, None, outputs, _batch(batch_size=1), 0)
  artefact = load_score_artefact(callback.artefacts[0])
  expected = outputs['y_pred']['clipwise_output'][0].max(dim=0)[0]
  assert np.allclose(artefact['clip'], expected.numpy(), atol=1e-6)


def test_the_json_dump_is_optional(tmp_path):
  '''The original's JSON dump is only written when it is asked for.'''
  callback = InferenceDumpCallback(tmp_path)
  callback.on_test_batch_end(None, None, _outputs(), _batch(), 0)
  assert callback.dumps == []
  assert not list(tmp_path.glob('*.json'))


def test_the_json_dump_has_the_original_keys(tmp_path):
  '''The JSON dump keeps the original's key set byte for byte.'''
  callback = InferenceDumpCallback(tmp_path, write_json=True)
  callback.on_test_batch_end(None, None, _outputs(), _batch(), 0)
  assert len(callback.dumps) == 2
  payload = json.loads(callback.dumps[0].read_text(encoding='utf-8'))
  assert sorted(payload) == ['filename', 'lraps_score_clip',
                             'lraps_score_frames', 'predicted_clip',
                             'predicted_frames', 'targets']
  assert payload['filename'] == 'REC0.wav'
  assert len(payload['predicted_frames']) == CLASSES
  assert len(payload['predicted_clip']) == CLASSES
  assert len(payload['targets']) == CLASSES
  assert payload['targets'][3] == 1.0
  assert isinstance(payload['lraps_score_clip'], float)


def test_the_json_dump_max_pools_the_frames(tmp_path):
  '''The dump keeps the original's pooling, the artefact does not.'''
  callback = InferenceDumpCallback(tmp_path, write_json=True)
  outputs = _outputs(batch_size=1)
  callback.on_test_batch_end(None, None, outputs, _batch(batch_size=1), 0)
  payload = json.loads(callback.dumps[0].read_text(encoding='utf-8'))
  expected = outputs['y_pred']['framewise_output'][0].reshape(-1,
                                                               CLASSES)
  pooled = expected.max(dim=0)[0]
  assert np.allclose(payload['predicted_frames'], pooled.numpy(), atol=1e-6)
  artefact = load_score_artefact(callback.artefacts[0])
  assert artefact['frames'].shape[0] == SNIPPETS * FRAMES
