#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''The inference dump, as a Keras callback.

The original's ``SedTest`` metric wrote one JSON per audio file, and it
had four blockers: a missing comma, a ``save_dir`` that had to be a
``pathlib.Path``, two config attributes that no config defined, and an
undefined local in the test branch.  The JSON format itself is
reproduced byte for byte by the shared
``birdcall.metrics.scoring.write_json_dump``.

The dump **max-pools the entire time axis**, exactly like the original.
That is right for the dump and wrong for the vote, which needs the time
axis; so the score artefact is written next to it, with the frame axis
**intact**.  The artefact is what the shared vote consumes, and it is
framework agnostic, so both backends write the same format.
'''


import tensorflow as tf
import logging
from pathlib import Path

import numpy as np

from birdcall.metrics.scoring import merge_json_dumps
from birdcall.metrics.scoring import save_score_artefact
from birdcall.metrics.scoring import write_json_dump
from birdcall.metrics.tf_metric_lraps import lrap_scores


LOGGER = logging.getLogger(__name__)

DEFAULT_FRAME_RATE = 100.0
'''The number of frames per second of the log-mel front-end.'''

class InferenceDumpCallback(tf.keras.callbacks.Callback):
  '''Writes the per-file JSON dump and the score artefact.

  Attributes:
    save_dir: The directory the JSON dumps are written to.
    artefact_dir: The directory the score artefacts are written to.  It
      defaults to the JSON directory.
    combine_files: Whether the JSON dumps are merged into one file when
      the run ends.
    frame_rate: The number of frames per second of the front-end.
    predictor: The model the predictions come from.  The callback takes
      the model from the training loop when it is not given.
  '''

  def __init__(self, save_dir, dataset=None, model=None,
               combine_files: bool = False,
               frame_rate: float = DEFAULT_FRAME_RATE) -> None:
    '''Builds the callback.

    Args:
      save_dir: The directory the JSON dumps are written to.
      dataset: The dataset of items, yielding dicts with the keys
        ``filename``, ``waveforms`` and ``all_labels``.
      model: The model the predictions come from.
      combine_files: Whether the JSON dumps are merged into one file.
      frame_rate: The number of frames per second of the front-end.
    '''
    super().__init__()
    self.save_dir = Path(save_dir)
    self.artefact_dir = self.save_dir
    self.dataset = dataset
    self.predictor = model
    self.combine_files = combine_files
    self.frame_rate = frame_rate

  def set_model(self, model) -> None:
    '''Takes the model from the training loop.

    Args:
      model: The model.
    '''
    super().set_model(model)
    if self.predictor is None:
      self.predictor = model

  def _write(self, filename: str, clip: np.ndarray, frames: np.ndarray,
             target: np.ndarray) -> None:
    '''Writes the two artefacts of one file.

    Args:
      filename: The name of the audio file.
      clip: The clip level scores, shape ``(classes,)``.
      frames: The frame level scores, shape ``(frames, classes)``.
      target: The label vector, shape ``(classes,)``.
    '''
    pooled = frames.max(axis=0) if frames.size else frames
    clip_score = float(lrap_scores(target[None, :], clip[None, :])[0])
    frame_score = float(lrap_scores(target[None, :], pooled[None, :])[0])
    write_json_dump(self.save_dir, filename, pooled, clip, target,
                    clip_score, frame_score)
    save_score_artefact(self.artefact_dir, filename, clip, frames,
                        self.frame_rate)

  def on_test_end(self, logs=None) -> None:
    '''Runs the inference and writes the artefacts.

    Args:
      logs: The evaluation logs, unused.
    '''
    del logs
    model = self.model or self.predictor
    if self.dataset is None or model is None:
      LOGGER.warning('The inference dump needs both a dataset and a model.')
      return
    for batch in self.dataset:
      filenames = batch['filename'].numpy()
      waveforms = batch['waveforms']
      targets = batch['all_labels'].numpy()
      prediction = model(waveforms, training=False)
      clips = np.asarray(prediction['clipwise_output'])
      frames = np.asarray(prediction['framewise_output'])
      for index, name in enumerate(filenames):
        if isinstance(name, bytes):
          name = name.decode('utf-8')
        self._write(str(name), clips[index].max(axis=0),
                    frames[index].max(axis=0), targets[index, 0, :])
    if self.combine_files:
      merge_json_dumps(self.save_dir)
