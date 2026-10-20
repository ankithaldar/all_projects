#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''The inference dump callback.

The original's ``SedTest`` metric wrote one JSON per audio file, and it
had four independent blockers: a missing comma, two configuration
attributes no config defines, and a string where a path was expected.
This callback writes the **score artefact** of
``birdcall.metrics.scoring``, which keeps the frame axis intact, and it
can also write the original's JSON dump.

The two artefacts are not the same thing, and the difference is the point
of the port:

* the **artefact** keeps the frame scores with their time axis, so the
  vote can cut them into five second rows;
* the **JSON dump** max-pools the frames over the entire time axis, which
  is the original's behaviour and the reason the original's output could
  never produce a five second row.
'''

import logging
from pathlib import Path

import numpy as np
from lightning.pytorch import Callback
from sklearn.metrics import label_ranking_average_precision_score

from birdcall.metrics.scoring import DEFAULT_FRAME_RATE
from birdcall.metrics.scoring import save_score_artefact
from birdcall.metrics.scoring import write_json_dump

LOGGER = logging.getLogger(__name__)


class InferenceDumpCallback(Callback):
  '''Writes the scores of every test file.

  Attributes:
    save_dir: The directory the artefacts are written to.
    write_json: Whether the original's JSON dump is written as well.
    frame_rate: The number of frames per second of the artefact.
    artefacts: The paths of the artefacts that were written.
    dumps: The paths of the JSON dumps that were written.
  '''

  def __init__(self, save_dir, write_json: bool = False,
               frame_rate: float = DEFAULT_FRAME_RATE) -> None:
    '''Builds the callback.

    Args:
      save_dir: The directory the artefacts are written to.
      write_json: Whether the original's JSON dump is written as well.
      frame_rate: The number of frames per second of the artefact.
    '''
    super().__init__()
    self.save_dir = Path(save_dir)
    self.write_json = write_json
    self.frame_rate = float(frame_rate)
    self.artefacts: list[Path] = []
    self.dumps: list[Path] = []

  def on_test_batch_end(self, trainer, pl_module, outputs, batch,
                        batch_idx) -> None:
    '''Writes the scores of one test batch.

    Args:
      trainer: The trainer.
      pl_module: The module of the run.
      outputs: What the test step returned.
      batch: The batch of the step.
      batch_idx: The index of the batch.
    '''
    if not outputs:
      return
    y_pred = outputs['y_pred']
    labels = outputs['y']
    filenames = labels.get('filename', batch.get('filename'))
    clipwise = y_pred['clipwise_output']
    framewise = y_pred['framewise_output']
    targets = labels['all_labels']
    for index, filename in enumerate(filenames):
      self._write_one(str(filename), clipwise[index], framewise[index],
                      targets[index])

  def _write_one(self, filename: str, clipwise, framewise, target) -> None:
    '''Writes the scores of one audio file.

    Args:
      filename: The name of the audio file.
      clipwise: The clip head of the file, ``(snippets, classes)``.
      framewise: The frame head of the file, ``(snippets, frames,
        classes)``.
      target: The labels of the file, ``(snippets, classes)``.
    '''
    clip = clipwise.max(dim=0)[0]
    frames = framewise.reshape(-1, framewise.shape[-1])
    label = target[0] if target.dim() > 1 else target
    artefact = save_score_artefact(self.save_dir, filename,
                                   clip.detach().cpu().numpy(),
                                   frames.detach().cpu().numpy(),
                                   self.frame_rate)
    self.artefacts.append(artefact)
    if not self.write_json:
      return
    pooled = frames.max(dim=0)[0]
    dump = write_json_dump(
        self.save_dir, filename,
        pooled.detach().cpu().numpy(),
        clip.detach().cpu().numpy(),
        label.detach().cpu().numpy(),
        self._lraps(label, clip),
        self._lraps(label, pooled))
    self.dumps.append(dump)

  def _lraps(self, target, prediction) -> float:
    '''Returns the label ranking score of one file.

    Args:
      target: The labels of the file.
      prediction: The scores of the file.

    Returns:
      The score, or zero when the file carries no label at all and the
      score is therefore undefined.
    '''
    try:
      return float(label_ranking_average_precision_score(
          [np.asarray(target)], [np.asarray(prediction)]))
    except ValueError as error:
      LOGGER.debug('The label ranking score is undefined: %s', error)
      return 0.0
