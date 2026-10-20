#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''The session wide import order.

The cross-backend parity tests need **both** frameworks in one process,
and the order of the imports is not free: importing ``torchvision``
after ``tensorflow`` segfaults the interpreter, because torchvision
pulls in triton, which does not survive the TensorFlow runtime in the
same process.

The import is therefore done here, in the conftest of the test
directory, which is loaded before any test module is imported.  Every
test module may then import the frameworks in any order.
'''

import torch  # noqa: F401  pylint: disable=unused-import
import torchvision  # noqa: F401  pylint: disable=unused-import

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from birdcall.config_params.configs import get_bird_code
from birdcall.config_params.configs import get_inv_ebird_label
from birdcall.datasets.pl_dataset_sed import SedDataset

LOSS_BATCH = 3
LOSS_SNIPPETS = 2
LOSS_CLASSES = 8
LOSS_SEED = 20240501


@pytest.fixture
def loss_labels() -> dict:
  '''The label dict the criteria are fed with.

  Every item carries one primary and one secondary species, so the
  secondary masking of the criteria has something to mask.

  Returns:
    A dict with `all_labels` and `secondary_labels`.
  '''
  all_labels = torch.zeros(LOSS_BATCH, LOSS_SNIPPETS, LOSS_CLASSES)
  all_labels[:, :, 1] = 1.0
  all_labels[:, :, 3] = 1.0
  secondary = torch.zeros(LOSS_BATCH, LOSS_SNIPPETS, LOSS_CLASSES)
  secondary[:, :, 3] = 1.0
  return {'all_labels': all_labels, 'secondary_labels': secondary}


@pytest.fixture
def loss_prediction() -> torch.Tensor:
  '''A fixed, already sigmoided prediction.

  Returns:
    A `(batch, snippets, classes)` float32 tensor.
  '''
  generator = torch.Generator().manual_seed(LOSS_SEED)
  return (torch.rand(LOSS_BATCH, LOSS_SNIPPETS, LOSS_CLASSES,
                      generator=generator) * 0.9 + 0.05)


METRIC_SNIPPETS = 2
METRIC_CLASSES = 6
METRIC_FRAMES = 5


@pytest.fixture
def metric_output():
  '''Builds the output dict the metrics are fed with.

  Returns:
    A callable `(clip, frame=None, labels=None) -> dict`.
  '''
  def build(clip, frame=None, labels=None):
    batch = clip.shape[0]
    if frame is None:
      frame = torch.zeros(batch, METRIC_SNIPPETS, METRIC_FRAMES,
                          METRIC_CLASSES)
    if labels is None:
      labels = torch.zeros(batch, METRIC_SNIPPETS, METRIC_CLASSES)
    return {'y_pred': {'clipwise_output': clip, 'framewise_output': frame},
            'y': {'all_labels': labels}}
  return build


@pytest.fixture
def metric_labels():
  '''Builds a label dict from one list of positive classes per item.

  Returns:
    A callable `(*rows) -> torch.Tensor`.
  '''
  def build(*rows):
    labels = torch.zeros(len(rows), METRIC_SNIPPETS, METRIC_CLASSES)
    for index, positives in enumerate(rows):
      for positive in positives:
        labels[index, :, positive] = 1.0
    return labels
  return build


FOLD_SAMPLE_RATE = 32000
FOLD_DURATION = 0.2
FOLD_PERIOD = 1


@pytest.fixture
def fold_species():
  '''The first three species of the label space.

  Returns:
    The three species codes, sorted.
  '''
  bird_code = get_bird_code()
  return sorted(bird_code)[:3]


@pytest.fixture
def fold_secondary_name(fold_species):
  '''The common name of the second species of the label space.

  Returns:
    The common name, as the CSV spells it.
  '''
  label_of = {code: name
              for name, code in get_inv_ebird_label().items()}
  return label_of[fold_species[1]]


@pytest.fixture
def write_wav():
  '''Writes a short mono tone.

  Returns:
    A callable `(path, frequency) -> None`.
  '''
  def build(path, frequency: float) -> None:
    time = np.arange(int(FOLD_SAMPLE_RATE * FOLD_DURATION)) / FOLD_SAMPLE_RATE
    signal = 0.5 * np.sin(2.0 * np.pi * frequency * time)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, signal.astype(np.float32), FOLD_SAMPLE_RATE)
  return build


@pytest.fixture
def make_fold(write_wav, fold_species):
  '''Builds a synthetic fold: an audio tree and its CSV.

  Returns:
    A callable `(tmp_path, counts=(4, 2, 1)) -> (root, csv_path, rows)`.
  '''
  def build(tmp_path, counts=(4, 2, 1)):
    audio_root = tmp_path / 'audio'
    rows = []
    for species_index, count in enumerate(counts):
      for row_index in range(count):
        filename = f'REC{species_index}_{row_index}.mp3'
        write_wav(audio_root / fold_species[species_index] / filename.replace(
            '.mp3', '.wav'), 440.0 * (species_index + 1))
        rows.append({
            'filename': filename,
            'ebird_code': fold_species[species_index],
            'secondary_labels': '[]',
        })
    csv_path = tmp_path / 'fold_0.csv'
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return audio_root, csv_path, rows
  return build


@pytest.fixture
def make_dataset(make_fold):
  '''Builds a dataset over a synthetic fold.

  Returns:
    A callable that takes the pytest temporary directory and the options
    of the dataset, and returns the dataset.
  '''
  def build(tmp_path, is_training, counts=(4, 2, 1), background_dir=None,
            num_test_samples=1):
    audio_root, csv_path, _ = make_fold(tmp_path, counts)
    return SedDataset(root_dir=audio_root,
                      csv_dir=csv_path,
                      period=FOLD_PERIOD,
                      bird_code=get_bird_code(),
                      inv_ebird_label=get_inv_ebird_label(),
                      isTraining=is_training,
                      num_test_samples=num_test_samples,
                      background_audio_dir=background_dir,
                      seed=42)
  return build


AUGMENT_SAMPLE_RATE = 32000
AUGMENT_DURATION = 1.0


@pytest.fixture
def augment_signal() -> np.ndarray:
  '''Builds a one second test signal.

  Returns:
    A float32 waveform.
  '''
  time = np.arange(int(AUGMENT_SAMPLE_RATE * AUGMENT_DURATION)) / (
      AUGMENT_SAMPLE_RATE)
  return (0.5 * np.sin(2.0 * np.pi * 440.0 * time)).astype(np.float32)
