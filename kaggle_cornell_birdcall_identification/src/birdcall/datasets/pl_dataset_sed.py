#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''The torch dataset of the SED pipeline.

The dataset turns a fold CSV and a tree of wav files into fixed-length
waveform snippets with three label tensors.  The framework-agnostic half
lives in ``datasets/sed_dataset_core.py``; this class adds the ``torch``
plumbing, the audio read and the per-item seeding.

The label is a **264-vector** one-hot broadcast to every snippet, not a
per-frame matrix: the weak file-level label says "this species is
somewhere in this window", so every snippet of the window carries it.

The per-item randomness is derived from ``(seed, epoch, index)``, so a
draw does not change when the number of dataloader workers changes.
'''

from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from birdcall.datasets.pl_sampler_imbalanced import ImbalancedDatasetSampler
from birdcall.datasets.sed_dataset_core import background_rows
from birdcall.datasets.sed_dataset_core import build_label_vector
from birdcall.datasets.sed_dataset_core import read_fold_rows
from birdcall.datasets.sed_dataset_core import resolve_audio_path
from birdcall.datasets.sed_dataset_core import slice_for_training
from birdcall.datasets.sed_dataset_core import slice_for_validation
from birdcall.helpers.pl_seed import worker_init_fn
from birdcall.helpers.seed import make_generator


class SedDataset(Dataset):
  '''The waveform dataset of the SED pipeline.

  Attributes:
    data: The flat row list.  A CSV row has the keys ``filename``,
      ``ebird_code`` and ``secondary_labels``; a background row has only
      ``filename`` and is a hard negative on every species.
    sampler: The sampler class the engine instantiates, exactly like the
      original did.
    additional_loader_params: The extra ``DataLoader`` kwargs, namely the
      worker init function that seeds numpy, ``random`` and torch.
  '''

  def __init__(self, root_dir, csv_dir, period, bird_code, inv_ebird_label,
               isTraining, num_test_samples, transform=None,
               background_audio_dir=None, seed: int = 42) -> None:
    '''Builds the dataset of one fold.

    Args:
      root_dir: The audio root, ``root_dir/{ebird_code}/{filename}.wav``.
      csv_dir: The fold CSV.
      period: The window length in seconds.
      bird_code: The species code to class index map.
      inv_ebird_label: The common name to species code map, used to turn
        the secondary labels into codes.
      isTraining: Whether to crop randomly (train) or slice (valid).
      num_test_samples: The number of snippets per item.
      transform: The waveform transform, called as ``transform(x, sr)``.
      background_audio_dir: A directory of label-less background wavs.
      seed: The root seed of the per-item randomness.
    '''
    self.root_dir = Path(root_dir)
    self.csv_dir = Path(csv_dir)
    self.period = period
    self.bird_code = bird_code
    self.inv_ebird_label = inv_ebird_label
    self.isTraining = isTraining
    self.num_test_samples = num_test_samples
    self.transform = transform
    self.background_audio_dir = background_audio_dir
    self.seed = seed
    self.epoch = 0
    self.data = read_fold_rows(self.csv_dir)
    self.data.extend(background_rows(background_audio_dir))
    self.sampler = ImbalancedDatasetSampler
    self.additional_loader_params = {'worker_init_fn': self.init_workers_fn}

  def set_epoch(self, epoch: int) -> None:
    '''Sets the epoch the next draws belong to.

    Args:
      epoch: The epoch index.
    '''
    self.epoch = epoch

  def init_workers_fn(self, worker_id: int) -> None:
    '''Seeds a dataloader worker.

    Args:
      worker_id: The worker index.
    '''
    worker_init_fn(worker_id, self.seed, self.epoch)

  def get_label(self, dataset, idx) -> str:
    '''Returns the primary species code of a row.

    The original read ``dataset.data[idx]['ebird_code']`` and raised a
    ``KeyError`` on a background row; the empty string is returned here
    instead, so the sampler can be built on a dataset that has background
    audio.

    Args:
      dataset: The dataset the row belongs to.
      idx: The row index.

    Returns:
      The primary species code, empty for a background row.
    '''
    return str(dataset.data[idx].get('ebird_code', ''))

  def _read_audio(self, row: dict) -> tuple[np.ndarray, int]:
    '''Reads the wav a row points at.

    Args:
      row: The dataset row.

    Returns:
      The samples and the sample rate of the file.
    '''
    audio, sample_rate = sf.read(resolve_audio_path(self.root_dir, row))
    samples = np.asarray(audio)
    if samples.ndim > 1:
      # the original assumed mono and broke on stereo files
      samples = samples.mean(axis=1)
    return samples, sample_rate

  def _label_codes(self, row: dict) -> tuple[list[str], list[str], list[str]]:
    '''Turns a row into its three label sets.

    Args:
      row: The dataset row.

    Returns:
      The all, primary and secondary species codes.  A background row
      has none of them.
    '''
    if 'ebird_code' not in row:
      return [], [], []
    primary = str(row['ebird_code'])
    all_codes = [primary]
    for name in row.get('secondary_labels', []):
      code = self.inv_ebird_label.get(name)
      if code is not None:
        all_codes.append(code)
    secondary_codes = [code for code in all_codes if code != primary]
    return all_codes, [primary], secondary_codes

  def __len__(self) -> int:
    '''Returns the number of rows.

    Returns:
      The size of the dataset.
    '''
    return len(self.data)

  def __getitem__(self, idx: int) -> dict:
    '''Returns one item of the dataset.

    Args:
      idx: The row index.

    Returns:
      A dict with the keys ``filename``, ``waveforms`` ``(C, S)`` float32,
      ``all_labels``, ``primary_labels`` and ``secondary_labels``, the
      three label tensors being ``(C, 264)`` float32.
    '''
    row = self.data[idx]
    audio, sample_rate = self._read_audio(row)
    if self.transform is not None:
      audio = self.transform(audio, sample_rate)
    length = sample_rate * self.period
    if self.isTraining:
      rng = make_generator(self.seed, self.epoch, idx)
      snippets = [slice_for_training(audio, length, rng)
                  for _ in range(self.num_test_samples)]
    else:
      snippets = slice_for_validation(audio, length, self.num_test_samples)
    waveforms = np.stack(snippets).astype(np.float32)
    all_codes, primary_codes, secondary_codes = self._label_codes(row)
    filename = str(row['filename'])
    if 'ebird_code' not in row:
      filename = Path(filename).stem
    return {
        'filename': filename,
        'waveforms': waveforms,
        'all_labels': self._label_matrix(all_codes, len(snippets)),
        'primary_labels': self._label_matrix(primary_codes, len(snippets)),
        'secondary_labels': self._label_matrix(secondary_codes, len(snippets)),
    }

  def _label_matrix(self, codes: list[str], count: int) -> torch.Tensor:
    '''Builds the label tensor of a row.

    Args:
      codes: The species codes of the label set.
      count: The number of snippets the tensor holds.

    Returns:
      A ``(count, 264)`` float32 tensor, the same one-hot vector on every
      row.
    '''
    vector = build_label_vector(codes, self.bird_code)
    return torch.from_numpy(np.tile(vector, (count, 1))).float()
