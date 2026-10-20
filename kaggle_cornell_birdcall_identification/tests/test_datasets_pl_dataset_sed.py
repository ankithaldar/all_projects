#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests of the torch dataset of the SED pipeline.

The audio is synthetic: a 0.2 second tone per row, written with
``soundfile`` into ``tmp_path``, plus a fold CSV written with pandas.

What matters here is the label: it is a **264-vector** one-hot broadcast
to every snippet, not a per-frame matrix, and the per-item randomness is
derived from `(seed, epoch, index)`, so the training draw changes with
the epoch while the validation draw does not.
'''

import random

import numpy as np
import pandas as pd
import torch

from birdcall.config_params.configs import get_bird_code
from birdcall.config_params.configs import get_inv_ebird_label
from birdcall.datasets.pl_dataset_sed import SedDataset

SAMPLE_RATE = 32000
PERIOD = 1


def test_training_draw_depends_on_the_epoch(tmp_path, make_dataset):
  '''The training window is placed at a random offset, per epoch.'''
  dataset = make_dataset(tmp_path, is_training=True)
  dataset.set_epoch(0)
  first = dataset[0]['waveforms']
  dataset.set_epoch(1)
  second = dataset[0]['waveforms']
  assert first.shape == second.shape == (1, SAMPLE_RATE * PERIOD)
  assert not np.array_equal(first, second)


def test_validation_draw_is_deterministic(tmp_path, make_dataset):
  '''The validation window is a deterministic contiguous slice.'''
  dataset = make_dataset(tmp_path, is_training=False)
  dataset.set_epoch(0)
  first = dataset[0]['waveforms']
  dataset.set_epoch(3)
  second = dataset[0]['waveforms']
  assert np.array_equal(first, second)


def test_validation_pads_at_the_tail(tmp_path, make_dataset):
  '''A short file is zero-padded at the tail of the buffer.'''
  dataset = make_dataset(tmp_path, is_training=False, num_test_samples=2)
  waveforms = dataset[0]['waveforms']
  assert waveforms.shape == (2, SAMPLE_RATE * PERIOD)
  assert np.count_nonzero(waveforms[0]) > 0
  assert np.count_nonzero(waveforms[1]) == 0


def test_label_tensor_is_a_broadcast_264_vector(tmp_path, make_dataset):
  '''The label is a 264-vector, the same one for every snippet.'''
  dataset = make_dataset(tmp_path, is_training=False, num_test_samples=2)
  item = dataset[0]
  for key in ('all_labels', 'primary_labels', 'secondary_labels'):
    assert item[key].shape == (2, 264)
    assert item[key].dtype == torch.float32
  assert item['all_labels'][0].equal(item['all_labels'][1])
  assert item['all_labels'][0].sum() == 1.0
  assert item['primary_labels'][0].sum() == 1.0
  assert item['secondary_labels'][0].sum() == 0.0


def test_secondary_labels_are_coded(tmp_path, make_fold, fold_secondary_name):
  '''A secondary species in the CSV becomes a one in the label vector.'''
  audio_root, csv_path, rows = make_fold(tmp_path, counts=(1, 0, 0))
  rows[0]['secondary_labels'] = f'["{fold_secondary_name}"]'
  pd.DataFrame(rows).to_csv(csv_path, index=False)
  dataset = SedDataset(root_dir=audio_root,
                        csv_dir=csv_path,
                        period=PERIOD,
                        bird_code=get_bird_code(),
                        inv_ebird_label=get_inv_ebird_label(),
                        isTraining=False,
                        num_test_samples=1)
  item = dataset[0]
  assert item['all_labels'].sum() == 2.0
  assert item['primary_labels'].sum() == 1.0
  assert item['secondary_labels'].sum() == 1.0


def test_background_rows_are_all_zero_and_do_not_raise(tmp_path, make_dataset,
                                                        write_wav):
  '''A background row is a hard negative on every species.'''
  background_dir = tmp_path / 'background'
  write_wav(background_dir / 'SSW_002.wav', 220.0)
  dataset = make_dataset(tmp_path, is_training=False,
                         background_dir=background_dir)
  background_index = len(dataset) - 1
  assert 'ebird_code' not in dataset.data[background_index]
  item = dataset[background_index]
  assert item['filename'] == 'SSW_002'
  for key in ('all_labels', 'primary_labels', 'secondary_labels'):
    assert item[key].shape == (1, 264)
    assert item[key].sum() == 0.0
  assert dataset.get_label(dataset, background_index) == ''


def test_worker_init_seeds_the_global_rngs(tmp_path, make_dataset):
  '''The worker init function seeds numpy, random and torch.'''
  dataset = make_dataset(tmp_path, is_training=True)
  assert 'worker_init_fn' in dataset.additional_loader_params
  dataset.init_workers_fn(0)
  first = (np.random.rand(), random.random(), torch.rand(1).item())
  dataset.init_workers_fn(1)
  second = (np.random.rand(), random.random(), torch.rand(1).item())
  assert first != second


def test_the_sampler_class_is_published(tmp_path, make_dataset):
  '''The dataset names the sampler class the engine instantiates.'''
  from birdcall.datasets.pl_sampler_imbalanced import ImbalancedDatasetSampler
  dataset = make_dataset(tmp_path, is_training=True)
  assert dataset.sampler is ImbalancedDatasetSampler
