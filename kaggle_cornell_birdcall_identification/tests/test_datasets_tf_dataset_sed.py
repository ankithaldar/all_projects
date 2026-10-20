#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''The tests of the Keras dataset pipeline.

The tests build a tiny tree of wav files and a fold CSV, so no
competition data is needed.  The behaviours that are asserted are the
ones that change the experiment: the order of the transform and the
slicing, the shape of the label, the determinism of the validation
split, and the cardinality of the sampler.
'''

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from birdcall.augmentations.sed_augment_background import (
    get_transforms as background_transforms)
from birdcall.augmentations.sed_augment_default import get_transforms
from birdcall.datasets.sed_dataset_core import build_label_vector
from birdcall.datasets.tf_dataset_sed import SedDataset
from birdcall.helpers.seed import derive_seed
from birdcall.helpers.seed import make_generator

SR = 32000
PERIOD = 1
BIRD_CODE = {'aldfly': 0, 'ameavo': 1, 'amered': 2}
INV_EBIRD_LABEL = {'Alder Flycatcher': 'aldfly', 'American Avocet': 'ameavo',
                   'American Redstart': 'amered'}


def write_wav(path, seconds: float = 1.0, seed: int = 1) -> None:
  '''Writes a synthetic wav file.

  Args:
    path: The path of the file.
    seconds: The length of the file, in seconds.
    seed: The seed of the generator.
  '''
  path.parent.mkdir(parents=True, exist_ok=True)
  rng = np.random.default_rng(seed)
  samples = rng.standard_normal(int(SR * seconds)).astype(np.float32)
  sf.write(path, samples, SR)


@pytest.fixture(name='tree')
def tree_fixture(tmp_path):
  '''Builds a tiny audio tree and a fold CSV.

  Args:
    tmp_path: The temporary directory of the test.

  Returns:
    A tuple ``(root, csv, background directory)``.
  '''
  root = tmp_path / 'audio'
  rows = []
  for index, code in enumerate(['aldfly', 'ameavo', 'amered']):
    write_wav(root / code / f'clip{index}.wav', seconds=1.5, seed=index)
    rows.append({'filename': f'clip{index}.mp3', 'ebird_code': code,
                 'secondary_labels': "[]"})
  write_wav(root / 'aldfly' / 'again.wav', seconds=0.5, seed=9)
  rows.append({'filename': 'again.mp3', 'ebird_code': 'aldfly',
               'secondary_labels': "['American Avocet']"})
  csv = tmp_path / 'fold.csv'
  pd.DataFrame(rows).to_csv(csv, index=False)
  background = tmp_path / 'background'
  write_wav(background / 'noise.wav', seconds=1.0, seed=11)
  return root, csv, background


def build(root, csv, is_training: bool, num_test_samples: int = 2,
          background=None, transform=None) -> SedDataset:
  '''Builds a dataset of the test tree.

  Args:
    root: The audio root.
    csv: The fold CSV.
    is_training: Whether the split is the training one.
    num_test_samples: The number of snippets per item.
    background: The directory of background wavs.
    transform: The waveform transform.

  Returns:
    The dataset.
  '''
  return SedDataset(root_dir=root, csv_dir=csv, period=PERIOD,
                    bird_code=BIRD_CODE, inv_ebird_label=INV_EBIRD_LABEL,
                    isTraining=is_training, num_test_samples=num_test_samples,
                    transform=transform, background_audio_dir=background,
                    seed=42)


def items(dataset: SedDataset) -> list:
  '''Reads every item of a dataset.

  Args:
    dataset: The dataset.

  Returns:
    The list of items.
  '''
  return list(dataset.items().as_numpy_iterator())


def test_the_item_holds_the_waveform_and_the_labels(tree) -> None:
  '''The item holds the snippets and the three label tensors.'''
  root, csv, _ = tree
  dataset = build(root, csv, is_training=False)
  item = items(dataset)[0]
  assert set(item) >= {'filename', 'waveforms', 'all_labels',
                        'primary_labels', 'secondary_labels'}
  assert item['waveforms'].shape == (2, SR * PERIOD)
  assert item['all_labels'].shape == (2, 264)
  assert item['primary_labels'].shape == (2, 264)
  assert item['secondary_labels'].shape == (2, 264)


def test_the_label_is_a_264_vector_broadcast_to_every_snippet(tree) -> None:
  '''The weak file level label is every snippet's label.'''
  root, csv, _ = tree
  dataset = build(root, csv, is_training=False)
  item = items(dataset)[0]
  vector = item['all_labels'][0]
  assert vector.shape == (264,)
  assert vector.sum() == 1.0
  assert vector[BIRD_CODE['aldfly']] == 1.0
  # The same one-hot on every snippet.
  assert np.array_equal(item['all_labels'][0], item['all_labels'][1])


def test_the_secondary_labels_are_kept_apart(tree) -> None:
  '''The secondary species are in the secondary tensor only.'''
  root, csv, _ = tree
  dataset = build(root, csv, is_training=False)
  item = items(dataset)[3]
  assert item['all_labels'][0].sum() == 2.0
  assert item['primary_labels'][0].sum() == 1.0
  assert item['secondary_labels'][0].sum() == 1.0
  assert item['secondary_labels'][0][BIRD_CODE['ameavo']] == 1.0


def test_the_background_rows_have_no_label(tree) -> None:
  '''A background row is a hard negative on every species.'''
  root, csv, background = tree
  dataset = build(root, csv, is_training=False, background=background)
  item = items(dataset)[-1]
  assert item['all_labels'].sum() == 0.0
  assert item['primary_labels'].sum() == 0.0


def test_the_validation_split_is_deterministic(tree) -> None:
  '''Two reads of the validation split are identical.'''
  root, csv, _ = tree
  dataset = build(root, csv, is_training=False)
  first = items(dataset)[0]['waveforms']
  second = items(dataset)[0]['waveforms']
  assert np.array_equal(first, second)


def test_the_training_split_crops_at_a_random_offset(tree) -> None:
  '''The training window is not the deterministic validation window.'''
  root, csv, _ = tree
  training = build(root, csv, is_training=True)
  validation = build(root, csv, is_training=False)
  trained = items(training)[0]['waveforms']
  validated = items(validation)[0]['waveforms']
  assert not np.array_equal(trained, validated)
  # The per-item randomness is derived from (seed, epoch, index), so a
  # second read of the same epoch is the same draw.
  assert np.array_equal(trained, items(training)[0]['waveforms'])


def test_the_transform_runs_on_the_whole_file_before_the_slicing(
    tree) -> None:
  '''The transform sees the whole recording, not a snippet.'''
  root, csv, _ = tree
  seen = []

  def transform(x, sr):
    seen.append(len(x))
    return x

  dataset = build(root, csv, is_training=False, transform=transform)
  items(dataset)
  # The first recording is 1.5 seconds long, and the snippet is one
  # second: the transform saw the whole file.
  assert seen[0] == int(SR * 1.5)
  assert seen[0] != SR * PERIOD


def test_the_cardinality_is_known_before_the_sampler(tree) -> None:
  '''The cardinality is the number of rows, not UNKNOWN.'''
  root, csv, background = tree
  dataset = build(root, csv, is_training=True, background=background)
  assert dataset.cardinality() == 5
  assert dataset.items().cardinality().numpy() == 5


def test_the_sampler_balances_the_species(tree) -> None:
  '''The declarative sampler draws the rare species more often.'''
  root, csv, _ = tree
  dataset = build(root, csv, is_training=True, num_test_samples=1)
  counts = {}
  for _ in range(20):
    for _, y in dataset.build(batch_size=1, balanced=True):
      label = int(np.argmax(y['all_labels'][0, 0].numpy()))
      counts[label] = counts.get(label, 0) + 1
  # Three species, and the sampler must not starve any of them.
  assert len(counts) == 3


def test_the_index_space_is_sharded(tree) -> None:
  '''Two shards hold disjoint rows.'''
  root, csv, _ = tree
  dataset = build(root, csv, is_training=False)
  first = {str(x['filename'].numpy()) for x, _ in
           dataset.build(batch_size=1, balanced=False, num_shards=2,
                         shard_index=0)}
  second = {str(x['filename'].numpy()) for x, _ in
            dataset.build(batch_size=1, balanced=False, num_shards=2,
                          shard_index=1)}
  assert first and second
  assert not (first & second)


def test_the_batched_dataset_yields_the_input_and_the_labels(tree) -> None:
  '''The batched dataset yields the waveform and the label dict.'''
  root, csv, _ = tree
  dataset = build(root, csv, is_training=False)
  batch = next(iter(dataset.build(batch_size=2, balanced=False)))
  x, y = batch
  assert x['waveforms'].shape == (2, 2, SR * PERIOD)
  assert y['all_labels'].shape == (2, 2, 264)
  assert y['primary_labels'].shape == (2, 2, 264)
  assert y['secondary_labels'].shape == (2, 2, 264)


def test_the_epoch_advances_the_per_item_randomness(tree) -> None:
  '''The per-item seed depends on the epoch.'''
  root, csv, _ = tree
  dataset = build(root, csv, is_training=True)
  dataset.set_epoch(0)
  first = items(dataset)[0]['waveforms']
  dataset.set_epoch(1)
  second = items(dataset)[0]['waveforms']
  assert not np.array_equal(first, second)


def test_the_primary_label_is_readable(tree) -> None:
  '''The sampler balances on the primary label.'''
  root, csv, _ = tree
  dataset = build(root, csv, is_training=False)
  assert dataset.get_label(dataset, 0) == 'aldfly'
  assert dataset.get_label(dataset, 1) == 'ameavo'


def test_the_label_vector_is_the_shared_one() -> None:
  '''The label coding is the shared, framework agnostic one.'''
  vector = build_label_vector(['ameavo'], BIRD_CODE)
  assert vector.shape == (264,)
  assert vector[BIRD_CODE['ameavo']] == 1.0
  assert vector.sum() == 1.0


def test_the_per_item_randomness_is_seeded() -> None:
  '''The per-item draw comes from the shared seed derivation.'''
  first = make_generator(42, 0, 3).integers(0, 1000)
  second = make_generator(42, 0, 3).integers(0, 1000)
  assert first == second
  # The epoch is part of the derivation, so the draw moves with it.
  assert derive_seed(42, 0, 3) != derive_seed(42, 1, 3)


def test_the_augmentation_stack_is_imported_lazily() -> None:
  '''The waveform stacks need audiomentations, which may be missing.'''
  for builder in (get_transforms, background_transforms):
    with pytest.raises(ImportError):
      builder()
