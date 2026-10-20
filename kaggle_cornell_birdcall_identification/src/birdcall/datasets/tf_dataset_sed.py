#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''The ``tf.data`` pipeline of the SED pipeline.

The dataset turns a fold CSV and a tree of wav files into fixed length
waveform snippets with three label tensors.  The framework agnostic half
lives in ``datasets/sed_dataset_core.py``; this class adds the
``tf.data`` plumbing, the audio read and the per-item seeding.

Four behaviours are inherited verbatim from the original, because each of
them changes the experiment:

* the **waveform transform runs on the whole file, before the slicing**.
  Augmenting per snippet is a different experiment.
* the label is a **264-vector** one-hot broadcast to every snippet, not
  a per-frame matrix: the weak file-level label says "this species is
  somewhere in this window", so every snippet of the window carries it.
* the **training** window is stochastic and the **validation** window is
  deterministic.
* the **imbalanced sampler** balances the species by inverse frequency.
  ``tf.data.Dataset.sample_from_datasets`` is the declarative
  equivalent of the torch sampler's ``multinomial`` draw.

Two ``tf.data`` traps are handled here:

* ``sample_from_datasets`` and ``filter`` make the cardinality
  **unknown**, so the cardinality is computed **before** any lossy
  operation, and it is the number ``steps_per_epoch`` needs.
* ``sample_from_datasets`` does **not** shard across replicas, so the
  index space is sharded by hand when a strategy is in use.
'''

import logging
from pathlib import Path

import numpy as np
import soundfile as sf
import tensorflow as tf

from birdcall.datasets.sed_dataset_core import background_rows
from birdcall.datasets.sed_dataset_core import build_label_vector
from birdcall.datasets.sed_dataset_core import read_fold_rows
from birdcall.datasets.sed_dataset_core import resolve_audio_path
from birdcall.datasets.sed_dataset_core import slice_for_training
from birdcall.datasets.sed_dataset_core import slice_for_validation
from birdcall.helpers.seed import make_generator

LOGGER = logging.getLogger(__name__)


class SedDataset:
  '''The waveform dataset of the SED pipeline, as a ``tf.data`` pipeline.

  Attributes:
    rows: The flat row list.  A CSV row has the keys ``filename``,
      ``ebird_code`` and ``secondary_labels``; a background row has only
      ``filename`` and is a hard negative on every species.
    epoch: The epoch the per-item randomness belongs to.
  '''

  def __init__(self, root_dir, csv_dir, period, bird_code, inv_ebird_label,
               isTraining, num_test_samples, transform=None,
               background_audio_dir=None, seed: int = 42,
               num_classes: int = 264) -> None:
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
        It is applied to the **whole file**, before the slicing.
      background_audio_dir: A directory of label-less background wavs.
      seed: The root seed of the per-item randomness.
      num_classes: The width of the label vector.
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
    self.num_classes = num_classes
    self.epoch = 0
    self.rows = read_fold_rows(self.csv_dir)
    self.rows.extend(background_rows(self.background_audio_dir))
    self._paths = [str(resolve_audio_path(self.root_dir, row))
                   for row in self.rows]
    self._labels = [self._label_sets(row) for row in self.rows]
    self._filenames = [self._filename(row) for row in self.rows]
    self._primary_codes = [self._primary_label(row) for row in self.rows]

  def set_epoch(self, epoch: int) -> None:
    '''Sets the epoch the next draws belong to.

    Args:
      epoch: The epoch index.
    '''
    self.epoch = epoch

  def get_label(self, dataset, idx) -> str:
    '''Returns the primary species code of a row.

    Args:
      dataset: The dataset the row belongs to.
      idx: The row index.

    Returns:
      The primary species code, empty for a background row.
    '''
    del dataset
    return self._primary_codes[idx]

  def _filename(self, row: dict) -> str:
    '''Returns the name of the file of a row.

    Args:
      row: The dataset row.

    Returns:
      The file name, without the extension for a background row.
    '''
    filename = str(row['filename'])
    if 'ebird_code' not in row:
      return Path(filename).stem
    return filename

  def _primary_label(self, row: dict) -> str:
    '''Returns the primary species code of a row.

    Args:
      row: The dataset row.

    Returns:
      The primary species code, empty for a background row.
    '''
    if 'ebird_code' not in row:
      return ''
    return str(row['ebird_code'])

  def _label_sets(self, row: dict) -> tuple[np.ndarray, np.ndarray,
                                            np.ndarray]:
    '''Turns a row into its three label vectors.

    Args:
      row: The dataset row.

    Returns:
      The all, primary and secondary label vectors, each of width
      ``num_classes``.
    '''
    if 'ebird_code' not in row:
      zeros = np.zeros(self.num_classes, dtype=np.float32)
      return zeros, zeros.copy(), zeros.copy()
    primary = str(row['ebird_code'])
    all_codes = [primary]
    for name in row.get('secondary_labels', []):
      code = self.inv_ebird_label.get(name)
      if code is not None:
        all_codes.append(code)
    secondary_codes = [code for code in all_codes if code != primary]
    return (build_label_vector(all_codes, self.bird_code),
            build_label_vector([primary], self.bird_code),
            build_label_vector(secondary_codes, self.bird_code))

  @staticmethod
  def _as_path(path) -> str:
    '''Turns the path of a row into a string.

    Args:
      path: The path, as a tensor, as bytes or as a string.

    Returns:
      The path, as a string.
    '''
    if hasattr(path, 'numpy'):
      path = path.numpy()
    if isinstance(path, np.ndarray):
      path = path.item() if path.ndim == 0 else path
    if isinstance(path, bytes):
      return path.decode('utf-8')
    return str(path)

  def _read_audio(self, path: str) -> tuple[np.ndarray, int]:
    '''Reads a wav file.

    Args:
      path: The path of the file.

    Returns:
      The samples and the sample rate of the file.
    '''
    audio, sample_rate = sf.read(path)
    samples = np.asarray(audio)
    if samples.ndim > 1:
      samples = samples.mean(axis=1)
    return samples, sample_rate

  def _apply_transform(self, audio: np.ndarray, sample_rate: int):
    '''Applies the waveform transform to the whole file.

    Args:
      audio: The samples of the file.
      sample_rate: The sample rate of the file.

    Returns:
      The transformed samples, of the same length.
    '''
    if self.transform is None:
      return audio
    try:
      return self.transform(audio, sample_rate)
    except ImportError as error:
      # audiomentations is optional: it pins an old librosa, which
      # conflicts with the mel matrix build.
      LOGGER.warning('Skipping the waveform transform: %s', error)
      return audio

  def _build_waveforms(self, path: str, index: int) -> np.ndarray:
    '''Reads, transforms and slices one recording.

    Args:
      path: The path of the file.
      index: The index of the row, which seeds the per-item randomness.

    Returns:
      A ``(num_test_samples, samples)`` float32 array.
    '''
    audio, sample_rate = self._read_audio(self._as_path(path))
    audio = self._apply_transform(audio, sample_rate)
    length = sample_rate * self.period
    rng = make_generator(self.seed, self.epoch, index)
    if self.isTraining:
      snippets = [slice_for_training(audio, length, rng)
                  for _ in range(self.num_test_samples)]
    else:
      snippets = slice_for_validation(audio, length, self.num_test_samples)
    return np.stack(snippets).astype(np.float32)

  def _row_dataset(self) -> tf.data.Dataset:
    '''Builds the dataset of rows.

    Returns:
      A dataset of dicts with the keys ``path``, ``all_labels``,
      ``primary_labels``, ``secondary_labels`` and ``filename``.
    '''
    return tf.data.Dataset.from_tensor_slices({
        'path': self._paths,
        'all_labels': [labels[0] for labels in self._labels],
        'primary_labels': [labels[1] for labels in self._labels],
        'secondary_labels': [labels[2] for labels in self._labels],
        'filename': self._filenames,
    })

  def _map_row(self, index, item):
    '''Turns one row into one item.

    Args:
      index: The index of the row.
      item: The row.

    Returns:
      A dict with the keys ``filename``, ``waveforms``,
      ``all_labels``, ``primary_labels`` and ``secondary_labels``.
    '''
    waveforms = tf.py_function(self._build_waveforms,
                               [item['path'], index], tf.float32)
    waveforms.set_shape((self.num_test_samples, None))
    count = self.num_test_samples
    return {
        'index': index,
        'filename': item['filename'],
        'waveforms': waveforms,
        'all_labels': tf.tile(tf.expand_dims(item['all_labels'], 0),
                              [count, 1]),
        'primary_labels': tf.tile(tf.expand_dims(item['primary_labels'], 0),
                                  [count, 1]),
        'secondary_labels': tf.tile(
            tf.expand_dims(item['secondary_labels'], 0), [count, 1]),
    }

  def items(self, num_parallel_calls: int = 1) -> tf.data.Dataset:
    '''Returns the unbatched dataset of items.

    Args:
      num_parallel_calls: The parallelism of the map.

    Returns:
      A dataset of item dicts.
    '''
    dataset = self._row_dataset().enumerate()
    return dataset.map(self._map_row,
                       num_parallel_calls=num_parallel_calls)

  def cardinality(self) -> int:
    '''Returns the number of rows of the dataset.

    The cardinality is computed on the **row** dataset, before any lossy
    operation, because ``sample_from_datasets`` and ``filter`` make it
    unknown, and ``steps_per_epoch`` cannot always be inferred
    afterwards.

    Returns:
      The number of rows.
    '''
    return len(self.rows)

  def _balanced(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
    '''Resamples the dataset with inverse frequency weights.

    Args:
      dataset: The dataset of items.

    Returns:
      The resampled dataset.
    '''
    groups: dict[str, list[int]] = {}
    for index, code in enumerate(self._primary_codes):
      groups.setdefault(code, []).append(index)
    subsets = []
    for indices in groups.values():
      keep = tf.constant(sorted(indices), dtype=tf.int64)

      def predicate(item, keep=keep):
        return tf.reduce_any(tf.equal(item['index'], keep))

      subsets.append(dataset.filter(predicate))
    # Every species carries the same total weight, which is what the
    # inverse frequency draw of the original does.
    weights = [1.0] * len(subsets)
    return tf.data.Dataset.sample_from_datasets(subsets, weights=weights,
                                                stop_on_empty_dataset=False)

  def build(self, batch_size: int = 1, balanced: bool | None = None,
            num_shards: int = 1, shard_index: int = 0,
            drop_remainder: bool = False,
            num_parallel_calls: int = 1) -> tf.data.Dataset:
    '''Builds the batched dataset.

    Args:
      batch_size: The batch size.
      balanced: Whether the inverse frequency sampler runs.  The default
        is to balance the training split only.
      num_shards: The number of shards the index space is split into.
        The declarative sampler does not shard by itself.
      shard_index: The shard this replica reads.
      drop_remainder: Whether the last batch of an epoch is dropped.
      num_parallel_calls: The parallelism of the map.

    Returns:
      A dataset of ``(x, y)`` pairs, where ``x`` is the waveform dict
      and ``y`` is the label dict.
    '''
    if balanced is None:
      balanced = self.isTraining
    dataset = self.items()
    if num_shards > 1:
      dataset = dataset.shard(num_shards, shard_index)
    if balanced:
      dataset = self._balanced(dataset)
    else:
      dataset = dataset.shuffle(len(self.rows), seed=self.seed,
                                reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset.map(self._split_item)

  @staticmethod
  def _split_item(item):
    '''Splits an item into the model input and the labels.

    Args:
      item: The batched item.

    Returns:
      A tuple ``(x, y)``, where ``x`` is the waveform dict and ``y`` is
      the label dict.
    '''
    x = {'waveforms': item['waveforms'], 'filename': item['filename']}
    y = {
        'all_labels': item['all_labels'],
        'primary_labels': item['primary_labels'],
        'secondary_labels': item['secondary_labels'],
    }
    return x, y
