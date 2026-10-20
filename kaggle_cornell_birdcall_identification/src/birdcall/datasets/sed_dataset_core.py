#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''The framework-agnostic half of the SED dataset.

The original kept the CSV read, the path resolution, the label coding and
the window slicing in one ``torch.utils.data`` class.  Everything except
the ``torch`` plumbing lives here, so the Keras port can reuse it
unchanged.

Three behaviours are inherited verbatim from the original, because each of
them changes the experiment:

* ``secondary_labels`` is a **Python literal** in the CSV.  The original
  ran ``eval`` on it; this module runs ``ast.literal_eval``, which parses
  the same literals without executing code out of a data file.
* The **training** window places a short clip at a random offset inside
  the window and crops a long one at a random offset.  The offset range
  excludes the flush-right position, exactly like the original's
  ``np.random.randint(high)``.
* The **validation** window is a deterministic contiguous slice; a short
  file is zero-padded at the tail and a long one is silently truncated.

Every function that needs randomness takes an explicit generator, so no
global RNG is ever touched.
'''

import ast
from pathlib import Path

import numpy as np
import pandas as pd

NUM_CLASSES = 264
'''The width of the label vector: the competition has 264 species.'''


def _parse_secondary_labels(cell: object) -> list[str]:
  '''Parses one ``secondary_labels`` cell into a list of strings.

  Args:
    cell: The raw cell value.  ``None`` and ``NaN`` mean "no secondary
      label".

  Returns:
    The parsed list, empty when the cell is missing.

  Raises:
    ValueError: If the cell is neither missing nor a Python list literal.
  '''
  if cell is None:
    return []
  if not isinstance(cell, str) and bool(pd.isna(cell)):
    return []
  text = str(cell).strip()
  if not text:
    return []
  try:
    parsed = ast.literal_eval(text)
  except (ValueError, SyntaxError) as error:
    raise ValueError(
        f'secondary_labels is not a Python literal: {cell!r}') from error
  if not isinstance(parsed, (list, tuple)):
    raise ValueError(
        f'secondary_labels is not a list literal: {cell!r}')
  return [str(item) for item in parsed]


def read_fold_rows(csv_dir: Path) -> list[dict]:
  '''Reads a fold CSV into the flat row list the dataset indexes.

  Args:
    csv_dir: The fold CSV.  It must have the ``filename``, ``ebird_code``
      and ``secondary_labels`` columns; a missing ``secondary_labels``
      column is read as "no secondary label".

  Returns:
    One dict per row, with the keys ``filename``, ``ebird_code`` and
    ``secondary_labels``.
  '''
  frame = pd.read_csv(csv_dir)
  rows = []
  for record in frame.to_dict('records'):
    rows.append({
        'filename': str(record['filename']),
        'ebird_code': str(record['ebird_code']),
        'secondary_labels': _parse_secondary_labels(
            record.get('secondary_labels')),
    })
  return rows


def build_label_vector(codes: list[str],
                       bird_code: dict[str, int]) -> np.ndarray:
  '''Builds the one-hot label vector of a row.

  Args:
    codes: The species codes of the row, primary first.
    bird_code: The species code to class index map.

  Returns:
    A float32 vector of width ``max(264, len(bird_code))`` with a one at
    every known code.  Codes that are not in ``bird_code`` are skipped,
    which is what the original's ``in self.inv_ebird_label`` guard did.
  '''
  width = max(NUM_CLASSES, len(bird_code))
  vector = np.zeros(width, dtype=np.float32)
  for code in codes:
    index = bird_code.get(code)
    if index is None or index >= width:
      continue
    vector[index] = 1.0
  return vector


def slice_for_training(audio: np.ndarray, length: int,
                        rng: np.random.Generator) -> np.ndarray:
  '''Cuts one training window out of a recording.

  A clip shorter than the window is zero-padded and dropped at a random
  offset; a longer clip is cropped at a random offset.  The offset range
  is half-open, so the flush-right position is never drawn -- the
  original's ``np.random.randint(high)`` behaved the same way.

  Args:
    audio: The recording, one dimensional.
    length: The window length in samples.
    rng: The generator the offset is drawn from.

  Returns:
    The window, float32.
  '''
  samples = np.asarray(audio)
  total = samples.shape[0]
  if total < length:
    window = np.zeros(length, dtype=samples.dtype)
    start = int(rng.integers(0, length - total))
    window[start:start + total] = samples
    return window.astype(np.float32)
  if total > length:
    start = int(rng.integers(0, total - length))
    return samples[start:start + length].astype(np.float32)
  return samples.astype(np.float32)


def slice_for_validation(audio: np.ndarray, length: int,
                          num_samples: int) -> list[np.ndarray]:
  '''Cuts the validation windows out of a recording.

  The windows are contiguous and non-overlapping.  A short file is
  zero-padded at the tail of the buffer, a long one is silently
  truncated.

  Args:
    audio: The recording, one dimensional.
    length: The window length in samples.
    num_samples: The number of windows to return.

  Returns:
    ``num_samples`` windows, float32.
  '''
  samples = np.asarray(audio)
  total = length * num_samples
  if samples.shape[0] < total:
    buffer = np.zeros(total, dtype=samples.dtype)
    buffer[:samples.shape[0]] = samples
  else:
    buffer = samples[:total]
  return [buffer[index * length:(index + 1) * length].astype(np.float32)
          for index in range(num_samples)]


def resolve_audio_path(root_dir: Path, row: dict) -> Path:
  '''Resolves the file a row points at.

  The original rewrote ``.mp3`` to ``.wav`` and looked the file up under
  ``root_dir/{ebird_code}/{filename}``.  A background row carries no
  ``ebird_code`` and its filename is the path itself.

  Args:
    root_dir: The audio root.
    row: The dataset row.

  Returns:
    The path of the wav file.
  '''
  filename = str(row['filename'])
  if 'ebird_code' not in row:
    return Path(filename)
  return Path(root_dir) / str(row['ebird_code']) / filename.replace(
      '.mp3', '.wav')


def background_rows(background_audio_dir: Path | None) -> list[dict]:
  '''Lists the label-less background rows of a directory.

  Args:
    background_audio_dir: The directory of background wavs, or ``None``.

  Returns:
    One dict per wav, with the single key ``filename``.  The rows are
    hard negatives on every species.
  '''
  if background_audio_dir is None:
    return []
  paths = sorted(Path(background_audio_dir).glob('**/*.wav'))
  return [{'filename': str(path)} for path in paths]
