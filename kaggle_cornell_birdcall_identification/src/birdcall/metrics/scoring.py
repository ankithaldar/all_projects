#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''The score artefact: the handoff between a model and the vote.

The original's ``SedTest`` metric wrote one JSON per audio file, but it
max-pooled the **entire time axis** away before writing, so a ten minute
file produced a single record instead of the 120 five second rows the
vote needs.  Temporal information was structurally unrecoverable from it.

This module defines the artefact that fixes that: a ``.npz`` per audio file
per model, holding the clip scores and the **frame scores with their time
axis intact**.  The vote consumes it, and it is framework agnostic, so both
backends write the same format and the resulting submission can be compared
byte for byte.

The original's JSON dump is also reproduced here, with its four documented
blockers fixed: the missing comma, the undefined ``save_dir`` and
``combine_files`` config attributes, and the ``save_dir`` string that had no
``mkdir``.
'''

import json
from pathlib import Path

import numpy as np

FRAME_KEY = 'frames'
CLIP_KEY = 'clip'
AUDIO_ID_KEY = 'audio_id'
FRAME_RATE_KEY = 'frame_rate'
NUM_CLASSES_KEY = 'num_classes'
VERSION_KEY = 'version'

ARTEFACT_VERSION = 1
DEFAULT_FRAME_RATE = 100.0
NUM_CLASSES = 264

JSON_KEYS = (
    'filename',
    'predicted_frames',
    'predicted_clip',
    'targets',
    'lraps_score_clip',
    'lraps_score_frames',
)


def save_score_artefact(path: Path, audio_id: str, clip: np.ndarray,
                        frames: np.ndarray, frame_rate: float) -> Path:
  '''Writes one model's scores for one audio file.

  Args:
    path: The directory the artefact is written to.
    audio_id: The identifier of the audio file.
    clip: The clip level scores, shape ``(num_classes,)``.
    frames: The frame level scores, shape ``(num_frames, num_classes)``.
      The time axis must be preserved.
    frame_rate: The number of frames per second.

  Returns:
    The path the artefact was written to.
  '''
  path = Path(path)
  path.mkdir(parents=True, exist_ok=True)
  target = path / f'{audio_id}.npz'
  np.savez(
      target,
      **{CLIP_KEY: np.asarray(clip, dtype=np.float32)},
      **{FRAME_KEY: np.asarray(frames, dtype=np.float32)},
      **{AUDIO_ID_KEY: np.asarray(audio_id)},
      **{FRAME_RATE_KEY: np.asarray(frame_rate, dtype=np.float64)},
      **{NUM_CLASSES_KEY: np.asarray(NUM_CLASSES, dtype=np.int64)},
      **{VERSION_KEY: np.asarray(ARTEFACT_VERSION, dtype=np.int64)})
  return target


def load_score_artefact(path: Path) -> dict:
  '''Reads one model's scores for one audio file.

  Args:
    path: The path of the ``.npz`` artefact.

  Returns:
    A dictionary with the keys ``audio_id``, ``clip``, ``frames`` and
    ``frame_rate``.
  '''
  with np.load(Path(path), allow_pickle=False) as data:
    return {
        'audio_id': str(data[AUDIO_ID_KEY]),
        'clip': data[CLIP_KEY],
        'frames': data[FRAME_KEY],
        'frame_rate': float(data[FRAME_RATE_KEY]),
    }


def write_json_dump(path: Path, filename: str, predicted_frames: np.ndarray,
                    predicted_clip: np.ndarray, targets: np.ndarray,
                    lraps_score_clip: float,
                    lraps_score_frames: float) -> Path:
  '''Writes the original's per-file JSON dump.

  The key set is byte identical to the original's, so the two artefacts can
  be diffed against each other.

  Args:
    path: The directory the JSON is written to.
    filename: The name of the audio file.
    predicted_frames: The 264 frame scores, max-pooled over time.
    predicted_clip: The 264 clip scores.
    targets: The 264 label values.
    lraps_score_clip: The per file label ranking score on the clip head.
    lraps_score_frames: The per file label ranking score on the frame head.

  Returns:
    The path the JSON was written to.
  '''
  path = Path(path)
  path.mkdir(parents=True, exist_ok=True)
  payload = {
      'filename': filename,
      'predicted_frames': np.asarray(predicted_frames).tolist(),
      'predicted_clip': np.asarray(predicted_clip).tolist(),
      'targets': np.asarray(targets).tolist(),
      'lraps_score_clip': float(lraps_score_clip),
      'lraps_score_frames': float(lraps_score_frames),
  }
  target = path / f'{filename}.json'
  target.write_text(json.dumps(payload), encoding='utf-8')
  return target


def merge_json_dumps(path: Path) -> Path:
  '''Merges every per-file JSON dump into one JSON array.

  The original built the array by concatenating the raw file texts.  That
  is reproduced, because the output has to stay byte comparable, but the
  file handles are closed and a truncated file is detected rather than
  silently corrupting the array.

  Args:
    path: The directory holding the per-file JSON dumps.

  Returns:
    The path of the merged file.
  '''
  path = Path(path)
  final = path / 'final'
  final.mkdir(parents=True, exist_ok=True)
  texts = []
  for source in sorted(path.glob('*.json')):
    text = source.read_text(encoding='utf-8')
    if not text.strip().startswith('{') or not text.strip().endswith('}'):
      raise ValueError(f'Truncated JSON dump: {source}')
    texts.append(text)
  target = final / 'merged_results.json'
  target.write_text('[{}]'.format(','.join(texts)), encoding='utf-8')
  return target
