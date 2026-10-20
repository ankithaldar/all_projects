#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''The label space: 264 species codes and their class indices.

The original kept ``BIRD_CODE``, ``EBIRD_LABEL`` and ``INV_EBIRD_LABEL`` in
``src/config_params/configs.py``.  The index order is **the** label order
everywhere in the codebase, so it is data, not code, and it lives in
``bird_codes.json`` next to this module.

The committed JSON is a **placeholder** until
``scripts/build_bird_codes.py`` regenerates it from the competition's
``train.csv``.  The placeholder has the right shape -- 264 codes with
contiguous indices -- so that everything downstream is developable and
testable, but it is not the real label space and must not be used to train.
'''

import json
from pathlib import Path
from typing import Any

LABEL_SPACE_PATH = Path(__file__).resolve().parent / 'bird_codes.json'
NUM_CLASSES = 264

_BIRD_CODE: dict[str, int] = {}
_EBIRD_LABEL: dict[str, str] = {}
_INV_EBIRD_LABEL: dict[str, str] = {}
_INV_BIRD_CODE: dict[int, str] = {}


class AUDIO_CONFIG:
  '''The legacy audio configuration of the original.

  It was defined but never instantiated: the live dataset takes a
  ``period`` in seconds, not a ``duration``.
  '''

  sampling_rate = 32000
  duration = 1
  hop_length = 251
  fmin = 500
  fmax = 15000
  n_mels = 128
  n_fft = n_mels * 20
  samples = sampling_rate * duration
  padmode = 'reflect'


def _load_label_space() -> dict[str, Any]:
  '''Reads and validates the committed label space.

  Returns:
    The parsed JSON document.

  Raises:
    FileNotFoundError: If the label space has not been generated yet.
    ValueError: If the label space does not have 264 contiguous entries.
  '''
  if not LABEL_SPACE_PATH.exists():
    raise FileNotFoundError(
        f'Label space not found at {LABEL_SPACE_PATH}. Generate it with '
        '`python -m birdcall.scripts.build_bird_codes`.')
  document = json.loads(LABEL_SPACE_PATH.read_text(encoding='utf-8'))
  codes = document.get('codes', {})
  labels = document.get('labels', {})
  if len(codes) != NUM_CLASSES:
    raise ValueError(
        f'Label space has {len(codes)} codes, expected {NUM_CLASSES}. '
        'Regenerate it with `python -m birdcall.scripts.build_bird_codes`.')
  if sorted(codes.values()) != list(range(NUM_CLASSES)):
    raise ValueError('Label space indices are not contiguous from 0.')
  return document


def _initialise() -> None:
  '''Populates the module level label maps from the committed JSON.'''
  global _BIRD_CODE, _EBIRD_LABEL, _INV_EBIRD_LABEL, _INV_BIRD_CODE
  document = _load_label_space()
  _BIRD_CODE = dict(document['codes'])
  _EBIRD_LABEL = dict(document.get('labels', {}))
  _INV_EBIRD_LABEL = {
      label: code for code, label in _EBIRD_LABEL.items()
  }
  _INV_BIRD_CODE = {index: code for code, index in _BIRD_CODE.items()}


_initialise()

BIRD_CODE = _BIRD_CODE
EBIRD_LABEL = _EBIRD_LABEL
INV_EBIRD_LABEL = _INV_EBIRD_LABEL
INV_BIRD_CODE = _INV_BIRD_CODE


def get_bird_code() -> dict[str, int]:
  '''Returns the species code to class index map.

  Returns:
    A copy of the map, so a caller cannot corrupt the shared one.
  '''
  return dict(_BIRD_CODE)


def get_inv_ebird_label() -> dict[str, str]:
  '''Returns the class index to label map.

  Returns:
    A copy of the map, so a caller cannot corrupt the shared one.
  '''
  return dict(_INV_EBIRD_LABEL)


def get_inv_bird_code() -> dict[int, str]:
  '''Returns the class index to species code map.

  This is the map that turns a 264 wide score vector into the species
  codes a submission's ``labels`` column is built from.  The original
  defined it in its inference metric and never used it.

  Returns:
    A copy of the map, so a caller cannot corrupt the shared one.
  '''
  return dict(_INV_BIRD_CODE)
