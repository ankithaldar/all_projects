#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests of the label space.

The index order is **the** label order everywhere in the codebase, so it
is data, not code.  The committed JSON holds 264 codes with contiguous
indices.
'''

import pytest

from birdcall.config_params.configs import AUDIO_CONFIG
from birdcall.config_params.configs import NUM_CLASSES
from birdcall.config_params.configs import get_bird_code
from birdcall.config_params.configs import get_inv_bird_code
from birdcall.config_params.configs import get_inv_ebird_label


def test_the_label_space_has_264_codes():
  '''The competition has 264 species.'''
  bird_code = get_bird_code()
  assert len(bird_code) == NUM_CLASSES
  assert NUM_CLASSES == 264


def test_the_indices_are_contiguous_from_zero():
  '''The class indices are 0 to 263, with no gap.'''
  bird_code = get_bird_code()
  assert sorted(bird_code.values()) == list(range(NUM_CLASSES))


def test_the_inverse_label_map_turns_names_into_codes():
  '''The name to code map is the mirror of the code to index map.'''
  bird_code = get_bird_code()
  inv_ebird_label = get_inv_ebird_label()
  assert len(inv_ebird_label) == NUM_CLASSES
  assert set(inv_ebird_label.values()) == set(bird_code)


def test_the_inverse_bird_code_map_turns_indices_into_codes():
  '''A 264 wide score vector becomes species codes.'''
  inv_bird_code = get_inv_bird_code()
  bird_code = get_bird_code()
  assert len(inv_bird_code) == NUM_CLASSES
  for code, index in bird_code.items():
    assert inv_bird_code[index] == code


def test_the_maps_are_copies():
  '''A caller cannot corrupt the shared label space.'''
  bird_code = get_bird_code()
  bird_code['not_a_species'] = 999
  assert 'not_a_species' not in get_bird_code()


def test_the_legacy_audio_configuration_is_untouched():
  '''The original's legacy constants are still there.'''
  assert AUDIO_CONFIG.sampling_rate == 32000
  assert AUDIO_CONFIG.duration == 1
  assert AUDIO_CONFIG.n_mels == 128
  assert AUDIO_CONFIG.n_fft == AUDIO_CONFIG.n_mels * 20
  assert AUDIO_CONFIG.samples == AUDIO_CONFIG.sampling_rate * (
      AUDIO_CONFIG.duration)


def test_an_unknown_species_is_not_in_the_label_space():
  '''A code that is not in the space is simply absent.'''
  assert 'not_a_species' not in get_bird_code()
  with pytest.raises(KeyError):
    get_bird_code()['not_a_species']
