#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests of the default waveform augmentation stack.

``audiomentations`` is not installed in this environment -- it pins
``librosa<=0.7.2``, which conflicts with the mel matrix build -- so the
tests that need it skip cleanly.  The tests that do not need it still
run: the module must be importable without ``audiomentations``.
'''

import pytest

from birdcall.augmentations import sed_augment_default

SAMPLE_RATE = 32000


def test_the_module_imports_without_audiomentations():
  '''The import of ``audiomentations`` is lazy, inside the function.'''
  assert callable(sed_augment_default.get_transforms)


def test_the_default_stack_keeps_the_length(augment_signal):
  '''The transform never changes the length of the recording.'''
  pytest.importorskip('audiomentations')
  transforms = sed_augment_default.get_transforms()
  signal = augment_signal
  augmented = transforms['train'](signal, SAMPLE_RATE)
  assert len(augmented) == len(signal)


def test_the_valid_transform_is_the_identity(augment_signal):
  '''The validation transform returns the waveform unchanged.'''
  pytest.importorskip('audiomentations')
  transforms = sed_augment_default.get_transforms()
  signal = augment_signal
  assert transforms['valid'](signal, SAMPLE_RATE) is signal
