#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests of the augmentation registry.

The engine dispatches on ``hparams.aug_name`` and the registry fails
loudly on an unknown one, instead of the original's ``if``/``elif``
chain, which had no ``else`` and failed later with an unrelated
``AttributeError``.
'''

import pytest

from birdcall.augmentations import registry
from birdcall.augmentations import sed_augment_background
from birdcall.augmentations import sed_augment_default


def test_the_registry_holds_both_stacks():
  '''The two shipped stacks are registered under their config names.'''
  assert set(registry.AUGMENTATIONS) == {'default', 'secondary_default'}


def test_build_transforms_raises_on_an_unknown_name():
  '''An unknown name is a clear error that names the valid ones.'''
  with pytest.raises(ValueError) as error:
    registry.build_transforms('nope')
  message = str(error.value)
  assert 'nope' in message
  assert 'default' in message
  assert 'secondary_default' in message


def test_the_registry_maps_every_name_to_its_stack():
  '''The names point at the two stack builders.'''
  assert registry.AUGMENTATIONS['default'] is sed_augment_default.get_transforms
  assert (registry.AUGMENTATIONS['secondary_default']
          is sed_augment_background.get_transforms)
