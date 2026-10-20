#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''The registry of the waveform augmentation stacks.

The engine dispatches on ``hparams.aug_name``; this module turns that
name into a transform dict and fails loudly on an unknown one, instead of
the original's ``if``/``elif`` chain, which had no ``else`` and failed
later with an unrelated ``AttributeError``.
'''

from typing import Callable

from birdcall.augmentations import sed_augment_background
from birdcall.augmentations import sed_augment_default

AUGMENTATIONS: dict[str, Callable] = {
    'default': sed_augment_default.get_transforms,
    'secondary_default': sed_augment_background.get_transforms,
}
'''The augmentation stacks, keyed by the name the config selects.'''


def build_transforms(name, bckgrd_aug_dir=None,
                     secondary_bckgrd_aug_dir=None) -> dict[str, Callable]:
  '''Builds the transform dict of an augmentation name.

  Args:
    name: The name of the stack, one of the keys of ``AUGMENTATIONS``.
    bckgrd_aug_dir: A directory of background noise wavs.
    secondary_bckgrd_aug_dir: A second directory of background noise
      wavs, only read by the ``secondary_default`` stack.

  Returns:
    A dict with the keys ``train`` and ``valid``.

  Raises:
    ValueError: If the name is unknown.
  '''
  if name not in AUGMENTATIONS:
    raise ValueError(
        f'Unknown augmentation {name!r}. '
        f'Valid names: {sorted(AUGMENTATIONS)}.')
  if name == 'secondary_default':
    return AUGMENTATIONS[name](
        bckgrd_aug_dir=bckgrd_aug_dir,
        secondary_bckgrd_aug_dir=secondary_bckgrd_aug_dir)
  return AUGMENTATIONS[name](bckgrd_aug_dir=bckgrd_aug_dir)
