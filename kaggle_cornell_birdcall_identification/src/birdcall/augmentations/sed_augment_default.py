#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''The default waveform augmentation stack.

This is the ``aug_name="default"`` stack: gaussian noise, gaussian SNR
and, when a noise directory is given, background noise.  The validation
transform is the identity.

``audiomentations`` is imported **lazily**, inside the function that needs
it, so this module stays importable without it -- the package pins
``audiomentations==0.11.0``, which conflicts with the librosa version the
mel matrix is built with, and the two cannot always be installed
together.

The transform signature is ``transform(x, sr)``: the whole recording and
its sample rate.  The length of the recording is never changed.
'''

from typing import Callable


def get_transforms(bckgrd_aug_dir=None) -> dict[str, Callable]:
  '''Builds the default train and validation transforms.

  Args:
    bckgrd_aug_dir: A directory of background noise wavs.  When it is not
      ``None`` the stack gains an ``AddBackgroundNoise`` transform.

  Returns:
    A dict with the keys ``train`` and ``valid``, both callables of
    ``(x, sr)``.
  '''
  from audiomentations import AddBackgroundNoise
  from audiomentations import AddGaussianNoise
  from audiomentations import AddGaussianSNR
  from audiomentations import Compose

  stack = [
      AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.3),
      AddGaussianSNR(p=0.3),
  ]
  if bckgrd_aug_dir is not None:
    stack.append(AddBackgroundNoise(bckgrd_aug_dir, p=0.5))
  augmenter = Compose(stack)

  def train(x, sr):
    '''Augments a whole recording.

    Args:
      x: The waveform.
      sr: The sample rate of the waveform.

    Returns:
      The augmented waveform, of the same length.
    '''
    return augmenter(samples=x, sample_rate=sr)

  def valid(x, sr):
    '''Returns the waveform unchanged.

    Args:
      x: The waveform.
      sr: The sample rate of the waveform.

    Returns:
      The waveform, unchanged.
    '''
    return x

  return {'train': train, 'valid': valid}
