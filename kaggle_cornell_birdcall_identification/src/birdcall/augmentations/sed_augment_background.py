#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''The background waveform augmentation stack.

This is the ``aug_name="secondary_default"`` stack, and its defining
feature is the **palindromic bracket**: the noise, SNR and gain trio is
applied before *and* after the two background augmenters, so an additive
perturbation is sandwiched between two gain stages.

The bracket is reproduced faithfully, quirks included: ``Gain`` can reach
+30 dB with no clipping guard, and the SNR is perturbed twice.

``audiomentations`` is imported **lazily**, inside the function that needs
it, so this module stays importable without it.
'''

from typing import Callable


def get_transforms(bckgrd_aug_dir=None,
                   secondary_bckgrd_aug_dir=None) -> dict[str, Callable]:
  '''Builds the background train and validation transforms.

  Args:
    bckgrd_aug_dir: A directory of background noise wavs.  When it is not
      ``None`` the stack gains an ``AddBackgroundNoise`` transform.
    secondary_bckgrd_aug_dir: A second directory of background noise
      wavs.  When it is not ``None`` the stack gains an
      ``AddShortNoises`` transform.

  Returns:
    A dict with the keys ``train`` and ``valid``, both callables of
    ``(x, sr)``.
  '''
  from audiomentations import AddBackgroundNoise
  from audiomentations import AddGaussianNoise
  from audiomentations import AddGaussianSNR
  from audiomentations import AddShortNoises
  from audiomentations import Compose
  from audiomentations import Gain

  stack = [
      AddGaussianNoise(p=0.2),
      AddGaussianSNR(p=0.2),
      Gain(min_gain_in_db=-15, max_gain_in_db=15, p=0.3),
  ]
  if bckgrd_aug_dir is not None:
    stack.append(AddBackgroundNoise(bckgrd_aug_dir, p=0.2))
  if secondary_bckgrd_aug_dir is not None:
    stack.append(AddShortNoises(secondary_bckgrd_aug_dir,
                                min_time_between_sounds=0.0,
                                max_time_between_sounds=15.0,
                                burst_probability=0.5,
                                p=0.6))
  stack += [
      AddGaussianNoise(p=0.2),
      AddGaussianSNR(p=0.2),
      Gain(min_gain_in_db=-15, max_gain_in_db=15, p=0.3),
  ]
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
