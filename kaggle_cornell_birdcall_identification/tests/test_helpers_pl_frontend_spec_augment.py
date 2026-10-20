#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests for the PyTorch SpecAugment.

The two droppers zero random bands along the time axis and the frequency
axis.  The bands are drawn per batch item, so two items of the same
batch are dropped in different places.  In eval mode nothing is dropped
and the input is not modified.
'''

import pytest
import torch

from birdcall.helpers.pl_frontend_spec_augment import DropStripes
from birdcall.helpers.pl_frontend_spec_augment import SpecAugmentation

MEL_BINS = 64
FRAMES = 201


def test_spec_augmentation_is_a_noop_in_eval():
  '''In eval mode the stripes are not dropped, and nothing is mutated.'''
  augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2,
                               freq_drop_width=8, freq_stripes_num=2)
  augmenter.eval()
  spectrogram = torch.randn(3, 1, FRAMES, MEL_BINS)
  before = spectrogram.clone()
  out = augmenter(spectrogram)
  assert torch.equal(out, spectrogram)
  assert torch.equal(spectrogram, before)


def test_spec_augmentation_drops_an_independent_band_per_item():
  '''The band positions differ between two items of the same batch.'''
  torch.manual_seed(0)
  dropper = DropStripes(dim=2, drop_width=64, stripes_num=2)
  dropper.train()
  spectrogram = torch.randn(4, 1, FRAMES, MEL_BINS)
  out = dropper(spectrogram)
  dropped = (out == 0).all(dim=3)
  assert dropped[0].any()
  assert dropped[1].any()
  assert not torch.equal(dropped[0], dropped[1])


def test_spec_augmentation_drops_frequency_bands_per_item():
  '''The frequency dropper also draws an independent band per item.'''
  torch.manual_seed(0)
  dropper = DropStripes(dim=3, drop_width=8, stripes_num=2)
  dropper.train()
  spectrogram = torch.randn(4, 1, FRAMES, MEL_BINS)
  out = dropper(spectrogram)
  dropped = (out == 0).all(dim=2)
  assert dropped[0].any()
  assert not torch.equal(dropped[0], dropped[1])


def test_spec_augmentation_keeps_the_frame_count():
  '''The stripes change values, never the shape.'''
  torch.manual_seed(0)
  augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2,
                               freq_drop_width=8, freq_stripes_num=2)
  augmenter.train()
  spectrogram = torch.randn(2, 1, FRAMES, MEL_BINS)
  out = augmenter(spectrogram)
  assert out.shape == spectrogram.shape
  assert not torch.equal(out, spectrogram)


def test_the_input_is_not_modified_in_place():
  '''The original mutated its input; this port does not.'''
  torch.manual_seed(0)
  augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2,
                               freq_drop_width=8, freq_stripes_num=2)
  augmenter.train()
  spectrogram = torch.randn(2, 1, FRAMES, MEL_BINS)
  before = spectrogram.clone()
  augmenter(spectrogram)
  assert torch.equal(spectrogram, before)


def test_a_three_dimensional_input_is_refused():
  '''The dropper only accepts a 4-D spectrogram.'''
  dropper = DropStripes(dim=2, drop_width=8, stripes_num=1)
  with pytest.raises(ValueError, match='4-D'):
    dropper(torch.randn(2, FRAMES, MEL_BINS))


def test_an_unsupported_axis_is_refused():
  '''Only the time and the frequency axes are droppable.'''
  dropper = DropStripes(dim=1, drop_width=8, stripes_num=1)
  with pytest.raises(ValueError, match='dim'):
    dropper(torch.randn(2, 1, FRAMES, MEL_BINS))
