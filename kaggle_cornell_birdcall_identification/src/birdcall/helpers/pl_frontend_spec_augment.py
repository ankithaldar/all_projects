#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''SpecAugment, as two stripe droppers.

`DropStripes` zeroes `stripes_num` random bands along one axis of a
`(B, C, n_frames, n_bins)` tensor, and `SpecAugmentation` runs the time
dropper (axis 2) and then the frequency dropper (axis 3).

The original mutated its input in place, in a Python loop over the
batch.  Here the bands are drawn per batch item and applied as a mask,
which keeps the observable behaviour -- an independent band per item --
without the in-place mutation.
'''

import torch
from torch import nn


class DropStripes(nn.Module):
  '''Zeroes random bands along one axis of a spectrogram.'''

  def __init__(self, dim: int, drop_width: int, stripes_num: int) -> None:
    '''Builds the stripe dropper.

    Args:
      dim: The axis to drop along, 2 for time and 3 for frequency.
      drop_width: The maximum width of a band, in frames or bins.  The
        drawn width is in `[0, drop_width)`.
      stripes_num: The number of bands dropped per item.
    '''
    super().__init__()
    self.dim = dim
    self.drop_width = drop_width
    self.stripes_num = stripes_num

  def forward(self, input: torch.Tensor) -> torch.Tensor:
    '''Drops the stripes.

    Args:
      input: A `(B, C, n_frames, n_bins)` spectrogram.

    Returns:
      The same spectrogram, with the bands zeroed.  The input is not
      modified in place, and it is returned unchanged when the module is
      not in training mode.

    Raises:
      ValueError: If the input is not 4-D, or `dim` is not 2 or 3.
    '''
    if input.ndim != 4:
      raise ValueError(f'Expected a 4-D spectrogram, got {input.ndim}-D')
    if self.dim not in (2, 3):
      raise ValueError(f'Unsupported dim: {self.dim!r}')
    if not self.training:
      return input

    batch_size = input.shape[0]
    total_width = input.shape[self.dim]
    mask = self._band_mask(batch_size, total_width, input.device)
    mask = mask.to(input.dtype)
    if self.dim == 2:
      # (B, 1, n_frames, 1)
      mask = mask[:, None, :, None]
    else:
      # (B, 1, 1, n_bins)
      mask = mask[:, None, None, :]
    return input * mask

  def _band_mask(self, batch_size: int, total_width: int,
                 device: torch.device) -> torch.Tensor:
    '''Draws the dropped bands.

    Args:
      batch_size: The number of items in the batch.
      total_width: The length of the dropped axis.
      device: The device to draw on.

    Returns:
      A `(batch_size, total_width)` boolean mask, `True` where the
      spectrogram is kept.
    '''
    stripes = torch.randint(0, self.drop_width,
                             (batch_size, self.stripes_num), device=device)
    # The original drew `bgn` from `[0, total_width - distance)`, with
    # a per-item upper bound.  Scaling a uniform draw by that bound is
    # the same distribution, and it vectorises.
    high = torch.clamp(total_width - stripes, min=1)
    starts = (torch.rand(batch_size, self.stripes_num,
                          device=device) * high).long()
    positions = torch.arange(total_width, device=device)
    bands = ((positions[None, None, :] >= starts[:, :, None]) &
             (positions[None, None, :] < (starts + stripes)[:, :, None]))
    return ~bands.any(dim=1)


class SpecAugmentation(nn.Module):
  '''Drops time stripes and then frequency stripes.'''

  def __init__(self, time_drop_width: int, time_stripes_num: int,
               freq_drop_width: int, freq_stripes_num: int) -> None:
    '''Builds the two droppers.

    Args:
      time_drop_width: The maximum width of a time band, in frames.
      time_stripes_num: The number of time bands dropped per item.
      freq_drop_width: The maximum width of a frequency band, in bins.
      freq_stripes_num: The number of frequency bands dropped per item.
    '''
    super().__init__()
    self.time_dropper = DropStripes(dim=2, drop_width=time_drop_width,
                                     stripes_num=time_stripes_num)
    self.freq_dropper = DropStripes(dim=3, drop_width=freq_drop_width,
                                     stripes_num=freq_stripes_num)

  def forward(self, input: torch.Tensor) -> torch.Tensor:
    '''Drops the time stripes and then the frequency stripes.

    Args:
      input: A `(B, C, n_frames, n_bins)` spectrogram.

    Returns:
      The same spectrogram, with the bands zeroed.
    '''
    x = self.time_dropper(input)
    return self.freq_dropper(x)
