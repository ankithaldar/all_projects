#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''The power spectrogram.

`Spectrogram` owns an `STFT` and turns its real and imaginary parts into
a power spectrogram, `real ** 2 + imag ** 2`.  The output is
`(B, 1, n_frames, n_fft // 2 + 1)`.

The power, not the magnitude, is load bearing: the dB formula downstream
is `10 * log10(power)`, and feeding the magnitude is a silent factor-of-two
error.
'''

import torch
from torch import nn

from birdcall.helpers.pl_frontend_stft import STFT


class Spectrogram(nn.Module):
  '''The power spectrogram of a waveform batch.'''

  def __init__(self, n_fft: int = 2048, hop_length: int | None = None,
               win_length: int | None = None, window: str | None = 'hann',
               center: bool = True, pad_mode: str = 'reflect',
               power: float = 2.0, freeze_parameters: bool = True) -> None:
    '''Builds the spectrogram extractor.

    Args:
      n_fft: The FFT size.
      hop_length: The hop between frames, in samples.  Defaults to
        `win_length // 4`.
      win_length: The length of the analysis window, in samples.
        Defaults to `n_fft`.
      window: The window name.
      center: Whether to reflect-pad the input by `n_fft // 2`.
      pad_mode: The padding mode, `'reflect'` or `'constant'`.
      power: The exponent of the magnitude.  `2.0` gives the power
        spectrogram; anything else is raised to `power / 2`.
      freeze_parameters: Whether to freeze the STFT weights.
    '''
    super().__init__()
    self.power = power
    self.stft = STFT(n_fft=n_fft, hop_length=hop_length,
                     win_length=win_length, window=window, center=center,
                     pad_mode=pad_mode, freeze_parameters=freeze_parameters)

  def forward(self, input: torch.Tensor) -> torch.Tensor:
    '''Transforms a batch of waveforms into a power spectrogram.

    Args:
      input: A `(B, n_samples)` waveform batch.

    Returns:
      A `(B, 1, n_frames, n_fft // 2 + 1)` power spectrogram.
    '''
    real, imag = self.stft(input)
    # (B, n_bins, n_frames) -> (B, 1, n_frames, n_bins)
    real = real[:, None, :, :].transpose(2, 3)
    imag = imag[:, None, :, :].transpose(2, 3)
    spectrogram = real ** 2 + imag ** 2
    if self.power != 2.0:
      spectrogram = spectrogram ** (self.power / 2.0)
    return spectrogram
